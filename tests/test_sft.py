# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid.sft on CPU: a shrunk vLLM-shaped hybrid LM (attention with a
sliding window, a Qwen3.5-layout GDN layer, a Gemma-style KV-shared layer,
packed segments) whose serving-path ops CUT autograd like real kernels do
(fused norms / activation / RoPE run under no_grad, attention and GDN read a
paged cache = raise, the inner model's __call__ is "compiled" = raise).

(a) the chunked gated delta rule == the token recurrence (values + grads)
(b) adapters: untrained = bitwise no-op on the serving path, isinstance
    preserved, base_apply bypasses, trained delta = (x A) B * scale
(c) TrainMode loss + every adapter grad == an independent pure-torch oracle
    (explicit attention, recurrent GDN, dense LoRA), packed == separate,
    host offload on/off identical, every swap restored on exit
(d) an op without autograd is caught by the gradient-reach guard
(e) worker run: serving unchanged until commit, commit = trained values,
    save / reset / load round trip, fingerprint refusal, errors never raise
(f) TP=2 (gloo): loss and TP-reduced grads == TP=1, replicated factors stay
    bit-identical after the host AdamW step
(g) HTTP state machine end to end with a fake engine: train -> local
    checkpoint + latest.json -> served; load base / run; startup restore;
    validation; one run at a time
"""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from suffix_hybrid.mtp_tune.lora import base_apply
from suffix_hybrid.sft import Config
from suffix_hybrid.sft import api as A
from suffix_hybrid.sft import model as M
from suffix_hybrid.sft import worker as W

H, NH, NKV, D, I = 32, 4, 2, 8, 48
HK, HV, DK, DV, CK = 2, 4, 8, 8, 4
V, VPAD = 60, 64
REF = {"on": False, "params": {}}     # oracle mode: plain autograd everywhere


# ---------------------------------------------------------------------------
# toy vLLM surface
# ---------------------------------------------------------------------------
class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class ProbeMethod(UnquantizedLinearMethod):
    """A 'quantized' method (unknown layout) -> FrozenLinear must probe."""


class UnquantizedEmbeddingMethod(UnquantizedLinearMethod):
    pass


class Lin(nn.Module):
    def __init__(self, w, kind="rep", group=None, idx=None, probe=False):
        super().__init__()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.tp_size = group.world_size if (group is not None and kind != "rep") else 1
        self.group, self.shard_idx = group, idx
        if kind == "col":
            self.gather_output = False
        elif kind == "row":
            self.input_is_parallel, self.reduce_results = True, True
        self.quant_method = ProbeMethod() if probe else UnquantizedLinearMethod()

    def forward(self, x):
        if REF["on"]:
            y = F.linear(x, self.weight)
            p = REF["params"].get(id(self))
            if p is not None:
                y = y + (x @ p[0]) @ p[1] * p[2]
        else:
            y = self.quant_method.apply(self, x, None)
        if self.tp_size > 1 and hasattr(self, "input_is_parallel"):
            y = self.group.all_reduce(y)
        return y, None


class CustomOp(nn.Module):
    """vLLM CustomOp dispatch: the 'fused' kernel has no autograd."""

    def __init__(self):
        super().__init__()
        self._forward_method = self.forward_fused

    def forward(self, *a, **kw):
        return (self.forward_native if REF["on"] else self._forward_method)(*a, **kw)

    def forward_fused(self, *a, **kw):
        with torch.no_grad():
            return self.forward_native(*a, **kw)


class Norm(CustomOp):
    def __init__(self, n, g):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(n, generator=g), requires_grad=False)

    def forward_native(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return y if residual is None else (y, residual)


class GatedNorm(CustomOp):
    def __init__(self, n, g):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(n, generator=g), requires_grad=False)
        self.activation = "silu"

    def forward_native(self, x, z):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return y * F.silu(z)


class SiluAndMul(CustomOp):
    def forward_native(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class Rope(CustomOp):
    def forward_native(self, pos, q, k=None):
        inv = 1.0 / (100.0 ** (torch.arange(0, D, 2).float() / D))
        f = pos.float()[:, None] * inv[None]
        cos, sin = f.cos()[:, None], f.sin()[:, None]

        def rot(x):
            x = x.view(x.shape[0], -1, D)
            a, b = x[..., :D // 2], x[..., D // 2:]
            return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).flatten(1)
        return rot(q), (None if k is None else rot(k))


class Attention(nn.Module):
    def __init__(self, nh, nkv, name, sliding=None, target=None):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_size, self.head_size_v = nh, nkv, D, D
        self.impl = SimpleNamespace(scale=D ** -0.5, logits_soft_cap=None)
        self.layer_name, self.sliding_window, self.kv_sharing_target_layer_name = name, sliding, target

    def forward(self, q, k, v):
        if not REF["on"]:
            raise RuntimeError("paged-KV attention called in training")
        t = q.shape[0]
        if k is None:
            k, v = REF["kv"][self.kv_sharing_target_layer_name]
        else:
            REF["kv"][self.layer_name] = (k, v)
        q, kk, vv = q.view(t, -1, D), k.view(t, -1, D), v.view(t, -1, D)
        rep = q.shape[1] // kk.shape[1]
        kk, vv = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
        out = torch.zeros_like(q)
        for s, e in REF["segs"]:
            sc = torch.einsum("qhd,khd->hqk", q[s:e], kk[s:e]) * self.impl.scale
            i = torch.arange(e - s)
            ok = i[:, None] >= i[None, :]
            if self.sliding_window:
                ok &= (i[:, None] - i[None, :]) < self.sliding_window
            p = sc.masked_fill(~ok, float("-inf")).softmax(-1)
            out[s:e] = torch.einsum("hqk,khd->qhd", p, vv[s:e])
        return out.flatten(1)


class AttnBlock(nn.Module):
    def __init__(self, ws, group, name, sliding=None, target=None):
        super().__init__()
        self.shared = target is not None
        loc = (group.world_size if group else 1)
        self.nh, self.nkv = NH // loc, NKV // loc
        if self.shared:
            self.q_proj = Lin(ws["q"], group=group, kind="col")
        else:
            self.qkv_proj = Lin(ws["qkv"], group=group, kind="col")
        self.o_proj = Lin(ws["o"], group=group, kind="row")
        self.rotary_emb = Rope()
        self.attn = Attention(self.nh, self.nkv, name, sliding, target)

    def forward(self, pos, x):
        if self.shared:
            q, _ = self.q_proj(x)
            q, _ = self.rotary_emb(pos, q, None)
            o = self.attn(q, None, None)
        else:
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.nh * D, self.nkv * D, self.nkv * D], -1)
            q, k = self.rotary_emb(pos, q, k)
            o = self.attn(q, k, v)
        return self.o_proj(o)[0]


def ref_gdn_rule(q, k, v, g, beta):
    """Token recurrence (HF torch_recurrent_gated_delta_rule), [T, H, D]."""
    q = F.normalize(q.float(), dim=-1, eps=1e-6) * q.shape[-1] ** -0.5
    k = F.normalize(k.float(), dim=-1, eps=1e-6)
    s = q.new_zeros(q.shape[1], q.shape[2], v.shape[2])
    out = []
    for t in range(q.shape[0]):
        s = s * g[t].exp()[:, None, None]
        mem = (s * k[t][:, :, None]).sum(1)
        delta = (v[t].float() - mem) * beta[t][:, None]
        s = s + k[t][:, :, None] * delta[:, None, :]
        out.append((s * q[t][:, :, None]).sum(1))
    return torch.stack(out)


class GatedDeltaNet(nn.Module):
    def __init__(self, ws, group, g):
        super().__init__()
        self.tp_size = group.world_size if group else 1
        self.key_dim, self.value_dim, self.head_k_dim, self.head_v_dim = HK * DK, HV * DV, DK, DV
        self.gqa_interleaved_layout, self.activation = False, "silu"
        self.in_proj_qkvz = Lin(ws["qkvz"], group=group, kind="col", probe=True)
        self.in_proj_ba = Lin(ws["ba"], group=group, kind="col")
        self.conv1d = Lin(ws["conv"], group=group, kind="col")
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)   # [C, 1, K] like vLLM
        self.conv1d.bias = None
        self.A_log = nn.Parameter(ws["A_log"], requires_grad=False)
        self.dt_bias = nn.Parameter(ws["dt_bias"], requires_grad=False)
        self.norm = GatedNorm(DV, g)
        self.out_proj = Lin(ws["out"], group=group, kind="row")

    def split_ba(self, ba):
        return ba.chunk(2, -1)

    def forward(self, hidden_states):
        if not REF["on"]:
            raise RuntimeError("GDN state-cache path called in training")
        kd, vd = self.key_dim // self.tp_size, self.value_dim // self.tp_size
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        mixed, z = qkvz.split([2 * kd + vd, vd], -1)
        z = z.reshape(z.shape[0], -1, DV)
        b, a = self.split_ba(self.in_proj_ba(hidden_states)[0])
        out = []
        w = self.conv1d.weight[:, 0]                     # [C, K]
        for s, e in REF["segs"]:
            x = mixed[s:e]
            xp = torch.cat([x.new_zeros(CK - 1, x.shape[1]), x])
            conv = torch.stack([(xp[t:t + CK] * w.t()).sum(0) for t in range(e - s)])
            conv = F.silu(conv)
            q, k, v = conv.split([kd, kd, vd], -1)
            q, k, v = q.view(e - s, -1, DK), k.view(e - s, -1, DK), v.view(e - s, -1, DV)
            rep = v.shape[1] // q.shape[1]
            q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
            gg = -self.A_log.float().exp() * F.softplus(a[s:e].float() + self.dt_bias.float())
            out.append(ref_gdn_rule(q, k, v, gg, b[s:e].float().sigmoid()))
        core = torch.cat(out).to(hidden_states.dtype)
        return self.out_proj(self.norm(core, z).flatten(-2))[0]


class Mlp(nn.Module):
    def __init__(self, ws, group):
        super().__init__()
        self.gate_up_proj = Lin(ws["gu"], group=group, kind="col")
        self.act = SiluAndMul()
        self.down_proj = Lin(ws["down"], group=group, kind="row")

    def forward(self, x):
        return self.down_proj(self.act(self.gate_up_proj(x)[0]))[0]


class Layer(nn.Module):
    def __init__(self, kind, ws, group, g, idx):
        super().__init__()
        self.kind = kind
        self.input_layernorm, self.post_attention_layernorm = Norm(H, g), Norm(H, g)
        if kind == "gdn":
            self.linear_attn = GatedDeltaNet(ws, group, g)
        else:
            self.self_attn = AttnBlock(ws, group, f"model.layers.{idx}.self_attn.attn",
                                       sliding=5 if idx == 0 else None,
                                       target="model.layers.0.self_attn.attn" if kind == "shared" else None)
        self.mlp = Mlp(ws, group)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual, h = hidden_states, self.input_layernorm(hidden_states)
        else:
            h, residual = self.input_layernorm(hidden_states, residual)
        h = self.linear_attn(hidden_states=h) if self.kind == "gdn" else self.self_attn(positions, h)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class Embed(nn.Module):
    def __init__(self, w, group, start):
        super().__init__()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)
        self.group, self.start = group, start

    def forward(self, ids):
        loc = ids - self.start
        ok = (loc >= 0) & (loc < self.weight.shape[0])
        e = F.embedding(loc.clamp(0, self.weight.shape[0] - 1), self.weight) * ok[:, None]
        return self.group.all_reduce(e) if self.group is not None else e


class Inner(nn.Module):
    def __init__(self, ws, group, g, start):
        super().__init__()
        self.do_not_compile = False
        self.embed_tokens = Embed(ws["emb"], group, start)
        self.layers = nn.ModuleList([Layer(k, ws["layers"][i], group, g, i)
                                     for i, k in enumerate(("attn", "gdn", "shared"))])
        self.norm = Norm(H, g)

    def __call__(self, *a, **kw):        # vLLM support_torch_compile dispatch
        if not (self.do_not_compile or REF["on"]):
            raise RuntimeError("compiled forward called in training")
        return self.forward(*a, **kw)

    def forward(self, input_ids, positions):
        h, res = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, res = layer(positions=positions, hidden_states=h, residual=res)
        return self.norm(h, res)[0]


class LM(nn.Module):
    def __init__(self, ws, group, rank, world):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        vloc = VPAD // world
        start = rank * vloc
        self.model = Inner(ws, group, g, start)
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(ws["head"][start:start + vloc].clone(), requires_grad=False)
        self.lm_head.quant_method = UnquantizedEmbeddingMethod()
        self.lm_head.shard_indices = SimpleNamespace(org_vocab_start_index=min(start, V),
                                                     org_vocab_end_index=min(start + vloc, V))
        self.logits_processor = SimpleNamespace(scale=1.0, soft_cap=None)


def full_weights(seed=3):
    g = torch.Generator().manual_seed(seed)

    def r(*s, sc=0.2):
        return torch.randn(*s, generator=g) * sc
    layers = []
    for _ in range(3):
        layers.append(dict(
            qkv=r((NH + 2 * NKV) * D, H), q=r(NH * D, H), o=r(H, NH * D),
            qkvz=r(2 * HK * DK + 2 * HV * DV, H), ba=r(2 * HV, H), conv=r(2 * HK * DK + HV * DV, CK, sc=0.5),
            A_log=r(HV, sc=0.5), dt_bias=r(HV, sc=0.5), out=r(H, HV * DV),
            gu=r(2 * I, H), down=r(H, I)))
    head = r(VPAD, H)
    head[V:] = 0
    return dict(emb=r(VPAD, H, sc=1.0), head=head, layers=layers)


def _cat(*ranges):
    return torch.cat([torch.arange(a, b) for a, b in ranges])


def shard(ws, rank, world):
    """vLLM sharding of the full weights: (local weight, global out / in idx)."""
    def heads(off, n, d):
        per = n // world
        return (off + rank * per * d, off + (rank + 1) * per * d)
    out = dict(emb=ws["emb"][rank * VPAD // world:(rank + 1) * VPAD // world], head=ws["head"], layers=[])
    for L in ws["layers"]:
        qkv = _cat(heads(0, NH, D), heads(NH * D, NKV, D), heads((NH + NKV) * D, NKV, D))
        qi = _cat(heads(0, NH, D))
        kd, vd = HK * DK, HV * DV
        conv = _cat(heads(0, HK, DK), heads(kd, HK, DK), heads(2 * kd, HV, DV))
        qkvz = torch.cat([conv, _cat(heads(2 * kd + vd, HV, DV))])
        hv = _cat(heads(0, HV, 1))
        ba = torch.cat([hv, hv + HV])
        vo = _cat(heads(0, HV, DV))
        gi = _cat(heads(0, I, 1))
        gu = torch.cat([gi, gi + I])
        out["layers"].append(dict(
            qkv=L["qkv"][qkv], qkv_idx=qkv, q=L["q"][qi], q_idx=qi, o=L["o"][:, qi], o_idx=qi,
            qkvz=L["qkvz"][qkvz], qkvz_idx=qkvz, ba=L["ba"][ba], ba_idx=ba, conv=L["conv"][conv],
            A_log=L["A_log"][hv], dt_bias=L["dt_bias"][hv], out=L["out"][:, vo], out_idx=vo,
            gu=L["gu"][gu], gu_idx=gu, down=L["down"][:, gi], down_idx=gi))
    return out


def build(rank=0, world=1, group=None, seed=3):
    ws = shard(full_weights(seed), rank, world)
    lm = LM(ws, group, rank, world)
    for i, layer in enumerate(lm.model.layers):     # record shard maps on the Lins
        L = ws["layers"][i]
        mods = {"mlp.gate_up_proj": "gu", "mlp.down_proj": "down"}
        if layer.kind == "gdn":
            mods.update({"linear_attn.in_proj_qkvz": "qkvz", "linear_attn.in_proj_ba": "ba",
                         "linear_attn.out_proj": "out"})
        elif layer.kind == "shared":
            mods.update({"self_attn.q_proj": "q", "self_attn.o_proj": "o"})
        else:
            mods.update({"self_attn.qkv_proj": "qkv", "self_attn.o_proj": "o"})
        for path, key in mods.items():
            layer.get_submodule(path).shard_idx = L[key + "_idx"]
    return lm


def sft_for(lm, group=None, rank=4, targets=M.DEFAULT_TARGETS, **cfg):
    c = Config(rank=rank, alpha=8.0, targets=targets, min_free_gib=0.0, **cfg)
    return W.WorkerSFT(lm, None, c, group)


def set_adapters(sft, seed=0, world=1):
    """Deterministic FULL adapters, sliced like the layer's shard."""
    for name, (layer, lo) in sft.adapters.items():
        g = torch.Generator().manual_seed(len(name) * 7 + seed)
        kf = lo.A.shape[0] * (world if lo.kind == "row" else 1)
        nf = lo.B.shape[1] * (world if lo.kind == "col" else 1)
        a, b = torch.randn(kf, lo.r, generator=g) * 0.2, torch.randn(lo.r, nf, generator=g) * 0.2
        if lo.kind == "col":
            b = b[:, layer.shard_idx]
        if lo.kind == "row":
            a = a[layer.shard_idx]
        lo.load_(a, b)


BATCH = dict(ids=[3, 17, 22, 9, 41, 5, 33, 12, 8, 50, 2, 7, 19, 44, 21, 30, 11],
             seg_lens=[10, 7],
             mask=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])


def oracle(lm, sft, batch=BATCH):
    """Loss + adapter grads with plain autograd (no TrainMode, no FrozenLinear,
    explicit attention, recurrent GDN, full logits)."""
    params = {}
    for name, (layer, lo) in sft.adapters.items():
        params[id(layer)] = (lo.A.detach().clone().requires_grad_(True),
                             lo.B.detach().clone().requires_grad_(True), lo.scale)
    segs, rows, tgt = M.batch_rows(batch["seg_lens"], batch["mask"], "cpu")
    ids = torch.tensor(batch["ids"])
    pos = torch.cat([torch.arange(e - s) for s, e in segs])
    REF.update(on=True, params=params, segs=segs, kv={})
    try:
        h = lm.model(input_ids=ids, positions=pos)
        logits = h[rows] @ lm.lm_head.weight[:V].T
        loss = F.cross_entropy(logits, ids[tgt], reduction="sum")
        loss.backward()
    finally:
        REF.update(on=False, params={}, kv={})
    return float(loss.detach()), {n: (params[id(l)][0].grad, params[id(l)][1].grad)
                         for n, (l, _) in sft.adapters.items()}


def run_micro(sft, batch=BATCH, offload=False):
    sft.cfg.offload = offload
    sft.begin(dict(lr=1e-2))
    r = sft.micro(dict(batch, norm=1.0, check=True))
    grads = {n: (sft.run.flat.grad[sft.run.spans[(n, "A")][0]:][:lo.A.numel()].view_as(lo.A).clone(),
                 sft.run.grad_of(n).view_as(lo.B).clone())
             for n, (_, lo) in sft.adapters.items()}
    return r["loss_sum"], grads


# ---------------------------------------------------------------------------
# (a) gated delta rule
# ---------------------------------------------------------------------------
def test_chunk_rule_matches_recurrence():
    g = torch.Generator().manual_seed(0)
    t, h, d = 70, 2, 8          # 70 = one full chunk + a padded one
    q, k, v = (torch.randn(t, h, d, generator=g, requires_grad=True) for _ in range(3))
    a = torch.randn(t, h, generator=g, requires_grad=True)
    b = torch.randn(t, h, generator=g, requires_grad=True)
    gg, beta = -F.softplus(a), b.sigmoid()
    got = M.chunk_gated_delta_rule(q[None], k[None], v[None], gg[None], beta[None])[0]
    want = ref_gdn_rule(q, k, v, gg, beta)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    w = torch.randn(t, h, d, generator=g)
    g1 = torch.autograd.grad((got * w).sum(), (q, k, v, a, b), retain_graph=True)
    g2 = torch.autograd.grad((want * w).sum(), (q, k, v, a, b))
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-4)


class EmulatedNvfp4:
    """Stands in for the NVFP4 GEMM: y = x @ W_eff^T (W4A16 emulation)."""

    def __init__(self, w_eff):
        self.w_eff = w_eff

    def apply(self, layer, x, bias=None):
        return F.linear(x, self.w_eff.to(x.dtype), bias)


def nvfp4_layer(n, k, seed, scrambled=False):
    import numpy as np
    from suffix_hybrid.kernels import nvfp4_gemm as G
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.1
    g_w, g_x = np.float32(448 * 6 / np.abs(w).max()), np.float32(3.0)
    packed, sf_bits, sf = G.quantize(w, g_w)
    w_eff = torch.from_numpy(G.dequant(packed, sf) / g_w)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.from_numpy(packed), requires_grad=False)
    layer.weight_scale = torch.from_numpy(G.swizzle_sf(sf_bits)).view(torch.float8_e4m3fn)
    layer.alpha = torch.tensor(1.0 / (g_x * g_w))
    layer.input_global_scale_inv = torch.tensor(g_x)
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    # a backend with another weight layout (Marlin-style repack): the
    # dequant formula no longer describes what the kernel multiplies by
    layer.quant_method = EmulatedNvfp4(w_eff.flip(0) if scrambled else w_eff)
    return layer, layer.quant_method.w_eff


@pytest.mark.parametrize("scrambled", [False, True])
def test_frozen_linear_nvfp4_dequant_dx_or_probe_fallback(scrambled):
    from suffix_hybrid.mtp_tune.train import _NVFP4_OK, FrozenLinear
    layer, w_eff = nvfp4_layer(160, 96, seed=1, scrambled=scrambled)
    x = torch.randn(7, 96, requires_grad=True)
    dy = torch.randn(7, 160)
    FrozenLinear.apply(x, layer).backward(dy)
    torch.testing.assert_close(x.grad, dy @ w_eff, rtol=1e-4, atol=1e-5)
    assert _NVFP4_OK[id(layer)] is (not scrambled)      # fast path only when verified


# ---------------------------------------------------------------------------
# (b) serving seam
# ---------------------------------------------------------------------------
def test_adapters_zero_noop_isinstance_and_delta():
    lm = build()
    x = torch.randn(5, H)
    lin = lm.model.layers[0].mlp.gate_up_proj
    before = lin(x)[0]
    sft = sft_for(lm)
    assert "model.layers.0.mlp.gate_up_proj" in sft.adapters
    assert not any("conv1d" in n or "in_proj_ba" in n for n in sft.adapters)
    assert isinstance(lin.quant_method, UnquantizedLinearMethod)
    assert torch.equal(lin(x)[0], before)                        # B == 0: exact no-op
    set_adapters(sft)
    lo = sft.adapters["model.layers.0.mlp.gate_up_proj"][1]
    torch.testing.assert_close(lin(x)[0], before + (x @ lo.A) @ lo.B * lo.scale)
    assert torch.equal(base_apply(lin, x), before)               # frozen path skips the delta
    names = {n for n, _ in lin.named_buffers()}
    assert {"sft_A", "sft_B"} <= names and "sft_A" not in lin.state_dict()


# ---------------------------------------------------------------------------
# (c) training forward == oracle
# ---------------------------------------------------------------------------
def test_trainmode_matches_oracle_and_restores():
    lm = build()
    sft = sft_for(lm)
    set_adapters(sft)
    want_loss, want = oracle(lm, sft)
    loss, got = run_micro(sft)
    assert abs(loss - want_loss) < 1e-4 * want_loss
    for n in want:
        for i, f in enumerate("AB"):
            torch.testing.assert_close(got[n][i], want[n][i], rtol=2e-3, atol=2e-5,
                                       msg=f"{n}.{f}")
    # packed == the two segments trained separately
    sft.abort()
    a = dict(ids=BATCH["ids"][:10], seg_lens=[10], mask=BATCH["mask"][:10])
    b = dict(ids=BATCH["ids"][10:], seg_lens=[7], mask=BATCH["mask"][10:])
    la, _ = run_micro(sft, a)
    sft.abort()
    lb, _ = run_micro(sft, b)
    assert abs(la + lb - loss) < 1e-4 * loss
    # every swap restored: serving paths raise / dispatch fused again
    with pytest.raises(RuntimeError, match="compiled"):
        lm.model(input_ids=torch.tensor([1, 2]), positions=torch.arange(2))
    assert lm.model.layers[0].input_layernorm._forward_method == lm.model.layers[0].input_layernorm.forward_fused
    assert "forward" not in lm.model.layers[0].__dict__ and M._TRAIN is None


def test_offload_and_eval_agree():
    lm = build()
    sft = sft_for(lm)
    set_adapters(sft)
    l0, g0 = run_micro(sft, offload=False)
    sft.abort()
    l1, g1 = run_micro(sft, offload=True)
    assert l0 == pytest.approx(l1, rel=1e-6)
    for n in g0:
        torch.testing.assert_close(g0[n][1], g1[n][1])
    sft.abort()
    ev = sft.eval(dict(BATCH))
    assert ev["loss_sum"] == pytest.approx(l0, rel=1e-5) and ev["tokens"] == 12


# ---------------------------------------------------------------------------
# (d) gradient-reach guard
# ---------------------------------------------------------------------------
class OpaqueKernel(nn.Module):
    """A fused op TrainMode cannot swap (no forward_native): output detached."""

    def forward(self, x):
        return SiluAndMul.forward_native(None, x).detach()


def test_grad_reach_catches_op_without_autograd():
    lm = build()
    sft = sft_for(lm)
    lm.model.layers[1].mlp.act = OpaqueKernel()
    W.SFT = sft
    try:
        sft.begin(dict(lr=1e-3))
        r = W.rpc("micro", dict(BATCH, norm=1.0, check=True))
    finally:
        W.SFT = None
    assert "gradient does not reach" in r["error"] and sft.run is None


# ---------------------------------------------------------------------------
# (e) worker run lifecycle
# ---------------------------------------------------------------------------
def _served(lm, ids=(3, 17, 22, 9, 41)):
    """Serving-path output of one adapted layer stack (no TrainMode)."""
    x = lm.model.embed_tokens(torch.tensor(ids))
    return lm.model.layers[0].mlp(x)


def test_run_commit_save_load_reset(tmp_path):
    lm = build()
    sft = sft_for(lm)
    base = _served(lm)
    W.SFT = sft
    try:
        assert W.rpc("begin", dict(lr=5e-2))["ok"]
        for _ in range(3):
            assert "error" not in W.rpc("micro", dict(BATCH, norm=12.0))
            assert W.rpc("step")["grad_norm"] > 0
        assert torch.equal(_served(lm), base)                   # not served before commit
        trained = {k: v.clone() for k, v in sft.run.tensors().items()}
        W.rpc("save", {"dir": str(tmp_path)})
        assert W.rpc("commit")["ok"] and sft.run is None
        for (n, f), t in trained.items():
            assert torch.equal(getattr(sft.adapters[n][1], f), t)
        served = _served(lm)
        assert not torch.allclose(served, base)
        assert W.rpc("reset")["ok"] and torch.equal(_served(lm), base)
        assert W.rpc("load", {"dir": str(tmp_path)})["ok"]
        assert torch.equal(_served(lm), served)
        assert "error" in W.rpc("bogus")
        assert "error" in W.rpc("step")                         # no run: error, never raises
    finally:
        W.SFT = None
    other = sft_for(build(), rank=2)                            # different adapter config
    W.SFT = other
    try:
        assert "fingerprint" in W.rpc("load", {"dir": str(tmp_path)})["error"]
    finally:
        W.SFT = None


def test_training_lowers_loss():
    lm = build()
    sft = sft_for(lm)
    sft.begin(dict(lr=3e-2))
    losses = []
    for _ in range(8):
        losses.append(sft.micro(dict(BATCH, norm=12.0))["loss_sum"] / 12.0)
        sft.step()
    sft.commit()
    assert losses[-1] < losses[0] - 0.05
    assert sft.eval(dict(BATCH))["loss_sum"] / 12.0 < losses[0] - 0.05


# ---------------------------------------------------------------------------
# (f) TP=2 over gloo
# ---------------------------------------------------------------------------
class GlooGroup:
    def __init__(self):
        import torch.distributed as dist
        self.d = dist
        self.world_size, self.rank_in_group = dist.get_world_size(), dist.get_rank()

    def all_reduce(self, x):     # like vLLM's custom-op all-reduce: no autograd
        y = x.detach().clone()
        self.d.all_reduce(y)
        return y

    def all_gather(self, x, dim=-1):
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        self.d.all_gather(parts, x.detach().contiguous())
        return torch.cat(parts, dim)


def _tp_run(rank, world, group):
    lm = build(rank, world, group)
    sft = sft_for(lm, group)
    set_adapters(sft, world=world)
    sft.begin(dict(lr=1e-2))
    r = sft.micro(dict(BATCH, norm=1.0, check=True))
    sft.run.reduce_grads()
    grads = {}
    for n, (layer, lo) in sft.adapters.items():
        o, _ = sft.run.spans[(n, "A")]
        grads[n] = dict(kind=lo.kind, idx=layer.shard_idx,
                        A=sft.run.flat.grad[o:o + lo.A.numel()].view_as(lo.A).clone(),
                        B=sft.run.grad_of(n).view_as(lo.B).clone())
    sft.step()
    after = {n: (sft.run.view(n, "A").detach().clone(), sft.run.view(n, "B").detach().clone())
             for n in sft.adapters}
    return dict(loss=r["loss_sum"], grads=grads, after=after)


def _tp_child(rank, world, path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{path}/pg", rank=rank, world_size=world)
    try:
        torch.save(_tp_run(rank, world, GlooGroup()), f"{path}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_tp2_matches_tp1(tmp_path, monkeypatch):
    import torch.multiprocessing as mp
    for key in [k for k in os.environ if k.startswith("SUFFIX_")]:
        monkeypatch.delenv(key)
    ref = _tp_run(0, 1, None)
    mp.spawn(_tp_child, args=(2, str(tmp_path)), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"r{r}.pt", weights_only=False) for r in range(2)]
    for r in ranks:
        assert abs(r["loss"] - ref["loss"]) < 1e-4 * ref["loss"]
    for n, g in ref["grads"].items():
        parts = [r["grads"][n] for r in ranks]
        kind = parts[0]["kind"]
        for f in "AB":
            sharded = (kind == "col" and f == "B") or (kind == "row" and f == "A")
            for p in parts:
                want = (g[f][:, p["idx"]] if f == "B" else g[f][p["idx"]]) if sharded else g[f]
                torch.testing.assert_close(p[f], want, rtol=2e-3, atol=2e-5, msg=f"{n}.{f} ({kind})")
        rep = {"col": 0, "row": 1}.get(kind)
        if rep is not None:   # replicated factor: bit-identical after the host AdamW step
            assert torch.equal(ranks[0]["after"][n][rep], ranks[1]["after"][n][rep])


# ---------------------------------------------------------------------------
# (g) HTTP state machine
# ---------------------------------------------------------------------------
class Tok:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return ([2] if add_special_tokens else []) + [3 + (ord(c) % (V - 3)) for c in text]

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}|" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")


class FakeEngine:
    def __init__(self, sft):
        self.sft, self.resets = sft, 0
        self.output_processor = SimpleNamespace(get_num_unfinished_requests=lambda: 0)

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        assert method == "suffix_sft"
        return [W.rpc(*args)]

    def get_tokenizer(self):
        return Tok()

    async def reset_prefix_cache(self):
        self.resets += 1

    def generate(self, *a, **kw):
        raise RuntimeError("no engine in tests")


def _state(tmp_path, sft):
    W.SFT = sft
    cfg = Config(dir=str(tmp_path), max_tokens=64, min_free_gib=0.0, rank=4, alpha=8.0,
                 targets=M.DEFAULT_TARGETS)
    s = A.State(FakeEngine(sft), cfg, SimpleNamespace())
    s.info = {k: v for k, v in W.rpc("info").items() if k != "ms"}
    return s


CHATS = [{"messages": [{"role": "user", "content": f"q{i}"},
                       {"role": "assistant", "content": f"answer {i}!"}]} for i in range(3)]


def test_api_train_checkpoint_load_restore(tmp_path):
    lm = build()
    sft = sft_for(lm, rank=4)
    base = _served(lm)
    s = _state(tmp_path, sft)
    try:
        res = asyncio.run(s.train(dict(examples=CHATS + [{"prompt": "x", "completion": "yz"}],
                                       steps=4, batch_size=4, learning_rate=5e-2)))
        assert res["served"] and res["loss"] < res["loss_first"]
        assert res["parity_before"]["served_nll"] is None and s.engine.resets == 1
        trained = _served(lm)
        assert not torch.allclose(trained, base)
        run_dir = s.hub.directory / res["checkpoint"]
        assert (run_dir / "adapter-tp0-of-1.safetensors").is_file()
        assert json.loads((s.hub.directory / "latest.json").read_text())["run_id"] == res["run_id"]
        assert s.status["state"] == "idle" and s.status["checkpoint"]["run_id"] == res["run_id"]
        asyncio.run(s.load("base"))
        assert torch.equal(_served(lm), base)
        assert json.loads((s.hub.directory / "latest.json").read_text())["checkpoint"] is None
        asyncio.run(s.load(res["run_id"]))
        assert torch.equal(_served(lm), trained)
        # a restarted server restores latest.json before serving
        sft.reset({})
        s2 = _state(tmp_path, sft)
        asyncio.run(s2.restore())
        assert s2.status["state"] == "idle" and torch.equal(_served(lm), trained)
    finally:
        W.SFT = None


def test_api_validation_and_exclusivity(tmp_path):
    from fastapi import HTTPException
    s = _state(tmp_path, sft_for(build()))
    try:
        for bad in (dict(examples=[]), dict(examples=CHATS, steps=0),
                    dict(examples=CHATS, learning_rate=float("nan")),
                    dict(examples=[{"messages": [{"role": "user", "content": "only"}]}]),
                    dict(examples=[{"prompt": "x" * 80, "completion": "y"}])):
            with pytest.raises(HTTPException) as e:
                asyncio.run(s.train(bad))
            assert e.value.status_code == 400
        s.status["state"] = "restore_failed"
        with pytest.raises(HTTPException) as e:
            asyncio.run(s.train(dict(examples=CHATS)))
        assert e.value.status_code == 409

        async def busy():
            A._S = s
            s.status["state"] = "idle"
            s.running = asyncio.ensure_future(asyncio.sleep(0.2))
            req = SimpleNamespace(json=lambda: _aval({"examples": CHATS}))
            with pytest.raises(HTTPException) as e2:
                await A.train_route(req)
            assert e2.value.status_code == 409
            await s.running
        asyncio.run(busy())
    finally:
        W.SFT, A._S = None, None


async def _aval(v):
    return v


def test_batch_rows_never_predicts_across_segments():
    segs, rows, tgt = M.batch_rows([3, 2], [1, 1, 1, 1, 1], "cpu")
    assert segs == [(0, 3), (3, 5)]
    assert tgt.tolist() == [1, 2, 4] and rows.tolist() == [0, 1, 3]


def test_pack_schedule_encode():
    mbs = A.pack([([1] * 5, [0] * 5), ([2] * 4, [1] * 4), ([3] * 9, [1] * 9)], 10)
    assert [mb["seg_lens"] for mb in mbs] == [[5, 4], [9]]
    sch = A.schedule(5, 4, 3, seed=1)
    assert len(sch) == 4 and all(len(b) == 3 for b in sch)
    assert sorted(sch[0] + sch[1][:2]) == [0, 1, 2, 3, 4]      # epochs cover every example
    ids, mask = A.encode(Tok(), CHATS[0])
    prompt = Tok().apply_chat_template(CHATS[0]["messages"][:1], add_generation_prompt=True)
    assert mask == [0] * len(prompt) + [1] * (len(ids) - len(prompt)) and sum(mask) == len("answer 0!|")
    ids, mask = A.encode(Tok(), {"prompt": "ab", "completion": "c"})
    assert ids[0] == 2 and ids[-1] == Tok.eos_token_id and mask == [0, 0, 0, 1, 1]
