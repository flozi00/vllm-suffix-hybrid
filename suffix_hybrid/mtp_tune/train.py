# SPDX-License-Identifier: Apache-2.0
"""Training-mode functional replay of the MTP layer + the adapter trainer.

Why a functional replay: the live vLLM modules run custom kernels (fused
norms, Triton HC glue, fused MoE, paged sparse attention) that have no
autograd and read the paged KV cache. The replay below re-states the MTP
layer forward in plain torch over a WINDOW of captured positions, calling
the live frozen weights through two autograd Functions:

* ``FrozenLinear``: forward = the layer's own ``quant_method.apply`` (exact
  serving numerics for bf16 / FP8 / NVFP4 / custom kernels); backward = dx
  only, from the EFFECTIVE weight probed through the same apply in K-chunks
  (``apply(layer, I[k0:k1])`` = W_eff^T rows). No dW, no persistent copy.
  Exact for bf16 and FP8 (identity rows quantize exactly); W4A4 NVFP4 has a
  per-input-column scale error <= one e4m3 ulp (~6 %) in the GRADIENT only.
* ``FrozenExperts``: routed experts from per-expert DEQUANTIZED tiles
  (FP8 128x128 block via kernels.fp8_moe.expert_w, or float), forward and
  backward (dx + d topk_weights), one expert resident at a time: no BF16
  shadow of the expert bank. FP8 activation quant (x, h) is emulated in the
  forward (fp8_moe.qdq) and straight-through in the backward.

Attention is dense causal over the window (``WindowAttention``); window
positions carry their TRUE absolute positions (RoPE exact). Context before
the window is not visible (serving sees the whole prefix through the KV
cache) and QSA/DSA top-k sparsity is not applied (exact while the index
top-k >= visible keys). Both are fidelity gaps measured on silicon by the
depth-1 parity metric (replay argmax vs the draft the server produced).

Draft chain (gate replay, depths 1..k): depth j at anchor p is the query
at position p+j-1 with input (next_hidden of depth j-1 at p, x_{p+j}); it
attends depth-1 keys at positions <= p plus its own chain keys (depths
2..j) at anchor p only - exactly the KV a serving draft step sees.
Teacher forcing with the TRUE tokens equals the served chain on every
accepted prefix, so ``accepted length = #leading correct depths``.

Families: ``Qwen4ExpFunctional`` (vllm/models/qwen4_exp/nvidia/mtp.py
@ v0.30.0: residual_linear_shared input fusion, HyperConnection gated
residual, QSA gated attention, Qwen3Next MoE + gated shared expert, HC
final mixer). GLM-5.3 (GlmMoeDsaForCausalLM -> deepseek_v32/nvidia/mtp.py:
enorm/hnorm/eh_proj, MLA + DSA indexer, grouped sigmoid routing) is not
implemented yet: ``build_functional`` raises and tuning stays off.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from suffix_hybrid.mtp_tune.lora import (
    SingleGroup, allreduce_grads, base_apply, copy_to_tp, gather_from_tp,
    reduce_from_tp, tp_kind)

PROBE_CHUNK = 512


# ---------------------------------------------------------------------------
# frozen ops
# ---------------------------------------------------------------------------
class FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer):
        ctx.layer = layer
        ctx.k = x.shape[-1]
        with torch.no_grad():
            return base_apply(layer, x)

    @staticmethod
    def backward(ctx, dy):
        if not ctx.needs_input_grad[0]:
            return None, None
        k = ctx.k
        dx = dy.new_empty(*dy.shape[:-1], k)
        with torch.no_grad():
            for k0 in range(0, k, PROBE_CHUNK):
                k1 = min(k, k0 + PROBE_CHUNK)
                eye = torch.zeros(k1 - k0, k, dtype=dy.dtype, device=dy.device)
                eye[:, k0:k1].fill_diagonal_(1.0)
                wt = base_apply(ctx.layer, eye)          # [c, N] = W_eff^T rows
                dx[..., k0:k1] = dy @ wt.to(dy.dtype).T
        return dx, None


def expert_weights(handle, e):
    """(w13 [2I, H], w2 [H, I]) f32 of local expert e."""
    if handle["kind"] == "fp8":
        from suffix_hybrid.kernels.fp8_moe import expert_w
        return (expert_w(handle["w13"], handle["w13_s"], e),
                expert_w(handle["w2"], handle["w2_s"], e))
    return handle["w13"][e].float(), handle["w2"][e].float()


def _qdq(x, handle, stage):
    if handle["kind"] != "fp8" or not handle.get("act_qdq", True):
        return x
    from suffix_hybrid.kernels.fp8_moe import qdq
    return qdq(x, handle.get("ue8m0", False), stage)


def _groups(ids, e_count):
    """{local expert: flat pair indices}; ids < 0 or >= e_count are off-rank."""
    flat = ids.reshape(-1)
    out = {}
    for e in torch.unique(flat).tolist():
        if 0 <= e < e_count:
            out[e] = (flat == e).nonzero().squeeze(1)
    return out


class FrozenExperts(torch.autograd.Function):
    """y[t] = sum_k tw[t,k] * W2_e (silu(g) * u), [g; u] = W13_e x[t],
    e = ids[t,k] (local id; off-rank pairs contribute 0)."""

    @staticmethod
    def _expert(handle, e, xe):
        w13, w2 = expert_weights(handle, e)
        gu = _qdq(xe, handle, "x") @ w13.T
        i = gu.shape[-1] // 2
        g, u = gu[:, :i], gu[:, i:]
        h = F.silu(g) * u
        return w13, w2, g, u, h, _qdq(h, handle, "h") @ w2.T

    @staticmethod
    def forward(ctx, x, tw, ids, handle):
        ctx.handle = handle
        ctx.save_for_backward(x, tw, ids)
        m, topk = ids.shape
        xf = x.float()
        out = torch.zeros(m, x.shape[-1], dtype=torch.float32, device=x.device)
        with torch.no_grad():
            for e, pairs in _groups(ids, handle["w13"].shape[0]).items():
                y = FrozenExperts._expert(handle, e, xf[pairs // topk])[-1]
                out.index_add_(0, pairs // topk, y * tw.reshape(-1)[pairs, None].float())
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, tw, ids = ctx.saved_tensors
        handle = ctx.handle
        m, topk = ids.shape
        xf, dyf = x.float(), dy.float()
        dx = torch.zeros_like(xf)
        dtw = torch.zeros(m * topk, dtype=torch.float32, device=x.device)
        with torch.no_grad():
            for e, pairs in _groups(ids, handle["w13"].shape[0]).items():
                rows = pairs // topk
                w13, w2, g, u, h, y = FrozenExperts._expert(handle, e, xf[rows])
                d = dyf[rows]
                dtw[pairs] = (d * y).sum(-1)
                dh = (d * tw.reshape(-1)[pairs, None].float()) @ w2
                sg = torch.sigmoid(g)
                dg = dh * u * (sg * (1 + g * (1 - sg)))
                du = dh * g * sg
                dx.index_add_(0, rows, torch.cat([dg, du], -1) @ w13)
        return dx.to(x.dtype), dtw.view_as(tw).to(tw.dtype), None, None


def experts_handle(experts):
    """Frozen routed-experts handle from a vLLM FusedMoE layer (or a test
    fake with the same attribute surface)."""
    if getattr(experts, "enable_eplb", False):
        raise NotImplementedError("EPLB redundant experts not supported")
    w13, w2 = experts.w13_weight, experts.w2_weight
    if w13.dtype.is_floating_point and w13.element_size() >= 2:
        return dict(kind="float", w13=w13, w2=w2)
    if w13.dtype == torch.float8_e4m3fn:
        s13 = getattr(experts, "w13_weight_scale_inv", None)
        s2 = getattr(experts, "w2_weight_scale_inv", None)
        e, two_i, h = w13.shape
        shape13, shape2 = (e, two_i // 128, h // 128), (e, h // 128, two_i // 256)
        if s13 is None or tuple(s13.shape) != shape13 or s13.dtype != torch.float32:
            from suffix_hybrid.kernels import fp8_moe
            info, sc = fp8_moe.layer_info(experts, {})
            if sc is None or sc[0] is None or sc[1] is None:
                raise NotImplementedError(f"FP8 expert scales unresolvable: {info.get('scales')}")
            s13, s2 = sc
        if tuple(s2.shape) != shape2:
            raise NotImplementedError(f"FP8 w2 scale shape {tuple(s2.shape)} != {shape2}")
        ue8m0 = False
        try:
            from suffix_hybrid.kernels.fp8_moe import e8m0_used
            ue8m0 = e8m0_used()
        except Exception:
            pass
        return dict(kind="fp8", w13=w13, w13_s=s13, w2=w2, w2_s=s2, ue8m0=ue8m0,
                    act_qdq=True)
    # ponytail: NVFP4 routed experts (GLM prod) need nvfp4_moe._expert_w + the
    # global scale; add with the GLM family.
    raise NotImplementedError(f"routed expert dtype {w13.dtype} not supported")


# ---------------------------------------------------------------------------
# functional primitives
# ---------------------------------------------------------------------------
def gemma_rms(x, w, eps, groups=1):
    """GemmaRMSNorm (x * rsqrt(mean x^2 + eps) * (1 + w)), per group."""
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], groups, shp[-1] // groups)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y.reshape(shp) * (1.0 + w.float())).to(x.dtype)


def rope_neox(x, cos_sin, rotary_dim):
    """x [N, heads, D]; cos_sin [N, rotary_dim] = (cos | sin) halves."""
    half = rotary_dim // 2
    cos, sin = cos_sin[:, None, :half].float(), cos_sin[:, None, half:].float()
    rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = rot[..., :half], rot[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return torch.cat([out.to(x.dtype), rest], -1)


class WindowAttention:
    """Batched window attention for depth 1 and the draft chain.

    Rows are [B, T] flattened to N = B*T. Depth 1: causal within each
    window. Depth j >= 2 (row p = anchor p): depth-1 keys at rows <= p plus
    this anchor's own chain keys of depths 2..j. Windows are right-padded,
    so causal masking alone keeps padding out of every real row.
    """

    def __init__(self, b, t):
        self.b, self.t, self.kv = b, t, []

    def __call__(self, q, k, v, scale):
        b, t = self.b, self.t
        q = q.reshape(b, t, q.shape[-2], q.shape[-1]).float()
        k = k.reshape(b, t, k.shape[-2], k.shape[-1]).float()
        v = v.reshape(b, t, v.shape[-2], v.shape[-1]).float()
        self.kv.append((k, v))
        rep = q.shape[2] // k.shape[2]
        k1, v1 = (z.repeat_interleave(rep, 2) for z in self.kv[0])
        s = torch.einsum("bqhd,bkhd->bhqk", q, k1) * scale
        causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~causal, float("-inf"))
        extra = [(ki.repeat_interleave(rep, 2), vi.repeat_interleave(rep, 2))
                 for ki, vi in self.kv[1:]]
        if extra:
            se = torch.stack([(q * ki).sum(-1) for ki, _ in extra], -1) * scale
            s = torch.cat([s, se.permute(0, 2, 1, 3)], -1)       # [b,h,q,t+E]
        p = torch.softmax(s, -1)
        o = torch.einsum("bhqk,bkhd->bqhd", p[..., :t], v1)
        for i, (_, vi) in enumerate(extra):
            o = o + p[..., t + i].permute(0, 2, 1)[..., None] * vi
        return o.reshape(b * t, o.shape[2], o.shape[3])


# ---------------------------------------------------------------------------
# vocab-parallel head
# ---------------------------------------------------------------------------
def _vocab_range(head, group):
    si = getattr(head, "shard_indices", None)
    if si is not None:
        return int(si.org_vocab_start_index), int(si.org_vocab_end_index)
    n = int(getattr(head, "num_embeddings_per_partition", head.weight.shape[0]))
    r = group.rank_in_group
    return r * n, (r + 1) * n


def vocab_parallel_ce(local, target, start, end, group):
    """CE over a vocab sharded by TP; local [N, V_local] (padding columns
    beyond end - start already -inf), target [N] global ids. Autograd-correct
    across ranks."""
    local = local.float()
    m = local.detach().max(-1, keepdim=True).values
    if group.world_size > 1:
        m = group.all_gather(m.contiguous(), dim=-1).max(-1, keepdim=True).values
    sumexp = reduce_from_tp((local - m).exp().sum(-1), group)
    idx = target - start
    hit = (idx >= 0) & (idx < end - start)   # never a padding column
    t = local.gather(-1, idx.clamp(0, local.shape[-1] - 1)[:, None]).squeeze(-1)
    t = reduce_from_tp(torch.where(hit, t, torch.zeros_like(t)), group)
    return sumexp.log() + m.squeeze(-1) - t


def vocab_parallel_argmax(local, start, group):
    v, i = local.float().max(-1)
    if group.world_size == 1:
        return i + start
    both = group.all_gather(torch.stack([v, (i + start).float()], -1).contiguous(), dim=-1)
    both = both.view(*v.shape, group.world_size, 2)
    best = both[..., 0].argmax(-1, keepdim=True)
    return both[..., 1].gather(-1, best).squeeze(-1).long()


# ---------------------------------------------------------------------------
# Qwen4Exp MTP (vllm/models/qwen4_exp/nvidia/mtp.py @ v0.30.0)
# ---------------------------------------------------------------------------
class Qwen4ExpFunctional:
    def __init__(self, model, group=None):
        self.m, self.mp = model, model.model
        self.g = group or SingleGroup()
        self.eps = float(self.mp.pre_fc_norm_embedding.variance_epsilon)
        self._const = {}
        self.head = model.lm_head
        self.vstart, self.vend = _vocab_range(self.head, self.g)
        layer = self.mp.layers[0]
        self.experts = experts_handle(layer.mlp.experts)

    # frozen small tensors: detached normal-tensor copies (vLLM params may be
    # inference tensors, which autograd refuses to save)
    def c(self, t):
        key = id(t)
        if key not in self._const:
            with torch.inference_mode(False):
                self._const[key] = t.detach().clone()
        return self._const[key]

    def lin(self, layer, x, params):
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        kind = tp_kind(layer)
        if kind == "col":
            x = copy_to_tp(x, self.g)
        y = FrozenLinear.apply(x, layer)
        lora = getattr(layer, "_mtp_lora", None)
        if lora is not None:
            ab = params.get(lora.name) if params is not None else None
            y = y + (lora.delta(x) if ab is None else lora.delta(x, *ab))
        bias = getattr(layer, "bias", None)
        skip = getattr(layer, "skip_bias_add", False)
        if kind == "row":
            if getattr(layer, "reduce_results", True):
                y = reduce_from_tp(y, self.g)
            if bias is not None and not skip:
                y = y + self.c(bias)
        else:
            if bias is not None and not skip:
                y = y + self.c(bias)
            if kind == "col" and getattr(layer, "gather_output", False):
                y = gather_from_tp(y, self.g)
        return y.reshape(*lead, y.shape[-1])

    def hc_combine_and_mix(self, hc, res, block, inj, params):
        n, c, hs = res.shape[0], hc.hc_count, hc.hidden_size
        add = block.float()[:, None, :]
        if inj is not None:
            add = add * (2.0 * torch.sigmoid(inj.float() / c))[:, :, None]
        out = (res.float().view(n, c, hs) + add).to(res.dtype)
        w = self.c(hc.hc_norm.weight)
        y = gemma_rms(out, w.view(-1, hs) if w.numel() == c * hs else w, self.eps)
        y = y.reshape(n, c * hs)
        if hc.use_combine:
            d = self.lin(hc.input_mix_weight_down_block_inject, y, params)
            low, new_inj = d[:, :hc.lora_rank], d[:, hc.lora_rank:hc.lora_rank + c]
        else:
            low, new_inj = self.lin(hc.input_mix_weight_down, y, params), None
        gate = self.lin(hc.input_mix_weight_up, F.silu(low.float() / c).to(y.dtype), params)
        blk = (torch.sigmoid(gate.float()) * y.float()).view(n, c, hs).mean(1)
        return out.reshape(n, c * hs), blk.to(y.dtype), new_inj

    def attention(self, sa, x, pos, attn, params):
        n = x.shape[0]
        nh, nkv, d = sa.num_heads, sa.num_kv_heads, sa.head_dim
        qkv = self.lin(sa.qkv_proj, x, params)
        qg, k, v = qkv.split([2 * nh * d, nkv * d, nkv * d], -1)
        q, gate = qg.reshape(n, nh, 2 * d).chunk(2, -1)
        eps = float(sa.q_norm.variance_epsilon)
        q = gemma_rms(q, self.c(sa.q_norm.weight), eps)
        k = gemma_rms(k.reshape(n, nkv, d), self.c(sa.k_norm.weight), eps)
        re = sa.rotary_emb
        with torch.no_grad():
            cs = re.cos_sin_cache[pos.long()].to(q.device)
        q, k = rope_neox(q, cs, re.rotary_dim), rope_neox(k, cs, re.rotary_dim)
        o = attn(q, k, v.reshape(n, nkv, d), sa.scaling).to(x.dtype)
        o = o.reshape(n, nh * d) * torch.sigmoid(gate.reshape(n, nh * d).float()).to(x.dtype)
        return self.lin(sa.o_proj, o, params)

    def moe(self, mlp, x, params):
        ex = mlp.experts
        logits = self.lin(mlp.gate, x, params).float()
        tw, ids = torch.softmax(logits, -1).topk(int(ex.top_k), -1)
        if getattr(ex, "renormalize", True):
            tw = tw / tw.sum(-1, keepdim=True)
        emap = getattr(ex, "expert_map", None)
        local = ids if emap is None else emap.to(ids.device)[ids]
        # x and tw are replicated but only this rank's experts see them: both
        # grads are partial sums over the EP/TP ranks
        partial = FrozenExperts.apply(copy_to_tp(x, self.g), copy_to_tp(tw.to(x.dtype), self.g),
                                      local, self.experts)
        full = None
        se = getattr(mlp, "shared_expert", None)
        if se is not None:
            gu = self.lin(se.gate_up_proj, x, params)
            g, u = gu.chunk(2, -1)
            sd = self.lin(se.down_proj, (F.silu(g.float()) * u.float()).to(x.dtype), params)
            gate = torch.sigmoid(self.lin(mlp.shared_expert_gate, x, params).float()).to(x.dtype)
            if tp_kind(se.down_proj) == "row" and not getattr(se.down_proj, "reduce_results", True):
                # vLLM gates the TP-PARTIAL down_proj output: the gate's grad
                # is a partial sum too
                partial = partial + copy_to_tp(gate, self.g) * sd
            else:
                full = gate * sd
        out = reduce_from_tp(partial, self.g)
        return out if full is None else out + full

    def step(self, h, tok, pos, attn, params=None):
        """One MTP forward over N rows -> (logit hidden [N,H], next hidden)."""
        mp = self.mp
        n, c, hs = h.shape[0], mp.hc_count, mp.hidden_size
        with torch.no_grad():
            e = mp.embed_input_ids(tok)
        e = gemma_rms(e, self.c(mp.pre_fc_norm_embedding.weight), self.eps)
        e = self.lin(mp.fc_embedding, e, params)
        hh = gemma_rms(h, self.c(mp.pre_fc_norm_hidden.weight), self.eps).view(n, c, hs)
        hh = self.lin(mp.fc_hidden, hh, params).reshape(n, c * hs)
        layer = mp.layers[0]
        hid, blk, inj = self.hc_combine_and_mix(layer.attn_hyper_connection, hh, e, None, params)
        a = self.attention(layer.self_attn, blk, pos, attn, params)
        hid, blk, inj = self.hc_combine_and_mix(layer.mlp_hyper_connection, hid, a, inj, params)
        mo = self.moe(layer.mlp, blk, params)
        multi, sample, _ = self.hc_combine_and_mix(mp.hyper_connection_mixer, hid, mo, inj, params)
        return sample, multi

    def logits_local(self, x):
        # vocab-parallel head = column-parallel over the vocab: the replicated
        # hidden's grad is the SUM of every rank's vocab slice
        y = FrozenLinear.apply(copy_to_tp(x, self.g), self.head)
        n_org = self.vend - self.vstart
        if y.shape[-1] > n_org:     # vocab padding rows of the shard
            y = torch.cat([y[..., :n_org], torch.full_like(y[..., n_org:], float("-inf"))], -1)
        return y


def build_functional(model, group=None):
    mp = getattr(model, "model", None)
    if mp is not None and hasattr(mp, "hyper_connection_mixer") and hasattr(mp, "fc_hidden"):
        return Qwen4ExpFunctional(model, group)
    raise NotImplementedError(
        f"MTP functional replay not implemented for {type(model).__name__} "
        "(GLM-5.3 = DeepseekV32MTP: MLA + DSA, phase 2)")


# ---------------------------------------------------------------------------
# windows -> batch, forward passes
# ---------------------------------------------------------------------------
def collate(windows, device, dtype):
    """list of dict(h [L,Hd], tok [L], pos [L], anchor [L] bool) -> padded
    batch dict with lens [B]."""
    b, t = len(windows), max(int(w["tok"].shape[0]) for w in windows)
    hd = windows[0]["h"].shape[-1]
    h = torch.zeros(b, t, hd, dtype=dtype)
    tok = torch.zeros(b, t, dtype=torch.long)
    pos = torch.zeros(b, t, dtype=torch.long)
    anchor = torch.zeros(b, t, dtype=torch.bool)
    lens = torch.zeros(b, dtype=torch.long)
    for i, w in enumerate(windows):
        n = int(w["tok"].shape[0])
        h[i, :n] = w["h"].to(dtype)
        tok[i, :n] = w["tok"].long()
        pos[i, :n] = w["pos"].long()
        pos[i, n:] = pos[i, n - 1] + 1 + torch.arange(t - n)  # monotone padding
        anchor[i, :n] = w["anchor"]
        lens[i] = n
    return {k: v.to(device) for k, v in dict(h=h, tok=tok, pos=pos, anchor=anchor,
                                             lens=lens).items()}


def _shift(x, j):
    """x[:, p + j] at column p (zero padded)."""
    if j == 0:
        return x
    return torch.cat([x[:, j:], torch.zeros_like(x[:, :j])], 1)


def depth1_loss(fam, batch, params):
    """Teacher-forced MTP CE: input (h_p, x_{p+1}) -> label x_{p+2}."""
    b, t = batch["tok"].shape
    attn = WindowAttention(b, t)
    col = torch.arange(t, device=batch["tok"].device)[None]
    mask = batch["anchor"] & (col + 2 < batch["lens"][:, None])
    n = int(mask.sum())
    if n == 0:   # identical on every TP rank (same broadcast batch)
        return torch.zeros((), device=mask.device), 0
    sample, _ = fam.step(batch["h"].reshape(b * t, -1), _shift(batch["tok"], 1).reshape(-1),
                         batch["pos"].reshape(-1), attn, params)
    m = mask.reshape(-1)
    ce = vocab_parallel_ce(fam.logits_local(sample[m]), _shift(batch["tok"], 2).reshape(-1)[m],
                           fam.vstart, fam.vend, fam.g)
    return ce.mean(), n


@torch.no_grad()
def chain_replay(fam, batch, params, k):
    """Greedy draft chain replay. Returns dict(correct [k,B,T] bool,
    full [B,T] bool = anchors with labels for all k depths, pred1 [B,T])."""
    b, t = batch["tok"].shape
    attn = WindowAttention(b, t)
    col = torch.arange(t, device=batch["tok"].device)[None]
    h = batch["h"].reshape(b * t, -1)
    correct, pred1 = [], None
    for j in range(1, k + 1):
        sample, h = fam.step(h, _shift(batch["tok"], j).reshape(-1),
                             (batch["pos"] + (j - 1)).reshape(-1), attn, params)
        pred = vocab_parallel_argmax(fam.logits_local(sample), fam.vstart, fam.g).view(b, t)
        pred1 = pred if pred1 is None else pred1
        correct.append(pred == _shift(batch["tok"], j + 1))
    full = batch["anchor"] & (col + k + 1 < batch["lens"][:, None])
    return dict(correct=torch.stack(correct), full=full, pred1=pred1)


# ---------------------------------------------------------------------------
# trainer (candidate adapter only)
# ---------------------------------------------------------------------------
class Trainer:
    """AdamW over fp32 master copies of every adapter's (A, B); micro-steps
    accumulate `accum` backward passes per optimizer step; the micro-batch
    (windows per micro-step) adapts to hit `target_ms`."""

    def __init__(self, fam, loras, lr=1e-4, wd=0.0, accum=4, target_ms=50.0,
                 max_windows=8, offload=False, clip=1.0):
        self.fam, self.loras = fam, loras
        self.accum, self.target_ms, self.max_windows, self.clip = accum, target_ms, max_windows, clip
        with torch.inference_mode(False):
            self.params = {n: (l.A.detach().float().clone().requires_grad_(True),
                               l.B.detach().float().clone().requires_grad_(True))
                           for n, l in loras.items()}
        flat = [p for ab in self.params.values() for p in ab]
        self.opt = torch.optim.AdamW(flat, lr=lr, weight_decay=wd)
        self.offload = offload   # ponytail: states stay on the adapter device;
        # host offload only pays off for large ranks (tiny at r=16), add if needed
        self.micro, self.steps, self.windows_per_micro = 0, 0, 1
        self.ms_per_window = None
        self.loss_ewma = None

    def serving_params(self):
        """Candidate (A, B) cast to each live adapter's dtype (what a
        promotion would copy)."""
        return {n: (a.detach().to(self.loras[n].A.dtype), b.detach().to(self.loras[n].B.dtype))
                for n, (a, b) in self.params.items()}

    def micro_step(self, batch):
        t0 = time.perf_counter()
        with torch.inference_mode(False), torch.enable_grad():
            loss, n = depth1_loss(self.fam, batch, self.params)
            if n:
                (loss / self.accum).backward()
        self.micro += 1
        if self.micro % self.accum == 0:
            reps = [(l, self.params[name][0 if l.replicated_factor() == "A" else 1])
                    for name, l in self.loras.items() if l.replicated_factor()]
            allreduce_grads(reps, self.fam.g)
            flat = [p for ab in self.params.values() for p in ab]
            torch.nn.utils.clip_grad_norm_(flat, self.clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.steps += 1
        lv = float(loss.detach())
        self.loss_ewma = lv if self.loss_ewma is None else 0.9 * self.loss_ewma + 0.1 * lv
        return lv, (time.perf_counter() - t0) * 1e3

    def adapt(self, ms, n_windows):
        per = ms / max(n_windows, 1)
        self.ms_per_window = per if self.ms_per_window is None else 0.8 * self.ms_per_window + 0.2 * per
        self.windows_per_micro = max(1, min(self.max_windows,
                                            int(math.floor(self.target_ms / self.ms_per_window))))
