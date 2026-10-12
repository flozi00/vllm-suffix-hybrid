# SPDX-License-Identifier: Apache-2.0
"""Shrunk Qwen4ExpMTP with vLLM's attribute surface, for CPU tests.

Same module tree / attribute names the functional replay reads from the
live vllm/models/qwen4_exp/nvidia/mtp.py modules (fc_embedding, fc_hidden,
pre_fc_norm_*, layers[0].{attn,mlp}_hyper_connection, self_attn.{qkv_proj,
o_proj,q_norm,k_norm,rotary_emb}, mlp.{gate,shared_expert,shared_expert_gate,
experts}, hyper_connection_mixer, lm_head.shard_indices). Linears carry
vLLM's LinearBase attributes (tp_size, gather_output / input_is_parallel,
reduce_results, *_size_per_partition) and compute through
quant_method.apply. Routed experts are FP8 e4m3 with 128x128 block scales
(the qwen draft layout). ``build(tp_rank, tp_size, group)`` shards ONE
seeded full model the way vLLM does (column: output rows, row: input
columns, experts: EP by expert, vocab: by rank with padding).
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from suffix_hybrid.kernels.fp8_moe import block_quant

H, HC, HEADS, KVH, D, ROT, E, TOPK, I, SI, V, VPAD, HCR = (
    128, 2, 4, 2, 32, 16, 4, 2, 128, 128, 96, 8, 8)


class Unquantized:
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class Fp8Block:
    """Weight-only FP8 128x128 block method (dequant GEMM) - a quantized
    dense method the probe backward must handle."""

    def apply(self, layer, x, bias=None):
        w = layer.weight.float().reshape(layer.weight.shape[0] // 128, 128, -1, 128)
        w = (w * layer.weight_scale_inv[:, None, :, None]).reshape(layer.weight.shape)
        return F.linear(x, w.to(x.dtype), bias)


class Lin(nn.Module):
    def __init__(self, w, kind="rep", tp_rank=0, tp_size=1, group=None, gather=False,
                 reduce=True, fp8=False):
        super().__init__()
        self.tp_size, self.tp_rank, self.group = (tp_size, tp_rank, group) if kind != "rep" else (1, 0, None)
        self.bias, self.skip_bias_add = None, False
        if kind == "col":
            self.gather_output = gather
        elif kind == "row":
            self.input_is_parallel, self.reduce_results = True, reduce
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        if fp8:
            q, s = block_quant(w)
            self.weight = nn.Parameter(q, requires_grad=False)
            self.weight_scale_inv = s
            self.quant_method = Fp8Block()
        else:
            self.weight = nn.Parameter(w.clone(), requires_grad=False)
            self.quant_method = Unquantized()

    def forward(self, x):   # vLLM LinearBase semantics, return_bias=False
        y = self.quant_method.apply(self, x, None)
        if self.tp_size > 1 and hasattr(self, "gather_output") and self.gather_output:
            y = self.group.all_gather(y, dim=-1)
        if self.tp_size > 1 and hasattr(self, "input_is_parallel") and self.reduce_results:
            y = self.group.all_reduce(y)
        return y


class Norm(nn.Module):
    def __init__(self, n, g):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, generator=g) * 0.1, requires_grad=False)
        self.variance_epsilon = 1e-6


class HCMod(nn.Module):
    def __init__(self, g, use_combine):
        super().__init__()
        self.hc_count, self.hidden_size, self.lora_rank, self.use_combine = HC, H, HCR, use_combine
        self.hc_norm = Norm(HC * H, g)
        if use_combine:
            pad = (-(HCR + HC)) % 16
            self.input_mix_weight_down_block_inject = Lin(torch.randn(HCR + HC + pad, HC * H, generator=g) * 0.05)
        else:
            self.input_mix_weight_down = Lin(torch.randn(HCR, HC * H, generator=g) * 0.05)
        self.input_mix_weight_up = Lin(torch.randn(HC * H, HCR, generator=g) * 0.3)


def _cos_sin(max_pos=4096):
    inv = 1.0 / (10000 ** (torch.arange(0, ROT, 2).float() / ROT))
    f = torch.arange(max_pos).float()[:, None] * inv[None]
    return torch.cat([f.cos(), f.sin()], -1)


def build(tp_rank=0, tp_size=1, group=None, seed=0, fp8_dense=False):
    g = torch.Generator().manual_seed(seed)
    r, t = tp_rank, tp_size

    def rnd(*s, std=0.08):
        return torch.randn(*s, generator=g) * std

    def col(w, rows, **kw):   # shard_idx = global output rows held locally
        lin = Lin(w[rows], "col", r, t, group, **kw)
        lin.shard_idx = rows
        return lin

    def row(w, cols, **kw):   # shard_idx = global input cols held locally
        lin = Lin(w[:, cols], "row", r, t, group, **kw)
        lin.shard_idx = cols
        return lin

    def span(n):          # contiguous local slice of n global units
        return torch.arange(r * n // t, (r + 1) * n // t)

    m = nn.Module()
    mp = m.model = nn.Module()
    mp.hidden_size, mp.hc_count = H, HC
    emb_full = rnd(V, H, std=1.0)
    vs = span(V)
    emb_local = emb_full[vs]

    def embed_input_ids(ids, _e=emb_local, _s=int(vs[0]), _n=len(vs)):
        loc = ids - _s
        hit = (loc >= 0) & (loc < _n)
        out = _e[loc.clamp(0, _n - 1)] * hit[:, None]
        return group.all_reduce(out) if t > 1 else out
    mp.embed_input_ids = embed_input_ids
    mp.pre_fc_norm_embedding = Norm(H, g)
    mp.pre_fc_norm_hidden = Norm(HC * H, g)
    hs = span(H)
    mp.fc_embedding = col(rnd(H, H), hs, gather=True, fp8=fp8_dense)
    mp.fc_hidden = col(rnd(H, H), hs, gather=True)
    layer = nn.Module()
    mp.layers = nn.ModuleList([layer])
    layer.attn_hyper_connection = HCMod(g, True)
    layer.mlp_hyper_connection = HCMod(g, True)
    sa = layer.self_attn = nn.Module()
    sa.num_heads, sa.num_kv_heads, sa.head_dim, sa.scaling = HEADS // t, KVH // t, D, D ** -0.5
    hl, kl = span(HEADS), span(KVH)
    qg_rows = torch.cat([torch.arange(h * 2 * D, (h + 1) * 2 * D) for h in hl.tolist()])
    k_rows = HEADS * 2 * D + torch.cat([torch.arange(h * D, (h + 1) * D) for h in kl.tolist()])
    v_rows = k_rows + KVH * D
    sa.qkv_proj = col(rnd(HEADS * 2 * D + 2 * KVH * D, H), torch.cat([qg_rows, k_rows, v_rows]))
    o_cols = torch.cat([torch.arange(h * D, (h + 1) * D) for h in hl.tolist()])
    sa.o_proj = row(rnd(H, HEADS * D), o_cols)
    sa.q_norm, sa.k_norm = Norm(D, g), Norm(D, g)
    sa.rotary_emb = SimpleNamespace(cos_sin_cache=_cos_sin(), rotary_dim=ROT)
    mlp = layer.mlp = nn.Module()
    mlp.gate = Lin(rnd(E, H, std=0.3))
    mlp.shared_expert_gate = Lin(rnd(1, H))
    se = mlp.shared_expert = nn.Module()
    si = span(SI)
    se.gate_up_proj = col(rnd(2 * SI, H), torch.cat([si, SI + si]))
    se.down_proj = row(rnd(H, SI), si, reduce=False)
    ex = mlp.experts = nn.Module()
    ex.top_k, ex.renormalize = TOPK, True
    w13, s13, w2, s2 = [], [], [], []
    for _ in range(E):
        q, s = block_quant(rnd(2 * I, H, std=0.06))
        w13.append(q), s13.append(s)
        q, s = block_quant(rnd(H, I, std=0.06))
        w2.append(q), s2.append(s)
    el = span(E)
    ex.w13_weight, ex.w13_weight_scale_inv = torch.stack(w13)[el], torch.stack(s13)[el]
    ex.w2_weight, ex.w2_weight_scale_inv = torch.stack(w2)[el], torch.stack(s2)[el]
    emap = torch.full((E,), -1, dtype=torch.long)
    emap[el] = torch.arange(len(el))
    ex.expert_map = emap if t > 1 else None
    mp.hyper_connection_mixer = HCMod(g, False)
    head_full = rnd(V, H, std=0.2)
    head_local = torch.cat([head_full[vs], torch.zeros(VPAD, H)])  # vocab padding rows
    m.lm_head = Lin(head_local)
    m.lm_head.shard_indices = SimpleNamespace(org_vocab_start_index=int(vs[0]),
                                              org_vocab_end_index=int(vs[-1]) + 1)
    return m


def hidden_table(seed=1):
    return torch.randn(V, HC * H, generator=torch.Generator().manual_seed(seed))
