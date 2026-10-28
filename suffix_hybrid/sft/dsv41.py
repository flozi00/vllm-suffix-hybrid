# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 (DeepseekV4Model on the nvidia path) training twin.

vLLM's V4.1 forward cannot carry autograd: every hot op (mHC TileLang,
fused q/kv norm + RoPE + KV insert, compressor, indexer, FlashInfer sparse
MLA into a preallocated buffer, MoE runner, Engram Triton) writes into
buffers or KV caches, and without attention metadata the attention, the
compressor and Engram silently produce zeros / are skipped. So TrainMode
does not run ``DeepseekV4Model.forward`` at all: ``forward`` below drives
the decoder layers itself, in torch, reading the LIVE modules' weights
(linears still go through the serving quant kernels via TrainMode.linear,
i.e. FrozenLinear + the LoRA delta). Math source: vLLM 39983a5e7
(models/deepseek_v41/**, model_executor/kernels/mhc/torch.py); map in
inference-console plugin-harness/results/dsv41-sft-2026-10-24/MAP.md.

Per layer l (state: hc stream S [T, 4, H] bf16, post/comb/pre mixes):
  S = mhc_post(y_prev, S, post, comb); S = engram(S) on Engram layers
  post, comb, li, pre_a = mhc_pre_delayed(S, hc_attn_fn, pre_mix=pre_prev)
  y = attention(attn_norm(li)); S = mhc_post(y, S, post, comb)
  post, comb, li, pre_f = mhc_pre_delayed(S, hc_ffn_fn, pre_mix=pre_a)
  y = moe(ffn_norm(li))
end: S = mhc_post(y, S, ...); h = norm(sum_i pre_f[i] S_i)
Attention: own 128-token window of RoPE'd kv (fp8 V4-record qdq, STE) +
the kv-source layer's compressed entries (ratio 2: per-dim softmax pooling
of pairs; ratio 1: per token) selected by the index-source layer's top-512
(torch twin of the DSA indexer, no gradient; every visible entry when
there are <= 512), one softmax with the per-head sink, V = K, inverse RoPE,
grouped wo_a (bf16 bmm), wo_b. MoE: sqrt(softplus) scores, top-6 by
score + bias, renormalised to routed_scaling_factor; frozen NVFP4 experts
with the swiglu clamp; clamped shared MLP. Every layer is checkpointed
(TrainMode._checkpointed: inputs parked on the host within the budget).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from suffix_hybrid.mtp_tune.lora import copy_to_tp, reduce_from_tp
from suffix_hybrid.mtp_tune.train import FrozenExperts, experts_handle

TARGETS = ("*.attn.fused_wqa_wkv", "*.attn.wq_b", "*.attn.wo_b",
           "*.shared_experts.gate_up_proj", "*.shared_experts.down_proj")
Q_CHUNK = 256              # query rows per attention / indexer chunk
IMAGE_TOKEN_ID = 129264
FP8_MAX = 448.0
ROPE = 64                  # qk_rope_head_dim: the bf16 tail of a V4 KV record


def is_dsv4(inner) -> bool:
    layers = getattr(inner, "layers", None)
    return (layers is not None and len(layers) > 0 and hasattr(layers[0], "hc_attn_fn")
            and hasattr(getattr(layers[0], "attn", None), "fused_wqa_wkv"))


def check(lm, vllm_config=None):
    """Raise NotImplementedError for a V4.1 configuration the twin does not model."""
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is not None:
        if pc.pipeline_parallel_size > 1:
            raise NotImplementedError("pipeline parallel")
        if int(getattr(pc, "prefill_context_parallel_size", 1) or 1) > 1:
            raise NotImplementedError("prefill context parallel")
        if getattr(pc, "enable_eplb", False):
            raise NotImplementedError("EPLB")
    for i, layer in enumerate(lm.model.layers):
        a, ffn = layer.attn, layer.ffn
        if getattr(layer, "use_sequence_parallel", False) or getattr(a, "use_pcp", False):
            raise NotImplementedError(f"layer {i}: sequence / context parallel")
        if a.compress_ratio not in (0, 1, 2):
            raise NotImplementedError(f"layer {i}: compress ratio {a.compress_ratio}")
        if a.wo_a.weight.dtype != torch.bfloat16:
            raise NotImplementedError(f"layer {i}: wo_a weight {a.wo_a.weight.dtype} (expected the BF16 "
                                      "emulation weight)")
        if getattr(ffn, "scoring_func", None) != "sqrtsoftplus":
            raise NotImplementedError(f"layer {i}: MoE scoring {getattr(ffn, 'scoring_func', None)}")
        if getattr(ffn.gate, "tid2eid", None) is not None:
            raise NotImplementedError(f"layer {i}: hash routing")
        if getattr(ffn, "use_native_mega_moe", False):
            raise NotImplementedError(f"layer {i}: native mega MoE")
        experts_handle(ffn.experts)


def install(lm, adapters):
    """Serving side: an adapted wq_b needs the bf16 normed q, so the fused
    q/kv RMSNorm + MXFP8 quant producer (a cached_property per attention)
    is switched off where it would feed one."""
    n = 0
    for layer in lm.model.layers:
        a = layer.attn
        if getattr(a.wq_b, "sft_A", None) is not None:
            a.__dict__["_can_fuse_query_quant"] = False
            n += 1
    return n


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _diag(key):
    """Dev-only ablations (SUFFIX_SFT_DIAG=1 /v1/sft/diag, the dsv41_sft lab tool):
    dsv4_no_attn / _no_moe / _no_engram / _no_comp / _no_kvqdq."""
    from suffix_hybrid.sft import model as M
    return bool(M.DIAG.get(key))


def _lin(m, x):
    out = m(x)
    return out[0] if isinstance(out, tuple) else out


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _mhc_pre(layer, S, fn, scale, base, pre_mix=None, x=None):
    """vllm.model_executor.kernels.mhc.torch.mhc_pre_delayed_torch (inlined:
    same op order)."""
    hc = S.shape[1]
    xf = (S.flatten(1) if x is None else x).float()
    mixes = (xf @ fn.float().t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.rms_norm_eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + layer.hc_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * layer.hc_post_alpha
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * scale[2] + base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + layer.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + layer.hc_eps)
    for _ in range(layer.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + layer.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + layer.hc_eps)
    li = S[:, 0] if pre_mix is None else (pre_mix.unsqueeze(-1) * S.float()).sum(1).to(S.dtype)
    return post.unsqueeze(-1), comb, li, pre


def _mhc_post(y, S, post, comb):
    mixed = torch.einsum("tij,tih->tjh", comb.float(), S.float())
    return (mixed + post.float() * y.unsqueeze(-2).float()).to(S.dtype)


def _ste(x, q):
    return x + (q - x).detach()


def _kv_qdq(x):
    """SM120 fp8_ds_mla V4 record: 448 NoPE dims e4m3 with one UE8M0 scale
    per 64 (exponent = ceil(log2(max(amax, 1e-4) / 448))), 64 RoPE dims bf16.
    Straight-through: the value serving reads, the identity gradient."""
    nd = x.shape[-1] - ROPE
    with torch.no_grad():
        nope = x[..., :nd].float().unflatten(-1, (nd // 64, 64))
        amax = nope.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        s = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
        q = (nope / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
        q = torch.cat([q.flatten(-2).to(x.dtype), x[..., nd:]], -1)
    return _ste(x, q)


def _rope(emb, pos, x, inverse=False):
    """GPT-J RoPE on the last rotary_dim dims (the module's own torch path)."""
    return emb.forward_native(pos, x, None, inverse=inverse)[0]


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def _compress(a, x, pos, segs):
    """kv-source compressor: (C [sum Tc, 512] RoPE'd + qdq, latent [sum Tc, 512])."""
    c = a.compressor
    r = a.compress_ratio
    ks = F.linear(x.float(), c.fused_wkv_wgate.weight.float())       # serving: bf16 mm, fp32 out
    lat, cpos = [], []
    for s, e in segs:
        n = e - s
        if r == 1:
            pooled, p = ks[s:e, :a.head_dim], pos[s:e]
        else:
            m = n // 2
            pair = ks[s:s + 2 * m].view(m, 2, -1)
            kv, sc = pair[..., :a.head_dim], pair[..., a.head_dim:]
            pooled = (kv * torch.softmax(sc, dim=1)).sum(1)
            p = pos[s:s + 2 * m:2]
        lat.append(_rms(pooled, c.norm.weight, c.rms_norm_eps).to(torch.bfloat16))
        cpos.append(p)
    latent = torch.cat(lat)
    cp = torch.cat(cpos)
    C = _kv_qdq(_rope(a.rotary_emb, cp, latent.unsqueeze(1))[:, 0])
    return C, latent, cp


def _index_k(a, latent, cp):
    ix = a.indexer
    with torch.no_grad():
        k = ix.k_norm(_lin(ix.wk, latent))
        return _rope(a.indexer_rotary_emb, cp, k.unsqueeze(1))[:, 0]


def _c_offsets(segs, r):
    out, off = [], 0
    for s, e in segs:
        out.append(off)
        off += (e - s) // r
    return out


def _select(a, qr, x, pos, segs, KI, r):
    """Indexer top-k per token (segment-local compressed ids, -1 padded) or
    None when every segment has <= topk compressed entries (all visible
    entries attended: vLLM's short-context path, identical selection)."""
    ix = a.indexer
    topk = ix.topk_tokens
    if max((e - s) // r for s, e in segs) <= topk:
        return None
    with torch.no_grad():
        q = _lin(ix.wq_b, qr).view(-1, ix.n_head, ix.head_dim)
        q = _rope(a.indexer_rotary_emb, pos, q).float()
        w = _lin(ix.weights_proj, x).float()                          # [T, heads]
        sel = torch.full((qr.shape[0], topk), -1, dtype=torch.long, device=qr.device)
        co = _c_offsets(segs, r)
        for (s, e), o in zip(segs, co):
            n, nc = e - s, (e - s) // r
            k = KI[o:o + nc].float()
            for r0 in range(0, n, Q_CHUNK):
                r1 = min(n, r0 + Q_CHUNK)
                vis = (pos[s + r0:s + r1, None] + 1) // r                     # visible count
                j = torch.arange(nc, device=qr.device)
                lg = (torch.einsum("thd,jd->thj", q[s + r0:s + r1], k).relu()
                      * w[s + r0:s + r1, :, None]).sum(1)
                lg = lg.masked_fill(j[None] >= vis, float("-inf"))
                kk = min(topk, nc)
                v, idx = lg.topk(kk, -1)
                sel[s + r0:s + r1, :kk] = torch.where(torch.isfinite(v), idx, -1)
        return sel


def _attend(q, kw, lo, C, cmask, r0, sink, scale, window):
    """One query chunk, no per-row key copies: q [t, H, D] at segment
    positions r0..r0+t-1; kw [r1 - lo, D] = the segment's window keys at
    positions lo..r1-1 (contiguous slice, band-masked); C [nc, D] the
    segment's compressed entries or None, cmask [t, nc] bool = visible and
    selected. One softmax over window + compressed + the per-head sink."""
    t, h, _ = q.shape
    dev = q.device
    p = r0 + torch.arange(t, device=dev)
    d = p[:, None] - (lo + torch.arange(kw.shape[0], device=dev))[None]
    qf = q.float()
    kwf = kw.float()
    s = torch.einsum("thd,kd->thk", qf, kwf) * scale
    s = s.masked_fill(~((d >= 0) & (d < window))[:, None, :], float("-inf"))
    parts = [s]
    if C is not None:
        Cf = C.float()
        sc = torch.einsum("thd,jd->thj", qf, Cf) * scale
        parts.append(sc.masked_fill(~cmask[:, None, :], float("-inf")))
    parts.append(sink.float().view(1, h, 1).expand(t, h, 1))
    pr = torch.softmax(torch.cat(parts, -1), -1)
    nw = kw.shape[0]
    o = torch.einsum("thk,kd->thd", pr[..., :nw], kwf)
    if C is not None:
        o = o + torch.einsum("thj,jd->thd", pr[..., nw:nw + C.shape[0]], Cf)
    return o.to(q.dtype)


def _cmask(p, nc, r, sel):
    """[t, nc] bool: compressed entry j visible (j < (p + 1) // r) and, with
    an indexer selection sel [t, K] (-1 padded), selected."""
    with torch.no_grad():
        j = torch.arange(nc, device=p.device)
        m = j[None] < ((p[:, None] + 1) // r)
        if sel is not None:
            pick = torch.zeros(p.shape[0], nc + 1, dtype=torch.bool, device=p.device)
            pick.scatter_(1, torch.where(sel >= 0, sel, nc), True)
            m &= pick[:, :nc]
        return m


def attention(tm, a, x, pos, segs, carry):
    """Returns (y [T, H] all-reduced, carry'): carry = (C, KI, sel) of the
    current kv-source / index-source group."""
    g = tm.g
    T = x.shape[0]
    qr_kv = _lin(a.fused_wqa_wkv, x)
    qr, kv = qr_kv.split([a.q_lora_rank, a.head_dim], -1)
    qr, kv = _rms(qr, a.q_norm.weight, a.eps), _rms(kv, a.kv_norm.weight, a.eps)
    q = _lin(a.wq_b, qr).view(T, a.n_local_heads, a.head_dim)
    q = _rope(a.rotary_emb, pos, q)
    kvr = _kv_qdq(_rope(a.rotary_emb, pos, kv.unsqueeze(1))[:, 0])
    kvl = copy_to_tp(kvr, g)
    C, KI, sel = carry
    r = a.compress_ratio
    if r:
        if a.compressor is not None:                 # kv source: new compressed cache
            C, latent, cp = _compress(a, x, pos, segs)
            KI = _index_k(a, latent, cp) if (a.indexer is not None and a.indexer.owns_k) else None
        if a.indexer is not None:                    # index source: new selection
            sel = _select(a, qr, x, pos, segs, KI, r)
    Cl = copy_to_tp(C, g) if r else None
    if r and _diag("dsv4_no_comp"):
        Cl = Cl * 0
    sink = a.attn_sink[:a.n_local_heads]
    co = _c_offsets(segs, r) if r else None
    out = []
    W = a.window_size
    for i, (s, e) in enumerate(segs):
        n = e - s
        nc = n // r if r else 0
        Cs = Cl[co[i]:co[i] + nc] if r else None
        # score rows bounded at ~64M fp32 elements per chunk (quadratic in nc: ponytail,
        # a gather path for > 64k-token segments if training ever needs them)
        step = max(16, min(Q_CHUNK, (1 << 26) // max(1, a.n_local_heads * (nc + Q_CHUNK + W))))
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            lo = max(0, r0 - W + 1)
            cm = None
            if r:
                cm = _cmask(pos[s + r0:s + r1], nc, r, None if sel is None else sel[s + r0:s + r1])
            args = (q[s + r0:s + r1], kvl[s + lo:s + r1], lo, Cs, cm, r0, sink, a.scale, W)
            if torch.is_grad_enabled():
                out.append(checkpoint(_attend, *args, use_reentrant=False))
            else:
                out.append(_attend(*args))
    o = torch.cat(out)
    o = _rope(a.rotary_emb, pos, o, inverse=True)
    G = a.n_local_groups
    o = o.reshape(T, G, -1)
    z = torch.einsum("tgk,grk->tgr", o, a.wo_a.weight.view(G, a.o_lora_rank, -1))
    y = _lin(a.wo_b, z.flatten(1))
    if _diag("dsv4_no_attn"):
        y = y * 0
    return y, (C if r else None, KI if r else None, sel if r else None)


# ---------------------------------------------------------------------------
# MoE / Engram
# ---------------------------------------------------------------------------
def moe(tm, sft, ffn, x):
    g = tm.g
    logits = F.linear(x.float(), ffn.gate.weight.float())
    w = torch.sqrt(F.softplus(logits))
    with torch.no_grad():
        bias = ffn.gate.e_score_correction_bias
        ids = (w + (bias.float() if bias is not None else 0.0)).topk(ffn.n_activated_experts, -1).indices
    tw = w.gather(-1, ids)
    if ffn.renormalize:
        tw = tw / tw.sum(-1, keepdim=True)
    tw = tw * float(ffn.routed_scaling_factor)
    re_ = ffn.experts.routed_experts
    emap = getattr(re_, "expert_map", None)
    local = ids if emap is None else emap.to(ids.device)[ids]
    cache = sft.__dict__.setdefault("_sft_experts", {})
    if id(re_) not in cache:
        cache[id(re_)] = dict(experts_handle(ffn.experts), limit=float(ffn.swiglu_limit or 0.0))
    out = FrozenExperts.apply(copy_to_tp(x, g), copy_to_tp(tw.to(x.dtype), g), local, cache[id(re_)])
    se = ffn.shared_experts
    if se is not None:
        gu = _lin(se.gate_up_proj, x).float()
        gt, up = gu.chunk(2, -1)
        lim = ffn.swiglu_limit
        if lim:
            gt, up = gt.clamp(max=lim), up.clamp(-lim, lim)
        out = out + _lin(se.down_proj, (F.silu(gt) * up).to(x.dtype))
    out = reduce_from_tp(out, g)
    return out * 0 if _diag("dsv4_no_moe") else out


def engram_kv(inner, eng, hashes):
    with torch.no_grad():
        rows = eng.embed_tokens(hashes[:, eng.layer_hash_index])
        return _lin(eng.wkv, rows.flatten(-2))


def engram(eng, S, kv):
    """_fused_engram_post_wkv_kernel in torch (per hc copy c)."""
    t, hc, d = S.shape
    h = S.float()
    key = kv[:, :hc * d].view(t, hc, d).float()
    val = kv[:, hc * d:].float()
    qk = eng.q_weight.float() * eng.k_weight.float()
    dot = (h * qk * key).sum(-1)
    dot = dot * torch.rsqrt(h.square().mean(-1) + eng.eps) * torch.rsqrt(key.square().mean(-1) + eng.eps) \
        / math.sqrt(d)
    gi = torch.sqrt(torch.clamp(dot.abs(), min=eng.clamp_value))
    gate = torch.sigmoid(torch.where(dot < 0, -gi, gi))
    return (h + gate[..., None] * val[:, None, :]).to(S.dtype)


def engram_hashes(inner, ids, pos, segs):
    hs = getattr(inner, "engram_hash", None)
    if hs is None:
        return None
    with torch.no_grad():
        qsl = torch.tensor([0] + [e for _, e in segs], dtype=torch.int32, device=ids.device)
        dead = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        lb = ids.new_full((len(segs), int(hs.lookback_depth)), -1)
        return hs(ids, pos, qsl, dead, lb, torch.zeros_like(lb, dtype=torch.bool), None, None)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def forward(tm, inner, input_ids, positions, **_):
    """The language model's final normed hidden rows [T, H] with autograd."""
    sft, segs = tm.s, tm.segments
    ids, pos = input_ids.reshape(-1), positions.reshape(-1)
    if bool((ids == IMAGE_TOKEN_ID).any()):
        raise ValueError("image placeholder tokens in an SFT example (text-only training)")
    e = inner.embed_input_ids(ids)
    hashes = engram_hashes(inner, ids, pos, segs)
    layers = list(inner.layers)
    tm.dsv4_engaged = tm.__dict__.get("dsv4_engaged", 0) + 1

    def make(idx, layer):
        a, ffn = layer.attn, layer.ffn

        def body(y, S, post, comb, pre, C, KI, sel):
            if sel is not None:          # int carry: host checkpoints only move float inputs
                sel = sel.to(pos.device, non_blocking=True)
            if idx == 0:
                S = y.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
                post, comb, li, pre_a = _mhc_pre(layer, S, layer.hc_attn_fn_broadcast, layer.hc_attn_scale,
                                                 layer.hc_attn_base, None, x=y)
            else:
                S = _mhc_post(y, S, post, comb)
                if layer.engram is not None:
                    if hashes is None:
                        raise RuntimeError("Engram layer without hashes")
                    if not _diag("dsv4_no_engram"):
                        S = engram(layer.engram, S, engram_kv(inner, layer.engram, hashes))
                post, comb, li, pre_a = _mhc_pre(layer, S, layer.hc_attn_fn, layer.hc_attn_scale,
                                                 layer.hc_attn_base, pre)
            x = _rms(li, layer.attn_norm.weight, layer.attn_norm.variance_epsilon)
            y, (C, KI, sel) = attention(tm, a, x, pos, segs, (C, KI, sel))
            S = _mhc_post(y, S, post, comb)
            post, comb, li, pre_f = _mhc_pre(layer, S, layer.hc_ffn_fn, layer.hc_ffn_scale,
                                             layer.hc_ffn_base, pre_a)
            x = _rms(li, layer.ffn_norm.weight, layer.ffn_norm.variance_epsilon)
            y = moe(tm, sft, ffn, x)
            return y, S, post, comb, pre_f, C, KI, sel
        return body

    y, S, post, comb, pre, C, KI, sel = e, None, None, None, None, None, None, None
    for idx, layer in enumerate(layers):
        run = tm._checkpointed(make(idx, layer), last=idx == len(layers) - 1, mod=layer)
        y, S, post, comb, pre, C, KI, sel = run(y, S, post, comb, pre, C, KI, sel)
    S = _mhc_post(y, S, post, comb)
    h = (pre.unsqueeze(-1) * S.float()).sum(1).to(S.dtype)
    return _rms(h, inner.norm.weight, inner.norm.variance_epsilon)
