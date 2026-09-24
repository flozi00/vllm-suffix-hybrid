# SPDX-License-Identifier: Apache-2.0
"""On-silicon numerics oracle for the SM120 NVFP4-KV fa2 route.

Gate (c) of dossiers/gemma-hd512-nvfp4.md: must pass on the real GPU before
any pool boots ``--kv-cache-dtype nvfp4`` at a head_dim the route has not yet
been proven at (first target: gemma-4 full-attn head_dim=512).

What it exercises is exactly the production byte path, nothing re-implemented:
  store  = vLLM's own ``_C_cache_ops.reshape_and_cache_flash(..., "nvfp4")``
           into a (B, 2H, N, full_dim) HND cache (FlashInferImpl layout),
  views  = vLLM's ``nvfp4_split_data_scale`` (the forward() split),
  kernel = FlashInfer ``fa2`` Batch{Prefill,Decode} wrappers with
           kv_data_type=uint8 + kv_cache_sf (patch H6/H7/H9 route), JIT'd with
           the V-SF de-swizzle define (patch ensure_deswizzle_flag()).
The reference dequantizes the cache BYTES in torch (e2m1 LUT x e4m3 SF x
global scale; K SF linear, V SF 4-token swizzled per the store kernel) and runs
fp32 attention — independent of the kernel under test. Gates:
  * decoder sanity: dequant(cache) vs the bf16 input  (store/decoder bug)
  * kernel vs dequant reference, tight                 (kernel/SF/layout bug)
  * kernel vs bf16-input attention, loose               (end-to-end sanity)
mm_prefix cases (gate for SUFFIX_SM120_NVP4KV_MM=1): same path plus the
builder's packed custom mask (the backend's own _nvfp4_kv_mm_prefill_mask
text, installed on the wrapper exactly like patch H18); the reference mask is
re-derived here from TRITON_ATTN's compute_kv_seq_mask (causal AND window, OR
image-range bidirectional clamped to the window), and each case must also
differ measurably from plain causal (proves the mask was applied).

own_* cases (gate for SUFFIX_SM120_NVP4KV_OWN_ATTN=1, --own): the same store
and views through OUR K2-NVFP4 kernel (own_attn.run: cutile-rs, src/
nvfp4_attn_gpu.rs), gated against the same dequant reference AND compared to
the FA2 route's output on identical inputs (cos_vs_fa2); graph cases capture
our op once and replay it after changing seq_lens (grid must not depend on
KV lengths). --bench prints microseconds per call, ours vs the FA2 route vLLM
uses today (decode wrapper for q_len 1, paged prefill for spec verify).

Run in the pod/CI image (PYTHONPATH=/plugins):
    python -m nvfp4_kv_patch.oracle [--shapes 512:16:2,256:16:8] [--json]
        [--cases mm_]   (substring filter on case names)
        [--own]         (K2-NVFP4 cases instead of the FA2 cases)
        [--bench [--batches 1,2,4,8,16,32] [--kvs 1024,4096,16384,65536,131072]
                 [--q-lens 1,9] [--max-gb 6] [--iters 20]]
Exit 0 = PASS, 1 = FAIL, 2 = could not run (no SM120 / import error).
"""

import argparse
import json
import math
import sys

from . import SF_VEC_SIZE, ensure_deswizzle_flag, is_sm120, mm_helper_namespace

PAGE = 64
# e2m1 code -> value (bit 3 = sign).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# Gates. The kernel accumulates in fp32 over bf16-rounded P, so vs an exact
# fp32 reference over the SAME dequantized bytes it must be near-exact.
COS_KERNEL_MIN = 0.9995
REL_KERNEL_MAX = 2e-2
COS_DECODE_MIN = 0.98     # nvfp4 round-trip of gaussian data is ~0.99
COS_E2E_MIN = 0.95        # attention output vs un-quantized inputs
REL_MM_VS_CAUSAL_MIN = 0.05  # mm reference must differ from causal
COS_VS_FA2_MIN = 0.999    # ours vs FA2 on identical bytes (both ~exact)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _rel(a, b):
    return float((a.float() - b.float()).abs().max()
                 / b.float().abs().max().clamp_min(1e-30))


def dequant_side(data, sf, swizzled, global_scale):
    """(B,H,N,D/2) uint8 + (B,H,N,D/16) e4m3 views -> (B,H,N,D) fp32."""
    import torch

    lut = torch.tensor(_E2M1, device=data.device)
    d = data.to(torch.int64)
    vals = torch.stack((lut[d & 0xF], lut[d >> 4]), dim=-1).flatten(-2)
    s = sf.float()
    if swizzled:
        # Store wrote logical (t, g) at swizzle_scale_offset(t, g, S) in the
        # per-(page, head) [N*S] region; gather it back.
        n, g = s.shape[2], s.shape[3]
        t = torch.arange(n, device=s.device).view(n, 1)
        c = torch.arange(g, device=s.device).view(1, g)
        grp = g // 4
        flat = ((t // 4) * 4 + c // grp) * g + (c % grp) * 4 + t % 4
        s = s.flatten(2)[:, :, flat.flatten()].view(s.shape)
    return vals * s.repeat_interleave(SF_VEC_SIZE, dim=-1) * global_scale


def _allowed(lq, lk, window, ranges, device):
    """TRITON_ATTN compute_kv_seq_mask for q at the tail: (causal AND
    window) OR (q, k in one inclusive range [s, e], s < e, AND window —
    gemma-4 clamp; no window = full attention)."""
    import torch

    q = torch.arange(lk - lq, lk, device=device).view(lq, 1)
    k = torch.arange(lk, device=device).view(1, lk)
    sw = (q - k < window) if window > 0 else torch.ones_like(q - k, dtype=torch.bool)
    ok = (k <= q) & sw
    for s, e in ranges:
        if s < e:
            ok |= (q >= s) & (q <= e) & (k >= s) & (k <= e) & sw
    return ok


def _ref_attn(q, k, v, sm_scale, allowed=None):
    """q (Lq,Hq,D), k/v (L,Hkv,D) fp32; causal with q at the tail unless an
    explicit (Lq, L) ``allowed`` mask is given."""
    import torch

    lq, hq, _ = q.shape
    lk, hkv, _ = k.shape
    if allowed is None:
        allowed = _allowed(lq, lk, -1, (), q.device)
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    logits.masked_fill_(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)


def _make_kv(n_tok, hkv, d, pattern, gen):
    import torch

    x = torch.randn(n_tok, hkv, d, generator=gen, device="cuda")
    if pattern == "adversarial":
        # Per 16-element block magnitudes over 1e-4..1e2 (e4m3 SF subnormals
        # through large), plus all-zero blocks (SF=0 path).
        g = x.view(n_tok, hkv, d // SF_VEC_SIZE, SF_VEC_SIZE)
        mag = 10 ** (torch.rand(g.shape[:-1], generator=gen, device="cuda")
                     * 6 - 4)
        mag[torch.rand(mag.shape, generator=gen, device="cuda") < 0.05] = 0
        x = (g * mag.unsqueeze(-1)).view(n_tok, hkv, d)
    return x.to(torch.bfloat16)


def run_case(shape, pattern, kv_lens, q_len, scales, wrapper_kind, seed=0,
             page=PAGE, window=-1, mm=None, own=False, graph=False):
    """window: sliding window in tokens (-1 = full); mm: per-request lists of
    inclusive image ranges (absolute positions), or None for plain causal.
    own: gate OUR kernel (K2-NVFP4) instead of FA2 (FA2 output is kept as
    the cos_vs_fa2 comparison); graph: run ours via CUDA-graph replay with
    seq_lens changed after capture."""
    import torch
    from vllm.utils.torch_utils import (nvfp4_kv_cache_full_dim,
                                        nvfp4_split_data_scale)

    import flashinfer

    d, hq, hkv = shape
    k_scale, v_scale = scales
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages_per = [math.ceil(n / page) for n in kv_lens]
    num_pages = sum(pages_per) + 3  # spare pages stay garbage-free zeros
    page_ids = torch.randperm(num_pages, generator=gen, device="cuda")

    cache = torch.zeros(num_pages, 2 * hkv, page, nvfp4_kv_cache_full_dim(d),
                        dtype=torch.uint8, device="cuda")
    keys, vals, slots, indices, indptr, last = [], [], [], [], [0], []
    cur = 0
    for n, p in zip(kv_lens, pages_per):
        ids = page_ids[cur:cur + p]
        cur += p
        pos = torch.arange(n, device="cuda")
        slots.append(ids[pos // page] * page + pos % page)
        keys.append(_make_kv(n, hkv, d, pattern, gen))
        vals.append(_make_kv(n, hkv, d, pattern, gen))
        indices.append(ids)
        indptr.append(indptr[-1] + p)
        last.append(n - (p - 1) * page)

    # --- production store (FlashInferImpl.do_kv_cache_update, nvfp4 branch)
    k_cache, v_cache = cache.transpose(1, 2).split(hkv, dim=-2)
    ks = torch.tensor([k_scale], dtype=torch.float32, device="cuda")
    vs = torch.tensor([v_scale], dtype=torch.float32, device="cuda")
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        torch.cat(keys), torch.cat(vals), k_cache, v_cache,
        torch.cat(slots).to(torch.int64), "nvfp4", ks, vs)

    # --- production read views (FlashInferImpl.forward, HND identity order)
    k_side, v_side = cache.split(hkv, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)

    k_deq = dequant_side(k_data, k_sf, False, k_scale)  # (B,H,N,D)
    v_deq = dequant_side(v_data, v_sf, True, v_scale)

    sm_scale = 1.0 / math.sqrt(d)
    qs = [torch.randn(q_len, hq, d, generator=gen, device="cuda")
          .to(torch.bfloat16) for _ in kv_lens]
    q = torch.cat(qs)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor(indptr, **i32)
    kv_indices = torch.cat(indices).to(torch.int32)
    kv_last = torch.tensor(last, **i32)
    dtypes = dict(q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
                  o_data_type=torch.bfloat16)
    if wrapper_kind == "decode":
        # q_len > 1: FlashInfer's uniform multi-token decode (fa2 tensor-core
        # path, causal within each request) — the candidate FA2 route for
        # FULL-graph spec verify.
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "HND", use_tensor_cores=True, backend="fa2")
        w.plan(kv_indptr, kv_indices, kv_last, hq, hkv, d, page,
               pos_encoding_mode="NONE", sm_scale=sm_scale,
               window_left=window - 1 if window > 0 else -1,
               **(dict(q_len_per_req=q_len) if q_len > 1 else {}), **dtypes)
    else:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "HND", backend="fa2")
        qo_indptr = torch.arange(len(kv_lens) + 1, **i32) * q_len
        w.plan(qo_indptr, kv_indptr, kv_indices, kv_last, hq, hkv, d, page,
               causal=True, sm_scale=sm_scale, window_left=window - 1
               if window > 0 else -1, **dtypes)
        if mm is not None:
            # Production mask builder + install, exactly as patch H18 does.
            build = mm_helper_namespace()["_nvfp4_kv_mm_prefill_mask"]
            res = build(dict(enumerate(mm)), 0, qo_indptr.cpu(),
                        torch.tensor(kv_lens), "cuda")
            assert res is not None, "mm case must need the mask"
            w._custom_mask_buf, w._mask_indptr_buf = res
    out = w.run(q, (k_data, v_data), k_scale=k_scale, v_scale=v_scale,
                kv_cache_sf=(k_sf, v_sf))
    torch.cuda.synchronize()
    fa2_out = None
    if own:
        fa2_out, out = out, _run_own(q, (k_data, k_sf, v_data, v_sf), indices,
                                     kv_lens, q_len, window, sm_scale, scales,
                                     graph)

    worst = {"cos_kernel": 1.0, "rel_kernel": 0.0, "cos_e2e": 1.0,
             "cos_decode": 1.0}
    if mm is not None:
        worst["rel_mm_vs_causal"] = 0.0
    for i, (n, ids) in enumerate(zip(kv_lens, indices)):
        # Gather through the page table (proves paging math, not just data).
        kd = k_deq[ids].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        vd = v_deq[ids].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        o = out[i * q_len:(i + 1) * q_len]
        allowed = _allowed(q_len, n, window, mm[i] if mm else (), q.device)
        ref = _ref_attn(qs[i], kd, vd, sm_scale, allowed)
        e2e = _ref_attn(qs[i], keys[i].float(), vals[i].float(),
                        sm_scale, allowed)
        if mm is not None:
            causal = _ref_attn(qs[i], kd, vd, sm_scale,
                               _allowed(q_len, n, window, (), q.device))
            worst["rel_mm_vs_causal"] = max(worst["rel_mm_vs_causal"],
                                            _rel(ref, causal))
        if fa2_out is not None:
            worst["cos_vs_fa2"] = min(worst.get("cos_vs_fa2", 1.0), _cos(
                o, fa2_out[i * q_len:(i + 1) * q_len]))
        worst["cos_kernel"] = min(worst["cos_kernel"], _cos(o, ref))
        worst["rel_kernel"] = max(worst["rel_kernel"], _rel(o, ref))
        worst["cos_e2e"] = min(worst["cos_e2e"], _cos(o, e2e))
        worst["cos_decode"] = min(worst["cos_decode"],
                                  _cos(kd, keys[i]), _cos(vd, vals[i]))
    # e2e/decode gates only mean something on well-scaled gaussian data; the
    # adversarial pattern deliberately loses precision in tiny blocks.
    strict = pattern == "gauss"
    worst["pass"] = (
        worst["cos_kernel"] >= COS_KERNEL_MIN
        and worst["rel_kernel"] <= REL_KERNEL_MAX
        and (not strict or worst["cos_decode"] >= COS_DECODE_MIN)
        and (not strict or worst["cos_e2e"] >= COS_E2E_MIN)
        and (mm is None
             or worst["rel_mm_vs_causal"] >= REL_MM_VS_CAUSAL_MIN)
        and worst.get("cos_vs_fa2", 1.0) >= COS_VS_FA2_MIN)
    return worst


def _block_table(indices, device):
    import torch

    bt = torch.zeros(len(indices), max(len(i) for i in indices),
                     dtype=torch.int32, device=device)
    for r, ids in enumerate(indices):
        bt[r, :len(ids)] = ids
    return bt


def _run_own(q, views, indices, kv_lens, q_len, window, sm_scale, scales,
             graph):
    """OUR kernel on the production views (own_attn.run = what the patched
    backend's DecodeWrapper.run calls)."""
    import torch

    from . import own_attn

    k_data, k_sf, v_data, v_sf = views
    d, hq, hkv = q.shape[2], q.shape[1], k_data.shape[1]
    own_attn.prepare(own_attn.native(), (d, hq, hkv, k_data.shape[2],
                                         window - 1 if window > 0 else -1),
                     (q_len,))
    bt = _block_table(indices, q.device)
    sl = torch.tensor(kv_lens, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    args = (q, k_data, k_sf, v_data, v_sf, bt, sl, out, q_len,
            window - 1 if window > 0 else -1, sm_scale * scales[0], scales[1])
    if not graph:
        own_attn.run(*args)
    else:
        # Capture with DIFFERENT lengths, then replay with the real ones: the
        # grid and workspace must not depend on seq_lens.
        sl.copy_((sl - q_len).clamp(min=q_len))
        own_attn.run(*args)  # eager warmup (module load outside capture)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            own_attn.run(*args)
        sl.copy_(torch.tensor(kv_lens, dtype=torch.int32, device=q.device))
        out.zero_()
        g.replay()
    torch.cuda.synchronize()
    return out


# (name, kv_lens, q_len, wrapper[, run_case kwargs]). kv lens hit: 1 token,
# exact page multiple, ragged tails, > 4 pages, split-KV scale (>=256-token
# chunks).
# mm ranges (inclusive, absolute) per request with q_len=300: image split
# across the chunk boundary + image longer than the 128 window; an image in
# the cached context (ignored) + a tiny one + one ending at the last token; a
# text-only row; an image running past kv_len (continues next chunk).
_MM = ([(380, 460), (470, 650)],
       [(100, 300), (1250, 1260), (1480, 1499)],
       [],
       [(1000, 1100)])
_MM_KV = (700, 1500, 333, 1030)
CASES = (
    ("decode", (1, 64, 65, 300, 777, 1024, 2049, 4100), 1, "decode"),
    ("spec_verify_k8", (9, 130, 700, 3001), 9, "prefill"),
    ("prefill", (1500,), 1500, "prefill"),
    ("prefix_cached", (1200, 2500), 300, "prefill"),
    ("mm_prefix_swa", _MM_KV, 300, "prefill", dict(window=128, mm=_MM)),
    ("mm_prefix_swa_p16", _MM_KV, 300, "prefill",
     dict(window=128, mm=_MM, page=16)),
    ("mm_prefix_full", _MM_KV, 300, "prefill", dict(mm=_MM)),
    ("mm_fresh_prefill", (900,), 900, "prefill",
     dict(window=128, mm=([(100, 400), (899, 950)],))),
)
# K2-NVFP4 (--own): decode q_len 1, MTP verify q_len 9 (uniform, as the H19
# spec-as-decode route feeds it), SWA window, page 16, and graph replay.
OWN_CASES = (
    ("own_decode", (1, 64, 65, 300, 777, 1024, 2049, 4100), 1, "decode",
     dict(own=True)),
    ("own_spec_verify_k8", (9, 130, 700, 3001), 9, "prefill", dict(own=True)),
    ("own_swa_verify_k8", (9, 700, 3001), 9, "prefill",
     dict(own=True, window=128)),
    ("own_swa_decode_p16", (1, 300, 2049), 1, "decode",
     dict(own=True, window=128, page=16)),
    ("own_graph_decode", (65, 777, 4100), 1, "decode",
     dict(own=True, graph=True)),
    ("own_graph_verify_k8", (130, 3001), 9, "prefill",
     dict(own=True, graph=True)),
    # FA2 decode wrapper at q_len 9 (q_len_per_req) vs the torch reference:
    # evidence for a graph-capturable FA2 verify path (no K2 involved).
    ("fa2_decode_verify_k8", (9, 130, 700, 3001), 9, "decode"),
    ("fa2_decode_verify_k8_swa", (9, 700, 3001), 9, "decode",
     dict(window=128)),
)
PATTERNS = (("gauss", (1.0, 1.0)), ("gauss", (0.5, 2.0)),
            ("adversarial", (1.0, 1.0)))


def _bench_cache(batch, kv, shape, page, device):
    """Random NVFP4 pages (valid 1.0 scales) + a shuffled page table."""
    import torch
    from vllm.utils.torch_utils import (nvfp4_kv_cache_full_dim,
                                        nvfp4_split_data_scale)

    d, _, hkv = shape
    per = math.ceil(kv / page)
    cache = torch.randint(0, 256, (batch * per + 1, 2 * hkv, page,
                                   nvfp4_kv_cache_full_dim(d)),
                          dtype=torch.uint8, device=device)
    k_side, v_side = cache.split(hkv, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)
    for sf in (k_sf, v_sf):
        sf.view(torch.uint8).fill_(0x38)  # e4m3 1.0
    ids = torch.randperm(batch * per, device=device, dtype=torch.int64)
    return (k_data, k_sf, v_data, v_sf), ids.view(batch, per).to(torch.int32)


def _time_us(fn, iters):
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def bench(shapes, batches, kvs, q_lens, max_gb, iters, page, as_json):
    """us/call, ours (graph-replayed: pure GPU time, as served in FULL decode
    graphs) vs the FA2 route vLLM runs today (decode wrapper, q_len 1; paged
    prefill wrapper, spec verify — eager in production, timed eager here)."""
    import torch

    import flashinfer

    from . import own_attn

    dev = "cuda"
    bw = 1792.0  # GB/s, RTX PRO 6000 Blackwell GDDR7 spec
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    dt = dict(q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
              o_data_type=torch.bfloat16)
    for shape in shapes:
        d, hq, hkv = shape
        for q_len in q_lens:
            for batch in batches:
                for kv in kvs:
                    kv_bytes = batch * kv * hkv * 2 * (d // 2 + d // 16)
                    if kv_bytes > max_gb * 2 ** 30:
                        continue
                    views, bt = _bench_cache(batch, kv, shape, page, dev)
                    q = torch.randn(batch * q_len, hq, d, device=dev).to(
                        torch.bfloat16)
                    sl = torch.full((batch,), kv, **i32)
                    per = bt.shape[1]
                    indptr = torch.arange(batch + 1, **i32) * per
                    last = torch.full((batch,), kv - (per - 1) * page, **i32)
                    sm = 1.0 / math.sqrt(d)
                    if q_len == 1:
                        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                            ws, "HND", use_tensor_cores=True, backend="fa2")
                        w.plan(indptr, bt.flatten(), last, hq, hkv, d, page,
                               pos_encoding_mode="NONE", sm_scale=sm, **dt)
                    else:
                        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                            ws, "HND", backend="fa2")
                        w.plan(torch.arange(batch + 1, **i32) * q_len, indptr,
                               bt.flatten(), last, hq, hkv, d, page,
                               causal=True, sm_scale=sm, **dt)
                    k_data, k_sf, v_data, v_sf = views
                    fa2 = _time_us(lambda: w.run(
                        q, (k_data, v_data), k_scale=1.0, v_scale=1.0,
                        kv_cache_sf=(k_sf, v_sf)), iters)
                    out = torch.empty_like(q)
                    args = (q, k_data, k_sf, v_data, v_sf, bt, sl, out,
                            q_len, -1, sm, 1.0)
                    own_attn.prepare(own_attn.native(),
                                     (d, hq, hkv, page, -1), (q_len,))
                    own_attn.run(*args)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        own_attn.run(*args)
                    ours = _time_us(g.replay, iters)
                    ours_eager = _time_us(lambda: own_attn.run(*args), iters)
                    row = {"head_dim": d, "hq": hq, "hkv": hkv, "q_len": q_len,
                           "batch": batch, "kv": kv, "fa2_us": round(fa2, 1),
                           "own_us": round(ours, 1),
                           "own_eager_us": round(ours_eager, 1),
                           "speedup": round(fa2 / ours, 3),
                           "own_pct_bw": round(
                               100 * kv_bytes / (ours * 1e3) / bw, 1),
                           "fa2_pct_bw": round(
                               100 * kv_bytes / (fa2 * 1e3) / bw, 1)}
                    print(json.dumps(row) if as_json else
                          "BENCH " + " ".join(f"{k}={v}" for k, v in
                                              row.items()), flush=True)
                    del views, bt, g
                    torch.cuda.empty_cache()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default="512:16:2,256:16:8",
                    help="head_dim:num_q_heads:num_kv_heads,... "
                         "(default: gemma-4 full-attn, then SWA)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cases", default="",
                    help="only run cases whose name contains this substring")
    ap.add_argument("--own", action="store_true",
                    help="run the K2-NVFP4 (own kernel) cases")
    ap.add_argument("--bench", action="store_true",
                    help="microbenchmark ours vs the FA2 route, then exit")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--kvs", default="1024,4096,16384,65536,131072")
    ap.add_argument("--q-lens", default="1,9")
    ap.add_argument("--max-gb", type=float, default=6.0,
                    help="skip bench configs whose KV bytes exceed this")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--page", type=int, default=PAGE)
    args = ap.parse_args(argv)

    if not is_sm120():
        print("oracle: not an SM120 (cc 12.x) device; nothing to verify",
              file=sys.stderr)
        return 2
    ensure_deswizzle_flag()  # before flashinfer JITs anything
    try:
        import flashinfer  # noqa: F401
        import vllm._custom_ops  # noqa: F401  (registers _C_cache_ops)
    except Exception as exc:
        print(f"oracle: import failed: {exc}", file=sys.stderr)
        return 2

    shapes = [tuple(int(x) for x in spec.split(":"))
              for spec in args.shapes.split(",")]
    if args.bench:
        ints = lambda v: [int(x) for x in v.split(",")]  # noqa: E731
        bench(shapes, ints(args.batches), ints(args.kvs), ints(args.q_lens),
              args.max_gb, args.iters, args.page, args.json)
        return 0
    ok = True
    for shape in shapes:
        for name, kv_lens, q_len, kind, *kw in (OWN_CASES if args.own
                                                else CASES):
            if args.cases not in name:
                continue
            for pattern, scales in PATTERNS:
                row = {"head_dim": shape[0], "hq": shape[1], "hkv": shape[2],
                       "case": name, "pattern": pattern, "scales": scales}
                try:
                    row.update(run_case(shape, pattern, kv_lens, q_len,
                                        scales, kind, **(kw[0] if kw else {})))
                except Exception as exc:  # JIT/smem/plan failure = verdict
                    row.update({"pass": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                ok &= row["pass"]
                if args.json:
                    print(json.dumps(row), flush=True)
                else:
                    tag = "PASS" if row["pass"] else "FAIL"
                    detail = row.get("error") or " ".join(
                        f"{k}={row[k]:.5f}" for k in
                        ("cos_kernel", "rel_kernel", "cos_decode", "cos_e2e",
                         "rel_mm_vs_causal", "cos_vs_fa2") if k in row)
                    print(f"{tag} hd={shape[0]} {name:<15} {pattern:<11} "
                          f"ks/vs={scales} {detail}", flush=True)
    print(f"oracle verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
