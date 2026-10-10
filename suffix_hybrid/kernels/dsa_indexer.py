# SPDX-License-Identifier: Apache-2.0
"""DSA sparse-attention indexer logits on SM120 (GLM-5.3 / DeepSeek-V3.2):
our cuda-oxide FP8 tensor-core kernel (kernels-oxide/dsa_indexer, e4m3
m16n8k32 mma, sm_120a SASS) behind the sm120 deep_gemm shim.

Gate ``SUFFIX_SM120_DSA_INDEXER=1`` (default off; needs the shim's own
``SUFFIX_SM120=1`` path on an SM120 GPU): the shim's ``fp8_fp4_paged_mqa_logits``
(decode) and ``fp8_fp4_mqa_logits`` (prefill) are served here; shapes outside
the kernel contract fall back to the Triton path (sm120_fallback) with one log
line per reason. A gate that is on but whose cubin/native op is missing fails
closed (raises at first call).

vLLM 0.30 contract (read from v0.30.0 sources):
* decode ``fp8_fp4_paged_mqa_logits(q [B, next_n, H, 128] e4m3, cache uint8
  [blocks, 64, 1, 132] split page, weights [>= B*next_n, H] f32, context_lens
  int32 [B, next_n] per-row causal, block_tables int32 [B, *], ...,
  max_model_len, clean_logits=False, indices=None)``. On SM120 vLLM runs MTP
  next_n > 2 FLATTENED: next_n = 1, one row per draft token, each carrying a
  copy of its request's block-table row. The kernel groups G consecutive rows
  per CTA and stages each tile once per distinct physical page, so a request's
  rows share one K read either way (G = next_n natively, else the MTP row
  count from the vLLM config / SUFFIX_SM120_DSA_INDEXER_GROUP).
* output [rows, max_model_len] f32; only [0, context_len) is written (vLLM's
  persistent_topk / top_k_per_row_* read nothing else; the oracle proves it on
  a NaN-poisoned buffer). clean_logits=True (never used by vLLM) pre-fills -inf.
* prefill ``fp8_fp4_mqa_logits`` q [M, H, 128], kv [N, 128] e4m3 + k_scale [N]
  f32, window [ks, ke) per row: same kernel, contiguous K, groups of 8 rows.

CLI (one GPU):
  python -m suffix_hybrid.kernels.dsa_indexer oracle   # vs Triton + f64 + top-k sets
  python -m suffix_hybrid.kernels.dsa_indexer bench    # us/call, GB/s vs HBM roofline
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_SM120_DSA_INDEXER"
GROUP_ENV = "SUFFIX_SM120_DSA_INDEXER_GROUP"
MIN_TILES_ENV = "SUFFIX_SM120_DSA_INDEXER_MIN_TILES"
MARKER = "[suffix dsa-indexer]"
FAMILY = "dsa_indexer"
PARAMS = {"dsa_logits": 24}  # = the #[kernel] parameter count (tests pin it)
TILE = 64                    # K tokens per tile == vLLM indexer block_size
MAX_WARPS = 12               # kernel launch_bounds(384)
PREFILL_GROUP = 8
HBM_BPS = 1.79e12            # RTX PRO 6000 (nvfp4_moe.HBM_BPS)
_state = {"loaded": set(), "said": set(), "cfg_group": None, "sms": {}}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def _once(key: str, msg: str) -> None:
    if key not in _state["said"]:
        _state["said"].add(key)
        _log(msg)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


# ---------------------------------------------------------------------------
# launch plan (pure; tests/test_dsa_indexer.py runs the kernel twin on it)
# ---------------------------------------------------------------------------
def slices_for(group: int) -> int:
    """Warps per row: small groups split a tile's 8 n-tiles over more warps."""
    return 4 if group == 1 else 2 if group <= 3 else 1


def plan(rows: int, group: int, n_cols: int, n_sms: int, min_tiles: int = 2) -> dict:
    """Launch plan: grid (splits, ceil(rows / group)). ~2 CTAs per SM when the
    windows are long; the kernel gives each CTA >= min_tiles tiles, so short
    contexts use fewer, longer CTAs (Q stays in registers per CTA)."""
    group = max(1, min(group, MAX_WARPS))
    slices = slices_for(group)
    groups = -(-rows // group)
    max_tiles = max(1, -(-n_cols // TILE))
    splits = max(1, min(max_tiles, -(-2 * n_sms // groups)))
    return {"group": group, "slices": slices, "splits": splits,
            "min_tiles": max(1, min_tiles), "grid": (splits, groups),
            "threads": 32 * group * slices}


def mtp_rows(vllm_config) -> int | None:
    """Rows per request of a flattened MTP decode (1 + k, parallel drafting
    2k): nvfp4_kv_patch.own_attn.max_decode_q_len's rule."""
    spec = getattr(vllm_config, "speculative_config", None)
    k = getattr(spec, "num_speculative_tokens", None) if spec else None
    if not k:
        return None
    return 1 + (2 if getattr(spec, "parallel_drafting", False) else 1) * int(k)


def decode_group(next_n: int) -> int:
    """Rows per CTA group for decode. Native [B, next_n] layout: next_n.
    Flattened (next_n == 1): SUFFIX_SM120_DSA_INDEXER_GROUP, else the vLLM
    config's MTP row count, else 1. Any value is CORRECT (the kernel dedups
    pages per tile); a matching one reads each request's K once."""
    if next_n > 1:
        return min(next_n, MAX_WARPS)
    env = os.environ.get(GROUP_ENV, "").strip()
    if env:
        return max(1, min(int(env), MAX_WARPS))
    if _state["cfg_group"] is None:
        try:
            from vllm.config import get_current_vllm_config_or_none
            _state["cfg_group"] = mtp_rows(get_current_vllm_config_or_none()) or None
        except Exception:
            pass
    return min(_state["cfg_group"] or 1, MAX_WARPS)


def _min_tiles() -> int:
    return int(os.environ.get(MIN_TILES_ENV, "2"))


def _sms(dev) -> int:
    import torch
    n = _state["sms"].get(dev.index)
    if n is None:
        n = _state["sms"][dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


# ---------------------------------------------------------------------------
# kernel launches
# ---------------------------------------------------------------------------
def _native(dev):
    import torch
    from suffix_hybrid import oxide_kernels
    nat = oxide_kernels.native()
    if dev.index not in _state["loaded"]:
        if torch.cuda.get_device_capability(dev)[0] != 12:
            raise RuntimeError("dsa_indexer cubin is sm_120a SASS (cc 12.x only)")
        oxide_kernels.ensure_loaded(FAMILY, dev.index, params=PARAMS)
        _state["loaded"].add(dev.index)
    return nat


def paged_ineligible(qv, kv_cache, weights, block_tables, q_scale, indices) -> str | None:
    import torch
    if q_scale is not None or qv.dtype != torch.float8_e4m3fn:
        return "non-FP8 (MXFP4) query"
    if indices is not None:
        return "varlen indices"
    if qv.dim() not in (3, 4) or qv.shape[-1] != 128 or qv.shape[-2] not in (32, 64):
        return f"q shape {tuple(qv.shape)} (need H 32|64, D 128)"
    kc = kv_cache.squeeze(2) if kv_cache.dim() == 4 else kv_cache
    if kc.dtype != torch.uint8 or kc.dim() != 3 or tuple(kc.shape[1:]) != (TILE, 132) \
            or not kc.is_contiguous():
        return f"kv cache {tuple(kv_cache.shape)} {kv_cache.dtype} (need [*, 64, 132] uint8)"
    if weights.dtype != torch.float32 or block_tables.dtype != torch.int32:
        return "weights/block_tables dtype"
    return None


def paged_logits(qv, kv_cache, weights, context_lens, block_tables, max_model_len,
                 clean_logits=False, out=None, group=None, min_tiles=None):
    """deep_gemm fp8_fp4_paged_mqa_logits (FP8) on our kernel; the caller
    checked paged_ineligible(). `out` (oracle/bench): a preallocated
    [rows, >= max_model_len] f32 buffer; `group` / `min_tiles`: plan overrides."""
    import torch
    dev = qv.device
    if qv.dim() == 3:
        qv = qv.unsqueeze(1)
    b, next_n, h, d = qv.shape
    rows = b * next_n
    kc = kv_cache.squeeze(2) if kv_cache.dim() == 4 else kv_cache
    if context_lens.dim() == 2:
        cl = context_lens.reshape(-1)
    else:  # 1-D per-request lens: every row of the request sees the same window
        cl = context_lens.repeat_interleave(next_n)
    cl = cl.to(torch.int32).contiguous()
    if cl.shape[0] != rows:
        raise ValueError(f"dsa_indexer: context_lens has {cl.shape[0]} rows, q has {rows}")
    if block_tables.dim() == 1:
        block_tables = block_tables.unsqueeze(-1)
    if block_tables.stride(-1) != 1:
        block_tables = block_tables.contiguous()
    if out is None:
        out = torch.empty((rows, max_model_len), dtype=torch.float32, device=dev)
    if clean_logits:
        out.fill_(float("-inf"))
    if rows == 0 or max_model_len == 0:
        return out
    p = plan(rows, group or decode_group(next_n), max_model_len, _sms(dev),
             min_tiles or _min_tiles())
    _native(dev).dsa_indexer_logits_cuda(
        out, qv.reshape(rows, h, d).contiguous(), weights, kc, None, block_tables,
        None, cl, max_model_len, p["group"], p["slices"], next_n, p["splits"],
        p["min_tiles"], torch.cuda.current_stream(dev).cuda_stream)
    return out


def mqa_ineligible(qv, kv_v, kv_scale, weights, q_scale) -> str | None:
    import torch
    if q_scale is not None or qv.dtype != torch.float8_e4m3fn or kv_v.dtype != torch.float8_e4m3fn:
        return "non-FP8 (MXFP4) inputs"
    if qv.dim() != 3 or qv.shape[2] != 128 or qv.shape[1] not in (32, 64):
        return f"q shape {tuple(qv.shape)} (need H 32|64, D 128)"
    if kv_v.dim() != 2 or kv_v.shape[1] != 128 or kv_v.stride(1) != 1 or kv_v.stride(0) % 16 \
            or kv_v.data_ptr() % 16 or kv_scale.numel() != kv_v.shape[0]:
        return f"kv {tuple(kv_v.shape)} stride {tuple(kv_v.stride())} / k_scale {tuple(kv_scale.shape)}"
    if weights.dtype != torch.float32:
        return "weights dtype"
    return None


def mqa_logits(qv, kv_v, kv_scale, weights, ks, ke, clean_logits=False, out=None):
    """deep_gemm fp8_fp4_mqa_logits (FP8) on our kernel (caller checked
    mqa_ineligible())."""
    import torch
    dev = qv.device
    m, n = qv.shape[0], kv_v.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=dev)
    if clean_logits:
        out.fill_(float("-inf"))
    if m == 0 or n == 0:
        return out
    p = plan(m, PREFILL_GROUP, n, _sms(dev), _min_tiles())
    _native(dev).dsa_indexer_logits_cuda(
        out, qv.contiguous(), weights, kv_v, kv_scale.reshape(-1).to(torch.float32).contiguous(),
        None, ks.to(torch.int32).contiguous(), ke.to(torch.int32).contiguous(), n,
        p["group"], p["slices"], 1, p["splits"], p["min_tiles"],
        torch.cuda.current_stream(dev).cuda_stream)
    return out


# ---------------------------------------------------------------------------
# deep_gemm shim wiring (sm120/deep_gemm_shim/__init__.py)
# ---------------------------------------------------------------------------
def wrap_paged(fallback):
    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                                 schedule_metadata, max_model_len, clean_logits=False,
                                 indices=None):
        qv, q_scale = q
        why = paged_ineligible(qv, kv_cache, weights, block_tables, q_scale, indices)
        if why is not None:
            _once("paged:" + why, f"paged logits -> Triton fallback: {why}")
            return fallback(q, kv_cache, weights, context_lens, block_tables,
                            schedule_metadata, max_model_len, clean_logits, indices)
        nn = qv.shape[1] if qv.dim() == 4 else 1
        _once("paged", f"paged logits ACTIVE (oxide e4m3 mma, H={qv.shape[-2]}, "
                       f"rows/CTA group={decode_group(nn)})")
        return paged_logits(qv, kv_cache, weights, context_lens, block_tables,
                            max_model_len, clean_logits)
    return fp8_fp4_paged_mqa_logits


def wrap_mqa(fallback):
    def fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
        (qv, q_scale), (kv_v, kv_scale) = q, kv
        why = mqa_ineligible(qv, kv_v, kv_scale, weights, q_scale)
        if why is not None:
            _once("mqa:" + why, f"prefill logits -> Triton fallback: {why}")
            return fallback(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)
        _once("mqa", f"prefill logits ACTIVE (oxide e4m3 mma, H={qv.shape[1]})")
        return mqa_logits(qv, kv_v, kv_scale, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)
    return fp8_fp4_mqa_logits


# ---------------------------------------------------------------------------
# numerics helpers (CPU-testable)
# ---------------------------------------------------------------------------
def build_paged_cache(k_fp8, scales, pages, num_blocks):
    """uint8 [num_blocks, 64, 132] split pages: token t of a sequence lives in
    physical page pages[t // 64] at slot t % 64 (values [slot*128, +128),
    scale at 64*128 + slot*4)."""
    import torch
    cache = torch.zeros((num_blocks, TILE * 132), dtype=torch.uint8, device=k_fp8.device)
    t = torch.arange(k_fp8.shape[0], device=k_fp8.device)
    pg, slot = pages[t // TILE].long(), t % TILE
    vidx = (pg * TILE * 132 + slot * 128).unsqueeze(1) + torch.arange(128, device=k_fp8.device)
    cache.view(-1)[vidx.reshape(-1)] = k_fp8.view(torch.uint8).reshape(-1)
    sidx = (pg * TILE * 132 + TILE * 128 + slot * 4).unsqueeze(1) + torch.arange(4, device=k_fp8.device)
    cache.view(-1)[sidx.reshape(-1)] = scales.to(torch.float32).contiguous().view(torch.uint8).reshape(-1)
    return cache.view(num_blocks, TILE, 132)


def gather_paged(cache, bt_row, n):
    """(K e4m3 [n, 128], scale f32 [n]) of the first n tokens through one
    block-table row (the split-byte page layout, read back)."""
    import torch
    t = torch.arange(n, device=cache.device)
    pg = bt_row[t // TILE].long()
    flat = cache.reshape(cache.shape[0], -1)
    slot = t % TILE
    vals = flat[pg.unsqueeze(1), (slot * 128).unsqueeze(1) + torch.arange(128, device=cache.device)]
    sc = flat[pg.unsqueeze(1), (TILE * 128 + slot * 4).unsqueeze(1) + torch.arange(4, device=cache.device)]
    return vals.view(torch.float8_e4m3fn), sc.contiguous().view(torch.float32).reshape(n)


def ref_logits(q, w, k, ksc, dtype=None):
    """f64 (default) logits of q [H, 128] e4m3, w [H], k [n, 128] e4m3,
    ksc [n] -> ([n] logits, [n] |term| magnitude A = sum_h |w| sum_d |q k| |s|)."""
    import torch
    dt = dtype or torch.float64
    qf, kf = q.to(dt), k.to(dt)
    dots = kf @ qf.T                                   # [n, H]
    wt = w.to(dt)
    val = (dots.clamp_min(0) * wt).sum(1) * ksc.to(dt)
    mag = ((kf.abs() @ qf.abs().T) * wt.abs()).sum(1) * ksc.to(dt).abs()
    return val, mag


def vllm_set_ok(idx, vals, k):
    """vLLM decode top-k output row `idx` (k int32, -1 fill) is exactly the
    top-k of this row's window `vals` (ties at the k-th value aside): rows
    <= k long select 0..n-1 without reading any logit; nothing >= n or
    duplicated may appear (a read past the window of a NaN-poisoned buffer
    would show up here)."""
    import torch
    n = vals.shape[0]
    got = [i for i in idx if i >= 0]
    if len(set(got)) != len(got) or any(i >= n for i in got):
        return False
    if n <= k:
        return set(got) == set(range(n))
    top = torch.topk(vals, k)
    diff = set(got) ^ set(top.indices.tolist())
    return len(got) == k and all(float(vals[i]) == float(top.values[-1]) for i in diff)


def topk_agree(a, b, ref, k, band):
    """Top-k index SETS of logits rows a and b (1-D, one row's window) agree up
    to ties: every index in the symmetric difference has a reference value
    within `band` of the k-th largest reference value. -> (ok, n_diff)."""
    import torch
    n = a.shape[0]
    if n <= k:
        return True, 0
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    diff = sorted(sa ^ sb)
    if not diff:
        return True, 0
    kth = torch.topk(ref, k).values[-1]
    return bool(((ref[diff] - kth).abs() <= band).all()), len(diff)


# ---------------------------------------------------------------------------
# on-silicon oracle + bench
# ---------------------------------------------------------------------------
TOL = 1e-5  # |ours - f64| <= TOL * A (sum of |terms|): f32 accumulation-order class


def _fallback_mod():
    import importlib.util
    import pathlib
    try:
        import deep_gemm
        if getattr(deep_gemm, "__suffix_shim__", False):
            from deep_gemm import sm120_fallback
            return sm120_fallback
    except Exception:
        pass
    path = pathlib.Path(__file__).resolve().parents[2] / "sm120" / "deep_gemm_shim" / "sm120_fallback.py"
    spec = importlib.util.spec_from_file_location("sm120_fallback_dsa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_decode(dev, heads, batch, next_n, ctx, flattened, max_model_len, seed=0):
    """GLM-5.3-like decode problem: per request ctx_b ~ U[ctx/2, ctx] tokens in
    shuffled physical pages; rows ordered by request; row j of a request sees
    L - next_n + 1 + j (vLLM's per-row causal lens). flattened: vLLM's SM120
    layout (q [rows, 1], cl [rows, 1], block-table row copied per row)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = torch.randint(max(next_n, ctx // 2), ctx + 1, (batch,), generator=g)
    nblk = [-(-int(n) // TILE) for n in lens]
    total = sum(nblk)
    perm = torch.randperm(total + 3, generator=g)[:total]  # scattered pages, spare ones unused
    width = max(nblk)
    bt = torch.zeros((batch, width), dtype=torch.int32)
    o = 0
    for i, nb in enumerate(nblk):
        bt[i, :nb] = perm[o:o + nb].to(torch.int32)
        o += nb
    num_blocks = total + 3
    # big tensors generated on the device (200k-token caches)
    gd = torch.Generator(device=dev).manual_seed(seed)
    # K ~ a few "heavy" tokens + noise, like trained indexer keys; ue8m0 scales
    kf = torch.randn((num_blocks * TILE, 128), generator=gd, device=dev) * 0.5
    kf[torch.randint(0, num_blocks * TILE, (num_blocks * 4,), generator=gd, device=dev)] *= 6
    k_fp8 = kf.clamp(-448, 448).to(torch.float8_e4m3fn)
    del kf
    ksc = 2.0 ** torch.randint(-8, -3, (num_blocks * TILE,), generator=gd, device=dev).float()
    rows = batch * next_n
    q = (torch.randn((rows, heads, 128), generator=gd, device=dev) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
    w = torch.randn((rows, heads), generator=gd, device=dev) * (128 ** -0.5) * (heads ** -0.5)
    cl = (lens.unsqueeze(1) - next_n + 1 + torch.arange(next_n)).clamp_min(0).to(torch.int32)
    # every physical page filled (page p slot s = row p*64 + s of kf)
    cache = build_paged_cache(k_fp8, ksc, torch.arange(num_blocks, dtype=torch.int32, device=dev),
                              num_blocks)
    if flattened:
        qd = q.view(rows, 1, heads, 128)
        cl_d = cl.reshape(rows, 1)
        bt_d = bt.repeat_interleave(next_n, 0)
    else:
        qd = q.view(batch, next_n, heads, 128)
        cl_d, bt_d = cl, bt
    to = lambda t: t.to(dev)  # noqa: E731
    return {"q": to(qd), "cache": to(cache).unsqueeze(2), "w": to(w), "cl": to(cl_d),
            "bt": to(bt_d), "lens": lens, "bt_req": bt, "next_n": next_n, "rows": rows,
            "max_model_len": max_model_len, "heads": heads}


ORACLE_CASES = (  # (heads, batch, next_n, ctx, flattened)
    (32, 1, 1, 1000, False), (32, 1, 6, 1000, True), (32, 4, 6, 16384, True),
    (32, 2, 2, 70000, False), (32, 1, 6, 200000, True), (32, 32, 1, 4096, False),
    (32, 32, 6, 2048, True), (64, 4, 6, 16384, True), (64, 3, 6, 8192, False),
    (32, 3, 5, 3000, True),  # rows/request 5 while the group is 6: straddling groups
)
MAX_MODEL_LEN = 202752


def _check_rows(pr, ours, trit, rows_idx, cols_per_row=4096, seed=0):
    """Per sampled row: f64 reference on <= cols_per_row sampled columns ->
    max |ours - ref| / A, Triton's, and top-2048 set agreement ours vs Triton
    on the full window. -> (worst_ours, worst_triton, topk_ok, n_diff)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    worst = worst_t = 0.0
    ok_all, ndiff = True, 0
    nn = pr["next_n"]
    for r in rows_idx:
        b = r // nn
        n = int(pr["cl"].reshape(-1)[r])
        if n == 0:
            continue
        k, s = gather_paged(pr["cache"].squeeze(2), pr["bt_req"][b].to(pr["cache"].device), n)
        q = pr["q"].reshape(pr["rows"], pr["heads"], 128)[r]
        w = pr["w"][r]
        cols = torch.randperm(n, generator=g)[:cols_per_row].to(k.device)
        ref, mag = ref_logits(q, w, k[cols], s[cols])
        o, t = ours[r, cols].double(), trit[r, cols].double()
        den = mag + 1e-30
        worst = max(worst, float(((o - ref).abs() / den).max()))
        worst_t = max(worst_t, float(((t - ref).abs() / den).max()))
        full_ref, full_mag = ref_logits(q, w, k, s, torch.float32)
        ok, nd = topk_agree(ours[r, :n], trit[r, :n], full_ref, 2048, 4 * TOL * float(full_mag.max()))
        ok_all &= ok
        ndiff += nd
    return worst, worst_t, ok_all, ndiff


def _vllm_topk(logits, lens2d):
    """vLLM's SM120 decode top-k (persistent_topk, 2048) -> int32 [rows, 2048]."""
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    out = torch.empty((logits.shape[0], 2048), dtype=torch.int32, device=logits.device)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(logits, lens2d, out, ws, 2048, logits.shape[1])
    return out


def oracle(cases=ORACLE_CASES):
    """Per case: ours vs the Triton fallback vs an f64 reference (sampled
    columns, normalized by the sum of |terms|), top-2048 sets vs Triton,
    determinism, no write outside [0, cl) (NaN-poisoned output), vLLM's
    persistent_topk on the poisoned output == on Triton's; then prefill
    (fp8_fp4_mqa_logits) cases. Fatal on mismatch."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    fb = _fallback_mod()
    lines = []
    for heads, batch, nn, ctx, flat in cases:
        pr = make_decode(dev, heads, batch, nn, ctx, flat, MAX_MODEL_LEN, seed=ctx + batch)
        rows = pr["rows"]
        args = ((pr["q"], None), pr["cache"], pr["w"], pr["cl"], pr["bt"], None, MAX_MODEL_LEN)
        trit = fb.fp8_fp4_paged_mqa_logits(*args, clean_logits=False)
        poison = torch.full((rows, MAX_MODEL_LEN), float("nan"), device=dev)
        # flattened: the MTP row count vLLM's config gives; next_n 5 runs with
        # a deliberately mismatched group of 6 (groups straddle requests)
        grp = (6 if nn == 5 else nn) if flat else None
        ours = paged_logits(pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], MAX_MODEL_LEN,
                            out=poison, group=grp)
        again = paged_logits(pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], MAX_MODEL_LEN,
                             group=grp)
        torch.cuda.synchronize()
        cl = pr["cl"].reshape(-1)
        col = torch.arange(MAX_MODEL_LEN, device=dev)
        inside = col.unsqueeze(0) < cl.unsqueeze(1)
        untouched = bool(torch.isnan(ours[~inside]).all())
        finite = bool(torch.isfinite(ours[inside]).all())
        det = bool(torch.equal(ours[inside], again[inside]))
        sample = sorted({0, rows - 1, rows // 2, *range(0, rows, max(1, rows // 6))})
        worst, worst_t, tk_ok, nd = _check_rows(pr, ours, trit, sample)
        lens2d = pr["cl"].reshape(rows, -1) if pr["cl"].dim() == 2 else pr["cl"]
        vt = _vllm_topk(ours, lens2d).cpu()
        v_ok = all(vllm_set_ok(vt[r].tolist(), ours[r, :int(cl[r])].cpu(), 2048)
                   for r in range(rows))
        ok = (worst <= TOL and untouched and finite and det and tk_ok and v_ok)
        lines.append(f"decode H={heads} B={batch} next_n={nn} {'flattened' if flat else 'native'} "
                     f"ctx<={ctx} rows={rows}: err/A ours {worst:.2e} triton {worst_t:.2e} "
                     f"(tol {TOL:.0e}) top2048 sets {'==' if tk_ok else '!='} triton "
                     f"({nd} tie swaps) vllm persistent_topk {'OK' if v_ok else 'MISMATCH'} "
                     f"no-write-outside-cl={untouched} finite={finite} deterministic={det} "
                     f"{'OK' if ok else 'FAIL'}")
        _log(lines[-1])
        if not ok:
            raise RuntimeError(f"{MARKER} DSA-INDEXER ORACLE FAIL: {lines[-1]}")
        del pr, trit, poison, ours, again
        torch.cuda.empty_cache()
    for heads, m, n in ((32, 300, 5000), (64, 128, 20000), (32, 1, 70)):
        g = torch.Generator(device="cpu").manual_seed(m + n)
        q = (torch.randn((m, heads, 128), generator=g) * 2).to(torch.float8_e4m3fn).to(dev)
        kv = (torch.randn((n, 128), generator=g) * 0.5).to(torch.float8_e4m3fn).to(dev)
        ks_ = (2.0 ** torch.randint(-8, -3, (n,), generator=g).float()).to(dev)
        w = (torch.randn((m, heads), generator=g) * 0.02).to(dev)
        # two requests packed in K (vLLM prefill chunk): request 0 = K [0, n/2)
        # with the first m/2 rows, request 1 = K [n/2, n) with the rest; row i
        # of a request with L keys and Mq rows sees [a, a + L - Mq + 1 + i)
        half = m // 2
        ks_l, ke_l = [], []
        for a, ln, mq in ((0, n // 2, half), (n // 2, n - n // 2, m - half)):
            ks_l += [a] * mq
            ke_l += [a + ln - mq + 1 + i for i in range(mq)]
        ks = torch.tensor(ks_l, dtype=torch.int32, device=dev)
        ke = torch.tensor(ke_l, dtype=torch.int32, device=dev)
        trit = fb.fp8_fp4_mqa_logits((q, None), (kv, ks_), w, ks, ke, clean_logits=True)
        ours = mqa_logits(q, kv, ks_, w, ks, ke, clean_logits=True)
        torch.cuda.synchronize()
        worst = 0.0
        for row in sorted({0, m - 1, half, m // 3}):
            a, bnd = int(ks[row]), int(ke[row])
            ref, mag = ref_logits(q[row], w[row], kv[a:bnd], ks_[a:bnd])
            worst = max(worst, float(((ours[row, a:bnd].double() - ref).abs() / (mag + 1e-30)).max()))
        mask_ok = bool(torch.equal(torch.isneginf(ours), torch.isneginf(trit)))
        ok = worst <= TOL and mask_ok
        lines.append(f"prefill H={heads} M={m} N={n}: err/A {worst:.2e} -inf mask == triton "
                     f"{mask_ok} {'OK' if ok else 'FAIL'}")
        _log(lines[-1])
        if not ok:
            raise RuntimeError(f"{MARKER} DSA-INDEXER ORACLE FAIL: {lines[-1]}")
    return f"{MARKER} DSA-INDEXER ORACLE PASS ({len(lines)} cases, sm_120a e4m3 m16n8k32 mma)"


BENCH_CASES = tuple((h, b, nn, ctx, nn > 2) for h in (32,) for nn in (1, 6)
                    for b in (1, 4, 16, 32) for ctx in (1024, 8192, 32768, 70000, 131072, 200000)
                    if b * ctx <= 32 * 70000) + ((64, 4, 6, 70000, True),)


def bench(cases=BENCH_CASES, iters=50):
    """us per decode logits call (CUDA graphs, cold L2 per replay): ours vs
    the Triton fallback; effective GB/s over the bytes that must move (each
    request's K pages once + Q/weights + written logits) vs the HBM roofline."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    dev = torch.device("cuda", torch.cuda.current_device())
    fb = _fallback_mod()
    scrub = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    flush = lambda: scrub.fill_(1)  # noqa: E731  (> the 128 MB L2)
    res = {}
    for heads, batch, nn, ctx, flat in cases:
        pr = make_decode(dev, heads, batch, nn, ctx, flat, MAX_MODEL_LEN, seed=1)
        args = ((pr["q"], None), pr["cache"], pr["w"], pr["cl"], pr["bt"], None, MAX_MODEL_LEN)
        out = torch.empty((pr["rows"], MAX_MODEL_LEN), dtype=torch.float32, device=dev)
        grp = nn if flat else None
        t_o = nm._graph_us(lambda s: paged_logits(pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"],
                                                  MAX_MODEL_LEN, out=out, group=grp), dev, iters, flush)
        t_t = nm._graph_us(lambda s: fb.fp8_fp4_paged_mqa_logits(*args), dev, max(3, iters // 10), flush)
        toks = int(pr["lens"].sum())
        need = (sum(-(-int(n) // TILE) for n in pr["lens"]) * TILE * 132
                + pr["rows"] * heads * 132 + int(pr["cl"].sum()) * 4)
        gbs = need / (t_o * 1e-6) / 1e9
        roof = need / HBM_BPS * 1e6
        tflops = 2 * heads * 128 * int(pr["cl"].sum()) / (t_o * 1e-6) / 1e12
        res[(heads, batch, nn, ctx, flat)] = (t_o, t_t)
        sweep = {mt: nm._graph_us(lambda s, mt=mt: paged_logits(
            pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], MAX_MODEL_LEN, out=out, group=grp,
            min_tiles=mt), dev, iters, flush) for mt in (1, 4, 8)}
        sweep[_min_tiles()] = t_o
        _log(f"bench H={heads} B={batch} next_n={nn} {'flattened' if flat else 'native'} "
             f"ctx<={ctx} ({toks} K tokens, group {nn}): ours {t_o:.1f} us, "
             f"triton {t_t:.1f} us (x{t_t / t_o:.1f}); {gbs:.0f} GB/s = {roof / t_o * 100:.0f}% "
             f"of HBM roofline ({roof:.1f} us), {tflops:.0f} TFLOPS e4m3; min_tiles sweep "
             + ", ".join(f"{k}: {v:.1f}" for k, v in sorted(sweep.items())) + " us")
        torch.cuda.empty_cache()
    return res


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        import torch
        _native(torch.device("cuda", torch.cuda.current_device()))
        if mode in ("oracle", "both"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:
        print(f"{MARKER} DSA-INDEXER {mode.upper()} FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
