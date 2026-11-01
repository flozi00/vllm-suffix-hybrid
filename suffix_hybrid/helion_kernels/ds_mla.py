# SPDX-License-Identifier: Apache-2.0
"""Helion port of the nvfp4_ds_mla reader (kernels-oxide/nvfp4_ds_mla):
SM120 sparse-MLA attention over vLLM's 352 B/token ``nvfp4_ds_mla`` cache —
GLM 5.3 / DeepSeek-V3.2 decode, MTP verify rows and the prefill slices
(HiSparse direct/staged, SUFFIX_SM120_HISPARSE_MIXED), which all reach the
same op: _run_mqa_kernel -> sm120/nvfp4_ds_mla_patch._suffix_nvfp4_ds_mla_decode.

Contract (the oxide op's, src/nvfp4_ds_mla_oxide.rs; dossier
dsmla-kernel-contracts-glm53.md §2/§4):
  q      [T, HQ, 576] bf16 (512 NoPE latent | 64 RoPE), HQ = heads per rank
  cache  uint8 [blocks, block_size, 352]; slot s at (s // block_size) *
         stride(0) + (s % block_size) * 352 (padded HiSparse strides fine);
         row [0,256) e2m1 pairs (low nibble = even dim), [256,320) raw e4m3
         RoPE, [320,352) 32 e4m3 SFs at byte 8*(s&3)+(s>>2)
  topk   [T, C] int32 physical slots from the DSA indexer, -1 = masked
  out    [T, HQ, 512] bf16 = softmax(sm_scale * q.k) . latent (the latent IS
         the value); an all -1 row is exactly 0

ONE launch for both oxide entries (attn_partial + attn_merge). Program =
(token t, top-k split s), grid (T, num_splits): each head tile (all 8/16
per-rank heads by default) shares one gathered latent tile, like the oxide
CTA's 8 heads; online softmax in the exp2 domain; each split stores a bf16
partial + f32 lse. The LAST split of token t to arrive (acq_rel atomic
counter, no spin-wait: any grid size, any co-running stream) merges the
splits and resets its counter to 0 (graph-replay safe). The writer (nvfp4_ds_mla_quant_store)
stays on oxide: one launch-bound 352 B row per token, and its bytes are a
bit-exact contract (IEEE 1/sf) Triton's default f32 division does not give.

Numerics (within the oracle's 2e-2 row rel-L2 reader gate): e2m1 x e4m3 SF
products have <= 6 significant bits and |x| <= 2688, so the dequantized
latent is EXACT in f16 (PV operand, P in f16 like the oxide kernel) and in
bf16 (QK operand next to the bf16 query: no f16 staging, no prescale); S,
P.V and the merge accumulate in f32, partials are bf16 (as oxide). Tiles are
bounded (top-k tile 16..64 rows, 16-row head tile): every operand tile is
16-bit, the only f32 tile is the [heads, 512] accumulator. -1 slots are zero-gathered
(stale NaN SF bytes cannot poison P.V) and masked to S = -inf.

Serving: ``SUFFIX_HELION_DS_MLA`` = oxide (default; this module is never
imported) | helion | auto, resolved once per impl by load(); decode() runs
the bucketed tuned kernel or returns None (auto -> the oxide op).
"""
from __future__ import annotations

import logging
import os
import sys

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU CI / console: the lab ABI must still import
    helion = hl = PowerOfTwoFragment = None

KNOB = "SUFFIX_HELION_DS_MLA"
KERNEL = "ds_mla_decode"
FAMILY = "nvfp4_ds_mla"  # oxide cubin family (baseline + fallback)
ROW, DIM, PE = 352, 512, 64
TOPK = 2048  # GLM 5.3 / DeepSeek-V3.2 index_topk
SCALE = 256 ** -0.5  # GLM 5.3 bmm1 scale (qk_head_dim 192 + 64)
ARCHS = ("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM")
# decode rows: batch x (MTP k + 1), k <= 5 -> 6/12/24/48/96/192, plus short
# prefill slices; T > 1024 (long prefill) stays on oxide in every mode
BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024)
COUNTERS = 16384  # rows of the persistent counter buffer (larger T: per call)
GATE = 2e-2  # oracle reader gate: max (token, head) row rel-L2


# ---------------------------------------------------------------------------
# Helion body (module level, undecorated: the lab wraps it, serving binds it
# to the shipped config via helion_kernels.compiled)
# ---------------------------------------------------------------------------
def _e2m1_f16(code):
    """e2m1 codes (int32 0..15, sign bit 3) -> exact float16 by bit
    construction (|x| = 0, .5, 1, 1.5, 2, 3, 4, 6): no f32 tile."""
    m = code & 7
    bits = torch.where(m >= 2, (m << 9) + 0x3800, m * 0x3800) | ((code & 8) << 12)
    return bits.to(torch.int16).view(torch.float16)


def ds_mla_decode(q: torch.Tensor, kv: torch.Tensor, topk: torch.Tensor, out: torch.Tensor,
                  counters: torch.Tensor, block_size: int, block_stride: int,
                  sm_scale: float) -> torch.Tensor:
    """q [T, HQ, 576] bf16, kv = the cache as a flat uint8 byte view, topk
    [T, C] int32 slots (-1 masked), out [T, HQ, 512] bf16 (filled, returned),
    counters int32 [>= T] all 0 (left 0). Tunables: num_splits, top-k tile
    (16..64), merge head tile, num_warps, num_stages, indexing, pid layout."""
    n_tok = q.size(0)
    heads, _ = hl.specialize(q.shape[1:])  # q dim 576: constant slices, no masks
    cap = hl.specialize(topk.size(1))
    block_size = hl.specialize(block_size)  # 64 in practice: shift/mask slot math
    num_splits = hl.register_tunable("num_splits", PowerOfTwoFragment(1, 128, 16))
    bk = hl.register_block_size(16, 64)
    # top-k columns per split: a power of two, so whole bk tiles once per >= bk
    per = helion.next_power_of_2(helion.cdiv(cap, num_splits))
    o_part = torch.empty([n_tok, heads, num_splits, 512], dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty([n_tok, heads, num_splits], dtype=torch.float32, device=q.device)
    for t, s in hl.grid([n_tok, num_splits]):
        c0 = s * per
        c1 = min(c0 + per, cap)
        # 16 q rows per tile (the m16 MMA; HQ 8 = 8 real + 8 masked like the
        # oxide CTA): fixed, as an autotuned dot dim overshoots to 64 rows
        for tile_h in hl.tile(heads, block_size=16):
            q_lat = q[t, tile_h, 0:512]
            q_pe = q[t, tile_h, 512:576]
            m_i = hl.full([tile_h], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_h], dtype=torch.float32)
            acc = hl.zeros([tile_h, 512], dtype=torch.float32)
            for tile_k in hl.tile(c0, c1, block_size=bk):
                slot = topk[t, tile_k]
                valid = (slot >= 0) & (tile_k.index < c1)
                keep = valid.to(torch.uint8)
                s64 = torch.clamp(slot, min=0).to(torch.int64)
                row = (s64 // block_size) * block_stride + (s64 % block_size) * 352
                blk = hl.arange(32)
                # latent bytes [tk, 32 SF blocks, 8 bytes]; -1 rows gather zeros
                lat_b = kv[row[:, None, None] + (blk * 8)[None, :, None]
                           + hl.arange(8)[None, None, :]] * keep[:, None, None]
                codes = hl.join(lat_b & 15, lat_b >> 4)  # [tk, 32, 8, 2]: dim 16b + 2j + {0,1}
                val = _e2m1_f16(codes.to(torch.int32))
                sf_b = kv[row[:, None] + (320 + 8 * (blk & 3) + (blk >> 2))[None, :]] * keep[:, None]
                sf = sf_b.view(torch.float8_e4m3fn).to(torch.float16)
                k16 = (val * sf[:, :, None, None]).reshape(lat_b.size(0), 512)  # exact f16
                pe_b = kv[row[:, None] + (256 + hl.arange(64))[None, :]] * keep[:, None]
                k_pe = pe_b.view(torch.float8_e4m3fn).to(torch.float16).to(torch.bfloat16)
                sc = hl.dot(q_lat, k16.to(torch.bfloat16).T, out_dtype=torch.float32)
                sc = hl.dot(q_pe, k_pe.T, acc=sc)
                sc = torch.where(valid[None, :], sc * (sm_scale * 1.4426950408889634),
                                 float("-inf"))
                m_new = torch.maximum(m_i, torch.amax(sc, dim=1))
                m_ref = torch.clamp(m_new, min=-3.0e38)  # all-masked so far: exp2(-inf) = 0
                alpha = torch.exp2(m_i - m_ref)
                p = torch.exp2(sc - m_ref[:, None])
                l_i = l_i * alpha + torch.sum(p, dim=1)
                acc = hl.dot(p.to(torch.float16), k16, acc=acc * alpha[:, None])
                m_i = m_new
            l_div = l_i + (l_i == 0.0).to(torch.float32)  # empty split: o 0, lse -inf
            o_part[t, tile_h, s, :] = (acc / l_div[:, None]).to(torch.bfloat16)
            lse_part[t, tile_h, s] = m_i + torch.log2(l_div)
        old = hl.atomic_add(counters, [t], 1, sem="acq_rel")
        if old == num_splits - 1:
            # every split of token t has stored its partial: merge them
            for tile_g in hl.tile(heads):
                mg_m = hl.full([tile_g], float("-inf"), dtype=torch.float32)
                mg_l = hl.zeros([tile_g], dtype=torch.float32)
                mg_o = hl.zeros([tile_g, 512], dtype=torch.float32)
                for s2 in hl.grid(num_splits):
                    lse = lse_part[t, tile_g, s2]
                    mg_new = torch.maximum(mg_m, lse)
                    mg_ref = torch.clamp(mg_new, min=-3.0e38)
                    mg_a = torch.exp2(mg_m - mg_ref)
                    mg_w = torch.exp2(lse - mg_ref)
                    mg_l = mg_l * mg_a + mg_w
                    mg_o = (mg_o * mg_a[:, None]
                            + o_part[t, tile_g, s2, :].to(torch.float32) * mg_w[:, None])
                    mg_m = mg_new
                mg_div = mg_l + (mg_l == 0.0).to(torch.float32)  # all -1 token: exact 0
                out[t, tile_g, :] = (mg_o / mg_div[:, None]).to(out.dtype)
            counters[t] = 0
    return out


# ---------------------------------------------------------------------------
# serving entry
# ---------------------------------------------------------------------------
def case(numtokens: int, heads: int, topk: int) -> dict:
    return {"numtokens": numtokens, "heads": heads, "topk": topk}


def views(q, kv_cache, topk, out):
    """The oxide op's accepted views -> (q [T,H,576], flat uint8 bytes, topk
    [T,C], out [T,H,512], block_size, block_stride). RuntimeError otherwise."""
    q3 = q.squeeze(1) if q.dim() == 4 else q
    if q3.dim() != 3 or q3.shape[-1] != DIM + PE or q3.dtype != torch.bfloat16:
        raise RuntimeError(f"{hk.MARKER} ds_mla: q must be bf16 [T, HQ, 576], got "
                           f"{q.dtype} {tuple(q.shape)}")
    t, h = q3.shape[:2]
    tk = topk.reshape(t, -1) if topk.dim() == 3 else topk
    o3 = out.squeeze(1) if out.dim() == 4 else out
    if tk.dtype != torch.int32 or tk.dim() != 2 or tk.shape[0] != t or tk.shape[1] == 0:
        raise RuntimeError(f"{hk.MARKER} ds_mla: topk must be int32 [T, C] (T={t}), got "
                           f"{topk.dtype} {tuple(topk.shape)}")
    if o3.dtype != torch.bfloat16 or tuple(o3.shape) != (t, h, DIM):
        raise RuntimeError(f"{hk.MARKER} ds_mla: out must be bf16 [{t}, {h}, 512], got "
                           f"{out.dtype} {tuple(out.shape)}")
    kv = kv_cache.view(torch.uint8)
    if kv.dim() != 3 or kv.shape[2] != ROW or kv.shape[0] == 0 or kv.stride(2) != 1 \
            or (kv.shape[1] > 1 and kv.stride(1) != ROW):
        raise RuntimeError(f"{hk.MARKER} ds_mla: kv_cache must be uint8 [blocks, block_size, "
                           f"352] with dense rows, got {tuple(kv.shape)} {kv.stride()}")
    nb, bs = kv.shape[:2]
    stride = kv.stride(0) if nb > 1 else bs * ROW
    if stride < bs * ROW or stride % 16 or kv.data_ptr() % 16:
        raise RuntimeError(f"{hk.MARKER} ds_mla: kv_cache needs 16 B aligned, non-overlapping "
                           f"blocks (stride {stride})")
    flat = kv.as_strided(((nb - 1) * stride + bs * ROW,), (1,))
    return q3, flat, tk, o3, bs, stride


def load(heads: int, topk: int, device) -> dict | None:
    """Load time (impl init, before any CUDA-graph capture): the lane's route
    for this knob — {bucket: compiled kernel} — or None (auto, nothing
    tuned: pure oxide). helion mode with any untuned bucket fails closed.
    Every bucket is compiled and checked against the oracle once per device."""
    mode = hk.backend(KNOB)
    if mode == "oxide":
        return None
    missing = [b for b in BUCKETS if hk.config(KERNEL, case(b, heads, topk)) is None]
    if mode == "helion" and missing:
        raise RuntimeError(
            f"{hk.MARKER} {KNOB}=helion: no tuned {KERNEL} config for heads {heads} topk "
            f"{topk} buckets {missing} on {hk.platform()} (tune them in the autotune lab "
            f"or use {KNOB}=auto)")
    if helion is None and len(missing) < len(BUCKETS):
        raise RuntimeError(f"{hk.MARKER} {KNOB}={mode} with tuned {KERNEL} configs, but "
                           "helion is not importable")
    kernels = {b: hk.compiled(ds_mla_decode, KERNEL, case(b, heads, topk))
               for b in BUCKETS if b not in missing}
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    key = (str(dev), heads, topk)
    # last-arriver counters: all 0 between launches (each merge resets its
    # token's); one buffer per route, sized for max-num-batched-tokens
    counters = torch.zeros(COUNTERS, dtype=torch.int32, device=dev) if kernels else None
    route = {"mode": mode, "heads": heads, "topk": topk, "kernels": kernels,
             "counters": counters}
    if key not in _VERIFIED:  # once per device (vLLM builds one impl per layer)
        if kernels:
            _verify(route, dev)
        _VERIFIED.add(key)
        hk.log(f"{KNOB}={mode}: {KERNEL} Helion buckets {sorted(kernels) or 'none'} "
               f"(verified on {dev}), oxide {missing or 'none'} (heads {heads}, topk {topk})")
    return route if kernels else None


_VERIFIED: set = set()


def decode(route: dict, q, kv_cache, topk, out, sm_scale: float):
    """The serving op (oxide op semantics minus its o_part/lse_part workspace
    and stream: Helion allocates its own and runs on torch's current stream;
    one serving stream per worker shares the route's counter buffer).
    Returns `out`, or None when this shape has no tuned kernel (auto) or
    T > the largest bucket (any mode): the caller runs the oxide op."""
    t, h, c = q.shape[0], q.shape[-2], topk.shape[-1]
    if t == 0:
        return out
    if t > BUCKETS[-1]:
        return None  # long prefill: oxide (owner decision 2026-10-17)
    kern = None
    if (h, c) == (route["heads"], route["topk"]):
        kern = route["kernels"].get(hk.bucket(t, BUCKETS))
    if kern is None:
        if route["mode"] == "helion":
            raise RuntimeError(f"{hk.MARKER} {KNOB}=helion: no {KERNEL} kernel for T={t} "
                               f"heads {h} topk {c} (loaded heads {route['heads']} topk "
                               f"{route['topk']})")
        return None
    q3, flat, tk, o3, bs, stride = views(q, kv_cache, topk, out)
    cnt = route["counters"]
    if t > cnt.numel():
        cnt = torch.zeros(t, dtype=torch.int32, device=cnt.device)
    kern(q3, flat, tk, o3, cnt, bs, stride, float(sm_scale))
    return out


def _oracle():
    """sm120/nvfp4_ds_mla_patch/oracle.py: flat at /plugins in the bundle,
    under <repo>/sm120 in a checkout."""
    try:
        from nvfp4_ds_mla_patch import oracle
    except ImportError:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "sm120"))
        from nvfp4_ds_mla_patch import oracle
    return oracle


def _verify(route: dict, dev) -> None:
    """Load-time oracle: each bucket's kernel at T = bucket on max-SF rows,
    |q| ~ 1e5 rows, -1 holes/tails and an all -1 token vs the f64 reference
    (sampled tokens). Fails closed. Eager, CPU-generated (vLLM inits impls
    under a CUDA default device and bf16 default dtype)."""
    O = _oracle()
    heads, cap = route["heads"], route["topk"]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    ctx = torch.device(dev)
    ctx.__enter__()
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        kv_c = torch.randn(256, DIM, generator=g, device="cpu")
        kv_c[:32] = kv_c[:32].sign() * 5000.0  # SF 448, every nibble +-6
        k_pe = torch.randn(256, PE, generator=g, device="cpu")
        rows = O.quant_rows_ref(kv_c.bfloat16().to(dev), k_pe.bfloat16().to(dev))
        cache = rows.view(4, 64, ROW)
        lat, rope = (x.double() for x in O.dequant_rows_ref(rows))
        for b in sorted(route["kernels"]):
            q = torch.randn(b, heads, DIM + PE, generator=g, device="cpu")
            q[0] *= 2.0 ** 15  # |q| > f16 max
            tk = torch.randint(0, 256, (b, cap), generator=g, device="cpu", dtype=torch.int32)
            tk[1::2, cap // 3:] = -1
            tk[1::2] = tk[1::2][:, torch.randperm(cap, generator=g, device="cpu")]
            if b > 1:
                tk[1] = -1
            q, tk = q.bfloat16().to(dev), tk.to(dev)
            out = torch.empty(b, heads, DIM, dtype=torch.bfloat16, device=dev)
            decode(route, q, cache, tk, out, SCALE)
            pick = torch.tensor(sorted({0, 1 % b, b // 2, b - 1}), device=dev)
            ref = O.attention_ref(q[pick].double(), lat, rope, tk[pick], SCALE)
            keep = (tk[pick] >= 0).any(-1)[:, None].expand(-1, heads)
            err = O.row_rel_l2(out[pick], ref, keep)
            zero = bool((out[pick][~keep] == 0).all())
            finite = bool(torch.isfinite(out).all())
            if not (finite and zero and err <= GATE):
                raise RuntimeError(
                    f"{hk.MARKER} {KERNEL} load-time oracle FAILED at T={b} heads {heads} "
                    f"topk {cap} on {dev}: max row rel-L2 {err:.3e} (gate {GATE}), finite "
                    f"{finite}, all -1 token zero {zero}")
        del q, tk, out, cache
    finally:
        ctx.__exit__(None, None, None)
        torch.set_default_dtype(prev)
    if torch.device(dev).type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def cases(model: dict) -> list:
    """GLM 5.3 / DeepSeek-V3.2 sparse-MLA models: the decode-row buckets at
    the per-rank heads and the indexer's top-k; [] for every other model."""
    if model.get("architecture") not in ARCHS or not model.get("heads"):
        return []
    return [case(b, model["heads"], TOPK) for b in BUCKETS]


def _cache_view(kv, block_size, block_stride):
    nb = (kv.numel() - block_size * ROW) // block_stride + 1
    return kv.as_strided((nb, block_size, ROW), (block_stride, ROW, 1))


def make(c: dict, seed: int):
    """A realistic cache (64k slots written by the oracle's row recipe =
    the oxide/vLLM writer bytes), top-k rows with -1 tails / holes / one all
    -1 padding row, q, out and zeroed counters — the body's args (dense
    64-row blocks)."""
    O = _oracle()
    t, h, cap = c["numtokens"], c["heads"], c["topk"]
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    n, bs = 1 << 16, 64
    chan = torch.exp(torch.randn(DIM, generator=g) * 0.75)
    kv_c = (torch.randn(n, DIM, generator=g) * chan).bfloat16().cuda()
    k_pe = (torch.randn(n, PE, generator=g) * 2.0).bfloat16().cuda()
    flat = O.quant_rows_ref(kv_c, k_pe).view(-1)
    topk = torch.randint(0, n, (t, cap), generator=g, dtype=torch.int32)
    for r in range(3, t, 4):  # short contexts: -1 tail
        topk[r, int(torch.randint(1, cap, (1,), generator=g)):] = -1
    for r in range(7, t, 8):  # holes anywhere
        topk[r] = topk[r, torch.randperm(cap, generator=g)]
    if t > 5:
        topk[5] = -1  # padding row
    q = torch.randn(t, h, DIM + PE, generator=g).bfloat16().cuda()
    out = torch.empty(t, h, DIM, dtype=torch.bfloat16, device="cuda")
    counters = torch.zeros(t, dtype=torch.int32, device="cuda")
    return q, flat, topk.cuda(), out, counters, bs, bs * ROW, SCALE


def reference(q, kv, topk, out, counters, block_size, block_stride, sm_scale):
    O = _oracle()
    lat, rope = O.dequant_rows_ref(_cache_view(kv, block_size, block_stride).reshape(-1, ROW))
    return O.attention_ref(q.float(), lat, rope, topk, sm_scale)


def baseline(q, kv, topk, out, counters, block_size, block_stride, sm_scale):
    """The oxide op exactly as the patch launches it (plan-sized workspace)."""
    from suffix_hybrid import _native, oxide_kernels

    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    oxide_kernels.ensure_loaded(FAMILY, dev)
    t, h = q.shape[:2]
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ns = _native.nvfp4_ds_mla_plan(t, h, topk.shape[-1], sms)["ns"]
    o_part = torch.empty(t, h, ns, DIM, dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty(t, h, ns, dtype=torch.float32, device=q.device)
    _native.nvfp4_ds_mla_decode_cuda(
        q, _cache_view(kv, block_size, block_stride), topk, out, o_part, lse_part,
        float(sm_scale), torch.cuda.current_stream(q.device).cuda_stream)
    return out


def _tune_check(actual, expected):
    """Autotuner accuracy check against the oxide op's output (the tuning
    baseline): both sit within the 2e-2 reader gate of the reference, so
    rows may differ by up to twice that; masked rows are exactly 0 in both.
    Raises AssertionError (Helion's contract)."""
    keep = expected.float().norm(dim=-1) > 0
    d = (actual.float() - expected.float()).norm(dim=-1)
    rel = d / expected.float().norm(dim=-1).clamp_min(1e-30)
    if not bool(torch.isfinite(actual.float()).all()) or bool((actual[~keep] != 0).any()) \
            or (bool(keep.any()) and float(rel[keep].max()) > 2 * GATE):
        raise AssertionError(f"ds_mla vs oxide: max row rel-L2 "
                             f"{float(rel[keep].max()) if bool(keep.any()) else 0.0:.2e}")


def check(out, ref):
    """The oracle's reader gate: max (token, head) row rel-L2 <= 2e-2, all
    rows finite, masked (all -1) tokens exactly 0."""
    keep = ref.float().norm(dim=-1) > 0
    finite = bool(torch.isfinite(out.float()).all())
    zero = bool((out[~keep] == 0).all())
    d = (out.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-30)
    err = float(d[keep].max()) if bool(keep.any()) else 0.0
    return (finite and zero and err <= GATE,
            f"max row rel-L2 {err:.2e} (gate {GATE}), finite {finite}, masked rows zero {zero}")


KERNELS = [
    {"name": KERNEL, "body": ds_mla_decode, "cases": cases, "make": make,
     "reference": reference, "baseline": baseline, "baseline_name": "oxide nvfp4_ds_mla",
     "check": check,
     "replaces": [r"nvfp4_ds_mla_attn_partial", r"nvfp4_ds_mla_attn_merge"],
     # lab only: the accuracy baseline is the oxide op on the body's args (no
     # in-process, no-timeout compile of Helion's reference config, like
     # dsa_indexer); DEBUG shows every autotuner step in the run log
     "settings": {"autotune_baseline_fn": baseline,
                  "autotune_baseline_accuracy_check_fn": _tune_check,
                  "autotune_log_level": logging.DEBUG}},
]
