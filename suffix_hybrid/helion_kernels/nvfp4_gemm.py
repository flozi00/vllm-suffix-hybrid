# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/nvfp4_gemm: the NVFP4 W4A4 decode GEMM of the
dense NVFP4 linears (kernels/nvfp4_linear.py) and the NVFP4 lm_head
(kernels/nvfp4_lm_head.py), M <= 64 rows.

    y[M, N] = bf16(alpha * sum_k deq(a[m, k]) * deq(w[n, k]))

Layouts (vLLM, = the oxide kernel's; kernels/nvfp4_gemm.py `quantize` /
`swizzle_sf` / `sf_offset` are the spec):
  w     uint8 [N, K/2], two e2m1 per byte, LOW nibble = even k
  w_sf  e4m3 [round128(N), round4(K/16)] 128x4-swizzled (byte of (row, kb):
        (row/128)*kb_pad*128 + (kb/4)*512 + (row%32)*16 + ((row/32)%4)*4 + kb%4)
  bf16 route  x bf16 [M, K] quantized like vLLM scaled_fp4_quant with IEEE
        math: sf = e4m3(amax16 * g/6), q = e2m1_rne(x * g/sf); g/sf comes
        from a 256-entry table per g (`inv_table`: exact, no f64 divide)
  prequant route  xq uint8 [M, K/2] + xsf swizzled like w_sf (vLLM's fused
        SiLU*mul / RMSNorm quant; the lm_head's scaled_fp4_quant)

Bodies (module-level, undecorated: the lab wraps them; serving binds the
shipped config via helion_kernels.compiled):
  nvfp4_gemm_q_body      prequant GEMM
  nvfp4_quant_act_body   bf16 x -> (xq, swizzled xsf), once per linear: the
                         bf16 route as TWO launches = this + nvfp4_gemm_q
                         (the oxide M>1 path: quant_act + gemm)
  nvfp4_gemm_body        bf16 route in ONE launch, the quant fused per
                         program (the oxide FUSE_QUANT prologue; redundant
                         per N tile, so it can only win at tiny M)
Both GEMMs split K by the tunable split_k (1 = no split, the default): each
(M, N, K-split) program writes its fp32 partial [split, M, N] and draws an
acq_rel ticket per (M, N) tile; the LAST arriver sums the splits in order
0..S-1 and re-zeroes its counter (the oxide FUSE_REDUCE fixup: deterministic,
no spin -> no residency deadlock, graph-replay safe).
tl.dot_scaled only takes e8m0 block-32 (MX) scales, so NVFP4's e4m3 block-16
scales are applied in registers: e2m1 x e4m3 is exact in bf16, hl.dot
accumulates fp32. Weights load as 2-D byte rows [BN, BK] (16 B+ vectors,
tensor-descriptor capable), the nibbles become f16 BITS (e2m1 s.ee.m placed
at f16 bit 15 / 10..9: value * 2^-14, subnormals included) -> f32 * (scale *
2^14) -> bf16, one e4m3 scale per 16-k block. Tuned by Helion: block sizes
over M / N / K bytes, split_k, loop order, pid type, L2 grouping, warps,
stages, unrolling, indexing, eviction.
Not ported: the oxide L2-prefetch hint (pf; Triton pipelining replaces it).

Serving (knob SUFFIX_HELION_NVFP4_GEMM, helion_kernels.backend): the lanes
call `prepare` at load and the oxide-signature twins `nvfp4_gemm_cuda` /
`nvfp4_gemm_q_cuda` per call. Per M bucket (BUCKETS), shipped configs only:
prequant = nvfp4_gemm_q; bf16 = nvfp4_gemm, else nvfp4_quant_act +
nvfp4_gemm_q; else oxide (`auto`) / a load-time RuntimeError (`helion`).
"""
from __future__ import annotations

import numpy as np
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no Helion: cases, plans and references still import
    helion = hl = PowerOfTwoFragment = None

ENV = "SUFFIX_HELION_NVFP4_GEMM"
FAMILY = "nvfp4_gemm"
MARKER = "[suffix nvfp4-gemm helion]"
BUCKETS = (1, 8, 16, 32, 64)  # tuned M per case; serving M -> hk.bucket(m, BUCKETS)
MAX_M = 64  # the lanes' cap (oxide: 4 m16 tiles)
FP4_RANGE = 448.0 * 6.0
SPLIT_COLS = 4 * 188 * 32  # nvfp4_gemm.TARGET_CTAS x 32 columns: the split-K cap's scale
GEMM, GEMM_Q, QUANT = "nvfp4_gemm", "nvfp4_gemm_q", "nvfp4_quant_act"


# ---------------------------------------------------------------------------
# Helion bodies. Literals only inside (the bodies are traced); swizzle offsets
# with shifts / masks (indices >= 0: Python floor-div semantics cost a sign
# fixup per element). The dequant / quant blocks are copies on purpose
# (no helper calls in traced code); tests/test_helion_nvfp4_gemm.py runs each
# body against the exact reference in interpret mode.
# ---------------------------------------------------------------------------
def nvfp4_gemm_q_body(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor,
                      out: torch.Tensor, partial: torch.Tensor, counters: torch.Tensor,
                      alpha: float) -> torch.Tensor:
    """xq uint8 [M, K/2] + xsf e4m3 flat swizzled (rows padded to 128, M <= 64),
    w uint8 [N, K/2] + wsf e4m3 flat swizzled, out bf16 [M, N]; partial f32
    [cap, M, N] scratch (cap = the largest split_k tried, a power of two;
    serving passes the tuned one), counters int32 zeros >= `counter_elems`
    (left zero). Writes and returns out."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    nkb = kh // 8
    kb_pad = (nkb + 3) // 4 * 4
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 16)) * 16  # whole 16-k blocks
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 128)
    bk = hl.register_block_size(16, 128)  # bytes = 2..16 scale blocks per k step
    n_split = helion.cdiv(kh, k_split)  # host counts (RefTile.count is per tile)
    n_tiles = helion.cdiv(n, bn)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        rm = torch.clamp(tile_m.index, max=127)[:, None]
        mok = (tile_m.index < m)[:, None]  # padded xsf rows may hold anything
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            kb = torch.clamp(torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3,
                             max=nkb - 1)[None, :]
            ab = xq[tile_m, tile_k].to(torch.int16)
            ab = ab.reshape(ab.size(0), -1, 8)
            ha = hl.join(((ab & 7) << 9) | ((ab & 8) << 12),
                         ((ab & 112) << 5) | ((ab & 128) << 8)).view(torch.float16)
            sa = xsf[(rm >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rm & 31) * 16
                     + ((rm >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            sa = torch.where(mok, sa * 16384.0, 0.0)
            av = (ha.to(torch.float32) * sa[:, :, None, None]).to(torch.bfloat16)
            wb = w[tile_n, tile_k].to(torch.int16)
            wb = wb.reshape(wb.size(0), -1, 8)
            hw = hl.join(((wb & 7) << 9) | ((wb & 8) << 12),
                         ((wb & 112) << 5) | ((wb & 128) << 8)).view(torch.float16)
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32) * 16384.0
            wv = (hw.to(torch.float32) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
            # last arriver of this (M, N) tile sums the splits in order 0..S-1
            tid = tile_m.id * n_tiles + tile_n.id
            old = hl.atomic_add(counters, [tid], 1, sem="acq_rel")
            if old == n_split - 1:
                red = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_r in hl.tile(kh, block_size=k_split):
                    red = red + partial[tile_r.id, tile_m, tile_n]
                out[tile_m, tile_n] = (red * alpha).to(torch.bfloat16)
                counters[tid] = 0
    return out


def nvfp4_gemm_body(x: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor, out: torch.Tensor,
                    partial: torch.Tensor, counters: torch.Tensor, inv: torch.Tensor,
                    g6: float, alpha: float) -> torch.Tensor:
    """bf16 route in one launch: x bf16 [M, K/2, 2] (element pairs = bytes),
    inv f32 [256] = `inv_table(g)`, g6 = f32(g) / 6; the rest as
    nvfp4_gemm_q_body."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    nkb = kh // 8
    kb_pad = (nkb + 3) // 4 * 4
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 16)) * 16
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 128)
    bk = hl.register_block_size(16, 128)
    n_split = helion.cdiv(kh, k_split)
    n_tiles = helion.cdiv(n, bn)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            kb = torch.clamp(torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3,
                             max=nkb - 1)[None, :]
            # activation quant (nvfp4_gemm.quantize): sf = e4m3(amax * g/6),
            # s = x * (g / sf) with g / sf from the exact table, q = e2m1_rne(s)
            xb = x[tile_m, tile_k, :].to(torch.float32)
            xb = xb.reshape(xb.size(0), -1, 16)
            s8 = torch.clamp(torch.amax(torch.abs(xb), dim=-1) * g6, max=448.0)
            s8 = s8.to(torch.float8_e4m3fn)
            s = xb * inv[s8.view(torch.uint8).to(torch.int32)][:, :, None]
            a = torch.abs(s)
            q = torch.where(a > 5.0, 6.0, torch.where(a >= 3.5, 4.0, torch.where(
                a > 2.5, 3.0, torch.where(a >= 1.75, 2.0, torch.where(
                    a > 1.25, 1.5, torch.where(a >= 0.75, 1.0, (a > 0.25).to(torch.float32) * 0.5))))))
            av = (torch.where(s < 0.0, -q, q) * s8.to(torch.float32)[:, :, None])
            av = av.to(torch.bfloat16)
            wb = w[tile_n, tile_k].to(torch.int16)
            wb = wb.reshape(wb.size(0), -1, 8)
            hw = hl.join(((wb & 7) << 9) | ((wb & 8) << 12),
                         ((wb & 112) << 5) | ((wb & 128) << 8)).view(torch.float16)
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32) * 16384.0
            wv = (hw.to(torch.float32) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
            tid = tile_m.id * n_tiles + tile_n.id
            old = hl.atomic_add(counters, [tid], 1, sem="acq_rel")
            if old == n_split - 1:
                red = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_r in hl.tile(kh, block_size=k_split):
                    red = red + partial[tile_r.id, tile_m, tile_n]
                out[tile_m, tile_n] = (red * alpha).to(torch.bfloat16)
                counters[tid] = 0
    return out


def nvfp4_quant_act_body(x: torch.Tensor, xq: torch.Tensor, xsf: torch.Tensor,
                         inv: torch.Tensor, g6: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x bf16 [M, K/16, 16] -> xq uint8 [M, K/16, 8] (low nibble = even k) and
    xsf e4m3 flat 128x4-swizzled (>= 128 rows; rows >= M untouched), bit for
    bit nvfp4_gemm.quantize (the oxide nvfp4_quant_act codes)."""
    m = x.size(0)
    nkb = hl.specialize(x.size(1))
    kb_pad = (nkb + 3) // 4 * 4
    bm = hl.register_block_size(1, 64)
    bkb = hl.register_block_size(1, 64)
    for tile_m, tile_kb in hl.tile([m, nkb], block_size=[bm, bkb]):
        xb = x[tile_m, tile_kb, :].to(torch.float32)
        s8 = torch.clamp(torch.amax(torch.abs(xb), dim=-1) * g6, max=448.0)
        s8 = s8.to(torch.float8_e4m3fn)
        s = xb * inv[s8.view(torch.uint8).to(torch.int32)][:, :, None]
        a = torch.abs(s)
        code = ((a > 0.25).to(torch.int32) + (a >= 0.75).to(torch.int32)
                + (a > 1.25).to(torch.int32) + (a >= 1.75).to(torch.int32)
                + (a > 2.5).to(torch.int32) + (a >= 3.5).to(torch.int32)
                + (a > 5.0).to(torch.int32)) | ((s < 0.0).to(torch.int32) << 3)
        lo, hi = hl.split(code.reshape(code.size(0), code.size(1), 8, 2))
        xq[tile_m, tile_kb, :] = (lo | (hi << 4)).to(torch.uint8)
        rm = torch.clamp(tile_m.index, max=127)[:, None]
        kb = tile_kb.index[None, :]
        hl.store(xsf, [(rm >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rm & 31) * 16
                       + ((rm >> 5) & 3) * 4 + (kb & 3)], s8,
                 extra_mask=(tile_m.index < m)[:, None] & (kb < nkb))
    return xq, xsf


BODIES = {GEMM_Q: nvfp4_gemm_q_body, QUANT: nvfp4_quant_act_body, GEMM: nvfp4_gemm_body}


def counter_elems(n: int) -> int:
    """Split-K arrival counters a GEMM body may index: (M tiles) x (N tiles)
    at the smallest blocks (16) and M <= MAX_M."""
    return -(-MAX_M // 16) * -(-n // 16)


def max_splits(n: int) -> int:
    """Split-K cap the lab tunes under (partial [cap, M, N]): enough (N x
    split) tiles of 32 columns for ~4 CTAs per SM (nvfp4_gemm.splits_for's
    target), >= 2, <= 64 (the oxide MAX_SPLITS); a power of two."""
    want = -(-SPLIT_COLS // n)
    return min(64, max(2, 1 << (want - 1).bit_length()))


def inv_table(g: float) -> np.ndarray:
    """f32 [256]: IEEE f32(g) / e4m3(i) for every positive finite e4m3 byte i
    (0 elsewhere = the spec's `sf == 0 -> 0`): the quant's only division."""
    sf = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    sf = sf.float().numpy()
    out = np.zeros(256, np.float32)
    ok = (np.arange(256) < 128) & (sf > 0) & np.isfinite(sf)
    out[ok] = np.float32(g) / sf[ok]
    return out


# ---------------------------------------------------------------------------
# serving: per-M plan at load + the oxide op's call signatures
# ---------------------------------------------------------------------------
_PLANS: dict = {}  # (prequant, n, k) -> [None, entry for M = 1..max_m]
_NEED = {"counters": 0, "xq": 0, "xsf": 0}  # scratch every prepared plan needs
_SCRATCH: dict = {}  # (device, name) -> tensor (zeros; grown at load, before capture)
_INV: dict = {}  # (device, g) -> inv_table(g) on device


def _scratch(device, name: str, dtype) -> torch.Tensor:
    """Module scratch sized for every prepared plan: split-K counters (left
    zero by every launch), the two-launch bf16 route's xq / xsf. First used
    by the lanes' load-time oracle / warm-up, i.e. before graph capture."""
    buf = _SCRATCH.get((device, name))
    if buf is None or buf.numel() < _NEED[name]:
        buf = _SCRATCH[(device, name)] = torch.zeros(max(1, _NEED[name]), dtype=dtype,
                                                     device=device)
    return buf


def _inv(g: float, device) -> torch.Tensor:
    key = (device, float(g))
    if key not in _INV:
        _INV[key] = torch.from_numpy(inv_table(g)).to(device)
    return _INV[key]


def _gemm_entry(name: str, n: int, k: int, b: int):
    """(compiled, split_k) of GEMM `name` for (N, K, bucket b), or None."""
    case = {"k": k, "n": n, "numtokens": b}
    cfg = hk.config(name, case)
    if cfg is None:
        return None
    return hk.compiled(BODIES[name], name, case), int(cfg.get("split_k", 1))


def prepare(n: int, k: int, prequant: bool, max_m: int, mode: str) -> list:
    """[None] + per M = 1..max_m: (label, kernels, split_k, bucket) or None
    (oxide serves). kernels = (gemm,) or (quant, gemm_q). mode =
    hk.backend(ENV): "auto" takes Helion where configs are shipped (= where
    it beat oxide in the lab); "helion" raises when a bucket <= max_m has
    none. Binds configs only: the lane's load oracle runs (= compiles) every
    bucket (`warm_ms`)."""
    if mode not in ("auto", "helion"):
        raise RuntimeError(f"{MARKER} prepare: mode {mode!r} (oxide never reaches Helion)")
    per = {}
    for b in sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)}):
        e = None
        if prequant:
            g = _gemm_entry(GEMM_Q, n, k, b)
            e = g and (GEMM_Q, (g[0],), g[1], b)
        else:
            g = _gemm_entry(GEMM, n, k, b)
            if g:
                e = (GEMM, (g[0],), g[1], b)
            else:
                q = _gemm_entry(GEMM_Q, n, k, b)
                qc = {"k": k, "numtokens": b}
                if q and hk.config(QUANT, qc) is not None:
                    e = (f"{QUANT}+{GEMM_Q}", (hk.compiled(BODIES[QUANT], QUANT, qc), q[0]),
                         q[1], b)
        if e is None and mode == "helion":
            names = GEMM_Q if prequant else f"{GEMM} or {QUANT} + {GEMM_Q}"
            raise RuntimeError(
                f"{MARKER} {ENV}=helion: no tuned {names} config for N={n} K={k} M bucket "
                f"{b} on {hk.platform()} (tune it in the autotune lab and ship "
                f"configs/helion/<kernel>/<platform>.json, or set {ENV}=auto)")
        per[b] = e
    plan = [None] + [per[hk.bucket(m, BUCKETS)] for m in range(1, max_m + 1)]
    if any(plan[1:]):
        _NEED["counters"] = max(_NEED["counters"], counter_elems(n))
    if any(e and len(e[1]) == 2 for e in plan[1:]):
        _NEED["xq"] = max(_NEED["xq"], MAX_M * k // 2)
        _NEED["xsf"] = max(_NEED["xsf"], 128 * (-(-k // 64) * 4))
    key = (prequant, n, k)
    if len(plan) > len(_PLANS.get(key, ())):
        _PLANS[key] = plan
    return plan


def partial_elems(n: int, plans) -> int:
    """f32 split-K scratch the plans need: split_k x bucket M x N."""
    return max([e[2] * e[3] * n for p in plans for e in p if e is not None] or [0])


def warm_ms(max_m: int) -> list:
    """Ms that compile every Triton specialization the plans can launch:
    per bucket its top row count and the one below (M == 1 and M % 16 == 0
    are separate Triton specializations), clipped to max_m."""
    out, prev = set(), 0
    for b in BUCKETS:
        if prev >= max_m:
            break
        top = min(b, max_m)
        out |= {top, max(prev + 1, top - 1)}
        prev = b
    return sorted(out)


def describe(plan: list) -> str:
    """'M1:nvfp4_gemm M2-8:oxide ...' for the load log."""
    runs, last = [], object()
    for m, e in enumerate(plan[1:], 1):
        name = e[0] + (f"(split<={e[2]})" if e[2] > 1 else "") if e else "oxide"
        if name == last:
            runs[-1][1] = m
        else:
            runs.append([m, m, name])
            last = name
    return " ".join(f"M{a}{'' if a == b else f'-{b}'}:{s}" for a, b, s in runs)


def _entry(prequant: bool, n: int, k: int, m: int):
    plan = _PLANS.get((prequant, n, k))
    e = plan[m] if plan is not None and 1 <= m < len(plan) else None
    if e is None:
        raise RuntimeError(f"{MARKER} no Helion plan for N={n} K={k} M={m} "
                           f"({'prequant' if prequant else 'bf16'} route): the lane routes "
                           "only planned M here")
    return e


def _sf_flat(sf: torch.Tensor) -> torch.Tensor:
    return sf.reshape(-1).view(torch.float8_e4m3fn)


def nvfp4_gemm_cuda(x, w, w_sf, aq, asf, partial, out, gscale, alpha, splits, stream_ptr,
                    fused=0, counters=None):
    """Helion twin of _native.nvfp4_gemm_cuda (bf16 route), same call: out =
    bf16(alpha * nvfp4(x) @ w^T). aq / asf / splits / stream_ptr / fused /
    counters are oxide launch state and unused (Helion launches on torch's
    current stream, its split count is its tuned config, its codes go to
    module scratch: aq / asf are NOT written). partial: f32 scratch >=
    split_k x M x N (`partial_elems`)."""
    m, k = x.shape
    n = w.shape[0]
    _, kerns, split, _ = _entry(False, n, k, m)
    dev = x.device
    inv, g6 = _inv(gscale, dev), float(np.float32(gscale) / np.float32(6.0))
    tail = (partial[: split * m * n].view(split, m, n),
            _scratch(dev, "counters", torch.int32))
    if len(kerns) == 1:
        return kerns[0](x.view(m, k // 2, 2), w, _sf_flat(w_sf), out, *tail, inv, g6, alpha)
    xq = _scratch(dev, "xq", torch.uint8)[: m * k // 2]
    xsf = _scratch(dev, "xsf", torch.uint8).view(torch.float8_e4m3fn)
    kerns[0](x.view(m, k // 16, 16), xq.view(m, k // 16, 8), xsf, inv, g6)
    return kerns[1](xq.view(m, k // 2), xsf, w, _sf_flat(w_sf), out, *tail, alpha)


def nvfp4_gemm_q_cuda(xq, xsf, w, w_sf, partial, out, alpha, splits, stream_ptr, fused=0,
                      counters=None):
    """Helion twin of _native.nvfp4_gemm_q_cuda (vLLM-prequantized x: packed
    [M, K/2] + 128x4-swizzled scales), same call; unused args as above."""
    m, kh = xq.shape
    n = w.shape[0]
    _, kerns, split, _ = _entry(True, n, 2 * kh, m)
    return kerns[0](xq, _sf_flat(xsf), w, _sf_flat(w_sf), out,
                    partial[: split * m * n].view(split, m, n),
                    _scratch(xq.device, "counters", torch.int32), alpha)


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def _arch_shapes():
    from suffix_hybrid.kernels.nvfp4_gemm import GEMMA_SHAPES, QWEN_FLASH_SHAPES, SHAPES
    # (name substring, TP the table is per rank for, {name: (N, K)})
    return (("qwen3.8-27b", 1, SHAPES), ("gemma-4-26b", 1, GEMMA_SHAPES),
            ("qwen3.8-flash", 2, QWEN_FLASH_SHAPES))


def _ok(n, k) -> bool:
    return bool(n) and bool(k) and n % 16 == 0 and k % 64 == 0


def dense_shapes(model: dict) -> list:
    """Per-TP-rank (N, K) of the model's dense NVFP4 linears: the plugin's
    per-architecture tables (kernels/nvfp4_gemm.py, from the vLLM model code)
    where they exist for this TP, else the console's generic GQA/MLP linears."""
    name, tp = str(model.get("name") or "").lower(), model.get("tp") or 1
    for key, at_tp, table in _arch_shapes():
        if key in name and tp == at_tp:
            shapes = list(table.values())
            break
    else:
        shapes = [(lin["n"], lin["k"]) for lin in model.get("linears") or []
                  if lin.get("name") != "lm_head"]
    return [s for s in dict.fromkeys(shapes) if _ok(*s)]


def head_shape(model: dict):
    """(N, K) of the NVFP4 lm_head per rank (nvfp4_lm_head.eligible), or None."""
    n, k = model.get("vocab"), model.get("hidden")
    return (n, k) if _ok(n, k) and n % 32 == 0 else None



def _cases(model, kernel: str) -> list:
    """Every (N, K) x bucket: GEMM_Q = dense + lm_head (prequant route),
    GEMM = dense, QUANT = each dense K (n not part of its case)."""
    if model.get("quant_format") != "nvfp4":
        return []
    shapes = dense_shapes(model)
    if kernel == QUANT:
        return [{"k": k, "numtokens": m} for m in BUCKETS for k in dict.fromkeys(
            k for _, k in shapes)][:64]
    head = head_shape(model) if kernel == GEMM_Q else None
    if head and head not in shapes:
        shapes.append(head)
    return [{"k": k, "n": n, "numtokens": m} for m in BUCKETS for n, k in shapes][:64]


def _swizzle(sf: torch.Tensor) -> torch.Tensor:
    """e4m3 [R, C] -> flat 128x4-swizzled e4m3 (nvfp4_gemm.swizzle_sf, zero padded)."""
    r, c = sf.shape
    pad = torch.zeros(-(-r // 128) * 128, -(-c // 4) * 4, dtype=torch.uint8, device=sf.device)
    pad[:r, :c] = sf.view(torch.uint8)
    rp, cp = pad.shape
    return (pad.reshape(rp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)
            .view(torch.float8_e4m3fn))


def problem(case: dict, seed: int, device="cuda", prequant: bool = False) -> dict:
    """Random NVFP4 operands in vLLM layout: uniform e2m1 codes, e4m3 block
    scales in [16, 448) (what scaled_fp4_quant emits at g = 2688 / amax),
    weight global scale g_w = 2688 (amax 1); bf16 route x ~ N(0, 1) with
    g = 2688 / amax(x), prequant route random codes + scales at g_x = 2688.
    A case without n (QUANT) gets no weight."""
    m, n, k = case["numtokens"], case.get("n", 0), case["k"]
    gen = torch.Generator(device=device).manual_seed(1000 + 7919 * seed + n + 31 * k)
    codes = lambda r: torch.randint(0, 256, (r, k // 2), generator=gen, dtype=torch.uint8,
                                    device=device)
    scales = lambda r: _swizzle((torch.rand(r, k // 16, generator=gen, device=device) * 432.0
                                 + 16.0).to(torch.float8_e4m3fn))
    p = dict(m=m, n=n, k=k)
    if n:
        p.update(w=codes(n), wsf=scales(n),
                 out=torch.empty(m, n, dtype=torch.bfloat16, device=device),
                 partial=torch.empty(max_splits(n), m, n, dtype=torch.float32, device=device),
                 counters=torch.zeros(counter_elems(n), dtype=torch.int32, device=device))
    if prequant:
        p.update(xq=codes(m), xsf=scales(m), alpha=float(np.float32(1.0 / FP4_RANGE ** 2)))
    else:
        x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
        g = float(np.float32(FP4_RANGE) / np.float32(float(x.abs().amax().float())))
        p.update(x=x, g=g, g6=float(np.float32(g) / np.float32(6.0)),
                 inv=torch.from_numpy(inv_table(g)).to(device),
                 alpha=float(np.float32(1.0) / (np.float32(g) * np.float32(FP4_RANGE))))
    return p


def body_args(kernel: str, p: dict) -> tuple:
    """A problem as `kernel`'s body arguments."""
    m, k = p["m"], p["k"]
    if kernel == GEMM_Q:
        return (p["xq"], p["xsf"], p["w"], p["wsf"], p["out"], p["partial"], p["counters"],
                p["alpha"])
    if kernel == GEMM:
        return (p["x"].view(m, k // 2, 2), p["w"], p["wsf"], p["out"], p["partial"],
                p["counters"], p["inv"], p["g6"], p["alpha"])
    xsf = torch.zeros(128 * (-(-k // 64) * 4), dtype=torch.uint8, device=p["x"].device)
    return (p["x"].view(m, k // 16, 16), torch.empty(m, k // 16, 8, dtype=torch.uint8,
                                                    device=p["x"].device),
            xsf.view(torch.float8_e4m3fn), p["inv"], p["g6"])


def _g_of(inv: torch.Tensor) -> float:
    return float(inv[0x38])  # e4m3 byte 0x38 = 1.0: inv[0x38] = g / 1 = g


def reference_q(xq, xsf, w, wsf, out, partial, counters, alpha):
    """Exact (f64) alpha * deq(xq) @ deq(w).T (nvfp4_gemm.exact_ref)."""
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref
    return exact_ref(xq, xsf, w, wsf, alpha, w.shape[0])


def _spec_quant(x2, g):
    """nvfp4_gemm.quantize (IEEE; the oxide codes) -> (xq [M, K/2], swizzled xsf)."""
    from suffix_hybrid.kernels.nvfp4_gemm import quantize, swizzle_sf
    q, sf_bits, _ = quantize(x2.float().cpu().numpy(), g)
    return (torch.from_numpy(q).to(x2.device),
            torch.from_numpy(swizzle_sf(sf_bits)).to(x2.device).view(torch.float8_e4m3fn))


def reference(x, w, wsf, out, partial, counters, inv, g6, alpha):
    """Exact product of the spec quant of x and the weight."""
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref
    xq, xsf = _spec_quant(x.reshape(x.shape[0], -1), _g_of(inv))
    return exact_ref(xq, xsf, w, wsf, alpha, w.shape[0])


def reference_quant(x, xq, xsf, inv, g6):
    q, sf = _spec_quant(x.reshape(x.shape[0], -1), _g_of(inv))
    full = torch.zeros_like(xsf.view(torch.uint8))
    full[: sf.numel()] = sf.view(torch.uint8)
    return q.view(xq.shape), full.view(torch.float8_e4m3fn)


def check(out, ref):
    """Rel. Frobenius error <= 1e-2 (the oxide oracle's floor) AND every
    element within 2^-7 |ref| + 1e-3 max|ref| (bf16 output rounding is
    2^-9; fp32 accumulation order is far below 1e-3): a wrong column,
    row or tail cannot hide in the norm."""
    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite output"
    err = (o - r).abs()
    rel = float(err.norm() / r.norm().clamp_min(1e-30))
    bad = int((err > 2.0 ** -7 * r.abs() + 1e-3 * r.abs().max()).sum())
    return rel <= 1e-2 and bad == 0, (f"rel {rel:.2e} (gate 1e-2), {bad} elements past "
                                      "2^-7 |ref| + 1e-3 max|ref|")


def check_quant(out, ref):
    """Codes and scales bit-identical to the spec (= the oxide quant)."""
    (q, sf), (rq, rsf) = out, ref
    bq = int((q != rq).sum())
    bs = int((sf.view(torch.uint8) != rsf.view(torch.uint8)).sum())
    return bq == 0 and bs == 0, f"{bq} code bytes, {bs} scale bytes differ from the spec"


_NATIVE: list = []


def _oxide():
    """suffix_hybrid._native with the nvfp4_gemm cubin loaded (lab baseline)."""
    if not _NATIVE:
        from suffix_hybrid import oxide_kernels
        from suffix_hybrid.kernels.nvfp4_gemm import PARAMS
        native = oxide_kernels.native()
        oxide_kernels.ensure_loaded(FAMILY, params=PARAMS)
        _NATIVE.append(native)
    return _NATIVE[0]


_PROD_PF: dict = {}  # (prequant, case key) -> the L2-prefetch lookahead prod serves


def _prod_pf(case, prequant) -> int:
    """The prefetch the lane serves: SUFFIX_NVFP4_GEMM_PF when pinned, else
    (nvfp4_linear AUTOROUTE) the faster of nvfp4_gemm.PF_AUTO, timed the
    lane's way (CUDA graph, L2 flushed before every replay, median of
    AUTOROUTE_REPS). Per case, so oxide's bar is never below what it serves."""
    from suffix_hybrid.kernels.nvfp4_gemm import PF_AUTO, _graph_us, pf_env
    from suffix_hybrid.kernels.nvfp4_linear import AUTOROUTE_REPS
    if pf_env() is not None:
        return pf_env()
    key = (prequant, hk.case_key(case))
    if key not in _PROD_PF:
        dev = torch.device("cuda", torch.cuda.current_device())
        l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0) or 128 << 20)
        buf = torch.zeros(max(2 * l2, 256 << 20) // 4, dtype=torch.float32, device=dev)
        times = {}
        for pf in PF_AUTO:
            args = _baseline_args(case, 0, prequant, pf)
            fn = _baseline_q if prequant else _baseline
            times[pf] = _graph_us(lambda s, a=args: fn(*a), dev, AUTOROUTE_REPS, lambda: buf.sum())
        del buf
        _PROD_PF[key] = min(times, key=times.get)
    return _PROD_PF[key]


def _baseline_args(case, seed, prequant, pf=None):
    """The oxide op's arguments, PRODUCTION launch: kernels/nvfp4_gemm.plan
    with the env defaults (fused single launch at M=1, split-K per
    splits_for) and the prefetch the lane serves (`_prod_pf`)."""
    from suffix_hybrid.kernels.nvfp4_gemm import plan, tiles_for
    _oxide()
    if pf is None:
        pf = _prod_pf(case, prequant)
    p = problem(case, seed, "cuda", prequant)
    m, n, k = p["m"], p["n"], p["k"]
    pl = plan(n, k, m, prequant=prequant, pf=pf)
    dev = p["w"].device
    partial = torch.empty(max(1, pl["partial"]), dtype=torch.float32, device=dev)
    counters = torch.zeros(-(-n // 32), dtype=torch.int32, device=dev)
    if prequant:
        return (p["xq"], p["xsf"], p["w"], p["wsf"], partial, p["out"], p["alpha"],
                pl["splits"], pl["flags"], counters)
    rows = 16 * tiles_for(m)
    return (p["x"], p["w"], p["wsf"], torch.empty(rows, k // 2, dtype=torch.uint8, device=dev),
            torch.empty(rows, k // 16, dtype=torch.uint8, device=dev), partial, p["out"], p["g"],
            p["alpha"], pl["splits"], pl["flags"], counters)


def _baseline(x, w, wsf, aq, asf, partial, out, g, alpha, splits, flags, counters):
    _NATIVE[0].nvfp4_gemm_cuda(x, w, wsf, aq, asf, partial, out, g, alpha, splits,
                               torch.cuda.current_stream().cuda_stream, flags, counters)
    return out


def _baseline_q(xq, xsf, w, wsf, partial, out, alpha, splits, flags, counters):
    _NATIVE[0].nvfp4_gemm_q_cuda(xq, xsf, w, wsf, partial, out, alpha, splits,
                                 torch.cuda.current_stream().cuda_stream, flags, counters)
    return out


def _quant_baseline_args(case, seed):
    p = problem(case, seed, "cuda")
    return p["x"], torch.tensor([p["g"]], dtype=torch.float32, device=p["x"].device)


def _quant_baseline(x, g):
    """vLLM scaled_fp4_quant, swizzled scales: the oxide nvfp4_quant_act is
    only reachable inside nvfp4_gemm_cuda, this is the same contract alone."""
    from vllm._custom_ops import scaled_fp4_quant
    return scaled_fp4_quant(x, g, is_sf_swizzled_layout=True)


# PTX .entry names of kernels-oxide/nvfp4_gemm (oxide manifest) the prod bf16
# route launches: nvfp4_gemm_f* at M=1 (quant + reduce fused), else
# nvfp4_quant_act + nvfp4_gemm_t* (_f* with prefetch) + nvfp4_splitk_reduce.
# Only nvfp4_gemm claims them: the prequant route (lm_head, vLLM-fused quant)
# launches the SAME gemm / reduce entries (a profile cannot tell them apart)
# and the dense linears' bf16 route is what the lane serves by default.
_OXIDE_BF16 = [r"^nvfp4_quant_act$", r"^nvfp4_gemm_[tf][1-4]$", r"^nvfp4_splitk_reduce$"]


def _pf_name() -> str:
    from suffix_hybrid.kernels.nvfp4_gemm import PF_AUTO, PF_ENV, pf_env
    pin = pf_env()
    return (f"pf{pin} ({PF_ENV})" if pin is not None else
            f"pf {'|'.join(map(str, PF_AUTO))}: the faster per case, like the lane's autoroute")


def _kernel(name):
    q = name == GEMM_Q
    base = {
        "name": name, "body": BODIES[name],
        "cases": lambda model: _cases(model, name),
        "make": lambda case, seed: body_args(name, problem(case, seed, "cuda", q)),
    }
    if name == QUANT:
        return {**base, "make_baseline": _quant_baseline_args, "reference": reference_quant,
                "baseline": _quant_baseline, "check": check_quant, "replaces": [],
                "baseline_name": "vllm scaled_fp4_quant (swizzled; oxide's quant is fused "
                                 "into nvfp4_gemm_cuda)"}
    return {**base, "make_baseline": lambda case, seed: _baseline_args(case, seed, q),
            "reference": reference_q if q else reference,
            "baseline": _baseline_q if q else _baseline, "check": check,
            "baseline_name": f"oxide {name} prod launch, {_pf_name()}",
            "replaces": _OXIDE_BF16 if name == GEMM else []}


# Most promising first (the lab's run budget): the prequant GEMM (also the
# bf16 route's second launch), the one-pass quant, then the fused one-launch.
KERNELS = [_kernel(GEMM_Q), _kernel(QUANT), _kernel(GEMM)]
