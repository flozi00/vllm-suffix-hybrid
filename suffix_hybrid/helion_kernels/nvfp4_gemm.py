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
  bf16 route  x bf16 [M, K], quantized in the GEMM (vLLM scaled_fp4_quant with
        IEEE math: sf = e4m3(amax16 * g/6), q = e2m1_rne(x * g/sf))
  prequant route  xq uint8 [M, K/2] + xsf swizzled like w_sf (vLLM's fused
        SiLU*mul / RMSNorm quant; the lm_head's scaled_fp4_quant)

Bodies (module-level, undecorated: the lab wraps them; serving binds the
shipped config via helion_kernels.compiled):
  nvfp4_gemm_body           bf16 route, the act quant fused per program
                            (the oxide FUSE_QUANT prologue: each N tile
                            quantizes its rows in registers)
  nvfp4_gemm_q_body         prequant route
  nvfp4_gemm_splitk_body    the same two with split-K: phase 1 writes fp32
  nvfp4_gemm_q_splitk_body  partials [split, M, N] per (M, N, K-split) tile,
                            hl.barrier(), phase 2 sums them in split order
                            0..S-1 (deterministic, like the oxide
                            nvfp4_splitk_reduce / FUSE_REDUCE fixup)
tl.dot_scaled only takes e8m0 block-32 (MX) scales, so NVFP4's e4m3 block-16
scales are applied in registers: e2m1 x e4m3 is exact in bf16, hl.dot
accumulates fp32. One k step = BKB 16-element blocks: weight bytes
[BN, BKB, 8] -> hl.join(lo, hi) nibbles [BN, BKB, 8, 2] (= k order) ->
dequant -> [BN, 16*BKB]. Tuned by Helion: block sizes over M / N / K
blocks, split_k (split bodies), loop order, pid type, L2 grouping, warps,
stages, unrolling, indexing, eviction.
Not ported: the oxide L2-prefetch hint (pf; Triton pipelining replaces it)
and the bf16 route's aq/asf readback (the Helion bf16 route keeps the codes
in registers; the lane oracle checks its output against the exact product
of the spec quant).

Serving (knob SUFFIX_HELION_NVFP4_GEMM, helion_kernels.backend): the lanes
call `prepare` at load and the oxide-signature twins `nvfp4_gemm_cuda` /
`nvfp4_gemm_q_cuda` per call. Per M bucket (BUCKETS): the single-pass
kernel's shipped config, else the split-K one's, else oxide (`auto`) / a
load-time RuntimeError (`helion`).
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
KERNEL_NAMES = {False: ("nvfp4_gemm", "nvfp4_gemm_splitk"),
                True: ("nvfp4_gemm_q", "nvfp4_gemm_q_splitk")}


# ---------------------------------------------------------------------------
# Helion bodies. Literals only inside (the bodies are traced); swizzle
# offsets with shifts / masks (indices are >= 0: Python floor-div semantics
# cost a sign fixup per element). Keep the four dequant / quant blocks identical — tests/test_helion_nvfp4_gemm.py runs each
# body against the exact reference.
# ---------------------------------------------------------------------------
def nvfp4_gemm_body(x: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor, out: torch.Tensor,
                    g: float, g6: float, alpha: float) -> torch.Tensor:
    """x bf16 [M, K/16, 16], w uint8 [N, K/16, 8], wsf e4m3 flat swizzled,
    out bf16 [M, N]; g = activation global scale, g6 = f32(g) / 6 (host:
    IEEE), alpha = 1 / (g * g_w). Writes and returns out."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    kb_pad = (nkb + 3) // 4 * 4
    bkb = hl.register_block_size(1, 32)
    for tile_m, tile_n in hl.tile([m, n]):
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]  # N tail: stay inside w_sf
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_kb in hl.tile(nkb, block_size=bkb):
            kb = torch.clamp(tile_kb.index, max=nkb - 1)[None, :]
            # activation quant (nvfp4_gemm.quantize): sf = e4m3(amax * g/6),
            # inv = g / sf (f64 divide = the IEEE f32 quotient), q = e2m1_rne
            xb = x[tile_m, tile_kb, :].to(torch.float32)
            sa = torch.clamp(torch.amax(torch.abs(xb), dim=-1) * g6, max=448.0)
            sa = sa.to(torch.float8_e4m3fn).to(torch.float32)
            inv = torch.where(sa == 0.0, 0.0, (g / sa.to(torch.float64)).to(torch.float32))
            s = xb * inv[:, :, None]
            a = torch.abs(s)
            q = torch.where(a > 5.0, 6.0, torch.where(a >= 3.5, 4.0, torch.where(
                a > 2.5, 3.0, torch.where(a >= 1.75, 2.0, torch.where(
                    a > 1.25, 1.5, torch.where(a >= 0.75, 1.0, (a > 0.25).to(torch.float32) * 0.5))))))
            av = (torch.where(s < 0.0, -q, q) * sa[:, :, None]).to(torch.bfloat16)
            # weight: nibbles in k order, e2m1 value * e4m3 block scale
            wb = w[tile_n, tile_kb, :]
            c = hl.join(wb & 15, wb >> 4)
            t = (c & 7).to(torch.float32)
            v = torch.where(t < 4.0, t * 0.5, torch.where(t < 6.0, t - 2.0, t * 2.0 - 8.0))
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            wv = (torch.where(c > 7, -v, v) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out


def nvfp4_gemm_q_body(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor,
                      out: torch.Tensor, alpha: float) -> torch.Tensor:
    """xq uint8 [M, K/16, 8] + xsf e4m3 flat swizzled (rows padded to 128:
    M-tail rows < 128 stay in bounds), w / wsf as nvfp4_gemm_body."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    kb_pad = (nkb + 3) // 4 * 4
    bkb = hl.register_block_size(1, 32)
    for tile_m, tile_n in hl.tile([m, n]):
        rm = tile_m.index[:, None]
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_kb in hl.tile(nkb, block_size=bkb):
            kb = torch.clamp(tile_kb.index, max=nkb - 1)[None, :]
            ab = xq[tile_m, tile_kb, :]
            ca = hl.join(ab & 15, ab >> 4)
            ta = (ca & 7).to(torch.float32)
            va = torch.where(ta < 4.0, ta * 0.5, torch.where(ta < 6.0, ta - 2.0, ta * 2.0 - 8.0))
            sa = xsf[(rm >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rm & 31) * 16
                     + ((rm >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            av = (torch.where(ca > 7, -va, va) * sa[:, :, None, None]).to(torch.bfloat16)
            wb = w[tile_n, tile_kb, :]
            c = hl.join(wb & 15, wb >> 4)
            t = (c & 7).to(torch.float32)
            v = torch.where(t < 4.0, t * 0.5, torch.where(t < 6.0, t - 2.0, t * 2.0 - 8.0))
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            wv = (torch.where(c > 7, -v, v) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out


def nvfp4_gemm_splitk_body(x: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor, out: torch.Tensor,
                           partial: torch.Tensor, g: float, g6: float,
                           alpha: float) -> torch.Tensor:
    """nvfp4_gemm_body with split-K: partial fp32 [cap, M, N] scratch, cap =
    the largest split_k tried (a power of two; serving passes the tuned one)."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    kb_pad = (nkb + 3) // 4 * 4
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, min(4, cap)))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))
    bkb = hl.register_block_size(1, 32)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[None, None, kb_split]):
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_kb in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            kb = torch.clamp(tile_kb.index, max=nkb - 1)[None, :]
            xb = x[tile_m, tile_kb, :].to(torch.float32)
            sa = torch.clamp(torch.amax(torch.abs(xb), dim=-1) * g6, max=448.0)
            sa = sa.to(torch.float8_e4m3fn).to(torch.float32)
            inv = torch.where(sa == 0.0, 0.0, (g / sa.to(torch.float64)).to(torch.float32))
            s = xb * inv[:, :, None]
            a = torch.abs(s)
            q = torch.where(a > 5.0, 6.0, torch.where(a >= 3.5, 4.0, torch.where(
                a > 2.5, 3.0, torch.where(a >= 1.75, 2.0, torch.where(
                    a > 1.25, 1.5, torch.where(a >= 0.75, 1.0, (a > 0.25).to(torch.float32) * 0.5))))))
            av = (torch.where(s < 0.0, -q, q) * sa[:, :, None]).to(torch.bfloat16)
            wb = w[tile_n, tile_kb, :]
            c = hl.join(wb & 15, wb >> 4)
            t = (c & 7).to(torch.float32)
            v = torch.where(t < 4.0, t * 0.5, torch.where(t < 6.0, t - 2.0, t * 2.0 - 8.0))
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            wv = (torch.where(c > 7, -v, v) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        partial[tile_s.id, tile_m, tile_n] = acc
    hl.barrier()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_r in hl.tile(nkb, block_size=kb_split):  # fixed split order
            acc = acc + partial[tile_r.id, tile_m, tile_n]
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out


def nvfp4_gemm_q_splitk_body(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor,
                             wsf: torch.Tensor, out: torch.Tensor, partial: torch.Tensor,
                             alpha: float) -> torch.Tensor:
    """nvfp4_gemm_q_body with split-K (partial as nvfp4_gemm_splitk_body)."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    kb_pad = (nkb + 3) // 4 * 4
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, min(4, cap)))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))
    bkb = hl.register_block_size(1, 32)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[None, None, kb_split]):
        rm = tile_m.index[:, None]
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_kb in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            kb = torch.clamp(tile_kb.index, max=nkb - 1)[None, :]
            ab = xq[tile_m, tile_kb, :]
            ca = hl.join(ab & 15, ab >> 4)
            ta = (ca & 7).to(torch.float32)
            va = torch.where(ta < 4.0, ta * 0.5, torch.where(ta < 6.0, ta - 2.0, ta * 2.0 - 8.0))
            sa = xsf[(rm >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rm & 31) * 16
                     + ((rm >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            av = (torch.where(ca > 7, -va, va) * sa[:, :, None, None]).to(torch.bfloat16)
            wb = w[tile_n, tile_kb, :]
            c = hl.join(wb & 15, wb >> 4)
            t = (c & 7).to(torch.float32)
            v = torch.where(t < 4.0, t * 0.5, torch.where(t < 6.0, t - 2.0, t * 2.0 - 8.0))
            sb = wsf[(rn >> 7) * (kb_pad * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)].to(torch.float32)
            wv = (torch.where(c > 7, -v, v) * sb[:, :, None, None]).to(torch.bfloat16)
            acc = hl.dot(av.reshape(av.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        partial[tile_s.id, tile_m, tile_n] = acc
    hl.barrier()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_r in hl.tile(nkb, block_size=kb_split):
            acc = acc + partial[tile_r.id, tile_m, tile_n]
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out


BODIES = {"nvfp4_gemm": nvfp4_gemm_body, "nvfp4_gemm_splitk": nvfp4_gemm_splitk_body,
          "nvfp4_gemm_q": nvfp4_gemm_q_body, "nvfp4_gemm_q_splitk": nvfp4_gemm_q_splitk_body}


def max_splits(n: int) -> int:
    """Split-K cap the lab tunes under (partial [cap, M, N]): enough (N x
    split) tiles of 32 columns for ~4 CTAs per SM (nvfp4_gemm.splits_for's
    target), >= 2, <= 64 (the oxide MAX_SPLITS); a power of two."""
    want = -(-SPLIT_COLS // n)
    return min(64, max(2, 1 << (want - 1).bit_length()))


# ---------------------------------------------------------------------------
# serving: per-M plan at load + the oxide op's call signatures
# ---------------------------------------------------------------------------
_PLANS: dict = {}  # (prequant, n, k) -> [None, entry for M = 1..max_m]


def prepare(n: int, k: int, prequant: bool, max_m: int, mode: str) -> list:
    """[None] + per M = 1..max_m: (kernel name, compiled kernel, split_k (0 =
    single pass), bucket) or None (oxide serves). mode = hk.backend(ENV):
    "auto" takes Helion where a config is shipped (= where it beat oxide);
    "helion" raises when a bucket <= max_m has none. Binds configs only:
    the lane's load oracle runs (= compiles) every bucket (`warm_ms`)."""
    if mode not in ("auto", "helion"):
        raise RuntimeError(f"{MARKER} prepare: mode {mode!r} (oxide never reaches Helion)")
    per = {}
    for b in sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)}):
        case = {"k": k, "n": n, "numtokens": b}
        per[b] = None
        for name in KERNEL_NAMES[prequant]:
            cfg = hk.config(name, case)
            if cfg is not None:
                split = int(cfg.get("split_k", 4)) if name.endswith("_splitk") else 0
                per[b] = (name, hk.compiled(BODIES[name], name, case), split, b)
                break
        if per[b] is None and mode == "helion":
            raise RuntimeError(
                f"{MARKER} {ENV}=helion: no tuned {' / '.join(KERNEL_NAMES[prequant])} config "
                f"for {hk.case_key(case)} on {hk.platform()} (tune it in the autotune lab and "
                f"ship configs/helion/<kernel>/<platform>.json, or set {ENV}=auto)")
    plan = [None] + [per[hk.bucket(m, BUCKETS)] for m in range(1, max_m + 1)]
    key = (prequant, n, k)
    if len(plan) > len(_PLANS.get(key, ())):
        _PLANS[key] = plan
    return plan


def partial_elems(n: int, plans) -> int:
    """f32 split-K scratch the plans need: split_k x bucket M x N."""
    return max([e[2] * e[3] * n for p in plans for e in p if e is not None and e[2]] or [0])


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
        name = e[0] + (f"(split<={e[2]})" if e[2] else "") if e else "oxide"
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
    current stream; its split count is its tuned config; the codes stay in
    registers, aq / asf are NOT written). partial: f32 scratch >= split_k x
    M x N (`partial_elems`)."""
    m, k = x.shape
    n = w.shape[0]
    _, kern, split, _ = _entry(False, n, k, m)
    g6 = float(np.float32(gscale) / np.float32(6.0))
    args = (x.view(m, k // 16, 16), w.view(n, k // 16, 8), _sf_flat(w_sf), out)
    if split:
        return kern(*args, partial[: split * m * n].view(split, m, n), gscale, g6, alpha)
    return kern(*args, gscale, g6, alpha)


def nvfp4_gemm_q_cuda(xq, xsf, w, w_sf, partial, out, alpha, splits, stream_ptr, fused=0,
                      counters=None):
    """Helion twin of _native.nvfp4_gemm_q_cuda (vLLM-prequantized x: packed
    [M, K/2] + 128x4-swizzled scales), same call; unused args as above."""
    m, kh = xq.shape
    n = w.shape[0]
    _, kern, split, _ = _entry(True, n, 2 * kh, m)
    args = (xq.view(m, kh // 8, 8), _sf_flat(xsf), w.view(n, kh // 8, 8), _sf_flat(w_sf), out)
    if split:
        return kern(*args, partial[: split * m * n].view(split, m, n), alpha)
    return kern(*args, alpha)


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


def _cases(model, prequant: bool, split: bool) -> list:
    if model.get("quant_format") != "nvfp4":
        return []
    from suffix_hybrid.kernels.nvfp4_gemm import splits_for
    shapes = dense_shapes(model)
    head = head_shape(model) if prequant else None
    if head and head not in shapes:
        shapes.append(head)
    # decode-sized M first; split-K only where the oxide plan splits too
    out = [{"k": k, "n": n, "numtokens": m} for m in BUCKETS for n, k in shapes
           if not split or splits_for(n, k, m) > 1]
    return out[:64]


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
    g = 2688 / amax(x), prequant route random codes + scales at g_x = 2688."""
    m, n, k = case["numtokens"], case["n"], case["k"]
    gen = torch.Generator(device=device).manual_seed(1000 + 7919 * seed + n + 31 * k)
    codes = lambda r: torch.randint(0, 256, (r, k // 2), generator=gen, dtype=torch.uint8,
                                    device=device)
    scales = lambda r: _swizzle((torch.rand(r, k // 16, generator=gen, device=device) * 432.0
                                 + 16.0).to(torch.float8_e4m3fn))
    p = dict(m=m, n=n, k=k, w=codes(n), wsf=scales(n),
             out=torch.empty(m, n, dtype=torch.bfloat16, device=device))
    if prequant:
        p.update(xq=codes(m), xsf=scales(m), alpha=float(np.float32(1.0 / FP4_RANGE ** 2)))
    else:
        x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
        g = float(np.float32(FP4_RANGE) / np.float32(float(x.abs().amax().float())))
        p.update(x=x, g=g, g6=float(np.float32(g) / np.float32(6.0)),
                 alpha=float(np.float32(1.0) / (np.float32(g) * np.float32(FP4_RANGE))))
    return p


def body_args(p: dict, split: bool) -> tuple:
    """A problem as a body's arguments (split: + the [max_splits, M, N] scratch)."""
    m, n, k = p["m"], p["n"], p["k"]
    w3 = p["w"].view(n, k // 16, 8)
    part = (torch.empty(max_splits(n), m, n, dtype=torch.float32, device=p["w"].device),) \
        if split else ()
    if "xq" in p:
        return (p["xq"].view(m, k // 16, 8), p["xsf"], w3, p["wsf"], p["out"], *part, p["alpha"])
    return (p["x"].view(m, k // 16, 16), w3, p["wsf"], p["out"], *part, p["g"], p["g6"],
            p["alpha"])


def reference_q(xq, xsf, w, wsf, out, *rest):
    """Exact (f64) alpha * deq(xq) @ deq(w).T (nvfp4_gemm.exact_ref)."""
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref
    m, n = xq.shape[0], w.shape[0]
    return exact_ref(xq.reshape(m, -1), xsf, w.reshape(n, -1), wsf, rest[-1], n)


def reference(x, w, wsf, out, *rest):
    """Exact product of the spec quant of x (nvfp4_gemm.quantize, IEEE math
    = the oxide kernel's codes bit for bit) and the weight."""
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref, quantize, swizzle_sf
    g, alpha = rest[-3], rest[-1]
    m, n = x.shape[0], w.shape[0]
    q, sf_bits, _ = quantize(x.reshape(m, -1).float().cpu().numpy(), g)
    xq = torch.from_numpy(q).to(x.device)
    xsf = torch.from_numpy(swizzle_sf(sf_bits)).to(x.device)
    return exact_ref(xq, xsf, w.reshape(n, -1), wsf, alpha, n)


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


# PTX .entry names of kernels-oxide/nvfp4_gemm (oxide manifest) the prod bf16
# route launches: nvfp4_gemm_f* at M=1 (quant + reduce fused), else
# nvfp4_quant_act + nvfp4_gemm_t* (_f* with prefetch) + nvfp4_splitk_reduce.
_OXIDE_BF16 = [r"^nvfp4_quant_act$", r"^nvfp4_gemm_[tf][1-4]$", r"^nvfp4_splitk_reduce$"]
# Only nvfp4_gemm claims them: the prequant route (lm_head, vLLM-fused quant)
# and both split-K variants launch the SAME gemm / reduce entries, a profile
# cannot tell them apart, and the dense linears' bf16 route is what the lane
# serves by default (its quant is inside the op) -> [] on the other three.
def _pf_name() -> str:
    from suffix_hybrid.kernels.nvfp4_gemm import PF_AUTO, PF_ENV, pf_env
    pin = pf_env()
    return (f"pf{pin} ({PF_ENV})" if pin is not None else
            f"pf {'|'.join(map(str, PF_AUTO))}: the faster per case, like the lane's autoroute")


def _kernel(name, prequant, split):
    return {
        "name": name, "body": BODIES[name],
        "cases": lambda model, q=prequant, s=split: _cases(model, q, s),
        "make": lambda case, seed, q=prequant, s=split: body_args(
            problem(case, seed, "cuda", q), s),
        "make_baseline": lambda case, seed, q=prequant: _baseline_args(case, seed, q),
        "reference": reference_q if prequant else reference,
        "baseline": _baseline_q if prequant else _baseline,
        "baseline_name": (f"oxide nvfp4_gemm_q prod launch, {_pf_name()}" if prequant else
                          f"oxide nvfp4_gemm prod launch, {_pf_name()}"),
        "check": check,
        "replaces": _OXIDE_BF16 if name == "nvfp4_gemm" else [],
    }


KERNELS = [_kernel("nvfp4_gemm", False, False), _kernel("nvfp4_gemm_splitk", False, True),
           _kernel("nvfp4_gemm_q", True, False), _kernel("nvfp4_gemm_q_splitk", True, True)]
