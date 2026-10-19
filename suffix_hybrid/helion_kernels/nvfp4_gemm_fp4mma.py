# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A4 GEMM on NATIVE FP4 MMA: Helion `hl.dot_scaled` bodies (lane I of the
DeepSeek-V4.1 blind speed program; docs/dsv41-fp4.md is the long form).

Triton >= 3.6.0 lowers `tl.dot_scaled` with e2m1 x e2m1 operands and e4m3 block-16
scales on SM120 to `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.
scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` (PR triton#8494; vLLM#38338 measured it on a
5090). The Helion ports in nvfp4_gemm.py never used it (they decode in ALU). These
bodies do. NOTHING here is measured on silicon: CPU eager mode proves indexing and
numerics, not speed, registers or that the compiler really takes the native path
(K5: kernels/fp4mma_probe.py asserts the PTX at warm-up and fails closed).

    y[M, N] = bf16(alpha * sum_k deq(a[m, k]) * deq(w[n, k]))

Operands (vLLM's layout, = nvfp4_gemm.py / kernels/nvfp4_gemm.py `quantize`):
  xq  uint8 [M, K/2]   two e2m1 per byte, LOW nibble = even k   (prequant route)
  w   uint8 [N, K/2]
  *sf e4m3 block-16 scales, 128x4-swizzled (byte of (row, kb): (row/128)*cp*128 +
      (kb/4)*512 + (row%32)*16 + ((row/32)%4)*4 + kb%4, cp = round4(K/16))
Contract: K % 64 == 0 (every lane already requires it) so cp == K/16.

Scale access, the part Triton cares about. `tl.dot_scaled` takes the scales as a
[rows, K/16] e4m3 tile. The swizzled buffer stores the four scales of one (row, k64
step) as ONE aligned 32-bit word, so:
  words (primary) the buffer is passed as int32 words; one 4-byte gather per (row,
          k64 step), unpacked in registers (shift/mask + hl.join) and bit-cast to
          e4m3. Every element is a 4-byte load, so Triton's pipeliner may turn it
          into a cp.async (width >= 32 bit) whatever its contiguity analysis can
          prove about our index arithmetic; zero extra bytes in HBM.
  bytes   the buffer as e4m3 bytes, `sf[offset]` per scale. The scale operand is a
          plain loaded tile (the layout Triton's native pattern is known to take);
          loads are 1-byte unless AxisInfo proves 4-byte runs, which Helion's
          floor-div / mask index helpers hide. The safe fallback.
  lin     a LINEAR [rows, K/16] copy (+1/9 weight bytes; free for activations the
          lane quantizes itself, vLLM#38338's layout). Read as words too.
`bytes` and `words` differ only in two index lines; `words` could lose the native
lowering if Triton hands `ScaledBlockedToMMA` a LinearEncoding scale operand (it
returns failure and the op silently decomposes to a bf16 upcast) -> the P0 probe
(scripts/fp4mma_p0_probe.py) compiles both patterns for sm_120a before anything
else, and K5 re-asserts the PTX in the serving process.

Bodies (module level, undecorated, literals only; the lab wraps them):
  nvfp4_gemm_q_fp4mma       words, swizzled   (kernel name nvfp4_gemm_q_fp4mma)
  nvfp4_gemm_q_fp4mma_b     bytes, swizzled
  nvfp4_gemm_q_fp4mma_lin   words, linear scales
  nvfp4_gemm_q_fp4cvt       control arm: hardware `cvt.rn.f16x2.e2m1x2` decode of
                            both operands (hl.load_float4_e2m1fn_x16_to_float16),
                            scale in f16 (exact), f16 `hl.dot` with f32 accumulate
Split-K is TWO-STAGE only (capture-safe, lane D's contract): split programs store
alpha-scaled f32 partials to `partial[split, M, N]` (torch.empty per call = the
graph pool under capture), a fixed-order torch sum + copy_ finishes on the same
stream. No atomics, no semaphores, no counters, no module-level scratch.

The bf16-activation route is TWO launches: the existing `nvfp4_quant_act` Helion
kernel (nvfp4_gemm.py, bit-exact vs the spec quant; its config table is shared) +
the prequant GEMM here. Fusing the quant into the GEMM prologue costs
R = ceil(N/BN) redundant quantizations of the [M, K] tile, each M*K*(2 B read + ~25 ALU
ops): at N = K = 5120, BN = 64 (R = 80) that is 0.8 MB / 10 Mop at M=1 (0.2 us at 45 Top/s)
and 52 MB / 0.66 Gop at M=64 (15 us, more than the ~10 us weight-stream floor of that GEMM)
against ONE saved launch (measured for the oxide single-launch arm: -2 us/linear at M=1,
MINED-LESSONS:713, which also lost from M=5): at best ~2 us of a ~13 us call at M <= ~4,
and the MMA operand would then be a computed tensor instead of a load (one more way for
Triton to leave the native lowering). Not built. Estimates, not measurements.

Env knob SUFFIX_HELION_NVFP4_GEMM_FP4MMA = 0 (default: this family is never
imported by a lane) | 1 (native FP4 MMA arm on; fails closed: SM12x, Triton >= 3.6,
Helion, the K5 PTX assertion; a bucket without a shipped config stays on the
incumbent unless SUFFIX_HELION_NVFP4_GEMM_FP4MMA_STRICT=1).
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no Helion: layout math, references, cases and plans still import
    helion = hl = PowerOfTwoFragment = None

ENV = "SUFFIX_HELION_NVFP4_GEMM_FP4MMA"
STRICT_ENV = "SUFFIX_HELION_NVFP4_GEMM_FP4MMA_STRICT"
MARKER = "[suffix nvfp4-gemm fp4mma]"
FAMILY = "nvfp4_gemm_fp4mma"

# lab kernel names (= configs/helion/<name>/<platform>.json)
WORDS, BYTES, LIN, CVT = ("nvfp4_gemm_q_fp4mma", "nvfp4_gemm_q_fp4mma_b",
                          "nvfp4_gemm_q_fp4mma_lin", "nvfp4_gemm_q_fp4cvt")
QUANT = "nvfp4_quant_act"  # shared with nvfp4_gemm.py (same body, same config table)
SERVED = (WORDS, BYTES, CVT)  # prepare() prefers them in this order per bucket
PREFER = SERVED
BUCKETS = (1, 8, 16, 32, 64, 128, 192)  # tuned M; serving M -> hk.bucket(m, BUCKETS)
# The bf16-activation route quantizes with nvfp4_gemm.py's `nvfp4_quant_act_body`, which clamps the row
# of its swizzled scale write to ONE 128-row block (it was written for the oxide lane's M <= 64): above
# 128 rows it would overwrite block 0. The prequant route (vLLM's own quant, the lm_head) has no such cap.
QUANT_MAX_M = 128
PREFILL_BUCKETS = (512, 2048, 8192)  # lab only: FP4MMA_PREFILL_CASES=1
MAX_M = 192  # V4.1 verify batch 32 x (1 + 5); the lanes' own caps decide what actually calls us
FP4_RANGE = 448.0 * 6.0
SPLIT_COLS = 4 * 188 * 32  # ~4 CTAs per SM at 32 columns: the split-K cap's scale

# Search-space bounds, module constants so tests can force small tiles in eager mode (the eager
# mode of register_block_size returns the MAX). BK is in packed BYTES (2 e2m1 each): 32 bytes =
# 64 elements = the m16n8k64 instruction's K, the Triton minimum (K < 64 per tile is a known
# crash: open PR triton#11776 per the oxide-retirement dossier). BM >= 16 (m16), BN >= 32 (believed,
# untested: four warps side by side at n8 each; 8 warps would need BN >= 64, and an invalid
# combination is a config the autotuner skips) until the P0 table says better. The P0 probe
# (scripts/fp4mma_p0_probe.py) prints the legal (BM, BN, BK, warps, stages) table; tighten here.
BM_MIN, BM_MAX = 16, 128
BN_MIN, BN_MAX = 32, 256
BK_MIN, BK_MAX = 32, 256

# the Triton function name of each Helion body (`_helion_<body.__name__>`), what K5 matches
TRITON_NAMES = {WORDS: "_helion_nvfp4_gemm_q_fp4mma", BYTES: "_helion_nvfp4_gemm_q_fp4mma_b",
                LIN: "_helion_nvfp4_gemm_q_fp4mma_lin", CVT: "_helion_nvfp4_gemm_q_fp4cvt"}


# ---------------------------------------------------------------------------
# knob
# ---------------------------------------------------------------------------
def enabled() -> bool:
    """SUFFIX_HELION_NVFP4_GEMM_FP4MMA: 0 (default) | 1. Loud on anything else."""
    raw = os.environ.get(ENV, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise RuntimeError(f"{MARKER} {ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def strict() -> bool:
    raw = os.environ.get(STRICT_ENV, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise RuntimeError(f"{MARKER} {STRICT_ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


# ---------------------------------------------------------------------------
# scale layout math (K1 contract; byte-exact tests in tests/test_fp4mma_layout.py)
# ---------------------------------------------------------------------------
def quads(nkb: int) -> int:
    """4-block groups (k64 steps) per scale row, padded: round4(K/16) / 4."""
    return -(-nkb // 4)


def swz_byte_index(row, kb, nkb: int):
    """Flat byte offset of scale (row, kb) in the 128x4-swizzled buffer (any int
    numpy / torch arrays or scalars). = kernels/nvfp4_gemm.sf_offset."""
    atom = (row // 128) * quads(nkb) + kb // 4
    return atom * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


def swz_word_index(row, q, nkb: int):
    """Flat int32-WORD offset of the 4 scales (blocks 4q..4q+3) of `row` in the swizzled
    buffer: the byte offset of (row, 4q) / 4. The four bytes are contiguous because kb % 4
    is the innermost swizzle coordinate; little-endian: byte j of the word = block 4q + j."""
    return (row // 128) * (quads(nkb) * 128) + q * 128 + (row % 32) * 4 + (row // 32) % 4


def lin_word_index(row, q, nkb: int):
    """Same for a LINEAR [rows, K/16] scale matrix (row stride K/16 = 4 * words)."""
    return row * (nkb // 4) + q


def as_words(sf: torch.Tensor) -> torch.Tensor:
    """Flat int32 view of a scale buffer (e4m3 or uint8 bytes, any shape, contiguous, numel % 4
    == 0; an int32 buffer is taken as the words already): metadata only, the same bytes.
    Little-endian hosts and GPUs only (the unpack assumes byte j of a word = block 4q + j)."""
    if not sf.is_contiguous():
        raise RuntimeError(f"{MARKER} scale buffer must be contiguous, got shape "
                           f"{tuple(sf.shape)} strides {tuple(sf.stride())}")
    if sf.dtype == torch.int32:
        return sf.reshape(-1)
    if sf.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise RuntimeError(f"{MARKER} scale buffer dtype {sf.dtype}: want float8_e4m3fn, uint8 "
                           "or int32 words")
    if sf.numel() % 4:
        raise RuntimeError(f"{MARKER} scale buffer numel {sf.numel()} % 4 != 0 (the swizzled "
                           "layout pads rows to 128 and columns to 4)")
    return sf.reshape(-1).view(torch.uint8).view(torch.int32)


def words_to_bytes(words: torch.Tensor) -> torch.Tensor:
    """int32 [..] -> uint8 [.., 4] in k order (block 4q + j = byte j): the host twin of the
    in-kernel unpack (shift / mask / join)."""
    w = words.to(torch.int64)
    return torch.stack([(w >> (8 * j)) & 255 for j in range(4)], -1).to(torch.uint8)


def linearize(sf_swz: torch.Tensor, rows: int, nkb: int) -> torch.Tensor:
    """Swizzled scale buffer (any shape) -> uint8 [rows, nkb] row-major (the `lin` copy)."""
    return ng_unswizzle(sf_swz, rows, nkb).contiguous()


def ng_unswizzle(sf_swz, rows, nkb):
    from suffix_hybrid.kernels.nvfp4_gemm import unswizzle_sf
    return unswizzle_sf(sf_swz, rows, nkb)


def validate_scales(sf: torch.Tensor, what: str = "scales") -> None:
    """Fail closed (AGENTS.md 1.7) on scale bytes the MMA does not define: ue4m3 has no sign
    bit and no NaN (0x7F); a checkpoint byte >= 0x7F would feed NaN / garbage to the tensor core.
    One device sync, at load."""
    b = sf.reshape(-1).view(torch.uint8)
    if bool((b >= 0x7F).any()):
        raise RuntimeError(f"{MARKER} {what}: e4m3 scale byte >= 0x7F (sign bit or NaN code) — "
                           "ue4m3 block scales must be in [0x00, 0x7E]; refusing to serve it on the "
                           "native FP4 MMA")


# ---------------------------------------------------------------------------
# Helion bodies. Literals only (BM_MIN.. are module constants: eager tests patch them).
# `kh` = K / 2 bytes; a K step of `bk` bytes is bk / 32 quads = bk / 8 scale blocks.
# Split program tiles: m rows, n rows of w, a range of k_split bytes (a multiple of 32).
# ---------------------------------------------------------------------------
def nvfp4_gemm_q_fp4mma(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor,
                        out: torch.Tensor, partial: torch.Tensor, alpha: float) -> torch.Tensor:
    """xq uint8 [M, K/2], xsf int32 words of the swizzled activation scales (rows padded to
    128), w uint8 [N, K/2], wsf int32 words of the swizzled weight scales, out bf16 [M, N];
    partial f32 [cap, M, N] (cap = the largest split_k; serving passes the tuned split, or
    [1, 1, 1] when it does not split). Writes and returns out."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    cp4 = (kh // 8 + 3) // 4
    rw = cp4 * 128
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 32)) * 32
    n_split = helion.cdiv(kh, k_split)
    bm = hl.register_block_size(BM_MIN, BM_MAX)
    bn = hl.register_block_size(BN_MIN, BN_MAX)
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        rm = torch.clamp(tile_m.index, max=m - 1)[:, None]
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        ra = (rm >> 7) * rw + (rm & 31) * 4 + ((rm >> 5) & 3)
        rb = (rn >> 7) * rw + (rn & 31) * 4 + ((rn >> 5) & 3)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5,
                            max=cp4 - 1)[None, :] * 128
            wa = xsf[ra + q]
            wb = wsf[rb + q]
            sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
            sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
            sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
            sb = sb.reshape(sb.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
            acc = hl.dot_scaled(xq[tile_m, tile_k], sa, "e2m1", w[tile_n, tile_k].T, sb, "e2m1",
                                acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc * alpha
    if n_split > 1:
        out.copy_(partial[:n_split].sum(0))
    return out


def nvfp4_gemm_q_fp4mma_b(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor,
                          wsf: torch.Tensor, out: torch.Tensor, partial: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    """As nvfp4_gemm_q_fp4mma, but xsf / wsf are the swizzled scales as e4m3 BYTES (flat) and
    the scale operand is a plain byte gather: sf[offset] per (row, 16-block)."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    nkb = kh // 8
    cp = (nkb + 3) // 4 * 4
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 32)) * 32
    n_split = helion.cdiv(kh, k_split)
    bm = hl.register_block_size(BM_MIN, BM_MAX)
    bn = hl.register_block_size(BN_MIN, BN_MAX)
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        rm = torch.clamp(tile_m.index, max=m - 1)[:, None]
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            kb = torch.clamp(torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3,
                             max=nkb - 1)[None, :]
            sa = xsf[(rm >> 7) * (cp * 128) + (kb >> 2) * 512 + (rm & 31) * 16
                     + ((rm >> 5) & 3) * 4 + (kb & 3)]
            sb = wsf[(rn >> 7) * (cp * 128) + (kb >> 2) * 512 + (rn & 31) * 16
                     + ((rn >> 5) & 3) * 4 + (kb & 3)]
            acc = hl.dot_scaled(xq[tile_m, tile_k], sa, "e2m1", w[tile_n, tile_k].T, sb, "e2m1",
                                acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc * alpha
    if n_split > 1:
        out.copy_(partial[:n_split].sum(0))
    return out


def nvfp4_gemm_q_fp4mma_lin(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, ws: torch.Tensor,
                            out: torch.Tensor, partial: torch.Tensor, alpha: float) -> torch.Tensor:
    """As nvfp4_gemm_q_fp4mma, but xs / ws are int32 words of LINEAR [rows, K/16] scales
    (row stride K/64 words, no row padding): `linearize` of the swizzled weight scales, and
    vLLM's scaled_fp4_quant(is_sf_swizzled_layout=False) for the activation."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    cp4 = kh // 32
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 32)) * 32
    n_split = helion.cdiv(kh, k_split)
    bm = hl.register_block_size(BM_MIN, BM_MAX)
    bn = hl.register_block_size(BN_MIN, BN_MAX)
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        ra = torch.clamp(tile_m.index, max=m - 1)[:, None] * cp4
        rb = torch.clamp(tile_n.index, max=n - 1)[:, None] * cp4
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5,
                            max=cp4 - 1)[None, :]
            wa = xs[ra + q]
            wb = ws[rb + q]
            sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
            sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
            sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
            sb = sb.reshape(sb.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
            acc = hl.dot_scaled(xq[tile_m, tile_k], sa, "e2m1", w[tile_n, tile_k].T, sb, "e2m1",
                                acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc * alpha
    if n_split > 1:
        out.copy_(partial[:n_split].sum(0))
    return out


def nvfp4_gemm_q_fp4cvt(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor,
                        out: torch.Tensor, partial: torch.Tensor, alpha: float) -> torch.Tensor:
    """Control arm, same operands as nvfp4_gemm_q_fp4mma (swizzled scale WORDS): both operands
    decoded with the hardware cvt.rn.f16x2.e2m1x2 (hl.load_float4_e2m1fn_x16_to_float16: one
    64-bit load = 16 values), times the e4m3 block scale in f16 (exact: <= 6 significant bits,
    |v| <= 2688), f16 hl.dot with f32 accumulate. Separates "decode ALU" from "FP4 MMA"."""
    m = xq.size(0)
    n = hl.specialize(w.size(0))
    kh = hl.specialize(w.size(1))
    kg = kh // 8
    cp4 = (kg + 3) // 4
    rw = cp4 * 128
    xf = xq.view(-1)
    wf = w.view(-1)
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 32)) * 32
    n_split = helion.cdiv(kh, k_split)
    bm = hl.register_block_size(BM_MIN, BM_MAX)
    bn = hl.register_block_size(BN_MIN, BN_MAX)
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[bm, bn, k_split]):
        rm = torch.clamp(tile_m.index, max=m - 1)[:, None]
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        ra = (rm >> 7) * rw + (rm & 31) * 4 + ((rm >> 5) & 3)
        rb = (rn >> 7) * rw + (rn & 31) * 4 + ((rn >> 5) & 3)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            g = torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3
            gok = (g < (tile_s.end >> 3))[None, :]
            gc = torch.clamp(g, max=kg - 1)[None, :]
            q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5,
                            max=cp4 - 1)[None, :] * 128
            wa = xsf[ra + q]
            wb = wsf[rb + q]
            sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
            sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float16)
            sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
            sb = sb.reshape(sb.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float16)
            # lane i of an 8-byte group = element i (low nibble first); activation rows past M and
            # K groups past the split range decode to 0 (extra_mask)
            ma = (tile_m.index < m)[:, None] & gok
            mb = (tile_n.index < n)[:, None] & gok
            xo = rm * kg + gc
            wo = rn * kg + gc
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = \
                hl.load_float4_e2m1fn_x16_to_float16(xf, xo, ma)
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 = \
                hl.load_float4_e2m1fn_x16_to_float16(wf, wo, mb)
            av = hl.join(hl.join(hl.join(hl.join(a0, a8), hl.join(a4, a12)),
                                 hl.join(hl.join(a2, a10), hl.join(a6, a14))),
                         hl.join(hl.join(hl.join(a1, a9), hl.join(a5, a13)),
                                 hl.join(hl.join(a3, a11), hl.join(a7, a15))))
            bv = hl.join(hl.join(hl.join(hl.join(b0, b8), hl.join(b4, b12)),
                                 hl.join(hl.join(b2, b10), hl.join(b6, b14))),
                         hl.join(hl.join(hl.join(b1, b9), hl.join(b5, b13)),
                                 hl.join(hl.join(b3, b11), hl.join(b7, b15))))
            av = (av.reshape(av.size(0), av.size(1), 16) * sa[:, :, None]).reshape(av.size(0), -1)
            bv = (bv.reshape(bv.size(0), bv.size(1), 16) * sb[:, :, None]).reshape(bv.size(0), -1)
            acc = hl.dot(av, bv.T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc * alpha
    if n_split > 1:
        out.copy_(partial[:n_split].sum(0))
    return out


BODIES = {WORDS: nvfp4_gemm_q_fp4mma, BYTES: nvfp4_gemm_q_fp4mma_b, LIN: nvfp4_gemm_q_fp4mma_lin,
          CVT: nvfp4_gemm_q_fp4cvt}


# ---------------------------------------------------------------------------
# eager (HELION_INTERPRET=1) references for the two ops Helion ships WITHOUT a CPU twin that
# handles NVFP4: its dot_scaled ref is E8M0-only and treats packed bytes as plain numbers; its
# FP4 decode primitives have no ref at all. CPU tests / the CPU capture-gate runtime install these.
# ---------------------------------------------------------------------------
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [R, C/2] -> f64 [R, C], element 2i = LOW nibble (sign bit 3, magnitude bits 0-2)."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=packed.device)
    code = torch.stack([packed & 15, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    val = lut[code & 7]
    return torch.where((code & 8) != 0, -val, val)


def dot_scaled_ref(mat1, mat1_scale, mat1_format, mat2, mat2_scale, mat2_format, acc=None,
                   out_dtype=None):
    """hl.dot_scaled for e2m1 x e2m1 with e4m3 block-16 scales: mat1 [M, K/2] packed, mat1_scale
    [M, K/16]; mat2 [K/2, N] packed along K, mat2_scale [N, K/16]. The product is computed in f64
    (exact: <= 12 significant bits per term) and rounded to f32 once per call; `acc + result`
    in f32 like the compiled op's accumulate."""
    if mat1_format != "e2m1" or mat2_format != "e2m1":
        raise NotImplementedError(f"{MARKER} eager dot_scaled ref: e2m1 x e2m1 only")
    if mat1_scale.dtype != torch.float8_e4m3fn or mat2_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{MARKER} eager dot_scaled ref: NVFP4 scales must be float8_e4m3fn, got "
                        f"{mat1_scale.dtype} / {mat2_scale.dtype} (an E8M0 uint8 scale is the "
                        "MX format: the compiled op would pick the 2X e8m0 instruction)")
    a = unpack_e2m1(mat1) * mat1_scale.to(torch.float64).repeat_interleave(16, dim=1)
    b = unpack_e2m1(mat2.T.contiguous()) * mat2_scale.to(torch.float64).repeat_interleave(16, dim=1)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"{MARKER} eager dot_scaled ref: K {a.shape[1]} vs {b.shape[1]} "
                         "(scale tiles must be [rows, K/16])")
    res = (a @ b.T).to(torch.float32)
    return res if acc is None else acc + res


def load_fp4x16_ref(storage, group_offsets, extra_mask=None):
    """hl.load_float4_e2m1fn_x16_to_float16: 16 f16 tensors shaped like group_offsets; lane i =
    element i of the 8-byte group (low nibble first); masked-off lanes are 0."""
    flat = storage.reshape(-1)
    idx = group_offsets.to(torch.int64)[..., None] * 8 + torch.arange(8, device=flat.device)
    byt = flat[idx]
    if extra_mask is not None:
        byt = torch.where(extra_mask[..., None], byt, torch.zeros_like(byt))
    vals = unpack_e2m1(byt.reshape(-1, 8)).reshape(*group_offsets.shape, 16).to(torch.float16)
    return tuple(vals[..., i] for i in range(16))


@contextlib.contextmanager
def eager_ref():
    """Install the NVFP4 eager refs for `hl.dot_scaled` and the FP4 x16 load (HELION_INTERPRET
    runs only; restores whatever was registered)."""
    if hl is None:
        yield
        return
    ops = (hl.dot_scaled, hl.load_float4_e2m1fn_x16_to_float16)
    fns = (dot_scaled_ref, load_fp4x16_ref)
    old_fns = [op._ref_fn for op in ops]
    for op, fn in zip(ops, fns):
        op._ref_fn = fn
    try:
        yield
    finally:
        for op, fn in zip(ops, old_fns):
            op._ref_fn = fn


# ---------------------------------------------------------------------------
# references / checks (pure torch)
# ---------------------------------------------------------------------------
def _u8(sf: torch.Tensor) -> torch.Tensor:
    return sf.reshape(-1).view(torch.uint8)


def reference_q(xq, xsf, w, wsf, out, partial, alpha):
    """Exact (f64) alpha * deq(xq) @ deq(w).T for the swizzled-scale bodies (words, bytes, cvt):
    xsf / wsf are the swizzled scale buffers as int32 words or e4m3 bytes (same bytes)."""
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref
    return exact_ref(xq, _u8(xsf), w, _u8(wsf), alpha, w.shape[0])


def reference_lin(xq, xs, w, ws, out, partial, alpha):
    """Exact (f64) reference of the `lin` body: scales are int32 words of linear [rows, K/16]."""
    from suffix_hybrid.kernels.nvfp4_gemm import dequant_torch
    m, n, nkb = xq.shape[0], w.shape[0], w.shape[1] // 8
    a = dequant_torch(xq.contiguous(), _u8(xs).reshape(m, nkb))
    step = max(1, (1 << 24) // (2 * w.shape[1]))
    res = torch.empty(m, n, dtype=torch.float64, device=xq.device)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        res[:, r0:r1] = a @ dequant_torch(w[r0:r1], _u8(ws).reshape(n, nkb)[r0:r1]).T
    return res * alpha


check = None  # bound below to the shared bf16 checker


def _bind_check():
    global check
    from suffix_hybrid.helion_kernels import nvfp4_gemm as old
    check = old.check


_bind_check()


# ---------------------------------------------------------------------------
# lab operands (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def _ignore() -> list:
    """Helion warns that the stage-2 torch reduce is not fused into the kernel: intended."""
    return [helion.exc.TensorOperationInWrapper] if helion is not None else []


def _old():
    from suffix_hybrid.helion_kernels import nvfp4_gemm as old
    return old


def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def problem(case: dict, seed: int, device=None, kind: str = WORDS) -> dict:
    """Random NVFP4 operands in vLLM layout (prequant route; see nvfp4_gemm.problem) as the
    body of `kind` wants them. `out` / `partial` are ZEROS, not empty: Helion's autotuner
    decides which args a kernel mutates by comparing them before / after the baseline run and
    NaN garbage never compares equal (lane D lesson)."""
    device = device or _dev()
    p = _old().problem(case, seed, device, True)
    m, n, k = p["m"], p["n"], p["k"]
    nkb = k // 16
    out = dict(m=m, n=n, k=k, xq=p["xq"], w=p["w"], alpha=p["alpha"], xsf_swz=p["xsf"],
               wsf_swz=p["wsf"],
               out=torch.zeros(m, n, dtype=torch.bfloat16, device=device),
               partial=torch.zeros(_old().max_splits(n), m, n, dtype=torch.float32, device=device))
    if kind == BYTES:
        out.update(xsf=p["xsf"].reshape(-1).view(torch.float8_e4m3fn),
                   wsf=p["wsf"].reshape(-1).view(torch.float8_e4m3fn))
    elif kind == LIN:
        out.update(xsf=as_words(linearize(p["xsf"], m, nkb)), wsf=as_words(linearize(p["wsf"], n, nkb)))
    else:
        out.update(xsf=as_words(p["xsf"]), wsf=as_words(p["wsf"]))
    return out


def body_args(kernel: str, p: dict) -> tuple:
    return (p["xq"], p["xsf"], p["w"], p["wsf"], p["out"], p["partial"], p["alpha"])


# ---------------------------------------------------------------------------
# serving: per-M plan at load + the oxide-signature twins (drop-in for the lanes'
# `hx` seam: nvfp4_linear.py / nvfp4_lm_head.py call prepare / describe / warm_ms /
# partial_elems / nvfp4_gemm_cuda / nvfp4_gemm_q_cuda on whichever module they bound).
# Capture-safety contract (docs/dsv41-fp4.md section 7): the ONLY allocations are
# torch.empty per call (graph-private under capture): the split-K partial and, on
# the bf16 route, the quantized activation; no module scratch, no counters, no
# atomics; the lane's own `partial` / `counters` / `aq` / `asf` / `stream_ptr` /
# `splits` / `fused` arguments are oxide launch state and unused.
# ---------------------------------------------------------------------------
_PLANS: dict = {}  # (prequant, n, k) -> [None, entry for M = 1..max_m]
GATE_NAME = "fp4mma_capture_replay"


def case_of(n: int, k: int, b: int) -> dict:
    return {"k": k, "n": n, "numtokens": b}


def _entry_for(name: str, n: int, k: int, b: int):
    """(compiled kernel, split_k) of body `name` for (N, K, bucket b), or None."""
    case = case_of(n, k, b)
    cfg = hk.config(name, case)
    if cfg is None:
        return None
    return hk.compiled(BODIES[name], name, case, ignore_warnings=_ignore()), int(cfg.get("split_k", 1))


def prepare(n: int, k: int, prequant: bool, max_m: int, mode: str) -> list:
    """[None] + per M = 1..max_m: (label, kernels, split_k, bucket) or None (the incumbent
    serves). label = the body's kernel name, 'nvfp4_quant_act+<body>' on the bf16 route;
    kernels = (gemm,) or (quant, gemm). mode: 'auto' takes the family where a config is
    shipped (= where it beat the incumbent in the lab), 'helion' (and STRICT=1) raises when a
    bucket <= max_m has none. Binds configs only: the lane's load oracle runs (= compiles)
    every `warm_ms` and K5 (fp4mma_probe) then asserts the PTX of what compiled."""
    if not enabled():
        raise RuntimeError(f"{MARKER} prepare called with {ENV} unset/0: the lane must not "
                           "import this family when the knob is off")
    if mode not in ("auto", "helion"):
        raise RuntimeError(f"{MARKER} prepare: mode {mode!r} (the incumbent never reaches the family)")
    if k % 64 or n % 16:
        raise RuntimeError(f"{MARKER} prepare: N={n} % 16 or K={k} % 64 (the family's contract)")
    if not prequant and max_m > QUANT_MAX_M:
        raise RuntimeError(f"{MARKER} prepare: the bf16-activation route is capped at M <= {QUANT_MAX_M} "
                           f"(got {max_m}): the shared quant kernel writes one 128-row scale block; "
                           "use the prequant route (vLLM's quant) above that")
    from suffix_hybrid.kernels import fp4mma_probe
    fp4mma_probe.require_native_stack()
    fp4mma_probe.install_recorder()
    old = _old()
    fail_closed = mode == "helion" or strict()
    per = {}
    for b in sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)}):
        e = None
        for name in PREFER:
            g = _entry_for(name, n, k, b)
            if g is None:
                continue
            if prequant:
                e = (name, (g[0],), g[1], b)
            else:
                qc = {"k": k, "numtokens": b}
                if hk.config(QUANT, qc) is not None:
                    e = (f"{QUANT}+{name}", (hk.compiled(old.BODIES[QUANT], QUANT, qc), g[0]), g[1], b)
            if e is not None:
                break
        if e is None and fail_closed:
            raise RuntimeError(
                f"{MARKER} {ENV}=1 ({'helion' if mode == 'helion' else 'STRICT'}): no tuned "
                f"{' / '.join(PREFER)}{'' if prequant else ' + ' + QUANT} config for N={n} K={k} "
                f"M bucket {b} on {hk.platform()} (tune it in the autotune lab / fp4mma_bench and "
                f"ship configs/helion/<kernel>/<platform>.json, or unset {STRICT_ENV})")
        per[b] = e
    plan = [None] + [per[hk.bucket(m, BUCKETS)] for m in range(1, max_m + 1)]
    key = (prequant, n, k)
    if len(plan) > len(_PLANS.get(key, ())):
        _PLANS[key] = plan
    return plan


def verify_served(plans, allow_local: bool = True) -> list:
    """K5 for the dense lane after its warm-up (`warm_ms` of every plan, under the recorder that
    `prepare` installed): the dot_scaled bodies the plans use must have compiled to the native
    kind::mxf4nvf4 instruction for sm_120a (fp4mma_probe.assert_native); the hardware-cvt control
    body must carry cvt.rn.f16x2.e2m1x2. Fp4mmaError (fail closed) otherwise. No-op for no plan."""
    from suffix_hybrid.kernels import fp4mma_probe
    used = {_body_of(e[0]) for plan in plans for e in (plan or [])[1:] if e}
    native = sorted(TRITON_NAMES[b] for b in used if b != CVT)
    out = fp4mma_probe.assert_native(native, allow_local=allow_local) if native else []
    if CVT in used:
        out += fp4mma_probe.assert_contains([TRITON_NAMES[CVT]], ["cvt.rn.f16x2.e2m1x2"],
                                            "hardware FP4 decode")
    return out


def partial_elems(n: int, plans) -> int:
    """f32 scratch the LANE must reserve for these plans: 0 (the family allocates its split-K
    partial per call from torch's allocator, graph-private under capture)."""
    return 0


def warm_ms(max_m: int) -> list:
    """Ms that compile every Triton specialization the plans can launch: M == 1, and per
    bucket its top row count and the one below (M % 16 == 0 is its own specialization),
    clipped to max_m."""
    out, prev = {1}, 0
    for b in BUCKETS:
        if prev >= max_m:
            break
        top = min(b, max_m)
        out |= {top, max(prev + 1, top - 1)}
        prev = b
    return sorted(out)


def describe(plan: list) -> str:
    """'M1:nvfp4_gemm_q_fp4mma M2-8:incumbent ...' for the load log."""
    runs, last = [], object()
    for m, e in enumerate(plan[1:], 1):
        name = e[0] + (f"(split<={e[2]})" if e[2] > 1 else "") if e else "incumbent"
        if name == last:
            runs[-1][1] = m
        else:
            runs.append([m, m, name])
            last = name
    return " ".join(f"M{a}{'' if a == b else f'-{b}'}:{s}" for a, b, s in runs)


def _plan_entry(prequant: bool, n: int, k: int, m: int):
    plan = _PLANS.get((prequant, n, k))
    e = plan[m] if plan is not None and 1 <= m < len(plan) else None
    if e is None:
        raise RuntimeError(f"{MARKER} no fp4mma plan for N={n} K={k} M={m} "
                           f"({'prequant' if prequant else 'bf16'} route): the lane routes only "
                           "planned M here")
    return e


def _body_of(label: str) -> str:
    return label.split("+")[-1]


def gemm(xq, xsf, w, w_sf, out, entry, alpha: float, poison_nan: bool = False):
    """out[M, N] = alpha * the planned prequant GEMM of `entry` (the lanes' twins, the
    capture/replay gate and the bench all run exactly this). xq uint8 [M, K/2], xsf / w_sf the
    swizzled scales (e4m3 or uint8, any shape, contiguous; ONE int32-word view or e4m3-byte view
    per call, metadata only), out bf16 [M, N] (row stride free). The only allocation: the f32
    split-K partial from torch's allocator, [split, M, N] or a [1, 1, 1] stub when the config does
    not split (the kernel never touches it). poison_nan (checks only): NaN-fill the partial first,
    so a split a kernel forgets to write fails the exact check instead of reading stale bytes."""
    label, kerns, split, _ = entry
    name = _body_of(label)
    m, n = xq.shape[0], w.shape[0]
    partial = torch.empty((split, m, n) if split > 1 else (1, 1, 1), dtype=torch.float32,
                          device=xq.device)
    if poison_nan:
        partial.fill_(float("nan"))
    if name == BYTES:
        sx, sw = _u8(xsf).view(torch.float8_e4m3fn), _u8(w_sf).view(torch.float8_e4m3fn)
    else:
        sx, sw = as_words(xsf), as_words(w_sf)
    return kerns[-1](xq, sx, w, sw, out, partial, alpha)


def nvfp4_gemm_q_cuda(xq, xsf, w, w_sf, partial, out, alpha, splits, stream_ptr, fused=0,
                      counters=None):
    """Twin of _native.nvfp4_gemm_q_cuda (vLLM-prequantized x: packed [M, K/2] + 128x4-swizzled
    scales), same call. `partial` / `splits` / `stream_ptr` / `fused` / `counters` are oxide
    launch state: unused (Helion launches on torch's current stream, its split count is its tuned
    config, its partial comes from torch's allocator)."""
    m, kh = xq.shape
    entry = _plan_entry(True, w.shape[0], 2 * kh, m)
    return gemm(xq, xsf, w, w_sf, out, entry, float(alpha))


def quant_gemm(x, w, w_sf, out, entry, gscale: float, alpha: float, poison_nan: bool = False):
    """bf16 route of `entry` ('nvfp4_quant_act+<body>'): the existing Helion quant kernel
    (bit-exact vs the spec quant; its inv table is a per-(device, g) cache filled at the first
    call = the lane's warm-up, before any capture) then the prequant GEMM. The quantized activation
    is a per-call torch.empty: every byte of rows < M is written, rows >= M are never read (masked)."""
    m, k = x.shape
    old = _old()
    quant = entry[1][0]
    dev = x.device
    inv, g6 = old._inv(gscale, dev), float(np.float32(gscale) / np.float32(6.0))
    rows = -(-m // 128) * 128
    xq = torch.empty(m, k // 16, 8, dtype=torch.uint8, device=dev)
    xsf = torch.empty(rows * (k // 16), dtype=torch.float8_e4m3fn, device=dev)
    quant(x.reshape(m, k // 16, 16), xq, xsf, inv, g6)
    return gemm(xq.view(m, k // 2), xsf, w, w_sf, out, entry, alpha, poison_nan)


def nvfp4_gemm_cuda(x, w, w_sf, aq, asf, partial, out, gscale, alpha, splits, stream_ptr, fused=0,
                    counters=None):
    """Twin of _native.nvfp4_gemm_cuda (bf16 route), same call: out = bf16(alpha * nvfp4(x) @
    w^T). aq / asf / partial / splits / stream_ptr / fused / counters are oxide launch state:
    unused (aq / asf are NOT written)."""
    m, k = x.shape
    entry = _plan_entry(False, w.shape[0], k, m)
    return quant_gemm(x, w, w_sf, out, entry, float(gscale), float(alpha))


# ---------------------------------------------------------------------------
# capture/replay probes (kernels/fp4mma_probe.py runs them; the CPU tests use EagerRuntime)
# ---------------------------------------------------------------------------
def probe_config(m: int, split: int) -> dict:
    """Explicit, shape-independent config: 64-byte (128-element) K steps, 64-column tiles, rows
    grow with M. block_sizes order = registration order (bm, bn, bk)."""
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    return {"split_k": split, "block_sizes": [bm, 64, 64], "num_warps": 4, "num_stages": 3}


def probes(tier: str) -> list:
    from suffix_hybrid.kernels.fp4mma_probe import Probe

    def p(label, body, n, k, m, split, route="prequant", **spec):
        return Probe(label, "gemm", dict(body=body, n=n, k=k, m=m, split=split, route=route, **spec))

    if tier == "tiny":  # CPU tests
        return [p("tiny.words.s1", WORDS, 64, 192, 5, 1), p("tiny.words.s2", WORDS, 160, 640, 17, 2),
                p("tiny.bytes.s2", BYTES, 64, 192, 5, 2), p("tiny.lin.s2", LIN, 64, 192, 5, 2),
                p("tiny.cvt.s2", CVT, 64, 192, 5, 2),
                p("tiny.bf16.s2", WORDS, 64, 192, 5, 2, route="bf16")]
    if tier == "quick":
        return [p("lmhead.m1.s1", WORDS, 16160, 5120, 1, 1), p("lmhead.m64.s1", WORDS, 16160, 5120, 64, 1),
                p("lmhead.m192.s1", WORDS, 16160, 5120, 192, 1),
                p("lmhead.m192.bytes", BYTES, 16160, 5120, 192, 1),
                p("lmhead.m16.lin", LIN, 16160, 5120, 16, 1),
                p("lmhead.m64.cvt", CVT, 16160, 5120, 64, 1),
                p("mlp_down.m16.s4", WORDS, 5120, 17408, 16, 4),
                p("gate_up.m8.s1", WORDS, 34816, 5120, 8, 1),
                p("gemma_down.m17.s2", WORDS, 2816, 2112, 17, 2),
                p("attn_o.m5.bf16.s2", WORDS, 5120, 6144, 5, 2, route="bf16")]
    from suffix_hybrid.kernels.nvfp4_gemm import ALL_SHAPES
    shapes = dict.fromkeys([*ALL_SHAPES.values(), V41_LM_HEAD])
    out = []
    for (n, k) in shapes:
        for m in (1, 8, 64, 192):
            for split in (1, 2, 8):
                for body in (WORDS, BYTES, CVT):
                    out.append(p(f"{n}x{k}.m{m}.{body[len('nvfp4_gemm_q_'):]}.s{split}", body, n, k, m, split))
    return out


def shipped_probes() -> list:
    """Every config the serving lane would run on this platform (configs/helion/<kernel>/...)."""
    import json
    from suffix_hybrid.kernels.fp4mma_probe import Probe
    out = []
    for kernel in SERVED:
        for key, cfg in hk.table(kernel).items():
            case = json.loads(key)
            out.append(Probe(f"shipped.{kernel}.{case['n']}x{case['k']}.m{case['numtokens']}", "gemm",
                             dict(body=kernel, n=case["n"], k=case["k"], m=case["numtokens"],
                                  split=int(cfg.get("split_k", 1)), route="prequant", cfg=dict(cfg)),
                             "shipped"))
    return out


def build_probe(pr, dev):
    """Case for a dense probe: fixed weights, three distinct activations, the explicit config."""
    from suffix_hybrid.kernels.fp4mma_probe import Case
    sp = pr.spec
    body, n, k, m, split, route = (sp[x] for x in ("body", "n", "k", "m", "split", "route"))
    cfg = {**probe_config(m, split), **sp.get("cfg", {})}
    old = _old()
    kern = helion.kernel(BODIES[body], config=helion.Config(**cfg), static_shapes=False,
                         ignore_warnings=_ignore())
    case = {"k": k, "n": n, "numtokens": m}
    base = problem(case, 10, dev, body)
    w, wsf, alpha = base["w"], base["wsf"], base["alpha"]
    wsf_swz = base["wsf_swz"]
    g = FP4_RANGE / 4.5
    if route == "bf16":
        qkern = helion.kernel(old.BODIES[QUANT], config=helion.Config(block_sizes=[16, 16]),
                              static_shapes=False)
        entry = (f"{QUANT}+{body}", (qkern, kern), split, m)
        alpha = float(np.float32(1.0 / (np.float32(g) * np.float32(FP4_RANGE))))
        gen = torch.Generator(device="cpu").manual_seed(77 + n + k + m)
        inputs = [(torch.randn(m, k, generator=gen).clamp(-4.5, 4.5)).to(torch.bfloat16).to(dev)
                  for _ in range(3)]
        copy_in = lambda st, x: st.copy_(x)

        def run(static, out=None):
            o = out if out is not None else torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            o.fill_(float("nan"))
            return quant_gemm(static, w, wsf, o, entry, g, alpha, poison_nan=True)

        def exact(x, out):
            xq, xsf = old._spec_quant(x, g)
            return check(out, _exact_swizzled(xq, xsf, w, wsf_swz, alpha, n))
    else:
        entry = (body, (kern,), split, m)
        inputs = []
        for i in range(3):
            pi = problem(case, 11 + i, dev, body)
            inputs.append({"xq": pi["xq"], "xsf": pi["xsf"], "xsf_swz": pi["xsf_swz"]})
        copy_in = lambda st, x: [st[key].copy_(x[key]) for key in st]

        def run(static, out=None):
            o = out if out is not None else torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            o.fill_(float("nan"))
            return gemm(static["xq"], static["xsf"], w, wsf, o, entry, alpha, poison_nan=True)

        def exact(x, out):
            if body == LIN:
                ref = reference_lin(x["xq"], x["xsf"], w, wsf, None, None, alpha)
            else:
                ref = _exact_swizzled(x["xq"], x["xsf_swz"], w, wsf_swz, alpha, n)
            return check(out, ref)
    return Case(inputs=inputs, run=run, copy_in=copy_in, exact=exact, out_shape=(m, n))


def _exact_swizzled(xq, xsf_swz, w, wsf_swz, alpha, n):
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref
    return exact_ref(xq, xsf_swz, w, wsf_swz, alpha, n)


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
V41_LM_HEAD = (16160, 5120)  # vocab 129280 / TP8 x hidden: bf16 in the checkpoint, NVFP4 via nvfp4_lm_head


def _is_dsv41(model: dict) -> bool:
    return "deepseek-v4" in str(model.get("name") or model.get("architecture") or "").lower()


def dense_shapes_for(model: dict) -> list:
    """Per-rank (N, K) of the NVFP4 linears the family serves for `model`: the plugin's per-arch
    tables (nvfp4_gemm.dense_shapes) plus the lm_head; DeepSeek-V4.1 has NO NVFP4 dense layer
    (attention / shared experts are MXFP8, mxfp8_gemm.py), only the NVFP4 lm_head."""
    old = _old()
    shapes = [] if _is_dsv41(model) else list(old.dense_shapes(model))
    head = old.head_shape(model)
    if head and head not in shapes:
        shapes.append(head)
    return shapes


def _cases(model: dict, kernel: str) -> list:
    """Every (N, K) x bucket, decode buckets first (FP4MMA_PREFILL_CASES=1 appends the prefill
    buckets); 64 cases max (the lab's per-kernel cap)."""
    if model.get("quant_format") != "nvfp4":
        return []
    shapes = dense_shapes_for(model)
    buckets = BUCKETS + (PREFILL_BUCKETS if os.environ.get("FP4MMA_PREFILL_CASES") == "1" else ())
    return [{"k": k, "n": n, "numtokens": m} for m in buckets for n, k in shapes][:64]


def _fi_args(case: dict, seed: int):
    """The production arm the family has to beat on V4.1 (no oxide there): vLLM's FlashInfer
    CUTLASS FP4 GEMM on the same prequantized operands (weights padded to N % 32, scales as the
    2-D swizzled matrices FlashInfer reads)."""
    p = problem(case, seed, "cuda", WORDS)
    m, n = p["m"], p["n"]
    xsf2d = p["xsf_swz"].reshape(-1).view(torch.float8_e4m3fn).view(-(-m // 128) * 128, -1)
    wsf2d = p["wsf_swz"].reshape(-1).view(torch.float8_e4m3fn).view(-(-n // 128) * 128, -1)
    w_fi = torch.nn.functional.pad(p["w"], (0, 0, 0, -n % 32)).contiguous()
    alpha = torch.tensor([p["alpha"]], dtype=torch.float32, device=p["w"].device)
    return p["xq"], w_fi, xsf2d, wsf2d, alpha, n


def _fi_baseline(xq, w_fi, xsf2d, wsf2d, alpha, n):
    try:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
        y = flashinfer_scaled_fp4_mm(xq, w_fi, xsf2d, wsf2d, alpha, torch.bfloat16, backend="cutlass")
    except Exception:  # FlashInfer absent: vLLM's own CUTLASS kernel
        from vllm import _custom_ops as ops
        y = ops.cutlass_scaled_fp4_mm(xq, w_fi, xsf2d, wsf2d, alpha, torch.bfloat16)
    return y[:, :n]


def _kernel(name: str):
    ref = reference_lin if name == LIN else reference_q
    return {"name": name, "body": BODIES[name], "cases": lambda model: _cases(model, name),
            "make": lambda case, seed: body_args(name, problem(case, seed, _dev(), name)),
            "make_baseline": _fi_args, "baseline": _fi_baseline,
            "reference": ref, "check": check,
            "settings": {"ignore_warnings": _ignore()},
            "baseline_name": "vLLM FlashInfer CUTLASS FP4 GEMM (prequant operands, cold L2 via the "
                             "lab's rotating copies); the oxide arm is fp4mma_bench's job",
            # CUTLASS kernel names do not tell which linear is which
            "replaces": []}


# most promising first (the lab's run budget): the primary body, the byte-gather fallback, the
# hardware-cvt control, the linear-scale arm (needs the lane's linear copy to ever serve)
KERNELS = [_kernel(WORDS), _kernel(BYTES), _kernel(CVT), _kernel(LIN)]
