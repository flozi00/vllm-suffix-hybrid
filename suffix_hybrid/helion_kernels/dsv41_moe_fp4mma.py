# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 routed experts on NATIVE FP4 MMA: Helion `hl.dot_scaled` fc1 / fc2 bodies (lane I
of the blind program) as a drop-in ALTERNATIVE to lane B's dequant-to-bf16 bodies
(helion_kernels/nvfp4_moe.py, docs/dsv41-moe.md). docs/dsv41-fp4.md is the long form.

Same layer contract as lane B (vLLM FLASHINFER_CUTLASS after weight processing): w13 uint8
[E, 2I, H/2] = [up; gate], w2 [E, H, I/2], low nibble = even element, e4m3 block-16 scales 128x4-
swizzled PER EXPERT, per-expert alphas g1 / g2, ONE activation gscale per GEMM (a1 for x, a2 for h),
swiglu clamp `gate = min(gate, L)`, `up = clamp(up, -L, L)` BEFORE silu (the placement lane B proved
against vLLM at 39983a5e7), W4A4: x and h are quantized to NVFP4 (e4m3 block-16 scales, the spec
rounding of kernels/nvfp4_moe.quant_codes), f32 accumulate. Eligibility: H % 64 == 0 and
I % 64 == 0 (a k64 step = 4 scale blocks = one 32-bit scale word; lane B only needs I % 16): V4.1 EP8
(48 of 384 experts, H 5120, I 2304) qualifies, the kernel-only TP8-no-EP rank (I = 288) does not.

What differs from lane B (the part that decides speed, none of it measured):
  * the MMA is the block-scaled FP4 instruction (no ALU decode of weights, no re-decode per row tile);
    x and h travel PACKED (u8 [rows, K/2] + e4m3 scales) instead of lane B's bf16 dequantized rows
  * the tile map is EXPERT-major: the route kernel emits per-expert counts / starts (an extra root
    that scans the keys: O(E * P), E = 48) next to lane B's per-pair counting sort, so fc1 / fc2 launch
    ceil(E * ncols / tile) programs instead of one program per sorted PAIR (M = 192: ~20k programs of
    which ~1k work in lane B -> ~0.9k)
  * above `ROUTE_KERNEL_MAX_M` the route is torch ops (argsort of key * pairs + pair, scatter_add of
    the counts, cumsum: sync-free, capturable) because lane B's O(P^2) pair loop is a prefill non-starter
    (P = 49152 at 8192 tokens) — the prefill-range arm; the decode / verify arm keeps the one-launch
    kernel route
  * activation scales are LINEAR (we write them), weight scales are read in place from the swizzled
    layout as int32 words (nvfp4_gemm_fp4mma.py explains the word gather)

Plans (the seam: `serving_op` wraps lane B's op and routes per M, `served_ms` is its table):
  fp4mma4   route_quant (one launch) , fc1 , fc2 , combine          M <= ROUTE_KERNEL_MAX_M
  fp4mma4t  torch route + quant      , fc1 , fc2 , combine          prefill range
A plan serves bucket(M) only when EVERY kernel of it has a shipped config for that bucket; any other
M calls the wrapped op (lane B's Helion plan / oxide / FlashInfer), exactly as before.

Capture safety (docs/dsv41-fp4.md section 7): no atomics / barriers / semaphores / counters; every
scratch is a torch.empty inside the bodies' host code (graph pool under capture); the route's
counts / starts are rewritten completely by every launch; nothing is created on first use but the
Triton compile, which `served_ms` + the lane's warm-up do before capture.

Env knob SUFFIX_HELION_NVFP4_MOE_FP4MMA = 0 (default) | 1; STRICT=1 makes a layer whose plan has a
gap an error (default: the gap stays on lane B's op). SUFFIX_FP4MMA_MOE_SCALE_READ = words (default) | bytes
picks how fc1 / fc2 read the scale buffers (see SCALE_READ below): `bytes` is the fallback for the case that
the P0 probe shows Triton decomposing the `words` unpack; configs are tuned per mode.
"""
from __future__ import annotations

import functools
import os

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU / no Helion: plans, cases, references still import
    helion = hl = None

ENV = "SUFFIX_HELION_NVFP4_MOE_FP4MMA"
STRICT_ENV = "SUFFIX_HELION_NVFP4_MOE_FP4MMA_STRICT"
MARKER = "[suffix dsv41-moe fp4mma]"

ROUTE_QUANT, QUANT, FC1, FC2, COMBINE = (
    "dsv41_fp4mma_route_quant", "dsv41_fp4mma_quant", "dsv41_fp4mma_fc1", "dsv41_fp4mma_fc2",
    "dsv41_fp4mma_combine")
PLANS = {"fp4mma4": (ROUTE_QUANT, FC1, FC2, COMBINE), "fp4mma4t": (QUANT, FC1, FC2, COMBINE)}
TRITON_NAMES = {k: "_helion_" + k for k in (ROUTE_QUANT, QUANT, FC1, FC2, COMBINE)}
# decode / verify buckets (lane B's CLAMP_BUCKETS) + the prefill-range ones (a call is chunked to <= chunk_size)
BUCKETS = (1, 6, 8, 16, 32, 64, 128, 192, 256, 512, 1024, 2048, 4096)
ROUTE_KERNEL_MAX_M = 512  # above: torch route (lane B's pair loop is O(P^2))
# y is [P+1, H] f32 per call (1 GB at M = 8192, topk 6, H 5120). A call whose y would exceed Y_MB runs in
# token chunks (rows are independent), each chunk its own bucket / plan. The price is weight traffic: every
# chunk re-reads the touched experts (~955 MB per layer at V4.1 EP8 once every expert has rows), so the
# default 512 MB budget = 4096-token chunks (2 passes at 8192; an ESTIMATE of 2x the one-pass 0.66 ms weight
# floor), not 1024 (8 passes). CHUNK_M pins the size (tests); SUFFIX_FP4MMA_MOE_Y_MB sets the budget.
Y_MB_ENV = "SUFFIX_FP4MMA_MOE_Y_MB"
CHUNK_M = None
NO_CLAMP = 3.0e38

# How fc1 / fc2 read the scale buffers (the one open Triton question, docs/dsv41-fp4.md section 5), a process-wide
# choice read at trace time (module constant, like the tile bounds below):
#   words (default)  int32 words: one aligned 4-byte gather per (row, k64 step) holds the 4 scales of that step;
#                    unpacked in registers (hl.join + reshape + bitcast to e4m3). 4-byte loads, cp.async-able.
#   bytes            e4m3 byte gathers: the scale tile IS the load (the pattern vLLM#38338 lowers natively), no
#                    join/reshape; 1-byte loads. The fallback if P0 shows `words` is decomposed by Triton.
# Configs are tuned per mode: ship the ones tuned in the mode the pool runs.
SCALE_ENV = "SUFFIX_FP4MMA_MOE_SCALE_READ"

# search-space bounds (module constants so the eager tests can force many small tiles; the eager
# mode of register_block_size returns the MAX). K tiles are in packed BYTES, >= 32 = one k64 step.
BROW_MIN, BROW_MAX = 16, 64
BCOL_MIN, BCOL_MAX = 2, 8  # fc1: 16-column blocks of I per program (x16 = 32..128 columns, up AND gate)
BH_MIN, BH_MAX = 32, 256  # fc2: columns of H per program
BK_MIN, BK_MAX = 32, 256
BP_MIN, BP_MAX = 16, 64  # route pair tiles
BQ_MIN, BQ_MAX = 16, 128
BX_ROWS_MAX, BX_BLKS_MAX = 8, 64  # quant tiles: tokens x 16-element blocks


def _scale_read() -> str:
    raw = os.environ.get(SCALE_ENV, "words").strip() or "words"
    if raw not in ("words", "bytes"):
        raise RuntimeError(f"{MARKER} {SCALE_ENV}={raw!r}: expected words or bytes")
    return raw


SCALE_READ = _scale_read()


def enabled() -> bool:
    raw = os.environ.get(ENV, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise RuntimeError(f"{MARKER} {ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def strict() -> bool:
    raw = os.environ.get(STRICT_ENV, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise RuntimeError(f"{MARKER} {STRICT_ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def _f32(v: float) -> float:
    import numpy as np
    return float(np.float32(v))


def _div6(g: float) -> float:
    import numpy as np
    return float(np.float32(g) / np.float32(6.0))


def eligible(hdim: int, idim: int, e_count: int) -> str | None:
    """Why a layer cannot use the FP4 MMA bodies, or None (a k64 step is one scale word)."""
    if e_count < 1 or hdim % 64 or idim % 64 or idim < 64:
        return (f"shape E={e_count} H={hdim} I={idim}: needs H % 64 == 0 and I % 64 == 0 "
                "(lane B's dequant bodies take I % 16)")
    return None


# ---------------------------------------------------------------------------
# Helion bodies (module level, undecorated, literals only). Expert-major tile map:
# counts[e] rows start at pair_sorted[starts[e]], local experts ascending, pairs ascending.
# ---------------------------------------------------------------------------
def dsv41_fp4mma_route_quant(x: torch.Tensor, topk_ids: torch.Tensor, g: float, g6: float,
                             e_count: int, id_base: int):
    """(xq u8 [M, H/2], xs e4m3 [M, H/16] LINEAR, pair_sorted i32 [P+1], counts i32 [E], starts
    i32 [E]). Three roots in one launch: (1) lane B's counting sort of the pairs (key = local expert,
    off-rank = E sorts last; sorted position = #{q: (key_q, q) < (key_p, p)}), (2) per-expert row
    counts / first positions (a scan of the keys, one program per expert), (3) the NVFP4 quant of x
    (spec rounding: sf = e4m3(amax16 * g/6), q = e2m1_rne(x * g/sf), g/sf as an exact f64 divide)."""
    m, hdim = x.size()
    pairs = topk_ids.numel()
    ids = topk_ids.view(-1)
    x3 = x.view(m, hdim // 16, 16)
    xq = torch.empty([m, hdim // 2], dtype=torch.uint8, device=x.device)
    xq3 = xq.view(m, hdim // 16, 8)
    xs = torch.empty([m, hdim // 16], dtype=torch.float8_e4m3fn, device=x.device)
    pair_sorted = torch.empty([pairs + 1], dtype=torch.int32, device=x.device)
    counts = torch.empty([e_count], dtype=torch.int32, device=x.device)
    starts = torch.empty([e_count], dtype=torch.int32, device=x.device)
    bp = hl.register_block_size(BP_MIN, BP_MAX)
    bq = hl.register_block_size(BQ_MIN, BQ_MAX)
    bm = hl.register_block_size(1, BX_ROWS_MAX)
    bb = hl.register_block_size(1, BX_BLKS_MAX)
    for tile_p in hl.tile(pairs, block_size=bp):
        kp = ids[tile_p].to(torch.int32) - id_base
        kp = torch.where((kp >= 0) & (kp < e_count), kp, e_count)
        pos = hl.zeros([tile_p], dtype=torch.int32)
        for tile_q in hl.tile(pairs, block_size=bq):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            kq = torch.where(tile_q.index < pairs, kq, e_count + 1)  # tail lanes rank nothing
            eq = kq[None, :] == kp[:, None]
            earlier = eq & (tile_q.index[None, :] < tile_p.index[:, None])
            less = (kq[None, :] < kp[:, None]) | earlier
            pos = pos + torch.sum(less.to(torch.int32), dim=1).to(torch.int32)
        dst = torch.where(tile_p.index < pairs, pos, pairs)  # tail lanes -> trash slot P
        pair_sorted[dst] = tile_p.index
    for tile_e in hl.tile(e_count, block_size=1):
        e = tile_e.begin
        cnt = hl.zeros([tile_e], dtype=torch.int32)
        bef = hl.zeros([tile_e], dtype=torch.int32)
        for tile_q in hl.tile(pairs, block_size=bq):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            kq = torch.where(tile_q.index < pairs, kq, e_count + 1)
            cnt = cnt + torch.sum((kq == e).to(torch.int32))
            bef = bef + torch.sum((kq < e).to(torch.int32))
        counts[tile_e] = cnt
        starts[tile_e] = bef
    for tile_m, tile_b in hl.tile([m, hdim // 16], block_size=[bm, bb]):
        v = x3[tile_m, tile_b, :].to(torch.float32)
        amax = torch.amax(torch.abs(v), dim=-1)
        s8 = torch.clamp(amax * g6, max=448.0).to(torch.float8_e4m3fn)
        sf = s8.to(torch.float32)
        inv = torch.where(sf == 0.0, 0.0,
                          (g / torch.where(sf == 0.0, 1.0, sf).to(torch.float64)).to(torch.float32))
        sv = v * inv[:, :, None]
        a = torch.abs(sv)
        code = ((a > 0.25).to(torch.int32) + (a >= 0.75).to(torch.int32)
                + (a > 1.25).to(torch.int32) + (a >= 1.75).to(torch.int32)
                + (a > 2.5).to(torch.int32) + (a >= 3.5).to(torch.int32)
                + (a > 5.0).to(torch.int32)) | ((sv < 0.0).to(torch.int32) << 3)
        lo, hi = hl.split(code.reshape(code.size(0), code.size(1), 8, 2))
        xq3[tile_m, tile_b, :] = (lo | (hi << 4)).to(torch.uint8)
        xs[tile_m, tile_b] = s8
    return xq, xs, pair_sorted, counts, starts


def dsv41_fp4mma_quant(x: torch.Tensor, g: float, g6: float):
    """(xq u8 [M, H/2], xs e4m3 [M, H/16] linear): the quant root of the route kernel alone (the
    prefill-range plan routes with torch ops)."""
    m, hdim = x.size()
    x3 = x.view(m, hdim // 16, 16)
    xq = torch.empty([m, hdim // 2], dtype=torch.uint8, device=x.device)
    xq3 = xq.view(m, hdim // 16, 8)
    xs = torch.empty([m, hdim // 16], dtype=torch.float8_e4m3fn, device=x.device)
    bm = hl.register_block_size(1, BX_ROWS_MAX)
    bb = hl.register_block_size(1, BX_BLKS_MAX)
    for tile_m, tile_b in hl.tile([m, hdim // 16], block_size=[bm, bb]):
        v = x3[tile_m, tile_b, :].to(torch.float32)
        amax = torch.amax(torch.abs(v), dim=-1)
        s8 = torch.clamp(amax * g6, max=448.0).to(torch.float8_e4m3fn)
        sf = s8.to(torch.float32)
        inv = torch.where(sf == 0.0, 0.0,
                          (g / torch.where(sf == 0.0, 1.0, sf).to(torch.float64)).to(torch.float32))
        sv = v * inv[:, :, None]
        a = torch.abs(sv)
        code = ((a > 0.25).to(torch.int32) + (a >= 0.75).to(torch.int32)
                + (a > 1.25).to(torch.int32) + (a >= 1.75).to(torch.int32)
                + (a > 2.5).to(torch.int32) + (a >= 3.5).to(torch.int32)
                + (a > 5.0).to(torch.int32)) | ((sv < 0.0).to(torch.int32) << 3)
        lo, hi = hl.split(code.reshape(code.size(0), code.size(1), 8, 2))
        xq3[tile_m, tile_b, :] = (lo | (hi << 4)).to(torch.uint8)
        xs[tile_m, tile_b] = s8
    return xq, xs


def dsv41_fp4mma_fc1(xq: torch.Tensor, xs: torch.Tensor, pair_sorted: torch.Tensor,
                     counts: torch.Tensor, starts: torch.Tensor, w13: torch.Tensor,
                     w13s: torch.Tensor, g1: torch.Tensor, a2g: float, a2g6: float, act: int,
                     topk: int, limit: float):
    """(hq u8 [P+1, I/2], hs e4m3 [P+1, I/16] linear): per routed pair the NVFP4 requant (a2 gscale)
    of act(min(g1*gate, limit)) * clamp(g1*up, -limit, limit); rows of off-rank pairs unwritten, row P =
    trash. xs / w13s are int32 WORDS: xs the linear activation scales (row stride H/64 words), w13s
    the swizzled per-expert weight scales (the whole [E, rows128, H/16] buffer, expert stride
    round128(2I) * H/64 words). Grid (expert, blocks of 16 I-columns); up and gate dots share A."""
    e_count = hl.specialize(w13.size(0))
    two_i = hl.specialize(w13.size(1))
    kh = hl.specialize(w13.size(2))
    idim = two_i // 2
    nb = idim // 16
    nq = kh // 32
    rw = nq * 128
    ew = (two_i + 127) // 128 * 128 * nq
    pairs = pair_sorted.size(0) - 1
    wf = w13.view(e_count * two_i, kh)
    hq = torch.empty([pairs + 1, idim // 2], dtype=torch.uint8, device=w13.device)
    hs = torch.empty([pairs + 1, nb], dtype=torch.float8_e4m3fn, device=w13.device)
    hq3 = hq.view(pairs + 1, nb, 8)
    brow = hl.register_block_size(BROW_MIN, BROW_MAX)
    bcol = hl.register_block_size(BCOL_MIN, min(BCOL_MAX, nb))
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_e, tile_b in hl.tile([e_count, nb], block_size=[1, bcol]):
        e = tile_e.begin
        cnt = counts[e]
        s0 = starts[e]
        alpha = g1[e]
        colf = torch.clamp(tile_b.index[:, None] * 16 + hl.arange(16)[None, :], max=idim - 1)
        colf = colf.reshape(-1)
        ru = e * two_i + colf
        rg = ru + idim
        gcol = colf + idim
        if SCALE_READ == "words":
            wu = e * ew + (colf >> 7) * rw + (colf & 31) * 4 + ((colf >> 5) & 3)
            wg = e * ew + (gcol >> 7) * rw + (gcol & 31) * 4 + ((gcol >> 5) & 3)
        else:  # byte offset of (row, kb = 0): the word index * 4 with kb % 4 added per column below
            wu = e * ew * 4 + (colf >> 7) * (nq * 512) + (colf & 31) * 16 + ((colf >> 5) & 3) * 4
            wg = e * ew * 4 + (gcol >> 7) * (nq * 512) + (gcol & 31) * 16 + ((gcol >> 5) & 3) * 4
        for tile_r in hl.tile(0, cnt, block_size=brow):
            pair = pair_sorted[torch.clamp(s0 + tile_r.index, max=pairs)]
            tok = torch.clamp(pair, max=pairs - 1) // topk
            acc_u = hl.zeros([tile_r, tile_b, 16], dtype=torch.float32)
            acc_u = acc_u.reshape(acc_u.size(0), -1)
            acc_g = hl.zeros([tile_r, tile_b, 16], dtype=torch.float32)
            acc_g = acc_g.reshape(acc_g.size(0), -1)
            for tile_k in hl.tile(kh, block_size=bk):
                if SCALE_READ == "words":
                    q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5, max=nq - 1)
                    wa = xs[tok[:, None] * nq + q[None, :]]
                    sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
                    sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                    wbu = w13s[wu[:, None] + q[None, :] * 128]
                    su = hl.join(hl.join(wbu & 255, (wbu >> 16) & 255), hl.join((wbu >> 8) & 255, (wbu >> 24) & 255))
                    su = su.reshape(su.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                    wbg = w13s[wg[:, None] + q[None, :] * 128]
                    sg = hl.join(hl.join(wbg & 255, (wbg >> 16) & 255), hl.join((wbg >> 8) & 255, (wbg >> 24) & 255))
                    sg = sg.reshape(sg.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                else:
                    kb = torch.clamp(torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3, max=nq * 4 - 1)
                    sa = xs[tok[:, None] * (nq * 4) + kb[None, :]]
                    su = w13s[wu[:, None] + (kb >> 2)[None, :] * 512 + (kb & 3)[None, :]]
                    sg = w13s[wg[:, None] + (kb >> 2)[None, :] * 512 + (kb & 3)[None, :]]
                a = xq[tok, tile_k]
                acc_u = hl.dot_scaled(a, sa, "e2m1", wf[ru, tile_k].T, su, "e2m1", acc=acc_u)
                acc_g = hl.dot_scaled(a, sa, "e2m1", wf[rg, tile_k].T, sg, "e2m1", acc=acc_g)
            gate = torch.clamp(alpha * acc_g, max=limit)
            up = torch.clamp(alpha * acc_u, min=-limit, max=limit)
            # act: silu (0) | gelu_tanh (1) = x * sigmoid(2u) (oxide `act`); exact blend
            z = (1.5957691216057308 * (gate + 0.044715 * gate * gate * gate)) * act \
                + gate * (1 - act)
            hv = gate * torch.sigmoid(z) * up
            hv = hv.reshape(hv.size(0), -1, 16)
            amax = torch.amax(torch.abs(hv), dim=-1)
            sfh = torch.clamp(amax * a2g6, max=448.0).to(torch.float8_e4m3fn)
            sff = sfh.to(torch.float32)
            inv = torch.where(sff == 0.0, 0.0,
                              (a2g / torch.where(sff == 0.0, 1.0, sff).to(torch.float64)).to(torch.float32))
            sv = hv * inv[:, :, None]
            av = torch.abs(sv)
            code = ((av > 0.25).to(torch.int32) + (av >= 0.75).to(torch.int32)
                    + (av > 1.25).to(torch.int32) + (av >= 1.75).to(torch.int32)
                    + (av > 2.5).to(torch.int32) + (av >= 3.5).to(torch.int32)
                    + (av > 5.0).to(torch.int32)) | ((sv < 0.0).to(torch.int32) << 3)
            lo, hi = hl.split(code.reshape(code.size(0), code.size(1), 8, 2))
            dst = torch.where(tile_r.index < cnt, pair, pairs)
            hq3[dst, tile_b, :] = (lo | (hi << 4)).to(torch.uint8)
            hs[dst, tile_b] = sfh
    return hq, hs


def dsv41_fp4mma_fc2(hq: torch.Tensor, hs: torch.Tensor, pair_sorted: torch.Tensor,
                     counts: torch.Tensor, starts: torch.Tensor, topk_w: torch.Tensor,
                     w2: torch.Tensor, w2s: torch.Tensor, g2: torch.Tensor):
    """y f32 [P+1, H] = g2[e] * topk_w[p] * (h[p] . w2[e]) per routed pair, h packed (hq u8
    [P+1, I/2], hs int32 WORDS of its linear e4m3 scales, row stride I/64 words), w2s the swizzled
    per-expert weight scale words; off-rank rows unwritten, row P = trash. Grid (expert, H tile)."""
    e_count = hl.specialize(w2.size(0))
    hdim = hl.specialize(w2.size(1))
    kh = hl.specialize(w2.size(2))
    nq = kh // 32
    rw = nq * 128
    ew = (hdim + 127) // 128 * 128 * nq
    pairs = pair_sorted.size(0) - 1
    wf = w2.view(e_count * hdim, kh)
    tw = topk_w.view(-1)
    y = torch.empty([pairs + 1, hdim], dtype=torch.float32, device=w2.device)
    brow = hl.register_block_size(BROW_MIN, BROW_MAX)
    bh = hl.register_block_size(BH_MIN, BH_MAX)
    bk = hl.register_block_size(BK_MIN, BK_MAX)
    for tile_e, tile_h in hl.tile([e_count, hdim], block_size=[1, bh]):
        e = tile_e.begin
        cnt = counts[e]
        s0 = starts[e]
        alpha = g2[e]
        row = torch.clamp(tile_h.index, max=hdim - 1)
        rows = e * hdim + row
        if SCALE_READ == "words":
            wb0 = e * ew + (row >> 7) * rw + (row & 31) * 4 + ((row >> 5) & 3)
        else:
            wb0 = e * ew * 4 + (row >> 7) * (nq * 512) + (row & 31) * 16 + ((row >> 5) & 3) * 4
        for tile_r in hl.tile(0, cnt, block_size=brow):
            pair = torch.clamp(pair_sorted[torch.clamp(s0 + tile_r.index, max=pairs)], max=pairs - 1)
            acc = hl.zeros([tile_r, tile_h], dtype=torch.float32)
            for tile_k in hl.tile(kh, block_size=bk):
                if SCALE_READ == "words":
                    q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5, max=nq - 1)
                    wa = hs[pair[:, None] * nq + q[None, :]]
                    sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
                    sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                    wb = w2s[wb0[:, None] + q[None, :] * 128]
                    sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
                    sb = sb.reshape(sb.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                else:
                    kb = torch.clamp(torch.amin(tile_k.index.reshape(-1, 8), dim=1) >> 3, max=nq * 4 - 1)
                    sa = hs[pair[:, None] * (nq * 4) + kb[None, :]]
                    sb = w2s[wb0[:, None] + (kb >> 2)[None, :] * 512 + (kb & 3)[None, :]]
                acc = hl.dot_scaled(hq[pair, tile_k], sa, "e2m1", wf[rows, tile_k].T, sb, "e2m1",
                                    acc=acc)
            dst = torch.where(tile_r.index < cnt, pair, pairs)
            y[dst, tile_h] = acc * (alpha * tw[pair])[:, None]
    return y


def dsv41_fp4mma_combine(y: torch.Tensor, topk_ids: torch.Tensor, id_base: int, e_count: int,
                         out: torch.Tensor):
    """out bf16 [M, H] = bf16(sum_k y[t*topk + k]) over local ids, k ascending (lane B's body,
    under this family's own config key so the prefill buckets can be tuned)."""
    m, topk = topk_ids.size()
    hdim = out.size(1)
    ids = topk_ids.view(-1)
    bt = hl.register_block_size(1, 16)
    bh = hl.register_block_size(16, 512)
    for tile_t, tile_h in hl.tile([m, hdim], block_size=[bt, bh]):
        acc = hl.zeros([tile_t, tile_h], dtype=torch.float32)
        tok = torch.clamp(tile_t.index, max=m - 1)
        for k in hl.grid(topk):
            p = tok * topk + k
            loc = ids[p].to(torch.int32) - id_base
            ok = (loc >= 0) & (loc < e_count)
            acc = acc + torch.where(ok[:, None], y[p, tile_h], 0.0)
        out[tile_t, tile_h] = acc.to(torch.bfloat16)
    return out


BODIES = {ROUTE_QUANT: dsv41_fp4mma_route_quant, QUANT: dsv41_fp4mma_quant, FC1: dsv41_fp4mma_fc1,
          FC2: dsv41_fp4mma_fc2, COMBINE: dsv41_fp4mma_combine}


# ---------------------------------------------------------------------------
# torch route for the prefill range (sync-free, capturable) and its test twin
# ---------------------------------------------------------------------------
def route_torch(topk_ids: torch.Tensor, e_count: int, id_base: int):
    """(pair_sorted i32 [P+1], counts i32 [E], starts i32 [E]) with torch ops only: the stable
    sort of (local expert, pair) (off-rank pairs last), per-expert counts by scatter_add (int32,
    exact, deterministic) and the exclusive cumsum. No data-dependent shape, no host sync."""
    pairs = topk_ids.numel()
    dev = topk_ids.device
    loc = topk_ids.reshape(-1).to(torch.int64) - id_base
    key = torch.where((loc >= 0) & (loc < e_count), loc, torch.full_like(loc, e_count))
    order = torch.argsort(key * pairs + torch.arange(pairs, device=dev, dtype=torch.int64))
    hist = torch.zeros(e_count + 1, dtype=torch.int64, device=dev)
    hist.scatter_add_(0, key, torch.ones_like(key))
    first = torch.cumsum(hist, 0) - hist
    pair_sorted = torch.cat([order.to(torch.int32), torch.zeros(1, dtype=torch.int32, device=dev)])
    return pair_sorted, hist[:e_count].to(torch.int32), first[:e_count].to(torch.int32)


def route_ref(topk_ids: torch.Tensor, e_count: int, id_base: int):
    """The spec of the route: same outputs as route_torch / the kernel route, by a plain python
    sort (the independent oracle of the tests)."""
    flat = [int(v) - id_base for v in topk_ids.reshape(-1).tolist()]
    key = [v if 0 <= v < e_count else e_count for v in flat]
    order = sorted(range(len(key)), key=lambda p: (key[p], p))
    counts = [sum(1 for v in key if v == e) for e in range(e_count)]
    starts = [sum(counts[:e]) for e in range(e_count)]
    dev = topk_ids.device
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    return t(order + [0]), t(counts), t(starts)


# ---------------------------------------------------------------------------
# plans: which M runs which plan, the wrapper op (the seam), the plan runner
# ---------------------------------------------------------------------------
def bucket_case(e_count: int, hdim: int, idim: int, topk: int, act: int, m: int,
                limit: float | None = None, e_global: int | None = None) -> dict:
    """The tuned case a serving call of M tokens runs as: lane B's `lane_case` schema (so lab
    and bench rows line up) with this family's bucket list."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    case = lb.lane_case(e_count, hdim, idim, topk, act, m, limit, e_global)
    case["numtokens"] = hk.bucket(m, BUCKETS)
    return case


def plan_name(m: int) -> str:
    return "fp4mma4" if m <= ROUTE_KERNEL_MAX_M else "fp4mma4t"


def missing(case: dict, plan: str) -> list:
    return [k for k in PLANS[plan] if hk.config(k, case) is None]


def chunk_size(topk: int, hdim: int) -> int:
    """Tokens per chunk: CHUNK_M when pinned, else the largest POWER OF TWO whose y buffer
    (tokens * topk * H * 4 bytes) fits SUFFIX_FP4MMA_MOE_Y_MB (default 512), at least 256 (so an
    8192-token prefill chunk is exactly 2 x 4096 at V4.1's topk 6 / H 5120)."""
    if CHUNK_M:
        return int(CHUNK_M)
    raw = os.environ.get(Y_MB_ENV, "512").strip() or "512"
    if not raw.isdigit() or int(raw) < 16:
        raise RuntimeError(f"{MARKER} {Y_MB_ENV}={raw!r}: want an integer MB >= 16")
    fit = int(raw) * (1 << 20) // (topk * hdim * 4)
    return max(256, 1 << (fit.bit_length() - 1))


def chunks(m: int, size: int) -> list:
    """Token spans one call of m tokens runs as: [(0, m)] up to `size`, else `size`-sized spans."""
    return [(0, m)] if m <= size else [(a, min(a + size, m)) for a in range(0, m, size)]


def served_ms(e_count: int, hdim: int, idim: int, topk: int, act: int, max_m: int,
              limit: float | None = None, e_global: int | None = None,
              fail_closed: bool = False) -> dict:
    """{M: (bucket, plan)} for every M in 1..max_m this family serves: every kernel of the plan of
    EACH chunk of M has a shipped config for that chunk's bucket (the value is the first chunk's).
    fail_closed (STRICT=1): a gap is an error naming each (bucket, plan, kernel). The lane warms
    exactly the `warm_ms` of this table before graph capture and K5 asserts their PTX."""
    size = chunk_size(topk, hdim)
    direct, gaps = {}, {}
    for m in range(1, min(max_m, size) + 1):
        case = bucket_case(e_count, hdim, idim, topk, act, m, limit, e_global)
        plan = plan_name(m)
        lack = missing(case, plan)
        if lack:
            gaps.setdefault((case["numtokens"], plan), lack)
        else:
            direct[m] = (case["numtokens"], plan)
    out = {m: direct[m] for m in direct}
    for m in range(size + 1, max_m + 1):
        if all(b - a in direct for a, b in chunks(m, size)):
            first = chunks(m, size)[0]
            out[m] = direct[first[1] - first[0]]
    if gaps and fail_closed:
        raise RuntimeError(
            f"{MARKER} {STRICT_ENV}=1 but no tuned config on {hk.platform()} for E={e_count}/"
            f"{e_global or e_count} H={hdim} I={idim} topk={topk} limit={limit}: "
            + "; ".join(f"M<={b} {p}: {', '.join(k)}" for (b, p), k in sorted(gaps.items()))
            + " (tune them with fp4mma_bench / the autotune lab, ship "
            "configs/helion/<kernel>/<platform>.json)")
    return out


@functools.lru_cache(maxsize=None)
def _kernels(case_key: str, plan: str, untuned_ok: bool = False):
    """The plan's serving kernels for one bucketed case. untuned_ok (the lane's standalone
    oracle / bench only, never serving): a kernel with no shipped config runs Helion's default."""
    import json
    case = json.loads(case_key)
    out = []
    for k in PLANS[plan]:
        if untuned_ok and hk.config(k, case) is None:
            hk.log(f"dsv41-moe-fp4mma UNTUNED {k} {case_key}: Helion default config (bench only)")
            out.append(helion.kernel(BODIES[k], autotune_effort="none", static_shapes=False))
        else:
            out.append(hk.compiled(BODIES[k], k, case))
    return out


def words(t: torch.Tensor) -> torch.Tensor:
    """Flat int32 words of a scale buffer (e4m3 / uint8 / int32): metadata only."""
    from suffix_hybrid.helion_kernels.nvfp4_gemm_fp4mma import as_words
    return as_words(t)


def sf_arg(t: torch.Tensor) -> torch.Tensor:
    """A scale buffer as the bodies read it in the current SCALE_READ mode: flat int32 words or flat e4m3
    bytes (the same bytes, metadata only)."""
    if SCALE_READ == "words":
        return words(t)
    if not t.is_contiguous():
        raise RuntimeError(f"{MARKER} scale buffer must be contiguous, got strides {tuple(t.stride())}")
    return t.reshape(-1).view(torch.uint8).view(torch.float8_e4m3fn)


def run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out, a1_gscale,
             a2_gscale, act, id_base, limit=None):
    """One MoE forward on the family's kernels `kerns` (PLANS[plan] order) -> out. The scale
    buffers may be the lane's [E, rows, cols] e4m3 tensors (viewed as words here)."""
    e_count = w13.shape[0]
    if plan == "fp4mma4":
        xq, xs, ps, counts, starts = kerns[0](x, topk_ids, _f32(a1_gscale), _div6(a1_gscale),
                                              e_count, id_base)
    else:
        ps, counts, starts = route_torch(topk_ids, e_count, id_base)
        xq, xs = kerns[0](x, _f32(a1_gscale), _div6(a1_gscale))
    hq, hs = kerns[1](xq, sf_arg(xs), ps, counts, starts, w13, sf_arg(w13_sf), g1, _f32(a2_gscale),
                      _div6(a2_gscale), act, topk_ids.shape[1],
                      NO_CLAMP if limit is None else float(limit))
    y = kerns[2](hq, sf_arg(hs), ps, counts, starts, topk_w, w2, sf_arg(w2_sf), g2)
    return kerns[3](y, topk_ids, id_base, e_count, out)


def serving_op(base_op, limit: float | None = None, e_global: int | None = None,
               served: dict | None = None, untuned_ok: bool = False):
    """THE SEAM. Wraps the lane's host op (lane B's `serving_op(...)`, or the oxide op: any
    callable with `_native.nvfp4_moe_cuda`'s signature) into one with the same signature that runs
    this family's plan for every M in `served` ({M: (bucket, plan)}, from `served_ms`) and calls
    `base_op` for every other M. Calls above CHUNK_M tokens run in CHUNK_M-token chunks (y is
    [P+1, H] f32). ws_* / mode / tune / stream_ptr are the oxide op's: unused here (Helion runs on
    torch's current stream = the lane's). untuned_ok: standalone oracle / bench only."""
    served = served if served is not None else {}

    def op(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, ws_aq, ws_asf, ws_inter, ws_hq,
           ws_hsf, ws_y, ws_route, out, a1_gscale, a2_gscale, act, id_base, mode, tune, stream_ptr):
        m = x.shape[0]
        spans = chunks(m, chunk_size(topk_ids.shape[1], x.shape[1]))
        if not untuned_ok and not all((b - a) in served for a, b in spans):
            return base_op(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, ws_aq, ws_asf,
                           ws_inter, ws_hq, ws_hsf, ws_y, ws_route, out, a1_gscale, a2_gscale,
                           act, id_base, mode, tune, stream_ptr)
        for a, b in spans:
            n = b - a
            plan = plan_name(n)
            case = bucket_case(w13.shape[0], x.shape[1], w13.shape[1] // 2, topk_ids.shape[1], act,
                               n, limit, e_global)
            kerns = _kernels(hk.case_key(case), plan, untuned_ok)
            run_plan(kerns, plan, x[a:b], topk_ids[a:b], topk_w[a:b], w13, w13_sf, g1, w2, w2_sf,
                     g2, out[a:b], a1_gscale, a2_gscale, act, id_base, limit)
        return out

    op.backend = "fp4mma"
    op.served = served
    return op


def warm_ms(served: dict, size: int | None = None) -> list:
    """Ms the lane must run once before graph capture. Triton specializes int arguments on == 1
    and on divisibility by 16 (M, and P = M * topk, 6M % 16 == 0 <=> M % 8 == 0 at topk 6): per
    (bucket, plan) the first and last served M of each class (M == 1, M % 16 == 0, M % 8 == 0,
    other); only the direct Ms (<= the layer's chunk size `size`; default: the longest unbroken
    1..m run of the table), chunked calls reuse them."""
    size = size or next((m for m in range(len(served) + 1, 0, -1) if all(k in served for k in range(1, m + 1))), 0)
    by = {}
    for m, key in served.items():
        if m <= size:
            by.setdefault((key, m == 1, m % 16 == 0, m % 8 == 0), []).append(m)
    out = set()
    for ms in by.values():
        out |= {min(ms), max(ms)}
    return sorted(out)


NATIVE_KERNELS = (FC1, FC2)  # the kernels K5 asserts carry the native FP4 MMA


def verify_served(served: dict, allow_local: bool = True) -> list:
    """K5 for a layer's served table: after the lane warmed `warm_ms(served)`, every fc1 / fc2
    specialization compiled under the recorder must carry kind::mxf4nvf4 for sm_120a (else
    Fp4mmaError, fail closed). No-op for an empty table."""
    if not served:
        return []
    from suffix_hybrid.kernels import fp4mma_probe
    return fp4mma_probe.assert_native([TRITON_NAMES[k] for k in NATIVE_KERNELS],
                                      allow_local=allow_local)


def prepare_layer(base_op, hdim: int, idim: int, e_count: int, topk: int, act: int, max_m: int,
                  limit: float | None = None, e_global: int | None = None):
    """What the lane calls once per layer when SUFFIX_HELION_NVFP4_MOE_FP4MMA=1: (op, served).
    Fails closed on the stack (SM12x, Triton >= 3.6, Helion), on an ineligible shape and (STRICT=1)
    on a gap; installs the K5 recorder BEFORE any kernel of the family compiles. An empty `served`
    (no shipped config anywhere) returns `base_op` unchanged: lane B's behaviour exactly."""
    if not enabled():
        raise RuntimeError(f"{MARKER} prepare_layer called with {ENV} unset/0")
    why = eligible(hdim, idim, e_count)
    if why:
        raise RuntimeError(f"{MARKER} {ENV}=1 but the layer is not eligible: {why}")
    from suffix_hybrid.kernels import fp4mma_probe
    fp4mma_probe.require_native_stack()
    fp4mma_probe.install_recorder()
    served = served_ms(e_count, hdim, idim, topk, act, max_m, limit, e_global, strict())
    if not served:
        hk.log(f"dsv41-moe-fp4mma: no shipped config on {hk.platform()} — the layer stays on its "
               "incumbent op")
        return base_op, served
    hk.log(f"dsv41-moe-fp4mma: serves {len(served)}/{max_m} Ms {sorted(set(served.values()))} on the native FP4 MMA "
           f"(scale read {SCALE_READ}, E={e_count} H={hdim} I={idim}), {max_m - len(served)} Ms stay on the wrapped op")
    return serving_op(base_op, limit, e_global, served), served


# ---------------------------------------------------------------------------
# references (pure torch; the independent spec lives in kernels/nvfp4_moe.py: quant_codes, moe_ref)
# ---------------------------------------------------------------------------
def words_to_e4m3(w: torch.Tensor, shape) -> torch.Tensor:
    """int32 words -> float8_e4m3fn tensor of `shape` (the same bytes)."""
    return w.reshape(-1).view(torch.uint8).view(torch.float8_e4m3fn).reshape(shape)


def seg_from(ps: torch.Tensor, counts: torch.Tensor, starts: torch.Tensor):
    """Lane B's per-position route arrays (seg, seg_e) from the expert-major (counts, starts), so
    lane B's stage references (`_h_ref`, `_y_ref`) judge this family's stages unchanged."""
    pairs = ps.numel() - 1
    e_count = counts.numel()
    seg = torch.zeros(pairs + 1, dtype=torch.int32, device=ps.device)
    seg_e = torch.full((pairs + 1,), e_count, dtype=torch.int32, device=ps.device)
    seg_e[pairs] = 0  # the trash slot, like lane B's route_ref pad
    for e in range(e_count):
        c, s0 = int(counts[e]), int(starts[e])
        if c:
            seg[s0] = c
            seg_e[s0:s0 + c] = e
    return seg, seg_e


def route_quant_ref(x, topk_ids, g, g6, e_count, id_base):
    """Spec of dsv41_fp4mma_route_quant: (xq, xs e4m3, pair_sorted, counts, starts)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    q, sfb = nm.quant_codes(x.float(), g)
    ps, counts, starts = route_ref(topk_ids, e_count, id_base)
    return q, sfb.view(torch.float8_e4m3fn), ps, counts, starts


def quant_ref(x, g, g6):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    q, sfb = nm.quant_codes(x.float(), g)
    return q, sfb.view(torch.float8_e4m3fn)


def _frac(a, b) -> float:
    return float((a != b).float().mean()) if a.numel() else 0.0


def check_route_quant(out, ref):
    """Routing exact (integer math); the x quant like lane B's xq stage (rare rounding ties:
    STAGE_TOL['xq'] of the bytes may differ)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ref[2].numel() - 1
    exact = (torch.equal(out[2][:pairs].int(), ref[2][:pairs]) and torch.equal(out[3].int(), ref[3])
             and torch.equal(out[4].int(), ref[4]))
    fq = _frac(out[0], ref[0])
    fs = _frac(out[1].view(torch.uint8), ref[1].view(torch.uint8))
    return (exact and fq <= nm.STAGE_TOL["xq"] and fs <= nm.STAGE_TOL["xq"],
            f"route exact={exact} xq bytes differing={fq:.1e} scale bytes differing={fs:.1e}")


def check_quant(out, ref):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    fq = _frac(out[0], ref[0])
    fs = _frac(out[1].view(torch.uint8), ref[1].view(torch.uint8))
    return (fq <= nm.STAGE_TOL["xq"] and fs <= nm.STAGE_TOL["xq"],
            f"xq bytes differing={fq:.1e} scale bytes differing={fs:.1e}")


def _layer_p(w13, w13s, g1, a2g, act, limit, w2=None, w2s=None, g2=None):
    """A `p` dict for kernels/nvfp4_moe's references from kernel arguments (scale WORDS back to
    the per-expert byte rows `_expert_w` reads)."""
    e = w13.shape[0]
    p = dict(w13=w13, w13_sf=w13s.reshape(-1).view(torch.uint8).reshape(e, -1), g1=g1, a2g=a2g,
             act=act, limit=None if limit is None or limit >= NO_CLAMP else limit)
    if w2 is not None:
        p.update(w2=w2, w2_sf=w2s.reshape(-1).view(torch.uint8).reshape(e, -1), g2=g2)
    return p


def fc1_ref(xq, xs, ps, counts, starts, w13, w13s, g1, a2g, a2g6, act, topk, limit):
    """(h bf16 [P+1, I] = spec requant (a2 gscale) of the f64 clamped fc1 output, valid rows)
    for the packed activation (xq, xs words); lane B's `_h_ref` through (seg, seg_e)."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    from suffix_hybrid.kernels import nvfp4_moe as nm
    m, kh = xq.shape
    xsf = words_to_e4m3(xs, (m, kh // 8)).float()
    xdq = nm.dequant(xq, xsf).bfloat16()
    seg, seg_e = seg_from(ps, counts, starts)
    return lb._h_ref(_layer_p(w13, w13s, g1, a2g, act, limit), xdq, ps, seg, seg_e, topk)


def check_fc1(out, ref):
    """out = (hq, hs): the requantized h as dequantized values vs the spec, on the rows a consumer
    reads (lane B's tolerance: STAGE_TOL['hq'] of the values may differ)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    hq, hs = out
    want, valid = ref
    pairs1, idim = want.shape
    got = nm.dequant(hq.reshape(pairs1, -1), hs.reshape(pairs1, -1).float()).bfloat16()
    f = _frac(got[valid], want[valid])
    ok = bool(torch.isfinite(got[valid].float()).all()) and f <= nm.STAGE_TOL["hq"]
    return ok, f"hq differing={f:.1e} rows={int(valid.sum())}"


def fc2_ref(hq, hs, ps, counts, starts, topk_w, w2, w2s, g2):
    """(y f64 [P+1, H] = g2 * tw * (h . w2), valid rows) for the packed h (hq, hs words)."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs1, kh = hq.shape
    hsf = words_to_e4m3(hs, (pairs1, kh // 8)).float()
    hd = nm.dequant(hq, hsf)
    seg, seg_e = seg_from(ps, counts, starts)
    return lb._y_ref(hd, ps, seg, seg_e, topk_w, w2, w2s.reshape(-1).view(torch.uint8).reshape(
        w2.shape[0], -1), g2)


def check_fc2(out, ref):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return lb._check_rows("fc2", rel=True)(out, ref)


# ---------------------------------------------------------------------------
# lab operands + ABI (inference-console api/autotune_kernels.py); cases = lane B's schema
# ---------------------------------------------------------------------------
def pk_inputs(case: dict, seed: int):
    """Everything a stage needs for one lab case: the synthetic layer + routing (lane B's
    generators: balanced / skewed by seed), the spec-quantized packed x, and the expert-major route."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    from suffix_hybrid.kernels import nvfp4_moe as nm
    p, x, ids, tw = lb._problem(case, seed)
    base = lb.ep_base(case)
    e = p["w13"].shape[0]
    xq, xsb = nm.quant_codes(x.float(), p["a1g"])
    ps, counts, starts = route_ref(ids, e, base)
    return dict(p=p, x=x, ids=ids, tw=tw, base=base, xq=xq, xs=sf_arg(xsb), ps=ps, counts=counts,
                starts=starts, w13s=sf_arg(p["w13_sf"]), w2s=sf_arg(p["w2_sf"]))


def make_route_quant(case, seed):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return lb._make_route_quant(case, seed)


def make_quant(case, seed):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    x, _ids, g, g6, _e, _base = lb._make_route_quant(case, seed)
    return x, g, g6


def make_fc1(case, seed):
    d = pk_inputs(case, seed)
    p, lim = d["p"], case.get("limit")
    return (d["xq"], d["xs"], d["ps"], d["counts"], d["starts"], p["w13"], d["w13s"], p["g1"],
            _f32(p["a2g"]), _div6(p["a2g"]), p["act"], d["ids"].shape[1],
            NO_CLAMP if lim is None else float(lim))


def make_fc2(case, seed):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    d = pk_inputs(case, seed)
    p, ids = d["p"], d["ids"]
    hpre = nm.fc1_ref(p, d["x"], nm.to_local(ids, d["base"], p["w13"].shape[0]).int()).float()
    hq, hsb = nm.quant_codes(hpre, p["a2g"])
    pad = lambda t: torch.cat([t, torch.zeros(1, *t.shape[1:], dtype=t.dtype, device=t.device)])
    return (pad(hq), sf_arg(pad(hsb)), d["ps"], d["counts"], d["starts"], d["tw"], p["w2"],
            d["w2s"], p["g2"])


def make_combine(case, seed):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return lb._make_combine(case, seed)


# Helion's autotuner compares every candidate with the default config's output over the WHOLE output
# tensor. Ours are not fully written (h / y rows of off-rank pairs and the trash slot P are
# torch.empty garbage, the trash slot of pair_sorted is a benign race), so a whole-tensor comparison
# rejects candidates at random: `autotune_check` compares only what a consumer reads, masked by the
# args of the latest make() (the lab calls make(case, 0) right before it autotunes).
LAST_ARGS: dict = {}
AUTOTUNE_FLIP = 5e-3  # fraction of requantized values / bytes that may differ between two configs
AUTOTUNE_REL = 2e-3  # rel-L2 between the f32 outputs of two configs (accumulation order)


def remember(name, make):
    def wrapped(case, seed):
        LAST_ARGS[name] = args = make(case, seed)
        return args
    wrapped.__wrapped__ = make
    return wrapped


def autotune_check(name):
    """autotune_baseline_accuracy_check_fn of kernel `name` (raises AssertionError)."""
    def check(actual, expected):
        args = LAST_ARGS.get(name)
        got = actual if isinstance(actual, (tuple, list)) else (actual,)
        want = expected if isinstance(expected, (tuple, list)) else (expected,)
        if len(got) != len(want):
            raise AssertionError(f"{name}: {len(got)} outputs vs {len(want)}")

        def flips(a, b, what):
            f = _frac(a.reshape(-1).view(torch.uint8) if a.dtype != torch.bfloat16 else a.float(),
                      b.reshape(-1).view(torch.uint8) if b.dtype != torch.bfloat16 else b.float())
            if f > AUTOTUNE_FLIP:
                raise AssertionError(f"{name}.{what}: {f:.2e} of the values differ (> {AUTOTUNE_FLIP})")

        def local_rows(ps_i, c_i):
            ps, counts = args[ps_i], args[c_i]
            return ps[:int(counts.sum())].long()  # sorted position < sum(counts) = local pairs

        if name in (ROUTE_QUANT, QUANT):
            if name == ROUTE_QUANT:
                pairs = want[2].numel() - 1
                for i, w in ((2, "pair_sorted"), (3, "counts"), (4, "starts")):
                    a, b = (got[i][:pairs], want[i][:pairs]) if i == 2 else (got[i], want[i])
                    if not torch.equal(a, b):
                        raise AssertionError(f"{name}.{w}: routing differs between configs")
            flips(got[0], want[0], "xq")
            flips(got[1], want[1], "xs")
        elif name == FC1:
            r = local_rows(2, 3)
            flips(got[0][r], want[0][r], "hq")
            flips(got[1][r], want[1][r], "hs")
        elif name == FC2:
            r = local_rows(2, 3)
            a, b = got[0][r].float(), want[0][r].float()
            rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
            if not bool(torch.isfinite(a).all()) or rel > AUTOTUNE_REL:
                raise AssertionError(f"{name}.y: rel {rel:.2e} (> {AUTOTUNE_REL})")
        else:  # combine: every row of out is written
            flips(got[0], want[0], "out")
    return check


def _kernel_entry(name, make, reference, check_fn, stage):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return {"name": name, "body": BODIES[name], "cases": lambda model: cases(model, name),
            "make": remember(name, make), "reference": reference, "check": check_fn,
            "settings": {"ignore_warnings": [helion.exc.TensorOperationInWrapper] if helion else [],
                         "autotune_baseline_accuracy_check_fn": autotune_check(name)},
            "baseline": lb._oxide_baseline,
            "make_baseline": lambda case, seed: lb._oxide_args(case, seed, 1, stage),
            "baseline_name": "vLLM FlashInfer CUTLASS (V4.1: oxide cannot run it), WHOLE layer, "
                             "this EP rank, cold L2; sum a plan's stages to compare; the verdict is "
                             "fp4mma_bench's end-to-end table vs lane B's plans",
            "replaces": []}


def cases(model: dict, kernel: str | None = None) -> list:
    """Lane B's EP-rank layouts x this family's decode buckets (FP4MMA_PREFILL_CASES=1 adds the
    chunked prefill range), only layouts the bodies take (H % 64, I % 64)."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    want = {ROUTE_QUANT: lambda m: m <= ROUTE_KERNEL_MAX_M,
            QUANT: lambda m: m > ROUTE_KERNEL_MAX_M}.get(kernel, lambda m: True)
    top = BUCKETS[-1] if os.environ.get("FP4MMA_PREFILL_CASES") == "1" else 192
    return [{**d, "numtokens": m} for d in lb._layouts(model)
            if eligible(d["hidden"], d["moe_intermediate"], d["experts"]) is None
            for m in BUCKETS if m <= top and want(m)]


def _lab_kernels():
    if helion is None:
        return []
    return [
        _kernel_entry(ROUTE_QUANT, make_route_quant, route_quant_ref, check_route_quant, 0),
        _kernel_entry(QUANT, make_quant, quant_ref, check_quant, 0),
        _kernel_entry(FC1, make_fc1, fc1_ref, check_fc1, 1),
        _kernel_entry(FC2, make_fc2, fc2_ref, check_fc2, 2),
        _kernel_entry(COMBINE, make_combine,
                      lambda y, ids, base, e, out: _ref_combine(y, ids, base, e, out),
                      lambda out, ref: _check_combine(out, ref), 3),
    ]


def _ref_combine(y, ids, base, e, out):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return lb._ref_combine(y, ids, base, e, out)


def _check_combine(out, ref):
    from suffix_hybrid.helion_kernels import nvfp4_moe as lb
    return lb._check_combine(out, ref)


KERNELS = _lab_kernels()


# ---------------------------------------------------------------------------
# capture/replay probes (kernels/fp4mma_probe.py)
# ---------------------------------------------------------------------------
PROBE_CFG = {
    ROUTE_QUANT: dict(block_sizes=[32, 64, 4, 32], num_warps=4, num_stages=1),
    QUANT: dict(block_sizes=[4, 32], num_warps=4, num_stages=1),
    FC1: dict(block_sizes=[16, 4, 128], num_warps=4, num_stages=3),
    FC2: dict(block_sizes=[16, 64, 128], num_warps=4, num_stages=3),
    COMBINE: dict(block_sizes=[4, 256], num_warps=4, num_stages=1),
}
DIST_ROTATION = ("balanced", "skewed", "hot", "dead")


def probes(tier: str) -> list:
    from suffix_hybrid.kernels.fp4mma_probe import Probe

    def p(label, plan, m, e=48, eg=384, h=5120, i=2304, topk=6, limit=10.0, dists=DIST_ROTATION):
        return Probe(label, "moe", dict(plan=plan, m=m, e=e, eg=eg, h=h, i=i, topk=topk, limit=limit,
                                        dists=list(dists)))

    if tier == "tiny":  # CPU tests
        return [p("tiny.fp4mma4.m7", "fp4mma4", 7, 6, 12, 128, 128, 3, 0.8),
                p("tiny.fp4mma4t.m16", "fp4mma4t", 16, 6, 12, 128, 128, 3, 0.8)]
    if tier == "quick":  # the V4.1 EP8 rank: 48 of 384 experts, H 5120, I 2304
        return [p("ep8.fp4mma4.m1", "fp4mma4", 1), p("ep8.fp4mma4.m6", "fp4mma4", 6),
                p("ep8.fp4mma4.m48", "fp4mma4", 48), p("ep8.fp4mma4.m192", "fp4mma4", 192),
                p("ep8.fp4mma4t.m600", "fp4mma4t", 600)]
    return [p(f"ep8.{pl}.m{m}", pl, m) for pl in ("fp4mma4", "fp4mma4t")
            for m in (1, 2, 6, 8, 16, 32, 48, 64, 96, 128, 192, 300, 512, 600, 1024)
            if (pl == "fp4mma4") == (m <= ROUTE_KERNEL_MAX_M)]


def build_probe(pr, dev):
    """Case for a MoE probe: ONE synthetic layer, four distinct inputs (balanced / skewed / hot
    expert loads and a batch whose tokens all route off-rank, x redrawn), the family's kernels at
    explicit configs. K3 = rel_vs_f64 <= REF_TOL on the layer, exact zeros for off-rank tokens."""
    from suffix_hybrid.helion_kernels.nvfp4_gemm_fp4mma import eager_ref  # noqa: F401 (CPU runs)
    from suffix_hybrid.kernels import nvfp4_moe as nm
    from suffix_hybrid.kernels.fp4mma_probe import Case
    sp = pr.spec
    e, eg, h, i, topk, m = (sp[k] for k in ("e", "eg", "h", "i", "topk", "m"))
    limit, plan = sp["limit"], sp["plan"]
    base = (eg // e) // 2 * e
    p = nm.make_problem(e, h, i, "silu", device=dev, seed=3, limit=limit)
    amp = 2688.0 / p["a1g"]
    inputs = []
    for j, dist in enumerate((sp["dists"] * 2)[:4]):
        ids, tw = nm.rand_routing(m, eg, topk, dev, seed=100 + j, base=base, local=e,
                                  dead=True if dist == "dead" else None,
                                  dist="balanced" if dist == "dead" else dist)
        gen = torch.Generator(device="cpu").manual_seed(500 + j)
        x = (torch.randn(m, h, generator=gen) * amp / 4.5 * (4.0 if dist == "hot" else 1.0)).to(
            dev, torch.bfloat16)
        inputs.append({"x": x, "ids": ids, "tw": tw})
    kerns = [helion.kernel(BODIES[k], config=helion.Config(**PROBE_CFG[k]), static_shapes=False,
                           ignore_warnings=[helion.exc.TensorOperationInWrapper]) for k in PLANS[plan]]

    def run(static, out=None):
        o = out if out is not None else torch.empty(m, h, dtype=torch.bfloat16, device=dev)
        o.fill_(float("nan"))
        return run_plan(kerns, plan, static["x"], static["ids"], static["tw"], p["w13"], p["w13_sf"],
                        p["g1"], p["w2"], p["w2_sf"], p["g2"], o, p["a1g"], p["a2g"], p["act"], base,
                        limit)

    def exact(x, out):
        lid = nm.to_local(x["ids"], base, e).int()
        ref = nm.moe_ref(p, x["x"], lid, x["tw"])
        rel = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30))
        dead = (lid < 0).all(1)
        zero = bool((out[dead] == 0).all())
        ok = bool(torch.isfinite(out.float()).all()) and rel <= nm.REF_TOL and zero
        return ok, f"rel_vs_f64={rel:.2e} (REF_TOL {nm.REF_TOL}) nonlocal_tokens={int(dead.sum())} zero={zero}"

    return Case(inputs=inputs, run=run,
                copy_in=lambda st, x: [st[k].copy_(x[k]) for k in st], exact=exact, out_shape=(m, h))
