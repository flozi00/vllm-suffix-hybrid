# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/nvfp4_moe (NVFP4 W4A4 routed-experts decode,
suffix_hybrid/kernels/nvfp4_moe.py): bodies, serving entry, lab KERNELS.

Same stages and layouts as the oxide plans (vLLM FLASHINFER_CUTLASS after
weight processing: w13 uint8 [E, 2I, H/2] = [up; gate], w2 [E, H, I/2],
e4m3 block-16 scales 128x4-swizzled per expert, low nibble = even element,
g1/g2 alphas [E], one activation gscale per GEMM; pair p = token*topk + k):

  nvfp4_moe_route_quant  (= moe_route_quant) one launch, two roots:
      route   counting sort of the pairs by local expert (ascending expert,
              ascending pair: route_par's order) -> pair_sorted / seg / seg_e
              per sorted position; seg[s] = row count where s starts an
              expert's segment, else 0 (no slot compaction, no zero-fill)
      quant x vLLM scaled_fp4_quant math (a1 gscale) -> xdq bf16 [M, H] =
              e2m1 * e4m3 dequantized (exact in bf16; 1/g folded into g1)
  nvfp4_moe_fc1          (= moe_fc1_quant) per (sorted position, 16-col
      blocks of I): that expert's rows, up/gate tiles share the A operand,
      h = act(g1*gate) * g1*up in f32, NVFP4 requant (a2 gscale) in the
      epilogue -> h bf16 [P+1, I] dequantized (row P = trash for masked rows)
  nvfp4_moe_fc2          (= moe_fc2) y[p] = g2[e] * tw[p] * (h[p] . w2[e]) f32
  nvfp4_moe_combine      (= moe_combine) out[t] = bf16(sum_k y[t*topk+k]),
      k ascending over local experts (0 for a token with no local expert)
  nvfp4_moe_fc2_combine  (= moe_fc2_combine) token-major fc2 + combine as one
      GEMV per (token, H tile), k ascending (decode M)

Plans (the lane's tune table picks the mode per M, both backends):
  mode 2 (fused3)          route_quant, fc1, fc2_combine
  mode 0/1 (legacy/front4) route_quant, fc1, fc2, combine

FP4 math: Triton's dot_scaled takes e8m0 group-32 scales only, so NVFP4
(e4m3 group-16) is decoded in registers: nibble -> fp16 bits (e2m1 is a
denormal-exact subset of fp16) * 2^14 * e4m3 scale -> bf16, exact (<= 6
significant bits), and hl.dot accumulates bf16 x bf16 in f32. The packed
K axis is never interleaved: even and odd elements are two dots over the
byte index (A is read through an [.., K/16, 8, 2] view). Products are
exact; only f32 summation order differs from the block-scaled mma, so the
lane's oracle bounds (REF_TOL vs the f64 spec) are the contract.
Not bit-identical to the oxide plans; deterministic run to run.

Serving: `serving_op` has the oxide host op's call signature, routed by
SUFFIX_HELION_NVFP4_MOE (helion_kernels.backend): per M, the plan's kernels
run on Helion when every one has a shipped config for bucket(M) (auto falls
back to the oxide op per M, helion makes a gap a load-time error, see
`served_ms`). Serving never autotunes (helion_kernels.compiled).
"""
from __future__ import annotations

import functools
import importlib
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
except ImportError:  # CPU / no Helion: the module still imports (lab ABI, tests)
    helion = hl = None

KNOB = "SUFFIX_HELION_NVFP4_MOE"
MARKER = "[suffix helion nvfp4-moe]"
# decode M buckets tuned in the lab; bucket(M) = the smallest >= M (the last
# serves above). GLM MTP verify rows: 6 per sequence, up to MAX_M 192.
BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
ACT = {0: "silu", 1: "gelu_tanh"}  # nvfp4_moe.ACT_CODE inverted
NO_CLAMP = 3.0e38  # fc1 `limit` of an unclamped layer (identity on finite values)
# DeepSeek-V4 (DeepseekV4*/V41*: swiglu_limit 10, the lab's model dims do not
# carry it) serves TP8 here: cases() emits both rank layouts of that TP.
DS_LIMIT, DS_TP = 10.0, 8
# DS layers have no oxide fallback (clamp, E 384, I 288): the lane's tune
# table picks fused3 up to M 8, front4 above; the lab tunes each plan's own
# kernels a bucket past that split (6..16 overlap) so the split is measured.
FUSED3_MAX, FRONT4_MIN = 16, 6
# Clamped (Helion-only) layers tune fewer buckets: one config serves its
# whole bucket (dynamic shapes), DSpark verifies 6 tokens per sequence.
CLAMP_BUCKETS = (1, 6, 8, 16, 32, 64, 128, 192)


def buckets_for(limit) -> tuple:
    return BUCKETS if limit is None else CLAMP_BUCKETS
# Search-space bounds (K2; the lab lesson of ds_mla/fp8_moe: unbounded tiles time out
# compiling or spill). Every tunable block size is registered with literal [min, max]
# (powers of two) inside the bodies; tests/test_helion_nvfp4_moe.py reads them back from
# the bound ConfigSpec. Rows per expert segment (nm.expected_load: EP8 allgather dispatch
# gives an expert M/64 rows, ~3 per HIT expert and <= ~10 for the heaviest at M=192; skewed
# puts ~80 on the top expert, hot M): the 16-row minimum is the dot's M floor, mostly
# padding on balanced loads, and the 64-row maximum only matters for the heavy tail (longer
# segments loop). fc1: rows 16..64, 16-col N blocks 1..4
# (16..64 cols: the four dots + reshape scratch overflowed sm_120's ~99 KB smem at
# N 256 x K 128 on gemma), K = 8 * bkb elements per half-dot with bkb 2..8; largest live
# set = 4 bf16 [64, 64] weight tiles + 2 f32 [64, 64] accumulators. fc2: rows 16..64,
# H tile 16..256, bkb 2..8. fc2_combine: H tile 16..128, K blocks 1..16. combine: 1..16
# tokens x 16..512 cols. route_quant: 16..64 x 16..128 pair tiles (O(P^2) compare),
# 1..8 tokens x 1..64 16-col blocks. An over-budget config fails to compile and the
# autotuner drops it; none ships unvalidated.
ROUTE_QUANT, FC1, FC2, COMBINE, FC2_COMBINE = (
    "nvfp4_moe_route_quant", "nvfp4_moe_fc1", "nvfp4_moe_fc2", "nvfp4_moe_combine",
    "nvfp4_moe_fc2_combine")
PLANS = {"front4": (ROUTE_QUANT, FC1, FC2, COMBINE), "fused3": (ROUTE_QUANT, FC1, FC2_COMBINE)}


# ---------------------------------------------------------------------------
# Helion bodies (module-level, undecorated: the lab and `compiled` wrap them).
# Literals only inside: the bodies are traced.
# ---------------------------------------------------------------------------
def nvfp4_moe_route_quant(x: torch.Tensor, topk_ids: torch.Tensor, g: float, g6: float,
                          e_count: int, id_base: int):
    """(xdq bf16 [M, H], pair_sorted, seg, seg_e int32 [P+1]). Key of pair p
    = topk_ids[p] - id_base when in [0, E) else E (off-rank, sorts last);
    its sorted position = #{q: (key_q, q) < (key_p, p)}. g6 = f32(g)/f32(6)
    computed IEEE on the host (the kernel's g / 6)."""
    m, hdim = x.size()
    pairs = topk_ids.numel()
    ids = topk_ids.view(-1)
    x3 = x.view(m, hdim // 16, 16)
    xdq = torch.empty([m, hdim], dtype=torch.bfloat16, device=x.device)
    xq3 = xdq.view(m, hdim // 16, 16)
    pair_sorted = torch.empty([pairs + 1], dtype=torch.int32, device=x.device)
    seg = torch.empty([pairs + 1], dtype=torch.int32, device=x.device)
    seg_e = torch.empty([pairs + 1], dtype=torch.int32, device=x.device)
    bp = hl.register_block_size(16, 64)
    bq = hl.register_block_size(16, 128)
    bm = hl.register_block_size(1, 8)
    bb = hl.register_block_size(1, 64)
    for tile_p in hl.tile(pairs, block_size=bp):
        kp = ids[tile_p].to(torch.int32) - id_base
        kp = torch.where((kp >= 0) & (kp < e_count), kp, e_count)
        pos = hl.zeros([tile_p], dtype=torch.int32)
        before = hl.zeros([tile_p], dtype=torch.int32)
        same = hl.zeros([tile_p], dtype=torch.int32)
        for tile_q in hl.tile(pairs, block_size=bq):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            kq = torch.where(tile_q.index < pairs, kq, e_count + 1)  # tail lanes rank nothing
            eq = kq[None, :] == kp[:, None]
            earlier = eq & (tile_q.index[None, :] < tile_p.index[:, None])
            less = (kq[None, :] < kp[:, None]) | earlier
            pos = pos + torch.sum(less.to(torch.int32), dim=1).to(torch.int32)
            before = before + torch.sum(earlier.to(torch.int32), dim=1).to(torch.int32)
            same = same + torch.sum(eq.to(torch.int32), dim=1).to(torch.int32)
        dst = torch.where(tile_p.index < pairs, pos, pairs)  # tail lanes -> trash slot P
        pair_sorted[dst] = tile_p.index
        seg[dst] = torch.where((before == 0) & (kp < e_count), same, 0)
        seg_e[dst] = kp
    for tile_m, tile_b in hl.tile([m, hdim // 16], block_size=[bm, bb]):
        v = x3[tile_m, tile_b, :].to(torch.float32)
        amax = torch.amax(torch.abs(v), dim=-1)
        sf = torch.clamp(amax * g6, max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
        # g / sf correctly rounded (the spec's IEEE division; Triton's f32 `/` is
        # div.full): f64 divide + round to f32 is exact (53 >= 2*24 + 2 bits)
        inv = torch.where(sf == 0.0, 0.0,
                          (g / torch.where(sf == 0.0, 1.0, sf).to(torch.float64)).to(torch.float32))
        sv = v * inv[:, :, None]
        a = torch.clamp(torch.abs(sv), max=6.0)
        # e2m1 round-to-nearest-even (ties at 0.75, 1.75, 3.5 round up)
        q = (0.5 * ((a > 0.25).to(torch.float32) + (a >= 0.75).to(torch.float32)
                    + (a > 1.25).to(torch.float32) + (a >= 1.75).to(torch.float32))
             + (a > 2.5).to(torch.float32) + (a >= 3.5).to(torch.float32)
             + 2.0 * (a > 5.0).to(torch.float32))
        xq3[tile_m, tile_b, :] = (torch.where(sv < 0.0, -q, q) * sf[:, :, None]).to(torch.bfloat16)
    return xdq, pair_sorted, seg, seg_e


def nvfp4_moe_fc1(xdq: torch.Tensor, pair_sorted: torch.Tensor, seg: torch.Tensor,
                  seg_e: torch.Tensor, w13: torch.Tensor, w13_sf: torch.Tensor,
                  g1: torch.Tensor, a2g: float, a2g6: float, act: int, topk: int,
                  limit: float):
    """h bf16 [P+1, I]: NVFP4-dequantized act(min(g1*gate, limit)) *
    clamp(g1*up, -limit, limit) per routed pair (vLLM's swiglu clamp; NO_CLAMP
    = none), rows of off-rank pairs unwritten, row P = trash. Grid (sorted
    position, blocks of 16 I-columns: whole NVFP4 blocks for the requant). Block
    caps keep the 4 dot operands + reshape scratch inside sm_120's ~99 KB smem
    (N <= 64, K <= 64)."""
    m, hdim = xdq.size()
    e_count, two_i, kh = w13.size()
    idim = two_i // 2
    pairs = pair_sorted.size(0) - 1
    nb = idim // 16
    nkb = hdim // 16
    cp = (nkb + 3) // 4 * 4
    estride = (two_i + 127) // 128 * 128 * cp
    x4 = xdq.view(m, nkb, 8, 2)
    w4 = w13.view(e_count * 2 * nb, 16, nkb, 8)  # row block (e, up|gate, nb)
    sf = w13_sf.view(-1)
    h = torch.empty([pairs + 1, idim], dtype=torch.bfloat16, device=xdq.device)
    h3 = h.view(pairs + 1, nb, 16)
    bkb = hl.register_block_size(2, min(8, nkb))  # dot K = 8 * bkb >= 16
    brow = hl.register_block_size(16, 64)
    bcol = hl.register_block_size(1, min(4, nb))  # dot N = 16 * bcol <= 64
    for tile_s, tile_b in hl.tile([pairs, nb], block_size=[1, bcol]):
        s = tile_s.begin
        cnt = seg[s]
        e = torch.clamp(seg_e[s], max=e_count - 1)
        alpha = g1[e]
        col = torch.clamp(tile_b.index[:, None] * 16 + hl.arange(16)[None, :], max=idim - 1)
        colg = col + idim
        rb_u = torch.clamp(tile_b.index, max=nb - 1) + 2 * e * nb  # 1-D expert row blocks (tail-safe)
        rb_g = rb_u + nb
        su_row = (e * estride + (col // 128) * (cp // 4) * 512 + (col % 32) * 16
                  + ((col // 32) % 4) * 4)
        sg_row = (e * estride + (colg // 128) * (cp // 4) * 512 + (colg % 32) * 16
                  + ((colg // 32) % 4) * 4)
        for tile_r in hl.tile(0, cnt, block_size=brow):
            # lanes past the segment read the next segment's entries (or the trash slot):
            # clamp so every gather below stays inside the arrays (their stores go to row P)
            pair = pair_sorted[torch.clamp(s + tile_r.index, max=pairs)]
            tok = torch.clamp(pair, max=pairs - 1) // topk
            acc_u = hl.zeros([tile_r, tile_b, 16], dtype=torch.float32)
            acc_u = acc_u.reshape(acc_u.size(0), -1)
            acc_g = hl.zeros([tile_r, tile_b, 16], dtype=torch.float32)
            acc_g = acc_g.reshape(acc_g.size(0), -1)
            for tile_k in hl.tile(nkb, block_size=bkb):
                a_lo = x4[tok, tile_k, :, 0]
                a_lo = a_lo.reshape(a_lo.size(0), -1)
                a_hi = x4[tok, tile_k, :, 1]
                a_hi = a_hi.reshape(a_hi.size(0), -1)
                kb = torch.clamp(tile_k.index, max=nkb - 1)
                kofs = (kb // 4) * 512 + kb % 4
                su = sf[su_row[:, :, None] + kofs[None, None, :]].to(torch.float32) * 16384.0
                sg = sf[sg_row[:, :, None] + kofs[None, None, :]].to(torch.float32) * 16384.0
                bu = w4[rb_u, :, tile_k, :]
                bg = w4[rb_g, :, tile_k, :]
                # e2m1 nibble -> fp16 bits (exact incl. the 0.5 denormal) * 2^14 * sf
                n = (bu & 15).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * su[:, :, :, None]
                wu_lo = v.to(torch.bfloat16).reshape(v.size(0) * 16, -1)
                n = (bu >> 4).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * su[:, :, :, None]
                wu_hi = v.to(torch.bfloat16).reshape(v.size(0) * 16, -1)
                n = (bg & 15).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * sg[:, :, :, None]
                wg_lo = v.to(torch.bfloat16).reshape(v.size(0) * 16, -1)
                n = (bg >> 4).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * sg[:, :, :, None]
                wg_hi = v.to(torch.bfloat16).reshape(v.size(0) * 16, -1)
                acc_u = hl.dot(a_lo, wu_lo.T, acc=acc_u)
                acc_u = hl.dot(a_hi, wu_hi.T, acc=acc_u)
                acc_g = hl.dot(a_lo, wg_lo.T, acc=acc_g)
                acc_g = hl.dot(a_hi, wg_hi.T, acc=acc_g)
            gate = torch.clamp(alpha * acc_g, max=limit)
            up = torch.clamp(alpha * acc_u, min=-limit, max=limit)
            # act: silu (0) | gelu_tanh (1) = x * sigmoid(2u) (oxide `act`); exact blend
            z = (1.5957691216057308 * (gate + 0.044715 * gate * gate * gate)) * act \
                + gate * (1 - act)
            hv = gate * torch.sigmoid(z) * up
            hv = hv.reshape(hv.size(0), -1, 16)
            amax = torch.amax(torch.abs(hv), dim=-1)
            sfh = torch.clamp(amax * a2g6, max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
            inv = torch.where(sfh == 0.0, 0.0, (a2g / torch.where(sfh == 0.0, 1.0, sfh).to(
                torch.float64)).to(torch.float32))  # IEEE g / sf, as in route_quant
            sv = hv * inv[:, :, None]
            a = torch.clamp(torch.abs(sv), max=6.0)
            q = (0.5 * ((a > 0.25).to(torch.float32) + (a >= 0.75).to(torch.float32)
                        + (a > 1.25).to(torch.float32) + (a >= 1.75).to(torch.float32))
                 + (a > 2.5).to(torch.float32) + (a >= 3.5).to(torch.float32)
                 + 2.0 * (a > 5.0).to(torch.float32))
            dst = torch.where(tile_r.index < cnt, pair, pairs)
            h3[dst, tile_b, :] = (torch.where(sv < 0.0, -q, q) * sfh[:, :, None]).to(torch.bfloat16)
    return h


def nvfp4_moe_fc2(h: torch.Tensor, pair_sorted: torch.Tensor, seg: torch.Tensor,
                  seg_e: torch.Tensor, topk_w: torch.Tensor, w2: torch.Tensor,
                  w2_sf: torch.Tensor, g2: torch.Tensor):
    """y f32 [P+1, H] = g2[e] * topk_w[p] * (h[p] . w2[e]) per routed pair
    (off-rank rows unwritten, row P = trash). Grid (sorted position, H tile)."""
    e_count, hdim, kh = w2.size()
    idim = kh * 2
    pairs = pair_sorted.size(0) - 1
    nkb = idim // 16
    cp = (nkb + 3) // 4 * 4
    estride = (hdim + 127) // 128 * 128 * cp
    h4 = h.view(pairs + 1, nkb, 8, 2)
    w3 = w2.view(e_count * hdim, nkb, 8)
    tw = topk_w.view(-1)
    sf = w2_sf.view(-1)
    y = torch.empty([pairs + 1, hdim], dtype=torch.float32, device=h.device)
    bkb = hl.register_block_size(2, min(8, nkb))
    brow = hl.register_block_size(16, 64)
    bh = hl.register_block_size(16, 256)
    for tile_s, tile_h in hl.tile([pairs, hdim], block_size=[1, bh]):
        s = tile_s.begin
        cnt = seg[s]
        e = torch.clamp(seg_e[s], max=e_count - 1)
        alpha = g2[e]
        row = torch.clamp(tile_h.index, max=hdim - 1)
        rows = row + e * hdim
        s_row = (e * estride + (row // 128) * (cp // 4) * 512 + (row % 32) * 16
                 + ((row // 32) % 4) * 4)
        for tile_r in hl.tile(0, cnt, block_size=brow):
            pair = torch.clamp(pair_sorted[torch.clamp(s + tile_r.index, max=pairs)], max=pairs - 1)
            acc = hl.zeros([tile_r, tile_h], dtype=torch.float32)
            for tile_k in hl.tile(nkb, block_size=bkb):
                a_lo = h4[pair, tile_k, :, 0]
                a_lo = a_lo.reshape(a_lo.size(0), -1)
                a_hi = h4[pair, tile_k, :, 1]
                a_hi = a_hi.reshape(a_hi.size(0), -1)
                kb = torch.clamp(tile_k.index, max=nkb - 1)
                sc = sf[s_row[:, None] + ((kb // 4) * 512 + kb % 4)[None, :]].to(torch.float32) \
                    * 16384.0
                b = w3[rows, tile_k, :]
                n = (b & 15).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * sc[:, :, None]
                w_lo = v.to(torch.bfloat16).reshape(v.size(0), -1)
                n = (b >> 4).to(torch.int16)
                v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                v = torch.where((n & 8) != 0, -v, v) * sc[:, :, None]
                w_hi = v.to(torch.bfloat16).reshape(v.size(0), -1)
                acc = hl.dot(a_lo, w_lo.T, acc=acc)
                acc = hl.dot(a_hi, w_hi.T, acc=acc)
            dst = torch.where(tile_r.index < cnt, pair, pairs)
            y[dst, tile_h] = acc * (alpha * tw[pair])[:, None]
    return y


def nvfp4_moe_combine(y: torch.Tensor, topk_ids: torch.Tensor, id_base: int, e_count: int,
                      out: torch.Tensor):
    """out bf16 [M, H] = bf16(sum_k y[t*topk + k]) over local ids, k ascending."""
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


def nvfp4_moe_fc2_combine(h: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                          w2: torch.Tensor, w2_sf: torch.Tensor, g2: torch.Tensor,
                          id_base: int, out: torch.Tensor):
    """Token-major fc2 + combine: out[t] = bf16(sum_k g2[e]*tw[p] * (h[p] . w2[e]))
    over local ids, k ascending; one GEMV per (token, H tile), no y workspace.
    Off-rank pairs skip their expert's weights entirely."""
    m, topk = topk_ids.size()
    e_count, hdim, kh = w2.size()
    nkb = kh // 8
    cp = (nkb + 3) // 4 * 4
    estride = (hdim + 127) // 128 * 128 * cp
    ids = topk_ids.view(-1)
    tw = topk_w.view(-1)
    h4 = h.view(h.size(0), nkb, 8, 2)
    w3 = w2.view(e_count * hdim, nkb, 8)
    sf = w2_sf.view(-1)
    bh = hl.register_block_size(16, 128)
    bk = hl.register_block_size(1, min(16, nkb))
    for tile_t, tile_h in hl.tile([m, hdim], block_size=[1, bh]):
        t = tile_t.begin
        row = torch.clamp(tile_h.index, max=hdim - 1)
        s_row = (row // 128) * (cp // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4
        tot = hl.zeros([tile_h], dtype=torch.float32)
        for k in hl.grid(topk):
            p = t * topk + k
            loc = ids[p].to(torch.int32) - id_base
            if (loc >= 0) & (loc < e_count):
                acc = hl.zeros([tile_h], dtype=torch.float32)
                rows = row + loc * hdim
                for tile_k in hl.tile(nkb, block_size=bk):
                    a_lo = h4[p, tile_k, :, 0].to(torch.float32)
                    a_hi = h4[p, tile_k, :, 1].to(torch.float32)
                    kb = torch.clamp(tile_k.index, max=nkb - 1)
                    sc = sf[loc * estride + s_row[:, None]
                            + ((kb // 4) * 512 + kb % 4)[None, :]].to(torch.float32) * 16384.0
                    b = w3[rows, tile_k, :]
                    n = (b & 15).to(torch.int16)
                    v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                    w_lo = torch.where((n & 8) != 0, -v, v) * sc[:, :, None]
                    n = (b >> 4).to(torch.int16)
                    v = ((n & 7) << 9).view(torch.float16).to(torch.float32)
                    w_hi = torch.where((n & 8) != 0, -v, v) * sc[:, :, None]
                    acc = acc + torch.sum(torch.sum(w_lo * a_lo[None, :, :]
                                                    + w_hi * a_hi[None, :, :], dim=2), dim=1)
                tot = tot + acc * (g2[loc] * tw[p])
        out[t, tile_h] = tot.to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------
# serving entry
# ---------------------------------------------------------------------------
def _f32(v: float) -> float:
    import numpy as np
    return float(np.float32(v))


def _div6(g: float) -> float:
    """f32(g) / f32(6) with IEEE rounding (the oxide quant16's g / 6.0)."""
    import numpy as np
    return float(np.float32(g) / np.float32(6.0))


def lane_case(e_count: int, hdim: int, idim: int, topk: int, act: int, m: int,
              limit: float | None = None, e_global: int | None = None) -> dict:
    """The tuned case a serving call of M tokens runs as (lab `cases` keys;
    `limit` only on clamped layers, `experts_global` only under EP)."""
    case = {"experts": int(e_count), "hidden": int(hdim), "moe_intermediate": int(idim),
            "topk": int(topk), "act": ACT[int(act)], "numtokens": hk.bucket(m, buckets_for(limit))}
    if limit is not None:
        case["limit"] = float(limit)
    if e_global is not None and int(e_global) != int(e_count):
        case["experts_global"] = int(e_global)
    return case


def plan_name(mode: int) -> str:
    return "fused3" if int(mode) == 2 else "front4"


# ---------------------------------------------------------------------------
# body variants per M bucket: the seam to the dot_scaled bodies (docs/dsv41-moe.md section 11)
#
# A VARIANT is a whole plan's worth of bodies (the inter-stage carriers differ: this module's
# dequant bodies hand bf16-dequantized NVFP4 between stages, a dot_scaled plan hands packed
# e2m1 + e4m3 block scales), so it is selected per M BUCKET, never per stage. Selection is
# DATA: rows of the pseudo-kernel VARIANT_KERNEL in the shipped config table,
#   configs/helion/dsv41_moe_variant/<platform>.json = [{"key": <lane_case>, "config": {"body": "dot_scaled"}}]
# no row = "dequant". The lab writes the rows (the tuner gate's bench compares the variants per
# bucket and prints/writes them); serving only reads them. A module serving a variant exports
#   VARIANT PLANS BODIES run_plan KERNELS   (check_variant pins the shape of that contract)
# and this module is the "dequant" one.
# ---------------------------------------------------------------------------
DEQUANT, DOT_SCALED = "dequant", "dot_scaled"
VARIANT = DEQUANT  # this module is the "dequant" variant
VARIANT_KERNEL = "dsv41_moe_variant"
VARIANT_MODULES = {DOT_SCALED: "suffix_hybrid.helion_kernels.dsv41_moe_fp4mma"}
_VARIANTS: dict = {}  # name -> module-like (imported lazily; tests / lanes may register)
_KERNEL_KEYS = ("name", "body", "cases", "make", "reference", "check", "baseline")


def check_variant(name: str, mod) -> None:
    """Fail closed on a variant module that does not honour the seam (K5: refuse, never guess)."""
    bad = []
    if getattr(mod, "VARIANT", None) != name:
        bad.append(f"VARIANT {getattr(mod, 'VARIANT', None)!r} != {name!r}")
    plans = getattr(mod, "PLANS", None)
    bodies = getattr(mod, "BODIES", None)
    if not isinstance(plans, dict) or not plans or not set(plans) <= set(PLANS):
        bad.append(f"PLANS must be a non-empty dict over {sorted(PLANS)}")
    if not isinstance(bodies, dict):
        bad.append("BODIES must be {kernel name: undecorated Helion body}")
    if not callable(getattr(mod, "run_plan", None)):
        bad.append("run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, "
                   "out, a1_gscale, a2_gscale, act, id_base, limit) is missing")
    if not bad:
        for plan, names in plans.items():
            for k in names:
                if k not in bodies and k not in BODIES:
                    bad.append(f"PLANS[{plan!r}] names {k!r}: no body in BODIES")
        kerns = getattr(mod, "KERNELS", None)
        if not isinstance(kerns, list) or not kerns:
            bad.append("KERNELS (lab ABI list) is missing")
        else:
            for e in kerns:
                lacking = [key for key in _KERNEL_KEYS if key not in e]
                if lacking:
                    bad.append(f"KERNELS entry {e.get('name')!r} lacks {lacking}")
                elif e["name"] in BODIES:
                    bad.append(f"KERNELS entry {e['name']!r} collides with a dequant kernel name")
    if bad:
        raise RuntimeError(f"{MARKER} body variant {name!r} breaks the seam: " + "; ".join(bad))


def variant_module(name: str):
    """The module-like that serves variant `name`."""
    if name == DEQUANT:
        return sys.modules[__name__]
    mod = _VARIANTS.get(name)
    if mod is None:
        path = VARIANT_MODULES.get(name)
        if path is None:
            raise RuntimeError(f"{MARKER} unknown body variant {name!r} "
                               f"(known: {[DEQUANT, *VARIANT_MODULES]})")
        try:
            mod = importlib.import_module(path)
        except ModuleNotFoundError as exc:
            if exc.name != path:
                raise  # a missing dependency INSIDE the variant module is a real error
            raise RuntimeError(f"{MARKER} body variant {name!r} is selected but {path} is not "
                               "part of this build") from exc
        check_variant(name, mod)
        _VARIANTS[name] = mod
    return mod


def available_variants() -> tuple:
    """(dequant, + every variant whose module is in this build): what the lab can compare."""
    out = [DEQUANT]
    for name in VARIANT_MODULES:
        try:
            variant_module(name)
            out.append(name)
        except RuntimeError as exc:
            if "not part of this build" not in str(exc):
                raise
    return tuple(out)


def variant_for(case: dict) -> str:
    """The body variant a bucketed lane case runs: its row in the config table, else dequant."""
    cfg = hk.config(VARIANT_KERNEL, case)
    if cfg is None:
        return DEQUANT
    name = cfg.get("body")
    if name != DEQUANT and name not in VARIANT_MODULES and name not in _VARIANTS:
        raise RuntimeError(f"{MARKER} {VARIANT_KERNEL} row for {hk.case_key(case)} names body "
                           f"{name!r} (known: {[DEQUANT, *VARIANT_MODULES, *_VARIANTS]})")
    return name


def plan_kernels(variant: str, plan: str) -> tuple:
    """Kernel names of `plan` under `variant` (launch order)."""
    names = variant_module(variant).PLANS.get(plan)
    if names is None:
        raise RuntimeError(f"{MARKER} body variant {variant!r} has no {plan!r} plan "
                           f"(it serves {sorted(variant_module(variant).PLANS)})")
    return tuple(names)


def missing(case: dict, mode: int, variant: str | None = None) -> list:
    """Kernels of mode's plan (under `variant`, default = the case's table row) with no shipped
    config for `case` ([] = Helion serves it)."""
    variant = variant_for(case) if variant is None else variant
    return [k for k in plan_kernels(variant, plan_name(mode)) if hk.config(k, case) is None]


def served_ms(backend: str, e_count: int, hdim: int, idim: int, topk: int, act: int,
              max_m: int, mode_for, limit: float | None = None,
              e_global: int | None = None) -> dict:
    """{M: (bucket, plan, variant)} for every M in 1..max_m that Helion serves under
    `backend` (mode_for(M) = the lane's plan mode; variant = the bucket's table row). helion: a
    gap is a startup error naming every missing (M bucket, variant, kernel); auto: gaps stay
    on the oxide op; oxide: {}."""
    if backend == "oxide":
        return {}
    out, gaps = {}, {}
    for m in range(1, max_m + 1):
        case = lane_case(e_count, hdim, idim, topk, act, m, limit, e_global)
        mode = mode_for(m)
        variant = variant_for(case)
        lack = missing(case, mode, variant)
        if lack:
            gaps.setdefault((case["numtokens"], plan_name(mode), variant), lack)
        else:
            out[m] = (case["numtokens"], plan_name(mode), variant)
    if gaps and backend == "helion":
        raise RuntimeError(
            f"{MARKER} {KNOB}=helion (or a layer only Helion can serve) but no tuned config on "
            f"{hk.platform()} for E={e_count}/{e_global or e_count} H={hdim} I={idim} "
            f"topk={topk} {ACT[int(act)]} limit={limit}: "
            + "; ".join(f"M<={b} {p}/{v}: {', '.join(k)}" for (b, p, v), k in sorted(gaps.items()))
            + " (tune them in the autotune lab, ship configs/helion/<kernel>/<platform>.json)")
    return out


BODIES = {ROUTE_QUANT: nvfp4_moe_route_quant, FC1: nvfp4_moe_fc1, FC2: nvfp4_moe_fc2,
          COMBINE: nvfp4_moe_combine, FC2_COMBINE: nvfp4_moe_fc2_combine}


@functools.lru_cache(maxsize=None)
def _kernels(case_key: str, plan: str, untuned_ok: bool = False, variant: str = DEQUANT):
    """The plan's serving kernels for one bucketed case under `variant`. untuned_ok (the
    lane's standalone oracle / bench only, never serving): a kernel with no shipped config
    runs Helion's default config instead of failing."""
    import json
    case = json.loads(case_key)
    mod = variant_module(variant)
    out = []
    for k in plan_kernels(variant, plan):
        body = mod.BODIES.get(k) or BODIES[k]
        if untuned_ok and hk.config(k, case) is None:
            hk.log(f"nvfp4-moe UNTUNED {k} {case_key}: Helion default config (bench/oracle only)")
            out.append(helion.kernel(body, autotune_effort="none", static_shapes=False))
        else:
            out.append(hk.compiled(body, k, case))
    return out


def run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out,
             a1_gscale, a2_gscale, act, id_base, limit=None):
    """One MoE forward on Helion kernels `kerns` (PLANS[plan] order) -> out."""
    e_count = w13.shape[0]
    f8 = lambda t: t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)
    xdq, ps, seg, seg_e = kerns[0](x, topk_ids, _f32(a1_gscale), _div6(a1_gscale),
                                   e_count, id_base)
    h = kerns[1](xdq, ps, seg, seg_e, w13, f8(w13_sf), g1, _f32(a2_gscale), _div6(a2_gscale),
                 act, topk_ids.shape[1], NO_CLAMP if limit is None else float(limit))
    if plan == "fused3":
        return kerns[2](h, topk_ids, topk_w, w2, f8(w2_sf), g2, id_base, out)
    y = kerns[2](h, ps, seg, seg_e, topk_w, w2, f8(w2_sf), g2)
    return kerns[3](y, topk_ids, id_base, e_count, out)


def serving_op(native, backend: str, limit: float | None = None, e_global: int | None = None,
               untuned_ok: bool = False, variant: str | None = None):
    """The oxide host op's signature (_native.nvfp4_moe_cuda) on Helion: per M
    the plan's kernels when all are tuned for bucket(M), else (auto) the oxide
    op itself; helion fails closed (native may then be None: the oxide-free
    DeepSeek-V4 route never imports it). ws_* / tune / stream_ptr are the oxide
    op's (Helion runs on torch's current stream = the lane's stream). limit /
    e_global: the layer's swiglu clamp and global expert count (case keys).
    untuned_ok: standalone oracle / bench only (see _kernels). variant: force one body
    variant (the lab's bench compares them); None = each bucket's table row."""

    def op(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, ws_aq, ws_asf, ws_inter,
           ws_hq, ws_hsf, ws_y, ws_route, out, a1_gscale, a2_gscale, act, id_base, mode,
           tune, stream_ptr):
        case = lane_case(w13.shape[0], x.shape[1], w13.shape[1] // 2, topk_ids.shape[1],
                         act, x.shape[0], limit, e_global)
        plan = plan_name(mode)
        # ponytail: per-call config lookup (json key, dict get); free under CUDA
        # graphs, ~us per layer eager — memoize per (case, mode) if eager decode matters
        body = variant or variant_for(case)
        lack = [] if untuned_ok else missing(case, mode, body)
        if lack:
            if backend == "helion":
                raise RuntimeError(f"{MARKER} {KNOB}=helion: no tuned {lack} for "
                                   f"{hk.case_key(case)} (load-time check missed it)")
            if native is None:
                raise RuntimeError(f"{MARKER} no tuned {lack} for {hk.case_key(case)} and no oxide "
                                   "kernel to fall back to on this (oxide-free) route")
            return native.nvfp4_moe_cuda(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2,
                                         ws_aq, ws_asf, ws_inter, ws_hq, ws_hsf, ws_y, ws_route,
                                         out, a1_gscale, a2_gscale, act, id_base, mode, tune,
                                         stream_ptr)
        return variant_module(body).run_plan(
            _kernels(hk.case_key(case), plan, untuned_ok, body), plan, x, topk_ids, topk_w, w13,
            w13_sf, g1, w2, w2_sf, g2, out, a1_gscale, a2_gscale, act, id_base, limit)

    op.backend = backend
    return op


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py)
# ---------------------------------------------------------------------------
def _is_ds(arch: str) -> bool:
    return arch.startswith("deepseekv4")  # DeepseekV4ForCausalLM, DeepseekV41ForCausalLM


def _layouts(model: dict) -> list:
    """Lane-eligible NVFP4 MoE dims of the TP rank(s) to tune, else [].
    Default: the combination's rank (no EP: E global, I = I/tp). DeepSeek-V4
    (clamped silu, TP8 serving): the EP8 rank (E/8 local of E, whole I; prod:
    FlashInfer refuses TP8 without EP) whether the console hands the dims of
    a TP8 target without EP (experts = global, moe_intermediate = I/8) or of
    a TP8 + EP8 target (api/autotune.model_dims: experts = this rank's,
    experts_total = global, moe_intermediate = whole I, ep = 8); the no-EP
    rank (E, I/8) only with NVFP4_MOE_TP8_NOEP=1 — it has no stock baseline
    (nvfp4_moe.fi_supported), so a lab row for it can only judge correctness."""
    if model.get("quant_format") != "nvfp4":
        return []
    e, h, i, k = (model.get(n) for n in ("experts", "hidden", "moe_intermediate", "topk"))
    if not (e and h and i and k) or h % 64 or i % 16:
        return []
    arch = str(model.get("architecture") or model.get("name") or "").lower()
    if _is_ds(arch):
        ep = int(model.get("ep") or 1)
        eg = int(model.get("experts_total") or e * ep)  # global experts
        ig = i if ep > 1 else i * int(model.get("tp") or 1)  # whole per-expert intermediate
        d = {"experts": eg, "hidden": h, "topk": k, "act": "silu",
             "limit": float(model.get("swiglu_limit") or DS_LIMIT)}
        # EP8 first (prod: vLLM's FlashInfer CUTLASS refuses TP8 without EP,
        # "intermediate size padding for w1 and w3"), then the no-EP shard
        out = [{**d, "experts": eg // DS_TP, "moe_intermediate": ig, "experts_global": eg}] \
            if eg % DS_TP == 0 else []
        if ig % (16 * DS_TP) == 0 and os.environ.get("NVFP4_MOE_TP8_NOEP") == "1":
            out.append({**d, "moe_intermediate": ig // DS_TP})
        return out
    if e > 256 or i % 64:  # the oxide baseline's limits (only DS layers are Helion-only)
        return []
    return [{"experts": e, "hidden": h, "moe_intermediate": i, "topk": k,
             "act": "gelu_tanh" if "gemma" in arch else "silu"}]


def cases(model: dict, kernel: str | None = None) -> list:
    """Every layout x M bucket; on a Helion-only (clamped) layout the
    CLAMP_BUCKETS `kernel`'s plan can serve (FUSED3_MAX / FRONT4_MIN)."""
    want = {FC2_COMBINE: lambda m: m <= FUSED3_MAX, FC2: lambda m: m >= FRONT4_MIN,
            COMBINE: lambda m: m >= FRONT4_MIN}.get(kernel, lambda m: True)
    return [{**d, "numtokens": m} for d in _layouts(model) for m in buckets_for(d.get("limit"))
            if "limit" not in d or want(m)]


@functools.lru_cache(maxsize=2)
def _weights(e_count, hdim, idim, act, seed, limit=None):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return nm.make_problem(e_count, hdim, idim, act, device=_device(), seed=seed, limit=limit)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def route_ref(topk_ids, e_count: int, id_base: int):
    """Torch twin of the route root: (pair_sorted, seg, seg_e) int32 [P+1]
    (slot P = trash, 0 here)."""
    pairs = topk_ids.numel()
    dev = topk_ids.device
    loc = topk_ids.reshape(-1).long() - id_base
    key = torch.where((loc >= 0) & (loc < e_count), loc, torch.full_like(loc, e_count))
    order = torch.argsort(key * pairs + torch.arange(pairs, device=dev))
    sk = key[order]
    start = torch.ones(pairs, dtype=torch.bool, device=dev)
    start[1:] = sk[1:] != sk[:-1]
    counts = torch.bincount(key, minlength=e_count + 1)
    seg = torch.where(start & (sk < e_count), counts[sk], torch.zeros_like(sk))
    pad = lambda t: torch.cat([t.int(), torch.zeros(1, dtype=torch.int32, device=dev)])
    return pad(order), pad(seg), pad(sk)


def ep_base(case) -> int:
    """id_base of the lab's EP rank: a middle rank of experts_global/experts."""
    e = case["experts"]
    return e * (case.get("experts_global", e) // e // 2)


ROUTING = None  # force one nm.ROUTING_DISTS entry (the in-lab tuner / bench sets it)
LAB_DISTS = ("balanced", "skewed")  # make(case, seed) -> LAB_DISTS[seed % 2]


def routing_dist(seed) -> str:
    """Expert-load distribution of make(case, seed): the runner gates correctness
    on seed 0 (balanced) and times the rotating copies seed 1, 2.. (skewed,
    balanced..), so one timed number covers both; baseline and Helion arms see the
    same routing (same seed)."""
    return ROUTING or LAB_DISTS[seed % 2]


def _problem(case, seed):
    """(p, x, ids, tw) for one lab case: the lane's synthetic layer (random
    NVFP4 experts, calibrated gscales, the case's clamp) + top-k routing over
    the GLOBAL experts (EP: ~1/ep of the pairs are local), balanced or skewed
    (routing_dist)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    e, hd, i, k, m = (case[n] for n in ("experts", "hidden", "moe_intermediate", "topk",
                                        "numtokens"))
    p = _weights(e, hd, i, case["act"], seed % 2, case.get("limit"))
    dev = p["w13"].device
    eg = case.get("experts_global", e)
    ids, tw = nm.rand_routing(m, eg, k, dev, seed=1000 + seed, dead=False,
                              base=ep_base(case) if eg != e else 0, local=e,
                              dist=routing_dist(seed))
    gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
    x = torch.randn(m, hd, generator=gen).bfloat16().to(dev)
    return p, x, ids, tw


def _h_ref(p, xdq, ps, seg, seg_e, topk):
    """(h bf16 [P+1, I] = spec requant of the f64 (clamped) fc1 output, valid rows)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ps.numel() - 1
    idim = p["w13"].shape[1] // 2
    h = torch.zeros(pairs + 1, idim, dtype=torch.float64, device=xdq.device)
    valid = torch.zeros(pairs + 1, dtype=torch.bool, device=xdq.device)
    for s in torch.nonzero(seg[:pairs] > 0).reshape(-1).tolist():
        cnt, e = int(seg[s]), int(seg_e[s])
        pr = ps[s:s + cnt].long()
        gu = float(p["g1"][e]) * (xdq[pr // topk].double()
                                  @ nm._expert_w(p["w13"], p["w13_sf"], e).double().T)
        gu = nm.clamp_ref(gu, p.get("limit"))
        h[pr] = nm.act_ref(gu[:, idim:], p["act"]) * gu[:, :idim]
        valid[pr] = True
    return nm.qdq(h.float(), p["a2g"]).bfloat16(), valid


def _y_ref(h, ps, seg, seg_e, tw, w2, w2_sf, g2):
    """(y f64 [P+1, H] = g2 * tw * (h . w2), valid rows)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ps.numel() - 1
    y = torch.zeros(pairs + 1, w2.shape[1], dtype=torch.float64, device=h.device)
    valid = torch.zeros(pairs + 1, dtype=torch.bool, device=h.device)
    for s in torch.nonzero(seg[:pairs] > 0).reshape(-1).tolist():
        cnt, e = int(seg[s]), int(seg_e[s])
        pr = ps[s:s + cnt].long()
        y[pr] = (float(g2[e]) * (h[pr].double() @ nm._expert_w(w2, w2_sf, e).double().T)
                 * tw.reshape(-1)[pr, None].double())
        valid[pr] = True
    return y, valid


def _stage_inputs(case, seed):
    p, x, ids, tw = _problem(case, seed)
    from suffix_hybrid.kernels import nvfp4_moe as nm
    xdq = nm.qdq(x.float(), p["a1g"]).bfloat16()
    ps, seg, seg_e = route_ref(ids, p["w13"].shape[0], ep_base(case))
    return p, x, ids, tw, xdq, ps, seg, seg_e


def _make_route_quant(case, seed):
    p, x, ids, _ = _problem(case, seed)
    return x, ids, _f32(p["a1g"]), _div6(p["a1g"]), p["w13"].shape[0], ep_base(case)


def _make_fc1(case, seed):
    p, _, ids, _, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    lim = case.get("limit")
    return (xdq, ps, seg, seg_e, p["w13"], p["w13_sf"], p["g1"], _f32(p["a2g"]),
            _div6(p["a2g"]), p["act"], ids.shape[1], NO_CLAMP if lim is None else float(lim))


def _make_fc2(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    return h, ps, seg, seg_e, tw, p["w2"], p["w2_sf"], p["g2"]


def _make_combine(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    y, _ = _y_ref(h, ps, seg, seg_e, tw, p["w2"], p["w2_sf"], p["g2"])
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    return y.float(), ids, ep_base(case), p["w2"].shape[0], out


def _make_fc2_combine(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    return h, ids, tw, p["w2"], p["w2_sf"], p["g2"], ep_base(case), out


def _ref_route_quant(x, topk_ids, g, g6, e_count, id_base):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return (nm.qdq(x.float(), g).bfloat16(), *route_ref(topk_ids, e_count, id_base))


def _ref_fc1(xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, act, topk, limit):
    return _h_ref(dict(w13=w13, w13_sf=w13_sf, g1=g1, a2g=a2g, act=act,
                       limit=None if limit >= NO_CLAMP else limit), xdq, ps, seg, seg_e, topk)


def _ref_fc2(h, ps, seg, seg_e, tw, w2, w2_sf, g2):
    return _y_ref(h, ps, seg, seg_e, tw, w2, w2_sf, g2)


def _ref_combine(y, topk_ids, id_base, e_count, out):
    """f32 sum in the kernels' k order (bit-exact target), bf16 once."""
    m, topk = topk_ids.shape
    loc = topk_ids.long() - id_base
    ok = (loc >= 0) & (loc < e_count)
    acc = torch.zeros(m, y.shape[1], dtype=torch.float32, device=y.device)
    y3 = y[:m * topk].view(m, topk, -1)
    for k in range(topk):
        acc = torch.where(ok[:, k, None], acc + y3[:, k], acc)
    return acc.bfloat16()


def _ref_fc2_combine(h, topk_ids, topk_w, w2, w2_sf, g2, id_base, out):
    """f64 sum over local pairs of g2 * tw * (h . w2) (bf16 rounding is the floor)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    m, topk = topk_ids.shape
    loc = (topk_ids.long() - id_base).reshape(-1)
    acc = torch.zeros(m * topk, w2.shape[1], dtype=torch.float64, device=h.device)
    for e in torch.unique(loc).tolist():
        if 0 <= e < w2.shape[0]:
            pr = torch.nonzero(loc == e).reshape(-1)
            acc[pr] = (float(g2[e]) * (h[pr].double() @ nm._expert_w(w2, w2_sf, e).double().T)
                       * topk_w.reshape(-1)[pr, None].double())
    return acc.view(m, topk, -1).sum(1)


def _frac(a, b) -> float:
    return float((a != b).float().mean()) if a.numel() else 0.0


def _rel(a, b) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _check_route_quant(out, ref):
    """Routing exact (the counting sort is integer math); x quant like the
    lane's xq stage (STAGE_TOL: rare rounding ties)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ref[1].numel() - 1
    route = all(torch.equal(o[:pairs].int(), r[:pairs]) for o, r in zip(out[1:], ref[1:]))
    fx = _frac(out[0].float(), ref[0].float())
    return route and fx <= nm.STAGE_TOL["xq"], f"route exact={route} xq differing={fx:.1e}"


def _check_rows(tol_key, rel: bool):
    def check(out, ref):
        from suffix_hybrid.kernels import nvfp4_moe as nm
        want, valid = ref
        got, want = out[valid].float(), want[valid]
        err = _rel(got, want) if rel else _frac(got, want.float())
        ok = bool(torch.isfinite(got).all()) and err <= nm.STAGE_TOL[tol_key]
        return ok, f"{tol_key} {'rel' if rel else 'differing'}={err:.1e} rows={int(valid.sum())}"
    return check


def _check_combine(out, ref):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    f = _frac(out.float(), ref.float())
    return f <= nm.STAGE_TOL["comb"], f"comb differing={f:.1e}"


def _check_end(out, ref):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    r = _rel(out, ref)
    return bool(torch.isfinite(out.float()).all()) and r <= nm.REF_TOL, \
        f"rel_vs_f64={r:.2e} (REF_TOL {nm.REF_TOL})"


# Served SUFFIX_NVFP4_MOE_TUNE per model (lab jobs cannot set env): the
# baseline is the prod oxide launch. GLM-5.3 TP8 (silu, E 256, H 6144); every
# other model (gemma: the lane documents none) gets the lane's DEFAULT_TUNE.
GLM_PROD_TUNE = ("1:fused3/32/255/1,5:front4/8/255/1,10:front4/8/0/1,32:front4/8/255/1,"
                 "64:front4/8/255/0")


def baseline_tune(case: dict):
    """(TUNE table, label) the oxide baseline runs with: SUFFIX_NVFP4_MOE_TUNE
    when set, else the model's prod table."""
    import os
    from suffix_hybrid.kernels import nvfp4_moe as nm
    if os.environ.get(nm.TUNE_ENV, "").strip():
        return nm.tune_table(), "env tune"
    if (case["experts"], case["hidden"], case["act"]) == (256, 6144, "silu"):
        return nm.parse_tune(GLM_PROD_TUNE), "glm prod tune"
    return nm.parse_tune(nm.DEFAULT_TUNE), "lane default tune"


def oxide_can(case) -> bool:
    """The oxide kernel runs this case (no clamp, E <= 256, I % 64; nvfp4_moe.oxide_gap)."""
    return "limit" not in case and case["experts"] <= 256 and case["moe_intermediate"] % 64 == 0


def _fi_args(case, seed):
    """Baseline args where oxide cannot run: vLLM's FLASHINFER_CUTLASS path
    for the WHOLE layer (x quant + both GEMMs + clamp + finalize, this EP
    rank's ep_size/ep_rank) — the production op every stage is part of."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    if not nm.fi_supported(case["moe_intermediate"]):
        raise RuntimeError(f"no FlashInfer baseline for I={case['moe_intermediate']} per rank "
                           "(it pads w1/w3 to 128 rows: DeepSeek-V4.1 TP8 without EP)")
    p, x, ids, tw = _problem(case, seed)
    e, dev = p["w13"].shape[0], p["w13"].device
    fi = dict(p, a1g_t=torch.full((e,), p["a1g"], dtype=torch.float32, device=dev),
              a2g_t=torch.full((e,), p["a2g"], dtype=torch.float32, device=dev),
              ep_size=case.get("experts_global", e) // e, ep_rank=ep_base(case) // e)
    out = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=dev)
    return ("fi", fi, x, ids, tw, out)


def _oxide_args(case, seed, mode: int, keep: int):
    """The oxide op's argument tuple (stream excluded) that runs ONLY launch
    `keep` of `mode`'s plan (tune skip mask), workspace pre-filled by one full
    run so the kept launch reads real upstream data; prefetch tune = the M's
    row of baseline_tune (prod). Cases oxide cannot run: _fi_args."""
    if not oxide_can(case):
        return _fi_args(case, seed)
    from suffix_hybrid import oxide_kernels
    from suffix_hybrid.kernels import nvfp4_moe as nm
    oxide_kernels.ensure_loaded(nm.FAMILY, params=nm.PARAMS)
    native = oxide_kernels.native()
    p, x, ids, tw = _problem(case, seed)
    m, (e, hd, i, k) = x.shape[0], (case[n] for n in ("experts", "hidden",
                                                      "moe_intermediate", "topk"))
    ws = nm.workspace(x.device, m, k, hd, i, e)
    out = torch.empty(m, hd, dtype=torch.bfloat16, device=x.device)
    tune = nm.tune_for(m, baseline_tune(case)[0])[1]
    head = (x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"], *ws, out,
            float(p["a1g"]), float(p["a2g"]), int(p["act"]), 0, mode)
    native.nvfp4_moe_cuda(*head, tune, torch.cuda.current_stream().cuda_stream)
    launches = 3 if mode == 2 else 4
    skip = ((1 << launches) - 1) & ~(1 << keep)
    return (*head, nm.tune_word(tune & 0xFF, (tune >> 8) & 0xFF, (tune >> 16) & 1, skip))


def _oxide_baseline(*args):
    if isinstance(args[0], str):  # _fi_args
        from suffix_hybrid.kernels import nvfp4_moe as nm
        try:
            return nm._fi_call(*args[1:])
        except TypeError:  # a FlashInfer without swiglu_limit: same GEMMs, unclamped (timing only)
            return nm._fi_call(dict(args[1], limit=None), *args[2:])
    from suffix_hybrid import oxide_kernels
    return oxide_kernels.native().nvfp4_moe_cuda(*args, torch.cuda.current_stream().cuda_stream)


# Helion's autotuner compares every candidate with the default config's output over the WHOLE
# output tensor (settings.autotune_baseline_accuracy_check_fn replaces that check). Ours are not
# fully written: h / y rows of off-rank pairs and the trash slot P are torch.empty garbage, and
# the trash slot of the route arrays is a data race between tail lanes, so a whole-tensor
# comparison rejects (or, by luck of the allocator, accepts) candidates at random. The check
# below compares only what a consumer reads, masked by the args of the latest make() (the lab
# calls make(case, 0) right before it autotunes; the in-lab tuner does too). This module's own
# KERNELS skip that: their baseline is the exact eager reference and the lab check judges each
# candidate (_tune_check, masked the same way); dsv41_moe_shared's kernels use autotune_check.
LAST_ARGS: dict = {}
AUTOTUNE_FLIP = 5e-3  # fraction of requantized / bf16 values that may differ between configs
AUTOTUNE_REL = 2e-3   # rel-L2 between float outputs of two configs (accumulation order)


def remember(name, make):
    """make(case, seed) that records its args under `name` for autotune_check."""
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

        def rows(ps_i, se_i, e_i):  # rows a consumer reads: the local pairs
            ps, seg_e, e_count = args[ps_i], args[se_i], args[e_i].shape[0]
            pairs = ps.numel() - 1
            return ps[:pairs][seg_e[:pairs] < e_count].long()

        def flips(a, b, what):
            f = _frac(a.float(), b.float())
            if not bool(torch.isfinite(a.float()).all()) or f > AUTOTUNE_FLIP:
                raise AssertionError(f"{name}.{what}: {f:.2e} of the values differ (> {AUTOTUNE_FLIP})")

        def close(a, b, what):
            r = _rel(a.float(), b.float())
            if not bool(torch.isfinite(a.float()).all()) or r > AUTOTUNE_REL:
                raise AssertionError(f"{name}.{what}: rel {r:.2e} (> {AUTOTUNE_REL})")

        if name == ROUTE_QUANT:
            pairs = want[1].numel() - 1
            flips(got[0], want[0], "xdq")
            for i, w in zip(range(1, 4), ("pair_sorted", "seg", "seg_e")):
                if not torch.equal(got[i][:pairs], want[i][:pairs]):
                    raise AssertionError(f"{name}.{w}: routing differs between configs")
        elif name in (FC1, "dsv41_moe_fc1_sh"):
            r = rows(1, 3, 4)
            flips(got[0][r], want[0][r], "h")
            if len(got) > 1:
                close(got[1], want[1], "hs")
        elif name == FC2:
            r = rows(1, 3, 5)
            close(got[0][r], want[0][r], "y")
        else:  # combine kernels: every row of out is written
            flips(got[0], want[0], "out")
    return check


def _tune_check(check):
    """Autotuner accuracy check = the lab gate (defined rows only, requant-flip
    tolerant); raises AssertionError like torch.testing."""
    def tune_check(actual, expected):
        ok, msg = check(actual, expected)
        if not ok:
            raise AssertionError(msg)
    return tune_check


def _entry(name, body, make, reference, check, mode, keep, oxide_entry):
    # Lab-only settings (serving never autotunes): the autotuner's baseline is
    # the exact eager reference (the oxide launches write packed/slot formats,
    # not these outputs), so a failing default config cannot abort a case;
    # candidates are judged by the lab check; DEBUG logs show real errors.
    return {"name": name, "body": body, "cases": functools.partial(cases, kernel=name),
            "make": make, "reference": reference, "check": check,
            "settings": {"autotune_baseline_fn": reference,
                         "autotune_baseline_accuracy_check_fn": _tune_check(check),
                         "autotune_log_level": logging.DEBUG},
            "baseline": _oxide_baseline,
            "make_baseline": lambda case, seed: _oxide_args(case, seed, mode, keep),
            "baseline_name": f"oxide nvfp4_moe {oxide_entry} (glm prod tune; env "
                             f"SUFFIX_NVFP4_MOE_TUNE wins; else lane default tune); where oxide "
                             f"cannot run (clamp, E>256, I%64): vLLM FlashInfer CUTLASS, WHOLE "
                             f"layer (sum a plan's stages to compare)",
            "replaces": [rf"^{oxide_entry}$"]}


KERNELS = [
    _entry(ROUTE_QUANT, nvfp4_moe_route_quant, _make_route_quant, _ref_route_quant,
           _check_route_quant, 1, 0, "moe_route_quant"),
    _entry(FC1, nvfp4_moe_fc1, _make_fc1, _ref_fc1, _check_rows("hq", rel=False), 1, 1,
           "moe_fc1_quant"),
    _entry(FC2, nvfp4_moe_fc2, _make_fc2, _ref_fc2, _check_rows("fc2", rel=True), 1, 2,
           "moe_fc2"),
    _entry(COMBINE, nvfp4_moe_combine, _make_combine, _ref_combine, _check_combine, 1, 3,
           "moe_combine"),
    _entry(FC2_COMBINE, nvfp4_moe_fc2_combine, _make_fc2_combine, _ref_fc2_combine, _check_end,
           2, 2, "moe_fc2_combine"),
]
