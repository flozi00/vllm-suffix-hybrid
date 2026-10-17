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
      epilogue -> h bf16 [P, I] dequantized
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
import logging

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
    """(xdq bf16 [M, H], pair_sorted, seg, seg_e int32 [P]). Key of pair p
    = topk_ids[p] - id_base when in [0, E) else E (off-rank, sorts last);
    its sorted position = #{q: (key_q, q) < (key_p, p)}. g6 = f32(g)/f32(6)
    computed IEEE on the host (the kernel's g / 6)."""
    m, hdim = x.size()
    pairs = topk_ids.numel()
    ids = topk_ids.view(-1)
    x3 = x.view(m, hdim // 16, 16)
    xdq = torch.empty([m, hdim], dtype=torch.bfloat16, device=x.device)
    xq3 = xdq.view(m, hdim // 16, 16)
    pair_sorted = torch.empty([pairs], dtype=torch.int32, device=x.device)
    seg = torch.empty([pairs], dtype=torch.int32, device=x.device)
    seg_e = torch.empty([pairs], dtype=torch.int32, device=x.device)
    for tile_p in hl.tile(pairs):
        kp = ids[tile_p].to(torch.int32) - id_base
        kp = torch.where((kp >= 0) & (kp < e_count), kp, e_count)
        pos = hl.zeros([tile_p], dtype=torch.int32)
        before = hl.zeros([tile_p], dtype=torch.int32)
        same = hl.zeros([tile_p], dtype=torch.int32)
        for tile_q in hl.tile(pairs):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            kq = torch.where(tile_q.index < pairs, kq, e_count + 1)  # tail lanes rank nothing
            eq = kq[None, :] == kp[:, None]
            earlier = eq & (tile_q.index[None, :] < tile_p.index[:, None])
            less = (kq[None, :] < kp[:, None]) | earlier
            pos = pos + torch.sum(less.to(torch.int32), dim=1).to(torch.int32)
            before = before + torch.sum(earlier.to(torch.int32), dim=1).to(torch.int32)
            same = same + torch.sum(eq.to(torch.int32), dim=1).to(torch.int32)
        # every output fully written (the autotuner compares all of it); the
        # tile's tail lanes are masked off these scatters
        pair_sorted[pos] = tile_p.index
        seg[pos] = torch.where((before == 0) & (kp < e_count), same, 0)
        seg_e[pos] = kp
    for tile_m, tile_b in hl.tile([m, hdim // 16]):
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
                  g1: torch.Tensor, a2g: float, a2g6: float, act: int, topk: int):
    """h bf16 [P, I]: NVFP4-dequantized act(g1*gate) * g1*up per routed pair
    (rows of off-rank pairs unwritten). Grid (sorted position, blocks of 16
    I-columns: whole NVFP4 blocks for the requant). Block caps keep the 4 dot
    operands + reshape scratch inside sm_120's ~99 KB smem (N <= 64, K <= 128)."""
    m, hdim = xdq.size()
    e_count, two_i, kh = w13.size()
    idim = two_i // 2
    pairs = pair_sorted.size(0)
    nb = idim // 16
    nkb = hdim // 16
    cp = (nkb + 3) // 4 * 4
    estride = (two_i + 127) // 128 * 128 * cp
    x4 = xdq.view(m, nkb, 8, 2)
    w4 = w13.view(e_count * 2 * nb, 16, nkb, 8)  # row block (e, up|gate, nb)
    sf = w13_sf.view(-1)
    h = torch.empty([pairs, idim], dtype=torch.bfloat16, device=xdq.device)
    h3 = h.view(pairs, nb, 16)
    bkb = hl.register_block_size(2, 16)  # dot K = 8 * bkb in [16, 128]
    bnb = hl.register_block_size(1, 4)   # dot N = 16 * bnb in [16, 64]
    for tile_s, tile_b in hl.tile([pairs, nb], block_size=[1, bnb]):
        s = tile_s.begin
        cnt = seg[s]
        e = torch.clamp(seg_e[s], max=e_count - 1)
        alpha = g1[e]
        col = torch.clamp(tile_b.index[:, None] * 16 + hl.arange(16)[None, :], max=idim - 1)
        colg = col + idim
        rb_u = tile_b.index + 2 * e * nb  # 1-D expert row blocks: no 0-d tensor index
        rb_g = rb_u + nb
        su_row = (e * estride + (col // 128) * (cp // 4) * 512 + (col % 32) * 16
                  + ((col // 32) % 4) * 4)
        sg_row = (e * estride + (colg // 128) * (cp // 4) * 512 + (colg % 32) * 16
                  + ((colg // 32) % 4) * 4)
        for tile_r in hl.tile(0, cnt):
            pair = pair_sorted[s + tile_r.index]
            tok = pair // topk
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
            gate = alpha * acc_g
            up = alpha * acc_u
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
            h3[pair, tile_b, :] = (torch.where(sv < 0.0, -q, q) * sfh[:, :, None]).to(torch.bfloat16)
    return h


def nvfp4_moe_fc2(h: torch.Tensor, pair_sorted: torch.Tensor, seg: torch.Tensor,
                  seg_e: torch.Tensor, topk_w: torch.Tensor, w2: torch.Tensor,
                  w2_sf: torch.Tensor, g2: torch.Tensor):
    """y f32 [P, H] = g2[e] * topk_w[p] * (h[p] . w2[e]) per routed pair
    (off-rank rows unwritten). Grid (sorted position, H tile)."""
    e_count, hdim, kh = w2.size()
    idim = kh * 2
    pairs = pair_sorted.size(0)
    nkb = idim // 16
    cp = (nkb + 3) // 4 * 4
    estride = (hdim + 127) // 128 * 128 * cp
    h4 = h.view(pairs, nkb, 8, 2)
    w3 = w2.view(e_count * hdim, nkb, 8)
    tw = topk_w.view(-1)
    sf = w2_sf.view(-1)
    y = torch.empty([pairs, hdim], dtype=torch.float32, device=h.device)
    bkb = hl.register_block_size(2, 16)  # dot K = 8 * bkb in [16, 128]
    bh = hl.register_block_size(16, 128)
    for tile_s, tile_h in hl.tile([pairs, hdim], block_size=[1, bh]):
        s = tile_s.begin
        cnt = seg[s]
        e = torch.clamp(seg_e[s], max=e_count - 1)
        alpha = g2[e]
        rows = tile_h.index + e * hdim
        row = torch.clamp(tile_h.index, max=hdim - 1)
        s_row = (e * estride + (row // 128) * (cp // 4) * 512 + (row % 32) * 16
                 + ((row // 32) % 4) * 4)
        for tile_r in hl.tile(0, cnt):
            pair = pair_sorted[s + tile_r.index]
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
            y[pair, tile_h] = acc * (alpha * tw[pair])[:, None]
    return y


def nvfp4_moe_combine(y: torch.Tensor, topk_ids: torch.Tensor, id_base: int, e_count: int,
                      out: torch.Tensor):
    """out bf16 [M, H] = bf16(sum_k y[t*topk + k]) over local ids, k ascending."""
    m, topk = topk_ids.size()
    hdim = out.size(1)
    ids = topk_ids.view(-1)
    for tile_t, tile_h in hl.tile([m, hdim]):
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
    for tile_t, tile_h in hl.tile([m, hdim], block_size=[1, None]):
        t = tile_t.begin
        row = torch.clamp(tile_h.index, max=hdim - 1)
        s_row = (row // 128) * (cp // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4
        tot = hl.zeros([tile_h], dtype=torch.float32)
        for k in hl.grid(topk):
            p = t * topk + k
            loc = ids[p].to(torch.int32) - id_base
            if (loc >= 0) & (loc < e_count):
                acc = hl.zeros([tile_h], dtype=torch.float32)
                rows = tile_h.index + loc * hdim
                for tile_k in hl.tile(nkb):
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


def lane_case(e_count: int, hdim: int, idim: int, topk: int, act: int, m: int) -> dict:
    """The tuned case a serving call of M tokens runs as (lab `cases` keys)."""
    return {"experts": int(e_count), "hidden": int(hdim), "moe_intermediate": int(idim),
            "topk": int(topk), "act": ACT[int(act)], "numtokens": hk.bucket(m, BUCKETS)}


def plan_name(mode: int) -> str:
    return "fused3" if int(mode) == 2 else "front4"


def missing(case: dict, mode: int) -> list:
    """Kernels of mode's plan with no shipped config for `case` ([] = Helion serves it)."""
    return [k for k in PLANS[plan_name(mode)] if hk.config(k, case) is None]


def served_ms(backend: str, e_count: int, hdim: int, idim: int, topk: int, act: int,
              max_m: int, mode_for) -> dict:
    """{M: (bucket, plan)} for every M in 1..max_m that Helion serves under
    `backend` (mode_for(M) = the lane's plan mode). helion: a gap is a
    startup error naming every missing (M bucket, kernel); auto: gaps stay on
    the oxide op; oxide: {}."""
    if backend == "oxide":
        return {}
    out, gaps = {}, {}
    for m in range(1, max_m + 1):
        case = lane_case(e_count, hdim, idim, topk, act, m)
        mode = mode_for(m)
        lack = missing(case, mode)
        if lack:
            gaps.setdefault((case["numtokens"], plan_name(mode)), lack)
        else:
            out[m] = (case["numtokens"], plan_name(mode))
    if gaps and backend == "helion":
        raise RuntimeError(
            f"{MARKER} {KNOB}=helion but no tuned config on {hk.platform()} for E={e_count} "
            f"H={hdim} I={idim} topk={topk} {ACT[int(act)]}: "
            + "; ".join(f"M<={b} {p}: {', '.join(k)}" for (b, p), k in sorted(gaps.items()))
            + " (tune them in the autotune lab, ship configs/helion/<kernel>/<platform>.json)")
    return out


@functools.lru_cache(maxsize=None)
def _kernels(case_key: str, plan: str):
    import json
    case = json.loads(case_key)
    bodies = {ROUTE_QUANT: nvfp4_moe_route_quant, FC1: nvfp4_moe_fc1, FC2: nvfp4_moe_fc2,
              COMBINE: nvfp4_moe_combine, FC2_COMBINE: nvfp4_moe_fc2_combine}
    return [hk.compiled(bodies[k], k, case) for k in PLANS[plan]]


def run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out,
             a1_gscale, a2_gscale, act, id_base):
    """One MoE forward on Helion kernels `kerns` (PLANS[plan] order) -> out."""
    e_count = w13.shape[0]
    f8 = lambda t: t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)
    xdq, ps, seg, seg_e = kerns[0](x, topk_ids, _f32(a1_gscale), _div6(a1_gscale),
                                   e_count, id_base)
    h = kerns[1](xdq, ps, seg, seg_e, w13, f8(w13_sf), g1, _f32(a2_gscale), _div6(a2_gscale),
                 act, topk_ids.shape[1])
    if plan == "fused3":
        return kerns[2](h, topk_ids, topk_w, w2, f8(w2_sf), g2, id_base, out)
    y = kerns[2](h, ps, seg, seg_e, topk_w, w2, f8(w2_sf), g2)
    return kerns[3](y, topk_ids, id_base, e_count, out)


def serving_op(native, backend: str):
    """The oxide host op's signature (_native.nvfp4_moe_cuda) on Helion: per M
    the plan's kernels when all are tuned for bucket(M), else (auto) the oxide
    op itself; helion fails closed. ws_* / tune / stream_ptr are the oxide
    op's (Helion runs on torch's current stream = the lane's stream)."""

    def op(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, ws_aq, ws_asf, ws_inter,
           ws_hq, ws_hsf, ws_y, ws_route, out, a1_gscale, a2_gscale, act, id_base, mode,
           tune, stream_ptr):
        case = lane_case(w13.shape[0], x.shape[1], w13.shape[1] // 2, topk_ids.shape[1],
                         act, x.shape[0])
        plan = plan_name(mode)
        # ponytail: per-call config lookup (json key, dict get); free under CUDA
        # graphs, ~us per layer eager — memoize per (case, mode) if eager decode matters
        lack = missing(case, mode)
        if lack:
            if backend == "helion":
                raise RuntimeError(f"{MARKER} {KNOB}=helion: no tuned {lack} for "
                                   f"{hk.case_key(case)} (load-time check missed it)")
            return native.nvfp4_moe_cuda(x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2,
                                         ws_aq, ws_asf, ws_inter, ws_hq, ws_hsf, ws_y, ws_route,
                                         out, a1_gscale, a2_gscale, act, id_base, mode, tune,
                                         stream_ptr)
        return run_plan(_kernels(hk.case_key(case), plan), plan, x, topk_ids, topk_w, w13,
                        w13_sf, g1, w2, w2_sf, g2, out, a1_gscale, a2_gscale, act, id_base)

    op.backend = backend
    return op


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py)
# ---------------------------------------------------------------------------
def _dims(model: dict) -> dict | None:
    """Lane-eligible NVFP4 MoE dims of one TP rank (no EP: E global), else None."""
    if model.get("quant_format") != "nvfp4":
        return None
    e, h, i, k = (model.get(n) for n in ("experts", "hidden", "moe_intermediate", "topk"))
    if not (e and h and i and k) or e > 256 or h % 64 or i % 64:
        return None
    arch = str(model.get("architecture") or model.get("name") or "").lower()
    return {"experts": e, "hidden": h, "moe_intermediate": i, "topk": k,
            "act": "gelu_tanh" if "gemma" in arch else "silu"}


def cases(model: dict) -> list:
    d = _dims(model)
    return [] if d is None else [{**d, "numtokens": m} for m in BUCKETS]


@functools.lru_cache(maxsize=2)
def _weights(e_count, hdim, idim, act, seed):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return nm.make_problem(e_count, hdim, idim, act, device=_device(), seed=seed)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def route_ref(topk_ids, e_count: int, id_base: int):
    """Torch twin of the route root: (pair_sorted, seg, seg_e) int32 [P]."""
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
    return order.int(), seg.int(), sk.int()


def _problem(case, seed):
    """(p, x, ids, tw) for one lab case: the lane's synthetic layer (random
    NVFP4 experts, calibrated gscales) + uniform top-k routing."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    e, hd, i, k, m = (case[n] for n in ("experts", "hidden", "moe_intermediate", "topk",
                                        "numtokens"))
    p = _weights(e, hd, i, case["act"], seed % 2)
    dev = p["w13"].device
    ids, tw = nm.rand_routing(m, e, k, dev, seed=1000 + seed, dead=False)
    gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
    x = torch.randn(m, hd, generator=gen).bfloat16().to(dev)
    return p, x, ids, tw


def _h_ref(p, xdq, ps, seg, seg_e, topk):
    """(h bf16 [P, I] = spec requant of the f64 fc1 output, valid rows)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ps.numel()
    idim = p["w13"].shape[1] // 2
    h = torch.zeros(pairs, idim, dtype=torch.float64, device=xdq.device)
    valid = torch.zeros(pairs, dtype=torch.bool, device=xdq.device)
    for s in torch.nonzero(seg[:pairs] > 0).reshape(-1).tolist():
        cnt, e = int(seg[s]), int(seg_e[s])
        pr = ps[s:s + cnt].long()
        gu = float(p["g1"][e]) * (xdq[pr // topk].double()
                                  @ nm._expert_w(p["w13"], p["w13_sf"], e).double().T)
        h[pr] = nm.act_ref(gu[:, idim:], p["act"]) * gu[:, :idim]
        valid[pr] = True
    return nm.qdq(h.float(), p["a2g"]).bfloat16(), valid


def _y_ref(h, ps, seg, seg_e, tw, w2, w2_sf, g2):
    """(y f64 [P, H] = g2 * tw * (h . w2), valid rows)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    pairs = ps.numel()
    y = torch.zeros(pairs, w2.shape[1], dtype=torch.float64, device=h.device)
    valid = torch.zeros(pairs, dtype=torch.bool, device=h.device)
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
    ps, seg, seg_e = route_ref(ids, p["w13"].shape[0], 0)
    return p, x, ids, tw, xdq, ps, seg, seg_e


def _make_route_quant(case, seed):
    p, x, ids, _ = _problem(case, seed)
    return x, ids, _f32(p["a1g"]), _div6(p["a1g"]), p["w13"].shape[0], 0


def _make_fc1(case, seed):
    p, _, ids, _, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    return (xdq, ps, seg, seg_e, p["w13"], p["w13_sf"], p["g1"], _f32(p["a2g"]),
            _div6(p["a2g"]), p["act"], ids.shape[1])


def _make_fc2(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    return h, ps, seg, seg_e, tw, p["w2"], p["w2_sf"], p["g2"]


def _make_combine(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    y, _ = _y_ref(h, ps, seg, seg_e, tw, p["w2"], p["w2_sf"], p["g2"])
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    return y.float(), ids, 0, p["w2"].shape[0], out


def _make_fc2_combine(case, seed):
    p, _, ids, tw, xdq, ps, seg, seg_e = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    return h, ids, tw, p["w2"], p["w2_sf"], p["g2"], 0, out


def _ref_route_quant(x, topk_ids, g, g6, e_count, id_base):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return (nm.qdq(x.float(), g).bfloat16(), *route_ref(topk_ids, e_count, id_base))


def _ref_fc1(xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, act, topk):
    return _h_ref(dict(w13=w13, w13_sf=w13_sf, g1=g1, a2g=a2g, act=act), xdq, ps, seg,
                  seg_e, topk)


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
    route = all(torch.equal(o.int(), r) for o, r in zip(out[1:], ref[1:]))
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


def _oxide_args(case, seed, mode: int, keep: int):
    """The oxide op's argument tuple (stream excluded) that runs ONLY launch
    `keep` of `mode`'s plan (tune skip mask), workspace pre-filled by one full
    run so the kept launch reads real upstream data; prefetch tune = the M's
    row of baseline_tune (prod)."""
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
    from suffix_hybrid import oxide_kernels
    return oxide_kernels.native().nvfp4_moe_cuda(*args, torch.cuda.current_stream().cuda_stream)


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
    # not these outputs), so a failing reference config cannot abort a case;
    # candidates are judged by the lab check; DEBUG logs show real errors.
    return {"name": name, "body": body, "cases": cases, "make": make, "reference": reference,
            "check": check, "baseline": _oxide_baseline,
            "make_baseline": lambda case, seed: _oxide_args(case, seed, mode, keep),
            "baseline_name": f"oxide nvfp4_moe {oxide_entry} (glm prod tune; env "
                             f"SUFFIX_NVFP4_MOE_TUNE wins; else lane default tune)",
            "replaces": [rf"^{oxide_entry}$"],
            "settings": {"autotune_baseline_fn": reference,
                         "autotune_baseline_accuracy_check_fn": _tune_check(check),
                         "autotune_log_level": logging.DEBUG}}


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
