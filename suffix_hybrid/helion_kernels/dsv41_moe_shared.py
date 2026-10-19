# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared expert fused into the Helion routed-MoE plan (lane B,
SUFFIX_DSV41_MOE_FUSE_SHARED, lane glue: suffix_hybrid/kernels/dsv41_moe_shared.py).

Why: stock runs the shared MLP as ~6 launches on vLLM's aux stream (act quant,
mm_mxfp8, clamp-silu-mul, act quant, mm_mxfp8) plus the runner's `shared + routed`
add. Here the shared expert rides inside launches the routed plan already has:

  route_quant   unchanged (base module)
  fc1           dsv41_moe_fc1_sh: the routed fc1 + a SECOND ROOT LOOP = its own
                program-id region of the SAME launch (Helion emits one Triton kernel,
                `if pid < N_region0 ... else ...`): shared gate/up GEMM tiled over
                (M tiles x 16-column blocks), clamp + silu*up, h_sh bf16 [M, I_sh].
                The shared region is the FIRST root, so its programs get the lowest
                program ids and start first (longest chain first)
  fc2/combine   front4: fc2 unchanged, dsv41_moe_combine_sh = the routed combine +
                the shared down projection as an M-tiled dot (token tile 16..64 x H
                tile), out = bf16(sum_k y + S);  fused3: dsv41_moe_fc2_combine_sh =
                the token-major routed fc2+combine + one more GEMV slot per (token,
                H tile), out = bf16(sum_k + S)
  No extra launch, no scratch beyond h_sh (allocated by torch inside the body: the
  graph pool under capture), no semaphores, tickets or other state: every launch
  rewrites everything it reads later, so the plan is stateless and capture-safe.

Why it cannot be the long pole (the glm-int fused slot was: it walked all M rows
serially inside one CTA, -5.4 ms/step at M=96, -22 ms at M=192): the shared work
is tiled over M exactly like routed work (grid = ceil(M/BM) x I_sh/16/BJ programs
in fc1, ceil(M/BT) x H/BH in the combine, each with a bounded K chain: 160 steps of
32 for fc1 as long as a routed fc1 program, 9 for the down projection), the work
per token is a constant ~12 % of ONE routed expert's, and the K4 harness
(kernels/dsv41_moe_shared.py `bench`) measures fused vs routed + stock shared on
vLLM's aux stream at M in 1/6/12/24/48/96/192 and prints the ceiling
(SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M); above it the lane runs routed-only + vLLM's
own overlapped shared MLP. At boot the first fused layer is timed against exactly that
composition (kernels/dsv41_moe_shared.long_pole_gate) and a loss is a startup error.

Math (W8A16, deterministic): the shared weights are vLLM's post-processed MXFP8
linears (FlashInferCutlassMxfp8LinearKernel on SM120: weight float8_e4m3fn [N, K]
contiguous, scale uint8 e8m0 (2^(s-127)) per (row, 32-K block) in the F8_128x4
swizzle of utils/mxfp8_utils.swizzle_mxfp8_scale), read IN PLACE (no copy):
    gate_up [2*I_sh, H] = [gate; up]   (stacked w1, w3), down [H, I_sh]
    gate = min(x . Wg^T, L); up = clamp(x . Wu^T, -L, L)      (L = swiglu_limit 10)
    h = bf16(silu(gate) * up);  S = h . Wd^T                  (f32 accumulate)
vLLM additionally quantizes x and h to MXFP8 per call (W8A8); here they stay
bf16, so ours is at least as accurate as stock (the oracle bounds the distance
to stock at SHARED_STOCK_TOL, the distance to the f64 spec of THIS math tight).
Dequant is exact: e4m3 x 2^k is a bf16 value; s is bit-built (s << 23), not
exp2. Combine order: out = bf16(((... + y_k) ascending k) + S), one rounding;
stock is bf16(shared_bf16 + routed_bf16) (runner `shared_output + fused_output`,
routed scaling 1.5 already folded into the top-k weights by dsv4_topk and
apply_routed_scale_to_output False, runner rsf 1.0).

Search spaces (literal bounds, powers of two, LFBO-full ready): see BOUNDS.
Lab ABI: KERNELS (api/autotune_kernels.py), cases keyed by the routed EP8 case
plus `sh_inter`; baselines are the UNFUSED Helion composition of the same math
(the fusion gain alone); the verdict vs stock is the `bench` gate.
"""
from __future__ import annotations

import functools

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor
from suffix_hybrid.helion_kernels import nvfp4_moe as hm

ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU / no Helion: the module still imports (lab ABI, tests)
    helion = hl = None

MARKER = "[suffix dsv41-moe-shared]"
KNOB = "SUFFIX_DSV41_MOE_FUSE_SHARED"
FC1_SH, FC2_COMBINE_SH, COMBINE_SH = ("dsv41_moe_fc1_sh", "dsv41_moe_fc2_combine_sh",
                                      "dsv41_moe_combine_sh")
# the plan = base kernels where the shared expert does not touch the stage
PLANS_SH = {"front4": (hm.ROUTE_QUANT, FC1_SH, hm.FC2, COMBINE_SH),
            "fused3": (hm.ROUTE_QUANT, FC1_SH, FC2_COMBINE_SH)}
NEW_KERNELS = (FC1_SH, COMBINE_SH, FC2_COMBINE_SH)
E8M0_BIAS, SCALE_BLOCK = 127, 32
S_MIN, S_MAX = 1, 246  # accepted e8m0 bytes: 0 would be 2^-127 (bit trick gives 0.0), > 246 can overflow
# search-space bounds, literal inside the bodies (tests/test_dsv41_moe_shared.py reads them back
# from the bound ConfigSpec): (min, max) per tunable block size
BOUNDS = {
    FC1_SH: {"shared rows (dot M)": (16, 64), "shared 16-col blocks": (1, 8),
             "shared K blocks of 32": (1, 8), "routed K (x8 bytes)": (2, 8),
             "routed rows": (16, 64), "routed 16-col blocks": (1, 8)},
    COMBINE_SH: {"tokens (dot M)": (16, 64), "H": (16, 256), "shared K blocks of 32": (1, 8)},
    FC2_COMBINE_SH: {"H": (16, 128), "routed K": (1, 16), "shared K blocks of 32": (1, 8)},
}
CAPS = {FC1_SH: 64, COMBINE_SH: 256, FC2_COMBINE_SH: 128}  # largest max_size of any block size


# ---------------------------------------------------------------------------
# Helion bodies (module-level, undecorated; literals only inside: they are traced)
# ---------------------------------------------------------------------------
def dsv41_moe_fc1_sh(xdq: torch.Tensor, pair_sorted: torch.Tensor, seg: torch.Tensor,
                     seg_e: torch.Tensor, w13: torch.Tensor, w13_sf: torch.Tensor,
                     g1: torch.Tensor, a2g: float, a2g6: float, topk: int, limit: float,
                     x: torch.Tensor, wgu: torch.Tensor, sgu: torch.Tensor):
    """(h, hs). h = nvfp4_moe_fc1's routed intermediate (silu only: act(min(g1*gate, limit)) *
    clamp(g1*up, -limit, limit), NVFP4 requant, bf16 dequantized [P+1, I], row P = trash).
    hs bf16 [M, I_sh] = silu(min(x.Wg^T, limit)) * clamp(x.Wu^T, -limit, limit) of the shared
    expert (wgu float8_e4m3fn [2*I_sh, H] = [gate; up], sgu its F8_128x4 e8m0 scales).
    Root 1 (shared, tiles [M, I_sh/16]) runs first, root 2 (routed) is nvfp4_moe_fc1."""
    m, hdim = xdq.size()
    e_count, two_i, kh = w13.size()
    idim = two_i // 2
    pairs = pair_sorted.size(0) - 1
    nb = idim // 16
    nkb = hdim // 16
    cp = (nkb + 3) // 4 * 4
    estride = (two_i + 127) // 128 * 128 * cp
    x4 = xdq.view(m, nkb, 8, 2)
    w4 = w13.view(e_count * 2 * nb, 16, nkb, 8)
    sf = w13_sf.view(-1)
    h = torch.empty([pairs + 1, idim], dtype=torch.bfloat16, device=xdq.device)
    h3 = h.view(pairs + 1, nb, 16)
    isz = wgu.size(0) // 2
    nbs = isz // 16
    nks = hdim // 32
    cps = (nks + 3) // 4 * 4
    xs3 = x.view(m, nks, 32)
    ws4 = wgu.view(2 * nbs, 16, nks, 32)
    ssf = sgu.view(-1)
    hs = torch.empty([m, isz], dtype=torch.bfloat16, device=x.device)
    hs3 = hs.view(m, nbs, 16)
    sm = hl.register_block_size(16, 64)
    sj = hl.register_block_size(1, min(8, nbs))
    sk = hl.register_block_size(1, min(8, nks))
    bkb = hl.register_block_size(2, min(8, nkb))
    brow = hl.register_block_size(16, 64)
    bcol = hl.register_block_size(1, min(8, nb))
    for tile_m, tile_j in hl.tile([m, nbs], block_size=[sm, sj]):
        jb = torch.clamp(tile_j.index, max=nbs - 1)
        col = jb[:, None] * 16 + hl.arange(16)[None, :]
        colu = col + isz
        sg_row = (col // 128) * (cps // 4) * 512 + (col % 32) * 16 + ((col // 32) % 4) * 4
        su_row = (colu // 128) * (cps // 4) * 512 + (colu % 32) * 16 + ((colu // 32) % 4) * 4
        s_acc_g = hl.zeros([tile_m, tile_j, 16], dtype=torch.float32)
        s_acc_g = s_acc_g.reshape(s_acc_g.size(0), -1)
        s_acc_u = hl.zeros([tile_m, tile_j, 16], dtype=torch.float32)
        s_acc_u = s_acc_u.reshape(s_acc_u.size(0), -1)
        for tile_k in hl.tile(nks, block_size=sk):
            a_s = xs3[tile_m, tile_k, :]
            a_s = a_s.reshape(a_s.size(0), -1)
            kk = torch.clamp(tile_k.index, max=nks - 1)
            kofs = (kk // 4) * 512 + kk % 4
            sg_b = (ssf[sg_row[:, :, None] + kofs[None, None, :]].to(torch.int32) << 23).view(
                torch.float32)
            su_b = (ssf[su_row[:, :, None] + kofs[None, None, :]].to(torch.int32) << 23).view(
                torch.float32)
            wg_q = ws4[jb, :, tile_k, :]
            wu_q = ws4[jb + nbs, :, tile_k, :]
            wg_s = (wg_q.to(torch.float32) * sg_b[:, :, :, None]).to(torch.bfloat16)
            wg_s = wg_s.reshape(wg_s.size(0) * 16, -1)
            wu_s = (wu_q.to(torch.float32) * su_b[:, :, :, None]).to(torch.bfloat16)
            wu_s = wu_s.reshape(wu_s.size(0) * 16, -1)
            s_acc_g = hl.dot(a_s, wg_s.T, acc=s_acc_g)
            s_acc_u = hl.dot(a_s, wu_s.T, acc=s_acc_u)
        s_gate = torch.clamp(s_acc_g, max=limit)
        s_up = torch.clamp(s_acc_u, min=-limit, max=limit)
        s_hv = s_gate * torch.sigmoid(s_gate) * s_up
        hs3[tile_m, tile_j, :] = s_hv.reshape(s_hv.size(0), -1, 16).to(torch.bfloat16)
    for tile_s, tile_b in hl.tile([pairs, nb], block_size=[1, bcol]):
        s = tile_s.begin
        cnt = seg[s]
        e = torch.clamp(seg_e[s], max=e_count - 1)
        alpha = g1[e]
        col = torch.clamp(tile_b.index[:, None] * 16 + hl.arange(16)[None, :], max=idim - 1)
        colg = col + idim
        rb_u = torch.clamp(tile_b.index, max=nb - 1) + 2 * e * nb
        rb_g = rb_u + nb
        su_row = (e * estride + (col // 128) * (cp // 4) * 512 + (col % 32) * 16
                  + ((col // 32) % 4) * 4)
        sg_row = (e * estride + (colg // 128) * (cp // 4) * 512 + (colg % 32) * 16
                  + ((colg // 32) % 4) * 4)
        for tile_r in hl.tile(0, cnt, block_size=brow):
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
            hv = gate * torch.sigmoid(gate) * up
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
    return h, hs


def dsv41_moe_combine_sh(y: torch.Tensor, topk_ids: torch.Tensor, id_base: int, e_count: int,
                         hs: torch.Tensor, wd: torch.Tensor, sd: torch.Tensor,
                         out: torch.Tensor):
    """out bf16 [M, H] = bf16((sum_k y[t*topk + k] over local ids, k ascending) + S[t]) with
    S = hs[t] . Wd^T on the shared down projection (wd float8_e4m3fn [H, I_sh], sd its
    F8_128x4 e8m0 scales): an M-tiled dot (token tile 16..64, so its work scales with M like
    the routed combine's)."""
    m, topk = topk_ids.size()
    hdim = out.size(1)
    ids = topk_ids.view(-1)
    isz = wd.size(1)
    nksd = isz // 32
    cpd = (nksd + 3) // 4 * 4
    hs3 = hs.view(m, nksd, 32)
    wd3 = wd.view(hdim, nksd, 32)
    ssf = sd.view(-1)
    bt = hl.register_block_size(16, 64)
    bh = hl.register_block_size(16, 256)
    bk = hl.register_block_size(1, min(8, nksd))
    for tile_t, tile_h in hl.tile([m, hdim], block_size=[bt, bh]):
        acc = hl.zeros([tile_t, tile_h], dtype=torch.float32)
        tok = torch.clamp(tile_t.index, max=m - 1)
        for k in hl.grid(topk):
            p = tok * topk + k
            loc = ids[p].to(torch.int32) - id_base
            ok = (loc >= 0) & (loc < e_count)
            acc = acc + torch.where(ok[:, None], y[p, tile_h], 0.0)
        row = torch.clamp(tile_h.index, max=hdim - 1)
        srow = (row // 128) * (cpd // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4
        sacc = hl.zeros([tile_t, tile_h], dtype=torch.float32)
        for tile_k in hl.tile(nksd, block_size=bk):
            a_s = hs3[tile_t, tile_k, :]
            a_s = a_s.reshape(a_s.size(0), -1)
            kk = torch.clamp(tile_k.index, max=nksd - 1)
            sc = (ssf[srow[:, None] + ((kk // 4) * 512 + kk % 4)[None, :]].to(torch.int32)
                  << 23).view(torch.float32)
            w_q = wd3[row, tile_k, :]
            w_s = (w_q.to(torch.float32) * sc[:, :, None]).to(torch.bfloat16)
            w_s = w_s.reshape(w_s.size(0), -1)
            sacc = hl.dot(a_s, w_s.T, acc=sacc)
        out[tile_t, tile_h] = (acc + sacc).to(torch.bfloat16)
    return out


def dsv41_moe_fc2_combine_sh(h: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                             w2: torch.Tensor, w2_sf: torch.Tensor, g2: torch.Tensor,
                             id_base: int, hs: torch.Tensor, wd: torch.Tensor,
                             sd: torch.Tensor, out: torch.Tensor):
    """Token-major fc2 + combine (nvfp4_moe_fc2_combine) with the shared down projection as one
    more GEMV slot per (token, H tile) after the routed slots: out[t] = bf16(sum_k ... + S[t])."""
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
    isz = wd.size(1)
    nksd = isz // 32
    cpd = (nksd + 3) // 4 * 4
    hs3 = hs.view(m, nksd, 32)
    wd3 = wd.view(hdim, nksd, 32)
    ssf = sd.view(-1)
    bh = hl.register_block_size(16, 128)
    bk = hl.register_block_size(1, min(16, nkb))
    bks = hl.register_block_size(1, min(8, nksd))
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
        s_row_d = (row // 128) * (cpd // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4
        sacc = hl.zeros([tile_h], dtype=torch.float32)
        for tile_ks in hl.tile(nksd, block_size=bks):
            a_s = hs3[t, tile_ks, :].to(torch.float32)
            kk = torch.clamp(tile_ks.index, max=nksd - 1)
            sc_d = (ssf[s_row_d[:, None] + ((kk // 4) * 512 + kk % 4)[None, :]].to(torch.int32)
                    << 23).view(torch.float32)
            w_q = wd3[row, tile_ks, :]
            w_s = w_q.to(torch.float32) * sc_d[:, :, None]
            sacc = sacc + torch.sum(torch.sum(w_s * a_s[None, :, :], dim=2), dim=1)
        out[t, tile_h] = (tot + sacc).to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------
# MXFP8 shared-expert operands + the exact spec (any device, f64 reference)
# ---------------------------------------------------------------------------
def _bf(t):
    return t.to(torch.bfloat16).to(t.dtype)


def quantize_mxfp8(w, rows32: bool = False):
    """f32 [N, K] -> (q float8_e4m3fn [N, K], s uint8 [N, K/32]): vLLM's quantizer
    (utils/mxfp8_utils._mxfp8_e4m3_quantize_torch: s = ceil(log2(amax / 448)) + 127),
    clamped into [S_MIN, S_MAX]. rows32: one scale per 32 rows x 32 columns, the
    checkpoint's block shape (vLLM repeat_interleaves it to one byte per row)."""
    n, k = w.shape
    blk = w.float().reshape(n, k // SCALE_BLOCK, SCALE_BLOCK)
    amax = blk.abs().amax(-1)
    if rows32:
        amax = amax.reshape(n // 32, 32, k // SCALE_BLOCK).amax(1).repeat_interleave(32, 0)
    s = (torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0)) + E8M0_BIAS).clamp(S_MIN, S_MAX)
    q = (blk / torch.exp2(s - E8M0_BIAS)[..., None]).reshape(n, k).to(torch.float8_e4m3fn)
    return q, s.to(torch.uint8)


def dequant_mxfp8(q, s, dtype=torch.float64):
    """The exact dequantized weight [N, K] (e4m3 x 2^(s-127))."""
    n, k = q.shape
    return (q.to(dtype).reshape(n, k // SCALE_BLOCK, SCALE_BLOCK)
            * torch.exp2(s.to(dtype) - E8M0_BIAS)[..., None]).reshape(n, k)


def swizzle_scales(s):
    """uint8 [R, C] -> the flat F8_128x4 layout of vLLM's swizzle_mxfp8_scale (the NVFP4
    lane's swizzle is the same permutation: pinned by tests/test_dsv41_moe_shared.py)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return nm.swizzle(s)


def unswizzle_scales(flat, r: int, c: int):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    return nm.unswizzle(flat.reshape(-1).view(torch.uint8), r, c)


def adversarial_mxfp8(q, s, gen):
    """In place scale/value stress on a quantized (q [N, K], s [N, K/32]): 8 % of the blocks all
    zero under a random scale byte (a stray scale must not leak into zeros), 4 % at the smallest
    accepted scale, 4 % at scale 246 with q zeroed (largest accepted byte, no overflow),
    5 % one element at the e4m3 maximum +-448 (an outlier block)."""
    n, kb = s.shape
    u = torch.rand(n, kb, generator=gen)
    qb = q.view(torch.uint8).reshape(n, kb, SCALE_BLOCK)
    zero = u < 0.08
    qb[zero] = 0
    s[zero] = torch.randint(S_MIN, S_MAX + 1, (int(zero.sum()),), generator=gen).to(torch.uint8)
    s[(u >= 0.08) & (u < 0.12)] = S_MIN
    big = (u >= 0.12) & (u < 0.16)
    qb[big] = 0
    s[big] = S_MAX
    out = (u >= 0.16) & (u < 0.21)
    pos = torch.randint(0, SCALE_BLOCK, (n, kb), generator=gen)
    sign = torch.randint(0, 2, (n, kb), generator=gen) * 0x80
    code = (0x7E + sign).to(torch.uint8)  # 0x7E = +448, 0xFE = -448 (e4m3fn max, not NaN)
    rows, cols = torch.nonzero(out, as_tuple=True)
    qb[rows, cols, pos[rows, cols]] = code[rows, cols]
    return q, s


def make_shared_problem(hdim: int, isz: int, seed: int = 0, device="cpu", limit: float = 10.0,
                        wstd: float = 0.02, adversarial: bool = False, rows32: bool = True):
    """Random MXFP8 shared expert in vLLM's post-processed layout: wgu float8_e4m3fn
    [2*isz, hdim] = [gate; up], wd [hdim, isz], scales in the F8_128x4 swizzle, plus the exact
    dequantized weights (f64) the references use."""
    gen = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    qs = []
    for shape in ((2 * isz, hdim), (hdim, isz)):
        w = torch.randn(*shape, generator=gen) * wstd
        q, s = quantize_mxfp8(w, rows32=rows32)
        if adversarial:
            adversarial_mxfp8(q, s, gen)
        qs.append((q, s))
    dev = torch.device(device)
    (qg, sg), (qd, sd) = qs
    return dict(wgu=qg.to(dev), sgu=swizzle_scales(sg).to(dev), wd=qd.to(dev),
                sd=swizzle_scales(sd).to(dev), limit=float(limit), isz=isz, hdim=hdim,
                deq_gu=dequant_mxfp8(qg, sg).to(dev), deq_d=dequant_mxfp8(qd, sd).to(dev))


def shared_h_ref(sp, x):
    """f64 [M, I_sh] = bf16-rounded silu(min(x.Wg^T, L)) * clamp(x.Wu^T, -L, L) (what hs holds)."""
    lim, i = sp["limit"], sp["isz"]
    gu = x.double() @ sp["deq_gu"].double().T
    gate = gu[:, :i].clamp(max=lim)
    up = gu[:, i:].clamp(-lim, lim)
    return _bf(gate * torch.sigmoid(gate) * up)


def shared_down_ref(sp, hs):
    """f64 [M, H] = hs . Wd^T."""
    return hs.double() @ sp["deq_d"].double().T


def shared_ref(sp, x):
    """f64 [M, H] shared expert partial of this lane's math (module doc)."""
    return shared_down_ref(sp, shared_h_ref(sp, x))


# ---------------------------------------------------------------------------
# serving: the fused plan (no oxide fallback exists for a clamp / E 384 layer)
# ---------------------------------------------------------------------------
BODIES_SH = {**hm.BODIES, FC1_SH: dsv41_moe_fc1_sh, COMBINE_SH: dsv41_moe_combine_sh,
             FC2_COMBINE_SH: dsv41_moe_fc2_combine_sh}


def lane_case_sh(e_count: int, hdim: int, idim: int, topk: int, m: int, limit: float,
                 e_global: int | None, isz: int) -> dict:
    """Case of the NEW kernels (base case + sh_inter); route_quant / fc2 use the base case."""
    return {**hm.lane_case(e_count, hdim, idim, topk, 0, m, limit, e_global), "sh_inter": int(isz)}


def base_case(case_sh: dict) -> dict:
    return {k: v for k, v in case_sh.items() if k != "sh_inter"}


def _case_of(kernel: str, case_sh: dict) -> dict:
    return case_sh if kernel in NEW_KERNELS else base_case(case_sh)


def missing_sh(case_sh: dict, mode: int) -> list:
    """Kernels of the fused plan with no shipped config for this bucket."""
    return [k for k in PLANS_SH[hm.plan_name(mode)] if hk.config(k, _case_of(k, case_sh)) is None]


def served_ms_sh(e_count, hdim, idim, topk, max_m, mode_for, limit, e_global, isz) -> dict:
    """{M: (bucket, plan)} for every M in 1..max_m the fused plan serves. A gap is a startup error
    naming every missing (bucket, plan, kernel): there is no oxide kernel for this layer to fall
    back to, and a silent unfused M below the ceiling would hide the very thing being measured."""
    out, gaps = {}, {}
    for m in range(1, max_m + 1):
        case = lane_case_sh(e_count, hdim, idim, topk, m, limit, e_global, isz)
        mode = mode_for(m)
        body = hm.variant_for(base_case(case))
        if body != hm.DEQUANT:  # the shared expert rides inside the dequant bodies only
            raise RuntimeError(
                f"{MARKER} {KNOB}=1: M={m} (bucket {case['numtokens']}) is served by the "
                f"{body!r} body variant ({hm.VARIANT_KERNEL} row), but the fused shared expert "
                f"lives in the dequant bodies: lower SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M below "
                f"M={m} or select dequant for that bucket")
        lack = missing_sh(case, mode)
        if lack:
            gaps.setdefault((case["numtokens"], hm.plan_name(mode)), lack)
        else:
            out[m] = (case["numtokens"], hm.plan_name(mode))
    if gaps:
        raise RuntimeError(
            f"{MARKER} {KNOB}=1 but no tuned config on {hk.platform()} for E={e_count}/"
            f"{e_global or e_count} H={hdim} I={idim} I_sh={isz} topk={topk} limit={limit}: "
            + "; ".join(f"M<={b} {p}: {', '.join(k)}" for (b, p), k in sorted(gaps.items()))
            + " (tune them in the autotune lab, ship configs/helion/<kernel>/<platform>.json, "
            "or lower SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M)")
    return out


@functools.lru_cache(maxsize=None)
def _kernels_sh(case_key: str, plan: str, untuned_ok: bool = False):
    import json
    case = json.loads(case_key)
    out = []
    for k in PLANS_SH[plan]:
        c = _case_of(k, case)
        if untuned_ok and hk.config(k, c) is None:
            hk.log(f"dsv41-moe-shared UNTUNED {k} {hk.case_key(c)}: Helion default config "
                   "(bench/oracle only)")
            out.append(helion.kernel(BODIES_SH[k], autotune_effort="none", static_shapes=False))
        else:
            out.append(hk.compiled(BODIES_SH[k], k, c))
    return out


def sh_views(wgu, sgu, wd, sd) -> tuple:
    """The shared operands as the bodies read them: weights float8_e4m3fn, scales uint8."""
    f8 = lambda t: t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)
    u8 = lambda t: t if t.dtype == torch.uint8 else t.view(torch.uint8)
    return f8(wgu), u8(sgu), f8(wd), u8(sd)


def run_plan_sh(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out,
                a1_gscale, a2_gscale, id_base, limit, sh):
    """One routed + shared MoE forward on the fused plan's kernels -> out (bf16 [M, H]):
    out = bf16(sum_k routed_k + shared). sh = sh_views(...)."""
    wgu, sgu, wd, sd = sh
    e_count = w13.shape[0]
    f8 = lambda t: t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)
    xdq, ps, seg, seg_e = kerns[0](x, topk_ids, hm._f32(a1_gscale), hm._div6(a1_gscale),
                                   e_count, id_base)
    h, hs = kerns[1](xdq, ps, seg, seg_e, w13, f8(w13_sf), g1, hm._f32(a2_gscale),
                     hm._div6(a2_gscale), topk_ids.shape[1], float(limit), x, wgu, sgu)
    if plan == "fused3":
        return kerns[2](h, topk_ids, topk_w, w2, f8(w2_sf), g2, id_base, hs, wd, sd, out)
    y = kerns[2](h, ps, seg, seg_e, topk_w, w2, f8(w2_sf), g2)
    return kerns[3](y, topk_ids, id_base, e_count, hs, wd, sd, out)


class FusedPlan:
    """The serving entry of one layer: M -> (bucketed case, plan) -> compiled kernels -> run.
    Built at load (configs resolved), every served M run once by warm() (Triton specializes on
    sizes: each M compiles its own variant), then called per forward with no allocation but the
    kernels' own torch.empty (the graph pool under capture). An M that was not warmed is refused
    inside a CUDA-graph capture: a Triton compile there would be the capture-time crash the
    g14 boot hit."""

    def __init__(self, e_count, hdim, idim, topk, limit, e_global, isz, max_m, mode_for,
                 untuned_ok: bool = False):
        self.dims = (int(e_count), int(hdim), int(idim), int(topk), float(limit),
                     int(e_global), int(isz))
        self.max_m, self.untuned_ok, self.mode_for = int(max_m), untuned_ok, mode_for
        self._hot = {}  # M -> (kernels, plan): filled by the first call outside any capture
        if untuned_ok:  # bench / oracle on synthetic weights: Helion's default config per kernel
            self.served = {m: (hk.bucket(m, hm.CLAMP_BUCKETS), hm.plan_name(mode_for(m)))
                           for m in range(1, self.max_m + 1)}
        else:
            self.served = served_ms_sh(e_count, hdim, idim, topk, self.max_m, mode_for, limit,
                                       e_global, isz)

    def key(self, m: int):
        e, hd, i, k, lim, eg, isz = self.dims
        case = lane_case_sh(e, hd, i, k, m, lim, eg, isz)
        return hk.case_key(case), self.served[m][1]

    def kernels(self, m: int):
        ck, plan = self.key(m)
        return _kernels_sh(ck, plan, self.untuned_ok), plan, ck

    def warm(self, run_m) -> int:
        """Run every served M once now (run_m(m) calls __call__ on real shapes); returns the
        number of distinct (bucket, plan) pairs."""
        for m in sorted(self.served):
            run_m(m)
        return len({self.key(m) for m in self.served})

    def __call__(self, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, a1g, a2g, id_base,
                 sh, out=None):
        m = x.shape[0]
        hot = self._hot.get(m)
        if hot is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{MARKER} M={m} was not warmed before CUDA-graph capture "
                                   "(a Triton compile inside a capture)")
            kerns, plan, _ = self.kernels(m)
            hot = self._hot[m] = (kerns, plan)
        if out is None:
            out = torch.empty(m, x.shape[1], dtype=torch.bfloat16, device=x.device)
        return run_plan_sh(hot[0], hot[1], x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2,
                           out, a1g, a2g, id_base, self.dims[4], sh)


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py): the three new kernels
# ---------------------------------------------------------------------------
SHARED_H_TOL = 2e-3  # hs vs the f64 spec (same bf16 rounding on both sides; f32 vs f64 flips)
DS_N_SHARED = 1  # n_shared_experts of the V4 family (config.json)


def sh_inter_of(model: dict, layout: dict) -> int:
    """Shared-expert intermediate of one TP rank: whole per-expert I x n_shared / tp."""
    ep_layout = "experts_global" in layout
    whole = layout["moe_intermediate"] * (1 if ep_layout else hm.DS_TP)
    n = int(model.get("n_shared_experts") or DS_N_SHARED)
    return whole * n // int(model.get("tp") or hm.DS_TP)


def cases(model: dict, kernel: str | None = None) -> list:
    """The routed lab layouts (hm._layouts: DeepSeek-V4 EP8 [+ no-EP TP8 opt-in]) x the M buckets
    the kernel's plan serves, each with the shared intermediate `sh_inter`."""
    want = {FC2_COMBINE_SH: lambda m: m <= hm.FUSED3_MAX,
            COMBINE_SH: lambda m: m >= hm.FRONT4_MIN}.get(kernel, lambda m: True)
    out = []
    for lay in hm._layouts(model):
        if "limit" not in lay:
            continue
        isz = sh_inter_of(model, lay)
        if isz % SCALE_BLOCK:
            continue
        out += [{**lay, "numtokens": m, "sh_inter": isz} for m in hm.CLAMP_BUCKETS if want(m)]
    return out


@functools.lru_cache(maxsize=2)
def _shared(hidden, isz, seed, limit, device):
    return make_shared_problem(hidden, isz, seed=seed, device=device, limit=limit)


def _sp(case, seed):
    return _shared(case["hidden"], case["sh_inter"], seed % 2, float(case["limit"]),
                   hm._device())


def _deq(wq, sflat, n, k):
    """Dequantized f64 [n, k] from the kernel's own operands (weight fp8, swizzled scales)."""
    return dequant_mxfp8(wq, unswizzle_scales(sflat, n, k // SCALE_BLOCK))


def _hs_bf16(case, seed, x):
    return shared_h_ref(_sp(case, seed), x).bfloat16()


def _make_fc1_sh(case, seed):
    p, x, ids, _, xdq, ps, seg, seg_e = hm._stage_inputs(case, seed)
    sp = _sp(case, seed)
    return (xdq, ps, seg, seg_e, p["w13"], p["w13_sf"], p["g1"], hm._f32(p["a2g"]),
            hm._div6(p["a2g"]), ids.shape[1], float(case["limit"]), x, *sh_views(
                sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])[:2])


def _make_combine_sh(case, seed):
    p, x, ids, tw, xdq, ps, seg, seg_e = hm._stage_inputs(case, seed)
    h, _ = hm._h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    y, _ = hm._y_ref(h, ps, seg, seg_e, tw, p["w2"], p["w2_sf"], p["g2"])
    sp = _sp(case, seed)
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    _, _, wd, sd = sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])
    return (y.float(), ids, hm.ep_base(case), p["w2"].shape[0], _hs_bf16(case, seed, x), wd, sd,
            out)


def _make_fc2_combine_sh(case, seed):
    p, x, ids, tw, xdq, ps, seg, seg_e = hm._stage_inputs(case, seed)
    h, _ = hm._h_ref(p, xdq, ps, seg, seg_e, ids.shape[1])
    sp = _sp(case, seed)
    out = torch.empty(ids.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=h.device)
    _, _, wd, sd = sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])
    return (h, ids, tw, p["w2"], p["w2_sf"], p["g2"], hm.ep_base(case),
            _hs_bf16(case, seed, x), wd, sd, out)


def _ref_fc1_sh(xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, topk, limit, x, wgu, sgu):
    routed = hm._ref_fc1(xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, 0, topk, limit)
    n, hdim = wgu.shape
    sp = {"deq_gu": _deq(wgu, sgu, n, hdim), "isz": n // 2, "limit": limit}
    return routed, shared_h_ref(sp, x)


def _check_fc1_sh(out, ref):
    routed_ok, routed_msg = hm._check_rows("hq", rel=False)(out[0], ref[0])
    got, want = out[1].float(), ref[1].float()
    rel = hm._rel(got, want)
    ok = routed_ok and bool(torch.isfinite(got).all()) and rel <= SHARED_H_TOL
    return ok, f"{routed_msg}; shared hs rel_vs_f64={rel:.2e} (tol {SHARED_H_TOL})"


def _shared_s_ref(hs, wd, sd, m):
    return hs.double() @ _deq(wd, sd, wd.shape[0], wd.shape[1]).T


def _ref_combine_sh(y, topk_ids, id_base, e_count, hs, wd, sd, out):
    """f64 (sum_k y over local ids) + hs . Wd^T."""
    m, topk = topk_ids.shape
    loc = topk_ids.long() - id_base
    ok = (loc >= 0) & (loc < e_count)
    y3 = y[:m * topk].view(m, topk, -1).double()
    routed = (y3 * ok[:, :, None]).sum(1)
    return routed + _shared_s_ref(hs, wd, sd, m)


def _ref_fc2_combine_sh(h, topk_ids, topk_w, w2, w2_sf, g2, id_base, hs, wd, sd, out):
    return hm._ref_fc2_combine(h, topk_ids, topk_w, w2, w2_sf, g2, id_base, out) \
        + _shared_s_ref(hs, wd, sd, topk_ids.shape[0])


def _check_end_sh(out, ref):
    return hm._check_end(out, ref)


# baselines: the UNFUSED composition of the same math (module doc): the base stage kernel (its
# shipped config, Helion's default otherwise) + the shared MLP as torch bf16 GEMMs on the
# dequantized weights (2x the bytes of the FP8 ones). The verdict against stock vLLM (CUTLASS
# mm_mxfp8 on the aux stream) is kernels/dsv41_moe_shared.py `bench`.
@functools.lru_cache(maxsize=None)
def _base_stage(name: str, case_key: str):
    import json
    case = json.loads(case_key)
    if hk.config(name, case) is not None:
        return hk.compiled(hm.BODIES[name], name, case)
    return helion.kernel(hm.BODIES[name], autotune_effort="none", static_shapes=False)


def _bf16_deq(wq, sflat):
    n, k = wq.shape
    return _deq(wq, sflat, n, k).bfloat16()


def _baseline_args(case, seed, make):
    """(stage args..., base case key, dequantized bf16 shared weights) for the unfused arms."""
    args = make(case, seed)
    sp = _sp(case, seed)
    return (*args, hk.case_key(base_case(case)), _bf16_deq(sp["wgu"], sp["sgu"]),
            _bf16_deq(sp["wd"], sp["sd"]), float(case["limit"]))


def _torch_shared_h(x, deq_gu, limit):
    i = deq_gu.shape[0] // 2
    gu = (x @ deq_gu.T).float()
    gate = gu[:, :i].clamp(max=limit)
    up = gu[:, i:].clamp(-limit, limit)
    return (gate * torch.sigmoid(gate) * up).bfloat16()


def _baseline_fc1_sh(*a):
    (xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, topk, limit, x, _wgu, _sgu,
     ck, deq_gu, _deq_d, _lim) = a
    h = _base_stage(hm.FC1, ck)(xdq, ps, seg, seg_e, w13, w13_sf, g1, a2g, a2g6, 0, topk, limit)
    return h, _torch_shared_h(x, deq_gu, limit)


def _baseline_combine_sh(*a):
    y, ids, base, e_count, hs, _wd, _sd, out, ck, _deq_gu, deq_d, _lim = a
    routed = _base_stage(hm.COMBINE, ck)(y, ids, base, e_count, out)
    return (routed.float() + (hs @ deq_d.T).float()).bfloat16()


def _baseline_fc2_combine_sh(*a):
    h, ids, tw, w2, w2_sf, g2, base, hs, _wd, _sd, out, ck, _deq_gu, deq_d, _lim = a
    routed = _base_stage(hm.FC2_COMBINE, ck)(h, ids, tw, w2, w2_sf, g2, base, out)
    return (routed.float() + (hs @ deq_d.T).float()).bfloat16()


_BASELINE = ("unfused: base Helion stage kernel (shipped config else default) + torch bf16 GEMMs "
             "on the dequantized shared weights; vs stock vLLM: kernels/dsv41_moe_shared.py bench")


def _entry_sh(name, body, make, reference, check, baseline):
    return {"name": name, "body": body, "cases": functools.partial(cases, kernel=name),
            "make": hm.remember(name, make), "reference": reference, "check": check,
            "settings": {"autotune_baseline_accuracy_check_fn": hm.autotune_check(name)},
            "baseline": baseline,
            "make_baseline": lambda case, seed: _baseline_args(case, seed, make),
            "baseline_name": _BASELINE, "replaces": []}


KERNELS = [
    _entry_sh(FC1_SH, dsv41_moe_fc1_sh, _make_fc1_sh, _ref_fc1_sh, _check_fc1_sh,
              _baseline_fc1_sh),
    _entry_sh(COMBINE_SH, dsv41_moe_combine_sh, _make_combine_sh, _ref_combine_sh,
              _check_end_sh, _baseline_combine_sh),
    _entry_sh(FC2_COMBINE_SH, dsv41_moe_fc2_combine_sh, _make_fc2_combine_sh,
              _ref_fc2_combine_sh, _check_end_sh, _baseline_fc2_combine_sh),
]
