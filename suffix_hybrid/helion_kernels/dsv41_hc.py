# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 hyper-connection (mHC) boundary in Helion (SM120).

Replaces vLLM's ``mhc_fused_post_pre_delayed_tilelang`` (nightly 39983a5e7,
vllm/model_executor/kernels/mhc/tilelang.py). On SM120 the SM100-only paths (DeepGEMM
prenorm, mega-mHC, mHC side stream, fused all-reduce) are all off, so each of the 77 sublayer
boundaries per step this module serves (2 x 40 layers, minus the attention pre of layers 0, 1 and
14: the broadcast embedding and the two Engram layers use other entry points) runs

    <= 32 tokens : [post + split-K prenorm GEMM] -> [MHC_PRE_NORM epilogue]      2 launches
    >  32 tokens : [post] -> [prenorm GEMM] -> [MHC_PRE_NORM epilogue]           3 launches

Nothing here fuses across an all-reduce (the boundary starts after the
sublayer's all-reduce) and nothing synchronises CTAs: no grid barrier, no split-K
semaphore, no atomics. Every path is deterministic (the 8 TP ranks recompute the mixes
independently and must agree bit for bit).

K1 contract (V4.1 per rank; mHC is replicated, hidden H = 5120, hc_mult 4)
  x [T, H] bf16 sublayer output            res [T, 4, H] bf16 residual streams
  post [T, 4] f32, comb [T, 4, 4] f32 [t, i, j]   the previous boundary's mixes
  fn [24, 4H] f32 (rows 0:4 pre, 4:8 post, 8:24 comb[jp, k]), scale [3] f32, base [24] f32
  pre_in [T, 4] f32 | None (None: stream 0)       norm_w [H] bf16
Math (= mhc_post_torch + mhc_pre_delayed_torch + RMSNorm of vLLM's torch.py, with the
BF16 rounding boundaries of the TileLang kernels):
  rc[t,j,:]  = bf16( sum_i comb[t,i,j] res[t,i,:] + post[t,j] x[t,:] )          residual_cur
  mixes      = (flat(rc) @ fn^T) * rsqrt(sum(rc^2) / (4H) + rms_eps)             f32 statistics
  pre'       = sigmoid(mixes[:4] s0 + b[:4]) + pre_eps                           next pre mix
  post'      = sigmoid(mixes[4:8] s1 + b[4:8]) * post_mult
  comb'      = sinkhorn(mixes[8:] s2 + b[8:]): row softmax + eps, column normalise, then
               (repeat-1) x (row, column) normalisations, eps in every denominator
  pooled     = bf16( sum_j pre_in[t,j] rc[t,j,:] )   (pre_in None: rc[t,0,:])
  layer_in   = bf16( pooled * rsqrt(mean(pooled^2) + norm_eps) * norm_w )
  aux        = bf16( mean_j rc[t,j,:] )                                          (capture_aux)
  q, qs      = MXFP8 quantisation of layer_in (FlashInfer UE8M0 scales, F8_128x4 swizzle),
               optional: the next MXFP8 linear then skips its activation-quant launch.

Fusion boundaries kept in the search space (the lab / bench picks per M bucket; stock stays
the baseline and the fallback):
  split2    dsv41_hc_post_partials + dsv41_hc_epilogue   2 launches, split-K partials through a
            torch-allocated scratch: -1 launch vs stock above 32 tokens, equal below.
  resident  dsv41_hc_resident                            1 launch, one program per token tile
            streams the whole fn (1.97 MB f32) itself. The aggressive-fusion NEGATIVE CONTROL
            (a more aggressively fused HC path already lost once on silicon): at small T it is
            bounded by one SM's L2 bandwidth, at large T by the program count.
  [post | GEMM | epilogue] (3 launches) is the stock path above 32 tokens.

Search spaces (bounded so an LFBO full-effort run does not spend its compiles on spilling
tiles; all block sizes are powers of two, enforced by Helion): K1 bt 1..16 tokens x bh 32..256
columns x ns 1..16 splits; K2 bt 1..8 x bg 1..8 groups of 32 columns; resident bt 1..16 x
bh 32..256 x bg 1..8; num_warps / num_stages / indexing stay in Helion's default space.

Knobs: SUFFIX_HELION_DSV41_HC = oxide (default = stock vLLM) | auto | helion;
SUFFIX_DSV41_HC_BOUNDARY = split2 (default) | resident; Wiring:
suffix_hybrid/kernels/dsv41_fuse_hook.py. Boot gates: <kernel>_capture_replay (CLI below).
"""
import os
import sys
from types import SimpleNamespace

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_common as cm
from suffix_hybrid.helion_kernels.dsv41_common import (  # noqa: F401  (traced into the bodies)
    mxfp8_group_quant, mxfp8_scale_offsets, sinkhorn)

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU host: the module stays importable (lab ABI, tests)
    helion = hl = None
CE = hl.constexpr if hl is not None else int

ENV = "SUFFIX_HELION_DSV41_HC"
BOUNDARY_ENV = "SUFFIX_DSV41_HC_BOUNDARY"
K_POST = "dsv41_hc_post_partials"
K_EPI = "dsv41_hc_epilogue"
K_RES = "dsv41_hc_resident"
MARKER = "[suffix dsv41-hc]"
HC = 4
MIX = HC * (HC + 2)
BUCKETS = (1, 2, 4, 6, 12, 24, 48, 96, 192)   # decode tokens/step: 6 per sequence at DSpark k=5
MAX_TOKENS = BUCKETS[-1]   # above: stock (prefill chunks / mixed batches), counted in the summary
SEARCH_SPACE = {            # (min, max) per registered block size, in registration order
    K_POST: {"bt": (1, 16), "bh": (32, 256), "ns": (1, 16)},
    K_EPI: {"bt": (1, 8), "bg": (1, 8)},
    K_RES: {"bt": (1, 16), "bh": (32, 256), "bg": (1, 8)},
}
DEFAULT_CONSTS = {"rms_eps": 1e-6, "pre_eps": 1e-6, "sk_eps": 1e-6, "post_mult": 2.0,
                  "norm_eps": 1e-6, "repeat": 20}


# ---------------------------------------------------------------------------
# traced helpers (pure tensor ops; unsqueeze, never [:, None], on a helper's inputs: Inductor cannot
# lower a slice of a graph input)
# ---------------------------------------------------------------------------
def _post_stream(c0, c1, c2, c3, p, r0, r1, r2, r3, xh):
    """rc_j = bf16( sum_i c_ij r_i + p_j x ): c_i [bt], p [bt], r_i / xh [bt, bh] f32; i ascending."""
    return (c0.unsqueeze(1) * r0 + c1.unsqueeze(1) * r1 + c2.unsqueeze(1) * r2
            + c3.unsqueeze(1) * r3 + p.unsqueeze(1) * xh).to(torch.bfloat16)


def _mix_acc(a, f0, f1, f2, f3, w0, w1, w2, w3):
    """a [bt, 4] += sum_j sum_h f_j [bt, bh] * w_j [4, bh]  (one output group of 4 mixes)."""
    return a + (torch.sum(f0.unsqueeze(1) * w0.unsqueeze(0), dim=2)
                + torch.sum(f1.unsqueeze(1) * w1.unsqueeze(0), dim=2)
                + torch.sum(f2.unsqueeze(1) * w2.unsqueeze(0), dim=2)
                + torch.sum(f3.unsqueeze(1) * w3.unsqueeze(0), dim=2))


def _pool(p0, p1, p2, p3, r0, r1, r2, r3):
    """bf16-rounded collapse sum_j p_j r_j of [bt, bg, 32] streams with [bt] weights, j ascending."""
    q0 = p0.unsqueeze(1).unsqueeze(2)
    q1 = p1.unsqueeze(1).unsqueeze(2)
    q2 = p2.unsqueeze(1).unsqueeze(2)
    q3 = p3.unsqueeze(1).unsqueeze(2)
    return (q0 * r0 + q1 * r1 + q2 * r2 + q3 * r3).to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# Helion bodies (module level, undecorated: the lab and hk.compiled wrap them;
# only literals / constexpr flags inside, they are traced)
# ---------------------------------------------------------------------------
def dsv41_hc_post_partials(x: torch.Tensor, res: torch.Tensor, post: torch.Tensor,
                           comb: torch.Tensor, w_pre: torch.Tensor, w_post: torch.Tensor,
                           w_comb: torch.Tensor):
    """K1. x [T,H] bf16, res [T,4,H] bf16, post [T,4] f32, comb [T,4,4] f32 [t,i,j];
    w_pre / w_post = fn[0:4] / fn[4:8] viewed [i, j, H]; w_comb = fn[8:24] viewed [jp, k, j, H].
    Returns (rc [T,4,H] bf16, part_sq [ns,T], part_pre [ns,T,4], part_post [ns,T,4],
    part_comb [ns,T,4,4]) f32: the post-mapped residual and the split-K partials of
    sum(rc^2) and of the 24 mixes (split s = columns [s*per, (s+1)*per))."""
    T = x.size(0)
    H = hl.specialize(x.size(1))
    hc = hl.specialize(res.size(1))
    bt = hl.register_block_size(1, 16)
    bh = hl.register_block_size(32, 256)
    ns = hl.register_block_size(1, 16)
    rc = torch.empty_like(res)
    part_sq = torch.empty([ns, T], dtype=torch.float32, device=x.device)
    part_pre = torch.empty([ns, T, hc], dtype=torch.float32, device=x.device)
    part_post = torch.empty([ns, T, hc], dtype=torch.float32, device=x.device)
    part_comb = torch.empty([ns, T, hc, hc], dtype=torch.float32, device=x.device)
    per = ((H + ns - 1) // ns + bh - 1) // bh * bh
    for tile_t, tile_s in hl.tile([T, ns], block_size=[bt, 1]):
        h0 = tile_s.begin * per
        h1 = min(h0 + per, H)
        c00 = comb[tile_t, 0, 0]
        c01 = comb[tile_t, 0, 1]
        c02 = comb[tile_t, 0, 2]
        c03 = comb[tile_t, 0, 3]
        c10 = comb[tile_t, 1, 0]
        c11 = comb[tile_t, 1, 1]
        c12 = comb[tile_t, 1, 2]
        c13 = comb[tile_t, 1, 3]
        c20 = comb[tile_t, 2, 0]
        c21 = comb[tile_t, 2, 1]
        c22 = comb[tile_t, 2, 2]
        c23 = comb[tile_t, 2, 3]
        c30 = comb[tile_t, 3, 0]
        c31 = comb[tile_t, 3, 1]
        c32 = comb[tile_t, 3, 2]
        c33 = comb[tile_t, 3, 3]
        p0 = post[tile_t, 0]
        p1 = post[tile_t, 1]
        p2 = post[tile_t, 2]
        p3 = post[tile_t, 3]
        sq = hl.zeros([tile_t], dtype=torch.float32)
        a_pre = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_post = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c0 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c1 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c2 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c3 = hl.zeros([tile_t, hc], dtype=torch.float32)
        for tile_h in hl.tile(h0, h1, block_size=bh):
            xh = x[tile_t, tile_h].to(torch.float32)               # [bt, bh]
            r0 = res[tile_t, 0, tile_h].to(torch.float32)
            r1 = res[tile_t, 1, tile_h].to(torch.float32)
            r2 = res[tile_t, 2, tile_h].to(torch.float32)
            r3 = res[tile_t, 3, tile_h].to(torch.float32)
            q0 = _post_stream(c00, c10, c20, c30, p0, r0, r1, r2, r3, xh)     # rc stream j = 0..3
            q1 = _post_stream(c01, c11, c21, c31, p1, r0, r1, r2, r3, xh)
            q2 = _post_stream(c02, c12, c22, c32, p2, r0, r1, r2, r3, xh)
            q3 = _post_stream(c03, c13, c23, c33, p3, r0, r1, r2, r3, xh)
            rc[tile_t, 0, tile_h] = q0
            rc[tile_t, 1, tile_h] = q1
            rc[tile_t, 2, tile_h] = q2
            rc[tile_t, 3, tile_h] = q3
            f0 = q0.to(torch.float32)
            f1 = q1.to(torch.float32)
            f2 = q2.to(torch.float32)
            f3 = q3.to(torch.float32)
            sq = sq + torch.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3, dim=1)
            a_pre = _mix_acc(a_pre, f0, f1, f2, f3, w_pre[:, 0, tile_h], w_pre[:, 1, tile_h],
                             w_pre[:, 2, tile_h], w_pre[:, 3, tile_h])
            a_post = _mix_acc(a_post, f0, f1, f2, f3, w_post[:, 0, tile_h], w_post[:, 1, tile_h],
                              w_post[:, 2, tile_h], w_post[:, 3, tile_h])
            a_c0 = _mix_acc(a_c0, f0, f1, f2, f3, w_comb[0, :, 0, tile_h], w_comb[0, :, 1, tile_h],
                            w_comb[0, :, 2, tile_h], w_comb[0, :, 3, tile_h])
            a_c1 = _mix_acc(a_c1, f0, f1, f2, f3, w_comb[1, :, 0, tile_h], w_comb[1, :, 1, tile_h],
                            w_comb[1, :, 2, tile_h], w_comb[1, :, 3, tile_h])
            a_c2 = _mix_acc(a_c2, f0, f1, f2, f3, w_comb[2, :, 0, tile_h], w_comb[2, :, 1, tile_h],
                            w_comb[2, :, 2, tile_h], w_comb[2, :, 3, tile_h])
            a_c3 = _mix_acc(a_c3, f0, f1, f2, f3, w_comb[3, :, 0, tile_h], w_comb[3, :, 1, tile_h],
                            w_comb[3, :, 2, tile_h], w_comb[3, :, 3, tile_h])
        part_sq[tile_s.begin, tile_t] = sq
        part_pre[tile_s.begin, tile_t, :] = a_pre
        part_post[tile_s.begin, tile_t, :] = a_post
        part_comb[tile_s.begin, tile_t, 0, :] = a_c0
        part_comb[tile_s.begin, tile_t, 1, :] = a_c1
        part_comb[tile_s.begin, tile_t, 2, :] = a_c2
        part_comb[tile_s.begin, tile_t, 3, :] = a_c3
    return rc, part_sq, part_pre, part_post, part_comb


def dsv41_hc_epilogue(rc: torch.Tensor, part_sq: torch.Tensor, part_pre: torch.Tensor,
                      part_post: torch.Tensor, part_comb: torch.Tensor, scale: torch.Tensor,
                      b_pre: torch.Tensor, b_post: torch.Tensor, b_comb: torch.Tensor,
                      pre_in: torch.Tensor, norm_w: torch.Tensor, rms_eps: float,
                      pre_eps: float, sk_eps: float, post_mult: float, norm_eps: float,
                      repeat: CE, use_pre_in: CE, write_aux: CE, write_q: CE):
    """K2. rc and the K1 partials -> (post' [T,4], comb' [T,4,4], layer_in [T,H] bf16,
    pre' [T,4], aux [T|0,H] bf16, q [T|0,H] e4m3, qs scale bytes). One program per token
    tile; pre_in is only read when use_pre_in (else the stream-0 selector is used)."""
    T = rc.size(0)
    hc = hl.specialize(rc.size(1))
    H = hl.specialize(rc.size(2))
    G = H // 32
    bt = hl.register_block_size(1, 8)
    bg = hl.register_block_size(1, 8)
    post_o = torch.empty([T, hc], dtype=torch.float32, device=rc.device)
    comb_o = torch.empty([T, hc, hc], dtype=torch.float32, device=rc.device)
    pre_o = torch.empty([T, hc], dtype=torch.float32, device=rc.device)
    layer_in = torch.empty([T, H], dtype=torch.bfloat16, device=rc.device)
    if write_aux:
        aux = torch.empty([T, H], dtype=torch.bfloat16, device=rc.device)
    else:
        aux = torch.empty([0, H], dtype=torch.bfloat16, device=rc.device)
    if write_q:
        q = torch.empty([T, H], dtype=torch.float8_e4m3fn, device=rc.device)
        qs = torch.empty([(T + 127) // 128 * 128 * ((G + 3) // 4) * 4], dtype=torch.uint8,
                         device=rc.device)
    else:
        q = torch.empty([0, H], dtype=torch.float8_e4m3fn, device=rc.device)
        qs = torch.empty([0], dtype=torch.uint8, device=rc.device)
    rc4 = rc.view(T, hc, G, 32)
    li3 = layer_in.view(T, G, 32)
    ax3 = aux.view(aux.size(0), G, 32)
    q3 = q.view(q.size(0), G, 32)
    nw2 = norm_w.view(G, 32)
    for tile_t in hl.tile(T, block_size=bt):
        sqv = torch.sum(part_sq[:, tile_t], 0)                              # [bt]
        m_pre = torch.sum(part_pre[:, tile_t, :], 0)                        # [bt, 4]
        m_post = torch.sum(part_post[:, tile_t, :], 0)
        m_comb = torch.sum(part_comb[:, tile_t, :, :], 0)                   # [bt, 4, 4]
        rms = torch.rsqrt(sqv / (hc * H) + rms_eps)
        s0 = scale[0]
        s1 = scale[1]
        s2 = scale[2]
        pre_o[tile_t, :] = torch.sigmoid(m_pre * rms[:, None] * s0 + b_pre[:]) + pre_eps
        post_o[tile_t, :] = torch.sigmoid(m_post * rms[:, None] * s1 + b_post[:]) * post_mult
        comb_o[tile_t, :, :] = sinkhorn(m_comb * rms[:, None, None] * s2 + b_comb[:, :], sk_eps,
                                        repeat)
        if use_pre_in:
            p0 = pre_in[tile_t, 0]
            p1 = pre_in[tile_t, 1]
            p2 = pre_in[tile_t, 2]
            p3 = pre_in[tile_t, 3]
        else:
            p0 = hl.full([tile_t], 1.0, dtype=torch.float32)
            p1 = hl.zeros([tile_t], dtype=torch.float32)
            p2 = hl.zeros([tile_t], dtype=torch.float32)
            p3 = hl.zeros([tile_t], dtype=torch.float32)
        ssq = hl.zeros([tile_t, 1, 1], dtype=torch.float32)
        for tile_g in hl.tile(G, block_size=bg):
            r0 = rc4[tile_t, 0, tile_g, :].to(torch.float32)                # [bt, bg, 32]
            r1 = rc4[tile_t, 1, tile_g, :].to(torch.float32)
            r2 = rc4[tile_t, 2, tile_g, :].to(torch.float32)
            r3 = rc4[tile_t, 3, tile_g, :].to(torch.float32)
            pooled = _pool(p0, p1, p2, p3, r0, r1, r2, r3)
            ssq = ssq + torch.sum(torch.sum(pooled * pooled, dim=2, keepdim=True), dim=1, keepdim=True)
        rstd = torch.rsqrt(ssq / H + norm_eps)                              # [bt, 1, 1]
        for tile_g in hl.tile(G, block_size=bg):
            r0 = rc4[tile_t, 0, tile_g, :].to(torch.float32)
            r1 = rc4[tile_t, 1, tile_g, :].to(torch.float32)
            r2 = rc4[tile_t, 2, tile_g, :].to(torch.float32)
            r3 = rc4[tile_t, 3, tile_g, :].to(torch.float32)
            pooled = _pool(p0, p1, p2, p3, r0, r1, r2, r3)
            ly = (pooled * rstd * nw2[tile_g, :].to(torch.float32)).to(torch.bfloat16)   # [bt, bg, 32]
            li3[tile_t, tile_g, :] = ly
            if write_aux:
                ax3[tile_t, tile_g, :] = ((((r0 + r1) + r2) + r3) / hc).to(torch.bfloat16)
            if write_q:
                qv, sf = mxfp8_group_quant(ly.to(torch.float32))
                q3[tile_t, tile_g, :] = qv
                qs[mxfp8_scale_offsets(tile_t.index, tile_g.index, G)] = sf.to(torch.uint8)
        if write_q:
            # F8_128x4 scale padding must be zero like the stock producer writes it: the padding
            # group columns of this tile's rows, and (last tile only) the padding rows. Inner
            # loops of the one top-level loop: Helion refuses a second top-level write of qs.
            for tile_gp in hl.tile(G, (G + 3) // 4 * 4, block_size=4):
                qs[mxfp8_scale_offsets(tile_t.index, tile_gp.index, G)] = hl.zeros(
                    [tile_t, tile_gp], dtype=torch.uint8)
            if tile_t.begin + bt >= T:
                for tile_r in hl.tile(T, (T + 127) // 128 * 128, block_size=32):
                    for tile_gp in hl.tile((G + 3) // 4 * 4, block_size=32):
                        qs[mxfp8_scale_offsets(tile_r.index, tile_gp.index, G)] = hl.zeros(
                            [tile_r, tile_gp], dtype=torch.uint8)
    return post_o, comb_o, layer_in, pre_o, aux, q, qs


def dsv41_hc_resident(x: torch.Tensor, res: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
                      w_pre: torch.Tensor, w_post: torch.Tensor, w_comb: torch.Tensor,
                      scale: torch.Tensor, b_pre: torch.Tensor, b_post: torch.Tensor,
                      b_comb: torch.Tensor, pre_in: torch.Tensor, norm_w: torch.Tensor,
                      rms_eps: float, pre_eps: float, sk_eps: float, post_mult: float,
                      norm_eps: float, repeat: CE, use_pre_in: CE, write_aux: CE, write_q: CE):
    """K0 (aggressive-fusion negative control): the whole boundary in ONE launch, one program
    per token tile, no cross-program state. Returns (rc, post', comb', layer_in, pre', aux, q, qs)."""
    T = x.size(0)
    H = hl.specialize(x.size(1))
    hc = hl.specialize(res.size(1))
    G = H // 32
    bt = hl.register_block_size(1, 16)
    bh = hl.register_block_size(32, 256)
    bg = hl.register_block_size(1, 8)
    rc = torch.empty_like(res)
    post_o = torch.empty([T, hc], dtype=torch.float32, device=x.device)
    comb_o = torch.empty([T, hc, hc], dtype=torch.float32, device=x.device)
    pre_o = torch.empty([T, hc], dtype=torch.float32, device=x.device)
    layer_in = torch.empty([T, H], dtype=torch.bfloat16, device=x.device)
    if write_aux:
        aux = torch.empty([T, H], dtype=torch.bfloat16, device=x.device)
    else:
        aux = torch.empty([0, H], dtype=torch.bfloat16, device=x.device)
    if write_q:
        q = torch.empty([T, H], dtype=torch.float8_e4m3fn, device=x.device)
        qs = torch.empty([(T + 127) // 128 * 128 * ((G + 3) // 4) * 4], dtype=torch.uint8,
                         device=x.device)
    else:
        q = torch.empty([0, H], dtype=torch.float8_e4m3fn, device=x.device)
        qs = torch.empty([0], dtype=torch.uint8, device=x.device)
    rc4 = rc.view(T, hc, G, 32)
    li3 = layer_in.view(T, G, 32)
    ax3 = aux.view(aux.size(0), G, 32)
    q3 = q.view(q.size(0), G, 32)
    nw2 = norm_w.view(G, 32)
    for tile_t in hl.tile(T, block_size=bt):
        c00 = comb[tile_t, 0, 0]
        c01 = comb[tile_t, 0, 1]
        c02 = comb[tile_t, 0, 2]
        c03 = comb[tile_t, 0, 3]
        c10 = comb[tile_t, 1, 0]
        c11 = comb[tile_t, 1, 1]
        c12 = comb[tile_t, 1, 2]
        c13 = comb[tile_t, 1, 3]
        c20 = comb[tile_t, 2, 0]
        c21 = comb[tile_t, 2, 1]
        c22 = comb[tile_t, 2, 2]
        c23 = comb[tile_t, 2, 3]
        c30 = comb[tile_t, 3, 0]
        c31 = comb[tile_t, 3, 1]
        c32 = comb[tile_t, 3, 2]
        c33 = comb[tile_t, 3, 3]
        pp0 = post[tile_t, 0]
        pp1 = post[tile_t, 1]
        pp2 = post[tile_t, 2]
        pp3 = post[tile_t, 3]
        sq = hl.zeros([tile_t], dtype=torch.float32)
        a_pre = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_post = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c0 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c1 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c2 = hl.zeros([tile_t, hc], dtype=torch.float32)
        a_c3 = hl.zeros([tile_t, hc], dtype=torch.float32)
        for tile_h in hl.tile(H, block_size=bh):
            xh = x[tile_t, tile_h].to(torch.float32)
            r0 = res[tile_t, 0, tile_h].to(torch.float32)
            r1 = res[tile_t, 1, tile_h].to(torch.float32)
            r2 = res[tile_t, 2, tile_h].to(torch.float32)
            r3 = res[tile_t, 3, tile_h].to(torch.float32)
            q0 = _post_stream(c00, c10, c20, c30, pp0, r0, r1, r2, r3, xh)
            q1 = _post_stream(c01, c11, c21, c31, pp1, r0, r1, r2, r3, xh)
            q2 = _post_stream(c02, c12, c22, c32, pp2, r0, r1, r2, r3, xh)
            q3_ = _post_stream(c03, c13, c23, c33, pp3, r0, r1, r2, r3, xh)
            rc[tile_t, 0, tile_h] = q0
            rc[tile_t, 1, tile_h] = q1
            rc[tile_t, 2, tile_h] = q2
            rc[tile_t, 3, tile_h] = q3_
            f0 = q0.to(torch.float32)
            f1 = q1.to(torch.float32)
            f2 = q2.to(torch.float32)
            f3 = q3_.to(torch.float32)
            sq = sq + torch.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3, dim=1)
            a_pre = _mix_acc(a_pre, f0, f1, f2, f3, w_pre[:, 0, tile_h], w_pre[:, 1, tile_h],
                             w_pre[:, 2, tile_h], w_pre[:, 3, tile_h])
            a_post = _mix_acc(a_post, f0, f1, f2, f3, w_post[:, 0, tile_h], w_post[:, 1, tile_h],
                              w_post[:, 2, tile_h], w_post[:, 3, tile_h])
            a_c0 = _mix_acc(a_c0, f0, f1, f2, f3, w_comb[0, :, 0, tile_h], w_comb[0, :, 1, tile_h],
                            w_comb[0, :, 2, tile_h], w_comb[0, :, 3, tile_h])
            a_c1 = _mix_acc(a_c1, f0, f1, f2, f3, w_comb[1, :, 0, tile_h], w_comb[1, :, 1, tile_h],
                            w_comb[1, :, 2, tile_h], w_comb[1, :, 3, tile_h])
            a_c2 = _mix_acc(a_c2, f0, f1, f2, f3, w_comb[2, :, 0, tile_h], w_comb[2, :, 1, tile_h],
                            w_comb[2, :, 2, tile_h], w_comb[2, :, 3, tile_h])
            a_c3 = _mix_acc(a_c3, f0, f1, f2, f3, w_comb[3, :, 0, tile_h], w_comb[3, :, 1, tile_h],
                            w_comb[3, :, 2, tile_h], w_comb[3, :, 3, tile_h])
        rms = torch.rsqrt(sq / (hc * H) + rms_eps)
        s0 = scale[0]
        s1 = scale[1]
        s2 = scale[2]
        pre_o[tile_t, :] = torch.sigmoid(a_pre * rms[:, None] * s0 + b_pre[:]) + pre_eps
        post_o[tile_t, :] = torch.sigmoid(a_post * rms[:, None] * s1 + b_post[:]) * post_mult
        jp = hl.arange(0, hc)[None, :, None]            # comb row index of each accumulator
        m_comb = (a_c0[:, None, :] * (jp == 0) + a_c1[:, None, :] * (jp == 1)
                  + a_c2[:, None, :] * (jp == 2) + a_c3[:, None, :] * (jp == 3))   # [bt, 4, 4]
        comb_o[tile_t, :, :] = sinkhorn(m_comb * rms[:, None, None] * s2 + b_comb[:, :], sk_eps,
                                        repeat)
        if use_pre_in:
            p0 = pre_in[tile_t, 0]
            p1 = pre_in[tile_t, 1]
            p2 = pre_in[tile_t, 2]
            p3 = pre_in[tile_t, 3]
        else:
            p0 = hl.full([tile_t], 1.0, dtype=torch.float32)
            p1 = hl.zeros([tile_t], dtype=torch.float32)
            p2 = hl.zeros([tile_t], dtype=torch.float32)
            p3 = hl.zeros([tile_t], dtype=torch.float32)
        ssq = hl.zeros([tile_t, 1, 1], dtype=torch.float32)
        for tile_g in hl.tile(G, block_size=bg):
            v0 = rc4[tile_t, 0, tile_g, :].to(torch.float32)
            v1 = rc4[tile_t, 1, tile_g, :].to(torch.float32)
            v2 = rc4[tile_t, 2, tile_g, :].to(torch.float32)
            v3 = rc4[tile_t, 3, tile_g, :].to(torch.float32)
            pooled = _pool(p0, p1, p2, p3, v0, v1, v2, v3)
            ssq = ssq + torch.sum(torch.sum(pooled * pooled, dim=2, keepdim=True), dim=1, keepdim=True)
        rstd = torch.rsqrt(ssq / H + norm_eps)
        for tile_g in hl.tile(G, block_size=bg):
            v0 = rc4[tile_t, 0, tile_g, :].to(torch.float32)
            v1 = rc4[tile_t, 1, tile_g, :].to(torch.float32)
            v2 = rc4[tile_t, 2, tile_g, :].to(torch.float32)
            v3 = rc4[tile_t, 3, tile_g, :].to(torch.float32)
            pooled = _pool(p0, p1, p2, p3, v0, v1, v2, v3)
            ly = (pooled * rstd * nw2[tile_g, :].to(torch.float32)).to(torch.bfloat16)
            li3[tile_t, tile_g, :] = ly
            if write_aux:
                ax3[tile_t, tile_g, :] = ((((v0 + v1) + v2) + v3) / hc).to(torch.bfloat16)
            if write_q:
                qv, sf = mxfp8_group_quant(ly.to(torch.float32))
                q3[tile_t, tile_g, :] = qv
                qs[mxfp8_scale_offsets(tile_t.index, tile_g.index, G)] = sf.to(torch.uint8)
        if write_q:
            for tile_gp in hl.tile(G, (G + 3) // 4 * 4, block_size=4):
                qs[mxfp8_scale_offsets(tile_t.index, tile_gp.index, G)] = hl.zeros(
                    [tile_t, tile_gp], dtype=torch.uint8)
            if tile_t.begin + bt >= T:
                for tile_r in hl.tile(T, (T + 127) // 128 * 128, block_size=32):
                    for tile_gp in hl.tile((G + 3) // 4 * 4, block_size=32):
                        qs[mxfp8_scale_offsets(tile_r.index, tile_gp.index, G)] = hl.zeros(
                            [tile_r, tile_gp], dtype=torch.uint8)
    return rc, post_o, comb_o, layer_in, pre_o, aux, q, qs


# ---------------------------------------------------------------------------
# problem containers + argument builders (vLLM tensors <-> body args; views, no copies)
# ---------------------------------------------------------------------------
def _p2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def problem_from_stock(x, residual, post, comb, fn, scale, base, pre_mix, norm_weight,
                       rms_eps, pre_eps, sk_eps, post_mult, repeat, norm_eps):
    """vLLM's mhc_fused_post_pre_delayed_tilelang arguments -> the container the builders use."""
    t, h = x.shape
    if h % 32 or residual.shape != (t, HC, h) or fn.shape != (MIX, HC * h):
        raise ValueError(f"{MARKER} shapes res {tuple(residual.shape)} fn {tuple(fn.shape)} "
                         f"(need [T, {HC}, H], [{MIX}, {HC}*H], H % 32 == 0)")
    return SimpleNamespace(t=t, h=h, x=x, res=residual, post=post.view(t, HC), comb=comb, fn=fn,
                           scale=scale, base=base, pre_in=pre_mix, norm_w=norm_weight,
                           rms_eps=float(rms_eps), pre_eps=float(pre_eps), sk_eps=float(sk_eps),
                           post_mult=float(post_mult), repeat=int(repeat), norm_eps=float(norm_eps))


def k1_args(P):
    fn6 = P.fn.view(6, HC, HC, P.h)
    return (P.x, P.res, P.post, P.comb, fn6[0], fn6[1], fn6[2:6])


def _tail(P, write_aux: bool, write_q: bool):
    return (P.rms_eps, P.pre_eps, P.sk_eps, P.post_mult, P.norm_eps, P.repeat,
            P.pre_in is not None, bool(write_aux), bool(write_q))


def _coef(P):
    b = P.base
    return (P.scale, b[0:4], b[4:8], b[8:24].view(HC, HC))


def k2_args(P, k1_out, write_aux: bool = False, write_q: bool = False):
    rc, part_sq, part_pre, part_post, part_comb = k1_out
    pre_in = P.pre_in if P.pre_in is not None else P.post
    return (rc, part_sq, part_pre, part_post, part_comb, *_coef(P), pre_in, P.norm_w,
            *_tail(P, write_aux, write_q))


def k0_args(P, write_aux: bool = False, write_q: bool = False):
    fn6 = P.fn.view(6, HC, HC, P.h)
    pre_in = P.pre_in if P.pre_in is not None else P.post
    return (P.x, P.res, P.post, P.comb, fn6[0], fn6[1], fn6[2:6], *_coef(P), pre_in, P.norm_w,
            *_tail(P, write_aux, write_q))


# ---------------------------------------------------------------------------
# serving (the hook: kernels/dsv41_fuse_hook.py)
# ---------------------------------------------------------------------------
def case(tokens: int, hidden: int = 5120) -> dict:
    return {"numtokens": tokens, "hidden": hidden}


def serving_case(tokens: int, hidden: int) -> dict:
    return case(hk.bucket(tokens, BUCKETS), hidden)


def boundary_knob() -> str:
    v = os.environ.get(BOUNDARY_ENV, "split2").strip().lower() or "split2"
    if v not in ("split2", "resident"):
        raise RuntimeError(f"{BOUNDARY_ENV}={v!r}: expected split2 | resident")
    return v


def kernels_of(boundary: str) -> tuple:
    return (K_RES,) if boundary == "resident" else (K_POST, K_EPI)


def missing(boundary: str, hidden: int) -> list:
    """(kernel, bucket) pairs with no shipped config on this GPU."""
    return [(k, b) for k in kernels_of(boundary) for b in BUCKETS
            if hk.config(k, case(b, hidden)) is None]


_WARM: set = set()
_BODIES = {K_POST: dsv41_hc_post_partials, K_EPI: dsv41_hc_epilogue, K_RES: dsv41_hc_resident}


def _kernel(name: str, case_: dict):
    return hk.compiled(_BODIES[name], name, case_)


def _check_warm(key: tuple) -> None:
    """No first-use compile / state inside a CUDA graph capture: every (kernel, bucket,
    flags) variant must have run eagerly (vLLM's warmup pass does) before it is captured."""
    if key in _WARM:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{MARKER} {key} first used while capturing a CUDA graph: warm every "
                           "bucket / flag variant eagerly before capture (never compile in capture)")
    _WARM.add(key)


def hc_post_pre(x, residual, post, comb, fn, scale, base, rms_eps, pre_eps, sk_eps, post_mult,
                repeat, pre_mix=None, norm_weight=None, norm_eps=1e-6, capture_aux=False,
                want_q=False):
    """Drop-in for ``mhc_fused_post_pre_delayed_tilelang`` (same 6-tuple: residual_cur,
    post' [T,4,1], comb' [T,4,4], layer_input, next_pre_mix, aux). want_q appends
    (q, qs): the MXFP8 quantisation of layer_input (see the module docstring)."""
    if norm_weight is None:
        raise RuntimeError(f"{MARKER} norm_weight is None: the hook must route this call to stock")
    P = problem_from_stock(x, residual, post, comb, fn, scale, base, pre_mix, norm_weight,
                           rms_eps, pre_eps, sk_eps, post_mult, repeat, norm_eps)
    case_ = serving_case(P.t, P.h)
    boundary = boundary_knob()
    flags = (bool(capture_aux), bool(want_q), P.pre_in is not None)
    if boundary == "resident":
        _check_warm((K_RES, case_["numtokens"], P.h, flags))
        rc, post_o, comb_o, layer_in, pre_o, aux, q, qs = _kernel(K_RES, case_)(
            *k0_args(P, capture_aux, want_q))
    else:
        _check_warm((K_POST, case_["numtokens"], P.h))
        _check_warm((K_EPI, case_["numtokens"], P.h, flags))
        k1 = _kernel(K_POST, case_)(*k1_args(P))
        post_o, comb_o, layer_in, pre_o, aux, q, qs = _kernel(K_EPI, case_)(
            *k2_args(P, k1, capture_aux, want_q))
        rc = k1[0]
    out = (rc, post_o.unsqueeze(-1), comb_o, layer_in, pre_o, aux)
    return out + (q, qs) if want_q else out


# ---------------------------------------------------------------------------
# exact references (CPU or GPU, float64) - K3 oracle
# ---------------------------------------------------------------------------
def fn_of(w_pre, w_post, w_comb):
    return torch.cat([w_pre.reshape(HC, -1), w_post.reshape(HC, -1),
                      w_comb.reshape(HC * HC, -1)]).double()


def reference_residual(x, res, post, comb):
    """residual_cur = bf16(einsum(comb, res) + post * x) (mhc_post_torch, f64 sums)."""
    mixed = torch.einsum("tij,tih->tjh", comb.double(), res.double())
    return (mixed + post.double()[:, :, None] * x.double()[:, None, :]).to(torch.bfloat16)


def reference_stats(rc, fn):
    """-> (sq [T] f64, raw mixes [T, 24] f64) of a given residual_cur."""
    rf = rc.double()
    return (rf * rf).sum((1, 2)), rf.reshape(rf.shape[0], -1) @ fn.double().T


def reference_epilogue(rc, sq, mix, scale, base, pre_in, norm_w, rms_eps, pre_eps, sk_eps,
                       post_mult, norm_eps, repeat, write_aux=False, write_q=False):
    """Everything downstream of (rc, sum-of-squares, raw mixes): the K2 / epilogue spec.
    -> (post' [T,4], comb' [T,4,4], layer_in, pre' [T,4], aux, q, qs)."""
    t, hc, h = rc.shape
    rf = rc.double()
    mixes = mix * torch.rsqrt(sq / (hc * h) + rms_eps)[:, None]
    s, b = scale.double(), base.double()
    pre = torch.sigmoid(mixes[:, :4] * s[0] + b[:4]) + pre_eps
    post = torch.sigmoid(mixes[:, 4:8] * s[1] + b[4:8]) * post_mult
    c = (mixes[:, 8:] * s[2] + b[8:]).view(t, HC, HC)
    c = torch.softmax(c, dim=-1) + sk_eps
    c = c / (c.sum(1, keepdim=True) + sk_eps)
    for _ in range(repeat - 1):
        c = c / (c.sum(2, keepdim=True) + sk_eps)
        c = c / (c.sum(1, keepdim=True) + sk_eps)
    if pre_in is None:
        pin = torch.zeros(t, HC, dtype=torch.float64, device=rc.device)
        pin[:, 0] = 1.0
    else:
        pin = pre_in.double()
    pooled = (pin[:, :, None] * rf).sum(1).to(torch.bfloat16).double()
    out = (pooled * torch.rsqrt((pooled * pooled).mean(-1, keepdim=True) + norm_eps)
           * norm_w.double()).to(torch.bfloat16)
    aux = (rf.sum(1) / HC).to(torch.bfloat16) if write_aux else rc.new_empty(0, h)
    if write_q:
        q, qs = cm.mxfp8_quant_reference(out)
    else:
        q, qs = rc.new_empty(0, h, dtype=torch.float8_e4m3fn), rc.new_empty(0, dtype=torch.uint8)
    return post.float(), c.float(), out, pre.float(), aux, q, qs


def reference_args(args):
    """Full-boundary reference (rc + everything downstream) of a k0_args tuple."""
    (x, res, post, comb, w_pre, w_post, w_comb, scale, b_pre, b_post, b_comb, pre_in, norm_w,
     rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat, use_pre_in, write_aux, write_q) = args
    rc = reference_residual(x, res, post, comb)
    sq, mix = reference_stats(rc, fn_of(w_pre, w_post, w_comb))
    base = torch.cat([b_pre, b_post, b_comb.reshape(-1)])
    return (rc, *reference_epilogue(rc, sq, mix, scale, base, pre_in if use_pre_in else None,
                                    norm_w, rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat,
                                    write_aux, write_q))


def _ulp_close(a, b, ulps=1):
    a, b = a.float(), b.float()
    tol = ulps * (2.0 ** -7) * torch.maximum(a.abs(), b.abs()) + 1e-30
    return (a - b).abs() <= tol


def _same_bits(a, b) -> bool:
    return cm._bits_equal(a.detach().cpu(), b.detach().cpu())


def judge_rc(rc, ref_rc):
    """rc: <= 0.2% of elements may differ, by <= 1 bf16 ulp (fp32 vs f64 summation ties)."""
    bad = float((~_ulp_close(rc, ref_rc)).float().mean()) if rc.numel() else 0.0
    exact = float((rc != ref_rc).float().mean()) if rc.numel() else 0.0
    return bad == 0.0 and exact <= 2e-3, f"rc exact-mismatch {exact:.1e} (>1ulp {bad:.1e})"


def judge_epilogue(got, ref):
    """got / ref = (post', comb', layer_in, pre', aux, q, qs). post'/comb'/pre' rel 1e-4
    (fp32 split sums), layer_in / aux within 1 bf16 ulp except <= 0.2% of elements, and q / qs
    BIT-exact against the exact quantiser applied to the KERNEL'S OWN layer_in."""
    msgs, ok = [], True
    for i, name in ((0, "post'"), (1, "comb'"), (3, "pre'")):
        d = (got[i].float() - ref[i].float()).abs().max()
        rel = float(d / ref[i].float().abs().max().clamp_min(1e-30))
        msgs.append(f"{name} rel {rel:.1e}")
        ok &= rel < 1e-4
    for i, name in ((2, "layer_in"), (4, "aux")):
        if got[i].numel() == 0:
            continue
        bad = float((~_ulp_close(got[i], ref[i])).float().mean())
        msgs.append(f"{name} >1ulp {bad:.1e}")
        ok &= bad <= 2e-3
    if got[5].numel():
        q_ref, qs_ref = cm.mxfp8_quant_reference(got[2])
        same = _same_bits(got[5], q_ref) and _same_bits(got[6], qs_ref)
        msgs.append("q/qs " + ("bit-exact" if same else "MISMATCH"))
        ok &= same
    ok &= all(bool(t.float().isfinite().all()) for t in got[:5])
    return bool(ok), ", ".join(msgs)


def judge_boundary(got, args):
    """Full boundary (resident / pair): (rc, post', comb', layer_in, pre', aux, q, qs)."""
    ref = reference_args(args)
    ok_rc, m_rc = judge_rc(got[0], ref[0])
    # downstream judged on the kernel's own rc, so rc ties cannot mask or fake an epilogue error
    (x, res, post, comb, w_pre, w_post, w_comb, scale, b_pre, b_post, b_comb, pre_in, norm_w,
     rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat, use_pre_in, write_aux, write_q) = args
    sq, mix = reference_stats(got[0], fn_of(w_pre, w_post, w_comb))
    own = reference_epilogue(got[0], sq, mix, scale, torch.cat([b_pre, b_post, b_comb.reshape(-1)]),
                             pre_in if use_pre_in else None, norm_w, rms_eps, pre_eps, sk_eps,
                             post_mult, norm_eps, repeat, write_aux, write_q)
    ok, msg = judge_epilogue(got[1:], own)
    return ok_rc and ok, f"{m_rc}, {msg}"


def totals_of_partials(parts):
    """Sum the K1 split partials: -> (sq [T], pre [T,4], post [T,4], comb [T,4,4]) f64."""
    return tuple(p.double().sum(0) for p in parts)


def judge_post_partials(got, args):
    """K1: rc + the SUM of the split partials vs the exact statistics of the kernel's own rc."""
    x, res, post, comb, w_pre, w_post, w_comb = args[:7]
    ok_rc, m_rc = judge_rc(got[0], reference_residual(x, res, post, comb))
    sq, mix = reference_stats(got[0], fn_of(w_pre, w_post, w_comb))
    tot = totals_of_partials(got[1:])
    want = (sq, mix[:, :4], mix[:, 4:8], mix[:, 8:].reshape(-1, HC, HC))
    msgs, ok = [m_rc], ok_rc
    for name, g, w in zip(("sq", "pre", "post", "comb"), tot, want):
        rel = float((g - w).abs().max() / w.abs().max().clamp_min(1e-30))
        msgs.append(f"{name} rel {rel:.1e}")
        ok &= rel < 1e-4
    return bool(ok), ", ".join(msgs)


def split_partials(rc, fn, ns: int = 4):
    """Realistic K1 outputs for a given rc: the exact statistics cut into ns column splits."""
    t, hc, h = rc.shape
    rf = rc.double()
    per = -(-h // ns)
    sq = torch.zeros(ns, t, dtype=torch.float32, device=rc.device)
    pre = torch.zeros(ns, t, HC, dtype=torch.float32, device=rc.device)
    post = torch.zeros_like(pre)
    comb = torch.zeros(ns, t, HC, HC, dtype=torch.float32, device=rc.device)
    f4 = fn.double().view(MIX, HC, h)
    for s in range(ns):
        sl = slice(s * per, min((s + 1) * per, h))
        part = rf[:, :, sl]
        sq[s] = (part * part).sum((1, 2)).float()
        mix = torch.einsum("tjh,mjh->tm", part, f4[:, :, sl])
        pre[s], post[s], comb[s] = mix[:, :4].float(), mix[:, 4:8].float(), mix[:, 8:].reshape(t, HC, HC).float()
    return sq, pre, post, comb


# ---------------------------------------------------------------------------
# problems for tests / lab
# ---------------------------------------------------------------------------
def make_problem(tokens: int, hidden: int = 5120, seed: int = 0, device: str = "cpu",
                 use_pre_in: bool = True, consts: dict | None = None):
    """Decode-like activations, plausible mHC weights (fn ~ N(0, 0.02): the mixes stay O(1)
    after the rms scaling), non-degenerate gates."""
    g = torch.Generator().manual_seed(1234 + seed)

    def rnd(*shape, std=1.0, dtype=torch.float32):
        return (torch.randn(shape, generator=g) * std).to(dtype).to(device)

    t, h = tokens, hidden
    k = dict(DEFAULT_CONSTS, **(consts or {}))
    return SimpleNamespace(
        t=t, h=h, x=rnd(t, h, dtype=torch.bfloat16), res=rnd(t, HC, h, std=2.0, dtype=torch.bfloat16),
        post=torch.rand(t, HC, generator=g).mul(2.0).to(device),
        comb=torch.softmax(torch.randn(t, HC, HC, generator=g), -1).to(device),
        fn=rnd(MIX, HC * h, std=0.02), scale=(torch.rand(3, generator=g) * 0.5 + 0.1).to(device),
        base=rnd(MIX, std=0.5),
        pre_in=(torch.sigmoid(torch.randn(t, HC, generator=g)) + 1e-6).to(device) if use_pre_in else None,
        norm_w=(1.0 + 0.1 * torch.randn(h, generator=g)).to(torch.bfloat16).to(device), **k)


def stock_call_args(P, write_aux: bool = False):
    """The same problem in vLLM's mhc_fused_post_pre_delayed_tilelang calling convention."""
    return (P.x, P.res, P.post.view(P.t, HC, 1).contiguous(), P.comb, P.fn, P.scale, P.base,
            P.rms_eps, P.pre_eps, P.sk_eps, P.post_mult, P.repeat, P.pre_in, P.norm_w, P.norm_eps,
            bool(write_aux))


def baseline(x, residual, post, comb, fn, scale, base, rms_eps, pre_eps, sk_eps, post_mult,
             repeat, pre_mix, norm_weight, norm_eps, capture_aux):
    """The stock launch sequence on SM120: vLLM's own fused post+pre (TileLang)."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_delayed_tilelang
    return mhc_fused_post_pre_delayed_tilelang(
        x, residual, post, comb, fn, scale, base, rms_eps, pre_eps, sk_eps, post_mult, repeat,
        pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps, capture_aux=capture_aux)


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py): one entry per kernel
# ---------------------------------------------------------------------------
def _is_v41(model: dict) -> bool:
    return str(model.get("architecture", "")).startswith(("DeepseekV41", "DeepseekV4"))


def cases(model: dict) -> list:
    """The V4.1 config has hidden 5120 / hc_mult 4: mHC is replicated per rank."""
    if not _is_v41(model):
        return []
    hidden = int(model.get("hidden", 5120) or 5120)
    return [case(b, hidden) for b in BUCKETS]


def default_config(case_: dict, kernel: str = K_POST) -> dict:
    """A starting point (NOT tuned) inside the bounded search space; --autotune searches."""
    t = case_["numtokens"]
    cap = _p2(t)
    if kernel == K_POST:
        # fn (1.97 MB) is re-read once per token tile: bigger tiles at bigger T
        return {"block_sizes": [min(1 if t <= 2 else 4 if t <= 24 else 8, cap), 64 if t > 24 else 128,
                                8 if t <= 16 else 4], "num_warps": 4 if t <= 24 else 8}
    if kernel == K_EPI:
        return {"block_sizes": [min(2, cap), 4], "num_warps": 4}
    return {"block_sizes": [min(1, cap), 128, 2], "num_warps": 4}


def _make_k1(c, seed):
    return k1_args(make_problem(c["numtokens"], c["hidden"], seed, device=cm.DEVICE))


def _make_k2(c, seed, write_q=True):
    P = make_problem(c["numtokens"], c["hidden"], seed, device=cm.DEVICE)
    rc = reference_residual(P.x, P.res, P.post, P.comb)
    return k2_args(P, (rc, *split_partials(rc, P.fn)), write_aux=True, write_q=write_q)


def _make_k0(c, seed):
    # the heaviest variant (aux + MXFP8 quant): the constexpr flags only prune work, so a config tuned on it serves the rest
    return k0_args(make_problem(c["numtokens"], c["hidden"], seed, device=cm.DEVICE), True, True)


def _ref_k1(*args):
    x, res, post, comb, w_pre, w_post, w_comb = args[:7]
    return reference_residual(x, res, post, comb)


def _ref_k2(*args):
    (rc, p_sq, p_pre, p_post, p_comb, scale, b_pre, b_post, b_comb, pre_in, norm_w,
     rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat, use_pre_in, write_aux, write_q) = args
    t = rc.shape[0]
    mix = torch.cat([p_pre.double().sum(0), p_post.double().sum(0),
                     p_comb.double().sum(0).reshape(t, HC * HC)], 1)
    return reference_epilogue(rc, p_sq.double().sum(0), mix, scale,
                              torch.cat([b_pre, b_post, b_comb.reshape(-1)]),
                              pre_in if use_pre_in else None, norm_w, rms_eps, pre_eps, sk_eps,
                              post_mult, norm_eps, repeat, write_aux, write_q)


def _cpu(t):
    return tuple(a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in t)


_LAST: dict = {}


def _lab_make(kind):
    make = {"k1": _make_k1, "k2": _make_k2, "k0": _make_k0}[kind]

    def wrapped(c, seed):
        args = make(c, seed)
        _LAST[kind] = args
        return args
    return wrapped


def _check_k1(got, ref):
    return judge_post_partials(_cpu(got), _cpu(_LAST["k1"]))


def _check_k2(got, ref):
    return judge_epilogue(_cpu(got), _cpu(ref))


def _check_k0(got, ref):
    return judge_boundary(_cpu(got), _cpu(_LAST["k0"]))


def _stock_k0_baseline(*args):
    """Whole stock boundary on the same data (the resident kernel replaces all of it)."""
    (x, res, post, comb, w_pre, w_post, w_comb, scale, b_pre, b_post, b_comb, pre_in, norm_w,
     rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat, use_pre_in, write_aux, write_q) = args
    t = x.shape[0]
    return baseline(x, res, post.view(t, HC, 1), comb, fn_of(w_pre, w_post, w_comb).float(), scale,
                    torch.cat([b_pre, b_post, b_comb.reshape(-1)]), rms_eps, pre_eps, sk_eps,
                    post_mult, repeat, pre_in if use_pre_in else None, norm_w, norm_eps, write_aux)


def _stock_k1_baseline(*args):
    """Stock stage 1: [post + prenorm GEMM] (T <= 32: the fused TileLang kernel; above: post +
    GEMM). vLLM internals @ 39983a5e7 (tilelang.py:373-419); lab image only."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL, _MHC_FUSED_TILELANG_KERNEL, _MHC_POST_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config)
    x, res, post, comb, w_pre, w_post, w_comb = args[:7]
    t, h = x.shape
    fn = fn_of(w_pre, w_post, w_comb).float().contiguous()
    if mhc_fused_post_pre_split_config(t, h, HC) is not None:
        return _MHC_FUSED_TILELANG_KERNEL(comb, res, post, x, fn.view(MIX, HC, h), HC, h, MIX)
    rc = torch.empty_like(res)
    _MHC_POST_TILELANG_KERNEL(comb, res, post, x, rc, HC, h)
    mixes = torch.empty(1, t, MIX, dtype=torch.float32, device=x.device)
    sqr = torch.empty(1, t, dtype=torch.float32, device=x.device)
    _HC_PRENORM_GEMM_TILELANG_KERNEL(rc.view(t, HC * h), fn, mixes, sqr, HC * h, 1)
    return rc, mixes, sqr


def _stock_k2_baseline(*args):
    """Stock stage 2: the MHC_PRE_NORM epilogue on the stage-1 statistics (lab image only)."""
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL
    (rc, p_sq, p_pre, p_post, p_comb, scale, b_pre, b_post, b_comb, pre_in, norm_w,
     rms_eps, pre_eps, sk_eps, post_mult, norm_eps, repeat, use_pre_in, write_aux, write_q) = args
    t, _, h = rc.shape
    mixes = torch.cat([p_pre.sum(0), p_post.sum(0), p_comb.sum(0).reshape(t, HC * HC)], 1)[None]
    sqr = p_sq.sum(0)[None]
    post_o = torch.empty(t, HC, dtype=torch.float32, device=rc.device)
    comb_o = torch.empty(t, HC * HC, dtype=torch.float32, device=rc.device)
    nxt = torch.empty_like(post_o)
    li = torch.empty(t, h, dtype=torch.bfloat16, device=rc.device)
    MHC_PRE_NORM_KERNEL(mixes.contiguous(), sqr.contiguous(), scale, torch.cat([b_pre, b_post, b_comb.reshape(-1)]),
                        rc, post_o, comb_o, li, norm_w, pre_in if use_pre_in else post_o, nxt, li,
                        hidden_size=h, rms_eps=rms_eps, hc_pre_eps=pre_eps, hc_sinkhorn_eps=sk_eps,
                        hc_post_mult_value=post_mult, sinkhorn_repeat=repeat, norm_eps=norm_eps,
                        hc_mult=HC, use_pre_mix_in=use_pre_in, save_pre_mix=True, rms_numel=HC * h,
                        write_aux=False)
    return post_o, comb_o, li, nxt


def _fi_k2(args, got):
    return cm.flashinfer_quant_matches(got[2], got[5], got[6]) if args[-1] else (True, "")


def _fi_k0(args, got):
    return cm.flashinfer_quant_matches(got[3], got[6], got[7]) if args[-1] else (True, "")


SILICON_CHECKS = {K_EPI: _fi_k2, K_RES: _fi_k0}   # dsv41_bench.main: parity with flashinfer.mxfp8_quantize


KERNELS = [
    {"name": K_POST, "body": dsv41_hc_post_partials, "cases": cases, "make": _lab_make("k1"),
     "reference": _ref_k1, "baseline": _stock_k1_baseline,
     "baseline_name": "vLLM TileLang stage 1: fused post + split-K prenorm GEMM (<= 32 tokens) / "
                      "post + prenorm GEMM (above)",
     "check": _check_k1,
     "replaces": [r"mhc_post_tilelang", r"mhc_fused_post_pre(?!.*norm)", r"hc_prenorm_gemm",
                  r"tf32_hc_prenorm"]},
    {"name": K_EPI, "body": dsv41_hc_epilogue, "cases": cases, "make": _lab_make("k2"),
     "reference": _ref_k2, "baseline": _stock_k2_baseline,
     "baseline_name": "vLLM TileLang stage 2: MHC_PRE_NORM epilogue (reduce, sinkhorn, collapse, RMSNorm)",
     "check": _check_k2, "replaces": [r"mhc_pre_big_fuse"]},
    {"name": K_RES, "body": dsv41_hc_resident, "cases": cases, "make": _lab_make("k0"),
     "reference": lambda *a: reference_args(a), "baseline": _stock_k0_baseline,
     "baseline_name": "the whole stock boundary: vLLM mhc_fused_post_pre_delayed_tilelang "
                      "(2 launches <= 32 tokens, 3 above)",
     "check": _check_k0,
     "replaces": [r"mhc_pre_big_fuse", r"mhc_post_tilelang", r"mhc_fused_post_pre",
                  r"hc_prenorm_gemm", r"tf32_hc_prenorm"]},
]


# ---------------------------------------------------------------------------
# boot gates <kernel>_capture_replay (CLI: python -m ... capture_replay [--tokens N])
# ---------------------------------------------------------------------------
def _built(name, case_, cfg=None):
    import helion as _h
    cfg = cfg or hk.config(name, case_) or default_config(case_, name)
    return _h.kernel(_BODIES[name], config=_h.Config(**cfg), static_shapes=False)


def _gate(name, kern, make_args, runner_factory=None):
    """Boot gate over a body whose args are (tensors..., python scalars/flags...): the leading tensors
    are the captured inputs, the scalar tail is bound once."""
    sample = make_args(0)
    split = next((i for i, a in enumerate(sample) if not isinstance(a, torch.Tensor)), len(sample))
    return cm.capture_replay_gate(name, lambda *t: kern(*t, *sample[split:]),
                                  lambda i: list(make_args(i)[:split]), runner_factory=runner_factory)


def dsv41_hc_post_partials_capture_replay(tokens: int = 6, hidden: int = 5120, cfg=None,
                                          runner_factory=None) -> bool:
    kern = _built(K_POST, case(hk.bucket(tokens, BUCKETS), hidden), cfg)
    return _gate(K_POST, kern, lambda i: k1_args(make_problem(tokens, hidden, i, cm.DEVICE)),
                 runner_factory)


def dsv41_hc_epilogue_capture_replay(tokens: int = 6, hidden: int = 5120, cfg=None,
                                     write_q: bool = True, write_aux: bool = True,
                                     runner_factory=None) -> bool:
    kern = _built(K_EPI, case(hk.bucket(tokens, BUCKETS), hidden), cfg)

    def make(i):
        P = make_problem(tokens, hidden, i, cm.DEVICE)
        rc = reference_residual(P.x, P.res, P.post, P.comb)
        return k2_args(P, (rc, *split_partials(rc, P.fn)), write_aux, write_q)
    return _gate(K_EPI, kern, make, runner_factory)


def dsv41_hc_resident_capture_replay(tokens: int = 6, hidden: int = 5120, cfg=None,
                                     write_q: bool = True, write_aux: bool = True,
                                     runner_factory=None) -> bool:
    kern = _built(K_RES, case(hk.bucket(tokens, BUCKETS), hidden), cfg)
    return _gate(K_RES, kern, lambda i: k0_args(make_problem(tokens, hidden, i, cm.DEVICE), write_aux,
                                                write_q), runner_factory)


BOOT_GATES = {f"{K_POST}_capture_replay": dsv41_hc_post_partials_capture_replay,
              f"{K_EPI}_capture_replay": dsv41_hc_epilogue_capture_replay,
              f"{K_RES}_capture_replay": dsv41_hc_resident_capture_replay}


def bench_boundary(tokens=BUCKETS, hidden: int = 5120, iters: int = 100, cfg_file=None) -> int:
    """K4: the whole boundary, ours (split2 and resident, default or --config configs) vs the
    stock public function, CUDA graph, cold L2, per M bucket. Needs SM120 + the vLLM image."""
    import json

    import helion as _h

    from suffix_hybrid.kernels import dsv41_bench as bench
    table = json.load(open(cfg_file)) if cfg_file else {}
    status = 0
    print(f"== dsv41 hc boundary: stock mhc_fused_post_pre_delayed_tilelang vs split2 vs resident "
          f"(H={hidden}, us per boundary, cold L2)")
    for t in tokens:
        c = case(t, hidden)
        P = make_problem(t, hidden, 0, cm.DEVICE)
        stock_args = stock_call_args(P)
        t_stock = bench.graph_us(lambda: baseline(*stock_args), iters)
        row = f"  T={t:<4d} stock {t_stock:8.1f} us"
        for name, kerns in (("split2", (K_POST, K_EPI)), ("resident", (K_RES,))):
            cfgs = [table.get(f"{k}:{t}") or hk.config(k, c) or default_config(c, k) for k in kerns]
            built = [_h.kernel(_BODIES[k], config=_h.Config(**cf), static_shapes=False)
                     for k, cf in zip(kerns, cfgs)]
            if name == "split2":
                def run():
                    k1 = built[0](*k1_args(P))
                    return (k1[0], *built[1](*k2_args(P, k1)))
            else:
                def run():
                    return built[0](*k0_args(P))
            ok, msg = judge_boundary(run(), k0_args(P))
            us = bench.graph_us(run, iters)
            row += f" | {name} {us:8.1f} us x{t_stock / us:4.2f} {'ok' if ok else 'FAIL ' + msg}"
            status |= 0 if ok else 1
        print(row, flush=True)
    return status


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog=__name__)
    ap.add_argument("cmd", choices=["oracle", "bench", "capture_replay"])
    ap.add_argument("--config")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--buckets", type=int, nargs="*")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=6)
    ns, rest = ap.parse_known_args(argv)
    if ns.cmd == "capture_replay":
        ok = all(gate(tokens=ns.tokens) for gate in BOOT_GATES.values())
        return 0 if ok else 1
    if ns.cmd == "bench" and not ns.autotune:
        return bench_boundary(tuple(ns.buckets or BUCKETS), iters=ns.iters, cfg_file=ns.config)
    from suffix_hybrid.kernels import dsv41_bench
    return dsv41_bench.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
