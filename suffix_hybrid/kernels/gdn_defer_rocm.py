# SPDX-License-Identifier: Apache-2.0
"""GDN MTP-verify with deferred state commit (SUFFIX_ROCM_GDN_DEFER=1, ROCm).

vLLM's spec-decode contract for a GDN layer: the verify kernel reads the state from
slot num_accepted - 1 of the request's 1 + num_spec state slots and writes the state
after every verified token t to slot t, so whatever gets accepted can be read next
step. At MTP-4 that is 5 full fp32 state writes (48 heads x 128 x 128 x 4 B = 3.1 MB
each) per request and layer; only one of them is ever read.

Here a step writes the state after token 0 (always accepted) to slot 0, and for tokens
t >= 1 only their inputs (raw k, the program's v tile, a, b: 162 floats) to a record
inside slot 1's state tile. The next step rebuilds the accepted state from slot 0 by
replaying the first num_accepted - 1 recorded tokens with the same per-token code,
so states and outputs are bit-identical to the stock kernel's: one state read and one
write instead of one read and five writes.

Who else reads the slots: only vLLM's mamba "align" prefix-caching copies, and only
around block boundaries (post-step: the boundary state when the accepted tokens cross
one; pre-step: the committed state when the running block moves). A step whose
positions c + 1 .. c + 2 (num_spec + 1) contain a boundary (ZONE = the mamba block
size) writes every slot the stock way, and the step after one reads the stock way, so
those copies see what they always saw. num_accepted == 1 (first step after a prefill,
after vLLM's align reset) reads slot 0 under both contracts.

Tried and measured slower on the MI350P (Triton 3.8, 2026-10-10, oracle in the history
of 4f10e0f6): a rank-update replay from stored u / normalized k plus a chunk-form verify
(10 independent row reductions, k / q dot products from a per-head pre-pass). Fewer
instructions (4.1k vs 4.8k), but 180 VGPRs -> occupancy 2 instead of 3: c1/c8/c32
19.2/57.8/204.5 vs 15.0/52.2/177.5 us; a [NC, BV, KC] tile costs Triton ~30% more again.

    python -m suffix_hybrid.kernels.gdn_defer_rocm   # multi-step GPU oracle vs AITER + us/call
    python -m suffix_hybrid.kernels.gdn_defer_rocm mfma   # SUFFIX_ROCM_GDN_DEFER_MFMA vs fp64,
                                                          # AITER, v1 + us/call sweep
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix gdn-defer]"


@triton.jit
def _gdn_token(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold,
               USE_QK_L2NORM: tl.constexpr):
    # One token of AITER's fused_rearrange_sigmoid_gated_delta_rule_update_kernel
    # (v0.1.24.post1, = FLA's fused_sigmoid_gating_delta_rule_update), same op order.
    x = a_raw + dt_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    b_g = -tl.exp(A_log_v) * softplus_x
    b_beta = tl.sigmoid(b_raw)
    if USE_QK_L2NORM:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    return b_h


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, o, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr, REC: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    # Fully unrolled over the WIN = 1 + num_spec token slots (masked past the request's
    # query length): no data-dependent loop, so every load (state, record, the tokens'
    # q/k/v/a/b) can issue before the recurrence starts.
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    two = tl.arange(0, 2)
    tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]  # this program's [BV, K] state tile
    tile_base = i_hv * V * K + i_v * BV * K  # ... contiguous from here (BV x K floats)
    rec_v = K + tl.arange(0, BV)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_v = tl.load(dt_bias + i_hv).to(tl.float32)
    acc = tl.load(num_accepted + i_n).to(tl.int64)
    c_now = tl.load(seq_lens + i_n).to(tl.int64) - n_tok  # tokens computed before this step
    idx_row = state_indices + i_n * stride_idx_seq
    slot0 = tl.load(idx_row).to(tl.int64)
    slot1 = tl.load(idx_row + 1).to(tl.int64)
    if ZONE > 0:
        cur_zone = (c_now + ZONE_REACH) // ZONE > c_now // ZONE
        c_prev = c_now - acc
        prev_zone = (c_prev + ZONE_REACH) // ZONE > c_prev // ZONE
    else:
        cur_zone = c_now < 0
        prev_zone = c_now < 0
    # Deferred: slot 0 holds the state after the previous step's token 0, slot 1's tile
    # the inputs of its tokens 1.. -> replay the accepted ones. Else vLLM's contract.
    deferred = (acc >= 2) & (prev_zone == 0)
    read_slot = tl.where(deferred, slot0, tl.load(idx_row + acc - 1).to(tl.int64))
    if read_slot <= 0:
        return
    if deferred & (slot1 <= 0):
        return
    b_h = tl.zeros([BV, K], dtype=tl.float32)  # AITER's 0 + load (signed zeros)
    b_h += tl.load(state + read_slot * stride_state_block + tile).to(tl.float32)
    rec = state + slot1 * stride_state_block + tile_base
    for t in tl.static_range(1, WIN):
        use = deferred & (t < acc)
        r = rec + (t - 1) * REC
        r_k = tl.load(r + o_k, mask=use, other=0.0)  # issued up front, masked
        r_v = tl.load(r + rec_v, mask=use, other=0.0)
        r_ab = tl.load(r + K + BV + two, mask=use, other=0.0)
        if use:  # only the accepted tokens pay for the recurrence
            b_h = _gdn_token(b_h, r_k, r_v, tl.sum(tl.where(two == 0, r_ab, 0.0)),
                             tl.sum(tl.where(two == 1, r_ab, 0.0)), A_log_v, dt_v, beta,
                             threshold, USE_QK_L2NORM)

    p_q = qkv + bos * stride_qkv_l + i_h * K + o_k
    p_k = qkv + bos * stride_qkv_l + H * K + i_h * K + o_k
    p_v = qkv + bos * stride_qkv_l + 2 * H * K + i_hv * V + o_v
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + o_v
    for i_t in tl.static_range(WIN):
        valid = i_t < n_tok
        b_q = tl.load(p_q + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_k = tl.load(p_k + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_v = tl.load(p_v + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        a_raw = tl.load(p_a + i_t * stride_a_l, mask=valid, other=0.0).to(tl.float32)
        b_raw = tl.load(p_b + i_t * stride_b_l, mask=valid, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        h_new = _gdn_token(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold,
                           USE_QK_L2NORM)
        b_h = tl.where(valid, h_new, b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o + i_t * stride_o_l, b_o.to(p_o.dtype.element_ty), mask=valid)
        if i_t == 0:
            tl.store(state + slot0 * stride_state_block + tile, b_h,
                     mask=valid & (slot0 > 0))
        else:
            slot_t = tl.load(idx_row + i_t).to(tl.int64)
            tl.store(state + slot_t * stride_state_block + tile, b_h,
                     mask=valid & cur_zone & (slot_t > 0))
            r = rec + (i_t - 1) * REC
            keep = valid & (cur_zone == 0) & (slot1 > 0)
            tl.store(r + o_k, b_k, mask=keep)
            tl.store(r + rec_v, b_v, mask=keep)
            tl.store(r + K + BV + two, tl.where(two == 0, a_raw, b_raw), mask=keep)


# _gdn_defer_kernel's contract (slots, records, zones) in chunk form on the fp32 matrix cores.
# 16 token columns per (request, value head): WIN - 1 replayed records, the WIN new tokens' k,
# their q; an unused column is an identity token (k = 0, g = 0, beta = 0). With G the
# cumulative log decay and A[t, s] = beta_t e^(G_t - G_s) k_s.k_t (s < t), the delta rule is
# U (I + A)^T = beta (V - e^G S0 K), S_c = e^(G_c) S0 + sum_(s<=c) e^(G_c - G_s) u_s k_s^T,
# o_t = S_t q_t: [BV, K] x [K, 16] and [BV, 16] x [16, K] dots plus a few 16 x 16 ones instead
# of per-token row reductions. Records keep v1's 16-row-tile layout, so v1 and these kernels
# replay each other's. One kernel (_gdn_defer_mfma_kernel) or, NB > 0, a per-head prep kernel
# (k / q, G, P, bm to a workspace) + a main kernel over NB row blocks with S0 prefetch.


@triton.jit
def _mfma_slots(state_indices, num_accepted, seq_lens, i_n, n_tok, stride_idx_seq,
                ZONE: tl.constexpr, ZONE_REACH: tl.constexpr):
    # _gdn_defer_kernel's slot and zone decisions for request i_n (n_tok > 0)
    acc = tl.load(num_accepted + i_n).to(tl.int64)
    c_now = tl.load(seq_lens + i_n).to(tl.int64) - n_tok
    idx_row = state_indices + i_n * stride_idx_seq
    slot0 = tl.load(idx_row).to(tl.int64)
    slot1 = tl.load(idx_row + 1).to(tl.int64)
    if ZONE > 0:
        cur_zone = (c_now + ZONE_REACH) // ZONE > c_now // ZONE
        c_prev = c_now - acc
        prev_zone = (c_prev + ZONE_REACH) // ZONE > c_prev // ZONE
    else:
        cur_zone = c_now < 0
        prev_zone = c_now < 0
    deferred = (acc >= 2) & (prev_zone == 0)
    read_slot = tl.where(deferred, slot0, tl.load(idx_row + acc - 1).to(tl.int64))
    return acc, idx_row, slot0, slot1, cur_zone, deferred, read_slot


@triton.jit
def _mfma_cols(deferred, acc, n_tok, WIN: tl.constexpr):
    NREP: tl.constexpr = WIN - 1
    cols = tl.arange(0, 16)
    is_new = (cols >= NREP) & (cols < NREP + WIN)
    is_q = (cols >= NREP + WIN) & (cols < NREP + 2 * WIN)
    rep_use = (cols < NREP) & deferred & (cols + 1 < acc)
    tok = tl.maximum(tl.where(is_q, cols - NREP - WIN, cols - NREP), 0)
    return cols, is_new, is_q, rep_use, tok, is_new & (tok < n_tok), is_q & (tok < n_tok)


@triton.jit
def _mfma_kq(rec0, qkv, bos, tok, cols, rep_use, row_ok, is_q, stride_qkv_l, i_h, scale,
             H: tl.constexpr, K: tl.constexpr, REC: tl.constexpr):
    # k / q of the 16 columns [16, K] fp32 (replayed k from the record tile at rec0, the new
    # tokens' k / q where row_ok from qkv): k l2-normalized, q normalized and scaled.
    o_k = tl.arange(0, K)
    x = tl.load(rec0 + cols[:, None] * REC + o_k[None, :], mask=rep_use[:, None], other=0.0)
    x += tl.load(qkv + (bos + tok)[:, None] * stride_qkv_l
                 + (tl.where(is_q, 0, H * K) + i_h * K)[:, None] + o_k[None, :],
                 mask=row_ok[:, None], other=0.0).to(tl.float32)
    nrm = tl.rsqrt(tl.sum(x * x, 1) + 1e-6)
    return x * tl.where(is_q, nrm * scale, nrm)[:, None]


@triton.jit
def _mfma_egq(G, cols, is_q, WIN: tl.constexpr):
    # G of q column j's chunk column (j - WIN), and its exp on the q columns
    gq = tl.sum(tl.where((cols[None, :] == cols[:, None] + WIN) & is_q[None, :], G[:, None],
                         0.0), 0)
    return gq, tl.where(is_q, tl.exp(gq), 0.0)


@triton.jit
def _mfma_solve(kq, a_c, b_c, use, cols, is_q, A_log_v, dt_v, beta, threshold,
                WIN: tl.constexpr):
    # Per request and value head: _gdn_token's gating per column, G, P = (I + A)^-1 and bm,
    # the outputs' coefficients of u.
    NREP: tl.constexpr = WIN - 1
    x = a_c + dt_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    g = tl.where(use, -tl.exp(A_log_v) * softplus_x, 0.0)
    bet = tl.where(use, tl.sigmoid(b_c), 0.0)
    G = tl.cumsum(g, 0)
    t2 = cols[:, None]
    s2 = cols[None, :]
    gr = tl.dot(kq, tl.trans(kq), input_precision="ieee")  # k_s . k_t / k_s . q_j
    am = tl.where((s2 < t2) & (t2 < NREP + WIN),
                  bet[:, None] * tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0)) * gr, 0.0)
    eye = tl.where(t2 == s2, 1.0, 0.0)
    x1 = -am
    x2 = tl.dot(x1, x1, input_precision="ieee")
    x4 = tl.dot(x2, x2, input_precision="ieee")
    x8 = tl.dot(x4, x4, input_precision="ieee")
    p = tl.dot(eye + x1, eye + x2, input_precision="ieee")
    p = tl.dot(p, eye + x4, input_precision="ieee")
    p = tl.dot(p, eye + x8, input_precision="ieee")  # A^16 = 0
    gq, _ = _mfma_egq(G, cols, is_q, WIN)
    bm = tl.where(is_q[None, :] & (t2 <= s2 - WIN),
                  tl.exp(tl.minimum(gq[None, :] - G[:, None], 0.0)) * gr, 0.0)
    return bet, G, p, bm


@triton.jit
def _mfma_out(s0, vt, kq, bet, G, p, bm, cols, is_q, q_ok, p_o, WIN: tl.constexpr):
    # One block of state rows: w = S0 k / S0 q, U = beta (V - e^G w) P^T, outputs
    # e^(G_c) S0 q + U bm to p_o [BV, 16] (q columns); returns U.
    wz = tl.dot(s0, tl.trans(kq), input_precision="ieee")
    u = tl.dot(bet[None, :] * (vt - tl.exp(G)[None, :] * wz), tl.trans(p),
               input_precision="ieee")
    _, egq = _mfma_egq(G, cols, is_q, WIN)
    out = wz * egq[None, :] + tl.dot(u, bm, input_precision="ieee")
    tl.store(p_o, out.to(p_o.dtype.element_ty), mask=q_ok[None, :])
    return u


@triton.jit
def _mfma_commit(u, s_acc, kq, G, vt, cols, is_new, tok, n_tok, cur_zone, state, tile, slot0,
                 slot1, idx_row, rec_rows, stride_state_block, K: tl.constexpr,
                 WIN: tl.constexpr, REC: tl.constexpr):
    # The state after new token 0 to slot 0; near an align boundary the state after every
    # token to its slot (vLLM's contract), else the v slices of the new tokens' records.
    NREP: tl.constexpr = WIN - 1
    g0 = tl.sum(tl.where(cols == NREP, G, 0.0), 0)
    d0 = tl.where(cols <= NREP, tl.exp(tl.minimum(g0 - G, 0.0)), 0.0)
    s_c = tl.dot(u * d0[None, :], kq, acc=s_acc * tl.exp(g0), input_precision="ieee")
    tl.store(state + slot0 * stride_state_block + tile, s_c, mask=slot0 > 0)
    tl.debug_barrier()  # every warp's record loads are done before slot 1 is written
    if cur_zone:
        for t in tl.static_range(1, WIN):
            slot_t = tl.load(idx_row + t).to(tl.int64)
            gc = tl.sum(tl.where(cols == NREP + t, G, 0.0), 0)
            dc = tl.where(cols <= NREP + t, tl.exp(tl.minimum(gc - G, 0.0)), 0.0)
            s_t = tl.dot(u * dc[None, :], kq, acc=s_acc * tl.exp(gc), input_precision="ieee")
            tl.store(state + slot_t * stride_state_block + tile, s_t,
                     mask=(t < n_tok) & (slot_t > 0))
    else:
        keep = is_new & (cols > NREP) & (tok < n_tok) & (slot1 > 0)
        tl.store(rec_rows + (tl.maximum(cols - NREP - 1, 0) * REC)[None, :] + K, vt,
                 mask=keep[None, :])


@triton.jit
def _mfma_rec_kab(rec0, qkv, a, b, bos, tok, keep, i_h, i_hv, stride_qkv_l, stride_a_l,
                  stride_b_l, NSUB: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                  WIN: tl.constexpr, REC: tl.constexpr):
    # raw k, a, b of the new tokens 1.. (keep) into the NSUB 16-row record tiles from rec0,
    # the rows this program owns (slot 1 can be the slot it read S0 from)
    o_k = tl.arange(0, K)
    r_off = tl.maximum(tl.arange(0, 16) - WIN, 0) * REC
    k_raw = tl.load(qkv + (bos + tok)[:, None] * stride_qkv_l + H * K + i_h * K + o_k[None, :],
                    mask=keep[:, None], other=0.0).to(tl.float32)
    a_raw = tl.load(a + (bos + tok) * stride_a_l + i_hv, mask=keep, other=0.0).to(tl.float32)
    b_raw = tl.load(b + (bos + tok) * stride_b_l + i_hv, mask=keep, other=0.0).to(tl.float32)
    for j in tl.static_range(NSUB):
        rec_j = rec0 + j * 16 * K
        tl.store(rec_j + r_off[:, None] + o_k[None, :], k_raw, mask=keep[:, None])
        tl.store(rec_j + r_off + K + 16, a_raw, mask=keep)
        tl.store(rec_j + r_off + K + 17, b_raw, mask=keep)


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer_mfma_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, o, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr, REC: tl.constexpr,
    RELOAD: tl.constexpr, LEAN: tl.constexpr,
):
    # One program per (request, value head, BV rows), everything in one launch. RELOAD: S0
    # for the commit's accumulator from L2 again; LEAN: k reloaded for the commit instead of
    # staying live through the solve (fewer VGPRs).
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    acc, idx_row, slot0, slot1, cur_zone, deferred, read_slot = _mfma_slots(
        state_indices, num_accepted, seq_lens, i_n, n_tok, stride_idx_seq, ZONE, ZONE_REACH)
    if (read_slot <= 0) | (deferred & (slot1 <= 0)):
        return
    o_k = tl.arange(0, K)
    rows = tl.arange(0, BV)
    o_v = i_v * BV + rows
    tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    s_base = state + read_slot * stride_state_block
    s0 = tl.load(s_base + tile)  # the bulk of the traffic: first, in flight during the solve
    cols, is_new, is_q, rep_use, tok, new_ok, q_ok = _mfma_cols(deferred, acc, n_tok, WIN)
    slot1_base = state + slot1 * stride_state_block + i_hv * V * K
    rec0 = slot1_base + i_v * BV * K
    rec_rows = slot1_base + (o_v // 16 * 16)[:, None] * K + (o_v % 16)[:, None]
    kq = _mfma_kq(rec0, qkv, bos, tok, cols, rep_use, new_ok | q_ok, is_q, stride_qkv_l, i_h,
                  scale, H, K, REC)
    vt = tl.load(rec_rows + cols[None, :] * REC + K, mask=rep_use[None, :], other=0.0)
    vt += tl.load(qkv + (bos + tok)[None, :] * stride_qkv_l + 2 * H * K + i_hv * V
                  + o_v[:, None], mask=new_ok[None, :], other=0.0).to(tl.float32)
    a_c = tl.load(rec0 + cols * REC + K + 16, mask=rep_use, other=0.0)
    a_c += tl.load(a + (bos + tok) * stride_a_l + i_hv, mask=new_ok, other=0.0).to(tl.float32)
    b_c = tl.load(rec0 + cols * REC + K + 17, mask=rep_use, other=0.0)
    b_c += tl.load(b + (bos + tok) * stride_b_l + i_hv, mask=new_ok, other=0.0).to(tl.float32)
    bet, G, p, bm = _mfma_solve(kq, a_c, b_c, rep_use | new_ok, cols, is_q,
                                tl.load(A_log + i_hv).to(tl.float32),
                                tl.load(dt_bias + i_hv).to(tl.float32), beta, threshold, WIN)
    u = _mfma_out(s0, vt, kq, bet, G, p, bm, cols, is_q, q_ok,
                  o + (bos + tok)[None, :] * stride_o_l + i_hv * V + o_v[:, None], WIN)
    if RELOAD:
        s_acc = tl.load(s_base + tile)
    else:
        s_acc = s0
    if LEAN:  # the commit needs k only: replayed and new rows again
        kq = _mfma_kq(rec0, qkv, bos, tok, cols, rep_use, new_ok, is_q, stride_qkv_l, i_h,
                      scale, H, K, REC)
    _mfma_commit(u, s_acc, kq, G, vt, cols, is_new, tok, n_tok, cur_zone, state, tile, slot0,
                 slot1, idx_row, rec_rows, stride_state_block, K, WIN, REC)
    keep = is_new & (cols >= WIN) & (tok < n_tok) & (cur_zone == 0) & (slot1 > 0)
    _mfma_rec_kab(rec0, qkv, a, b, bos, tok, keep, i_h, i_hv, stride_qkv_l, stride_a_l,
                  stride_b_l, BV // 16, H, K, WIN, REC)


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_idx_seq"])
def _gdn_defer_mfma_prep_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale, ws,
    stride_qkv_l, stride_a_l, stride_b_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, WIN: tl.constexpr,
    ZONE: tl.constexpr, ZONE_REACH: tl.constexpr, REC: tl.constexpr, WS: tl.constexpr,
):
    # One program per (request, value head): k / q, bet, G, P, bm to ws[i_nh] for the main
    # kernel. No records here: slot 1 may be the slot the main kernel still reads S0 from.
    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    acc, idx_row, slot0, slot1, cur_zone, deferred, read_slot = _mfma_slots(
        state_indices, num_accepted, seq_lens, i_n, n_tok, stride_idx_seq, ZONE, ZONE_REACH)
    if (read_slot <= 0) | (deferred & (slot1 <= 0)):
        return
    cols, is_new, is_q, rep_use, tok, new_ok, q_ok = _mfma_cols(deferred, acc, n_tok, WIN)
    rec0 = state + slot1 * stride_state_block + i_hv * V * K
    kq = _mfma_kq(rec0, qkv, bos, tok, cols, rep_use, new_ok | q_ok, is_q, stride_qkv_l,
                  i_hv // (HV // H), scale, H, K, REC)
    a_c = tl.load(rec0 + cols * REC + K + 16, mask=rep_use, other=0.0)
    a_c += tl.load(a + (bos + tok) * stride_a_l + i_hv, mask=new_ok, other=0.0).to(tl.float32)
    b_c = tl.load(rec0 + cols * REC + K + 17, mask=rep_use, other=0.0)
    b_c += tl.load(b + (bos + tok) * stride_b_l + i_hv, mask=new_ok, other=0.0).to(tl.float32)
    bet, G, p, bm = _mfma_solve(kq, a_c, b_c, rep_use | new_ok, cols, is_q,
                                tl.load(A_log + i_hv).to(tl.float32),
                                tl.load(dt_bias + i_hv).to(tl.float32), beta, threshold, WIN)
    w = ws + i_nh * WS
    sq = cols[:, None] * 16 + cols[None, :]
    tl.store(w + cols[:, None] * K + tl.arange(0, K)[None, :], kq)
    tl.store(w + 16 * K + sq, p)
    tl.store(w + 16 * K + 256 + sq, bm)
    tl.store(w + 16 * K + 512 + cols, bet)
    tl.store(w + 16 * K + 528 + cols, G)


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer_mfma_main_kernel(
    a, b, qkv, o, state, cu_seqlens, state_indices, num_accepted, seq_lens, ws,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    NB: tl.constexpr, WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr,
    REC: tl.constexpr, WS: tl.constexpr, RELOAD: tl.constexpr, LEAN: tl.constexpr,
):
    # NB blocks of BV state rows per program from the prep kernel's per-head terms; the next
    # block's S0 load is in flight while this one computes. LEAN: k / q from ws (L2) for
    # each dot instead of staying live across the blocks.
    i_g, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    acc, idx_row, slot0, slot1, cur_zone, deferred, read_slot = _mfma_slots(
        state_indices, num_accepted, seq_lens, i_n, n_tok, stride_idx_seq, ZONE, ZONE_REACH)
    if (read_slot <= 0) | (deferred & (slot1 <= 0)):
        return
    cols, is_new, is_q, rep_use, tok, new_ok, q_ok = _mfma_cols(deferred, acc, n_tok, WIN)
    keep = is_new & (cols >= WIN) & (tok < n_tok) & (cur_zone == 0) & (slot1 > 0)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    rows = tl.arange(0, BV)
    w = ws + i_nh * WS
    p_kq = w + cols[:, None] * K + o_k[None, :]
    sq = cols[:, None] * 16 + cols[None, :]
    p = tl.load(w + 16 * K + sq)
    bm = tl.load(w + 16 * K + 256 + sq)
    bet = tl.load(w + 16 * K + 512 + cols)
    G = tl.load(w + 16 * K + 528 + cols)
    if not LEAN:
        kq = tl.load(p_kq)
    s_base = state + read_slot * stride_state_block
    slot1_base = state + slot1 * stride_state_block + i_hv * V * K
    v0 = i_g * NB * BV
    s_nxt = tl.load(s_base + i_hv * V * K + (v0 + rows)[:, None] * K + o_k[None, :])
    for blk in range(NB):
        o_v = v0 + blk * BV + rows
        tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        s0 = s_nxt
        s_nxt = tl.load(s_base + tile + BV * K, mask=blk + 1 < NB, other=0.0)
        rec_rows = slot1_base + (o_v // 16 * 16)[:, None] * K + (o_v % 16)[:, None]
        vt = tl.load(rec_rows + cols[None, :] * REC + K, mask=rep_use[None, :], other=0.0)
        vt += tl.load(qkv + (bos + tok)[None, :] * stride_qkv_l + 2 * H * K + i_hv * V
                      + o_v[:, None], mask=new_ok[None, :], other=0.0).to(tl.float32)
        if LEAN:
            kq = tl.load(p_kq)
        u = _mfma_out(s0, vt, kq, bet, G, p, bm, cols, is_q, q_ok,
                      o + (bos + tok)[None, :] * stride_o_l + i_hv * V + o_v[:, None], WIN)
        if RELOAD:
            s_acc = tl.load(s_base + tile)
        else:
            s_acc = s0
        if LEAN:
            kq = tl.load(p_kq)
        _mfma_commit(u, s_acc, kq, G, vt, cols, is_new, tok, n_tok, cur_zone, state, tile,
                     slot0, slot1, idx_row, rec_rows, stride_state_block, K, WIN, REC)
        _mfma_rec_kab(slot1_base + (v0 + blk * BV) * K, qkv, a, b, bos, tok, keep, i_h, i_hv,
                      stride_qkv_l, stride_a_l, stride_b_l, BV // 16, H, K, WIN, REC)


@triton.jit(do_not_specialize=["stride_x_l", "stride_a_l", "stride_b_l", "stride_o_l"])
def _gdn_prefill_kernel(
    A_log, a, b, dt_bias, beta, threshold, x, o, state, cu_seqlens, slots, has_init, scale,
    stride_x_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    # Short prefill chunks, token by token with _gdn_token: the state comes from the
    # request's slot (zeros without prior state) and goes back to it after the last token,
    # which is what vLLM's chunk path does through gather / FLA chunk kernels / scatter.
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    slot = tl.load(slots + i_n).to(tl.int64)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_v = tl.load(dt_bias + i_hv).to(tl.float32)
    b_h = tl.zeros([BV, K], dtype=tl.float32)
    if (tl.load(has_init + i_n) != 0) & (slot > 0):
        b_h += tl.load(state + slot * stride_state_block + tile).to(tl.float32)
    p_q = x + bos * stride_x_l + i_h * K + o_k
    p_k = x + bos * stride_x_l + H * K + i_h * K + o_k
    p_v = x + bos * stride_x_l + 2 * H * K + i_hv * V + o_v
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + o_v
    for _ in range(0, n_tok):
        b_q = tl.load(p_q).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        b_h = _gdn_token(b_h, tl.load(p_k).to(tl.float32), tl.load(p_v).to(tl.float32),
                         tl.load(p_a).to(tl.float32), tl.load(p_b).to(tl.float32), A_log_v,
                         dt_v, beta, threshold, USE_QK_L2NORM)
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty))
        p_q += stride_x_l
        p_k += stride_x_l
        p_v += stride_x_l
        p_a += stride_a_l
        p_b += stride_b_l
        p_o += stride_o_l
    if slot > 0:
        tl.store(state + slot * stride_state_block + tile, b_h)


def gdn_prefill(x, a, b, A_log, dt_bias, state, cu_seqlens, slots, has_init, out, num_k_heads,
                head_k_dim, head_v_dim, config=None):
    """Prefill chunks over post-conv packed qkv x [T, 2 H K + HV V] (row-strided), a / b
    [T, HV] views, the requests' state slots / has-initial-state flags, out [>= T, HV, V]."""
    BV, warps, stages = config or PREFILL_CONFIG
    n, hv = cu_seqlens.shape[0] - 1, a.shape[1]
    K, V = head_k_dim, head_v_dim
    assert state.dtype == torch.float32 and state.shape[1:] == (hv, V, K) and state[0].is_contiguous()
    assert x.stride(1) == a.stride(1) == b.stride(1) == 1 and out.stride(-2) == V
    if n:
        _gdn_prefill_kernel[(V // BV, n * hv)](
            A_log, a, b, dt_bias, 1.0, 20.0, x, out, state, cu_seqlens, slots, has_init,
            K**-0.5, x.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
            H=num_k_heads, HV=hv, K=K, V=V, BV=BV, USE_QK_L2NORM=True, num_warps=warps,
            num_stages=stages)
    return out


PREFILL_CONFIG = (16, 1, 3)  # BV, num_warps, num_stages of gdn_prefill (pipelined token loop)
CONFIG = (16, 1, 1)  # BV, num_warps, num_stages. MI350P k16 oracle, graphed us c1/c8/c32:
# 16/1w 15.0/52.5/179.1, 32/1w 20.3/54.0/173.8, 32/2w 20.0/56.3/196.2; AITER 19.9/66.4/225.5.
# SUFFIX_ROCM_GDN_DEFER_MFMA=1: gdn_defer runs the MFMA kernels. MFMA_CONFIG = BV, num_warps,
# RELOAD, LEAN, NB (0: one kernel; > 0: prep + main kernel over NB row blocks per program).
# Records are shared with v1 at its BV 16 (CONFIG above). MI350P c3014ec7 oracle, graphed us
# c1/c8/c32: 16/1w/rl 11.9/51.0/158.4 (v1 15.9/53.3/181.0; 232 VGPRs, occupancy 2).
MFMA = os.environ.get("SUFFIX_ROCM_GDN_DEFER_MFMA", "").strip() == "1"
MFMA_CONFIG = (16, 1, True, False, 0)
# From MFMA_PP_MIN_REQ requests on, the per-head pre-pass + main kernel (MI350P gate6,
# graphed us c1/c8/c32: v1 15.8/53.7/181.0, MFMA_CONFIG 11.7/50.0/162.0, this 13.8/42.8/127.6).
MFMA_PP_CONFIG = (16, 1, False, False, 1)
MFMA_PP_MIN_REQ = 4
# Tried (gate10): a Gluon main kernel loading each S0 tile once as both the fp32 MFMA operand and
# the commit accumulator (AITER's gfx950 Gluon GDN layouts): correct, but c8 / c32 51.5-56.5 /
# 172-201 us vs this pre-pass kernel's 43.1 / 127.3.


def gdn_defer(qkv, a, b, A_log, dt_bias, state, cu_seqlens, state_indices, num_accepted,
              seq_lens, out, num_k_heads, head_k_dim, head_v_dim, zone, config=None,
              mfma=None):
    """Spec-verify delta rule over packed post-conv qkv [T, 2 H K + HV V] (row-strided),
    a / b [T, HV] views (row-strided), state [blocks, HV, V, K] fp32 (vLLM's layer
    kv_cache[1]; slot ids <= 0 are NULL), state_indices [N, 1 + num_spec] (block ids),
    num_accepted / seq_lens [N], out [>= T, HV, V]. zone = mamba block size in align
    mode, else 0."""
    n = cu_seqlens.shape[0] - 1
    hv = a.shape[1]
    K, V = head_k_dim, head_v_dim
    win = state_indices.shape[1]
    if (MFMA if mfma is None else mfma) and 3 * win - 1 <= 16:
        BV, warps, reload, lean, nb = config or (MFMA_PP_CONFIG if n >= MFMA_PP_MIN_REQ else MFMA_CONFIG)
        rec = (K + 16 + 2 + 63) // 64 * 64  # v1's record at its BV 16
        assert K == triton.next_power_of_2(K) and BV % 16 == 0 and V % (BV * max(nb, 1)) == 0
        assert (win - 1) * rec <= 16 * K and state.dtype == torch.float32
        assert state.shape[1:] == (hv, V, K) and state[0].is_contiguous()
        assert qkv.stride(1) == a.stride(1) == b.stride(1) == 1 and state_indices.stride(1) == 1
        assert out.stride(-1) == 1 and out.stride(-2) == V
        if n == 0:
            return out
        common = dict(H=num_k_heads, HV=hv, K=K, V=V, WIN=win, ZONE=zone, ZONE_REACH=2 * win,
                      REC=rec, num_stages=1)
        if nb == 0:
            _gdn_defer_mfma_kernel[(V // BV, n * hv)](
                A_log, a, b, dt_bias, 1.0, 20.0, qkv, out, state, cu_seqlens, state_indices,
                num_accepted, seq_lens, K**-0.5,
                qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
                state_indices.stride(0), BV=BV, RELOAD=reload, LEAN=lean, num_warps=warps,
                **common)
            return out
        ws_size = 16 * K + 544  # k / q [16, K], P, bm [16, 16], bet, G [16]
        ws = torch.empty(n * hv * ws_size, device=qkv.device, dtype=torch.float32)
        _gdn_defer_mfma_prep_kernel[(n * hv,)](
            A_log, a, b, dt_bias, 1.0, 20.0, qkv, state, cu_seqlens, state_indices,
            num_accepted, seq_lens, K**-0.5, ws,
            qkv.stride(0), a.stride(0), b.stride(0), state.stride(0), state_indices.stride(0),
            WS=ws_size, num_warps=1, **common)
        _gdn_defer_mfma_main_kernel[(V // (BV * nb), n * hv)](
            a, b, qkv, out, state, cu_seqlens, state_indices, num_accepted, seq_lens, ws,
            qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
            state_indices.stride(0),
            BV=BV, NB=nb, WS=ws_size, RELOAD=reload, LEAN=lean, num_warps=warps, **common)
        return out
    BV, warps, stages = config or CONFIG
    rec = (K + BV + 2 + 63) // 64 * 64  # floats per recorded token: raw k, the v tile, a, b
    assert K == triton.next_power_of_2(K) and V % BV == 0 and (win - 1) * rec <= BV * K
    assert state.dtype == torch.float32 and state.stride(-1) == 1
    assert state.shape[1:] == (hv, V, K) and state[0].is_contiguous()
    assert qkv.stride(1) == a.stride(1) == b.stride(1) == 1 and state_indices.stride(1) == 1
    assert out.stride(-1) == 1 and out.stride(-2) == V
    if n == 0:
        return out
    _gdn_defer_kernel[(V // BV, n * hv)](
        A_log, a, b, dt_bias, 1.0, 20.0, qkv, out, state, cu_seqlens, state_indices,
        num_accepted, seq_lens, K**-0.5,
        qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
        state_indices.stride(0),
        H=num_k_heads, HV=hv, K=K, V=V, BV=BV, WIN=win, ZONE=zone, ZONE_REACH=2 * win,
        REC=rec,
        USE_QK_L2NORM=True, num_warps=warps, num_stages=stages)
    return out


def defer_spec(layer, mixed_qkv_spec, a_spec, b_spec, ssm_state, md):
    """_forward_core's spec verify under the deferred contract (mixed batches, and the
    all-spec ones forward_spec declines); returns core_attn_out_spec [1, T, HV, V]."""
    hv = layer.num_v_heads // layer.tp_size
    out = mixed_qkv_spec.new_empty((mixed_qkv_spec.shape[0], hv, layer.head_v_dim))
    gdn_defer(mixed_qkv_spec, a_spec, b_spec, layer.A_log, layer.dt_bias, ssm_state,
              md.spec_query_start_loc[: md.num_spec_decodes + 1], md.spec_state_indices_tensor,
              md.num_accepted_tokens, md.suffix_spec_seq_lens, out,
              layer.num_k_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim,
              md.suffix_zone)
    return out.unsqueeze(0)


def install(module) -> None:
    """rocm_patches `after` hook for qwen_gdn_linear_attn: its rewritten _forward_core
    calls _suffix_gdn_defer_spec (forward_spec checks SUFFIX_ROCM_GDN_DEFER itself)."""
    module._suffix_gdn_defer_spec = defer_spec


def _isa_stats(fn) -> list:
    """Compact AMDGCN facts of every compiled variant of a Triton kernel: registers,
    scratch spill bytes, LDS, occupancy, instruction mix (the oracle prints them)."""
    import re
    from collections import Counter

    rows = []
    for entry in getattr(fn, "device_caches", {}).values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for key, ck in getattr(cache, "items", lambda: [])():
            asm = (getattr(ck, "asm", {}) or {}).get("amdgcn", "")
            if not asm:
                continue

            def grab(tag):
                m = re.search(rf"; {tag}: (\d+)", asm)
                return m.group(1) if m else "?"

            ins = [ln.split()[0] for ln in asm.splitlines()
                   if ln.startswith("\t") and ln.strip() and ln.strip()[0] not in ";."]
            cls = Counter()
            for i in ins:
                cls["mfma" if "mfma" in i else "dpp/perm" if ("dpp" in i or "permlane" in i
                    or "bpermute" in i or "swizzle" in i or "readlane" in i) else
                    "lds" if i.startswith("ds_") else "mem" if i.startswith(("global_", "buffer_"))
                    else "valu" if i.startswith("v_") else "salu" if i.startswith("s_") else "other"] += 1
            consts = {k: v for k, v in (getattr(ck, "metadata", None)._asdict().items()
                                        if hasattr(getattr(ck, "metadata", None), "_asdict") else [])
                      if k in ("num_warps", "num_stages")}
            rows.append(f"vgpr {grab('NumVgprs')} agpr {grab('NumAgprs')} sgpr {grab('NumSgprs')} "
                        f"scratch {grab('ScratchSize')} lds {grab('LDSByteSize')} occ "
                        f"{grab('Occupancy')} instrs {len(ins)} "
                        + " ".join(f"{k} {v}" for k, v in cls.most_common()) + f" {consts} "
                        f"key {str(key)[-120:]}")
    return rows


def oracle_mfma() -> int:
    """SUFFIX_ROCM_GDN_DEFER_MFMA on silicon. No kernel shares its op order, so the yardstick
    is the fp64 sequential recurrence E on the same bf16 inputs. Pools stepped together
    through random acceptance with vLLM's align copies emulated: S = AITER, R = MFMA_CONFIG
    zone 1 (every slot, vLLM's read), X = deferred with v1 and MFMA_CONFIG alternating per
    step (shared record layout), one deferred pool per candidate config. Checks: R's
    committed states as close to E as AITER's; every pool's outputs within bf16 rounding of
    E; every deferred pool's boundary-copy slots equal to R's to fp32 rounding. Then graphed
    us/call v1 vs every MFMA config in the deferred steady state, and the AMDGCN facts of
    every compiled variant. Exit 0 = every check held."""
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as aiter_gdr)

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    mark = "[suffix gdn-mfma]"
    dev, bf16, f64 = "cuda", torch.bfloat16, torch.float64
    H, HV, K, V, win = 16, 48, 128, 128, 5
    key_dim, value_dim = H * K, HV * V
    torch.manual_seed(1)
    A_log = (0.5 * torch.randn(HV, device=dev)).float()
    dt_bias = (0.5 * torch.randn(HV, device=dev)).to(bf16)
    ok_all = True

    def run_stock(pool, qkv, ba, cu, idx, acc, out):
        b, a = ba.unflatten(-1, (2, HV)).transpose(0, 1).contiguous()
        aiter_gdr(A_log=A_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv, key_dim=key_dim,
                  value_dim=value_dim, head_k_dim=K, head_v_dim=V, initial_state=pool[1:],
                  inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=idx - 1,
                  num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True, core_attn_out=out)

    def run(pool, qkv, ba, cu, idx, acc, seq, out, zone, mfma, config=None):
        gdn_defer(qkv, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, idx, acc, seq, out,
                  H, K, V, zone, config, mfma)

    def ref_tokens(h, qkv_rows, ba_rows):  # fp64: states after each token and outputs
        x, a, bb = qkv_rows.to(f64), ba_rows[:, HV:].to(f64), ba_rows[:, :HV].to(f64)
        g = -torch.exp(A_log.to(f64)) * torch.nn.functional.softplus(a + dt_bias.to(f64))
        beta = torch.sigmoid(bb)
        states, outs = [], []
        for t in range(x.shape[0]):
            q = x[t, :key_dim].view(H, K)
            k = x[t, key_dim:2 * key_dim].view(H, K)
            q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            q, k = q.repeat_interleave(HV // H, 0), k.repeat_interleave(HV // H, 0)
            v = x[t, 2 * key_dim:].view(HV, V)
            h = h * torch.exp(g[t])[:, None, None]
            u = (v - torch.einsum("hvk,hk->hv", h, k)) * beta[t][:, None]
            h = h + u[:, :, None] * k[:, None, :]
            states.append(h)
            outs.append(torch.einsum("hvk,hk->hv", h, q))
        return states, torch.stack(outs)

    def bound(x, ref):  # in units of bf16 rounding of the reference
        return ((x.to(f64) - ref).abs() / (2**-7 * ref.abs() + 1e-3)).max().item()

    def q_of(step, r):
        return win if r % 7 else 1 + (step + r) % win

    def name(cfg):
        return (f"{cfg[0]}/{cfg[1]}w{'/rl' if cfg[2] else ''}{'/lean' if cfg[3] else ''}"
                f"{f'/pp{cfg[4]}' if cfg[4] else ''}")

    cands = ((16, 1, True, False, 0), (16, 1, True, True, 0), (32, 1, True, True, 0),
             (16, 1, False, False, 1), (16, 1, True, True, 2), (16, 1, False, False, 8),
             (32, 1, False, False, 4))
    arms = {"S": None, "R": None, "X": None, **{name(c): c for c in cands}}
    for zone, n_req, steps in ((0, 16, 10), (24, 16, 14), (1664, 16, 8)):
        pool_blocks = n_req * win * 4 + 1
        base = torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1
        pools = {k: base.clone() for k in arms}
        free = (torch.randperm(pool_blocks - 1) + 1).tolist()
        windows = [[free.pop() for _ in range(win)] for _ in range(n_req)]
        E = [base[w[0]].to(f64) for w in windows]  # committed fp64 states (acc = 1)
        base = None
        c = torch.randint(0, 3 * zone + 50 if zone else 4000, (n_req,)).tolist()
        acc = [1] * n_req
        st = {"R": 0.0, "S": 0.0, "Escale": 0.0}
        ob = {k: 0.0 for k in arms}
        copies = {k: True for k in arms if k not in ("S", "R")}
        checks, moves = 0, 0
        for step in range(steps):
            qlens = [q_of(step, r) for r in range(n_req)]
            cu = torch.zeros(n_req + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(qlens).cumsum(0)
            T, cu = int(cu[-1]), cu.to(dev)
            qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
            ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
            idx = torch.tensor(windows, dtype=torch.int32, device=dev)
            acc_t = torch.tensor(acc, dtype=torch.int32, device=dev)
            seq_t = torch.tensor([ci + q for ci, q in zip(c, qlens)], dtype=torch.int32,
                                 device=dev)
            outs = {k: torch.full((T, HV, V), float("nan"), device=dev, dtype=bf16)
                    for k in arms}
            run_stock(pools["S"], qkv, ba, cu, idx, acc_t, outs["S"])
            run(pools["R"], qkv, ba, cu, idx, acc_t, seq_t, outs["R"], 1, True)
            run(pools["X"], qkv, ba, cu, idx, acc_t, seq_t, outs["X"], zone, bool(step % 2))
            for k, cfg in arms.items():
                if cfg is not None:
                    run(pools[k], qkv, ba, cu, idx, acc_t, seq_t, outs[k], zone, True, cfg)
            new_acc = [int(torch.randint(1, q + 1, ()).item()) for q in qlens]
            for r in range(n_req):
                lo, hi = int(cu[r]), int(cu[r + 1])
                states, o_ref = ref_tokens(E[r], qkv[lo:hi], ba[lo:hi])
                for k in arms:
                    ob[k] = max(ob[k], bound(outs[k][lo:hi], o_ref))
                E[r] = states[new_acc[r] - 1]
                sl = windows[r][new_acc[r] - 1]
                st["Escale"] = max(st["Escale"], E[r].abs().max().item())
                for k in ("R", "S"):
                    st[k] = max(st[k], (pools[k][sl].to(f64) - E[r]).abs().max().item())
            for r in range(n_req):
                if not zone:
                    continue
                lo, hi = c[r] + 1, c[r] + new_acc[r]
                bnd = hi // zone * zone
                if bnd >= lo:
                    sl = windows[r][bnd - lo]
                    checks += 1
                    for k in copies:
                        copies[k] &= torch.allclose(pools[k][sl], pools["R"][sl], rtol=1e-4,
                                                    atol=1e-5)
                if ((c[r] + qlens[r] - 1) // zone
                        != (c[r] + new_acc[r] + q_of(step + 1, r) - 1) // zone):
                    src = windows[r][new_acc[r] - 1]
                    new_win = [free.pop() for _ in range(win)]
                    for pool in pools.values():
                        pool[new_win[0]].copy_(pool[src])
                    free.extend(windows[r])
                    windows[r], new_acc[r] = new_win, 1
                    moves += 1
            c = [ci + ai for ci, ai in zip(c, new_acc)]
            acc = new_acc
        bad = [k for k in arms if ob[k] > 1.0 or not copies.get(k, True)]
        ok = st["R"] <= max(4 * st["S"], 1e-5 * st["Escale"]) and not bad
        ok_all &= ok
        print(f"{mark} zone {zone}, {n_req} requests x {steps} steps: committed state max abs "
              f"err vs fp64 mfma {st['R']:.2e} / AITER {st['S']:.2e} (|state| <= "
              f"{st['Escale']:.1f}); outputs vs fp64 in bf16 bounds (R = {name(MFMA_CONFIG)} "
              f"stock-way, X = v1 / MFMA alternating): "
              + " ".join(f"{k} {v:.2f}" for k, v in ob.items())
              + f"; boundary copies {checks}, moves {moves}; failing {bad or 'none'} -> "
              f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
        pools = None  # free the pools before the next size

    # Steady state (accepted 1..5, zone 0: every step replays), graphed. Correctness of
    # every config is the scenarios above; this only times.
    for n_req in (1, 8, 32):
        pool_blocks = n_req * win * 4 + 1
        pools = [torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1 for _ in range(4)]
        cu = (torch.arange(n_req + 1, dtype=torch.int32) * win).to(dev)
        T = n_req * win
        qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
        ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
        idx = ((torch.randperm(pool_blocks - 1)[: n_req * win] + 1).view(n_req, win)
               .to(torch.int32).to(dev))
        acc_t = torch.randint(1, win + 1, (n_req,), dtype=torch.int32, device=dev)
        seq_t = (torch.randint(100, 1500, (n_req,), dtype=torch.int32) + win).to(dev)
        out = torch.empty(T, HV, V, device=dev, dtype=bf16)
        t_v1 = _graph_us(lambda i: run(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t, out, 0,
                                       False), 8)[0]
        sweep = []
        for cfg in ((16, 1, False, False, 0), (16, 1, True, False, 0), (16, 1, False, True, 0),
                    (16, 1, True, True, 0), (32, 1, True, True, 0), (32, 2, True, True, 0),
                    (16, 1, False, False, 1), (16, 1, True, True, 1), (16, 1, False, False, 2),
                    (16, 1, False, True, 2), (16, 1, True, True, 2), (16, 1, False, False, 4),
                    (16, 1, False, True, 4), (16, 1, False, False, 8), (16, 1, False, True, 8),
                    (32, 1, False, False, 2), (32, 1, False, False, 4), (32, 2, False, False, 2),
                    (64, 4, False, False, 2)):
            try:
                us = _graph_us(lambda i: run(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t,
                                             out, 0, True, cfg), 8)[0]
                sweep.append(f"{name(cfg)} {us:.1f}")
            except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                sweep.append(f"{name(cfg)} {type(exc).__name__}: {str(exc)[:160]!r}")
        print(f"{mark} c{n_req} deferred steady state graphed: v1 {t_v1:.1f} us | mfma "
              f"{' | '.join(sweep)}", flush=True)
        pools = None  # free the pools before the next size
    for kname, fn in (("v1", _gdn_defer_kernel), ("mfma", _gdn_defer_mfma_kernel),
                      ("mfma-prep", _gdn_defer_mfma_prep_kernel),
                      ("mfma-main", _gdn_defer_mfma_main_kernel)):
        for row in _isa_stats(fn):
            print(f"{mark} isa {kname}: {row}", flush=True)
    return 0 if ok_all else 1


def main() -> int:
    """Multi-step oracle on silicon at Qwen3.8-Flash-Next GDN shapes (16 qk / 48 v heads
    x 128, MTP-4), three state pools stepped together through random acceptance with
    vLLM's align-mode copies emulated on each (post-step boundary copy, pre-step
    running-block move with the accepted-count reset):
      S  AITER's fused_rearrange_sigmoid_gated_delta_rule (what SUFFIX_ROCM_GDN_MTP serves),
      R  gdn_defer with zone 1: every step writes all slots and reads vLLM's way,
      D  gdn_defer under test (zone 0 = never stock-way, 24 = often, 1664 = serving).
    D must equal R bitwise (outputs and every slot a boundary copy reads): the replay is
    exact. R must stay within bf16 rounding of S (same math, other fp32 op schedule).
    Then graphed us/call at c1/c8/c32 in the deferred steady state."""
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as aiter_gdr)

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    dev, bf16 = "cuda", torch.bfloat16
    H, HV, K, V, spec = 16, 48, 128, 128, 4
    win, key_dim, value_dim = spec + 1, H * K, HV * V
    torch.manual_seed(0)
    A_log = (0.5 * torch.randn(HV, device=dev)).float()
    dt_bias = (0.5 * torch.randn(HV, device=dev)).to(bf16)
    failed = False

    def run_stock(pool, qkv, ba, cu, idx, acc, out):
        b, a = ba.unflatten(-1, (2, HV)).transpose(0, 1).contiguous()
        aiter_gdr(A_log=A_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv, key_dim=key_dim,
                  value_dim=value_dim, head_k_dim=K, head_v_dim=V, initial_state=pool[1:],
                  inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=idx - 1,
                  num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True, core_attn_out=out)

    def run_defer(pool, qkv, ba, cu, idx, acc, seq, out, zone, config=None):
        gdn_defer(qkv, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, idx, acc, seq, out,
                  H, K, V, zone, config)

    def q_of(step, r):  # a few rows draft fewer tokens
        return win if r % 7 else 1 + (step + r) % win

    for zone, n_req, steps in ((0, 32, 10), (24, 32, 14), (1664, 32, 10), (24, 8, 14)):
        pool_blocks = n_req * win * 4 + 1
        pools = {"S": torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1}
        pools["R"], pools["D"] = pools["S"].clone(), pools["S"].clone()
        free = (torch.randperm(pool_blocks - 1) + 1).tolist()
        windows = [[free.pop() for _ in range(win)] for _ in range(n_req)]
        c = torch.randint(0, 3 * zone + 50 if zone else 4000, (n_req,)).tolist()
        acc = [1] * n_req
        exact, worst_rs, checks, moves, stock_way = True, 0.0, 0, 0, 0
        for step in range(steps):
            qlens = [q_of(step, r) for r in range(n_req)]
            cu = torch.zeros(n_req + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(qlens).cumsum(0)
            cu = cu.to(dev)
            T = int(cu[-1])
            qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
            ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
            idx = torch.tensor(windows, dtype=torch.int32, device=dev)
            acc_t = torch.tensor(acc, dtype=torch.int32, device=dev)
            seq_t = torch.tensor([ci + q for ci, q in zip(c, qlens)], dtype=torch.int32,
                                 device=dev)
            if zone:
                stock_way += sum((ci + 2 * win) // zone > ci // zone for ci in c)
            outs = {k: torch.full((T, HV, V), float("nan"), device=dev, dtype=bf16)
                    for k in pools}
            run_stock(pools["S"], qkv, ba, cu, idx, acc_t, outs["S"])
            run_defer(pools["R"], qkv, ba, cu, idx, acc_t, seq_t, outs["R"], 1)
            run_defer(pools["D"], qkv, ba, cu, idx, acc_t, seq_t, outs["D"], zone)
            exact &= torch.equal(outs["D"], outs["R"])
            worst_rs = max(worst_rs, ((outs["R"].float() - outs["S"].float()).abs()
                                      / (2**-7 * outs["S"].float().abs() + 1e-3)).max().item())
            new_acc = [int(torch.randint(1, q + 1, ()).item()) for q in qlens]
            for r in range(n_req):
                if not zone:
                    continue
                # Post-step align copy source: the slot holding the state at the boundary.
                lo, hi = c[r] + 1, c[r] + new_acc[r]
                bnd = hi // zone * zone
                if bnd >= lo:
                    sl = windows[r][bnd - lo]
                    checks += 1
                    exact &= torch.equal(pools["D"][sl], pools["R"][sl])
                # Pre-step running-block move: copy the committed state (slot acc - 1) to a
                # new window's slot 0 and reset acc to 1, on every pool.
                if ((c[r] + qlens[r] - 1) // zone
                        != (c[r] + new_acc[r] + q_of(step + 1, r) - 1) // zone):
                    src = windows[r][new_acc[r] - 1]
                    new_win = [free.pop() for _ in range(win)]
                    for pool in pools.values():
                        pool[new_win[0]].copy_(pool[src])
                    free.extend(windows[r])
                    windows[r], new_acc[r] = new_win, 1
                    moves += 1
            c = [ci + ai for ci, ai in zip(c, new_acc)]
            acc = new_acc
        ok = exact and worst_rs <= 1.0
        failed |= not ok
        print(f"{MARK} zone {zone}, {n_req} requests x {steps} steps: deferred vs stock-way "
              f"{'bitwise' if exact else 'MISMATCH'}, vs AITER worst {worst_rs:.2f} of the "
              f"bf16 bound -> {'MATCH' if ok else 'MISMATCH'} | stock-way steps {stock_way}, "
              f"boundary copies checked {checks}, running-block moves {moves}", flush=True)

    # Short prefills: gdn_prefill vs vLLM's chunk path (fused_post_conv_prep + FLA chunk
    # kernels + state gather / scatter), same inputs and slots.
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.index import (prepare_chunk_indices,
                                                                   prepare_chunk_offsets)
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    for lens, init in (((60,), (True,)), ((3, 61, 17), (True, False, True)), ((256,), (False,)),
                       ((1, 2, 128, 40), (True, True, False, True))):
        cu_cpu = torch.tensor([0] + list(lens), dtype=torch.int32).cumsum(0).to(torch.int32)
        cu = cu_cpu.to(dev)
        ci = prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to(dev)  # as the GDN builder does
        co = prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to(dev)
        T, n = int(cu[-1]), len(lens)
        x = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
        ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
        pool = torch.randn(2 * n + 2, HV, V, K, device=dev) * 0.1
        slots = (torch.randperm(2 * n + 1)[:n] + 1).to(torch.int32).to(dev)
        has = torch.tensor(init, device=dev)
        ref_pool, pool_before = pool.clone(), pool.clone()
        q, k, v, g, beta = fused_post_conv_prep(conv_output=x, a=ba[:, HV:], b=ba[:, :HV],
                                                A_log=A_log, dt_bias=dt_bias, num_k_heads=H,
                                                head_k_dim=K, head_v_dim=V, apply_l2norm=True,
                                                output_g_exp=False)
        init_state = ref_pool[slots.long()]
        init_state[~has] = 0
        o_ref, last = chunk_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0),
            initial_state=init_state, output_final_state=True, cu_seqlens=cu, chunk_indices=ci,
            chunk_offsets=co)
        ref_pool[slots.long()] = last.to(ref_pool.dtype)
        out = torch.full((T, HV, V), float("nan"), device=dev, dtype=bf16)
        gdn_prefill(x, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, slots, has, out, H, K, V)
        do = (out.float() - o_ref.squeeze(0).float()).abs().max().item()
        ds = (pool - ref_pool).abs().max().item()
        # Exact reference: the same recurrence in fp64 on the bf16 inputs; which path is closer?
        exact = pool_before.double().clone()
        xf, af, bf = x.double(), ba[:, HV:].double(), ba[:, :HV].double()
        g64 = -torch.exp(A_log.double()) * torch.nn.functional.softplus(af + dt_bias.double())
        b64 = torch.sigmoid(bf)
        for r in range(n):
            h = exact[int(slots[r])] if bool(has[r]) else torch.zeros_like(exact[0])
            for t in range(int(cu[r]), int(cu[r + 1])):
                kk = xf[t, key_dim:2 * key_dim].view(H, K)
                kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
                kk = kk.repeat_interleave(HV // H, 0)  # [HV, K]
                vv = xf[t, 2 * key_dim:].view(HV, V)
                h = h * torch.exp(g64[t])[:, None, None]
                u = (vv - torch.einsum("hvk,hk->hv", h, kk)) * b64[t][:, None]
                h = h + u[:, :, None] * kk[:, None, :]
            exact[int(slots[r])] = h
        err_new = (pool.double() - exact).abs().max().item()
        err_ref = (ref_pool.double() - exact).abs().max().item()
        # The chunk path carries bf16 intermediates (WY / solve_tril), the recurrent one fp32:
        # they agree to bf16 rounding, not to fp32.
        ok = (torch.allclose(out.float(), o_ref.squeeze(0).float(), rtol=2e-2, atol=2e-2)
              and torch.allclose(pool, ref_pool, rtol=2e-2, atol=2e-2))
        failed |= not ok
        t_ref = _graph_us(lambda i: chunk_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0),
            initial_state=init_state, output_final_state=True, cu_seqlens=cu, chunk_indices=ci,
            chunk_offsets=co), 4)[0]
        t_new = _graph_us(lambda i: gdn_prefill(x, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool,
                                                cu, slots, has, out, H, K, V), 4)[0]
        sweep = []
        for cfg in ((16, 1, 1), (16, 1, 3), (32, 1, 3), (32, 2, 3), (64, 2, 3), (64, 4, 3)):
            us = _graph_us(lambda i: gdn_prefill(x, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool,
                                                 cu, slots, has, out, H, K, V, cfg), 4)[0]
            sweep.append(f"{cfg[0]}/{cfg[1]}w/{cfg[2]}s {us:.1f}")
        print(f"{MARK} prefill lens {lens}: {'MATCH' if ok else 'MISMATCH'} vs vLLM chunk path "
              f"(max abs diff out {do:.2e}, state {ds:.2e}; vs fp64: recurrent {err_new:.2e}, "
              f"chunk {err_ref:.2e}) | graphed FLA chunk {t_ref:.1f} us "
              f"(+post-conv, gather, scatter) -> recurrent {t_new:.1f} us | sweep "
              f"{' | '.join(sweep)}", flush=True)

    # Steady state (accepted 1..5), graphed: AITER (5 state writes), gdn_defer stock-way
    # (zone 1, 5 writes) and deferred (zone 0: 1 state write + the record).
    for n_req in (1, 8, 32):
        pool_blocks = n_req * win * 4 + 1
        pools = [torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1 for _ in range(4)]
        cu = (torch.arange(n_req + 1, dtype=torch.int32) * win).to(dev)
        T = n_req * win
        qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
        ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
        idx = ((torch.randperm(pool_blocks - 1)[: n_req * win] + 1).view(n_req, win)
               .to(torch.int32).to(dev))
        acc_t = torch.randint(1, win + 1, (n_req,), dtype=torch.int32, device=dev)
        seq_t = (torch.randint(100, 1500, (n_req,), dtype=torch.int32) + win).to(dev)
        out = torch.empty(T, HV, V, device=dev, dtype=bf16)
        t_s = _graph_us(lambda i: run_stock(pools[i % 4], qkv, ba, cu, idx, acc_t, out), 8)[0]
        sweep = {0: [], 1: []}
        for cfg in ((32, 2, 1), (32, 1, 1), (16, 1, 1), (16, 2, 1), (64, 1, 1), (64, 2, 1),
                    (64, 4, 1), (128, 2, 1), (128, 4, 1)):
            for zone in (0, 1):  # deferred, stock-way
                try:
                    us = _graph_us(lambda i: run_defer(pools[i % 4], qkv, ba, cu, idx, acc_t,
                                                       seq_t, out, zone, cfg), 8)[0]
                    sweep[zone].append(f"{cfg[0]}/{cfg[1]}w {us:.1f}")
                except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                    sweep[zone].append(f"{cfg[0]}/{cfg[1]}w {type(exc).__name__}")
        print(f"{MARK} c{n_req} x mtp5 steady state graphed: AITER {t_s:.1f} us | deferred "
              f"BV/warps: {' | '.join(sweep[0])} | stock-way: {' | '.join(sweep[1])}",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(oracle_mfma() if sys.argv[1:] == ["mfma"] else main())
