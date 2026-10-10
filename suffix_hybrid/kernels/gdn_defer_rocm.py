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

    python -m suffix_hybrid.kernels.gdn_defer_rocm   # multi-step GPU oracle vs AITER + us/call
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


@triton.jit
def _dot3(x, y):  # [NC, 1, KC] . [NC, 1, KC] -> scalar
    return tl.sum(tl.sum(tl.sum(x * y, axis=0), axis=1), axis=0)


@triton.jit
def _rows(h, x):  # state tile [NC, BV, KC] times a K vector [NC, 1, KC] -> [BV]
    return tl.sum(tl.sum(h * x, axis=0), axis=1)  # register chunks first, then KC lanes


@triton.jit
def _v2_tok(p_q, p_k, p_v, p_a, p_b, i_t, n_tok, s_qkv, s_a, s_b, A_log_v, dt_v, beta,
            threshold, scale, h):
    # Token i_t's normalized q (x scale) / k, v tile, log decay g (0 past n_tok), beta,
    # and the initial state's rows against k and q (P_t, R_t).
    valid = i_t < n_tok
    b_q = tl.load(p_q + i_t * s_qkv, mask=valid, other=0.0).to(tl.float32)
    b_k = tl.load(p_k + i_t * s_qkv, mask=valid, other=0.0).to(tl.float32)
    b_v = tl.load(p_v + i_t * s_qkv, mask=valid, other=0.0).to(tl.float32)
    a_raw = tl.load(p_a + i_t * s_a, mask=valid, other=0.0).to(tl.float32)
    b_raw = tl.load(p_b + i_t * s_b, mask=valid, other=0.0).to(tl.float32)
    b_q = b_q * (tl.rsqrt(_dot3(b_q, b_q) + 1e-6) * scale)
    b_k = b_k * tl.rsqrt(_dot3(b_k, b_k) + 1e-6)
    x = a_raw + dt_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    b_g = tl.where(valid, -tl.exp(A_log_v) * softplus_x, 0.0)
    return b_q, b_k, b_v, b_g, tl.sigmoid(b_raw), _rows(h, b_k), _rows(h, b_q)


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer2_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, o, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr,
    REC: tl.constexpr,
):
    # Same slots / zones / record placement as _gdn_defer_kernel (WIN = 5 only), other math:
    # a record holds token t's normalized k, its u = beta (v - decayed S k) tile and its
    # log decay since token 0, so the replay is one rank-(acc - 1) update of slot 0's state
    # instead of acc - 1 sequential tokens; the step's 5 tokens run in chunk form off the
    # initial state S: P_t = S k_t and R_t = S q_t (10 independent row reductions), then
    #   u_t = beta_t (v_t - e^G_t P_t - sum_{i<t} e^(G_t - G_i) (k_i . k_t) u_i)
    #   o_t = e^G_t R_t + sum_{i<=t} e^(G_t - G_i) (k_i . q_t) u_i
    # on [BV] vectors (G = cumulative log decay). Only the state after token 0 is built
    # (slot 0), the others only in stock-way zones. The tile is [NC, BV, KC] (K = NC x KC
    # chunks in registers) so a row reduction is mostly in-lane.
    KC: tl.constexpr = K // NC
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return

    o_bv = tl.arange(0, BV)
    kofs = tl.arange(0, NC)[:, None, None] * KC + tl.arange(0, KC)[None, None, :]
    tile = i_hv * V * K + (i_v * BV + o_bv)[None, :, None] * K + kofs  # [NC, BV, KC]
    tile_base = i_hv * V * K + i_v * BV * K
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_v = tl.load(dt_bias + i_hv).to(tl.float32)
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
    if read_slot <= 0:
        return
    if deferred & (slot1 <= 0):
        return
    b_h = tl.load(state + read_slot * stride_state_block + tile)
    rec = state + slot1 * stride_state_block + tile_base
    if deferred:  # S = e^Gr_last S0 + sum_j e^(Gr_last - Gr_j) u_j k_j^T, j = 1 .. acc - 1
        gr_last = tl.load(rec + (acc - 2) * REC + K + BV)
        b_h = b_h * tl.exp(gr_last)
        for j in tl.static_range(1, WIN):
            if j < acc:
                r = rec + (j - 1) * REC
                c_j = tl.exp(gr_last - tl.load(r + K + BV))
                b_h += (c_j * tl.load(r + K + o_bv))[None, :, None] * tl.load(r + kofs)

    p_q = qkv + bos * stride_qkv_l + i_h * K + kofs
    p_k = qkv + bos * stride_qkv_l + H * K + i_h * K + kofs
    p_v = qkv + bos * stride_qkv_l + 2 * H * K + i_hv * V + i_v * BV + o_bv
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + i_v * BV + o_bv
    sq, sa, sb, so = stride_qkv_l, stride_a_l, stride_b_l, stride_o_l
    q0, k0, v0, g0, be0, P0, R0 = _v2_tok(p_q, p_k, p_v, p_a, p_b, 0, n_tok, sq, sa, sb,
                                          A_log_v, dt_v, beta, threshold, scale, b_h)
    G0 = g0
    u0 = be0 * (v0 - tl.exp(G0) * P0)
    tl.store(p_o, (tl.exp(G0) * R0 + _dot3(k0, q0) * u0).to(p_o.dtype.element_ty))

    q1, k1, v1, g1, be1, P1, R1 = _v2_tok(p_q, p_k, p_v, p_a, p_b, 1, n_tok, sq, sa, sb,
                                          A_log_v, dt_v, beta, threshold, scale, b_h)
    G1 = G0 + g1
    d10 = tl.exp(G1 - G0)
    u1 = be1 * (v1 - (tl.exp(G1) * P1 + (d10 * _dot3(k0, k1)) * u0))
    o1 = tl.exp(G1) * R1 + (d10 * _dot3(k0, q1)) * u0 + _dot3(k1, q1) * u1
    tl.store(p_o + so, o1.to(p_o.dtype.element_ty), mask=1 < n_tok)

    q2, k2, v2, g2, be2, P2, R2 = _v2_tok(p_q, p_k, p_v, p_a, p_b, 2, n_tok, sq, sa, sb,
                                          A_log_v, dt_v, beta, threshold, scale, b_h)
    G2 = G1 + g2
    d20, d21 = tl.exp(G2 - G0), tl.exp(G2 - G1)
    u2 = be2 * (v2 - (tl.exp(G2) * P2 + (d20 * _dot3(k0, k2)) * u0
                      + (d21 * _dot3(k1, k2)) * u1))
    o2 = (tl.exp(G2) * R2 + (d20 * _dot3(k0, q2)) * u0 + (d21 * _dot3(k1, q2)) * u1
          + _dot3(k2, q2) * u2)
    tl.store(p_o + 2 * so, o2.to(p_o.dtype.element_ty), mask=2 < n_tok)

    q3, k3, v3, g3, be3, P3, R3 = _v2_tok(p_q, p_k, p_v, p_a, p_b, 3, n_tok, sq, sa, sb,
                                          A_log_v, dt_v, beta, threshold, scale, b_h)
    G3 = G2 + g3
    d30, d31, d32 = tl.exp(G3 - G0), tl.exp(G3 - G1), tl.exp(G3 - G2)
    u3 = be3 * (v3 - (tl.exp(G3) * P3 + (d30 * _dot3(k0, k3)) * u0
                      + (d31 * _dot3(k1, k3)) * u1 + (d32 * _dot3(k2, k3)) * u2))
    o3 = (tl.exp(G3) * R3 + (d30 * _dot3(k0, q3)) * u0 + (d31 * _dot3(k1, q3)) * u1
          + (d32 * _dot3(k2, q3)) * u2 + _dot3(k3, q3) * u3)
    tl.store(p_o + 3 * so, o3.to(p_o.dtype.element_ty), mask=3 < n_tok)

    q4, k4, v4, g4, be4, P4, R4 = _v2_tok(p_q, p_k, p_v, p_a, p_b, 4, n_tok, sq, sa, sb,
                                          A_log_v, dt_v, beta, threshold, scale, b_h)
    G4 = G3 + g4
    d40, d41, d42, d43 = tl.exp(G4 - G0), tl.exp(G4 - G1), tl.exp(G4 - G2), tl.exp(G4 - G3)
    u4 = be4 * (v4 - (tl.exp(G4) * P4 + (d40 * _dot3(k0, k4)) * u0
                      + (d41 * _dot3(k1, k4)) * u1 + (d42 * _dot3(k2, k4)) * u2
                      + (d43 * _dot3(k3, k4)) * u3))
    o4 = (tl.exp(G4) * R4 + (d40 * _dot3(k0, q4)) * u0 + (d41 * _dot3(k1, q4)) * u1
          + (d42 * _dot3(k2, q4)) * u2 + (d43 * _dot3(k3, q4)) * u3 + _dot3(k4, q4) * u4)
    tl.store(p_o + 4 * so, o4.to(p_o.dtype.element_ty), mask=4 < n_tok)

    # State after token 0 -> slot 0; tokens 1..4 -> records, or (zone) their states.
    h = b_h * tl.exp(g0) + u0[None, :, None] * k0
    tl.store(state + slot0 * stride_state_block + tile, h, mask=slot0 > 0)
    if cur_zone:
        h = h * tl.exp(g1) + u1[None, :, None] * k1
        s = tl.load(idx_row + 1).to(tl.int64)
        tl.store(state + s * stride_state_block + tile, h, mask=(1 < n_tok) & (s > 0))
        h = h * tl.exp(g2) + u2[None, :, None] * k2
        s = tl.load(idx_row + 2).to(tl.int64)
        tl.store(state + s * stride_state_block + tile, h, mask=(2 < n_tok) & (s > 0))
        h = h * tl.exp(g3) + u3[None, :, None] * k3
        s = tl.load(idx_row + 3).to(tl.int64)
        tl.store(state + s * stride_state_block + tile, h, mask=(3 < n_tok) & (s > 0))
        h = h * tl.exp(g4) + u4[None, :, None] * k4
        s = tl.load(idx_row + 4).to(tl.int64)
        tl.store(state + s * stride_state_block + tile, h, mask=(4 < n_tok) & (s > 0))
    else:
        keep = slot1 > 0
        tl.store(rec + kofs, k1, mask=keep & (1 < n_tok))
        tl.store(rec + K + o_bv, u1, mask=keep & (1 < n_tok))
        tl.store(rec + K + BV, G1 - G0, mask=keep & (1 < n_tok))
        tl.store(rec + REC + kofs, k2, mask=keep & (2 < n_tok))
        tl.store(rec + REC + K + o_bv, u2, mask=keep & (2 < n_tok))
        tl.store(rec + REC + K + BV, G2 - G0, mask=keep & (2 < n_tok))
        tl.store(rec + 2 * REC + kofs, k3, mask=keep & (3 < n_tok))
        tl.store(rec + 2 * REC + K + o_bv, u3, mask=keep & (3 < n_tok))
        tl.store(rec + 2 * REC + K + BV, G3 - G0, mask=keep & (3 < n_tok))
        tl.store(rec + 3 * REC + kofs, k4, mask=keep & (4 < n_tok))
        tl.store(rec + 3 * REC + K + o_bv, u4, mask=keep & (4 < n_tok))
        tl.store(rec + 3 * REC + K + BV, G4 - G0, mask=keep & (4 < n_tok))


@triton.jit
def _gdn_token3(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold):
    # _gdn_token on the [NC, BV, KC] tile (b_k [NC, 1, KC], b_v [BV]).
    x = a_raw + dt_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    b_g = -tl.exp(A_log_v) * softplus_x
    b_beta = tl.sigmoid(b_raw)
    b_k = b_k * tl.rsqrt(_dot3(b_k, b_k) + 1e-6)
    b_h *= tl.exp(b_g)
    b_v -= _rows(b_h, b_k)
    b_v *= b_beta
    b_h += b_v[None, :, None] * b_k
    return b_h


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer3_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, o, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr,
    REC: tl.constexpr,
):
    # _gdn_defer_kernel line for line (same records, same sequential per-token math) on a
    # [NC, BV, KC] tile: the oracle's layout probe (only reduction orders differ).
    KC: tl.constexpr = K // NC
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    o_bv = tl.arange(0, BV)
    kofs = tl.arange(0, NC)[:, None, None] * KC + tl.arange(0, KC)[None, None, :]
    tile = i_hv * V * K + (i_v * BV + o_bv)[None, :, None] * K + kofs
    tile_base = i_hv * V * K + i_v * BV * K
    two = tl.arange(0, 2)
    rec_v = K + o_bv
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_v = tl.load(dt_bias + i_hv).to(tl.float32)
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
    if read_slot <= 0:
        return
    if deferred & (slot1 <= 0):
        return
    b_h = tl.load(state + read_slot * stride_state_block + tile)
    rec = state + slot1 * stride_state_block + tile_base
    for t in tl.static_range(1, WIN):
        use = deferred & (t < acc)
        r = rec + (t - 1) * REC
        r_k = tl.load(r + kofs, mask=use, other=0.0)
        r_v = tl.load(r + rec_v, mask=use, other=0.0)
        r_ab = tl.load(r + K + BV + two, mask=use, other=0.0)
        if use:
            b_h = _gdn_token3(b_h, r_k, r_v, tl.sum(tl.where(two == 0, r_ab, 0.0)),
                              tl.sum(tl.where(two == 1, r_ab, 0.0)), A_log_v, dt_v, beta,
                              threshold)
    p_q = qkv + bos * stride_qkv_l + i_h * K + kofs
    p_k = qkv + bos * stride_qkv_l + H * K + i_h * K + kofs
    p_v = qkv + bos * stride_qkv_l + 2 * H * K + i_hv * V + i_v * BV + o_bv
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + i_v * BV + o_bv
    for i_t in tl.static_range(WIN):
        valid = i_t < n_tok
        b_q = tl.load(p_q + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_k = tl.load(p_k + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_v = tl.load(p_v + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        a_raw = tl.load(p_a + i_t * stride_a_l, mask=valid, other=0.0).to(tl.float32)
        b_raw = tl.load(p_b + i_t * stride_b_l, mask=valid, other=0.0).to(tl.float32)
        b_q = b_q * tl.rsqrt(_dot3(b_q, b_q) + 1e-6)
        b_q = b_q * scale
        h_new = _gdn_token3(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold)
        b_h = tl.where(valid, h_new, b_h)
        b_o = _rows(b_h, b_q)
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
            tl.store(r + kofs, b_k, mask=keep)
            tl.store(r + rec_v, b_v, mask=keep)
            tl.store(r + K + BV + two, tl.where(two == 0, a_raw, b_raw), mask=keep)


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
# SUFFIX_ROCM_GDN_DEFER_V2=1: gdn_defer runs _gdn_defer2_kernel (rank-update replay, chunk-form
# verify; its records differ from v1's, so one pod never mixes the two).
_V2 = os.environ.get("SUFFIX_ROCM_GDN_DEFER_V2", "").strip() == "1"
CONFIG2 = (16, 1, 1, 4)  # BV, num_warps, num_stages, NC (K chunks held in registers)
CONFIG = (16, 1, 1)  # BV, num_warps, num_stages. MI350P k16 oracle, graphed us c1/c8/c32:
# 16/1w 15.0/52.5/179.1, 32/1w 20.3/54.0/173.8, 32/2w 20.0/56.3/196.2; AITER 19.9/66.4/225.5.


def gdn_defer(qkv, a, b, A_log, dt_bias, state, cu_seqlens, state_indices, num_accepted,
              seq_lens, out, num_k_heads, head_k_dim, head_v_dim, zone, config=None, v2=None):
    """Spec-verify delta rule over packed post-conv qkv [T, 2 H K + HV V] (row-strided),
    a / b [T, HV] views (row-strided), state [blocks, HV, V, K] fp32 (vLLM's layer
    kv_cache[1]; slot ids <= 0 are NULL), state_indices [N, 1 + num_spec] (block ids),
    num_accepted / seq_lens [N], out [>= T, HV, V]. zone = mamba block size in align
    mode, else 0."""
    n = cu_seqlens.shape[0] - 1
    hv = a.shape[1]
    K, V = head_k_dim, head_v_dim
    win = state_indices.shape[1]
    if v2 == "v13":  # oracle only: v1 on the [NC, BV, KC] tile (v1's records)
        BV, warps, stages, nc = config
        rec = (K + BV + 2 + 63) // 64 * 64
        if n:
            _gdn_defer3_kernel[(V // BV, n * hv)](
                A_log, a, b, dt_bias, 1.0, 20.0, qkv, out, state, cu_seqlens, state_indices,
                num_accepted, seq_lens, K**-0.5,
                qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
                state_indices.stride(0),
                H=num_k_heads, HV=hv, K=K, V=V, BV=BV, NC=nc, WIN=win, ZONE=zone,
                ZONE_REACH=2 * win, REC=rec, num_warps=warps, num_stages=stages)
        return out
    if _V2 if v2 is None else v2:
        BV, warps, stages, nc = config or CONFIG2
        rec = (K + BV + 1 + 63) // 64 * 64  # floats per recorded token: k, the u tile, Gr
        assert win == 5 and K % nc == 0 and V % BV == 0 and (win - 1) * rec <= BV * K
        assert state.dtype == torch.float32 and state.shape[1:] == (hv, V, K)
        assert state[0].is_contiguous() and state_indices.stride(1) == 1
        assert qkv.stride(1) == a.stride(1) == b.stride(1) == 1
        assert out.stride(-1) == 1 and out.stride(-2) == V
        if n:
            _gdn_defer2_kernel[(V // BV, n * hv)](
                A_log, a, b, dt_bias, 1.0, 20.0, qkv, out, state, cu_seqlens, state_indices,
                num_accepted, seq_lens, K**-0.5,
                qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
                state_indices.stride(0),
                H=num_k_heads, HV=hv, K=K, V=V, BV=BV, NC=nc, WIN=win, ZONE=zone,
                ZONE_REACH=2 * win, REC=rec, num_warps=warps, num_stages=stages)
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


def _oracle_v2() -> bool:
    """v2 on silicon: no kernel shares its op order, so the yardstick is the fp64 sequential
    recurrence E on the same bf16 inputs. Pools stepped together through random acceptance
    with vLLM's align copies emulated: S = AITER, R = v2 zone 1 (every slot, vLLM's read),
    D = v2 deferred. Checks: R's committed states as close to E as AITER's (fp32 level),
    R / D / S outputs within bf16 rounding of E, D's boundary-copy slots equal to R's to
    fp32 rounding. Then graphed us/call v1 vs v2 configs, deferred steady state."""
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as aiter_gdr)

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

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

    def run_v2(pool, qkv, ba, cu, idx, acc, seq, out, zone, config=None, v2=True):
        gdn_defer(qkv, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, idx, acc, seq, out,
                  H, K, V, zone, config, v2)

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

    for zone, n_req, steps in ((0, 16, 10), (24, 16, 14), (1664, 16, 8)):
        pool_blocks = n_req * win * 4 + 1
        pools = {"S": torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1}
        pools["R"], pools["D"] = pools["S"].clone(), pools["S"].clone()
        free = (torch.randperm(pool_blocks - 1) + 1).tolist()
        windows = [[free.pop() for _ in range(win)] for _ in range(n_req)]
        E = [pools["S"][w[0]].to(f64) for w in windows]  # committed fp64 states (acc = 1)
        c = torch.randint(0, 3 * zone + 50 if zone else 4000, (n_req,)).tolist()
        acc = [1] * n_req
        st = {"R": 0.0, "S": 0.0, "Escale": 0.0}
        ob = {"R": 0.0, "D": 0.0, "S": 0.0, "DR": 0.0}
        copies_ok, checks, moves = True, 0, 0
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
                    for k in pools}
            run_stock(pools["S"], qkv, ba, cu, idx, acc_t, outs["S"])
            run_v2(pools["R"], qkv, ba, cu, idx, acc_t, seq_t, outs["R"], 1)
            run_v2(pools["D"], qkv, ba, cu, idx, acc_t, seq_t, outs["D"], zone)
            new_acc = [int(torch.randint(1, q + 1, ()).item()) for q in qlens]
            for r in range(n_req):
                lo, hi = int(cu[r]), int(cu[r + 1])
                states, o_ref = ref_tokens(E[r], qkv[lo:hi], ba[lo:hi])
                for k in ("R", "D", "S"):
                    ob[k] = max(ob[k], bound(outs[k][lo:hi], o_ref))
                ob["DR"] = max(ob["DR"], bound(outs["D"][lo:hi], outs["R"][lo:hi].to(f64)))
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
                    copies_ok &= torch.allclose(pools["D"][sl], pools["R"][sl], rtol=1e-4,
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
        ok = (st["R"] <= max(4 * st["S"], 1e-5 * st["Escale"]) and copies_ok
              and max(ob["R"], ob["D"], ob["DR"]) <= 1.0)
        ok_all &= ok
        print(f"{MARK} v2 zone {zone}, {n_req} requests x {steps} steps: committed state max "
              f"abs err vs fp64 v2 {st['R']:.2e} / AITER {st['S']:.2e} (|state| <= "
              f"{st['Escale']:.1f}); outputs vs fp64 in bf16 bounds: v2 stock-way {ob['R']:.2f}, "
              f"deferred {ob['D']:.2f}, AITER {ob['S']:.2f}, deferred vs stock-way "
              f"{ob['DR']:.2f}; boundary copies {checks} {'ok' if copies_ok else 'DIFFER'}, "
              f"moves {moves} -> {'MATCH' if ok else 'MISMATCH'}", flush=True)

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
        t_v1 = _graph_us(lambda i: run_v2(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t, out, 0,
                                          None, False), 8)[0]
        sweep = []
        for cfg in ((16, 1, 1, 4), (16, 1, 1, 2), (16, 1, 1, 8), (8, 1, 1, 4), (32, 1, 1, 4),
                    (16, 2, 1, 4), (32, 2, 1, 4), (64, 4, 1, 4), (16, 1, 2, 4)):
            try:
                us = _graph_us(lambda i: run_v2(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t,
                                                out, 0, cfg), 8)[0]
                sweep.append(f"{cfg[0]}/{cfg[1]}w/{cfg[2]}s/nc{cfg[3]} {us:.1f}")
            except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                sweep.append(f"{cfg[0]}/{cfg[1]}w/{cfg[2]}s/nc{cfg[3]} {type(exc).__name__}")
        probe = []
        for cfg in ((16, 1, 1, 4), (16, 1, 1, 2), (32, 1, 1, 4), (16, 1, 1, 1)):
            try:
                us = _graph_us(lambda i: run_v2(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t,
                                                out, 0, cfg, "v13"), 8)[0]
                probe.append(f"{cfg[0]}/{cfg[1]}w/nc{cfg[3]} {us:.1f}")
            except Exception as exc:  # noqa: BLE001
                probe.append(f"{cfg[0]}/{cfg[1]}w/nc{cfg[3]} {type(exc).__name__}")
        print(f"{MARK} v2 c{n_req} deferred steady state graphed: v1 {t_v1:.1f} us | v2 "
              f"{' | '.join(sweep)} | v1 on the 3D tile {' | '.join(probe)}", flush=True)
    for name, fn in (("v1", _gdn_defer_kernel), ("v1-3D", _gdn_defer3_kernel),
                     ("v2", _gdn_defer2_kernel)):
        for row in _isa_stats(fn):
            print(f"{MARK} isa {name}: {row}", flush=True)
    return ok_all


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
    failed |= not _oracle_v2()
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    raise SystemExit((0 if _oracle_v2() else 1) if "--v2" in sys.argv else main())
