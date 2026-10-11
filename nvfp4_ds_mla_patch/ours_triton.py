# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of nvfp4_ds_mla_patch.ours (imported lazily: needs triton).
Contract and design notes: ours.py."""
import triton
import triton.language as tl


@triton.jit
def _e2m1(c):
    # e2m1 nibble in bits 0..3 -> f16 holding value * 2^-14 (exact)
    return (((c & 7) << 9) | ((c & 8) << 12)).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _e4m3(b):
    # e4m3fn byte -> f16 holding value * 2^-8 (exact, subnormals too)
    return (((b & 0x7F) << 7) | ((b & 0x80) << 8)).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _e2m1x2(b):
    # e2m1 byte pairs (uint8) -> (low nibble, high nibble) as exact f16 values
    return tl.inline_asm_elementwise(
        "{ .reg .b8 b<4>; .reg .b32 r<4>;"
        " mov.b32 {b0, b1, b2, b3}, $4;"
        " cvt.rn.f16x2.e2m1x2 r0, b0; cvt.rn.f16x2.e2m1x2 r1, b1;"
        " cvt.rn.f16x2.e2m1x2 r2, b2; cvt.rn.f16x2.e2m1x2 r3, b3;"
        " prmt.b32 $0, r0, r1, 0x5410; prmt.b32 $1, r2, r3, 0x5410;"
        " prmt.b32 $2, r0, r1, 0x7632; prmt.b32 $3, r2, r3, 0x7632; }",
        "=r,=r,=r,=r,r", [b], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)


@triton.jit
def _load_q(q_ptr, row_tok, h, rok, q_st, q_sh, qk_scale_log2):
    # q rows -> f16 (even dims, odd dims, RoPE) after the per-row
    # power-of-two prescale; returns the matching log2 logit scale.
    i256 = tl.arange(0, 256)
    i64 = tl.arange(0, 64)
    base = q_ptr + tl.where(rok, row_tok, 0).to(tl.int64) * q_st + h * q_sh
    m2 = rok[:, None]
    qe = tl.load(base[:, None] + 2 * i256[None, :], mask=m2, other=0.0).to(tl.float32)
    qo = tl.load(base[:, None] + 2 * i256[None, :] + 1, mask=m2, other=0.0).to(tl.float32)
    qp = tl.load(base[:, None] + 512 + i64[None, :], mask=m2, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(qe), 1), tl.max(tl.abs(qo), 1)),
                      tl.max(tl.abs(qp), 1))
    e = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
    k = tl.minimum(tl.maximum(141 - e, -114), 126)
    up = ((k + 127) << 23).to(tl.float32, bitcast=True)
    dn = ((127 - k) << 23).to(tl.float32, bitcast=True)
    return ((qe * up[:, None]).to(tl.float16), (qo * up[:, None]).to(tl.float16),
            (qp * up[:, None]).to(tl.float16), qk_scale_log2 * dn)


@triton.jit
def _attend(q_e, q_o, q_p, rowscale, kv_ptr, topk_ptr, row_tok, rok, ref_tok,
            lo, hi, topk_st, BS, BSTRIDE, TN: tl.constexpr, ASM: tl.constexpr):
    # online softmax over list columns [lo, hi) of token ref_tok; row r
    # is live at column c iff topk[row_tok[r], c] >= 0 (== the ref slot)
    M: tl.constexpr = q_e.shape[0]
    offs = tl.arange(0, TN)
    i256 = tl.arange(0, 256)
    i64 = tl.arange(0, 64)
    j32 = tl.arange(0, 32)
    sfo = 320 + 8 * (j32 & 3) + (j32 >> 2)  # SF byte of 16-dim block j
    tok = tl.where(rok, row_tok, 0)
    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc_e = tl.zeros([M, 256], tl.float32)
    acc_o = tl.zeros([M, 256], tl.float32)
    for n0 in range(lo, hi, TN):
        cols = n0 + offs
        cin = cols < hi
        slot = tl.load(topk_ptr + ref_tok * topk_st + cols, mask=cin, other=-1)
        live = tl.load(topk_ptr + tok[:, None] * topk_st + cols[None, :],
                       mask=rok[:, None] & cin[None, :], other=-1) >= 0
        rv = slot >= 0
        sl = tl.where(rv, slot, 0).to(tl.int64)
        rowp = kv_ptr + (sl // BS) * BSTRIDE + (sl % BS) * 352
        b = tl.load(rowp[:, None] + i256[None, :], mask=rv[:, None], other=0)
        sf = _e4m3(tl.load(rowp[:, None] + sfo[None, :], mask=rv[:, None], other=0).to(tl.int32))
        pe = tl.load(rowp[:, None] + 256 + i64[None, :], mask=rv[:, None], other=0).to(tl.int32)
        k_pe = _e4m3(pe)
        # byte i holds dims 2i (low nibble) / 2i + 1 (high); its SF is block i // 8
        if ASM:
            sf = tl.reshape(tl.broadcast_to(sf[:, :, None], [TN, 32, 8]), [TN, 256])
            e_lo, e_hi = _e2m1x2(b)
            k_lo = e_lo * sf
            k_hi = e_hi * sf
        else:
            sf = (sf.to(tl.float32) * 16384.0).to(tl.float16)  # e2m1 bits carry 2^-14
            sf = tl.reshape(tl.broadcast_to(sf[:, :, None], [TN, 32, 8]), [TN, 256])
            b = b.to(tl.int32)
            k_lo = _e2m1(b) * sf
            k_hi = _e2m1(b >> 4) * sf
        s = tl.dot(q_e, tl.trans(k_lo))
        s += tl.dot(q_o, tl.trans(k_hi))
        s += tl.dot(q_p, tl.trans(k_pe))
        x = tl.where(live, s * rowscale[:, None], float("-inf"))
        m_new = tl.maximum(m_i, tl.max(x, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(x - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(tl.float16)
        acc_e = acc_e * alpha[:, None] + tl.dot(p16, k_lo)
        acc_o = acc_o * alpha[:, None] + tl.dot(p16, k_hi)
        m_i = m_new
    return m_i, l_i, acc_e, acc_o


@triton.jit
def _finish(out_ptr, opart_ptr, lse_ptr, m_i, l_i, acc_e, acc_o, row_tok, h, rok,
            s, NS, o_st, o_sh, HQ: tl.constexpr):
    M: tl.constexpr = acc_e.shape[0]
    has = l_i > 0
    inv = tl.where(has, 256.0 / l_i, 0.0)  # SCALE_FIX / l (V carries 2^-8)
    o = tl.reshape(tl.join(acc_e * inv[:, None], acc_o * inv[:, None]), [M, 512])
    o = o.to(tl.bfloat16)
    d = tl.arange(0, 512)
    tok = tl.where(rok, row_tok, 0).to(tl.int64)
    if NS == 1:
        tl.store(out_ptr + tok[:, None] * o_st + h[:, None] * o_sh + d[None, :], o,
                 mask=rok[:, None])
    else:
        # empty rows: lse -inf only (the merge never reads their partial)
        prow = (tok * HQ + h) * NS + s
        tl.store(opart_ptr + prow[:, None] * 512 + d[None, :], o, mask=(rok & has)[:, None])
        tl.store(lse_ptr + prow, tl.where(has, m_i + tl.log2(l_i), float("-inf")),
                 mask=rok)


@triton.jit
def _rows(t0, t_last, h0, HQ: tl.constexpr, HB: tl.constexpr, G: tl.constexpr,
          M: tl.constexpr):
    # q rows r = j * HB + (h - h0) over tokens t0 + j <= t_last
    r = tl.arange(0, M)
    j = r // HB
    h = h0 + r % HB
    return t0 + j, h, (j < G) & (t0 + j <= t_last) & (h < HQ)


@triton.jit
def _group(q_e, q_o, q_p, rs, row_tok, h, rok, kv_ptr, topk_ptr, out_ptr, opart_ptr,
           lse_ptr, t_last, L, s, NS, topk_st, o_st, o_sh, BS, BSTRIDE,
           HQ: tl.constexpr, TN: tl.constexpr, ASM: tl.constexpr):
    # this split's whole tiles of list columns [0, L) of token t_last
    M: tl.constexpr = q_e.shape[0]
    chunk = TN * tl.cdiv(tl.cdiv(L, TN), NS)
    lo = s * chunk
    hi = tl.minimum(lo + chunk, L)
    if lo < hi:
        m_i, l_i, acc_e, acc_o = _attend(q_e, q_o, q_p, rs, kv_ptr, topk_ptr, row_tok, rok,
                                         t_last, lo, hi, topk_st, BS, BSTRIDE, TN, ASM)
    else:  # past the live list (short contexts): no rows
        m_i = tl.full([M], float("-inf"), tl.float32)
        l_i = tl.zeros([M], tl.float32)
        acc_e = tl.zeros([M, 256], tl.float32)
        acc_o = tl.zeros([M, 256], tl.float32)
    _finish(out_ptr, opart_ptr, lse_ptr, m_i, l_i, acc_e, acc_o, row_tok, h, rok, s, NS,
            o_st, o_sh, HQ)


@triton.jit(do_not_specialize=["T", "C", "NS", "topk_st", "q_st", "q_sh", "o_st",
                               "o_sh", "BS", "BSTRIDE"])
def _fwd(q_ptr, kv_ptr, topk_ptr, out_ptr, opart_ptr, lse_ptr, stats_ptr,
         T, C, NS, topk_st, q_st, q_sh, o_st, o_sh, BS, BSTRIDE, qk_scale_log2,
         HQ: tl.constexpr, HB: tl.constexpr, HQT: tl.constexpr, G: tl.constexpr,
         M: tl.constexpr, TN: tl.constexpr, CP: tl.constexpr, STATS: tl.constexpr,
         ASM: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.program_id(1)
    t0 = (pid // HQT) * G
    h0 = (pid % HQT) * HB
    t_last = tl.minimum(t0 + G, T) - 1
    # union-row q first: its loads overlap the list loads below
    row_tok, h, rok = _rows(t0, t_last, h0, HQ, HB, G, M)
    q_e, q_o, q_p, rs = _load_q(q_ptr, row_tok, h, rok, q_st, q_sh, qk_scale_log2)
    offs_c = tl.arange(0, CP)
    cin = offs_c < C
    ref = tl.load(topk_ptr + t_last * topk_st + offs_c, mask=cin, other=-1)
    L = tl.max(tl.where(ref >= 0, offs_c + 1, 0), 0)
    bad = L * 0
    for jj in tl.static_range(G - 1):
        lst = tl.load(topk_ptr + (t0 + jj) * topk_st + offs_c,
                      mask=cin & (t0 + jj < t_last), other=-1)
        bad += tl.sum(((lst >= 0) & (lst != ref)).to(tl.int32), 0)
    if STATS:
        if s == 0:
            tl.store(stats_ptr + pid, (bad == 0).to(tl.int32))
    if G == 1:
        _group(q_e, q_o, q_p, rs, row_tok, h, rok, kv_ptr, topk_ptr, out_ptr, opart_ptr,
               lse_ptr, t_last, L, s, NS, topk_st, o_st, o_sh, BS, BSTRIDE, HQ, TN, ASM)
    else:
        if bad == 0:
            _group(q_e, q_o, q_p, rs, row_tok, h, rok, kv_ptr, topk_ptr, out_ptr, opart_ptr,
                   lse_ptr, t_last, L, s, NS, topk_st, o_st, o_sh, BS, BSTRIDE, HQ, TN, ASM)
        else:
            # per token: the same tile path with one token's rows (M = 16)
            for jj in range(G):
                tj = t0 + jj
                if tj <= t_last:
                    rt1, h1, rok1 = _rows(tj, tj, h0, HQ, HB, 1, 16)
                    q1e, q1o, q1p, rs1 = _load_q(q_ptr, rt1, h1, rok1, q_st, q_sh,
                                                 qk_scale_log2)
                    lst = tl.load(topk_ptr + tj * topk_st + offs_c, mask=cin, other=-1)
                    Lj = tl.max(tl.where(lst >= 0, offs_c + 1, 0), 0)
                    _group(q1e, q1o, q1p, rs1, rt1, h1, rok1, kv_ptr, topk_ptr, out_ptr,
                           opart_ptr, lse_ptr, tj, Lj, s, NS, topk_st, o_st, o_sh, BS,
                           BSTRIDE, HQ, TN, ASM)


@triton.jit(do_not_specialize=["NS", "o_st", "o_sh"])
def _merge(out_ptr, opart_ptr, lse_ptr, NS, o_st, o_sh, HQ: tl.constexpr,
           NSP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, NSP)
    lse = tl.load(lse_ptr + row * NS + offs, mask=offs < NS, other=float("-inf"))
    mx = tl.max(lse, 0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    den = tl.sum(tl.exp2(lse - mx), 0)
    d = tl.arange(0, 512)
    acc = tl.zeros([512], tl.float32)
    for s in range(NS):
        ls = tl.load(lse_ptr + row * NS + s)
        if ls != float("-inf"):
            acc += tl.exp2(ls - mx) * tl.load(opart_ptr + (row * NS + s) * 512 + d).to(tl.float32)
    o = tl.where(den > 0, acc / den, 0.0)
    tl.store(out_ptr + (row // HQ) * o_st + (row % HQ) * o_sh + d, o.to(tl.bfloat16))

