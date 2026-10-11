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
def _pf_l2(ptrs, mask):
    # L2 prefetch hint per address (never writes; masked lanes skip)
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.u32 $0, 0; }",
        "=r,l,r", [ptrs, mask.to(tl.int32)], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _fence_gpu():
    # every thread: its own partial stores (release side) / the partial reads that follow
    # (acquire side) ordered at gpu scope, besides the CTA barrier + thread 0's acq_rel atomic
    tl.inline_asm_elementwise("fence.acq_rel.gpu;\nmov.u32 $0, 0;", "=r", [],
                              dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _row_ptrs(kv_ptr, slot, BS, BSTRIDE):
    # rows are 352 B = 22 x 16 B: with kv_ptr / BSTRIDE specialized 16 B divisible (the
    # usual cache; a padded view that is not compiles a scalar-load variant), Triton proves
    # 16 B alignment and the latent / RoPE gathers become cp.async / vector loads
    sl = tl.where(slot >= 0, slot, 0).to(tl.int64)
    return kv_ptr + (sl // BS) * BSTRIDE + (sl % BS) * 352


@triton.jit
def _prefetch(kv_ptr, tbase, n0, hi, BS, BSTRIDE, TN: tl.constexpr):
    # the 352 B rows of list columns [n0, n0 + TN) < hi into L2: one hint per 128 B line
    cols = n0 + tl.arange(0, TN)
    slot = tl.load(tbase + cols, mask=cols < hi, other=-1)
    rowp = _row_ptrs(kv_ptr, slot, BS, BSTRIDE)
    off = tl.arange(0, 4)
    off = tl.where(off == 3, 351, off * 128)
    _pf_l2(rowp[:, None] + off[None, :], (slot >= 0)[:, None] & (off >= 0)[None, :])


@triton.jit
def _tile(rowp, rv, CM: tl.constexpr):
    # one read of a tile's raw bytes, every load issued before any is used (one memory
    # latency per tile): (latent chunks 0..3 [TN, 64], RoPE [TN, 64], SF of chunks 0..3
    # [TN, 8]). Each pass reads its own copy (CM differs: no CSE) so every load has one dot
    # consumer and Triton stages the raw u8 through shared memory, dequantizing after the
    # operand load (a shared read feeding both dots is dequantized first: 72 KB of f16).
    # Triton does not pipeline a gather whose addresses come from a load in the same loop:
    # the L2 prefetch (_prefetch, PFD tiles ahead) hides the DRAM part instead.
    j = tl.arange(0, 64)
    out = ()
    for c in tl.static_range(4):
        out = out + (tl.load(rowp[:, None] + c * 64 + j[None, :], mask=rv[:, None], other=0,
                             cache_modifier=CM),)
    out = out + (tl.load(rowp[:, None] + 256 + j[None, :], mask=rv[:, None], other=0,
                         cache_modifier=CM),)
    for c in tl.static_range(4):
        # byte j of chunk c: dims 2j / 2j + 1, SF block 8c + j // 8 at 320 + 8 (blk & 3) + blk >> 2
        blk = c * 8 + tl.arange(0, 8)
        out = out + (tl.load(rowp[:, None] + 320 + 8 * (blk & 3)[None, :] + (blk >> 2)[None, :],
                             mask=rv[:, None], other=0, cache_modifier=CM),)
    return out


@triton.jit
def _kchunk(b, sfb, TN: tl.constexpr, ASM: tl.constexpr):
    # latent bytes [TN, 64] + their SF bytes [TN, 8] -> (even dims, odd dims) f16 [TN, 64]
    # = value * 2^-8
    sf = tl.reshape(tl.broadcast_to(_e4m3(sfb.to(tl.int32))[:, :, None], [TN, 8, 8]), [TN, 64])
    if ASM:
        lo, hi = _e2m1x2(b)
        return lo * sf, hi * sf
    else:
        sf = (sf.to(tl.float32) * 16384.0).to(tl.float16)  # e2m1 bits carry 2^-14
        bi = b.to(tl.int32)
        return _e2m1(bi) * sf, _e2m1(bi >> 4) * sf


@triton.jit
def _qt(qb, rok, up, off, STEP: tl.constexpr, W: tl.constexpr):
    # q^T chunk [W, M] f16: dims off + STEP * i (i < W) after the per-row prescale
    d = off + STEP * tl.arange(0, W)
    q = tl.load(qb[:, None] + d[None, :], mask=rok[:, None], other=0.0).to(tl.float32)
    return tl.trans((q * up[:, None]).to(tl.float16))


@triton.jit
def _merge(out_ptr, opart_ptr, lse_ptr, tok, hs, rok, NS, o_st, o_sh, HQ: tl.constexpr,
           M: tl.constexpr, NSP: tl.constexpr):
    # the last split of (token, head tile): exp2-domain LSE merge of the NS bf16 partials
    # in split order (deterministic whoever arrives last), all -inf row -> 0
    prow = (tok * HQ + hs) * NS
    ns = tl.arange(0, NSP)
    lse = tl.load(lse_ptr + prow[:, None] + ns[None, :], mask=rok[:, None] & (ns < NS)[None, :],
                  other=float("-inf"), cache_modifier=".cg")
    mx = tl.max(lse, 1)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    den = tl.sum(tl.exp2(lse - mx[:, None]), 1)
    d = tl.arange(0, 128)
    o = ()
    for c in tl.static_range(4):
        o = o + (tl.zeros([M, 128], tl.float32),)
    for j in range(NS):
        lj = tl.load(lse_ptr + prow + j, mask=rok, other=float("-inf"), cache_modifier=".cg")
        w = tl.exp2(lj - mx)
        live = rok & (lj != float("-inf"))
        new = ()
        for c in tl.static_range(4):
            part = tl.load(opart_ptr + (prow + j)[:, None] * 512 + c * 128 + d[None, :],
                           mask=live[:, None], other=0.0, cache_modifier=".cg")
            new = new + (o[c] + w[:, None] * part.to(tl.float32),)
        o = new
    for c in tl.static_range(4):
        res = tl.where(den[:, None] > 0, o[c] / den[:, None], 0.0)
        tl.store(out_ptr + tok * o_st + hs[:, None] * o_sh + c * 128 + d[None, :],
                 res.to(tl.bfloat16), mask=rok[:, None])


@triton.jit(do_not_specialize=["T", "NS"])  # strides: fixed per model, specialized (vector loads)
def _fwd(q_ptr, kv_ptr, topk_ptr, out_ptr, opart_ptr, lse_ptr, sem_ptr,
         T, C, NS, topk_st, q_st, q_sh, o_st, o_sh, BS, BSTRIDE, qk_scale_log2,
         HQ: tl.constexpr, HB: tl.constexpr, HQT: tl.constexpr, M: tl.constexpr,
         TN: tl.constexpr, CP: tl.constexpr, NSP: tl.constexpr, PFD: tl.constexpr,
         ASM: tl.constexpr, INTERPRET: tl.constexpr = False):
    # one program = (token, HB-head tile, split s of the token's live list [0, L))
    pid = tl.program_id(0)
    s = tl.program_id(1)
    tok = (pid // HQT).to(tl.int64)
    r = tl.arange(0, M)
    h = (pid % HQT) * HB + r
    rok = (r < HB) & (h < HQ)
    hs = tl.where(rok, h, 0)
    qb = q_ptr + tok * q_st + hs * q_sh
    # the token's list first: its latency overlaps the q staging below
    tbase = topk_ptr + tok * topk_st
    oc = tl.arange(0, CP)
    lst = tl.load(tbase + oc, mask=oc < C, other=-1)
    # per-row power-of-two prescale (oxide recipe): amax -> [2^14, 2^15), inverse in rs
    qa = tl.load(qb[:, None] + tl.arange(0, 512)[None, :], mask=rok[:, None], other=0.0)
    qr = tl.load(qb[:, None] + 512 + tl.arange(0, 64)[None, :], mask=rok[:, None], other=0.0)
    amax = tl.maximum(tl.max(tl.abs(qa.to(tl.float32)), 1), tl.max(tl.abs(qr.to(tl.float32)), 1))
    e = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
    k = tl.minimum(tl.maximum(141 - e, -114), 126)
    up = ((k + 127) << 23).to(tl.float32, bitcast=True)
    rs = qk_scale_log2 * ((127 - k) << 23).to(tl.float32, bitcast=True)
    # S^T = K q^T: q^T chunks are the (loop-invariant) B operands
    qe = ()
    qo = ()
    for c in tl.static_range(4):
        qe = qe + (_qt(qb, rok, up, 128 * c, 2, 64),)
        qo = qo + (_qt(qb, rok, up, 128 * c + 1, 2, 64),)
    qp = _qt(qb, rok, up, 512, 1, 64)
    # live length L = last live column + 1; this split's whole tiles of [0, L)
    L = tl.max(tl.where(lst >= 0, oc + 1, 0), 0)
    chunk = TN * tl.cdiv(tl.cdiv(L, TN), NS)
    lo = s * chunk
    hi = tl.minimum(lo + chunk, L)
    for j in tl.static_range(PFD):
        _prefetch(kv_ptr, tbase, lo + j * TN, hi, BS, BSTRIDE, TN)
    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = ()  # (even, odd) output dims of latent bytes [64c, 64c + 64), c = 0..3
    for c in tl.static_range(8):
        acc = acc + (tl.zeros([M, 64], tl.float32),)
    jt = tl.arange(0, TN)
    for n0 in range(lo, hi, TN):
        if PFD > 0:
            _prefetch(kv_ptr, tbase, n0 + PFD * TN, hi, BS, BSTRIDE, TN)
        cols = n0 + jt
        slot = tl.load(tbase + cols, mask=cols < hi, other=-1)
        rv = slot >= 0
        rowp = _row_ptrs(kv_ptr, slot, BS, BSTRIDE)
        cur = _tile(rowp, rv, "")
        st = tl.zeros([TN, M], tl.float32)
        for c in tl.static_range(4):
            klo, khi = _kchunk(cur[c], cur[5 + c], TN, ASM)
            st = tl.dot(klo, qe[c], st)
            st = tl.dot(khi, qo[c], st)
        st = tl.dot(_e4m3(cur[4].to(tl.int32)), qp, st)
        x = tl.where(rv[:, None], st * rs[None, :], float("-inf"))
        m_new = tl.maximum(m_i, tl.max(x, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(x - m_use[None, :])
        l_i = l_i * alpha + tl.sum(p, 0)
        pt = tl.trans(p.to(tl.float16))  # [M, TN]
        new = ()
        cur = _tile(rowp, rv, ".ca")  # L1 hits
        for c in tl.static_range(4):
            vlo, vhi = _kchunk(cur[c], cur[5 + c], TN, ASM)
            new = new + (tl.dot(pt, vlo, acc[2 * c] * alpha[:, None]),
                         tl.dot(pt, vhi, acc[2 * c + 1] * alpha[:, None]))
        acc = new
        m_i = m_new
    has = l_i > 0
    inv = tl.where(has, 256.0 / l_i, 0.0)  # SCALE_FIX / l (V carries 2^-8)
    d = tl.arange(0, 128)
    if NS == 1:
        for c in tl.static_range(4):
            o = tl.reshape(tl.join(acc[2 * c] * inv[:, None], acc[2 * c + 1] * inv[:, None]),
                           [M, 128])
            tl.store(out_ptr + tok * o_st + hs[:, None] * o_sh + c * 128 + d[None, :],
                     o.to(tl.bfloat16), mask=rok[:, None])
    else:
        # empty rows: lse -inf only (the merge never reads their partial)
        prow = (tok * HQ + hs) * NS + s
        for c in tl.static_range(4):
            o = tl.reshape(tl.join(acc[2 * c] * inv[:, None], acc[2 * c + 1] * inv[:, None]),
                           [M, 128])
            tl.store(opart_ptr + prow[:, None] * 512 + c * 128 + d[None, :], o.to(tl.bfloat16),
                     mask=(rok & has)[:, None])
        tl.store(lse_ptr + prow, tl.where(has, m_i + tl.log2(l_i), float("-inf")), mask=rok)
        if not INTERPRET:
            _fence_gpu()
        tl.debug_barrier()  # every warp's partial is issued before the release
        cnt = tl.atomic_add(sem_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if cnt == NS - 1:
            if not INTERPRET:
                _fence_gpu()
            _merge(out_ptr, opart_ptr, lse_ptr, tok, hs, rok, NS, o_st, o_sh, HQ, M, NSP)
            tl.atomic_xchg(sem_ptr + pid, 0)
