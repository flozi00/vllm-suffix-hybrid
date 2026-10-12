# SPDX-License-Identifier: Apache-2.0
"""CPU proof of the nvfp4_ds_mla attention kernel's fragment plumbing
(kernels-oxide/nvfp4_ds_mla/src/main.rs), transcribed from the kernel:

* S phase over the full 576-dim query (512 e2m1 latent dims with permuted
  SF bytes + 64 raw e4m3 RoPE dims) with the kernel's Q staging permutation
  (dim 32blk+8a+4b+2e+f -> smem column 32blk+16b+8e+2a+f) — the k2
  k-permutation lesson: a straight staging scrambles every score;
* PV phase: two-token nibble interleave, SF block sf_perm(4w + gq/2) per
  row pair, and the C -> latent-dim mapping of the O stores.
Emulated with the documented m16n8k16 fragment layouts (helpers shared with
test_k2_oxide_fragments.py) against plain numpy matmuls.
"""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "k2frag", Path(__file__).with_name("test_k2_oxide_fragments.py"))
K2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K2)

rng = np.random.default_rng(7)
ROW, PE_BASE, SF_BASE = 352, 256, 320


def sf_perm(s):
    return 8 * (s & 3) + (s >> 2)


def e4m3_vals(b):
    return np.vectorize(K2.e4m3_ref)(b)


def rand_rows(n):
    """n valid 352 B rows (finite e4m3 bytes) + their dequantized f64 dims."""
    rows = rng.integers(0, 256, (n, ROW)).astype(np.int64)
    rows[:, PE_BASE:] = rng.integers(0, 0x7F, (n, ROW - PE_BASE))  # no NaN
    rows[:, PE_BASE:SF_BASE] |= rng.integers(0, 2, (n, 64)) << 7   # signs
    rows[:, SF_BASE:] = rng.integers(0x18, 0x50, (n, 32))          # scales
    nib = np.stack((rows[:, :256] & 0xF, rows[:, :256] >> 4), -1).reshape(n, 512)
    sf = np.stack([e4m3_vals(rows[:, SF_BASE + sf_perm(b)]) for b in range(32)], -1)
    lat = K2.E2M1[nib] * np.repeat(sf, 16, -1)
    rope = e4m3_vals(rows[:, PE_BASE:SF_BASE])
    return rows, lat, rope


def word(row, byte):
    return sum(int(row[byte + k]) << (8 * k) for k in range(4))


def bf16(x):
    """f64 -> nearest bf16 value (RNE via the f32 bit pattern)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def stage_q(Q, prescale=True):
    """Kernel Q staging: per-row power-of-two prescale (k = 141 - biased
    bf16 exponent of the row amax, clamped <= 126) then pair (d0, d0+1) ->
    smem column col (+1) as f16. Returns (smem, per-row fold 2^-k).
    prescale=False is the pre-hardening kernel (plain bf16 -> f16)."""
    qs = np.zeros((16, 584), dtype=np.uint16)
    fold = np.ones(16)
    with np.errstate(over="ignore"):
        for row in range(Q.shape[0]):
            bits = np.asarray(Q[row], np.float32).view(np.uint32) >> 16
            e = int((bits & 0x7FFF).max()) >> 7
            k = min(141 - e, 126) if prescale else 0
            fold[row] = 2.0 ** -k
            for d0 in range(0, 576, 2):
                o = d0 % 32
                ka, kb, ke = o // 8, (o % 8) // 4, (o % 4) // 2
                col = 32 * (d0 // 32) + 16 * kb + 8 * ke + 2 * ka
                qs[row, col] = K2.f2h(Q[row, d0] * 2.0 ** k)
                qs[row, col + 1] = K2.f2h(Q[row, d0 + 1] * 2.0 ** k)
    return qs, fold


def test_q_staging_is_a_column_bijection():
    cols = set()
    for d0 in range(0, 576, 2):
        o = d0 % 32
        col = 32 * (d0 // 32) + 16 * ((o % 8) // 4) + 8 * ((o % 4) // 2) + 2 * (o // 8)
        cols |= {col, col + 1}
    assert cols == set(range(576))


def s_phase(qs, rows):
    """Kernel S phase for one n-tile of 8 gathered rows (f64 matmuls of the
    emulated fragments): returns the raw accumulator [16, 8]."""
    arow = lambda l: (l % 8) + 8 * ((l // 8) % 2)  # noqa: E731
    acol = lambda l: 8 * (l // 16)                 # noqa: E731
    S = np.zeros((16, 8))
    for p in range(16):  # latent: B from e2m1 u32 x permuted SF
        aa = K2.ldmatrix_x4(qs, lambda l: (arow(l), 32 * p + acol(l)))
        ab = K2.ldmatrix_x4(qs, lambda l: (arow(l), 32 * p + 16 + acol(l)))
        ba, bb = [], []
        for lane in range(32):
            g, t = lane // 4, lane % 4
            sc = K2.e4m3_f16(int(rows[g, SF_BASE + sf_perm(2 * p + t // 2)]))
            f = K2.nib8(word(rows[g], 16 * p + 4 * t))
            sc2 = sc | (sc << 16)
            ba.append([K2.hmul2(f[0], sc2), K2.hmul2(f[1], sc2)])
            bb.append([K2.hmul2(f[2], sc2), K2.hmul2(f[3], sc2)])
        S += K2.a_matrix(aa) @ K2.b_matrix(ba) + K2.a_matrix(ab) @ K2.b_matrix(bb)
    e2 = lambda w: K2.e4m3_f16(w & 0xFF) | (K2.e4m3_f16((w >> 8) & 0xFF) << 16)  # noqa: E731
    for pr in range(2):  # RoPE: raw e4m3 bytes
        aa = K2.ldmatrix_x4(qs, lambda l: (arow(l), 512 + 32 * pr + acol(l)))
        ab = K2.ldmatrix_x4(qs, lambda l: (arow(l), 512 + 32 * pr + 16 + acol(l)))
        ba, bb = [], []
        for lane in range(32):
            g, t = lane // 4, lane % 4
            w0 = word(rows[g], PE_BASE + 32 * pr + 8 * t)
            w1 = word(rows[g], PE_BASE + 32 * pr + 8 * t + 4)
            ba.append([e2(w0 & 0xFFFF), e2(w0 >> 16)])
            bb.append([e2(w1 & 0xFFFF), e2(w1 >> 16)])
        S += K2.a_matrix(aa) @ K2.b_matrix(ba) + K2.a_matrix(ab) @ K2.b_matrix(bb)
    return S


def test_s_phase_computes_q_dot_k_over_latent_and_rope():
    rows, lat, rope = rand_rows(8)  # one n-tile: 8 gathered rows
    K = np.concatenate([lat, rope], -1)
    Q = bf16(rng.standard_normal((16, 576)))
    Q[8:] = 0  # mirror rows
    qs, fold = stage_q(Q)
    S = s_phase(qs, rows) * fold[:, None]  # kernel: S * qk_scale * 2^-k
    np.testing.assert_allclose(S * 256.0, Q @ K.T, rtol=1e-6, atol=1e-6)


def adversarial_q():
    """bf16 query rows the pre-hardening f16 staging cannot represent."""
    Q = np.zeros((16, 576))
    Q[0] = rng.standard_normal(576) * 2.0 ** 15        # |q| ~ 1e5 everywhere
    Q[1] = rng.standard_normal(576)                    # benign
    Q[1, 7], Q[1, 520] = 1e5, -1e5                     # mixed: 2 huge dims (latent + RoPE)
    Q[2] = rng.standard_normal(576) * 1e-6             # f16-subnormal range
    Q[3] = rng.standard_normal(576) * 1e30             # far past f16
    Q[4, 3] = 1e33                                     # one dim, near the f32-logit bound
    # Q[5..7] = 0: all-zero rows (k clamps; S must be exactly 0)
    return bf16(Q)


def max_sf_rows(n):
    """Rows at the format's magnitude ceiling: every nibble +-6, every SF
    0x7E (448), RoPE bytes +-448 -> |latent| = 2688 (6 x 448)."""
    rows = np.zeros((n, ROW), np.int64)
    rows[:, :256] = rng.choice([0x77, 0xFF, 0x7F, 0xF7], (n, 256))
    rows[:, PE_BASE:SF_BASE] = rng.choice([0x7E, 0xFE], (n, 64))
    rows[:, SF_BASE:] = 0x7E
    return rows


def test_s_phase_prescale_is_exact_where_plain_f16_staging_breaks():
    """Fails on the OLD kernel (bf16 -> f16 overflow to inf / subnormal
    loss), passes on the prescaled staging, for benign AND max-SF rows."""
    Q = adversarial_q()
    for rows in (rand_rows(8)[0], max_sf_rows(8)):
        lat = np.stack([np.concatenate(_deq(r)) for r in rows])
        with np.errstate(over="ignore", invalid="ignore"):
            ref = Q[:8] @ lat.T
            qs, fold = stage_q(Q)
            new = (s_phase(qs, rows) * fold[:, None] * 256.0)[:8]
            qs0, fold0 = stage_q(Q, prescale=False)
            old = (s_phase(qs0, rows) * fold0[:, None] * 256.0)[:8]
        # staged values are exact: f16(q * 2^k) * 2^-k == q for rows 0..4
        # (row 2's 1e-6 values stay f16 normals after the prescale)
        for r in (0, 1, 2, 3):
            got = np.array([K2.h2f(int(b)) for b in qs[r, :576]]) * fold[r]
            assert sorted(got) == sorted(Q[r]), f"row {r} staging inexact"
        assert np.isfinite(new[:5]).all()
        np.testing.assert_allclose(new[:5], ref[:5], rtol=1e-6)
        assert (new[5:] == 0).all()
        # the old staging: inf/NaN scores (rows 0, 1, 3, 4) and a
        # subnormal-truncated row 2
        assert (~np.isfinite(old[[0, 1, 3, 4]])).all(axis=1).all()
        assert np.abs(old[2] - ref[2]).max() > 1e-3 * np.abs(ref[2]).max()


def _deq(row):
    nib = np.stack((row[:256] & 0xF, row[:256] >> 4), -1).reshape(512)
    sf = np.array([K2.e4m3_ref(int(row[SF_BASE + sf_perm(b)])) for b in range(32)])
    return K2.E2M1[nib] * np.repeat(sf, 16), e4m3_vals(row[PE_BASE:SF_BASE])


def test_max_magnitude_rows_stay_inside_f16_and_f32():
    """V/PV and lse headroom at the format ceiling (SF 448 x e2m1 6)."""
    # in-kernel f16 operands carry 2^-8: |K|, |V| <= 6 * 448 / 256 = 10.5
    rows = max_sf_rows(8)
    f = [K2.h2f(K2.hmul2(K2.nib8(word(r, 0))[0], K2.e4m3_f16(0x7E) * 0x10001)) for r in rows]
    assert max(abs(x) for x in f) == 10.5
    # S accumulator: 576 products of |q| < 2^15 (prescaled) and <= 10.5
    assert 576 * 2.0 ** 15 * 10.5 < np.finfo(np.float32).max
    # P in [0, 1] (f16), O partial <= l * 10.5 with l <= 2048 rows
    assert 2048 * 10.5 * 256 < 3.39e38  # merge output (x v_scale 2^8), bf16 range
    # true logits stay finite in f32 up to |q| ~ 2e33 (bf16 max is 3.4e38):
    # 576 * 2e33 * 2688 * (256^-0.5 * log2 e) < f32 max
    assert 576 * 2e33 * 2688 * (1 / 16) * 1.4427 < np.finfo(np.float32).max


def test_pv_phase_and_output_store_mapping():
    tn = 16
    rows, lat, _ = rand_rows(tn)
    P = rng.random((16, tn)).astype(np.float16).astype(np.float64)
    ps = np.zeros((16, tn + 8), dtype=np.uint16)
    for r in range(16):
        for c in range(tn):
            ps[r, c] = K2.f2h(P[r, c])
    a = K2.ldmatrix_x4(ps, lambda l: ((l % 8) + 8 * ((l // 8) % 2), 8 * (l // 16)))
    A = K2.a_matrix(a)
    out = np.full((16, 512), np.nan)
    for w in range(8):  # warp w owns latent dims 64w..64w+63
        b0s, b1s = [], []
        for lane in range(32):
            g, t = lane // 4, lane % 4
            gblk = sf_perm(4 * w + g // 2)
            bs = []
            for pr2 in range(2):
                ta = 2 * t + 8 * pr2
                wa, wb = word(rows[ta], 32 * w + 4 * g), word(rows[ta + 1], 32 * w + 4 * g)
                sc2 = (K2.e4m3_f16(int(rows[ta, SF_BASE + gblk]))
                       | (K2.e4m3_f16(int(rows[ta + 1, SF_BASE + gblk])) << 16))
                lo = (wa & 0x0F0F0F0F) | ((wb & 0x0F0F0F0F) << 4)
                hi = ((wa >> 4) & 0x0F0F0F0F) | (wb & 0xF0F0F0F0)
                fl, fh = K2.nib8(lo), K2.nib8(hi)
                b = [0] * 8
                for k in range(4):
                    b[2 * k], b[2 * k + 1] = K2.hmul2(fl[k], sc2), K2.hmul2(fh[k], sc2)
                bs.append(b)
            b0s.append(bs[0])
            b1s.append(bs[1])
        for jn in range(8):
            C = A @ K2.b_matrix([[b0s[l][jn], b1s[l][jn]] for l in range(32)])
            for lane in range(32):
                g, t = lane // 4, lane % 4
                c = K2.c_of_lane(C, lane)
                base = 64 * w + 16 * t + jn  # kernel store: +0 (c0/c2), +8 (c1/c3)
                out[g, base], out[g, base + 8] = c[0], c[1]
                out[g + 8, base], out[g + 8, base + 8] = c[2], c[3]
    np.testing.assert_allclose(out * 256.0, P @ lat, rtol=1e-6, atol=1e-6)


# ---- multi-token decode: grid / offsets / S mask / merge ------------------
# Silicon (2026-09-25): T=1 PASS, every T>1 FAIL (row rel-L2 ~1). T=1 in the
# oracle is one FULL top-k token (no -1); T>1 adds -1 holes / tails. The S
# mask tested the B-fragment row krow = nt*8 + gq of each lane, but a lane's
# C fragment holds COLUMNS 2t4, 2t4+1 (k2 masks kp0 = nt*8 + 2t4 + e): live
# rows were dropped and zero-staged -1 rows entered with S = 0.
def _s_mask_src(fixed):
    """[16, 8] map: S element (row, col) of an n-tile -> the n-tile row
    whose liveness the kernel's lane applies to it (from K2.c_of_lane)."""
    pos = np.arange(128).reshape(16, 8)
    src = np.full((16, 8), -1)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        for i, p in enumerate(K2.c_of_lane(pos, lane)):
            e = i % 2
            src[p // 8, p % 8] = (2 * t + e) if fixed else g
    assert (src >= 0).all()
    return src


def emulate_decode(q, lat, rope, topk, ns, fixed=True, with_lse=False):
    """Index-level transcription of nvfp4_ds_mla_attn_partial + _merge with
    the host op's launch args (grid (T*HQT, NS), c_per_split = split_rows,
    topk_stride = topk_len = C, q_stride = HQ*576) over flat buffers."""
    T, HQ, _ = q.shape
    C = topk.shape[1]
    hqt = -(-HQ // 8)
    cps = 64 * -(-(-(-C // 64)) // ns)  # split_rows
    assert -(-C // cps) == ns
    qf, tk = q.reshape(-1), topk.reshape(-1)
    o_part = np.full(T * HQ * ns * 512, np.nan)
    lse = np.full(T * HQ * ns, np.nan)
    src = _s_mask_src(fixed)
    K = np.concatenate([lat, rope], -1)
    for r in range(T * hqt):
        for s in range(ns):
            ht, t = r % hqt, r // hqt
            h0 = ht * 8
            c0, c1 = s * cps, min(s * cps + cps, C)
            assert c0 < c1, "empty split launched"  # kernel would exit -inf
            heads = [h for h in range(8) if h0 + h < HQ]
            Q = np.zeros((16, 576))
            for h in heads:
                base = t * HQ * 576 + (h0 + h) * 576
                Q[h] = qf[base:base + 576]
            S_all, V_all = [], []
            for j in range(c0, c1, 64):
                for nt in range(8):
                    cc = j + nt * 8 + np.arange(8)
                    slot = np.where(cc < c1, tk[t * C + np.minimum(cc, C - 1)], -1)
                    live = slot >= 0
                    Kt = np.where(live[:, None], K[np.maximum(slot, 0)], 0.0)  # zero-staged
                    S = Q @ Kt.T
                    ok = (np.arange(16)[:, None] < 8) & live[src]
                    S_all.append(np.where(ok, S, -np.inf))
                    V_all.append(Kt[:, :512])
            S, V = np.concatenate(S_all, 1), np.concatenate(V_all, 0)
            m = S.max(1, keepdims=True)
            P = np.exp(S - np.where(np.isinf(m), 0, m))
            lsum = P.sum(1)
            Ot = (P @ V) / np.where(lsum == 0, 1, lsum)[:, None]
            for h in heads:
                row = (t * HQ + h0 + h) * ns + s
                o_part[row * 512:(row + 1) * 512] = Ot[h]
                lse[row] = -np.inf if lsum[h] == 0 else m[h, 0] + np.log(lsum[h])
    out = np.zeros(T * HQ * 512)
    for r in range(T * HQ):
        ls = lse[r * ns:(r + 1) * ns]
        assert not np.isnan(ls).any(), "lse slot never written"
        mx = ls.max()
        w = np.exp(ls - (0 if np.isinf(mx) else mx))
        acc = sum(w[s] * o_part[(r * ns + s) * 512:(r * ns + s + 1) * 512]
                  for s in range(ns) if w[s] != 0)
        out[r * 512:(r + 1) * 512] = 0 if w.sum() == 0 else acc / w.sum()
    if with_lse:
        return out.reshape(T, HQ, 512), lse.reshape(T, HQ, ns)
    return out.reshape(T, HQ, 512)


def decode_ref(q, lat, rope, topk):
    K = np.concatenate([lat, rope], -1)
    out = np.zeros(q.shape[:2] + (512,))
    for t in range(q.shape[0]):
        idx = topk[t][topk[t] >= 0]
        if idx.size:
            S = q[t] @ K[idx].T
            P = np.exp(S - S.max(1, keepdims=True))
            out[t] = (P / P.sum(1, keepdims=True)) @ lat[idx]
    return out


def oracle_topk(T, C, nslots):
    """oracle.make_topk's shape: token 0 full, token 1 all -1, others a
    random count, odd tokens with the -1s scattered as holes."""
    tk = np.full((T, C), -1)
    for i in range(T):
        n = C if i == 0 else (0 if i == 1 else int(rng.integers(1, C + 1)))
        tk[i, :n] = rng.permutation(nslots)[:n]
        if i % 2:
            tk[i] = tk[i, rng.permutation(C)]
    return tk


def test_multi_token_decode_grid_offsets_mask_and_merge():
    nslots, C = 1024, 512
    lat, rope = rng.standard_normal((nslots, 512)), rng.standard_normal((nslots, 64))
    for T, HQ, ns in [(6, 8, 8), (6, 8, 3), (3, 12, 4), (6, 64, 2), (2, 8, 1)]:
        q = rng.standard_normal((T, HQ, 576)) * 0.05
        tk = oracle_topk(T, C, nslots)
        ref = decode_ref(q, lat, rope, tk)
        np.testing.assert_allclose(emulate_decode(q, lat, rope, tk, ns), ref,
                                   rtol=1e-9, atol=1e-12)
        assert (emulate_decode(q, lat, rope, tk, ns)[1] == 0).all()
        # the pre-fix mask: wrong on every token with -1 slots
        old = emulate_decode(q, lat, rope, tk, ns, fixed=False)
        err = np.linalg.norm(old - ref, axis=-1) / np.maximum(
            np.linalg.norm(ref, axis=-1), 1e-30)
        assert T <= 2 or err[2:].max() > 0.3, (T, HQ, ns, err.max())
    # ... and exact when no slot is -1: why T=1 (one full token) passed
    q = rng.standard_normal((1, 8, 576)) * 0.05
    tk = rng.permutation(nslots)[None, :C]
    np.testing.assert_allclose(emulate_decode(q, lat, rope, tk, 8, fixed=False),
                               decode_ref(q, lat, rope, tk), rtol=1e-9, atol=1e-12)


# ---- wave-aware plan (src/nvfp4_ds_mla.rs wave_splits mirror) -------------
SMS, WAVE_C2 = 188, 4  # RTX PRO 6000; WAVE_OVERHEAD_HALF_TILES


def cdiv(a, b):
    return -(-a // b)


def wave_splits(rows, tiles, sms=SMS, c2=WAVE_C2):
    best = (None, 1)
    for ns in range(1, min(tiles, 256) + 1):
        k = cdiv(tiles, ns)
        if cdiv(tiles, k) != ns:
            continue
        cost = cdiv(rows * ns, sms) * (2 * k + c2)
        if best[0] is None or cost < best[0]:
            best = (cost, ns)
    return best[1]


def plan_ns(T, HQ, C):
    return wave_splits(T * cdiv(HQ, 8), cdiv(C, 64))


def split_rows(C, ns):
    return 64 * cdiv(cdiv(C, 64), ns)


PLAN_CS = (2048, 1000, 2000, 65)


def emitted_plans():
    """{(HQ, C, ns)} over T 1..8192 x HQ {8,16,64} x C (2048 + tails)."""
    return {(hq, C, plan_ns(T, hq, C)) for hq in (8, 16, 64) for C in PLAN_CS
            for T in range(1, 8193)}


def test_wave_plan_pins_and_native_agreement():
    old = lambda T, C: cdiv(C, split_rows(C, max(1, min(SMS // T, cdiv(C, 64)))))  # noqa: E731
    assert [plan_ns(T, 8, 2048) for T in (1, 6, 32, 64, 96, 128, 192, 256, 8192)] \
        == [32, 16, 5, 2, 3, 4, 4, 2, 1]
    assert all(plan_ns(T, 8, 2048) == old(T, 2048) for T in range(1, 65))
    for hq in (8, 16, 64):
        assert plan_ns(8192, hq, 2048) == 1
    try:
        from suffix_hybrid import _native
        native = _native.nvfp4_ds_mla_plan
    except (ImportError, AttributeError):
        return  # Rust twin: src/nvfp4_ds_mla.rs wave_plan_* tests
    if native(192, 8, 2048, SMS)["ns"] != 4:
        return  # stale local _native (pre wave plan)
    for T in (1, 6, 12, 32, 64, 96, 97, 192, 500, 1411, 8192):
        for hq in (8, 16, 64):
            for C in PLAN_CS:
                assert native(T, hq, C, SMS)["ns"] == plan_ns(T, hq, C), (T, hq, C)


def test_every_emitted_plan_decodes_exactly():
    """Every (HQ, C, ns) the planner emits over T 1..8192: the host op's
    launch (c_per_split = split_rows(C, ns), ceil(C / c_per_split) == ns),
    no empty split, and partial + merge == reference, with the merged lse
    == the full-row logsumexp (merge weights exact) and all-masked splits /
    tokens at lse -inf (weight 0)."""
    plans = emitted_plans()
    assert {ns for hq, C, ns in plans if C == 2048} >= {1, 2, 3, 4, 5, 16, 32}
    nslots = 4096
    lat, rope = rng.standard_normal((nslots, 512)), rng.standard_normal((nslots, 64))
    K = np.concatenate([lat, rope], -1)
    for hq, C, ns in sorted(plans):
        cps = split_rows(C, ns)
        assert cdiv(C, cps) == ns and (ns - 1) * cps < C <= ns * cps
        T = 3
        q = rng.standard_normal((T, hq, 576)) * 0.05
        tk = oracle_topk(T, C, nslots)
        tk[2, cps:] = -1  # token 2: every split past the first all -1
        out, lse = emulate_decode(q, lat, rope, tk, ns, with_lse=True)
        np.testing.assert_allclose(out, decode_ref(q, lat, rope, tk), rtol=1e-9, atol=1e-12)
        assert (out[1] == 0).all() and np.isneginf(lse[1]).all()
        assert np.isneginf(lse[2, :, 1:]).all()
        for t in (0, 2):
            idx = tk[t][tk[t] >= 0]
            S = q[t] @ K[idx].T
            full = S.max(1) + np.log(np.exp(S - S.max(1, keepdims=True)).sum(1))
            mx = lse[t].max(1, keepdims=True)
            merged = mx[:, 0] + np.log(np.exp(lse[t] - mx).sum(1))
            np.testing.assert_allclose(merged, full, rtol=1e-12, atol=1e-12)


# ---- issue_tile row ownership (4 threads / row) == old chunk-major -------
TN, THREADS, ROW_PITCH, ROW_CHUNKS = 64, 256, 368, 22


def _row_offset(slot, bs, stride):
    return (slot // bs) * stride + (slot % bs) * ROW


def _issue(new, dst, cache, tk, cap_base, cap_len, c0, ln, bs, stride):
    """Transcription of issue_tile (new: row = tid/4, chunks tid%4 + 4j;
    old: c = tid + 256j, row = c/22, chunk = c%22). Returns per-(row,
    chunk) write counts and the set of cp.async (dst, src) pairs."""
    hits, cps = np.zeros((TN, ROW_CHUNKS), int), set()
    for tid in range(THREADS):
        work = ([(tid // 4, ch) for ch in range(tid % 4, ROW_CHUNKS, 4)] if new
                else [divmod(c, ROW_CHUNKS) for c in range(tid, TN * ROW_CHUNKS, THREADS)])
        for row, ch in work:
            cc = c0 + row
            slot = int(tk[cap_base + cc]) if row < ln and cc < cap_len else -1
            d = row * ROW_PITCH + ch * 16
            hits[row, ch] += 1
            if slot >= 0:
                src = _row_offset(slot, bs, stride) + ch * 16
                cps.add((d, src))
                dst[d:d + 16] = cache[src:src + 16]
            else:
                dst[d:d + 16] = 0
    return hits, cps


def test_issue_tile_row_ownership_stages_identical_bytes():
    """New 4-threads-per-row mapping vs the silicon-proven chunk-major one:
    bitwise-identical staged stage (pad bytes untouched), identical cp.async
    set, every (row, chunk) written exactly once — for every (c0, ln) tile
    of every emitted plan, C 2048 / tails 1000, 2000, 65, -1 holes and
    trailing -1s, dense and padded block strides, block_size 64 and 1."""
    tiles = {(C, j, min(TN, min(s * split_rows(C, ns) + split_rows(C, ns), C) - j))
             for _, C, ns in emitted_plans() for s in range(ns)
             for j in range(s * split_rows(C, ns), min((s + 1) * split_rows(C, ns), C), TN)}
    assert any(ln < TN for _, _, ln in tiles)  # tail tiles present
    # ln < TN with capacity left (off-plan today: split ends are TN-aligned)
    tiles |= {(C, j, ln) for C in PLAN_CS for j in (0, 64) for ln in (1, 13, 40)
              if j + ln < C}
    for bs, pad in ((64, 0), (64, 128), (1, 48)):
        nb = 300 if bs == 64 else 4000
        stride = bs * ROW + pad
        cache = rng.integers(0, 256, nb * stride, dtype=np.uint8)
        nslots = nb * bs
        for C in PLAN_CS:
            tk = oracle_topk(3, C, nslots).reshape(-1)  # T=3, token 1 all -1
            for t in (0, 1, 2):
                for (c, j, ln) in sorted(tiles):
                    if c != C:
                        continue
                    stages, res = [], []
                    for new in (True, False):
                        dst = np.full(TN * ROW_PITCH, 0xA5, np.uint8)
                        res.append(_issue(new, dst, cache, tk, t * C, C, j, ln, bs, stride))
                        stages.append(dst)
                    assert (stages[0] == stages[1]).all(), (bs, pad, C, t, j, ln)
                    assert res[0][1] == res[1][1]
                    assert (res[0][0] == 1).all() and (res[1][0] == 1).all()
                    st = stages[0].reshape(TN, ROW_PITCH)
                    assert (st[:, ROW:] == 0xA5).all()  # row pad untouched
                    for row in range(TN):  # staged == gather reference
                        cc = j + row
                        slot = tk[t * C + cc] if row < ln and cc < C else -1
                        ref = (cache[_row_offset(slot, bs, stride):][:ROW] if slot >= 0
                               else np.zeros(ROW, np.uint8))
                        assert (st[row, :ROW] == ref).all()
