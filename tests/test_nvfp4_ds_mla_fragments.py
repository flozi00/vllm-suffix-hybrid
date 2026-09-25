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
