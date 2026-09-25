# SPDX-License-Identifier: Apache-2.0
"""CPU proof of the K2 v2 kernel's register-level tricks
(kernels-oxide/k2_nvfp4_attn/src/main.rs), transcribed bit for bit:

* nib8: 8 e2m1 nibbles -> 4 f16x2 via two `prmt` LUT lookups + sign bits;
* e4m3_f16: e4m3 byte -> f16 bits of value * 2^-8 (exact, subnormals too);
* V two-token nibble interleave;
* the S = Q K^T fragment plumbing: head-dim permutation `perm_k`, Q via
  ldmatrix.x4 from the permuted smem rows, K B-fragments from one u32;
* the O += P V fragment plumbing: V B-fragments for 8 n-tiles from one u32
  per token, and the C -> head-dim mapping of the 2 x 16 B stores;
all checked against plain numpy matmuls with the documented m16n8k16
fragment layouts. A layout bug here would otherwise only show on silicon.
"""
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
M32 = 0xFFFFFFFF


def prmt(a, b, sel):
    """PTX prmt.b32 default mode: byte i of d = byte sel[i] of {b:a}."""
    src = [(a >> (8 * k)) & 0xFF for k in range(4)] + [(b >> (8 * k)) & 0xFF for k in range(4)]
    out = 0
    for i in range(4):
        nib = (sel >> (4 * i)) & 0xF
        byte = src[nib & 7]
        if nib & 8:  # sign-replicate mode (never used by the kernel)
            byte = 0xFF if byte & 0x80 else 0
        out |= byte << (8 * i)
    return out


def nib8(w):
    mags = w & 0x77777777
    lb = prmt(0x3E3C3800, 0x46444240, mags)
    hb = prmt(0x3E3C3800, 0x46444240, mags >> 16)
    s = w & 0x88888888
    return [
        prmt(lb, 0, 0x1404) | ((s << 12) & 0x8000) | ((s << 24) & 0x80000000),
        prmt(lb, 0, 0x3424) | ((s << 4) & 0x8000) | ((s << 16) & 0x80000000),
        prmt(hb, 0, 0x1404) | ((s >> 4) & 0x8000) | ((s << 8) & 0x80000000),
        prmt(hb, 0, 0x3424) | ((s >> 12) & 0x8000) | (s & 0x80000000),
    ]


def e4m3_f16(b):
    return ((b & 0x7F) << 7) | ((b & 0x80) << 8)


def h2f(bits):
    return float(np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0])


def pair(word):
    return h2f(word), h2f(word >> 16)


def f2h(x):
    return int(np.array([x], dtype=np.float16).view(np.uint16)[0])


def pack(lo, hi):
    return f2h(lo) | (f2h(hi) << 16)


def hmul2(a, b):
    return pack(h2f(a) * h2f(b), h2f(a >> 16) * h2f(b >> 16))


def perm_k(kl):
    s, c = kl // 16, kl % 16
    return (s // 2) * 32 + ((c % 8) // 2) * 8 + (s % 2) * 4 + (c // 8) * 2 + c % 2


def e4m3_ref(b):
    s = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 7
    return s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


# ---- m16n8k16 fragment layouts (cuda-device wmma.rs doc) ---------------------
def a_matrix(frags):  # frags[lane] = 4 u32 (A regs)
    A = np.zeros((16, 16))
    for lane in range(32):
        g, t = lane // 4, lane % 4
        for j in range(8):
            row = g if j in (0, 1, 4, 5) else g + 8
            col = 2 * t + (j & 1) + (8 if j >= 4 else 0)
            A[row, col] = pair(frags[lane][j // 2])[j % 2]
    return A


def b_matrix(frags):  # frags[lane] = 2 u32 (B regs)
    B = np.zeros((16, 8))
    for lane in range(32):
        g, t = lane // 4, lane % 4
        for j in range(4):
            B[2 * t + (j & 1) + (8 if j >= 2 else 0), g] = pair(frags[lane][j // 2])[j % 2]
    return B


def c_of_lane(C, lane):
    g, t = lane // 4, lane % 4
    return [C[g, 2 * t], C[g, 2 * t + 1], C[g + 8, 2 * t], C[g + 8, 2 * t + 1]]


def ldmatrix_x4(smem, base_of_lane):
    """smem: 2-D f16-bits array; base_of_lane(l) -> (row, col) of lane l's
    8-element row address. Returns per-lane 4 u32 (matrix i -> reg i)."""
    rows = [base_of_lane(l) for l in range(32)]
    out = []
    for lane in range(32):
        regs = []
        for i in range(4):
            r, c = rows[8 * i + lane // 4]
            c += 2 * (lane % 4)
            regs.append(int(smem[r, c]) | (int(smem[r, c + 1]) << 16))
        out.append(regs)
    return out


rng = np.random.default_rng(0)


def test_nib8_decodes_every_nibble_in_order():
    for w in list(rng.integers(0, 2 ** 32, 2000, dtype=np.uint64)) + [0, M32, 0x88888888]:
        w = int(w)
        got = [x for word in nib8(w) for x in pair(word)]
        want = [E2M1[(w >> (4 * k)) & 0xF] for k in range(8)]
        assert got == want, hex(w)


def test_e4m3_f16_is_exact_times_2_pow_minus_8():
    for b in range(256):
        if (b & 0x7F) == 0x7F:
            continue  # NaN codes (masked in-kernel)
        assert h2f(e4m3_f16(b)) * 256.0 == e4m3_ref(b), b


def test_v_interleave_pairs_two_tokens_per_dim():
    for _ in range(500):
        wa, wb = (int(x) for x in rng.integers(0, 2 ** 32, 2, dtype=np.uint64))
        lo = (wa & 0x0F0F0F0F) | ((wb & 0x0F0F0F0F) << 4)
        hi = ((wa >> 4) & 0x0F0F0F0F) | (wb & 0xF0F0F0F0)
        fl, fh = nib8(lo), nib8(hi)
        for dim in range(8):
            word = fl[dim // 2] if dim % 2 == 0 else fh[dim // 2]
            assert pair(word) == (E2M1[(wa >> 4 * dim) & 0xF], E2M1[(wb >> 4 * dim) & 0xF])


def rand_e2m1_bytes(n):
    return rng.integers(0, 256, n, dtype=np.uint64).astype(np.int64)


def test_s_fragments_compute_q_kT_with_the_head_dim_permutation():
    d = 64  # two k-step pairs
    Q = rng.standard_normal((16, d)).astype(np.float16).astype(np.float64)
    kbytes = rand_e2m1_bytes(8 * d // 2).reshape(8, d // 2)   # 8 tokens
    ksf = rng.integers(0x18, 0x50, (8, d // 16))              # e4m3 scales
    K = np.stack((E2M1[kbytes & 0xF], E2M1[kbytes >> 4]), -1).reshape(8, d)
    K = K * np.repeat(np.vectorize(e4m3_ref)(ksf), 16, -1)
    # smem Q rows in LOGICAL k order: qs[row][kl] = Q[row][perm_k(kl)]
    qs = np.zeros((16, d + 8), dtype=np.uint16)
    for row in range(16):
        for kl in range(d):
            qs[row, kl] = f2h(Q[row, perm_k(kl)])
    arow = lambda l: (l % 8) + 8 * ((l // 8) % 2)  # noqa: E731
    acol = lambda l: 8 * (l // 16)                 # noqa: E731
    S = np.zeros((16, 8))
    for p in range(d // 32):
        aa = ldmatrix_x4(qs, lambda l: (arow(l), 32 * p + acol(l)))
        ab = ldmatrix_x4(qs, lambda l: (arow(l), 32 * p + 16 + acol(l)))
        ba, bb = [], []
        for lane in range(32):
            g, t = lane // 4, lane % 4
            w = 0
            for k in range(4):  # u32 at bytes p*16 + 4t of token g
                w |= int(kbytes[g, p * 16 + 4 * t + k]) << (8 * k)
            sc = e4m3_f16(int(ksf[g, 2 * p + t // 2]))
            sc2 = sc | (sc << 16)
            f = nib8(w)
            ba.append([hmul2(f[0], sc2), hmul2(f[1], sc2)])
            bb.append([hmul2(f[2], sc2), hmul2(f[3], sc2)])
        S += a_matrix(aa) @ b_matrix(ba) + a_matrix(ab) @ b_matrix(bb)
    np.testing.assert_allclose(S * 256.0, Q @ K.T, rtol=1e-6, atol=1e-6)


def test_pv_fragments_and_output_dim_mapping():
    dh, cj, tn = 64, 1, 16                     # 128-dim head, column group 1
    P = rng.random((16, tn)).astype(np.float16).astype(np.float64)
    vbytes = rand_e2m1_bytes(tn * dh).reshape(tn, dh)
    V = np.stack((E2M1[vbytes & 0xF], E2M1[vbytes >> 4]), -1).reshape(tn, 2 * dh)
    ps = np.zeros((16, tn + 8), dtype=np.uint16)
    for r in range(16):
        for c in range(tn):
            ps[r, c] = f2h(P[r, c])
    a = ldmatrix_x4(ps, lambda l: ((l % 8) + 8 * ((l // 8) % 2), 8 * (l // 16)))
    one = f2h(1.0) | (f2h(1.0) << 16)
    b0s, b1s = [], []
    for lane in range(32):
        g, t = lane // 4, lane % 4
        bs = []
        for pr in range(2):
            ta = 2 * t + 8 * pr
            word = lambda tok: sum(int(vbytes[tok, cj * 32 + 4 * g + k]) << 8 * k  # noqa: E731
                                   for k in range(4))
            wa, wb = word(ta), word(ta + 1)
            lo = (wa & 0x0F0F0F0F) | ((wb & 0x0F0F0F0F) << 4)
            hi = ((wa >> 4) & 0x0F0F0F0F) | (wb & 0xF0F0F0F0)
            fl, fh = nib8(lo), nib8(hi)
            b = [0] * 8
            for k in range(4):
                b[2 * k], b[2 * k + 1] = hmul2(fl[k], one), hmul2(fh[k], one)
            bs.append(b)
        b0s.append(bs[0])
        b1s.append(bs[1])
    A = a_matrix(a)
    out = np.full((16, 2 * dh), np.nan)
    for jn in range(8):
        C = A @ b_matrix([[b0s[l][jn], b1s[l][jn]] for l in range(32)])
        for lane in range(32):
            g, t = lane // 4, lane % 4
            c = c_of_lane(C, lane)
            # kernel store: row g/g+8, dims 64cj + 16t + jn (c0/c2), +8 (c1/c3)
            out[g, 64 * cj + 16 * t + jn] = c[0]
            out[g, 64 * cj + 16 * t + 8 + jn] = c[1]
            out[g + 8, 64 * cj + 16 * t + jn] = c[2]
            out[g + 8, 64 * cj + 16 * t + 8 + jn] = c[3]
    want = P @ V
    cols = slice(64 * cj, 64 * cj + 64)
    np.testing.assert_allclose(out[:, cols], want[:, cols], rtol=1e-9, atol=1e-9)
