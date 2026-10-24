# SPDX-License-Identifier: Apache-2.0
"""The TEST-LOCAL NVFP4 oracle of the native-FP4-MMA family (lane I), independent of the code under
test, plus shared helpers for tests/test_fp4mma_*.py.

Helion's registered CPU ref for `hl.dot_scaled` is E8M0-only (`2 ** (scale - 127)`) and does not unpack
packed e2m1 (helion/language/matmul_ops.py `_decorators.ref(dot_scaled)`), and its FP4 decode primitives
have no ref at all, so HELION_INTERPRET runs of an NVFP4 body need a ref that knows e2m1 + e4m3. This file
implements it a second time, in numpy float64 with explicit bit tricks, and tests the shipped
`eager_ref` (nvfp4_gemm_fp4mma.dot_scaled_ref / load_fp4x16_ref) against it, so the shipped ref cannot
drift into agreeing with a wrong body.

Everything runs with HELION_INTERPRET=1 (set before any helion.kernel is built) on CPU.
"""
import os

os.environ["HELION_INTERPRET"] = "1"

import contextlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3_f64(bits: np.ndarray) -> np.ndarray:
    """Independent e4m3fn decode of raw bytes (sign bit 7, exp 6..3 bias 7, mantissa 2..0; 0x7F/0xFF = NaN)."""
    b = bits.astype(np.int64)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 15
    man = b & 7
    val = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1.0 + man / 8.0) * 2.0 ** (exp.astype(np.float64) - 7))
    val = np.where((exp == 15) & (man == 7), np.nan, val)
    return sign * val


def unpack_np(packed: np.ndarray) -> np.ndarray:
    """uint8 [R, C/2] -> f64 [R, C]; element 2i = LOW nibble."""
    lo, hi = packed & 15, packed >> 4
    code = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    return np.where(code & 8, -1.0, 1.0) * E2M1[code & 7]


def ref_dot_scaled(mat1, s1, f1, mat2, s2, f2, acc=None, out_dtype=None):
    """The oracle: mat1 [M, K/2] u8 + s1 [M, K/16] e4m3; mat2 [K/2, N] u8 (packed along K) + s2 [N, K/16]."""
    assert f1 == f2 == "e2m1" and s1.dtype == s2.dtype == torch.float8_e4m3fn
    a = unpack_np(mat1.numpy()) * np.repeat(e4m3_f64(s1.view(torch.uint8).numpy()), 16, axis=1)
    b = unpack_np(np.ascontiguousarray(mat2.numpy().T)) * np.repeat(e4m3_f64(s2.view(torch.uint8).numpy()), 16, axis=1)
    res = torch.from_numpy((a @ b.T).astype(np.float32))
    return res if acc is None else acc + res


def ref_load_fp4x16(storage, group_offsets, extra_mask=None):
    flat = storage.reshape(-1).numpy()
    go = group_offsets.numpy().astype(np.int64)
    byt = flat[go[..., None] * 8 + np.arange(8)]
    if extra_mask is not None:
        byt = np.where(extra_mask.numpy()[..., None], byt, 0).astype(np.uint8)
    vals = unpack_np(byt.reshape(-1, 8)).reshape(*go.shape, 16)
    return tuple(torch.from_numpy(vals[..., i].astype(np.float16)) for i in range(16))


@pytest.fixture
def cpu_mm(monkeypatch):
    """CPU aten::mm has no out_dtype (hl.dot's eager reference uses it)."""
    mm = torch.mm
    monkeypatch.setattr(torch, "mm", lambda a, b, *, out_dtype=None, **kw: (
        mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)))


@contextlib.contextmanager
def local_ref():
    """Patch the TEST-LOCAL oracle as the eager ref of both ops (restores whatever was there)."""
    import helion.language as hl
    ops = (hl.dot_scaled, hl.load_float4_e2m1fn_x16_to_float16)
    old = [op._ref_fn for op in ops]
    ops[0]._ref_fn, ops[1]._ref_fn = ref_dot_scaled, ref_load_fp4x16
    try:
        yield
    finally:
        for op, fn in zip(ops, old):
            op._ref_fn = fn


@pytest.fixture
def nvfp4_ref():
    pytest.importorskip("helion")
    with local_ref():
        yield


def adversarial_scales(rows: int, nkb: int, seed: int = 0) -> torch.Tensor:
    """uint8 e4m3 scale bytes [rows, nkb] mixing every hostile pattern a checkpoint or a quantizer can
    produce: zero blocks, e4m3 SUBNORMALS (0x01..0x07), the smallest normal, the saturated 448 (0x7E),
    1.0, and random mid values. Never NaN (0x7F) or negative (the contract; validate_scales refuses them)."""
    rng = np.random.default_rng(seed)
    pool = np.array([0x00, 0x01, 0x03, 0x07, 0x08, 0x38, 0x7E, 0x7D, 0x30, 0x44], dtype=np.uint8)
    pick = rng.integers(0, len(pool), size=(rows, nkb))
    out = pool[pick]
    mid = rng.random((rows, nkb)) < 0.5
    out = np.where(mid, rng.integers(0x20, 0x70, size=(rows, nkb)).astype(np.uint8), out)
    return torch.from_numpy(out.astype(np.uint8))


# ---------------------------------------------------------------------------
# the shipped eager refs agree with the independent oracle
# ---------------------------------------------------------------------------
def _rand_packed(rows, cols_el, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, cols_el // 2), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("m,n,k", [(1, 8, 64), (3, 16, 128), (16, 32, 192), (17, 40, 640)])
def test_shipped_dot_scaled_ref_matches_the_independent_oracle(m, n, k):
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    a, b = _rand_packed(m, k, 1), _rand_packed(n, k, 2)
    sa = adversarial_scales(m, k // 16, 3).view(torch.float8_e4m3fn)
    sb = adversarial_scales(n, k // 16, 4).view(torch.float8_e4m3fn)
    acc = torch.randn(m, n)
    got = fm.dot_scaled_ref(a, sa, "e2m1", b.T, sb, "e2m1", acc=acc)
    want = ref_dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc=acc)
    assert torch.equal(got, want)  # same exact f64 product, same single f32 rounding


def test_oracle_pins_the_format_by_hand():
    # one row, K = 64: nibble pairs (lo = even k, hi = odd k); the block scale applies per 16 elements
    a = torch.zeros(1, 32, dtype=torch.uint8)
    a[0, 0] = 0x32  # lo nibble 2 -> 1.0, hi nibble 3 -> 1.5  (k = 0, 1)
    a[0, 8] = 0x9F  # k = 16, 17: lo 0xF -> -6.0, hi 0x9 -> -0.5   (second 16-block)
    b = torch.zeros(32, 1, dtype=torch.uint8)  # [K/2, N]
    b[0, 0] = 0x22  # b[k=0] = 1.0, b[k=1] = 1.0
    b[8, 0] = 0x77  # b[16] = b[17] = 6.0
    sa = torch.zeros(1, 4, dtype=torch.uint8)
    sa[0, 0], sa[0, 1] = 0x38, 0x40  # 1.0, 2.0
    sb = torch.zeros(1, 4, dtype=torch.uint8)
    sb[0, 0], sb[0, 1] = 0x38, 0x30  # 1.0, 0.5
    got = ref_dot_scaled(a, sa.view(torch.float8_e4m3fn), "e2m1", b, sb.view(torch.float8_e4m3fn), "e2m1")
    # block 0: (1.0*1 + 1.5*1) * 1.0 * 1.0 = 2.5 ; block 1: (-6*2)*(6*0.5) + (-0.5*2)*(6*0.5) = -36 - 3 = -39
    assert float(got[0, 0]) == 2.5 - 39.0


def test_e4m3_decode_matches_torch():
    bits = np.arange(256, dtype=np.uint8)
    want = torch.from_numpy(bits).view(torch.float8_e4m3fn).double().numpy()
    got = e4m3_f64(bits)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])


def test_shipped_fp4x16_load_ref_matches_the_independent_oracle():
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    st = _rand_packed(5, 128, 9).reshape(-1)
    go = torch.tensor([[0, 3, 7], [10, 2, 39]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    for m in (None, mask):
        got, want = fm.load_fp4x16_ref(st, go, m), ref_load_fp4x16(st, go, m)
        assert len(got) == 16 and all(torch.equal(g, w) for g, w in zip(got, want))


def test_eager_ref_context_installs_and_restores():
    helion_lang = pytest.importorskip("helion.language")
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    before = helion_lang.dot_scaled._ref_fn
    with fm.eager_ref():
        assert helion_lang.dot_scaled._ref_fn is fm.dot_scaled_ref
    assert helion_lang.dot_scaled._ref_fn is before


def test_shipped_ref_refuses_the_mx_format_instead_of_computing_garbage():
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    a, b = _rand_packed(2, 64, 1), _rand_packed(8, 64, 2)
    e8m0 = torch.full((2, 4), 127, dtype=torch.uint8)  # an MX (ue8m0) scale
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        fm.dot_scaled_ref(a, e8m0, "e2m1", b.T, torch.full((8, 4), 127, dtype=torch.uint8), "e2m1")
    with pytest.raises(NotImplementedError):
        fm.dot_scaled_ref(a, e8m0.view(torch.float8_e4m3fn), "e4m3", b.T, e8m0.new_zeros(8, 4).view(
            torch.float8_e4m3fn), "e2m1")
