"""Kernel-provider tests that run CPU-only in CI: guards, gate, refusal path.

The Triton numerics comparison against the IR native reference runs only when
a CUDA+triton+vllm host is present (skipped on CI and on the Mac); the guards
and the gate contract are what must never regress blindly. Guards are pure
attribute checks, so they are exercised against a fake CUDA tensor here —
production code keeps its real is_cuda check untouched.
"""
import pytest
import torch

from suffix_hybrid.kernels import rmsnorm
from suffix_hybrid.kernels.install import install_kernels


class FakeCuda:
    """Attribute-only stand-in for a CUDA tensor: guards must never touch
    data, so shape/dtype/stride/is_cuda/device are all they may read."""

    def __init__(self, shape, dtype=torch.bfloat16, contiguous=True,
                 is_cuda=True, device="cuda:0"):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = device
        n = 1
        for s in self._shape:
            n *= s
        self._numel = n
        strides, acc = [], 1
        for s in reversed(self._shape):
            strides.insert(0, acc)
            acc *= s
        if not contiguous and len(self._shape) == 2:
            strides = [1, self._shape[0]]  # column-major
        self._strides = tuple(strides)
        self._contiguous = contiguous

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def numel(self):
        return self._numel

    def stride(self, i=-1):
        return self._strides[i]

    def is_contiguous(self):
        return self._contiguous


def _fake(rows=4, hidden=256, **kw):
    return FakeCuda((rows, hidden), **kw)


# ---------------------------------------------------------------- guards
def test_guard_accepts_canonical_call():
    assert rmsnorm._supports_rms_norm(_fake(), _fake(1, 256), 1e-5)
    assert rmsnorm._supports_rms_norm(_fake(), None, 1e-5)


@pytest.mark.parametrize("hidden", [8, 64000])
def test_guard_hidden_bounds(hidden):
    assert not rmsnorm._supports_rms_norm(_fake(hidden=hidden), None, 1e-5)


def test_guard_variance_size_declined():
    assert not rmsnorm._supports_rms_norm(_fake(), None, 1e-5, variance_size=128)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_guard_supported_dtypes(dtype):
    assert rmsnorm._supports_rms_norm(_fake(dtype=dtype), None, 1e-5)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int8, torch.uint8])
def test_guard_rejects_other_dtypes(dtype):
    assert not rmsnorm._supports_rms_norm(_fake(dtype=dtype), None, 1e-5)


def test_guard_dtype_mismatch_weight_declined():
    w = _fake(1, 256, dtype=torch.float32)
    assert not rmsnorm._supports_rms_norm(_fake(dtype=torch.bfloat16), w, 1e-5)


def test_guard_device_mismatch_weight_declined():
    w = _fake(1, 256, device="cuda:1")
    assert not rmsnorm._supports_rms_norm(_fake(device="cuda:0"), w, 1e-5)


def test_guard_non_contiguous_declined():
    # The fused variant is in-place: a strided view must decline, never reach
    # the kernel where view() would raise (a mid-dispatch crash, not a
    # fallback). Declining at the guard keeps the degradation clean.
    assert not rmsnorm._supports_rms_norm(_fake(contiguous=False), None, 1e-5)


def test_guard_cpu_declined():
    assert not rmsnorm._supports_rms_norm(_fake(is_cuda=False), None, 1e-5)


def test_guard_fused_pairwise():
    x = _fake()
    assert rmsnorm._supports_fused_add(x, _fake(), None, 1e-5)
    assert not rmsnorm._supports_fused_add(x, _fake(contiguous=False), None, 1e-5)
    assert not rmsnorm._supports_fused_add(x, _fake(rows=2), None, 1e-5)
    assert not rmsnorm._supports_fused_add(x, _fake(dtype=torch.float32), None, 1e-5)


# ---------------------------------------------------------------- gate
def test_gate_off_is_silent(monkeypatch):
    monkeypatch.delenv("SUFFIX_KERNELS", raising=False)
    assert install_kernels() is None


def test_gate_on_refuses_loudly_without_host(monkeypatch, capsys):
    # CI host: no triton/vllm (or no CUDA). install_kernels must RETURN a
    # summary with errors, never raise — the pool keeps serving on vllm_c.
    monkeypatch.setenv("SUFFIX_KERNELS", "1")
    monkeypatch.delenv("SUFFIX_KERNELS_OPS", raising=False)
    summary = install_kernels()
    assert summary is not None and summary["gated"]
    assert "rmsnorm" in summary["ops"] or "rmsnorm" in summary["errors"]
    if "rmsnorm" in summary["errors"]:
        assert "unavailable" in summary["errors"]["rmsnorm"]
    captured = capsys.readouterr()
    assert "suffix_hybrid kernels:" in captured.err


def test_gate_unknown_op_group_is_error_not_crash(monkeypatch):
    monkeypatch.setenv("SUFFIX_KERNELS", "1")
    monkeypatch.setenv("SUFFIX_KERNELS_OPS", "bogus")
    summary = install_kernels()
    assert summary is not None and "bogus" in summary["errors"]


# ---------------------------------------------------------------- numerics
_HAS_HOST = False
try:
    import triton  # noqa: F401
    import vllm  # noqa: F401
    _HAS_HOST = torch.cuda.is_available()
except Exception:
    pass


@pytest.mark.skipif(not _HAS_HOST, reason="needs CUDA + triton + vllm host")
@pytest.mark.parametrize("hidden", [128, 1000, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_matches_native(hidden, dtype):
    from vllm import ir

    summary = rmsnorm.register_impls()
    assert "rms_norm" in summary["registered"] or \
        summary.get("registered") == "already"
    impl = ir.ops.rms_norm.impls[rmsnorm.PROVIDER]
    native = ir.ops.rms_norm.impls["native"]
    for rows in (1, 16):
        x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
        w = torch.randn(hidden, dtype=dtype, device="cuda")
        ref = native.impl_fn(x.clone(), w, 1e-5)
        got = impl.impl_fn(x.clone(), w, 1e-5)
        # Triton reduction order differs from eager's; fp32 math keeps it tight.
        assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _HAS_HOST, reason="needs CUDA + triton + vllm host")
def test_fused_add_matches_native():
    from vllm import ir

    rmsnorm.register_impls()
    impl = ir.ops.fused_add_rms_norm.impls[rmsnorm.PROVIDER]
    native = ir.ops.fused_add_rms_norm.impls["native"]
    x = torch.randn(16, 2048, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x)
    w = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
    ref_out, ref_res = native.impl_fn(x.clone(), r.clone(), w, 1e-5)
    got_out, got_res = impl.impl_fn(x.clone(), r.clone(), w, 1e-5)
    assert torch.allclose(got_res.float(), ref_res.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(got_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
