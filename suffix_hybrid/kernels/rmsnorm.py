# SPDX-License-Identifier: Apache-2.0
"""Triton RMSNorm kernels registered as a vLLM IR op provider.

Dispatch is priority-driven (see kernels/__init__.py): importing this module
and even calling register() changes nothing until --kernel-config names
"suffix_kernels" in an op priority list. Every kernel pairs with a strict
supports_args guard so an unsupported call degrades to the next provider
(vllm_c / native) instead of producing wrong numbers.

Numerics replicate the vllm.ir native op order in float32 (fp32 add, fp32
variance, rsqrt, weight-mul in weight dtype, cast back to the input dtype) —
the only divergence from `native` is the block-reduction summation order, the
same class of delta the vllm_c CUDA kernel already carries. Decode batches on
this fleet are small, so launch shape matters as much as bandwidth: one
program per row, the hidden axis in a single blocked load (hidden sizes to
32768 fit registers comfortably; beyond that we decline and let vllm_c serve).

triton/vllm are imported ONLY inside register_impls(): the module must stay
importable in CI (CPU torch, no triton, no vllm) for the pure-Python guards
and the refusal path to be testable.
"""
from __future__ import annotations

import torch
from torch import Tensor

PROVIDER = "suffix_kernels"

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MIN_HIDDEN = 16
_MAX_HIDDEN = 32768


def _supports_rms_norm(x, weight, epsilon, variance_size=None) -> bool:
    """Guard for rms_norm. Mirrors vllm_c's guard (no variance_size override,
    weight dtype matches input) and adds what our kernel assumes: CUDA fp-ish
    dtype, row-major contiguity on the reduced axis, bounded hidden width.
    Declining costs one dispatch hop to vllm_c, never correctness."""
    if variance_size is not None:
        return False
    if x.dtype not in _DTYPES:
        return False
    if weight is not None and (
        weight.dtype != x.dtype
        or weight.device != x.device
        or weight.stride(-1) != 1
    ):
        return False
    hidden = x.shape[-1] if x.dim() else 0
    if not (_MIN_HIDDEN <= hidden <= _MAX_HIDDEN):
        return False
    if x.numel() == 0 or x.stride(-1) != 1:
        return False
    # Both impls .view() the tensor to (rows, hidden): require full
    # contiguity so a strided view can never be silently COPIED by reshape
    # and leave the in-place fused variant mutating the copy.
    if not x.is_contiguous():
        return False
    return bool(x.is_cuda)


def _supports_fused_add(x, x_residual, weight, epsilon, variance_size=None) -> bool:
    """Guard for fused_add_rms_norm: both operands validated pairwise."""
    if not _supports_rms_norm(x, weight, epsilon, variance_size):
        return False
    if x_residual.shape != x.shape or x_residual.dtype != x.dtype:
        return False
    if x_residual.stride(-1) != 1 or not x_residual.is_contiguous():
        return False
    return True


def register_impls():
    """Import vllm.ir + triton and register both providers. Returns a dict
    summary; raises RuntimeError with a precise cause when the host lacks any
    piece (the gated installer turns that into a logged refusal).

    Idempotent: re-registering the same provider name overwrites in vLLM's
    IrOp.impls — safe across worker restarts in one process."""
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - host without triton
        raise RuntimeError(f"triton unavailable: {exc}") from exc
    try:
        from vllm import ir
    except Exception as exc:  # pragma: no cover - host without vllm
        raise RuntimeError(f"vllm unavailable: {exc}") from exc

    if PROVIDER in ir.ops.rms_norm.impls:
        return {"provider": PROVIDER, "registered": "already"}

    def _warps(hidden):
        if hidden <= 1024:
            return 4
        if hidden <= 4096:
            return 8
        if hidden <= 16384:
            return 16
        return 32

    # ---- fused add + rms norm, in place on x and x_residual --------------
    @triton.jit
    def _fused_add_rms_norm_kernel(
        x_ptr, res_ptr, w_ptr,
        stride_x, stride_r, stride_w,
        hidden: tl.constexpr,
        eps,
        HAS_W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                    other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * stride_r + offs, mask=mask,
                    other=0.0).to(tl.float32)
        s = x + r
        dtype = x_ptr.dtype.element_ty
        tl.store(res_ptr + row * stride_r + offs, s.to(dtype), mask=mask)
        var = tl.sum(s * s, axis=0) / hidden
        normed = s * tl.rsqrt(var + eps)
        # Mirror native exactly: x.to(weight.dtype) * weight, i.e. round the
        # normed value to the STORE dtype BEFORE the weight multiply (torch
        # bf16 elementwise mul computes in fp32 and rounds once — the .to
        # above is the extra rounding native performs via x.to(weight.dtype)).
        out = normed.to(dtype)
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            out = (out.to(tl.float32) * w.to(tl.float32)).to(dtype)
        tl.store(x_ptr + row * stride_x + offs, out, mask=mask)

    # ---- functional rms_norm into a fresh output --------------------------
    @triton.jit
    def _rms_norm_kernel(
        x_ptr, out_ptr, w_ptr,
        stride_x, stride_o, stride_w,
        hidden: tl.constexpr,
        eps,
        HAS_W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                    other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / hidden
        normed = x * tl.rsqrt(var + eps)
        dtype = x_ptr.dtype.element_ty
        out = normed.to(dtype)
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            out = (out.to(tl.float32) * w.to(tl.float32)).to(dtype)
        tl.store(out_ptr + row * stride_o + offs, out, mask=mask)

    @ir.ops.rms_norm.register_impl(
        PROVIDER, supports_args=_supports_rms_norm, supported=True
    )
    def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float,
                 variance_size: int | None = None) -> Tensor:
        assert variance_size is None
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x2 = x.view(rows, hidden)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(hidden)
        _rms_norm_kernel[(rows,)](  # type: ignore[index]
            x2, out, x2 if weight is None else weight.view(hidden),
            x2.stride(0), out.stride(0), 0 if weight is None else 1,
            hidden, epsilon, HAS_W=weight is not None,
            BLOCK=BLOCK, num_warps=_warps(hidden),
        )
        return out.reshape(x.shape)

    @ir.ops.fused_add_rms_norm.register_impl(
        PROVIDER, supports_args=_supports_fused_add, supported=True, inplace=True
    )
    def fused_add_rms_norm(x: Tensor, x_residual: Tensor, weight: Tensor | None,
                           epsilon: float,
                           variance_size: int | None = None) -> tuple[Tensor, Tensor]:
        assert variance_size is None
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x2 = x.view(rows, hidden)
        r2 = x_residual.view(rows, hidden)
        BLOCK = triton.next_power_of_2(hidden)
        _fused_add_rms_norm_kernel[(rows,)](  # type: ignore[index]
            x2, r2, x2 if weight is None else weight.view(hidden),
            x2.stride(0), r2.stride(0), 0 if weight is None else 1,
            hidden, epsilon, HAS_W=weight is not None,
            BLOCK=BLOCK, num_warps=_warps(hidden),
        )
        return x, x_residual

    return {"provider": PROVIDER,
            "registered": ["rms_norm", "fused_add_rms_norm"]}
