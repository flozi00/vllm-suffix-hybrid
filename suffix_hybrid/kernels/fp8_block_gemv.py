# SPDX-License-Identifier: Apache-2.0
"""Small-M W8A8 FP8 128x128-block GEMV for vLLM's CUTLASS block-FP8 linears.

On SM120 vLLM serves compressed-tensors FP8_BLOCK linears (with DeepGEMM off)
through ``CutlassFp8BlockScaledMMKernel`` -> ``cutlass_scaled_mm`` blockwise
(128-row M tile). At decode M = 1..4 that reaches ~37% of DRAM bandwidth on an
RTX PRO 6000 (LFM2.5 attention qkv/o + dense FFN: 24 GEMMs, 0.27 ms/step).

Gate ``SUFFIX_FP8_GEMV_MAX_M`` (default 0 = off). ``register()`` (vllm.general_plugins
entry ``suffix_fp8_gemv``) swaps ``CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm``
for the custom op ``torch.ops.suffix_fp8.block_mm`` (opaque to torch.compile, so the
M branch is a runtime branch inside the op, also under CUDA graphs where M is the
capture size): M <= max -> Triton GEMV (grid N/BN x M, one row of A per CTA, per
128-K block acc += (dot(a, w) * s_a) * s_w in fp32 = the blockwise contract),
else the stock CUTLASS call. The input quantization (vLLM QuantFP8) is unchanged.

The first eager call per (N, K) with M <= max compares ours with CUTLASS on the
real operands (rel <= 1e-2 and max|d| <= 2 bf16 ulps of max|ref|) and raises on a
miss; CUDA-graph capture calls never run the check.
"""
import os
import sys

import torch  # real annotations: torch.library infer_schema reads them at registration

GATE = "SUFFIX_FP8_GEMV_MAX_M"
MARKER = "[suffix fp8-gemv]"
_state = {"armed": False, "checked": set(), "launch_logged": False}
_kernel = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def max_m() -> int:
    v = int(os.environ.get(GATE, "0") or 0)
    if not 0 <= v <= 16:
        raise ValueError(f"{GATE}={v} outside [0, 16]")
    return v


def _get():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv(a_ptr, s_am, as_ptr, s_as0, s_as1, b_ptr, bs_ptr, s_bs0, out_ptr, s_om,
              N, K: tl.constexpr, BN: tl.constexpr):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < N
        rk0 = tl.arange(0, 128)
        s_row = (pid_n * BN) // 128
        acc = tl.zeros([BN], dtype=tl.float32)
        for kb in range(K // 128):
            rk = kb * 128 + rk0
            a = tl.load(a_ptr + m * s_am + rk).to(tl.float32)
            w = tl.load(b_ptr + rn[:, None] * K + rk[None, :], mask=mn[:, None], other=0.0)
            part = tl.sum(w.to(tl.float32) * a[None, :], axis=1)
            acc += (part * tl.load(as_ptr + m * s_as0 + kb * s_as1)) * tl.load(
                bs_ptr + s_row * s_bs0 + kb)
        tl.store(out_ptr + m * s_om + rn, acc.to(tl.bfloat16), mask=mn)

    _kernel = _gemv
    return _kernel


def gemv(a, b, a_s, b_s, out=None):
    """a e4m3 [M, K]; b e4m3 [N, K]; a_s f32 [M, K/128] (any strides); b_s f32
    [N/128, K/128]; -> bf16 [M, N]."""
    import torch
    import triton
    m, k = a.shape
    n = b.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    bn = 32 if n >= 8192 else 16
    _get()[(triton.cdiv(n, bn), m)](a, a.stride(0), a_s, a_s.stride(0), a_s.stride(1), b, b_s,
                                    b_s.stride(0), out, out.stride(0), n, K=k, BN=bn, num_warps=4)
    return out


def _check(y, ref):
    rel = ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
    dmax = (y.float() - ref.float()).abs().max().item()
    tol = 2 * 2.0 ** -7 * ref.float().abs().max().item()
    return rel <= 1e-2 and dmax <= tol, f"rel={rel:.2e} max|d|={dmax:.3e} (tol {tol:.3e})"


def _block_mm(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    from vllm import _custom_ops as ops
    if A.shape[0] <= _state["max_m"] and A.shape[0] > 0:
        y = gemv(A, B, As, Bs)
        key = (B.shape[0], B.shape[1])
        if key not in _state["checked"] and not torch.cuda.is_current_stream_capturing():
            ref = ops.cutlass_scaled_mm(A, B.T, out_dtype=torch.bfloat16, scale_a=As, scale_b=Bs.T)
            ok, msg = _check(y, ref)
            if not ok:
                raise RuntimeError(f"{MARKER} ORACLE FAIL N={key[0]} K={key[1]} M={A.shape[0]}: "
                                   f"{msg} — refusing to serve with {GATE}")
            _state["checked"].add(key)
            _log(f"ORACLE PASS N={key[0]} K={key[1]} M={A.shape[0]} vs CUTLASS: {msg}")
        return y
    return ops.cutlass_scaled_mm(A, B.T, out_dtype=torch.bfloat16, scale_a=As, scale_b=Bs.T)


def _block_mm_fake(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                   Bs: torch.Tensor) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[0], dtype=torch.bfloat16, device=A.device)


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_FP8_GEMV_MAX_M > 0."""
    mm = max_m()
    if not mm:
        return None
    if _state["armed"]:
        return _state
    import torch
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
    from vllm.utils.torch_utils import direct_register_custom_op
    import inspect
    params = list(inspect.signature(CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm).parameters)
    if params != ["self", "A", "B", "As", "Bs"]:
        raise RuntimeError(f"{MARKER} anchor drift: apply_block_scaled_mm{params}")
    lib = torch.library.Library("suffix_fp8", "FRAGMENT")
    direct_register_custom_op("block_mm", _block_mm, mutates_args=[], fake_impl=_block_mm_fake,
                              target_lib=lib)
    _state["lib"] = lib
    _state["max_m"] = mm

    def apply_block_scaled_mm(self, A, B, As, Bs):
        if self.config.out_dtype != torch.bfloat16:
            return orig(self, A, B, As, Bs)
        return torch.ops.suffix_fp8.block_mm(A, B, As, Bs)

    orig = CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm
    CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm = apply_block_scaled_mm
    _state["armed"] = True
    _log(f"armed: CutlassFp8BlockScaledMMKernel -> suffix_fp8.block_mm (Triton GEMV for M <= {mm})")
    return _state
