# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_HAR_FUSED_NORM=1: vLLM's fused_allreduce_rms_norm (vllm/models/common/ops, used by
DeepseekV32 / GLM 5.3 decoder layers for both all-reduces of every layer) through one host-memory
kernel: two-shot all-reduce + residual add + RMSNorm (suffix_hybrid/host_ar.py _har_rms_kernel).
Upstream's fused path needs FlashInfer's IPC workspace (P2P), so on PCIe-only nodes it falls back
to all-reduce + a separate fused_add_rms_norm launch per call.

Static checks (dtype, rank, plain RMSNorm with a bf16 weight) pick our op at trace time; the
token-count choice (one fused round vs all-reduce + vLLM's kernel) happens inside the custom op,
so torch.compile's skip-guards tracing cannot bake one branch. Inert unless the tp group built a
host all-reduce (SUFFIX_NCCL_BANDS ... har)."""
from __future__ import annotations

import importlib.util
import os
import sys

TARGET = "vllm.models.common.ops.fused_allreduce_rms_norm"
TAG = "suffix har-norm"
_registered = False


def _register_op() -> None:
    global _registered
    if _registered:
        return
    import torch

    from suffix_hybrid import host_ar

    @torch.library.custom_op("suffix_har::ar_rmsnorm", mutates_args=())
    def ar_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return host_ar.TP_COMM.ar_rmsnorm(x, residual, weight, eps)

    @ar_rmsnorm.register_fake
    def _(x, residual, weight, eps):
        return torch.empty_like(x), torch.empty_like(x)

    _registered = True


def apply(module) -> None:
    import torch

    from suffix_hybrid import host_ar

    _register_op()
    orig = module.fused_allreduce_rms_norm

    def fused_allreduce_rms_norm(hidden_states, residual, norm):
        w = getattr(norm, "weight", None)
        if (host_ar.TP_COMM is not None and hidden_states.dtype == torch.bfloat16
                and hidden_states.dim() == 2 and type(norm).__name__ == "RMSNorm"
                and getattr(norm, "has_weight", True) and w is not None and w.dim() == 1
                and w.dtype == torch.bfloat16 and residual is not None):
            return torch.ops.suffix_har.ar_rmsnorm(hidden_states.contiguous(), residual, w,
                                                   float(norm.variance_epsilon))
        return orig(hidden_states, residual, norm)

    module.fused_allreduce_rms_norm = fused_allreduce_rms_norm
    print(f"[{TAG}] ACTIVE: {TARGET}.fused_allreduce_rms_norm -> host all-reduce + add + RMSNorm "
          f"in one kernel (rows that fit one round)", file=sys.stderr, flush=True)


def install_post_import_hook() -> None:
    if os.environ.get("SUFFIX_HAR_FUSED_NORM", "").strip() != "1":
        return

    def run(module):
        try:
            apply(module)
        except Exception as exc:
            raise SystemExit(f"[{TAG}] enabled but installation FAILED: {exc!r}") from exc

    if TARGET in sys.modules:
        run(sys.modules[TARGET])
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != TARGET:
                return None
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                run(module)

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _Finder())
