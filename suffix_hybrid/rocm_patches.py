# SPDX-License-Identifier: Apache-2.0
"""ROCm patches for vLLM modules, one gate each (armed by sitecustomize).

A patch rewrites a module's source before its first (only) execution and/or
calls an `after` hook with the executed module. Gate on and the anchor not
found exactly once -> that import fails: an enabled pool must never serve the
unpatched path silently. Stdlib only at import (hooks import lazily).

SUFFIX_ROCM_AITER_PAD=1: vLLM pre-rounds hidden_pad/intermediate_pad before
  rocm_aiter_ops.fused_moe and doubles intermediate_pad at TP1; AITER's CK
  stage1 doubles it again, so the kernel zero-pads live rows (MXFP4 W4A4 MoE at
  TP1 on gfx950: Qwen3.8-Flash-Next GSM8K 0.735 vs 0.964). Same edit as
  vllm-project/vllm#46201 (open).
SUFFIX_ROCM_QSA_TOPK_ROWS=1: the Qwen4Exp QSA indexer runs top_k_per_row_decode
  over every row of a step at once. Above 384 rows vLLM's gfx950 dispatch
  leaves the length-aware kernel for the generic radix topKPerRowDecode, whose
  code object declares a hostcall buffer, and HIP refuses that kernel on hosts
  without PCIe atomics (worker-09: "Pcie atomics not enabled, hostcall not
  supported", then a sticky HIP 401): the first step with >384 new tokens
  killed the engine. Chunks of <= 384 rows stay on the length-aware kernel;
  rows are independent, so the selection is unchanged.
SUFFIX_ROCM_QSA_MQA=1: the QSA indexer scores only each row's visible columns
  (suffix_hybrid/kernels/qsa_mqa_rocm.py) instead of the whole 262k-context
  page-table width: 16.1 of 62.9 ms/step at 32 concurrent MTP-4 requests.
SUFFIX_ROCM_MXFP4_A16=1: dense MXFP4 linears (non-ASM AITER path) at M <=
  SUFFIX_ROCM_MXFP4_A16_MAX_M (default 32) run as one AITER gemm_a16wfp4 launch
  that quantizes the activations in-kernel, instead of dynamic_mxfp4_quant +
  gemm_afp4wfp4 (+ split-K reduce): ~180 fewer launches per MTP-4 decode step
  (suffix_hybrid/kernels/mxfp4_a16_rocm.py). Source rewrite, because
  gemm_with_dynamic_quant is registered as a custom op when the module executes.
"""
import importlib.util
import os
import sys
from typing import NamedTuple


class Patch(NamedTuple):
    target: str
    label: str
    old: str = ""    # source anchor replaced by `new` before the first exec ("" = none)
    new: str = ""
    after: str = ""  # "module:function" called with the executed module ("" = none)


PATCHES = {
    "SUFFIX_ROCM_AITER_PAD": Patch(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        "raw MoE padding to AITER",
        "            hidden_pad = hidden_pad // 128 * 128\n"
        "            intermediate_pad = (\n"
        "                intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)\n"
        "            )\n",
        "            pass  # suffix rocm-aiter-pad: AITER aligns internally (vllm#46201)\n"),
    "SUFFIX_ROCM_QSA_TOPK_ROWS": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA top-k in <=384-row chunks",
        "    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))\n",
        "    rows_per_chunk = min(384, max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))"
        "  # suffix: hostcall-free top-k\n"),
    "SUFFIX_ROCM_QSA_MQA": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA scores over visible columns only",
        after="suffix_hybrid.kernels.qsa_mqa_rocm:install"),
    "SUFFIX_ROCM_MXFP4_A16": Patch(
        "vllm.model_executor.kernels.linear.mxfp4.aiter",
        "dense MXFP4 small M via AITER gemm_a16wfp4",
        "            if x_scales is None:\n"
        "                x_q, x_s = dynamic_mxfp4_quant(x)\n",
        "            if x_scales is None:  # suffix SUFFIX_ROCM_MXFP4_A16: small M -> gemm_a16wfp4\n"
        "                from suffix_hybrid.kernels.mxfp4_a16_rocm import gemm_a16\n"
        "                y = gemm_a16(x, weight, weight_scale, out_dtype)\n"
        "                if y is not None:\n"
        "                    return y\n"
        "                x_q, x_s = dynamic_mxfp4_quant(x)\n"),
}
_MARK = "_suffix_rocm_patch"


def enabled() -> dict:
    """Target module -> [Patch] for every gate set to 1, in PATCHES order."""
    todo: dict = {}
    for gate, p in PATCHES.items():
        if os.environ.get(gate, "").strip() == "1":
            todo.setdefault(p.target, []).append(p)
    return todo


def patch_source(patch: Patch, src: str) -> str:
    n = src.count(patch.old)
    if n != 1:
        raise RuntimeError(f"[suffix rocm-patch] anchor found {n}x in {patch.target} "
                           "(expected 1): vLLM drifted, refusing to serve unpatched")
    return src.replace(patch.old, patch.new)


def install_post_import_hook() -> bool:
    todo = enabled()
    if not todo:
        return False
    early = sorted(t for t in todo if t in sys.modules)
    if early:
        raise SystemExit(f"[suffix rocm-patch] {early} imported before the hook armed")
    if any(getattr(f, _MARK, False) for f in sys.meta_path):
        return True

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
            patches = todo.pop(fullname, None)
            if patches is None:
                return None
            # importlib.util.find_spec walks sys.meta_path: step aside meanwhile,
            # for good once every target is armed.
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if todo:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            def exec_module(module):
                src = loader.get_source(fullname)
                for p in patches:
                    if p.old:
                        src = patch_source(p, src)
                exec(compile(src, module.__file__, "exec"), module.__dict__)
                for p in patches:
                    if p.after:
                        mod_name, fn = p.after.split(":")
                        getattr(importlib.import_module(mod_name), fn)(module)
                    print(f"[suffix rocm-patch] ACTIVE: {p.label} ({fullname})",
                          file=sys.stderr, flush=True)
                setattr(module, _MARK, True)

            loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
    return True
