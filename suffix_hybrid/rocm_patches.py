# SPDX-License-Identifier: Apache-2.0
"""ROCm source patches for vLLM, one gate each (armed by sitecustomize).

Each patch rewrites one module's source before its first (only) execution.
Gate on and the anchor not found exactly once -> that import fails: an enabled
pool must never serve the unpatched path silently. Stdlib only at import.

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
"""
import importlib.util
import os
import sys
from typing import NamedTuple


class Patch(NamedTuple):
    target: str
    old: str
    new: str
    label: str


PATCHES = {
    "SUFFIX_ROCM_AITER_PAD": Patch(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        "            hidden_pad = hidden_pad // 128 * 128\n"
        "            intermediate_pad = (\n"
        "                intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)\n"
        "            )\n",
        "            pass  # suffix rocm-aiter-pad: AITER aligns internally (vllm#46201)\n",
        "raw MoE padding to AITER"),
    "SUFFIX_ROCM_QSA_TOPK_ROWS": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))\n",
        "    rows_per_chunk = min(384, max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))"
        "  # suffix: hostcall-free top-k\n",
        "QSA top-k in <=384-row chunks"),
}
_MARK = "_suffix_rocm_patch"


def enabled() -> dict:
    """Target module -> Patch for every gate set to 1."""
    return {p.target: p for gate, p in PATCHES.items() if os.environ.get(gate, "").strip() == "1"}


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
            patch = todo.pop(fullname, None)
            if patch is None:
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
                src = patch_source(patch, loader.get_source(fullname))
                exec(compile(src, module.__file__, "exec"), module.__dict__)
                setattr(module, _MARK, True)
                print(f"[suffix rocm-patch] ACTIVE: {patch.label} ({fullname})",
                      file=sys.stderr, flush=True)

            loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
    return True
