# SPDX-License-Identifier: Apache-2.0
"""ROCm: pass raw MoE padding to AITER (SUFFIX_ROCM_AITER_PAD=1).

vLLM pre-rounds hidden_pad/intermediate_pad before rocm_aiter_ops.fused_moe
and doubles intermediate_pad at TP1; AITER's CK stage1 doubles it again, so
the kernel zero-pads live rows (MXFP4 W4A4 MoE at TP1 on gfx950: Qwen3.8-Flash-
Next GSM8K 0.735 vs 0.964). Same edit as vllm-project/vllm#46201 (open).

The module source is rewritten before its first (only) execution. Gate on:
fail closed if the anchor drifted, an enabled pool must never serve the
corrupting path silently. Stdlib only at import (sitecustomize).
"""
import importlib.util
import os
import sys

GATE = "SUFFIX_ROCM_AITER_PAD"
TARGET = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
OLD = (
    "            hidden_pad = hidden_pad // 128 * 128\n"
    "            intermediate_pad = (\n"
    "                intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)\n"
    "            )\n"
)
NEW = "            pass  # suffix rocm-aiter-pad: AITER aligns internally (vllm#46201)\n"
_MARK = "_suffix_rocm_aiter_pad"


def enabled() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def patch_source(src: str) -> str:
    n = src.count(OLD)
    if n != 1:
        raise RuntimeError(f"[suffix rocm-aiter-pad] anchor found {n}x in {TARGET} "
                           "(expected 1): vLLM drifted, refusing to serve unpatched")
    return src.replace(OLD, NEW)


def install_post_import_hook() -> bool:
    if not enabled():
        return False
    if TARGET in sys.modules:
        raise SystemExit(f"[suffix rocm-aiter-pad] {TARGET} imported before the hook armed")
    if any(getattr(f, _MARK, False) for f in sys.meta_path):
        return True

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
            if fullname != TARGET:
                return None
            # Step aside first: importlib.util.find_spec walks sys.meta_path.
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            def exec_module(module):
                src = patch_source(loader.get_source(fullname))
                exec(compile(src, module.__file__, "exec"), module.__dict__)
                setattr(module, _MARK, True)
                print("[suffix rocm-aiter-pad] ACTIVE: raw MoE padding to AITER",
                      file=sys.stderr, flush=True)

            loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
    return True
