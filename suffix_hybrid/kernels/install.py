# SPDX-License-Identifier: Apache-2.0
"""Installer gate for the kernels package (sitecustomize-side entry point).

SUFFIX_KERNELS=1 is the gate (dev pools only — prod never sets it). It is
INDEPENDENT of SUFFIX_HYBRID_WRAP so kernel experiments run without the
speculative wrap and vice versa. Registration is inert by design: dispatch
still follows --kernel-config ir_op_priority, so a pool with the gate on but
no priority entry behaves exactly like before.

Unlike the wrap (an installation failure there is fatal by design — an
enabled worker must not silently run unpatched), a KERNEL registration failure
is a logged refusal: the serving path never depended on our kernels, so
degrading to vllm_c/native is the correct outcome, not a crash.
"""
from __future__ import annotations

import os
import sys


def install_kernels() -> dict | None:
    """Register kernel providers if gated. Returns a summary dict, or None
    when the gate is off (silent, zero cost on every non-dev process)."""
    if os.environ.get("SUFFIX_KERNELS", "").strip() != "1":
        return None
    ops = [o.strip() for o in
           os.environ.get("SUFFIX_KERNELS_OPS", "rmsnorm").split(",") if o.strip()]
    summary = {"gated": True, "ops": {}, "errors": {}}
    for op in ops:
        try:
            if op == "rmsnorm":
                from .rmsnorm import register_impls
            else:
                raise ValueError(f"unknown kernel op group {op!r}")
            summary["ops"][op] = register_impls()
        except Exception as exc:  # logged refusal, never fatal
            summary["errors"][op] = f"{type(exc).__name__}: {exc}"
    print(f"suffix_hybrid kernels: {summary}", file=sys.stderr, flush=True)
    return summary
