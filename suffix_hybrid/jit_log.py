# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_JIT_LOG=1: one "[suffix jit]" stderr line per Triton compile (JITFunction._do_compile,
i.e. an in-process cache miss) with the kernel name and wall ms. A compile while serving stalls
every in-flight request; the timestamps in the pod log show which kernels still compile after
warm-up and when."""
import functools
import os
import sys
import time


def install(module) -> None:
    cls = module.JITFunction
    orig = cls._do_compile

    @functools.wraps(orig)
    def _do_compile(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            name = getattr(getattr(self, "fn", None), "__qualname__", None) or repr(self)
            print(f"[suffix jit] {name} compiled in {(time.perf_counter() - t0) * 1e3:.0f} ms "
                  f"(pid {os.getpid()})", file=sys.stderr, flush=True)

    cls._do_compile = _do_compile
