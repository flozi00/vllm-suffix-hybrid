# SPDX-License-Identifier: Apache-2.0
"""ROCm: keep every Triton kernel within SUFFIX_ROCM_LDS_CAP bytes of LDS.

worker-09's MI350P (in-tree amdgpu driver) fails Triton launches above 64 KiB
LDS with HIP 401, sticky for the process, although gfx950 reports 160 KiB
(rocm_lds_probe, ROCm 7.2.3 and 10 images alike). With the gate set (bytes,
e.g. 65536) a compile whose kernel needs more LDS is redone with one software
pipeline stage less until it fits: same math, less prefetch. A kernel still
over the cap at one stage raises Triton's OutOfResources at launch (autotuners
skip that config, fixed configs fail loudly) instead of poisoning the queue.

Patched when triton.compiler.compiler first executes, so the package's
re-exports (triton.compile, JITFunction's binder) see the wrapper. Gate on and
Triton drifted (no compile / max_shared_mem) -> the import fails: fail closed.
Stdlib only at import (sitecustomize).
"""
import importlib.util
import os
import sys

GATE = "SUFFIX_ROCM_LDS_CAP"
TARGET = "triton.compiler.compiler"
_MARK = "_suffix_rocm_lds_cap"


def _say(msg: str) -> None:
    print(f"[suffix rocm-lds-cap] {msg}", file=sys.stderr, flush=True)


def cap() -> int:
    raw = os.environ.get(GATE, "").strip()
    return int(raw) if raw else 0


def patch(mod, limit: int) -> None:
    real_compile, real_max = mod.compile, mod.max_shared_mem

    def compile(src, target=None, options=None, **kw):  # noqa: A001 - Triton's name
        kernel = real_compile(src, target=target, options=options, **kw)
        first = kernel.metadata.shared
        while kernel.metadata.shared > limit and (options or {}).get("num_stages", 0) > 1:
            options = dict(options, num_stages=options["num_stages"] - 1)
            kernel = real_compile(src, target=target, options=options, **kw)
        if first > limit:
            _say(f"{getattr(kernel, 'name', '?')}: {first} B -> {kernel.metadata.shared} B "
                 f"at {(options or {}).get('num_stages')} stage(s)"
                 + ("" if kernel.metadata.shared <= limit else ", still over: OutOfResources"))
        return kernel

    mod.compile = compile
    mod.max_shared_mem = lambda device: min(real_max(device), limit)
    setattr(mod, _MARK, True)


def install_post_import_hook() -> bool:
    limit = cap()
    if limit <= 0:
        return False
    if TARGET in sys.modules:
        raise SystemExit(f"[suffix rocm-lds-cap] {TARGET} imported before the hook armed")
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
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                patch(module, limit)
                _say(f"ACTIVE: Triton kernels capped at {limit} B LDS")

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
    return True
