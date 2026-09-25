# SPDX-License-Identifier: Apache-2.0
"""Loader for our cuda-oxide kernels (shared by every kernel lane).

Bundle layout (CI: scripts/oxide_build.py -> runtime_bundle.py --oxide-cubins):
    suffix_hybrid/oxide_cubins/manifest.json
    suffix_hybrid/oxide_cubins/<name>.cubin     one per kernels-oxide/<name>/
manifest: {"arch": "sm_120", "ptxas_version": "...", "cuda_oxide_rev": "...",
           "kernels": [{"name", "file", "sha256", "ptx_isa", "entries": [...]}]}

``ensure_loaded(name)`` verifies the cubin's sha256 and loads it with
cuModuleLoadData on torch's PRIMARY context (``_native.oxide_load_cubin``);
a driver rejection is a hard, specific error — pods never JIT PTX.
``probe()`` loads EVERY manifest cubin and runs the toolchain probe kernel,
printing the boot marker ``OXIDE-PROBE PASS`` (or ``OXIDE-PROBE FAIL``).

Pod boot probe: ``SUFFIX_OXIDE_PROBE=1`` -> sitecustomize runs
``python -m suffix_hybrid.oxide_kernels`` once per pod, before vLLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

MARKER = "[suffix oxide]"
_MANIFEST: dict | None = None
_LOADED: set = set()


def cubin_dir() -> str:
    d = os.environ.get("SUFFIX_OXIDE_CUBINS", "").strip()
    return d or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "oxide_cubins")


def native():
    from suffix_hybrid import _native

    if not getattr(_native, "HAS_OXIDE_KERNELS", False):
        raise RuntimeError(
            "suffix_hybrid._native was built without the `oxide-kernels` "
            "cargo feature (no cuda-oxide kernel host ops)")
    return _native


def manifest(directory: str | None = None) -> dict:
    global _MANIFEST
    if _MANIFEST is not None and directory is None:
        return _MANIFEST
    path = os.path.join(directory or cubin_dir(), "manifest.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"no cuda-oxide cubin manifest at {path} "
                           "(bundle built without scripts/oxide_build.py?)")
    with open(path) as f:
        man = json.load(f)
    if man.get("arch") != "sm_120" or not man.get("kernels"):
        raise RuntimeError(f"oxide manifest {path}: wrong arch or no kernels")
    if directory is None:
        _MANIFEST = man
    return man


def _entry(name: str, directory: str | None = None) -> dict:
    for k in manifest(directory)["kernels"]:
        if k["name"] == name:
            return k
    raise RuntimeError(f"oxide manifest has no kernel {name!r}")


def ensure_loaded(name: str, device: int | None = None,
                  directory: str | None = None) -> int:
    """Load kernel family ``name`` on ``device`` (idempotent). Returns the
    number of resolved entries."""
    import torch

    dev = torch.cuda.current_device() if device is None else device
    key = (name, dev)
    if key in _LOADED:
        return 0
    k = _entry(name, directory)
    with open(os.path.join(directory or cubin_dir(), k["file"]), "rb") as f:
        cubin = f.read()
    if hashlib.sha256(cubin).hexdigest() != k["sha256"]:
        raise RuntimeError(f"oxide cubin sha256 mismatch: {k['file']}")
    n = native().oxide_load_cubin(name, cubin, dev, list(k["entries"]))
    _LOADED.add(key)
    return n


def probe(device: int | None = None) -> str:
    """Driver-load every manifest cubin, run the probe kernel, verify.
    Returns the PASS marker line; raises with the specific cause."""
    import torch

    dev = torch.cuda.current_device() if device is None else device
    man = manifest()
    for k in man["kernels"]:
        ensure_loaded(k["name"], dev)
    n = 4096
    a = torch.arange(n, dtype=torch.int32, device=dev) * 3
    out = torch.zeros(n, dtype=torch.int32, device=dev)
    native().oxide_probe_launch(a.data_ptr(), out.data_ptr(), n, dev,
                                torch.cuda.current_stream(dev).cuda_stream)
    torch.cuda.synchronize(dev)
    want = torch.arange(n, dtype=torch.int32, device=dev) * 4
    if not torch.equal(out, want):
        bad = int((out != want).sum())
        raise RuntimeError(f"oxide probe computed wrong results ({bad}/{n})")
    return (f"{MARKER} OXIDE-PROBE PASS: {len(man['kernels'])} cubins "
            f"driver-loaded + probe kernel verified on cuda:{dev} "
            f"(ptxas {man.get('ptxas_version', '?')}, "
            f"PTX ISA {sorted({k.get('ptx_isa') for k in man['kernels']})}, "
            f"cuda-oxide {man.get('cuda_oxide_rev', '?')[:12]})")


def main() -> int:
    try:
        print(probe(), file=sys.stderr, flush=True)
        return 0
    except Exception as exc:  # the verdict is the evidence
        print(f"{MARKER} OXIDE-PROBE FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
