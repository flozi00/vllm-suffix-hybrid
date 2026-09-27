# SPDX-License-Identifier: Apache-2.0
"""Size-routed NCCL all-reduce: SUFFIX_NCCL_SMALL_ALGO=<NCCL_ALGO value>.

On PCIe nodes without GPU P2P (worker-06: every fast all-reduce in vLLM 0.30.0
is disabled, NCCL goes through host memory) no single NCCL algorithm wins at
all sizes: tree is 2-2.7x faster than ring at decode-sized messages (384 KiB -
1.5 MiB) but a global NCCL_ALGO=allreduce:tree made 60k-token prefills 27 %
slower (12 MiB chunk all-reduces) -- and NCCL_ALGO=Tree globally kills
all-gather ("invalid usage"). NCCL has no size threshold knob, so this builds a
SECOND PyNcclCommunicator per CudaCommunicator with NCCL_ALGO set only while
it initialises (NCCL reads the variable at communicator init) and routes
all-reduces of <= SUFFIX_NCCL_SMALL_BYTES (default 4 MiB) to it; everything
else keeps the default communicator. Fail-closed on anchor drift; unset = inert.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ENV = "SUFFIX_NCCL_SMALL_ALGO"
BYTES_ENV = "SUFFIX_NCCL_SMALL_BYTES"
TARGET = "vllm.distributed.device_communicators.cuda_communicator"
TAG = "suffix nccl-split"
OLD = ("        assert pynccl_comm is not None\n"
       "        out = pynccl_comm.all_reduce(input_)\n")
NEW = ("        assert pynccl_comm is not None\n"
       f"        _small = getattr(self, '_suffix_small_nccl', None)  # {TAG}\n"
       "        if _small is not None and (input_.numel() * input_.element_size()\n"
       "                                   <= self._suffix_small_bytes):\n"
       "            pynccl_comm = _small\n"
       "        out = pynccl_comm.all_reduce(input_)\n")


class PatchDriftError(RuntimeError):
    pass


def _say(msg: str) -> None:
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def patch_all_reduce(src: str) -> str:
    """CudaCommunicator.all_reduce source with the size route (dedented)."""
    import ast
    import textwrap

    cls = next((n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)
                and n.name == "CudaCommunicator"), None)
    fn = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef)
               and n.name == "all_reduce"), None)
    if fn is None:
        raise PatchDriftError("CudaCommunicator.all_reduce missing")
    body = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    if body.count(OLD) != 1:
        raise PatchDriftError(f"all_reduce anchor: expected 1, found {body.count(OLD)}")
    out = textwrap.dedent(body.replace(OLD, NEW))
    compile(out, f"<{TAG}>", "exec")
    return out


def apply(module) -> None:
    algo = os.environ.get(ENV, "").strip()
    if not algo:
        return
    if os.environ.get("NCCL_ALGO"):
        raise PatchDriftError(f"{ENV} needs NCCL_ALGO unset (it would apply to BOTH communicators)")
    small_bytes = int(os.environ.get(BYTES_ENV, str(4 << 20)))
    import __future__
    import linecache
    from pathlib import Path

    path = Path(module.__file__)
    src = patch_all_reduce(path.read_text())
    fname = f"{path}.{TAG.replace(' ', '-')}.py"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns: dict = {}
    exec(compile(src, fname, "exec", __future__.annotations.compiler_flag,
                 dont_inherit=True), module.__dict__, ns)
    cls = module.CudaCommunicator
    orig_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._suffix_small_nccl = None
        self._suffix_small_bytes = small_bytes
        if self.world_size <= 1 or self.pynccl_comm is None or self.pynccl_comm.disabled:
            return
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        os.environ["NCCL_ALGO"] = algo
        try:
            self._suffix_small_nccl = PyNcclCommunicator(group=self.cpu_group, device=self.device)
        finally:
            os.environ.pop("NCCL_ALGO", None)
        _say(f"second communicator NCCL_ALGO={algo} for all-reduce <= {small_bytes} B "
             f"(world {self.world_size}, {self.device})")

    ns["all_reduce"].__qualname__ = "CudaCommunicator.all_reduce"
    cls.all_reduce = ns["all_reduce"]
    cls.__init__ = __init__
    _say(f"ACTIVE: CudaCommunicator.all_reduce routes <= {small_bytes} B to NCCL_ALGO={algo}")


def install_post_import_hook() -> None:
    """sitecustomize entry; fail closed (SystemExit) on drift while enabled."""
    if not os.environ.get(ENV, "").strip():
        return

    def run(module):
        try:
            apply(module)
        except Exception as exc:
            raise SystemExit(f"[{TAG}] enabled but installation FAILED: {exc}") from exc

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
