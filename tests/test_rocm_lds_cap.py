# SPDX-License-Identifier: Apache-2.0
"""CPU test for suffix_hybrid.rocm_lds_cap (no Triton, no GPU).

Silicon check: the rocm_lds_probe boot gate's "cap64k" rows launch OK where
the default rows fail, and an MI350P pod with SUFFIX_ROCM_LDS_CAP=65536 logs
"[suffix rocm-lds-cap] ACTIVE" before vLLM's first Triton compile.
"""
import importlib
import sys
import types

from suffix_hybrid import rocm_lds_cap as cap

SHARED = {1: 16384, 2: 68016, 3: 120000}  # bytes per num_stages (probe 128x128x64)


def _fake():
    calls = []

    def compile(src, target=None, options=None, **kw):  # noqa: A001, ARG001
        calls.append(options["num_stages"])
        return types.SimpleNamespace(name="mm", metadata=types.SimpleNamespace(
            shared=SHARED[options["num_stages"]]))

    return types.SimpleNamespace(compile=compile, max_shared_mem=lambda d: 163840), calls


def test_refits_stages_until_under_cap():
    mod, calls = _fake()
    cap.patch(mod, 65536)
    assert mod.compile("src", options={"num_stages": 3, "num_warps": 4}).metadata.shared == 16384
    assert calls == [3, 2, 1]
    assert mod.max_shared_mem(0) == 65536


def test_fitting_kernel_compiles_once_and_overflow_is_left_to_launch():
    mod, calls = _fake()
    cap.patch(mod, 65536)
    assert mod.compile("src", options={"num_stages": 1}).metadata.shared == 16384 and calls == [1]
    mod, calls = _fake()
    cap.patch(mod, 8192)  # over even at 1 stage: Triton's launch check raises OutOfResources
    assert mod.compile("src", options={"num_stages": 2}).metadata.shared == 16384 and calls == [2, 1]
    assert mod.max_shared_mem(0) == 8192


def test_hook_patches_before_package_reexport(tmp_path, monkeypatch):
    pkg = tmp_path / "ftriton" / "compiler"
    pkg.mkdir(parents=True)
    (tmp_path / "ftriton" / "__init__.py").write_text("from .compiler import compile\n")
    (pkg / "__init__.py").write_text("from .compiler import compile, max_shared_mem\n")
    (pkg / "compiler.py").write_text(
        "import types\n"
        "def max_shared_mem(device):\n    return 163840\n"
        "def compile(src, target=None, options=None, _env_vars=None):\n"
        "    shared = {1: 16384, 2: 68016}[options['num_stages']]\n"
        "    return types.SimpleNamespace(name='k', metadata=types.SimpleNamespace(shared=shared))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cap, "TARGET", "ftriton.compiler.compiler")
    monkeypatch.setenv(cap.GATE, "65536")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert cap.install_post_import_hook()
    try:
        tri = importlib.import_module("ftriton")
        assert tri.compile("s", options={"num_stages": 2}).metadata.shared == 16384
        assert tri.compiler.max_shared_mem(0) == 65536
        assert getattr(tri.compiler.compiler, cap._MARK)
    finally:
        for name in [n for n in sys.modules if n.startswith("ftriton")]:
            sys.modules.pop(name)


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(cap.GATE, raising=False)
    assert cap.install_post_import_hook() is False
