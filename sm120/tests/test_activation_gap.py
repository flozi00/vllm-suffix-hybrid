# SPDX-License-Identifier: Apache-2.0
"""Activation-gap regression tests (qwen-nvfp4kv-dev crash-1, 2026-09-24).

The first live silicon attempt died ARMED-BUT-INERT: marker 1 (armed) fired
but markers 2-4 never did, and stock
FlashInferBackend.validate_configuration rejected ``kv_cache_dtype=nvfp4``
from ``platforms/cuda.py:get_attn_backend_cls``. Root cause (H1,
reproduced locally): the ``sys.meta_path`` post-import finder was APPENDED
to the END of ``sys.meta_path`` — the import machinery stops at the FIRST
finder that returns a spec, and PathFinder answers any regular
site-packages import before an appended meta finder is ever consulted, so
the wrapping loader was never used and ``apply()`` never ran.

These tests pin the fixed contract on CPU:
* the finder fires on a REAL ``importlib.import_module`` of the target
  (the exact path ``AttentionBackendEnum.get_class()`` ->
  ``resolve_obj_by_qualname`` uses at selector/cuda validation time),
* with a Qwen3_5-style configuration (nvfp4 kv_cache_dtype + FLASHINFER
  forced), markers 2 and 3 are emitted on stderr and the patched
  ``supports_kv_cache_dtype`` accepts nvfp4 while the STOCK source
  (same process, same fake platform) rejects it,
* every silent-decline branch now raises or logs loudly (fail-closed):
  a hook that fires but declines on an SM120 host is a process-killing
  error, not a silent inert.
"""
import importlib
import importlib.util
import os
import sys
import textwrap
from pathlib import Path

import pytest

# CI runs pytest --import-mode=importlib, which does not put the test dir on
# sys.path; the sibling import needs it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nvfp4_kv_patch import (  # noqa: E402 - same test dir
    FAKE_STACK,
    FIXTURES,
    PATCH,
    REPO,
)

STACK_HEAD_MARK = "# fake torch.cuda capability"


def _stack(backend_pkg: Path) -> str:
    """FAKE_STACK up to the patch-module part; we script the tail per test."""
    head, _, _tail = FAKE_STACK.partition(STACK_HEAD_MARK)
    return head.format(repo=str(REPO), str_backend_pkg=str(backend_pkg))


# ---------------------------------------------------------------------------
# Pure mechanism: the front-inserted finder MUST fire on a real import.
# ---------------------------------------------------------------------------

def _make_pkg(tmp_path: Path, mod_body: str = "FLAG = 'stock'\n") -> str:
    pkg = tmp_path / "gap_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(mod_body)
    for name in ("gap_pkg", "gap_pkg.mod"):
        sys.modules.pop(name, None)
    return str(tmp_path)


def test_front_finder_fires_on_real_import(tmp_path, monkeypatch):
    # Regression for crash-1: an APPENDED finder is never consulted for a
    # regular import (PathFinder answers first). The fix front-inserts.
    sys.path.insert(0, _make_pkg(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        fired = []
        holder = {}
        finder = PATCH._PostImportFinder("gap_pkg.mod", lambda m: (fired.append(1), m.__dict__.update(FLAG="patched"))[0])
        sys.meta_path.insert(0, finder)
        m = importlib.import_module("gap_pkg.mod")
        assert fired == [1], "post-import callback did NOT fire on a real import"
        assert m.FLAG == "patched"
        # exact single execution of the real module, and one-shot disarm
        assert sys.modules["gap_pkg.mod"] is m
        assert all(f is not finder for f in sys.meta_path)
        holder["ok"] = True
    finally:
        sys.path.remove(str(tmp_path))
    assert holder["ok"]


def test_appended_finder_is_never_consulted(tmp_path, monkeypatch):
    # Documents the OLD (broken) placement — the live-pod failure mode.
    monkeypatch.syspath_prepend(str(tmp_path))
    _make_pkg(tmp_path)
    fired = []
    finder = PATCH._PostImportFinder("gap_pkg.mod", lambda m: fired.append(1))
    sys.meta_path.append(finder)  # the old placement
    try:
        importlib.import_module("gap_pkg.mod")
        assert fired == [], "an appended finder fired (unexpected new behavior)"
        assert sys.modules["gap_pkg.mod"].FLAG == "stock"
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if f is not finder]


def test_install_front_inserts_and_announces(monkeypatch, capsys):
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV", "1")
    try:
        assert PATCH.install_post_import_hook() is True
        assert sys.meta_path[0].__class__.__name__ == "_PostImportFinder"
        err = capsys.readouterr().err
        assert "armed at sys.meta_path[0]" in err
        assert PATCH.install_post_import_hook() is True  # idempotent
        assert sum(
            f.__class__.__name__ == "_PostImportFinder" for f in sys.meta_path) == 1
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path
                            if f.__class__.__name__ != "_PostImportFinder"]


# ---------------------------------------------------------------------------
# End-to-end: real import machinery on the Qwen3_5-style configuration.
# ---------------------------------------------------------------------------

def _run_stack(backend_pkg: Path, tail: str, env_extra=None):
    script = _stack(backend_pkg) + textwrap.dedent(tail)
    env = dict(os.environ, SUFFIX_SM120_NVP4KV="1")
    env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    env.update(env_extra or {})
    import subprocess
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=env, timeout=300)


def test_activation_fires_and_validation_gate_opens(tmp_path):
    """The crash-1 scenario end to end: armed hook + REAL import of the
    backend module (the path AttentionBackendEnum.get_class() takes) with
    a Qwen3_5-style config (nvfp4 KV + FLASHINFER forced) on a fake cc-12
    GPU. Assert apply() ran (marker attr), markers 2+3 printed, and the
    patched validate gate accepts nvfp4 where stock rejects it."""
    backend_pkg = tmp_path / "vllm" / "v1" / "attention"
    tail = '''
        # fake torch.cuda capability: real torch is present (CPU) — monkey the
        # patch module instead (below).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "nvfp4_kv_patch", repo / "sm120" / "nvfp4_kv_patch" / "__init__.py")
        patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
        patch.is_sm120 = lambda capability=None: True   # pretend cc 12.0

        # Explicit SM120-only platform so the STOCK source measurably rejects
        # nvfp4 (family(100) is False on SM120) in this same process.
        class _Platform:
            def is_device_capability_family(self, fam):
                return fam == 120
            def __getattr__(self, k):
                return _Any()
        sys.modules["vllm.platforms"].current_platform = _Platform()

        # The earlier GDN-layer import already loaded the flashinfer PACKAGE
        # (per the pod log, 18:47:35Z) — the hook must not care.
        assert "flashinfer" in sys.modules

        mod_name = "vllm.v1.attention.backends.flashinfer"
        # Real parent package path so the REAL import machinery loads the
        # staged backend through find_spec/PathFinder — the exact code path
        # resolve_obj_by_qualname uses inside AttentionBackendEnum.get_class().
        sys.modules["vllm.v1.attention.backends"].__path__ = [
            str(Path(r"{pkg}") / "backends")]

        # ARM (as sitecustomize does), then import like vLLM does.
        assert patch.install_post_import_hook() is True
        assert type(sys.meta_path[0]).__name__ == "_PostImportFinder"
        backend = importlib.import_module(mod_name)

        # apply() ran inside the load:
        assert getattr(backend, patch.MARKER_ATTR) == patch.PATCH_REVISION
        assert callable(backend._use_fa2_for_nvfp4_kv_on_sm120)
        # activation gate flipped on the module (H2 widen H2):
        assert backend.FlashInferBackend.supports_kv_cache_dtype("nvfp4") is True

        # Stock source, SAME process, SAME fake platform: must still reject
        # (i.e. without the patch the cuda.py validation would have failed
        # with ['kv_cache_dtype not supported'] exactly as the pod did).
        stock_src = (repo / "sm120" / "tests" / "fixtures" / "vllm_0.30.0" /
                     "flashinfer_backend.py").read_text()
        stock_ns = {}
        exec(compile(stock_src, "stock_backend.py", "exec"), stock_ns)
        assert stock_ns["FlashInferBackend"].supports_kv_cache_dtype("nvfp4") is False

        # validate_configuration's kv check (the exact line that killed the
        # pod) now passes for the patched backend:
        reasons = []
        if not backend.FlashInferBackend.supports_kv_cache_dtype("nvfp4"):
            reasons.append("kv_cache_dtype not supported")
        assert reasons == []
        print("ACTIVATION-OK")
    '''.replace("{pkg}", str(backend_pkg))
    proc = _run_stack(backend_pkg, tail)
    assert "ACTIVATION-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    err = proc.stderr
    # Marker set per dossier §6.1 (marker 1 is now position-exact):
    assert "armed at sys.meta_path[0]" in err                       # 1
    assert "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1" in err            # 2
    assert "ACTIVE on SM120: NVFP4 KV -> FlashInfer fa2 route" in err  # 3
    # Marker 4 text is injected by anchor H3a into the live module; verify
    # it's in the transformed source (in-process replay of the same fixture):
    _src, _new, _ = (None, None, None)
    _fixture = (REPO / "sm120" / "tests" / "fixtures" / "vllm_0.30.0" /
                "flashinfer_backend.py").read_text()
    _new, _ = PATCH.patch_backend_source(_fixture)
    assert ("suffix sm120 nvfp4-kv ACTIVE: NVFP4 KV on SM120 routed "
            in _new)  # marker 4 part 1
    assert ("through the FlashInfer fa2 backend (head_dim=%d)" in _new)


def test_gate_flip_decline_is_fail_closed(tmp_path):
    """apply() declining while the host reports SM120 must kill the process
    (armed-but-inert guard), never serve stock FlashInfer silently."""
    backend_pkg = tmp_path / "vllm" / "v1" / "attention"
    tail = '''
        # fake torch.cuda capability
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "nvfp4_kv_patch", repo / "sm120" / "nvfp4_kv_patch" / "__init__.py")
        patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
        patch.is_sm120 = lambda capability=None: True
        assert patch.install_post_import_hook() is True
        # Operator flips the gate off between arm and import:
        del os.environ["SUFFIX_SM120_NVP4KV"]
        sys.modules["vllm.v1.attention.backends"].__path__ = [
            str(Path(r"{pkg}") / "backends")]
        importlib.import_module("vllm.v1.attention.backends.flashinfer")
        print("SHOULD-NOT-GET-HERE")
    '''.replace("{pkg}", str(backend_pkg))
    proc = _run_stack(backend_pkg, tail)
    assert proc.returncode != 0
    assert "SHOULD-NOT-GET-HERE" not in proc.stdout
    assert "armed-but-inert guard" in proc.stderr, proc.stderr


def test_late_arm_patches_immediately_loudly(monkeypatch, capsys):
    """Module already imported when the hook arms (H3 flavour): patch NOW,
    and the inert decline still logs loudly (no silent arm)."""
    import types
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV", "1")
    stub = types.ModuleType("vllm.v1.attention.backends.flashinfer")
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.flashinfer",
                        stub)
    # On this CPU host is_sm120() is False -> apply declines -> loud inert.
    assert PATCH.install_post_import_hook() is True
    err = capsys.readouterr().err
    assert "applying patch immediately" in err
    assert "hook fired; apply() inert" in err


def test_hook_failure_fails_closed(tmp_path, monkeypatch, capsys):
    """If apply() raises inside the hook on an SM120 host, SystemExit kills
    the load (fail-closed) — the pod cannot serve unpatched."""
    import types
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV", "1")
    monkeypatch.setattr(PATCH, "is_sm120", lambda capability=None: True)
    boom = types.ModuleType("vllm.v1.attention.backends.flashinfer")
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.flashinfer",
                        boom)
    with pytest.raises(SystemExit, match="enabled but installation FAILED"):
        # force apply() to raise via a module whose read fails
        boom.__file__ = str(tmp_path / "does_not_exist.py")
        PATCH._hook_callback(boom)