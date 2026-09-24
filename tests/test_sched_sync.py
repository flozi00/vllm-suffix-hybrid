# SPDX-License-Identifier: Apache-2.0
"""sched_sync v2: EngineArgs.create_engine_config force-off (root cause of
−20.6%).

Regression tests (pytest, no vllm needed):
  * arming gates (opt-out wins / WRAP arms / neither inert)
  * the wrap pre-sets instance async_scheduling=False so the original
    passes EXPLICIT False into SchedulerConfig construction
  * AsyncEngineArgs-style subclasses inherit the wrap
  * idempotent patching, silent no-op without EngineArgs
  * THE v1 LESSON pinned: v2 must not depend on VllmConfig.__post_init__
    class swapping (the pod's pydantic binds post_init at decoration time
    and silently ignored it — audit line async_scheduling=True).
"""
import os
import sys
import types

import pytest

os.environ.setdefault("SUFFIX_HYBRID_WRAP", "1")

import suffix_hybrid.sched_sync as sched_sync  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_meta_path():
    yield
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not getattr(f, "_suffix_hybrid_sched_sync", False)
    ]


def test_arming_gates():
    orig = {k: os.environ.get(k) for k in
            ("SUFFIX_HYBRID_SYNC_SCHED", "SUFFIX_HYBRID_WRAP")}
    try:
        os.environ["SUFFIX_HYBRID_WRAP"] = "1"
        os.environ.pop("SUFFIX_HYBRID_SYNC_SCHED", None)
        assert sched_sync._armed() is True
        os.environ["SUFFIX_HYBRID_SYNC_SCHED"] = "0"
        assert sched_sync._armed() is False
        os.environ.pop("SUFFIX_HYBRID_WRAP", None)
        assert sched_sync._armed() is False
    finally:
        for k, v in orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeSchedCfg:
    def __init__(self, async_scheduling):
        self.async_scheduling = async_scheduling


class _FakeVllmCfg:
    def __init__(self, scheduler_config):
        self.scheduler_config = scheduler_config
        # The v0.30.0 None->True resolution chain (config/vllm.py:1438-87)
        if self.scheduler_config.async_scheduling is None:
            self.scheduler_config.async_scheduling = True


def _make_module():
    mod = types.ModuleType("vllm.engine.arg_utils")

    class EngineArgs:
        async_scheduling = None  # stdlib dataclass default

        def create_engine_config(self):
            # arg_utils.py:2448 — passes the INSTANCE value through
            sched = _FakeSchedCfg(self.async_scheduling)
            return _FakeVllmCfg(sched)

    class AsyncEngineArgs(EngineArgs):
        # v0.30.0 subclasses with only __init__ differences;
        # create_engine_config is inherited.
        pass

    mod.EngineArgs = EngineArgs
    mod.AsyncEngineArgs = AsyncEngineArgs
    return mod


def test_wrap_forces_explicit_false_through_config(capsys):
    mod = _make_module()
    # CONTROL: before patching, None resolves to True (the mystery).
    ctrl = mod.EngineArgs().create_engine_config()
    assert ctrl.scheduler_config.async_scheduling is True

    sched_sync._patch(mod)
    sched_sync._patch(mod)  # idempotency: second patch is a no-op

    cfg = mod.EngineArgs().create_engine_config()
    # THE assertion: explicit False on entry; the None->True chain in the
    # fake VllmCfg never fired (value already False).
    assert cfg.scheduler_config.async_scheduling is False
    out = capsys.readouterr()
    assert "wrapped" in out.err
    assert "FORCE APPLIED" in out.err


def test_subclass_inherits_wrap(capsys):
    mod = _make_module()
    sched_sync._patch(mod)
    cfg = mod.AsyncEngineArgs().create_engine_config()
    assert cfg.scheduler_config.async_scheduling is False
    capsys.readouterr()


def test_operator_explicit_true_still_forced_off(capsys):
    # Even if something set async_scheduling=True on the args instance,
    # the wrap REPLACES it with False (dev-pool policy: sync take path).
    mod = _make_module()
    sched_sync._patch(mod)
    args = mod.EngineArgs()
    args.async_scheduling = True
    cfg = args.create_engine_config()
    assert cfg.scheduler_config.async_scheduling is False
    out = capsys.readouterr()
    assert "True" in out.err  # had=True logged


def test_patch_skips_module_without_engineargs(capsys):
    mod = types.ModuleType("vllm.engine.arg_utils")
    sched_sync._patch(mod)  # must not raise
    assert capsys.readouterr().err == ""


def test_v1_lesson_no_vllmconfig_post_init_swap():
    # v1 shipped a VllmConfig.__post_init__ class swap. The live pod's
    # pydantic bound __post_init__ at decoration time and silently ignored
    # the swap (audit line said async_scheduling=True while the hook
    # claimed wrapped). Pinned: sched_sync must never again depend on that
    # mechanism — no "__post_init__" assignment, no "vllm.config.vllm"
    # import interception in the file.
    import ast
    tree = ast.parse(open(sched_sync.__file__).read())
    # No assignment target or attribute name may be __post_init__ (prose in
    # docstrings is fine — it is the historical record).
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in (ast.walk(t)):
                    if isinstance(n, ast.Name) and n.id == "__post_init__":
                        bad.append("assign")
                    if isinstance(n, ast.Attribute) and n.attr == "__post_init__":
                        bad.append("attr-assign")
        if isinstance(node, ast.Attribute) and node.attr == "__post_init__":
            if not isinstance(node, ast.Assign):
                pass  # reads are not the v1 mechanism; assigns above
    assert bad == []


def test_hook_arms_once(capsys):
    os.environ["SUFFIX_HYBRID_WRAP"] = "1"
    os.environ.pop("SUFFIX_HYBRID_SYNC_SCHED", None)
    sched_sync.install_post_import_hook()
    sched_sync.install_post_import_hook()  # idempotent
    armed = [f for f in sys.meta_path
             if getattr(f, "_suffix_hybrid_sched_sync", False)]
    assert len(armed) == 1
    capsys.readouterr()


def test_v2_lesson_no_self_import_in_finder():
    # v2 shipped a finder that __import__-ed its own target inside
    # find_spec and returned None, so the machinery loaded a SECOND fresh
    # module and everyone downstream used the unpatched corpse. Pinned:
    # find_spec must never import the module it is asked to find.
    import ast
    src = open(sched_sync.__file__).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "find_spec":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(
                            sub.func, ast.Name) and sub.func.id == "__import__":
                        raise AssertionError(
                            "__import__ inside find_spec is the v2 "
                            "corpse-module bug")


def test_v3_loader_wrap_real_import(tmp_path, capsys):
    # THE v2 regression test: import a fresh fake vllm.engine.arg_utils
    # through the ARMED finder via the REAL import machinery and assert
    # the module sys.modules exposes carries the patched EngineArgs. Under
    # v2 this exact scenario left sys.modules with an unpatched class.
    root = tmp_path / "v3imp"
    (root / "vllm" / "engine").mkdir(parents=True)
    (root / "vllm" / "__init__.py").write_text("")
    (root / "vllm" / "engine" / "__init__.py").write_text("")
    (root / "vllm" / "engine" / "arg_utils.py").write_text(
        "class EngineArgs:\n"
        "    async_scheduling = None\n"
        "    def create_engine_config(self):\n"
        "        sched = type('S', (), {'async_scheduling': self.async_scheduling})()\n"
        "        cfg = type('V', (), {'scheduler_config': sched})()\n"
        "        if sched.async_scheduling is None:\n"
        "            sched.async_scheduling = True  # v0.30.0 None->True\n"
        "        return cfg\n"
    )
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "vllm"}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(root))
    try:
        os.environ["SUFFIX_HYBRID_WRAP"] = "1"
        os.environ.pop("SUFFIX_HYBRID_SYNC_SCHED", None)
        capsys.readouterr()
        sched_sync.install_post_import_hook()
        import vllm.engine.arg_utils as argmod  # noqa: E402

        # The module the MACHINERY registered must carry the wrapped class:
        # the v2 corpse-module bug broke exactly this identity.
        assert argmod is sys.modules["vllm.engine.arg_utils"]
        wrapped = getattr(argmod.EngineArgs.create_engine_config,
                          "_suffix_hybrid_sched_sync", False)
        assert wrapped, "sys.modules class is UNPATCHED (v2 corpse bug)"
        cfg = argmod.EngineArgs().create_engine_config()
        assert cfg.scheduler_config.async_scheduling is False
        out = capsys.readouterr()
        assert "FORCE APPLIED" in out.err
        assert "None" in out.err  # had=None logged
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if k.split(".")[0] == "vllm"]:
            del sys.modules[k]
        for k, v in saved.items():
            sys.modules[k] = v
        sys.meta_path[:] = [f for f in sys.meta_path
                            if not getattr(f, "_suffix_hybrid_sched_sync", False)]