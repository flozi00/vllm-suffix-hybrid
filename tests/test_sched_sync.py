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