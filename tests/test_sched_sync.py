# SPDX-License-Identifier: Apache-2.0
"""sched_sync: the async-scheduling force-off fix (root cause of −20.6%).

Regression tests (pytest, no vllm needed): the meta_path hook must arm
idempotently, patch exactly once, force False at VllmConfig.__post_init__
entry (before the None→True resolution chain), and print its loud lines so
an operator can see it from the pod log.
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
    # Opt-out wins; WRAP arm; neither -> inert.
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


def test_patch_forces_false_before_resolution_chain(capsys):
    calls = {"post": 0, "chain": None}

    class SchedCfg:
        async_scheduling = None

    class Cfg:
        scheduler_config = SchedCfg()

        def __post_init__(self):
            calls["post"] += 1
            # Simulate the v0.30.0 None→True resolution chain
            # (config/vllm.py:1438-1487): if still None here, async wins.
            if self.scheduler_config.async_scheduling is None:
                self.scheduler_config.async_scheduling = True
                calls["chain"] = "resolved-True"
            else:
                calls["chain"] = "pre-set-kept"

    mod = types.ModuleType("vllm.config.vllm")
    mod.VllmConfig = Cfg
    sys.modules["vllm.config.vllm"] = mod
    sched_sync._patch(mod)
    # Idempotency: patching twice must be a no-op.
    sched_sync._patch(mod)

    c = Cfg()
    c.__post_init__()
    # THE assertion: False survived, the None→True chain never fired.
    assert c.scheduler_config.async_scheduling is False
    assert calls["chain"] == "pre-set-kept"
    assert calls["post"] == 1
    out = capsys.readouterr()
    assert "wrapped" in out.err
    # The loud force line fired at post_init time (writes to stderr).
    assert "async_scheduling" in out.err


def test_patch_skips_module_without_vllmconfig(capsys):
    mod = types.ModuleType("vllm.config.vllm")
    sys.modules["vllm.config.vllm"] = mod
    sched_sync._patch(mod)  # must not raise
    out = capsys.readouterr()
    assert out.err == ""  # silent when there is nothing to patch


def test_hook_arms_once(capsys):
    os.environ["SUFFIX_HYBRID_WRAP"] = "1"
    os.environ.pop("SUFFIX_HYBRID_SYNC_SCHED", None)
    sched_sync.install_post_import_hook()
    sched_sync.install_post_import_hook()  # idempotent
    armed = [f for f in sys.meta_path
             if getattr(f, "_suffix_hybrid_sched_sync", False)]
    assert len(armed) == 1
    capsys.readouterr()