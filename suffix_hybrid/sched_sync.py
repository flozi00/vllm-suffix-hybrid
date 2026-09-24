# SPDX-License-Identifier: Apache-2.0
"""Force synchronous scheduling when the suffix-hybrid wrap is armed.

Why this exists (2026-09-24, wake #18): vLLM v0.30.0 resolves
``SchedulerConfig.async_scheduling`` default ``None`` to **True** inside
``VllmConfig.__post_init__`` (config/vllm.py:1438-1487: "Enable async
scheduling unless there is an incompatible option" — MTP spec decode is in
the *supported* set, UniprocExecutor.supports_async_scheduling() is True,
and an absent CLI flag means "default", not "off"). Empirically confirmed on
the gemma-spec dev pools (diag build 5f23d918, echo burst): the wrapped
RejectionSampler ran 1000+ times ("VTRACE2 heartbeat call#1000") while the
patched DraftTokenIds take handler NEVER fired (VERIFYTRACE: zero lines in
the same window). The take is gated at engine_core.py:625:

    if self.check_for_draft_tokens and not self.async_scheduling and ...

Under async scheduling the draft-width publication path is worker-side and
structured-output-only, so the scheduler never sees our ragged per-row
widths; the engine schedules K-wide pad/placeholder rows instead and the
grader rejects nearly everything ("num_draft_tokens rows but widths are
pad[-1]"), which is the whole −20.6% mystery.

The console write allowlist (api/state.py PARAM_GROUPS) has no
async-scheduling flag at all and dropping "--async-scheduling" is a no-op
(the default resolves ON), so this cannot be fixed from the serving flags.

Fix: a sys.meta_path post-import hook (the pattern proven by
nvfp4_kv_patch) that wraps ``vllm.config.vllm.VllmConfig.__post_init__``.
The wrap sets ``scheduler_config.async_scheduling = False`` BEFORE the
original runs, so:

  * the None→True resolution chain (config/vllm.py:1438-1487) is skipped
    entirely ("is None" reads False),
  * get_scheduler_cls() (config/scheduler.py:212-220) picks the sync
    Scheduler, not AsyncScheduler,
  * engine_core post_step (:625) runs the take every executed step →
    scheduler.update_draft_token_ids sees our ragged widths per row.

Arming (checked once, per process, at install):
  * SUFFIX_HYBRID_SYNC_SCHED=0  → never arm (explicit operator opt-out)
  * else SUFFIX_HYBRID_WRAP=1   → arm (prod never sets WRAP, so this is
    dev-pool-only by construction, and the force line is loud either way)

The hook is registered at interpreter start (sitecustomize) and fires the
first time vllm.config.vllm is imported in whichever process later builds
the VllmConfig — both the API server and the EngineCore child inherit
PYTHONPATH=/plugins, but only the process that actually constructs the
config is affected functionally.
"""
import os
import sys

_gate_marker = "_suffix_hybrid_sched_sync"


def _armed() -> bool:
    if os.environ.get("SUFFIX_HYBRID_SYNC_SCHED", "").strip() == "0":
        return False
    return os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1"


def install_post_import_hook() -> None:
    """Arm the meta_path hook; idempotent, never raises."""
    if not _armed():
        return
    if any(getattr(finder, _gate_marker, False) for finder in sys.meta_path):
        return

    class _SchedSyncFinder:
        """Intercept vllm.config.vllm ONCE, patch, then step aside."""

        def find_module(self, fullname, path=None):  # noqa: ARG002
            # Returning a finder (not a loader) would stall real imports;
            # meta_path finders that return None defer to the next finder,
            # so the real import machinery does all the loading. We only
            # need post-import notification, which find_spec hooks give.
            return None

        # Some Python versions call find_spec on every finder until one
        # returns a spec — we must never return one; just watch.
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            return None

    # The watcher scheme above does not give an after-import callback, so
    # use the standard trick instead: import the module ourselves through
    # the REGULAR finders, patch it, and record it in sys.modules before
    # anything else asks for it.
    class _Hook:
        def find_spec(self, fullname, path, target=None):
            if fullname != "vllm.config.vllm":
                return None
            if "vllm.config.vllm" in sys.modules:
                _patch(sys.modules["vllm.config.vllm"])
                return None
            # Skip ourselves while importing for real.
            sys.meta_path.remove(self)
            try:
                module = __import__(fullname, fromlist=["VllmConfig"])
            finally:
                if _armed():
                    sys.meta_path.insert(0, self)
            _patch(module)
            return None

    hook = _Hook()
    setattr(hook, _gate_marker, True)
    sys.meta_path.insert(0, hook)
    print("suffix_hybrid sched-sync: armed (async scheduling will be "
          "forced OFF for this process tree when the wrap is active)",
          file=sys.stderr, flush=True)


def _patch(module) -> None:
    cfg = getattr(module, "VllmConfig", None)
    if cfg is None or getattr(cfg.__post_init__, _gate_marker, False):
        return
    original = cfg.__post_init__

    import functools

    @functools.wraps(original)
    def post_init(self, *args, **kwargs):
        # PRE-set before the original resolution chain runs so the
        # None→True branch (config/vllm.py:1438-1487) is bypassed for
        # real, not corrected after other code already read True.
        sched = getattr(self, "scheduler_config", None)
        had = getattr(sched, "async_scheduling", None) if sched else None
        if sched is not None:
            sched.async_scheduling = False
        result = original(self, *args, **kwargs)
        print(f"suffix_hybrid sched-sync: scheduler_config.async_scheduling "
              f"{had!r} -> False (sync take path restored: post_step "
              f"engine_core.py:625 now runs)", file=sys.stderr, flush=True)
        return result

    setattr(post_init, _gate_marker, True)
    cfg.__post_init__ = post_init
    print("suffix_hybrid sched-sync: VllmConfig.__post_init__ wrapped",
          file=sys.stderr, flush=True)