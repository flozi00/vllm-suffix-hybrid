# SPDX-License-Identifier: Apache-2.0
"""Force synchronous scheduling when the suffix-hybrid wrap is armed.

Why this exists (2026-09-24): vLLM v0.30.0 resolves
``SchedulerConfig.async_scheduling`` default ``None`` to **True** inside
``VllmConfig.__post_init__`` (config/vllm.py:1438-1487) for MTP spec decode
on UniprocExecutor — v0.30.0 serves *with* async scheduling by default.
Empirically confirmed on the gemma-spec dev pools (diag build 5f23d918,
echo burst): the wrapped RejectionSampler ran 1000+ times ("VTRACE2
heartbeat call#1000") while the patched DraftTokenIds take handler NEVER
fired (VERIFYTRACE: zero lines) — the take is gated at engine_core.py:625:

    if self.check_for_draft_tokens and not self.async_scheduling and ...

Under async scheduling the draft-width publication path is worker-side and
structured-output-only, so the scheduler never sees our ragged per-row
widths; the engine schedules K-wide pad/placeholder rows instead — the
whole "97% zero-accept" mystery.

Attempt v1 (2026-09-24, wrong — kept for the record): wrap
``VllmConfig.__post_init__`` via a sys.meta_path hook. It armed and
replaced the class attribute in the live pod's processes, but the pod's own
ENGINE-CONFIG audit line still printed ``async_scheduling=True``: v0.30.0's
``@config`` decorator (vllm/config/utils.py:52-80) wraps config classes
with pydantic's ``dataclasses.dataclass``, whose generated ``__init__``
binds ``__post_init__`` AT DECORATION TIME — a class-level replacement
after import is silently ignored. (A mechanism probe on a different local
pydantic build PASSED, which is exactly why the loud audit line existed:
never trust a mechanism probe over the runtime loud line.)

Fix v2 (this file): wrap ``EngineArgs.create_engine_config`` instead
(vllm/engine/arg_utils.py:2040). EngineArgs is a PLAIN stdlib @dataclass
(arg_utils.py:431-432) — method lookup at call time hits our class-level
wrap — and the wrap sets ``args.async_scheduling = False`` (an instance
attribute shadowing the dataclass default) BEFORE the original builds
VllmConfig. SchedulerConfig is then constructed with an EXPLICIT False
(arg_utils.py:2448 ``async_scheduling=self.async_scheduling``), which the
None→True resolution chain in ``VllmConfig.__post_init__``
(config/vllm.py:1438-1487) provably skips ("is None" reads False).
``AsyncEngineArgs`` subclasses EngineArgs and overrides only __init__, so
async setups inherit the wrapped method through normal attribute lookup.

Arming (checked once, per process, at install):
  * SUFFIX_HYBRID_SYNC_SCHED=0  → never arm (explicit operator opt-out)
  * else SUFFIX_HYBRID_WRAP=1   → arm (prod never sets WRAP, so this is
    dev-pool-only by construction)

The sys.meta_path watcher remains only as the on-import TRIGGER: it patches
EngineArgs the first time ``vllm.engine.arg_utils`` is imported in this
process. (The v1 VllmConfig class-swap is gone — it provably does nothing
under the pod's pydantic, and a dead hook breeds false confidence.)
"""
import functools
import os
import sys

_marker = "_suffix_hybrid_sched_sync"


def _armed() -> bool:
    if os.environ.get("SUFFIX_HYBRID_SYNC_SCHED", "").strip() == "0":
        return False
    return os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1"


def install_post_import_hook() -> None:
    """Arm the meta_path hook; idempotent, never raises."""
    if not _armed():
        return
    if any(getattr(finder, _marker, False) for finder in sys.meta_path):
        return

    class _Hook:
        """On first request of vllm.engine.arg_utils: step aside, import
        it through the REAL finders, patch EngineArgs, then go quiet
        (returning None lets the normal import machinery proceed with the
        now-patched module)."""

        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != "vllm.engine.arg_utils":
                return None
            sys.meta_path.remove(self)
            try:
                module = __import__(fullname, fromlist=["EngineArgs"])
            finally:
                # Re-insert lazily-guarded: idempotent patch makes a
                # double fire harmless (env may flip between calls).
                if _armed() and not any(
                        getattr(f, _marker, False) for f in sys.meta_path):
                    sys.meta_path.insert(0, self)
            _patch(module)
            return None

    hook = _Hook()
    setattr(hook, _marker, True)
    sys.meta_path.insert(0, hook)
    print("suffix_hybrid sched-sync: armed (async scheduling will be "
          "forced OFF for this process tree when the wrap is active)",
          file=sys.stderr, flush=True)


def _patch(module) -> None:
    """Wrap EngineArgs.create_engine_config; idempotent per class."""
    cls = getattr(module, "EngineArgs", None)
    if cls is None or getattr(
            getattr(cls, "create_engine_config", None), _marker, False):
        return
    original = cls.create_engine_config

    @functools.wraps(original)
    def create_engine_config(self, *args, **kwargs):
        # Instance attribute shadowing the dataclass default: the original
        # reads self.async_scheduling at arg_utils.py:2448 and passes our
        # explicit False into SchedulerConfig, so the None→True chain in
        # VllmConfig.__post_init__ (config/vllm.py:1438-1487) never fires.
        had = getattr(self, "async_scheduling", None)
        self.async_scheduling = False
        result = original(self, *args, **kwargs)
        sched = getattr(result, "scheduler_config", None)
        final = (getattr(sched, "async_scheduling", "ATTR-MISSING")
                 if sched is not None else "NO-SCHED-CFG")
        print(f"suffix_hybrid sched-sync: FORCE APPLIED "
              f"async_scheduling {had!r} -> scheduler_config.{final!r} "
              f"(sync take path: engine_core.py:625)",
              file=sys.stderr, flush=True)
        return result

    setattr(create_engine_config, _marker, True)
    cls.create_engine_config = create_engine_config
    print("suffix_hybrid sched-sync: EngineArgs.create_engine_config "
          "wrapped (explicit async_scheduling=False at config build)",
          file=sys.stderr, flush=True)