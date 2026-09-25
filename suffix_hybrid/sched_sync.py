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
``VllmConfig.__post_init__`` via a sys.meta_path finder. It armed and
replaced the class attribute in the live pod's processes, but the pod's own
ENGINE-CONFIG audit line still printed ``async_scheduling=True``: v0.30.0's
``@config`` decorator (vllm/config/utils.py:52-80) wraps config classes
with pydantic's ``dataclasses.dataclass``, whose generated ``__init__``
binds ``__post_init__`` AT DECORATION TIME — a class-level replacement
after import is silently ignored. (A mechanism probe on a different local
pydantic build PASSED, which is exactly why the loud audit line existed:
never trust a mechanism probe over the runtime loud line.)

Attempt v2 (2026-09-24, wrong — kept for the record): the meta_path
hook's ``find_spec`` executed ``__import__("vllm.engine.arg_utils")``
(module #1: executed, patched, "wrapped" loud line fired), then returned
None — which told the outer import machinery "not my module", so it loaded
module #2 fresh through the real finders. The machinery registered #2 in
sys.modules and every downstream importer imported #2. The patched class
never ran; the live pod proved it: "wrapped" printed at 13:23:38Z, config
built at 13:24:02Z with NO "FORCE APPLIED" line and audit line
async_scheduling=True. Reproduced locally in-process
(PATCHED-CLASS-REACHED: False). LESSON (v2): a meta_path finder that
imports its own target and returns None has patched a corpse — the module
in sys.modules is a SECOND, fresh execution of the target.

Fix v3 (this file): no second execution, no post-import race. The finder
returns a spec whose LOADER is a thin proxy wrapping the real loader's
``exec_module``: the import machinery executes the real module exactly once
(through the wrapped loader), and we patch ``EngineArgs`` the instant its
execution finishes, INSIDE the load. There is no window where an unpatched
EngineArgs is reachable downstream, and no corpse-module to confuse. The
wrap sets ``args.async_scheduling = False`` (instance shadowing the
dataclass default) before the original runs; ``create_engine_config`` then
passes the EXPLICIT False into SchedulerConfig (arg_utils.py:2448
"async_scheduling=self.async_scheduling"), so the None→True resolution
chain in ``VllmConfig.__post_init__`` (config/vllm.py:1438-1487)
provably never fires ("is None" reads False). ``AsyncEngineArgs``
subclasses EngineArgs and inherits the wrap through normal attribute
lookup.

Arming (checked once, per process, at install):
  * SUFFIX_HYBRID_SYNC_SCHED=0  → never arm (explicit operator opt-out)
  * else SUFFIX_HYBRID_WRAP=1   → arm (prod never sets WRAP, so this is
    dev-pool-only by construction)
"""
import functools
import importlib.util
import os
import sys

_marker = "_suffix_hybrid_sched_sync"


def _armed() -> bool:
    if os.environ.get("SUFFIX_HYBRID_SYNC_SCHED", "").strip() == "0":
        return False
    return os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1"


def install_post_import_hook() -> None:
    """Arm the loader-wrap finder; idempotent, never raises."""
    if not _armed():
        return
    if any(getattr(finder, _marker, False) for finder in sys.meta_path):
        return

    class _Finder:
        """On request of vllm.engine.arg_utils: return a spec whose loader
        wraps the real loader's exec_module. The machinery loads the real
        module exactly once; we patch EngineArgs right after the module's
        own execution, before the import completes. No second execution,
        no corpse module (the v2 bug)."""

        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != "vllm.engine.arg_utils":
                return None
            if "vllm.engine.arg_utils" in sys.modules:
                # Already imported before our finder was armed: patch the
                # live module directly (idempotent).
                _patch(sys.modules[fullname])
                sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
                return None
            # Step aside FIRST: importlib.util.find_spec itself walks
            # sys.meta_path and would hit our finder again (recursion —
            # caught by test_v3_loader_wrap_real_import).
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            real_spec = importlib.util.find_spec(fullname)
            if real_spec is None or real_spec.loader is None:
                # Never block serving: step aside for the real finders. The
                # audit line will then show async_scheduling=True and the
                # FORCE APPLIED line will be absent — a loud, diagnosable,
                # deliberate failure mode (silence must be loud).
                sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
                return None
            real_exec = real_spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                try:
                    _patch(module)
                except Exception as exc:  # noqa: BLE001 - never kill the load
                    print(f"suffix_hybrid sched-sync: PATCH FAILED: {exc!r}",
                          file=sys.stderr, flush=True)

            real_spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            print("suffix_hybrid sched-sync: v3 loader-wrap installed on "
                  "vllm.engine.arg_utils (single load, mid-load patch)",
                  file=sys.stderr, flush=True)
            return real_spec

    finder = _Finder()
    setattr(finder, _marker, True)
    sys.meta_path.insert(0, finder)
    print("suffix_hybrid sched-sync: armed (async scheduling will be forced "
          "OFF for this process tree when the wrap is active)",
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