"""Scheduler-side spec-decode probe for suffix-hybrid.

Wraps ``Scheduler.update_draft_token_ids`` (class level) so that every
step the EngineCore performs (engine_core.py:625-628:
``model_executor.take_draft_token_ids()`` ->
``scheduler.update_draft_token_ids(draft_token_ids)``) is observable on
stderr, independent of the DraftTokensHandler / rejection-sampler wraps
that production pods showed going silent.

vLLM v0.30.0 reference (mirror):
  scheduler.py:2404  ``def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None``
  scheduler.py:2405-2407  zips ``draft_token_ids.req_ids`` with
  ``draft_token_ids.draft_token_ids`` (both are plain lists:
  req_ids: list[str] ([num_reqs]), draft_token_ids: list[list[int]]
  (num_reqs x num_draft_tokens) — vllm/v1/outputs.py ``DraftTokenIds``).

stdlib only; self-contained.
"""

from __future__ import annotations

import functools
import os
import sys

GUARD_ATTR = "_suffix_hybrid_sched_probe"

_ENV_MAX = "SUFFIX_HYBRID_SCHEDTRACE_MAX"
_ENV_ON = "SUFFIX_HYBRID_SCHEDTRACE"
_DEFAULT_MAX = 24


def _trace_max() -> int:
    try:
        return max(0, int(os.environ.get(_ENV_MAX, "") or _DEFAULT_MAX))
    except (TypeError, ValueError):
        return _DEFAULT_MAX


def _describe(draft_token_ids) -> tuple[str, str]:
    """Extract (n, widths, rid0) exactly as update_draft_token_ids
    consumes the argument (scheduler.py:2405-2407: zip(req_ids,
    draft_token_ids))."""
    if draft_token_ids is None:
        return "n=none", ""
    req_ids = getattr(draft_token_ids, "req_ids", None)
    rows = getattr(draft_token_ids, "draft_token_ids", None)
    if req_ids is None and rows is None:
        return "n=none", ""
    if req_ids is None:
        req_ids = []
    if rows is None:
        rows = []
    n = min(len(req_ids), len(rows))
    widths = [len(list(r)) if r is not None else 0 for r in rows[:n]]
    rid0 = req_ids[0] if req_ids else "-"
    widths_s = "[" + ",".join(str(w) for w in widths) + "]"
    return f"n={n} widths={widths_s}", str(rid0)


def _emit(draft_token_ids, counter: list) -> None:
    # Silence-must-be-loud: even a None/absent payload logs n=none.
    desc, rid0 = _describe(draft_token_ids)
    if counter[0] >= _trace_max():
        if counter[0] == _trace_max():
            print("suffix_hybrid SCHEDTRACE limit reached; going quiet",
                  file=sys.stderr, flush=True)
            counter[0] += 1
        return
    print(f"suffix_hybrid SCHEDTRACE {desc} rid0={rid0}",
          file=sys.stderr, flush=True)
    counter[0] += 1


def install(engine_core_module=None) -> bool:
    """Wrap ``Scheduler.update_draft_token_ids`` at class level (once).

    ``engine_core_module`` is accepted for API symmetry but unused: the
    Scheduler class is imported directly, which is robust in the
    EngineCore worker process where the plugin installs via the
    load_model hook (uniproc executor: same process).
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"suffix_hybrid SCHEDTRACE install failed: {exc}",
              file=sys.stderr, flush=True)
        return False

    original = getattr(Scheduler, "update_draft_token_ids", None)
    if original is None:
        print("suffix_hybrid SCHEDTRACE install failed: "
              "Scheduler.update_draft_token_ids not found",
              file=sys.stderr, flush=True)
        return False

    if getattr(original, GUARD_ATTR, False):
        return True  # already wrapped

    counter = [0]

    @functools.wraps(original)
    def wrapped(self, draft_token_ids, *args, **kwargs):
        try:
            _emit(draft_token_ids, counter)
        except Exception as exc:
            print(f"suffix_hybrid SCHEDTRACE emit failed: {exc}",
                  file=sys.stderr, flush=True)
        return original(self, draft_token_ids, *args, **kwargs)

    setattr(wrapped, GUARD_ATTR, True)
    Scheduler.update_draft_token_ids = wrapped
    print(f"suffix_hybrid SCHEDTRACE installed on "
          f"{Scheduler.__module__}.Scheduler.update_draft_token_ids "
          f"max={_trace_max()}",
          file=sys.stderr, flush=True)
    return True