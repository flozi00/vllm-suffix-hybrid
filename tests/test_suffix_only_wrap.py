# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the SUFFIX_HYBRID_SUFFIX_ONLY=1 propose path.

Audit (2026-09-24): _suffix_only_wrap's propose path, _totals_now and
dummy-run short-circuit were previously exercised ONLY on the live pod.
The fake runner here mirrors vLLM's req-state layout: live requests sit
at HIGH row indices (states.py add_request pops rows from the END of the
free list, so a fresh engine hands out rows max_reqs-1, max_reqs-2, ...)
and rows 0..1 hold free-list garbage — exactly the layout that made the
old positional totals_gpu[:n] slice read garbage and freeze the mirror
(fixed in 692fca96 by the indexed gather). The Rust V2SuffixProposer is
real; only CUDA buffer access is faked with CPU tensors/numpy.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from suffix_hybrid import wrap_v2
from suffix_hybrid._native import HybridMixer

K = 4            # draft width
NROWS = 4        # rows 0,1 = free-list garbage; rows 2,3 = live ('a','b')
NCOLS = 64       # token buffer width (row capacity)

SEQ_A = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]


@pytest.fixture
def suffix_env(monkeypatch):
    """Isolate the wrapper's env knobs from ambient/CI env."""
    for var in ("SUFFIX_HYBRID_SUFFIX_ONLY", "SUFFIX_HYBRID_SYNC_HOOK",
                "SUFFIX_HYBRID_TP_MODE", "SUFFIX_HYBRID_SUFFIX_MIN",
                "SUFFIX_HYBRID_UNIFORM_K", "SUFFIX_HYBRID_TRACE",
                "SUFFIX_HYBRID_TRACE_VERIFY", "SUFFIX_HYBRID_TRACE_VERIFY2",
                "SUFFIX_HYBRID_SCHEDTRACE", "SUFFIX_HYBRID_LOG_INTERVAL",
                "SUFFIX_HYBRID_ZERO_OP"):
        monkeypatch.delenv(var, raising=False)


class FakeTotals:
    """Stand-in for req_states.total_len.gpu ([max_reqs] int64 tensor).

    Records every __getitem__ so the tests can pin that the gather is
    INDEXED (rows [2]/[3] from req_id_to_index) and never the positional
    [:n] slice that read free-list garbage (wake-#1/#2, fixed 692fca96).
    """

    def __init__(self, totals):
        self.t = torch.tensor(list(totals), dtype=torch.int64)
        self.gathers = []      # (index_used, gathered_values)

    def __getitem__(self, idx):
        g = self.t[idx]
        try:
            self.gathers.append((idx.tolist(), g.tolist()))
        except (AttributeError, TypeError):
            self.gathers.append((None, g.tolist()))
        return g


class Speculator:
    """Fake speculator exposing exactly what _suffix_only_wrap reads."""

    def propose(self, *args, **kwargs):   # signature target for wraps()
        raise AssertionError("suffix-only replaces propose; never called")

    def __init__(self, k=K, max_reqs=NROWS, max_model_len=None):
        self.draft_tokens = torch.zeros((max_reqs, k), dtype=torch.int64)
        self.max_num_reqs = max_reqs
        self.max_model_len = max_model_len   # None -> wrapper uses 32768
        self.num_speculative_steps = k
        self.draft_logits = None            # greedy contract


class TP:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, value, src=0):
        return value


def make_runner(nrows=NROWS, ncols=NCOLS, total_a=None, row_b=None,
                total_b=None):
    """Fake runner with the end-of-freelist row layout.

    Rows 0,1: garbage (junk tokens, totals 3). 'a' lives at row nrows-2
    with SEQ_A as its passage; 'b' lives at row nrows-1. row_b/total_b
    default to an unrelated 6-token row (always a cache miss).
    """
    ats = np.zeros((nrows, ncols), dtype=np.int32)
    ats[0, :] = 666          # free-list garbage; must never be observed
    ats[1, :] = 667
    ats[nrows - 2, :len(SEQ_A)] = np.array(SEQ_A, dtype=np.int32)
    if ncols >= len(SEQ_A):
        ats[nrows - 2, len(SEQ_A):] = 668   # post-total junk, ignores
    if row_b is None:
        row_b = [201, 202, 203, 204, 205, 206]
    ats[nrows - 1, :len(row_b)] = np.array(row_b, dtype=np.int32)
    total_a = len(SEQ_A) if total_a is None else total_a
    total_b = len(row_b) if total_b is None else total_b
    totals = FakeTotals([3, 3, 0] * 0 + [3, 3, total_a, total_b])
    states = NS(total_len=NS(gpu=totals),
                all_token_ids=NS(_uva_buf=NS(np=ats)),
                req_id_to_index={"a": nrows - 2, "b": nrows - 1})
    return NS(req_states=states), ats, totals


def wrap(runner, k=K):
    """Wire up the suffix-only wrapper the way install_v2 does."""
    spec = Speculator(k=k)
    mixer = HybridMixer(k, NCOLS if NCOLS >= 256 else 512)
    wrapped = wrap_v2._suffix_only_wrap(runner, spec, mixer, TP(), k)
    wrapped._suffix_hybrid_hook = True
    return wrapped, spec


def propose(wrapped, runner, req_ids, **kw):
    """Invoke the wrapped propose with a fake input_batch of req_ids."""
    batch = NS(req_ids=list(req_ids), num_reqs=len(req_ids),
               idx_mapping=torch.zeros(len(req_ids), dtype=torch.long))
    return wrapped(input_batch=batch, attn_metadata={}, slot_mappings={},
                   last_hidden_states=torch.zeros(1), aux_hidden_states=None,
                   num_sampled=torch.ones(len(req_ids), dtype=torch.int64),
                   num_rejected=torch.zeros(len(req_ids), dtype=torch.int64),
                   last_sampled=torch.zeros(2),
                   next_prefill_tokens=torch.zeros(1, 2),
                   temperature=torch.tensor([0.8] * len(req_ids)),
                   seeds=torch.ones(2), **kw)


# ---------------------------------------------------------------------------
# (c) totals read is indexed, not positional — pins 692fca96
# ---------------------------------------------------------------------------

def test_totals_read_is_indexed_not_positional(suffix_env):
    # Live row at a HIGH index carries total 700; free-list row 0 carries
    # total 3. The old positional totals_gpu[:n] slice read row 0's 3 and
    # froze the mirror (wake-#1/#2). The gather must go through
    # req_id_to_index and yield 700.
    runner, ats, totals = make_runner(ncols=768, total_a=700)
    ats[2, :len(SEQ_A)] = np.array(SEQ_A, dtype=np.int32)  # 12 real tokens
    runner.req_states.req_id_to_index = {"a": 2, "b": 3}
    wrapped, spec = wrap(runner)
    out = propose(wrapped, runner, ["a"])
    assert out.shape == (1, K)
    assert out.tolist() == [[0] * K]        # cold cache: miss, zeros
    idx_used, values = totals.gathers[0]
    assert idx_used == [2]                  # indexed by req_id_to_index
    assert values == [700]                  # NOT row 0's garbage total 3
    # No positional/prefix gather ever touched rows 0..1.
    for idx_used_i, _ in totals.gathers:
        assert 0 not in idx_used_i and 1 not in idx_used_i


# ---------------------------------------------------------------------------
# (a)+(f) the D2H copy is blocking and gathers by index, never slices
# ---------------------------------------------------------------------------

def test_totals_now_blocking_and_indexed(suffix_env, monkeypatch):
    # _totals_now must (1) gather totals_gpu[idx] by index and (2) issue
    # the D2H copy with non_blocking=False — the blocking copy is also the
    # fence for the postprocess_sampled UVA writes; an optimistic copy
    # could race half-updated rows (same reasoning as the sync body).
    runner, ats, totals = make_runner(
        row_b=SEQ_A[:10], total_b=10)       # 'b' set up for later hits
    wrapped, spec = wrap(runner)

    copies = []
    orig_copy = torch.Tensor.copy_

    def spy(self, src, non_blocking=False):
        copies.append((tuple(self.shape), bool(non_blocking)))
        return orig_copy(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", spy)
    out = propose(wrapped, runner, ["a", "b"])
    assert out.shape == (2, K)
    # The totals D2H: dst is the 1-D [n] int64 staging row.
    assert (2,) in [shape for shape, _ in copies]
    tot_copies = [nb for shape, nb in copies if shape == (2,)]
    assert tot_copies == [False], "totals D2H must be non_blocking=False"
    # Gather indices came from the LUT rows [2, 3], in batch order.
    assert totals.gathers[0] == ([2, 3], [12, 10])


# ---------------------------------------------------------------------------
# (b) widths published via the accessor, not an install-time snapshot
# ---------------------------------------------------------------------------

def test_publishes_widths_via_accessor_not_snapshot(suffix_env):
    # wake-#13 root cause: install_v2 once bound the widths VALUE
    # (an empty dict captured at install time) into the
    # get_draft_tokens patch; the @getter builds a FRESH dict per access,
    # so a snapshot stays empty forever and ragged publication goes dead.
    # Pin: dicts read across two proposes differ, and earlier reads are
    # not mutated in place (accessor semantics, not shared state).
    runner, ats, totals = make_runner(row_b=SEQ_A[:10], total_b=10)
    wrapped, spec = wrap(runner)
    proposer = wrapped._suffix_proposer

    at_install = dict(proposer.widths_table)
    assert at_install == {}                  # nothing published yet

    propose(wrapped, runner, ["a"])          # 'a': cold miss, width 0
    read1 = dict(proposer.widths_table)
    assert read1 == {"a": 0}
    assert at_install == {}                  # install read NOT retro-filled

    # 'a' departs (mirror ingested on departure inside Rust), 'b' repeats
    # 'a's 10-token prefix -> a hit with a nonzero published width.
    out = propose(wrapped, runner, ["b"])
    read2 = dict(proposer.widths_table)
    assert read2["b"] == 2
    assert read2 != read1                   # post-propose state moved on
    assert read1 == {"a": 0}                 # fresh dict per access Proof
    # And the hit actually published the suffix continuation into the
    # persistent draft buffer: SEQ_A[10:12].
    assert out.tolist() == [[SEQ_A[10], SEQ_A[11], 0, 0]]


# ---------------------------------------------------------------------------
# (d) dummy/profile runs zero the buffer and short-circuit before any read
# ---------------------------------------------------------------------------

def test_dummy_run_zeroes_and_short_circuits(suffix_env):
    runner, ats, totals = make_runner()
    wrapped, spec = wrap(runner)
    proposer = wrapped._suffix_proposer
    spec.draft_tokens[0] = 777               # junk: dummy must overwrite

    for kw in ({"dummy_run": True}, {"is_profile": True}):
        out = propose(wrapped, runner, ["a"], **kw)
        assert out.shape == (1, K)
        assert out.tolist() == [[0] * K]     # zeroed, correctly shaped
    assert totals.gathers == []              # no totals read at all
    st = proposer.get_stats()
    assert st["steps"] == 0                  # Rust proposer never invoked
    assert dict(proposer.widths_table) == {}  # no widths published


# ---------------------------------------------------------------------------
# (e) an exception after widths were published must retract them
# ---------------------------------------------------------------------------

def test_exception_retracts_widths(suffix_env):
    # The except path in propose() zeroes the widths table via
    # clear_widths(): a failed step uploads zeroed drafts, and widths
    # left behind from the LAST successful step would make the scheduler
    # verify zeroed drafts at those phantom widths.
    runner, ats, totals = make_runner(row_b=SEQ_A[:10], total_b=10)
    wrapped, spec = wrap(runner)
    proposer = wrapped._suffix_proposer

    propose(wrapped, runner, ["a"])           # miss; widths {'a': 0}
    propose(wrapped, runner, ["b"])           # hit; widths {'b': 2}
    assert dict(proposer.widths_table) == {"b": 2}

    # Next step keeps 'b' live (so its width entry survives the gone-row
    # eviction inside Rust) but blows up adapter-side BEFORE the Rust
    # call: an unknown rid misses req_id_to_index -> KeyError -> except.
    out = propose(wrapped, runner, ["b", "c-unknown"])
    assert out.tolist() == [[0] * K, [0] * K, [0] * K][:2]
    # Widths retracted: the scheduler cannot consume phantom width 2 for
    # a row whose draft the step failed to upload.
    assert dict(proposer.widths_table) == {"b": 0, "c-unknown": 0} \
        or all(w == 0 for w in proposer.widths_table.values())
    st = proposer.get_stats()
    assert st["steps"] == 2                  # the failed step never ran