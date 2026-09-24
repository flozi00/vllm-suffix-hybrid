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
                "SUFFIX_HYBRID_ZERO_OP", "SUFFIX_HYBRID_W0_FASTPATH"):
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


# ---------------------------------------------------------------------------
# (g) W0 fast path: bit-for-bit publish equivalence across arms
# ---------------------------------------------------------------------------

def _arm_env(monkeypatch, fast, trace=None, interval=None):
    """Env for one arm: W0_FASTPATH on/off (+ optional MTRACE knobs)."""
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "1" if fast else "0")
    if trace is not None:
        monkeypatch.setenv("SUFFIX_HYBRID_TRACE", trace)
    if interval is not None:
        monkeypatch.setenv("SUFFIX_HYBRID_LOG_INTERVAL", str(interval))


def _w0_runner():
    """Runner for the W0 scenarios: 'a' (12 tokens) row 2, 'b' (unrelated,
    6 tokens) row 3, replay request 'c' (SEQ_A[:10]) at row 1."""
    runner, ats, totals = make_runner()
    ats[1, :] = 0
    ats[1, :10] = np.array(SEQ_A[:10], dtype=np.int32)
    totals.t[1] = 10
    runner.req_states.req_id_to_index["c"] = 1
    return runner, ats, totals


def _w0_scenario(wrapped, runner):
    """Live/depart/replay script exercising stock steps, a fast step, a
    departure and a repeat hit. Returns (outputs, widths snapshots)."""
    outs, widths = [], []
    for ids in (["a", "b"], ["a"], ["a"], [], ["c"], ["c"]):
        o = propose(wrapped, runner, ids)
        outs.append(o.tolist())
        widths.append(dict(wrapped._suffix_proposer.widths_table))
    return outs, widths


def test_w0_fastpath_default_on_and_fires(suffix_env, monkeypatch):
    # No env set at all: the arm must be armed by default (gate value 0
    # disables), and the cache-empty steady steps must take the fast path.
    monkeypatch.delenv("SUFFIX_HYBRID_W0_FASTPATH", raising=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = wrap(runner)
    assert wrapped._suffix_w0fp["enabled"] is True
    _w0_scenario(wrapped, runner)
    assert wrapped._suffix_w0fp["fast_steps"] >= 1
    # The published [n,K] buffer stays correctly shaped.
    assert spec.draft_tokens.shape == (NROWS, K)


def test_w0_fastpath_off_reproduces_stock_widths_bit_for_bit(
        suffix_env, monkeypatch):
    # FASTPATH=0 is the stock arm (current behavior, pinned by the tests
    # above): step-separated widths snapshots must equal the exact
    # expected ragged publish -- miss/miss/miss/depart/hit/hit.
    _arm_env(monkeypatch, fast=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = wrap(runner)
    outs, widths = _w0_scenario(wrapped, runner)
    assert wrapped._suffix_w0fp["fast_steps"] == 0
    # Widths semantics, step by step (stock arm):
    assert widths[0] == {"a": 0, "b": 0}     # cold: both rows miss
    assert widths[1] == {"a": 0}             # 'b' departed: width entry gone
    assert widths[2] == {"a": 0}
    assert widths[3] == {}                   # 'a' departed (ingested)
    assert widths[4] == {"c": 2}              # repeat hit
    assert widths[5] == {"c": 2}
    # And the drafted continuation is the true suffix: SEQ_A[10:12] + pad
    assert outs[4] == [[SEQ_A[10], SEQ_A[11], 0, 0]]


def test_w0_fastpath_on_publishes_identical_widths_vs_slow(
        suffix_env, monkeypatch):
    # The fast arm must publish BIT-FOR-BIT the same draft rows and the
    # same widths-table snapshots on every step as the stock arm.
    results = {}
    for arm, fast in (("slow", False), ("fast", True)):
        _arm_env(monkeypatch, fast=fast)
        runner, ats, totals = _w0_runner()
        wrapped, spec = wrap(runner)
        results[arm] = (wrapped,) + _w0_scenario(wrapped, runner)
    slow_w, slow_outs, slow_widths = results["slow"]
    fast_w, fast_outs, fast_widths = results["fast"]
    assert fast_w._suffix_w0fp["fast_steps"] >= 1   # fast path actually fired
    assert fast_outs == slow_outs                   # drafts bit-for-bit
    assert fast_widths == slow_widths               # widths-table publication
    # Proposal steps: it is fine for the two arms to differ (fast steps skip
    # the Rust call); ingestion must not:
    assert (fast_w._suffix_proposer.get_stats()["ingested"]
            == slow_w._suffix_proposer.get_stats()["ingested"])


# ---------------------------------------------------------------------------
# (h) ingestion still happens when the fast path skips
# ---------------------------------------------------------------------------

def _growth_scenario(wrapped, runner, ats, totals):
    """'a' grows 9 -> 12 tokens WHILE living through a fast step, then
    departs (12 > 8 => ingestible); 'c' repeats its 10-token prefix."""
    # step 1: cold cache, 'a' unknown to the Rust state -> stock call,
    # mirror tracks 9 tokens.
    propose(wrapped, runner, ["a"])
    # step 2: 'a' appended 3 more tokens (engine wrote 9..11). Cache is
    # still empty and 'a' is known -> fast step: the Rust call is skipped
    # and its mirror goes STALE at 9 tokens.
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])
    # step 3: 'a' has departed; an empty batch forces the departure step.
    propose(wrapped, runner, [])
    # step 4: 'c' repeats 'a's 10-token prefix -> must hit the ingested
    # history and draft the true continuation.
    return propose(wrapped, runner, ["c"]).tolist()


def test_ingestion_survives_fastpath_skips(suffix_env, monkeypatch):
    # HARD GATE: the fast step (step 2) skips the Rust call, so 'a''s
    # mirror is stale; the adapter must re-feed the missing prefix at
    # departure so the gone-ingestion sees the FULL 12-token history
    # (byte-identical to what the stock arm ingests) and the repeat
    # request still hits.
    _arm_env(monkeypatch, fast=True)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9                      # 'a' starts with only 9 tokens
    wrapped, spec = wrap(runner)
    fp = wrapped._suffix_w0fp
    out = _growth_scenario(wrapped, runner, ats, totals)
    assert fp["fast_steps"] >= 1         # the skip actually happened
    assert fp["ghosts_fed"] == 1         # departure re-fed the prefix
    st = wrapped._suffix_proposer.get_stats()
    assert st["ingested"] == 1           # full history ingested at departure
    # IDENTICAL published draft to the stock arm on the same script:
    _arm_env(monkeypatch, fast=False)
    runner2, ats2, totals2 = _w0_runner()
    ats2[2, 9:] = 668
    totals2.t[2] = 9
    wrapped2, spec2 = wrap(runner2)
    out2 = _growth_scenario(wrapped2, runner2, ats2, totals2)
    assert out == out2 == [[SEQ_A[10], SEQ_A[11], 0, 0]]
    assert (wrapped2._suffix_proposer.get_stats()["ingested"]
            == st["ingested"])
    assert wrapped2._suffix_w0fp["ghosts_fed"] == 0   # stock needs no ghost


# ---------------------------------------------------------------------------
# (i) arm-M instrumentation: histogram accumulates + trace line at interval
# ---------------------------------------------------------------------------

def test_mtrace_histogram_accumulates_and_fires_at_interval(
        suffix_env, monkeypatch, capsys):
    # SUFFIX_HYBRID_TRACE arms the arm-M histogram (the existing trace
    # env gate); SUFFIX_HYBRID_LOG_INTERVAL is its line cadence. After N
    # steps the MTRACE line must carry the step / width-0 / batch-size
    # counters and the split host-time histogram (t_d2h fence-inclusive,
    # t_rust, t_upload) in microseconds -- the dossier decision-table
    # discriminators -- all non-negative and step-accurate.
    _arm_env(monkeypatch, fast=True, trace="1", interval=1)
    runner, ats, totals = _w0_runner()
    wrapped, spec = wrap(runner)
    _w0_scenario(wrapped, runner)
    m = wrapped._suffix_mtrace
    # Histogram counters saw every step (6 steps in the script; the fast
    # step is counted too) with the split components populated.
    assert m["steps"] == 6
    assert m["steps_w0"] >= 1
    assert m["batch"] == 6   # per-step scheduled batch size: 2+1+1+0+1+1
    assert m["totals_us"] >= 0.0 and m["rust_us"] > 0.0
    assert m["upload_us"] >= 0.0
    assert m["fast_steps"] >= 1
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if ln.startswith("suffix_hybrid MTRACE ")]
    assert lines, "MTRACE line must fire at LOG_INTERVAL cadence"
    import json as _json
    payload = _json.loads(lines[-1][len("suffix_hybrid MTRACE "):])
    for key in ("steps", "steps_w0", "batch_tokens", "totals_us",
                "rust_us", "upload_us"):
        assert key in payload
    # The final line reports the CURRENT histogram totals (interval=1).
    assert payload["steps"] == m["steps"]
    assert payload["steps_w0"] == m["steps_w0"]
    assert payload["batch_tokens"] == m["batch"]


# ---------------------------------------------------------------------------
# (j) NIT-1: ghost records survive the exception path (no silent loss)
# ---------------------------------------------------------------------------

def _growth_setup(monkeypatch):
    """Arm the fast path and drive 'a' through a stale-making fast step.

    Returns everything needed to then depart 'a' as a dirty ghost."""
    _arm_env(monkeypatch, fast=True)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = wrap(runner)
    propose(wrapped, runner, ["a"])        # cold: stock, mirror tracks 9
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])        # fast step: mirror goes stale
    fp = wrapped._suffix_w0fp
    assert fp["fast_steps"] >= 1
    assert fp["seen"]["a"]["dirty"] is True
    return wrapped, runner, ats, totals, fp


def test_ghost_fingerprint_mismatch_counts_drop(suffix_env, monkeypatch):
    # The engine reuses the departed row before the re-feed: the
    # head4/tail1 fingerprint no longer matches, no provable recovery
    # exists. The adapter must count it EXPLICITLY (ghosts_dropped) --
    # never silently -- and the ledger entry must be gone.
    wrapped, runner, ats, totals, fp = _growth_setup(monkeypatch)
    # Corrupt the row's head: fingerprint check at departure must fail.
    ats[2, :4] = 999
    propose(wrapped, runner, [])            # 'a' departs -> ghost drop
    # Explicitly counted, never silent; ledger entry consumed either way.
    assert fp["ghosts_dropped"] == 1
    assert fp["ghosts_fed"] == 0
    assert fp["ghosts_pending"] == 0
    assert "a" not in fp["seen"]


def test_ghost_exception_restores_and_retries(suffix_env, monkeypatch):
    # An exception DURING the ghost re-feed must not lose the ledger
    # record: the except block re-inserts it and counts ghosts_pending;
    # the next step re-attempts the re-feed and the gone-ingestion of
    # the full 12-token history eventually happens (repeat request 'c'
    # still hits). The pyo3 proposer's attributes are read-only, so the
    # throw is injected via a poisoning head whose __eq__ raises at the
    # fingerprint check -- the record is already popped at that point,
    # exactly the loss window NIT-1 closes.
    wrapped, runner, ats, totals, fp = _growth_setup(monkeypatch)
    ingested_before = wrapped._suffix_proposer.get_stats()["ingested"]

    class PoisonHead(list):
        def __eq__(self, other):
            raise RuntimeError("boom during ghost re-feed")

        __hash__ = None

    fp["seen"]["a"]["head"] = PoisonHead()
    out = propose(wrapped, runner, [])     # departure: fingerprint throws
    assert fp["ghosts_fed"] == 0
    assert fp["ghosts_pending"] == 1
    assert fp["seen"]["a"]["dirty"] is True
    # Restore a sane fingerprint: the next departure step retries the
    # re-feed and it succeeds now.
    fp["seen"]["a"]["head"] = ats[2, :4].tolist()
    out = propose(wrapped, runner, [])
    assert fp["ghosts_fed"] == 1
    assert "a" not in fp["seen"]
    ingested = wrapped._suffix_proposer.get_stats()["ingested"]
    assert ingested == ingested_before + 1
    # Gone-ingestion saw the FULL 12-token history: the repeat request
    # 'c' hits and drafts the true continuation.
    out = propose(wrapped, runner, ["c"]).tolist()
    assert out == [[SEQ_A[10], SEQ_A[11], 0, 0]]


def test_two_ghosts_departed_same_step(suffix_env, monkeypatch):
    # Two departed rids in ONE step: both dirty (each lived through a
    # fast step), both re-fed, both gone-ingested; a replay of either
    # prefix must hit. Uses the real Rust V2SuffixProposer.
    _arm_env(monkeypatch, fast=True)
    runner, ats, totals = make_runner()
    # 'b' gets a unique 9-token prefix that later grows to 12.
    seq_b = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312]
    ats[3, :] = 668
    ats[3, :9] = np.array(seq_b[:9], dtype=np.int32)
    totals.t[3] = 9
    wrapped, spec = wrap(runner)
    propose(wrapped, runner, ["a", "b"])   # cold: stock, mirrors track 9
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    ats[3, 9:12] = np.array(seq_b[9:12], dtype=np.int32)
    totals.t[3] = 12
    propose(wrapped, runner, ["a", "b"])   # fast step: both mirrors stale
    fp = wrapped._suffix_w0fp
    assert fp["fast_steps"] >= 1
    assert fp["seen"]["a"]["dirty"] and fp["seen"]["b"]["dirty"]
    propose(wrapped, runner, [])           # both depart in one step
    assert fp["ghosts_fed"] == 2
    assert "a" not in fp["seen"] and "b" not in fp["seen"]
    # Ingestion accounting: each re-feed rebuilds a mirror, and a
    # re-feed call ALSO gone-ingests previously-rebuilt ghosts (departed
    # rows) -- so ingested is >= 2, exact count an internal detail.
    assert wrapped._suffix_proposer.get_stats()["ingested"] >= 2
    # Replays of either prefix hit and draft the true continuations
    # (10-token replay prefix of a 12-token ingested history).
    runner.req_states.req_id_to_index["ra"] = 2
    ats[2, :10] = np.array(SEQ_A[:10], dtype=np.int32)
    totals.t[2] = 10
    assert propose(wrapped, runner, ["ra"]).tolist() \
        == [[SEQ_A[10], SEQ_A[11], 0, 0]]
    runner.req_states.req_id_to_index["rb"] = 3
    ats[3, :10] = np.array(seq_b[:10], dtype=np.int32)
    totals.t[3] = 10
    assert propose(wrapped, runner, ["rb"]).tolist() \
        == [[seq_b[10], seq_b[11], 0, 0]]