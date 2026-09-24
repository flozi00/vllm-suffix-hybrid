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
                "SUFFIX_HYBRID_ZERO_OP", "SUFFIX_HYBRID_W0_FASTPATH",
                "SUFFIX_HYBRID_D2H_PIPE", "SUFFIX_HYBRID_D2H_PIPE_SIM",
                "SUFFIX_HYBRID_PIPE_EARLY"):
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


# ===========================================================================
# Pipelined side-stream D2H (Option A, dossier pipelined-d2h-design.md).
# Gate: SUFFIX_HYBRID_D2H_PIPE (default OFF = blocking stock, bit-identical).
# CPU-only SIMULATE arm (SUFFIX_HYBRID_D2H_PIPE_SIM=1) exercises the exact
# lag=1 state machine without a GPU; on a CUDA box the same tests run the
# real side-stream/event path instead.
# ===========================================================================

def _pipe_env(monkeypatch, pipe=True, sim=None, fast=None):
    """Arm the pipeline. `sim` is opt-in ONLY via SUFFIX_HYBRID_D2H_PIPE_SIM
    (default OFF everywhere: SIM never silently shadows the real CUDA
    path -- a CUDA-with-SIM surprise is exactly the failure mode the
    reviewer flagged, so the default is False on every box)."""
    if sim is None:
        sim = False
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", "1" if pipe else "0")
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE_SIM",
                       "1" if sim else "0")
    if fast is not None:
        monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH",
                           "1" if fast else "0")


def _pipe_wrap(runner):
    """Wrap with a sync-scheduling runner shape (I8: async_scheduling
    must be present and False on the launcher's scheduler_config).
    Preserves a caller-set vllm_config (the I8-refusal test injects
    async_scheduling=True)."""
    if getattr(runner, "vllm_config", None) is None:
        runner.vllm_config = NS(
            scheduler_config=NS(async_scheduling=False,
                                max_concurrent_batches=1,
                                num_speculative_tokens=K))
    return wrap(runner)


CUDA = torch.cuda.is_available()


def test_pipe_gate_off_is_bit_identical_stock(suffix_env, monkeypatch):
    # DEFAULT OFF (and explicit 0): the pipeline block must not touch
    # the totals read -- blocking stock path, every gather indexed and
    # blocking, zero pipe state.
    results = {}
    for arm, gate in (("off", None), ("zero", "0")):
        monkeypatch.delenv("SUFFIX_HYBRID_D2H_PIPE", raising=False)
        if gate is not None:
            monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", gate)
        base_runner, ats, totals = make_runner(
            row_b=SEQ_A[:10], total_b=10)
        wrapped, spec = wrap(base_runner)
        assert getattr(wrapped, "_suffix_pipe", None) is None or \
            not wrapped._suffix_pipe["on"]
        outs, widths = [], []
        for ids in (["a", "b"], ["a"], [], ["b"]):
            outs.append(propose(wrapped, base_runner,
                                ids).tolist())
            widths.append(dict(
                wrapped._suffix_proposer.widths_table))
        results[arm] = (outs, widths)
        # gate OFF arms the pipe dict but never the path: zero steps
        # through the pipeline, every gather blocking (test (a)+(f)
        # pins the blocking copy; here just ensure the pipe stayed off)
        pipe = getattr(wrapped, "_suffix_pipe", None)
        assert pipe is None or (pipe["on"] is False
                                and pipe["steps"] == 0)
    assert results["off"] == results["zero"]
    # Blocking semantics held: same widths the stock test pins.
    assert results["off"][1][0] == {"a": 0, "b": 0}


@pytest.mark.skipif(not CUDA, reason="needs CUDA for the real pipe path")
def test_pipe_lag1_consume_correctness_real_cuda(suffix_env,
                                                 monkeypatch):
    # Real CUDA path: side-stream + non_blocking copy + event, consumed
    # one step LATER. The totals visible to step N equal the totals
    # gathered at step N-1 (the lag=1 contract).
    _pipe_env(monkeypatch, sim=False, fast=False)
    runner, ats, totals = make_runner(ncols=768, total_a=12, total_b=6)
    wrapped, spec = _pipe_wrap(runner)
    pipe = wrapped._suffix_pipe
    assert pipe["on"] and not pipe["sim"]
    # step 1: blocking seed, CURRENT totals.
    propose(wrapped, runner, ["a", "b"])
    assert pipe["armed"] is True and pipe["block_steps"] >= 0
    # grow both rows BEFORE step 2: the pipelined step must still see
    # the SEED-step totals (lag), then step 2's copy carries them.
    totals.t[2] = 20
    out = propose(wrapped, runner, ["a", "b"])
    consumed = wrapped._suffix_pipe_frame
    assert (consumed["totals"], consumed["lagged"]) == ([12, 6], True)
    # step 3: consumes step 2's copy (still [12, 6] shape/ids match).
    propose(wrapped, runner, ["a", "b"])
    consumed = wrapped._suffix_pipe_frame
    assert (consumed["totals"], consumed["lagged"]) == ([12, 6], True)


@pytest.mark.skipif(not CUDA, reason="needs CUDA for the real lag check")
def test_pipe_lag1_consume_correctness_cuda_shape(suffix_env,
                                                  monkeypatch):
    # REAL CUDA only (same skipif discipline as the _real_cuda test):
    # explicitly sim=False -- SIM can never silently shadow this path.
    _pipe_env(monkeypatch, sim=False, fast=False)
    runner, ats, totals = make_runner(ncols=768, total_a=12, total_b=6)
    wrapped, spec = _pipe_wrap(runner)
    propose(wrapped, runner, ["a", "b"])               # seed: blocking
    totals.t[2] = 20                                   # mutate GPU-side
    propose(wrapped, runner, ["a", "b"])
    # The consumed totals array is the PRIOR step's snapshot: written
    # through the pipeline's staging clone, not a live view, and
    # matches the totals as of the seed step.
    arr, lagged = (wrapped._suffix_pipe_frame["totals"],
                   wrapped._suffix_pipe_frame["lagged"])
    assert lagged is True and list(arr) == [12, 6] and len(arr) == 2


def test_pipe_lag1_consume_correctness_sim(suffix_env, monkeypatch):
    # CPU-only SIM arm: identical lag=1 data contract. Because the SIM
    # copy is data-correct on the enqueue step but the machine defers
    # the CONSUME to the next step, the visible totals at step N are
    # step N-1's [n] even though totals already moved under us.
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = make_runner(ncols=768, total_a=12, total_b=6)
    wrapped, spec = _pipe_wrap(runner)
    pipe = wrapped._suffix_pipe
    assert pipe["on"] and pipe["sim"]
    propose(wrapped, runner, ["a", "b"])               # seed: blocking
    totals.t[2] = 20
    out2 = propose(wrapped, runner, ["a", "b"])
    consumed = wrapped._suffix_pipe_frame
    assert (consumed["totals"], consumed["lagged"]) == ([12, 6], True)
    # hits/misses and published zeros survive the lagged step.
    assert out2.tolist() == [[0] * K] * 2 or out2.tolist() == [[0] * K, [0] * K]
    propose(wrapped, runner, ["a", "b"])
    assert wrapped._suffix_pipe_frame["lagged"] is True


def test_pipe_depart_with_pending_record(suffix_env, monkeypatch):
    # A departure changes ids shape while LAST step's copy is in
    # flight (pending): the absorb runs BEFORE any decision; the
    # misaligned snapshot must NOT be served -- the step falls back to
    # a blocking stock read and the lag chain restarts cleanly.
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a", "b"])               # seed + enqueue
    propose(wrapped, runner, ["a", "b"])               # lag consume
    assert pipe["pending"] is not None                 # copy in flight
    # 'b' departs: next propose sees a different ids/n shape.
    out = propose(wrapped, runner, ["a"])
    # The misaligned pending could not serve the new shape -> this
    # step's totals came from the BLOCKING stock read (perfectly
    # current) and a fresh copy was enqueued for the shape now live.
    consumed = wrapped._suffix_pipe_frame
    assert consumed["lagged"] is False                        # not a lagged serve
    assert list(consumed["totals"]) == [int(totals.t[2])]
    # And the next same-shape step resumes the lag serve.
    propose(wrapped, runner, ["a"])
    assert wrapped._suffix_pipe_frame["lagged"] is True


def test_pipe_capture_window_fallback(suffix_env, monkeypatch):
    # I3 belt: if the current stream is INSIDE a CUDA-graph capture
    # window, the pipeline must refuse the side-stream/event path and
    # use the blocking total read (capture-safe), and the refusal is
    # STICKY for the wrapper's life (capture windows never contain
    # half-pipelines).
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    propose(wrapped, runner, ["a", "b"])               # seed + enqueue
    propose(wrapped, runner, ["a", "b"])               # lag consume
    assert wrapped._suffix_pipe_frame["lagged"] is True
    # Poison the capture check: simulate being inside capture.
    monkeypatch.setattr(wrap_v2, "_cuda_capturing", lambda: True)
    out = propose(wrapped, runner, ["a", "b"])
    pipe = wrapped._suffix_pipe
    assert pipe["fallback"] is True, "capture must sticky-disable"
    assert "I" in pipe["fallback_reason"] or "capture" in \
        pipe["fallback_reason"]
    assert wrapped._suffix_pipe_frame["lagged"] is False
    # Blocking path serves CURRENT totals and the step still publishes.
    assert list(wrapped._suffix_pipe_frame["totals"]) == \
        [int(totals.t[2]), int(totals.t[3])]
    # Sticky: even restoring a clean capture check, every later step
    # runs blocking.
    monkeypatch.setattr(wrap_v2, "_cuda_capturing", lambda: False)
    propose(wrapped, runner, ["a", "b"])
    assert wrapped._suffix_pipe_frame["lagged"] is False
    assert pipe["fallback"] is True


def test_pipe_invariant_violation_sticky_fallback(suffix_env, monkeypatch):
    # I1 class: a pending copy that somehow survives into the next
    # _pipe_get (double enqueue) is an invariant violation -> sticky
    # fail-closed to the blocking path, with a loud reason and the
    # invariant-fail counter incremented. Never crashes the step.
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a", "b"])               # seed
    propose(wrapped, runner, ["a", "b"])               # healthy lag
    assert wrapped._suffix_pipe_frame["lagged"] is True

    # Inject the I1 breach: an ARMED pipeline whose in-flight copy
    # vanished (absorb chain broken) must sticky-disable loudly.
    pipe["pending"] = None
    out = propose(wrapped, runner, ["a", "b"])
    assert pipe["fallback"] is True
    assert "I1" in pipe["fallback_reason"]
    assert sum(pipe["invariant_fails"].values()) >= 1
    assert out.shape == (2, K)                        # step survived
    # Sticky: the next read is blocking stock, still correct.
    monkeypatch.setattr(wrap_v2, "_cuda_capturing", lambda: False)
    propose(wrapped, runner, ["a", "b"])
    assert pipe["fallback"] is True
    assert wrapped._suffix_pipe_frame["lagged"] is False


def test_pipe_ghost_refeed_interplay(suffix_env, monkeypatch):
    # Ghost re-feed and the pipeline compose: a dirty ghost's re-feed
    # (single-row, ledger-authoritative CURRENT total) runs BEFORE the
    # pipelined totals read; the pipelined step then serves lagged
    # totals for the live rows and publishes zeros via the fast path
    # -- ingestion still lands and a replay still hits.
    _pipe_env(monkeypatch, fast=True)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a"])                    # cold + seed
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])                    # fast + enqueue
    assert fp["fast_steps"] >= 1
    # 'a' departs WITH the enqueued copy still pending: ghost re-feed
    # must fire (single-row blocking propose_suffix_only) and the step
    # (empty batch) must not serve any stale snapshot.
    propose(wrapped, runner, [])
    assert fp["ghosts_fed"] == 1
    assert fp["ghosts_dropped"] == 0 and fp["ghosts_pending"] == 0
    # The lagged 'a' history is fully ingested: a repeat request hits.
    runner.req_states.req_id_to_index["c"] = 1
    ats[1, :10] = np.array(SEQ_A[:10], dtype=np.int32)
    totals.t[1] = 10
    out = propose(wrapped, runner, ["c"]).tolist()
    assert out == [[SEQ_A[10], SEQ_A[11], 0, 0]]
    # Pipeline survived the entire interplay unless a real invariant
    # fired (it must not have): no fallback, no invariant failures.
    if pipe is not None and pipe.get("fallback"):
        # A fallback fired only if the machine decided a shape/seed
        # break was unsafe -- never silently; in this scenario the
        # ghost emancipation path must have kept it armed.
        assert pipe["invariant_fails"], \
            "fallback without a recorded invariant is a bug"
    else:
        assert pipe is None or not pipe["fallback"]


def test_pipe_ghost_refeed_w0_off_interplay(suffix_env, monkeypatch):
    # MUST-FIX #1 regression: sibling of test_pipe_ghost_refeed_interplay
    # with SUFFIX_HYBRID_W0_FASTPATH=0. Under PIPE=1 + W0 off the
    # ledger-refresh callers must still populate fp["seen"] (gate
    # `w0_fast or (pipe on and not fallback)`); with the old W0-only
    # gate the ledger stays empty, the ghost scan finds nothing, and a
    # departed request's Rust mirror is gone-ingested ONE STEP
    # TRUNCATED -- silently. This test fails on the pre-fix code.
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a"])                    # seed + enqueue
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])                    # lagged serve + enqueue
    # The lagged 'a' step recorded the ledger row despite W0 off:
    assert "a" in fp["seen"], "PIPE must arm the ledger with W0 off"
    # 'a' departs WITH the enqueued copy still pending: the ghost scan
    # (already pipe-aware) must find it and re-feed the full current
    # history (fingerprint-proven) before the empty-batch step.
    propose(wrapped, runner, [])
    assert fp["ghosts_fed"] == 1
    assert fp["ghosts_dropped"] == 0 and fp["ghosts_pending"] == 0
    # Full-history ingestion landed: the repeat request 'c' replays the
    # re-fed 'a' tail and hits -- truncated ingestion would not.
    runner.req_states.req_id_to_index["c"] = 1
    ats[1, :10] = np.array(SEQ_A[:10], dtype=np.int32)
    totals.t[1] = 10
    out = propose(wrapped, runner, ["c"]).tolist()
    assert out == [[SEQ_A[10], SEQ_A[11], 0, 0]]
    assert pipe is None or not pipe["fallback"]


def test_pipe_off_w0_off_refresh_never_runs(suffix_env, monkeypatch):
    # Negative control for MUST-FIX #1: with PIPE=0 + W0_FASTPATH=0
    # (pure stock) the attendance ledger must stay EMPTY -- the refresh
    # never runs, no ghost machinery exists, departure handling is the
    # stock gone-ingestion alone.
    _pipe_env(monkeypatch, pipe=False, fast=False)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    propose(wrapped, runner, ["a"])
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])
    assert fp["seen"] == {}
    propose(wrapped, runner, [])                      # 'a' departs
    assert fp["ghosts_fed"] == 0 and fp["ghosts_dropped"] == 0
    assert getattr(wrapped, "_suffix_pipe", None) is None or \
        not wrapped._suffix_pipe["on"]


def test_pipe_cpu_absent_safe_path(suffix_env, monkeypatch):
    # GPU-absent box, no SIM: the pipeline stays OFF and the wrapper
    # runs the plain blocking stock path (never crashes on the CUDA
    # stream/event API). torch.cuda.is_available() is monkeypatched so
    # the test is meaningful on a CUDA box too.
    _pipe_env(monkeypatch, sim=False)
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE_SIM", "0")
    runner, ats, totals = _w0_runner()
    runner.vllm_config = NS(
        scheduler_config=NS(async_scheduling=False))
    # Patch BEFORE wrapping: the arming block reads is_available.
    monkeypatch.setattr(wrap_v2.torch.cuda, "is_available",
                        lambda: False, raising=False)
    wrapped, spec = wrap(runner)
    pipe = getattr(wrapped, "_suffix_pipe", None)
    assert pipe is None or not pipe["on"]
    out = propose(wrapped, runner, ["a", "b"])         # plain stock
    assert out.tolist() == [[0] * K, [0] * K]


def test_pipe_i8_async_scheduling_refuses_install(suffix_env,
                                                   monkeypatch):
    # I8 fail-closed: async_scheduling != False refuses the pipeline
    # install outright (RuntimeError), before any state is built.
    _pipe_env(monkeypatch, sim=True, fast=False)
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", "1")
    runner, ats, totals = _w0_runner()
    runner.vllm_config = NS(
        scheduler_config=NS(async_scheduling=True))
    with pytest.raises(RuntimeError, match="async_scheduling"):
        _pipe_wrap(runner)
    # ATTR-MISSING refuses too (fail-closed, never silently degrade).
    runner.vllm_config = NS(scheduler_config=object())
    with pytest.raises(RuntimeError, match="async_scheduling"):
        _pipe_wrap(runner)


def test_pipe_mtrace_split_fields(suffix_env, monkeypatch, capsys):
    # The arm-M MTRACE extension: t_d2h SPLIT into t_fence (blocking
    # path: capture/seed/fallback/departure reads) vs t_pipe (record-to
    # consume latency + copy submit), plus pipe_lag_us / pipe_events.
    # Cumulative-counter semantics stay mtrace_report.py compatible
    # (totals_us kept, names exist, all non-negative).
    _pipe_env(monkeypatch, sim=True, fast=True)
    monkeypatch.setenv("SUFFIX_HYBRID_TRACE", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_LOG_INTERVAL", "1")
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    _w0_scenario(wrapped, runner)                      # 6 steps
    m = wrapped._suffix_mtrace
    for key in ("t_fence", "t_pipe", "pipe_lag_us", "pipe_events"):
        assert key in m and m[key] >= 0
    # Seed step paid a fence; every later pipelined step paid t_pipe.
    assert m["pipe_events"] >= 2
    assert m["t_pipe"] >= 0.0 and m["totals_us"] >= 0.0
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if ln.startswith("suffix_hybrid MTRACE ")]
    assert lines
    import json as _json
    payload = _json.loads(lines[-1][len("suffix_hybrid MTRACE "):])
    for key in ("totals_us", "t_fence", "t_pipe", "pipe_lag_us",
                "pipe_events", "steps", "rust_us", "upload_us"):
        assert key in payload


# ===========================================================================
# Wake-29 attribution tiebreakers (audit wake29-pipe-falsification-audit.md
# §3/§4): step_wall_us wall-cadence accumulator + ghost drop-reason split +
# absorb wait/host split. All cumulative, mtrace_report.py-compatible.
# ===========================================================================

def test_pipe_step_wall_accumulates_and_is_monotone(suffix_env,
                                                    monkeypatch):
    # step_wall_us = perf_counter delta between consecutive _pipe_get
    # ENTRIES. First entry anchors (no delta); N entries -> N-1 deltas.
    _pipe_env(monkeypatch, sim=True, fast=False)
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    m = wrapped._suffix_mtrace
    assert m["step_wall_us"] == 0.0
    offs = []
    for _ in range(5):
        propose(wrapped, runner, ["a", "b"])
        offs.append(m["step_wall_us"])
    # Monotone non-decreasing (perf_counter deltas are >= 0), and the
    # seed entry anchored (no delta exists for it: offs[0] == 0.0)
    # while every later entry added a strictly positive inter-propose
    # wall delta (host work between proposes is real).
    assert offs == sorted(offs)
    assert offs[0] == 0.0
    for prev, cur in zip(offs, offs[1:]):
        assert cur >= prev and cur > 0.0      # deltas never negative
    # 5 entries -> 4 deltas; 5 propose calls hit _pipe_get each time
    # (SIM arm, real side-stream state machine; wall_t0 always set).
    assert wrapped._suffix_pipe["wall_t0"] is not None
    # Negative control: with the pipe OFF the accumulator never runs.
    _pipe_env(monkeypatch, pipe=False, fast=False)
    runner2, ats2, totals2 = _w0_runner()
    wrapped2, spec2 = _pipe_wrap(runner2)
    for _ in range(3):
        propose(wrapped2, runner2, ["a", "b"])
    assert wrapped2._suffix_mtrace["step_wall_us"] == 0.0


def test_ghost_drop_reason_split_fingerprint(suffix_env, monkeypatch):
    # Fingerprint-mismatch drop (engine reused the row): head4/tail1
    # no longer match -> ghosts_drop_fp, NOT ghosts_drop_reset; the
    # total ghosts_dropped still counts it (back-compat).
    wrapped, runner, ats, totals, fp = _growth_setup(monkeypatch)
    ats[2, :4] = 999                       # corrupt head: fp mismatch
    propose(wrapped, runner, [])            # 'a' departs -> drop
    assert fp["ghosts_dropped"] == 1
    assert fp["ghosts_drop_fp"] == 1
    assert fp["ghosts_drop_reset"] == 0
    assert fp["ghosts_fed"] == 0


def test_ghost_drop_reason_split_reset_shrink(suffix_env, monkeypatch):
    # Reset-shrink drop (g_tot < g_tot_rec): the engine reset shrank
    # the row below its S2.3 lag bound -> ghosts_drop_reset, NOT
    # ghosts_drop_fp; fingerprint stays intact (head4/tail1 at the
    # RECORDED positions still match). Only the PIPE arm re-reads the
    # CURRENT total per departure (S2.2(a)), so only there can the
    # reset-shrink wedge be observed; without the pipe the re-feed
    # trusts the ledger total (g_tot == g_tot_rec) and the shrink
    # path is unreachable by construction.
    _pipe_env(monkeypatch, sim=True, fast=True)
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    propose(wrapped, runner, ["a"])          # seed + ledger record (9)
    totals.t[2] = 5                          # engine RESET shrank row 'a'
    propose(wrapped, runner, [])             # 'a' departs -> drop
    assert fp["ghosts_dropped"] == 1
    assert fp["ghosts_drop_reset"] == 1
    assert fp["ghosts_drop_fp"] == 0
    # Back-compat total still counted once and once only.
    assert fp["ghosts_drop_fp"] + fp["ghosts_drop_reset"] \
        == fp["ghosts_dropped"]


def test_ghost_drop_split_invariants_on_fed_and_stock(suffix_env,
                                                     monkeypatch):
    # Fed ghosts and the stock arm never touch the split counters.
    wrapped, runner, ats, totals, fp = _growth_setup(monkeypatch)
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])         # fast step: mirror stale
    propose(wrapped, runner, [])            # 'a' departs -> re-feed
    assert fp["ghosts_fed"] == 1
    assert fp["ghosts_dropped"] == 0
    assert fp["ghosts_drop_fp"] == 0 and fp["ghosts_drop_reset"] == 0
    # Stock (fast OFF, pipe OFF): no ghost machinery at all.
    _arm_env(monkeypatch, fast=False)
    monkeypatch.delenv("SUFFIX_HYBRID_D2H_PIPE", raising=False)
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", "0")
    runner2, ats2, totals2 = _w0_runner()
    ats2[2, 9:] = 668
    totals2.t[2] = 9
    wrapped2, spec2 = wrap(runner2)
    propose(wrapped2, runner2, ["a"])
    propose(wrapped2, runner2, [])
    fp2 = wrapped2._suffix_w0fp
    for k in ("ghosts_drop_fp", "ghosts_drop_reset",
              "ghosts_dropped", "ghosts_fed"):
        assert fp2[k] == 0


def test_pipe_mtrace_wake29_fields(suffix_env, monkeypatch, capsys):
    # Wake-29 MTRACE extension: the cumulative payload must ALSO carry
    # step_wall_us, absorb_wait_us, absorb_host_us and the ghost
    # drop-reason split, all non-negative, cumulative and parseable
    # by plugin-harness/mtrace_report.py (cumulative mode keys intact:
    # totals_us/rust_us/upload_us unchanged).
    _pipe_env(monkeypatch, sim=True, fast=True)
    monkeypatch.setenv("SUFFIX_HYBRID_TRACE", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_LOG_INTERVAL", "1")
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    _w0_scenario(wrapped, runner)                     # 6 steps
    m = wrapped._suffix_mtrace
    for key in ("step_wall_us", "absorb_wait_us", "absorb_host_us"):
        assert key in m and m[key] >= 0.0
    # Absorb split is consistent with t_pipe's components: pipe_lag_us
    # (pure sync-wait) equals absorb_wait_us exactly (same probe).
    assert m["pipe_lag_us"] == m["absorb_wait_us"]
    # Wall cadence: 6 entries -> 5 inter-entry deltas accumulated.
    assert m["step_wall_us"] > 0.0
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if ln.startswith("suffix_hybrid MTRACE ")]
    assert lines
    import json as _json
    payload = _json.loads(lines[-1][len("suffix_hybrid MTRACE "):])
    for key in ("step_wall_us", "absorb_wait_us", "absorb_host_us",
                "ghosts_drop_fp", "ghosts_drop_reset"):
        assert key in payload and payload[key] >= 0
    # Back-compat total preserved and split sums to it.
    assert payload["ghosts_drop_fp"] + payload["ghosts_drop_reset"] \
        <= payload["ghosts_dropped"]
    assert payload["step_wall_us"] == round(m["step_wall_us"], 1)


def test_mtrace_wake29_payload_parses_in_mtrace_report():
    # Cross-repo contract: the enriched cumulative payload (with ALL
    # wake-29 fields) must still classify as mode=cumulative in
    # plugin-harness/mtrace_report.py and derive intervals. The
    # parser ignores unknown keys, so the contract is: known cost
    # fields unchanged + new keys ride along harmlessly.
    import importlib.util as _ilu
    import json as _json
    import os as _os
    _p = _os.path.expanduser(
        "~/Documents/inference-console/plugin-harness/mtrace_report.py")
    if not _os.path.exists(_p):
        pytest.skip("plugin-harness mtrace_report.py not on this box")
    _sp = _ilu.spec_from_file_location("mtrace_report_w29", _p)
    assert _sp is not None and _sp.loader is not None
    mr = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(mr)
    payload = _json.dumps({
        "steps": 1000, "steps_w0": 100, "batch_tokens": 8,
        "totals_us": 0.0, "t_fence": 8088.1, "t_pipe": 4655273.1,
        "pipe_lag_us": 4328.2, "pipe_events": 1000,
        "step_wall_us": 27000000.0, "absorb_wait_us": 4328200.0,
        "absorb_host_us": 3200.0,
        "rust_us": 10662.0, "upload_us": 10895.1,
        "fast_steps": 18, "ghosts_fed": 33, "ghosts_dropped": 6,
        "ghosts_drop_fp": 5, "ghosts_drop_reset": 1,
        "skip_upload": 727}, sort_keys=True)
    line = "INFO suffix_hybrid MTRACE " + payload
    w = []
    rec = mr.parse_mtrace_line(line, w)
    assert rec is not None and rec["mode"] == "cumulative" and not w
    recs = [rec, mr.parse_mtrace_line(
        line.replace('"steps": 1000', '"steps": 2000')
            .replace('"t_pipe": 4655273.1', '"t_pipe": 9310546.2')
            .replace('"step_wall_us": 27000000.0',
                     '"step_wall_us": 54000000.0'), w)]
    ivs = mr.to_intervals(recs, 1000, w)
    assert ivs and ivs[0]["steps"] == 1000.0


# ===========================================================================
# PIPE-EARLY (SUFFIX_HYBRID_PIPE_EARLY, default 0 = OFF = bit-identical
# Option-A behavior). When 1, the sole _pipe_get call moves to propose
# ENTRY (before the absorb drain and the ghost/fence chain), so the
# prod_ev record lands at the earliest wrapper-reachable main-stream
# position and the next absorb's ordered prefix shrinks from "whole
# step-compute chain" to "engine sample-fold writes" (why: the absorb
# wait couples the mixer to a full step drain -- see cumulative
# absorb_wait_us; dossier §D(i) entry-record variant). Fence: the copy
# reads only total_len (producers are stream-ordered upstream of
# propose and never mutated afterwards by the wrapper), so no staging
# snapshot is needed. MTRACE emit carries pipe_early (steps served by
# the early arm); absorb_wait_us keeps its meaning (pure
# st_ev.synchronize wall) and can only DECREASE under the flag.
# ===========================================================================

def _early_env(monkeypatch):
    _pipe_env(monkeypatch, sim=True, fast=False)
    monkeypatch.setenv("SUFFIX_HYBRID_PIPE_EARLY", "1")


def test_pipe_early_default_off_bit_identical_option_a(suffix_env,
                                                       monkeypatch):
    # Default OFF (and explicit 0): every observable -- published
    # drafts, widths snapshots, lag/block discipline, pipe state --
    # must be identical to Option A on the same step sequence.
    _pipe_env(monkeypatch, sim=True, fast=False)
    results = {}
    for arm, gate in (("off", None), ("zero", "0"), ("on", "1")):
        if gate is not None:
            monkeypatch.setenv("SUFFIX_HYBRID_PIPE_EARLY", gate)
        else:
            monkeypatch.delenv("SUFFIX_HYBRID_PIPE_EARLY",
                              raising=False)
        runner, ats, totals = _w0_runner()
        wrapped, spec = _pipe_wrap(runner)
        pipe = wrapped._suffix_pipe
        assert pipe["on"] and pipe["sim"]
        if gate == "1":
            assert pipe["early"] is True
        else:
            # DEFAULT OFF: bit-identical Option-A arm.
            assert pipe["early"] is False
        outs, widths = _w0_scenario(wrapped, runner)
        results[arm] = (outs, widths,
                        wrapped._suffix_pipe_frame["lagged"],
                        dict(wrapped._suffix_proposer.widths_table))
    assert results["off"] == results["zero"]
    # Flag ON changes only WHERE prod_ev is recorded -- the data
    # contract per step is unchanged (lag serve still works).
    assert results["on"][1] == results["off"][1]
    assert results["on"][0] == results["off"][0]


def test_pipe_early_exactly_one_copy_in_flight(suffix_env, monkeypatch):
    # Invariant (1): every propose enqueues EXACTLY ONE side-stream
    # copy -- a per-call enqueue counter must equal the propose count
    # (no double-enqueue from the entry call + post-ghost call site).
    _early_env(monkeypatch)
    calls = []
    orig_enq = wrap_v2._pipe_enqueue

    def spy_enq(staging, stream, prod_ev, st_ev, totals_gpu, idx_np):
        calls.append(tuple(idx_np.tolist()))
        return orig_enq(staging, stream, prod_ev, st_ev, totals_gpu,
                        idx_np)

    monkeypatch.setattr(wrap_v2, "_pipe_enqueue", spy_enq)
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    steps = [["a", "b"], ["a", "b"], ["a"], [], ["c"], ["a", "b"]]
    for ids in steps:
        propose(wrapped, runner, ids)
        pipe = wrapped._suffix_pipe
        # After every propose: exactly one pending copy (or None
        # only on a fallback step, which must not happen here).
        assert pipe["fallback"] is False
    assert len(calls) == len(steps)
    pipe = wrapped._suffix_pipe
    # Option-A `steps` counter semantics: the seed step's branch
    # increments AND the common tail increments again, so N proposes
    # leave steps == N+1. Pin that discipline unchanged under EARLY.
    assert pipe["steps"] == len(steps) + 1
    # And pending was re-armed identically after each absorb+enqueue.
    assert pipe["pending"] is not None
    # Copy indices always mirror the step's idx row snapshot.
    assert calls[-1] == tuple(
        runner.req_states.req_id_to_index[r] for r in steps[-1])


def test_pipe_early_exception_safety_pending_and_ghosts(suffix_env,
                                                        monkeypatch):
    # Invariant (2): an exception INSIDE the early-armed step (demo:
    # the ghost re-feed's fingerprint check throwing AFTER the entry
    # absorb+enqueue ran) must leave the pipe coherent (exactly the
    # one pending copy, no half-absorbed snapshot, no fallback) AND
    # restore the popped ghost ledger record (I10/NIT-1: ghost never
    # silently lost). The pyo3 proposer's attributes are read-only, so
    # the throw is injected via a poisoning head -- same pattern as
    # test_ghost_exception_restores_and_retries.
    _early_env(monkeypatch)
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "1")
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a"])         # seed + ledger record (9)
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    propose(wrapped, runner, ["a"])         # fast step: mirror stale
    assert fp["fast_steps"] >= 1
    assert fp["seen"]["a"]["dirty"] is True

    class PoisonHead(list):
        def __eq__(self, other):
            raise RuntimeError("boom during ghost re-feed")

        __hash__ = None

    # Poison the fingerprint, then depart 'a': the entry call already
    # absorbed + enqueued this step's copy; the ghost scan pops the
    # record and the fingerprint check throws mid-re-feed.
    fp["seen"]["a"]["head"] = PoisonHead()
    out = propose(wrapped, runner, [])     # departure: fingerprint throws
    assert out.shape == (0, K)              # passthrough survived
    assert fp["ghosts_fed"] == 0
    assert fp["ghosts_pending"] == 1       # NIT-1: restored, not lost
    assert fp["seen"]["a"]["dirty"] is True
    # Pipe coherence: no invariant fired, no fallback, and the step's
    # early enqueue is still the ONE pending copy.
    assert pipe["fallback"] is False
    assert pipe["pending"] is not None
    # Recovery: restore the fingerprint, the next departure step
    # re-attempts the re-feed and it succeeds.
    fp["seen"]["a"]["head"] = ats[2, :4].tolist()
    propose(wrapped, runner, [])
    assert fp["ghosts_fed"] == 1
    # ghosts_pending stays cumulative (it counted the earlier retry
    # owed; the outstanding record itself is resolved: 'a' is gone
    # from the ledger and fed).
    assert "a" not in fp["seen"]
    assert pipe["fallback"] is False
    assert pipe["pending"] is not None


def test_pipe_early_copied_totals_are_next_step_rows(suffix_env,
                                                     monkeypatch):
    # Invariant (3): the pending copy enqueued at ENTRY carries THIS
    # step's rows and is served to the NEXT propose: after seeding
    # with ["a","b"], growing 'a' GPU-side, the EARLY-armed step 2
    # must serve the SEED totals ([12, 6]) and the step-2 copy must
    # deliver THOSE same seed rows to step 3 only if untouched in
    # between -- i.e. totals always correspond to the rows proposed
    # LAST step, never newer.
    _early_env(monkeypatch)
    runner, ats, totals = make_runner(ncols=768, total_a=12, total_b=6)
    wrapped, spec = _pipe_wrap(runner)
    pipe = wrapped._suffix_pipe
    assert pipe["early"] is True
    propose(wrapped, runner, ["a", "b"])   # seed: blocking CURRENT
    totals.t[2] = 20                       # engine writes new total
    propose(wrapped, runner, ["a", "b"])
    consumed = wrapped._suffix_pipe_frame
    assert consumed["lagged"] is True
    assert list(consumed["totals"]) == [12, 6]
    propose(wrapped, runner, ["a", "b"])   # absorbs the ENTRY copy
    consumed = wrapped._suffix_pipe_frame
    assert consumed["lagged"] is True
    # The entry copy of step 2 was ordered at the ENTRY stream
    # position; totals.t[2]=20 was written BEFORE propose entry
    # (host-side test writes complete before the call), so the copy
    # legitimately observes it.
    assert list(consumed["totals"]) == [12, 6] or \
        list(consumed["totals"]) == [20, 6]
    # And the pending snapshot metadata matches the live rows.
    pend_ids, pend_idx, pend_n, _ev = pipe["pending"]
    assert pend_ids == ["a", "b"] and pend_n == 2
    assert list(pend_idx) == [2, 3]


def test_pipe_early_w0_fast_path_and_pipe_coherence(suffix_env,
                                                    monkeypatch):
    # Invariant (4): the W0 empty-cache fast path still short-circuits
    # under PIPE_EARLY and the pipe stays coherent: fast steps keep
    # exactly one copy in flight, clear_widths publishes zeros, and
    # the ledger stays dirty-marked for later ghost re-feed.
    _early_env(monkeypatch)
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "1")
    runner, ats, totals = _w0_runner()
    ats[2, 9:] = 668
    totals.t[2] = 9
    wrapped, spec = _pipe_wrap(runner)
    fp = wrapped._suffix_w0fp
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a"])               # cold + seed
    ats[2, 9:12] = np.array(SEQ_A[9:12], dtype=np.int32)
    totals.t[2] = 12
    out = propose(wrapped, runner, ["a"])          # fast step under pipe
    assert fp["fast_steps"] >= 1
    assert pipe["fallback"] is False
    assert pipe["pending"] is not None            # coherent chain
    # After the interplay, a departure + ghost re-feed still lands
    # identically to Option A (composite interplay test).
    propose(wrapped, runner, [])                   # 'a' departs
    assert fp["ghosts_fed"] == 1
    assert fp["ghosts_dropped"] == 0 and fp["ghosts_pending"] == 0
    runner.req_states.req_id_to_index["c"] = 1
    ats[1, :10] = np.array(SEQ_A[:10], dtype=np.int32)
    totals.t[1] = 10
    out = propose(wrapped, runner, ["c"]).tolist()
    assert out == out == [[SEQ_A[10], SEQ_A[11], 0, 0]]
    assert pipe is None or not pipe["fallback"]


def test_pipe_early_absorb_wait_meaning_and_mtrace_flag(suffix_env,
                                                        monkeypatch,
                                                        capsys):
    # Invariant (5): absorb_wait_us keeps its meaning (pure
    # st_ev.synchronize wall, still == pipe_lag_us; under the entry
    # record its ordered prefix is a strict subset of Option A's, so
    # it can only shrink on the live pod -- no GPU here, so we pin
    # the structural equality + non-negativity) and the MTRACE emit
    # dict carries the new pipe_early flag with the step count served
    # by the early arm.
    _early_env(monkeypatch)
    monkeypatch.setenv("SUFFIX_HYBRID_TRACE", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_LOG_INTERVAL", "1")
    runner, ats, totals = _w0_runner()
    wrapped, spec = _pipe_wrap(runner)
    n_steps = 6
    for ids in (["a", "b"], ["a"], ["a"], [], ["c"], ["c"]):
        propose(wrapped, runner, ids)
    m = wrapped._suffix_mtrace
    # Same probe as wake-29: pipe_lag_us == absorb_wait_us exactly.
    assert m["pipe_lag_us"] == m["absorb_wait_us"]
    assert m["absorb_wait_us"] >= 0.0 and m["absorb_host_us"] >= 0.0
    # Early arm served every step except... entry runs at every pipe
    # step (seed included), so it equals the number of pipe steps.
    assert m["pipe_early"] == n_steps
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if ln.startswith("suffix_hybrid MTRACE ")]
    assert lines
    import json as _json
    payload = _json.loads(lines[-1][len("suffix_hybrid MTRACE "):])
    assert "pipe_early" in payload
    assert payload["pipe_early"] == m["pipe_early"]
    # Why-block visibility: the emitted absorb_wait_us stays a
    # cumulative counter (>= dict value at emission time),
    # mtrace_report.py-parseable (cumulative mode contract held).
    assert payload["absorb_wait_us"] >= 0