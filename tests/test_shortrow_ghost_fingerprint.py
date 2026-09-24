# SPDX-License-Identifier: Apache-2.0
"""Regression tests for review aa5accb1 findings 1 and 3.

Finding 3: _ledger_refresh recorded row[:4] unconditionally while the
ghost re-feed checks row[:min(4, tot)] -- a request whose recorded
total < 4 padded its head with stale free-list tokens, so the
fingerprint NEVER matched and its history was always dropped as
ghosts_drop_fp. Pin: a short-row (total < 4) ghost is RETAINED (re-fed),
not dropped.

Finding 1: a lagged pipe snapshot that raced next step's post_update
can carry torn totals that pass the aligned (n/ids/idx) check. The
absorb-side monotonicity assert must fail-closed on a SHRINKING
row-persistent rid total. Test simulates the torn mixed-generation
snapshot by mutating staging_cpu between enqueue and absorb (exact
torn-read shape: enqueue copy correct, later overwrite shrinks).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from suffix_hybrid import wrap_v2
from suffix_hybrid._native import HybridMixer

K = 4
NROWS = 4
NCOLS = 64

SEQ_A = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]


@pytest.fixture
def suffix_env(monkeypatch):
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
    def __init__(self, totals):
        self.t = torch.tensor(list(totals), dtype=torch.int64)
        self.gathers = []

    def __getitem__(self, idx):
        g = self.t[idx]
        self.gathers.append(g.tolist())
        return g


class Speculator:
    def propose(self, *args, **kwargs):
        raise AssertionError("suffix-only replaces propose; never called")

    def __init__(self, k=K, max_reqs=NROWS):
        self.draft_tokens = torch.zeros((max_reqs, k), dtype=torch.int64)
        self.max_num_reqs = max_reqs
        self.max_model_len = None
        self.num_speculative_steps = k
        self.draft_logits = None


class TP:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, value, src=0):
        return value


def make_runner(nrows=NROWS, ncols=NCOLS, total_a=None, row_b=None,
                total_b=None):
    ats = np.zeros((nrows, ncols), dtype=np.int32)
    ats[0, :] = 666
    ats[1, :] = 667
    ats[nrows - 2, :len(SEQ_A)] = np.array(SEQ_A, dtype=np.int32)
    if ncols >= len(SEQ_A):
        ats[nrows - 2, len(SEQ_A):] = 668
    if row_b is None:
        row_b = [201, 202, 203]
    ats[nrows - 1, :len(row_b)] = np.array(row_b, dtype=np.int32)
    total_a = len(SEQ_A) if total_a is None else total_a
    total_b = len(row_b) if total_b is None else total_b
    ats[nrows - 1, len(row_b):] = 669
    totals = FakeTotals([3, 3, total_a, total_b])
    states = NS(total_len=NS(gpu=totals),
                all_token_ids=NS(_uva_buf=NS(np=ats)),
                req_id_to_index={"a": nrows - 2, "b": nrows - 1})
    runner = NS(req_states=states)
    runner.vllm_config = NS(scheduler_config=NS(async_scheduling=False,
                                                max_concurrent_batches=1,
                                                num_speculative_tokens=K))
    return runner, ats, totals


def wrap(runner, k=K):
    spec = Speculator(k=k)
    mixer = HybridMixer(k, 512)
    wrapped = wrap_v2._suffix_only_wrap(runner, spec, mixer, TP(), k)
    wrapped._suffix_hybrid_hook = True
    return wrapped, spec


def propose(wrapped, runner, req_ids, **kw):
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


def arm_sim(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE_SIM", "1")


# ---------------------------------------------------------------------------
# Finding 3: short-row (total < 4) ghost is RETAINED, not dropped
# ---------------------------------------------------------------------------

def test_short_row_ghost_retained_not_dropped(suffix_env, monkeypatch):
    # 'b' has a THREE-token passage (total_b=3 < 4): the recorded head
    # is row[:3] now, exactly the [:min(4, tot)] slice the re-feed
    # checks. Before the fix the ledger stored row[:4] which included
    # the stale 669 free-list pad -> fingerprint never matched ->
    # ghosts_drop_fp + lost history.
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "1")
    runner, ats, totals = make_runner(
        row_b=[201, 202, 203], total_b=3)
    wrapped, spec = wrap(runner)
    fp = wrapped._suffix_w0fp
    # step 1: ledger records both rows (head at [:min(4, tot)])
    propose(wrapped, runner, ["a", "b"])
    # step 2: live through a fast step -> mirror goes stale (dirty)
    propose(wrapped, runner, ["a", "b"])
    assert fp["fast_steps"] >= 1
    # step 3: both depart, then a ghost re-feed would be attempted
    propose(wrapped, runner, [])
    # The SHORT-ROW ghost ('b', total 3) must have matched its
    # fingerprint and been re-fed (retained), NOT dropped.
    assert fp["ghosts_fed"] == 2, (
        f"short-row ghosts must be re-fed; fed={fp['ghosts_fed']} "
        f"dropped={fp['ghosts_dropped']} drop_fp={fp['ghosts_drop_fp']}")
    assert fp["ghosts_dropped"] == 0
    assert fp["ghosts_drop_fp"] == 0, "short-row fingerprint miscounted"


def test_short_row_ghost_ledger_head_shape(suffix_env, monkeypatch):
    # Direct ledger-shape pin: every recorded head has len == min(4, tot).
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "1")
    runner, ats, totals = make_runner(
        row_b=[201, 202, 203], total_b=3)
    wrapped, spec = wrap(runner)
    propose(wrapped, runner, ["a", "b"])
    seen = wrapped._suffix_w0fp["seen"]
    assert seen["a"]["head"] == SEQ_A[:4]
    assert seen["b"]["head"] == [201, 202, 203]   # len 3, not padded to 4


# ---------------------------------------------------------------------------
# Finding 1: absorb-side monotonicity assert fails closed on torn totals
# ---------------------------------------------------------------------------

def test_pipe_absorb_monotonicity_assert_fails_closed(suffix_env,
                                                      monkeypatch):
    # Simulate the torn mixed-generation snapshot (review #1): the
    # enqueue step wrote the CORRECT totals into staging, then a
    # racing write lands a SMALLER total for a row-persistent rid
    # before the absorb. The absorb assert must raise I7 -> the pipe
    # sticky-disables (fallback) and this step serves a BLOCKING
    # stock read; nothing lagged reaches the mixer.
    arm_sim(monkeypatch)
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "0")
    runner, ats, totals = make_runner()
    wrapped, spec = wrap(runner)
    pipe = wrapped._suffix_pipe
    assert pipe["on"] and pipe["sim"]
    propose(wrapped, runner, ["a", "b"])           # seed + enqueue
    propose(wrapped, runner, ["a", "b"])           # absorb + enqueue
    assert pipe["fallback"] is False
    assert pipe["last_totals"] == {"a": 12, "b": 3}
    assert pipe["pending"] is not None
    # TORN READ: overwrite the in-flight staging rows with shrunken
    # mixed-generation values (exact thing the dossier says "can still
    # pass the aligned check"): 'a' 12 -> 2, 'b' 3 -> 1.
    st = pipe["staging_cpu"]
    st[0] = 2
    st[1] = 1
    out = propose(wrapped, runner, ["a", "b"])      # absorb must raise I7
    assert pipe["fallback"] is True, "torn totals must sticky-disable"
    assert "I7" in pipe["fallback_reason"]
    assert pipe["invariant_fails"].get("I7", 0) >= 1
    # Fail-closed: this step's totals came from the BLOCKING stock read
    # (current, correct) and the draft is published normally.
    assert wrapped._suffix_pipe_frame["lagged"] is False
    assert list(wrapped._suffix_pipe_frame["totals"]) == [12, 3]
    assert out.shape == (2, K)


def test_pipe_absorb_monotone_growth_passes_quietly(suffix_env,
                                                    monkeypatch):
    # Non-regression of the assert itself: normal append-only growth
    # across steps must never trip I7 (totals only INCREASE).
    arm_sim(monkeypatch)
    monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", "0")
    runner, ats, totals = make_runner()
    wrapped, spec = wrap(runner)
    pipe = wrapped._suffix_pipe
    propose(wrapped, runner, ["a", "b"])           # seed
    totals.t[2] = 15                               # 'a' grows
    totals.t[3] = 9                                # 'b' grows
    propose(wrapped, runner, ["a", "b"])           # lag absorb (old totals)
    totals.t[2] = 16
    propose(wrapped, runner, ["a", "b"])           # absorb grown totals
    assert pipe["fallback"] is False
    assert wrapped._suffix_pipe_frame["lagged"] is True
    # Only a RID RETAINED across snapshots is checked; a completely
    # fresh batch (rid not in last_totals) never trips the assert.
    totals.t[2] = 1                                # shrink AFTER a gap
    pipe["last_totals"] = {}                       # sim of post-departure reset
    propose(wrapped, runner, ["a"])
    assert pipe["fallback"] is False