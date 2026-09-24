# SPDX-License-Identifier: Apache-2.0
"""Acceptance-EWMA draft-width gate in V2SuffixProposer (SUFFIX_HYBRID_EWMA_WIDTH).

Covers verify-econ dossier knob C.#1: at c1-warm the suffix cache hits and
publishes full 8-token drafts while acceptance collapses to ~0.024 — the
engine pays a (k+1)-wide verify forward per step and rejects ~97.6% of it.
The gate ports mix_core's accept_estimate/pick EWMA into the drafter-free
proposer: per-request EWMA of accepted draft tokens, publishing
w = ceil(EWMA).clamp(1, k) (floor 1, ceiling k).

Feedback source: the engine verifies LAST step's published width, so on a
continuing row the next authoritative total exceeds the previous mirror by
(accepted drafts + 1 sampled token) — accepted = mirror delta - 1, the same
ns-1 arithmetic the sync body uses for the hybrid mixer. Tests drive it
through propose_suffix_only exactly as the live adapter does.

OFF semantics (env unset -> width_gate=false) must be BIT-IDENTICAL to the
stock arm: identical packed drafts and widths for any input sequence.
"""
import os

import numpy as np
import pytest

from suffix_hybrid import wrap_v2
from suffix_hybrid._native import V2SuffixProposer

K = 4
NCOLS = 384

# 5 repeats of a 20-token stanza: long enough that no test runs off the
# passage, and periodic so n-gram lookups (INDEX_N=2 via conftest) always
# find a continuation.
STANZA = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
          113, 114, 115, 116, 117, 118, 119, 120]
SEQ = STANZA * 8


def _buf():
    return np.zeros((4, NCOLS), dtype=np.int32)


def _seed(p):
    """Depart 'a' with its full passage to prime the cache."""
    tokens = _buf()
    tokens[0, :len(SEQ)] = np.array(SEQ, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    p.propose_suffix_only(["a"], idx0,
                          np.array([len(SEQ)], dtype=np.int64), tokens)
    p.propose_suffix_only([], np.array([], dtype=np.int64),
                          np.array([], dtype=np.int64), tokens)
    assert p.get_stats()["ingested"] == 1


def _drive(p, accepted_seq, n_steps):
    """'b' repeats a's passage; the engine accepts accepted_seq[i] of the
    published drafts at step i (mirror advances accepted+1). Returns
    (packed, width) per step."""
    _seed(p)
    tokens = _buf()
    tokens[1, :10] = np.array(SEQ[:10], dtype=np.int32)
    idx1 = np.array([1], dtype=np.int64)
    out = []
    total = 10
    for step in range(n_steps):
        packed, widths = p.propose_suffix_only(
            ["b"], idx1, np.array([total], dtype=np.int64), tokens)
        out.append((packed.copy(), int(widths[0])))
        accepted = accepted_seq[step % len(accepted_seq)]
        total += accepted + 1         # accepted drafts + 1 sampled token
        tokens[1, :total] = np.array(SEQ[:total], dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# EWMA convergence DOWN (the c1-warm pathology: full hit, zero acceptance)
# ---------------------------------------------------------------------------

def test_gate_converges_down_under_rejection():
    # Every draft rejected: accepted = 0 each step. EWMA decays
    # 4 -> 2.8 -> 1.96 -> 1.37 -> 0.96 -> 0.67 -> ... and the published
    # width follows ceil(EWMA) with a floor of 1: [4, 3, 2, 2, 1, 1, 1, 1].
    # The floor matters: the row still HITS, so it keeps one speculative
    # position (the verify step stays width-consistent with what the
    # scheduler consumes) instead of dead-ending the EWMA at width 0.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    steps = _drive(p, [0], 8)
    widths = [w for _, w in steps]
    assert widths == [4, 3, 2, 2, 1, 1, 1, 1], widths
    assert p.get_stats()["hits"] == 8      # gate narrowed, never stopped
    assert p.get_stats()["hit_tokens"] < 8 * K   # and paid fewer rows
    # EWMA decayed geometrically: 7 feedbacks of accepted=0 -> 4*0.7**7.
    assert p.gate_table["b"][0] == 4.0 * 0.7 ** 7


def test_gate_ewma_matches_mix_core_decay_constants():
    # Port fidelity: partial acceptance decays by exactly 0.7/0.3 like
    # mix_core's accept_estimate (mixer.rs mix_core feedback pass).
    # 4 steps -> 3 feedbacks: 4 -> 3.1 -> 2.47 -> 2.029.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    _drive(p, [1], 4)
    expected = 4.0
    for _ in range(3):
        expected = 0.7 * expected + 0.3 * 1
    assert abs(p.gate_table["b"][0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# EWMA convergence UP (acceptance improves -> width widens, ceiling k)
# ---------------------------------------------------------------------------

def test_gate_rewiden_on_full_acceptance():
    # The engine accepts every published draft: creep-up raises the EWMA
    # to (last_width+1) per step, but the published width never exceeds
    # the raw matched width or the k ceiling — a healthy row stays at k.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    steps = _drive(p, [4], 6)
    widths = [w for _, w in steps]
    assert widths == [4, 4, 4, 4, 4, 4], widths


def test_gate_recovers_after_decay_when_acceptance_returns():
    # Decay down to the floor, then acceptance returns: the width must
    # climb back up (creep-up on full accept, ceil on partial), never
    # sticking at the floor.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    down = _drive(p, [0], 6)
    assert [w for _, w in down][-1] == 1
    # The same 'b' row now accepts everything (mirror already at 16).
    tokens = _buf()
    tokens[1, :] = 0
    tokens[1, :17] = np.array(SEQ[:17], dtype=np.int32)
    idx1 = np.array([1], dtype=np.int64)
    total = 17
    widths = []
    for _ in range(5):
        packed, w = p.propose_suffix_only(
            ["b"], idx1, np.array([total], dtype=np.int64), tokens)
        wi = int(w[0])
        widths.append(wi)
        total += wi + 1              # every draft accepted + 1 sampled
        tokens[1, :total] = np.array(SEQ[:total], dtype=np.int32)
    assert all(1 <= w <= K for w in widths)
    assert widths[-1] > widths[0]          # recovered upward
    assert widths == sorted(widths)        # monotone recovery


def test_gate_clamps_between_floor_and_ceiling():
    # Cycling accept counts (0..4): published width must ALWAYS land in
    # [1, k], whatever the feedback history.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    steps = _drive(p, [0, 1, 2, 3, 4], 40)
    widths = [w for _, w in steps]
    assert all(1 <= w <= K for w in widths), widths


# ---------------------------------------------------------------------------
# OFF semantics: env unset = zero behavior change, byte-identical drafts
# ---------------------------------------------------------------------------

def test_gate_off_is_byte_identical_to_default_ctor():
    # V2SuffixProposer(k, ctx) with width_gate omitted (env-unset
    # semantics; the adapter reads SUFFIX_HYBRID_EWMA_WIDTH which defaults
    # off) vs width_gate=False: byte-identical packed drafts and widths
    # on varied feedback.
    sched = [2, 0, 3, 1]
    a = _drive(V2SuffixProposer(K, NCOLS), sched, 6)
    b = _drive(V2SuffixProposer(K, NCOLS, 1, False, False), sched, 6)
    for (pa, wa), (pb, wb) in zip(a, b):
        assert (pa == pb).all()
        assert wa == wb


def test_gate_off_publishes_raw_width_on_arm():
    # The OFF arm's widths track the RAW matched-suffix width (clamped at
    # k) every step even while acceptance collapses — the exact c1-warm
    # pathology the gate exists to fix. Identical feedback under the gate
    # must NOT reproduce the raw widths.
    sched = [2, 0, 1, 0, 2]
    off = _drive(V2SuffixProposer(K, NCOLS, 1, False, False), sched, 6)
    on = _drive(V2SuffixProposer(K, NCOLS, 1, False, True), sched, 6)
    w_off = [w for _, w in off]
    w_on = [w for _, w in on]
    assert w_off == [K] * 6              # stock arm: always full width
    assert any(w < K for w in w_on)      # gate arm actually narrowed


def test_gate_off_keeps_no_gate_state_and_off_stats():
    p_off = V2SuffixProposer(K, NCOLS, 1, False, False)
    assert p_off.get_stats()["width_gate"] is False
    assert p_off.gate_table == {}
    p_on = V2SuffixProposer(K, NCOLS, 1, False, True)
    assert p_on.get_stats()["width_gate"] is True


# ---------------------------------------------------------------------------
# Publish path at width < k: gated truncation keeps the packed contract
# ---------------------------------------------------------------------------

def test_publish_path_at_width_below_k():
    # After the EWMA decays, the published draft is the gated PREFIX of
    # the matched suffix: content identical to what the ungated draft
    # would carry in [0:w], packed row stays [1, k] with zeros past w,
    # and widths_table (what get_draft_tokens schedules) matches.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    _drive(p, [0], 4)                    # widths [4, 3, 2, 2]; EWMA 1.37
    tokens = _buf()
    tokens[1, :14] = np.array(SEQ[:14], dtype=np.int32)
    idx1 = np.array([1], dtype=np.int64)
    total = 14
    packed, widths = p.propose_suffix_only(
        ["b"], idx1, np.array([total], dtype=np.int64), tokens)
    w = int(widths[0])
    assert 1 <= w < K                     # decayed: ceil(1.37) == 2
    # Prefix identity vs the ungated arm on identical fresh state.
    p_ref = V2SuffixProposer(K, NCOLS, 1, False, False)
    _seed(p_ref)
    packed_ref, w_ref = p_ref.propose_suffix_only(
        ["b"], idx1, np.array([total], dtype=np.int64), tokens)
    assert int(w_ref[0]) == K
    assert packed.shape == (1, K)
    assert packed[0, :w].tolist() == packed_ref[0, :w].tolist()
    assert packed[0, w:].tolist() == [0] * (K - w)
    assert p.widths_table["b"] == w


def test_clear_widths_censors_gate_feedback():
    # Retraction (the adapter's exception path) must zero last_width so
    # the NEXT step's mirror delta cannot be misread as accepted drafts
    # at a phantom width.
    p = V2SuffixProposer(K, NCOLS, 1, False, True)
    _drive(p, [2], 3)
    tokens = _buf()
    tokens[1, :17] = np.array(SEQ[:17], dtype=np.int32)
    idx1 = np.array([1], dtype=np.int64)
    total = 17
    p.propose_suffix_only(["b"], idx1,
                          np.array([total], dtype=np.int64), tokens)
    p.clear_widths()
    assert p.widths_table["b"] == 0
    ewma_before = p.gate_table["b"][0]
    # A big mirror jump right after a retraction must NOT train the EWMA
    # upward: censoring held (a retracted width was never verified).
    total += 4
    tokens[1, :total] = np.array(SEQ[:total], dtype=np.int32)
    p.propose_suffix_only(["b"], idx1,
                          np.array([total], dtype=np.int64), tokens)
    assert p.gate_table["b"][0] == ewma_before
    # The retracted row resumes gating from the decayed EWMA and never
    # publishes an out-of-range width.
    assert 0 <= p.widths_table["b"] <= K
    assert p.get_stats()["resets"] == 0   # continuity held throughout


# ---------------------------------------------------------------------------
# Adapter env wiring: SUFFIX_HYBRID_EWMA_WIDTH arms the gate in the live
# wrap (suffix-only path); unset / "0" / "false" keep it OFF. Mirrors
# test_suffix_only_wrap's fake-runner wiring.
# ---------------------------------------------------------------------------

class _EnvIsolated:
    """Clear every SUFFIX_HYBRID_* env the suffix-only wrap reads."""

    VARS = ("SUFFIX_HYBRID_SUFFIX_ONLY", "SUFFIX_HYBRID_SYNC_HOOK",
            "SUFFIX_HYBRID_TP_MODE", "SUFFIX_HYBRID_SUFFIX_MIN",
            "SUFFIX_HYBRID_UNIFORM_K", "SUFFIX_HYBRID_TRACE",
            "SUFFIX_HYBRID_TRACE_VERIFY", "SUFFIX_HYBRID_TRACE_VERIFY2",
            "SUFFIX_HYBRID_SCHEDTRACE", "SUFFIX_HYBRID_LOG_INTERVAL",
            "SUFFIX_HYBRID_ZERO_OP", "SUFFIX_HYBRID_W0_FASTPATH",
            "SUFFIX_HYBRID_D2H_PIPE", "SUFFIX_HYBRID_D2H_PIPE_SIM",
            "SUFFIX_HYBRID_PIPE_EARLY", "SUFFIX_HYBRID_EWMA_WIDTH")

    def __enter__(self):
        self.saved = {v: os.environ.get(v) for v in self.VARS}
        for v in self.VARS:
            os.environ.pop(v, None)
        return self

    def __exit__(self, *exc):
        for v, val in self.saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
        return False


class _FakeSpeculator:
    """Fake speculator exposing exactly what _suffix_only_wrap reads."""

    def __init__(self):
        import torch
        k = 4
        self.draft_tokens = torch.zeros((4, k), dtype=torch.int64)
        self.max_num_reqs = 4
        self.max_model_len = 256
        self.num_speculative_steps = k
        self.draft_logits = None

    def propose(self, *args, **kwargs):   # wraps() target, never called
        raise AssertionError("suffix-only replaces propose; never called")


def _install_wrap():
    """Wire up the suffix-only wrap against the fake runner shape used by
    test_suffix_only_wrap (imports are lazy; torch is required there but
    _suffix_only_wrap itself only needs numpy/speculator attrs)."""
    from types import SimpleNamespace as NS

    import torch

    k = 4
    ats = np.zeros((4, 256), dtype=np.int32)
    totals = NS(gpu=torch.zeros(4, dtype=torch.int64))
    runner = NS(req_states=NS(
        total_len=totals,
        all_token_ids=NS(_uva_buf=NS(np=ats)),
        req_id_to_index={"a": 2, "b": 3}))
    speculator = _FakeSpeculator()
    group = NS(rank_in_group=0,
               broadcast=lambda value, src=0: value)
    mixer = NS()                      # unused on the suffix-only path
    wrapped = wrap_v2._suffix_only_wrap(runner, speculator, mixer, group, k)
    return wrapped


def test_env_ewma_width_arms_and_disarms_gate():
    torch = pytest.importorskip("torch")
    for env_val, expected in ((None, False), ("", False), ("0", False),
                              ("false", False), ("1", True), ("on", True)):
        with _EnvIsolated():
            if env_val is not None:
                os.environ["SUFFIX_HYBRID_EWMA_WIDTH"] = env_val
            wrapped = _install_wrap()
            proposer = wrapped._suffix_proposer
            assert proposer.get_stats()["width_gate"] is expected, env_val