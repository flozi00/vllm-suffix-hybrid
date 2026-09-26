# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_HYBRID_SPLIT=diverge: counterfactual divergence arbitration."""
import importlib.util
import os
from pathlib import Path

import pytest

from suffix_hybrid import _native

CTX = list(range(1, 9))


@pytest.fixture
def diverge(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_SPLIT", "diverge")
    monkeypatch.setenv("SUFFIX_HYBRID_INDEX_N", "2")


def _mixer():
    m = _native.HybridMixer(5, 1024)
    m.suffix_cache.add_sequence(list(range(1, 200)))
    return m


def test_cold_trust_publishes_native_at_divergence(diverge):
    m = _mixer()
    # Suffix continues 9,10,11,..; native diverges at j=2.
    assert m.mix(["a"], [CTX], [[9, 10, 99, 98, 97]], None) == [[9, 10, 99, 98, 97]]
    assert m.last_native_counts() == [5]
    assert m.get_stats()["split_mode"] == "diverge"


def test_counterfactual_truth_trains_trust_then_suffix_overlays(diverge):
    m = _mixer()
    ctx, native = list(CTX), [9, 10, 99, 98, 97]
    m.mix(["a"], [ctx], [native], None)
    # Verifier accepted 9,10 and emitted the TRUE token 11 at j=2 although
    # native was published: the suffix was right, native wrong.
    ctx = ctx + [9, 10, 11]
    got = m.mix(["a"], [ctx], [[12, 13, 77, 76, 75]], [2])
    assert m.get_stats()["div_observed"] == 1
    # Trust now > 0.5: suffix overlays from the divergence on, full width.
    assert got == [[12, 13, 14, 15, 16]]
    assert m.last_native_counts() == [2]


def test_native_right_at_divergence_keeps_native(diverge):
    m = _mixer()
    ctx = list(CTX)
    m.mix(["a"], [ctx], [[9, 10, 99, 98, 97]], None)
    # Native was right at j=2 (truth 99): trust drops below the prior.
    ctx = ctx + [9, 10, 99]
    got = m.mix(["a"], [ctx], [[5, 6, 7, 8, 9]], [2])
    assert got == [[5, 6, 7, 8, 9]]
    assert max(m.get_stats()["div_trust"]) <= 0.5


def test_unreached_divergence_is_censored(diverge):
    m = _mixer()
    ctx = list(CTX)
    m.mix(["a"], [ctx], [[9, 10, 99, 98, 97]], None)
    # Rejected at position 1 < j=2: position 2 was never verified.
    ctx = ctx + [9, 55]
    m.mix(["a"], [ctx], [[1, 2, 3, 4, 5]], [1])
    assert m.get_stats()["div_observed"] == 0


def test_self_ngram_drafts_from_own_history(diverge, monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_SELF_NGRAM_MIN", "3")
    m = _native.HybridMixer(5, 1024)           # empty global corpus
    ctx = [7, 8, 9, 40, 41, 42, 43, 44, 3, 3, 7, 8, 9]
    m.mix(["a"], [ctx], [[40, 41, 90, 91, 92]], None)
    assert m.get_stats()["self_hits"] == 1
    # Divergence at j=2 recorded against the in-request lookup (42 vs 90).
    ctx = ctx + [40, 41, 42]
    m.mix(["a"], [ctx], [[43, 44, 1, 1, 1]], [2])
    assert m.get_stats()["div_observed"] == 1


def test_legacy_default_unchanged(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_SPLIT", raising=False)
    m = _native.HybridMixer(5, 128)
    assert m.get_stats()["split_mode"] == "legacy"


def _sim():
    path = Path(__file__).resolve().parents[1] / "bench" / "spec_sim.py"
    spec = importlib.util.spec_from_file_location("spec_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_simulator_diverge_beats_legacy_and_never_hurts(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_INDEX_N", "8")
    sim = _sim()
    try:
        res = {(c, arm): sim.run(c, arm, k=5, conc=4, n_req=24)["tokens_per_step"]
               for c in ("unique", "reuse", "self-repeat")
               for arm in ("native", "legacy", "diverge")}
    finally:
        os.environ.pop("SUFFIX_HYBRID_SPLIT", None)
    assert res[("reuse", "diverge")] > 1.3 * res[("reuse", "legacy")]
    assert res[("self-repeat", "diverge")] > 1.05 * res[("self-repeat", "legacy")]
    assert res[("unique", "diverge")] >= 0.99 * res[("unique", "native")]
