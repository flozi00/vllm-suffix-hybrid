# SPDX-License-Identifier: Apache-2.0
"""Rust FFI guard tests feasible CPU-only (audit 2026-09-24).

- Cache::add short-row clamp exactness (src/lib.rs ~161-175): a row with
  total == n leaves nothing indexed (no partial window, no cached tokens);
  total == n+1 is fully ingested. Exercised through the public SuffixCache
  Python API (add_sequence / stats / speculate) so no Rust source changes
  are needed.
- Poison-panic recovery (lock_cache, lib.rs ~62-90) and the guard_py panic
  boundary (lib.rs ~76-82): the audit asked for these only IF an existing
  seam can force a panic under lock. The compiled extension exposes no
  such seam (all ndarray indexing goes through ndarray's checked bounds,
  which return PyValueError, not panic), and adding one would require a
  Rust src change — out of scope for this test-only change. Skipped with
  reasons pinned here so the gap is explicit, not silent.
"""
import pytest
from suffix_hybrid._native import SuffixCache

n = 2   # conftest pins SUFFIX_HYBRID_INDEX_N=2; re-pin defensively below


def _cache():
    c = SuffixCache()
    c.set_test_n(2)      # deterministic n-gram order, independent of env
    return c


def test_short_row_clamp_total_eq_n_ingests_nothing():
    # total == n: add's `tokens.len() <= cfg.n` guard returns BEFORE any
    # window is indexed — even partial ingestion (e.g. cached_tokens or
    # a len-1 pseudo-window) would corrupt the index.
    c = _cache()
    c.add_sequence([1, 2])            # exactly n=2 tokens
    st = c.stats()
    assert st["cached_tokens"] == 0
    assert st["num_sequences"] == 0        # nothing stored at all
    assert st["num_index_keys"] == 0       # and nothing indexed
    # And nothing is retrievable: a context ending in the row's only
    # n-gram must NOT speculate a continuation.
    suffix, score, matched = c.speculate([1, 2], 4)
    assert suffix == [] and matched == 0


def test_short_row_clamp_total_eq_n_plus_1_ingests_1_window():
    # total == n+1: exactly one n-gram window fits — fully ingested and
    # retrievable. This is the exactness boundary the audit sketch asked
    # for: the clamp skips AT n, ingests AT n+1.
    c = _cache()
    c.add_sequence([1, 2, 3])          # n+1 = 3 tokens
    st = c.stats()
    assert st["cached_tokens"] == 3
    suffix, score, matched = c.speculate([1, 2], 4)
    assert suffix == [3]              # the one window's continuation
    assert matched >= 2


def test_short_row_clamp_boundary_via_proposer_ingest_gate():
    # The ingest boundary through V2SuffixProposer's mirror-departure
    # path (mixer.rs ~795: row.len() > 8 caller gate feeds Cache::add):
    # a departed mirror of exactly k+1-cacheable length ingests fully and
    # a later request repeating a PREFIX of it drafts the continuation.
    # Row lengths 8 and 9 pin the caller gate exactness: len == 8 is NOT
    # ingested (gate is `> 8`), len == 9 IS (the audit's n / n+1 shape at
    # the caller boundary).
    import numpy as np
    from suffix_hybrid._native import V2SuffixProposer

    def run_departure(row_len, expect_ingested):
        p = V2SuffixProposer(4, 512)
        tokens = np.zeros((2, 512), dtype=np.int32)
        row = list(range(100, 100 + row_len))
        tokens[1, :row_len] = np.array(row, dtype=np.int32)
        i1 = np.array([1], dtype=np.int64)
        t_full = np.array([row_len], dtype=np.int64)
        p.propose_suffix_only(["a"], i1, t_full, tokens)   # track 'a'
        # 'a' departs; batch is now 'b' at row 0 (unrelated content).
        p.propose_suffix_only(["b"], np.array([0], dtype=np.int64),
                              np.array([1], dtype=np.int64),
                              np.zeros((2, 512), dtype=np.int32))
        assert p.get_stats()["ingested"] == (1 if expect_ingested else 0)
        if not expect_ingested:
            return None, None
        # A fresh request repeats 'a's prefix (row 0 <- row's first 6
        # tokens) and must draft the ingested continuation.
        tokens[0, :] = 0
        tokens[0, :6] = np.array(row[:6], dtype=np.int32)
        packed, widths = p.propose_suffix_only(
            ["c"], np.array([0], dtype=np.int64),
            np.array([6], dtype=np.int64), tokens)
        # row has 9 tokens; a 6-token prefix leaves 3 to draft
        # (width is the true remaining continuation, capped at k=4).
        assert widths.tolist() == [3]
        # packed rows are always k-wide; slots past the width are 0.
        assert packed[0].tolist() == row[6:9] + [0]
        return widths, packed

    # len == 8: caller gate (> 8) rejects -> no ingestion, no hit possible.
    run_departure(8, expect_ingested=False)
    # len == 9: ingested; repeat request drafts 4 continuation tokens.
    run_departure(9, expect_ingested=True)


@pytest.mark.skip(reason="no test seam: lock_cache poison recovery "
                         "(lib.rs ~62-90) requires forcing a panic while "
                         "the cache mutex is held; no exposed API can "
                         "panic under lock (ndarray access is bounds-"
                         "checked), and adding a panic seam would need a "
                         "Rust src change, which this test-only change "
                         "forbids. Live behavior is covered by the "
                         "poison-safe lock_cache unwrap_or_else design; "
                         "revisit if a #[cfg(test)]-gated seam is added.")
def test_poison_panic_recovery_under_lock():
    raise AssertionError("unreachable; seam does not exist")


@pytest.mark.skip(reason="no test seam: guard_py (lib.rs ~76-82) converts "
                         "a Rust panic to RuntimeError so wrap.py's "
                         "except Exception degrades to native; forcing a "
                         "panic through the compiled extension is not "
                         "possible from CPU-only Python (every ndarray "
                         "index path bounds-checks into PyValueError). "
                         "Requires a Rust-side panic hook/seam first.")
def test_guard_py_converts_panic_to_runtime_error():
    raise AssertionError("unreachable; seam does not exist")