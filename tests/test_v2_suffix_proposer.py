# SPDX-License-Identifier: Apache-2.0
"""V2SuffixProposer (drafter-free suffix-only mode) unit tests.

Covers the Rust core of SUFFIX_HYBRID_SUFFIX_ONLY=1 (gemma lane, wake
#6-7): per-request mirror state, boundary-window continuity, echo-hit
drafting, ingestion on departure, ragged widths, and the packed numpy
return contract the thin Python adapter consumes.
"""
import numpy as np
import pytest

from suffix_hybrid._native import V2SuffixProposer


def _buf(nrows, ncols):
    return np.zeros((nrows, ncols), dtype=np.int32)


def test_fresh_row_no_cache_misses_width_zero():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    tokens[0, :10] = np.arange(10, dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["a"], np.array([0], dtype=np.int64),
        np.array([10], dtype=np.int64), tokens)
    assert widths.tolist() == [0]
    assert packed.shape == (4,)
    assert (packed == 0).all()


def test_echo_hit_drafts_exact_continuation():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    seq = list(range(100, 122))
    tokens[0, :22] = np.array(seq, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    for t in (10, 22):
        p.propose_suffix_only(["a"], idx0, np.array([t], dtype=np.int64), tokens)
    # b repeats a's passage prefix -> drafts a's continuation
    tokens[1, :10] = np.array(seq[:10], dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["b"], np.array([1], dtype=np.int64),
        np.array([10], dtype=np.int64), tokens)
    assert widths.tolist() == [4]
    assert packed[:4].tolist() == seq[10:14]
    st = p.get_stats()
    assert st["hits"] == 1 and st["hit_tokens"] == 4 and st["ingested"] == 1


def test_incremental_growth_sustains_hits_no_reset():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    seq = list(range(100, 122))
    tokens[0, :22] = np.array(seq, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    for t in (10, 22):
        p.propose_suffix_only(["a"], idx0, np.array([t], dtype=np.int64), tokens)
    tokens[1, :10] = np.array(seq[:10], dtype=np.int32)
    idx1 = np.array([1], dtype=np.int64)
    p.propose_suffix_only(["b"], idx1, np.array([10], dtype=np.int64), tokens)
    for t in (11, 12, 13):
        tokens[1, :t] = np.array(seq[:t], dtype=np.int32)
        packed, widths = p.propose_suffix_only(
            ["b"], idx1, np.array([t], dtype=np.int64), tokens)
        assert widths[0] == 4
        assert packed[:4].tolist() == seq[t:t + 4]
    assert p.get_stats()["resets"] == 0


def test_row_swap_same_slot_resets_not_crosscontaminates():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    seq = list(range(100, 122))
    tokens[0, :22] = np.array(seq, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    for t in (10, 22):
        p.propose_suffix_only(["a"], idx0, np.array([t], dtype=np.int64), tokens)
    # a different request id, same slot, different content: fresh row,
    # miss (no cross-contamination), no reset (new id was never tracked)
    other = list(range(500, 540))
    tokens[0, :12] = np.array(other[:12], dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["c"], idx0, np.array([12], dtype=np.int64), tokens)
    assert widths[0] == 0
    # a tracked id whose content CHANGED (e.g. preempted+re-prefilled)
    # must reset, not silently stitch unrelated contexts together
    tokens[0, :12] = np.array(other[:12], dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["c"], idx0, np.array([12], dtype=np.int64), tokens)
    assert p.get_stats()["resets"] == 0  # same content: continuing
    tokens[0, :12] = np.array(range(700, 712), dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["c"], idx0, np.array([12], dtype=np.int64), tokens)
    assert p.get_stats()["resets"] == 1  # content changed: reset


def test_widths_table_tracks_last_published_width():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    seq = list(range(100, 122))
    tokens[0, :22] = np.array(seq, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    for t in (10, 22):
        p.propose_suffix_only(["a"], idx0, np.array([t], dtype=np.int64), tokens)
    wt = p.widths_table
    assert dict(wt) == {"a": 0}


def test_int64_tokens_view_accepted():
    p = V2SuffixProposer(4, 512)
    tokens = np.zeros((8, 512), dtype=np.int64)
    seq = list(range(100, 122))
    tokens[0, :22] = np.array(seq, dtype=np.int64)
    idx0 = np.array([0], dtype=np.int64)
    for t in (10, 22):
        p.propose_suffix_only(["a"], idx0, np.array([t], dtype=np.int64), tokens)
    tokens[1, :10] = np.array(seq[:10], dtype=np.int64)
    packed, widths = p.propose_suffix_only(
        ["b"], np.array([1], dtype=np.int64),
        np.array([10], dtype=np.int64), tokens)
    assert widths.tolist() == [4]
    assert packed[:4].tolist() == seq[10:14]


def test_batch_dimensions_mismatch_raises():
    p = V2SuffixProposer(4, 512)
    tokens = _buf(8, 512)
    with pytest.raises(Exception):
        p.propose_suffix_only(
            ["a", "b"], np.array([0], dtype=np.int64),
            np.array([10], dtype=np.int64), tokens)


def test_min_len_floor_suppresses_short_drafts():
    # min_len=3: a 1-2 token continuation must NOT publish a draft
    p = V2SuffixProposer(4, 512, 3)
    tokens = _buf(8, 512)
    seq = list(range(100, 110))  # a's passage: b matches first 9, 1 left
    tokens[0, :10] = np.array(seq, dtype=np.int32)
    idx0 = np.array([0], dtype=np.int64)
    p.propose_suffix_only(["a"], idx0, np.array([10], dtype=np.int64), tokens)
    tokens[1, :9] = np.array(seq[:9], dtype=np.int32)
    packed, widths = p.propose_suffix_only(
        ["b"], np.array([1], dtype=np.int64),
        np.array([9], dtype=np.int64), tokens)
    # only 1 continuation token available (seq[9]) -> below min_len -> miss
    assert widths.tolist() == [0]