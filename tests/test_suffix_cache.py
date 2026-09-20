# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SuffixCache — pure stdlib, no numpy/torch/vllm."""

import os
import sys
import unittest



from suffix_hybrid.suffix_cache import SuffixCache


def set_env(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


class TestSuffixCache(unittest.TestCase):
    def setUp(self):
        # Reset all env knobs so tests are deterministic.
        keys = [k for k in os.environ if k.startswith("SUFFIX_HYBRID")]
        for k in keys:
            os.environ.pop(k)

    def test_no_match_on_empty_cache(self):
        cache = SuffixCache()
        draft, score, match_len = cache.speculate([1, 2, 3, 4, 5, 6, 7, 8], 4)
        self.assertEqual(draft, [])
        self.assertEqual(score, 0.0)
        self.assertEqual(match_len, 0)

    def test_exact_continuation(self):
        cache = SuffixCache()
        # Corpus: a JSON-schema-ish repeating sequence.
        cache.add_sequence(list(range(10, 30)))
        context = [99, 12, 13, 14, 15, 16, 17, 18, 19]  # ends mid-sequence
        draft, score, match_len = cache.speculate(context, 3)
        self.assertEqual(draft, [20, 21, 22])
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(match_len, 8)

    def test_cap_by_spec_factor(self):
        # With factor 1.0 and offset 0, draft length <= match_len.
        cache = SuffixCache()
        seq = list(range(30))
        cache.add_sequence(seq)
        context = seq[:12]  # match_len 12 (full back-extension)
        draft, _score, match_len = cache.speculate(context, 100)
        self.assertLessEqual(len(draft), match_len)

    def test_offset_extends_cap(self):
        set_env(SUFFIX_HYBRID_MAX_SPEC_OFFSET=4)
        cache = SuffixCache()
        seq = list(range(30))
        cache.add_sequence(seq)
        draft, _s, match_len = cache.speculate(seq[:12], 100)
        self.assertLessEqual(len(draft), match_len + 4)
        self.assertGreater(len(draft), 0)

    def test_longer_back_extension_wins(self):
        cache = SuffixCache()
        # Two sequences share the index n-gram tail but one matches further back.
        cache.add_sequence([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cache.add_sequence([9, 9, 9, 9, 2, 3, 4, 5, 6, 7, 8, 9])
        context = [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]
        draft, _s, _m = cache.speculate(context, 4)
        self.assertEqual(draft, [9])  # continuation of the longer match

    def test_frequency_tiebreak(self):
        cache = SuffixCache()
        # Same back-extension length, different continuations; the one seen
        # more often (more candidates) should win.
        for _ in range(3):
            cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 100])
        cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 100])
        context = [1, 2, 3, 4, 5, 6, 7, 8]
        draft, _s, _m = cache.speculate(context, 1)
        self.assertEqual(draft, [100])

    def test_min_token_count(self):
        set_env(SUFFIX_HYBRID_MIN_TOKEN_COUNT=5)
        cache = SuffixCache()
        cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 42])
        context = [1, 2, 3, 4, 5, 6, 7, 8]
        draft, _s, _m = cache.speculate(context, 2)
        self.assertEqual(draft, [])  # support 1 < min 5

    def test_fifo_eviction(self):
        set_env(SUFFIX_HYBRID_MAX_CACHED_REQUESTS=2)
        cache = SuffixCache()
        cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 11])
        cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 22])
        cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, 33])  # evicts the first
        draft, _s, _m = cache.speculate([1, 2, 3, 4, 5, 6, 7, 8], 1)
        self.assertEqual(draft, [33])
        stats = cache.stats()
        self.assertEqual(stats["num_sequences"], 2)

    def test_negative_cap_uses_bounded_default(self):
        set_env(SUFFIX_HYBRID_MAX_CACHED_REQUESTS=-1)
        cache = SuffixCache()
        for i in range(20):
            cache.add_sequence([1, 2, 3, 4, 5, 6, 7, 8, i])
        self.assertEqual(cache.stats()["num_sequences"], 20)
        self.assertEqual(cache.stats()["max_cached_requests"], 512)

    def test_short_sequence_rejected(self):
        cache = SuffixCache()
        cache.add_sequence([1, 2])  # too short to index
        self.assertEqual(cache.stats()["num_sequences"], 0)

    def test_window_respects_max_tree_depth(self):
        set_env(SUFFIX_HYBRID_MAX_TREE_DEPTH=16)
        cache = SuffixCache()
        seq = list(range(100))
        cache.add_sequence(seq)
        # Long context; only last 16 tokens are the window.
        draft, _s, _m = cache.speculate(seq[:80], 4)
        # tail window = seq[64:80], matches corpus -> continuation follows
        self.assertEqual(draft, seq[80:84])

    def test_score_between_zero_and_one(self):
        cache = SuffixCache()
        cache.add_sequence(list(range(40)))
        _d, score, _m = cache.speculate(list(range(20)), 5)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_thread_safety_smoke(self):
        import threading

        cache = SuffixCache()
        errors = []

        def writer(base):
            try:
                for i in range(50):
                    cache.add_sequence(
                        [base, base + 1, base + 2, base + 3, base + 4,
                         base + 5, base + 6, base + 7, i]
                    )
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(50):
                    cache.speculate([1, 2, 3, 4, 5, 6, 7, 8], 2)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(b,)) for b in (1, 100)] + \
                  [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_accepts_plain_and_other_sequences(self):
        cache = SuffixCache()
        cache.add_sequence((1, 2, 3, 4, 5, 6, 7, 8, 9))  # tuple
        # numpy absent is fine; bytes-like not required
        draft, _s, _m = cache.speculate((1, 2, 3, 4, 5, 6, 7, 8), 1)
        self.assertEqual(draft, [9])


if __name__ == "__main__":
    unittest.main()