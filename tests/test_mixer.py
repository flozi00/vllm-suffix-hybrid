import unittest
from suffix_hybrid import _native


class MixerTests(unittest.TestCase):
    def test_fresh_row_publishes_native_then_learns_split(self):
        # Fresh row: publish the native draft untouched (no per-row
        # evidence yet). From the second proposal on the learned split
        # binds and the suffix tail replaces the unearned native tail.
        self.assertTrue(hasattr(_native, 'HybridMixer'), 'Rust hybrid token mixer is required')
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 30)))
        result = mixer.mix(['a'], [list(range(1, 9))], [[9, 10, 11, 12, 99]], None)
        self.assertEqual(result, [[9, 10, 11, 12, 99]])
        self.assertEqual(mixer.last_native_counts(), [5])
        result = mixer.mix(['a'], [list(range(1, 9))], [[9, 10, 11, 12, 99]], [5])
        self.assertEqual(result, [[9, 10, 11, 12, 13]])
        self.assertEqual(mixer.last_native_counts(), [4])

    def test_no_suffix_evidence_preserves_native_draft(self):
        # No suffix evidence: publish the full native draft. Shortening
        # without evidence would only shrink the window the EWMA learns
        # from. The split still learns: the native prefix IS the budget.
        mixer = _native.HybridMixer(5, 128)
        self.assertEqual(mixer.mix(['a'], [[1, 2]], [[3, 4, 5, 6, 7]], None), [[3, 4, 5, 6, 7]])
        self.assertEqual(mixer.last_native_counts(), [5])

    def test_unreached_suffix_is_not_counted_rejected(self):
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 30)))
        mixer.mix(['a'], [list(range(1, 9))], [[9, 10, 11, 12, 99]], None)
        mixer.mix(['a'], [list(range(1, 11))], [[11, 12, 13, 14, 99]], [1])
        stats = mixer.get_stats()
        self.assertEqual(stats['native_tested'], 2)
        self.assertEqual(stats['native_accepted'], 1)
        self.assertEqual(stats['suffix_tested'], 0)

    def test_budget_follows_row_acceptance(self):
        # The split is per-request: a row that accepts 2 of 5 converges its
        # NATIVE PREFIX toward its own EWMA (~2) while the cache tail covers
        # the replaced slots; a fully-accepted row keeps the full draft.
        # The native draft DIVERGES from the cache ([..,99,98] vs [..,13..])
        # so the published tail is genuinely non-native and measurable.
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 100)))
        native = [[9, 10, 11, 99, 98]]
        accepted = None
        for _ in range(10):
            mixer.mix(['a'], [list(range(1, 9))], native, accepted)
            accepted = [2]
        # Row a: budget ~2 -> prefix max(budget, floor=cap-1)=4 on the way
        # down, settling at 3. Slots past the prefix carry cache evidence.
        got = mixer.mix(['a'], [list(range(1, 9))], native, [2])
        self.assertLessEqual(mixer.last_native_counts()[0], 4)
        self.assertEqual(len(got[0]), 5)
        self.assertEqual(got[0][:2], [9, 10])
        self.assertNotEqual(got[0][3:], [99, 98])
        # Row b with full acceptance keeps the full native draft. The cache
        # tail matches the draft here, so the publish is identical either
        # way and every slot counts as native evidence.
        b_acc = None
        for _ in range(12):
            got_b = mixer.mix(['b'], [list(range(1, 9))], [[9, 10, 11, 12, 13]], b_acc)
            b_acc = [5]
        self.assertEqual(got_b, [[9, 10, 11, 12, 13]])
        self.assertEqual(mixer.last_native_counts()[0], 5)

    def test_numpy_buffers_mix_without_python_token_lists(self):
        import numpy as np
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 30)))
        for dtype in (np.int32, np.int64):
            data = np.arange(1, 257, dtype=dtype).reshape(1, -1)
            got = mixer.mix_numpy(['a'], np.array([8], dtype=dtype), data, [[9, 10, 11, 12, 99]], None)
            self.assertEqual(got[0][:4], [9, 10, 11, 12])
        with self.assertRaises(ValueError):
            mixer.mix_numpy(['a'], np.array([-1], dtype=np.int32), data, [[9]], None)

    def test_dimensions_fail_before_mutation(self):
        mixer = _native.HybridMixer(4, 128)
        with self.assertRaises(ValueError):
            mixer.mix(['a'], [], [[1, 2]], None)
        self.assertEqual(mixer.get_stats()['calls'], 0)

    def test_max_length_cap_and_duplicate_ids(self):
        mixer = _native.HybridMixer(5, 10)
        self.assertEqual(mixer.mix(['a'], [list(range(9))], [[9, 10, 11]], None), [[9]])
        with self.assertRaises(ValueError):
            mixer.mix(['a', 'a'], [[1], [1]], [[2], [2]], None)

if __name__ == '__main__':
    unittest.main()
