import unittest
from suffix_hybrid import _native


class MixerTests(unittest.TestCase):
    def test_native_prefix_conditions_suffix_tail(self):
        self.assertTrue(hasattr(_native, 'HybridMixer'), 'Rust hybrid token mixer is required')
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 30)))
        result = mixer.mix(['a'], [list(range(1, 9))], [[9, 10, 11, 12, 99]], None)
        self.assertEqual(result, [[9, 10, 11, 12, 13]])
        self.assertEqual(mixer.last_native_counts(), [4])

    def test_no_suffix_evidence_preserves_native_draft(self):
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

    def test_split_learns_from_verification(self):
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 100)))
        counts = []
        accepted = None
        for i in range(200):
            mixer.mix(['a'], [list(range(1, 9))], [[9, 10, 11, 12, 13]], accepted)
            split = mixer.last_native_counts()[0]
            counts.append(split)
            accepted = [5 if split == 2 else 0]
        self.assertGreater(counts[-50:].count(2), 30)
        self.assertGreater(len(set(counts)), 1)

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
