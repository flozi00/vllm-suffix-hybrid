"""Native contract tests use real NumPy buffers, not fake tensor APIs."""
import os
import unittest
from suffix_hybrid.suffix_cache import SuffixCache


class NativeCacheTests(unittest.TestCase):
    def setUp(self):
        # Keep the conftest-pinned INDEX_N (file order must not change
        # lookup semantics); reset every other knob for determinism.
        for key in list(os.environ):
            if key.startswith('SUFFIX_HYBRID') and key != 'SUFFIX_HYBRID_INDEX_N':
                del os.environ[key]
        os.environ.setdefault('SUFFIX_HYBRID_INDEX_N', '2')

    def tearDown(self):
        for key in list(os.environ):
            if key.startswith('SUFFIX_HYBRID') and key != 'SUFFIX_HYBRID_INDEX_N':
                del os.environ[key]
        os.environ['SUFFIX_HYBRID_INDEX_N'] = '2'

    def test_oversized_sequence_allocation_is_bounded(self):
        os.environ['SUFFIX_HYBRID_MAX_CACHED_TOKENS'] = '16'
        cache = SuffixCache()
        cache.add_sequence(list(range(100_000)))
        self.assertEqual(cache.stats()['cached_tokens'], 16)
        self.assertLessEqual(cache.stats()['allocated_token_capacity'], 16)
        self.assertEqual(cache.speculate(list(range(99984, 99992)), 2)[0], [99992, 99993])

    def test_native_cache_and_bounded_index(self):
        os.environ['SUFFIX_HYBRID_MAX_CACHED_REQUESTS'] = '2'
        os.environ['SUFFIX_HYBRID_MAX_CACHED_TOKENS'] = '20'
        cache = SuffixCache()
        self.assertEqual(type(cache).__module__, 'suffix_hybrid._native')
        for i in range(100):
            cache.add_sequence(list(range(i * 20, i * 20 + 10)))
        stats = cache.stats()
        self.assertLessEqual(stats['num_sequences'], 2)
        self.assertLessEqual(stats['cached_tokens'], 20)
        # n=2 (conftest-pinned): 10-token sequences index 9 bigrams each,
        # 2 live sequences => <= 18 keys. (Was 6 at n=8.)
        self.assertLessEqual(stats['num_index_keys'], 18)
        self.assertEqual(cache.speculate(list(range(8)), 2)[0], [])
        self.assertEqual(cache.speculate(list(range(1980, 1988)), 2)[0], [1988, 1989])
