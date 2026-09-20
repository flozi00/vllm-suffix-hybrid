"""vLLM 0.29 custom_class contract: rows ALREADY include sampled IDs.

Legacy fake-InputBatch and count-as-identity tests were intentionally retired:
those asserted unsupported signatures and duplicated tokens. Acceptance must
come from vLLM, never equality of a later sampled token with a draft.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from suffix_hybrid.hybrid_proposer import HybridProposer


def config(max_len=256, k=4):
    return SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=k),
                           model_config=SimpleNamespace(max_model_len=max_len))


class TestHybridProposer(unittest.TestCase):
    def setUp(self):
        for key in list(os.environ):
            if key.startswith('SUFFIX_HYBRID'):
                del os.environ[key]
        os.environ['SUFFIX_HYBRID_INDEX_N'] = '2'
        os.environ['SUFFIX_HYBRID_NGRAM_MIN'] = '2'
        os.environ['SUFFIX_HYBRID_NGRAM_MAX'] = '4'

    def call(self, p, rows, sampled=None, dtype=np.int32):
        if sampled is None:
            sampled = [[r[-1]] if r else [] for r in rows]
        buf = np.full((len(rows), 256), -999, dtype=dtype)
        for i, row in enumerate(rows):
            buf[i, :len(row)] = row
        return p.propose(sampled, np.array([len(r) for r in rows], dtype=dtype), buf,
                         slot_mappings=None)

    def test_native_ngram_without_vllm(self):
        p = HybridProposer(config())
        self.assertEqual(self.call(p, [[1, 2, 3, 4, 1, 2]]), [[3, 4, 1, 2]])
        self.assertEqual(p.get_stats()['ngram_proposals'], 1)

    def test_no_duplicate_sampled_tokens(self):
        os.environ['SUFFIX_HYBRID_USE_NGRAM'] = '0'
        p = HybridProposer(config())
        self.call(p, [[1, 2, 3, 4]], [[3, 4]])
        self.call(p, [[7, 8, 9, 10]])
        self.assertEqual(p.suffix_cache.speculate([1, 2], 4)[0], [3, 4])
        self.assertEqual(p.suffix_cache.stats()['cached_tokens'], 4)
        self.assertEqual(self.call(p, [[1, 2, 3]]), [[4]])

    def test_same_counts_not_identity(self):
        p = HybridProposer(config())
        self.call(p, [[1, 2, 3]])
        self.call(p, [[8, 9, 10, 11]])
        self.call(p, [])
        self.assertEqual(p.suffix_cache.stats()['cached_tokens'], 7)
        self.assertEqual(p.suffix_cache.speculate([2, 3], 4)[0], [])

    def test_reorder_shrink_and_incremental_prefix(self):
        p = HybridProposer(config())
        self.call(p, [[1, 2, 3], [8, 9, 10]])
        self.call(p, [[8, 9, 10, 11], [1, 2, 3, 4]])
        self.assertEqual(p.suffix_cache.stats()['num_sequences'], 0)
        self.call(p, [[1, 2, 3, 4, 5]])
        self.assertEqual(p.suffix_cache.stats()['cached_tokens'], 4)
        self.call(p, [])
        self.assertEqual(p.suffix_cache.stats()['cached_tokens'], 9)
        self.assertEqual(p.get_stats()['active_requests'], 0)

    def test_empty_sampled_and_max_model_len(self):
        p = HybridProposer(config(max_len=7))
        row = [1, 2, 3, 4, 1, 2]
        self.assertEqual(self.call(p, [row], [[]]), [[]])
        self.assertEqual(self.call(p, [row]), [[3]])
        self.assertEqual(self.call(p, [row + [3]]), [[]])

    def test_int64_and_strided_buffers(self):
        p = HybridProposer(config())
        self.assertEqual(self.call(p, [[1, 2, 3, 4, 1, 2]], dtype=np.int64), [[3, 4, 1, 2]])
        a = np.array([[1, 99, 2, 99, 3, 99, 4, 99, 1, 99, 2, 99]], dtype=np.int32)[:, ::2]
        self.assertEqual(p.propose([[2]], np.array([6], dtype=np.int32), a), [[3, 4, 1, 2]])

    def test_invalid_bounds_fail_before_state_mutation(self):
        p = HybridProposer(config())
        for counts in ([-1], [5], []):
            with self.assertRaises(ValueError):
                p.propose([[1]], np.array(counts, dtype=np.int32), np.zeros((1, 4), dtype=np.int32))
        self.assertEqual(p.get_stats()['active_requests'], 0)

    def test_metrics_are_not_claimed_acceptance(self):
        p = HybridProposer(config())
        self.call(p, [[1, 2, 3, 4, 1, 2]])
        stats = p.get_stats()
        self.assertIsNone(stats['accepted'])
        self.assertIsNone(stats['acceptance_rate'])
        self.assertGreater(stats['native_time_ns'], 0)
        self.assertGreaterEqual(stats['total_proposer_time_ns'], stats['native_time_ns'])
        self.assertIn('rust', stats['version'])

    def test_stats_file_and_log(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
            os.environ['SUFFIX_HYBRID_STATS_FILE'] = d + '/stats.jsonl'
            os.environ['SUFFIX_HYBRID_STATS_INTERVAL'] = '1'
            os.environ['SUFFIX_HYBRID_LOG_INTERVAL'] = '1'
            p = HybridProposer(config())
            with self.assertLogs('suffix_hybrid', level='INFO') as logs:
                self.call(p, [[1, 2, 3]])
            with open(d + '/stats.jsonl') as f:
                rec = json.loads(f.readline())
            self.assertIn('native_time_ns', rec)
            self.assertIn('rust', logs.output[0])

    def test_variable_sampled_growth_uses_authoritative_rows(self):
        p = HybridProposer(config())
        self.call(p, [[1, 2, 3]], [[3]])
        self.call(p, [[1, 2, 3, 4, 5, 6]], [[4, 5, 6]])
        self.call(p, [[1, 2, 3, 4, 5, 6, 7]], [[7]])
        self.assertEqual(p.get_stats()['active_context_tokens'], 7)
        self.assertEqual(p.suffix_cache.stats()['num_sequences'], 0)
        self.call(p, [])
        self.assertEqual(p.suffix_cache.stats()['cached_tokens'], 7)

    def test_buffers_never_convert_to_python_lists(self):
        class NoList(np.ndarray):
            def tolist(self):
                raise AssertionError('padded array converted to list')
        p = HybridProposer(config())
        tokens = np.zeros((8, 200_000), dtype=np.int32).view(NoList)
        tokens[0, :6] = [1, 2, 3, 4, 1, 2]
        counts = np.array([6] + [-999] * 7, dtype=np.int32).view(NoList)
        self.assertEqual(p.propose([[2]], counts, tokens), [[3, 4, 1, 2]])

    def test_suffix_confidence_arbitrates_against_ngram(self):
        p = HybridProposer(config())
        p.suffix_cache.add_sequence([3, 4, 1, 2, 91, 92, 93, 94])
        self.assertEqual(self.call(p, [[1, 2, 3, 4, 1, 2]]), [[91, 92, 93, 94]])
        self.assertEqual(p.get_stats()['suffix_proposals'], 1)

    def test_wrap_explicitly_unsupported(self):
        from suffix_hybrid import wrap
        self.assertFalse(wrap.install())
        os.environ['SUFFIX_HYBRID_WRAP'] = '1'
        with self.assertRaisesRegex(RuntimeError, 'unsupported'):
            wrap.install()
        with self.assertRaisesRegex(RuntimeError, 'unsupported'):
            HybridProposer(config())
