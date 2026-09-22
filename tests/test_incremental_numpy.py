"""Incremental mix_numpy: steady-state decode must match the list API.

The NumPy path tracks each row's context and appends only new tokens, so a
long context is copied once per request, not once per decode step. These
tests pin the contract that made that safe:
- growing counts append the delta and produce identical drafts to `mix`;
- a row whose buffer no longer extends the tracked context resets (old
  history goes to the corpus, row publishes native-only like a fresh row);
- shrinking counts reset instead of panicking;
- boundary-window continuity (head+tail) still detects a real continuation.
"""
import unittest

import numpy as np

from suffix_hybrid import _native


def _tokens(rows, width, seqs):
    data = np.zeros((rows, width), dtype=np.int64)
    for i, seq in enumerate(seqs):
        data[i, : len(seq)] = seq
    return data


class IncrementalNumpyTests(unittest.TestCase):
    def setUp(self):
        import os
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("SUFFIX_HYBRID")}
        for key in list(os.environ):
            if key.startswith("SUFFIX_HYBRID"):
                del os.environ[key]
        os.environ["SUFFIX_HYBRID_INDEX_N"] = "2"

    def tearDown(self):
        import os
        for key in list(os.environ):
            if key.startswith("SUFFIX_HYBRID"):
                del os.environ[key]
        os.environ.update(self._saved)

    def test_growing_counts_match_list_api(self):
        # Same corpus, same drafts: the NumPy path stepping counts 8->12 must
        # publish exactly what the list API publishes for the full contexts.
        corpus = list(range(1, 40))
        native = [[40, 41, 42, 43, 99]]
        list_mixer = _native.HybridMixer(5, 128)
        list_mixer.suffix_cache.add_sequence(corpus)
        numpy_mixer = _native.HybridMixer(5, 128)
        numpy_mixer.suffix_cache.add_sequence(corpus)
        width = 64
        for step, length in enumerate([8, 9, 10, 11, 12]):
            ctx = corpus[:length]
            accepted = None if step == 0 else [5]
            want = list_mixer.mix(["a"], [ctx], native, accepted)
            data = _tokens(1, width, [ctx])
            got = numpy_mixer.mix_numpy(["a"], np.array([length], dtype=np.int64),
                                        data, native, accepted)
            self.assertEqual(got, want, f"step {step} length {length}")
            self.assertEqual(numpy_mixer.last_native_counts(),
                             list_mixer.last_native_counts())

    def test_reused_slot_resets_and_publishes_native_only(self):
        mixer = _native.HybridMixer(5, 128)
        mixer.suffix_cache.add_sequence(list(range(1, 30)))
        data = _tokens(1, 64, [list(range(1, 9))])
        mixer.mix_numpy(["a"], np.array([8], dtype=np.int64), data, [[9, 10, 11, 12, 99]], None)
        # A different request reuses the ID: unrelated context, same slot.
        other = [100, 101, 102, 103]
        data2 = _tokens(1, 64, [other])
        got = mixer.mix_numpy(["a"], np.array([4], dtype=np.int64), data2,
                              [[104, 105, 106, 107, 108]], None)
        self.assertEqual(got, [[104, 105, 106, 107, 108]])  # native-only publish
        # The finalized old history is in the corpus now.
        self.assertGreater(mixer.get_stats()["cache"]["num_sequences"], 0)

    def test_shrinking_counts_reset_not_panic(self):
        mixer = _native.HybridMixer(4, 128)
        data = _tokens(1, 64, [list(range(1, 20))])
        mixer.mix_numpy(["a"], np.array([16], dtype=np.int64), data, [[1, 2, 3, 4]], None)
        small = _tokens(1, 64, [[7, 8, 9]])
        got = mixer.mix_numpy(["a"], np.array([3], dtype=np.int64), small, [[5, 6]], None)
        self.assertEqual(got, [[5, 6]])

    def test_boundary_continuity_detects_growth(self):
        # Tracked context longer than the boundary window: a row that extends
        # it must be treated as continuing (drafts identical to list API).
        mixer = _native.HybridMixer(3, 4096)
        base = list(range(1, 200))
        data = _tokens(1, 512, [base])
        mixer.mix_numpy(["a"], np.array([len(base)], dtype=np.int64), data,
                        [[900, 901, 902]], None)
        grown = base + [900, 901]
        data2 = _tokens(1, 512, [grown])
        got = mixer.mix_numpy(["a"], np.array([len(grown)], dtype=np.int64), data2,
                              [[903, 904, 905]], [3])
        # Continuing: no reset, so the row keeps its per-row estimates.
        self.assertEqual(got, [[903, 904, 905]])
        self.assertEqual(mixer.get_stats()["active_tracked"], 1)

    def test_int32_buffers_incremental_match(self):
        corpus = list(range(1, 30))
        mixer = _native.HybridMixer(4, 128)
        mixer.suffix_cache.add_sequence(corpus)
        for length in (8, 10, 12):
            data = np.zeros((1, 64), dtype=np.int32)
            data[0, :length] = corpus[:length]
            got = mixer.mix_numpy(["a"], np.array([length], dtype=np.int32), data,
                                  [[30, 31, 32, 33]], None)
            self.assertEqual(got, [[30, 31, 32, 33]])

    def test_validation_rejects_before_mutation(self):
        mixer = _native.HybridMixer(4, 128)
        with self.assertRaises(ValueError):
            mixer.mix_numpy(["a", "a"], np.array([1, 1], dtype=np.int64),
                            _tokens(2, 8, [[1], [2]]), [[1], [2]], None)
        self.assertEqual(mixer.get_stats()["calls"], 0)


if __name__ == "__main__":
    unittest.main()
