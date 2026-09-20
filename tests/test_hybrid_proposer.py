# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HybridProposer — no torch/vllm/numpy required."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from suffix_hybrid.hybrid_proposer import HybridProposer


class FakeInputBatch:
    """Duck-typed stand-in for vllm.v1.worker.gpu_input_batch.InputBatch."""

    def __init__(self, rows, prompt_lens, ids=None):
        # rows: list[list[int]] token ids (prompt+generated so far)
        self.rows = rows
        self.req_ids = list(ids) if ids is not None else list(range(len(rows)))
        self.num_tokens_no_spec = [len(r) for r in rows]
        self.req_id_to_index = {rid: i for i, rid in enumerate(self.req_ids)}
        self.num_prompt_tokens = prompt_lens
        self.token_ids_cpu = _FakeTensor(rows)


class _FakeTensor:
    """Duck-typed 2D token tensor: indexable, rows are list-convertible."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        return _FakeRow(self.rows[idx])

    def numpy(self):
        return self


class _FakeRow:
    def __init__(self, row):
        self.row = row

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            return self.row[sl]
        return self.row[sl]

    def tolist(self):
        return list(self.row)


def reset_env():
    for k in [k for k in os.environ if k.startswith("SUFFIX_HYBRID")]:
        os.environ.pop(k)


class _Fake2DTensor:
    """2-D token_ids_cpu stand-in with a .shape (tensor introspection)."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), max((len(r) for r in rows), default=0))

    def __getitem__(self, idx):
        return _FakeRow(self.rows[idx])


class _Fake1DTensor:
    """1-D num_tokens_no_spec stand-in with a .shape."""

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def tolist(self):
        return list(self.values)


class TestHybridProposer(unittest.TestCase):
    def setUp(self):
        reset_env()

    def _cfg(self):
        class Spec:
            num_speculative_tokens = 4

        class Model:
            max_model_len = 256

        class Cfg:
            speculative_config = Spec()
            model_config = Model()

        return Cfg()

    def test_no_vllm_import_needed(self):
        # Importing/constructing must work without vllm installed.
        p = HybridProposer(self._cfg())
        self.assertEqual(p.num_speculative_tokens, 4)

    def test_partial_prefill_rows_return_empty(self):
        p = HybridProposer(self._cfg())
        batch = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
        out = p.propose(4, batch, [[]])
        self.assertEqual(out, [[]])

    def test_registers_prompt_and_finalizes_on_disappearance(self):
        p = HybridProposer(self._cfg())
        batch = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8], ids=["r0"])
        out = p.propose(4, batch, [[9]])
        self.assertEqual(len(out), 1)
        # Sequence is finalized only once the request has DISAPPEARED and
        # another propose call runs.
        self.assertEqual(p.suffix_cache.stats()["num_sequences"], 0)
        batch2 = FakeInputBatch([[9, 9, 9, 9, 9, 9, 9, 9]], [8], ids=["r1"])
        p.propose(4, batch2, [[1]])
        self.assertEqual(p.suffix_cache.stats()["num_sequences"], 1)

    def test_cross_request_improvement(self):
        p = HybridProposer(self._cfg())
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Request 0 generates [9, 10] and disappears; its full output enters
        # the corpus at the end of the NEXT propose call.
        batch = FakeInputBatch([seq[:8]], [8], ids=["r0"])
        p.propose(4, batch, [[9, 10]])
        # Request 1 arrives (req 0 absent -> finalized at end of this call).
        batch1 = FakeInputBatch([[7, 7, 7, 7, 7, 7, 7, 7]], [8], ids=["r1"])
        out1 = p.propose(4, batch1, [[1]])
        # Request 1 continues; suffix cache now knows the schema. In real vLLM the
        # just-sampled token is already present in token_ids_cpu.
        batch2 = FakeInputBatch([seq[:9]], [8], ids=["r1"])
        out2 = p.propose(4, batch2, [[9]])
        self.assertEqual(out2[0], [10])

    def test_v029_positional_shape_drafts(self):
        """vLLM 0.29 custom_class: propose(sampled, num_tokens_no_spec,
        token_ids_cpu, slot_mappings=...) — no InputBatch. Regression for
        the always-empty-drafts bug (introspection found no req_ids)."""
        p = HybridProposer(self._cfg())
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Step 1: request r starts, one token sampled.
        rows = [seq[:8]]
        out = p.propose([[9]], _Fake1DTensor([8]), _Fake2DTensor(rows))
        self.assertEqual(len(out), 1)
        # No corpus yet -> no draft (or whatever suffix has), but NOT the
        # blanket empty-from-failed-introspection: assert the row was
        # registered by checking stats.
        self.assertEqual(p.get_stats()["active_requests"], 1)

        # Step 2: same request continues (8 -> 9 tokens), corpus built from
        # a finished neighbour below. Feed a finished request first.
        p2 = HybridProposer(self._cfg())
        # Request A finishes: [1..8] prompt, samples [9, 10].
        p2.propose([[9, 10]], _Fake1DTensor([8]), _Fake2DTensor([seq[:8]]))
        # New request B on the same row (count reset 10 -> 8): row re-keyed,
        # A finalized into corpus.
        out = p2.propose([[9]], _Fake1DTensor([8]), _Fake2DTensor([seq[:8]]))
        # B continues: 8 -> 9 tokens, context [1..9] — corpus knows [10].
        out = p2.propose([[9]], _Fake1DTensor([9]), _Fake2DTensor([seq[:9]]))
        self.assertEqual(out[0], [10])

    def test_v029_row_adapter_continuity(self):
        """The persistent row adapter: consecutive steps on the same row
        keep ONE request id; a count reset re-keys the row."""
        from suffix_hybrid.hybrid_proposer import _RowAdapter

        a = _RowAdapter(None, _Fake1DTensor([8]), 1)
        a.refresh([[9]])            # step 1: new request, sampled len 1
        rid1 = a.req_ids[0]
        a.update(None, _Fake1DTensor([9]), 1)
        a.refresh([[10]])           # 9 == 8 + 1 -> same request
        self.assertEqual(a.req_ids[0], rid1)
        a.update(None, _Fake1DTensor([8]), 1)
        a.refresh([[5]])            # 8 != 9 + 1 -> new request
        self.assertNotEqual(a.req_ids[0], rid1)

    def test_max_model_len_rows_get_empty_draft(self):
        class Big:
            class speculative_config:
                num_speculative_tokens = 4

            class model_config:
                max_model_len = 10

        p = HybridProposer(Big())
        batch = FakeInputBatch([[1] * 10], [10])
        out = p.propose(4, batch, [[5]])
        self.assertEqual(out, [[]])

    def test_signature_variants(self):
        p = HybridProposer(self._cfg())
        batch = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
        # Newer main: (k, input_batch, sampled, slot_mappings=...)
        out = p.propose(4, batch, [[9]], slot_mappings=None)
        self.assertEqual(len(out), 1)
        # Some versions: (input_batch, sampled)
        batch2 = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
        out2 = p.propose(batch2, [[9]])
        self.assertEqual(len(out2), 1)
        # Keyword form
        batch3 = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
        out3 = p.propose(
            num_speculative_tokens=4, input_batch=batch3, sampled_token_ids=[[9]]
        )
        self.assertEqual(len(out3), 1)

    def test_acceptance_telemetry_and_get_stats(self):
        p = HybridProposer(self._cfg())
        # Seed the cache so a real draft is produced (see improvement test).
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p.propose(4, FakeInputBatch([seq[:8]], [8], ids=["r0"]), [[9, 10]])  # req 0
        p.propose(4, FakeInputBatch([[7] * 8], [8], ids=["r1"]), [[1]])      # req 1
        out = p.propose(4, FakeInputBatch([seq[:9]], [8], ids=["r1"]), [[9]])  # cont.
        draft = out[0]
        self.assertEqual(draft, [10])
        # Next step: sampled ids start with our draft -> accepted.
        p.propose(4, FakeInputBatch([seq[:10]], [8], ids=["r1"]), [draft[:1]])
        stats = p.get_stats()
        self.assertGreaterEqual(stats["proposed"], len(draft))
        self.assertGreaterEqual(stats["accepted"], 1)
        self.assertIn("acceptance_rate", stats)
        self.assertIn("cache", stats)

    def test_stats_file_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stats.jsonl")
            os.environ["SUFFIX_HYBRID_STATS_FILE"] = path
            os.environ["SUFFIX_HYBRID_STATS_INTERVAL"] = "1"
            p = HybridProposer(self._cfg())
            for _ in range(3):
                batch = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
                p.propose(4, batch, [[9]])
            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 3)
            for rec in lines:
                for key in ("ts", "proposed", "accepted", "suffix_proposals",
                            "ngram_proposals", "avg_match_len"):
                    self.assertIn(key, rec)

    def test_ngram_disabled(self):
        os.environ["SUFFIX_HYBRID_USE_NGRAM"] = "0"
        p = HybridProposer(self._cfg())
        self.assertFalse(p.use_ngram)
        self.assertIsNone(p._get_ngram_proposer())

    def test_ngram_fallback_safe_without_vllm(self):
        # _native_ngram_draft must not blow up when vllm/numpy are missing.
        p = HybridProposer(self._cfg())
        p._ngram_proposer = object()  # pretend we have one
        batch = FakeInputBatch([[1, 2, 3, 4, 5, 6, 7, 8]], [8])
        draft, score = p._native_ngram_draft(
            object(), [[9]], batch.num_tokens_no_spec, batch, 0, 4
        )
        self.assertEqual((draft, score), ([], 0.0))

    def test_no_input_batch_returns_empties(self):
        p = HybridProposer(self._cfg())
        self.assertEqual(p.propose(4, None, [[1], [2]]), [[], []])

    def test_env_override_max_spec_tokens(self):
        os.environ["SUFFIX_HYBRID_MAX_SPEC_TOKENS"] = "2"
        p = HybridProposer(self._cfg())
        self.assertEqual(p.num_speculative_tokens, 2)


class TestWrapMode(unittest.TestCase):
    def setUp(self):
        reset_env()

    def test_install_off_by_default(self):
        from suffix_hybrid import wrap

        self.assertFalse(wrap.install())
        self.assertFalse(any(
            getattr(f, "_suffix_hybrid_wrap", False) for f in sys.meta_path
        ))

    def test_install_and_patch(self):
        os.environ["SUFFIX_HYBRID_WRAP"] = "1"
        try:
            # Build a fake "vllm.v1.spec_decode.eagle" module.
            import types

            mod = types.ModuleType("vllm.v1.spec_decode.eagle")

            class EagleProposer:
                def propose(self, *a, **k):
                    return [[5, 6], []]

                def load_model(self, *a, **k):
                    pass

            mod.EagleProposer = EagleProposer
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
            sys.modules["vllm.v1.spec_decode"] = types.ModuleType("vllm.v1.spec_decode")
            sys.modules["vllm.v1.spec_decode.eagle"] = mod

            from suffix_hybrid import wrap

            try:
                self.assertTrue(wrap.install())
                # Any vllm import consults the watchdog -> module patched.
                import importlib

                importlib.invalidate_caches()
                wrap.patch_all()
                patched = sys.modules["vllm.v1.spec_decode.eagle"].EagleProposer
                self.assertTrue(getattr(patched, "_suffix_hybrid_arbitrated", False))
                # isinstance still holds (subclass).
                self.assertTrue(issubclass(patched, EagleProposer))
                # propose still works and returns the native draft (no
                # suffix evidence -> native rows preserved).
                out = patched().propose(4, None, [[1]])
                self.assertEqual(out, [[5, 6], []])
            finally:
                sys.meta_path = [f for f in sys.meta_path
                                 if not getattr(f, "_suffix_hybrid_wrap", False)]
                for name in ("vllm", "vllm.v1", "vllm.v1.spec_decode",
                             "vllm.v1.spec_decode.eagle"):
                    sys.modules.pop(name, None)
                import suffix_hybrid.wrap as w
                w._TARGETS  # module still usable
        finally:
            os.environ.pop("SUFFIX_HYBRID_WRAP", None)

    def test_sitecustomize_is_noop_without_env(self):
        import subprocess

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import suffix_hybrid.wrap; "
            "print(suffix_hybrid.wrap.install())" % repo
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")

    def test_sitecustomize_never_breaks_pod(self):
        # Even with SUFFIX_HYBRID_WRAP=1 and no vllm present, a Python
        # interpreter with the repo on PYTHONPATH must start cleanly.
        import subprocess

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["SUFFIX_HYBRID_WRAP"] = "1"
        env["PYTHONPATH"] = repo
        out = subprocess.run(
            [sys.executable, "-c", "print('ok')"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()