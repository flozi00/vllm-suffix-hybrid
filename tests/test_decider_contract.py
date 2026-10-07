"""CPU gates: reject behavior-changing execution choices before allocation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

try:
    import suffix_hybrid
except ModuleNotFoundError as error:
    if error.name != "suffix_hybrid":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suffix_hybrid.decider_contract import (
    checkpoint_contract, configure_model, validate_engine, validate_tokenizer)
from suffix_hybrid.decider_plugin import register


class Dtype:
    def __str__(self):
        return "torch.bfloat16"


def engine_config():
    return NS(model_config=NS(runner_type="pooling", max_model_len=8192,
                             pooler_config=NS(seq_pooling_type="LAST"),
                             enforce_eager=True, dtype=Dtype()),
              cache_config=NS(enable_prefix_caching=False, mamba_cache_mode="none",
                              mamba_ssm_cache_dtype="float32"),
              scheduler_config=NS(enable_chunked_prefill=False,
                                  max_num_batched_tokens=8192),
              parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1,
                                 enable_dbo=False, ubatch_size=0),
              speculative_config=None, use_v2_model_runner=True)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.saved = dict(format_version=1,
                          attention_mode="noncausal_full_attention", pooling="last",
                          temperature=1.0087417621345625,
                          codes=["C" + str(i) for i in range(255)],
                          token_ids=list(range(255)))
        (self.root / "readout.safetensors").write_bytes(b"fixture")

    def tearDown(self):
        self.directory.cleanup()

    def write(self, **changes):
        value = self.saved | changes
        (self.root / "decision_config.json").write_text(json.dumps(value))

    def test_rejects_altered_checkpoint_semantics_and_bad_calibration(self):
        for changes in (dict(attention_mode="causal"), dict(pooling="mean"),
                        dict(temperature=0), dict(temperature=float("nan")),
                        dict(token_ids=[0] * 255), dict(codes=["A"] * 255)):
            with self.subTest(changes=changes):
                self.write(**changes)
                with self.assertRaises(ValueError):
                    checkpoint_contract(self.root)

    def test_updates_outer_and_text_noncausal_without_changing_layer_types(self):
        self.write()
        text = NS(hidden_size=5120, layer_types=["linear_attention", "full_attention"])
        model = NS(model=self.root, hf_config=NS(), hf_text_config=text)
        configure_model(model)
        self.assertFalse(model.hf_config.is_causal)
        self.assertFalse(text.is_causal)
        self.assertEqual(text.layer_types, ["linear_attention", "full_attention"])
        self.assertEqual(model._suffix_decider_contract["temperature"], self.saved["temperature"])

    def test_rejects_engine_choices_that_break_bidirectional_or_fresh_state(self):
        validate_engine(engine_config())
        changes = [("cache_config", "enable_prefix_caching", True),
                   ("cache_config", "mamba_cache_mode", "align"),
                   ("cache_config", "mamba_ssm_cache_dtype", "bfloat16"),
                   ("scheduler_config", "enable_chunked_prefill", True),
                   ("scheduler_config", "max_num_batched_tokens", 1024),
                   ("model_config", "max_model_len", 16384),
                   ("model_config", "enforce_eager", False),
                   ("parallel_config", "enable_dbo", True),
                   ("parallel_config", "tensor_parallel_size", 2),
                   ("parallel_config", "pipeline_parallel_size", 2)]
        for parent, field, value in changes:
            with self.subTest(field=field):
                config = engine_config()
                setattr(getattr(config, parent), field, value)
                with self.assertRaises(ValueError):
                    validate_engine(config)
        config = engine_config()
        config.use_v2_model_runner = False
        with self.assertRaisesRegex(ValueError, "runner V2"):
            validate_engine(config)

    def test_accepts_native_disabled_microbatch_defaults_and_single_batch(self):
        # vLLM 0.30.0 ParallelConfig.ubatch_size defaults to 0; 0 and 1 both
        # select one physical batch when DBO is off. Only >1 enables ubatching.
        config = engine_config()
        self.assertEqual(config.parallel_config.ubatch_size, 0)
        validate_engine(config)
        config.parallel_config.ubatch_size = 1
        validate_engine(config)
        config.parallel_config.ubatch_size = 2
        with self.assertRaisesRegex(ValueError, "no DBO"):
            validate_engine(config)

    def test_plugin_off_does_not_import_vllm(self):
        with patch.dict("os.environ", {"SUFFIX_PPLX_DECIDER": "0"}):
            with patch("suffix_hybrid.decider_plugin.version", side_effect=AssertionError("imported runtime")):
                register()

    def test_armed_plugin_rejects_version_drift_before_import(self):
        with patch.dict("os.environ", {"SUFFIX_PPLX_DECIDER": "1"}):
            with patch("suffix_hybrid.decider_plugin.version", return_value="0.31.0"):
                with self.assertRaisesRegex(RuntimeError, "pinned"):
                    register()

    def test_rejects_tokenizer_boundary_changes(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "PREFIX"
            def encode(self, text, **kwargs):
                if text == "PREFIX":
                    return [999]
                if text.startswith("PREFIXC"):
                    return [999, int(text[7:])]
                return [int(text[1:])]
        validate_tokenizer(Tokenizer(), self.saved)
        bad = self.saved | {"token_ids": [254] + list(range(1, 255))}
        with self.assertRaises(ValueError):
            validate_tokenizer(Tokenizer(), bad)


if __name__ == "__main__":
    unittest.main()
