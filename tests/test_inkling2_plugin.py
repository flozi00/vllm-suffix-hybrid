"""CPU gates for the Inkling 2 loop port: off means untouched, wrong vLLM or a drifted anchor refuses."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import suffix_hybrid
except ModuleNotFoundError as error:
    if error.name != "suffix_hybrid":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suffix_hybrid import inkling2_plugin


class Inkling2PluginTest(unittest.TestCase):
    def test_flag_off_is_a_no_op(self):
        with patch.dict(os.environ, {"SUFFIX_INKLING2_LOOP": "0"}), \
                patch.object(inkling2_plugin, "version", side_effect=AssertionError("must not look")):
            inkling2_plugin.register()

    def test_unvalidated_vllm_refuses(self):
        with patch.dict(os.environ, {"SUFFIX_INKLING2_LOOP": "1"}), \
                patch.object(inkling2_plugin, "version", return_value="0.34.0"):
            with self.assertRaisesRegex(RuntimeError, "pinned to vLLM"):
                inkling2_plugin.register()

    def test_drifted_anchor_refuses(self):
        def nightly():
            return 'if hf_config.model_type == "inkling2":'
        inkling2_plugin._anchor(nightly, 'model_type == "inkling2"')
        with self.assertRaisesRegex(RuntimeError, "anchor drifted"):
            inkling2_plugin._anchor(nightly, 'model_type in ("inkling2", "inkling2_loop")')

    def test_moe_tune_is_off_unless_armed_and_only_for_swept_shapes(self):
        from suffix_hybrid.inkling2.nvidia.ops import fast_sm120 as f
        with patch.object(f, "MOE_TUNE", False):
            self.assertIsNone(f.moe_config("nvfp4", 128, 2304, 2304))
        with patch.object(f, "MOE_TUNE", True):
            self.assertEqual(f.moe_config("nvfp4", 128, 2304, 2304), (128, 128, 256, 8, 3))
            self.assertIsNone(f.moe_config("nvfp4", 8, 2304, 2304))  # decode tiles: dropped
            self.assertIsNone(f.moe_config("nvfp4", 64, 2304, 2304))  # <10%: dropped
            self.assertEqual(f.moe_config("mxfp8", 32, 2304, 2304), (128, 32, 128, 4, 2))

    def test_vendored_package_has_no_upstream_self_imports(self):
        root = Path(inkling2_plugin.__file__).parent / "inkling2"
        stale = [str(p) for p in root.rglob("*.py")
                 if "vllm.models.inkling2" in p.read_text() and p.name != "quant_config.py"]
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
