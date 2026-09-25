# SPDX-License-Identifier: Apache-2.0
"""NVFP4 checkpoint quantizer (suffix_hybrid.tools.quantize_nvfp4): recipe,
exclusions, bit-exactness vs the kernel reference, and an end-to-end write of
a tiny synthetic Qwen3.5-named checkpoint (CPU, --calib none)."""
import json

import numpy as np
import pytest
import torch
from suffix_hybrid.kernels import nvfp4_gemm as ng
from suffix_hybrid.tools import calib_prompts
from suffix_hybrid.tools import quantize_nvfp4 as q

L = "model.language_model.layers.3."


@pytest.mark.parametrize("key,group", [
    (L + "self_attn.q_proj.weight", L + "self_attn.qkv"),
    (L + "self_attn.v_proj.weight", L + "self_attn.qkv"),
    (L + "self_attn.o_proj.weight", L + "self_attn.o_proj"),
    (L + "mlp.up_proj.weight", L + "mlp.gate_up"),
    (L + "mlp.down_proj.weight", L + "mlp.down_proj"),
    (L + "linear_attn.in_proj_qkv.weight", L + "linear_attn.in_proj_qkvz"),
    (L + "linear_attn.in_proj_z.weight", L + "linear_attn.in_proj_qkvz"),
    (L + "linear_attn.out_proj.weight", L + "linear_attn.out_proj"),
])
def test_quantized_allowlist_and_fused_groups(key, group):
    assert q.classify(key)[2] == group


@pytest.mark.parametrize("key", [
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    L + "linear_attn.in_proj_b.weight",
    L + "linear_attn.in_proj_a.weight",
    L + "linear_attn.conv1d.weight",
    L + "linear_attn.A_log",
    L + "linear_attn.dt_bias",
    L + "linear_attn.norm.weight",
    L + "input_layernorm.weight",
    L + "self_attn.q_norm.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.merger.linear_fc1.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    L + "self_attn.q_proj.bias",
])
def test_everything_else_stays_high_precision(key):
    assert q.classify(key) is None


def test_quant_config_vllm_schema():
    c = q.quant_config()
    assert c["quant_method"] == "modelopt"
    assert c["quantization"]["quant_algo"] == "NVFP4"
    assert c["quantization"]["group_size"] == 16
    ex = c["quantization"]["exclude_modules"]
    assert "visual" in ex and "lm_head" in ex and "in_proj_b" in ex and "mtp" in ex


def test_weight_quant_bit_exact_vs_kernel_reference():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.03
    amax = float(w.abs().max())
    packed, sf, ws2 = q.quantize_weight(w, amax)
    ref_packed, ref_sf_bits, _ = ng.quantize(w.numpy(), np.float32(2688.0 / amax))
    np.testing.assert_array_equal(packed.numpy(), ref_packed)
    np.testing.assert_array_equal(sf.view(torch.uint8).numpy(), ref_sf_bits)
    assert ws2.shape == () and abs(float(ws2) / (amax / 2688.0) - 1) < 1e-6
    deq = q.dequantize_weight(packed, sf, ws2)
    rel = float((deq - w).norm() / w.norm())
    assert rel < 0.12


def test_rne_ties_match_e2m1_grid():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, 7.0, -9.0])
    codes = q.e2m1_codes(x)
    lut = [0, .5, 1, 1.5, 2, 3, 4, 6]
    vals = [lut[c & 7] * (-1 if c & 8 else 1) for c in codes.tolist()]
    assert vals == [0, 1, 1, 2, 2, 4, 4, -0.0, 6, -6]


def test_calibration_prompts_shipped():
    assert len(calib_prompts.PROMPTS) == 64
    assert len(set(calib_prompts.PROMPTS)) == 64


def _tiny_checkpoint(d):
    pytest.importorskip("safetensors")  # in the vLLM image; not in the CPU CI venv
    from safetensors.torch import save_file
    torch.manual_seed(1)
    t = {
        L + "self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        L + "self_attn.k_proj.weight": torch.randn(32, 64, dtype=torch.bfloat16) * 3,
        L + "self_attn.v_proj.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        L + "linear_attn.in_proj_b.weight": torch.randn(8, 64, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(16, 64, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(48, 16, dtype=torch.bfloat16),
    }
    save_file(t, str(d / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    (d / "tokenizer_config.json").write_text("{}")
    return t


def test_end_to_end_write(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    orig = _tiny_checkpoint(src)
    assert q.main(["--src", str(src), "--out", str(out), "--calib", "none"]) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quantization"]["quant_algo"] == "NVFP4"
    assert (out / "hf_quant_config.json").is_file() and (out / "tokenizer_config.json").is_file()
    man = json.loads((out / "suffix_quant_manifest.json").read_text())
    assert man["stats"]["quantized"] == 3 and man["stats"]["copied"] == 3
    assert set(man["source_sha256"]) == {"config.json", "model-00001-of-00001.safetensors"}
    assert man["recipe"]["exclude_modules"] == q.EXCLUDE_MODULES
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    from safetensors import safe_open
    with safe_open(str(out / "model-00001-of-00001.safetensors"), "pt") as st:
        assert set(idx["weight_map"]) == set(st.keys())
        wq = st.get_tensor(L + "self_attn.q_proj.weight")
        assert wq.dtype == torch.uint8 and wq.shape == (64, 32)
        assert st.get_tensor(L + "self_attn.q_proj.weight_scale").dtype == torch.float8_e4m3fn
        # fused q/k/v share ONE global weight scale (vLLM takes the max)
        s2 = {st.get_tensor(L + f"self_attn.{p}_proj.weight_scale_2").item() for p in "qkv"}
        assert len(s2) == 1
        amax_qkv = max(float(orig[L + f"self_attn.{p}_proj.weight"].float().abs().max())
                       for p in "qkv")
        assert abs(s2.pop() - amax_qkv / 2688.0) < 1e-9
        for k in ("lm_head.weight", L + "linear_attn.in_proj_b.weight",
                  "model.visual.blocks.0.attn.qkv.weight"):
            assert torch.equal(st.get_tensor(k), orig[k])  # byte-identical copy


def test_refuses_nonempty_out(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (out / "x").write_text("x")
    _tiny_checkpoint(src)
    with pytest.raises(SystemExit, match="not empty"):
        q.main(["--src", str(src), "--out", str(out), "--calib", "none"])
