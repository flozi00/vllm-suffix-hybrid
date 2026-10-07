"""Native-packed decision weights without head/index contamination or CUDA."""
import json
import os
from pathlib import Path
import tempfile

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from suffix_hybrid.tools import quantize_nvfp4 as shared
from suffix_hybrid.tools import quantize_pplx_nvfp4 as quantizer

PREFIX = "language_model.layers.3."


def fixture_checkpoint(root):
    torch.manual_seed(17)
    tensors = {PREFIX + f"self_attn.{part}_proj.weight":
               torch.randn(128, 128, dtype=torch.bfloat16) * scale
               for part, scale in (("q", 1), ("k", 3), ("v", 2))}
    tensors.update({PREFIX + "linear_attn.in_proj_b.weight":
                    torch.randn(48, 128, dtype=torch.bfloat16),
                    PREFIX + "linear_attn.in_proj_qkv.weight": torch.randn(128, 128, dtype=torch.bfloat16),
                    PREFIX + "linear_attn.in_proj_z.weight": torch.randn(128, 128, dtype=torch.bfloat16) * 4,
                    PREFIX + "linear_attn.A_log": torch.randn(48, dtype=torch.float32),
                    "language_model.embed_tokens.weight": torch.randn(256, 128, dtype=torch.bfloat16),
                    "visual.blocks.0.attn.qkv.weight": torch.randn(128, 128, dtype=torch.bfloat16)})
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(root / name), metadata={"format": "pt"})
    save_file({"weight": torch.zeros(255, 5120, dtype=torch.bfloat16)}, str(root / "readout.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: name for key in tensors}}))
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5Model"]}))
    (root / "decision_config.json").write_text(json.dumps({"format_version": 1, "pooling": "last",
        "attention_mode": "noncausal_full_attention", "codes": [f"C{i}" for i in range(255)],
        "token_ids": list(range(255)), "temperature": 1.0087417621345625}))
    (root / "tokenizer.json").write_text('{"unchanged":"tokenizer"}')
    (root / "release-manifest.json").write_text('{"source_revision":"original"}')
    (root / ".complete.json").write_text('{"must_not_copy":true}')
    return tensors


def test_native_weight_only_conversion_keeps_head_and_fused_scales(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    original = fixture_checkpoint(source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA touched")))
    output = tmp_path / "output"
    stats = quantizer.convert(source, output, row_chunk=17, shard_mib=1, threads=1)
    assert stats["quantized"] == 5 and stats["copied"] == 4
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert "weight" not in index["weight_map"]
    assert "readout.safetensors" not in index["weight_map"].values()
    assert not any(name.endswith(".input_scale") for name in index["weight_map"])
    for name in ("readout.safetensors", "decision_config.json", "tokenizer.json"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert (output / "source-release-manifest.json").read_bytes() == (source / "release-manifest.json").read_bytes()
    assert not (output / "release-manifest.json").exists()
    assert not (output / ".complete.json").exists()
    groups = {}
    for key, tensor in original.items():
        group = quantizer.classify(key)
        if group:
            groups[group] = max(groups.get(group, 0), float(tensor.float().abs().amax()))
    expected_keys = set(original)
    for key in original:
        if quantizer.classify(key):
            expected_keys.update({key[:-7] + ".weight_scale", key[:-7] + ".weight_scale_2"})
    assert set(index["weight_map"]) == expected_keys
    for key, file in index["weight_map"].items():
        with safe_open(str(output / file), framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(key)
            if key in original and quantizer.classify(key):
                packed, scales, scale2 = shared.quantize_weight(original[key], groups[quantizer.classify(key)])
                assert torch.equal(tensor, packed)
                assert torch.equal(shard.get_tensor(key[:-7] + ".weight_scale").view(torch.uint8), scales.view(torch.uint8))
                assert torch.equal(shard.get_tensor(key[:-7] + ".weight_scale_2"), scale2)
            elif key in original:
                assert tensor.dtype == original[key].dtype
                assert torch.equal(tensor.view(torch.uint8), original[key].view(torch.uint8))
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["quantization"]["quant_algo"] == "W4A16_NVFP4"
    assert "readout" in config["quantization_config"]["quantization"]["exclude_modules"]
    manifest = json.loads((output / "suffix_quant_manifest.json").read_text())
    assert manifest["source_sha256"]["readout.safetensors"] == shared.sha256_file(source / "readout.safetensors")


def test_rejects_head_in_backbone_index_and_nonempty_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    fixture_checkpoint(source)
    output = tmp_path / "output"
    output.mkdir()
    (output / "owner-file").write_text("keep")
    with pytest.raises(ValueError, match="refusing overwrite"):
        quantizer.convert(source, output)
    (output / "owner-file").unlink()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"]["weight"] = "readout.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="readout must remain separate"):
        quantizer.convert(source, output)


def test_publication_stays_inside_output_mount_and_never_replaces_mount_root(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "mounted-output"
    source.mkdir()
    output.mkdir()
    fixture_checkpoint(source)
    real_mkdtemp, real_replace, real_rmdir = tempfile.mkdtemp, os.replace, Path.rmdir
    moved = []
    def mounted_mkdtemp(*args, **kwargs):
        assert Path(kwargs["dir"]) == output, "staging escaped the mounted output filesystem"
        return real_mkdtemp(*args, **kwargs)
    def mounted_replace(src, dst):
        src, dst = Path(src), Path(dst)
        assert src.parent.parent == output and dst.parent == output
        assert dst != output, "attempted to replace output mount root"
        moved.append(dst.name)
        return real_replace(src, dst)
    def mounted_rmdir(path):
        assert path != output, "attempted to remove output mount root"
        return real_rmdir(path)
    monkeypatch.setattr(tempfile, "mkdtemp", mounted_mkdtemp)
    monkeypatch.setattr(os, "replace", mounted_replace)
    monkeypatch.setattr(Path, "rmdir", mounted_rmdir)
    quantizer.convert(source, output, threads=1)
    assert moved[-3:] == ["hf_quant_config.json", "config.json", "model.safetensors.index.json"]
    assert not any(path.name.startswith(".pending-") for path in output.iterdir())


def test_partial_publish_failure_cleans_own_files_and_leaves_mount_root(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "source", tmp_path / "mounted-output"
    source.mkdir()
    output.mkdir()
    fixture_checkpoint(source)
    real_replace = os.replace
    moves = 0
    def failing_replace(src, dst):
        nonlocal moves
        moves += 1
        if moves == 3:
            raise OSError("simulated publication failure")
        return real_replace(src, dst)
    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        quantizer.convert(source, output, threads=1)
    assert output.is_dir() and not list(output.iterdir())
    assert "[quantize-nvfp4] DONE" not in capsys.readouterr().out


@pytest.mark.parametrize("key", [PREFIX + "linear_attn.in_proj_a.weight",
                                PREFIX + "linear_attn.in_proj_b.weight",
                                PREFIX + "linear_attn.conv1d.weight", PREFIX + "input_layernorm.weight",
                                "visual.blocks.0.attn.qkv.weight", "language_model.embed_tokens.weight",
                                "readout.weight", "weight"])
def test_excluded_parameters_are_not_quantized(key):
    assert quantizer.classify(key) is None
