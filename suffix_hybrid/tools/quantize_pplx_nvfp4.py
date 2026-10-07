"""CPU-stream pplx decision backbone to native vLLM W4A16_NVFP4.

Reuse the existing NVFP4 pack/scales; activations, vision, embeddings, GDN
gates/norm/state and the separately saved decision readout stay high precision.
No model construction, CUDA allocation, activation calibration or download.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from suffix_hybrid.decider_contract import checkpoint_contract
from suffix_hybrid.tools import quantize_nvfp4 as shared

QUANT_RE = re.compile(
    r"^language_model\.layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$")
EXCLUDE_MODULES = shared.EXCLUDE_MODULES + ["readout"]


def classify(key):
    match = QUANT_RE.fullmatch(key)
    if match is None:
        return None
    module = match.group(2)
    base = shared.module_name(key)
    return base[:-len(module)] + shared.FUSED.get(module, module)


def quant_config():
    return {"quant_method": "modelopt",
            "producer": {"name": "suffix_hybrid.tools.quantize_pplx_nvfp4", "version": "1"},
            "quantization": {"quant_algo": "W4A16_NVFP4", "group_size": 16,
                             "kv_cache_quant_algo": None,
                             "exclude_modules": EXCLUDE_MODULES}}


def source_layout(src):
    checkpoint_contract(src)
    config = json.loads((src / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("quantization_config"):
        raise ValueError("expected an unquantized Qwen3.5 decision backbone")
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source backbone requires a nonempty safetensors index")
    files = {}
    for key, name in weight_map.items():
        if (not isinstance(key, str) or not isinstance(name, str)
                or Path(name).name != name or not name.endswith(".safetensors")
                or name == "readout.safetensors"):
            raise ValueError("invalid backbone index; readout must remain separate")
        files.setdefault(name, set()).add(key)
    if not any(classify(key) for key in weight_map):
        raise ValueError("source index has no supported bare language_model.layers linears")
    return config, files


def inspect_source(src, files, row_chunk):
    import torch
    from safetensors import safe_open
    with safe_open(str(src / "readout.safetensors"), framework="pt", device="cpu") as head:
        if list(head.keys()) != ["weight"]:
            raise ValueError("unexpected decision readout keys")
        weight = head.get_tensor("weight")
        if weight.shape != (255, 5120) or weight.dtype != torch.bfloat16:
            raise ValueError("decision readout must remain BF16 255x5120")
    amax, key_group = {}, {}
    for name, indexed_keys in sorted(files.items()):
        with safe_open(str(src / name), framework="pt", device="cpu") as shard:
            if set(shard.keys()) != indexed_keys:
                raise ValueError(f"source shard/index mismatch: {name}")
            for key in shard.keys():
                group = classify(key)
                if group is None:
                    continue
                tensor = shard.get_slice(key)
                shape = tensor.get_shape()
                if (len(shape) != 2 or min(shape) < 1 or shape[0] % 128 or shape[1] % 128
                        or tensor.get_dtype() != "BF16"):
                    raise ValueError(f"unsupported BF16 Marlin linear shape: {key} {shape}")
                maximum = 0.0
                for start in range(0, shape[0], row_chunk):
                    part = tensor[start:start + row_chunk].float()
                    if not bool(torch.isfinite(part).all()):
                        raise ValueError(f"nonfinite source weight: {key}")
                    maximum = max(maximum, float(part.abs().amax()))
                amax[group] = max(amax.get(group, 0.0), maximum)
                key_group[key] = group
    return amax, key_group


def write_backbone(src, out, files, amax, key_group, row_chunk, shard_bytes):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    weight_map, tensors = {}, {}
    stats = {"quantized": 0, "copied": 0, "tensor_bytes": 0, "file_bytes": 0}
    pending_bytes, shard_number = 0, 0

    def flush():
        nonlocal tensors, pending_bytes, shard_number
        if not tensors:
            return
        shard_number += 1
        name = f"model-{shard_number:05d}.safetensors"
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        stats["file_bytes"] += (out / name).stat().st_size
        weight_map.update({key: name for key in tensors})
        tensors, pending_bytes = {}, 0
        print(f"[pplx-nvfp4] wrote {name}", flush=True)

    def add(items):
        nonlocal pending_bytes
        size = sum(t.numel() * t.element_size() for t in items.values())
        if pending_bytes and pending_bytes + size > shard_bytes:
            flush()
        tensors.update(items)
        pending_bytes += size
        stats["tensor_bytes"] += size
        # A single untouched embedding may exceed the shard target. Flush it
        # immediately, never retain it alongside later tensors or a full shard.
        if pending_bytes >= shard_bytes:
            flush()

    for name in sorted(files):
        with safe_open(str(src / name), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                group = key_group.get(key)
                if group is None:
                    add({key: shard.get_tensor(key)})
                    stats["copied"] += 1
                    continue
                tensor = shard.get_slice(key)
                rows = tensor.get_shape()[0]
                packed_parts, scale_parts = [], []
                for start in range(0, rows, row_chunk):
                    packed, scales, global_scale = shared.quantize_weight(
                        tensor[start:start + row_chunk], amax[group])
                    packed_parts.append(packed)
                    scale_parts.append(scales)
                base = shared.module_name(key)
                add({key: torch.cat(packed_parts),
                     base + ".weight_scale": torch.cat(scale_parts),
                     base + ".weight_scale_2": global_scale})
                stats["quantized"] += 1
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": stats["tensor_bytes"]}, "weight_map": weight_map}, indent=2))
    return stats


def convert(src, out, row_chunk=128, shard_mib=256, threads=4):
    import torch
    src, out = Path(src).absolute(), Path(out).absolute()
    if row_chunk < 1 or shard_mib < 1 or threads < 1:
        raise ValueError("row chunk, shard size and CPU threads must be positive")
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("output must be absent or an empty directory; refusing overwrite")
    torch.set_num_threads(threads)
    config, files = source_layout(src)
    started = time.time()
    amax, key_group = inspect_source(src, files, row_chunk)
    source_names = sorted(set(files) | {"config.json", "model.safetensors.index.json",
                                      "decision_config.json", "readout.safetensors"})
    if (src / "release-manifest.json").is_file():
        source_names.append("release-manifest.json")
    source_hashes = {name: shared.sha256_file(src / name) for name in source_names}
    # In the console, out is itself a PVC subPath mountpoint. Keep staging
    # on that filesystem and never remove/replace the output directory root.
    out.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=out))
    published = []
    try:
        stats = write_backbone(src, temporary, files, amax, key_group,
                               row_chunk, shard_mib * 2**20)
        skip = {"config.json", "model.safetensors.index.json", "hf_quant_config.json",
                "suffix_quant_manifest.json", "release-manifest.json"}
        for file in src.iterdir():
            if (file.is_file() and not file.name.startswith(".")
                    and file.name not in skip
                    and (file.suffix in shared.COPY_FILES
                         or file.name in ("LICENSE", "NOTICE", "readout.safetensors"))):
                shutil.copy2(file, temporary / file.name)
        # Original release metadata describes the BF16 source shards. Preserve
        # it explicitly as source provenance rather than implying it describes
        # this new set of packed shards and hashes.
        if (src / "release-manifest.json").is_file():
            shutil.copy2(src / "release-manifest.json", temporary / "source-release-manifest.json")
        config["quantization_config"] = quant_config()
        (temporary / "config.json").write_text(json.dumps(config, indent=2))
        (temporary / "hf_quant_config.json").write_text(json.dumps(quant_config(), indent=2))
        if shared.sha256_file(temporary / "readout.safetensors") != source_hashes["readout.safetensors"]:
            raise ValueError("readout copy changed")
        manifest = {"kind": "pplx_decider_w4a16_nvfp4", "source_sha256": source_hashes,
                    "source_release_manifest": "source-release-manifest.json"
                        if "release-manifest.json" in source_hashes else None,
                    "quantization": quant_config(), "stats": stats,
                    "recipe": {"weights": "RTN NVFP4, fused-group amax, CPU-stream",
                               "activations": "BF16, unquantized; no calibration",
                               "row_chunk": row_chunk, "shard_mib": shard_mib,
                               "threads": threads, "allowlist": QUANT_RE.pattern},
                    "versions": {"torch": torch.__version__, "safetensors": importlib.metadata.version("safetensors")},
                    "seconds": round(time.time() - started, 2)}
        (temporary / "suffix_quant_manifest.json").write_text(json.dumps(manifest, indent=2))
        if set(out.iterdir()) != {temporary}:
            raise ValueError("output changed while converting; refusing overwrite")
        # Upload is gated on DONE after this transaction. Publish weights/head
        # and provenance first, then the loader configuration and index last.
        last = {"hf_quant_config.json": 1, "config.json": 2,
                "model.safetensors.index.json": 3}
        for file in sorted(temporary.iterdir(), key=lambda path: (last.get(path.name, 0), path.name)):
            target = out / file.name
            os.replace(file, target)
            published.append(target)
        temporary.rmdir()
    except BaseException:
        for file in reversed(published):
            file.unlink(missing_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[quantize-nvfp4] DONE W4A16_NVFP4 {stats['quantized']} linears, {stats['file_bytes'] / 2**30:.2f} GiB backbone", flush=True)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--shard-mib", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    convert(args.src, args.out, args.row_chunk, args.shard_mib, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
