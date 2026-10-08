# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next block-FP8 checkpoint -> AMD Quark OCP-MX MXFP4 (W4A4), on CPU.

    python -m suffix_hybrid.tools.quantize_qwen4exp_mxfp4 --src DIR --out DIR \
        [--risky mxfp4|bf16] [--workers 4] [--threads 1]
    S3_BUCKET=... python -m suffix_hybrid.tools.quantize_qwen4exp_mxfp4 --s3-src PREFIX \
        --s3-out PREFIX --out DIR [--tmp DIR] [--overwrite] [...]   (streaming, no staging)

Source: the official Qwen/Qwen3.8-Flash-Next-FP8 layout (Qwen4ExpForConditionalGeneration,
quant_method "fp8", weight_block_size [128, 128]): routed experts are F8_E4M3 with a
BF16 weight_scale_inv per 128x128 block, the PLE n-gram table is F8_E4M3 with one
global scale, and every other tensor is BF16 (byte-identical to the BF16 base).

Target: what vLLM's ROCm build loads natively on gfx950 (quant_method "quark", OCP MX
MXFP4 weights + dynamic MXFP4 activations; the native dense kernel needs the dynamic
MXFP4 activations, so there is no W4A16 variant):
  <linear>.weight        uint8 [N, K/2]   e2m1 codes, element 2i in the low nibble
  <linear>.weight_scale  uint8 [N, K/32]  e8m0, 2**(v - 127), one per 32 inputs
No input scales and no swizzle; vLLM shuffles for AITER at load.
The rounding is bit-exact with amd/Qwen3.8-Flash-Next-Quark-MXFP4
(tests/test_quantize_qwen4exp_mxfp4.py).

Layer policy (the PLE table needs text_config.ple_embedding_dtype, set here):
  MXFP4  routed + shared experts (main layers and the MTP layer), linear_attn.out_proj,
         self_attn.o_proj; with --risky mxfp4 (default) also linear_attn.in_proj_qkv/z
         and self_attn.q/k/v_proj (--risky bf16 keeps those four BF16 for the A/B)
  FP8    PLE n-gram table, copied unchanged with its global scale
  BF16   everything else, copied unchanged: embeddings, lm_head, hyper-connection
         mixers, router + shared-expert gates, linear_attn in_proj_a/b + conv1d +
         A_log/dt_bias/norm, the QSA indexer, PLE key/value projections, the MTP
         attention and fc_embedding/fc_hidden, the vision tower

Output: shards named like the input (input shard i -> output shard i), the
safetensors index, config.json, side files and suffix_quant_manifest.json. Each
worker converts one input shard at a time, so RAM stays at a few GiB per worker.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TOOL_VERSION = "1"
DONE_MARKER = "[quantize-nvfp4] DONE"  # the console's run gate greps for this (api/quantize.py)
MANIFEST_NAME = "suffix_quant_manifest.json"
GROUP = 32  # OCP MX block size
FP8_BLOCK = (128, 128)
PLE_EMBEDDING_DTYPE = "float8_e4m3fn"
RISKY_CHOICES = ("mxfp4", "bf16")
ROW_CHUNK = 4096  # rows per quantization call; bounds float32 temporaries
COPY_SUFFIXES = (".json", ".jinja", ".txt", ".model", ".tiktoken")
SKIP_FILES = {"config.json", "model.safetensors.index.json", "hf_quant_config.json",
              MANIFEST_NAME}

# Checkpoint (HF) names. The MTP layer is "mtp.layers.0" in the checkpoint.
_LAYER = r"(?:model\.language_model\.layers\.\d+|mtp\.layers\.\d+)"
EXPERT_RE = re.compile(rf"^{_LAYER}\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$")
SHARED_RE = re.compile(rf"^{_LAYER}\.mlp\.shared_expert\.(?:gate|up|down)_proj\.weight$")
SAFE_DENSE_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.(?:linear_attn\.out_proj|self_attn\.o_proj)\.weight$")
RISKY_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.in_proj_(?:qkv|z)|self_attn\.[qkv]_proj)\.weight$")
PLE_TABLE_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\."
    r"(?:shard_\d+\.weight|weight_scale)$")


def classify(key: str, risky: str = "mxfp4") -> str:
    """-> "expert" (block FP8 -> MXFP4), "mxfp4" (BF16 -> MXFP4) or "copy"."""
    if EXPERT_RE.match(key):
        return "expert"
    if SHARED_RE.match(key) or SAFE_DENSE_RE.match(key):
        return "mxfp4"
    if RISKY_RE.match(key):
        return "mxfp4" if risky == "mxfp4" else "copy"
    return "copy"


# ---------------------------------------------------------------------------
# Quark config. vLLM matches `exclude` entries against its own module names
# (exact string, or "re:" + re.match), and a fused module is excluded when all
# of its shards are (quantization/quark/utils.py should_ignore_layer). For the
# target model the list first goes through the model's prefix renames
# (model.language_model. -> language_model.model., model.visual. -> visual.).
# The MTP draft uses the list as written (models/qwen4_exp/amd/mtp.py only
# renumbers ignored_layers/exclude_modules, which QuarkConfig does not have), and
# its modules are mtp.fc_embedding, mtp.fc_hidden and mtp.layers.48.*
# (48 = num_hidden_layers), never the checkpoint's mtp.layers.0.*. The "re:"
# entries below match both models' names and pass through the renames unchanged.
# ---------------------------------------------------------------------------
_MX = {"dtype": "fp4", "qscheme": "per_group", "ch_axis": -1, "group_size": GROUP,
       "block_size": None, "symmetric": None, "round_method": "half_even",
       "scale_type": "float", "scale_format": "e8m0", "scale_calculation_mode": "even",
       "mx_element_dtype": None, "observer_cls": "PerBlockMXObserver",
       "is_scale_quant": False, "enable_buffer_reuse": False, "max_input_numel": 4194304}

EXCLUDE = [
    "lm_head",
    r"re:.*\.mlp\.gate$",                                # router
    r"re:.*\.shared_expert_gate$",
    r"re:.*\.linear_attn\.(in_proj_a|in_proj_b|conv1d)$",  # shards of fused in_proj_ba
    r"re:.*\.self_attn\.indexer\.",                      # QSA indexer (index_qk_proj)
    r"re:.*hyper_connection",
    r"re:.*\.ple\.",                                     # PLE kv_proj (key_proj + value_proj)
    r"re:.*visual\.",                                    # vision tower
    r"re:^mtp\.fc_(embedding|hidden)$",                  # MTP draft input projections
    r"re:^mtp\.layers\.\d+\.self_attn\.",                # MTP draft attention (mtp.layers.48)
]
EXCLUDE_RISKY_BF16 = [
    r"re:.*\.linear_attn\.in_proj_(qkv|z)$",             # shards of fused in_proj_qkvz
    r"re:.*\.self_attn\.(q|k|v)_proj$",                  # shards of fused qkv_proj
]


def quark_config(risky: str = "mxfp4") -> dict:
    if risky not in RISKY_CHOICES:
        raise ValueError(f"risky must be one of {RISKY_CHOICES}")
    exclude = EXCLUDE + (EXCLUDE_RISKY_BF16 if risky == "bf16" else [])
    return {
        "quant_method": "quark",
        "quant_mode": "eager_mode",
        "version": "0.13",
        "global_quant_config": {
            "weight": dict(_MX, is_dynamic=False),
            "input_tensors": dict(_MX, is_dynamic=True),
            "output_tensors": None, "bias": None, "target_device": None,
        },
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "kv_cache_quant_config": {},
        "kv_cache_post_rope": False,
        "softmax_quant_spec": None,
        "algo_config": None,
        "exclude": exclude,
        "export": {"kv_cache_group": [], "min_kv_scale": 0.0, "pack_method": "reorder",
                   "weight_format": "real_quantized", "weight_merge_groups": None},
    }


# ---------------------------------------------------------------------------
# MXFP4 math (torch, CPU)
# ---------------------------------------------------------------------------
_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # boundaries between the e2m1 magnitudes
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_quantize(w):
    """[N, K] float -> (uint8 [N, K/2] packed e2m1, uint8 [N, K/32] e8m0).

    Quark's scale_calculation_mode "even" (quark/torch/quantization/utils.py even_round):
    amax's fp32 mantissa is rounded at 1.75 (add 2**21 to the bits), and the shared
    exponent is floor(log2(that)) - 2 (emax of e2m1), clamped to [-127, 127].
    Elements: x / 2**exp, saturated at 6, round-half-to-even onto
    {0, .5, 1, 1.5, 2, 3, 4, 6}; a value that rounds to 0 is written as +0.
    """
    import torch
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    blk = w.to(torch.float32).reshape(n, k // GROUP, GROUP)
    amax = blk.abs().amax(-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError("non-finite weight")
    biased = ((amax.view(torch.int32) + 0x200000) >> 23) & 0xFF
    exp = (biased - 129).clamp_(-127, 127)
    # 2**exp built from its bits (exact; 2**-127 is the fp32 subnormal 0x400000)
    bits = torch.where(exp > -127, (exp + 127) << 23, torch.full_like(exp, 0x400000))
    q = blk / bits.view(torch.float32).unsqueeze(-1)
    mag = q.abs().clamp_(max=6.0)
    mids = torch.tensor(_MIDS + (float("inf"),), dtype=torch.float32)
    code = torch.bucketize(mag, mids[:-1])  # midpoints strictly below mag
    code += (code & 1).bool() & (mag == mids[code])  # a tie above an odd code rounds up
    code = code.to(torch.uint8)
    code |= ((q < 0) & (code > 0)).to(torch.uint8) << 3
    code = code.reshape(n, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), (exp + 127).to(torch.uint8)


def mxfp4_dequantize(packed, scale):
    """Inverse of mxfp4_quantize -> float32 [N, K] (checks and tests)."""
    import torch
    code = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    val = torch.tensor(E2M1)[code & 7] * (1.0 - 2.0 * (code >> 3).float())
    n, k = val.shape
    s = torch.ldexp(torch.ones(scale.shape), scale.int() - 127)
    return (val.reshape(n, k // GROUP, GROUP) * s.unsqueeze(-1)).reshape(n, k)


def quantize_rows(w, rows: int = ROW_CHUNK):
    """mxfp4_quantize in row chunks, so big dense matrices stay within a few 100 MB."""
    import torch
    if w.shape[0] <= rows:
        return mxfp4_quantize(w)
    parts = [mxfp4_quantize(w[i:i + rows]) for i in range(0, w.shape[0], rows)]
    return torch.cat([p for p, _ in parts]), torch.cat([s for _, s in parts])


def dequant_fp8_block(w8, scale_inv, block=FP8_BLOCK):
    """Fine-grained FP8 [N, K] * per-block dequant scale [ceil(N/bn), ceil(K/bk)] -> f32."""
    import torch
    n, k = w8.shape
    bn, bk = block
    gn, gk = -(-n // bn), -(-k // bk)
    if tuple(scale_inv.shape) != (gn, gk):
        raise ValueError(f"block scale {tuple(scale_inv.shape)} does not fit {n}x{k} / {block}")
    w = w8.to(torch.float32)
    s = scale_inv.to(torch.float32)
    if n % bn == 0 and k % bk == 0:
        return (w.view(gn, bn, gk, bk) * s.view(gn, 1, gk, 1)).view(n, k)
    return w * s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]


# ---------------------------------------------------------------------------
# source checks
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def source_plan(src: Path, risky: str) -> tuple[dict, dict, Counter]:
    """Validate the FP8 source -> (config, weight_map, expected action counts)."""
    config = json.loads((src / "config.json").read_text())
    if "Qwen4ExpForConditionalGeneration" not in (config.get("architectures") or []):
        raise ValueError(f"expected Qwen4ExpForConditionalGeneration, got {config.get('architectures')}")
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "fp8" or list(qc.get("weight_block_size") or []) != list(FP8_BLOCK):
        raise ValueError(f"expected block-FP8 {list(FP8_BLOCK)} source, got {qc.get('quant_method')} "
                         f"{qc.get('weight_block_size')}")
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    for key, name in weight_map.items():
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"invalid shard name {name!r}")
    for key in weight_map:
        if key.endswith(".weight_scale_inv"):
            if not EXPERT_RE.match(key[:-len("_scale_inv")]):
                raise ValueError(f"block-FP8 tensor outside the routed experts: {key}")
        elif EXPERT_RE.match(key) and key + "_scale_inv" not in weight_map:
            raise ValueError(f"routed expert without weight_scale_inv: {key}")
    if not any(PLE_TABLE_RE.match(k) and k.endswith(".weight_scale") for k in weight_map):
        raise ValueError("no FP8 PLE table scale (ngram_embedding.weight_scale) in the source")
    text = config.get("text_config") or {}
    layers = int(text["num_hidden_layers"]) + int(text.get("mtp_num_hidden_layers") or 0)
    expected = Counter(classify(k, risky) for k in weight_map if not k.endswith("_scale_inv"))
    want_experts = layers * int(text["num_experts"]) * 3
    if expected["expert"] != want_experts:
        raise ValueError(f"{expected['expert']} routed-expert weights, expected {want_experts}")
    return config, weight_map, expected


# ---------------------------------------------------------------------------
# conversion (one input shard per call; runs in worker processes)
# ---------------------------------------------------------------------------
def convert_shard(src: str, out: str, name: str, keys: list, scale_files: dict, risky: str,
                  threads: int) -> dict:
    """Convert input shard `name` (holding exactly `keys`) into out/name.

    scale_files maps each expert weight_scale_inv this shard needs to the shard that
    holds it (normally this one)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(threads)
    src_dir, started = Path(src), time.time()
    tensors, kinds = {}, Counter()
    with contextlib.ExitStack() as stack:
        shard = stack.enter_context(safe_open(str(src_dir / name), framework="pt", device="cpu"))
        if set(shard.keys()) != set(keys):
            raise ValueError(f"{name}: shard keys differ from the safetensors index")
        handles = {name: shard}

        def scale_inv(key):
            other = scale_files[key]
            if other not in handles:
                handles[other] = stack.enter_context(
                    safe_open(str(src_dir / other), framework="pt", device="cpu"))
            return handles[other].get_tensor(key)

        for key in shard.keys():
            if key.endswith(".weight_scale_inv"):
                continue  # consumed with its weight
            kind = classify(key, risky)
            t = shard.get_tensor(key)
            if kind == "expert":
                if t.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"{key}: expected F8_E4M3, got {t.dtype}")
                packed, scale = quantize_rows(dequant_fp8_block(t, scale_inv(key + "_scale_inv")))
            elif kind == "mxfp4":
                if t.dtype != torch.bfloat16 or t.dim() != 2:
                    raise ValueError(f"{key}: expected a 2-D BF16 weight, got {t.dtype} {tuple(t.shape)}")
                packed, scale = quantize_rows(t)
            else:
                ple = PLE_TABLE_RE.match(key)
                if t.dtype == torch.float8_e4m3fn and not ple:
                    raise ValueError(f"FP8 tensor without a conversion rule: {key}")
                if ple and key.endswith(".weight") and t.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"{key}: the PLE table must be F8_E4M3, got {t.dtype}")
                tensors[key] = t
                kinds["copy"] += 1
                continue
            tensors[key] = packed
            tensors[key + "_scale"] = scale
            kinds[kind] += 1
    save_file(tensors, str(Path(out) / name), metadata={"format": "pt"})
    return {"name": name, "keys": sorted(tensors), "kinds": dict(kinds),
            "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "file_bytes": (Path(out) / name).stat().st_size,
            "sha256": sha256_file(src_dir / name), "seconds": round(time.time() - started, 1)}


def _shard_args(weight_map: dict) -> list[tuple]:
    """(shard name, its keys, {expert scale_inv key: shard holding it}) per input shard."""
    by_shard = {}
    for key, name in weight_map.items():
        by_shard.setdefault(name, []).append(key)
    return [(n, by_shard[n], {k + "_scale_inv": weight_map[k + "_scale_inv"]
                              for k in by_shard[n] if EXPERT_RE.match(k)})
            for n in sorted(by_shard)]


def _pool(fn_name: str, args: list[tuple], workers: int) -> list[dict]:
    # Imported by name so spawn workers resolve it the same way under `python -m`.
    from suffix_hybrid.tools import quantize_qwen4exp_mxfp4 as tool
    fn, results = getattr(tool, fn_name), []
    if workers == 1:
        it = (fn(*a) for a in args)
    else:
        ctx = multiprocessing.get_context("spawn")  # fresh interpreters, no forked torch pools
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        it = pool.map(fn, *zip(*args))
    try:
        for r in it:
            results.append(r)
            print(f"[mxfp4] wrote {r['name']} ({len(results)}/{len(args)}, {r['seconds']} s)",
                  flush=True)
    finally:
        if workers != 1:
            pool.shutdown(cancel_futures=True)
    return results


def _write_meta(dst: Path, src_meta: Path, config: dict, results: list, expected: Counter,
                risky: str, workers: int, threads: int, started: float):
    """Index, side files, config.json and the manifest into dst -> (stats, kinds)."""
    import torch
    kinds, out_map = Counter(), {}
    for r in results:
        kinds.update(r["kinds"])
        for key in r["keys"]:
            if key in out_map:
                raise ValueError(f"duplicate output tensor {key}")
            out_map[key] = r["name"]
    if any(kinds[k] != expected[k] for k in ("expert", "mxfp4", "copy")):
        raise ValueError(f"converted {dict(kinds)}, planned {dict(expected)}")
    tensor_bytes = sum(r["tensor_bytes"] for r in results)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": tensor_bytes}, "weight_map": dict(sorted(out_map.items()))},
        indent=2))
    for f in src_meta.iterdir():
        if (f.is_file() and not f.name.startswith(".") and f.name not in SKIP_FILES
                and (f.suffix in COPY_SUFFIXES or f.name in ("LICENSE", "NOTICE"))):
            shutil.copy2(f, dst / f.name)
    source_sha = {r["name"]: r["sha256"] for r in results}
    for f in ("config.json", "model.safetensors.index.json"):
        source_sha[f] = sha256_file(src_meta / f)
    config["quantization_config"] = quark_config(risky)
    config.setdefault("text_config", {})["ple_embedding_dtype"] = PLE_EMBEDDING_DTYPE
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    stats = {"tensors": dict(kinds), "tensor_bytes": tensor_bytes,
             "file_bytes": sum(r["file_bytes"] for r in results),
             "seconds": round(time.time() - started, 1)}
    manifest = {
        "kind": "qwen4exp_quark_mxfp4",
        "source_sha256": source_sha,
        "quantization": config["quantization_config"],
        "text_config_overrides": {"ple_embedding_dtype": PLE_EMBEDDING_DTYPE},
        "recipe": {"weights": "RTN OCP-MX MXFP4 (Quark scale_calculation_mode even), CPU",
                   "experts": "block-FP8 source dequantized (w * weight_scale_inv) then MXFP4",
                   "activations": "dynamic MXFP4 at runtime; no calibration",
                   "risky": risky, "ple": "FP8 table copied unchanged",
                   "workers": workers, "threads": threads},
        "versions": {"tool": TOOL_VERSION, "python": sys.version.split()[0],
                     "torch": torch.__version__,
                     "safetensors": importlib.metadata.version("safetensors"),
                     "plugin_revision": os.environ.get("SUFFIX_PLUGIN_REVISION")},
        "stats": stats,
    }
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return stats, kinds


def _done(kinds, stats, risky) -> None:
    print(f"{DONE_MARKER} quark-mxfp4 risky={risky}: {kinds['expert']} experts + "
          f"{kinds['mxfp4']} dense -> MXFP4, {kinds['copy']} copied, "
          f"{stats['file_bytes'] / 2**30:.2f} GiB in {stats['seconds']} s", flush=True)


def convert(src, out, risky="mxfp4", workers=4, threads=1) -> dict:
    src, out = Path(src).absolute(), Path(out).absolute()
    if risky not in RISKY_CHOICES or workers < 1 or threads < 1:
        raise ValueError("bad --risky/--workers/--threads")
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise SystemExit(f"--out {out} is not empty (refusing to overwrite)")
    started = time.time()
    config, weight_map, expected = source_plan(src, risky)
    print(f"[mxfp4] {len(weight_map)} source tensors in {len(set(weight_map.values()))} shards; "
          f"plan {dict(expected)} (risky={risky})", flush=True)
    # The console mounts --out as a PVC subPath: stage beside it on the same
    # filesystem and never replace the directory itself.
    out.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=out))
    published = []
    try:
        results = _pool("convert_shard", [(str(src), str(pending), n, keys, scales, risky, threads)
                                          for n, keys, scales in _shard_args(weight_map)], workers)
        stats, kinds = _write_meta(pending, src, config, results, expected, risky, workers,
                                   threads, started)
        if set(out.iterdir()) != {pending}:
            raise ValueError("output changed while converting; refusing to publish")
        last = {"config.json": 2, "model.safetensors.index.json": 1}
        for f in sorted(pending.iterdir(), key=lambda p: (last.get(p.name, 0), p.name)):
            os.replace(f, out / f.name)
            published.append(out / f.name)
        pending.rmdir()
    except BaseException:
        for f in reversed(published):
            f.unlink(missing_ok=True)
        shutil.rmtree(pending, ignore_errors=True)
        raise
    _done(kinds, stats, risky)
    return stats


# ---------------------------------------------------------------------------
# streaming mode: bucket -> bucket, one shard per worker on local scratch
# (no node in the cluster has room to stage 173 GiB in + 115 GiB out)
# ---------------------------------------------------------------------------
def _s3():
    """boto3 client for the in-cluster bucket (endpoint + keys: the weights Secret env)."""
    try:
        import boto3
    except ImportError:  # the console's job scripts install it the same way
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--disable-pip-version-check", "boto3"])
        import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                        config=Config(s3={"addressing_style": "path"},
                                      retries={"max_attempts": 5}))


def _list(s3, bucket: str, prefix: str) -> dict:
    return {o["Key"][len(prefix):]: o["Size"]
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", [])}


def stream_shard(bucket, src_prefix, out_prefix, tmp, name, keys, scale_files, risky, threads):
    """Worker: fetch input shard `name` (+ any shard holding its scales), convert it,
    upload the output shard, delete the local copies."""
    s3 = _s3()
    work = Path(tmp) / f"w-{name}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "in").mkdir(parents=True)
    (work / "out").mkdir()
    try:
        for shard in sorted({name, *scale_files.values()}):
            s3.download_file(bucket, src_prefix + shard, str(work / "in" / shard))
        r = convert_shard(str(work / "in"), str(work / "out"), name, keys, scale_files, risky,
                          threads)
        s3.upload_file(str(work / "out" / name), bucket, out_prefix + name)
        return r
    finally:
        shutil.rmtree(work, ignore_errors=True)


def convert_s3(bucket, src_prefix, out_prefix, out, tmp, risky="mxfp4", workers=4, threads=1,
               overwrite=False) -> dict:
    """s3://bucket/src_prefix (block FP8) -> s3://bucket/out_prefix (MXFP4) without a
    staged copy: scratch holds ~2 shards per worker. config.json is uploaded LAST
    (weight staging treats it as the commit marker); `out` keeps the metadata files
    and the manifest, where the console job's run gate looks for them."""
    if risky not in RISKY_CHOICES or workers < 1 or threads < 1:
        raise ValueError("bad --risky/--workers/--threads")
    started, s3 = time.time(), _s3()
    sp, op = src_prefix.strip("/") + "/", out_prefix.strip("/") + "/"
    if sp.startswith(op) or op.startswith(sp):
        raise ValueError(f"source {sp} and output {op} prefixes overlap")
    out, tmp = Path(out).absolute(), Path(tmp).absolute()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty (refusing to overwrite)")
    existing = sorted(_list(s3, bucket, op))
    if existing and not overwrite:
        raise SystemExit(f"s3://{bucket}/{op} already holds {len(existing)} objects "
                         "(refusing to overwrite)")
    for i in range(0, len(existing), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": op + k}
                                                             for k in existing[i:i + 1000]]})
    meta = tmp / "src-meta"
    shutil.rmtree(meta, ignore_errors=True)
    meta.mkdir(parents=True)
    src_objs = _list(s3, bucket, sp)
    for n in src_objs:
        if "/" not in n and not n.endswith(".safetensors"):
            s3.download_file(bucket, sp + n, str(meta / n))
    config, weight_map, expected = source_plan(meta, risky)
    missing = sorted(set(weight_map.values()) - set(src_objs))
    if missing:
        raise ValueError(f"index names {len(missing)} shards missing from s3://{bucket}/{sp}: "
                         f"{missing[:3]}")
    print(f"[mxfp4] streaming s3://{bucket}/{sp} -> {op}: {len(weight_map)} tensors in "
          f"{len(set(weight_map.values()))} shards; plan {dict(expected)} (risky={risky})",
          flush=True)
    results = _pool("stream_shard", [(bucket, sp, op, str(tmp), n, keys, scales, risky, threads)
                                     for n, keys, scales in _shard_args(weight_map)], workers)
    out.mkdir(parents=True, exist_ok=True)
    stats, kinds = _write_meta(out, meta, config, results, expected, risky, workers, threads,
                               started)
    for f in sorted((p for p in out.iterdir() if p.is_file()),
                    key=lambda p: (p.name == "config.json", p.name)):  # config.json last
        s3.upload_file(str(f), bucket, op + f.name)
    shutil.rmtree(meta, ignore_errors=True)
    _done(kinds, stats, risky)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, help="staged FP8 checkpoint directory")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory (streaming mode: metadata + manifest only)")
    ap.add_argument("--s3-src", help="streaming mode: source prefix in $S3_BUCKET")
    ap.add_argument("--s3-out", help="streaming mode: output prefix in $S3_BUCKET")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/mxfp4"),
                    help="streaming mode: shard scratch directory")
    ap.add_argument("--overwrite", action="store_true",
                    help="streaming mode: replace a non-empty output prefix")
    ap.add_argument("--risky", choices=RISKY_CHOICES, default="mxfp4",
                    help="attention q/k/v + linear_attn in_proj_qkv/z: MXFP4 (default) or BF16")
    ap.add_argument("--workers", type=int, default=4, help="input shards converted in parallel")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    args = ap.parse_args(argv)
    if bool(args.s3_src) != bool(args.s3_out) or bool(args.s3_src) == bool(args.src):
        ap.error("give either --src, or --s3-src with --s3-out")
    if args.s3_src:
        convert_s3(os.environ["S3_BUCKET"], args.s3_src, args.s3_out, args.out, args.tmp,
                   args.risky, args.workers, args.threads, args.overwrite)
    else:
        convert(args.src, args.out, args.risky, args.workers, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
