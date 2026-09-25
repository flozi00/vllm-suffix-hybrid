# SPDX-License-Identifier: Apache-2.0
"""Quantize qwen3.8-27b-fable-distill (Qwen3_5ForConditionalGeneration) to an
NVFP4 W4A4 checkpoint in ModelOpt ``modelopt_fp4`` format that vLLM 0.30.0
loads natively and routes through ``init_nvfp4_linear_kernel`` (so
SUFFIX_NVFP4_GEMM's SuffixNvFp4LinearKernel applies).

Runs inside vllm/vllm-openai:v0.30.0 with NO extra packages (torch,
transformers, safetensors, numpy from the image); nothing is downloaded and
nothing leaves the pod (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced).

    python -m suffix_hybrid.tools.quantize_nvfp4 --src DIR --out DIR \
        [--samples 256] [--gen-tokens 192] [--seed 0] [--calib selfgen|none]

Pipeline (fits one 96 GB RTX PRO 6000):
  1. calibrate: load the bf16 model onto the GPU (54 GB), self-generate
     ``--samples`` chat continuations from the bundled seed prompts
     (tools/calib_prompts.py), run them through the model with forward
     pre-hooks recording each quantized linear's input |x| max;
  2. free the model, then STREAM the source safetensors tensor by tensor:
     quantize the allowlisted language-model linears (RTN NVFP4, vLLM math:
     block-16 e4m3 scales, fused-group global scale), copy everything else
     byte-identical;
  3. write shards + model.safetensors.index.json + config.json
     (quantization_config) + hf_quant_config.json + tokenizer/processor
     files + suffix_quant_manifest.json (source sha256s, recipe, versions).

Format (vllm/model_executor/layers/quantization/modelopt.py):
  <linear>.weight          uint8 [N, K/2]   e2m1, element 2i in the low nibble
  <linear>.weight_scale    float8_e4m3fn [N, K/16]   (unswizzled; vLLM swizzles)
  <linear>.weight_scale_2  float32 []   = amax_group / 2688  (dequant scale)
  <linear>.input_scale     float32 []   = amax_act_group / 2688
Fused groups share one weight_scale_2 and input_scale (vLLM takes the max
over shards and warns if they differ: modelopt.py KNvfp4Static/KNvfp4Dynamic
process()).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

FP4_MAX = 6.0
E4M3_MAX = 448.0
GLOBAL = FP4_MAX * E4M3_MAX  # 2688
GROUP = 16
TOOL_VERSION = "1"

# Allowlist: the ONLY tensors quantized. Everything else (embeddings, lm_head,
# vision tower, MTP head, conv1d, in_proj_b / in_proj_a gates, A_log,
# dt_bias, all norms) is copied bf16/fp32 byte-identical.
QUANT_RE = re.compile(
    r"^(?:model\.language_model\.|language_model\.model\.|model\.)layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$"
)
# Fused in vLLM (qwen3_5.py packed_modules_mapping :301-311 / hf mapper :220-227)
FUSED = {
    "self_attn.q_proj": "self_attn.qkv", "self_attn.k_proj": "self_attn.qkv",
    "self_attn.v_proj": "self_attn.qkv",
    "mlp.gate_proj": "mlp.gate_up", "mlp.up_proj": "mlp.gate_up",
    "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz",
    "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
}
# vLLM exclude_modules (substring + fnmatch semantics, modelopt.py:181-215).
# 'visual' is required: vLLM's hard-coded vision skip only matches
# vision_tower/vision_model, not Qwen's `visual.` (modelopt.py:233-241).
EXCLUDE_MODULES = ["lm_head", "embed_tokens", "visual", "mtp", "in_proj_b", "in_proj_a"]
COPY_FILES = (".json", ".jinja", ".txt", ".model", ".tiktoken", ".py")


def classify(key: str):
    """-> (layer_idx, module, fused_group_key) for an allowlisted weight, else
    None. fused_group_key identifies tensors that must share global scales."""
    m = QUANT_RE.match(key)
    if not m:
        return None
    layer, mod = int(m.group(1)), m.group(2)
    prefix = key[: -len(".weight")]
    group = prefix[: -len(mod)] + FUSED.get(mod, mod)
    return layer, mod, group


def module_name(key: str) -> str:
    return key[: -len(".weight")]


# Checkpoint <-> transformers naming. Qwen3.5 checkpoints come in several
# layouts: `model.language_model.layers.N` (HF release), `model.layers.N`
# (text-only / transformers>=5 save: its Qwen3_5ForConditionalGeneration
# conversion is PrefixChange ^model.language_model.(.+) <-> model.\1),
# legacy `language_model.model.layers.N`; vision `model.visual.` or `visual.`.
# Both sides are reduced to one canonical name, so calibration never depends
# on which layout the source or the transformers class uses.
_TEXT_PREFIX = re.compile(
    r"^(?:model\.language_model\.|language_model\.model\.|language_model\.(?!lm_head)"
    r"|model\.(?!visual\.))")
_VISUAL_PREFIX = re.compile(r"^(?:model\.)?visual\.")
_HEAD_PREFIX = re.compile(r"^(?:language_model\.)?lm_head\.")


def canonical(name: str) -> str:
    """'model.language_model.layers.3.mlp.up_proj' == 'model.layers.3.mlp.up_proj'
    -> 'text.layers.3.mlp.up_proj' (also visual. / lm_head. / mtp. as-is)."""
    if _HEAD_PREFIX.match(name):
        return _HEAD_PREFIX.sub("lm_head.", name)
    if _VISUAL_PREFIX.match(name):
        return _VISUAL_PREFIX.sub("visual.", name)
    if _TEXT_PREFIX.match(name):
        return _TEXT_PREFIX.sub("text.", name)
    return name


def match_checkpoint(ckpt_shapes: dict, model_shapes: dict) -> tuple[dict, list]:
    """-> ({ckpt_key: model_param}, missing model params). Fails loud on a
    canonical-name collision or a shape mismatch (never skips silently).
    Missing vision params are tolerated (calibration is text-only)."""
    by_canon = {}
    for name in model_shapes:
        c = canonical(name)
        if c in by_canon:
            raise RuntimeError(f"canonical name collision: {by_canon[c]} / {name}")
        by_canon[c] = name
    mapping = {}
    for key, shape in ckpt_shapes.items():
        name = by_canon.get(canonical(key))
        if name is None:
            continue  # e.g. mtp.* (not in the transformers model)
        if tuple(model_shapes[name]) != tuple(shape):
            raise RuntimeError(f"shape mismatch {key} {tuple(shape)} vs model {name} "
                               f"{tuple(model_shapes[name])}")
        mapping[key] = name
    loaded = set(mapping.values())
    missing = [n for n in model_shapes
               if n not in loaded and not canonical(n).startswith("visual.")]
    return mapping, missing


# ---------------------------------------------------------------------------
# NVFP4 math (torch; identical to vLLM scaled_fp4_quant / nvfp4_gemm.quantize)
# ---------------------------------------------------------------------------
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e2m1_codes(x):
    """float tensor -> e2m1 codes (sign in bit 3), RNE ties to even code."""
    import torch
    a = x.abs().clamp(max=FP4_MAX)
    mids = torch.tensor(_MID, dtype=a.dtype, device=a.device)
    gt = (a.unsqueeze(-1) > mids).sum(-1)
    odd = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.bool, device=a.device)
    tie_up = ((a.unsqueeze(-1) == mids) & odd).sum(-1)
    code = (gt + tie_up).to(torch.uint8)
    return code | ((x < 0).to(torch.uint8) << 3)


def quantize_weight(w, amax: float):
    """w [N, K] -> (packed uint8 [N, K/2], scales float8_e4m3fn [N, K/16],
    weight_scale_2 float32 []). `amax` is the fused group's global |w| max."""
    import torch
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} not a multiple of {GROUP}")
    wf = w.float()
    g = GLOBAL / max(float(amax), 1e-12)  # quant multiplier
    blk = wf.reshape(n, k // GROUP, GROUP)
    sf = (blk.abs().amax(-1) * (g / FP4_MAX)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    inv = torch.where(sf_f == 0, torch.zeros_like(sf_f), g / sf_f.clamp_min(1e-30))
    code = e2m1_codes(blk * inv.unsqueeze(-1)).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf.contiguous(), torch.tensor(1.0 / g, dtype=torch.float32)


def dequantize_weight(packed, sf, ws2):
    """Inverse (for tests / sanity): -> float32 [N, K]."""
    import torch
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    lo, hi = packed & 0xF, packed >> 4
    code = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1).long()
    val = lut[code & 7] * torch.where(code & 8 > 0, -1.0, 1.0)
    n, k = val.shape
    return (val.reshape(n, k // GROUP, GROUP) * sf.float().unsqueeze(-1)).reshape(n, k) * ws2


# ---------------------------------------------------------------------------
# config / manifest
# ---------------------------------------------------------------------------
def quant_config() -> dict:
    return {
        "quant_method": "modelopt",
        "producer": {"name": "suffix_hybrid.tools.quantize_nvfp4", "version": TOOL_VERSION},
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": GROUP,
            "kv_cache_quant_algo": None,
            "exclude_modules": EXCLUDE_MODULES,
        },
    }


def recipe(args) -> dict:
    return {
        "format": "modelopt_fp4 (NVFP4 W4A4, group 16, e4m3 block scales, fp32 global)",
        "weights": "RTN, per-fused-group global amax",
        "activations": "static per-tensor input_scale = calibrated amax / 2688",
        "quantized": QUANT_RE.pattern,
        "fused_groups": sorted(set(FUSED.values())),
        "exclude_modules": EXCLUDE_MODULES,
        "calibration": {"mode": args.calib, "samples": args.samples,
                        "gen_tokens": args.gen_tokens, "seed": args.seed,
                        "source": "self-generated from suffix_hybrid/tools/calib_prompts.py"},
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {"tool": TOOL_VERSION, "python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "safetensors", "numpy", "vllm", "suffix_hybrid"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    rev = os.environ.get("SUFFIX_PLUGIN_REVISION")
    if rev:
        out["plugin_revision"] = rev
    return out


def build_manifest(src: Path, source_hashes: dict, args, stats: dict) -> dict:
    return {
        "kind": "suffix_nvfp4_checkpoint",
        "source_dir": str(src),
        "source_sha256": source_hashes,
        "recipe": recipe(args),
        "versions": versions(),
        "stats": stats,
        "created_unix": int(time.time()),
    }


# ---------------------------------------------------------------------------
# calibration (GPU; transformers; offline)
# ---------------------------------------------------------------------------
def calibrate(src: Path, args) -> dict[str, float]:
    """-> {canonical(module name): input |x| max} for every allowlisted linear."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    from suffix_hybrid.tools.calib_prompts import PROMPTS

    torch.manual_seed(args.seed)
    cfg = AutoConfig.from_pretrained(src)
    # transformers>=5 prints "incorrect regex pattern ... fix_mistral_regex" for
    # any local tokenizer whose config.json lacks transformers_version
    # (tokenization_utils_tokenizers.py:1332-1374), not only Mistral. Setting
    # it would swap in Mistral's pre-tokenizer regex -> wrong for Qwen. The
    # calibration text is self-generated, so tokenization only has to be
    # self-consistent: keep the checkpoint's own tokenizer untouched.
    tok = AutoTokenizer.from_pretrained(src)
    # Build directly on the GPU (host RAM on the pods is ~52 GB < 54 GB model),
    # then copy the checkpoint in shard by shard.
    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    params = dict(model.named_parameters())  # tied lm_head appears once
    shards = sorted(src.glob("*.safetensors"))
    ckpt_shapes, where = {}, {}
    for f in shards:
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                ckpt_shapes[key] = st.get_slice(key).get_shape()
                where[key] = f
    mapping, missing = match_checkpoint(
        ckpt_shapes, {n: tuple(p.shape) for n, p in params.items()})
    if missing:
        raise RuntimeError(f"calibration model ({type(model).__name__}) has {len(missing)} "
                           f"params not in the checkpoint, e.g. {missing[:3]}")
    for f in shards:
        with safe_open(str(f), framework="pt", device="cuda") as st:
            for key in st.keys():
                if key in mapping:
                    with torch.no_grad():
                        params[mapping[key]].copy_(st.get_tensor(key))
    print(f"[quantize-nvfp4] calibration model {type(model).__name__}: "
          f"{len(mapping)} checkpoint tensors loaded", flush=True)

    amax: dict[str, float] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and classify(name + ".weight"):
            def hook(m, inp, _n=canonical(name)):
                v = float(inp[0].detach().abs().amax())
                if v > amax.get(_n, 0.0):
                    amax[_n] = v
            hooks.append(mod.register_forward_pre_hook(hook))
    n_prompts = len(PROMPTS)
    per = max(1, -(-args.samples // n_prompts))
    texts = []
    for i, p in enumerate(PROMPTS):
        texts += [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)] * per
    texts = texts[: args.samples]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bs = args.batch
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
            # generation forwards feed the hooks too (prefill + every decode step)
            model.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=True,
                           temperature=0.8, top_p=0.95)
            print(f"[quantize-nvfp4] calib {min(i + bs, len(texts))}/{len(texts)} "
                  f"({len(amax)} linears seen)", flush=True)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return amax


# ---------------------------------------------------------------------------
# streaming quantization + write
# ---------------------------------------------------------------------------
def group_amax(src: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Pass 1 over the source: per fused group |w| max. Returns
    (group -> amax, key -> group)."""
    from safetensors import safe_open
    amax, key_group = {}, {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                c = classify(key)
                if c is None:
                    continue
                t = st.get_tensor(key)
                key_group[key] = c[2]
                amax[c[2]] = max(amax.get(c[2], 0.0), float(t.float().abs().amax()))
    return amax, key_group


def act_group_amax(act: dict[str, float], key_group: dict[str, str]) -> dict[str, float]:
    """Calibrated input amax per fused group (shards share their input)."""
    out: dict[str, float] = {}
    for key, g in key_group.items():
        v = act.get(canonical(module_name(key)))
        if v is None:
            raise RuntimeError(f"no calibration activations for {module_name(key)}")
        out[g] = max(out.get(g, 0.0), v)
    return out


def write_checkpoint(src: Path, out: Path, w_amax, key_group, a_amax, device: str) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    weight_map, stats = {}, {"quantized": 0, "copied": 0, "bytes_out": 0}
    for f in sorted(src.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(f), framework="pt", device="cpu") as st:
            meta = st.metadata() or {}
            for key in st.keys():
                t = st.get_tensor(key)
                g = key_group.get(key)
                if g is None:
                    tensors[key] = t
                    stats["copied"] += 1
                    continue
                packed, sf, ws2 = quantize_weight(t.to(device), w_amax[g])
                base = module_name(key)
                tensors[base + ".weight"] = packed.cpu()
                tensors[base + ".weight_scale"] = sf.cpu()
                tensors[base + ".weight_scale_2"] = ws2
                tensors[base + ".input_scale"] = torch.tensor(
                    a_amax[g] / GLOBAL, dtype=torch.float32)
                stats["quantized"] += 1
        name = f.name
        save_file(tensors, str(out / name), metadata={**meta, "format": "pt"})
        stats["bytes_out"] += (out / name).stat().st_size
        for k in tensors:
            weight_map[k] = name
        print(f"[quantize-nvfp4] wrote {name}", flush=True)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": stats["bytes_out"]}, "weight_map": weight_map},
        indent=1))
    return stats


def copy_side_files(src: Path, out: Path):
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.suffix in COPY_FILES and f.name not in (
                "model.safetensors.index.json", "config.json"):
            shutil.copy2(f, out / f.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = quant_config()
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "hf_quant_config.json").write_text(json.dumps(quant_config(), indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calib", choices=["selfgen", "none"], default="selfgen",
                    help="none = input_scale from weights only (TEST ONLY; not servable)")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    src, out = args.src, args.out
    if not (src / "config.json").is_file() or not list(src.glob("*.safetensors")):
        raise SystemExit(f"--src {src}: no config.json / *.safetensors")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty (refusing to overwrite)")
    t0 = time.time()
    hashes = {f.name: sha256_file(f) for f in sorted(src.iterdir())
              if f.is_file() and (f.suffix == ".safetensors" or f.name == "config.json")}
    w_amax, key_group = group_amax(src)
    print(f"[quantize-nvfp4] {len(key_group)} linears in {len(w_amax)} fused groups",
          flush=True)
    if args.calib == "selfgen":
        act = calibrate(src, args)
    else:
        act = {canonical(module_name(k)): 1.0 for k in key_group}
    a_amax = act_group_amax(act, key_group)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = write_checkpoint(src, out, w_amax, key_group, a_amax, device)
    copy_side_files(src, out)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["act_amax_min_max"] = [min(a_amax.values()), max(a_amax.values())]
    man = build_manifest(src, hashes, args, stats)
    (out / "suffix_quant_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[quantize-nvfp4] DONE {stats['quantized']} quantized, {stats['copied']} "
          f"copied, {stats['bytes_out'] / 2**30:.2f} GiB -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
