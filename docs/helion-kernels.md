# Helion ports of the cuda-oxide kernels

Owner, 2026-10-17: our new focus is Helion autotuning — port every cuda-oxide
kernel (`kernels-oxide/*`) to Helion and let idle GPUs autotune it hard.
The oxide kernel stays the **baseline to beat and the fallback**: per shape
the faster one serves (owner rule, custom kernels always; even 1% counts).
An oxide kernel is deleted only once Helion wins every shape it serves.

```
suffix_hybrid/helion_kernels/__init__.py   runtime: config tables, bucket, backend knob, compiled()
suffix_hybrid/helion_kernels/<family>.py   one port: Helion bodies + serving entry + lab KERNELS
suffix_hybrid/configs/helion/<kernel>/<platform>.json   tuned WINS from the autotune lab
```

## Families and knobs

| family | oxide crate | serving lane | knob |
|---|---|---|---|
| nvfp4_gemm | kernels-oxide/nvfp4_gemm | kernels/nvfp4_linear.py, kernels/nvfp4_lm_head.py | `SUFFIX_HELION_NVFP4_GEMM` |
| nvfp4_moe | kernels-oxide/nvfp4_moe | kernels/nvfp4_moe.py | `SUFFIX_HELION_NVFP4_MOE` |
| fp8_moe | kernels-oxide/fp8_moe | kernels/fp8_moe.py | `SUFFIX_HELION_FP8_MOE` |
| k2_attn | kernels-oxide/k2_nvfp4_attn | sm120/nvfp4_kv_patch/own_attn.py | `SUFFIX_HELION_K2_ATTN` |
| dsa_indexer | kernels-oxide/dsa_indexer | kernels/dsa_indexer.py | `SUFFIX_HELION_DSA_INDEXER` |
| ds_mla | kernels-oxide/nvfp4_ds_mla | sm120/nvfp4_ds_mla_patch/ | `SUFFIX_HELION_DS_MLA` |
| kgdn1 | kernels-oxide/kgdn1 | kernels/qwen_gdn.py | `SUFFIX_HELION_KGDN1` |
| dsv41_attn_{partial,row} | none (FlashInfer SM120 sparse MLA is the baseline) | kernels/dsv41_attn.py | `SUFFIX_HELION_DSV41_ATTN` |

Every knob is `helion_kernels.backend(env)`:

- `oxide` (default) — unchanged behaviour, the Helion module is never imported.
- `helion` — every shape on Helion; a shape with no shipped config is a
  startup error (fail closed, like every gate here).
- `auto` — Helion where a config is shipped, oxide elsewhere. The config
  pull ships only cases where Helion beat the oxide baseline in the lab, so
  `auto` IS "the faster kernel per shape".

The knob only picks the implementation inside an already-enabled lane
(`SUFFIX_NVFP4_MOE=1` etc.): the lane's gates, shape checks and load-time
oracles stay in force for both backends.

## Serving never autotunes

`helion_kernels.compiled(body, kernel, case)` binds a body to the config
shipped for exactly `case` (`bucket(m, buckets)` maps a serving M to the
tuned bucket first) with `static_shapes=False`: one config serves its whole
M bucket, compiled once per process (Triton cache), before CUDA-graph
capture — every port warms all its buckets in the lane's load-time oracle.

## The lab (inference-console autotune)

Each port module is also a lab kernel module: a module-level `KERNELS` list
in the console ABI (inference-console `api/autotune_kernels.py` docstring):
`name`, undecorated module-level `body`, `cases(model)` (flat JSON dicts with
`numtokens`, `[]` when the model does not use the kernel), `make(case, seed)`,
`reference(*args)`, `baseline(*args)` = **the oxide kernel** (via
`suffix_hybrid._native`; the lab mounts this plugin), `baseline_name`,
`atol/rtol` or `check`, `replaces` (regexes over the oxide kernel names in a
profile). The lab submits a two-line shim per family:

```python
import sys; sys.path.append("/plugins")
from suffix_hybrid.helion_kernels.nvfp4_moe import KERNELS
```

Autotune effort = the lab setting (`full`). Winners land in the run's
`configs/<kernel>/<platform>.json`; the pull script copies the WINNING rows
(correct, speedup > 1 vs oxide) into `suffix_hybrid/configs/helion/`.

## CPU checks

`tests/test_helion_<family>.py` run each body in Helion's eager reference
mode (`HELION_INTERPRET=1`) on small CPU tensors against the reference, and
skip when Helion is not installed. They prove indexing and math, not speed.

## Bundle

CI (`.github/workflows/native.yml`, env `HELION_VENDOR`) runs
`pip install --no-deps --only-binary=:all: --target vendor` for helion 1.5.0
plus the pure-python deps the vLLM image lacks (rich, markdown-it-py, mdurl,
pygments), fails on any compiled file, and passes the tree to
`scripts/runtime_bundle.py --vendor-dir`: it lands in `/plugins/_vendor/`,
hashed in BUILD.json. `ensure_vendor()` appends it to `sys.path`, so an image
that ships Helion wins. Not vendored: numpy, psutil, typing-extensions,
torch, triton (the image has them), filecheck (Helion never imports it) and
scikit-learn/scipy: only the autotuner's surrogate searches import sklearn,
inside `try/except ImportError`, and serving never autotunes. Tuned configs
ride in the wheel like `configs/afp4` (maturin ships every file under
`suffix_hybrid/`). The CI test step installs helion + triton 3.8.0 (the pin
of PyPI torch 2.14.x, matching the CPU torch it installs) so the interpret
checks run instead of skipping.

## mxfp8_gemm (no oxide twin: DeepSeek-V4.1 FP8 linears on SM120)

`helion_kernels/mxfp8_gemm.py` + lane `kernels/mxfp8_linear.py`, knob
`SUFFIX_HELION_MXFP8_GEMM` = oxide (default: lane off, vLLM unchanged) |
auto | helion, `SUFFIX_MXFP8_GEMM_MAX_M` (default 192). The baseline is
vLLM's own SM120 pick, not an oxide kernel: FlashInfer CUTLASS `mm_mxfp8` +
dynamic MXFP8 quant for the plain linears (pre-quantized input at K = 1280,
the fused q/kv-RMSNorm producer), and for `wo_a` (is_bmm: DeepGEMM BMM is
SM100-only) the emulation kernel's BF16-at-load weight + `torch.bmm`.
Bodies: `mxfp8_gemm` (A16: bf16 x, e4m3 weight dequantized in registers,
fp32 accumulate — better numerics than vLLM's W8A8; the only wo_a route)
and `mxfp8_gemm_a8` (vLLM's MX activation quant in the prologue + FP8
`dot_scaled`). Cases: `cases(model)` keys on `deepseek-v4` in the model
name (shapes hard-coded from the checkpoint's config.json, TP8 then TP4,
buckets 8 / 192 / 32 / 64); `wo_a` cases carry `g: 1` (one launch per local
group). The lane swaps `modelopt.select_linear_kernel` (where vLLM picked
FlashInfer CUTLASS; `.engram.` prefixes excluded) and the bmm's
`init_mxfp8_linear_kernel`, and wraps `deep_gemm_fp8_o_proj` in
`deepseek_v41/nvidia/ops/o_proj.py`; M > max M and unplanned buckets stay on
vLLM's kernel. `lab_main(plan)` is a kernel-lab driver (runner-equivalent
autotune + graph timing vs vLLM's kernel, JSON lines on stdout).
`validate_weight` (load, fatal) refuses e4m3 NaN codes and e8m0 scales above 246 (the A16
decode builds 2^(ws - 119) from f32 bits). `lab_main` also times vLLM's Marlin MXFP8 W8A16 kernel
(`marlin_us`, `vs_marlin`) as a second reference. Checkpoint anatomy, ranking, drift-gate plans:
`docs/dsv41-dense.md`.

Split-K is two-stage only in serving (capture-safe): the Helion kernel writes fp32 partials
(`torch.empty` per call, graph-private under capture), host code in the same body then launches
`out.copy_(partial[:n_split].sum(0))` on the same stream; no semaphore, no counter, no module-level
scratch, so the lab times the composite that serves. The old last-arriver bodies (`mxfp8_gemm_sem`,
`mxfp8_gemm_a8_sem`) stay behind `SUFFIX_MXFP8_GEMM_SEMAPHORE=1` for lab and boot-gate comparison;
`prepare()` / `register()` raise with the knob set (unproven under CUDA-graph capture, glm g14). The
layer oracle poisons out / partials with NaN and replays a CUDA-graph capture bit-exactly per bucket;
the `mxfp8_capture_replay` boot gate (`python -m suffix_hybrid.kernels.mxfp8_linear capture_replay`,
sanitizer-ready) and the one-command lab session (`kernels/dsv41_dense_lab.py`: Markov bench first,
then the gate, then the cases) are described in `docs/dsv41-dense.md` section 3b / 5.
