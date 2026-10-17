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

## Results (inference-console lab, RTX PRO 6000, LFBO full effort)

Kernel time, Helion vs the oxide kernel at its production launch (> 1 = Helion faster).

| family | status | evidence |
|---|---|---|
| dsa_indexer | **wins** decode rows 1-192: 1.22-2.12x; configs shipped (`configs/helion/dsa_indexer_decode`) | GLM-5.3, runs 20261017T100330Z-...-9020, ...134337Z-...-af3f |
| k2_attn | **wins** hd256 sliding layers b1/b8: 1.22-1.53x (hd512 full layers still tuning) | gemma-4-26b/31b, runs ...145256Z-...-6a8c, ...145859Z-...-3050 |
| nvfp4_gemm | **loses** 0.17-0.62x on every shape: stays on oxide | Triton 3.8 `tl.dot_scaled` has no native NVFP4 MMA on sm_120 (tcgen05 is sm_100 only; cc 12.x decomposes to a bf16 upcast), so Helion cannot reach oxide's native block-scaled FP4 MMA. Revisit when Triton lowers sm_120 `mma.sync ... block_scale`. |
| nvfp4_moe, fp8_moe, ds_mla, kgdn1 | in the lab | |

Lessons that changed the ports: `hl.barrier()` persistent kernels deadlock on silicon (use
last-arriver merges or separate launches); Helion's autotuner checks a candidate's whole
output, so nothing may stay `torch.empty`; pass `autotune_baseline_fn`, because without it a
reference config runs in-process with no timeout; and `make()` must not build fresh weights per seed.
