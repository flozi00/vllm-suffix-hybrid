# SPDX-License-Identifier: Apache-2.0
"""One-GPU, full-depth, PER-RANK slice of DeepSeek-V4.1-Flash (served as `glm`) for vLLM.

Prod runs the model TP8 + expert-parallel 8 on 8 GPUs. Every kernel on one rank only sees that
rank's shapes, so one GPU can stand in for one rank: this tool writes a dummy-weight model dir whose
config.json holds ONE rank's dimensions at the full 40-layer depth (heads 64 -> 8, o_groups 8 -> 1,
experts 384 -> 48, vocab 129280 -> 16160, ...), boots vLLM on it with the prod serving flags and
the DSpark speculative config, and measures decode step time (and, with profile=true, the kernel
time shares) under whatever serving options `--opts` selects. Collectives do not exist in the
slice; everything else does. docs/dsv41-slice.md lists every place where the slice is NOT
per-rank-equivalent, with a ms/step estimate for each.

    python3 -m suffix_hybrid.tools.dsv41_slice --opts '{"conc":[1,8,32],"k":5}'
    python3 -m suffix_hybrid.tools.dsv41_slice --dry --opts '{}'     # config + plan, no vLLM import
    python3 -m suffix_hybrid.tools.dsv41_slice --print-schema        # every opt, bounds, default

stdout is ONE JSON object (schema "dsv41-slice/1"); logs go to stderr. Exit codes: 0 ok, 2 bad
opts/usage, 3 the engine refused the configuration (JSON `engine_error` holds vLLM's text verbatim),
4 deadline hit (partial JSON), 5 measurement failed, 6 a pinned vLLM source anchor drifted,
7 environment (no GPU / vLLM import failed), 1 unexpected.

Nothing here imports torch or vLLM at module level: the config-building half runs on a CPU box.
Written blind (no GPU, no vLLM on the author's machine): everything under "runtime" is read from the
vLLM source at the pinned commit, not executed. See the "Unverified" section of the doc.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from pathlib import Path

RESULT_SCHEMA = "dsv41-slice/1"  # the id of the result JSON (SCHEMA below is the opts table)

# --------------------------------------------------------------------------------------------
# Pinned facts. Everything below was read from vLLM at this commit (nightly of the live pool).
# --------------------------------------------------------------------------------------------
VLLM_COMMIT = "39983a5e7a5a575a7829f5d5ada50a33d9337ec1"
VLLM_VERSION_PREFIX = "0.32.0rc1.dev329"
PROD_TP = 8
PROD_EP = 8
ASSUMED_BW_GBS = 1400.0  # effective GDDR7 read bandwidth of one RTX PRO 6000 (docs/dsv41-comms.md)
# Live pool `glm`, 2026-10-19 (512k + host all-reduce roll): decode ms per engine step.
LIVE_STEP_MS = {1: 12.2, 8: 25.9, 32: 55.3}
# Estimated per-step collective time that the slice does not have (docs/dsv41-comms.md section 3:
# host all-reduce, 81 verify + 7 drafter all-reduces; section 11: vocab logits gather). Estimates.
COLLECTIVE_MS_BY_C = ((1, 1.75), (8, 7.0), (16, 13.1), (32, 25.1))

# The router kernel behind `fused_topk_bias` for sqrtsoftplus gating (csrc/moe/topk_softplus_sqrt_kernels.cu,
# switch at :786-841) is instantiated for exactly these expert counts and throws "Unsupported expert number"
# otherwise. Prod's target layers (384 experts, top-6) never reach it: `dsv4_topk` (Triton, router/dsv4_topk.py:26)
# takes 256/384 experts with top-6. A slice has neither, so its expert count must come from this table.
GATE_KERNEL_EXPERTS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576)
SLICE_EXPERTS = tuple(n for n in GATE_KERNEL_EXPERTS if 16 <= n <= 192)  # (16, 32, 64, 128, 192)
PROD_RANK_EXPERTS = 48  # 384 routed experts / EP8
CARD_GIB = 94.97  # RTX PRO 6000 Blackwell (Server Edition and Max-Q)
# Memory model, from the glm boot (ROLL.md, boot5.log) and the first slice boot (Model loading took 39.45 GiB
# at 48 experts = 38.46 modeled + 0.99): GiB per card outside the weights and the KV pool.
UNMODELLED_GIB = 0.99  # rope caches, buffers: logged "Model loading took" minus rank_modules()
VISION_GIB = 0.97  # prod logged 40.66 - 38.70 modeled - 0.99 unmodelled: the ViT + aligner the slice does not build
NONTORCH_GIB = 1.72  # CUDA context etc. (ROLL.md: 42.38 weights+non-torch against 40.66 weights)
ACTIVATION_GIB = 5.66  # profile run at max_num_batched_tokens=8192 (ROLL.md)
GRAPH_GIB = 2.69  # CUDA graph pool (boot5.log: "Graph capturing finished ... took 2.69 GiB")
KV_TOKENS_PER_GIB = 292000  # glm: 10,867,561 tokens in 37.17 GiB at 262144 (390k at 524288): the lower, hybrid-cache figure
KV_MIN_GIB = 1.0  # below this the weights leave no KV pool: refused at parse time
KV_WARN_GIB = 8.0

# Exact bytes of LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4 config.json (hf download, 2026-10-19).
REAL_CONFIG_SHA256 = "fbb1660e71f4d05dad68aa1849188791a1e80a8b1d2a031ce543ff24c65a448e"
_REAL_CONFIG_JSON = r'''{
  "architectures": [
    "DeepseekV41ForCausalLM"
  ],
  "bos_token_id": 0,
  "dtype": "bfloat16",
  "eos_token_id": 1,
  "image_token_id": 129264,
  "model_type": "deepseek_v41",
  "pad_token_id": 2,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "config_groups": {
      "group_0": {
        "input_activations": {
          "dynamic": false,
          "group_size": 16,
          "num_bits": 4,
          "type": "float"
        },
        "targets": [
          "Linear"
        ],
        "weights": {
          "dynamic": false,
          "group_size": 16,
          "num_bits": 4,
          "type": "float"
        }
      }
    },
    "expert_dtype": "fp4",
    "group_size": 16,
    "ignore": [
      "*.attn.*",
      "*.ffn.shared_experts.*",
      "head",
      "mtp.*"
    ],
    "kv_cache_quant_algo": null,
    "moe_quant_algo": "NVFP4",
    "producer": {
      "name": "modelopt",
      "version": "dsv4-nvfp4-experts"
    },
    "quant_algo": "MIXED_PRECISION",
    "quant_method": "fp8",
    "quantized_layers": {
      "layers.0.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.1.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.10.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.11.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.12.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.13.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.14.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.15.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.16.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.17.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.18.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.19.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.2.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.20.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.21.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.22.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.23.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.24.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.25.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.26.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.27.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.28.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.29.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.3.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.30.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.31.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.32.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.33.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.34.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.35.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.36.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.37.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.38.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.39.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.4.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.5.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.6.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.7.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.8.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      },
      "layers.9.ffn.experts": {
        "group_size": 16,
        "quant_algo": "NVFP4"
      }
    },
    "scale_fmt": "ue8m0",
    "weight_block_size": [
      32,
      32
    ],
    "engram_embed_quant": "mxfp4",
    "engram_embed_block_size": 32,
    "engram_embed_scale_dtype": "ue8m0",
    "engram_embed_layers": [
      1,
      14
    ]
  },
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "candidate_block_size": 8,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "compress_ratios": [
      0,
      0,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    "compress_rope_theta": 160000,
    "dspark_block_size": 5,
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_noise_token_id": 128799,
    "dspark_num_experts_per_tok": 3,
    "dspark_target_layer_ids": [
      37,
      38,
      39
    ],
    "engram_compressed_vocab_size": 99092,
    "engram_head_dim": 256,
    "engram_layer_ids": [
      1,
      14
    ],
    "engram_max_ngram_size": 4,
    "engram_n_heads": 8,
    "engram_num_embeddings": [
      384006168,
      384016682
    ],
    "engram_pad_token_id": 2,
    "engram_vocab_size": 16000000,
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_source_layer_ids": [
      2,
      8,
      14,
      20,
      24,
      28,
      32,
      36
    ],
    "index_topk": 512,
    "initializer_range": 0.02,
    "kv_source_layer_ids": [
      2,
      8,
      14,
      20
    ],
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v41_text",
    "moe_intermediate_size": 2304,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "norm_topk_prob": true,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 40,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 3,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1280,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-20,
    "rope_scaling": {
      "beta_fast": 32,
      "beta_slow": 1,
      "factor": 16,
      "original_max_position_embeddings": 65536,
      "rope_type": "yarn"
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": false,
    "topk_method": "noaux_tc",
    "use_cache": true,
    "vocab_size": 129280
  },
  "transformers_version": "5.6.0",
  "vision_config": {
    "downsample_ratio": 3,
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "max_image_tokens": 1024,
    "max_wh_ratio": null,
    "min_pixels": 295936,
    "model_type": "deepseek_v41_vision",
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "rope_theta": 10000
  }
}
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if _sha256(_REAL_CONFIG_JSON.encode()) != REAL_CONFIG_SHA256:  # fail loud on an edited literal
    raise ImportError("dsv41_slice: embedded V4.1 config.json no longer matches REAL_CONFIG_SHA256")
REAL_CONFIG: dict = json.loads(_REAL_CONFIG_JSON)


def real_config() -> dict:
    return copy.deepcopy(REAL_CONFIG)


class SliceError(Exception):
    """A refusal with an exit code; the message is what the operator reads."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------------
# Measured acceptance profiles (live glm, 2026-10-19; cumulative P(first i+1 drafts accepted)).
# These are vLLM's `synthetic_acceptance_rates` semantics (unconditional, non-increasing).
# --------------------------------------------------------------------------------------------
SYNTHETIC_PROFILES = {
    "chat": (0.61, 0.34, 0.17, 0.09, 0.05),
    "thinking": (0.86, 0.71, 0.58, 0.46, 0.375),
    "edit": (0.89, 0.73, 0.53, 0.33, 0.19),
    "rag8k": (0.67, 0.44, 0.26, 0.14, 0.08),
}

# --------------------------------------------------------------------------------------------
# Environment allowlists. The plugin reads SUFFIX_* at interpreter start (sitecustomize), so the
# tool re-execs itself with the full env; only these keys may be set, with shell-inert values.
# Excluded on purpose: SUFFIX_ROCM_* (AMD), SUFFIX_NCCL_*/SUFFIX_HAR_* (collectives: none at TP=1),
# SUFFIX_MTP_TUNE* (online weight tuning), SUFFIX_HYBRID_* (suffix-decoding wrap, not DSpark),
# markers/_DONE flags, SUFFIX_HYBRID_DEV_ARGS (argv splice). Snapshot of plugin af97edb7.
# --------------------------------------------------------------------------------------------
ALLOWED_SUFFIX_ENV = frozenset("""
SUFFIX_PRESET SUFFIX_DSV41_ENGRAM_MXFP4 SUFFIX_DSV41_SM120_BLOCK64 SUFFIX_DSV41_FUSE
SUFFIX_DSV41_ATTN_GATE SUFFIX_DSV41_ATTN_ROUTE SUFFIX_DSV41_HC_BOUNDARY SUFFIX_DSV41_HC_QUANT
SUFFIX_DSV41_MOE_FUSE_SHARED SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M SUFFIX_DSV41_MOE_FUSE_SHARED_POLE_TOL
SUFFIX_FI_PLAN_HOSTFREE
SUFFIX_HELION_DSA_INDEXER SUFFIX_HELION_DSV41_ATTN SUFFIX_HELION_DSV41_ENGRAM SUFFIX_HELION_DSV41_HC
SUFFIX_HELION_DSV41_ROUTER SUFFIX_HELION_DSV41_SWIGLU SUFFIX_HELION_DS_MLA SUFFIX_HELION_FP8_MOE
SUFFIX_HELION_K2_ATTN SUFFIX_HELION_MXFP8_GEMM SUFFIX_HELION_NVFP4_GEMM SUFFIX_HELION_NVFP4_GEMM_FP4MMA
SUFFIX_HELION_NVFP4_GEMM_FP4MMA_STRICT SUFFIX_HELION_NVFP4_MOE SUFFIX_HELION_NVFP4_MOE_FP4MMA
SUFFIX_OXIDE_PROBE
SUFFIX_SM120 SUFFIX_SM120_DSA_INDEXER SUFFIX_SM120_DSA_INDEXER_CAND SUFFIX_SM120_DSA_INDEXER_CANDSEL
SUFFIX_SM120_DSA_INDEXER_GROUP SUFFIX_SM120_DSA_INDEXER_MIN_TILES SUFFIX_SM120_DSA_INDEXER_STRICT
SUFFIX_SM120_HISPARSE_MIXED SUFFIX_SM120_HISPARSE_MTP SUFFIX_SM120_HISPARSE_NO_MIRROR_STAGING
SUFFIX_SM120_HISPARSE_OWN_RESOLVER SUFFIX_SM120_HISPARSE_PREFETCH SUFFIX_SM120_NVP4DSMLA SUFFIX_SM120_NVP4KV
SUFFIX_DSA_INDEXER_CTX SUFFIX_DSA_INDEXER_GEOMETRY SUFFIX_DSA_INDEXER_STOCK
SUFFIX_NVFP4_MOE SUFFIX_NVFP4_MOE_MAX_M SUFFIX_NVFP4_MOE_TUNE SUFFIX_NVFP4_MOE_BACKEND SUFFIX_MTP_MOE_NVFP4
SUFFIX_MOE_FUSED SUFFIX_FP4MMA_MOE_SCALE_READ SUFFIX_FP4MMA_MOE_Y_MB SUFFIX_FP8_MOE SUFFIX_FP8_MOE_MAX_M
SUFFIX_FP8_DENSE SUFFIX_FP8_DENSE_BACKEND SUFFIX_FP8_DENSE_LAYERS SUFFIX_FP8_DENSE_MARLIN_MAX_M
SUFFIX_NVFP4_DENSE SUFFIX_NVFP4_DENSE_ACT_AMAX SUFFIX_NVFP4_DENSE_LAYERS
SUFFIX_NVFP4_GEMM SUFFIX_NVFP4_GEMM_ROUTE SUFFIX_NVFP4_GEMM_MAX_M SUFFIX_NVFP4_GEMM_FUSED
SUFFIX_NVFP4_GEMM_FUSED_MAX_M SUFFIX_NVFP4_GEMM_PF SUFFIX_NVFP4_GEMM_AUTOROUTE
SUFFIX_NVFP4_GEMM_AUTOROUTE_MARGIN SUFFIX_NVFP4_GEMM_AUTOROUTE_CACHE
SUFFIX_NVFP4_LMHEAD SUFFIX_NVFP4_LMHEAD_BACKEND SUFFIX_NVFP4_LMHEAD_MAX_M SUFFIX_NVFP4_LMHEAD_GRAPH
SUFFIX_MXFP4_LMHEAD SUFFIX_MXFP4_LMHEAD_MAX_M SUFFIX_MXFP4_LMHEAD_GRAPH
SUFFIX_MXFP8_GEMM_MAX_M SUFFIX_MXFP8_GEMM_SEMAPHORE
SUFFIX_BF16_SKINNY SUFFIX_BF16_SKINNY_LAYERS SUFFIX_BF16_SKINNY_MAX_M
SUFFIX_SILU_GRID_CLAMP SUFFIX_SILU_SKIP_PAD SUFFIX_POW2_MIRROR
SUFFIX_SAMPLER_WARMUP SUFFIX_PREFILL_WARMUP SUFFIX_PREFILL_WARMUP_SIZES SUFFIX_JIT_LOG SUFFIX_FICACHE
SUFFIX_KERNELS SUFFIX_KERNELS_OPS
""".split())
# VLLM_* knobs that exist at the pinned commit (vllm/envs.py) and are safe to flip in a slice run.
ALLOWED_VLLM_ENV = frozenset("""
VLLM_USE_V2_MODEL_RUNNER VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
VLLM_DISABLE_SHARED_EXPERTS_STREAM VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE VLLM_USE_DEEP_GEMM VLLM_USE_DEEP_GEMM_E8M0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES VLLM_DEEP_GEMM_WARMUP VLLM_USE_FLASHINFER_SAMPLER
VLLM_COMPUTE_NANS_IN_LOGITS VLLM_ENABLE_CUDAGRAPH_GC VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE VLLM_LOGGING_LEVEL VLLM_LOG_STATS_INTERVAL
VLLM_CUSTOM_SCOPES_FOR_PROFILING VLLM_NVTX_SCOPES_FOR_PROFILING VLLM_TRITON_JIT_WARMUP_NUM_THREADS
""".split())
# Set by the tool itself; an opts.env entry for one of these is refused (they carry design decisions).
TOOL_OWNED_ENV = frozenset({"VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_LOGGING_STREAM",
                            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "CUDA_VISIBLE_DEVICES"})
_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:/*+@~=-]{0,1024}$")
# Mirrors the prod pool env (d7-spec.json) minus the parts that need >1 GPU (host all-reduce).
DEFAULT_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "SUFFIX_PRESET": "dsv41-sm120",  # Engram MXFP4 overlay + block-64 hook; dsv41-sm120-comms adds HAR
    "MALLOC_ARENA_MAX": "16",
    "MALLOC_MMAP_THRESHOLD_": "131072",
    "HF_HUB_OFFLINE": "1",  # the model dir is local and the lab has no internet: fail fast, never hang
    "TRANSFORMERS_OFFLINE": "1",
}
_REEXEC_MARK = "DSV41_SLICE_ENV_APPLIED"


# --------------------------------------------------------------------------------------------
# Engram hash-table layout (mirror of vllm EngramLayout.__init__ at VLLM_COMMIT)
# --------------------------------------------------------------------------------------------
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):  # deterministic Miller-Rabin below 2**32
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set) -> int:
    cand = start + 1
    while not _is_prime(cand) or cand in seen:
        cand += 1
    return cand


def engram_primes(layer_ids, max_ngram: int, n_heads: int, vocab_size: int):
    """[layer][ngram size - 2][head] bucket-range sizes, drawn exactly like vLLM does; the
    table of a layer has sum(sizes) rows, and config.engram_num_embeddings is that sum."""
    seen: set = set()
    out = []
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur = _next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(tuple(sizes))
        out.append(tuple(per_ngram))
    return tuple(out)


def engram_rows(layer_ids, max_ngram: int, n_heads: int, vocab_size: int) -> list:
    return [sum(sum(g) for g in layer)
            for layer in engram_primes(layer_ids, max_ngram, n_heads, vocab_size)]


# --------------------------------------------------------------------------------------------
# opts
# --------------------------------------------------------------------------------------------
MOE_BACKENDS = ("auto", "flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_trtllm",
                "flashinfer_b12x", "b12x", "cutlass", "triton", "marlin", "humming", "emulation")
LINEAR_BACKENDS = ("auto", "cutlass", "flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_trtllm",
                   "flashinfer_cudnn", "flashinfer_b12x", "b12x", "marlin", "triton", "deep_gemm",
                   "torch", "emulation")
TOPK_BACKENDS = ("auto", "deep_select", "cooperative", "persistent", "per_row", "flashinfer", "torch")
CUDAGRAPH_MODES = ("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
MAX_ENGRAM_HOST_GIB = 32.0


def _spec(kind, default, doc, lo=None, hi=None, enum=None, nullable=False):
    return {"type": kind, "default": default, "doc": doc, "min": lo, "max": hi, "enum": enum,
            "nullable": nullable}


SCHEMA = {
    # what to measure
    "conc": _spec("int_list", [1, 8, 32], "closed-loop stream counts (each <= max_num_seqs, 1..8 entries)", 1, 256),
    "prompt_tokens": _spec("int", 2048, "prompt length per request (random token ids) so KV is realistic", 16, 262144),
    "warmup_steps": _spec("int", 16, "pure-decode engine steps after all streams started, excluded", 0, 1000),
    "measure_steps": _spec("int", 64, "engine steps timed per concurrency", 4, 4000),
    "temperature": _spec("float", 0.2, "sampling temperature (prod default override is 0.2)", 0.0, 2.0),
    "profile": _spec("bool", False, "also collect a torch.profiler kernel table per concurrency"),
    "profile_steps": _spec("int", 8, "engine steps inside the profiler window", 1, 64),
    # speculative decoding (DSpark)
    "k": _spec("int", 5, "num_speculative_tokens; 0 = no speculation (plain decode baseline)", 0, 8),
    "draft_sample_method": _spec("enum", "greedy", "draft sampling", enum=("greedy", "probabilistic")),
    "rejection_sample_method": _spec("enum", None, "default: synthetic when k>0 (dummy weights accept ~nothing otherwise)",
                                     enum=("standard", "synthetic", "block"), nullable=True),
    "synthetic": _spec("synthetic", None, "profile name chat|thinking|edit|rag8k or a list of k cumulative rates; "
                       "default chat", nullable=True),
    "synthetic_acceptance_length": _spec("float", None, "mean accepted length in [1,k+1] (exclusive with synthetic)",
                                         1.0, 9.0, nullable=True),
    "enable_adaptive_verification": _spec("bool", False, "vLLM adaptive verification (confidence head is random here!)"),
    "num_speculative_tokens_per_batch_size": _spec("schedule", None, "[[start,end,k],...] inclusive batch ranges, "
                                                   "first start 1", nullable=True),
    # serving flags (prod values)
    "gpu_memory_utilization": _spec("float", 0.93, "--gpu-memory-utilization", 0.30, 0.98),
    "max_model_len": _spec("int", 524288, "--max-model-len", 4096, 1048576),
    "max_num_seqs": _spec("int", 32, "--max-num-seqs", 1, 256),
    "max_num_batched_tokens": _spec("int", 8192, "--max-num-batched-tokens", 512, 131072),
    "block_size": _spec("enum", 64, "--block-size (SM12x: 64 needs the plugin's block-64 hook)", enum=(32, 64)),
    "kv_cache_dtype": _spec("enum", "fp8", "--kv-cache-dtype", enum=("fp8", "auto")),
    "enable_prefix_caching": _spec("bool", True, "prod runs with prefix caching on"),
    "cudagraph_mode": _spec("enum", None, "compilation_config.cudagraph_mode; default = what the arch picks "
                            "(prod: FULL_AND_PIECEWISE)", enum=CUDAGRAPH_MODES, nullable=True),
    "async_scheduling": _spec("bool", None, "default: vLLM decides (prod)", nullable=True),
    "moe_backend": _spec("enum", "auto", "--moe-backend (prod: auto -> FLASHINFER_CUTLASS NVFP4)", enum=MOE_BACKENDS),
    "linear_backend": _spec("enum", "auto", "--linear-backend", enum=LINEAR_BACKENDS),
    "sparse_indexer_topk_backend": _spec("enum", "auto", "--sparse-indexer-topk-backend", enum=TOPK_BACKENDS),
    "enable_flashinfer_autotune": _spec("bool", None, "kernel_config.enable_flashinfer_autotune", nullable=True),
    # slice design
    "moe_routing": _spec("enum", "model", "model = the dummy-weight router (prod kernels); uniform_random/"
                         "normal_routing = vLLM's routing simulator (balanced / skewed; adds rand+topk kernels)",
                         enum=("model", "uniform_random", "normal_routing")),
    "engram_heads": _spec("enum", 1, "Engram hash heads: 1 = the per-rank 3 columns (prod rank looks up 3 of 24); "
                          "8 = all 24 columns at TP=1 (8x lookup rows and wkv bytes)", enum=(1, 8)),
    "engram_rows": _spec("int", 1048576, "Engram rows per hash column (prod: 16,000,000; pinned host memory)",
                         1024, 16000000),
    "slice_experts": _spec("experts", 64, "routed experts RESIDENT per layer. The model router kernel only exists for "
                           "16|32|64|128|192 here (csrc/moe/topk_softplus_sqrt_kernels.cu:786-841), so the EP8 rank's 48 is "
                           "impossible; 128 and 192 do not fit the card. moe_routing=uniform_random|normal_routing bypasses "
                           "the kernel and accepts any count in 16-192", 16, 192),
    "active_experts": _spec("int", None, "experts the router may pick; the other slice_experts - active_experts stay "
                            "resident but get gate bias -1e4 right after the engine is up, so the weight traffic is the "
                            "EP8 rank's. null = 48 (moe_routing=model) or all (the simulator)", 1, 192, nullable=True),
    "moe_topk": _spec("int", 1, "routed experts per token in the slice (prod rank: 6/8 = 0.75 pairs; 1 is the nearest "
                      "integer above; raise it only if a backend refuses top-1)", 1, 6),
    "shared_expert_tp": _spec("bool", True, "patch the shared expert to its TP8 width (288) in-process; "
                              "false = full 2304 width (+~1.0 ms/step)"),
    "seed": _spec("int", 0, "engine seed and prompt rng seed", 0, 2 ** 31 - 1),
    "env": _spec("env", {}, "extra env, allowlisted SUFFIX_*/VLLM_* keys only"),
    "deadline_s": _spec("int", 7200, "wall clock after which the partial JSON is printed (exit 4)", 60, 43200),
    "work_dir": _spec("path", None, "parent of the temp model dir (default: system tmp)", nullable=True),
    "log_level": _spec("enum", "INFO", "VLLM_LOGGING_LEVEL", enum=("DEBUG", "INFO", "WARNING")),
}


def _bad(msg: str) -> SliceError:
    return SliceError(f"opts: {msg}", 2)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _coerce(name: str, v, spec: dict):
    kind = spec["type"]
    if v is None:
        if spec["nullable"]:
            return None
        raise _bad(f"{name} may not be null")
    lo, hi = spec["min"], spec["max"]
    if kind == "bool":
        if not isinstance(v, bool):
            raise _bad(f"{name} must be true/false, got {v!r}")
        return v
    if kind in ("int", "experts"):
        if not _is_int(v) or not lo <= v <= hi:
            raise _bad(f"{name} must be an integer in [{lo}, {hi}], got {v!r}")
        return v
    if kind == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise _bad(f"{name} must be a number in [{lo}, {hi}], got {v!r}")
        return float(v)
    if kind == "enum":
        if v not in spec["enum"] or isinstance(v, bool):
            raise _bad(f"{name} must be one of {list(spec['enum'])}, got {v!r}")
        return v
    if kind == "int_list":
        if not isinstance(v, list) or not 1 <= len(v) <= 8 or not all(_is_int(x) and lo <= x <= hi for x in v):
            raise _bad(f"{name} must be a list of 1..8 integers in [{lo}, {hi}], got {v!r}")
        if len(set(v)) != len(v):
            raise _bad(f"{name} has duplicates: {v!r}")
        return sorted(v)
    if kind == "synthetic":
        if isinstance(v, str):
            if v not in SYNTHETIC_PROFILES:
                raise _bad(f"synthetic must be one of {sorted(SYNTHETIC_PROFILES)} or a list of rates, got {v!r}")
            return v
        if (isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                              and math.isfinite(x) and 0.0 <= x <= 1.0 for x in v)):
            return [float(x) for x in v]
        raise _bad(f"synthetic must be a profile name or a non-empty list of rates in [0,1], got {v!r}")
    if kind == "schedule":
        return validate_schedule(v)
    if kind == "env":
        return validate_env(v)
    if kind == "path":
        if not isinstance(v, str) or not v.startswith("/") or len(v) > 400 or "\x00" in v:
            raise _bad(f"{name} must be an absolute path, got {v!r}")
        return v
    raise AssertionError(kind)


def validate_schedule(v) -> list:
    """num_speculative_tokens_per_batch_size, validated like vllm.v1.spec_decode.dynamic.utils."""
    if not isinstance(v, list) or not v:
        raise _bad("num_speculative_tokens_per_batch_size must be a non-empty list of [start, end, k]")
    out = []
    for e in v:
        if not isinstance(e, (list, tuple)) or len(e) != 3 or not all(_is_int(x) for x in e):
            raise _bad(f"schedule entry {e!r} must be [start, end, k] with integers")
        a, b, kk = e
        if not 1 <= a <= b <= 4096 or not 0 <= kk <= 8:
            raise _bad(f"schedule entry {e!r}: need 1 <= start <= end <= 4096 and 0 <= k <= 8")
        out.append([a, b, kk])
    out.sort()
    prev = 0
    for a, b, _ in out:
        if a <= prev:
            raise _bad("schedule ranges must not overlap")
        prev = b
    if out[0][0] != 1:
        raise _bad("the first schedule range must start at batch size 1")
    return out


def effective_k(o: dict, c: int) -> int:
    """Draft tokens per stream the scheduler uses at batch size c: vLLM's build_dynamic_sd_schedule_lookup
    (v1/spec_decode/dynamic/utils.py:74-140): ranges are inclusive, a gap or the tail carries the previous k forward,
    every k is clamped to num_speculative_tokens. No schedule: k."""
    k, sched = o["k"], o["num_speculative_tokens_per_batch_size"]
    if not sched:
        return k
    last = sched[0][2]
    for a, b, kk in sched:
        if c < a:
            break  # in the gap before this range: the previous range's k carries forward
        last = kk
        if c <= b:
            break
    return min(k, last)


def validate_env(v) -> dict:
    if not isinstance(v, dict):
        raise _bad("env must be an object of string -> string")
    out = {}
    for key, val in v.items():
        if not isinstance(key, str) or not isinstance(val, str):
            raise _bad(f"env {key!r}: keys and values must be strings")
        if key in TOOL_OWNED_ENV:
            raise _bad(f"env {key}: set by the tool (see --print-schema); use the matching opt")
        if key not in ALLOWED_SUFFIX_ENV and key not in ALLOWED_VLLM_ENV:
            hint = ("collective knobs (SUFFIX_NCCL_*/SUFFIX_HAR_*) have no effect at TP=1"
                    if key.startswith(("SUFFIX_NCCL_", "SUFFIX_HAR_")) else
                    "to allow a knob, add it to ALLOWED_SUFFIX_ENV / ALLOWED_VLLM_ENV in dsv41_slice.py")
            raise _bad(f"env {key} is not on the allowlist ({hint})")
        if not _ENV_VALUE_RE.match(val):
            raise _bad(f"env {key}: value {val!r} has characters outside [A-Za-z0-9_.,:/*+@~=-] or is too long")
        out[key] = val
    return out


def max_tokens_for(o: dict) -> int:
    """Per-request max_tokens that keeps every stream alive through all timed windows."""
    steps = o["warmup_steps"] + o["measure_steps"] + (o["profile_steps"] if o["profile"] else 0) + 8
    return steps * (o["k"] + 1) + 1


def synthetic_rates(o: dict):
    """The cumulative per-position acceptance vector for rejection_sample_method=synthetic, or None."""
    if o["rejection_sample_method"] != "synthetic":
        return None
    k = o["k"]
    if o["synthetic_acceptance_length"] is not None:
        length = o["synthetic_acceptance_length"]
        if not 1.0 <= length <= k + 1:
            raise _bad(f"synthetic_acceptance_length must be in [1, {k + 1}] for k={k}, got {length}")
        drafts = length - 1
        full = int(drafts)
        return ([1.0] * full + [drafts - full] + [0.0] * (k - full - 1))[:k]
    syn = o["synthetic"]
    if isinstance(syn, str):
        rates = list(SYNTHETIC_PROFILES[syn])
        if k > len(rates):
            raise _bad(f"profile {syn!r} has {len(rates)} positions; pass an explicit list for k={k}")
        rates = rates[:k]
    else:
        rates = list(syn)
        if len(rates) != k:
            raise _bad(f"synthetic needs exactly k={k} rates, got {len(rates)}")
    if any(rates[i] > rates[i - 1] for i in range(1, len(rates))):
        raise _bad(f"synthetic rates must be non-increasing (cumulative), got {rates}")
    return rates


def engram_host_gib(o: dict, text: dict | None = None) -> float:
    """Pinned host memory of the slice's Engram tables (MXFP4 rows: dim/2 + dim/32 bytes)."""
    t = text or REAL_CONFIG["text_config"]
    cols = (t["engram_max_ngram_size"] - 1) * o["engram_heads"]
    row = t["engram_head_dim"] // 2 + t["engram_head_dim"] // 32
    return len(t["engram_layer_ids"]) * cols * o["engram_rows"] * row / 2 ** 30


def parse_opts(raw) -> dict:
    """Validate and complete opts. Every number is bounded, every string an enum, env allowlisted."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _bad("must be a JSON object")
    unknown = sorted(set(raw) - set(SCHEMA))
    if unknown:
        raise _bad(f"unknown keys {unknown}; valid keys: {sorted(SCHEMA)}")
    o = {name: _coerce(name, raw[name] if name in raw else copy.deepcopy(spec["default"]), spec)
         for name, spec in SCHEMA.items()}
    if max(o["conc"]) > o["max_num_seqs"]:
        raise _bad(f"conc {o['conc']} exceeds max_num_seqs={o['max_num_seqs']}")
    k = o["k"]
    spec_keys = ("rejection_sample_method", "synthetic", "synthetic_acceptance_length",
                 "num_speculative_tokens_per_batch_size")
    if k == 0:
        extra = [n for n in spec_keys if o[n] is not None]
        extra += [n for n in ("draft_sample_method", "enable_adaptive_verification")
                  if o[n] != SCHEMA[n]["default"]]
        if extra:
            raise _bad(f"k=0 means no speculation; remove {extra}")
    else:
        if o["rejection_sample_method"] is None:
            o["rejection_sample_method"] = "synthetic"
        if o["rejection_sample_method"] == "synthetic":
            if o["synthetic"] is not None and o["synthetic_acceptance_length"] is not None:
                raise _bad("synthetic and synthetic_acceptance_length are mutually exclusive")
            if o["synthetic"] is None and o["synthetic_acceptance_length"] is None:
                o["synthetic"] = "chat"
        elif o["synthetic"] is not None or o["synthetic_acceptance_length"] is not None:
            raise _bad(f"synthetic acceptance needs rejection_sample_method=synthetic, got "
                       f"{o['rejection_sample_method']!r}")
        for a, b, kk in o["num_speculative_tokens_per_batch_size"] or []:
            if kk > k:
                raise _bad(f"schedule k={kk} exceeds k={k}")
        synthetic_rates(o)  # raises on a bad length / rate vector
    need = o["prompt_tokens"] + max_tokens_for(o) + 1
    if need > o["max_model_len"]:
        raise _bad(f"prompt_tokens + decode budget = {need} exceeds max_model_len={o['max_model_len']}")
    if o["max_num_batched_tokens"] < o["block_size"]:
        raise _bad("max_num_batched_tokens must be >= block_size")
    gib = engram_host_gib(o)
    if gib > MAX_ENGRAM_HOST_GIB:
        raise _bad(f"engram_rows={o['engram_rows']} x engram_heads={o['engram_heads']} would pin "
                   f"{gib:.1f} GiB of host memory (limit {MAX_ENGRAM_HOST_GIB:.0f})")
    _check_experts(o)
    broken = [c for c in kernel_contracts(o) if not c["ok"]]
    if broken:
        raise _bad("kernel contract violated: " + "; ".join(f"{c['id']}: {c['detail']} [{c['source']}]" for c in broken))
    mem = memory_plan(o)
    if mem["kv_gib_est"] < KV_MIN_GIB:
        raise _bad(f"slice_experts={o['slice_experts']}: the weights alone need ~{mem['weights_gib_logged_est']} GiB and "
                   f"the card budget is {mem['budget_gib']} GiB ({o['gpu_memory_utilization']} x {CARD_GIB}); with "
                   f"{NONTORCH_GIB + ACTIVATION_GIB + GRAPH_GIB:.1f} GiB of context/activations/graphs that leaves "
                   f"{mem['kv_gib_est']} GiB for KV. Use slice_experts 64 or fewer")
    return o


def _check_experts(o: dict) -> None:
    """slice_experts must have a router kernel instance (model routing); active_experts is a subset of it."""
    n, routing = o["slice_experts"], o["moe_routing"]
    if routing == "model" and n not in SLICE_EXPERTS:
        raise _bad(f"slice_experts={n}: the model router kernel (csrc/moe/topk_softplus_sqrt_kernels.cu:786-841) is "
                   f"instantiated only for {list(SLICE_EXPERTS)} experts in this range (the EP8 rank's {PROD_RANK_EXPERTS} "
                   "throws 'Unsupported expert number'); moe_routing=uniform_random|normal_routing bypasses the kernel "
                   "and takes any count in 16-192")
    act = o["active_experts"]
    if act is None:
        act = min(n, PROD_RANK_EXPERTS) if routing == "model" else n
    elif routing != "model" and act != n:
        raise _bad("active_experts masks the model router's gate bias; the routing simulator picks over all "
                   f"slice_experts={n}")
    if not o["moe_topk"] <= act <= n:
        raise _bad(f"active_experts={act} must be in [moe_topk={o['moe_topk']}, slice_experts={n}]")
    o["active_experts"] = act


def schema_doc() -> dict:
    return {name: {key: spec[key] for key in ("type", "default", "min", "max", "enum", "nullable", "doc")
                   if spec[key] is not None or key in ("default",)} for name, spec in SCHEMA.items()}


# --------------------------------------------------------------------------------------------
# Per-rank config derivation
# --------------------------------------------------------------------------------------------
def _pad64(v: int) -> int:
    return -(-v // 64) * 64


def vocab_per_rank(vocab: int, tp: int = PROD_TP) -> int:
    """VocabParallelEmbedding pads the vocab to a multiple of 64, then splits it over TP."""
    return _pad64(vocab) // tp


def slice_text(real_text: dict, o: dict, tp: int = PROD_TP, ep: int = PROD_EP):
    """text_config of ONE rank at TP=tp / EP=ep, as a TP=1 model. Returns (text, rules)."""
    t = copy.deepcopy(real_text)
    rules = []

    def put(key, new, rule, why):
        rules.append({"key": key, "real": t.get(key), "slice": new, "rule": rule, "why": why})
        t[key] = new

    for key, div in (("num_attention_heads", tp), ("o_groups", tp), ("n_routed_experts", ep),
                     ("dspark_n_routed_experts", ep)):
        if t[key] % div:
            raise SliceError(f"{key}={t[key]} is not divisible by {div}", 1)
    per_rank = t["n_routed_experts"] // ep
    put("num_attention_heads", t["num_attention_heads"] // tp, "heads/TP",
        "wq_b is column-parallel over heads*head_dim; attention n_local_heads = heads/TP")
    put("o_groups", t["o_groups"] // tp, "o_groups/TP",
        "wo_a (out) and wo_b (in) shard by group: n_local_groups = o_groups/TP; wo_a input = heads*head_dim/o_groups stays 4096")
    put("n_routed_experts", o["slice_experts"], "slice_experts (EP8 rank: %d)" % per_rank,
        "EP8 = %d whole experts per rank at the full moe_intermediate_size (2304, unchanged). The slice holds slice_experts "
        "because the router kernel (csrc/moe/topk_softplus_sqrt_kernels.cu:786-841) has no instance for %d; only "
        "active_experts=%d of them are ever selected (gate bias mask), so the per-step weight traffic is the rank's"
        % (per_rank, per_rank, o["active_experts"]))
    put("num_experts_per_tok", o["moe_topk"], "pairs per token -> moe_topk",
        "a prod rank sees top_k/EP = 0.75 expert pairs per token; the integer nearest above is 1 "
        "(+33% pairs, touched experts +32%/+20%/+3% at M=6/48/192: see non-equivalences)")
    put("vocab_size", vocab_per_rank(t["vocab_size"], tp), "pad64(vocab)/TP",
        "embed_tokens and lm_head are vocab-parallel; the sampler vocabulary shrinks with it (non-equivalence)")
    put("dspark_n_routed_experts", t["dspark_n_routed_experts"] // ep, "draft experts/EP",
        "the DSpark layers are MoE layers with their own expert count, EP-sharded the same way")
    put("dspark_num_experts_per_tok", min(o["moe_topk"], t["dspark_n_routed_experts"]), "pairs per token -> moe_topk",
        "draft top-3/EP8 = 0.375 -> 1")
    per = vocab_per_rank(real_text["vocab_size"], tp)
    noise = real_text["dspark_noise_token_id"]
    put("dspark_noise_token_id", noise - (noise // per) * per, "local index on the owning shard",
        "the draft embeds this id; vocab_parallel zeroes ids outside a rank's shard, so one valid in-range id stands in")
    heads = o["engram_heads"]
    layers = tuple(t["engram_layer_ids"])
    put("engram_n_heads", heads, "per-rank hash columns",
        "a prod rank looks up cdiv(24, TP)=3 of the 24 hash columns; 1 head x 3 n-gram sizes = 3 columns "
        "(its wkv [768 -> 25600] has the same bytes as the rank's [6144 -> 3200])" if heads == 1 else
        "all 24 hash columns on one GPU (8x the per-rank lookup rows and wkv bytes)")
    put("engram_vocab_size", o["engram_rows"], "table rows per hash column",
        "bucket-range prime base; prod 16,000,000 rows per column in pinned host memory")
    put("engram_num_embeddings", engram_rows(layers, t["engram_max_ngram_size"], heads, o["engram_rows"]),
        "sum of the prime ranges", "must equal the sum of the drawn primes (vLLM asserts sum <= num_embeddings)")
    put("engram_compressed_vocab_size", t["vocab_size"], "= synthetic tokenizer's compressed vocab",
        "NgramHashState builds the id map from a tokenizer and refuses a size mismatch; the tool ships a "
        "tokenizer of vocab_size distinct tokens")
    return t, rules


def slice_config(o: dict, tp: int = PROD_TP, ep: int = PROD_EP):
    """The full config.json of the slice (text_config rewritten, everything else verbatim)."""
    cfg = real_config()
    cfg["text_config"], rules = slice_text(cfg["text_config"], o, tp, ep)
    return cfg, rules


def memory_plan(o: dict) -> dict:
    """GiB per card: weights (modeled + what the model leaves out), the fixed overheads, and what is left for KV.
    Constants come from the glm boot and the first slice boot; the KV token figure is the conservative one."""
    patched = o["shared_expert_tp"]
    modeled = resident_bytes(rank_modules(slice_config(o)[0], 1, 1, shared_tp=PROD_TP if patched else 1)) / 2 ** 30
    weights = modeled + UNMODELLED_GIB
    budget = o["gpu_memory_utilization"] * CARD_GIB
    kv_gib = budget - weights - NONTORCH_GIB - ACTIVATION_GIB - GRAPH_GIB
    per_request = o["prompt_tokens"] + max_tokens_for(o) + 1
    kv_tokens = max(kv_gib, 0.0) * KV_TOKENS_PER_GIB
    return {"weights_gib_modeled": round(modeled, 2), "weights_gib_logged_est": round(weights, 2),
            "budget_gib": round(budget, 2), "kv_gib_est": round(kv_gib, 2), "kv_tokens_est": int(kv_tokens),
            "tokens_needed": {str(c): c * per_request for c in o["conc"]},
            "fits": {str(c): c * per_request <= 0.9 * kv_tokens for c in o["conc"]}}


def _flatten(d, prefix=""):
    for key, val in d.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            yield from _flatten(val, path + ".")
        else:
            yield path, val


def config_diff(real: dict, sl: dict) -> dict:
    a, b = dict(_flatten(real)), dict(_flatten(sl))
    return {path: [a.get(path), b.get(path)] for path in sorted(set(a) | set(b)) if a.get(path) != b.get(path)}


def tokenizer_files(vocab: int, max_len: int = 1048576) -> dict:
    """A WordLevel tokenizer of `vocab` distinct tokens t0..t{vocab-1}. Engram's hash state loads a
    tokenizer from the model dir even with skip_tokenizer_init, and refuses a compressed-vocab size
    that differs from the config; with distinct tokens the compressed size is exactly `vocab`."""
    tok = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
           "pre_tokenizer": None, "post_processor": None, "decoder": None,
           "model": {"type": "WordLevel", "vocab": {f"t{i}": i for i in range(vocab)}, "unk_token": "t0"}}
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": max_len, "unk_token": "t0",
           "bos_token": "t0", "eos_token": "t1", "pad_token": "t2", "clean_up_tokenization_spaces": False}
    return {"tokenizer.json": json.dumps(tok), "tokenizer_config.json": json.dumps(cfg, indent=1)}


def write_model_dir(base: str | None, cfg: dict) -> tuple:
    """Create the temp model dir; returns (path, {file: {bytes, sha256}})."""
    path = Path(tempfile.mkdtemp(prefix="dsv41-slice-", dir=base))
    files = {"config.json": json.dumps(cfg, indent=2) + "\n",
             **tokenizer_files(cfg["text_config"]["vocab_size"])}
    info = {}
    for name, text in files.items():
        data = text.encode()
        (path / name).write_bytes(data)
        info[name] = {"bytes": len(data), "sha256": _sha256(data)}
    return str(path), info


# --------------------------------------------------------------------------------------------
# Per-rank weight inventory (read from the vLLM constructors at VLLM_COMMIT) and byte model
# --------------------------------------------------------------------------------------------
BPE = {"mxfp8": 1 + 1 / 32, "bf16": 2.0, "fp32": 4.0, "nvfp4": 0.5 + 1 / 16, "mxfp4": 0.5 + 1 / 32}
SAMPLER_PASSES = 4  # assumed fp32 passes over [tokens, vocab] logits (temperature, softmax/gumbel, argmax, reject)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def touched_experts(m_tokens: int, top_k: int, n_global: int, ep: int) -> float:
    """Expected distinct local experts hit by m tokens routed uniformly at random (balanced)."""
    local = n_global / ep
    return local * (1.0 - (1.0 - top_k / n_global) ** m_tokens)


def rank_modules(cfg: dict, tp: int, ep: int, shared_tp: int | None = None) -> dict:
    """What ONE rank holds when `cfg` runs at (tp, ep). name -> {layers, elems, bpe, kind, streams}.
    kind: dense (streamed every step), expert (touched subset streamed), gather (resident only).
    shared_tp: TP degree the shared expert is sharded with (the slice emulates 8 by patching)."""
    t = cfg["text_config"]
    h, hd, heads, ql = t["hidden_size"], t["head_dim"], t["num_attention_heads"], t["q_lora_rank"]
    og, ol, mi = t["o_groups"], t["o_lora_rank"], t["moe_intermediate_size"]
    ne, ns = t["n_routed_experts"], t["n_shared_experts"] or 0
    layers, draft = t["num_hidden_layers"], t["num_nextn_predict_layers"]
    hc = t["hc_mult"]
    mix, hc_dim = (2 + hc) * hc, hc * h
    kv, ix, ratios = t["kv_source_layer_ids"], t["index_source_layer_ids"], t["compress_ratios"]
    vocab_local = _pad64(t["vocab_size"]) // tp
    # DeepseekV4MLP: block-aligned (32) per-rank width = cdiv(blocks, TP) * 32
    shared_local = _cdiv((mi * ns) // 32, shared_tp or tp) * 32
    cols = (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"]
    out: dict = {}

    def add(name, n, elems, fmt, kind="dense", streams=1):
        out[name] = {"layers": n, "elems": elems, "bpe": BPE[fmt], "kind": kind, "streams": streams}

    add("attn.fused_wqa_wkv", layers + draft, h * (ql + hd), "mxfp8")
    add("attn.wq_b", layers + draft, ql * (heads * hd // tp), "mxfp8")
    add("attn.wo_a", layers + draft, (heads * hd // og) * (og * ol // tp), "bf16")  # dequantized at load
    add("attn.wo_b", layers + draft, (og * ol // tp) * h, "mxfp8")
    add("hc.fn", layers + draft, 2 * mix * hc_dim, "fp32")
    add("moe.gate", layers, h * ne, "bf16")
    add("moe.shared", layers + draft, 3 * h * shared_local, "mxfp8")
    add("moe.experts", layers, (ne // ep) * 3 * h * mi, "nvfp4", "expert")
    add("indexer.wq_b", len(ix), ql * t["index_n_heads"] * t["index_head_dim"], "mxfp8")
    add("indexer.weights_proj", len(ix), h * t["index_n_heads"], "bf16")
    add("indexer.wk", len(set(kv) & set(ix)), hd * t["index_head_dim"], "bf16")
    add("compressor", 1, sum(h * hd * (2 if ratios[i] > 1 else 1) for i in kv), "bf16")
    add("embed", 1, vocab_local * h, "bf16", "gather")
    add("lm_head", 1, vocab_local * h, "bf16", streams=2)  # target logits + draft logits
    add("engram.wkv", len(t["engram_layer_ids"]), cols * t["engram_head_dim"] * (h * (hc + 1) // tp), "mxfp8")
    dne = t["dspark_n_routed_experts"]
    add("draft.gate", draft, h * dne, "bf16")
    add("draft.experts", draft, (dne // ep) * 3 * h * mi, "mxfp4", "expert")
    add("draft.main_proj", 1, len(t["dspark_target_layer_ids"]) * h * h, "mxfp8")
    add("draft.markov_w1", 1, t["vocab_size"] * t["dspark_markov_rank"], "bf16", "gather")
    add("draft.markov_w2", 1, t["vocab_size"] * t["dspark_markov_rank"], "bf16")  # streams = k, set per step
    return out


def resident_bytes(mods: dict) -> float:
    return sum(m["layers"] * m["elems"] * m["bpe"] for m in mods.values())


def rank_diff(real: dict, sl: dict) -> dict:
    """Modules whose per-rank bytes differ between the prod rank and the slice: name -> (real, slice)."""
    out = {}
    for name in real:
        a = real[name]["layers"] * real[name]["elems"] * real[name]["bpe"]
        b = sl[name]["layers"] * sl[name]["elems"] * sl[name]["bpe"]
        if abs(a - b) > 1e-6:
            out[name] = (a, b)
    return out


def step_bytes(cfg: dict, tp: int, ep: int, c: int, k: int, shared_tp: int | None = None,
               active_experts: int | None = None) -> dict:
    """Bytes one rank streams from HBM in one decode step (weights only): the roofline view.
    active_experts: routed experts the router can reach (slice: the unmasked ones; default: all of them)."""
    t = cfg["text_config"]
    mods = rank_modules(cfg, tp, ep, shared_tp)
    m_tok, m_draft = c * (k + 1), c * k
    out = {"dense": {}, "experts": 0.0, "draft_experts": 0.0}
    for name, m in mods.items():
        if m["kind"] != "dense" or (k == 0 and name.startswith("draft.")):
            continue
        streams = k if name == "draft.markov_w2" else (1 if k == 0 and name == "lm_head" else m["streams"])
        out["dense"][name] = m["layers"] * m["elems"] * m["bpe"] * streams
    ne = active_experts or t["n_routed_experts"]
    per_expert = 3 * t["hidden_size"] * t["moe_intermediate_size"]
    out["experts"] = (t["num_hidden_layers"] * touched_experts(m_tok, t["num_experts_per_tok"], ne, ep)
                      * per_expert * BPE["nvfp4"])
    if k:
        out["draft_experts"] = (t["num_nextn_predict_layers"]
                                * touched_experts(m_draft, t["dspark_num_experts_per_tok"],
                                                  t["dspark_n_routed_experts"], ep)
                                * per_expert * BPE["mxfp4"])
    out["total"] = sum(out["dense"].values()) + out["experts"] + out["draft_experts"]
    return out


def _interp(points, x: float) -> float:
    if x <= points[0][0]:
        return points[0][1] * x / points[0][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y1 + (y1 - y0) * (x - x1) / (x1 - x0)


def _ms(nbytes: float, bw_gbs: float = ASSUMED_BW_GBS) -> float:
    return nbytes / (bw_gbs * 1e9) * 1e3


def bias_terms(o: dict, c: int, bw_gbs: float = ASSUMED_BW_GBS, active: int | None = None) -> dict:
    """(slice - prod) contribution in ms/step, per effect, for c streams. Positive = the slice is slower.
    Only effects that follow from bytes (or a cited table) are quantified; see non_equivalences().
    active: experts the router actually reaches (default o["active_experts"]; the run passes slice_experts
    when the routing mask could not be applied)."""
    k = effective_k(o, c)  # the draft tokens verified per stream at this batch size (the schedule's, if any)
    real = real_config()
    sl, _ = slice_config(o)
    patched = o["shared_expert_tp"]
    r = step_bytes(real, PROD_TP, PROD_EP, c, k)
    s = step_bytes(sl, 1, 1, c, k, shared_tp=PROD_TP if patched else 1,
                   active_experts=active if active is not None else o["active_experts"])
    dense = lambda d, *names: sum(d["dense"].get(n, 0.0) for n in names)  # noqa: E731
    vocab_real, vocab_slice = real["text_config"]["vocab_size"], sl["text_config"]["vocab_size"]
    tokens = c * (k + 1) + c * k
    terms = {
        "collectives": -_interp(COLLECTIVE_MS_BY_C, c),
        "moe_pairs": _ms(s["experts"] - r["experts"], bw_gbs),
        "draft_moe_pairs": _ms(s["draft_experts"] - r["draft_experts"], bw_gbs),
        "shared_expert": _ms(dense(s, "moe.shared") - dense(r, "moe.shared"), bw_gbs),
        "gate_width": _ms(dense(s, "moe.gate", "draft.gate") - dense(r, "moe.gate", "draft.gate"), bw_gbs),
        "markov_head": _ms(dense(s, "draft.markov_w2") - dense(r, "draft.markov_w2"), bw_gbs),
        "sampler_vocab": _ms(tokens * (vocab_slice - vocab_real) * 4 * SAMPLER_PASSES, bw_gbs),
        "vocab_padding": _ms(dense(s, "lm_head") - dense(r, "lm_head"), bw_gbs),
    }
    terms["quantified_total"] = sum(terms.values())
    return {name: round(v, 4) for name, v in terms.items()}


def non_equivalences(o: dict) -> list:
    """Every known place where the slice is not per-rank-equivalent; ms are (slice - prod) per step."""
    terms = {c: bias_terms(o, c) for c in o["conc"]}

    def per_c(*names):
        return {str(c): round(sum(terms[c][n] for n in names), 3) for c in o["conc"]}

    none = {str(c): None for c in o["conc"]}
    shared_on = o["shared_expert_tp"]
    st = slice_config(o)[0]["text_config"]
    layers = st["num_hidden_layers"]
    return [
        {"id": "collectives", "basis": "estimate (docs/dsv41-comms.md section 3 host all-reduce table; not measured on this model)",
         "prod": "TP8: 2 all-reduces per layer x 43 + embedding + vocab-logits gather + 2 Engram head gathers (~88/step)",
         "slice": "none (TP=1)", "ms": per_c("collectives"),
         "note": "the slice step is SHORTER by this much; add it back to compare with the live pool"},
        {"id": "moe_pairs", "basis": "computed (touched experts x 19.9 MB at %.0f GB/s)" % ASSUMED_BW_GBS,
         "prod": "6 of 384 experts per token, 48 local: 0.75 pairs/token, E[distinct local experts]=48(1-(63/64)^M)",
         "slice": "top-%d of %d reachable experts: %d pairs/token, E=%d(1-(1-%d/%d)^M)"
                  % (o["moe_topk"], o["active_experts"], o["moe_topk"], o["active_experts"], o["moe_topk"], o["active_experts"]),
         "ms": per_c("moe_pairs"),
         "note": "balanced random routing assumed on both sides; real routers are skewed (fewer distinct experts). "
                 + ("The gate-bias mask keeps the other %d resident experts untouched, so this is the %d-expert bias."
                    % (o["slice_experts"] - o["active_experts"], o["active_experts"])
                    if o["active_experts"] < o["slice_experts"] else
                    "No mask (active_experts = slice_experts): every resident expert is reachable.")},
        {"id": "resident_experts", "basis": "computed (memory)",
         "prod": "%d experts resident per layer" % PROD_RANK_EXPERTS,
         "slice": "%d resident (the router kernel has no %d-expert instance)" % (o["slice_experts"], PROD_RANK_EXPERTS),
         "ms": none, "note": "%+d experts x %d layers x 19.9 MB = %+.1f GiB of weights, so the KV pool is that much smaller "
                             "(plan.memory_estimate); no time effect while the extra experts are masked"
                             % (o["slice_experts"] - PROD_RANK_EXPERTS, layers,
                                (o["slice_experts"] - PROD_RANK_EXPERTS) * layers * 3 * st["hidden_size"]
                                * st["moe_intermediate_size"] * BPE["nvfp4"] / 2 ** 30)},
        {"id": "router_kernel", "basis": "unquantified (a few us per layer)",
         "prod": "target layers: dsv4_topk (Triton, 384 experts, top-6; router/dsv4_topk.py:26); draft layers: topkGatingSoftplusSqrt<128>",
         "slice": "topkGatingSoftplusSqrt<%d> (CUDA) on the target layers, <%d> on the draft layers: not the kernel prod's target "
                  "layers run" % (st["n_routed_experts"], st["dspark_n_routed_experts"]),
         "ms": none, "note": "<= ~0.2 ms per step over 43 layers; prod also filters non-local pairs inside the MoE kernel"},
        {"id": "draft_moe_pairs", "basis": "computed",
         "prod": "DSpark MoE: top-3 of 128, 16 local", "slice": "top-1 of 16", "ms": per_c("draft_moe_pairs"), "note": ""},
        {"id": "shared_expert", "basis": "computed",
         "prod": "TP8-sharded: 288 wide (3.0 + 1.5 MB per layer)",
         "slice": "patched to 288 (exact)" if shared_on else "2304 wide, unsharded (8x the bytes)",
         "ms": per_c("shared_expert"), "note": "opts.shared_expert_tp"},
        {"id": "gate_and_router", "basis": "gate computed; router top-k kernel width and EP filtering unquantified (<~0.3 ms, prod-only)",
         "prod": "gate [5120 x 384] (and [5120 x 128] in the draft); top-6 over 384; non-local pairs filtered in the kernel",
         "slice": "[%d x %d] / [%d x %d]; top-%d over %d resident (%d reachable)"
                  % (st["hidden_size"], st["n_routed_experts"], st["hidden_size"], st["dspark_n_routed_experts"],
                     o["moe_topk"], st["n_routed_experts"], o["active_experts"]),
         "ms": per_c("gate_width"), "note": ""},
        {"id": "vocab", "basis": "markov head computed; sampler assumes %d fp32 passes over [tokens, vocab]" % SAMPLER_PASSES,
         "prod": "lm_head/embedding 16160 per rank, logits all-gathered to 129280 for the sampler and rejection kernels; "
                 "Markov head (replicated) 129280 x 256",
         "slice": "vocab 16160 everywhere (lm_head exact per rank, sampler/rejection/Markov 8x narrower)",
         "ms": per_c("markov_head", "sampler_vocab", "vocab_padding"), "note": "slice step is shorter"},
        {"id": "engram", "basis": "bound: 2 layers x M x 3 rows x 136 B over PCIe is < 0.2 MB per step",
         "prod": "24 hash columns hashed per rank, 3 looked up from a 6.08 GiB pinned table per layer, head all-gather",
         "slice": "3 hash columns hashed and looked up from 3 x engram_rows rows (default 1M rows/column, 0.4 GiB per layer)",
         "ms": none, "note": "<= ~0.1 ms either way; lookups from a smaller table are more cache-friendly; host NUMA differs from worker-06"},
        {"id": "vision_tower", "basis": "memory only", "prod": "ViT + aligner resident (data-parallel mm encoder)",
         "slice": "not built: limit_mm_per_prompt image=0 (no tokenizer files exist for the MM processor)",
         "ms": none, "note": "decode never runs the encoder; ~0.97 GiB less weight than a prod rank. AGENTS rule 1.5 is about pools"},
        {"id": "inprocess_engine", "basis": "unquantified", "prod": "scheduler and worker are separate processes (async scheduling overlaps them)",
         "slice": "one process, VLLM_ENABLE_V1_MULTIPROCESSING=0 (needed for the profiler and the shared-expert patch)",
         "ms": none, "note": "adds at most the scheduler's CPU time per step when the GPU would otherwise wait"},
        {"id": "dummy_weights", "basis": "unquantified", "prod": "trained weights, skewed routing, correlated tokens",
         "slice": "random weights/tokens: ~uniform routing, constant E8M0 scales, tiny activations (power/clock data dependence)",
         "ms": none, "note": "expect a few percent either way; use moe_routing=normal_routing to bracket skew"},
        {"id": "card", "basis": "unquantified", "prod": "RTX PRO 6000 Server Edition (600 W)",
         "slice": "lab card may be the Max-Q (300 W): same GDDR7 bandwidth, lower sustained clocks",
         "ms": none, "note": "bandwidth-bound decode unaffected; compute-bound parts (c=32 GEMMs, attention) can be slower; "
                             "the JSON carries power limit and sampled SM clock"},
        {"id": "context", "basis": "parameter", "prod": "live traffic: mean prompt up to ~75k tokens",
         "slice": "opts.prompt_tokens (default 2048)", "ms": none,
         "note": "SPEED.md measured flat decode speed 0 -> 500k at c=1; sweep prompt_tokens to check at c>1"},
        {"id": "adaptive_verification", "basis": "unquantified", "prod": "trained confidence head drives the verification budget",
         "slice": "random confidence head (~0.5 everywhere): trimming does not match prod", "ms": none,
         "note": "only the cost side of enable_adaptive_verification is meaningful"},
        {"id": "acceptance_process", "basis": "by construction", "prod": "content-dependent, correlated acceptance",
         "slice": "independent Bernoulli per position with the measured cumulative rates (synthetic rejection)",
         "ms": none, "note": "marginals match; step time depends only on the verified token count"},
    ]


def expected_slice_ms(o: dict, c: int):
    """What the slice should read at c streams if this model of the rank is right: the live step plus the
    (slice - prod) bias. Only defined for the live configuration (k=5, no schedule, no adaptive trimming)."""
    live = LIVE_STEP_MS.get(c)
    if live is None or o["k"] != 5 or o["num_speculative_tokens_per_batch_size"] or o["enable_adaptive_verification"]:
        return None
    return round(live + bias_terms(o, c)["quantified_total"], 3)


def prod_equivalent_ms(o: dict, c: int, slice_step_ms: float) -> float:
    """slice step minus the quantified (slice - prod) bias: the number to compare with LIVE_STEP_MS."""
    return round(slice_step_ms - bias_terms(o, c)["quantified_total"], 3)


# --------------------------------------------------------------------------------------------
# Kernel contracts: the hard allowlists and integer tables that the shrunken dimensions must satisfy.
# Each entry cites the pinned vLLM source; parse_opts refuses a violated one (exit 2) instead of
# spending a 10-minute boot to find it. The first silicon run found one by running into it: the
# router kernel has no 48-expert instance (csrc/moe/topk_softplus_sqrt_kernels.cu:841).
# --------------------------------------------------------------------------------------------
def mxfp8_linears(cfg: dict, tp: int, shared_tp: int | None = None) -> dict:
    """(N, K) of every MXFP8 linear one rank runs through FlashInferCutlassMxfp8LinearKernel. wo_a is a BMM and
    takes the emulation kernel on SM12x (kernels/linear/__init__.py:961, mxfp8/deep_gemm.py:35): no shape rule."""
    t = cfg["text_config"]
    h, hd, heads, ql = t["hidden_size"], t["head_dim"], t["num_attention_heads"], t["q_lora_rank"]
    shared = _cdiv((t["moe_intermediate_size"] * (t["n_shared_experts"] or 0)) // 32, shared_tp or tp) * 32
    cols = (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"]
    return {
        "attn.fused_wqa_wkv": (ql + hd, h),
        "attn.wq_b": (heads * hd // tp, ql),
        "attn.wo_b": (h, t["o_groups"] * t["o_lora_rank"] // tp),
        "shared_expert.gate_up": (2 * shared, h),
        "shared_expert.down": (h, shared),
        "indexer.wq_b": (t["index_n_heads"] * t["index_head_dim"], ql),
        "engram.wkv": (h * (t["hc_mult"] + 1) // tp, cols * t["engram_head_dim"]),
        "draft.main_proj": (h, len(t["dspark_target_layer_ids"]) * h),
    }


def kernel_contracts(o: dict) -> list:
    cfg, _ = slice_config(o)
    t = cfg["text_config"]
    simulated = o["moe_routing"] != "model"
    out: list = []

    def add(name, ok, detail, source):
        out.append({"id": name, "ok": bool(ok), "detail": detail, "source": source})

    router_src = "csrc/moe/topk_softplus_sqrt_kernels.cu:786-841"
    add("router_experts_target", simulated or t["n_routed_experts"] in GATE_KERNEL_EXPERTS,
        f"{t['n_routed_experts']} experts; kernel instances exist for {list(GATE_KERNEL_EXPERTS)}"
        + ("; bypassed by the routing simulator (router/router_factory.py:120)" if simulated else ""), router_src)
    add("router_experts_draft", simulated or t["dspark_n_routed_experts"] in GATE_KERNEL_EXPERTS,
        f"{t['dspark_n_routed_experts']} draft experts", router_src)
    shapes = mxfp8_linears(cfg, 1, PROD_TP if o["shared_expert_tp"] else 1)
    bad = {n: (N, K) for n, (N, K) in shapes.items() if not (K >= 128 and K % 32 == 0 and N >= 128 and N % 32 == 0)}
    add("mxfp8_linear_shapes", not bad,
        "all (N,K) satisfy K>=128, K%32==0, N>=128, N%32==0 (SM12x): " + ", ".join(f"{n}={nk}" for n, nk in shapes.items())
        if not bad else f"violated by {bad}", "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:27-70")
    inter = t["moe_intermediate_size"]
    aligned = t["hidden_size"] % 128 == 0 and inter % 128 == 0
    add("flashinfer_cutlass_moe_alignment", aligned,
        f"hidden_size={t['hidden_size']}, moe_intermediate_size={inter}: " + ("both multiples of 128" if aligned else
                                                                             "must both be multiples of 128")
        + " (the strictest of the binding's alignment checks); no expert-count or top-k allowlist exists: the fused "
        "expert-map prologue returns false for top-k outside {1,2,4,6,8}, more than 256 tokens or more than 510 experts "
        "and the three-step path takes over; EP=1 here",
        "flashinfer 0.7.2 csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu:422-438; "
        "cutlass_fused_moe_kernels.cuh:552-648,4148-4191,4319-4336")
    heads = t["num_attention_heads"]
    add("sparse_mla_heads", heads <= 128,
        f"{heads} local heads -> padded {next((h for h in (8, 16, 32, 64, 128) if heads <= h), None)} "
        "(natively supported h_q: 8, 16, 32, 64, 128; the prod rank has 8 too)",
        "vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:75-86")
    window, k = t["sliding_window"], o["k"]
    draft_topk = -(-(window + k) // 64) * 64 if k else window
    add("sm120_sparse_mla_dispatch", draft_topk == 192 or k == 0,
        f"vLLM refuses the backend unless FlashInfer's DSV4 decode envelope contains ({heads}, {draft_topk}); in 0.7.2 that is a "
        "predicate (1 <= heads <= 128 and top_k >= 1: top-k is a runtime argument, there is no table of instances) and the C++ "
        "dispatcher stays authoritative. k > 0: ceil((window+k)/64)*64 = 192 for every k <= 64, the (8, 192) pair prod's k=5 rank "
        f"runs; k = 0: ({heads}, {window}), a pair prod never ran",
        "vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:88-100,633; vllm/utils/flashinfer.py:473-486; "
        "v1/attention/backends/mla/compressor_utils.py:19-28; flashinfer 0.7.2 flashinfer/mla/_sparse_mla_sm120/_policy.py:98-182")
    add("noise_token_in_vocab", 0 <= t["dspark_noise_token_id"] < t["vocab_size"],
        f"dspark_noise_token_id={t['dspark_noise_token_id']} feeds the draft embedding as the parallel-drafting mask token; "
        f"vocab_size={t['vocab_size']} (TP=1 embedding has no out-of-shard masking)",
        "vllm/v1/worker/gpu/spec_decode/utils.py:92-114")
    primes = engram_rows(t["engram_layer_ids"], t["engram_max_ngram_size"], t["engram_n_heads"], t["engram_vocab_size"])
    add("engram_rows_cover_primes", all(n >= r for n, r in zip(t["engram_num_embeddings"], primes)),
        f"engram_num_embeddings={t['engram_num_embeddings']} vs sum of bucket ranges {primes}",
        "vllm/models/deepseek_v41/common/engram.py:965")
    add("engram_tokenizer_vocab", t["engram_compressed_vocab_size"] == t["vocab_size"],
        "the model dir's tokenizer has vocab_size distinct tokens, so its compressed vocab is vocab_size",
        "vllm/models/deepseek_v41/common/engram.py:449-461")
    add("hash_tables_absent", True,
        "V4.1 builds every MoE with num_hash_layers=0: no tid2eid table, so no integer expert-id table to violate",
        "vllm/models/deepseek_v41/nvidia/model.py:150; vllm/models/deepseek_v4/nvidia/model.py:284-300")
    add("routing_within_experts", o["moe_topk"] <= o["active_experts"] <= t["n_routed_experts"],
        f"top-{o['moe_topk']} of {o['active_experts']} reachable of {t['n_routed_experts']} resident experts",
        "csrc/moe/topk_softplus_sqrt_kernels.cu:510-516 (gate bias added before the top-k)")
    padded = _pad64(t["vocab_size"])
    add("vocab_stride", True,
        f"lm_head is {padded} rows for vocab {t['vocab_size']}; logits[..., :vocab] is a row-strided view"
        if padded != t["vocab_size"] else "no vocab padding",
        "vllm/model_executor/layers/logits_processor.py:198; v1/worker/gpu/sample/gumbel.py:41; "
        "v1/worker/gpu/spec_decode/rejection_sampler_utils.py:1061 (stride(-1)==1 only)")
    return out


ENGINE_HINTS = (
    (r"Unsupported expert number: (\d+)",
     "the router kernel has no instance for that expert count; slice_experts must be one of "
     + "/".join(str(n) for n in SLICE_EXPERTS) + " (csrc/moe/topk_softplus_sqrt_kernels.cu:786-841)"),
    (r"larger than the available KV cache memory|To serve at least one request with the model's max seq len",
     "the weights leave too little KV memory for max_model_len: the tool never shrinks the context silently; lower "
     "max_model_len, slice_experts, or raise gpu_memory_utilization"),
    (r"Free memory on device|less than desired GPU memory utilization",
     "another process holds memory on this GPU: lower gpu_memory_utilization or free the card"),
    (r"out of memory|OutOfMemoryError",
     "GPU or host memory exhausted: check engram_rows (pinned host RAM), slice_experts and gpu_memory_utilization"),
    (r"No available memory for the cache blocks|Insufficient GPU memory|Available KV cache memory.*negative",
     "no memory left for KV after the weights: lower slice_experts or raise gpu_memory_utilization"),
)


def engine_hint(text: str) -> str | None:
    for rx, hint in ENGINE_HINTS:
        if re.search(rx, text):
            return hint
    return None


# --------------------------------------------------------------------------------------------
# Plan: env + engine kwargs (everything the engine will be given), buildable without vLLM
# --------------------------------------------------------------------------------------------
def env_plan(o: dict) -> dict:
    """The complete env the run needs at interpreter start (sitecustomize reads SUFFIX_* there)."""
    env = dict(DEFAULT_ENV)
    env.update(o["env"])
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine: profiler + shared-expert patch
    env["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"  # vLLM logs to stdout by default; stdout is the JSON
    env["VLLM_LOGGING_LEVEL"] = env.get("VLLM_LOGGING_LEVEL", o["log_level"])
    if o["moe_routing"] != "model":
        env["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = o["moe_routing"]
    return env


def speculative_config(o: dict):
    """Only keys that differ from vLLM's defaults, like the prod pool's `non-default args`."""
    k = o["k"]
    if k == 0:
        return None
    spec = {"method": "dspark", "num_speculative_tokens": k}
    if o["draft_sample_method"] != "greedy":
        spec["draft_sample_method"] = o["draft_sample_method"]
    if o["rejection_sample_method"] != "standard":
        spec["rejection_sample_method"] = o["rejection_sample_method"]
    rates = synthetic_rates(o)
    if rates is not None:
        spec["synthetic_acceptance_rates"] = rates
    if o["enable_adaptive_verification"]:
        spec["enable_adaptive_verification"] = True
    if o["num_speculative_tokens_per_batch_size"]:
        spec["num_speculative_tokens_per_batch_size"] = [tuple(x) for x in o["num_speculative_tokens_per_batch_size"]]
    return spec


def engine_kwargs(o: dict, model_dir: str) -> dict:
    """LLM(**kwargs): the prod serving flags on a TP=1 slice."""
    kw = {
        "model": model_dir,
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "tensor_parallel_size": 1,
        "seed": o["seed"],
        "max_model_len": o["max_model_len"],
        "max_num_seqs": o["max_num_seqs"],
        "max_num_batched_tokens": o["max_num_batched_tokens"],
        "gpu_memory_utilization": o["gpu_memory_utilization"],
        "block_size": o["block_size"],
        "enable_prefix_caching": o["enable_prefix_caching"],
        "limit_mm_per_prompt": {"image": 0},  # text-only: no tokenizer files for the MM processor (see doc)
        "disable_log_stats": False,  # LLM.get_metrics() needs it (spec-decode counters)
    }
    if o["kv_cache_dtype"] != "auto":
        kw["kv_cache_dtype"] = o["kv_cache_dtype"]
    for opt in ("moe_backend", "linear_backend", "sparse_indexer_topk_backend"):
        if o[opt] != "auto":
            kw[opt] = o[opt]
    for opt in ("enable_flashinfer_autotune", "async_scheduling"):
        if o[opt] is not None:
            kw[opt] = o[opt]
    if o["cudagraph_mode"]:
        kw["compilation_config"] = {"cudagraph_mode": o["cudagraph_mode"]}
    spec = speculative_config(o)
    if spec:
        kw["speculative_config"] = spec
    return kw


def plan_warnings(o: dict, mem: dict) -> list:
    """Things worth knowing before a 10-minute boot (none of them is a refusal)."""
    out = []
    if mem["kv_gib_est"] < KV_WARN_GIB:
        out.append(f"estimated KV pool {mem['kv_gib_est']} GiB (~{mem['kv_tokens_est']:,} tokens at the conservative "
                   f"{KV_TOKENS_PER_GIB:,} tokens/GiB): long prompts at high concurrency may not fit")
    for c, fits in mem["fits"].items():
        if not fits:
            out.append(f"c={c}: needs ~{mem['tokens_needed'][c]:,} KV tokens, the estimate is ~{mem['kv_tokens_est']:,}: "
                       "that concurrency is skipped with a warning if the booted pool is smaller")
    if o["active_experts"] < o["slice_experts"]:
        out.append(f"routing mask: {o['slice_experts'] - o['active_experts']} of {o['slice_experts']} experts per layer get "
                   "gate bias -1e4 after boot; if that cannot be applied the run keeps going, records "
                   "routing_mask.applied=false and charges the unmasked MoE bias")
    return out


def build_plan(o: dict, model_dir: str | None = None, files: dict | None = None) -> dict:
    cfg, rules = slice_config(o)
    real, sl = real_config(), cfg
    patched = o["shared_expert_tp"]
    mods_real = rank_modules(real, PROD_TP, PROD_EP)
    mods_slice = rank_modules(sl, 1, 1, shared_tp=PROD_TP if patched else 1)
    mem = memory_plan(o)
    return {
        "engine_kwargs": engine_kwargs(o, model_dir or "<model_dir>"),
        "env": env_plan(o),
        "shared_expert_patch": patched,
        "max_tokens_per_request": max_tokens_for(o),
        "rules": rules,
        "config_diff": config_diff(real, sl),
        "model_dir_files": files or {},
        "weights_gib_per_rank": {
            "prod_rank_modeled": round(resident_bytes(mods_real) / 2 ** 30, 3),
            "slice_modeled": round(resident_bytes(mods_slice) / 2 ** 30, 3),
            "prod_rank_logged": 40.66,
            "note": "modeled = linears/experts/gates/mHC/embeddings; logged = vLLM 'Model loading took' on glm "
                    "(adds the vision tower ~0.97, rope caches, buffers ~0.99). The slice has no vision tower: its logged "
                    "value should be slice_modeled + 0.99 (the 48-expert slice logged 39.45 for 38.458 modeled on silicon); "
                    "memory_estimate.weights_gib_logged_est is that number",
        },
        "rank_diff_mb": {n: [round(a / 1e6, 2), round(b / 1e6, 2)] for n, (a, b) in rank_diff(mods_real, mods_slice).items()},
        "step_stream_gb": {str(c): {"prod_rank": round(step_bytes(real, PROD_TP, PROD_EP, c, effective_k(o, c))["total"] / 1e9, 2),
                                    "slice": round(step_bytes(sl, 1, 1, c, effective_k(o, c), PROD_TP if patched else 1,
                                                              active_experts=o["active_experts"])["total"] / 1e9, 2)}
                           for c in o["conc"]},
        "engram_host_gib": round(engram_host_gib(o, sl["text_config"]), 3),
        "k_per_conc": {str(c): effective_k(o, c) for c in o["conc"]},
        "memory_estimate": mem,
        "contracts": kernel_contracts(o),
        "routing": {"moe_routing": o["moe_routing"], "slice_experts": o["slice_experts"],
                    "active_experts": o["active_experts"], "moe_topk": o["moe_topk"],
                    "mask": ("gate bias -1e4 on experts >= active_experts, applied with LLM.apply_model after the engine "
                             "is up (CUDA graphs read the same storage)")
                    if o["active_experts"] < o["slice_experts"] else None},
        "warnings": plan_warnings(o, mem),
        "non_equivalences": non_equivalences(o),
        "bias_terms_ms": {str(c): bias_terms(o, c) for c in o["conc"]},
        "live_reference_ms": {str(c): LIVE_STEP_MS[c] for c in o["conc"] if c in LIVE_STEP_MS},
        "expected_slice_ms": {str(c): expected_slice_ms(o, c) for c in o["conc"]},
    }


# --------------------------------------------------------------------------------------------
# Measurement helpers (pure; unit-tested on CPU)
# --------------------------------------------------------------------------------------------
def pct(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def step_stats(gaps_s: list, tokens: list, wall_s: float, c: int) -> dict:
    """gaps_s: time between consecutive step outputs; tokens: tokens emitted per step (all streams)."""
    n = len(gaps_s)
    ms = sorted(g * 1e3 for g in gaps_s)
    total_tokens = sum(tokens)
    return {
        "steps": n,
        "step_ms": {"mean": wall_s / n * 1e3 if n else None, "p50": pct(ms, 0.5), "p90": pct(ms, 0.9),
                    "p99": pct(ms, 0.99), "min": ms[0] if ms else None, "max": ms[-1] if ms else None},
        "wall_s": wall_s,
        "tokens_emitted": total_tokens,
        "tokens_per_stream_per_step": total_tokens / (c * n) if n else None,
        "tok_s_per_stream": total_tokens / c / wall_s if wall_s > 0 else None,
        "tok_s_total": total_tokens / wall_s if wall_s > 0 else None,
    }


def spec_delta(before: dict | None, after: dict | None, c: int) -> dict | None:
    """Spec-decode counter deltas over a window -> verified tokens per step and accepted length."""
    if not before or not after:
        return None
    d = {key: after[key] - before[key] for key in ("drafts", "draft_tokens", "accepted")}
    pos = [a - b for a, b in zip(after["per_pos"], before["per_pos"])]
    drafts = d["drafts"]
    if drafts <= 0:
        return {**d, "per_pos_accepted": pos}
    return {**d, "per_pos_accepted": pos,
            "per_pos_cum_accept": [p / drafts for p in pos],
            "accepted_len": 1 + d["accepted"] / drafts,
            "verified_per_stream": 1 + d["draft_tokens"] / drafts,
            "verified_tokens_per_step": c * (1 + d["draft_tokens"] / drafts),
            "acceptance_rate": d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else None}


# Kernel families: ordered, first match wins, patterns on the lower-cased kernel name. Heuristic:
# unmatched kernels stay in "other" and are listed so the patterns can be refined from a real run.
KERNEL_FAMILIES = (
    ("memcpy", r"^memcpy|^memset|memcpy |memset "),
    ("engram", r"engram|ngram|hash_ids|hash_cache"),
    ("mhc", r"mhc|hc_collapse|hc_head|hc_pre|hc_post|sinkhorn|tilelang|^main_kernel"),
    ("indexer", r"indexer|mqa_logits|top_k_per_row|topk_per_row|paged_mqa|fp8_mqa|lightning|short_context_topk|candidate"),
    ("compressor", r"compress|state_cache|save_partial|kv_state|kv_score"),
    ("draft", r"markov|dspark|draft|confidence"),
    ("sampler", r"gumbel|sample_|rejection|reject_|argmax|softmax|bonus|penalt|multinomial|philox|topk_topp|logits"),
    ("moe", r"moe|expert|grouped|fusedbuildexpertmaps|expandinputrows|doactivation|finalize|computestrides|"
            r"fused_topk|topk_gating|router|swiglu|silu_and_mul|act_and_mul|align_block|m_grouped"),
    ("attention", r"mla|attention|attn|sparse|fmha|flash_|paged|swa|rope|kv_insert|kv_rope|qnorm|"
                  r"reshape_and_cache|concat_and_cache|cache_kernel"),
    ("norm_quant", r"rms|norm|quant|amax|per_token|dequant|mxfp|nvfp4|fp4_|cast"),
    ("dense_gemm", r"gemm|gemv|cutlass|cublas|nvjet|xmma|splitk|sm\d+_|wgmma|mma_|bmm|einsum|matmul"),
    ("elementwise", r"elementwise|vectorized|unrolled|add|mul|copy|fill|cat|index|gather|scatter|where|reduce|"
                    r"clamp|silu|gelu|relu|sigmoid|exp|log|arange|iota"),
)
_FAMILY_RES = tuple((name, re.compile(rx)) for name, rx in KERNEL_FAMILIES)


def classify_kernel(name: str) -> str:
    low = name.lower()
    for family, rx in _FAMILY_RES:
        if rx.search(low):
            return family
    return "other"


def merge_span_us(intervals: list) -> float:
    """Length of the union of [start, end] intervals (kernels on several streams overlap)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if e < s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def aggregate_kernels(events: list, steps: int, wall_ms_per_step: float | None = None,
                      top: int = 40, name_cap: int = 160) -> dict:
    """events: [(kernel name, start_us, end_us)] over `steps` engine steps -> per-step table."""
    by: dict = {}
    for name, s, e in events:
        row = by.setdefault(name, [0.0, 0])
        row[0] += max(0.0, e - s)
        row[1] += 1
    total = sum(r[0] for r in by.values())
    steps = max(steps, 1)
    fam: dict = {}
    rows = []
    for name, (us, calls) in by.items():
        family = classify_kernel(name)
        f = fam.setdefault(family, [0.0, 0])
        f[0] += us
        f[1] += calls
        rows.append({"name": name[:name_cap], "family": family, "calls_per_step": round(calls / steps, 3),
                     "us_per_step": round(us / steps, 2), "share": round(us / total, 5) if total else 0.0})
    rows.sort(key=lambda r: -r["us_per_step"])
    span = merge_span_us([(s, e) for _, s, e in events])
    out = {
        "steps": steps,
        "kernels_per_step": round(len(events) / steps, 1),
        "kernel_us_per_step": round(total / steps, 1),
        "span_us_per_step": round(span / steps, 1),
        "families": sorted(({"family": f, "us_per_step": round(us / steps, 1),
                             "calls_per_step": round(calls / steps, 1),
                             "share": round(us / total, 4) if total else 0.0}
                            for f, (us, calls) in fam.items()), key=lambda r: -r["us_per_step"]),
        "top_kernels": rows[:top],
        "unclassified_top": [r for r in rows if r["family"] == "other"][:15],
    }
    if wall_ms_per_step:
        out["wall_ms_per_step_profiled"] = round(wall_ms_per_step, 3)
        out["gpu_busy_fraction"] = round(span / steps / (wall_ms_per_step * 1e3), 4)
    return out


# --------------------------------------------------------------------------------------------
# Runtime. Everything below needs torch/vLLM/a GPU and was written from the vLLM source only.
# --------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[dsv41-slice {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _clean(x):
    """JSON-safe: NaN/inf -> None, floats to 6 significant digits."""
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


class Recorder:
    """The one result document. emit() prints it exactly once (also from the deadline thread)."""

    def __init__(self, out):
        self.out = out
        self.t0 = time.time()
        self.lock = threading.RLock()
        self.emitted = False
        self.doc: dict = {"schema": RESULT_SCHEMA, "ok": False, "exit": 1, "partial": False, "dry": False,
                          "engine_error": None, "error": None, "warnings": [], "phases": {}, "results": {}}

    def set(self, key, value):
        with self.lock:
            self.doc[key] = value

    def warn(self, msg: str):
        log("WARNING " + msg)
        with self.lock:
            self.doc["warnings"].append(msg)

    def checkpoint(self):
        """Write the document so far to out_path (atomic), so a SIGKILL keeps finished concurrencies."""
        if not getattr(self, "out_path", None):
            return
        with self.lock:
            text = json.dumps(_clean({**self.doc, "partial": True, "exit": None}), default=str)
        tmp = self.out_path + ".tmp"
        with contextlib.suppress(OSError):
            Path(tmp).write_text(text + "\n")
            os.replace(tmp, self.out_path)

    def emit(self, code: int, partial: bool = False):
        with self.lock:
            if self.emitted:
                return
            self.emitted = True
            self.doc.update(exit=code, ok=(code == 0), partial=partial, elapsed_s=round(time.time() - self.t0, 1))
            text = json.dumps(_clean(self.doc), default=str)
        self.out.write(text + "\n")
        self.out.flush()


class Clock:
    def __init__(self, seconds: int):
        self.deadline = time.time() + seconds

    def check(self):
        if time.time() > self.deadline:
            raise SliceError("deadline_s reached", 4)


@contextlib.contextmanager
def phase(rec: Recorder, name: str):
    t = time.time()
    log(f"phase {name}: start")
    try:
        yield
    finally:
        dt = time.time() - t
        with rec.lock:
            rec.doc["phases"][name] = round(dt, 2)
        log(f"phase {name}: {dt:.1f}s")


class Tee:
    """After this, fd 1 and fd 2 are one pipe: library prints, child processes and C-level writes all
    land on the real stderr AND in `lines` (so plugin markers and vLLM backend choices can be checked).
    The result JSON goes to the saved original stdout, never through here."""

    def __init__(self, real_stderr_fd: int, keep: int = 30000):
        self.lines: deque = deque(maxlen=keep)
        r, w = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        self._out = os.fdopen(real_stderr_fd, "w", errors="replace")
        self._token = ""
        self._seen = threading.Event()
        threading.Thread(target=self._pump, args=(r,), daemon=True, name="tee").start()

    def _handle(self, line: str):
        try:
            self._out.write(line)
            self._out.flush()
        except Exception:  # noqa: BLE001 - never kill the run over a closed stderr
            pass
        self.lines.append(line.rstrip("\n")[:2000])
        if self._token and self._token in line:
            self._seen.set()

    def _pump(self, r: int):
        """A writer blocks once the pipe is full, so this must keep draining whatever happens."""
        f = os.fdopen(r, "r", errors="replace")
        while True:
            try:
                for line in f:
                    self._handle(line)
                return
            except Exception:  # noqa: BLE001
                continue

    def sync(self, timeout: float = 10.0) -> None:
        """Block until everything written so far went through the pump (a marker line round-trips)."""
        self._token = f"@@dsv41-slice-sync-{time.time_ns()}@@"
        self._seen.clear()
        print(self._token, file=sys.stderr, flush=True)
        self._seen.wait(timeout)

    def text(self) -> str:
        return "\n".join(self.lines)


def maybe_reexec(env: dict) -> None:
    """The plugin's sitecustomize reads SUFFIX_* (and glibc MALLOC_*) when the interpreter starts, so a
    changed env needs a fresh interpreter. Re-exec once with the full env and a marker."""
    differs = {k: v for k, v in env.items() if os.environ.get(k) != v}
    if not differs or os.environ.get(_REEXEC_MARK) == "1":
        os.environ.update(env)
        return
    new = dict(os.environ)
    new.update(env)
    new[_REEXEC_MARK] = "1"
    log(f"re-exec with env {sorted(differs)}")
    sys.stderr.flush()
    os.execve(sys.executable, sys.orig_argv, new)


class Telemetry:
    """NVML sampler (SM clock, power, temperature, throttle reasons) during the timed windows."""

    def __init__(self):
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            self.h = (pynvml.nvmlDeviceGetHandleByIndex(int(first)) if first.isdigit()
                      else pynvml.nvmlDeviceGetHandleByUUID(first))
        except Exception:  # noqa: BLE001 - telemetry is best effort
            self.h = None
        self._stop = threading.Event()
        self._rows: list = []
        self._t: threading.Thread | None = None

    def info(self) -> dict:
        if not self.h:
            return {}
        nv, out = self.nv, {}
        for key, fn in (("name", lambda: nv.nvmlDeviceGetName(self.h)),
                        ("power_limit_w", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000),
                        ("max_sm_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(self.h, nv.NVML_CLOCK_SM)),
                        ("max_mem_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(self.h, nv.NVML_CLOCK_MEM)),
                        ("driver", lambda: nv.nvmlSystemGetDriverVersion())):
            try:
                v = fn()
                out[key] = v.decode() if isinstance(v, bytes) else v
            except Exception:  # noqa: BLE001
                pass
        return out

    def _loop(self):
        nv = self.nv

        def get(fn, default):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - NVML names differ across versions
                return default

        reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        while not self._stop.is_set():
            self._rows.append((get(lambda: nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM), 0),
                               get(lambda: nv.nvmlDeviceGetPowerUsage(self.h) / 1000, 0.0),
                               get(lambda: nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU), 0),
                               get(lambda: reasons(self.h), 0) if reasons else 0))
            self._stop.wait(0.2)

    def start(self):
        if self.h:
            self._stop.clear()
            self._rows = []
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def stop(self) -> dict | None:
        if not self.h or not self._t:
            return None
        self._stop.set()
        self._t.join(2)
        rows = self._rows
        if not rows:
            return None
        reasons = 0
        for r in rows:
            reasons |= int(r[3])
        return {"samples": len(rows), "sm_mhz_mean": sum(r[0] for r in rows) / len(rows),
                "sm_mhz_min": min(r[0] for r in rows), "power_w_mean": sum(r[1] for r in rows) / len(rows),
                "power_w_max": max(r[1] for r in rows), "temp_c_max": max(r[2] for r in rows),
                "throttle_reasons_mask": reasons}


def collect_versions() -> dict:
    import importlib
    v = {"python": sys.version.split()[0], "argv": sys.orig_argv}
    for mod in ("torch", "vllm", "triton", "flashinfer", "transformers", "tokenizers", "tilelang", "helion"):
        try:
            v[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            v[mod] = f"unavailable: {type(exc).__name__}"
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["cuda_available"] = bool(torch.cuda.is_available())  # does not create a context
    except Exception as exc:  # noqa: BLE001
        v["cuda"], v["cuda_available"] = f"unavailable: {exc}", False
    return v


def gpu_properties() -> dict:
    """Only after the engine exists: reading device properties creates the CUDA context."""
    import torch
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "cc": f"{p.major}.{p.minor}", "mem_gib": round(p.total_memory / 2 ** 30, 2),
            "sms": p.multi_processor_count}


# One vLLM anchor the tool relies on. Same doctrine as the plugin: pin the source, refuse on drift.
_SHARED_ANCHORS = (
    "intermediate_size = config.moe_intermediate_size * config.n_shared_experts",
    "self.shared_experts = DeepseekV4MLP(",
    'prefix=f"{prefix}.shared_experts"',
)


def apply_shared_expert_patch(factor: int = PROD_TP) -> None:
    """At TP=1 the shared expert is 2304 wide; a TP8 rank holds 288 (cdiv(72 blocks, 8) x 32). Wrap
    DeepseekV4MLP so the shared-expert MLPs are built at 1/factor width. In-process only."""
    import importlib
    import inspect
    mod = importlib.import_module("vllm.models.deepseek_v4.nvidia.model")
    src = inspect.getsource(mod.DeepseekV4MoE.__init__)
    missing = [a for a in _SHARED_ANCHORS if a not in src]
    if missing:
        raise SliceError(f"vLLM source drifted from {VLLM_COMMIT[:9]}: DeepseekV4MoE.__init__ lacks {missing}; "
                         "rerun with shared_expert_tp=false to accept the 8x shared-expert over-count", 6)
    base = mod.DeepseekV4MLP

    class ShardedSharedMLP(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            if str(kwargs.get("prefix", "")).endswith(".shared_experts") and "intermediate_size" in kwargs:
                full = kwargs["intermediate_size"]
                if full % (32 * factor):
                    raise SliceError(f"shared expert width {full} is not a multiple of 32 x {factor}", 6)
                kwargs["intermediate_size"] = full // factor
            super().__init__(*args, **kwargs)

    mod.DeepseekV4MLP = ShardedSharedMLP
    log(f"shared expert patched to 1/{factor} width (anchors ok)")


ROUTING_MASK_BIAS = -1.0e4


def _mask_routing(model, active: int, width: int) -> dict:
    """Runs inside the worker (LLM.apply_model). Writes gate bias -1e4 for routed experts >= `active` into every
    [width] e_score_correction_bias reachable from `model` (the gate parameter and the router's reference), then
    routes random logits through the first live router to show that its top-k never leaves [0, active)."""
    import torch
    seen: dict = {}
    routers: list = []
    for _, mod in model.named_modules():
        router = getattr(mod, "router", None)
        if router is not None and hasattr(router, "select_experts"):
            routers.append(router)
        for tensor in (getattr(mod, "e_score_correction_bias", None), getattr(router, "e_score_correction_bias", None)):
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.shape[0] == width:
                seen.setdefault(tensor.data_ptr(), tensor)
    with torch.no_grad():
        for tensor in seen.values():
            tensor.data[active:] = ROUTING_MASK_BIAS
    probe: dict = {}
    if routers and seen:
        dev = next(iter(seen.values())).device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        try:
            _, ids = routers[0].select_experts(
                hidden_states=torch.zeros(512, 8, dtype=torch.bfloat16, device=dev),
                router_logits=torch.randn(512, width, dtype=torch.float32, device=dev),
                topk_indices_dtype=torch.int32,
                input_ids=torch.randint(0, 1000, (512,), device=dev))
            probe = {"max_expert_id": int(ids.max())}
        except Exception as exc:  # noqa: BLE001 - the probe is evidence; its failure is reported, not raised
            probe = {"error": f"{type(exc).__name__}: {exc}"}
    return {"bias_tensors": len(seen), "routers": len(routers), "probe": probe}


def apply_routing_mask(llm, active: int, width: int, layers: int) -> dict:
    """Mask routed experts >= `active` after the engine is up, in place. The router adds the bias to its scores
    before the top-k (csrc/moe/topk_softplus_sqrt_kernels.cu:510-516) and reads `e_score_correction_bias.data` on
    every call (router/fused_topk_bias_router.py:439); the captured CUDA graphs read the same storage, so the write
    needs no recapture. In-process engine only (LLM.apply_model -> UniProcExecutor.collective_rpc, no pickling).
    `applied` is true only if >= `layers` bias tensors were written AND the probe saw no expert id >= active."""
    try:
        got = llm.apply_model(functools.partial(_mask_routing, active=active, width=width))[0]
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "active": active, "error": f"{type(exc).__name__}: {exc}"}
    ok = got["bias_tensors"] >= layers and got["probe"].get("max_expert_id", active) < active and "max_expert_id" in got["probe"]
    return {"applied": ok, "active": active, **got,
            **({} if ok else {"error": "mask not proven: need >= %d bias tensors and a probe whose largest expert id is "
                                       "< %d, got %s" % (layers, active, got)})}


ENGINE_CHECKS = (
    # name, regex over the captured log, regex the prod value must match (boot5.log / ROLL.md)
    ("nvfp4_moe_backend", r"Using '(\w+)' NvFp4 MoE backend", r"^FLASHINFER_CUTLASS$"),
    ("draft_mxfp4_moe_backend", r"Using '(\w+)' Mxfp4 MoE backend", r"^DEEPGEMM_MXFP4$"),
    ("mxfp8_gemm_kernel", r"Using (\w+) for MXFP8 GEMM", r"^FlashInferCutlassMxfp8LinearKernel$"),
    ("kv_cache_format", r"Using DeepSeek's (\w+) KV cache format", r"^fp8_ds_mla$"),
    ("indexer_cache", r"Using (\w+) indexer cache for Lightning Indexer", r"^FP8$"),
    ("model_runner_v2", r"(Using V2 Model Runner)", r"V2"),
    ("breakable_cudagraph", r"(Breakable CUDA graph enabled)", r"enabled"),
    ("engram_mxfp4_overlay", r"\[suffix dsv41-engram-mxfp4\] (ACTIVE)", r"^ACTIVE$"),
    ("block64_hook", r"\[suffix dsv41-sm120-block64\] (ACTIVE)", r"^ACTIVE$"),
    ("engram_offload", r"(Engram table offloaded to pinned host memory: [^\n]*)", r"offloaded"),
    ("dspark_draft_loaded", r"(DSpark draft model loaded: \d+ params)", r"loaded"),
    ("model_load_gib", r"Model loading took ([\d.]+) GiB", None),
    ("kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens", None),
    ("kv_max_concurrency", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", None),
    ("kv_available_gib", r"Available KV cache memory: ([\d.]+) GiB", None),
    ("graph_capture_s", r"Graph capturing finished in (\d+) secs", None),
    ("graph_pool_gib", r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB", None),
)


def evaluate_checks(text: str) -> dict:
    out = {}
    for name, rx, want in ENGINE_CHECKS:
        m = re.search(rx, text)
        got = m.group(1) if m else None
        out[name] = {"found": got, "expected_prod": want,
                     "match": (None if want is None else (got is not None and re.search(want, got) is not None))}
    return out


def _int(text) -> int | None:
    try:
        return int(str(text).replace(",", ""))
    except ValueError:
        return None


def _verbatim(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-24000:]


def spec_snapshot(llm) -> dict | None:
    try:
        metrics = llm.get_metrics()
    except Exception:  # noqa: BLE001
        return None
    snap = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": []}
    found = False
    for m in metrics:
        if m.name == "vllm:spec_decode_num_drafts":
            snap["drafts"] += m.value
            found = True
        elif m.name == "vllm:spec_decode_num_draft_tokens":
            snap["draft_tokens"] += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            snap["accepted"] += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            snap["per_pos"] = list(m.values)
    return snap if found else None


def run_steps(eng, n_steps: int, clock: Clock):
    """Step the engine until n_steps outputs arrived. Returns (gaps_s, tokens per step, wall_s, finished ids).
    A step that returns nothing (async pipeline fill) is folded into the next gap."""
    gaps, toks, finished = [], [], []
    t_start = last = time.perf_counter()
    empty = 0
    while len(gaps) < n_steps:
        clock.check()
        outs = eng.step()
        now = time.perf_counter()
        if not outs:
            empty += 1
            if empty > 200000:
                raise SliceError("engine produced no output for 200000 consecutive steps", 5)
            continue
        empty = 0
        n = 0
        for out in outs:
            if out.outputs:
                n += len(out.outputs[0].token_ids)
            if out.finished:
                finished.append(out.request_id)
        gaps.append(now - last)
        toks.append(n)
        last = now
    return gaps, toks, last - t_start, finished


def start_streams(eng, ids: list, clock: Clock, timeout_s: float) -> float:
    t0 = time.perf_counter()
    started: set = set()
    while len(started) < len(ids):
        clock.check()
        if time.perf_counter() - t0 > timeout_s:
            raise SliceError(f"only {len(started)}/{len(ids)} streams started within {timeout_s:.0f} s "
                             "(KV pool too small for conc x prompt_tokens, or the engine is stuck)", 5)
        for out in eng.step():
            if out.outputs and out.outputs[0].token_ids:
                started.add(out.request_id)
            if out.finished:
                raise SliceError(f"request {out.request_id} finished during prefill", 5)
    return time.perf_counter() - t0


def drain(eng, clock: Clock, limit: int = 5000) -> None:
    for _ in range(limit):
        if not eng.has_unfinished_requests():
            return
        clock.check()
        eng.step()
    raise SliceError("engine did not drain after abort", 5)


def profile_window(eng, n_steps: int, clock: Clock) -> dict:
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gaps, _, wall, _ = run_steps(eng, n_steps, clock)
        torch.cuda.synchronize()
    events = [(e.name, e.time_range.start, e.time_range.end)
              for e in prof.events() if e.device_type == DeviceType.CUDA]
    agg = aggregate_kernels(events, len(gaps), wall / len(gaps) * 1e3)
    if not events:
        agg["warning"] = ("profiler saw no CUDA events (graph replays invisible to CUPTI here?): rerun with "
                          "cudagraph_mode=NONE to get a table")
    return agg


def measure_concurrency(llm, c: int, o: dict, vocab: int, clock: Clock, tel: Telemetry, rec: Recorder) -> dict:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    eng = llm.llm_engine
    rng = random.Random(o["seed"] * 7919 + c)
    params = SamplingParams(temperature=o["temperature"], max_tokens=max_tokens_for(o), ignore_eos=True,
                            detokenize=False, output_kind=RequestOutputKind.DELTA)
    ids = [f"slice-c{c}-{int(time.time() * 1000) % 10 ** 8}-{i}" for i in range(c)]
    for rid in ids:
        eng.add_request(rid, {"prompt_token_ids": [rng.randrange(16, vocab) for _ in range(o["prompt_tokens"])]}, params)
    res: dict = {"c": c, "max_tokens": max_tokens_for(o)}
    try:
        res["prefill_s"] = round(start_streams(eng, ids, clock, max(300.0, 3.0 * c * o["prompt_tokens"] / 5000)), 3)
        if o["warmup_steps"]:
            run_steps(eng, o["warmup_steps"], clock)
        before = spec_snapshot(llm)
        tel.start()
        try:
            gaps, toks, wall, finished = run_steps(eng, o["measure_steps"], clock)
        finally:
            res["telemetry"] = tel.stop()
        res.update(step_stats(gaps, toks, wall, c))
        res["spec"] = spec_delta(before, spec_snapshot(llm), c)
        res["k_effective"] = k_eff = effective_k(o, c)
        vps = (res["spec"] or {}).get("verified_per_stream")
        if vps is not None and abs(vps - (k_eff + 1)) > 0.05:
            rec.warn(f"c={c}: vLLM verified {vps:.2f} tokens per stream per step, k={o['k']} with the schedule "
                     f"{o['num_speculative_tokens_per_batch_size']} says {k_eff + 1}: the schedule is not in effect "
                     "for this batch size")
        if finished:
            rec.warn(f"c={c}: {len(finished)} stream(s) finished inside the window; step stats are not steady state")
        res["slice_step_ms"] = res["step_ms"]["mean"]
        res["prod_equivalent_ms_est"] = prod_equivalent_ms(o, c, res["slice_step_ms"])
        res["live_step_ms"] = LIVE_STEP_MS.get(c)
        res["expected_slice_ms"] = expected_slice_ms(o, c)
        if res["expected_slice_ms"]:
            res["ratio_to_expected"] = round(res["slice_step_ms"] / res["expected_slice_ms"], 3)
        rates = synthetic_rates(o)
        if rates and res["spec"] and res["spec"].get("per_pos_cum_accept"):
            n = res["spec"]["drafts"]
            worst = max(abs(a - b) / max(math.sqrt(max(b * (1 - b), 1e-4) / n), 1e-9)
                        for a, b in zip(res["spec"]["per_pos_cum_accept"], rates))
            res["synthetic_check"] = {"target": rates, "max_abs_z": round(worst, 2)}
            if worst > 4.5:
                rec.warn(f"c={c}: measured cumulative acceptance deviates from the synthetic target (max |z|={worst:.1f})")
        if o["profile"]:
            res["profile"] = profile_window(eng, o["profile_steps"], clock)
    finally:
        eng.abort_request(ids)
        drain(eng, clock)
    return res


def run_slice(o: dict, rec: Recorder, tee: Tee, clock: Clock, model_dir: str, plan: dict) -> int:
    with phase(rec, "import_vllm"):
        try:
            import torch  # noqa: F401
            from vllm import LLM
        except Exception as exc:  # noqa: BLE001
            rec.set("error", {"phase": "import_vllm", "type": type(exc).__name__, "message": str(exc)})
            raise SliceError(f"cannot import torch/vllm: {exc}", 7) from exc
        versions = collect_versions()
    rec.set("versions", versions)
    if not versions.get("cuda_available"):
        raise SliceError("no CUDA device visible to torch", 7)
    if not str(versions.get("vllm", "")).startswith(VLLM_VERSION_PREFIX):
        rec.warn(f"vLLM {versions.get('vllm')} is not the pinned {VLLM_VERSION_PREFIX}: derivations in this tool "
                 "were read from that commit")
    tel = Telemetry()
    rec.set("gpu_nvml", tel.info())
    log("engine kwargs: " + json.dumps(plan["engine_kwargs"], default=str))
    log("config diff vs real: " + json.dumps(plan["config_diff"], default=str))
    log(f"versions: torch {versions['torch']} vllm {versions['vllm']} cuda {versions.get('cuda')}")
    for w in plan.get("warnings", []):  # the pre-flight estimates (KV pool, routing mask) belong in the result too
        rec.warn(w)
    if o["shared_expert_tp"]:
        apply_shared_expert_patch()
    with phase(rec, "engine_init"):
        try:
            llm = LLM(**plan["engine_kwargs"])
        except BaseException as exc:  # noqa: BLE001 - verbatim, whatever vLLM (or the plugin's SystemExit) raised
            if isinstance(exc, (SliceError, KeyboardInterrupt)):
                raise
            text = f"{type(exc).__name__}: {exc}"
            hint = engine_hint(text)
            rec.set("engine_error", text)
            rec.set("engine_traceback", _verbatim(exc))
            rec.set("engine_hint", hint)
            raise SliceError(f"the engine refused this configuration: {text[:500]}" + (f" -- hint: {hint}" if hint else ""),
                             3) from exc
    with contextlib.suppress(Exception):
        versions["gpu"] = gpu_properties()
        log(f"gpu: {versions['gpu']}")
    tee.sync()
    checks = evaluate_checks(tee.text())
    facts = {"log_checks": checks}
    cc = getattr(getattr(llm.llm_engine, "vllm_config", None), "cache_config", None)
    if cc is not None:
        facts["num_gpu_blocks"] = getattr(cc, "num_gpu_blocks", None)
    try:
        free, total = torch.cuda.mem_get_info()
        facts["gpu_mem_free_gib"], facts["gpu_mem_total_gib"] = round(free / 2 ** 30, 2), round(total / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        pass
    rec.set("engine", facts)
    for name, ch in checks.items():
        if ch["match"] is False:
            rec.warn(f"engine check {name}: got {ch['found']!r}, prod has {ch['expected_prod']!r}")
        elif ch["match"] is None and ch["expected_prod"] and ch["found"] is None:
            rec.warn(f"engine check {name}: not seen in the log (prod logs it)")
    o_run = o
    if o["active_experts"] < o["slice_experts"]:
        with phase(rec, "routing_mask"):
            mask = apply_routing_mask(llm, o["active_experts"], o["slice_experts"], plan["layers"])
        rec.set("routing_mask", mask)
        if mask["applied"]:
            log(f"routing mask applied: {mask}")
        else:
            o_run = {**o, "active_experts": o["slice_experts"]}
            rec.warn(f"routing mask NOT applied ({mask.get('error')}): all {o['slice_experts']} resident experts are "
                     "reachable, so the MoE term is charged unmasked in prod_equivalent_ms_est / expected_slice_ms")
    kv_tokens = _int(checks["kv_cache_tokens"]["found"])
    vocab = plan["slice_vocab"]
    for c in o["conc"]:
        need = c * (o["prompt_tokens"] + max_tokens_for(o) + 1)
        if kv_tokens and need > 0.9 * kv_tokens:
            why = (f"c={c}: needs ~{need:,} KV tokens (c x (prompt_tokens + max_tokens)), the booted pool holds "
                   f"{kv_tokens:,} (vLLM's own count): skipped")
            rec.warn(why)
            with rec.lock:
                rec.doc["results"][str(c)] = {"c": c, "skipped": why, "tokens_needed": need, "kv_cache_tokens": kv_tokens}
            continue
        with phase(rec, f"measure_c{c}"):
            try:
                res = measure_concurrency(llm, c, o_run, vocab, clock, tel, rec)
            except SliceError:
                raise
            except Exception as exc:  # noqa: BLE001
                rec.set("error", {"phase": f"measure_c{c}", "type": type(exc).__name__, "message": str(exc),
                                  "traceback": _verbatim(exc)})
                raise SliceError(f"measurement at c={c} failed: {type(exc).__name__}: {exc}", 5) from exc
        with rec.lock:
            rec.doc["results"][str(c)] = res
        rec.checkpoint()
    if not any("skipped" not in r for r in rec.doc["results"].values()):
        raise SliceError("no concurrency fit the KV pool: lower prompt_tokens/conc or free memory "
                         "(slice_experts, gpu_memory_utilization)", 5)
    return 0


def real_run(o: dict, args) -> int:
    env = env_plan(o)
    maybe_reexec(env)
    out = os.fdopen(os.dup(1), "w")
    tee = Tee(os.dup(2))
    rec = Recorder(out)
    rec.out_path = args.out
    clock = Clock(o["deadline_s"])
    model_dir = None
    state = {"dir": None}

    def cleanup():
        if state["dir"] and not args.keep_dir:
            shutil.rmtree(state["dir"], ignore_errors=True)

    def fire(code, why):
        rec.warn(why)
        rec.doc["deadline_hit"] = (code == 4)
        rec.emit(code, partial=True)
        cleanup()
        os._exit(code)

    timer = threading.Timer(o["deadline_s"] + 30, fire, args=(4, f"deadline_s={o['deadline_s']} (+30 s grace) reached"))
    timer.daemon = True
    timer.start()
    signal.signal(signal.SIGTERM, lambda *_: fire(143, "SIGTERM"))
    code = 1
    try:
        cfg, _ = slice_config(o)
        with phase(rec, "build_config"):
            model_dir, files = write_model_dir(o["work_dir"], cfg)
            state["dir"] = model_dir
            plan = build_plan(o, model_dir, files)
            plan["slice_vocab"] = cfg["text_config"]["vocab_size"]
            plan["layers"] = cfg["text_config"]["num_hidden_layers"]
        rec.set("opts", o)
        rec.set("model_dir", model_dir if args.keep_dir else "<removed at exit>")
        rec.set("plan", plan)
        rec.set("env_applied", {k: os.environ.get(k) for k in sorted(env)})
        code = run_slice(o, rec, tee, clock, model_dir, plan)
    except SliceError as exc:
        log(f"ERROR ({exc.code}): {exc}")
        if rec.doc["error"] is None:
            rec.set("error", {"phase": "run", "type": "SliceError", "message": str(exc)})
        code = exc.code
        if code == 4:
            rec.doc["deadline_hit"] = True
    except KeyboardInterrupt:
        code = 130
    except BaseException as exc:  # noqa: BLE001
        log("UNEXPECTED " + _verbatim(exc))
        rec.set("error", {"phase": "run", "type": type(exc).__name__, "message": str(exc), "traceback": _verbatim(exc)})
        code = 1
    finally:
        timer.cancel()
        tee.sync(3.0)
        tee_text = tee.text()
        if args.log_file:
            with contextlib.suppress(OSError):
                Path(args.log_file).write_text(tee_text)
        rec.doc["log_tail"] = tee_text[-6000:]
        rec.emit(code, partial=code in (4, 130, 143) or (code != 0 and bool(rec.doc["results"])))
        if args.out:
            with contextlib.suppress(OSError):
                Path(args.out).write_text(json.dumps(_clean(rec.doc), default=str) + "\n")
        cleanup()
    return code


def dry_run(o: dict, args, stdout) -> int:
    cfg, _ = slice_config(o)
    model_dir, files = write_model_dir(o["work_dir"], cfg)
    try:
        plan = build_plan(o, model_dir, files)
        plan["slice_vocab"] = cfg["text_config"]["vocab_size"]
        plan["layers"] = cfg["text_config"]["num_hidden_layers"]
        doc = {"schema": RESULT_SCHEMA, "ok": True, "exit": 0, "dry": True, "opts": o, "plan": plan,
               "model_dir": model_dir if args.keep_dir else "<removed after the dry run>"}
        text = json.dumps(_clean(doc), default=str, indent=None if not args.pretty else 1)
        (stdout or sys.stdout).write(text + "\n")
        if args.out:
            Path(args.out).write_text(text + "\n")
    finally:
        if not args.keep_dir:
            shutil.rmtree(model_dir, ignore_errors=True)
    return 0


def main(argv=None, stdout=None) -> int:
    ap = argparse.ArgumentParser(prog="dsv41_slice", description=__doc__.split("\n\n")[0])
    ap.add_argument("--opts", help="JSON object of options (see --print-schema)")
    ap.add_argument("--opts-file", help="file holding the JSON options")
    ap.add_argument("--dry", action="store_true", help="build the model dir and print the plan; no vLLM import")
    ap.add_argument("--print-schema", action="store_true", help="print every option with bounds and defaults")
    ap.add_argument("--keep-dir", action="store_true", help="do not delete the temp model dir")
    ap.add_argument("--out", help="also write the result JSON here")
    ap.add_argument("--log-file", help="write the full captured log here")
    ap.add_argument("--pretty", action="store_true", help="indent the dry-run JSON")
    args = ap.parse_args(argv)
    try:
        if args.print_schema:
            (stdout or sys.stdout).write(json.dumps({"schema": RESULT_SCHEMA, "opts": schema_doc(),
                                                     "families": [f for f, _ in KERNEL_FAMILIES],
                                                     "allowed_env": {"SUFFIX": sorted(ALLOWED_SUFFIX_ENV),
                                                                     "VLLM": sorted(ALLOWED_VLLM_ENV)}},
                                                    indent=1, default=str) + "\n")
            return 0
        if args.opts and args.opts_file:
            raise SliceError("pass --opts or --opts-file, not both", 2)
        text = Path(args.opts_file).read_text() if args.opts_file else (args.opts or "{}")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SliceError(f"--opts is not valid JSON: {exc}", 2) from exc
        o = parse_opts(raw)
    except SliceError as exc:
        print(f"dsv41-slice: {exc}", file=sys.stderr)
        return exc.code
    except OSError as exc:
        print(f"dsv41-slice: {exc}", file=sys.stderr)
        return 2
    return dry_run(o, args, stdout) if args.dry else real_run(o, args)


def _hard_exit(code: int) -> None:
    """vLLM's atexit teardown (pinned Engram tables, engine shutdown) can hang or print after the JSON."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _hard_exit(main())
