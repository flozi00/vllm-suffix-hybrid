# SPDX-License-Identifier: Apache-2.0
"""ROCm patches for vLLM modules, one gate each (armed by sitecustomize).

A patch rewrites a module's source before its first (only) execution and/or
calls an `after` hook with the executed module. Gate on and the anchor not
found exactly once -> that import fails: an enabled pool must never serve the
unpatched path silently. Stdlib only at import (hooks import lazily).

SUFFIX_ROCM_AITER_PAD=1: vLLM pre-rounds hidden_pad/intermediate_pad before
  rocm_aiter_ops.fused_moe and doubles intermediate_pad at TP1; AITER's CK
  stage1 doubles it again, so the kernel zero-pads live rows (MXFP4 W4A4 MoE at
  TP1 on gfx950: Qwen3.8-Flash-Next GSM8K 0.735 vs 0.964). Same edit as
  vllm-project/vllm#46201 (open).
SUFFIX_ROCM_QSA_TOPK_ROWS=1: the Qwen4Exp QSA indexer runs top_k_per_row_decode
  over every row of a step at once. Above 384 rows vLLM's gfx950 dispatch
  leaves the length-aware kernel for the generic radix topKPerRowDecode, whose
  code object declares a hostcall buffer, and HIP refuses that kernel on hosts
  without PCIe atomics (worker-09: "Pcie atomics not enabled, hostcall not
  supported", then a sticky HIP 401): the first step with >384 new tokens
  killed the engine. Chunks of <= 384 rows stay on the length-aware kernel;
  rows are independent, so the selection is unchanged.
SUFFIX_ROCM_QSA_MQA=1: the QSA indexer scores only each row's visible columns
  (suffix_hybrid/kernels/qsa_mqa_rocm.py) instead of the whole 262k-context
  page-table width: 16.1 of 62.9 ms/step at 32 concurrent MTP-4 requests.
SUFFIX_ROCM_AITER_FLYDSL_PAD=1: AITER v0.1.24.post1 bug. A tuned fMoE row that pairs a
  FlyDSL fp4 stage 1 (flydsl_moe1_*_fp4) with a flydsl_moe2_layout_* stage 2 passes the
  inter pad (vLLM zero-pads 640 -> 768) to stage 1, which leaves columns 640..767 and
  their e8m0 scales unwritten in torch.empty buffers, but stage 2 reads all 768 (no
  K-pad skip): stale 0xFF scales are NaN -> garbage (GSM8K 0.105 on the MI350P with
  our tuned table). Stage 1 now writes the zero tail for layout stage 2s (boot gate
  moe_fp4_oracle); the default and prefill paths keep the pad.
SUFFIX_ROCM_AITER_FLYDSL_ZERO=1: the cheaper fix for the same AITER bug (use instead of
  _FLYDSL_PAD): stage 1 keeps skipping the padded inter tail, but its fp4 output and
  sorted e8m0 scale buffers are allocated zeroed when inter_dim_pad > 0 (AITER's a16w4
  stage 1 already does this), so the layout stage 2 reads 0 * 2^-127 = 0 there.
SUFFIX_ROCM_AITER_FLYDSL_ZBUF=1: the same fix without the per-call memsets of _ZERO (use
  instead of _ZERO / _PAD; two fills per MoE layer = 100 launches per MTP-4 step): the
  padded stage-1 output and scales come from persistent zero-initialized buffers, one per
  layout, grown only. Stage 1 never writes the padded tail, so it stays zero.
SUFFIX_ROCM_QSA_SPARSE_SKIP=1: the QSA sparse attention kernel skips selection
  tiles whose slots are all -1 (every context shorter than the 2048-token
  budget pads the 2051-slot selection with -1); such a tile is a no-op in the
  online softmax, so the output is bit-identical (boot gate qsa_sparse_bench).
SUFFIX_ROCM_MXFP4_A16=1: dense MXFP4 linears (non-ASM AITER path) at M <=
  SUFFIX_ROCM_MXFP4_A16_MAX_M (default 32) run as one AITER gemm_a16wfp4 launch
  that quantizes the activations in-kernel, instead of dynamic_mxfp4_quant +
  gemm_afp4wfp4 (+ split-K reduce): ~180 fewer launches per MTP-4 decode step
  (suffix_hybrid/kernels/mxfp4_a16_rocm.py). Source rewrite, because
  gemm_with_dynamic_quant is registered as a custom op when the module executes.
SUFFIX_ROCM_GDN_MTP=1: the GDN MTP-verify core runs AITER's strided gated delta
  rule on the packed qkv (suffix_hybrid/kernels/gdn_mtp_rocm.py), 11 -> 3
  launches per GDN layer, and the output gate reads z from qkvz instead of a
  copy. Two anchors in one module: a gate may carry a list of patches.
SUFFIX_ROCM_HC_FUSE=1: Qwen4Exp GatedResidual.mix / combine_and_mix run
  hc_silu -> up GEMM -> hc_gate_mix as one kernel
  (suffix_hybrid/kernels/hc_fused_rocm.py): 2 launches fewer per HC site (218
  per MTP-4 decode step) and no [M, 10240] gate round trip.
SUFFIX_ROCM_GDN_ASYNC_IDX=1: the GDN metadata builder's mixed (prefill + spec
  decode) branch indexes GPU tensors with CPU bool masks; each one is a pageable
  blocking H2D copy (hipMemcpyWithStream) that waits for the GPU to drain, so
  no mixed step overlaps under async scheduling. The masks become device row
  indices once per build (nonzero + pinned async H2D, as vLLM's short_conv
  builder does), kept on the metadata for update_block_table's other GDN KV
  groups. Same rows in the same ascending order: bit-identical. Not ROCm-specific.
SUFFIX_ROCM_HC_DOWN=1: the same two methods run both HC down projections (merged
  down+inject 336x10240, final mixers' down 320x10240) through hc_down
  (suffix_hybrid/kernels/hc_down_rocm.py): at M <= SUFFIX_ROCM_HC_DOWN_MAX_M
  (default 64) a deterministic split-K Triton GEMM + fp32 reduce instead of
  hipBLASLt's split-K pair, above it vLLM's rocm_unquantized_gemm as before. Its
  anchors end where HC_FUSE's begin, so either gate works alone or with the other.
SUFFIX_ROCM_QSA_DENSE=1: Qwen4Exp QSA attention takes a request with seq_len <= the
  token top-k (2048) and <= 5 query rows (MTP-4 verify, decode) as one dense block over
  its context (suffix_hybrid/kernels/qsa_dense_rocm.py): its selection is exactly tokens
  0..p, which vLLM's sparse kernel gathers once per query row (179 us per layer at c32).
  Other requests keep vLLM's sparse kernel (copied with a row skip + the all -1 tile
  skip); MTP steps that reuse the step-0 selection (indexer.skip_topk) stay stock.
SUFFIX_ROCM_GDN_DEFER=1 (needs _GDN_MTP and _GDN_ASYNC_IDX): GDN MTP verify with deferred
  state commit (suffix_hybrid/kernels/gdn_defer_rocm.py): the state after token 0 goes to
  slot 0 and tokens 1.. leave only their inputs in slot 1's tile; the next step replays
  the accepted ones (bit-identical). 1 state write per request and layer instead of 5;
  steps near a mamba align block boundary keep vLLM's slot contract for its copies. The
  GDN metadata carries the spec rows' seq_lens and the align block size for that.
  With SUFFIX_ROCM_GDN_DEFER_MFMA=1 the same contract runs in chunk form on the fp32
  matrix cores (gdn_defer_rocm._gdn_defer_mfma_kernel; fp32-accurate, not bitwise; no
  source patch, read by gdn_defer).
SUFFIX_ROCM_TOPK_GATING=1: the MoE router's aiter.topk_softmax (vllm._aiter_ops) runs as one
  Triton program per token (suffix_hybrid/kernels/topk_gating_rocm.py): same indices, AITER's
  weight math; ~12 us per MoE layer in AITER's 16-threads-per-row kernel at c8..c32.
SUFFIX_ROCM_AFP4_CONFIGS=1: AITER's Triton MXFP4 GEMM (gemm_afp4wfp4, vLLM's non-ASM dense
  MXFP4 path) takes GEMM-AFP4WFP4-N=<N>-K=<K>.json from suffix_hybrid/configs/afp4/ (boot
  gate afp4_tune) before AITER's own tree. AITER keys these JSONs by arch + (N, K) only, so
  the 128-CU MI350P otherwise runs the 256-CU MI355X DEFAULT.json tiles. A shape without a
  plugin file resolves exactly as before. vLLM's preshuffle-tuned guard (probes 2x K) only
  gates the ASM path (VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1), not this one.
SUFFIX_ROCM_MOE_ROUTE=1: at M <= SUFFIX_ROCM_MOE_ROUTE_MAX_M (16) the MoE router's top-k,
  AITER's moe_sorting (P0_v2 + P23) and the stage-1 MXFP4 quant-sort run as two Triton
  launches (suffix_hybrid/kernels/moe_route_rocm.py): ids, sort and quant bit-identical, the
  gating weights too with the calibrated exp (SUFFIX_ROCM_MOE_ROUTE_EXP, 0 or 2); the
  AiterSharedRoutedFusedMoERouter defers its topk_softmax call to aiter.fused_moe's sort.
SUFFIX_ROCM_HC_BIG=1: Qwen4Exp GatedResidual.mix / combine_and_mix return through
  suffix_hybrid/kernels/hc_big_rocm.py: vLLM's norm, then one custom op for the rest of the
  site. At SUFFIX_ROCM_HC_BIG_MIN_M < M <= _MAX_M (16 < M <= 256: c8+ verify, c32 drafts)
  hc_down's split-K partials at a large-M config, their reduce + silu in one launch and
  the up GEMM + gate mix without the [M, 10240] gate (6 launches -> 4 at M = 160); at
  M <= MIN_M HC_DOWN's + HC_FUSE's kernels, above MAX_M vLLM's stock chain. The patch
  only inserts the return at the top of each method: composes with HC_FUSE / HC_DOWN.
SUFFIX_ROCM_QK_FUSED=1: the Qwen4Exp QSA layers run vLLM's fused split + QK GemmaRMSNorm +
  partial NeoX RoPE + gate copy (vllm/model_executor/layers/fused_qk_norm_rope.py, Triton)
  instead of the eager chain inductor compiles into ~4-5 launches: the AMD layer allows it
  on CUDA and text-only; here also ROCm and interleaved mRoPE (multimodal), with
  Qwen3NextAttention's own conditions (oracle: suffix_hybrid/kernels/qk_fused_rocm.py).
SUFFIX_ROCM_ACT_QUANT_FUSE=1: the GDN output (RMSNormGated -> out_proj) and the QSA output
  (attn * sigmoid(gate) -> o_proj) quantize their own MXFP4 activations
  (suffix_hybrid/kernels/act_quant_rocm.py): producer + AITER's _mxfp4_quant_op in one
  Triton launch, then gemm_afp4wfp4 as vLLM's non-ASM MXFP4 linear calls it; one launch
  fewer per site (52 per MTP-4 step). Only for out_proj / o_proj served by that kernel
  at TP 1; anything else keeps the stock code below the patched branch.
SUFFIX_JIT_LOG=1: one "[suffix jit]" line per Triton compile (kernel, wall ms): which kernels
  still JIT-compile while serving (each compile stalls every in-flight request).
"""
import glob
import importlib.util
import linecache
import os
import sys
import textwrap
from typing import NamedTuple


class Patch(NamedTuple):
    target: str
    label: str
    old: str = ""    # source anchor replaced by `new` before the first exec ("" = none)
    new: str = ""
    after: str = ""  # "module:function" called with the executed module ("" = none)


# GatedResidual.mix and .combine_and_mix end in the same three ops; the next def
# tells the two anchors apart.
_HC = "vllm.models.qwen4_exp.amd.hyperconnection"
_HC_TAIL = (
    "        lora = hc_silu(lora, self.hc_count)\n"
    "        gate = self.input_mix_weight_up(lora)  # [M, D]\n"
    "        block_input = hc_gate_mix(xn, gate, self.hc_count)\n"
    "\n"
    "        return hidden_states, block_input, injection\n"
    "\n")
_HC_FUSED = (
    "        block_input = hc_up_gate_mix(  # suffix rocm-hc-fuse: silu + up GEMM + gate mix\n"
    "            lora, self.input_mix_weight_up.weight, xn, self.hc_count\n"
    "        )\n"
    "\n"
    "        return hidden_states, block_input, injection\n"
    "\n")
# HC_BIG anchors: mix's signature and combine_and_mix's docstring end, which neither
# HC_FUSE nor HC_DOWN touches.
_HC_MIX_SIG = (
    "    def mix(\n"
    "        self, hidden_states: torch.Tensor\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n")
_HC_CAM_DOC = (
    "        block's mix. Its combine with ``block_output`` is fused with this\n"
    "        module's input RMSNorm.\n"
    '        """\n')
# Both methods also share the down projection branch; the norm call's argument before
# self.hc_norm.weight (hidden_states in mix, prev_injection in combine_and_mix) tells the
# two anchors apart. They end before _HC_TAIL: disjoint from the HC_FUSE anchors.
_HC_DOWN = (
    "            self.hc_norm.weight,\n"
    "            self.config.rms_norm_eps,\n"
    "            self.hc_count,\n"
    "        )\n"
    "\n"
    "        if self.use_combine:\n"
    "            # produce injection logits for combine\n"
    "            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]\n"
    "            down_and_injection = self.input_mix_weight_down_block_inject(xn)\n"
    "            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)\n"
    "        else:\n"
    "            lora = self.input_mix_weight_down(xn)\n")
_HC_DOWN_NEW = _HC_DOWN.replace(
    "self.input_mix_weight_down_block_inject(xn)",
    "hc_down(  # suffix rocm-hc-down: split-K at small M\n"
    "                xn, self.input_mix_weight_down_block_inject.weight\n"
    "            )").replace(
    "self.input_mix_weight_down(xn)", "hc_down(xn, self.input_mix_weight_down.weight)")

_QSA_SPARSE_BODY = (  # vllm/models/qwen4_exp/amd/ops/qsa.py @81198e97: tile body of
    # _qsa_sparse_paged_gqa_splitk_kernel's loop, byte-exact
    '        safe_token = tl.maximum(logical_token, 0)\n'
    '        logical_page = safe_token // PAGE_SIZE\n'
    '        page_offset = safe_token % PAGE_SIZE\n'
    '        valid = (\n'
    '            (request >= 0)\n'
    '            & (request < num_requests)\n'
    '            & (logical_token >= 0)\n'
    '            & (logical_page < PAGE_TABLE_WIDTH)\n'
    '        )\n'
    '        physical_page = tl.load(\n'
    '            block_table_ptr\n'
    '            + safe_request * stride_table_req\n'
    '            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),\n'
    '            mask=valid,\n'
    '            other=-1,\n'
    '        )\n'
    '        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)\n'
    '        # physical_page * block stride can overflow int32 for large caches.\n'
    '        safe_page = tl.maximum(physical_page, 0).to(tl.int64)\n'
    '        keys = tl.load(\n'
    '            k_cache_ptr\n'
    '            + safe_page[None, :] * stride_k_block\n'
    '            + page_offset[None, :] * stride_k_token\n'
    '            + kv_head * stride_k_head\n'
    '            + dim_offsets[:, None],\n'
    '            mask=valid[None, :],\n'
    '            other=0.0,\n'
    '        )\n'
    '        values = tl.load(\n'
    '            v_cache_ptr\n'
    '            + safe_page[:, None] * stride_v_block\n'
    '            + page_offset[:, None] * stride_v_token\n'
    '            + kv_head * stride_v_head\n'
    '            + dim_offsets[None, :],\n'
    '            mask=valid[:, None],\n'
    '            other=0.0,\n'
    '        )\n'
    '        scores = tl.dot(query, keys)\n'
    '        # Scaling scores avoids re-quantizing a scaled query to BF16.\n'
    '        scores *= softmax_scale_log2\n'
    '        scores = tl.where(valid[None, :], scores, -1.0e20)\n'
    '        next_max = tl.maximum(max_value, tl.max(scores, axis=1))\n'
    '        alpha = tl.math.exp2(max_value - next_max)\n'
    '        probabilities = tl.where(\n'
    '            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0\n'
    '        )\n'
    '        accumulator = tl.dot(\n'
    '            probabilities.to(values.dtype),\n'
    '            values,\n'
    '            acc=accumulator * alpha[:, None],\n'
    '        )\n'
    '        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)\n'
    '        max_value = next_max\n'
)

# (site, anchor, rewrite) in vllm/v1/attention/backends/gdn_attn.py @81198e97, applied
# in order: every GPU[cpu_mask] of the mixed branch becomes GPU[device row index].
# CPU tensors (query_lens_cpu) keep the CPU masks.
_GDN_TAG = "  # suffix gdn-async-idx\n"
_GDN_FIELD = "    token_chunk_offset_ptr: torch.Tensor | None = None\n"
_GDN_INIT = "        spec_sequence_masks_cpu: torch.Tensor | None = None\n"
_GDN_CTOR = "            spec_sequence_masks_cpu=spec_sequence_masks_cpu,\n"
_GDN_ASYNC_IDX = (
    ("metadata fields", _GDN_FIELD, _GDN_FIELD +
     "    # suffix gdn-async-idx: a mixed batch's device row indices, for update_block_table\n"
     "    spec_req_idx: torch.Tensor | None = None\n"
     "    non_spec_req_idx: torch.Tensor | None = None\n"),
    ("build init", _GDN_INIT, _GDN_INIT + "        spec_req_idx = non_spec_req_idx = None" + _GDN_TAG),
    ("state indices",
     "                spec_state_indices_tensor = block_table_tensor[\n"
     "                    spec_sequence_masks_cpu, : self.num_spec + 1\n"
     "                ]\n"
     "                non_spec_state_indices_tensor = block_table_tensor[\n"
     "                    non_spec_sequence_masks_cpu, 0\n"
     "                ]\n",
     "                # suffix gdn-async-idx: GPU[cpu_mask] is a blocking pageable H2D copy;\n"
     "                # the row indices go over once, pinned and async.\n"
     "                spec_req_idx = async_tensor_h2d(\n"
     "                    spec_sequence_masks_cpu.nonzero().flatten(),\n"
     "                    device=query_start_loc.device,\n"
     "                )\n"
     "                non_spec_req_idx = async_tensor_h2d(\n"
     "                    non_spec_sequence_masks_cpu.nonzero().flatten(),\n"
     "                    device=query_start_loc.device,\n"
     "                )\n"
     "                spec_state_indices_tensor = block_table_tensor[\n"
     "                    spec_req_idx, : self.num_spec + 1\n"
     "                ]\n"
     "                non_spec_state_indices_tensor = block_table_tensor[\n"
     "                    non_spec_req_idx, 0\n"
     "                ]\n"),
    ("spec query_start_loc",
     "                    query_lens[spec_sequence_masks_cpu],\n",
     "                    query_lens[spec_req_idx]," + _GDN_TAG),
    ("non-spec query_start_loc",
     "                    query_lens[non_spec_sequence_masks_cpu],\n",
     "                    query_lens[non_spec_req_idx]," + _GDN_TAG),
    ("num_accepted_tokens",
     "                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]\n",
     "                num_accepted_tokens = num_accepted_tokens[spec_req_idx]" + _GDN_TAG),
    ("has_initial_state",
     "                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]\n",
     "                has_initial_state = has_initial_state[non_spec_req_idx]" + _GDN_TAG),
    ("metadata ctor", _GDN_CTOR, _GDN_CTOR +
     "            spec_req_idx=spec_req_idx," + _GDN_TAG +
     "            non_spec_req_idx=non_spec_req_idx,\n"),
    ("update_block_table",
     "            spec_indices = blk_table[masks, : self.num_spec + 1]\n"
     "            non_spec_indices = prefill_indices = blk_table[~masks, 0]\n",
     "            # suffix gdn-async-idx: build()'s device row indices, no blocking H2D\n"
     "            assert m.spec_req_idx is not None and m.non_spec_req_idx is not None\n"
     "            spec_indices = blk_table[m.spec_req_idx, : self.num_spec + 1]\n"
     "            non_spec_indices = prefill_indices = blk_table[m.non_spec_req_idx, 0]\n"),
)

# Tuned AITER gemm_afp4wfp4 JSONs (afp4_tune), named as AITER names them (K logical).
AFP4_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "afp4")
# aiter/ops/triton/utils/gemm_config_utils.py @ v0.1.24.post1, _get_gemm_config_cached:
# AITER's N/K file probe, byte-exact (once in the file).
_AFP4_PROBE = (
    "    for suffix in specialized_suffixes:\n"
    "        specialized_config = load_config_json(\n"
    '            f"{cfg_dir}/{config_name}-{suffix}.json", required=False\n'
    "        )\n")

# ACT_QUANT_FUSE anchors: _output_projection's body (vLLM 81198e97) and QSA forward's tail.
_AQ_GDN = (
    "        core_attn_out = self.norm(core_attn_out, z)\n"
    "        output, _ = self.out_proj(core_attn_out.flatten(-2))\n"
    "        return output\n")
_AQ_QSA = (
    "        flat_output = attn_output.view(num_tokens, -1)\n"
    "        if gate is not None:\n"
    "            flat_output = flat_output * torch.sigmoid(gate)\n"
    "        output, _ = self.o_proj(flat_output)\n"
    "        return output\n")

# gate -> Patch, or a tuple of Patches the gate applies together.
PATCHES = {
    "SUFFIX_ROCM_AITER_PAD": Patch(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        "raw MoE padding to AITER",
        "            hidden_pad = hidden_pad // 128 * 128\n"
        "            intermediate_pad = (\n"
        "                intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)\n"
        "            )\n",
        "            pass  # suffix rocm-aiter-pad: AITER aligns internally (vllm#46201)\n"),
    "SUFFIX_ROCM_QSA_TOPK_ROWS": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA top-k in <=384-row chunks",
        "    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))\n",
        "    rows_per_chunk = min(384, max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))"
        "  # suffix: hostcall-free top-k\n"),
    "SUFFIX_ROCM_QSA_MQA": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA scores over visible columns only",
        after="suffix_hybrid.kernels.qsa_mqa_rocm:install"),
    "SUFFIX_ROCM_AITER_FLYDSL_PAD": Patch(
        "aiter.fused_moe",
        "FlyDSL fp4 stage 1 writes the padded inter columns for a layout stage 2",
        "            stage1_func = functools.partial(\n"
        "                _flydsl_stage1_wrapper,\n"
        "                kernelName=kernelName1,\n"
        "                activation=activation,\n"
        "                inter_dim_pad=intermediate_pad,\n"
        "                model_dim_pad=hidden_pad,\n"
        "            )\n",
        "            stage1_func = functools.partial(\n"
        "                _flydsl_stage1_wrapper,\n"
        "                kernelName=kernelName1,\n"
        "                activation=activation,\n"
        "                # suffix SUFFIX_ROCM_AITER_FLYDSL_PAD: moe2_layout GEMM2 has no K-pad skip\n"
        "                inter_dim_pad=0 if is_flydsl2_layout else intermediate_pad,\n"
        "                model_dim_pad=hidden_pad,\n"
        "            )\n"),
    "SUFFIX_ROCM_AITER_FLYDSL_ZERO": (
        Patch("aiter.ops.flydsl.moe_kernels",
              "FlyDSL fp4 stage 1 output zeroed when the inter dim is padded",
              "            if _need_fp4:\n"
              "                out = torch.empty(\n"
              "                    (_sorted_rows, inter_dim // 2), dtype=dtypes.fp4x2, device=dev\n"
              "                )\n",
              "            if _need_fp4:\n"
              "                out = torch.empty(\n"
              "                    (_sorted_rows, inter_dim // 2), dtype=dtypes.fp4x2, device=dev\n"
              "                )\n"
              "                if inter_dim_pad > 0:  # suffix SUFFIX_ROCM_AITER_FLYDSL_ZERO: no stale tail\n"
              "                    out.view(torch.uint8).zero_()  # fill_cuda has no fp4x2 kernel\n"),
        Patch("aiter.ops.flydsl.moe_kernels",
              "FlyDSL fp4 stage 1 scales zeroed when the inter dim is padded",
              "    out_scale_sorted_flat = (\n"
              "        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)\n"
              "        if _need_sort\n",
              "    out_scale_sorted_flat = (  # suffix SUFFIX_ROCM_AITER_FLYDSL_ZERO\n"
              "        (torch.zeros if inter_dim_pad > 0 else torch.empty)(\n"
              "            padded_rows * padded_cols, dtype=torch.uint8, device=dev)\n"
              "        if _need_sort\n"),
    ),
    "SUFFIX_ROCM_AITER_FLYDSL_ZBUF": (
        Patch("aiter.ops.flydsl.moe_kernels", "FlyDSL zeroed-buffer helper",
              "_KERNEL_PARAMS: dict[str, dict] = {}\n",
              "_KERNEL_PARAMS: dict[str, dict] = {}\n"
              "# suffix SUFFIX_ROCM_AITER_FLYDSL_ZBUF: stage 1 never writes the padded inter tail of\n"
              "# its fp4 output / sorted e8m0 scales. One zero-initialized buffer per layout (key),\n"
              "# grown only, every generation kept alive (captured HIP graphs hold the addresses):\n"
              "# the tail stays zero from the allocation on, with no per-call memset.\n"
              "_suffix_zbufs: dict = {}\n"
              "\n"
              "\n"
              "def _suffix_zeroed(numel, key, dev):\n"
              "    bufs = _suffix_zbufs.setdefault((key, str(dev)), [])\n"
              "    if not bufs or bufs[-1].numel() < numel:\n"
              "        size = max(numel, 2 * bufs[-1].numel()) if bufs else numel\n"
              "        bufs.append(torch.zeros(size, dtype=torch.uint8, device=dev))\n"
              "    return bufs[-1][:numel]\n"),
        Patch("aiter.ops.flydsl.moe_kernels",
              "FlyDSL fp4 stage 1 output from a persistent zeroed buffer when the inter dim is padded",
              "            if _need_fp4:\n"
              "                out = torch.empty(\n"
              "                    (_sorted_rows, inter_dim // 2), dtype=dtypes.fp4x2, device=dev\n"
              "                )\n",
              "            if _need_fp4 and inter_dim_pad > 0:  # suffix SUFFIX_ROCM_AITER_FLYDSL_ZBUF\n"
              "                out = _suffix_zeroed(\n"
              "                    _sorted_rows * (inter_dim // 2), (\"out\", inter_dim, inter_dim_pad), dev\n"
              "                ).view(dtypes.fp4x2).view(_sorted_rows, inter_dim // 2)\n"
              "            elif _need_fp4:\n"
              "                out = torch.empty(\n"
              "                    (_sorted_rows, inter_dim // 2), dtype=dtypes.fp4x2, device=dev\n"
              "                )\n"),
        Patch("aiter.ops.flydsl.moe_kernels",
              "FlyDSL fp4 stage 1 scales from a persistent zeroed buffer when the inter dim is padded",
              "    out_scale_sorted_flat = (\n"
              "        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)\n"
              "        if _need_sort\n",
              "    out_scale_sorted_flat = (  # suffix SUFFIX_ROCM_AITER_FLYDSL_ZBUF\n"
              "        _suffix_zeroed(padded_rows * padded_cols,\n"
              "                       (\"scale\", padded_cols, inter_dim, inter_dim_pad), dev)\n"
              "        if _need_sort and inter_dim_pad > 0\n"
              "        else torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)\n"
              "        if _need_sort\n"),
    ),
    "SUFFIX_ROCM_QSA_SPARSE_SKIP": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA sparse attention skips all-padding tiles",
        _QSA_SPARSE_BODY,
        "        if tl.max(logical_token, axis=0) >= 0:  # suffix: an all -1 tile is a no-op\n"
        + textwrap.indent(_QSA_SPARSE_BODY, "    ")),
    "SUFFIX_ROCM_QSA_DENSE": Patch(
        "vllm.models.qwen4_exp.amd.qsa",
        "QSA attention: dense path for requests within the token top-k",
        "        from .ops.qsa import qsa_sparse_paged_attention\n"
        "\n"
        "        qsa_sparse_paged_attention(\n"
        "            query[:num_tokens],\n"
        "            key_cache,\n"
        "            value_cache,\n"
        "            logical_indices,\n"
        "            attn_metadata.block_table,\n"
        "            token_to_req,\n"
        "            output[:num_tokens],\n"
        "        )\n",
        "        _suffix_qsa_attention(  # suffix SUFFIX_ROCM_QSA_DENSE: dense short requests\n"
        "            query[:num_tokens],\n"
        "            key_cache,\n"
        "            value_cache,\n"
        "            logical_indices,\n"
        "            attn_metadata.block_table,\n"
        "            token_to_req,\n"
        "            output[:num_tokens],\n"
        "            attn_metadata.seq_lens,\n"
        "            attn_metadata.query_start_loc,\n"
        "            layer.indexer.token_topk,\n"
        "            dense=not layer.indexer.skip_topk,  # MTP steps > 0 reuse step 0's rows\n"
        "        )\n",
        "suffix_hybrid.kernels.qsa_dense_rocm:install"),
    "SUFFIX_ROCM_TOPK_GATING": Patch(
        "vllm._aiter_ops", "MoE top-k gating: one Triton program per token",
        "    from aiter import topk_softmax\n"
        "\n"
        "    topk_softmax(\n"
        "        topk_weights,\n",
        "    from suffix_hybrid.kernels.topk_gating_rocm import topk_softmax  # suffix\n"
        "\n"
        "    topk_softmax(\n"
        "        topk_weights,\n"),
    "SUFFIX_ROCM_MXFP4_A16": Patch(
        "vllm.model_executor.kernels.linear.mxfp4.aiter",
        "dense MXFP4 small M via AITER gemm_a16wfp4",
        "            if x_scales is None:\n"
        "                x_q, x_s = dynamic_mxfp4_quant(x)\n",
        "            if x_scales is None:  # suffix SUFFIX_ROCM_MXFP4_A16: small M -> gemm_a16wfp4\n"
        "                from suffix_hybrid.kernels.mxfp4_a16_rocm import gemm_a16\n"
        "                y = gemm_a16(x, weight, weight_scale, out_dtype)\n"
        "                if y is not None:\n"
        "                    return y\n"
        "                x_q, x_s = dynamic_mxfp4_quant(x)\n"),
    "SUFFIX_ROCM_GDN_MTP": [
        Patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            "GDN MTP verify via AITER strided gated delta rule",
            "        core_attn_out.zero_()\n"
            "        num_tokens_all = qkvz.shape[0]\n",
            "        if _suffix_gdn_mtp(self, qkvz, ba, core_attn_out, attn_metadata):\n"
            "            return  # suffix gdn-mtp: all-spec verify batch\n"
            "        core_attn_out.zero_()\n"
            "        num_tokens_all = qkvz.shape[0]\n",
            after="suffix_hybrid.kernels.gdn_mtp_rocm:install"),
        Patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            "GDN output gate reads z from qkvz",
            "            return self._output_projection(core_attn_out, z)\n"
            "        else:\n"
            "            return self.forward_cuda(hidden_states)\n",
            "            if not self.gqa_interleaved_layout:  # suffix gdn-mtp: z stays in qkvz\n"
            "                qkv_size = (2 * self.key_dim + self.value_dim) // self.tp_size\n"
            "                z = projected_states_qkvz[:, qkv_size:].view(z.shape)\n"
            "            return self._output_projection(core_attn_out, z)\n"
            "        else:\n"
            "            return self.forward_cuda(hidden_states)\n"),
    ],
    "SUFFIX_ROCM_HC_FUSE": (
        Patch(_HC, "HC silu + up GEMM + gate mix in one kernel (mix)",
              _HC_TAIL + "    def combine_and_mix(\n", _HC_FUSED + "    def combine_and_mix(\n",
              "suffix_hybrid.kernels.hc_fused_rocm:install"),
        Patch(_HC, "HC silu + up GEMM + gate mix in one kernel (combine_and_mix)",
              _HC_TAIL + "    def combine(\n", _HC_FUSED + "    def combine(\n"),
    ),
    "SUFFIX_ROCM_GDN_ASYNC_IDX": tuple(
        Patch("vllm.v1.attention.backends.gdn_attn",
              f"GDN mixed-batch gathers by device row index ({site})", old, new)
        for site, old, new in _GDN_ASYNC_IDX),
    "SUFFIX_ROCM_GDN_DEFER": (
        Patch("vllm.v1.attention.backends.gdn_attn", "GDN metadata: spec seq_lens + align zone",
              _GDN_FIELD, _GDN_FIELD +
              "    # suffix SUFFIX_ROCM_GDN_DEFER: positions of the spec rows, align block size\n"
              "    suffix_spec_seq_lens: torch.Tensor | None = None\n"
              "    suffix_zone: int = 0\n"
              "    suffix_max_prefill: int = 0  # longest prefill chunk (host)\n"),
        Patch("vllm.v1.attention.backends.gdn_attn", "GDN metadata: spec seq_lens (ctor)",
              _GDN_CTOR, _GDN_CTOR +
              "            suffix_spec_seq_lens=(  # suffix SUFFIX_ROCM_GDN_DEFER\n"
              "                None if spec_sequence_masks is None\n"
              "                else m.seq_lens[:batch_size] if spec_req_idx is None\n"
              "                else m.seq_lens[spec_req_idx]\n"
              "            ),\n"
              "            suffix_zone=(\n"
              "                self.kv_cache_spec.block_size\n"
              "                if self.vllm_config.cache_config.mamba_cache_mode == \"align\"\n"
              "                else 0\n"
              "            ),\n"
              "            suffix_max_prefill=(\n"
              "                int(prefill_query_start_loc_cpu.diff().max()) if num_prefills > 0 else 0\n"
              "            ),\n"),
        Patch("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
              "GDN spec verify (generic path) with deferred state commit",
              "        if spec_sequence_masks is not None:\n"
              "            core_attn_out_spec, last_recurrent_state = (\n"
              "                fused_sigmoid_gating_delta_rule_update(\n"
              "                    A_log=self.A_log,\n"
              "                    a=a_spec,\n"
              "                    b=b_spec,\n"
              "                    dt_bias=self.dt_bias,\n"
              "                    q=query_spec,\n"
              "                    k=key_spec,\n"
              "                    v=value_spec,\n"
              "                    initial_state=ssm_state,\n"
              "                    inplace_final_state=True,\n"
              "                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]\n"
              "                        : attn_metadata.num_spec_decodes\n"
              "                        + 1  # type: ignore[attr-defined]\n"
              "                    ],\n"
              "                    ssm_state_indices=spec_state_indices_tensor,\n"
              "                    num_accepted_tokens=num_accepted_tokens,\n"
              "                    use_qk_l2norm_in_kernel=True,\n"
              "                )\n"
              "            )\n",
              "        if spec_sequence_masks is not None:  # suffix SUFFIX_ROCM_GDN_DEFER\n"
              "            core_attn_out_spec = _suffix_gdn_defer_spec(\n"
              "                self, mixed_qkv_spec, a_spec, b_spec, ssm_state, attn_metadata\n"
              "            )\n"
              "            last_recurrent_state = None\n",
              "suffix_hybrid.kernels.gdn_defer_rocm:install"),
        Patch("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
              "GDN spec verify (generic path): no q/k/v repack for the deferred kernel",
              "        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)\n",
              "        # suffix SUFFIX_ROCM_GDN_DEFER: the verify reads the packed mixed_qkv_spec\n"
              "        query_spec = key_spec = value_spec = None\n"),
    ),
    "SUFFIX_ROCM_HC_DOWN": (
        Patch(_HC, "HC down projection split-K at small M (mix)",
              "            hidden_states,\n" + _HC_DOWN,
              "            hidden_states,\n" + _HC_DOWN_NEW,
              "suffix_hybrid.kernels.hc_down_rocm:install"),
        Patch(_HC, "HC down projection split-K at small M (combine_and_mix)",
              "            prev_injection,\n" + _HC_DOWN,
              "            prev_injection,\n" + _HC_DOWN_NEW),
    ),
    "SUFFIX_ROCM_HC_BIG": (
        Patch(_HC, "HC site tail in one op, 4 launches at decode M (mix)", _HC_MIX_SIG,
              _HC_MIX_SIG + "        return hc_big_mix(self, hidden_states)  # suffix rocm-hc-big\n",
              "suffix_hybrid.kernels.hc_big_rocm:install"),
        Patch(_HC, "HC site tail in one op, 4 launches at decode M (combine_and_mix)", _HC_CAM_DOC,
              _HC_CAM_DOC + "        return hc_big_combine_and_mix(  # suffix rocm-hc-big\n"
              "            self, hidden_states, prev_block_output, prev_injection\n"
              "        )\n"),
    ),
    "SUFFIX_ROCM_QK_FUSED": Patch(
        "vllm.models.qwen4_exp.amd.qsa",
        "QSA split + QK norm + mRoPE + gate in vLLM's fused Triton kernel on ROCm",
        "            and current_platform.is_cuda()\n"
        "            and text_only\n"
        "        )\n",
        "            and current_platform.is_cuda_alike()  # suffix SUFFIX_ROCM_QK_FUSED\n"
        "            and getattr(self.rotary_emb, \"dtype\", None) in (torch.float16, torch.bfloat16)\n"
        "            and (text_only or (  # Qwen3NextAttention's interleaved-mRoPE condition\n"
        "                type(self.rotary_emb).__name__ == \"MRotaryEmbedding\"\n"
        "                and getattr(self.rotary_emb, \"mrope_interleaved\", False)\n"
        "                and len(getattr(self.rotary_emb, \"mrope_section\", None) or ()) == 3\n"
        "                and sum(self.rotary_emb.mrope_section) == self.rotary_emb.rotary_dim // 2))\n"
        "        )\n"),
    "SUFFIX_ROCM_ACT_QUANT_FUSE": (
        Patch("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
              "GDN RMSNormGated + MXFP4 quant in one launch before out_proj",
              _AQ_GDN,
              "        if _suffix_aq_gdn_ok(self):  # suffix SUFFIX_ROCM_ACT_QUANT_FUSE\n"
              "            return _suffix_aq_gdn_out(self, core_attn_out, z)\n" + _AQ_GDN,
              "suffix_hybrid.kernels.act_quant_rocm:install_gdn"),
        Patch("vllm.models.qwen4_exp.amd.qsa",
              "QSA sigmoid gate + MXFP4 quant in one launch before o_proj",
              _AQ_QSA,
              "        flat_output = attn_output.view(num_tokens, -1)\n"
              "        if gate is not None and _suffix_aq_qsa_ok(self):  # suffix SUFFIX_ROCM_ACT_QUANT_FUSE\n"
              "            return _suffix_aq_qsa_out(self, flat_output, gate, qkv)\n"
              + _AQ_QSA.split("\n", 1)[1],
              "suffix_hybrid.kernels.act_quant_rocm:install_qsa"),
    ),
    "SUFFIX_JIT_LOG": Patch("triton.runtime.jit", "log every Triton compile",
                            after="suffix_hybrid.jit_log:install"),
    # Inside the lru-cached lookup: one file probe per (shape, M) per process; a plugin
    # miss (None) falls through to AITER's own probe unchanged.
    "SUFFIX_ROCM_AFP4_CONFIGS": Patch(
        "aiter.ops.triton.utils.gemm_config_utils",
        f"AITER GEMM-AFP4WFP4 configs from {AFP4_DIR} first "
        f"({len(glob.glob(os.path.join(AFP4_DIR, 'GEMM-AFP4WFP4-N=*-K=*.json')))} shapes)",
        _AFP4_PROBE,
        "    for suffix in specialized_suffixes:\n"
        "        specialized_config = (  # suffix SUFFIX_ROCM_AFP4_CONFIGS: plugin JSON first\n"
        '            config_name == "GEMM-AFP4WFP4" and backend == "triton" and load_config_json(\n'
        f'                {AFP4_DIR!r} f"/{{config_name}}-{{suffix}}.json", required=False)\n'
        "        ) or load_config_json(\n"
        '            f"{cfg_dir}/{config_name}-{suffix}.json", required=False\n'
        "        )\n"),
    "SUFFIX_ROCM_MOE_ROUTE": (
        Patch("aiter.fused_moe", "MoE top-k + sort + MXFP4 quant-sort in two launches (sort site)",
              after="suffix_hybrid.kernels.moe_route_rocm:install_aiter"),
        Patch("vllm.model_executor.layers.fused_moe.router.aiter_shared_routed_fused_moe_router",
              "MoE top-k deferred to the fused sort (router)",
              after="suffix_hybrid.kernels.moe_route_rocm:install_router"),
    ),
}
_MARK = "_suffix_rocm_patch"


def enabled() -> dict:
    """Target module -> [Patch] for every gate set to 1, in PATCHES order."""
    todo: dict = {}
    for gate, patches in PATCHES.items():
        if os.environ.get(gate, "").strip() == "1":
            for p in (patches,) if isinstance(patches, Patch) else patches:
                todo.setdefault(p.target, []).append(p)
    return todo


def patch_source(patch: Patch, src: str) -> str:
    n = src.count(patch.old)
    if n != 1:
        raise RuntimeError(f"[suffix rocm-patch] anchor found {n}x in {patch.target} "
                           "(expected 1): vLLM drifted, refusing to serve unpatched")
    return src.replace(patch.old, patch.new)


def install_post_import_hook() -> bool:
    todo = enabled()
    if not todo:
        return False
    early = sorted(t for t in todo if t in sys.modules)
    if early:
        raise SystemExit(f"[suffix rocm-patch] {early} imported before the hook armed")
    if any(getattr(f, _MARK, False) for f in sys.meta_path):
        return True

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
            patches = todo.pop(fullname, None)
            if patches is None:
                return None
            # importlib.util.find_spec walks sys.meta_path: step aside meanwhile,
            # for good once every target is armed.
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if todo:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            def exec_module(module):
                src = loader.get_source(fullname)
                for p in patches:
                    if p.old:
                        src = patch_source(p, src)
                # Triton's @jit parses inspect.getsource(fn), which reads the file named
                # by co_filename: without this the patched module's kernels compile from
                # the unpatched file on disk (wrong lines once a patch shifts them).
                fname = f"{module.__file__}.suffix-rocm-patch.py"
                linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
                exec(compile(src, fname, "exec"), module.__dict__)
                for p in patches:
                    if p.after:
                        mod_name, fn = p.after.split(":")
                        getattr(importlib.import_module(mod_name), fn)(module)
                    print(f"[suffix rocm-patch] ACTIVE: {p.label} ({fullname})",
                          file=sys.stderr, flush=True)
                setattr(module, _MARK, True)

            loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
    return True
