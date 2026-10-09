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
"""
import importlib.util
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
    "SUFFIX_ROCM_QSA_SPARSE_SKIP": Patch(
        "vllm.models.qwen4_exp.amd.ops.qsa",
        "QSA sparse attention skips all-padding tiles",
        _QSA_SPARSE_BODY,
        "        if tl.max(logical_token, axis=0) >= 0:  # suffix: an all -1 tile is a no-op\n"
        + textwrap.indent(_QSA_SPARSE_BODY, "    ")),
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
                exec(compile(src, module.__file__, "exec"), module.__dict__)
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
