# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.32.0 adapter for the existing, oracle-gated Gemma K2 lane.

Upstream now supports SM120 NVFP4 itself. Keep its dtype/view/forward code;
port only our image masks, graph-safe K2 dispatch and host planning changes.
The cache writer remains vLLM's C++ writer: its V scales are swizzled, as
required by both oxide and the tuned Helion kernels. The new FlashInfer
slot writer uses linear V scales and MUST NOT be mixed with these readers.
Consequently this adapter uses FA2 (with the existing deswizzle define),
not XQA. A separate native-format port needs its own numerical gate.
"""
from __future__ import annotations


def guard_gather_state(gather):
    """Keep K2's old invariant: every real prefill stays out of FULL graphs.

    v0.32 also admits one-token prompt extends with placeholder drafts into
    decode graphs. Keep their batch classification, but use piecewise graphs
    so the image-aware metadata builder can choose its exact mask.
    """
    import functools

    @functools.wraps(gather)
    def guarded(*args, **kwargs):
        state, uniform = gather(*args, **kwargs)
        if state is not None and state.has_prefill:
            return state._replace(decode_graph_eligible=False), None
        return state, uniform

    guarded._suffix_k2_prefill_graph_guard = True
    return guarded


def install_prefill_graph_guard():
    import hashlib
    import inspect
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        gather = GPUModelRunner.gather_batch_req_state
        if getattr(gather, "_suffix_k2_prefill_graph_guard", False):
            return True
        # Exact published v0.32.0 function. Unknown request-state/control-flow
        # contracts keep the conservative single-token support tier.
        source = inspect.getsource(gather).rstrip() + "\n"
        if hashlib.sha256(source.encode()).hexdigest() != (
            "18da780f3c3ffdea5e38b6d663083792796b92e8bfe2159ae16b76d0af9db7ca"
        ):
            return False
        GPUModelRunner.gather_batch_req_state = guard_gather_state(gather)
        return True
    except (ImportError, AttributeError, OSError, TypeError):
        return False


def probe_headers(files, probe):
    # FI 0.7 also lets a custom attention variant own the mask. Our wrapper
    # supplies _custom_mask_buf, so the additional OR preserves its dispatch.
    # Normalize only this exact, audited condition; every other probe stays.
    files = dict(files)
    files["prefill.py"] = files.get("prefill.py", "").replace(
        "if self._custom_mask_buf is not None or self._variant_owns_mask:\n"
        "            mask_mode = MaskMode.CUSTOM.value",
        "if self._custom_mask_buf is not None:\n"
        "            mask_mode = MaskMode.CUSTOM.value")
    return probe(files)


def edits(base):
    keep = {
        "supports_mm_prefix_nvfp4", "nvfp4_head_major_layouts", "helper_after_logger",
        "impl_route_flag", "cudagraph_support_fa2", "own_attn_no_paged_indices",
        "hostfree_pinned_buffers", "hostfree_no_kv_lens_pin", "builder_mm_prefix_mode",
        "build_mm_prefix_resplit", "prefill_mm_prefix_mask", "own_attn_spec_as_decode",
        "own_attn_decode_wrapper",
    }
    result = [e for e in base._BACKEND_EDITS if e[0] in keep]
    result.extend([
        ("v032_builder_route",
         "        # Compute per-phase Q dtype.  On SM90 (XQA decode), the prefill and",
         '''        self.use_fa2_nvfp4_kv = (
            self.nvfp4_fa2 and _use_fa2_for_nvfp4_kv_on_sm120()
        )
        if self.use_fa2_nvfp4_kv:
            if self.use_dcp or not _nvfp4_kv_head_dim_ok(self.head_dim):
                raise ValueError("suffix K2: unsupported DCP or head dimension")
            if get_flashinfer_layout_string(self.kv_cache_layout) != "HND":
                raise ValueError("suffix K2: swizzled V scales require HND pages")

        # Compute per-phase Q dtype.  On SM90 (XQA decode), the prefill and''', 1),
        ("v032_swizzled_cache_no_xqa",
         "        self.use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode",
         '''        if self.use_fa2_nvfp4_kv:
            can_use_xqa_or_trtllm_gen_decode = False
        self.use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode''', 1),
        ("v032_keep_cpp_cache_writer",
         '''            if self.nvfp4_fa2:
                data, scales, kv_layout = _nvfp4_kv_views(kv_cache, self.num_kv_heads)''',
         '''            if self.nvfp4_fa2 and not self.use_fa2_nvfp4_kv:
                data, scales, kv_layout = _nvfp4_kv_views(kv_cache, self.num_kv_heads)''', 1),
        ("v032_ragged_k2_split",
         "                        self.use_xqa or self.use_trtllm_gen_varlen_decode",
         '''                        self.use_xqa or self.use_trtllm_gen_varlen_decode
                        or getattr(self, "use_own_nvfp4_attn", False)''', 1),
        ("v032_k2_no_seq_lens_sync",
         "        needs_seq_lens_cpu = self.use_dcp or not all_uses_trtllm",
         '''        needs_seq_lens_cpu = self.use_dcp or not all_uses_trtllm
        if (needs_seq_lens_cpu and num_prefills == 0 and not self.use_dcp
                and getattr(self, "use_own_nvfp4_attn", False)
                and _nvfp4_own_attn_takes(qo_indptr_cpu, num_decodes)):
            needs_seq_lens_cpu = False''', 1),
        ("v032_hostfree_build_fence", "    def build(\n",
         "    @_fi_hostfree_build\n    def build(\n", 1),
    ])
    return result


def patch_backend_source(src, base):
    transforms = edits(base)
    bad = [f"{name}: expected {count}, found {src.count(old)}"
           for name, old, _new, count in transforms if src.count(old) != count]
    if bad:
        raise base.PatchDriftError("vLLM 0.32.0 K2 adapter anchor drift: " + "; ".join(bad))
    out, applied = src, []
    for name, old, new, count in transforms:
        if out.count(old) != count:
            raise base.PatchDriftError(f"vLLM 0.32.0 overlapping anchor: {name}")
        out = out.replace(old, new)
        applied.append(name)
    compile(out, "<nvfp4_kv_patch_v032>", "exec")
    return out, applied
