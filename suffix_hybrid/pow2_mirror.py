# SUFFIX_POW2_MIRROR: value-level mirror of the baseline's DeepGEMM E8M0
# quantization on the VLLM_USE_DEEP_GEMM=0 arm (RTX 5090, lfm2.5-24b-a2b-fp8).
#
# WHY: ad7b (env-only VLLM_USE_DEEP_GEMM=0) won +9.5/+18.6/+21.9% tok/s over
# the baseline but drift-FAILed (de KL 0.1182, ppl ratio 0.9587 - LOWER
# perplexity than the baseline), repeat-confirmed by 4e29 (identical KL). Two
# candidate causes: (a) the arm's quant values differ from the baseline's
# (DG=0 turns the E8M0 oracle off, so activations AND weights quantize with
# unceiled FP32 per-block scales while the baseline uses pow2-CEILED scales
# everywhere), or (b) the GEMM backends themselves (CUTLASS/Triton instead of
# DeepGEMM) produce the drift. This gate isolates (a): it forces the SAME
# pow2-ceiled scale VALUES the baseline produces at every quant/requant site,
# while the GEMM backends stay the DG=0 arm's. Drift collapses toward zero =>
# (a) scale rounding was the drift; drift unchanged (~0.118 de KL) => (b) the
# backend is, and ad7b's drift is intrinsic to CUTLASS/Triton-vs-DeepGEMM math.
#
# MECHANISM (4 patches, 3 modules; the arm's own sites need no patch):
#   P1 vllm.utils.deep_gemm.is_deep_gemm_e8m0_used: return True when
#      SUFFIX_POW2_MIRROR=1 even though is_deep_gemm_supported() is False
#      (DG=0). Every oracle-deferring site (use_ue8m0=None) then quantizes
#      with ceiled scales: MoE act1 via moe_kernel_quantize_input, and MoE
#      act2 via triton_moe.py's "not is_deep_gemm_e8m0_used()" branch, which
#      now takes the UNFUSED silu_mul + per_token_group_quant_fp8 path (the
#      fused helion op at :510 has no ue8m0 knob and stays off, exactly like
#      the baseline). Blast radius on this arm: only the quant sites - the
#      packed-scale path stays unreachable (QuantFP8 packs only when
#      is_deep_gemm_supported() AND ue8m0; deepgemm_post_process_* runs only
#      from the DeepGemm kernel / DEEPGEMM-family MoE backends, pruned from
#      selection under DG=0; flashinfer + cutlass is_supported never consult
#      e8m0). The flip is env-checked INSIDE the cached function, so the
#      patched source is inert without SUFFIX_POW2_MIRROR=1.
#   P2 cutlass.py CutlassFp8BlockScaledMMKernel.__init__: build
#      QuantFP8(use_ue8m0=True) - the dense activation quant then produces
#      the baseline's ceiled column-major FP32 scales (deep_gemm.py's kernel
#      builds use_ue8m0=oracle=True, column_major_scales=True; only the
#      value-changing flag differs from stock CUTLASS here).
#   P3 cutlass.py: ADD process_weights_after_loading - the CUTLASS kernel
#      inherits the base's pad-only processing; the baseline's DeepGemm kernel
#      does pad -> requant_weight_ue8m0_inplace -> deepgemm packing. The
#      mirror runs the SAME requant (same function, same pad-first order) and
#      skips only the deepgemm scale packing: ceiled values in the FP32
#      layout CUTLASS consumes.
#   P4 fp8.py Fp8MoEMethod.process_weights_after_loading: requant w13/w2
#      in place (same requant_weight_ue8m0_inplace, same position: post-shard,
#      pre-_setup_kernel) before the runtime shuffle.
#
# EXACTNESS: value-level mirror of the baseline's quantization - same quant
# ops, same flags, same order; the only differences are the GEMM backends and
# the physical scale container (ceiled FP32 instead of deepgemm's packed
# layout). Scale VALUES are bit-equal to the baseline's pre-packing scales.
# The 5%-relative quality gate + drift check decide like any trial.
#
# ENGAGEMENT MARKERS (grep the trial log): "[suffix rocm-patch] ACTIVE" x4
# plus "DeepGEMM E8M0 scale-format MIRRORED" from P1's info_once. Refuted if:
# (a) any anchor count != 1 at boot (fail-closed refuses to serve), or
# (b) speed loses ad7b's gains, or (c) drift is NOT materially reduced toward
# the baseline while quality holds - that leaves the GEMM backend as the
# drift's cause and ad7b's fork open for the owner's full A/B gate.
#
# NOTE: sitecustomize.py's arming tuple must list SUFFIX_POW2_MIRROR (the
# SUFFIX_SILU_SKIP_PAD gate was registered in rocm_patches.py but never armed
# there, so trial 481f ran stock - see learning 20261016T200732Z-03087b).

_P2M_ORACLE_OLD = (
    "    if not is_deep_gemm_supported():\n"
    "        logger.debug_once(\n"
    "            \"DeepGEMM E8M0 disabled: DeepGEMM not supported on this system.\"\n"
    "        )\n"
    "        return False\n"
)

_P2M_ORACLE_NEW = (
    "    if not is_deep_gemm_supported():\n"
    "        import os  # suffix SUFFIX_POW2_MIRROR\n"
    "        if os.environ.get(\"SUFFIX_POW2_MIRROR\") == \"1\":\n"
    "            logger.info_once(\n"
    "                \"DeepGEMM E8M0 scale-format MIRRORED (SUFFIX_POW2_MIRROR=1): \"\n"
    "                \"E8M0 quant values active while DeepGEMM stays unselected.\"\n"
    "            )\n"
    "            return True\n"
    "        logger.debug_once(\n"
    "            \"DeepGEMM E8M0 disabled: DeepGEMM not supported on this system.\"\n"
    "        )\n"
    "        return False\n"
)

_P2M_ACT_OLD = (
    "        self.quant_fp8 = QuantFP8(\n"
    "            static=act_scale_descriptor.static,\n"
    "            group_shape=act_scale_descriptor.group_shape,\n"
    "            num_token_padding=self.get_output_padding(),\n"
    "            use_ue8m0=False,\n"
    "            column_major_scales=True,\n"
    "        )\n"
)

_P2M_ACT_NEW = (
    "        self.quant_fp8 = QuantFP8(\n"
    "            static=act_scale_descriptor.static,\n"
    "            group_shape=act_scale_descriptor.group_shape,\n"
    "            num_token_padding=self.get_output_padding(),\n"
    "            use_ue8m0=True,  # suffix SUFFIX_POW2_MIRROR: baseline's ceiled act scales\n"
    "            column_major_scales=True,\n"
    "        )\n"
)

_P2M_WDEF_OLD = (
    "    def apply_block_scaled_mm(\n"
    "        self,\n"
    "        A: torch.Tensor,\n"
)

_P2M_WDEF_NEW = (
    "    def process_weights_after_loading(self, layer):\n"
    "        super().process_weights_after_loading(layer)\n"
    "        import os  # suffix SUFFIX_POW2_MIRROR\n"
    "        if os.environ.get(\"SUFFIX_POW2_MIRROR\") != \"1\":\n"
    "            return\n"
    "        # suffix SUFFIX_POW2_MIRROR: pow2-ceiled weight scales - the same\n"
    "        # requant_weight_ue8m0_inplace the baseline's DeepGEMM kernel runs\n"
    "        # after the same pad-first order - kept in the FP32 layout CUTLASS\n"
    "        # consumes (no deepgemm scale packing).\n"
    "        params = self._get_layer_params(layer)\n"
    "        from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
    "            requant_weight_ue8m0_inplace,\n"
    "        )\n"
    "        requant_weight_ue8m0_inplace(\n"
    "            params.weight,\n"
    "            params.block_scale,\n"
    "            block_size=tuple(layer.weight_block_size),\n"
    "        )\n"
    "\n"
    "    def apply_block_scaled_mm(\n"
    "        self,\n"
    "        A: torch.Tensor,\n"
)

_P2M_MOE_OLD = (
    "        # Shuffle weights to runtime format and setup kernel.\n"
    "        self._setup_kernel(\n"
    "            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale\n"
    "        )\n"
)

_P2M_MOE_NEW = (
    "        if self.block_quant:\n"
    "            import os  # suffix SUFFIX_POW2_MIRROR\n"
    "            if os.environ.get(\"SUFFIX_POW2_MIRROR\") == \"1\":\n"
    "                # suffix SUFFIX_POW2_MIRROR: pow2-ceiled expert weight scales\n"
    "                # (the baseline's DeepGEMM E8M0 requant) before the runtime\n"
    "                # shuffle; FP32 layout, no deepgemm scale packing.\n"
    "                from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
    "                    requant_weight_ue8m0_inplace,\n"
    "                )\n"
    "                requant_weight_ue8m0_inplace(\n"
    "                    w13, w13_scale, block_size=tuple(self.moe_block_shape)\n"
    "                )\n"
    "                requant_weight_ue8m0_inplace(\n"
    "                    w2, w2_scale, block_size=tuple(self.moe_block_shape)\n"
    "                )\n"
    "        # Shuffle weights to runtime format and setup kernel.\n"
    "        self._setup_kernel(\n"
    "            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale\n"
    "        )\n"
)
