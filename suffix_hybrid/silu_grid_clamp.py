# SUFFIX_SILU_GRID_CLAMP: NVIDIA DeepGEMM MoE (SM120 fp8 block).
#
# The fused SiLU*mul+FP8-quant wrapper (silu_mul_quant_fp8_packed_triton) sizes
# its launch grid from the PADDED grouped workspace rows M_sum (>= 64 experts x
# 128 alignment) instead of the token count: the expert branch launches
# (grid_n, grid_m, num_experts) = up to (3, 32, 64) = 6144 CTAs, whose
# per-expert row-blocks are mostly empty (each expert holds at most one row per
# token with distinct top-k). The kernel is a persistent grid-stride loop over
# row-blocks, so ANY grid_m >= 1 processes every live row exactly once:
# clamping grid_m to cdiv(tokens, BM) keeps full per-expert parallelism and
# drops the empty CTAs. Bit-identical outputs (same live rows, same math;
# only the CTA -> row-block partition changes).
#
# Profile 20261016T111728Z-e038 (lfm2.5-24b-a2b-fp8, RTX 5090): 2.15 / 9.03 /
# 31.3 us per call at c1 / c8 / c32 against ~25 KB / 0.2 / 0.87 MB of live
# bytes -> 28 GB/s effective at c32 (pure launch/metadata cost), 21.55% of the
# c32 decode step, 7.24% at c8, 2.29% at c1. Prefill is untouched: at 512
# live tokens the clamp is inactive (cdiv(512, 4) = 128 >= stock grid_m).

_SGC_SIG_OLD = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    "    *,\n"
    "    expert_ends: torch.Tensor | None = None,\n"
    "    expert_alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n")
_SGC_SIG_NEW = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    "    *,\n"
    "    expert_ends: torch.Tensor | None = None,\n"
    "    expert_alignment: int = 0,\n"
    "    live_rows: int | None = None,  # suffix SUFFIX_SILU_GRID_CLAMP\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n")

_SGC_GRID_OLD = (
    "        grid_m = min(\n"
    "            triton.cdiv(M, BM * num_experts),\n"
    "            max(1, triton.cdiv(max_ctas, grid_n * num_experts)),\n"
    "        )\n"
    "        grid = (grid_n, grid_m, num_experts)\n")
_SGC_GRID_NEW = (
    "        grid_m = min(\n"
    "            triton.cdiv(M, BM * num_experts),\n"
    "            max(1, triton.cdiv(max_ctas, grid_n * num_experts)),\n"
    "        )\n"
    "        if live_rows is not None:  # suffix SUFFIX_SILU_GRID_CLAMP\n"
    "            # Each expert holds <= live_rows rows (one per token at most\n"
    "            # with distinct top-k), so cdiv(live_rows, BM) row-blocks per\n"
    "            # expert is full parallelism; the persistent loop keeps every\n"
    "            # live row processed exactly once.\n"
    "            grid_m = max(1, min(triton.cdiv(live_rows, BM), grid_m))\n"
    "        grid = (grid_n, grid_m, num_experts)\n")

_SGC_DEF_OLD = (
    "    def _act_mul_quant(\n"
    "        self,\n"
    "        input: torch.Tensor,\n"
    "        output: torch.Tensor,\n"
    "        activation: MoEActivation,\n"
    "        expert_ends: torch.Tensor | None = None,\n"
    "        expert_alignment: int = 0,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n")
_SGC_DEF_NEW = (
    "    def _act_mul_quant(\n"
    "        self,\n"
    "        input: torch.Tensor,\n"
    "        output: torch.Tensor,\n"
    "        activation: MoEActivation,\n"
    "        expert_ends: torch.Tensor | None = None,\n"
    "        expert_alignment: int = 0,\n"
    "        num_tokens: int | None = None,  # suffix SUFFIX_SILU_GRID_CLAMP\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n")

_SGC_FUSED_OLD = (
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                    alpha=self.gemm1_alpha,\n"
    "                    beta=self.gemm1_beta,\n"
    "                    expert_ends=expert_ends,\n"
    "                    expert_alignment=expert_alignment,\n"
    "                )\n")
_SGC_FUSED_NEW = (
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                    alpha=self.gemm1_alpha,\n"
    "                    beta=self.gemm1_beta,\n"
    "                    expert_ends=expert_ends,\n"
    "                    expert_alignment=expert_alignment,\n"
    "                    live_rows=num_tokens,  # suffix SUFFIX_SILU_GRID_CLAMP\n"
    "                )\n")

_SGC_CALL_OLD = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N),\n"
    "                output=quant_out,\n"
    "                activation=activation,\n"
    "                expert_ends=grouped_layout if use_psum_layout else None,\n"
    "                expert_alignment=align_used if use_psum_layout else 0,\n"
    "            )\n")
_SGC_CALL_NEW = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N),\n"
    "                output=quant_out,\n"
    "                activation=activation,\n"
    "                expert_ends=grouped_layout if use_psum_layout else None,\n"
    "                expert_alignment=align_used if use_psum_layout else 0,\n"
    "                num_tokens=topk_ids.size(0),  # suffix SUFFIX_SILU_GRID_CLAMP\n"
    "            )\n")
