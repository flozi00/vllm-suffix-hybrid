# SUFFIX_SILU_SKIP_PAD: NVIDIA DeepGEMM MoE (SM120 fp8 block), v3 - re-anchored
# for vLLM v0.31.0 by wake 80 (v2 dd422a00 anchors were nightly-2eb491e18b:
# that kernel still had the expert/psum branch + expert_ends/EXPERT_ALIGNMENT
# args; v0.31.0's kernel is the dense-only persistent one, so 9/10 anchors
# drifted and the fail-closed check refused to boot - trials 25d3/2125 DOA).
#
# Mechanism (unchanged): deepgemm_moe_permute returns expert_ids = m_indices,
# (M_sum,) int32, -1 on padding rows (vLLM v0.31.0 deep_gemm_utils.py:517 "we
# always initialize expert_ids to -1 so any row that is not explicitly written
# by the scatter kernel will be treated as invalid and skipped by DeepGEMM's
# scheduler"). DeepGemmExperts.apply() runs the fused SiLU*mul+FP8-quant over
# the PADDED grouped workspace (M_sum rows for M*num_topk live rows): the
# kernel stores q+scale for every row including m_indices<0 padding. At c32
# (e038 profile, v0.30.0-based, 56debb caveat): 31.3 us/call, 38 calls, 21.55%
# of the decode step. Thread expert_ids in as row_experts and mask the dense
# loop: live rows keep identical math (bit-identical), padding rows go stale -
# the standing workspace contract (GEMM2 skips m_indices<0 blocks; the
# unpermute reads live rows only, exactly as it already does for mm1_out).
#
# v0.31.0 shape notes: silu_mul_quant_fp8_packed_triton lives in
# vllm/model_executor/layers/quantization/utils/fp8_utils.py (moved from
# quantization/fp8_utils.py); the MoE runner is
# vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py (moved from
# fused_moe/deep_gemm_moe.py). The CALL anchor spans to the mm2 launch line
# because DeepGemmFP4Experts._act_mul_quant's call site (line ~638) is
# textually identical otherwise. The kernel has no expert branch anymore, so
# HAS_ROW_EXPERTS=False (all stock callers pass row_experts=None) compiles the
# mask out; Triton accepts the None arg exactly as 2eb's expert_ends=None did.
# First engaged call prints "[suffix SUFFIX_SILU_SKIP_PAD] engaged: M=... live=
# ..." once (eager warmup precedes cudagraph capture, so the one .item() sync
# is safe); grep the trial log for it as the engagement marker.

_SSP_SIG_OLD = (
    "def _silu_mul_quant_fp8_packed_kernel(\n"
    "    input_ptr,\n"
    "    output_q_ptr,\n"
    "    output_scale_ptr,\n"
    "    M,\n"
)

_SSP_SIG_NEW = (
    "def _silu_mul_quant_fp8_packed_kernel(\n"
    "    input_ptr,\n"
    "    output_q_ptr,\n"
    "    output_scale_ptr,\n"
    "    row_experts_ptr,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "    M,\n"
)

_SSP_CE_OLD = (
    "    PACKS_PER_CTA: tl.constexpr,\n"
    "    BLOCK_M: tl.constexpr,\n"
    "    HAS_CLAMP: tl.constexpr,\n"
    "):\n"
)

_SSP_CE_NEW = (
    "    PACKS_PER_CTA: tl.constexpr,\n"
    "    BLOCK_M: tl.constexpr,\n"
    "    HAS_CLAMP: tl.constexpr,\n"
    "    HAS_ROW_EXPERTS: tl.constexpr,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "):\n"
)

_SSP_MASK_OLD = (
    "    while row_start < M:\n"
    "        rows = row_start + row_offsets\n"
    "        row_mask = rows < M\n"
)

_SSP_MASK_NEW = (
    "    while row_start < M:\n"
    "        rows = row_start + row_offsets\n"
    "        row_mask = rows < M\n"
    "        if HAS_ROW_EXPERTS:  # suffix SUFFIX_SILU_SKIP_PAD: m_indices<0 rows are padding\n"
    "            row_mask &= tl.load(\n"
    "                row_experts_ptr + rows, mask=rows < M, other=-1\n"
    "            ) >= 0\n"
)

_SSP_WRAP_OLD = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

_SSP_WRAP_NEW = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    "    row_experts: torch.Tensor | None = None,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

# v0.31.0 has no expert/psum else-branch: the validation + engagement marker
# slot in between the grid computation and the launch.
_SSP_ELSE_OLD = (
    "    grid_n = triton.cdiv(packs_per_row, packs_per_cta)\n"
    "    grid_m = min(triton.cdiv(M, BM), 4096)\n"
    "    grid = (grid_n, grid_m)\n"
    "\n"
    "    has_clamp = clamp_limit is not None\n"
)

_SSP_ELSE_NEW = (
    "    grid_n = triton.cdiv(packs_per_row, packs_per_cta)\n"
    "    grid_m = min(triton.cdiv(M, BM), 4096)\n"
    "    grid = (grid_n, grid_m)\n"
    "\n"
    "    if row_experts is not None:  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "        assert row_experts.dim() == 1 and row_experts.numel() == M\n"
    "        assert row_experts.dtype == torch.int32 and row_experts.is_contiguous()\n"
    "        assert row_experts.device == input.device\n"
    "        if not getattr(silu_mul_quant_fp8_packed_triton, '_ssp_seen', False):\n"
    "            silu_mul_quant_fp8_packed_triton._ssp_seen = True\n"
    "            print('[suffix SUFFIX_SILU_SKIP_PAD] engaged: M=', M,\n"
    "                  ' live=', (row_experts >= 0).sum().item(), flush=True)\n"
    "\n"
    "    has_clamp = clamp_limit is not None\n"
)

_SSP_LAUNCH_OLD = (
    "    _silu_mul_quant_fp8_packed_kernel[grid](\n"
    "        input,\n"
    "        output_q,\n"
    "        output_scale_packed,\n"
    "        M,\n"
)

_SSP_LAUNCH_NEW = (
    "    _silu_mul_quant_fp8_packed_kernel[grid](\n"
    "        input,\n"
    "        output_q,\n"
    "        output_scale_packed,\n"
    "        row_experts,  # suffix SUFFIX_SILU_SKIP_PAD (None for stock callers)\n"
    "        M,\n"
)

_SSP_LCONST_OLD = (
    "        BLOCK_M=BM,\n"
    "        HAS_CLAMP=has_clamp,\n"
    "        num_warps=num_warps,\n"
    "        num_stages=num_stages,\n"
    "    )\n"
)

_SSP_LCONST_NEW = (
    "        BLOCK_M=BM,\n"
    "        HAS_CLAMP=has_clamp,\n"
    "        HAS_ROW_EXPERTS=row_experts is not None,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "        num_warps=num_warps,\n"
    "        num_stages=num_stages,\n"
    "    )\n"
)

# Class-1 (fp8-block) _act_mul_quant only: the trailing assert line
# disambiguates from DeepGemmFP4Experts._act_mul_quant (line ~517, followed
# by "block_k = self._ACT_BLOCK_K").
_SSP_DEF_OLD = (
    "    def _act_mul_quant(\n"
    "        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        assert self.block_shape is not None\n"
)

_SSP_DEF_NEW = (
    "    def _act_mul_quant(\n"
    "        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation,\n"
    "        row_experts: torch.Tensor | None = None,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        assert self.block_shape is not None\n"
)

# Class-1 fused call only: alpha/beta kwargs are absent from the FP4 class.
_SSP_FUSED_OLD = (
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                    alpha=self.gemm1_alpha,\n"
    "                    beta=self.gemm1_beta,\n"
    "                )\n"
)

_SSP_FUSED_NEW = (
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                    alpha=self.gemm1_alpha,\n"
    "                    beta=self.gemm1_beta,\n"
    "                    row_experts=row_experts,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "                )\n"
)

# Class-1 apply() call site only: spans to the mm2 launch (the FP4 class calls
# m_grouped_fp8_fp4_gemm_nt_contiguous there instead).
_SSP_CALL_OLD = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N), output=quant_out, activation=activation\n"
    "            )\n"
    "\n"
    "            mm2_out = _resize_cache(workspace2, (M_sum, K))\n"
    "            m_grouped_fp8_gemm_nt_contiguous(\n"
)

_SSP_CALL_NEW = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N), output=quant_out, activation=activation,\n"
    "                row_experts=expert_ids,  # suffix SUFFIX_SILU_SKIP_PAD\n"
    "            )\n"
    "\n"
    "            mm2_out = _resize_cache(workspace2, (M_sum, K))\n"
    "            m_grouped_fp8_gemm_nt_contiguous(\n"
)
