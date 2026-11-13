"""Exact verify and commit kernels for the GLM RecoverSSM wiring (glm_recoverssm.py): bit-identical to the stock spec snapshots of `fused_recurrent_kda`.

Upstream's RecoverSSM kernels (Kimi-K3) equal the stock snapshots only up to fp32 rounding (rsqrt instead of sqrt-and-divide, sigmoid instead of the divided gate form, another tile
and reduction tree, a reverse-accumulation fold). GLM-5.3-Flash is chaotic enough (288 experts, top-8, NVFP4 activations) that such rounding flips expert choices and moves logprobs by
tenths of a nat in a pruned model, which makes the stock-vs-patched comparison unreadable. These kernels keep the upstream protocol (records during the verify, ONE commit launch after
sampling, conv compaction and block-boundary states from upstream's plan and conv kernels) and replace the arithmetic:

  verify  = the stock recurrence (BV 8, 1 warp, 3 stages, Triton's default argument specialization, the same tl.sum reduction order) from the committed state of the row's checkpoint
            slot, read only; outputs, and per token and head the records u [V] (the stock b_v after beta), kn [K] (L2-normalised key), dec [K] (exp of the gate), fp32, in a per-layer
            tensor [max_rows, Q, H, V + 2K] indexed by the row of the step's spec batch;
  fold    = sequential replay of the first commit_len records onto the checkpoint state, b_h = fma(u, kn, b_h * dec) per token with an explicit FMA (the same fp32 element operations
            the stock kernel applies), the final state to final_state_idx and, when the window reaches the next block boundary, the state after boundary_recovery_len tokens to
            boundary_state_idx. fp32 and bf16 state: the committed value is the rounded stock snapshot.

`GlmCommitContext` subclasses upstream's KDARecoverSSMCommitContext: its plan kernel (commit_lens, final/boundary block ids from the block table and the computed tokens) and its conv
state compaction run unchanged, only the fold launch is ours. Check: `python -m suffix_hybrid.glm_recoverssm oracle` (stock snapshots against verify + commit, incl. block boundaries)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

BV = 8
FOLD_BV = 32


@dataclass(frozen=True)
class GlmRecords:
    """One layer's verify records: rec [max_rows, Q, H, V + 2K] fp32 = u | kn | dec per (row of the step's spec batch, token, head). Alive from verify to the commit of the same step."""

    rec: torch.Tensor

    @property
    def correction(self) -> torch.Tensor:      # upstream's KDARecoverSSMCommitContext.create reads these two for pointers and strides only
        return self.rec

    @property
    def key_gate(self) -> torch.Tensor:
        return self.rec

    @classmethod
    def allocate(cls, max_rows: int, num_heads: int, q_len: int, head_dim: int, device: torch.device) -> "GlmRecords":
        return cls(torch.empty((max_rows, q_len, num_heads, 3 * head_dim), dtype=torch.float32, device=device))


@triton.jit
def kda_glm_verify_kernel(
    q, k, v, g, beta, o, state, rec, cu_seqlens, idx, a_log, g_bias, scale,
    stride_q_t, stride_k_t, stride_v_t, stride_beta_t, stride_state_slot, stride_idx_seq, stride_rec_row,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, LOWER_BOUND: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(idx + i_n * stride_idx_seq).to(tl.int64)
    if slot <= 0:
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k < K)[None, :]
    b_h = tl.load(state + slot * stride_state_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :], mask=mask_h, other=0).to(tl.float32)
    b_a_log = tl.exp(tl.load(a_log + i_hv).to(tl.float32))
    p_q = q + bos * stride_q_t + i_hv * K + o_k
    p_k = k + bos * stride_k_t + i_hv * K + o_k
    p_v = v + bos * stride_v_t + i_hv * V + o_v
    p_beta = beta + bos * stride_beta_t + i_hv
    p_gk = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_nu = rec + i_n * stride_rec_row + i_hv * (V + 2 * K) + o_v
    p_nk = rec + i_n * stride_rec_row + i_hv * (V + 2 * K) + V + o_k
    p_nd = p_nk + K
    tok = HV * (V + 2 * K)
    for i_t in range(0, T):
        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_gk = tl.load(p_gk).to(tl.float32)
        b_gk += tl.load(g_bias + i_hv * K + o_k).to(tl.float32)
        b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
        b_dk = exp(b_gk)
        b_h = b_h * b_dk[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = tl.sigmoid(tl.load(p_beta).to(tl.float32))
        b_v *= b_beta
        b_h = tl.fma(tl.broadcast_to(b_v[:, None], (BV, K)), tl.broadcast_to(b_k[None, :], (BV, K)), b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_nu, b_v, mask=mask_v)
        if i_v == 0:
            tl.store(p_nk, b_k)
            tl.store(p_nd, b_dk)
        p_q += stride_q_t
        p_k += stride_k_t
        p_o += HV * V
        p_v += stride_v_t
        p_gk += HV * K
        p_beta += stride_beta_t
        p_nu += tok
        p_nk += tok
        p_nd += tok


@triton.jit
def kda_glm_fold_kernel(
    state_ref, state_addrs, state_strides, rec_addrs, rec_strides, state_indices, commit_lens, final_idx, boundary_idx, boundary_lens,
    null_block_id, stride_state_indices, N,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, ALIGN_MODE: tl.constexpr,
):
    """Replay commit_len records of row n onto the checkpoint state of layer l (all layers of the group in one launch); no reductions, so the tile size does not matter for the bits."""
    i_lnh, i_v = tl.program_id(0), tl.program_id(1)
    i_l = i_lnh // (N * HV)
    i_nh = i_lnh % (N * HV)
    i_n, i_hv = i_nh // HV, i_nh % HV
    src = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
    if src <= null_block_id:
        return
    clen = tl.load(commit_lens + i_n)
    if clen == 0:
        return
    fin = tl.load(final_idx + i_n).to(tl.int64)
    if fin <= null_block_id:
        return
    st = tl.load(state_addrs + i_l).to(tl.pointer_type(state_ref.dtype.element_ty))
    stride_slot = tl.load(state_strides + i_l)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k < K)[None, :]
    tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(st + src * stride_slot + tile, mask=mask_h, other=0).to(tl.float32)
    rec = tl.load(rec_addrs + i_l).to(tl.pointer_type(tl.float32)) + i_n * tl.load(rec_strides + i_l)
    p_u = rec + i_hv * (V + 2 * K) + o_v
    p_k = rec + i_hv * (V + 2 * K) + V + o_k
    p_d = p_k + K
    if ALIGN_MODE:
        bnd = tl.load(boundary_idx + i_n).to(tl.int64)
        blen = tl.load(boundary_lens + i_n)
    for i_t in range(0, clen):
        b_u = tl.load(p_u, mask=mask_v, other=0)
        b_kn = tl.load(p_k)
        b_dn = tl.load(p_d)
        b_h = b_h * b_dn[None, :]
        b_h = tl.fma(tl.broadcast_to(b_u[:, None], (BV, K)), tl.broadcast_to(b_kn[None, :], (BV, K)), b_h)
        if ALIGN_MODE:
            if (i_t + 1 == blen) & (bnd > null_block_id):
                tl.store(st + bnd * stride_slot + tile, b_h.to(st.dtype.element_ty), mask=mask_h)
        p_u += HV * (V + 2 * K)
        p_k += HV * (V + 2 * K)
        p_d += HV * (V + 2 * K)
    tl.store(st + fin * stride_slot + tile, b_h.to(st.dtype.element_ty), mask=mask_h)


def kda_glm_verify(q, k, v, raw_g, raw_beta, A_log, dt_bias, lower_bound, checkpoint_state, records: GlmRecords, query_start_loc, state_indices, spec_query_len, out=None):
    """Drop-in for upstream's kda_recoverssm_verify (same argument order): the checkpoint state of state_indices[n] is read, never written."""
    _, tokens, hv, kd = q.shape
    vd = v.shape[-1]
    n = state_indices.shape[0]
    if query_start_loc.shape[0] != n + 1 or tokens > n * spec_query_len or records.rec.shape[0] < n or records.rec.shape[1] < spec_query_len:
        raise ValueError(f"RecoverSSM verify: {tokens} tokens, {n} rows, window {spec_query_len}, records {tuple(records.rec.shape)}")
    if raw_g.shape != q.shape or not raw_g.is_contiguous() or checkpoint_state.shape[1:] != (hv, vd, kd) or checkpoint_state.stride()[1:] != (vd * kd, kd, 1):
        raise ValueError("RecoverSSM verify: unsupported layout")
    for x, name in ((q, "q"), (k, "k"), (v, "v")):
        if x.stride(-1) != 1 or x.stride(2) != x.shape[3] or x.stride(1) < hv * x.shape[3]:
            raise ValueError(f"RecoverSSM verify: {name} layout {tuple(x.shape)} {x.stride()}")
    if raw_beta.stride(-1) != 1 or raw_beta.stride(1) < hv:
        raise ValueError("RecoverSSM verify: beta layout")
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    elif tuple(out.shape) != tuple(v.shape) or not out.is_contiguous():
        raise ValueError("RecoverSSM verify: output layout")
    kda_glm_verify_kernel[(n * hv, vd // BV)](
        q, k, v, raw_g, raw_beta, out, checkpoint_state, records.rec, query_start_loc, state_indices, A_log.reshape(-1), dt_bias.reshape(-1), kd ** -0.5,
        q.stride(1), k.stride(1), v.stride(1), raw_beta.stride(1), checkpoint_state.stride(0), state_indices.stride(0), records.rec.stride(0),
        HV=hv, K=kd, V=vd, BV=BV, LOWER_BOUND=float(lower_bound), num_warps=1, num_stages=3)
    return out


class GlmCommitContext(KDARecoverSSMCommitContext):
    """Upstream's context (arrays, plan buffers, plan + conv kernels) with the exact fold."""

    def commit(self, num_accepted_tokens, state_indices, query_start_loc, request_indices=None, block_table=None, num_computed_tokens=None, mamba_block_size=None) -> None:
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0] or query_start_loc.shape[0] != batch + 1:
            raise ValueError("RecoverSSM commit: batch exceeds the plan capacity or metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("RecoverSSM commit: request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(a is not None for a in align_args) and any(a is None for a in align_args):
            raise ValueError("RecoverSSM commit: align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError("RecoverSSM commit: the align block size must cover one speculative window")
        align = block_table is not None
        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens, request_indices, state_indices, query_start_loc, block_table, num_computed_tokens,
            self.commit_lens, self.final_state_indices, self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, mamba_block_size or 1, block_table.shape[1] if align else 1,
            num_accepted_tokens.stride(0), request_indices.stride(0) if request_indices is not None else 0, state_indices.stride(0), query_start_loc.stride(0),
            block_table_stride[0], block_table_stride[1], 0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len, num_warps=1)
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref, self.conv_state_base_addrs, self.conv_state_block_strides, self.conv_state_dim_strides, self.conv_state_token_strides, state_indices,
            self.commit_lens, self.final_state_indices, self.boundary_state_indices, self.boundary_recovery_lens, NULL_BLOCK_ID, conv_dim, self.conv_history_len,
            state_indices.stride(0), BLOCK_D=256, BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len), ALIGN_MODE=align, num_warps=4)
        state_ref = self.checkpoints[0]
        _, hv, vd, kd = state_ref.shape
        kda_glm_fold_kernel[(num_layers * batch * hv, vd // FOLD_BV)](
            state_ref, self.state_base_addrs, self.state_block_strides, self.correction_cache_base_addrs, self.correction_cache_block_strides, state_indices,
            self.commit_lens, self.final_state_indices, self.boundary_state_indices, self.boundary_recovery_lens, NULL_BLOCK_ID, state_indices.stride(0), batch,
            HV=hv, K=kd, V=vd, BV=FOLD_BV, ALIGN_MODE=align, num_warps=4)
