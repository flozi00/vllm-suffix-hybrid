"""Triton kernel and per-layer state for the lazy-state KDA speculative verify (see kda_lazy.py for the vLLM hook and the doctrine).

Stock vLLM (0.33.0, glm5next KDA layer) writes a [H, V, K] fp32 state snapshot per verified token and per stream (T x 1 MiB per layer, 288 MiB per layer at
32 streams x 8 tokens) so the rejection sampler can roll back; the next step starts from the snapshot of the last accepted token. Here a request keeps ONE
state (column 0 of its block row), always one window behind, plus compact records of its last window in column 1 of the block row (no new allocation) and a
4-byte header per column-0 block in a small per-layer aux tensor.

One launch per layer and step, per program (row n, head, BV-wide slice of V), in this order:
  1. read the header of the block: which of the two record buffers holds the previous window and how many tokens it has (0 = fresh request, or cleared by a prefill);
  2. replay r = min(num_accepted, tokens) records onto the state (b_h = b_h * dec + u (x) kn, the same fp32 element operations the stock kernel applies, with an
     explicit FMA so the contraction does not depend on the loop's context): the committed state is bit-identical to the stock snapshot of the last accepted
     token. It is stored back to column 0 (only if r > 0);
  3. run the stock recurrence over the T new tokens from there (outputs) and write, per token and head, the records u [V] (the stock b_v after beta), kn [K]
     (L2-normalised key) and dec [K] (exp of the gate) into the OTHER record buffer; no snapshots;
  4. count itself done in a per-row counter; the LAST program of the row (every sibling has read the header by then) flips the header (other buffer, T) and
     resets the counter.
The sibling programs of a row run concurrently and share the header and the kn/dec records. The records are double-buffered and the header is only written by the
last program of the row, so no program can see a half-updated view. A prefill step clears the header of its block (a new request must not replay the records a
previous owner of the block left behind).

Record layout in the column-1 block, as fp32: [buffer 0/1][token][head][V + 2K] = u | kn | dec. At T = 8, H = 16, K = V = 128 one buffer is 192 KiB, both 384 KiB, in
the 1 MiB SSM region of a slot (the state tensor's stride(0) may be larger, the conv state shares the page). Header (int32 per block): bits 0-7 T of the previous
window, bit 8 its buffer.

Verified bit-exact against the stock fused_recurrent_kda over many cycles (random acceptance, variable window lengths, null rows, rows that sit out a cycle, a
prefill clear of a block holding records, fp32 and bf16 state, eager and CUDA-graph replay): `python -m suffix_hybrid.kda_lazy oracle`."""
from __future__ import annotations

import torch
from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

BV = 8


# No do_not_specialize on the strides: the stock kernel is compiled with Triton's default argument specialization (divisibility by 16 of q/k/v/state strides gives the
# vectorised loads and with them the reduction order of tl.sum), and the replay/verify is bit-identical to it only when it gets the same code. The strides are
# constants of the model, so the variants are all compiled at warmup.
@triton.jit
def kda_lazy_kernel(
    q, k, v, g, beta, o, state, hdr, cnt, cu_seqlens, idx, acc, a_log, g_bias, scale,
    stride_q_t, stride_k_t, stride_v_t, stride_beta_t, stride_state_slot, stride_idx_seq,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, TREC: tl.constexpr, LOWER_BOUND: tl.constexpr,
    REPLAY_ONLY: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot0 = tl.load(idx + i_n * stride_idx_seq).to(tl.int64)
    slot1 = tl.load(idx + i_n * stride_idx_seq + 1).to(tl.int64)
    if slot0 <= 0:
        return
    if slot1 <= 0:
        return
    # 1. header: the previous window (buffer, tokens); it only changes after every program of the row has read it (step 4)
    h = tl.load(hdr + slot0)
    old_t = (h & 0xFF).to(tl.int64)
    old_buf = ((h >> 8) & 1).to(tl.int64)
    new_buf = 1 - old_buf
    r = tl.minimum(tl.load(acc + i_n).to(tl.int64), old_t)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k < K)[None, :]
    p_h0 = state + slot0 * stride_state_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    rec = (state + slot1 * stride_state_slot).to(tl.pointer_type(tl.float32))
    tok = HV * (V + 2 * K)
    p_ou = rec + (old_buf * TREC * HV + i_hv) * (V + 2 * K) + o_v
    p_ok = rec + (old_buf * TREC * HV + i_hv) * (V + 2 * K) + V + o_k
    p_od = p_ok + K
    # 2. replay the accepted tokens of the previous window onto the state, store the committed state
    for i_t in range(0, r):
        b_u = tl.load(p_ou, mask=mask_v, other=0)
        b_kn = tl.load(p_ok)
        b_dn = tl.load(p_od)
        b_h = b_h * b_dn[None, :]
        b_h = tl.fma(tl.broadcast_to(b_u[:, None], (BV, K)), tl.broadcast_to(b_kn[None, :], (BV, K)), b_h)
        p_ou += tok
        p_ok += tok
        p_od += tok
    if r > 0:
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
        b_h = b_h.to(p_h0.dtype.element_ty).to(tl.float32)   # stock reloads the rounded snapshot as the next window's initial state (no-op for fp32 state)
    if not REPLAY_ONLY:
        # 3. verify the new window from the committed state, write the outputs and the records into the other buffer
        b_a_log = tl.exp(tl.load(a_log + i_hv).to(tl.float32))
        p_q = q + bos * stride_q_t + i_hv * K + o_k
        p_k = k + bos * stride_k_t + i_hv * K + o_k
        p_v = v + bos * stride_v_t + i_hv * V + o_v
        p_beta = beta + bos * stride_beta_t + i_hv
        p_gk = g + (bos * HV + i_hv) * K + o_k
        p_o = o + (bos * HV + i_hv) * V + o_v
        p_nu = rec + (new_buf * TREC * HV + i_hv) * (V + 2 * K) + o_v
        p_nk = rec + (new_buf * TREC * HV + i_hv) * (V + 2 * K) + V + o_k
        p_nd = p_nk + K
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
        # 4. the last program of the row publishes the new header (other buffer, T) and resets the counter
        done = tl.atomic_add(cnt + i_n, 1)
        if done == HV * (V // BV) - 1:
            tl.store(hdr + slot0, ((new_buf << 8) | T).to(tl.int32))
            tl.store(cnt + i_n, 0)


def token_stride(x: torch.Tensor) -> int:
    """Token stride (elements) of a [1, T, H, D] or [1, T, H] tensor whose per-token [H, D] block is contiguous (what the stock kernel also requires)."""
    st = x.stride()
    if not (x.dim() in (3, 4) and x.shape[0] == 1 and st[-1] == 1 and (x.dim() == 3 or st[2] == x.shape[3])
            and st[1] >= x.shape[2] * (x.shape[3] if x.dim() == 4 else 1)):
        raise RuntimeError(f"kda_lazy: unsupported activation layout shape={tuple(x.shape)} stride={st}")
    return st[1]


class LazyKDA:
    """One KDA layer: its state tensor [num_blocks, H, V, K] (any float dtype), the header per column-0 block, the per-row completion counters."""

    def __init__(self, state: torch.Tensor, spec_len: int, max_rows: int = 1024):
        nb, hv, v, k = state.shape
        if state.stride()[1:] != (v * k, k, 1):
            raise RuntimeError(f"kda_lazy: state layout {tuple(state.shape)} stride {state.stride()} (want dense [H, V, K] inside a slot)")
        if v % BV or (k & (k - 1)):
            raise RuntimeError(f"kda_lazy: V={v} must be a multiple of {BV} and K={k} a power of two")
        if 2 * spec_len * hv * (v + 2 * k) * 4 > hv * v * k * state.element_size():
            raise RuntimeError(f"kda_lazy: the records of two {spec_len}-token windows do not fit the {hv * v * k * state.element_size()} B state slot")
        self.state, self.spec_len, self.hv, self.v, self.k = state, spec_len, hv, v, k
        self.hdr = torch.zeros(nb, dtype=torch.int32, device=state.device)
        self.cnt = torch.zeros(max_rows, dtype=torch.int32, device=state.device)

    def run(self, q, k, v, g, beta, cu_seqlens, idx, acc, a_log, g_bias, lower_bound, out=None, replay_only=False):
        """Replace of fused_recurrent_kda(..., ssm_state_indices=[N, T+], num_accepted_tokens=acc) on the spec path: returns the outputs [1, tokens, H, V]."""
        hv, vd, kd = self.hv, self.v, self.k
        n = cu_seqlens.numel() - 1
        if idx.dim() != 2 or idx.shape[0] < n or idx.shape[1] < 2 or idx.shape[1] > self.spec_len:
            raise RuntimeError(f"kda_lazy: ssm_state_indices {tuple(idx.shape)} vs spec_len {self.spec_len}")
        if n > self.cnt.numel():
            raise RuntimeError(f"kda_lazy: {n} rows > {self.cnt.numel()} counters")
        if tuple(g.shape) != (1, q.shape[1], hv, kd) or not g.is_contiguous():
            raise RuntimeError(f"kda_lazy: raw gate layout shape={tuple(g.shape)} stride={g.stride()}")
        if out is None:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        elif tuple(out.shape) != tuple(v.shape) or not out.is_contiguous():
            raise RuntimeError(f"kda_lazy: output layout shape={tuple(out.shape)} stride={out.stride()}")
        kda_lazy_kernel[(n * hv, vd // BV)](
            q, k, v, g, beta, out, self.state, self.hdr, self.cnt, cu_seqlens, idx, acc, a_log.reshape(-1), g_bias.reshape(-1), kd ** -0.5,
            token_stride(q), token_stride(k), token_stride(v), token_stride(beta), self.state.stride(0), idx.stride(0),
            HV=hv, K=kd, V=vd, BV=BV, TREC=self.spec_len, LOWER_BOUND=float(lower_bound), REPLAY_ONLY=replay_only, num_warps=1, num_stages=3)
        return out

    def clear(self, blocks: torch.Tensor) -> None:
        """Non-spec (prefill) sequences: their block must not replay records of a previous owner (eager only; prefill never runs in a full graph)."""
        self.hdr.index_fill_(0, blocks.clamp(min=0).long(), 0)
