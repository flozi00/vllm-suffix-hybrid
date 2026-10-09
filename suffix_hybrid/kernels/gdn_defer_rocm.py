# SPDX-License-Identifier: Apache-2.0
"""GDN MTP-verify with deferred state commit (SUFFIX_ROCM_GDN_DEFER=1, ROCm).

vLLM's spec-decode contract for a GDN layer: the verify kernel reads the state from
slot num_accepted - 1 of the request's 1 + num_spec state slots and writes the state
after every verified token t to slot t, so whatever gets accepted can be read next
step. At MTP-4 that is 5 full fp32 state writes (48 heads x 128 x 128 x 4 B = 3.1 MB
each) per request and layer; only one of them is ever read.

Here a step writes the state after token 0 (always accepted) to slot 0, and for tokens
t >= 1 only their inputs (raw k, the program's v tile, a, b: 162 floats) to a record
inside slot 1's state tile. The next step rebuilds the accepted state from slot 0 by
replaying the first num_accepted - 1 recorded tokens with the same per-token code,
so states and outputs are bit-identical to the stock kernel's: one state read and one
write instead of one read and five writes.

Who else reads the slots: only vLLM's mamba "align" prefix-caching copies, and only
around block boundaries (post-step: the boundary state when the accepted tokens cross
one; pre-step: the committed state when the running block moves). A step whose
positions c + 1 .. c + 2 (num_spec + 1) contain a boundary (ZONE = the mamba block
size) writes every slot the stock way, and the step after one reads the stock way, so
those copies see what they always saw. num_accepted == 1 (first step after a prefill,
after vLLM's align reset) reads slot 0 under both contracts.

    python -m suffix_hybrid.kernels.gdn_defer_rocm   # multi-step GPU oracle vs AITER + us/call
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix gdn-defer]"
REC = 256  # floats per recorded token: raw k (K = 128), v tile (BV = 32), a, b


@triton.jit
def _gdn_token(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold,
               USE_QK_L2NORM: tl.constexpr):
    # One token of AITER's fused_rearrange_sigmoid_gated_delta_rule_update_kernel
    # (v0.1.24.post1, = FLA's fused_sigmoid_gating_delta_rule_update), same op order.
    x = a_raw + dt_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    b_g = -tl.exp(A_log_v) * softplus_x
    b_beta = tl.sigmoid(b_raw)
    if USE_QK_L2NORM:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    return b_h


@triton.jit(do_not_specialize=["stride_qkv_l", "stride_a_l", "stride_b_l", "stride_o_l",
                               "stride_idx_seq"])
def _gdn_defer_kernel(
    A_log, a, b, dt_bias, beta, threshold, qkv, o, state, cu_seqlens, state_indices,
    num_accepted, seq_lens, scale,
    stride_qkv_l, stride_a_l, stride_b_l, stride_o_l, stride_state_block, stride_idx_seq,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    ZONE: tl.constexpr, ZONE_REACH: tl.constexpr, REC: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    tile = i_hv * V * K + o_v[:, None] * K + o_k[None, :]  # this program's [BV, K] state tile
    tile_base = i_hv * V * K + i_v * BV * K  # ... contiguous from here (BV x K floats)
    rec_v = K + tl.arange(0, BV)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_v = tl.load(dt_bias + i_hv).to(tl.float32)
    acc = tl.load(num_accepted + i_n).to(tl.int64)
    c_now = tl.load(seq_lens + i_n).to(tl.int64) - n_tok  # tokens computed before this step
    idx_row = state_indices + i_n * stride_idx_seq
    slot0 = tl.load(idx_row).to(tl.int64)
    slot1 = tl.load(idx_row + 1).to(tl.int64)
    if ZONE > 0:
        cur_zone = (c_now + ZONE_REACH) // ZONE > c_now // ZONE
        c_prev = c_now - acc
        prev_zone = (c_prev + ZONE_REACH) // ZONE > c_prev // ZONE
    else:
        cur_zone = c_now < 0
        prev_zone = c_now < 0

    if (acc >= 2) & (prev_zone == 0):
        # Deferred: slot 0 holds the state after the previous step's token 0, slot 1's
        # tile the inputs of its tokens 1.. -> replay the accepted ones.
        if (slot0 <= 0) | (slot1 <= 0):
            return
        b_h = tl.zeros([BV, K], dtype=tl.float32)  # AITER's 0 + load (signed zeros)
        b_h += tl.load(state + slot0 * stride_state_block + tile).to(tl.float32)
        rec = state + slot1 * stride_state_block + tile_base
        for t in range(1, acc):
            r = rec + (t - 1) * REC
            b_h = _gdn_token(b_h, tl.load(r + o_k), tl.load(r + rec_v), tl.load(r + K + BV),
                             tl.load(r + K + BV + 1), A_log_v, dt_v, beta, threshold,
                             USE_QK_L2NORM)
    else:
        read_slot = tl.load(idx_row + acc - 1).to(tl.int64)
        if read_slot <= 0:
            return
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        b_h += tl.load(state + read_slot * stride_state_block + tile).to(tl.float32)

    p_q = qkv + bos * stride_qkv_l + i_h * K + o_k
    p_k = qkv + bos * stride_qkv_l + H * K + i_h * K + o_k
    p_v = qkv + bos * stride_qkv_l + 2 * H * K + i_hv * V + o_v
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + o_v
    for i_t in range(0, n_tok):
        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        a_raw = tl.load(p_a).to(tl.float32)
        b_raw = tl.load(p_b).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        b_h = _gdn_token(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold,
                         USE_QK_L2NORM)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        if i_t == 0:
            if slot0 > 0:
                tl.store(state + slot0 * stride_state_block + tile, b_h)
        elif cur_zone:
            slot_t = tl.load(idx_row + i_t).to(tl.int64)
            if slot_t > 0:
                tl.store(state + slot_t * stride_state_block + tile, b_h)
        elif slot1 > 0:
            r = state + slot1 * stride_state_block + tile_base + (i_t - 1) * REC
            tl.store(r + o_k, b_k)
            tl.store(r + rec_v, b_v)
            tl.store(r + K + BV, a_raw)
            tl.store(r + K + BV + 1, b_raw)
        p_q += stride_qkv_l
        p_k += stride_qkv_l
        p_v += stride_qkv_l
        p_a += stride_a_l
        p_b += stride_b_l
        p_o += stride_o_l


def gdn_defer(qkv, a, b, A_log, dt_bias, state, cu_seqlens, state_indices, num_accepted,
              seq_lens, out, num_k_heads, head_k_dim, head_v_dim, zone):
    """Spec-verify delta rule over packed post-conv qkv [T, 2 H K + HV V] (row-strided),
    a / b [T, HV] views (row-strided), state [blocks, HV, V, K] fp32 (vLLM's layer
    kv_cache[1]; slot ids <= 0 are NULL), state_indices [N, 1 + num_spec] (block ids),
    num_accepted / seq_lens [N], out [>= T, HV, V]. zone = mamba block size in align
    mode, else 0."""
    n = cu_seqlens.shape[0] - 1
    hv = a.shape[1]
    K, V, BV = head_k_dim, head_v_dim, 32
    win = state_indices.shape[1]
    assert K == triton.next_power_of_2(K) and V % BV == 0 and (win - 1) * REC <= BV * K
    assert REC >= K + BV + 2 and state.dtype == torch.float32 and state.stride(-1) == 1
    assert state.shape[1:] == (hv, V, K) and state[0].is_contiguous()
    assert qkv.stride(1) == a.stride(1) == b.stride(1) == 1 and state_indices.stride(1) == 1
    assert out.stride(-1) == 1 and out.stride(-2) == V
    if n == 0:
        return out
    _gdn_defer_kernel[(V // BV, n * hv)](
        A_log, a, b, dt_bias, 1.0, 20.0, qkv, out, state, cu_seqlens, state_indices,
        num_accepted, seq_lens, K**-0.5,
        qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), state.stride(0),
        state_indices.stride(0),
        H=num_k_heads, HV=hv, K=K, V=V, BV=BV, ZONE=zone, ZONE_REACH=2 * win, REC=REC,
        USE_QK_L2NORM=True, num_warps=4, num_stages=3)
    return out


def defer_spec(layer, mixed_qkv_spec, a_spec, b_spec, ssm_state, md):
    """_forward_core's spec verify under the deferred contract (mixed batches, and the
    all-spec ones forward_spec declines); returns core_attn_out_spec [1, T, HV, V]."""
    hv = layer.num_v_heads // layer.tp_size
    out = mixed_qkv_spec.new_empty((mixed_qkv_spec.shape[0], hv, layer.head_v_dim))
    gdn_defer(mixed_qkv_spec, a_spec, b_spec, layer.A_log, layer.dt_bias, ssm_state,
              md.spec_query_start_loc[: md.num_spec_decodes + 1], md.spec_state_indices_tensor,
              md.num_accepted_tokens, md.suffix_spec_seq_lens, out,
              layer.num_k_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim,
              md.suffix_zone)
    return out.unsqueeze(0)


def install(module) -> None:
    """rocm_patches `after` hook for qwen_gdn_linear_attn: its rewritten _forward_core
    calls _suffix_gdn_defer_spec (forward_spec checks SUFFIX_ROCM_GDN_DEFER itself)."""
    module._suffix_gdn_defer_spec = defer_spec


def main() -> int:
    """Multi-step oracle on silicon at Qwen3.8-Flash-Next GDN shapes (16 qk / 48 v heads
    x 128, MTP-4): AITER's fused_rearrange_sigmoid_gated_delta_rule (what
    SUFFIX_ROCM_GDN_MTP serves) on one state pool, gdn_defer on a copy, 14 steps of random
    acceptance with vLLM's align-mode copies emulated on both pools (post-step boundary
    copy, pre-step running-block move with the accepted-count reset). Every step's outputs
    must be bitwise equal; so must every slot a boundary copy reads. Then graphed us/call
    at c1/c8/c32 in the deferred steady state."""
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as aiter_gdr)

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    dev, bf16 = "cuda", torch.bfloat16
    H, HV, K, V, spec = 16, 48, 128, 128, 4
    win, key_dim, value_dim = spec + 1, H * K, HV * V
    torch.manual_seed(0)
    A_log = (0.5 * torch.randn(HV, device=dev)).float()
    dt_bias = (0.5 * torch.randn(HV, device=dev)).to(bf16)
    failed = False

    def run_stock(pool, qkv, ba, cu, idx, acc, out):
        b, a = ba.unflatten(-1, (2, HV)).transpose(0, 1).contiguous()
        aiter_gdr(A_log=A_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv, key_dim=key_dim,
                  value_dim=value_dim, head_k_dim=K, head_v_dim=V, initial_state=pool[1:],
                  inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=idx - 1,
                  num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True, core_attn_out=out)

    def run_defer(pool, qkv, ba, cu, idx, acc, seq, out, zone):
        gdn_defer(qkv, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, idx, acc, seq, out,
                  H, K, V, zone)

    for zone, n_req, steps in ((0, 32, 10), (24, 32, 14), (1664, 32, 10), (24, 8, 14)):
        pool_blocks = n_req * win * 4 + 1
        stock = torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1
        defer = stock.clone()
        free = (torch.randperm(pool_blocks - 1) + 1).tolist()
        windows = [[free.pop() for _ in range(win)] for _ in range(n_req)]
        c = torch.randint(0, 3 * zone + 50 if zone else 4000, (n_req,)).tolist()
        acc = [1] * n_req
        ok, boundary_checks, moves, zone_steps = True, 0, 0, 0
        def q_of(step, r):  # a few rows draft fewer tokens
            return win if r % 7 else 1 + (step + r) % win

        for step in range(steps):
            qlens = [q_of(step, r) for r in range(n_req)]
            cu = torch.zeros(n_req + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(qlens).cumsum(0)
            T = int(cu[-1])
            qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
            ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
            idx = torch.tensor(windows, dtype=torch.int32, device=dev)
            acc_t = torch.tensor(acc, dtype=torch.int32, device=dev)
            seq_t = torch.tensor([ci + q for ci, q in zip(c, qlens)], dtype=torch.int32,
                                 device=dev)
            if zone:
                zone_steps += sum((ci + 2 * win) // zone > ci // zone for ci in c)
            o_s = torch.full((T, HV, V), float("nan"), device=dev, dtype=bf16)
            o_d = o_s.clone()
            run_stock(stock, qkv, ba, cu.to(dev), idx, acc_t, o_s)
            run_defer(defer, qkv, ba, cu.to(dev), idx, acc_t, seq_t, o_d, zone)
            same = torch.equal(o_s, o_d)
            ok &= same
            if not same:
                print(f"{MARK} zone {zone} step {step}: output mismatch, max abs diff "
                      f"{(o_s.float() - o_d.float()).abs().nan_to_num(1e9).max().item():.2e}",
                      flush=True)
            new_acc = [int(torch.randint(1, q + 1, ()).item()) for q in qlens]
            for r in range(n_req):
                if not zone:
                    continue
                # Post-step align copy source: the slot holding the state at the boundary.
                lo, hi = c[r] + 1, c[r] + new_acc[r]
                bnd = hi // zone * zone
                if bnd >= lo:
                    s = windows[r][bnd - lo]
                    boundary_checks += 1
                    ok &= torch.equal(stock[s], defer[s])
                # Pre-step running-block move: copy the committed state (slot acc - 1) to
                # a new window's slot 0 and reset acc to 1, on both pools.
                if ((c[r] + qlens[r] - 1) // zone
                        != (c[r] + new_acc[r] + q_of(step + 1, r) - 1) // zone):
                    src = windows[r][new_acc[r] - 1]
                    new_win = [free.pop() for _ in range(win)]
                    for pool in (stock, defer):
                        pool[new_win[0]].copy_(pool[src])
                    free.extend(windows[r])
                    windows[r], new_acc[r] = new_win, 1
                    moves += 1
            c = [ci + ai for ci, ai in zip(c, new_acc)]
            acc = new_acc
        failed |= not ok
        print(f"{MARK} zone {zone}, {n_req} requests x {steps} steps: "
              f"{'MATCH (bitwise)' if ok else 'MISMATCH'} | stock-way steps {zone_steps}, "
              f"boundary copies checked {boundary_checks}, running-block moves {moves}",
              flush=True)

    # Steady state (no zone, accepted 1..5), graphed: stock (5 state writes) vs deferred.
    for n_req in (1, 8, 32):
        pool_blocks = n_req * win * 4 + 1
        pools = [torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1 for _ in range(4)]
        cu = (torch.arange(n_req + 1, dtype=torch.int32) * win).to(dev)
        T = n_req * win
        qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
        ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
        idx = ((torch.randperm(pool_blocks - 1)[: n_req * win] + 1).view(n_req, win)
               .to(torch.int32).to(dev))
        acc_t = torch.randint(1, win + 1, (n_req,), dtype=torch.int32, device=dev)
        seq_t = (torch.randint(100, 1500, (n_req,), dtype=torch.int32) + win).to(dev)
        out = torch.empty(T, HV, V, device=dev, dtype=bf16)
        t_s = _graph_us(lambda i: run_stock(pools[i % 4], qkv, ba, cu, idx, acc_t, out), 8)[0]
        t_d = _graph_us(lambda i: run_defer(pools[i % 4], qkv, ba, cu, idx, acc_t, seq_t, out,
                                            0), 8)[0]
        print(f"{MARK} c{n_req} x mtp5 steady state: graphed stock {t_s:.1f} us -> "
              f"deferred {t_d:.1f} us", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
