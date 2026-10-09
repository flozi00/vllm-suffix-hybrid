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
    WIN: tl.constexpr, ZONE: tl.constexpr, ZONE_REACH: tl.constexpr, REC: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    # Fully unrolled over the WIN = 1 + num_spec token slots (masked past the request's
    # query length): no data-dependent loop, so every load (state, record, the tokens'
    # q/k/v/a/b) can issue before the recurrence starts.
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    two = tl.arange(0, 2)
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
    # Deferred: slot 0 holds the state after the previous step's token 0, slot 1's tile
    # the inputs of its tokens 1.. -> replay the accepted ones. Else vLLM's contract.
    deferred = (acc >= 2) & (prev_zone == 0)
    read_slot = tl.where(deferred, slot0, tl.load(idx_row + acc - 1).to(tl.int64))
    if read_slot <= 0:
        return
    if deferred & (slot1 <= 0):
        return
    b_h = tl.zeros([BV, K], dtype=tl.float32)  # AITER's 0 + load (signed zeros)
    b_h += tl.load(state + read_slot * stride_state_block + tile).to(tl.float32)
    rec = state + slot1 * stride_state_block + tile_base
    for t in tl.static_range(1, WIN):
        use = deferred & (t < acc)
        r = rec + (t - 1) * REC
        r_k = tl.load(r + o_k, mask=use, other=0.0)  # issued up front, masked
        r_v = tl.load(r + rec_v, mask=use, other=0.0)
        r_ab = tl.load(r + K + BV + two, mask=use, other=0.0)
        if use:  # only the accepted tokens pay for the recurrence
            b_h = _gdn_token(b_h, r_k, r_v, tl.sum(tl.where(two == 0, r_ab, 0.0)),
                             tl.sum(tl.where(two == 1, r_ab, 0.0)), A_log_v, dt_v, beta,
                             threshold, USE_QK_L2NORM)

    p_q = qkv + bos * stride_qkv_l + i_h * K + o_k
    p_k = qkv + bos * stride_qkv_l + H * K + i_h * K + o_k
    p_v = qkv + bos * stride_qkv_l + 2 * H * K + i_hv * V + o_v
    p_a = a + bos * stride_a_l + i_hv
    p_b = b + bos * stride_b_l + i_hv
    p_o = o + bos * stride_o_l + i_hv * V + o_v
    for i_t in tl.static_range(WIN):
        valid = i_t < n_tok
        b_q = tl.load(p_q + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_k = tl.load(p_k + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        b_v = tl.load(p_v + i_t * stride_qkv_l, mask=valid, other=0.0).to(tl.float32)
        a_raw = tl.load(p_a + i_t * stride_a_l, mask=valid, other=0.0).to(tl.float32)
        b_raw = tl.load(p_b + i_t * stride_b_l, mask=valid, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        h_new = _gdn_token(b_h, b_k, b_v, a_raw, b_raw, A_log_v, dt_v, beta, threshold,
                           USE_QK_L2NORM)
        b_h = tl.where(valid, h_new, b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o + i_t * stride_o_l, b_o.to(p_o.dtype.element_ty), mask=valid)
        if i_t == 0:
            tl.store(state + slot0 * stride_state_block + tile, b_h,
                     mask=valid & (slot0 > 0))
        else:
            slot_t = tl.load(idx_row + i_t).to(tl.int64)
            tl.store(state + slot_t * stride_state_block + tile, b_h,
                     mask=valid & cur_zone & (slot_t > 0))
            r = rec + (i_t - 1) * REC
            keep = valid & (cur_zone == 0) & (slot1 > 0)
            tl.store(r + o_k, b_k, mask=keep)
            tl.store(r + rec_v, b_v, mask=keep)
            tl.store(r + K + BV + two, tl.where(two == 0, a_raw, b_raw), mask=keep)


CONFIG = (16, 1, 1)  # BV, num_warps, num_stages. MI350P k16 oracle, graphed us c1/c8/c32:
# 16/1w 15.0/52.5/179.1, 32/1w 20.3/54.0/173.8, 32/2w 20.0/56.3/196.2; AITER 19.9/66.4/225.5.


def gdn_defer(qkv, a, b, A_log, dt_bias, state, cu_seqlens, state_indices, num_accepted,
              seq_lens, out, num_k_heads, head_k_dim, head_v_dim, zone, config=None):
    """Spec-verify delta rule over packed post-conv qkv [T, 2 H K + HV V] (row-strided),
    a / b [T, HV] views (row-strided), state [blocks, HV, V, K] fp32 (vLLM's layer
    kv_cache[1]; slot ids <= 0 are NULL), state_indices [N, 1 + num_spec] (block ids),
    num_accepted / seq_lens [N], out [>= T, HV, V]. zone = mamba block size in align
    mode, else 0."""
    n = cu_seqlens.shape[0] - 1
    hv = a.shape[1]
    BV, warps, stages = config or CONFIG
    K, V = head_k_dim, head_v_dim
    win = state_indices.shape[1]
    rec = (K + BV + 2 + 63) // 64 * 64  # floats per recorded token: raw k, the v tile, a, b
    assert K == triton.next_power_of_2(K) and V % BV == 0 and (win - 1) * rec <= BV * K
    assert state.dtype == torch.float32 and state.stride(-1) == 1
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
        H=num_k_heads, HV=hv, K=K, V=V, BV=BV, WIN=win, ZONE=zone, ZONE_REACH=2 * win,
        REC=rec,
        USE_QK_L2NORM=True, num_warps=warps, num_stages=stages)
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
    x 128, MTP-4), three state pools stepped together through random acceptance with
    vLLM's align-mode copies emulated on each (post-step boundary copy, pre-step
    running-block move with the accepted-count reset):
      S  AITER's fused_rearrange_sigmoid_gated_delta_rule (what SUFFIX_ROCM_GDN_MTP serves),
      R  gdn_defer with zone 1: every step writes all slots and reads vLLM's way,
      D  gdn_defer under test (zone 0 = never stock-way, 24 = often, 1664 = serving).
    D must equal R bitwise (outputs and every slot a boundary copy reads): the replay is
    exact. R must stay within bf16 rounding of S (same math, other fp32 op schedule).
    Then graphed us/call at c1/c8/c32 in the deferred steady state."""
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

    def run_defer(pool, qkv, ba, cu, idx, acc, seq, out, zone, config=None):
        gdn_defer(qkv, ba[:, HV:], ba[:, :HV], A_log, dt_bias, pool, cu, idx, acc, seq, out,
                  H, K, V, zone, config)

    def q_of(step, r):  # a few rows draft fewer tokens
        return win if r % 7 else 1 + (step + r) % win

    for zone, n_req, steps in ((0, 32, 10), (24, 32, 14), (1664, 32, 10), (24, 8, 14)):
        pool_blocks = n_req * win * 4 + 1
        pools = {"S": torch.randn(pool_blocks, HV, V, K, device=dev) * 0.1}
        pools["R"], pools["D"] = pools["S"].clone(), pools["S"].clone()
        free = (torch.randperm(pool_blocks - 1) + 1).tolist()
        windows = [[free.pop() for _ in range(win)] for _ in range(n_req)]
        c = torch.randint(0, 3 * zone + 50 if zone else 4000, (n_req,)).tolist()
        acc = [1] * n_req
        exact, worst_rs, checks, moves, stock_way = True, 0.0, 0, 0, 0
        for step in range(steps):
            qlens = [q_of(step, r) for r in range(n_req)]
            cu = torch.zeros(n_req + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(qlens).cumsum(0)
            cu = cu.to(dev)
            T = int(cu[-1])
            qkv = torch.randn(T, 2 * key_dim + value_dim, device=dev).to(bf16)
            ba = torch.randn(T, 2 * HV, device=dev).to(bf16)
            idx = torch.tensor(windows, dtype=torch.int32, device=dev)
            acc_t = torch.tensor(acc, dtype=torch.int32, device=dev)
            seq_t = torch.tensor([ci + q for ci, q in zip(c, qlens)], dtype=torch.int32,
                                 device=dev)
            if zone:
                stock_way += sum((ci + 2 * win) // zone > ci // zone for ci in c)
            outs = {k: torch.full((T, HV, V), float("nan"), device=dev, dtype=bf16)
                    for k in pools}
            run_stock(pools["S"], qkv, ba, cu, idx, acc_t, outs["S"])
            run_defer(pools["R"], qkv, ba, cu, idx, acc_t, seq_t, outs["R"], 1)
            run_defer(pools["D"], qkv, ba, cu, idx, acc_t, seq_t, outs["D"], zone)
            exact &= torch.equal(outs["D"], outs["R"])
            worst_rs = max(worst_rs, ((outs["R"].float() - outs["S"].float()).abs()
                                      / (2**-7 * outs["S"].float().abs() + 1e-3)).max().item())
            new_acc = [int(torch.randint(1, q + 1, ()).item()) for q in qlens]
            for r in range(n_req):
                if not zone:
                    continue
                # Post-step align copy source: the slot holding the state at the boundary.
                lo, hi = c[r] + 1, c[r] + new_acc[r]
                bnd = hi // zone * zone
                if bnd >= lo:
                    sl = windows[r][bnd - lo]
                    checks += 1
                    exact &= torch.equal(pools["D"][sl], pools["R"][sl])
                # Pre-step running-block move: copy the committed state (slot acc - 1) to a
                # new window's slot 0 and reset acc to 1, on every pool.
                if ((c[r] + qlens[r] - 1) // zone
                        != (c[r] + new_acc[r] + q_of(step + 1, r) - 1) // zone):
                    src = windows[r][new_acc[r] - 1]
                    new_win = [free.pop() for _ in range(win)]
                    for pool in pools.values():
                        pool[new_win[0]].copy_(pool[src])
                    free.extend(windows[r])
                    windows[r], new_acc[r] = new_win, 1
                    moves += 1
            c = [ci + ai for ci, ai in zip(c, new_acc)]
            acc = new_acc
        ok = exact and worst_rs <= 1.0
        failed |= not ok
        print(f"{MARK} zone {zone}, {n_req} requests x {steps} steps: deferred vs stock-way "
              f"{'bitwise' if exact else 'MISMATCH'}, vs AITER worst {worst_rs:.2f} of the "
              f"bf16 bound -> {'MATCH' if ok else 'MISMATCH'} | stock-way steps {stock_way}, "
              f"boundary copies checked {checks}, running-block moves {moves}", flush=True)

    # Steady state (accepted 1..5), graphed: AITER (5 state writes), gdn_defer stock-way
    # (zone 1, 5 writes) and deferred (zone 0: 1 state write + the record).
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
        sweep = {0: [], 1: []}
        for cfg in ((32, 2, 1), (32, 1, 1), (16, 1, 1), (16, 2, 1), (64, 1, 1), (64, 2, 1),
                    (64, 4, 1), (128, 2, 1), (128, 4, 1)):
            for zone in (0, 1):  # deferred, stock-way
                try:
                    us = _graph_us(lambda i: run_defer(pools[i % 4], qkv, ba, cu, idx, acc_t,
                                                       seq_t, out, zone, cfg), 8)[0]
                    sweep[zone].append(f"{cfg[0]}/{cfg[1]}w {us:.1f}")
                except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                    sweep[zone].append(f"{cfg[0]}/{cfg[1]}w {type(exc).__name__}")
        print(f"{MARK} c{n_req} x mtp5 steady state graphed: AITER {t_s:.1f} us | deferred "
              f"BV/warps: {' | '.join(sweep[0])} | stock-way: {' | '.join(sweep[1])}",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
