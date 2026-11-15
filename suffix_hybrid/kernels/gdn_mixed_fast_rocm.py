# SPDX-License-Identifier: Apache-2.0
"""Mixed-step GDN prefill from a per-step launch plan (SUFFIX_ROCM_GDN_MIXED_FAST, ROCm).

gdn_mtp_rocm.forward_mixed runs a mixed step's prefill rows through vLLM's
causal_conv1d_fn, fused_post_conv_prep and FLA's chunk_gated_delta_rule (cumsum, kkt,
solve_tril, recompute_w_u, fwd_h, fwd_o) for each of the 36 GDN layers. Those sit behind
autograd.Function / input_guard / autotune / heuristics wrappers, ~15 allocations and a
zeros_like fill per layer: 40-70 us of Python between kernels while the GPU idles (k34b
PIECEWISE steps, ~0.45 ms per c8 step). Everything the wrappers derive is the same for every
GDN layer of a step: grids, chunk indices / offsets and the conv's program map (already in
the metadata), heuristics flags, the autotuned configs (their keys hold no token count) and
the buffer shapes. The first layer of a step builds that once (`_build`); each layer then
launches the same kernels with the same arguments and configs:

  1  through each kernel's JITFunction (binder + compile cache), skipping the wrappers;
  2  also, from the step's second layer on, straight through the CompiledKernel's
     launcher with JITFunction.run's argument list (signature order) whenever every
     tensor's 16-byte alignment matches the first call's (ints, dtypes and shapes are the
     step's own, so that is the whole specialization).

Bit for bit the stock path: same kernels, configs and inputs. Differences are confined to
memory stock leaves uninitialized and its kernels mask: buffers live for the step instead
of the layer, the conv reads its rows straight from qkvz (stock reads a contiguous copy of
the same values), and solve_tril's Ai is zeroed once per step instead of per layer (the
merge kernel never writes its upper 16x16 blocks or rows past a sequence's end, so those
stay zero). Steps with a prefill longer than MAX_T tokens keep the stock calls: their GPU
work per layer dwarfs the Python, and their buffers would outlive the layer.

    python -m suffix_hybrid.kernels.gdn_mixed_fast_rocm   # GPU oracle (boot gate gdn_mixed_bench)
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import torch

from vllm.triton_utils import triton

MARK = "[suffix gdn-mixed]"
MODE = int(os.environ.get("SUFFIX_ROCM_GDN_MIXED_FAST", "0") or 0)
MAX_T = int(os.environ.get("SUFFIX_ROCM_GDN_MIXED_FAST_MAX_T", "2048"))
TUNED = ("cumsum", "kkt", "merge", "wu", "h", "o")  # FLA's autotuned kernels
_CFG: dict = {}  # name -> autotuned config kwargs, captured after a stock call (capture())
_K = None
_SEEN: set = set()  # first built / declined plan of the process -> one log line each


def _kernels():
    global _K
    if _K is None:
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc
        from vllm.platforms import current_platform
        from vllm.third_party.flash_linear_attention.ops import (
            chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum, solve_tril, wy_fast)
        from vllm.third_party.flash_linear_attention.ops import (
            fused_gdn_prefill_post_conv as post)

        sig = inspect.signature(cc.causal_conv1d_fn).parameters
        _K = SimpleNamespace(
            conv=cc._causal_conv1d_fwd_kernel, post=post._fused_post_conv_kernel,
            cumsum=cumsum.chunk_local_cumsum_scalar_kernel,
            kkt=chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
            merge=solve_tril.merge_16x16_to_64x64_inverse_kernel,
            wu=wy_fast.recompute_w_u_fwd_kernel,
            h=chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            o=chunk_o.chunk_fwd_kernel_o,
            pad_slot_id=sig["pad_slot_id"].default, null_block_id=sig["null_block_id"].default,
            cast_dot=chunk_scaled_dot_kkt._CAST_DOT_TO_K_DTYPE, use_tma=solve_tril.is_tma_supported,
            tril_precision=solve_tril.FLA_TRIL_PRECISION,
            pdl=current_platform.is_arch_support_pdl())
    return _K


def _inner(k, cls_name):
    while type(k).__name__ != cls_name:
        k = k.fn
    return k


def capture() -> None:
    """Record the config every FLA autotuner used in the stock call that just ran (their keys
    are head counts / dims / BT / flags: one config per kernel for the process)."""
    if len(_CFG) == len(TUNED):
        return
    cfgs = {n: _inner(getattr(_kernels(), n), "Autotuner").best_config for n in TUNED}
    if all(c is not None for c in cfgs.values()):
        _CFG.update({n: c.all_kwargs() for n, c in cfgs.items()})


class _Launch:
    """One kernel launch of the step plan: JITFunction.run on the first call (and in MODE 1),
    then, in MODE 2, the first call's CompiledKernel straight through its launcher."""

    def __init__(self, kernel, grid, const):
        self.fn, self.const, self.kernel = _inner(kernel, "JITFunction"), const, None
        self.grid = tuple(grid) + (1,) * (3 - len(grid))

    def __call__(self, **args):
        if self.kernel is not None and MODE >= 2:
            if tuple(v.data_ptr() & 15 == 0 for v in args.values()
                     if isinstance(v, torch.Tensor)) == self.align:
                vals = [args[n] if n in args else self.const[n] for n in self.names]
                k, (g0, g1, g2) = self.kernel, self.grid
                stream = self.drv.get_current_stream(self.drv.get_current_device())
                k.run(g0, g1, g2, stream, k.function, k.packed_metadata,
                      k.launch_metadata(self.grid, stream, *vals),
                      self.knobs.runtime.launch_enter_hook, self.knobs.runtime.launch_exit_hook,
                      *vals)
                return
        kernel = self.fn[self.grid](**args, **self.const)
        if self.kernel is None:
            from triton import knobs
            from triton.runtime import driver

            self.kernel, self.names, self.knobs, self.drv = kernel, self.fn.arg_names, knobs, \
                driver.active
            self.align = tuple(v.data_ptr() & 15 == 0 for v in args.values()
                               if isinstance(v, torch.Tensor))


def _build(layer, md, nst, n, qkv_dim, hv, conv_state, w):
    """The step's buffers and launches, or False: this step keeps the stock calls."""
    m = layer.chunk_gated_delta_rule
    conv_map = (md.nums_dict or {}).get(8) or {}
    T = n - nst
    if (not 0 < T <= MAX_T or getattr(m, "gdn_prefill_backend", None) != "triton"
            or md.chunk_indices is None or md.chunk_offsets is None
            or md.aiter_prefill_metadata is not None or conv_map.get("batch_ptr") is None):
        return False
    K = _kernels()
    dev, bf16, f32 = conv_state.device, torch.bfloat16, torch.float32
    hk, dk, dv = layer.num_k_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim
    N = md.num_prefills
    NT, BT = md.chunk_indices.shape[0], 64
    f = SimpleNamespace(
        conv=torch.empty(T, qkv_dim, dtype=bf16, device=dev),
        q=torch.empty(T, hk, dk, dtype=bf16, device=dev), k=torch.empty(T, hk, dk, dtype=bf16, device=dev),
        v=torch.empty(T, hv, dv, dtype=bf16, device=dev), g=torch.empty(T, hv, dtype=f32, device=dev),
        beta=torch.empty(T, hv, dtype=f32, device=dev), h0=torch.empty(N, hv, dv, dk, dtype=f32, device=dev),
        g_cs=torch.empty(1, T, hv, dtype=f32, device=dev), A=torch.empty(1, T, hv, BT, dtype=f32, device=dev),
        Ai=torch.zeros(1, T, hv, BT, dtype=bf16, device=dev),
        w=torch.empty(1, T, hv, dk, dtype=bf16, device=dev), u=torch.empty(1, T, hv, dv, dtype=bf16, device=dev),
        h=torch.empty(1, NT, hv, dv, dk, dtype=bf16, device=dev),
        v_new=torch.empty(1, T, hv, dv, dtype=bf16, device=dev),
        ht=torch.empty(N, hv, dv, dk, dtype=f32, device=dev),
        batch_ptr=conv_map["batch_ptr"], tco_ptr=conv_map["token_chunk_offset_ptr"])
    act = "silu" if layer.activation is True else layer.activation
    width = w.shape[1]
    f.L_conv = _Launch(K.conv, (conv_map["tot"], triton.cdiv(qkv_dim, 256)), dict(
        dim=qkv_dim, stride_x_dim=1, stride_w_dim=w.stride(0), stride_w_width=w.stride(1),
        stride_istate_seq=conv_state.stride(0), stride_istate_dim=conv_state.stride(1),
        stride_istate_token=conv_state.stride(2),
        stride_cache_indices=md.non_spec_state_indices_tensor.stride(0), stride_o_dim=1,
        stride_block_m=0, pad_slot_id=K.pad_slot_id, null_block_id=K.null_block_id,
        HAS_BIAS=layer.conv1d.bias is not None, KERNEL_WIDTH=width,
        SILU_ACTIVATION=act in ("silu", "swish"), IS_APC_ENABLED=False,
        HAS_NULL_BLOCK=K.null_block_id is not None,
        NP2_STATELEN=triton.next_power_of_2(width - 1), BLOCK_M=8, BLOCK_N=256,
        launch_pdl=K.pdl, num_stages=2))
    f.L_post = _Launch(K.post, (triton.cdiv(T, 16), hk + hv), dict(
        stride_x_tok=f.conv.stride(0), stride_q_tok=f.q.stride(0), stride_k_tok=f.k.stride(0),
        stride_v_tok=f.v.stride(0), L=T, H=hk, HV=hv, K=dk, V=dv, APPLY_L2NORM=True,
        L2NORM_EPS=1e-6, OUTPUT_G_EXP=False, SOFTPLUS_THRESHOLD=20.0, BLOCK_T=16,
        BK=triton.next_power_of_2(dk), BV=triton.next_power_of_2(dv), num_warps=4,
        num_stages=2))
    # chunk_gated_delta_rule_fwd's kernels: wrapper arguments + heuristics + autotuned config
    cu, ci, co = md.prefill_query_start_loc, md.chunk_indices, md.chunk_offsets
    f.L_cumsum = _Launch(K.cumsum, (NT, hv), dict(
        s=f.g.unsqueeze(0), o=f.g_cs, cu_seqlens=cu, chunk_indices=ci, T=T, B=1, H=hv, BT=BT,
        HEAD_FIRST=False, REVERSE=False, IS_VARLEN=True, **_CFG["cumsum"]))
    f.L_kkt = _Launch(K.kkt, (NT, hv), dict(
        k=f.k.unsqueeze(0), g=f.g_cs, beta=f.beta.unsqueeze(0), A=f.A, cu_seqlens=cu,
        chunk_indices=ci, T=T, H=hv, Hg=hk, K=dk, BT=BT, CAST_DOT_TO_K_DTYPE=K.cast_dot,
        USE_G=True, IS_VARLEN=True, **_CFG["kkt"]))
    f.L_merge = _Launch(K.merge, (NT, hv), dict(
        A=f.A, Ai=f.Ai, cu_seqlens=cu, chunk_indices=ci, T=T, H=hv, BT=BT, USE_TMA=K.use_tma,
        DOT_PRECISION=K.tril_precision, IS_VARLEN=True, **_CFG["merge"]))
    f.L_wu = _Launch(K.wu, (NT, hv), dict(
        k=f.k.unsqueeze(0), v=f.v.unsqueeze(0), beta=f.beta.unsqueeze(0), w=f.w, u=f.u,
        A=f.Ai, g=f.g_cs, cu_seqlens=cu, chunk_indices=ci, T=T, H=hv, Hg=hk, K=dk, V=dv,
        BT=BT, BK=64, BV=64, IS_VARLEN=True, **_CFG["wu"]))
    f.L_h = _Launch(K.h, (triton.cdiv(dv, _CFG["h"]["BV"]), N * hv), dict(
        k=f.k.unsqueeze(0), v=f.u, w=f.w, v_new=f.v_new, g=f.g_cs, gk=None, h=f.h, h0=f.h0,
        ht=f.ht, cu_seqlens=cu, chunk_offsets=co, T=T, H=hv, Hg=hk, K=dk, V=dv, BT=BT,
        USE_EXP2=False, USE_G=True, USE_GK=False, USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True, IS_VARLEN=True, **_CFG["h"]))
    f.L_o = _Launch(K.o, (triton.cdiv(dv, _CFG["o"]["BV"]), NT, hv), dict(
        q=f.q.unsqueeze(0), k=f.k.unsqueeze(0), v=f.v_new, h=f.h, g=f.g_cs,
        cu_seqlens=cu, chunk_indices=ci, scale=dk ** -0.5, T=T, H=hv, Hg=hk, K=dk, V=dv, BT=BT,
        USE_G=True, IS_VARLEN=True, **_CFG["o"]))
    return f


def prefill(layer, qkvz, ba, core_attn_out, md, plan, nst, n, qkv_dim, hv, conv_state,
            ssm_state, w) -> bool:
    """forward_mixed's prefill suffix (rows [nst, n)) from the step plan: conv, post-conv
    prep, state gather, FLA chunk kernels, state scatter. False = run the stock calls."""
    if not _CFG:
        return False  # the stock call populates the autotuners first; capture() follows it
    f = getattr(plan, "fast", None)
    if f is None:
        f = plan.fast = _build(layer, md, nst, n, qkv_dim, hv, conv_state, w)
        if bool(f) not in _SEEN:
            _SEEN.add(bool(f))
            import sys
            print(f"{MARK} mode {MODE}: first step plan "
                  f"{'built' if f else 'declined (stock calls)'}: {n - nst} prefill tokens, "
                  f"{getattr(md, 'num_prefills', '?')} sequences, backend "
                  f"{getattr(getattr(layer, 'chunk_gated_delta_rule', None), 'gdn_prefill_backend', None)}",
                  file=sys.stderr, flush=True)
    if f is False:
        return False
    x = qkvz[nst:n, :qkv_dim]
    f.L_conv(x_ptr=x.T, w_ptr=w, bias_ptr=layer.conv1d.bias, initial_states_ptr=conv_state,
             cache_indices_ptr=md.non_spec_state_indices_tensor,
             has_initial_states_ptr=md.has_initial_state,
             query_start_loc_ptr=md.non_spec_query_start_loc, batch_ptr=f.batch_ptr,
             token_chunk_offset_ptr=f.tco_ptr, block_idx_first_scheduled_token=None,
             block_idx_last_scheduled_token=None, initial_state_idx=None,
             num_computed_tokens=None, o_ptr=f.conv.T, num_cache_lines=conv_state.size(0),
             stride_x_token=x.stride(0), stride_o_token=f.conv.stride(0))
    a, b = ba[nst:n, hv:], ba[nst:n, :hv]
    f.L_post(mixed_qkv_ptr=f.conv, a_ptr=a, b_ptr=b, A_log_ptr=layer.A_log,
             dt_bias_ptr=layer.dt_bias, q_ptr=f.q, k_ptr=f.k, v_ptr=f.v, g_ptr=f.g,
             beta_ptr=f.beta, stride_a_tok=a.stride(0), stride_b_tok=b.stride(0))
    torch.index_select(ssm_state, 0, plan.pre_idx, out=f.h0)
    f.h0.masked_fill_(plan.no_init, 0.0)
    f.L_cumsum()
    f.L_kkt()
    f.L_merge()
    f.L_wu()
    f.L_h()
    f.L_o(o=core_attn_out[nst:n].view(1, n - nst, hv, -1))
    ssm_state.index_copy_(0, plan.pre_idx, f.ht)
    if n < core_attn_out.shape[0]:
        core_attn_out[n:].zero_()
    return True


def main() -> int:
    """Oracle on silicon at Qwen3.8-Flash-Next GDN shapes (16 qk / 48 v heads x 128, conv 4,
    MTP-4, fp32 state): mixed steps of verify rows + prefill sequences, 36 layers' worth of
    forward_mixed calls (each its own state pages and inputs) on one step's metadata, stock
    vs plan MODE 1 vs MODE 2. Every output row, the zeroed tail and every state page byte
    must be equal; then host us per layer call (CPU timer around the Python, no sync), eager
    wall us per layer, and graphed GPU us per layer (36 calls in one HIP graph)."""
    import math
    import time

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as Q
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.index import (prepare_chunk_indices,
                                                                   prepare_chunk_offsets)
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    from suffix_hybrid.kernels import gdn_mtp_rocm as gm

    global MODE
    gm.install(Q)
    gm._DEFER = gm._MIXED = True
    dev, bf16 = "cuda", torch.bfloat16
    hk, hv, dk, dv, width, spec, layers = 16, 48, 128, 128, 4, 4, 36
    win, qkv_dim = spec + 1, 2 * hk * dk + hv * dv
    torch.manual_seed(0)

    class _Fla:  # ChunkGatedDeltaRule.forward_native, the triton backend
        gdn_prefill_backend = "triton"

        def __call__(self, aiter_prefill_metadata=None, **kw):
            return chunk_gated_delta_rule(**kw)

    layer = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.__dict__.update(
        prefix="gdn", tp_size=1, num_k_heads=hk, num_v_heads=hv, head_k_dim=dk, head_v_dim=dv,
        key_dim=hk * dk, value_dim=hv * dv, gqa_interleaved_layout=False, activation="silu",
        conv1d=SimpleNamespace(weight=0.3 * torch.randn(qkv_dim, 1, width, device=dev,
                                                        dtype=bf16), bias=None),
        A_log=0.5 * torch.randn(hv, device=dev), dt_bias=0.5 * torch.randn(hv, device=dev,
                                                                           dtype=bf16),
        chunk_gated_delta_rule=_Fla())
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, hk, hv, dk, dv, width, spec)
    cbytes, sbytes = math.prod(conv_shape) * 2, math.prod(ssm_shape) * 4
    page = cbytes + sbytes + 4096

    def views(pages):
        return (pages[:, :cbytes].view(bf16).view(-1, *conv_shape),
                pages[:, cbytes:cbytes + sbytes].view(torch.float32).view(-1, *ssm_shape))

    def case(ns, lens, init, tail=8):
        n_pre, T = len(lens), sum(lens)
        nst, n = ns * win, ns * win + T
        pool = ns * win + n_pre + 2
        slots = (torch.randperm(pool - 1)[: ns * win + n_pre] + 1).to(torch.int32)
        cu_cpu = torch.tensor([0] + list(lens), dtype=torch.int32).cumsum(0).to(torch.int32)
        masks = torch.tensor([True] * ns + [False] * n_pre)
        has = torch.tensor(init, device=dev)
        nums_dict, batch_ptr, tco = compute_causal_conv1d_metadata(cu_cpu, device=dev)
        md = GDNAttentionMetadata(
            num_prefills=n_pre, num_prefill_tokens=T, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=ns, num_spec_decode_tokens=nst, num_actual_tokens=n,
            has_initial_state=has,
            spec_query_start_loc=(torch.arange(ns + 1, dtype=torch.int32) * win).to(dev),
            non_spec_query_start_loc=cu_cpu.to(dev),
            spec_state_indices_tensor=slots[: ns * win].view(ns, win).to(dev),
            non_spec_state_indices_tensor=slots[ns * win:].to(dev),
            spec_sequence_masks=masks.to(dev), spec_sequence_masks_cpu=masks,
            num_accepted_tokens=torch.randint(1, win + 1, (ns,), dtype=torch.int32).to(dev),
            chunk_indices=prepare_chunk_indices(cu_cpu, 64).to(dev),
            chunk_offsets=prepare_chunk_offsets(cu_cpu, 64).to(dev),
            prefill_query_start_loc=cu_cpu.to(dev), prefill_state_indices=slots[ns * win:].to(dev),
            prefill_has_initial_state=has, nums_dict=nums_dict, batch_ptr=batch_ptr,
            token_chunk_offset_ptr=tco)
        md.suffix_spec_seq_lens = (torch.randint(10, 3000, (ns,), dtype=torch.int32) + win).to(dev)
        md.suffix_zone, md.suffix_max_prefill = 0, max(lens)
        inputs = []
        for _ in range(layers):
            pages = torch.zeros(pool, page, dtype=torch.int8, device=dev)
            conv, ssm = views(pages)
            conv.normal_()
            ssm.normal_(std=0.1)
            inputs.append((pages, torch.randn(n + tail, qkv_dim + hv * dv, device=dev, dtype=bf16),
                           torch.randn(n + tail, 2 * hv, device=dev, dtype=bf16)))
        return md, inputs

    def run(md, inputs, fast, mode, graph=False):
        global MODE
        gm._FAST, MODE = fast, mode
        vars(md).pop("_suffix_mixed_plan", None)
        bufs = [(p.clone(), q.clone(), b.clone(),
                 torch.full((q.shape[0], hv, dv), float("nan"), device=dev, dtype=bf16))
                for p, q, b in inputs]

        def calls():
            for pages, qkvz, ba, out in bufs:
                layer.kv_cache = views(pages)
                if not gm.forward_mixed(layer, qkvz, ba, out, md):
                    raise RuntimeError("forward_mixed declined the mixed layout")
        if not graph:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            calls()
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            return bufs, (t1 - t0) * 1e6 / layers, (time.perf_counter() - t0) * 1e6 / layers
        calls()  # warm: plan, configs, compiles
        torch.cuda.synchronize()
        vars(md).pop("_suffix_mixed_plan", None)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            calls()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay()
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3 / (5 * layers)

    def same(ref, got, n):
        return all(torch.equal(rp, gp) and torch.equal(ro[:n], go[:n])
                   and bool((go[n:] == 0).all()) and bool((ro[n:] == 0).all())
                   for (rp, _, _, ro), (gp, _, _, go) in zip(ref, got))

    failed = False
    for ns, lens, init in ((31, [200], [False]), (31, [200], [True]), (7, [130, 777], [True, False]),
                           (1, [64], [True]), (31, [2048], [True]), (31, [MAX_T + 52], [True])):
        md, inputs = case(ns, lens, init)
        n = md.num_actual_tokens
        ref, _, _ = run(md, inputs, False, 0)  # stock; its FLA calls fill the autotuners
        capture()
        verdict, host, wall = [], {}, {}
        for mode in (1, 2):
            got, _, _ = run(md, inputs, True, mode)
            ok = same(ref, got, n)
            failed |= not ok
            verdict.append(f"mode {mode} {'bitwise' if ok else 'DIFFERS'}")
        for label, fast, mode in (("stock", False, 0), ("mode1", True, 1), ("mode2", True, 2)):
            samples = [run(md, inputs, fast, mode)[1:] for _ in range(5)]
            host[label] = sorted(s[0] for s in samples)[2]
            wall[label] = sorted(s[1] for s in samples)[2]
        gpu = {label: run(md, inputs, fast, mode, graph=True)
               for label, fast, mode in (("stock", False, 0), ("mode1", True, 1),
                                         ("mode2", True, 2))}
        built = getattr(vars(md).get("_suffix_mixed_plan", (None, None))[1], "fast", None)
        print(f"{MARK} {ns}x{win} verify + prefill {lens} init {init} ({n} tokens, plan "
              f"{'built' if built else 'stock (declined)'}): {', '.join(verdict)} | host us per "
              f"layer call: " + " / ".join(f"{k} {v:.0f}" for k, v in host.items())
              + " | eager wall us per layer: " + " / ".join(f"{k} {v:.0f}" for k, v in wall.items())
              + " | graphed GPU us per layer: " + " / ".join(f"{k} {v:.1f}" for k, v in gpu.items())
              + f" -> {'MATCH' if 'DIFFERS' not in str(verdict) else 'MISMATCH'}", flush=True)
    MODE = int(os.environ.get("SUFFIX_ROCM_GDN_MIXED_FAST", "0") or 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
