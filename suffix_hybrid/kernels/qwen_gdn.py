# SPDX-License-Identifier: Apache-2.0
"""K-GDN1: our fused Gated-DeltaNet decode + MTP-verify kernels for Qwen GDN
hybrid models (Qwen3.5 / qwen3.8-27b-fable, Qwen3-Next, Qwen4Exp), wired into
vLLM 0.30.0 and 0.32.0 through the official out-of-tree layer seam — no vLLM
fork, no `_custom_ops` patching.

Gate: ``SUFFIX_QWEN_GDN=1`` (default OFF). Entry: ``register()``, exported as
the ``vllm.general_plugins`` entry point ``suffix_qwen_gdn``
(``suffix_qwen_gdn_ep-1.0.dist-info``), which vLLM runs in EngineCore and
every worker before the model is built. With the gate off ``register()``
returns immediately: nothing is imported, nothing is registered.

Backend knob ``SUFFIX_HELION_KGDN1`` = oxide (default; Helion never imported)
| helion | auto (docs/helion-kernels.md, suffix_hybrid/helion_kernels/kgdn1.py).

Layer contract (K1; QwenGatedDeltaNetAttention, qwen_gdn_linear_attn.py, the
same in 0.30.0 and 0.32.0). Per TP rank r of tp: H = linear_num_key_heads/tp,
HV = linear_num_value_heads/tp, G = HV/H, K = linear_key_head_dim,
V = linear_value_head_dim (head dims are never split; heads must divide).
  * in_proj_qkvz output [T, 2HK + 2HV*V] bf16 (MergedColumnParallelLinear,
    rows may be padded/strided). Flat layout (Qwen3.5, Qwen4Exp:
    gqa_interleaved_layout=False): [q (H,K) | k (H,K) | v (HV,V) | z (HV,V)],
    every block this rank's head slice. Interleaved (Qwen3-Next): one shard
    of H groups [q_h (K) | k_h (K) | v_h (G,V) | z_h (G,V)];
    fix_query_key_value_ordering de-interleaves it.
  * in_proj_ba [T, 2HV] bf16: flat [b (HV) | a (HV)]; interleaved H groups
    [b_h (G) | a_h (G)]. (disable_tp_for_ba_proj — AWQ/GPTQ/INC Marlin —
    replicates it at TP>1: refused, see layer_contract_violations.)
  * conv1d (causal_conv1d_update, bf16 conv state over 2HK+HV*V channels)
    runs over the de-interleaved [q | k | v] channels; its output is the
    kernel's mixed_qkv. Interleaved layers therefore copy q/k/v into that
    order before the conv — the stock path does the same copy (the conv
    weights and the conv state prefill writes are in that order); z stays a
    strided [T, H, G, V] view, b/a are repacked into one tiny [T, 2HV].
  * recurrent state kv_cache[1]: [S, HV, V, K] (row v, column k), fp32
    (contract), slot pages possibly padded (mamba_utils as_strided).
    NULL_BLOCK_ID 0 (slot <= 0): no state IO, zero output rows.
  * A_log, dt_bias [HV]; RMSNormGated(V, norm_before_gate, no groups, no
    bias, silu|sigmoid) weight [V]; core_attn_out [T, HV, V] bf16 (zeros).
  * non-spec decode (spec_sequence_masks None, no prefill): token t uses
    slot non_spec_state_indices_tensor[t], state updated in place.
  * MTP verify (spec_sequence_masks set, num_prefills == num_decodes == 0 —
    vLLM 0.32 reclassifies non-spec decodes as prefills when spec decodes
    exist; both versions): N = num_spec_decodes sequences, tokens spec_query_start_loc[n] ..
    [n+1], W = num_spec + 1 per-position slots spec_state_indices_tensor
    [N, W]; the recurrence starts from slot [n, num_accepted_tokens[n]-1]
    and writes the state after token t to slot [n, t] (so the next step
    resumes at whatever was accepted; rejected drafts need no rollback).
    Conv: causal_conv1d_update with num_accepted_tokens / query_start_loc /
    max_query_len = W. Cudagraph padding: NULL rows, 0-token sequences.

Per-layer contract (at construction; every violation is a ValueError, fail
loud — the layer never silently stays stock while K-GDN1 is advertised):
  * oxide: TP=1, flat layout, K == V, and the cubin's (H, HV, K) = (16, 48,
    128) (load_prebuilt) — the qwen-fable path, unchanged;
  * helion | auto: any TP whose heads divide, flat or interleaved layout,
    K and V pow2 <= 256 (K != V allowed); not a replicated in_proj_ba;
  * both: HV % H == 0, RMSNormGated variant above, bf16 conv state, fp32
    recurrent state, bf16 model dtype.
Plus, with the gate ON: the native wheel with the oxide op
(``_native.QWEN_GDN_BACKEND == "oxide"``) and SM family 120 — else
RuntimeError at register().

Seam: ``PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")``
(vllm/model_executor/custom_op.py); ``PluggableLayer.__new__`` then
instantiates our subclass wherever the model builds the layer. The subclass
overrides ONE method, ``_forward_core_fused_norm_packed``, and sets vLLM's
routing flag that sends forward_cuda there (0.30.0 ``enable_fused_gdn_decode``,
0.32.0 ``enable_fused_gdn_spec_decode``, which 0.32 only sets with a
speculative config — without it the override would be unreachable; the
lane restores 0.30's routing, whose stock non-K-GDN1 batches run
_forward_core + _rms_norm_gated_cuda as before). Routing by batch shape:
  * non-spec decode -> stock causal_conv1d_update + K-GDN1 decode;
  * MTP verify, sequence bucket with a Helion verify config -> stock conv +
    kgdn1_verify or kgdn1_verify_vt (helion | auto only);
  * everything else (prefill, mixed, unshipped verify buckets) -> vLLM's own
    method (interleaved: de-interleave, then vLLM's _forward_core_fused_norm).
Per Helion bucket the full-tile body serves where its config is shipped,
else the V-tiled one (kgdn1.pick); the layer log names the body per bucket
and the boot oracles replay a CUDA graph per Helion bucket.

Startup kernel-path assertion (K5, log markers, see the dossier §6 protocol):
  ``suffix qwen-gdn K-GDN1 armed``            register() succeeded
  ``suffix qwen-gdn K-GDN1 ACTIVE``           first layer constructed
  ``suffix qwen-gdn K-GDN1 layer <prefix>``   per layer: decode / verify path per bucket
  ``suffix qwen-gdn K-GDN1 prebuilt SASS loaded, ptxas 13.0``  driver-load check (oxide)
  ``suffix qwen-gdn K-GDN1 oracle PASS``      on-GPU identity check at profile run
  ``suffix qwen-gdn K-GDN1 verify oracle PASS``  same for the verify kernel
  ``suffix qwen-gdn K-GDN1 decode-launch``    first real decode launch (graph capture)
  ``suffix qwen-gdn K-GDN1 verify-launch``    first real verify launch
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_QWEN_GDN"
HELION_ENV = "SUFFIX_HELION_KGDN1"
LAYER_NAME = "QwenGatedDeltaNetAttention"
ACT_CODES = {"silu": 0, "swish": 0, "sigmoid": 1}
# Numerics gate (oracle): out is bf16 (1 ulp ~ 7.8e-3 relative); the state
# stays fp32. The kernel reads o in fp32 where the stock chain round-trips
# core_attn_out through bf16, so allow ~2 bf16 ulps of output difference.
OUT_ATOL = 3e-2
OUT_RTOL = 2e-2
STATE_RTOL = 1e-4
STATE_ATOL = 1e-5

_state = {"armed": False, "active_layers": 0, "oracle": None, "voracle": None,
          "launch_logged": False, "vlaunch_logged": False}


def _log(msg: str) -> None:
    print(f"suffix qwen-gdn {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


# ---------------------------------------------------------------------------
# torch reference (vLLM 0.30.0 semantics; the oracle for both the Rust CPU
# reference and the GPU kernel)
# ---------------------------------------------------------------------------
def gdn_decode_fused_torch(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state,
                           state_idx, num_k_heads, scale, norm_eps, act,
                           round_o_bf16=False):
    """Reference of one K-GDN1 step. Mutates ``state`` in place (like the
    kernel) and returns ``out`` [T, HV, V] in fp32.

    Mirrors fused_recurrent.py:256-340 (packed decode, use_qk_l2norm) +
    layernorm_guard.py:74-172 (RMSNormGated, norm_before_gate). With
    ``round_o_bf16=True`` it reproduces the stock two-kernel chain exactly
    (o is stored to bf16 core_attn_out before the norm reads it).
    """
    import torch

    T = mixed_qkv.shape[0]
    S, HV, V, K = state.shape
    H = num_k_heads
    g_ = HV // H
    out = torch.zeros(T, HV, V, dtype=torch.float32, device=mixed_qkv.device)
    x = mixed_qkv.float()
    q_all = x[:, : H * K].view(T, H, K)
    k_all = x[:, H * K: 2 * H * K].view(T, H, K)
    v_all = x[:, 2 * H * K:].view(T, HV, V)
    b_all = ba[:, :HV].float()
    a_all = ba[:, HV:].float()
    zf = z.float().reshape(T, HV, V)
    w = norm_w.float()
    for t in range(T):
        slot = int(state_idx[t])
        if slot <= 0:
            continue
        for hv in range(HV):
            h = hv // g_
            q = q_all[t, h] / torch.sqrt((q_all[t, h] ** 2).sum() + 1e-6) * scale
            k = k_all[t, h] / torch.sqrt((k_all[t, h] ** 2).sum() + 1e-6)
            xg = a_all[t, hv] + dt_bias[hv].float()
            sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
            g = -torch.exp(a_log[hv].float()) * sp
            beta = torch.sigmoid(b_all[t, hv])
            s = state[slot, hv].float() * torch.exp(g)          # [V, K]
            dv = (v_all[t, hv] - s @ k) * beta
            s = s + dv[:, None] * k[None, :]
            o = s @ q
            state[slot, hv] = s.to(state.dtype)
            if round_o_bf16:
                o = o.to(torch.bfloat16).float()
            rstd = torch.rsqrt((o * o).mean() + norm_eps)
            zz = zf[t, hv]
            gate = zz * torch.sigmoid(zz) if act == 0 else torch.sigmoid(zz)
            out[t, hv] = o * rstd * w * gate
    return out


def make_inputs(T, H, HV, K, slots, device="cpu", seed=0, idx=None, page_pad=0, V=None,
                interleaved=False):
    """Random K-GDN1 inputs at a given shape (bf16 activations, fp32 state).
    ``mixed_qkv``/``z`` are row-strided views into one packed qkvz buffer,
    exactly like the in_proj_qkvz split the layer hands the kernel. With
    ``page_pad`` > 0 the state is an as_strided view with a padded per-slot
    page stride, like vLLM's mamba cache (v1/worker/mamba_utils.py:354-357).
    ``interleaved``: the qkvz / ba buffers are Qwen3-Next's grouped layout and
    the kernel inputs come from ``deinterleave`` (what the layer hands over)."""
    import torch

    V = V or K
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qkv_w = 2 * H * K + HV * V
    qkvz = torch.randn(T, qkv_w + HV * V, generator=gen).to(torch.bfloat16)
    ba = torch.randn(T, 2 * HV, generator=gen).to(torch.bfloat16)
    a_log = (torch.rand(HV, generator=gen) * 2 - 1).float()
    dt_bias = torch.randn(HV, generator=gen).float()
    norm_w = (1 + 0.1 * torch.randn(V, generator=gen)).float()
    state = (0.1 * torch.randn(slots, HV, V, K, generator=gen)).float()
    if idx is None:
        idx = [(t % (slots - 1)) + 1 for t in range(T)]  # distinct, >0
    state_idx = torch.tensor(idx, dtype=torch.int32)
    tens = [qkvz, ba, a_log, dt_bias, norm_w, state, state_idx]
    qkvz, ba, a_log, dt_bias, norm_w, state, state_idx = [t.to(device) for t in tens]
    if page_pad:
        page = HV * V * K + page_pad
        raw = torch.zeros(slots * page, dtype=state.dtype, device=state.device)
        view = raw.as_strided((slots, HV, V, K), (page, V * K, K, 1))
        view.copy_(state)
        state = view
    if interleaved:
        mixed_qkv, z, ba = deinterleave(qkvz, ba, H, HV // H, K, V)
    else:
        mixed_qkv = qkvz[:, :qkv_w]
        z = qkvz[:, qkv_w:].view(T, HV, V)
    return dict(mixed_qkv=mixed_qkv, z=z, ba=ba, a_log=a_log, dt_bias=dt_bias,
                norm_w=norm_w, state=state, state_idx=state_idx)


def make_verify_inputs(N, W, H, HV, K, V=None, device="cpu", seed=0, lens=None,
                       accepted=None, page_pad=0, interleaved=False, pad_tokens=0):
    """Random MTP-verify inputs: N sequences of ``lens`` (default random 1..W)
    tokens (+ ``pad_tokens`` rows no sequence covers, like cudagraph token
    padding), per-position slots state_idx [N, W] (distinct, shuffled),
    cu [N+1], acc [N] (default random 1..W). A 0 in ``lens`` makes that
    sequence a padded one (NULL slot row, like the metadata builder's)."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    if lens is None:
        lens = (torch.randint(1, W + 1, (N,), generator=gen)).tolist()
    if accepted is None:
        accepted = (torch.randint(1, W + 1, (N,), generator=gen)).tolist()
    T = sum(lens) + pad_tokens
    slots = N * W + 1
    inp = make_inputs(T, H, HV, K, slots, device=device, seed=seed, idx=[0] * T,
                      page_pad=page_pad, V=V, interleaved=interleaved)
    state_idx = (torch.randperm(N * W, generator=gen) + 1).view(N, W).to(torch.int32)
    for n, ln in enumerate(lens):
        if ln == 0:
            state_idx[n] = 0
    cu = torch.zeros(N + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    acc = torch.tensor(accepted, dtype=torch.int32)
    inp.update(state_idx=state_idx.to(device), cu=cu.to(device), acc=acc.to(device))
    return inp


def deinterleave(mixed_qkvz, ba, H, G, K, V):
    """Qwen3-Next's grouped per-rank projections -> the kernel's inputs:
    mixed_qkvz [T, H*(2K+2GV)], ba [T, H*2G] ->
      mixed_qkv [T, 2HK+HGV] contiguous, [q | k | v] — exactly
        fix_query_key_value_ordering + the "l p d -> l (p d)" cat of vLLM's
        forward_cuda (the conv1d channel order, so this copy is the one the
        stock path makes too; the conv state prefill writes is in it);
      z [T, H, G, V] strided view (no copy: the kernel takes 4-D z);
      ba [T, 2HG] = [b | a], b[:, h*G+g] = b_h[g] (one tiny repack copy, the
        packed-ba signature the oxide op and both bodies share)."""
    import torch

    T = mixed_qkvz.shape[0]
    x = mixed_qkvz.view(T, H, 2 * K + 2 * G * V)
    qkv = torch.empty(T, 2 * H * K + H * G * V, dtype=x.dtype, device=x.device)
    qkv[:, :H * K].view(T, H, K).copy_(x[..., :K])
    qkv[:, H * K:2 * H * K].view(T, H, K).copy_(x[..., K:2 * K])
    qkv[:, 2 * H * K:].view(T, H, G * V).copy_(x[..., 2 * K:2 * K + G * V])
    z = x[..., 2 * K + G * V:].unflatten(-1, (G, V))
    ba = ba.view(T, H, 2, G).transpose(1, 2).reshape(T, 2 * H * G)
    return qkv, z, ba


# ---------------------------------------------------------------------------
# on-GPU oracle (runs once per process at the profile run, before capture)
# ---------------------------------------------------------------------------
def run_gpu_oracle(native, H, HV, K, act, norm_eps, row_stride=None, op=None, extra_T=(),
                   V=None, interleaved=False):
    """JIT/load the kernel at the serving shape and check it against the
    torch reference on the GPU, plus a CUDA-graph capture/replay round.
    Raises RuntimeError on any mismatch (the gate is fail-closed).
    ``op`` = the lane's launcher (decode_op; default the oxide op); every T in
    ``extra_T`` (its Helion buckets) runs too, compiling it before capture —
    plus bucket-1 above 16: Triton specializes int args (T) on %16.
    ``V`` / ``interleaved``: the layer's value head dim and qkvz layout."""
    import torch

    op = op or native.gdn_decode_fused_cuda
    V = V or K
    dev = torch.device("cuda", torch.cuda.current_device())
    worst = {"out": 0.0, "state": 0.0}
    cases = [(1, [1]), (3, [2, 0, 5]), (16, None)]
    # 23 live distinct slots (the oracle pool), NULL rows beyond
    cases += [(T, [t + 1 if t < 23 else 0 for t in range(T)])
              for b in extra_T for T in ((b - 1, b) if b > 16 else (b,))]
    for T, idx in cases:
        inp = make_inputs(T, H, HV, K, slots=24, device=dev, seed=T, idx=idx,
                          page_pad=4096, V=V, interleaved=interleaved)
        if row_stride is not None and row_stride != inp["mixed_qkv"].stride(0):
            raise RuntimeError(
                f"K-GDN1 oracle: layer row stride {row_stride} != oracle "
                f"{inp['mixed_qkv'].stride(0)} (JIT specialization would differ)")
        ref_state = inp["state"].clone()
        ref = gdn_decode_fused_torch(
            inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], ref_state, inp["state_idx"].cpu(), H, K ** -0.5,
            norm_eps, act, round_o_bf16=True)
        out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
        op(
            inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], out, H, K ** -0.5,
            norm_eps, act, torch.cuda.current_stream(dev).cuda_stream)
        torch.cuda.synchronize(dev)
        _check(out.float(), ref, inp["state"], ref_state, worst, f"T={T}")

    # CUDA-graph capture/replay: the kernel must be capturable (no sync, no
    # alloc on the launch path) and replay must match an eager launch. Once
    # per Helion bucket (each may serve on another body: full tile / V-tiled
    # + norm launch), plus T=4 (the fallback when no bucket is Helion).
    for T in sorted({4, *extra_T}):
        idx = [t + 1 if t < 23 else 0 for t in range(T)]
        inp = make_inputs(T, H, HV, K, slots=24, device=dev, seed=99, idx=idx, page_pad=4096,
                          V=V, interleaved=interleaved)
        s_eager = inp["state"].clone()
        out_eager = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
        out_graph = torch.empty_like(out_eager)
        args = [inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
                inp["norm_w"]]
        op(*args, s_eager, inp["state_idx"], out_eager, H, K ** -0.5, norm_eps, act,
           torch.cuda.current_stream(dev).cuda_stream)
        s_graph = inp["state"].clone()
        s_graph0 = s_graph.clone()
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph, stream=side):
                op(*args, s_graph, inp["state_idx"], out_graph, H, K ** -0.5, norm_eps, act,
                   side.cuda_stream)
        torch.cuda.current_stream(dev).wait_stream(side)
        s_graph.copy_(s_graph0)  # capture may or may not have executed the kernel
        graph.replay()
        torch.cuda.synchronize(dev)
        if not torch.equal(out_graph, out_eager) or not torch.equal(s_graph, s_eager):
            raise RuntimeError(f"K-GDN1 oracle FAIL: CUDA-graph replay != eager launch (T={T})")
    return worst


def run_gpu_verify_oracle(op, H, HV, K, V, W, act, norm_eps, buckets, interleaved=False,
                          device=None):
    """The verify kernel at the serving shape vs the exact eager verify
    reference (kgdn1.verify_reference): ragged lengths, every accepted count,
    and from N > 3 an out-of-range accepted count, a NULL-source sequence, a
    padded 0-token sequence, a NULL destination slot, token padding. Every
    shipped sequence bucket (and bucket-1 above 16) compiles here, before
    capture; then a CUDA-graph capture/replay round (CUDA only; ``device``
    = a CPU device runs the checks alone, for tests). RuntimeError on any
    mismatch."""
    import torch

    from suffix_hybrid.helion_kernels import kgdn1

    dev = device or torch.device("cuda", torch.cuda.current_device())
    cuda = dev.type == "cuda"
    worst = {"out": 0.0, "state": 0.0}
    scale = K ** -0.5

    def inputs(N, seed):
        lens = [(n % W) + 1 for n in range(N)]
        acc = [((n * 7) % W) + 1 for n in range(N)]
        if N > 2:
            lens[-1] = 0                    # padded sequence (NULL row, 0 tokens)
            acc[1] = W + 1                  # out-of-range accepted -> zero rows
        inp = make_verify_inputs(N, W, H, HV, K, V=V, device=dev, seed=seed, lens=lens,
                                 accepted=acc, page_pad=4096, interleaved=interleaved,
                                 pad_tokens=3)
        if N > 3:
            inp["state_idx"][2, acc[2] - 1] = 0  # NULL source: zero rows, no state IO
            inp["state_idx"][3, 0] = 0           # NULL destination: that state is dropped
        T = inp["mixed_qkv"].shape[0]
        out = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=dev)
        return [inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
                inp["norm_w"], inp["state"], inp["state_idx"], inp["cu"], inp["acc"], out]

    # only sizes whose bucket is shipped (op = the Helion launcher, KeyError otherwise)
    sizes = sorted({N for b in buckets for N in ((b - 1, b) if b > 16 else (b,))})
    for N in sizes:
        args = inputs(N, N)
        ref_out, ref_state = kgdn1.verify_reference(
            *kgdn1.verify_body_args(*args, H), scale, norm_eps, act)
        op(*args, H, scale, norm_eps, act)
        if cuda:
            torch.cuda.synchronize(dev)
        _check(args[10].float(), ref_out.float().view(args[10].shape), args[6],
               ref_state.view(args[6].shape), worst, f"verify N={N}")
    if not cuda:
        return worst

    for N in sorted(set(buckets)):  # one graph round per bucket: each may serve on its own body
        args = inputs(N, 99)
        s0 = args[6].clone()
        op(*args, H, scale, norm_eps, act)
        out_eager, s_eager = args[10].clone(), args[6].clone()
        args[6].copy_(s0)
        args[10].zero_()
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph, stream=side):
                op(*args, H, scale, norm_eps, act)
        torch.cuda.current_stream(dev).wait_stream(side)
        args[6].copy_(s0)
        args[10].zero_()
        graph.replay()
        torch.cuda.synchronize(dev)
        if not torch.equal(args[10], out_eager) or not torch.equal(args[6], s_eager):
            raise RuntimeError(f"K-GDN1 verify oracle FAIL: CUDA-graph replay != eager launch "
                               f"(N={N})")
    return worst


def _check(out, ref, state, ref_state, worst, tag):
    import torch

    d_out = (out - ref).abs()
    bad = d_out > (OUT_ATOL + OUT_RTOL * ref.abs())
    d_state = (state - ref_state).abs()
    bad_s = d_state > (STATE_ATOL + STATE_RTOL * ref_state.abs())
    worst["out"] = max(worst["out"], float(d_out.max()))
    worst["state"] = max(worst["state"], float(d_state.max()))
    if bool(bad.any()) or bool(bad_s.any()):
        raise RuntimeError(
            f"K-GDN1 oracle FAIL ({tag}): out max|d|={float(d_out.max()):.3e} "
            f"({int(bad.sum())} bad), state max|d|={float(d_state.max()):.3e} "
            f"({int(bad_s.sum())} bad)")
    if not torch.isfinite(out).all():
        raise RuntimeError(f"K-GDN1 oracle FAIL ({tag}): non-finite output")


# ---------------------------------------------------------------------------
# prebuilt sm_120 SASS (cuda-oxide track, docs/oxide-kernels.md): the cubin
# kernels-oxide/kgdn1 -> ptxas 13.0 ships in suffix_hybrid/oxide_cubins/;
# pods only driver-load it (no JIT of any kind).
# ---------------------------------------------------------------------------
OXIDE_FAMILY = "kgdn1"
OXIDE_SHAPE = (16, 48, 128)  # (H, HV, K) the shipped cubin is built for


def oxide_manifest_entry() -> dict:
    """The kgdn1 entry of the bundle's oxide manifest (RuntimeError if the
    bundle lacks it — checked at register(), before any model build)."""
    from suffix_hybrid import oxide_kernels

    man = oxide_kernels.manifest()
    for k in man["kernels"]:
        if k["name"] == OXIDE_FAMILY:
            return dict(k, ptxas_version=man.get("ptxas_version", "?"))
    raise RuntimeError(f"{GATE}=1 but the oxide manifest has no {OXIDE_FAMILY!r} "
                       "cubin (bundle built without kernels-oxide/kgdn1?)")


def load_prebuilt(H, HV, K, device=None) -> str:
    """Driver-load the kgdn1 SASS on this device (boot driver-load check;
    idempotent). Returns the marker suffix for the log lines."""
    if (H, HV, K) != OXIDE_SHAPE:
        raise ValueError(f"K-GDN1 SASS is built for (H, HV, K)={OXIDE_SHAPE}, "
                         f"layer has {(H, HV, K)}")
    from suffix_hybrid import oxide_kernels

    ent = oxide_manifest_entry()
    oxide_kernels.ensure_loaded(OXIDE_FAMILY, device)
    tag = (f"prebuilt SASS loaded, ptxas {ent['ptxas_version']} "
           f"(PTX ISA {ent.get('ptx_isa', '?')}, sha256 {ent['sha256'][:12]})")
    if not _state.get("sass_logged"):
        _state["sass_logged"] = True
        _log(f"K-GDN1 {tag}")
    return tag


def decode_op(native, H, HV, K, V=None, layout="flat"):
    """(op, helion_buckets): the K-GDN1 launcher for this layer shape under
    SUFFIX_HELION_KGDN1. oxide -> the native op itself and Helion is never
    imported; helion|auto -> helion_kernels.kgdn1.decode_op (a missing config
    under helion raises here, at load)."""
    from suffix_hybrid import helion_kernels
    if helion_kernels.backend(HELION_ENV) == "oxide":
        return native.gdn_decode_fused_cuda, ()
    from suffix_hybrid.helion_kernels import kgdn1
    return kgdn1.decode_op(native, H, HV, K, V or K, layout)


def stock_decode_fused(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                       num_k_heads, scale, norm_eps, act):
    """vLLM's stock chain for the same step, into ``out`` [T, HV, V] bf16:
    ba.chunk -> 2x .contiguous() + FLA packed recurrent decode + FLA
    layer_norm_fwd (qwen_gdn_linear_attn.py:1657-1709, 1842-1874, 1912-1913).
    The bench's and the Helion lab's reference point."""
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd

    V = state.shape[2]
    b, a = ba.chunk(2, dim=-1)
    b, a = b.contiguous(), a.contiguous()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=a_log, dt_bias=dt_bias, scale=scale,
        initial_state=state, out=out.unsqueeze(1), ssm_state_indices=state_idx,
        use_qk_l2norm_in_kernel=True)
    x2 = out.reshape(-1, V)
    layer_norm_fwd(x2, norm_w, None, norm_eps, z=z.reshape(-1, V), out=x2, group_size=V,
                   norm_before_gate=True, is_rms_norm=True,
                   activation="silu" if act == 0 else "sigmoid")


def stock_verify_fused(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, cu, acc,
                       out, num_k_heads, scale, norm_eps, act):
    """vLLM's spec-verify step after the conv, into ``out`` [T, HV, V] bf16:
    the fused CUDA MTP kernel where vLLM takes it (_can_use_fused_gdn_mtp_decode:
    K == V == 128, HV/H in 1,2,3,4,8, W <= 8, op built ->
    _forward_core_decode_spec_post_conv_fused_norm), else the _forward_core
    spec branch (FLA fused_sigmoid_gating_delta_rule_update) +
    _rms_norm_gated_cuda. The lab's verify reference point."""
    import torch
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd

    S, HV, V, K = state.shape
    H = num_k_heads
    b, a = ba[:, :HV], ba[:, HV:2 * HV]
    gate = "silu" if act == 0 else "sigmoid"
    if (K == V == 128 and HV // H in (1, 2, 3, 4, 8) and state_idx.shape[1] <= 8
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")):
        from vllm import _custom_ops as ops
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=a_log, dt_bias=dt_bias,
            state_indices=state_idx, cu_seqlens=cu, num_accepted_tokens=acc, state=state,
            output_gate=z.reshape(-1, HV, V), norm_weight=norm_w, out=out, scale=scale,
            norm_eps=norm_eps, output_gate_activation=gate)
        return
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    T = mixed_qkv.shape[0]
    q = mixed_qkv[:, :H * K].reshape(1, T, H, K)
    k = mixed_qkv[:, H * K:2 * H * K].reshape(1, T, H, K)
    v = mixed_qkv[:, 2 * H * K:2 * H * K + HV * V].reshape(1, T, HV, V)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log, a=a.contiguous(), b=b.contiguous(), dt_bias=dt_bias, q=q, k=k, v=v,
        scale=scale, initial_state=state, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=state_idx, num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True)
    out.copy_(o.reshape(T, HV, V))
    x2 = out.reshape(-1, V)
    layer_norm_fwd(x2, norm_w, None, norm_eps, z=z.reshape(-1, V), out=x2, group_size=V,
                   norm_before_gate=True, is_rms_norm=True, activation=gate)


# ---------------------------------------------------------------------------
# vLLM wiring
# ---------------------------------------------------------------------------
def _native():
    from suffix_hybrid import _native as native
    if (not getattr(native, "HAS_QWEN_GDN_CUDA", False)
            or getattr(native, "QWEN_GDN_BACKEND", "none") != "oxide"):
        raise RuntimeError(
            f"{GATE}=1 but suffix_hybrid._native has no cuda-oxide K-GDN1 op "
            "(needs cargo feature `oxide-kernels`; the cutile build cannot load "
            "on CUDA 13.0 nodes) — refusing to serve the stock GDN chain while "
            "the K-GDN1 gate is armed")
    return native


def check_sm120() -> tuple:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{GATE}=1 requires a CUDA device (none visible)")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 12:
        raise RuntimeError(
            f"{GATE}=1: K-GDN1 is built for SM family 120 only (device cc "
            f"{cap[0]}.{cap[1]}); refusing to run on another arch")
    return cap


# vLLM source anchors the lane relies on (pinned, refuse on drift). The
# routing flag forward_cuda tests before calling the fused-norm-packed op
# (our override's only caller) was renamed and narrowed in 0.32.0.
ROUTE_FLAGS = {"enable_fused_gdn_decode": "0.30.0", "enable_fused_gdn_spec_decode": "0.32.0"}
FUSED_PACKED_OP = "qwen_gdn_attention_core_fused_norm_packed"
ANCHOR_SIGS = {
    "_forward_core_fused_norm_packed": ["self", "mixed_qkvz", "ba", "core_attn_out"],
    "_forward_core_fused_norm": ["self", "mixed_qkv", "b", "a", "output_gate", "core_attn_out"],
}
MD_FIELDS = ("spec_sequence_masks", "num_prefills", "num_decodes", "num_actual_tokens",
             "non_spec_state_indices_tensor", "num_spec_decodes", "spec_state_indices_tensor",
             "spec_query_start_loc", "num_accepted_tokens")


def anchor_flag(forward_cuda_src: str, sigs: dict, md_fields) -> str:
    """Check the pinned anchors and return the routing flag of this vLLM:
    ``forward_cuda_src`` = QwenGatedDeltaNetAttention.forward_cuda source,
    ``sigs`` = {method: parameter names}, ``md_fields`` = GDNAttentionMetadata
    field names. RuntimeError on any drift (the gate refuses to arm)."""
    import re
    for name, want in ANCHOR_SIGS.items():
        if sigs.get(name) != want:
            raise RuntimeError(f"K-GDN1 anchor drift: {name}{sigs.get(name)} != {want}")
    missing = sorted(set(MD_FIELDS) - set(md_fields))
    if missing:
        raise RuntimeError(f"K-GDN1 anchor drift: GDNAttentionMetadata lacks {missing}")
    hits = [f for f in ROUTE_FLAGS if re.search(rf"\bself\.{f}\b", forward_cuda_src)]
    if len(hits) != 1 or FUSED_PACKED_OP not in forward_cuda_src:
        raise RuntimeError(f"K-GDN1 anchor drift: forward_cuda reaches {FUSED_PACKED_OP} via "
                           f"{hits or 'no known flag'} (known: {sorted(ROUTE_FLAGS)})")
    return hits[0]


def layer_contract_violations(layer, backend="oxide") -> list:
    """Every reason this GDN layer cannot run K-GDN1 under ``backend``
    (SUFFIX_HELION_KGDN1 value; empty = OK)."""
    import torch
    why = []
    tp = layer.tp_size
    k, v = layer.head_k_dim, layer.head_v_dim
    if backend == "oxide":
        if tp != 1:
            why.append(f"tp_size={tp} (the oxide K-GDN1 cubin is TP=1; "
                       f"{HELION_ENV}=helion|auto serves TP>1)")
        if layer.gqa_interleaved_layout:
            why.append(f"interleaved (Qwen3-Next) qkvz layout (the oxide cubin is flat-only; "
                       f"{HELION_ENV}=helion|auto serves it)")
        if k != v:
            why.append(f"K={k} != V={v}")
    else:
        if layer.num_k_heads % tp or layer.num_v_heads % tp:
            why.append(f"H={layer.num_k_heads} / HV={layer.num_v_heads} heads do not divide "
                       f"tp_size={tp}")
        if tp > 1 and getattr(layer, "disable_tp_for_ba_proj", False):
            why.append("replicated in_proj_ba at TP>1 (AWQ/GPTQ/INC Marlin: b/a are not the "
                       "rank's packed [b | a])")
    for name, d in (("K", k), ("V", v)):
        if d & (d - 1) or d > 256:
            why.append(f"{name}={d} not a power of two <= 256")
    if layer.num_v_heads % layer.num_k_heads:
        why.append("HV not a multiple of H")
    norm = layer.norm
    if norm.activation not in ACT_CODES:
        why.append(f"gate activation {norm.activation!r}")
    if not norm.norm_before_gate or norm.group_size is not None or norm.bias is not None:
        why.append("RMSNormGated variant (needs norm_before_gate, no groups, no bias)")
    conv_dt, ssm_dt = layer.get_state_dtype()
    if conv_dt != torch.bfloat16:
        why.append(f"conv state dtype {conv_dt} (needs bf16: conv output feeds the kernel in place)")
    if ssm_dt != torch.float32:
        why.append(f"recurrent state dtype {ssm_dt} (K-GDN1 v1 is fp32-state)")
    if layer.model_config.dtype != torch.bfloat16:
        why.append(f"model dtype {layer.model_config.dtype} (needs bf16)")
    return why


def _make_layer_cls(native):
    import dataclasses
    import inspect

    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from suffix_hybrid import helion_kernels as hk

    Q = QwenGatedDeltaNetAttention
    route_flag = anchor_flag(
        inspect.getsource(Q.forward_cuda),
        {m: list(inspect.signature(getattr(Q, m)).parameters) for m in ANCHOR_SIGS},
        [f.name for f in dataclasses.fields(GDNAttentionMetadata)])
    mode = hk.backend(HELION_ENV)

    class SuffixQwenGDN(QwenGatedDeltaNetAttention):
        """QwenGatedDeltaNetAttention + K-GDN1 on the non-spec decode and
        (helion|auto) the MTP verify step."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            why = layer_contract_violations(self, mode)
            if why:
                raise ValueError(f"{GATE}=1 but layer {self.prefix} violates the "
                                 f"K-GDN1 contract ({HELION_ENV}={mode}): {'; '.join(why)}")
            tp = self.tp_size
            H, HV = self.num_k_heads // tp, self.num_v_heads // tp
            K, V = self.head_k_dim, self.head_v_dim
            W = self.num_spec + 1 if self.num_spec else 0
            layout = "interleaved" if self.gqa_interleaved_layout else "flat"
            self._kgdn_dims = (H, HV, K, V)
            self._kgdn_act = ACT_CODES[self.norm.activation]
            self._kgdn_params = None
            self._kgdn_sass = ""
            if mode == "oxide":
                fallback, self._kgdn_vop, self._kgdn_vbuckets = "oxide", None, ()
                served = ({}, {})
            else:
                from suffix_hybrid.helion_kernels import kgdn1
                fits = kgdn1.oxide_fits(H, HV, K, V, layout)
                fallback = "oxide" if fits else "vLLM stock FLA chain"
                self._kgdn_vop, self._kgdn_vbuckets = kgdn1.verify_op(H, HV, K, V, W)
                self._kgdn_buckets = kgdn1.BUCKETS
                served = (kgdn1.served(H, HV, K, V), kgdn1.served(H, HV, K, V, W))
            self._kgdn_op, self._kgdn_helion = decode_op(native, H, HV, K, V, layout)
            self._kgdn_fallback = fallback
            if fallback == "oxide" and (mode == "oxide"
                                        or len(self._kgdn_helion) < len(self._kgdn_buckets)):
                self._kgdn_sass = load_prebuilt(H, HV, K)
            # forward_cuda reaches our override only through this flag; 0.32
            # sets it only with a speculative config (anchor_flag).
            setattr(self, route_flag, True)
            _state["active_layers"] += 1
            if _state["active_layers"] == 1:
                _log(f"K-GDN1 ACTIVE: {LAYER_NAME} -> SuffixQwenGDN (first layer "
                     f"{self.prefix}; H={H} HV={HV} K={K} V={V} tp={tp} {layout} "
                     f"act={self.norm.activation}; helion buckets "
                     f"{list(self._kgdn_helion) or 'none'})")
            verify = ("n/a (no spec decode)" if not W else
                      f"helion@{served[1]} seqs (w={W}), else vLLM stock"
                      if self._kgdn_vbuckets else f"vLLM stock (w={W})")
            _log(f"K-GDN1 layer {self.prefix}: {HELION_ENV}={mode} tp={tp} H={H} HV={HV} "
                 f"K={K} V={V} {layout}; decode helion@{served[0] or '[]'} else "
                 f"{fallback}; verify {verify}; route flag {route_flag}")

        def _kgdn_f32(self):
            # fp32 copies of the per-head params, made once (outside capture:
            # the profile run calls this before any graph is recorded).
            if self._kgdn_params is None:
                self._kgdn_params = (
                    self.A_log.detach().float().contiguous(),
                    self.dt_bias.detach().float().contiguous(),
                    self.norm.weight.detach().float().contiguous(),
                )
            return self._kgdn_params

        def _kgdn_conv(self):
            conv_state = (self.kv_cache[0] if is_conv_state_dim_first()
                          else self.kv_cache[0].transpose(-1, -2))
            conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                             self.conv1d.weight.size(2))
            return conv_state, conv_w

        def _kgdn_stock(self, mixed_qkvz, ba, qkv, z, ba_p, core_attn_out):
            if not self.gqa_interleaved_layout:
                return super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)
            HV, V = self._kgdn_dims[1], self._kgdn_dims[3]
            return self._forward_core_fused_norm(
                mixed_qkv=qkv, b=ba_p[:, :HV], a=ba_p[:, HV:],
                output_gate=z.reshape(z.shape[0], HV, V), core_attn_out=core_attn_out)

        def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
            H, HV, K, V = self._kgdn_dims
            if self.gqa_interleaved_layout:
                qkv, z, ba_p = deinterleave(mixed_qkvz, ba, H, HV // H, K, V)
            else:
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                qkv = mixed_qkvz[:, :qkv_size]
                z = mixed_qkvz[:, qkv_size:].unflatten(-1, (-1, V))
                ba_p = ba
            md = get_forward_context().attn_metadata
            md = md.get(self.prefix) if isinstance(md, dict) else None
            if md is None:
                # profile / dummy run: pre-JIT + on-GPU oracles before capture.
                self._kgdn_f32()
                inter = self.gqa_interleaved_layout
                if _state["oracle"] is None:
                    _state["oracle"] = run_gpu_oracle(
                        native, H, HV, K, self._kgdn_act, self.layer_norm_epsilon,
                        row_stride=None if inter else mixed_qkvz.stride(0), op=self._kgdn_op,
                        extra_T=self._kgdn_helion, V=V, interleaved=inter)
                    _log(f"K-GDN1 oracle PASS max|d_out|={_state['oracle']['out']:.3e} "
                         f"max|d_state|={_state['oracle']['state']:.3e} (+graph replay) "
                         f"{self._kgdn_sass or 'helion/stock (no oxide SASS)'}")
                if self._kgdn_vop is not None and _state["voracle"] is None:
                    _state["voracle"] = run_gpu_verify_oracle(
                        self._kgdn_vop, H, HV, K, V, self.num_spec + 1, self._kgdn_act,
                        self.layer_norm_epsilon, self._kgdn_vbuckets, interleaved=inter)
                    _log(f"K-GDN1 verify oracle PASS max|d_out|="
                         f"{_state['voracle']['out']:.3e} max|d_state|="
                         f"{_state['voracle']['state']:.3e} (+graph replay) seq buckets "
                         f"{list(self._kgdn_vbuckets)}")
                return self._kgdn_stock(mixed_qkvz, ba, qkv, z, ba_p, core_attn_out)
            spec = md.spec_sequence_masks is not None
            if not spec and md.num_prefills == 0 and md.num_decodes > 0:
                return self._kgdn_decode(md, qkv, z, ba_p, core_attn_out)
            if (spec and self._kgdn_vop is not None and md.num_prefills == 0
                    and md.num_decodes == 0 and md.num_spec_decodes > 0
                    and hk.bucket(md.num_spec_decodes, self._kgdn_buckets) in self._kgdn_vbuckets):
                return self._kgdn_verify(md, qkv, z, ba_p, core_attn_out)
            return self._kgdn_stock(mixed_qkvz, ba, qkv, z, ba_p, core_attn_out)

        def _kgdn_decode(self, md, qkv, z, ba, core_attn_out):
            n = md.num_actual_tokens
            idx = md.non_spec_state_indices_tensor[:n]
            conv_state, conv_w = self._kgdn_conv()
            mixed_qkv = causal_conv1d_update(
                qkv[:n], conv_state, conv_w, self.conv1d.bias,
                self.activation, conv_state_indices=idx, validate_data=False)
            a_log, dt_bias, norm_w = self._kgdn_f32()
            self._kgdn_op(
                mixed_qkv, z[:n], ba[:n], a_log, dt_bias, norm_w, self.kv_cache[1], idx,
                core_attn_out[:n], self._kgdn_dims[0], self.head_k_dim ** -0.5,
                self.layer_norm_epsilon, self._kgdn_act,
                torch.cuda.current_stream().cuda_stream)
            if not _state["launch_logged"]:
                _state["launch_logged"] = True
                path = ("helion" if self._kgdn_helion and hk.bucket(n, self._kgdn_buckets)
                        in self._kgdn_helion else self._kgdn_fallback)
                _log(f"K-GDN1 decode-launch: first non-spec decode batch T={n} "
                     f"({self.prefix}) on {path}")

        def _kgdn_verify(self, md, qkv, z, ba, core_attn_out):
            N = md.num_spec_decodes
            n = md.num_actual_tokens
            idx = md.spec_state_indices_tensor[:N]
            cu = md.spec_query_start_loc[:N + 1]
            acc = md.num_accepted_tokens[:N]
            conv_state, conv_w = self._kgdn_conv()
            # = _forward_core_decode_spec_fused_norm's conv call
            mixed_qkv = causal_conv1d_update(
                qkv[:n], conv_state, conv_w, self.conv1d.bias, self.activation,
                conv_state_indices=idx[:, 0], num_accepted_tokens=acc, query_start_loc=cu,
                max_query_len=idx.size(1), validate_data=False)
            a_log, dt_bias, norm_w = self._kgdn_f32()
            self._kgdn_vop(
                mixed_qkv, z[:n], ba[:n], a_log, dt_bias, norm_w, self.kv_cache[1], idx, cu,
                acc, core_attn_out[:n], self._kgdn_dims[0], self.head_k_dim ** -0.5,
                self.layer_norm_epsilon, self._kgdn_act)
            if not _state["vlaunch_logged"]:
                _state["vlaunch_logged"] = True
                _log(f"K-GDN1 verify-launch: first MTP verify batch N={N} T={n} "
                     f"({self.prefix}); vLLM spec recurrence + norm replaced")

    return SuffixQwenGDN


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_QWEN_GDN=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    native = _native()
    cap = check_sm120()
    oxide_manifest_entry()  # fail closed now if the bundle lacks the SASS
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    if LAYER_NAME not in op_registry_oot:
        PluggableLayer.register_oot(_make_layer_cls(native), name=LAYER_NAME)
    _state["armed"] = True
    _log(f"K-GDN1 armed: {LAYER_NAME} OOT-registered (cc {cap[0]}.{cap[1]})")
    return _state


# ---------------------------------------------------------------------------
# on-silicon CLI: `python -m suffix_hybrid.kernels.qwen_gdn {oracle,bench}`
# ---------------------------------------------------------------------------
def bench(Ts=(1, 4, 8, 16), layers=48, iters=50, H=16, HV=48, K=128, eps=1e-6):
    """Per-decode-step GPU time of the 48-layer GDN non-GEMM chain, stock vs
    K-GDN1, each captured in ONE CUDA graph (= how decode replays in vLLM).
    Stock = stock_decode_fused; K-GDN1 = decode_op (SUFFIX_HELION_KGDN1 picks
    oxide | helion | auto). causal_conv1d_update is identical in both arms and
    excluded. Returns {T: (stock_us, kgdn1_us)}."""
    import torch

    native = _native()
    check_sm120()
    load_prebuilt(H, HV, K)
    op, _ = decode_op(native, H, HV, K)
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    for T in Ts:
        inp = make_inputs(T, H, HV, K, slots=T + 1, device=dev, seed=T)
        dt_bf16 = inp["dt_bias"].to(torch.bfloat16)  # stock param dtype
        core = torch.zeros(T, HV, K, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(core)
        scale = K ** -0.5

        def stock():
            stock_decode_fused(inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"],
                               dt_bf16, inp["norm_w"], inp["state"], inp["state_idx"],
                               core, H, scale, eps, 0)

        def ours():
            op(
                inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
                inp["norm_w"], inp["state"], inp["state_idx"], out, H, scale, eps, 0,
                torch.cuda.current_stream(dev).cuda_stream)

        times = []
        for fn in (stock, ours):
            fn()  # JIT / autotune outside capture
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(layers):
                    fn()
            g.replay()
            torch.cuda.synchronize(dev)
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                g.replay()
            t1.record()
            torch.cuda.synchronize(dev)
            times.append(t0.elapsed_time(t1) * 1000.0 / iters)
        res[T] = tuple(times)
        _log(f"K-GDN1 bench T={T}: stock {times[0]:.1f} us/step, K-GDN1 "
             f"{times[1]:.1f} us/step ({layers} layers) ratio {times[1] / times[0]:.3f}")
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m suffix_hybrid.kernels.qwen_gdn")
    ap.add_argument("cmd", choices=["oracle", "bench"])
    args = ap.parse_args(argv)
    if args.cmd == "oracle":
        native = _native()
        check_sm120()
        tag = load_prebuilt(16, 48, 128)
        op, extra = decode_op(native, 16, 48, 128)
        for act in (0, 1):
            w = run_gpu_oracle(native, 16, 48, 128, act, 1e-6, op=op, extra_T=extra)
            _log(f"K-GDN1 oracle PASS act={act} max|d_out|={w['out']:.3e} "
                 f"max|d_state|={w['state']:.3e} (+graph replay) {tag}")
    else:
        bench()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
