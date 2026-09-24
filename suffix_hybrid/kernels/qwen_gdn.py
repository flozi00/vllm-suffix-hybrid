# SPDX-License-Identifier: Apache-2.0
"""K-GDN1: our fused Gated-DeltaNet decode kernel for Qwen3.5-class hybrid
models (qwen3.8-27b-fable-distill), wired into vLLM 0.30.0 through its
official out-of-tree layer seam — no vLLM fork, no `_custom_ops` patching.

Gate: ``SUFFIX_QWEN_GDN=1`` (default OFF). Entry: ``register()``, exported as
the ``vllm.general_plugins`` entry point ``suffix_qwen_gdn``
(``suffix_qwen_gdn_ep-1.0.dist-info``), which vLLM runs in EngineCore and
every worker before the model is built. With the gate off ``register()``
returns immediately: nothing is imported, nothing is registered.

With the gate ON every precondition is a HARD error (fail loud, never a
silent fallback to the stock Triton chain while advertising K-GDN1):
  * native wheel built with cargo feature ``qwen-gdn-kernels``
    (``_native.HAS_QWEN_GDN_CUDA``) — else RuntimeError;
  * a CUDA device of SM family 120 — else RuntimeError;
  * per layer (at construction): TP=1, non-interleaved Qwen3.5 layout,
    K == V (pow2 <= 256), RMSNormGated(norm_before_gate, no bias, no groups,
    silu|sigmoid), fp32 recurrent state, bf16 activations — else ValueError.

Seam: ``PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")``
(vllm/model_executor/custom_op.py:84-100); ``PluggableLayer.__new__``
(:47-66) then instantiates our subclass wherever the model builds
``QwenGatedDeltaNetAttention``. The subclass overrides ONE method,
``_forward_core_fused_norm_packed`` (qwen_gdn_linear_attn.py:1793): for a
pure non-spec decode batch it runs stock ``causal_conv1d_update`` + our
kernel (conv + 1 launch instead of conv + 2 copies + 2 Triton kernels);
every other batch shape (prefill, mixed, spec-verify) goes to the stock
method unchanged — that is routing by batch shape, not a fallback: K-GDN1
only implements the non-spec decode step.

Startup kernel-path assertion (log markers, see the dossier §6 protocol):
  ``suffix qwen-gdn K-GDN1 armed``            register() succeeded
  ``suffix qwen-gdn K-GDN1 ACTIVE``           first layer constructed
  ``suffix qwen-gdn K-GDN1 oracle PASS``      on-GPU identity check at profile run
  ``suffix qwen-gdn K-GDN1 decode-launch``    first real decode launch (graph capture)
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_QWEN_GDN"
LAYER_NAME = "QwenGatedDeltaNetAttention"
ACT_CODES = {"silu": 0, "swish": 0, "sigmoid": 1}
# Numerics gate (oracle): out is bf16 (1 ulp ~ 7.8e-3 relative); the state
# stays fp32. The kernel reads o in fp32 where the stock chain round-trips
# core_attn_out through bf16, so allow ~2 bf16 ulps of output difference.
OUT_ATOL = 3e-2
OUT_RTOL = 2e-2
STATE_RTOL = 1e-4
STATE_ATOL = 1e-5

_state = {"armed": False, "active_layers": 0, "oracle": None, "launch_logged": False}


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
    zf = z.float()
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


def make_inputs(T, H, HV, K, slots, device="cpu", seed=0, idx=None):
    """Random K-GDN1 inputs at a given shape (bf16 activations, fp32 state).
    ``mixed_qkv``/``z`` are row-strided views into one packed qkvz buffer,
    exactly like the in_proj_qkvz split the layer hands the kernel."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    qkv_w = 2 * H * K + HV * K
    qkvz = torch.randn(T, qkv_w + HV * K, generator=gen).to(torch.bfloat16)
    ba = torch.randn(T, 2 * HV, generator=gen).to(torch.bfloat16)
    a_log = (torch.rand(HV, generator=gen) * 2 - 1).float()
    dt_bias = torch.randn(HV, generator=gen).float()
    norm_w = (1 + 0.1 * torch.randn(K, generator=gen)).float()
    state = (0.1 * torch.randn(slots, HV, K, K, generator=gen)).float()
    if idx is None:
        idx = [(t % (slots - 1)) + 1 for t in range(T)]  # distinct, >0
    state_idx = torch.tensor(idx, dtype=torch.int32)
    tens = [qkvz, ba, a_log, dt_bias, norm_w, state, state_idx]
    qkvz, ba, a_log, dt_bias, norm_w, state, state_idx = [t.to(device) for t in tens]
    mixed_qkv = qkvz[:, :qkv_w]
    z = qkvz[:, qkv_w:].view(T, HV, K)
    return dict(mixed_qkv=mixed_qkv, z=z, ba=ba, a_log=a_log, dt_bias=dt_bias,
                norm_w=norm_w, state=state, state_idx=state_idx)


# ---------------------------------------------------------------------------
# on-GPU oracle (runs once per process at the profile run, before capture)
# ---------------------------------------------------------------------------
def run_gpu_oracle(native, H, HV, K, act, norm_eps, row_stride=None):
    """JIT/load the kernel at the serving shape and check it against the
    torch reference on the GPU, plus a CUDA-graph capture/replay round.
    Raises RuntimeError on any mismatch (the gate is fail-closed)."""
    import torch

    dev = torch.device("cuda", torch.cuda.current_device())
    worst = {"out": 0.0, "state": 0.0}
    cases = [(1, [1]), (3, [2, 0, 5]), (16, None)]
    for T, idx in cases:
        inp = make_inputs(T, H, HV, K, slots=24, device=dev, seed=T, idx=idx)
        if row_stride is not None and row_stride != inp["mixed_qkv"].stride(0):
            raise RuntimeError(
                f"K-GDN1 oracle: layer row stride {row_stride} != oracle "
                f"{inp['mixed_qkv'].stride(0)} (JIT specialization would differ)")
        ref_state = inp["state"].clone()
        ref = gdn_decode_fused_torch(
            inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], ref_state, inp["state_idx"].cpu(), H, K ** -0.5,
            norm_eps, act, round_o_bf16=True)
        out = torch.empty(T, HV, K, dtype=torch.bfloat16, device=dev)
        native.gdn_decode_fused_cuda(
            inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], out, H, K ** -0.5,
            norm_eps, act, torch.cuda.current_stream(dev).cuda_stream)
        torch.cuda.synchronize(dev)
        _check(out.float(), ref, inp["state"], ref_state, worst, f"T={T}")

    # CUDA-graph capture/replay: the kernel must be capturable (no sync, no
    # alloc on the launch path) and replay must match an eager launch.
    T = 4
    inp = make_inputs(T, H, HV, K, slots=24, device=dev, seed=99)
    s_eager = inp["state"].clone()
    out_eager = torch.empty(T, HV, K, dtype=torch.bfloat16, device=dev)
    out_graph = torch.empty_like(out_eager)
    args = [inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"]]
    native.gdn_decode_fused_cuda(*args, s_eager, inp["state_idx"], out_eager, H,
                                 K ** -0.5, norm_eps, act,
                                 torch.cuda.current_stream(dev).cuda_stream)
    s_graph = inp["state"].clone()
    s_graph0 = s_graph.clone()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph, stream=side):
            native.gdn_decode_fused_cuda(*args, s_graph, inp["state_idx"], out_graph,
                                         H, K ** -0.5, norm_eps, act,
                                         side.cuda_stream)
    torch.cuda.current_stream(dev).wait_stream(side)
    s_graph.copy_(s_graph0)  # capture may or may not have executed the kernel
    graph.replay()
    torch.cuda.synchronize(dev)
    if not torch.equal(out_graph, out_eager) or not torch.equal(s_graph, s_eager):
        raise RuntimeError("K-GDN1 oracle FAIL: CUDA-graph replay != eager launch")
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
# vLLM wiring
# ---------------------------------------------------------------------------
def _native():
    from suffix_hybrid import _native as native
    if not getattr(native, "HAS_QWEN_GDN_CUDA", False):
        raise RuntimeError(
            f"{GATE}=1 but suffix_hybrid._native was built WITHOUT cargo feature "
            "`qwen-gdn-kernels` (no K-GDN1 GPU op) — refusing to serve the stock "
            "GDN chain while the K-GDN1 gate is armed")
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


def layer_contract_violations(layer) -> list:
    """Every reason this GDN layer cannot run K-GDN1 (empty = OK)."""
    import torch
    why = []
    if layer.tp_size != 1:
        why.append(f"tp_size={layer.tp_size} (K-GDN1 is TP=1)")
    if layer.gqa_interleaved_layout:
        why.append("interleaved (Qwen3-Next) qkvz layout")
    if layer.head_k_dim != layer.head_v_dim:
        why.append(f"K={layer.head_k_dim} != V={layer.head_v_dim}")
    k = layer.head_k_dim
    if k & (k - 1) or k > 256:
        why.append(f"K={k} not a power of two <= 256")
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
    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    # Source-anchor pin (vLLM 0.30.0, qwen_gdn_linear_attn.py:1793): the one
    # method we override must keep its contract, or we refuse to arm.
    import inspect
    params = list(inspect.signature(
        QwenGatedDeltaNetAttention._forward_core_fused_norm_packed).parameters)
    if params != ["self", "mixed_qkvz", "ba", "core_attn_out"]:
        raise RuntimeError(f"K-GDN1 anchor drift: _forward_core_fused_norm_packed{params}")

    class SuffixQwenGDN(QwenGatedDeltaNetAttention):
        """QwenGatedDeltaNetAttention + K-GDN1 on the non-spec decode step."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            why = layer_contract_violations(self)
            if why:
                raise ValueError(f"{GATE}=1 but layer {self.prefix} violates the "
                                 f"K-GDN1 contract: {'; '.join(why)}")
            self._kgdn_act = ACT_CODES[self.norm.activation]
            self._kgdn_params = None
            _state["active_layers"] += 1
            if _state["active_layers"] == 1:
                _log(f"K-GDN1 ACTIVE: {LAYER_NAME} -> SuffixQwenGDN (first layer "
                     f"{self.prefix}; H={self.num_k_heads} HV={self.num_v_heads} "
                     f"K={self.head_k_dim} act={self.norm.activation})")

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

        def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
            md = get_forward_context().attn_metadata
            md = md.get(self.prefix) if isinstance(md, dict) else None
            if md is None:
                # profile / dummy run: pre-JIT + on-GPU oracle before capture.
                self._kgdn_f32()
                if _state["oracle"] is None:
                    _state["oracle"] = run_gpu_oracle(
                        native, self.num_k_heads, self.num_v_heads, self.head_k_dim,
                        self._kgdn_act, self.layer_norm_epsilon,
                        row_stride=mixed_qkvz.stride(0))
                    compiles, hits = native.qwen_gdn_jit_stats()
                    if (os.environ.get("SUFFIX_QWEN_GDN_REQUIRE_PREBUILT", "").strip() == "1"
                            and compiles):
                        raise RuntimeError(
                            f"K-GDN1: {compiles} tileiras JIT compile(s) on this pod "
                            "with SUFFIX_QWEN_GDN_REQUIRE_PREBUILT=1 — the bundled "
                            "cubin store did not cover the serving specialization")
                    _log(f"K-GDN1 oracle PASS max|d_out|={_state['oracle']['out']:.3e} "
                         f"max|d_state|={_state['oracle']['state']:.3e} (+graph replay) "
                         f"jit backend_compiles={compiles} disk_hits={hits}")
                return super()._forward_core_fused_norm_packed(mixed_qkvz, ba,
                                                               core_attn_out)
            if not (md.spec_sequence_masks is None and md.num_prefills == 0
                    and md.num_decodes > 0):
                return super()._forward_core_fused_norm_packed(mixed_qkvz, ba,
                                                               core_attn_out)
            n = md.num_actual_tokens
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            idx = md.non_spec_state_indices_tensor[:n]
            conv_state = (self.kv_cache[0] if is_conv_state_dim_first()
                          else self.kv_cache[0].transpose(-1, -2))
            conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                             self.conv1d.weight.size(2))
            mixed_qkv = causal_conv1d_update(
                mixed_qkvz[:n, :qkv_size], conv_state, conv_w, self.conv1d.bias,
                self.activation, conv_state_indices=idx, validate_data=False)
            z = mixed_qkvz[:n, qkv_size:].unflatten(-1, (-1, self.head_v_dim))
            a_log, dt_bias, norm_w = self._kgdn_f32()
            native.gdn_decode_fused_cuda(
                mixed_qkv, z, ba[:n], a_log, dt_bias, norm_w, self.kv_cache[1], idx,
                core_attn_out[:n], self.num_k_heads // self.tp_size,
                self.head_k_dim ** -0.5, self.layer_norm_epsilon, self._kgdn_act,
                torch.cuda.current_stream().cuda_stream)
            if not _state["launch_logged"]:
                _state["launch_logged"] = True
                _log(f"K-GDN1 decode-launch: first non-spec decode batch T={n} "
                     f"({self.prefix}); stock FLA packed-decode + layer_norm_fwd "
                     "replaced")

    return SuffixQwenGDN


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_QWEN_GDN=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    native = _native()
    cap = check_sm120()
    store = os.environ.get("SUFFIX_QWEN_GDN_JIT_STORE", "").strip()
    if store:
        _log(f"K-GDN1 JIT store: {native.qwen_gdn_enable_jit_store(store)}")
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    if LAYER_NAME not in op_registry_oot:
        PluggableLayer.register_oot(_make_layer_cls(native), name=LAYER_NAME)
    _state["armed"] = True
    _log(f"K-GDN1 armed: {LAYER_NAME} OOT-registered (cc {cap[0]}.{cap[1]})")
    return _state
