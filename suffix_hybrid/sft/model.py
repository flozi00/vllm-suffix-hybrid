# SPDX-License-Identifier: Apache-2.0
"""LoRA on the SERVED language model + an autograd forward over the same
live modules (no second model, no weight copy).

Serving seam (``install_adapters``, from the Worker.load_model hook, i.e.
BEFORE torch.compile tracing and CUDA-graph capture): every vLLM linear of
the language model gets its ``quant_method`` instance re-classed to a
per-class subclass (``_sft_class``) whose ``apply`` adds
``(x @ sft_A) @ sft_B * sft_scale`` when the layer carries adapter buffers.
A subclass (not a proxy object) keeps isinstance checks and Dynamo tracing
plain; the adapters are non-persistent module buffers, so the compiled graph
reads them as inputs and the captured CUDA graphs see in-place ``copy_``
updates. B starts at zero: an untrained adapter adds an exact 0.0. TP layout
is mtp_tune.lora's (column: B sharded, row: A sharded, delta added before the
layer's own all-reduce -> no extra collective at serve time).

Training seam (``TrainMode``, eager, inside one collective_rpc between engine
steps): the language model's own forward (vLLM module code: residuals,
gating, KV sharing, per-layer inputs, MRoPE) runs with these swaps, all
restored on exit:
  linears      -> FrozenLinear (the serving quant kernel forward; dx from
                  the bf16 weight or the identity probe) + trainable delta;
                  copy_to_tp at column-parallel inputs
  TP all-reduce-> autograd-aware (fwd all-reduce, bwd identity)
  Attention    -> SDPA per packed segment (causal, GQA, sliding window,
                  KV-shared layers read the target layer's k/v)
  GDN layers   -> conv1d + gated delta rule with autograd per segment
                  (flash-linear-attention when importable, else the chunked
                  torch form below)
  CustomOps    -> forward_native; vLLM IR ops -> their native impl; known
                  fused fast paths off; compiled modules -> eager
  decoder layers -> activation checkpointing; layer inputs offloaded to
                  pinned host memory (non_blocking) between forward and
                  backward
Any op this list misses (a fused kernel writing into a preallocated buffer)
silently cuts the gradient; ``grad_reach`` detects it on the first micro-step
of every run and the run is refused.

  short conv  -> LFM2 ShortConv: B*x, depthwise causal conv per segment, C*,
                 out_proj (zero conv state at every segment)
  MoE block   -> router twin (softmax or sigmoid, selection-only bias,
                 renormalize, routed scaling) + frozen routed experts
Unsupported (refused at install, serving untouched): pipeline parallel,
MLA, multi-group routing, attention logit soft-capping, non-GDN/short-conv
mamba layers, encoder-only attention.
"""
from __future__ import annotations

import contextlib
import os
import fnmatch
import inspect
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from suffix_hybrid.mtp_tune.lora import (
    LoRA, SingleGroup, _ReduceFromTP, copy_to_tp, in_out, reduce_from_tp, tp_kind)
from suffix_hybrid.mtp_tune.train import (
    FrozenExperts, FrozenLinear, _vocab_range, experts_handle, vocab_parallel_ce)

DEFAULT_TARGETS = (
    "*.qkv_proj", "*.q_proj", "*.o_proj",          # attention (incl. KV-shared)
    "*.in_proj_qkvz", "*.out_proj",                # GDN (out_proj: also attention / short conv)
    "*.in_proj",                                   # LFM2 short conv (B, C, x gates)
    "*.gate_up_proj", "*.down_proj",               # dense MLP
    "*.w13", "*.w2",                               # LFM2 dense MLP (routed experts have no linear)
)
FUSED_FLAGS = ("use_fused_qk_norm_rope_gate",)    # Qwen3NextAttention Triton fusion
QSA_DENSE_MAX = 2048        # QSA selects every key up to its 2048-block budget: dense = exact
CE_BYTES = int(os.environ.get("SUFFIX_SFT_CE_MB", "256")) << 20   # fp32 logits per CE chunk
GDN_HEAD_GROUP = None       # heads per delta-rule backward group; None = from GDN_GROUP_BYTES
GDN_GROUP_BYTES = 3 << 29   # ~1.5 GiB of fp32 intermediates (~8 KiB per head-token)


def _heads_per_group(t, h):
    return GDN_HEAD_GROUP or max(1, min(h, GDN_GROUP_BYTES // max(1, t * 8192)))

_TRAIN = None   # the active TrainMode (only ever set inside an RPC)
DIAG = {}       # dev-only forward toggles (SUFFIX_SFT_DIAG=1 /v1/sft/diag): moe_qdq, moe_ue8m0, conv32, attn32


# ---------------------------------------------------------------------------
# model surface
# ---------------------------------------------------------------------------
def language_model(model):
    """(lm, inner, lm_head, logits_processor) of a vLLM CausalLM or VLM."""
    lm = model.get_language_model() if hasattr(model, "get_language_model") else model
    inner = lm.model
    head = getattr(lm, "lm_head", None) or model.lm_head
    lp = getattr(lm, "logits_processor", None) or model.logits_processor
    return lm, inner, head, lp


def is_linear(m) -> bool:
    return (getattr(m, "quant_method", None) is not None
            and hasattr(m, "input_size_per_partition") and hasattr(m, "output_size_per_partition"))


def is_attention(m) -> bool:
    return hasattr(m, "impl") and hasattr(m, "kv_sharing_target_layer_name")


def is_gdn(m) -> bool:
    return hasattr(m, "in_proj_qkvz") and hasattr(m, "A_log") and hasattr(m, "conv1d")


def is_qsa(m) -> bool:      # Qwen4Exp query-sparse attention owner (qkv + indexer + o)
    return hasattr(m, "_run_qsa") and hasattr(m, "qkv_proj") and hasattr(m, "o_proj")


def is_shortconv(m) -> bool:  # LFM2 gated short convolution (vLLM ShortConv)
    return (hasattr(m, "L_cache") and hasattr(m, "in_proj") and hasattr(m, "out_proj")
            and hasattr(m, "conv"))


def is_moe_block(m) -> bool:  # Qwen3Next / Qwen4Exp / LFM2 sparse MoE block (router + runner)
    return hasattr(getattr(m, "experts", None), "routed_experts") and hasattr(m, "gate")


def is_hc(m) -> bool:       # Qwen4Exp GatedResidual hyper-connection
    return hasattr(m, "hc_norm") and hasattr(m, "input_mix_weight_up")


def is_ple(m) -> bool:      # Qwen4Exp per-layer n-gram embedding + short conv
    return hasattr(m, "ple_embedding") and hasattr(m, "kv_proj") and hasattr(m, "norm_conv")


def is_mlp(m) -> bool:      # dense MLP block: token-wise, so it can be tiled
    return ((hasattr(m, "w13") and hasattr(m, "w2"))
            or (hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")))


def check_supported(lm, vllm_config=None):
    """Raise NotImplementedError naming the first module SFT cannot train."""
    if vllm_config is not None and vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("pipeline parallel")
    if vllm_config is not None and getattr(vllm_config.cache_config, "kv_sharing_fast_prefill", False):
        raise NotImplementedError("kv_sharing_fast_prefill (in-place buffer copies)")
    moe = [name for name, m in lm.named_modules() if is_moe_block(m)]
    for name, m in lm.named_modules():
        if not is_moe_block(m):
            continue
        re_ = m.experts.routed_experts
        if (getattr(re_, "enable_eplb", False) or getattr(m, "is_sequence_parallel", False)
                or getattr(m, "is_fused_shared_expert_enabled", False)):
            raise NotImplementedError(f"MoE with EPLB / sequence parallel ({name})")
        if (getattr(re_, "use_grouped_topk", False) and int(getattr(re_, "num_expert_group", 1) or 1) > 1
                or getattr(re_, "scoring_func", "softmax") not in ("softmax", "sigmoid")
                or getattr(re_, "custom_routing_function", None) is not None
                or getattr(re_, "hash_indices_table", None) is not None):
            raise NotImplementedError(f"MoE routing other than (grouped x1) softmax / sigmoid top-k ({name})")
        experts_handle(m.experts)   # raises for unsupported expert dtypes
    for name, m in lm.named_modules():
        cls = type(m).__name__
        inside = any(name == b or name.startswith(b + ".") for b in moe)
        if not inside and (hasattr(m, "w13_weight") or cls.endswith(("FusedMoE", "RoutedExperts"))):
            raise NotImplementedError(f"MoE experts outside a supported MoE block ({name}: {cls})")
        if "MLA" in cls:
            raise NotImplementedError(f"MLA attention ({name}: {cls})")
        if is_attention(m) and not is_qsa(m):
            if getattr(m.impl, "logits_soft_cap", None):
                raise NotImplementedError(f"attention soft-capping ({name})")
            if str(getattr(m, "attn_type", "decoder")).lower().split(".")[-1] != "decoder":
                raise NotImplementedError(f"non-causal attention ({name})")
        if hasattr(m, "get_state_shape") and not (is_gdn(m) or is_ple(m) or is_qsa(m) or is_shortconv(m)):
            raise NotImplementedError(f"recurrent layer {name}: {cls}")


# ---------------------------------------------------------------------------
# serving seam
# ---------------------------------------------------------------------------
_CLASSES: dict = {}


def _sft_class(cls):
    if getattr(cls, "_sft_base", None) is not None:
        return cls
    if cls not in _CLASSES:
        def apply(self, layer, x, bias=None, *args, **kwargs):
            if _TRAIN is not None:
                return _TRAIN.linear(layer, x, bias)
            out = cls.apply(self, layer, x, bias, *args, **kwargs)
            a = getattr(layer, "sft_A", None)
            if a is None:
                return out
            # 2 kernels per adapter: shrink GEMM + expand GEMM whose epilogue
            # does the scale and the add (decode runs ~250 of these per step)
            x2 = x.reshape(-1, x.shape[-1]).to(a.dtype)
            return torch.addmm(out.reshape(-1, out.shape[-1]), x2 @ a, layer.sft_B,
                               alpha=layer.sft_scale).view(out.shape)

        _CLASSES[cls] = type(f"Sft{cls.__name__}", (cls,), {"apply": apply, "_sft_base": cls})
    return _CLASSES[cls]


def install_adapters(lm, targets=DEFAULT_TARGETS, rank=16, alpha=32.0, tp_rank=0, dtype=None,
                     rslora=False):
    """Re-class every linear's quant method; zero-init adapters (in the
    model's activation `dtype`) on the linears matching `targets`. Returns
    {name: (layer, LoRA)} (LM-relative names, sorted). Every LM parameter is
    frozen (vLLM never trains; stray .grad on served norms would accumulate)."""
    for p in lm.parameters():
        p.requires_grad_(False)
    out = {}
    for name, m in lm.named_modules():
        if not is_linear(m):
            continue
        m.quant_method.__class__ = _sft_class(type(m.quant_method))
        if not any(fnmatch.fnmatch(name, p) for p in targets):
            continue
        k, n = in_out(m)
        w = getattr(m, "weight", None)   # NVFP4 layers: uint8 weight, fp32 scales
        dt = dtype or (w.dtype if w is not None and w.dtype in (torch.bfloat16, torch.float16,
                                                                 torch.float32) else torch.bfloat16)
        lora = LoRA(name, k, n, rank, alpha, tp_kind(m), tp_rank,
                    int(getattr(m, "tp_size", 1) or 1), dt, next(m.parameters()).device,
                    rslora=rslora)
        m.requires_unquantized_input = True    # the delta reads the bf16 input (as vLLM LoRA)
        m.register_buffer("sft_A", lora.A, persistent=False)
        m.register_buffer("sft_B", lora.B, persistent=False)
        m.sft_scale = lora.scale
        out[name] = (m, lora)
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# GDN core with autograd
# ---------------------------------------------------------------------------
def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk=64):
    """Gated delta rule, q/k/v [B, T, H, D] (q/k already at v's head count),
    g/beta [B, T, H]; q/k L2-normalized inside, q scaled by D^-1/2. Chunked
    UT-transform form of HF transformers' torch_chunk_gated_delta_rule
    (Apache-2.0); fp32, differentiable."""
    dt = v.dtype
    q, k, v, beta, g = (x.transpose(1, 2).float() for x in (q, k, v, beta, g))
    q, k = _l2norm(q) * q.shape[-1] ** -0.5, _l2norm(k)
    t = q.shape[2]
    pad = -t % chunk
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    b, h, tp, d = k.shape
    q, k, kb, vb = (x.reshape(b, h, tp // chunk, chunk, x.shape[-1])
                    for x in (q, k, k * beta[..., None], v * beta[..., None]))
    g = g.reshape(b, h, tp // chunk, chunk).cumsum(-1)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(upper, float("-inf")).exp()
    ut = (kb @ k.transpose(-1, -2)) * decay
    attn = (q @ k.transpose(-1, -2)) * decay
    u = torch.linalg.solve_triangular(ut, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(ut, kb * g.exp()[..., None], upper=False,
                                      unitriangular=True)
    q = q * g.exp()[..., None]
    k = k * (g[..., -1:] - g).exp()[..., None]
    last = g[..., -1].exp()[..., None, None]
    s = q.new_zeros(b, h, d, v.shape[-1])
    out = []
    for i in range(tp // chunk):
        vn = u[:, :, i] - w[:, :, i] @ s
        out.append(q[:, :, i] @ s + attn[:, :, i] @ vn)
        s = s * last[:, :, i] + k[:, :, i].transpose(-1, -2) @ vn
    o = torch.stack(out, 2).reshape(b, h, tp, -1)[:, :, :t]
    return o.transpose(1, 2).to(dt)


def _fla():
    """vLLM's bundled FLA chunk kernel (forward only: serving's prefill
    kernel), None where it is not importable."""
    try:
        from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as f
    except Exception:  # noqa: BLE001
        try:
            from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as f
        except Exception:  # noqa: BLE001
            return None
    return lambda q, k, v, g, beta: f(q, k, v, g=g, beta=beta,
                                      use_qk_l2norm_in_kernel=True)[0]


class _FlaRule(torch.autograd.Function):
    """[T, H, D] gated delta rule: forward = the serving kernel (exact serving
    numerics, Triton speed); backward = the chunked torch form recomputed
    once, GDN_HEAD_GROUP heads at a time (bounded fp32 intermediates)."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, fwd):
        ctx.save_for_backward(q, k, v, g, beta)
        with torch.no_grad():
            return fwd(q[None], k[None], v[None], g[None], beta[None])[0].to(v.dtype)

    @staticmethod
    def backward(ctx, do):
        ins = ctx.saved_tensors
        grads = [torch.zeros_like(t) for t in ins]
        n = _heads_per_group(*ins[0].shape[:2])
        for h0 in range(0, ins[0].shape[1], n):
            sl = slice(h0, h0 + n)
            with torch.enable_grad():
                part = [t[:, sl].detach().requires_grad_(True) for t in ins]
                o = chunk_gated_delta_rule(*(x[None] for x in part))[0]
                for dst, gr in zip(grads, torch.autograd.grad(o, part, do[:, sl])):
                    dst[:, sl] = gr
        return (*grads, None)


# ---------------------------------------------------------------------------
# Qwen4Exp custom ops: torch twins (vllm/models/qwen4_exp/nvidia/ops/hc.py,
# ops/ple.py @ v0.30.0, same rounding boundaries)
# ---------------------------------------------------------------------------
def _gemma_rms_groups(x, w, eps, groups):
    n = x.shape[0]
    xf = x.float().view(n, groups, -1)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y = y + y * w.float().view(-1 if w.numel() == y.shape[-1] else groups, y.shape[-1])
    return y.reshape(n, -1)


def hc_grouped_gemma_rmsnorm(x, weight, eps, num_groups):
    return _gemma_rms_groups(x, weight, eps, num_groups).to(x.dtype)


def hc_silu(x, hc_count):
    xf = x.float() / hc_count
    return (xf * torch.sigmoid(xf)).to(x.dtype)


def hc_gate_mix(x, gate, hc_count):
    n = x.shape[0]
    mixed = torch.sigmoid(gate.float()).view(n, hc_count, -1) * x.float().view(n, hc_count, -1)
    return (mixed.sum(1) / hc_count).to(x.dtype)


def hc_combine(residual, block_output, injection_logits, hc_count):
    n = residual.shape[0]
    blk = block_output.float()[:, None, :]
    if injection_logits is not None:
        blk = blk * (2.0 * torch.sigmoid(injection_logits.float() / hc_count))[:, :, None]
    return (residual.float().view(n, hc_count, -1) + blk).reshape(n, -1).to(residual.dtype)


def hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count):
    out = hc_combine(residual, block_output, injection_logits, hc_count)
    return out, _gemma_rms_groups(out, norm_weight, eps, hc_count).to(out.dtype)


HC_TWINS = dict(grouped_gemma_rmsnorm=hc_grouped_gemma_rmsnorm, hc_silu=hc_silu,
                hc_gate_mix=hc_gate_mix, hc_combine=hc_combine, hc_combine_norm=hc_combine_norm)


def ple_gate(key, value, hidden, nk, nq, ncw, eps):
    """(gated, normed) [T, HC*H]: per stream g = sigmoid(sign(d) sqrt|d|),
    d = <rms(key)(1+nk), rms(hidden)(1+nq)> / sqrt(H); gated = g * value."""
    t, h = value.shape[0], value.shape[-1]
    hc, dt = hidden.shape[-1] // h, hidden.dtype

    def rms(x, w):
        xf = x.float().view(t, hc, h)
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
                * (1.0 + w.float().view(hc, h))).to(dt)
    kn, qn = rms(key, nk), rms(hidden, nq)
    d = (kn.float() * qn.float()).to(dt).float().sum(-1).to(dt).float() / h ** 0.5
    d = d.to(dt).float()
    mag = torch.sqrt(torch.clamp(d.abs(), min=1e-6)).to(dt).float()
    g = torch.sigmoid(torch.sign(d) * mag).to(dt).float()
    gated = (g[..., None] * value.float()[:, None, :]).to(dt)
    normed = rms(gated, ncw)
    return gated.reshape(t, -1), normed.reshape(t, -1)


# ---------------------------------------------------------------------------
# training mode
# ---------------------------------------------------------------------------
class _Raw:
    """The TP group's original collectives (what the autograd Functions call)."""

    def __init__(self, g, all_reduce):
        self.all_reduce, self.world_size, self.rank_in_group = all_reduce, g.world_size, g.rank_in_group


class _Park(torch.autograd.Function):
    """x -> (1-element handle broadcast to x's shape); x's data parked on the
    host. The handle is what activation checkpointing keeps; its gradient
    (full-size, on the device) is x's gradient: no gradient crosses PCIe."""

    @staticmethod
    def forward(ctx, x, slot):
        try:
            h = torch.empty(x.shape, dtype=x.dtype, pin_memory=x.is_cuda)
        except RuntimeError:   # pinned pool exhausted: pageable works, just slower
            h = torch.empty(x.shape, dtype=x.dtype)
        slot.append(h.copy_(x.detach(), non_blocking=False))
        return x.new_zeros(()).expand(x.shape)

    @staticmethod
    def backward(ctx, g):
        return g, None


class _Unpark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, handle, slot):
        return slot[0].to(handle.device, non_blocking=True, copy=True)

    @staticmethod
    def backward(ctx, g):
        return g, None


def _to_host(t):
    """Pinned host copy of a device tensor (pageable if the pinned pool is full)."""
    try:
        h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    except RuntimeError:
        h = torch.empty(t.shape, dtype=t.dtype)
    h.copy_(t.detach(), non_blocking=False)
    return h


def _swap_out(mod):
    """Move a frozen layer's weights to host (block swap, Unsloth-style)."""
    for p in mod.parameters():
        if p.is_cuda:
            p.data = p.data.to("cpu", non_blocking=False)


def _swap_in(mod, dev):
    for p in mod.parameters():
        if not p.is_cuda:
            p.data = p.data.to(dev, non_blocking=False)


class _HostCkpt(torch.autograd.Function):
    """Checkpoint whose saved inputs live in pinned host memory.

    ``torch.utils.checkpoint`` keeps one activation per layer on the device --
    layers x seq x hidden x bytes, which is 7.6 GiB at 32k for this model and
    more than the card has. This saves the same tensors to host and recomputes
    the layer in backward, which is what HF's
    ``gradient_checkpointing(offload=True)`` does. It also cannot be expressed
    with ``saved_tensors_hooks``: checkpoint installs its own hooks inside the
    region and shadows any outer ones.
    """

    @staticmethod
    def forward(ctx, fn, device, dummy, mod, *args):
        ctx.fn, ctx.device, ctx.mod = fn, device, mod
        ctx.host = []
        for a in args:
            if torch.is_tensor(a) and a.is_floating_point():
                h = a if not a.is_cuda else _to_host(a)
                ctx.host.append(h)
            else:
                ctx.host.append(a)
        dev = [h.to(device, non_blocking=False) if torch.is_tensor(h) else h for h in ctx.host]
        with torch.no_grad():
            out = fn(*dev)
        if not isinstance(out, tuple):
            out = (out,)
        # Return host tensors: the graph then holds host memory between layers,
        # so no device activation survives the forward.
        return tuple(_to_host(o) if torch.is_tensor(o) else o for o in out)

    @staticmethod
    def backward(ctx, *grads):
        args = [h.to(ctx.device, non_blocking=False) if torch.is_tensor(h) else h
                for h in ctx.host]
        for a in args:
            if torch.is_tensor(a) and a.is_floating_point():
                a.requires_grad_(True)
        with torch.enable_grad():
            out = ctx.fn(*args)
        if not isinstance(out, tuple):
            out = (out,)
        need = [a for a in args if torch.is_tensor(a) and a.is_floating_point()]
        g = [x.to(ctx.device, non_blocking=False) if torch.is_tensor(x) else x for x in grads]
        # autograd.grad only fills the *input* grads; the LoRA leaves need
        # autograd.backward to accumulate into .grad, so do both.
        torch.autograd.backward(out, g, retain_graph=True)
        got = torch.autograd.grad(out, need, g, allow_unused=True)
        if ctx.mod is not None:
            _swap_out(ctx.mod)   # only after the recompute's gradient is taken
        # inputs were host tensors, so their gradients must be host tensors too
        got = [x.to("cpu", non_blocking=False) if torch.is_tensor(x) else x for x in got]
        it = iter(got)
        return (None, None, None, None) + tuple(
            next(it) if (torch.is_tensor(a) and a.is_floating_point()) else None for a in args)


class TrainMode(contextlib.ContextDecorator):
    """Context in which ``lm``'s live modules compute with autograd.
    ``params``: {id(layer): (A, B, scale)} trainable views; ``segments``:
    [(start, end)] of the packed sequences (attention / GDN never cross them).
    """

    def __init__(self, sft, params, segments, grad=True, offload_bytes=0, check=False):
        self.s, self.params, self.segments = sft, params, segments
        self.grad, self.check = grad, check
        self.offload_left = offload_bytes   # host budget for parked layer inputs
        self.mlp_tile = int(getattr(sft.cfg, "mlp_tile", 0) or 0)
        self.offload_saved = bool(getattr(sft.cfg, "offload_saved", False))
        self.mem_trace = bool(getattr(sft.cfg, "mem_trace", False))
        self.host_ckpt = bool(getattr(sft.cfg, "host_ckpt", False))
        # LoRA freezes the base weights, so hidden states never require grad and
        # autograd.Function.apply would attach no node. This forces one.
        self._dummy = torch.zeros(1, device=self.s.device, requires_grad=True)
        self._trace_n = 0
        self.saved_min = int(getattr(sft.cfg, "saved_min_mb", 1) or 1) * 2**20
        self.cut = []      # modules whose output lost the input's grad (check=True)
        self.g = sft.g
        self.kv = {}
        self.fla = _fla()
        self._stack = None

    # -- swaps ---------------------------------------------------------------
    def _set(self, obj, attr, value):
        had = attr in obj.__dict__
        old = obj.__dict__.get(attr)
        setattr(obj, attr, value)
        self._stack.callback(lambda: setattr(obj, attr, old) if had else delattr(obj, attr))

    def __enter__(self):
        global _TRAIN
        if _TRAIN is not None:
            raise RuntimeError("TrainMode is not reentrant")
        st = self._stack = contextlib.ExitStack()
        try:
            st.enter_context(torch.inference_mode(False))
            st.enter_context(torch.enable_grad() if self.grad else torch.no_grad())
            if self.offload_saved:
                # Unsloth's offload_activations, generalised: every tensor the
                # backward will need (activations *and* the frozen base weights
                # the linear backward reads) is parked in pinned host memory and
                # pulled back on unpack. This is what makes a 64k micro-batch
                # fit on a 12 GB card that is also serving 128k.
                st.enter_context(torch.autograd.graph.saved_tensors_hooks(
                    self._pack_saved, self._unpack_saved))
            mods = self.s.__dict__.get("_sft_mods")
            if mods is None:   # classify once per model (thousands of modules)
                mods = self.s.__dict__["_sft_mods"] = [
                    (m, "qsa" if is_qsa(m) else "moe" if is_moe_block(m) else
                     "sconv" if is_shortconv(m) else
                     "hc" if is_hc(m) else "ple" if is_ple(m) else
                     "attn" if is_attention(m) else "gdn" if is_gdn(m) else
                     "mlp" if is_mlp(m) else
                     "op" if hasattr(m, "_forward_method") and hasattr(m, "forward_native") else None,
                     hasattr(m, "do_not_compile"), [f for f in FUSED_FLAGS if hasattr(m, f)])
                    for m in self.s.lm.modules()]
                self.s.__dict__["_sft_kv_targets"] = {
                    m.kv_sharing_target_layer_name for m, kind, _, _ in mods
                    if kind == "attn" and m.kv_sharing_target_layer_name}
            self.kv_targets = self.s.__dict__["_sft_kv_targets"]
            if self.check:
                from torch.nn.modules.module import register_module_forward_hook
                st.callback(register_module_forward_hook(self._cut_hook).remove)
            hc_mods = {sys.modules[type(m).__module__] for m, kind, _, _ in mods if kind == "hc"}
            for mod in hc_mods:      # the HC Triton ops have no autograd: torch twins
                for name, fn in HC_TWINS.items():
                    if hasattr(mod, name):
                        self._set(mod, name, fn)
            if getattr(self.s.inner, "_mtp_hidden_buffer", None) is not None:
                self._set(self.s.inner, "_mtp_hidden_buffer", None)   # serving's drafter state
            for m, kind, compiled, flags in mods:
                if kind == "qsa":
                    self._set(m, "forward", lambda positions, hidden_states, _m=m:
                              self.qsa(_m, positions, hidden_states))
                elif kind == "moe":
                    self._set(m, "forward", lambda hidden_states, *a, _m=m, **kw:
                              self.moe(_m, hidden_states))
                elif kind == "ple":
                    self._set(m, "forward", lambda *a, _m=m, **kw: self.ple(_m, *a, **kw))
                elif kind == "attn":
                    self._set(m, "forward", lambda q, k, v, *a, _m=m, **kw: self.attention(_m, q, k, v))
                elif kind == "gdn":
                    self._set(m, "forward", lambda hidden_states, _m=m: self.gdn(_m, hidden_states))
                elif kind == "sconv":
                    self._set(m, "forward", lambda hidden_states, output, _m=m:
                              self.shortconv(_m, hidden_states, output))
                elif kind == "mlp":
                    orig = m.forward
                    self._set(m, "forward", lambda x, _o=orig: self._tiled_mlp(_o, x))
                elif kind == "op":
                    self._set(m, "_forward_method", m.forward_native)
                if compiled:
                    self._set(m, "do_not_compile", True)
                for flag in flags:
                    if getattr(m, flag):
                        self._set(m, flag, False)
            _layers = list(self.s.inner.layers)
            _keep = int(getattr(self.s.cfg, "swap_keep", 0) or 0)
            for _i, layer in enumerate(_layers):
                _swap = bool(_keep) and _i >= _keep
                if _swap:
                    _swap_out(layer)   # start every step with the tail on host
                self._set(layer, "forward",
                          self._checkpointed(layer.forward, last=(_i == len(_layers) - 1),
                                             mod=layer, swap=_swap))
            if _keep:
                # serving shares these modules: put every layer back before the
                # context exits, or the parity check runs against CPU weights
                self._stack.callback(
                    lambda: [_swap_in(l, self.s.device) for l in _layers])
            if self.g.world_size > 1:
                raw = _Raw(self.g, self.g.all_reduce)
                self._set(self.g, "all_reduce", lambda x: _ReduceFromTP.apply(x, raw))
            self._native_ir(st)
            self._workspace()
            if self.s.vllm_config is not None:
                from vllm.forward_context import set_forward_context
                st.enter_context(set_forward_context(None, self.s.vllm_config))
        except BaseException:
            st.close()
            raise
        _TRAIN = self
        return self

    def __exit__(self, *exc):
        global _TRAIN
        _TRAIN = None
        self.kv.clear()
        self._stack.close()
        return False

    def _cut_hook(self, m, args, out):
        """A module that received a grad-requiring tensor but returns float
        tensors without grad ran an op with no autograd (a fused kernel)."""
        flat = lambda x: [t for t in (x if isinstance(x, (tuple, list)) else [x]) if torch.is_tensor(t)]
        if (torch.is_grad_enabled() and any(t.requires_grad for t in flat(args))
                and any(t.is_floating_point() and not t.requires_grad for t in flat(out))):
            self.cut.append(type(m).__name__)

    def _workspace(self):
        """Serving kernels (Marlin FP8: a per-CUDA-stream lock tensor) register
        small PERSISTENT workspace resources lazily; after warmup the manager is
        locked and refuses a key it has not seen on this stream. Training runs
        on another stream: let missing persistent resources be created (they
        never touch captured graphs). Scratch growth stays locked (it would
        reallocate storage the CUDA graphs captured) and fails loudly."""
        try:
            from vllm.v1.worker.workspace import current_workspace_manager
            wm = current_workspace_manager()
        except Exception:  # noqa: BLE001 - vLLM without a workspace manager
            return
        orig = wm.get_persistent_resource

        def lenient(key, factory):
            if getattr(wm, "_locked", False) and key not in wm._persistent_resources[wm._get_workspace_id()]:
                wm._locked = False
                try:
                    return orig(key, factory)
                finally:
                    wm._locked = True
            return orig(key, factory)
        self._set(wm, "get_persistent_resource", lenient)

    @staticmethod
    def _native_ir(st):
        try:
            from vllm import ir
            from vllm.ir.op import IrOp
        except Exception:  # noqa: BLE001 - vLLM without the IR layer
            return
        st.enter_context(ir.op.enable_torch_wrap(False))
        for mod in (getattr(ir.ops, n) for n in dir(ir.ops)):
            for op in (getattr(mod, n, None) for n in dir(mod)):
                if isinstance(op, IrOp) and op.get_priority():
                    st.enter_context(op.set_priority(["native"]))

    def _checkpointed(self, fwd, last=False, mod=None, swap=False):
        def run(*args, **kwargs):
            if not torch.is_grad_enabled():
                if swap:
                    _swap_in(mod, self.s.device)
                out = fwd(*args, **kwargs)
                if swap:
                    _swap_out(mod)
                return out
            # float tensors (hidden states / residual) travel through the
            # checkpoint as host copies; ints (positions) and the rest stay
            slots = [(i, a) for i, a in enumerate(args)] + list(kwargs.items())
            keys = [k for k, v in slots if torch.is_tensor(v) and v.is_floating_point()]
            size = sum(v.numel() * v.element_size() for k, v in slots if k in keys)
            parked = {}
            if not (self.offload_saved or self.host_ckpt) and size <= self.offload_left:   # past the host budget inputs stay on the GPU
                self.offload_left -= size
                parked = {k: [] for k in keys}

            def body(*vals):
                got = {k: (_Unpark.apply(v, parked[k]) if parked else v) for k, v in zip(keys, vals)}
                a = [got.get(i, v) for i, v in enumerate(args)]
                kw = {k: got.get(k, v) for k, v in kwargs.items()}
                if self.mem_trace:
                    self._trace_n += 1
                    free, _ = torch.cuda.mem_get_info()
                    print(f"[sft mem] layer {self._trace_n} enter: free {free / 2**30:.2f} GiB, "
                          f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                          f"parked={bool(parked)} left={self.offload_left / 2**30:.1f} GiB, "
                          f"size={size / 2**20:.0f} MB",
                          file=sys.stderr, flush=True)
                if swap:
                    _swap_in(mod, self.s.device)
                out = fwd(*a, **kw)
                if swap and not torch.is_grad_enabled():
                    # forward pass only: no graph holds the weights, so they can
                    # go back to host. The recompute's swap-out happens in
                    # _HostCkpt.backward, after autograd.grad has read them.
                    _swap_out(mod)
                return out

            # Two parking designs. HF's `offload=True` lets the checkpoint save
            # the real activation and moves *that* to host via saved_tensors_hooks
            # -- the graph then holds a host tensor, not a device one. Our _Park
            # handle is smaller still, but the autograd edge from _Park keeps the
            # producer's device tensor alive, so it does not actually free it.
            # When offload_saved is on, use the HF design and skip _Park.
            if self.host_ckpt:
                out = _HostCkpt.apply(body, self.s.device, self._dummy,
                                      mod if swap else None,
                                      *[v for k, v in slots if k in keys])
                if last:   # the model's own norm/CE expect device tensors
                    out = tuple(o.to(self.s.device, non_blocking=False)
                                if torch.is_tensor(o) else o for o in out)
                return out
            if self.offload_saved:
                vals = [v for k, v in slots if k in keys]
            else:
                vals = [(_Park.apply(v, parked[k]) if parked else v) for k, v in slots if k in keys]
            return checkpoint(body, *vals, use_reentrant=False)
        return run

    def _pack_saved(self, t):
        """saved_tensors_hooks pack: park a tensor the backward will need."""
        if (t.is_cuda and t.is_floating_point() and t.numel() * t.element_size() >= self.saved_min):
            try:
                h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            except RuntimeError:   # pinned pool exhausted: pageable still works
                h = torch.empty(t.shape, dtype=t.dtype)
            h.copy_(t, non_blocking=True)
            return ("h", h)
        return ("g", t)

    def _unpack_saved(self, h):
        tag, t = h
        if tag == "g":
            return t
        return t.to(self.s.device, non_blocking=True)

    def _tiled_mlp(self, orig, x):
        """Run a token-wise MLP in sequence tiles.

        The gate/up intermediate is [T, 2*ff] and dominates peak memory at
        long context: 64k tokens x 21504 x 2 B = 2.7 GB for LFM2.5-2.6B, plus
        the activated copy. Tiling bounds the transient to tile x 2*ff and is
        exact -- an MLP has no cross-token mixing, so the tiles are independent.
        """
        tile = self.mlp_tile
        if tile <= 0 or x.shape[0] <= tile:
            return orig(x)
        return torch.cat([orig(x[i:i + tile]) for i in range(0, x.shape[0], tile)], 0)

    # -- swapped ops ---------------------------------------------------------
    def linear(self, layer, x, bias):
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if tp_kind(layer) == "col":
            x = copy_to_tp(x, self.g)
        y = FrozenLinear.apply(x, layer)
        p = self.params.get(id(layer))
        if p is not None:
            a, b, scale = p
            y = y + ((x.to(a.dtype) @ a) @ b * scale).to(y.dtype)
        if bias is not None:
            y = y + bias.detach()
        return y.reshape(*lead, y.shape[-1])

    def attention(self, m, q, k, v):
        t = q.shape[0]
        nh, nkv, hd = m.num_heads, m.num_kv_heads, m.head_size
        hdv = getattr(m, "head_size_v", hd)
        q = q.reshape(t, nh, hd)
        if k is None:
            k, v = self.kv[m.kv_sharing_target_layer_name]
        else:
            k, v = k.reshape(t, nkv, hd), v.reshape(t, nkv, hdv)
            if m.layer_name in self.kv_targets:
                self.kv[m.layer_name] = (k, v)
        if nh != nkv:   # explicit GQA: SDPA's enable_gqa forces the O(n^2) math kernel
            k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
        sw = getattr(m, "sliding_window", None)
        out = []
        for s, e in self.segments:
            qs, ks, vs = (z[s:e].transpose(0, 1)[None] for z in (q, k, v))
            n = e - s
            mask = None
            if sw is not None and sw < n:
                i = torch.arange(n, device=q.device)
                d = i[:, None] - i[None, :]
                mask = (d >= 0) & (d < sw)
            if DIAG.get("attn32"):
                qs, ks, vs = qs.float(), ks.float(), vs.float()
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, is_causal=mask is None,
                                               scale=m.impl.scale)
            out.append(o[0].transpose(0, 1).to(q.dtype))
        return torch.cat(out).reshape(t, nh * hdv)

    def gdn(self, m, hidden_states):
        qkvz, _ = m.in_proj_qkvz(hidden_states)
        ba, _ = m.in_proj_ba(hidden_states)
        if m.gqa_interleaved_layout:     # Qwen3-Next
            q, k, v, z, b, a = m.fix_query_key_value_ordering(qkvz, ba)
            mixed = torch.cat([x.reshape(x.shape[0], -1) for x in (q, k, v)], -1)
        else:                            # Qwen3.5: [q, k, v, z] / [b, a]
            qkv = (m.key_dim * 2 + m.value_dim) // m.tp_size
            mixed, z = qkvz.split([qkv, m.value_dim // m.tp_size], -1)
            z = z.reshape(z.shape[0], -1, m.head_v_dim)
            b, a = m.split_ba(ba)
        w = m.conv1d.weight.detach()     # [C, 1, K]
        kd, vd = m.key_dim // m.tp_size, m.value_dim // m.tp_size
        out = []
        for s, e in self.segments:
            x = F.conv1d(mixed[s:e].t()[None], w, getattr(m.conv1d, "bias", None),
                         padding=w.shape[-1] - 1, groups=w.shape[0])[..., :e - s]
            x = (F.silu(x) if m.activation in ("silu", "swish") else x)[0].t()
            qs, ks, vs = x.split([kd, kd, vd], -1)
            qs = qs.reshape(e - s, -1, m.head_k_dim)
            ks = ks.reshape(e - s, -1, m.head_k_dim)
            vs = vs.reshape(e - s, -1, m.head_v_dim)
            rep = vs.shape[1] // qs.shape[1]
            if rep > 1:
                qs, ks = qs.repeat_interleave(rep, 1), ks.repeat_interleave(rep, 1)
            beta = b[s:e].sigmoid()
            g = -m.A_log.detach().float().exp() * F.softplus(a[s:e].float()
                                                             + m.dt_bias.detach().float())
            out.append(self._rule_grouped(qs, ks, vs, g, beta))
        core = torch.cat(out).to(hidden_states.dtype)
        core = m.norm(core, z)
        o, _ = m.out_proj(core.flatten(-2))
        return o

    def shortconv(self, m, hidden_states, output):
        """LFM2 ShortConv: y = out_proj(C * causal_depthwise_conv(B * x)); the
        conv starts from a zero state at every packed segment. vLLM's layer
        writes into `output` (a fresh tensor of the caller): copy_ keeps autograd."""
        bcx, _ = m.in_proj(hidden_states)
        b, c, x = bcx.chunk(3, -1)
        bx = b * x
        w = m.conv.weight.detach()                       # [C, 1, K]
        bias = getattr(m.conv, "bias", None)
        out = []
        for s, e in self.segments:
            dt = torch.float32 if DIAG.get("conv32") else bx.dtype
            y = F.conv1d(bx[s:e].t()[None].to(dt), w.to(dt), None if bias is None else bias.detach().to(dt),
                         padding=w.shape[-1] - 1, groups=w.shape[0])[..., :e - s]
            out.append(y[0].t().to(bx.dtype))
        o, _ = m.out_proj(c * torch.cat(out))
        output[:o.shape[0]].copy_(o)

    def qsa(self, m, positions, hidden_states):
        """Qwen4Exp QSA as dense causal attention: exact while every segment
        fits the indexer budget (every key selected); sigma(gate) in f32."""
        qkv, _ = m.qkv_proj(hidden_states)
        q, k, v, gate = m._project_qkv_gate(qkv, positions)
        t = hidden_states.shape[0]
        nh, nkv, hd = m.num_heads, m.num_kv_heads, m.head_dim
        q, k, v = q.reshape(t, nh, hd), k.reshape(t, nkv, hd), v.reshape(t, nkv, hd)
        if nh != nkv:
            k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
        out = []
        for s, e in self.segments:
            if e - s > QSA_DENSE_MAX:
                raise ValueError(f"sequence of {e - s} tokens > {QSA_DENSE_MAX}: QSA would be "
                                 "sparse there; use shorter examples")
            qs, ks, vs = (z[s:e].transpose(0, 1)[None] for z in (q, k, v))
            out.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=True,
                                                      scale=m.scaling)[0].transpose(0, 1))
        o = torch.cat(out).reshape(t, nh * hd)
        o = (o.float() * torch.sigmoid(gate.float())).to(hidden_states.dtype)
        return m.o_proj(o)[0]

    def route(self, m, re_, x):
        """(top-k weights [T, k] with autograd, ids). softmax / sigmoid scoring,
        optional selection bias (`e_score_correction_bias`, selection only),
        renormalize, routed scaling factor (router- or runner-side). Ids come
        from serving's own gate, weights from an fp32 re-evaluation so the
        gate input gets its gradient (GateLinear's kernels have none)."""
        k = int(re_.top_k)
        sigmoid = getattr(re_, "scoring_func", "softmax") == "sigmoid"
        bias = getattr(m.gate, "e_score_correction_bias", None)
        if bias is None:
            bias = getattr(re_, "e_score_correction_bias", None)
        scale = float(getattr(getattr(re_, "router", None), "routed_scaling_factor", None)
                      or getattr(re_, "routed_scaling_factor", 1.0) or 1.0)
        scale *= float(getattr(m.experts, "routed_scaling_factor", 1.0) or 1.0)
        renorm = getattr(re_, "renormalize", True)
        logits = m.gate(x)
        logits = (logits[0] if isinstance(logits, tuple) else logits).float()
        if not sigmoid and bias is None:       # plain softmax top-k
            tw, ids = torch.softmax(logits, -1).topk(k, -1)
            if renorm:
                tw = tw / tw.sum(-1, keepdim=True)
            return tw * scale, ids
        with torch.no_grad():
            score = logits.sigmoid() if sigmoid else logits.softmax(-1)
            ids = (score if bias is None else score + bias.float()).topk(k, -1).indices
        gate_w = getattr(m.gate, "weight", None)
        if gate_w is not None and gate_w.dim() == 2 and gate_w.shape[0] == logits.shape[-1]:
            # serving's (bf16-rounded) logit VALUES, an fp32 re-evaluation as the gradient carrier
            carrier = F.linear(x.float(), gate_w.detach().float())
            logits = carrier + (logits - carrier).detach()
        score = logits.sigmoid() if sigmoid else logits.softmax(-1)
        tw = score.gather(-1, ids)
        if renorm:
            tw = tw / tw.sum(-1, keepdim=True)
        return tw * scale, ids

    def moe(self, m, hidden_states):
        """Router softmax top-k -> frozen routed experts (local to this EP
        rank) + gated shared expert, partials summed by one TP all-reduce."""
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        re_ = m.experts.routed_experts
        tw, ids = self.route(m, re_, x)
        emap = getattr(re_, "expert_map", None)
        local = ids if emap is None else emap.to(ids.device)[ids]
        handle = self.s.__dict__.setdefault("_sft_experts", {})
        if id(re_) not in handle:
            handle[id(re_)] = experts_handle(re_)
        # x and the routing weights are replicated but only this rank's
        # experts see them: their grads are partial sums over the TP/EP ranks
        hd = handle[id(re_)]
        if DIAG:
            hd = dict(hd, **{k: DIAG[f"moe_{k}"] for k in ("act_qdq", "ue8m0") if f"moe_{k}" in DIAG})
        out = FrozenExperts.apply(copy_to_tp(x, self.g), copy_to_tp(tw.to(x.dtype), self.g),
                                  local, hd)
        full = None
        se = getattr(m, "shared_expert", None)
        if se is not None:
            gu, _ = se.gate_up_proj(x)
            sd, _ = se.down_proj(se.act_fn(gu))
            gate = torch.sigmoid(m.shared_expert_gate(x)[0])
            if getattr(m, "replicate_shared_expert", False):
                full = gate * sd
            else:   # TP-partial down_proj output (reduce_results=False): partial gate grad
                out = out + copy_to_tp(gate, self.g) * sd
        out = reduce_from_tp(out, self.g)
        return (out if full is None else out + full).reshape(shape)

    def ple(self, m, hidden_states, input_ids, query_start_loc, ngram_context):
        """Qwen4Exp PLE: frozen n-gram lookup (serving's own path, no grad),
        gate + dilated causal short conv per segment with autograd."""
        with torch.no_grad():
            emb = m.ple_embedding(hidden_states, input_ids.reshape(-1), query_start_loc,
                                  ngram_context)
            emb = m._dequantize_embeddings(emb, hidden_states.dtype)
        kv, _ = m.kv_proj(emb)
        key, value = kv.split(m.kv_proj.output_sizes, dim=-1)
        gated, normed = ple_gate(key, value, hidden_states, m.norm_key.weight.detach(),
                                 m.norm_query.weight.detach(), m.norm_conv.weight.detach(),
                                 m.norm_key.eps)
        w = m.conv1d.weight.detach()        # [C, 1, K]
        dil, k = m.short_conv_dilation, w.shape[-1]
        conv = []
        for s, e in self.segments:
            c = F.conv1d(normed[s:e].t()[None].float(), w.float(), padding=(k - 1) * dil,
                         dilation=dil, groups=w.shape[0])[0, :, :e - s].t()
            c = c.to(hidden_states.dtype).float()
            conv.append((c * torch.sigmoid(c)).to(hidden_states.dtype))
        ple_out = (gated.float() + torch.cat(conv).float()).to(hidden_states.dtype)
        return (hidden_states.float() + ple_out.float()).to(hidden_states.dtype)

    def _rule_grouped(self, q, k, v, g, beta):
        """[T, H, D] delta rule. With the serving kernel available: its
        forward + a grouped torch backward (_FlaRule). Without (CPU tests):
        the torch form per checkpointed head group."""
        if self.fla is not None:
            if not torch.is_grad_enabled():
                return self.fla(q[None], k[None], v[None], g[None], beta[None])[0].to(v.dtype)
            return _FlaRule.apply(q, k, v, g, beta, self.fla)
        heads = _heads_per_group(*q.shape[:2])
        if not torch.is_grad_enabled() or q.shape[1] <= heads:
            return chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None])[0]
        outs = []
        for h0 in range(0, q.shape[1], heads):
            sl = slice(h0, h0 + heads)
            outs.append(checkpoint(lambda *t: chunk_gated_delta_rule(*(x[None] for x in t))[0],
                                   q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
                                   use_reentrant=False))
        return torch.cat(outs, 1)

    # -- loss ------------------------------------------------------------------
    def nll(self, ids, rows, tgt, per_token=False):
        """Summed next-token NLL of ids[tgt] predicted from hidden rows (or
        the per-token vector, no grad)."""
        targets = ids[tgt]
        n = ids.shape[0]
        pos = torch.cat([torch.arange(e - s, device=ids.device) for s, e in self.segments])
        if self.s.mrope:
            pos = pos.expand(3, n)
        kw = {}
        params = self.s.__dict__.get("_sft_fwd_params")
        if params is None:
            params = self.s.__dict__["_sft_fwd_params"] = set(
                inspect.signature(type(self.s.inner).forward).parameters)
        if "query_start_loc" in params:     # Qwen4Exp PLE: segment layout + n-gram context
            ng = next((m.ple_embedding for m in self.s.lm.modules() if is_ple(m)), None)
            if ng is not None:
                kw["query_start_loc"] = torch.tensor([0] + [e for _, e in self.segments],
                                                     dtype=torch.int32, device=ids.device)
                kw["ngram_context"] = torch.full((len(self.segments), ng.ngram_size - 1),
                                                 ng.eos_token_id, dtype=torch.int32,
                                                 device=ids.device)
        h = self.s.inner(input_ids=ids, positions=pos, intermediate_tensors=None, **kw)
        if isinstance(h, tuple):
            h = h[0]
        h = h[rows]
        head, lp, g = self.s.head, self.s.lp, self.g
        start, end = _vocab_range(head, g)
        cap, scale = getattr(lp, "soft_cap", None), float(getattr(lp, "scale", 1.0))

        def chunk_ce(hs, tgt):
            y = FrozenLinear.apply(copy_to_tp(hs, g), head).float()
            if cap:
                y = torch.tanh(y / cap) * cap
            if scale != 1.0:
                y = y * scale
            if y.shape[-1] > end - start:   # vocab padding of the shard
                y = torch.cat([y[:, :end - start], torch.full_like(y[:, end - start:], float("-inf"))],
                              -1)
            ce = vocab_parallel_ce(y, tgt, start, end, g)
            return ce if per_token else ce.sum()

        step = max(16, CE_BYTES // (4 * max(1, end - start)))
        if per_token:
            return torch.cat([chunk_ce(h[c:c + step], targets[c:c + step])
                              for c in range(0, h.shape[0], step)])
        total = h.new_zeros((), dtype=torch.float32)
        for c in range(0, h.shape[0], step):
            args = (h[c:c + step], targets[c:c + step])
            total = total + (checkpoint(chunk_ce, *args, use_reentrant=False)
                             if torch.is_grad_enabled() else chunk_ce(*args))
        return total


def batch_rows(seg_lens, mask, device):
    """Packed segments + per-token loss mask -> (segments, rows, target idx):
    token i (mask=1, not a segment start) is predicted from hidden row i-1."""
    segs, off = [], 0
    for n in seg_lens:
        segs.append((off, off + int(n)))
        off += int(n)
    m = torch.as_tensor(mask, dtype=torch.bool)
    for s, _ in segs:
        m[s] = False
    tgt = m.nonzero().squeeze(1)
    return segs, (tgt - 1).to(device), tgt.to(device)


def grad_reach(adapters, flat_grad_of) -> list:
    """Adapter names whose B received no gradient (graph cut below them)."""
    bad = []
    for name in adapters:
        gb = flat_grad_of(name)
        if gb is None or not bool(gb.ne(0).any()):
            bad.append(name)
    return bad


__all__ = ["DEFAULT_TARGETS", "TrainMode", "batch_rows", "check_supported", "grad_reach",
           "install_adapters", "language_model", "SingleGroup", "chunk_gated_delta_rule"]
