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

Unsupported (refused at install, serving untouched): pipeline parallel,
MoE / routed experts, MLA, attention logit soft-capping, non-GDN mamba
layers, encoder-only attention.
"""
from __future__ import annotations

import contextlib
import fnmatch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from suffix_hybrid.mtp_tune.lora import (
    LoRA, SingleGroup, _ReduceFromTP, copy_to_tp, in_out, tp_kind)
from suffix_hybrid.mtp_tune.train import FrozenLinear, _vocab_range, vocab_parallel_ce

DEFAULT_TARGETS = (
    "*.qkv_proj", "*.q_proj", "*.o_proj",          # attention (incl. KV-shared)
    "*.in_proj_qkvz", "*.out_proj",                # GDN
    "*.gate_up_proj", "*.down_proj",               # dense MLP
)
FUSED_FLAGS = ("use_fused_qk_norm_rope_gate",)    # Qwen3NextAttention Triton fusion
CE_BYTES = 256 << 20                              # fp32 logits per CE chunk
GDN_HEAD_GROUP = 8                                # heads per checkpointed delta-rule group

_TRAIN = None   # the active TrainMode (only ever set inside an RPC)


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


def check_supported(lm, vllm_config=None):
    """Raise NotImplementedError naming the first module SFT cannot train."""
    if vllm_config is not None and vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("pipeline parallel")
    if vllm_config is not None and getattr(vllm_config.cache_config, "kv_sharing_fast_prefill", False):
        raise NotImplementedError("kv_sharing_fast_prefill (in-place buffer copies)")
    for name, m in lm.named_modules():
        cls = type(m).__name__
        if hasattr(m, "w13_weight") or cls.endswith(("FusedMoE", "RoutedExperts")):
            raise NotImplementedError(f"MoE experts ({name}: {cls})")
        if "MLA" in cls:
            raise NotImplementedError(f"MLA attention ({name}: {cls})")
        if is_attention(m):
            if getattr(m.impl, "logits_soft_cap", None):
                raise NotImplementedError(f"attention soft-capping ({name})")
            if str(getattr(m, "attn_type", "decoder")).lower().split(".")[-1] != "decoder":
                raise NotImplementedError(f"non-causal attention ({name})")
        if hasattr(m, "get_state_shape") and not is_gdn(m):
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
            return out + ((x.to(a.dtype) @ a) @ layer.sft_B * layer.sft_scale).to(out.dtype)

        _CLASSES[cls] = type(f"Sft{cls.__name__}", (cls,), {"apply": apply, "_sft_base": cls})
    return _CLASSES[cls]


def install_adapters(lm, targets=DEFAULT_TARGETS, rank=16, alpha=32.0, tp_rank=0, dtype=None):
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
                    int(getattr(m, "tp_size", 1) or 1), dt, next(m.parameters()).device)
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
        for h0 in range(0, ins[0].shape[1], GDN_HEAD_GROUP):
            sl = slice(h0, h0 + GDN_HEAD_GROUP)
            with torch.enable_grad():
                part = [t[:, sl].detach().requires_grad_(True) for t in ins]
                o = chunk_gated_delta_rule(*(x[None] for x in part))[0]
                for dst, gr in zip(grads, torch.autograd.grad(o, part, do[:, sl])):
                    dst[:, sl] = gr
        return (*grads, None)


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
        h = torch.empty(x.shape, dtype=x.dtype, pin_memory=x.is_cuda)
        slot.append(h.copy_(x.detach(), non_blocking=True))
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


class TrainMode(contextlib.ContextDecorator):
    """Context in which ``lm``'s live modules compute with autograd.
    ``params``: {id(layer): (A, B, scale)} trainable views; ``segments``:
    [(start, end)] of the packed sequences (attention / GDN never cross them).
    """

    def __init__(self, sft, params, segments, grad=True, offload_bytes=0, check=False):
        self.s, self.params, self.segments = sft, params, segments
        self.grad, self.check = grad, check
        self.offload_left = offload_bytes   # host budget for parked layer inputs
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
            mods = self.s.__dict__.get("_sft_mods")
            if mods is None:   # classify once per model (thousands of modules)
                mods = self.s.__dict__["_sft_mods"] = [
                    (m, "attn" if is_attention(m) else "gdn" if is_gdn(m) else
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
            for m, kind, compiled, flags in mods:
                if kind == "attn":
                    self._set(m, "forward", lambda q, k, v, *a, _m=m, **kw: self.attention(_m, q, k, v))
                elif kind == "gdn":
                    self._set(m, "forward", lambda hidden_states, _m=m: self.gdn(_m, hidden_states))
                elif kind == "op":
                    self._set(m, "_forward_method", m.forward_native)
                if compiled:
                    self._set(m, "do_not_compile", True)
                for flag in flags:
                    if getattr(m, flag):
                        self._set(m, flag, False)
            for layer in self.s.inner.layers:
                self._set(layer, "forward", self._checkpointed(layer.forward))
            if self.g.world_size > 1:
                raw = _Raw(self.g, self.g.all_reduce)
                self._set(self.g, "all_reduce", lambda x: _ReduceFromTP.apply(x, raw))
            self._native_ir(st)
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

    def _checkpointed(self, fwd):
        def run(*args, **kwargs):
            if not torch.is_grad_enabled():
                return fwd(*args, **kwargs)
            # float tensors (hidden states / residual) travel through the
            # checkpoint as host copies; ints (positions) and the rest stay
            slots = [(i, a) for i, a in enumerate(args)] + list(kwargs.items())
            keys = [k for k, v in slots if torch.is_tensor(v) and v.is_floating_point()]
            size = sum(v.numel() * v.element_size() for k, v in slots if k in keys)
            parked = {}
            if size <= self.offload_left:   # past the host budget inputs stay on the GPU
                self.offload_left -= size
                parked = {k: [] for k in keys}

            def body(*vals):
                got = {k: (_Unpark.apply(v, parked[k]) if parked else v) for k, v in zip(keys, vals)}
                a = [got.get(i, v) for i, v in enumerate(args)]
                kw = {k: got.get(k, v) for k, v in kwargs.items()}
                return fwd(*a, **kw)

            vals = [(_Park.apply(v, parked[k]) if parked else v) for k, v in slots if k in keys]
            return checkpoint(body, *vals, use_reentrant=False)
        return run

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
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, is_causal=mask is None,
                                               scale=m.impl.scale)
            out.append(o[0].transpose(0, 1))
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

    def _rule_grouped(self, q, k, v, g, beta):
        """[T, H, D] delta rule. With the serving kernel available: its
        forward + a grouped torch backward (_FlaRule). Without (CPU tests):
        the torch form per checkpointed head group."""
        if self.fla is not None:
            if not torch.is_grad_enabled():
                return self.fla(q[None], k[None], v[None], g[None], beta[None])[0].to(v.dtype)
            return _FlaRule.apply(q, k, v, g, beta, self.fla)
        heads = GDN_HEAD_GROUP
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
        h = self.s.inner(input_ids=ids, positions=pos, intermediate_tensors=None)
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
