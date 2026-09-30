# SPDX-License-Identifier: Apache-2.0
"""LoRA adapters on the MTP layer's dense linears + TP autograd collectives.

Serving contract
----------------
``attach(layer, name, rank, alpha)`` puts a ``LoRA`` (A [K_local, r], B
[r, N_local], scale alpha/r) on a vLLM linear. vLLM linears compute their
local GEMM through ``layer.quant_method.apply(layer, x, bias)`` whatever the
quant method (bf16 / FP8 / NVFP4 / our SUFFIX_* kernels / the qwen
low-latency method), so the adapter replaces ``layer.quant_method`` with a
per-layer proxy whose ``apply`` returns ``base + (x @ A) @ B * scale`` and
delegates every other attribute to the original method. A module without a
quant method (plain ``nn.Linear``, e.g. GLM eh_proj) gets its instance
``forward`` wrapped instead. B starts at zero: an untrained adapter adds an
exact 0.0 to every output element (bitwise no-op, tested).

TP layout (no extra collective in the serving forward): the delta is added
where ``apply`` returns, i.e. BEFORE the layer's own gather / all-reduce.
  column-parallel (output sharded, incl. QKV / merged): A replicated [K, r],
      B sharded [r, N_local] -> the local output shard gets its own columns;
  row-parallel (input sharded): A sharded [K_local, r], B replicated [r, N]
      -> each rank adds a PARTIAL (x_s A_s) B that the layer's all-reduce
      sums to x A B;
  replicated / disable_tp: both full.
A is initialised from ONE seeded full [K_total, r] matrix (seed = layer
name) and sliced by rank, so every rank agrees on the conceptual adapter.

Backward consistency: every TP rank trains on identical data (rank 0's
windows are broadcast). The replicated factor of a sharded adapter (column
A, row B) receives only this rank's partial gradient from local autograd,
so its grad is all-reduced (summed) after backward (``allreduce_grads``);
the sharded factor's grad is local and exact. The frozen base structure
uses the Megatron operators below: ``copy_to_tp`` (fwd identity, bwd
all-reduce) where a replicated activation enters column-parallel compute,
``reduce_from_tp`` (fwd all-reduce, bwd identity) where row-parallel
partials are summed, ``gather_from_tp`` (fwd all-gather, bwd local slice)
for gather_output. With those, every rank ends a backward with identical
replicated grads and its own shard grads, so AdamW keeps replicated
factors bit-identical across ranks.

CUDA-graph in-place contract: the drafter's CUDA graphs capture the ADDRESS
of A and B. ``LoRA.load_(src)`` / ``zero_()`` only ever ``copy_`` into the
existing storage, so a captured graph replays with the new values without
re-capture. Rebinding ``lora.A = new_tensor`` would silently keep the graph
on the old values - never do it (tested).
"""
from __future__ import annotations

import fnmatch
import zlib

import torch


# ---------------------------------------------------------------------------
# TP group + autograd-aware collectives
# ---------------------------------------------------------------------------
class SingleGroup:
    """world_size 1: every collective is the identity."""
    world_size = 1
    rank_in_group = 0

    def all_reduce(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x

    def broadcast_tensor_dict(self, d=None, src=0):
        return d


def group_of(g):
    return g if g is not None else SingleGroup()


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        return ctx.group.all_reduce(dy.contiguous()), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return group.all_reduce(x.contiguous())

    @staticmethod
    def backward(ctx, dy):
        return dy, None


class _GatherFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group, ctx.n = group, x.shape[-1]
        return group.all_gather(x.contiguous(), dim=-1)

    @staticmethod
    def backward(ctx, dy):
        r = ctx.group.rank_in_group
        return dy[..., r * ctx.n:(r + 1) * ctx.n].contiguous(), None


def copy_to_tp(x, group):
    return x if group.world_size == 1 else _CopyToTP.apply(x, group)


def reduce_from_tp(x, group):
    return x if group.world_size == 1 else _ReduceFromTP.apply(x, group)


def gather_from_tp(x, group):
    return x if group.world_size == 1 else _GatherFromTP.apply(x, group)


# ---------------------------------------------------------------------------
# layer classification (vLLM LinearBase attribute surface)
# ---------------------------------------------------------------------------
def tp_kind(layer) -> str:
    """'row' | 'col' | 'rep' from vLLM's attribute surface (RowParallelLinear
    has input_is_parallel, ColumnParallelLinear gather_output; disable_tp /
    ReplicatedLinear have tp_size 1 and count as replicated)."""
    if int(getattr(layer, "tp_size", 1) or 1) <= 1:
        return "rep"
    if hasattr(layer, "input_is_parallel"):
        return "row"
    if hasattr(layer, "gather_output"):
        return "col"
    return "rep"


def in_out(layer) -> tuple[int, int]:
    """Local (K, N) of the layer's GEMM."""
    k = getattr(layer, "input_size_per_partition", None)
    n = getattr(layer, "output_size_per_partition", None)
    if k is None or n is None:
        w = layer.weight  # plain nn.Linear
        n, k = int(w.shape[0]), int(w.shape[1])
    return int(k), int(n)


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------
class LoRA:
    """Persistent adapter tensors; values change only through copy_."""

    def __init__(self, name, k, n, rank, alpha, kind, tp_rank=0, tp_size=1,
                 dtype=torch.bfloat16, device="cpu"):
        self.name, self.kind, self.r = name, kind, int(rank)
        self.scale = float(alpha) / float(rank)
        gen = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        k_total = k * tp_size if kind == "row" else k
        a_full = torch.randn(k_total, self.r, generator=gen) / (k_total ** 0.5)
        if kind == "row":
            a_full = a_full[tp_rank * k:(tp_rank + 1) * k]
        self.A = a_full.to(device=device, dtype=dtype).contiguous()
        self.B = torch.zeros(self.r, n, device=device, dtype=dtype)

    # replicated factor of a sharded adapter -> grad must be summed over TP
    def replicated_factor(self):
        return {"col": "A", "row": "B"}.get(self.kind)

    def delta(self, x, A=None, B=None):
        A = self.A if A is None else A
        B = self.B if B is None else B
        return ((x.to(A.dtype) @ A) @ B * self.scale).to(x.dtype)

    @torch.no_grad()
    def load_(self, A, B):
        self.A.copy_(A)
        self.B.copy_(B)

    @torch.no_grad()
    def zero_(self):
        self.B.zero_()


class _QuantProxy:
    """Per-layer stand-in for layer.quant_method: apply() adds the delta."""

    def __init__(self, inner, lora):
        self._inner, self._lora = inner, lora

    def apply(self, layer, x, bias=None, *args, **kwargs):
        out = self._inner.apply(layer, x, bias, *args, **kwargs)
        return out + self._lora.delta(x)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def base_apply(layer, x):
    """The frozen local GEMM (no bias, no collective) whatever the method."""
    qm = getattr(layer, "quant_method", None)
    if isinstance(qm, _QuantProxy):
        qm = qm._inner
    if qm is not None:
        return qm.apply(layer, x, None)
    return torch.nn.functional.linear(x, layer.weight)


def attach(layer, name, rank=16, alpha=32.0, tp_rank=0):
    """Install a zero-init adapter on `layer`; returns the LoRA."""
    if getattr(layer, "_mtp_lora", None) is not None:
        return layer._mtp_lora
    k, n = in_out(layer)
    kind = tp_kind(layer)
    w = next((p for p in layer.parameters() if p.dtype.is_floating_point
              and p.element_size() >= 2), None)
    dtype = w.dtype if w is not None else torch.bfloat16
    device = next(layer.parameters()).device
    lora = LoRA(name, k, n, rank, alpha, kind, tp_rank,
                int(getattr(layer, "tp_size", 1) or 1), dtype, device)
    qm = getattr(layer, "quant_method", None)
    if qm is not None:
        layer.quant_method = _QuantProxy(qm, lora)
    else:
        base = layer.forward
        layer.forward = lambda x, *a, **kw: base(x, *a, **kw) + lora.delta(x)
    layer._mtp_lora = lora
    return lora


def attach_all(model, patterns, rank=16, alpha=32.0, tp_rank=0):
    """Attach to every linear-like submodule whose qualified name matches a
    glob in `patterns`; returns {name: LoRA}."""
    out = {}
    for name, mod in model.named_modules():
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        if getattr(mod, "quant_method", None) is None and not isinstance(
                mod, torch.nn.Linear):
            continue
        out[name] = attach(mod, name, rank, alpha, tp_rank)
    return out


def allreduce_grads(params_by_lora, group):
    """Sum the replicated-factor grads over TP (one flat all-reduce)."""
    if group.world_size == 1:
        return
    grads = [p.grad for lora, p in params_by_lora if p.grad is not None]
    if not grads:
        return
    flat = group.all_reduce(torch.cat([g.reshape(-1) for g in grads]))
    off = 0
    for g in grads:
        g.copy_(flat[off:off + g.numel()].view_as(g))
        off += g.numel()
