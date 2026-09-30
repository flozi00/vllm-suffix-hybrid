# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel micro-batch overlap: hide TP all-reduces behind compute.

Design, hazards and the remaining serving integration: docs/tp-overlap.md.

A TP decoder stack is a chain of SEGMENTS separated by all-reduces
(DeepseekV32DecoderLayer: [norm, attention -> partial] AR [norm, MLP/MoE ->
partial] AR ... final norm). With the batch split into two micro-batches at a
request boundary, `pipelined_forward` runs segment k of micro-batch 0, issues
its all-reduce on ONE side stream, runs segment k of micro-batch 1 (overlapping
that all-reduce), issues its all-reduce, then segment k+1 of micro-batch 0
after waiting for its all-reduce, and so on: per segment max(comm, compute)
instead of comm + compute.

Single thread, program order = issue order (no ubatch threads, no CPU
ping-pong), so it is CUDA-graph capturable as one fork/join per all-reduce:
  compute: ... seg(k, ub) -> record ready[ub]
  comm:    wait ready[ub] -> all_reduce(partial -> out) -> record done[ub]
  compute: wait done[ub] -> seg(k+1, ub) ...
NCCL ordering: every overlapped all-reduce (whatever nccl_split band
communicator it uses) is enqueued on the same comm stream in program order,
identical on every rank, and at most one NCCL kernel per GPU is in flight;
collectives outside the pipelined region (embedding, lm_head) run on the
compute stream strictly before the first ready / after the last done.
Allocator safety without record_stream: the all-reduce output is allocated on
the compute stream before the fork and the partial input stays referenced
until the compute stream has waited on done.

Numerics: each micro-batch runs exactly the ops of the unsplit forward on its
rows; only scheduling differs. Bit-identical to the unsplit forward when every
kernel is row-invariant (proved on CPU at TP=2), and always bit-identical to
running the two micro-batches one after the other (the on-GPU oracle's gate).

Gate: SUFFIX_TP_OVERLAP=1 (default off); SUFFIX_TP_OVERLAP_MIN_TOKENS (48).
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager, nullcontext

ENV = "SUFFIX_TP_OVERLAP"
MIN_TOKENS_ENV = "SUFFIX_TP_OVERLAP_MIN_TOKENS"
DEFAULT_MIN_TOKENS = 48
TAG = "suffix tp-overlap"


class DriftError(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == "1"


def min_tokens() -> int:
    return int(os.environ.get(MIN_TOKENS_ENV, "").strip() or DEFAULT_MIN_TOKENS)


def plan_split(query_lens, decode_query_len: int, threshold: int | None = None) -> int | None:
    """Tokens in micro-batch 0, or None = run the batch unsplit (the stock path).

    Splits only uniform decode batches (every request has exactly
    decode_query_len tokens, e.g. MTP verify k+1) with >= 2 requests and
    >= threshold tokens, at the request boundary nearest the middle (ub0 gets
    the odd request), so no request straddles micro-batches. Gate off,
    prefill, mixed or ragged batches -> None.
    """
    if not enabled():
        return None
    lens = list(query_lens)
    if len(lens) < 2 or any(q != decode_query_len for q in lens):
        return None
    if sum(lens) < (min_tokens() if threshold is None else threshold):
        return None
    return (len(lens) + 1) // 2 * decode_query_len


# ----------------------------------------------------------------- streams
class CudaStreams:
    """compute = the stream current while the driver runs (the capture stream
    under graph capture); comm = ONE persistent side stream for every
    overlapped all-reduce. Build once, outside capture: the four events are
    persistent per (kind, micro-batch) and each is waited on before it is
    recorded again, so reuse is safe eagerly and under capture."""

    def __init__(self, comm):
        import torch

        self.comm = comm
        self._events = {(k, ub): torch.cuda.Event() for k in ("ready", "done") for ub in (0, 1)}

    @property
    def compute(self):
        import torch

        return torch.cuda.current_stream()

    def record(self, stream, key):
        ev = self._events[key]
        ev.record(stream)
        return ev

    def wait(self, stream, ev) -> None:
        stream.wait_event(ev)

    def run(self, stream, fn) -> None:
        import torch

        with torch.cuda.stream(stream):
            fn()

    def sync(self) -> None:
        """Nothing to do: the last segment runs on compute after its wait."""


class FakeStreams:
    """CPU stand-in that PROVES the event discipline: every launch is deferred
    and executed in a random order that respects only what a GPU guarantees
    (FIFO per stream, a wait blocks its stream until the record it refers to
    ran). A missing wait shows up as wrong numbers; a collective outside the
    comm stream as a cross-rank order mismatch."""

    def __init__(self, seed: int = 0):
        self.compute, self.comm = "compute", "comm"
        self.q: dict = {self.compute: [], self.comm: []}
        self.issued: dict = {}
        self.fired: dict = {}
        self.rng = random.Random(seed)

    def record(self, stream, key):
        n = self.issued[key] = self.issued.get(key, 0) + 1
        self.q[stream].append(("record", key, n))
        return key, n

    def wait(self, stream, ev) -> None:
        self.q[stream].append(("wait", *ev))

    def run(self, stream, fn) -> None:
        self.q[stream].append(("run", fn))

    def sync(self) -> None:
        while any(self.q.values()):
            ready = [s for s, q in self.q.items()
                     if q and (q[0][0] != "wait" or self.fired.get(q[0][1], 0) >= q[0][2])]
            if not ready:
                raise RuntimeError("stream deadlock (wait on a record that never runs)")
            op = self.q[self.rng.choice(ready)].pop(0)
            if op[0] == "record":
                self.fired[op[1]] = op[2]
            elif op[0] == "run":
                op[1]()


# ---------------------------------------------------------------- pipeline
def pipelined_forward(segments, inputs, streams, allreduce, enter=None, log=None, joint=()):
    """Run `segments` over the micro-batches `inputs`, overlapping all-reduces.

    segments: [fn(h, residual, ub) -> (out, residual)]; every segment but the
      last returns a TP-PARTIAL tensor that is all-reduced before the next one.
    inputs: per micro-batch first hidden state (already reduced).
    allreduce(x, out): sum x over the TP group into out, on the current stream.
    enter(ub): context manager active around every per-micro-batch segment
      (its forward context / cross-layer row buffers).
    log: optional list; the comm stream appends (segment, ub, shape) in the
      order collectives are issued.
    joint: indices of segments run ONCE on the micro-batches' rows
      concatenated (ub=None, caller's context), e.g. the MoE, whose split
      would stream every expert's weights once per micro-batch. Their
      all-reduces are still issued per micro-batch, so micro-batch 0's next
      segment overlaps micro-batch 1's all-reduce.
    Returns per micro-batch slots; after streams.sync(), slot["h"] is the
    output of the last segment.
    """
    enter = enter or (lambda ub: nullcontext())
    last = len(segments) - 1
    slots = [{"h": h, "res": None} for h in inputs]
    for k, seg in enumerate(segments):
        for group in ([list(range(len(slots)))] if k in joint else [[ub] for ub in range(len(slots))]):
            for ub in group:  # s["part"] (k-1) stays referenced until this wait: no early reuse
                if k:
                    streams.wait(streams.compute, slots[ub].pop("done"))
            streams.run(streams.compute, _compute([slots[ub] for ub in group], seg, group,
                                                  k < last, enter))
            if k == last:
                continue
            for ub in group:
                streams.wait(streams.comm, streams.record(streams.compute, ("ready", ub)))
                streams.run(streams.comm, _comm(slots[ub], allreduce, log, k, ub))
                slots[ub]["done"] = streams.record(streams.comm, ("done", ub))
    return slots


def _compute(ss, seg, group, partial, enter):
    def fn():
        import torch

        if len(ss) == 1:
            with enter(group[0]):
                out, res = seg(ss[0]["h"], ss[0]["res"], group[0])
            outs, ress = [out], [res]
        else:
            sizes = [s["h"].shape[0] for s in ss]
            res = None if ss[0]["res"] is None else torch.cat([s["res"] for s in ss])
            out, res = seg(torch.cat([s["h"] for s in ss]), res, None)
            outs, ress = out.split(sizes), [None] * len(ss) if res is None else res.split(sizes)
        for s, o, r in zip(ss, outs, ress):
            s["res"] = r
            if partial:  # the all-reduce output: allocated here (compute), filled by comm
                s["part"], s["h"] = o, torch.empty_like(o)
            else:
                s["h"] = o
    return fn


def _comm(s, allreduce, log, k, ub):
    def fn():
        x = s["part"]
        if log is not None:
            log.append((k, ub, tuple(x.shape)))
        allreduce(x, s["h"])
    return fn


@contextmanager
def rows_from(tensors, start: int):
    """Re-point shared cross-layer row buffers at rows [start:] in place.

    DSA IndexShare: an indexer layer writes topk_indices_buffer[:n] and the
    follower layers read it back several segments later; the sparse-MLA index
    group keeps its physical top-k workspace the same way. Micro-batch 1 runs
    between micro-batch 0's writer and readers, so each micro-batch needs its
    own rows. Tensor.set_ keeps the tensor OBJECT (every module holding it
    sees the new rows) and costs no copy; kernels launched meanwhile bake the
    shifted pointer (graph capture included)."""
    saved = [(t, t.storage_offset(), t.size(), t.stride()) for t in tensors]
    try:
        for t, off, size, stride in saved:
            if start:
                if start > size[0]:
                    raise ValueError(f"row offset {start} past a {size[0]}-row buffer")
                t.set_(t.untyped_storage(), off + start * stride[0],
                       (size[0] - start, *size[1:]), stride)
        yield
    finally:
        for t, off, size, stride in saved:
            t.set_(t.untyped_storage(), off, size, stride)


# ------------------------------------------------ GLM-5.3 / DeepSeek-V3.2
# The statements of vLLM 0.30.0 DeepseekV32DecoderLayer.forward /
# DeepseekV32Model.forward (vllm/models/deepseek_v32/nvidia/model.py) that
# glm_segments re-implements; any drift refuses the overlap.
ANCHORS = (
    "hidden_states = ( attn_in if attn_in is not None else self.input_layernorm(hidden_states) )",
    "hidden_states, residual = fused_allreduce_rms_norm( hidden_states, residual, self.input_layernorm )",
    "hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)",
    "hidden_states, residual = fused_allreduce_rms_norm( hidden_states, residual, "
    "self.post_attention_layernorm )",
    "hidden_states = self.mlp(hidden_states)",
    "hidden_states, residual = layer(positions, hidden_states, residual, attn_in)",
    "hidden_states, _ = fused_allreduce_rms_norm( hidden_states, residual, self.norm )",
)


def check_source(src: str) -> None:
    flat = " ".join(src.replace("(", "( ").replace(")", " )").split())
    missing = [a for a in ANCHORS if " ".join(a.replace("(", "( ").replace(")", " )").split()) not in flat]
    if missing:
        raise DriftError(f"DeepseekV32 forward drifted, missing: {missing}")


def glm_refusal(model, world_size: int) -> str | None:
    """Why the pipelined forward cannot serve this model instance (None = ok)."""
    if world_size < 2:
        return "tensor parallel size 1 (nothing to overlap)"
    if getattr(model, "use_sequence_parallel", False):
        return "sequence-parallel MoE (reduce-scatter/all-gather, not all-reduce)"
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        return "pipeline parallelism"
    if getattr(model, "aux_hidden_state_layers", ()):
        return "aux hidden states (EAGLE3)"
    for m in model.modules():
        group = getattr(getattr(m, "impl", None), "index_group", None)
        if type(group).__name__ == "HiSparseMLAIndexGroup":
            return ("HiSparse: hot-slot state is keyed by micro-batch-local request "
                    "rows (docs/tp-overlap.md H2)")
    return None


def cross_layer_buffers(model) -> list:
    """model.topk_indices_buffer + every sparse-MLA index group's workspaces."""
    out, seen = [], set()
    for t in [getattr(model, "topk_indices_buffer", None)] + [
            getattr(g, name, None)
            for m in model.modules()
            for g in [getattr(getattr(m, "impl", None), "index_group", None)] if g is not None
            for name in ("physical_topk_indices", "valid_topk_counts")]:
        if t is not None and id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def glm_segments(model, positions, attn_in=None):
    """DeepseekV32DecoderLayer.forward (non-SP) cut at its all-reduces:
    [attn_0, mlp_0, ..., attn_{L-1}, mlp_{L-1}, final norm]. `positions` /
    `attn_in` are per micro-batch lists. fused_allreduce_rms_norm(h, r, norm)
    == norm(all_reduce(h), r) off the FlashInfer fused path (never taken on
    PCIe without P2P); the all-reduce is the pipeline's."""
    segs = []
    for i, layer in enumerate(model.layers):
        segs += [_attn_seg(layer, positions, attn_in if i == 0 else None), _mlp_seg(layer)]
    segs.append(lambda h, res, ub: (model.norm(h, res)[0], None))
    return segs


def _attn_seg(layer, positions, attn_in):
    def seg(h, res, ub):
        if res is None:
            res, x = h, attn_in[ub] if attn_in is not None else layer.input_layernorm(h)
        else:
            x, res = layer.input_layernorm(h, res)
        return layer.self_attn(positions=positions[ub], hidden_states=x), res
    return seg


def _mlp_seg(layer):
    def seg(h, res, ub):
        x, res = layer.post_attention_layernorm(h, res)
        return layer.mlp(x), res
    return seg


def glm_forward(model, input_ids, positions, split: int, streams, allreduce,
                enter=None, log=None, joint_mlp: bool = False):
    """DeepseekV32Model.forward with the layer stack pipelined over the
    micro-batches rows [:split] and [split:]. The embedding (and its
    vocab-parallel all-reduce) runs unsplit on the compute stream first.
    joint_mlp: MLP/MoE segments run on the whole batch (no expert weight
    re-read; hides half of the all-reduce time instead of all of it)."""
    import torch

    attn_in = None
    if getattr(model, "replicated_embed", False):
        from vllm.model_executor.layers.fused_embed_norm import fused_embed_norm

        h, attn_in = fused_embed_norm(
            input_ids, model.embed_tokens.weight,
            chain_weight=model.layers[model.start_layer].input_layernorm.weight,
            eps=model.config.rms_norm_eps)
    else:
        h = model.embed_input_ids(input_ids)
    rows = (slice(0, split), slice(split, h.shape[0]))
    segs = glm_segments(model, [positions[r] for r in rows],
                        None if attn_in is None else [attn_in[r] for r in rows])
    joint = range(1, len(segs) - 1, 2) if joint_mlp else ()
    slots = pipelined_forward(segs, [h[r] for r in rows], streams, allreduce, enter, log, joint)
    streams.sync()
    return torch.cat([s["h"] for s in slots])


def vllm_enter(contexts, buffers, starts):
    """enter(ub) for vLLM: that micro-batch's ForwardContext (attention
    metadata, slot mappings, is_padding) + its rows of the cross-layer
    buffers."""
    from vllm import forward_context as fc

    @contextmanager
    def enter(ub):
        saved = fc._forward_context
        fc._forward_context = contexts[ub]
        try:
            with rows_from(buffers, starts[ub]):
                yield
        finally:
            fc._forward_context = saved
    return enter


def pynccl_allreduce(device_communicator):
    """allreduce(x, out) on the current stream through vLLM's PyNccl; with the
    nccl_split router installed, through the size band's communicator (all of
    them are driven from the one comm stream). QAR allocates on the calling
    stream and is lossy: refused."""
    qar = os.environ.get("SUFFIX_NCCL_QAR", "").strip().lower()
    if qar not in ("", "0", "off"):
        raise ValueError(f"{ENV} with SUFFIX_NCCL_QAR={qar} is not supported")
    base = device_communicator.pynccl_comm
    if base is None or base.disabled:
        raise ValueError("PyNccl communicator unavailable")
    pick = getattr(device_communicator, "_suffix_pick", None)

    def allreduce(x, out):
        comm = pick(x, base) if pick is not None else base
        comm.all_reduce(x, out)
    return allreduce
