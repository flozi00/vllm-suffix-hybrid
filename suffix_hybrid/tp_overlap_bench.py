# SPDX-License-Identifier: Apache-2.0
"""TP micro-batch overlap: toy DeepSeek-V3.2 stack + on-GPU oracle / bench.

Boot gates tp_overlap_oracle / tp_overlap_bench (sitecustomize.py). The toy
has exactly the vLLM DeepseekV32Model interface tp_overlap.glm_segments drives
(layers[i].input_layernorm / self_attn(positions=, hidden_states=) /
post_attention_layernorm / mlp, norm, embed_input_ids, topk_indices_buffer)
and prod's TP layout: vocab-parallel embedding (all-reduce), MLA-style
attention (column-parallel q_b, row-parallel o_proj -> partial) over a
per-request KV cache read through the toy forward context, a DSA indexer on
IndexShare leader layers (hisparse_mtp_patch.oracle.index_leaders) whose
top-k rows the follower layers re-read, routed experts split EP=TP plus a
TP-sharded shared expert (partial). Comm is real: vLLM PyNccl (optionally a
nccl_split band spec) at prod message sizes (tokens x hidden x 2 B).

    python -m suffix_hybrid.tp_overlap_bench oracle [--tp 2,4,8] [--tokens 48,96,192]
    python -m suffix_hybrid.tp_overlap_bench bench  [--tp 2,4,8] [--tokens 48,96,192,384]
        [--layers 4] [--spec default|allreduce:ring/Simple|...] [--iters 20]

oracle PASS (exit 0) = on every rank and token count, for both schedules
(split: every segment per micro-batch; hybrid: MLP/MoE on the whole batch):
overlapped forward (eager and CUDA-graph replay, replayed twice) bit-identical
to the SAME schedule with the comm stream = the compute stream (same kernels,
same message sizes, no concurrency), and the serialized split schedule
bit-identical to the model's own forward run per micro-batch. Differences to
the unsplit forward are reported, not gated (GPU kernels are not batch-
invariant). bench: CUDA-graph step ms of unsplit / split-sequential /
overlap-split / overlap-hybrid / compute-only / comm-only (2L+1 all-reduces),
MAX over ranks, + a 78-layer extrapolation.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
from torch import nn
from torch.nn import functional as F

from suffix_hybrid import tp_overlap as tpo

MARK = "[suffix tp-overlap]"
Q = 6  # tokens per request per decode step (MTP k=5)
CTX = {"req": None}  # the toy ForwardContext: request row of every token


def mm(x, w, exact):
    """x @ w; exact=True is row-invariant on CPU (no M-dependent GEMM path)."""
    return (x[:, :, None] * w).sum(1) if exact else x @ w


class RMSNorm(nn.Module):
    """vLLM RMSNorm semantics: norm(x) or (norm(x + r), x + r)."""

    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        y = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps))
        y = y.to(x.dtype) * self.weight
        return y if residual is None else (y, residual)


def _w(shape, seed, dev, dtype, scale=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    t = torch.randn(shape, generator=g, device=dev, dtype=torch.float32)
    return (t * (scale if scale is not None else shape[-2] ** -0.5)).to(dtype)


class Attention(nn.Module):
    def __init__(self, c, rank, tp, seed, leader, topk_buf, dev, dtype, exact):
        super().__init__()
        hl, d = c["heads"] // tp, c["head_dim"]
        self.hl, self.d, self.leader, self.exact, self.topk = hl, d, leader, exact, c["topk"]
        self.topk_indices_buffer = topk_buf
        self.wqa = nn.Parameter(_w((c["hidden"], c["q_lora"]), seed, dev, dtype), False)
        self.q_norm = RMSNorm(c["q_lora"]).to(dev, dtype)
        self.wqb = nn.Parameter(_w((c["q_lora"], hl * d), seed + 1 + 7 * rank, dev, dtype), False)
        self.wo = nn.Parameter(_w((hl * d, c["hidden"]), seed + 2 + 7 * rank, dev, dtype), False)
        self.register_buffer("kv", _w((c["max_reqs"], c["ctx"], d), seed + 3, dev, dtype, 1.0))
        if leader:
            self.wiq = nn.Parameter(_w((c["hidden"], c["idx_dim"]), seed + 4, dev, dtype), False)
            self.register_buffer("idx_k", _w((c["max_reqs"], c["ctx"], c["idx_dim"]), seed + 5,
                                             dev, dtype, 1.0))

    def forward(self, positions, hidden_states):
        x, req, n = hidden_states, CTX["req"], hidden_states.shape[0]
        if req is None or req.shape[0] != n:
            raise RuntimeError("toy forward context does not match this (micro-)batch")
        q = mm(self.q_norm(mm(x, self.wqa, self.exact)), self.wqb, self.exact).view(n, self.hl, self.d)
        if self.leader:  # DSA indexer: write the top-k rows the followers re-read
            iq, keys = mm(x, self.wiq, self.exact), self.idx_k[req]
            s = (iq[:, None] * keys).sum(-1) if self.exact else torch.einsum("nd,ncd->nc", iq, keys)
            pos = torch.arange(s.shape[1], device=s.device)
            s = s.float() - 1e-3 * (pos[None] - positions[:, None] % s.shape[1]).abs()
            self.topk_indices_buffer[:n] = s.topk(self.topk, dim=-1).indices.to(torch.int32)
        kv = self.kv[req[:, None], self.topk_indices_buffer[:n].long()]  # (n, topk, d)
        if self.exact:
            p = ((q[:, :, None] * kv[:, None]).sum(-1) * self.d ** -0.5).float().softmax(-1)
            o = (p.to(kv.dtype)[..., None] * kv[:, None]).sum(2)
        else:
            p = (torch.einsum("nhd,ntd->nht", q, kv) * self.d ** -0.5).float().softmax(-1)
            o = torch.einsum("nht,ntd->nhd", p.to(kv.dtype), kv)
        return mm(o.reshape(n, -1), self.wo, self.exact)  # row-parallel: TP-partial


class MLP(nn.Module):
    """Column-parallel gate_up + row-parallel down (dense layer / shared expert)."""

    def __init__(self, hidden, inter, seed, dev, dtype, exact):
        super().__init__()
        self.exact = exact
        self.w13 = nn.Parameter(_w((hidden, 2 * inter), seed, dev, dtype), False)
        self.w2 = nn.Parameter(_w((inter, hidden), seed + 1, dev, dtype), False)

    def forward(self, x):
        g, u = mm(x, self.w13, self.exact).chunk(2, dim=-1)
        return mm(F.silu(g) * u, self.w2, self.exact)


class MoE(nn.Module):
    """Sigmoid top-k routing, local experts = EP rank slice, + shared expert."""

    def __init__(self, c, rank, tp, seed, dev, dtype, exact):
        super().__init__()
        self.ne, self.el, self.top, self.exact = c["experts"], c["experts"] // tp, c["top_e"], exact
        self.e0 = rank * self.el
        h, i = c["hidden"], c["moe_inter"]
        self.wg = nn.Parameter(_w((h, c["experts"]), seed, dev, dtype), False)
        self.w13 = nn.Parameter(_w((self.el, h, 2 * i), seed + 1 + 7 * rank, dev, dtype), False)
        self.w2 = nn.Parameter(_w((self.el, i, h), seed + 2 + 7 * rank, dev, dtype), False)
        self.shared = MLP(h, c["shared_inter"] // tp, seed + 3 + 7 * rank, dev, dtype, exact)

    def forward(self, x):
        n = x.shape[0]
        topv, topi = torch.sigmoid(mm(x, self.wg, self.exact).float()).topk(self.top, dim=-1)
        w = (topv / topv.sum(-1, keepdim=True)).to(x.dtype)
        local = topi - self.e0
        mask = (local >= 0) & (local < self.el)
        e = local.clamp(0, self.el - 1)
        if self.exact:  # per (token, slot) expert GEMV: row-invariant
            g, u = (x[:, None, :, None] * self.w13[e]).sum(2).chunk(2, dim=-1)
            y = ((F.silu(g) * u)[..., None] * self.w2[e]).sum(2)
            routed = (y * (w * mask)[..., None]).sum(1)
        else:  # fixed-capacity grouped GEMM: static shapes, CUDA-graph capturable
            cap = min(n, max(4, -(-2 * n * self.top // self.ne)))  # 2x the mean load
            onehot = ((e[..., None] == torch.arange(self.el, device=e.device)) & mask[..., None]).long()
            slot = ((onehot.reshape(-1, self.el).cumsum(0) - 1) * onehot.reshape(-1, self.el)).sum(-1)
            keep = mask.reshape(-1) & (slot < cap)
            dest = torch.where(keep, e.reshape(-1) * cap + slot.clamp(max=cap - 1), self.el * cap)
            buf = x.new_zeros(self.el * cap + 1, x.shape[1])
            buf.index_copy_(0, dest, x[:, None].expand(n, self.top, -1).reshape(n * self.top, -1))
            g, u = torch.bmm(buf[:-1].view(self.el, cap, -1), self.w13).chunk(2, dim=-1)
            y = torch.bmm(F.silu(g) * u, self.w2).reshape(self.el * cap, -1)
            y = torch.cat([y, y.new_zeros(1, y.shape[1])])[dest]
            routed = (y * (w.reshape(-1) * keep)[:, None]).view(n, self.top, -1).sum(1)
        return routed + self.shared(x)  # TP-partial


class Layer(nn.Module):
    def __init__(self, model, attn, mlp, dev, dtype):
        super().__init__()
        self._model = [model]  # not a submodule
        self.input_layernorm = RMSNorm(attn.wqa.shape[0]).to(dev, dtype)
        self.post_attention_layernorm = RMSNorm(attn.wqa.shape[0]).to(dev, dtype)
        self.self_attn, self.mlp = attn, mlp

    def forward(self, positions, hidden_states, residual):
        """vLLM DeepseekV32DecoderLayer.forward (non-SP), all-reduce inline."""
        ar = self._model[0].ar
        if residual is None:
            residual, hidden_states = hidden_states, self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(ar(hidden_states), residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(ar(hidden_states), residual)
        return self.mlp(hidden_states), residual


def toy_config(**kw) -> dict:
    """CPU-test sized; the bench passes prod-like per-rank widths."""
    c = dict(vocab=128, hidden=64, heads=4, head_dim=16, q_lora=32, idx_dim=16, ctx=32,
             topk=8, experts=8, top_e=2, moe_inter=32, shared_inter=32, dense_inter=64,
             layers=4, dense_layers=1, index_freq=2, index_offset=2, max_reqs=16,
             max_tokens=96)
    c.update(kw)
    return c


class ToyModel(nn.Module):
    start_layer = 0
    use_sequence_parallel = False
    replicated_embed = False
    aux_hidden_state_layers = ()

    def __init__(self, c, rank, tp, ar, dev="cpu", dtype=torch.float32, exact=True, seed=0):
        super().__init__()
        from hisparse_mtp_patch.oracle import index_leaders

        self.ar, self.exact, self.c = ar, exact, c
        vl = c["vocab"] // tp
        self.v0, self.vl = rank * vl, vl
        self.embed = nn.Parameter(_w((vl, c["hidden"]), seed + 7 * rank, dev, dtype, 1.0), False)
        self.topk_indices_buffer = torch.zeros(c["max_tokens"], c["topk"], dtype=torch.int32,
                                               device=dev)
        lead = index_leaders(c["layers"], c["index_freq"], c["index_offset"], mtp=0)
        layers = []
        for i in range(c["layers"]):
            s = seed + 1000 * (i + 1)
            attn = Attention(c, rank, tp, s, lead[i], self.topk_indices_buffer, dev, dtype, exact)
            mlp = (MLP(c["hidden"], c["dense_inter"] // tp, s + 100 + 7 * rank, dev, dtype, exact)
                   if i < c["dense_layers"] else MoE(c, rank, tp, s + 200, dev, dtype, exact))
            layers.append(Layer(self, attn, mlp, dev, dtype))
        self.layers = nn.ModuleList(layers)
        self.end_layer = len(layers)
        self.norm = RMSNorm(c["hidden"]).to(dev, dtype)

    def embed_input_ids(self, ids):
        mask = (ids >= self.v0) & (ids < self.v0 + self.vl)
        return self.ar(self.embed[(ids - self.v0).clamp(0, self.vl - 1)] * mask[:, None])

    def forward(self, input_ids, positions):
        """vLLM DeepseekV32Model.forward (non-SP, one PP rank)."""
        h, residual = self.embed_input_ids(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        return self.norm(self.ar(h), residual)[0]


def toy_batch(c, n, seed=0, dev="cpu"):
    """Uniform decode batch: n // Q requests x Q tokens, distinct request rows."""
    g = torch.Generator().manual_seed(seed)
    reqs = torch.randperm(c["max_reqs"], generator=g)[: n // Q].repeat_interleave(Q)
    ids = torch.randint(0, c["vocab"], (n,), generator=g)
    pos = torch.randint(0, 4 * c["ctx"], (n,), generator=g)
    return ids.to(dev), pos.to(dev), reqs.to(dev)


def toy_enter(model, reqs, split, rebase=True):
    """enter(ub) for the toy: that micro-batch's request rows (its forward
    context) + its rows of the cross-layer top-k buffer."""
    from contextlib import contextmanager

    parts, starts = (reqs[:split], reqs[split:]), (0, split)

    @contextmanager
    def enter(ub):
        saved = CTX["req"]
        CTX["req"] = parts[ub]
        try:
            with tpo.rows_from(tpo.cross_layer_buffers(model) if rebase else [], starts[ub]):
                yield
        finally:
            CTX["req"] = saved
    return enter


def run_full(model, ids, pos, reqs):
    CTX["req"] = reqs
    try:
        return model(ids, pos)
    finally:
        CTX["req"] = None


def run_split_sequential(model, ids, pos, reqs, split):
    """The two micro-batches one after the other: same kernels and message
    sizes as the overlap, no concurrency -- the oracle's bitwise reference."""
    return torch.cat([run_full(model, ids[r], pos[r], reqs[r])
                      for r in (slice(0, split), slice(split, None))])


def run_overlap(model, ids, pos, reqs, split, streams, allreduce, log=None, rebase=True,
                joint_mlp=False):
    CTX["req"] = reqs  # the caller's (full-batch) context: joint segments run in it
    try:
        return tpo.glm_forward(model, ids, pos, split, streams, allreduce,
                               toy_enter(model, reqs, split, rebase), log, joint_mlp)
    finally:
        CTX["req"] = None


# ------------------------------------------------------------------ GPU side
def bench_config(tp: int, layers: int) -> dict:
    """GLM-5.3-like per-rank widths. moe_inter 576 in bf16 = the bytes of
    NVFP4 experts at intermediate 2048 (weight streaming is what doubles when
    a decode batch is split)."""
    return toy_config(vocab=8 * 4096, hidden=6144, heads=64, head_dim=128, q_lora=2048,
                      idx_dim=64, ctx=2048, topk=512, experts=256, top_e=8, moe_inter=576,
                      shared_inter=2048, dense_inter=12288, layers=layers, dense_layers=0,
                      index_freq=2, index_offset=2, max_reqs=128, max_tokens=768)


def _graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()  # warmup (lazy init, NCCL first use)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def _time_ms(g, iters):
    runs = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        runs.append(a.elapsed_time(b) / iters)
    return statistics.median(runs)


SCHEDULES = {"split": False, "hybrid": True}  # name -> joint_mlp


def _rank(rank, world, port, mode, tokens, layers, spec, iters, q):
    import torch.distributed as dist

    os.environ[tpo.ENV] = "1"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    out = {"rank": rank, "rows": []}
    try:
        from suffix_hybrid import nccl_split as ns

        comm = ns.make_comm(spec, dist.group.WORLD, dev)
        streams = tpo.CudaStreams(torch.cuda.Stream())  # outside capture: persistent events
        serial = tpo.CudaStreams(torch.cuda.current_stream())  # comm == compute: no overlap
        c = bench_config(world, layers)
        sync_ar = [lambda t: comm.all_reduce(t)]
        model = ToyModel(c, rank, world, lambda t: sync_ar[0](t), dev, torch.bfloat16, exact=False)

        def overlap_ar(x, o):
            comm.all_reduce(x, o)

        for n in tokens:
            ids, pos, reqs = toy_batch(c, n, seed=n, dev=dev)
            split = tpo.plan_split([Q] * (n // Q), Q, threshold=0)
            row = {"n": n, "split": split}
            with torch.inference_mode():
                full = lambda: run_full(model, ids, pos, reqs)  # noqa: E731
                seq = lambda: run_split_sequential(model, ids, pos, reqs, split)  # noqa: E731

                def ovl(st, joint):
                    return lambda: run_overlap(model, ids, pos, reqs, split, st, overlap_ar,
                                               joint_mlp=joint)
                if mode == "oracle":
                    base, checks = full(), {}
                    for name, joint in SCHEDULES.items():
                        ref, eager = ovl(serial, joint)(), ovl(streams, joint)()
                        g, static = _graph(ovl(streams, joint))
                        g.replay()
                        first = static.clone()
                        g.replay()
                        torch.cuda.synchronize()
                        checks.update({f"{name}-eager": torch.equal(eager, ref),
                                       f"{name}-graph": torch.equal(first, ref),
                                       f"{name}-replay": torch.equal(static, first)})
                        row[f"{name}_vs_unsplit"] = float((eager.float() - base.float()).abs().max())
                        if not joint:  # the adapter == the model's own forward per micro-batch
                            checks["split-serial==seq"] = torch.equal(ref, seq())
                        del g, static
                    row["checks"] = {k: bool(v) for k, v in checks.items()}
                else:
                    graphs = {"base": _graph(full)[0], "seq": _graph(seq)[0]}
                    for name, joint in SCHEDULES.items():
                        graphs[name] = _graph(ovl(streams, joint))[0]
                    sync_ar[0] = lambda t: t  # compute only
                    graphs["compute"] = _graph(full)[0]
                    sync_ar[0] = lambda t: comm.all_reduce(t)
                    x = torch.randn(n, c["hidden"], device=dev, dtype=torch.bfloat16)
                    graphs["comm"] = _graph(lambda: [comm.all_reduce(x)
                                                     for _ in range(2 * layers + 1)])[0]
                    for name, g in graphs.items():
                        dist.barrier()
                        row[name] = _time_ms(g, iters)
                    del graphs
            out["rows"].append(row)
    except Exception as exc:  # noqa: BLE001 - reported as FAIL by the parent
        out["error"] = repr(exc)[:500]
    q.put(out)
    dist.destroy_process_group()


def _launch(world, mode, tokens, layers, spec, iters):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29600 + os.getpid() % 1000 + world
    procs = [ctx.Process(target=_rank, args=(r, world, port, mode, tokens, layers, spec, iters, q))
             for r in range(world)]
    [p.start() for p in procs]
    res = sorted((q.get(timeout=1800) for _ in procs), key=lambda r: r["rank"])
    [p.join(timeout=120) for p in procs]
    return res


def _report(world, mode, res, layers) -> bool:
    errs = [f"rank{r['rank']}: {r['error']}" for r in res if "error" in r]
    if errs:
        print(f"{MARK} tp={world} {mode} ERROR {errs}", flush=True)
        return False
    ok = True
    for i, row in enumerate(res[0]["rows"]):
        rows = [r["rows"][i] for r in res]
        n = row["n"]
        if mode == "oracle":
            bad = sorted({k for r in rows for k, v in r["checks"].items() if not v})
            ok &= not bad
            print(f"{MARK} ORACLE tp={world} n={n} split={row['split']} "
                  f"{'FAIL ' + ','.join(bad) if bad else 'PASS'} (bitwise: overlapped eager + "
                  f"graph replay == same schedule serialized; split serialized == split-seq) "
                  + " ".join(f"max|{s}-unsplit| {max(r[s + '_vs_unsplit'] for r in rows):.3g}"
                             for s in SCHEDULES), flush=True)
            continue
        t = {k: max(r[k] for r in rows) for k in ("base", "seq", "compute", "comm", *SCHEDULES)}
        best = min(SCHEDULES, key=t.get)
        saved = t["base"] - t[best]
        print(f"{MARK} BENCH tp={world} n={n} layers={layers} ms: unsplit {t['base']:.3f} | "
              f"split-seq {t['seq']:.3f} | overlap-split {t['split']:.3f} | overlap-hybrid "
              f"{t['hybrid']:.3f} | compute {t['compute']:.3f} | comm {t['comm']:.3f} | best "
              f"{best} saves {saved:.3f} ({100 * saved / t['base']:.1f} %) | 78-layer est: "
              f"unsplit {78 * t['base'] / layers:.1f} {best} {78 * t[best] / layers:.1f}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("oracle", "bench"))
    ap.add_argument("--tp", default="2,4,8", help="TP sizes (capped by visible GPUs)")
    ap.add_argument("--tokens", default="", help="decode tokens (multiples of 6)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--spec", default="default", help="nccl_split communicator spec")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args(argv)
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    tps = [t for t in (int(x) for x in a.tp.split(",")) if 2 <= t <= gpus]
    if not tps:
        print(f"{MARK} SKIP: {gpus} GPU(s)", flush=True)
        return 0
    tokens = [int(x) for x in (a.tokens or ("48,96,192" if a.mode == "oracle"
                                             else "48,96,192,384")).split(",")]
    if any(n % Q or n < 2 * Q for n in tokens):
        raise SystemExit(f"--tokens must be multiples of {Q} and >= {2 * Q}")
    ok = True
    for world in tps:
        ok &= _report(world, a.mode, _launch(world, a.mode, tokens, a.layers, a.spec, a.iters),
                      a.layers)
    if a.mode == "oracle":
        print(f"{MARK} ORACLE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok or a.mode == "bench" else 1


if __name__ == "__main__":
    sys.exit(main())
