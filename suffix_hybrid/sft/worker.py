# SPDX-License-Identifier: Apache-2.0
"""Worker side of SFT: adapters on the served model, the trainer, shard IO.

Every op is one ``collective_rpc("suffix_sft", op, payload)`` and runs on
all TP ranks in lockstep between engine steps (RPCs and execute_model share
the worker's queue, so serving never runs concurrently with an op). Ops
never raise into vLLM: they return ``{"error": ...}`` and drop the run.

Memory (per rank, adapters of P params): live adapters 2P bytes (bf16,
resident, read by serving); during a run only +2P working copy +2P grad on
the GPU, the fp32 master and both AdamW moments (12P) live in pinned host
memory and the optimizer step runs on the CPU. Activations are bounded by
the micro-batch token cap (checkpointed + host-offloaded layer inputs,
chunked vocab-parallel CE). Between micro-steps the run holds only the
working copy + grad; ``torch.cuda.empty_cache`` returns everything when a
run ends.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import torch

from suffix_hybrid.mtp_tune.lora import SingleGroup
from suffix_hybrid.sft import model as M

MARK = "[suffix sft]"
SFT = None          # this worker's WorkerSFT (set by the load_model hook)
INSTALL_ERROR = "not installed"


def _log(msg):
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


class Run:
    """One training run: flat bf16 working params on the GPU (views per
    adapter, replicated TP factors first so their grads all-reduce as one
    slice), fp32 master + AdamW on the host."""

    def __init__(self, sft, lr, weight_decay=0.0, clip=1.0):
        self.s, self.clip = sft, clip
        spans, off = {}, 0
        for want_rep in (True, False):
            for name, (_, lora) in sft.adapters.items():
                for f in ("A", "B"):
                    if (lora.replicated_factor() == f) == want_rep:
                        t = getattr(lora, f)
                        spans[(name, f)] = (off, tuple(t.shape))
                        off += t.numel()
            if want_rep:
                self.n_rep = off
        self.spans = spans
        dev, dt = sft.device, sft.dtype
        with torch.no_grad():
            self.flat = torch.empty(off, dtype=dt, device=dev)
            for (name, f), (o, shp) in spans.items():
                self.flat[o:o + _numel(shp)].copy_(getattr(sft.adapters[name][1], f).reshape(-1))
        self.flat.requires_grad_(True)
        self.master = self.flat.detach().float().cpu()
        if torch.cuda.is_available():
            self.master = self.master.pin_memory()
        self.opt = torch.optim.AdamW([self.master], lr=lr, weight_decay=weight_decay)
        self.steps = 0

    def view(self, name, f):
        o, shp = self.spans[(name, f)]
        return self.flat[o:o + _numel(shp)].view(shp)

    def params(self):
        return {id(layer): (self.view(n, "A"), self.view(n, "B"), lora.scale)
                for n, (layer, lora) in self.s.adapters.items()}

    def grad_of(self, name):
        g = self.flat.grad
        if g is None:
            return None
        o, shp = self.spans[(name, "B")]
        return g[o:o + _numel(shp)]

    def reduce_grads(self):
        """Sum the replicated factors' partial grads over TP (in place) ->
        the global grad norm (identical on every rank)."""
        g, grp = self.flat.grad, self.s.g
        if g is None:
            raise RuntimeError("step without gradients")
        with torch.no_grad():
            if grp.world_size > 1 and self.n_rep:
                g[:self.n_rep] = grp.all_reduce(g[:self.n_rep].contiguous())
            rep = g[:self.n_rep].float().square().sum()
            shard = g[self.n_rep:].float().square().sum()
            if grp.world_size > 1:
                shard = grp.all_reduce(shard.reshape(1))[0]
            return float((rep + shard).sqrt())

    def step(self):
        """Clip by the global grad norm, AdamW on the host, copy back.
        Bit-identical on every rank for the replicated slice."""
        norm = self.reduce_grads()
        if not torch.isfinite(torch.tensor(norm)):
            raise FloatingPointError(f"non-finite grad norm {norm}")
        g = self.flat.grad
        with torch.no_grad():
            self.master.grad = g.float().cpu() * min(1.0, self.clip / (norm + 1e-6))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.flat.grad = None
            self.flat.copy_(self.master)
        self.steps += 1
        return norm

    def tensors(self):
        return {(n, f): self.view(n, f).detach() for (n, f) in self.spans}


def _numel(shp):
    n = 1
    for d in shp:
        n *= d
    return n


class WorkerSFT:
    def __init__(self, model, vllm_config, cfg, group=None, worker=None):
        self.cfg, self.vllm_config = cfg, vllm_config
        self.vw = worker          # the vLLM Worker: sleep/wake_up live there
        self.g = group if group is not None and group.world_size > 1 else SingleGroup()
        self.lm, self.inner, self.head, self.lp = M.language_model(model)
        M.check_supported(self.lm, vllm_config)
        mc = getattr(vllm_config, "model_config", None)
        self.mrope = bool(getattr(mc, "uses_mrope", False))
        self.adapters = M.install_adapters(self.lm, cfg.targets, cfg.rank, cfg.alpha,
                                           self.g.rank_in_group, getattr(mc, "dtype", None))
        ac = getattr(vllm_config, "additional_config", None)
        if isinstance(ac, dict):   # compile-cache key: graphs built for these adapters only
            ac["suffix_sft"] = dict(rank=cfg.rank, alpha=cfg.alpha, targets=list(cfg.targets))
        if not self.adapters:
            raise RuntimeError(f"no linear matched {cfg.targets}")
        first = next(iter(self.adapters.values()))[1]
        self.device, self.dtype = first.A.device, first.A.dtype
        self.run = None
        # basename: the same weights staged at /models/x or s3://.../x match
        self.model_id = os.path.basename(str(getattr(mc, "model", "?")).rstrip("/"))
        self.fingerprint = hashlib.sha256(json.dumps(
            dict(model=self.model_id, tp=self.g.world_size, rank=cfg.rank, alpha=cfg.alpha,
                 shapes={n: [list(lo.A.shape), list(lo.B.shape)]
                         for n, (_, lo) in self.adapters.items()}),
            sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ ops
    def kv_free(self, _p=None):
        """Unmap the KV cache for the duration of a run (vLLM sleep mode).

        ``sleep(1)`` offloads the weights to host and *discards* every other
        tagged allocation -- the KV cache. ``wake_up(["weights"])`` brings the
        weights back and leaves the KV cache unmapped, so training gets the
        whole card minus the weights. ``kv_restore`` maps it back at the same
        virtual address, so the captured CUDA graphs stay valid. Serving is
        paused while the cache is gone: the run owns the card.
        """
        if not self.cfg.free_kv or self.vw is None:
            return dict(ok=False, reason="free_kv off")
        try:
            self.vw.sleep(1)
            self.vw.wake_up(["weights"])
        except Exception as exc:  # noqa: BLE001 - fall back to sharing the card
            return dict(ok=False, reason=f"{type(exc).__name__}: {exc}")
        return dict(ok=True, free_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 2))

    def kv_restore(self, _p=None):
        if not self.cfg.free_kv or self.vw is None:
            return dict(ok=False, reason="free_kv off")
        try:
            self.vw.wake_up()
        except Exception as exc:  # noqa: BLE001
            return dict(ok=False, reason=f"{type(exc).__name__}: {exc}")
        return dict(ok=True, free_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 2))

    def info(self, _p=None):
        n = sum(lo.A.numel() + lo.B.numel() for _, lo in self.adapters.values())
        return dict(adapters=len(self.adapters), params=n * self.g.world_size, rank=self.cfg.rank,
                    alpha=self.cfg.alpha, tp=self.g.world_size, fingerprint=self.fingerprint,
                    model=self.model_id, targets=list(self.cfg.targets),
                    run=self.run is not None)

    def begin(self, p):
        self.run = Run(self, float(p["lr"]), float(p.get("weight_decay", 0.0)),
                       float(p.get("clip", 1.0)))
        return dict(ok=True)

    def _batch(self, p):
        ids = torch.as_tensor(p["ids"], dtype=torch.long, device=self.device)
        segs, rows, tgt = M.batch_rows(p["seg_lens"], p["mask"], self.device)
        return ids, segs, rows, tgt

    def micro(self, p):
        if self.run is None:
            raise RuntimeError("micro without begin")
        free = self._mem_vote()
        ids, segs, rows, tgt = self._batch(p)
        if not tgt.numel():
            raise ValueError("micro-batch without completion tokens")
        check = bool(p.get("check"))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device)
        t0 = time.perf_counter()
        with M.TrainMode(self, self.run.params(), segs, grad=True,
                         offload_bytes=int(self.cfg.offload_gib * 2**30), check=check) as tm:
            loss = tm.nll(ids, rows, tgt)
            fwd = float(loss.detach())        # syncs: forward time
            t1 = time.perf_counter()
            fwd_peak = (torch.cuda.max_memory_allocated(self.device)
                        if self.device.type == "cuda" else 0)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            (loss / float(p["norm"])).backward()
            _sync()
        out = dict(loss_sum=fwd, tokens=int(tgt.numel()), fwd_ms=round((t1 - t0) * 1e3, 1),
                   bwd_ms=round((time.perf_counter() - t1) * 1e3, 1))
        if self.device.type == "cuda":   # what training adds on top of serving's resident set
            out.update(peak_gib=round((torch.cuda.max_memory_allocated(self.device) - base) / 2**30, 3),
                       fwd_peak_gib=round((fwd_peak - base) / 2**30, 3),
                       free_gib=round(free / 2**30, 2), host_pinned_gib=_pinned_gib())
        if check:
            bad = M.grad_reach(self.adapters, self.run.grad_of)
            bad += [f"op:{c}" for c in dict.fromkeys(tm.cut)]
            flag = torch.tensor([float(len(bad))], device=self.device)
            if self.g.world_size > 1:
                flag = self.g.all_reduce(flag)
            if float(flag[0]):
                raise RuntimeError(f"gradient cut in the training forward (an op without "
                                   f"autograd): {len(bad)} adapters / modules, e.g. {bad[:4]}")
        return out

    def step(self, _p=None):
        return dict(grad_norm=self.run.step(), steps=self.run.steps)

    def eval(self, p):
        """Summed NLL with the LIVE adapters (what serving computes)."""
        self._mem_vote()
        ids, segs, rows, tgt = self._batch(p)
        live = {id(layer): (lo.A, lo.B, lo.scale) for layer, lo in self.adapters.values()}
        with M.TrainMode(self, live, segs, grad=False) as tm:
            loss = tm.nll(ids, rows, tgt, per_token=bool(p.get("per_token")))
        if p.get("per_token"):
            return dict(loss_sum=float(loss.sum()), tokens=int(tgt.numel()),
                        nll=[round(x, 4) for x in loss.tolist()])
        return dict(loss_sum=float(loss), tokens=int(tgt.numel()))

    def diag(self, p):
        """Dev only (SUFFIX_SFT_DIAG=1): per-token NLL of one example under
        toggled training-forward variants {name: {toggle: value}}."""
        if os.environ.get("SUFFIX_SFT_DIAG", "").strip() != "1":
            raise PermissionError("SUFFIX_SFT_DIAG != 1")
        self._mem_vote()
        ids, segs, rows, tgt = self._batch(p)
        live = {id(layer): (lo.A, lo.B, lo.scale) for layer, lo in self.adapters.values()}
        out = {}
        try:
            for name, flags in p["variants"].items():
                M.DIAG.clear()
                M.DIAG.update(flags)
                with M.TrainMode(self, live, segs, grad=False) as tm:
                    out[name] = [round(x, 4) for x in tm.nll(ids, rows, tgt, per_token=True).tolist()]
        finally:
            M.DIAG.clear()
        return dict(nll=out, tokens=int(tgt.numel()))

    def save(self, p):
        """This rank's shard of the run (or the live adapters) as
        <dir>/adapter-tp<r>-of-<n>.safetensors."""
        from safetensors.torch import save_file
        src = (self.run.tensors() if self.run is not None else
               {(n, f): getattr(lo, f) for n, (_, lo) in self.adapters.items() for f in "AB"})
        path = os.path.join(p["dir"], shard_name(self.g.rank_in_group, self.g.world_size))
        tmp = path + ".pending"
        save_file({f"{n}.{f}": t.detach().cpu().contiguous() for (n, f), t in src.items()}, tmp,
                  metadata={"fingerprint": self.fingerprint})
        os.replace(tmp, path)
        return dict(file=os.path.basename(path), fingerprint=self.fingerprint)

    def commit(self, _p=None):
        """Trained values -> the live (served) adapters, in place."""
        _sync()
        with torch.no_grad():
            for (n, f), t in self.run.tensors().items():
                getattr(self.adapters[n][1], f).copy_(t)
        self.abort()
        return dict(ok=True)

    def abort(self, _p=None):
        self.run = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # pinned offload buffers count against the pod's memory limit
            free_host = (getattr(torch._C, "_accelerator_emptyHostCache", None)
                         or getattr(torch._C, "_host_emptyCache", None))
            if free_host is not None:
                free_host()
        return dict(ok=True)

    def load(self, p):
        from safetensors import safe_open
        path = os.path.join(p["dir"], shard_name(self.g.rank_in_group, self.g.world_size))
        with safe_open(path, framework="pt", device="cpu") as f:
            if (f.metadata() or {}).get("fingerprint") != self.fingerprint:
                raise ValueError(f"checkpoint fingerprint {(f.metadata() or {}).get('fingerprint')} "
                                 f"!= served {self.fingerprint} (model / TP / rank / targets differ)")
            want = {f"{n}.{x}" for n in self.adapters for x in "AB"}
            if set(f.keys()) != want:
                raise ValueError("checkpoint adapter names differ from the served model")
            _sync()
            with torch.no_grad():
                for n, (_, lo) in self.adapters.items():
                    lo.load_(f.get_tensor(f"{n}.A"), f.get_tensor(f"{n}.B"))
        return dict(ok=True)

    def reset(self, _p=None):
        _sync()
        for _, lo in self.adapters.values():
            lo.zero_()
        return dict(ok=True)

    def _mem_vote(self):
        """Uniform refusal (no rank enters a collective-heavy forward alone)
        when any rank has less than min_free_gib free (driver free + cached)."""
        ok = 1.0
        if self.device.type == "cuda":
            # cached blocks may sit in CUDA-graph pools / fragments: count
            # only what the driver can hand out
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(self.device)
            ok = float(free >= self.cfg.min_free_gib * 2**30)
        else:
            free = 0
        flag = torch.tensor([1.0 - ok], device=self.device)
        if self.g.world_size > 1:
            flag = self.g.all_reduce(flag)
        if float(flag[0]):
            raise MemoryError(f"less than {self.cfg.min_free_gib} GiB free VRAM on a rank")
        return free


def _pinned_gib():
    try:
        st = torch.cuda.host_memory_stats()
        return round(st.get("allocated_bytes.current", st.get("allocated_bytes", 0)) / 2**30, 3)
    except Exception:  # noqa: BLE001 - telemetry only
        return None


def _sync():
    """An async-scheduled serving step may still be reading the adapters."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def shard_name(rank, world):
    return f"adapter-tp{rank}-of-{world}.safetensors"


OPS = ("info", "begin", "micro", "step", "eval", "save", "commit", "abort", "load", "reset", "diag",
       "kv_free", "kv_restore")
RUN_OPS = ("micro", "step", "commit")    # need the run on EVERY rank


def rpc(op, payload=None):
    """collective_rpc target (Worker.suffix_sft). Rank-uniform liveness vote
    first, so a rank without SFT never leaves the others in a collective."""
    group = SFT.g if SFT is not None else _tp_group()
    ok = SFT is not None and op in OPS and (op not in RUN_OPS or SFT.run is not None)
    if group is not None and group.world_size > 1:
        dev = SFT.device if SFT is not None else torch.device("cuda", torch.cuda.current_device())
        if float(group.all_reduce(torch.tensor([0.0 if ok else 1.0], device=dev))[0]):
            if SFT is not None and op in RUN_OPS:
                SFT.abort()
            return {"error": f"{op} refused: SFT or its run is missing on a TP rank"}
    if not ok:
        return {"error": INSTALL_ERROR if SFT is None else
                f"unknown op {op}" if op not in OPS else f"{op} without an active run"}
    t0 = time.perf_counter()
    try:
        out = getattr(SFT, op)(payload or {})
    except Exception as exc:  # noqa: BLE001 - never raise into the engine
        oom = isinstance(exc, (torch.OutOfMemoryError, MemoryError))
        if op not in ("info", "eval", "save", "diag"):
            SFT.abort()
        _log(f"{op} failed: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        return {"error": f"{type(exc).__name__}: {exc}", "oom": oom}
    out["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return out


def _tp_group():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return get_tp_group()
    except Exception:  # noqa: BLE001
        return None


def patch_worker(mod, cfg):
    """Worker.load_model hook (adapters BEFORE compile / graph capture) and
    the ``suffix_sft`` RPC method."""
    import functools
    W = mod.Worker
    if getattr(W.load_model, "_suffix_sft", False):
        return
    orig = W.load_model

    @functools.wraps(orig)
    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        global SFT, INSTALL_ERROR
        try:
            SFT = WorkerSFT(self.get_model(), self.vllm_config, cfg, _tp_group(), worker=self)
            info = SFT.info()
            _log(f"installed: {info['adapters']} adapters, {info['params'] / 1e6:.1f}M params, "
                 f"r={cfg.rank} tp={info['tp']} fingerprint={info['fingerprint']}")
        except Exception as exc:  # noqa: BLE001 - optional feature, serving unaffected
            SFT, INSTALL_ERROR = None, f"{type(exc).__name__}: {exc}"
            _log(f"not installed (serving unaffected): {INSTALL_ERROR}")
        return out

    load_model._suffix_sft = True
    W.load_model = load_model
    W.suffix_sft = lambda self, op, payload=None: rpc(op, payload)
