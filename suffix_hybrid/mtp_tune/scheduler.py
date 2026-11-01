# SPDX-License-Identifier: Apache-2.0
"""Idle-time scheduling: engine-side idle hook + worker-side tuner.

Engine side (EngineCoreProc, the process that owns the scheduler): the busy
loop blocks in ``_process_input_queue`` when ``has_work()`` is False (no
running / waiting requests, no batch in flight). The hook waits a grace
period for input first; if the queue is still empty it issues ONE
``collective_rpc(worker_step)`` per micro-step and re-checks the queue
between micro-steps. RPCs are serialized with execute_model, so a training
micro-step never overlaps a serving step, and every TP rank executes the
same micro-step in lockstep (the RPC is the synchronization - no worker
thread, no rank-local idle decision that could desync collectives). A
request arriving mid-micro-step waits at most one micro-step; the hook
records that wait (RPC duration when the queue was non-empty afterwards) as
``max_added_ms``. Any engine-side exception or a worker reporting
``disabled`` turns the hook off for the process. DP engines are skipped.

Worker side (``Tuner``): rank 0 owns capture + planning; each micro-step
rank 0 broadcasts the plan (op + batch tensors, broadcast_tensor_dict), all
ranks agree on a free-memory flag (one scalar all-reduce), run the op on a
side CUDA stream, then synchronize that stream (idle: free) so the RPC
duration is the true added latency. Any exception disables tuning for the
process (logged once); serving is never touched.
"""
from __future__ import annotations

import os
import queue
import random
import sys
import time
import traceback

import torch

from suffix_hybrid.mtp_tune import lora as L
from suffix_hybrid.mtp_tune.capture import Store, wrap_propose
from suffix_hybrid.mtp_tune.gate import Gate, lang_tag
from suffix_hybrid.mtp_tune.train import Trainer, build_functional, collate

MARK = "[suffix mtp-tune]"
TUNER = None


def _log(msg):
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


class Tuner:
    def __init__(self, model, k, width, cfg, group=None, dtype=torch.bfloat16,
                 device="cpu", speculator=None, tokenizer=None):
        self.cfg, self.k, self.g = cfg, k, L.group_of(group)
        self.rank0 = self.g.rank_in_group == 0
        self.device = torch.device(device)
        self.loras = L.attach_all(model, cfg.targets, cfg.rank, cfg.alpha, self.g.rank_in_group)
        if not self.loras:
            raise RuntimeError(f"no MTP linear matched {cfg.targets}")
        self.fam = build_functional(model, self.g)
        self.trainer = Trainer(self.fam, self.loras, lr=cfg.lr, accum=cfg.accum,
                               target_ms=cfg.step_ms, max_windows=cfg.max_windows)
        self.gate = Gate(k, cfg.margin, cfg.min_anchors)
        self.store = None
        if self.rank0:
            row_bytes = width * torch.tensor([], dtype=dtype).element_size() + 16
            self.store = Store(width, k, int(cfg.mem_gib * 2**30) // row_bytes, cfg.window,
                               cfg.rate, cfg.ctx, max_rows_per_step=cfg.max_rows,
                               heldout_pct=cfg.heldout_pct, dtype=dtype, device=device)
            if speculator is not None:
                speculator.propose = wrap_propose(speculator.propose, self.store)
        self.tok = tokenizer
        self.rng = random.Random(0)
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self.disabled = self.rolled_back = False
        self.eval_queue, self.eval_steps_at = [], 0
        self.micro_ms_max, self.last_log = 0.0, time.monotonic()
        self.counts = dict(train=0, eval=0, skip_mem=0)

    # ---------------------------------------------------------------- plan
    def _tag(self, w):
        if "tag" not in w:
            try:
                w["tag"] = lang_tag(self.tok.decode(w["tok"].tolist())) if self.tok else "?"
            except Exception:
                w["tag"] = "?"
        return w["tag"]

    def _windows(self, recs):
        out = []
        for r in recs:
            w = self.store.materialize(r)
            if w is not None:
                w["tag"] = r.get("tag") or self._tag(w)
                r["tag"] = w["tag"]
                out.append(w)
        return out

    def plan(self):
        """Rank 0: next op (dict with CPU/GPU tensors) - pure host work."""
        c, s = self.cfg, self.store
        if c.rollback_file and not self.rolled_back and os.path.exists(c.rollback_file):
            return {"op": "rollback"}
        if self.eval_queue:
            recs = self.eval_queue[:self.trainer.windows_per_micro]
            del self.eval_queue[:len(recs)]
            return self._batch("eval", recs, last=not self.eval_queue)
        held = s.alive_windows(heldout=True)
        if (self.trainer.steps - self.eval_steps_at >= c.eval_every
                and len(held) >= c.min_eval_windows):
            self.eval_queue = held[-c.eval_windows:]
            self.eval_steps_at = self.trainer.steps
            self.gate.reset()
            return self.plan()
        train = s.alive_windows(heldout=False)
        if len(train) >= c.min_train_windows:
            n = min(self.trainer.windows_per_micro, len(train))
            return self._batch("train", self.rng.sample(train, n))
        return {"op": "none"}

    def _batch(self, op, recs, last=False):
        ws = self._windows(recs)
        if not ws:
            return {"op": "none"}
        b = collate(ws, self.device, self.fam_dtype())
        d1 = torch.full_like(b["tok"], -1)
        for i, w in enumerate(ws):
            d1[i, :w["d1"].shape[0]] = w["d1"].to(d1.device).long()
        b.update(op=op, last=last, tags=[w["tag"] for w in ws], d1=d1)
        return b

    def fam_dtype(self):
        return next(iter(self.loras.values())).A.dtype

    # ---------------------------------------------------------------- step
    def step(self, engine_stats=None):
        if self.disabled:
            return {"worked": False, "disabled": True}
        t0 = time.perf_counter()
        try:
            worked = self._step()
        except Exception as exc:  # noqa: BLE001 - never kill the engine
            self.disabled = True
            _log(f"DISABLED after trainer exception (serving unaffected): "
                 f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
            return {"worked": False, "disabled": True}
        ms = (time.perf_counter() - t0) * 1e3
        if worked:
            self.micro_ms_max = max(self.micro_ms_max, ms)
        self.maybe_log(engine_stats)
        return {"worked": worked, "ms": ms}

    def _step(self):
        fault = os.environ.get("SUFFIX_MTP_TUNE_FAULT_RANK")
        if fault is not None and int(fault) == self.g.rank_in_group:
            raise RuntimeError("injected fault (SUFFIX_MTP_TUNE_FAULT_RANK)")
        p = self.plan() if self.rank0 else None
        p = self.g.broadcast_tensor_dict(p, src=0)
        op = p["op"]
        if op == "none":
            return False
        if op == "rollback":
            for lo in self.loras.values():
                lo.zero_()
            self.rolled_back = True
            _log("ROLLBACK: live adapters zeroed, promotions off")
            return True
        if not self._mem_ok():
            self.counts["skip_mem"] += 1
            return False
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _Null()
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with ctx:
            if op == "train":
                _, ms = self.trainer.micro_step(p)
                self.trainer.adapt(ms, p["tok"].shape[0])
            else:
                live = {n: (lo.A, lo.B) for n, lo in self.loras.items()}
                self.gate.feed(self.fam, p, p["tags"], live, self.trainer.serving_params(), p["d1"])
        if self.stream is not None:
            self.stream.synchronize()
        self.counts[op] += 1
        if op == "eval" and p["last"]:
            d = {"promote": self.gate.decide() and not self.rolled_back and not self.cfg.shadow}
            d = self.g.broadcast_tensor_dict(d if self.rank0 else None, src=0)
            if d["promote"]:
                self.gate.promote(self.loras, self.trainer.serving_params())
                _log(f"PROMOTED #{self.gate.promotions}: {self._eval_str()}")
        return True

    def _mem_ok(self):
        ok = 1.0
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            ok = 1.0 if free >= self.cfg.min_free_gib * 2**30 else 0.0
        flag = torch.tensor([1.0 - ok], device=self.device)
        return float(self.g.all_reduce(flag)[0]) == 0.0

    # ----------------------------------------------------------- telemetry
    def _eval_str(self):
        e = self.gate.last
        if not e:
            return "eval=none"
        par = e["live_buckets"]
        parity = [v["parity"] for v in par.values() if v["parity"] is not None]
        per = " ".join(f"{t}:{v['acc_len']:.3f}->{e['cand_buckets'].get(t, {}).get('acc_len', 0):.3f}"
                       f"(n={v['n']})" for t, v in par.items())
        return (f"eval live={e['live']:.3f} cand={e['cand']:.3f} n={e['n']} promote={e['promote']} "
                f"parity_d1={sum(parity) / len(parity) if parity else float('nan'):.3f} [{per}]")

    def line(self, engine_stats=None):
        s = self.store.stats if self.store else {}
        t = self.trainer
        eng = engine_stats or {}
        return (f"windows train={s.get('windows_train', 0)} held={s.get('windows_held', 0)} "
                f"rows={s.get('rows', 0)} drops={s.get('dropped_steps', 0)} "
                f"gaps={s.get('gaps', 0)} evicted={s.get('evicted_windows', 0)} | "
                f"steps={t.steps} micro={t.micro} loss={t.loss_ewma if t.loss_ewma is not None else float('nan'):.4f} "
                f"w/micro={t.windows_per_micro} ms/w={t.ms_per_window or 0:.1f} "
                f"max_micro_ms={self.micro_ms_max:.1f} skip_mem={self.counts['skip_mem']} | "
                f"{self._eval_str()} promotions={self.gate.promotions} | "
                f"idle_used={eng.get('idle_used', 0):.1%} max_added_ms={eng.get('max_added_ms', 0):.1f} "
                f"rpcs={eng.get('rpcs', 0)}")

    def maybe_log(self, engine_stats):
        now = time.monotonic()
        if self.rank0 and now - self.last_log >= self.cfg.log_s:
            self.last_log = now
            _log(self.line(engine_stats))


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _tp_group():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return get_tp_group()
    except Exception:
        return None


def worker_step(worker=None, engine_stats=None):
    """collective_rpc target (runs on every worker, in lockstep). First a
    rank-uniform liveness vote: a rank whose tuner failed to install or got
    disabled must not leave the others waiting in a broadcast."""
    g = TUNER.g if TUNER is not None else _tp_group()
    ok = TUNER is not None and not TUNER.disabled
    if g is not None and g.world_size > 1:
        dev = TUNER.device if TUNER is not None else torch.device("cuda", torch.cuda.current_device())
        if float(g.all_reduce(torch.tensor([0.0 if ok else 1.0], device=dev))[0]) > 0:
            if TUNER is not None and not TUNER.disabled:
                TUNER.disabled = True
                _log("DISABLED: another TP rank has no live tuner")
            return {"worked": False, "disabled": True}
    if not ok:
        return {"worked": False, "disabled": True}
    return TUNER.step(engine_stats)


# ---------------------------------------------------------------------------
# engine side
# ---------------------------------------------------------------------------
class EngineIdle:
    def __init__(self, cfg, rpc=None):
        self.cfg, self.on = cfg, True
        self.rpc = rpc  # tests inject; default core.collective_rpc(worker_step)
        self.busy_s, self.t_start = 0.0, time.monotonic()
        self.max_added_ms, self.rpcs = 0.0, 0

    def snapshot(self):
        wall = max(time.monotonic() - self.t_start, 1e-9)
        return dict(idle_used=self.busy_s / wall, max_added_ms=self.max_added_ms, rpcs=self.rpcs)

    def _call(self, core):
        if self.rpc is not None:
            return self.rpc(self.snapshot())
        return core.collective_rpc(worker_step, timeout=self.cfg.rpc_timeout_s,
                                   args=(self.snapshot(),))[0]

    def run(self, core):
        if (not self.on or core.has_work() or not core.is_running()
                or not core.input_queue.empty()):
            return
        try:
            req = core.input_queue.get(timeout=self.cfg.idle_grace_ms / 1e3)
            core._handle_client_request(*req)
            return
        except queue.Empty:
            pass
        while self.on and core.input_queue.empty() and not core.has_work() and core.is_running():
            t0 = time.monotonic()
            res = self._call(core)
            dt = time.monotonic() - t0
            self.busy_s += dt
            self.rpcs += 1
            if not core.input_queue.empty():
                self.max_added_ms = max(self.max_added_ms, dt * 1e3)
            if not res or not res.get("worked"):
                if res and res.get("disabled"):
                    self.on = False
                return


def install_engine_hook(cfg):
    from vllm.v1.engine import core
    cls = core.EngineCoreProc
    orig = cls._process_input_queue
    if getattr(orig, "_mtp_tune", False):
        return
    idle = EngineIdle(cfg)
    dp_cls = getattr(core, "DPEngineCoreProc", ())

    def _process_input_queue(self):
        if idle.on and not isinstance(self, dp_cls):
            try:
                idle.run(self)
            except Exception as exc:  # noqa: BLE001
                idle.on = False
                _log(f"engine idle hook disabled: {type(exc).__name__}: {exc}")
        return orig(self)

    _process_input_queue._mtp_tune = True
    cls._process_input_queue = _process_input_queue
    _log("engine idle hook installed")


def install_worker_hook(cfg):
    """Patch the V2 GPUModelRunner.load_model: after the speculator loaded
    (and BEFORE capture_model records the drafter's CUDA graphs) attach the
    adapters and the capture wrapper."""
    import functools
    module = __import__("vllm.v1.worker.gpu.model_runner", fromlist=["GPUModelRunner"])
    original = module.GPUModelRunner.load_model
    if getattr(original, "_mtp_tune", False):
        return

    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        global TUNER
        try:
            spec = getattr(self, "speculator", None)
            if spec is None or type(spec).__name__ != "MTPSpeculator":
                _log(f"no MTPSpeculator ({type(spec).__name__}) - tuning off")
                return result
            from vllm.distributed.parallel_state import get_tp_group
            tok = None
            if cfg.lang:
                try:
                    from transformers import AutoTokenizer
                    mc = self.vllm_config.model_config
                    tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=mc.trust_remote_code)
                except Exception as exc:  # noqa: BLE001
                    _log(f"no tokenizer for language tags: {exc}")
            TUNER = Tuner(spec.model, int(spec.num_speculative_steps), int(spec.hidden_size), cfg,
                          group=get_tp_group(), dtype=spec.dtype, device=spec.device,
                          speculator=spec, tokenizer=tok)
            _log(f"installed: {len(TUNER.loras)} adapters r={cfg.rank} k={TUNER.k} "
                 f"tp={TUNER.g.world_size} capture={'rank0' if TUNER.rank0 else 'off'} "
                 f"targets={sorted(TUNER.loras)}")
        except Exception as exc:  # noqa: BLE001 - tuning is optional
            TUNER = None
            _log(f"install failed (serving unaffected): {type(exc).__name__}: {exc}")
        return result

    load_model._mtp_tune = True
    module.GPUModelRunner.load_model = load_model
