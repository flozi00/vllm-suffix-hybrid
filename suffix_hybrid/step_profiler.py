# SPDX-License-Identifier: Apache-2.0
"""In-pod engine-step profiler: SUFFIX_PROFILE_STEPS=<N>:<skip>[:<per>].

Default OFF. When set, the EngineCore process wraps
``EngineCoreProc._process_engine_step`` (vllm/v1/engine/core.py). Windows
are keyed by LOAD BUCKET (running requests: c1, c2-6, c7-12, c13-24, c25+),
so one boot under a bench chain yields one window per concurrency cell:
once the bucket has held for <skip> executing steps, the next <N> executing
steps run under torch.profiler (CPU + CUDA activities, record_shapes off),
at most <per> (default 1) windows per bucket, 12 per process. After each
window the profiler stops, every temporary wrap is removed, the chrome trace
goes to /tmp/suffix-prof-<pid>-<k>.json and ONE summary block is printed to
stderr, every line prefixed with ``[suffix-prof]`` (grep the pod log):

  * top-40 GPU kernels (calls, total ms, % of summed GPU time)
  * CPU step wall vs GPU busy (union over streams) -> host gap per step
  * sync / launch API counts and CPU time, with the host function that owns
    each sync (seq_lens.cpu() inside an attention builder shows up here)
  * CPU ms/step in scheduler, executor, attention-metadata builders
  * cudagraph mode per step (target/draft FULL vs PIECEWISE vs eager)
  * category rollup

CUDA graphs: a FULL replay is ONE cudaGraphLaunch on the CPU timeline; CUPTI
still reports the kernels inside the graph, so kernel totals stay complete
(the summary prints how many graph kernels were seen).

Fail-soft: any profiler error is logged and profiling is abandoned; the
wrapped engine function is always called exactly once with its result and
exceptions untouched. The worker must share the EngineCore process
(UniProcExecutor, TP=1) for GPU activity to be visible.

Offline: ``python -m suffix_hybrid.step_profiler <trace.json>`` prints the same
summary from a kept trace. stdlib + torch only.
"""
from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict

ENV = "SUFFIX_PROFILE_STEPS"
MARK = "[suffix-prof]"
STEP = "suffix_step"
TOP_K = 40
_TARGET_MODULE = "vllm.v1.engine.core"
_FINDER_MARK = "_suffix_step_profiler"

SYNC_APIS = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
             "cudaEventSynchronize", "cudaMemcpy", "cudaMemcpyAsync",
             "cuStreamSynchronize", "cuCtxSynchronize", "cuMemcpyDtoH")
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel",
               "cuLaunchKernelEx", "cudaGraphLaunch", "cuGraphLaunch")

# Ordered: first match wins. MoE before attention (flashinfer ships both).
CATEGORIES = [
    ("attn K2 own (verify)", r"nvfp4_attn_"),
    ("MoE", r"moe|expert|topk|router|grouped|align_block|expandinput|b12x"),
    ("attn FlashInfer fa2 (prefill/mm + q1 decode)",
     r"batchprefillwith|batchdecodewith|singleprefillwith|singledecodewith"
     r"|mergestate|variablelengthmerge|batchpodwith"),
    ("attn other (triton/...)",
     r"unified_attention|paged_attention|_fwd_kernel|attention|attn|fmha"),
    ("kv-cache write", r"reshape_and_cache|concat_and_cache|cache_kernel"
     r"|kv_cache"),
    ("dense GEMM", r"gemm|gemv|cutlass|cublas|nvjet|xmma|matmul|scaled_mm"
     r"|splitk|sm\d+_|mma"),
    ("quant (act/Q)", r"quant|fp4|fp8"),
    ("norm/rope/activation", r"norm|rms|rotary|rope|silu|gelu|act_and_mul"
     r"|softcap|tanh|activation"),
    ("sampling/spec", r"sampl|reject|argmax|softmax|top_p|gumbel|draft"
     r"|_prepare_|exponential|multinomial|logit|penalt|philox|random"),
    ("copies/memset", r"^memcpy|^memset|copy|catarray|index|gather|scatter"
     r"|fill"),
    ("other", r""),
]
_CAT_RE = [(n, re.compile(p, re.I)) for n, p in CATEGORIES]


MAX_WINDOWS = 12
BUCKETS = ((1, "c1"), (6, "c2-6"), (12, "c7-12"), (24, "c13-24"))


def parse_env(value: str | None) -> tuple[int, int, int] | None:
    """'<N>[:<skip>[:<per>]]' -> (N, skip, per); None when unset/invalid."""
    if not value or not value.strip():
        return None
    parts = value.strip().split(":")
    try:
        n = int(parts[0])
        skip = int(parts[1]) if len(parts) > 1 and parts[1] else 50
        per = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    except ValueError:
        return None
    if n <= 0 or skip < 0 or per <= 0 or len(parts) > 3:
        return None
    return n, skip, per


def bucket(running: int) -> str | None:
    if running <= 0:
        return None
    return next((lab for top, lab in BUCKETS if running <= top), "c25+")


def category(name: str) -> str:
    for cat, rx in _CAT_RE:
        if rx.search(name):
            return cat
    return "other"


def _short(name: str, width: int = 110) -> str:
    name = re.sub(r"\s+", " ", name.removeprefix("void ")).strip()
    return name if len(name) <= width else name[: width - 3] + "..."


def _union(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def summarize(trace: dict, py_stats: dict | None = None) -> list[str]:
    """Chrome-trace dict (torch export) -> summary lines (no marker)."""
    evs = [e for e in trace.get("traceEvents", [])
           if e.get("ph") == "X" and "ts" in e and "dur" in e]
    steps = sorted((e for e in evs if e.get("cat") == "user_annotation"
                    and e.get("name") == STEP), key=lambda e: e["ts"])
    if steps:
        w0, w1 = steps[0]["ts"], max(s["ts"] + s["dur"] for s in steps)
    else:
        w0 = min((e["ts"] for e in evs), default=0.0)
        w1 = max((e["ts"] + e["dur"] for e in evs), default=0.0)
    n = max(1, len(steps))
    span = max(w1 - w0, 1e-9)
    gpu = [e for e in evs if e.get("cat") in ("kernel", "gpu_memcpy",
                                                "gpu_memset")
           and w0 <= e["ts"] <= w1]
    busy = _union([(e["ts"], min(e["ts"] + e["dur"], w1)) for e in gpu])
    gpu_sum = sum(e["dur"] for e in gpu) or 1e-9
    step_cpu = sum(s["dur"] for s in steps)
    graph_kernels = sum(1 for e in gpu if e.get("cat") == "kernel" and any(
        "graph" in k.lower() for k in (e.get("args") or {})))

    out = [f"window: {len(steps)} steps, span {span / 1e3:.1f} ms, "
           f"{span / n / 1e3:.3f} ms/step"]
    if py_stats:
        reqs = py_stats.get("reqs", 0)
        out.append(
            f"load: {reqs:.1f} reqs/step, {py_stats.get('toks', 0):.1f} "
            f"scheduled tokens/step, {py_stats.get('emitted', 0):.1f} emitted "
            f"tokens/step ({py_stats.get('emitted', 0) / max(reqs, 1e-9):.2f}"
            " per req-step = accepted+1)")
    busy_in = sum(_union([(max(e["ts"], s["ts"]),
                           min(e["ts"] + e["dur"], s["ts"] + s["dur"]))
                          for e in gpu if e["ts"] < s["ts"] + s["dur"]
                          and e["ts"] + e["dur"] > s["ts"]])
                  for s in steps)
    out.append(
        f"time/step ms: cpu step {step_cpu / n / 1e3:.3f} | gpu busy "
        f"{busy / n / 1e3:.3f} ({100 * busy / span:.1f}% of span) | gpu idle "
        f"inside steps {(step_cpu - busy_in) / n / 1e3:.3f} | between steps "
        f"(engine loop) {(span - step_cpu) / n / 1e3:.3f} | summed kernel "
        f"{gpu_sum / n / 1e3:.3f}")

    # CUDA API: launches + syncs, sync owners by innermost annotation.
    rt = [e for e in evs if e.get("cat") in ("cuda_runtime", "cuda_driver")
          and w0 <= e["ts"] <= w1]
    anns = [e for e in evs if e.get("cat") == "user_annotation"
            and e.get("name") != STEP and w0 <= e["ts"] <= w1]
    api = defaultdict(lambda: [0, 0.0])
    owners = defaultdict(lambda: [0, 0.0])
    for e in rt:
        name = e.get("name", "")
        if name in SYNC_APIS or name in LAUNCH_APIS:
            api[name][0] += 1
            api[name][1] += e["dur"]
        if name in SYNC_APIS:
            enc = [a for a in anns if a.get("tid") == e.get("tid")
                   and a["ts"] <= e["ts"]
                   and a["ts"] + a["dur"] >= e["ts"] + e["dur"]]
            own = min(enc, key=lambda a: a["dur"])["name"] if enc else "-"
            owners[f"{own} / {name}"][0] += 1
            owners[f"{own} / {name}"][1] += e["dur"]
    out.append("cuda api /step (calls, cpu ms): " + (", ".join(
        f"{k} {v[0] / n:.1f} {v[1] / n / 1e3:.3f}"
        for k, v in sorted(api.items(), key=lambda kv: -kv[1][1])) or "none"))
    d2h = sum(1 for e in gpu if e.get("cat") == "gpu_memcpy"
              and "dtoh" in e.get("name", "").lower())
    out.append(f"D2H copies/step {d2h / n:.1f}; sync owners (calls/step, "
               "cpu ms/step): " + (", ".join(
                   f"{k} {v[0] / n:.1f} {v[1] / n / 1e3:.3f}"
                   for k, v in sorted(owners.items(),
                                      key=lambda kv: -kv[1][1])[:10])
                   or "none"))
    host = defaultdict(lambda: [0, 0.0])
    for a in anns:
        host[a["name"]][0] += 1
        host[a["name"]][1] += a["dur"]
    if host:
        out.append("host fn /step (calls, cpu ms): " + ", ".join(
            f"{k} {v[0] / n:.1f} {v[1] / n / 1e3:.3f}"
            for k, v in sorted(host.items(), key=lambda kv: -kv[1][1])))
    if py_stats and py_stats.get("cg"):
        out.append("cudagraph per step (signature: steps): " + " | ".join(
            f"{sig}: {c}" for sig, c in py_stats["cg"].most_common(8)))
    out.append(
        f"note: FULL graph replay = 1 cudaGraphLaunch on CPU; CUPTI kernels "
        f"inside graphs seen: {graph_kernels} of "
        f"{sum(1 for e in gpu if e.get('cat') == 'kernel')}")

    cats = defaultdict(float)
    per = defaultdict(lambda: [0, 0.0])
    for e in gpu:
        name = e.get("name", "?")
        cats[category(name)] += e["dur"]
        per[name][0] += 1
        per[name][1] += e["dur"]
    out.append("categories (ms/step, %): " + ", ".join(
        f"{c} {v / n / 1e3:.3f} {100 * v / gpu_sum:.1f}%"
        for c, v in sorted(cats.items(), key=lambda kv: -kv[1])))
    out.append(f"top {TOP_K} kernels: calls/step | ms/step | % | category | "
               "name")
    for name, (calls, dur) in sorted(per.items(),
                                     key=lambda kv: -kv[1][1])[:TOP_K]:
        out.append(f"  {calls / n:7.1f} | {dur / n / 1e3:7.3f} | "
                   f"{100 * dur / gpu_sum:5.1f} | {category(name)} | "
                   f"{_short(name)}")
    return out


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


class _Session:
    """Load-bucketed profiling windows. Every entry point is fail-soft."""

    def __init__(self, n: int, skip: int, per: int = 1):
        self.n, self.skip, self.per = n, skip, per
        self.phase = "wait"          # wait -> prof -> wait ... -> done
        self.cur, self.stable = None, 0
        self.done: Counter = Counter()
        self.windows = 0
        self.label = "-"
        self.prof = None
        self.restore: list[tuple[object, str, object]] = []
        self.cg_step: Counter = Counter()
        self.cg_depth = 0
        self._reset_window()

    def _reset_window(self) -> None:
        self.cg: Counter = Counter()
        self.reqs = self.toks = self.emitted = self.steps = 0
        self.t_start = 0.0

    def bucket_of(self, engine) -> str | None:
        return bucket(len(getattr(getattr(engine, "scheduler", None),
                                  "running", ()) or ()))

    def observe(self, b: str | None, executed: bool) -> None:
        """Track how long the load bucket has held (executing steps)."""
        if not executed or b is None:
            return
        if b == self.cur:
            self.stable += 1
        else:
            self.cur, self.stable = b, 1

    def ready(self, b: str | None) -> bool:
        return (b is not None and b == self.cur and self.stable >= self.skip
                and self.done[b] < self.per)

    # -- temporary wraps (installed at start, removed at stop) --
    def _wrap(self, owner, name: str, label: str, hook=None) -> None:
        orig = owner.__dict__.get(name)
        if not inspect.isfunction(orig):
            return
        from torch.autograd.profiler import record_function

        @functools.wraps(orig)
        def w(*a, **kw):
            rf = None
            try:
                rf = record_function(label)
                rf.__enter__()
            except Exception:  # noqa: BLE001
                rf = None
            try:
                r = orig(*a, **kw)
            finally:
                if rf is not None:
                    try:
                        rf.__exit__(None, None, None)
                    except Exception:  # noqa: BLE001
                        pass
            if hook is not None:
                try:
                    hook(a, r)
                except Exception:  # noqa: BLE001
                    pass
            return r

        setattr(owner, name, w)
        self.restore.append((owner, name, orig))

    def _wrap_cg(self, owner, name: str, mode: str) -> None:
        orig = owner.__dict__.get(name)
        if not inspect.isfunction(orig):
            return
        sess = self

        @functools.wraps(orig)
        def w(this, *a, **kw):
            if sess.cg_depth == 0:
                role = {"ModelCudaGraphManager": "target",
                        "SpeculatorCudaGraphManager": "draft"}.get(
                            type(this).__name__, type(this).__name__)
                sess.cg_step[f"{role}:{mode}"] += 1
            sess.cg_depth += 1
            try:
                return orig(this, *a, **kw)
            finally:
                sess.cg_depth -= 1

        setattr(owner, name, w)
        self.restore.append((owner, name, orig))

    def _instrument(self, engine) -> None:
        sched = getattr(engine, "scheduler", None)

        def on_schedule(_a, r):
            self.toks += int(getattr(r, "total_num_scheduled_tokens", 0))
            self.reqs += len(getattr(r, "num_scheduled_tokens", {}) or {})

        def on_update(a, _r):
            ids = getattr(a[2] if len(a) > 2 else None, "sampled_token_ids",
                          None)
            if ids:
                self.emitted += sum(len(x) for x in ids)

        if sched is not None:
            self._wrap(type(sched), "schedule", "sched.schedule", on_schedule)
            self._wrap(type(sched), "update_from_output",
                       "sched.update_from_output", on_update)
        ex = getattr(engine, "model_executor", None)
        if ex is not None:
            for m in ("execute_model", "sample_tokens",
                      "take_draft_token_ids"):
                for klass in type(ex).__mro__:
                    if m in klass.__dict__:
                        self._wrap(klass, m, f"exec.{m}")
                        break
        try:
            from vllm.v1.attention.backend import AttentionMetadataBuilder
            todo, seen = [AttentionMetadataBuilder], set()
            while todo:
                k = todo.pop()
                for sub in k.__subclasses__():
                    if sub not in seen:
                        seen.add(sub)
                        todo.append(sub)
                        self._wrap(sub, "build", f"attn.build:{sub.__name__}")
        except Exception as exc:  # noqa: BLE001
            _say(f"builder instrumentation skipped: {exc!r}")
        try:
            from vllm.v1.worker.gpu import cudagraph_utils as cgu
            for k in (getattr(cgu, "CudaGraphManager", None),
                      getattr(cgu, "ModelCudaGraphManager", None)):
                if k is not None:
                    self._wrap_cg(k, "run_fullgraph", "FULL")
                    self._wrap_cg(k, "run_pw_graph", "PIECEWISE")
            self.cg_tracked = True
        except Exception:  # noqa: BLE001 - V1 runner: no per-step modes
            self.cg_tracked = False

    def _uninstrument(self) -> None:
        for owner, name, orig in reversed(self.restore):
            try:
                setattr(owner, name, orig)
            except Exception:  # noqa: BLE001
                pass
        self.restore.clear()

    # -- lifecycle --
    def start(self, engine, label: str = "-") -> None:
        import torch
        from torch.profiler import ProfilerActivity, profile
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA in this process")
        self._reset_window()
        self.label = label
        self._instrument(engine)
        self.prof = profile(activities=[ProfilerActivity.CPU,
                                        ProfilerActivity.CUDA],
                            record_shapes=False, with_stack=False,
                            profile_memory=False)
        self.prof.__enter__()
        self.phase = "prof"
        self.t_start = time.perf_counter()
        _say(f"START window {self.windows + 1} [{label}]: {self.n} steps "
             f"after {self.stable} steps in bucket (pid {os.getpid()})")

    def end_step(self, executed: bool) -> None:
        if getattr(self, "cg_tracked", False) and executed:
            sig = " ".join(f"{k}x{v}" for k, v in sorted(self.cg_step.items()))
            if not any(k.startswith("target:") for k in self.cg_step):
                sig = ("target:eager " + sig).strip()
            self.cg[sig] += 1
        self.cg_step.clear()

    def stop(self) -> None:
        import torch
        self.windows += 1
        self.done[self.label] += 1
        self.stable = 0
        self.phase = "wait" if self.windows < MAX_WINDOWS else "done"
        wall = time.perf_counter() - self.t_start
        try:
            torch.cuda.synchronize()
        finally:
            prof, self.prof = self.prof, None
            try:
                prof.__exit__(None, None, None)
            finally:
                self._uninstrument()
        path = f"/tmp/suffix-prof-{os.getpid()}-{self.windows}.json"
        prof.export_chrome_trace(path)
        py = {"reqs": self.reqs / self.n, "toks": self.toks / self.n,
              "emitted": self.emitted / self.n, "cg": self.cg}
        head = (f"BEGIN summary window {self.windows} [{self.label}] "
                f"({self.n} steps, trace {path})")
        _say(f"STOP after {wall * 1e3:.0f} ms wall; trace {path}; "
             "summarizing in a background thread")

        def work():
            try:
                with open(path) as fh:
                    lines = summarize(json.load(fh), py)
                block = "\n".join(f"{MARK} {ln}" for ln in
                                  [head] + lines + [f"END summary window "
                                                    f"{self.windows}"])
                print(block, file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                _say(f"summary FAILED (trace kept at {path}): {exc!r}")

        threading.Thread(target=work, name="suffix-prof-summary",
                         daemon=True).start()

    def abandon(self, where: str, exc: BaseException) -> None:
        self.phase = "done"
        _say(f"profiler error in {where}, profiling abandoned (serving "
             f"unaffected): {exc!r}")
        prof, self.prof = self.prof, None
        if prof is not None:
            try:
                prof.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._uninstrument()


def wrap_step(orig, sess: _Session):
    """Wrap EngineCoreProc._process_engine_step. Never alters its result."""

    @functools.wraps(orig)
    def step(self, *a, **kw):
        if sess.phase == "done":
            return orig(self, *a, **kw)
        if sess.phase == "wait":
            try:
                b = sess.bucket_of(self)
                go = sess.ready(b)
            except Exception:  # noqa: BLE001
                b, go = None, False
            if go:
                try:
                    sess.start(self, b)
                except Exception as exc:  # noqa: BLE001
                    sess.abandon("start", exc)
            if sess.phase != "prof":
                r = orig(self, *a, **kw)
                try:
                    sess.observe(b, bool(r))
                except Exception:  # noqa: BLE001
                    pass
                return r
        rf = None
        try:
            from torch.autograd.profiler import record_function
            rf = record_function(STEP)
            rf.__enter__()
        except Exception:  # noqa: BLE001
            rf = None
        try:
            r = orig(self, *a, **kw)
        finally:
            if rf is not None:
                try:
                    rf.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
        try:
            sess.end_step(bool(r))
            sess.steps += bool(r)
            if sess.steps >= sess.n:
                sess.stop()
        except Exception as exc:  # noqa: BLE001
            sess.abandon("step", exc)
        return r

    return step


def _patch(module) -> None:
    cfg = parse_env(os.environ.get(ENV))
    cls = getattr(module, "EngineCoreProc", None)
    if cfg is None or cls is None or getattr(cls, _FINDER_MARK, False):
        return
    cls._process_engine_step = wrap_step(cls._process_engine_step,
                                         _Session(*cfg))
    cls._suffix_step_profiler = True
    _say(f"armed on EngineCoreProc._process_engine_step: {cfg[0]} steps "
         f"per window after {cfg[1]} steady steps, {cfg[2]} window(s) per "
         f"load bucket (c1, c2-6, c7-12, c13-24, c25+)")


def install_post_import_hook() -> None:
    """Patch vllm.v1.engine.core right after it executes. Never raises."""
    value = os.environ.get(ENV)
    if parse_env(value) is None:
        if value:
            _say(f"{ENV}={value!r} invalid (want <N>:<skip>[:<per>]); "
                 "profiler off")
        return
    if _TARGET_MODULE in sys.modules:
        _patch(sys.modules[_TARGET_MODULE])
        return
    if any(getattr(f, _FINDER_MARK, False) for f in sys.meta_path):
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != _TARGET_MODULE:
                return None
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                try:
                    _patch(module)
                except Exception as exc:  # noqa: BLE001
                    _say(f"patch failed, profiler off: {exc!r}")

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _FINDER_MARK, True)
    sys.meta_path.insert(0, finder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m suffix_hybrid.step_profiler <trace.json>")
    with open(sys.argv[1]) as fh:
        for line in summarize(json.load(fh)):
            print(line)
