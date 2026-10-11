# SPDX-License-Identifier: Apache-2.0
"""L2 weight prefetch overlapped with the host all-reduce (SUFFIX_HAR_L2_PREFETCH=1).

GLM 5.3 decode at c1 runs ~170 host all-reduces per step (~20 us each, PCIe-latency
bound) and after each one reads its next dense weights cold from GDDR (qkv_a / q_b /
indexer wq_b / o_proj after the input all-reduce, router + shared expert after the
post-attention one). While the all-reduce waits on PCIe the GDDR idles, so the HAR
kernel's waiting blocks issue `cp.async.bulk.prefetch.L2` for the weights the next
window will read (host_ar._l2_prefetch; the block that publishes the barrier never does:
it is on the critical path). Prefetch never writes: outputs stay bit-identical.

Schedule: after GPUModelRunner.load_model (V2 runner) the target model and the MTP draft
model get a forward pre/post hook (one record per root forward with <= MAX_TOKENS tokens:
decode shapes, never prefill chunks) and every Linear-like module (vLLM LinearBase,
nn.Linear) a pre-hook that appends its weights to the current window; MLA modules add
W_UK_T after q_b_proj and W_UV before o_proj. Every HAR all_reduce inside a record opens
the next window. A (root, tokens) key's first complete record becomes its schedule and
its tables are built at the root's post-hook (never during CUDA-graph capture); vLLM
warms every captured shape up eagerly first, so the capture already prefetches. A later
record of the key that differs (window count or contents) marks the key unstable: no
more prefetch for it, its rows are zeroed (deferred out of capture) -- fail closed.

Rows: int64 [addr, bytes] per chunk (CHUNK bytes, 16 B aligned inside the tensor's own
storage, tensors kept referenced by their table), first BUDGET MB of each window in use
order, padded with zero rows to a multiple of 16 (one Triton specialization). Boot:
attach() proves on every rank that the all-reduce with prefetch (two-shot, one-shot and
int8 transport when enabled) is bit-identical to without, else the worker refuses to start.

    SUFFIX_HAR_L2_PREFETCH=1          gate (needs the "har" nccl_split spec and the V2 runner)
    SUFFIX_HAR_L2_PREFETCH_MB=32      bytes prefetched per window
    python -m suffix_hybrid.har_prefetch bench   # boot gate har_l2_prefetch_bench
"""
from __future__ import annotations

import importlib.util
import os
import sys

MARK = "[suffix har-l2pf]"
GATE_ENV = "SUFFIX_HAR_L2_PREFETCH"
BUDGET_ENV = "SUFFIX_HAR_L2_PREFETCH_MB"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
CHUNK = 128 << 10      # bytes per prefetch row
MIN_BYTES = 4096       # smaller tensors are not worth a row
ALIGN = 16             # cp.async.bulk: 16 B aligned address and size
ROW_PAD = 16           # rows padded to a multiple of this (stable Triton specialization)
MAX_TOKENS = 512       # records only for forwards up to this many tokens (decode shapes)


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def budget_bytes() -> int:
    return int(float(os.environ.get(BUDGET_ENV, "32").strip() or "32") * (1 << 20))


# ------------------------------------------------------------------ rows (pure)
def tensor_span(t) -> tuple[int, int] | None:
    """(addr, nbytes) of the memory a strided tensor covers, checked to lie inside its
    own storage (= a live allocation while the tensor is referenced); None if empty or not."""
    if t.numel() == 0 or any(st < 0 for st in t.stride()):
        return None
    start = t.data_ptr()
    span = (sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1) * t.element_size()
    st = t.untyped_storage()
    base, size = st.data_ptr(), st.nbytes()
    if not (base <= start and start + span <= base + size):
        return None
    return start, span


def rows_for(spans, budget: int, chunk: int = CHUNK) -> list[tuple[int, int]]:
    """[(addr, nbytes)] in use order -> [(addr, size)] rows: each span shrunk to 16 B
    boundaries (never past its ends), split into <= chunk pieces, cut at budget bytes."""
    out, left = [], budget
    for addr, nbytes in spans:
        a = -(-addr // ALIGN) * ALIGN
        b = (addr + nbytes) // ALIGN * ALIGN
        while b - a >= MIN_BYTES and left >= ALIGN:
            size = min(chunk, b - a, left // ALIGN * ALIGN)
            out.append((a, size))
            a += size
            left -= size
        if left < ALIGN:
            break
    return out


def make_table(rows, device):
    """int64 [cap, 2] rows on device, cap padded to ROW_PAD with zero (skipped) rows."""
    import torch

    cap = max(ROW_PAD, -(-len(rows) // ROW_PAD) * ROW_PAD)
    flat = [v for r in rows for v in r] + [0] * (2 * (cap - len(rows)))
    return torch.tensor(flat, dtype=torch.int64, device=device).view(cap, 2), cap


def watched(m) -> bool:
    import torch

    if isinstance(m, torch.nn.Linear):
        return True
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError:  # CPU tests without vLLM
        return False
    return isinstance(m, LinearBase)


def own_tensors(m, device) -> list:
    """The module's own (non-recursive) parameters and buffers on device, in order."""
    out = []
    for t in list(m._parameters.values()) + list(m._buffers.values()):
        if t is not None and t.device == device and t.numel() * t.element_size() >= MIN_BYTES:
            out.append(t)
    return out


# ------------------------------------------------------------------ recorder
class Recorder:
    """Per-process schedule/tables (see the module docstring)."""

    def __init__(self, budget: int):
        self.budget = budget
        self.device = None
        self.hars = 0
        self.roots: dict[int, str] = {}
        self.seen: set[int] = set()
        self.schedules: dict[tuple, list] = {}   # key -> [window signature after AR k]
        self.unstable: set[tuple] = set()
        self.tables: dict[tuple, tuple] = {}     # signature -> (rows, cap, tensors)
        self.keep: dict[tuple, list] = {}        # signature -> tensors keeping its memory alive
        self.pending: list = []                  # signatures to build (seen during capture)
        self.zero: list = []                     # rows to zero (key went unstable in capture)
        self.cur = None                          # {"key", "plan", "wins": [[tensors]]}
        self.off = None                          # reason, once disabled entirely

    # -- hooks
    def attach_root(self, name: str, root) -> int:
        import torch

        self.device = self.device or torch.device("cuda", torch.cuda.current_device())
        if id(root) in self.roots:  # load_model again (weights reload): hooks already there
            return 0
        self.roots[id(root)] = name
        root.register_forward_pre_hook(self._root_pre, with_kwargs=True)
        root.register_forward_hook(self._root_post, with_kwargs=True)
        n = 0
        for m in root.modules():  # parents before children; modules shared by roots once
            if id(m) in self.seen:
                continue
            self.seen.add(id(m))
            for sub, when, t in _mla_extras(m):  # W_UK_T after q_b_proj, W_UV before o_proj
                sub.__dict__.setdefault("_suffix_l2pf_extra", {"before": [], "after": []})[when].append(t)
            if watched(m):
                m.register_forward_pre_hook(self._collect)
                n += 1
        return n

    def _root_pre(self, module, args, kwargs):
        try:
            self._flush_deferred()
            pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
            toks = int(pos.shape[-1]) if pos is not None else None
            if self.off or toks is None or toks > MAX_TOKENS:
                self.cur = None
                return
            import torch

            key = (id(module), toks)
            if key in self.unstable:
                self.cur = None
                return
            plan = self.schedules.get(key)
            # record the first forward of a key and every captured one (checked against the
            # schedule); eager forwards of a known key only replay it (no Python per module)
            rec = plan is None or torch.cuda.is_current_stream_capturing()
            self.cur = {"key": key, "plan": plan, "wins": [[]], "rec": rec}
        except Exception as exc:  # noqa: BLE001 - never break a forward
            self.disable(f"root pre-hook: {exc!r}")

    def _collect(self, m, args):
        cur = self.cur
        if cur is None or not cur["rec"]:
            return
        ex = m.__dict__.get("_suffix_l2pf_extra")
        win = cur["wins"][-1]
        win.extend(ex["before"] if ex else ())
        win.extend(own_tensors(m, self.device))
        win.extend(ex["after"] if ex else ())

    def on_ar(self):
        """Per HAR all_reduce: open the next window; rows for it from the key's plan."""
        cur = self.cur
        if cur is None:
            return None
        k = len(cur["wins"]) - 1
        cur["wins"].append([])
        plan = cur["plan"]
        if self.off or plan is None or k >= len(plan):
            return None
        tab = self.tables.get(plan[k])
        return None if tab is None else (tab[0], tab[1])

    def _root_post(self, module, args, kwargs, out):
        cur, self.cur = self.cur, None
        if cur is None or self.off or not cur["rec"]:
            return
        try:
            import torch

            got = [self._sig(w) for w in cur["wins"][1:]]  # windows after AR 0..n-1
            key = cur["key"]
            capturing = torch.cuda.is_current_stream_capturing()
            have = self.schedules.get(key)
            if have is None:
                self.schedules[key] = got
                self.pending.extend(s for s in got if s not in self.tables)
            elif have != got:
                self.unstable.add(key)
                live = {s for k, sch in self.schedules.items() if k not in self.unstable for s in sch}
                self.zero.extend(self.tables[s][0] for s in have if s in self.tables and s not in live)
                _say(f"schedule of {self.roots.get(key[0], '?')} at {key[1]} tokens changed "
                     f"({len(have)} -> {len(got)} windows): prefetch off for it (fail closed)")
            if not capturing:
                self._flush_deferred()
        except Exception as exc:  # noqa: BLE001
            self.disable(f"root post-hook: {exc!r}")

    # -- tables
    def _sig(self, tensors) -> tuple:
        """Window signature: the distinct (addr, nbytes) spans of its tensors on this GPU."""
        spans = []
        for t in tensors:
            sp = tensor_span(t) if t.device == self.device else None
            if sp is not None and sp not in spans:
                spans.append(sp)
        sig = tuple(spans)
        self.keep.setdefault(sig, list(tensors))
        return sig

    def _flush_deferred(self) -> None:
        import torch

        if torch.cuda.is_current_stream_capturing():
            return
        for rows in self.zero:
            rows[:, 1].zero_()
        self.zero = []
        for sig in self.pending:
            rows = rows_for(sig, self.budget) if sig not in self.tables else None
            if rows:  # an empty window keeps no table: its all-reduce runs without prefetch
                self.tables[sig] = (*make_table(rows, self.device), self.keep.get(sig))
        self.pending = []

    def disable(self, reason: str) -> None:
        if self.off:
            return
        self.off = reason
        self.cur = None
        self.zero.extend(t[0] for t in self.tables.values())
        _say(f"DISABLED (fail closed, serving unaffected): {reason}")


RECORDER: Recorder | None = None


def _mla_extras(m) -> list:
    """MLA attention: absorbed W_UK_T is read right after q_b_proj, W_UV just before o_proj."""
    import torch

    out = []
    for attr, sub, when in (("W_UK_T", "q_b_proj", "after"), ("W_UV", "o_proj", "before")):
        t, s = getattr(m, attr, None), getattr(m, sub, None)
        if isinstance(t, torch.Tensor) and isinstance(s, torch.nn.Module):
            out.append((s, when, t))
    return out


# ------------------------------------------------------------------ install
def selftest(har, mb: int = 16) -> str:
    """Collective (every rank of har's group): all_reduce with prefetch == without, bit for
    bit, for two-shot, one-shot and (if enabled) int8 transport. Returns a summary; raises."""
    import torch
    import torch.distributed as dist

    dev = har.device
    scratch = torch.empty(mb << 20, dtype=torch.uint8, device=dev)
    rows, cap = make_table(rows_for([tensor_span(scratch)], mb << 20), dev)
    g = torch.Generator(device=dev).manual_seed(4242 + har.rank)
    cases = [("two-shot", 6 * 6144, {"oneshot": False, "q8": False}),
             ("one-shot", 6144, {"oneshot": True, "q8": False})]
    if har.q8_min < (1 << 61):
        cases.append(("int8", 48 * 6144, {"oneshot": False, "q8": True}))
    bad = []
    for name, n, kw in cases:
        x = torch.randn(n, dtype=torch.float32, device=dev, generator=g).to(torch.bfloat16)
        ref = har.all_reduce(x, **kw)  # attach() sets har.l2pf only after this
        on = har.all_reduce(x, pf=(rows, cap), **kw)
        torch.cuda.synchronize(dev)
        if not torch.equal(ref, on):
            bad.append(name)
    flag = torch.tensor([float(len(bad))], dtype=torch.float32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=har.cpu_group)
    del scratch, rows
    if bad or flag.item():
        raise RuntimeError(f"all-reduce with L2 prefetch not bit-identical ({bad or 'other rank'})")
    return "/".join(c[0] for c in cases)


def attach(har) -> None:
    """nccl_split.make_comm, gate on: prove the prefetch variants exact, then let every
    all_reduce of har ask the recorder for its window's rows."""
    global RECORDER
    if RECORDER is None:
        RECORDER = Recorder(budget_bytes())
    checked = selftest(har)
    har.l2pf = RECORDER.on_ar
    RECORDER.hars += 1
    if har.rank == 0:
        _say(f"armed: {checked} bit-identical with prefetch on every rank; budget "
             f"{RECORDER.budget >> 20} MiB/window, {CHUNK >> 10} KiB rows; waiting for "
             "GPUModelRunner.load_model")


def _patch_runner(module) -> None:
    import functools

    cls = module.GPUModelRunner
    orig = cls.load_model
    if getattr(orig, "_suffix_l2pf", False):
        return

    @functools.wraps(orig)
    def load_model(self, *a, **kw):
        out = orig(self, *a, **kw)
        if RECORDER is None or RECORDER.hars == 0:
            raise SystemExit(f"{MARK} {GATE_ENV}=1 but no host all-reduce in this worker "
                             "(needs SUFFIX_NCCL_BANDS with the har spec)")
        roots = [("target", getattr(self, "model", None)),
                 ("draft", getattr(getattr(self, "speculator", None), "model", None))]
        roots = [(n, r) for n, r in roots if r is not None]
        if not roots:
            raise SystemExit(f"{MARK} {GATE_ENV}=1 but GPUModelRunner has no model after load")
        hooked = {n: RECORDER.attach_root(n, r) for n, r in roots}
        if RECORDER.device is not None and RECORDER.device.index == 0:
            _say("ACTIVE: " + ", ".join(f"{n} {type(r).__name__} ({hooked[n]} modules)"
                                       for n, r in roots)
                 + f"; records forwards <= {MAX_TOKENS} tokens")
        return out

    load_model._suffix_l2pf = True
    cls.load_model = load_model


def install_post_import_hook() -> None:
    """sitecustomize entry (gate on): patch the V2 GPUModelRunner right after it imports."""
    if not enabled():
        return

    def run(module):
        try:
            _patch_runner(module)
        except Exception as exc:
            raise SystemExit(f"{MARK} enabled but installation FAILED: {exc!r}") from exc

    if RUNNER_MODULE in sys.modules:
        run(sys.modules[RUNNER_MODULE])
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != RUNNER_MODULE:
                return None
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                run(module)

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _Finder())


# ------------------------------------------------------------------ boot gate bench
# Per TP8 rank (glm_dense_marlin.SHAPES): attention window after the input all-reduce
# (FP8 W8A8 cutlass, our fp8_dense path) and MoE window after the post-attention one (BF16).
ATTN = (("qkv_a", 2624, 6144), ("q_b", 2048, 2048), ("idx.wq_b", 4096, 2048), ("o_proj", 6144, 2048))
MOE = (("router", 256, 6144), ("shared.gate_up", 512, 6144), ("shared.down", 6144, 256))
MS = (6, 48)
ITERS = 30


def _timed(prep, fn, iters: int = ITERS) -> float:
    """Median us of a CUDA graph of fn() replayed after prep() each time (prep untimed)."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        prep()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    del g
    return sorted(ts)[len(ts) // 2]


def _bench_worker(rank, world, port, q, hidden) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from suffix_hybrid import host_ar as H
    from suffix_hybrid import nccl_split as ns

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=dev)
    cpu = dist.new_group(backend="gloo")
    out = {"rank": rank, "rows": [], "err": None}
    try:
        import vllm._custom_ops as ops
        from fp8_dense_patch import runtime as R

        R._OPS = ops
        har = H.HostAllReduce(cpu, dev, oneshot_max_bytes=48 << 10, timeout_spins=1 << 26)
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        flush_buf = torch.empty(max(2 * l2, 64 << 20), dtype=torch.uint8, device=dev)
        g = torch.Generator(device=dev).manual_seed(77 + rank)

        def rnd(*shape):
            return torch.randn(*shape, dtype=torch.float32, device=dev, generator=g)

        def tables(tensors, mb):
            return make_table(rows_for([tensor_span(t) for t in tensors], mb << 20), dev)

        attn = {nm: R.quantize_weight((rnd(n, k) * 0.02).to(torch.bfloat16)) for nm, n, k in ATTN}
        moe = {nm: (rnd(n, k) * 0.02).to(torch.bfloat16) for nm, n, k in MOE}
        attn_t = [t for nm, _, _ in ATTN for t in attn[nm]]
        moe_t = list(moe.values())
        pf = {f"attn{mb}": tables(attn_t, mb) for mb in (16, 32, 64)}
        pf.update({f"moe{mb}": tables(moe_t, mb) for mb in (16, 32)})
        pf_kernel = H._pf_kernel  # standalone prefetch (GEMM rows)

        def flush():
            flush_buf.zero_()

        def align():  # flush, then line the ranks up before the timed replay
            flush_buf.zero_()
            dist.barrier(group=cpu)

        for m in MS:
            x = rnd(m, hidden).to(torch.bfloat16)
            a_out = rnd(m, 2048).to(torch.bfloat16)
            nw = torch.ones(hidden, dtype=torch.bfloat16, device=dev)

            def norm(y):
                v = y.float()
                return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * nw

            def attn_chain(p):
                h = norm(har.all_reduce(x, pf=p))
                qa = R.fp8_linear(h, *attn["qkv_a"])
                qc = qa[:, :2048].contiguous()
                R.fp8_linear(qc, *attn["q_b"])
                R.fp8_linear(qc, *attn["idx.wq_b"])
                return R.fp8_linear(a_out, *attn["o_proj"])

            def moe_chain(p):
                h = norm(har.all_reduce(x, pf=p))
                F.linear(h, moe["router"])
                gu = F.linear(h, moe["shared.gate_up"])
                return F.linear(F.silu(gu[:, :256]) * gu[:, 256:], moe["shared.down"])

            # (i) GEMMs alone: cold vs prefetched (standalone kernel + 40 us sleep) vs hot
            for nm, n, k in ATTN:
                xi = rnd(m, k).to(torch.bfloat16)
                wq, ws = attn[nm]
                rows, cap = tables([wq, ws], 64)

                def pre_pf(rows=rows, cap=cap):
                    flush_buf.zero_()
                    pf_kernel[(-(-cap // 128),)](rows, cap, num_warps=4)
                    torch.cuda._sleep(80_000)
                fn = (lambda xi=xi, wq=wq, ws=ws: R.fp8_linear(xi, wq, ws))
                out["rows"].append({"t": "gemm", "name": nm, "m": m, "cold": _timed(flush, fn),
                                    "pf": _timed(pre_pf, fn), "hot": _timed(lambda: None, fn)})
            # (ii) all-reduce alone: rows rotating over 4 x 32 MiB so every prefetch streams
            regions = [torch.empty(32 << 20, dtype=torch.uint8, device=dev) for _ in range(4)]
            rot = [make_table(rows_for([tensor_span(r)], mb << 20), dev)
                   for mb in (16, 32) for r in regions]
            for label, sel in (("off", None), ("pf16", rot[:4]), ("pf32", rot[4:])):
                calls = [(lambda i=i: har.all_reduce(x, pf=None if sel is None else sel[i % 4]))
                         for i in range(20)]
                us = ns.graph_time_us(lambda: [c() for c in calls], reps=1, replays=7) / 20
                out["rows"].append({"t": "ar", "m": m, "v": label, "us": us})
            del regions, rot
            # (iii) all-reduce -> norm -> window GEMMs, cold L2 before every replay
            ys = {}
            for label, p in (("off", None), ("pf16", pf["attn16"]), ("pf32", pf["attn32"]),
                             ("pf64", pf["attn64"])):
                out["rows"].append({"t": "attn", "m": m, "v": label,
                                    "us": _timed(align, lambda p=p: attn_chain(p))})
                ys[label] = har.all_reduce(x, pf=p)
            for label, p in (("off", None), ("pf16", pf["moe16"]), ("pf32", pf["moe32"])):
                out["rows"].append({"t": "moe", "m": m, "v": label,
                                    "us": _timed(align, lambda p=p: moe_chain(p))})
            out["rows"].append({"t": "exact", "m": m,
                                "ok": all(torch.equal(ys["off"], y) for y in ys.values())})
            out["rows"].append({"t": "warm", "m": m, "us": _timed(lambda: dist.barrier(group=cpu),
                                                                lambda: attn_chain(None))})
        har.destroy()
    except Exception:  # noqa: BLE001
        import traceback

        out["err"] = traceback.format_exc()[-1500:]
    q.put(out)
    try:
        dist.barrier(group=cpu)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def bench(hidden: int = 6144) -> int:
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world not in (2, 4, 8):
        print(f"{MARK} SKIP: {world} GPU(s)", flush=True)
        return 0
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29700 + os.getpid() % 1000
    procs = [ctx.Process(target=_bench_worker, args=(r, world, port, q, hidden)) for r in range(world)]
    [p.start() for p in procs]
    res = []
    try:
        for _ in procs:
            res.append(q.get(timeout=1800))
    except Exception as exc:  # noqa: BLE001
        print(f"{MARK} FAIL: {len(res)}/{world} ranks reported ({exc!r})", flush=True)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    errs = [r for r in res if r["err"]]
    for r in errs[:2]:
        print(f"{MARK} rank {r['rank']} ERROR: {r['err']}", flush=True)
    if errs or len(res) < world:
        return 1
    res.sort(key=lambda r: r["rank"])
    rows = res[0]["rows"]
    exact = all(row["ok"] for r in res for row in r["rows"] if row["t"] == "exact")

    def worst(i, k):  # max over ranks (the step waits for the slowest)
        return max(r["rows"][i][k] for r in res)

    print(f"{MARK} cold L2 = 2x L2 written before every replay; median of {ITERS}; us; "
          "max over ranks for chains", flush=True)
    chain = {}
    for i, row in enumerate(rows):
        if row["t"] == "gemm":
            print(f"{MARK} gemm {row['name']:9s} M={row['m']:3d}: cold {row['cold']:7.1f} | "
                  f"prefetched {row['pf']:7.1f} | hot {row['hot']:7.1f}", flush=True)
        elif row["t"] in ("ar", "attn", "moe"):
            chain[(row["t"], row["m"], row["v"])] = worst(i, "us")
        elif row["t"] == "warm":
            chain[("warm", row["m"], "-")] = worst(i, "us")
    for m in MS:
        for t, vs in (("ar", ("off", "pf16", "pf32")), ("attn", ("off", "pf16", "pf32", "pf64")),
                      ("moe", ("off", "pf16", "pf32"))):
            base = chain[(t, m, "off")]
            print(f"{MARK} {t:4s} M={m:3d}: " + " | ".join(
                f"{v} {chain[(t, m, v)]:7.1f}" + (f" ({chain[(t, m, v)] - base:+.1f})" if v != "off" else "")
                for v in vs) + (f" | hot (no flush) {chain[('warm', m, '-')]:7.1f}" if t == "attn" else ""),
                flush=True)
    m = MS[0]
    gain = (chain[("attn", m, "off")] - chain[("attn", m, "pf32")]
            + chain[("moe", m, "off")] - chain[("moe", m, "pf32")])
    print(f"{MARK} VERDICT M={m}: attention + MoE window {gain:+.1f} us per layer with 32 MiB "
          f"(x78 layers ~ {gain * 78 / 1000:+.2f} ms/step, estimate); prefetch "
          f"{'bit-exact' if exact else 'NOT bit-exact'}", flush=True)
    print(f"{MARK} {'PASS' if exact else 'FAIL'}: all-reduce outputs identical with and without "
          "prefetch on every rank", flush=True)
    return 0 if exact else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench"])
    ap.add_argument("--hidden", type=int, default=6144)
    a = ap.parse_args()
    return bench(a.hidden)


if __name__ == "__main__":
    sys.exit(main())
