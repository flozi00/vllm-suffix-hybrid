# SPDX-License-Identifier: Apache-2.0
"""Size-routed NCCL all-reduce: per-size-band communicators, startup autotune, QAR.

On PCIe nodes without GPU P2P (worker-06: every fast all-reduce in vLLM 0.30.0
is disabled, NCCL goes through host memory) no single NCCL algorithm/protocol
wins at all sizes: default is 2-2.7x slower than tree at 384 KiB - 1.5 MiB (and
1.5 MiB is 3x slower than 3 MiB), a global NCCL_ALGO=allreduce:tree made
60k-token prefills 27 % slower -- and NCCL_ALGO=Tree globally kills all-gather
("invalid usage"). NCCL has no size threshold knob, but it reads NCCL_ALGO /
NCCL_PROTO at communicator init, so each band gets its OWN PyNcclCommunicator
built while those variables are set, and CudaCommunicator.all_reduce picks the
communicator by message bytes (static per captured shape -> CUDA-graph safe;
all communicators exist before graph capture). Specs: "default" (the stock
communicator) or "<NCCL_ALGO>[/<NCCL_PROTO>]", e.g. "allreduce:ring/Simple".

  SUFFIX_NCCL_BANDS="0-256K:default,256K-4M:allreduce:ring/Simple,4M-inf:default"
      half-open [lo, hi) byte bands, K/M/G = KiB/MiB/GiB; gaps = default.
  SUFFIX_NCCL_SMALL_ALGO / _PROTO / _MIN_BYTES / _BYTES (legacy, one band
      [MIN, BYTES] inclusive; defaults 256 KiB, 2 MiB). Not with _BANDS.
  SUFFIX_NCCL_AUTOTUNE=1  (tp group only) every rank times CANDIDATES x sizes
      (16 KiB..SUFFIX_NCCL_AUTOTUNE_MAX_BYTES log2-spaced + tokens x hidden x 2)
      on the real group inside CudaCommunicator.__init__ (before model load and
      graph capture), results MAX-reduced over ranks (gloo) so every rank
      derives the same bands; a candidate that errors or sums wrong anywhere is
      never chosen. Env: _BUDGET_S (20), _ITERS (7; 3 above 8 MiB), _MARGIN
      (0.05: a candidate must beat default by 5 %), _CANDIDATES (comma list),
      _HIDDEN (else model config). Overrides _BANDS on the tp group. One
      "[suffix nccl-split] AUTOTUNE" line (rank 0) with table + bands.
  SUFFIX_NCCL_QAR=int8|fp8 (tp group only, default off): bf16/fp16 all-reduces
      in [SUFFIX_NCCL_QAR_MIN_BYTES (4 MiB), SUFFIX_NCCL_QAR_MAX_BYTES (inf))
      go through suffix_hybrid/nccl_qar.py (LOSSY -- quality-gate first).
Fail-closed (SystemExit) on anchor drift or bad config; all unset = inert.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

ENV = "SUFFIX_NCCL_SMALL_ALGO"
BYTES_ENV = "SUFFIX_NCCL_SMALL_BYTES"
MIN_ENV = "SUFFIX_NCCL_SMALL_MIN_BYTES"
PROTO_ENV = "SUFFIX_NCCL_SMALL_PROTO"
BANDS_ENV = "SUFFIX_NCCL_BANDS"
AUTOTUNE_ENV = "SUFFIX_NCCL_AUTOTUNE"
QAR_ENV = "SUFFIX_NCCL_QAR"
TARGET = "vllm.distributed.device_communicators.cuda_communicator"
TAG = "suffix nccl-split"
CANDIDATES = ("default", "allreduce:ring/Simple", "allreduce:ring/LL128", "allreduce:ring/LL",
              "allreduce:tree/Simple", "allreduce:tree/LL128", "allreduce:tree/LL")
TOKENS = (1, 6, 12, 24, 48, 96, 192, 384, 1024, 2048, 4096, 8192)
INF = 1 << 62
OLD = ("        assert pynccl_comm is not None\n"
       "        out = pynccl_comm.all_reduce(input_)\n")
NEW = ("        assert pynccl_comm is not None\n"
       f"        _pick = getattr(self, '_suffix_pick', None)  # {TAG}\n"
       "        if _pick is not None:\n"
       "            pynccl_comm = _pick(input_, pynccl_comm)\n"
       "        out = pynccl_comm.all_reduce(input_)\n")


class PatchDriftError(RuntimeError):
    pass


def _say(msg: str) -> None:
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ----------------------------------------------------------------- band specs
def parse_size(s: str) -> int:
    s = s.strip().upper().removesuffix("B").removesuffix("I")
    if s == "INF":
        return INF
    unit = s[-1:] if s[-1:] in ("K", "M", "G") else ""
    return int(float(s[:len(s) - len(unit)]) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[unit])


def fmt_size(n: int) -> str:
    if n >= INF:
        return "inf"
    for unit, u in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n and n % u == 0:
            return f"{n // u}{unit}"
    return str(n)


def spec_env(spec: str) -> dict[str, str]:
    if spec == "default":
        return {}
    algo, _, proto = spec.partition("/")
    if not algo.strip():
        raise ValueError(f"bad communicator spec {spec!r}")
    return {"NCCL_ALGO": algo.strip(), **({"NCCL_PROTO": proto.strip()} if proto.strip() else {})}


def parse_bands(text: str) -> list[tuple[int, int, str]]:
    bands = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        rng, _, spec = item.partition(":")
        lo, dash, hi = rng.partition("-")
        if not dash or not spec.strip():
            raise ValueError(f"bad band {item!r} (want LO-HI:SPEC)")
        band = (parse_size(lo), parse_size(hi), spec.strip())
        if band[0] >= band[1]:
            raise ValueError(f"empty band {item!r}")
        spec_env(band[2])
        bands.append(band)
    bands.sort()
    for a, b in zip(bands, bands[1:]):
        if a[1] > b[0]:
            raise ValueError(f"overlapping bands {a} {b}")
    return bands


def bands_str(bands) -> str:
    return ",".join(f"{fmt_size(lo)}-{fmt_size(hi)}:{spec}" for lo, hi, spec in bands)


def make_pick(routes, qar=None, qar_lo: int = 0, qar_hi: int = INF):
    """routes: [(lo, hi, comm)]; returns pick(x, default_comm) -> object with .all_reduce."""
    import torch

    def pick(x, default):
        nbytes = x.numel() * x.element_size()
        if qar is not None and qar_lo <= nbytes < qar_hi and x.dtype in (torch.bfloat16, torch.float16):
            return qar
        for lo, hi, comm in routes:
            if lo <= nbytes < hi:
                return comm
        return default
    return pick


# ------------------------------------------------------------------- autotune
def autotune_sizes(hidden: int | None, max_bytes: int = 128 << 20) -> list[int]:
    sizes = {1 << k for k in range(14, 63) if (1 << k) <= max_bytes}
    if hidden:
        sizes |= {t * hidden * 2 for t in TOKENS if t * hidden * 2 <= max_bytes}
    return sorted(sizes)


def autotune(cpu_group, default, candidates, sizes, make_comm, measure,
             budget_s: float = 20.0, prune: float = 4.0, clock=time.monotonic):
    """Collective over cpu_group (gloo). Returns (table {spec: [us per size]}, comms, errors).

    Every per-rank outcome (creation failure, measured time, error, elapsed) is
    MAX-all-reduced before any decision, so all ranks issue the same NCCL calls
    and end with the same table. inf = unmeasured/pruned/failed. The caller owns
    destroying comms it does not keep (never `default`).
    """
    import torch
    import torch.distributed as dist

    def agree(*vals):
        t = torch.tensor(vals, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
        return t.tolist()

    t0 = clock()
    table = {c: [math.inf] * len(sizes) for c in candidates}
    comms, errors = {"default": default}, {}
    for spec in candidates:
        if spec == "default" or agree(clock() - t0)[0] > budget_s:
            continue
        comm = None
        try:
            comm = make_comm(spec)
        except Exception as exc:  # noqa: BLE001 - a candidate may fail, the pool must not
            errors[spec] = f"init: {exc!r}"[:200]
        if agree(float(comm is None))[0]:
            errors.setdefault(spec, "init failed on another rank")
            if comm is not None:
                comm.destroy()
            continue
        comms[spec] = comm
    live = [c for c in candidates if c in comms]
    for i, nbytes in enumerate(sizes):
        for spec in list(live):
            try:
                t, bad = measure(comms[spec], nbytes), 0.0
            except Exception as exc:  # noqa: BLE001
                t, bad = math.inf, 1.0
                errors[spec] = f"{fmt_size(nbytes)}: {exc!r}"[:200]
            t, bad, elapsed = agree(t, bad, clock() - t0)
            if bad and spec != "default":
                errors.setdefault(spec, f"{fmt_size(nbytes)}: failed on another rank")
                table[spec] = [math.inf] * len(sizes)  # never pick a candidate that errored
                live.remove(spec)
            elif not bad:
                table[spec][i] = t
            if elapsed > budget_s:
                return table, comms, errors
        best = min(table[c][i] for c in live)
        live = [c for c in live if c == "default" or table[c][i] <= prune * best]
    return table, comms, errors


def _cross(a: int, b: int) -> int:
    return min(b, max(a + 1, round(math.sqrt(a * b) / 1024) * 1024))


def build_bands(sizes, table, margin: float = 0.05) -> list[tuple[int, int, str]]:
    """Winner per size (default unless beaten by > margin), merged, log-midpoint crossovers."""
    win = []
    for i in range(len(sizes)):
        base, w = table["default"][i], "default"
        if math.isfinite(base):
            t, c = min(((table[c][i], c) for c in table if c != "default"), default=(math.inf, w))
            if t < base * (1 - margin):
                w = c
        win.append(w)
    bands, lo = [], 0
    for i, w in enumerate(win):
        if i + 1 < len(win) and win[i + 1] == w:
            continue
        hi = INF if i + 1 == len(win) else _cross(sizes[i], sizes[i + 1])
        bands.append((lo, hi, w))
        lo = hi
    return bands


def table_str(sizes, table) -> str:
    def cell(t):
        return "-" if not math.isfinite(t) else f"{t:.0f}"
    return ";".join(f"{fmt_size(s)}:" + "/".join(cell(table[c][i]) for c in table)
                    for i, s in enumerate(sizes))


def time_us(fn, iters: int) -> float:
    """Median CUDA-event time of `iters` back-to-back fn() calls (us)."""
    import statistics

    import torch

    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in ev)


def make_comm(spec: str, cpu_group, device):
    """PyNcclCommunicator built while the spec's NCCL_ALGO/NCCL_PROTO are set."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    env = spec_env(spec)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        comm = PyNcclCommunicator(group=cpu_group, device=device)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if comm.disabled:
        raise RuntimeError("PyNcclCommunicator disabled")
    return comm


def tune(cpu_group, device, default, sizes, candidates=CANDIDATES, budget_s=20.0, iters=7):
    """autotune() on real GPUs: exact-sum check + median CUDA-event time per cell."""
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(cpu_group), dist.get_world_size(cpu_group)
    n_max = max(sizes) // 2
    buf = (torch.arange(9, device=device) - 4).to(torch.bfloat16).repeat(-(-n_max // 9))
    buf += rank % 3  # integers in [-4, 6]: every 8-rank sum is exact in bf16
    offs = sum(r % 3 for r in range(world))

    def measure(comm, nbytes):
        x = buf[: nbytes // 2]
        out = comm.all_reduce(x)
        if out is None:
            raise RuntimeError("communicator disabled")
        ok = torch.equal(out, (x - rank % 3) * world + offs)
        del out
        t = time_us(lambda: comm.all_reduce(x), iters if nbytes <= 8 << 20 else 3)
        if not ok:  # after the timed calls: every rank issues the same collectives
            raise RuntimeError("wrong all-reduce result")
        return t

    try:
        return autotune(cpu_group, default, list(candidates), sizes,
                        lambda spec: make_comm(spec, cpu_group, device), measure, budget_s)
    finally:
        buf = None  # noqa: F841 - frees the closure's cell before empty_cache
        torch.cuda.empty_cache()


def _hidden() -> int | None:
    if _env("SUFFIX_NCCL_AUTOTUNE_HIDDEN"):
        return int(_env("SUFFIX_NCCL_AUTOTUNE_HIDDEN"))
    try:
        from vllm.config import get_current_vllm_config_or_none

        return int(get_current_vllm_config_or_none().model_config.get_hidden_size())
    except Exception:  # noqa: BLE001 - log2 sizes alone still cover the range
        return None


# ---------------------------------------------------------------- install
def config():
    """(static bands, autotune, qar mode) from env; ValueError/PatchDriftError on bad config."""
    small, text = _env(ENV), _env(BANDS_ENV)
    if small and text:
        raise PatchDriftError(f"set {BANDS_ENV} or {ENV}, not both")
    bands = []
    if small:
        proto = _env(PROTO_ENV)
        # Measured on worker-06 (allreduce_split_bench 2026-09-27): NCCL's default
        # is only bad in a band (~384 KiB..2 MiB) -> legacy default [256 KiB, 2 MiB].
        bands = [(int(_env(MIN_ENV, str(256 << 10))), int(_env(BYTES_ENV, str(2 << 20))) + 1,
                  small + (f"/{proto}" if proto else ""))]
        spec_env(bands[0][2])
    elif text:
        bands = parse_bands(text)
    autotune_on = _env(AUTOTUNE_ENV) == "1"
    qar = _env(QAR_ENV).lower()
    qar = "" if qar in ("", "0", "off") else qar
    if qar not in ("", "int8", "fp8"):
        raise PatchDriftError(f"{QAR_ENV}={qar!r}: want int8|fp8")
    if (bands or autotune_on) and (os.environ.get("NCCL_ALGO") or os.environ.get("NCCL_PROTO")):
        raise PatchDriftError("size routing needs NCCL_ALGO/NCCL_PROTO unset "
                              "(they would apply to EVERY communicator)")
    return bands, autotune_on, qar


def patch_all_reduce(src: str) -> str:
    """CudaCommunicator.all_reduce source with the size route (dedented)."""
    import ast
    import textwrap

    cls = next((n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)
                and n.name == "CudaCommunicator"), None)
    fn = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef)
               and n.name == "all_reduce"), None)
    if fn is None:
        raise PatchDriftError("CudaCommunicator.all_reduce missing")
    body = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    if body.count(OLD) != 1:
        raise PatchDriftError(f"all_reduce anchor: expected 1, found {body.count(OLD)}")
    out = textwrap.dedent(body.replace(OLD, NEW))
    compile(out, f"<{TAG}>", "exec")
    return out


def _autotune_group(self):
    hidden = _hidden()
    sizes = autotune_sizes(hidden, parse_size(_env("SUFFIX_NCCL_AUTOTUNE_MAX_BYTES", "128M")))
    cands = _env("SUFFIX_NCCL_AUTOTUNE_CANDIDATES", ",".join(CANDIDATES)).split(",")
    cands = ["default"] + [c for c in dict.fromkeys(c.strip() for c in cands) if c and c != "default"]
    budget = float(_env("SUFFIX_NCCL_AUTOTUNE_BUDGET_S", "20"))
    t0 = time.monotonic()
    table, comms, errors = tune(self.cpu_group, self.device, self.pynccl_comm, sizes, cands,
                                budget, int(_env("SUFFIX_NCCL_AUTOTUNE_ITERS", "7")))
    bands = build_bands(sizes, table, float(_env("SUFFIX_NCCL_AUTOTUNE_MARGIN", "0.05")))
    keep = {s for *_, s in bands}
    for spec in [s for s in comms if s not in keep and s != "default"]:
        comms.pop(spec).destroy()
    if self.rank == 0:
        _say(f"AUTOTUNE {self.unique_name} world={self.world_size} hidden={hidden} "
             f"used={time.monotonic() - t0:.1f}s/{budget:g}s bands={bands_str(bands)} "
             f"failed={errors or '-'} cols={'/'.join(table)} us={table_str(sizes, table)}")
    return bands, comms


def setup(self, bands, autotune_on: bool, qar: str) -> None:
    """Runs at the end of CudaCommunicator.__init__ on every rank of the group."""
    self._suffix_pick = None
    if self.world_size <= 1 or self.pynccl_comm is None or self.pynccl_comm.disabled:
        return
    tp = self.unique_name.split(":")[0] == "tp"
    comms = {"default": self.pynccl_comm}
    tuned = False
    if autotune_on and tp:
        try:
            bands, comms = _autotune_group(self)
            tuned = True
        except Exception as exc:  # noqa: BLE001 - symmetric failures (OOM, bad env) degrade
            _say(f"AUTOTUNE FAILED on {self.unique_name} rank {self.rank}: {exc!r}; static bands")
    if not tuned:
        for spec in sorted({s for *_, s in bands} - {"default"}):  # same order on every rank
            comms[spec] = make_comm(spec, self.cpu_group, self.device)
    routes = [(lo, hi, comms[s]) for lo, hi, s in bands if s != "default"]
    qar_obj, qar_lo, qar_hi = None, 0, INF
    if qar and tp:
        from suffix_hybrid.nccl_qar import PyncclQar

        qar_obj = PyncclQar(self.pynccl_comm, qar)
        qar_lo = parse_size(_env("SUFFIX_NCCL_QAR_MIN_BYTES", "4M"))
        qar_hi = parse_size(_env("SUFFIX_NCCL_QAR_MAX_BYTES", "inf"))
    if routes or qar_obj is not None:
        self._suffix_pick = make_pick(routes, qar_obj, qar_lo, qar_hi)
    if self.rank == 0:
        _say(f"{self.unique_name}: bands={bands_str(bands) or '-'} "
             f"qar={f'{qar}[{fmt_size(qar_lo)},{fmt_size(qar_hi)})' if qar_obj else 'off'} "
             f"(world {self.world_size}, {len(comms)} communicator(s))")


def apply(module) -> None:
    bands, autotune_on, qar = config()
    if not (bands or autotune_on or qar):
        return
    import __future__
    import linecache
    from pathlib import Path

    path = Path(module.__file__)
    src = patch_all_reduce(path.read_text())
    fname = f"{path}.{TAG.replace(' ', '-')}.py"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns: dict = {}
    exec(compile(src, fname, "exec", __future__.annotations.compiler_flag,
                 dont_inherit=True), module.__dict__, ns)
    cls = module.CudaCommunicator
    orig_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        setup(self, bands, autotune_on, qar)

    ns["all_reduce"].__qualname__ = "CudaCommunicator.all_reduce"
    cls.all_reduce = ns["all_reduce"]
    cls.__init__ = __init__
    _say(f"ACTIVE: static bands={bands_str(bands) or '-'} autotune={'tp' if autotune_on else 'off'} "
         f"qar={qar or 'off'}")


def install_post_import_hook() -> None:
    """sitecustomize entry; fail closed (SystemExit) on drift/bad config while enabled."""
    if not any(_env(k) for k in (ENV, BANDS_ENV, AUTOTUNE_ENV, QAR_ENV)):
        return

    def run(module):
        try:
            apply(module)
        except Exception as exc:
            raise SystemExit(f"[{TAG}] enabled but installation FAILED: {exc}") from exc

    if TARGET in sys.modules:
        run(sys.modules[TARGET])
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != TARGET:
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
