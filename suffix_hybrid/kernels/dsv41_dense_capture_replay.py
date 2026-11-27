# SPDX-License-Identifier: Apache-2.0
"""Boot gate ``mxfp8_capture_replay``: is the DeepSeek-V4.1 Helion MXFP8 GEMM lane safe under CUDA-graph
capture? (helion_kernels/mxfp8_gemm.py, kernels/mxfp8_linear.py.) Motivation: glm g14 hit an illegal memory
access in the drafter's prefill capture on all ranks with our own small-M dense GEMM (per-tile counters,
last-arriving CTA reduces) among the two suspects; the bisect never finished. The serving lane therefore
runs the two-stage split-K only (fp32 partials from torch's allocator + a separate reduce launch, no
semaphores, no scratch state) and this gate is its proof.

    python -m suffix_hybrid.kernels.mxfp8_linear capture_replay [--quick | --full]
        [--variant two_stage | semaphore] [--replays N] [--streams N] [--only SUBSTR]
        [--no-capture] [--no-guards] [--json]

Per probe (body x real V4.1 per-rank shape x M x explicit config; plus every SHIPPED config of this platform):
  1. eager on three inputs (compiles / loads outside any capture); the first output must pass the K3 check
     against the exact f64 product (out and every partial NaN-poisoned first: a skipped tile fails);
  2. capture the SAME call on a side stream (torch.cuda.graph), replay N times with NEW data copied into the
     static input, bit-exact against eager each time;
  3. guarded output: the same, with `out` carved from an arena with sentinel bands on both sides (a stray
     store past the output is caught in software, no sanitizer needed);
  4. --streams 2: two graphs of the probe (private pools) replayed concurrently on two streams, repeatedly,
     bit-exact (the lane's scratch must not be shared state; the semaphore variant's counters are, on purpose).
The label is printed (and flushed) BEFORE a probe runs, so a crash names the culprit; the verdict line is
``[suffix mxfp8-capture] MXFP8-CAPTURE-REPLAY PASS|FAIL|NOT RUN``, exit 0 / 1 / 2.

compute-sanitizer (the gate does no timing and no autotuning, so it is sanitizer-friendly; use --quick and
few replays; the caching allocator stays ON because torch cannot allocate inside a capture otherwise):
    compute-sanitizer --tool memcheck  python -m suffix_hybrid.kernels.mxfp8_linear capture_replay --quick --replays 2
    compute-sanitizer --tool synccheck python -m ...   # barrier misuse (the semaphore variant's debug_barrier in a branch)
    compute-sanitizer --tool racecheck python -m ...   # shared-memory hazards of the cp.async pipelines
Isolate capture from kernel bugs with --no-capture (eager only) and CUDA_LAUNCH_BLOCKING=1; the semaphore
variant needs SUFFIX_MXFP8_GEMM_SEMAPHORE=1 and --variant semaphore (boot gate ``mxfp8_capture_replay_sem``).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import mxfp8_gemm as hx

MARKER = "[suffix mxfp8-capture]"
GUARD = 4096  # sentinel elements on each side of a guarded output
SENT = 0x5A5A  # int16 sentinel bit pattern (a bf16 NaN-free finite value, not what a kernel writes)
INPUTS = 3


@dataclass
class Probe:
    label: str
    kernel: str
    n: int
    k: int
    m: int
    split: int = 1
    groups: int = 1
    bmm: bool = False
    cfg: dict = field(default_factory=dict)
    source: str = "probe"  # probe | shipped

    def config(self) -> dict:
        # explicit, shape-independent defaults: 16x32 tiles, 64 k per step; bm grows with M
        bm = 16 if self.m <= 16 else 64
        return {"split_k": self.split, "block_sizes": [bm, 32, 2], "num_warps": 4, "num_stages": 2,
                **self.cfg}


def _p(label, kernel, n, k, m, split, groups=1, bmm=False):
    return Probe(label, kernel, n, k, m, split, groups, bmm)


def probes(tier: str, variant: str = "two_stage") -> list:
    """quick: nine probes that cover single launch, two-stage, K tails, high split, big partials,
    wo_a (bmm) with strided groups. full: every V4.1 TP8 shape x bucket x body x split {1, 2, 8}."""
    sem = variant == "semaphore"
    a16, a8 = (hx.A16_SEM, hx.A8_SEM) if sem else (hx.A16, hx.A8)
    if tier == "tiny":  # CPU tests
        return [_p("tiny.single", a16, 64, 160, 5, 1), _p("tiny.split2", a16, 64, 160, 5, 2),
                _p("tiny.a8.split2", a8, 64, 160, 5, 2),
                _p("tiny.bmm2", a16, 32, 96, 3, 2, groups=2, bmm=True)]
    if tier == "quick":
        return [_p("wq_b.m8.s1", a16, 4096, 1280, 8, 1),
                _p("wq_b.m8.s4", a16, 4096, 1280, 8, 4),
                _p("wq_b.m48.a8.s2", a8, 4096, 1280, 48, 2),
                _p("shared.down.m48.s2", a16, 5120, 288, 48, 2),
                _p("shared.gate_up.m8.s16", a16, 576, 5120, 8, 16),
                _p("wo_a.g1.m8.s4", a16, 1024, 4096, 8, 4, 1, True),
                _p("wo_a.g2.m192.s2", a16, 1024, 4096, 192, 2, 2, True),
                _p("fused_wqa_wkv.m192.a8.s8", a8, 1792, 5120, 192, 8),
                _p("main_proj.m64.s4", a16, 5120, 15360, 64, 4)]
    out = []
    for lin, (n, k) in hx.dsv41_shapes(8).items():
        for m in (8, 32, 64, 192):
            for split in (1, 2, 8):
                if lin == "wo_a":
                    out.append(_p(f"{lin}.m{m}.s{split}", a16, n, k, m, split, 1, True))
                else:
                    out += [_p(f"{lin}.m{m}.s{split}", a16, n, k, m, split),
                            _p(f"{lin}.m{m}.a8.s{split}", a8, n, k, m, split)]
    return out


def shipped_probes() -> list:
    """Every config the serving lane would run on this platform (configs/helion/<kernel>/<platform>.json)."""
    out = []
    for kernel in (hx.A16, hx.A8):
        for key, cfg in hk.table(kernel).items():
            case = json.loads(key)
            n, k, m = case["n"], case["k"], case["numtokens"]
            out.append(Probe(f"shipped.{kernel}.{n}x{k}.m{m}" + (".wo_a" if "g" in case else ""),
                             kernel, n, k, m, int(cfg.get("split_k", 1)), 1, "g" in case, dict(cfg),
                             "shipped"))
    return out


# ---------------------------------------------------------------------------------------------
def _operands(pr: Probe, dev):
    sem = pr.kernel in hx.SEM_BODIES
    ps = [hx.problem({"k": pr.k, "n": pr.n, "numtokens": pr.m}, 10 + i, dev, sem=False)
          for i in range(pr.groups)]
    w = torch.cat([p["w"] for p in ps])
    ws = torch.cat([p["ws"] for p in ps])
    gen = torch.Generator(device=dev).manual_seed(77 + pr.n + pr.k + pr.m)
    shape = (pr.m, pr.groups, pr.k) if pr.bmm else (pr.m, pr.k)
    xs = [torch.randn(*shape, generator=gen, device=dev).to(torch.bfloat16) for _ in range(INPUTS)]
    counters = torch.zeros(hx.counter_elems(pr.n), dtype=torch.int32, device=dev) if sem else None
    return w, ws, xs, counters


def _build(pr: Probe, dev):
    """(entry, w, ws, xs, counters) with the probe's kernel bound to its explicit config."""
    import helion
    w, ws, xs, counters = _operands(pr, dev)
    kern = helion.kernel(hx.ALL_BODIES[pr.kernel], config=helion.Config(**pr.config()),
                         static_shapes=False, ignore_warnings=hx._ignore())
    return (pr.kernel, kern, int(pr.config()["split_k"]), pr.m), w, ws, xs, counters


def _exact_ok(pr: Probe, out, x, w, ws):
    """K3 per group against the exact f64 product (A8: of the spec-quantized x)."""
    n = pr.n
    for i in range(pr.groups):
        xi = x[:, i, :] if pr.bmm else x
        wi, si = w[i * n:(i + 1) * n], ws[i * n:(i + 1) * n]
        if pr.kernel in (hx.A8, hx.A8_SEM):
            q, s = hx.quant_spec(xi)
            ref = hx.exact(hx.dequant(q, s, torch.float32), wi, si)
        else:
            ref = hx.exact(xi, wi, si)
        ok, msg = hx.check(out[:, i * n:(i + 1) * n], ref)
        if not ok:
            return False, f"group {i}: {msg}"
    return True, msg


def _guarded(m: int, cols: int, dev):
    arena = torch.full((GUARD + m * cols + GUARD,), SENT, dtype=torch.int16, device=dev)
    return arena, arena[GUARD:GUARD + m * cols].view(torch.bfloat16).view(m, cols)


def _guards_intact(arena) -> bool:
    return bool((arena[:GUARD] == SENT).all()) and bool((arena[-GUARD:] == SENT).all())


def _concurrent(pr: Probe, entry, w, ws, xs, counters, dev, rounds: int) -> None:
    """Two graphs of the same call (separate static inputs / private pools; ONE counters buffer for
    the semaphore variant, like a lane that shares scratch) replayed together on two streams."""
    n_out = pr.groups * pr.n
    statics = [xs[0].clone(), xs[1].clone()]
    graphs, outs = [], []
    for st in statics:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream()):
            outs.append(hx.apply(st, w, ws, entry, pr.n, pr.groups, pr.bmm, poison_nan=True,
                                 counters=counters))
        graphs.append(g)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    eager = [hx.apply(x, w, ws, entry, pr.n, pr.groups, pr.bmm, poison_nan=True,
                      counters=counters).clone() for x in xs]
    torch.cuda.synchronize()
    cur = torch.cuda.current_stream()
    for r in range(rounds):
        pick = [(r + i) % len(xs) for i in range(2)]
        for st, j in zip(statics, pick):
            st.copy_(xs[j])
        for g, s in zip(graphs, streams):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                g.replay()
        for s in streams:
            cur.wait_stream(s)
        torch.cuda.synchronize()
        for o, j in zip(outs, pick):
            if not hx.bits_equal(o, eager[j]):
                raise RuntimeError(f"concurrent replay {r}: output of input {j} differs from eager "
                                   f"({n_out} columns) — shared state between two graphs")


def run_probe(pr: Probe, rt, dev, replays: int, streams: int, capture: bool, guards: bool) -> dict:
    rec = {"label": pr.label, "kernel": pr.kernel, "n": pr.n, "k": pr.k, "m": pr.m,
           "split": pr.split, "groups": pr.groups, "source": pr.source}
    try:  # --- 1. eager, compile outside any capture (a failure here is NOT RUN, not FAIL)
        entry, w, ws, xs, counters = _build(pr, dev)
        run = lambda x: hx.apply(x, w, ws, entry, pr.n, pr.groups, pr.bmm, poison_nan=True,
                                 counters=counters)
        eager0 = run(xs[0]).clone()
        rt.sync()
    except Exception as exc:  # noqa: BLE001
        rec.update(status="NOT RUN", detail=f"{type(exc).__name__}: {exc}"[:600])
        return rec
    try:
        ok, msg = _exact_ok(pr, eager0, xs[0], w, ws)
        if not ok:
            rec.update(status="FAIL", detail=f"eager vs exact: {msg}")
            return rec
        rec["check"] = msg
        if not capture:
            rec.update(status="PASS", detail="eager only (--no-capture)")
            return rec
        n_ok = hx.capture_check(run, xs, rt, replays, what=pr.label)                      # --- 2
        if guards:                                                                          # --- 3
            arena, view = _guarded(pr.m, pr.groups * pr.n, dev)

            def run_g(x):
                view.fill_(float("nan"))
                return hx.apply(x, w, ws, entry, pr.n, pr.groups, pr.bmm, out=view,
                                poison_nan=True, counters=counters)
            hx.capture_check(run_g, xs, rt, replays, what=pr.label + " [guarded out]")
            rt.sync()
            if not _guards_intact(arena):
                raise RuntimeError(f"{MARKER} GUARD FAIL {pr.label}: a store landed outside `out`")
        if streams >= 2 and rt is hx.CudaRuntime:                                          # --- 4
            _concurrent(pr, entry, w, ws, xs, counters, dev, max(2, replays))
        if counters is not None and bool(counters.any()):
            raise RuntimeError(f"{MARKER} {pr.label}: split-K tickets not back at zero")
        rec.update(status="PASS", detail=f"{n_ok} replays bit-exact" + (", guards intact" if guards else "")
                   + (", 2-stream concurrent ok" if streams >= 2 and rt is hx.CudaRuntime else ""))
    except Exception as exc:  # noqa: BLE001 — a capture / replay / guard failure is the finding
        rec.update(status="FAIL", detail=f"{type(exc).__name__}: {exc}"[:900])
    return rec


def run_gate(tier="quick", variant="two_stage", rt=None, dev=None, replays=6, streams=2,
             capture=True, guards=True, only="", plist=None,
             emit=lambda s: print(s, flush=True)) -> dict:
    """The gate as a function (the lab driver and the CPU tests call it). Returns
    {"verdict": PASS | FAIL | NOT RUN, "variant", "runtime", "probes": [...]}."""
    rt = rt or hx.CudaRuntime
    dev = dev or ("cuda" if rt is hx.CudaRuntime else "cpu")
    if variant == "semaphore" and not hx.semaphore_on():
        msg = (f"{MARKER} --variant semaphore needs {hx.SEM_ENV}=1: the last-arriver variant is UNPROVEN under "
               "CUDA-graph capture and only this gate and lab_main may run it")
        emit(msg)
        return {"verdict": "NOT RUN", "variant": variant, "runtime": rt.name, "probes": [], "reason": msg}
    plist = plist if plist is not None else (
        probes(tier, variant) + (shipped_probes() if variant == "two_stage" else []))
    plist = [p for p in plist if only in p.label]
    emit(f"{MARKER} {variant}: {len(plist)} probes, {replays} replays each, runtime {rt.name}, "
         f"capture={capture} guards={guards} streams={streams}")
    recs = []
    for pr in plist:
        emit(f"{MARKER} PROBE {pr.label}: {pr.kernel} N={pr.n} K={pr.k} M={pr.m} split={pr.split} "
             f"groups={pr.groups}{' bmm' if pr.bmm else ''} [{pr.source}]")
        rec = run_probe(pr, rt, dev, replays, streams, capture, guards)
        recs.append(rec)
        emit(f"{MARKER} PROBE {pr.label}: {rec['status']} — {rec.get('detail', '')}")
        if rec["status"] == "FAIL" and ("CUDA" in rec["detail"] or "illegal" in rec["detail"]):
            emit(f"{MARKER} sticky CUDA error: stopping at the first faulting probe")
            break
    fail = [r for r in recs if r["status"] == "FAIL"]
    ran = [r for r in recs if r["status"] == "PASS"]
    verdict = "FAIL" if fail else ("PASS" if ran else "NOT RUN")
    emit(f"{MARKER} MXFP8-CAPTURE-REPLAY {verdict} variant={variant} probes {len(ran)} pass / "
         f"{len(fail)} fail / {len(recs) - len(ran) - len(fail)} not run")
    return {"verdict": verdict, "variant": variant, "runtime": rt.name, "probes": recs}


# The sitecustomize.py `_BOOT_GATES` entries (paste them there; tests/test_dsv41_dense_capture_replay.py
# parses these argv with the gate's own parser, so a flag typo cannot ship).
BOOT_GATES = {
    "mxfp8_capture_replay": (["-m", "suffix_hybrid.kernels.mxfp8_linear", "capture_replay"], {}),
    "mxfp8_capture_replay_sem": (["-m", "suffix_hybrid.kernels.mxfp8_linear", "capture_replay",
                                  "--variant", "semaphore"], {hx.SEM_ENV: "1"}),
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="capture_replay", description=__doc__.split("\n")[0])
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--quick", action="store_true", help="nine probes (default)")
    tier.add_argument("--full", action="store_true", help="every V4.1 TP8 shape x bucket x body x split")
    ap.add_argument("--variant", choices=("two_stage", "semaphore"), default="two_stage")
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--streams", type=int, default=2, help="2 = also replay two graphs concurrently")
    ap.add_argument("--only", default="", help="run probes whose label contains this")
    ap.add_argument("--no-capture", action="store_true", help="eager only (sanitizer isolation)")
    ap.add_argument("--no-guards", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the result dict as the last line")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if not torch.cuda.is_available():
        print(f"{MARKER} MXFP8-CAPTURE-REPLAY NOT RUN: no CUDA device", flush=True)
        return 2
    cc = torch.cuda.get_device_capability()
    print(f"{MARKER} gpu {torch.cuda.get_device_name(0)} cc {cc[0]}.{cc[1]} platform {hk.platform()}",
          flush=True)
    res = run_gate("full" if a.full else "quick", a.variant, replays=a.replays, streams=a.streams,
                   capture=not a.no_capture, guards=not a.no_guards, only=a.only,
                   emit=lambda s: print(s, flush=True))
    if a.json:
        print(json.dumps(res), flush=True)
    return {"PASS": 0, "FAIL": 1, "NOT RUN": 2}[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
