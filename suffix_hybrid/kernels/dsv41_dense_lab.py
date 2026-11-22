# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 dense-GEMM kernel-lab session (lane D) in ONE command, in a fixed order:

    python3 -c "import sys; sys.path.append('/plugins'); \\
        from suffix_hybrid.kernels.dsv41_dense_lab import main; raise SystemExit(main())" [--effort quick]

  1. markov   the DSpark Markov head chain: five serial BF16 [B, 256] @ [129280, 256]^T calls per step
              (scripts/dsv41_dense_inventory.py: 331 MB/rank/step, as much as lm_head). FIRST because it
              decides whether the biggest uncovered byte stream is real (cold vs warm L2) before any
              MXFP8 case is tuned for it; the MXFP8 candidate cases run with the others in stage 3.
  2. capture  the mxfp8_capture_replay gate on the two-stage lane (helion_kernels/mxfp8_gemm.py): a FAIL
              stops the session, because timing a lane that is not capture-safe proves nothing.
  3. cases    mxfp8_gemm.lab_main: autotune + CUDA-graph timing of the composite that serves (stage 1 +
              torch reduce) vs vLLM's pick and Marlin, A16 then A8 (+ the Markov W2 shapes on A16).

One JSON line per result on stdout, progress on stderr. Needs vLLM only for the stage-3 baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

STAGES = ("markov", "capture", "cases")
MARKOV_N, MARKOV_K, MARKOV_CHAIN = 129280, 256, 5
ROOF_GBS = 1597 * 0.85  # assumed achievable DRAM bandwidth, RTX PRO 6000 Blackwell Server (estimate)
MARKOV_CASES = [{"k": MARKOV_K, "n": MARKOV_N, "numtokens": 8},
                {"k": MARKOV_K, "n": MARKOV_N, "numtokens": 32}]  # B = 1..6 / up to 32 sequences


def _time_us(fn, device: str, iters: int) -> float:
    """us per call of fn(): a CUDA graph + events on cuda, perf_counter on cpu (tests)."""
    fn()
    if device != "cuda":
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1e6 / iters
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def markov_bench(rows=(1, 8, 32), iters=200, device="cuda", n=MARKOV_N, k=MARKOV_K,
                 chain=MARKOV_CHAIN, flush_mb=512, emit=None) -> list:
    """Per row count B: us per call of the serial chain with W2 warm (L2 may hold the 66 MB between
    calls) and cold (L2 flushed before every call, the flush time subtracted), effective GB/s, the
    roofline, and whether the warm chain looks L2-resident (a requantised W2 then buys nothing)."""
    emit = emit or (lambda rec: print(json.dumps(rec), flush=True))
    w = torch.randn(n, k, device=device).to(torch.bfloat16)
    flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device=device)
    w_bytes = w.numel() * w.element_size()
    roof_us = w_bytes / (ROOF_GBS * 1e9) * 1e6
    out = []
    for b in rows:
        xs = [torch.randn(b, k, device=device).to(torch.bfloat16) for _ in range(chain)]
        per = {}
        for cold in (False, True):
            def run():
                prev = None
                for i in range(chain):
                    if cold:
                        flush.zero_()
                    y = torch.nn.functional.linear(xs[i], w)
                    prev = y.float().argmax(-1)  # the sampler's dependent work between calls
                return prev

            def only_flush():
                for _ in range(chain):
                    flush.zero_()
            total = _time_us(run, device, iters)
            if cold:
                total = max(total - _time_us(only_flush, device, iters), 0.0)
            per[cold] = total / chain
        rec = {"stage": "markov", "rows": b, "warm_us_per_call": round(per[False], 2),
               "cold_us_per_call": round(per[True], 2), "roofline_us": round(roof_us, 2),
               "warm_gbps": round(w_bytes / per[False] / 1e3, 1) if per[False] else None,
               "cold_gbps": round(w_bytes / per[True] / 1e3, 1) if per[True] else None,
               "chain_calls": chain, "w2_mb": round(w_bytes / 1e6, 1)}
        rec["l2_resident"] = per[False] < 0.6 * per[True]
        rec["hint"] = ("calls 2..5 hit L2: a requantised W2 buys nothing"
                       if rec["l2_resident"] else
                       "W2 re-read from DRAM each call: an MXFP8 W2 (stage 3 case) halves the bytes")
        emit(rec)
        out.append(rec)
    return out


def markov_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DSpark Markov head chain bench (stage 1 of the lab)")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args(argv)
    print(f"gpu {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
    markov_bench(tuple(a.rows), a.iters)
    return 0


def capture_stage(replays: int = 6, emit=None) -> dict:
    """Stage 2: the boot gate's probes, in-process; the verdict decides whether the session continues."""
    from suffix_hybrid.kernels import dsv41_dense_capture_replay as gate
    emit = emit or (lambda rec: print(json.dumps(rec), flush=True))
    res = gate.run_gate("quick", "two_stage", replays=replays,
                        emit=lambda s: print(s, file=sys.stderr, flush=True))
    rec = {"stage": "capture", "verdict": res["verdict"], "variant": res["variant"],
           "probes": [{k: r.get(k) for k in ("label", "status", "detail")} for r in res["probes"]]}
    emit(rec)
    return rec


def cases_stage(effort: str, budget_s: float, markov_cases: bool = True, kernels=None) -> None:
    """Stage 3: lab_main for A16 (+ the Markov W2 shapes) then A8."""
    from suffix_hybrid.helion_kernels import mxfp8_gemm as hx
    names = list(kernels or (hx.A16, hx.A8))
    for name in names:
        plan = {"kernels": [name], "effort": effort, "budget_s": budget_s}
        if name == hx.A16 and markov_cases:
            plan["cases"] = hx._cases({"name": "deepseek-v4.1"}, hx.A16) + MARKOV_CASES
        print(json.dumps({"stage": "cases", "kernel": name,
                          "cases": len(plan.get("cases") or hx._cases({"name": "deepseek-v4.1"}, name))}),
              flush=True)
        hx.lab_main(plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsv41_dense_lab", description=__doc__.split("\n")[0])
    ap.add_argument("--effort", choices=("none", "quick", "full"), default="quick")
    ap.add_argument("--budget-s", type=float, default=1500, help="per kernel body, stage 3")
    ap.add_argument("--skip", default="", help=f"comma list of {STAGES} to skip")
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--no-markov-cases", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a capture FAIL (never for a ship decision)")
    a = ap.parse_args(argv)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    bad = skip - set(STAGES)
    if bad:
        ap.error(f"unknown stages {sorted(bad)}")
    if not torch.cuda.is_available():
        print("dsv41_dense_lab: no CUDA device", file=sys.stderr)
        return 2
    print(f"gpu {torch.cuda.get_device_name(0)}; stages "
          f"{[s for s in STAGES if s not in skip]}", file=sys.stderr, flush=True)
    for stage in STAGES:
        if stage in skip:
            continue
        print(f"== stage {stage}", file=sys.stderr, flush=True)
        if stage == "markov":
            markov_bench()
        elif stage == "capture":
            rec = capture_stage(a.replays)
            if rec["verdict"] == "FAIL" and not a.keep_going:
                print("dsv41_dense_lab: capture/replay FAIL — stopping before any timing", file=sys.stderr,
                      flush=True)
                return 1
        else:
            cases_stage(a.effort, a.budget_s, not a.no_markov_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
