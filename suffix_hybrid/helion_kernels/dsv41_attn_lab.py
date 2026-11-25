# SPDX-License-Identifier: Apache-2.0
"""K4 harness: DeepSeek-V4.1 Helion sparse-MLA attention vs FlashInfer, one command.

Runs in a kernel-lab pod (ONE RTX PRO 6000 Blackwell, vLLM nightly 39983a5e7
image so FlashInfer is the one that serves; the plugin mounted at /plugins):

    python -m suffix_hybrid.helion_kernels.dsv41_attn_lab --rec v4 --kernels partial,row \\
        --buckets 1,6,12,24,48,96,192,512,2048,8192 --search --out /results/dsv41-attn

For every (kernel, T bucket, layer shape, context class): builds the realistic problem of
``dsv41_attn._make`` (sequences of 6 rows, ~80% top-k overlap row to row, compressed
pool sized by the context class), runs FlashInfer (the baseline: exactly vLLM's SM120
call) and the Helion candidates, checks each candidate (K3) against the exact f64
reference on sampled rows, times baseline and candidate (CUDA events, back-to-back
launches, warm L2 and a cold-L2 variant, host call cost without sync) and writes

    <out>/table.csv      one row per case: baseline us, best us, speedup, K3 message, achieved GB/s
    <out>/configs/<kernel>/<platform>.json   ONLY the winning rows (speedup > --margin on the warm AND the
                         cold timing, K3 PASS), in the exact shape of suffix_hybrid/configs/helion/
    <out>/step_estimate.txt  per-step attention ms, FlashInfer vs Helion, summed over the layer mix
                         (38 compressed layers + 2 SWA-only + 3 draft) at the measured per-call times

A ``partial`` candidate is timed as the PAIR it ships as: dsv41_attn_partial followed by
dsv41_attn_merge (two launches, no cross-CTA synchronisation); the autotune lab (Autotune tab,
KERNELS ABI in dsv41_attn.py) times the partial alone and is therefore optimistic: use it to
search, and this script to decide what ships. ``--search`` runs a small manual grid
(num_splits x bk x num_warps x num_stages); without it only the configs already shipped are
timed. Host cost per call is reported separately (the attention runs in the eager gap of
vLLM's breakable CUDA graphs, so Helion's per-call Python overhead is part of the price).
``--plan`` prints the cases without importing CUDA (used by the CPU test).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

from suffix_hybrid.helion_kernels import dsv41_attn as D
from suffix_hybrid.helion_kernels import dsv41_attn_ref as R

SEARCH = {  # manual search space per body (the lab's LFBO covers the same axes and more)
    D.PARTIAL: {"num_splits": (1, 2, 4, 8, 16, 32), "bk": (16, 32, 64), "num_warps": (4, 8), "num_stages": (1, 2, 3)},
    D.ROW: {"bk": (16, 32, 64), "num_warps": (4, 8), "num_stages": (1, 2, 3)},
}
# layer mix of one forward (cost model): (swa, topk) -> number of layers
LAYER_MIX = {(R.WINDOW, R.TOPK): 38, (R.WINDOW, 0): 2, (192, 0): R.DRAFT_LAYERS}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rec", choices=("v4", "v41"), default="v4")
    ap.add_argument("--kernels", default="partial,row")
    ap.add_argument("--buckets", default=",".join(str(b) for b in D.ROW_BUCKETS))
    ap.add_argument("--ctx", default=",".join(c for c, _ in D.CTX_CLASSES),
                    help="context classes for the decode buckets of compressed layers")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--shapes", default="128:512,128:0,192:0", help="swa:topk list")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="dsv41-attn-lab")
    ap.add_argument("--search", action="store_true", help="manual config search (default: shipped configs only)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--overlap", default="", help="torch file with a dumped topk_indices_buffer [T, 512] (rows of "
                    "consecutive spec-decode rows): print how much their top-k sets overlap and exit")
    ap.add_argument("--margin", type=float, default=1.0, help="ship a row only when baseline/best > margin")
    return ap.parse_args(argv)


def plan_cases(a) -> list:
    """[(kernel, case)] in run order; buckets outside a body's range are dropped."""
    names = {"partial": D.PARTIAL, "row": D.ROW}
    shapes = [tuple(int(x) for x in s.split(":")) for s in a.shapes.split(",")]
    ctxs = a.ctx.split(",")
    out = []
    for k in a.kernels.split(","):
        kernel = names[k]
        for b in (int(x) for x in a.buckets.split(",")):
            if b not in D.buckets(kernel):
                continue
            for sw, tk in shapes:
                cls = ctxs if D.case_ctx(b, tk, "c512k") != "any" else ["any"]
                out += [(kernel, D.case(b, a.heads, sw, tk, a.rec, c)) for c in cls]
    return out


def configs_for(kernel, search: bool):
    if not search:
        return [None]
    sp = SEARCH[kernel]
    keys = list(sp)
    return [dict(zip(keys, v)) for v in itertools.product(*sp.values())]


def to_config(kernel, c: dict) -> dict:
    cfg = {"block_sizes": [c["bk"]], "num_warps": c["num_warps"], "num_stages": c["num_stages"]}
    if kernel == D.PARTIAL:
        cfg["num_splits"] = c["num_splits"]
    return cfg


def time_it(fn, iters: int, cold: bool):
    import torch

    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda") if cold else None
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    host = 0.0
    for s, e in evs:
        if flush is not None:
            flush.zero_()
        s.record()
        t0 = time.perf_counter()
        fn()
        host += time.perf_counter() - t0
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in evs)
    return ms[len(ms) // 2] * 1e3, host / iters * 1e6  # median device us, mean host us per call


def runner(kernel, kern, merge, args, out):
    """The candidate as it ships: the row kernel, or partial + merge (two launches)."""
    if kernel == D.ROW:
        a = list(args)
        a[12] = out
        return lambda: kern(*a)
    sink = args[11]
    return lambda: merge(*kern(*args)[:2], sink, out)


def main(argv=None) -> int:
    a = parse(argv)
    if a.overlap:
        import torch

        idx = torch.load(a.overlap, map_location="cpu")
        idx = idx.get("topk_indices", idx) if isinstance(idx, dict) else idx
        print(json.dumps({"rows": int(idx.shape[0]), **R.overlap_stats(idx.reshape(idx.shape[0], -1), group=6)}))
        return 0
    cases = plan_cases(a)
    if a.plan:
        for kernel, c in cases:
            print(kernel, json.dumps(c, sort_keys=True))
        return 0
    import helion
    import torch

    from suffix_hybrid import helion_kernels as hk

    os.makedirs(a.out, exist_ok=True)
    plat = hk.platform()
    merge = D.merge_kernel()
    rows, wins, per_call = [], {}, {}
    for kernel, c in cases:
        args = D._make(kernel)(c, 0)
        ref = D._reference(kernel)(*args)
        t = c["numtokens"]
        out_b = torch.empty(t, c["heads"], R.HD, dtype=torch.bfloat16, device="cuda")
        baseline = D._baseline(kernel)
        ok_b, msg_b = D._check(baseline(*args), ref)      # (row args carry their own out buffer)
        run_base = lambda: baseline(*args)  # noqa: E731
        t_base, h_base = time_it(run_base, a.iters, False)
        t_base_cold, _ = time_it(run_base, max(10, a.iters // 5), True)
        best = None
        for cand in configs_for(kernel, a.search):
            try:
                if cand is None:
                    kern = hk.compiled(D._BODY[kernel], kernel, c, **D.SETTINGS)
                else:
                    kern = helion.kernel(D._BODY[kernel], config=helion.Config(**to_config(kernel, cand)),
                                         static_shapes=False, **D.SETTINGS)
                out = torch.zeros_like(out_b)
                fn = runner(kernel, kern, merge, args, out)
                fn()
                ok, msg = D._check(out, ref)
                if not ok:
                    print(f"K3 FAIL {kernel} {json.dumps(c)} {cand}: {msg}", file=sys.stderr)
                    continue
                t_us, h_us = time_it(fn, a.iters, False)
                if best is None or t_us < best[0]:
                    best = (t_us, h_us, cand, msg, time_it(fn, max(10, a.iters // 5), True)[0])
            except Exception as exc:  # noqa: BLE001 -- a config that does not compile is data, not a crash
                print(f"SKIP {kernel} {json.dumps(c)} {cand}: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        srec, crec = args[-2], args[-1]
        nbytes = t * (c["swa"] * R.ROW_BYTES[srec] + c["topk"] * R.ROW_BYTES[crec])  # no-reuse upper bound
        row = {"kernel": kernel, "T": t, "swa": c["swa"], "topk": c["topk"], "rec": c["rec"], "ctx": c["ctx"],
               "base_us": round(t_base, 2), "base_cold_us": round(t_base_cold, 2), "base_host_us": round(h_base, 1),
               "base_k3": ok_b, "best_us": None, "best_cold_us": None, "best_host_us": None, "speedup": None,
               "speedup_cold": None, "config": None, "k3": None, "gbs_best": None}
        if best:
            t_us, h_us, cand, msg, t_cold = best
            row.update(best_us=round(t_us, 2), best_cold_us=round(t_cold, 2), best_host_us=round(h_us, 1),
                       speedup=round(t_base / t_us, 3), speedup_cold=round(t_base_cold / t_cold, 3),
                       config=json.dumps(cand), k3=msg, gbs_best=round(nbytes / (t_us * 1e3), 1))
            per_call[(kernel, c["swa"], c["topk"], t, c["ctx"])] = (t_base, t_us)
            if t_base / t_us > a.margin and t_base_cold / t_cold > a.margin and cand is not None:
                wins.setdefault(kernel, []).append({"key": c, "config": to_config(kernel, cand)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(a.out, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["kernel"])
        w.writeheader()
        w.writerows(rows)
    for kernel, rs in wins.items():
        d = os.path.join(a.out, "configs", kernel)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, plat + ".json"), "w") as f:
            json.dump(rs, f, indent=1)
    write_step_estimate(os.path.join(a.out, "step_estimate.txt"), per_call)
    return 0


def write_step_estimate(path: str, per_call: dict) -> None:
    """Per-step attention time = sum over the layer mix of the per-call time at the
    bucket that serves T = batch * rows (decode) -- baseline vs best, per context class."""
    lines = []
    for kernel in sorted({k for k, *_ in per_call}):
        for t in sorted({t for k, _, _, t, _ in per_call if k == kernel}):
            for ctx in sorted({c for k, _, _, tt, c in per_call if k == kernel and tt == t}):
                base = best = 0.0
                for (sw, tk), n in LAYER_MIX.items():
                    got = per_call.get((kernel, sw, tk, t, ctx)) or per_call.get((kernel, sw, tk, t, "any"))
                    if got:
                        base, best = base + n * got[0], best + n * got[1]
                if best:
                    lines.append(f"{kernel} T={t} ctx={ctx}: FlashInfer {base / 1e3:.3f} ms/step, Helion "
                                 f"{best / 1e3:.3f} ms/step ({base / best:.2f}x) [sum over "
                                 f"{sum(LAYER_MIX.values())} layers; measured per-call us]")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
