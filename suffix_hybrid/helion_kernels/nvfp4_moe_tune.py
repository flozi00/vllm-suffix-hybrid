# SPDX-License-Identifier: Apache-2.0
"""In-lab tuner for the nvfp4_moe Helion port's DeepSeek-V4 cases.

A kernel lab (inference-console) runs only its fixed gate argv
(`python -m suffix_hybrid.kernels.nvfp4_moe bench|oracle`) with a 30 min cap,
and the autotune lab's queue can be hours deep. A DEV bundle routes the
nvfp4_moe_bench gate to `gate()` here:

  pending    starts ONE detached tuner process (survives the gate call) over
             every pending (kernel, case) of cases(DS_MODEL) (EP8; the no-EP
             TP8 rank too with NVFP4_MOE_TP8_NOEP=1), at the first effort of EFFORTS not finished yet (a
             later pass only replaces a config it beats); later calls print
             progress + the configs not yet printed (paged, stdout)
  pass done  the lane's e2e bench (cold L2: FlashInfer CUTLASS vs the Helion
             plans) on the best tuned config per case, once per effort, then the
             K4 of the fused shared expert (kernels/dsv41_moe_shared.py `bench`:
             fused vs routed-only + vLLM's shared MLP on the aux stream, the
             FUSE_SHARED_MAX_M ceiling), and the next effort's tuner starts

The todo list is the routed plan's 5 kernels plus the 3 fused-shared ones
(helion_kernels/dsv41_moe_shared.py): 53 jobs on EP8, 106 with the no-EP rank.

State lives in ROOT (the lab pod's /tmp): records.jsonl (one row per tune,
the autotune runner's fields) and configs/<kernel>/<platform>.json (the
runner's / helion_kernels' format: [{"key": case, "config": {...}}], best
correct row per case). `use_tuned()` points helion_kernels at it, so the
oracle gate judges the tuned kernels too. Never imported by serving.

  python -m suffix_hybrid.helion_kernels.nvfp4_moe_tune run     # the tuner itself
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from suffix_hybrid import helion_kernels as hk

ROOT = os.environ.get("NVFP4_MOE_TUNE_DIR", "/tmp/nvfp4_moe_tune")
EFFORTS = ("quick", "full")
CASE_S = {"quick": 900, "full": 2400}  # per-case cap (the runner: 1800 at full)
DS_MODEL = {"name": "deepseek-v4.1-flash-1017-nvfp4", "architecture": "DeepseekV41ForCausalLM",
            "tp": 8, "hidden": 5120, "moe_intermediate": 288, "experts": 384, "topk": 6,
            "quant_format": "nvfp4"}
# Expert-load distributions (nm.rand_routing dist). The autotune search runs on TUNE_DIST; the
# winner is then gated and timed on ALL of DISTS, and a row's score is the worse of balanced /
# skewed (hot, one expert with every row, only has to be correct). The search runs on BALANCED
# since the load is re-derived (nm.expected_load, docs/dsv41-moe.md section 8): EP8 allgather
# dispatch gives an expert M/64 rows, so production decode is ~1-3 rows per hit expert (the
# heaviest of a batch ~9 at M=192), a 16-row MMA tile is padding and the skewed tail (80 rows
# on the top expert at M=192) must not steer the row tile; it still gates and scores.
TUNE_DIST = os.environ.get("NVFP4_MOE_TUNE_DIST", "balanced")
DISTS = ("balanced", "skewed", "hot")
SCORED = ("balanced", "skewed")
M_ORDER = (8, 1, 32, 192, 6, 16, 64, 128)  # a whole plan for the key Ms first
PAGE = 7000  # stdout tail the lab returns is 8000 chars


def _path(*p):
    return os.path.join(ROOT, *p)


def variant_kernels() -> list:
    """Lab kernels of every non-default body variant in this build (lane I's dot_scaled bodies
    when helion_kernels/dsv41_moe_fp4mma.py exists): tuned on the same cases so the bench can
    compare variants per bucket."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    return [k for v in hm.available_variants() if v != hm.DEQUANT
            for k in hm.variant_module(v).KERNELS]


def kernels() -> dict:
    """{name: lab kernel} of the routed plan's kernels, the body variants' (dot_scaled) and,
    with the shared expert fused (SUFFIX_DSV41_MOE_FUSE_SHARED), the three fused ones
    (helion_kernels/dsv41_moe_shared.py)."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as ds
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    return {k["name"]: k for k in hm.KERNELS + variant_kernels() + ds.KERNELS}


def todo() -> list:
    """[(kernel, case)] in tuning order: layout (EP8 = prod first), then
    M_ORDER, then the routed plan's kernels, then the fused-shared ones."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as ds
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    per = {n: k["cases"](DS_MODEL) for n, k in kernels().items()}
    layouts = []
    for c in per[hm.FC1]:
        lay = {k: v for k, v in c.items() if k != "numtokens"}
        if lay not in layouts:
            layouts.append(lay)
    order = (hm.ROUTE_QUANT, hm.FC1, hm.FC2_COMBINE, hm.FC2, hm.COMBINE,
             *[k["name"] for k in variant_kernels()],
             ds.FC1_SH, ds.FC2_COMBINE_SH, ds.COMBINE_SH)
    out = []
    for lay in layouts:
        sh = {"sh_inter": ds.sh_inter_of(DS_MODEL, lay)}
        for m in M_ORDER:
            for k in order:
                case = {**lay, "numtokens": m, **(sh if k in ds.NEW_KERNELS else {})}
                if case in per[k]:
                    out.append((k, case))
    return out


def write_variant_rows(rows: list, benched: list) -> None:
    """Merge one bench's body-variant selection rows into <ROOT>/configs/dsv41_moe_variant/
    <platform>.json (the runner's / helion_kernels' format): rows of every benched bucket are
    replaced by this bench's (a bucket where dequant won again loses its row), others stay."""
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    path = _path("configs", hm.VARIANT_KERNEL, hk.platform() + ".json")
    old = []
    if os.path.isfile(path):
        with open(path) as f:
            old = json.load(f)
    case_of = lambda r: r["key"] if isinstance(r["key"], dict) else json.loads(r["key"])
    stale = {hk.case_key(c) for c in benched}
    keep = [r for r in old if hk.case_key(case_of(r)) not in stale]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # by layout, then numerically by bucket
    order = lambda r: (hk.case_key({k: v for k, v in case_of(r).items() if k != "numtokens"}),
                       case_of(r).get("numtokens", 0))
    with open(path, "w") as f:
        json.dump(sorted(keep + rows, key=order), f, indent=1, sort_keys=True)


def records() -> list:
    try:
        with open(_path("records.jsonl")) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


def best(rows) -> dict:
    """{(kernel, case_key): row} — the fastest correct row per case."""
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["kernel"], hk.case_key(r["case"]))
        if key not in out or r["helion_us"] < out[key]["helion_us"]:
            out[key] = r
    return out


def write_configs(rows) -> None:
    by = {}
    for (kernel, ck), r in sorted(best(rows).items()):
        by.setdefault((kernel, r["platform"]), []).append({"key": r["case"], "config": r["config"]})
    for (kernel, plat), entries in by.items():
        os.makedirs(_path("configs", kernel), exist_ok=True)
        with open(_path("configs", kernel, plat + ".json"), "w") as f:
            json.dump(entries, f, indent=1, sort_keys=True)


def use_tuned() -> None:
    """Point helion_kernels at ROOT's configs (this process only)."""
    if os.path.isdir(_path("configs")):
        hk.CONFIG_ROOT = _path("configs")
        hk._TABLES.clear()


class _Timeout(BaseException):  # Helion's autotuner swallows Exception (runner.CaseTimeout)
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _graph_us(fn, args, iters=50) -> float:
    import torch
    fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        fn(*args)
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def tune_one(kern: dict, case: dict, effort: str) -> dict:
    """The autotune runner's _run_case for one (kernel, case): autotune with
    static shapes on TUNE_DIST routing, gate on the kernel's check for every
    distribution of DISTS, time tuned vs its baseline (graph replays, same args:
    L2-warm for both arms; the cold-L2 e2e number is gate()'s bench)."""
    import helion
    import torch
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    rec = {"kernel": kern["name"], "case": case, "effort": effort, "platform": hk.platform(),
           "autotuner": os.environ.get("HELION_AUTOTUNER"), "tune_dist": TUNE_DIST}
    hm.ROUTING = TUNE_DIST
    try:
        args = kern["make"](case, 0)
        t0 = time.time()
        signal.alarm(CASE_S[effort])
        try:
            cfg = helion.kernel(kern["body"], autotune_effort=effort, static_shapes=True,
                                **(kern.get("settings") or {})).autotune(args, force=True)
        finally:
            signal.alarm(0)
        rec["autotune_s"] = round(time.time() - t0, 1)
        rec["config"] = json.loads(cfg.to_json())
        tuned = helion.kernel(kern["body"], config=cfg, static_shapes=True,
                              **(kern.get("settings") or {}))
        del args
        by, msgs = {}, []
        for dist in DISTS:
            hm.ROUTING = dist
            a = kern["make"](case, 0)
            ok, msg = kern["check"](tuned(*a), kern["reference"](*a))
            msgs.append(f"{dist}: {msg}")
            if not ok:
                rec.update(correct=False, check="; ".join(msgs), status="incorrect")
                return rec
            if dist in SCORED:
                by[dist] = {"helion_us": round(_graph_us(tuned, a), 3), "baseline_us": None}
                try:  # TP8 without EP: FlashInfer refuses I = 288, there is no baseline
                    by[dist]["baseline_us"] = round(_graph_us(
                        kern["baseline"], kern["make_baseline"](case, 0)), 3)
                except Exception as exc:  # noqa: BLE001 — a missing baseline never drops a config
                    by[dist]["baseline_error"] = f"{type(exc).__name__}: {exc}"[:200]
            del a
        rec.update(correct=True, check="; ".join(msgs), by_dist=by,
                   helion_us=max(v["helion_us"] for v in by.values()),
                   baseline_us=(None if any(v["baseline_us"] is None for v in by.values())
                                else max(v["baseline_us"] for v in by.values())), status="ok")
    finally:
        hm.ROUTING = None
    torch.cuda.empty_cache()
    return rec


def run(effort: str) -> int:
    """The detached tuner: every pending (kernel, case) at `effort`, in order."""
    hk.ensure_vendor()
    try:  # the runner's default_autotuner: LFBO when sklearn loads
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        os.environ["HELION_AUTOTUNER"] = "LFBOTreeSearch"
    except Exception:  # noqa: BLE001
        os.environ["HELION_AUTOTUNER"] = "PatternSearch"
    signal.signal(signal.SIGALRM, _alarm)
    kerns = kernels()
    for name, case in todo():
        done = {(r["kernel"], hk.case_key(r["case"]), r["effort"]) for r in records()}
        if (name, hk.case_key(case), effort) in done:
            continue
        try:
            rec = tune_one(kerns[name], case, effort)
        except _Timeout:
            rec = {"kernel": name, "case": case, "effort": effort, "status": "timeout"}
        except Exception as exc:  # noqa: BLE001 — a failed case is a result
            rec = {"kernel": name, "case": case, "effort": effort, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"[:1500]}
        with open(_path("records.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        write_configs(records())
        print(json.dumps({k: rec.get(k) for k in ("kernel", "status", "helion_us",
                                                  "baseline_us", "autotune_s")}),
              flush=True)
    return 0


def _alive() -> bool:
    try:
        with open(_path("pid")) as f:
            os.kill(int(f.read()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False


def _tag(case) -> str:
    lay = "ep" if "experts_global" in case else "tp"
    return f"{lay}M{case['numtokens']}"


def _start(effort: str) -> None:
    log = open(_path("tune.log"), "a")
    proc = subprocess.Popen([sys.executable, "-m", "suffix_hybrid.helion_kernels.nvfp4_moe_tune",
                             "run", effort], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, env=dict(os.environ))
    with open(_path("pid"), "w") as f:
        f.write(str(proc.pid))


def gate() -> int:
    """The DEV nvfp4_moe_bench gate (module doc)."""
    os.makedirs(ROOT, exist_ok=True)
    rows = records()
    plan = todo()
    have = {(r["kernel"], hk.case_key(r["case"]), r["effort"]) for r in rows}
    pending = [e for e in EFFORTS if any((k, hk.case_key(c), e) not in have for k, c in plan)]
    finished = [e for e in EFFORTS if e not in pending]
    running = _alive()
    bench_now = (not running and finished
                 and not os.path.exists(_path(f"benched-{finished[-1]}")))
    if not running and pending and not bench_now:
        _start(pending[0])
        running = True
    print(f"tuner {'RUNNING' if running else 'idle'}: done "
          + ", ".join(f"{e} {sum(1 for k, c in plan if (k, hk.case_key(c), e) in have)}/{len(plan)}"
                      for e in EFFORTS), file=sys.stderr)
    try:
        with open(_path("tune.log")) as f:
            print("tune.log: " + f.read()[-600:], file=sys.stderr)
    except FileNotFoundError:
        pass
    for r in rows[-25:]:
        print(f"  {r.get('effort')} {r['kernel']} {_tag(r['case'])} {r.get('status')} "
              f"helion={r.get('helion_us')} base={r.get('baseline_us')} "
              f"t={r.get('autotune_s')} {(r.get('error') or r.get('check') or '')[:120]}",
              file=sys.stderr)
    try:  # configs not printed yet, paged
        with open(_path("printed.json")) as f:
            printed = set(json.load(f))
    except FileNotFoundError:
        printed = set()
    out, size = [], 0
    for (kernel, ck), r in sorted(best(rows).items()):
        tag = f"{kernel}|{ck}|{json.dumps(r['config'], sort_keys=True)}"
        line = json.dumps({"kernel": kernel, "key": r["case"], "config": r["config"],
                           "us": r["helion_us"], "effort": r["effort"]}, sort_keys=True)
        if tag in printed or size + len(line) > PAGE:
            continue
        out.append(line)
        printed.add(tag)
        size += len(line) + 1
    print("\n".join(out))
    with open(_path("printed.json"), "w") as f:
        json.dump(sorted(printed), f)
    if not bench_now:
        return 0
    use_tuned()
    from suffix_hybrid.kernels import nvfp4_moe as nm
    nm.bench(cases=((nm.DS_EP, nm.DS_MS),)
             + (((nm.DS_TP8, nm.DS_MS),) if os.environ.get("NVFP4_MOE_TP8_NOEP") == "1" else ()))
    # the per-bucket body choice of THIS bench (dequant vs dot_scaled), shipped like a config
    write_variant_rows(list(nm.VARIANT_ROWS), list(nm.VARIANT_BENCHED))
    print("variant rows: " + json.dumps(nm.VARIANT_ROWS, sort_keys=True), file=sys.stderr)
    from suffix_hybrid.kernels import dsv41_moe_shared  # K4 of the fused shared expert
    try:  # its own verdict; a failure is printed loudly but must not re-run the routed bench forever
        dsv41_moe_shared.bench()
    except Exception as exc:  # noqa: BLE001
        print(f"{dsv41_moe_shared.MARKER} FUSED-SHARED BENCH FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    open(_path(f"benched-{finished[-1]}"), "w").close()
    if pending:
        _start(pending[0])
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[2]) if sys.argv[1:2] == ["run"] else gate())
