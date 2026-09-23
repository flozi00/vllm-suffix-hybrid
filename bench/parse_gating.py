#!/usr/bin/env python3
"""Parse suffix_hybrid gating counters from a pod log — the verification step
every bench arm MUST run before its numbers count (the p16-floor hole: an arm
was benched without ever measuring its full-path share).

Accepts either:
  * a saved MCP get_logs spillover file ({"result": "{\"pod\": ..., \"log\": ...}"}),
  * raw log text.

Usage:
  python3 bench/parse_gating.py <file> [--pid-regex]

Prints per-rank counter maxima, total steps, full-path share, and the
skips_frozen/skips_probe/skips_unchanged split. Exit 1 when no counters are
found (wrap not armed / wrong log window) so CI-style gating can catch it.

Counter line format (Worker pids differ per rank):
  ... (Worker_TP0_EP0 pid=1561) suffix_hybrid v2 skips_frozen #100
Full-path steps are the steps NOT covered by any skip counter; the mixer
stats line carries pid=NNNN too:
  ... suffix_hybrid v2 mixer pid=1561 mixes=... (older builds)
The full-path share = 1 - sum(skips)/steps, where steps is the largest
#N seen on any counter (they all tick on the same step loop).
"""
import argparse
import json
import re
import sys

COUNTER = re.compile(
    r"Worker_TP(\d)_EP\d pid=\d+\) suffix_hybrid v2 (skips_\w+) #(\d+)")
# suffix_hybrid_native {"active_tracked": 7, "cache": {...}, "mixes": N, ...}
STATS = re.compile(r"suffix_hybrid v2 mixer pid=(\d+) mixes=(\d+)")
STATS_JSON = re.compile(
    r"suffix_hybrid_native (\{.*\"mixes\": ?(\d+).*\})")
# RFC3339 k8s timestamp at line start: 2026-09-22T21:38:48.225796327Z
TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _parse_ts(line):
    m = TS.match(line)
    return m.group(1) if m else None


def window_report(log, pod):
    """Per-minute full-path share from timestamped cumulative counters.

    Tests H1 (frozen-share mis-measurement): cumulative counters over the pod's
    whole life can hide a much higher full-path share inside the bench window.
    Counters tick on disjoint steps, so per-minute steps = sum of counter
    deltas; full-path = delta of skips_unchanged + delta of mixes.
    """
    # series[rank][counter] -> {minute: max_N}  (max within minute: prints
    # are cadenced, but re-prints of the same value collapse via max)
    series = {}
    mixes = {}  # minute -> max mixes value seen
    for line in log.splitlines():
        ts = _parse_ts(line)
        if ts is None:
            continue
        m = COUNTER.search(line)
        if m:
            rank, kind, n = int(m.group(1)), m.group(2), int(m.group(3))
            d = series.setdefault(rank, {}).setdefault(kind, {})
            d[ts] = max(d.get(ts, 0), n)
            continue
        m = STATS.search(line)
        if m:
            mixes[ts] = max(mixes.get(ts, 0), int(m.group(2)))
            continue
        m = STATS_JSON.search(line)
        if m:
            try:
                obj = json.loads(m.group(1))
                mixes[ts] = max(mixes.get(ts, 0), int(obj["mixes"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
    if not series:
        print(f"{pod}: no timestamped counters for window analysis")
        return 1
    for rank in sorted(series):
        kinds = series[rank]
        minutes = sorted({ts for d in kinds.values() for ts in d} |
                         set(mixes))
        if len(minutes) < 2:
            print(f"  TP{rank}: need >=2 minutes of counters, have "
                  f"{len(minutes)}")
            continue
        prev = {}
        print(f"  TP{rank} per-minute full-path share "
              f"(steps=ΣΔcounters, full=Δunchanged+Δmixes):")
        for ts in minutes:
            cur = {k: d.get(ts, None) for k, d in kinds.items()}
            cur["mixes"] = mixes.get(ts)
            # carry forward last known value within this minute set
            deltas = {}
            for k, v in cur.items():
                if v is None:
                    continue
                if k in prev and v >= prev[k]:
                    deltas[k] = v - prev[k]
                prev[k] = v
            skip_d = sum(v for k, v in deltas.items()
                         if k.startswith("skips_"))
            full_d = deltas.get("skips_unchanged", 0) + \
                (deltas.get("mixes", 0) or 0)
            steps_d = skip_d + (deltas.get("mixes", 0) or 0)
            share = (full_d / steps_d) if steps_d else 0.0
            bar = "#" * int(share * 40)
            print(f"    {ts} steps={steps_d:5d} full={full_d:5d} "
                  f"share={share:6.1%} {bar}")
    return 0


def extract_log(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    # Spillover JSON wrapper ({"result": "{\"pod\":...,\"log\":...}"})
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and "result" in outer:
            inner = outer["result"]
            if isinstance(inner, str):
                inner = json.loads(inner)
            if isinstance(inner, dict) and isinstance(inner.get("log"), str):
                return inner["log"], inner.get("pod", "?")
    except (json.JSONDecodeError, ValueError):
        pass
    return text, "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--window", action="store_true",
                    help="per-minute full-path share (tests H1: bench-window "
                         "share vs cumulative share)")
    args = ap.parse_args()
    log, pod = extract_log(args.file)
    if args.window:
        return window_report(log, pod)
    per_rank = {}   # rank -> {counter: max N}
    for line in log.splitlines():
        m = COUNTER.search(line)
        if not m:
            continue
        rank, kind, n = int(m.group(1)), m.group(2), int(m.group(3))
        d = per_rank.setdefault(rank, {})
        d[kind] = max(d.get(kind, 0), n)
    # Full-path mixes tick no skip counter; recover them from the mixer
    # stats line. mixes= is the count of FULL-path mixes (state["mixes"]
    # only increments inside _publish's changed path... it ticks on every
    # _publish call including unchanged publishes).
    mixes_by_pid = {}
    for line in log.splitlines():
        m = STATS.search(line)
        if m:
            mixes_by_pid[m.group(1)] = int(m.group(2))
        m = STATS_JSON.search(line)
        if m:
            try:
                obj = json.loads(m.group(1))
                mixes = obj.get("mixes")
                if mixes is not None:
                    # pid is not in the JSON body; attribute to rank -1 (all)
                    mixes_by_pid.setdefault("*", 0)
                    mixes_by_pid["*"] = max(mixes_by_pid["*"], int(mixes))
            except (json.JSONDecodeError, ValueError):
                pass
    if not per_rank:
        if mixes_by_pid:
            print(f"pod={pod} (mixes only)")
            print(f"  full mixes: {mixes_by_pid}")
            return 0
        print(f"{pod}: NO suffix_hybrid counters found — wrap not armed or "
              f"log window misses the counters. DO NOT TRUST a bench from "
              f"this log.", file=sys.stderr)
        return 1
    print(f"pod={pod}")
    mixes_known = None
    if mixes_by_pid:
        vals = [v for v in mixes_by_pid.values()]
        mixes_known = max(vals) if vals else None
    for rank in sorted(per_rank):
        d = per_rank[rank]
        # Total steps = every skip counter + changed publishes (mixes).
        # p12-era builds tick skips_unchanged on full-path publishes only,
        # so max()-based steps is wrong across builds; sum() is uniform.
        skipped = sum(v for k, v in d.items() if k.startswith("skips_"))
        changed = mixes_known if mixes_known is not None else 0
        steps = skipped + changed
        full_path = d.get("skips_unchanged", 0) + changed
        stall_share = (full_path / steps) if steps else 0.0
        print(f"  TP{rank}: steps≈{steps} " +
              " ".join(f"{k}={v}" for k, v in sorted(d.items())) +
              (f" changed_mixes={changed}" if mixes_known is not None
               else " changed_mixes=?"))
        print(f"        full-path share={stall_share:.1%} "
              f"(unchanged={d.get('skips_unchanged', 0)}, "
              f"probe-gated={d.get('skips_probe', 0)}, "
              f"frozen={d.get('skips_frozen', 0)})")
    if mixes_by_pid:
        print(f"  mixer full-mix counts (by pid): {mixes_by_pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())