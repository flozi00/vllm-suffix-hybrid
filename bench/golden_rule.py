#!/usr/bin/env python3
"""Golden-rule verdict from bench/results.jsonl.

The golden rule: the plugin's key metrics (c1/c8 tokens/s, sequence length)
must BEAT the vllm baseline (no hybrid speculative decoding). This tool
computes the verdict from the registry instead of hand-assembled numbers:

  python3 bench/golden_rule.py [--arm p24-verifierwin] [--baseline p18-inert]

Each results.jsonl row: {"arm", "pod", "c1": [...], "c8": [...],
"share": 0.0-1.0 or null, "note", "date", "reuse_c1"/"reuse_c8": [...]}.
Baseline defaults to the best-measured inert arm (p18).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REG = Path(__file__).with_name("results.jsonl")


def load():
    rows = {}
    for line in REG.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["arm"]] = r
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="single arm to compare (default: best c1 armed arm)")
    ap.add_argument("--baseline", default="p18-inert")
    args = ap.parse_args()
    rows = load()
    if args.baseline not in rows:
        print(f"baseline {args.baseline!r} not in registry", file=sys.stderr)
        return 1
    base = rows[args.baseline]
    base_c1, base_c8 = _mean(base.get("c1")), _mean(base.get("c8"))
    print(f"baseline: {args.baseline} pod={base.get('pod')} "
          f"c1={base_c1:.0f} c8={base_c8:.0f}" if base_c1 and base_c8
          else f"baseline: {args.baseline} (incomplete)")
    print()
    hdr = f"{'arm':26s} {'c1':>6} {'Δc1':>7} {'c8':>7} {'Δc8':>7} {'share':>6} verdict"
    print(hdr)
    print("-" * len(hdr))
    for arm, r in rows.items():
        if arm == args.baseline:
            continue
        c1, c8 = _mean(r.get("c1")), _mean(r.get("c8"))
        if c1 is None:
            continue
        d1 = c1 - base_c1 if base_c1 else None
        d8 = (c8 - base_c8) if (c8 and base_c8) else None
        share = r.get("share")
        sh = f"{share:5.1%}" if share is not None else "    -"
        beats = (d1 or -1) > 0 and (d8 is None or d8 > 0)
        verdict = "GOLDEN RULE MET" if beats else "below baseline"
        print(f"{arm:26s} {c1:6.0f} {d1:+7.0f} "
              f"{c8 if c8 else 0:7.0f} {d8 if d8 is not None else 0:+7.0f} "
              f"{sh} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())