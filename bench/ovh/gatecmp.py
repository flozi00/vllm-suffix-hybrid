#!/usr/bin/env python3
"""gatecmp.py <TAG> [ledger]: owner gate verdict from aba5.sh rows (gate5.jsonl).

Per concurrency and metric: A = A1+A2 rounds, B = B rounds. WIN if B's range is
strictly better than A's whole range, LOSS if strictly worse, else TIE.
TTFT metrics may lose only if end-to-end latency (ttft_p50 + gen/stream_tps_med)
wins (owner relaxation 2026-10-02).
"""
import json, os, sys
from collections import defaultdict

HIGHER = {"agg_tps": 1, "stream_tps_med": 1, "stream_tps_p10": 1,
          "ttft_p50": -1, "ttft_max": -1, "itl_p99_ms": -1, "e2e_s": -1}
tag = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "gate5.jsonl")
rows = [json.loads(l) for l in open(path) if f'"{tag}-' in l]
cells = defaultdict(lambda: defaultdict(list))  # (arm, conc) -> metric -> values
for r in rows:
    arm = r["arm"][len(tag) + 1:].split("-r")[0]
    side = "B" if arm == "B" else "A"
    if r.get("errors"):
        print(f"WARN errors={r['errors']} in {r['arm']} c{r['conc']}")
    if r.get("stream_tps_med"):
        r["e2e_s"] = r["ttft_p50"] + r["gen_tokens"] / r["stream_tps_med"]
    for m in HIGHER:
        if r.get(m) is not None:
            cells[(side, r["conc"])][m].append(r[m])
loss = 0
for c in sorted({k[1] for k in cells}):
    print(f"--- c={c}")
    e2e_win = False
    for m, sign in HIGHER.items():
        a, b = cells[("A", c)][m], cells[("B", c)][m]
        if not a or not b:
            continue
        if sign > 0:
            v = "WIN" if min(b) > max(a) else "LOSS" if max(b) < min(a) else "TIE"
        else:
            v = "WIN" if max(b) < min(a) else "LOSS" if min(b) > max(a) else "TIE"
        if m == "e2e_s":
            e2e_win = v == "WIN"
        print(f"  {m:15} A {min(a):9.3f}..{max(a):9.3f} (n={len(a)})  B {min(b):9.3f}..{max(b):9.3f} (n={len(b)})  {v}")
        if v == "LOSS":
            loss += 1 if not m.startswith("ttft") else 0
            if m.startswith("ttft"):
                print(f"    (TTFT loss tolerated only if e2e WIN)")
    if not e2e_win:
        print("  NOTE: e2e not a separated WIN at this concurrency")
print(f"VERDICT {tag}: {'PASS' if loss == 0 else f'FAIL ({loss} non-TTFT losses)'}")
