#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# A/B runbook for the GLM pool (TP=8, k=12 MTP): native-MTP reference vs the
# OLD synchronous V2 hook vs the NEW (async mirror + cold-skip) hook.
#
# THIS FILE IS A RUNBOOK. It is never executed by CI or by agents. It only
# drives bench/compare.py against the live endpoint; every pool change is an
# OPERATOR ACTION done in the inference-console UI between arms.
#
# Operator preconditions (once per arm switch, in the console pool env edit):
#   ARM native : SUFFIX_HYBRID_WRAP=0
#   ARM oldhook: SUFFIX_HYBRID_WRAP=1  SUFFIX_HYBRID_SYNC_HOOK=1
#   ARM newhook: SUFFIX_HYBRID_WRAP=1  (new default; no SYNC_HOOK flip)
# After ANY env edit the pool must be redeployed from the console (GLM is a
# console pool — the devsync.sh kill-PID1 in-place restart flow is for the
# next-spec-dev pool only and does NOT apply here). Wait for the pod ready +
# one smoke completion before timing.
#
# Per-section CPU timing inside a hook arm (optional, low overhead): set
# SUFFIX_HYBRID_PROFILE=1 in the same env edit before redeploy; it prints
# periodic per-section timing lines to the worker stderr.
#
# Gates:
#   correctness: the multiset of per-request content/reasoning SHA-256s must
#     be identical across all arms (compare.py pins temperature=0 + seed, so
#     outputs are deterministic up to scheduling; a hook must not change text).
#   speed: summary.aggregate_tok_s per arm; expect oldhook << native and
#     newhook recovering most of the gap (native-MTP reference ~100 tok/s at
#     this workload shape).
# 2 repeats per arm; we report per-repeat and the best-of for speed, and the
# correctness hashes across every run of every arm.
#
# Usage:  OPENAI_API_KEY=*** scripts/ab_glm.sh [results-dir]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENDPOINT="https://glm.pl-ai.net/v1"
MODEL="glm"
OUT_DIR="${1:-$REPO/results}"
mkdir -p "$OUT_DIR"

run_arm () {
  local arm="$1" rep="$2"
  echo "== ARM=$arm repeat=$rep  (operator: confirm console env is $arm and pool was redeployed; press Enter)"
  read -r _
  "$REPO/.venv/bin/python" "$REPO/bench/compare.py" \
    --endpoint "$ENDPOINT" --model "$MODEL" \
    --num-prompts 32 --concurrency 4 --max-tokens 512 \
    --label "$arm" --output "$OUT_DIR/${arm}_r${rep}.json" \
    || echo "WARN: $arm r$rep exited nonzero (compare.py: nonzero = request errors; see $OUT_DIR/${arm}_r${rep}.json)"
}

for rep in 1 2; do
  run_arm native  "$rep"   # SUFFIX_HYBRID_WRAP=0 — native-MTP reference (~100 tok/s expected)
  run_arm oldhook "$rep"   # SUFFIX_HYBRID_WRAP=1 + SUFFIX_HYBRID_SYNC_HOOK=1 — old sync hook
  run_arm newhook "$rep"   # SUFFIX_HYBRID_WRAP=1 clean — new async-mirror/cold-skip hook
done

echo "== correctness gate: per-request output-hash fingerprints (must be equal across ALL arms/repeats)"
"$REPO/.venv/bin/python" - "$OUT_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
fingerprints = {}
for f in sorted(out.glob("*_r*.json")):
    art = json.loads(f.read_text())
    blob = "\n".join(sorted(r.get("content_sha256", "ERR") + "|" +
                            r.get("reasoning_sha256", "") for r in art["requests"]))
    fingerprints[f.name] = hashlib.sha256(blob.encode()).hexdigest()
for name, fp in fingerprints.items():
    print(f"{fp}  {name}")
uniq = set(fingerprints.values())
print("CORRECTNESS:", "PASS" if len(uniq) == 1 and fingerprints else f"FAIL ({len(uniq)} distinct outputs)")
PY

echo "== speed gate: aggregate tok/s per run (compare best-of-2 per arm)"
"$REPO/.venv/bin/python" - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
per_arm = {}
for f in sorted(out.glob("*_r*.json")):
    s = json.loads(f.read_text())["summary"]
    per_arm.setdefault(f.name.split("_r")[0], []).append(
        (f.name, s.get("aggregate_tok_s"), s.get("errors")))
for arm, rows in per_arm.items():
    for name, tps, errors in rows:
        print(f"{arm:8s} {name:16s} aggregate_tok_s={tps} errors={errors}")
best = {a: max((t for _, t, _ in rows if t), default=None) for a, rows in per_arm.items()}
print("BEST", json.dumps(best))
if all(a in best and best[a] for a in ("native", "oldhook", "newhook")):
    print(f"oldhook/native={best['oldhook']/best['native']:.2f}  newhook/native={best['newhook']/best['native']:.2f}")
PY
