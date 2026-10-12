"""python3 bench/ovh/test_gatecmp.py — gatecmp verdict on a synthetic A/B/A ledger."""
import json, os, subprocess, sys, tempfile

def row(arm, tps, ttft):
    return {"arm": arm, "conc": 1, "gen_tokens": 384, "agg_tps": tps, "stream_tps_med": tps,
            "stream_tps_p10": tps, "ttft_p50": ttft, "ttft_max": ttft, "itl_p99_ms": 20, "errors": 0}

def verdict(b_tps, b_ttft):
    rows = [row(f"T-A{s}-r{i}", 100 + i, 0.10) for s in (1, 2) for i in range(3)]
    rows += [row(f"T-B-r{i}", b_tps + i, b_ttft) for i in range(3)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "gatecmp.py"), "T", fh.name],
                         capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]

assert verdict(150, 0.10).endswith("PASS"), "separated throughput win must pass"
assert verdict(150, 0.20).endswith("PASS"), "TTFT loss tolerated when e2e wins"
assert "FAIL" in verdict(50, 0.10), "separated throughput loss must fail"
print("gatecmp ok")
