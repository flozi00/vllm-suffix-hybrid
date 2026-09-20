#!/usr/bin/env python3
"""Schema-repeating benchmark against an OpenAI-compatible endpoint.

Generates N agentic-style prompts that all ask for the same JSON output shape
(multi-record summaries with varied content), drives the endpoint with a
bounded thread pool, and reports tokens/sec percentiles from usage counters.

Stdlib only (urllib + threading). Python 3.9+.

Usage:
    python3 bench/bench.py --endpoint http://host:8000/v1 --model my-model \
        --num-prompts 64 --concurrency 8 --label baseline
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

FIRST = ["Anton", "Berta", "Cem", "Dilara", "Emil", "Fatma", "Greta", "Hakan"]
LAST = ["Meyer", "Schulz", "Yilmaz", "Weber", "Krause", "Aydin", "Fuchs", "Brandt"]
DEPTS = ["Sales", "Engineering", "Support", "Finance", "Logistics"]
PRODUCTS = ["Widget Pro", "Mega Widget", "Widget Lite", "Widget XL"]


def make_prompt(seed: int) -> str:
    rng = random.Random(seed)
    records = []
    for i in range(8):
        records.append(
            {
                "id": 1000 + i,
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "department": rng.choice(DEPTS),
                "product": rng.choice(PRODUCTS),
                "quantity": rng.randint(1, 50),
                "revenue_eur": round(rng.uniform(500, 20000), 2),
            }
        )
    return (
        "You are a reporting assistant. Convert the following records into a "
        "markdown table, then a JSON summary with total_revenue, avg_quantity "
        "and top_department. Reply with exactly this structure.\n\n"
        + json.dumps(records, indent=2)
    )


def run_one(endpoint: str, model: str, seed: int, timeout: float):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": make_prompt(seed)}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - started
    usage = data.get("usage", {})
    ct = int(usage.get("completion_tokens", 0))
    return ct, elapsed, ct / elapsed if elapsed > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default="run")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    seeds = list(range(args.num_prompts))
    results = []
    lock = threading.Lock()
    idx = [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(seeds):
                    return
                seed = seeds[idx[0]]
                idx[0] += 1
            try:
                r = run_one(args.endpoint, args.model, seed, args.timeout)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    results.append({"error": str(exc)})
                continue
            with lock:
                results.append({"ct": r[0], "elapsed": r[1], "tps": r[2]})

    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(args.concurrency)
    ]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall

    ok = [r for r in results if "error" not in r]
    errs = len(results) - len(ok)
    tps = sorted(r["tps"] for r in ok)
    total_ct = sum(r["ct"] for r in ok)

    def pct(p):
        if not tps:
            return 0.0
        return tps[min(len(tps) - 1, int(len(tps) * p))]

    print(f"[{args.label}] prompts={len(results)} errors={errs} "
          f"wall={wall:.1f}s completion_tokens={total_ct}")
    if tps:
        print(f"[{args.label}] aggregate_tok_s={total_ct / wall:.1f} "
              f"p50={pct(0.50):.1f} p95={pct(0.95):.1f} "
              f"mean={statistics.mean(tps):.1f}")
    else:
        print(f"[{args.label}] no successful requests")


if __name__ == "__main__":
    main()