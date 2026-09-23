#!/usr/bin/env python3
"""Corpus-reuse benchmark — the workload the suffix cache is built for.

bench.py's prompts are all UNIQUE (seeded records), so cross-request draft
reuse can never materialize; mixer stats across p12-p17 confirm it (suffix
contribution ~0.5% of accepted tokens, flat). This variant measures the
plugin's designed upside: many requests sharing long repeated context.

Phases:
  1. warm:     a few requests ingest the shared corpus into the suffix cache
  2. measured: N requests whose prompts share (a) a long common system
               preamble and (b) verbatim-repeated document blocks, with only
               a small per-request question varied. Expected-output drift is
               minor (greedy), so suffix continuations of the repeated blocks
               have high hit probability.

Usage mirrors bench.py:
    python3 bench/bench_reuse.py --endpoint https://... --model m \
        --num-prompts 48 --concurrency 8 --label reuse
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

DOC = (
    "Quarterly compliance report extract follows.\n"
    "Section 1 Scope: This report covers the data-processing activities of "
    "the organisation for the reporting quarter, including all automated "
    "decision systems, profiling activities, and processor engagements "
    "notified under Article 30 of the regulation.\n"
    "Section 2 Legal basis: Processing occurs under legitimate interest, "
    "contract performance, and where applicable explicit consent; each "
    "recorded activity lists its basis, retention period, and the categories "
    "of data subjects affected.\n"
    "Section 3 Technical measures: Pseudonymisation of identifiers at "
    "ingestion, AES-256 encryption at rest, TLS 1.3 in transit, role-based "
    "access control with quarterly review, and immutable audit logging with "
    "a fourteen-month retention window.\n"
    "Section 4 Transfers: No transfers to third countries occurred during "
    "the reporting period; all processors are located within the union and "
    "bound by standard contractual clauses.\n"
)

QUESTIONS = [
    "Summarise Section 3 in two sentences.",
    "Summarise Section 2 in two sentences.",
    "What retention window applies to audit logs?",
    "Which sections mention encryption, and how?",
    "List the legal bases named in the report.",
    "Does the report record any third-country transfers?",
]


def make_prompt(seed: int) -> str:
    rng = random.Random(seed)
    doc = DOC * 2  # verbatim repetition inside one prompt as well
    q = QUESTIONS[seed % len(QUESTIONS)]
    return (
        "You are a compliance assistant. Use only the document below.\n\n"
        + doc
        + f"\n\nQuestion: {q}\nAnswer concisely."
    )


def run_one(endpoint: str, model: str, seed: int, timeout: float):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": make_prompt(seed)}],
            "max_tokens": 512,
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
    ct = int(data.get("usage", {}).get("completion_tokens", 0))
    return ct, elapsed, ct / elapsed if elapsed > 0 else 0.0


def drive(endpoint, model, seeds, concurrency, timeout, label):
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
                r = run_one(endpoint, model, seed, timeout)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    results.append({"error": str(exc)})
                continue
            with lock:
                results.append({"ct": r[0], "elapsed": r[1], "tps": r[2]})

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(concurrency)]
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

    print(f"[{label}] prompts={len(results)} errors={errs} "
          f"wall={wall:.1f}s completion_tokens={total_ct}")
    if tps:
        print(f"[{label}] aggregate_tok_s={total_ct / wall:.1f} "
              f"p50={pct(0.50):.1f} p95={pct(0.95):.1f} "
              f"mean={statistics.mean(tps):.1f}")
    else:
        print(f"[{label}] no successful requests")
    return total_ct / wall if wall else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=48)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4,
                    help="corpus-ingestion requests before the measured run")
    ap.add_argument("--label", default="reuse")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    if args.warmup > 0:
        drive(args.endpoint, args.model, list(range(args.warmup)),
              min(args.concurrency, 4), args.timeout, f"{args.label}-warm")
    drive(args.endpoint, args.model,
          list(range(args.warmup, args.warmup + args.num_prompts)),
          args.concurrency, args.timeout, args.label)


if __name__ == "__main__":
    main()