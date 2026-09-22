# SPDX-License-Identifier: Apache-2.0
"""Hot-path scaling benchmark: mixer.mix_numpy + Engine.propose.

Measures the per-decode-step cost of the plugin's own code as concurrency and
context length grow. This is the cost the plugin ADDS to every engine step, so
it is what turns a single-request win into a 16-way throughput drop.

Run from the repo root:  python bench/scale_bench.py [--rows 16] [--len 8192]
"""
import argparse
import os
import time

import numpy as np

from suffix_hybrid._native import Engine, HybridMixer


def _batch(rows, ctx_len, k, vocab=50_000, seed=7):
    """A steady-state batch: rows requests, each with ctx_len tokens of history.

    Token stream is pseudo-random with a repeated block so the suffix cache has
    real continuations to find (an empty cache would skip the search work).
    """
    rng = np.random.default_rng(seed)
    block = rng.integers(1000, vocab, size=256, dtype=np.int64)
    width = ctx_len + k + 8
    tokens = np.zeros((rows, width), dtype=np.int64)
    for i in range(rows):
        reps = (width + 255) // 256
        pattern = np.tile(block, reps)[:width]
        tokens[i, :] = pattern
        # A per-row tail so rows are distinct (no cross-row aliasing).
        tokens[i, ctx_len - 8 : ctx_len] = rng.integers(1000, vocab, size=8)
    counts = np.full(rows, ctx_len, dtype=np.int64)
    ids = [f"req-{i}" for i in range(rows)]
    drafts = [[int(tokens[i, ctx_len + j]) for j in range(k)] for i in range(rows)]
    return ids, counts, tokens, drafts


def bench_mixer(rows, ctx_len, k, steps, feed_cache):
    mixer = HybridMixer(k, ctx_len + 4096)
    if feed_cache:
        # Realistic warm cache: the corpus is what finished requests leave.
        for s in range(64):
            mixer.suffix_cache.add_sequence(
                [int(x) for x in np.tile(np.arange(256) + 1000 * s, ctx_len // 256)]
            )
    ids, counts, tokens, drafts = _batch(rows, ctx_len, k)
    accepted = [k] * rows
    mixer.mix_numpy(ids, counts, tokens, drafts, None)  # warm: seed per-row state
    best = float("inf")
    total = 0.0
    for s in range(steps):
        started = time.perf_counter_ns()
        mixer.mix_numpy(ids, counts, tokens, drafts, accepted)
        elapsed = time.perf_counter_ns() - started
        best = min(best, elapsed)
        total += elapsed
    return best / 1000.0, total / steps / 1000.0


def bench_engine(rows, ctx_len, k, steps):
    engine = Engine(k, ctx_len + 4096)
    ids, counts, tokens, drafts = _batch(rows, ctx_len, k)
    sampled = [[int(tokens[i, ctx_len])] for i in range(rows)]
    engine.propose(sampled, counts, tokens)  # warm
    best = float("inf")
    total = 0.0
    for _ in range(steps):
        started = time.perf_counter_ns()
        engine.propose(sampled, counts, tokens)
        elapsed = time.perf_counter_ns() - started
        best = min(best, elapsed)
        total += elapsed
    return best / 1000.0, total / steps / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--len", dest="ctx_len", type=int, default=8192)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    print(f"rows={args.rows} ctx_len={args.ctx_len} k={args.k} steps={args.steps}")
    print(f"{'path':<26}{'best_us':>10}{'mean_us':>10}")
    for label, fn in (
        ("mixer (cold cache)", lambda: bench_mixer(args.rows, args.ctx_len, args.k, args.steps, False)),
        ("mixer (warm cache)", lambda: bench_mixer(args.rows, args.ctx_len, args.k, args.steps, True)),
        ("engine (suffix+ngram)", lambda: bench_engine(args.rows, args.ctx_len, args.k, args.steps)),
    ):
        best, mean = fn()
        print(f"{label:<26}{best:>10.3f}{mean:>10.3f}")


if __name__ == "__main__":
    main()
