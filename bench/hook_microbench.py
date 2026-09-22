#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only microbenchmark of the V2 hook's propose() hot path.

Fakes replace the CUDA runner/TP group exactly like tests/test_wrap_v2_install.py
(SimpleNamespace runner whose req_states buffers are plain CPU tensors; TP stub
rank 0 / world 8 with identity broadcast). The compiled Rust HybridMixer is
REAL, so the numbers cover the whole per-step wire-up the wrapper pays:
clone, idx/total_len gather + copy, full-history slice+copy, .tolist() of the
native draft, mix_numpy, per-row torch.tensor row writes, broadcast.

Arms:
  warm  -- corpus pre-fed into the mixer's suffix cache (mix_core finalizes
           fed context the way tests/test_mixer.py drives it), so the cache
           path with real speculation depth is measured.
  cold  -- fresh, never-fed mixer. The baseline pays the IDENTICAL full
           per-step cost even though the mixer returns the native draft
           untouched: this is the headline the async/cold-skip rewrite must
           collapse toward zero.

Module under test is swappable: --module takes any dotted path (a wrap_v2
lookalike exposing _wrap_propose with the same signature). Add its import root
to sys.path with --sys-path (repeatable) — that is how the HEAD-baseline
snapshot package (sh_baseline_pkg.wrap_v2 in TMPDIR) is loaded without
touching the live tree.

Usage:
  .venv/bin/python bench/hook_microbench.py --module suffix_hybrid.wrap_v2 \
      --rows 32 --context 8192 --steps 2000 --arm both
"""
import argparse
import cProfile
import importlib
import io
import json
import pstats
import random
import statistics
import sys
import time
from types import SimpleNamespace as NS

import torch

K = 12  # GLM pool MTP draft steps


class TP:
    """TP stub: owner rank of a world-8 group, identity broadcast."""
    rank_in_group = 0
    world_size = 8

    def broadcast(self, value, src=0):
        return value


def build_corpus(rows, context, seed=1234):
    """Structured repeatable token sequences so n-gram speculation has real
    depth (a flat random vocab over tiny ids would be noise for n=8)."""
    rng = random.Random(seed)
    blocks = [[rng.randrange(10_000, 60_000) for _ in range(16)]
              for _ in range(24)]
    sequences = []
    for _ in range(4):
        seq = []
        for _ in range(rows * context // (4 * 16)):
            seq.extend(rng.choice(blocks))
        sequences.append(seq)
    return sequences


def make_env(module, rows, context, fed, extra_cols):
    """Fresh (runner, batch, wrapped propose, step, advance).

    warm/fed rows turn over like production: row u starts at an offset into
    the buffer slack so its wrap (context departs -> mix_core finalizes the
    old history into the cache -> Reset) lands at a staggered step instead of
    all rows finalizing simultaneously. cold rows never advance (state frozen,
    cache stays unfed — the arm's whole point).
    """
    from suffix_hybrid._native import HybridMixer

    mixer = HybridMixer(K, 2 * context)
    if fed:
        for seq in build_corpus(rows, context):
            mixer.suffix_cache.add_sequence(seq)

    # Row u's history is the arithmetic tail of u*1e6 — guaranteed to depart
    # from every other row so finalize/reset paths in mix_core see realistic
    # distinct contexts. Each step appends the native draft at the row's own
    # `len` (Continuing path, delta of K tokens); at buffer end the row's
    # len snaps back to `context` and the next mix sees a Reset.
    history = torch.zeros((rows, context + extra_cols), dtype=torch.int64)
    for u in range(rows):
        history[u, :context] = torch.arange(context) + u * 1_000_000
    total_len = torch.full((rows,), context, dtype=torch.int64)
    offset = torch.zeros(rows, dtype=torch.int64)
    if fed and extra_cols >= rows * K:
        # Spread first wraps across the buffer so finalize/Reset churn is
        # ~one row per step (steady serving) not 32 rows in one step.
        offset = (torch.arange(rows, dtype=torch.int64) * extra_cols) // rows
        offset -= offset % K  # keep writes buffer-aligned
        total_len += offset
    width = context + extra_cols
    runner = NS(req_states=NS(total_len=NS(gpu=total_len),
                              all_token_ids=NS(gpu=history)))
    batch = NS(req_ids=[f"r{i}" for i in range(rows)],
               num_reqs=rows,
               idx_mapping=torch.arange(rows, dtype=torch.int64))
    native = torch.tensor(
        [[u * 1000 + s + 1 for s in range(K)] for u in range(rows)],
        dtype=torch.int64)

    # ns=K, nr=0: the full width is verified every step except one slot
    # rejected (ns+nr-1 = K-1 of K published) — a healthy steady row.
    num_sampled = torch.full((rows,), K - 1, dtype=torch.int64)
    num_rejected = torch.ones(rows, dtype=torch.int64)
    temperature = torch.zeros(rows, dtype=torch.float32)
    seeds = torch.ones(rows, dtype=torch.int64)
    last_sampled = torch.zeros(rows, dtype=torch.int64)
    next_prefill = torch.zeros(rows, K, dtype=torch.int64)

    def original(*args, **kwargs):
        return native

    wrapped = module._wrap_propose(runner, original, mixer, TP())

    def step():
        wrapped(batch, {}, {}, torch.zeros(1), None,
                num_sampled, num_rejected, last_sampled,
                next_prefill, temperature, seeds)

    def advance():
        # Inter-step postprocess: persist the published draft into the
        # authoritative buffers. A row whose write would pass the buffer end
        # wraps back to `context + offset[u]` instead — its history now
        # departs from the tracked context, so the NEXT mix sees a Reset and
        # mix_core finalizes the old history into the cache (the ids-depart
        # path the warm arm must exercise at steady, staggered turnover).
        pos = total_len.clone()
        wrap = pos + K > width
        # Boolean-row mask + per-row column gather: native[~wrap] picked all
        # K draft tokens; we need the [non-wrapped rows, K] block.
        rows_keep = torch.nonzero(~wrap).squeeze(1)
        cols = pos[rows_keep].unsqueeze(1) + torch.arange(K).unsqueeze(0)
        history[rows_keep.unsqueeze(1), cols] = native[rows_keep]
        # copy_ (not rebind): the wrapper reads runner.req_states.total_len.gpu.
        total_len.copy_(torch.where(wrap, context + offset, pos + K))

    return step, advance, mixer


def _stats(samples_us):
    return {
        "mean_us": statistics.fmean(samples_us),
        "median_us": statistics.median(samples_us),
        "p99_us": statistics.quantiles(samples_us, n=100)[-1],
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def run_arm(module, rows, context, steps, arm):
    # Buffer slack ~ one turnover per row: extra = rows*K so staggered rows
    # wrap at least once across the measured window.
    extra = max(rows * K, 64 * K)
    step, advance, mixer = make_env(module, rows, context,
                                    fed=(arm == "warm"), extra_cols=extra)
    # Warm the code paths before timing (imports settled, allocator warm),
    # past the first all-rows-fresh mix.
    for _ in range(3):
        step()
        if arm == "warm":
            advance()
    samples_us = []
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        step()
        samples_us.append((time.perf_counter_ns() - t0) / 1000.0)
        # Cold arm: state frozen — no history growth, no finalize, cache
        # stays unfed. This is exactly the steady state the rewrite must
        # short-circuit; the baseline still pays the full per-step path.
        if arm == "warm":
            advance()

    result = {"arm": arm, "steps": steps}
    result.update(_stats(samples_us))
    result["cache_tokens"] = mixer.get_stats()["cache"]["cached_tokens"]
    result["mixer_calls"] = mixer.get_stats()["calls"]
    return result


def profile_arm(module, rows, context, steps, arm, top=6):
    extra = max(rows * K, 64 * K)
    step, advance, mixer = make_env(module, rows, context,
                                    fed=(arm == "warm"), extra_cols=extra)
    step()
    advance()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(steps):
        step()
        if arm == "warm":
            advance()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top + 8)
    lines = []
    for line in stream.getvalue().splitlines():
        fields = line.split()
        # cProfile table rows: ncalls tottime percall cumtime percall funcname
        def _num(s):
            return s.replace('.', '', 1).replace('/', '', 1).isdigit()
        if (len(fields) >= 6 and _num(fields[0]) and _num(fields[1])
                and _num(fields[3])):
            func = " ".join(fields[5:])
            lines.append(f"cum {fields[3]}s self {fields[1]}s calls {fields[0]} :: {func}")
        if len(lines) >= top:
            break
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", default="suffix_hybrid.wrap_v2",
                        help="dotted path to a wrap_v2-like module exposing "
                             "_wrap_propose (live or snapshot package)")
    parser.add_argument("--sys-path", action="append", default=[],
                        help="extra sys.path entry (import root for --module); "
                             "repeatable")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--arm", choices=["warm", "cold", "both"],
                        default="both")
    parser.add_argument("--profile-steps", type=int, default=300,
                        help="cProfile window per arm (0 disables)")
    args = parser.parse_args(argv)

    for p in args.sys_path:
        if p not in sys.path:
            sys.path.insert(0, p)
    module = importlib.import_module(args.module)
    if not hasattr(module, "_wrap_propose"):
        parser.error(f"{args.module} has no _wrap_propose")

    out = {"module": args.module, "rows": args.rows,
           "context": args.context, "K": K, "arms": []}
    arms = ["warm", "cold"] if args.arm == "both" else [args.arm]
    for arm in arms:
        res = run_arm(module, args.rows, args.context, args.steps, arm)
        if args.profile_steps:
            res["profile_top"] = profile_arm(
                module, args.rows, args.context, args.profile_steps, arm)
        out["arms"].append(res)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
