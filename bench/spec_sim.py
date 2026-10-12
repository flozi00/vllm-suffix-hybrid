# SPDX-License-Identifier: Apache-2.0
"""Trace-driven accepted-tokens/step simulator for the hybrid mixer (CPU only).

Drives the REAL Rust HybridMixer through its list API the way the wrappers do
(verified contexts + native drafts + accepted-length feedback; departures
ingest into the shared corpus) with:

- ground-truth greedy token streams per corpus (below),
- a synthetic MTP drafter: position t is correct with conditional probability
  q[t] (higher on tokens that copy earlier text), and garbage once an earlier
  position was wrong -- its later tokens were conditioned on a wrong prefix,
- greedy verification: accept the longest matching draft prefix + 1 bonus.

Corpora (mirroring plugin-harness bench legs, token ids synthesized):
  unique        fresh chat, 5% short self-echo chunks
  reuse         6 fixed prompts cycled -> identical greedy completions
  self-repeat   "repeat the passage": completion echoes the prompt, 2% edits
  agentic       multi-turn session, each turn quotes 16-64 token spans of the
                history (tool output / file lines) 60% of the time

Absolute numbers depend on the drafter model; the arm-vs-arm gap on the same
seeded streams is the signal. Usage:
  python bench/spec_sim.py [--k 5] [--conc 1,8] [--requests 48] [--json]
"""
import argparse
import json
import os
import random
import time

V = 32000
Q_FRESH = (0.80, 0.72, 0.66, 0.60, 0.55, 0.50, 0.46, 0.42)
Q_COPY = (0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81)


def _fresh(rng, n):
    # Skewed unigram so short n-grams recur by chance, like real text.
    return [int(V * rng.random() ** 3) for _ in range(n)]


def corpus(name, n_req, seed=0, gen=256):
    """Yield (prompt, completion, copy_flags) per request, deterministic."""
    rng = random.Random(seed)
    if name == "unique":
        for _ in range(n_req):
            out, flags = [], []
            while len(out) < gen:
                if len(out) > 32 and rng.random() < 0.05:
                    s = rng.randrange(len(out) - 8)
                    out += out[s:s + 8]
                    flags += [True] * 8
                else:
                    out += _fresh(rng, 1)
                    flags.append(False)
            yield _fresh(rng, 64), out[:gen], flags[:gen]
    elif name == "reuse":
        templates = [(_fresh(rng, 64), _fresh(rng, gen)) for _ in range(6)]
        for i in range(n_req):
            p, c = templates[i % 6]
            yield list(p), list(c), [False] * gen
    elif name == "self-repeat":
        for _ in range(n_req):
            passage = _fresh(rng, 200)
            out = [t if rng.random() > 0.02 else _fresh(rng, 1)[0]
                   for t in (passage * 2)[:gen]]
            yield passage + _fresh(rng, 12), out, [True] * gen
    elif name == "agentic":
        sessions = max(1, n_req // 6)
        for _ in range(sessions):
            history = _fresh(rng, 400)
            for _turn in range(6):
                prompt = history + _fresh(rng, 100)       # new tool output
                out, flags = [], []
                while len(out) < gen:
                    if rng.random() < 0.6:
                        L = rng.randrange(16, 65)
                        s = rng.randrange(max(1, len(prompt) - L))
                        out += prompt[s:s + L]
                        flags += [True] * L
                    else:
                        L = rng.randrange(4, 24)
                        out += _fresh(rng, L)
                        flags += [False] * L
                out, flags = out[:gen], flags[:gen]
                yield prompt, out, flags
                history = prompt + out
    else:
        raise ValueError(name)


def native_draft(rng, truth, flags, k):
    d, ok = [], True
    for t in range(k):
        if t >= len(truth):
            d.append(_fresh(rng, 1)[0])
            continue
        q = (Q_COPY if flags[t] else Q_FRESH)[min(t, len(Q_FRESH) - 1)]
        if ok and rng.random() < q:
            d.append(truth[t])
        else:
            ok = False
            w = _fresh(rng, 1)[0]
            d.append(w if w != truth[t] else (w + 1) % V)
    return d


def run(corpus_name, arm, k=5, conc=8, n_req=48, seed=0):
    """Return dict(tokens_per_step, accepted_per_step, steps, mix_us)."""
    for key in ("SUFFIX_HYBRID_SPLIT",):
        os.environ.pop(key, None)
    if arm == "diverge":
        os.environ["SUFFIX_HYBRID_SPLIT"] = "diverge"
    os.environ.setdefault("SUFFIX_HYBRID_INDEX_N", "8")
    from suffix_hybrid._native import HybridMixer
    mixer = None if arm == "native" else HybridMixer(k, 1 << 20)
    rng = random.Random(seed + 1)
    queue = list(corpus(corpus_name, n_req, seed))
    active = {}      # rid -> [context, truth, flags, pos, last_accepted]
    next_id = 0
    row_steps = tokens = accepted_sum = 0
    mix_ns = calls = 0
    while queue or active:
        while queue and len(active) < conc:
            p, c, f = queue.pop(0)
            # Prefill emits the first completion token.
            active[f"r{next_id}"] = [p + c[:1], c, f, 1, -1]
            next_id += 1
        ids = list(active)
        natives = []
        for rid in ids:
            ctx, truth, flags, pos, _ = active[rid]
            natives.append(native_draft(rng, truth[pos:pos + k],
                                        flags[pos:pos + k], k))
        if mixer is None:
            drafts = natives
        else:
            t0 = time.perf_counter_ns()
            drafts = mixer.mix(ids, [active[r][0] for r in ids], natives,
                               [active[r][4] for r in ids])
            mix_ns += time.perf_counter_ns() - t0
            calls += 1
        for rid, d in zip(ids, drafts):
            st = active[rid]
            ctx, truth, _, pos, _ = st
            a = 0
            while a < len(d) and pos + a < len(truth) and d[a] == truth[pos + a]:
                a += 1
            emit = truth[pos:pos + a + 1]
            st[0] = ctx + emit
            st[3] = pos + len(emit)
            st[4] = a
            row_steps += 1
            tokens += len(emit)
            accepted_sum += a
            if st[3] >= len(truth):
                del active[rid]
    if mixer is not None:
        mixer.mix([], [], [], [])        # flush departures (API parity)
    return {"tokens_per_step": tokens / row_steps,
            "accepted_per_step": accepted_sum / row_steps,
            "row_steps": row_steps,
            "mix_us_per_call": mix_ns / 1e3 / max(calls, 1),
            "stats": ({key: v for key, v in mixer.get_stats().items()
                       if key in ("div_observed", "div_suffix_published",
                                  "self_hits", "suffix_proposed")}
                      if mixer is not None else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--conc", default="1,8")
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--corpora", default="unique,reuse,self-repeat,agentic")
    ap.add_argument("--arms", default="native,legacy,diverge")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = []
    for name in args.corpora.split(","):
        for c in map(int, args.conc.split(",")):
            for arm in args.arms.split(","):
                r = run(name, arm, args.k, c, args.requests)
                rows.append({"corpus": name, "conc": c, "arm": arm, **r})
                if not args.json:
                    print(f"{name:12s} c={c:<3d} {arm:8s} "
                          f"tok/step={r['tokens_per_step']:.3f} "
                          f"acc/step={r['accepted_per_step']:.3f} "
                          f"mix={r['mix_us_per_call']:.1f}us {r['stats']}")
    if args.json:
        print(json.dumps(rows))


if __name__ == "__main__":
    main()
