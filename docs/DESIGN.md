# Design notes

## Component map

- `suffix_hybrid/suffix_cache.py` — `SuffixCache`: corpus of finished request
  sequences + incremental n-gram position index. `speculate(context, max)`
  returns `(draft, score, match_len)`; the draft is the continuation of the
  longest back-extended match, length-capped by
  `factor*match_len + offset`, scored by the product of per-token empirical
  frequencies over the tied candidates.
- `suffix_hybrid/hybrid_proposer.py` — `HybridProposer`: the `custom_class`
  proposer. Feeds the cache (prompt on first sight, sampled ids each step,
  finalize-on-disappearance), computes suffix + native n-gram drafts per row,
  proposes the higher-scored one. Version-tolerant `propose(*args)` that
  introspects the input batch / sampled ids / token cap.
- `suffix_hybrid/wrap.py` + `sitecustomize.py` — opt-in (`SUFFIX_HYBRID_WRAP=1`)
  meta-path hook that subclasses eagle/dflash/draft-model proposers to add
  suffix arbitration on top of the native draft.
- `bench/bench.py` — schema-repeating load generator with tok/s percentiles.

## Why drafts compose safely

Every proposal — ours or the native drafter's — goes through vLLM's rejection
sampler: the target model scores the drafted tokens in one verification pass
and discards any that don't match its own distribution. Mixing sources per
step cannot change outputs, only how many tokens a single forward pass
retires. The failure mode is purely performance: bad drafts waste compute.
That is why arbitration is score-based and why wrap mode (which perturbs
eagle-family draft chains mid-flight) is opt-in.

## What "self-improving" means here

The corpus grows with every finished request. `index_n`-gram keys accumulate
positions; frequency counts sharpen over time, so the same schema match that
scored 0.6 on first sight scores higher after 50 repetitions, and the
draft-length heuristic (`factor*match_len`) lets long, well-supported matches
speculate deeper. A cold cache behaves like plain n-gram; a warm cache
approaches suffix-decoding acceptance rates from the literature (~2-3x on
repetitive agentic traffic).

## Known approximations (documented trade-offs)

- Acceptance telemetry compares previous-draft prefixes against the next
  step's sampled ids — a coincidental match counts as accepted. Directionally
  correct, cheap, no extra GPU work.
- Frequency counts are over the index bucket (capped at
  `MAX_POSITIONS_PER_NGRAM`), not the whole corpus — bounded work per step.
- The native n-gram draft is scored heuristically (`0.5 * len`), since the
  native proposer returns no match info; the suffix cache wins whenever it has
  real evidence, which is the intended bias.