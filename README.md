# vllm-suffix-hybrid

A pure-Python vLLM plugin that combines **cross-request suffix decoding** with
vLLM's **native speculative drafting** (n-gram today; eagle-family drafters via
opt-in wrap mode) — because vLLM's `speculative_config.method` is a single
choice: `suffix` XOR `dspark`/`dflash`/MTP/eagle. This plugin runs them
**together** and picks the better draft per request, per decode step.

## Why

- **Agentic workloads repeat schemas.** Tool-call envelopes, multi-record JSON,
  structured reports — the same token patterns recur across requests. A suffix
  cache that remembers *finished requests* (not just the current one) matches
  those patterns with high confidence.
- **Static draft models don't improve; suffix decoding does.** An MTP/eagle
  drafter is frozen at training time. The suffix cache's acceptance rate climbs
  as the corpus accumulates examples of your workload's patterns.
- **Native n-gram only sees the current request.** vLLM's `ngram` method
  matches within the request's own sequence (prompt + output). The hybrid
  reuses it as one signal and adds cross-request memory as the other; the
  higher-scored draft wins each step. Either way there is exactly one
  verification forward pass, so tokens/sec only goes up when drafts are good
  and never correctness-wise down.

## The two modes

### 1. `custom_class` hybrid (recommended, no draft model needed)

```
--speculative-config '{"method":"custom_class",
                       "model":"suffix_hybrid.hybrid_proposer.HybridProposer",
                       "num_speculative_tokens":8}'
```

Deliver the repo's files onto the pod's filesystem (we use an init container
that copies them from S3) with the directory on `PYTHONPATH`, and vLLM's
`custom_class` proposer loads `HybridProposer` at engine start. Per row and per
step it computes the suffix-cache draft and vLLM's native in-request n-gram
draft, then proposes the better one.

### 2. Wrap mode (experimental, opt-in) — combine with dspark / dflash / MTP / eagle

Keep the native method exactly as you run it today:

```
SUFFIX_HYBRID_WRAP=1  # plus PYTHONPATH pointing at this repo
--speculative-config '{"method":"dspark", ...}'
```

`sitecustomize.py` (auto-imported by CPython at interpreter start) then
subclasses the eagle/dflash/draft-model proposer classes so that each
`propose()` call runs the native drafter **and** the suffix cache, returning
the better draft. The native machinery (draft weights, KV bookkeeping,
rejection sampling) stays fully intact; our drafts are only proposals, so
outputs remain correct. Because replacing tokens in an eagle-family draft
chain can degrade the drafter's internal state on the *next* step (still
correct, occasionally slower), wrap mode is OFF by default.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SUFFIX_HYBRID_USE_NGRAM` | `1` | also run the native n-gram drafter in custom_class mode |
| `SUFFIX_HYBRID_NGRAM_MIN` / `_MAX` | `4` / `16` | n-gram sizes for the native drafter (custom_class zeroes `prompt_lookup_*`) |
| `SUFFIX_HYBRID_MAX_SPEC_TOKENS` | `0` (use config) | per-row draft cap override |
| `SUFFIX_HYBRID_MAX_CACHED_REQUESTS` | `512` | FIFO cap on finished sequences (`-1` unlimited) |
| `SUFFIX_HYBRID_INDEX_PROMPTS` | `0` | also index prompts, not just outputs |
| `SUFFIX_HYBRID_INDEX_N` | `8` | n-gram index key size |
| `SUFFIX_HYBRID_MAX_POSITIONS_PER_NGRAM` | `64` | positions kept per n-gram key |
| `SUFFIX_HYBRID_MAX_TREE_DEPTH` | `64` | trailing-context match window |
| `SUFFIX_HYBRID_MAX_SPEC_FACTOR` / `_OFFSET` | `1.0` / `0` | draft length ≤ factor×match_len + offset |
| `SUFFIX_HYBRID_MIN_TOKEN_COUNT` | `1` | min corpus support for a continuation |
| `SUFFIX_HYBRID_MAX_CANDIDATES` | `32` | candidate positions examined per step (latency guard) |
| `SUFFIX_HYBRID_TIME_BUDGET_MS` | `2.0` | soft per-request speculate budget |
| `SUFFIX_HYBRID_STATS_FILE` | unset | append JSONL acceptance telemetry every `_INTERVAL` calls |
| `SUFFIX_HYBRID_STATS_INTERVAL` | `100` | telemetry cadence (propose calls) |
| `SUFFIX_HYBRID_LOG_INTERVAL` | `1000` | log-line cadence |
| `SUFFIX_HYBRID_WRAP` | `0` | enable wrap mode (sitecustomize hook) |

## Telemetry

`HybridProposer.get_stats()` reports proposed/accepted tokens, suffix vs n-gram
win counts and average match length. With `SUFFIX_HYBRID_STATS_FILE=/tmp/suffix_stats.jsonl`
the engine appends `{ts, proposed, accepted, suffix_proposals, ngram_proposals,
avg_match_len}` lines — acceptance rate over time is the number to watch: it
should climb on schema-repeating traffic.

## Benchmarking

`bench/bench.py` drives an OpenAI-compatible endpoint with schema-repeating
agentic prompts (multi-record JSON of a fixed shape, varied content) and
reports p50/p95 tokens/sec:

```
python3 bench/bench.py --endpoint http://host:8000/v1 --model my-model \
    --num-prompts 64 --concurrency 8 --label baseline
```

Run it once against the vanilla deployment, then against the plugin-enabled
one; re-run after warm-up traffic to see the self-improvement effect.

## License

Apache-2.0. The suffix-matching approach follows Snowflake's Suffix Decoding
(https://arxiv.org/abs/2411.04975); this is an independent pure-Python
implementation with cross-request frequency arbitration added.