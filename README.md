# vllm-suffix-hybrid

Rust-native hybrid speculative decoding for vLLM: **the native model drafter
plus a conditional suffix tail in the same draft**. A learned token-budget
split starts near 80% native / 20% suffix and adjusts from verification feedback.
vLLM still verifies every proposal with the target model. **Better acceptance
does not automatically mean faster inference; benchmark the actual workload.**

The decoding algorithm, corpus/index maintenance, eviction and proposal
bookkeeping live in Rust. Python is only the adapter to vLLM's Python proposer
interface. An unavailable native extension is a startup error, not a silent
fallback to Python.

## Native model + suffix usage

Mount the extracted Linux runtime bundle at `/plugins`; it includes both the
native extension and `sitecustomize.py`. Keep vLLM's native MTP/drafter config:

```sh
PYTHONPATH=/plugins SUFFIX_HYBRID_WRAP=1 SUFFIX_HYBRID_LOG_INTERVAL=100 \
vllm serve YOUR_MODEL --speculative-config \
  '{"method":"mtp","num_speculative_tokens":5,"disable_padded_drafter_batch":true,"draft_sample_method":"greedy"}' \
  --override-generation-config '{"temperature":0}'
```

**Initial support is deliberately narrow:** exact vLLM `d05da62e9` source
contracts, synchronous unpadded scheduling, greedy requests, standard rejection,
no pipeline parallelism. Six source hashes reject unaudited revisions. The
adapter permits the inspected Eagle/MTP/DFlash class contracts; Qwen3.5 MTP's
full-attention draft layer is allowed with its hybrid target. This is not a
claim of live validation for every MTP/DSpark/DFlash checkpoint.

Do not send stochastic requests to this experimental deployment: unsupported
contracts raise rather than silently approximate, and a worker error can stop
the engine. Keep it isolated until additional paths are validated.

The Rust mixer retains a native prefix, conditions suffix lookup on that
prefix, and learns the split from verified acceptance. No suffix evidence →
native-only fallback. The full native draft is still computed, so this version
does not claim reduced native drafting compute. Telemetry's drafting wall time
is not full GPU decode-step latency.

## Auxiliary weight-free mode

With `SUFFIX_HYBRID_WRAP` unset, the native wheel also exposes the original
custom-class interface, now implemented in Rust:

```sh
vllm serve YOUR_MODEL --speculative-config \
  '{"method":"custom_class","model":"suffix_hybrid.hybrid_proposer.HybridProposer","num_speculative_tokens":4}'
```

This mode is suffix + in-request n-gram, **not** the model-drafter hybrid.
The vLLM positional contract is
`propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=...)`.
Its CPU row already contains sampled tokens; never append them a second time.

## Build and deliver

```sh
python -m pip install maturin
maturin build --release --locked
python -m pip install target/wheels/*.whl
```

GitHub Actions builds a manylinux x86_64 wheel and exercises the installed
extension. Its `runtime-linux-x86_64` artifact contains the importable package
and `BUILD.json` with source revision and file SHA-256 hashes. The public
`runtime-linux-x86_64` branch carries this bundle for deployments that fetch
files rather than install wheels. Deploy by **full runtime commit SHA** into a
**new, revision-specific S3 prefix**. Do not overwrite a shared plugin prefix:
the console's downloader currently skips same-size objects, not same-content
objects, and old files are not automatically removed.

This reuses the console's plugin-fetch init container; neither a new vLLM image
nor a compiler in serving containers is required. Native artifacts are
platform-specific. A macOS development build cannot be deployed to Linux.

## Correctness and isolation

- A corpus belongs to one engine process and is lost on restart.
- Scope instances by trust domain: do not share a cross-request corpus across
  unrelated tenants or models. Drafts still undergo target verification, but
  cached tokens are sensitive data held in process memory.
- Request row position is not an identity. Continuous batching can reorder or
  reuse rows; corpus updates must use authoritative token histories, not
  count-only synthetic request tracking.
- Corpus warming is not guaranteed to improve acceptance monotonically.
- No universal speedup or compatibility with every native model drafter is
  claimed.

## High-concurrency hot paths and crash safety

Every per-decode-step path is incremental: continuing rows append only their
new tokens (identity is gated by 64-token head/tail boundary windows, never a
full-context compare), the n-gram scan is one backward pass with no reversed
copy, and a strong suffix draft skips the n-gram scan entirely when no
n-gram can win its score ceiling. Measured on `bench/scale_bench.py`
(16 rows x 8k context): mixer 108 µs → 4.7 µs, engine 2319 µs → 118 µs per
step.

Knobs (all default to previous behavior):

- `SUFFIX_HYBRID_NGRAM_WINDOW` (default 0 = unbounded): caps the n-gram
  backward scan to matches ending within the last N tokens. At 128k context
  a window of 8192 bounds engine cost per step (~16x at 128k) at some draft
  quality for long-range repeats.
- `SUFFIX_HYBRID_USE_NGRAM=0`: suffix-only drafts.

Crash safety: every Rust hot path runs behind a panic boundary
(`catch_unwind` → `RuntimeError`) and poison-safe locks, so a Rust panic
surfaces as a catchable Python error and the wrapper degrades to native
drafts instead of escaping as pyo3's `PanicException` (a `BaseException`)
through `except Exception` and killing EngineCore. Drafts are advisory — the
target model verifies every token — so degrading is always safe; a lost draft
costs acceptance, a dead engine costs the pod.

## Benchmarking

Served-model speed, quality and A/B gating use the suite in
`inference-console/benchmarks` (its README documents tasks, scenarios and gate
rules). The old served-model scripts under `bench/` are removed. Gate a
candidate build against the reference build of the SAME model (same weights,
node and image digest) on an idle dev pool, never on production or `glm`, in
blocks A1 -> B -> A2, from the inference-console checkout:

```sh
uv run benchmarks/quality.py --url https://<pool>.pl-ai.net --arm A1             # quick tier
python3 benchmarks/speed.py  --url https://<pool>.pl-ai.net --arm A1 --rounds 5
# roll the candidate: same two commands with --arm B; roll back: --arm A2
python3 benchmarks/compare.py --a runs/<A1-quality> runs/<A2-quality> --b runs/<B-quality>
python3 benchmarks/compare.py --a runs/<A1-speed> runs/<A2-speed> --b runs/<B-speed>
```

`compare.py` ends with `VERDICT: PASS|FAIL|INCONCLUSIVE`. Request-level
numbers cannot establish GPU step time or CPU proposer overhead; for the
plugin's own hot paths use the in-process microbenchmarks in `bench/`
(`bench/README.md`).

## License

Apache-2.0. Independent implementation inspired by Snowflake's
[Suffix Decoding](https://arxiv.org/abs/2411.04975), not a vendored copy of
ArcticInference.
