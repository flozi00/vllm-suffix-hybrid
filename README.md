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

## Benchmarking

`bench/compare.py` records streamed TTFT, request latency, per-request decode
throughput, aggregate throughput, output hashes, and raw metric snapshots.
Use identical prompts, concurrency, output budgets, sampling parameters,
model weights, node and **image digest** across baseline and plugin runs.

```sh
python bench/compare.py --endpoint https://YOUR_ENDPOINT/v1 --model YOUR_MODEL \
  --num-prompts 32 --concurrency 4 --max-tokens 512 --label rust-warm \
  --output results/rust-warm.json
```

Run baseline and plugin sequentially on the same dedicated GPU; do not change
production deployments. Warm each engine before timing. Repeated prompts
measure corpus reuse; a disjoint prompt-seed run is needed to assess
same-schema generalization. Server metric deltas are authoritative for draft
acceptance. Request-rate percentiles cannot establish GPU step time or CPU
proposer overhead; use direct instrumentation for that.

`bench/bench.py` is retained as the legacy non-streaming benchmark. Its reported
per-request tok/s includes prefill/network time and should not be interpreted
as pure decode speed.

## License

Apache-2.0. Independent implementation inspired by Snowflake's
[Suffix Decoding](https://arxiv.org/abs/2411.04975), not a vendored copy of
ArcticInference.
