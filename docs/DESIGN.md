# Native hybrid design

## Required behavior

Native model-based speculation and suffix lookup must contribute to **one
coherent draft**. A five-token budget starts with four native tokens and one
suffix token (80/20), not a choice between two entire independent drafts.

For a chosen native prefix length `n`:

1. Run the configured vLLM native drafter.
2. Preserve its first `n` tokens.
3. Query the Rust suffix corpus using `verified_context + native_prefix`.
4. Fill the tail with that conditional suffix continuation.
5. If suffix evidence is absent, retain the original native draft.
6. Never append original native-tail tokens after a changed suffix: those
   native tokens were conditioned on a different prefix.

The initial integration computes the full native draft before mixing. It does
**not** claim to save native model forward passes. Dynamic compute allocation
requires a deeper drafter-specific integration; current allocation concerns
the tokens sent to the target verifier.

## Components

- `src/lib.rs`: bounded suffix corpus, position index, matching and support.
  Eviction eagerly removes dead index entries and keys; both corpus sequence
  count and token count are bounded.
- `src/engine.rs`: weight-free custom-class engine (suffix + Rust n-gram).
  This auxiliary mode is not the native-model hybrid requirement.
- `src/mixer.rs`: `HybridMixer`, native-prefix/suffix-tail composition and
  discounted online split selection. Rust reads NumPy CPU buffers directly.
- `suffix_hybrid/wrap.py`: version-scoped vLLM integration glue. Its actual
  guards define the supported serving modes; do not generalize to untested
  runtime revisions or stochastic samplers.
- `suffix_hybrid/hybrid_proposer.py`: thin Python custom-class ABI adapter.
- `bench/compare.py`: measured streaming requests, usage, hashes and metrics.
- `scripts/runtime_bundle.py`: wheel extraction with revision/hash manifest.

## Learning the allocation

The learner explores different native-prefix lengths and updates each split's
reward from the verifier's actual accepted-prefix length. Discounting prevents
old traffic from permanently fixing the allocation. There is no guaranteed
monotonic improvement and no oracle for the better source before verification.

Source attribution counts the accepted prefix plus at most one rejected
position as tested. Positions after the first rejection are censored, not
failures. Missing or ambiguous feedback must be skipped. Scheduler truncation
must not be mistaken for rejection. A native-only fallback is not evidence
that an untested suffix split worked.

The default learning objective is verified acceptance per proposed slot.
An optional measured step-cost input supports cost-aware rewards. Acceptance
alone does not establish throughput; native-only and hybrid must be measured
on the same GPU, image digest, checkpoint, sampling and output budgets.

## Correctness boundaries

The old claim that arbitrary drafts always preserve stochastic correctness
merely because the target verifies them was too broad. A rejection sampler's
proposal probabilities must describe the actual proposal process; stateful
native drafter caches and scheduling metadata must remain aligned. The first
model-hybrid integration is deliberately scoped to contracts proven by source
inspection and tests. Unsupported paths must fail explicitly or bypass mixing
with visible telemetry, never silently claim hybrid participation.

vLLM 0.29 custom-class CPU token rows already contain the current sampled tokens.
Appending `sampled_token_ids` again corrupts corpus sequences. Row positions and
counts are not identities. Native hybrid mode should use the runner's stable
request IDs; the weight-free adapter validates complete authoritative prefixes
before reusing snapshots.

Corpora contain sensitive token histories. Scope deployments by trust domain;
do not share a corpus across unrelated tenants or models. Process-local corpus
and policy state reset on restart.
