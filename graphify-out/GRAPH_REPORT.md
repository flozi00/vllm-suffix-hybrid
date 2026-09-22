# Graph Report - .  (2026-09-22)

## Corpus Check
- 27 files · ~17,423 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 280 nodes · 468 edges · 20 communities (13 shown, 7 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Rust cache core (lib.rs)|Rust cache core (lib.rs)]]
- [[_COMMUNITY_vLLM wrap contract tests|vLLM wrap contract tests]]
- [[_COMMUNITY_Python adapter & integration glue|Python adapter & integration glue]]
- [[_COMMUNITY_Native cache contract tests|Native cache contract tests]]
- [[_COMMUNITY_Benchmarks (bench.py, compare.py)|Benchmarks (bench.py, compare.py)]]
- [[_COMMUNITY_HybridMixer incremental paths|HybridMixer incremental paths]]
- [[_COMMUNITY_Weight-free Engine (engine.rs)|Weight-free Engine (engine.rs)]]
- [[_COMMUNITY_Scale benchmark (scale_bench.py)|Scale benchmark (scale_bench.py)]]
- [[_COMMUNITY_Docs & CI workflow|Docs & CI workflow]]
- [[_COMMUNITY_wrap_v2 feedback tests|wrap_v2 feedback tests]]
- [[_COMMUNITY_Incremental NumPy path tests|Incremental NumPy path tests]]
- [[_COMMUNITY_Mixer behavior tests|Mixer behavior tests]]
- [[_COMMUNITY_Design rationale (DESIGN.md)|Design rationale (DESIGN.md)]]
- [[_COMMUNITY_Runtime bundle tests|Runtime bundle tests]]
- [[_COMMUNITY_Package init|Package init]]
- [[_COMMUNITY_Crate manifest|Crate manifest]]
- [[_COMMUNITY_Std HashMap type|Std HashMap type]]
- [[_COMMUNITY_Std HashMap type (dup)|Std HashMap type (dup)]]
- [[_COMMUNITY_Generic type param|Generic type param]]

## God Nodes (most connected - your core abstractions)
1. `invoke()` - 23 edges
2. `HybridProposer` - 19 edges
3. `runner()` - 19 edges
4. `TestHybridProposer` - 17 edges
5. `TestSuffixCache` - 16 edges
6. `SuffixCache` - 14 edges
7. `HybridMixer` - 14 edges
8. `config()` - 14 edges
9. `Cache` - 12 edges
10. `Engine` - 11 edges

## Surprising Connections (you probably didn't know these)
- `High-concurrency hot paths and crash safety` --references--> `ngram()`  [INFERRED]
  README.md → src/engine.rs
- `High-concurrency hot paths and crash safety` --references--> `lock_cache()`  [EXTRACTED]
  README.md → src/lib.rs
- `Native plugin CI workflow` --references--> `bundle()`  [EXTRACTED]
  .github/workflows/native.yml → scripts/runtime_bundle.py
- `High-concurrency hot paths and crash safety` --references--> `guard_py()`  [EXTRACTED]
  README.md → src/lib.rs
- `TestHybridProposer` --uses--> `HybridProposer`  [INFERRED]
  tests/test_hybrid_proposer.py → suffix_hybrid/hybrid_proposer.py

## Import Cycles
- None detected.

## Communities (20 total, 7 thin omitted)

### Community 0 - "Rust cache core (lib.rs)"
Cohesion: 0.09
Nodes (20): Arc, Default, Hasher, HashMap, K, Mutex, MutexGuard, Cache (+12 more)

### Community 1 - "vLLM wrap contract tests"
Cohesion: 0.11
Nodes (31): as_lists(), invoke(), Contract tests with real tensors; vLLM/CUDA execution is a deployment gate., Normalize the wrapper return: Tensor on the live path, list on the list path., Wrapper overhead on the live all-greedy path must not touch torch.      Monkeypa, SUFFIX_HYBRID_LOG_INTERVAL is read once, not once per propose (~300ns)., Regression: the scheduler verifying MORE slots than our last mixed     proposal, A mixer/adapter raise AFTER the native drafter ran must never kill     EngineCor (+23 more)

### Community 2 - "Python adapter & integration glue"
Cohesion: 0.14
Nodes (12): HybridProposer, _check_runner_capability(), _fingerprint_sources(), install(), Fail closed when the live runner lacks the hooked shape.      Raises RuntimeErro, Compare live sources against the audited pin; warn, never raise.      Returns th, Legacy name kept for tests/callers: log drift, do not raise., _verify_sources() (+4 more)

### Community 3 - "Native cache contract tests"
Cohesion: 0.09
Nodes (4): NativeCacheTests, Native contract tests use real NumPy buffers, not fake tensor APIs., set_env(), TestSuffixCache

### Community 4 - "Benchmarks (bench.py, compare.py)"
Cohesion: 0.12
Nodes (13): make_prompt(), run_one(), consume_stream(), main(), metric_snapshot(), Parse SSE data fields, including comments, CRLF and multiline events., run_one(), sse_events() (+5 more)

### Community 5 - "HybridMixer incremental paths"
Cohesion: 0.20
Nodes (15): Decide, Map, Option, boundary_matches(), Continuity, HybridMixer, Previous, Bound (+7 more)

### Community 6 - "Weight-free Engine (engine.rs)"
Cohesion: 0.15
Nodes (11): ArrayView2, Engine, ngram(), Bound, PyAny, PyDict, PyResult, Python (+3 more)

### Community 7 - "Scale benchmark (scale_bench.py)"
Cohesion: 0.19
Nodes (13): _batch(), bench_engine(), bench_mixer(), main(), A steady-state batch: rows requests, each with ctx_len tokens of history.      T, F, PyModule, High-concurrency hot paths and crash safety (+5 more)

### Community 8 - "Docs & CI workflow"
Cohesion: 0.17
Nodes (9): Native plugin CI workflow, Auxiliary weight-free mode, Benchmarking, Build and deliver, Correctness and isolation, License, Native model + suffix usage, vllm-suffix-hybrid (+1 more)

### Community 9 - "wrap_v2 feedback tests"
Cohesion: 0.27
Nodes (8): _wrap_propose(), fixture(), _mixer(), CPU contract tests: real Torch tensors and the compiled Rust mixer.  The vLLM ru, test_feedback_follows_request_ids_after_slot_reorder(), test_native_then_full_width_mix_stochastic_target(), test_truncated_success_and_zero_sample_are_censored(), TP

### Community 10 - "Incremental NumPy path tests"
Cohesion: 0.24
Nodes (3): IncrementalNumpyTests, Incremental mix_numpy: steady-state decode must match the list API.  The NumPy p, _tokens()

### Community 12 - "Design rationale (DESIGN.md)"
Cohesion: 0.33
Nodes (5): Components, Correctness boundaries, Learning the allocation, Native hybrid design, Required behavior

## Knowledge Gaps
- **8 isolated node(s):** `suffix-hybrid`, `Continuity`, `Native model + suffix usage`, `Auxiliary weight-free mode`, `Correctness and isolation` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SuffixCache` connect `Rust cache core (lib.rs)` to `HybridMixer incremental paths`, `Weight-free Engine (engine.rs)`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `High-concurrency hot paths and crash safety` connect `Scale benchmark (scale_bench.py)` to `Rust cache core (lib.rs)`, `Docs & CI workflow`, `HybridMixer incremental paths`, `Weight-free Engine (engine.rs)`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `HybridMixer` connect `HybridMixer incremental paths` to `Rust cache core (lib.rs)`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `invoke()` (e.g. with `fixture()` and `test_feedback_follows_request_ids_after_slot_reorder()`) actually correct?**
  _`invoke()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Parse SSE data fields, including comments, CRLF and multiline events.`, `A steady-state batch: rows requests, each with ctx_len tokens of history.      T`, `Synthetic local fixtures only; these are not benchmark results.` to the rest of the system?**
  _32 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Rust cache core (lib.rs)` be split into smaller, more focused modules?**
  _Cohesion score 0.09358974358974359 - nodes in this community are weakly interconnected._
- **Should `vLLM wrap contract tests` be split into smaller, more focused modules?**
  _Cohesion score 0.10661268556005399 - nodes in this community are weakly interconnected._