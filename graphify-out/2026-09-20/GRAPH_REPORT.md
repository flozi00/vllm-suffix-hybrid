# Graph Report - vllm-suffix-hybrid  (2026-09-20)

## Corpus Check
- 21 files · ~8,990 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 192 nodes · 309 edges · 14 communities (10 shown, 4 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `970e24dd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_SuffixCache|SuffixCache]]
- [[_COMMUNITY_HybridProposer|HybridProposer]]
- [[_COMMUNITY_TestSuffixCache|TestSuffixCache]]
- [[_COMMUNITY_compare.py|compare.py]]
- [[_COMMUNITY_Engine|Engine]]
- [[_COMMUNITY_HybridMixer|HybridMixer]]
- [[_COMMUNITY_test_native_wrap.py|test_native_wrap.py]]
- [[_COMMUNITY_MixerTests|MixerTests]]
- [[_COMMUNITY_vllm-suffix-hybrid|vllm-suffix-hybrid]]
- [[_COMMUNITY_Native hybrid design|Native hybrid design]]
- [[_COMMUNITY_BundleTest|BundleTest]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_suffix-hybrid|suffix-hybrid]]

## God Nodes (most connected - your core abstractions)
1. `HybridProposer` - 19 edges
2. `TestHybridProposer` - 17 edges
3. `TestSuffixCache` - 16 edges
4. `config()` - 14 edges
5. `SuffixCache` - 13 edges
6. `HybridMixer` - 13 edges
7. `Engine` - 11 edges
8. `Cache` - 11 edges
9. `runner()` - 10 edges
10. `invoke()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `TestHybridProposer` --uses--> `HybridProposer`  [INFERRED]
  tests/test_hybrid_proposer.py → suffix_hybrid/hybrid_proposer.py
- `run_one()` --calls--> `make_prompt()`  [INFERRED]
  bench/compare.py → bench/bench.py
- `Engine` --references--> `SuffixCache`  [EXTRACTED]
  src/engine.rs → src/lib.rs
- `HybridMixer` --references--> `SuffixCache`  [EXTRACTED]
  src/mixer.rs → src/lib.rs

## Import Cycles
- None detected.

## Communities (14 total, 4 thin omitted)

### Community 0 - "SuffixCache"
Cohesion: 0.12
Nodes (17): Arc, Default, Mutex, PyModule, Cache, Config, env_float(), env_usize() (+9 more)

### Community 1 - "HybridProposer"
Cohesion: 0.18
Nodes (7): HybridProposer, install(), _verify_sources(), _wrap_propose(), config(), vLLM 0.29 custom_class contract: rows ALREADY include sampled IDs.  Legacy fake-, TestHybridProposer

### Community 2 - "TestSuffixCache"
Cohesion: 0.10
Nodes (4): NativeCacheTests, Native contract tests use real NumPy buffers, not fake tensor APIs., set_env(), TestSuffixCache

### Community 3 - "compare.py"
Cohesion: 0.12
Nodes (13): make_prompt(), run_one(), consume_stream(), main(), metric_snapshot(), Parse SSE data fields, including comments, CRLF and multiline events., run_one(), sse_events() (+5 more)

### Community 4 - "Engine"
Cohesion: 0.15
Nodes (11): ArrayView2, Engine, ngram(), Bound, PyAny, PyDict, PyResult, Python (+3 more)

### Community 5 - "HybridMixer"
Cohesion: 0.20
Nodes (12): Option, HybridMixer, Previous, Bound, HashMap, PyAny, PyDict, PyResult (+4 more)

### Community 6 - "test_native_wrap.py"
Cohesion: 0.28
Nodes (12): invoke(), Contract tests with real tensors; vLLM/CUDA execution is a deployment gate., runner(), test_feedback_uses_accepted_and_scheduled_counts_with_discard_censoring(), test_native_list_not_mutated_and_short_mixed_lists_not_padded(), test_native_probabilities_preserved_suffix_tail_is_one_hot(), test_native_runs_first_and_rust_mixes_with_authoritative_context(), test_qwen35_full_attention_mtp_allowed_with_hybrid_target() (+4 more)

### Community 8 - "vllm-suffix-hybrid"
Cohesion: 0.29
Nodes (6): Benchmarking, Build and deliver, Correctness and isolation, License, Usage, vllm-suffix-hybrid

### Community 9 - "Native hybrid design"
Cohesion: 0.33
Nodes (5): Components, Correctness boundaries, Learning the allocation, Native hybrid design, Required behavior

## Knowledge Gaps
- **10 isolated node(s):** `suffix-hybrid`, `Usage`, `Build and deliver`, `Correctness and isolation`, `Benchmarking` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SuffixCache` connect `SuffixCache` to `Engine`, `HybridMixer`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `HybridMixer` connect `HybridMixer` to `SuffixCache`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `Engine` connect `Engine` to `SuffixCache`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **What connects `Parse SSE data fields, including comments, CRLF and multiline events.`, `Synthetic local fixtures only; these are not benchmark results.`, `suffix-hybrid` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SuffixCache` be split into smaller, more focused modules?**
  _Cohesion score 0.11724137931034483 - nodes in this community are weakly interconnected._
- **Should `TestSuffixCache` be split into smaller, more focused modules?**
  _Cohesion score 0.09666666666666666 - nodes in this community are weakly interconnected._
- **Should `compare.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11857707509881422 - nodes in this community are weakly interconnected._