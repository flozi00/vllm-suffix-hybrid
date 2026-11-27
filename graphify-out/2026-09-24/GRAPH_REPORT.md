# Graph Report - vllm-suffix-hybrid  (2026-09-24)

## Corpus Check
- 61 files · ~112,645 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1200 nodes · 2229 edges · 56 communities (46 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8870d5af`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- [[_COMMUNITY_KernelTraits|KernelTraits]]
- [[_COMMUNITY_prefill.cuh|prefill.cuh]]
- [[_COMMUNITY_test_activation_gap.py|test_activation_gap.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_kernel_twin_batch|kernel_twin_batch]]
- [[_COMMUNITY_SharedStorageQKVO|SharedStorageQKVO]]
- [[_COMMUNITY_test_deep_gemm_shim.py|test_deep_gemm_shim.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_FlashInferBackend|FlashInferBackend]]
- [[_COMMUNITY_Tensor|Tensor]]
- [[_COMMUNITY_flashinfer_backend.py|flashinfer_backend.py]]
- [[_COMMUNITY_tvm_ffi_utils.h|tvm_ffi_utils.h]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_test_v2_suffix_proposer.py|test_v2_suffix_proposer.py]]
- [[_COMMUNITY__fi_fixture_files|_fi_fixture_files]]
- [[_COMMUNITY_test_sched_sync.py|test_sched_sync.py]]
- [[_COMMUNITY_FlashInferMetadataBuilder|FlashInferMetadataBuilder]]
- [[_COMMUNITY_uint_fastdiv|uint_fastdiv]]
- [[_COMMUNITY_test_late_arm_patches_immediately_loudly|test_late_arm_patches_immediately_loudly]]
- [[_COMMUNITY_test_hook_failure_fails_closed|test_hook_failure_fails_closed]]
- [[_COMMUNITY_.__init__|.__init__]]
- [[_COMMUNITY_BatchPrefillWithPagedKVCacheDevice|BatchPrefillWithPagedKVCacheDevice]]
- [[_COMMUNITY_dtype|dtype]]
- [[_COMMUNITY__make_xqa_draft_block_mask|_make_xqa_draft_block_mask]]
- [[_COMMUNITY_test_native_guards.py|test_native_guards.py]]
- [[_COMMUNITY_SinglePrefillWithKVCacheDispatched|SinglePrefillWithKVCacheDispatched]]
- [[_COMMUNITY_ab_glm.sh|ab_glm.sh]]
- [[_COMMUNITY_sm120 — SM120 (Blackwell consumerworkstation) support|sm120 — SM120 (Blackwell consumer/workstation) support]]
- [[_COMMUNITY_KVScaleFactorSmemDTypeKV, CTA_TILE_KV, HEAD_DIM_QK, HEAD_DIM_VO, true|KVScaleFactorSmem<DTypeKV, CTA_TILE_KV, HEAD_DIM_QK, HEAD_DIM_VO, true>]]
- [[_COMMUNITY_devsync.sh|devsync.sh]]
- [[_COMMUNITY_README|README.md]]
- [[_COMMUNITY_test_warmstart.py|test_warmstart.py]]
- [[_COMMUNITY_VerifyFusionTests|VerifyFusionTests]]
- [[_COMMUNITY_rejection_greedy_accept|rejection_greedy_accept]]

## God Nodes (most connected - your core abstractions)
1. `invoke()` - 50 edges
2. `paged_kv_t` - 42 edges
3. `KernelTraits` - 42 edges
4. `propose()` - 33 edges
5. `paged_kv_mla_t` - 32 edges
6. `_w0_runner()` - 27 edges
7. `TP` - 27 edges
8. `SharedStorageQKVO` - 24 edges
9. `wrap()` - 23 edges
10. `_pipe_wrap()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Native plugin CI workflow` --references--> `bundle()`  [EXTRACTED]
  .github/workflows/native.yml → scripts/runtime_bundle.py
- `test_bad_entries_rejected()` --calls--> `splice()`  [EXTRACTED]
  tests/test_dev_args.py → suffix_hybrid/dev_args.py
- `test_bare_flag_appends_once()` --calls--> `splice()`  [EXTRACTED]
  tests/test_dev_args.py → suffix_hybrid/dev_args.py
- `test_equals_form_parses_and_overrides()` --calls--> `splice()`  [EXTRACTED]
  tests/test_dev_args.py → suffix_hybrid/dev_args.py
- `test_new_flag_appends()` --calls--> `splice()`  [EXTRACTED]
  tests/test_dev_args.py → suffix_hybrid/dev_args.py

## Import Cycles
- 1-file cycle: `sm120/deep_gemm_shim/__init__.py -> sm120/deep_gemm_shim/__init__.py`

## Communities (56 total, 10 thin omitted)

### Community 0 - "Rust cache core (lib.rs)"
Cohesion: 0.05
Nodes (38): Arc, Default, F, Hasher, HashMap, K, Mutex, MutexGuard (+30 more)

### Community 1 - "vLLM wrap contract tests"
Cohesion: 0.05
Nodes (73): as_lists(), invoke(), Contract tests with real tensors; vLLM/CUDA execution is a deployment gate., Normalize the wrapper return: Tensor on the live path, list on the list path., Wrapper overhead on the live all-greedy path must not touch torch.      Monkeypa, SUFFIX_HYBRID_LOG_INTERVAL is read once, not once per propose (~300ns)., Regression: the scheduler verifying MORE slots than our last mixed     proposal, A mixer/adapter raise AFTER the native drafter ran must never kill     EngineCor (+65 more)

### Community 2 - "Python adapter & integration glue"
Cohesion: 0.16
Nodes (12): HybridProposer, _check_runner_capability(), _fingerprint_sources(), install(), Fail closed when the live runner lacks the hooked shape.      Raises RuntimeErro, Compare live sources against the audited pin; warn, never raise.      Returns th, Legacy name kept for tests/callers: log drift, do not raise., _verify_sources() (+4 more)

### Community 3 - "Native cache contract tests"
Cohesion: 0.09
Nodes (4): NativeCacheTests, Native contract tests use real NumPy buffers, not fake tensor APIs., set_env(), TestSuffixCache

### Community 4 - "Benchmarks (bench.py, compare.py)"
Cohesion: 0.15
Nodes (21): Decide, Fn, Map, Option, Py, boundary_matches(), Continuity, HybridMixer (+13 more)

### Community 5 - "HybridMixer incremental paths"
Cohesion: 0.05
Nodes (62): __global__, __host__, nv_bfloat16, nv_half, AppendPagedKVCache(), AppendPagedKVCacheDecode(), AppendPagedKVCacheDecodeKernel(), AppendPagedKVCacheKernel() (+54 more)

### Community 6 - "Weight-free Engine (engine.rs)"
Cohesion: 0.07
Nodes (33): _FastAPIRequest, attach_warmstart_route(), _armed(), Satisfies the EndpointPlugin protocol from     vllm/plugins/endpoint_plugins/int, Zero-arg factory the vllm.endpoint_plugins entry point resolves to     (see the, register(), _WarmstartEndpointPlugin, fill() (+25 more)

### Community 7 - "Scale benchmark (scale_bench.py)"
Cohesion: 0.67
Nodes (3): BaseStorage, SharedStorageWithRopeFreq, rope_freq_smem

### Community 8 - "Docs & CI workflow"
Cohesion: 0.15
Nodes (10): Native plugin CI workflow, Auxiliary weight-free mode, Benchmarking, Build and deliver, Correctness and isolation, High-concurrency hot paths and crash safety, License, Native model + suffix usage (+2 more)

### Community 9 - "wrap_v2 feedback tests"
Cohesion: 0.06
Nodes (40): RuntimeError, _describe(), _emit(), install(), Scheduler-side spec-decode probe for suffix-hybrid.  Wraps ``Scheduler.update_dr, Extract (n, widths, rid0) exactly as update_draft_token_ids     consumes the arg, Wrap ``Scheduler.update_draft_token_ids`` at class level (once).      ``engine_c, _trace_max() (+32 more)

### Community 10 - "Incremental NumPy path tests"
Cohesion: 0.24
Nodes (3): IncrementalNumpyTests, Incremental mix_numpy: steady-state decode must match the list API.  The NumPy p, _tokens()

### Community 12 - "Design rationale (DESIGN.md)"
Cohesion: 0.33
Nodes (5): Components, Correctness boundaries, Learning the allocation, Native hybrid design, Required behavior

### Community 16 - "Std HashMap type"
Cohesion: 0.05
Nodes (46): # NOTE: the WARM-START debug endpoint is NO LONGER armed here. The old, apply_dev_args(), [(flag, [flag, values...]) | (None, [positionals...])] preserving order., Mutate sys.argv in place. Returns the applied entries, or None (no-op)., splice(), _tokenize(), install_kernels(), Register kernel providers if gated. Returns a summary dict, or None     when the (+38 more)

### Community 17 - "Std HashMap type (dup)"
Cohesion: 0.13
Nodes (41): JitSpec, _append_nvfp4_sf_stride_setter(), _fa2_head_dim_nvcc_flags(), _fa2_prefill_head_dim_nvcc_flags(), gen_batch_attention_module(), gen_batch_decode_mla_module(), gen_batch_decode_module(), gen_batch_mla_module() (+33 more)

### Community 18 - "Generic type param"
Cohesion: 0.09
Nodes (65): _arm_env(), _early_env(), FakeTotals, _growth_scenario(), _growth_setup(), make_runner(), _pipe_env(), _pipe_wrap() (+57 more)

### Community 20 - "KernelTraits"
Cohesion: 0.06
Nodes (34): MaskMode, PosEncodingMode, KernelTraits, HEAD_DIM_VO_TILE, KV_THR_LAYOUT_COL, KV_THR_LAYOUT_ROW, MASK_MODE, NUM_D_VO_TILES (+26 more)

### Community 21 - "prefill.cuh"
Cohesion: 0.10
Nodes (35): DT, DTypeQKAccum, false_type, __grid_constant__, PagedKV, Params, compute_rope_freq(), __device__ (+27 more)

### Community 22 - "test_activation_gap.py"
Cohesion: 0.24
Nodes (11): _make_pkg(), Path, The crash-1 scenario end to end: armed hook + REAL import of the     backend mod, apply() declining while the host reports SM120 must kill the process     (armed-, FAKE_STACK up to the patch-module part; we script the tail per test., _run_stack(), _stack(), test_activation_fires_and_validation_gate_opens() (+3 more)

### Community 23 - "__init__.py"
Cohesion: 0.10
Nodes (28): allow_drift(), apply(), _capability(), ensure_deswizzle_flag(), _fail(), gate_enabled(), _hook_callback(), install_post_import_hook() (+20 more)

### Community 24 - "kernel_twin_batch"
Cohesion: 0.28
Nodes (7): GpuTwinAlgorithmTests, kernel_block_twin(), kernel_twin_batch(), Statement-for-statement transcription of the tile kernel's per-block     body in, Grid = batch: apply the block twin per request., Torch mirror of vLLM's greedy kernel semantics (:739-773): the     reference the, torch_reference_counts_mask()

### Community 25 - "SharedStorageQKVO"
Cohesion: 0.08
Nodes (25): DTypeKV, DTypeQ, float2, kEnableVOSplitOpt, CTA_TILE_KV, CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO (+17 more)

### Community 27 - "test_deep_gemm_shim.py"
Cohesion: 0.12
Nodes (24): build_paged_cache(), _fallback(), _make_inputs(), Simulate the bundle layout: <tmp>/deep_gemm/ (our shim, package) must     win ov, On a machine with neither deep_gemm nor vllm installed, importing the     shim m, # NOTE: triton is NOT import-skipped at module level: the shadowing tests, CPU-side: shape checks fire before any launch; degenerate windows are     repres, sap.py slices weights/ks/ke to the same token window as q: the shim     must key (+16 more)

### Community 28 - "__init__.py"
Cohesion: 0.09
Nodes (22): _announce(), _fallback_symbol(), _is_sm120(), _load_vendor(), _MissingVendor, Fallback-backed symbol, or a loud proxy to the vendor/missing path., Proxy for a DeepGEMM symbol with no vendor: resolves nothing, raises     a preci, One loud line, only inside a vLLM process (never for unrelated     importers, ne (+14 more)

### Community 29 - "FlashInferBackend"
Cohesion: 0.10
Nodes (7): AttentionBackend, CacheDType, DeviceCapability, KVCacheLayout, MultipleOf, FlashInferBackend, NVFP4 stores K and V as separate per-head slots of packed fp4 data         plus

### Community 30 - "Tensor"
Cohesion: 0.16
Nodes (9): AttentionImpl, AttentionType, BatchPrefillWithPagedKVCacheWrapper, Module, QuantKey, BatchDCPPrefillWrapper, FlashInferImpl, Tensor (+1 more)

### Community 31 - "flashinfer_backend.py"
Cohesion: 0.11
Nodes (19): Enum, _copy_page_indices_kernel(), FIDecode, FIPrefill, FlashInferDecodeKernel, FlashInferMetadata, FlashInferTrtllmAPIDecode, _get_trtllm_workspace_buffer() (+11 more)

### Community 32 - "tvm_ffi_utils.h"
Cohesion: 0.09
Nodes (33): DLDataType, DLDevice, Error, Result, Shape, alloc_tensor(), check_shape(), encode_dlpack_dtype() (+25 more)

### Community 33 - "__init__.py"
Cohesion: 0.13
Nodes (26): Loader, _cache_roots(), encode_dump(), _env_flag(), _harvest_loop(), install(), _log(), _patch_resolve() (+18 more)

### Community 34 - "test_v2_suffix_proposer.py"
Cohesion: 0.18
Nodes (16): _buf(), Regression (gemma wake #8): n>1 batches crashed the adapter with     'size of te, Regression (gemma wake #8): the adapter's exception path must be     able to ret, Uniform-k pallet (wake #8 counter-arm): the engine compiles its     uniform-deco, Uniform-k pallet partial hit: the stored corpus ends 2 tokens     beyond the rep, test_batch_dimensions_mismatch_raises(), test_clear_widths_retracts_all_published_widths(), test_echo_hit_drafts_exact_continuation() (+8 more)

### Community 35 - "_fi_fixture_files"
Cohesion: 0.67
Nodes (3): _fi_fixture_files(), test_header_probe_all_features_upstream_on_0_6_18(), test_header_probe_fails_on_missing_feature()

### Community 36 - "test_sched_sync.py"
Cohesion: 0.15
Nodes (6): _FakeSchedCfg, _FakeVllmCfg, _make_module(), test_operator_explicit_true_still_forced_off(), test_subclass_inherits_wrap(), test_wrap_forces_explicit_false_through_config()

### Community 37 - "FlashInferMetadataBuilder"
Cohesion: 0.22
Nodes (6): CommonAttentionMetadata, ndarray, FlashInferMetadataBuilder, Return the query width, ragged offsets, and effective query lengths., Return the packed XQA draft mask for speculative decode., Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer

### Community 38 - "uint_fastdiv"
Cohesion: 0.26
Nodes (15): compute_qk(), compute_qk_fp4_rope(), compute_sfm_v(), uint_fastdiv, k_smem_inplace_apply_rotary(), SWIZZLE_MODE_KV, SWIZZLE_MODE_Q, load_fp4_k_frag_scaled() (+7 more)

### Community 42 - ".__init__"
Cohesion: 0.20
Nodes (7): AttentionCGSupport, AttentionSpec, KVCacheSpec, _is_xqa_head_dim_supported(), Whether FlashInfer can serve attention sinks on this platform., Get the cudagraph support level for FlashInfer attention., VllmConfig

### Community 46 - "BatchPrefillWithPagedKVCacheDevice"
Cohesion: 0.39
Nodes (5): BatchPrefillWithPagedKVCacheDevice(), page_produce_kv(), page_produce_kv_sf(), produce_kv_sf(), SmemStorage

### Community 47 - "dtype"
Cohesion: 0.38
Nodes (4): fast_plan_decode(), dtype, A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for     cudagr, Plan the prefill operation with given parameters.

### Community 48 - "_make_xqa_draft_block_mask"
Cohesion: 0.33
Nodes (7): _make_xqa_draft_block_mask(), _make_xqa_ragged_draft_block_mask(), _pack_draft_block_bool_mask(), device, Pack a boolean draft mask into XQA's uint16 layout., Build a packed XQA draft mask for a uniform request., Build a packed XQA draft mask for a ragged batch.

### Community 49 - "test_native_guards.py"
Cohesion: 0.38
Nodes (3): _cache(), test_short_row_clamp_total_eq_n_ingests_nothing(), test_short_row_clamp_total_eq_n_plus_1_ingests_1_window()

### Community 51 - "SinglePrefillWithKVCacheDispatched"
Cohesion: 0.40
Nodes (6): cudaError_t, cudaStream_t, get_num_mma_q(), get_num_warps_kv(), get_num_warps_q(), SinglePrefillWithKVCacheDispatched()

### Community 57 - "KVScaleFactorSmem<DTypeKV, CTA_TILE_KV, HEAD_DIM_QK, HEAD_DIM_VO, true>"
Cohesion: 0.67
Nodes (3): KVScaleFactorSmem<DTypeKV, CTA_TILE_KV, HEAD_DIM_QK, HEAD_DIM_VO, true>, k_sf_smem, v_sf_smem

### Community 62 - "test_warmstart.py"
Cohesion: 0.15
Nodes (14): _load_bench(), _propose(), Pure-logic assertions on the route handler without fastapi installed:     - reje, EPP gateway carrier keys: the POST body must carry "model" + "prompt"     alongs, test_bench_dump_histories_format(), test_bench_hist_reader_and_flags(), test_bench_hist_reader_rejects_garbage(), test_bench_payload_gains_return_token_ids_only_when_dumping() (+6 more)

### Community 63 - "VerifyFusionTests"
Cohesion: 0.29
Nodes (5): emit_tokens_from_rust(), vLLM greedy verify semantics, in torch, on CPU.      draft_token_ids: [num_token, Reconstruct vLLM output_token_ids from the fused op's (count, mask).      Where, torch_reference(), VerifyFusionTests

### Community 64 - "rejection_greedy_accept"
Cohesion: 0.25
Nodes (7): PyArray1, PyArray2, rejection_greedy_accept(), Bound, PyAny, PyResult, Python

## Knowledge Gaps
- **94 isolated node(s):** `suffix-hybrid`, `devsync.sh script`, `page_size`, `num_heads`, `head_dim` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_probe_worker()` connect `Weight-free Engine (engine.rs)` to `wrap_v2 feedback tests`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `KernelTraits` connect `KernelTraits` to `SharedStorageQKVO`, `prefill.cuh`, `uint_fastdiv`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `invoke()` (e.g. with `fixture()` and `test_feedback_follows_request_ids_after_slot_reorder()`) actually correct?**
  _`invoke()` has 31 INFERRED edges - model-reasoned connections that need verification._
- **What connects `suffix-hybrid`, `devsync.sh script`, `# NOTE: the WARM-START debug endpoint is NO LONGER armed here. The old` to the rest of the system?**
  _238 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Rust cache core (lib.rs)` be split into smaller, more focused modules?**
  _Cohesion score 0.0504828797190518 - nodes in this community are weakly interconnected._
- **Should `vLLM wrap contract tests` be split into smaller, more focused modules?**
  _Cohesion score 0.054945054945054944 - nodes in this community are weakly interconnected._
- **Should `Native cache contract tests` be split into smaller, more focused modules?**
  _Cohesion score 0.09230769230769231 - nodes in this community are weakly interconnected._