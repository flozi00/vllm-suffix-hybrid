# DeepSeek-V4.1 one-GPU per-rank slice (`dsv41_slice`)

Lane P of the DSv4.1 speed program. Tool: `suffix_hybrid/tools/dsv41_slice.py` (stdlib at import; torch and vLLM only
inside the run). Tests: `tests/test_dsv41_slice.py`. **Written blind**: no GPU, no vLLM on the author's machine, no
cluster. Everything vLLM-side was read from the source at the pinned commit
`39983a5e7a5a575a7829f5d5ada50a33d9337ec1` (0.32.0rc1.dev329), not executed. Section 8 says exactly what is proven.

## 1. What it is

Prod `glm` serves DeepSeek-V4.1-Flash-1017-NVFP4 as TP8 + expert-parallel 8 on 8 RTX PRO 6000. Every kernel of one rank
sees only that rank's shapes, so one GPU can stand in for one rank. The tool writes a **dummy-weight model dir** whose
`config.json` holds one rank's dimensions at the **full 40-layer depth**, boots vLLM on it with the prod serving flags and the
DSpark speculative config, and measures decode step time (and, with `profile=true`, where the time goes) under whatever
serving options you pass. Nothing needs the cluster, the checkpoint or the 17-minute prod downtime. Collectives do not exist
in the slice; everything else does.

Why not the generic rank-slice profiler (`api/autotune_runner.py`): it does not shrink the Engram tables (384M rows x 2 layers,
~98 GB pinned at world size 1: the job is OOM-killed), it extrapolates from a layer prefix that has none of the compressed or
indexer layers, and its rank cut is not TP-aware for the shared expert, `o_groups`, the vocabulary or the Engram head shards.

## 2. Run it

```
# on any machine, no vLLM: build the config, print the plan, the diff, the estimates
python3 -m suffix_hybrid.tools.dsv41_slice --dry --opts '{}' --pretty
python3 -m suffix_hybrid.tools.dsv41_slice --print-schema

# in a one-GPU kernel-lab pod: vLLM nightly-39983a5e image, plugin bundle on PYTHONPATH (as in the pool spec)
python3 -m suffix_hybrid.tools.dsv41_slice --opts '{"conc":[1,8,32]}' --out /tmp/slice.json --log-file /tmp/slice.log
```

The module is standalone (no package-relative imports), so `python3 /path/to/dsv41_slice.py ...` works too. stdout is one
JSON object, logs go to stderr (the tool moves fd 1 and 2 into a pipe, so library prints and child processes cannot corrupt the
JSON). `--out` is rewritten after every finished concurrency (a SIGKILL keeps them); `--keep-dir` keeps the temp model dir.
Delete the lab pod when you are done (AGENTS section 3).

The plugin reads `SUFFIX_*` when the interpreter starts (`sitecustomize.py`), so the tool computes the full env and
**re-execs itself once** with it (`DSV41_SLICE_ENV_APPLIED=1` marks the second run). Default env mirrors `d7-spec.json`:
`VLLM_USE_V2_MODEL_RUNNER=1`, `SUFFIX_PRESET=dsv41-sm120` (Engram MXFP4 overlay + block-64 hook; the `-comms` preset only adds
host-all-reduce knobs that have nothing to do at TP=1), `MALLOC_ARENA_MAX=16`, `MALLOC_MMAP_THRESHOLD_=131072`, plus
`HF_HUB_OFFLINE=1`. Fixed by the tool: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process engine: the profiler and the
shared-expert patch need it), `VLLM_LOGGING_STREAM=ext://sys.stderr` (vLLM logs to stdout by default).

Exit codes: 0 ok; 2 bad opts; 3 the engine refused the configuration (`engine_error` has vLLM's text verbatim); 4 deadline hit
(partial JSON); 5 a measurement failed; 6 a pinned vLLM source anchor drifted; 7 environment (no GPU, import failure); 1 other.

Suggested first runs (each is one boot; `conc` and `profile` share it):

```
{}                                                       # calibration: compare ratio_to_expected (section 7)
{"profile": true}                                        # + kernel table per concurrency
{"k": 0}                                                 # plain decode: what speculation costs and buys
{"k": 3, "synthetic": "chat"}                            # k sweep (k=6..8 needs an explicit rate list)
{"synthetic": "thinking"}  /  {"synthetic": "rag8k"}     # acceptance mix
{"enable_adaptive_verification": true}                   # the error text if the backend cannot do it is in engine_error
{"num_speculative_tokens_per_batch_size": [[1,8,5],[9,32,3]]}
{"draft_sample_method": "probabilistic", "rejection_sample_method": "block"}
{"cudagraph_mode": "FULL_DECODE_ONLY"}   {"moe_backend": "flashinfer_cutedsl"}   {"sparse_indexer_topk_backend": "per_row"}
{"env": {"SUFFIX_DSV41_MOE_FUSE_SHARED": "1"}}           # any allowlisted plugin / VLLM knob
{"prompt_tokens": 32768, "conc": [1, 8]}                 # context depth
{"moe_routing": "normal_routing"}                        # skewed expert load (uniform_random = balanced)
```

## 3. How the rank is built

Every row was read from the vLLM source named in the last column. `slice config` is what `config.json` carries; all other
`text_config` keys, `vision_config`, `quantization_config` (verbatim: it decides NVFP4 experts via ModelOpt, MXFP8 dense via
`deepseek_v4_fp8`, MXFP4 Engram tables, the ignore list) and the top-level keys are unchanged. The tool prints the exact diff
(`plan.config_diff`) and a reason per key (`plan.rules`) and logs both to stderr.

| dimension | prod rank (TP8, EP8) | slice config (TP=1) | vLLM source |
|---|---|---|---|
| attention heads | 64 -> 8 local | `num_attention_heads=8` | `deepseek_v41/attention.py`: `n_local_heads = heads // tp`, `wq_b` ColumnParallel over `heads*head_dim` |
| output groups | `o_groups` 8 -> 1 local | `o_groups=1` | `wo_a` ColumnParallel out `o_groups*o_lora/tp`, `wo_b` RowParallel in; `n_local_groups = o_groups // tp`. Gives wo_a [4096->1024], wo_b [1024->5120] |
| KV | 1 head, replicated; SWA 128 | unchanged | MLA-like; the KV cache is not sharded |
| `fused_wqa_wkv`, indexer (8 layers), compressor (4), mHC `hc_*_fn`, norms | replicated | unchanged | `disable_tp=True` / `ReplicatedLinear` |
| routed experts | 384 -> 48 whole experts at I=2304 | `n_routed_experts=48`, `num_experts_per_tok` 6 -> `moe_topk` (1) | `FusedMoE` with EP; the top-k choice is the one approximation here (section 6) |
| shared expert | 2304 -> 288 per rank (`cdiv(72 blocks, 8) x 32`) | patched at runtime to 1/8 width | `DeepseekV4MLP` has no independent width knob (`moe_intermediate_size * n_shared_experts`) |
| gate | replicated [5120 -> 384] | [5120 -> 48] | `GateLinear`; non-equivalence |
| vocabulary | `pad64(129280)/8 = 16160` rows per rank | `vocab_size=16160` | `VocabParallelEmbedding` / `ParallelLMHead`; non-equivalence for the sampler |
| DSpark draft | 3 MoE layers, 128 experts top-3, EP8 | `dspark_n_routed_experts=16`, `dspark_num_experts_per_tok=1`, `dspark_noise_token_id` 128799 -> 15679 | `deepseek_v41/nvidia/dspark.py`; draft layers are ordinary decoder layers with ids 40..42 |
| Markov head | replicated 129280 x 256 | 16160 x 256 | `DSparkMarkovHead` (replicated by design); non-equivalence |
| Engram hash columns | 24 (3 n-gram sizes x 8 heads), sharded over TP: **3 per rank** | `engram_n_heads=1` (3 columns) | `ParallelEngramEmbedding`: `part_n_hash_cols = cdiv(n_cols, tp)`. The rank's `wkv` [6144 -> 3200] and the slice's [768 -> 25600] are the same weight bytes |
| Engram tables | 48.0M rows per rank and layer, MXFP4 rows of 136 B = 6.08 GiB, pinned host memory (`cpu_offload=True`) | 3 columns x `engram_rows` (default 1,048,576): 0.4 GiB per layer, still pinned host memory | `EngramLayout`: a table has `sum(primes)` rows, so `engram_vocab_size`, `engram_num_embeddings` (recomputed with vLLM's own prime draw; the tests reproduce the real 384006168 / 384016682 and the logged 48000217 / 48001463) move together |
| Engram tokenizer | real tokenizer, compressed vocab 99092 | a WordLevel tokenizer of 16160 distinct tokens, `engram_compressed_vocab_size=16160` | `NgramHashState` loads a tokenizer from the model dir even with `skip_tokenizer_init` and refuses a size mismatch |
| depth | 40 layers + 3 draft | unchanged: `compress_ratios` (43), `kv_source_layer_ids`, `index_source_layer_ids`, `engram_layer_ids`, `dspark_target_layer_ids` | |
| vision tower | resident (data-parallel mm encoder) | not built: `limit_mm_per_prompt={"image": 0}` | the MM processor needs the real tokenizer's `<image>` token; AGENTS rule 1.5 is about serving pools |

Serving flags (`plan.engine_kwargs`, mirroring the `non-default args` line of the prod boot): `kv_cache_dtype=fp8`,
`block_size=64`, `max_model_len=524288`, `max_num_seqs=32`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.93`,
`enable_prefix_caching=True`, V2 runner, `skip_tokenizer_init=True`, cudagraph mode and compilation config left to the
architecture (prod: `FULL_AND_PIECEWISE`, mode NONE), `speculative_config={"method": "dspark", "num_speculative_tokens": k, ...}`
with only the non-default keys, `disable_log_stats=False` (the spec-decode counters).

Weight check: the model of what one rank holds (`rank_modules`) gives 38.70 GiB for the prod rank (vLLM logged 40.66 GiB; the gap
is the vision tower, rope caches and buffers) and 38.46 GiB for the slice. The slice's own `Model loading took` line should read
about 40.66 - (vision 0.6-1.1) - 0.24 = **39.3-39.8 GiB**. A larger number means a dimension is wrong.

## 4. Options (`--opts` JSON; unknown keys, wrong types and out-of-range values are refused with exit 2)

| key | domain | default | meaning |
|---|---|---|---|
| `conc` | list of 1-8 ints in [1, 256] | `[1, 8, 32]` | closed-loop stream counts, each <= `max_num_seqs` |
| `prompt_tokens` | int [16, 262144] | `2048` | prompt length per request (random token ids) so the KV is realistic |
| `warmup_steps` | int [0, 1000] | `16` | pure-decode engine steps after all streams started, excluded |
| `measure_steps` | int [4, 4000] | `64` | engine steps timed per concurrency |
| `temperature` | float [0, 2] | `0.2` | sampling temperature (the prod override) |
| `profile` / `profile_steps` | bool / int [1, 64] | `false` / `8` | torch.profiler kernel table over `profile_steps` more steps per concurrency |
| `k` | int [0, 8] | `5` | `num_speculative_tokens`; 0 = no speculation |
| `draft_sample_method` | greedy, probabilistic | `greedy` | |
| `rejection_sample_method` | standard, synthetic, block, or null | null = `synthetic` when k>0 | dummy-weight drafts accept ~nothing under `standard` |
| `synthetic` | chat, thinking, edit, rag8k, or k floats | null = `chat` | cumulative P(first i+1 drafts accepted): chat [.61 .34 .17 .09 .05], thinking [.86 .71 .58 .46 .375], edit [.89 .73 .53 .33 .19], rag8k [.67 .44 .26 .14 .08] (live glm, 2026-10-19) |
| `synthetic_acceptance_length` | float [1, k+1] or null | null | mean accepted length instead of a rate list (exclusive with `synthetic`) |
| `enable_adaptive_verification` | bool | `false` | the confidence head is random here (section 6) |
| `num_speculative_tokens_per_batch_size` | `[[start,end,k],...]` or null | null | validated like vLLM: first start 1, no overlap, k <= `k` |
| `gpu_memory_utilization` | float [0.3, 0.98] | `0.93` | |
| `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` | int | `524288` / `32` / `8192` | |
| `block_size` | 32, 64 | `64` | 64 needs the plugin's block-64 hook |
| `kv_cache_dtype` | fp8, auto | `fp8` | |
| `enable_prefix_caching` | bool | `true` | |
| `cudagraph_mode` | NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE, or null | null | null = what the architecture picks |
| `async_scheduling`, `enable_flashinfer_autotune` | bool or null | null | null = vLLM decides |
| `moe_backend`, `linear_backend`, `sparse_indexer_topk_backend` | vLLM `--moe-backend` etc. (enum) | `auto` | prod: `auto` -> FLASHINFER_CUTLASS NVFP4 |
| `moe_routing` | model, uniform_random, normal_routing | `model` | model = the dummy-weight router (prod kernels). The other two are vLLM's routing simulator: balanced / skewed, and they add rand + top-k kernels |
| `moe_topk` | int [1, 6] | `1` | routed experts per token in the slice; raise it only if a backend refuses top-1 |
| `engram_heads` | 1, 8 | `1` | 1 = the per-rank 3 columns; 8 = all 24 columns (8x lookup rows, 8x `wkv` bytes). Use 8 if a kernel keyed to 24 columns refuses 3 |
| `engram_rows` | int [1024, 16,000,000] | `1048576` | rows per hash column; refused if the pinned tables would exceed 32 GiB |
| `shared_expert_tp` | bool | `true` | false = full 2304-wide shared expert, +0.98 ms/step |
| `seed`, `deadline_s`, `work_dir`, `log_level` | | `0`, `7200`, tmp, `INFO` | `deadline_s`: partial JSON + exit 4 (a watchdog fires at +30 s even if vLLM is stuck in init) |
| `env` | `{KEY: "value"}` | `{}` | allowlisted keys only (below); values limited to `[A-Za-z0-9_.,:/*+@~=-]` |

Env allowlist (`ALLOWED_SUFFIX_ENV`, `ALLOWED_VLLM_ENV`; `--print-schema` lists them): the V4.1 / SM120 / Helion / NVFP4 /
FP8 / lm_head / indexer knobs of plugin `af97edb7`, and the `VLLM_*` names that exist in `vllm/envs.py` at the pinned commit and
are safe to flip (cudagraph, shared-expert and multi-stream thresholds, DeepGEMM, indexer logits, logging, profiling scopes).
Refused on purpose: `SUFFIX_ROCM_*` (AMD), `SUFFIX_NCCL_*` / `SUFFIX_HAR_*` (no collectives at TP=1), `SUFFIX_MTP_TUNE*`
(online weight tuning), `SUFFIX_HYBRID_*` (suffix-decoding wrap), marker flags, `PYTHONPATH`, and the tool-owned variables (`VLLM_ENABLE_V1_MULTIPROCESSING`, `VLLM_LOGGING_STREAM`, `VLLM_MOE_ROUTING_SIMULATION_STRATEGY`, `CUDA_VISIBLE_DEVICES`).
To allow another plugin knob, add it to the frozenset.

## 5. Output (stdout, one JSON object, schema `dsv41-slice/1`; ms are milliseconds)

```
schema, ok, exit, partial, dry, deadline_hit?, elapsed_s, warnings[str]
engine_error        null | "ExcType: message" exactly as vLLM raised it;  engine_traceback  full traceback
error               null | {phase, type, message, traceback?}
phases              {import_vllm, engine_init, measure_c<N>, build_config: seconds}
opts, env_applied, model_dir
plan                {engine_kwargs, env, rules, config_diff, model_dir_files{sha256}, weights_gib_per_rank, rank_diff_mb,
                     step_stream_gb{c:{prod_rank, slice}}, engram_host_gib, non_equivalences[], bias_terms_ms{c:{...}},
                     live_reference_ms, expected_slice_ms, slice_vocab}
versions            {python, torch, vllm, triton, flashinfer, transformers, tokenizers, tilelang, helion, cuda, gpu{name,cc,mem_gib,sms}, argv}
gpu_nvml            {name, power_limit_w, max_sm_mhz, max_mem_mhz, driver}
engine              {log_checks{name:{found, expected_prod, match}}, num_gpu_blocks, gpu_mem_free_gib, gpu_mem_total_gib}
results.<c>         c, max_tokens, prefill_s, steps, wall_s, tokens_emitted, tokens_per_stream_per_step, tok_s_per_stream, tok_s_total,
                    step_ms{mean, p50, p90, p99, min, max},
                    spec{drafts, draft_tokens, accepted, per_pos_accepted, per_pos_cum_accept, accepted_len,
                         verified_per_stream, verified_tokens_per_step, acceptance_rate},
                    telemetry{samples, sm_mhz_mean, sm_mhz_min, power_w_mean, power_w_max, temp_c_max, throttle_reasons_mask},
                    slice_step_ms, prod_equivalent_ms_est, live_step_ms, expected_slice_ms, ratio_to_expected,
                    synthetic_check{target, max_abs_z},
                    profile{steps, kernels_per_step, kernel_us_per_step, span_us_per_step, wall_ms_per_step_profiled,
                            gpu_busy_fraction, families[{family, us_per_step, calls_per_step, share}],
                            top_kernels[40]{name, family, calls_per_step, us_per_step, share}, unclassified_top[15], warning?}
log_tail            last 6000 characters of the combined log
```

How a step is timed: all `c` requests are added with `ignore_eos` and a `max_tokens` large enough that none finishes
(`(warmup + measure + profile + 8) * (k+1) + 1`); the loop steps `LLMEngine` until all `c` have produced their first token
(prefill, mixed steps), runs `warmup_steps` pure decode steps, then times `measure_steps` steps. `step_ms.mean` is wall time of the
window divided by the number of steps that produced output (an async pipeline-fill call that returns nothing is folded into the next
gap). `spec.*` are deltas of vLLM's `spec_decode_*` counters over the same window: `verified_per_stream = 1 + draft_tokens / drafts`,
`accepted_len = 1 + accepted / drafts`, `verified_tokens_per_step = c * verified_per_stream` (6 / 48 / 192 in the live config).
`per_pos_cum_accept` must reproduce the synthetic target; `synthetic_check.max_abs_z` and a warning say if it does not.

Kernel table: `torch.profiler` (CUDA activity only) over `profile_steps` steps, run after the timed window so it never disturbs
`step_ms`. Shares are of summed kernel time; `span_us_per_step` is the union of kernel intervals (kernels on side streams
overlap), `gpu_busy_fraction = span / profiled wall`, so `1 - gpu_busy_fraction` is launch/CPU/sync gap. Families
(`KERNEL_FAMILIES`, ordered regexes on the lower-cased kernel name): memcpy, engram, mhc, indexer, compressor, draft, sampler,
moe, attention, norm_quant, dense_gemm, elementwise, other. **The patterns are a heuristic written without seeing a V4.1 kernel
name**; `unclassified_top` lists what stayed in `other`, send it back and the regexes get fixed. The draft backbone runs the same
kernels as the target, so the `draft` family only holds the Markov/draft-sampling kernels: get the draft's cost from `k=0` versus
`k=5` instead. If the profiler sees no kernels (CUPTI and graph replays), rerun with `cudagraph_mode=NONE`.

## 6. Where the slice is NOT per-rank-equivalent

`slice - prod`, ms per decode step, default options, k=5 (the tool recomputes this for your options: `plan.non_equivalences`,
`plan.bias_terms_ms`). "computed" = from the byte model at an assumed 1400 GB/s effective bandwidth (the figure of
`docs/dsv41-comms.md`; the 1.79 TB/s peak would shrink the byte-derived terms by 22%), "estimate" = from a cited table.

| effect | prod rank | slice | c=1 | c=8 | c=32 | basis |
|---|---|---|---|---|---|---|
| collectives | ~88 per step (2 per layer x 43, embedding, vocab-logits gather, Engram head gathers) | none | -1.75 | -7.00 | -25.10 | estimate: `dsv41-comms.md` section 3 host all-reduce table (1.7 / 6.6 / 23.5 ms) + logits gather (0.05 / 0.4 / 1.6) |
| routed MoE pairs | top-6 of 384, 48 local: 0.75 pairs/token, E[distinct local experts] = 48(1-(63/64)^M) = 4.33 / 25.5 / 45.7 | top-1 of 48: 1.0 pair/token, E = 48(1-(47/48)^M) = 5.70 / 30.5 / 47.2 | +0.78 | +2.88 | +0.85 | computed (19.9 MB per expert x 40 layers; matches the comms doc's 86 / 507 / 909 MB); balanced routing assumed |
| DSpark MoE pairs | top-3 of 128, 16 local | top-1 of 16 | +0.11 | +0.20 | +0.01 | computed |
| shared expert | 288 wide | patched to 288 (exact) | 0 | 0 | 0 | with `shared_expert_tp=false`: +0.98 at every c |
| gate GEMM width | 384 (draft 128) | 48 (draft 16) | -0.10 | -0.10 | -0.10 | computed (router top-k kernel width and EP filtering of non-local pairs, prod-only, are not included: < ~0.3 ms) |
| Markov head | 129280 x 256, read k times | 16160 x 256 | -0.21 | -0.21 | -0.21 | computed |
| sampler / rejection vocabulary | 129280 columns | 16160 | -0.01 | -0.11 | -0.46 | estimate: 4 fp32 passes over [tokens, vocab] |
| lm_head / embedding padding | 16160 rows | 16192 | +0.001 | +0.001 | +0.001 | computed |
| **quantified total** | | | **-1.19** | **-4.34** | **-25.0** | |

Not quantified: Engram (3 hash columns instead of 24 hashed, lookups from a 0.4 GiB instead of a 6.08 GiB pinned table; the volume is
< 0.2 MB per step, so <= ~0.1 ms either way, but the lab host's NUMA and PCIe differ from worker-06); the vision tower is absent
(memory only: the KV pool is 0.6-1.1 GiB larger, decode never runs the encoder); the engine is in-process (scheduler and worker
share a thread; prod overlaps them with async scheduling, so the slice can be slower by up to the scheduler's CPU time per step when
the GPU would otherwise wait); dummy weights (routing is ~uniform, activations tiny, E8M0 scales constant: expect a few percent
either way from power/clock data dependence; real routers are skewed, i.e. fewer distinct experts, which makes the slice's MoE
bytes an over-count; `moe_routing=normal_routing` brackets it); the lab card may be the Max-Q (same GDDR7 bandwidth, lower power
cap: bandwidth-bound decode is unaffected, compute-bound parts such as c=32 GEMMs and attention can be slower; `gpu_nvml.power_limit_w`
and `telemetry.sm_mhz_*` show it); context depth is a parameter (`prompt_tokens`), `SPEED.md` measured flat decode speed from 0 to
500k at c=1; `enable_adaptive_verification` runs on a random confidence head (about 0.5 everywhere), so only its cost side is
meaningful; synthetic acceptance matches the marginals, not the content correlation.

## 7. Calibrate before trusting it

`expected_slice_ms = live step + quantified bias`: **11.0 / 21.6 / 30.3 ms** at c = 1 / 8 / 32 (live 12.2 / 25.9 / 55.3). The run
reports `ratio_to_expected` per concurrency and `prod_equivalent_ms_est = slice step - bias`. Read it like this:

* ratio within about 10% at c=8 and c=32: the rank is right, deltas between option sets are credible, absolute numbers are
  within the stated estimate error (the collective table alone is an estimate; 25 of the live 55 ms at c=32).
* c=1 off but c=32 right: launch/CPU effects (in-process engine, graph breaks): look at `gpu_busy_fraction` with `profile=true`.
* everything off by the same factor: card (Max-Q) or clocks: see `telemetry`.
* `engine.log_checks` shows a backend that differs from prod (`FLASHINFER_CUTLASS` NVFP4 MoE, `DEEPGEMM_MXFP4` for the draft,
  `FlashInferCutlassMxfp8LinearKernel`, `fp8_ds_mla`, FP8 indexer cache, `[suffix dsv41-engram-mxfp4] ACTIVE`,
  `[suffix dsv41-sm120-block64] ACTIVE`): the slice is running other kernels than prod, fix that first (each mismatch is also a warning).

Prod weight-streaming roofline for the same rank (same byte model, `plan.step_stream_gb`): 6.17 / 23.45 / 39.87 GB per step
= 4.4 / 16.8 / 28.5 ms at 1.4 TB/s (3.4 / 13.1 / 22.3 ms at the 1.79 TB/s peak). The slice streams 6.98 / 27.34 / 40.65 GB.

## 8. Verified, not verified

CPU-proven by `tests/test_dsv41_slice.py` (137 tests, exact result lines in the lane report): the embedded config is the HF
file byte for byte (sha256); the Engram prime arithmetic reproduces the real `engram_num_embeddings` and the logged per-rank rows;
the slice's per-rank module shapes equal the prod rank's to the byte except the documented set; the weight model lands at the logged
40.66 GiB minus the unmodelled vision/rope part; the expert-streaming model reproduces the comms doc table; every opt's bounds and
enums, cross-checks and the env allowlist; engine kwargs and env; the synthetic tokenizer loads with `AutoTokenizer`
(transformers 5.20.2) and yields exactly the declared compressed vocabulary through a copy of vLLM's `build_compressed_token_map`;
the stepping loop, spec-counter deltas, error mapping, deadline and engine-check parsing against a fake engine and the real prod
boot-log lines; the non-dry path end to end without vLLM (re-exec, log tee, one JSON line, exit 7, temp dir removed).

**Not verified (needs the first lab run):**

1. That vLLM boots this config: nothing here ran vLLM. Static reading found no blocker, but the backend choice at EP=1 / top-1
   (NVFP4 MoE), Triton kernels with `BLOCK_H=1` in the Engram hash (`engram_heads=1`; fallback `engram_heads=8`), and DSpark with
   dummy weights are exactly where a refusal would come from. `engine_error` holds the text; `moe_topk`, `engram_heads` and
   `moe_backend` are the built-in escape hatches.
2. The tokenizer: tested with transformers 5.20.2 here, not with the pod's version.
3. The shared-expert patch: the three source anchors are checked against the fetched text, the wrapper is unit-tested on a stand-in
   class, neither ran inside vLLM. It refuses (exit 6) when the anchors drift.
4. In-process stepping with async scheduling (`step()` may return empty while the pipeline fills; handled, never observed) and the
   profiler on CUDA-graph replays.
5. The kernel-family regexes (written without a V4.1 kernel name) and the 4-pass sampler assumption.
6. The collective estimates, which dominate the c=32 calibration, are estimates from `dsv41-comms.md`, not measurements on this model.
7. The plugin must be on the lab pod's `PYTHONPATH` for the Engram overlay and the block-64 hook; the plugin's Helion/oxide
   kernels keyed to full-model shapes may refuse the slice's shapes (set their family switch to `oxide`/`auto`).

## 9. Follow-ups worth doing after the first run

* Change the synthetic rates at runtime (poke `RejectionSampler.synthetic_conditional_rates`) to sweep acceptance profiles in one boot.
* A custom `RoutingSimulator` strategy that sends the missing 25% of pairs to one hot expert would match the touched-expert counts of
  an EP rank exactly (E = 25.5 at M=48 instead of 30.5) at the price of three tiny elementwise kernels per layer.
* Real tokenizer files in the model dir would allow the vision tower and the MM processor (`limit_mm_per_prompt` default).
* Refine `KERNEL_FAMILIES` from the first `unclassified_top` lists.
