# DeepSeek-V4.1-Flash (`glm`): collectives, speculation, launch overhead, presets — lane E, blind

Branch `dsv41-comms` (base `dsv41` @ d0fec38). Brief: `plugin-harness/dossiers/dsv41-blind-brief-2026-10-19.md`
in the console repo. **Nothing here was measured on this model.** Evidence is the bring-up log
(`plugin-harness/results/dsv41-2026-10-17/boot5.log`, d6-spec.json), vLLM source at the pinned commit
`39983a5e7a5a575a7829f5d5ada50a33d9337ec1` (read from GitHub, line numbers below are in that tree), and the GLM-5.3
numbers of `plugin-harness/dossiers/glm53-speed-2026-10-11.md`. Every number marked *est.* is derived on paper.

## Report (brief format)

1. **Files.** `sitecustomize.py` (preset `dsv41-sm120-comms`, boot gates `host_ar_bench_dsv41`,
   `allreduce_matrix_bench_dsv41`), `suffix_hybrid/host_ar.py` (`route()`, `rounds()`, bench `--hidden/--vocab`, Engram
   gather shapes), `suffix_hybrid/nccl_split.py` (estimate tables + `table` CLI, per-band / per-fallback route log),
   `suffix_hybrid/mtp_tune/dspark_k.py` (k(c) hypothesis), `tests/test_dsv41_comms.py`,
   `tests/fixtures/vllm_39983a5e/cuda_communicator.py` (verbatim pin), this file. `nccl_qar.py`, `mtp_tune/*` otherwise untouched.
2. **Tests.** See the last section for the exact result lines.
3. **Gates.** Host all-reduce/all-gather at hidden 5120: K1 contract written (below), K2 unchanged (same Triton kernels,
   only the runtime `n` differs), K3 CPU = routing/size legality/anchors against the pin; **exactness on silicon is the boot gate
   `host_ar_bench_dsv41`** (the existing "every rank bit-exact vs the fp32 rank-order sum" check at V4.1 sizes), K4 silicon,
   K5 = the new route log lines (CPU-tested), K6 = arms C1..C5 below. Band table: an estimate, no gate. k(c): a hypothesis, no gate.
4. **Knobs.** New preset `SUFFIX_PRESET=dsv41-sm120-comms`; boot gates `host_ar_bench_dsv41`, `allreduce_matrix_bench_dsv41`;
   `python -m suffix_hybrid.nccl_split table --hidden 5120`; `python -m suffix_hybrid.mtp_tune.dspark_k`.
   Existing knobs used as arms: `SUFFIX_HAR_ONESHOT_MAX_BYTES`, `SUFFIX_HAR_ALLGATHER`, `SUFFIX_HAR_Q8_MIN_BYTES`,
   `SUFFIX_NCCL_AUTOTUNE`. vLLM knobs used as arms: see section 7.
5. **Handoffs.** Section 8.
6. **Risks / silicon first.** Section 8.
7. **Expected effect (est.).** Section 9.

**Follow-up 2026-10-19** (coordinator): hard constraints (section 10), the DP-attention layout study (11), the L2 prefetch port (12), overlap options
ranked (13). New files: `suffix_hybrid/har_prefetch.py` (port of the glm-int lane, V4.1 windows), `suffix_hybrid/comm_layout.py` (layout model + the
`dp_ag_rs_bench_dsv41` gate), `tests/test_har_prefetch.py`, `tests/test_dsv41_layout.py`; the PF path of `host_ar.py` (Triton section identical to glm-int),
the attach in `nccl_split.make_comm`, the `SUFFIX_HAR_L2_PREFETCH` hook and two boot gates (`har_l2_prefetch_bench_dsv41`, `dp_ag_rs_bench_dsv41`) in
`sitecustomize.py`. New knobs: `SUFFIX_HAR_L2_PREFETCH` (default off), `SUFFIX_HAR_L2_PREFETCH_MB` (32).

## 1. What crosses ranks (vLLM 39983a5e, TP8 + `--enable-expert-parallel`, dp=1)

Source: `vllm/models/deepseek_v41/nvidia/model.py`, `attention.py`, `common/engram.py`,
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`, `logits_processor.py`, `nvidia/dspark.py`.

| collective | where | tensor | count per target step |
|---|---|---|---|
| all-reduce | `VocabParallelEmbedding` | `[T, 5120]` bf16 | 1 |
| all-reduce | `wo_b` `RowParallelLinear` (attention output) | `[T, 5120]` bf16 | 40 |
| all-reduce | MoE late reduce (`_maybe_reduce_final_output`, shared + routed summed first) | `[T, 5120]` bf16 | 40 |
| all-gather (dim 1) | Engram layers 1 and 14 (`Engram.embed`, main stream; only the table lookup is prefetched on a side stream) | `[T, h, 256]` bf16, h = ceil(hash columns / 8) = 1 or 3 | 2 |
| all-gather | `LogitsProcessor` (`use_all_gather()` is True) | `[R, 16160]` bf16 per rank, R = verify rows (6 per request) | 1 |
| DSpark pass | embedding AR, 3 decoder layers x 2 AR, base-logit gather | `[5c, 5120]` AR, `[5c, 16160]` gather | 7 AR + 1 AG |

**Which tensor crosses with hc_mult = 4: the 1x sublayer output `[T, 5120]`, not the 4x stream.** The embedding is expanded to
`[T, 4, 5120]` locally (`model.py:528`), the mHC post/pre mixes run replicated on every rank, and only the attention and FFN
block outputs are reduced. The fused all-reduce + mHC kernel (`ops/cute_dsl/all_reduce_mhc.py`) is gated on TP=4, no EP, NVLink
multicast and SM100 (`ops/mhc.py:44-72`): unavailable here.

**EP in this build has no all-to-all.** `do_naive_dispatch_combine` needs `dp_size > 1` or sequence parallel
(`moe_runner.py:800`); `_use_sequence_parallel` needs MegaMoE (SM100) or dp > 1 (`model.py:258`). With dp = 1 every rank holds all
tokens, runs its 48 local experts (`MoEPrepareAndFinalizeNoDPEPModular` in the boot log) and the combine is the one all-reduce
above. So EP and a TP-sharded MoE move **identical bytes**.

Already ruled out by the boot log (`boot5.log:103-128`): vLLM custom all-reduce ("not supported on more than two PCIe-only GPUs"),
SymmMem ("capability 12.0 not supported"), FlashInfer all-reduce ("not supported for world_size=8"), FlashInfer PCIe-IPC (not
in the enabled list). Only `PYNCCL` is active, with no bands: the pool runs plain NCCL.

### Message sizes (bf16; 10 KiB per token)

Per step at concurrency c, k = 5: verify rows `6c`, DSpark pass rows `5c`.

| c | verify AR | draft AR | logits gather per rank (verify + draft) | Engram gather per rank (h=1) |
|---|---|---|---|---|
| 1 | 60 KiB | 50 KiB | 189 + 158 KiB | 3 KiB |
| 8 | 480 KiB | 400 KiB | 1.48 + 1.23 MiB | 24 KiB |
| 16 | 960 KiB | 800 KiB | 2.96 + 2.47 MiB | 48 KiB |
| 32 | 1.875 MiB | 1.56 MiB | 5.92 + 4.93 MiB | 96 KiB |
| prefill chunk 8192 | 80 MiB | n/a | last token per request only | 4 MiB |
| prefill chunk 16384 | 160 MiB | n/a | | 8 MiB |

Count per target step: 81 all-reduces + 2 Engram gathers + 1 logits gather; per drafter pass: 7 all-reduces + 1 gather.

## 2. EP8 versus TP8-sharded MoE, on paper

Communication: **no difference** (section 1). The decision is compute balance plus whether a kernel exists.

Per expert: 3 x 5120 x 2304 params x 0.5625 B (NVFP4 + fp8 scales) = 19.9 MB. Distinct experts read per layer, independent
top-6 routing over 384 experts (real routers are skewed: more imbalance, fewer distinct experts). EP per rank = 48 experts at full
I = 2304; TP = all active experts at I/8 = 288. Busiest rank = mean + 1.42 sigma of 8 ranks.

| tokens T | EP mean MB/rank | EP busiest MB | TP MB/rank | EP busiest / TP |
|---|---|---|---|---|
| 6 (c1) | 86 | 142 | 86 | 1.65 |
| 12 | 164 | 238 | 164 | 1.45 |
| 48 (c8) | 507 | 604 | 507 | 1.19 |
| 96 | 745 | 826 | 745 | 1.11 |
| 192 (c32) | 909 | 951 | 909 | 1.05 |

At ~1.4 TB/s that is 40 layers x (142 - 86 MB) / 1.4 TB/s = **1.6 ms/step at c1, 2.8 ms at c8, 1.2 ms at c32** in favour of TP
(est., bandwidth view only). GLM measured c1 +17 %, c8 +8.5 %, **c32 -3.7 %** for the same switch (narrow-tile kernels from
M ~ 96, g10): not transferable, but the shape is the same.

Blockers for TP-MoE on V4.1: (1) I/8 = 288 is not a multiple of 128; this is exactly the FlashInfer CUTLASS padding failure that
forced EP. Our own NVFP4 MoE lane (lane B) would have to pad `w2` K to 320 (+11 % on one of three matrices, ~+3.7 % bytes) or tile
at 32. (2) Dual layout (EP for c32, TP for c1-c8) needs 2 x 40 GiB of expert weights per rank: impossible. (3) Mixed
SP/all-to-all EP (`_use_sequence_parallel`) would move ~2.5 S per layer instead of 3.5 S (S = tokens x 10 KiB; AG 0.875 + RS 0.875
+ dispatch 0.05 + combine 0.66) but with 4 collectives per layer instead of 2 on a latency-bound host-routed fabric, and it
needs MegaMoE (SM100) or dp > 1: rejected.

**Decision: stay on EP8.** Only if lane B ships an I = 288 NVFP4 kernel that wins K4 at M <= 48 is "TP-MoE for everyone" worth an
arm (it would lose ~4 % at c32 if GLM repeats). That is a lane B / main-session call; there is nothing for comms to change.

## 3. Host all-reduce / all-gather at V4.1 sizes

### Contract (K1)

bf16, contiguous, `numel % 8 == 0` (`[T, 5120]`: always), one node, TP world 8 (`WORLDS = 2, 4, 8`), shared pinned `/dev/shm`
buffer of 72 MiB (`ROUND_BYTES` 4 MiB per rank per parity + reduced slots), int8 transport needs `numel % 128 == 0` (5120 = 40 x
128: always). 8192 x 5120 = 20 in-kernel rounds, 16384 x 5120 = 40. Other dtypes fall through to the default NCCL communicator
(logged once per distinct case since this branch, see K5). `all_gather` is exact bytes for any 2- or 4-byte dtype.
`tests/test_dsv41_comms.py::test_every_v41_message_is_host_ar_legal_and_round_counts` pins these.

### Routing (`host_ar.route`, refactored out of `all_reduce`, behaviour unchanged)

| tokens | bytes | kernel at `ONESHOT_MAX = 48K` |
|---|---|---|
| 1-4 | 10-40 KiB | one-shot |
| 5 (DSpark pass at c1) | 50 KiB | two-shot |
| 6 (verify at c1) | 60 KiB | two-shot |
| 48, 192, 8192 | 480 KiB .. 80 MiB | two-shot |

The 48 KiB threshold was tuned for GLM, where the MTP drafter ran 1-token passes. At V4.1 the 5- and 6-token passes sit just above
it. The boot gate prints one-shot and two-shot for 6 tokens; if one-shot wins at 60 KiB set `SUFFIX_HAR_ONESHOT_MAX_BYTES=64K`
(arm C2). Size of the prize: 88 all-reduces x (0-8 us) = at most ~0.7 ms/step at c1 (est.).

### Estimated all-reduce cost per size (`python -m suffix_hybrid.nccl_split table --hidden 5120`)

**ESTIMATE.** Log-log interpolation in *bytes* between the GLM bench points (tokens x 6144; host all-reduce, best NCCL
algorithm, int8 transport) and the stock-NCCL curve of 2026-09-29 (vLLM 0.30.0). Reproduces the GLM points exactly at GLM sizes
(test). NCCL 2.30.7 (this image) was never measured at these sizes: the "default" column is the stalest input. Above 6 MiB the
"default" curve is a straight-line extrapolation and the two NCCL columns agree within 2 % (GLM measured 17.0 ms best at 96 MiB): read the 8192 row
as "NCCL ~14 ms".

| tokens | bytes | NCCL default | NCCL best | host AR | host AR int8 |
|---|---|---|---|---|---|
| 1 | 10 KiB | 42 | 37 | 10 | |
| 6 | 60 KiB | 49 | 44 | 19 | |
| 12 | 120 KiB | 81 | 58 | 29 | |
| 24 | 240 KiB | 151 | 82 | 47 | |
| 48 | 480 KiB | 390 | 115 | 76 | |
| 96 | 960 KiB | 985 | 195 | 140 | 90 |
| 192 | 1.875 MiB | 1121 | 351 | 271 | 166 |
| 8192 | 80 MiB | ~14 000 | ~14 000 | 11 498 | 6 520 |

Per-step all-reduce time (81 verify + 7 drafter all-reduces; est.):

| c | host AR | NCCL best | NCCL default |
|---|---|---|---|
| 1 | 1.7 ms | 3.9 ms | 4.3 ms |
| 8 | 6.6 ms | 10.1 ms | 33.7 ms |
| 16 | 12.2 ms | 16.9 ms | 85.5 ms |
| 32 | 23.5 ms | 30.5 ms | 101.3 ms |

PCIe floor of the host two-shot (write 1.0 S, read 1.75 S per rank at 50 GB/s each way): 0.19 / 1.5 / 6.0 ms per step at c1 / c8 / c32,
so the kernel is latency-bound (barriers), not bandwidth-bound, at decode sizes. A prefill chunk of 8192 tokens costs 81 x 11.5 =
**0.93 s of all-reduce** (0.53 s with int8), which is why section 6 treats `max-num-batched-tokens` as an ITL knob.

The generated band is `SUFFIX_NCCL_BANDS=0-inf:har` for hidden 5120 and for 6144: host all-reduce wins at every estimated size,
same as glm53-sm120. Regenerate after the silicon matrix with `allreduce_matrix_bench_dsv41` (candidates x sizes, same timing
code as `SUFFIX_NCCL_AUTOTUNE`).

### Preset `dsv41-sm120-comms`

`dsv41-sm120` + `SUFFIX_NCCL_BANDS=0-inf:har`, `SUFFIX_HAR_ONESHOT_MAX_BYTES=48K`, `SUFFIX_HAR_ALLGATHER=1`. A new name, so the
live `dsv41-sm120` pool is not changed by a rebuild. No int8 transport. Explicit pod env still wins (`setdefault`).

### K5 evidence in the boot log

`[suffix nccl-split] ACTIVE: static bands=0-inf:har autotune=off qar=off host-all-gather=on`,
`tp:0: bands=0-inf:har ... (world 8, 2 communicator(s))`, then **new**: one line per band at its first use
(`route [0, inf) -> HostAllReduce (first 60K torch.bfloat16)`) and one line per distinct fallback to the default communicator
(`FALLBACK to the default communicator: <size> <dtype> contiguous=.. numel%8=..`). Rank 0 only, never per call, so the log
stays small across the 45 captured sizes and arbitrary eager prefill sizes. Engagement check: the `route` line exists, no
`FALLBACK` line for bf16.

### Int8 transport: quality-gate note (not in the preset)

GLM: rel-l2 0.9 % per all-reduce vs 0.17 % for bf16, 81-156 sequential all-reduces per step, g8q quality still passed (GSM8K-400
0.980) and gave +6.6 % / +9.1 % at c8 / c32 on top of g7n. For V4.1 it would apply from 256 KiB: every decode step at c >= 5 and every prefill
chunk. Gate before any speed number: `quality.py --tier full`, drift (FAIL lines KL > 0.02 or ppl ratio > 1.02, WARN 0.005 / 1.005),
the generation tasks (drift only sees prefill; decode needs the tasks), tools 5/5, a 512k needle. Arm C4, off by default.

### Host all-gather

Used for the logits gather and the two Engram gathers; exact bytes, same buffer and epoch counter as the all-reduce (all of them run
on the main stream, so one in-order collective stream: the Engram side stream only does the table lookup). Floor at c32: logits
payload 5.9 + 4.9 MiB per rank, read 7x = 76 MiB = **1.6 ms/step** at 50 GB/s, versus *est.* 7.7 ms for NCCL (GLM profile 9.2 ms at
c32 scaled by 129280 / 154880). The boot gate adds the V4.1 shapes (vocab 129280 and Engram h = 1 and 3).

## 4. Speculative decoding (DSpark, block 5)

### What runs and what does not

* DSpark drafts all 5 positions in **one parallel pass** (3 decoder layers with 128 experts top-3, EP8) and then 5 sequential
  Markov steps (`dspark/speculator.py`). The Markov weights are replicated (`disable_tp`) on purpose: sharding them would add a
  collective per position. Collectives of a drafter pass: 7 all-reduces `[5c, 5120]` and one full-vocab logits gather `[5c, 16160]`
  per rank. That is on the critical path (the sequential loop needs the gathered logits).
* The V2 runner does **not** use the per-step K in the drafter (`num_speculative_tokens` is accepted by `propose` and never read),
  so `num_speculative_tokens_per_batch_size` trims **verify width only**; the drafter always runs 5 wide. Capture: the V2 runner
  keeps FULL graphs (`config/vllm.py:1040` only downgrades on V1) and captures one decode graph per distinct K.
* `use_local_argmax_reduction` exists in the config but the DSpark speculator never reads it (it uses `logits.argmax` on the gathered
  logits). Adaptive verification does support DSpark.
* Online MTP tuning (`mtp_tune`) does not apply: `install_worker_hook` requires `MTPSpeculator` and logs "tuning off" for
  `DSparkSpeculator`. A DSpark head tuner would be a new phase (markov head + confidence head), not blind work.

### Acceptance data and its limits

`boot5.log:2472`: 39 steps, 195 drafted, 129 accepted, mean length 4.31, per-position (unconditional) 0.872, 0.769, 0.641, 0.590,
0.436. Sum of rates = 3.308 = 129/39, so they are the nested survival probabilities `P(accepted >= i)` and truncating k is exact.
One smoke window: the standard error of position 1 alone is ~0.05; the prompt mix is unknown. Expected tokens per step by k:
1.00, 1.87, 2.64, 3.28, 3.87, 4.31.

### k(c) hypothesis (`python -m suffix_hybrid.mtp_tune.dspark_k`)

Model: `spec_k_policy` (same as the GLM schedule), unit = one plain decode step at batch 1, `verify = lat + max(bytes, compute)` with
the MoE expert union growing with `N = c (k+1)` tokens, drafter = a fixed 0.11 unit pass independent of K (0.022 x 5). Central
constants: lat 0.55, dense 0.30, experts 0.15 (81 all-reduces + 40 eager attention regions + mHC/sampler launches dominate c1; 384
experts top-6). Two sensitivities: latency-bound (0.70 / 0.20 / 0.10) and bandwidth-bound (0.40 / 0.35 / 0.25).

`--speculative-config.num_speculative_tokens_per_batch_size` values (ranges are inclusive running requests):

| acceptance profile | central | latency-bound | bandwidth-bound |
|---|---|---|---|
| bring-up (above) | `[[1,3,5],[4,10,4],[11,32,5]]` | `[[1,32,5]]` | `[[1,1,5],[2,13,4],[14,32,5]]` |
| organic-hyp (0.764, 0.531, 0.345, 0.238, 0.132) | `[[1,1,4],[2,3,3],[4,27,2],[28,29,3],[30,32,4]]` | `[[1,2,4],[3,5,3],[6,23,2],[24,27,3],[28,32,4]]` | `[[1,1,3],[2,5,2],[6,21,1],[22,30,2],[31,31,3],[32,32,4]]` |

"organic-hyp" is the bring-up chain scaled by the GLM organic / burst ratio at each position: a *what-if for a lower-acceptance
mix*, not a measurement. Modelled speedup vs no speculation (central), k = 0..5:

| c | bring-up | organic-hyp |
|---|---|---|
| 1 | 1.00 1.54 2.01 2.32 2.55 **2.67** | 1.00 1.45 1.74 1.86 **1.90** 1.86 |
| 8 | 1.00 1.25 1.39 1.45 **1.49** 1.49 | 1.00 1.18 **1.21** 1.17 1.11 1.04 |
| 16 | 1.00 1.19 1.31 1.37 1.43 **1.45** | 1.00 1.12 **1.14** 1.10 1.06 1.01 |
| 32 | 1.00 1.20 1.36 1.49 1.62 **1.71** | 1.00 1.13 1.18 1.20 **1.21** 1.20 |

Reading it: with the bring-up acceptance a static k = 5 is within ~1 % of optimal at every c (the table's 4 at c4-c10 wins by <= 0.4 %), so **dynamic K is worth ~0 on that traffic and worth +12-17 % at c8-c16 if real traffic looks like the organic what-if**.
The rise at c32 for both profiles follows from the cost model, not from data: at 192 verify tokens nearly all 384 experts are read
anyway, so extra verify tokens are almost free until the compute ridge (the ridge constant, 140 tokens, is the weakest input). The decision therefore needs per-scenario acceptance, not
a guess: run arm S0 first, read `spec_decode_num_accepted_tokens_per_pos` per scenario (chat-en, chat-de, rag-8k, multiturn, tools,
thinking), then feed the rates to `spec_k_policy --rates` (needs a `dsv41` preset there: handoff) or to `dspark_k.schedule`.

### Better than a table: vLLM's own adaptive verification (`enable_adaptive_verification`)

`v1/worker/gpu/spec_decode/adaptive_verification.py`: DSpark-only. It profiles draft and verify step cost curves at startup
(graph replays at every capture size plus a tail beyond it), takes the confidence head's per-position probabilities, and each step
admits the global best prefix of (request, position) slots that maximises `expected accepted tokens / (draft cost + verify cost)`.
That is the k(c) table, learned on this hardware and per request. Requirements it checks at startup: the attention backends must
tolerate device/CPU query-length mismatch (`supports_device_cpu_query_lens_mismatch`, default True for non-SSM) and varlen decode
graphs (the FlashInfer sparse MLA builders declare `AttentionCGSupport.ALWAYS`; the DeepGEMM paged-MQA indexer backend is not
confirmed: **silicon-first check**). Cost: a profile pass at boot (`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`, default 8192) and
a D2H of confidences per step (stale by one step by design). Arm S4.

### Other DSpark levers (hypotheses, none implemented)

* `--speculative-config.dspark_draft_topk=64` (vLLM-native): the 5 sequential Markov steps stop reading the 66 MB replicated `markov_w2`
  and gather 64 rows per position instead. Proposal distribution truncated, verification exact. Est. <= 0.4 ms per drafter pass
  (5 x ~80 us) = ~1 %. Arm S5.
* Vocab-sharded local argmax for the drafter (greedy drafts only): per rank slice of `markov_w2` (8 MB), local max, 5 tiny
  (value, index) gathers instead of one 5.9 MiB-per-rank logits gather. Floor 5 x ~10 us vs 30 us (c1) / 1.6 ms host (c32): break-even
  at c ~ 2, saves ~0.3-0.8 ms per pass at c >= 8. It needs a patch of `DSparkSpeculator._sample_sequential` plus an oracle
  (bit-identical tokens versus the full-logits argmax including lowest-index tie-break). Not written blind: after `SUFFIX_HAR_ALLGATHER`
  the remaining prize is ~1 % per step. Candidate for the plugin if the profile shows the draft logits gather above 0.5 ms.
* Target-side local argmax (`LogitsProcessor.get_top_tokens`) exists but the V2 sampler needs full logits (temperature, penalties,
  logprobs); only a greedy fast path would use it. After host all-gather: <= 1.6 ms at c32. Skip.

## 5. Launch overhead

### Breakable CUDA graphs

* `DeepseekV41ForCausalLM` is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`config/vllm.py:74`). When on,
  `_maybe_enable_breakable_cudagraph` forces `CompilationMode.NONE` (`config/vllm.py:~838`); the boot log shows exactly that
  (`mode: NONE`, `cudagraph_mode FULL_AND_PIECEWISE`, 45 capture sizes up to 384, 89 s and 2.69 GiB per rank).
* The model has **no `@support_torch_compile`** (nothing in `model.py`, `attention.py`, `dspark.py`); the source comment at
  `config/vllm.py:114` says "DeepseekV41ForCausalLM cannot torch.compile". "Enabled custom fusions: norm_quant, act_quant" in the log is the
  config being printed; those are Inductor FX passes (`pass_config.fuse_norm_quant`, `fuse_act_quant`) that never see this model with or
  without breakable graphs. **So nothing available on this architecture is lost to NONE mode.** (GLM was different: its
  0.30.0 build had the fusions and the nightly dropped them.)
* `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is **not viable**: the default `FULL_AND_PIECEWISE` then needs piecewise compilation
  (`config/vllm.py:2137-2145` assertion) and attention is written as `eager_break_during_capture` regions
  (`attention.py:820,1023`); outside a breakable capture they would be captured into a plain graph, with CPU planning inside.
  Expected: a boot error, 25 minutes of node time. Only worth a boot if the owner wants closure (arm L3, run last).
* What the breakable design costs instead: the indexer + sparse attention (V2 runner: the narrow `_sparse_indexer_and_attn`) of each of the 40
  layers runs **eagerly from the CPU between graph segments** at every replay. If one eager launch burst exceeds the GPU time of the
  preceding segment, the GPU idles. This is the first number to get (arm L1): `step_profiler` reports CPU step wall vs GPU busy and
  cudagraph mode per step. Fusions that could still be ours: RMSNorm + activation quant ahead of NVFP4/FP8 GEMMs and SiLU-mul + quant
  in the shared expert, but the mHC kernels already fuse the two block norms and `fused_q_kv_rmsnorm` fuses the q/kv norm and the
  q quant; a kernel trace (L1) lists what is left before anyone writes a kernel.

### Scheduler arguments

* `--max-num-batched-tokens`: prefill all-reduce time is linear in tokens (1.4 us/token/all-reduce x 81 = 0.93 s per 8192 chunk), so 16384
  does not make prefill faster per token, it doubles the longest step every decode stream sees (ITL p99). 4096 halves it.
  A 512k prefill is 64 chunks of 8192: >= 60 s of all-reduce alone (est.), 34 s with int8. TTFT, not throughput, is the
  trade. Arms L5a (4096), L5b (16384).
* `--long-prefill-token-threshold N`: caps one request's chunk when other requests exist (`scheduler.py:763`); with `max_num_partial_prefills`
  at 1 it only matters when a long prompt competes with short ones. Arm L6 (2048).
* Async scheduling: with `dspark` it is auto-enabled (`config/vllm.py:1629-1688`), so the live pool already runs it; arm L2
  (`--no-async-scheduling`) measures what it is worth. `stream_interval` 2 (API-server host work) is a cheap add-on, ITL smoothness cost.
* Capture sizes: default max is `min(32 x 6 x 2, 512) = 384`. Decode needs <= 192. `--max-cudagraph-capture-size 192` saves capture time
  and part of the 2.69 GiB of graph memory (KV), but steps with a small prefill chunk riding on decode (192 < T <= 384) would run uncaptured.
  Arm L4; only if L1 shows few mixed steps.
* `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (default 1024): the three attention input projections on aux streams. Arm L7 (0 and 8192).

## 6. Prefill and long context

Prefill is all-reduce-bound at 8192 x 5120 (section 3). Levers in order: host all-reduce (C1), int8 transport (C4, gated), then chunk
size (L5). GLM's measured prefill gain from host AR + int8 was 1.65x TTFT at 14-179k tokens.

## 7. Arm file (not run, not added to the console repo)

`pools.yml` format of `plugin-harness/pools.yml` (the harness reads `url`, `model`, `plugin_prefix`, `env_fingerprint`, `install_markers`,
`spec_counters`, `bench`; arm names are labels, the `delta` blocks are the edit_pool changes versus `d6-spec.json`, applied with
`edit_pool` which **replaces the whole env list**: re-list `SUFFIX_PRESET`, `VLLM_USE_V2_MODEL_RUNNER` etc., never `PYTHONPATH`, then
`workload_detail` to diff). Two blockers for the main session: `bench-chain` refuses the `glm` workload and `glm.pl-ai.net`
(`harness.py:920-929`); AGENTS 3.6 lets the model's A/B run on the prod pool only on the owner's go with a downtime window, and the node is
whole-node (no dev pool can coexist). The entry below assumes a dev pool `dsv41-dev` created from d6-spec.json (hostname and served name
different) during an approved window, or the same arms rolled on `glm` with the guard lifted by the owner.

```yaml
pools:
  dsv41-dev:                       # whole worker-06; needs the owner's go + downtime window (AGENTS 1.1, 3.6)
    url: https://dsv41-dev.pl-ai.net
    model: dsv41-dev
    plugin_prefix: <vllm-suffix-hybrid-<full SHA of dsv41-comms head after CI-green>>   # new prefix per revision (P3, P4)
    env_fingerprint:               # the pod env list, not the preset expansion
      SUFFIX_PRESET: dsv41-sm120-comms
      VLLM_USE_V2_MODEL_RUNNER: "1"
    install_markers:
      - "[suffix dsv41-sm120-block64] ACTIVE"
      - "[suffix nccl-split] ACTIVE: static bands=0-inf:har"
      - "[suffix nccl-split] tp:0: bands=0-inf:har"
      - "route [0, inf) -> HostAllReduce"
    spec_counters:
      draft: spec_decode_num_drafts_total
    bench:
      baseline_arm: S0-dsv41-sm120
      speed_scenarios: chat-en,chat-de,rag-8k,multiturn,tools,thinking
      speed_conc: "1,8,32"
      speed_rounds: 5
      quality_tier: quick

arms:
  # --- baseline: d6-spec.json as served (SUFFIX_PRESET=dsv41-sm120). Measure FIRST and TWICE (A1, A2).
  S0-dsv41-sm120:
    note: plain NCCL, static k=5, async scheduling auto, breakable graphs auto. Also records per-position acceptance per scenario.
    delta: {env: {SUFFIX_PRESET: dsv41-sm120}}

  # --- C: collectives (one change at a time against S0; C1 is the headline)
  C1-har:
    note: host all-reduce + host all-gather, one-shot <= 48K (preset dsv41-sm120-comms). Engagement: route line, no bf16 FALLBACK.
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms}}
    boot_gates: [host_ar_bench_dsv41, allreduce_matrix_bench_dsv41]   # SUFFIX_BOOT_GATES, once per pod, read the exactness lines
  C1a-har-no-gather:
    note: isolates the all-reduce from the all-gather
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_ALLGATHER: "0"}}
  C2-har-oneshot64k:
    note: one-shot also for the 5-token drafter pass and 6-token verify (60 KiB)
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_ONESHOT_MAX_BYTES: 64K}}
  C3-nccl-autotune:
    note: no host AR; NCCL candidates timed at boot on the real group (tokens x 5120 sizes). Shows whether plain NCCL tuning alone closes the gap.
    delta: {env: {SUFFIX_PRESET: dsv41-sm120, SUFFIX_NCCL_AUTOTUNE: "1", SUFFIX_NCCL_AUTOTUNE_HIDDEN: "5120"}}
  C4-har-q8:                      # LOSSY: quality.py --tier full + drift + generation tasks + 512k needle before any speed claim
    note: int8 transport from 256 KiB
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_Q8_MIN_BYTES: 256K}}
  C6-har-l2pf:                    # section 12; bit-exact by construction (the boot self-test refuses otherwise)
    note: C1 + L2 weight prefetch of the next attention / ffn window during the all-reduce wait, 32 MiB per window
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_L2_PREFETCH: "1"}}
    install_markers:
      - "[suffix har-l2pf] armed"
      - "[suffix har-l2pf] ACTIVE: target"
      - "[suffix har-l2pf] schedule target @"           # 80 (or 81) windows; a missing line = nothing engaged
    boot_gates: [har_l2_prefetch_bench_dsv41]
  C6a-l2pf-16:
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_L2_PREFETCH: "1", SUFFIX_HAR_L2_PREFETCH_MB: "16"}}
  C6b-l2pf-64:
    delta: {env: {SUFFIX_PRESET: dsv41-sm120-comms, SUFFIX_HAR_L2_PREFETCH: "1", SUFFIX_HAR_L2_PREFETCH_MB: "64"}}

  # --- S: speculation (on top of the best C arm; same K values must be re-derived from S0's per-scenario acceptance)
  S1-k4:
    delta: {args: {--speculative-config.num_speculative_tokens: "4"}}
  S2-k3:
    delta: {args: {--speculative-config.num_speculative_tokens: "3"}}
  S3-dyn-bringup:
    note: central table for the bring-up acceptance; expect ~S0 (static 5 is optimal there)
    delta: {args: {--speculative-config.num_speculative_tokens_per_batch_size: "[[1,3,5],[4,10,4],[11,32,5]]"}}
  S3b-dyn-organic:
    note: what-if table for lower acceptance; wins only if the mix really looks like it
    delta: {args: {--speculative-config.num_speculative_tokens_per_batch_size: "[[1,1,4],[2,3,3],[4,27,2],[28,29,3],[30,32,4]]"}}
  S4-adaptive-verification:
    note: vLLM's profile-driven budget; startup error means the indexer / attention backend refuses varlen graphs (a finding, not a failure of the arm)
    delta: {args: {--speculative-config.enable_adaptive_verification: "true"}}
  S5-draft-topk64:
    delta: {args: {--speculative-config.dspark_draft_topk: "64"}}

  # --- L: launch / scheduling
  L1-profile:                      # not a speed arm: SUFFIX_PROFILE_WORKER windows at c1/c8/c32, read host gap + mixed-step share
    delta: {env: {SUFFIX_PROFILE_WORKER: "60:20:1", SUFFIX_PROFILE_WORKER_BUCKETS: "1,8,32"}}
  L2-no-async:
    delta: {args: {--no-async-scheduling: true}}
  L4-capture192:
    delta: {args: {--max-cudagraph-capture-size: "192"}}
  L5a-batched4096:
    delta: {args: {--max-num-batched-tokens: "4096"}}
  L5b-batched16384:
    delta: {args: {--max-num-batched-tokens: "16384"}}
  L6-long-prefill-2048:
    delta: {args: {--long-prefill-token-threshold: "2048"}}
  L7-multistream-off:
    delta: {env: {VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: "0"}}
  L7b-multistream-8192:
    delta: {env: {VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: "8192"}}
  L3-breakable-off:                # expected boot error; last, only for closure
    delta: {env: {VLLM_USE_BREAKABLE_CUDAGRAPH: "0"}}

  # --- X: layouts (section 11). X0 is a measurement, X1/X2 only if X0 passes the c32 rule (DP layer <= 2 x host AR - 120 us at 192 tokens)
  X0-layout-gate:                  # no serving load: boot gates only (SUFFIX_BOOT_GATES=dp_ag_rs_bench_dsv41,host_ar_bench_dsv41)
    boot_gates: [dp_ag_rs_bench_dsv41]
  X1-tp4-dp2:                      # expected: c1 +3.9 ms, c32 -4.9 ms (est.); sequence parallel on SM120 is untried
    delta:
      args: {--tensor-parallel-size: "4", --data-parallel-size: "2", --max-num-seqs: "16"}
      pod: {shm_mib: 110000}       # Engram dp_shared_memory holds one 97 GiB table copy in /dev/shm
  X2-dp8:                          # expected: c1 +9.3 ms, c8 +4.0 ms, c32 -7.8 ms (est.); dynamic K and the host AR lanes are inert
    delta:
      args: {--tensor-parallel-size: "1", --data-parallel-size: "8", --max-num-seqs: "4"}
      pod: {shm_mib: 110000}
```

Order: S0 (twice) -> L1 (read the profile) -> C1 -> C2/C1a/C3 -> C6 on C1 -> S-arms on the winning C arm -> L-arms -> C4 only after its quality
gate; X0 whenever 8 GPUs are free for a few minutes, X1/X2 only on a pass. Each arm is A/B against S0/C1 per AGENTS section 5 (A1 -> B -> A2, 5 rounds, c = 1/8/32); `verify-engagement --probe` before
every bench. Long-context cells (512k prefill TTFT, needle) for L5/L6/C4 come from the `rag-8k` scenario plus one manual 512k prompt.

## 8. Handoffs, risks, silicon-first checklist

Handoffs (nobody else edits my files; these are for the main session):

* **Lane B (NVFP4 MoE)**: TP-MoE only if an I = 288 kernel exists (pad `w2` K to 320 or 32-wide tiles) and wins K4 at M <= 48; comms
  does not change either way (section 2).
* **spec_k_policy owner**: add a `dsv41` `CostModel` preset (constants in `mtp_tune/dspark_k.py`: lat 0.55, dense 0.30, experts 0.15,
  384 experts top-6, draft 0.022, draft_ridge 32) so `--model dsv41` works on live logs. `tools/` is not in my ownership.
* **Attention lane**: the profile (L1) decides whether the eager indexer + attention regions leave GPU bubbles; `SUFFIX_FI_PLAN_HOSTFREE`
  exists for gemma's FlashInfer planning and may apply to the sparse-MLA wrapper.
* **Main**: `harness.py` refuses `glm` / `glm.pl-ai.net` for `bench-chain` (section 7). Add the `dsv41-dev` entry only with the owner's go.
  The console must know the new preset name (nothing greps `SUFFIX_PRESET` values today).
* **Prefill-side custom all-reduce lanes** (if any): the 80 MiB all-reduce (0.93 s per chunk) is the single biggest prefill item.
* **Attention lane**: FlashInfer's `benchmarks/bench_sparse_mla_sm120.py` with `(8, 128, 6)` and `(64, 128, 6)` is the only attention number the layout study still needs.
* **Main, if X arms ever run**: pod `shm_mib` > 100 GiB (Engram `dp_shared_memory`), per-replica `--max-num-seqs`, and a client-side `data_parallel_rank` pin per
  session or the prefix cache fragments.
* **Plugin owner of `host_ar.py` (me, later)**: a host reduce-scatter (phases 1-2 of the two-shot kernel) and a 1-byte-capable all-gather are the two kernels a DP layout
  would need; neither is written because the model says the layout does not pay.

Risks, in the order silicon should check them:

1. `host_ar_bench_dsv41` exactness + the 5/6-token one-shot vs two-shot numbers; Engram gather with h = 1 and 3.
2. The "default NCCL" column is a 2026-09-29 curve on a different NCCL: if NCCL 2.30.7 is fine at 0.4-2 MiB, C1 is worth ~6-7 %
   (versus best-NCCL), not 2x. `allreduce_matrix_bench_dsv41` answers it before any serving boot.
3. The 81 all-reduces inside **breakable** graph segments with the host kernel: proven on GLM nightly 2eb491e1 (also breakable), not on this one.
4. `SUFFIX_HAR_ALLGATHER` patches `CudaCommunicator.all_gather` for every group; non-tp groups keep NCCL (tested). The patch replays on the pin
   (`test_pin_*`), but the fixture is one file: re-check the anchors when the nightly moves.
5. DSpark adaptive verification may refuse the indexer backend at startup (a result, not a bug).
6. Acceptance numbers are one 39-step window.
7. L2 prefetch (C6) hooks by module structure: it needs the V2 runner, the DSpark draft model called through `__call__`, and `LinearBase` modules for the
   projections; the `ACTIVE` / `schedule` log lines say whether it engaged. The proxy bench uses equal-byte BF16 for the MXFP8 weights.
8. The layout model's NCCL reduce-scatter / all-gather are 0.5 x the best all-reduce, a guess until `dp_ag_rs_bench_dsv41` runs; DP lockstep sync is 200 us, a guess.

No strict fail-on-fallback mode was added: vLLM legitimately reduces other dtypes in the TP group; a crash on the first fp32
reduction during graph capture would cost a 25-minute boot to learn nothing. The log line makes every fallback visible instead.

## 9. Expected effect (estimates, with derivation)

| lever | c1 | c8 | c32 | derivation |
|---|---|---|---|---|
| C1 host AR vs NCCL best (est.) | -2.2 ms/step (+8 % tok/s at a 28.7 ms step) | -3.5 ms | -7 ms | table in section 3; 28.7 ms = 4.31 tokens / 150 tok/s (one-request read, not an A/B number) |
| C1 vs the stale NCCL-default curve | -2.6 ms | -27 ms | -78 ms | same table; **upper bound, depends on NCCL 2.30.7** |
| host all-gather | -0.8 ms | -1.6 ms | -6 ms | logits gathers, verify + draft: est. NCCL 0.9 / 2.0 / 7.7 ms (GLM profile x 129280/154880) vs host floor 0.05 / 0.4 / 1.6 ms (+ barriers) |
| dynamic K | ~0 | 0 to +17 % | 0 to +1 % | section 4 (bring-up vs organic-hyp; organic c8 k=2 vs 5, c32 k=4 vs 5) |
| draft top-k, local argmax | <= 1 % | <= 1 % | <= 1 % | section 4 |
| TP-MoE (if lane B delivers) | saves 1.6 ms | saves 2.8 ms | about -4 % (GLM's c32) | section 2 |
| int8 transport | n/a (below 256 KiB) | -1.5 ms (extrapolated below the first measured int8 size) | -8.5 ms | 166 vs 271 us at 1.9 MiB x 81; prefill 0.53 vs 0.93 s per chunk; GLM g8q +6.6 / +9.1 % |
| L2 prefetch (C6) | 0.6-1.4 ms | ~1 ms | ~1 ms | section 12: GLM -14..-36 us per layer x 40 layers; capped by the cold-read share of the window GEMMs |
| DP8 attention instead of TP8 | +9.3 ms (-24 % tok/s) | +4.0 ms | -7.8 ms | section 11.4; not recommended |
| TP4 x DP2 instead of TP8 | +3.9 ms | +1.9 ms | -4.9 ms | section 11.4; break-even c = 16 |

GLM anchors for plausibility only: g4 host all-reduce +15 % / +15 % / +23 % at c1 / c8 / c32 over an autotuned NCCL baseline.

## 10. Hard constraints (recorded 2026-10-19; reopen only with new silicon data)

1. **No fused all-reduce + residual + RMSNorm.** Exact but slower: 27.9 vs 23.7 us at 6 tokens, 0.73-0.96x at 1-96 tokens, because the
   local grid barrier before the row norm costs more than the separate norm launch (`host_ar.py` header, GLM g4; evidence ledger L026).
   vLLM's own `fuse_allreduce_rms` pass is off here anyway (`pass_config` in the boot log) and the model is not compiled.
2. **One poller per GPU.** Per-program flags in one shared line were 2-2.7x slower at 72-576 KiB (line contention); on separate 64 B lines 6 tokens
   took 42-93 us against 21.6 us; the epoch-in-every-8-byte-unit "LL" variant was exact but 2-100x slower (6 tokens 2.0 ms two-shot, 138 us
   one-shot vs 21 us): spinning volatile 8-byte loads flood PCIe with tiny reads. The barrier design stays: blocks arrive on a device counter,
   the last one polls the 8 flag lines.
3. **Comm kernels stay hand-written Triton (inline PTX release/acquire), no Rust CUDA, no oxide in the path.** Helion cannot express cross-rank
   spin loops and its grid barriers deadlocked on silicon (owner/coordinator 2026-10-19); Helion's own all-reduce example needs symmetric peer
   memory, which this box does not have (no P2P). Helion is for the dense/attention/MoE kernels around the collectives.
4. **Int8 transport is lossy**: per-128 absmax, rel-l2 0.9 % per all-reduce against 0.17 % for bf16 (GLM). It lives behind its own switch
   (`SUFFIX_HAR_Q8_MIN_BYTES`), never in a preset, and needs the quality gate of section 3 before any speed number.
5. **Infra**: no GPU P2P (PCIe cards), no cluster-admin. vLLM custom all-reduce, SymmMem, FlashInfer all-reduce and the PCIe-IPC path are all
   off in the boot log; anything that needs them is out.
6. **Every rank takes the same collective sequence.** Host all-reduce and all-gather share one in-order stream, one epoch counter and one 72 MiB
   buffer per group. Routing is a pure function of shape and dtype (`host_ar.route`), failures are boot errors (self-tests), never a per-rank
   runtime fallback: a rank that falls back alone deadlocks the others.
7. **No microbatching / DBO on V4.1**: the Engram layers do not support it (vLLM `docs/features/engram.md`, Limitations), so engine-level
   compute/communication overlap through dual-batch overlap is closed.

## 11. Layout study: DP8 attention + EP8 versus TP8 + EP8 (and TP4xDP2, TP2xDP4)

Question (coordinator, 2026-10-19): V4.1 has one KV head, so is the KV cache replicated on every TP rank, and would `--data-parallel-size 8
--enable-expert-parallel` (attention weights replicated, requests sharded, MoE through all-gather / reduce-scatter) beat TP8 + EP?
Model: `suffix_hybrid/comm_layout.py` (`python -m suffix_hybrid.comm_layout table`, tests in `tests/test_dsv41_layout.py`). All times are
ESTIMATES; the one input that decides c32 is measured by the boot gate `dp_ag_rs_bench_dsv41`.

### 11.1 The KV cache is replicated under TP (source)

`DeepseekV4Attention.get_kv_cache_spec` returns `MLAAttentionSpec(num_kv_heads=1, head_size=512, ...)` (`attention.py:1195`), the indexer cache
the same (`attention.py:1263`), and the SWA cache likewise (`SlidingWindowMLASpec(num_kv_heads=1)`, `v1/attention/backends/mla/sparse_swa.py`): the page size
has no TP term, so each of the 8 TP ranks stores every token. The only
mechanism that shards it is decode context parallelism (`dcp_sharded` on the spec); `deepseek_v41/` has no DCP code (the one hit is a guard that
refuses PCP together with DCP, `model.py:702`) and the engine runs `decode_context_parallel_size=1`. The bring-up log agrees: every rank reports
the same pool, `Available KV cache memory: 37.17 GiB`, `GPU KV cache size: 10,867,561 tokens` (3.65 KB per token over all layers), "Maximum
concurrency for 262,144 tokens per request: 41.46x". At `--gpu-memory-utilization 0.93` the pool is ~40 GiB, ~11.7 M tokens: a 524288-token request
needs ~1.8 GiB, so ~21 of them fit. KV is not scarce at TP8; 32 concurrent 512k contexts (the `max-num-seqs` ceiling) are the only case that does not fit.

### 11.2 Collectives and bytes per layer (S = tokens x 10 KiB; ring bytes per rank)

| layout | attention | MoE | bytes per layer |
|---|---|---|---|
| TP8 + EP8 (today) | all-reduce of the replica's rows | all-reduce (late reduce, no dispatch) | 2 x 1.75 S = 3.5 S |
| DP8 + EP8 | none (attention is local, `tp` group has world 1) | `AgRsAll2AllManager`: `all_gatherv` of NVFP4 activations + scales + topk (2.9 KB per token, 0.29 S_tot) and `reduce_scatterv` of the bf16 output (S_tot); uniform padded sizes fall back to plain `all_gather` / `reduce_scatter` (`cuda_communicator.py:all_gatherv`) | 0.875 S_tot + 0.25 S_tot = 1.13 S_tot |
| TPa x DPb + EP8 | sequence parallel (turned on by tp > 1 and dp > 1, `model.py:258`): `sp_all_gather` + `sp_reduce_scatter` over the TP group = one all-reduce equivalent | as DP8 over the 8-rank EP group | 1.75 S_rep + 1.13 S_tot |

S_tot is the token count the MoE sees: every DP replica pads to the busiest one (`sync_cudagraph_and_dp_padding`, `v1/worker/gpu/dp_utils.py`),
so c = 1 and c = 8 both ship 8 x 6 rows. "MoE AG + RS = one all-reduce equivalent instead of two" holds: 7 + 7 ring hops against 2 x 14, 3.1x fewer
bytes. Per-layer microseconds (est.; TP-group reduces on the host-AR curve, DP/EP collectives NCCL with RS = AG = 0.5 x best-NCCL all-reduce):

| layout | c=1 | c=8 | c=16 | c=32 |
|---|---|---|---|---|
| TP8 + EP8 (2 host all-reduces) | 38 | 152 | 280 | 541 |
| TP4 x DP2 | 81 | 153 | 236 | 404 |
| TP2 x DP4 | 95 | 132 | 205 | 333 |
| DP8 + EP8 | 89 | 89 | 141 | 237 |

The logits gather disappears too (the full-vocab `lm_head` is local): 0.05 / 0.4 / 1.6 ms host-floor per step at c1 / c8 / c32 saved.

### 11.3 What replication costs (lane D inventory, `docs/dsv41-dense.md` section 2)

Per rank at TP8 the TP-sharded tensors are wq_b 5.4, wo_a 8.4 (BF16 dequantized at load), wo_b 5.4, shared gate_up 3.0 + down 1.5 MB per layer
x 43 layers, plus `lm_head`, embedding (165 MB each) and the two Engram `wkv` (20 MB). At tp = 1 they are 8x larger.

| layout | dense MB streamed per rank per step | = ms at 1357 GB/s | extra resident weights | KV GiB per replica | aggregate KV (M tokens) |
|---|---|---|---|---|---|
| TP8 + EP8 | 2 621 | 1.9 | 0 | 40.0 | 11.7 |
| TP4 x DP2 | 5 190 | 3.8 | 2.5 GiB | 37.5 | 21.9 |
| TP2 x DP4 | 7 584 | 5.6 | 4.8 GiB | 35.2 | 41.1 |
| DP8 + EP8 | 12 372 | 9.1 | 9.5 GiB | 30.5 | 71.3 (6.1x, not 8x: the replicas' weights eat KV) |

Each request is limited to one replica's pool (DP8: 30.5 GiB, ~8.9 M tokens, against ~1.8 GiB for a 524288-token request): never binding.

Every rank streams all of this every step, also the idle ones: **+7.2 ms per step at every concurrency for DP8** (+1.9 ms TP4xDP2, +3.7 ms TP2xDP4, the
sequence-parallel shared expert counted: it is replicated on the TP group, `DeepseekV4MLP is_sequence_parallel -> disable_tp`). Lane D's FP8
`wo_a` saves 1.4 GB of the 12.4 (-1 ms). Lane D's floors assume every GEMM reaches the byte roofline; the TP8 shapes (N <= 1792) probably do not
(14 CTAs on 188 SMs), which would shrink the DP penalty: `small_gemm_x` in the model (TP8 dense stream 2x / 3x its floor) moves the DP8 delta at c1
from +9.3 to +7.4 / +5.4 ms and at c8 from +4.0 to +2.1 / +0.1 ms. The c1 loss survives; c8 only approaches a tie.

### 11.4 Net step time against TP8 + EP8 (est., ms per decode step; negative = faster)

| layout | c=1 | c=8 | c=16 | c=32 | no slower than TP8 from c = |
|---|---|---|---|---|---|
| TP4 x DP2 | +3.9 | +1.9 | -0.3 | -4.9 | 16 |
| TP2 x DP4 | +6.3 | +2.6 | -0.1 | -6.7 | 18 |
| DP8 + EP8 | +9.3 | +4.0 | +0.3 | -7.8 | 20 |

At a 28.7 ms c1 step that is -24 % tok/s for DP8. At c32 the saving is ~8 ms of a step nobody has measured (GLM's c32 step was 81-128 ms): single digits percent. The attention
kernel itself is a wash: at c1 one rank reads the same KV bytes (the 640 selected positions per token), only with 64 heads instead of 8
(FlashInfer v0.7.2 lists dedicated head counts {8, 16, 32, 64, 128} for the DSv4.1 decode path in `_STATIC_DECODE_ENVELOPE`, and its own
`benchmarks/bench_sparse_mla_sm120.py` benches (64 heads, topk 128, 1 and 16 tokens)); at c32 TP8 ranks all
gather the same ~12 MB of unique KV per layer while a DP8 rank gathers its 1/8 (~0.3 ms per step, not modelled).

### 11.5 The other items, from source

* **DP lockstep.** Every step the DP ranks agree the padded token count with a CPU all-reduce (`sync_cudagraph_and_dp_padding`, `get_dp_group().cpu_all_reduce_`;
  200 us in the model, unmeasured). A rank without requests runs `execute_dummy_batch` (`v1/engine/core.py`, busy loop) for as long as any
  rank has work: the full forward with `is_padding` rows, which `VLLM_MOE_SKIP_PADDING` (default on) turns into `topk_id = -1` so MoE experts skip them,
  if the FlashInfer CUTLASS NVFP4 kernel honours that sentinel (check on silicon). The dispatch / combine buffers keep the padded size. `dp_sync_interval` 16.
* **Engram.** Per-replica tables would be 8 x 97 GiB: impossible in a 229 GiB pod. `dp_shared_memory` (default on for dp > 1, `config/engram.py`)
  keeps ONE copy in `/dev/shm` (97 GiB, the same total as today's 8 x 12.2 GiB shards) and removes the per-step Engram DP collectives; the pod's
  `shm_mib: 32768` must grow to > 100 GiB, otherwise vLLM falls back (with a warning) to per-replica tables, 8 x 97 GiB, and the pod is OOM-killed. The #61530 overlay
  (`dsv41_engram_pr61530.txt`) contains the whole DP machinery (`get_engram_dp_group`, `gather_engram_hashes`, `dp_shared_memory`). Not exercised by any test.
  `embedding_across_dp: true` is the alternative (table sharded over all ranks, hash ids gathered and rows exchanged back each step).
* **DSpark.** One speculator per DP rank; the drafter's 3 MoE layers join the EP collectives, so vLLM runs a dummy draft pass on idle ranks to keep the
  ranks in step (`model_runner.py` dummy run). `num_speculative_tokens_per_batch_size` is **disabled for dp > 1** (`config/vllm.py:1059-1076`: ranks
  could pick different counts and deadlock), so the dynamic-K tables of section 4 do not apply; adaptive verification has no DP guard and is untested
  there (per-rank budgets change the per-rank token count the sync pads to).
* **Prefix cache.** The internal load balancer scores engines by queue depth and KV usage only (`core_client.py:get_core_engine_for_request`): a multi-turn
  or tool conversation lands on whichever replica is least loaded, so the prefix-cache hit rate falls towards 1/8 unless the client pins
  `data_parallel_rank` per session. Every miss re-prefills at ~0.93 s per 8192-token chunk (section 3). TP8 has one global cache.
* **Prefill.** One replica carries a whole chunk: the dense GEMMs that TP8 splits 8 ways run on one GPU (~0.3 s per 8192 chunk at the lane D
  throughput assumptions) against the 0.93 s all-reduce-bound TP8 chunk, plus a variable-size reduce-scatter of 80 MiB per layer. Same order; not modelled.
* **Plugin pieces that are TP-group specific.** `host_ar` / `HostAllReduce` and every nccl_split band exist for the `tp` group only (world 2, 4 or 8, one
  shared buffer per group); with tp = 1 the `tp` communicator has world 1 and `nccl_split.setup` returns at once, so the DP8 MoE collectives (`dp` / `ep`
  groups, `all_gatherv` / `reduce_scatterv`) stay on plain NCCL: there is no host reduce-scatter (it would be phases 1-2 of the two-shot kernel without the
  final gather, ~0.6x its time, not written) and the host all-gather moves 2- and 4-byte tensors only (the NVFP4 payload is 1 byte). `har_prefetch` needs a
  host all-reduce in the worker and refuses to start without one. `dspark_k` tables are void under DP (above). `mtp_tune` is not wired for DSpark at all.
  The dense lanes (MXFP8 / NVFP4 / Helion) are keyed on (N, K, M): DP changes N and K of wq_b / wo_a / wo_b / shared / lm_head, so the lab cases would
  need TP1 shapes.
* **Mixed TP2 x DP4 / TP4 x DP2.** Sequence parallel switches on (`_use_sequence_parallel`), `wo_b.reduce_results = False`, the shared expert is replicated on the
  TP group, `sp_all_gather` goes through the host all-gather patch (tp group) while `sp_reduce_scatter` stays NCCL; the GEMM-RS fusion is SM100-only. Two or four
  `HostAllReduce` buffers (one per TP group) on the same node. Untried on SM120.

### 11.6 Recommendation and the measurements that would change it

**Keep TP8 + EP8.** DP8 attention loses 9 ms (c1) and 4 ms (c8) per step (TP4 x DP2: 3.9 / 1.9 ms), because the replicated attention / shared / lm_head weights are
streamed on every rank every step. That fails the prod bar ("no LOSS in any metric", c1/c8/c32 all) on its own; it also switches off two plugin lanes (host
all-reduce / all-gather, dynamic K) and fragments the prefix cache. Its only win is a saturated pool (c >= 16-20), 5-8 ms per step, and 6x aggregate KV that is not scarce. If the owner wants a
throughput-oriented replica, make it a separate pool; do not change this one.

What would reopen it, in order of cost:

1. **Attention kernel at 8 vs 64 heads (one GPU, minutes)**: FlashInfer's `benchmarks/bench_sparse_mla_sm120.py` (add `(8, 128, 6)` to `decode_configs`; it already has
   (16 | 32 | 64, 128, 1 | 16)). Its Python dispatch table is only a coarse pre-filter (the C++ dispatcher is authoritative), so a zero-GPU probe would be
   vacuous, and vLLM's init check (`has_flashinfer_sparse_mla_sm120_config`) consults that same pre-filter: a missing 64-head instance would first show at
   the first decode.
2. **`dp_ag_rs_bench_dsv41`** (one boot gate, minutes): NCCL reduce-scatter, the four-tensor all-gather group, NCCL and host all-reduce at 48 / 96 / 192 padded
   tokens, one TP8 layer (2 host all-reduces) against one DP8 layer (RS + AG). Go / no-go for c32: DP layer <= 2 x host-AR - 120 us (192 tokens: <= ~420 us;
   the model says 237 us). Anything above rules DP out even at c32. Feed the measured numbers to `comm_layout.step(..., costs=(ar, rs, ag))`.
3. **A TP8 step profile at c1 / c8 / c32** (`SUFFIX_PROFILE_WORKER`, arm L1): the share of attention all-reduces, dense GEMMs (are they at 2-3x their floor?)
   and the logits gather replaces three of the model's guesses.
4. Only then a dev boot of DP8 or TP4 x DP2: pod `shm_mib` > 100 GiB, `--max-num-seqs 4..8` per replica (it is per DP rank), the `topk_id = -1` check, Engram
   `dp_shared_memory`, static k.

## 12. L2 weight prefetch during the host all-reduce: port decision (SUFFIX_HAR_L2_PREFETCH)

**Decision: port, both windows, off by default (arm C6), not in the preset.** Code: `suffix_hybrid/har_prefetch.py`, the PF path of `host_ar.py` (the
Triton section is byte-identical to glm-int 5efcfd8, where it ran on the same RTX PRO 6000 nodes: outputs bit-identical, all-reduce unchanged at
20.8 us, attention + MoE window -14 to -36 us per layer at M=6, three boots, noisy), the attach in `nccl_split.make_comm`, the sitecustomize hook and
the boot gate `har_l2_prefetch_bench_dsv41`. Tests: `tests/test_har_prefetch.py` (20).

What V4.1 reads after each of its 80 target all-reduces (per rank at TP8; MXFP8 weights count their ue8m0 scales):

| window (starts after) | tensors in use order | MiB |
|---|---|---|
| attention (the MoE all-reduce of the previous layer) | `hc_attn_fn` f32 [24, 20480] 1.9, `fused_wqa_wkv` 9.3, `wq_b` 5.2, `wo_a` 8.0 (BF16; read through `.weight` by `deep_gemm_fp8_o_proj`, never as a module call: recorded as an extra before `wo_b`), `wo_b` 5.2 | 29.2 |
| ffn (the attention all-reduce of the same layer) | `hc_ffn_fn` 1.9, router `gate` 3.8, shared `gate_up` 2.9, `down` 1.5 | 10.0 |
| attention window of an indexer / kv-source layer | + `indexer.wq_b` 5.2, `weights_proj`, compressor `wkv` / `wgate` 5.2 each (and `engram.wkv` 20 on layers 1 and 14) | cut at the budget, tail first |

* **Prefetch the next attention weights AND the ffn window**, in use order, 32 MiB per window (the default fits the 29.2 MiB attention window; the
  boot gate sweeps 16 / 32 / 64). The attention window is the bigger prize: 29 MB at ~1.36 TB/s is ~22 us of cold GDDR reads against the ~19 us the
  60 KiB all-reduce spends waiting on PCIe, so nearly all of it can be hidden; the ffn window is ~7 us.
* **Not prefetched**: the routed experts (48 x 19.9 MB per rank, 86-142 MB touched per layer at M=6: which ones is known only after the router, and the
  working set exceeds the 128 MB L2), `lm_head` (165 MB, outside the root forward), the hash-routing table `tid2eid`, the embedding.
* Hooks cover the target and the DSpark root; windows come from forward pre-hooks on `LinearBase` modules plus two extras (`wo_a` before `wo_b`, `hc_*_fn`
  before `attn` / `ffn`). A record's first all-reduce gets the window after it: in the DSpark forward that is the embedding reduce (`inputs_embeds=None`, so
  it runs inside `forward`) and it prefetches draft layer 0; the target's embedding reduce runs outside `forward` (the V2 runner precomputes `inputs_embeds`
  for multimodal models, `model_runner.py:269,1946`), so layer 0's attention window is not prefetched (1 of 80).
* **Expected effect (est.)**: 14-36 us x 40 layers = 0.6-1.4 ms per step at c1 (2-5 % of a 28.7 ms step); at c8 / c32 the all-reduce wait (76 / 271 us) exceeds
  the ~22 us weight read, so the cap is the cold-read share of the window GEMMs, ~1 ms per step (1-2 %). Plus the DSpark pass (3 layers). Not end-to-end
  measured even on GLM.
* **Fail-closed**: gate `SUFFIX_HAR_L2_PREFETCH=1` (inert otherwise: the sitecustomize hook is not even installed); at `make_comm` every rank proves the
  two-shot, one-shot and (when enabled) int8 all-reduce bit-identical with and without prefetch, else the worker refuses to start; `load_model` refuses
  without a host all-reduce in the worker, without a model, or with no Linear-like module under a root; a `(root, tokens)` schedule that changes between
  eager warm-up and capture turns that key off (its rows are zeroed after capture); any exception in a hook disables everything and zeroes all tables.
  Rows are 16 B aligned inside the tensor's own storage and the tensors stay referenced, so a prefetch never points at freed memory.
* **K gates**: K1 rows contract and windows above; K2 kernel unchanged from silicon; K3 bit-exact self-test (silicon, every boot) + 20 CPU tests; K4
  `har_l2_prefetch_bench_dsv41` (proxy windows: MXFP8 weights as equal-byte BF16; the verdict line prints us per layer and ms per step); K5 log lines
  `[suffix har-l2pf] armed`, `ACTIVE: target <class> (N modules), draft <class> (M modules)` and up to two
  `schedule target @ 6 tokens: 80 windows after an all-reduce, ...` lines per root (80 for the target, 7 for the draft root);
  a missing `ACTIVE` or `schedule` line means the gate engaged nothing (V1 runner, draft not called through `__call__`); a line
  `first record of ... happened inside CUDA-graph capture` means vLLM captured that shape without an eager warm-up, so its graph replays without prefetch
  (safe, but that shape gains nothing); K6 arm C6.
* **Silicon first**: the bench's verdict line and its `exact` bit; the two log lines above; then C6 against C1.

## 13. Communication / compute overlap options, ranked (est. ms per step at c1 / c8 / c32)

Nothing in the TP8 layer is independent of the all-reduce it waits for (mHC post needs the reduced block output, the next GEMM needs the norm), so
overlap means hiding *memory* work in the wait, or shrinking what waits.

| rank | option | est. ms/step | status |
|---|---|---|---|
| 0 | shared expert on an aux stream next to the routed experts | ~0.4 at every c | stock (`SharedFusedMoE`); nothing to do |
| 1 | **L2 prefetch of the next window during the all-reduce wait** | 0.6-1.4 / ~1 / ~1 | implemented, arm C6, off by default (section 12) |
| 2 | one-shot all-reduce for the 5- and 6-token passes (`SUFFIX_HAR_ONESHOT_MAX_BYTES=64K`) | 0-0.7 / 0 / 0 | arm C2, one env |
| 3 | mHC coefficient generation on a side stream (vLLM's `mhc_stream` overlap, SM100-only today: needs DeepGEMM prenorm) | 0.2-0.4 / ~0.3 / ~0.3 | would need a Triton / TileLang port of the overlap path; the mHC lane, not comms |
| 4 | programmatic dependent launch (PDL) on the all-reduce kernel boundaries (graph gaps ~2 us x 2 x 88) | 0.15-0.3 | unproven in Triton for these kernels; low |
| 5 | routed-expert prefetch after the router (first tiles of the selected experts while the shared expert and the launch run) | <= 0.4 | lane B, inside the MoE kernel |
| 6 | split the attention block into two token halves so half B's compute hides half A's all-reduce | 0 / 0 / <= 4 | **reject**: the spin-barrier kernel holds 64 of 188 SMs while it waits (it would slow the overlapped compute by up to a third), the halves re-read the weights, the attention eager breaks double, and AR(S/2) costs the same at c <= 8 |
| 7 | split the MoE by experts or tokens to overlap its all-reduce | negative | **reject**: expert split doubles the all-reduce bytes for the same exposed tail; token split re-reads ~all expert weights (bandwidth-bound at c >= 8) |
| 8 | overlap drafter and verify | 0 | impossible: the drafter needs the accepted tokens and the aux hidden states of the verify step |
| 9 | DBO / microbatching | n/a | blocked by Engram (section 10, item 7) |
| 10 | fused all-reduce + residual + RMSNorm | negative | closed (section 10, item 1) |

The biggest lever hiding behind these numbers is not overlap but the eager indexer + attention regions between the graph segments (section 5, arm
L1): if the profile shows a host gap there, graphing them is worth more than everything above combined.

## 14. Tests

Run with `/tmp/lane-c/venv/bin/python` (torch 2.14.1, pytest 9.1.1, Python 3.12), the worktree on `PYTHONPATH`, no GPU, no `_native` build needed.

```
PYTHONPATH=$PWD python -m pytest tests/test_har_prefetch.py tests/test_dsv41_layout.py tests/test_dsv41_comms.py tests/test_nccl_split.py \
    tests/test_spec_k_policy.py tests/test_mtp_tune.py tests/test_rocm_preset.py scripts/test_runtime_bundle.py tests/test_runtime_bundle_vendor.py -p no:cacheprovider
============================= 114 passed in 10.09s =============================
PYTHONPATH=$PWD:$PWD/sm120 python -m pytest sm120/tests/test_boot_gates.py sm120/tests/test_vllm_nightly_anchors.py tests/test_dsv41_engram.py -p no:cacheprovider
============================== 28 passed in 0.95s ==============================
```

New this follow-up: `tests/test_har_prefetch.py` 20, `tests/test_dsv41_layout.py` 9 (the first lane E pass added `tests/test_dsv41_comms.py` 21). `ruff check --select F,E9,E7`
is clean on the new files; its 7 F821 on `host_ar.py` (`x` after a `del` in the bench) are in the base file.

Not run: `tests/test_wrap_v2_install.py` and the other `_native` tests (no maturin build here); nothing in them touches these files.
CPU cannot run: the Triton host kernels (no Triton on the Mac; the kernel section of `host_ar.py` is byte-identical to the silicon-proven glm-int one, checked with `diff`),
`cp.async.bulk.prefetch`, the three benches (`host_ar`, `har_prefetch`, `comm_layout` need 8 GPUs), the DSpark / vLLM behaviour claims (read from source only).
