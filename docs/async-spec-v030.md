# Spec decode under async scheduling in vLLM v0.30.0 (V2 runner), and SUFFIX_HYBRID_ASYNC

Pinned source: `vllm-0.30.0/vllm`, `VLLM_USE_V2_MODEL_RUNNER=1` (`vllm/v1/worker/gpu/*`).
Anchors are `path:line` relative to `vllm/`. Implementation: `suffix_hybrid/async_spec.py`.
Tests: `tests/test_async_spec.py`.

## 1. What turns async on or off

- `VllmConfig.__post_init__` (`config/vllm.py:1407-1487`). An explicit `async_scheduling=True` hard-fails for
  `disable_padded_drafter_batch=True` (`:1429-1433`). With `None`, async resolves to `True` for
  EAGLE/MTP/draft_model/NGram-GPU/DSpark unless `disable_padded_drafter_batch=True` (`:1463-1471`), the executor
  lacks support, or the model is a pooling model.
- **In the V2 runner `disable_padded_drafter_batch` is read nowhere except
  `spec_decode/extract_hidden_states.py:33`.** Its only effect on our MTP models is turning async off.
  The V1 meaning (an unpadded drafter batch built from CPU `sampled_token_ids`, `v1/worker/gpu_model_runner.py:5172-5240`)
  does not exist in V2. The V2 drafter always receives `num_sampled` and `num_rejected` as device tensors
  (`model_runner.py:2143-2157`), so it is "padded" by construction.
- `max_concurrent_batches = pp_size + 1` under async on V2 (`config/vllm.py:588-599`), so 2 at PP=1.
  `EngineCore` then runs `step_with_batch_queue` (`v1/engine/core.py:234-237`).
- Today `sched_sync.py` forces `async_scheduling=False` whenever `SUFFIX_HYBRID_WRAP=1`.

## 2. The async step pipeline with spec decode

### Engine core (`v1/engine/core.py:630-744`)
Step N+1 is scheduled and dispatched (`execute_model` plus `sample_tokens`, both non-blocking) while step N is still
in flight. The engine blocks only when the queue (depth 2) is full: `batch_queue.pop()` then `future.result()`
(`:701-706`), then `update_from_output(N)`. So the scheduler has not seen step N's output when it builds step N+1.
`post_step` fetches draft ids only when async is off (`:621-628`). Under async the only draft fetch is the
structured-output deferred path (`:715-742`), which calls `take_draft_token_ids` to grammar-validate drafts.

### Scheduler: placeholders and reconciliation
- `AsyncScheduler._update_after_schedule` (`v1/core/sched/async_scheduler.py:19-50`) adds
  `1 + len(spec)` to `num_output_placeholders` for every scheduled decode request and sets
  `request.spec_token_ids = [-1] * num_spec_tokens_to_schedule` (`:44`). These are **placeholder draft tokens**:
  the scheduler never sees real drafts.
- On the next `schedule()`, those K placeholders become `scheduled_spec_decode_tokens[req] = [-1]*K`
  (`scheduler.py:811-827`), and `num_scheduled_tokens = 1 + K`. KV slots are allocated for 1+K.
- **Reconciliation** in `update_from_output` (`scheduler.py:1977-1995`):
  `num_rejected = len(scheduled_spec) - (len(generated) - 1)`, then
  `num_computed_tokens -= num_rejected` and `num_output_placeholders -= num_rejected`. The scheduler charged 1+K
  optimistically and gets back exactly `1 + accepted`. **Any worker-side trim of a row to w < K drafts is
  therefore invisible to the scheduler and still exact.** The trimmed slots count as rejected, which is
  how adaptive verification (AV) gets away with device-side trimming. Side effect: vLLM's spec-decode
  acceptance metrics use K as the denominator (`make_spec_decoding_stats`, `:2801-2813`).
- New or resumed decode rows are padded to 1+K with `[-1]` regardless of async (`scheduler.py:1043-1060,
  1266-1270`) to keep FULL decode graphs.

### Worker (V2 runner)
- The CPU `num_computed_tokens_np` is an **optimistic mirror** (`states.py:61`; set from the scheduler's
  optimistic value at `model_runner.py:1189`). Under async it overestimates by at most one step of rejections.
  It only feeds upper bounds (`seq_lens_cpu_upper_bound`, `model_runner.py:1412-1420`).
  `model_states/mamba_hybrid.py:204-209` notes the same thing.
- The GPU is authoritative: `_post_update_kernel` (`input_batch.py:552-609`) advances
  `num_computed_tokens += query_len - num_rejected`, writes sampled tokens into `all_token_ids` (UVA) and bumps
  `total_len`. `num_rejected = num_logits - num_sampled` per row (`input_batch.py:500-548`).
- Draft tokens never leave the GPU. `propose()` output is scattered into `req_states.draft_tokens[idx]`
  (`model_runner.py:2158`). Step N+1's `combine_sampled_and_draft_tokens` (`:1389-1400`, kernel
  `input_batch.py:397-497`) reads that row for `cu_num_logits[i+1]-cu_num_logits[i]-1` drafts. Those widths come
  from the CPU `scheduled_spec_decode_tokens` lengths (`model_runner.py:1292-1324`).
- `DraftTokensHandler` (`spec_decode/utils.py:29-70`) copies real drafts to the host only under structured
  output. Otherwise `get_draft_tokens` returns K-wide `[-1]` rows. The wrap's ragged widths ride on this handler
  (`wrap_v2._patch_get_draft_tokens`), which async never calls.

### Where the CPU waits

| | sync (`step`) | async (`step_with_batch_queue`) |
|---|---|---|
| output N | engine thread, `future.result()` → `AsyncOutput.get_output` → `copy_event.synchronize()` (`async_utils.py:167-168`) | Multiproc: `WorkerAsyncOutputCopy` thread (`multiproc_executor.py:683-691, 1011-1028`); the worker main thread never waits. UniProc: `AsyncOutputFuture.result` in the engine thread (`uniproc_executor.py:38-48`) when the queue is full |
| drafts | `post_step` → `take_draft_token_ids` RPC every step (`core.py:625-628`); host sync only with structured output (`spec_decode/utils.py:63-66`) | none (structured output: deferred path `core.py:715-742`) |
| AV | — | `record_confidences` synchronizes the **previous** step's copy event (`adaptive_verification.py:259-262`) |

The output copy event is recorded **before** `postprocess_sampled` and `propose` (`model_runner.py:2081-2121`).
So even in sync mode, the engine's turnaround (update, take_draft RPC, schedule, prepare) overlaps the MTP
drafter's GPU time. Sync loses only `max(0, turnaround - drafter_gpu_time)` per step.

### CUDA-graph mode of verify steps
FULL decode graphs are captured only for uniform `1+K` rows (`cudagraph_utils.py:310-330`). The exceptions are
`varlen_decode` (enabled only with AV, `model_runner.py:722`) and dynamic-SD multi-length capture (`:265-291`).
Dispatch requires `uniform_tok_count == decode_query_len`.
- Native MTP (sync or async) gives all decode rows 1+K, so FULL. Qwen4Exp/GLM are breakable graphs: attention runs
  eager between segments (`config/vllm.py:77-104`).
- The wrap's ragged widths, and the async trim below, give mixed widths, so PIECEWISE. An all-miss step (every
  row 1 token) is also PIECEWISE, because no uniform-1 graph is captured when K>0.

### Adaptive verification and query-lens mismatch (`spec_decode/adaptive_verification.py`)
The CPU picks a total draft **budget** from stale (lag-1) confidences and cost tables (`get_num_tokens`,
`:280-348`). `compact_batch` (`:350-388`) spreads the budget evenly on the CPU as an upper bound.
`reallocate_drafts` (`:390-445`) then picks per-row counts on the GPU (top-k survival) and rewrites
`query_start_loc` and `cu_num_logits` **on the device**. The CPU and device query lengths now disagree, so every
target backend must declare `supports_device_cpu_query_lens_mismatch()` (`attention/backend.py:212`, checked at
`:340-347` and in the factory `:448-504`) and report `AttentionCGSupport.ALWAYS`. FlashInfer declares it only on
the SM100 family with trtllm-gen decode (`backends/flashinfer.py:456-468`). The DSA indexer declares it only on
SM90/SM100 (`mla/indexer.py:190`), and SSM backends (GDN) opt out. That is why AV is refused on all our SM120
pools (MINED-LESSONS 2026-09-29).

## 3. qwen3.8-flash c8 async regression: hypothesis and check

Observation (MINED-LESSONS, qwen A/B/A 2026-09-30): native MTP k=4, TP2, wrap off. Async (flag removed) lost c8
per-stream throughput (med 164 vs 173) and ITL p99 (0.118 vs 0.101 s) against sync.

**Hypothesis H-A (primary): qwen-flash decode is host-launch-bound, so async has no GPU slack to hide and only adds
host contention.**
1. Qwen4Exp runs breakable cudagraphs: eager QSA/GDN attention between graph segments (`config/vllm.py:77-104`),
   about 97 eager launches per call, GPU busy 69-85 %. The host must issue every segment and attention launch, so
   step time is roughly max(host, GPU) with host ≥ GPU at small batch.
2. Sync already overlaps the engine turnaround with the drafter (see "Where the CPU waits"), so async's upside is small
   (upstream +2-7 %).
3. Async adds a second Python thread in the output-rank worker. `WorkerAsyncOutputCopy` (`multiproc_executor.py:683-691`)
   wakes on `copy_event`, then `tolist()` and MessageQueue-enqueues under the GIL (`async_utils.py:167-190`,
   `multiproc_executor.py:987-1009`) while the main thread launches step N+1. Contention runs both ways. With the
   default 5 ms GIL switch interval, the output of step N can sit behind N+1's launch loop, and the engine needs
   output N to schedule N+2 (queue depth 2, `core.py:687-706`). On a CPU-quota pod, CFS throttling of the extra
   runnable thread compounds this.
4. Minor contributors: rows that finished at N are still computed at N+1 (scheduled before output N), and the
   align-mode mamba pre-copy launches every step because `num_computed_tokens_np` is optimistic (`model_states/mamba_hybrid.py:204-209`,
   about 0.3 % TPOT).

**Predictions and measurable check** (same dev pod and GPUs, wrap off, native MTP, c1/c8/c32, A = sync via the flag,
B = async):
- B shows **no higher GPU busy fraction** than A at c8, plus a larger per-step host gap (rank-0 worker
  nsys/torch.profiler, since the EngineCore step profiler is blind at TP>1).
- `py-spy record --gil --idle -p <output-rank worker>` under B at c8: a visible share of samples where
  `WorkerAsyncOutputCopy` holds or waits for the GIL during decode. It is absent in A, because that thread does not exist there.
- Falsifier knob: in B, set `sys.setswitchinterval(2e-4)` in the worker processes (one line in
  `sitecustomize.py`). If c8 ITL p99 returns to A, the GIL convoy is confirmed. If B's GPU busy exceeds A's and
  ITL is still worse, H-A is refuted and the next suspects are item 4 and scheduler-side effects.
- H-A predicts the regression **shrinks or flips at c32**, where GPU work per step grows (MoE) and slack appears.

## 4. SUFFIX_HYBRID_ASYNC: design

Full contract and invariants I1-I4 are in the `async_spec.py` docstring. Summary:

1. **Lag-1 host lookup.** `propose(N)` enqueues a non-blocking D2H of `total_len[idx]` into pinned staging and
   records an event at the end of the step. `propose(N+1)` synchronizes that previous-step event (the same pattern as
   AV `record_confidences`, wrapped in `gpu_sync_allowed()`). It then runs the Rust `V2SuffixProposer`
   (unchanged, constructed with `k=2K+1`) on the UVA history `H_N` to get a candidate `C` anchored at `p = |H_N|`.
2. **Device resolve** (`resolve()`, about 10 small torch ops, no sync). `o = total_len - p` ∈ [1, K+1].
   If `all_token_ids[p:p+o] == C[:o]`, the drafts are `C[o:o+K]` (positions ≥ |C| are filler 0).
   Otherwise the whole row is filler. Filler is always in-vocab, and correctness never depends on it.
3. **Host-exact trim** (suffix-only): the host knows `w = min(K, |C|-1)` per row before the next step. A wrapper on
   `runner.execute_model` hands the runner a **shallow copy** of `SchedulerOutput` with spec rows cut to `w`
   (width 0 = plain decode, no candidate = 0). The original is never mutated, because under UniProc it is the
   scheduler's own object and reconciliation needs the K-wide lists. The CPU and device `query_start_loc` agree, so
   **no backend needs query-lens-mismatch support** and the verify shapes are exactly those of today's sync ragged
   path. Reconciliation stays exact (§2). Trim is skipped when the batch has structured-output requests (the engine's
   bitmask is K-wide).
4. **Hybrid** (native MTP present): no trim. Native drafts every row at K (FULL graphs kept). A row is replaced
   by the suffix only when the resolve yields ≥ `SUFFIX_HYBRID_ASYNC_HYB_MIN` real drafts (default K, so suffix and
   native tokens are never stitched). In probabilistic mode only temperature==0 rows are replaced. TP>1 defaults to
   broadcast: rank 0 computes, and every rank broadcasts the `[n,K]` rows on the device on every real step.
5. **sched_sync**: with the gate on it does not force sync. It drops `disable_padded_drafter_batch=True` (V2 ignores
   it otherwise) and respects an explicit `--no-async-scheduling`. The drafter works under sync too, since the trim
   applies to the K-wide stock rows.

### Options considered for rows with no or misaligned suffix

| option | verify cost | verdict |
|---|---|---|
| pad every row to K with filler (`SUFFIX_HYBRID_ASYNC_TRIM=0`) | K wasted tokens per miss row per step. Measured on gemma suffix-only at k=8: width-0 rows still paying the verify gave ITL +10..+36 %/token (MINED-LESSONS 2026-09-24); NVFP4-FA2 9-token verify ≈100x decode on the same KV (hd512) | kept as an A/B knob only (FULL graphs) |
| pad with native MTP drafts | free when native already runs K-wide | used in hybrid mode |
| device-side trim (AV-style, CPU upper bound ≠ device) | proportional to real drafts | needs `supports_device_cpu_query_lens_mismatch` on every target backend. FlashInfer on SM120 plans off `qo_indptr_cpu` (that is why upstream returns False), so declaring it on gemma's FlashInfer/K2 path would be wrong until the plan reads device offsets. GDN opts out. Not done |
| **host-exact trim to w** | proportional to host-known candidates. Residual waste only on rows whose candidate misaligns on device | **chosen** |

**Residual cost vs sync.** No-candidate rows cost nothing extra. A candidate row that misaligns (the target diverged
from C at step N+1) is verified at width w with filler, and its fresh lookup comes one step later than in sync.
So each divergence costs one wasted w-token verify plus one step without drafts. On long verbatim runs
(the regime where suffix wins) misalignment is rare. The live meter is the stats line
(`SUFFIX_HYBRID_LOG_INTERVAL`): `sched_drafts` (host-scheduled verify tokens) vs `real_drafts` (device-resolved,
read back with lag 1, no sync). The waste fraction is `1 - real/sched`.

**Graph mode.** Trimmed steps are PIECEWISE unless every row is at K, the same as the sync ragged path. Async
removes the host gap but not the piecewise cost. The orthogonal lever is capturing 1-token and `1+K` uniform graphs
(dynamic-SD capture path, `cudagraph_utils.py:265-291`) or varlen graphs (#57263 backport).

### Known limits
- The install still needs a `DraftModelSpeculator` (MTP checkpoint) for its `draft_tokens` buffer, the same as the sync
  suffix-only arm. A drafter-free model is not covered.
- Suffix-only at TP>1 relies on replay determinism. A rank-local exception (e.g. OOM) makes that rank trim to 0
  while others do not, so token counts diverge across ranks and collectives can hang. TP1 (gemma 26B) is not
  exposed. Do not arm suffix-only at TP>1 before a silicon soak.
- Corpus ingestion of a departing request misses its last step (≤ K+1 tokens), because the mirror is lag-1.
- qwen-flash QSA splits non-uniform decode rows out of its decode path (`models/qwen4_exp/common/qsa_cache.py:657-680`,
  `require_uniform=True`), so use hybrid mode (no trim) there, never suffix-only.
- PP>1, probabilistic suffix-only, and k > 31 (2k+1 > Rust cap 64) refuse install and stay native.

## 5. Arming on a dev pool (never prod)

Env (in addition to the pool's existing plugin env):

```
SUFFIX_HYBRID_WRAP=1
SUFFIX_HYBRID_ASYNC=1
SUFFIX_HYBRID_SUFFIX_ONLY=1        # gemma suffix-only arm; omit for hybrid (qwen-flash/GLM)
SUFFIX_HYBRID_LOG_INTERVAL=200     # async stats line: sched_drafts vs real_drafts
VLLM_GPU_SYNC_CHECK=warn           # optional: vLLM flags unexpected syncs in execute_model/sample_tokens
```

Args: keep the pool's `--speculative-config` (e.g. gemma-spec-dev: method mtp, assistant model,
`num_speculative_tokens` 8, `draft_sample_method` greedy). Do **not** pass `--no-async-scheduling`.
`disable_padded_drafter_batch` may stay, because the gate neutralizes it, but removing it is cleaner.
`SUFFIX_HYBRID_D2H_PIPE` / `W0_FASTPATH` have no effect on this path.

Engagement lines to grep:
- `sched-sync: armed in ASYNC mode`
- `sched-sync: ASYNC gate, sync NOT forced async_scheduling None -> scheduler_config.True`
- `suffix_hybrid v2 ASYNC installed mode=suffix-only ... async_scheduling=True`
- periodic `suffix_hybrid async {...}` with `real_drafts > 0` on repeat traffic

## 6. Verified vs needs silicon

Verified on CPU (Rust proposer real, CUDA faked): gate off leaves install_v2 and sched_sync unchanged.
Gate on, propose makes no host sync except the previous-step event. Every staging copy is non-blocking, and
host reads come only from pinned staging. Draft contents for hit, miss, offset (o=K+1), short-candidate,
misaligned, re-indexed and new rows are covered, as are trim semantics, hybrid select, the temperature gate,
broadcast discipline and the config path. Mutation-checked: removing the alignment check, mutating the
SchedulerOutput in place, a blocking H2D, a same-step event sync, or dropping the temperature gate each fail a test.

Needs silicon: real CUDA stream ordering and UVA visibility, the cost of the resolve kernels (a Triton fusion
is the upgrade if launch cost shows), the `VLLM_GPU_SYNC_CHECK=warn` silence, the waste fraction on real traffic,
end-to-end ITL/TPS under the A/B gate, gemma FlashInfer/K2 behavior with the trimmed ragged batches under async
(the same shapes as the sync ragged arm, but now overlapped), and TP>1 replay stability.
