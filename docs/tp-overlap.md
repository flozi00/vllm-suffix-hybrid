# TP micro-batch overlap (hide all-reduce behind compute)

Status 2026-09-30: **pipeline core + proofs + on-GPU oracle/bench done; vLLM
runner integration NOT done** (section 7). `SUFFIX_TP_OVERLAP=1` only logs that
serving is not wired. Code: `suffix_hybrid/tp_overlap.py` (core),
`suffix_hybrid/tp_overlap_bench.py` (toy DeepSeek-V3.2 stack, oracle, bench),
`tests/test_tp_overlap.py` (CPU proofs).

## 1. Problem

Prod GLM-5.3 decode (TP=8 + EP, PCIe, no P2P, MTP k=5 -> 6 tokens/request):
step 25 / 75 / 124 / 140 ms at c1 / c4 / c16 / c32. ~160 bf16 all-reduces per
step (o_proj + MoE output per layer x 78 + MTP), tokens x 6144 x 2 B, through
host memory. At c32 (192 tokens, 2.25 MiB per all-reduce, ~0.45 ms each) that
is ~72 ms of comm serialized with ~68 ms of compute.

vLLM 0.30.0 DBO (`docs/design/dbo.md`, V2: `v1/worker/gpu/ubatch_utils.py`,
`v1/worker/ubatching.py`) splits the batch into two micro-batches run on two
Python threads that ping-pong at `dbo_yield` points. It is built for DP+EP with
DeepEP all-to-all: the runner only builds it for `data_parallel_size > 1`
(`maybe_build_ubatch_runner`), the config asserts a DeepEP/nixl all2all backend
(`config/vllm.py`), the only yield points are in DeepEP prepare/finalize, and V2
lists "dual batch overlap with speculative decoding" as unsupported.

## 2. Design choice

A decoder layer is `[norm, attention -> partial] AR [norm, MLP/MoE -> partial]
AR`; DeepseekV32DecoderLayer leaves o_proj and the MoE un-reduced and fuses the
AR into the next norm (`fused_allreduce_rms_norm`, which off the FlashInfer
fused path -- never taken without P2P -- is `norm(all_reduce(h), residual)`).
So the stack is a chain of **segments** separated by all-reduces.

**(b) single-thread per-layer software pipeline (chosen).** `pipelined_forward`
runs segment k of micro-batch 0, issues its AR on one comm stream, runs segment
k of micro-batch 1 (overlapping that AR), issues its AR, then segment k+1 of
micro-batch 0 after waiting on its AR, ...

```
compute: seg(k,A) rec ready_A | seg(k,B) rec ready_B | wait done_A seg(k+1,A) ...
comm:    wait ready_A AR(k,A) rec done_A | wait ready_B AR(k,B) rec done_B | ...
```

vs **(a) reuse vLLM's ubatch threads with yields at the TP ARs**:

| | (a) DBO threads | (b) segment pipeline |
|---|---|---|
| collective order | implicit: thread hand-off discipline; any AR anywhere in the forward yields | explicit program order; the adapter lists every overlapped AR |
| graph capture | 2 Python threads parked around capture, forward-context global swapped across threads | ordinary single-thread capture, one fork/join per AR |
| failure | a dying micro-batch hangs its sibling (vLLM's own comment in `UBatchRunner.begin_capturable_run`) | exception propagates |
| model code | none | GLM adapter re-implements the layer forward (~30 lines), anchored against the pinned source (fail closed on drift) |
| eager overhead | 2 thread hand-offs per AR (~320/step) | none |
| MoE on whole batch ("hybrid") | impossible (each thread owns its rows) | `joint` segments |

Both need the same runner work (per-micro-batch attention metadata, dispatch at
DP=1, graph descriptors) and the same model-state fixes (H1/H2 below), so (b)'s
only extra cost is the adapter; it buys determinism and debuggability.

### Two schedules

* **split**: every segment per micro-batch. Hides (ideally) all comm, but a
  decode MoE is expert-weight-bandwidth bound and both halves activate ~95 % of
  the experts at c32 (1-(1-8/256)^96), so the expert weights are streamed twice.
* **hybrid** (`joint_mlp=True`): attention per micro-batch, MLP/MoE once on the
  concatenated rows (no re-stream), its AR still issued per micro-batch. Per
  layer AR(o, A) hides behind attention B and AR(moe, B) behind attention A of
  the next layer; AR(o, B) and AR(moe, A) stay exposed: ~half the comm hidden.

## 3. NCCL ordering (deadlock freedom)

* Every overlapped AR, whatever `nccl_split` band communicator it picks, is
  enqueued on **one** comm stream in program order -> identical order on every
  rank, and at most one NCCL kernel per GPU in flight (two communicators never
  run concurrently, which is what can deadlock on shared GPUs).
* Collectives outside the pipelined region (vocab-parallel embedding AR,
  lm_head gather) run on the compute stream strictly before the first `ready`
  record / after the last `done` wait, i.e. ordered against the comm stream.
* Nothing inside a segment may issue a TP collective (true for GLM at DP=1:
  MLA o_proj / MoE / shared expert are un-reduced; DSA attention all-gathers
  only under PCP/DCP). The CPU test counts the synchronous ARs during the
  pipelined forward: exactly one (the embedding).
* QAR (`SUFFIX_NCCL_QAR`) is refused: it allocates on the calling stream.

## 4. CUDA graphs, allocator, numerics

* compute = the stream current while the driver runs (the capture stream);
  comm = one persistent side stream; 4 persistent events (ready/done x micro-
  batch), each waited on before it is re-recorded. Every comm fork is joined
  (compute waits `done` before the next segment / the final norm).
* Allocator safety without `record_stream`: the AR output is allocated on the
  compute stream before the fork; the partial input stays referenced until the
  compute stream has waited `done`.
* Numerics: each micro-batch runs exactly the unsplit forward's ops on its rows.
  Bit-identical to the unsplit forward iff every kernel is row-invariant
  (proved on CPU); always bit-identical to the same schedule serialized (comm
  stream = compute stream), which is the on-GPU oracle's gate. On GPU the
  split changes GEMM M (cuBLAS/NVFP4 tile choice) and NCCL message size
  (reduction order at TP>2), so tokens can differ in the last bits vs unsplit --
  same class of difference as a different batch size today.

## 5. Model-state hazards found (must be fixed before serving)

* **H1 DSA IndexShare rows.** An indexer layer writes `topk_indices_buffer[:n]`,
  follower layers read it several segments later; micro-batch B runs in
  between and overwrites rows 0..n_B. Same for each `SparseMLAIndexGroup`'s
  `physical_topk_indices` / `valid_topk_counts` (converted once by the group
  leader, reused by followers). Fix implemented: `rows_from` re-points these
  tensors in place (`Tensor.set_`, same object, no copy) at the micro-batch's
  own rows; `cross_layer_buffers(model)` collects them. The CPU test shows the
  hazard (wrong output without it) and the fix (bit-identical with it).
* **H2 HiSparse per-forward state.** `HiSparseMLAIndexGroup` uses
  `request_ids = arange` (batch-local rows), `prepare_group_for_batch` resets
  per-forward fields and `begin_forward` resets `_swap_step`, whose
  `_step_rows` walk the shared physical-top-k workspace
  (`v1/hisparse/runtime.py:869`). Two interleaved micro-batches corrupt each
  other's swap rows. Not fixed: `glm_refusal` refuses HiSparse. Prod GLM runs
  HiSparse, so this is on the critical path.
* **H3 MTP drafter.** After a micro-batched target step the runner has no
  full-batch `attn_metadata` for `speculator.propose` (draft prefill reuses the
  target's), and the drafter's FULL graph reads builder-0's persistent
  buffers, which now hold micro-batch 0's rows. Needs a full-batch
  `prepare_attn` before `propose` (one extra metadata build per step).
* **H4 workspace manager.** Single thread -> `dbo_current_ubatch_id()` is 0 for
  both micro-batches (shared workspace). Safe iff no workspace view outlives a
  segment; verify for the NVFP4 MoE / DS-MLA kernels.

## 6. What is proven (CPU, `tests/test_tp_overlap.py`)

`FakeStreams` defers every launch and executes in a random order that keeps
only what a GPU guarantees (per-stream FIFO, event waits). On a shrunk
DeepSeek-V3.2-like TP stack (vocab-parallel embedding, MLA-style attention with
column/row-parallel projections over a per-request KV cache read through a
per-micro-batch forward context, DSA indexer with IndexShare followers via
`hisparse_mtp_patch.oracle.index_leaders`, sigmoid top-k MoE with EP=TP experts +
TP-sharded shared expert, dense first layer):

1. overlapped == unsplit == split-sequential, bitwise, split and hybrid, 12
   random interleavings, 12/48/96 tokens; at **TP=2 over gloo** with different
   interleavings per rank (rank-local shards, real all-reduces);
2. collective log identical on both ranks and equal to the program order
   `[(segment, micro-batch, shape)]`; only the embedding AR is synchronous;
3. dropping the compute wait on `done` or the comm wait on `ready` gives wrong
   outputs (the harness has teeth); IndexShare hazard reproduced and fixed;
4. `plan_split` fallbacks: gate off, < threshold (48), < 2 requests, prefill /
   mixed / ragged -> unsplit; split at the request boundary (odd -> ub0 +1);
5. anchors match the pinned `deepseek_v32/nvidia/model.py` (fixture), drift
   refused; refusals (TP 1, SP, PP, EAGLE3 aux, HiSparse); the band-communicator
   pick and QAR refusal; forward-context swap.

Not provable here: real stream concurrency, NCCL-in-graph on a forked stream,
and the speed. That is what the GPU gates are for.

## 7. Remaining (serving integration), in order

1. **Measure first** (section 9). Only continue if the bench shows a win.
2. V2 runner, DP=1: `maybe_build_ubatch_runner` refuses `dp <= 1`;
   `dispatch_cg_and_sync_dp` returns `num_ubatches=1` at DP=1 -> branch on
   `plan_split` (identical on all TP ranks: same scheduler output) and dispatch
   the `num_ubatches=2` descriptor; per-micro-batch metadata builders need
   `get_num_ubatches() == 2` in `attn_utils` (set `enable_dbo` in the worker
   after config validation, which asserts DeepEP).
3. Replace `UBatchRunner.begin_capturable_run`'s threads with
   `glm_forward(..., enter=vllm_enter(ubatch_state.forward_contexts,
   cross_layer_buffers(model), starts))`, returning it as the `finish`
   callable (capture path `ubatch_forward_fn` works unchanged); split at the
   request boundary (`create_ubatch_slices` splits at the token midpoint).
4. H2 (HiSparse micro-batch state), H3 (full-batch metadata before propose),
   H4 audit.
5. Boot the glm_stack oracles with the flag, then an A/B at c16/c32.

## 8. Expected step time at c32 (192 tokens)

Inputs: compute C ~68 ms, of which ~32 ms is NVFP4 expert-weight streaming
(assumed GLM-5.3 dims 256 experts, top-8, intermediate 2048: 32 experts/rank x
3 x 6144 x 2048 x 0.5625 B ~ 0.68 GB/layer x 76 MoE layers at ~1.6 TB/s;
consistent with the 25 ms c1 step, where only ~17 % of experts are active);
comm 160 x ~0.45 ms ~72 ms (3 MiB measured 534 us on worker-06). Half-batch AR (1.1 MiB) ~0.25 ms
**only with the nccl_split band for 0.4-2 MiB**; with NCCL's default protocol
it is ~1.2-1.6 ms on worker-06 and overlap would lose.

* split: compute ~68 + 29 (experts streamed twice at 95 % activation) + ~5
  (2x kernels) ~ 102 ms; comm 320 x 0.25 ~ 80 ms -> step ~ max + fill ~
  **~105 ms (-25 %)**.
* hybrid: compute ~71 ms + exposed 2 half-ARs/layer (78 x 0.5 ms ~ 39 ms) ->
  **~110 ms (-21 %)**.

Per-stream at c32 ~26 -> ~33-35 tok/s. The >= 50 tok/s target (step <= ~70 ms)
is **not reachable with TP overlap alone**: after the split the step is compute
bound (split) or half the comm stays exposed (hybrid). It needs comm volume cut
as well (e.g. hybrid + QAR on the exposed half-ARs -> ~90 ms, lossy) or a
different parallel layout. These numbers are estimates; the bench decides.

## 9. Pod validation protocol

Idle 8-GPU node, not prod (the gates take all GPUs before vLLM sizes memory):

1. `SUFFIX_BOOT_GATES=allreduce_matrix_bench` -> confirm the band/communicator
   that is fast at 0.5-1.1 MiB (half-AR sizes at c16/c32).
2. `SUFFIX_BOOT_GATES=tp_overlap_oracle` -> `[suffix tp-overlap] ORACLE PASS`
   for tp 2/4/8 x n 48/96/192 (overlapped eager + graph replay bitwise == same
   schedule serialized; serialized split == model forward per micro-batch).
   Any FAIL = stream/event or NCCL-in-graph problem: stop.
3. `SUFFIX_BOOT_GATES=tp_overlap_bench,tp_overlap_bench_ring_simple` ->
   `BENCH tp=8 n=192 ... best <schedule> saves X %` and the compute/comm
   columns (toy per-layer compute vs comm should resemble prod's ~1:1 at c32;
   if not, scale with `--layers`/widths by hand:
   `python -m suffix_hybrid.tp_overlap_bench bench --tp 8 --tokens 96,192,384`).
   Go: best schedule saves >= 15 % at n=192 with the router spec.
4. Only then section 7, and prod A/B with the owner's go.

## 10. Risks

* NCCL kernels on the side stream take SMs (channels) and the host-memory path
  may contend with HBM traffic: overlap efficiency < 100 % (bench measures).
* Smaller messages have a larger per-AR latency floor: 2x the ARs.
* HiSparse (H2) may need upstream-level changes to its per-forward state.
* The adapter duplicates the layer forward: anchored, fail closed on drift.
* Graph memory: micro-batched descriptors are extra FULL graphs.
