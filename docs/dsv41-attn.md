# DeepSeek-V4.1-Flash sparse-MLA attention (lane G): contract, kernels, KV formats, runbook

Blind lane (2026-10-19): no GPU, no cluster. Everything below that says PROVEN was run on CPU (Helion eager
mode, Helion's compiler front end without a GPU, torch references); everything that says SILICON is a ready-to-run
command with its pass condition. Estimates are labelled ESTIMATE with their derivation. Nothing here is a measurement.

## Report (brief format)

1. **Files** (all new, worktree `~/Documents/vllm-suffix-hybrid-dsv41-attn`, branch `dsv41-attn`, nothing committed):
   `suffix_hybrid/helion_kernels/dsv41_attn.py` (3 Helion functions + serving entry + lab ABI), `dsv41_attn_ref.py` (K1 contract,
   record writers/readers, exact f64 reference, cost + capacity model), `dsv41_attn_lab.py` (K4 harness), `suffix_hybrid/kernels/dsv41_attn.py`
   (K5 hook), `suffix_hybrid/kernels/dsv41_attn_capture.py` (boot gate `dsv41_attn_capture_replay` + CLI),
   `tests/test_dsv41_attn.py`, `tests/test_dsv41_attn_helion.py`, `docs/dsv41-attn.md`.
2. **Tests** (`/tmp/lane-c/venv/bin/python -m pytest`, `PYTHONPATH=<worktree>`): `tests/test_dsv41_attn.py` -> `39 passed in 1.86s`;
   `tests/test_dsv41_attn_helion.py` -> `63 passed in 40.02s`; together with the neighbouring helion suites
   (`test_helion_ds_mla/runtime/k2_attn/dsa_indexer`) -> `167 passed in 59.42s`; `pyflakes` clean.
3. **Gates**: K1 contract PROVEN (byte-level, 3 records + row views). K2 PARTIAL: Helion front end lowers each function to one Triton
   kernel, 4 `tl.dot`, bf16 K tile, no atomics/barriers; ptxas/registers/spills UNKNOWN (silicon). K3 PROVEN on CPU eager (adversarial,
   both record pairs, all split counts). K4 harness + FlashInfer baseline READY, needs silicon. K5 hook + anchors (verified against the pinned
   source text) + per-layer log + refusals PROVEN on CPU; the real CUDA-graph backend of the boot gate untested.
4. **Env**: `SUFFIX_HELION_DSV41_ATTN=oxide|helion|auto` (default oxide), `SUFFIX_DSV41_ATTN_ROUTE=auto|row`, `SUFFIX_DSV41_ATTN_GATE=0` (skip boot gate).
5. **Handoffs**: sitecustomize hook lines, `docs/helion-kernels.md` row, preset only after A/B (section 9); the NVFP4-KV enabling patch (section 6);
   lab shim `from suffix_hybrid.helion_kernels.dsv41_attn import KERNELS`; the autotune pull must not ship `dsv41_attn_partial` rows on the lab
   leaderboard alone (the lab times the partial without the merge): use `dsv41_attn_lab` for the ship decision.
6. **Risks / verify first on silicon**: section 8 (compile + registers + spills of all three functions; Helion per-call host cost x2 launches in the
   eager gap; async gathers; real CUDA-graph gate; V4.1 pools need block stride % 576 / 528 / 288 = 0).
7. **Expected effect** (ESTIMATE, nothing measured): decode attention at c32 is 0.7-2.1 ms per step by the roofline model (section 1); I do NOT claim
   a speed win over FlashInfer. The only derived gain is the NVFP4 compressed pool: -42% gathered bytes per row and -41% KV bytes per context token
   (capacity x1.25..x1.70 at 512k), subject to the quality gate.

## 1. What runs today (from vLLM `39983a5e7` and `boot5.log`)

* Target model path on SM120: `vllm/models/deepseek_v41/nvidia/model.py` picks `DeepseekV4FlashInferSM120Attention`
  (backend name `FLASHINFER_MLA_SPARSE_DSV41`, the plugin's `SUFFIX_DSV41_SM120_BLOCK64` declares 64-token kernel blocks).
  `boot5.log`: "Using DeepSeek's fp8_ds_mla KV cache format", "Warming up DeepSeek V4 sparse MLA attention",
  "FlashInfer SM120 sparse MLA DSv4.1 decode autotune" (FlashInfer 0.7.2). Block size 64, 8 local heads (64/TP8), 1 KV head.
* Record on SM120: the **V4 584 B record** (`_use_v41_mxfp8_kv_record()` is true only for SM100, so SM120 never uses V4.1's
  528 B MXFP8 record). One call per layer: `flashinfer_trtllm_batch_decode_sparse_mla_dsv4(query, swa_kv_cache,
  sparse_indices=SWA slots [T,1,128], swa_topk_lens, compressed_kv_cache, extra_sparse_indices=top-k slots [T,1,512],
  extra_sparse_topk_lens, sinks, kv_layout="NHD", bmm1_scale=512**-0.5)`. Decode and prefill chunks use the same call.
* FlashInfer plans decode split-K only for `num_tokens <= 64` (`_DECODE_MAX_TOKENS`); batch 32 x 6 spec rows = 192 tokens
  therefore runs its **prefill-family kernels** (swapAB / SG / MG), not the decode kernel. The decode kernel quantizes q
  (NoPE dims) and P to **fp8** and keeps RoPE in bf16 (`kernels/fp8_decode/decode.cuh`: `quantize_q_to_smem`, fp8 `W`).
  Our kernel keeps q and every dequantized K element exact in bf16 (see section 3), so it is more accurate than the baseline.
* Layer mix (config.json `compress_ratios`): layers 0,1 = SWA only; 2..19 ratio 2 (18 layers); 20..39 ratio 1 (20 layers);
  3 DSpark draft layers (40..42): SWA only with the non-causal window `cdiv(128+5, 64)*64 = 192`. Compressed pools and
  indexer pools exist only on `kv_source_layer_ids` (2, 8, 14, 20); the other compressed layers read those pools.
* Gathered per q row per layer (per TP rank; every rank reads the FULL kv, there is one kv head): 128 SWA rows (+512 selected
  compressed rows on 38 layers) x 584 B = 74.8 KB (SWA only) / 373.8 KB (compressed layers). Per q row per forward over
  43 attention layers: **14.59 MB** (`decode_bytes_per_row`). Batch 32 x 6 rows = 192 rows.

### Cost model (ESTIMATE; `dsv41_attn_ref.decode_step_model`, hardware constants in `HW`)
Constants (assumptions): HBM 1792 GB/s x 0.75 achievable, bf16 mma.sync 250 TFLOPs sustained, 40 TOPS elementwise.
Per decode step, one rank, V4 584 B records:

| batch x rows | DRAM bytes (no row reuse / 80% / full reuse) | t_dram ms | t_mma ms (M padded 8 -> 16) | t_dequant ms |
|---|---|---|---|---|
| 1 x 6 | 87 / 29 / 15 MB | 0.065 / 0.022 / 0.011 | 0.020 | 0.015 |
| 8 x 6 | 700 / 233 / 117 MB | 0.52 / 0.17 / 0.09 | 0.157 | 0.123 |
| 32 x 6 | 2800 / 934 / 467 MB | 2.08 / 0.70 / 0.35 | 0.629 | 0.491 |

Reading: at c32 the three terms are the same order, so a good kernel is balanced; the lever that is NOT in the table is
**row reuse**: the six spec rows of a sequence read near-identical SWA windows (127/128 shared) and, if the indexer's top-512
sets overlap, near-identical compressed rows. Scheduling consecutive rows as adjacent programs (the kernels do) lets L2
(128 MB) serve the repeats. At c1 the attention is latency bound: ESTIMATE 43 layers x ~10 us (two launches + one gather
round trip each) ~ 0.4 ms/step, far above the bandwidth term. The attention share of a step is UNKNOWN (no profile exists).

## 2. K1 contract (`suffix_hybrid/helion_kernels/dsv41_attn_ref.py`, module docstring is normative)

q [T,8,512] bf16 (RoPE already applied, last 64 dims); logits `512**-0.5 * q.k` over the FULL cached 512-dim row; value = the
same row; per-head sink = one extra softmax column of logit `sink` (NOT scaled) with a zero value; SWA list [T,W] and
compressed list [T,512] of global slots (`-1` masked, `len[t]` live prefix) share ONE softmax; no live column -> output
exactly 0 (also with a finite sink). Records (uint8 pools, block stride in bytes = `stride(0)`, must be a whole multiple of 576 /
528 / 288 so the kernels can address rows; vLLM aligns pages to 576 B and V4.1 blocks to lcm(528,288) = 3168 B):

| record | bytes | layout |
|---|---|---|
| `REC_V4` (serves today) | 584 | page = `[page*576 data \| page*8 scales \| pad]`; token: 448 fp8 e4m3 + 128 B bf16 RoPE; scales: 7 UE8M0 (one per 64 NoPE dims) + 1 zero byte; scale = 2^(byte-127) |
| `REC_FP8` (V4.1 MXFP8, vLLM `nvfp4_ds_mla` SWA pool) | 528 | interleaved: 512 fp8 (RoPE included) + 16 UE8M0 (one per 32 dims) |
| `REC_FP4` (V4.1 NVFP4, compressed pool only) | 288 | interleaved: 256 e2m1 bytes (even dim = low nibble) + 32 e4m3 scales (one per 16 dims, `amax/6` clamped to [2^-9, 448]); RoPE dims are fp4 too |

**Finding: FlashInfer 0.7.2 stores the V4.1 records as page FOOTERS** (data of the whole page, then the scales:
`dsv41_{fp8,fp4}_quantize_pack_sparse_mla_cache`) whereas vLLM writes them interleaved per token. They are NOT byte
compatible, so FlashInfer's `fp8_dsv41` / `fp8_dsv41_fp4_ca` readers cannot read a vLLM-written V4.1 pool. `to_flashinfer_footer`
converts for the K4 baseline; the shim refuses to fall back to FlashInfer for V4.1 pools. `REC_V4` is identical in both.
PROVEN: byte-level layout tests, row-view addressing tests (`tests/test_dsv41_attn.py`).

## 3. Kernels (`dsv41_attn.py`; three Helion functions, no cross-CTA synchronisation)

Owner rule applied: **no grid barrier, no semaphore, no last-arriving-split merge**. We verified why: in Helion 1.5.0 two
dependent top-level loops require `hl.barrier()` (`helion.exc.LoopDependencyError`, found by running the front end on a
fused partial+merge body), and `hl.barrier` forces a persistent kernel with an in-kernel grid barrier
(`helion/language/barrier.py`, `exc.BarrierRequiresPersistent`; `autotuner/config_spec.py`: "the grid barrier ... waits for
every CTA ... needs every CTA resident at once") = the deadlock. So dependent stages are separate Helion functions, i.e.
separate launches:

* `dsv41_attn_partial` (T <= 1024): grid (token, split). Split s walks the s-th slice of the SWA list, then of the
  compressed list, one online softmax across both pools; stores bf16 partial + f32 base-2 lse (`torch.empty` inside = torch's
  allocator) and returns `(o_part, lse_part, sink)`. Tunables: `num_splits` 1..32, `bk` 16..64, `num_warps`, `num_stages`,
  `loop_orders`, `l2_groupings`, load eviction policies.
* `dsv41_attn_merge`: grid (token); lse merge of S partials plus the sink. Fixed default config (`block_sizes=[512]`, 4 warps):
  traffic S x 8 KB per token. Not a lab entry (its cost is part of the split-count trade-off, timed by `dsv41_attn_lab`).
* `dsv41_attn_row` (all T up to 8192): one kernel, grid (token), no partials: the prefill body and the decode alternative.

Numerics: every dequantized K element is exact in bf16, QK = bf16 x bf16 -> f32, P rounded to bf16 for P.V (V == K), softmax
state / lse / merge f32, one bf16 output rounding (+ one per split partial). Measured on CPU against the f64 reference:
row rel-L2 2-4.5e-3 (gate 2e-2), both record pairs, every split count. Deterministic (bit-exact repeats; tested).

Memory / register discipline (lessons of the glm-int Triton MLA, 255 regs / 78-138 spills): pools are read through **row views**
(`[rows,576]` fp8+RoPE bytes, `[rows,8]` scales, an `int16` view for the RoPE, `[rows,528]`, `[rows,288]`) so every gather is a
contiguous 16-byte-vectorizable slice; the K tile is dequantized once in bf16 (no f32 `[bk,512]` tile; asserted on the
generated Triton), q is loaded once as the A operand; the tile (16..64 rows), warps (4/8) and stages are searched, small tiles first.
KV bandwidth-delay product per SM is ~6 KB (ESTIMATE: 600 ns x 1792 GB/s / 188 SMs), one bk=16 tile is 9.3 KB, so latency is
hidden by occupancy; `num_stages` lets Triton's pipeliner hoist the next tile's loads, and Helion adds per-loop
`range_num_stages / range_unroll_factors / range_multi_buffers / range_flattens` (Triton `tl.range` kwargs) to the config
spec whenever Triton is importable (`language/loops.py::_add_config_range_choice`; the Triton-less front-end run on the Mac
cannot show them) so each `tile_k` loop is pipelined separately by the LFBO search. Whether Triton emits `cp.async` for
loads that feed ALU (not a dot) must be read from the IR on silicon, see section 8.

Search spaces: T buckets decode (1,2,4,6,8,12,16,24,32,48,64,96,128,192) + prefill (256..8192; partial stops at 1024);
layer shapes (swa,topk) = (128,512), (128,0), (192,0); records v4 / v41; **context class** c16k / c128k / c512k on the decode
buckets of compressed layers (pool footprint decides L2-resident vs DRAM-random gathers, so LFBO searches `num_splits` per
class). The class is a host-known hint (`max_seq_len` of the compressed-cache metadata, a Python int, no sync); correctness
never depends on it. Lab cases: partial 158, row 176.

PROVEN (CPU): 63 Helion tests (interpret mode, front end, boot-gate logic) (adversarial rows, all splits/tiles/heads/widths, both record pairs, NaN-poisoned rows,
sink variants, teeth tests) + Helion's compiler front end accepts every body and lowers each to exactly one Triton kernel
with 4 `tl.dot`, no atomics/barriers, no f32 K tile. NOT proven: Triton/ptxas compile, registers, spills, speed.

## 4. Capture safety and the boot gate `dsv41_attn_capture_replay`

Rules enforced in code: scratch only from `torch.empty` inside the kernels (torch's allocator, graph-pool safe) or load-time
persistent buffers in the route (`neg`, `zero`, `sink_none`, dummies); nothing allocated on the call path (tested by patching
`torch.zeros/full/empty/...` to raise); no `.item()`/H2D/sync; lens must be int32, q/out unit last stride (a new dtype/stride
would compile a new Triton variant at serving time). Triton also compiles a new variant when `T == 1`, `T % 16`, or a pool
block stride `% 16` changes: `load()` warms every bucket at `warm_sizes` (upper edge, lower edge + 1, first multiple of 16),
and `attend` raises (loudly) if a CUDA graph capture reaches a specialization no eager call has compiled (the real pool strides
are warmed by the first eager step; vLLM runs warmup forwards before capture). Layer `__init__` loads the geometries a layer
can see (decode width and the 128-wide prefill lists); a geometry first seen inside a capture raises.

Gate (`suffix_hybrid/kernels/dsv41_attn_capture.py`): per kernel/bucket/context class at the bucket's edge T values: eager warm ->
capture on a side stream (sync-debug "error" inside) -> replay x trials with the static buffers overwritten in place (q, both
index lists, live lengths including 0 and full, sink, pool bytes), outputs pre-filled with NaN -> replay must equal the eager twin
BIT-EXACTLY, be finite, pass K3 against the f64 reference, and leave `memory_allocated` and the output pointer unchanged.
Run at boot by the hook (`SUFFIX_DSV41_ATTN_GATE=0` skips) or by hand, also under the sanitizer:

    python -m suffix_hybrid.kernels.dsv41_attn_capture --small                      # default configs, no autotune needed
    compute-sanitizer --tool memcheck --print-limit 10 python -m suffix_hybrid.kernels.dsv41_attn_capture --small --buckets 6,12,32
    compute-sanitizer --tool racecheck  python -m suffix_hybrid.kernels.dsv41_attn_capture --small --buckets 6
    compute-sanitizer --tool initcheck  python -m suffix_hybrid.kernels.dsv41_attn_capture --small --buckets 6 --rec v41 --shape 128:512

PROVEN: the gate's logic on CPU with an eager stand-in backend (healthy route passes; a replay that writes nothing, state that
differs between capture and eager, unwritten rows, a wrong-but-deterministic kernel and a capture exception are all caught).
SILICON: the real CUDA-graph backend (`CudaBackend`) is untested.

## 5. K5 hook (`suffix_hybrid/kernels/dsv41_attn.py`)

Seam: the module global `flashinfer_trtllm_batch_decode_sparse_mla_dsv4` of `vllm.models.deepseek_v41.nvidia.flashinfer_sparse`
is replaced by a shim with the same keyword signature (none of vLLM's metadata code is copied). Exact-text anchors in
`_forward_decode`, `_forward_prefill`, `forward_mqa` are count-verified (PROVEN against the pinned source). `__init__`/`forward_mqa`
are wrapped (route loading before capture, per-layer path log at first call, host ctx hint). An unknown call form raises; in auto
mode FlashInfer fallbacks are counted per (layer, bucket) and logged once, never for V4.1 pools. Env: see section 9.

## 6. NVFP4 KV cache for V4.1 (owner rule: NVFP4 everywhere)

* The upstream record exists: `nvfp4_ds_mla` = MXFP8 528 B SWA pool + NVFP4 288 B compressed pool, both writers in vLLM
  (`quantize_and_insert_k_cache`, `_rope_quant_insert_nvfp4_kernel`, Triton, PTX `cvt.rn.satfinite.e2m1x2.f32`) and a bf16
  dequant-gather. Upstream gates it to SM100 (FlashMLA). Contract is crisp -> **the reader is implemented and CPU-proven**
  (`rec` 1/2 in both bodies, oracle against vLLM's exact writer semantics); the enabling patch is a ready design, not code:
  1. `vllm.models.deepseek_v41.attention._use_v41_mxfp8_kv_record` -> true on SM12x (module-global call, post-import hook);
  2. `DeepseekV4FlashInferMLASparseBackend.supports_combination` (major == 12 branch) and `supported_kv_cache_dtypes` accept
     `nvfp4_ds_mla`; serve `--kv-cache-dtype nvfp4_ds_mla`;
  3. confirm the C++ SWA writer `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(..., kv_mxfp8=True)` is in the SM120 build and
     that Triton targets `sm_120a` for the e2m1 PTX;
  4. skip FlashInfer's sparse-MLA warmup for these pools; our hook already routes the pair (`rec` from `swa_bytes_per_token` 528 /
     `compressed_bytes_per_token` 288) and refuses any FlashInfer fallback.
* Sizes (per rank, bottom-up over the 4 kv-source compressed pools + 4 indexer pools, `kv_bytes_per_ctx_token`): V4 584 B ->
  1790 B/ctx token; with the 288 B compressed record 1050 B (-41%); with an FP4 indexer pool (68 B instead of 132) 890 B.
  Per sequence at 512k: 938 MB -> 551 MB (-41%) -> 467 MB. Gathered bytes per row: 14.59 MB -> 8.52 MB per step (-42%).
* Capacity gain at 512k: `boot5.log` reports 37.17 GiB -> 10,867,561 tokens = 3672 B/ctx token aggregate, of which the bottom-up
  model explains 1790 B (the rest is allocator/other groups: unmodelled). ESTIMATE: capacity x1.25 if the remainder is fixed,
  up to x1.70 (x2.01 with the FP4 indexer) if it scales like the modelled pools; the boot log of a real nvfp4_ds_mla roll gives the number.
* Quality risk (the reason this needs the gate): fp4 rows have ~3x the per-row error of fp8 (CPU, Gaussian heavy-tailed rows,
  rel-L2 0.10 vs 0.037) and RoPE dims are quantized too; on synthetic attention the output error is ~3.5x fp8's
  (0.16 vs 0.045 at q-scale 1; 0.58 vs 0.18 at q-scale 4). Real KV is structured, so this is only a warning.
* Gate plan (AGENTS section 5, `--tier full` because it is an approximation-class change, drift on the raw path): K3 reader oracle
  (done on CPU) -> K4 vs FlashInfer on repacked pools (`dsv41_attn_lab --rec v41`) -> rank-slice integration check on one GPU
  (capture/replay gate with `--rec v41`) -> prod gate on the whole-node pool with the owner's go and a downtime window
  (A1 = fp8_ds_mla, B = nvfp4_ds_mla + this reader, A2 = rollback; speed c1/8/32 x 5 rounds, quality full, drift KL < 0.02, ppl ratio < 1.02,
  needle at 512k, multimodal intact). Expect arm drift vs A1 to include FlashInfer's own fp8 q/P noise: measure A1-vs-A2 first.

## 7. Prefill (8192-token chunks)

Implemented: the same bodies with T up to 8192 (row body for T >= 256 by default; partial+merge up to 1024). Cost model ESTIMATE per 8k
chunk and layer: compressed gathers 8192 x 512 x 584 B = 2.45 GB (1.4 ms at the achievable HBM rate if top-k sets do not overlap
between adjacent tokens; SWA windows overlap 127/128 so they are L2 hits), MMA 172 GFLOP padded (0.69 ms), dequant 0.5 ms; x38
layers = 26-55 ms per chunk. Designed, NOT implemented (D=512 caps M at ~32 rows per CTA, 99 KB smem): (a) SWA band tile, 4 tokens x
8 heads = 32 rows share one K tile (4x fewer SWA gathers/dequant, no 2x M padding); (b) row-grouped compressed tile (6 spec rows =
48 = 3 x m16, zero padding) that gathers the UNION of the rows' top-k once. (b) pays only if adjacent rows' top-k sets overlap:
**silicon probe first** - dump `topk_indices_buffer` of one request and run `dsv41_attn_ref.overlap_stats(idx, group=6)`
(`union_over_total` 0.17 = identical sets, 1.0 = disjoint; build (b) only below ~0.4).

## 8. Silicon runbook (what to verify first, in order)

1. `python -m suffix_hybrid.kernels.dsv41_attn_capture --small` in a kernel-lab pod (one RTX PRO 6000, the serving image): proves
   Triton/ptxas compile of all three functions at both record pairs, K3, capture/replay bit-exactness. Then the three sanitizer runs above.
   Read `n_regs`/`n_spills` of the compiled kernels (Triton compile metadata or `ptxas -v`): any spill at bk=16 -> try 8 warps,
   then the alternatives below.
2. `python -m suffix_hybrid.helion_kernels.dsv41_attn_lab --rec v4 --kernels partial,row --search --out ...` (hand grid, ships only
   composite wins on warm AND cold L2 with K3 PASS, writes `configs/<kernel>/<platform>.json`, `table.csv` with host us per call and
   achieved GB/s, `step_estimate.txt`). Then the autotune lab (Autotune tab: `from suffix_hybrid.helion_kernels.dsv41_attn import
   KERNELS`; the partial entry is timed WITHOUT the merge, so its leaderboard is optimistic - decide shipping with the harness).
3. Host overhead: the attention runs in the eager gap of vLLM's breakable graphs; Helion's per-call Python cost (two calls on the
   split path) is in `table.csv` (`best_host_us`); if it exceeds ~30 us per call, capture the pair per bucket in an own CUDA graph.
4. IR check: does the K-gather loop contain `ttg.async_copy_global_to_local` (`MLIR_ENABLE_DUMP=1`)? If not (loads feeding ALU, not
   dot, are likely not async-copied), variants in order: (i) `num_stages` 2-3 with bk=16; (ii) software prefetch of the next tile's raw
   bytes as loop-carried registers (9 KB at bk=16); (iii) K as the A operand (`S^T = K q^T`, one shared K tile serving both products).
5. Rank-slice integration check on one GPU (AGENTS 3, step 5) with `SUFFIX_HELION_DSV41_ATTN=helion`, then the prod gate (owner go).

## 9. Env knobs and handoffs

* `SUFFIX_HELION_DSV41_ATTN` = `oxide` (default; FlashInfer, module never imported) | `helion` (all calls ours, missing config or unknown
  call form = error) | `auto` (ours where a lab-won config is shipped, FlashInfer elsewhere, never for V4.1 pools).
* `SUFFIX_DSV41_ATTN_ROUTE` = `auto` | `row` (row body only, no partials).  `SUFFIX_DSV41_ATTN_GATE=0` skips the boot gate.
* Handoff sitecustomize.py (not my file), next to the block64 hook and BEFORE `SUFFIX_HYBRID_WRAP`:

      if os.environ.get("SUFFIX_HELION_DSV41_ATTN", "oxide").strip().lower() not in ("", "oxide"):
          from suffix_hybrid.kernels.dsv41_attn import install_post_import_hook as _dsv41_attn_hook
          _dsv41_attn_hook()

  Preset `dsv41-sm120` gets `SUFFIX_HELION_DSV41_ATTN=auto` only after the A/B passes. `docs/helion-kernels.md` family table row:
  `dsv41_attn_{partial,row}` | FlashInfer SM120 sparse MLA | `kernels/dsv41_attn.py` | `SUFFIX_HELION_DSV41_ATTN`.
* Lab `replaces` regexes (`sparse_mla_sm120`, `cake_sparse_mla`, `sm120_sparse_mla`) are guesses until a profile of the model exists.
