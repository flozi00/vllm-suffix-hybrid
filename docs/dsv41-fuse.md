# DeepSeek-V4.1-Flash lane F: fusing the small, launch-bound kernels (blind, 2026-10-19)

Branch `dsv41-fuse` (base `dsv41` @ d0fec38), nothing committed. Blind = CPU only: Helion bodies are proven
in Helion's own interpret mode (`HELION_INTERPRET=1`) against exact float64 references and lowered to Triton
source by the real Helion 1.5.0 front end; nothing here was compiled by ptxas or run on a GPU. **No speed number
below is a measurement.** Every "est." has its derivation next to it; K4 and K6 are the only thing that can say
whether any of this wins. Stock behaviour with every knob at its default is bit-for-bit unchanged.

## 0. What exists (all new files)

| file | role |
|---|---|
| `suffix_hybrid/helion_kernels/dsv41_common.py` | shared traced helpers (Sinkhorn, MXFP8 quantiser + F8_128x4 swizzle), exact quantiser reference, the `capture_replay_gate` boot-gate runner |
| `suffix_hybrid/helion_kernels/dsv41_hc.py` | mHC boundary: `dsv41_hc_post_partials` + `dsv41_hc_epilogue` (split2), `dsv41_hc_resident` (negative control), optional MXFP8 output |
| `suffix_hybrid/helion_kernels/dsv41_router.py` | `dsv41_gate_gemv` (replaces the GateLinear GEMM), `dsv41_gate_topk_resident` (fused negative control) |
| `suffix_hybrid/helion_kernels/dsv41_swiglu.py` | `dsv41_swiglu_quant`: shared-expert clamped SwiGLU + MXFP8 quant |
| `suffix_hybrid/helion_kernels/dsv41_engram.py` | `dsv41_engram_gather_packed` / `dsv41_engram_dequant_packed` (packed-row transport, design P1) |
| `suffix_hybrid/kernels/dsv41_fuse_hook.py` | post-import hooks, pinned sha256 anchors, env gates, K5 accounting |
| `suffix_hybrid/kernels/dsv41_bench.py` | K3-on-silicon + K4 harness (`oracle`, `bench`), CUDA-graph cold-L2 timer |
| `tests/test_dsv41_{stub,common,hc,router,swiglu,engram_fuse,fuse_hook,cli}.py` | CPU tests (stand-in + real Helion interpret mode + Triton lowering) |

Every kernel module is also a lab module (`KERNELS` in the console ABI) and has a CLI:

```
python -m suffix_hybrid.helion_kernels.dsv41_hc      oracle | bench [--autotune|--config F] | capture_replay
python -m suffix_hybrid.helion_kernels.dsv41_router  oracle | bench | capture_replay
python -m suffix_hybrid.helion_kernels.dsv41_swiglu  oracle | bench | capture_replay
python -m suffix_hybrid.helion_kernels.dsv41_engram  oracle | capture_replay
compute-sanitizer --tool memcheck python -m suffix_hybrid.helion_kernels.dsv41_hc capture_replay
```

## 1. Rules this lane follows (and what they changed)

* **No cross-CTA coordination of any kind.** My first HC/router designs used a split-K partial buffer plus a
  last-arriver atomic counter (the k2_attn merge idiom). Dropped on the coordinator's instruction (grid-level
  sync deadlocked on silicon; a barrier costs more than a launch). Every kernel here is deterministic and
  semaphore-free; the lowered Triton of all bodies is asserted to contain no `atomic`, grid barrier, spin loop,
  `gdc_*` or cluster op. (Helion itself inserts a CTA-level `tl.debug_barrier()` between a store and the re-read of
  the same buffer inside the resident kernel: `__syncthreads`, not a grid barrier.)
* **Nothing fuses across an all-reduce.** The mHC boundary starts after the sublayer's reduction; the router
  and SwiGLU sit between collectives.
* **Alternative fusion boundaries stay in the search space** (a more aggressive HC fusion and a fused all-reduce
  both lost on silicon before): HC `split2` (2 launches) vs `resident` (1 launch, negative control) vs stock;
  router unfused (Helion GEMV + stock top-k) vs `resident` (1 launch) vs stock. The K4 commands print the
  whole-boundary comparison, not just per-kernel speedups.
* **Capture safety.** Scratch comes only from torch's allocator (inside the bodies); there is no persistent
  state at all (the counters of the first design are gone); nothing syncs the host; a (kernel, bucket, flag)
  variant that has not run eagerly raises when first seen while a CUDA graph is being captured (`_check_warm`,
  vLLM's warm-up pass before each capture runs every variant first). Boot gates `<kernel>_capture_replay`
  (capture on a side stream, replay with changing inputs, bit-exact vs eager, forward and reverse order, run
  twice) exist for all 8 kernels and are exercised end to end on CPU with a stand-in graph runner.
* **Bounded search spaces** (`SEARCH_SPACE` in each module, asserted against the bound Helion `ConfigSpec`):
  K1 bt 1..16 x bh 32..256 x ns 1..16; K2 bt 1..8 x bg 1..8; resident bt 1..16 x bh 32..256 x bg 1..8; gemv
  bm 16..64 x bn 16..128 x bk 32..256; router-resident bt 16..64 x bk 32..256; swiglu bt 1..16 x bg 1..16. All
  powers of two. num_warps/num_stages/indexing stay in Helion's default space; `default_config()` is a starting
  point inside the space, not a tuned config.
* Env knob per family, default off; flag off == previous behaviour (`oxide` = stock vLLM, module never imported).

## 2. Launch census (stock vLLM 39983a5e7 on SM120, decode, TP8 + EP8, 1 token = 6 per sequence at DSpark k=5)

Read from `vllm/models/deepseek_v41/**`, `deepseek_v4/nvidia/model.py`, `model_executor/kernels/mhc/**`,
`layers/fused_moe/router/**`, `common/engram.py` (our PR-61530 overlay) and cross-checked against the lane-R
dossier (`dsv41-speed-2026-10-19.md`, ~1,750 kernels/step) and boot5.log.

**Correcting the brief's premise.** The stock V4.1 graph is already far more fused than "thousands of tiny
kernels" suggests; most of what the brief lists as separate launches is one kernel:

| brief item | what stock actually launches |
|---|---|
| q / kv RMSNorms (+ quant) | `fused_q_kv_rmsnorm_quant` (Triton) or `fused_q_kv_rmsnorm`: 1 launch for q and kv |
| rope + fp8 KV write + q pad | `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`: Q-side RoPE+pad and KV-side RoPE + UE8M0 fp8 quant + paged insert in ONE op |
| indexer q / k | `fused_indexer_q_rope_quant`, `indexer_k_norm_rope_store`: 1 launch each |
| compressor (ratio 2/1) | `fused_save_compress_norm` (state save + pool + RMSNorm) + `rope_quant_insert`: 2 launches, only on the 4 kv-source layers |
| router scoring | `dsv4_topk` Triton: sqrtsoftplus + bias + top-6 + renorm + x1.5 + padding + image bias in ONE launch (PDL), after the gate GEMM |
| mHC | `mhc_fused_post_pre_delayed_tilelang`: sinkhorn/norm are inside the epilogue; 2 launches (<= 32 tokens) or 3 |
| Engram hash | one Triton launch (`_hash_ids_kernel`) + 3 tiny mask ops |

So item (4) of the brief (rope + KV write + norm) needs no kernel: it is already fused upstream. What is left are
the launches the missing torch.compile passes would have removed (no `@support_torch_compile` on either V4.1
class, so norm+quant / act+quant never run) and the places upstream leaves two launches where one would do.

Per-step counts (decode, backbone 40 layers; "ours" = a launch this lane removes or restructures):

| group | launches per occurrence | occurrences / step | launches / step | note |
|---|---|---|---|---|
| mHC boundary | 2 (<= 32 tok) / 3 (> 32) | 77 served + 3 (layer 0 / Engram attn pre) | 154 / 231 | ours |
| attn input act quant (cute-dsl) + `fused_wqa_wkv` GEMM | 2 | 40 | 80 | quant: ours (-37) |
| `fused_q_kv_rmsnorm_quant`, `wq_b`, qnorm/rope/kv insert | 3 | 40 | 120 | already fused |
| `compute_global_topk_indices_and_lens` | 1 | 38 | 38 | dedupe candidate (below) |
| sparse MLA decode (FlashInfer) | 1-2 | 40 | 40-80 | |
| o-proj: inv-rope, `wo_a` bmm, `wo_b` quant + GEMM | 4 | 40 | 160 | |
| compressor / indexer extras | 3 / 7 | 4 / 8 | ~68 | |
| all-reduce (attn out, MoE combine) | 1 | 80 | 80 | not fused (rule) |
| router: gate GEMM + `dsv4_topk` | 2 | 40 | 80 | ours (GEMV) |
| shared expert: quant, gate_up, SwiGLU, quant, down | 5 | 40 | 200 | ours (-80) |
| routed experts (FlashInfer CUTLASS NVFP4) | ~7 | 40 | ~280 | lane B |
| Engram (2 layers) | ~12 | 2 | ~25 | P1/P2 |
| DSpark draft layers, heads, sampler, Markov loop | | | ~225 | |

Ranking of what this lane removes, by launches x the lane-R floor of 5-7 us per graphed launch (est.; the GLM
profile it comes from is 7.1 us per kernel including kernel time, so these are upper bounds on the saving):

| # | change | launches / step | est. time | share of lane-R bottom-up step (20.4 / 39.9 / 82.5 ms at c1 / c8 / c32) |
|---|---|---|---|---|
| 1 | HC epilogue emits the MXFP8 activation; attn input projection and shared-expert gate_up consume it | -37 attn, -40 ffn (if the attribute survives the MoE call chain) | 0.19-0.54 ms | 0.9-2.6% / 0.5-1.4% / 0.2-0.7% |
| 2 | HC `split2` above 32 tokens (c >= 6) | -77 (0 at c <= 5: stock already fuses post + GEMM) | 0.39-0.54 ms | 0 / 1.0-1.4% / 0.5-0.7% |
| 3 | SwiGLU + MXFP8 quant (shared experts) | -40 | 0.20-0.28 ms | 1.0-1.4% / 0.5-0.7% / 0.2-0.3% |
| 4 | router GEMV | 0 | 0-56 us (GLM: ~1.4 us/call x 40) | <= 0.3% |
| 5 | Engram P1 (packed rows through NCCL) | +2 | see 3.5 | c32 only |
| 6 | Engram P2 (replicated wkv, no output gather) | -4 NCCL + copies | see 3.5 | c32 mostly |
| - | `compute_global_topk_indices_and_lens` dedupe (not built: plugin-sized, 30 of 38 launches are repeats) | -30 | 0.15-0.21 ms | |

Upper bound of 1+2+3 together: ~194 launches (c >= 6) = 1.0-1.4 ms; at c <= 5 117 launches = 0.6-0.8 ms. Both
ignore the extra epilogue work and assume the Helion kernels match the stock kernels one for one.

## 3. Kernels

### 3.1 mHC boundary (`dsv41_hc.py`)

**K1 contract.** `x` bf16 [T,5120], `res` bf16 [T,4,5120], `post` f32 [T,4], `comb` f32 [T,4,4] ([t,i,j], the previous
boundary's output), `fn` f32 [24,4*5120] (rows 0:4 pre, 4:8 post, 8:24 comb[jp,k]), `scale` f32 [3], `base` f32 [24], `pre_in`
f32 [T,4] or None, `norm_w` bf16 [5120]. Math (identical to `mhc_post_torch` + `mhc_pre_delayed_torch` + RMSNorm with the
TileLang BF16 boundaries): `rc = bf16(sum_i comb[t,i,j] res[t,i] + post[t,j] x)`; `mixes = (rc @ fn^T) * rsqrt(sum(rc^2)/(4H) + eps)`;
`pre' = sigmoid(m s0 + b) + eps`, `post' = sigmoid(m s1 + b) * 2`, `comb' = sinkhorn(m s2 + b)` (row softmax + eps, column
normalise, 19 x (row, column), eps in every denominator); `layer_in = bf16(bf16(sum_j pre_in[j] rc[j]) * rsqrt(mean(.^2)+eps) * w)`;
`aux = bf16(mean_j rc[j])`; optionally `q, qs = MXFP8(layer_in)` (FlashInfer UE8M0, F8_128x4 swizzle, padding rows and group
columns zero).

**K2 design.** `split2`: K1 = one program per (token tile, hidden split) computes the post-mapped residual chunk (written once, BF16)
and the split-K partials of the 24 mixes and of sum(rc^2) (fp32 FMA, not TF32: the mixes feed a 20-step Sinkhorn; fn is
read once per token tile, so bt up to 16 keeps the L2 traffic at 12 x 2 MB for T = 192); K2 = one program per token tile reduces the
partials, runs the Sinkhorn, collapses the 4 streams with the carried pre-mix, RMSNorms, writes aux and (optionally) the quantised
activation. `resident`: the same boundary in one launch, one program per token tile, whole K inside.

**Launch derivation.** Stock: `mhc_fused_post_pre_split_config` fuses post + GEMM only up to 32 tokens (2 launches);
above it `post`, prenorm GEMM, epilogue (3). `split2` = 2 always: -1 launch x 77 served boundaries at c >= 6, 0 at c <= 5.
`resident` = 1 (-1 / -2). `norm_weight is None`, `pre_mix is None` and `capture_aux` are all supported.

**K3 (CPU).** Reference = float64 `reference_args`; checks in `tests/test_dsv41_hc.py`: the reference vs a copy of vLLM's own
torch formulas; K1/K2/resident bodies (ragged token tiles, splits past the end of the hidden dim, uneven hidden = 96/160, zero rows,
x300 rows, identity comb, `repeat` = 1/2/5, with and without aux/quant/pre_in); the quantised outputs bit-exact against the
independent MXFP8 transcription over dirty scratch; split2 == resident up to f32 split sums; judges catch injected errors; the
serving wrapper is a drop-in for the stock 6-tuple.

**K5.** Hook `hc` (below): logs the first call of every (path, bucket), exit summary of helion calls per bucket and stock calls per
reason (`tokens>192`, `no-config`); `helion` mode with a missing config raises at the first call; unexpected dtype/layout/hidden raises.

### 3.2 MXFP8 activation hand-over (HC epilogue -> attention input projection, shared-expert gate_up)

The stock linears (`FlashInferCutlassMxfp8LinearKernel`, SM120) quantise their bf16 input with `flashinfer.mxfp8_quantize` in a launch of
their own, but accept a `QuantizedActivation` (`as_quantized_activation`; `fused_q_kv_rmsnorm_quant` already feeds `wq_b` that way). With
`SUFFIX_DSV41_HC_QUANT=1` the epilogue writes `layer_in` (bf16, still needed by the compressor / indexer GEMMs) **and** its MXFP8
form; the hook hangs the `QuantizedActivation` on the layer-input tensor (`_suffix_mxfp8`); the patched
`DeepseekV4Attention._fused_wqa_wkv_gemm` and `DeepseekV4MLP.forward` pick it up. Fail-loud checks: the consuming linear must be a
FlashInfer MXFP8 kernel (stock `can_fuse_query_quant`), and if 80 activations were emitted without one reaching the attention
projection the run raises (the attribute got lost). The FFN side depends on the tensor object surviving the `FusedMoE` call chain;
the exit summary counts `ffn_hit` / `ffn_miss`, so a zero is visible.
**The silicon must prove flashinfer parity**: `oracle` compares our q/qs bytes with `mxfp8_e4m3_quantize` (HC: bit-exact required).

### 3.3 SwiGLU + MXFP8 quant (`dsv41_swiglu.py`)

Stock: `SiluAndMulWithClamp` (CUDA) then `down_proj`'s own quant launch; the op is transcribed from `csrc/activation_kernels.cu`
(packed path: `g' = bf16(min(g, L))`, `sil = bf16(g'/(1+exp(-g')))`, `out = bf16(sil * clamp(u, -L, L))`). Output = `down_proj`
input as `QuantizedActivation`. d = 2304/8 = 288 = 9 groups -> scales pad to 12 columns (zeroed in-kernel, last token tile also
zeroes the padding rows). Bit-exact vs the torch transcription on CPU; on GPU Helion lowers `exp` to `libdevice.exp` (the accurate expf
the CUDA op uses), so differences should be rare 1-step flips, quality-gated (K6). -1 launch x 40 layers.

### 3.4 Router (`dsv41_router.py`)

The scoring/top-6/renorm/scale kernel is already one launch (`dsv4_topk`) and already writes exactly lane B's input contract
(`topk_weights` f32 [M,6], `topk_ids` i32 [M,6] **global** ids, -1 and 0 on padding rows; `nvfp4_moe` takes `id - ep_rank*E_local`
itself and its Helion route root ranks the pairs in parallel CTAs inside the launch that quantises x), so no layout conversion and no
plan output is needed. The only unfused part is the gate GEMM. Two Helion bodies:

* `dsv41_gate_gemv`: bf16 x bf16 -> f32, full-K loop per (token tile, expert tile), no split-K and no atomics so all 8 TP ranks (which recompute the
  router on identical data and must pick identical experts) agree bit for bit. Hook `gate` replaces only the (384, 5120) `GateLinear`.
* `dsv41_gate_topk_resident`: GEMM + scoring + top-6 in one launch. A cross-CTA split-K + last-arriver version would need the banned
  semaphore; this one streams the 3.9 MB weight through one SM per token tile and is expected to lose (GLM's fused router did). It exists so K4 prints the fused
  vs unfused boundary: `python -m ...dsv41_router bench` times stock (GEMM + dsv4_topk), Helion GEMV + stock dsv4_topk, and resident.

K3 includes a literal transcription of `_dsv4_topk_kernel` (ties to the lowest expert id, image rows use `bias_vl`, NaN -> -1e30,
softplus branch above 20, padding rows) and a tie-aware judge (rows whose 6th/7th scores are within 1e-4 are not judged on membership).

### 3.5 Engram (`dsv41_engram.py`; measurement-gated, not wired)

Per layer (layers 1 and 14) at TP8/DP1 the stock path is: lookup of this rank's 3 of 24 hash heads from pinned host (side stream,
prefetched at step start; layer 1 consumes it after layer 0, so the ~2.3k small PCIe reads at c32 are hidden - est. <= 15 us, nothing to
fuse there) -> `tensor_model_parallel_all_gather(rows, dim=1)` (**bf16** [T,3,256] -> [T,24,256], 512 B per row) -> `wkv` =
`ColumnParallelLinear(24*256 -> 5*5120, gather_output=True)` -> a **second all-gather** of [T,25600] bf16 -> gate kernel.

* **P1 (built).** `gather_packed` copies the raw 128 B + 8 B record into a packed staging row, the all-gather moves uint8 (136 B per row,
  -73% bytes), `dequant_packed` decodes after the exchange, bit-identical to the PR's `PACKED_FP4` decode. +2 launches per layer for
  2.36 MB -> 0.63 MB per layer at c32 (est. saves ~0.15 ms per layer at the GLM-measured ~10 GB/s algorithm bandwidth; loses at c <= 8 where the all-gather is latency bound).
  Coalescing the data row and the scale row into one 136 B record does **not** reduce PCIe requests (a 136 B record straddles two 128 B lines;
  the current 128 B data row is exactly one line), so it is not proposed.
* **P2 (spec, one line, main session).** The bigger bytes are the wkv output gather: [T,25600] bf16 = 9.8 MB at c32. Replicate the wkv weight
  (`disable_tp=use_sequence_parallel or REPLICATE`, line 1607 of `dsv41_engram_pr61530.txt`) and every rank computes all 25600 columns itself:
  no output gather, +~140 MB (7/8 of ~162 MB, MXFP8 assumed; NVFP4 would halve it) per layer per rank, +0.17 ms compute at c32 vs est. 0.7-1.1 ms of NCCL (8.6 MB per rank at
  6-10 GB/s). It loses at small T (reading 157 MB per layer costs ~0.1 ms vs a ~50 us gather), so it is a c >= 8 decision for K6. lane R's
  0.5 ms Engram estimate at c32 omits this gather; one nsys of an Engram layer settles it.
* **Persistent prefetch design.** Nothing to add at decode: the lookup is issued at the start of `forward` on its own stream and consumed
  >= 400 us later; next-step prefetch is impossible (the next ids are not known until sampling finishes).

### 3.6 Not built, with reason

* Gate GEMM + top-k fusion across CTAs (semaphore: banned). `compute_global_topk_indices_and_lens` dedupe (a per-step memo keyed by the indexer
  generation; -30 launches, ~0.2 ms): plugin-sized, no kernel. PDL for the Helion kernels: Helion 1.5 supports `hl.pdl_wait()` /
  `hl.pdl_launch_dependents()` (emits `gdc_wait` + `launch_pdl=True`), but only as unconditional top-level statements, so a flag cannot gate
  it - it needs separate PDL body copies; stock TileLang/Triton kernels around ours use PDL, so K4 may be biased against us by ~1-2 us per kernel. Worth a follow-up
  only if K4 shows ours within a few us of stock.

## 4. Hooks (`kernels/dsv41_fuse_hook.py`, off by default)

`SUFFIX_DSV41_FUSE=1` arms them; families: `SUFFIX_HELION_DSV41_HC`, `..._ROUTER`, `..._SWIGLU` = `oxide` (default) | `auto` | `helion`,
`SUFFIX_DSV41_HC_QUANT=1`, `SUFFIX_DSV41_HC_BOUNDARY=split2|resident`. `auto` serves a bucket only where the lab shipped a config
(`configs/helion/<kernel>/<platform>.json` = lab WINS only), stock elsewhere; `helion` raises on a missing config.

| anchor (sha256 pinned, nightly 39983a5e7) | patch |
|---|---|
| `vllm.models.deepseek_v41.nvidia.ops.mhc` c8233a70... | global `mhc_fused_post_pre_delayed_tilelang` |
| `vllm.models.deepseek_v41.attention` 1b22884b... | `DeepseekV4Attention._fused_wqa_wkv_gemm` |
| `vllm.models.deepseek_v4.nvidia.model` 57028950... | `DeepseekV4MLP.forward` |
| `vllm.model_executor.layers.fused_moe.router.gate_linear` f82a00c1... | `GateLinear.forward` (384, 5120 only) |

Sha pins come from the GitHub blobs at the pinned commit; the same method reproduces the existing `dsv41_engram.py` pin (80d6f149...), but
confirm the four files against the pod's installed copies on first boot (a mismatch refuses to start, by design).
Calls above 192 tokens always go stock. Anything else off-contract raises.

Wiring for `sitecustomize.py` (main session; I may not edit it), after the block64 line:

```python
if os.environ.get("SUFFIX_DSV41_FUSE", "").strip() == "1":
    from suffix_hybrid.kernels.dsv41_fuse_hook import install_post_import_hook as _dsv41_fuse_hook
    _dsv41_fuse_hook()
```

First A/B arms once configs exist: F1 `HC=auto`; F2 + `HC_QUANT=1`, `SWIGLU=auto`; F3 + `ROUTER=auto`. With no configs shipped every arm is
identical to stock (the log says `stock for no-config`).

## 5. What silicon must verify first (in this order)

1. `oracle` for every module on one PRO 6000: ptxas/sm_120a compile of all variants, no spills (check `-Xptxas -v` via Helion's logs), shared memory <= 99 KB,
   and the K3 verdicts. Expect to prune the largest tiles (K1 bt 16 x bh 256 is likely over budget).
2. flashinfer parity of the emitted quant bytes (HC: bit-exact; SwiGLU: <= 0.5% of bytes) and that `ffn_hit` > 0 in the exit summary.
3. `capture_replay` boot gates for all kernels, then under `compute-sanitizer` (the g14 drafter capture fault is unresolved, nothing here may add to it).
4. `bench` K4 per bucket: HC `split2` and `resident` vs `mhc_fused_post_pre_delayed_tilelang`; router unfused/resident vs stock; SwiGLU vs act + quant; ship only winners.
5. Compile budget: each (kernel, bucket, flag) variant is a separate Triton compile (9 buckets x ~4 variants for HC); the unrolled Sinkhorn makes K2 the slowest. If
   boot time hurts, ship fewer buckets (`auto` runs every unshipped bucket on stock).
6. K6 (section 5 of AGENTS.md) on a dev pool; quality drift from the GPU `exp` in SwiGLU and from fp32-vs-TF32 mixes (ours is the more exact side).

## 6. Handoffs

* **main session**: `sitecustomize.py` snippet above; presets F1-F3; decide Engram P2; ship configs from the lab; add ledger rows (one per kernel, status `todo`).
* **lane B (MoE)**: nothing required; the router output contract is unchanged (global ids, -1 padding).
* **comms lane**: the Engram all-gathers (bf16 rows 12 KB/token, wkv output 51 KB/token) are the largest Engram cost at c32; include them in the collective sweep.
* **lane R**: the 0.5 ms Engram estimate at c32 omits the wkv output gather (9.8 MB).

## 7. Tests run (exact result lines)

Run from the worktree root (2026-10-19, macOS, CPU only):

* `HELION_INTERPRET=1 PYTHONPATH=$PWD:$PWD/tests /tmp/lane-c/venv/bin/python -m pytest tests/test_dsv41_*.py -q`
  (Helion 1.5.0, torch 2.14.1 CPU; includes the pre-existing `test_dsv41_engram.py`) -> `228 passed in 58.42s`
* the same with `~/Documents/vllm-suffix-hybrid/.venv/bin/python` (no Helion: stand-in only, the real-Helion oracles skip) -> `128 passed, 19 skipped in 2.10s`
* Triton-lowering sweep (real Helion `bound.to_triton_code()` over every kernel x bucket x flag, CPU, no ptxas): `lowered 243 variants; FAILURES: 0`.
* `python3` itself (system) has no pytest or torch on this Mac, so it cannot run these. The plugin venv alone silently skips the Helion oracles; only the
  lane-c venv result counts for K3.
