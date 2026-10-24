# DeepSeek-V4.1-Flash routed experts (lane B, blind program 2026-10-19)

Target: `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` on 8x RTX PRO 6000 (SM120), TP8 + expert
parallel, vLLM nightly 39983a5e7. Everything below is CPU-proven (K1-K3 + lab-ready K4) or
explicitly marked as needing silicon. No speed number here is measured.

## 1. Contract (K1)

Read from config.json `text_config` and the safetensors headers (HTTP range, no download):

| item | value |
|---|---|
| layers | 40, ALL MoE (`mlp_layer_types` null: no dense/hash layer); DSpark/MTP layers `mtp.*` separate |
| experts | 384 routed, top-6, +1 shared; moe_intermediate 2304 (full per expert under EP), hidden 5120 |
| routing | sqrtsoftplus + `e_score_correction_bias` (noaux_tc), norm_topk_prob, routed_scaling 1.5 |
| clamp | `swiglu_limit` 10.0 |
| routed w1/w3 | U8 `[2304, 2560]` (packed e2m1, low nibble first), `weight_scale` F8_E4M3 `[2304, 320]` |
| routed w2 | U8 `[5120, 1152]`, `weight_scale` F8_E4M3 `[5120, 144]` |
| per expert scalars | `weight_scale_2` F32, `input_scale` F32 (per matrix) |
| NOT NVFP4 | shared experts (FP8 e4m3 `[2304,5120]`, e8m0 `[72,160]`, in the quant `ignore` list), gate (BF16 `[384,5120]`), `mtp.*` experts (MXFP4, e8m0) |

After vLLM's modelopt processing (FLASHINFER_CUTLASS backend, confirmed live in boot5.log) the
lane reads the layout the GLM/gemma lane already uses: `w13` U8 `[E, 2I, H/2]` = `[up; gate]`,
128x4-swizzled e4m3 block-16 scales per expert, per-expert alphas g1/g2, ONE activation gscale per
GEMM (`gscale_shared` is checked at load; per-expert input scales that differ refuse the layer).

Shapes per rank (K1, from config.json + layout):

| layout | E local | I | status |
|---|---|---|---|
| EP8 (prod: TP8 + `--enable-expert-parallel`) | 48 of 384 | 2304 | tuned + served |
| TP8, no EP (kernel-only: FlashInfer CUTLASS refuses it, I/8 = 288 not a 128 multiple) | 384 | 288 | opt-in tuning, served by us only, no baseline |

**Clamp placement (verified against vLLM source at the pinned commit):** `silu_and_mul_with_clamp`
(`vllm/model_executor/layers/activation.py`, `csrc/activation_kernels.cu` HAS_CLAMP) and the
FlashInfer CUTLASS expert (`gemm1_clamp_limit` -> `swiglu_limit`): `gate = min(gate, L)` (above
only), `up = clamp(up, -L, L)`, then `silu(gate) * up`, on the fc1 output before the activation. The
NVFP4 backend selector at this commit lists FLASHINFER_CUTLASS in `NVFP4_BACKENDS_WITH_CLAMP`, so
the stock path (our baseline and the M > MAX_M path) applies the same clamp. The Helion fc1
epilogue clamps in f32 (`clamp_ref` is the f64 spec; `tests/test_nvfp4_moe.py::test_clamp_edge_table`
pins values at / above / below +-10 and the negative gate, `test_clamp_is_one_sided_on_the_gate` proves a
symmetric gate clamp does NOT match). The shared expert carries the same clamp (vLLM's
`DeepseekV4MLP(swiglu_limit=...)`), run by vLLM, not by this lane.

**Router and scaling:** `dsv4_topk` (Triton) renormalizes the top-6 sqrtsoftplus scores and
multiplies by 1.5 inside the router. The lane receives final f32 weights (sum 1.5 per token) and
applies nothing itself. Padding rows arrive as id -1 / weight 0 and are treated as off-rank (exact
zeros). Hash routing is not used by V4.1 (`num_hash_layers=0`).

## 2. What was built

| file | change |
|---|---|
| `suffix_hybrid/kernels/dsv41_moe.py` (new) | the V4.1 shape/arch gate: `refuse(info)`; supported set = hidden 5120, silu, topk 6, limit 10.0, 384 experts, layouts EP8 / TP8 |
| `suffix_hybrid/kernels/nvfp4_moe.py` | clamp in the references (`clamp_ref`, `gate_up_ref`), eligibility accepts silu+clamp, I % 16, E > 256 (`oxide_gap` -> Helion-only), startup error for any NVFP4+clamp layer the lane refuses (and for any backend but `helion`), the oxide-free route (`oxide_free_route`, `NO_WS`: arm/layer/oracle/bench import no `_native`), `rand_routing(dist=balanced/skewed/hot)`, the EP load model (`expected_load`, `load_stats`, `load_markdown`), `adversarial_blocks`, DS oracle/bench cases, FlashInfer baseline with `swiglu_limit`, cold-L2 bench under two load distributions, `winner_table`, `pick_variants` / `variant_table` |
| `suffix_hybrid/helion_kernels/nvfp4_moe.py` | clamp in the fc1 epilogue, EP (`experts_global`, `id_base`) in case keys + lab, CLAMP_BUCKETS, plan split fused3 <= 16 / front4 >= 6, bounded search spaces, FlashInfer whole-layer baseline where oxide cannot run, routing distribution by seed, the body-variant seam (`variant_for`, `variant_module`, `check_variant`: section 11) |
| `suffix_hybrid/helion_kernels/nvfp4_moe_tune.py` (new) | in-lab tuner for 30 min lab gates: detached process, records + configs, searches on balanced routing, gates and scores on balanced/skewed/hot, tunes the variants' kernels too and writes the per-bucket body rows |
| `suffix_hybrid/helion_kernels/dsv41_moe_shared.py`, `suffix_hybrid/kernels/dsv41_moe_shared.py`, `tests/test_dsv41_moe_shared.py` (new) | the shared-expert fusion (section 7) |
| tests | `tests/test_nvfp4_moe.py`, `tests/test_helion_nvfp4_moe.py`, `tests/test_dsv41_moe_shared.py`, `tests/test_dsv41_moe_variants.py` (load model + the variant seam), `tests/test_dsv41_moe_oxide_free.py` (the route under an import guard) |

### Gate behaviour (fail closed, K5)

`SUFFIX_NVFP4_MOE=1` plus a V4.1 NVFP4 layer: `eligibility()` (parallel contract, scale layout,
gscale_shared, no EPLB/all2all/shared-overlap) then `dsv41_moe.refuse()` then
`dsv41_moe.backend_refusal()`. Any failure raises
`[suffix nvfp4-moe] DeepSeek-V4 NVFP4 MoE layer <name> cannot run on the SUFFIX_NVFP4_MOE lane: <why>`
(naming the supported layouts, or, for the backend, `SUFFIX_HELION_NVFP4_MOE='oxide': the DeepSeek-V4
route is oxide-free and runs on Helion only (the oxide kernel cannot serve swiglu clamp 10.0, E=384 >
256, ...); set SUFFIX_HELION_NVFP4_MOE=helion`) instead of quietly staying on vLLM's path.
Non-NVFP4 layers (shared experts, `mtp.*` MXFP4) are not touched. V4.1 therefore REQUIRES
`SUFFIX_HELION_NVFP4_MOE=helion` (the oxide backend cannot serve it, `auto` would import the oxide
family for a fallback that does not exist): every M <= MAX_M runs on Helion and a missing config is
a load error. Section 12 is the oxide-free guarantee and its test.
The layer oracle runs on the real weights at load (ours vs FlashInfer vs f64 spec, zero contract,
clamp engaged via a x4 activation batch, skewed and hot loads) and refuses on mismatch.
`arm()` requires cc 12.x. Family 100 is never checked.

## 3. Kernel design and search spaces (K2 design, compile unproven)

Plans (all W4A4, the A operand is the exact NVFP4 dequant in bf16, f32 accumulate, as the GLM port):
`front4` = route_quant, fc1 (+clamp, act, NVFP4 requant of h), fc2, combine; `fused3` = route_quant,
fc1, token-major fc2+combine. Tune table (`DEFAULT_TUNE`): fused3 for M <= 8, front4 above;
the lab tunes each plan's own kernels one bucket past that (fused3 <= 16, front4 >= 6) so the
split is measured. Buckets (clamped layers): 1, 6, 8, 16, 32, 64, 128, 192 (6 = one DSpark verify
sequence, 192 = 32 x 6). 34 (kernel, case) tuning jobs for EP8 (68 with `NVFP4_MOE_TP8_NOEP=1`, which adds the no-EP rank).

Search spaces (literal bounds inside the bodies; `test_search_spaces_are_bounded` reads them back
from the bound ConfigSpec at the real shapes; before this the fc1/fc2 row tile could reach 8192 and
the H tile 5120, the ds_mla/fp8_moe timeout pattern in the commit log):
fc1 rows 16..64, N blocks 1..8 (x16 cols), K bkb 2..8; fc2 rows 16..64, H 16..256, bkb 2..8;
fc2_combine H 16..128, K 1..16; combine tokens 1..16 x H 16..512; route_quant 16..64 x 16..128 pair
tiles + 1..8 x 1..64 quant tiles. No grid barriers (deadlock lesson). Rows per expert segment
(re-derived, section 8): an expert gets M / 64 rows on average, ~1-3 per HIT expert and <= ~10 for
the heaviest of a batch at M = 192 on balanced loads, so a 16-row dot tile is the MMA's M floor and
mostly padding there (the layer is bandwidth-bound); 16..64 is kept because the skewed tail (~80
rows on the top expert at M = 192) and the hot case (M rows) loop over 64-row tiles.

Why `fused3` (token-major fc2 + combine, one launch less) is cheap up to M ~ 16: it re-reads an
expert's w2 once per PAIR instead of once per expert, extra bytes = (local pairs - hit experts) x
6.6 MB = +0.3 experts at M = 8 (1.8 % of the layer's bytes), +1.3 at M = 16 (4.0 %), +5 at M = 32
(8.8 %), +17.5 at M = 64 (19 %), from the load table; the tune table's split at M = 8 and the lab's
fused3 <= 16 overlap come from this, not from rows per expert.

## 4. Oracle (K3, CPU)

- f64 spec reference = vLLM's numerics with the clamp (`moe_ref`), exact zeros off-rank.
- Helion bodies run in Helion eager mode (`HELION_INTERPRET=1`) against it, both plans, EP rank
  with off-rank and all-off-rank batches, top-6, I = 288 (18 whole NVFP4 blocks, masked tails),
  I % 64 != 0.
- Load distributions: balanced, skewed (rank^-0.8 popularity, top expert ~8 %), hot (one local
  expert takes all M rows). Measured rel error vs the f64 spec 1.5e-3..1.7e-3 (REF_TOL 8e-3: GLM g13 refused a correct layer at 5e-3 on silicon with 5.39e-3, plugin branch glm-int 1536bb2).
- Adversarial scale patterns (`adversarial_blocks`, weights AND activations): 10 % zero blocks,
  5 % e4m3-subnormal scales, 3 % non-zero blocks whose scale flushes to 0, 5 % saturated
  scale blocks, 5 % x64 single-element outliers. Same bound.
- Clamp: edge table, one-sided gate, clamp engaged (> 2 % of outputs) with the unclamped spec NOT
  matched.
- Silicon-only: Triton's `.to(float8_e4m3fn)` / e2m1 rounding vs torch (the CPU run uses torch for
  both); layer oracle at load; per-stage readback is oxide-only (Helion gets bitwise determinism).

## 5. K4 on silicon: one lab run

Two entry points, same bodies:

1. **Kernel lab (one PRO 6000, DEV bundle)**: the lab runs only its gate argv
   (`python -m suffix_hybrid.kernels.nvfp4_moe bench`, 30 min cap). The DEV bundle routes that gate
   to `suffix_hybrid.helion_kernels.nvfp4_moe_tune.gate()` (handoff to main: the routing lives in
   the lab bundle, not in this repo). Call it repeatedly: the first call starts ONE detached tuner
   (quick, then full effort) over the 34 routed jobs (EP8; `NVFP4_MOE_TP8_NOEP=1` adds the other
   34) + 19 fused-shared jobs (+ 8 per kernel of a dot_scaled variant when lane I's module is in
   the build); it searches on BALANCED routing (the load is ~1-3 rows per hit expert, section 8)
   and gates / scores on balanced, skewed and hot; later calls print progress and the new
   configs (paged); when a pass is done the same call runs the cold-L2 e2e bench: FlashInfer
   CUTLASS vs the Helion plans under EVERY body variant on the tuned configs, balanced AND skewed,
   for M in 1/6/8/16/32/64/128/192, and prints per layout the **VARIANT TABLE** (per bucket every
   variant's worst-dist time, the pick, WIN/LOSS vs FlashInfer; dequant keeps ties within 1 %) and
   the **WINNER TABLE** (per M and distribution: helion/flashinfer ratio, WIN/LOSS) with the line
   `SUFFIX_NVFP4_MOE_MAX_M=<n>`: the largest M with an unbroken win from the bottom (M above it
   delegates to FlashInfer, "ships only where it wins"). The picked non-default bodies are written
   as `<ROOT>/configs/dsv41_moe_variant/<platform>.json` (section 11). Configs:
   `<ROOT>/configs/<kernel>/<platform>.json`, copy into `suffix_hybrid/configs/helion/` only for a
   winning layout.
2. **Autotune job** with the two-line shim (`from suffix_hybrid.helion_kernels.nvfp4_moe import
   KERNELS`): `cases(model)` derives the EP8 rank (and with `NVFP4_MOE_TP8_NOEP=1` the no-EP rank) from a `DeepseekV41ForCausalLM` model dict with
   `experts` 384, `topk` 6, `moe_intermediate` per rank (I x tp = global), `quant_format` nvfp4.
   Gate = reference check on seed 0 (balanced); timing rotates seeds (balanced, skewed).

Baseline = vLLM FLASHINFER_CUTLASS for the whole layer (x quant + both GEMMs + clamp + finalize,
this EP rank), because oxide cannot run V4.1. The per-stage lab rows therefore compare one stage
against the whole layer; only the e2e bench is the verdict. K4 rule: a bucket where Helion loses
keeps FlashInfer (MAX_M). The TP8-no-EP rank (I = 288) has no stock baseline at all
(`fi_supported`: FlashInfer pads to 128 rows and refuses it): it is opt-in
(`NVFP4_MOE_TP8_NOEP=1`), the oracle judges it vs the f64 spec only, the bench prints Helion
times without a ratio, and a missing baseline never drops a tuned config.

## 6. EP integration (who does what)

- Router: vLLM `dsv4_topk` -> global ids + final weights. No change.
- Dispatch: none. TP8 + EP without DP/all2all is "no_dp_ep": every rank holds all M tokens;
  FlashInfer and our lane compute only experts in `[ep_rank*48, +48)` (id_base from the linear
  expert_map, any other map is refused) and emit a partial `[M, 5120]` with exact zeros for tokens
  with no local expert.
- Combine: our kernels (fc2+combine / combine) sum the token's local top-k in fixed k order.
- Shared expert: `MoERunner` runs it BEFORE the routed call (NO_OVERLAP; the lane refuses
  MK-overlapped shared experts), TP-sharded (288 per rank, FP8), output not reduced. Then
  `result = shared_output + fused_output` and ONE all-reduce of the sum (`_maybe_reduce_final_output`,
  or the fused mHC all-reduce when the model uses it: same un-reduced-partial contract). The lane
  returns the same partial FlashInfer does, so the all-reduce count per layer stays at 1.
- M > MAX_M (prefill chunks up to 8192, mixed batches) and any M the lane does not serve call the
  parent `forward_modular` = FlashInfer CUTLASS with the clamp. Decode and the 192-token DSpark
  verify are the lane's range (`SUFFIX_NVFP4_MOE_MAX_M` allows <= 256).

## 7. Shared-expert + routed-combine fusion (built: `SUFFIX_DSV41_MOE_FUSE_SHARED`, default off)

Code: `helion_kernels/dsv41_moe_shared.py` (bodies, plan, lab ABI), `kernels/dsv41_moe_shared.py`
(knobs, eligibility, serving state, layer oracle, `moe_capture_replay`, K4 bench),
`tests/test_dsv41_moe_shared.py`; hooks in `kernels/nvfp4_moe.py` (`forward_modular` ->
`_sfx_routed`, `_sfx_prepare`, `arm`).

**What the checkpoint and vLLM give us.** The shared expert is NOT BF16 and not NVFP4: it is MXFP8
(`F8_E4M3` weights, 32x32 `F8_E8M0` blocks that vLLM `repeat_interleave`s to one byte per
(row, 32-K block)), run by `DeepseekV4MLP` through `ModelOptLinearMethod` -> FlashInfer CUTLASS
`mm_mxfp8` (boot5.log: `Using FlashInferCutlassMxfp8LinearKernel`) with dynamic MXFP8 quantization
of both activations. Per TP8 rank: gate_up `[576, 5120]` = `[gate; up]`, down `[5120, 288]`, scales
in the F8_128x4 swizzle. The fused bodies read those parameters IN PLACE (no copy, no re-quant):
the layout is pinned by a test against vLLM's `swizzle_mxfp8_scale`, and a layer whose linears run
on another kernel (Emulation / TRTLLM layouts) is refused with the kernel's name.

**Design.** Math W8A16: `gate = min(x.Wg, L)`, `up = clamp(x.Wu, -L, L)`, `h = bf16(silu(gate) *
up)`, `S = h.Wd` (f32 accumulate; x and h stay bf16, so ours is at least as accurate as stock's
W8A8). Fusion points (zero extra launches):
- `route_quant` unchanged. `fc1` -> `dsv41_moe_fc1_sh`: a SECOND ROOT LOOP in the same kernel =
  its own program-id region of the same launch (Helion emits one `@triton.jit`, `if pid_shared <
  N_shared: ... else: ...`; checked in the generated code by a test): the shared gate/up GEMM tiled
  over (M tiles of 16..64 rows) x (16-column blocks), clamp, silu*up -> `hs` bf16 `[M, I_sh]`.
  It is the first root, so it gets the lowest program ids and starts first.
- front4: `fc2` unchanged, `combine` -> `dsv41_moe_combine_sh`: the routed combine plus the
  shared down projection as an M-tiled dot; `out = bf16(sum_k y + S)`.
- fused3 (M <= 8): `fc2_combine` -> `dsv41_moe_fc2_combine_sh`: one more GEMV slot per (token, H
  tile) after the routed slots (what the old oxide slot did, but per token and per tile: parallel
  over M, not a serial walk over all M rows inside one CTA).

**Why it cannot be the long pole** (glm-int measured -5.4 ms/step at M=96 and -22 ms at M=192
because its slot walked all M rows serially): the shared work is tiled over M like routed work
(grid = ceil(M/BM) x I_sh/16/BJ in fc1, ceil(M/BT) x H/BH in the combine); a shared fc1 program
has the same K chain (160 steps of 32) as a routed fc1 program, so the region adds no latency to
the launch; per token the shared FLOPs/bytes are ~12 % of ONE routed expert's (I_sh 288 vs 2304);
the region needs no barrier and nothing waits on it inside the launch. It is still an estimate:
the K4 `bench` measures it (below), the default ceiling is 8 until it does, and a boot guard
(`long_pole_gate`, first fused layer) times fused vs routed-only + vLLM's overlapped shared MLP in
CUDA graphs with a cold L2 at M=1 and at the ceiling on the real weights: slower than
`SUFFIX_DSV41_MOE_FUSE_SHARED_POLE_TOL` (default 1.05) x the stock composition is a startup error
naming both numbers, so a wrong ceiling or a config tuned on another card cannot ship silently.

**Combine order vs vLLM (pinned commit).** `MoERunner.forward`: `result = shared_output +
fused_output` (bf16), then ONE `tensor_model_parallel_all_reduce` of the sum
(`_maybe_reduce_final_output`; FlashInfer CUTLASS `output_is_reduced()` False). Routed scaling:
`dsv4_topk` folds 1.5 into the top-k weights and `apply_routed_scale_to_output` is False (V4's
`FusedMoEFactory` call does not pass it), so the runner's own factor is 1.0 and the shared expert
is not scaled. Ours: `out = bf16(sum_k y + S)` (one rounding instead of two; test: within 4e-3 of
`bf16(shared_bf16 + routed_bf16)`). The runner is told to stop: at load, after the oracle,
`runner._shared_experts = None` and `runner._forward_entry` is re-selected (`moe_forward` instead
of `moe_forward_shared`); the lane returns `routed_partial + shared_partial` in the same
`[M, H]` bf16 partial layout, so the reduce count per layer stays 1.

**Above the fused ceiling** (`SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M`, default 8, <= 192): routed-only
(our lane up to `SUFFIX_NVFP4_MOE_MAX_M`, vLLM's FlashInfer path above) + vLLM's own shared MLP
through the SAME `SharedExperts` object (`maybe_forward_async` / `forward` / `wait` / `output`:
aux stream and events exist since the runner's `__init__`) in the runner's order, then `shared +
routed`: bit for bit what the runner did (the layer oracle asserts it).

**Capture-safety rules (g14).** The two g14 suspects were the oxide shared slot and a Triton
dense GEMM with split-K semaphores bound to streams on first use; the language is not established
as the cause, so the design removes both classes of hazard instead of trusting Helion:
1. stateless kernels: no tickets / counters / semaphores / atomics (the generated code is
   asserted free of them); every launch rewrites everything it reads later;
2. scratch only from torch's allocator inside the bodies (graph pool under capture) or load-time
   parameters read in place (stable pointers); nothing is created on first use (streams and events
   are vLLM's, made in `MoERunner.__init__`);
3. no host sync, stream/event creation or `.item()` in the serving path (AST audit test over
   `SharedState.forward/split/fused`, `FusedPlan.__call__`, `run_plan_sh`, ...);
4. every served M is run once at load (`FusedPlan.warm`, Triton specializes on sizes); an M that
   was not warmed raises INSIDE a capture instead of compiling there;
5. boot gate `moe_capture_replay` per fused layer (first layer: 5-6 M x 3 rounds; the others: M
   in {1, ceiling, ceiling+1}): capture in a `torch.cuda.graph` on a side stream (gc/empty_cache
   patched out like vLLM's CUDAGraphWrapper), replay with changing inputs (balanced / skewed /
   hot / all-off-rank routing, x redrawn), graphs interleaved in random order, output poisoned
   with NaN before each replay: bit-exact vs eager, two replays equal. CLI under
   `compute-sanitizer --tool memcheck python -m suffix_hybrid.kernels.dsv41_moe_shared
   moe_capture_replay` (`DSV41_MOE_GATE_MS=1,6,48` shrinks it). The split path is captured with
   the aux-stream fork/join inside, like the g14 context. CPU: `capture_replay(cpu_graph=True)`
   runs the same logic with a re-run-into-a-fixed-buffer stand-in; it catches non-determinism and
   carried state, not baked host values (the AST audit covers those).

**Fail closed.** `SUFFIX_DSV41_MOE_FUSE_SHARED=1` needs `SUFFIX_NVFP4_MOE=1` and
`SUFFIX_HELION_NVFP4_MOE=helion`; every V4 NVFP4 routed layer must fuse (gated / latent /
sequence-parallel / dp / reduced-output / rsf != 1 / other linear kernel / limit mismatch /
unswizzled scales are named); a missing tuned config for any served M names every (bucket, plan,
kernel); a failed layer oracle (vs the f64 spec 8e-3, shared partial alone 4e-3, vs vLLM's module
0.10, vs FlashInfer + module 0.15, split path exact) or capture gate is a startup error, and the
runner keeps its shared expert until both passed.

**K4.** `python -m suffix_hybrid.kernels.dsv41_moe_shared bench` (also run by the in-lab tuner's
gate after the routed bench): fused vs routed-only + vLLM's own MXFP8 shared MLP (CUTLASS
`mm_mxfp8`, `silu_and_mul_with_clamp`) on an aux stream inside one CUDA graph, cold L2, balanced
and skewed loads, M in 1/6/12/24/48/96/192; prints per M both shared costs (arm minus routed-only)
and `SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M=<n>` = the largest M of an unbroken fused win from M=1.
The tuner's todo list adds the 3 fused kernels (19 jobs on EP8, 53 total).

**Search spaces** (literal bounds, bound ConfigSpec read back by tests, extreme configs generate
clean Triton): fc1_sh shared rows 16..64, shared 16-col blocks 1..8, shared K blocks of 32 1..8,
then the routed fc1's (K 2..8, rows 16..64, cols 1..8); combine_sh tokens 16..64, H 16..256, K
blocks 1..8; fc2_combine_sh H 16..128, routed K 1..16, shared K 1..8. The autotuner's accuracy
check is replaced by a masked one (`autotune_check`): Helion compares candidates over the WHOLE
output, ours has unwritten rows (off-rank pairs, the trash row/slot) that differ between configs.

## 8. Expected effect (ESTIMATES, derivation shown)

**Expert load of an EP8 rank (corrected 2026-10-19: the first version of the rows column was 8x too
high, M x 6 / 48 instead of M x 6 / 384).** EP8 with allgather dispatch gives every rank all M
tokens; each token picks 6 of the 384 experts, so a given expert receives M x 6 / 384 = M / 64 rows
on average, a rank holds 48 x M / 64 = 0.75 M pairs (1/8 of the 6 M, not all of them) and hits
d(M) = 48 x (1 - (63/64)^M) distinct experts (the d column and the floors were right all along:
they never used the rows). Bytes per expert = 3 x 2304 x 5120 x (1/2 + 1/16) = 19.9 MB; floor at the
PRO 6000 DRAM peak of 1.79 TB/s. `python -m suffix_hybrid.kernels.nvfp4_moe load` prints both tables
(`nvfp4_moe.expected_load`, `load_stats`, `simulated_load`; `tests/test_dsv41_moe_variants.py`
pins the first table to the code and a Monte Carlo of the lane's router to the formula):

| M | rows/expert (all 48) | hit experts d | rows per hit expert | local pairs | MB read | floor us/layer |
|---|---|---|---|---|---|---|
| 1 | 0.016 | 0.8 | 1.00 | 0.8 | 15 | 8 |
| 6 | 0.094 | 4.3 | 1.04 | 4.5 | 86 | 48 |
| 8 | 0.125 | 5.7 | 1.06 | 6.0 | 113 | 63 |
| 16 | 0.250 | 10.7 | 1.12 | 12.0 | 213 | 119 |
| 32 | 0.500 | 19.0 | 1.26 | 24.0 | 378 | 211 |
| 64 | 1.000 | 30.5 | 1.57 | 48.0 | 607 | 339 |
| 128 | 2.000 | 41.6 | 2.31 | 96.0 | 828 | 463 |
| 192 | 3.000 | 45.7 | 3.15 | 144.0 | 909 | 508 |

| M | dist | hit experts / rank | rows per hit expert | heaviest expert of a batch (max over ranks) | heaviest ever seen |
|---|---|---|---|---|---|
| 1 | balanced | 0.8 | 1.00 | 1.0 | 1 |
| 1 | skewed | 0.8 | 1.00 | 1.0 | 1 |
| 1 | hot | 0.8 | 1.00 | 1.0 | 1 |
| 6 | balanced | 4.3 | 1.04 | 1.8 | 3 |
| 6 | skewed | 3.8 | 1.18 | 3.1 | 5 |
| 6 | hot | 3.8 | 1.20 | 6.0 | 6 |
| 8 | balanced | 5.7 | 1.05 | 2.0 | 4 |
| 8 | skewed | 5.0 | 1.20 | 3.6 | 6 |
| 8 | hot | 4.9 | 1.22 | 8.0 | 8 |
| 16 | balanced | 10.7 | 1.12 | 2.7 | 4 |
| 16 | skewed | 8.5 | 1.41 | 6.8 | 11 |
| 16 | hot | 9.3 | 1.30 | 16.0 | 16 |
| 32 | balanced | 19.1 | 1.26 | 3.6 | 5 |
| 32 | skewed | 14.5 | 1.66 | 13.2 | 20 |
| 32 | hot | 16.6 | 1.44 | 32.0 | 32 |
| 64 | balanced | 30.5 | 1.58 | 4.9 | 6 |
| 64 | skewed | 22.5 | 2.13 | 26.9 | 34 |
| 64 | hot | 27.4 | 1.75 | 64.0 | 64 |
| 128 | balanced | 41.7 | 2.30 | 6.9 | 8 |
| 128 | skewed | 32.3 | 2.98 | 53.0 | 60 |
| 128 | hot | 39.2 | 2.45 | 128.0 | 128 |
| 192 | balanced | 45.8 | 3.15 | 9.0 | 10 |
| 192 | skewed | 37.9 | 3.80 | 79.1 | 94 |
| 192 | hot | 44.2 | 3.26 | 192.0 | 192 |

(Second table: `rand_routing` over 32 batches x the 8 EP ranks, the same ids on every rank. The
heaviest expert of a batch is the max over ranks; "heaviest ever seen" is the single worst expert
in any batch. Skewed = rank^-0.8 popularity, hot = one expert takes every token.)

What follows from the corrected numbers:
- **Tiles.** Balanced decode puts 1.0-3.2 rows on a hit expert and at most ~9-10 on the heaviest
  (M = 192): every segment fits one 16-row MMA tile, 80 % or more of which is padding. The row-tile
  bound stays 16..64 (16 is the dot's M floor); only skewed (~80 rows) and hot (M rows) loop.
  Nothing in the kernels' constants depended on the wrong column (grep: no code used it); the
  lab's SEARCH distribution did rest on "skewed covers both singleton and heavy segments" and is
  now BALANCED (skewed still gates and scores), so the rare 80-row tail does not steer the row
  tile of the typical 1-3 row workload.
- **Floors.** Unchanged (they use d(M), not rows): x 40 layers, 8.4 ms/step at M = 32 and 20 ms
  at M = 192 as a hard floor for routed-expert weight streaming alone.
- **Arithmetic intensity.** At 3 rows per expert the layer does 2 x 3 x 35.4 M = 212 Mflop per
  19.9 MB = ~11 flop/B, one to two orders of magnitude below the ridge of a bf16 MMA (~280 flop/B
  at an ASSUMED ~500 TFLOP/s over 1.79 TB/s; FP4 MMA higher still): decode is bandwidth-bound at
  EVERY bucket on balanced loads. A compute-bound expert
  needs hundreds of rows: the skewed/hot tail, or prefill-sized M (>= 128 rows per expert avg at
  M ~ 8k), which this lane does not serve (M > SUFFIX_NVFP4_MOE_MAX_M, cap 256, goes to vLLM).
- **What can still make the dequant bodies lose.** ALU: dequantizing costs ~6 ops per element, 12
  per weight byte, against ~25 ops/B the SMs can issue at the DRAM peak (188 SMs x 128 lanes x
  1.9 GHz = 45.7 Tops/s over 1.79 TB/s): ~47 % of the issue budget, a co-bottleneck once enough
  experts are hit to saturate DRAM (tens of them, M >= ~32; a hypothesis, not a measurement). A hardware block-scaled MMA (`dot_scaled`, lane I) removes exactly that work,
  so it can win in the bandwidth-bound buckets too; the lab decides per bucket (section 11).

x 40 layers, as above. The nvfp4_moe module doc measured FlashInfer at about 58 % of the DRAM
peak and the oxide lane at 47 % at mid M on qwen/gemma/GLM (not V4.1); at 58 % the M = 32 layer is
about 360 us. A kernel that is 10 % faster than that saves about 0.8 ms of the 8.4 ms step. Where
the Helion lane can win is small M (fewer launches: 3-4 vs FlashInfer's chain) and a clamp
epilogue without a separate pass; where it can lose is mid/high M (ALU, above). Dequant happens once
per hit expert on balanced loads (one row tile); only skewed / hot segments re-dequantize per
row tile. The winner table decides; no gain is claimed.

**Shared-expert fusion (ESTIMATE, not measured).** Stock runs ~6 small launches per layer on the
aux stream (act quant, mm_mxfp8, clamp-silu-mul, act quant, mm_mxfp8, plus the runner's add and
the event fork/join); lane D's inventory puts the two GEMMs at about 4.3 + 3.1 us per layer
(1597 GB/s x 0.85 floors) and the glm-int fused BF16 slot saved 0.45-0.70 ms/step at M <= 8 for 4
launches + add. Scaling that by the launch count gives roughly 0.2-0.9 ms/step at M <= 8 over 40
layers (about 1-4 % of a step of ~20-25 ms), and it shrinks toward zero at large M where the
shared expert hides behind the routed experts on the aux stream anyway. K4 decides the ceiling.

## 9. Risks / what silicon must verify first

1. Helion compile of the five bodies at the EP8 shapes inside the bounded spaces (register / smem
   spills; K2 for Triton output is the lab's compile, there is no `oxide_probe` cubin step).
2. Layer oracle at load on the real V4.1 weights: `gscale_shared` (per-expert `input_scale` may differ;
   then vLLM's max-reduction decides and the lane refuses if the processed scales are not equal),
   and a1/a2 calibration range (clamp engaged fraction logged as `max_clamped`).
3. Winner table vs FlashInfer on EP8 (cold L2, balanced + skewed): sets MAX_M; may be empty.
4. CUDA-graph capture of the Helion plan at every served M (warm-up compiles all buckets first).
5. Triton e4m3/e2m1 cast rounding vs torch (CPU proof used torch for both sides).
6. DSpark/MTP experts are MXFP4 and stay on vLLM's path; hidden-state dtype must be bf16 at the call.
7. Body variants: lane I's module must honour the seam (`check_variant` refuses otherwise), its
   kernels tuned on the same cases, and a bucket's selection row needs every kernel of its plan.
8. Oxide-free: a V4.1 pod boots with `SUFFIX_HELION_NVFP4_MOE=helion`; any other value is a named
   startup error. The route imports no oxide module (tested on CPU under an import guard).
9. Fused shared expert: the layer oracle against vLLM's module (the first call of the real MXFP8
   kernels at load), the K4 ceiling (default 8 until measured), a Triton compile of the 3 fused
   bodies at the real shapes, `tensor_descriptor` indexing in a tuned config under capture, and
   lane D (`SuffixMxfp8LinearKernel` on the shared linears is accepted; its first-forward check
   must not require the shared linears to be CALLED, because fused layers do not call them).

## 10. Env knobs

| name | default | meaning |
|---|---|---|
| `SUFFIX_NVFP4_MOE` | off | the lane (required) |
| `SUFFIX_HELION_NVFP4_MOE` | oxide | V4.1 requires `helion` (oxide / auto refused: the route is oxide-free); `helion` also skips arm()'s oxide import and cubin checks |
| `SUFFIX_NVFP4_MOE_MAX_M` | 32 | set to the winner table's value (<= 192 for the verify batch, <= 256 cap) |
| `SUFFIX_NVFP4_MOE_TUNE` | table | plan per M (fused3 <= 8, front4 above); lab-only effect on Helion layers: plan choice |
| `NVFP4_MOE_TUNE_DIR`, `NVFP4_MOE_TUNE_DIST` | /tmp/nvfp4_moe_tune, balanced | in-lab tuner state dir; distribution the search runs on (gating / scoring always cover balanced, skewed, hot) |
| `NVFP4_MOE_TP8_NOEP` | off | `1` adds the no-EP TP8 rank (I 288) to the lab cases |
| `SUFFIX_DSV41_MOE_FUSE_SHARED` | off | `1` fuses the shared expert into the Helion plan (section 7); needs `SUFFIX_NVFP4_MOE=1` + a Helion backend; fail closed |
| `SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M` | 8 | largest fused M (1..192, <= the lane's MAX_M); K4 `bench` recommends it |
| `SUFFIX_DSV41_MOE_FUSE_SHARED_POLE_TOL` | 1.05 | boot long-pole guard: fused slower than this x (routed-only + vLLM shared MLP) at M=1 or the ceiling = startup error |
| `DSV41_MOE_GATE_MS` | 1,2,6,7,13,48,96,192 | M set of the `moe_capture_replay` gate (comma list) |

## 11. Body variants per M bucket: the seam to lane I's dot_scaled bodies

**Premise corrected (2026-10-19):** the brief's "Triton 3.8 has no native FP4 MMA on sm_120" is
false. Triton >= 3.6.0 lowers `tl.dot_scaled` (e2m1 x e2m1, e4m3 block-16 scales) to `mma.sync ...
kind::mxf4nvf4.block_scale` on SM120 and Helion's `hl.dot_scaled` emits it (Triton PR #8494,
MMAv2.cpp at v3.6.0 / 3.7.1 / 3.8.0). Lane I (worktree `vllm-suffix-hybrid-dsv41-fp4`, new files only:
`helion_kernels/dsv41_moe_fp4mma.py`) builds those bodies. The dequant-to-bf16 bodies of this lane
stay (they are the bandwidth-bound choice and the baseline); which body serves a bucket is DATA.

**Selection = a row in the config table.** The pseudo-kernel `dsv41_moe_variant` (constant
`nvfp4_moe.VARIANT_KERNEL`) has rows keyed exactly like every other kernel of the lane (the
`lane_case` dict, `numtokens` = the bucket, `limit`, `experts_global`, ...):

```json
configs/helion/dsv41_moe_variant/<platform>.json
[{"key": {"act": "silu", "experts": 48, "experts_global": 384, "hidden": 5120, "limit": 10.0,
          "moe_intermediate": 2304, "numtokens": 64, "topk": 6},
  "config": {"body": "dot_scaled"}}]
```

No row = `dequant`. `body` is `dequant` or `dot_scaled`; anything else, or a row without `body`, is a
startup error. Serving only reads the table: `served_ms` resolves the variant of every M <=
`SUFFIX_NVFP4_MOE_MAX_M`, then demands a shipped config for EVERY kernel of that variant's plan at
that bucket (a gap is a startup error naming `M<=64 front4/dot_scaled: <kernels>`), and
`serving_op` runs `variant_module(body).run_plan` for the call's bucket. A variant is selected per
BUCKET and as a WHOLE PLAN, never per stage: a dot_scaled plan hands packed e2m1 + e4m3 block scales
between stages, this lane's plans hand bf16-dequantized NVFP4, so the stages are not interchangeable.
The fused shared expert (section 7) lives in the dequant bodies: a bucket <= its ceiling that selects
another body is a startup error naming both knobs.

**What the lab does.** `nvfp4_moe_tune.todo()` tunes the dot_scaled kernels on the same buckets
(`variant_kernels()`); the gate's e2e bench times FlashInfer and EVERY variant at the table plan,
balanced and skewed, cold L2 (`_bench_helion`), picks per bucket the variant with the lowest
worst-dist time (dequant keeps ties within 1 %: `pick_variants`), prints the VARIANT TABLE and the
WINNER TABLE (vs FlashInfer) over the picked bodies, and writes the non-default picks as the rows
above (`write_variant_rows`: a bucket where dequant wins again loses its row). Shipping = copy the
variant kernels' configs and this file into `suffix_hybrid/configs/helion/`.

**What lane I must export** (`suffix_hybrid/helion_kernels/dsv41_moe_fp4mma.py`; `check_variant`
refuses the module at the first use with the exact violation):

| name | contract |
|---|---|
| `VARIANT = "dot_scaled"` | the key of `nvfp4_moe.VARIANT_MODULES` |
| `PLANS` | `{"front4": (names...), "fused3": (names...)}` (a subset is fine, but the tune table assigns the plan by M: fused3 for M <= 8, front4 above, so a bucket whose plan the variant lacks fails closed); kernel names are the config-table / lab names, prefix `dsv41_moe_`; a name that is also a dequant body (e.g. `nvfp4_moe_combine`, whose interface is unchanged: y f32 + ids -> out) is served by that body and ITS tuned config |
| `BODIES` | `{name: undecorated module-level Helion body}` for every new name |
| `run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out, a1_gscale, a2_gscale, act, id_base, limit=None) -> out` | same arguments as `nvfp4_moe.run_plan`; `kerns` are the compiled kernels of `PLANS[plan]` in order; `limit` is the swiglu clamp as a float (None = no clamp); `act` is 0 (silu) |
| `KERNELS` | lab ABI list (`api/autotune_kernels.py`): `name body cases make reference check baseline`; `cases(model)` must return the SAME case dicts as `nvfp4_moe.cases(model, "nvfp4_moe_fc1")` for the buckets the kernel serves (so rows and selection share keys); names must not collide with a dequant kernel; baselines may be the dequant stage or FlashInfer |

*Layout assumptions (identical to the dequant bodies; vLLM FlashInfer CUTLASS after weight
processing):* `w13` uint8 `[E_local, 2I, H/2]` rows `[up; gate]`, low nibble = even element;
`w13_sf` float8_e4m3fn, block 16 along K, 128x4-swizzled PER EXPERT, flat index
`e * estride + (r // 128) * (cp // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + (kb // 4) * 512
+ kb % 4` with `cp = round_up(K/16, 4)`, `estride = round_up(rows, 128) * cp` (scale of row `r`,
K block `kb`); `w2` uint8 `[E_local, H, I/2]` and `w2_sf` the same way; `g1`, `g2` float32
`[E_local]` per-expert alphas (they fold 1/(g_w * g_a)); `a1_gscale`, `a2_gscale` scalar activation
global scales (x and h are requantized with them: `sf = e4m3(amax16 * g / 6)`, `q = e2m1_rne(x * g /
sf)`, f64-spec in `nvfp4_moe.quant_codes`); `topk_ids` GLOBAL expert ids int32 (-1 = padding), local
experts are `[id_base, id_base + E_local)` and everything else is off-rank (exact zeros);
`topk_w` float32 with the 1.5 routed scaling already folded in; `out` bf16 `[M, H]`.

*Clamp:* after the per-expert alpha, `gate = min(alpha * acc_gate, L)`, `up = clamp(alpha * acc_up,
-L, L)`, `h = silu(gate) * up`, then the a2 requantization, exactly `nvfp4_moe.clamp_ref` /
`fc1_ref` (gate clamped from above only; tests pin the edge table).

*Output contract:* bf16 `[M, H]` = sum over the token's local experts of `tw * E_k(x)`, k ascending,
exact zeros for a token with no local expert; deterministic; STATELESS (no tickets, atomics, grid
barriers; scratch only from torch's allocator inside the bodies) so it is capture-safe like the dequant
plan (section 7 rules). Numerics may differ from the dequant plan in accumulation order, not in
quantization: the bound is the lane's (`REF_TOL` 8e-3 end to end vs the f64 spec, stage checks as
for the dequant bodies), not bitwise equality with it.

*Search spaces:* literal `register_block_size(min, max)` bounds (powers of two), tested like the
dequant bodies (`BOUNDS`); the autotune accuracy check must compare only the rows a consumer reads
(`nvfp4_moe.autotune_check` is the model: unwritten rows of off-rank pairs differ between configs).

*Where it can win, from section 8:* not at "compute-bound decode" (decode is ~1-3 rows per hit
expert) but where the dequant ALU work binds (many hit experts, M >= ~32) and on the heavy tail;
the per-bucket selection exists because that is a measurement, not a rule. M above
`SUFFIX_NVFP4_MOE_MAX_M` (cap 256) never reaches either body (vLLM's FlashInfer path); a prefill-sized
compute-bound regime would need, beyond a body: a routing kernel that is not O(P^2), a larger cap,
and warming by Triton specialization class instead of every M. That is a decision for main, not
something the selection table can express.

## 12. Oxide-free (owner direction): nothing on the V4.1 route imports `_native` or oxide

`SUFFIX_HELION_NVFP4_MOE=helion` (without `SUFFIX_FP8_MOE`) is the oxide-free route
(`nvfp4_moe.oxide_free_route()`): `arm()` imports neither `suffix_hybrid.oxide_kernels` nor
`suffix_hybrid._native`, checks no manifest or cubin, `_make_layer_cls(None)` closes over no native
module, the layer prepare loads no cubin and allocates no oxide workspace (`NO_WS`), the Helion op is
the only op (`run_ours(None, ...)` raises a named error if a layer was never prepared), the
standalone oracle/bench take `native` lazily and never for the Helion-only cases (the CLI modes
`python -m suffix_hybrid.kernels.nvfp4_moe oracle_v41|bench_v41` run only the V4.1 cases; plain
`oracle|bench` go on to the gemma/GLM/qwen cases, which do need the oxide family; the lab gate
names `nvfp4_moe_oracle|bench` in sitecustomize.py map to the plain modes), and the K4 bench of the
fused shared expert times the routed-only arm on the same Helion op. `oxide` and `auto` are
refused for V4 geometry at the first layer with the message of section 2.

Tested on CPU (`tests/test_dsv41_moe_oxide_free.py`): an import guard that raises on
`suffix_hybrid.oxide_kernels` and `suffix_hybrid._native` (and fails the test if a swallowed
`ImportError` hides an attempt) wraps the WHOLE route end to end: `arm()`, the layer class, the
weight-processing hook (eligibility, V4 gate, served-M table, the layer oracle and its Helion
warm-up), the fused shared expert (`prepare_layer`: layer oracle, capture-replay, long-pole gate,
runner detach) and `forward_modular` at fused / split / vLLM-path M; vLLM is faked, the kernels are
the real Helion bodies. A source-level test pins the only functions that may import the oxide
family (`arm` behind `not oxide_free`, `_sfx_prepare` behind `hb != "helion" and not gap`,
`_native_ready`, and the two oxide lab baselines): a new import elsewhere fails it.

Outside this lane's files (handoffs): a V4.1 preset must set `SUFFIX_NVFP4_MOE=1` and
`SUFFIX_HELION_NVFP4_MOE=helion` (today's `dsv41-sm120` preset sets neither) and must leave
`SUFFIX_OXIDE_PROBE` unset (sitecustomize.py runs `python -m suffix_hybrid.oxide_kernels` for it);
the plugin core's suffix-decoding proposer (`wrap.py`, `wrap_v2.py`, `hybrid_proposer.py`) imports
`suffix_hybrid._native` when ITS knobs are on, which is not this route's business but is where a pod
that must not load `_native` at all needs a check.
