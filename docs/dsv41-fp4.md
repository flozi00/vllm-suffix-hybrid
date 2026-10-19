# DeepSeek-V4.1 lane I: native FP4 MMA (Helion `hl.dot_scaled`) for the NVFP4 GEMM and the routed experts

Blind lane (no GPU, no cluster, no commits), branch `dsv41-fp4` on `dsv41-speed`. Everything below was
written and proven on a Mac. Nothing that needs Triton ran at all (Triton has no macOS wheel), and nothing
ran on a GPU. Every claim is labelled PROVEN (CPU test named), DESIGN (written, not run) or ESTIMATE
(derivation shown). Section 12 lists every claim that could not be verified.

## 0. Status

| kernel | K1 contract | K2 compile | K3 oracle | K4 bench | K5 startup assert |
|---|---|---|---|---|---|
| dense `nvfp4_gemm_q_fp4mma` (words) | written, section 3 | DESIGN: P0 probe + Linux CI test, unrun | CPU PASS (eager vs exact f64) | harness ready, unrun | code + plumbing CPU-proven; never saw a real PTX |
| dense `..._fp4mma_b` (bytes), `..._fp4mma_lin` (linear scales) | same | same | CPU PASS | same | same |
| dense `nvfp4_gemm_q_fp4cvt` (hardware-cvt control arm) | same | DESIGN (asm assembles for sm_120a?) | CPU PASS | same | asserts `cvt.rn.f16x2.e2m1x2` instead |
| MoE `dsv41_fp4mma_fc1` / `_fc2` (scale read `words` and `bytes`) | section 3.5 | DESIGN | CPU PASS in both modes (layer vs f64 spec, stages vs lane B refs) | harness ready, unrun | as the dense dot bodies |
| MoE `route_quant`, `quant`, `combine`, `route_torch` | section 3.5 | no MMA, nothing to assert | CPU PASS | part of the whole-layer bench | n/a |

Gate K6 (serving A/B) is not claimed. K2, K4 and the silicon half of K3/K5 need a GPU.

PROVEN on CPU (257 tests, section 13): index maths of the scale layouts byte for byte; every body against an
independent exact f64 reference in Helion's eager mode incl. zero blocks, subnormal e4m3, saturated 448,
single-element outliers, K/N tails, M in {1,2,15,16,17,64} and above one scale atom (M>128), low nibble first,
forced small tiles, split-K 1/2/4+, bit-exact run to run; Helion's real front end lowers every body (at the
configs the tests use) to Triton source with one launch, no atomic/semaphore/barrier, `tl.dot_scaled(.. 'e2m1' ..
'e2m1' ..)` and the host reduce after the launch; the K5 plumbing under a stub Triton; the knobs, plans,
twins and fail-closed branches; the capture/replay gate machinery on an eager runtime.

NOT proven: that Triton keeps the native lowering for our scale-access patterns, speed, registers, spills,
shared memory, accumulation precision of the block-scaled MMA, CUDA-graph behaviour of the Triton launches.

## 1. The premise was wrong; what Triton does

The brief said "Triton 3.8 has no native FP4 MMA on sm_120". False: Triton PR #8494 (in v3.6.0) lowers
`tl.dot_scaled` with e2m1 x e2m1 operands and e4m3 block-16 scales to
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
(`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`, table `mmaInstrPtxScaled`, v3.8.0; the
e8m0 variant is `scale_vec::2X ... ue8m0`). Helion's `hl.dot_scaled` emits that call on the default Triton
backend (not the cute backend, which is SM100-only for block scaling). vLLM#38338 measured it on an RTX 5090
(Triton 3.7.1, hot L2): faster than CUTLASS up to M=64 (gate_up), 128 (QKV), 512 (O, down) and 0.68-0.91 of
CUTLASS above. Our Helion NVFP4 ports (`helion_kernels/nvfp4_gemm.py`) decode in ALU and never used it.

Triton can silently fall back: the SM120 pattern `ScaledBlockedToMMA` (v3.8.0 `AccelerateMatmul.cpp:710-760`)
requires cc 12.x, one CTA, both scales present with the same element type, FP4 x FP4 or FP8 x FP8, and scale
tensors that do NOT already carry a linear encoding; otherwise the op becomes a bf16 upcast + regular MMA. The
result is correct and several times slower, so K5 (section 5) asserts the PTX instead of trusting the lowering.
(K < 64 per tile is a separate, known crash.)

Statements this invalidates (not my files, listed for the integrator): `AGENTS.md` section 3 ("cuda-oxide SASS
stays wherever Helion loses (today: NVFP4 GEMM, because Triton 3.8 has no native FP4 MMA on sm_120)");
`docs/dsv41-dense.md:119` ("NVFP4 GEMM: stays oxide (Triton 3.8 has no FP4 MMA on sm_120)");
`helion_kernels/nvfp4_gemm.py:33` ("tl.dot_scaled only takes e8m0 block-32 (MX) scales": true for FP8 data,
false for e2m1 x e2m1); memory note `helion-port-of-oxide-kernels.md`; `oxide-retirement-2026-10-19.md:179`
quotes the AGENTS.md sentence.

## 2. Files (all new; nothing existing was edited)

| path | lines | role |
|---|---|---|
| `suffix_hybrid/helion_kernels/nvfp4_gemm_fp4mma.py` | 956 | dense family: bodies, scale-layout maths, plans, oxide-signature twins, lab ABI, probes |
| `suffix_hybrid/helion_kernels/dsv41_moe_fp4mma.py` | 1012 | V4.1 routed experts: route/quant/fc1/fc2/combine bodies, plans, the seam wrapper, references, lab ABI |
| `suffix_hybrid/kernels/fp4mma_probe.py` | 659 | stack gate, PTX analysis, K5 recorder/assertions, capture/replay boot gate |
| `suffix_hybrid/kernels/fp4mma_bench.py` | 724 | bake-off harness (dense arms, whole-layer MoE, winner table, config shipping) |
| `scripts/fp4mma_p0_probe.py` | 602 | P0: raw-Triton mirror kernels compiled for `GPUTarget("cuda", 120, 32)`, no GPU needed |
| `tests/test_fp4mma_{ref,layout,gemm,moe,probe,ptx}.py` | 178 / 187 / 606 / 550 / 411 / 87 | CPU tests; `_ptx` is the Linux-only P0 as a test |
| `docs/dsv41-fp4.md` | this file | |

## 3. K1: contracts

### 3.1 Dense math and operands

`y[M, N] = bf16(alpha * sum_k deq(a[m, k]) * deq(w[n, k]))`, W4A4, f32 accumulate, vLLM layout:
`xq`, `w` uint8 `[rows, K/2]` (two e2m1, LOW nibble = even k); scales e4m3, one per 16 k, 128x4-swizzled
(byte of (row, kb) = `(row/128)*cp*128 + (kb/4)*512 + (row%32)*16 + ((row/32)%4)*4 + kb%4`, cp = round4(K/16)).
K % 64 == 0 (every lane already needs it), N % 16 == 0. Shapes come from the checkpoint config.json per TP
rank: V4.1 has NO NVFP4 dense layer except the lm_head (attention and shared experts are MXFP8,
`docs/dsv41-dense.md:45`): `16160 x 5120` (vocab 129280 / TP8). The other shapes are the plugin's existing
`ALL_SHAPES` (fable, gemma, qwen-flash). Scale bytes >= 0x7F (sign bit / e4m3 NaN) are refused at load
(`validate_scales`): ue4m3 has no sign and no NaN.

### 3.2 Scale layout: which, and why (the part Triton cares about)

`tl.dot_scaled` takes the scales as a `[rows, K/16]` e4m3 tile. Three ways to feed it from the buffers we have:

| option | extra HBM | how it is read | risk |
|---|---|---|---|
| `words` (primary, `nvfp4_gemm_q_fp4mma`) | 0 | swizzled buffer as int32 words: one aligned 4-byte gather per (row, k64 step) holds the 4 scales of that step; unpack by shift/mask + `hl.join` + bitcast to e4m3 | Triton must keep the native lowering through join/reshape/bitcast (P0) |
| `bytes` (fallback, `..._b`) | 0 | e4m3 byte gather `sf[offset(row, kb)]` | loads are 1-byte unless AxisInfo proves 4-byte runs (Helion's floor-div/mask helpers hide them): pipelining may suffer |
| `lin` (`..._lin`) | +1/9 of the weight bytes (a linearized copy at load); free for activations we quantize ourselves | linear `[rows, K/16]` as words, row stride K/64 words | needs the lane to hold a second scale copy |

Why `words` is primary: the swizzle puts `kb % 4` innermost, so the four scales of a k64 step are one aligned
little-endian 32-bit word at `word(row, q) = (row//128)*(quads*128) + q*128 + (row%32)*4 + (row//32)%4`
(quads = round4(K/16)/4; q = k64 step). Every scale load is then a 4-byte load, which is the minimum width
for Triton's pipeliner to turn it into a cp.async whatever AxisInfo can prove about our index arithmetic
(Triton 3.8.0 `pipeliner/LowerLoops.cpp:453-471` and `AssignLatencies.cpp:140`: an async copy needs a >= 32-bit
element and >= 4 contiguous bytes per thread, which an int32 element always has). It costs no bytes and no launch. The weights'
scales come swizzled from the checkpoint loader and vLLM's `scaled_fp4_quant(is_sf_swizzled_layout=True)`
gives the activation scales swizzled too, so reading both in place is the cheapest correct option. `lin` is
used where WE write the scales (the MoE activations x and h: our own writers emit linear scales) and as the
fallback if P0 shows the word gather cannot stay native. Cheapest correct choice if `words` loses native: ship
`bytes` (no code change: only the configs that are shipped decide, `PREFER = (words, bytes, cvt)` per bucket).

The MoE bodies (fc1, fc2) have the same two ways, as a process-wide switch instead of separate bodies:
`SUFFIX_FP4MMA_MOE_SCALE_READ=words` (default) or `bytes` (module constant `SCALE_READ`, read by Helion at
trace time; `bytes` loads the e4m3 scale tile directly: activation scales `xs` / `hs` are linear (our writers),
weight scales are the per-expert swizzled bytes, so no join / reshape / bitcast is involved). The numbers are
identical in both modes (the MoE tests run every layer case in both). Configs are tuned per mode, so ship the
ones tuned in the mode the pool runs.

Pinned by `tests/test_fp4mma_layout.py` (25 tests): swizzle vs the spec builder and byte index, word index
for the four scales, V4.1 shapes, per-expert word stride, linear word index, little-endian assert, `as_words`
(same bytes, rejects non-contiguous / wrong dtype / numel % 4), `hl.join` unpack order == `words_to_bytes`,
linearize roundtrip, `validate_scales`, and a real Helion gather with the bodies' own index expressions.

### 3.3 Fused or separate activation quant (byte / launch math)

The bf16-activation route is two launches: the existing `nvfp4_quant_act` Helion kernel (bit-exact vs the spec
quant, config table shared) + the prequant GEMM here. Fusing the quant into the GEMM prologue costs
R = ceil(N/BN) redundant quantizations of the `[M, K]` tile. At N = K = 5120, BN = 64 (R = 80), ~25 ALU ops and
2 B read per element: M = 1 -> 0.82 MB / 10 Mop (0.23 us at 45 Top/s); M = 64 -> 52 MB / 0.66 Gop (14.6 us,
more than the 10.3 us weight-stream floor of that GEMM: 14.75 MB at 1.437 TB/s). It saves ONE launch;
measured for the oxide single-launch arm: -2 us/linear at M = 1 and a loss from M = 5
(MINED-LESSONS.md:713). At best ~2 us of a ~13 us call at M <= ~4, and the MMA operand would be a computed
tensor instead of a load (one more way to leave the native lowering). Decision: separate. Not built. The
V4.1 lm_head does not even have this question: it uses vLLM's `scaled_fp4_quant` (prequant route).
The old quant body clamps the row of its swizzled scale write to one 128-row block (written for M <= 64):
`prepare()` refuses the bf16 route above `QUANT_MAX_M = 128` (fail closed; test
`test_bf16_route_refuses_m_above_one_scale_atom`). The prequant route has no such cap.

### 3.4 Tiles, split-K

Tile bounds are module constants (eager mode returns the MAX, tests force small tiles): BM 16..128 (m16 is
the instruction's M; BLOCK_M >= 16 compiles on SM120, vLLM#38338), BN 32..256, BK 32..256 packed BYTES (one
k64 step = 32 bytes, the instruction's K and Triton's minimum: K < 64 per tile is a known crash, open PR
triton#11776 per the oxide-retirement dossier, 3.2). BN 32 is four warps side by side at 8 columns each (a
guess from `createMMAEncodingForDot`, not tested; 8 warps would need BN >= 64, and an invalid combination is a
config the Helion autotuner skips). The body bound BK_MAX = 256 bytes (512 elements) is above the P0 sweep
(64..256 elements, the dossier's range), so the autotuner may try an unprobed size; invalid ones are skipped.
The P0 table prints the legal (BM, BN, BK, warps, stages) set; tighten the constants from it (`summarize()`
prints the suggested bounds).

Split-K is two-stage only, lane D's capture contract: split programs store alpha-scaled f32 partials
`partial[split, M, N]` (a `torch.empty` per call = the graph pool under capture; a `[1,1,1]` stub when not
splitting), then `out.copy_(partial[:n_split].sum(0))` on the same stream. No atomics, no semaphores, no
counters, no module scratch, fixed summation order (bit-exact run to run). The old port's last-arriver ticket
fixup is deliberately not used. `split_k` is a power-of-two tunable capped by `partial.size(0)`.

### 3.5 V4.1 routed experts

Same layer contract as lane B (vLLM FLASHINFER_CUTLASS after weight processing): `w13` uint8
`[E, 2I, H/2]` = [up; gate], `w2` `[E, H, I/2]`, low nibble first, e4m3 block-16 scales 128x4-swizzled per
expert (`[E, round128(rows), cols]`), per-expert `g1` / `g2`, one activation gscale per GEMM (`a1g` for x,
`a2g` for h). Clamp (vLLM `silu_and_mul_with_clamp`, applied to the fc1 output BEFORE the activation):
`gate = min(gate, L)`, `up = clamp(up, -L, L)`, `h = silu(gate) * up`, L = 10. W4A4: x and h are quantized to
NVFP4 with the spec rounding of `kernels/nvfp4_moe.quant_codes` (sf = e4m3(amax16 * g/6), q = e2m1_rne(x*g/sf),
g/sf as an exact f64 divide), then the dots run on the FP4 MMA with f32 accumulate. EP8 rank: 48 of 384
experts, H 5120, I 2304 (eligible: H % 64 == 0 and I % 64 == 0; lane B needs only I % 16). The kernel-only
TP8-no-EP rank (I = 288) is NOT eligible and stays on lane B.

Plans (`PLANS`): `fp4mma4` = route_quant (one launch: lane B's counting sort + per-expert counts/starts + x quant),
fc1, fc2, combine for M <= `ROUTE_KERNEL_MAX_M` = 512; `fp4mma4t` = torch route (argsort of key*pairs+pair,
scatter_add of the counts, cumsum; sync-free, capturable; lane B's O(P^2) pair loop is a non-starter at
P = 49152) + quant, fc1, fc2, combine for the prefill range. Buckets
(1, 6, 8, 16, 32, 64, 128, 192, 256, 512, 1024, 2048, 4096). A plan serves a bucket only when EVERY kernel of it
has a shipped config for that bucket; any other M calls the wrapped op (lane B's Helion plan / oxide /
FlashInfer) unchanged. Differences from lane B that decide speed (none measured): the MMA is the block-scaled
FP4 instruction (no ALU decode of weights, no re-decode per row tile); x and h travel packed (u8 + e4m3
scales) instead of bf16 dequantized rows; the tile map is expert-major (program count ceil(E*ncols/tile)
instead of one program per sorted pair: at M = 192 roughly 20k programs of which ~1k work in lane B, ~0.9k
here; ESTIMATE from lane B's grid).

fc1 programs are (expert, 16-column blocks of I): up and gate dots share the activation tile, the epilogue clamps,
activates, requantizes h (a2 gscale) and writes `hq` (packed) + `hs` (linear scales); fc2 programs are (expert,
H tile): `y[pair] = g2[e] * topk_w[pair] * (h . w2[e])` in f32 into `y[P+1, H]` (row P = trash); combine sums
the top-6 rows per token in a fixed order into bf16 and masks off-rank ids (a select, so uninitialised y rows
cannot leak NaN). Off-rank tokens give exact zeros. `y` is `[P+1, H]` f32 per call (1 GB at M = 8192): calls
whose y would exceed `SUFFIX_FP4MMA_MOE_Y_MB` (default 512 MB) run in power-of-two token chunks (>= 256; 4096 at
V4.1 topk 6 / H 5120 = 2 passes at 8192, each pass re-reads the touched experts, ~955 MB per layer once every
expert has rows, a ~0.66 ms floor per pass).

## 4. Kernel list (lab names = `configs/helion/<name>/<platform>.json`)

Dense (`KERNELS` order = lab budget order): `nvfp4_gemm_q_fp4mma`, `nvfp4_gemm_q_fp4mma_b`,
`nvfp4_gemm_q_fp4cvt`, `nvfp4_gemm_q_fp4mma_lin`. MoE: `dsv41_fp4mma_route_quant`, `dsv41_fp4mma_quant`,
`dsv41_fp4mma_fc1`, `dsv41_fp4mma_fc2`, `dsv41_fp4mma_combine`. Dense buckets (1, 8, 16, 32, 64, 128, 192); lab
baseline = vLLM FlashInfer CUTLASS FP4 GEMM on the same prequantized operands (V4.1 has no oxide to compare to;
the oxide arm is the bench's job). The console auto-discovers every `suffix_hybrid/helion_kernels/<family>.py`
as a port family (`api/autotune.py` `port_families`, name <= 33 chars): once this branch is merged, `tune_model`
will queue both families on its own.

The hardware-cvt arm (`..._fp4cvt`) is the "ALU vs MMA" control: `cvt.rn.f16x2.e2m1x2` (PTX ISA 8.6+, sm_100a /
sm_120a) via Helion's `hl.load_float4_e2m1fn_x16_to_float16`, both operands decoded to f16, scale applied in
f16 (exact: e2m1 x e4m3 has <= 6 significant bits), f16 `hl.dot`, f32 accumulate.

## 5. K5 and the P0 probe

K5 (`fp4mma_probe.py`): at lane warm-up every kernel of the family must show, in its compiled PTX,
`.target sm_120a` and `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
(a bf16 MMA in its place, or the 2X ue8m0 form, is named in the error); spills (`.local`) are a warning in
serving and a failure in the lab gate. Failure raises `Fp4mmaError` (startup error, no fallback).
`require_native_stack()` names every reason the arm cannot run: no CUDA device, GPU not cc 12.x ("family 120;
family 100 uses tcgen05"), Triton < 3.6.0 (triton#8494), Helion missing.

Plumbing (read from Helion 1.5.x and Triton 3.7.1 / 3.8.0 sources, then run under a stub Triton in
`test_generated_module_binds_the_launcher_at_import_and_the_recorder_sees_the_launch`): Helion's generated host
code calls `_launcher(triton_kernel, grid, *args, num_warps=, num_stages=)` and drops the return value;
`helion.runtime.default_launcher` (bound into every generated module at its first import) returns
`triton_kernel.run(...)` = the `CompiledKernel` (`.asm["ptx"]`, `.n_regs`, `.n_spills`, `.metadata.shared`).
There is no official hook, so `install_recorder()` swaps `default_launcher` for a recording wrapper BEFORE
the family's first kernel compiles (`prepare` / `prepare_layer` do it). A Helion without that attribute makes
the lane refuse to start; the documented fallback for such a build is a fresh `TRITON_CACHE_DIR` +
`ptx_from_cache`. Kernels compiled earlier by other lanes keep the original launcher and are never recorded.
Dense lanes call `verify_served(plans)` after their warm-up; the MoE lane calls `verify_served(served)`.

P0 (`scripts/fp4mma_p0_probe.py`, Linux, no GPU): `triton.compile(ASTSource(...), target=GPUTarget("cuda", 120,
32))` needs no driver (the wheel's bundled `ptxas-blackwell` assembles; `CompiledKernel.__init__` is lazy in both
3.7.1 and 3.8.0). It compiles raw-Triton mirrors of the exact `tl.dot_scaled` call plus the load/unpack
patterns of our bodies (`pr` = vLLM#38338's kernel, `pr_perm`, `bytes`, `words`, `linwords`, and `cvt` with
`--cvt`) over the dossier's tile sweep (`--quick` = 12 configs), asserts native instruction / `.target sm_120a` /
no `.local` / smem <= 101376 B, prints regs, spills and smem from `ptxas -v`, and summarizes the legal tiles
with suggested register_block_size bounds. `--helion` pushes the REAL generated bodies (the 4 dense GEMM bodies
and the 5 MoE bodies at the V4.1 H 5120 / I 2304 with 4 experts: E only changes the grid; `--helion-families`,
`--moe-e/-h/-i` adjust it) through `JITFunction.run` with a duck-typed driver and `knobs.runtime.jit_cache_hook`
(the 2026-10-11 fake-driver technique), so fc1/fc2 (two accumulators, data-dependent row loop) are probed too, in BOTH MoE scale-read
modes (`moe.dsv41_fp4mma_fc1` / `_fc2` = words, `...[bytes]`). Their Helion front-end half (all eleven rows lower to
parseable Triton with the expected `tl.dot_scaled` count, `tl.join` only in the words programs) is a CPU test, and
`tests/test_fp4mma_ptx.py::test_real_helion_bodies_compile_for_sm120a_with_the_native_mma` runs the compile half on
the Linux CI (no GPU needed). `--compare a.json b.json` diffs a 3.7.1 run against a 3.8.0 run. `tests/test_fp4mma_ptx.py` is the
same probe as a pytest for the Linux CI (native.yml installs Triton 3.8.0): a red cell there is the finding
"Triton decomposed this pattern". Exit 0 = `pr` and `words` native on every config.

Decision table after P0:

| P0 result | action |
|---|---|
| `pr` and `words` native | ship `words` configs (primary) |
| `words` decomposed, `bytes` native | ship only `_b` configs (PREFER falls through); file the Helion/Triton finding |
| neither, `pr` native | the problem is the Helion-generated pattern, not Triton: the family stays off until a body loads the scale tile the way `pr` does |
| `pr` not native | wheel/ptxas problem: family stays off, nothing else breaks |
| MoE `fc1`/`fc2` words decomposed, `[bytes]` native | pool env `SUFFIX_FP4MMA_MOE_SCALE_READ=bytes`; tune and ship the MoE configs in that mode |
| 3.7.1 native, 3.8.0 not (or the reverse) | the image decides; the startup check fails closed in the bad image |

Lab (3.7.1) vs serving V4.1 image (3.8.0): `add_prefetch` runs on every capability in 3.7.1
(`third_party/nvidia/backend/compiler.py:303`) but only on cc 8.x in 3.8.0 (`:312`), so SM120 kernels are
scheduled differently. Helion configs are only valid for the Triton they were tuned on: run `fp4mma_bench`
inside the V4.1 nightly image for the numbers that ship; the lab's 3.7.1 numbers are a screen.

## 6. Capture safety and the boot gate

Contract (tests `test_served_paths_have_no_host_sync_no_streams_no_atomics`, `test_module_holds_no_device_scratch`,
`test_bodies_write_no_state_other_than_out_and_partial`, `test_bodies_hold_no_state_across_launches`, the AST check
`test_bodies_are_module_level_literal_only_and_never_import_the_oxide_stack`): no atomics, barriers, semaphores or
counters; every scratch is a `torch.empty` inside the call (graph pool under capture); the route's
counts/starts are rewritten completely by every launch; nothing is created on first use except the Triton
compile, which the lane's warm-up does before capture (`warm_ms`: M == 1, and per bucket the top rows and the one
below, because Triton specializes int arguments on == 1 and on divisibility by 16; the MoE variant covers
M % 16 and M % 8 since P = 6M).

Boot gate `fp4mma_capture_replay` (`python -m suffix_hybrid.kernels.fp4mma_probe capture_replay [--full]`;
exit 0 PASS / 1 FAIL / 2 NOT RUN; output line `FP4MMA-CAPTURE-REPLAY <verdict>`): per probe (body x real shape x M x
explicit config, plus every SHIPPED config of the platform): eager on three inputs with the output and every
partial NaN-poisoned first (K3 vs the exact f64 product; K5 PTX on the first compile), capture the same call on a
side stream, replay with NEW data bit-exact against eager, a guarded output (4096 sentinel elements each side),
and two graphs replayed concurrently on two streams. A poisoned partial makes a split a kernel forgot to write
fail the exact check (`test_poisoned_partial_exposes_an_unwritten_split`, and the gate fails a deliberately
broken kernel: `test_capture_replay_gate_fails_a_kernel_with_an_unwritten_split`). `compute-sanitizer` works (the
gate times and tunes nothing). Quick tier uses the V4.1 lm_head (16160x5120, M 1/16/64/192, bytes/lin/cvt
bodies) and the qwen/gemma shapes with split 1/2/4.

## 7. Bake-off (`fp4mma_bench.py`, K4)

One command per question, same script on the RTX PRO 6000 and the RTX 5090 (platform name in every record):

```
python -m suffix_hybrid.kernels.fp4mma_bench dense --screen 21 --tune quick --out /tmp/fp4     # 7 shapes x M{1,16,64}
python -m suffix_hybrid.kernels.fp4mma_bench dense --screen dsv41 --tune full --out /tmp/fp4   # V4.1 lm_head, M 1..192
python -m suffix_hybrid.kernels.fp4mma_bench moe --m 1,6,16,48,192 --tune quick --out /tmp/fp4 # EP8 layer
python -m suffix_hybrid.kernels.fp4mma_bench report /tmp/fp4 ; ... selftest                     # selftest runs on CPU
```

Dense arms (same operands, same exact f64 reference, a wrong arm is reported WRONG and never wins):
`fp4mma_words/_bytes/_lin`, `fp4cvt`, `alu_decode` (the current Helion body), `oxide` (production launch where
the bundle has the cubin), `flashinfer` (vLLM's CUTLASS FP4 GEMM, the stock kernel), `triton_pr38338` (the
raw-Triton kernel of the PR with its 9 tile configs: the control that separates "Triton dot_scaled" from "what
Helion generates"), `scaled_mm` (`F.scaled_mm` NVFP4 two-level recipe, else `torch._scaled_mm`; unverified on
SM120, the record holds "runs" or the exception). Timing is the lane's autoroute method
(`kernels/nvfp4_gemm._graph_us`): one CUDA graph per arm, the L2 flushed by a 2x-L2 scratch read before EVERY
replay, the median of `--iters`; the Helion arms include their stage-2 reduce and per-call allocations (what
serves). Every record carries the toolchain that compiled it (torch, Triton, the ptxas Triton uses and its
path, Helion): the lab-vs-serving skew decides whether a timing transfers (dossier 3.7, T3). MoE: whole-layer cold-L2 time per M and per routing distribution (balanced, skewed) for FlashInfer
CUTLASS, lane B's plan and this family's plan; a bucket ships only if the family wins on EVERY distribution
(lane B's rule); the bench prints the largest M of an unbroken win from the bottom as
`SUFFIX_NVFP4_MOE_FP4MMA_MAX_M=<n>` (a recommendation printed for the person copying configs, not a knob the code reads).

Shipping convention (docs/helion-kernels.md): a case where the best fp4mma arm beats the stock arms
(flashinfer, oxide) by >= `--margin` (3 %, the autoroute's) is a shipping row in
`<out>/configs/<kernel>/<platform>.json = [{"key": case, "config": {...}}]`; copy those files into
`suffix_hybrid/configs/helion/` for the winners only. Several M of one bucket must all win for the bucket to ship
(a bucket is served for every M of it; the config shipped is the one tuned at the bucket's top M). CPU-tested:
case lists, winner logic, bucket rule, PR config pruning, max-M line (`selftest`). The GPU-only paths
(`run_moe`, `_moe_kernels`, `_triton_pr_runner`, `_scaled_mm_runner`) were statically reviewed (every module
attribute and call signature resolved against the repo, the PR kernel's strides/masks against the PR text,
the `F.scaled_mm` call against torch 2.14's signature) and never executed.

## 8. Env knobs

| knob | default | meaning |
|---|---|---|
| `SUFFIX_HELION_NVFP4_GEMM_FP4MMA` | `0` | `1`: dense GEMM native-FP4-MMA arm. `0`: no lane imports the module. Not 0/1: RuntimeError. With 1, `prepare()` raises naming the reason on non-SM12x, Triton < 3.6.0, Helion missing |
| `SUFFIX_HELION_NVFP4_GEMM_FP4MMA_STRICT` | `0` | `1`: a bucket <= max_m with no shipped config is a startup error instead of the incumbent serving it (mode `helion` of the lane implies it) |
| `SUFFIX_HELION_NVFP4_MOE_FP4MMA` | `0` | `1`: routed-expert arm (`prepare_layer`); fails closed on stack, ineligible shape (H or I % 64) |
| `SUFFIX_HELION_NVFP4_MOE_FP4MMA_STRICT` | `0` | `1`: a plan gap is an error naming each (bucket, plan, kernel); default: the gap stays on lane B's op |
| `SUFFIX_FP4MMA_MOE_SCALE_READ` | `words` | `words` or `bytes`: how fc1 / fc2 read scales (section 3.2); the fallback if P0 shows Triton decomposing the words unpack. Anything else: RuntimeError at import. Configs are tuned per mode |
| `SUFFIX_FP4MMA_MOE_Y_MB` | `512` | y-buffer budget per call -> prefill chunk size (power of two tokens, >= 256) |
| `SUFFIX_BOOT_GATES=fp4mma_capture_replay` (or `..._bytes` when the pool runs the bytes mode) | unset | after the sitecustomize entries of section 9.4 exist: run the gate in a child before vLLM sizes memory |
| `FP4MMA_PREFILL_CASES=1` | unset | lab only: append the prefill buckets (512, 2048, 8192) to the dense lab cases |

With every flag off the behaviour equals the previous plugin (nothing imports these modules; checked by the
knob tests). The existing family knobs (`SUFFIX_HELION_<FAMILY>=oxide|helion|auto`) pick a backend inside a
lane; this one is an additional arm, so it is a plain on/off. The lane owner may fold it into the tri-state.

## 9. Handoffs

### 9.1 Lane B / MoE lane (`kernels/nvfp4_moe.py`, `_sfx_prepare`, right after `cfg["helion_ms"] = hmoe.served_ms(...)`)

The seam is `serving_op(base_op, ...)`: it has the oxide host op's 24-argument signature, so `run_ours`
(`op = p.get("op") or native.nvfp4_moe_cuda`) already routes through it. The lane's `_sfx_oracle` consumes
`cfg["helion_ms"]` = `{M: (bucket, plan)}` (the same schema `fe.served_ms` returns): it runs EVERY served M once
before capture ("no compile can happen later"), judges one M per (bucket, plan) against FlashInfer and the f64
reference, runs the imbalanced-load cases and checks determinism. Merging the family's table into it gives the
family all of that with no new warm-up code.

```python
if os.environ.get("SUFFIX_HELION_NVFP4_MOE_FP4MMA", "0") == "1":      # off: byte-identical to lane B
    from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
    cfg["op"], fp4_ms = fe.prepare_layer(
        cfg.get("op") or native.nvfp4_moe_cuda, info["H"], info["I"], info["E"], cfg["topk"], cfg["act"],
        cfg["max_m"], limit=cfg["limit"], e_global=cfg["E_global"])
    cfg["helion_ms"] = {**(cfg.get("helion_ms") or {}), **fp4_ms}     # the family's Ms override lane B's
    cfg["fp4mma_ms"] = fp4_ms
# after `self._sfx_oracle(cfg, info)` in the same function (the oracle's warm-up has compiled everything):
if cfg.get("fp4mma_ms"):
    fe.verify_served(cfg["fp4mma_ms"])        # K5: Fp4mmaError = fail closed, names the kernel and the PTX problem
```

`prepare_layer` returns `(base_op, {})` (lane B exactly) when no config is shipped. Calls above the chunk size
run chunked inside the wrapper. M's not in the table call `base_op`. On V4.1 the oxide cannot serve the clamp
(`oxide_gap`), so `base_op` is lane B's Helion op.

### 9.2 Dense / lm_head lanes (`kernels/nvfp4_lm_head.py` l.268-300 and `select`, `kernels/nvfp4_linear.py` same pattern)

`nvfp4_gemm_fp4mma` has the SAME call surface as `helion_kernels/nvfp4_gemm`: `prepare(n, k, prequant, max_m,
mode)`, `describe`, `warm_ms(max_m)`, `partial_elems`, and the `nvfp4_gemm_q_cuda` / `nvfp4_gemm_cuda` twins (the
lane's `partial`, `counters`, `aq`, `asf`, `stream_ptr`, `splits`, `fused` are oxide launch state and ignored).
Edits for the lane owner:

1. The `hx` binding (`if hb != "oxide": from ...helion_kernels import nvfp4_gemm as hx`): with
   `SUFFIX_HELION_NVFP4_GEMM_FP4MMA=1` bind `nvfp4_gemm_fp4mma` instead. `SUFFIX_HELION_NVFP4_GEMM=auto|helion`
   stays the backend tri-state (`prepare` takes that mode), the new knob picks WHICH Helion module.
2. In `select()`, after the `for m in hx.warm_ms(max_m): run(...)` loop: `hx.verify_served([st["hx"]])` when `hx`
   has it (K5; Fp4mmaError = fail closed).
3. M cap: `MAX_M = 64` and the `plan(n, k, m, prequant=True)` launch table for `m in 1..max_m` are oxide-kernel
   limits (4 m16 tiles, `docs/dsv41-dense.md` lm_head handoff). For this arm raise the cap to
   `nvfp4_gemm_fp4mma.MAX_M` (192 = V4.1 verify, 32 seqs x 6) and skip the oxide table (the twins never read it).
4. `ops = hx if st["hx"][M] else native`: for an M without a plan the lane must take its FlashInfer arm, not
   `native`, once the oxide is retired (the autoroute assumes "ours" exists for every M).
5. Add `(16160, 5120)` to the lm_head `SHAPES` (already requested in `docs/dsv41-dense.md`).

Unchanged and now judging the new arm for free: the cold-L2 autoroute ("ours" vs FlashInfer per M), the LOAD
ORACLE against the exact quantized reference, the graph-vs-eager bit-identity check, the `HELION n x k` load line
(`describe`: `M1:nvfp4_gemm_q_fp4mma M2-8:incumbent ...`).

### 9.3 `docs/helion-kernels.md` family table (add)

```
| nvfp4_gemm_fp4mma | none (arm of nvfp4_gemm; lab baseline FlashInfer CUTLASS) | kernels/nvfp4_linear.py, kernels/nvfp4_lm_head.py via `hx` | `SUFFIX_HELION_NVFP4_GEMM_FP4MMA` (+ `_STRICT`) |
| dsv41_moe_fp4mma  | none (alternative to nvfp4_moe's dequant bodies)           | kernels/nvfp4_moe.py via `serving_op`                      | `SUFFIX_HELION_NVFP4_MOE_FP4MMA` (+ `_STRICT`) |
```

plus a sentence that the autotune console auto-queues both families once merged, and the corrected
"FP4 MMA" statements listed in section 1.

### 9.4 `sitecustomize.py` `_BOOT_GATES` (add; identical to `fp4mma_probe.BOOT_GATES`)

```python
    # native-FP4-MMA family (lane I): K3 exact + K5 PTX + capture/replay + two concurrent graphs
    "fp4mma_capture_replay": (["-m", "suffix_hybrid.kernels.fp4mma_probe", "capture_replay"],
                              {"SUFFIX_HELION_NVFP4_GEMM_FP4MMA": "1",
                               "SUFFIX_HELION_NVFP4_MOE_FP4MMA": "1"}),
    # the same with the MoE scale-read fallback (a gate child gets the pod env minus its SUFFIX_* keys, so the
    # mode the pool runs must be a gate of its own)
    "fp4mma_capture_replay_bytes": (["-m", "suffix_hybrid.kernels.fp4mma_probe", "capture_replay"],
                                    {"SUFFIX_HELION_NVFP4_GEMM_FP4MMA": "1",
                                     "SUFFIX_HELION_NVFP4_MOE_FP4MMA": "1",
                                     "SUFFIX_FP4MMA_MOE_SCALE_READ": "bytes"}),
```

### 9.5 CI / other

`native.yml` already installs Triton 3.8.0 on Linux for the test step, so `tests/test_fp4mma_ptx.py` runs there
and un-skips; it needs no GPU. Run the CPU suites with `/tmp/lane-c/venv/bin/python` (Helion present) and
`PYTHONPATH=<worktree>`; the plugin `.venv` has no Helion, those tests skip.

## 10. Expected effect (ESTIMATES; nothing measured; no V4.1 profile exists)

Constants (dossier `oxide-retirement-2026-10-19.md` 3.5): weight bytes 0.5625 B/element, usable HBM
1.437 TB/s (90 % of 1.597), FP4 dense 2015.2 TFLOP/s at eta 0.6, 16-bit decode path 503.8 TFLOP/s at eta 0.5.
Tensor time / memory time = 2*M*W/(eta*P) / (W*0.5625/BW):

| M | 48 | 96 | 192 |
|---|---|---|---|
| 16-bit register-decode path | 0.97 | 1.95 | 3.89 |
| native FP4 | 0.20 | 0.41 | 0.81 |

1. Dense / lm_head (V4.1: 16160 x 5120, NVFP4 46.5 MB vs bf16 165.5 MB per call, 2 calls per step). At M <= 16
   every FP4 kernel streams the same bytes: floor 32.4 us/call (bf16 115 us). Native MMA cannot beat a
   bandwidth-optimal oxide/CUTLASS kernel there (expect parity +- a few %); against our current ALU-decode
   Helion port it removes the decode (~0.4 x the memory time by the dossier's 7-op count): up to ~1.4x
   from that term alone, if pipelining holds. The measured gap those ports left is far larger (0.17-0.62x of
   the oxide kernel on every shape, run d614 gemma-31b TP2 0.26-0.62x: oxide-retirement dossier, section 2), so
   whether the native arm recovers it is exactly what the bake-off decides. The structural gain is M in 48..192, where the 16-bit paths turn tensor-bound (ratios
   above) and the FP4 path stays at the floor, and where the oxide stops (M <= 64). Lane D's accounting for
   the lm_head (`docs/dsv41-dense.md:93`, same constants): 331 -> 93 MB/step = ~175 us/step at M <= 48,
   0 at M = 192; with the cap lifted the M = 192 row would be 318 us (bf16, tensor-bound) - 69 us = ~250 us/step,
   i.e. about +1 % of an unmeasured 17-20 ms step, only on steps whose verify batch exceeds 64 rows.
2. Routed experts (EP8 rank, 48 experts, 19.9 MB each, 955 MB per layer if all are touched). With uniform
   routing (no skew; real routing is skewed) the expected touched experts are 48*(1-(1-1/384)^(6M)):

   | M | 1 | 6 | 16 | 48 | 96 | 192 |
   |---|---|---|---|---|---|---|
   | touched experts | 0.7 | 4.3 | 10.6 | 25.3 | 37.3 | 45.6 |
   | floor per layer (us) | 10 | 60 | 147 | 351 | 517 | 632 |
   | floor, 40 layers (ms/step) | 0.4 | 2.4 | 5.9 | 14.1 | 20.7 | 25.3 |

   Rows per expert are ~3 at M = 192, so the per-expert tensor time with a 16-row tile is 0.94 us against a 13.9 us
   stream (ratio 0.07): the experts are bandwidth/latency/occupancy-bound, not tensor-bound, and the ceiling of
   any arm is the floor above. The FP4 MMA arm removes lane B's ALU decode and its wasted programs; the realistic
   claim is "closer to the floor than lane B's dequant bodies", and parity-to-modest against FlashInfer's
   native-FP4 CUTLASS MoE (which already uses the same instruction). Sensitivity, not a prediction: each 1 % of
   expert time at M = 192 is 0.25 ms/step (40 x 632 us x 1 %). The routed experts dominate the step at c >= 8
   (14-25 ms of floor) and are small at c = 1 (2.4 ms at M = 6), so this arm matters for throughput, not c1.
3. No end-to-end tok/s is derivable without a V4.1 decode profile. The only A/B that counts is section 5 of
   AGENTS.md on a dev pool (K6).

## 11. Risks and what silicon must verify first (in order)

1. P0 on Linux (no GPU): native instruction for `pr` and `words` (and `bytes`, `linwords`) on both wheels; the legal
   tile table, regs, spills, smem; `cvt` assembles for sm_120a; `--helion` for the real generated bodies. If `words`
   is not native the family ships `bytes` or stays off (section 5). This decides everything else.
2. `fp4mma_probe capture_replay --quick` on one GPU: K3 vs the exact f64 product ON SILICON (the CPU oracle
   computes exact products with f32/f64 accumulation; the tensor core's internal accumulation of the
   block-scaled MMA is not modelled), K5 on real PTX, capture/replay bit-exactness, guards, concurrent graphs.
3. `fp4mma_bench dense --screen dsv41 --tune quick` (lm_head, M 1..192) with all arms, then `--screen 21`; then
   `moe --m 1,6,16,48,192`. Run in the V4.1 serving image (Triton 3.8.0), not only the lab (3.7.1).
4. Ship configs for winners only; dev-pool A/B per AGENTS.md section 5 (K6) before any prod change.

Risks: (a) `words` pattern (join/reshape/bitcast of a gathered tile) may give Triton a linear-encoded scale
operand -> silent decomposition (K5 catches it, speed lost; dense falls back to the `_b` body, the MoE to
`SUFFIX_FP4MMA_MOE_SCALE_READ=bytes`; reading Triton 3.8.0's layout inference I could neither confirm nor rule it
out: join of a blocked tensor stays blocked, `inferReshapeOpEncoding` tries a blocked result first and falls back
to a linear layout, and `remove-layout-conversions` runs before `accelerate-matmul`); (b) scale-load pipelining (cp.async needs >= 32-bit
contiguity; the words are 4-byte loads by construction, but unmeasured); (c) 3.7.1 vs 3.8.0 scheduling
difference (`add_prefetch`); (d) register pressure of fc1 (two accumulators `[BROW, 16*BCOL]` + requant epilogue)
may spill at the large tile bounds: P0 `--helion` prints it; (e) per-call `torch.empty` for the split-K partial
and y inside graphs relies on the graph pool (lane D precedent), gate unrun; (f) the MoE prefill route
(`route_torch`: argsort over P = 6*8192) and the y chunking are estimates; (g) tile bounds BM 16, BN 32 are
untested minimums; (h) `hl.dot_scaled` and `hl.join` behave identically in Helion 1.5.0 (the bundle pin) and
1.5.1 on CPU (both suites pass); the Triton side is untested; (i) boot compile time: every served
(bucket, kernel, specialization class) is a Triton compile at warm-up and the Triton cache is not persisted
across pods. `warm_ms` for the full V4.1 decode table (M 1..192) is 30 Ms x 4 kernels = at most 120 launches
(Triton dedups identical specializations, so ~60-120 compiles; 42 Ms with the prefill buckets), the lm_head
adds 13 Ms x 1 kernel. At a few seconds each [ESTIMATE] that is minutes per pod. Mitigation: ship only the
winning buckets (only shipped buckets are served and warmed), seed `TRITON_CACHE_DIR` from CI.

## 12. Claims I could NOT verify

1. That any scale-access pattern (`words`, `bytes`, `linwords`) compiles to `kind::mxf4nvf4` in Triton 3.7.1 or
   3.8.0 for `sm_120a` (P0 not run: Linux only). Source reading says the lowering exists; whether it survives our
   address arithmetic is open.
2. That ptxas accepts every generated PTX (including the `cvt.rn.f16x2.e2m1x2` inline asm, which is copied from
   Helion's primitive) and the register / spill / shared-memory numbers of any config.
3. Any speed or bandwidth number, including all of section 10 (derived from dossier constants and routing maths).
4. The accuracy of the block-scaled MMA's internal accumulation versus the exact f64 oracle (CPU emulates exact).
5. That the launcher swap works on the real Helion runtime of the lab image (verified by executing a generated
   module under a stub Triton and by reading Helion 1.5.0 / 1.5.1 sources), and that `CompiledKernel.n_regs` is
   populated after the first launch.
6. CUDA-graph capture and replay of these kernels (the gate machinery is tested on an eager runtime only), including
   the graph-private allocation of the per-call split-K partial and the MoE `y`.
7. Triton 3.7.1 vs 3.8.0 behaviour differences beyond the `add_prefetch` line I read.
8. That `torch._scaled_mm` / `F.scaled_mm` NVFP4 runs on SM120 (the bench records the outcome).
9. vLLM#38338's numbers are hot-L2 on an RTX 5090 (clock ~2.7 GHz); the cold-L2 PRO 6000 behaviour is unknown.
10. The GPU-only code of `fp4mma_bench.py` and the Triton side of `fp4mma_p0_probe.py` (statically reviewed, never run).
11. The 20k-vs-0.9k program-count comparison with lane B (read from lane B's grid, not executed).
12. That the lab (Helion 1.5.0, Triton 3.7.1) autotunes these bodies without invalid-config storms; the Helion
    autotuner skips configs that fail to compile, so a bad bound shows as a slow search, not a wrong result.

## 13. Tests run (macOS, CPU, Helion eager; exact result lines of the final run)

```
$ PYTHONPATH=<worktree> /tmp/lane-c/venv/bin/python -m pytest tests/test_fp4mma_ref.py tests/test_fp4mma_layout.py \
    tests/test_fp4mma_gemm.py tests/test_fp4mma_moe.py tests/test_fp4mma_probe.py tests/test_fp4mma_ptx.py -q
257 passed, 1 skipped in 31.65s                 (Helion 1.5.1; the skip is test_fp4mma_ptx.py: Triton is Linux-only)
$ PYTHONPATH=/tmp/lane-i/helion150:<worktree> ... (the five CPU files)
257 passed in 37.79s                            (Helion 1.5.0, the plugin bundle's pin)
per file: ref 9, layout 25, gemm 131, moe 71, probe 21
$ neighbours + fp4mma in ONE process (helion_runtime, helion_nvfp4_gemm, helion_nvfp4_moe, helion_mxfp8_gemm,
  dsv41_dense_capture_replay, dsv41_moe_shared + the six fp4mma files)
523 passed, 1 skipped in 82.60s
$ python scripts/fp4mma_p0_probe.py --selftest          -> fp4mma_p0_probe selftest OK
$ python -m suffix_hybrid.kernels.fp4mma_bench selftest -> fp4mma_bench selftest OK
$ python -m compileall -q <every new file>              -> compileall-ok;  pyflakes: no undefined names
```

`git status`: twelve new files, no tracked file changed. Not run, by design: `tests/test_fp4mma_ptx.py` (Linux +
Triton: the mirror kernels and the real generated bodies), `scripts/fp4mma_p0_probe.py` without `--selftest`,
`fp4mma_bench dense|moe|report`, `fp4mma_probe capture_replay`.
