# nvfp4_ds_mla — status ledger

| Step | State |
|---|---|
| Crate compiles to PTX (.target sm_120a, ISA 8.7, no .local) | done locally (macOS, cargo-oxide) |
| ptxas 13.0 -> sm_120a cubin, cuobjdump SASS check, manifest | CI (`scripts/oxide_build.py`; no ptxas on macOS) |
| Host ops + plan build (`--features oxide-kernels`), Rust tests | done (cargo test, all CI feature sets) |
| CPU proofs: fragment algebra (S 576-dim + PV), ABI, patch replay/drift, patched impl on stub vLLM, oracle harness dry run | done (`tests/`, `sm120/tests/`) |
| Silicon oracle `python -m nvfp4_ds_mla_patch.oracle` | ready, NOT RUN (needs one RTX PRO 6000) |
| Microbench `--bench` vs stock fp8_ds_mla | ready, NOT RUN |
| GLM clone pool with `--kv-cache-dtype nvfp4_ds_mla` + quality eval | not started (prod `glm` untouched) |

Fixed vs the blind draft: host ops did not compile; cp.async stage parity
was always 0; NS = ceil(C/64) made prefill o_part ~2 GiB/layer at T=8192;
no block-stride support (HiSparse hot views); PTX target lacked `a` for
e2m1 cvt; patch rewrote the shared base class, never applied its writer /
backend / HiSparse edits, no hooks, no fixtures.

Hardened 2026-09-26 (branch dsmla-harden):
- q staging: per-(token, head) power-of-two prescale, amax -> [2^14, 2^15),
  inverse folded into the logit scale. bf16 (8 sig. bits) -> f16 (11) is
  exact for every value >= amax * 2^-28; the f32 S accumulator is exactly
  2^k x the unscaled one, so logits are bit-identical to the old path
  wherever it was finite, and finite for |q| up to ~2e33 (was 65504).
  bf16 mma (`m16n8k16.f32.bf16.bf16.f32`, sm_80+, legal on sm_120a) was the
  alternative; rejected: it rewrites every B-fragment dequant (f16 LUT,
  e4m3 shift, mul.f16x2) for no accuracy gain over the exact prescale.
- V/PV/lse headroom at the format ceiling (SF 448 x e2m1 6): in-kernel f16
  K/V <= 10.5, P in [0,1], all accumulators f32 — no overflow (CPU-proved).
- launch guard `check_launch` (plan + host op): grid.x, grid.y and every
  u32 element index; T=8192 prefill (ns 1) uses <= 2^25 of 2^32; o_part =
  T*HQ*1 KiB (64 MiB at HQ 8), same bytes as the output.
- boot self-test at impl init (once per device, fail closed): |q| ~ 1e5 +
  max-SF rows vs f64 reference — catches a stale pre-prescale cubin.
- oracle: `adversarial_*` gates (q 1e5, single-dim 1e5 latent/RoPE, 1e-6,
  1e30 rows, max-SF rows, all -1 tokens, T 1/6/32/8192, HQ 64) and a
  T=8192 bench row. CPU: the old f16 staging fails them, the fix passes.

Silicon 2026-09-25 (bundle dc3c688c, RTX PRO 6000): T=1 PASS, every T>1
reader gate FAIL (row rel-L2 1.1-1.8). Fixed:
- S mask tested each lane's B-fragment row (nt*8 + gq) instead of its C
  columns (nt*8 + 2t4 + e; k2 does it right). Exact only without -1 slots,
  and the oracle's T=1 token was the one full top-k row. CPU repro:
  tests/test_nvfp4_ds_mla_fragments.py (index-level partial + merge with
  the host op's launch args; the old mask fails every token with -1s).
  Oracle T=1 now uses a partial token; T {1,6,32,256} x HQ {8,64}.
- plan: ns = ceil(SMs / rows) overshot one wave (T=6: 192 CTAs on 188
  SMs, the 0.79x bench row; T=64: 3 splits x 64 = 2 waves, 0.68x). Now
  floor(SMs / rows): T=6 -> ns 16 (96 CTAs), T=32 -> 5, T=64 -> 2.
- writer_vs_vllm_triton (2.54 % bytes): not a convention fork. Triton
  quantizes its f32 normed/roped values, the gate fed our writer the bf16
  copies; that rounding alone gives 2.46 % / 3.70e-2 on CPU. The gate now
  takes f32 kv_c_out / k_pe_out and requires Triton == the row reference
  byte for byte (same SF/e2m1/e4m3/perm convention).

Residual (needs silicon): oracle + bench NOT RUN; the PTX is built locally
(sm_120a, ISA 8.7, no .local), ptxas/SASS is CI-only. lse = m + lg2(l) in
f32 loses lg2(l) resolution when |m| is huge (merge weight error <=
ln2 * ulp(m): 0.5 % at |m| ~ 1e5), only matters when two splits' maxima
are that close. Single 1e5 dims against e2m1 x SF ties are ill-conditioned
for any f32 kernel (the oracle exempts rows an f32 reference cannot
resolve). NVFP4 vs fp8_ds_mla attention output differs ~15-18 % rel-L2 on
synthetic data (format noise) — model-level eval needed.
