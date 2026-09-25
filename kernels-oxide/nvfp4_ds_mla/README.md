<!-- SPDX-License-Identifier: Apache-2.0 -->
# nvfp4_ds_mla — SM120 NVFP4 sparse-MLA KV cache kernels (cuda-oxide)

Writer + split-capacity decode for vLLM's `nvfp4_ds_mla` cache (GLM 5.3 /
DeepSeek-V3.2 sparse MLA) on SM120. Contract dossier:
`inference-console/plugin-harness/dossiers/dsmla-kernel-contracts-glm53.md`.

Build: cargo-oxide -> PTX `.target sm_120a` (ISA 8.7) -> ptxas 13.0 ->
sm_120a cubin (`oxide-variants.json`; the writer's
`cvt.rn.satfinite.e2m1x2.f32` is arch-specific). Products are f16
`mma.sync.m16n8k16` with in-register e2m1/e4m3 dequant (k2 recipe).

## Row layout — 352 B/token

| bytes | content |
|---|---|
| `[0,256)` | 512 latent dims, e2m1 pairs, **low nibble = even dim** |
| `[256,320)` | 64 RoPE dims, raw **unscaled** e4m3 |
| `[320,352)` | 32 e4m3 SFs (one per 16 dims) at byte `8*(s&3) + (s>>2)` |

`sf = e4m3(max(amax16 * f32(1/6), 2^-9))`, `x = e2m1 * sf` — bit-identical to
vLLM's fused Triton writer (`fused_norm_rope`, non-HiSparse path). Cache
views are `[blocks, block_size, 352]`; slot `s` lives at
`(s / block_size) * stride(0) + (s % block_size) * 352` (padded block
strides, e.g. HiSparse hot views, are fine).

## Kernels (ABI checked by tests/test_oxide_abi.py)

1. `nvfp4_ds_mla_quant_store` — grid `(T)`, block 64 (warp 0: 32 latent SF
   blocks, warp 1 lanes 0..3: RoPE). Negative slots skipped.
2. `nvfp4_ds_mla_attn_partial` — grid `(T * ceil(HQ/8), NS)`, 256 threads,
   69,312 B smem (1 CTA/SM). One CTA = (token, 8-head tile, capacity split of
   `c_per_split` rows, 64-row cp.async double-buffered tiles). Q is staged
   through the k-permutation that cancels the B-fragment dim order
   (tests/test_nvfp4_ds_mla_fragments.py proves S and PV on CPU). `-1` rows
   are zero-staged and masked; empty splits write lse = -inf.
3. `nvfp4_ds_mla_attn_merge` — grid `(T * HQ)`, exp2-domain LSE merge,
   all -inf => zero row.

Plan (`src/nvfp4_ds_mla.rs`, `_native.nvfp4_ds_mla_plan`): NS = fewest splits
that fill one wave of SMs, capped at ceil(C/64); a function of (T, HQ, C,
SMs) only (CUDA-graph stable). GLM TP=8 (HQ 8/rank), C 2048, 188 SMs:
T=1..31 -> NS 32..7, T>=188 -> NS 1 (prefill: o_part = T*HQ*1 KiB).

Host ops: `src/nvfp4_ds_mla_oxide.rs`. vLLM patch + oracle:
`sm120/nvfp4_ds_mla_patch/`. Status: `STATUS.md`.
