<!-- SPDX-License-Identifier: Apache-2.0 -->
# nvfp4_ds_mla — SM120 NVFP4 sparse-MLA reader-cache kernels (cuda-oxide track)

Writer + split-capacity attention for vLLM's `nvfp4_ds_mla` cache
(DeepSeek/GLM-style sparse MLA decode) on SM120 (cc 12.0), blind-coded
per the pinned dossier
(`plugin-harness/dossiers/dsmla-kernel-contracts-glm53.md`).

Built like the k2 family: plain `sm_120` PTX (no `a` suffix — tcgen05/UMMA and
block-scaled `mxf4nvf4` mma are NOT used; products run as **f16 `mma.sync`
m16n8k16** with `cvt.rn.satfinite.e2m1x2.f32` / `e4m3x2` SIMT conversions),
launched through the cuda-device API with raw-pointer ABI.

## Row layout — 352 B/token, EXACTLY the flashinfer contract

Per token, `uint8 [num_blocks, block_size, 352]` flat cache:

| bytes        | content                                             |
|--------------|-----------------------------------------------------|
| `[0,256)`    | 512 NoPE latent dims, e2m1-packed, **low nibble = even dim** |
| `[256,320)`  | 64 RoPE dims, e4m3, **unscaled** (no global scale)  |
| `[320,352)`  | 32 e4m3 scale factors, one per 16 latent dims, **byte-permuted `s -> 8*(s&3) + (s>>2)`** |

Quantization: `sf = e4m3(max(amax16 / 6, 2^-9))`, dequant `x = e2m1 * sf`.
`bmm1_scale`/`bmm2_scale` are applied OUTSIDE as floats (bmm2_scale is
pinned 1.0 at the call site — nothing is folded into V). There is NO global
scale byte and NO `kv_scale_format` (fp8-only concept).

## Kernels

1. **`nvfp4_ds_mla_quant_store` (writer)** — grid `(tokens,)`, block **64**
   (mirrors the SM100 kernel geometry): warp 0's 32 lanes each own one
   16-dim latent tile (infers its e4m3 SF and stores both e2m1 bytes,
   `prmt`-packing low nibble = even dim), warp 1's lanes each own 2 RoPE
   dims (32 × e4m3). ABI: `kv_c [T,512] bf16, k_pe [T,64] bf16, slot_mapping [T] i64, kv_cache u8*, block_size u32`; slot `< 0` skipped.
2. **`nvfp4_ds_mla_attn_partial` (gather attention)** — 256 threads (8 warps),
   grid = `(T, HQ_tiles, NS)` with `NS = ceil(topk / 64)`. One q token ×
   `(MTQ heads)`: 512 NoPE S-dims come from f16-mma on 256 B of gathered
   rows, 64 RoPE S-dims from 64 B of e4m3 rows; the SAME gathered rows
   dequantize V (the 512-dim latent IS the value). `topk_indices [T, C]`
   int32 holds **physical token-slot** indices into the flat cache
   (page size inferred from the kv cache's `block_size` dim); `-1` rows are
   masked (score `-inf`, split lse `-inf`). Fully 2D-block-parallel —
   CUDA-graph stable; rows are consumed via 16 B `cp.async`.
3. **`nvfp4_ds_mla_attn_merge`** — grid `(T*HQ)`, 256 threads, 512 lanes of
   the CTA front: SSE-sum reduces lse parts via max/`lg2`-delta, rescales
   each o_part split by `exp2(lse_i - lse_max)` (f32 accumulate), writes
   out [T, HQ, 512] bf16. Skips `-inf` splits.

Partial buffers: `o_part [T, HQ, NS, 512] bf16` + `lse_part [T, HQ, NS] f32`
(≈2.1 MiB/token at topk 2048 — a fraction of flashinfer's 394 MiB plan; the
patch never allocates the shared workspace). Output [T, HQ, 512] is the
concatenated nope+rope `q`-driven attention over latent values — the
MLA `V` projection is downstream in the model.

## ABI (all kernels)

Raw pointers + u32/f32 scalars, cuda_device API launch:

```
quant_store(kv_c:*u16, k_pe:*u16, slots:*i64, kv:*u8, t:*u32,
            kv_row_stride:u32, n_toks:u32, scale:f32)
attn_partial(q:*u16, kv:*u8, topk_indices:*i32, q_stride:u32,
             tok_stride:u32, topk_stride:u32, kv_stride:u32,
             o_part:*u16, lse_part:*f32, sm_scale:f32, hq:u32,
             topk_len:u32, t_pad:u32)
attn_merge(o_part:*u16, lse_part:*f32, out:*u16, sm_scale:f32, t_pad:u32, hq:u32)
```

`hq` is a runtime u32 (GLM 5.3: 64, `kv_lora_rank` 512, RoPE 64, topk 2048;
kpool/GLM-5.3-Flash out of scope v1). No per-shape cubin recompile.

## Variants

`oxide-variants.json` keeps the k2 pattern: `_w1/_w2/_w3` register/tile
variants (`w1` pins `ptxas -maxrregcount=96`). Specialization knob for
GLM's constant shapes lives in block-size features, not arch features (plain
`sm_120` PTX, matching the k2 precedent of f16 `mma.sync` over
block-scaled fp4 mma — which would need `sm_120a`).

## Numerics notes

- e2m1 values ∈ {0, .5, 1, 1.5, 2, 3, 4, ±6}, e4m3 SFs have 3 mantissa bits
  — **e2m1 × e4m3 is exact in f16** (4-bit mantissa products fit in 11
  bits), the same claim the k2 kernels rely on; all products/accumulation
  are f16/f32 mma, no f64.
- Online softmax in the exp2 domain: `exp(x) = exp2(x * log2(e))`,
  f32 max-subtraction per split, `-inf` masked rows never move the running
  max.
- SF dequant uses the k2 `prmt` LUT nibble recipe; both `bmm_scales`
  applied as float multiplies on the f32 S/P accumulators, never folded
  into the quantized rows.

## Host side

- Plan module `src/nvfp4_ds_mla.rs` (pure Rust, CPU-testable, no GPU
  imports): `block_size` inferred, `NS = ceil(topk/64)` split plan, smem &
  grid invariants, quantization-reference constants; `#[cfg(test)]` suites
  for quantization round-trip error bounds, SF permutation, plan invariants.
- Host op `src/nvfp4_ds_mla_oxide.rs` (`oxide-kernels` feature): validates
  the torch views, hands raw pointers/strides to the kernels, launches
  writer + partial + merge. Registered as `nvfp4_ds_mla_quant_store_cuda`
  / `nvfp4_ds_mla_decode_cuda`.
- vLLM patch sketch `sm120-drafts/nvfp4_ds_mla_patch.py` (gated on
  `SUFFIX_SM120_NVP4DSMLA=1`, cc 12): replaces the
  `flashinfer_trtllm_batch_decode_with_kv_cache_mla` call in
  `FlashInferMLASparseSM120Impl._run_mqa_kernel` with our op (identical
  wrapper semantics) and routes `concat_and_cache_mla` to the writer —
  count-verified fail-closed anchors, `PatchDriftError` on any drift.

## Build / validation blinds

- `cargo check` runs only where cuda-bindings' build script finds a CUDA
  13 toolkit — **TODO(CI)**: nvcc/ptxas generation against `.target sm_120`
  is untested on this host (no toolkit). AST parse + the pure-plan test
  crate pass locally.
- Open items to verify on silicon: register usage per variant at `HQ=64,
  topk=2048` (roofline vs the 394 MiB plan), `cp.async` conflict-free
  gather patterns at `block_size=64`, and end-to-end numerics vs the
  flashinfer fp8_ds_mla reference path.