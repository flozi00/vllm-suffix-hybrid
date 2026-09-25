# NVFP4 ds-MLA kernel — SM120 GLM 5.3 (status ledger)

Branch: **blind-coded, NOT deployed, NOT built (CI-only), prod GLM 5.3 untouched.**
Pinned vLLM: 0.30.0 sdist at /tmp/vllm-0.30.0. Crate: `kernels-oxide/nvfp4_ds_mla`.

## Deliverables (all NEW files, zero edits to deployed files)
| Artifact | Path |
|---|---|
| Kernel crate (3 kernels: writer + split-capacity attn partial/merge) | `kernels-oxide/nvfp4_ds_mla/src/main.rs` (~860 L) |
| Kernel variants (w1 = -maxrregcount=96) | `kernels-oxide/nvfp4_ds_mla/{Cargo.toml,oxide-variants.json,README.md}` |
| Plan module (always built; 7 unit tests) | `src/nvfp4_ds_mla.rs` |
| Host ops (decode + writer pyfunctions) | `src/nvfp4_ds_mla_oxide.rs` |
| lib.rs wiring (mod decls, pyfunction reg, HAS_ marker) | `src/lib.rs` (registered under `oxide-kernels`) |
| vLLM patch sketch (fail-closed, gated SUFFIX_SM120_NVP4DSMLA=1) | `sm120-drafts/nvfp4_ds_mla_patch.py` |
| Contracts dossier (research-pinned) | `inference-console/plugin-harness/dossiers/dsmla-kernel-constandcts lead`: see `plugin-harness/dossiers/dsmla-kernel-contracts-glm53.md` |
| Audit report (post-fix) | `~/.hermes/cache/scratch/nvfp4_ds_mla_audit.md` |

## Kernel ABI (the ATK contract the patch plugs into)
- `nvfp4_ds_mla_quant_store(kv_c [T,512] bf16, k_pe [T,64] bf16, slot_mapping [T] i64, rows [slots,352] u8, tokens, kc_stride, kp_stride)`
- `nvfp4_ds_mla_attn_partial(q [T,1,HQ,576] bf16, ...)` grid `(T*HQT, NS, 1)`, block 256, dyn-smem 69,312 B
- `nvfp4_ds_mla_attn_merge(...)` grid `(T*HQ, 1, 1)`, block 256, dyn-smem `(NS+1)*4` B
- Workspace contract: o_part `[T,HQ,NS,512]` bf16 + lse_part `[T,HQ,NS]` f32, both newly allocated per call (`q.new_empty`), so no stale-memory hazard — BUG 2's exposure was narrowed to hypothetical stale workspace reuse, and BUG 3 is dead-code today (grid never pads); both fixed anyway.

## Contracts pinned during the review (audit § 5, verified)
- **352 B row**: `[0,256)` 512-dim NoPE latent e2m1 (LOW nibble = even dim), `[256,320)` 64 RoPE raw UNSCALED e4m3, `[320,352)` 32 permuted SFs, bytes via `nvfp4_ds_byte(s) = 8*(s&3)+(s>>2)`; sf = e4m3(max(amax16/6, 2^-9)); quant → dequant x = e2m1 · sf (global scale 1.0, 2^8 folded into qk_scale_log2/v_scale).
- **S-phase k-permutation (k2 precedent)**: B fragments dequant dims 32p+8t4+{0..7} (u32 at byte 16p+4t4) into hw k {2t4,2t4+1,2t4+8,(+16)}; Q staging must write physical dim 32blk+8a+4b+2e+f at smem column 32blk+16b+8e+2a+f — else every attention score scrambles dims (critical bug fixed in blind code; k2's k-inverse `kl = 16(2P+b)+8e+2a+f` is the silicon-proven ground truth).
- **lse merge contract**: partials emit `lse = m + lg2(l)`; merge `w_s = ex2(lse_s − mx)`; all-(-inf) ⇒ o=0 (defined).
- **`-1` slots**: skip row gather (zero-staged rows, SF zeroed too ⇒ 0·NaN safe); lse `-inf` for -1-only splits via empty_exit (padding tokens now actually write it).
- **Graph stability**: grid = f(T, HQ, C) only; ns = ceil(C/64) host-side; all i32 topk indices physical token slots.

## Remaining verification (silicon-gated; blind coding stands)
1. CI builds cubins (cargo oxide → PTX sm_120 → ptxas 13.0) — requires CUDA 13 toolkit, not on this Mac.
2. Kernel numerics vs fp8_ds_mla reference (oracle harness in sm120/fixtures, to be run on-silicon by a later task).
3. ptxas register report at HQ=64/K=2048 (w1 pins 96) — check spills via `cuobjdump -res-usage`.
4. Numerics vs fp8_ds_mla reference — on-silicon oracle comparison, queued post-merge.