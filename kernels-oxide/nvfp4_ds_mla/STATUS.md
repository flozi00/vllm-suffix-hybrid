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

Known risks: q bf16 -> f16 staging overflows above 65504 (absorbed q_nope
magnitudes unverified on GLM); NVFP4 vs fp8_ds_mla attention output differs
~15-18 % rel-L2 on synthetic data (format noise) — model-level eval needed.
