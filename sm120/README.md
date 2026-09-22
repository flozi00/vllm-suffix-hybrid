# sm120 — SM120 (Blackwell consumer/workstation) support

vLLM 0.30 gates DeepGEMM on SM90/SM100/SM120 (`support_deep_gemm`), and the
DeepSeek-style sparse attention indexer
(`vllm/model_executor/layers/sparse_attn_indexer.py`) hard-requires the
`deep_gemm` package for its MQA-logits kernels. The vLLM wheel vendors
`deep_gemm` under `vllm.third_party.deep_gemm`, but its JIT cannot compile the
SM90/SM100 tcgen05/TMA kernels for SM120 (`cc 12.0`) — the indexer then aborts
at layer construction.

This directory ships a directory-shadowing shim: `runtime_bundle.py` copies
`sm120/deep_gemm_shim/` to `<bundle>/deep_gemm/` so that, with
`PYTHONPATH=/plugins`, `import deep_gemm` resolves to the shim *before* any
site-packages copy (vLLM's `_import_deep_gemm` prefers an external
`deep_gemm` first). The shim:

- re-exports the vendored `vllm.third_party.deep_gemm` verbatim on every GPU
  where it works (Hopper/SM100: zero behaviour change),
- on SM120 (`torch.cuda.get_device_capability() == (12, 0)`) — or whenever the
  vendored module fails to import — serves `fp8_fp4_mqa_logits`,
  `fp8_fp4_paged_mqa_logits` and `get_paged_mqa_logits_metadata` from a
  Triton fallback that reproduces the wrapper contract in
  `vllm/utils/deep_gemm.py` exactly (FP8 path; MXFP4 raises clearly),
- keeps every other symbol as a loud proxy so a missing vendor on SM120 fails
  with a precise message instead of a silent `AttributeError`.

Engagement gate: the overrides activate when `SUFFIX_SM120=1` and the device
is SM120, or whenever the vendored module is unusable (auto — the shim is
strictly a superset of "no vendor", which already hard-fails). With the gate
unset on a working vendor the shim is inert by design, so shipping it in every
bundle is safe. `sitecustomize.py` logs the shim state once per process.

## Layout

- `deep_gemm_shim/` — becomes the top-level `deep_gemm` package in the bundle.
- `tests/` — CPU-safe contract tests (run by the repo pytest suite; CUDA
  numerics tests skip cleanly without a GPU).
