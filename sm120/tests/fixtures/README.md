# Pinned upstream sources for CPU-verifiable probes

Verbatim upstream files used by `test_nvfp4_kv_patch.py` to prove, on CPU,
that the SM120 NVFP4-KV overlay is a no-op against these exact versions:

- `flashinfer_0.6.18.post1/` — fetched from GitHub raw at tag `v0.6.18.post1`
  (`include/flashinfer/page.cuh`, `include/flashinfer/attention/prefill.cuh`,
  `flashinfer/jit/attention/modules.py`, `flashinfer/jit/attention/utils.py`,
  `csrc/tvm_ffi_utils.h`, plus for the mm-prefix probe
  `include/flashinfer/attention/variants.cuh` and `flashinfer/prefill.py`
  — copied from an installed 0.6.18.post1 wheel whose prefill.cuh and
  modules.py are byte-identical to the fixtures above). The bundle's header probe reads the *installed*
  copies of these paths; the fixtures let the probe logic run on CI with no
  GPU and no flashinfer wheel.
- `vllm_0.30.0/flashinfer_backend.py` — `vllm/v1/attention/backends/flashinfer.py`
  at v0.30.0. The monkeypatch anchors its source edits against exact text from
  this file; the tests replay the patch over the fixture and byte-verify the
  result.

- `vllm_0.30.0/hisparse_layout.py`, `vllm_0.30.0/gpu_cudagraph_utils.py` —
  `vllm/v1/hisparse/layout.py`, `vllm/v1/worker/gpu/cudagraph_utils.py` at
  v0.30.0 (profiling-KV-init host-pool sizing, hisparse_mtp_patch rev .5).

- `vllm_0.30.0/attention_backends_utils.py` — `vllm/v1/attention/backends/utils.py`
  at v0.30.0 (`split_decodes_and_prefills`, for the HiSparse staging-plan replay).

Do not edit these files; they are regression fixtures. If upstream drifts
(newer vLLM/FI in the image), the patch layer must fail closed and a new
fixture + anchor set is required.
