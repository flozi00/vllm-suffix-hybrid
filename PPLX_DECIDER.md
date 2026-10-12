# Pplx Decider v1.1 through the existing vLLM plugin

This is an additive, default-off vLLM 0.30.0 model/endpoint plugin. No image,
Dockerfile, inference server, or replacement backbone is introduced. The owner
approved using vLLM's upstream Python API frontend for this decision pool:
the stock Rust frontend has neither the pooling engine protocol nor the
EndpointPlugin HTTP seam. All decision formatting, candidate masking,
calibration, and answer calculations run in the existing Rust extension.

## Model contract

`PplxDeciderForSequenceClassification` reuses vLLM's native multimodal
Qwen3.5 backbone. The generic pooling wrapper suppresses the vocabulary LM
head; a replicated, unquantized BF16 255x5120 readout is loaded separately.
Full attention is encoder-only/noncausal through the stock Qwen attention
constructor. GDN stays causal. LAST pooling returns raw float32 logits to
the native Rust answer helper, which masks to the request's option count
and divides by the checkpoint's saved temperature exactly once.

The V2 model-state resolver normally chooses encoder-only state before GDN
state. This model supplies `get_model_state_cls()`, composing native
`EncoderOnlyModelState` with `MambaHybridModelState` via cooperative MRO.
Encoder metadata supplements, rather than replaces, the GDN metadata and
state preparation. No kernels are replaced by this adapter.

Initial support is text-only, at most 8192 tokens, PP=1, eager execution,
fresh GDN state, and complete uncached prefills. Unsupported checkpoint
formats, temperature, token vocabulary, runner versions, or settings fail
closed. GPU parity and performance remain deployment gates; passing CPU
contracts does not prove GPU execution or model accuracy.

## Existing engine invocation

Stage the original checkpoint including `decision_config.json`,
`readout.safetensors`, tokenizer/config files, backbone safetensors and index
at `/weights`. Mount the existing CI-built runtime bundle at `/plugins`.
The console already owns that PYTHONPATH mount and plugin-fetch workflow.

```sh
SUFFIX_PPLX_DECIDER=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_USE_RUST_FRONTEND=0 \
VLLM_PLUGINS=suffix_pplx_decider,suffix_pplx_decider_endpoint \
vllm serve /weights \
  --served-model-name pplx-decider-v1.1-27b \
  --runner pooling \
  --hf-overrides '{"architectures":["PplxDeciderForSequenceClassification"],"is_causal":false}' \
  --pooler-config '{"task":"classify","seq_pooling_type":"LAST","use_activation":false}' \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --language-model-only \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --mamba-cache-mode none \
  --enforce-eager
```

Use BF16 first on a 96GB GPU or a CPU-offloaded reference pool. FP8 was
validated on one RTX 5090 with 25.24 GiB model memory: a 20-case comparison
against native vLLM BF16 matched all 13 choices, with maximum probability
drift 0.0042902. Publisher PyTorch GPU parity and performance remain
unmeasured. Add `--quantization fp8` for that arm. It uses upstream vLLM quantization and supported GEMM
kernels on the backbone, preserving the BF16 decision head. Do not enable
the speculative decoding, suffix-cache, NVFP4 KV, or GDN decode-only hooks:
decision inference consists of a prefill and readout, so those paths do
not provide its performance evidence.

For the RTX 5090 memory budget, the decision-specific offline converter
creates a native vLLM **W4A16_NVFP4** checkpoint. This is weight-only NVFP4
with BF16 activations through upstream Marlin; it requires no activation
calibration or complete model on a GPU:

```sh
python3 -m suffix_hybrid.tools.quantize_pplx_nvfp4 \
  --src /work/in --out /work/out --threads 4
```

It streams the indexed BF16 backbone on CPU in row chunks and bounded output
shards. Embeddings, vision, GDN gates/norm/state and the separate decision
head stay high precision. Decision config and tokenizer files are preserved;
original release metadata is retained as `source-release-manifest.json`,
with source hashes in `suffix_quant_manifest.json`. The head remains outside
the backbone index. Serve the converted checkpoint with the same plugin
settings and `--quantization modelopt_fp4`; the adapter accepts only the
serialized `W4A16_NVFP4` variant, preserving its BF16 readout. This path uses
native Marlin, so do not arm the custom W4A4 GEMM hook. GPU engagement,
endpoint and repeat checks passed on RTX 5090 at 15.54 GiB model memory.
All 13 choices in 20 bounded synthetic cases matched the native vLLM BF16
reference; maximum probability drift was 0.0312534. This does not establish
publisher PyTorch GPU parity, broader accuracy, or a performance speedup.
Keep `--language-model-only` for the supported text route to suppress unused
vision allocation/profiling. The conversion produced 73 objects totaling
16.04 GiB, including 59 backbone shards; the readout and decision/tokenizer
metadata were preserved. The inference-console deployment recipes and
recorded quality fixtures are the reproducible cluster configuration.
For this NVFP4 arm on RTX 5090, `--gpu-memory-utilization 0.60` passed
8192-token warmup with an 18.81 GiB engine memory budget. Marlin weight-only
compression saves memory; latency and throughput have not been benchmarked.

The normal vLLM API authentication middleware also covers this `/v1/` route.
Example request:

```sh
curl -fsS http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-decider-v1.1-27b","state":"The card payment was declined.","questions":{"route":{"type":"choice","criteria":{"billing":"Payment and invoice issues","technical":"Software troubleshooting"}}}}'
```

Check the actual request schema against the checkpoint's reference when
adding client code. The endpoint does not expose image input in this first
adapter. Generic `/classify` returns head logits when activation is disabled;
it cannot perform request-dependent candidate masking, so clients should
use `/v1/systemone` for calibrated decisions.

## vllm-sr/d3 (Decision 3.0)

`vllm-sr/d3` uses the same backbone shape, 255 codes/token IDs, LAST pooling
and noncausal full attention. Its `decision_config.json` names `prompt: "d3"`
and `format_id: "d3-code-readout-v1"`; the contract maps that to the `d3`
profile: d3's own system prompt and trailing line, `(empty)` for a missing or
empty state, any JSON value as descriptions/instructions, an optional
`videos` field, entropy confidence (choice and score) and a canonical score
legend. The readout is FP32 (`readout_dtype`, default `float32` for d3) and
the pooled hidden state is cast to FP32 before it, matching
`hidden.float() @ readout.T` in `d3_runtime.py`. Temperature is 1.0. The
pplx release keeps the `pplx` profile and BF16 readout unchanged.
`tests/test_pplx_decider_reference.py` checks prompts and answers against
d3's pinned `d3_format.py`/`d3_runtime.py` (revision `5ce995df`). The same
serve command and `quantize_pplx_nvfp4` apply.

Images and videos (d3 only): drop `--language-model-only` and serve
`--mm-processor-kwargs` exactly as `decider_media.D3_MM_PROCESSOR_KWARGS`
(1.6 MP images; video frames at 0.2 MP, `cap_pixels_per_frame`, 196 tokens
per frame pair); the model and endpoint refuse to start otherwise. The
endpoint decodes base64 PNG/JPEG/WebP and MP4/WebM/QuickTime/Matroska like
`d3_runtime.py` (OpenCV, 2 fps, 4-32 frames, even count, source indices and
fps as metadata), enforces 16,384 video tokens per request, rejects literal
placeholder tokens, and passes `multi_modal_data` so vLLM's Qwen3-VL
processor expands the placeholders (timestamps included). Usage counts the
expanded prompt. vLLM request errors (over-length media) answer 400. The d3
input limit is 32,768 tokens; pplx stays text-only at 8,192.

## Validation and shipping

```sh
python3 tests/test_decider_contract.py
python3 -m pytest tests/test_quantize_pplx_nvfp4.py
python3 scripts/test_runtime_bundle.py
cargo test --no-default-features pplx_decider
```

Ship with the existing inference-console `plugin-harness/harness.py ship`
workflow and a full immutable commit SHA. Never copy a macOS native library
to Linux. Before benchmarking, prove the model registration/readout markers
and a live completed decision request from the same serving pod. Compare
BF16 GPU decisions with the published reference on varied choice/noul/score
requests, including repeated and interleaved inputs, then compare FP8 with
that BF16 arm. Report decision latency, decisions/s, and probability/choice
agreement; generation tokens/s and speculative counters are inapplicable.
