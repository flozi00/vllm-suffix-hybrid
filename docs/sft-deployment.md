# Deploying `SUFFIX_SFT` in a container (single GPU, small model)

A working recipe for serving a small model with the in-place SFT endpoint on
one consumer GPU, plus the four failure modes that cost the most time. The
reference deployment is LFM2.5-2.6B on a 12 GB card with vLLM 0.32.0.

The Dockerfile lives outside this repo (it needs the repo as build context):

```bash
docker build -f /path/to/Dockerfile -t lfm-sft:dev /path/to/vllm-suffix-hybrid
```

## Why a two-stage build on the same base image

`suffix_hybrid._native` is a pyo3 extension. Building it in a generic Rust
image produces a wheel linked against a different Python ABI and it fails to
import at runtime. Both stages therefore start from `vllm/vllm-openai`, and
the builder only adds rustup + maturin:

```dockerfile
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0

FROM ${VLLM_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/opt/cargo/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.9,<2"
WORKDIR /src
COPY . /src
RUN maturin build --release --locked --out /wheels

FROM ${VLLM_IMAGE}
COPY --from=builder /wheels/*.whl /tmp/plugin/
RUN pip install --no-cache-dir /tmp/plugin/*.whl && rm -rf /tmp/plugin
```

`--locked` matters: an unlocked resolve can pick a different pyo3 and change
the ABI contract with the runtime stage.

## Failure mode 1 — there is no `python` on PATH

The vLLM image ships `python3` only. A `RUN python -c ...` health check fails
the build with exit 127, and `ENTRYPOINT ["python", ...]` fails at start.
Use `python3` in both places.

## Failure mode 2 — `sitecustomize.py` is shadowed by the stdlib

The wheel installs `sitecustomize.py` into `dist-packages`, but Python
resolves `sitecustomize` from the stdlib first
(`/usr/lib/python3.12/sitecustomize.py`), so the plugin's copy never runs and
`SUFFIX_SFT=1` does nothing. Put it on `PYTHONPATH` instead:

```dockerfile
RUN mkdir -p /plugins \
    && cp /usr/local/lib/python3.12/dist-packages/sitecustomize.py /plugins/
ENV PYTHONPATH=/plugins
```

Verify with `python3 -c "import sitecustomize; print(sitecustomize.__file__)"`
— it must print `/plugins/sitecustomize.py`.

## Failure mode 3 — the endpoint plugin is never discovered

vLLM discovers endpoint plugins through the `vllm.endpoint_plugins` entry
point group, not by import. The repo ships that metadata as a bare dist-info
(the runtime bundle copies it the same way):

```
suffix_qwen_gdn_ep-1.0.dist-info/entry_points.txt
  [vllm.endpoint_plugins]
  suffix_sft = suffix_hybrid.sft.api:register
```

Without it the log still shows `[suffix sft] installed: 120 adapters, ...`
(the worker patch ran) but every `/v1/sft/*` request returns 404. Copy the
dist-info into `dist-packages` in the image.

## Failure mode 4 — `VLLM_PLUGINS` filters general plugins too

`VLLM_PLUGINS=suffix_sft` allowlists the endpoint plugin **and** disables
every other plugin. For a plain bf16 model that is what you want (none of the
`suffix_*` kernel plugins are needed). If the pod uses FP8/NVFP4 kernels,
name them all: `VLLM_PLUGINS=suffix_sft,suffix_nvfp4_gemm,...`.

## Memory budget on a 12 GB card

LFM2.5-2.6B bf16 weights are ~5.2 GB. The SFT backward pass needs headroom on
top of the KV cache, and vLLM sizes the KV pool to fill
`gpu_memory_utilization`, so raising the context window directly steals from
training.

| flag | value | why |
|---|---|---|
| `--max-model-len` | `131072` | the model's native window |
| `--kv-cache-dtype` | `fp8` | 128k needs 2.0 GiB of bf16 KV; fp8 halves it to 1.0 GiB |
| `--gpu-memory-utilization` | `0.70` | 128k bf16 KV does not fit at 0.70 (1.43 GiB available vs 2.0 needed); fp8 does |
| `SUFFIX_SFT_MIN_FREE_GIB` | `1` | 2.85 GiB free at 0.70; the default 4 refuses every run |
| `SUFFIX_SFT_MAX_TOKENS` | `8192` | micro-batch token budget, **not** the context |

Measured on a 12 GB card: 8.66 GiB resident while serving 128k, 2.85 GiB free,
`train_peak_gib` 2.58 during an 8k-token step.

### The context/training tradeoff is real

`SUFFIX_SFT_MAX_TOKENS` bounds activation memory per forward/backward, and the
cost per example is strongly superlinear because the worker checkpoints and
host-offloads layer inputs:

| example length | time per example | corpus (855 examples) |
|---|---|---|
| ~2.3k tokens | ~3 s | ~45 min |
| ~5.6k tokens | 21 s | ~5 h |
| ~24k tokens | 405 s | ~4 days |

So 128k **inference** is free, but 128k **training** is not: a 24k-token
example costs 405 s on this card. The traceforge corpus has a median trace
length of 24.5k tokens, so full-length training is a multi-day job here. The
reference run uses an 8k micro-batch budget, which keeps the task head and the
conversation tail and drops the middle.

If you need full-length training, the lever is a bigger card, not a flag: the
worker already checkpoints and offloads, and `train_peak_gib` was 10.3 GiB on
the 24k-token example.

## Verifying the seam before training

```bash
curl -s localhost:8000/v1/sft/status
# {"state":"idle","sft":{"adapters":120,"params":22953984,"rank":16,...}}
```

`adapters` and `params` non-zero means the LoRA seam is installed on the
served linears. For LFM2 the targets are `*.in_proj` / `*.out_proj` (short
conv), `*.w13` / `*.w2` (dense MLP) and `*.qkv_proj` / `*.o_proj`
(attention).

## Example shape: prefer raw prompt/completion

`encode()` splits a `messages` example by rendering the history twice and
matching the generation prompt. For LFM2.5 the template emits
`<|im_start|>assistant\n` followed by ` thinking` (no space), while
`OPEN_THINK` is `" thinking"`; replicating `encode()` locally on real traces
returned `tail is None`, i.e. the request would be rejected with *"the chat
template renders the history differently once the final assistant message is
appended"*. Short two-message examples happen to match, which makes this look
like a data problem rather than a template problem.

Raw pairs sidestep the split entirely and are what the benchmark harness
uses:

```python
base = tok.apply_chat_template(msgs[:-1], tokenize=False,
                               add_generation_prompt=False)
gen  = tok.apply_chat_template(msgs[:-1], tokenize=False,
                               add_generation_prompt=True)
full = tok.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=False)
header = "<|im_start|>assistant\n"
think_open = gen[len(base) + len(header):]      # "" or " thinking"
body = full[len(base) + len(header):]           # content + "<|im_end|>\n"
completion = ("</think>" if think_open else "") + body
# POST {"prompt": gen, "completion": completion}
```

The completion starts with `</think>` because the served model continues from
an open think block and the example answers with an empty reasoning block.

## Correctness signal to watch

Every run returns `parity_before` / `parity_after`: the mean NLL of the first
example under the serving forward vs the training forward. A healthy run is
within a few hundredths of a nat. Measured on a 2576-token real trace:
`delta 0.00484` before training, `0.00746` after. A large delta means the
training forward is not the serving forward and the run should not be trusted.
