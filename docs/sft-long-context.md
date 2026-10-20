# Long-context training on a 12 GB card

Goal: train at half the served context (serve 128k → train 64k). This is what
the card actually does, what the TRL / Unsloth / Liger tricks buy here, and
where the wall is.

All numbers measured on the reference box: **RTX A2000 12 GB** (70 W power
limit, 2100 MHz max SM clock, ~1815 MHz sustained, 84 °C under load),
LFM2.5-2.6B, LoRA r=16, serving at `--max-model-len 131072`,
`--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.70`.

## What the training forward is

`suffix_hybrid/sft/model.py` is a hand-written training path, not HF Trainer:
it swaps `forward` on the decoder layers, checkpoints each layer, parks the
layer inputs in pinned host memory, and runs a chunked vocab-parallel CE. So
none of the three libraries can be `pip install`ed into it — their *techniques*
have to be ported into that loop. That is what the sections below do.

## The three libraries, and what actually applies

| Technique | Where it comes from | Applies here? |
|---|---|---|
| Fused linear cross-entropy (chunked logits, no `[T, vocab]` fp32 tensor) | Liger | **Already present** — `chunk_ce` + `CE_BYTES` |
| Fused RMSNorm / SwiGLU / RoPE (fewer memory-bound kernels) | Liger, Unsloth | Marginal: the peak is activations, not kernel transients |
| Gradient checkpointing + `use_reentrant=False` | Unsloth, TRL | **Already present** — per decoder layer |
| Offload checkpointed activations to host | Unsloth (`offload_activations`) | **Already present** — `_Park` / `_Unpark` |
| Sequence-tiled MLP (bound the `[T, 2*ff]` intermediate) | — | **Added here** — `_tiled_mlp`, `SUFFIX_SFT_MLP_TILE` |
| 4-bit / 8-bit frozen base weights (QLoRA) | Unsloth, bitsandbytes | **Not present** — the biggest remaining lever, see below |
| CPU-offload of frozen base weights, streamed per layer | DeepSpeed ZeRO-Offload, FSDP | **Not present** — the other big lever |
| Free the KV cache for the duration of a run | — (vLLM sleep mode) | **Added here** — `SUFFIX_SFT_FREE_KV` |
| Paged / 8-bit optimizer states | TRL + bitsandbytes | Already host-side: fp32 master + AdamW moments live in pinned host memory |

The two that matter and are *not* implemented are both about the **frozen base
weights** (5.47 GiB of the 11.24 GiB card). Everything else is already done or
is a rounding error.

## Measured: what fits

Serving idle at 128k/fp8/util 0.70: **8648 MiB** of 11514 MiB.

| Micro-batch | Time / example | Forward | Backward | Peak (nvidia-smi) |
|---|---|---|---|---|
| 16k tokens | 207 s | 36 s | 167 s | ~10.4 GiB |
| 32k tokens | 433–527 s | 122 s | 400 s | **11312 MiB (98 % of card)** |
| 50k tokens | killed at 50 min | — | — | thrashing, ~200 MiB free |

Scaling is **superlinear in time** (2× tokens → 2.5× time: the 8 full-attention
layers are O(T²)) and **linear in memory** (2× tokens → 2× activation peak).

At 64k the activation peak doubles again: 32k already sits at 98 % of the card,
so 64k needs roughly **2 GiB more headroom than exists**.

## Measured: the KV-cache offload

`SUFFIX_SFT_FREE_KV=1` unmaps the KV cache for the duration of a run and maps
it back afterwards, using vLLM's own sleep mode:

```
worker.sleep(1)              # weights -> host, KV cache discarded
worker.wake_up(["weights"])  # weights back, KV cache stays unmapped
... train ...
worker.wake_up()             # KV cache re-mapped at the same VA
```

`kv_cache` is in `DEFERRABLE_TAGS`, so `wake_up(["weights"])` leaves it asleep;
`cuMemMap` restores it at the same virtual address, so the captured CUDA graphs
stay valid. Verified in the container log:

```
CuMemAllocator: sleep freed 0.00 GiB ... (weights are offloaded, not discarded)
Sleep mode freed 1.03 GiB memory, 9.29 GiB memory is still in use.
```

**Gain: 1.03 GiB.** Real, but it is 1 GiB against a 2 GiB shortfall, and the
training activations immediately expand into it (`min_free_gib` stays ~1.9).

Serving is paused while the cache is unmapped — the run owns the card. That is
the intended trade: `SUFFIX_SFT_DUTY` and `yield_to_serving` become moot.

## Measured: the corpus at 64k

`build_examples.py --max-tokens 64000` over the 955-trace corpus:

```
train 955  held-out 0  dropped {'unusable': 0, 'too_long': 0, 'no_split': 0}
tokens: min 145  median 25347  max 50901
```

**Nothing is dropped and nothing is truncated at 64k.** At the 4k budget the
reference run used, the median trace was cut from 25 347 to ~2 300 tokens —
90 % of every example was thrown away. That is the quality argument for 64k,
independent of the memory argument.

## Where the wall is

| Consumer | GiB |
|---|---|
| Base weights (bf16) | 5.47 |
| KV cache (128k, fp8) | 1.38 → **0 with `FREE_KV`** |
| CUDA graphs | 0.61 |
| Peak activation (32k) | ~2.0 |
| **Total at 32k** | **11.05 of 11.24** |

To reach 64k, one of these has to give:

1. **Offload the frozen base weights** (5.47 GiB) and stream each layer's
   weights H2D on demand. The base is frozen — LoRA only needs it transiently.
   Cost: ~22 GB of PCIe traffic per optimizer step (~2 s at 10 GB/s), which is
   noise next to a 9-minute step. **This is the change that unlocks 64k.**
2. **4-bit base weights** (5.47 → ~1.4 GiB). Frees 4 GiB, but adds
   bitsandbytes and a dequant path to the training forward, and breaks the
   served-vs-trained parity check that currently holds at 0.0037 nats.
3. **A bigger card.** The 70 W power limit is also why a 32k step takes 9
   minutes; a 24 GB card would fit 64k with no code change at all.

## Throughput reality

Even once 64k fits, this card does ~9 min per 32k example. A 955-example epoch
at 64k is on the order of **100+ hours**. Long-context training here is a
correctness/quality tool, not a throughput tool — run it on a rented GPU for
real epochs and use the local card to prove the recipe.

## Reproducing

```bash
# 64k examples: nothing dropped, nothing truncated
docker exec lfm-sft python3 /work/sft/build_examples.py \
  --dataset /data/train.jsonl --out /work/sft/examples64k.jsonl --max-tokens 64000

# one 32k step with the KV cache freed
SUFFIX_SFT_MAX_TOKENS=65536 SUFFIX_SFT_FREE_KV=1 \
SUFFIX_SFT_MLP_TILE=4096 SUFFIX_SFT_OFFLOAD_GIB=10 bash run.sh
curl -X POST localhost:8000/v1/sft/train --data-binary @results/one_32k.json
```

`memory` in the response reports `kv_freed_gib`, `fwd_peak_gib`, `min_free_gib`
and `host_pinned_gib` alongside the loss.
