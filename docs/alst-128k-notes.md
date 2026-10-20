# ALST (Snowflake Arctic Long Sequence Training) → 128k on a 12 GB A2000

Source: https://www.snowflake.com/en/blog/engineering/arctic-long-sequence-training-multi-million-token-ai/
(plus the paper's ablation table, arXiv 2506.13996 §5.4, which the blog's Table 1 only shows as an image)

Target: LFM2.5-2.6B, 30 layers (8 full-attn + 22 short-conv), H=2048, FF=10752, V=128000,
bf16, frozen base + LoRA r=64. 11.24 GiB usable, serving 128k concurrently.

## 0. Byte budget at 128k (T=128000, bf16=2 B)

| Tensor | Shape | Bytes |
|---|---|---|
| one hidden state | [128000, 2048] | **500 MiB** |
| 30 layer inputs (what checkpointing keeps) | 30 × above | **14.65 GiB** |
| MLP gate/up intermediate | [128000, 21504] | **5.13 GiB** |
| MLP silu output | [128000, 10752] | 2.56 GiB |
| full logits, fp32 | [128000, 128000] | **65.5 GB** (impossible) |
| CE chunk @ CE_BYTES=256 MiB | [524, 128000] fp32 | 256 MiB |
| KV, 8 full-attn layers (8 kv × 64 hd) | 2·T·8·64·2 | 1.95 GiB |
| dense bool attn mask | [128000, 128000] | **16.4 GB** |
| math-kernel scores, nh=16 | [16, T, T] bf16 | 524 GB |

The dominant term is **not** attention — it is 30 × 500 MiB of layer inputs. That is why
activation-checkpoint offload, not tiling, is the make-or-break technique here.

## 1. Ablation multipliers (paper Table 1, Llama-8B, 8×H100)

| Config | Max seq | Multiplier |
|---|---|---|
| baseline (ZeRO3 + grad-ckpt + optim offload + expandable_segments + FA2) | 32K | 1× |
| + tiled logits & loss | 160K | **5.0×** |
| + Ulysses SP | 1.1M | 6.9× |
| + tiled MLP | 1.2M | 1.09× |
| + AC offload to CPU (instead of tiled MLP) | 2.4M | **2.0×** |
| + both | 3.7M | 1.54× over 2.4M |

Single-GPU result: 32K → 500K (16×) using tiled logits + tiled MLP + AC offload.
Note the ordering: tiled MLP contributes almost nothing (1.09×) *until* AC offload is on,
then adds +54%. Same will hold for us.

## 2. Technique-by-technique

### T1. Ulysses sequence parallelism — **does not apply**
Splits the sequence across GPUs, all-to-all to head-parallel for attention, all-to-all back.
Needs ≥2 ranks. On one 12 GB card there is nothing to shard. 0 bytes saved. Skip.

### T2. Sequence tiling / TiledCompute — **applies, ~60% done**
Mechanism: token-wise ops (linear, embedding lookup, per-token logits+loss) have no
cross-token mixing, so they can run in sequence tiles; peak transient becomes
`tile × width` instead of `T × width`. Exact — no numerics change.
- Already done: `chunk_ce` (256 MiB chunks, 524 tokens, checkpointed) kills the 65.5 GB
  logits; `_tiled_mlp` (tile 2048 → 84 MB vs 5.13 GiB).
- Missing: the `h = h[rows]` gather at model.py:894 materialises a 500 MiB copy at 128k
  plus a 500 MiB scatter in backward. Fold the gather into `chunk_ce` (index per chunk)
  → saves ~500 MiB.
- Missing: embedding gather / final norm are untiled. ~0.5 GiB.

### T3. Activation-checkpoint offload to CPU — **applies, already done, biggest win**
Mechanism: the tensors `torch.utils.checkpoint` retains (the layer *inputs*) are copied to
pinned host memory during forward and pulled back during backward. Saves
**14.65 GiB at 128k** (7.32 GiB at 64k). Implemented as `_Park`/`_Unpark` + `offload_left`
budget. Paper says 2.0× on its own; here it is worth more, because our per-layer input is
500 MiB and we have 30 of them.
Blog notes the copies run on the compute stream — our `non_blocking=True` matches.

### T4. `expandable_segments:True` — **applies, NOT currently set**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Reduces fragmentation at large
sequence lengths. Grep shows it is set nowhere in the repo. Free; expect 0.3–1 GiB of
headroom at 128k where allocation sizes swing between 84 MB tiles and 500 MiB states.

### T5. Avoid `all_reduce_object` — **N/A**
>3 GB/GPU overhead, but only with tensor parallelism. `world_size == 1` here; the
`_ReduceFromTP` path is never taken. No action.

### T6. PyTorch 2.6–2.7 `dist.barrier` leak (~3 GB/GPU) — **N/A, but pin the version**
Single-process, no barrier. Worth confirming the venv is not on a 2.6/2.7 build anyway.

### T7. 1D position ids instead of a 4D attention mask — **applies, mostly done**
A 4D mask is O(T²); at 128k a dense bool mask is **16.4 GB**. Code already passes
`positions` and uses `is_causal=True`. But `attention()` (model.py:668-677) builds a dense
`[n,n]` bool mask whenever `sliding_window` is set — that path would OOM instantly at 128k
and would also force SDPA off the flash kernel. `sliding_window` is only read, never set,
so it comes from the vLLM config; add a guard/assert for the 8 full-attn layers.

## 3. What the blog does NOT contain
No optimizer-state reduction (Adafactor/8-bit), no activation quantization, no
ZeRO-Offload/FSDP CPU offload for single GPU, no weight streaming/paging, no LoRA/PEFT
(they do full fine-tuning), no gradient-accumulation-vs-sequence-length analysis.
Do not expect those from this source.

## 4. Implementation sketch (ordered by GiB per unit effort)

**a) Set the offload budget — required, else silent OOM.**
`Config.offload_gib` defaults to **2.0** (`__init__.py:98`). At 128k the layer inputs are
14.65 GiB; `_checkpointed` decrements `offload_left` per layer and once exhausted keeps
inputs **on the GPU** (500 MiB each). With the default, ~4 layers park and 26 × 500 MiB =
12.7 GiB stays resident → guaranteed OOM. Set `OFFLOAD_GIB=16` (host has 127 GiB).

**b) Pinned arena instead of 30 fresh pinned allocations.**
`_Park.forward` calls `torch.empty(shape, pin_memory=True)` per layer input — 30 × 500 MiB
of pinned allocations at 128k. Pinned memory is a scarce, slow-to-grow resource on Windows;
the `except RuntimeError` fallback to pageable is **silent** and turns the step 2-3× slower
rather than failing loudly. Preallocate one arena:

```python
class _Arena:
    def __init__(self, nbytes):
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off = 0
    def take(self, shape, dtype):
        n = 1
        for s in shape: n *= s
        nb = (n * torch.empty((), dtype=dtype).element_size() + 511) & ~511
        if self.off + nb > self.buf.numel(): return None
        v = self.buf[self.off:self.off + nb].view(dtype).view(shape)
        self.off += nb
        return v
```
Set `_ARENA = _Arena(int(offload_bytes))` in `TrainMode.__enter__`, reset `off` on exit,
and have `_Park.forward` use `_ARENA.take(...)` with the old path as fallback.

**c) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the worker env. Test against
vLLM sleep mode — expandable segments and vLLM's own pool/CUDA-graph capture can interact.

**d) Fold the target gather into the CE chunk** (model.py:894 + 899):
```python
# before: h = h[rows]  -> 500 MiB copy + 500 MiB scatter
# after:  pass full h and rows; inside chunk_ce:
#         hs = h[rows[c:c + step]]
```
Saves ~500 MiB forward, same backward.

**e) Guard the dense-mask branch**: `assert sw is None or n <= 8192` in `attention()`, or
route windowed layers to a fused kernel.

## 5. Pitfalls

1. **Tiling outside a checkpoint is a memory regression.** `_tiled_mlp` returns
   `torch.cat([...])`; each tile's w13/silu intermediates are *saved for backward*
   (84 + 42 MB × 63 tiles ≈ 7.9 GiB) unless the layer is checkpointed and they are
   recomputed. It works today only because `_tiled_mlp` runs inside `_checkpointed`.
   Never lift it out.
2. **`offload_gib` default 2.0** — see 4a. Silent GPU fallback, not an error.
3. **Pinned-pool exhaustion is silent** — pageable fallback costs 2-3× step time.
4. **`_pack_saved` / `_unpack_saved` share no budget with `offload_left`.** At 128k you can
   have 14.65 GiB parked layer inputs *plus* the saved-tensor offload live at once. Fine on
   127 GiB host, but do not assume one budget covers both.
5. **Numerics**: tiling and chunked CE are exact for token-wise ops, but chunked CE changes
   fp32 reduction order — already inside the 0.15-nat parity tolerance. AC offload is
   bit-exact (pure copy). Nothing here should move the parity check.
6. **Blog numbers are Llama-8B / 80 GB H100.** The 5× from tiled logits is already banked
   by `chunk_ce`; the 2× from AC offload is already banked by `_Park`. Do not expect
   another 16× from re-applying them.
7. **Realistic ceiling**: with T3+T4+T2-remainder, 128k needs ~2 GiB of transient per layer
   (input 500 MiB + recompute 500 MiB-1 GiB + grad 500 MiB) on top of 5.47 GiB weights +
   0.61 GiB graphs. After sleep-mode frees 1.03 GiB that is ~4.6 GiB free — tight but
   plausible. 64k should fit comfortably; 128k likely needs `mlp_tile` 2048 and the
   gather fix to land.
