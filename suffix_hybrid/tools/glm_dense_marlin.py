# SPDX-License-Identifier: Apache-2.0
"""Boot gate glm_dense_marlin_bench: GLM 5.3 TP8 per-rank dense projections on ONE GPU,
BF16 cuBLAS vs our FP8 W8A8 cutlass path (+ its activation quant) vs Marlin W8A16 vs
Marlin W4A16 (sm120/fp8_dense_patch backends), cold L2 (flush 2x L2 before every
replay), CUDA-graph timed (median), numerics vs an fp32 reference.

One "[suffix dense-marlin]" line per (shape, M), a per-step sum over the target layers
per M, and the recommended SUFFIX_FP8_DENSE_MARLIN_MAX_M (largest M up to which Marlin
W8A16 beats cutlass at every smaller benched M, summed over the layers backend auto
serves). Exit 1 if a Marlin path breaks its fp8_dense_patch self-test bounds.

    python -m suffix_hybrid.tools.glm_dense_marlin
"""
from __future__ import annotations

import sys

MARK = "[suffix dense-marlin]"
# (name, N, K, calls per target forward) per TP8 rank: hidden 6144, q_lora 2048,
# kv_lora 512 + rope 64, 64 heads x (192 + 64) / 8, v_head 256, indexer 32 x 128 on the 21
# "full" indexer layers, shared expert I 2048 / 8 on the 75 MoE layers, MTP eh_proj.
SHAPES = (
    ("qkv_a", 2624, 6144, 78),
    ("q_b", 2048, 2048, 78),
    ("o_proj", 6144, 2048, 78),
    ("idx.wq_b", 4096, 2048, 21),
    ("shared.gate_up", 512, 6144, 75),
    ("shared.down", 6144, 256, 75),
    ("mtp.eh_proj", 6144, 12288, 0),
)
SERVED = ("qkv_a", "q_b", "o_proj", "idx.wq_b")  # today's FP8 layers (prod allowlist)
MS = (1, 6, 8, 32, 48, 96, 192)
ITERS = 30


def _graph_us(fn, flush, dev) -> float:
    import torch

    s = torch.cuda.Stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(ITERS)]
    for a, b in ev:
        flush()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize(dev)
    return sorted(a.elapsed_time(b) for a, b in ev)[ITERS // 2] * 1000.0


def recommend_max_m(step, ms=MS) -> int:
    """Largest benched M such that Marlin W8A16 is no slower than cutlass (summed per step
    over the served layers) at it and at every smaller benched M; 0 = never."""
    best = 0
    for m in ms:
        row = step.get(m, {})
        if not ("marlin8" in row and "cutlass" in row and row["marlin8"] <= row["cutlass"]):
            break
        best = m
    return best


def _module(n, k, packed):
    from torch import nn

    mod = nn.Module()
    for attr, t in packed.items():
        setattr(mod, attr, t)
    mod._sfx_fp8d_nk = (n, k)
    return mod


def main() -> int:
    import torch
    from torch import nn

    import vllm._custom_ops as ops
    from fp8_dense_patch import runtime as R

    R._OPS = ops
    dev = torch.device("cuda", 0)
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    flush_buf = torch.empty(max(2 * l2, 64 << 20), dtype=torch.uint8, device=dev)
    flush = flush_buf.zero_
    print(f"{MARK} {torch.cuda.get_device_name(dev)} cc {torch.cuda.get_device_capability(dev)}, "
          f"L2 {l2 >> 20} MiB (flushed with {flush_buf.numel() >> 20} MiB before every replay), "
          f"median of {ITERS} graph replays", flush=True)
    bad, step, shared = [], {m: {} for m in MS}, {m: {} for m in MS}
    for name, n, k, count in SHAPES:
        g = torch.Generator(device=dev).manual_seed(n * 7 + k)
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).bfloat16()
        q, s = R.quantize_weight(w)
        paths = {"bf16": lambda x: nn.functional.linear(x, w),
                 "cutlass": lambda x: R.fp8_linear(x, q, s)}
        errs = {}
        for tag, build in (("marlin8", lambda: R.marlin_fp8_pack(q, s)),
                           ("marlin4", lambda: R.marlin_fp4_pack(*R.fp4_quantize(w)))):
            try:
                mod = _module(n, k, build())
                paths[tag] = lambda x, mod=mod: R.marlin_linear(x, mod)
            except Exception as exc:  # noqa: BLE001 - report, keep benching the rest
                errs[tag] = f"{type(exc).__name__}: {exc}"[:160]
        for m in MS:
            x = torch.randn(m, k, device=dev, generator=g).bfloat16()
            ref = x.float() @ w.float().t()
            us, rel = {}, {}
            for tag, fn in paths.items():
                try:
                    y = fn(x)
                    torch.cuda.synchronize(dev)
                    rel[tag] = float((y.float() - ref).norm() / ref.norm())
                    us[tag] = _graph_us(lambda fn=fn: fn(x), flush, dev)
                except Exception as exc:  # noqa: BLE001
                    errs[tag] = f"{type(exc).__name__}: {exc}"[:160]
            for tag, bound in (("marlin8", R.QUANT_BOUND), ("marlin4", R.FP4_QUANT_BOUND)):
                if tag in rel and not rel[tag] <= bound:
                    bad.append(f"{name} M={m} {tag} rel {rel[tag]:.3e} > {bound}")
            cells = " | ".join(f"{t} {us[t]:7.1f}" if t in us else f"{t}     n/a"
                               for t in ("bf16", "cutlass", "marlin8", "marlin4"))
            errl = " ".join(f"{t} {rel[t]:.2e}" for t in ("bf16", "cutlass", "marlin8", "marlin4")
                            if t in rel)
            print(f"{MARK} {name:14s} N={n:5d} K={k:5d} M={m:3d}: {cells} us | rel-l2 vs fp32 {errl}",
                  flush=True)
            for t in us:
                acc = step[m] if name in SERVED else shared[m] if name.startswith("shared.") else None
                if acc is not None:
                    acc[t] = acc.get(t, 0.0) + count * us[t] / 1000.0
        for tag, e in errs.items():
            print(f"{MARK} {name}: {tag} unavailable: {e}", flush=True)
    for m in MS:
        print(f"{MARK} per-step ms over {'/'.join(SERVED)} at M={m:3d}: "
              + " | ".join(f"{t} {v:6.2f}" for t, v in step[m].items())
              + " || shared experts (bf16 today): "
              + " | ".join(f"{t} {v:6.2f}" for t, v in shared[m].items()), flush=True)
    print(f"{MARK} recommend SUFFIX_FP8_DENSE_MARLIN_MAX_M={recommend_max_m(step)} "
          "(0 = keep cutlass)", flush=True)
    for line in bad:
        print(f"{MARK} FAIL {line}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
