# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP4 linears at small M in one AITER launch (SUFFIX_ROCM_MXFP4_A16=1, ROCm).

vLLM's AiterMxfp4LinearKernel (non-ASM path) runs every dense W4A4 linear as Triton
dynamic_mxfp4_quant + gemm_afp4wfp4 (+ a split-K reduce at M <= 8): two or three
launches, ~180 quant launches per Qwen3.8-Flash-Next decode step. With the gate, the
rocm_patches rewrite of gemm_with_dynamic_quant hands M <= SUFFIX_ROCM_MXFP4_A16_MAX_M
(default 32) to gemm_a16(): AITER gemm_a16wfp4, which quantizes the BF16 activations
inside the GEMM, on vLLM's layouts as loaded (weight [N, K/2], weight_scale [K/32, N]
passed .T like the stock call). Pre-quantized x_scales and the ASM path stay stock.

Numerics: every K tile goes through the same _mxfp4_quant_op as dynamic_mxfp4_quant
("even" E8M0 scale, RNE E2M1, fp32 math, the same 32-aligned groups), so dot_scaled
sees bit-identical activation codes and scales; only the fp32 summation order (split-K,
MFMA shape) can differ: rare one-ulp bf16 output differences.
Graph safety: no host sync, allocations depend on M only, atomic_add=False (no zeroed
output, deterministic). EVEN_K is enforced (_config): the a16 kernel loads weight
scales without a K mask, so a K tail (K=640 at AITER's BLOCK_K 512) would read past
weight_scale, where a 0xFF byte is an E8M0 NaN that dot_scaled propagates.

    python -m suffix_hybrid.kernels.mxfp4_a16_rocm   # GPU oracle + us/call (boot gate mxfp4_a16_bench)
"""
from __future__ import annotations

import functools
import os

MAX_M = int(os.environ.get("SUFFIX_ROCM_MXFP4_A16_MAX_M", "32"))
MARKER = "[suffix mxfp4-a16]"


def _even_k(cfg: dict, k: int) -> dict:
    """cfg such that the gemm_a16wfp4 wrapper launches EVEN_K (it re-runs get_splitk itself)."""
    from aiter.ops.triton.gemm_a16wfp4 import get_splitk

    if cfg["NUM_KSPLIT"] > 1:  # kept if the wrapper's split (fp32 partials + reduce) is even
        spk, bk, _ = get_splitk(k // 2, cfg["BLOCK_SIZE_K"], cfg["NUM_KSPLIT"])
        if k % spk == 0 and spk % bk == 0 and bk >= 64:
            return cfg
        cfg["NUM_KSPLIT"] = 1
    while k % cfg["BLOCK_SIZE_K"]:
        cfg["BLOCK_SIZE_K"] //= 2
    return cfg


@functools.lru_cache(maxsize=None)
def _config(m: int, n: int, k: int) -> dict:
    """AITER's own gemm_a16wfp4 config (DEFAULT.json or a tuned N/K JSON), made EVEN_K."""
    from aiter.ops.triton.gemm_a16wfp4 import _get_config

    return _even_k(_get_config(m, n, k // 2)[0], k)


def gemm_a16(x, weight, weight_scale, out_dtype, max_m: int = MAX_M):
    """y = x @ W.T for vLLM's non-ASM MXFP4 layouts, or None: caller runs the stock path."""
    k = weight.shape[1] * 2
    if x.shape[0] > max_m or k % 64:
        return None
    from aiter.ops.triton.gemm_a16wfp4 import gemm_a16wfp4

    return gemm_a16wfp4(x, weight, weight_scale.T, False, out_dtype, None,
                        _config(x.shape[0], weight.shape[0], k))


SHAPES = ((16384, 2560, "in_proj_qkvz"), (2560, 6144, "out_proj/o_proj"),
          (13312, 2560, "qkv_proj"), (1280, 2560, "shared gate_up"), (2560, 640, "shared down"))
MS = (1, 5, 8, 16, 32, 40)


def _graph_us(fn, x, weights, reps: int = 5):
    """us/call of fn over `weights` (copies of one weight: cold reads, as in decode) in one
    HIP graph, and the last call's replayed output."""
    import torch

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w, s in weights:
            out = fn(x, w, s)
    g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * len(weights)), out


def main() -> int:
    """Oracle on silicon: stock (vLLM's dynamic_mxfp4_quant + gemm_afp4wfp4 call) vs gemm_a16.
    Pass per case: a16 finite; |a16 - stock| <= 2^-7 max(|a16|, |stock|) + 2^-16 max|stock|
    elementwise (one bf16 ulp + an accumulation-order floor near zero); >= 99% of elements
    bitwise equal (same quantized operands); HIP-graph replay == eager bitwise.
    Timing only: "ksN" = the same a16 tile with a non-atomic split-K filling the CUs (AITER's
    DEFAULT never splits); if it wins, ship it as a tuned GEMM-A16WFP4-N=..-K=.. JSON."""
    import torch
    from aiter.ops.triton.gemm_a16wfp4 import gemm_a16wfp4, get_splitk
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    def stock(x, weight, weight_scale):  # vLLM 81198e97 gemm_with_dynamic_quant, non-ASM branch
        x_q, x_s = dynamic_mxfp4_quant(x)
        y = torch.empty(x_q.shape[0], weight.shape[0], device=x_q.device, dtype=torch.bfloat16)
        gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, torch.bfloat16, y)
        return y

    def a16(x, weight, weight_scale):
        return gemm_a16(x, weight, weight_scale, torch.bfloat16, max_m=1 << 30)

    dev, gen = "cuda", torch.Generator(device="cuda").manual_seed(0)
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    failed, wins = False, dict.fromkeys(MS, True)
    for n, k, name in SHAPES:
        w = torch.randn(n, k, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
        wq, ws = dynamic_mxfp4_quant(w)  # checkpoint layout: [N, K/2] e2m1 pairs, [N, K/32] e8m0
        ws = ws.T.contiguous()  # process_weights_after_loading, non-ASM: [K/32, N]
        copies = [(wq.clone(), ws.clone()) for _ in range(max(1, min(512, (1 << 30) // wq.numel())))]
        for m in MS:
            # per-group scales 2^-6..2^6 (many E8M0 values); half of the last row zero (padding)
            x = (torch.randn(m, k // 32, 32, generator=gen, device=dev) * torch.exp2(
                torch.randint(-6, 7, (m, k // 32, 1), generator=gen, device=dev).float())
                 ).reshape(m, k).to(torch.bfloat16)
            x[-1, :k // 2] = 0
            ref, out = stock(x, wq, ws).float(), a16(x, wq, ws).float()
            diff = (out - ref).abs()
            tol = 2 ** -7 * torch.maximum(out.abs(), ref.abs()) + 2 ** -16 * ref.abs().max()
            same = (out == ref).float().mean().item()
            t_stock, _ = _graph_us(stock, x, copies)
            t_a16, replayed = _graph_us(a16, x, copies)
            ok = (bool(torch.isfinite(out).all()) and bool((diff <= tol).all()) and same >= 0.99
                  and torch.equal(replayed, a16(x, *copies[-1])))
            failed |= not ok
            wins[m] &= t_a16 < t_stock
            c = _config(m, n, k)
            tiles = -(-m // c["BLOCK_SIZE_M"]) * -(-n // c["BLOCK_SIZE_N"])
            split = _even_k(dict(c, NUM_KSPLIT=max(2, min(8, cus // tiles))), k)
            t_split, _ = _graph_us(lambda x, w, s: gemm_a16wfp4(
                x, w, s.T, False, torch.bfloat16, None, split), x, copies)
            ks = (get_splitk(k // 2, split["BLOCK_SIZE_K"], split["NUM_KSPLIT"])[2]
                  if split["NUM_KSPLIT"] > 1 else 1)  # what the wrapper runs
            print(f"{MARKER} N={n} K={k} M={m}: max abs diff {diff.max().item():.2e} rel "
                  f"{(diff.norm() / ref.norm()).item():.1e} | stock {t_stock:.1f} us -> a16 "
                  f"{t_a16:.1f} us | bitwise {same:.2%} | cfg {c['BLOCK_SIZE_M']}x"
                  f"{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']} ks{c['NUM_KSPLIT']}, ks{ks} "
                  f"{t_split:.1f} us | {name}{'' if ok else ' FAIL'}",
                  flush=True)
        del copies
    print(f"{MARKER} a16 faster on every shape at M in {[m for m in MS if wins[m]]}", flush=True)
    print(f"{MARKER} oracle {'FAIL' if failed else 'PASS'} (|d| <= 2^-7 max|y| + 2^-16 max|stock|, "
          ">= 99% bitwise, replay == eager)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
