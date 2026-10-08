# SPDX-License-Identifier: Apache-2.0
"""ROCm card probe (boot gate rocm_probe): what the AMD pod really has.

Prints "[suffix rocm-probe]" lines: torch/HIP/arch/CU count/VRAM, AITER version
and how many rows of its tuned configs match this CU count (MI350P = 128 CUs,
AITER ships 256-CU tables), HBM copy bandwidth, and BF16 vs AITER MXFP4 GEMM
us/call (activation quant included) at the Qwen3.8-Flash-Next TP1 decode
shapes. Each section is independent; evidence only, exit 0 unless it crashed.

    python -m suffix_hybrid.tools.rocm_probe [--m 1,4,16,64] [--iters 50]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

MARK = "[suffix rocm-probe]"
# (name, N, K) at TP1: GDN in_proj_qkvz / out_proj, attn qkv(+gate) / o,
# shared expert gate_up / down, lm_head.
SHAPES = [("gdn_in_qkvz", 16384, 2560), ("gdn_out", 2560, 6144),
          ("attn_qkv", 13312, 2560), ("attn_o", 2560, 6144),
          ("shared_gate_up", 1280, 2560), ("shared_down", 2560, 640),
          ("lm_head", 248320, 2560)]


def say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def _time_us(fn, iters: int) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def device() -> int:
    import torch

    p = torch.cuda.get_device_properties(0)
    say(f"torch {torch.__version__} hip {torch.version.hip} arch "
        f"{getattr(p, 'gcnArchName', '?')} cus {p.multi_processor_count} "
        f"vram {p.total_memory / 2**30:.1f} GiB name {p.name!r}")
    return p.multi_processor_count


def aiter_info(cus: int) -> None:
    import aiter

    root = os.path.dirname(aiter.__file__)
    say(f"aiter {getattr(aiter, '__version__', '?')} at {root}")
    for path in sorted(glob.glob(os.path.join(root, "configs", "**", "*.csv"), recursive=True)):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and "cu_num" in rows[0]:
            mine = sum(1 for r in rows if str(r["cu_num"]).strip() == str(cus))
            cu_set = sorted({str(r["cu_num"]).strip() for r in rows})
            say(f"config {os.path.relpath(path, root)}: {len(rows)} rows, "
                f"{mine} for cu_num={cus} (has {','.join(cu_set[:6])})")


def hbm(iters: int) -> None:
    import torch

    x = torch.empty(2 * 2**30, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    us = _time_us(lambda: y.copy_(x), iters)
    say(f"hbm copy 2 GiB: {2 * x.numel() / us / 1e3:.0f} GB/s (read+write)")


def gemms(ms: list[int], iters: int) -> None:
    """BF16 vs both vLLM MXFP4 linear paths, operands laid out exactly as
    vllm kernels/linear/mxfp4/aiter.py prepares them: Triton afp4wfp4 (scales
    stored [K/32, N]) and, for the ASM-eligible shapes, per_1x32 HIP quant +
    CK/ASM gemm_a4w4 on (16, 16)-shuffled weights (VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1)."""
    import torch
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    asm = None
    try:
        from aiter import gemm_a4w4, per_1x32_f4_quant_hip
        from aiter.ops.shuffle import shuffle_weight
        asm = (gemm_a4w4, per_1x32_f4_quant_hip, shuffle_weight)
    except Exception as exc:  # noqa: BLE001
        say(f"asm gemm_a4w4 unavailable: {exc!r}")
    for name, n, k in SHAPES:
        w16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        wq, ws = dynamic_mxfp4_quant(w16)
        ws_t = ws.T.contiguous()
        wq_a = ws_a = None
        if asm is not None and n % 32 == 0 and (k // 32) % 8 == 0:
            sm, sn = ws.shape
            ws_a = ws.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(
                0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
            wq_a = asm[2](wq, layout=(16, 16))
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            ref = torch.nn.functional.linear(x, w16).float()
            bf16 = _time_us(lambda: torch.nn.functional.linear(x, w16), iters)
            line = f"gemm {name} M={m} N={n} K={k}: bf16 {bf16:.1f} us"

            def tri():
                xq, xs = dynamic_mxfp4_quant(x)
                y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                gemm_afp4wfp4(xq, wq, xs, ws_t.T, torch.bfloat16, y)
                return y

            def asm_fn():
                xq, xs = asm[1](x, shuffle=True)
                return asm[0](xq, wq_a.view(xq.dtype), xs, ws_a.view(xs.dtype),
                              dtype=torch.bfloat16, bpreshuffle=True)[:m]

            for label, fn in (("triton", tri), ("asm", asm_fn if wq_a is not None else None)):
                if fn is None:
                    continue
                try:
                    err = ((fn().float() - ref).norm() / ref.norm()).item()
                    line += f" | {label} {_time_us(fn, iters):.1f} us (rel {err:.3f})"
                except Exception as exc:  # noqa: BLE001 - evidence, keep going
                    line += f" | {label} FAILED {exc!r}"[:200]
            say(line)
        del w16, wq, ws, ws_t, wq_a, ws_a
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,4,16,64")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    cus = device()
    for fn in (lambda: aiter_info(cus), lambda: hbm(a.iters),
               lambda: gemms([int(v) for v in a.m.split(",")], a.iters)):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one section failing is evidence
            say(f"section failed: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
