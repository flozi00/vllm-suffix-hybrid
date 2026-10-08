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
    import torch
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    for name, n, k in SHAPES:
        w16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        wq, ws = dynamic_mxfp4_quant(w16)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            bf16 = _time_us(lambda: torch.nn.functional.linear(x, w16), iters)

            def fp4():
                xq, xs = dynamic_mxfp4_quant(x)
                return gemm_afp4wfp4(xq, wq, xs, ws, torch.bfloat16)

            try:
                ref = torch.nn.functional.linear(x, w16).float()
                err = ((fp4().float() - ref).norm() / ref.norm()).item()
                q4 = _time_us(fp4, iters)
                say(f"gemm {name} M={m} N={n} K={k}: bf16 {bf16:.1f} us | "
                    f"mxfp4(triton, quant incl) {q4:.1f} us | rel-l2 {err:.3f}")
            except Exception as exc:  # noqa: BLE001 - evidence, keep going
                say(f"gemm {name} M={m}: bf16 {bf16:.1f} us | mxfp4 FAILED {exc!r}")
        del w16, wq, ws
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
