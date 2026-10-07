# SPDX-License-Identifier: Apache-2.0
"""FP8-dense on-silicon gate: GLM 5.3 TP=8 per-rank linear shapes, one GPU.

    python -m fp8_dense_patch.oracle [--json] [--ms 6,12,24,48,96,192]

Per layer type, random BF16 weights (N(0, 0.02)) and activations (N(0, 1)
with 8 outlier channels x20): max over M of
  q_err = rel. Frobenius(FP8 path, BF16 cuBLAS F.linear)   <= QUANT_BOUND
  k_err = rel. Frobenius(FP8 path, exact-dequant fp32 ref)   <= KERNEL_BOUND
(q_err is the e4m3 W8A8 floor, ~3.5 % on Gaussian data; 2e-2 is below what
any per-channel/per-token e4m3 GEMM can reach, k_err is the kernel check).
Latency: BF16 F.linear vs FP8 (quant + cutlass_scaled_mm) and mm-only, CUDA
graph of R calls over rotating weight copies (> 2x the 128 MB L2, as in a
real step), CUDA events. Exit 0 = numerics PASS (latency is evidence).
"""

import argparse
import json
import math
import sys

MARK = "[suffix fp8-dense]"

# (layer type, N, K per rank at TP=8, calls per decode step: 78 main layers
#  + MTP k=5 drafter passes). GLM 5.3: hidden 6144, q_lora 2048, kv_lora 512
#  + rope 64, 64 heads x (qk 256 / v 256), indexer 32 x 128 on 21 layers
#  (+ MTP step 0 only: index_share_for_mtp_iteration), dense MLP 12288 on
#  layers 0-2, shared expert 2048 on 75 MoE layers + MTP.
INVENTORY = [
    ("fused_qkv_a_proj", 2624, 6144, 78 + 5),   # replicated
    ("q_b_proj", 2048, 2048, 78 + 5),
    ("o_proj", 6144, 2048, 78 + 5),
    ("indexer.wq_b", 4096, 2048, 21 + 1),       # replicated
    ("dense.gate_up_proj", 3072, 6144, 3),
    ("dense.down_proj", 6144, 1536, 3),
    ("shared.gate_up_proj", 512, 6144, 75 + 5),
    ("shared.down_proj", 6144, 256, 75 + 5),
    ("mtp.eh_proj", 6144, 12288, 5),            # replicated nn.Linear
]


def _graph_time(fn, reps, iters):
    """us per fn() call: R calls captured in one CUDA graph, replayed."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (iters * reps)


def run(ms, iters, seed=0):
    import torch
    from vllm import _custom_ops

    from . import runtime as R

    R._OPS = _custom_ops
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    rows, ok = [], True
    for name, n, k, calls in INVENTORY:
        copies = max(2, min(256, math.ceil(512 * 2**20 / (n * k * 2))))
        ws = [(torch.randn(n, k, device=dev) * 0.02).bfloat16() for _ in range(copies)]
        qs = [R.quantize_weight(w) for w in ws]
        row = {"layer": name, "N": n, "K": k, "calls": calls, "q_err": 0.0,
               "k_err": 0.0, "us": {}}
        try:
            for m in ms:
                x = torch.randn(m, k, device=dev)
                x[:, :8] *= 20.0
                x = x.bfloat16()
                out = R.fp8_linear(x, *qs[0]).float()
                row["q_err"] = max(row["q_err"], R.rel_frob(
                    out, torch.nn.functional.linear(x, ws[0]).float()))
                row["k_err"] = max(row["k_err"], R.rel_frob(out, R.reference(x, *qs[0])))
                if not bool(torch.isfinite(out).all()):
                    row["k_err"] = float("inf")
                xq, xs = _custom_ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                reps = copies * max(1, 32 // copies)
                t_bf16 = _graph_time(
                    lambda i: torch.nn.functional.linear(x, ws[i % copies]), reps, iters)
                t_fp8 = _graph_time(lambda i: R.fp8_linear(x, *qs[i % copies]), reps, iters)
                t_mm = _graph_time(lambda i: _custom_ops.cutlass_scaled_mm(
                    xq, qs[i % copies][0].t(), xs, qs[i % copies][1],
                    out_dtype=torch.bfloat16), reps, iters)
                row["us"][m] = {"bf16": round(t_bf16, 2), "fp8": round(t_fp8, 2),
                                "fp8_mm": round(t_mm, 2)}
        except Exception as exc:  # e.g. cutlass_gemm_caller Invalid status
            row["error"] = repr(exc)
        row["pass"] = ("error" not in row and row["q_err"] <= R.QUANT_BOUND
                       and row["k_err"] <= R.KERNEL_BOUND)
        ok &= row["pass"]
        rows.append(row)
        del ws, qs
        torch.cuda.empty_cache()
    return ok, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ms", type=lambda s: [int(x) for x in s.split(",")],
                    default=[6, 12, 24, 48, 96, 192])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"{MARK} FP8-DENSE ORACLE NOT RUN: no CUDA device", flush=True)
            return 2
        cap = torch.cuda.get_device_capability()
        ok, rows = run(args.ms, args.iters)
    except Exception as exc:
        print(f"{MARK} FP8-DENSE ORACLE FAIL: {exc!r}", flush=True)
        return 1
    from .runtime import KERNEL_BOUND, QUANT_BOUND

    print(f"{MARK} sm_{cap[0]}{cap[1]} bounds q_err<={QUANT_BOUND} "
          f"k_err<={KERNEL_BOUND}; us/call bf16 | fp8(quant+mm) | fp8 mm-only",
          flush=True)
    for r in rows:
        lat = " ".join(f"M{m}:{v['bf16']}|{v['fp8']}|{v['fp8_mm']}"
                       for m, v in r["us"].items())
        print(f"{MARK} {'PASS' if r['pass'] else 'FAIL'} {r['layer']:<20} "
              f"{r['N']}x{r['K']} x{r['calls']}/step q_err {r['q_err']:.3e} "
              f"k_err {r['k_err']:.3e} {lat}"
              + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
    for m in args.ms:  # per-step estimate at this M, every layer converted
        per = {r["layer"]: r["calls"] * (r["us"][m]["bf16"] - r["us"][m]["fp8"]) / 1e3
               for r in rows if m in r["us"]}
        wins = {k: round(v, 3) for k, v in per.items() if v > 0}
        print(f"{MARK} M={m}: est. step saving {sum(per.values()):.2f} ms all "
              f"layers, {sum(wins.values()):.2f} ms winners only "
              f"(losers: {sorted(set(per) - set(wins)) or 'none'})", flush=True)
    if args.json:
        print(json.dumps({"pass": ok, "results": rows}), flush=True)
    print(f"{MARK} FP8-DENSE ORACLE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
