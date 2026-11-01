# SPDX-License-Identifier: Apache-2.0
"""NVFP4-dense on-silicon gate: Gemma-4-26B-A4B (TP1) and Gemma-4-31B (TP2)
per-rank linear shapes, one GPU, through the SAME per-layer conversion the
patch runs at load (nvfp4.convert_layer -> vLLM ModelOptLinearMethod NVFP4
W4A4, vLLM's kernel selection; SUFFIX_* env is stripped by the boot-gate
runner, so this is the stock kernel, not SUFFIX_NVFP4_GEMM).

    python -m fp8_dense_patch.nvfp4_oracle [--json] [--ms 1,2,4,8,16,32,64]

Per layer type, random BF16 weights N(0, 0.02), activations N(0, 1) with 8
outlier channels x20 (amax <~ 100), static activation amax A = 256:
  q_err  = rel. Frobenius(NVFP4 layer, BF16 F.linear), max over M <= QUANT_BOUND
           (W4A4 floor ~13.4 % on this data; 0.16)
  k_err  = rel. Frobenius(NVFP4 layer, exact-dequant fp32 ref), max over M
           <= KERNEL_BOUND (2e-2: kernel / layout / scale plumbing check)
  wq_mis = fraction of e2m1 codes where vLLM scaled_fp4_quant (checkpoint
           layout) differs from the repo's numpy-verified torch quantizer (a
           differing block scale counts 16); <= 1e-2 (exact-midpoint ties
           round differently under rcp.approx: ~0.3 % on bf16 weights)
  head   = q_err at M=16 with A = amax(x) x {1, 2^8, 2^12, 2^14}: the price of
           the static conservative activation global scale (evidence only).
Latency: us/call BF16 F.linear | NVFP4 (quant + GEMM) | FP8 W8A8 (fp8-dense
runtime: per-token quant + cutlass_scaled_mm), CUDA graph over rotating weight
copies (> 2x the 128 MB L2), CUDA events. Exit 0 = numerics PASS.
"""

import argparse
import json
import math
import sys

MARK = "[suffix nvfp4-dense]"
ACT_AMAX = 256.0
HEADROOM = (1, 2 ** 8, 2 ** 12, 2 ** 14)

# (model, layer, N, K per rank, calls per decode step). HF text_config:
# 26B-A4B hidden 2816, 16 heads, sliding head_dim 256 x 8 kv / full (5 of 30)
# global_head_dim 512 x 2 kv, dense MLP 2112 (+ NVFP4 MoE, not here).
# 31B TP2: hidden 5376, 32 -> 16 heads/rank, sliding 256 x 16 -> 8 kv/rank,
# full (10 of 60) 512 x 4 -> 2 kv/rank, MLP 21504 -> 10752/rank.
INVENTORY = [
    ("g26", "qkv_proj.sliding", 8192, 2816, 25),
    ("g26", "qkv_proj.full", 10240, 2816, 5),
    ("g26", "o_proj.sliding", 2816, 4096, 25),
    ("g26", "o_proj.full", 2816, 8192, 5),
    ("g26", "mlp.gate_up_proj", 4224, 2816, 30),
    ("g26", "mlp.down_proj", 2816, 2112, 30),
    ("g31tp2", "qkv_proj.sliding", 8192, 5376, 50),
    ("g31tp2", "qkv_proj.full", 10240, 5376, 10),
    ("g31tp2", "o_proj.sliding", 5376, 4096, 50),
    ("g31tp2", "o_proj.full", 5376, 8192, 10),
    ("g31tp2", "mlp.gate_up_proj", 21504, 5376, 60),
    ("g31tp2", "mlp.down_proj", 5376, 10752, 60),
]


def _dist_init():
    """BasevLLMParameter reads the TP rank: single-rank groups, like vLLM's tests."""
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="tcp://127.0.0.1:0")
    ensure_model_parallel_initialized(1, 1)


def _layer(w, act_amax):
    """A bare module with the LinearBase attributes convert_layer reads."""
    import torch

    from . import nvfp4 as N

    m = torch.nn.Module()
    n, k = w.shape
    m.weight = torch.nn.Parameter(w, requires_grad=False)
    m.bias = None
    m.input_size = m.input_size_per_partition = k
    m.output_size = n
    m.output_partition_sizes = [n]
    m.params_dtype = torch.bfloat16
    m.weight_loader = lambda *a, **kw: None
    q, sf, ws2 = N.convert_layer("oracle", m, act_amax)
    return m, (q, sf, ws2)


def _wq_mismatch(w, q, sf):
    """Fraction of e2m1 codes that differ from the torch reference, a differing
    block scale counting all 16 of its codes."""
    import torch

    from suffix_hybrid.tools.quantize_nvfp4 import quantize_weight as qw

    amax = float(w.abs().amax())
    bad = 0
    for i in range(0, w.shape[0], 2048):
        rq, rsf, _ = qw(w[i:i + 2048], amax)
        d = rq ^ q[i:i + 2048]
        bad += int(((d & 0xF) != 0).sum() + ((d >> 4) != 0).sum())
        bad += 16 * int((rsf.view(torch.uint8) != sf[i:i + 2048].view(torch.uint8)).sum())
    return bad / (2 * q.numel())


def run(ms, iters, seed=0):
    import torch
    from vllm import _custom_ops

    from . import nvfp4 as N
    from . import runtime as R
    from .oracle import _graph_time

    N._OPS = R._OPS = _custom_ops
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    rows, ok = [], True
    for model, name, n, k, calls in INVENTORY:
        copies = max(2, min(256, math.ceil(512 * 2**20 / (n * k * 2))))
        ws = [(torch.randn(n, k, device=dev) * 0.02).bfloat16() for _ in range(copies)]
        row = {"model": model, "layer": name, "N": n, "K": k, "calls": calls,
               "q_err": 0.0, "k_err": 0.0, "wq_mis": 1.0, "head": {}, "us": {}}
        try:
            layers = [_layer(w.clone(), ACT_AMAX) for w in ws]
            f8 = [R.quantize_weight(w) for w in ws]
            lay0, (q0, sf0, ws20) = layers[0]
            row["wq_mis"] = _wq_mismatch(ws[0], q0, sf0)
            nv = lambda lay, x: lay.quant_method.apply(lay, x)  # noqa: E731
            for m in ms:
                x = torch.randn(m, k, device=dev)
                x[:, :8] *= 20.0
                x = x.bfloat16()
                out = nv(lay0, x).float()
                row["q_err"] = max(row["q_err"], N.rel_frob(
                    out, torch.nn.functional.linear(x, ws[0]).float()))
                row["k_err"] = max(row["k_err"], N.rel_frob(
                    out, N.reference(x, q0, sf0, ws20, N.layer_act_amax(lay0))))
                if not bool(torch.isfinite(out).all()):
                    row["k_err"] = float("inf")
                reps = copies * max(1, 32 // copies)
                t_bf16 = _graph_time(
                    lambda i: torch.nn.functional.linear(x, ws[i % copies]), reps, iters)
                t_nv = _graph_time(lambda i: nv(layers[i % copies][0], x), reps, iters)
                t_f8 = _graph_time(lambda i: R.fp8_linear(x, *f8[i % copies]), reps, iters)
                row["us"][m] = {"bf16": round(t_bf16, 2), "nvfp4": round(t_nv, 2),
                                "fp8": round(t_f8, 2)}
            x = torch.randn(16, k, device=dev)
            x[:, :8] *= 20.0
            x = x.bfloat16()
            ref = torch.nn.functional.linear(x, ws[0]).float()
            amax = float(x.abs().amax())
            for h in HEADROOM:
                lay, _ = _layer(ws[0].clone(), amax * h)
                row["head"][h] = round(N.rel_frob(nv(lay, x).float(), ref), 4)
                del lay
        except Exception as exc:  # kernel selection / Invalid status / OOM
            row["error"] = repr(exc)
        row["pass"] = ("error" not in row and row["q_err"] <= N.QUANT_BOUND
                       and row["k_err"] <= N.KERNEL_BOUND and row["wq_mis"] <= 1e-2)
        ok &= row["pass"]
        rows.append(row)
        layers = f8 = ws = None
        torch.cuda.empty_cache()
    return ok, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ms", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"{MARK} NVFP4-DENSE ORACLE NOT RUN: no CUDA device", flush=True)
            return 2
        cap = torch.cuda.get_device_capability()
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            _dist_init()
            ok, rows = run(args.ms, args.iters)
    except Exception as exc:
        print(f"{MARK} NVFP4-DENSE ORACLE FAIL: {exc!r}", flush=True)
        return 1
    from .nvfp4 import KERNEL_BOUND, QUANT_BOUND

    print(f"{MARK} sm_{cap[0]}{cap[1]} bounds q_err<={QUANT_BOUND} k_err<={KERNEL_BOUND} "
          f"wq_mis<=1e-2; A={ACT_AMAX}; head = q_err at A=amax x {HEADROOM}; "
          "us/call bf16 | nvfp4 | fp8", flush=True)
    for r in rows:
        lat = " ".join(f"M{m}:{v['bf16']}|{v['nvfp4']}|{v['fp8']}" for m, v in r["us"].items())
        print(f"{MARK} {'PASS' if r['pass'] else 'FAIL'} {r['model']:<7}{r['layer']:<18} "
              f"{r['N']}x{r['K']} x{r['calls']}/step q_err {r['q_err']:.3e} k_err "
              f"{r['k_err']:.3e} wq_mis {r['wq_mis']:.1e} head "
              f"{'/'.join(str(v) for v in r['head'].values())} {lat}"
              + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
    for model in sorted({r["model"] for r in rows}):
        for m in args.ms:  # per-step estimate, every listed layer converted
            rs = [r for r in rows if r["model"] == model and m in r["us"]]
            d_nv = sum(r["calls"] * (r["us"][m]["bf16"] - r["us"][m]["nvfp4"]) for r in rs)
            d_f8 = sum(r["calls"] * (r["us"][m]["bf16"] - r["us"][m]["fp8"]) for r in rs)
            print(f"{MARK} {model} M={m}: est. step saving vs bf16 nvfp4 "
                  f"{d_nv / 1e3:.2f} ms, fp8 {d_f8 / 1e3:.2f} ms", flush=True)
    if args.json:
        print(json.dumps({"pass": ok, "results": rows}), flush=True)
    print(f"{MARK} NVFP4-DENSE ORACLE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
