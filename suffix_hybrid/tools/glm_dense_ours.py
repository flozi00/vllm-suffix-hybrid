# SPDX-License-Identifier: Apache-2.0
"""Boot gate glm_dense_ours_bench: our own small-M dense GEMM (sm120/fp8_dense_patch/ours.py)
against every existing option at the GLM 5.3 TP8 per-rank dense shapes on ONE GPU, cold L2
(flushed before every replay), CUDA-graph timed (median), through the same tuner the serving
backends (SUFFIX_FP8_DENSE_BACKEND=ours|best|ours_fp4) run at load.

Lines "[suffix dense-ours]":
  per (shape, M): us of bf16 (cuBLAS) | cutlass (W8A8 + its per-token act-quant kernel) |
    marlin8 (W8A16) | marlin4 (W4A16) | ours8 (W8A16, best gemv/mma tile) | ours8a8 (W8A8,
    activation quantized inside the GEMM) | ours4 (W4A16), the winning tile of each of
    ours, and rel-l2 vs the fp32 BF16-weight product (quantization + kernel error);
  per shape: the route tables backends ours / best / ours_fp4 would serve;
  per M: per-step ms per group (attention projections, shared expert, dense MLP, MTP
    eh_proj) for every column and every route;
  batched heads (absorbed MLA W_UK_T / W_UV, 8 heads per rank): torch.bmm BF16 vs ours
    batched FP8 / NVFP4.
Exit 1 if one of our candidates misses the exact-dequant bound (ours.OURS_BOUND: it is
excluded from routing either way) or a column misses its quantization bound.

    python -m suffix_hybrid.tools.glm_dense_ours
"""
from __future__ import annotations

import sys

MARK = "[suffix dense-ours]"
# (name, N, K, calls per step at that M, group) per TP8 rank (glm_dense_marlin.SHAPES + the
# dense MLP of layers 0-2); MTP eh_proj once per drafter pass.
SHAPES = (
    ("qkv_a", 2624, 6144, 78, "attn"),
    ("q_b", 2048, 2048, 78, "attn"),
    ("o_proj", 6144, 2048, 78, "attn"),
    ("idx.wq_b", 4096, 2048, 21, "attn"),
    ("shared.gate_up", 512, 6144, 75, "shared"),
    ("shared.down", 6144, 256, 75, "shared"),
    ("dense.gate_up", 3072, 6144, 3, "dense"),
    ("dense.down", 6144, 1536, 3, "dense"),
    ("mtp.eh_proj", 6144, 12288, 1, "mtp"),
)
# absorbed MLA per rank (8 heads): q_nope [T, 8, 192] x W_UK_T -> [T, 8, 512];
# attention out [T, 8, 512] x W_UV -> [T, 8, 256]; both on all 78 layers.
HEADS = (("W_UK_T", 8, 512, 192, 78), ("W_UV", 8, 256, 512, 78))
MS = (1, 2, 4, 6, 8, 16, 32, 48, 96, 192)
HEAD_MS = (1, 6, 8, 32, 96, 192)
COLS = ("bf16", "cutlass", "marlin8", "marlin4", "ours8", "ours8a8", "ours4")
ROUTES = {"ours": ("cutlass",), "best": ("cutlass", "marlin8"), "ours_fp4": ()}


def best_of(row: dict, a8: bool):
    hit = [(us, p) for us, p in row["ours"] if p.cfg.a8 == a8]
    return hit[0] if hit else (None, None)


def columns(t8: dict, t4: dict, m: int) -> dict:
    """{column: (us, plan | None)} at M = m from the two tuning tables."""
    cols = {c: (t8[m]["base"][c], None) for c in ("bf16", "cutlass", "marlin8")
            if c in t8[m]["base"]}
    if m in t4 and "marlin4" in t4[m]["base"]:
        cols["marlin4"] = (t4[m]["base"]["marlin4"], None)
    for c, row, a8 in (("ours8", t8[m], False), ("ours8a8", t8[m], True),
                       ("ours4", t4.get(m), False)):
        if row is not None and best_of(row, a8)[0] is not None:
            cols[c] = best_of(row, a8)
    return cols


def route_us(table: dict, m: int, choice) -> float:
    path, plan = choice
    if path == "ours":
        return next(us for us, p in table[m]["ours"] if p == plan)
    return table[m]["base"][path]


def main() -> int:
    import torch
    from torch import nn

    import vllm._custom_ops as ops
    from fp8_dense_patch import ours as O
    from fp8_dense_patch import runtime as R
    from suffix_hybrid.kernels.nvfp4_gemm import _graph_us
    from suffix_hybrid.tools.glm_dense_marlin import _module

    R._OPS = ops
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    flush_buf = torch.empty(max(2 * props.L2_cache_size, 64 << 20), dtype=torch.uint8, device=dev)
    sem = torch.zeros(O.SEM_SLOTS, dtype=torch.int32, device=dev)
    nsm = props.multi_processor_count

    def log(line):
        print(f"{MARK} {line}", flush=True)

    kw = dict(dev=dev, flush=flush_buf.zero_, sem=sem, nsm=nsm, log=log)
    src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    bw = 2 * src.numel() / _graph_us(lambda st, a=src, b=dst: b.copy_(a), dev, 5) / 1e3  # GB/s
    del src, dst
    log(f"{props.name} cc {torch.cuda.get_device_capability(dev)}, {nsm} SMs, L2 "
        f"{props.L2_cache_size >> 20} MiB (flushed with {flush_buf.numel() >> 20} MiB before "
        f"every replay), median of {O.REPS} graph replays; DRAM copy {bw:.0f} GB/s (the "
        f"'%bw' reference); tiles: {O.MMA_TILES} gemv {O.GEMV_TILES}")
    bad, sums = [], {}

    def quant_err(fn, x, ref):
        y = fn(x)
        torch.cuda.synchronize(dev)
        return float((y.float() - ref).norm() / ref.norm())

    for name, n, k, count, group in SHAPES:
        gen = torch.Generator(device=dev).manual_seed(n * 7 + k)
        w = (torch.randn(n, k, device=dev, generator=gen, dtype=torch.float32) * 0.02).bfloat16()
        q, s = R.quantize_weight(w)
        wt8 = O.Weights.fp8(q, s)
        extra8 = {"bf16": lambda x, w=w: nn.functional.linear(x, w),
                  "cutlass": lambda x, q=q, s=s: R.fp8_linear(x, q, s)}
        extra4, wt4 = {}, None
        try:
            m8 = _module(n, k, R.marlin_fp8_pack(q, s))
            extra8["marlin8"] = lambda x, m8=m8: R.marlin_linear(x, m8)
        except Exception as exc:  # noqa: BLE001 - report, keep benching the rest
            log(f"{name}: marlin8 unavailable: {type(exc).__name__}: {exc}"[:200])
        try:
            wq, wsf, gs = R.fp4_quantize(w)
            wt4 = O.Weights.fp4(wq, wsf, gs)
            m4 = _module(n, k, R.marlin_fp4_pack(wq, wsf, gs))
            extra4["marlin4"] = lambda x, m4=m4: R.marlin_linear(x, m4)
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: fp4 / marlin4 unavailable: {type(exc).__name__}: {exc}"[:200])
        t8, b8 = O.tune_shape(wt8, MS, extra=extra8, **kw)
        t4, b4 = O.tune_shape(wt4, MS, extra=extra4, **kw) if wt4 is not None else ({}, [])
        bad += b8 + b4
        dec = {"ours": O.decide(t8, ROUTES["ours"]), "best": O.decide(t8, ROUTES["best"])}
        if t4:
            dec["ours_fp4"] = O.decide(t4, ())
        for m in MS:
            cols = columns(t8, t4, m)
            x = torch.randn(m, k, device=dev, generator=gen, dtype=torch.float32).bfloat16()
            with O._exact_fp32():
                ref = x.float() @ w.float().t()
            fns = {c: extra8[c] for c in ("bf16", "cutlass", "marlin8") if c in extra8}
            fns.update({c: extra4[c] for c in ("marlin4",) if c in extra4})
            for c in ("ours8", "ours8a8", "ours4"):
                if c in cols:
                    wt = wt4 if c == "ours4" else wt8
                    fns[c] = lambda x, wt=wt, p=cols[c][1]: O.run(x, wt, p, sem)
            rel = {c: quant_err(fn, x, ref) for c, fn in fns.items() if c in cols}
            bad += [f"{name} M={m}: no working {c} plan" for c in ("ours8", "ours4")
                    if c not in cols and (c == "ours8" or wt4 is not None)]
            for c, r in rel.items():
                bound = R.FP4_QUANT_BOUND if c in ("marlin4", "ours4") else R.QUANT_BOUND
                if not r <= bound:
                    bad.append(f"{name} M={m} {c}: rel vs fp32 {r:.3e} > {bound}")
            cell = " | ".join(f"{c} {cols[c][0]:6.1f}" if c in cols else f"{c}    n/a"
                              for c in COLS)
            tiles = " ".join(f"{c}={cols[c][1].tag()}" for c in ("ours8", "ours8a8", "ours4")
                             if c in cols)
            rl = " ".join(f"{c} {rel[c]:.2e}" for c in COLS if c in rel)
            served = " ".join(f"{r}={route_us(t4 if r == 'ours_fp4' else t8, m, d[m]):.1f}"
                              for r, d in dec.items())
            io = m * k * 2 + m * n * 2
            pct = " ".join(f"{c} {(nb + io) / cols[c][0] / 1e3 / bw:.0%}" for c, nb in
                           (("ours8", n * k + 4 * n), ("ours4", n * k // 2 + n * k // 16))
                           if c in cols)
            log(f"{name:14s} N={n:5d} K={k:5d} M={m:3d}: {cell} us | routes {served} | "
                f"%bw {pct} | {tiles} | rel-l2 vs fp32 {rl}")
            acc = sums.setdefault((group, m), {})
            for c, (us, _p) in cols.items():
                acc[c] = acc.get(c, 0.0) + count * us / 1000.0
            for r, d in dec.items():
                us = route_us(t4 if r == "ours_fp4" else t8, m, d[m])
                acc[f"route:{r}"] = acc.get(f"route:{r}", 0.0) + count * us / 1000.0
        for r, d in dec.items():
            route = O.Route(MS, [d[m] for m in MS], ("cutlass", None) if r != "ours_fp4"
                            else d[MS[-1]])
            log(f"route {name} {r}: {route.line(t4 if r == 'ours_fp4' else t8)}")

    for name, b, n, k, count in HEADS:  # batched heads, vLLM's [T, H, K] -> [H, T, K] view
        gen = torch.Generator(device=dev).manual_seed(b * n + k)
        w = (torch.randn(b, n, k, device=dev, generator=gen, dtype=torch.float32) * 0.02).bfloat16()
        q, s = R.quantize_weight(w.reshape(b * n, k))
        wt8 = O.Weights.fp8(q.view(b, n, k), s.view(b, n))
        wt4 = None
        try:
            per = [R.fp4_quantize(w[h].contiguous()) for h in range(b)]
            wt4 = O.Weights.fp4(torch.stack([p[0] for p in per]), torch.stack([p[1] for p in per]),
                                torch.cat([p[2].reshape(1) for p in per]))
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: fp4 unavailable: {type(exc).__name__}: {exc}"[:200])
        wtT = w.transpose(1, 2)
        extra = {"bmm": lambda x, wtT=wtT: torch.bmm(x, wtT)}
        t8, b8 = O.tune_shape(wt8, HEAD_MS, extra=extra, **kw)
        t4, b4 = O.tune_shape(wt4, HEAD_MS, **kw) if wt4 is not None else ({}, [])
        bad += b8 + b4
        for m in HEAD_MS:
            x = torch.randn(m, b, k, device=dev, generator=gen,
                            dtype=torch.float32).bfloat16().transpose(0, 1)
            with O._exact_fp32():
                ref = x.float() @ w.float().transpose(1, 2)
            cols = {"bmm": (t8[m]["base"]["bmm"], None)} if "bmm" in t8[m]["base"] else {}
            for c, row, a8 in (("ours8", t8[m], False), ("ours8a8", t8[m], True),
                               ("ours4", t4.get(m), False)):
                if row is not None and best_of(row, a8)[0] is not None:
                    cols[c] = best_of(row, a8)
            rel = {"bmm": quant_err(extra["bmm"], x, ref)} if "bmm" in cols else {}
            for c in ("ours8", "ours8a8", "ours4"):
                if c in cols:
                    wt = wt4 if c == "ours4" else wt8
                    rel[c] = quant_err(lambda x, wt=wt, p=cols[c][1]: O.run(x, wt, p, sem), x, ref)
                    bound = R.FP4_QUANT_BOUND if c == "ours4" else R.QUANT_BOUND
                    if not rel[c] <= bound:
                        bad.append(f"{name} M={m} {c}: rel vs fp32 {rel[c]:.3e} > {bound}")
            cell = " | ".join(f"{c} {cols[c][0]:6.1f}" if c in cols else f"{c}    n/a"
                              for c in ("bmm", "ours8", "ours8a8", "ours4"))
            tiles = " ".join(f"{c}={cols[c][1].tag()}" for c in ("ours8", "ours8a8", "ours4")
                             if c in cols)
            rl = " ".join(f"{c} {v:.2e}" for c, v in rel.items())
            log(f"heads {name:7s} {b}x(N={n} K={k}) M={m:3d}: {cell} us | {tiles} | "
                f"rel-l2 vs fp32 {rl} | per step x{count}: "
                + " ".join(f"{c} {count * v[0] / 1000:.2f}ms" for c, v in cols.items()))

    # Ceiling of a qkv_a + q_a/kv_a-layernorm fusion: the two RMSNorm launches it would remove.
    nq = (torch.randn(2048, device=dev, dtype=torch.float32) * 0.1 + 1).bfloat16()
    nkv = (torch.randn(512, device=dev, dtype=torch.float32) * 0.1 + 1).bfloat16()
    for m in MS:
        qa, kva = (torch.randn(m, 2048, device=dev, dtype=torch.bfloat16),
                   torch.randn(m, 512, device=dev, dtype=torch.bfloat16))
        oq, okv = torch.empty_like(qa), torch.empty_like(kva)
        try:
            us = _graph_us(lambda st, qa=qa, kva=kva, oq=oq, okv=okv: (
                ops.rms_norm(oq, qa, nq, 1e-6), ops.rms_norm(okv, kva, nkv, 1e-6)),
                dev, O.REPS, flush_buf.zero_)
        except Exception as exc:  # noqa: BLE001
            log(f"fusion ceiling: rms_norm unavailable: {type(exc).__name__}: {exc}"[:200])
            break
        log(f"fusion ceiling M={m:3d}: q_a + kv_a rms_norm {us:5.1f} us -> x78 "
            f"{78 * us / 1000:.2f} ms/step (an exact qkv_a epilogue fusion saves at most this)")

    for m in MS:
        for group in ("attn", "shared", "dense", "mtp"):
            acc = sums.get((group, m), {})
            log(f"per-step ms {group:6s} M={m:3d}: "
                + " | ".join(f"{c} {acc[c]:6.2f}" for c in COLS if c in acc) + " || "
                + " | ".join(f"{r} {acc[f'route:{r}']:6.2f}" for r in ROUTES
                             if f"route:{r}" in acc))
    for line in bad:
        log(f"FAIL {line}")
    log(f"verdict: {'FAIL' if bad else 'PASS'} ({len(bad)} violations)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
