# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 TP8 MoE layer per rank WITH its shared expert, one GPU, cold L2.

Boot gate glm_moe_shared_bench. vLLM's path today: our NVFP4 routed kernel, then the BF16
shared MLP (gate_up GEMM [6144 -> 512], silu_and_mul, down GEMM [256 -> 6144]: cuBLAS) and
the runner's `shared + routed` add — serial, and with the shared MLP on a second stream the
way vLLM's SharedExperts overlaps it (M <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD).
Fused: the shared MLP as our kernel's extra slot (SUFFIX_NVFP4_MOE_FUSE_SHARED, NVFP4 W4A4
from the same BF16 weights). Both sides take their best tune of glm_moe_tp's TP grid. Per M
one line, then numerics at M=8 (fused vs the f64 spec, vs the stock output, the shared slot
vs its BF16 MLP), the fused tune table + the largest M where fused wins (paste into
SUFFIX_NVFP4_MOE_TUNE / _MAX_M), and the per-step saving over 75 MoE layers.
Lines: [suffix moe-shared]. Evidence only (exit 0 unless it crashed).

    python -m suffix_hybrid.tools.glm_moe_shared
"""
from __future__ import annotations

import math
import sys

MARK = "[suffix moe-shared]"
MS = (1, 6, 8, 48, 96, 192)
LAYERS = 75  # GLM-5.3 MoE layers per decode step (+ the MTP layer per draft)


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def main() -> int:
    import torch

    from suffix_hybrid.kernels import nvfp4_moe as nm
    from suffix_hybrid.tools.glm_moe_tp import TP_GRID

    native = nm._native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) or (128 << 20)
    buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
    flush = lambda: buf.sum()  # noqa: E731 - a cold L2 per replay, like serving
    us = lambda fn: nm._graph_us(fn, dev, 20, flush)  # noqa: E731
    hdim, idim, topk = 6144, 256, 8
    p = nm._dev_problem(dev, nm.GLM_TP8, seed=1)
    gen = torch.Generator().manual_seed(7)
    w_gu = (torch.randn(2 * idim, hdim, generator=gen) * 0.05).bfloat16().to(dev)  # [gate; up]
    w_dn = (torch.randn(hdim, idim, generator=gen) * 0.05).bfloat16().to(dev)
    xp = (torch.randn(256, hdim, generator=gen) * (2688.0 / p["a1g"] / 4.5)).to(dev).bfloat16().float()
    gu = xp @ w_gu.float().T
    h_sh = float((nm.act_ref(gu[:, :idim], p["act"]) * gu[:, idim:]).abs().max())
    c = nm.shared_c(nm.probe_h_routed(p, xp), h_sh, nm.shared_headroom())
    pf = dict(p, shared=nm.quant_shared(w_gu, w_dn, p["a1g"], p["a2g"], c))
    ws = nm.workspace(dev, max(MS), topk, hdim, idim, 256, ns=1)
    h_buf = torch.empty(max(MS), idim, dtype=torch.bfloat16, device=dev)
    from vllm.platforms import current_platform

    current_platform.import_kernels()  # registers torch.ops._C (a gate runs before vLLM loads)
    silu_and_mul = torch.ops._C.silu_and_mul  # vLLM's SiluAndMul kernel

    def shared_bf16(x):  # DeepseekV2MLP at TP8 (reduce_results=False)
        h = h_buf[:x.shape[0]]
        silu_and_mul(h, torch.mm(x, w_gu.t()))
        return torch.mm(h, w_dn.t())

    aux = torch.cuda.Stream(dev)
    fork, join = torch.cuda.Event(), torch.cuda.Event()
    name = lambda cand: f"{nm.PLANS[cand[0]]}/{cand[1]}/{cand[2]}/{cand[3]}"  # noqa: E731
    _say(f"TP8 rank: 256 routed experts x I={idim} + 1 shared (BF16 -> NVFP4 slot, c=2^"
         f"{int(math.log2(c))}); cold L2 (flush {2 * l2 >> 20} MB per replay); "
         f"{len(TP_GRID)} tunes per side")
    rows, saves = [], []
    for m in MS:
        ids, tw = nm.rand_routing(m, 256, topk, dev, seed=100 + m, dead=False)
        x = torch.randn(m, hdim, device=dev).bfloat16()
        out = torch.empty(m, hdim, dtype=torch.bfloat16, device=dev)
        run = lambda q, s, cand: nm.run_ours(native, q, x, ids, tw, ws, s.cuda_stream, out,  # noqa: E731
                                             cand[0], nm.tune_word(*cand[1:]))
        routed = {cand: us(lambda s, cand=cand: run(p, s, cand)) for cand in TP_GRID}
        fused = {cand: us(lambda s, cand=cand: run(pf, s, cand)) for cand in TP_GRID}
        br, bf = min(routed, key=routed.get), min(fused, key=fused.get)

        def overlap(s):
            fork.record(s)
            with torch.cuda.stream(aux):
                aux.wait_event(fork)
                sh = shared_bf16(x)
                join.record(aux)
            r = run(p, s, br)
            s.wait_event(join)
            return sh + r

        t = {"shared": us(lambda s: shared_bf16(x)),
             "serial": us(lambda s: shared_bf16(x) + run(p, s, br)),
             "overlap": us(overlap)}
        stock = min(t["serial"], t["overlap"])
        rows.append((m, bf, fused[bf], stock))
        saves.append((m, stock - fused[bf]))
        _say(f"M={m:3d} routed {routed[br]:7.1f} us ({name(br)}) + bf16 shared {t['shared']:6.1f}: "
             f"serial {t['serial']:7.1f}, overlap {t['overlap']:7.1f} | fused {fused[bf]:7.1f} "
             f"({name(bf)}) | saves {stock - fused[bf]:6.1f} us/layer = "
             f"{(stock - fused[bf]) * LAYERS / 1000:5.2f} ms/step ({LAYERS} layers)")
    m = 8
    ids, tw = nm.rand_routing(m, 256, topk, dev, seed=7, dead=False)
    x = torch.randn(m, hdim, device=dev).bfloat16()
    stream = torch.cuda.current_stream(dev).cuda_stream
    ours = nm.run_ours(native, pf, x, ids, tw, ws, stream).float()
    ref = nm.moe_ref(pf, x, nm.to_local(ids, 0, 256), tw)
    stock = (shared_bf16(x) + nm.run_ours(native, p, x, ids, tw, ws, stream)).float()
    sh_rel = nm._rel(nm.shared_ref(pf, x), shared_bf16(x).double())
    _say(f"M=8 numerics: fused rel-l2 vs f64 spec {nm._rel(ours, ref):.2e} (tol {nm.REF_TOL:g}), "
         f"vs stock routed+bf16 shared {nm._rel(ours, stock):.2e}, shared slot vs its bf16 MLP "
         f"{sh_rel:.2e} (tol {nm.SHARED_TOL:g})")
    table, mx, lose = nm.sweep_verdict(rows)
    _say(f"fused vs stock: SUFFIX_NVFP4_MOE_TUNE={table} fused wins up to M="
         f"{mx if mx else '(never)'}" + (f" (loses below it at M={lose})" if lose else ""))
    _say("per-step saving: " + ", ".join(f"M={m} {d * LAYERS / 1000:.2f} ms" for m, d in saves))
    return 0


if __name__ == "__main__":
    sys.exit(main())
