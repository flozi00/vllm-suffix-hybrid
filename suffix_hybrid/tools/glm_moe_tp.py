# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 routed MoE per rank: EP8 (the busiest rank) vs TP8 (no EP), one GPU, cold L2.

Boot gate glm_moe_tp_bench. Prod runs TP8 + EP8: every rank holds 32 whole experts and a
decode step waits for the rank whose experts were hit most (c1: ~8.6 vs ~5.5 on average;
the host all-reduce then idles on the others). TP8 without EP gives every rank 1/8 of all
256 experts (I 2048 -> 256): the same bytes on average, no imbalance. Per M (decode tokens
per layer call, MTP k=5 -> 6 per sequence) and backend this prints one line:

  EP-worst  random top-8 routing rotated so this rank (5) gets the busiest rank's experts:
            vLLM's FLASHINFER_CUTLASS call and our NVFP4 MoE kernel (prod tune table);
  TP8       FLASHINFER_CUTLASS, ours (best of a small tune grid, bit-identical plans) and
            FlashInfer's b12x (CuteDSL; flashinfer_b12x) when it compiles and loads here.

Then numerics at M=8 (rel-l2 vs the f64 reference: FlashInfer, ours, b12x), the TP tune
table + MAX_M to paste (SUFFIX_NVFP4_MOE_TUNE / _MAX_M) and per M the faster layout.
Lines: [suffix moe-tp]. Evidence only (exit 0 unless it crashed).

    python -m suffix_hybrid.tools.glm_moe_tp
"""
from __future__ import annotations

import sys

MARK = "[suffix moe-tp]"
MS = (1, 6, 8, 48, 96, 192)
EP_RANK, EP, LOCAL = 5, 8, 32
# prod glm's SUFFIX_NVFP4_MOE_TUNE (2026-09-29 EP sweep)
PROD_EP_TUNE = "1:fused3/32/255/1,5:front4/8/255/1,10:front4/8/0/1,32:front4/8/255/1,64:front4/8/255/0"
TP_GRID = [(0, 0, 0, 0)] + [(md, a, b, c) for md in (1, 2) for a in (8, 32, 255)
                            for b in (0, 255) for c in (0, 1)]


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def worst_rank_ids(ids, ep: int = EP, local: int = LOCAL, rank: int = EP_RANK):
    """Rotate global expert ids so the EP rank with the most distinct routed experts lands on
    `rank`; returns (ids, distinct experts per original rank)."""
    import torch

    counts = [int(torch.unique(ids[(ids // local) == r]).numel()) for r in range(ep)]
    w = max(range(ep), key=counts.__getitem__)
    return (ids + (rank - w) * local) % (ep * local), counts


def _b12x(p, dev, max_m):
    """FlashInfer's b12x on our FI-CUTLASS-layout problem, prepared like vLLM's
    FlashInferB12xExperts.process_weights_after_loading (weight global scale baked into the
    block scales, alphas 1, dynamic FC2 input quant). Returns run(x, ids, tw) or a reason."""
    import torch

    try:
        from flashinfer.fused_moe import B12xMoEWrapper
        from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"[:200]
    e, two_i, hh = p["w13"].shape

    def mma(sf, scale2):
        sf = (sf.float() * scale2.view(-1, 1, 1)).to(torch.float8_e4m3fn)
        ne, m1, k1_sf = sf.shape
        return flashinfer_convert_sf_to_mma_layout(sf.reshape(ne * m1, k1_sf), m=m1, k=k1_sf * 16,
                                                   num_groups=ne)

    try:
        # g = 1 / (w_gs * a_gs) -> the checkpoint's weight_scale_2 = 1 / w_gs = g * a_gs
        w1_sf, w2_sf = mma(p["w13_sf"], p["g1"] * p["a1g"]), mma(p["w2_sf"], p["g2"] * p["a2g"])
        ones = torch.ones(e, dtype=torch.float32, device=dev)
        wr = B12xMoEWrapper(num_experts=e, top_k=8, hidden_size=hh * 2, intermediate_size=two_i // 2,
                            use_cuda_graph=True, max_num_tokens=max_m, activation="silu")
    except Exception as exc:  # noqa: BLE001
        return f"setup failed: {exc!r}"[:300]

    def run(x, ids, tw):
        return wr.run(x=x, w1_weight=p["w13"], w1_weight_sf=w1_sf, w1_alpha=ones,
                      fc2_input_scale=ones, w2_weight=p["w2"], w2_weight_sf=w2_sf, w2_alpha=ones,
                      token_selected_experts=ids.to(torch.int32), token_final_scales=tw)
    return run


def main() -> int:
    import torch

    from suffix_hybrid.kernels import nvfp4_moe as nm

    native = nm._native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = lambda s: s.cuda_stream  # noqa: E731
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) or (128 << 20)
    buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
    flush = lambda: buf.sum()  # noqa: E731 - a cold L2 per replay, like serving
    us = lambda fn: nm._graph_us(fn, dev, 20, flush)  # noqa: E731
    ep = nm._dev_problem(dev, nm.GLM_EP, seed=1)
    tp = nm._dev_problem(dev, nm.GLM_TP8, seed=1)
    ws_ep = nm.workspace(dev, max(MS), 8, 6144, 2048, LOCAL)
    ws_tp = nm.workspace(dev, max(MS), 8, 6144, 256, 256)
    prod = nm.parse_tune(PROD_EP_TUNE)
    b12x = _b12x(tp, dev, max(MS))
    _say(f"EP8 rank view: {LOCAL} experts x I=2048; TP8 view: 256 experts x I=256; cold L2 "
         f"(flush {2 * l2 >> 20} MB per replay); b12x: {'ready' if callable(b12x) else b12x}")
    rows, verdicts = [], []
    for m in MS:
        ids, tw = nm.rand_routing(m, 256, 8, dev, seed=100 + m, dead=False)
        ids_w, counts = worst_rank_ids(ids)
        x = torch.randn(m, 6144, device=dev).bfloat16()
        out = torch.empty(m, 6144, dtype=torch.bfloat16, device=dev)
        mode, tune = nm.tune_for(m, prod)
        t = {"ep_fi": us(lambda s: nm._fi_call(ep, x, ids_w, tw, out)),
             "ep_ours": us(lambda s: nm.run_ours(native, ep, x, ids_w, tw, ws_ep, stream(s), out,
                                                 mode, tune)),
             "tp_fi": us(lambda s: nm._fi_call(tp, x, ids, tw, out))}
        grid = {c: us(lambda s, c=c: nm.run_ours(native, tp, x, ids, tw, ws_tp, stream(s), out, c[0],
                                                  nm.tune_word(*c[1:])))
                for c in TP_GRID}
        best = min(grid, key=grid.get)
        t["tp_ours"] = grid[best]
        b12x_err = ""
        if callable(b12x):
            try:
                t["tp_b12x"] = us(lambda s: b12x(x, ids, tw))
            except Exception as exc:  # noqa: BLE001 - e.g. a cubin this driver cannot load
                b12x_err, b12x = f" b12x FAILED: {exc!r}"[:300], f"failed at M={m}: {exc!r}"[:300]
        rows.append((m, best, t["tp_ours"], t["tp_fi"]))
        ep_best = min(t["ep_fi"], t["ep_ours"])
        tp_best = min(v for k, v in t.items() if k.startswith("tp_"))
        verdicts.append((m, ep_best, tp_best))
        _say(f"M={m:3d} pairs={m * 8:4d} EP experts/rank max {max(counts)} avg {sum(counts) / EP:.1f} | "
             f"EP-worst flashinfer {t['ep_fi']:7.1f} us, ours {t['ep_ours']:7.1f} | TP8 flashinfer "
             f"{t['tp_fi']:7.1f}, ours {t['tp_ours']:7.1f} ({nm.PLANS[best[0]]}/{best[1]}/{best[2]}/"
             f"{best[3]}), b12x " + (f"{t['tp_b12x']:7.1f}" if "tp_b12x" in t else "     -")
             + f" | best EP {ep_best:7.1f} vs TP {tp_best:7.1f} us (x{ep_best / tp_best:4.2f})" + b12x_err)
    m = 8  # numerics on the TP problem: everyone vs the f64 spec reference
    ids, tw = nm.rand_routing(m, 256, 8, dev, seed=7, dead=False)
    x = torch.randn(m, 6144, device=dev).bfloat16()
    ref = nm.moe_ref(tp, x, nm.to_local(ids, 0, 256), tw)
    errs = {"flashinfer": nm._rel(nm._fi_call(tp, x, ids, tw, torch.empty_like(x)).float(), ref),
            "ours": nm._rel(nm.run_ours(native, tp, x, ids, tw, ws_tp,
                                        torch.cuda.current_stream(dev).cuda_stream).float(), ref)}
    if callable(b12x):
        try:
            errs["b12x"] = nm._rel(b12x(x, ids, tw).float(), ref)
        except Exception as exc:  # noqa: BLE001
            errs["b12x"] = float("nan")
            _say(f"b12x numerics FAILED: {exc!r}"[:300])
    _say("TP8 M=8 rel-l2 vs f64 reference: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
    table, mx, lose = nm.sweep_verdict(rows)
    _say(f"TP8 ours vs flashinfer: SUFFIX_NVFP4_MOE_TUNE={table} SUFFIX_NVFP4_MOE_MAX_M="
         f"{mx if mx else '(ours never wins)'}" + (f" (loses below it at M={lose})" if lose else ""))
    _say("faster layout per M: " + ", ".join(f"M={m} {'TP' if tp_ < ep_ else 'EP'} x{ep_ / tp_:.2f}"
                                             for m, ep_, tp_ in verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
