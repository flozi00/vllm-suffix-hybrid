# SPDX-License-Identifier: Apache-2.0
"""Small-M FP8 128x128-block routed experts (decode at M <= SUFFIX_FP8_MOE_SMALL_M,
default off): two Triton launches with enough CTAs to stream the selected experts'
weights at DRAM speed when only a handful of tokens decode (c1-c4).

The oxide fp8_moe fused plan runs fc1 on (I/128) x slots CTAs (48 at the LFM2.5
shape, M=1: 4 experts) and reaches ~75% of DRAM bandwidth there (serving profile
2026-10-15: fc1 18.6 + fc2 9.9 + route/quant 3.0 us per layer). Here:

  A  fc1   grid (I/BN1, P = M*topk): one pair x BN1 intermediate columns per CTA
           (192 CTAs at M=1, BN1=32). Quantizes its token's x per 128-col group
           in-kernel (vLLM per_token_group_quant_fp8: s = max(amax, 1e-10) / 448
           with IEEE division, q = e4m3_rne(clamp(x / s))), accumulates gate and
           up per 128-K block as (dot * s_x) * s_w in fp32, writes
           h = silu(gate) * up in fp32.
  B  fc2   grid (H/BN2, M): BN2 output columns of one token over its topk
           experts. Re-quantizes each pair's h per 128-col group in the prologue
           (vLLM silu_and_mul_per_block_quant: s = max(amax / 448, 1/(448*512)),
           no eps; UE8M0: per_token_group rule rounded up to 2^n), accumulates
           (dot * s_h) * s_w, y_k = topk_w * acc, out = sum_k y_k in ascending k
           (fp32) -> bf16.

No routing kernel and no workspace beyond h (P x I fp32). Same numerical
contract as fp8_moe.moe_ref (f64 spec); the load-time oracle demands
rel <= fp8_moe.REF_TOL vs the spec and agreement with the oxide kernel.
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_FP8_MOE_SMALL_M"
MARKER = "[suffix fp8-moe-small]"
BN1 = 32
BN2 = 8
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)

_kernels = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def small_m() -> int:
    v = int(os.environ.get(GATE, "0") or 0)
    if not 0 <= v <= 16:
        raise ValueError(f"{GATE}={v} outside [0, 16]")
    return v


def _get():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _quant(v, eps, smin, UE8M0: tl.constexpr):
        """f32 [128] -> (e4m3 value as f32, f32 scale) with vLLM's rounding."""
        amax = tl.maximum(tl.max(tl.abs(v), axis=0), eps)
        s = tl.maximum(tl.math.div_rn(amax, 448.0), smin)
        if UE8M0:
            s = tl.exp2(tl.ceil(tl.log2(tl.maximum(s, 1e-10))))
        q = tl.minimum(tl.maximum(tl.math.div_rn(v, s), -448.0), 448.0)
        return q.to(tl.float8e4nv).to(tl.float32), s

    @triton.jit
    def _fc1(x_ptr, s_x, ids_ptr, w13_ptr, w13s_ptr, h_ptr, id_base,
             TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
             BN: tl.constexpr, UE8M0: tl.constexpr):
        pid_n = tl.program_id(0)
        p = tl.program_id(1)
        t = p // TOPK
        e = tl.load(ids_ptr + p).to(tl.int64) - id_base
        rn = pid_n * BN + tl.arange(0, BN)
        if (e < 0) | (e >= E):
            tl.store(h_ptr + p * I + rn, tl.zeros([BN], dtype=tl.float32))
            return
        kb_n: tl.constexpr = H // 128
        wbase = w13_ptr + e * (2 * I * H)
        sbase = w13s_ptr + e * ((2 * I // 128) * kb_n)
        sg_row = (pid_n * BN) // 128
        su_row = (I + pid_n * BN) // 128
        rk0 = tl.arange(0, 128)
        acc_g = tl.zeros([BN], dtype=tl.float32)
        acc_u = tl.zeros([BN], dtype=tl.float32)
        for kb in range(kb_n):
            rk = kb * 128 + rk0
            xv = tl.load(x_ptr + t * s_x + rk).to(tl.float32)
            q, sx = _quant(xv, 1e-10, 0.0, UE8M0)
            wg = tl.load(wbase + rn[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (I + rn)[:, None] * H + rk[None, :]).to(tl.float32)
            pg = tl.sum(wg * q[None, :], axis=1)
            pu = tl.sum(wu * q[None, :], axis=1)
            acc_g += (pg * sx) * tl.load(sbase + sg_row * kb_n + kb)
            acc_u += (pu * sx) * tl.load(sbase + su_row * kb_n + kb)
        h = tl.math.div_rn(acc_g, 1.0 + tl.exp(-acc_g)) * acc_u  # silu(g) * u, IEEE div like CUDA
        tl.store(h_ptr + p * I + rn, h)

    @triton.jit
    def _fc2(h_ptr, ids_ptr, tw_ptr, w2_ptr, w2s_ptr, out_ptr, s_out, id_base,
             TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
             BN: tl.constexpr, UE8M0: tl.constexpr):
        pid_n = tl.program_id(0)
        t = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        kb_n: tl.constexpr = I // 128
        s_row = (pid_n * BN) // 128
        rk0 = tl.arange(0, 128)
        tot = tl.zeros([BN], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            p = t * TOPK + k
            e = tl.load(ids_ptr + p).to(tl.int64) - id_base
            if (e >= 0) & (e < E):
                wk = tl.load(tw_ptr + p)
                wbase = w2_ptr + e * (H * I)
                sbase = w2s_ptr + e * ((H // 128) * kb_n)
                acc = tl.zeros([BN], dtype=tl.float32)
                for kb in range(kb_n):
                    rk = kb * 128 + rk0
                    hv = tl.load(h_ptr + p * I + rk)
                    if UE8M0:
                        q, sh = _quant(hv, 1e-10, 0.0, True)
                    else:
                        q, sh = _quant(hv, 0.0, 1.0 / (448.0 * 512.0), False)
                    w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
                    acc += (tl.sum(w * q[None, :], axis=1) * sh) * tl.load(sbase + s_row * kb_n + kb)
                tot += wk * acc
        tl.store(out_ptr + t * s_out + rn, tot.to(tl.bfloat16))

    _kernels = (_fc1, _fc2)
    return _kernels


def run(cfg, x, topk_weights, topk_ids, out=None, h=None):
    """cfg = the fp8_moe layer cfg (w13/w13_s/w2/w2_s/ue8m0/H/I/E/topk/id_base)."""
    import torch
    m = x.shape[0]
    k, hdim, idim, e = cfg["topk"], cfg["H"], cfg["I"], cfg["E"]
    fc1, fc2 = _get()
    ids = topk_ids.reshape(-1)
    tw = topk_weights.reshape(-1)
    tw = tw if tw.dtype == torch.float32 else tw.float()
    if h is None:
        h = torch.empty(m * k, idim, dtype=torch.float32, device=x.device)
    if out is None:
        out = torch.empty(m, hdim, dtype=torch.bfloat16, device=x.device)
    ue = bool(cfg["ue8m0"])
    base = int(cfg.get("id_base", 0))
    fc1[(idim // BN1, m * k)](x, x.stride(0), ids, cfg["w13"], cfg["w13_s"], h, base,
                              TOPK=k, H=hdim, I=idim, E=e, BN=BN1, UE8M0=ue, num_warps=4)
    fc2[(hdim // BN2, m)](h, ids, tw, cfg["w2"], cfg["w2_s"], out, out.stride(0), base,
                          TOPK=k, H=hdim, I=idim, E=e, BN=BN2, UE8M0=ue, num_warps=2)
    return out


def oracle(cfg=None, native=None) -> str:
    """Ours vs the f64 spec (fp8_moe.moe_ref) and vs the oxide kernel at M=1..4
    on a synthetic LFM2.5-shaped problem (or a real layer cfg). Fatal on miss."""
    import torch
    from suffix_hybrid.kernels import fp8_moe as F
    from suffix_hybrid.kernels import nvfp4_moe as nm
    dev = torch.device("cuda", torch.cuda.current_device())
    lines = []
    for ue in ((cfg["ue8m0"],) if cfg else (False, True)):
        p = cfg or F.make_problem(64, 2048, 1536, device=dev, seed=3, ue8m0=ue)
        p = dict(p, E=p["w13"].shape[0], H=p["w2"].shape[1], I=p["w2"].shape[2],
                 topk=p.get("topk", 4), id_base=p.get("id_base", 0))
        for m in (1, 2, 3, 4):
            ids, tw = nm.rand_routing(m, p["E"], p["topk"], dev, seed=50 + m, base=0, local=p["E"],
                                      dead=None)
            x = torch.randn(m, p["H"], device=dev).bfloat16()
            ours = run(p, x, tw, ids).float()
            ref = F.moe_ref(p, x, nm.to_local(ids, 0, p["E"]), tw)
            rel = nm._rel(ours, ref)
            msg = f"ue8m0={ue} M={m}: rel_vs_ref={rel:.2e}"
            ok = rel <= F.REF_TOL
            if native is not None:
                ws = F.workspace(dev, 4, p["topk"], p["H"], p["I"], p["E"])
                ox = F.run_ours(native, p, x, ids, tw, ws,
                                torch.cuda.current_stream(dev).cuda_stream).float()
                r2 = nm._rel(ours, ox)
                msg += f" rel_vs_oxide={r2:.2e}"
                ok = ok and r2 <= 2 * F.REF_TOL
            lines.append(msg + (" OK" if ok else " FAIL"))
            if not ok:
                raise RuntimeError(f"{MARKER} ORACLE FAIL {msg}")
    for ln in lines:
        _log(ln)
    return f"{MARKER} ORACLE PASS ({len(lines)} cases)"


def bench(layers=8, iters=50) -> None:
    """Cold-L2 us per MoE layer call (8 distinct LFM2.5-shaped layers per graph):
    ours vs the oxide fused plan, M = 1..4, + achieved GB/s on the selected experts."""
    import torch
    from suffix_hybrid import oxide_kernels
    from suffix_hybrid.kernels import fp8_moe as F
    from suffix_hybrid.kernels import nvfp4_moe as nm
    native = F._native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    ps = [F.make_problem(64, 2048, 1536, device=dev, seed=10 + i) for i in range(layers)]
    for p in ps:
        p.update(E=64, H=2048, I=1536, topk=4, id_base=0)
    for m in (1, 2, 3, 4):
        ids, tw = nm.rand_routing(m, 64, 4, dev, seed=7 + m, base=0, local=64, dead=None)
        x = torch.randn(m, 2048, device=dev).bfloat16()
        ws = F.workspace(dev, 4, 4, 2048, 1536, 64)
        h = torch.empty(m * 4, 1536, dtype=torch.float32, device=dev)
        out = torch.empty(m, 2048, dtype=torch.bfloat16, device=dev)
        ours = lambda s: [run(p, x, tw, ids, out=out, h=h) for p in ps]
        oxide = lambda s: [F.run_ours(native, p, x, ids, tw, ws, s.cuda_stream, out, 2) for p in ps]
        t_ours = nm._graph_us(ours, dev, iters) / layers
        t_ox = nm._graph_us(oxide, dev, iters) / layers
        uniq = int(torch.unique(ids).numel())
        gbs = uniq * 3 * 2048 * 1536 / (t_ours * 1e-6) / 1e9
        _log(f"bench M={m} experts={uniq}: oxide {t_ox:.1f} us, ours {t_ours:.1f} us "
             f"(x{t_ours / t_ox:.2f}, {gbs:.0f} GB/s)")
    oxide_kernels  # noqa: B018 (keeps the import for the native check above)


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["both"])[0]
    try:
        import torch  # noqa: F401
        from suffix_hybrid.kernels import fp8_moe as F
        native = F._native_ready()
        if mode in ("oracle", "both"):
            _log(oracle(native=native))
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _log(f"{mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
