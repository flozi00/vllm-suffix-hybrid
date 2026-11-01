# SPDX-License-Identifier: Apache-2.0
"""BF16 skinny GEMM y[M, N] = x[M, K] @ W[N, K]^T for decode-sized M (Triton,
tensor cores via tl.dot with M padded to 16, fp32 accumulation, one bf16
rounding at the end — the same precision contract as cuBLAS' bf16 GEMM).

Target: the unquantized BF16 projections of LFM2 / LFM2-MoE short-conv
layers (in_proj 2048 -> 6144, out_proj 2048 -> 2048) where cuBLAS picks a
16x16 wmma GEMM (~77% of DRAM bandwidth) or gemv (~45%) at M <= 32. The
kernel streams W once with a deep cp.async pipeline (num_stages) over
BLOCK_N-row stripes, one program per stripe (no split-K, no reduction
launch, no atomics: deterministic and batch-invariant in M).

Gate: ``SUFFIX_BF16_SKINNY=1`` (default OFF) with
``SUFFIX_BF16_SKINNY_LAYERS`` = comma fnmatch globs over the module prefix
(default ``*.short_conv.in_proj,*.short_conv.out_proj``) and
``SUFFIX_BF16_SKINNY_MAX_M`` (default 32). Wiring: the allowlisted layers'
``UnquantizedLinearMethod`` instances are wrapped after weight loading (the
load-time oracle compares against F.linear on the layer's REAL weight at
M = 1, 3, 16, max_m: rel. error <= 1e-2 and max |d| within 2 bf16 ulps of
the reference magnitude). M > max_m and anything else runs F.linear.
"""
from __future__ import annotations

import fnmatch
import os
import sys

GATE = "SUFFIX_BF16_SKINNY"
LAYERS_ENV = "SUFFIX_BF16_SKINNY_LAYERS"
MAX_M_ENV = "SUFFIX_BF16_SKINNY_MAX_M"
DEFAULT_LAYERS = "*.short_conv.in_proj,*.short_conv.out_proj"
MARKER = "[suffix bf16-skinny]"

_state = {"armed": False, "layers": 0, "oracle": [], "launch_logged": False}
_kernel = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def max_m() -> int:
    v = int(os.environ.get(MAX_M_ENV, "32"))
    if not 1 <= v <= 64:
        raise ValueError(f"{MAX_M_ENV}={v} outside [1, 64]")
    return v


def layer_globs() -> list[str]:
    raw = os.environ.get(LAYERS_ENV, "").strip() or DEFAULT_LAYERS
    return [g.strip() for g in raw.split(",") if g.strip()]


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _skinny(x_ptr, w_ptr, y_ptr, M, N, K, s_xm, s_wn, s_ym,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm < M
        mn = rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            mk = kk < K
            x = tl.load(x_ptr + rm[:, None] * s_xm + kk[None, :], mask=mm[:, None] & mk[None, :],
                        other=0.0)
            w = tl.load(w_ptr + rn[:, None] * s_wn + kk[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        tl.store(y_ptr + rm[:, None] * s_ym + rn[None, :], acc.to(tl.bfloat16),
                 mask=mm[:, None] & mn[None, :])

    _kernel = _skinny
    return _kernel


def plan(m: int, n: int, k: int) -> dict:
    """Tile per shape: BN rows of W per program; aim for >= ~1.5 programs per
    SM on 188-SM parts while keeping each W stripe >= 32 KB."""
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    bn = 32 if n >= 4096 else 16
    return dict(BM=bm, BN=bn, BK=256, num_warps=4, num_stages=4)


def skinny(x, w, out=None):
    import torch
    import triton
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    p = plan(m, n, k)
    _get_kernel()[(triton.cdiv(n, p["BN"]),)](
        x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        BM=p["BM"], BN=p["BN"], BK=p["BK"], num_warps=p["num_warps"], num_stages=p["num_stages"])
    return out


def check(y, ref) -> tuple[bool, str]:
    rel = ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
    dmax = (y.float() - ref.float()).abs().max().item()
    # 2 bf16 ulps of the largest reference magnitude bounds two different fp32
    # summation orders each rounded once to bf16
    tol = 2 * 2.0 ** -7 * ref.float().abs().max().item()
    return rel <= 1e-2 and dmax <= tol, f"rel={rel:.2e} max|d|={dmax:.3e} (tol {tol:.3e})"


def layer_oracle(name, w, mm) -> str:
    import torch
    dev = w.device
    worst = ""
    for m in sorted({1, 3, min(16, mm), mm}):
        g = torch.Generator(device="cpu").manual_seed(m)
        x = torch.randn(m, w.shape[1], generator=g).to(dev, torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        ok, msg = check(skinny(x, w), ref)
        if not ok:
            raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL {name} M={m}: {msg} — "
                               f"refusing to serve with {GATE}=1")
        worst = msg
    return worst


def install(model) -> int:
    """Wrap the allowlisted unquantized BF16 linears' quant_method.apply (called
    after weight loading by the plugin's post-load hook)."""
    import torch
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    globs, mm, n = layer_globs(), max_m(), 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase) or not any(fnmatch.fnmatch(name, g) for g in globs):
            continue
        qm = mod.quant_method
        w = getattr(mod, "weight", None)
        if (type(qm) is not UnquantizedLinearMethod or w is None or w.dtype != torch.bfloat16
                or getattr(mod, "bias", None) is not None or w.shape[1] % 8):
            _log(f"{name}: stays on F.linear ({type(qm).__name__}, "
                 f"{getattr(w, 'dtype', None)})")
            continue
        msg = layer_oracle(name, w, mm)
        orig = qm.apply

        def apply(layer, x, bias=None, _orig=orig, _w=w):
            if bias is None and x.dim() == 2 and 1 <= x.shape[0] <= mm and x.dtype == torch.bfloat16:
                return skinny(x, _w)
            return _orig(layer, x, bias)

        qm.apply = apply
        n += 1
        if n == 1:
            _log(f"LAYER ORACLE PASS {name} N={w.shape[0]} K={w.shape[1]} ({msg})")
    _state["layers"] = n
    _log(f"ACTIVE on {n} layers (M <= {mm}, globs {','.join(globs)})")
    if not n:
        raise RuntimeError(f"{MARKER} {GATE}=1 but no layer matched {globs}")
    return n


def bench(shapes=((6144, 2048), (2048, 2048), (128000, 2048)), ms=(1, 2, 4, 8, 16, 32),
          iters=200, layers=30) -> dict:
    """Graph-replayed us per GEMM (cold-ish: `layers` distinct weights per
    graph so W is not L2-resident) vs cuBLAS (F.linear) + achieved GB/s."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    for n, k in shapes:
        nl = layers if n * k * 2 * layers < 6e9 else 2
        ws = [torch.randn(n, k, device=dev).bfloat16() for _ in range(nl)]
        for m in ms:
            x = torch.randn(m, k, device=dev).bfloat16()
            outs = {}
            for name, fn in (("cublas", lambda w: torch.nn.functional.linear(x, w)),
                             ("ours", lambda w: skinny(x, w))):
                for w in ws:
                    fn(w)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                    for w in ws:
                        fn(w)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                outs[name] = e0.elapsed_time(e1) * 1e3 / iters / nl
            gbs = n * k * 2 / (outs["ours"] * 1e-6) / 1e9
            ok, msg = check(skinny(x, ws[0]), torch.nn.functional.linear(x, ws[0]))
            res[(n, k, m)] = outs
            _log(f"bench N={n} K={k} M={m}: cublas {outs['cublas']:.2f} us, ours {outs['ours']:.2f} us "
                 f"(x{outs['ours'] / outs['cublas']:.2f}, {gbs:.0f} GB/s) {msg} {'OK' if ok else 'FAIL'}")
        del ws
        torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    try:
        bench()
        return 0
    except Exception as exc:  # noqa: BLE001
        _log(f"BENCH FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
