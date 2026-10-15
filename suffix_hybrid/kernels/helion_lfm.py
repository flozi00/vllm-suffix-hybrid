# SPDX-License-Identifier: Apache-2.0
"""Helion kernels for LFM2.5-24B-A2B serving on SM120 (Helion -> Triton ->
Triton's own ptxas -> SASS: loads on the driver-580 nodes with no PTX JIT,
like every vLLM Triton kernel).

Helion is vendored in the bundle under /plugins/_vendor (appended to
sys.path by sitecustomize, so the image's own packages always win). Configs
are autotuned ON the target GPU by the boot gate ``helion_lfm_tune``
(``python -m suffix_hybrid.kernels.helion_lfm tune``), which prints one
``TUNED`` JSON line per (kernel, shape, M bucket); serving pins those configs
(PINNED below, keyed by vLLM's canonical GPU name) and never autotunes.

Kernels:
  skinny_mm   y[M, N] = x[M, K] @ w[N, K]^T, bf16 in/out, fp32 accumulate
              (the LFM2 short-conv in/out projections at decode M).
"""
from __future__ import annotations

import json
import os
import sys
import time

MARKER = "[suffix helion]"
SHAPES = ((6144, 2048), (2048, 2048))  # (N, K): short_conv in_proj, out_proj
MS = (1, 8, 32)
# canonical GPU name (vllm.kernels.helion.utils) -> {(kernel, N, K, M bucket): config dict}
PINNED: dict = {}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def ensure_vendor() -> str:
    """Make the vendored Helion importable (lowest sys.path priority)."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "_vendor")
    if os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)
    return d


ensure_vendor()
import helion  # noqa: E402
import helion.language as hl  # noqa: E402
import torch  # noqa: E402


def m_bucket(m: int) -> int:
    for b in MS:
        if m <= b:
            return b
    return MS[-1]


def _skinny_mm(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = wt.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], wt[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def make_skinny(config=None, effort: str = "quick"):
    """Helion kernel over the module-level body (Helion refuses closures);
    config pinned when given, else autotuned on first call / .autotune()."""
    kw = dict(static_shapes=True, autotune_effort=effort)
    if config is not None:
        kw["config"] = helion.Config(**config) if isinstance(config, dict) else config
    return helion.kernel(_skinny_mm, **kw)


def _graph_us(fn, ws, iters=100):
    """us per call, CUDA-graph replayed over `ws` distinct weights (W not L2-hot)."""
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
    return e0.elapsed_time(e1) * 1e3 / iters / len(ws)


def tune(effort: str = "quick") -> int:
    from vllm.kernels.helion.utils import get_canonical_gpu_name
    from suffix_hybrid.kernels import bf16_skinny
    plat = get_canonical_gpu_name()
    from importlib.metadata import version
    _log(f"helion {version('helion')} torch {torch.__version__} on {plat}; effort={effort}")
    dev = torch.device("cuda", torch.cuda.current_device())
    for n, k in SHAPES:
        ws = [torch.randn(n, k, device=dev).bfloat16() for _ in range(24)]
        for m in MS:
            x = torch.randn(m, k, device=dev).bfloat16()
            kern = make_skinny(effort=effort)
            t0 = time.time()
            cfg = kern.autotune((x, ws[0].t()), force=True)
            secs = time.time() - t0
            best = make_skinny(config=cfg)
            ref = torch.nn.functional.linear(x, ws[0])
            ok, msg = bf16_skinny.check(best(x, ws[0].t()), ref)
            t_cub = _graph_us(lambda w: torch.nn.functional.linear(x, w), ws)
            t_hel = _graph_us(lambda w: best(x, w.t()), ws)
            t_tri = _graph_us(lambda w: bf16_skinny.skinny(x, w), ws)
            gbs = n * k * 2 / (t_hel * 1e-6) / 1e9
            _log(f"TUNED {json.dumps({'kernel': 'skinny_mm', 'platform': plat, 'N': n, 'K': k, 'M': m, 'config': json.loads(cfg.to_json())})}")
            _log(f"skinny_mm N={n} K={k} M={m}: cublas {t_cub:.2f} us, triton {t_tri:.2f} us, "
                 f"helion {t_hel:.2f} us ({gbs:.0f} GB/s, tuned in {secs:.0f} s) {msg} "
                 f"{'OK' if ok else 'FAIL'}")
        del ws
        torch.cuda.empty_cache()
    return 0


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["tune"])[0]
    try:
        if mode == "tune":
            return tune(os.environ.get("SUFFIX_HELION_EFFORT", "quick"))
        raise SystemExit(f"unknown mode {mode}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _log(f"{mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
