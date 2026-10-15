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
PINNED: dict = {
    # helion_lfm_tune on worker-01 Max-Q 2026-10-15 (PatternSearch, effort quick); us at M 1/8/32:
    # in_proj 6144x2048 helion 17.08/17.50/19.13 vs cuBLAS 17.95/17.99/18.95;
    # out_proj 2048x2048 helion 7.55/8.25/- vs our Triton 6.86/6.94 vs cuBLAS 8.33/13.29.
    'nvidia_rtx_pro_6000_blackwell_max_q_workstation_edition': {
        ('skinny_mm', 6144, 2048, 1): {'block_sizes': [1, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat', 'host_tensor_descriptors': False},
        ('skinny_mm', 6144, 2048, 8): {'block_sizes': [16, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat', 'host_tensor_descriptors': False},
        ('skinny_mm', 6144, 2048, 32): {'block_sizes': [32, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat', 'host_tensor_descriptors': False},
        ('skinny_mm', 2048, 2048, 1): {'block_sizes': [1, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat', 'host_tensor_descriptors': False},
        ('skinny_mm', 2048, 2048, 8): {'block_sizes': [8, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat', 'host_tensor_descriptors': False},
    },
}


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
# Helion's default search (LFBO) needs scikit-learn + scipy (not vendored, ~50 MB of binaries);
# PatternSearch needs neither. An explicit HELION_AUTOTUNER in the env still wins.
os.environ.setdefault("HELION_AUTOTUNER", "PatternSearch")
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


def _fp8_block_mm(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor,
                  ws: torch.Tensor) -> torch.Tensor:
    """y[M, N] bf16 = (xq * xs) @ (wq * ws)^T: xq e4m3 [M, K] with fp32 1x128 group scales
    xs [M, K/128]; wq e4m3 [N, K] with fp32 128x128 block scales ws [N/128, K/128]. Same
    per-128-K-block rescale order as vLLM's block-FP8 GEMMs (fp32 accumulate)."""
    m, k = xq.size()
    n = wq.size(0)
    out = torch.empty([m, n], dtype=torch.bfloat16, device=xq.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=128):
            p = hl.dot(xq[tile_m, tile_k], wq[tile_n, tile_k].T, out_dtype=torch.float32)
            sa = xs[tile_m, tile_k.id]
            sb = ws[tile_n.begin // 128, tile_k.id]
            acc = acc + p * sa[:, None] * sb
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def make_fp8_block(config=None, effort: str = "quick"):
    kw = dict(static_shapes=True, autotune_effort=effort)
    if config is not None:
        kw["config"] = helion.Config(**config) if isinstance(config, dict) else config
    return helion.kernel(_fp8_block_mm, **kw)


def _skinny_mm_out(x: torch.Tensor, wt: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Serving form of _skinny_mm: writes into `out` (the caller's rows, e.g. the
    short-conv op's output buffer), so no copy follows the GEMM."""
    m, k = x.size()
    _, n = wt.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], wt[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


_serving: dict = {}
_checked: set = set()


def pinned_config(kernel: str, n: int, k: int, mb: int) -> dict:
    from vllm.kernels.helion.utils import get_canonical_gpu_name
    plat = get_canonical_gpu_name()
    cfg = PINNED.get(plat, {}).get((kernel, n, k, mb))
    if cfg is None:
        raise KeyError(f"{MARKER} no pinned {kernel} config for N={n} K={k} M<={mb} on {plat} "
                       f"(run the helion_lfm_tune boot gate and pin its TUNED lines)")
    return cfg


def skinny(x, w, out=None):
    """Serving entry (bf16_skinny.skinny signature): y = x @ w.T with the config pinned for
    (N, K, M bucket) on this GPU, dynamic M inside the bucket. The first eager call per
    bucket checks against F.linear (bf16_skinny.check) and raises on a miss."""
    m = x.shape[0]
    n, k = w.shape
    key = (n, k, m_bucket(m))
    kern = _serving.get(key)
    if kern is None:
        cfg = pinned_config("skinny_mm", *key)
        kern = _serving[key] = helion.kernel(_skinny_mm_out, config=helion.Config(**cfg),
                                             static_shapes=False)
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    kern(x, w.t(), out)
    if key not in _checked and not torch.cuda.is_current_stream_capturing():
        from suffix_hybrid.kernels import bf16_skinny
        ok, msg = bf16_skinny.check(out, torch.nn.functional.linear(x, w))
        if not ok:
            raise RuntimeError(f"{MARKER} skinny ORACLE FAIL N={n} K={k} M={m}: {msg}")
        _checked.add(key)
        _log(f"skinny ORACLE PASS N={n} K={k} M={m} (bucket {key[2]}) vs F.linear: {msg}")
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


FP8_SHAPES = ((3072, 2048), (2048, 2048), (23552, 2048), (2048, 11776))  # qkv, o, dense w13, w2


def _fp8_problem(m, n, k, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xq = (torch.randn(m, k, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn).to(dev)
    xs_t = (torch.rand(k // 128, m, generator=g) * 0.01 + 1e-3).to(dev)  # CUTLASS: column-major
    wq = (torch.randn(n, k, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn).to(dev)
    ws = (torch.rand(n // 128, k // 128, generator=g) * 0.01 + 1e-3).to(dev)
    return xq, xs_t.t(), wq, ws


def _fp8_ref(xq, xs, wq, ws):
    a = xq.float() * xs.repeat_interleave(128, 1)
    b = wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return a @ b.t()


def tune_fp8(effort: str = "quick") -> int:
    from vllm import _custom_ops as ops
    from vllm.kernels.helion.utils import get_canonical_gpu_name
    plat = get_canonical_gpu_name()
    dev = torch.device("cuda", torch.cuda.current_device())
    for n, k in FP8_SHAPES:
        for m in MS:
            xq, xs, wq, ws = _fp8_problem(m, n, k, dev)
            ref = _fp8_ref(xq, xs, wq, ws)
            t0 = time.time()
            cfg = make_fp8_block(effort=effort).autotune((xq, xs.contiguous(), wq, ws), force=True)
            secs = time.time() - t0
            best = make_fp8_block(config=cfg)
            y = best(xq, xs.contiguous(), wq, ws)
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            yc = ops.cutlass_scaled_mm(xq, wq.t(), out_dtype=torch.bfloat16, scale_a=xs,
                                       scale_b=ws.t())
            relc = ((yc.float() - ref).norm() / ref.norm()).item()
            nw = 24 if n * k < 64e6 else 4
            wqs = [wq] + [(torch.randn(n, k, device=dev) * 50).clamp(-448, 448)
                          .to(torch.float8_e4m3fn) for _ in range(nw - 1)]
            xsc = xs.contiguous()
            t_cut = _graph_us(lambda w: ops.cutlass_scaled_mm(
                xq, w.t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws.t()), wqs)
            t_hel = _graph_us(lambda w: best(xq, xsc, w, ws), wqs)
            gbs = n * k / (t_hel * 1e-6) / 1e9
            _log(f"TUNED {json.dumps({'kernel': 'fp8_block_mm', 'platform': plat, 'N': n, 'K': k, 'M': m, 'config': json.loads(cfg.to_json())})}")
            _log(f"fp8_block_mm N={n} K={k} M={m}: cutlass {t_cut:.2f} us (rel {relc:.1e}), "
                 f"helion {t_hel:.2f} us (rel {rel:.1e}, {gbs:.0f} GB/s, tuned in {secs:.0f} s) "
                 f"{'OK' if rel < 1e-2 else 'FAIL'}")
            del wqs
        torch.cuda.empty_cache()
    return 0


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["tune"])[0]
    try:
        effort = os.environ.get("SUFFIX_HELION_EFFORT", "quick")
        if mode == "tune":
            return tune(effort)
        if mode == "tune_fp8":
            return tune_fp8(effort)
        raise SystemExit(f"unknown mode {mode}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _log(f"{mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
