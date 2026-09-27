# SPDX-License-Identifier: Apache-2.0
"""NVFP4 lm_head for decode: the bf16 vocab projection is the largest single
weight read per decode step on our NVFP4 checkpoints (they exclude lm_head),
e.g. qwen3.8-27b 248320 x 5120 x 2 B = 2.54 GB/step = ~15 % of all bytes.
This serves it as W4A4 NVFP4 (0.5625 B/param -> 0.72 GB) through our sm_120a
decode GEMM (kernels-oxide/nvfp4_gemm, prequant route).

Gate: ``SUFFIX_NVFP4_LMHEAD=1`` (default OFF). Entry: ``register()``, the
``vllm.general_plugins`` entry point ``suffix_nvfp4_lm_head``.

Seam (no fork, no _custom_ops patching): ``ParallelLMHead`` is a vLLM
PluggableLayer (vocab_parallel_embedding.py:568) -> ``register_oot`` swaps in
a subclass whose ``quant_method`` wraps the stock unquantized one
(UnquantizedEmbeddingMethod, or UnquantizedLinearMethod when a ModelOpt
config excludes lm_head). ``LogitsProcessor._apply_head`` calls
``lm_head.quant_method.apply`` (logits_processor.py:144). Our method
subclasses UnquantizedEmbeddingMethod so vLLM's isinstance checks (weight
re-tying, head_dtype) behave as stock; the bf16 weight stays resident (tied
embeddings, gemma MTP's centroid head and fallbacks read it).

Load (``process_weights_after_loading``, before profiling / graph capture):
per-tensor weight global scale g_w = 448*6/amax(W), vLLM's own
``scaled_fp4_quant`` -> packed [N, K/2] + 128x4-swizzled e4m3 scales (the
exact layout our GEMM reads), then a LOAD ORACLE per head: ours vs the exact
quantized reference on sampled vocab rows (fatal) + fidelity vs the stock
bf16 head (rel / cos / top-1 agreement, logged).

Hot path (M = rows <= 16, bf16, no bias; else the stock method): dynamic
per-call activation scale on device (g_x = 448*6/amax(x), no host sync),
``scaled_fp4_quant`` -> ``nvfp4_gemm_q_cuda`` with alpha = 1/g_w -> ``*
1/g_x`` in place -> top-RESCORE candidates per row recomputed EXACTLY from
the bf16 rows and scattered back. W4A4 alone has rel ~0.13 logit noise
(flips greedy on ~28 % of random-x rows, CPU twin); the rescore makes
greedy / top-k picks see stock logits, only the tail keeps the noise.
CUDA-graph capturable.

Markers: ``[suffix nvfp4-lmhead] armed`` / ``LOAD ORACLE PASS`` / ``ACTIVE``.
CLI (in-pod, no model): ``python -m suffix_hybrid.kernels.nvfp4_lm_head
oracle|bench|both``; boot gates ``nvfp4_lmhead_oracle`` / ``nvfp4_lmhead_bench``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from suffix_hybrid.kernels.nvfp4_gemm import (
    dequant,
    e4m3_bits_to_f32,
    sf_offset,
)

GATE = "SUFFIX_NVFP4_LMHEAD"
MARKER = "[suffix nvfp4-lmhead]"
FAMILY = "nvfp4_gemm"
MAX_M = 16
FP4_RANGE = 448.0 * 6.0  # e4m3 max * e2m1 max: g = FP4_RANGE / amax
PLUMB_REL = 2e-2  # ours vs exact quantized reference (same bar as nvfp4_gemm)
RESCORE = 64  # NVFP4 top candidates per row re-scored exactly in bf16
# (N, K) per rank: qwen3.8-27b (TP1), gemma-4-26b-a4b (tied, TP1),
# GLM-5.3 vocab-parallel TP8 (154880 / 8).
SHAPES = {"qwen": (248320, 5120), "gemma": (262144, 2816), "glm_tp8": (19360, 6144)}
_state = {"armed": False, "heads": 0, "stock": 0, "active": False, "hook": None,
          "checked": False, "oracle": {}}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def eligible(n: int, k: int, dtype: str) -> str | None:
    """Why an lm_head shard [n, k] cannot use our kernel, or None."""
    if dtype != "torch.bfloat16":
        return f"dtype {dtype} (bf16 only)"
    if n % 32 or k % 64:
        return f"N={n} % 32 or K={k} % 64"
    return None


def logits_nvfp4(x2, w, g_w: float, quant, gemm):
    """The hot path with its two device ops injected (vLLM scaled_fp4_quant,
    our GEMM; CPU twins in tests). x2 bf16 [M, K], w the bf16 head [N, K]
    -> bf16 [M, N]: NVFP4 screen over the vocab, then the RESCORE best
    candidates per row recomputed exactly from the bf16 rows (M*RESCORE*K*2 B,
    ~0.6 MB/row) and scattered back, so greedy / top-k picks see stock
    logits; only the tail keeps W4A4 noise (rel ~0.13)."""
    import torch
    amax = x2.abs().amax().float().clamp_min(1e-12)
    xq, xsf = quant(x2, FP4_RANGE / amax)
    out = gemm(xq, xsf, 1.0 / g_w)  # = acc / g_w
    out.mul_(amax / FP4_RANGE)  # * 1/g_x, on device
    top = out.topk(min(RESCORE, out.shape[-1]), dim=-1).indices  # [M, R]
    exact = torch.bmm(w[top], x2.unsqueeze(-1)).squeeze(-1)  # [M, R] bf16
    return out.scatter_(1, top, exact)


def greedy_ok(ref, got) -> bool:
    """Every row's argmax under ``got`` is the stock argmax, or a near-tie
    of it (within bf16 resolution of the row max: accumulation order may
    legitimately flip exact bf16 ties)."""
    import torch
    ref = ref.float()
    pick = ref.gather(-1, got.float().argmax(-1, keepdim=True)).squeeze(-1)
    best = ref.max(-1).values
    return bool(torch.all(pick >= best - best.abs().clamp_min(1.0) * 2.0 ** -7))


# ---------------------------------------------------------------------------
# exact reference on sampled rows (numpy; shared by the load oracle + tests)
# ---------------------------------------------------------------------------
def sf_index(rows, nkb: int):
    """Flat byte offsets [len(rows), nkb] of rows' scales in a 128x4-swizzled
    scale buffer (vLLM swizzle_blockscale)."""
    rows = np.asarray(rows, dtype=np.int64)[:, None]
    kb = np.arange(nkb, dtype=np.int64)[None, :]
    return sf_offset(rows, kb, -(-nkb // 4) * 4)


def ref_logits(xq, xsf_bits, wq_rows, wsf_bits_rows, g_x: float, g_w: float):
    """fp32 [M, R] = dequant(x) @ dequant(W[rows]).T / (g_x * g_w); scales
    given as raw e4m3 bits in row-major [rows, K/16]."""
    a = dequant(np.asarray(xq), e4m3_bits_to_f32(np.asarray(xsf_bits)))
    b = dequant(np.asarray(wq_rows), e4m3_bits_to_f32(np.asarray(wsf_bits_rows)))
    return (a @ b.T / (np.float32(g_x) * np.float32(g_w))).astype(np.float32)


def fidelity(ref, got) -> dict:
    """Stock-vs-NVFP4 logits: relative error, cosine, top-1 (greedy) agreement."""
    import torch
    ref, got = ref.float(), got.float()
    return dict(rel=float((got - ref).norm() / ref.norm().clamp_min(1e-30)),
                cos=float(torch.nn.functional.cosine_similarity(
                    got.flatten(), ref.flatten(), dim=0)),
                top1=float((got.argmax(-1) == ref.argmax(-1)).float().mean()))


# ---------------------------------------------------------------------------
# device side (SM120 + oxide bundle + vLLM)
# ---------------------------------------------------------------------------
def _device_ops(native):
    import torch
    from vllm._custom_ops import scaled_fp4_quant

    def prepare(w):
        """bf16 [N, K] -> dict(wq, wsf, g_w, splits, partial). Load time only."""
        n, k = w.shape
        amax = float(torch.linalg.vector_norm(w, float("inf")).float())  # no |W| copy
        g_w = FP4_RANGE / max(amax, 1e-12)
        wq, wsf = scaled_fp4_quant(
            w, torch.tensor(g_w, dtype=torch.float32, device=w.device),
            is_sf_swizzled_layout=True)
        splits = native.nvfp4_gemm_splits(n, k)
        return dict(n=n, k=k, w=w, wq=wq, wsf=wsf, g_w=g_w, splits=splits,
                    partial=torch.empty(splits * 16 * n, dtype=torch.float32,
                                        device=w.device))

    def quant(x2, g):
        return scaled_fp4_quant(x2, g, is_sf_swizzled_layout=True)

    def run(st, x2):
        dev = x2.device

        def gemm(xq, xsf, alpha):
            out = torch.empty(xq.shape[0], st["n"], dtype=torch.bfloat16, device=dev)
            native.nvfp4_gemm_q_cuda(xq, xsf, st["wq"], st["wsf"], st["partial"], out,
                                     alpha, st["splits"],
                                     torch.cuda.current_stream(dev).cuda_stream)
            return out
        return logits_nvfp4(x2, st["w"], st["g_w"], quant, gemm)

    return prepare, quant, run


def load_oracle(st, run, quant, stock, n_rows=256, seed=0) -> dict:
    """FATAL checks: (1) the NVFP4 screen vs the exact quantized reference on
    n_rows sampled vocab rows that were NOT re-scored (<= PLUMB_REL); (2)
    greedy agreement with the stock bf16 head on every row (``greedy_ok``).
    Logged: rel / cos / top-1 vs stock (rel is the tail's W4A4 noise).
    ``stock(x)`` -> bf16 logits [M, N]."""
    import torch
    n, k = st["n"], st["k"]
    dev = st["wq"].device
    rows = np.linspace(0, n - 1, min(n_rows, n)).astype(np.int64)
    idx = torch.from_numpy(sf_index(rows, k // 16).reshape(-1)).to(dev)
    wsf_rows = st["wsf"].view(torch.uint8).reshape(-1)[idx].view(len(rows), -1).cpu().numpy()
    wq_rows = st["wq"][torch.from_numpy(rows).to(dev)].cpu().numpy()
    gen = torch.Generator(device=dev).manual_seed(seed)
    res = {"plumb_rel": 0.0}
    for m in (1, 4, MAX_M):
        x = torch.randn(m, k, generator=gen, device=dev, dtype=torch.bfloat16)
        got = run(st, x)
        amax = float(x.abs().amax().float())
        g_x = FP4_RANGE / amax
        xq, xsf = quant(x, torch.tensor(g_x, dtype=torch.float32, device=dev))
        xsf_bits = xsf.view(torch.uint8).reshape(-1)[
            torch.from_numpy(sf_index(np.arange(m), k // 16).reshape(-1)).to(dev)
        ].view(m, -1).cpu().numpy()
        ref = torch.from_numpy(ref_logits(xq.cpu().numpy(), xsf_bits, wq_rows, wsf_rows,
                                          g_x, st["g_w"]))
        sub = got.float()[:, torch.from_numpy(rows).to(dev)].cpu()
        top = got.float().topk(min(RESCORE, n), dim=-1).indices.cpu().numpy()
        keep = torch.from_numpy(~np.isin(rows[None, :], top))  # screen-only cells
        rel = float((sub - ref)[keep].norm() / ref[keep].norm().clamp_min(1e-30))
        want = stock(x)
        if not rel <= PLUMB_REL or not torch.isfinite(got).all():
            raise RuntimeError(f"{MARKER} LOAD ORACLE FAIL N={n} K={k} M={m}: "
                               f"rel_vs_exact={rel:.3e} (> {PLUMB_REL})")
        if not greedy_ok(want, got):
            raise RuntimeError(f"{MARKER} LOAD ORACLE FAIL N={n} K={k} M={m}: greedy "
                               "argmax differs from the stock bf16 head")
        res["plumb_rel"] = max(res["plumb_rel"], rel)
        res[f"m{m}"] = fidelity(want, got)
    return res


def _make_head_cls(native):
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )
    from suffix_hybrid import oxide_kernels

    prepare, quant, run = _device_ops(native)

    class SuffixNvFp4LMHeadMethod(UnquantizedEmbeddingMethod):
        """Stock bf16 head + NVFP4 copy for M <= 16 decode rows."""
        supports_pre_processed_weights = False  # IPC weight cache: fail loud

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def tie_weights(self, layer, embed_tokens):
            return self.inner.tie_weights(layer, embed_tokens)

        def process_weights_after_loading(self, layer):
            self.inner.process_weights_after_loading(layer)
            w = layer.weight
            n, k = w.shape
            why = eligible(n, k, str(w.dtype)) or (None if w.is_cuda else "not on CUDA")
            layer._sfx_lmhead = None
            if why is not None:
                _state["stock"] += 1
                _log(f"lm_head N={n} K={k} stays bf16: {why}")
                return
            oxide_kernels.ensure_loaded(FAMILY, w.device.index)
            st = prepare(w)
            key = (n, k, st["g_w"])
            if key not in _state["oracle"]:
                r = load_oracle(st, run, quant, lambda x: self.inner.apply(layer, x))
                torch.cuda.synchronize(w.device)
                _state["oracle"][key] = r
                fid = " ".join(f"M={m[1:]}:rel={v['rel']:.3f},cos={v['cos']:.5f},"
                               f"top1={v['top1']:.2f}" for m, v in r.items() if m != "plumb_rel")
                _log(f"LOAD ORACLE PASS N={n} K={k} splits={st['splits']} "
                     f"rel_vs_exact={r['plumb_rel']:.2e}; vs bf16 (random x) {fid}")
            layer._sfx_lmhead = st
            _state["heads"] += 1

        def apply(self, layer, x, bias=None):
            st = getattr(layer, "_sfx_lmhead", None)
            rows = x.numel() // x.shape[-1] if x.numel() else 0
            if (st is None or bias is not None or x.dtype != torch.bfloat16
                    or not 1 <= rows <= MAX_M):
                return self.inner.apply(layer, x, bias)
            out = run(st, x.reshape(rows, st["k"]))
            if not _state["active"]:
                _state["active"] = True
                _log(summary())
            return out.view(*x.shape[:-1], st["n"])

    class SuffixNvFp4ParallelLMHead(ParallelLMHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            qm = type(self.quant_method).__name__
            if qm in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
                self.quant_method = SuffixNvFp4LMHeadMethod(self.quant_method)
            else:  # a checkpoint-quantized head: not ours to replace
                _log(f"lm_head keeps {qm} (checkpoint-quantized)")

    return SuffixNvFp4ParallelLMHead


def _first_forward_check(module, args):
    """Engagement check at the first module call after load (before graph
    capture): gate on but no head converted -> fail startup, loud."""
    h = _state.pop("hook", None)
    if h is not None:
        h.remove()
    if _state["checked"]:
        return
    _state["checked"] = True
    if _state["heads"] == 0:
        msg = (f"{MARKER} NVFP4 lm_head NOT ENGAGED with {GATE}=1: 0 heads converted "
               f"({_state['stock']} stayed bf16) — see the lines above")
        _log(msg)
        raise RuntimeError(msg)
    _log(f"SELECTION: {_state['heads']} lm_head(s) on NVFP4, {_state['stock']} bf16")


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_NVFP4_LMHEAD=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    import torch
    from suffix_hybrid import oxide_kernels
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot

    native = oxide_kernels.native()
    for fn in ("nvfp4_gemm_q_cuda", "nvfp4_gemm_splits"):
        if not hasattr(native, fn):
            raise RuntimeError(f"{GATE}=1 but _native lacks {fn} (oxide-kernels build)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError(f"{GATE}=1: the NVFP4 GEMM cubin is sm_120a SASS (cc 12.x only)")
    if not any(k["name"] == FAMILY for k in oxide_kernels.manifest()["kernels"]):
        raise RuntimeError(f"{GATE}=1 but the oxide manifest has no {FAMILY!r} cubin")
    if "ParallelLMHead" in op_registry_oot:
        raise RuntimeError(f"{GATE}=1: ParallelLMHead already OOT-replaced by "
                           f"{op_registry_oot['ParallelLMHead']}")
    PluggableLayer.register_oot(_make_head_cls(native), name="ParallelLMHead")
    _state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        _first_forward_check)
    _state["armed"] = True
    _log(f"armed: ParallelLMHead OOT-registered (M<={MAX_M} -> NVFP4 decode GEMM, "
         "else stock bf16)")
    return _state


def summary() -> str:
    return (f"NVFP4 lm_head ACTIVE: {_state['heads']} head(s) on our decode GEMM, "
            f"{_state['stock']} stay bf16")


# ---------------------------------------------------------------------------
# in-pod CLI (no model): synthetic heads at our three shapes
# ---------------------------------------------------------------------------
def _synthetic(n, k, dev, seed=0):
    import torch
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(n, k, generator=g, device=dev) * 0.02).to(torch.bfloat16)


def main(argv=None) -> int:
    import torch
    from suffix_hybrid import oxide_kernels
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        native = oxide_kernels.native()
        if torch.cuda.get_device_capability()[0] != 12:
            raise RuntimeError("sm_120a SASS needs cc 12.x")
        oxide_kernels.ensure_loaded(FAMILY)
        prepare, quant, run = _device_ops(native)
        dev = torch.device("cuda", torch.cuda.current_device())
        for name, (n, k) in SHAPES.items():
            w = _synthetic(n, k, dev)
            st = prepare(w)
            if mode in ("oracle", "both"):
                r = load_oracle(st, run, quant, lambda x: x @ w.t())
                _log(f"ORACLE {name} N={n} K={k}: rel_vs_exact={r['plumb_rel']:.2e} "
                     + " ".join(f"{m}={v}" for m, v in r.items() if m != "plumb_rel"))
            if mode in ("bench", "both"):
                for m in (1, 4, MAX_M):
                    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
                    t = [_graph_us(lambda: x @ w.t(), dev), _graph_us(lambda: run(st, x), dev)]
                    roof = [n * k * 2 / 1.79e6, n * k * 0.5625 / 1.79e6]
                    _log(f"bench {name} M={m} N={n} K={k}: bf16 {t[0]:.1f} us "
                         f"(roofline {roof[0]:.1f}), nvfp4 {t[1]:.1f} us (roofline "
                         f"{roof[1]:.1f}), saved {t[0] - t[1]:.1f} us/step")
        if mode in ("oracle", "both"):
            _log(f"NVFP4-LMHEAD ORACLE PASS ({len(SHAPES)} shapes)")
        return 0
    except Exception as exc:
        _log(f"NVFP4-LMHEAD {mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


def _graph_us(fn, dev, iters=100):
    import torch
    s = torch.cuda.Stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream(dev).wait_stream(s)
    g.replay()
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize(dev)
    return e0.elapsed_time(e1) * 1000.0 / iters


if __name__ == "__main__":
    sys.exit(main())
