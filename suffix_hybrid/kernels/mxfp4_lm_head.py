# SPDX-License-Identifier: Apache-2.0
"""MXFP4 lm_head for decode on ROCm gfx950: the AMD twin of nvfp4_lm_head.

Qwen3.8-Flash-Next keeps lm_head BF16 (248320 x 2560 = 1.27 GB), and with MTP
the head is read once per draft step plus once per verify: at c=1, k=3 that is
~5 GB of the step's bytes. Served as W4A4 MXFP4 it is 0.34 GB per read.

Gate: ``SUFFIX_MXFP4_LMHEAD=1`` (default OFF). Entry: ``register()``, the
``vllm.general_plugins`` entry point ``suffix_mxfp4_lm_head``. Same seam as
nvfp4_lm_head (``ParallelLMHead`` register_oot, quant_method wrapper that
keeps the BF16 weight resident), so the two gates are mutually exclusive.

Load: the weight is quantized with AITER's dynamic MXFP4 quant (e8m0 per 32);
scales are stored [K/32, N] as vLLM's AITER Triton linear does. Hot path for
M <= SUFFIX_MXFP4_LMHEAD_MAX_M rows (default 16): dynamic MXFP4 activation
quant -> AITER Triton gemm_afp4wfp4 over the vocab -> the top-RESCORE
candidates per row recomputed EXACTLY from the BF16 rows and scattered back,
so greedy / top-k picks see stock logits and only the tail keeps W4A4 noise.
One HIP graph per M is captured at load (``SUFFIX_MXFP4_LMHEAD_GRAPH=0`` =
eager); graph replay must be bit-identical to eager and greedy picks must
agree with the stock head on random rows, or startup fails.

Markers: ``[suffix mxfp4-lmhead] armed`` / ``LOAD ORACLE PASS`` / ``ACTIVE``.
CLI (in-pod, no model): ``python -m suffix_hybrid.kernels.mxfp4_lm_head`` =
fidelity + us/call vs the stock BF16 head at M=1..max_m; boot gate
``mxfp4_lmhead_bench``.
"""
from __future__ import annotations

import os
import sys

from suffix_hybrid.kernels.nvfp4_lm_head import fidelity, greedy_ok, replay

GATE = "SUFFIX_MXFP4_LMHEAD"
MARKER = "[suffix mxfp4-lmhead]"
MAX_M_ENV = "SUFFIX_MXFP4_LMHEAD_MAX_M"
GRAPH_ENV = "SUFFIX_MXFP4_LMHEAD_GRAPH"
MAX_M = 16
RESCORE = 64
ORACLE_MS = (1, 2, 4, 16)
SHAPE = (248320, 2560)  # qwen3.8-flash-next
_state = {"armed": False, "heads": 0, "stock": 0, "active": False, "hook": None,
          "checked": False}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def max_m_env() -> int:
    raw = os.environ.get(MAX_M_ENV, "").strip()
    m = int(raw) if raw else MAX_M
    if not 1 <= m <= 64:
        raise ValueError(f"{MAX_M_ENV}={raw!r}: 1..64")
    return m


def graph_on() -> bool:
    return os.environ.get(GRAPH_ENV, "1").strip() != "0"


def eligible(n: int, k: int, dtype: str) -> str | None:
    if dtype != "torch.bfloat16":
        return f"dtype {dtype} (bf16 only)"
    if k % 32:
        return f"K={k} % 32"
    return None


def _ops():
    """(prepare, run, capture); imports aiter, i.e. initializes HIP: worker side only."""
    if "ops" in _state:
        return _state["ops"]
    import torch
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    def prepare(w, max_m):
        n, k = w.shape
        wq, ws = dynamic_mxfp4_quant(w)  # [N, K/2] e2m1 pairs, [N, K/32] e8m0
        return dict(n=n, k=k, w=w, wq=wq, ws_t=ws.T.contiguous(), max_m=max_m)

    def run(st, x2, out=None):
        """x2 bf16 [M, K] -> bf16 [M, N]: MXFP4 screen + exact rescore."""
        xq, xs = dynamic_mxfp4_quant(x2)
        y = out if out is not None else torch.empty(
            x2.shape[0], st["n"], dtype=torch.bfloat16, device=x2.device)
        gemm_afp4wfp4(xq, st["wq"], xs, st["ws_t"].T, torch.bfloat16, y)
        top = y.topk(min(RESCORE, st["n"]), dim=-1).indices  # [M, R]
        exact = torch.bmm(st["w"][top], x2.unsqueeze(-1)).squeeze(-1)  # [M, R] bf16
        return y.scatter_(1, top, exact)

    def capture(st):
        """One HIP graph of run() per M over static gx / gout (shared pool),
        then graph replay == eager on ORACLE_MS (fatal). Load time only."""
        from unittest import mock
        dev, max_m = st["w"].device, st["max_m"]
        cur = torch.cuda.current_stream(dev)
        gx = torch.zeros(max_m, st["k"], dtype=torch.bfloat16, device=dev)
        gout = torch.empty(max_m, st["n"], dtype=torch.bfloat16, device=dev)
        pool, s, graphs = torch.cuda.graph_pool_handle(), torch.cuda.Stream(dev), [None]
        s.wait_stream(cur)
        with torch.cuda.stream(s), mock.patch("gc.collect", lambda *a: 0), \
                mock.patch("torch.cuda.empty_cache", lambda: None):
            for m in range(1, max_m + 1):
                run(st, gx[:m], gout[:m])  # warm-up: Triton JIT + topk workspaces
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=s):
                    run(st, gx[:m], gout[:m])
                graphs.append(g)
        cur.wait_stream(s)
        st.update(gx=gx, gout=gout, graphs=graphs)
        gen = torch.Generator(device=dev).manual_seed(1)
        for m in (m for m in ORACLE_MS if m <= max_m):
            x = torch.randn(m, st["k"], generator=gen, device=dev, dtype=torch.bfloat16)
            if not torch.equal(replay(st, x), run(st, x)):
                raise RuntimeError(f"{MARKER} GRAPH CHECK FAIL M={m}: replay != eager "
                                   f"(set {GRAPH_ENV}=0)")
        torch.cuda.synchronize(dev)

    _state["ops"] = (prepare, run, capture)
    return _state["ops"]


def load_oracle(st, run, stock, seed=0) -> dict:
    """FATAL: greedy agreement with the stock BF16 head on random rows (the
    rescore makes the argmax exact). Logged: rel / cos / top-1 vs stock."""
    import torch
    gen = torch.Generator(device=st["w"].device).manual_seed(seed)
    res = {}
    for m in (m for m in ORACLE_MS if m <= st["max_m"]):
        x = torch.randn(m, st["k"], generator=gen, device=st["w"].device, dtype=torch.bfloat16)
        got, want = run(st, x), stock(x)
        if not torch.isfinite(got).all() or not greedy_ok(want, got):
            raise RuntimeError(f"{MARKER} LOAD ORACLE FAIL N={st['n']} K={st['k']} M={m}: "
                               "greedy argmax differs from the stock bf16 head")
        res[m] = fidelity(want, got)
    return res


def _make_head_cls():
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    class SuffixMxFp4LMHeadMethod(UnquantizedEmbeddingMethod):
        """Stock bf16 head + MXFP4 copy for M <= max M decode rows."""
        supports_pre_processed_weights = False

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def tie_weights(self, layer, embed_tokens):
            return self.inner.tie_weights(layer, embed_tokens)

        def process_weights_after_loading(self, layer):
            self.inner.process_weights_after_loading(layer)
            w = layer.weight
            n, k = w.shape
            arch = torch.cuda.get_device_properties(w.device).gcnArchName if w.is_cuda else ""
            why = eligible(n, k, str(w.dtype)) or (
                None if arch.startswith("gfx95") else f"arch {arch or 'cpu'} (gfx95x only)")
            layer._sfx_lmhead = None
            if why is not None:
                _state["stock"] += 1
                _log(f"lm_head N={n} K={k} stays bf16: {why}")
                return
            prepare, run, capture = _ops()
            st = prepare(w, max_m_env())
            r = load_oracle(st, run, lambda x: self.inner.apply(layer, x))
            _log(f"LOAD ORACLE PASS N={n} K={k}; vs bf16 (random x) " + " ".join(
                f"M={m}:rel={v['rel']:.3f},cos={v['cos']:.5f},top1={v['top1']:.2f}"
                for m, v in r.items()))
            if graph_on():
                capture(st)
                _log(f"GRAPHS N={n} K={k}: M=1..{st['max_m']} captured, bit-identical to eager")
            layer._sfx_lmhead = st
            _state["heads"] += 1

        def apply(self, layer, x, bias=None):
            st = getattr(layer, "_sfx_lmhead", None)
            rows = x.numel() // x.shape[-1] if x.numel() else 0
            if (st is None or bias is not None or x.dtype != torch.bfloat16
                    or not 1 <= rows <= st["max_m"]):
                return self.inner.apply(layer, x, bias)
            x2 = x.reshape(rows, st["k"])
            if "graphs" in st and not torch.cuda.is_current_stream_capturing():
                out = replay(st, x2)
            else:  # gate off, or inside a foreign capture (drafter graphs)
                out = _state["ops"][1](st, x2)
            if not _state["active"]:
                _state["active"] = True
                _log(f"ACTIVE: {_state['heads']} head(s) on MXFP4 (M<={st['max_m']}), "
                     f"{_state['stock']} bf16")
            return out.view(*x.shape[:-1], st["n"])

    class SuffixMxFp4ParallelLMHead(ParallelLMHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            qm = type(self.quant_method).__name__
            if qm in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
                self.quant_method = SuffixMxFp4LMHeadMethod(self.quant_method)
            else:
                _log(f"lm_head keeps {qm} (checkpoint-quantized)")

    return SuffixMxFp4ParallelLMHead


def _first_forward_check(module, args):
    h = _state.pop("hook", None)
    if h is not None:
        h.remove()
    if _state["checked"]:
        return
    _state["checked"] = True
    if _state["heads"] == 0:
        msg = (f"{MARKER} MXFP4 lm_head NOT ENGAGED with {GATE}=1: 0 heads converted "
               f"({_state['stock']} stayed bf16) — see the lines above")
        _log(msg)
        raise RuntimeError(msg)


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_MXFP4_LMHEAD=1.
    No HIP init here (API server processes load plugins too): the arch check
    runs at weight load, on the worker."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    import importlib.util

    import torch
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    if torch.version.hip is None:
        raise RuntimeError(f"{GATE}=1 needs a ROCm torch build (AITER MXFP4 GEMM)")
    if importlib.util.find_spec("aiter") is None:
        raise RuntimeError(f"{GATE}=1 but aiter is not installed")
    max_m_env()
    if "ParallelLMHead" in op_registry_oot:
        raise RuntimeError(f"{GATE}=1: ParallelLMHead already OOT-replaced by "
                           f"{op_registry_oot['ParallelLMHead']}")
    PluggableLayer.register_oot(_make_head_cls(), name="ParallelLMHead")
    _state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        _first_forward_check)
    _state["armed"] = True
    _log("armed: ParallelLMHead OOT-registered (M<=max_m -> MXFP4 screen + exact rescore)")
    return _state


def main(argv=None) -> int:
    """In-pod bench: fidelity + us/call vs the stock BF16 head (synthetic head)."""
    import torch
    prepare, run, capture = _ops()
    n, k = SHAPE
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.02
    st = prepare(w, max_m_env())
    capture(st)

    def us(fn, iters=50):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3 / iters

    for m in (1, 2, 4, 8, 16):
        if m > st["max_m"]:
            break
        x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
        f = fidelity(torch.nn.functional.linear(x, w), replay(st, x))
        print(f"{MARKER} bench N={n} K={k} M={m}: bf16 "
              f"{us(lambda: torch.nn.functional.linear(x, w)):.1f} us | mxfp4+rescore graph "
              f"{us(lambda: replay(st, x)):.1f} us eager {us(lambda: run(st, x)):.1f} us | "
              f"rel {f['rel']:.3f} top1 {f['top1']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
