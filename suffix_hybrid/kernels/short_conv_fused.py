# SPDX-License-Identifier: Apache-2.0
"""Fused LFM2 short-conv decode (Lfm2 / Lfm2Moe ``ShortConv``, vLLM
short_conv.py): one Triton launch replaces the decode chain between the two
projections — ``(B_d * x_d).contiguous()``, ``causal_conv1d_update``,
``C_d * Bx``, ``torch.vstack`` (4-5 launches -> 1 per conv layer; the in /
out projections are the stock calls).

Gate: ``SUFFIX_SHORTCONV_FUSED=1`` (default OFF). Entry: ``register()``, the
``vllm.general_plugins`` entry point ``suffix_short_conv``. Seam:
``PluggableLayer.register_oot(name="ShortConv")`` — the subclass overrides
``forward_cuda`` only; every batch that is not a pure non-spec decode batch
(prefill, mixed, spec verify, profile run) runs the stock method unchanged.

Numerics: bit-identical to the stock chain. B*x and C*conv are bf16 products
of bf16 operands (rounded once, like torch's bf16 mul); the conv accumulates
bf16-rounded tap products in fp32 in tap order (0, 1, 2) from 0.0 and rounds
the sum to bf16, exactly like ``_causal_conv1d_update_kernel`` (whose
``acc += matrix_x * matrix_w`` multiplies two bf16 tensors). The state roll
(shift left by one, append B*x) matches it for every state_len >= width - 1.
The load-time oracle demands bitwise equality of y AND the updated state
against the stock chain on random inputs (incl. null-block padding rows), so
a drift in either kernel refuses to serve instead of serving different math.
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_SHORTCONV_FUSED"
LAYER_NAME = "ShortConv"
MARKER = "[suffix short-conv]"
BLOCK_N = 256

_state = {"armed": False, "layers": 0, "oracle": None, "launch_logged": False}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


_kernel = None


def _get_kernel():
    """Build the Triton kernel lazily (Triton is imported only when armed)."""
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _shortconv_decode(bcx_ptr, stride_bcx, state_ptr, s_seq, s_dim, s_tok, w_ptr, s_wd, s_ww,
                          idx_ptr, s_idx, y_ptr, stride_y, null_block_id,
                          D: tl.constexpr, STATE_LEN: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = f < D
        base = bcx_ptr + row.to(tl.int64) * stride_bcx
        b = tl.load(base + f, mask=m, other=0.0)
        c = tl.load(base + D + f, mask=m, other=0.0)
        x = tl.load(base + 2 * D + f, mask=m, other=0.0)
        bx = b * x  # bf16 * bf16 -> bf16 (one rounding), = torch B_d * x_d
        line = tl.load(idx_ptr + row * s_idx).to(tl.int64)
        st = state_ptr + line * s_seq + f * s_dim
        # width 3 (LFM2 conv_L_cache): taps = state[0], state[1], new input
        col0 = tl.load(st, mask=m, other=0.0)
        col1 = tl.load(st + s_tok, mask=m, other=0.0)
        w0 = tl.load(w_ptr + f * s_wd, mask=m, other=0.0)
        w1 = tl.load(w_ptr + f * s_wd + s_ww, mask=m, other=0.0)
        w2 = tl.load(w_ptr + f * s_wd + 2 * s_ww, mask=m, other=0.0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        acc += col0 * w0
        acc += col1 * w1
        acc += bx * w2
        conv = acc.to(tl.bfloat16)
        y = c * conv
        tl.store(y_ptr + row.to(tl.int64) * stride_y + f, y, mask=m)
        if line != null_block_id:
            # roll: [s0 .. s_{L-1}] -> [s1 .. s_{L-1}, bx]
            for t in tl.static_range(STATE_LEN - 1):
                v = tl.load(st + (t + 1) * s_tok, mask=m, other=0.0)
                tl.store(st + t * s_tok, v, mask=m)
            tl.store(st + (STATE_LEN - 1) * s_tok, bx, mask=m)

    _kernel = _shortconv_decode
    return _kernel


def fused_decode(bcx, conv_state, conv_w, state_idx, y, null_block_id: int):
    """bcx bf16 [n, 3D] (in_proj output rows); conv_state [lines, D, L] (any
    strides, the dim-first view); conv_w [D, 3]; state_idx int [n] or
    [n, k] (column 0 used, like causal_conv1d_update); y bf16 [n, D] out."""
    import triton
    n, three_d = bcx.shape
    d = three_d // 3
    if conv_w.shape != (d, 3):
        raise ValueError(f"{MARKER} conv weight {tuple(conv_w.shape)} != ({d}, 3)")
    k = _get_kernel()
    k[(n, triton.cdiv(d, BLOCK_N))](
        bcx, bcx.stride(0), conv_state, conv_state.stride(0), conv_state.stride(1),
        conv_state.stride(2), conv_w, conv_w.stride(0), conv_w.stride(1),
        state_idx, state_idx.stride(0), y, y.stride(0), null_block_id,
        D=d, STATE_LEN=conv_state.shape[2], BLOCK=BLOCK_N, num_warps=4)
    return y


def stock_decode(bcx, conv_state, conv_w, state_idx, null_block_id: int):
    """vLLM's decode chain verbatim (short_conv.py forward_cuda, non-spec
    decode branch, no bias): returns y [n, D]; mutates conv_state."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    b, c, x = bcx.chunk(3, dim=-1)
    bx = (b * x).contiguous()
    out = causal_conv1d_update(bx, conv_state, conv_w, None, activation=None,
                               conv_state_indices=state_idx, null_block_id=null_block_id)
    return c * out


def oracle(device=None, d=2048, seeds=(0, 1, 2), ns=(1, 3, 8, 33, 64)) -> str:
    """Fused vs stock on random inputs: y AND the updated state bitwise equal
    for every n (null-block padding rows included; their y is don't-care, so
    only real rows are compared). Fatal on any difference."""
    import torch
    dev = device or torch.device("cuda", torch.cuda.current_device())
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    checked = 0
    lines = max(ns) + 2  # every row a distinct non-null cache line (+ the null block 0)
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for n in ns:
            bcx = (torch.randn(n, 3 * d, generator=g) * 2).to(dev, torch.bfloat16)
            w = (torch.randn(d, 3, generator=g) * 0.5).to(dev, torch.bfloat16)
            # dim-first view of a (lines, L, D) buffer = vLLM's default layout
            st0 = (torch.randn(lines, 2, d, generator=g)).to(dev, torch.bfloat16).transpose(1, 2)
            perm = torch.randperm(lines - 1, generator=g)[:n] + 1  # 0 = null block
            idx = perm.to(torch.int32)
            if n >= 3:
                idx[n // 2] = NULL_BLOCK_ID  # a cudagraph padding row
            idx = idx.to(dev)
            real = (idx != NULL_BLOCK_ID)
            s_ref, s_ours = st0.clone(), st0.clone()
            y_ref = stock_decode(bcx, s_ref, w, idx, NULL_BLOCK_ID)
            y = torch.empty(n, d, dtype=torch.bfloat16, device=dev)
            fused_decode(bcx, s_ours, w, idx, y, NULL_BLOCK_ID)
            torch.cuda.synchronize(dev)
            if not torch.equal(y[real], y_ref[real]):
                diff = (y[real].float() - y_ref[real].float()).abs().max().item()
                raise RuntimeError(f"{MARKER} ORACLE FAIL y seed={seed} n={n} max|d|={diff}")
            if not torch.equal(s_ours, s_ref):
                raise RuntimeError(f"{MARKER} ORACLE FAIL state seed={seed} n={n}")
            checked += 1
    return f"{MARKER} ORACLE PASS ({checked} cases, y + state bitwise == stock chain)"


def _make_layer_cls():
    import inspect

    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    params = list(inspect.signature(ShortConv.forward_cuda).parameters)
    if params != ["self", "hidden_states", "output"]:
        raise RuntimeError(f"{MARKER} anchor drift: ShortConv.forward_cuda{params}")

    class SuffixShortConv(ShortConv):
        """ShortConv + one fused launch on the pure non-spec decode step."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.L_cache != 3 or self.bias:
                raise ValueError(f"{GATE}=1 but {self.prefix}: L_cache={self.L_cache} "
                                 f"bias={self.bias} (fused kernel = width 3, no bias)")
            _state["layers"] += 1
            if _state["layers"] == 1:
                _log(f"ACTIVE: ShortConv -> SuffixShortConv (first layer {self.prefix}, "
                     f"dim={self.conv_dim})")

        def forward_cuda(self, hidden_states, output):
            md = get_forward_context().attn_metadata
            md = md.get(self.prefix) if isinstance(md, dict) else None
            if md is None:
                if _state["oracle"] is None:  # profile run, before any capture
                    _state["oracle"] = oracle(hidden_states.device, d=self.conv_dim)
                    _log(_state["oracle"])
                return super().forward_cuda(hidden_states, output)
            if md.num_prefills or md.num_accepted_tokens is not None or not md.num_decode_tokens:
                return super().forward_cuda(hidden_states, output)
            n = md.num_decode_tokens
            bcx, _ = self.in_proj(hidden_states[:n])
            conv_state = (self.kv_cache[0] if is_conv_state_dim_first()
                          else self.kv_cache[0].transpose(-1, -2))
            w = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            y = torch.empty(n, self.conv_dim, dtype=bcx.dtype, device=bcx.device)
            fused_decode(bcx, conv_state, w, md.state_indices_tensor_d, y, NULL_BLOCK_ID)
            output[:n], _ = self.out_proj(y)
            if not _state["launch_logged"]:
                _state["launch_logged"] = True
                _log(f"decode-launch: first fused decode batch n={n} ({self.prefix})")

    return SuffixShortConv


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_SHORTCONV_FUSED=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    if LAYER_NAME not in op_registry_oot:
        PluggableLayer.register_oot(_make_layer_cls(), name=LAYER_NAME)
    _state["armed"] = True
    _log(f"armed: {LAYER_NAME} OOT-registered")
    return _state


def bench(d=2048, ns=(1, 8, 32, 64), layers=30, iters=50) -> dict:
    """GPU us per decode step for `layers` conv layers' middle chain (stock vs
    fused), each captured in ONE CUDA graph like vLLM's FULL decode graph."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    res = {}
    for n in ns:
        bcx = [torch.randn(n, 3 * d, device=dev).bfloat16() for _ in range(layers)]
        w = [torch.randn(d, 3, device=dev).bfloat16() for _ in range(layers)]
        st = [torch.randn(n + 1, 2, d, device=dev).bfloat16().transpose(1, 2) for _ in range(layers)]
        idx = torch.arange(1, n + 1, dtype=torch.int32, device=dev)
        y = torch.empty(n, d, dtype=torch.bfloat16, device=dev)

        def stock():
            for i in range(layers):
                torch.vstack([stock_decode(bcx[i], st[i], w[i], idx, NULL_BLOCK_ID)])

        def fused():
            for i in range(layers):
                fused_decode(bcx[i], st[i], w[i], idx, y, NULL_BLOCK_ID)

        out = {}
        for name, fn in (("stock", stock), ("fused", fused)):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    fn()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                g.replay()
            t1.record()
            torch.cuda.synchronize()
            out[name] = t0.elapsed_time(t1) * 1e3 / iters
        res[n] = out
        _log(f"bench n={n} {layers} layers: stock {out['stock']:.1f} us, fused {out['fused']:.1f} us "
             f"(x{out['fused'] / out['stock']:.2f})")
    return res


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "both"):
            _log(oracle())
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:  # noqa: BLE001
        _log(f"{mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
