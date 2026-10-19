# SPDX-License-Identifier: Apache-2.0
"""Serve DeepSeek-V4.1's FP8 (MXFP8: e4m3 + 32-block ue8m0) dense linears on
SM120 with the Helion skinny GEMM (helion_kernels/mxfp8_gemm.py) for decode
M <= max M; vLLM's own kernel everywhere else.

Knob: ``SUFFIX_HELION_MXFP8_GEMM`` = oxide (default: lane OFF, vLLM unchanged;
"oxide" is helion_kernels.backend's name for the baseline, here vLLM's kernel)
| auto (Helion per (shape, M bucket) where a lab-tuned config is shipped =
where it beat vLLM's SM120 kernel; vLLM elsewhere) | helion (every bucket <=
max M on Helion; a missing config is a startup error). ``SUFFIX_MXFP8_GEMM_MAX_M``
(default 192 = max-num-seqs 32 x DSpark 5+1, 1..192): larger M -> vLLM.
Entry: ``register()`` (no-op when the knob is oxide) — wired by sitecustomize.

What vLLM runs on SM120 (nightly 39983a5e7) and what replaces it:
  * plain MXFP8 linears (fused_wqa_wkv, wq_b, wo_b, indexer.wq_b, shared
    experts gate_up / down, DSpark main_proj): ModelOptLinearMethod ->
    select_linear_kernel -> init_mxfp8_linear_kernel -> FlashInferCutlass
    Mxfp8LinearKernel (dynamic MXFP8 quant + mm_mxfp8). We patch
    ``modelopt.select_linear_kernel`` (module global): where vLLM picked
    exactly that class, the layer gets ``SuffixMxfp8LinearKernel`` (its
    subclass: same weight processing, M > max M -> the parent's
    apply_weights). Its input_quant_key() is None: the layer takes bf16, and
    the fused q/kv-RMSNorm MXFP8 producer (query_quant.can_fuse_query_quant
    checks the exact consumer class) stays off — our A16 GEMM needs no quant.
    Excluded (stay on vLLM's kernel): prefixes containing ``.engram.`` (its
    wkv input arrives pre-quantized from the FP8 table prefetch).
  * wo_a (is_bmm): modelopt's process_weights_after_loading calls
    init_mxfp8_linear_kernel(bmm_batch_size=groups) -> DeepGemm BMM is
    SM100-only -> EmulationMxfp8LinearKernel (BF16 dequant at load, torch.bmm
    in deep_gemm_fp8_o_proj). We patch that global too: the layer gets
    ``SuffixMxfp8BmmKernel`` (Emulation subclass) which KEEPS vLLM's BF16
    weight (the M > max M path, unchanged) and holds the checkpoint's FP8
    weight + scales for ours; ``deep_gemm_fp8_o_proj`` (module globals of
    deepseek_v41/nvidia/ops/o_proj.py, which dsv41_o_proj calls) is
    wrapped: inverse RoPE (quantize=False, exactly vLLM's call) -> our op ->
    wo_b. One GEMM launch per local group (TP8: 1).

Compile: the per-call M decision runs inside ``torch.ops.suffix_mxfp8.linear``
(opaque to Dynamo, fake impl for shapes), like nvfp4_linear.

Split-K is TWO-STAGE and nothing else (K5): fp32 partials from torch's
stream-ordered allocator per call + a separate reduce launch, no cross-CTA
semaphore, no counter or scratch state bound at first use (helion_kernels/
mxfp8_gemm.py). ``SUFFIX_MXFP8_GEMM_SEMAPHORE=1`` (the unproven last-arriver
variant, lab and boot gate only) makes ``register()`` / ``prepare()`` raise.

Startup (fatal): cc 12.x; per layer ``validate_weight`` (e4m3 NaN codes, e8m0
scale > 246); per (N, K, bmm) a LAYER ORACLE on the real weights at every warm
M of the plan (compiles every bucket before graph capture): ours (NaN-poisoned
out / partials) vs the exact f64 product (A16: of the bf16 x; A8: of the
spec-quantized x) with mxfp8_gemm.check, a replay bit-identical, a CUDA-graph
CAPTURE on a side stream replayed with new data and compared bit-exactly with
eager (``capture_check``), and the error vs vLLM's kernel logged next to
vLLM's own error. First forward: no layer selected -> RuntimeError.
Markers: ``[suffix mxfp8-gemm] MXFP8-GEMM armed`` / ``PLAN NxK`` / ``LAYER
ORACLE PASS`` / ``MXFP8-GEMM ACTIVE`` / ``in-graph: Helion captured at M=m``.
Boot gate (evidence, sanitizer-ready): ``python -m suffix_hybrid.kernels.
mxfp8_linear capture_replay`` (boot gate ``mxfp8_capture_replay``).
"""
from __future__ import annotations

import os
import sys

ENV = "SUFFIX_HELION_MXFP8_GEMM"
MAX_M_ENV = "SUFFIX_MXFP8_GEMM_MAX_M"
MARKER = "[suffix mxfp8-gemm]"
EXCLUDE = (".engram.",)
OP_NS = "suffix_mxfp8"
_state = {"armed": False, "layers": {}, "instances": 0, "ours": 0, "stock": 0, "oracle": {},
          "captured": set(), "hook": None, "checked": False, "classes": None}
_LIB = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def mode() -> str:
    from suffix_hybrid.helion_kernels import backend
    return backend(ENV)


def max_m_env() -> int:
    from suffix_hybrid.helion_kernels.mxfp8_gemm import MAX_M
    raw = os.environ.get(MAX_M_ENV, "").strip()
    if not raw:
        return MAX_M
    if not raw.isdigit() or not 1 <= int(raw) <= MAX_M:
        raise ValueError(f"{MAX_M_ENV}={raw!r}: need an integer in 1..{MAX_M}")
    return int(raw)


# ---------------------------------------------------------------------------
# the op: M-dependent route with concrete shapes
# ---------------------------------------------------------------------------
def _op_impl(x, n: int, layer: str):
    """x bf16 [M, K] (linear) or [M, G, K] (wo_a, any strides with a
    contiguous last dim) -> bf16 [M, n] (n = G * N for wo_a)."""
    import torch
    kernel, ly = _state["layers"][layer]
    cfg = ly._sfx_mx
    m = x.shape[0]
    e = cfg["plan"][m] if 1 <= m <= cfg["max_m"] else None
    if e is None:
        return kernel._stock(ly, x)
    out = kernel._ours(ly, cfg, x, e)
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() \
            and m not in _state["captured"]:
        _state["captured"].add(m)
        _log(f"in-graph: Helion captured at M={m} ({layer}: {e[0]})")
    return out


def _op_fake(x, n: int, layer: str):
    import torch
    return x.new_empty((x.shape[0], n), dtype=torch.bfloat16)


def register_op():
    """torch.ops.suffix_mxfp8.linear(x, n, layer) (idempotent)."""
    global _LIB
    import torch
    if _LIB is None:
        lib = torch.library.Library(OP_NS, "FRAGMENT")
        lib.define("linear(Tensor x, int n, str layer) -> Tensor")
        for key in ("CUDA", "CPU"):
            lib.impl("linear", _op_impl, key)
        lib._register_fake("linear", _op_fake)
        _LIB = lib
    return getattr(torch.ops, OP_NS).linear


def register_layer(kernel, layer, name: str) -> str:
    """Graph-constant key for (kernel, layer): the prefix, #i on collision."""
    key, i = name or "mxfp8_linear", 1
    while key in _state["layers"] and _state["layers"][key][1] is not layer:
        key, i = f"{name}#{i}", i + 1
    _state["layers"][key] = (kernel, layer)
    return key


# ---------------------------------------------------------------------------
# kernel classes (vLLM imports: built at register())
# ---------------------------------------------------------------------------
def _make_classes():
    import torch
    from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    from suffix_hybrid.helion_kernels import mxfp8_gemm as hx

    op = register_op()

    class _Lane:
        """Shared: plan at load, our launch, the load oracle."""

        def _setup(self, layer, w, ws, groups: int):
            """cfg on layer._sfx_mx (None: vLLM serves every M). w e4m3 [G*N, K]."""
            bmm = groups > 1 or isinstance(self, EmulationMxfp8LinearKernel)
            n, k = w.shape[0] // groups, w.shape[1]
            layer._sfx_mx = None
            _state["instances"] += 1
            cap = max_m_env()
            plan = hx.prepare(n, k, cap, mode(), bmm) if k % 32 == 0 else [None] * (cap + 1)
            tag = f"{n}x{k}{f' wo_a x{groups}' if bmm else ''}"
            if (tag, "plan") not in _state["oracle"]:
                _state["oracle"][(tag, "plan")] = True
                _log(f"PLAN {tag}: {hx.describe(plan)} [split-K two-stage: torch-allocated fp32 "
                     "partials + reduce, no semaphores]")
            if not any(plan[1:]):
                _state["stock"] += 1
                return
            hx.validate_weight(w, ws, getattr(layer, "prefix", "") or tag)  # fatal on NaN codes
            cfg = layer._sfx_mx = dict(n=n, k=k, g=groups, bmm=bmm, w=w, ws=ws, plan=plan,
                                       max_m=cap)
            cfg["key"] = register_layer(self, layer, getattr(layer, "prefix", ""))
            if tag not in _state["oracle"]:
                _state["oracle"][tag] = self._oracle(layer, cfg)
            _state["ours"] += 1

        def _ours(self, layer, cfg, x, e, poison_nan=False):
            return hx.apply(x, cfg["w"], cfg["ws"], e, cfg["n"], cfg["g"], cfg["bmm"],
                            poison_nan=poison_nan)

        def _oracle(self, layer, cfg):
            n, k, g = cfg["n"], cfg["k"], cfg["g"]
            dev = cfg["w"].device
            gen = torch.Generator(device=dev).manual_seed(n * 7 + k)
            worst = [0.0, 0.0, 0.0]  # ours vs exact, ours vs vLLM, vLLM vs exact
            ms = [m for m in hx.warm_ms(cfg["max_m"]) if cfg["plan"][m]]
            tops = {min(b, cfg["max_m"]) for b in hx.BUCKETS}
            captured = 0
            for m in ms:
                e = cfg["plan"][m]
                shape = (m, g, k) if cfg["bmm"] else (m, k)
                x = torch.randn(*shape, generator=gen, device=dev).to(torch.bfloat16)
                ours = self._ours(layer, cfg, x, e, poison_nan=True)
                again = self._ours(layer, cfg, x, e, poison_nan=True)
                stock = self._stock(layer, x).float()
                if not hx.bits_equal(ours, again):
                    raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL N={n} K={k} M={m} {e[0]}: "
                                       f"replay differs — refusing to serve with {ENV}={mode()}")
                for i in range(g):
                    xi = x[:, i, :] if cfg["bmm"] else x
                    wi, si = cfg["w"][i * n:(i + 1) * n], cfg["ws"][i * n:(i + 1) * n]
                    if e[0] == hx.A8:
                        q, s = hx.quant_spec(xi)
                        ref = hx.exact(hx.dequant(q, s, torch.float32), wi, si)
                    else:
                        ref = hx.exact(xi, wi, si)
                    oi = ours[:, i * n:(i + 1) * n]
                    ok, msg = hx.check(oi, ref)
                    if not ok:
                        raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL N={n} K={k} M={m} "
                                           f"{e[0]}: vs exact: {msg} — refusing to serve with "
                                           f"{ENV}={mode()}")
                    bx = hx.exact(xi, wi, si).float()  # the bf16-x product: both paths' target
                    si_ = stock[:, i * n:(i + 1) * n]
                    nrm = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))
                    worst = [max(worst[0], nrm(oi.float(), bx)), max(worst[1], nrm(oi.float(), si_)),
                             max(worst[2], nrm(si_, bx))]
                if m in tops:
                    # K5 capture/replay: the same call in a CUDA graph on a side stream, replayed
                    # with new data, bit-identical to eager (compiled above, outside capture);
                    # one M per bucket (the bucket decides kernel and config)
                    x2 = torch.randn(*shape, generator=gen, device=dev).to(torch.bfloat16)
                    captured += hx.capture_check(
                        lambda xx: self._ours(layer, cfg, xx, e, poison_nan=True), [x, x2],
                        what=f"N={n} K={k} M={m} {e[0]}")
            torch.cuda.synchronize(dev)
            _log(f"LAYER ORACLE PASS N={n} K={k}{' wo_a' if cfg['bmm'] else ''} M {ms}: "
                 f"rel vs exact(bf16 x) {worst[0]:.2e} (vLLM's own {worst[2]:.2e}), "
                 f"vs vLLM {worst[1]:.2e}; {captured} capture/replays bit-exact")
            return worst

    class SuffixMxfp8LinearKernel(_Lane, FlashInferCutlassMxfp8LinearKernel):
        """FlashInfer CUTLASS MXFP8 linear + the Helion decode GEMM for M <= max M."""

        def input_quant_key(self):
            return None  # bf16 in: no fused MXFP8 producer feeds this layer

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            n, k = w.shape
            ws = layer.weight_scale.data[:n, :k // 32].contiguous()  # before the swizzle
            super().process_weights_after_loading(layer)
            self._setup(layer, layer.weight.data, ws, 1)

        def _stock(self, layer, x2d):
            return super().apply_weights(layer, x2d)

        def apply_weights(self, layer, x, bias=None):
            cfg = getattr(layer, "_sfx_mx", None)
            if cfg is None or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
                return super().apply_weights(layer, x, bias)
            out = op(x.reshape(-1, x.shape[-1]), cfg["n"], cfg["key"])
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], cfg["n"])

    class SuffixMxfp8BmmKernel(_Lane, EmulationMxfp8LinearKernel):
        """wo_a: vLLM's BF16-at-load weight for M > max M, the checkpoint's FP8
        weight on the Helion GEMM below (served from the o_proj wrapper)."""

        def process_weights_after_loading(self, layer):
            w = layer.weight.data.contiguous()
            ws = layer.weight_scale.data[:w.shape[0], :w.shape[1] // 32].contiguous()
            super().process_weights_after_loading(layer)  # vLLM's BF16 dequant, unchanged
            self._setup(layer, w, ws, int(self.config.bmm_batch_size or 1))

        def _stock(self, layer, x):
            """vLLM's deep_gemm_fp8_o_proj bf16 branch on [M, G, K]."""
            g, m = layer._sfx_mx["g"], x.shape[0]
            wb = layer.weight.view(g, layer._sfx_mx["n"], -1)
            z = torch.empty(m, g, wb.shape[1], dtype=torch.bfloat16, device=x.device)
            torch.bmm(x.transpose(0, 1), wb.transpose(1, 2), out=z.transpose(0, 1))
            return z.view(m, -1)

    return SuffixMxfp8LinearKernel, SuffixMxfp8BmmKernel, FlashInferCutlassMxfp8LinearKernel, \
        EmulationMxfp8LinearKernel, op


def excluded(prefix: str) -> bool:
    return any(s in (prefix or "") for s in EXCLUDE)


def _wrap_select(orig, ours, stock_cls):
    def select_linear_kernel(spec, layer, rt, weight_shape=None):
        k = orig(spec, layer, rt, weight_shape=weight_shape)
        if type(k) is stock_cls and not excluded(getattr(layer, "prefix", "")):
            return ours(k.config)
        return k
    select_linear_kernel._sfx_mx = True
    return select_linear_kernel


def _wrap_init(orig, ours, stock_cls):
    def init_mxfp8_linear_kernel(*, weight_shape, bmm_batch_size=None):
        k = orig(weight_shape=weight_shape, bmm_batch_size=bmm_batch_size)
        if bmm_batch_size is not None and type(k) is stock_cls:
            return ours(k.config)
        return k
    init_mxfp8_linear_kernel._sfx_mx = True
    return init_mxfp8_linear_kernel


def _wrap_o_proj(orig, op):
    def deep_gemm_fp8_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, **kw):
        cfg = getattr(wo_a, "_sfx_mx", None)
        if cfg is None or type(o).__name__ == "QuantizedActivation":
            return orig(o, positions, cos_sin_cache, wo_a, wo_b, **kw)
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )
        x, _ = fused_inv_rope_fp8_quant(
            o, positions, cos_sin_cache, n_groups=kw["n_groups"],
            heads_per_group=kw["heads_per_group"], nope_dim=kw["nope_dim"],
            rope_dim=kw["rope_dim"], quant_group_size=kw["einsum_recipe"][2],
            tma_aligned_scales=kw["tma_aligned_scales"], quantize=False)
        return wo_b(op(x, cfg["g"] * cfg["n"], cfg["key"]))
    deep_gemm_fp8_o_proj._sfx_mx = True
    return deep_gemm_fp8_o_proj


# dsv41_o_proj (the SM120 DeepseekV4FlashInferSM120Attention._o_proj) calls
# deep_gemm_fp8_o_proj through this module's global.
O_PROJ_MODULES = ("vllm.models.deepseek_v41.nvidia.ops.o_proj",)


def _first_forward_check(module, args):
    h = _state.pop("hook", None)
    if h is not None:
        h.remove()
    if _state["checked"]:
        return
    _state["checked"] = True
    if _state["instances"] == 0:
        err = (f"{MARKER} MXFP8-GEMM NOT SELECTED with {ENV}={mode()}: no MXFP8 linear went "
               "through FlashInfer CUTLASS / the wo_a emulation bmm on this model")
        _log(err)
        raise RuntimeError(err)
    _log(summary())


def summary() -> str:
    return (f"MXFP8-GEMM ACTIVE: {_state['ours']} layers on the Helion GEMM (M <= "
            f"{max_m_env()}, per-bucket plans, two-stage split-K, capture/replay-checked at load), "
            f"{_state['stock']} on vLLM's kernel only, "
            f"{len([k for k in _state['oracle'] if isinstance(k, str)])} shapes oracle-checked")


def register():
    """vllm.general_plugins / sitecustomize entry. No-op unless the knob is
    auto | helion. Fatal on anything that would leave it half-armed."""
    hb = mode()
    if hb == "oxide" or _state["armed"]:
        return _state if _state["armed"] else None
    import importlib

    from suffix_hybrid.helion_kernels import mxfp8_gemm as _hx
    _hx.refuse_semaphore("register")  # K5: the unproven semaphore variant never serves

    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError(f"{ENV}={hb}: the MXFP8 Helion lane is for SM12x (cc 12.x) only")
    max_m_env()
    from suffix_hybrid.helion_kernels import ensure_vendor
    ensure_vendor()
    import helion  # noqa: F401 — fail now, not at the first layer
    ours, bmm, stock_cls, emu_cls, op = _make_classes()
    mo = importlib.import_module("vllm.model_executor.layers.quantization.modelopt")
    if not getattr(mo.select_linear_kernel, "_sfx_mx", False):
        mo.select_linear_kernel = _wrap_select(mo.select_linear_kernel, ours, stock_cls)
    if not getattr(mo.init_mxfp8_linear_kernel, "_sfx_mx", False):
        mo.init_mxfp8_linear_kernel = _wrap_init(mo.init_mxfp8_linear_kernel, bmm, emu_cls)
    wrapped = []
    for name in O_PROJ_MODULES:
        mod = importlib.import_module(name)
        fn = getattr(mod, "deep_gemm_fp8_o_proj", None)
        if fn is None:
            raise RuntimeError(f"{ENV}={hb}: {name}.deep_gemm_fp8_o_proj is gone (vLLM moved "
                               "the o_proj; refusing a half-armed wo_a lane)")
        if not getattr(fn, "_sfx_mx", False):
            mod.deep_gemm_fp8_o_proj = _wrap_o_proj(fn, op)
        wrapped.append(name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1])
    _state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        _first_forward_check)
    _state["armed"] = True
    _log(f"MXFP8-GEMM armed: {ENV}={hb}, M <= {max_m_env()} on Helion where planned; "
         f"patched modelopt.select_linear_kernel / init_mxfp8_linear_kernel (bmm) and "
         f"deep_gemm_fp8_o_proj in {', '.join(wrapped)}; excluded prefixes {EXCLUDE}")
    return _state


def main(argv=None) -> int:
    """python -m suffix_hybrid.kernels.mxfp8_linear capture_replay [args]: the
    mxfp8_capture_replay boot gate (suffix_hybrid/kernels/dsv41_dense_capture_replay.py)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["capture_replay"]:
        from suffix_hybrid.kernels import dsv41_dense_capture_replay as gate
        return gate.main(argv[1:])
    print("usage: python -m suffix_hybrid.kernels.mxfp8_linear capture_replay "
          "[--quick|--full] [--variant two_stage|semaphore] [--replays N] [--only SUBSTR] "
          "[--no-capture] [--streams N]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
