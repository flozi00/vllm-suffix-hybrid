# SPDX-License-Identifier: Apache-2.0
"""Serve our NVFP4 W4A4 decode GEMM (kernels-oxide/nvfp4_gemm, sm_120a SASS
from ptxas 13.0) inside vLLM 0.30.0 through its own linear-kernel selection.

Gate: ``SUFFIX_NVFP4_GEMM=1`` (default OFF). Entry: ``register()``, the
``vllm.general_plugins`` entry point ``suffix_nvfp4_gemm`` (shipped in
suffix_qwen_gdn_ep-1.0.dist-info). Gate off -> returns immediately.

Integration point (no fork, no _custom_ops patching):
  * vllm/model_executor/kernels/linear/__init__.py:1189
    ``register_linear_kernel(cls, PlatformEnum.CUDA, "nvfp4")`` — vLLM's
    public registry for extra linear kernels. It APPENDS to
    ``_POSSIBLE_NVFP4_KERNELS`` (:551-567), which ``init_nvfp4_linear_kernel``
    (:1069-1186) walks in order, skipping names listed in
    ``VLLM_DISABLED_KERNELS`` (:1154, envs.py:1190). register() therefore also
    adds every NVFP4 candidate listed before ours to VLLM_DISABLED_KERNELS
    (plugins load before engine config; envs are uncached until service init,
    envs.py:2179-2215) so selection lands on ``SuffixNvFp4LinearKernel``.
  * ``SuffixNvFp4LinearKernel`` subclasses vLLM's own
    ``FlashInferCutlassNvFp4LinearKernel`` (nvfp4/flashinfer.py:170-251): same
    weight processing (swizzled scales, padding), and every call that is not a
    decode-sized GEMM on an eligible shape (M > the layer's max M, padded K,
    non-bf16, ...) is routed to that parent's ``apply_weights`` — the stock
    FlashInfer CUTLASS path, i.e. exactly what vLLM would have run. Routing
    is by shape, decided per layer at load time and logged; never a silent
    substitute. Per-layer max M = min(SUFFIX_NVFP4_GEMM_MAX_M (default 64,
    16..64), ROUTE[(N, K)]); ROUTE is ``DEFAULT_ROUTE`` overridden by
    ``SUFFIX_NVFP4_GEMM_ROUTE="NxK:maxM,..."`` (maxM 0 = always FlashInfer),
    e.g. from the in-pod bench's suggested route line.
  * One launch per linear by default (nvfp4_gemm.plan: activation quant in
    the GEMM prologue, split-K reduce by the last CTA per column tile);
    ``SUFFIX_NVFP4_GEMM_FUSED=0`` restores the old quant + GEMM + reduce
    launches exactly (``=reduce``: fuse only the reduce). Read at load.
  * Pre-quantized inputs (vLLM's fused SiLU*mul / RMSNorm + NVFP4 quant,
    ``input_quant_key() == kNvfp4Dynamic`` inherited) run our GEMM on vLLM's
    packed activation + swizzled scales (kernel asf_mode 1).

Startup (fatal when the gate is on): oxide manifest has ``nvfp4_gemm``,
driver loads the cubin (``oxide_kernels.ensure_loaded``), SM family 12, and a
per-(N, K) LAYER ORACLE on the real checkpoint weights at M in {1, 5, 16,
17, 40, 64} (<= the layer's max M): ours vs the parent FlashInfer path AND vs
the f64 exact-dequant reference, both input routes, and the single-launch
path bit-identical to the old 3-launch one (outputs, aq/asf readback,
ticket counters reset), before any CUDA graph is captured.
Markers: ``[suffix nvfp4-gemm] NVFP4-GEMM armed`` / ``LAYER ORACLE PASS`` /
``NVFP4-GEMM ACTIVE`` / ``NVFP4-GEMM in-graph: N layers captured on ours at M=m``
(one per captured M; the proof the kernel is in the CUDA graphs).

Compile: the per-call M decision runs inside ``torch.ops.suffix_nvfp4.linear``
(opaque to Dynamo, fake impl for shapes), never as a Python branch in the
traced forward — vLLM traces once at M = max_num_batched_tokens with guards
skipped, which baked FlashInfer into every decode graph before 2026-10-07.
Prequantized activations cross the op as (packed data, swizzled scales).
The bf16 route's activation quant is inside the op, so vLLM's
norm/act + NVFP4-quant fusion passes see no scaled_fp4_quant for these
layers (decode quantizes in our GEMM prologue anyway).

Hot path: no allocation besides the output tensor (like every vLLM linear),
no host sync (alpha / global scale cached as floats at load), launches on
torch's current stream -> CUDA-graph capturable.
"""
from __future__ import annotations

import os
import sys

GATE = "SUFFIX_NVFP4_GEMM"
MARKER = "[suffix nvfp4-gemm]"
FAMILY = "nvfp4_gemm"
MAX_M_ENV = "SUFFIX_NVFP4_GEMM_MAX_M"
ROUTE_ENV = "SUFFIX_NVFP4_GEMM_ROUTE"
MAX_M = 64  # kernel limit (4 m16 tiles); default of MAX_M_ENV
ORACLE_REL = 2e-2
ORACLE_MS = (1, 5, 16, 17, 40, 64)
# (N, K) -> largest M served by our kernel (0 = never; larger M -> FlashInfer).
# Filled from the in-pod bench (python -m suffix_hybrid.kernels.nvfp4_gemm
# bench prints the suggested ROUTE); empty = every eligible shape up to MAX_M.
DEFAULT_ROUTE: dict = {}
_state = {"armed": False, "oracle": {}, "layers_ours": 0, "layers_stock": 0,
          "ws": {}, "instances": 0, "shapes": {}, "checked": False, "hook": None,
          "layers": {}, "captured": {}, "graph_logged": set()}
OP_NS = "suffix_nvfp4"
_LIB = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# M-dependent dispatch as an opaque torch custom op.
# vLLM compiles the model forward once with a symbolic batch hinted at
# max_num_batched_tokens and SKIPS all guards (compilation/wrapper.py
# skip_all_guards_unsafe): a Python `if rows <= max_m` in apply_weights is
# evaluated once at trace time (8192 -> FlashInfer) and baked into every
# CUDA graph. Like vLLM's own M-dependent kernels (torch.ops.vllm.*), the
# route decision lives inside an op Dynamo cannot see into; the op body runs
# with concrete shapes (eager, piecewise and at CUDA-graph capture).
# ---------------------------------------------------------------------------
def register_layer(kernel, layer, name: str) -> str:
    """Make (kernel, layer) reachable from the op by a graph-constant string
    (vLLM's layer_name convention). Deterministic per process (prefix, then
    #i on collision, e.g. a drafter reusing target prefixes); re-registering
    the same layer keeps its key."""
    key = getattr(layer, "_sfx_nvfp4_key", None)
    if key is None:
        key, i = name or "nvfp4_linear", 1
        while key in _state["layers"]:
            key, i = f"{name}#{i}", i + 1
        layer._sfx_nvfp4_key = key
    _state["layers"][key] = (kernel, layer)
    return key


def _capturing() -> bool:
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _note_capture(key: str, m: int) -> None:
    """One line per captured M once every layer that routes M to us has been
    captured on our kernel: the in-graph proof (the eager ACTIVE line is not)."""
    seen = _state["captured"].setdefault(m, set())
    seen.add(key)
    want = sum(1 for _, ly in _state["layers"].values()
               if ly._sfx_nvfp4 is not None and m <= ly._sfx_nvfp4["max_m"])
    if len(seen) >= want and m not in _state["graph_logged"]:
        _state["graph_logged"].add(m)
        _log(f"NVFP4-GEMM in-graph: {len(seen)} layers captured on ours at M={m}")


def _op_impl(x, xsf, n: int, layer: str):
    """x: bf16 [M, K] (xsf None) or vLLM-prequantized packed [M, K/2] u8 with
    its swizzled scales xsf. -> bf16 [M, n]. Our kernel for 1 <= M <= the
    layer's max M, else the parent's (stock FlashInfer) apply_weights."""
    kernel, ly = _state["layers"][layer]
    cfg = ly._sfx_nvfp4
    m = x.shape[0]
    if not 1 <= m <= cfg["max_m"]:
        return kernel._stock(ly, x, xsf)
    out = kernel._ours(ly, cfg, x if xsf is None else None,
                       None if xsf is None else (x, xsf))
    if not _state.get("active_logged"):
        _state["active_logged"] = True
        _log(summary())
    if _capturing():
        _note_capture(layer, m)
    return out


def _op_fake(x, xsf, n: int, layer: str):
    import torch
    return x.new_empty((x.shape[0], n), dtype=torch.bfloat16)


def register_op():
    """torch.ops.suffix_nvfp4.linear(x, xsf, n, layer) (idempotent)."""
    global _LIB
    import torch
    if _LIB is None:
        lib = torch.library.Library(OP_NS, "FRAGMENT")
        lib.define("linear(Tensor x, Tensor? xsf, int n, str layer) -> Tensor")
        for key in ("CUDA", "CPU"):
            lib.impl("linear", _op_impl, key)
        lib._register_fake("linear", _op_fake)
        _LIB = lib
    return getattr(torch.ops, OP_NS).linear


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def max_m_env() -> int:
    """SUFFIX_NVFP4_GEMM_MAX_M (default 64): the largest M our kernel serves."""
    raw = os.environ.get(MAX_M_ENV, "").strip()
    if not raw:
        return MAX_M
    if not raw.isdigit() or not 16 <= int(raw) <= MAX_M:
        raise ValueError(f"{MAX_M_ENV}={raw!r}: need an integer in 16..{MAX_M}")
    return int(raw)


def parse_route(spec: str) -> dict:
    """"NxK:maxM,..." (spaces allowed, e.g. "2560 x 3072:16, 336x10240:0")
    -> {(N, K): maxM}. Loud on malformed entries."""
    out = {}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        try:
            shape, mm = item.split(":")
            n, k = (int(v) for v in shape.lower().replace(" ", "").split("x"))
            out[(n, k)] = int(mm)
        except ValueError:
            raise ValueError(f"{ROUTE_ENV}: bad entry {item!r} (want NxK:maxM)") from None
        if not 0 <= out[(n, k)] <= MAX_M:
            raise ValueError(f"{ROUTE_ENV}: {item!r}: maxM must be in 0..{MAX_M}")
    return out


def route_table() -> dict:
    return {**DEFAULT_ROUTE, **parse_route(os.environ.get(ROUTE_ENV, ""))}


def layer_max_m(n: int, k: int, max_m: int, route: dict) -> int:
    return min(max_m, route.get((n, k), max_m))


def eligible(n: int, k: int, rows: int, pad_bytes: int) -> str | None:
    """Why a layer with N = output_size_per_partition, K, and `rows` weight
    rows (FlashInfer pads N to % 32 with zero rows; we read the first N)
    cannot use our kernel, or None."""
    if pad_bytes or rows < n:
        return "padded weight (K)" if pad_bytes else f"weight rows {rows} < N={n}"
    if n % 16 or k % 64:
        return f"N={n} % 16 or K={k} % 64"
    return None


def _workspace(dev, n: int, k: int, max_m: int = MAX_M):
    """Per-device scratch, grown at LOAD time only (never inside forward):
    aq [16*tiles*K/2] u8, asf [16*tiles*K/16] u8 (tiles = ceil(max_m/16)),
    split-K partials f32 (worst M bucket of `plan`), split-K ticket
    counters int32 [ceil(N/32)] (zeroed here, left zeroed by every launch)."""
    import torch
    from suffix_hybrid.kernels.nvfp4_gemm import partial_elems, tiles_for
    ws = _state["ws"].get(dev)
    rows = 16 * tiles_for(max_m)
    need = (rows * k // 2, rows * k // 16, max(1, partial_elems(n, k, max_m)), -(-n // 32))
    if ws is None or any(have.numel() < want for have, want in zip(ws, need)):
        old = ws or (None,) * 4
        sizes = [max(want, o.numel() if o is not None else 0)
                 for want, o in zip(need, old)]
        ws = (torch.empty(sizes[0], dtype=torch.uint8, device=dev),
              torch.empty(sizes[1], dtype=torch.uint8, device=dev),
              torch.empty(sizes[2], dtype=torch.float32, device=dev),
              torch.zeros(sizes[3], dtype=torch.int32, device=dev))
        _state["ws"][dev] = ws
    return ws


def _make_kernel_cls():
    import torch
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCutlassNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        as_quantized_activation,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        nvfp4_weight_padding_bytes,
    )
    from suffix_hybrid import oxide_kernels
    import numpy as np

    from suffix_hybrid.kernels.nvfp4_gemm import (
        PARAMS,
        dequant_torch,
        exact_ref,
        fused_vs_old,
        oracle_ok,
        plan,
        plan_str,
        quantize,
        unswizzle_sf,
    )

    native = oxide_kernels.native()
    op = register_op()

    class SuffixNvFp4LinearKernel(FlashInferCutlassNvFp4LinearKernel):
        """FlashInfer-CUTLASS NVFP4 linear + our sm_120a decode GEMM for M <= max M."""

        def __init__(self, config):
            super().__init__(config)
            _state["instances"] += 1  # selection witness (checked at first forward)

        @classmethod
        def is_supported(cls, compute_capability=None):
            ok, why = super().is_supported(compute_capability)
            if not ok:
                return ok, why
            if not gate_on():
                return False, f"{GATE} != 1"
            if torch.cuda.get_device_capability()[0] != 12:
                return False, "sm_120a SASS needs cc 12.x"
            return True, None

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            n = layer.output_size_per_partition
            k = layer.weight.shape[1] * 2
            why = eligible(n, k, layer.weight.shape[0], nvfp4_weight_padding_bytes(layer))
            max_m = layer_max_m(n, k, max_m_env(), route_table())
            if why is None and max_m < 1:
                why = f"{ROUTE_ENV} routes N={n} K={k} to FlashInfer"
            layer._sfx_nvfp4 = None
            route = "flashinfer" if why is not None else f"ours<=M{max_m}"
            key = f"{n}x{k}:{route}"
            _state["shapes"][key] = _state["shapes"].get(key, 0) + 1
            if why is not None:
                _state["layers_stock"] += 1
                _log(f"layer N={n} K={k} stays on FlashInfer CUTLASS: {why}")
                return
            dev = layer.weight.device
            oxide_kernels.ensure_loaded(FAMILY, dev.index, params=PARAMS)
            _workspace(dev, n, k, max_m)
            # One device sync per layer at LOAD time; never in forward.
            # (splits, fused flags) per M and route, fixed at load (graph-safe;
            # SUFFIX_NVFP4_GEMM_FUSED read here, not in forward)
            launch = {q: [(0, 0)] + [(pl["splits"], pl["flags"]) for pl in
                                     (plan(n, k, m, prequant=q) for m in range(1, max_m + 1))]
                      for q in (False, True)}
            cfg = dict(n=n, k=k, max_m=max_m, alpha=float(layer.alpha.item()),
                       g=float(layer.input_global_scale_inv.item()),
                       # rows past N are FlashInfer's zero padding (N % 32)
                       w=layer.weight[:n], launch=launch)
            layer._sfx_nvfp4 = cfg
            cfg["key"] = register_layer(self, layer, getattr(layer, "prefix", ""))
            if (n, k) not in _state["oracle"]:
                _state["oracle"][(n, k)] = self._layer_oracle(layer, cfg)
            _state["layers_ours"] += 1

        def _ours(self, layer, cfg, x2d, qa2d=None, fused=None):
            """`fused` None: the load-time plan; else plan(fused=...) (oracle)."""
            n = cfg["n"]
            dev = layer.weight.device
            aq, asf, partial, counters = _state["ws"][dev]
            stream = torch.cuda.current_stream(dev).cuda_stream
            m = (x2d if qa2d is None else qa2d[0]).shape[0]
            if fused is None:
                splits, flags = cfg["launch"][qa2d is not None][m]
            else:
                pl = plan(n, cfg["k"], m, fused=fused, prequant=qa2d is not None)
                splits, flags = pl["splits"], pl["flags"]
            out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            if qa2d is None:
                native.nvfp4_gemm_cuda(x2d, cfg["w"], layer.weight_scale, aq, asf,
                                       partial, out, cfg["g"], cfg["alpha"],
                                       splits, stream, flags, counters)
            else:
                xq, xsf = qa2d
                native.nvfp4_gemm_q_cuda(xq, xsf, cfg["w"], layer.weight_scale,
                                         partial, out, cfg["alpha"], splits, stream,
                                         flags, counters)
            return out

        def _layer_oracle(self, layer, cfg):
            """Ours vs the parent FlashInfer path AND vs the f64 exact-dequant
            reference on the REAL weights, both input routes, every M in
            ORACLE_MS up to the layer's max M (all M tile buckets, split
            plans, non-multiple-of-16 rows). Fatal on mismatch, with the
            standalone oracle's tolerances (nvfp4_gemm.oracle_ok)."""
            from vllm._custom_ops import scaled_fp4_quant
            n, k = cfg["n"], cfg["k"]
            dev = layer.weight.device
            gen = torch.Generator(device=dev).manual_seed(n * 7 + k)
            worst, worst_ref = 0.0, 0.0
            # activations at the layer's design point (amax ~ A/4 with A =
            # 2688 / g): a static global far above N(0,1) (SUFFIX_NVFP4_DENSE's
            # proven bounds) would otherwise zero most blocks on both paths.
            amp = max(1.0, 448.0 * 6.0 / cfg["g"] / 20.0)
            ms = [m for m in ORACLE_MS if m <= cfg["max_m"]]
            aq_ws, asf_ws, _, counters = _state["ws"][dev]
            for m in ms:
                x = (torch.randn(m, k, generator=gen, device=dev) * amp).bfloat16()
                ref = super().apply_weights(layer, x).float()
                xq, xsf = scaled_fp4_quant(x, layer.input_global_scale_inv,
                                           is_sf_swizzled_layout=True)
                # single launch == old 3-launch path, bit for bit (both routes)
                for q, run in ((False, lambda f: self._ours(layer, cfg, x, fused=f)),
                               (True, lambda f: self._ours(layer, cfg, None, (xq, xsf), fused=f))):
                    fq = plan(n, k, m, fused=True, prequant=q)["fused_quant"]
                    diff = fused_vs_old(run, aq_ws[: m * k // 2], asf_ws[: m * k // 16],
                                        counters, fq)
                    if diff is not None:
                        raise RuntimeError(
                            f"{MARKER} LAYER ORACLE FAIL N={n} K={k} M={m} "
                            f"{'prequant' if q else 'bf16'} {plan_str(plan(n, k, m, prequant=q))}: "
                            f"{diff} — refusing to serve with {GATE}=1")
                ours = self._ours(layer, cfg, x).float()  # production plan; aq/asf readback below
                ours_q = self._ours(layer, cfg, None, (xq, xsf)).float()
                exact = exact_ref(xq, xsf, layer.weight, layer.weight_scale,
                                  cfg["alpha"], n).bfloat16().float()
                enorm = exact.norm().clamp_min(1e-30)
                ref_rel = float((ref - exact).norm() / enorm)  # FlashInfer's own error
                # prequant route: vLLM's own quantized x -> must agree with the
                # FlashInfer parent and the exact product (standalone tolerances).
                rel = float((ours_q - ref).norm() / ref.norm().clamp_min(1e-30))
                rel_x = float((ours_q - exact).norm() / enorm)
                worst, worst_ref = max(worst, rel), max(worst_ref, rel_x)
                bad = (not oracle_ok(rel_x, rel, ref_rel) or not torch.isfinite(ours_q).all())
                why = (f"prequant rel_vs_flashinfer={rel:.3e} rel_vs_exact={rel_x:.3e} "
                       f"(flashinfer {ref_rel:.3e})")
                if not bad:
                    # bf16 route: OUR in-kernel quant is the NVFP4 spec with IEEE
                    # math (sf = e4m3(amax*g/6), q = e2m1(x*g/sf)); vLLM's
                    # scaled_fp4_quant uses rcp.approx.ftz and flips some block
                    # scales for some g (silicon 2026-09-29: 2e-2 at o_proj g=93).
                    # So: kernel quant == spec bit-for-bit, GEMM == exact product
                    # of that quant, and vs FlashInfer only the triangle bound.
                    q_own = aq_ws[: m * k // 2].view(m, k // 2)
                    sf_own = asf_ws[: m * k // 16].view(m, k // 16)
                    q_spec, sfb_spec, _ = quantize(x.float().cpu().numpy(), cfg["g"])
                    qmis = float(np.mean(q_own.cpu().numpy() != q_spec)
                                 + np.mean(sf_own.cpu().numpy() != sfb_spec))
                    a_own = dequant_torch(q_own, sf_own)
                    w_u = unswizzle_sf(layer.weight_scale, n, k // 16)
                    own = torch.empty(m, n, dtype=torch.float64, device=dev)
                    step = max(1, (1 << 24) // k)
                    for r0 in range(0, n, step):
                        r1 = min(n, r0 + step)
                        own[:, r0:r1] = a_own @ dequant_torch(layer.weight[r0:r1], w_u[r0:r1]).T
                    own = (own * cfg["alpha"]).bfloat16().float()
                    onorm = own.norm().clamp_min(1e-30)
                    rel_b = float((ours - ref).norm() / ref.norm().clamp_min(1e-30))
                    rel_bx = float((ours - own).norm() / onorm)
                    fi_own = float((ref - own).norm() / onorm)  # FlashInfer vs the spec product
                    worst, worst_ref = max(worst, rel_b), max(worst_ref, rel_bx)
                    bad = (qmis > 1e-3 or rel_bx > max(1e-2, 1.1 * ref_rel)
                           or rel_b > max(ORACLE_REL, 1.1 * (rel_bx + fi_own))
                           or not torch.isfinite(ours).all())
                    why = (f"bf16 quant_mismatch_vs_spec={qmis:.1e} rel_vs_own_exact={rel_bx:.3e} "
                           f"rel_vs_flashinfer={rel_b:.3e} (flashinfer_vs_spec={fi_own:.3e})")
                if bad:
                    raise RuntimeError(
                        f"{MARKER} LAYER ORACLE FAIL N={n} K={k} M={m} "
                        f"{plan_str(plan(n, k, m))}: {why} — refusing to serve with {GATE}=1")
            torch.cuda.synchronize(dev)
            plans = " ".join(f"M{m}:{plan(n, k, m)['tiles']}t/{plan(n, k, m)['splits']}s/"
                             f"{plan(n, k, m)['launches']}L" for m in ms)
            _log(f"LAYER ORACLE PASS N={n} K={k} max_m={cfg['max_m']} [{plans}] "
                 f"max_rel_vs_flashinfer={worst:.2e} max_rel_vs_exact={worst_ref:.2e} "
                 f"(bf16 + prequant routes; single launch == old 3-launch bits)")
            return worst

        def _stock(self, layer, x2d, xsf=None):
            """The parent's apply_weights on the op's 2D operands (M > max M)."""
            if xsf is None:
                return super().apply_weights(layer, x2d)
            qa = QuantizedActivation(x2d, xsf, torch.bfloat16,
                                     torch.Size((x2d.shape[0], x2d.shape[1] * 2)),
                                     self.input_quant_key())
            return super().apply_weights(layer, qa)

        def apply_weights(self, layer, x, bias=None):
            """Static checks (dtype, eligibility) here; the M-dependent route
            inside torch.ops.suffix_nvfp4.linear (see _op_impl)."""
            cfg = getattr(layer, "_sfx_nvfp4", None)
            if cfg is None:
                return super().apply_weights(layer, x, bias)
            qa = as_quantized_activation(x, self.input_quant_key())
            if qa is not None:
                if qa.orig_dtype != torch.bfloat16:
                    return super().apply_weights(layer, x, bias)
                data = qa.data
                out = op(data.reshape(-1, data.shape[-1]), qa.scale, cfg["n"], cfg["key"])
                shape = [*qa.orig_shape[:-1], cfg["n"]]
            else:
                if x.dtype != torch.bfloat16:
                    return super().apply_weights(layer, x, bias)
                out = op(x.reshape(-1, x.shape[-1]), None, cfg["n"], cfg["key"])
                shape = [*x.shape[:-1], cfg["n"]]
            if bias is not None:
                out = out + bias
            return out.view(*shape)

    return SuffixNvFp4LinearKernel


def target_quantization(cfg) -> tuple:
    """(target quantization, draft quantization) from a VllmConfig. With spec
    decode the CURRENT config at first forward can be the drafter's, so the
    target comes from speculative_config.target_model_config when present."""
    if cfg is None:
        return None, None
    sc = getattr(cfg, "speculative_config", None)
    tgt = getattr(sc, "target_model_config", None) or getattr(cfg, "model_config", None)
    dft = getattr(sc, "draft_model_config", None)
    return (getattr(tgt, "quantization", None),
            getattr(dft, "quantization", None) if dft is not None else None)


def layer_census(objects=None) -> dict:
    """Count built linear / MoE layers by quant method class (gc scan, once,
    at startup): e.g. {'linear:UnquantizedLinearMethod': 210,
    'moe:ModelOptNvFp4FusedMoE': 30}. Tells precisely what a checkpoint
    quantized (dense NVFP4 linears vs NVFP4 MoE experts only)."""
    import gc

    import torch
    out: dict = {}
    for o in (gc.get_objects() if objects is None else objects):
        # nn.Modules only, plain __dict__ lookup: never trigger lazy
        # attribute machinery on arbitrary gc objects (ctypes, pytest.mark).
        if not isinstance(o, torch.nn.Module):
            continue
        qm = o.__dict__.get("quant_method")
        if qm is None:
            continue
        cls = type(o).__name__
        if "MoE" in cls or "Experts" in cls:
            kind = "moe"
        elif "Linear" in cls or "LMHead" in cls:
            kind = "linear"
        else:
            continue
        key = f"{kind}:{type(qm).__name__}"
        out[key] = out.get(key, 0) + 1
    return out


def selection_verdict(instances: int, target_quant, disabled: list[str],
                      census: dict | None = None, draft_quant=None) -> str | None:
    """None when SuffixNvFp4LinearKernel was selected for >= 1 NVFP4 dense
    linear, else the loud startup error naming the precise reason."""
    if instances > 0:
        return None
    census = census or {}
    cen = ", ".join(f"{k} x{v}" for k, v in sorted(census.items())) or "-"
    nvfp4_moe = sum(v for k, v in census.items()
                    if k.startswith("moe:") and "Unquantized" not in k)
    if target_quant in (None, "None", ""):
        why = ("the TARGET checkpoint is NOT NVFP4-quantized (quantization=None, "
               "bf16 weights) — there is no NVFP4 linear to replace")
    elif nvfp4_moe:
        why = (f"TARGET quantization={target_quant} quantizes only MoE experts "
               f"({nvfp4_moe} quantized FusedMoE layers -> NVFP4 MoE backend, a "
               "different kernel family); its dense linears are excluded (bf16) — "
               "no NVFP4 dense linear to replace")
    else:
        why = (f"TARGET quantization={target_quant} built no NVFP4 dense linear "
               "through init_nvfp4_linear_kernel with SuffixNvFp4LinearKernel "
               f"(VLLM_DISABLED_KERNELS={','.join(disabled) or '-'})")
    extra = f"; drafter quantization={draft_quant}" if draft_quant is not None else ""
    return (f"{MARKER} NVFP4-GEMM NOT SELECTED with {GATE}=1: {why}{extra}; "
            f"layer census: {cen}")


def _first_forward_check(module, args):
    """Global forward pre-hook (PyTorch API, not a vLLM patch): fires on the
    first module call after model load (profile / encoder warmup, eager, before
    any CUDA-graph capture), removes itself, and fails startup if our kernel
    was never selected; otherwise logs the per-shape selection marker."""
    h = _state.pop("hook", None)
    if h is not None:
        h.remove()
    if _state["checked"]:
        return
    _state["checked"] = True
    tq = dq = None
    try:
        from vllm.config import get_current_vllm_config_or_none
        tq, dq = target_quantization(get_current_vllm_config_or_none())
    except Exception:  # verdict below still fires on instances == 0
        pass
    census = layer_census()
    disabled = [x for x in os.environ.get("VLLM_DISABLED_KERNELS", "").split(",") if x]
    err = selection_verdict(_state["instances"], tq, disabled, census, dq)
    if err is not None:
        _log(err)
        raise RuntimeError(err)
    shapes = ", ".join(f"{k} x{v}" for k, v in sorted(_state["shapes"].items()))
    cen = ", ".join(f"{k} x{v}" for k, v in sorted(census.items()))
    _log(f"NVFP4-GEMM SELECTION: NVFP4 dense linear kernel = SuffixNvFp4LinearKernel "
         f"({_state['instances']} instances; target quantization={tq}); per shape "
         f"(NxK:route x layers): {shapes}; layer census: {cen}")
    _log(summary())


def _disable_earlier(names: list[str]) -> list[str]:
    cur = [s for s in os.environ.get("VLLM_DISABLED_KERNELS", "").split(",") if s]
    add = [n for n in names if n not in cur]
    os.environ["VLLM_DISABLED_KERNELS"] = ",".join(cur + add)
    return add


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_NVFP4_GEMM=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    import torch
    from suffix_hybrid import oxide_kernels
    from vllm import envs
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_NVFP4_KERNELS,
        register_linear_kernel,
    )
    from vllm.platforms import PlatformEnum

    native = oxide_kernels.native()
    max_m, route = max_m_env(), route_table()  # malformed env: fail at startup
    for fn in ("nvfp4_gemm_cuda", "nvfp4_gemm_q_cuda"):
        if not hasattr(native, fn):
            raise RuntimeError(f"{GATE}=1 but _native lacks {fn} (oxide-kernels build)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError(f"{GATE}=1: the NVFP4 GEMM cubin is sm_120a SASS (cc 12.x only)")
    ent = [k for k in oxide_kernels.manifest()["kernels"] if k["name"] == FAMILY]
    if not ent:
        raise RuntimeError(f"{GATE}=1 but the oxide manifest has no {FAMILY!r} cubin")
    cls = _make_kernel_cls()
    earlier = [c.__name__ for c in _POSSIBLE_NVFP4_KERNELS.get(PlatformEnum.CUDA, [])
               if c is not cls]
    added = _disable_earlier(earlier)
    if any(name not in envs.VLLM_DISABLED_KERNELS for name in earlier):
        raise RuntimeError(f"{GATE}=1: VLLM_DISABLED_KERNELS is cached; cannot route "
                           "NVFP4 selection to SuffixNvFp4LinearKernel")
    register_linear_kernel(cls, PlatformEnum.CUDA, "nvfp4")
    if cls not in _POSSIBLE_NVFP4_KERNELS.get(PlatformEnum.CUDA, []):
        raise RuntimeError(f"{GATE}=1: register_linear_kernel did not add our kernel")
    _state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        _first_forward_check)
    _state["armed"] = True
    routes = ",".join(f"{n}x{k}:{mm}" for (n, k), mm in route.items()) or "-"
    _log(f"NVFP4-GEMM armed: SuffixNvFp4LinearKernel registered (M<={max_m} -> "
         f"sm_120a mxf4nvf4 SASS, sha256 {ent[0]['sha256'][:12]}; else FlashInfer "
         f"CUTLASS; route {routes}); disabled earlier candidates: {','.join(added) or '-'}")
    return _state


def summary() -> str:
    return (f"NVFP4-GEMM ACTIVE: {_state['layers_ours']} layers on our decode GEMM, "
            f"{_state['layers_stock']} stay on FlashInfer CUTLASS, "
            f"{len(_state['oracle'])} shapes oracle-checked")
