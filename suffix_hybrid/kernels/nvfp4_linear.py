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
    decode-sized GEMM on an eligible shape (M > 16, padded K, non-bf16, ...)
    is routed to that parent's ``apply_weights`` — the stock FlashInfer
    CUTLASS path, i.e. exactly what vLLM would have run. Routing is by shape,
    decided per layer at load time and logged; never a silent substitute.
  * Pre-quantized inputs (vLLM's fused SiLU*mul / RMSNorm + NVFP4 quant,
    ``input_quant_key() == kNvfp4Dynamic`` inherited) run our GEMM on vLLM's
    packed activation + swizzled scales (kernel asf_mode 1).

Startup (fatal when the gate is on): oxide manifest has ``nvfp4_gemm``,
driver loads the cubin (``oxide_kernels.ensure_loaded``), SM family 12, and a
per-(N, K) LAYER ORACLE on the real checkpoint weights (ours vs the parent
FlashInfer path, both input routes) before any CUDA graph is captured.
Markers: ``[suffix nvfp4-gemm] NVFP4-GEMM armed`` / ``LAYER ORACLE PASS`` /
``NVFP4-GEMM ACTIVE``.

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
MAX_M = 16
ORACLE_REL = 2e-2
_state = {"armed": False, "oracle": {}, "layers_ours": 0, "layers_stock": 0,
          "ws": {}, "instances": 0, "shapes": {}, "checked": False, "hook": None}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def eligible(n: int, k: int, output_size: int, pad_bytes: int) -> str | None:
    """Why (N, K) cannot use our kernel, or None."""
    if n != output_size or pad_bytes:
        return "padded weight"
    if n % 32 or k % 64:
        return f"N={n} % 32 or K={k} % 64"
    return None


def _workspace(dev, n: int, k: int):
    """Per-device scratch, grown at LOAD time only (never inside forward):
    aq [16*K/2] u8, asf [16*K/16] u8, split-K partials [8*16*N] f32."""
    import torch
    ws = _state["ws"].get(dev)
    need = (16 * k // 2, 16 * k // 16, 8 * 16 * n)
    if ws is None or any(have.numel() < want for have, want in zip(ws, need)):
        old = ws or (None, None, None)
        sizes = [max(want, o.numel() if o is not None else 0)
                 for want, o in zip(need, old)]
        ws = (torch.empty(sizes[0], dtype=torch.uint8, device=dev),
              torch.empty(sizes[1], dtype=torch.uint8, device=dev),
              torch.empty(sizes[2], dtype=torch.float32, device=dev))
        _state["ws"][dev] = ws
    return ws


def _make_kernel_cls():
    import torch
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCutlassNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import (
        as_quantized_activation,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        nvfp4_weight_padding_bytes,
    )
    from suffix_hybrid import oxide_kernels

    native = oxide_kernels.native()

    class SuffixNvFp4LinearKernel(FlashInferCutlassNvFp4LinearKernel):
        """FlashInfer-CUTLASS NVFP4 linear + our sm_120a decode GEMM for M<=16."""

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
            n = layer.weight.shape[0]
            k = layer.weight.shape[1] * 2
            why = eligible(n, k, layer.output_size_per_partition,
                           nvfp4_weight_padding_bytes(layer))
            layer._sfx_nvfp4 = None
            route = "flashinfer" if why is not None else "ours"
            key = f"{n}x{k}:{route}"
            _state["shapes"][key] = _state["shapes"].get(key, 0) + 1
            if why is not None:
                _state["layers_stock"] += 1
                _log(f"layer N={n} K={k} stays on FlashInfer CUTLASS: {why}")
                return
            dev = layer.weight.device
            oxide_kernels.ensure_loaded(FAMILY, dev.index)
            _workspace(dev, n, k)
            # One device sync per layer at LOAD time; never in forward.
            cfg = dict(n=n, k=k, alpha=float(layer.alpha.item()),
                       g=float(layer.input_global_scale_inv.item()),
                       splits=native.nvfp4_gemm_splits(n, k))
            layer._sfx_nvfp4 = cfg
            if (n, k) not in _state["oracle"]:
                _state["oracle"][(n, k)] = self._layer_oracle(layer, cfg)
            _state["layers_ours"] += 1

        def _ours(self, layer, cfg, x2d, qa2d=None):
            n = cfg["n"]
            dev = layer.weight.device
            aq, asf, partial = _state["ws"][dev]
            stream = torch.cuda.current_stream(dev).cuda_stream
            if qa2d is None:
                out = torch.empty(x2d.shape[0], n, dtype=torch.bfloat16, device=dev)
                native.nvfp4_gemm_cuda(x2d, layer.weight, layer.weight_scale, aq, asf,
                                       partial, out, cfg["g"], cfg["alpha"],
                                       cfg["splits"], stream)
            else:
                xq, xsf = qa2d
                out = torch.empty(xq.shape[0], n, dtype=torch.bfloat16, device=dev)
                native.nvfp4_gemm_q_cuda(xq, xsf, layer.weight, layer.weight_scale,
                                         partial, out, cfg["alpha"], cfg["splits"],
                                         stream)
            return out

        def _layer_oracle(self, layer, cfg):
            """Ours vs the parent FlashInfer path on the REAL weights, both
            input routes, before serving. Fatal on mismatch."""
            from vllm._custom_ops import scaled_fp4_quant
            n, k = cfg["n"], cfg["k"]
            dev = layer.weight.device
            gen = torch.Generator(device=dev).manual_seed(n * 7 + k)
            worst = 0.0
            for m in (1, 4, MAX_M):
                x = torch.randn(m, k, generator=gen, device=dev, dtype=torch.bfloat16)
                ref = super().apply_weights(layer, x).float()
                ours = self._ours(layer, cfg, x).float()
                xq, xsf = scaled_fp4_quant(x, layer.input_global_scale_inv,
                                           is_sf_swizzled_layout=True)
                ours_q = self._ours(layer, cfg, None, (xq, xsf)).float()
                for got in (ours, ours_q):
                    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
                    worst = max(worst, rel)
                    if not rel <= ORACLE_REL or not torch.isfinite(got).all():
                        raise RuntimeError(
                            f"{MARKER} LAYER ORACLE FAIL N={n} K={k} M={m}: rel={rel:.3e} "
                            f"(> {ORACLE_REL}) — refusing to serve with {GATE}=1")
            torch.cuda.synchronize(dev)
            _log(f"LAYER ORACLE PASS N={n} K={k} splits={cfg['splits']} "
                 f"max_rel_vs_flashinfer={worst:.2e} (bf16 + prequant routes)")
            return worst

        def apply_weights(self, layer, x, bias=None):
            cfg = getattr(layer, "_sfx_nvfp4", None)
            if cfg is None:
                return super().apply_weights(layer, x, bias)
            qa = as_quantized_activation(x, self.input_quant_key())
            if qa is not None:
                data = qa.data
                rows = data.numel() // data.shape[-1] if data.numel() else 0
                if not (1 <= rows <= MAX_M) or qa.orig_dtype != torch.bfloat16:
                    return super().apply_weights(layer, x, bias)
                out = self._ours(layer, cfg, None, (data.reshape(rows, -1), qa.scale))
                shape = [*qa.orig_shape[:-1], cfg["n"]]
            else:
                rows = x.numel() // x.shape[-1] if x.numel() else 0
                if not (1 <= rows <= MAX_M) or x.dtype != torch.bfloat16:
                    return super().apply_weights(layer, x, bias)
                out = self._ours(layer, cfg, x.reshape(rows, x.shape[-1]))
                shape = [*x.shape[:-1], cfg["n"]]
            if not _state.get("active_logged"):
                _state["active_logged"] = True
                _log(summary())
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
    for fn in ("nvfp4_gemm_cuda", "nvfp4_gemm_q_cuda", "nvfp4_gemm_splits"):
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
    _log(f"NVFP4-GEMM armed: SuffixNvFp4LinearKernel registered (M<={MAX_M} -> "
         f"sm_120a mxf4nvf4 SASS, sha256 {ent[0]['sha256'][:12]}; else FlashInfer "
         f"CUTLASS); disabled earlier candidates: {','.join(added) or '-'}")
    return _state


def summary() -> str:
    return (f"NVFP4-GEMM ACTIVE: {_state['layers_ours']} layers on our decode GEMM, "
            f"{_state['layers_stock']} stay on FlashInfer CUTLASS, "
            f"{len(_state['oracle'])} shapes oracle-checked")
