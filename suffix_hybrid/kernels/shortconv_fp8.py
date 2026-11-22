# SPDX-License-Identifier: Apache-2.0
"""LFM2.5 ShortConv in_proj/out_proj on the FP8-block path.

Gate: SUFFIX_SHORTCONV_FP8=1 (armed by the rocm-patch hook in sitecustomize;
the source patch adds ``process_weights_after_loading`` to
Lfm2MoeForCausalLM, which model_loader/utils.py calls with no args once
weights are loaded).

The LFM2.5-24B-A2B-FP8 checkpoint excludes the whole short_conv block from
FP8 (config.json ignore list ``re:.*\\.(?:conv|short_conv)\\..*``), so its
in_proj (6144x2048) and out_proj (2048x2048) stream BF16 weights through the
wmma bf16 GEMM at decode M: ~1008 MB/step over 30 layers at c1 (profile
20261015T204656Z run: the conv-projection cluster is ~13.7% of the c1 step).

What this does at load, per layer:
  - quantize the BF16 weight to e4m3 with 128x128-block f32 amax/448 scales
    (the same scale math the checkpoint's own fp8-block tensors carry);
  - swap in the parameters under the names the stock blockwise kernel reads
    (weight, weight_scale) via replace_parameter;
  - build the kernel with the stock chooser, exactly as the compressed-
    tensors scheme builds it for the model's other dense linears:
    init_fp8_linear_kernel(kFp8Dynamic128Sym, kFp8Static128BlockSym, ...)
    -- under the trial env (VLLM_USE_DEEP_GEMM=0) that is
    CutlassFp8BlockScaledMMKernel: ops.cutlass_scaled_mm with per-token-group
    (1,128) activation quant;
  - set layer.quant_method to a shim whose apply() forwards to the kernel.

Fail-closed: two oracles must pass per layer before the swap is committed --
  1. dequant round trip: every element within amax*0.20 of the BF16 weight
     (e4m3 round-to-nearest near a block's amax is <= amax*0.036; a scale /
     layout bug is off by the whole block);
  2. eager GEMM: kernel output vs the BF16 reference, rel Frobenius error
     <= 0.06 over 8 random probes (fp8 quant noise averages <1% over K=2048;
     a wiring bug is off by 100%+).
Any failure raises SystemExit -> the load aborts; never serve unverified
weights. Gate unset: this module is never imported.

Exactness class: APPROXIMATE (fp8 weight quantization). This changes outputs
vs the BF16 conv projections; the trial's agreement vs the baseline measures
how much, and the checkpoint's deliberate exclusion of these layers means a
quality regression is possible -- that is what the trial is for.

Bitwise-neutral composition: SUFFIX_SHORTCONV_FUSED fuses the conv1d chain
inside ShortConv.forward and never touches in_proj/out_proj, so both gates
compose.
"""
import os
import sys

import torch

GATE = "SUFFIX_SHORTCONV_FP8"
MARKER = "[suffix shortconv-fp8]"
BLOCK = 128
FP8_MAX = 448.0  # torch.float8_e4m3fn finite max
# e4m3 has a 3-bit mantissa: half-ulp near a block's amax is amax*2^-4/...
# ~amax*0.036 in dequant units; 0.20 catches any scale/layout bug (a wrong
# scale is off by the whole block) without tolerating real quant noise.
ELEM_TOL = 0.20
# rel Frobenius bound for the eager GEMM oracle: fp8 weight quant noise
# averages out over K=2048 (typically <1%); a wiring bug is 100%+ off.
GEMM_TOL = 0.06
# size of the eager GEMM probe batch (M=8: one probe per activation group
# row is unnecessary; 8 rows is enough signal and cheap at load).
PROBE_M = 8


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def convert(model: torch.nn.Module) -> None:
    """Model-level post-load hook; inert unless the gate is set."""
    if os.environ.get(GATE, "").strip() != "1":
        return
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if name.endswith(".short_conv.in_proj")
        or name.endswith(".short_conv.out_proj")
    ]
    if not targets:
        _log("no .short_conv.in_proj/.out_proj found; nothing converted")
        return
    n_done = 0
    for name, layer in targets:
        _convert_one(name, layer)
        n_done += 1
    _log(f"converted {n_done}/{len(targets)} linears to fp8-block")


def _quantize_weight(w: torch.Tensor):
    """BF16 (n,k) -> (e4m3 (n,k), f32 (n/128,k/128) scales). f32 math."""
    n, k = w.shape
    blocks = w.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).float()
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (blocks.float() / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(n, k), scale


def _dequant_oracle(w: torch.Tensor, w_fp8: torch.Tensor,
                    scale: torch.Tensor) -> None:
    """Every element of the round trip within amax*ELEM_TOL of the original."""
    n, k = w.shape
    nb, kb = scale.shape
    dq = (w_fp8.view(nb, BLOCK, kb, BLOCK).float()
          * scale[:, None, :, None]).view(n, k)
    err = (dq - w.float()).abs().view(nb, BLOCK, kb, BLOCK)
    # threshold = amax * ELEM_TOL, amax = scale * FP8_MAX
    over = (err > scale[:, None, :, None] * FP8_MAX * ELEM_TOL).any(dim=(1, 3))
    n_bad = int(over.sum().item())
    if n_bad:
        raise SystemExit(
            f"{MARKER} dequant oracle FAILED: {n_bad}/{nb * kb} blocks over "
            f"amax*{ELEM_TOL}; refusing to serve")


def _convert_one(name: str, layer: torch.nn.Module) -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        _log(f"{name}: quant_method is "
             f"{type(layer.quant_method).__name__}; left stock")
        return
    w = layer.weight.data
    if w.ndim != 2 or w.dtype != torch.bfloat16:
        raise SystemExit(f"{MARKER} {name}: expected 2D BF16 weight, got "
                         f"{tuple(w.shape)} {w.dtype}; refusing")
    n, k = w.shape
    if n % BLOCK or k % BLOCK:
        raise SystemExit(f"{MARKER} {name}: shape {n}x{k} not divisible by "
                         f"{BLOCK}; refusing")

    w_fp8, scale = _quantize_weight(w)
    _dequant_oracle(w, w_fp8, scale)

    # Eager GEMM oracle against the BF16 reference, on the STOCK kernel,
    # before the parameters are swapped in place.
    x = torch.randn(PROBE_M, k, dtype=torch.bfloat16, device=w.device) * 0.5
    ref = torch.nn.functional.linear(x, w)
    ref_norm = ref.float().norm().clamp_min(1e-6)

    kernel = _make_kernel(n, k, w.dtype, name)

    from vllm.model_executor.layers.quantization.utils import replace_parameter
    replace_parameter(layer, "weight",
                      torch.nn.Parameter(w_fp8, requires_grad=False))
    # 5b20 crash fix: a BF16 (UnquantizedLinearMethod) layer has NO
    # weight_scale attribute, and the deployed replace_parameter does
    # `old = getattr(mod, name)` before anything else -> AttributeError.
    # Register the block-scale parameter when absent (None counts as
    # absent); when the layer already carries one, keep the stock
    # replace_parameter path so the swap semantics stay identical.
    scale_param = torch.nn.Parameter(scale, requires_grad=False)
    if getattr(layer, "weight_scale", None) is not None:
        replace_parameter(layer, "weight_scale", scale_param)
    else:
        layer.register_parameter("weight_scale", scale_param)
    # 209d boot fix (run 20261017T023120Z-rtx-5090-46fe): under this trial env
    # the stock chooser returns MarlinFP8, whose process_weights_after_loading
    # -> prepare_fp8_layer_for_marlin reads BOTH layer.orig_dtype and
    # layer.weight_block_size off the layer. Native FP8 layers get them from
    # Fp8LinearMethod.create_weights; a converted unquantized BF16 layer has
    # neither -> AttributeError at marlin_utils_fp8.py:146, and without
    # weight_block_size the scales take the channel-wise branch and die in
    # view(). Both values below state what THIS quantization did: the weight
    # was BF16 before the e4m3 swap, and the scales are 128x128-block.
    layer.orig_dtype = w.dtype
    layer.weight_block_size = [BLOCK, BLOCK]
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x, None)
    rel = (out.float() - ref.float()).norm() / ref_norm
    if not torch.isfinite(rel) or rel.item() > GEMM_TOL:
        raise SystemExit(f"{MARKER} {name}: GEMM oracle FAILED "
                         f"(rel err {rel.item():.4f} > {GEMM_TOL}); refusing")

    layer.quant_method = _Fp8BlockShim(kernel)
    _log(f"{name}: {n}x{k} bf16 -> fp8-block; oracles pass "
         f"(dequant ok, gemm rel {rel.item():.4f})")


def _make_kernel(n: int, k: int, dtype: torch.dtype, name: str):
    """Stock kernel chooser, same call the compressed-tensors scheme makes
    for the model's other dense linears (same env -> same kernel type)."""
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    return init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=dtype,
        out_dtype=dtype,
        weight_shape=(n, k),
        module_name=name,
    )


class _Fp8BlockShim:
    """Minimal LinearMethodBase stand-in: the stock kernel does the math."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, layer, x, bias=None):
        return self._kernel.apply_weights(layer, x, bias)

    def process_weights_after_loading(self, layer):
        self._kernel.process_weights_after_loading(layer)
