# SPDX-License-Identifier: Apache-2.0
"""NVFP4 mode of the dense-linear patch (``SUFFIX_NVFP4_DENSE=1``): allowlisted
BF16 linears become exactly what a ModelOpt ``modelopt_fp4`` (quant_algo NVFP4,
W4A4) checkpoint layer is after vLLM's own post-load, through vLLM's own code:

  1. weight: per-tensor global g_w = 2688 / amax(W) (one scalar per fused
     layer: no "global scale differs across partitions" warning), vLLM's
     ``scaled_fp4_quant(W, g_w, is_sf_swizzled_layout=False)`` (the same call
     its online NVFP4 MoE quantizer makes) -> packed e2m1 uint8 [N, K/2] +
     linear e4m3 block-16 scales [N, K/16] = the on-disk checkpoint layout;
  2. ``build_linear_method(cfg, "NVFP4", prefix)`` -> vLLM's generic
     ``ModelOptLinearMethod`` (KNvfp4Static + KNvfp4Dynamic) ->
     ``create_weights`` (registers weight / weight_scale / weight_scale_2 /
     input_scale and selects the kernel through ``init_nvfp4_linear_kernel``,
     so ``SUFFIX_NVFP4_GEMM``'s SuffixNvFp4LinearKernel serves these layers
     when armed) -> we fill the checkpoint tensors -> its
     ``process_weights_after_loading`` (global scales, alpha, the selected
     kernel's swizzle / padding) -> ``layer.quant_method`` = that method.

Activation global scale (input_scale). ModelOpt stores a STATIC calibrated
per-tensor scalar (amax_act / 2688); per-16 block scales are dynamic in the
quant kernel. vLLM 0.30 has no dynamic-global NVFP4 linear path (its
per-token NVFP4 is MoE-only, SM100 + TRTLLM), and the global cannot be made
per-call without breaking our kernel's load-time float cache and vLLM's fused
norm/act + quant producers, which read input_global_scale_inv. So it stays
static and is chosen so the activation can NEVER saturate (a clipped outlier
is the unrecoverable failure; excess headroom only costs precision on blocks
whose amax < A / 28672 = the e4m3 normal floor, and on Gaussian data 256x
headroom is free, 4096x costs ~1 point, 16384x breaks). A = input amax bound,
proved from the model's own weights, first rule that applies:
  * ``SUFFIX_NVFP4_DENSE_ACT_AMAX`` = "glob=value,..." (explicit override);
  * ``*self_attn.qkv_proj`` / ``*self_attn.q_proj`` (input = input_layernorm
    output) and ``*mlp.gate_up_proj`` (pre_feedforward_layernorm): an RMSNorm
    output satisfies |y_i| <= sqrt(K) * max|w_norm|;
  * ``*self_attn.o_proj`` with an unscaled ``v_norm`` (Gemma 4): the attention
    output is a convex combination of RMS-normalised v -> |o| <= sqrt(head_dim);
  * ``*mlp.down_proj``: |act(g_j) u_j| <= |g_j u_j| (GELU / SiLU) <=
    R^2 (|a_j||b_j| + |a_j . b_j|) / 2 with a_j / b_j the gate / up rows and
    R = sqrt(K) max|w_norm| the input norm bound. Rigorous but LOOSE
    (~K x (max/rms w_norm)^2 above typical): the main precision risk, logged;
    override it once serving data exists;
  * none -> the layer stays BF16 (logged).
There is no calibration pass.

Fail closed: per (device, N, K) boot self-test on the real layer at M=6 with
activations at the layer's design point (amax ~ A/4): rel. Frobenius vs BF16
<= QUANT_BOUND and vs the exact-dequant reference <= KERNEL_BOUND.
"""

import fnmatch
import math
import os
import sys
from types import SimpleNamespace

import torch

from . import NVFP4_LAYERS_ENV, PATCH_REVISION, nvfp4_layer_patterns, selected

MARK = "[suffix nvfp4-dense]"
ACT_ENV = "SUFFIX_NVFP4_DENSE_ACT_AMAX"
FP4_RANGE = 448.0 * 6.0   # e4m3 max * e2m1 max: global g = 2688 / amax
GROUP = 16
QUANT_BOUND = 0.16   # W4A4 vs BF16; NVFP4 floor ~13.4 % on N(0,1) x N(0,.02)
KERNEL_BOUND = 2e-2  # vs exact dequant (same bar as SUFFIX_NVFP4_GEMM's oracle)
E4M3_NORMAL_SPAN = 448.0 / 2.0 ** -6  # 28672: block amax >= A/28672 is exact-scaled

_OPS = None  # vllm._custom_ops (tests inject a fake)
_BUILD = None  # vLLM build_linear_method (tests inject a fake)
_SELFTESTED = set()


def act_overrides():
    """SUFFIX_NVFP4_DENSE_ACT_AMAX="glob=float,..." -> ((glob, float), ...)."""
    out = []
    for item in os.environ.get(ACT_ENV, "").split(","):
        if item.strip():
            glob, _, val = item.rpartition("=")
            v = float(val)
            if not glob.strip() or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{ACT_ENV}: bad entry {item!r} (want glob=positive)")
            out.append((glob.strip(), v))
    return tuple(out)


def _norm_bound(norm, k):
    w = getattr(norm, "weight", None)
    if w is None:
        return None
    # 1.01: the norm output is rounded to bf16 (2^-8 relative)
    return 1.01 * math.sqrt(k) * float(w.detach().float().abs().max())


def act_bound(name, modules):
    """(A, rule) = proven |input| bound of linear `name`, or None."""
    for glob, v in act_overrides():
        if fnmatch.fnmatchcase(name, glob):
            return v, "env"
    parent, _, leaf = name.rpartition(".")
    block, _, sub = parent.rpartition(".")
    layer, mod = modules.get(block), modules[name]
    if layer is None:
        return None
    k = mod.input_size_per_partition
    if sub == "self_attn" and leaf in ("qkv_proj", "q_proj"):
        b = _norm_bound(getattr(layer, "input_layernorm", None), k)
        return None if b is None else (b, "input_layernorm")
    if sub == "mlp" and leaf == "gate_up_proj":
        b = _norm_bound(getattr(layer, "pre_feedforward_layernorm", None), k)
        return None if b is None else (b, "pre_feedforward_layernorm")
    if sub == "self_attn" and leaf == "o_proj":
        attn = modules[parent]
        vn, hd = getattr(attn, "v_norm", None), getattr(attn, "head_dim", None)
        if vn is None or getattr(vn, "has_weight", True) or not hd:
            return None
        return 1.01 * math.sqrt(hd), "v_norm"
    if sub == "mlp" and leaf == "down_proj":
        gu = getattr(modules[parent], "gate_up_proj", None)
        norm = getattr(layer, "pre_feedforward_layernorm", None)
        if gu is None or gu.weight.dtype != torch.bfloat16:
            return None
        r = _norm_bound(norm, gu.input_size_per_partition)
        if r is None:
            return None
        w = gu.weight.detach()
        half = w.shape[0] // 2
        if half != k:
            return None
        a, b = w[:half].float(), w[half:].float()
        pair = a.norm(dim=1) * b.norm(dim=1) + (a * b).sum(1).abs()
        return 1.02 * r * r * float(pair.max()) / 2, "gate_up cauchy-schwarz"
    return None


def quantize_weight(w):
    """bf16 [N, K] -> (packed uint8 [N, K/2], e4m3 [N, K/16], weight_scale_2)
    in the ModelOpt checkpoint layout, by vLLM's scaled_fp4_quant."""
    amax = max(float(w.abs().amax()), 1e-12)  # load time: host sync is fine
    g = torch.tensor(FP4_RANGE / amax, dtype=torch.float32, device=w.device)
    q, sf = _OPS.scaled_fp4_quant(w.contiguous(), g, is_sf_swizzled_layout=False)
    return q, sf, amax / FP4_RANGE


def reference(x, q, sf, ws2, act_amax, chunk=4096):
    """Exact-dequant fp32 result of the W4A4 layer (activation quantized with
    the static global 2688 / act_amax, per-16 dynamic block scales)."""
    from suffix_hybrid.tools.quantize_nvfp4 import dequantize_weight, quantize_weight as qw

    x2 = x.reshape(-1, x.shape[-1]).float()
    xq, xsf, xs2 = qw(x2, act_amax)
    xd = dequantize_weight(xq, xsf, float(xs2))
    return torch.cat([xd @ dequantize_weight(q[i:i + chunk], sf[i:i + chunk], ws2).t()
                      for i in range(0, q.shape[0], chunk)], 1)


def layer_act_amax(mod) -> float:
    """The activation amax the processed layer really quantizes with (fp32
    1/input_scale on device): the exact reference must use the same value, a
    1-ulp different global flips e2m1 ties."""
    return FP4_RANGE / float(mod.input_global_scale_inv)


def rel_frob(a, b) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _builder():
    global _BUILD
    if _BUILD is None:
        from vllm.model_executor.layers.quantization.modelopt import build_linear_method

        _BUILD = build_linear_method
    return _BUILD


def convert_layer(name, mod, act_amax):
    """BF16 LinearBase -> vLLM ModelOpt NVFP4 W4A4 layer. Returns the
    checkpoint-layout tensors (q, sf, weight_scale_2) for reference checks."""
    w = mod.weight.data
    q, sf, ws2 = quantize_weight(w)
    method = _builder()(SimpleNamespace(group_size=GROUP), "NVFP4", name)
    wl = getattr(mod, "weight_loader_v2", None) or getattr(mod, "weight_loader", None)
    with torch.device(w.device):
        method.create_weights(mod, mod.input_size_per_partition,
                              list(mod.output_partition_sizes), mod.input_size,
                              mod.output_size, mod.params_dtype, weight_loader=wl)
    if (tuple(mod.weight.shape) != tuple(q.shape) or mod.weight.dtype != q.dtype
            or tuple(mod.weight_scale.shape) != tuple(sf.shape)):
        raise RuntimeError(f"{MARK} {name}: vLLM NVFP4 params {tuple(mod.weight.shape)}/"
                           f"{tuple(mod.weight_scale.shape)} != quantized "
                           f"{tuple(q.shape)}/{tuple(sf.shape)}")
    mod.weight.data.copy_(q)
    mod.weight_scale.data.copy_(sf)
    mod.weight_scale_2.data.fill_(ws2)
    mod.input_scale.data.fill_(act_amax / FP4_RANGE)
    method.process_weights_after_loading(mod)
    mod.quant_method = method
    return q, sf, ws2


def _selftest(name, mod, w, bias, q, sf, ws2, act_amax) -> None:
    """Real kernel at M=6 vs BF16 and vs exact dequant. Fails closed."""
    x = (torch.randn(6, w.shape[1], device=w.device) * (act_amax / 20)).bfloat16()
    out = mod.quant_method.apply(mod, x, bias).float()
    ref = torch.nn.functional.linear(x, w, bias).float()
    exact = reference(x, q, sf, ws2, layer_act_amax(mod))
    exact = exact + (0 if bias is None else bias.float())
    eq, ek = rel_frob(out, ref), rel_frob(out, exact)
    if not (bool(torch.isfinite(out).all()) and eq <= QUANT_BOUND and ek <= KERNEL_BOUND):
        raise RuntimeError(
            f"{MARK} self-test FAILED on {name} {tuple(w.shape)}: rel err vs bf16 "
            f"{eq:.3e} (<= {QUANT_BOUND}), vs exact dequant {ek:.3e} (<= {KERNEL_BOUND})")


def convert_model(model, patterns=None) -> list:
    """Convert allowlisted BF16 linears in place; returns [(name, N, K, A, rule)]."""
    global _OPS
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from .runtime import _capability

    patterns = nvfp4_layer_patterns() if patterns is None else patterns
    dev = next(model.parameters()).device
    cap = _capability(dev)
    if cap is None or cap[0] != 12:
        print(f"{MARK} present but inert: {dev} capability {cap} is not SM120",
              file=sys.stderr, flush=True)
        return []
    if _OPS is None:
        from vllm import _custom_ops

        _OPS = _custom_ops
    modules = dict(model.named_modules())
    plan, skipped = [], []
    for name, mod in modules.items():  # bounds first: down_proj reads gate_up bf16
        if not selected(name, patterns):
            continue
        if not isinstance(mod, LinearBase) or type(mod.quant_method) is not UnquantizedLinearMethod:
            continue
        w = mod.weight
        n, k = w.shape if w.dim() == 2 else (0, 0)
        if w.dtype != torch.bfloat16 or w.device != dev or n % 16 or k % 16 or n == 0:
            skipped.append(f"{name}{tuple(w.shape)}:{w.dtype}")
            continue
        bound = act_bound(name, modules)
        if bound is None:
            skipped.append(f"{name}:no activation bound (set {ACT_ENV})")
            continue
        plan.append((name, mod, n, k, *bound))
    done, bf16_bytes = [], 0
    for name, mod, n, k, amax, rule in plan:
        w = mod.weight.data
        bias = getattr(mod, "bias", None)
        bias = None if bias is None else bias.data
        q, sf, ws2 = convert_layer(name, mod, amax)
        key = (w.device, n, k)
        if key not in _SELFTESTED:
            _selftest(name, mod, w, bias, q, sf, ws2, amax)
            _SELFTESTED.add(key)
        del w, q, sf
        bf16_bytes += n * k * 2
        done.append((name, n, k, amax, rule))
    torch.cuda.empty_cache()
    by_rule = {}
    for _n, _N, _K, a, rule in done:
        lo, hi = by_rule.get(rule, (a, a))
        by_rule[rule] = (min(lo, a), max(hi, a))
    bounds = "; ".join(f"{r}: A {lo:.3g}..{hi:.3g} (exact block scales down to "
                       f"amax {lo / E4M3_NORMAL_SPAN:.2g})" for r, (lo, hi) in by_rule.items())
    print(f"{MARK} converted {len(done)} linears to NVFP4 W4A4 (vLLM ModelOpt method) "
          f"on {dev} ({bf16_bytes / 2**30:.2f} GiB bf16 -> "
          f"{bf16_bytes * 0.5625 / 2 / 2**30:.2f} GiB nvfp4; shapes N x K "
          f"{sorted({(n, k) for _, n, k, _a, _r in done})}; STATIC activation "
          f"global scale from proven input bounds: {bounds or '-'}; rev {PATCH_REVISION}; "
          f"{NVFP4_LAYERS_ENV}={','.join(patterns)})"
          + (f"; skipped {skipped}" if skipped else ""), file=sys.stderr, flush=True)
    return done
