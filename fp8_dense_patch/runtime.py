# SPDX-License-Identifier: Apache-2.0
"""Worker-side half of fp8_dense_patch (imports torch; vLLM lazily so the CPU
tests can stub it). See the package docstring for the contract."""

import sys

import torch
from torch import nn

from . import PATCH_NAME, PATCH_REVISION, layer_patterns, selected

FP8_MAX = 448.0  # float8_e4m3fn
QUANT_BOUND = 5e-2   # rel. Frobenius FP8 vs BF16 (e4m3 floor ~3.5 % on N(0,1))
KERNEL_BOUND = 1e-2  # rel. Frobenius kernel vs exact dequantized reference

_OPS = None  # vllm._custom_ops, bound by convert_model (tests inject a fake)
_METHOD_CLS = None
_SELFTESTED = set()


def quantize_weight(w: torch.Tensor):
    """[N, K] -> (float8_e4m3fn [N, K], float32 [N]); w ~= q * s[:, None]."""
    w32 = w.float()
    s = w32.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX
    q = (w32 / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s.contiguous()


def reference(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor):
    """Exact-dequant fp32 reference of fp8_linear (per-token x quant)."""
    x32 = x.reshape(-1, x.shape[-1]).float()
    xs = x32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    xq = (x32 / xs).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * xs
    return xq @ (q.float() * s[:, None]).t()


def rel_frob(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def fp8_linear(x, weight, weight_scale, bias=None):
    """x [..., K] bf16, weight fp8 [N, K], weight_scale f32 [N] -> [..., N].
    Two kernels, static shapes, no host sync."""
    x2 = x.reshape(-1, x.shape[-1])
    qx, xs = _OPS.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
    out = _OPS.cutlass_scaled_mm(qx, weight.t(), xs, weight_scale,
                                 out_dtype=x.dtype, bias=bias)
    return out.view(*x.shape[:-1], weight.shape[0])


class Fp8DenseNNLinear(nn.Linear):
    """Class swapped onto converted plain nn.Linear modules (MTP eh_proj)."""

    def forward(self, x):
        return fp8_linear(x, self.weight, self.weight_scale, self.bias)


def method_cls():
    global _METHOD_CLS
    if _METHOD_CLS is None:
        from vllm.model_executor.layers.linear import LinearMethodBase

        class Fp8DenseLinearMethod(LinearMethodBase):
            """quant_method swapped onto converted LinearBase layers."""

            def create_weights(self, *args, **kwargs):
                raise RuntimeError(f"[suffix {PATCH_NAME}] post-load method only")

            def process_weights_after_loading(self, layer) -> None:
                pass

            def apply(self, layer, x, bias=None):
                return fp8_linear(x, layer.weight, layer.weight_scale, bias)

        _METHOD_CLS = Fp8DenseLinearMethod
    return _METHOD_CLS


def _selftest(name, w, q, s, bias) -> None:
    """Once per (device, N, K): the real kernels at M=6 vs BF16 and vs the
    exact-dequant reference, before the BF16 weight is freed. Fails closed."""
    x = torch.randn(6, w.shape[1], device=w.device, dtype=torch.bfloat16)
    out = fp8_linear(x, q, s, bias).float()
    ref = nn.functional.linear(x, w, bias).float()
    exact = reference(x, q, s) + (0 if bias is None else bias.float())
    eq, ek = rel_frob(out, ref), rel_frob(out, exact)
    if not (bool(torch.isfinite(out).all()) and eq <= QUANT_BOUND and ek <= KERNEL_BOUND):
        raise RuntimeError(
            f"[suffix {PATCH_NAME}] self-test FAILED on {name} {tuple(w.shape)}: "
            f"rel err vs bf16 {eq:.3e} (<= {QUANT_BOUND}), vs exact dequant "
            f"{ek:.3e} (<= {KERNEL_BOUND})")


def _capability(dev):
    return torch.cuda.get_device_capability(dev) if dev.type == "cuda" else None


def convert_model(model, patterns=None) -> list:
    """Convert allowlisted BF16 linears in place; returns [(name, N, K)]."""
    global _OPS
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    patterns = layer_patterns() if patterns is None else patterns
    dev = next(model.parameters()).device
    cap = _capability(dev)
    if cap is None or cap[0] != 12:
        print(f"[suffix {PATCH_NAME}] present but inert: {dev} capability {cap} "
              "is not SM120", file=sys.stderr, flush=True)
        return []
    if _OPS is None:
        from vllm import _custom_ops

        if not _custom_ops.cutlass_scaled_mm_supports_fp8(cap[0] * 10 + cap[1]):
            raise RuntimeError(f"[suffix {PATCH_NAME}] this vLLM build has no "
                               "SM120 cutlass_scaled_mm FP8 kernels")
        _OPS = _custom_ops
    method = method_cls()()
    done, skipped, bf16_bytes = [], [], 0
    for name, mod in list(model.named_modules()):
        if not selected(name, patterns):
            continue
        if isinstance(mod, LinearBase):
            if type(mod.quant_method) is not UnquantizedLinearMethod:
                continue
            kind = "linear"
        elif type(mod) is nn.Linear:
            kind = "nn"
        else:
            continue
        w = mod.weight
        n, k = w.shape if w.dim() == 2 else (0, 0)
        if w.dtype != torch.bfloat16 or w.device != dev or n % 16 or k % 16 or n == 0:
            skipped.append(f"{name}{tuple(w.shape)}:{w.dtype}")
            continue
        bias = getattr(mod, "bias", None)
        q, s = quantize_weight(w.data)
        key = (w.device, n, k)
        if key not in _SELFTESTED:
            _selftest(name, w.data, q, s, None if bias is None else bias.data)
            _SELFTESTED.add(key)
        bf16_bytes += w.numel() * 2
        mod.weight = nn.Parameter(q, requires_grad=False)
        mod.weight_scale = nn.Parameter(s, requires_grad=False)
        if kind == "linear":
            mod.quant_method = method
        else:
            mod.__class__ = Fp8DenseNNLinear
        done.append((name, n, k))
    torch.cuda.empty_cache()
    shapes = sorted({(n, k) for _, n, k in done})
    print(f"[suffix {PATCH_NAME}] converted {len(done)} linears to FP8 W8A8 on "
          f"{dev} ({bf16_bytes / 2**30:.2f} GiB bf16 -> "
          f"{bf16_bytes / 2**31:.2f} GiB fp8; shapes N x K {shapes}; rev "
          f"{PATCH_REVISION})" + (f"; skipped {skipped}" if skipped else ""),
          file=sys.stderr, flush=True)
    return done
