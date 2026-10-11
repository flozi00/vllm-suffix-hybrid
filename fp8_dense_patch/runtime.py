# SPDX-License-Identifier: Apache-2.0
"""Worker-side half of fp8_dense_patch (imports torch; vLLM lazily so the CPU
tests can stub it). See the package docstring for the contract."""

import os
import sys

import torch
from torch import nn

from . import PATCH_NAME, PATCH_REVISION, layer_patterns, selected

FP8_MAX = 448.0  # float8_e4m3fn
FP4_MAX = 6.0    # e2m1
QUANT_BOUND = 5e-2   # rel. Frobenius FP8 vs BF16 (e4m3 floor ~3.5 % on N(0,1))
KERNEL_BOUND = 1e-2  # rel. Frobenius kernel vs exact dequantized reference
FP4_QUANT_BOUND = 0.15   # weight-only NVFP4 vs BF16 (e2m1 floor ~9 % on N(0,1))
FP4_KERNEL_BOUND = 2e-2  # Marlin zeroes block scales < 2^-6 of the largest (S0E5M3)

# SUFFIX_FP8_DENSE_BACKEND:
#   cutlass     W8A8: dynamic per-token act quant + cutlass_scaled_mm (default, today)
#   marlin      W8A16 Marlin for every M (FP8 weights repacked in place: no extra memory,
#               no act-quant kernel; prefill runs bf16 MMA after dequant)
#   auto        Marlin W8A16 for M <= SUFFIX_FP8_DENSE_MARLIN_MAX_M, cutlass W8A8 above:
#               both FP8 layouts stay resident (+1x the converted FP8 bytes)
#   marlin_fp4  NVFP4 weight-only (W4A16 Marlin) for every M: half the FP8 bytes, LOSSY
#               (per-16 e4m3 scales + per-tensor global) -- quality-gate before serving
#   ours        our Triton W8A16 / in-GEMM-quant W8A8 kernels (ours.py) and cutlass, the
#               fastest per (shape, M bucket) from a load-time cold-L2 timing table; one
#               FP8 copy (cutlass reads the same [N, K] tensor transposed)
#   best        ours + Marlin W8A16 as a third route (its repack kept only for shapes
#               where Marlin wins a bucket)
#   ours_fp4    NVFP4 weight-only through our W4A16 kernel for every M, LOSSY like
#               marlin_fp4 -- quality-gate before serving
BACKEND_ENV = "SUFFIX_FP8_DENSE_BACKEND"
MARLIN_MAX_M_ENV = "SUFFIX_FP8_DENSE_MARLIN_MAX_M"
OURS_BACKENDS = ("ours", "best", "ours_fp4")
BACKENDS = ("cutlass", "marlin", "auto", "marlin_fp4") + OURS_BACKENDS
ROUTED = ("auto",) + OURS_BACKENDS  # per-call M route inside torch.ops.suffix_fp8d.linear
MARLIN_MAX_M = 64  # placeholder until glm_dense_marlin_bench recommends one
ORACLE_MS = (1, 6, 32)
OP_NS = "suffix_fp8d"

_OPS = None  # vllm._custom_ops, bound by convert_model (tests inject a fake)
_METHOD_CLS = None
_SELFTESTED = set()
_LAYERS = {}  # op key -> module (routed backends)
_ROUTES = {}  # (backend, device, N, K) -> ours.Route (tuned on the first layer of a shape)
_OURS = None  # the ours module, bound at load (no imports inside the traced forward)
_NO_SEM = set()  # streams already reported without split-K counters
_LIB = None
_APPLY = {}   # "fp8" / "fp4" -> vLLM's apply_fp{8,4}_marlin_linear, bound at load (no
              # imports inside the traced forward)


def backend() -> str:
    b = os.environ.get(BACKEND_ENV, "").strip() or "cutlass"
    if b not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={b!r}: want one of {BACKENDS}")
    return b


def marlin_max_m() -> int:
    raw = os.environ.get(MARLIN_MAX_M_ENV, "").strip()
    if not raw:
        return MARLIN_MAX_M
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{MARLIN_MAX_M_ENV}={raw!r}: want a positive integer")
    return int(raw)


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


# --- Marlin weight-only (vLLM's own repack + kernels) -----------------------
def _holder(n, k, **params):
    """The attributes vLLM's prepare_fp{8,4}_layer_for_marlin read and replace."""
    h = nn.Module()
    h.output_size_per_partition, h.input_size_per_partition = n, k
    h.orig_dtype = h.params_dtype = torch.bfloat16
    for name, t in params.items():
        setattr(h, name, nn.Parameter(t, requires_grad=False))
    return h


def marlin_fp8_pack(q, s):
    """FP8 [N, K] + f32 [N] per-channel scales -> Marlin W8A16 tensors."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

    _APPLY["fp8"] = apply_fp8_marlin_linear
    h = _holder(*q.shape, weight=q, weight_scale=s)
    prepare_fp8_layer_for_marlin(h, size_k_first=False)
    return dict(marlin_w=h.weight.data, marlin_s=h.weight_scale.data,
                marlin_ws=getattr(h, "workspace", None))  # 0.30.0 sets one; nightly shares


def fp4_quantize(w):
    """BF16 [N, K] -> (packed u8 [N, K/2], e4m3 [N, K/16] linear layout, g) with
    w ~= e2m1 * sf / g (vLLM scaled_fp4_quant, ModelOpt convention)."""
    g = torch.tensor([FP8_MAX * FP4_MAX], dtype=torch.float32, device=w.device) / (
        w.float().abs().amax().clamp_min(1e-12))
    wq, wsf = _OPS.scaled_fp4_quant(w, g, is_sf_swizzled_layout=False)
    return wq, wsf.view(torch.float8_e4m3fn).reshape(w.shape[0], w.shape[1] // 16), g


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_dequant(wq, wsf, g):
    """Exact fp32 value of fp4_quantize's output (element 2i = low nibble)."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=wq.device)
    vals = torch.stack((lut[(wq & 0xF).long()], lut[(wq >> 4).long()]), -1).reshape(wq.shape[0], -1)
    return (vals.view(wq.shape[0], -1, 16) * (wsf.float() / g)[..., None]).reshape(wq.shape[0], -1)


def marlin_fp4_pack(wq, wsf, g):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin)

    _APPLY["fp4"] = apply_fp4_marlin_linear
    h = _holder(wq.shape[0], wq.shape[1] * 2, weight=wq, weight_scale=wsf,
                weight_global_scale=(1.0 / g).reshape(()))
    prepare_fp4_layer_for_marlin(h)
    return dict(marlin_w=h.weight.data, marlin_s=h.weight_scale.data,
                marlin_gs=h.weight_global_scale.data, marlin_ws=getattr(h, "workspace", None))


def marlin_linear(x2, mod):
    """x2 [M, K] bf16 through the module's Marlin tensors -> [M, N]."""
    n, k = mod._sfx_fp8d_nk
    if getattr(mod, "marlin_gs", None) is not None:
        return _APPLY["fp4"](x2, mod.marlin_w, mod.marlin_s, mod.marlin_gs, mod.marlin_ws, n, k, None)
    return _APPLY["fp8"](x2, mod.marlin_w, mod.marlin_s, mod.marlin_ws, n, k, None)


def ours_apply(x2, mod):
    """Routed backends ours / best / ours_fp4: x2 [M, K] -> [M, N] on the path the
    load-time table picked for this M bucket."""
    path, plan = mod._sfx_ours.lookup(x2.shape[0])
    if path == "cutlass":
        return fp8_linear(x2, mod.weight, mod.weight_scale)
    if path == "marlin":
        return marlin_linear(x2, mod)
    if x2.stride(-1) != 1 or x2.data_ptr() % 16 or (x2.shape[0] > 1 and x2.stride(0) % 16):
        x2 = x2.clone(memory_format=torch.contiguous_format)  # keep one specialization
    sem = None
    if plan.n_split > 1 and not plan.two:
        stream = torch.cuda.current_stream(x2.device).cuda_stream
        sem = _OURS.sem_for(x2.device, stream)
        if sem is None:  # more streams than ours.N_SEMS: same tile and split, two-stage
            if stream not in _NO_SEM:
                _NO_SEM.add(stream)
                _log(f"stream {stream:#x} found no free split-K counters (ours.N_SEMS = "
                     f"{_OURS.N_SEMS}): its last-CTA plans run two-stage")
            plan = plan.two_stage()
    return _OURS.run(x2, mod._sfx_ours_w, plan, sem)


def _op_impl(x, n: int, layer: str):
    """torch.ops.suffix_fp8d.linear: the per-call M route of the routed backends (opaque
    to Dynamo: vLLM traces once at max_num_batched_tokens with guards skipped)."""
    mod = _LAYERS[layer]
    if mod._sfx_fp8d != "auto":
        return ours_apply(x, mod)
    if x.shape[0] <= mod._sfx_fp8d_max_m:
        return marlin_linear(x, mod)
    return fp8_linear(x, mod.weight, mod.weight_scale)


def _op_fake(x, n: int, layer: str):
    return x.new_empty((x.shape[0], n))


def register_op():
    """torch.ops.suffix_fp8d.linear(x, n, layer) (idempotent)."""
    global _LIB
    if _LIB is None:
        lib = torch.library.Library(OP_NS, "FRAGMENT")
        lib.define("linear(Tensor x, int n, str layer) -> Tensor")
        for key in ("CUDA", "CPU"):
            lib.impl("linear", _op_impl, key)
        lib._register_fake("linear", _op_fake)
        _LIB = lib
    return getattr(torch.ops, OP_NS).linear


def dense_linear(x, mod, bias=None):
    """The converted layer's forward for its backend (tag set at conversion)."""
    tag = getattr(mod, "_sfx_fp8d", "cutlass")
    if tag == "cutlass":
        return fp8_linear(x, mod.weight, mod.weight_scale, bias)
    n = mod._sfx_fp8d_nk[0]
    x2 = x.reshape(-1, x.shape[-1])
    out = register_op()(x2, n, mod._sfx_fp8d_key) if tag in ROUTED else marlin_linear(x2, mod)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], n)


class Fp8DenseNNLinear(nn.Linear):
    """Class swapped onto converted plain nn.Linear modules (MTP eh_proj)."""

    def forward(self, x):
        return dense_linear(x, self, self.bias)


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
                return dense_linear(x, layer, bias)

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


def _selftest_marlin(name, w, mod, bias, wq4=None) -> None:
    """Once per (backend, device, N, K): the Marlin path at ORACLE_MS vs BF16 and vs
    the exact dequantized weight it was built from, before the BF16 weight is freed.
    Fails closed."""
    fp4 = getattr(mod, "marlin_gs", None) is not None
    if fp4:
        deq = fp4_dequant(*wq4)
    else:  # Marlin reads the per-channel scales in the activation dtype
        deq = mod.weight_fp8_ref.float() * mod.weight_scale_ref.to(torch.bfloat16).float()[:, None]
    qb, kb = (FP4_QUANT_BOUND, FP4_KERNEL_BOUND) if fp4 else (QUANT_BOUND, KERNEL_BOUND)
    for m in ORACLE_MS:
        x = torch.randn(m, w.shape[1], device=w.device, dtype=torch.bfloat16)
        out = marlin_linear(x, mod).float()
        ref = nn.functional.linear(x, w).float()
        exact = x.float() @ deq.t()
        if bias is not None:
            out, ref, exact = out + bias.float(), ref + bias.float(), exact + bias.float()
        eq, ek = rel_frob(out, ref), rel_frob(out, exact)
        if not (bool(torch.isfinite(out).all()) and eq <= qb and ek <= kb):
            raise RuntimeError(
                f"[suffix {PATCH_NAME}] Marlin {'W4A16' if fp4 else 'W8A16'} self-test FAILED "
                f"on {name} {tuple(w.shape)} at M={m}: rel err vs bf16 {eq:.3e} (<= {qb}), "
                f"vs exact dequant {ek:.3e} (<= {kb})")


def _log(msg: str) -> None:
    print(f"[suffix {PATCH_NAME}] {msg}", file=sys.stderr, flush=True)


def _convert_ours(name, mod, w, bias, be, n, k):
    """Backends ours / best / ours_fp4 for one layer: quantize, attach, route (timed on
    the first layer of each (backend, device, N, K)). Returns (resident bytes, first)."""
    global _OURS
    if _OURS is None:
        from . import ours

        _OURS = ours
    rkey = (be, w.device, n, k)
    route = _ROUTES.get(rkey)
    if be == "ours_fp4":
        wq, wsf, g = fp4_quantize(w)
        wt = _OURS.Weights.fp4(wq, wsf, g)
        params = dict(weight=wq, ours_sf=wsf)
    else:
        q, s = quantize_weight(w)
        if route is None:  # cutlass serves above the last bucket: its own self-test
            _selftest(name, w, q, s, bias)
        wt = _OURS.Weights.fp8(q, s)
        params = dict(weight=q, weight_scale=wt.s)
    for attr, t in params.items():
        setattr(mod, attr, nn.Parameter(t, requires_grad=False))
    mod._sfx_fp8d_nk, mod._sfx_ours_w = (n, k), wt
    if be == "best" and (route is None or "marlin" in route.paths()):
        for attr, t in marlin_fp8_pack(q, s).items():
            setattr(mod, attr, None if t is None else nn.Parameter(t, requires_grad=False))
    first = route is None
    if first:
        extra = {}
        if be != "ours_fp4":
            extra["cutlass"] = lambda x: fp8_linear(x, q, s)
        if be == "best":
            extra["marlin"] = lambda x: marlin_linear(x, mod)
        route = _ROUTES[rkey] = _OURS.tune_route(
            wt, w.device, extra, None if be == "ours_fp4" else "cutlass",
            log=lambda line: _log(f"{name}: {line}"))
        if be == "best" and "marlin" not in route.paths():
            for attr in ("marlin_w", "marlin_s", "marlin_ws"):
                delattr(mod, attr)
    mod._sfx_ours = route
    kept = [getattr(mod, a, None) for a in ("marlin_w", "marlin_s", "marlin_ws")]
    return wt.nbytes() + sum(t.numel() * t.element_size() for t in kept if t is not None), first


def _oracle_ours(name, w, mod) -> None:
    """Once per (backend, device, N, K), counters in place: the routed path through
    ours_apply at ORACLE_MS vs BF16 and vs the exact reference of the path it took, before
    the BF16 weight is freed. Fails closed."""
    wt = mod._sfx_ours_w
    deq = _OURS.dequant(wt)
    qb = FP4_QUANT_BOUND if wt.w4 else QUANT_BOUND
    for m in ORACLE_MS:
        x = torch.randn(m, w.shape[1], device=w.device, dtype=torch.bfloat16)
        path, plan = mod._sfx_ours.lookup(m)
        out = ours_apply(x, mod)
        if path == "ours":
            ek, kb = _OURS.rel_err(out, _OURS.reference(x, deq, plan.cfg)), _OURS.bound(plan.cfg)
        elif path == "cutlass":
            ek, kb = rel_frob(out, reference(x, mod.weight, mod.weight_scale)), KERNEL_BOUND
        else:  # Marlin reads the channel scales in the activation dtype
            mdeq = mod.weight.float() * mod.weight_scale.to(torch.bfloat16).float()[:, None]
            ek, kb = rel_frob(out, x.float() @ mdeq.t()), KERNEL_BOUND
        eq = rel_frob(out, nn.functional.linear(x, w))
        if not (bool(torch.isfinite(out).all()) and eq <= qb and ek <= kb):
            raise RuntimeError(
                f"[suffix {PATCH_NAME}] {mod._sfx_fp8d} oracle FAILED on {name} "
                f"{tuple(w.shape)} at M={m} ({plan.tag() if plan else path}): rel err vs "
                f"bf16 {eq:.3e} (<= {qb}), vs exact {ek:.3e} (<= {kb})")


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
    be = backend()
    max_m = marlin_max_m() if be == "auto" else 0
    if be in ROUTED:
        register_op()  # here, not inside the traced forward
    method = method_cls()()
    done, skipped, bf16_bytes, kept_bytes = [], [], 0, 0
    pending = []  # (name, bf16 weight, module): routed-path oracle once counters exist
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
        bias_d = None if bias is None else bias.data
        key = (be, w.device, n, k)
        if be in OURS_BACKENDS:
            nbytes, first = _convert_ours(name, mod, w.data, bias_d, be, n, k)
            kept_bytes += nbytes
            if first:
                pending.append((name, w.data, mod))
        else:
            if be == "marlin_fp4":
                wq4 = fp4_quantize(w.data)
                packed = marlin_fp4_pack(*wq4)
            else:
                q, s = quantize_weight(w.data)
                if be != "marlin" and key not in _SELFTESTED:
                    _selftest(name, w.data, q, s, bias_d)
                packed = marlin_fp8_pack(q, s) if be in ("marlin", "auto") else {}
            if packed:
                for attr, t in packed.items():
                    setattr(mod, attr, None if t is None else nn.Parameter(t, requires_grad=False))
                mod._sfx_fp8d_nk = (n, k)
                if key not in _SELFTESTED:
                    if be != "marlin_fp4":
                        mod.weight_fp8_ref, mod.weight_scale_ref = q, s
                    _selftest_marlin(name, w.data, mod, bias_d, wq4 if be == "marlin_fp4" else None)
                    for ref_attr in ("weight_fp8_ref", "weight_scale_ref"):
                        mod.__dict__.pop(ref_attr, None)
            if be in ("cutlass", "auto"):
                mod.weight = nn.Parameter(q, requires_grad=False)
                mod.weight_scale = nn.Parameter(s, requires_grad=False)
                kept_bytes += q.numel() + s.numel() * 4
            else:  # Marlin only: the repacked tensor IS the weight (no second copy)
                mod.weight = mod.marlin_w
            kept_bytes += sum(t.numel() * t.element_size() for t in packed.values() if t is not None)
        _SELFTESTED.add(key)
        bf16_bytes += w.numel() * 2
        mod._sfx_fp8d = be
        if be == "auto":
            mod._sfx_fp8d_max_m = max_m
        if be in ROUTED:
            mod._sfx_fp8d_key = f"{name}#{len(_LAYERS)}"
            _LAYERS[mod._sfx_fp8d_key] = mod
        if kind == "linear":
            mod.quant_method = method
        else:
            mod.__class__ = Fp8DenseNNLinear
        done.append((name, n, k))
    banks = ""
    if be in OURS_BACKENDS:
        _OURS.ensure_sems(dev)
        need = max([0] + [r.ws(1, rk[2]) for rk, r in _ROUTES.items() if rk[:2] == (be, dev)])
        banks = (f" + split-K counters {_OURS.N_SEMS} streams x {_OURS.SEM_SLOTS * 4 >> 10} KiB"
                 f", partials <= {need * 4 / 2**20:.1f} MiB per call (caching allocator)")
        while pending:  # pop: no reference to a freed BF16 weight survives
            _oracle_ours(*pending.pop())
    torch.cuda.empty_cache()
    shapes = sorted({(n, k) for _, n, k in done})
    mode = {"cutlass": "FP8 W8A8 (cutlass)", "marlin": "FP8 W8A16 (Marlin, every M)",
            "auto": f"FP8 W8A16 Marlin M<={max_m} / W8A8 cutlass above",
            "marlin_fp4": "NVFP4 W4A16 (Marlin, every M)",
            "ours": "FP8 ours W8A16/W8A8 | cutlass routed per (shape, M bucket)",
            "best": "FP8 ours W8A16/W8A8 | cutlass | Marlin W8A16 routed per (shape, M bucket)",
            "ours_fp4": "NVFP4 W4A16 ours (every M)"}[be]
    print(f"[suffix {PATCH_NAME}] converted {len(done)} linears to {mode} on "
          f"{dev} ({bf16_bytes / 2**30:.2f} GiB bf16 -> "
          f"{kept_bytes / 2**30:.2f} GiB resident{banks}; shapes N x K {shapes}; rev "
          f"{PATCH_REVISION})" + (f"; skipped {skipped}" if skipped else ""),
          file=sys.stderr, flush=True)
    return done
