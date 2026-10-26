# SPDX-License-Identifier: Apache-2.0
"""BF16 skinny GEMM y[M, N] = x[M, K] @ W[N, K]^T for decode-sized M (Triton,
tensor cores via tl.dot with M padded to 16, fp32 accumulation, one bf16
rounding at the end — the same precision contract as cuBLAS' bf16 GEMM).

Target: the unquantized BF16 projections of LFM2 / LFM2-MoE short-conv
layers (in_proj 2048 -> 6144, out_proj 2048 -> 2048) where cuBLAS picks a
16x16 wmma GEMM (~77% of DRAM bandwidth) or gemv (~45%) at M <= 32. The
kernel streams W once with a deep cp.async pipeline (num_stages) over
BLOCK_N-row stripes, one program per stripe (no split-K, no reduction
launch, no atomics: deterministic and batch-invariant in M).

Gate: ``SUFFIX_BF16_SKINNY=1`` (default OFF) with
``SUFFIX_BF16_SKINNY_LAYERS`` = comma fnmatch globs over the module prefix
(default ``*.short_conv.in_proj,*.short_conv.out_proj``) and
``SUFFIX_BF16_SKINNY_MAX_M`` (default 32). Wiring: the allowlisted layers'
``UnquantizedLinearMethod`` instances are wrapped after weight loading (the
load-time oracle compares against F.linear on the layer's REAL weight at
M = 1, 3, 16, max_m: rel. error <= 1e-2 and max |d| within 2 bf16 ulps of
the reference magnitude). M > max_m and anything else runs F.linear.

Helion mode (``SUFFIX_HELION_BF16_SKINNY=auto|helion``; unset / ``oxide`` = the
Triton lane above, unchanged): the router linears of Qwen3-Next / Qwen4Exp MoE blocks
(``mlp.gate`` 2560 -> 512 replicated, ``mlp.shared_expert_gate`` 2560 -> 1), main
model and MTP drafter, on helion_kernels/bf16_skinny.py (deterministic two-stage
split-K, fp32 accumulate, one bf16 rounding). Per layer and M bucket the shipped
tuned config serves; a bucket without one stays on F.linear (``auto``) or is a
startup error (``helion``). ``SUFFIX_BF16_SKINNY_MAX_M`` may go to 192 here (MTP k=4
verify at c32 = 160 rows per rank). Wiring: the ``register`` general-plugin entry
point wraps ``UnquantizedLinearMethod.process_weights_after_loading`` (vLLM's
per-layer post-load call, main and drafter alike), so every allowlisted layer
(``layer.prefix`` vs the globs) is armed right after its weight is final:
  * load oracle on the REAL weight at every specialization the plan launches (per
    served bucket its first / last / an odd row count, rows spanning 1e-3..1e3):
    every element within the bf16-GEMM bound of the exact f64 product (2^-8 |ref| +
    2 K 2^-24 sum|x||w|: one bf16 rounding over any fp32 summation order);
  * CUDA-graph capture/replay bit-equality per (N, K, bucket) (mxfp8_gemm.capture_check);
  * K5: one line per layer naming which M range runs Helion (config bucket, split)
    and which stays on F.linear.
The serving call goes through ``torch.ops.suffix_bf16sk.linear`` (opaque to Dynamo,
so the M dispatch is never frozen by a trace). The first module forward after load
refuses to serve when no layer matched the globs, or when a later post-load pass
swapped an armed layer's quant method (fail closed); otherwise it logs ACTIVE.
"""
from __future__ import annotations

import fnmatch
import os
import sys

GATE = "SUFFIX_BF16_SKINNY"
LAYERS_ENV = "SUFFIX_BF16_SKINNY_LAYERS"
MAX_M_ENV = "SUFFIX_BF16_SKINNY_MAX_M"
HELION_ENV = "SUFFIX_HELION_BF16_SKINNY"  # oxide (default) = the Triton lane | auto | helion
OP_NS = "suffix_bf16sk"
DEFAULT_LAYERS = "*.short_conv.in_proj,*.short_conv.out_proj"
MARKER = "[suffix bf16-skinny]"

_state = {"armed": False, "layers": 0, "oracle": [], "launch_logged": False}
_kernel = None
# Helion mode: prefix -> (module, quant method, original apply, weight, plan)
_helion = {"layers": {}, "matched": 0, "captured": set(), "hook": None, "checked": False}
_LIB = None


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def helion_mode() -> str:
    """oxide (the Triton lane, default) | auto | helion (helion_kernels.backend)."""
    from suffix_hybrid.helion_kernels import backend
    return backend(HELION_ENV)


def max_m() -> int:
    cap = 64 if helion_mode() == "oxide" else 192
    v = int(os.environ.get(MAX_M_ENV, "32"))
    if not 1 <= v <= cap:
        raise ValueError(f"{MAX_M_ENV}={v} outside [1, {cap}]")
    return v


def layer_globs() -> list[str]:
    raw = os.environ.get(LAYERS_ENV, "").strip() or DEFAULT_LAYERS
    return [g.strip() for g in raw.split(",") if g.strip()]


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _skinny(x_ptr, w_ptr, y_ptr, M, N, K, s_xm, s_wn, s_ym,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm < M
        mn = rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            mk = kk < K
            x = tl.load(x_ptr + rm[:, None] * s_xm + kk[None, :], mask=mm[:, None] & mk[None, :],
                        other=0.0)
            w = tl.load(w_ptr + rn[:, None] * s_wn + kk[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        tl.store(y_ptr + rm[:, None] * s_ym + rn[None, :], acc.to(tl.bfloat16),
                 mask=mm[:, None] & mn[None, :])

    _kernel = _skinny
    return _kernel


def plan(m: int, n: int, k: int) -> dict:
    """Tile per shape: BN rows of W per program; aim for >= ~1.5 programs per
    SM on 188-SM parts while keeping each W stripe >= 32 KB."""
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    bn = 32 if n >= 4096 else 16
    return dict(BM=bm, BN=bn, BK=256, num_warps=4, num_stages=4)


def skinny(x, w, out=None):
    import torch
    import triton
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    p = plan(m, n, k)
    _get_kernel()[(triton.cdiv(n, p["BN"]),)](
        x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        BM=p["BM"], BN=p["BN"], BK=p["BK"], num_warps=p["num_warps"], num_stages=p["num_stages"])
    return out


def check(y, ref) -> tuple[bool, str]:
    rel = ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
    dmax = (y.float() - ref.float()).abs().max().item()
    # 2 bf16 ulps of the largest reference magnitude bounds two different fp32
    # summation orders each rounded once to bf16
    tol = 2 * 2.0 ** -7 * ref.float().abs().max().item()
    return rel <= 1e-2 and dmax <= tol, f"rel={rel:.2e} max|d|={dmax:.3e} (tol {tol:.3e})"


def layer_oracle(name, w, mm) -> str:
    import torch
    dev = w.device
    worst = ""
    for m in sorted({1, 3, min(16, mm), mm}):
        g = torch.Generator(device="cpu").manual_seed(m)
        x = torch.randn(m, w.shape[1], generator=g).to(dev, torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        ok, msg = check(skinny(x, w), ref)
        if not ok:
            raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL {name} M={m}: {msg} — "
                               f"refusing to serve with {GATE}=1")
        worst = msg
    return worst


def install(model) -> int:
    """Wrap the allowlisted unquantized BF16 linears' quant_method.apply (called
    after weight loading by the plugin's post-load hook)."""
    import torch
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    globs, mm, n = layer_globs(), max_m(), 0
    if helion_mode() != "oxide":
        for name, mod in model.named_modules():
            if isinstance(mod, LinearBase) and any(fnmatch.fnmatch(name, g) for g in globs):
                n += install_helion_layer(name, mod)
        if not _helion["matched"]:
            raise RuntimeError(f"{MARKER} {GATE}=1 but no layer matched {globs}")
        return n
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase) or not any(fnmatch.fnmatch(name, g) for g in globs):
            continue
        qm = mod.quant_method
        w = getattr(mod, "weight", None)
        if (type(qm) is not UnquantizedLinearMethod or w is None or w.dtype != torch.bfloat16
                or getattr(mod, "bias", None) is not None or w.shape[1] % 8):
            _log(f"{name}: stays on F.linear ({type(qm).__name__}, "
                 f"{getattr(w, 'dtype', None)})")
            continue
        msg = layer_oracle(name, w, mm)
        orig = qm.apply

        def apply(layer, x, bias=None, _orig=orig, _w=w):
            if bias is None and x.dim() == 2 and 1 <= x.shape[0] <= mm and x.dtype == torch.bfloat16:
                return skinny(x, _w)
            return _orig(layer, x, bias)

        qm.apply = apply
        n += 1
        if n == 1:
            _log(f"LAYER ORACLE PASS {name} N={w.shape[0]} K={w.shape[1]} ({msg})")
    _state["layers"] = n
    _log(f"ACTIVE on {n} layers (M <= {mm}, globs {','.join(globs)})")
    if not n:
        raise RuntimeError(f"{MARKER} {GATE}=1 but no layer matched {globs}")
    return n


# ---------------------------------------------------------------------------
# Helion mode
# ---------------------------------------------------------------------------
def _op_impl(x, n: int, layer: str):
    """F.linear-equivalent for one armed layer: its plan's Helion entry for this M, else
    the layer's original apply (F.linear)."""
    mod, _qm, orig, w, plan = _helion["layers"][layer]
    m = x.shape[0]
    e = plan[m] if m < len(plan) else None
    if e is None:
        return orig(mod, x, None)
    from suffix_hybrid.helion_kernels import bf16_skinny as hb
    return hb.run(x, w, e)


def _op_fake(x, n: int, layer: str):
    return x.new_empty((x.shape[0], n))


def register_op():
    """torch.ops.suffix_bf16sk.linear(x, n, layer) (idempotent)."""
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


def helion_oracle(name: str, w, plan: list, rt=None) -> str:
    """Load gate for one layer (module docstring): exact bound at every oracle M, then
    capture/replay bit-equality once per (N, K, bucket). RuntimeError = refuse to serve."""
    import torch
    from suffix_hybrid.helion_kernels import bf16_skinny as hb
    from suffix_hybrid.helion_kernels.mxfp8_gemm import CudaRuntime, capture_check
    n, k = w.shape
    worst, base = "", ""
    for m in hb.oracle_ms(plan):
        g = torch.Generator(device="cpu").manual_seed(1000 + m)
        x = (torch.randn(m, k, generator=g) * torch.logspace(-3, 3, m)[:, None]
             if m > 1 else torch.randn(m, k, generator=g)).to(w.device, torch.bfloat16)
        ref = hb.exact(x, w)
        ok, msg = hb.check_exact(hb.run(x, w, plan[m], poison_nan=True), x, w, ref)
        if not ok:
            raise RuntimeError(f"{MARKER} HELION ORACLE FAIL {name} N={n} K={k} M={m}: {msg} "
                               f"— refusing to serve with {HELION_ENV}={helion_mode()}")
        worst = msg
        base = hb.check_exact(torch.nn.functional.linear(x, w), x, w, ref)[1]
    tops = {e[2]: m for m, e in enumerate(plan) if e is not None}  # bucket -> its top M
    for m in sorted(tops.values()):
        key = (n, k, plan[m][2])
        if key in _helion["captured"]:
            continue
        xs = [torch.randn(m, k, generator=torch.Generator(device="cpu").manual_seed(s))
              .to(w.device, torch.bfloat16) for s in range(3)]
        capture_check(lambda x, _e=plan[m]: hb.run(x, w, _e), xs, rt or CudaRuntime,
                      what=f"{name} N={n} K={k} M={m}")
        _helion["captured"].add(key)
    return f"ours: {worst}; F.linear: {base}"


def install_helion_layer(name: str, mod, rt=None) -> int:
    """Arm one allowlisted layer (idempotent). 1 = Helion serves some M, 0 = it stays
    on F.linear (logged: wrong method / dtype / bias, or no shipped config in auto)."""
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from suffix_hybrid.helion_kernels import bf16_skinny as hb
    if getattr(mod, "_sfx_bf16_skinny", None) is not None:
        return 0
    mod._sfx_bf16_skinny = name
    _helion["matched"] += 1
    _arm_first_forward_check()
    qm, w = mod.quant_method, getattr(mod, "weight", None)
    on_device = w is not None and (w.is_cuda or rt is not None)  # rt: CPU tests only
    if (type(qm) is not UnquantizedLinearMethod or not on_device or w.dtype != torch.bfloat16
            or w.dim() != 2 or w.stride(1) != 1 or getattr(mod, "bias", None) is not None):
        _log(f"{name}: stays on F.linear ({type(qm).__name__}, {getattr(w, 'dtype', None)}, "
             f"bias={getattr(mod, 'bias', None) is not None})")
        return 0
    w = w.data
    n, k = w.shape
    plan = hb.prepare(n, k, max_m(), helion_mode())
    if not any(plan[1:]):
        _log(f"{name} N={n} K={k}: stays on F.linear (no shipped {hb.KERNEL} config on "
             "this GPU)")
        return 0
    msg = helion_oracle(name, w, plan, rt)
    op, orig = register_op(), qm.apply

    def apply(layer, x, bias=None, _name=name, _orig=orig, _n=n):
        if (bias is None and x.dim() == 2 and x.dtype == torch.bfloat16
                and x.stride(-1) == 1):
            return op(x, _n, _name)
        return _orig(layer, x, bias)

    qm.apply = apply
    _helion["layers"][name] = (mod, qm, orig, w, plan)
    _log(f"{name} N={n} K={k}: {hb.describe(plan)}, M>{len(plan) - 1}:F.linear "
         f"(LAYER ORACLE PASS, {msg})")
    return 1


def _first_forward_check(module, args):
    """Global forward pre-hook, armed by the first post-load call: fires on the first
    module call after load (profile run, eager, before graph capture), removes itself,
    refuses to serve on a misconfiguration, else logs the ACTIVE summary."""
    h, _helion["hook"] = _helion["hook"], None
    if h is not None:
        h.remove()
    if _helion["checked"]:
        return
    _helion["checked"] = True
    if not _helion["matched"]:
        raise RuntimeError(f"{MARKER} {GATE}=1 but no loaded layer matched {layer_globs()} "
                           f"({HELION_ENV}={helion_mode()}): refusing to serve a silent arm")
    swapped = [name for name, (mod, qm, _orig, w, _plan) in _helion["layers"].items()
               if mod.quant_method is not qm or mod.weight.data_ptr() != w.data_ptr()]
    if swapped:
        raise RuntimeError(f"{MARKER} {len(swapped)} armed layers had their quant method or "
                           f"weight replaced after arming (first: {swapped[0]}): a later "
                           f"post-load pass converts them — fix {LAYERS_ENV}")
    served = len(_helion["layers"])
    _log(f"BF16-SKINNY {'ACTIVE' if served else 'INERT'}: {served} of {_helion['matched']} "
         f"matched layers on Helion ({HELION_ENV}={helion_mode()}, M<={max_m()}, globs "
         f"{','.join(layer_globs())})")


def _arm_first_forward_check() -> None:
    import torch
    if _helion["hook"] is None and not _helion["checked"]:
        _helion["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
            _first_forward_check)


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_BF16_SKINNY=1 with
    SUFFIX_HELION_BF16_SKINNY=auto|helion (the Triton lane keeps its own wiring)."""
    if not gate_on() or helion_mode() == "oxide":
        return None
    if _state["armed"]:
        return _state
    max_m()
    globs = layer_globs()  # malformed env: fail at startup
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    post = UnquantizedLinearMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer, _post=post):
        _post(self, layer)
        _arm_first_forward_check()  # this process loads a model: check it at first forward
        name = getattr(layer, "prefix", "") or ""
        if name and any(fnmatch.fnmatch(name, g) for g in globs):
            install_helion_layer(name, layer)

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    _state["armed"] = True
    _log(f"armed: {HELION_ENV}={helion_mode()}, M<={max_m()}, globs {','.join(globs)}")
    return _state


def bench(shapes=((6144, 2048), (2048, 2048), (128000, 2048)), ms=(1, 2, 4, 8, 16, 32),
          iters=200, layers=30) -> dict:
    """Graph-replayed us per GEMM (cold-ish: `layers` distinct weights per
    graph so W is not L2-resident) vs cuBLAS (F.linear) + achieved GB/s."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    for n, k in shapes:
        nl = layers if n * k * 2 * layers < 6e9 else 2
        ws = [torch.randn(n, k, device=dev).bfloat16() for _ in range(nl)]
        for m in ms:
            x = torch.randn(m, k, device=dev).bfloat16()
            outs = {}
            for name, fn in (("cublas", lambda w: torch.nn.functional.linear(x, w)),
                             ("ours", lambda w: skinny(x, w))):
                for w in ws:
                    fn(w)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                    for w in ws:
                        fn(w)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                outs[name] = e0.elapsed_time(e1) * 1e3 / iters / nl
            gbs = n * k * 2 / (outs["ours"] * 1e-6) / 1e9
            ok, msg = check(skinny(x, ws[0]), torch.nn.functional.linear(x, ws[0]))
            res[(n, k, m)] = outs
            _log(f"bench N={n} K={k} M={m}: cublas {outs['cublas']:.2f} us, ours {outs['ours']:.2f} us "
                 f"(x{outs['ours'] / outs['cublas']:.2f}, {gbs:.0f} GB/s) {msg} {'OK' if ok else 'FAIL'}")
        del ws
        torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    try:
        bench()
        return 0
    except Exception as exc:  # noqa: BLE001
        _log(f"BENCH FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
