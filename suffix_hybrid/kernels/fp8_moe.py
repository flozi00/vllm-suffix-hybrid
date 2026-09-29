# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scaled routed-experts decode kernel (qwen3.8-flash-next-nvfp4 MTP
draft layer ``mtp.layers.0.mlp.experts``: 512 experts, EP=2 -> 256 local per
rank, top-10, hidden 2560, expert intermediate 640, silu; e4m3 weights with
128x128 f32 block scales, dynamic per-token 1x128-group activation scales)
— reference, vLLM wiring, on-silicon oracle + bench.

Kernel: kernels-oxide/fp8_moe (cuda-oxide -> PTX .target sm_120a -> ptxas
13.0 SASS, ``mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32``). Host op
``_native.fp8_moe_cuda``; same six-launch skeleton as nvfp4_moe (route,
quant x, fc1+silu_and_mul, quant h, fc2 * topk_w, combine), grids a function
of M only, torch's current stream, no host sync.

Numerical contract (the f64 spec ``moe_ref`` below; the kernel matches it up
to f32 accumulation):
  x  -> e4m3 per (token, 128 cols): vLLM per_token_group_quant_fp8 (CUDA op):
        amax = max(1e-10, max|x|), s = amax / 448 (IEEE div), q =
        e4m3_rne(clamp(x / s, +-448)); UE8M0 mode: s = 2^ceil(log2(max(s,
        1e-10))) (vLLM when DeepGEMM E8M0 is on; weights are then requantized
        to power-of-two block scales by vLLM itself).
  fc1   per 128-K block: acc += dot(xq, wq) * s_x[tok, kb] * s_w[n/128, kb]
        (Triton fused_moe_kernel order), f32; h = silu(gate) * up in F32
        (w13 rows [0, I) = gate (w1), [I, 2I) = up (w3): vLLM silu_and_mul).
  h  -> e4m3 per (pair, 128 cols): non-UE8M0 = vLLM
        silu_and_mul_per_block_quant (Triton path's fused op): s =
        max(amax / 448, 1/(448*512)), no eps; UE8M0 = per_token_group_quant.
  fc2   same block rescale, y = topk_w * acc (f32); combine sum_k in fixed
        ascending k order (f32) -> bf16.
Deliberate difference from vLLM's Triton path (same choice as nvfp4_moe):
Triton stores the fc1 output in bf16 before silu_and_mul, the fc2 output in
bf16 and sums bf16; we keep f32 until the final bf16 store. `vllm=True` in
the reference emulates vLLM's numerics so the oracle attributes the
ours-vs-vLLM distance instead of assuming it.

Serving (gate ``SUFFIX_FP8_MOE=1``, default OFF; entry point
``suffix_fp8_moe``): shares nvfp4_moe's single OOT class
(``RoutedExperts.register_oot`` runs once, whichever gate arms first), so
``SUFFIX_NVFP4_MOE=1`` and ``SUFFIX_FP8_MOE=1`` can be on together: FP8
block layers go here, NVFP4 layers to nvfp4_moe. After vLLM's own weight
processing each eligible layer runs a LOAD-time LAYER ORACLE on its real
weights: ours vs the parent ``forward_modular`` vs the f64 spec + per-stage
readback, fatal on mismatch. Decode-sized calls (1 <= M <=
SUFFIX_FP8_MOE_MAX_M, default 32) run ours; everything else calls the parent
unchanged. Gate on but no FP8 layer engaged -> startup error naming why.

Weight scales: the Fp8MoEMethod checkpoint layout f32 [E, N/128, K/128]. The
DEEPGEMM backend's process_weights_after_loading may re-layout (TMA-aligned
MN-major) or pack them (UE8M0 int32) — the kernel needs the logical f32
tensor, so the OOT class snapshots the pre-processing scale tensors (vLLM's
UE8M0 requant rewrites weights AND those scales in place, before packing)
and uses whichever of post/pre is the logical layout.

CLI (in-pod, SM120 + oxide bundle):
  python -m suffix_hybrid.kernels.fp8_moe oracle   # ours vs vLLM Triton vs f64 ref, per stage
  python -m suffix_hybrid.kernels.fp8_moe bench    # us: ours vs vLLM fused_experts (CUDA graphs)
"""
from __future__ import annotations

import os
import sys
import zlib

from suffix_hybrid.kernels import nvfp4_moe as nm

GATE = "SUFFIX_FP8_MOE"
MAX_M_ENV = "SUFFIX_FP8_MOE_MAX_M"
MARKER = "[suffix fp8-moe]"
FAMILY = "fp8_moe"
NATIVE_FN = "fp8_moe_cuda"
FP8_MAX = 448.0
EPS = 1e-10
MIN_SCALE = 1.0 / (448.0 * 512.0)
BACKENDS = ("TRITON", "DEEPGEMM")  # vLLM Fp8MoeBackend names whose layout we read
# (E global, H, I per rank, topk, ep_size, ep_rank)
QWEN_DRAFT_EP0 = (512, 2560, 640, 10, 2, 0)
QWEN_DRAFT_EP1 = (512, 2560, 640, 10, 2, 1)
_ws: dict = {}
_active_logged = [False]


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def max_m() -> int:
    v = int(os.environ.get(MAX_M_ENV, "32"))
    if not 1 <= v <= 256:
        raise ValueError(f"{MAX_M_ENV}={v} outside [1, 256]")
    return v


# ---------------------------------------------------------------------------
# reference (torch, any device): the FP8 block spec our kernel implements
# ---------------------------------------------------------------------------
def quant_params(ue8m0: bool, stage: str) -> tuple:
    """(eps on amax, scale floor) of vLLM's activation quant: x always
    per_token_group_quant_fp8; h the Triton path's fused
    silu_and_mul_per_block_quant unless UE8M0 (then per_token_group)."""
    return (EPS, 0.0) if stage == "x" or ue8m0 else (0.0, MIN_SCALE)


def pow2_ceil(s):
    """Smallest power of two >= s (s > 0), exactly: 2^ceil(log2 s)."""
    import torch
    m, e = torch.frexp(s)
    return torch.where(m == 0.5, s, torch.ldexp(torch.ones_like(s), e))


def quant_fp8(x, ue8m0: bool = False, stage: str = "x"):
    """f32 [R, K] -> (e4m3 bits uint8 [R, K], f32 scales [R, K/128]); the
    kernel's moe_quant_rows == vLLM per-token-group FP8 quant."""
    import torch
    eps, smin = quant_params(ue8m0, stage)
    r, k = x.shape
    g = x.float().reshape(r, k // 128, 128)
    amax = g.abs().amax(-1).clamp_min(eps)
    # tensor / tensor = IEEE division like the kernel (div.rn) and vLLM's CUDA
    # quant; tensor / python-scalar is a reciprocal multiply in torch (1 ulp off
    # in ~half the groups -> every dequantized value of the group differs).
    s = amax / torch.full_like(amax, FP8_MAX)
    s = s.clamp_min(smin)
    if ue8m0:
        s = pow2_ceil(s.clamp_min(1e-10))
    q = (g / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(r, k).view(torch.uint8), s


def dequant(q, s):
    """(e4m3 bits / e4m3 [R, K], f32 [R, K/128]) -> f32 [R, K]."""
    import torch
    r, k = q.shape
    v = q.view(torch.float8_e4m3fn).float().reshape(r, k // 128, 128)
    return (v * s[..., None]).reshape(r, k)


def qdq(x, ue8m0=False, stage="x"):
    return dequant(*quant_fp8(x, ue8m0, stage))


def block_quant(w, ue8m0=False):
    """f32 [N, K] -> (e4m3 [N, K], f32 [N/128, K/128]): per 128x128 block
    s = amax/448 (power of two when ue8m0, like vLLM's UE8M0 requant)."""
    import torch
    n, k = w.shape
    b = w.float().reshape(n // 128, 128, k // 128, 128)
    s = b.abs().amax((1, 3)).clamp_min(1e-12) / FP8_MAX
    if ue8m0:
        s = pow2_ceil(s)
    q = (b / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(n, k), s


def expert_w(w, s, e: int):
    """Dequantized f32 [N, K] weight of expert e."""
    return w[e].float() * s[e].repeat_interleave(128, 0).repeat_interleave(128, 1)


def silu(x):
    import torch
    return x * torch.sigmoid(x)


def fc1_ref(p, x, ids, vllm=False):
    """f64 [P, I] intermediate h = silu(gate) * up per routed pair (ids
    LOCAL, -1 = off-rank). vllm=True: vLLM's Triton numerics (bf16 fc1
    output; bf16 act output in UE8M0 mode)."""
    import torch
    e_count, two_i = p["w13"].shape[0], p["w13"].shape[1]
    i = two_i // 2
    topk = ids.shape[1]
    xd = qdq(x.float(), p["ue8m0"]).double()
    h = torch.zeros(ids.numel(), i, dtype=torch.float64, device=x.device)
    for e, pairs in nm._groups(ids, e_count).items():
        gu = xd[pairs // topk] @ expert_w(p["w13"], p["w13_s"], e).double().T
        if vllm:
            gu = nm._bf(gu)
        a = silu(gu[:, :i]) * gu[:, i:]
        h[pairs] = nm._bf(a) if vllm and p["ue8m0"] else a
    return h


def moe_ref(p, x, ids, tw, vllm=False):
    """Exact f64 [M, H] routed-experts output of the quantized problem (ids
    LOCAL): the FP8 block spec with f64 arithmetic between the two
    activation quantizations (the kernel keeps f32 there)."""
    import torch
    m, topk = ids.shape
    e_count, hdim = p["w2"].shape[0], p["w2"].shape[1]
    hd = qdq(fc1_ref(p, x, ids, vllm).float(), p["ue8m0"], "h").double()
    y = torch.zeros(m * topk, hdim, dtype=torch.float64, device=x.device)
    for e, pairs in nm._groups(ids, e_count).items():
        yy = (hd[pairs] @ expert_w(p["w2"], p["w2_s"], e).double().T) * tw.reshape(-1)[pairs, None].double()
        y[pairs] = nm._bf(yy) if vllm else yy
    valid = ((ids >= 0) & (ids < e_count)).reshape(-1, 1).double()
    y = (y * valid).reshape(m, topk, hdim)
    out = torch.zeros(m, hdim, dtype=torch.float64, device=x.device)
    for k in range(topk):  # the kernel's fixed k order
        out += y[:, k]
    return out


def make_problem(e_count, hdim, idim, device="cpu", seed=0, wstd=0.05, ue8m0=False):
    """Random FP8 block-quantized routed-experts layer in vLLM's
    Fp8MoEMethod layout: w13 e4m3 [E, 2I, H] = [gate; up], w13_s f32
    [E, 2I/128, H/128], w2 e4m3 [E, H, I], w2_s f32 [E, H/128, I/128].
    Generated per expert on `device` (the qwen draft shape is 3.3 GB f32)."""
    import torch
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.empty(e_count, 2 * idim, hdim, dtype=torch.float8_e4m3fn, device=dev)
    w2 = torch.empty(e_count, hdim, idim, dtype=torch.float8_e4m3fn, device=dev)
    s13 = torch.empty(e_count, 2 * idim // 128, hdim // 128, device=dev)
    s2 = torch.empty(e_count, hdim // 128, idim // 128, device=dev)
    for e in range(e_count):
        for w, s, shape in ((w13, s13, (2 * idim, hdim)), (w2, s2, (hdim, idim))):
            w[e], s[e] = block_quant(torch.randn(*shape, generator=gen, device=dev) * wstd, ue8m0)
    return dict(w13=w13, w13_s=s13, w2=w2, w2_s=s2, ue8m0=bool(ue8m0), id_base=0,
                E_global=e_count)


REF_TOL = 3e-3  # ours vs the f64 spec, end to end
STAGE_TOL = {"xq": 1e-3, "fc1": 1e-4, "hq": 1e-3, "fc2": 1e-4, "comb": 1e-3}


def oracle_ok(rel_vs_ref: float, rel_vs_vllm: float, vllm_rel_vs_ref: float) -> bool:
    """Ours vs the f64 spec <= REF_TOL (floor: bf16 output rounding ~1.1e-3
    rms; a wrong block scale / layout costs >= 1e-2). vLLM's own distance to
    the spec (bf16 intermediates) is NOT a tolerance for us; ours vs vLLM
    only has to respect the triangle bound of both errors (floor 1e-2)."""
    return (rel_vs_ref <= REF_TOL
            and rel_vs_vllm <= max(1e-2, 1.1 * (rel_vs_ref + vllm_rel_vs_ref)))


def judge(ours, stock, ref, local_ids, emu=None):
    """(ok, metrics): oracle_ok, finite, and the EP zero contract (tokens
    with no local expert exactly 0 in ours AND in vLLM's output)."""
    import torch
    r_ref, r_st, st_ref = nm._rel(ours, ref), nm._rel(ours, stock), nm._rel(stock, ref)
    dead = (local_ids < 0).all(1)
    zero = bool((ours[dead] == 0).all() and (stock[dead] == 0).all())
    ok = oracle_ok(r_ref, r_st, st_ref) and zero and bool(torch.isfinite(ours).all())
    e = "" if emu is None else f"vllm_vs_vllm_emulation={nm._rel(stock, emu):.2e} "
    return ok, (f"rel_vs_ref={r_ref:.2e} rel_vs_vllm={r_st:.2e} vllm_rel_vs_ref={st_ref:.2e} "
                f"{e}nonlocal_tokens={int(dead.sum())} nonlocal_zero={zero}")


def stages(p, x, ids, tw, ws, out):
    """Per-stage drift of OUR kernel read back from its workspace (`ids`
    LOCAL), each stage vs the f64 spec fed with the kernel's own previous
    stage (see nvfp4_moe.stages): xq / hq / comb = fraction of values
    differing (same f32 ops, bit-exact expected), fc1 / fc2 = rel error."""
    import torch
    aq, a_s, inter, hq, h_s, y, _ = ws
    m, topk = ids.shape
    e_count, hdim, idim = p["w2"].shape
    pairs, ue = m * topk, p["ue8m0"]
    frac = lambda a, b: float((a != b).float().mean()) if a.numel() else 0.0
    xq = dequant(aq[:m * hdim].view(m, hdim), a_s[:m * hdim // 128].view(m, -1))
    got = {"xq": frac(xq, qdq(x.float(), ue))}
    vp = torch.nonzero(ids.reshape(-1) >= 0).reshape(-1)
    h = inter[:pairs * idim].view(pairs, idim)[vp]
    hd = dequant(hq[:pairs * idim].view(pairs, idim), h_s[:pairs * idim // 128].view(pairs, -1))
    h_ref = torch.zeros(pairs, idim, dtype=torch.float64, device=x.device)
    y_ref = torch.zeros(pairs, hdim, dtype=torch.float64, device=x.device)
    for e, pr in nm._groups(ids, e_count).items():
        gu = xq[pr // topk].double() @ expert_w(p["w13"], p["w13_s"], e).double().T
        h_ref[pr] = silu(gu[:, :idim]) * gu[:, idim:]
        y_ref[pr] = ((hd[pr].double() @ expert_w(p["w2"], p["w2_s"], e).double().T)
                     * tw.reshape(-1)[pr, None].double())
    got["fc1"] = nm._rel(h, h_ref[vp]) if vp.numel() else 0.0
    got["hq"] = frac(hd[vp], qdq(h, ue, "h"))
    yk = y[:pairs * hdim].view(pairs, hdim)
    got["fc2"] = nm._rel(yk[vp], y_ref[vp]) if vp.numel() else 0.0
    acc = torch.zeros(m, hdim, dtype=torch.float32, device=x.device)
    yk = yk.view(m, topk, hdim)
    for k in range(topk):
        acc = torch.where((ids[:, k] >= 0)[:, None], acc + yk[:, k], acc)
    got["comb"] = frac(out.float(), acc.bfloat16().float())
    ok = all(got[k] <= STAGE_TOL[k] for k in got)
    return ok, "stages " + " ".join(f"{k}={v:.1e}" for k, v in got.items())


# ---------------------------------------------------------------------------
# native launch (shared by serving, oracle, bench)
# ---------------------------------------------------------------------------
def workspace(dev, m: int, topk: int, hdim: int, idim: int, e_count: int, ws=None):
    """(aq, a_s, inter, hq, h_s, y, route) sized for M tokens; grown, never
    shrunk. Serving allocates it at LOAD time only."""
    import torch
    p = m * topk
    need = (m * hdim, m * hdim // 128, p * idim, p * idim, p * idim // 128, p * hdim,
            3 * e_count + p)
    dts = (torch.uint8, torch.float32, torch.float32, torch.uint8, torch.float32,
           torch.float32, torch.int32)
    if ws is not None and all(t.numel() >= n for t, n in zip(ws, need)):
        return ws
    old = ws or (None,) * 7
    return tuple(torch.empty(max(n, o.numel() if o is not None else 0), dtype=d, device=dev)
                 for n, d, o in zip(need, dts, old))


def run_ours(native, p, x, ids, tw, ws, stream, out=None):
    import torch
    if out is None:
        out = torch.empty(x.shape[0], p["w2"].shape[1], dtype=torch.bfloat16, device=x.device)
    native.fp8_moe_cuda(x, ids, tw, p["w13"], p["w13_s"], p["w2"], p["w2_s"], *ws, out,
                        bool(p["ue8m0"]), int(p.get("id_base", 0)), stream)
    return out


# ---------------------------------------------------------------------------
# vLLM wiring (the OOT class lives in nvfp4_moe._make_layer_cls)
# ---------------------------------------------------------------------------
def eligibility(info: dict) -> str | None:
    """Why an FP8 RoutedExperts layer cannot use our kernel, or None. Pure
    (plain values) so the CPU tests pin every branch."""
    if info.get("quant_dtype") != "fp8":
        return f"not FP8 e4m3 (quant_dtype={info.get('quant_dtype')})"
    if info.get("block_shape") != [128, 128]:
        return f"block_shape {info.get('block_shape')} (128x128 block scales only)"
    if info.get("backend") not in BACKENDS:
        return f"Fp8 MoE backend {info.get('backend')} (standard layout only: {BACKENDS})"
    if info.get("act") != "silu":
        return f"activation {info.get('act')} (silu only)"
    if info.get("clamp_limit") is not None or info.get("swiglu_alpha") is not None:
        return "swiglu clamp/alpha not supported"
    if info.get("router_weight_on_input"):
        return "apply_router_weight_on_input"
    if info.get("bias"):
        return "expert biases"
    if info.get("dp") != 1 or info.get("all2all"):
        return f"DP/all2all dispatch (dp={info.get('dp')})"
    if info.get("eplb"):
        return "EPLB (physical expert ids / redundant experts)"
    if info.get("mk_shared_overlap"):
        return "shared experts overlapped inside the modular kernel"
    if info.get("tp") != 1 and info.get("ep") != 1:
        return f"parallel tp/ep={info.get('tp')}/{info.get('ep')} (TP-sharded or EP, not both)"
    if (info.get("ep") != 1 or info.get("expert_map")) and info.get("ep_base") is None:
        return "expert_map is not the linear ep_rank*E_local placement"
    e, h, i = info.get("E", 0), info.get("H", 0), info.get("I", 0)
    if not (1 <= e <= 256) or h <= 0 or i <= 0 or h % 128 or i % 128:
        return f"shape E={e} H={h} I={i} (E<=256, H,I % 128)"
    if not info.get("w_exact", False):
        return "weights not contiguous e4m3 [E, 2I, H] / [E, H, I]"
    if info.get("scales"):
        return info["scales"]
    return None


def resolve_scale(post, pre, shape):
    """(logical f32 block-scale tensor, None) or (None, reason): vLLM's
    processed scale when still the f32 [E, N/128, K/128] layout (possibly
    strided), else the pre-processing snapshot (same values; UE8M0 requant
    rewrote it in place)."""
    import torch
    for t in (post, pre):
        if t is not None and t.dtype == torch.float32 and tuple(t.shape) == tuple(shape):
            return t.contiguous(), None
    got = None if post is None else (post.dtype, tuple(post.shape))
    return None, f"weight scales {got} not resolvable to f32 {tuple(shape)} block layout"


def is_fp8(layer) -> bool:
    import torch
    qc = getattr(layer.quant_method, "moe_quant_config", None)
    return getattr(qc, "quant_dtype", None) == torch.float8_e4m3fn


def e8m0_used() -> bool:
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    except ImportError:
        return False
    return bool(is_deep_gemm_e8m0_used())


def layer_info(layer, pre):
    import torch
    qm = layer.quant_method
    qc = getattr(qm, "moe_quant_config", None)
    mc = layer.moe_config
    pc = mc.moe_parallel_config
    qd = getattr(qc, "quant_dtype", None)
    info = dict(quant_dtype="fp8" if qd == torch.float8_e4m3fn else str(qd),
                block_shape=list(getattr(qc, "block_shape", None) or []),
                backend=getattr(getattr(qm, "fp8_backend", None), "name", None),
                act=getattr(layer.activation, "value", layer.activation),
                clamp_limit=getattr(qc, "gemm1_clamp_limit", None),
                swiglu_alpha=getattr(qc, "gemm1_alpha", None),
                router_weight_on_input=bool(layer.apply_router_weight_on_input),
                bias=getattr(qc, "w1_bias", None) is not None,
                tp=mc.tp_size, ep=mc.ep_size, dp=mc.dp_size,
                all2all=bool(pc.use_all2all_kernels), eplb=bool(pc.enable_eplb),
                mk_shared_overlap=bool(getattr(qm, "mk_can_overlap_shared_experts", False)),
                expert_map=layer.expert_map is not None)
    w13, w2 = getattr(layer, "w13_weight", None), getattr(layer, "w2_weight", None)
    if info["quant_dtype"] != "fp8" or w13 is None or w13.dim() != 3:
        return info, None
    e, two_i, h = w13.shape
    i = two_i // 2
    em = layer.expert_map
    info["ep_base"] = nm.ep_base(None if em is None else em.tolist(), pc.ep_rank, mc.ep_size,
                                 layer.global_num_experts, e)
    f8 = torch.float8_e4m3fn
    info.update(E=e, H=h, I=i, w_exact=(w13.dtype == f8 and w2.dtype == f8 and two_i == 2 * i
                                        and tuple(w2.shape) == (e, h, i)
                                        and w13.is_contiguous() and w2.is_contiguous()))
    if h % 128 or i % 128:
        return info, None
    s13, why13 = resolve_scale(qc.w1_scale, pre.get("w13"), (e, two_i // 128, h // 128))
    s2, why2 = resolve_scale(qc.w2_scale, pre.get("w2"), (e, h // 128, i // 128))
    info["scales"] = why13 or why2
    return info, (s13, s2)


def prepare(layer, pre, parent_forward, state):
    """LOAD time, right after vLLM's weight processing: eligibility ->
    cubin load -> workspace -> fatal layer oracle. Returns the layer cfg
    (forward_modular runs ours for 1 <= M <= max_m) or None (stock)."""
    import torch
    from suffix_hybrid import oxide_kernels
    info, sc = layer_info(layer, pre)
    state["seen"] += 1
    why = eligibility(info)
    if why is not None:
        state["stock"][why] = state["stock"].get(why, 0) + 1
        _log(f"layer {layer.layer_name} stays on vLLM's FP8 MoE path: {why}")
        return None
    dev = layer.w13_weight.device
    oxide_kernels.ensure_loaded(FAMILY, dev.index)
    cfg = dict(kind="fp8", w13=layer.w13_weight, w13_s=sc[0], w2=layer.w2_weight, w2_s=sc[1],
               ue8m0=e8m0_used(), H=info["H"], I=info["I"], E=info["E"], topk=layer.top_k,
               max_m=max_m(), id_base=info["ep_base"], E_global=layer.global_num_experts)
    _ws[dev] = workspace(dev, cfg["max_m"], cfg["topk"], info["H"], info["I"], info["E"],
                         _ws.get(dev))
    state["oracle"].append(layer_oracle(layer.layer_name, cfg, parent_forward))
    state["ours"] += 1
    return cfg


def run(native, cfg, x, topk_weights, topk_ids):
    """Serving forward (decode-sized call)."""
    if not _active_logged[0]:
        _active_logged[0] = True
        _log(f"FP8-MOE ACTIVE: first decode call on our kernel (ue8m0={cfg['ue8m0']})")
    return _launch(native, cfg, x, topk_weights, topk_ids)


def _launch(native, cfg, x, topk_weights, topk_ids):
    import torch
    dev = x.device
    tw = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
    return run_ours(native, cfg, x.contiguous(), topk_ids.contiguous(), tw.contiguous(),
                    _ws[dev], torch.cuda.current_stream(dev).cuda_stream)


def layer_oracle(name, cfg, parent_forward):
    """Ours vs the parent (vLLM's FP8 MoE path on this rank) vs the f64
    reference on this layer's REAL weights, GLOBAL routing ids; EP adds a
    batch whose tokens all route off-rank (must be exact 0). Each path gets
    its own clone of the inputs. Fatal on mismatch; returns max rel vs vLLM."""
    import torch
    from suffix_hybrid import oxide_kernels
    native = oxide_kernels.native()
    dev = cfg["w13"].device
    base, local, mm = cfg["id_base"], cfg["E"], cfg["max_m"]
    cases = [(m, None) for m in sorted({1, min(8, mm), mm})]
    if local < cfg["E_global"]:
        cases.append((min(8, mm), True))
    worst = worst_ref = 0.0
    st = ""
    for m, dead in cases:
        seed = zlib.crc32(str(name).encode()) % 10007 + m
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(m, cfg["H"], generator=gen).to(dev, torch.bfloat16)
        ids, tw = nm.rand_routing(m, cfg["E_global"], cfg["topk"], dev, seed, base, local, dead)
        x0 = x.clone()
        stock = parent_forward(x.clone(), tw.clone(), ids.clone()).float()
        out = _launch(native, cfg, x.clone(), tw.clone(), ids.clone())
        lid = nm.to_local(ids, base, local)
        st_ok, st = stages(cfg, x0, lid, tw, _ws[dev], out)
        ours, ref = out.float(), moe_ref(cfg, x0, lid, tw)
        ok, msg = judge(ours, stock, ref, lid, moe_ref(cfg, x0, lid, tw, vllm=True).bfloat16())
        if stock.norm() > 0:
            worst = max(worst, nm._rel(ours, stock))
            worst_ref = max(worst_ref, nm._rel(ours, ref))
        if not (ok and st_ok):
            raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL {name} M={m} id_base={base}: "
                               f"{msg} {st} ue8m0={cfg['ue8m0']} — refusing to serve with {GATE}=1")
    torch.cuda.synchronize(dev)
    _log(f"LAYER ORACLE PASS {name} E={local}/{cfg['E_global']} id_base={base} H={cfg['H']} "
         f"I={cfg['I']} top{cfg['topk']} ue8m0={cfg['ue8m0']} max_rel_vs_vllm={worst:.2e} "
         f"max_rel_vs_ref={worst_ref:.2e} (last case {st})")
    return worst


def verdict(seen: int, ours: int, stock: dict) -> str | None:
    """None when >= 1 FP8 layer runs our kernel, else the loud startup
    error. Decided at the first module forward (after the MTP drafter's
    weights were processed: vLLM loads the drafter before any forward)."""
    if ours > 0:
        return None
    if seen == 0:
        return (f"{MARKER} FP8-MOE NOT ENGAGED with {GATE}=1: no FP8 block-quantized "
                "RoutedExperts layer was processed through the OOT class")
    why = "; ".join(f"{k} x{v}" for k, v in sorted(stock.items()))
    return f"{MARKER} FP8-MOE NOT ENGAGED with {GATE}=1: every FP8 MoE layer ineligible: {why}"


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_FP8_MOE=1; arms
    nvfp4_moe's shared OOT class (registered once for both gates)."""
    if not gate_on():
        return None
    return nm.arm()


# ---------------------------------------------------------------------------
# standalone on-silicon oracle + bench (synthetic weights, one GPU = one rank)
# ---------------------------------------------------------------------------
def _native_ready():
    import torch
    from suffix_hybrid import oxide_kernels
    native = oxide_kernels.native()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("FP8 MoE cubin is sm_120a SASS (cc 12.x only)")
    oxide_kernels.ensure_loaded(FAMILY)
    return native


def _vllm_call(p, x, ids, tw):
    """vLLM's Triton FP8 block MoE (fused_experts, the functional form of
    TritonExperts) on this rank: GLOBAL ids + the linear expert_map."""
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    qc = fp8_w8a8_moe_quant_config(w1_scale=p["w13_s"], w2_scale=p["w2_s"], block_shape=[128, 128])
    return fused_experts(x, p["w13"], p["w2"], tw, ids, global_num_experts=p["E_global"],
                         expert_map=p["expert_map"], quant_config=qc)


def _dev_problem(dev, case, ue8m0, seed=0):
    import torch
    e_global, hdim, idim, topk, ep_size, ep_rank = case
    local = e_global // ep_size
    p = make_problem(local, hdim, idim, device=dev, seed=seed, ue8m0=ue8m0)
    base = ep_rank * local
    em = torch.full((e_global,), -1, dtype=torch.int32, device=dev)
    em[base:base + local] = torch.arange(local, dtype=torch.int32, device=dev)
    p.update(id_base=base, E_global=e_global, expert_map=em if ep_size > 1 else None)
    return p


def _case_name(case, ue8m0) -> str:
    e_global, hdim, idim, topk, ep_size, ep_rank = case
    ep = f" EP{ep_size} rank{ep_rank} ({e_global // ep_size} local)" if ep_size > 1 else ""
    return f"E={e_global} H={hdim} I={idim} top{topk}{ep} ue8m0={ue8m0}"


DRAFT_MS = (1, 2, 4, 8, 16, 32)  # decode tokens per draft step (running seqs)
ORACLE_CASES = ((QWEN_DRAFT_EP0, DRAFT_MS), (QWEN_DRAFT_EP1, DRAFT_MS),
                ((64, 1024, 384, 6, 1, 0), (1, 5, 17, 32)))
BENCH_CASES = ((QWEN_DRAFT_EP0, (1, 2, 4, 8, 16, 32, 64)),)


def oracle(cases=ORACLE_CASES):
    """Per case: ours vs vLLM Triton (same EP rank) vs the f64 reference,
    vLLM vs our emulation of its numerics, per-stage readback, determinism;
    EP cases add rows routed only off-rank plus an all-off-rank batch. The
    first case also runs in the other activation-scale mode (UE8M0 flipped)
    against the reference only. Fatal on mismatch."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev).cuda_stream
    mode = e8m0_used()
    lines = []
    runs = [(c, ms, mode) for c, ms in cases] + [(cases[0][0], (1, 8), not mode)]
    for case, ms, ue in runs:
        e_global, hdim, idim, topk, ep_size, _ = case
        p = _dev_problem(dev, case, ue)
        base, local = p["id_base"], e_global // ep_size
        ws = workspace(dev, max(ms), topk, hdim, idim, local)
        batches = [(m, None) for m in ms] + ([(8, True)] if local < e_global else [])
        for m, dead in batches:
            ids, tw = nm.rand_routing(m, e_global, topk, dev, seed=m, base=base, local=local,
                                      dead=dead)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            lid = nm.to_local(ids, base, local)
            ref = moe_ref(p, x, lid, tw)
            again = run_ours(native, p, x, ids, tw, ws, stream).float()
            out = run_ours(native, p, x, ids, tw, ws, stream)
            st_ok, st = stages(p, x, lid, tw, ws, out)
            ours = out.float()
            det = bool(torch.equal(ours, again))
            if ue == mode:
                stock = _vllm_call(p, x.clone(), ids.clone(), tw.clone()).float()
                ok, msg = judge(ours, stock, ref, lid, moe_ref(p, x, lid, tw, vllm=True).bfloat16())
            else:  # vLLM runs one mode per process: spec + zero contract only
                zero = bool((ours[(lid < 0).all(1)] == 0).all())
                r = nm._rel(ours, ref)
                ok, msg = r <= REF_TOL and zero, f"rel_vs_ref={r:.2e} nonlocal_zero={zero} (no vLLM)"
            ok = ok and det and st_ok
            lines.append(f"{_case_name(case, ue)} M={m}{' all-nonlocal' if dead else ''} "
                         f"P={m * topk}: {msg} {st} deterministic={det} {'OK' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"{MARKER} FP8-MOE ORACLE FAIL: {lines[-1]}")
        del p, ws
        torch.cuda.empty_cache()
    for ln in lines:
        _log(ln)
    return f"{MARKER} FP8-MOE ORACLE PASS ({len(lines)} cases, sm_120a e4m3 m16n8k32 mma)"


def bench(cases=BENCH_CASES, iters=200):
    """us per MoE layer call on one rank (x quant + experts + combine), CUDA
    graphs, ours vs vLLM fused_experts (Triton). The crossover M sets
    SUFFIX_FP8_MOE_MAX_M."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    mode = e8m0_used()
    for case, ms in cases:
        e_global, hdim, idim, topk, ep_size, _ = case
        p = _dev_problem(dev, case, mode, seed=1)
        base, local = p["id_base"], e_global // ep_size
        ws = workspace(dev, max(ms), topk, hdim, idim, local)
        per_expert = 3 * idim * hdim * (1 + 4 / 128 ** 2)
        for m in ms:
            ids, tw = nm.rand_routing(m, e_global, topk, dev, seed=100 + m, dead=False)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            out = torch.empty(m, hdim, dtype=torch.bfloat16, device=dev)
            t_v = nm._graph_us(lambda s: _vllm_call(p, x, ids, tw), dev, iters)
            t_o = nm._graph_us(lambda s: run_ours(native, p, x, ids, tw, ws, s.cuda_stream, out),
                               dev, iters)
            lid = nm.to_local(ids, base, local)
            distinct = int(torch.unique(lid[lid >= 0]).numel())
            roof = distinct * per_expert / nm.HBM_BPS * 1e6
            res[(case, m)] = (t_v, t_o, roof)
            _log(f"bench {_case_name(case, mode)} M={m} routed_rows={m * topk} "
                 f"local_experts_hit={distinct}: vllm {t_v:.1f} us, ours {t_o:.1f} us "
                 f"(x{t_o / t_v:.2f}), weight-roofline {roof:.1f} us")
        del p, ws
        torch.cuda.empty_cache()
    return res


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "both"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:
        print(f"{MARKER} FP8-MOE {mode.upper()} FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
