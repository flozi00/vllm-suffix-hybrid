# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-experts decode kernel (gemma-4-26b-a4b-nvfp4: 30 layers x 128
experts, top-8, hidden 2816, expert intermediate 704; GLM-5.3-NVFP4: 256
experts, top-8, hidden 6144, intermediate 2048, TP=8 + expert parallel ->
32 local experts per rank) — reference, vLLM wiring, on-silicon oracle + bench.

Kernel: kernels-oxide/nvfp4_moe (cuda-oxide -> PTX .target sm_120a -> ptxas
13.0 SASS, block-scaled mxf4nvf4 m16n8k64 mma). Host op
``_native.nvfp4_moe_cuda`` launches, on torch's stream, with grids that are a
function of M only (CUDA-graph safe, no host sync):
  route    one CTA: local id = topk_id - id_base (EP), active local experts
           ascending, pairs (p = token*topk + k) ascending per expert ->
           deterministic; pairs routed to other ranks' experts are skipped
  quant x  vLLM scaled_fp4_quant math with the layer's a1 gscale
  fc1      one CTA per (expert slot, 32 intermediate cols) over only that
           expert's routed rows (16-row mma chunks):
           inter[p, j] = act(g1[e] * gate_j) * g1[e] * up_j
           (w13 rows [0, I) = up (w3), [I, 2I) = gate (w1): vLLM's FI layout)
  quant h  a2 gscale
  fc2      y[p, :] = g2[e] * (h_p @ w2[e]^T) * topk_w[p]
  combine  out[t] = sum_k y[t*topk + k] in ascending k (fixed order) over
           local experts only; a token with no local expert gets exactly 0

Parallel contract (= vLLM's stock FLASHINFER_CUTLASS path, vLLM 0.30):
  EP (TP=N + --enable-expert-parallel, DP=1): moe tp=1, ep=N, linear
  expert_map (global g -> g - ep_rank*E_local on this rank, else -1);
  topk_ids reach forward_modular as GLOBAL ids and FlashInfer only computes
  experts in [ep_rank*E_local, +E_local) (flashinfer_cutlass_moe.py passes
  ep_size/ep_rank, not expert_map), contributing 0 for the rest. The
  partial output is all-reduced by MoERunner._maybe_reduce_final_output
  (no_dp_ep finalize: output_is_reduced() False), then routed_scaling_factor.
  We take id_base = ep_rank*E_local and require expert_map to be exactly
  that linear map (else: stock, with the reason).
  TP-sharded (no EP): weights hold I/tp columns; per-rank partial output,
  same all-reduce -> the kernel just runs at I = I/tp.
  DP/all2all, EPLB, MK-overlapped shared experts: stock (fail closed).

Serving (gate ``SUFFIX_NVFP4_MOE=1``, default OFF; entry point
``suffix_nvfp4_moe``): ``RoutedExperts`` is a vLLM PluggableLayer
(fused_moe/routed_experts.py:45) -> ``RoutedExperts.register_oot`` swaps in
``SuffixNvFp4RoutedExperts``. After vLLM's own weight processing (FlashInfer
CUTLASS layout) each eligible layer runs a LAYER ORACLE on its real weights:
ours vs the parent ``forward_modular`` (= vLLM's FlashInfer
cutlass_fused_moe path) vs the f64 spec reference + per-stage readback of our
workspace (`stages`), fatal on mismatch.
Shared experts: MoERunner runs them itself unless the modular kernel
overlaps them (prepare_finalize.supports_async); the latter is ineligible,
so ignoring the forward_modular shared_experts argument matches stock.
Decode-sized calls (1 <= M <= SUFFIX_NVFP4_MOE_MAX_M, default 32) run ours;
everything else calls the parent unchanged. Gated on but no layer engaged ->
startup error naming why.

CLI (in-pod, SM120 + oxide bundle):
  python -m suffix_hybrid.kernels.nvfp4_moe oracle   # ours vs FlashInfer vs f64 ref, per stage
  python -m suffix_hybrid.kernels.nvfp4_moe bench    # us: ours vs FlashInfer (CUDA graphs)
  (gemma + GLM-5.3 EP rank + GLM-5.3 TP=8 shard, synthetic weights, one GPU)
"""
from __future__ import annotations

import os
import sys
import zlib

GATE = "SUFFIX_NVFP4_MOE"
MAX_M_ENV = "SUFFIX_NVFP4_MOE_MAX_M"
MARKER = "[suffix nvfp4-moe]"
FAMILY = "nvfp4_moe"
ACT_CODE = {"silu": 0, "gelu_tanh": 1}
# (E global, H, I per rank, topk, activation, ep_size, ep_rank)
GEMMA_MOE = (128, 2816, 704, 8, "gelu_tanh", 1, 0)
GLM_EP = (256, 6144, 2048, 8, "silu", 8, 5)  # TP=8+EP rank 5: experts 160..191
GLM_TP8 = (256, 6144, 256, 8, "silu", 1, 0)  # TP=8 without EP: I/8 shard
HBM_BPS = 1.79e12  # RTX PRO 6000 Max-Q
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_state = {"armed": False, "instances": 0, "layers_ours": 0, "stock": {},
          "ws": {}, "checked": False, "hook": None, "oracle": [], "active_logged": False}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def max_m() -> int:
    """Decode-size ceiling (tokens per call). Knob, not a constant: the
    crossover vs FlashInfer is measured by `bench`, not assumed."""
    v = int(os.environ.get(MAX_M_ENV, "32"))
    if not 1 <= v <= 256:
        raise ValueError(f"{MAX_M_ENV}={v} outside [1, 256]")
    return v


def _r(v: int, a: int) -> int:
    return -(-v // a) * a


# ---------------------------------------------------------------------------
# reference (torch f64, any device): the NVFP4 spec our kernel implements
# ---------------------------------------------------------------------------
# Accuracy anatomy (2026-09-26; CPU emulation, gemma shapes, vs this f64
# reference): the intermediate is RE-QUANTIZED to FP4 after act*up, so any
# sub-ulp perturbation before that point flips e2m1 codes / e4m3 block scales
# near rounding boundaries; each flip is a whole FP4 quantum (12-25 % of the
# element), which amplifies a relative perturbation d to ~sqrt(d * quantum)
# end-to-end. The dense GEMM has no second quantization, hence no such gain.
#   bf16 output rounding only (the floor, = this kernel)       1.7e-3
#   tanh.approx.f32 in gelu (rel 2^-11; kernel before 26.09)    2.5e-3..1.7e-2
#   FlashInfer: fc1 GEMM output stored bf16 before act          2.1e-2..2.8e-2
#   FlashInfer: act*up cast to bf16 before its FP4 quant        1.9e-2..3.3e-2
#   FlashInfer: fc2 output stored bf16 before finalize          2.3e-3
#   FlashInfer: erf GeGLU (vLLM maps GELU_TANH -> Geglu)        2.7e-3..6.5e-3
# (TRT-LLM cutlass_fused_moe_kernels.cuh: GemmOutputType = bf16 for fc1/fc2,
# quantizePackedFPXValue casts post_act to bf16; flashinfer_utils.py maps
# GELU_TANH to ActivationType.Geglu although FI 0.6.18 has GegluTanh.)
# So ours-vs-FlashInfer (1.3e-2..8e-2 on silicon) is FlashInfer's own
# distance from the spec; `fi=True` below emulates it so the oracle proves
# that attribution on silicon instead of assuming it.
def act_ref(x, kind: int, erf: bool = False):
    import torch
    if kind == 1:
        if erf:
            return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return x * torch.sigmoid(x)


def _bf(t):
    import torch
    return t.to(torch.bfloat16).to(t.dtype)


def quant_codes(x, g: float):
    """f32 [R, K] -> (packed uint8 [R, K/2] low-nibble-even, sf e4m3 bits
    uint8 [R, K/16]) — vLLM scaled_fp4_quant: sf = e4m3(amax16*g/6),
    q = e2m1_rne(x*g/sf)."""
    import torch
    r, k = x.shape
    blk = x.float().reshape(r, k // 16, 16)
    g = torch.tensor(g, dtype=torch.float32, device=x.device)
    amax = blk.abs().amax(-1)
    sf8 = torch.clamp(amax * (g / 6.0), max=448.0).to(torch.float8_e4m3fn)
    sf = sf8.float()
    inv = torch.where(sf == 0, torch.zeros_like(sf), g / torch.where(sf == 0, 1.0, sf))
    s = blk * inv[..., None]
    mid = torch.tensor(_MID, device=x.device)
    odd = torch.arange(7, device=x.device) % 2 == 1
    a = s.abs().clamp(max=6.0)[..., None]
    mag = (a > mid).sum(-1) + ((a == mid) & odd).sum(-1)
    code = (mag | torch.where(s < 0, 8, 0)).to(torch.uint8).reshape(r, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), sf8.view(torch.uint8)


def dequant(packed, sf):
    """(packed uint8 [R, K/2], sf f32 [R, K/16]) -> f32 [R, K] (no 1/g)."""
    import torch
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=packed.device)
    r, k = packed.shape[0], packed.shape[1] * 2  # explicit: R may be 0 (all off-rank)
    code = torch.stack([packed & 0xF, packed >> 4], -1).reshape(r, k)
    val = lut[code.long()]
    return (val.reshape(r, k // 16, 16) * sf[..., None]).reshape(r, k)


def qdq(x, g: float):
    """Quantize-dequantize (value * g, i.e. without the 1/g folded into alpha)."""
    import torch
    q, sfb = quant_codes(x, g)
    return dequant(q, sfb.view(torch.float8_e4m3fn).float())


def swizzle(sf_bits):
    """uint8 [R, C] -> flat 128x4-swizzled uint8 [r128(R) * r4(C)] (vLLM
    swizzle_blockscale / kernel sf_offset)."""
    import torch
    r, c = sf_bits.shape
    rp, cp = _r(r, 128), _r(c, 4)
    pad = torch.zeros(rp, cp, dtype=torch.uint8, device=sf_bits.device)
    pad[:r, :c] = sf_bits
    return pad.reshape(rp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def unswizzle(flat, r: int, c: int):
    """Inverse of `swizzle`: flat uint8 -> [R, C] uint8."""
    rp, cp = _r(r, 128), _r(c, 4)
    t = flat.reshape(rp // 128, cp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return t.reshape(rp, cp)[:r, :c]


def _expert_w(w, w_sf, e: int):
    """Dequantized f32 weight of expert e (without 1/g_w)."""
    import torch
    rows, kh = w.shape[1], w.shape[2]
    sf = unswizzle(w_sf[e].reshape(-1).view(torch.uint8), rows, kh * 2 // 16)
    return dequant(w[e], sf.contiguous().view(torch.float8_e4m3fn).float())


def _groups(ids, e_count: int):
    """{expert: pair indices ascending} over valid ids (p = t*topk + k)."""
    import torch
    flat = ids.reshape(-1).long()
    out = {}
    for e in torch.unique(flat).tolist():
        if 0 <= e < e_count:
            out[e] = torch.nonzero(flat == e).reshape(-1)
    return out


def fc1_ref(p, x, ids, fi=False):
    """f64 [P, I] intermediate h = act(g1*gate) * g1*up per routed pair.
    fi=True: FlashInfer's numerics (bf16 GEMM output, erf GeGLU, bf16 act)."""
    import torch
    e_count, two_i = p["w13"].shape[0], p["w13"].shape[1]
    i = two_i // 2
    topk = ids.shape[1]
    xd = qdq(x.float(), p["a1g"]).double()
    h = torch.zeros(ids.numel(), i, dtype=torch.float64, device=x.device)
    for e, pairs in _groups(ids, e_count).items():
        gu = float(p["g1"][e]) * (xd[pairs // topk] @ _expert_w(p["w13"], p["w13_sf"], e).double().T)
        if fi:
            gu = _bf(gu)
        a = act_ref(gu[:, i:], p["act"], erf=fi) * gu[:, :i]
        h[pairs] = _bf(a) if fi else a
    return h


def moe_ref(p, x, ids, tw, fi=False):
    """Exact f64 [M, H] routed-experts output of the quantized problem: the
    NVFP4 spec with f64 arithmetic between the two FP4 quantizations (the
    kernel keeps f32 there). fi=True: emulation of FlashInfer's numerics."""
    import torch
    m, topk = ids.shape
    e_count, hdim = p["w2"].shape[0], p["w2"].shape[1]
    hd = qdq(fc1_ref(p, x, ids, fi).float(), p["a2g"]).double()
    y = torch.zeros(m * topk, hdim, dtype=torch.float64, device=x.device)
    for e, pairs in _groups(ids, e_count).items():
        yy = float(p["g2"][e]) * (hd[pairs] @ _expert_w(p["w2"], p["w2_sf"], e).double().T)
        if fi:
            yy = _bf(yy)
        y[pairs] = yy * tw.reshape(-1)[pairs, None].double()
    valid = ((ids >= 0) & (ids < e_count)).reshape(-1, 1).float()
    y = (y * valid).reshape(m, topk, hdim)
    out = torch.zeros(m, hdim, dtype=torch.float64, device=x.device)
    for k in range(topk):  # the kernel's fixed k order
        out += y[:, k]
    return out


def to_local(ids, base: int, local: int):
    """Global expert ids -> this rank's local ids, -1 when not local (the
    kernel's topk_id - id_base range test; = vLLM's linear expert_map)."""
    import torch
    loc = ids.long() - base
    return torch.where((loc >= 0) & (loc < local), loc, -1)


def ep_base(expert_map, ep_rank: int, ep_size: int, global_e: int, local_e: int):
    """id_base when `expert_map` (list/None) is exactly the linear placement
    FlashInfer assumes (local = global - ep_rank*E_local, E_local*ep ==
    E_global), else None. No map (ep == 1) -> 0."""
    if expert_map is None:
        return 0 if ep_size == 1 and local_e == global_e else None
    base = ep_rank * local_e
    want = [g - base if base <= g < base + local_e else -1 for g in range(global_e)]
    return base if local_e * ep_size == global_e and list(expert_map) == want else None


def route_twin(ids, e_count: int, base: int = 0):
    """CPU twin of moe_route: (slot_expert, slot_off, slot_cnt, pair_list)
    with slots = min(E, P); active experts ascending, pairs ascending."""
    flat = [int(v) - base for v in ids.reshape(-1).tolist()]
    slots = min(e_count, len(flat))
    se, so, sc, pl = [-1] * slots, [0] * slots, [0] * slots, []
    s = 0
    for e in range(e_count):
        pairs = [q for q, v in enumerate(flat) if v == e]
        if pairs:
            se[s], so[s], sc[s] = e, len(pl), len(pairs)
            pl += pairs
            s += 1
    return se, so, sc, pl


def make_problem(e_count, hdim, idim, act="gelu_tanh", device="cpu", seed=0, wstd=0.05):
    """Random NVFP4 routed-experts layer in vLLM's post-processing FlashInfer
    CUTLASS layout (w13 = [up; gate], 128x4-swizzled per-expert scales, g1/g2
    = 1/(g_w*g_a), shared activation gscales calibrated on a random batch)."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    rp13, cp13 = _r(2 * idim, 128), _r(hdim // 16, 4)
    rp2, cp2 = _r(hdim, 128), _r(idim // 16, 4)
    w13 = torch.empty(e_count, 2 * idim, hdim // 2, dtype=torch.uint8, device=dev)
    w2 = torch.empty(e_count, hdim, idim // 2, dtype=torch.uint8, device=dev)
    s13 = torch.empty(e_count, rp13 * cp13, dtype=torch.uint8, device=dev)
    s2 = torch.empty(e_count, rp2 * cp2, dtype=torch.uint8, device=dev)
    gw13 = torch.empty(e_count)
    gw2 = torch.empty(e_count)
    for e in range(e_count):
        for w, s, gw, shape in ((w13, s13, gw13, (2 * idim, hdim)), (w2, s2, gw2, (hdim, idim))):
            wf = (torch.randn(*shape, generator=gen) * wstd).bfloat16().float().to(dev)
            gw[e] = 2688.0 / float(wf.abs().max())
            q, sfb = quant_codes(wf, float(gw[e]))
            w[e] = q
            s[e] = swizzle(sfb)
    a1g = 2688.0 / 4.5  # randn(H) amax ~ 4-4.5
    p = dict(w13=w13, w2=w2, act=ACT_CODE[act], act_name=act, a1g=a1g, a2g=1.0,
             w13_sf=s13.view(torch.float8_e4m3fn).reshape(e_count, rp13, cp13),
             w2_sf=s2.view(torch.float8_e4m3fn).reshape(e_count, rp2, cp2),
             g1=(1.0 / (gw13 * a1g)).float().to(dev), g2=None)
    # calibrate a2 like modelopt: amax of the intermediate on a random batch
    xc = torch.randn(16, hdim, generator=gen).bfloat16().to(dev)
    ids = torch.rand(16, e_count, generator=gen).topk(min(8, e_count), -1).indices.to(dev)
    p["a2g"] = 2688.0 / float(fc1_ref(p, xc, ids).abs().max().clamp_min(1e-6))
    p["g2"] = (1.0 / (gw2 * p["a2g"])).float().to(dev)
    return p


def rand_routing(m, e_count, topk, device, seed=0, base=0, local=None, dead=None):
    """Uniform random top-k experts per token (distinct), softmax weights.
    EP (local < e_count): rows t % 3 == 1 (or all rows when dead=True) are
    routed only to experts outside [base, base + local) — tokens this rank
    must return as exact zeros."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    score = torch.rand(m, e_count, generator=gen)
    local = e_count if local is None else local
    if local < e_count and dead is not False:
        rows = slice(None) if dead else slice(1, None, 3)
        score[rows, base:base + local] = -1.0
    ids = score.topk(topk, -1).indices.int()
    tw = torch.softmax(torch.randn(m, topk, generator=gen), -1).float()
    return ids.to(device), tw.to(device)


REF_TOL = 5e-3  # ours vs the f64 spec, end to end (see oracle_ok)
STAGE_TOL = {"xq": 1e-3, "fc1": 1e-4, "hq": 1e-3, "fc2": 1e-4, "comb": 1e-3}


def oracle_ok(rel_vs_ref: float, rel_vs_fi: float, fi_rel_vs_ref: float) -> bool:
    """Ours vs the f64 spec <= REF_TOL: the floor is the bf16 output
    rounding (~1.1e-3 rms) plus requant flips from f32-vs-f64 intermediates,
    1.7e-3 emulated; tanh.approx-class drift (2.5e-3..1.7e-2) or any
    layout/scale bug trips it. FlashInfer's own distance to the spec is NOT
    a tolerance for us (it is 1-3e-2 from bf16 intermediates, see the
    accuracy anatomy above). Ours vs FlashInfer only has to respect the
    triangle bound of both errors (floor 2e-2)."""
    return (rel_vs_ref <= REF_TOL
            and rel_vs_fi <= max(2e-2, 1.1 * (rel_vs_ref + fi_rel_vs_ref)))


def _rel(a, b) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def judge(ours, fi, ref, local_ids, fi_emu=None):
    """(ok, metrics) for one oracle case: oracle_ok vs FlashInfer and the
    f64 reference, finite, and the EP zero contract — tokens with no local
    expert are exactly 0 in ours AND in vLLM's stock output. fi_emu (our
    emulation of FlashInfer's numerics) is evidence only: flashinfer ~
    fi_emulation << flashinfer ~ ref attributes FlashInfer's error."""
    import torch
    r_ref, r_fi, fi_ref = _rel(ours, ref), _rel(ours, fi), _rel(fi, ref)
    dead = (local_ids < 0).all(1)
    zero = bool((ours[dead] == 0).all() and (fi[dead] == 0).all())
    ok = oracle_ok(r_ref, r_fi, fi_ref) and zero and bool(torch.isfinite(ours).all())
    emu = "" if fi_emu is None else f"flashinfer_vs_fi_emulation={_rel(fi, fi_emu):.2e} "
    return ok, (f"rel_vs_ref={r_ref:.2e} rel_vs_flashinfer={r_fi:.2e} "
                f"flashinfer_rel_vs_ref={fi_ref:.2e} {emu}nonlocal_tokens={int(dead.sum())} "
                f"nonlocal_zero={zero}")


def stages(p, x, ids, tw, ws, out):
    """Per-stage drift of OUR kernel, read back from its workspace right
    after a run (`ids` LOCAL, -1 = off-rank). Each stage is checked against
    the f64 spec fed with the kernel's own previous-stage output, so a drift
    is pinned to the stage that made it:
      xq    x -> NVFP4 (a1 gscale)        frac of dequantized values differing
      fc1   act(g1*gate) * g1*up           rel vs f64 from the kernel's xq
      hq    inter -> NVFP4 (a2 gscale)    frac differing vs quant of its inter
      fc2   g2 * tw * (hq @ w2^T)          rel vs f64 from the kernel's hq
      comb  bf16(sum_k y), fixed k order   frac differing
    Tolerances STAGE_TOL: quant/combine are the same f32 ops as the spec
    (bit-exact expected, 1e-3 allows rare rounding ties); GEMM stages are
    f32 accumulation over K <= 6144 (~sqrt(K) * 2^-24 ~ 5e-6), 20x headroom.
    Returns (ok, msg)."""
    import torch
    aq, asf, inter, hq, hsf, y, _ = ws
    m, topk = ids.shape
    e_count, hdim, ih = p["w2"].shape
    idim, pairs = ih * 2, m * topk
    f8 = lambda b: b.view(torch.float8_e4m3fn).float()
    frac = lambda a, b: float((a != b).float().mean()) if a.numel() else 0.0
    xq = dequant(aq[:m * hdim // 2].view(m, -1), f8(asf[:m * hdim // 16].view(m, -1)))
    got = {"xq": frac(xq, qdq(x.float(), p["a1g"]))}
    vp = torch.nonzero(ids.reshape(-1) >= 0).reshape(-1)
    h = inter[:pairs * idim].view(pairs, idim)[vp]
    h_ref = torch.zeros(pairs, idim, dtype=torch.float64, device=x.device)
    y_ref = torch.zeros(pairs, hdim, dtype=torch.float64, device=x.device)
    hd = dequant(hq[:pairs * idim // 2].view(pairs, -1), f8(hsf[:pairs * idim // 16].view(pairs, -1)))
    for e, pr in _groups(ids, e_count).items():
        gu = float(p["g1"][e]) * (xq[pr // topk].double()
                                  @ _expert_w(p["w13"], p["w13_sf"], e).double().T)
        h_ref[pr] = act_ref(gu[:, idim:], p["act"]) * gu[:, :idim]
        y_ref[pr] = (float(p["g2"][e]) * (hd[pr].double()
                                          @ _expert_w(p["w2"], p["w2_sf"], e).double().T)
                     * tw.reshape(-1)[pr, None].double())
    got["fc1"] = _rel(h, h_ref[vp]) if vp.numel() else 0.0
    got["hq"] = frac(hd[vp], qdq(h, p["a2g"]))
    yk = y[:pairs * hdim].view(pairs, hdim)
    got["fc2"] = _rel(yk[vp], y_ref[vp]) if vp.numel() else 0.0
    acc = torch.zeros(m, hdim, dtype=torch.float32, device=x.device)
    valid = (ids >= 0)
    yk = yk.view(m, topk, hdim)
    for k in range(topk):  # the kernel's order, f32
        acc = torch.where(valid[:, k, None], acc + yk[:, k], acc)
    got["comb"] = frac(out.float(), acc.bfloat16().float())
    ok = all(got[k] <= STAGE_TOL[k] for k in got)
    return ok, "stages " + " ".join(f"{k}={v:.1e}" for k, v in got.items())


# ---------------------------------------------------------------------------
# native launch (shared by serving, oracle, bench)
# ---------------------------------------------------------------------------
def workspace(dev, m: int, topk: int, hdim: int, idim: int, e_count: int, ws=None):
    """(aq, asf, inter, hq, hsf, y, route) sized for M tokens; grown, never
    shrunk. Called at LOAD time only when serving."""
    import torch
    p = m * topk
    need = (m * hdim // 2, m * hdim // 16, p * idim, p * idim // 2, p * idim // 16,
            p * hdim, 3 * e_count + p)
    dts = (torch.uint8, torch.uint8, torch.float32, torch.uint8, torch.uint8,
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
    native.nvfp4_moe_cuda(x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                          p["g2"], *ws, out, float(p["a1g"]), float(p["a2g"]),
                          int(p["act"]), int(p.get("id_base", 0)), stream)
    return out


# ---------------------------------------------------------------------------
# vLLM wiring
# ---------------------------------------------------------------------------
def eligibility(info: dict) -> str | None:
    """Why a RoutedExperts layer cannot use our kernel, or None. Pure (plain
    values) so the CPU tests pin every branch."""
    if info.get("quant_dtype") != "nvfp4":
        return f"not NVFP4 W4A4 (quant_dtype={info.get('quant_dtype')})"
    if info.get("backend") != "FLASHINFER_CUTLASS":
        return f"NvFp4 MoE backend {info.get('backend')} (layout is FlashInfer CUTLASS only)"
    if not info.get("scale_swizzled", False):
        return "activation scales not swizzled"
    if info.get("act") not in ACT_CODE:
        return f"activation {info.get('act')} not in {sorted(ACT_CODE)}"
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
        return "expert_map is not the linear ep_rank*E_local placement FlashInfer assumes"
    e, h, i = info.get("E", 0), info.get("H", 0), info.get("I", 0)
    if not (1 <= e <= 256) or h % 64 or i % 64:
        return f"shape E={e} H={h} I={i} (E<=256, H,I % 64)"
    if not info.get("sf_exact", False):
        return "block scales not in the per-expert padded swizzled layout"
    if not info.get("gscale_shared", False):
        return "per-expert activation gscales differ"
    return None


def _make_layer_cls():
    import torch
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from suffix_hybrid import oxide_kernels

    native = oxide_kernels.native()

    class SuffixNvFp4RoutedExperts(RoutedExperts):
        """vLLM RoutedExperts + our sm_120a NVFP4 decode MoE for M <= max_m."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _state["instances"] += 1
            self._sfx_moe = None
            qm = self.quant_method
            orig = qm.process_weights_after_loading

            # Instance-level wrap (not a vLLM class patch): run our prepare +
            # layer oracle right after vLLM's own FI weight processing.
            def _after_load(layer, _orig=orig):
                _orig(layer)
                if layer is self:
                    self._sfx_prepare()

            qm.process_weights_after_loading = _after_load

        def _sfx_info(self):
            qm = self.quant_method
            qc = getattr(qm, "moe_quant_config", None)
            mc = self.moe_config
            pc = mc.moe_parallel_config
            w13 = getattr(self, "w13_weight", None)
            info = dict(quant_dtype=getattr(qc, "quant_dtype", None),
                        backend=getattr(getattr(qm, "nvfp4_backend", None), "name", None),
                        scale_swizzled=getattr(qc, "is_scale_swizzled", False),
                        act=getattr(self.activation, "value", self.activation),
                        clamp_limit=getattr(qc, "gemm1_clamp_limit", None),
                        swiglu_alpha=getattr(qc, "gemm1_alpha", None),
                        router_weight_on_input=bool(self.apply_router_weight_on_input),
                        bias=getattr(qc, "w1_bias", None) is not None,
                        tp=mc.tp_size, ep=mc.ep_size, dp=mc.dp_size,
                        all2all=bool(pc.use_all2all_kernels), eplb=bool(pc.enable_eplb),
                        mk_shared_overlap=bool(getattr(qm, "mk_can_overlap_shared_experts",
                                                       False)),
                        expert_map=self.expert_map is not None)
            if info["quant_dtype"] != "nvfp4" or w13 is None or w13.dim() != 3:
                return info, None
            e, two_i, hh = w13.shape
            h, i = hh * 2, two_i // 2
            em = self.expert_map
            info["ep_base"] = ep_base(None if em is None else em.tolist(), pc.ep_rank,
                                      mc.ep_size, self.global_num_experts, e)
            s1, s2 = qc.w1_scale, qc.w2_scale
            a1, a2 = qc.a1_gscale.reshape(-1), qc.a2_gscale.reshape(-1)
            info.update(
                E=e, H=h, I=i,
                sf_exact=(s1.numel() == e * _r(2 * i, 128) * _r(h // 16, 4)
                          and s2.numel() == e * _r(h, 128) * _r(i // 16, 4)
                          and tuple(self.w2_weight.shape) == (e, h, i // 2)
                          and w13.dtype == torch.uint8),
                # one device sync per layer at LOAD time; never in forward
                gscale_shared=bool((a1 == a1[0]).all() and (a2 == a2[0]).all()))
            return info, qc

        def _sfx_prepare(self):
            info, qc = self._sfx_info()
            why = eligibility(info)
            if why is not None:
                _state["stock"][why] = _state["stock"].get(why, 0) + 1
                _log(f"layer {self.layer_name} stays on vLLM's MoE path: {why}")
                return
            dev = self.w13_weight.device
            oxide_kernels.ensure_loaded(FAMILY, dev.index)
            cfg = dict(w13=self.w13_weight, w13_sf=qc.w1_scale, w2=self.w2_weight,
                       w2_sf=qc.w2_scale, g1=qc.g1_alphas.float().contiguous(),
                       g2=qc.g2_alphas.float().contiguous(),
                       a1g=float(qc.a1_gscale.reshape(-1)[0]),
                       a2g=float(qc.a2_gscale.reshape(-1)[0]),
                       act=ACT_CODE[info["act"]], H=info["H"], topk=self.top_k,
                       max_m=max_m(), id_base=info["ep_base"],
                       E_global=self.global_num_experts)
            _state["ws"][dev] = workspace(dev, cfg["max_m"], cfg["topk"], info["H"],
                                          info["I"], info["E"], _state["ws"].get(dev))
            self._sfx_moe = cfg
            _state["oracle"].append(self._sfx_oracle(cfg, info))
            _state["layers_ours"] += 1

        def _sfx_run(self, cfg, x, topk_weights, topk_ids):
            dev = x.device
            tw = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
            return run_ours(native, cfg, x.contiguous(), topk_ids.contiguous(),
                            tw.contiguous(), _state["ws"][dev],
                            torch.cuda.current_stream(dev).cuda_stream)

        def _sfx_oracle(self, cfg, info):
            """Ours vs the parent (vLLM FlashInfer cutlass_fused_moe, with
            this rank's real ep_size/ep_rank) vs the f64 reference on this
            layer's REAL weights, GLOBAL routing ids. EP adds a batch whose
            tokens all route to other ranks (must be exact 0). Fatal."""
            dev = self.w13_weight.device
            worst, worst_ref, worst_fi_ref, worst_emu = 0.0, 0.0, 0.0, 0.0
            base, local = cfg["id_base"], info["E"]
            mm = cfg["max_m"]  # workspace capacity: every case M <= max_m
            cases = [(m, None) for m in sorted({1, min(8, mm), mm})]
            if local < cfg["E_global"]:
                cases.append((min(8, mm), True))
            for m, dead in cases:
                seed = zlib.crc32(str(self.layer_name).encode()) % 10007 + m
                gen = torch.Generator(device="cpu").manual_seed(seed)
                amax = 2688.0 / cfg["a1g"]  # calibrated activation range
                x = (torch.randn(m, info["H"], generator=gen) * (amax / 4.5)).to(
                    dev, torch.bfloat16)
                ids, tw = rand_routing(m, cfg["E_global"], cfg["topk"], dev, seed,
                                       base, local, dead)
                fi = super().forward_modular(x, tw, ids).float()
                out = self._sfx_run(cfg, x, tw, ids)
                lid = to_local(ids, base, local)
                st_ok, st = stages(cfg, x, lid, tw, _state["ws"][dev], out)
                ours = out.float()
                ref = moe_ref(cfg, x, lid, tw)
                emu = moe_ref(cfg, x, lid, tw, fi=True).bfloat16()
                ok, msg = judge(ours, fi, ref, lid, emu)
                ok, msg = ok and st_ok, f"{msg} {st}"
                if fi.norm() > 0:
                    worst = max(worst, _rel(ours, fi))
                    worst_ref = max(worst_ref, _rel(ours, ref))
                    worst_fi_ref = max(worst_fi_ref, _rel(fi, ref))
                    worst_emu = max(worst_emu, _rel(fi, emu))
                if not ok:
                    raise RuntimeError(
                        f"{MARKER} LAYER ORACLE FAIL {self.layer_name} M={m} "
                        f"id_base={base}: {msg} — refusing to serve with {GATE}=1")
            torch.cuda.synchronize(dev)
            _log(f"LAYER ORACLE PASS {self.layer_name} E={local}/{cfg['E_global']} "
                 f"id_base={base} H={info['H']} I={info['I']} act={info['act']} "
                 f"max_rel_vs_flashinfer={worst:.2e} max_rel_vs_ref={worst_ref:.2e} "
                 f"flashinfer_max_rel_vs_ref={worst_fi_ref:.2e} "
                 f"flashinfer_max_rel_vs_fi_emulation={worst_emu:.2e} (last case {st})")
            return worst

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                            shared_experts_input=None):
            # shared_experts: never run by a non-overlapping MK (eligibility
            # rejects mk_shared_overlap) -> MoERunner already ran them.
            cfg = self._sfx_moe
            if (cfg is None or x.dim() != 2
                    or not 1 <= x.shape[0] <= cfg["max_m"] or x.dtype != torch.bfloat16
                    or x.shape[1] != cfg["H"]):
                return super().forward_modular(x, topk_weights, topk_ids,
                                               shared_experts, shared_experts_input)
            if not _state["active_logged"]:
                _state["active_logged"] = True
                _log(summary())
            return self._sfx_run(cfg, x, topk_weights, topk_ids)

    return SuffixNvFp4RoutedExperts


def verdict(instances: int, ours: int, stock: dict) -> str | None:
    """None when >= 1 layer runs our kernel; '' when undecidable yet (no layer
    processed); else the loud startup error."""
    if ours > 0:
        return None
    if instances == 0:
        return (f"{MARKER} NVFP4-MOE NOT ENGAGED with {GATE}=1: no RoutedExperts was "
                "built through the OOT class (model has no routed experts, or "
                "register_oot ran after model construction)")
    if not stock:
        return ""
    why = "; ".join(f"{k} x{v}" for k, v in sorted(stock.items()))
    return f"{MARKER} NVFP4-MOE NOT ENGAGED with {GATE}=1: every MoE layer ineligible: {why}"


def _first_forward_check(module, args):
    """Global forward pre-hook (PyTorch API): at the first module call after
    weight processing, fail startup if no layer engaged."""
    if _state["checked"]:
        return
    err = verdict(_state["instances"], _state["layers_ours"], _state["stock"])
    if err == "":
        return  # weights not processed yet: decide at a later call
    _state["checked"] = True
    h = _state.pop("hook", None)
    if h is not None:
        h.remove()
    if err is not None:
        _log(err)
        raise RuntimeError(err)
    stock = "; ".join(f"{k} x{v}" for k, v in sorted(_state["stock"].items())) or "-"
    _log(f"NVFP4-MOE SELECTION: {_state['layers_ours']} RoutedExperts layers on our "
         f"decode kernel (M<={max_m()}), stock: {stock}")


def register():
    """vllm.general_plugins entry point. No-op unless SUFFIX_NVFP4_MOE=1."""
    if not gate_on():
        return None
    if _state["armed"]:
        return _state
    import torch
    from suffix_hybrid import oxide_kernels
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    native = oxide_kernels.native()
    if not hasattr(native, "nvfp4_moe_cuda"):
        raise RuntimeError(f"{GATE}=1 but _native lacks nvfp4_moe_cuda (oxide-kernels build)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError(f"{GATE}=1: the NVFP4 MoE cubin is sm_120a SASS (cc 12.x only)")
    ent = [k for k in oxide_kernels.manifest()["kernels"] if k["name"] == FAMILY]
    if not ent:
        raise RuntimeError(f"{GATE}=1 but the oxide manifest has no {FAMILY!r} cubin")
    max_m()  # validate the knob now, not at load
    cls = _make_layer_cls()
    RoutedExperts.register_oot(cls, name="RoutedExperts")
    _state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        _first_forward_check)
    _state["armed"] = True
    _log(f"NVFP4-MOE armed: RoutedExperts -> SuffixNvFp4RoutedExperts (M<={max_m()} -> "
         f"sm_120a mxf4nvf4 SASS, sha256 {ent[0]['sha256'][:12]}; else vLLM FlashInfer)")
    return _state


def summary() -> str:
    worst = max(_state["oracle"], default=0.0)
    return (f"NVFP4-MOE ACTIVE: {_state['layers_ours']} layers on our decode MoE, "
            f"{sum(_state['stock'].values())} on vLLM's path, "
            f"oracle max_rel_vs_flashinfer={worst:.2e}")


# ---------------------------------------------------------------------------
# standalone on-silicon oracle + bench (synthetic weights, gemma shapes)
# ---------------------------------------------------------------------------
def _native_ready():
    import torch
    from suffix_hybrid import oxide_kernels
    native = oxide_kernels.native()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("NVFP4 MoE cubin is sm_120a SASS (cc 12.x only)")
    oxide_kernels.ensure_loaded(FAMILY)
    return native


def _fi_call(p, x, ids, tw, out):
    """vLLM's FLASHINFER_CUTLASS NVFP4 path, verbatim argument mapping
    (prepare_finalize/no_dp_ep.py quant + experts/flashinfer_cutlass_moe.py),
    incl. the EP contract: GLOBAL ids + ep_size/ep_rank (FlashInfer computes
    only experts [ep_rank*E_local, +E_local) — no collective involved, so
    one GPU reproduces one EP rank exactly)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_type,
    )
    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
    xq, xsf = ops.scaled_fp4_quant(x, p["a1g_t"], is_sf_swizzled_layout=True)
    flashinfer_cutlass_fused_moe(
        input=xq, token_selected_experts=ids.to(torch.int), token_final_scales=tw,
        fc1_expert_weights=p["w13"].view(torch.long),
        fc2_expert_weights=p["w2"].view(torch.long), output=out,
        output_dtype=torch.bfloat16,
        quant_scales=[p["a1g_t"], p["w13_sf"].view(torch.int32), p["g1"], p["a2g_t"],
                      p["w2_sf"].view(torch.int32), p["g2"]],
        input_sf=xsf, tp_size=1, tp_rank=0, ep_size=p["ep_size"], ep_rank=p["ep_rank"],
        activation_type=activation_to_flashinfer_type(MoEActivation(p["act_name"])))
    return out


def _dev_problem(dev, case, seed=0):
    """Synthetic weights for ONE rank: E_global/ep_size local experts,
    id_base = ep_rank * E_local (vLLM linear expert_map)."""
    import torch
    e_global, hdim, idim, topk, act, ep_size, ep_rank = case
    local = e_global // ep_size
    p = make_problem(local, hdim, idim, act, device=dev, seed=seed)
    p["a1g_t"] = torch.full((local,), p["a1g"], dtype=torch.float32, device=dev)
    p["a2g_t"] = torch.full((local,), p["a2g"], dtype=torch.float32, device=dev)
    p.update(ep_size=ep_size, ep_rank=ep_rank, id_base=ep_rank * local, E_global=e_global)
    return p


def _case_name(case) -> str:
    e_global, hdim, idim, topk, act, ep_size, ep_rank = case
    ep = f" EP{ep_size} rank{ep_rank} ({e_global // ep_size} local)" if ep_size > 1 else ""
    return f"E={e_global} H={hdim} I={idim} top{topk} {act}{ep}"


CONCURRENCY = (1, 8, 16, 32)  # decode tokens per step -> routed rows = M * topk
GLM_MS = (1, 6, 32, 64)  # MTP k=5: one seq verifies 6 tokens per step
ORACLE_CASES = ((GEMMA_MOE, CONCURRENCY), ((64, 2048, 768, 8, "silu", 1, 0), CONCURRENCY),
                (GLM_EP, GLM_MS), (GLM_TP8, GLM_MS))
BENCH_CASES = ((GEMMA_MOE, CONCURRENCY), (GLM_EP, (1, 6, 12, 24, 48, 64, 96, 192)),
               (GLM_TP8, (1, 6, 12, 24, 48, 64, 96, 192)))


def oracle(cases=ORACLE_CASES):
    """Per case: ours vs vLLM's FlashInfer (same EP rank) vs the f64
    reference, FlashInfer vs our emulation of its numerics, and our kernel's
    per-stage drift read back from its workspace (`stages`); EP cases add rows routed only off-rank plus one batch whose
    tokens ALL route off-rank (exact-zero contract). Fatal on mismatch."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev).cuda_stream
    lines = []
    for case, ms in cases:
        e_global, hdim, idim, topk, act, ep_size, _ = case
        p = _dev_problem(dev, case)
        base, local = p["id_base"], e_global // ep_size
        ws = workspace(dev, max(ms), topk, hdim, idim, local)
        runs = [(m, None) for m in ms] + ([(8, True)] if local < e_global else [])
        for m, dead in runs:
            ids, tw = rand_routing(m, e_global, topk, dev, seed=m, base=base, local=local,
                                   dead=dead)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            lid = to_local(ids, base, local)
            ref = moe_ref(p, x, lid, tw)
            emu = moe_ref(p, x, lid, tw, fi=True).bfloat16()
            fi = _fi_call(p, x, ids, tw, torch.full((m, hdim), float("nan"),
                                                    dtype=torch.bfloat16, device=dev)).float()
            again = run_ours(native, p, x, ids, tw, ws, stream).float()
            out = run_ours(native, p, x, ids, tw, ws, stream)
            st_ok, st = stages(p, x, lid, tw, ws, out)
            ours = out.float()
            det = bool(torch.equal(ours, again))
            ok, msg = judge(ours, fi, ref, lid, emu)
            ok = ok and det and st_ok
            lines.append(f"{_case_name(case)} M={m}{' all-nonlocal' if dead else ''} "
                         f"P={m * topk}: {msg} {st} deterministic={det} "
                         f"{'OK' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"{MARKER} NVFP4-MOE ORACLE FAIL: {lines[-1]}")
        del p, ws
        torch.cuda.empty_cache()
    for ln in lines:
        print(f"{MARKER} {ln}", file=sys.stderr, flush=True)
    return f"{MARKER} NVFP4-MOE ORACLE PASS ({len(lines)} cases, sm_120a mxf4nvf4 mma)"


def _graph_us(fn, dev, iters):
    import torch
    s = torch.cuda.Stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        fn(s)  # warm (FlashInfer tactic selection happens here)
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn(s)
    torch.cuda.current_stream(dev).wait_stream(s)
    g.replay()
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize(dev)
    return e0.elapsed_time(e1) * 1000.0 / iters


def bench(cases=BENCH_CASES, iters=200):
    """us per MoE layer call on one rank (x quant + experts + combine), CUDA
    graphs. The crossover M where FlashInfer wins sets SUFFIX_NVFP4_MOE_MAX_M
    (above it the layer delegates to vLLM)."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    for case, ms in cases:
        e_global, hdim, idim, topk, act, ep_size, _ = case
        p = _dev_problem(dev, case, seed=1)
        base, local = p["id_base"], e_global // ep_size
        ws = workspace(dev, max(ms), topk, hdim, idim, local)
        per_expert = 2 * idim * hdim * (1 / 2 + 1 / 16) + hdim * idim * (1 / 2 + 1 / 16)
        for m in ms:
            ids, tw = rand_routing(m, e_global, topk, dev, seed=100 + m, dead=False)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            out = torch.empty(m, hdim, dtype=torch.bfloat16, device=dev)
            t_fi = _graph_us(lambda s: _fi_call(p, x, ids, tw, out), dev, iters)
            t_ours = _graph_us(lambda s: run_ours(native, p, x, ids, tw, ws, s.cuda_stream,
                                                  out), dev, iters)
            lid = to_local(ids, base, local)
            distinct = int(torch.unique(lid[lid >= 0]).numel())
            roof = distinct * per_expert / HBM_BPS * 1e6
            res[(case, m)] = (t_fi, t_ours, roof)
            print(f"{MARKER} bench {_case_name(case)} M={m} routed_rows={m * topk} "
                  f"local_experts_hit={distinct}: flashinfer {t_fi:.1f} us, ours "
                  f"{t_ours:.1f} us (x{t_ours / t_fi:.2f}), weight-roofline {roof:.1f} us",
                  file=sys.stderr, flush=True)
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
        print(f"{MARKER} NVFP4-MOE {mode.upper()} FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
