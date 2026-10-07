# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-experts decode kernel (gemma-4-26b-a4b-nvfp4: 30 layers x 128
experts, top-8, hidden 2816, expert intermediate 704) — reference, vLLM
wiring, on-silicon oracle + bench.

Kernel: kernels-oxide/nvfp4_moe (cuda-oxide -> PTX .target sm_120a -> ptxas
13.0 SASS, block-scaled mxf4nvf4 m16n8k64 mma). Host op
``_native.nvfp4_moe_cuda`` launches, on torch's stream, with grids that are a
function of M only (CUDA-graph safe, no host sync):
  route    one CTA: active experts ascending, pairs (p = token*topk + k)
           ascending per expert -> deterministic
  quant x  vLLM scaled_fp4_quant math with the layer's a1 gscale
  fc1      one CTA per (expert slot, 32 intermediate cols) over only that
           expert's routed rows (16-row mma chunks):
           inter[p, j] = act(g1[e] * gate_j) * g1[e] * up_j
           (w13 rows [0, I) = up (w3), [I, 2I) = gate (w1): vLLM's FI layout)
  quant h  a2 gscale
  fc2      y[p, :] = g2[e] * (h_p @ w2[e]^T) * topk_w[p]
  combine  out[t] = sum_k y[t*topk + k] in ascending k (fixed order)

Serving (gate ``SUFFIX_NVFP4_MOE=1``, default OFF; entry point
``suffix_nvfp4_moe``): ``RoutedExperts`` is a vLLM PluggableLayer
(fused_moe/routed_experts.py:45) -> ``RoutedExperts.register_oot`` swaps in
``SuffixNvFp4RoutedExperts``. After vLLM's own weight processing (FlashInfer
CUTLASS layout) each eligible layer runs a LAYER ORACLE on its real weights:
ours vs the parent ``forward_modular`` (= vLLM's FlashInfer
cutlass_fused_moe path) vs the exact fp32 reference, fatal on mismatch.
Decode-sized calls (1 <= M <= SUFFIX_NVFP4_MOE_MAX_M, default 32) run ours;
everything else calls the parent unchanged. Gated on but no layer engaged ->
startup error naming why.

CLI (in-pod, SM120 + oxide bundle):
  python -m suffix_hybrid.kernels.nvfp4_moe oracle   # ours vs FlashInfer vs exact ref
  python -m suffix_hybrid.kernels.nvfp4_moe bench    # us: ours vs FlashInfer (CUDA graphs)
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
# gemma-4-26b-a4b routed experts: (E, H, I, topk, activation)
GEMMA_MOE = (128, 2816, 704, 8, "gelu_tanh")
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
    if not 1 <= v <= 64:
        raise ValueError(f"{MAX_M_ENV}={v} outside [1, 64]")
    return v


def _r(v: int, a: int) -> int:
    return -(-v // a) * a


# ---------------------------------------------------------------------------
# reference (torch fp32, any device) — identical math to the kernel
# ---------------------------------------------------------------------------
def act_ref(x, kind: int):
    import torch
    if kind == 1:
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return x * torch.sigmoid(x)


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
    code = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1)
    val = lut[code.long()]
    return (val.reshape(val.shape[0], -1, 16) * sf[..., None]).reshape(val.shape[0], -1)


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


def fc1_ref(p, x, ids):
    """f32 [P, I] intermediate h = act(g1*gate) * g1*up per routed pair."""
    import torch
    e_count, two_i = p["w13"].shape[0], p["w13"].shape[1]
    i = two_i // 2
    topk = ids.shape[1]
    xd = qdq(x.float(), p["a1g"])
    h = torch.zeros(ids.numel(), i, dtype=torch.float32, device=x.device)
    for e, pairs in _groups(ids, e_count).items():
        gu = p["g1"][e].float() * (xd[pairs // topk] @ _expert_w(p["w13"], p["w13_sf"], e).T)
        h[pairs] = act_ref(gu[:, i:], p["act"]) * gu[:, :i]
    return h


def moe_ref(p, x, ids, tw):
    """Exact f32 [M, H] routed-experts output of the quantized problem (what
    the kernel computes, before the final bf16 rounding)."""
    import torch
    m, topk = ids.shape
    e_count, hdim = p["w2"].shape[0], p["w2"].shape[1]
    hd = qdq(fc1_ref(p, x, ids), p["a2g"])
    y = torch.zeros(m * topk, hdim, dtype=torch.float32, device=x.device)
    for e, pairs in _groups(ids, e_count).items():
        y[pairs] = (p["g2"][e].float() * (hd[pairs] @ _expert_w(p["w2"], p["w2_sf"], e).T)
                    * tw.reshape(-1)[pairs, None].float())
    valid = ((ids >= 0) & (ids < e_count)).reshape(-1, 1).float()
    y = (y * valid).reshape(m, topk, hdim)
    out = torch.zeros(m, hdim, dtype=torch.float32, device=x.device)
    for k in range(topk):  # the kernel's fixed k order
        out += y[:, k]
    return out


def route_twin(ids, e_count: int):
    """CPU twin of moe_route: (slot_expert, slot_off, slot_cnt, pair_list)
    with slots = min(E, P); active experts ascending, pairs ascending."""
    flat = [int(v) for v in ids.reshape(-1).tolist()]
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


def rand_routing(m, e_count, topk, device, seed=0):
    """Uniform random top-k experts per token (distinct), softmax weights."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.rand(m, e_count, generator=gen).topk(topk, -1).indices.int()
    tw = torch.softmax(torch.randn(m, topk, generator=gen), -1).float()
    return ids.to(device), tw.to(device)


def oracle_ok(rel_vs_ref: float, rel_vs_fi: float, fi_rel_vs_ref: float) -> bool:
    """Relative to vLLM's own error (like the dense gate): our error vs the
    exact reference within 10 % of FlashInfer's own (floor 1e-2), and our
    distance to FlashInfer bounded by the triangle inequality of both errors
    (floor 2e-2). NVFP4 requant of the intermediate makes bit-parity with a
    differently-ordered accumulation impossible; FlashInfer additionally maps
    gelu_tanh to erf-GeGLU (flashinfer_utils.py:47)."""
    return (rel_vs_ref <= max(1e-2, 1.1 * fi_rel_vs_ref)
            and rel_vs_fi <= max(2e-2, 2.2 * fi_rel_vs_ref))


def _rel(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


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
                          int(p["act"]), stream)
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
    if (info.get("tp"), info.get("ep"), info.get("dp")) != (1, 1, 1):
        return f"parallel tp/ep/dp={info.get('tp')}/{info.get('ep')}/{info.get('dp')} (single GPU only)"
    if info.get("expert_map"):
        return "expert_map (EP)"
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
                        expert_map=self.expert_map is not None)
            if info["quant_dtype"] != "nvfp4" or w13 is None or w13.dim() != 3:
                return info, None
            e, two_i, hh = w13.shape
            h, i = hh * 2, two_i // 2
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
                       max_m=max_m())
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
            """Ours vs the parent (vLLM FlashInfer cutlass_fused_moe) vs the
            exact reference on this layer's REAL weights. Fatal."""
            dev = self.w13_weight.device
            worst = 0.0
            for m in sorted({1, 8, cfg["max_m"]}):
                seed = zlib.crc32(str(self.layer_name).encode()) % 10007 + m
                gen = torch.Generator(device="cpu").manual_seed(seed)
                amax = 2688.0 / cfg["a1g"]  # calibrated activation range
                x = (torch.randn(m, info["H"], generator=gen) * (amax / 4.5)).to(
                    dev, torch.bfloat16)
                ids, tw = rand_routing(m, info["E"], cfg["topk"], dev, seed)
                fi = super().forward_modular(x, tw, ids).float()
                ours = self._sfx_run(cfg, x, tw, ids).float()
                ref = moe_ref(cfg, x, ids, tw).bfloat16().float()
                r_ref, r_fi, fi_ref = _rel(ours, ref), _rel(ours, fi), _rel(fi, ref)
                worst = max(worst, r_fi)
                if not (oracle_ok(r_ref, r_fi, fi_ref) and torch.isfinite(ours).all()):
                    raise RuntimeError(
                        f"{MARKER} LAYER ORACLE FAIL {self.layer_name} M={m}: rel_vs_ref="
                        f"{r_ref:.2e} rel_vs_flashinfer={r_fi:.2e} flashinfer_rel_vs_ref="
                        f"{fi_ref:.2e} — refusing to serve with {GATE}=1")
            torch.cuda.synchronize(dev)
            _log(f"LAYER ORACLE PASS {self.layer_name} E={info['E']} H={info['H']} "
                 f"I={info['I']} act={info['act']} max_rel_vs_flashinfer={worst:.2e}")
            return worst

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                            shared_experts_input=None):
            cfg = self._sfx_moe
            if (cfg is None or shared_experts is not None or x.dim() != 2
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
    (prepare_finalize/no_dp_ep.py quant + experts/flashinfer_cutlass_moe.py)."""
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
        input_sf=xsf, activation_type=activation_to_flashinfer_type(MoEActivation(p["act_name"])))
    return out


def _dev_problem(dev, e_count, hdim, idim, act, seed=0):
    import torch
    p = make_problem(e_count, hdim, idim, act, device=dev, seed=seed)
    p["a1g_t"] = torch.full((e_count,), p["a1g"], dtype=torch.float32, device=dev)
    p["a2g_t"] = torch.full((e_count,), p["a2g"], dtype=torch.float32, device=dev)
    return p


ORACLE_CASES = (GEMMA_MOE, (64, 2048, 768, 8, "silu"))
CONCURRENCY = (1, 8, 16, 32)  # decode tokens per step -> routed rows = M * topk


def oracle(cases=ORACLE_CASES, ms=CONCURRENCY):
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev).cuda_stream
    lines = []
    for e_count, hdim, idim, topk, act in cases:
        p = _dev_problem(dev, e_count, hdim, idim, act)
        ws = workspace(dev, max(ms), topk, hdim, idim, e_count)
        for m in ms:
            ids, tw = rand_routing(m, e_count, topk, dev, seed=m)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            ref = moe_ref(p, x, ids, tw).bfloat16().float()
            fi = _fi_call(p, x, ids, tw, torch.empty(m, hdim, dtype=torch.bfloat16,
                                                    device=dev)).float()
            ours = run_ours(native, p, x, ids, tw, ws, stream).float()
            again = run_ours(native, p, x, ids, tw, ws, stream).float()
            r_ref, r_fi, fi_ref = _rel(ours, ref), _rel(ours, fi), _rel(fi, ref)
            det = bool(torch.equal(ours, again))
            ok = oracle_ok(r_ref, r_fi, fi_ref) and det and bool(torch.isfinite(ours).all())
            lines.append(f"E={e_count} H={hdim} I={idim} top{topk} {act} M={m} P={m * topk}: "
                         f"rel_vs_ref={r_ref:.2e} rel_vs_flashinfer={r_fi:.2e} "
                         f"flashinfer_rel_vs_ref={fi_ref:.2e} deterministic={det} "
                         f"{'OK' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"{MARKER} NVFP4-MOE ORACLE FAIL: {lines[-1]}")
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


def bench(case=GEMMA_MOE, ms=CONCURRENCY, iters=200):
    """us per MoE layer call (x quant + experts + combine), CUDA graphs; one
    gemma layer = 1/30 of the MoE step."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    e_count, hdim, idim, topk, act = case
    p = _dev_problem(dev, e_count, hdim, idim, act, seed=1)
    ws = workspace(dev, max(ms), topk, hdim, idim, e_count)
    per_expert = 2 * idim * hdim * (1 / 2 + 1 / 16) + hdim * idim * (1 / 2 + 1 / 16)
    res = {}
    for m in ms:
        ids, tw = rand_routing(m, e_count, topk, dev, seed=100 + m)
        x = torch.randn(m, hdim, device=dev).bfloat16()
        out = torch.empty(m, hdim, dtype=torch.bfloat16, device=dev)
        t_fi = _graph_us(lambda s: _fi_call(p, x, ids, tw, out), dev, iters)
        t_ours = _graph_us(lambda s: run_ours(native, p, x, ids, tw, ws, s.cuda_stream, out),
                           dev, iters)
        distinct = int(torch.unique(ids).numel())
        roof = distinct * per_expert / HBM_BPS * 1e6
        res[m] = (t_fi, t_ours, roof)
        print(f"{MARKER} bench E={e_count} H={hdim} I={idim} top{topk} M={m} "
              f"routed_rows={m * topk} experts={distinct}: flashinfer {t_fi:.1f} us, ours "
              f"{t_ours:.1f} us (x{t_ours / t_fi:.2f}), weight-roofline {roof:.1f} us",
              file=sys.stderr, flush=True)
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
