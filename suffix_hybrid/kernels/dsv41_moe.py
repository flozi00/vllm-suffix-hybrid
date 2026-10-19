# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash routed-experts lane: the shape/arch gate (CPU-testable).

Contract (K1; checkpoint LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4, config.json
text_config + safetensors headers, 2026-10-19):

  layers 0..39 are ALL MoE (mlp_layer_types null: no dense/hash layer), 384 routed
  experts top-6 + 1 shared expert, hidden 5120, moe_intermediate 2304, silu,
  swiglu_limit 10.0, scoring sqrtsoftplus + noaux_tc bias, norm_topk_prob, routed
  scaling 1.5 (folded into the top-k weights by vLLM's dsv4_topk router: this lane
  gets final weights, sum = 1.5 per token, and applies nothing itself).
  Per expert e of layer l (tensor names `layers.l.ffn.experts.e.{w1,w2,w3}.*`):
    w1 (gate) / w3 (up) .weight  U8  [2304, 2560]  packed e2m1, low nibble first
    w2 (down)           .weight  U8  [5120, 1152]
    .weight_scale       F8_E4M3 [2304, 320] (w1/w3) | [5120, 144] (w2): block-16
    .weight_scale_2     F32 scalar (per expert, per matrix)
    .input_scale        F32 scalar (per expert, per matrix)
  vLLM's modelopt NVFP4 MoE method turns that into the FlashInfer CUTLASS layout
  this lane reads (w13 = [up; gate] U8 [E, 2I, H/2], 128x4-swizzled block scales
  per expert, g1/g2 per-expert alphas, ONE activation gscale per GEMM: the load
  checks gscale_shared and refuses a layer whose input scales differ).
  NOT NVFP4 (never this lane): shared experts (FP8 e4m3, e8m0 32x32 scales; in the
  quantization `ignore` list), gate (BF16), mtp.* / DSpark experts (MXFP4, e8m0).

Clamp (vLLM silu_and_mul_with_clamp / csrc/activation_kernels.cu HAS_CLAMP, the
DeepSeek V4 reference): gate = min(gate, L); up = clamp(up, -L, L); h = silu(gate)
* up, on the fc1 output BEFORE the activation, gate clamped from above only.

Layouts tuned and served (TP8 + expert parallel; config shapes per rank):
  EP8   48 local experts of 384, full I = 2304   (prod: FlashInfer CUTLASS refuses
        TP8 without EP, "intermediate size padding for w1 and w3")
  TP8   384 experts, I = 2304 / 8 = 288           (kernel-only: the lab tunes it,
        the stock lane cannot serve it)
Anything else carrying the clamp on an NVFP4 layer is a startup error naming the
supported set: `refuse` returns the reason, nvfp4_moe._sfx_prepare raises it.
"""
from __future__ import annotations

HIDDEN, TOPK, LIMIT = 5120, 6, 10.0
E_GLOBAL = 384
# (E local, I per rank) -> layout name; E_global is always 384
LAYOUTS = {(48, 2304): "EP8", (384, 288): "TP8-noEP"}
MAX_DECODE_M = 192  # 32 seqs x (1 + DSpark k 5)


def refuse(info: dict) -> str | None:
    """Why an NVFP4 clamp-carrying (DeepSeek-V4-class) layer cannot run on this
    lane, or None. `info` = nvfp4_moe's _sfx_info (+ topk, global_e). Called only
    for layers that are NVFP4 and carry a swiglu clamp; eligibility() has already
    passed (parallel contract, scale layout, gscale_shared)."""
    got = (info.get("E"), info.get("I"))
    if info.get("H") != HIDDEN:
        return f"hidden {info.get('H')} != {HIDDEN} (DeepSeek-V4.1-Flash)"
    if info.get("act") != "silu":
        return f"activation {info.get('act')} != silu"
    if info.get("topk") != TOPK:
        return f"topk {info.get('topk')} != {TOPK} (dsv4_topk router)"
    if info.get("clamp_limit") != LIMIT:
        return f"swiglu_limit {info.get('clamp_limit')} != {LIMIT}"
    if info.get("global_e") != E_GLOBAL:
        return f"{info.get('global_e')} routed experts != {E_GLOBAL}"
    if got not in LAYOUTS:
        return (f"layout E_local={got[0]} I={got[1]} not tuned; supported "
                + ", ".join(f"{n} (E_local {e}, I {i})" for (e, i), n in LAYOUTS.items())
                + " (TP8 + --enable-expert-parallel, or TP8 without EP)")
    ep = info.get("ep", 1)
    if (got == (48, 2304)) != (ep == 8):
        return f"layout {LAYOUTS[got]} needs ep={8 if got == (48, 2304) else 1}, got ep={ep}"
    return None
