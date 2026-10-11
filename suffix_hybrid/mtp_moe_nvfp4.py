# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_MTP_MOE_NVFP4=1: GLM 5.3's MTP draft-layer routed experts as NVFP4 W4A4.

GLM-5.3-NVFP4 quantizes the routed experts of layers 3..77 and excludes ``model.layers.78*``,
the MTP layer: its 256 BF16 experts run vLLM's unquantized MoE on every draft step (2.4-2.9 %
of a decode step at c1/c8/c32, nightly profile 2026-10-11). A draft only moves acceptance;
rejection sampling keeps outputs exact.

Two seams on vLLM's python side (rocm_patches after-hooks, no source anchors; same code in
0.30.0 and 0.31.1):
  1. ``get_quant_method`` of every ModelOpt config: ``ModelOptQuantConfigBase``'s (inherited by
     ``ModelOptNvFp4Config``, quantization=modelopt_fp4: the copy we serve declares quant_algo
     NVFP4) and ``ModelOptMixedPrecisionConfig``'s own (modelopt_mixed: HF's current export).
     A RoutedExperts at layer >= num_hidden_layers (the MTP layer) that the checkpoint leaves
     BF16 (method None: excluded or unlisted) gets ``ModelOptNvFp4FusedMoE`` with the NVFP4
     config of layers 3..77 (the modelopt_fp4 config itself / the mixed NVFP4 sub-config), so
     parameters, backend selection (FlashInfer CUTLASS, our SUFFIX_NVFP4_MOE kernel), EP / TP
     sharding and post-load processing are exactly theirs. One line per decision.
  2. ``DeepseekV32MTP.load_weights``: the layer's BF16 expert tensors are quantized on the way
     in into the checkpoint's NVFP4 tensor set (weight uint8, weight_scale e4m3 in the linear
     layout, weight_scale_2, input_scale) with vLLM's scaled_fp4_quant; gate and up share one
     global scale (vLLM serves w13 with ``w13_weight_scale_2[:, 0]`` for both halves).

Activation input scales: the checkpoint's calibrated scales of the last quantized MoE layer
(num_hidden_layers - 1), max over its experts and over gate/up, read through the safetensors
index (layer 77 sits in a later file than layer 78). FlashInfer's NVFP4 MoE shares one input
scale per GEMM across experts anyway (max over experts). Every global expert gets them, also
those an EP weight filter never streams.

Fail closed: every converted tensor must dequantize to within WEIGHT_REL_BOUND (rel-l2) of its
BF16 weight, and after vLLM's processing the layer's FlashInfer path (RoutedExperts.forward_
modular, global routing ids) must match a fp32 BF16-weight reference of this rank's shard for
M in ORACLE_MS within REL_BOUND; and with the gate on, a draft load that switched or converted
nothing is a loud FAIL line + error. Otherwise the pod refuses to start.
"""
from __future__ import annotations

import json
import os
import re
import sys

GATE = "SUFFIX_MTP_MOE_NVFP4"
MARK = "[suffix mtp-moe-nvfp4]"
FP4_MAX = 6.0 * 448.0  # e2m1 max x e4m3 max: the NVFP4 global-scale convention
WEIGHT_REL_BOUND = 0.2  # random-like weights quantize to ~0.1 rel-l2; a bad pack is ~1
REL_BOUND = 0.3         # W4A4 layer output vs BF16; a w1/w3 swap or bad layout lands near 1
ORACLE_MS = (1, 6, 32)
KEEP = 2                # BF16 reference experts kept per rank (CPU tensors from the loader)
_EXPERT = re.compile(r"^(?P<base>model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<e>\d+))\."
                     r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
_state: dict = {"keep": {}, "scales": None, "tensors": 0, "saved": 0, "switched": {}}


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def _fail(why: str):
    _say(f"FAIL: {GATE}=1 but {why} — refusing to start")
    raise RuntimeError(f"{MARK} {GATE}=1 but {why}")


def _model_config():
    from vllm.config import get_current_vllm_config  # set while the model is constructed

    return get_current_vllm_config().model_config


def _num_hidden_layers() -> int:
    return int(_model_config().hf_config.num_hidden_layers)


def is_mtp_experts(prefix: str, num_hidden_layers: int) -> bool:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$", prefix)
    return m is not None and int(m[1]) >= num_hidden_layers


def input_scales(path: str, layer: int) -> tuple[float, float]:
    """(w13, w2) input scale: max over layer `layer`'s experts in the checkpoint."""
    from safetensors import safe_open

    index = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise RuntimeError(f"{MARK} no {index}: the calibrated input scales need a local checkpoint")
    wmap = json.load(open(index))["weight_map"]
    pre = f"model.layers.{layer}.mlp.experts."
    by_file: dict = {}
    for n in wmap:
        if n.startswith(pre) and n.endswith(".input_scale"):
            by_file.setdefault(wmap[n], []).append(n)
    a13 = a2 = 0.0
    for f, names in by_file.items():
        with safe_open(os.path.join(path, f), framework="pt") as fh:
            for n in names:
                v = float(fh.get_tensor(n).float().max())
                if ".down_proj." in n:
                    a2 = max(a2, v)
                else:
                    a13 = max(a13, v)
    if not (a13 > 0 and a2 > 0):
        raise RuntimeError(f"{MARK} layer {layer} has no positive expert input scales in {index}")
    return a13, a2


def fp4_quant(w, gscale):
    """BF16 [N, K] -> (packed uint8 [N, K/2], e4m3 [N, K/16] linear) like the checkpoint."""
    from vllm._custom_ops import scaled_fp4_quant

    q, sf = scaled_fp4_quant(w, gscale, is_sf_swizzled_layout=False)
    return q, sf.view(-1).view(w.shape[0], w.shape[1] // 16)


def _check_weight(name, w, q, sf, gscale) -> None:
    import torch

    from suffix_hybrid.kernels.nvfp4_moe import dequant

    deq = dequant(q, sf.float()) / gscale
    rel = float((deq - w.float()).norm() / w.float().norm().clamp_min(1e-30))
    if not (rel <= WEIGHT_REL_BOUND and bool(torch.isfinite(deq).all())):
        raise RuntimeError(f"{MARK} {name}: NVFP4 dequant rel-l2 {rel:.3e} > {WEIGHT_REL_BOUND} "
                           f"vs BF16 — refusing to serve with {GATE}=1")


def convert(weights, first_mtp: int, n_experts: int, scales, local=lambda e: True,
            quant=fp4_quant, check=_check_weight, device=None):
    """The checkpoint stream with every BF16 MTP routed-expert weight replaced by its NVFP4
    tensor set; `scales()` -> (a13, a2) runs on the first MTP expert; `local(e)` picks the
    BF16 references the load oracle keeps."""
    import torch

    pending: dict = {}
    seen: dict = {}
    a13 = a2 = None
    for name, w in weights:
        m = _EXPERT.match(name)
        if m is None or int(m["layer"]) < first_mtp or not w.is_floating_point() \
                or w.dtype == torch.float8_e4m3fn:
            yield name, w
            continue
        if a13 is None:
            a13, a2 = scales()
        layer, e, proj = int(m["layer"]), int(m["e"]), m["proj"]
        seen.setdefault(layer, set()).add(e)
        if proj == "down_proj":
            group = {"down_proj": w}
        else:
            half = pending.setdefault(m["base"], {})
            half[proj] = w
            if len(half) < 2:
                continue
            group = pending.pop(m["base"])
        dev = device or torch.device("cuda", torch.cuda.current_device())
        ws = {p: t.to(dev, torch.bfloat16) for p, t in group.items()}
        amax = max(float(t.abs().amax()) for t in ws.values())
        gscale = torch.tensor(FP4_MAX / max(amax, 1e-30), dtype=torch.float32, device=dev)
        a = a2 if proj == "down_proj" else a13
        for p, t in ws.items():
            full = f"{m['base']}.{p}"
            q, sf = quant(t, gscale)
            check(full, t, q, sf, gscale)
            _state["tensors"] += 1
            _state["saved"] += t.numel() * 2 - q.numel() - sf.numel()
            yield f"{full}.weight", q
            yield f"{full}.weight_scale", sf
            yield f"{full}.weight_scale_2", (1.0 / gscale).reshape(())
            yield f"{full}.input_scale", torch.tensor(a, dtype=torch.float32)
        keep = _state["keep"].setdefault(layer, {})
        if (e in keep or len(keep) < KEEP) and local(e):
            keep.setdefault(e, {}).update(group)  # loader CPU tensors: no device memory held
    if pending:
        raise RuntimeError(f"{MARK} gate/up halves never paired: {sorted(pending)[:4]}")
    for layer, got in seen.items():  # global input scales also for experts never streamed here
        for e in range(n_experts):
            if e not in got:
                base = f"model.layers.{layer}.mlp.experts.{e}"
                for p, a in (("gate_proj", a13), ("up_proj", a13), ("down_proj", a2)):
                    yield f"{base}.{p}.input_scale", torch.tensor(a, dtype=torch.float32)


def oracle(layer, layer_idx: int, a13: float, a2: float, ms=ORACLE_MS, forward=None) -> float:
    """FlashInfer path of the converted layer vs fp32 math on the kept BF16 weights (this rank's
    TP shard), global routing ids, M in `ms`. Random hidden states are scaled into the calibrated
    FC1 input range and down until the FC2 input stays inside half its calibrated range: random
    data has far more energy than real activations at the same amax, and clipping there would
    measure the calibration, not the conversion. Returns the worst rel-l2; raises past
    REL_BOUND."""
    import torch

    if forward is None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        forward = RoutedExperts.forward_modular  # vLLM's path, not our SUFFIX_NVFP4_MOE kernel
    keep = _state["keep"].pop(layer_idx, {})
    check = sorted(e for e in keep if layer._map_global_expert_id_to_local_expert_id(e) >= 0
                   and len(keep[e]) == 3)[:KEEP]
    if not check:
        raise RuntimeError(f"{MARK} oracle: no complete BF16 reference expert is local to this "
                           f"rank (kept {sorted(keep)}) — refusing to serve with {GATE}=1")
    mc = layer.moe_config
    tp, r = mc.moe_parallel_config.tp_size, mc.moe_parallel_config.tp_rank
    dev = layer.w13_weight.device
    w = {}
    for e in check:
        w1, w3, w2 = (keep[e][p].to(dev, torch.float32) for p in ("gate_proj", "up_proj", "down_proj"))
        ip = w1.shape[0] // tp
        w[e] = (w1[r * ip:(r + 1) * ip], w3[r * ip:(r + 1) * ip], w2[:, r * ip:(r + 1) * ip])
    others = [e for e in range(layer.global_num_experts) if e not in check
              and layer._map_global_expert_id_to_local_expert_id(e) >= 0]
    k = layer.top_k
    slots = (check + others)[:k]
    hdim = w[check[0]][0].shape[1]
    worst = 0.0
    silu = torch.nn.functional.silu

    def h_amax(xf):
        return max(float((silu(xf @ w1.t()) * (xf @ w3.t())).abs().amax()) for w1, w3, _ in w.values())

    for m in ms:
        gen = torch.Generator(device="cpu").manual_seed(7919 + m)
        xu = torch.randn(m, hdim, generator=gen).to(dev)
        sc = a13 * FP4_MAX / float(xu.abs().amax())
        for _ in range(4):
            hm = h_amax(xu * sc)
            if hm <= 0.5 * a2 * FP4_MAX:
                break
            sc *= (0.5 * a2 * FP4_MAX / hm) ** 0.5
        x = (xu * sc).to(torch.bfloat16)
        ids = torch.tensor(slots, dtype=torch.int32, device=dev).expand(m, k).contiguous()
        tw = torch.zeros(m, k, dtype=torch.float32, device=dev)
        tw[:, :len(check)] = torch.rand(m, len(check), generator=gen).to(dev) + 0.25
        out = forward(layer, x.clone(), tw.clone(), ids.clone()).float()
        ref = torch.zeros(m, hdim, dtype=torch.float32, device=dev)
        for j, e in enumerate(check):
            w1, w3, w2 = w[e]
            xf = x.float()
            ref += tw[:, j:j + 1] * ((silu(xf @ w1.t()) * (xf @ w3.t())) @ w2.t())
        rel = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
        worst = max(worst, rel)
        if not (rel <= REL_BOUND and bool(torch.isfinite(out).all())):
            raise RuntimeError(f"{MARK} LAYER ORACLE FAIL layer {layer_idx} M={m} experts {check} "
                               f"rel-l2 {rel:.3e} > {REL_BOUND} vs BF16 — refusing to serve with {GATE}=1")
    return worst


def patch_quant(module) -> None:
    """after-hook on vllm.model_executor.layers.quantization.modelopt: wraps every ModelOpt
    config class that defines get_quant_method (the base, inherited by modelopt_fp4, and
    modelopt_mixed's override)."""
    for cls in [c for c in vars(module).values() if isinstance(c, type)
                and issubclass(c, module.ModelOptQuantConfigBase) and "get_quant_method" in vars(c)]:
        cls.get_quant_method = _switching(module, cls.get_quant_method)


def _switching(module, orig):
    def get_quant_method(self, layer, prefix):
        qm = orig(self, layer, prefix)
        if qm is not None or not isinstance(layer, module.RoutedExperts) \
                or not is_mtp_experts(prefix, _num_hidden_layers()):
            return qm
        nv = self if isinstance(self, module.ModelOptNvFp4Config) else getattr(self, "nvfp4_config", None)
        seen = f"{prefix}: {type(self).__name__} leaves it BF16 (excluded={self.is_layer_excluded(prefix)})"
        if nv is None:
            _fail(f"{seen} and has no NVFP4 config to serve it with")
        _say(f"{seen} -> ModelOptNvFp4FusedMoE with the NVFP4 config of the main MoE layers")
        qm = module.ModelOptNvFp4FusedMoE(quant_config=nv, moe_config=layer.moe_config)
        layer_idx = int(re.search(r"layers\.(\d+)\.mlp\.experts$", prefix)[1])
        # load_weights converts only layers switched here (their params are NVFP4 now)
        _state["switched"][layer_idx] = _model_config().model
        post = qm.process_weights_after_loading

        def process_weights_after_loading(lay, _post=post):
            _post(lay)
            if lay is layer:
                rel = oracle(lay, layer_idx, *_state["scales"])
                tp = lay.moe_config.moe_parallel_config.tp_size
                _say(f"ACTIVE {prefix}: BF16 experts -> NVFP4 W4A4 ({qm.nvfp4_backend.name}), "
                     f"{_state['tensors']} tensors, {_state['saved'] / tp / 2**30:.2f} GiB saved on "
                     f"this rank, input scales a13={_state['scales'][0]:.4g} a2={_state['scales'][1]:.4g} "
                     f"(layer {layer_idx - 1}), oracle max rel-l2 vs BF16 {rel:.3e} (bound {REL_BOUND})")

        qm.process_weights_after_loading = process_weights_after_loading
        return qm

    return get_quant_method


def patch_mtp(module) -> None:
    """after-hook on vllm.models.deepseek_v32.nvidia.mtp."""
    cls = module.DeepseekV32MTP
    orig = cls.load_weights

    def load_weights(self, weights):
        if not _state["switched"]:
            _fail("no MTP routed-expert layer was switched to NVFP4 (no decision line above: the "
                  "draft's quant config is not ModelOpt, the checkpoint already quantizes them, or "
                  "vLLM moved the get_quant_method seam)")
        first = min(_state["switched"])
        path = _state["switched"][first]
        experts = [getattr(lay.mtp_block.mlp.experts, "routed_experts", lay.mtp_block.mlp.experts)
                   for lay in self.model.layers.values()
                   if hasattr(getattr(lay.mtp_block, "mlp", None), "experts")]  # MoERunner -> RoutedExperts

        def scales():
            if _state["scales"] is None:
                _state["scales"] = input_scales(path, first - 1)
            return _state["scales"]

        def local(e):
            return all(x._map_global_expert_id_to_local_expert_id(e) >= 0 for x in experts)

        out = orig(self, convert(weights, first, int(self.config.n_routed_experts), scales, local))
        if not _state["tensors"]:
            _fail(f"layers {sorted(_state['switched'])} were switched to NVFP4 and the checkpoint "
                  "stream had no BF16 MTP expert weight to convert")
        return out

    cls.load_weights = load_weights
