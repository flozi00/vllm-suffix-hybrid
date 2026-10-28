# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared-expert activation: clamped SwiGLU + MXFP8 quantisation in one launch.

Stock (nightly 39983a5e7), per MoE layer, on the aux stream next to the routed experts:

  DeepseekV4MLP.forward: gate_up_proj (MXFP8 GEMM) -> SiluAndMulWithClamp (CUDA op, bf16) ->
  down_proj, whose FlashInferCutlassMxfp8LinearKernel quantises its bf16 input with
  flashinfer.mxfp8_quantize (cute-dsl) in a launch of its own.

torch.compile is off for this architecture (no @support_torch_compile), so vLLM's act+quant
fusion pass never runs. This kernel is that fusion: it reads gate_up [M, 2d] (d = 2304 / 8 = 288
per rank), applies the clamped SwiGLU exactly as csrc/activation_kernels.cu does (packed path,
rounding boundaries below), and writes the MXFP8 activation (e4m3 values + UE8M0 scales in the
F8_128x4 swizzle) that down_proj consumes through ``QuantizedActivation``
(vllm/model_executor/layers/fusion/quant_activation.py), skipping the quant launch:
-1 launch per MoE layer = -40 per step.

K1 contract
  gate_up bf16 [M, 2d] = [gate; up]   limit = swiglu_limit (10.0)
  g'  = bf16(min(f32(gate), limit))                    (gate clamped above only, rounded to bf16)
  sil = bf16(g' / (1 + exp(-g')))                      (the silu is rounded to bf16 before the multiply)
  out = bf16(sil * clamp(f32(up), -limit, limit))      (the up half is clamped in f32, not rounded)
  q, qs = MXFP8(out): per 32 columns UE8M0 scale (FlashInfer's conversion), e4m3 values, scales in
          F8_128x4 order; padding rows and padding group columns zero (d = 288 is 9 groups -> 12).
Numerics: bit-exact vs the torch transcription of that CUDA kernel on CPU; on the GPU the f32 exp
(Triton's fast exp vs CUDA expf) can differ by an ulp before the bf16 rounding, flipping a rare
element by one bf16 / fp8 step: a "reordering"-class change, quality-gated (K6), never claimed exact.

Search space: bt 1..16 tokens x bg 1..16 groups of 32 columns (powers of two).
Knob: SUFFIX_HELION_DSV41_SWIGLU = oxide (default = stock) | auto | helion; wiring in
kernels/dsv41_fuse_hook.py (DeepseekV4MLP.forward). Boot gate: dsv41_swiglu_quant_capture_replay.
"""
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_common as cm
from suffix_hybrid.helion_kernels.dsv41_common import (  # noqa: F401  (traced into the body)
    mxfp8_group_quant, mxfp8_scale_offsets)

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU host: the module stays importable (lab ABI, tests)
    helion = hl = None

ENV = "SUFFIX_HELION_DSV41_SWIGLU"
KERNEL = "dsv41_swiglu_quant"
MARKER = "[suffix dsv41-swiglu]"
INTERMEDIATE = 2304            # moe_intermediate_size x n_shared_experts (1)
LIMIT = 10.0                   # swiglu_limit
BUCKETS = (1, 2, 4, 6, 12, 24, 48, 96, 192)
MAX_TOKENS = BUCKETS[-1]
SEARCH_SPACE = {KERNEL: {"bt": (1, 16), "bg": (1, 16)}}


def dsv41_swiglu_quant(gate_up: torch.Tensor, limit: float):
    """gate_up bf16 [M, 2d] -> (q e4m3 [M, d], qs uint8 [ceil128(M) * ceil4(d/32) * 4])."""
    M = gate_up.size(0)
    D2 = hl.specialize(gate_up.size(1))
    G = D2 // 64                                       # 32-column groups of one half
    bt = hl.register_block_size(1, 16)
    bg = hl.register_block_size(1, 16)
    q = torch.empty([M, G * 32], dtype=torch.float8_e4m3fn, device=gate_up.device)
    qs = torch.empty([(M + 127) // 128 * 128 * ((G + 3) // 4) * 4], dtype=torch.uint8,
                     device=gate_up.device)
    gu4 = gate_up.view(M, 2, G, 32)
    q3 = q.view(M, G, 32)
    for tile_t in hl.tile(M, block_size=bt):
        for tile_g in hl.tile(G, block_size=bg):
            g = gu4[tile_t, 0, tile_g, :].to(torch.float32)             # [bt, bg, 32]
            u = gu4[tile_t, 1, tile_g, :].to(torch.float32)
            gc = torch.clamp(g, max=limit).to(torch.bfloat16).to(torch.float32)
            sil = (gc / (1.0 + torch.exp(-gc))).to(torch.bfloat16).to(torch.float32)
            out = (sil * torch.clamp(u, min=-limit, max=limit)).to(torch.bfloat16)
            qv, sf = mxfp8_group_quant(out.to(torch.float32))
            q3[tile_t, tile_g, :] = qv
            qs[mxfp8_scale_offsets(tile_t.index, tile_g.index, G)] = sf.to(torch.uint8)
        # F8_128x4 scale padding: the padding group columns of this tile's rows, and (last tile
        # only) the padding rows, are zero like the stock quantiser writes them
        for tile_gp in hl.tile(G, (G + 3) // 4 * 4, block_size=4):
            qs[mxfp8_scale_offsets(tile_t.index, tile_gp.index, G)] = hl.zeros(
                [tile_t, tile_gp], dtype=torch.uint8)
        if tile_t.begin + bt >= M:
            for tile_r in hl.tile(M, (M + 127) // 128 * 128, block_size=32):
                for tile_gp in hl.tile((G + 3) // 4 * 4, block_size=32):
                    qs[mxfp8_scale_offsets(tile_r.index, tile_gp.index, G)] = hl.zeros(
                        [tile_r, tile_gp], dtype=torch.uint8)
    return q, qs


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def case(tokens: int, d: int = 288) -> dict:
    return {"numtokens": tokens, "d": d}


def serving_case(tokens: int, d: int) -> dict:
    return case(hk.bucket(tokens, BUCKETS), d)


_WARM: set = set()


def swiglu_quant(gate_up, limit=LIMIT):
    """-> (q, qs) for the down_proj input; gate_up bf16 [M, 2d] contiguous, d % 32 == 0."""
    if gate_up.dtype != torch.bfloat16 or gate_up.dim() != 2 or not gate_up.is_contiguous():
        raise ValueError(f"{MARKER} need a contiguous bf16 [M, 2d] gate_up, got "
                         f"{gate_up.dtype} {tuple(gate_up.shape)}")
    if gate_up.shape[1] % 64:
        raise ValueError(f"{MARKER} 2d = {gate_up.shape[1]}: d must be a multiple of 32")
    c = serving_case(gate_up.shape[0], gate_up.shape[1] // 2)
    key = (KERNEL, c["numtokens"], c["d"])
    if key not in _WARM:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{MARKER} {key} first used while capturing a CUDA graph: warm it eagerly")
        _WARM.add(key)
    return hk.compiled(dsv41_swiglu_quant, KERNEL, c)(gate_up, float(limit))


def missing(d: int = 288) -> list:
    return [b for b in BUCKETS if hk.config(KERNEL, case(b, d)) is None]


# ---------------------------------------------------------------------------
# exact references
# ---------------------------------------------------------------------------
def swiglu_reference(gate_up, limit=LIMIT):
    """The CUDA kernel's packed path in torch f32: -> out bf16 [M, d]."""
    d = gate_up.shape[1] // 2
    g = gate_up[:, :d].to(torch.float32)
    u = gate_up[:, d:].to(torch.float32)
    gc = torch.clamp(g, max=limit).to(torch.bfloat16).to(torch.float32)
    sil = (gc / (1.0 + torch.exp(-gc))).to(torch.bfloat16).to(torch.float32)
    return (sil * torch.clamp(u, min=-limit, max=limit)).to(torch.bfloat16)


def reference(gate_up, limit=LIMIT):
    return cm.mxfp8_quant_reference(swiglu_reference(gate_up, limit))


def judge(got, gate_up, limit=LIMIT, exact=False):
    """-> (ok, msg). exact: bit-identical (the CPU stand-in runs the same torch ops); otherwise
    <= 0.5% of the q bytes / scale bytes may differ (GPU exp ulps) and every padding byte of the
    swizzled scales must be zero. Always: dequantised error vs the f32 SwiGLU <= 1 fp8 step."""
    q, qs = got
    q_ref, qs_ref = reference(gate_up, limit)
    q_bad = float((q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean())
    s_bad = float((qs != qs_ref).float().mean())
    t, d = q_ref.shape
    groups_mask = torch.zeros(qs_ref.numel(), dtype=torch.bool)
    rows = torch.arange(t)
    g4 = cm.cdiv(d // 32, 4)
    for gi in range(d // 32):
        groups_mask[((rows // 128) * g4 + gi // 4) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 + gi % 4] = True
    pad_ok = bool((qs[~groups_mask] == 0).all())
    ok = pad_ok and (q_bad + s_bad == 0.0 if exact else (q_bad <= 5e-3 and s_bad <= 5e-3))
    deq = cm.mxfp8_dequant(q, qs).float()
    full = swiglu_reference(gate_up, limit).float()
    err = float(((deq - full).abs() / (full.abs().amax(-1, keepdim=True) + 1e-30)).max())
    ok = ok and err < 0.07
    return ok, f"q bytes differ {q_bad:.1e}, scale bytes differ {s_bad:.1e}, padding {'zero' if pad_ok else 'DIRTY'}, deq err {err:.1e}"


# ---------------------------------------------------------------------------
# lab ABI
# ---------------------------------------------------------------------------
def _is_v41(model: dict) -> bool:
    return str(model.get("architecture", "")).startswith(("DeepseekV41", "DeepseekV4"))


def cases(model: dict) -> list:
    if not _is_v41(model):
        return []
    tp = int(model.get("tp", 8) or 8)
    d = INTERMEDIATE // tp
    return [case(b, d) for b in BUCKETS] if INTERMEDIATE % tp == 0 and d % 32 == 0 else []


def make_problem(tokens: int, d: int = 288, seed: int = 0, device: str = "cpu", hot: float = 4.0):
    """gate_up ~ N(0, hot) so the clamp engages on a few percent of the elements."""
    g = torch.Generator().manual_seed(500 + seed)
    return (torch.randn(tokens, 2 * d, generator=g) * hot).to(torch.bfloat16).to(device)


def default_config(case_: dict) -> dict:
    t = case_["numtokens"]
    return {"block_sizes": [min(2, 1 << max(0, (t - 1).bit_length())), 4], "num_warps": 4}


_LAST: dict = {}


def _make(c, seed):
    gu = make_problem(c["numtokens"], c["d"], seed, cm.DEVICE)
    _LAST["gu"] = gu
    return (gu, LIMIT)


def _baseline(gate_up, limit):
    """Stock: the CUDA SwiGLU op, then the quant launch down_proj's kernel would run."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    out = SiluAndMulWithClamp(limit)(gate_up)
    return mxfp8_e4m3_quantize(out, is_sf_swizzled_layout=True)


def _fi_parity(args, got):
    """flashinfer.mxfp8_quantize on the exact bf16 SwiGLU output: <= 0.5 % of the bytes may differ
    (the GPU exp ulps move a rare element by one step), the padding must not."""
    gu, limit = args
    out = swiglu_reference(gu.cpu(), limit).to(gu.device)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    q_fi, qs_fi = mxfp8_e4m3_quantize(out.contiguous(), is_sf_swizzled_layout=True)
    q_bad = float((got[0].view(torch.uint8) != q_fi.view(torch.uint8)).float().mean())
    s_bad = float((got[1] != qs_fi.view(-1)).float().mean()) if got[1].numel() == qs_fi.numel() else 1.0
    return q_bad <= 5e-3 and s_bad <= 5e-3, f"flashinfer parity: q differs {q_bad:.1e}, scales differ {s_bad:.1e}"


SILICON_CHECKS = {KERNEL: _fi_parity}

KERNELS = [
    {"name": KERNEL, "body": dsv41_swiglu_quant, "cases": cases, "make": _make,
     "reference": lambda gate_up, limit: tuple(t.cpu() for t in reference(gate_up.cpu(), limit)),
     "baseline": _baseline,
     "baseline_name": "SiluAndMulWithClamp (CUDA) + flashinfer.mxfp8_quantize (cute-dsl): 2 launches",
     "check": lambda got, ref: judge(tuple(t.cpu() for t in got), _LAST["gu"].cpu()),
     "replaces": [r"act_and_mul_kernel", r"mxfp8.*quant|quantize.*mxfp8|cvt.*mxfp8"]},
]


# ---------------------------------------------------------------------------
# boot gate dsv41_swiglu_quant_capture_replay
# ---------------------------------------------------------------------------
def dsv41_swiglu_quant_capture_replay(tokens: int = 6, d: int = 288, cfg=None, runner_factory=None) -> bool:
    import helion as _h
    c = serving_case(tokens, d)
    kern = _h.kernel(dsv41_swiglu_quant, config=_h.Config(**(cfg or hk.config(KERNEL, c) or default_config(c))),
                     static_shapes=False)
    return cm.capture_replay_gate(KERNEL, lambda gu: kern(gu, LIMIT),
                                  lambda i: [make_problem(tokens, d, i, cm.DEVICE)],
                                  runner_factory=runner_factory)


BOOT_GATES = {f"{KERNEL}_capture_replay": dsv41_swiglu_quant_capture_replay}


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog=__name__)
    ap.add_argument("cmd", choices=["oracle", "bench", "capture_replay"])
    ap.add_argument("--tokens", type=int, default=6)
    ns, _ = ap.parse_known_args(argv)
    if ns.cmd == "capture_replay":
        return 0 if all(g(tokens=ns.tokens) for g in BOOT_GATES.values()) else 1
    from suffix_hybrid.kernels import dsv41_bench
    return dsv41_bench.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
