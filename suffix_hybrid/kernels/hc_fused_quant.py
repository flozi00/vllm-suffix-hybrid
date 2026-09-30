# SPDX-License-Identifier: Apache-2.0
"""Fused NVFP4 activation quant in the Qwen4Exp HyperConnection glue kernels
(qwen3.8-flash-next, vLLM 0.30.0 vllm/models/qwen4_exp/nvidia/{hyperconnection,
ops/hc}.py). Gate: ``SUFFIX_HC_FUSED_QUANT=1`` (default OFF; needs
``SUFFIX_NVFP4_DENSE=1``: the consumers are the linears it converts).

Why: every NVFP4 dense linear runs a standalone activation quant launch
(vLLM ``cvt_fp16_to_fp4`` on the FlashInfer route, our ``nvfp4_quant_act``
on the decode-GEMM route) because its input comes from a HC glue Triton kernel
that vLLM's norm/act+quant fusion passes cannot see. Per ``GatedResidual``
boundary (combine_and_mix / mix):

  producer (fused here)                         -> NVFP4 consumer(s)
  A  hc_combine_norm / grouped_gemma_rmsnorm xn -> input_mix_weight_down(_block_inject)
  B  hc_silu(lora)                              -> input_mix_weight_up
  C  hc_gate_mix(xn, gate) = block_input        -> decoder-layer block linears:
       attn HC: linear_attn.in_proj_qkvz / in_proj_ba, self_attn.qkv_proj /
                self_attn.indexer.index_qk_proj;
       mlp HC:  mlp.gate_up_proj / mlp.shared_expert.gate_up_proj

Each fused kernel (hc_fused_quant_triton.py) is the stock vLLM Triton kernel
body verbatim plus an epilogue that ALSO writes the NVFP4 input of its consumer (packed e2m1
[M, K/2], element 2i in the low nibble + e4m3 block-16 scales in CUTLASS's
128x4 swizzled layout [round_up(M, 128), K/16], padded rows zeroed) using
that consumer's STATIC input global scale; the consumer then takes a vLLM
``QuantizedActivation`` (kNvfp4Dynamic, the fusion-pass contract) for every
M: our SuffixNvFp4LinearKernel prequant route (asf_mode 1) for M <= its max
M, FlashInfer CUTLASS with pre-quantized input above. B writes no bf16 at all
(nobody else reads hc_silu's output); A and C still write their bf16 outputs
(read by hc_gate_mix, the router / MoE experts / other bf16 consumers).
A, B and C cannot be merged into one kernel: a GEMM sits between each.
The norm epilogues re-read the bf16 row they just stored (after a CTA
barrier) instead of reshaping registers, so the stock reduction's layout and
order - hence its bf16 bits - cannot be perturbed by the added code.

Numeric contract (one, everywhere): the NVFP4 spec with IEEE math on the
bf16-ROUNDED glue output y, bit-identical to ``nvfp4_gemm.quantize`` and to
our decode GEMM's in-kernel quant (oxide quant_block):
    sf = e4m3_rne(min(amax16(y) * fp32(g / 6), 448)),
    q  = e2m1_rne(y * (g / sf))  (0 when sf == 0; sign = y*inv < 0),
with g = layer.input_global_scale_inv and IEEE div (tl.math.div_rn). (The
oxide cvt.rn.satfinite.e2m1 keeps the sign of an exact -0.0 input, code 0x8
vs 0x0 here: the same value, so GEMM outputs agree.) vLLM's
scaled_fp4_quant (rcp.approx) differs on some block scales, so vs the
unfused FlashInfer route the check is the triangle bound (``oracle_ok``);
vs the unfused decode-GEMM route (same spec quant) outputs are bit-identical.

Wiring (``wire``, called by the SUFFIX_NVFP4_DENSE converter right after
conversion, before any CUDA graph): per GatedResidual, detect its pairs,
fail closed per pair with a logged reason (consumer not NVFP4 kNvfp4Dynamic,
K padding, K % 64, bias, > QMAX distinct scales among C consumers, ...),
run the LOAD-TIME ORACLE on the real weights (fused bf16 outputs == stock
vLLM op bits; q / sf == stock op -> spec quant bits; padded scale rows 0;
consumer GEMM on the fused input == unfused GEMM bit-for-bit on our route,
``oracle_ok`` vs the f64 exact product on the FlashInfer route; M in
ORACLE_MS; fatal on mismatch), then switch the module's mix /
combine_and_mix to the fused path. C consumers are called by vLLM model code
with the bf16 tensor; their ``quant_method.apply`` is wrapped to swap in the
pending prequantized activation when handed that same buffer (data_ptr +
numel + width; consumed once, a miss logs once and runs unfused; q / sf get
record_stream'ed when the consumer runs on another stream, e.g. vLLM's
aux-stream MoE shared experts). Qwen4Exp is not torch.compiled (eager + CUDA
graphs), so this Python runs at capture.

CLI / boot gates (pod, SM120): ``python -m suffix_hybrid.kernels.hc_fused_quant
oracle`` (``hc_fused_quant_oracle``: every fused op vs stock op + spec quant
at the qwen shapes, M in BENCH_MS, zeros / saturation rows, + FlashInfer GEMM
on fused vs vLLM-quant input vs exact) and ``bench`` (``hc_fused_quant_bench``:
us per HC boundary glue+quant, fused vs unfused, CUDA graphs).
"""
from __future__ import annotations

import functools
import os
import sys
import types

import numpy as np

GATE = "SUFFIX_HC_FUSED_QUANT"
MARKER = "[suffix hc-fused-quant]"
QMAX = 2  # distinct consumer global scales one gate-mix launch can emit
ORACLE_MS = (1, 5, 160)  # load time: M=1/5 -> our GEMM route, 160 -> FlashInfer
BENCH_MS = (1, 5, 40, 160, 2700)
# block-input consumers relative to the decoder layer, per HC attribute
BLOCK_CONSUMERS = {
    "attn_hyper_connection": ("linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba",
                              "self_attn.qkv_proj", "self_attn.indexer.index_qk_proj"),
    "mlp_hyper_connection": ("mlp.gate_up_proj", "mlp.shared_expert.gate_up_proj"),
}
_V = types.SimpleNamespace(QA=None, KEY=None, ops=None)  # vLLM bits (tests inject fakes)
_state = {"missed": set(), "pairs": {}}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


# ---------------------------------------------------------------------------
# numeric contract (torch, CPU or GPU): the spec quant + its swizzled layout
# ---------------------------------------------------------------------------
def g6_of(g: float) -> float:
    """fp32(g) / fp32(6), exactly as the spec (and the oxide kernel) rounds it."""
    return float(np.float32(g) / np.float32(6.0))


def spec_quant(y, g: float):
    """[R, K] values (bf16-representable) -> (packed uint8 [R, K/2], e4m3 bits
    uint8 [R, K/16]); bit-identical to nvfp4_gemm.quantize (IEEE math)."""
    import torch
    r, k = y.shape
    blk = y.float().reshape(r, k // 16, 16)
    g32 = torch.tensor(np.float32(g), device=y.device)
    g6 = torch.tensor(np.float32(g6_of(g)), device=y.device)
    sf8 = torch.clamp(blk.abs().amax(-1) * g6, max=448.0).to(torch.float8_e4m3fn)
    sff = sf8.float()
    inv = torch.where(sff == 0, torch.zeros_like(sff), g32 / torch.where(sff == 0, 1.0, sff))
    s = blk * inv[..., None]
    code = _e2m1_code(s).reshape(r, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, sf8.view(torch.uint8)


def _e2m1_code(s):
    """e2m1 code of s (RNE on the grid, ties to the even code, saturating at
    6; sign bit when s < 0): the comparisons the Triton epilogue makes."""
    import torch
    a = torch.clamp(s.abs(), max=6.0)
    code = sum(((a > t) if strict else (a >= t)).to(torch.int32)
               for t, strict in ((0.25, 1), (0.75, 0), (1.25, 1), (1.75, 0),
                                 (2.5, 1), (3.5, 0), (5.0, 1)))
    return code | torch.where(s < 0, 8, 0).to(torch.int32)


def sf_offset(row, kb, kb_pad: int):
    """Byte offset of (row, k-block) in the 128x4 swizzled scale buffer
    (== nvfp4_gemm.sf_offset / the oxide kernel / vLLM swizzle_blockscale)."""
    atom = (row // 128) * (kb_pad // 4) + kb // 4
    return atom * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


def round_up(v: int, a: int) -> int:
    return -(-v // a) * a


# ---------------------------------------------------------------------------
# CPU twins: per-program structure of the Triton kernels below (partition,
# block validity, store addresses, padded-row zeroing) + torch references of
# the stock glue ops (CPU tests only; the pod oracle uses the real vLLM ops)
# ---------------------------------------------------------------------------
def twin_quant_store(q, sf, y, row: int, m: int, col0: int, valid_len: int, g: float,
                     k: int):
    """_quant_store of one program: y = fp32 tile (NB*16 elements from row
    element col0; bf16-rounded here, as the kernel does) -> flat q / sf."""
    import torch
    yq = y.to(torch.bfloat16).float().reshape(-1, 16)
    nb = yq.shape[0]
    codes, sfb = spec_quant(yq.reshape(1, -1), g)
    codes, sfb = codes.reshape(nb, 8), sfb.reshape(nb)
    kb_pad, rm = k // 16, round_up(m, 128)
    for b in range(nb):
        if b * 16 >= valid_len:
            continue
        kb = col0 // 16 + b
        q[row * (k // 2) + kb * 8:row * (k // 2) + kb * 8 + 8] = codes[b]
        sf[sf_offset(row, kb, kb_pad)] = sfb[b]
        for p in range(m + row, rm, m):
            sf[sf_offset(p, kb, kb_pad)] = 0


def twin_quantize(y, g: float, part: int, block: int):
    """Programs of width `part` (padded to `block`) over each row of y (the
    fused op's bf16 output) -> (q uint8 [M, K/2], sf uint8 [RM, K/16]).
    Buffers start as garbage: every valid byte must be written."""
    import torch
    m, k = y.shape
    q = torch.full((m * k // 2,), 0xAB, dtype=torch.uint8)
    sf = torch.full((round_up(m, 128) * k // 16,), 0xAB, dtype=torch.uint8)
    for row in range(m):
        for c0 in range(0, k, part):
            tile = torch.zeros(block)
            w = min(part, k - c0)
            tile[:w] = y[row, c0:c0 + w].float()
            twin_quant_store(q, sf, tile, row, m, c0, w, g, k)
    return q.view(m, k // 2), sf.view(-1, k // 16)


def ref_combine_norm(res, blk, inj, w, eps, hc):
    import torch
    m, d = res.shape
    hd = d // hc
    b = blk.float()[:, None, :]
    if inj is not None:
        b = b * (2.0 * torch.sigmoid(inj.float() / hc))[:, :, None]
    out = (res.float().view(m, hc, hd) + b).to(res.dtype)
    o = out.float()
    y = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
    wf = w.float().view(-1, hd) if w.numel() == d else w.float()[None]
    y = y + y * wf
    return out.view(m, d), y.view(m, d).to(res.dtype)


def ref_grouped_norm(x, w, eps, groups):
    import torch
    m, d = x.shape
    gd = d // groups
    xf = x.float().view(m, groups, gd)
    y = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)
    wf = w.float().view(-1, gd) if w.numel() == d else w.float()[None]
    return (y + y * wf).view(m, d).to(x.dtype)


def ref_silu(x, hc):
    import torch
    z = x.float() / hc
    return (z * torch.sigmoid(z)).to(x.dtype)


def ref_gate_mix(x, gate, hc):
    import torch
    m, d = x.shape
    s = torch.sigmoid(gate.float().view(m, hc, d // hc)) * x.float().view(m, hc, d // hc)
    return (s.sum(1) / hc).to(x.dtype)


def twin_combine_norm_q(res, blk, inj, w, eps, hc, g):
    out, y = ref_combine_norm(res, blk, inj, w, eps, hc)
    hd = res.shape[1] // hc
    return (out, y, *twin_quantize(y, g, hd, _np2(-(-hd // 512)) * 512))


def twin_grouped_norm_q(x, w, eps, groups, g):
    y = ref_grouped_norm(x, w, eps, groups)
    gd = x.shape[1] // groups
    return (y, *twin_quantize(y, g, gd, _np2(gd)))


def twin_silu_q(x, hc, g):
    return twin_quantize(ref_silu(x, hc), g, x.shape[1], _np2(x.shape[1]))


def twin_gate_mix_q(x, gate, hc, gs):
    y = ref_gate_mix(x, gate, hc)
    return y, [twin_quantize(y, g, 512, 512) for g in gs]


def _np2(v: int) -> int:
    return 1 << (v - 1).bit_length()


# ---------------------------------------------------------------------------
# Triton kernels (pod only; lazily built so the module imports on CPU CI)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _kernels():
    from suffix_hybrid.kernels import hc_fused_quant_triton as k

    return types.SimpleNamespace(grouped=k._grouped_gemma_rmsnorm_q_kernel,
                                 silu=k._hc_silu_q_kernel, gate_mix=k._hc_gate_mix_q_kernel,
                                 combine_norm=k._hc_combine_norm_q_kernel)


@functools.lru_cache(maxsize=1)
def _pdl() -> bool:
    from vllm.platforms import current_platform
    return bool(current_platform.is_arch_support_pdl())


def pad_pr(m: int) -> int:
    """PR constexpr: padded scale rows [M, round_up(M, 128)) are zeroed by
    row r's programs as rows M + r + i*M, i < PR (pow2 >= ceil((RM - M) / M))."""
    return _np2(max(1, -(-(round_up(m, 128) - m) // m)))


def _qbufs(m: int, k: int, dev):
    """(q uint8 [M, K/2], sf uint8 [round_up(M, 128), K/16]) — the layout of
    vLLM scaled_fp4_quant(is_sf_swizzled_layout=True) (K/16 % 4 == 0)."""
    import torch
    return (torch.empty(m, k // 2, dtype=torch.uint8, device=dev),
            torch.empty(round_up(m, 128), k // 16, dtype=torch.uint8, device=dev))


# ---------------------------------------------------------------------------
# host ops (launch shapes = the stock wrappers'); gs = consumer global scales
# ---------------------------------------------------------------------------
def combine_norm_q(residual, block_output, injection, norm_weight, eps, hc, g):
    """hc_combine_norm + NVFP4 of y -> (out, y, q, sf)."""
    n, dim = residual.shape
    hd = dim // hc
    assert residual.stride(1) == 1 and block_output.stride(1) == 1
    out, y = residual.new_empty(residual.shape), residual.new_empty(residual.shape)
    q, sf = _qbufs(n, dim, residual.device)
    _kernels().combine_norm[(n * hc,)](
        block_output, residual, injection, norm_weight, out, y, q, sf,
        block_output.stride(0), residual.stride(0),
        injection.stride(0) if injection is not None else 0, out.stride(0), y.stride(0),
        n, sf.shape[0], g, g6_of(g), hd, hc, W_SHARED=norm_weight.numel() == hd, EPS=eps,
        BLOCK_SIZE=512, KH=dim // 2, KB_PAD=dim // 16, PR=pad_pr(n), launch_pdl=_pdl())
    return out, y, q, sf


def grouped_norm_q(x, weight, eps, groups, g):
    """grouped_gemma_rmsnorm + NVFP4 of y -> (y, q, sf)."""
    n, dim = x.shape
    assert x.stride(1) == 1
    y = x.new_empty(x.shape)
    q, sf = _qbufs(n, dim, x.device)
    _kernels().grouped[(n * groups,)](
        x, weight, y, q, sf, x.stride(0), y.stride(0), n, sf.shape[0], g, g6_of(g),
        dim, groups, W_SHARED=weight.numel() == dim // groups, EPS=eps, KH=dim // 2,
        KB_PAD=dim // 16, PR=pad_pr(n), launch_pdl=_pdl())
    return y, q, sf


def silu_q(x, hc, g):
    """hc_silu -> NVFP4 only (q, sf); x may be a strided view (lora split)."""
    import torch
    n, dim = x.shape
    assert x.stride(1) == 1 and x.dtype == torch.bfloat16  # kernel rounds to bf16
    q, sf = _qbufs(n, dim, x.device)
    _kernels().silu[(n,)](x, q, sf, x.stride(0), n, sf.shape[0], g, g6_of(g), DIM=dim,
                          HC=hc, KH=dim // 2, KB_PAD=dim // 16, PR=pad_pr(n),
                          launch_pdl=_pdl())
    return q, sf


def gate_mix_q(x, gate, hc, gs):
    """hc_gate_mix + NVFP4 per distinct scale in gs (1..QMAX) -> (y, [(q, sf)])."""
    n, dim = gate.shape
    hd = dim // hc
    assert x.shape == gate.shape and x.stride(1) == 1 and gate.stride(1) == 1
    assert 1 <= len(gs) <= QMAX
    y = x.new_empty(n, hd)
    bufs = [_qbufs(n, hd, x.device) for _ in gs]
    (q, sf), (q2, sf2) = bufs[0], bufs[-1]
    g2 = gs[-1]
    _kernels().gate_mix[(n, -(-hd // 512))](
        x, gate, y, q, sf, q2, sf2, x.stride(0), gate.stride(0), y.stride(0), n,
        sf.shape[0], gs[0], g6_of(gs[0]), g2, g6_of(g2), dim, hc, 512, NQ=len(gs),
        KH=hd // 2, KB_PAD=hd // 16, PR=pad_pr(n), launch_pdl=_pdl())
    return y, bufs


# ---------------------------------------------------------------------------
# wiring: pairs per GatedResidual, eligibility (pure; tests use fakes)
# ---------------------------------------------------------------------------
def input_key(lin):
    """vLLM get_input_quant_key (fusion/quant_activation.py) without importing it."""
    if getattr(lin, "requires_unquantized_input", False):
        return None
    return getattr(lin, "_input_quant_key", None)


def consumer_reason(lin, k: int, key, replicated: bool = False) -> str | None:
    """Why `lin` cannot take a fused NVFP4 input of width k, or None."""
    if lin is None:
        return "absent"
    if key is None or input_key(lin) != key:
        return f"input quant key {input_key(lin)} is not NVFP4 kNvfp4Dynamic"
    kin = getattr(lin, "input_size_per_partition", None)
    if kin != k:
        return f"K={kin} != producer width {k}"
    if k % 64:
        return f"K={k} % 64 (scale columns would need padding)"
    w = getattr(lin, "weight", None)
    if w is None or w.shape[-1] * 2 != k:
        return "K-padded NVFP4 weight"
    gs = getattr(lin, "input_global_scale_inv", None)
    if gs is None or gs.numel() != 1:
        return "no scalar input_global_scale_inv"
    if replicated and (getattr(lin, "bias", None) is not None
                       or (getattr(lin, "gather_output", False)
                           and getattr(lin, "tp_size", 1) > 1)):
        return "bias or gathered output (called through quant_method.apply directly)"
    return None


def producer_reason(hc) -> str | None:
    for a in ("hc_norm", "hc_count", "hidden_size", "lora_rank", "use_combine",
              "input_mix_weight_up", "config"):
        if not hasattr(hc, a):
            return f"not a vLLM 0.30.0 GatedResidual (no {a})"
    if hc.hidden_size % 64 or hc.lora_rank % 64:
        return f"hidden {hc.hidden_size} / lora rank {hc.lora_rank} not % 64"
    if hc.hc_norm.weight.numel() not in (hc.hidden_size, hc.hidden_size * hc.hc_count):
        return "norm weight size"
    return None


def _gscale(lin) -> float:
    return float(lin.input_global_scale_inv.reshape(-1)[0].item())


def plan_model(model, key) -> tuple[list, list]:
    """-> (plans, skipped). plan = SimpleNamespace(name, hc, down, g_down, up,
    g_up, groups=[(g, [(name, lin)])], reasons=[...]) for every GatedResidual
    with >= 1 fusable pair; skipped = ["name: reason"] per rejected pair."""
    mods = dict(model.named_modules())
    plans, skipped = [], []
    for name, hc in mods.items():
        if type(hc).__name__ != "GatedResidual":
            continue
        why = producer_reason(hc)
        if why is not None:
            skipped.append(f"{name}: {why}")
            continue
        d = hc.hidden_size * hc.hc_count
        down = getattr(hc, "input_mix_weight_down_block_inject" if hc.use_combine
                       else "input_mix_weight_down", None)
        p = types.SimpleNamespace(name=name, hc=hc, down=None, g_down=None, up=None,
                                  g_up=None, groups=[])
        why = consumer_reason(down, d, key, replicated=True)
        if why is None:
            p.down, p.g_down = down, _gscale(down)
        else:
            skipped.append(f"{name} A norm->down: {why}")
        why = consumer_reason(hc.input_mix_weight_up, hc.lora_rank, key, replicated=True)
        if why is None:
            p.up, p.g_up = hc.input_mix_weight_up, _gscale(hc.input_mix_weight_up)
        else:
            skipped.append(f"{name} B silu->up: {why}")
        parent, _, attr = name.rpartition(".")
        groups: dict = {}
        for rel in BLOCK_CONSUMERS.get(attr, ()):
            cname = f"{parent}.{rel}" if parent else rel
            lin = mods.get(cname)
            if lin is None:
                continue
            why = consumer_reason(lin, hc.hidden_size, key)
            if why is None and _gscale(lin) not in groups and len(groups) >= QMAX:
                why = f"more than {QMAX} distinct input global scales at this boundary"
            if why is not None:
                skipped.append(f"{cname} C gate_mix->block: {why}")
                continue
            groups.setdefault(_gscale(lin), []).append((cname, lin))
        p.groups = list(groups.items())
        if p.down is not None or p.up is not None or p.groups:
            plans.append(p)
    return plans, skipped


def _qa(q, sf, shape):
    import torch
    return _V.QA(data=q, scale=sf.view(torch.float8_e4m3fn), orig_dtype=torch.bfloat16,
                 orig_shape=torch.Size(shape), quant_key=_V.KEY)


def _apply_q(lin, q, sf, shape):
    return lin.quant_method.apply(lin, _qa(q, sf, shape), None)


def _pending_apply(orig, layer, x, bias=None):
    """Block-consumer quant_method.apply: swap in the pending fused input when
    handed the producer's bf16 buffer (views allowed); consumed once."""
    p = layer.__dict__.pop("_sfx_hcq_pending", None)
    if p is not None:
        y, q, sf, stream = p
        if (hasattr(x, "data_ptr") and x.data_ptr() == y.data_ptr() and x.numel() == y.numel()
                and x.shape[-1] == y.shape[-1] and x.dtype == y.dtype and x.is_contiguous()):
            x = _qa(q, sf, x.shape)
            if stream is not None:
                # vLLM runs MoE shared experts on an aux stream: keep the
                # caching allocator from recycling q / sf under that GEMM
                import torch
                cur = torch.cuda.current_stream(q.device)
                if cur != stream:
                    q.record_stream(cur)
                    sf.record_stream(cur)
        elif id(layer) not in _state["missed"]:
            _state["missed"].add(id(layer))
            _log(f"MISS: {getattr(layer, 'prefix', type(layer).__name__)} was called with a "
                 "different tensor than its HC gate-mix output (fused quant wasted, ran unfused)")
    return orig(layer, x, bias)


def _tail(self, xn, down_out):
    """Shared by mix / combine_and_mix after the norm: split, silu, up, gate mix."""
    st, ops, hc = self._sfx_hcq, _V.ops, self.hc_count
    if self.use_combine:
        lora, injection, _ = down_out.split([self.lora_rank, hc, self.pad_size], dim=-1)
    else:
        lora, injection = down_out, None
    if st.up is not None:
        q, sf = ops.silu_q(lora, hc, st.g_up)
        gate = _apply_q(st.up, q, sf, lora.shape)
    else:
        gate = self.input_mix_weight_up(ops.hc_silu(lora, hc))
    if st.groups:
        block_input, bufs = ops.gate_mix_q(xn, gate, hc, [g for g, _ in st.groups])
        stream = None
        if block_input.is_cuda:
            import torch
            stream = torch.cuda.current_stream(block_input.device)
        for (_g, lins), (q, sf) in zip(st.groups, bufs):
            for _n, lin in lins:
                lin._sfx_hcq_pending = (block_input, q, sf, stream)
    else:
        block_input = ops.hc_gate_mix(xn, gate, hc)
    return block_input, injection


def _down(self, xn, q, sf):
    st = self._sfx_hcq
    if st.down is not None:
        return _apply_q(st.down, q, sf, xn.shape)
    return (self.input_mix_weight_down_block_inject if self.use_combine
            else self.input_mix_weight_down)(xn)


def _fused_mix(self, hidden_states):
    st, ops = self._sfx_hcq, _V.ops
    eps = self.config.rms_norm_eps
    if hidden_states.shape[0] == 0:
        return st.orig_mix(hidden_states)
    if st.down is not None:
        xn, q, sf = ops.grouped_norm_q(hidden_states, self.hc_norm.weight, eps,
                                       self.hc_count, st.g_down)
    else:
        xn, q, sf = ops.grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight, eps,
                                              self.hc_count), None, None
    return (hidden_states, *_tail(self, xn, _down(self, xn, q, sf)))


def _fused_combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
    st, ops = self._sfx_hcq, _V.ops
    eps = self.config.rms_norm_eps
    if hidden_states.shape[0] == 0:
        return st.orig_cam(hidden_states, prev_block_output, prev_injection)
    if st.down is not None:
        hidden_states, xn, q, sf = ops.combine_norm_q(
            hidden_states, prev_block_output, prev_injection, self.hc_norm.weight, eps,
            self.hc_count, st.g_down)
    else:
        (hidden_states, xn), q, sf = ops.hc_combine_norm(
            hidden_states, prev_block_output, prev_injection, self.hc_norm.weight, eps,
            self.hc_count), None, None
    return (hidden_states, *_tail(self, xn, _down(self, xn, q, sf)))


def install(plan) -> None:
    """Switch one GatedResidual (and its C consumers) to the fused path."""
    hc = plan.hc
    hc._sfx_hcq = types.SimpleNamespace(down=plan.down, g_down=plan.g_down, up=plan.up,
                                        g_up=plan.g_up, groups=plan.groups,
                                        orig_mix=hc.mix, orig_cam=hc.combine_and_mix)
    hc.mix = types.MethodType(_fused_mix, hc)
    hc.combine_and_mix = types.MethodType(_fused_combine_and_mix, hc)
    for _g, lins in plan.groups:
        for _n, lin in lins:
            qm = lin.quant_method
            if not isinstance(qm.__dict__.get("apply"), functools.partial):
                qm.apply = functools.partial(_pending_apply, qm.apply)


def pair_counts(plans) -> dict:
    return {"A norm->down": sum(p.down is not None for p in plans),
            "B silu->up": sum(p.up is not None for p in plans),
            "C gate_mix->block": sum(len(lins) for p in plans for _g, lins in p.groups),
            "C gate_mix launches": sum(bool(p.groups) for p in plans)}


# ---------------------------------------------------------------------------
# load-time oracle (real weights, pod) + wire()
# ---------------------------------------------------------------------------
def _bits_equal(a, b) -> bool:
    import torch
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                              b.contiguous().view(torch.uint8))


def check_quant(y, q, sf, g) -> str | None:
    """Fused (q, sf) == spec_quant(stock bf16 y) bits, padded scale rows 0."""
    import torch
    from suffix_hybrid.kernels.nvfp4_gemm import unswizzle_sf
    m, k = y.shape
    q_ref, sf_ref = spec_quant(y, g)
    sf_u = unswizzle_sf(sf, sf.shape[0], k // 16)
    if not torch.equal(q, q_ref):
        return f"codes differ at {int((q != q_ref).sum())}/{q.numel()} bytes"
    if not torch.equal(sf_u[:m], sf_ref):
        return f"block scales differ at {int((sf_u[:m] != sf_ref).sum())}/{sf_ref.numel()}"
    if sf_u[m:].any():
        return "padded scale rows not zero"
    return None


def check_gemm(lin, x_bf16, q, sf) -> tuple[str | None, str]:
    """Consumer GEMM on the fused input vs the unfused bf16 call: bit-exact on
    our decode-GEMM route (same spec quant), else oracle_ok vs f64 exact."""
    import torch
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref, oracle_ok
    m = x_bf16.shape[0]
    fused = lin.quant_method.apply(lin, _qa(q, sf, x_bf16.shape), None).float()
    unfused = lin.quant_method.apply(lin, x_bf16, None).float()
    cfg = getattr(lin, "_sfx_nvfp4", None)
    if not bool(torch.isfinite(fused).all()):
        return "non-finite output", "-"
    if cfg is not None and m <= cfg["max_m"]:
        ok = torch.equal(fused, unfused)  # values: -0 == +0 (see the contract)
        return (None if ok else "our route: fused != unfused output"), "ours exact"
    n = lin.output_size_per_partition
    exact = exact_ref(q, sf.view(torch.float8_e4m3fn), lin.weight, lin.weight_scale,
                      float(lin.alpha.reshape(-1)[0]), n).bfloat16().float()
    en = exact.norm().clamp_min(1e-30)
    rx, ux = float((fused - exact).norm() / en), float((unfused - exact).norm() / en)
    ru = float((fused - unfused).norm() / unfused.norm().clamp_min(1e-30))
    msg = f"flashinfer rel_vs_exact={rx:.2e} unfused_vs_exact={ux:.2e} rel_vs_unfused={ru:.2e}"
    return (None if oracle_ok(rx, ru, ux) else msg), msg


def _amp(g: float, div: float = 4.0) -> float:
    """Activation amplitude at the design point amax ~ A/div, A = 2688 / g."""
    return 448.0 * 6.0 / g / div


def oracle_plan(p, ms=ORACLE_MS) -> list[str]:
    """Load-time oracle of one GatedResidual's fused pairs; raises on mismatch."""
    import torch
    ops, hc = _V.ops, p.hc
    nh, h, r = hc.hc_count, hc.hidden_size, hc.lora_rank
    d, eps, w = nh * h, hc.config.rms_norm_eps, hc.hc_norm.weight
    dev = w.device
    gen = torch.Generator(device=dev).manual_seed(len(p.name))
    rnd = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
    lines = []

    def fail(what, m, why):
        raise RuntimeError(f"{MARKER} LOAD ORACLE FAIL {p.name} {what} M={m}: {why} "
                           f"— refusing to serve with {GATE}=1")

    for m in ms:
        if p.down is not None:
            res, blk = rnd(m, d).bfloat16(), rnd(m, h).bfloat16()
            res[0], blk[0] = 0, 0  # an all-zero row: sf 0, codes 0
            inj = rnd(m, nh).bfloat16() if hc.use_combine else None
            for variant in ("combine_norm", "combine_norm(no inj)", "grouped_norm"):
                if variant == "grouped_norm":
                    xn = ops.grouped_gemma_rmsnorm(res, w, eps, nh)
                    y, q, sf = ops.grouped_norm_q(res, w, eps, nh, p.g_down)
                    same = _bits_equal(y, xn)
                else:
                    i = inj if variant == "combine_norm" else None
                    out, xn = ops.hc_combine_norm(res, blk, i, w, eps, nh)
                    o2, y, q, sf = ops.combine_norm_q(res, blk, i, w, eps, nh, p.g_down)
                    same = _bits_equal(o2, out) and _bits_equal(y, xn)
                if not same:
                    fail(f"A {variant}", m, "bf16 outputs != stock vLLM op bits")
                why = check_quant(xn, q, sf, p.g_down)
                if why:
                    fail(f"A {variant}", m, why)
            why, msg = check_gemm(p.down, xn, q, sf)
            if why:
                fail("A down GEMM", m, why)
            lines.append(f"A M={m} {msg}")
        if p.up is not None:
            buf = rnd(m, r + 16) * (nh * _amp(p.g_up) / 3.5)
            lora = buf.bfloat16()[:, :r]  # strided, like the down-output split
            ref = ops.hc_silu(lora, nh)
            q, sf = ops.silu_q(lora, nh, p.g_up)
            why = check_quant(ref, q, sf, p.g_up)
            if why:
                fail("B silu", m, why)
            why, msg = check_gemm(p.up, ref, q, sf)
            if why:
                fail("B up GEMM", m, why)
            lines.append(f"B M={m} {msg}")
        if p.groups:
            gs = [g for g, _ in p.groups]
            x = (rnd(m, d) * (_amp(max(gs)) / 3.5)).bfloat16()
            gate = rnd(m, d).bfloat16()
            ref = ops.hc_gate_mix(x, gate, nh)
            y, bufs = ops.gate_mix_q(x, gate, nh, gs)
            if not _bits_equal(y, ref):
                fail("C gate_mix", m, "bf16 output != stock vLLM op bits")
            for (g, lins), (q, sf) in zip(p.groups, bufs):
                why = check_quant(ref, q, sf, g)
                if why:
                    fail(f"C gate_mix g={g:.4g}", m, why)
                for cname, lin in lins:
                    why, msg = check_gemm(lin, ref, q, sf)
                    if why:
                        fail(f"C {cname} GEMM", m, why)
                    lines.append(f"C {cname.rpartition('.')[2]} M={m} {msg}")
    return lines


def _load_vllm():
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
    from vllm.models.qwen4_exp.nvidia.ops import hc as stock

    _V.QA, _V.KEY = QuantizedActivation, kNvfp4Dynamic
    _V.ops = types.SimpleNamespace(
        hc_combine_norm=stock.hc_combine_norm, grouped_gemma_rmsnorm=stock.grouped_gemma_rmsnorm,
        hc_silu=stock.hc_silu, hc_gate_mix=stock.hc_gate_mix, combine_norm_q=combine_norm_q,
        grouped_norm_q=grouped_norm_q, silu_q=silu_q, gate_mix_q=gate_mix_q)


def wire(model) -> list:
    """SUFFIX_NVFP4_DENSE converter hook: fuse every eligible pair, oracle
    each on its real weights, log counts + per-pair skip reasons."""
    if not gate_on():
        return []
    try:
        _load_vllm()
    except ImportError as exc:  # not a Qwen4Exp-capable vLLM: nothing to fuse
        _log(f"inert: {exc}")
        return []
    import torch
    plans, skipped = plan_model(model, _V.KEY)
    checks = 0
    for p in plans:
        checks += len(oracle_plan(p))
        install(p)
    if plans:
        torch.cuda.synchronize()
    _state["pairs"] = pair_counts(plans)
    _log(f"HC-FUSED-QUANT ACTIVE on {len(plans)} GatedResidual modules: "
         + ", ".join(f"{k} x{v}" for k, v in _state["pairs"].items())
         + f"; LOAD ORACLE PASS ({checks} GEMM checks, M in {ORACLE_MS}: fused bf16 "
         "== stock op bits, q/sf == spec quant bits, consumer GEMM == unfused)"
         + (f"; unfused pairs: {skipped}" if skipped else ""))
    return plans


# ---------------------------------------------------------------------------
# standalone oracle + bench (boot gates, SM120 pod, no model)
# ---------------------------------------------------------------------------
H, HC, R = 2560, 4, 320  # qwen3.8-flash-next hidden / hc_count / lora rank
# consumer (N, K) per producer at TP2 (nvfp4_gemm.QWEN_FLASH_SHAPES)
SHAPES = {"A": (336, H * HC), "B": (H * HC, R), "C": (8192, H), "C2": (48, H)}
G_SAT = 2688 / 0.5  # static scale far below the data: saturation path


def _fi_gemm(xq, xsf, w):
    import torch
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
    return flashinfer_scaled_fp4_mm(xq, w["w_fi"], xsf, w["w_sf_2d"], w["alpha"],
                                    torch.bfloat16, backend="cutlass")[:, :w["n"]]


def _weight(n, k, g, dev):
    """Synthetic NVFP4 weight in vLLM layout, alpha for input global g."""
    import torch
    from suffix_hybrid.kernels import nvfp4_gemm as ng
    p = ng._weights(n, k)
    wt = torch.from_numpy(p["w_packed"]).to(dev)
    return dict(n=n, w=wt, w_fi=torch.nn.functional.pad(wt, (0, 0, 0, -n % 32)).contiguous(),
                w_sf=torch.from_numpy(p["w_sf_swz"]).to(dev).view(torch.float8_e4m3fn),
                w_sf_2d=torch.from_numpy(p["w_sf_swz"]).to(dev).view(torch.float8_e4m3fn)
                .view(round_up(n, 128), -1),
                alpha=torch.tensor([1.0 / (np.float32(g) * p["g_w"])], dtype=torch.float32,
                                   device=dev))


def _gemm_vs_unfused(x, q, sf, g, w) -> str:
    import torch
    from vllm import _custom_ops as vops
    from suffix_hybrid.kernels.nvfp4_gemm import exact_ref, oracle_ok
    gt = torch.tensor([g], dtype=torch.float32, device=x.device)
    fused = _fi_gemm(q, sf.view(torch.float8_e4m3fn), w).float()
    xq, xsf = vops.scaled_fp4_quant(x, gt)
    unfused = _fi_gemm(xq, xsf, w).float()
    exact = exact_ref(q, sf.view(torch.float8_e4m3fn), w["w"], w["w_sf"],
                      float(w["alpha"]), w["n"]).bfloat16().float()
    en = exact.norm().clamp_min(1e-30)
    rx, ux = float((fused - exact).norm() / en), float((unfused - exact).norm() / en)
    ru = float((fused - unfused).norm() / unfused.norm().clamp_min(1e-30))
    msg = f"rel_vs_exact={rx:.2e} unfused_vs_exact={ux:.2e} rel_vs_unfused={ru:.2e}"
    if not oracle_ok(rx, ru, ux):
        raise RuntimeError(f"GEMM {msg}")
    return msg


def oracle(ms=BENCH_MS):
    import torch
    _load_vllm()
    ops = _V.ops
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(0)
    rnd = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
    d, eps = H * HC, 1e-6
    w_norm = (rnd(d) * 0.1).bfloat16()
    w_shared = (rnd(H) * 0.1).bfloat16()
    g_a, g_b, g_c, g_c2 = 2688 / 150.0, 2688 / 8.0, 2688 / 40.0, 2688 / 25.0
    wts = {k: _weight(n, kk, g, dev) for (k, (n, kk)), g in
           zip(SHAPES.items(), (g_a, g_b, g_c, g_c2))}
    lines = []

    def need(tag, m, why):
        if why:
            raise RuntimeError(f"{tag} M={m}: {why}")

    for m in ms:
        res, blk, inj = rnd(m, d).bfloat16(), rnd(m, H).bfloat16(), rnd(m, HC).bfloat16()
        res[0], blk[0] = 0, 0  # all-zero row: sf 0, codes 0
        # each op at its design scale and at G_SAT (A = 0.5: block scales clamp
        # at 448, codes saturate at 6)
        for i, wn in ((inj, w_norm), (None, w_norm), (inj, w_shared)):
            out, xn = ops.hc_combine_norm(res, blk, i, wn, eps, HC)
            for g in (G_SAT, g_a):  # design scale last: its q / sf feed the GEMM check
                o2, y, q, sf = ops.combine_norm_q(res, blk, i, wn, eps, HC, g)
                need("A combine_norm", m, None if _bits_equal(o2, out) and _bits_equal(y, xn)
                     else "bf16 outputs != stock")
                need(f"A combine_norm g={g:.4g}", m, check_quant(xn, q, sf, g))
        xn2 = ops.grouped_gemma_rmsnorm(res, w_norm, eps, HC)
        for g in (G_SAT, g_a):
            y, q2, sf2 = ops.grouped_norm_q(res, w_norm, eps, HC, g)
            need("A grouped_norm", m, None if _bits_equal(y, xn2) else "bf16 output != stock")
            need(f"A grouped_norm g={g:.4g}", m, check_quant(xn2, q2, sf2, g))
        lines.append(f"A M={m}: {_gemm_vs_unfused(xn, q, sf, g_a, wts['A'])}")
        lora = (rnd(m, R + 16) * (HC * _amp(g_b) / 3.5)).bfloat16()[:, :R]
        lora[0] = 0
        ref = ops.hc_silu(lora, HC)
        for g in (G_SAT, g_b):
            q, sf = ops.silu_q(lora, HC, g)
            need(f"B silu g={g:.4g}", m, check_quant(ref, q, sf, g))
        lines.append(f"B M={m}: {_gemm_vs_unfused(ref, q, sf, g_b, wts['B'])}")
        x = (rnd(m, d) * (_amp(g_c) / 3.5)).bfloat16()
        x[0] = 0
        gate = rnd(m, d).bfloat16()
        ref = ops.hc_gate_mix(x, gate, HC)
        for gs in ([G_SAT], [g_c, g_c2]):
            y, bufs = ops.gate_mix_q(x, gate, HC, gs)
            need("C gate_mix", m, None if _bits_equal(y, ref) else "bf16 output != stock")
            for g, (q, sf) in zip(gs, bufs):
                need(f"C gate_mix NQ={len(gs)} g={g:.4g}", m, check_quant(ref, q, sf, g))
        lines.append(f"C M={m}: {_gemm_vs_unfused(ref, *bufs[0], g_c, wts['C'])}; "
                     f"C2: {_gemm_vs_unfused(ref, *bufs[1], g_c2, wts['C2'])}")
    for ln in lines:
        _log(ln)
    return (f"{MARKER} HC-FUSED-QUANT ORACLE PASS ({len(ms)} M x 4 fused ops: bf16 == stock "
            "bits, q/sf == spec bits incl. zero / saturated rows + zero padded scale rows; "
            "FlashInfer GEMM on fused input within oracle_ok of the vLLM-quant path)")


def bench(ms=BENCH_MS, iters=200):
    """us per HC boundary (glue + activation quant; the GEMMs are the same on
    both paths), CUDA graphs: unfused = stock op + vLLM scaled_fp4_quant per
    NVFP4 consumer, fused = the fused kernels. Boundaries: GDN attn HC
    (C -> in_proj_qkvz + in_proj_ba, 2 scales) and MoE mlp HC (C -> shared
    gate_up, 1 scale)."""
    import torch
    from vllm import _custom_ops as vops
    from suffix_hybrid.kernels.nvfp4_gemm import _graph_us
    _load_vllm()
    ops = _V.ops
    dev = torch.device("cuda", torch.cuda.current_device())
    d, eps = H * HC, 1e-6
    w = (torch.randn(d, device=dev) * 0.1).bfloat16()
    gt = torch.tensor([30.0], device=dev)
    for m in ms:
        res, blk = torch.randn(m, d, device=dev).bfloat16(), torch.randn(m, H, device=dev).bfloat16()
        inj = torch.randn(m, HC, device=dev).bfloat16()
        down = torch.randn(m, R + 16, device=dev).bfloat16()
        gate = torch.randn(m, d, device=dev).bfloat16()
        for label, nq in (("gdn-attn", 2), ("moe-mlp", 1)):
            def unfused(s, nq=nq):
                _out, xn = ops.hc_combine_norm(res, blk, inj, w, eps, HC)
                vops.scaled_fp4_quant(xn, gt)
                vops.scaled_fp4_quant(ops.hc_silu(down[:, :R], HC), gt)
                bi = ops.hc_gate_mix(xn, gate, HC)
                for _ in range(nq):
                    vops.scaled_fp4_quant(bi, gt)

            def fused(s, nq=nq):
                _o, xn, _q, _sf = ops.combine_norm_q(res, blk, inj, w, eps, HC, 30.0)
                ops.silu_q(down[:, :R], HC, 30.0)
                ops.gate_mix_q(xn, gate, HC, [30.0, 31.0][:nq])

            tu, tf = _graph_us(unfused, dev, iters), _graph_us(fused, dev, iters)
            _log(f"bench {label} boundary M={m}: unfused {tu:.1f} us ({5 + nq} launches), "
                 f"fused {tf:.1f} us (3 launches), delta {tf - tu:+.1f} us")


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "both"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:
        _log(f"HC-FUSED-QUANT {mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
