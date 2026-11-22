# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the DeepSeek-V4.1 fusion kernels (dsv41_hc / dsv41_router /
dsv41_swiglu / dsv41_engram).

* ``sinkhorn`` / ``mxfp8_group_quant`` / ``mxfp8_scale_offsets``: pure-torch helpers the
  Helion bodies call (Helion traces plain module-level helper functions into the
  kernel; they only use ops the bodies already use, no ``hl`` calls).
* ``mxfp8_*`` host side: the F8_128x4 swizzled-scale layout and its exact CPU
  reference, transcribed from vLLM's own producer
  ``vllm/models/deepseek_v41/common/ops/query_quant.py`` @ 39983a5e7
  (``_q_kv_norm_quant_kernel``: "match FlashInfer's UE8M0 conversion, including
  zero/subnormal scales").
* ``capture_replay_gate``: the boot gate every fused kernel ships
  (``<kernel>_capture_replay``): capture on a side stream, replay with changing
  inputs, bit-exact vs eager, runnable under compute-sanitizer.
"""
import os
import sys
import time

import torch

MARK = "[suffix dsv41-fuse]"
# lab / boot-gate device (a CPU stand-in run of the CLI sets SUFFIX_DSV41_FUSE_LAB_DEVICE=cpu)
DEVICE = os.environ.get("SUFFIX_DSV41_FUSE_LAB_DEVICE", "cuda")
MXFP8_BLOCK = 32
FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# device helpers (traced inside Helion bodies: torch ops only)
# ---------------------------------------------------------------------------
def sinkhorn(cm, sk_eps: float, repeat: int):
    """mHC Sinkhorn on [bt, 4, 4] logits (row softmax + eps, column normalise, then
    ``repeat - 1`` x (row, column) normalisations with eps in every denominator).
    Operation order = TileLang ``mhc_pre_big_fuse_with_norm_tilelang``."""
    e = torch.exp(cm - torch.amax(cm, dim=2, keepdim=True))
    cm = e / torch.sum(e, dim=2, keepdim=True) + sk_eps
    cm = cm / (torch.sum(cm, dim=1, keepdim=True) + sk_eps)
    for _ in range(repeat - 1):
        cm = cm / (torch.sum(cm, dim=2, keepdim=True) + sk_eps)
        cm = cm / (torch.sum(cm, dim=1, keepdim=True) + sk_eps)
    return cm


def mxfp8_group_quant(y):
    """y: f32 [..., 32] values that are already rounded to bf16. -> (q e4m3 [..., 32],
    sf int32 [...]) = FlashInfer's MXFP8 quantiser (UE8M0 scale per 32 values).
    sf is the biased exponent byte (0 = zero block, <= 254)."""
    amax = torch.amax(torch.abs(y), dim=-1)
    normalized = amax * (1.0 / 448.0)
    bits = normalized.view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = ((mantissa != 0) & ((exponent != 0) | (mantissa > 0x400000))).to(torch.int32)
    sf = torch.clamp(exponent + bump, max=254)
    sf = torch.where(normalized <= 0, 0, sf)
    inv = torch.where(sf == 0, 0, (254 - sf) << 23).view(torch.float32)
    q = (y * inv[..., None]).to(torch.float8_e4m3fn)
    return q, sf


def mxfp8_scale_offsets(rows, groups, n_groups: int):
    """F8_128x4 swizzle: byte offset of the scale of (row, 32-column group); ``rows``
    [r] and ``groups`` [g] int tensors -> offsets [r, g]. n_groups = hidden // 32.
    (unsqueeze, not [:, None]: Inductor cannot lower a slice of a helper's graph input.)"""
    r128 = (rows // 128).unsqueeze(1)
    r32 = (rows % 32).unsqueeze(1)
    rb = ((rows % 128) // 32).unsqueeze(1)
    gq = (groups // 4).unsqueeze(0)
    gr = (groups % 4).unsqueeze(0)
    return (r128 * ((n_groups + 3) // 4) + gq) * 512 + r32 * 16 + rb * 4 + gr


# ---------------------------------------------------------------------------
# host side: layout + exact CPU references
# ---------------------------------------------------------------------------
def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def mxfp8_scale_bytes(tokens: int, width: int) -> int:
    """Bytes of F8_128x4 scales for ``tokens`` rows of ``width`` MXFP8 values
    (= query_quant.mxfp8_scale_bytes)."""
    return cdiv(tokens, 128) * 128 * cdiv(width // MXFP8_BLOCK, 4) * 4


def mxfp8_quant_reference(y: torch.Tensor):
    """Exact CPU/GPU reference of the stock producer on bf16 rows y [T, K] (K % 32 == 0):
    -> (q e4m3 [T, K], scales uint8 [mxfp8_scale_bytes(T, K)] in F8_128x4 order with the
    padding rows / groups zero). Independent transcription (integer maths on the f32
    bits, scalar loops for the layout) so a bug in the traced helper cannot hide."""
    t, k = y.shape
    g = k // MXFP8_BLOCK
    yf = y.to(torch.bfloat16).to(torch.float32).reshape(t, g, MXFP8_BLOCK)
    amax = yf.abs().amax(-1)
    normalized = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = normalized.view(torch.int32).to(torch.int64)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = ((mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))).to(torch.int64)
    sf = torch.minimum(exponent + bump, torch.tensor(254))
    sf = torch.where(normalized <= 0, torch.zeros_like(sf), sf)
    inv_bits = torch.where(sf == 0, torch.zeros_like(sf), (254 - sf) << 23).to(torch.int32)
    inv = inv_bits.view(torch.float32)
    q = (yf * inv[..., None]).to(torch.float8_e4m3fn).reshape(t, k)
    scales = torch.zeros(mxfp8_scale_bytes(t, k), dtype=torch.uint8, device=y.device)
    groups4 = cdiv(g, 4)
    rows = torch.arange(t, device=y.device)
    for gi in range(g):
        off = ((rows // 128) * groups4 + gi // 4) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 \
            + gi % 4
        scales[off] = sf[:, gi].to(torch.uint8)
    return q, scales


def mxfp8_dequant(q: torch.Tensor, scales: torch.Tensor):
    """q [T, K] e4m3 + swizzled scale bytes -> f32 [T, K] (value * 2^(sf - 127))."""
    t, k = q.shape
    g = k // MXFP8_BLOCK
    rows = torch.arange(t, device=q.device)
    groups4 = cdiv(g, 4)
    sf = torch.stack([scales[((rows // 128) * groups4 + gi // 4) * 512 + (rows % 32) * 16
                             + ((rows % 128) // 32) * 4 + gi % 4] for gi in range(g)], 1)
    scale = (sf.to(torch.int32) << 23).view(torch.float32)
    return (q.to(torch.float32).reshape(t, g, MXFP8_BLOCK) * scale[..., None]).reshape(t, k)


# ---------------------------------------------------------------------------
# boot gate: <kernel>_capture_replay
# ---------------------------------------------------------------------------
def _bits_equal(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(width), b.contiguous().view(width)))


def _flat(out):
    if isinstance(out, torch.Tensor):
        return [out]
    return [t for t in out if isinstance(t, torch.Tensor)]


class _CudaGraphRunner:
    """The real thing: capture on a side stream, replay in place."""

    def __init__(self):
        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.Stream()

    def capture(self, fn):
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            fn()  # one more eager pass on the side stream (compile / allocator warm)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=self.stream, capture_error_mode="thread_local"):
            return fn()

    def replay(self):
        self.graph.replay()
        torch.cuda.synchronize()


def capture_replay_gate(name: str, run, make_inputs, variants: int = 4, replays: int = 2,
                        runner_factory=None, log=None) -> bool:
    """Boot gate ``<name>_capture_replay``.

    ``make_inputs(i)`` -> list of tensors (variant i; shapes identical across variants, the
    values differ). ``run(*static_inputs)`` -> tensor | tuple of tensors, only allowed to
    allocate through torch's allocator and to launch on the current stream.
      1. variant 0 goes into STATIC buffers; the kernel is warmed eagerly (compiles);
      2. every variant's eager output (same static buffers, so the same addresses) is kept;
      3. ``run`` is captured on a side stream into a CUDA graph (static buffers);
      4. each variant is copied into the static buffers, the graph replayed ``replays``
         times (the second replay catches state left behind by the first), and every
         output tensor compared BIT-exactly with the eager one; the variants are visited
         forward and then in reverse so a stale-buffer bug cannot match by accident.
    Returns ok; prints one PASS/FAIL line (exit-status material for the CLI)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    t0 = time.time()
    data = [make_inputs(i) for i in range(variants)]
    static = [t.clone() for t in data[0]]
    run(*static)  # eager warm: compile + allocator
    eager = []
    for v in data:
        for s, t in zip(static, v):
            s.copy_(t)
        eager.append([o.clone() for o in _flat(run(*static))])
    for s, t in zip(static, data[0]):
        s.copy_(t)
    runner = (runner_factory or _CudaGraphRunner)()
    out = runner.capture(lambda: run(*static))
    out = _flat(out)
    problems = []
    order = list(range(variants)) + list(reversed(range(variants)))
    for i in order:
        for s, t in zip(static, data[i]):
            s.copy_(t)
        for rep in range(replays):
            runner.replay()
            for k, (got, want) in enumerate(zip(out, eager[i])):
                if not _bits_equal(got, want):
                    problems.append(f"variant {i} replay {rep} output {k}: not bit-identical to eager")
    ok = not problems
    log(f"{MARK} {name}_capture_replay {'PASS' if ok else 'FAIL'}: {variants} input sets x "
        f"{replays} replays (forward + reverse), {len(out)} outputs bit-exact vs eager, "
        f"{time.time() - t0:.1f}s" + ("" if ok else " | " + "; ".join(problems[:4])))
    return ok


def flashinfer_quant_matches(y: torch.Tensor, q: torch.Tensor, qs: torch.Tensor):
    """SILICON parity: our (q, qs) vs vLLM's own quantiser on the same bf16 rows
    (vllm.model_executor.layers.quantization.utils.mxfp8_utils.mxfp8_e4m3_quantize =
    flashinfer.mxfp8_quantize, cute-dsl, F8_128x4 swizzle). -> (ok, message). Lab image only."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    q_fi, qs_fi = mxfp8_e4m3_quantize(y.contiguous(), is_sf_swizzled_layout=True)
    same_q = _bits_equal(q.cpu(), q_fi.cpu())
    n = min(qs.numel(), qs_fi.numel())
    same_s = qs.numel() == qs_fi.numel() and bool((qs.cpu() == qs_fi.cpu().view(-1)).all())
    return same_q and same_s, (f"flashinfer parity: q {'==' if same_q else '!='}, scales "
                              f"{'==' if same_s else '!='} ({qs.numel()} vs {qs_fi.numel()} bytes, first {n} compared)")
