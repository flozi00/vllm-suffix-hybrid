# SPDX-License-Identifier: Apache-2.0
"""Helion MXFP8 skinny GEMM for DeepSeek-V4.1's FP8 dense linears on SM120
(kernels/mxfp8_linear.py). No oxide twin: the BASELINE is vLLM's own SM120
pick for the layer (nightly 39983a5e7):

  plain linears  init_mxfp8_linear_kernel -> FlashInferCutlassMxfp8LinearKernel
                 (CuTe-DSL is SM100-only; CUTLASS takes >= SM100, N % 32 on
                 SM12x): dynamic MXFP8 activation quant + mm_mxfp8(cutlass).
                 wq_b / indexer.wq_b (K = q_lora_rank) get their activation
                 pre-quantized by the fused q/kv RMSNorm (query_quant.py), so
                 their baseline is mm_mxfp8 alone.
  wo_a (is_bmm)  DeepGemmMxfp8Bmm is SM100-only -> EmulationMxfp8LinearKernel:
                 BF16 dequant AT LOAD (2x the bytes) + torch.bmm per step
                 (deepseek_v4/nvidia/ops/o_proj.py, use_fp8 False).

    y[M, N] = bf16(sum_k x[m, k] * w[n, k] * 2^(ws[n, k / 32] - 127))

Layouts (the checkpoint's, as vLLM loads them: the 32x32 ue8m0 scale blocks
are repeated per row by KMxfp8Static's loader):
  w   float8_e4m3fn [N, K] (passed as [N, K/32, 32]; uint8 bits for the A16 body)
  ws  uint8 e8m0 [N, K/32], row-major, NOT swizzled (vLLM swizzles a copy
      for FlashInfer; the lane keeps this one)
  x   bf16 [M, K] (passed as [M, K/32, 32]; the row stride is free: wo_a's
      groups are strided views)

Bodies (module-level, undecorated: the lab wraps them; serving binds the
shipped config via helion_kernels.compiled):
  mxfp8_gemm_body     A16: bf16 x, weight dequantized in registers (e4m3 bits
                      -> f16 bits * 2^(e - 119) -> bf16, exact), hl.dot with
                      fp32 accumulate. Numerically BETTER than vLLM's W8A8
                      (no activation rounding); the only route for wo_a,
                      whose vLLM path is bf16 x bf16 too.
  mxfp8_gemm_a8_body  A8: the activation quantized per 32-block in the
                      prologue exactly like vLLM's mxfp8 quant (scale =
                      ceil(log2(amax / 448)) + 127, x * 2^-s -> e4m3 RNE),
                      tl.dot_scaled e4m3 x e4m3 with both e8m0 scales: the
                      same numerics as vLLM's path at FP8 MMA rate (the M=192
                      end, where bf16 MMA stops being free).
Both split K by the tunable split_k, TWO-STAGE (capture-safe, the only variant
the serving lane serves): stage 1 is the Helion kernel; a split program stores
its fp32 partial to partial[split, M, N]; stage 2 is a host-side torch reduce
launched after it on the same stream (partial[:n_split].sum(0) -> out), so the
whole composite is ONE Helion body (the lab tunes and times exactly what serves,
reduce included). The partial comes from torch's stream-ordered allocator per
call (graph-private under capture), no cross-CTA semaphore, no counter or any
other state that must be zero at launch, nothing bound at first use. The old
last-arriver variant (acq_rel ticket per (M, N) tile, persistent counters) is
kept as the *_sem bodies behind SUFFIX_MXFP8_GEMM_SEMAPHORE=1 for lab and boot
gate comparison only: it is UNPROVEN under CUDA-graph capture (glm g14: illegal
memory access in the drafter's prefill capture, bisect unfinished) and the
serving lane refuses it (K5).

Serving (knob SUFFIX_HELION_MXFP8_GEMM, kernels/mxfp8_linear.py): per (N, K,
M bucket) the A16 config when shipped, else the A8 one (never for wo_a),
else vLLM's kernel (`auto`) / a load-time RuntimeError (`helion`). The pull
ships only lab wins over vLLM's kernel; where both bodies win, keep the A8
row only if it also beats A16 (pull-time curation: configs carry no timing).
"""
from __future__ import annotations

import os

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no Helion: cases, plans and references still import
    helion = hl = PowerOfTwoFragment = None

ENV = "SUFFIX_HELION_MXFP8_GEMM"
SEM_ENV = "SUFFIX_MXFP8_GEMM_SEMAPHORE"  # 1 = lab / boot-gate only: the unproven last-arriver variant
MARKER = "[suffix mxfp8-gemm helion]"
A16, A8 = "mxfp8_gemm", "mxfp8_gemm_a8"
A16_SEM, A8_SEM = "mxfp8_gemm_sem", "mxfp8_gemm_a8_sem"
BUCKETS = (8, 32, 64, 192)  # decode M per rank: max-num-seqs 32 x DSpark (5+1)
MAX_M = 192
SPLIT_COLS = 4 * 188 * 32  # ~4 CTAs per SM at 32 columns: the split-K cap's scale
Q_LORA = 1280  # K of wq_b / indexer.wq_b: vLLM pre-quantizes their input (query_quant)


# ---------------------------------------------------------------------------
# Helion bodies (literals only inside; tests/test_helion_mxfp8_gemm.py runs
# each against the exact reference in interpret mode and lowers each to Triton
# source with Helion's real front end).
# ---------------------------------------------------------------------------
def mxfp8_gemm_body(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor,
                    partial: torch.Tensor) -> torch.Tensor:
    """x bf16 [M, K/32, 32], w uint8 [N, K/32, 32] (e4m3 bits), ws uint8
    [N, K/32] (e8m0 <= 246), out bf16 [M, N]; partial f32 [cap, M, N] (cap =
    the largest split_k tried; serving allocates the tuned split per call, or
    [1, 1, 1] when it does not split: never read then). Writes and returns out."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))  # 32-blocks per split
    bm = hl.register_block_size(16, 128)
    bn = hl.register_block_size(16, 128)
    bkb = hl.register_block_size(1, 8)  # 32..256 k per step
    n_split = helion.cdiv(nkb, kb_split)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[bm, bn, kb_split]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            xv = x[tile_m, tile_k, :]
            wb = w[tile_n, tile_k, :].to(torch.int16)
            # e4m3 s.eeee.mmm -> f16 bits s.0eeee.mmm0000000 = value * 2^-8
            # (subnormals included); * 2^(e - 127) * 2^8 = f32 bits (e + 8) << 23
            hw = (((wb & 127) << 7) | ((wb & 128) << 8)).view(torch.float16)
            sb = ((ws[tile_n, tile_k].to(torch.int32) + 8) << 23).view(torch.float32)
            wv = (hw.to(torch.float32) * sb[:, :, None]).to(torch.bfloat16)
            acc = hl.dot(xv.reshape(xv.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
    if n_split > 1:
        # stage 2: a separate launch on the same stream (stream order is the only
        # synchronisation), fp32 sum over the splits in a fixed order, one bf16 rounding
        out.copy_(partial[:n_split].sum(0))
    return out


def mxfp8_gemm_a8_body(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor,
                       partial: torch.Tensor) -> torch.Tensor:
    """As mxfp8_gemm_body, but w float8_e4m3fn [N, K/32, 32] and x quantized
    to MXFP8 per program (the spec: `quant_spec`) for an FP8 x FP8 dot_scaled."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))
    bm = hl.register_block_size(16, 128)
    bn = hl.register_block_size(16, 128)
    bkb = hl.register_block_size(1, 8)
    n_split = helion.cdiv(nkb, kb_split)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[bm, bn, kb_split]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            xb = x[tile_m, tile_k, :].to(torch.float32)
            # s = ceil(log2(amax / 448)) + 127 from the f32 bits (exact; a
            # subnormal quotient = an all-zero / denormal block, its codes are 0)
            yb = (torch.amax(torch.abs(xb), dim=-1) / 448.0).view(torch.int32)
            s = torch.clamp((yb >> 23) + ((yb & 8388607) != 0).to(torch.int32), 0, 254)
            inv = ((254 - s) << 23).view(torch.float32)  # 2^(127 - s)
            xq = (xb * inv[:, :, None]).to(torch.float8_e4m3fn)
            wq = w[tile_n, tile_k, :]
            acc = hl.dot_scaled(xq.reshape(xq.size(0), -1), s.to(torch.uint8), "e4m3",
                                wq.reshape(wq.size(0), -1).T, ws[tile_n, tile_k], "e4m3",
                                acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
    if n_split > 1:
        out.copy_(partial[:n_split].sum(0))
    return out


# --- the UNPROVEN-under-capture variant (SUFFIX_MXFP8_GEMM_SEMAPHORE=1: lab and
# boot gate only; the serving lane refuses it). Same device loop and the same
# ConfigSpec as the two-stage bodies (configs transfer); the tail is the
# last-arriver reduce: acq_rel ticket per (M, N) tile on a persistent zeroed
# counter array, the last split sums the partials in order and re-zeroes it.
def mxfp8_gemm_sem_body(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor,
                        partial: torch.Tensor, counters: torch.Tensor) -> torch.Tensor:
    """mxfp8_gemm_body with counters int32 zeros >= counter_elems(N) (left zero)."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))
    bm = hl.register_block_size(16, 128)
    bn = hl.register_block_size(16, 128)
    bkb = hl.register_block_size(1, 8)
    n_split = helion.cdiv(nkb, kb_split)
    n_tiles = helion.cdiv(n, bn)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[bm, bn, kb_split]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            xv = x[tile_m, tile_k, :]
            wb = w[tile_n, tile_k, :].to(torch.int16)
            hw = (((wb & 127) << 7) | ((wb & 128) << 8)).view(torch.float16)
            sb = ((ws[tile_n, tile_k].to(torch.int32) + 8) << 23).view(torch.float32)
            wv = (hw.to(torch.float32) * sb[:, :, None]).to(torch.bfloat16)
            acc = hl.dot(xv.reshape(xv.size(0), -1), wv.reshape(wv.size(0), -1).T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
            tid = tile_m.id * n_tiles + tile_n.id
            old = hl.atomic_add(counters, [tid], 1, sem="acq_rel")
            if old == n_split - 1:
                red = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_r in hl.tile(nkb, block_size=kb_split):
                    red = red + partial[tile_r.id, tile_m, tile_n]
                out[tile_m, tile_n] = red.to(torch.bfloat16)
                counters[tid] = 0
    return out


def mxfp8_gemm_a8_sem_body(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor,
                           partial: torch.Tensor, counters: torch.Tensor) -> torch.Tensor:
    """mxfp8_gemm_a8_body with the last-arriver tail of mxfp8_gemm_sem_body."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    nkb = hl.specialize(w.size(1))
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    kb_split = helion.next_power_of_2(helion.cdiv(nkb, split_k))
    bm = hl.register_block_size(16, 128)
    bn = hl.register_block_size(16, 128)
    bkb = hl.register_block_size(1, 8)
    n_split = helion.cdiv(nkb, kb_split)
    n_tiles = helion.cdiv(n, bn)
    for tile_m, tile_n, tile_s in hl.tile([m, n, nkb], block_size=[bm, bn, kb_split]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bkb):
            xb = x[tile_m, tile_k, :].to(torch.float32)
            yb = (torch.amax(torch.abs(xb), dim=-1) / 448.0).view(torch.int32)
            s = torch.clamp((yb >> 23) + ((yb & 8388607) != 0).to(torch.int32), 0, 254)
            inv = ((254 - s) << 23).view(torch.float32)
            xq = (xb * inv[:, :, None]).to(torch.float8_e4m3fn)
            wq = w[tile_n, tile_k, :]
            acc = hl.dot_scaled(xq.reshape(xq.size(0), -1), s.to(torch.uint8), "e4m3",
                                wq.reshape(wq.size(0), -1).T, ws[tile_n, tile_k], "e4m3",
                                acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
            tid = tile_m.id * n_tiles + tile_n.id
            old = hl.atomic_add(counters, [tid], 1, sem="acq_rel")
            if old == n_split - 1:
                red = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_r in hl.tile(nkb, block_size=kb_split):
                    red = red + partial[tile_r.id, tile_m, tile_n]
                out[tile_m, tile_n] = red.to(torch.bfloat16)
                counters[tid] = 0
    return out


BODIES = {A16: mxfp8_gemm_body, A8: mxfp8_gemm_a8_body}
SEM_BODIES = {A16_SEM: mxfp8_gemm_sem_body, A8_SEM: mxfp8_gemm_a8_sem_body}
ALL_BODIES = {**BODIES, **SEM_BODIES}


def semaphore_on() -> bool:
    """SUFFIX_MXFP8_GEMM_SEMAPHORE (default 0). 1 only ever enables the *_sem bodies in the lab and the
    capture/replay boot gate; the serving lane refuses it (prepare / register)."""
    raw = os.environ.get(SEM_ENV, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"{SEM_ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def refuse_semaphore(where: str) -> None:
    """K5: the serving lane serves the two-stage variant only."""
    if semaphore_on():
        raise RuntimeError(
            f"{MARKER} {where}: {SEM_ENV}=1 — the semaphore split-K variant (per-tile counters, "
            "last-arriver reduce) is UNPROVEN under CUDA-graph capture (glm g14: illegal memory "
            "access in the drafter's prefill capture, bisect unfinished) and the serving lane "
            "never serves it. Unset the knob; use it only in lab_main and the "
            "mxfp8_capture_replay boot gate (--variant semaphore).")


def counter_elems(n: int) -> int:
    """Split-K tickets a body may index: M tiles x N tiles at the smallest
    blocks (16) and M <= MAX_M."""
    return -(-MAX_M // 16) * -(-n // 16)


def max_splits(n: int) -> int:
    """Split-K cap the lab tunes under: ~4 CTAs per SM over (N / 32) x split,
    >= 2, <= 64, a power of two (nvfp4_gemm.max_splits)."""
    want = -(-SPLIT_COLS // n)
    return min(64, max(2, 1 << (want - 1).bit_length()))


MAX_SCALE = 246  # A16 decode: f32 exponent field (ws + 8) must stay <= 254 (255 is inf / NaN)


def validate_weight(w: torch.Tensor, ws: torch.Tensor, what: str = "") -> None:
    """Fail closed (AGENTS.md 1.7) on checkpoint values the bodies do not
    decode: e4m3 NaN codes (S.1111.111) and e8m0 scales above MAX_SCALE (the
    A16 body builds 2^(ws - 119) from f32 bits; 255 is NaN in e8m0). Real
    DeepSeek-V4.1 scales sit around 2^-12..2^-4 (ws ~115..123); one device
    sync per weight, at load."""
    if bool(((w.view(torch.uint8) & 127) == 127).any()):
        raise RuntimeError(f"{MARKER} {what or 'weight'}: e4m3 NaN code in the checkpoint weight "
                           "(S.1111.111) — refusing to serve it on the Helion MXFP8 GEMM")
    top = int(ws.max())
    if top > MAX_SCALE:
        raise RuntimeError(f"{MARKER} {what or 'weight'}: e8m0 scale {top} > {MAX_SCALE} "
                           "(2^-127 + 2^120 overflows the A16 decode) — refusing to serve it on "
                           "the Helion MXFP8 GEMM")


# ---------------------------------------------------------------------------
# DeepSeek-V4.1 shapes (LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4 config.json,
# vLLM deepseek_v41 at 39983a5e7): per TP rank (N, K).
# ---------------------------------------------------------------------------
def dsv41_shapes(tp: int) -> dict:
    inter = -(-2304 // 32 // tp) * 32  # shared expert, padded to whole 32-blocks per rank
    return {
        "wo_a": (1024, 4096),                 # per o_group (bmm); 8 / tp groups per rank
        "fused_wqa_wkv": (1280 + 512, 5120),  # replicated (disable_tp)
        "dspark.main_proj": (5120, 3 * 5120),  # replicated, once per drafter pass
        "wq_b": (64 * 512 // tp, 1280),       # = indexer.wq_b (4096 x 1280, replicated) at TP8
        "wo_b": (5120, 8 * 1024 // tp),
        "shared.gate_up": (2 * inter, 5120),
        "shared.down": (5120, inter),
        "indexer.wq_b": (4096, 1280),
    }


def _cases(model: dict, kernel: str) -> list:
    """TP8 (serving) then TP4, per bucket in decode order of value (8 = 6-token
    DSpark steps, 192 = a full batch), shapes by bytes moved; wo_a (case key
    g = 1: per group, also for TP4's 2 groups) is A16-only."""
    if "deepseek-v4" not in str(model.get("name") or "").lower():
        return []
    out = []
    for tp in (8, 4):
        shapes = dsv41_shapes(tp)
        for m in (8, 192, 32, 64):
            for lin, (n, k) in shapes.items():
                if lin == "wo_a" and kernel == A8:
                    continue
                c = {"k": k, "n": n, "numtokens": m, **({"g": 1} if lin == "wo_a" else {})}
                if c not in out:
                    out.append(c)
    return out[:64]


# ---------------------------------------------------------------------------
# serving: per-M plan at load + the call. Capture-safety contract (docs/dsv41-dense.md):
# no module-level scratch, no counters, no allocation but torch.empty, no host sync.
# ---------------------------------------------------------------------------
def case_of(n: int, k: int, b: int, bmm: bool) -> dict:
    return {"k": k, "n": n, "numtokens": b, **({"g": 1} if bmm else {})}


def _ignore() -> list:
    """Helion warns that the stage-2 torch reduce is not fused into the kernel: intended."""
    return [helion.exc.TensorOperationInWrapper] if helion is not None else []


def _entry(name: str, n: int, k: int, b: int, bmm: bool):
    case = case_of(n, k, b, bmm)
    cfg = hk.config(name, case)
    if cfg is None:
        return None
    return (name, hk.compiled(BODIES[name], name, case, ignore_warnings=_ignore()),
            int(cfg.get("split_k", 1)), b)


def prepare(n: int, k: int, max_m: int, mode: str, bmm: bool = False) -> list:
    """[None] + per M = 1..max_m: (body name, compiled, split_k, bucket) or
    None (vLLM's kernel serves). mode = hk.backend(ENV): "auto" takes Helion
    where a config is shipped; "helion" raises for a bucket without one.
    K5: only the two-stage bodies are ever planned (the semaphore knob refuses)."""
    if mode not in ("auto", "helion"):
        raise RuntimeError(f"{MARKER} prepare: mode {mode!r} (the lane is off)")
    refuse_semaphore("prepare")
    per = {}
    for b in sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)}):
        e = _entry(A16, n, k, b, bmm) or (None if bmm else _entry(A8, n, k, b, bmm))
        if e is None and mode == "helion":
            raise RuntimeError(
                f"{MARKER} {ENV}=helion: no tuned {A16}{'' if bmm else ' / ' + A8} config for "
                f"N={n} K={k}{' (wo_a bmm)' if bmm else ''} M bucket {b} on {hk.platform()} "
                f"(tune it in the autotune lab and ship configs/helion/<kernel>/<platform>.json, "
                f"or set {ENV}=auto)")
        per[b] = e
    return [None] + [per[hk.bucket(m, BUCKETS)] for m in range(1, max_m + 1)]


def gemm(x2d: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor, entry,
         poison_nan: bool = False, counters: torch.Tensor | None = None):
    """out[M, N] = x2d @ deq(w, ws).T via a plan entry. x2d bf16 [M, K]
    (row stride free), w e4m3 [N, K], ws uint8 [N, K/32], out bf16 [M, N]
    (row stride free). The only allocation is the fp32 partial, from torch's
    stream-ordered allocator (graph-private under capture): [split, M, N], or
    a [1, 1, 1] stub when the config does not split (the kernel never touches it).
    poison_nan (checks only): NaN-fill the partial first, so a split a kernel
    forgets to write fails the exact check instead of reading stale bytes.
    counters: the *_sem bodies' zeroed tickets (lab / boot gate only; the plan
    never holds a *_sem entry)."""
    m, k = x2d.shape
    n = w.shape[0]
    name, kern, split, _ = entry
    partial = torch.empty((split, m, n) if split > 1 else (1, 1, 1), dtype=torch.float32,
                          device=x2d.device)
    if poison_nan:
        partial.fill_(float("nan"))
    wv = w.view(torch.uint8) if name in (A16, A16_SEM) else w
    args = (x2d.reshape(m, k // 32, 32), wv.view(n, k // 32, 32), ws, out, partial)
    if name in SEM_BODIES:
        if counters is None:
            raise RuntimeError(f"{MARKER} {name} needs its counters buffer")
        args += (counters,)
    return kern(*args)


def apply(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, entry, n: int, groups: int = 1,
          bmm: bool = False, out: torch.Tensor | None = None, poison_nan: bool = False,
          counters: torch.Tensor | None = None) -> torch.Tensor:
    """The lane's GPU work for one call (the lane, its load check and the capture/replay
    gate all run exactly this). x bf16 [M, K] (bmm: [M, G, K], any strides with a unit last
    stride), w e4m3 [G * N, K], ws uint8 [G * N, K / 32] -> bf16 [M, G * N] (`out` when given:
    row stride free). One launch pair per local group (TP8: 1). poison_nan (checks only):
    NaN-fill out (when allocated here) and every partial before the kernels run."""
    m = x.shape[0]
    if out is None:
        out = torch.empty(m, groups * n, dtype=torch.bfloat16, device=x.device)
        if poison_nan:
            out.fill_(float("nan"))
    if not bmm:
        return gemm(x, w, ws, out, entry, poison_nan, counters)
    for i in range(groups):  # ponytail: one launch per local group (TP8: 1)
        gemm(x[:, i, :], w[i * n:(i + 1) * n], ws[i * n:(i + 1) * n], out[:, i * n:(i + 1) * n],
             entry, poison_nan, counters)
    return out


# --- capture/replay: the K5 proof the lane runs at load and the boot gate runs on probes -------
class CudaRuntime:
    """torch.cuda graph capture on a side stream."""
    name = "cuda"

    @staticmethod
    def sync() -> None:
        torch.cuda.synchronize()

    @staticmethod
    def capture(fn):
        """fn() -> tensor; returns (that tensor at its graph address, replay)."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            y = fn()
        return y, graph.replay


class EagerRuntime:
    """CPU twin for tests: 'capture' runs fn once, 'replay' re-runs it into the same tensor."""
    name = "eager"

    @staticmethod
    def sync() -> None:
        pass

    @staticmethod
    def capture(fn):
        y = fn()

        def replay():
            y.copy_(fn())
        return y, replay


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16),
                                              b.contiguous().view(torch.int16))


def capture_check(run, xs: list, rt=None, replays: int | None = None, what: str = "") -> int:
    """`run(x) -> bf16 tensor` is the lane's work. Eager result per input (the first call warms /
    compiles outside any capture); capture run(static) on a side stream; replay `replays` times
    (default one per input, cycling) with NEW data copied into the static input and compare each
    replay bit-exactly with its eager result. RuntimeError names the first mismatch. Returns the
    number of replays checked."""
    rt = rt or CudaRuntime
    eager = [run(x).clone() for x in xs]
    rt.sync()
    static = xs[0].clone()
    y, replay = rt.capture(lambda: run(static))
    rt.sync()
    count = replays or len(xs)
    for i in range(count):
        j = (i + 1) % len(xs)  # starts on NEW data (the capture saw input 0), then cycles
        static.copy_(xs[j])
        replay()
        rt.sync()
        if not bits_equal(y, eager[j]):
            bad = int((y.contiguous().view(torch.int16) != eager[j].contiguous().view(torch.int16)).sum())
            raise RuntimeError(
                f"{MARKER} CAPTURE/REPLAY FAIL {what}: replay {i} (input {j}) differs from the "
                f"eager result in {bad} of {y.numel()} elements")
    return count


def warm_ms(max_m: int) -> list:
    """Ms that compile every Triton specialization a plan can launch: M = 1,
    and per bucket its top row count and the one below (M % 16 == 0 is its own
    specialization), clipped to max_m."""
    out, prev = {1}, 0
    for b in BUCKETS:
        if prev >= max_m:
            break
        top = min(b, max_m)
        out |= {top, max(prev + 1, top - 1)}
        prev = b
    return sorted(out)


def describe(plan: list) -> str:
    """'M1-8:mxfp8_gemm(split<=8) M9-32:vllm ...' for the load log."""
    runs, last = [], object()
    for m, e in enumerate(plan[1:], 1):
        name = e[0] + (f"(split<={e[2]})" if e[2] > 1 else "") if e else "vllm"
        if name == last:
            runs[-1][1] = m
        else:
            runs.append([m, m, name])
            last = name
    return " ".join(f"M{a}{'' if a == b else f'-{b}'}:{s}" for a, b, s in runs)


# ---------------------------------------------------------------------------
# spec + references (pure torch: CPU tests, the lane's load oracle, the lab)
# ---------------------------------------------------------------------------
def dequant(w: torch.Tensor, ws: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """e4m3 [N, K] * 2^(ws - 127) per 32-block, exact in f32 / f64."""
    n, k = w.shape
    sc = torch.exp2(ws.to(dtype) - 127.0)
    return (w.to(dtype).view(n, k // 32, 32) * sc[:, :, None]).view(n, k)


def quant_spec(x: torch.Tensor):
    """vLLM's MXFP8 activation quant (mxfp8_utils._mxfp8_e4m3_quantize_torch):
    s = ceil(log2(f32(amax) / 448)) + 127 clamped to [0, 254], q = e4m3(x *
    2^(127 - s)). -> (q e4m3 [M, K], s uint8 [M, K/32])."""
    m, k = x.shape
    xb = x.float().view(m, k // 32, 32)
    y = xb.abs().amax(-1).clamp(min=torch.finfo(torch.float32).tiny) / 448.0
    s = (torch.ceil(torch.log2(y.double())) + 127.0).clamp(0, 254)
    q = (xb * torch.exp2(127.0 - s).float()[:, :, None]).to(torch.float8_e4m3fn)
    return q.view(m, k), s.to(torch.uint8)


def exact(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """f64 x @ deq(w).T, row-chunked (the dequantized weight is the big one)."""
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float64, device=x.device)
    xd = x.double()
    step = max(1, (1 << 24) // w.shape[1])
    for r0 in range(0, w.shape[0], step):
        out[:, r0:r0 + step] = xd @ dequant(w[r0:r0 + step], ws[r0:r0 + step]).T
    return out


def check(out, ref):
    """Rel. Frobenius error <= 1e-2 AND every element within 2^-7 |ref| +
    1e-3 max|ref| (bf16 output rounding is 2^-9; fp32 accumulation far below
    1e-3): a wrong column, row or tail cannot hide in the norm."""
    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite output"
    err = (o - r).abs()
    rel = float(err.norm() / r.norm().clamp_min(1e-30))
    bad = int((err > 2.0 ** -7 * r.abs() + 1e-3 * r.abs().max()).sum())
    return rel <= 1e-2 and bad == 0, (f"rel {rel:.2e} (gate 1e-2), {bad} elements past "
                                      "2^-7 |ref| + 1e-3 max|ref|")


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def problem(case: dict, seed: int, device="cuda", sem: bool = False) -> dict:
    """Checkpoint-like operands: e4m3 weights spread over the format's range,
    e8m0 scales 2^-10..2^-4 (what 32x32 blocks of |w| < 0.1 get), x ~ N(0, 1).
    `counters` (the *_sem bodies' zeroed tickets) only when sem."""
    m, n, k = case["numtokens"], case["n"], case["k"]
    gen = torch.Generator(device=device).manual_seed(1000 + 7919 * seed + n + 31 * k)
    w = (torch.randn(n, k, generator=gen, device=device) * 64.0).clamp(-448, 448)
    p = dict(m=m, n=n, k=k, w=w.to(torch.float8_e4m3fn),
             ws=torch.randint(117, 124, (n, k // 32), generator=gen, device=device,
                              dtype=torch.int32).to(torch.uint8),
             x=torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16),
             # zeros, not empty: Helion's autotuner decides which args a kernel mutates by comparing
             # them before / after the baseline run, and NaN garbage (freed gate poison!) never
             # compares equal, which would pin `partial` as "mutated" and reject every split config
             out=torch.zeros(m, n, dtype=torch.bfloat16, device=device),
             partial=torch.zeros(max_splits(n), m, n, dtype=torch.float32, device=device))
    if sem:
        p["counters"] = torch.zeros(counter_elems(n), dtype=torch.int32, device=device)
    return p


def body_args(kernel: str, p: dict) -> tuple:
    m, n, k = p["m"], p["n"], p["k"]
    w = p["w"].view(torch.uint8) if kernel in (A16, A16_SEM) else p["w"]
    args = (p["x"].view(m, k // 32, 32), w.view(n, k // 32, 32), p["ws"], p["out"], p["partial"])
    return args + ((p["counters"],) if kernel in SEM_BODIES else ())


def _w2(w3):
    return w3.reshape(w3.shape[0], -1).view(torch.float8_e4m3fn)


def reference(x, w, ws, out, partial, *_):
    """A16: exact f64 product of the bf16 x and the dequantized weight."""
    return exact(x.reshape(x.shape[0], -1), _w2(w), ws)


def reference_a8(x, w, ws, out, partial, *_):
    """A8: exact f64 product of the spec-quantized x and the weight."""
    q, s = quant_spec(x.reshape(x.shape[0], -1))
    return exact(dequant(q, s, torch.float32), _w2(w), ws)


def _vllm_kernel(n: int, k: int, w, ws, cls=None):
    """(vLLM's SM120 MXFP8 kernel for (N, K) via its own selection, or `cls`
    built directly (the Marlin W8A16 reference), a layer with the weights
    processed by it), FlashInfer autotuned like serving's warmup."""
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    if cls is None:
        kern = init_mxfp8_linear_kernel(weight_shape=(n, k))
    else:
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        kern = cls(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ws.clone(), requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.logical_widths, layer.params_dtype, layer.orig_dtype = [n], torch.bfloat16, torch.bfloat16
    kern.process_weights_after_loading(layer)
    return kern, layer


def _baseline_args(case, seed):
    p = problem(case, seed)
    if "g" in case:  # wo_a: vLLM's dequant-at-load BF16 weight, torch.bmm per group
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            dequant_mxfp8_to_bf16,
        )
        wb = dequant_mxfp8_to_bf16(p["w"], p["ws"]).view(1, p["n"], p["k"])
        z = torch.empty(p["m"], 1, p["n"], dtype=torch.bfloat16, device=wb.device)
        return p["x"].view(1, p["m"], p["k"]), wb, z
    kern, layer = _vllm_kernel(p["n"], p["k"], p["w"], p["ws"])
    x = p["x"]
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    if p["k"] == Q_LORA and kern.input_quant_key() == kMxfp8Dynamic:
        # serving hands these a QuantizedActivation from the fused q/kv RMSNorm
        from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        x = QuantizedActivation(q, s, torch.bfloat16, p["x"].shape, kMxfp8Dynamic)
    try:
        from flashinfer.autotuner import autotune
        with autotune(True):
            kern.apply_weights(layer, x)
    except ImportError:
        pass
    return x, kern, layer


def _marlin_args(case, seed):
    """vLLM's Marlin MXFP8 W8A16 kernel (SM75+, `--linear-backend marlin`): the
    zero-code alternative to FlashInfer CUTLASS, reference only (the plugin
    never serves it)."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    p = problem(case, seed)
    kern, layer = _vllm_kernel(p["n"], p["k"], p["w"], p["ws"], MarlinMxfp8LinearKernel)
    return p["x"], kern, layer


def _baseline(*args):
    if len(args) == 3 and isinstance(args[1], torch.Tensor):  # wo_a bmm
        x, wb, z = args
        return torch.bmm(x, wb.transpose(1, 2), out=z.transpose(0, 1))
    x, kern, layer = args
    return kern.apply_weights(layer, x)


def _kernel(name):
    a16 = name in (A16, A16_SEM)
    return {"name": name, "body": ALL_BODIES[name],
            "cases": lambda model: _cases(model, A16 if a16 else A8),
            "make": lambda case, seed: body_args(name, problem(case, seed, sem=name in SEM_BODIES)),
            "make_baseline": _baseline_args, "baseline": _baseline,
            "reference": reference if a16 else reference_a8, "check": check,
            "settings": {"ignore_warnings": _ignore()},
            "baseline_name": "vLLM SM120 MXFP8 pick (init_mxfp8_linear_kernel: FlashInfer "
                             "CUTLASS mm_mxfp8 + dynamic quant, prequantized at K=1280; "
                             "wo_a: dequant-at-load BF16 torch.bmm)",
            # FlashInfer CUTLASS / cuBLAS kernel names do not tell these linears apart
            "replaces": []}


# A16 first (better numerics, the only wo_a route), then A8. The *_sem bodies are listed
# only under SUFFIX_MXFP8_GEMM_SEMAPHORE=1 (lab comparison of the unproven variant).
KERNELS = [_kernel(A16), _kernel(A8)] + ([_kernel(A16_SEM), _kernel(A8_SEM)] if semaphore_on() else [])


# ---------------------------------------------------------------------------
# kernel-lab driver: the autotune runner's case loop + timing, one JSON line
# per (body, case) on stdout, inside one lab gate call's budget.
# ---------------------------------------------------------------------------
def _time_us(fn, sets, iters=50):
    """us per call: CUDA-graph replays over `sets` (eager if not capturable)."""
    for a in sets:
        fn(*a)
    torch.cuda.synchronize()
    try:
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            for a in sets:
                fn(*a)
        graph.replay()
        step, how = graph.replay, "graph"
    except Exception:  # noqa: BLE001
        torch.cuda.synchronize()
        step, how = (lambda: [fn(*a) for a in sets]), "eager"
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        step()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters / len(sets), how


def lab_main(plan: dict) -> None:
    """plan: kernels [names] (default A16, A8; the *_sem names need SUFFIX_MXFP8_GEMM_SEMAPHORE=1),
    cases [case dicts] (default: _cases), effort none | quick | full, config {..} (effort none: that
    config, else Helion's default), budget_s. Each body is tuned and timed as the composite that
    serves (stage 1 + the torch reduce). Prints {"kernel", "case", "correct", "check", "config",
    "us", "vllm_us", "speedup", "timing", "vllm_kernel", "marlin_us", "vs_marlin"} lines;
    "capture_fail": true when OUR kernel could not be CUDA-graph captured (never a timing detail)."""
    import json
    import time

    names = plan.get("kernels", [A16, A8])
    bad = [n for n in names if n not in ALL_BODIES]
    if bad:
        raise ValueError(f"unknown kernels {bad}: want {list(ALL_BODIES)}")
    if any(n in SEM_BODIES for n in names) and not semaphore_on():
        raise RuntimeError(f"{MARKER} lab_main: the *_sem bodies need {SEM_ENV}=1 (unproven under "
                           "CUDA-graph capture: lab and boot gate only)")
    try:
        import sklearn.ensemble  # noqa: F401 — what LFBO imports
        os.environ["HELION_AUTOTUNER"] = plan.get("autotuner", "LFBOTreeSearch")
    except Exception:  # noqa: BLE001
        os.environ["HELION_AUTOTUNER"] = plan.get("autotuner", "PatternSearch")
    end = time.time() + plan.get("budget_s", 1500)
    effort = plan.get("effort", "none")
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 128 << 20)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "platform": hk.platform(),
                      "autotuner": os.environ["HELION_AUTOTUNER"], "effort": effort}), flush=True)
    for name in names:
        a16, sem = name in (A16, A16_SEM), name in SEM_BODIES
        body = ALL_BODIES[name]
        for case in plan.get("cases") or _cases({"name": "deepseek-v4.1"}, A16 if a16 else A8):
            if "g" in case and not a16:
                continue
            rec = {"kernel": name, "case": case}
            if time.time() > end:
                print(json.dumps({**rec, "skipped": "budget"}), flush=True)
                continue
            t0 = time.time()
            try:
                args = body_args(name, problem(case, 0, sem=sem))
                ref = (reference if a16 else reference_a8)(*args)
                if effort == "none":
                    cfg = plan.get("config")
                    best = helion.kernel(body, static_shapes=True, ignore_warnings=_ignore(),
                                         **({"config": helion.Config(**cfg)} if cfg else
                                            {"autotune_effort": "none"}))
                    cfg = cfg or "helion default"
                else:
                    tuner = helion.kernel(body, autotune_effort=effort, static_shapes=True,
                                          ignore_warnings=_ignore())
                    cfg = tuner.autotune(args, force=True)
                    best = helion.kernel(body, config=cfg, static_shapes=True,
                                         ignore_warnings=_ignore())
                rec["config"] = json.loads(cfg.to_json()) if hasattr(cfg, "to_json") else cfg
                rec["correct"], rec["check"] = check(best(*args), ref)
                if rec["correct"]:
                    n_bytes = case["n"] * case["k"] * 1.03 + case["numtokens"] * case["k"] * 2
                    copies = max(1, min(32, int(-(-2 * l2 // max(1, int(n_bytes))))))
                    sets = [args] + [body_args(name, problem(case, s, sem=sem))
                                     for s in range(1, copies)]
                    base = [_baseline_args(case, s) for s in range(copies)]
                    rec["us"], rec["timing"] = _time_us(best, sets)
                    rec["capture_fail"] = rec["timing"] != "graph"
                    rec["vllm_us"], _ = _time_us(_baseline, base)
                    rec["speedup"] = round(rec["vllm_us"] / rec["us"], 4)
                    try:  # Marlin W8A16 as a second reference (never fatal)
                        mar = [_marlin_args(case, s) for s in range(copies)]
                        rec["marlin_us"] = round(_time_us(_baseline, mar)[0], 3)
                        rec["vs_marlin"] = round(rec["marlin_us"] / rec["us"], 4)
                        del mar
                    except Exception as exc:  # noqa: BLE001
                        rec["marlin_error"] = f"{type(exc).__name__}: {exc}"[:300]
                    rec["vllm_kernel"] = (type(base[0][1]).__name__ if "g" not in case
                                          else "torch.bmm(bf16)")
                    rec["us"], rec["vllm_us"] = round(rec["us"], 3), round(rec["vllm_us"], 3)
                    del sets, base
            except Exception as exc:  # noqa: BLE001 — a result, keep going
                rec["error"] = f"{type(exc).__name__}: {exc}"[:1500]
                if "illegal" in str(exc) or "CUDA error" in str(exc):
                    print(json.dumps(rec), flush=True)
                    return
            rec["s"] = round(time.time() - t0, 1)
            print(json.dumps(rec), flush=True)
            torch.cuda.empty_cache()
