# SPDX-License-Identifier: Apache-2.0
"""MoE routing glue in two Triton launches at small M (SUFFIX_ROCM_MOE_ROUTE=1, ROCm).

Per MoE layer of Qwen3.8-Flash-Next on vLLM's ROCm AITER path (shared expert fused as
expert 512: 10 routed + 1 shared = 11 slots, our tuned FlyDSL fMoE table) the experts
are preceded by four launches: aiter.topk_softmax (vllm::moe::topkGatingSoftmax,
~11 us on the MI350P), AITER's opus moe_sorting (513 experts -> the multi-phase
P0_v2 + P23 pair, ~8 us) and fused_dynamic_mxfp4_quant_moe_sort (~4 us); ~52 MoE
layers per MTP-4 step. At M <= MAX_M two launches do all of it, bit-for-bit:
launch 1 (grid M x 2, one wave each) gates token t (+ its 32-word expert bitmap) and,
in parallel, writes its MXFP4 row and its E8M0 bytes in token order; launch 2 (one
program, integer work) derives AITER's sort from the bitmaps (popcounts, no gather /
sort / scan / histogram: this Triton asserts on tl.gather layouts) and scatters each
valid slot's E8M0 bytes into the shuffled layout. (One program for everything was
latency-bound: 44 us at M 1 vs stock 20.)

1. Gating = AITER's topkGatingSoftmax (csrc/kernels/topk_softmax_kernels.cu @
   v0.1.24.post1, renormalize, shared scoring "sigmoid"): the 10 largest of the 512
   routed bf16 logits in descending order, ties to the lower expert;
   w_k = expf(l_k - l_max) * (1 / sum_k expf(l_k - l_max)), the sum in k order;
   w_10 = 1 / (1 + expf(-l_512)). Written where the router's call would have written:
   weights[:M, :11] and ids[:M, :10] of vLLM's aiter_topK_meta_data buffers (ids[:, 10]
   = 512 is vLLM's one-time prefill). AITER builds it with hipcc -O3
   -fgpu-flush-denormals-to-zero and no fast-math: expf = ocml -> llvm.exp.f32, "/" is the
   correctly rounded division. Triton's "/" is the same fdiv; its tl.exp is
   exp2(x * log2e), and FP fusion (on by default) contracts mul + add pairs inside an
   inlined exp, so this kernel compiles with enable_fp_fusion=False and its exp is
   SUFFIX_ROCM_MOE_ROUTE_EXP (oracle-calibrated): 0 = libdevice (ocml) exp, 1 = tl.exp,
   2 = LLVM's AMDGPU llvm.exp.f32 expansion written out (x * log2e split hi + lo with
   the 0x1.4ae0bep-26 term, rndne, v_exp_f32, ldexp, under/overflow), 3 = the same
   without the lo term. Top-k ids and everything below do not depend on it.

2. Sorting = aiter.fused_moe.moe_sorting -> moe_sorting_opus.h multi-phase path (513
   experts >= 512 is never "oneshot"; no expert mask, dispatch policy 0), unit B =
   block_m of the resolved fMoE row:
     experts ascend; an expert with c_e > 0 tokens owns nb_e = ceil(c_e / B) blocks of
     B slots starting at B * sum_{e' < e} nb_e' (experts without tokens own nothing);
     within an expert tokens ascend; sorted_ids[slot] = token | k << 24 (k = the
     token's column in ids), padding slots M | 11 << 24 (MOCK_ID(tokens, topk));
     sorted_weights[slot] = weights[token, k], padding 0.0; sorted_expert_ids[block] =
     e; num_valid_ids = [B * sum nb_e, M]. Slots and blocks past num_valid_ids[0] stay
     unwritten (torch.empty, as stock). accumulate (atomic stage 2): moe_buf [M, H]
     comes back zeroed, else it is a (0, 0) placeholder.

3. Stage-1 activation quant = fused_mx_quant_moe_sort_kernel (quant_kernels.cu, its
   top-k is 1 at stage 1: one row per token):
     a1 [M, H/2] fp4x2: per 32-column group amax = max(1e-10, |x|) over the bf16 row;
     E8M0 byte e = exponent(amax * fp32(1/6)), + 1 if that product's mantissa != 0
     (ceil_pow2(amax / 6)); nibble = RNE e2m1(x / 2^(e - 127)) with the sign kept
     (-0 -> 0x8), even column in the low nibble.
     a1_scale [pad32(len(sorted_ids)), SN = pad8(H / 32)] E8M0: valid slot r of token t
     holds t's bytes at mx_scale_shuffle_idx(SN, r, y) = (r/32*SN)*32 + (y/8)*256 +
     (y%4)*64 + (r%16)*4 + (y%8)/4*2 + (r%32)/16; padding slots stay unwritten.
     (Stock zeroes a token's whole a1 row when one of its slots has weight exactly 0.0,
     racing the other slots' writes of the same row; unreachable for real logits and
     not reproduced.)

Wiring: vLLM's AiterSharedRoutedFusedMoERouter defers its topk_softmax call (the
router runs inside the opaque moe_forward custom op, right before the experts);
aiter.fused_moe's module-level moe_sorting runs the fused kernel when the call is
eligible (same ids buffer, no expert mask / local tokens / FLAT / aux sort, policy 0),
else the router's own topk call first and then the stock sort. The stage-1
fused_dynamic_mxfp4_quant_moe_sort returns this kernel's (a1, a1_scale) when its input
is the router's hidden_states, else runs stock. A deferred job that nothing consumes
raises at the next MoE layer: a silent stale top-k is not possible.

    python -m suffix_hybrid.kernels.moe_route_rocm   # GPU oracle + us/call (boot gate moe_route_bench)
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Callable, NamedTuple

import torch

from vllm.triton_utils import tl, triton

try:  # the ROCm image; CPU tests only exercise the Python glue
    from triton.language.extra import libdevice
except ImportError:
    libdevice = None

MARK = "[suffix moe-route]"
# ponytail: one program does the whole batch; above 64 tokens the stock multi-CU kernels win.
# MI350P k33 gate, glue graphed us stock -> fused: M 1 20.1 -> 10.0, 5 21.9 -> 13.0, 8 21.9 -> 15.3,
# 16 24.9 -> 22.6, 32 28.9 -> 35.7, 40 30.1 -> 53.1, 64 33.1 -> 66.2: fused up to 16 rows.
MAX_M = min(64, int(os.environ.get("SUFFIX_ROCM_MOE_ROUTE_MAX_M", "16")))
EXP = int(os.environ.get("SUFFIX_ROCM_MOE_ROUTE_EXP", "2"))
BLOCKS = (16, 32, 64, 128)
STATS: Counter = Counter()
_JOBS: dict = {}   # router ids buffer data_ptr -> Job (at most one: layers run in order)
_QUANT: dict = {}  # sorted_ids data_ptr -> (hidden, a1, a1_scale) for stage 1
_ORIG: dict = {}   # aiter.fused_moe's own moe_sorting / fused_dynamic_mxfp4_quant_moe_sort


class Job(NamedTuple):
    logits: torch.Tensor   # [M, E + 1] bf16 router logits (column E: shared expert)
    hidden: torch.Tensor   # [M, H] bf16 MoE input
    weights: torch.Tensor  # [M, k + 1] fp32, the router's total_topk_weights[:M]
    ids: torch.Tensor      # [M, k + 1] int32, total_topk_ids[:M] (column k = E)
    stock: Callable | None  # the router's own topk call, for an ineligible sort


@triton.jit
def _exp(x, MODE: tl.constexpr):
    if MODE == 0:
        return libdevice.exp(x)
    elif MODE == 1:
        return tl.exp(x)
    else:  # AMDGPU lowering of llvm.exp.f32 (fast-FMA path), fusion off
        ph = x * 1.4426950216293335  # log2(e) rounded to fp32
        pl = tl.fma(x, 1.4426950216293335, -ph)
        if MODE == 2:
            pl = tl.fma(x, 1.925962855864327e-08, pl)  # 0x1.4ae0bep-26
        e = libdevice.rint(ph)
        r = libdevice.ldexp(tl.math.exp2((ph - e) + pl), e.to(tl.int32))
        r = tl.where(x < -103.2789306640625, 0.0, r)  # -0x1.9d1da0p+6
        return tl.where(x > 88.72283935546875, float("inf"), r)  # 0x1.62e430p+6


@triton.jit
def _gate_row(logits_ptr, w_ptr, ids_ptr, map_ptr, t, stride_l, stride_w, stride_ids,
              E: tl.constexpr, K: tl.constexpr, EXP_MODE: tl.constexpr):
    """AITER's topkGatingSoftmax for token t: weights [t, :K+1], ids [t, :K], and the token's
    expert set as a 32-word bitmap map[t, :] (bit e % 32 of word e // 32)."""
    cols = tl.arange(0, E)
    kk = tl.arange(0, 16)
    x = tl.load(logits_ptr + t * stride_l + cols).to(tl.float32)
    # Garbage rows (NaN / -inf, e.g. vLLM's warm-up passes over uninitialized state) must still
    # give K distinct experts: AITER's sort tolerates a repeated id, launch 2 does not (k34
    # crash). NaN counts as -inf and a tie among -inf takes the lowest free expert; rows with
    # K finite logits select exactly as before.
    x = tl.where(x == x, x, float("-inf"))
    free = cols >= 0
    top = tl.max(x, axis=0)
    ids = tl.zeros([16], dtype=tl.int32) + E  # slot K: the shared expert
    num = tl.zeros([16], dtype=tl.float32)
    lane0 = tl.zeros([16], dtype=tl.float32)  # math on [16] vectors with equal lanes, as
    den = lane0                               # gate7's tiles (no extern calls on scalars)
    for k in tl.static_range(K):
        v = tl.max(x, axis=0)
        e = tl.min(tl.where((x == v) & free, cols, E), axis=0)
        n = _exp(lane0 + (v - top), EXP_MODE)
        num = tl.where(kk == k, n, num)
        ids = tl.where(kk == k, e, ids)
        den += n  # AITER: k ascending
        free = free & (cols != e)
        x = tl.where(cols == e, float("-inf"), x)
    inv = tl.where(den != 0.0, 1.0 / den, 1.0)
    shared = lane0 + tl.load(logits_ptr + t * stride_l + E).to(tl.float32)
    sig = 1.0 / (1.0 + _exp(-shared, EXP_MODE))
    tl.store(w_ptr + t * stride_w + kk, tl.where(kk == K, sig, num * inv), mask=kk <= K)
    tl.store(ids_ptr + t * stride_ids + kk, ids, mask=kk < K)
    words = tl.arange(0, 32)
    hit = ((ids >> 5)[:, None] == words[None, :]) & (kk <= K)[:, None]
    tl.store(map_ptr + t * 32 + words, tl.sum(tl.where(hit, 1 << (ids & 31)[:, None], 0), axis=0))


@triton.jit(do_not_specialize=["M"])
def _gate_kernel(logits_ptr, w_ptr, ids_ptr, map_ptr, M, stride_l, stride_w, stride_ids,
                 E: tl.constexpr, K: tl.constexpr, EXP_MODE: tl.constexpr):
    _gate_row(logits_ptr, w_ptr, ids_ptr, map_ptr, tl.program_id(0), stride_l, stride_w,
              stride_ids, E, K, EXP_MODE)


@triton.jit
def _popc(x):
    v = x.to(tl.uint32, bitcast=True)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _e2m1_pack(qx, N: tl.constexpr, MP: tl.constexpr):
    # AITER v0.1.24.post1 _mxfp4_pack_bits (aiter/ops/triton/_triton_kernels/quant/quant.py):
    # RNE to E2M1 with the sign kept; AITER checked it against v_cvt_scalef32_pk_fp4_f32,
    # the instruction the HIP quant kernel uses, byte for byte.
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qf = qx.to(tl.float32, bitcast=True)
    sat = qf >= 6
    den = (not sat) & (qf < 1)
    nrm = not (sat | den)
    denormal_x = (qf + 4194304.0).to(tl.uint32, bitcast=True)  # + 2^22: RNE to 0.5
    denormal_x -= 149 << 23
    denormal_x = denormal_x.to(tl.uint8)
    normal_x = qx
    mant_odd = (normal_x >> 22) & 1
    normal_x += 3240099839  # ((1 - 127) << 23) + (1 << 21) - 1 mod 2^32: Triton rejects a
                            # negative scalar on an unsigned tensor; the add wraps the same
    normal_x += mant_odd
    normal_x = (normal_x >> 22).to(tl.uint8)
    e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1 = tl.where(nrm, normal_x, e2m1)
    e2m1 = tl.where(den, denormal_x, e2m1)
    e2m1 = e2m1 | (s >> 28).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(e2m1, [MP, N // 2, 2]))
    return lo | (hi << 4)


@triton.jit(do_not_specialize=["buf_cols"])
def _route_tok_kernel(
    logits_ptr, hidden_ptr, w_ptr, ids_ptr, map_ptr, a1_ptr, stage_ptr, buf_ptr,
    stride_l, stride_h, stride_w, stride_ids, buf_cols,
    E: tl.constexpr, K: tl.constexpr, H: tl.constexpr, EXP_MODE: tl.constexpr,
    ZERO_BUF: tl.constexpr,
):
    """Launch 1, grid (M, 2): program (t, 0) gates token t; program (t, 1) writes its MXFP4
    row a1[t], its E8M0 bytes in token order stage[t, :H/32] and zeroes moe_buf row t."""
    t = tl.program_id(0)
    if tl.program_id(1) == 0:
        _gate_row(logits_ptr, w_ptr, ids_ptr, map_ptr, t, stride_l, stride_w, stride_ids,
                  E, K, EXP_MODE)
    else:
        for c in tl.static_range(H // 512):
            xg = tl.reshape(tl.load(hidden_ptr + t * stride_h + c * 512 + tl.arange(0, 512))
                            .to(tl.float32), [16, 32])
            r = (tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-10) * (1.0 / 6.0)).to(tl.uint32,
                                                                               bitcast=True)
            ex = (r >> 23) & 255
            ex = tl.where((ex < 255) & ((r & 8388607) != 0), ex + 1, ex)
            q = xg * ((254 - ex) << 23).to(tl.float32, bitcast=True)[:, None]  # / 2^(ex-127)
            tl.store(a1_ptr + t * (H // 2) + c * 256 + tl.arange(0, 256)[None, :],
                     _e2m1_pack(tl.reshape(q, [1, 512]), 512, 1))
            tl.store(stage_ptr + t * (H // 32) + c * 16 + tl.arange(0, 16), ex.to(tl.uint8))
        if ZERO_BUF:
            zo = tl.arange(0, 512)
            for z in range(0, buf_cols, 512):
                tl.store(buf_ptr + t * buf_cols + z + zo, tl.zeros([512], dtype=buf_ptr.dtype.element_ty),
                         mask=z + zo < buf_cols)


@triton.jit(do_not_specialize=["M", "n_pad"])
def _route_sort_kernel(
    w_ptr, ids_ptr, map_ptr, stage_ptr, sid_ptr, sw_ptr, seid_ptr, nv_ptr, a1s_ptr,
    M, n_pad, stride_w, stride_ids,
    E: tl.constexpr, K: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
    MP: tl.constexpr, MAXB: tl.constexpr,
):
    """Launch 2, one program: AITER's sort from launch 1's ids / weights / bitmaps, and each
    valid slot's E8M0 bytes from the token-order staging rows into the shuffled layout."""
    P: tl.constexpr = MP * 16
    SN: tl.constexpr = (H // 32 + 7) // 8 * 8
    i = tl.arange(0, P)  # entry = (token t, slot k)
    t = i // 16
    k = i % 16
    valid = (t < M) & (k <= K)
    e = tl.load(ids_ptr + t * stride_ids + k, mask=valid & (k < K), other=E)
    e = tl.where(valid, e, 1023)
    # tokens of e_i: bit e_i of every token's bitmap; cnt = all of them, rank = earlier ones
    tp = tl.arange(0, MP)
    has = (tl.load(map_ptr + tp[None, :] * 32 + (e >> 5)[:, None],
                   mask=valid[:, None] & (tp < M)[None, :], other=0) >> (e & 31)[:, None]) & 1
    cnt = tl.sum(has, axis=1)
    rank = tl.sum(tl.where(tp[None, :] < t[:, None], has, 0), axis=1)
    nblk = (cnt + BLOCK - 1) // BLOCK
    first = valid & (rank == 0)
    # padded start = BLOCK * sum of lower experts' blocks = BLOCK * sum_s #(lower experts with
    # more than s blocks), each count a popcount over a 32-word expert bitmap
    wi = tl.arange(0, 32)
    bit = 1 << (e & 31)
    wsel = (e >> 5)[:, None] == wi[None, :]
    pre = tl.zeros([P], dtype=tl.int32)
    for s in tl.static_range(MAXB):
        m = tl.sum(tl.where(wsel & (first & (cnt > s * BLOCK))[:, None], bit[:, None], 0), axis=0)
        below = tl.where(wi[None, :] < (e >> 5)[:, None], m[None, :],
                         tl.where(wsel, m[None, :] & (bit - 1)[:, None], 0))
        pre += tl.sum(_popc(below), axis=1)
    dest = pre * BLOCK + rank
    put = valid & (dest < n_pad)  # always true for distinct ids per token; never write past
    tl.store(sid_ptr + dest, t | (k << 24), mask=put)
    tl.store(sw_ptr + dest, tl.load(w_ptr + t * stride_w + k, mask=valid, other=0.0), mask=put)
    opens = valid & (rank % BLOCK == 0)  # first slot of one of the expert's blocks
    tl.store(seid_ptr + dest // BLOCK, e, mask=opens & put)
    last = valid & (rank == cnt - 1)
    for j in tl.static_range(BLOCK):
        pad = last & (cnt + j < nblk * BLOCK) & (dest + 1 + j < n_pad)
        tl.store(sid_ptr + dest + 1 + j, M | ((K + 1) << 24), mask=pad)
        tl.store(sw_ptr + dest + 1 + j, 0.0, mask=pad)
    two = tl.arange(0, 2)
    total = tl.minimum(tl.sum(opens.to(tl.int32), axis=0) * BLOCK, n_pad)
    tl.store(nv_ptr + two, tl.where(two == 0, total, M))
    d = dest[:, None]
    for c in tl.static_range((H // 32 + 31) // 32):
        y = (c * 32 + tl.arange(0, 32))[None, :]
        ok = put[:, None] & (y < H // 32)
        sb = tl.load(stage_ptr + t[:, None] * (H // 32) + y, mask=ok, other=0)
        addr = ((d // 32 * SN) * 32 + (y // 8) * 256 + (y % 4) * 64 + (d % 16) * 4
                + (y % 8) // 4 * 2 + (d % 32) // 16)
        tl.store(a1s_ptr + addr, sb, mask=ok)


def _mp(m: int) -> int:
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64


def _route(job: Job, block: int, accumulate: bool, model_dim: int, buf_dtype, output,
           exp_mode: int = EXP):
    """moe_sorting's return tuple from the two launches; (a1, a1_scale) go to _QUANT."""
    from aiter import dtypes

    m, slots = job.ids.shape
    e1, h = job.logits.shape[1], job.hidden.shape[1]
    dev = job.ids.device
    n_blk = triton.cdiv(slots * m + e1 * block - slots, block)  # _moe_sorting_impl's sizes
    n_pad = n_blk * block
    sorted_ids = torch.empty(n_pad, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(n_pad, dtype=torch.float32, device=dev)
    sorted_e = torch.empty(n_blk, dtype=torch.int32, device=dev)
    nvalid = torch.empty(2, dtype=torch.int32, device=dev)
    if not accumulate:
        moe_buf = torch.empty((0, 0), dtype=buf_dtype, device=dev)
    else:
        moe_buf = output if output is not None else torch.empty(
            (m, model_dim), dtype=buf_dtype, device=dev)
    a1 = torch.empty(m, h // 2, dtype=dtypes.fp4x2, device=dev)
    a1s = torch.empty((n_pad + 31) // 32 * 32, (h // 32 + 7) // 8 * 8, dtype=dtypes.fp8_e8m0,
                      device=dev)
    stage = torch.empty(m, h // 32, dtype=torch.uint8, device=dev)
    rowmap = torch.empty(m, 32, dtype=torch.int32, device=dev)
    _route_tok_kernel[(m, 2)](
        job.logits, job.hidden, job.weights, job.ids, rowmap, a1.view(torch.uint8), stage,
        moe_buf, job.logits.stride(0), job.hidden.stride(0), job.weights.stride(0),
        job.ids.stride(0), model_dim if accumulate else 0,
        E=e1 - 1, K=slots - 1, H=h, EXP_MODE=exp_mode, ZERO_BUF=bool(accumulate),
        num_warps=1, enable_fp_fusion=False)
    mp = _mp(m)
    _route_sort_kernel[(1,)](
        job.weights, job.ids, rowmap, stage, sorted_ids, sorted_w, sorted_e, nvalid,
        a1s.view(torch.uint8), m, n_pad, job.weights.stride(0), job.ids.stride(0),
        E=e1 - 1, K=slots - 1, H=h, BLOCK=block, MP=mp, MAXB=triton.cdiv(mp, block),
        num_warps=4 if mp <= 16 else 8 if mp <= 32 else 16)
    _QUANT.clear()
    _QUANT[sorted_ids.data_ptr()] = (job.hidden, a1, a1s)
    return sorted_ids, sorted_w, sorted_e, nvalid, moe_buf


def _once(key: str, msg: str) -> None:
    STATS[key] += 1
    if STATS[key] == 1:
        print(f"{MARK} {msg}", flush=True)


def defer(job: Job) -> None:
    if _JOBS:
        raise RuntimeError(f"{MARK} the previous MoE layer's deferred top-k never reached "
                           "aiter.fused_moe's sort; refusing to route on stale ids")
    _QUANT.clear()
    _JOBS[job.ids.data_ptr()] = job


def moe_sorting(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, block_size=32,
                expert_mask=None, num_local_tokens=None, dispatch_policy=0,
                return_local_topk_ids=False, accumulate=True, flat=False, output_aux=False,
                output=None):
    """aiter.fused_moe.moe_sorting with the deferred router job folded in."""
    _QUANT.clear()
    job = _JOBS.pop(topk_ids.data_ptr(), None)
    if _JOBS:
        raise RuntimeError(f"{MARK} aiter.fused_moe got ids that are not the router's "
                           "buffer while a deferred top-k is pending")
    if job is not None:
        if (expert_mask is None and num_local_tokens is None and dispatch_policy == 0
                and not (return_local_topk_ids or flat or output_aux)
                and topk_ids.shape == job.ids.shape
                and topk_weights.data_ptr() == job.weights.data_ptr()
                and num_experts == job.logits.shape[1] and block_size in BLOCKS
                and moebuf_dtype == torch.bfloat16):
            _once("fused", f"first fused route: M={job.ids.shape[0]} block_m={block_size} "
                           f"accumulate={accumulate}")
            return _route(job, block_size, accumulate, model_dim, moebuf_dtype, output)
        _once("stock", f"sort not eligible (block_m={block_size} policy={dispatch_policy} "
                       f"mask={expert_mask is not None} flat={flat} aux={output_aux}): stock")
        job.stock()
    return _ORIG["sort"](topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
                         block_size, expert_mask, num_local_tokens, dispatch_policy,
                         return_local_topk_ids=return_local_topk_ids, accumulate=accumulate,
                         flat=flat, output_aux=output_aux, output=output)


def quant_moe_sort(input, sorted_ids, num_valid_ids, token_num, topk, block_size,
                   num_rows=None, group_size=32, sorted_weights=None,
                   num_experts_upper_bound=None):
    """aiter.fused_moe.fused_dynamic_mxfp4_quant_moe_sort: the fused kernel's stage-1 output
    when the input is the hidden_states it quantized."""
    hit = _QUANT.pop(sorted_ids.data_ptr(), None)
    if (hit is not None and num_rows is None and group_size == 32
            and hit[0].data_ptr() == input.data_ptr() and hit[0].shape == input.shape
            and hit[0].stride() == input.stride() and hit[0].dtype == input.dtype):
        return hit[1], hit[2]
    return _ORIG["quant"](input, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                          token_num=token_num, topk=topk, block_size=block_size,
                          num_rows=num_rows, group_size=group_size,
                          sorted_weights=sorted_weights,
                          num_experts_upper_bound=num_experts_upper_bound)


def install_aiter(module) -> None:
    """rocm_patches after-hook on aiter.fused_moe (both names are module globals there)."""
    if not _ORIG:
        _ORIG["sort"], _ORIG["quant"] = module.moe_sorting, module.fused_dynamic_mxfp4_quant_moe_sort
    module.moe_sorting, module.fused_dynamic_mxfp4_quant_moe_sort = moe_sorting, quant_moe_sort


def _eligible(router, hidden, logits, indices_type) -> bool:
    e = logits.shape[-1] - 1
    return (0 < hidden.shape[0] <= MAX_M and router.num_fused_shared_experts == 1
            and router.renormalize and router.scoring_func == "softmax"
            and router.capture_fn is None and router.eplb_state is None
            and getattr(router, "_routing_replay_out", None) is None  # copies ids on return
            and indices_type in (None, torch.int32) and router.top_k < 16
            and 0 < e < 1023 and e & (e - 1) == 0 and logits.dim() == 2
            and logits.stride(1) == 1 and logits.dtype == torch.bfloat16
            and hidden.dim() == 2 and hidden.is_contiguous() and hidden.dtype == torch.bfloat16
            and hidden.shape[1] % 512 == 0)


def install_router(module) -> None:
    """rocm_patches after-hook on vLLM's aiter_shared_routed_fused_moe_router: the
    fused-sigmoid branch registers its topk call as a job instead of launching it."""
    cls = module.AiterSharedRoutedFusedMoERouter
    stock = cls._compute_routing

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

        meta = rocm_aiter_moe.aiter_topK_meta_data
        if not (_eligible(self, hidden_states, router_logits, indices_type)
                and rocm_aiter_ops.fuse_sigmoid_in_kernel(meta)):
            if 0 < hidden_states.shape[0] <= MAX_M:
                _once("router", f"router not eligible (renormalize={self.renormalize} "
                                f"scoring={self.scoring_func} logits {router_logits.dtype} "
                                f"{tuple(router_logits.shape)}): stock top-k")
            return stock(self, hidden_states, router_logits, indices_type, input_ids=input_ids)
        m, k = hidden_states.shape[0], self.top_k
        weights, ids = meta[0][:m], meta[1][:m]
        topk = module.dispatch_topk_softmax_func(use_rocm_aiter=True)
        defer(Job(router_logits, hidden_states, weights, ids, lambda: topk(
            weights, ids[:, :k], torch.empty(m, k, dtype=torch.int32, device=ids.device),
            router_logits, self.renormalize, 1, "sigmoid")))
        return weights, ids

    cls._compute_routing = _compute_routing


def _scale_addr(r, y, sn):
    """aiter::mx_scale_shuffle_idx (mx_quant_utils.h), elementwise on integer tensors."""
    return ((r // 32 * sn) * 32 + (y // 8) * 256 + (y % 4) * 64 + (r % 16) * 4
            + (y % 8) // 4 * 2 + (r % 32) // 16)


def _row_rel(a, b):
    """Worst row of max |a - b| / max |b| (scale-aware: atomic stage-2 noise on near-zero
    elements does not count as an error)."""
    a, b = a.float(), b.float()
    return float(((a - b).abs().amax(1) / b.abs().amax(1).clamp_min(1e-30)).max())


def _sort_ok(srt, ids, m, block, shared):
    """AITER's sort contract on any routing (incl. garbage rows): num_valid in range, every
    valid slot's token routes to its block's expert at that slot, every (token, slot) once,
    padding = M | 11 << 24."""
    sid, sw, seid, nv = (x.cpu() for x in srt[:4])
    total, ids = int(nv[0]), ids.cpu().long()
    if not (0 < total <= sid.numel() and total % block == 0 and int(nv[1]) == m):
        return False
    sid, seid = sid[:total].long(), seid[: total // block].long()
    tok, slot = sid & 0xFFFFFF, sid >> 24
    real = tok < m
    exp = seid.repeat_interleave(block)
    routed = torch.where(slot.clamp(max=ids.shape[1] - 1) < ids.shape[1] - 1,
                         ids[tok.clamp(max=m - 1), slot.clamp(max=ids.shape[1] - 1)],
                         torch.full_like(tok, shared))
    pairs = tok[real] * 16 + slot[real]
    return (bool((routed[real] == exp[real]).all()) and pairs.numel() == m * ids.shape[1]
            and pairs.unique().numel() == pairs.numel()
            and bool(((tok[~real] == m) & (slot[~real] == ids.shape[1])).all())
            and bool(((seid >= 0) & (seid <= shared)).all()))


def main() -> int:
    """Oracle on silicon at Qwen3.8-Flash-Next's router + MXFP4 MoE (512 routed experts +
    fused shared expert 512, top-10 + 1, hidden 2560, inter 640 padded to 768).
    Criteria: top-k ids, sort structure (sorted ids / expert ids / num_valid), a1 and the
    valid a1_scale bytes bitwise vs aiter.topk_softmax -> moe_sorting ->
    fused_dynamic_mxfp4_quant_moe_sort; sorted weights = the kernel's own top-k weights
    slot for slot; top-k weights vs stock reported as bit-exact share + max fp32 ulps (the
    gating exp is calibrated first); aiter.fused_moe with the router deferral vs stock:
    worst-row max |diff| / max |stock| within max(2 x stock-vs-stock, 2^-8) (atomic stage-2
    rows are not deterministic; bitwise reported when equal); HIP-graph replay on new
    inputs == eager; graphed us/call of the glue and of a whole MoE layer."""
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")  # read at vllm._aiter_ops import
    os.environ.setdefault("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    os.environ.setdefault("AITER_CONFIG_FMOE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
        "mi350p_tuned_fmoe.csv"))
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType, dtypes, topk_softmax
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    install_aiter(fm)
    dev, e1, k1, h, inter, ip = "cuda", 513, 11, 2560, 640, 768
    failed = False

    def inputs(m, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        lg = (torch.randn(m, e1, device=dev, generator=g) * 2.0).to(torch.bfloat16)
        lg[: (m + 1) // 2, 7] = lg[: (m + 1) // 2, 3]                  # ties
        lg[m // 2:, 9] = lg[m // 2:, : e1 - 1].max(1).values           # tie for the top-1
        hs = torch.randn(m, h, device=dev, generator=g).to(torch.bfloat16)
        hs[:, :32] = 0                                                 # amax floor 1e-10
        hs[:, 64:96] = -hs[:, 64:96].abs() * 1e-3                      # -0 nibbles
        hs[:, 96:128] *= 1e4
        return lg, hs

    def bufs(m):
        w = torch.full((m, k1), float("nan"), device=dev)
        ids = torch.full((m, k1), -1, dtype=torch.int32, device=dev)
        ids[:, k1 - 1] = e1 - 1
        return w, ids

    def gate(lg, w, ids):
        return lambda: topk_softmax(w, ids[:, : k1 - 1], torch.empty(
            lg.shape[0], k1 - 1, dtype=torch.int32, device=dev), lg, True, 1, "sigmoid")

    def ulps32(a, b):
        return int((a.view(torch.int32).long() - b.view(torch.int32).long()).abs().max())

    # Gating exp calibration on one input set (fusion off, as in the fused kernels).
    lg, _ = inputs(40, 1)
    w0, i0 = bufs(40)
    gate(lg, w0, i0)()
    rowmap = torch.empty(40, 32, dtype=torch.int32, device=dev)
    for mode in (0, 1, 2, 3):
        w1, i1 = bufs(40)
        _gate_kernel[(40,)](lg, w1, i1, rowmap, 40, lg.stride(0), w1.stride(0), i1.stride(0),
                            E=e1 - 1, K=k1 - 1, EXP_MODE=mode, num_warps=1,
                            enable_fp_fusion=False)
        torch.cuda.synchronize()
        exact = (w0.view(torch.int32) == w1.view(torch.int32)).float().mean().item()
        print(f"{MARK} gating exp mode {mode}: ids {'bitwise' if torch.equal(i0, i1) else 'DIFFER'}, "
              f"weights bit-exact {100 * exact:.1f}% (max {ulps32(w0, w1)} fp32 ulps)"
              f"{' <- configured' if mode == EXP else ''}", flush=True)

    def stock_glue(lg, hs, w, ids, block, acc):
        gate(lg, w, ids)()
        srt = _ORIG["sort"](ids, w, e1, h, torch.bfloat16, block, None, None, 0,
                            accumulate=acc)
        a1, a1s = _ORIG["quant"](hs, sorted_ids=srt[0], num_valid_ids=srt[3],
                                 token_num=lg.shape[0], topk=k1, block_size=block,
                                 sorted_weights=srt[1], num_experts_upper_bound=e1)
        return srt, a1, a1s

    def fused_glue(lg, hs, w, ids, block, acc):
        srt = _route(Job(lg, hs, w, ids, None), block, acc, h, torch.bfloat16, None)
        return (srt, *_QUANT.pop(srt[0].data_ptr())[1:])

    worst_w = 0
    for case, (m, block, acc) in enumerate(((1, 32, False), (2, 32, False), (4, 32, False),
                                            (5, 32, False), (5, 16, False), (5, 64, False),
                                            (8, 32, False), (16, 32, False), (32, 32, True),
                                            (40, 32, False), (40, 64, True), (64, 32, False))):
        lg, hs = inputs(m, 100 + case)
        (w0, i0), (w1, i1) = bufs(m), bufs(m)
        s0, q0, qs0 = stock_glue(lg, hs, w0, i0, block, acc)
        s1, q1, qs1 = fused_glue(lg, hs, w1, i1, block, acc)
        torch.cuda.synchronize()
        nv = int(s0[3][0])
        sid = s0[0][:nv]
        tok, slot = sid & 0xFFFFFF, sid >> 24
        real = tok < m
        rows = torch.nonzero(real).flatten()
        sn = (h // 32 + 7) // 8 * 8
        addr = _scale_addr(rows[:, None], torch.arange(h // 32, device=dev)[None, :], sn)
        own = torch.where(real, w1[tok.clamp(max=m - 1).long(), slot.clamp(max=k1 - 1).long()],
                          torch.zeros_like(s1[1][:nv]))
        res = {
            "ids": torch.equal(i0[:, : k1 - 1], i1[:, : k1 - 1]),
            "num_valid": torch.equal(s0[3], s1[3]),
            "sorted_ids": torch.equal(sid, s1[0][:nv]),
            "expert_ids": torch.equal(s0[2][: nv // block], s1[2][: nv // block]),
            "sorted_w=own": torch.equal(s1[1][:nv].view(torch.int32), own.view(torch.int32)),
            "a1": torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)),
            "a1_scale": torch.equal(qs0.view(torch.uint8).flatten()[addr],
                                    qs1.view(torch.uint8).flatten()[addr]),
        }
        if acc:
            res["moe_buf=0"] = bool((s1[4] == 0).all()) and s1[4].shape == s0[4].shape
        exact = (w0.view(torch.int32) == w1.view(torch.int32)).float().mean().item()
        worst_w = max(worst_w, ulps32(w0, w1))
        good = all(res.values())
        failed |= not good
        bad = [k for k, v in res.items() if not v]
        print(f"{MARK} M={m} block={block} accumulate={acc}: {'MATCH' if good else 'MISMATCH'} "
              f"(valid slots {nv}; weights vs stock bit-exact {100 * exact:.1f}%, "
              f"max {ulps32(w0, w1)} ulps{'; differ: ' + ' '.join(bad) if bad else ''})",
              flush=True)

    # Whole MoE layer: tuned FlyDSL rows on vLLM-padded MXFP4 weights (tools/moe_fp4_oracle.py).
    def quant(x):
        y, s = per_1x32_f4_quant(x.reshape(-1, x.shape[-1]))
        return y.view(torch.uint8).view(*x.shape[:2], -1), s.view(torch.uint8).view(*x.shape[:2], -1)

    torch.manual_seed(0)
    w13 = torch.zeros(e1, 2 * ip, h // 2, dtype=torch.uint8, device=dev)
    w13s = torch.ones(e1, 2 * ip, h // 32, dtype=torch.uint8, device=dev)
    w2 = torch.zeros(e1, h, ip // 2, dtype=torch.uint8, device=dev)
    w2s = torch.ones(e1, h, ip // 32, dtype=torch.uint8, device=dev)
    for a in range(0, e1, 32):
        b = min(e1, a + 32)
        g, gs = quant(torch.randn(b - a, 2 * inter, h, device=dev, dtype=torch.bfloat16) / h**0.5)
        w13[a:b, :inter], w13s[a:b, :inter] = g[:, :inter], gs[:, :inter]
        w13[a:b, ip: ip + inter], w13s[a:b, ip: ip + inter] = g[:, inter:], gs[:, inter:]
        dw, dws = quant(torch.randn(b - a, h, inter, device=dev, dtype=torch.bfloat16) / inter**0.5)
        w2[a:b, :, : inter // 2], w2s[a:b, :, : inter // 32] = dw, dws
    W1 = shuffle_weight(w13.view(dtypes.fp4x2), (16, 16))
    W2 = shuffle_weight(w2.view(dtypes.fp4x2), (16, 16))
    S1 = e8m0_shuffle(w13s.view(e1 * 2 * ip, -1)).view(e1, 2 * ip, -1)
    S2 = e8m0_shuffle(w2s.view(e1 * h, -1)).view(e1, h, -1)
    picked = []
    cfgs = fm.get_2stage_cfgs

    def spy(*a, **k):
        md = cfgs(*a, **k)
        picked.append((md.block_m, getattr(md.stage1, "keywords", {}).get("kernelName")))
        return md

    fm.get_2stage_cfgs = spy

    def moe(hs, w, ids):
        return fm.fused_moe(hs, W1, W2, w, ids, None, ActivationType.Silu, QuantType.per_1x32,
                            False, S1, S2, None, None, dtype=torch.bfloat16, hidden_pad=0,
                            intermediate_pad=ip - inter, swiglu_limit=0.0)

    def stock_layer(lg, hs, w, ids):
        gate(lg, w, ids)()
        return moe(hs, w, ids)

    def fused_layer(lg, hs, w, ids):
        defer(Job(lg, hs, w, ids, gate(lg, w, ids)))
        return moe(hs, w, ids)

    def bound(m, lg, hs):  # stock's own run-to-run spread (atomic stage 2), row-relative
        (wa, ia), (wb, ib) = bufs(m), bufs(m)
        return max(2 * _row_rel(stock_layer(lg, hs, wa, ia), stock_layer(lg, hs, wb, ib)),
                   2.0 ** -8)

    for case, m in enumerate((1, 2, 5, 8, 32, 40, 64)):
        lg, hs = inputs(m, 200 + case)
        (w0, i0), (w1, i1) = bufs(m), bufs(m)
        picked.clear()
        ref = stock_layer(lg, hs, w0, i0)
        fused_before = STATS["fused"]
        out = fused_layer(lg, hs, w1, i1)
        torch.cuda.synchronize()
        used = "fused" if STATS["fused"] > fused_before else "stock fallback"
        rel, lim = _row_rel(out, ref), bound(m, lg, hs)
        good = (bool(torch.isfinite(out).all()) and rel <= lim
                and torch.equal(i0[:, :-1], i1[:, :-1]) and not _JOBS and not _QUANT)
        failed |= not good
        print(f"{MARK} fused_moe M={m} ({used}, row block_m={picked[0][0]} {picked[0][1]}): "
              f"{'MATCH' if good else 'MISMATCH'} ("
              f"{'bitwise' if torch.equal(ref, out) else f'row-relative max diff {rel:.2e}'}, "
              f"bound {lim:.2e})", flush=True)

    for m in (5, 40):  # one graph, replayed on new logits / hidden states
        lg, hs = inputs(m, 300 + m)
        w, ids = bufs(m)
        fused_layer(lg, hs, w, ids)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fused_layer(lg, hs, w, ids)
        nlg, nhs = inputs(m, 400 + m)
        lg.copy_(nlg)
        hs.copy_(nhs)
        g.replay()
        we, ie = bufs(m)
        eager = fused_layer(nlg, nhs, we, ie)
        torch.cuda.synchronize()
        good = torch.equal(eager, out) and torch.equal(ie, ids)
        failed |= not good
        print(f"{MARK} graph replay M={m} on new inputs == eager fused: "
              f"{'MATCH' if good else 'MISMATCH'}", flush=True)

    # vLLM's own path, as in serving: the AiterSharedRoutedFusedMoERouter (deferral installed)
    # and the experts call rocm_aiter_fused_experts makes, each size warmed up eagerly and
    # captured as a HIP graph, sizes descending into ONE shared pool (PIECEWISE capture),
    # replayed in mixed order on fresh inputs vs the stock router + experts; then garbage
    # logits (NaN / -inf / equal rows, as warm-up passes can produce: the k34 crash) through
    # the graphs and through the fused sort, whose contract must hold.
    import vllm.model_executor.layers.fused_moe.router.aiter_shared_routed_fused_moe_router as rmod
    from vllm._aiter_ops import _rocm_aiter_fused_moe_impl
    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    stock_routing = rmod.AiterSharedRoutedFusedMoERouter._compute_routing
    install_router(rmod)
    if rocm_aiter_moe.aiter_topK_meta_data is None:
        rocm_aiter_moe.init_aiter_topK_meta_data(e1 - 1, 1, k1 - 1, 0, 1, 1.0, 8192, False)
    router = rmod.AiterSharedRoutedFusedMoERouter(
        top_k=k1 - 1, global_num_experts=e1, num_fused_shared_experts=1, renormalize=True)

    def experts(hs, w, ids):
        return _rocm_aiter_fused_moe_impl(
            hs, W1, W2, w, ids, expert_mask=None, activation_method=0, quant_method=3,
            doweight_stage1=False, w1_scale=S1, w2_scale=S2, output_dtype=torch.bfloat16,
            hidden_pad=0, intermediate_pad=ip - inter, moe_sorting_dispatch_policy=0,
            swiglu_limit=0.0)

    def vllm_fused(lg, hs):
        w, ids = router.select_experts(hs, lg)
        return experts(hs, w, ids)

    def vllm_stock(lg, hs):
        w, ids = stock_routing(router, hs, lg, None)
        return experts(hs, w, ids)

    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    fused_before = STATS["fused"]
    for m in (16, 8, 5, 4, 2, 1):
        lg, hs = inputs(m, 600 + m)
        vllm_fused(lg, hs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            out = vllm_fused(lg, hs)
        graphs[m] = (g, lg, hs, out)
    for rnd, m in enumerate((1, 16, 5, 8, 2, 4, 16, 8, 1)):
        g, lg, hs, out = graphs[m]
        nlg, nhs = inputs(m, 700 + rnd)
        lg.copy_(nlg)
        hs.copy_(nhs)
        g.replay()
        torch.cuda.synchronize()
        got = out.clone()
        ref = vllm_stock(nlg, nhs)
        rel, lim = _row_rel(got, ref), bound(m, nlg, nhs)
        good = bool(torch.isfinite(got).all()) and rel <= lim and not _JOBS
        failed |= not good
        print(f"{MARK} vLLM path graph M={m} (shared pool, replay {rnd}): "
              f"{'MATCH' if good else 'MISMATCH'} "
              f"({'bitwise' if torch.equal(got, ref) else f'row-relative {rel:.2e}'}, bound {lim:.2e})",
              flush=True)
    for m in (16, 8, 5, 1):
        g, lg, hs, out = graphs[m]
        bad, _ = inputs(m, 800 + m)
        bad[0] = float("nan")
        if m > 1:
            bad[1] = float("-inf")
            bad[1, 100:103] = 1.0
        if m > 2:
            bad[2] = 0.0
        if m > 3:
            bad[3, :505] = float("nan")
        lg.copy_(bad)
        g.replay()
        torch.cuda.synchronize()  # a bad sort faults here (as k34 did)
        w, ids = bufs(m)
        srt = _route(Job(bad, hs, w, ids, None), 32, False, h, torch.bfloat16, None)
        _QUANT.clear()
        torch.cuda.synchronize()
        distinct = all(len(set(r)) == k1 for r in ids.cpu().tolist())
        good = _sort_ok(srt, ids, m, 32, e1 - 1) and distinct
        failed |= not good
        print(f"{MARK} vLLM path graph M={m} on garbage logits (NaN / -inf / equal rows): "
              f"replay ok, fused sort contract {'MATCH' if good else 'MISMATCH'} "
              f"(distinct ids per token {distinct})", flush=True)
    print(f"{MARK} vLLM path warm-up + capture used the fused route in "
          f"{STATS['fused'] - fused_before} of 12 calls (0 = deferral never engaged)", flush=True)
    failed |= STATS["fused"] == fused_before

    reps = 50
    for m in (1, 5, 8, 16, 32, 40, 64):
        data = [inputs(m, 500 + i) + bufs(m) for i in range(reps)]
        t_s = _graph_us(lambda i: stock_glue(*data[i], 32, False), reps)[0]
        t_f = _graph_us(lambda i: fused_glue(*data[i], 32, False), reps)[0]
        line = f"{MARK} M={m} glue (top-k + sort + quant-sort) graphed: stock {t_s:.1f} us -> {t_f:.1f} us"
        if m in (1, 5, 40):
            t_ls = _graph_us(lambda i: stock_layer(*data[i]), 20)[0]
            t_lf = _graph_us(lambda i: fused_layer(*data[i]), 20)[0]
            line += f" | whole MoE layer {t_ls:.1f} -> {t_lf:.1f} us"
        print(line, flush=True)
    print(f"{MARK} weights vs stock: max {worst_w} fp32 ulps over all cases", flush=True)
    print(f"{MARK} " + ("ALL MATCH" if not failed else "SOME MISMATCH"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
