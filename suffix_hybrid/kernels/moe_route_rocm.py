# SPDX-License-Identifier: Apache-2.0
"""MoE routing glue in one Triton launch at small M (SUFFIX_ROCM_MOE_ROUTE=1, ROCm).

Per MoE layer of Qwen3.8-Flash-Next on vLLM's ROCm AITER path (shared expert fused as
expert 512: 10 routed + 1 shared = 11 slots, our tuned FlyDSL fMoE table) the experts
are preceded by four launches: aiter.topk_softmax (vllm::moe::topkGatingSoftmax,
~11 us on the MI350P), AITER's opus moe_sorting (513 experts -> the multi-phase
P0_v2 + P23 pair, ~8 us) and fused_dynamic_mxfp4_quant_moe_sort (~4 us); ~52 MoE
layers per MTP-4 step. At M <= MAX_M one program does all of it, bit-for-bit:

1. Gating = AITER's topkGatingSoftmax (csrc/kernels/topk_softmax_kernels.cu @
   v0.1.24.post1, renormalize, shared scoring "sigmoid"): the 10 largest of the 512
   routed bf16 logits in descending order, ties to the lower expert;
   w_k = expf(l_k - l_max) * (1 / sum_k expf(l_k - l_max)), the sum in k order;
   w_10 = 1 / (1 + expf(-l_512)). Written where the router's call would have written:
   weights[:M, :11] and ids[:M, :10] of vLLM's aiter_topK_meta_data buffers (ids[:, 10]
   = 512 is vLLM's one-time prefill). Which exp / division Triton must use to hit the
   HIP kernel's bits is calibrated by the oracle: SUFFIX_ROCM_MOE_ROUTE_MATH="exp,div"
   (exp 0 = libdevice/ocml expf, 1 = tl.exp; div 0 = div_rn, 1 = Triton's "/").

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
MAX_M = min(64, int(os.environ.get("SUFFIX_ROCM_MOE_ROUTE_MAX_M", "64")))
MATH = tuple(int(v) for v in os.environ.get("SUFFIX_ROCM_MOE_ROUTE_MATH", "0,0").split(","))
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
def _keyed_add(x, y):
    # (key << 16 | count) run lengths over keys sorted ascending (triton_kernels routing)
    kx = x & -65536
    return tl.where(kx == (y & -65536), x + y - kx, y)


@triton.jit
def _exp(x, MODE: tl.constexpr):
    if MODE == 0:
        return libdevice.exp(x)
    else:
        return tl.exp(x)


@triton.jit
def _rcp(x, MODE: tl.constexpr):
    if MODE == 0:
        return tl.math.div_rn(tl.full(x.shape, 1.0, tl.float32), x)
    else:
        return 1.0 / x


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
    normal_x += ((1 - 127) << 23) + (1 << 21) - 1
    normal_x += mant_odd
    normal_x = (normal_x >> 22).to(tl.uint8)
    e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1 = tl.where(nrm, normal_x, e2m1)
    e2m1 = tl.where(den, denormal_x, e2m1)
    e2m1 = e2m1 | (s >> 28).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(e2m1, [MP, N // 2, 2]))
    return lo | (hi << 4)


@triton.jit(do_not_specialize=["M", "buf_numel"])
def _moe_route_kernel(
    logits_ptr, hidden_ptr, w_ptr, ids_ptr,
    sid_ptr, sw_ptr, seid_ptr, nv_ptr, a1_ptr, a1s_ptr, buf_ptr,
    M, stride_l, stride_h, stride_w, stride_ids, buf_numel,
    E: tl.constexpr, K: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
    MP: tl.constexpr, EXP_MODE: tl.constexpr, DIV_MODE: tl.constexpr,
    ZERO_BUF: tl.constexpr,
):
    KP: tl.constexpr = 16             # slots per token (K routed + 1 shared)
    P: tl.constexpr = MP * KP         # (token, slot) entries, row-major
    NB: tl.constexpr = 1024           # expert bins; the last one collects unused entries
    SENT: tl.constexpr = NB - 1
    CW: tl.constexpr = 512            # hidden columns per quant chunk
    G: tl.constexpr = CW // 32        # MX groups per chunk
    SN: tl.constexpr = (H // 32 + 7) // 8 * 8

    # 1. gating
    rows = tl.arange(0, MP)
    rok = rows < M
    cols = tl.arange(0, E)
    x = tl.load(logits_ptr + rows[:, None] * stride_l + cols[None, :], mask=rok[:, None],
                other=float("-inf")).to(tl.float32)
    top = tl.max(x, axis=1)
    kk = tl.arange(0, KP)
    ids = tl.zeros([MP, KP], dtype=tl.int32) + SENT
    num = tl.zeros([MP, KP], dtype=tl.float32)
    den = tl.zeros([MP], dtype=tl.float32)
    for k in tl.static_range(K):
        v = tl.max(x, axis=1)
        e = tl.min(tl.where(x == v[:, None], cols[None, :], E), axis=1)
        n = _exp(v - top, EXP_MODE)
        num = tl.where(kk[None, :] == k, n[:, None], num)
        ids = tl.where(kk[None, :] == k, e[:, None], ids)
        den += n  # AITER: k ascending
        x = tl.where(cols[None, :] == e[:, None], float("-inf"), x)
    inv = tl.where(den != 0.0, _rcp(den, DIV_MODE), 1.0)
    shared = tl.load(logits_ptr + rows * stride_l + E, mask=rok, other=0.0).to(tl.float32)
    sig = _rcp(1.0 + _exp(-shared, EXP_MODE), DIV_MODE)
    w = tl.where(kk[None, :] == K, sig[:, None], num * inv[:, None])
    ok = rok[:, None] & (kk[None, :] <= K)
    ids = tl.where(ok, tl.where(kk[None, :] == K, E, ids), SENT)
    tl.store(w_ptr + rows[:, None] * stride_w + kk[None, :], w, mask=ok)
    tl.store(ids_ptr + rows[:, None] * stride_ids + kk[None, :], ids, mask=ok & (kk[None, :] < K))

    # 2. sort: entries by (expert, token), run-length rank, padded expert starts
    flat = tl.reshape(ids, [P])
    skey = tl.sort((flat << 11) | tl.arange(0, P))
    se = skey >> 11
    si = skey & 2047
    sval = se != SENT
    rank = (tl.associative_scan((se << 16) | 1, 0, _keyed_add) & 65535) - 1
    cnt = tl.histogram(flat, NB)
    eid = tl.arange(0, NB)
    nblk = tl.where(eid != SENT, (cnt + BLOCK - 1) // BLOCK, 0)
    start = (tl.cumsum(nblk, 0) - nblk) * BLOCK
    dest = tl.gather(start, se, 0) + rank
    tok = si // KP
    tl.store(sid_ptr + dest, tok | ((si % KP) << 24), mask=sval)
    tl.store(sw_ptr + dest, tl.gather(tl.reshape(w, [P]), si, 0), mask=sval)
    used = nblk > 0
    for j in tl.static_range(BLOCK):
        pad = used & (cnt + j < nblk * BLOCK)
        tl.store(sid_ptr + start + cnt + j, M | ((K + 1) << 24), mask=pad)
        tl.store(sw_ptr + start + cnt + j, 0.0, mask=pad)
    for j in tl.static_range((MP + BLOCK - 1) // BLOCK):
        tl.store(seid_ptr + start // BLOCK + j, eid, mask=used & (j < nblk))
    two = tl.arange(0, 2)
    tl.store(nv_ptr + two, tl.where(two == 0, tl.sum(nblk, 0) * BLOCK, M))
    if ZERO_BUF:
        zo = tl.arange(0, 2048)
        for z in range(0, buf_numel, 2048):
            tl.store(buf_ptr + z + zo, tl.zeros([2048], dtype=buf_ptr.dtype.element_ty),
                     mask=z + zo < buf_numel)

    # 3. MXFP4 rows per token, E8M0 bytes per sorted slot
    d = dest[:, None]
    for c in tl.static_range(H // CW):
        hc = c * CW + tl.arange(0, CW)
        xg = tl.reshape(tl.load(hidden_ptr + rows[:, None] * stride_h + hc[None, :],
                                mask=rok[:, None], other=0.0).to(tl.float32), [MP, G, 32])
        r = (tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-10) * (1.0 / 6.0)).to(tl.uint32,
                                                                           bitcast=True)
        ex = (r >> 23) & 255
        ex = tl.where((ex < 255) & ((r & 8388607) != 0), ex + 1, ex)
        q = xg * ((254 - ex) << 23).to(tl.float32, bitcast=True)[:, :, None]  # / 2^(ex-127)
        bo = c * (CW // 2) + tl.arange(0, CW // 2)
        tl.store(a1_ptr + rows[:, None] * (H // 2) + bo[None, :],
                 _e2m1_pack(tl.reshape(q, [MP, CW]), CW, MP), mask=rok[:, None])
        sc = tl.gather(ex.to(tl.int32), tl.broadcast_to(tok[:, None], [P, G]), 0)
        y = (c * G + tl.arange(0, G))[None, :]
        addr = ((d // 32 * SN) * 32 + (y // 8) * 256 + (y % 4) * 64 + (d % 16) * 4
                + (y % 8) // 4 * 2 + (d % 32) // 16)
        tl.store(a1s_ptr + addr, sc.to(tl.uint8), mask=sval[:, None])


def _route(job: Job, block: int, accumulate: bool, model_dim: int, buf_dtype, output,
           math: tuple = MATH):
    """moe_sorting's return tuple from one launch; (a1, a1_scale) go to _QUANT."""
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
    mp = 8 if m <= 8 else 32 if m <= 32 else 64
    _moe_route_kernel[(1,)](
        job.logits, job.hidden, job.weights, job.ids, sorted_ids, sorted_w, sorted_e, nvalid,
        a1.view(torch.uint8), a1s.view(torch.uint8), moe_buf,
        m, job.logits.stride(0), job.hidden.stride(0), job.weights.stride(0),
        job.ids.stride(0), moe_buf.numel(),
        E=e1 - 1, K=slots - 1, H=h, BLOCK=block, MP=mp, EXP_MODE=math[0], DIV_MODE=math[1],
        ZERO_BUF=bool(accumulate), num_warps=4 if mp <= 8 else 8 if mp <= 32 else 16)
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


def main() -> int:
    """Oracle on silicon at Qwen3.8-Flash-Next's router + MXFP4 MoE (512 routed experts +
    fused shared expert 512, top-10 + 1, hidden 2560, inter 640 padded to 768): every
    output of aiter.topk_softmax -> moe_sorting -> fused_dynamic_mxfp4_quant_moe_sort vs
    the fused kernel bitwise; aiter.fused_moe with the router deferral vs stock bitwise;
    HIP-graph replay with new inputs; graphed us/call of the glue and of a whole MoE layer."""
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
    torch.manual_seed(0)
    failed = False

    def inputs(m):
        lg = (torch.randn(m, e1, device=dev) * 2.0).to(torch.bfloat16)
        lg[: (m + 1) // 2, 7] = lg[: (m + 1) // 2, 3]                  # ties
        lg[m // 2:, 9] = lg[m // 2:, : e1 - 1].max(1).values           # tie for the top-1
        hs = torch.randn(m, h, device=dev).to(torch.bfloat16)
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

    def stock_glue(lg, hs, w, ids, block, acc):
        gate(lg, w, ids)()
        srt = _ORIG["sort"](ids, w, e1, h, torch.bfloat16, block, None, None, 0,
                            accumulate=acc)
        a1, a1s = _ORIG["quant"](hs, sorted_ids=srt[0], num_valid_ids=srt[3],
                                 token_num=lg.shape[0], topk=k1, block_size=block,
                                 sorted_weights=srt[1], num_experts_upper_bound=e1)
        return srt, a1, a1s

    def fused_glue(lg, hs, w, ids, block, acc, math=MATH):
        srt = _route(Job(lg, hs, w, ids, None), block, acc, h, torch.bfloat16, None, math)
        return (srt, *_QUANT.pop(srt[0].data_ptr())[1:])

    def compare(m, block, acc, math=MATH):
        lg, hs = inputs(m)
        (w0, i0), (w1, i1) = bufs(m), bufs(m)
        s0, q0, qs0 = stock_glue(lg, hs, w0, i0, block, acc)
        s1, q1, qs1 = fused_glue(lg, hs, w1, i1, block, acc, math)
        torch.cuda.synchronize()
        nv = int(s0[3][0])
        rows = torch.nonzero((s0[0][:nv] & 0xFFFFFF) < m).flatten()
        sn = (h // 32 + 7) // 8 * 8
        addr = _scale_addr(rows[:, None], torch.arange(h // 32, device=dev)[None, :], sn)
        res = {
            "ids": torch.equal(i0[:, : k1 - 1], i1[:, : k1 - 1]),
            "num_valid": torch.equal(s0[3], s1[3]),
            "sorted_ids": torch.equal(s0[0][:nv], s1[0][:nv]),
            "sorted_w": torch.equal(s0[1][:nv].view(torch.int32), s1[1][:nv].view(torch.int32)),
            "expert_ids": torch.equal(s0[2][: nv // block], s1[2][: nv // block]),
            "a1": torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)),
            "a1_scale": torch.equal(qs0.view(torch.uint8).flatten()[addr],
                                    qs1.view(torch.uint8).flatten()[addr]),
        }
        if acc:
            res["moe_buf"] = bool((s1[4] == 0).all()) and s1[4].shape == s0[4].shape
        wexact = (w0.view(torch.int32) == w1.view(torch.int32)).float().mean().item()
        return res, wexact, nv

    # Which Triton exp / division reproduce the HIP kernel's weights.
    for math in ((0, 0), (0, 1), (1, 0), (1, 1)):
        _, wexact, _ = compare(40, 32, False, math)
        print(f"{MARK} gating math exp={math[0]} div={math[1]}: weights bit-exact "
              f"{100 * wexact:.1f}%{' <- configured' if math == MATH else ''}", flush=True)
    for m, block, acc in ((1, 32, False), (2, 32, False), (4, 32, False), (5, 32, False),
                          (5, 16, False), (5, 64, False), (8, 32, False), (16, 32, False),
                          (32, 32, True), (40, 32, False), (40, 64, True), (64, 32, False)):
        res, wexact, nv = compare(m, block, acc)
        good = all(res.values()) and wexact == 1.0
        failed |= not good
        bad = [k for k, v in res.items() if not v] + ([] if wexact == 1.0 else ["weights"])
        print(f"{MARK} M={m} block={block} accumulate={acc}: {'MATCH' if good else 'MISMATCH'} "
              f"(valid slots {nv}, weights bit-exact {100 * wexact:.1f}%"
              f"{', differ: ' + ' '.join(bad) if bad else ''})", flush=True)

    # Whole MoE layer: tuned FlyDSL rows on vLLM-padded MXFP4 weights (tools/moe_fp4_oracle.py).
    def quant(w):
        y, s = per_1x32_f4_quant(w.reshape(-1, w.shape[-1]))
        return y.view(torch.uint8).view(*w.shape[:2], -1), s.view(torch.uint8).view(*w.shape[:2], -1)

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

    for m in (1, 2, 5, 8, 32, 40, 64):
        lg, hs = inputs(m)
        (w0, i0), (w1, i1) = bufs(m), bufs(m)
        picked.clear()
        ref = stock_layer(lg, hs, w0, i0)
        before = dict(STATS)
        out = fused_layer(lg, hs, w1, i1)
        torch.cuda.synchronize()
        used = "fused" if STATS["fused"] > before.get("fused", 0) else "stock fallback"
        good = torch.equal(ref, out) and not _JOBS and not _QUANT
        failed |= not good
        print(f"{MARK} fused_moe M={m} ({used}, row block_m={picked[0][0]} "
              f"{picked[0][1]}): {'MATCH' if good else 'MISMATCH'} bitwise "
              f"(max abs diff {(ref.float() - out.float()).abs().max().item():.3e})", flush=True)

    for m in (5, 40):  # one graph, replayed on new logits / hidden states
        lg, hs = inputs(m)
        w, ids = bufs(m)
        fused_layer(lg, hs, w, ids)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fused_layer(lg, hs, w, ids)
        nlg, nhs = inputs(m)
        lg.copy_(nlg)
        hs.copy_(nhs)
        g.replay()
        w0, i0 = bufs(m)
        ref = stock_layer(nlg, nhs, w0, i0)
        torch.cuda.synchronize()
        good = torch.equal(ref, out) and torch.equal(i0[:, :-1], ids[:, :-1])
        failed |= not good
        print(f"{MARK} graph replay M={m} on new inputs vs eager stock: "
              f"{'MATCH' if good else 'MISMATCH'}", flush=True)

    reps = 50
    for m in (1, 5, 8, 40, 64):
        data = [inputs(m) + bufs(m) for _ in range(reps)]
        t_s = _graph_us(lambda i: stock_glue(*data[i], 32, False), reps)[0]
        t_f = _graph_us(lambda i: fused_glue(*data[i], 32, False), reps)[0]
        line = f"{MARK} M={m} glue (top-k + sort + quant-sort) graphed: stock {t_s:.1f} us -> {t_f:.1f} us"
        if m in (1, 5, 40):
            t_ls = _graph_us(lambda i: stock_layer(*data[i % 20]), 20)[0]
            t_lf = _graph_us(lambda i: fused_layer(*data[i % 20]), 20)[0]
            line += f" | whole MoE layer {t_ls:.1f} -> {t_lf:.1f} us"
        print(line, flush=True)
    print(f"{MARK} " + ("ALL MATCH" if not failed else "SOME MISMATCH"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
