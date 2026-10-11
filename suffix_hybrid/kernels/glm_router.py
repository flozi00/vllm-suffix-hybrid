# SPDX-License-Identifier: Apache-2.0
"""Our MoE router for GLM 5.3 decode on SM120 (SUFFIX_ROUTER_OURS): one Triton kernel for the
router GEMV (x [M, 6144] BF16 @ gate.weight [256, 6144]^T, fp32 logits) and, fused into its
split-K epilogue, vLLM's single-group noaux_tc selection (sigmoid + e_score_correction_bias,
top-k, renormalize, routed scaling).

Why: on SM120 vLLM's GateLinear has no specialized tier (Hopper / SM100 / gfx950 only) and
runs cuBLAS bf16 x bf16 -> fp32 (cutlass_80_wmma 32x32 + a split-K reduce, ~11 us at M=6
for a 3 MB weight that streams in ~1.7 us), then ops.grouped_topk's single_group_topk
block kernel (~2.6 us), per MoE layer and per draft step.

Kernel (grid: K split x N tile x M tile): each program multiplies its K slice (CUDA-core
GEMV for M <= 8, tl.dot above) and stores an fp32 partial; the last split of an (M, N) tile
(acq_rel tile counter, the CUTLASS semaphore pattern of fp8_dense_patch.ours, same per-stream
counters) sums the partials in split order -> logits (deterministic per plan). Fused: that
program also keeps its tile's TOPK largest selection keys per token; the last N tile of the
M tile (second counter) merges NT x TOPK candidates per token into topk weights / ids.

Selection semantics are vLLM's csrc/libtorch_stable/moe/grouped_topk_kernels.cu
(single_group_topk block kernel, nightly 2eb491e1, built without fast-math): unbiased =
0.5 * tanhf(0.5 x) + 0.5, selection = unbiased + bias (-0 -> +0), -inf when the logit or the
bias is not finite (its unbiased then counts 0); key = (cub TwiddleIn(selection) << 32) |
(65535 - expert), larger key first (ties: lower expert; for 256 experts x top-8 its
per-lane placeholders never surface, so it equals the flat key order); weight = unbiased *
(scale / (sum + 1e-20)) with scale the routed scaling factor and sum the warp butterfly
(cg::reduce = shfl_xor 16..1) over the selected lanes; 1/topk each when no selection is
finite. Bit-identity with ops.grouped_topk on the same logits is the load oracle (tanhf and
division rounding included); the logits are our own fp32 sums (rel <= TOL vs fp64, not
cuBLAS' order). TP-sharded experts need the same routing on every rank: plan timings are
max-all-reduced over the TP group, so every rank decides the same.

Gate SUFFIX_ROUTER_OURS: "1" (GEMV + fused selection everywhere up to the max M, both must
pass), "auto" (fastest of fused / our GEMV + vLLM's selection / vLLM per M bucket; fused only
where the oracle proves it exact), "gemv" (our GEMV + vLLM's selection where faster). M above
SUFFIX_ROUTER_OURS_MAX_M (default 192) or anything ineligible stays on vLLM's path.
Bench: python -m suffix_hybrid.kernels.glm_router bench (boot gate glm_router_bench).
"""
from __future__ import annotations

import bisect
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # CPU CI: plans / reference / install logic stay importable
    triton = tl = libdevice = None

GATE = "SUFFIX_ROUTER_OURS"
MAX_M_ENV = "SUFFIX_ROUTER_OURS_MAX_M"
MARK = "[suffix router-ours]"
MODES = ("1", "auto", "gemv")
BUCKETS = (1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192)
NSM = 188
REPS = 15
TOL = 1e-5            # rel. Frobenius of our fp32 logits vs fp64
KMIN = -(1 << 63)     # below every selection key
SEMS = 4096           # counter slots a launch may use: MT * NT + MT
# (BN, BK) for the GEMV family (M <= 8, BM = next pow2); (BN, BK, stages) for tl.dot.
GEMV_TILES = ((32, 256), (16, 512), (64, 128), (32, 128))
MMA_TILES = ((32, 128, 3), (64, 128, 3), (32, 256, 2), (16, 256, 3))

GATE_LINEAR = "vllm.model_executor.layers.fused_moe.router.gate_linear"
TOPK_ROUTER = "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
BASE_LOADER = "vllm.model_executor.model_loader.base_loader"

_STASH = [None]       # (logits, tw, ids, sig, bias_ptr): GateLinear.forward -> grouped_topk
_STATE = {"routes": {}, "layers": 0, "setup_ran": False}


def mode() -> str:
    v = os.environ.get(GATE, "").strip()
    if v in ("", "0"):
        return ""
    if v not in MODES:
        raise SystemExit(f"{MARK} {GATE}={v!r}: want one of {MODES}")
    return v


def max_m() -> int:
    raw = os.environ.get(MAX_M_ENV, "").strip()
    v = int(raw) if raw else BUCKETS[-1]
    if not 1 <= v <= BUCKETS[-1]:
        raise SystemExit(f"{MARK} {MAX_M_ENV}={raw!r}: want 1..{BUCKETS[-1]}")
    return v


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


# --- plans (pure Python) -------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    mma: bool
    bm: int
    bn: int
    bk: int
    kb_split: int     # BK blocks per K split
    n_split: int
    warps: int = 4
    stages: int = 1

    def tag(self) -> str:
        kind = f"mma{self.bm}" if self.mma else f"gemv{self.bm}"
        return f"{kind}x{self.bn}x{self.bk}/k{self.n_split}"


def _cdiv(a, b):
    return -(-a // b)


def plans(n: int, k: int, m: int, nsm: int = NSM) -> list:
    """Candidates at M = m: per tile family, K splits aiming at 1 and 2 waves of nsm."""
    fams = []
    if m <= 8:
        bm = 1 << (m - 1).bit_length()
        fams += [(False, bm, bn, bk, 1) for bn, bk in GEMV_TILES]
    bms = (16,) if m <= 16 else (32,) if m <= 32 else (32, 64)
    fams += [(True, bm, bn, bk, st) for bm in bms for bn, bk, st in MMA_TILES]
    out = []
    for mma, bm, bn, bk, st in fams:
        if n % bn or k % bk:
            continue
        kb, tiles = k // bk, (n // bn) * _cdiv(m, bm)
        seen = set()
        for waves in (1, 2):
            kbs = max(1, _cdiv(kb, max(1, round(waves * nsm / tiles))))
            ns = _cdiv(kb, kbs)
            if ns not in seen and tiles * (1 + 1) <= SEMS:
                seen.add(ns)
                out.append(Plan(mma, bm, bn, bk, kbs, ns, 4, st))
    return out


# --- reference semantics (torch; CPU tests and diagnostics) ---------------------------
_NEG_INF_S32 = int(torch.tensor(float("-inf")).view(torch.int32)) ^ 0x7FFFFFFF


def selection_keys(logits: torch.Tensor, bias: torch.Tensor) -> tuple:
    """(int64 keys [M, N], unbiased fp32 [M, N]) exactly as the kernel builds them."""
    lg, b = logits.float(), bias.float()
    ok = torch.isfinite(lg) & torch.isfinite(b)[None, :]
    u = 0.5 * torch.tanh(0.5 * lg) + 0.5
    sel = u + b[None, :]
    sel = torch.where(sel == 0, torch.zeros_like(sel), sel)
    sel = torch.where(ok, sel, torch.full_like(sel, float("-inf")))
    bits = sel.contiguous().view(torch.int32)
    s32 = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    col = torch.arange(lg.shape[1], dtype=torch.int64, device=lg.device)
    return (s32 << 32) | (65535 - col)[None, :], torch.where(ok, u, torch.zeros_like(u))


def butterfly_sum(v: torch.Tensor) -> torch.Tensor:
    """cg::reduce(warp, v, plus) for lanes [M, T] (T a power of two <= 32, the other lanes
    0): shfl_xor masks 16..1 fold the upper half onto the lower one."""
    s = v.float()
    while s.shape[1] > 1:
        h = s.shape[1] // 2
        s = s[:, :h] + s[:, h:]
    return s[:, 0]


def reference_topk(logits, bias, topk: int, renorm: bool, rsf: float) -> tuple:
    """(weights fp32 [M, topk], ids int32 [M, topk]): vLLM's single_group_topk on these
    logits (lane j = j-th largest key)."""
    keys, u = selection_keys(logits, bias)
    top = keys.topk(topk, dim=1).values
    ids = (65535 - (top & 0xFFFF)).to(torch.int64)
    ok = (top >> 32) != _NEG_INF_S32
    uu = torch.where(ok, u.gather(1, ids), torch.zeros(1, dtype=torch.float32, device=u.device))
    tp = 1 << (topk - 1).bit_length()
    s = butterfly_sum(torch.nn.functional.pad(uu, (0, tp - topk)))
    anyok = ok.any(1, keepdim=True)
    rsf32 = torch.tensor(rsf, dtype=torch.float32, device=u.device)
    if renorm:
        scale = rsf32 / (s + torch.tensor(1e-20, dtype=torch.float32, device=u.device))
        w = torch.where(ok, uu * scale[:, None], torch.zeros_like(uu))
        w = torch.where(anyok, w, torch.full_like(w, 1.0 / topk))
    else:
        w = torch.where(ok, uu * rsf32, torch.zeros_like(uu))
    return w, ids.to(torch.int32)


# --- the kernel ------------------------------------------------------------------------
if triton is not None:
    _KMIN = tl.constexpr(KMIN)

    @triton.jit
    def _keys(lg, bias, col):
        """Selection keys [BM, BN] (see selection_keys)."""
        ok = (tl.abs(lg) < float("inf")) & (tl.abs(bias) < float("inf"))[None, :]
        u = 0.5 * libdevice.tanh(0.5 * lg) + 0.5
        b = u + bias[None, :]
        sel = tl.where(ok, tl.where(b == 0.0, 0.0, b), float("-inf"))
        bits = tl.cast(sel, tl.int32, bitcast=True)
        s32 = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        return (s32.to(tl.int64) << 32) | (65535 - col).to(tl.int64)[None, :]

    @triton.jit
    def _top_rows(key, BM: tl.constexpr, TOPK: tl.constexpr):
        """[BM, C] unique keys -> [BM, TOPK], largest first."""
        cols = tl.arange(0, TOPK)
        out = tl.full((BM, TOPK), _KMIN, tl.int64)
        for j in tl.static_range(TOPK):
            kmax = tl.max(key, axis=1)
            out = tl.where(cols[None, :] == j, kmax[:, None], out)
            key = tl.where(key == kmax[:, None], _KMIN, key)
        return out

    @triton.jit
    def _half(s, BM: tl.constexpr, W: tl.constexpr):
        """[BM, W] -> [BM, W/2]: lane l + lane l + W/2 (one shfl_xor step)."""
        lo, hi = tl.split(tl.permute(tl.reshape(s, (BM, 2, W // 2)), (0, 2, 1)))
        return lo + hi

    @triton.jit
    def _finish(tot, offs_m, b_ptr, lg_ptr, cand_ptr, msem_ptr, tw_ptr, id_ptr, M, rsf, pid_n,
                N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, TOPK: tl.constexpr,
                RENORM: tl.constexpr):
        """This (M, N) tile's logits are final in `tot`: keep its TOPK candidates per token;
        the M tile's last N tile merges them into topk weights / ids."""
        NT: tl.constexpr = N // BN
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_ok = offs_m < M
        jt = tl.arange(0, TOPK)
        loc = _top_rows(_keys(tot, tl.load(b_ptr + offs_n), offs_n), BM, TOPK)
        tl.store(cand_ptr + ((pid_n * M + offs_m) * TOPK)[:, None] + jt[None, :], loc,
                 mask=m_ok[:, None])
        tl.debug_barrier()
        cnt = tl.atomic_add(msem_ptr, 1, sem="acq_rel", scope="gpu")
        if cnt == NT - 1:
            jc = tl.arange(0, NT * TOPK)
            c_ptrs = (cand_ptr + (((jc // TOPK) * M)[None, :] + offs_m[:, None]) * TOPK
                      + (jc % TOPK)[None, :])
            cand = tl.load(c_ptrs, mask=m_ok[:, None], other=_KMIN, cache_modifier=".cg")
            top = _top_rows(cand, BM, TOPK)
            ids = 65535 - (top & 0xFFFF).to(tl.int32)
            lsel = tl.load(lg_ptr + (offs_m * N)[:, None] + ids, mask=m_ok[:, None], other=0.0,
                           cache_modifier=".cg")
            bsel = tl.load(b_ptr + ids, mask=m_ok[:, None], other=0.0)
            ok = (tl.abs(lsel) < float("inf")) & (tl.abs(bsel) < float("inf"))
            u = tl.where(ok, 0.5 * libdevice.tanh(0.5 * lsel) + 0.5, 0.0)
            s = u
            if TOPK >= 16:
                s = _half(s, BM, 16)
            if TOPK >= 8:
                s = _half(s, BM, 8)
            if TOPK >= 4:
                s = _half(s, BM, 4)
            if TOPK >= 2:
                lo, hi = tl.split(s)
                s = lo + hi
            else:
                s = tl.reshape(s, (BM,))
            anyok = tl.max(ok.to(tl.int32), axis=1) > 0
            if RENORM:
                scale = tl.math.div_rn(rsf + tl.zeros_like(s), s + 1e-20)
                w = tl.where(ok, u * scale[:, None], 0.0)
                w = tl.where(anyok[:, None], w, 1.0 / TOPK)
            else:
                w = tl.where(ok, u * rsf, 0.0)
            o = (offs_m * TOPK)[:, None] + jt[None, :]
            tl.store(tw_ptr + o, w, mask=m_ok[:, None])
            tl.store(id_ptr + o, ids, mask=m_ok[:, None])
            tl.atomic_xchg(msem_ptr, 0)

    @triton.jit(do_not_specialize=["M", "kb_split", "n_split"])
    def _router_kernel(x_ptr, w_ptr, b_ptr, lg_ptr, ws_ptr, cand_ptr, sem_ptr, tw_ptr, id_ptr,
                       M, K, stride_xm, kb_split, n_split, rsf,
                       N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       MMA: tl.constexpr, FUSE: tl.constexpr, LOGITS_IN: tl.constexpr,
                       TOPK: tl.constexpr, RENORM: tl.constexpr):
        NT: tl.constexpr = N // BN
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_m = tl.program_id(2)
        mt = tl.num_programs(2)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_ok = offs_m < M
        msem = sem_ptr + mt * NT + pid_m          # merge counter of this M tile
        lg_ptrs = lg_ptr + (offs_m * N)[:, None] + offs_n[None, :]
        if LOGITS_IN:  # oracle mode: selection only, on given logits (one K split)
            tot = tl.load(lg_ptrs, mask=m_ok[:, None], other=0.0)
            if FUSE:
                _finish(tot, offs_m, b_ptr, lg_ptr, cand_ptr, msem, tw_ptr, id_ptr, M, rsf,
                        pid_n, N, BM, BN, TOPK, RENORM)
        else:
            k_lo = pid_k * kb_split * BK
            k_hi = tl.minimum(k_lo + kb_split * BK, K)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            if MMA:
                x_ptrs = x_ptr + (offs_m * stride_xm)[:, None] + (k_lo + tl.arange(0, BK))[None, :]
                w_ptrs = w_ptr + (offs_n * K)[:, None] + (k_lo + tl.arange(0, BK))[None, :]
                for _k0 in tl.range(k_lo, k_hi, BK):
                    xt = tl.load(x_ptrs, mask=m_ok[:, None], other=0.0)
                    wt = tl.load(w_ptrs, eviction_policy="evict_first")
                    acc = tl.dot(xt, tl.trans(wt), acc)
                    x_ptrs += BK
                    w_ptrs += BK
            else:  # [BK, BN] weight tiles in load layout; the next one loads during this one
                rows = tl.arange(0, BM)
                xk = tl.arange(0, BK)
                w_ptrs = w_ptr + (offs_n * K)[None, :] + (k_lo + xk)[:, None]
                w_nx = tl.load(w_ptrs, eviction_policy="evict_first")
                for k0 in tl.range(k_lo, k_hi, BK):
                    wf = w_nx.to(tl.float32)
                    w_ptrs += BK
                    w_nx = tl.load(w_ptrs, mask=k0 + BK < k_hi, other=0.0,
                                   eviction_policy="evict_first")
                    for mm in tl.static_range(BM):
                        r = pid_m * BM + mm
                        xr = tl.load(x_ptr + r * stride_xm + k0 + xk, mask=r < M, other=0.0)
                        part = tl.sum(wf * xr.to(tl.float32)[:, None], axis=0)
                        acc = tl.where(rows[:, None] == mm, acc + part[None, :], acc)
            if n_split == 1:
                tl.store(lg_ptrs, acc, mask=m_ok[:, None])
                if FUSE:
                    _finish(acc, offs_m, b_ptr, lg_ptr, cand_ptr, msem, tw_ptr, id_ptr, M,
                            rsf, pid_n, N, BM, BN, TOPK, RENORM)
            else:
                p_ptrs = ws_ptr + ((pid_k * M + offs_m) * N)[:, None] + offs_n[None, :]
                tl.store(p_ptrs, acc, mask=m_ok[:, None])
                tl.debug_barrier()  # every warp's partial is issued before the release
                tile = sem_ptr + pid_m * NT + pid_n
                cnt = tl.atomic_add(tile, 1, sem="acq_rel", scope="gpu")
                if cnt == n_split - 1:  # last split of this tile: sum in split order
                    tot = tl.zeros((BM, BN), dtype=tl.float32)
                    for j in tl.range(0, n_split):
                        q_ptrs = ws_ptr + ((j * M + offs_m) * N)[:, None] + offs_n[None, :]
                        tot += tl.load(q_ptrs, mask=m_ok[:, None], other=0.0,
                                       cache_modifier=".cg")
                    tl.store(lg_ptrs, tot, mask=m_ok[:, None])
                    tl.atomic_xchg(tile, 0)
                    if FUSE:
                        _finish(tot, offs_m, b_ptr, lg_ptr, cand_ptr, msem, tw_ptr, id_ptr, M,
                                rsf, pid_n, N, BM, BN, TOPK, RENORM)


def _dummy(dev):
    return torch.empty(1, dtype=torch.int64, device=dev)


def run(x, w, bias, plan: Plan, sem, *, fuse: bool, topk: int = 8, renorm: bool = True,
        rsf: float = 1.0, logits_in=None):
    """x [M, K] bf16 contiguous, w [N, K] bf16, bias [N] fp32 -> (logits fp32 [M, N],
    weights fp32 [M, topk] | None, ids int32 [M, topk] | None). logits_in: selection only
    on those logits (one K split). sem: zeroed int32 counters (>= MT * NT + MT slots)."""
    n, k = w.shape
    src = logits_in if logits_in is not None else x
    m, dev = src.shape[0], src.device
    lg = logits_in if logits_in is not None else torch.empty((m, n), dtype=torch.float32,
                                                             device=dev)
    nt, mt = n // plan.bn, _cdiv(m, plan.bm)
    splits = 1 if logits_in is not None else plan.n_split
    ws = (torch.empty(splits * m * n, dtype=torch.float32, device=dev) if splits > 1
          else _dummy(dev))
    if fuse:
        cand = torch.empty(nt * m * topk, dtype=torch.int64, device=dev)
        tw = torch.empty((m, topk), dtype=torch.float32, device=dev)
        ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    else:
        cand = tw = ids = _dummy(dev)
    _router_kernel[(splits, nt, mt)](
        src if logits_in is None else lg, w, bias, lg, ws, cand, sem, tw, ids,
        m, k, k, plan.kb_split, splits, float(rsf),
        N=n, BM=plan.bm, BN=plan.bn, BK=plan.bk, MMA=plan.mma, FUSE=fuse,
        LOGITS_IN=logits_in is not None, TOPK=topk, RENORM=renorm,
        num_warps=plan.warps, num_stages=plan.stages)
    return lg, (tw if fuse else None), (ids if fuse else None)


# --- checks ----------------------------------------------------------------------------
def gemv_rel(lg, x, w) -> float:
    ref = x.double() @ w.double().t()
    return float((lg.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def _stock_select(lg, bias, topk, renorm, rsf):
    import vllm._custom_ops as ops
    return ops.grouped_topk(lg, 1, 1, topk, renorm, rsf, bias, 1)


def _same(tw, ids, rw, rids) -> bool:
    return bool(torch.equal(ids, rids.to(torch.int32))
                and torch.equal(tw.view(torch.int32), rw.float().view(torch.int32)))


def adversarial_logits(n: int, dev, seed: int = 0) -> list:
    """(logits [32, n], bias [n]) pairs: ties within and across tiles, +-0, NaN / inf,
    no or few finite selections, saturation, tiny values, random."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    f32 = torch.float32
    lg = torch.randn(32, n, generator=g, dtype=f32)
    lg[0] = 0.25                                             # every expert tied
    lg[1] = torch.randint(-1, 2, (n,), generator=g).to(f32)  # three distinct values
    lg[2, ::7] = float("nan")
    lg[2, 3::11] = float("inf")
    lg[2, 5::13] = float("-inf")
    lg[3] = float("nan")                                     # no finite selection
    lg[4] = 0.0
    lg[4, ::2] = -0.0
    lg[5] = torch.randn(n, generator=g, dtype=f32) * 1e30    # saturated sigmoid
    lg[6, : n - 3] = float("nan")                            # fewer finite than topk
    lg[7] = torch.randn(n, generator=g, dtype=f32) * 1e-30
    b_real = torch.randn(n, generator=g, dtype=f32) * 1e-2
    b_tie = torch.zeros(n, dtype=f32)
    b_tie[1::2] = -0.5                                       # sigmoid(0) - 0.5 = +-0
    b_bad = b_real.clone()
    b_bad[::17] = float("nan")
    return [(lg.to(dev), b.to(dev)) for b in (b_real, b_tie, b_bad)]


def check_selection(bn: int, n: int, dev, sem, topk, renorm, rsf, select=None) -> str | None:
    """None when our fused selection (tile width bn) == `select` (default ops.grouped_topk)
    bit for bit on the adversarial logits, else the first difference."""
    select = select or _stock_select
    plan = Plan(False, 32, bn, 128, 1, 1)
    w_dummy = torch.empty((n, 128), dtype=torch.bfloat16, device=dev)
    for i, (lg, b) in enumerate(adversarial_logits(n, dev)):
        _, tw, ids = run(None, w_dummy, b, plan, sem, fuse=True, topk=topk, renorm=renorm,
                         rsf=rsf, logits_in=lg)
        rw, rids = select(lg, b, topk, renorm, rsf)
        if not _same(tw, ids, rw, rids):
            bad = (ids != rids.to(torch.int32)).any(1) | (
                tw.view(torch.int32) != rw.float().view(torch.int32)).any(1)
            r = int(bad.nonzero()[0])
            return (f"set {i} row {r}: ours ids {ids[r].tolist()} w {tw[r].tolist()} vs "
                    f"ids {rids[r].tolist()} w {rw[r].tolist()}")
    return None


def check_layer(w, bias, route: "Route", dev, sem, topk, renorm, rsf, ms=(1, 6, 32, 96),
                select=None) -> float:
    """Our routed path on this layer's real weights at each M: GEMV rel vs fp64 <= TOL,
    fused selection == `select` (ops.grouped_topk) on our logits. Worst rel; raises."""
    select = select or _stock_select
    worst = 0.0
    gen = torch.Generator(device=dev).manual_seed(int(w.shape[1]) + 17)
    for m in ms:
        path, plan = route.lookup(m)
        if path == "stock":
            continue
        x = torch.randn(m, w.shape[1], device=dev, generator=gen, dtype=torch.float32)
        x = x.to(torch.bfloat16)
        lg, tw, ids = run(x, w, bias, plan, sem, fuse=path == "fused", topk=topk,
                          renorm=renorm, rsf=rsf)
        rel = gemv_rel(lg, x, w)
        worst = max(worst, rel)
        if not rel <= TOL:
            raise RuntimeError(f"router GEMV {plan.tag()} M={m}: rel {rel:.3e} > {TOL}")
        if path == "fused" and not _same(tw, ids, *select(lg, bias, topk, renorm, rsf)):
            raise RuntimeError(f"fused selection {plan.tag()} M={m} differs from "
                               "ops.grouped_topk on the same logits")
    return worst


def check_graphs(w, bias, route: "Route", dev, sem, topk, renorm, rsf) -> None:
    """Each routed plan replayed from a CUDA graph == eager, bit for bit."""
    for m, (path, plan) in zip(route.ms, route.choices):
        if path == "stock":
            continue
        x = torch.randn(m, w.shape[1], device=dev, dtype=torch.float32).to(torch.bfloat16)
        fuse = path == "fused"
        ref = run(x, w, bias, plan, sem, fuse=fuse, topk=topk, renorm=renorm, rsf=rsf)
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            run(x, w, bias, plan, sem, fuse=fuse, topk=topk, renorm=renorm, rsf=rsf)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
                got = run(x, w, bias, plan, sem, fuse=fuse, topk=topk, renorm=renorm, rsf=rsf)
        torch.cuda.current_stream(dev).wait_stream(s)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize(dev)
        for a, b in zip(ref, got):
            if a is not None and not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                raise RuntimeError(f"router {plan.tag()} M={m}: graph replay != eager")


# --- routing decision ----------------------------------------------------------------
class Route:
    """Per-call M -> ("fused" | "gemv" | "stock", plan): first bucket >= M."""

    def __init__(self, ms, choices):
        self.ms, self.choices = tuple(ms), tuple(choices)

    def lookup(self, m: int):
        i = bisect.bisect_left(self.ms, m)
        return self.choices[i] if i < len(self.ms) else ("stock", None)

    def line(self) -> str:
        return " | ".join(f"M<={m} {c[0]}{'/' + c[1].tag() if c[1] else ''}"
                          for m, c in zip(self.ms, self.choices))


def decide(ms, best, row_us, md: str, fused_ok: bool) -> Route:
    """best {m: Plan | None}; row_us {m: {"stock", "gemv", "fused": us}} (inf = n/a) ->
    Route. "1": fused wherever a plan works (a missing plan is an error); otherwise the
    fastest allowed path per bucket."""
    choices = []
    for m in ms:
        p, row = best[m], row_us[m]
        if md == "1":
            if p is None or not fused_ok or not math.isfinite(row["fused"]):
                raise RuntimeError(f"{GATE}=1: no working fused plan at M<={m}")
            choices.append(("fused", p))
            continue
        opts = [("stock", row["stock"])]
        if p is not None:
            opts.append(("gemv", row["gemv"]))
            if fused_ok and md == "auto":
                opts.append(("fused", row["fused"]))
        path = min(opts, key=lambda o: o[1])[0]
        choices.append((path, p if path != "stock" else None))
    return Route(ms, choices)


def _graph_us(fn, dev, flush):
    from suffix_hybrid.kernels.nvfp4_gemm import _graph_us as g
    return g(lambda _s: fn(), dev, REPS, flush)


def _no_reduce(vals):
    return vals


def tp_max_reduce(vals: list) -> list:
    """Element-wise max over the TP group (CPU group), so every rank decides the same."""
    from vllm.distributed import get_tp_group
    grp = get_tp_group()
    if grp.world_size == 1:
        return vals
    t = torch.tensor(vals, dtype=torch.float64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=grp.cpu_group)
    return t.tolist()


def tune(w, bias, dev, sem, *, md: str, topk, renorm, rsf, fused_ok: bool,
         reduce=_no_reduce, log=_say) -> tuple:
    """Route for this router shape (all GLM gates share it), timed cold-L2 from CUDA graphs:
    round 1 every correct plan's GEMV per bucket (the fastest wins), round 2 vLLM's
    GateLinear GEMM + ops.grouped_topk vs our GEMV + ops.grouped_topk vs fused. Timings are
    `reduce`d (TP max) before each decision. Returns (Route, {m: {path: us}})."""
    props = torch.cuda.get_device_properties(dev)
    flush_buf = torch.empty(max(2 * props.L2_cache_size, 64 << 20), dtype=torch.uint8,
                            device=dev)
    flush = flush_buf.zero_
    n, k = w.shape
    inf = float("inf")
    ms = [b for b in BUCKETS if b <= max_m()]
    gen = torch.Generator(device=dev).manual_seed(k)
    xs = {m: torch.randn(m, k, device=dev, generator=gen, dtype=torch.float32).to(torch.bfloat16)
          for m in ms}
    cands = {m: plans(n, k, m, props.multi_processor_count) for m in ms}
    broken, t1 = {}, []
    for m in ms:
        x = xs[m]
        for p in cands[m]:
            key = (p.mma, p.bm, p.bn, p.bk, p.stages)
            if key in broken:
                t1.append(inf)
                continue
            try:
                lg, _, _ = run(x, w, bias, p, sem, fuse=False)
                torch.cuda.synchronize(dev)
            except Exception as exc:  # noqa: BLE001 - compile / resource limit: skip config
                torch.cuda.synchronize(dev)
                broken[key] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160]
                t1.append(inf)
                continue
            rel = gemv_rel(lg, x, w)
            if not rel <= TOL:
                log(f"WRONG {p.tag()} M={m}: rel {rel:.2e} > {TOL} -- excluded")
                t1.append(inf)
                continue
            t1.append(_graph_us(lambda p=p, x=x: run(x, w, bias, p, sem, fuse=False), dev,
                                flush))
    t1 = reduce(t1)
    best, i = {}, 0
    for m in ms:
        row = t1[i:i + len(cands[m])]
        i += len(cands[m])
        j = min(range(len(row)), key=lambda j: row[j]) if row else None
        best[m] = cands[m][j] if j is not None and math.isfinite(row[j]) else None
    t2 = []
    for m in ms:
        x, p = xs[m], best[m]
        t2.append(_graph_us(lambda x=x: _stock_select(torch.mm(x, w.t(), out_dtype=torch.float32),
                                                      bias, topk, renorm, rsf), dev, flush))
        t2.append(inf if p is None else _graph_us(
            lambda x=x, p=p: _stock_select(run(x, w, bias, p, sem, fuse=False)[0], bias, topk,
                                           renorm, rsf), dev, flush))
        t2.append(inf if p is None or not fused_ok else _graph_us(
            lambda x=x, p=p: run(x, w, bias, p, sem, fuse=True, topk=topk, renorm=renorm,
                                 rsf=rsf), dev, flush))
    t2 = reduce(t2)
    rows = {m: dict(zip(("stock", "gemv", "fused"), t2[3 * i:3 * i + 3])) for i, m in enumerate(ms)}
    for key, why in broken.items():
        log(f"plan {key} unavailable: {why}")
    del flush_buf
    return decide(ms, best, rows, md, fused_ok), rows


# --- vLLM integration ------------------------------------------------------------------
@dataclass
class _Gate:
    route: Route
    dev: object
    topk: int
    renorm: bool
    rsf: float
    sig: tuple
    max_m: int


def eligible(gate, router, use_fused_grouped_topk: bool = True) -> str | None:
    """None when this (GateLinear, router) pair is GLM's single-group noaux_tc on BF16."""
    w = getattr(gate, "weight", None)
    if type(router).__name__ != "GroupedTopKRouter":
        return f"router {type(router).__name__} is not GroupedTopKRouter"
    if (w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.device.type != "cuda"
            or not w.is_contiguous()):
        return "gate weight is not a contiguous 2-D CUDA bf16 tensor"
    if getattr(gate, "bias", None) is not None or not getattr(gate, "is_unquantized", False):
        return "gate has a bias or is quantized"
    if gate.out_dtype != torch.float32:
        return f"router logits dtype {gate.out_dtype} (want fp32)"
    if router.num_expert_group != 1 or router.topk_group != 1:
        return f"grouping {router.num_expert_group}/{router.topk_group} (want 1/1)"
    b = router.e_score_correction_bias
    if router.scoring_func != "sigmoid" or b is None or b.dtype != torch.float32:
        return "not sigmoid + an fp32 e_score_correction_bias"
    tk = router.top_k
    if tk & (tk - 1) or not 1 <= tk <= 16:
        return f"top_k {tk} (want a power of two <= 16)"
    if router.eplb_state is not None or getattr(router, "num_fused_shared_experts", 0):
        return "EPLB / fused shared experts in the router"
    if not use_fused_grouped_topk:
        return "VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 (stock selection is the torch path)"
    n, k = w.shape
    if n != 256 or tk != 8 or k % 512:
        return f"{n} experts x top-{tk}, K {k} (fused selection proven for 256 x top-8)"
    return None


def setup_model(model) -> None:
    """After a model loads: every eligible gate gets the route (tuned once per shape and
    process, identical on every TP rank) after the fail-closed checks on its real weights."""
    from fp8_dense_patch.ours import ensure_sems
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    _STATE["setup_ran"] = True
    md = mode()
    pairs, skipped = [], {}
    for name, mod in model.named_modules():
        gate, router = getattr(mod, "gate", None), getattr(mod, "router", None)
        if not isinstance(gate, GateLinear) or router is None:
            continue
        why = eligible(gate, router, envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK)
        if why is None:
            pairs.append((name, gate, router))
        else:
            skipped[why] = skipped.get(why, 0) + 1
    for why, c in skipped.items():
        _say(f"{c} router(s) stay on vLLM's path: {why}")
    if not pairs:
        if skipped:
            raise SystemExit(f"{MARK} {GATE}={md} but no router is eligible")
        return
    t0 = time.monotonic()
    dev = pairs[0][1].weight.device
    ensure_sems(dev)
    sem = torch.zeros(SEMS, dtype=torch.int32, device=dev)
    _, g0, r0 = pairs[0]
    topk, renorm, rsf = r0.top_k, bool(r0.renormalize), float(r0.routed_scaling_factor)
    n, k = g0.weight.shape
    shape = (n, k, topk, renorm, rsf)
    route = _STATE["routes"].get(shape)
    if route is None:
        why = None
        if md != "gemv":
            for bn in sorted({p.bn for m in (6, 64) for p in plans(n, k, m)}):
                why = why or check_selection(bn, n, dev, sem, topk, renorm, rsf)
        fused_ok = md != "gemv" and tp_max_reduce([0.0 if why is None else 1.0])[0] == 0.0
        if md == "1" and not fused_ok:
            raise SystemExit(f"{MARK} fused selection is not bit-identical to ops.grouped_topk "
                             f"({why or 'on another TP rank'}); {GATE}=auto or gemv serves our "
                             "GEMV with vLLM's selection")
        if md == "auto" and not fused_ok:
            _say(f"fused selection OFF: not bit-identical to ops.grouped_topk "
                 f"({why or 'on another TP rank'})")
        route, rows = tune(g0.weight, r0.e_score_correction_bias, dev, sem, md=md, topk=topk,
                           renorm=renorm, rsf=rsf, fused_ok=fused_ok, reduce=tp_max_reduce)
        check_graphs(g0.weight, r0.e_score_correction_bias, route, dev, sem, topk, renorm, rsf)
        _STATE["routes"][shape] = route
        _say(f"route {n}x{k} top{topk} [{time.monotonic() - t0:.1f} s]: {route.line()}")
        for m, row in rows.items():
            _say(f"  M={m}: " + ", ".join(f"{p} {us:.2f}us" for p, us in row.items()))
    worst = 0.0
    for _name, gate, router in pairs:
        worst = max(worst, check_layer(gate.weight, router.e_score_correction_bias, route, dev,
                                       sem, topk, renorm, rsf))
        gate._sfx_bias = router.e_score_correction_bias
        gate._sfx_router = _Gate(route, dev, topk, renorm, rsf,
                                 (topk, renorm, 1, 1, "sigmoid", rsf), max(route.ms))
    _STATE["layers"] += len(pairs)
    _say(f"ACTIVE: {len(pairs)} router(s) ({_STATE['layers']} in this process), paths "
         f"{sorted({c[0] for c in route.choices})}, GEMV max rel vs fp64 {worst:.2e} "
         f"(bound {TOL}), checks {time.monotonic() - t0:.1f} s")


def _forward(orig):
    from fp8_dense_patch.ours import sem_for

    def forward(self, x):
        st = getattr(self, "_sfx_router", None)
        if st is None:
            if not _STATE["setup_ran"]:
                raise SystemExit(f"{MARK} {GATE}={mode()} armed but no model was set up "
                                 "(BaseModelLoader.load_model not reached): refusing to serve")
            return orig(self, x)
        if (x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous()
                or not 0 < x.shape[0] <= st.max_m):
            return orig(self, x)
        path, plan = st.route.lookup(x.shape[0])
        if path == "stock":
            return orig(self, x)
        sem = sem_for(st.dev, torch.cuda.current_stream(st.dev).cuda_stream)
        if sem is None:  # no free counters for this stream: vLLM's path
            return orig(self, x)
        fuse = path == "fused"
        lg, tw, ids = run(x, self.weight, self._sfx_bias, plan, sem, fuse=fuse, topk=st.topk,
                          renorm=st.renorm, rsf=st.rsf)
        if fuse:
            _STASH[0] = (lg, tw, ids, st.sig, self._sfx_bias.data_ptr())
        return self._return(lg)
    forward.__qualname__ = "GateLinear.forward"
    return forward


def _grouped_topk(orig):
    def grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group=0,
                     topk_group=0, scoring_func="softmax", routed_scaling_factor=1.0,
                     e_score_correction_bias=None):
        hit, _STASH[0] = _STASH[0], None
        if (hit is not None and gating_output is hit[0] and e_score_correction_bias is not None
                and hit[3] == (topk, bool(renormalize), num_expert_group, topk_group,
                               scoring_func, float(routed_scaling_factor))
                and e_score_correction_bias.data_ptr() == hit[4]):
            return hit[1], hit[2]
        return orig(hidden_states=hidden_states, gating_output=gating_output, topk=topk,
                    renormalize=renormalize, num_expert_group=num_expert_group,
                    topk_group=topk_group, scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias)
    grouped_topk._sfx_router = True
    return grouped_topk


def _patch_gate(module) -> None:
    cls = module.GateLinear
    if not getattr(cls, "_sfx_router_patched", False):
        cls.forward = _forward(cls.forward)
        cls._sfx_router_patched = True


def _patch_topk(module) -> None:
    if not getattr(module.grouped_topk, "_sfx_router", False):
        module.grouped_topk = _grouped_topk(module.grouped_topk)


def _patch_loader(module) -> None:
    cls = module.BaseModelLoader
    if getattr(cls, "_sfx_router_patched", False):
        return
    orig = cls.load_model

    def load_model(self, *a, **kw):
        model = orig(self, *a, **kw)
        try:
            setup_model(model)
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"{MARK} {GATE}={mode()} load-time setup FAILED: {exc!r}") from exc
        return model

    cls.load_model = load_model
    cls._sfx_router_patched = True


PATCHES = {GATE_LINEAR: _patch_gate, TOPK_ROUTER: _patch_topk, BASE_LOADER: _patch_loader}


def install_post_import_hook() -> None:
    """sitecustomize entry (gate on): patch the three vLLM modules right after they import."""
    if not mode():
        return
    max_m()  # validate early

    def apply(name, module):
        try:
            PATCHES[name](module)
        except Exception as exc:
            raise SystemExit(f"{MARK} enabled but installation FAILED on {name}: "
                             f"{exc!r}") from exc

    todo = {name for name in PATCHES if name not in sys.modules}
    for name in set(PATCHES) - todo:
        apply(name, sys.modules[name])

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname not in todo:
                return None
            todo.discard(fullname)
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if todo:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                apply(fullname, module)

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    if todo:
        sys.meta_path.insert(0, _Finder())
    _say(f"armed ({GATE}={mode()}, M <= {max_m()})")


# --- boot-gate bench -------------------------------------------------------------------
GLM_ROUTER = (256, 6144)
# Other BF16 shapes of GLM 5.3 decode per TP8 rank (N, K) and the MLA head bmms (B, N, K):
# timed vs cuBLAS through fp8_dense_patch.ours' kernel with BF16 weights, not routed.
OTHER_LINEAR = (("indexer.wk", 128, 6144), ("indexer.weights_proj", 32, 6144),
                ("shared.gate_up", 512, 6144), ("shared.down", 6144, 256))
OTHER_BMM = (("W_UK_T", 8, 512, 192), ("W_UV", 8, 256, 512))
STEP_MS = (6, 32, 96)   # c1 (k=5) / c8 (k=3) / c32 (k=2) verify tokens; 80 router calls each


def _kernels_of(fn) -> str:
    """Kernel names one call of fn launches (torch.profiler), for the module mapping."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return " + ".join(dict.fromkeys(n[:72] for n in names)) or "-"


def bench() -> int:
    """vLLM (cuBLAS fp32-out GEMM + ops.grouped_topk) vs ours (GEMV + ops.grouped_topk,
    fused) per M on random GLM-shaped weights, checks, cuBLAS kernel names, the other BF16
    shapes vs cuBLAS, VERDICT. Exit 1 on any numerics violation."""
    from vllm.platforms import current_platform
    current_platform.import_kernels()  # torch.ops._moe_C: the gate runs before vLLM loads
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "sm120"))
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    n, k = GLM_ROUTER
    g = torch.Generator(device=dev).manual_seed(5)
    w = (torch.randn(n, k, device=dev, generator=g, dtype=torch.float32) * 0.02).bfloat16()
    bias = torch.randn(n, device=dev, generator=g, dtype=torch.float32) * 1e-2
    topk, renorm, rsf = 8, True, 2.5
    sem = torch.zeros(SEMS, dtype=torch.int32, device=dev)
    say = lambda s: print(f"{MARK} {s}", flush=True)  # noqa: E731
    why = None
    for bn in (16, 32, 64):
        why = why or check_selection(bn, n, dev, sem, topk, renorm, rsf)
    say("fused selection vs ops.grouped_topk (adversarial logits, BN 16/32/64): "
        + ("bit-identical" if why is None else "DIFFERS: " + why))
    os.environ.setdefault(MAX_M_ENV, str(BUCKETS[-1]))
    route, rows = tune(w, bias, dev, sem, md="auto", topk=topk, renorm=renorm, rsf=rsf,
                       fused_ok=why is None, log=say)
    bad = 0
    try:
        check_graphs(w, bias, route, dev, sem, topk, renorm, rsf)
        check_layer(w, bias, route, dev, sem, topk, renorm, rsf, ms=route.ms)
    except RuntimeError as exc:
        say(f"CHECK FAIL: {exc}")
        bad += 1
    for m in (1, 6, 32, 96):
        xm = torch.randn(m, k, device=dev, dtype=torch.float32).bfloat16()
        say(f"vLLM router kernels M={m}: " + _kernels_of(
            lambda: _stock_select(torch.mm(xm, w.t(), out_dtype=torch.float32), bias, topk,
                                  renorm, rsf)))
    for m in route.ms:
        say(f"router M={m:3d}: " + " | ".join(f"{p} {us:6.2f} us" for p, us in rows[m].items())
            + f" -> {route.lookup(m)[0]}")
    for m in STEP_MS:
        if m in rows:
            best = min(rows[m].values())
            say(f"per step at M={m} (80 router calls): vLLM {rows[m]['stock'] * 80 / 1e3:.3f} ms "
                f"-> ours {best * 80 / 1e3:.3f} ms (saves "
                f"{(rows[m]['stock'] - best) * 80 / 1e3:.3f} ms)")
    bad += _bench_other(dev, props, say)
    fast = all(min(rows[m].values()) < rows[m]["stock"] for m in route.ms)
    say(f"VERDICT: router ours {'faster at every M' if fast else 'NOT faster everywhere'} "
        f"(route {route.line()}); fused selection {'exact' if why is None else 'OFF'}; "
        f"numerics {'OK' if not bad else 'FAIL'}")
    return 1 if bad else 0


def _bench_other(dev, props, say) -> int:
    """Other BF16 shapes: fp8_dense_patch.ours' tuned kernel with BF16 weights (channel
    scale 1) vs cuBLAS, per M; its a8 configs do not apply to BF16 weights and are ignored."""
    from fp8_dense_patch import ours
    flush_buf = torch.empty(max(2 * props.L2_cache_size, 64 << 20), dtype=torch.uint8,
                            device=dev)
    sem = torch.zeros(ours.SEM_SLOTS, dtype=torch.int32, device=dev)
    bad = 0
    ms = (1, 6, 32, 96)
    for name, *shp in OTHER_LINEAR + OTHER_BMM:
        b, nn, kk = (1, *shp) if len(shp) == 2 else shp
        lead = (b,) if b > 1 else ()
        wq = (torch.randn(*lead, nn, kk, device=dev, dtype=torch.float32) * 0.02).bfloat16()
        wt = ours.Weights(False, b, nn, kk, wq, torch.ones(b * nn, dtype=torch.float32,
                                                           device=dev))

        def cublas(x, wq=wq, b=b):
            return torch.bmm(x, wq.transpose(-1, -2)) if b > 1 else x @ wq.t()
        table, wrong = ours.tune_shape(wt, ms, dev=dev, flush=flush_buf.zero_, sem=sem,
                                       nsm=props.multi_processor_count,
                                       extra={"cublas": cublas}, log=lambda s: None)
        bad += sum(1 for line in wrong if " a8" not in line and "a8mma" not in line)
        x6 = torch.randn(*lead, 6, kk, device=dev, dtype=torch.float32).bfloat16()
        cells = []
        for m in ms:
            row = table[m]
            ok = [(us, p) for us, p in row["ours"] if not p.cfg.a8]
            c = row["base"].get("cublas")
            cells.append(f"M={m} ours {ok[0][0]:.1f}" if ok else f"M={m} ours -")
            cells[-1] += f"/cuBLAS {c:.1f} us" if c is not None else "/cuBLAS -"
        say(f"other {name} {b}x{nn}x{kk} (cuBLAS M=6: {_kernels_of(lambda: cublas(x6))}): "
            + " | ".join(cells))
    del flush_buf
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        raise SystemExit(bench())
    raise SystemExit("usage: python -m suffix_hybrid.kernels.glm_router bench")
