# SPDX-License-Identifier: Apache-2.0
"""Our own small-M dense GEMM for SM120 (SUFFIX_FP8_DENSE_BACKEND ours / best / ours_fp4):
BF16 activations x FP8 e4m3 per-output-channel weights (W8A16) or NVFP4 weights (W4A16:
packed e2m1 + linear e4m3 block-16 scales + per-tensor global), dequantized in registers,
no activation-quant kernel; plus W8A8 that quantizes the activation INSIDE the GEMM (per
row and BK chunk, e4m3, finer than cutlass' per-token) for the FP8 MMA.

One Triton kernel (Triton ships its own ptxas: no PTX JIT on the driver-580 pods),
constexpr-selected families:
  gemv  M <= 8: CUDA-core FMA, BM rows per program, the next weight tile loaded while
        the current one is reduced, fp32 accumulate;
  mma   tl.dot on the dequantized tile (BM 16..128), cp.async pipeline (num_stages);
        W4 interleaves the decoded even/odd weights (join) or splits x into even/odd
        columns (xs: two dots);
  a8    FP8 x FP8 tl.dot, activation quantized per (row, BK chunk) in registers.
e2m1 decodes exactly through fp16 bits (sign << 15 | mag << 9, x 2^14); e2m1 x e4m3 has
<= 6 significant bits, so the dequantized BF16 weight is exact; 1/global folds into the
epilogue, the FP8 channel scale too.

Split-K (grid: split, N tile, batch x M tile), deterministic: each split stores its
scaled fp32 partial; the LAST arriving CTA of a tile (per-tile counter, acq_rel gpu-scope
atomic after a CTA barrier, the CUTLASS semaphore pattern) sums the partials in split
order, writes the output and rearms the counter -- or two-stage (partials + a reduce
kernel), timed against it. Partials come per call from torch's stream-ordered caching
allocator (graph-private under capture: concurrent streams never share them); the tile
counters must be zero at launch, so N_SEMS zeroed arrays are allocated at load and bound
to streams on first use; a stream beyond them runs the same plan two-stage.

Routing: per (backend, N, K) shape, every candidate (tile x split) is checked against the
exact dequantized weight (a wrong one is excluded and reported) and timed at each M
bucket (its upper bound) from a CUDA graph with the L2 flushed before every replay
(median), next to cutlass + its per-token quant kernel and, for ``best``, Marlin W8A16;
the fastest wins the bucket. Above the last bucket: cutlass (ours / best) or the fastest
unsplit W4 tile at BIG_M (ours_fp4). Timings are cached as JSON (CACHE_ENV); startup
timing is bounded by BUDGET_ENV.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU CI: the planning / routing half stays importable
    triton = tl = None

NSM = 188  # RTX PRO 6000 Blackwell Server; the tuner reads the real count
BUCKETS = (1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
BIG_M = 1024        # ours_fp4 above the last bucket: unsplit tiles timed here
REPS = 9            # cold-L2 graph replays per timing (median)
OURS_BOUND = 1e-3   # rel. Frobenius vs the exact reference rounded to the output dtype
# a8 plans vs the exact emulation of their own e4m3 activation quantization: the e4m3 x e4m3
# tensor-core products add 1.5-2.2e-3 on sm_120 at K 2048/6144 (g14 boot re-verify; the same
# plans passed 1e-3 on the tuning batch). Same error class as vLLM's cutlass W8A8 path.
A8_BOUND = 4e-3


# a8 plans (e4m3 activation quantized in the GEMM) are off unless SUFFIX_FP8_DENSE_OURS_A8=1:
# on silicon they sat 1.5e-3..4.1e-3 off the exact emulation of their own quantization, varying
# by batch (g14 boots) — unexplained, so not served until it is (exact-activation plans are).
A8_ENV = "SUFFIX_FP8_DENSE_OURS_A8"


def a8_on() -> bool:
    return os.environ.get(A8_ENV, "0").strip() == "1"


def bound(cfg) -> float:
    return A8_BOUND if cfg is not None and cfg.a8 else OURS_BOUND
WS_CAP = 4 << 20    # fp32 partial elements per split launch (16 MiB, transient)
SEM_SLOTS = 8192    # tile counters per stream
N_SEMS = 8          # streams with counters: eager, vLLM's profiling + serving capture
                    # streams, the shared-expert aux stream, drafter, spares
SMEM_BUDGET = 80 << 10
RED_BN = 1024
BUDGET_ENV = "SUFFIX_FP8_DENSE_OURS_BUDGET_S"
BUDGET_S = 180.0
CACHE_ENV = "SUFFIX_FP8_DENSE_OURS_CACHE"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Tile menus; bk counts weight elements. Per shape: bk halves until it divides K (K = 192
# for the MLA heads), then the SM120 shared-memory budget filters the pipelined ones.
GEMV_TILES = {"w8": ((32, 256, 4), (16, 512, 4)), "w4": ((32, 256, 4), (16, 512, 4))}
MMA_TILES = {  # bm -> (bn, bk, warps, stages)
    16: ((64, 128, 4, 4), (32, 256, 4, 4), (128, 128, 4, 3), (64, 256, 4, 3)),
    32: ((64, 128, 4, 4), (128, 128, 4, 3), (32, 256, 4, 3)),
    64: ((64, 128, 4, 3), (128, 64, 4, 4), (128, 128, 8, 2)),
    128: ((128, 64, 8, 3), (64, 64, 4, 4)),
}
A8_TILES = {16: ((64, 128, 4, 4), (128, 128, 4, 3)), 32: ((64, 128, 4, 4), (128, 128, 4, 3)),
            64: ((64, 128, 4, 3), (128, 128, 8, 2)), 128: ((128, 64, 8, 3),)}

_STATE = {"tune_s": 0.0}


def _cdiv(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Cfg:
    mma: bool
    bm: int
    bn: int
    bk: int
    warps: int = 4
    stages: int = 1
    a8: bool = False
    xs: bool = False

    def tag(self) -> str:
        if not self.mma:
            return f"gemv{self.bm}x{self.bn}x{self.bk}w{self.warps}"
        kind = "a8" if self.a8 else "xs" if self.xs else ""
        return f"{kind}mma{self.bm}x{self.bn}x{self.bk}w{self.warps}s{self.stages}"


@dataclass(frozen=True)
class Plan:
    cfg: Cfg
    kb_split: int       # BK blocks per split
    n_split: int
    two: bool = False   # two-stage reduction instead of the last-CTA one

    def tag(self) -> str:
        return f"{self.cfg.tag()}/k{self.n_split}" + ("r" if self.two else "")

    def two_stage(self) -> "Plan":
        return Plan(self.cfg, self.kb_split, self.n_split, True)

    def ws(self, b: int, m: int, n: int) -> int:
        """fp32 partial elements this plan allocates per call at M = m."""
        return b * self.n_split * m * n if self.n_split > 1 else 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Plan":
        return cls(Cfg(**d["cfg"]), d["kb_split"], d["n_split"], d["two"])


@dataclass
class Weights:
    """Kernel view of one (optionally head-batched) weight. w8: fp8 [B?, N, K] + f32
    channel scales [B?, N]; w4: packed u8 [B?, N, K/2] (low nibble = even k) + e4m3
    block scales [B?, N, K/16] + f32 1/global [B]."""
    w4: bool
    b: int
    n: int
    k: int
    w: object
    s: object
    sf: object = None

    @classmethod
    def fp8(cls, q, s):
        return cls(False, q.shape[0] if q.dim() == 3 else 1, q.shape[-2], q.shape[-1], q,
                   s.float().contiguous())

    @classmethod
    def fp4(cls, wq, wsf, g):
        return cls(True, wq.shape[0] if wq.dim() == 3 else 1, wq.shape[-2], wq.shape[-1] * 2,
                   wq, (1.0 / g.float()).reshape(-1).contiguous(), wsf)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.s, self.sf)
                   if t is not None)


# --- candidates and routes (pure Python) -------------------------------------------
def _smem(cfg: Cfg, w4: bool) -> int:
    if not cfg.mma:
        return 0
    wb = cfg.bn * cfg.bk // 2 + cfg.bn * cfg.bk // 16 if w4 else cfg.bn * cfg.bk
    return cfg.stages * (cfg.bm * cfg.bk * 2 + wb)


def _fit_bk(bk: int, k: int) -> int:
    while bk >= 32 and k % bk:
        bk //= 2
    return bk if bk >= 32 else 0


def configs(kind: str, k: int, m: int, big: bool = False) -> list:
    """Kernel configs worth timing at M = m for weight kind 'w8' / 'w4'."""
    w4 = kind == "w4"
    raw = []
    if m <= 8 and not big:
        bm = 1 << (m - 1).bit_length()
        raw += [Cfg(False, bm, bn, bk, w) for bn, bk, w in GEMV_TILES[kind]]
    bms = ((128,) if big else (16,) if m <= 16 else (32,) if m <= 32 else (64,) if m <= 64
           else (64, 128))
    for bm in bms:
        for bn, bk, w, st in MMA_TILES[bm]:
            for xs in ((False, True) if w4 and bm <= 32 else (False,)):
                raw.append(Cfg(True, bm, bn, bk, w, st, xs=xs))
        if not w4 and a8_on():
            raw += [Cfg(True, bm, bn, bk, w, st, a8=True) for bn, bk, w, st in A8_TILES[bm]]
    out = []
    for c in raw:
        bk = _fit_bk(c.bk, k)
        c = Cfg(c.mma, c.bm, c.bn, bk, c.warps, c.stages, c.a8, c.xs)
        if bk and _smem(c, w4) <= SMEM_BUDGET and c not in out:
            out.append(c)
    return out


def plans(kind: str, b: int, n: int, k: int, m: int, nsm: int = NSM, big: bool = False) -> list:
    """Every (config, split) candidate at M = m: splits aiming at 1 / 2 / 4 waves of
    nsm CTAs (and unsplit), deduplicated by the effective split count."""
    out = []
    for cfg in configs(kind, k, m, big):
        kb = k // cfg.bk
        tiles = _cdiv(n, cfg.bn) * b * _cdiv(m, cfg.bm)
        want = {1} if big else {1} | {max(1, min(kb, round(c * nsm / tiles))) for c in (1, 2, 4)}
        seen = set()
        for s in sorted(want):
            kbs = _cdiv(kb, s)
            ns = _cdiv(kb, kbs)
            if ns in seen or (ns > 1 and (tiles > SEM_SLOTS or b * ns * m * n > WS_CAP)):
                continue
            seen.add(ns)
            out.append(Plan(cfg, kbs, ns))
    return out


def default_plan(kind: str, b: int, n: int, k: int, m: int, nsm: int = NSM) -> Plan:
    """Untimed guess (startup budget spent): the first MMA tile at ~2 waves."""
    cands = [p for p in plans(kind, b, n, k, m, nsm) if p.cfg.mma and not p.cfg.a8]
    first = [p for p in cands if p.cfg == cands[0].cfg]
    tiles = _cdiv(n, first[0].cfg.bn) * b * _cdiv(m, first[0].cfg.bm)
    return min(first, key=lambda p: abs(p.n_split * tiles - 2 * nsm))


def decide(table: dict, routable=()) -> dict:
    """{m: (path, plan | None)}: per bucket the fastest of our checked plans and the
    routable baselines (by name)."""
    out = {}
    for m, row in table.items():
        cands = [(us, ("ours", plan)) for us, plan in row["ours"]]
        cands += [(us, (name, None)) for name, us in row["base"].items() if name in routable]
        if not cands:
            raise RuntimeError(f"no working kernel at M={m}")
        out[m] = min(cands, key=lambda c: c[0])[1]
    return out


class Route:
    """Per-call M -> (path, plan): bucket = the first bound >= M; above the last: big."""

    def __init__(self, ms, choices, big):
        self.ms, self.choices, self.big = tuple(ms), tuple(choices), big
        self._memo = {}

    def lookup(self, m: int):
        c = self._memo.get(m)
        if c is None:
            i = bisect.bisect_left(self.ms, m)
            c = self._memo[m] = self.choices[i] if i < len(self.ms) else self.big
        return c

    def paths(self) -> set:
        return {p for p, _ in self.choices + (self.big,)}

    def ws(self, b: int, n: int) -> int:
        return max([0] + [plan.ws(b, m, n) for m, (p, plan) in zip(self.ms, self.choices)
                          if p == "ours"])

    def line(self, table: dict | None = None) -> str:
        def one(m, c):
            us = ""
            if table and m in table:
                hit = [u for u, p in table[m]["ours"] if p == c[1]] if c[0] == "ours" else \
                    [table[m]["base"].get(c[0])]
                us = f" {hit[0]:.1f}us" if hit and hit[0] is not None else ""
            return f"M<={m} {c[1].tag() if c[1] else c[0]}{us}"
        last = self.ms[-1]
        return " | ".join([one(m, c) for m, c in zip(self.ms, self.choices)]
                          + [f"M>{last} {self.big[1].tag() if self.big[1] else self.big[0]}"])


# --- split-K tile counters: one zeroed int32 array per (device, stream) ------------
_SEMS = {}   # (device index, stream handle) -> int32 [SEM_SLOTS]
_FREE = {}   # device index -> zeroed counters not yet bound to a stream
_DUMMY = {}  # device -> typed stand-ins for unsplit launches (one signature per config)


def _index(dev) -> int:
    if dev.index is not None:
        return dev.index
    return torch.cuda.current_device() if dev.type == "cuda" else -1


def ensure_sems(dev) -> None:
    """N_SEMS zeroed counter arrays on dev (load time: never during graph capture)."""
    i = _index(dev)
    if i not in _FREE:
        _FREE[i] = [torch.zeros(SEM_SLOTS, dtype=torch.int32, device=dev) for _ in range(N_SEMS)]


def sem_for(dev, stream: int):
    """This stream's counters (bound on first use), None once N_SEMS are taken."""
    key = (_index(dev), stream)
    sem = _SEMS.get(key)
    if sem is None:
        free = _FREE.get(key[0])
        if not free:
            return None
        sem = _SEMS[key] = free.pop()
    return sem


def _dummy(dev):
    d = _DUMMY.get(dev)
    if d is None:
        d = _DUMMY[dev] = (torch.empty(1, dtype=torch.float32, device=dev),
                           torch.zeros(1, dtype=torch.int32, device=dev))
    return d


# --- references (torch) ----------------------------------------------------------
def dequant(wt: Weights):
    """Exact fp32 weight [B?, N, K] the kernel multiplies by."""
    if not wt.w4:
        return wt.w.float() * wt.s.reshape(*wt.w.shape[:-1], 1)
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=wt.w.device)
    lead = wt.w.shape[:-1]
    vals = torch.stack((lut[(wt.w & 0xF).long()], lut[(wt.w >> 4).long()]), -1)
    vals = vals.reshape(*lead, wt.k // 16, 16) * wt.sf.float()[..., None]
    gs = wt.s.reshape(-1, *([1] * len(lead))) if wt.w.dim() == 3 else wt.s.reshape(1, 1)
    return vals.reshape(*lead, wt.k) * gs


def a8_activation(x, bk: int):
    """What the a8 kernel multiplies: x quantized to e4m3 per (row, bk chunk), dequantized."""
    xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // bk, bk)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    xq = (xf * (448.0 / amax)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (xq * (amax / 448.0)).reshape(x.shape)


@contextmanager
def _exact_fp32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def reference(x, deq, cfg: Cfg | None = None):
    """fp32 x @ deq^T (a8 configs: x as that kernel quantizes it), TF32 off."""
    with _exact_fp32():
        xa = a8_activation(x, cfg.bk) if cfg is not None and cfg.a8 else x.float()
        return xa.matmul(deq.transpose(-1, -2))


def rel_err(y, ref) -> float:
    """rel. Frobenius of y vs ref rounded to y's dtype (the output rounding is not error)."""
    a, b = y.detach().double(), ref.to(y.dtype).double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# --- the kernels ------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _e2m1(v):
        """int16 nibble (bit 3 = sign) -> exact e2m1 value (fp16 subnormal bits x 2^14)."""
        bits = ((v & 8) << 12) | ((v & 7) << 9)
        return tl.cast(bits.to(tl.int16), tl.float16, bitcast=True) * 16384.0

    @triton.jit
    def _w4_tile(p, sc, BN: tl.constexpr, BK: tl.constexpr):
        """packed u8 [BN, BK/2] + e4m3 block scales [BN, BK/16] -> (even-k, odd-k) weight
        values [BN, BK/2], exact in fp16."""
        p16 = p.to(tl.int16)
        s3 = tl.reshape(sc.to(tl.float16), (BN, BK // 16, 1))
        lo = tl.reshape(tl.reshape(_e2m1(p16 & 15), (BN, BK // 16, 8)) * s3, (BN, BK // 2))
        hi = tl.reshape(tl.reshape(_e2m1(p16 >> 4), (BN, BK // 16, 8)) * s3, (BN, BK // 2))
        return lo, hi

    @triton.jit(do_not_specialize=["M", "kb_split", "n_split"])
    def _dense_kernel(
        x_ptr, w_ptr, s_ptr, sf_ptr, out_ptr, ws_ptr, sem_ptr,
        M, N, K,
        stride_xb, stride_xm, stride_ob, stride_om,
        stride_wb, stride_wn, stride_sfb, stride_sfn,
        kb_split, n_split,
        W4: tl.constexpr, A8: tl.constexpr, MMA: tl.constexpr, XS: tl.constexpr,
        PARTIAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_z = tl.program_id(2)
        m_tiles = tl.cdiv(M, BM)
        b = pid_z // m_tiles
        pid_m = pid_z - b * m_tiles
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_ok = offs_m < M
        n_ok = offs_n < N
        k_lo = pid_k * kb_split * BK
        k_hi = tl.minimum(k_lo + kb_split * BK, K)
        x_b = x_ptr + b * stride_xb
        w_b = w_ptr + b * stride_wb
        dt = x_ptr.dtype.element_ty
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sf_b = sf_ptr + b * stride_sfb
        if MMA:  # [BN, BK] weight tiles into tl.dot, cp.async-pipelined (num_stages)
            x_ptrs = x_b + offs_m[:, None] * stride_xm + (k_lo + tl.arange(0, BK))[None, :]
            if W4:
                w_ptrs = (w_b + offs_n[:, None] * stride_wn
                          + (k_lo // 2 + tl.arange(0, BK // 2))[None, :])
                sf_ptrs = (sf_b + offs_n[:, None] * stride_sfn
                           + (k_lo // 16 + tl.arange(0, BK // 16))[None, :])
            else:
                w_ptrs = w_b + offs_n[:, None] * stride_wn + (k_lo + tl.arange(0, BK))[None, :]
            for _k0 in tl.range(k_lo, k_hi, BK):
                xt = tl.load(x_ptrs, mask=m_ok[:, None], other=0.0)
                if W4:
                    lo, hi = _w4_tile(
                        tl.load(w_ptrs, mask=n_ok[:, None], other=0, eviction_policy="evict_first"),
                        tl.load(sf_ptrs, mask=n_ok[:, None], other=0.0,
                                eviction_policy="evict_first"), BN, BK)
                    if XS:
                        xe, xo = tl.split(tl.reshape(xt, (BM, BK // 2, 2)))
                        acc = tl.dot(xe, tl.trans(lo.to(dt)), acc)
                        acc = tl.dot(xo, tl.trans(hi.to(dt)), acc)
                    else:
                        wv = tl.reshape(tl.join(lo, hi), (BN, BK))
                        acc = tl.dot(xt, tl.trans(wv.to(dt)), acc)
                    w_ptrs += BK // 2
                    sf_ptrs += BK // 16
                else:
                    wt = tl.load(w_ptrs, mask=n_ok[:, None], other=0.0,
                                 eviction_policy="evict_first")
                    if A8:
                        xf = xt.to(tl.float32)
                        amax = tl.maximum(tl.max(tl.abs(xf), axis=1), 1e-30)
                        xq = tl.cast(xf * (448.0 / amax)[:, None], tl.float8e4nv,
                                     fp_downcast_rounding="rtne")  # as a8_activation
                        acc += tl.dot(xq, tl.trans(wt)) * (amax / 448.0)[:, None]
                    else:
                        acc = tl.dot(xt, tl.trans(wt.to(dt)), acc)
                    w_ptrs += BK
                x_ptrs += BK
        else:  # CUDA-core GEMV on [BK, BN] tiles kept in their load layout (the K sum lands
            # in acc's; only x rows convert); the next tile loads while this one is used.
            # W4 scales its 16-element block sums, not the weights (no tile-sized broadcast).
            rows = tl.arange(0, BM)
            live = k_lo < k_hi
            if W4:
                w_ptrs = (w_b + offs_n[None, :] * stride_wn
                          + (k_lo // 2 + tl.arange(0, BK // 2))[:, None])
                sf_ptrs = (sf_b + offs_n[None, :] * stride_sfn
                           + (k_lo // 16 + tl.arange(0, BK // 16))[:, None])
                p_nx = tl.load(w_ptrs, mask=n_ok[None, :] & live, other=0,
                               eviction_policy="evict_first")
                s_nx = tl.load(sf_ptrs, mask=n_ok[None, :] & live, other=0.0,
                               eviction_policy="evict_first")
                xk = tl.arange(0, BK // 2)[:, None] * 2 + tl.arange(0, 2)[None, :]
            else:
                w_ptrs = w_b + offs_n[None, :] * stride_wn + (k_lo + tl.arange(0, BK))[:, None]
                w_nx = tl.load(w_ptrs, mask=n_ok[None, :] & live, other=0.0,
                               eviction_policy="evict_first")
                xk = tl.arange(0, BK)
            for k0 in tl.range(k_lo, k_hi, BK):
                more = k0 + BK < k_hi
                if W4:
                    p_cur = p_nx
                    s_cur = s_nx
                    w_ptrs += BK // 2
                    sf_ptrs += BK // 16
                    p_nx = tl.load(w_ptrs, mask=n_ok[None, :] & more, other=0,
                                   eviction_policy="evict_first")
                    s_nx = tl.load(sf_ptrs, mask=n_ok[None, :] & more, other=0.0,
                                   eviction_policy="evict_first")
                    p16 = p_cur.to(tl.int16)
                    wlo = _e2m1(p16 & 15).to(tl.float32)  # [BK/2, BN], unscaled
                    whi = _e2m1(p16 >> 4).to(tl.float32)
                    sc = s_cur.to(tl.float32)  # [BK/16, BN]
                else:
                    w_cur = w_nx
                    w_ptrs += BK
                    w_nx = tl.load(w_ptrs, mask=n_ok[None, :] & more, other=0.0,
                                   eviction_policy="evict_first")
                    wf = w_cur.to(tl.float32)
                for mm in tl.static_range(BM):
                    r = pid_m * BM + mm
                    xr = tl.load(x_b + r * stride_xm + k0 + xk, mask=r < M, other=0.0)
                    if W4:  # [BK/2, 2] -> even / odd k; per-block sums x block scale
                        xe, xo = tl.split(xr.to(tl.float32))
                        prod = wlo * xe[:, None] + whi * xo[:, None]
                        blk = tl.sum(tl.reshape(prod, (BK // 16, 8, BN)), axis=1)
                        part = tl.sum(blk * sc, axis=0)
                    else:  # [BK] broadcast over N
                        part = tl.sum(wf * xr.to(tl.float32)[:, None], axis=0)
                    acc = tl.where(rows[:, None] == mm, acc + part[None, :], acc)
        # epilogue: scales folded here (FP8 channel scale / NVFP4 1/global)
        if W4:
            res = acc * tl.load(s_ptr + b)
        else:
            res = acc * tl.load(s_ptr + b * N + offs_n, mask=n_ok, other=0.0)[None, :]
        msk = m_ok[:, None] & n_ok[None, :]
        o_ptrs = out_ptr + b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :]
        p_ptrs = ws_ptr + (((b * n_split + pid_k) * M + offs_m) * N)[:, None] + offs_n[None, :]
        if PARTIAL:
            tl.store(p_ptrs, res, mask=msk)
        else:
            if n_split == 1:
                tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=msk)
            else:
                tl.store(p_ptrs, res, mask=msk)
                tl.debug_barrier()  # every warp's partial is issued before the release
                tile = pid_z * tl.num_programs(1) + pid_n
                cnt = tl.atomic_add(sem_ptr + tile, 1, sem="acq_rel", scope="gpu")
                if cnt == n_split - 1:  # last split of this tile: sum in split order
                    tot = tl.zeros((BM, BN), dtype=tl.float32)
                    for j in tl.range(0, n_split, loop_unroll_factor=4):
                        q_ptrs = (ws_ptr + (((b * n_split + j) * M + offs_m) * N)[:, None]
                                  + offs_n[None, :])
                        tot += tl.load(q_ptrs, mask=msk, other=0.0, cache_modifier=".cg")
                    tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=msk)
                    tl.atomic_xchg(sem_ptr + tile, 0)

    @triton.jit(do_not_specialize=["M", "n_split"])
    def _reduce_kernel(ws_ptr, out_ptr, M, N, stride_ob, stride_om, n_split, BN: tl.constexpr):
        row = tl.program_id(0)
        b = row // M
        m = row - b * M
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        n_ok = offs_n < N
        tot = tl.zeros((BN,), dtype=tl.float32)
        for j in tl.range(0, n_split, loop_unroll_factor=4):
            tot += tl.load(ws_ptr + ((b * n_split + j) * M + m) * N + offs_n, mask=n_ok, other=0.0)
        tl.store(out_ptr + b * stride_ob + m * stride_om + offs_n,
                 tot.to(out_ptr.dtype.element_ty), mask=n_ok)


def run(x, wt: Weights, plan: Plan, sem=None, out=None):
    """x [M, K] (one weight) or [B, M, K] (any strides, unit K stride) -> [M, N] / [B, M, N].
    sem: this stream's zeroed tile counters (last-CTA split plans only)."""
    cfg = plan.cfg
    m = x.shape[-2]
    if out is None:
        out = x.new_empty((wt.b, m, wt.n) if x.dim() == 3 else (m, wt.n))
    if m == 0:
        return out
    ws, dsem = _dummy(x.device)
    if plan.n_split > 1:
        ws = torch.empty(plan.ws(wt.b, m, wt.n), dtype=torch.float32, device=x.device)
        if not plan.two and sem is None:
            raise ValueError("last-CTA split-K plan without tile counters")
    if plan.n_split == 1 or plan.two:
        sem = dsem
    sf = wt.sf if wt.w4 else wt.s
    three = wt.w.dim() == 3
    sob, som = (out.stride(0) if out.dim() == 3 else 0), out.stride(-2)
    _dense_kernel[(plan.n_split, triton.cdiv(wt.n, cfg.bn), wt.b * triton.cdiv(m, cfg.bm))](
        x, wt.w, wt.s, sf, out, ws, sem, m, wt.n, wt.k,
        x.stride(0) if x.dim() == 3 else 0,
        x.stride(-2) if m > 1 else wt.k,  # M = 1: never a new stride specialization
        sob, som,
        wt.w.stride(0) if three else 0, wt.w.stride(-2),
        (sf.stride(0) if three else 0) if wt.w4 else 0, sf.stride(-2) if wt.w4 else 0,
        plan.kb_split, plan.n_split,
        W4=wt.w4, A8=cfg.a8, MMA=cfg.mma, XS=cfg.xs, PARTIAL=plan.two,
        BM=cfg.bm, BN=cfg.bn, BK=cfg.bk, num_warps=cfg.warps, num_stages=cfg.stages)
    if plan.two:
        _reduce_kernel[(wt.b * m, triton.cdiv(wt.n, RED_BN))](
            ws, out, m, wt.n, sob, som, plan.n_split, BN=RED_BN, num_warps=4)
    return out


# --- tuning (GPU) -------------------------------------------------------------------
def budget_s() -> float:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    try:
        v = float(raw) if raw else BUDGET_S
    except ValueError:
        v = -1.0
    if v < 0:
        raise ValueError(f"{BUDGET_ENV}={raw!r}: want seconds >= 0")
    return v


def _alive(dev, exc) -> None:
    """A config that failed to build is skipped; a sticky CUDA error is not."""
    try:
        torch.cuda.synchronize(dev)
    except Exception:  # noqa: BLE001
        raise exc from None


def tune_shape(wt: Weights, ms, *, dev, flush, sem, nsm: int = NSM, extra=None,
               big_m=None, default_only=False, log=print):
    """Check then time (cold L2, graphed, median) every candidate plan at each M in ms
    (+ unsplit plans at big_m) and every `extra` baseline fn(x) -> (table, bad):
    table {m: {"ours": [(us, plan), ...] fastest first, "base": {name: us}}}; bad =
    candidates that broke OURS_BOUND (excluded). Unbuildable configs are logged once.
    Load time runs under vLLM's bf16 default dtype: every tensor made here and in verify
    / dequant / reference carries an explicit dtype."""
    from suffix_hybrid.kernels.nvfp4_gemm import _graph_us

    kind = "w4" if wt.w4 else "w8"
    deq = dequant(wt)
    gen = torch.Generator(device=dev).manual_seed(wt.n * 131 + wt.k)
    table, bad, broken = {}, [], {}
    for m in list(ms) + ([big_m] if big_m else []):
        big = m == big_m
        if wt.b > 1:  # vLLM's absorbed-MLA bmm layout: [T, H, K] viewed as [H, T, K]
            x = torch.randn(m, wt.b, wt.k, device=dev, generator=gen,
                            dtype=torch.float32).bfloat16().transpose(0, 1)
        else:
            x = torch.randn(m, wt.k, device=dev, generator=gen, dtype=torch.float32).bfloat16()
        out = x.new_empty((wt.b, m, wt.n) if wt.b > 1 else (m, wt.n))
        refs = {}
        row = {"ours": [], "base": {}}
        cands = ([default_plan(kind, wt.b, wt.n, wt.k, m, nsm)] if default_only
                 else plans(kind, wt.b, wt.n, wt.k, m, nsm, big))

        def attempt(plan):
            tag = plan.cfg.tag() + ("r" if plan.two else "")
            if tag in broken:
                return
            try:
                run(x, wt, plan, sem, out)
                torch.cuda.synchronize(dev)
            except Exception as exc:  # noqa: BLE001 - compile / resource limits: skip config
                _alive(dev, exc)
                broken[tag] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
                return
            rk = plan.cfg.bk if plan.cfg.a8 else 0
            if rk not in refs:
                refs[rk] = reference(x, deq, plan.cfg if rk else None)
            err = rel_err(out, refs[rk])
            if not err <= bound(plan.cfg):
                bad.append(f"{kind} {wt.b}x{wt.n}x{wt.k} M={m} {plan.tag()}: rel {err:.2e} "
                           f"> {bound(plan.cfg)}")
                return
            us = _graph_us(lambda s: run(x, wt, plan, sem, out), dev, REPS, flush)
            row["ours"].append((us, plan))

        for plan in cands:
            attempt(plan)
        split = [p for _us, p in sorted(row["ours"], key=lambda t: t[0]) if p.n_split > 1]
        if split:  # the two-stage reduction against the fastest split plan
            attempt(split[0].two_stage())
        for name, fn in ({} if big else (extra or {})).items():
            if name in broken:
                continue
            try:
                fn(x)
                torch.cuda.synchronize(dev)
                row["base"][name] = _graph_us(lambda s, fn=fn: fn(x), dev, REPS, flush)
            except Exception as exc:  # noqa: BLE001
                _alive(dev, exc)
                broken[name] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
        row["ours"].sort(key=lambda t: t[0])
        table[m] = row
    for tag, why in broken.items():
        log(f"{kind} {wt.b}x{wt.n}x{wt.k}: {tag} unavailable: {why}")
    return table, bad


def _src_sig() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def cache_path() -> str:
    return os.environ.get(CACHE_ENV) or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "suffix_hybrid", "fp8_dense_ours.json")


def cache_get(key: str):
    try:
        with open(cache_path()) as f:
            hit = json.load(f).get(key)
        return {int(m): {"ours": [(us, Plan.from_json(p)) for us, p in row["ours"]],
                         "base": dict(row["base"])} for m, row in hit.items()}
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return None


def cache_put(key: str, table: dict) -> None:
    """Merge-write, atomic rename (TP ranks write concurrently; each entry is whole)."""
    path = cache_path()
    try:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data[key] = {str(m): {"ours": [(us, p.to_json()) for us, p in row["ours"][:4]],
                              "base": row["base"]} for m, row in table.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def verify(route: Route, wt: Weights, dev) -> None:
    """Every routed plan (and the two-stage form a stream without counters falls back
    to) once more at its bucket's M vs the exact reference: cached timings are never
    trusted for numerics, and nothing compiles or loads later, under graph capture.
    Raises."""
    deq = dequant(wt)
    sem = torch.zeros(SEM_SLOTS, dtype=torch.int32, device=dev)
    done = set()
    for m, (path, plan) in list(zip(route.ms, route.choices)) + [(BIG_M, route.big)]:
        if path != "ours":
            continue
        for p in (plan, plan.two_stage()) if plan.n_split > 1 and not plan.two else (plan,):
            if (m, p) in done:
                continue
            done.add((m, p))
            x = torch.randn(m, wt.k, device=dev, dtype=torch.bfloat16)
            if wt.b > 1:
                x = torch.randn(m, wt.b, wt.k, device=dev, dtype=torch.bfloat16).transpose(0, 1)
            err = rel_err(run(x, wt, p, sem), reference(x, deq, p.cfg))
            if not err <= bound(p.cfg):
                raise RuntimeError(f"routed plan {p.tag()} at M={m} on {wt.b}x{wt.n}x{wt.k}: "
                                   f"rel err {err:.3e} > {bound(p.cfg)}")


def tune_route(wt: Weights, dev, extra: dict, big: str | None, log=print) -> Route:
    """Route for one shape: cached or timed (bounded by budget_s()), verified. big =
    the baseline above the last bucket ('cutlass'), None = our fastest unsplit tile
    at BIG_M (ours_fp4)."""
    props = torch.cuda.get_device_properties(dev)
    nsm = props.multi_processor_count
    big_m = None if big else BIG_M
    shape = f"{'w4' if wt.w4 else 'w8'} {wt.b}x{wt.n}x{wt.k}"
    sig = hashlib.sha256(repr((_src_sig(), BUCKETS, big_m, REPS, nsm, sorted(extra))).encode())
    key = f"{props.name}|{shape}|{sig.hexdigest()[:16]}"
    table, note = cache_get(key), "cached"
    if table is None:
        over = _STATE["tune_s"] > budget_s()
        if over and big:
            log(f"ours route {shape}: startup budget {budget_s():.0f} s spent -> {big} "
                "for every M (a restart reuses the cached shapes and times the rest)")
            return Route(BUCKETS, [(big, None)] * len(BUCKETS), (big, None))
        t0 = time.monotonic()
        flush_buf = torch.empty(max(2 * props.L2_cache_size, 64 << 20), dtype=torch.uint8,
                                device=dev)
        sem = torch.zeros(SEM_SLOTS, dtype=torch.int32, device=dev)
        table, bad = tune_shape(wt, BUCKETS, dev=dev, flush=flush_buf.zero_, sem=sem, nsm=nsm,
                                extra=extra, big_m=big_m, default_only=over, log=log)
        del flush_buf, sem
        for line in bad:
            log(f"WRONG {line} -- excluded from routing")
        dt = time.monotonic() - t0
        _STATE["tune_s"] += dt
        if not over:
            cache_put(key, table)
        note = f"{'untimed guess, budget spent; ' if over else ''}timed {dt:.1f} s"
    choice = decide(table, routable=set(extra))
    if big:
        bigc = (big, None)
    else:
        unsplit = [p for _us, p in table[big_m]["ours"] if p.n_split == 1 and not p.two]
        if not unsplit:
            raise RuntimeError(f"ours route {shape}: no working unsplit tile at M={big_m}")
        bigc = ("ours", unsplit[0])
    route = Route(BUCKETS, [choice[m] for m in BUCKETS], bigc)
    verify(route, wt, dev)
    log(f"ours route {shape} [{note}]: {route.line(table)}")
    return route
