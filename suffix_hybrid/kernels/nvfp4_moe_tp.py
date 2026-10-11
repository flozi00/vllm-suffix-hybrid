# SPDX-License-Identifier: Apache-2.0
"""TP-sharded NVFP4 routed-experts decode MoE in Triton (GLM-5.3 TP8 without EP: every rank holds
all 256 experts at I = 2048/8 = 256, top-8, H 6144) — kernels, vLLM backend switch, load oracle,
boot gate.

Why a second family next to kernels-oxide/nvfp4_moe: under TP the per-expert work is tiny (I 256)
and spread over all 256 experts, so the oxide kernels (built for EP's 32 x I 2048) run their TP
shard at ~45-60 % of GDDR7 bandwidth (gate glm_moe_tp_bench: M=6 111.6 us for ~118 MB of weights,
M=192 595 us for ~680 MB). These kernels are built for exactly this layout:

  K1 fc1   grid (I / (BN/2), active experts ascending): one CTA per (expert, BN/2 intermediate
           columns), i.e. BN/2 up + BN/2 gate weight rows over the whole K = H. Routing is done
           in-kernel (histogram of topk_ids -> the CTA's expert, its pair list scattered to a
           workspace: no route launch). x is NVFP4-quantized on the fly with vLLM's
           scaled_fp4_quant math (a1 gscale), weights dequantized in registers (e2m1 bit
           placement + e4m3 block scale, exact), fp32 accumulation. Epilogue: g1 alpha,
           act(gate) * up, NVFP4 quant with the a2 gscale (each CTA owns whole 16-column
           blocks), hq/hsf rows per routed pair. Two bodies:
             fma  R (1|2) routed rows per weight pass, CUDA-core FMA on [BN/2, BK/2] tiles with
                  register double buffering (+ optional L2 prefetch of the CTA's whole weight
                  tile at start): no 16-row MMA padding / x-quant waste at M <= ~16.
             mma  BM (16|32) rows per pass, fp16 tensor-core dot (operands are exact:
                  e2m1 * e4m3 needs <= 6 significant bits and lies in [2^-10, 2688]).
  K2 fc2   tok  grid (M, H/BH): token-major fc2 + combine, 8 experts per CTA in ascending k,
                fp32 sum, bf16 once (oxide fused3's contract), all expert tiles L2-prefetched up
                front; L2 keeps the shared tiles of tokens that route to the same expert.
           exp  grid (H/BHC, active experts): expert-major fc2 (tensor cores) writing fp32 y per
                pair, then K3 combine (ascending k, bf16 once): every w2 tile is dequantized once
                at large M, where token-major would redo it per routed pair.
  PDL      (SUFFIX_NVFP4_MOE_TP_PDL, default on) K1/K2 call griddepcontrol.launch_dependents at
           start; K2/K3 launch programmatically dependent and prefetch their weights before
           griddepcontrol.wait: the w2 stream overlaps fc1's tail instead of following it.

Numerics: the NVFP4 spec of nvfp4_moe.moe_ref (x and h quantized exactly like vLLM / oxide,
e2m1 x e4m3 products exact, fp32 accumulation, y = (h.w2) * (g2 * topk_w), out = bf16(sum_k y)).
Deterministic (no atomics; fixed reduction trees per plan). Not bit-identical to oxide (different
fp32 summation order before the h requant), which the oracle bounds.

Backend switch (inside RoutedExperts.forward_modular, i.e. inside vLLM's opaque moe_forward
custom op: torch.compile never sees the branch; a function of M only, so CUDA-graph safe):
  SUFFIX_NVFP4_MOE_IMPL=oxide (default) | tp | auto
    tp    our kernels for every M <= SUFFIX_NVFP4_MOE_MAX_M (best plan per M)
    auto  per M the faster of ours (best plan) and the oxide kernel, timed at load
  SUFFIX_NVFP4_MOE_TP_TABLE="6:f1,16:f2,64:m64,192:m128"  the routing table pasted from the gate
    (plan names below or "oxide"; a bucket serves every M <= its key); unset = time at load:
    cold L2, CUDA graphs, real weights of the first layer, every PLANS candidate vs oxide.
  SUFFIX_NVFP4_MOE_TP_PDL=0  plain launches.
Fail-closed load oracle per layer (real weights, M 1/6/32/96/192 <= max_m + a hot-expert batch):
ours vs the f64 reference <= nvfp4_moe.REF_TOL, vs the oxide kernel within the triangle bound,
bitwise deterministic, CUDA-graph replay == eager; any miss refuses to serve.

CLI / boot gate glm_moe_tp_kernel_bench (one GPU, synthetic GLM TP8 shard, cold L2, CUDA graphs):
  python -m suffix_hybrid.kernels.nvfp4_moe_tp bench   # oracle, then us/layer per M: every plan,
      oxide (g10 table), FlashInfer CUTLASS; weight roofline and % of bandwidth; the paste-able
      SUFFIX_NVFP4_MOE_TP_TABLE / SUFFIX_NVFP4_MOE_IMPL. Lines: [suffix moe-tp-kernel].
  python -m suffix_hybrid.kernels.nvfp4_moe_tp oracle
"""
from __future__ import annotations

import os
import sys
import zlib

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
except ImportError:  # CPU CI: the Python side stays importable
    triton = tl = libdevice = gdc_launch_dependents = gdc_wait = None

MARK = "[suffix moe-tp-kernel]"
IMPL_ENV = "SUFFIX_NVFP4_MOE_IMPL"
TABLE_ENV = "SUFFIX_NVFP4_MOE_TP_TABLE"
PDL_ENV = "SUFFIX_NVFP4_MOE_TP_PDL"
IMPLS = ("oxide", "tp", "auto")
# plan -> (fc1, fc2)
#   fc1 ("fma", R rows/pass, BN, BK, warps, L2-prefetch whole tile) | ("mma", BM, BN, BK, warps, stages)
#   fc2 ("tok", BH, warps) | ("exp", BM, BH, BHC, warps, stages)
PLANS = {
    "f1": (("fma", 1, 32, 512, 8, 1), ("tok", 32, 4)),
    "f1s": (("fma", 1, 32, 256, 4, 1), ("tok", 32, 4)),
    "f1n": (("fma", 1, 32, 512, 8, 0), ("tok", 32, 4)),
    "f2": (("fma", 2, 32, 256, 8, 1), ("tok", 32, 4)),
    "m64": (("mma", 16, 64, 128, 4, 3), ("tok", 64, 4)),
    "m128": (("mma", 16, 128, 128, 8, 3), ("exp", 16, 64, 1024, 4, 3)),
    "m128w": (("mma", 32, 128, 128, 8, 3), ("exp", 32, 64, 1024, 4, 3)),
}
OXIDE = "oxide"
ORACLE_MS = (1, 6, 32, 96, 192)
TABLE_MS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
CH = 64            # topk ids per route chunk
BC = 1024          # combine columns per CTA
TIME_ITERS = 20
# prod glm g10 oxide tune (SUFFIX_NVFP4_MOE_TUNE): the gate's oxide reference
G10_TUNE = "1:front4/8/0/1,6:front4/8/0/0,8:fused3/8/255/0,192:front4/8/0/0"
_ST = {"ws": {}, "table": None, "pdl": None, "layers": 0, "logged": False}


def _log(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def impl() -> str:
    v = os.environ.get(IMPL_ENV, "").strip() or OXIDE
    if v not in IMPLS:
        raise ValueError(f"{IMPL_ENV}={v!r}: one of {'|'.join(IMPLS)}")
    return v


def pdl_on() -> bool:
    return os.environ.get(PDL_ENV, "1").strip() != "0"


def parse_table(text: str) -> list:
    """"M:plan,..." -> [(m_max, plan)] ascending; plan in PLANS or "oxide"."""
    out = []
    for item in (t.strip() for t in text.split(",")):
        if not item:
            continue
        m, _, plan = item.partition(":")
        if not m.strip().isdigit() or plan.strip() not in (*PLANS, OXIDE):
            raise ValueError(f"{TABLE_ENV}: bad bucket {item!r} (want M:plan, plan in "
                             f"{sorted(PLANS) + [OXIDE]})")
        row = (int(m), plan.strip())
        if not 1 <= row[0] <= 256 or (out and row[0] <= out[-1][0]):
            raise ValueError(f"{TABLE_ENV}: bucket {item!r} out of range / not ascending")
        out.append(row)
    if not out:
        raise ValueError(f"{TABLE_ENV}: empty table")
    return out


def format_table(table) -> str:
    return ",".join(f"{m}:{p}" for m, p in table)


def merge_table(rows) -> list:
    """[(m, plan)] ascending (one per timed M) -> buckets: a run of equal choices keeps its
    largest M."""
    out = []
    for m, plan in rows:
        if out and out[-1][1] == plan:
            out[-1] = (m, plan)
        else:
            out.append((m, plan))
    return out


def by_m(table, max_m: int) -> list:
    """[plan or None (= oxide)] indexed by M (0 unused): the first bucket with M <= its key,
    else the last (nvfp4_moe.tune_for's rule)."""
    out = [None]
    for m in range(1, max_m + 1):
        plan = next((p for k, p in table if m <= k), table[-1][1])
        out.append(None if plan == OXIDE else plan)
    return out


def eligibility(cfg: dict, info: dict) -> str | None:
    """Why this NVFP4 layer cannot use the TP family (None = it can). Pure for CPU tests."""
    if cfg.get("kind") == "fp8":
        return "FP8 block layer"
    if cfg.get("shared") is not None:
        return "shared expert fused into the oxide kernel (SUFFIX_NVFP4_MOE_FUSE_SHARED)"
    e, h, i, topk = info.get("E", 0), info.get("H", 0), info.get("I", 0), cfg.get("topk", 0)
    if info.get("ep", 1) != 1 or cfg.get("id_base", 0) != 0 or e != cfg.get("E_global", e):
        return "expert parallel (this family is the TP-sharded layout: all experts local)"
    if e < 1 or e & (e - 1) or e > 1024:
        return f"E={e} (power of two <= 1024)"
    if i & (i - 1) or not 32 <= i <= 1024:
        return f"I={i} (power of two in 32..1024)"
    if h % 512:
        return f"H={h} (multiple of 512)"
    if not 1 <= topk <= 16:
        return f"top-{topk} (1..16)"
    if cfg.get("act") not in (0, 1):
        return f"activation code {cfg.get('act')}"
    return None


def roofline_bytes(experts: int, h: int, i: int) -> float:
    """Weight bytes one MoE call must stream: distinct experts x (w13 + w2 + their e4m3 scales)."""
    return experts * 3 * i * h * (1 / 2 + 1 / 16)


def _p2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _sf_row(row, KB_PAD: tl.constexpr):
        """Row part of the CUTLASS 128x4 swizzled scale offset (nvfp4_moe.swizzle / sf_offset);
        the full offset is _sf_row(row) + _sf_col(kb)."""
        return (row // 128) * (KB_PAD // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4

    @triton.jit
    def _sf_col(kb):
        return (kb // 4) * 512 + kb % 4

    @triton.jit
    def _pf_l2(ptrs):
        """prefetch.global.L2 per element (a hint: no data, no fault semantics we rely on)."""
        tl.inline_asm_elementwise("prefetch.global.L2 [$1];\nmov.u32 $0, 0;", "=r,l", [ptrs],
                                  dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _act(v, ACT: tl.constexpr):
        """oxide act(): silu = v / (1 + 2^(-v log2e)); gelu_tanh = v * sigmoid(2u)."""
        if ACT == 1:
            z = 2.3022082 * (v + 0.044715 * v * v * v)
        else:
            z = v * 1.442695
        return tl.math.div_rn(v, 1.0 + tl.exp2(-z))

    @triton.jit
    def _e2m1_rne(a):
        """|s| in [0, 6] -> nearest e2m1 magnitude, ties to even (cvt.rn.satfinite.e2m1x2.f32):
        steps 0.5 / 1 / 2 below 2 / 4 / 6, so rint at the binade's step."""
        lo = a < 2.0
        mid = a < 4.0
        up = tl.where(lo, 2.0, tl.where(mid, 1.0, 0.5))
        return libdevice.rint(a * up) * tl.where(lo, 0.5, tl.where(mid, 1.0, 2.0))

    @triton.jit
    def _code(q):
        """e2m1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} -> its 3-bit code."""
        return tl.where(q < 2.0, q * 2.0, tl.where(q < 4.0, q + 2.0, q * 0.5 + 4.0)).to(tl.int32)

    @triton.jit
    def _quant(v, g, g6, AXIS: tl.constexpr):
        """NVFP4 quant of 16-element blocks along AXIS, vLLM scaled_fp4_quant / quant_codes:
        sf = e4m3(amax * (g / 6)), q = e2m1_rne(v * (g / sf)) (IEEE divisions).
        -> (s = v * g / sf, |q|, sf as fp32, sf as e4m3)."""
        amax = tl.max(tl.abs(v), axis=AXIS)
        sf8 = tl.minimum(amax * g6, 448.0).to(tl.float8e4nv)
        sf = sf8.to(tl.float32)
        nz = sf != 0.0
        inv = tl.where(nz, tl.math.div_rn(g + tl.zeros_like(sf), tl.where(nz, sf, 1.0)), 0.0)
        s = v * tl.expand_dims(inv, AXIS)
        return s, _e2m1_rne(tl.minimum(tl.abs(s), 6.0)), sf, sf8

    @triton.jit
    def _xq_row(xr, g, g6, BK: tl.constexpr):
        """x row [BK] fp32 -> quant-dequant (exact e2m1 * sf; 1/g lives in alpha) as the
        (even, odd) elements [BK // 2]: a packed weight byte's low / high nibble partners."""
        s, q, sf, _ = _quant(tl.reshape(xr, (BK // 16, 16)), g, g6, 1)
        d = tl.where(s < 0.0, -q, q) * tl.expand_dims(sf, 1)
        return tl.split(tl.reshape(d, (BK // 2, 2)))

    @triton.jit
    def _xq_rows(xr, g, g6, BM: tl.constexpr, BK: tl.constexpr):
        """_xq_row for [BM, BK] rows -> (even, odd) [BM, BK // 2]."""
        s, q, sf, _ = _quant(tl.reshape(xr, (BM, BK // 16, 16)), g, g6, 2)
        d = tl.where(s < 0.0, -q, q) * tl.expand_dims(sf, 2)
        return tl.split(tl.reshape(d, (BM, BK // 2, 2)))

    @triton.jit
    def _fp4(b):
        """Packed e2m1 bytes (low nibble = even element) -> (lo, hi) fp16 = e2m1 * 2^-14 by bit
        placement (sign -> bit 15, exponent:mantissa -> bits 11..9; codes 0 / 1 are fp16
        subnormals 0 / 2^-15)."""
        bi = b.to(tl.int32)
        lo = ((bi & 7) << 9) | ((bi & 8) << 12)
        hi = ((bi & 0x70) << 5) | ((bi & 0x80) << 8)
        return (tl.cast(lo.to(tl.int16), tl.float16, bitcast=True),
                tl.cast(hi.to(tl.int16), tl.float16, bitcast=True))

    @triton.jit
    def _deq32(b, s8, ROWS: tl.constexpr, NB: tl.constexpr):
        """bytes [ROWS, NB*8] + e4m3 scale bytes [ROWS, NB] -> (lo, hi) fp32 = e2m1 * sf, exact."""
        lo, hi = _fp4(b)
        s = tl.expand_dims(tl.cast(s8, tl.float8e4nv, bitcast=True).to(tl.float32) * 16384.0, 2)
        lo = tl.reshape(tl.reshape(lo.to(tl.float32), (ROWS, NB, 8)) * s, (ROWS, NB * 8))
        hi = tl.reshape(tl.reshape(hi.to(tl.float32), (ROWS, NB, 8)) * s, (ROWS, NB * 8))
        return lo, hi

    @triton.jit
    def _deq16(b, s8, ROWS: tl.constexpr, NB: tl.constexpr):
        """_deq32 as fp16 MMA operands: (e2m1 * 2^-14 * 2^14) * sf, both products exact and
        every nonzero value fp16-normal (e2m1 * e4m3 in [2^-10, 2688])."""
        lo, hi = _fp4(b)
        s = tl.expand_dims(tl.cast(s8, tl.float8e4nv, bitcast=True).to(tl.float16), 2)
        lo = tl.reshape(tl.reshape(lo * 16384.0, (ROWS, NB, 8)) * s, (ROWS, NB * 8))
        hi = tl.reshape(tl.reshape(hi * 16384.0, (ROWS, NB, 8)) * s, (ROWS, NB * 8))
        return lo, hi

    @triton.jit
    def _route(ids_ptr, P, slot, E: tl.constexpr, CHUNK: tl.constexpr):
        """Slot -> (expert, its routed pairs, its first pair-list index). Slots are the active
        experts ascending (oxide moe_route); a slot past the last active expert gets n_e = 0."""
        cnt = tl.zeros([E], dtype=tl.int32)
        for c0 in range(0, P, CHUNK):
            o = c0 + tl.arange(0, CHUNK)
            v = tl.load(ids_ptr + o, mask=o < P, other=-1).to(tl.int32)
            ok = (v >= 0) & (v < E)
            cnt = cnt + tl.histogram(tl.where(ok, v, 0), E, mask=ok)
        act = (cnt > 0).to(tl.int32)
        ev = tl.arange(0, E)
        sel = (act > 0) & ((tl.cumsum(act, 0) - act) == slot)
        e = tl.sum(tl.where(sel, ev, 0), 0)
        n_e = tl.sum(tl.where(sel, cnt, 0), 0)
        return e, n_e, tl.sum(tl.where(ev < e, cnt, 0), 0)

    @triton.jit
    def _pairs(ids_ptr, pair_ptr, P, e, off, CHUNK: tl.constexpr):
        """pair_list[off ..] = expert e's pairs (p = token * topk + k) ascending. Every CTA of the
        slot writes the same values; the caller's barrier publishes them inside the CTA."""
        base = off
        for c0 in range(0, P, CHUNK):
            o = c0 + tl.arange(0, CHUNK)
            v = tl.load(ids_ptr + o, mask=o < P, other=-1).to(tl.int32)
            me = ((v == e) & (o < P)).to(tl.int32)
            tl.store(pair_ptr + base + tl.cumsum(me, 0) - me, o, mask=me > 0)
            base = base + tl.sum(me, 0)

    @triton.jit
    def _fc1_out(u, g, alpha, p, rv, hq_ptr, hs_ptr, j0, a2g, g6h, I: tl.constexpr,
                 BM: tl.constexpr, HN: tl.constexpr, ACT: tl.constexpr):
        """[BM, HN] fp32 up / gate dots -> h = act(g1 * gate) * (g1 * up) -> NVFP4 (a2 gscale):
        packed bytes + e4m3 scales at hq / hsf row p (valid rows rv)."""
        h = _act(g * alpha, ACT) * (u * alpha)
        s, q, _, sf8 = _quant(tl.reshape(h, (BM, HN // 16, 16)), a2g, g6h, 2)
        code = tl.reshape(_code(q) | tl.where(s < 0.0, 8, 0), (BM, HN // 2, 2))
        lo, hi = tl.split(code)
        tl.store(hq_ptr + p[:, None] * (I // 2) + j0 // 2 + tl.arange(0, HN // 2)[None, :],
                 (lo | (hi << 4)).to(tl.uint8), mask=rv[:, None])
        tl.store(hs_ptr + p[:, None] * (I // 16) + j0 // 16 + tl.arange(0, HN // 16)[None, :],
                 tl.cast(sf8, tl.uint8, bitcast=True), mask=rv[:, None])

    @triton.jit(do_not_specialize=["M"])
    def _fc1_fma(x_ptr, ids_ptr, w_ptr, s_ptr, g1_ptr, hq_ptr, hs_ptr, pair_ptr, M, a1g, a2g,
                 H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 KB_PAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, R: tl.constexpr,
                 PF: tl.constexpr, LN: tl.constexpr, ACT: tl.constexpr, PDL: tl.constexpr,
                 CHUNK: tl.constexpr):
        HN: tl.constexpr = BN // 2
        BKH: tl.constexpr = BK // 2
        NBK: tl.constexpr = BK // 16
        if PDL:
            gdc_launch_dependents()
        P = M * TOPK
        e, n_e, off = _route(ids_ptr, P, tl.program_id(1), E, CHUNK)
        if n_e == 0:
            return
        j0 = tl.program_id(0) * HN
        rows = j0 + tl.arange(0, HN)
        w_e = w_ptr + e.to(tl.int64) * (2 * I * (H // 2))
        if PF:  # the CTA's whole weight tile -> L2 now: every DRAM request in flight at once
            ln = tl.minimum(tl.arange(0, LN) * 128, H // 2 - 128)
            _pf_l2(w_e + rows[:, None] * (H // 2) + ln[None, :])
            _pf_l2(w_e + (I + rows)[:, None] * (H // 2) + ln[None, :])
        wu_ptr = w_e + rows[:, None] * (H // 2) + tl.arange(0, BKH)[None, :]
        wg_ptr = wu_ptr + I * (H // 2)
        s_e = s_ptr + e.to(tl.int64) * (((2 * I + 127) // 128 * 128) * KB_PAD)
        kcol = _sf_col(tl.arange(0, NBK))[None, :]
        su_ptr = s_e + _sf_row(rows, KB_PAD)[:, None] + kcol
        sg_ptr = s_e + _sf_row(I + rows, KB_PAD)[:, None] + kcol
        wu_0 = tl.load(wu_ptr)  # the first K tile, shared by every row pass
        wg_0 = tl.load(wg_ptr)
        su_0 = tl.load(su_ptr)
        sg_0 = tl.load(sg_ptr)
        _pairs(ids_ptr, pair_ptr, P, e, off, CHUNK)
        tl.debug_barrier()
        alpha = tl.load(g1_ptr + e)
        g6x = tl.math.div_rn(a1g, 6.0)
        g6h = tl.math.div_rn(a2g, 6.0)
        kx = tl.arange(0, BK)
        for r0 in range(0, n_e, R):
            p0 = tl.load(pair_ptr + off + r0, cache_modifier=".cg")
            x0_ptr = x_ptr + (p0 // TOPK).to(tl.int64) * H + kx
            acc_u0 = tl.zeros([HN, BKH], dtype=tl.float32)
            acc_g0 = tl.zeros([HN, BKH], dtype=tl.float32)
            x0_n = tl.load(x0_ptr)
            if R == 2:
                v1 = r0 + 1 < n_e
                p1 = tl.load(pair_ptr + off + r0 + 1, mask=v1, other=0, cache_modifier=".cg")
                x1_ptr = x_ptr + (p1 // TOPK).to(tl.int64) * H + kx
                acc_u1 = tl.zeros([HN, BKH], dtype=tl.float32)
                acc_g1 = tl.zeros([HN, BKH], dtype=tl.float32)
                x1_n = tl.load(x1_ptr, mask=v1, other=0.0)
            wu_n = wu_0
            wg_n = wg_0
            su_n = su_0
            sg_n = sg_0
            for k0 in range(0, H, BK):  # register double buffer: tile k0 + BK in flight
                wu = wu_n
                wg = wg_n
                su = su_n
                sg = sg_n
                x0 = x0_n
                nx = k0 + BK < H
                wu_n = tl.load(wu_ptr + (k0 + BK) // 2, mask=nx, other=0)
                wg_n = tl.load(wg_ptr + (k0 + BK) // 2, mask=nx, other=0)
                su_n = tl.load(su_ptr + (k0 + BK) // 64 * 512, mask=nx, other=0)
                sg_n = tl.load(sg_ptr + (k0 + BK) // 64 * 512, mask=nx, other=0)
                x0_n = tl.load(x0_ptr + k0 + BK, mask=nx, other=0.0)
                ul, uh = _deq32(wu, su, HN, NBK)
                gl, gh = _deq32(wg, sg, HN, NBK)
                xe, xo = _xq_row(x0.to(tl.float32), a1g, g6x, BK)
                acc_u0 = acc_u0 + ul * xe[None, :] + uh * xo[None, :]
                acc_g0 = acc_g0 + gl * xe[None, :] + gh * xo[None, :]
                if R == 2:
                    x1 = x1_n
                    x1_n = tl.load(x1_ptr + k0 + BK, mask=v1 & nx, other=0.0)
                    xe1, xo1 = _xq_row(x1.to(tl.float32), a1g, g6x, BK)
                    acc_u1 = acc_u1 + ul * xe1[None, :] + uh * xo1[None, :]
                    acc_g1 = acc_g1 + gl * xe1[None, :] + gh * xo1[None, :]
            one = tl.zeros([1], dtype=tl.int32)
            _fc1_out(tl.reshape(tl.sum(acc_u0, 1), (1, HN)), tl.reshape(tl.sum(acc_g0, 1), (1, HN)),
                     alpha, one + p0, (one + r0) < n_e, hq_ptr, hs_ptr, j0, a2g, g6h, I, 1, HN, ACT)
            if R == 2:
                _fc1_out(tl.reshape(tl.sum(acc_u1, 1), (1, HN)),
                         tl.reshape(tl.sum(acc_g1, 1), (1, HN)),
                         alpha, one + p1, (one + r0 + 1) < n_e, hq_ptr, hs_ptr, j0, a2g, g6h, I, 1,
                         HN, ACT)

    @triton.jit(do_not_specialize=["M"])
    def _fc1_mma(x_ptr, ids_ptr, w_ptr, s_ptr, g1_ptr, hq_ptr, hs_ptr, pair_ptr, M, a1g, a2g,
                 H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 KB_PAD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 STAGES: tl.constexpr, ACT: tl.constexpr, PDL: tl.constexpr, CHUNK: tl.constexpr):
        HN: tl.constexpr = BN // 2
        BKH: tl.constexpr = BK // 2
        NBK: tl.constexpr = BK // 16
        if PDL:
            gdc_launch_dependents()
        P = M * TOPK
        e, n_e, off = _route(ids_ptr, P, tl.program_id(1), E, CHUNK)
        if n_e == 0:
            return
        j0 = tl.program_id(0) * HN
        rows = j0 + tl.arange(0, HN)
        w_e = w_ptr + e.to(tl.int64) * (2 * I * (H // 2))
        wu_ptr = w_e + rows[:, None] * (H // 2) + tl.arange(0, BKH)[None, :]
        wg_ptr = wu_ptr + I * (H // 2)
        s_e = s_ptr + e.to(tl.int64) * (((2 * I + 127) // 128 * 128) * KB_PAD)
        kcol = _sf_col(tl.arange(0, NBK))[None, :]
        su_ptr = s_e + _sf_row(rows, KB_PAD)[:, None] + kcol
        sg_ptr = s_e + _sf_row(I + rows, KB_PAD)[:, None] + kcol
        _pairs(ids_ptr, pair_ptr, P, e, off, CHUNK)
        tl.debug_barrier()
        alpha = tl.load(g1_ptr + e)
        g6x = tl.math.div_rn(a1g, 6.0)
        g6h = tl.math.div_rn(a2g, 6.0)
        for r0 in range(0, n_e, BM):
            rr = r0 + tl.arange(0, BM)
            rv = rr < n_e
            p = tl.load(pair_ptr + off + rr, mask=rv, other=0, cache_modifier=".cg")
            x_rows = x_ptr + (p // TOPK).to(tl.int64)[:, None] * H + tl.arange(0, BK)[None, :]
            acc_u = tl.zeros([BM, HN], dtype=tl.float32)
            acc_g = tl.zeros([BM, HN], dtype=tl.float32)
            for k0 in tl.range(0, H, BK, num_stages=STAGES):
                xr = tl.load(x_rows + k0, mask=rv[:, None], other=0.0)
                xe, xo = _xq_rows(xr.to(tl.float32), a1g, g6x, BM, BK)
                ul, uh = _deq16(tl.load(wu_ptr + k0 // 2), tl.load(su_ptr + k0 // 64 * 512), HN,
                                NBK)
                gl, gh = _deq16(tl.load(wg_ptr + k0 // 2), tl.load(sg_ptr + k0 // 64 * 512), HN,
                                NBK)
                xe = xe.to(tl.float16)
                xo = xo.to(tl.float16)
                acc_u = tl.dot(xe, tl.trans(ul), acc_u)
                acc_u = tl.dot(xo, tl.trans(uh), acc_u)
                acc_g = tl.dot(xe, tl.trans(gl), acc_g)
                acc_g = tl.dot(xo, tl.trans(gh), acc_g)
            _fc1_out(acc_u, acc_g, alpha, p, rv, hq_ptr, hs_ptr, j0, a2g, g6h, I, BM, HN, ACT)

    @triton.jit
    def _fc2_tok(out_ptr, hq_ptr, hs_ptr, ids_ptr, tw_ptr, w_ptr, s_ptr, g2_ptr,
                 H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 TKP: tl.constexpr, KB_PAD: tl.constexpr, BH: tl.constexpr, LNI: tl.constexpr,
                 PDL: tl.constexpr):
        IH: tl.constexpr = I // 2
        NBI: tl.constexpr = I // 16
        SE: tl.constexpr = ((H + 127) // 128 * 128) * KB_PAD
        t = tl.program_id(0)
        rows = tl.program_id(1) * BH + tl.arange(0, BH)
        kk = tl.arange(0, TKP)
        kv = kk < TOPK
        idv = tl.load(ids_ptr + t * TOPK + kk, mask=kv, other=-1).to(tl.int32)
        okv = (idv >= 0) & (idv < E) & kv
        ev = tl.where(okv, idv, 0)
        scv = tl.load(g2_ptr + ev) * tl.load(tw_ptr + t * TOPK + kk, mask=kv, other=0.0)
        wrow = rows[:, None] * IH + tl.arange(0, IH)[None, :]
        srow = _sf_row(rows, KB_PAD)[:, None] + _sf_col(tl.arange(0, NBI))[None, :]
        for k in tl.static_range(TOPK):  # every expert tile -> L2 before waiting on fc1
            ek = tl.sum(tl.where(kk == k, ev, 0), 0).to(tl.int64)
            _pf_l2(w_ptr + ek * (H * IH) + rows[:, None] * IH + (tl.arange(0, LNI) * 128)[None, :])
        e0 = tl.sum(tl.where(kk == 0, ev, 0), 0).to(tl.int64)
        w_n = tl.load(w_ptr + e0 * (H * IH) + wrow)
        s_n = tl.load(s_ptr + e0 * SE + srow)
        if PDL:
            gdc_wait()  # hq / hsf come from fc1
        acc = tl.zeros([BH], dtype=tl.float32)
        for k in tl.static_range(TOPK):  # ascending k: oxide's combine order
            w_c = w_n
            s_c = s_n
            if k + 1 < TOPK:
                en = tl.sum(tl.where(kk == k + 1, ev, 0), 0).to(tl.int64)
                w_n = tl.load(w_ptr + en * (H * IH) + wrow)
                s_n = tl.load(s_ptr + en * SE + srow)
            p = t * TOPK + k
            hl, hh = _deq32(tl.load(hq_ptr + p * IH + tl.arange(0, IH), cache_modifier=".cg")[None, :],
                            tl.load(hs_ptr + p * NBI + tl.arange(0, NBI), cache_modifier=".cg")[None, :],
                            1, NBI)
            wl, wh = _deq32(w_c, s_c, BH, NBI)
            y = tl.sum(wl * hl + wh * hh, 1)
            ok = tl.sum(tl.where(kk == k, okv.to(tl.int32), 0), 0) > 0
            sc = tl.sum(tl.where(kk == k, scv, 0.0), 0)
            acc = tl.where(ok, acc + y * sc, acc)
        tl.store(out_ptr + t * H + rows, acc.to(tl.bfloat16))

    @triton.jit(do_not_specialize=["M"])
    def _fc2_exp(y_ptr, hq_ptr, hs_ptr, ids_ptr, tw_ptr, pair_ptr, w_ptr, s_ptr, g2_ptr, M,
                 H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 KB_PAD: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, BHC: tl.constexpr,
                 STAGES: tl.constexpr, LNI: tl.constexpr, PDL: tl.constexpr,
                 CHUNK: tl.constexpr):
        IH: tl.constexpr = I // 2
        NBI: tl.constexpr = I // 16
        SE: tl.constexpr = ((H + 127) // 128 * 128) * KB_PAD
        if PDL:
            gdc_launch_dependents()
        e, n_e, off = _route(ids_ptr, M * TOPK, tl.program_id(1), E, CHUNK)
        if n_e == 0:
            return
        h_lo = tl.program_id(0) * BHC
        w_e = w_ptr + e.to(tl.int64) * (H * IH)
        s_e = s_ptr + e.to(tl.int64) * SE
        _pf_l2(w_e + (h_lo + tl.arange(0, BHC))[:, None] * IH + (tl.arange(0, LNI) * 128)[None, :])
        alpha = tl.load(g2_ptr + e)
        cb = tl.arange(0, IH)
        nb = tl.arange(0, NBI)
        if PDL:
            gdc_wait()  # pair list + hq / hsf come from fc1
        for r0 in range(0, n_e, BM):
            rr = r0 + tl.arange(0, BM)
            rv = rr < n_e
            p = tl.load(pair_ptr + off + rr, mask=rv, other=0, cache_modifier=".cg")
            he, ho = _deq16(tl.load(hq_ptr + p[:, None] * IH + cb[None, :], mask=rv[:, None], other=0,
                                    cache_modifier=".cg"),
                            tl.load(hs_ptr + p[:, None] * NBI + nb[None, :], mask=rv[:, None], other=0,
                                    cache_modifier=".cg"),
                            BM, NBI)
            sc = alpha * tl.load(tw_ptr + p, mask=rv, other=0.0)
            for h0 in tl.range(0, BHC, BH, num_stages=STAGES):
                rows = h_lo + h0 + tl.arange(0, BH)
                wl, wh = _deq16(tl.load(w_e + rows[:, None] * IH + cb[None, :]),
                                tl.load(s_e + _sf_row(rows, KB_PAD)[:, None] + _sf_col(nb)[None, :]),
                                BH, NBI)
                yb = tl.dot(he, tl.trans(wl))
                yb = tl.dot(ho, tl.trans(wh), yb)
                tl.store(y_ptr + p.to(tl.int64)[:, None] * H + rows[None, :], yb * sc[:, None],
                         mask=rv[:, None])

    @triton.jit
    def _combine(out_ptr, y_ptr, ids_ptr, H: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 BCOL: tl.constexpr, PDL: tl.constexpr):
        """out[t] = bf16(sum_k y[t * topk + k]) over valid ids, ascending k (oxide moe_combine)."""
        t = tl.program_id(0)
        cols = tl.program_id(1) * BCOL + tl.arange(0, BCOL)
        if PDL:
            gdc_wait()
        acc = tl.zeros([BCOL], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            p = t * TOPK + k
            ek = tl.load(ids_ptr + p)
            ok = (ek >= 0) & (ek < E)
            acc = tl.where(ok, acc + tl.load(y_ptr + p.to(tl.int64) * H + cols, mask=ok, other=0.0,
                                             cache_modifier=".cg"), acc)
        tl.store(out_ptr + t * H + cols, acc.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def layer_state(cfg: dict, info: dict) -> dict:
    """Kernel view of a prepared nvfp4_moe layer cfg (no copies: the layer's own tensors)."""
    e, h, i = info["E"], info["H"], info["I"]
    st = dict(w13=cfg["w13"], w2=cfg["w2"], g1=cfg["g1"], g2=cfg["g2"],
              s13=cfg["w13_sf"].contiguous().view(torch.uint8).reshape(-1),
              s2=cfg["w2_sf"].contiguous().view(torch.uint8).reshape(-1),
              a1g=float(cfg["a1g"]), a2g=float(cfg["a2g"]), act=int(cfg["act"]),
              E=e, H=h, I=i, topk=int(cfg["topk"]), kb13=-(-(h // 16) // 4) * 4,
              kb2=-(-(i // 16) // 4) * 4)
    for k in ("w13", "w2"):
        if st[k].dtype != torch.uint8 or not st[k].is_contiguous():
            raise RuntimeError(f"{MARK} {k}: want contiguous uint8, got {st[k].dtype}")
    if (st["s13"].numel() != e * -(-2 * i // 128) * 128 * st["kb13"]
            or st["s2"].numel() != e * -(-h // 128) * 128 * st["kb2"]):
        raise RuntimeError(f"{MARK} block scales not in the per-expert padded swizzled layout")
    return st


def workspace(dev, p_max: int, h: int, i: int, y: bool) -> dict:
    """hq / hsf rows per pair, the pair list, and (expert-major fc2) fp32 y. Grown, never
    shrunk; serving sizes it at LOAD for max_m (an allocation inside graph capture would
    land in the graph's private pool)."""
    ws = _ST["ws"].get(dev)
    if ws is None or ws["pairs"].numel() < p_max or ws["hq"].shape[1] != i // 2:
        ws = dict(hq=torch.zeros(p_max, i // 2, dtype=torch.uint8, device=dev),
                  hs=torch.zeros(p_max, i // 16, dtype=torch.uint8, device=dev),
                  pairs=torch.zeros(p_max, dtype=torch.int32, device=dev),
                  y=None if ws is None else ws["y"])
        _ST["ws"][dev] = ws
    if y and (ws["y"] is None or ws["y"].shape[0] < p_max or ws["y"].shape[1] != h):
        ws["y"] = torch.empty(p_max, h, dtype=torch.float32, device=dev)
    return ws


def run(st: dict, x, ids, tw, plan: str, pdl: bool | None = None, out=None):
    """One MoE call (bf16 [M, H] out) with `plan` on torch's current stream. ids int32
    [M, topk], tw fp32 [M, topk], x bf16 [M, H], all contiguous."""
    if pdl is None:
        pdl = pdl_on() if _ST["pdl"] is None else bool(_ST["pdl"])
    k1, k2 = PLANS[plan]
    m = x.shape[0]
    e, h, i, topk = st["E"], st["H"], st["I"], st["topk"]
    p = m * topk
    ws = workspace(x.device, p, h, i, k2[0] == "exp")
    if out is None:
        out = torch.empty(m, h, dtype=torch.bfloat16, device=x.device)
    slots = min(e, p)
    shape = dict(H=h, I=i, E=e, TOPK=topk)
    dep = {"launch_pdl": True} if pdl else {}
    lni = max(1, i // 256)  # 128 B lines per w2 row
    if k1[0] == "fma":
        _, r, bn, bk, warps, pf = k1
        _fc1_fma[(i // (bn // 2), slots)](
            x, ids, st["w13"], st["s13"], st["g1"], ws["hq"], ws["hs"], ws["pairs"], m,
            st["a1g"], st["a2g"], KB_PAD=st["kb13"], BN=bn, BK=bk, R=r, PF=pf,
            LN=_p2(h // 256), ACT=st["act"], PDL=pdl, CHUNK=CH, num_warps=warps, **shape)
    else:
        _, bm, bn, bk, warps, stages = k1
        _fc1_mma[(i // (bn // 2), slots)](
            x, ids, st["w13"], st["s13"], st["g1"], ws["hq"], ws["hs"], ws["pairs"], m,
            st["a1g"], st["a2g"], KB_PAD=st["kb13"], BM=bm, BN=bn, BK=bk, STAGES=stages,
            ACT=st["act"], PDL=pdl, CHUNK=CH, num_warps=warps, **shape)
    if k2[0] == "tok":
        _, bh, warps = k2
        _fc2_tok[(m, h // bh)](out, ws["hq"], ws["hs"], ids, tw, st["w2"], st["s2"], st["g2"],
                               TKP=_p2(topk), KB_PAD=st["kb2"], BH=bh, LNI=lni, PDL=pdl,
                               num_warps=warps, **dep, **shape)
    else:
        _, bm, bh, bhc, warps, stages = k2
        bhc = min(bhc, h)
        _fc2_exp[(h // bhc, slots)](ws["y"], ws["hq"], ws["hs"], ids, tw, ws["pairs"], st["w2"],
                                    st["s2"], st["g2"], m, KB_PAD=st["kb2"], BM=bm, BH=bh,
                                    BHC=bhc, STAGES=stages, LNI=lni, PDL=pdl, CHUNK=CH,
                                    num_warps=warps, **dep, **shape)
        bc = min(BC, h)
        _combine[(m, h // bc)](out, ws["y"], ids, H=h, E=e, TOPK=topk, BCOL=bc, PDL=pdl,
                               num_warps=4, **dep)
    return out


class TPLayer:
    """What nvfp4_moe's forward_modular dispatches to: `tp(x, ids, tw)` -> out, or None when
    this M belongs to the oxide kernel (a function of M only: CUDA-graph safe)."""

    __slots__ = ("st", "plans")

    def __init__(self, st, plans):
        self.st, self.plans = st, plans

    def __call__(self, x, ids, tw):
        m = x.shape[0]
        plan = self.plans[m] if m < len(self.plans) else None
        if plan is None:
            return None
        if not _ST["logged"]:
            _ST["logged"] = True
            _log(f"ACTIVE: first MoE call on our TP kernels (M={m}, plan {plan})")
        return run(self.st, x, ids if ids.dtype == torch.int32 else ids.int(), tw, plan)


# ---------------------------------------------------------------------------
# load: plan validation, timing table, per-layer oracle
# ---------------------------------------------------------------------------
def _batch(st, m, dev, seed, hot=False):
    """x at the layer's calibrated activation range + uniform top-k routing (hot: every token
    on the first token's experts, so n_e = M drives the multi-pass row loops)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(m, st["H"], generator=gen) * (2688.0 / st["a1g"] / 4.5)).to(dev, torch.bfloat16)
    ids, tw = nm.rand_routing(m, st["E"], st["topk"], dev, seed=seed)
    if hot:
        ids = ids[:1].expand(m, -1)
    return x, ids.to(torch.int32).contiguous(), tw.float().contiguous()


def check(st, ref_p, plan, m, dev, seed, oxide=None, hot=False, cache=None):
    """One oracle case -> (ok, rel_vs_ref, msg): ours vs the f64 spec <= REF_TOL, finite,
    bitwise deterministic on rerun, and vs the oxide kernel (if given) within the triangle
    bound of both errors. cache: {(m, seed, hot): f64 reference} shared across plans."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    x, ids, tw = _batch(st, m, dev, seed, hot)
    ours = run(st, x, ids, tw, plan)
    again = run(st, x, ids, tw, plan)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    key = (m, seed, hot)
    ref = cache.get(key) if cache is not None else None
    if ref is None:
        ref = nm.moe_ref(ref_p, x, ids, tw)
        if cache is not None:
            cache[key] = ref
    r_ref = nm._rel(ours.float(), ref)
    det = bool(torch.equal(ours.view(torch.int16), again.view(torch.int16)))
    ok = r_ref <= nm.REF_TOL and det and bool(torch.isfinite(ours.float()).all())
    msg = f"rel_vs_ref={r_ref:.2e} deterministic={det}"
    if oxide is not None:
        ox = oxide(x, ids, tw).float()
        r_ox, ox_ref = nm._rel(ours.float(), ox), nm._rel(ox, ref)
        ok = ok and r_ox <= 1.05 * (r_ref + ox_ref) + 1e-6
        msg = msg + f" rel_vs_oxide={r_ox:.2e} oxide_rel_vs_ref={ox_ref:.2e}"
    return ok, r_ref, msg


def graph_check(st, plan, m, dev) -> bool:
    """CUDA-graph replay (PDL edges included) bit-identical to eager; a capture error is False."""
    x, ids, tw = _batch(st, m, dev, 4242)
    try:
        eager = run(st, x, ids, tw, plan).clone()
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            run(st, x, ids, tw, plan)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                out = run(st, x, ids, tw, plan)
        torch.cuda.current_stream(dev).wait_stream(s)
        g.replay()
        torch.cuda.synchronize(dev)
    except Exception as exc:  # noqa: BLE001
        _log(f"graph replay of plan {plan} M={m} failed: {type(exc).__name__}: {exc}"[:300])
        return False
    return bool(torch.equal(out.view(torch.int16), eager.view(torch.int16)))


def graph_gate(st, plans, dev, max_m: int) -> list:
    """Plans whose graphed replay equals eager (M = min(6, max_m)); a plan that only fails with
    PDL turns PDL off for every plan."""
    m = min(6, max_m)
    bad = [p for p in plans if not graph_check(st, p, m, dev)]
    if bad and _ST["pdl"]:
        _ST["pdl"] = False
        if all(graph_check(st, p, m, dev) for p in bad):
            _log(f"graph replay != eager with PDL ({bad}): plain launches from here on")
            return list(plans)
        _ST["pdl"] = True
    for p in bad:
        _log(f"plan {p} REJECTED: CUDA-graph replay != eager (M={m}, PDL {_ST['pdl']})")
    return [p for p in plans if p not in bad]


def _try_plan(st, ref_p, plan, dev, max_m, cache) -> str | None:
    ms = sorted({min(m, max_m) for m in (1, 6, 32)})
    try:
        for m in ms:
            ok, _, msg = check(st, ref_p, plan, m, dev, seed=11 + m, cache=cache)
            if not ok:
                return f"M={m}: {msg}"
        ok, _, msg = check(st, ref_p, plan, ms[-1], dev, seed=7, hot=True, cache=cache)
        return None if ok else f"hot M={ms[-1]}: {msg}"
    except Exception as exc:  # noqa: BLE001 - a plan that fails is never served
        return f"{type(exc).__name__}: {exc}"[:300]


def validate(st, ref_p, dev, max_m: int, plans) -> dict:
    """{plan: None (ok) | reason} on this layer's weights: M 1 / 6 / 32 (<= max_m) + a hot batch.
    A plan that fails with PDL but passes without turns PDL off for every plan."""
    res, cache = {}, {}
    for plan in plans:
        why = _try_plan(st, ref_p, plan, dev, max_m, cache)
        if why is not None and _ST["pdl"]:
            _ST["pdl"] = False
            if _try_plan(st, ref_p, plan, dev, max_m, cache) is None:
                _log(f"plan {plan} fails only with PDL ({why}): plain launches from here on")
                return validate(st, ref_p, dev, max_m, plans)
            _ST["pdl"] = True
        if why is not None:
            _log(f"plan {plan} REJECTED: {why}")
        res[plan] = why
    return res


def time_table(st, oxide, dev, max_m: int, plans, auto: bool, iters: int = TIME_ITERS):
    """Cold-L2 graphed us per TABLE_MS M for every plan (and oxide when auto) -> (table, rows)."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) or (128 << 20)
    buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
    flush = lambda: buf.sum()  # noqa: E731 - every replay starts with a cold L2, like serving
    rows, picks = [], []
    for m in [v for v in TABLE_MS if v <= max_m]:
        x, ids, tw = _batch(st, m, dev, 100 + m)
        t = {p: nm._graph_us(lambda s, p=p: run(st, x, ids, tw, p), dev, iters, flush) for p in plans}
        if auto and oxide is not None:
            t[OXIDE] = nm._graph_us(lambda s: oxide(x, ids, tw), dev, iters, flush)
        picks.append((m, min(t, key=t.get)))
        rows.append((m, t))
    del buf
    torch.cuda.empty_cache()
    return merge_table(picks), rows


def prepare(layer, cfg: dict, info: dict, oxide) -> None:
    """nvfp4_moe._sfx_prepare hook, after the oxide layer oracle (oxide(x, ids, tw) -> out runs
    that kernel): decide the per-M table once per process, run this layer's fail-closed oracle,
    attach cfg["tp_impl"]."""
    mode = impl()
    if mode == OXIDE:
        return
    why = eligibility(cfg, info)
    if why is None and triton is None:
        why = "triton not importable"
    if why is not None:
        if mode == "tp":
            raise RuntimeError(f"{MARK} {IMPL_ENV}=tp but layer {layer.layer_name}: {why}")
        _log(f"layer {layer.layer_name} stays on the oxide kernel: {why}")
        return
    dev = cfg["w13"].device
    st = layer_state(cfg, info)
    max_m = int(cfg["max_m"])
    if _ST["table"] is None:
        _ST["pdl"] = pdl_on()
        env = os.environ.get(TABLE_ENV, "").strip()
        table = parse_table(env) if env else None
        need = sorted({p for _, p in table if p != OXIDE}) if table else list(PLANS)
        res = validate(st, cfg, dev, max_m, need)
        good = [p for p in need if res[p] is None]
        if good and dev.type == "cuda":
            good = graph_gate(st, good, dev, max_m)
        if table is not None and len(good) != len(need):
            raise RuntimeError(f"{MARK} {TABLE_ENV} plans failed validation: "
                               f"{sorted(set(need) - set(good))} ({ {p: r for p, r in res.items() if r} })")
        if not good:
            if mode == "tp":
                raise RuntimeError(f"{MARK} {IMPL_ENV}=tp: every plan failed validation: {res}")
            table = [(256, OXIDE)]
            _log("every plan failed validation: the oxide kernel serves every M")
        elif table is None:
            table, rows = time_table(st, oxide, dev, max_m, good, auto=mode == "auto")
            for m, t in rows:
                _log(f"load timing M={m}: " + ", ".join(
                    f"{k} {v:.1f}" for k, v in sorted(t.items(), key=lambda kv: kv[1]))
                    + " us (cold L2, graphed)")
        if mode == "tp" and any(p == OXIDE for _, p in table):
            raise RuntimeError(f"{MARK} {IMPL_ENV}=tp with oxide buckets in {TABLE_ENV}: use auto")
        _ST["table"] = table
        _log(f"routing table ({mode}, PDL {'on' if _ST['pdl'] else 'off'}): "
             f"{TABLE_ENV}={format_table(table)}")
    plans = by_m(_ST["table"], max_m)
    workspace(dev, max_m * st["topk"], st["H"], st["I"],
              any(PLANS[p][1][0] == "exp" for p in plans[1:] if p))
    oracle_layer(layer, st, cfg, plans, oxide, dev, max_m)
    cfg["tp_impl"] = TPLayer(st, plans)
    _ST["layers"] += 1


def oracle_layer(layer, st, cfg, plans, oxide, dev, max_m: int) -> None:
    """Fail-closed on this layer's real weights: every ORACLE_MS M <= max_m that the table gives
    to us + a hot-expert batch, vs the f64 reference and the oxide kernel."""
    worst, msgs, cache = 0.0, [], {}
    seed0 = zlib.crc32(str(layer.layer_name).encode()) % 10007
    cases = [(m, False) for m in ORACLE_MS if m <= max_m and plans[m]]
    hot_m = min(32, max_m)
    if plans[hot_m]:
        cases.append((hot_m, True))
    for m, hot in cases:
        ok, r_ref, msg = check(st, cfg, plans[m], m, dev, seed0 + m, oxide=oxide, hot=hot,
                               cache=cache)
        msgs.append(f"M={m}{' hot' if hot else ''} {plans[m]}: {msg}")
        worst = max(worst, r_ref)
        if not ok:
            raise RuntimeError(f"{MARK} LAYER ORACLE FAIL {layer.layer_name} {msgs[-1]} — "
                               f"refusing to serve with {IMPL_ENV}={impl()}")
    _log(f"LAYER ORACLE PASS {layer.layer_name} max_rel_vs_ref={worst:.2e} ({'; '.join(msgs)})")


# ---------------------------------------------------------------------------
# boot gate glm_moe_tp_kernel_bench (synthetic GLM-5.3 TP8 shard, one GPU)
# ---------------------------------------------------------------------------
def _gate_problem(dev):
    from suffix_hybrid.kernels import nvfp4_moe as nm
    p = nm._dev_problem(dev, nm.GLM_TP8, seed=1)
    e, two_i, kh = p["w13"].shape
    info = dict(E=e, H=kh * 2, I=two_i // 2, ep=1)
    p.update(topk=nm.GLM_TP8[3], E_global=e, id_base=0)
    return p, layer_state(p, info)


def gate_oracle(dev, p, st, native, max_m: int) -> dict:
    """{plan: passed} — every plan at ORACLE_MS + a hot batch vs f64 and oxide, graph at M=6."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    ws = nm.workspace(dev, max_m, st["topk"], st["H"], st["I"], st["E"])
    p["tune"] = nm.parse_tune(G10_TUNE)
    oxide = lambda x, i, w: nm.run_ours(native, p, x, i, w, ws,  # noqa: E731
                                        torch.cuda.current_stream(dev).cuda_stream)
    cache, res = {}, {}
    for plan in PLANS:
        res[plan] = True
        for m in [v for v in ORACLE_MS if v <= max_m] + [-min(32, max_m)]:
            try:
                ok, _, msg = check(st, p, plan, abs(m), dev, seed=31 + abs(m), oxide=oxide,
                                   hot=m < 0, cache=cache)
                if ok and m == 6:
                    ok = graph_check(st, plan, 6, dev)
                    msg = msg + f" graph_replay_bitwise={ok}"
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{type(exc).__name__}: {exc}"[:400]
            res[plan] = res[plan] and ok
            _log(f"oracle plan {plan} M={abs(m)}{' hot' if m < 0 else ''}: {msg} "
                 f"{'OK' if ok else 'FAIL'}")
    return res


def gate_bench(dev, p, st, native, max_m: int, plans, iters: int = 30):
    """us/layer per M (cold L2, graphed): every oracle-passing plan, the winner with PDL off, oxide
    at the g10 table, FlashInfer CUTLASS; % of the weight roofline; the paste-able table."""
    from suffix_hybrid.kernels import nvfp4_moe as nm
    ws = nm.workspace(dev, max_m, st["topk"], st["H"], st["I"], st["E"])
    p["tune"] = nm.parse_tune(G10_TUNE)
    oxide = lambda x, i, w, o=None: nm.run_ours(native, p, x, i, w, ws,  # noqa: E731
                                                torch.cuda.current_stream(dev).cuda_stream, o)
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) or (128 << 20)
    buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
    flush = lambda: buf.sum()  # noqa: E731
    us = lambda fn: nm._graph_us(fn, dev, iters, flush)  # noqa: E731
    picks, picks_tp = [], []
    for m in [v for v in TABLE_MS if v <= max_m]:
        x, ids, tw = _batch(st, m, dev, 100 + m)
        out = torch.empty(m, st["H"], dtype=torch.bfloat16, device=dev)
        t = {}
        for plan in plans:
            try:
                t[plan] = us(lambda s, plan=plan: run(st, x, ids, tw, plan, out=out))
            except Exception as exc:  # noqa: BLE001
                _log(f"bench M={m} plan {plan} FAILED: {type(exc).__name__}: {exc}"[:300])
        best = min(t, key=t.get) if t else None
        t_nopdl = us(lambda s: run(st, x, ids, tw, best, pdl=False, out=out)) if best else float("nan")
        t_ox = us(lambda s: oxide(x, ids, tw, out))
        try:
            t_fi = us(lambda s: nm._fi_call(p, x, ids, tw, out))
        except Exception as exc:  # noqa: BLE001
            t_fi = float("nan")
            _log(f"bench M={m} flashinfer FAILED: {type(exc).__name__}: {exc}"[:200])
        hit = int(torch.unique(ids).numel())
        roof = roofline_bytes(hit, st["H"], st["I"]) / nm.HBM_BPS * 1e6
        tb = t[best] if best else float("nan")
        picks_tp.append((m, best))
        picks.append((m, best if best and tb < t_ox else OXIDE))
        _log(f"bench M={m:3d} experts_hit={hit:3d} weights {roofline_bytes(hit, st['H'], st['I']) / 1e6:6.1f} MB "
             f"roofline {roof:6.1f} us | ours {best} {tb:7.1f} us ({100 * roof / tb:3.0f}% BW; "
             f"PDL off {t_nopdl:7.1f}) | oxide g10 {t_ox:7.1f} | flashinfer {t_fi:7.1f} | "
             f"x{t_ox / tb:4.2f} vs oxide | plans " + " ".join(f"{k} {v:.1f}" for k, v in t.items()))
    table = merge_table(picks)
    mode = "tp" if all(p_ != OXIDE for _, p_ in table) else "auto"
    _log(f"routing table: {TABLE_ENV}={format_table(table)} {IMPL_ENV}={mode}")
    if mode == "auto":
        _log(f"ours everywhere: {TABLE_ENV}={format_table(merge_table(picks_tp))} {IMPL_ENV}=tp")
    return table


def main(argv=None) -> int:
    from suffix_hybrid.kernels import nvfp4_moe as nm
    mode = (argv or sys.argv[1:] or ["bench"])[0]
    try:
        if triton is None:
            raise RuntimeError("triton not importable")
        native = nm._native_ready()
        dev = torch.device("cuda", torch.cuda.current_device())
        _ST["pdl"] = pdl_on()
        max_m = 192
        p, st = _gate_problem(dev)
        _log(f"GLM-5.3 TP8 shard: E={st['E']} H={st['H']} I={st['I']} top{st['topk']}, "
             f"{torch.cuda.get_device_name(dev)}, PDL {'on' if _ST['pdl'] else 'off'}")
        res = gate_oracle(dev, p, st, native, max_m)
        good = [k for k, v in res.items() if v]
        _log(f"ORACLE {'PASS' if len(good) == len(res) else 'FAIL'}: plans passing {good}, failing "
             f"{[k for k, v in res.items() if not v]} (vs f64 reference + oxide, deterministic, "
             "graph replay)")
        if mode == "bench" and good:
            gate_bench(dev, p, st, native, max_m, good)
        return 0 if len(good) == len(res) else 1
    except Exception as exc:  # noqa: BLE001
        _log(f"{mode.upper()} FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
