# SPDX-License-Identifier: Apache-2.0
"""Helion (-> Triton) sparse-MLA attention for DeepSeek-V4.1-Flash on SM120.

Replaces flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4 (SM120 "sparse"
backend) in vLLM's DeepseekV4FlashInferSM120Attention._forward_decode /
_forward_prefill, for decode rows (spec decode: 1..6 q rows per sequence,
batch <= 32) and prefill chunks (<= 8192 tokens). The math, the record
layouts, the sink and the empty-row rules are the K1 contract in
dsv41_attn_ref.py (docs/dsv41-attn.md has the derivations and the cost model).

Three Helion functions share the tile arithmetic. NONE has any cross-CTA
synchronisation: no grid barrier (``hl.barrier`` makes Helion emit a persistent
kernel with an in-kernel barrier: it deadlocked on silicon), no semaphore, no
last-arriving-split merge (atomics make graph replays stateful). Dependent
stages are separate Helion functions = separate launches; kernel boundaries are
the only ordering.

dsv41_attn_partial program = (token t, split s): split s walks the s-th slice of
    the SWA list, then the s-th slice of the compressed list, with ONE online
    softmax state (a split mixes both pools) and writes a bf16 partial + f32
    base-2 lse (allocated inside, torch's allocator). Decode rows and short
    prefill chunks (T <= 1024); the split count is a tunable (LFBO searches it
    per T bucket and per context class).
dsv41_attn_merge   program = token t: log-sum-exp merge of the S partials AND
    the sink into ``out`` (reads S x 8 KB per token; fixed default config).
dsv41_attn_row     ONE kernel, program = token t, no split, no partials: the
    prefill body (T >= 256 already fills the SMs) and the decode alternative
    when the partials' traffic loses.

Numerics (K3 reader gate: row rel-L2 <= 2e-2 vs the exact f64 reference; the
FlashInfer SM120 decode quantizes q and P to fp8, this does not): every
dequantized K element is EXACT in bf16 (fp8/e2m1/e4m3 value x 2^k), so QK uses
the bf16 query and bf16 K with f32 accumulation; P is rounded to bf16 for P.V
(V == K, also exact); softmax state, lse and merge in f32; one bf16 output
rounding (+ one bf16 rounding of each split partial in split mode). Masked / -1
columns gather as zero bytes (stale NaN bytes cannot reach P.V) and get S = -inf.
Deterministic: same inputs -> same bits (no atomics, fixed reduction order), so
eager and CUDA-graph replays can be compared bit-exactly (the boot gate does).

Memory and registers (the lessons of the glm-int Triton MLA: 255 regs / 78-138
spills at rev 1): pools are read through ROW views ([rows, 576] fp8+rope bytes,
[rows, 8] scales, an int16 view of the rope; [rows, 528]; [rows, 288]) so every
gather is a contiguous 16-byte-vectorizable row slice, not a byte gather; the K
tile is dequantized once, in bf16 (no f32 [bk, 512] tile), and q is loaded once
as the A operand; the tile (bk 16..64), warps and stages are searched, small
tiles first. The gathers are plain loads inside a ``tl.range`` loop: the lab
searches ``num_stages`` / ``range_num_stages`` so Triton's pipeliner overlaps the
next tile's loads with this tile's math (cp.async needs a silicon check, see docs).

Capture safety: scratch is ``torch.empty`` inside the Helion function (torch's
allocator, graph-pool safe) or load-time persistent buffers (``route``); no host
sync, no H2D copy, no first-use state binding on the call path. Every Triton
specialization a call can hit (T == 1, T % 16, pool strides % 16) is compiled at
load or on the first EAGER call; ``attend`` refuses (loudly) to be the first call
of a specialization while a CUDA graph is being captured.

Serving: ``SUFFIX_HELION_DSV41_ATTN`` = oxide|helion|auto (the repo-wide knob;
"oxide" here means "FlashInfer, this module never imported"), resolved once per
layer geometry by ``load``. Configs: configs/helion/<kernel>/<platform>.json
(kernels dsv41_attn_partial, dsv41_attn_row; the merge uses a fixed default config), shipped only for lab WINS vs the
FlashInfer baseline.
"""
from __future__ import annotations

import logging
import os

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_attn_ref as R
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU CI / console: the lab ABI must still import
    helion = hl = PowerOfTwoFragment = None

KNOB = "SUFFIX_HELION_DSV41_ATTN"
ROUTE_ENV = "SUFFIX_DSV41_ATTN_ROUTE"  # auto (default) | row
PARTIAL, MERGE, ROW = "dsv41_attn_partial", "dsv41_attn_merge", "dsv41_attn_row"
ARCHS = ("DeepseekV41ForCausalLM",)
SETTINGS = {"index_dtype": torch.int64}  # KV pools exceed 2^31 bytes
DECODE_BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
PARTIAL_BUCKETS = DECODE_BUCKETS + (256, 512, 1024)
ROW_BUCKETS = DECODE_BUCKETS + (256, 512, 1024, 2048, 4096, 8192)
# Context classes: the decode config is searched per class because the pool's
# footprint decides L2-resident vs DRAM-random gathers (8k ctx = a few MB of
# compressed rows per layer, 512k ctx = 150-300 MB). The class is a pure
# PERFORMANCE hint (host-known max_seq_len of the step); every class's config is
# correct for every context. Only decode buckets of layers with a compressed
# pool are class-specific; everything else uses "any".
CTX_CLASSES = (("c16k", 16384), ("c128k", 131072), ("c512k", 1 << 40))
LOG2E = 1.4426950408889634
QK_SCALE_LOG2 = 0.044194173824159216 * LOG2E  # 512 ** -0.5 * log2(e), asserted against R.SCALE below
assert abs(QK_SCALE_LOG2 - R.SCALE * R.LOG2E) < 1e-15


def buckets(kernel: str) -> tuple:
    return PARTIAL_BUCKETS if kernel == PARTIAL else ROW_BUCKETS


def ctx_class(max_seq_len) -> str:
    """Context class of a step from the host-known max sequence length (None: the
    largest class, the worst case)."""
    if max_seq_len is None:
        return CTX_CLASSES[-1][0]
    for name, hi in CTX_CLASSES:
        if int(max_seq_len) <= hi:
            return name
    return CTX_CLASSES[-1][0]


def case_ctx(t: int, topk: int, ctx: str) -> str:
    """Config-key ctx field: only decode buckets of compressed layers are class-specific."""
    return ctx if (topk > 0 and t <= DECODE_BUCKETS[-1]) else "any"


# ---------------------------------------------------------------------------
# tile arithmetic (module level, undecorated: inlined into both bodies).
# Helion forbids indexing HOST tensors inside helpers, so every pool gather is
# written inline in the bodies; the helpers only see device values.
# ---------------------------------------------------------------------------
def _pow2(byte):
    """UE8M0 byte -> 2^(byte-127) as f32 by exponent placement (exact for
    bytes 1..254; the writers never emit 0 / 255 for finite data)."""
    return (byte.to(torch.int32) << 23).view(torch.float32)


def _e2m1_bf16(code):
    """e2m1 codes (int32 0..15, sign bit 3) -> exact bf16 by bit construction."""
    m = code & 7
    bits = torch.where(m >= 2, (((m >> 1) + 126) << 7) | ((m & 1) << 6), m * 0x3F00)
    return (bits | ((code & 8) << 12)).to(torch.int16).view(torch.bfloat16)


def _step(q16, k16, live, m_i, l_i, acc):
    """One online-softmax step over a [16, bk] score tile (exp2 domain). No
    ``x[None, :]`` subscripts here: helpers are traced as plain torch, where
    those lower to aten.slice (rejected by Helion's Inductor lowering)."""
    sc = hl.dot(q16, k16.T, out_dtype=torch.float32)
    sc = torch.where(live.unsqueeze(0), sc * QK_SCALE_LOG2, float("-inf"))
    m_new = torch.maximum(m_i, torch.amax(sc, dim=1))
    m_ref = torch.where(m_new == float("-inf"), 0.0, m_new)  # all masked so far: exp2(-inf) = 0
    alpha = torch.exp2(m_i - m_ref)
    p = torch.exp2(sc - m_ref.unsqueeze(1))
    l_i = l_i * alpha + torch.sum(p, dim=1)
    acc = hl.dot(p.to(torch.bfloat16), k16, acc=acc * alpha.unsqueeze(1))
    return m_new, l_i, acc


# The pool arguments of the bodies (per pool: d, d16, s, shift, bd, bs):
#   REC_V4   d   uint8 [rows576, 576]   fp8 NoPE [0,448) + bf16 RoPE [448,576) per token
#            d16 int16 [rows576, 288]   the same bytes as int16 (RoPE read as 64 contiguous int16)
#            s   uint8 [rows8, 8]       the scale footers (7 UE8M0 + pad per token)
#            token slot -> d row blk*bd + r ; scale row blk*bs + page*72 + r
#   REC_FP8  d   uint8 [rows528, 528]   512 fp8 then 16 UE8M0 per token (d16 / s: unused dummies)
#   REC_FP4  d   uint8 [rows288, 288]   256 e2m1 pairs then 32 e4m3 per token
#   shift = log2(page tokens); bd = block stride in d rows; bs = block stride in s rows (V4).
# ``swa_`` / ``cmp_`` prefixes name the two pools; recs: SWA in {0, 1}, compressed in {0, 2}.
POOL_ARGS = ("q", "swa_d", "swa_d16", "swa_s", "cmp_d", "cmp_d16", "cmp_s", "swa_idx", "swa_len",
             "cmp_idx", "cmp_len", "sink")          # dsv41_attn_partial(*POOL_ARGS, *SCALAR_ARGS)
SCALAR_ARGS = ("swa_shift", "swa_bd", "swa_bs", "cmp_shift", "cmp_bd", "cmp_bs", "swa_rec", "cmp_rec")
# dsv41_attn_row(*POOL_ARGS, out, *SCALAR_ARGS)


# ---------------------------------------------------------------------------
# body 1: split-KV decode, stage 1 of 2 (partials) + the merge function
# ---------------------------------------------------------------------------
def dsv41_attn_partial(q: torch.Tensor, swa_d: torch.Tensor, swa_d16: torch.Tensor, swa_s: torch.Tensor,
                     cmp_d: torch.Tensor, cmp_d16: torch.Tensor, cmp_s: torch.Tensor,
                     swa_idx: torch.Tensor, swa_len: torch.Tensor, cmp_idx: torch.Tensor,
                     cmp_len: torch.Tensor, sink: torch.Tensor, swa_shift: int,
                     swa_bd: int, swa_bs: int, cmp_shift: int, cmp_bd: int, cmp_bs: int,
                     swa_rec: int, cmp_rec: int):
    """q [T, H, 512] bf16; pools as the comment above describes; *_idx int32
    [T, C] global slots (-1 masked) with *_len int32 [T] live prefix lengths (a
    ratio-0 layer passes cmp_len = 0 and any [T, 1] cmp_idx); sink f32 [H]
    (-inf = none; passed through for the merge). Returns (o_part bf16 [T, H, S,
    512], lse_part f32 [T, H, S], sink). Config: block_sizes=[bk], num_splits,
    num_warps, num_stages, indexing, pid layout, per-loop range_* options."""
    n_tok = q.size(0)
    heads = hl.specialize(q.size(1))
    s_cap = hl.specialize(swa_idx.size(1))
    c_cap = hl.specialize(cmp_idx.size(1))
    s_shift = hl.specialize(swa_shift)
    c_shift = hl.specialize(cmp_shift)
    s_rec = hl.specialize(swa_rec)
    c_rec = hl.specialize(cmp_rec)
    num_splits = hl.register_tunable("num_splits", PowerOfTwoFragment(1, 32, 8))
    bk = hl.register_block_size(16, 64)
    per_s = helion.next_power_of_2(helion.cdiv(s_cap, num_splits))
    per_c = helion.next_power_of_2(helion.cdiv(c_cap, num_splits))
    o_part = torch.empty([n_tok, heads, num_splits, 512], dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty([n_tok, heads, num_splits], dtype=torch.float32, device=q.device)
    for t, s in hl.grid([n_tok, num_splits]):
        n_s = torch.clamp(swa_len[t], min=0, max=s_cap)
        n_c = torch.clamp(cmp_len[t], min=0, max=c_cap)
        a0 = s * per_s
        a1 = torch.clamp(n_s, max=a0 + per_s)
        b0 = s * per_c
        b1 = torch.clamp(n_c, max=b0 + per_c)
        for tile_h in hl.tile(heads, block_size=16):
            q16 = q[t, tile_h, :]
            m_i = hl.full([tile_h], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_h], dtype=torch.float32)
            acc = hl.zeros([tile_h, 512], dtype=torch.float32)
            for tile_k in hl.tile(a0, a1, block_size=bk):
                slot = swa_idx[t, tile_k]
                live = (slot >= 0) & (tile_k.index < a1)
                s64 = torch.clamp(slot, min=0).to(torch.int64)
                blk = s64 >> s_shift
                r = s64 & ((1 << s_shift) - 1)
                kp = live.to(torch.uint8)
                rowd = blk * swa_bd + r
                if s_rec == 0:  # V4 584 B: fp8 NoPE + bf16 RoPE, scales in the page footer
                    j4 = hl.arange(8)
                    e4 = hl.arange(64)
                    nb4 = swa_d[rowd[:, None, None], (j4 * 64)[None, :, None] + e4[None, None, :]] * kp[:, None, None]
                    sb4 = swa_s[(blk * swa_bs + (72 << s_shift) + r)[:, None], j4[None, :]] * kp[:, None]
                    rp4 = swa_d16[rowd[:, None], 224 + e4[None, :]] * kp[:, None].to(torch.int16)
                    nope4 = nb4.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb4).to(torch.bfloat16)[:, :, None]
                    # block 7 = the 64 RoPE dims (its pad scale byte is 0; nope4[:, 7] is dropped)
                    k4 = torch.where((j4 == 7)[None, :, None], rp4.view(torch.bfloat16)[:, None, :], nope4)
                    k16 = k4.reshape(k4.size(0), 512)
                else:  # V4.1 MXFP8 528 B: 512 fp8 (RoPE included) + 16 UE8M0 per token
                    j8 = hl.arange(16)
                    e8 = hl.arange(32)
                    nb8 = swa_d[rowd[:, None, None], (j8 * 32)[None, :, None] + e8[None, None, :]] * kp[:, None, None]
                    sb8 = swa_d[rowd[:, None], 512 + j8[None, :]] * kp[:, None]
                    k8 = nb8.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb8).to(torch.bfloat16)[:, :, None]
                    k16 = k8.reshape(k8.size(0), 512)
                m_i, l_i, acc = _step(q16, k16, live, m_i, l_i, acc)
            for tile_k in hl.tile(b0, b1, block_size=bk):
                slot = cmp_idx[t, tile_k]
                live = (slot >= 0) & (tile_k.index < b1)
                s64 = torch.clamp(slot, min=0).to(torch.int64)
                blk = s64 >> c_shift
                r = s64 & ((1 << c_shift) - 1)
                kp = live.to(torch.uint8)
                rowd = blk * cmp_bd + r
                if c_rec == 0:  # V4 584 B: fp8 NoPE + bf16 RoPE, scales in the page footer
                    j4 = hl.arange(8)
                    e4 = hl.arange(64)
                    nb4 = cmp_d[rowd[:, None, None], (j4 * 64)[None, :, None] + e4[None, None, :]] * kp[:, None, None]
                    sb4 = cmp_s[(blk * cmp_bs + (72 << c_shift) + r)[:, None], j4[None, :]] * kp[:, None]
                    rp4 = cmp_d16[rowd[:, None], 224 + e4[None, :]] * kp[:, None].to(torch.int16)
                    nope4 = nb4.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb4).to(torch.bfloat16)[:, :, None]
                    # block 7 = the 64 RoPE dims (its pad scale byte is 0; nope4[:, 7] is dropped)
                    k4 = torch.where((j4 == 7)[None, :, None], rp4.view(torch.bfloat16)[:, None, :], nope4)
                    k16 = k4.reshape(k4.size(0), 512)
                else:  # V4.1 NVFP4 288 B: 256 e2m1 pairs (even dim = low nibble) + 32 e4m3 per token
                    jq = hl.arange(32)
                    nbq = cmp_d[rowd[:, None], hl.arange(256)[None, :]] * kp[:, None]     # [bk, 256]: one 256 B run per row
                    sfq = cmp_d[rowd[:, None], 256 + jq[None, :]] * kp[:, None]
                    codes = hl.join(nbq & 15, nbq >> 4)  # [bk, 256, 2]: dim 2i + {0, 1} (even dim = low nibble)
                    vq = _e2m1_bf16(codes.to(torch.int32)).reshape(codes.size(0), 32, 8, 2)   # 16 dims per scale
                    kq = vq * sfq.view(torch.float8_e4m3fn).to(torch.bfloat16)[:, :, None, None]
                    k16 = kq.reshape(kq.size(0), 512)
                m_i, l_i, acc = _step(q16, k16, live, m_i, l_i, acc)
            l_ok = l_i > 0.0
            l_one = torch.where(l_ok, l_i, 1.0)
            o_part[t, tile_h, s, :] = (acc / l_one[:, None]).to(torch.bfloat16)
            lse_part[t, tile_h, s] = torch.where(l_ok, m_i + torch.log2(l_one), float("-inf"))
    return o_part, lse_part, sink


def dsv41_attn_merge(o_part: torch.Tensor, lse_part: torch.Tensor, sink: torch.Tensor,
                     out: torch.Tensor) -> torch.Tensor:
    """Stage 2: out [T, H, 512] bf16 = log-sum-exp merge of the partials and the
    sink (a sink column of weight exp(sink), zero value). Fixed default config
    (block_sizes=[512], 4 warps): the traffic is S x 8 KB per token."""
    n_tok = o_part.size(0)
    heads = hl.specialize(o_part.size(1))
    n_split = hl.specialize(o_part.size(2))
    bd = hl.register_block_size(128, 512)
    for t in hl.grid(n_tok):
        for tile_g in hl.tile(heads, block_size=16):
            snk = sink[tile_g] * LOG2E
            for tile_d in hl.tile(512, block_size=bd):
                mg_m = snk
                mg_l = torch.where(snk == float("-inf"), 0.0, 1.0)
                mg_o = hl.zeros([tile_g, tile_d], dtype=torch.float32)
                for s2 in hl.grid(n_split):
                    lse = lse_part[t, tile_g, s2]
                    mg_new = torch.maximum(mg_m, lse)
                    mg_ref = torch.where(mg_new == float("-inf"), 0.0, mg_new)
                    mg_a = torch.exp2(mg_m - mg_ref)
                    mg_w = torch.exp2(lse - mg_ref)
                    mg_l = mg_l * mg_a + mg_w
                    mg_o = mg_o * mg_a[:, None] + o_part[t, tile_g, s2, tile_d].to(torch.float32) * mg_w[:, None]
                    mg_m = mg_new
                mg_div = mg_l + (mg_l == 0.0).to(torch.float32)  # nothing live, no sink: exact 0
                out[t, tile_g, tile_d] = (mg_o / mg_div[:, None]).to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# body 2: one program per token (prefill; decode alternative)
# ---------------------------------------------------------------------------
def dsv41_attn_row(q: torch.Tensor, swa_d: torch.Tensor, swa_d16: torch.Tensor, swa_s: torch.Tensor,
                   cmp_d: torch.Tensor, cmp_d16: torch.Tensor, cmp_s: torch.Tensor,
                   swa_idx: torch.Tensor, swa_len: torch.Tensor, cmp_idx: torch.Tensor,
                   cmp_len: torch.Tensor, sink: torch.Tensor, out: torch.Tensor, swa_shift: int,
                   swa_bd: int, swa_bs: int, cmp_shift: int, cmp_bd: int, cmp_bs: int,
                   swa_rec: int, cmp_rec: int) -> torch.Tensor:
    """Same args as dsv41_attn_partial plus ``out`` after ``sink``. Config: block_sizes=[bk], num_warps,
    num_stages, indexing, pid layout, per-loop range_* options. Adjacent tokens
    run as adjacent programs: their SWA windows (127/128 shared) hit L2."""
    n_tok = q.size(0)
    heads = hl.specialize(q.size(1))
    s_cap = hl.specialize(swa_idx.size(1))
    c_cap = hl.specialize(cmp_idx.size(1))
    s_shift = hl.specialize(swa_shift)
    c_shift = hl.specialize(cmp_shift)
    s_rec = hl.specialize(swa_rec)
    c_rec = hl.specialize(cmp_rec)
    bk = hl.register_block_size(16, 64)
    for t in hl.grid(n_tok):
        n_s = torch.clamp(swa_len[t], min=0, max=s_cap)
        n_c = torch.clamp(cmp_len[t], min=0, max=c_cap)
        for tile_h in hl.tile(heads, block_size=16):
            q16 = q[t, tile_h, :]
            m_i = hl.full([tile_h], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_h], dtype=torch.float32)
            acc = hl.zeros([tile_h, 512], dtype=torch.float32)
            for tile_k in hl.tile(0, n_s, block_size=bk):
                slot = swa_idx[t, tile_k]
                live = (slot >= 0) & (tile_k.index < n_s)
                s64 = torch.clamp(slot, min=0).to(torch.int64)
                blk = s64 >> s_shift
                r = s64 & ((1 << s_shift) - 1)
                kp = live.to(torch.uint8)
                rowd = blk * swa_bd + r
                if s_rec == 0:  # V4 584 B: fp8 NoPE + bf16 RoPE, scales in the page footer
                    j4 = hl.arange(8)
                    e4 = hl.arange(64)
                    nb4 = swa_d[rowd[:, None, None], (j4 * 64)[None, :, None] + e4[None, None, :]] * kp[:, None, None]
                    sb4 = swa_s[(blk * swa_bs + (72 << s_shift) + r)[:, None], j4[None, :]] * kp[:, None]
                    rp4 = swa_d16[rowd[:, None], 224 + e4[None, :]] * kp[:, None].to(torch.int16)
                    nope4 = nb4.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb4).to(torch.bfloat16)[:, :, None]
                    # block 7 = the 64 RoPE dims (its pad scale byte is 0; nope4[:, 7] is dropped)
                    k4 = torch.where((j4 == 7)[None, :, None], rp4.view(torch.bfloat16)[:, None, :], nope4)
                    k16 = k4.reshape(k4.size(0), 512)
                else:  # V4.1 MXFP8 528 B: 512 fp8 (RoPE included) + 16 UE8M0 per token
                    j8 = hl.arange(16)
                    e8 = hl.arange(32)
                    nb8 = swa_d[rowd[:, None, None], (j8 * 32)[None, :, None] + e8[None, None, :]] * kp[:, None, None]
                    sb8 = swa_d[rowd[:, None], 512 + j8[None, :]] * kp[:, None]
                    k8 = nb8.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb8).to(torch.bfloat16)[:, :, None]
                    k16 = k8.reshape(k8.size(0), 512)
                m_i, l_i, acc = _step(q16, k16, live, m_i, l_i, acc)
            for tile_k in hl.tile(0, n_c, block_size=bk):
                slot = cmp_idx[t, tile_k]
                live = (slot >= 0) & (tile_k.index < n_c)
                s64 = torch.clamp(slot, min=0).to(torch.int64)
                blk = s64 >> c_shift
                r = s64 & ((1 << c_shift) - 1)
                kp = live.to(torch.uint8)
                rowd = blk * cmp_bd + r
                if c_rec == 0:  # V4 584 B: fp8 NoPE + bf16 RoPE, scales in the page footer
                    j4 = hl.arange(8)
                    e4 = hl.arange(64)
                    nb4 = cmp_d[rowd[:, None, None], (j4 * 64)[None, :, None] + e4[None, None, :]] * kp[:, None, None]
                    sb4 = cmp_s[(blk * cmp_bs + (72 << c_shift) + r)[:, None], j4[None, :]] * kp[:, None]
                    rp4 = cmp_d16[rowd[:, None], 224 + e4[None, :]] * kp[:, None].to(torch.int16)
                    nope4 = nb4.view(torch.float8_e4m3fn).to(torch.bfloat16) * _pow2(sb4).to(torch.bfloat16)[:, :, None]
                    # block 7 = the 64 RoPE dims (its pad scale byte is 0; nope4[:, 7] is dropped)
                    k4 = torch.where((j4 == 7)[None, :, None], rp4.view(torch.bfloat16)[:, None, :], nope4)
                    k16 = k4.reshape(k4.size(0), 512)
                else:  # V4.1 NVFP4 288 B: 256 e2m1 pairs (even dim = low nibble) + 32 e4m3 per token
                    jq = hl.arange(32)
                    nbq = cmp_d[rowd[:, None], hl.arange(256)[None, :]] * kp[:, None]     # [bk, 256]: one 256 B run per row
                    sfq = cmp_d[rowd[:, None], 256 + jq[None, :]] * kp[:, None]
                    codes = hl.join(nbq & 15, nbq >> 4)  # [bk, 256, 2]: dim 2i + {0, 1} (even dim = low nibble)
                    vq = _e2m1_bf16(codes.to(torch.int32)).reshape(codes.size(0), 32, 8, 2)   # 16 dims per scale
                    kq = vq * sfq.view(torch.float8_e4m3fn).to(torch.bfloat16)[:, :, None, None]
                    k16 = kq.reshape(kq.size(0), 512)
                m_i, l_i, acc = _step(q16, k16, live, m_i, l_i, acc)
            snk = sink[tile_h] * LOG2E
            m_all = torch.maximum(m_i, snk)
            m_ref = torch.where(m_all == float("-inf"), 0.0, m_all)
            a = torch.exp2(m_i - m_ref)
            den = l_i * a + torch.exp2(snk - m_ref)
            den = den + (den == 0.0).to(torch.float32)  # nothing live, no sink: exact 0
            out[t, tile_h, :] = (acc * a[:, None] / den[:, None]).to(out.dtype)
    return out


_BODY = {PARTIAL: dsv41_attn_partial, ROW: dsv41_attn_row}


# ---------------------------------------------------------------------------
# pools -> row views (K1 contract at the call boundary)
# ---------------------------------------------------------------------------
def rec_label(swa_rec: int, cmp_rec: int, topk: int = R.TOPK) -> str:
    """Config-key record label: "v4" = today's 584 B pair, "v41" = the vLLM
    nvfp4_ds_mla pair (528 B SWA + 288 B compressed). Ratio-0 layers (topk 0)
    have no compressed pool: the SWA record alone names them."""
    if (swa_rec, cmp_rec if topk else swa_rec) in ((R.REC_V4, R.REC_V4), (R.REC_FP8, R.REC_FP4),
                                                   (R.REC_FP8, R.REC_FP8)):
        return "v4" if swa_rec == R.REC_V4 else "v41"
    raise RuntimeError(f"{hk.MARKER} dsv41_attn: unsupported record pair "
                       f"({R.REC_NAME[swa_rec]}, {R.REC_NAME[cmp_rec]})")


def case(numtokens: int, heads: int, swa: int, topk: int, rec: str, ctx: str = "any") -> dict:
    """One tuned shape: ``swa`` = SWA list width (128 causal; 192 = the DSpark
    non-causal draft layers), ``topk`` = compressed width (512; 0 = no pool),
    ``ctx`` = context class (decode buckets of compressed layers) or "any"."""
    return {"numtokens": numtokens, "heads": heads, "swa": swa, "topk": topk, "rec": rec, "ctx": ctx}


def pool_view(cache: torch.Tensor, what: str):
    """vLLM's pool tensor (uint8; [B, N, 1, C] NHD or [B, N, C]) -> (flat bytes,
    rec, page tokens, block stride bytes). RuntimeError otherwise (K1)."""
    kv = cache.view(torch.uint8)
    if kv.dim() == 4 and kv.shape[2] == 1:
        kv = kv.squeeze(2)
    if kv.dim() != 3 or kv.shape[2] not in R.REC_BY_BYTES or kv.stride(2) != 1:
        raise RuntimeError(f"{hk.MARKER} dsv41_attn {what}: expected uint8 [blocks, page, 1|-, "
                           f"584|528|288] rows, got {tuple(cache.shape)} {tuple(cache.stride())}")
    rec = R.REC_BY_BYTES[kv.shape[2]]
    nb, page = kv.shape[:2]
    stride = kv.stride(0) if nb > 1 else R.block_stride(rec, page)
    g = R.STRIDE_GRANULE[rec]
    if page < 1 or page & (page - 1) or stride < R.block_bytes(rec, page) or stride % g:
        raise RuntimeError(f"{hk.MARKER} dsv41_attn {what}: page {page} tokens must be a power of two and "
                           f"the block stride ({stride} B) >= {R.block_bytes(rec, page)} B and a multiple "
                           f"of {g} B (the row views need whole rows per block; vLLM aligns the page)")
    flat = kv.as_strided(((nb - 1) * stride + R.block_bytes(rec, page),), (1,))
    return flat, rec, page, stride


_VIEWS: dict = {}


def pool_args(cache: torch.Tensor, what: str, dummies: dict):
    """(d, d16, s, shift, bd, bs, rec, flat) of a pool, cached per (pointer, shape,
    strides): the views are stable tensors over the cache's own storage (no copies)."""
    key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()), cache.dtype, id(dummies))
    got = _VIEWS.get(key)
    if got is None:
        flat, rec, page, stride = pool_view(cache, what)
        d, d16, s, bd, bs = row_views(flat, rec, stride, dummies)
        if len(_VIEWS) > 512:
            _VIEWS.clear()
        got = _VIEWS[key] = (d, d16, s, page.bit_length() - 1, bd, bs, rec, flat)
    return got


def row_views(flat: torch.Tensor, rec: int, stride: int, dummies: dict):
    """(d, d16, s, bd, bs) row views of a flat pool (see the comment above POOL_ARGS)."""
    n = flat.numel()
    if rec == R.REC_V4:
        return (flat.as_strided((n // 576, 576), (576, 1)),
                flat[: n // 2 * 2].view(torch.int16).as_strided((n // 576, 288), (288, 1)),
                flat.as_strided((n // 8, 8), (8, 1)), stride // 576, stride // 8)
    w = R.ROW_BYTES[rec]
    return flat.as_strided((n // w, w), (w, 1)), dummies["d16"], dummies["s"], stride // w, 0


def flat_of(d, s, rec):
    """The pool's flat byte view back from its row views (reference / baseline)."""
    return (s if rec == R.REC_V4 else d).reshape(-1)


# ---------------------------------------------------------------------------
# serving entry
# ---------------------------------------------------------------------------
def load(heads: int, swa_cap: int, topk: int, swa_rec: int, cmp_rec: int, device, cmp_page: int = 32) -> dict | None:
    """Impl-init time (before any CUDA-graph capture): {kernel: {(bucket, ctx):
    compiled}} for this knob, or None (oxide / FlashInfer; auto with nothing
    tuned). helion mode with any untuned T bucket fails closed. Every compiled
    kernel is warmed at every Triton specialization it can meet and checked
    against the exact reference once per device and geometry. ``cmp_page`` =
    tokens per block of the compressed pool (attention block size // compress
    ratio: 32 for the ratio-2 layers, 64 for ratio 1; the SWA pool is 32)."""
    mode = hk.backend(KNOB)
    if mode == "oxide":
        return None
    lab = rec_label(swa_rec, cmp_rec, topk)
    route_mode = os.environ.get(ROUTE_ENV, "auto").strip().lower() or "auto"
    if route_mode not in ("auto", "row"):
        raise RuntimeError(f"{ROUTE_ENV}={route_mode!r}: expected auto | row")
    forced_row = route_mode == "row"
    kernels = {PARTIAL: {}, ROW: {}}
    for name in ((ROW,) if forced_row else (PARTIAL, ROW)):
        for b in buckets(name):
            for ctx in ([c for c, _ in CTX_CLASSES] if case_ctx(b, topk, "c512k") != "any" else ["any"]):
                c = case(b, heads, swa_cap, topk, lab, ctx)
                if hk.config(name, c) is not None:
                    if helion is None:
                        raise RuntimeError(f"{hk.MARKER} {KNOB}={mode} with tuned {name} configs, but "
                                           "helion is not importable")
                    kernels[name][(b, ctx)] = hk.compiled(_BODY[name], name, c, **SETTINGS)
    covered = {b for name in kernels for (b, _) in kernels[name]}
    if mode == "helion":
        need = [b for b in ROW_BUCKETS if b not in covered]
        if need:
            raise RuntimeError(
                f"{hk.MARKER} {KNOB}=helion: no tuned {PARTIAL}/{ROW} config for heads {heads} "
                f"swa {swa_cap} topk {topk} rec {lab} T-buckets {need} on {hk.platform()} (tune "
                f"them in the autotune lab or use {KNOB}=auto)")
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    route = new_route(mode, heads, swa_cap, topk, (swa_rec, cmp_rec), kernels, lab, dev, (32, cmp_page))
    key = (str(dev), heads, swa_cap, topk, lab, forced_row, cmp_page,
           tuple(sorted((n, k) for n in kernels for k in kernels[n])))
    if covered and key not in _VERIFIED:  # once per device+geometry (vLLM builds one impl per layer)
        warm_and_verify(route, dev)
        _VERIFIED.add(key)
        hk.log(f"{KNOB}={mode}: dsv41 attention Helion buckets partial+merge "
               f"{sorted({b for b, _ in kernels[PARTIAL]}) or 'none'} row "
               f"{sorted({b for b, _ in kernels[ROW]}) or 'none'} (warmed + verified on {dev}), FlashInfer "
               f"elsewhere (heads {heads}, swa {swa_cap}, topk {topk}, rec {lab})")
    return route if covered else None


def new_route(mode: str, heads: int, swa_cap: int, topk: int, recs: tuple, kernels: dict, lab: str, dev,
              pages: tuple = (32, 32)) -> dict:
    """A route: the loaded kernels plus the load-time persistent buffers (stable
    pointers; sliced per call, never allocated on the call path)."""
    return {"mode": mode, "heads": heads, "swa": swa_cap, "topk": topk, "recs": recs, "kernels": kernels,
            "label": lab, "device": dev, "pages": pages,
            "dummies": {"d16": torch.zeros(1, 1, dtype=torch.int16, device=dev),
                        "s": torch.zeros(1, 8, dtype=torch.uint8, device=dev)},
            "neg": torch.full((ROW_BUCKETS[-1], 1), -1, dtype=torch.int32, device=dev),
            "zero": torch.zeros(ROW_BUCKETS[-1], dtype=torch.int32, device=dev),
            "sink_none": torch.full((heads,), float("-inf"), dtype=torch.float32, device=dev),
            # stage 2 of the split path: ONE fixed-config kernel (compiled per S on first eager use)
            "merge": merge_kernel() if kernels.get(PARTIAL) else None}


_VERIFIED: set = set()
_WARM: set = set()  # Triton specialization keys already executed eagerly (see spec_key)


def pick(route: dict, t: int, ctx: str = "c512k"):
    """(name, kernel) for T=t rows in context class ``ctx``, or None (no tuned
    kernel -> FlashInfer in auto). The partial+merge pair wins where it has a
    config, else row; a missing class falls back to the next larger one, then
    smaller."""
    if t > ROW_BUCKETS[-1]:
        return None
    names = [c for c, _ in CTX_CLASSES]
    order = [case_ctx(t, route["topk"], ctx)]
    if order[0] != "any":
        i = names.index(ctx) if ctx in names else len(names) - 1
        order = names[i:] + names[:i][::-1]
    for name in (PARTIAL, ROW):
        if name == PARTIAL and t > PARTIAL_BUCKETS[-1]:
            continue
        b = hk.bucket(t, buckets(name))
        for c in order:
            k = route["kernels"][name].get((b, c))
            if k is not None:
                return name, k
    return None


def merge_kernel():
    """The fixed-config stage-2 kernel (None without helion: CPU CI)."""
    if helion is None:
        return None
    return helion.kernel(dsv41_attn_merge, config=helion.Config(block_sizes=[512], num_warps=4, num_stages=1),
                         static_shapes=False, **SETTINGS)


def spec_key(name: str, t: int, a: tuple) -> tuple:
    """Everything that makes Triton compile a NEW variant of an already compiled kernel:
    t == 1, t % 16 (n_tok), the pool block strides' % 16 (plain int args), page shifts, records."""
    swa_shift, swa_bd, swa_bs, cmp_shift, cmp_bd, cmp_bs, swa_rec, cmp_rec = a
    return (name, t == 1, t % 16 == 0, swa_shift, swa_bd % 16 == 0, swa_bs % 16 == 0, cmp_shift,
            cmp_bd % 16 == 0, cmp_bs % 16 == 0, swa_rec, cmp_rec)


def capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def attend(route: dict, q, swa_cache, swa_idx, swa_len, cmp_cache, cmp_idx, cmp_len, sink, out,
           ctx: str = "c512k"):
    """The serving op: FlashInfer's trtllm_batch_decode_sparse_mla_dsv4 SM120
    call semantics (K1) on vLLM's tensors. q/out [T, H, 512] bf16; *_idx
    [T, (1,) C] int32; *_len [T] int32; cmp_* None on ratio-0 layers; sink [H]
    f32 or None; ``ctx`` the step's context class (performance hint only).
    Returns ``out``, or None when this shape has no tuned kernel (auto -> the
    caller runs FlashInfer); helion mode raises instead. No allocation outside
    the kernels, no host sync."""
    t, h = q.shape[0], q.shape[1]
    if t == 0:
        return out
    got = pick(route, t, ctx) if (h == route["heads"] and q.shape[-1] == R.HD) else None
    if got is None:
        if route["mode"] == "helion":
            raise RuntimeError(f"{hk.MARKER} {KNOB}=helion: no dsv41 attention kernel for T={t} "
                               f"heads {h} (route heads {route['heads']})")
        return None
    name, kern = got
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or tuple(out.shape) != tuple(q.shape) \
            or q.stride(-1) != 1 or out.stride(-1) != 1 or swa_len.dtype != torch.int32 \
            or (cmp_cache is not None and cmp_len.dtype != torch.int32) \
            or (sink is not None and sink.dtype != torch.float32):
        raise RuntimeError(f"{hk.MARKER} dsv41_attn: q/out must be bf16 [T, H, 512] with unit last stride and the "
                           f"lens int32 (a new dtype/stride would compile a new variant at serving time), got "
                           f"{q.dtype} {tuple(q.shape)} {q.stride()} / {out.dtype} {tuple(out.shape)} / "
                           f"lens {swa_len.dtype}, sink {None if sink is None else sink.dtype}")
    dm = route["dummies"]
    sd, sd16, ss, sshift, sbd, sbs, srec, _ = pool_args(swa_cache, "swa", dm)
    sidx = swa_idx.reshape(t, -1)
    if cmp_cache is None:
        cd, cd16, cs, cshift, cbd, cbs, crec = sd, sd16, ss, sshift, sbd, sbs, srec
        cidx, clen = route["neg"][:t], route["zero"][:t]
    else:
        cd, cd16, cs, cshift, cbd, cbs, crec, _ = pool_args(cmp_cache, "cmp", dm)
        cidx, clen = cmp_idx.reshape(t, -1), cmp_len
    want_c = route["topk"] or 1
    if srec != route["recs"][0] or (cmp_cache is not None and crec != route["recs"][1]) \
            or (cmp_cache is None) != (route["topk"] == 0) or sidx.shape[1] != route["swa"] \
            or cidx.shape[1] != want_c or sidx.dtype != torch.int32 or cidx.dtype != torch.int32 \
            or sshift != route["pages"][0].bit_length() - 1 \
            or (cmp_cache is not None and cshift != route["pages"][1].bit_length() - 1):
        raise RuntimeError(
            f"{hk.MARKER} dsv41_attn: pools / index widths differ from the loaded route (rec "
            f"{route['label']}, swa {route['swa']}, topk {route['topk']}): got "
            f"{R.REC_NAME[srec]}, swa {tuple(sidx.shape)}, cmp {None if cmp_cache is None else R.REC_NAME[crec]} "
            f"{tuple(cidx.shape)}, pages {1 << sshift}/{1 << cshift} (route {route['pages']})")
    scalars = (sshift, sbd, sbs, cshift, cbd, cbs, srec, crec)
    key = spec_key(name, t, scalars)
    if key not in _WARM and capturing():
        raise RuntimeError(
            f"{hk.MARKER} dsv41_attn: a CUDA graph capture reached the Triton specialization {key} that "
            "no eager call has compiled yet (compiling inside a capture is illegal); the load-time "
            "warm-up covers T==1 / T%16 / its synthetic strides, the first eager step covers the real "
            "pool strides: run one eager forward before capturing")
    snk = sink if sink is not None else route["sink_none"]
    pools = (q, sd, sd16, ss, cd, cd16, cs, sidx, swa_len, cidx, clen, snk)
    if name == ROW:
        kern(*pools, out, *scalars)
    else:
        o_part, lse_part, _ = kern(*pools, *scalars)
        mkey = ("merge", int(lse_part.shape[2]), t == 1, t % 16 == 0)
        if mkey not in _WARM and capturing():
            raise RuntimeError(f"{hk.MARKER} dsv41_attn: a CUDA graph capture reached the merge "
                               f"specialization {mkey} that no eager call has compiled yet; run one eager "
                               "forward before capturing")
        route["merge"](o_part, lse_part, snk, out)
        _WARM.add(mkey)
    _WARM.add(key)
    return out


def warm_sizes(name: str, bucket: int) -> list:
    """The T values that reach every Triton specialization of a bucket: its upper
    edge, its lower edge + 1 and the first multiple of 16 inside it."""
    bs = buckets(name)
    prev = bs[bs.index(bucket) - 1] if bs.index(bucket) else 0
    sizes = {bucket, prev + 1}
    m16 = -(-(prev + 1) // 16) * 16
    if m16 <= bucket:
        sizes.add(m16)
    return sorted(sizes)


def warm_and_verify(route: dict, dev) -> None:
    """Load-time warm-up + oracle (fails closed): every compiled kernel at every
    warm size (compiles all Triton specializations before any capture) and, at
    the bucket's upper edge, adversarial rows (scale extremes, short/empty
    lists, duplicates, -1 holes, stale NaN bytes in unwritten rows, a finite and
    a dominating sink) vs the exact f64 reference on sampled rows. Eager,
    CPU-generated (vLLM inits impls under a CUDA default device / bf16 default)."""
    heads, swa_cap, topk = route["heads"], route["swa"], route["topk"]
    srec, crec = route["recs"]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    ctx_mgr = torch.device(dev)
    ctx_mgr.__enter__()
    try:
        for name in (PARTIAL, ROW):
            for (b, cx), kern in sorted(route["kernels"][name].items()):
                for t in warm_sizes(name, b):
                    prob = adversarial_problem(t, heads, swa_cap, srec, crec, seed=t, sample=24,
                                               spage=route["pages"][0], cpage=route["pages"][1],
                                               topk=topk or R.TOPK, with_cmp=topk > 0, reference=(t == b))
                    one = {**route, "mode": "helion", "kernels": {PARTIAL: {}, ROW: {}}}
                    one["kernels"][name][(b, cx)] = kern
                    got = torch.zeros(prob["q"].shape, dtype=torch.bfloat16, device=dev)
                    run_problem(one, prob, got, cx)
                    if t == b:
                        ok, msg = R.check(got.cpu()[prob["rows"]], prob["ref"][prob["rows"]])
                        if not ok:
                            raise RuntimeError(
                                f"{hk.MARKER} {name} load-time oracle FAILED at T={b} ctx {cx} heads "
                                f"{heads} swa {swa_cap} topk {topk} rec {route['label']} on {dev}: {msg}")
    finally:
        ctx_mgr.__exit__(None, None, None)
        torch.set_default_dtype(prev)
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# problems (tests, load-time oracle, capture gate, lab)
# ---------------------------------------------------------------------------
def _pools(rows_s, rows_c, srec, crec, spage, cpage, pad):
    sflat, sstride = R.make_pool(srec, spage, rows_s, pad=pad, fill=0xFF)
    cflat, cstride = R.make_pool(crec, cpage, rows_c, pad=pad, fill=0xFF)
    return (sflat, srec, spage, sstride), (cflat, crec, cpage, cstride)


def adversarial_problem(t: int, heads: int, swa_cap: int, srec: int, crec: int, seed: int = 0,
                        spage: int = 32, cpage: int = 32, pad: int = 1, n_swa: int = 192,
                        n_cmp: int = 640, topk: int = R.TOPK, with_cmp: bool = True,
                        sample: int | None = None, reference: bool = True) -> dict:
    """CPU problem + its f64 reference (``sample``: only that many rows, always
    including the special ones, are referenced; ``rows`` lists them; ``reference``
    False skips it). Unwritten pool rows hold 0xFF (NaN scales / codes; slot n_*
    is never live, slot 0 -- the kernels' clamp target for -1 -- is written);
    ``pad`` = extra granules per block stride. Rows:
      0 normal | 1 nothing live | 2 three live columns (ctx < window) |
      3 -1 holes + shuffled | 4 one slot repeated over half the list |
      5 len = cap/2 with live-looking slots past len | 6 |q| ~ 2^15 |
      >= 7 random lengths (every 7th full).
    Rows 8 / 9 read slots whose scale bytes are poked to 4 / 235 (2^-123 /
    2^108); slots 10.. hold saturating (3e4), tiny (2^-30), 2^40 and
    RoPE-dominated magnitudes; the sink is -inf / finite / dominating (20)."""
    g = torch.Generator().manual_seed(seed)
    chan = torch.exp(torch.randn(R.HD, generator=g) * 0.75)

    def rows(n):
        x = torch.randn(n, R.HD, generator=g) * chan
        x[10, :64] = 3e4
        x[11, 64:128] = 2.0 ** -30
        x[12, :16] = 2.0 ** 40
        x[13, R.NOPE:] *= 50.0
        return x.bfloat16()

    rs, rc = rows(n_swa), rows(n_cmp)
    swa, cmp = _pools(rs, rc, srec, crec, spage, cpage, pad)
    for pool, rec in ((swa, srec), (cmp, crec)):
        R.poke_scale(pool[0], rec, pool[2], pool[3], 8, 1, 4)
        R.poke_scale(pool[0], rec, pool[2], pool[3], 9, 2, 235)
    q = torch.randn(t, heads, R.HD, generator=g)
    if t > 6:
        q[6] *= 2.0 ** 15
    q = q.bfloat16()

    def lists(cap, n):
        # random slots 14..n-1; the special slots 0 (the -1 clamp target) and
        # 8..13 (extreme scales / magnitudes) only where a row asks for them
        idx = torch.randint(14, n, (t, cap), generator=g, dtype=torch.int32)
        for r in range(min(t, 14)):
            if r == 0 or 8 <= r <= 13:
                idx[r, 0] = r
        ln = torch.randint(1, cap + 1, (t,), generator=g, dtype=torch.int32)
        ln[::7] = cap
        fixed = {0: cap, 1: 0, 2: min(cap, 3), 3: cap, 4: cap, 5: cap // 2, 6: cap}
        for r, v in fixed.items():
            if r < t:
                ln[r] = v
        for r in range(8, min(t, 14)):
            ln[r] = max(int(ln[r]), 1)
        if t > 1:
            idx[1] = -1
        if t > 3:
            idx[3, ::3] = -1
            idx[3] = idx[3, torch.randperm(cap, generator=g)]
        if t > 4:
            idx[4, : cap // 2] = idx[4, 0]
        return idx, ln

    sidx, sln = lists(swa_cap, n_swa)
    if with_cmp:
        cidx, cln = lists(topk, n_cmp)
    else:
        cidx, cln = torch.full((t, 1), -1, dtype=torch.int32), torch.zeros(t, dtype=torch.int32)
    sink = torch.full((heads,), float("-inf"))
    sink[::2] = torch.linspace(-3.0, 4.0, len(sink[::2]))
    sink[1::4] = 20.0
    if sample and t > sample:
        keep = sorted(set(range(14)) | set(torch.linspace(14, t - 1, sample - 14).long().tolist()))
    else:
        keep = list(range(t))
    ref = (R.attention_ref(q, swa, cmp if with_cmp else None, sidx, sln, cidx, cln, sink, rows=keep)
           if reference else None)
    return {"q": q, "swa": swa, "cmp": cmp if with_cmp else None, "swa_idx": sidx, "swa_len": sln,
            "cmp_idx": cidx, "cmp_len": cln, "sink": sink, "ref": ref, "heads": heads, "rows": keep}


def vllm_view_of(flat: torch.Tensor, rec: int, page: int, stride: int) -> torch.Tensor:
    """A flat device pool as vLLM hands it over: uint8 [blocks, page, 1, row bytes] (NHD)."""
    nb = (flat.numel() - R.block_bytes(rec, page)) // stride + 1
    return flat.as_strided((nb, page, 1, R.ROW_BYTES[rec]), (stride, R.ROW_BYTES[rec], R.ROW_BYTES[rec], 1))


def vllm_view(pool, dev):
    """A pool tuple (flat, rec, page, stride) as vLLM hands it over, on ``dev``."""
    flat, rec, page, stride = pool
    return vllm_view_of(flat.to(dev), rec, page, stride)


def run_problem(route: dict, p: dict, out: torch.Tensor, ctx: str = "c512k"):
    """``attend`` on a problem dict through vLLM-shaped pool views, on out's device."""
    dev = out.device
    res = attend(route, p["q"].to(dev), vllm_view(p["swa"], dev), p["swa_idx"].to(dev)[:, None, :],
                 p["swa_len"].to(dev), vllm_view(p["cmp"], dev) if p["cmp"] is not None else None,
                 p["cmp_idx"].to(dev), p["cmp_len"].to(dev), p["sink"].to(dev), out, ctx)
    if res is None:
        raise RuntimeError("no kernel for the problem's T bucket")
    return res


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
LAYER_SHAPES = ((R.WINDOW, R.TOPK),   # target layers with a compressed pool (38 of 40)
                (R.WINDOW, 0),        # target ratio-0 layers (0, 1)
                (192, 0))             # DSpark draft layers: non-causal window, cdiv(128 + 5, 64) * 64
SAMPLE_ROWS = 64
CTX_SLOTS = {"c16k": 8192, "c128k": 65536, "c512k": 262144, "any": 65536}  # compressed-pool rows per ctx class


def _cases(kernel):
    def cases(model: dict) -> list:
        """DeepSeek-V4.1: the T buckets at the per-rank heads for every layer shape
        (window width x top-k), both record pairs (the V4.1 NVFP4 KV pair is a
        candidate: it ships only if the lab wins AND quality passes) and, for the
        decode buckets of compressed layers, the three context classes."""
        if model.get("architecture") not in ARCHS or not model.get("heads"):
            return []
        out = []
        for lab in ("v4", "v41"):
            for sw, tk in LAYER_SHAPES:
                for b in buckets(kernel):
                    cls = [c for c, _ in CTX_CLASSES] if case_ctx(b, tk, "c512k") != "any" else ["any"]
                    out += [case(b, int(model["heads"]), sw, tk, lab, c) for c in cls]
        return out
    return cases


def _recs(lab: str, topk: int):
    if lab == "v4":
        return R.REC_V4, R.REC_V4
    return R.REC_FP8, (R.REC_FP4 if topk else R.REC_FP8)


def random_pool(rec: int, page: int, n_slots: int, dev, g: torch.Generator):
    """A valid random pool of n_slots tokens on ``dev`` (no CPU quantization: lab
    pools reach 300 MB): random fp8 / e2m1 codes without NaNs, scale bytes
    118..134, bf16 RoPE from randn, e4m3 scales 0x28..0x48. (flat, stride)."""
    nb = -(-n_slots // page)
    stride = R.block_stride(rec, page)
    n = nb * page
    rnd = lambda *shape: torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(dev)  # noqa: E731
    fix = lambda x: torch.where((x & 0x7F) == 0x7F, x - 1, x)  # noqa: E731  e4m3fn NaN codes -> max normal
    sc = lambda *shape: torch.randint(118, 135, shape, generator=g, dtype=torch.uint8).to(dev)  # noqa: E731
    flat = torch.zeros(nb, stride, dtype=torch.uint8, device=dev)
    if rec == R.REC_V4:
        data = torch.cat([fix(rnd(n, 448)),
                          torch.randn(n, 64, generator=g).bfloat16().view(torch.uint8).reshape(n, 128).to(dev)], 1)
        scales = torch.cat([sc(n, 7), torch.zeros(n, 1, dtype=torch.uint8, device=dev)], 1)
        flat[:, : page * 576] = data.reshape(nb, page * 576)
        flat[:, page * 576: page * 584] = scales.reshape(nb, page * 8)
    elif rec == R.REC_FP8:
        flat[:, : page * 528] = torch.cat([fix(rnd(n, 512)), sc(n, 16)], 1).reshape(nb, page * 528)
    else:
        sf = torch.randint(0x28, 0x49, (n, 32), generator=g, dtype=torch.uint8).to(dev)
        flat[:, : page * 288] = torch.cat([rnd(n, 256), sf], 1).reshape(nb, page * 288)
    return flat.reshape(-1), stride


def _make(kernel):
    def make(c: dict, seed: int):
        """Realistic serving shape: sequences of up to 6 consecutive rows (decode
        verify) or 512 (prefill), SWA windows of consecutive slots shifting by 1
        row to row, top-k compressed sets drawn over the whole context-class pool
        (worst-case scatter) with ~80% row to row overlap, short lists in a few
        rows and one padding row. Returns the body's args: (*POOL_ARGS, [out,]
        *SCALAR_ARGS), ``out`` for the row body only."""
        t, h, sw, tk, lab = c["numtokens"], c["heads"], c["swa"], c["topk"], c["rec"]
        srec, crec = _recs(lab, tk)
        dev = os.environ.get("DSV41_ATTN_LAB_DEVICE", "cuda")   # "cpu": the CPU test drives this plumbing
        g = torch.Generator().manual_seed(1000 + seed)
        n_swa, n_cmp, page = 1 << 14, CTX_SLOTS[c.get("ctx", "any")], 32
        sflat, sstride = random_pool(srec, page, n_swa, dev, g)
        cflat, cstride = random_pool(crec, page, n_cmp, dev, g) if tk else (sflat, sstride)
        grp = 6 if t <= DECODE_BUCKETS[-1] else 512
        sidx = torch.empty(t, sw, dtype=torch.int32)
        for r in range(t):
            sq, j = divmod(r, grp)
            start = (sq * 977 + j) % (n_swa - sw)
            sidx[r] = torch.arange(start, start + sw, dtype=torch.int32)
        sln = torch.full((t,), sw, dtype=torch.int32)
        if tk:
            cidx = torch.empty(t, tk, dtype=torch.int32)
            for r in range(t):
                if r % grp == 0:
                    cidx[r] = torch.randperm(n_cmp, generator=g)[:tk].to(torch.int32)
                else:
                    cidx[r] = cidx[r - 1]
                    swap = torch.randperm(tk, generator=g)[: tk // 5]
                    cidx[r, swap] = torch.randint(0, n_cmp, (len(swap),), generator=g, dtype=torch.int32)
            cln = torch.full((t,), tk, dtype=torch.int32)
        else:
            cidx, cln = torch.full((t, 1), -1, dtype=torch.int32), torch.zeros(t, dtype=torch.int32)
        for r in range(3, t, 9):  # short-context rows
            sln[r] = max(1, sw // 4)
            if tk:
                cln[r] = max(1, tk // 8)
        if t > 5:
            sln[5] = cln[5] = 0
        sink = torch.linspace(-2.0, 2.0, h)
        q = torch.randn(t, h, R.HD, generator=g).bfloat16()
        dm = {"d16": torch.zeros(1, 1, dtype=torch.int16, device=dev), "s": torch.zeros(1, 8, dtype=torch.uint8, device=dev)}
        sd, sd16, ss, sbd, sbs = row_views(sflat, srec, sstride, dm)
        cd, cd16, cs, cbd, cbs = row_views(cflat, crec, cstride, dm)
        pools = (q.to(dev), sd, sd16, ss, cd, cd16, cs, sidx.to(dev), sln.to(dev), cidx.to(dev), cln.to(dev),
                 sink.to(dev))
        scalars = (page.bit_length() - 1, sbd, sbs, page.bit_length() - 1, cbd, cbs, srec, crec)
        if kernel == ROW:
            return (*pools, torch.empty(t, h, R.HD, dtype=torch.bfloat16, device=dev), *scalars)
        return (*pools, *scalars)
    return make


def _unpack(args, kernel):
    """(q, swa pool, cmp pool, sidx, sln, cidx, cln, sink) of a lab/body args
    tuple; a pool is (flat, rec, page, stride)."""
    (q, sd, _, ss, cd, _, cs, sidx, sln, cidx, cln, sink), rest = args[:12], args[12:]
    (sshift, sbd, sbs, cshift, cbd, cbs, srec, crec) = rest[1:] if kernel == ROW else rest
    w = lambda rec: R.ROW_BYTES[rec] if rec != R.REC_V4 else 576  # noqa: E731
    swa = (flat_of(sd, ss, srec), srec, 1 << sshift, sbd * w(srec))
    cmp = (flat_of(cd, cs, crec), crec, 1 << cshift, cbd * w(crec))
    return q, swa, cmp, sidx, sln, cidx, cln, sink


def merge_ref(o_part, lse_part, sink):
    """Torch (f64) merge of ``dsv41_attn_partial``'s outputs and the sink -> bf16 [T, H, 512]
    (the lab gate runs it on the partial body's tuple; the maths of dsv41_attn_merge)."""
    lse = lse_part.double()
    snk = sink.double() * LOG2E
    m = torch.maximum(lse.amax(-1), snk[None, :])
    m_ref = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = torch.exp2(lse - m_ref[..., None])
    den = w.sum(-1) + torch.exp2(snk[None, :] - m_ref)
    num = (w[..., None] * o_part.double()).sum(2)
    return (num / torch.where(den == 0, torch.ones_like(den), den)[..., None]).to(torch.bfloat16)


def _reference(kernel):
    def reference(*args):
        """Exact f64 reference; above SAMPLE_ROWS rows only an evenly spaced
        sample is computed (the others are NaN and skipped by ``check``)."""
        q, swa, cmp, sidx, sln, cidx, cln, sink = _unpack(args, kernel)
        t = q.shape[0]
        rows = list(range(t)) if t <= SAMPLE_ROWS else \
            sorted(set(torch.linspace(0, t - 1, SAMPLE_ROWS).long().tolist()) | {5})
        has_cmp = cidx.shape[1] > 1 or bool((cln > 0).any())
        cpu = lambda pool: (pool[0].cpu(), *pool[1:])  # noqa: E731
        ref = torch.full(q.shape, float("nan"), dtype=torch.float64)
        ref[rows] = R.attention_ref(q.cpu(), cpu(swa), cpu(cmp) if has_cmp else None, sidx.cpu(), sln.cpu(),
                                    cidx.cpu(), cln.cpu(), sink.cpu(), rows=rows)[rows]
        return ref.to(q.device)
    return reference


def _baseline(kernel):
    def baseline(*args):
        """FlashInfer's SM120 sparse backend exactly as vLLM's
        DeepseekV4FlashInferSM120Attention._forward_decode calls it. The V4.1
        pair is repacked to FlashInfer's footer layout first (cached on the
        pools' data_ptr, so the timed call is the attention alone)."""
        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

        q, swa, cmp, sidx, sln, cidx, cln, sink = _unpack(args, kernel)
        out = args[12] if kernel == ROW else torch.empty_like(q)
        has_cmp = cidx.shape[1] > 1 or bool((cln > 0).any())
        view = lambda pool: _fi_view(_footer(*pool), pool)  # noqa: E731
        t = q.shape[0]
        sr = swa[1]
        kw = {"kv_cache_format": "fp8" if sr == R.REC_V4 else
              ("fp8_dsv41_fp4_ca" if has_cmp else "fp8_dsv41")}
        if has_cmp:
            kw.update(compressed_kv_cache=view(cmp), extra_sparse_indices=cidx.view(t, 1, -1),
                      extra_sparse_topk_lens=cln)
        trtllm_batch_decode_sparse_mla_dsv4(
            query=q, swa_kv_cache=view(swa), workspace_buffer=_workspace(q.device),
            sparse_indices=sidx.view(t, 1, -1), out=out, bmm1_scale=R.SCALE, sinks=sink, kv_layout="NHD",
            swa_topk_lens=sln, **kw)
        return out
    return baseline


def _fi_view(f, pool):
    _, rec, page, stride = pool
    return f.as_strided(((f.numel() - R.block_bytes(rec, page)) // stride + 1, page, 1, R.ROW_BYTES[rec]),
                        (stride, R.ROW_BYTES[rec], R.ROW_BYTES[rec], 1))


_FOOTER: dict = {}
_WS: dict = {}


def _footer(flat, rec, page, stride):
    key = (flat.data_ptr(), rec)
    if key not in _FOOTER:
        _FOOTER.clear()
        _FOOTER[key] = flat if rec == R.REC_V4 else \
            R.to_flashinfer_footer(flat.cpu(), rec, page, stride).to(flat.device)
    return _FOOTER[key]


def _workspace(device):
    if device not in _WS:
        _WS[device] = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WS[device]


def _merged(out):
    """The partial body returns (o_part, lse_part, sink): merge it (the lab's
    gate and accuracy checks see the final attention output either way)."""
    return merge_ref(*out) if isinstance(out, tuple) else out


def _check(out, ref):
    """K3 reader gate on the referenced rows (NaN rows of ``ref`` were skipped)."""
    out = _merged(out)
    live = ~torch.isnan(ref[:, 0, 0])
    return R.check(out[live], ref[live])


def _tune_check(actual, expected):
    """Autotuner accuracy check against FlashInfer's output (the tuning
    baseline): FlashInfer quantizes q and P to fp8, so it sits a few 1e-2 from
    the exact reference on random data and rows may differ by up to that.
    Empty rows are exactly 0 in both. Raises AssertionError (Helion's contract)."""
    actual = _merged(actual)
    keep = expected.float().norm(dim=-1) > 0
    rel = (actual.float() - expected.float()).norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-30)
    if not bool(torch.isfinite(actual.float()).all()) or bool((actual[~keep] != 0).any()) \
            or (bool(keep.any()) and float(rel[keep].max()) > 0.12):
        raise AssertionError(f"dsv41 attention vs FlashInfer: max row rel-L2 "
                             f"{float(rel[keep].max()) if bool(keep.any()) else 0.0:.2e}")


def _entry(kernel):
    return {"name": kernel, "body": _BODY[kernel], "cases": _cases(kernel), "make": _make(kernel),
            "reference": _reference(kernel), "baseline": _baseline(kernel),
            "baseline_name": "FlashInfer SM120 sparse MLA (DSV4)" + (
                " [partial stage only is timed: ship via dsv41_attn_lab, which times partial+merge]"
                if kernel == PARTIAL else ""),
            "check": _check,
            "replaces": [r"sparse_mla_sm120", r"cake_sparse_mla", r"sm120_sparse_mla"],
            # lab only: the accuracy baseline is FlashInfer on the body's args (no
            # in-process, no-timeout compile of Helion's reference config)
            "settings": {"autotune_baseline_fn": _baseline(kernel),
                         "autotune_baseline_accuracy_check_fn": _tune_check,
                         "autotune_log_level": logging.DEBUG, **SETTINGS}}


KERNELS = [_entry(PARTIAL), _entry(ROW)]
