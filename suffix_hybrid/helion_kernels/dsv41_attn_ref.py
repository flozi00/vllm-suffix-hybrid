# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash sparse-MLA attention: K1 contract + exact CPU reference.

Pure torch (no helion, no triton, no CUDA): the oracle for
suffix_hybrid/helion_kernels/dsv41_attn.py, the record writers/readers the
tests and the lab build pools with, and the layout converters the K4 harness
needs to feed FlashInfer. Dossier: docs/dsv41-attn.md (sources: vLLM
39983a5e7 vllm/models/deepseek_v41/{attention.py,common/ops/cache_utils.py,
common/ops/fused_compress_quant_cache.py}; FlashInfer 0.7.2
include/flashinfer/attention/sparse_mla_sm120/model/*).

K1 contract (per layer, per TP rank: 8 q heads, ONE kv head, head dim 512)
--------------------------------------------------------------------------
q       [T, H, 512] bf16. RoPE (GPT-J interleaved, last 64 dims) is already
        applied by vLLM's fused qnorm/rope/insert op; H = padded local heads (8).
logits  s = sm_scale * (q . k), sm_scale = 512 ** -0.5, k = the FULL 512-dim
        cached row (448 NoPE + 64 rotated RoPE dims; V4.1 keeps all 512).
value   v = k (MLA latent: the row IS the value; the inverse RoPE on the last
        64 output dims happens later in vLLM's o_proj, not here).
sink    f32 [H] (-inf = none): one extra softmax column of logit ``sink``
        (NOT scaled by sm_scale) with a zero value vector:
        out = sum_i exp(s_i - M) v_i / (sum_i exp(s_i - M) + exp(sink - M)).
lists   two candidate lists per token, softmaxed JOINTLY (one softmax):
          SWA  idx int32 [T, W]  global slots into the SWA pool (W = 128, or
               the DSpark non-causal width), -1 masked, len int32 [T]
          CMP  idx int32 [T, 512] global slots into the compressed pool of the
               layer's kv-source (absent on ratio-0 layers), -1 masked, len [T]
        a column is live iff col < len[t] and idx >= 0. All live columns count
        once each (duplicate slots count twice). No live column in either
        list -> output row exactly 0 (also with a finite sink).
slot    global = block_id * page_tokens + token_in_block (page_tokens = the
        pool tensor's shape[1]: 32 for the SWA pool, attn_block_size //
        compress_ratio for the compressed pool).
out     [T, H, 512] bf16.

Records (uint8 pools [num_blocks, ...], block stride in BYTES = pool.stride(0))
  REC_V4  584 B/token (what SM120 serves today; FlashInfer ModelType::DSV4)
          page = [page*576 data | page*8 scales | pad]; token r of a page:
          data  base + r*576 : [0,448) fp8 e4m3 NoPE, [448,576) bf16 RoPE
          scale base + page*576 + r*8 : 7 UE8M0 bytes (one per 64 NoPE dims),
          byte 7 = 0 pad. scale = 2^(byte-127). Writer: exponent =
          ceil(log2(max(amax,1e-4)/448)), x/2^exponent clamped to +-448.
  REC_FP8 528 B/token (vLLM "V4.1 MXFP8", interleaved, SWA pool of nvfp4_ds_mla)
          token r: base + r*528 : [0,512) fp8 e4m3 (RoPE dims included),
          [512,528) 16 UE8M0 bytes, one per 32 dims.
  REC_FP4 288 B/token (vLLM "V4.1 NVFP4", interleaved, compressed pool only)
          token r: base + r*288 : [0,256) e2m1 pairs (even dim = low nibble),
          [256,288) 32 e4m3 scales, one per 16 dims, scale = clamp(amax/6,
          2^-9, 448) rounded to e4m3. All 512 dims (RoPE too) are fp4.
NOTE FlashInfer 0.7.2 packs REC_FP8/REC_FP4 as page-level FOOTERS (data of the
whole page, then the scales); vLLM's V4.1 records are interleaved per token.
They are NOT byte compatible: ``to_flashinfer_footer`` converts for the K4
baseline; never point FlashInfer at a vLLM-written V4.1 pool.
"""
from __future__ import annotations

import torch

HD, NOPE, ROPE = 512, 448, 64
SCALE = HD ** -0.5
WINDOW = 128
TOPK = 512
FP8_MAX = 448.0
LOG2E = 1.4426950408889634
REC_V4, REC_FP8, REC_FP4 = 0, 1, 2
ROW_BYTES = {REC_V4: 584, REC_FP8: 528, REC_FP4: 288}
REC_BY_BYTES = {v: k for k, v in ROW_BYTES.items()}
REC_NAME = {REC_V4: "v4_fp8_584", REC_FP8: "v41_mxfp8_528", REC_FP4: "v41_nvfp4_288"}
GATE = 2e-2  # K3 reader gate: max (token, head) row rel-L2 vs the exact reference


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------
def block_bytes(rec: int, page: int) -> int:
    """Payload bytes of one block (page tokens)."""
    return page * ROW_BYTES[rec]


# Block strides are whole multiples of this many bytes in vLLM (V4: the 576 B
# kv_page_alignment of attention.py; V4.1 records: lcm(528, 288) = 3168 B, a
# multiple of both row sizes): the kernels view the pool as ROW-addressed
# tensors ([rows, 576] + [rows, 8] for V4, [rows, 528], [rows, 288]) so the
# block stride must be a whole number of rows (K1 contract, checked at bind).
STRIDE_GRANULE = {REC_V4: 576, REC_FP8: 528, REC_FP4: 288}


def block_stride(rec: int, page: int, pad_granules: int = 0) -> int:
    """vLLM's padded block stride in bytes: payload rounded up to the granule
    (+ ``pad_granules`` extra granules, e.g. a packed neighbour layer)."""
    g = STRIDE_GRANULE[rec]
    return (-(-block_bytes(rec, page) // g) + pad_granules) * g


def new_pool(rec: int, page: int, n_blocks: int, pad: int = 0, fill: int = 0) -> tuple:
    """(flat uint8 [n_blocks * stride], stride bytes). ``pad`` extra granules per
    block mimic vLLM's page/lcm alignment padding; ``fill`` the stale content
    (0xFF = NaN scales/codes: nothing may read unwritten rows)."""
    stride = block_stride(rec, page, pad)
    return torch.full((n_blocks * stride,), fill, dtype=torch.uint8), stride


def _addr(rec: int, page: int, stride: int, slot: torch.Tensor):
    s = slot.to(torch.int64)
    base = (s // page) * stride
    r = s % page
    if rec == REC_V4:
        return base + r * 576, base + page * 576 + r * 8
    return base + r * ROW_BYTES[rec], None


# ---------------------------------------------------------------------------
# quantizers (reference writers: float32 math in vLLM's op order)
# ---------------------------------------------------------------------------
def _ue8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    return torch.ceil(torch.log2(torch.clamp(amax, min=1e-4) * (1.0 / FP8_MAX)))


def _fp8_bytes(scaled: torch.Tensor) -> torch.Tensor:
    return scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e2m1_code(v: torch.Tensor) -> torch.Tensor:
    """f32 -> 4-bit e2m1 code (sign<<3 | magnitude code), round-to-nearest-even
    on the code lattice (cvt.rn.satfinite.e2m1x2.f32), saturating at 6."""
    a = v.abs()
    m = ((a > 0.25).int() + (a >= 0.75) + (a > 1.25) + (a >= 1.75)
         + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    return (m + 8 * (v < 0).int()).to(torch.uint8)


def quantize_rows(rec: int, x: torch.Tensor):
    """bf16/f32 rows [n, 512] (RoPE already applied) -> (data, scales) byte
    tensors for ``rec``: V4 (data [n,576], scales [n,8]); FP8 (data [n,512],
    scales [n,16]); FP4 (data [n,256], scales [n,32])."""
    x = x.float()
    n = x.shape[0]
    if rec == REC_V4:
        nope = x[:, :NOPE].reshape(n, 7, 64)
        e = _ue8m0_exponent(nope.abs().amax(-1))
        q = _fp8_bytes(nope * torch.exp2(-e)[..., None]).reshape(n, NOPE)
        rope = x[:, NOPE:].to(torch.bfloat16).view(torch.uint8).reshape(n, 128)
        enc = (e + 127.0).clamp(0, 255).to(torch.uint8)
        return torch.cat([q, rope], 1), torch.cat([enc, torch.zeros(n, 1, dtype=torch.uint8)], 1)
    if rec == REC_FP8:
        tiles = x.reshape(n, 16, 32)
        e = _ue8m0_exponent(tiles.abs().amax(-1))
        return (_fp8_bytes(tiles * torch.exp2(-e)[..., None]).reshape(n, 512),
                (e + 127.0).clamp(0, 255).to(torch.uint8))
    tiles = x.reshape(n, 32, 16)
    amax = tiles.abs().amax(-1)
    sf = (amax * (1.0 / 6.0)).clamp(2.0 ** -9, 448.0).to(torch.float8_e4m3fn)
    code = e2m1_code(tiles / sf.float()[..., None]).reshape(n, 256, 2)
    return (code[..., 0] | (code[..., 1] << 4)), sf.view(torch.uint8)


def write_rows(pool: torch.Tensor, rec: int, page: int, stride: int,
               slots: torch.Tensor, rows: torch.Tensor) -> None:
    """vLLM's writer, reference form: quantize ``rows`` into ``slots``."""
    data, scales = quantize_rows(rec, rows)
    d0, s0 = _addr(rec, page, stride, slots)
    ar = torch.arange(data.shape[1])
    pool[d0[:, None] + ar[None, :]] = data
    if rec == REC_V4:
        pool[s0[:, None] + torch.arange(8)[None, :]] = scales
    else:
        pool[(d0 + data.shape[1])[:, None] + torch.arange(scales.shape[1])[None, :]] = scales


def poke_scale(pool: torch.Tensor, rec: int, page: int, stride: int, slot: int, block: int,
               value: int) -> None:
    """Overwrite one scale byte (adversarial scale extremes the writers never emit)."""
    d0, s0 = _addr(rec, page, stride, torch.tensor([slot]))
    off = s0 + block if rec == REC_V4 else d0 + (512 if rec == REC_FP8 else 256) + block
    pool[int(off)] = value


def make_pool(rec: int, page: int, rows: torch.Tensor, pad: int = 0, fill: int = 0,
              extra_blocks: int = 0):
    """A pool holding ``rows`` [n, 512] at slots 0..n-1: (flat, stride)."""
    n = rows.shape[0]
    nb = -(-n // page) + extra_blocks
    flat, stride = new_pool(rec, page, nb, pad, fill)
    write_rows(flat, rec, page, stride, torch.arange(n), rows)
    return flat, stride


# ---------------------------------------------------------------------------
# exact readers (float64)
# ---------------------------------------------------------------------------
def _e2m1_mag(code: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    c = code.to(torch.int64)
    return torch.where(c >= 8, -lut[c & 7], lut[c & 7])


def dequant_rows(pool: torch.Tensor, rec: int, page: int, stride: int,
                 slots: torch.Tensor) -> torch.Tensor:
    """Exact [n, 512] float64 rows at ``slots`` (all must be >= 0)."""
    d0, s0 = _addr(rec, page, stride, slots)
    n = slots.numel()
    g = lambda off, w: pool[off[:, None] + torch.arange(w)[None, :]]  # noqa: E731
    if rec == REC_V4:
        nope = g(d0, NOPE).view(torch.float8_e4m3fn).double().reshape(n, 7, 64)
        sc = torch.exp2(g(s0, 7).double() - 127.0)
        rope = g(d0 + NOPE, 128).contiguous().view(torch.bfloat16).double()
        return torch.cat([(nope * sc[..., None]).reshape(n, NOPE), rope], 1)
    if rec == REC_FP8:
        v = g(d0, 512).view(torch.float8_e4m3fn).double().reshape(n, 16, 32)
        sc = torch.exp2(g(d0 + 512, 16).double() - 127.0)
        return (v * sc[..., None]).reshape(n, 512)
    code = g(d0, 256)
    lo, hi = _e2m1_mag(code & 15), _e2m1_mag(code >> 4)
    v = torch.stack([lo, hi], -1).reshape(n, 32, 16)
    sf = g(d0 + 256, 32).view(torch.float8_e4m3fn).double()
    return (v * sf[..., None]).reshape(n, 512)


Pool = tuple  # (flat uint8, rec, page, stride)


def attention_ref(q: torch.Tensor, swa: Pool, cmp: Pool | None, swa_idx, swa_len,
                  cmp_idx=None, cmp_len=None, sink=None, sm_scale: float = SCALE, rows=None):
    """The K1 math, float64, one token at a time. q [T,H,512]; idx [T,C] int32
    (-1 masked); len [T]; sink [H] (None / -inf = no sink). Returns [T,H,512]
    (``rows``: only these token rows are computed, the rest stay 0)."""
    t_n, h = q.shape[:2]
    out = torch.zeros(t_n, h, HD, dtype=torch.float64)
    snk = torch.full((h,), float("-inf"), dtype=torch.float64) if sink is None else sink.double()
    for t in (range(t_n) if rows is None else rows):
        ks = []
        for pool, idx, ln in ((swa, swa_idx, swa_len), (cmp, cmp_idx, cmp_len)):
            if pool is None:
                continue
            row = idx[t]
            live = (torch.arange(row.numel()) < int(ln[t])) & (row >= 0)
            if bool(live.any()):
                flat, rec, page, stride = pool
                ks.append(dequant_rows(flat, rec, page, stride, row[live].long()))
        if not ks:
            continue
        k = torch.cat(ks, 0)
        s = sm_scale * (q[t].double() @ k.T)                      # [H, n]
        m = torch.maximum(s.amax(1), snk)
        w = torch.exp(s - m[:, None])
        den = w.sum(1) + torch.exp(snk - m)
        out[t] = (w @ k) / den[:, None]
    return out


def check(out: torch.Tensor, ref: torch.Tensor):
    """K3 reader gate: all finite, rows with no live column exactly 0, max
    (token, head) row rel-L2 <= 2e-2. Returns (ok, message)."""
    out, ref = out.double(), ref.double()
    live = ref.norm(dim=-1) > 0
    finite = bool(torch.isfinite(out).all())
    zero = bool((out[~live] == 0).all())
    d = (out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-300)
    err = float(d[live].max()) if bool(live.any()) else 0.0
    return (finite and zero and err <= GATE,
            f"max row rel-L2 {err:.2e} (gate {GATE}), finite {finite}, empty rows zero {zero}")


# ---------------------------------------------------------------------------
# FlashInfer 0.7.2 interop (K4 baseline only)
# ---------------------------------------------------------------------------
def to_flashinfer_footer(pool: torch.Tensor, rec: int, page: int, stride: int) -> torch.Tensor:
    """vLLM interleaved REC_FP8/REC_FP4 pool -> FlashInfer's page-footer pack
    (data of the whole page, then its scales; same block stride). REC_V4 is
    already FlashInfer's DSV4 layout and is returned as is."""
    if rec == REC_V4:
        return pool
    data_w = 512 if rec == REC_FP8 else 256
    nb = pool.numel() // stride
    out = pool.clone().view(nb, stride)
    src = pool.view(nb, stride)
    rows = src[:, : page * ROW_BYTES[rec]].reshape(nb, page, ROW_BYTES[rec])
    out[:, : page * data_w] = rows[:, :, :data_w].reshape(nb, -1)
    out[:, page * data_w: page * ROW_BYTES[rec]] = rows[:, :, data_w:].reshape(nb, -1)
    return out.reshape(-1)


def overlap_stats(idx: torch.Tensor, group: int = 6) -> dict:
    """How much the top-k sets of ``group`` consecutive rows (one sequence's
    spec-decode rows) share: mean |union| / (group * k) over complete groups
    (1/group = identical sets, 1.0 = disjoint). Feed it a dump of the
    indexer's topk_indices_buffer to decide whether q rows can share K tiles
    (docs/dsv41-attn.md, 'row-grouped CMP tile')."""
    t_n = (idx.shape[0] // group) * group
    ratios = []
    for g0 in range(0, t_n, group):
        rows = idx[g0:g0 + group]
        live = rows[rows >= 0]
        if live.numel():
            ratios.append(live.unique().numel() / live.numel())
    return {"groups": len(ratios), "union_over_total": (sum(ratios) / len(ratios)) if ratios else None,
            "ideal_shared": 1.0 / group}


# ---------------------------------------------------------------------------
# cost + capacity model (arithmetic only; every hardware constant is an
# ESTIMATE, nothing here is measured -- docs/dsv41-attn.md has the derivation)
# ---------------------------------------------------------------------------
# config.json compress_ratios: layers 0,1 = 0 (SWA only); 2..19 = 2; 20..39 = 1.
# vLLM keeps a compressed pool + an indexer-K pool only on kv_source_layer_ids.
COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
KV_SOURCE_LAYERS = (2, 8, 14, 20)
DRAFT_LAYERS = 3          # DSpark draft head layers 40..42: SWA only, non-causal window
DRAFT_SWA_LIVE = 128 + 5  # causal window + the block's spec tokens
INDEXER_ROW_BYTES = 132   # 128 fp8 + 4 f32 scale (68 with the FP4 indexer)
HW = {  # RTX PRO 6000 Blackwell Server Edition, per card (ESTIMATES from public specs / assumptions)
    "hbm_gbs": 1792.0,        # GDDR7 512-bit, peak
    "hbm_eff": 0.75,          # assumed achievable fraction for 584 B random-row gathers
    "bf16_tflops": 250.0,     # assumed mma.sync sustained (peak dense ~500)
    "alu_tops": 40.0,         # assumed sustained int/fp32 elementwise throughput
}


def decode_bytes_per_row(srec: int, crec: int, swa_live: int = WINDOW, cmp_live: int = TOPK) -> int:
    """Pool bytes one q row gathers per forward over all 43 attention layers
    (40 backbone + 3 draft), per TP rank (every rank reads the full KV: one kv head)."""
    total = DRAFT_LAYERS * DRAFT_SWA_LIVE * ROW_BYTES[srec]
    for ratio in COMPRESS_RATIOS:
        total += swa_live * ROW_BYTES[srec] + (cmp_live * ROW_BYTES[crec] if ratio else 0)
    return total


def decode_step_model(batch: int = 32, rows: int = 6, srec: int = REC_V4, crec: int = REC_V4,
                      heads_padded: int = 16, overlap: float = 1.0) -> dict:
    """Per-step attention work for ``batch`` sequences x ``rows`` q rows, one
    rank. ``overlap`` in [0, 1] = share of a row's gathered rows already read
    by the previous row of the same sequence (served from L2): 0 = every row
    re-reads, 1 = a sequence reads its bytes once. Returns bytes, FLOPs and the
    three roofline times in ms (ESTIMATES: HW constants above)."""
    per_row = decode_bytes_per_row(srec, crec)
    rows_total = batch * rows
    dram = batch * per_row * (1 + (rows - 1) * (1 - overlap))
    l2 = rows_total * per_row
    cols = [WINDOW + (TOPK if r else 0) for r in COMPRESS_RATIOS] + [DRAFT_SWA_LIVE] * DRAFT_LAYERS
    flops = rows_total * sum(4 * heads_padded * c * HD for c in cols)       # QK + PV, padded M
    deq = rows_total * sum(c * HD for c in cols)
    return {"bytes_per_row": per_row, "dram_bytes": dram, "l2_bytes": l2, "mma_flops_padded": flops,
            "dequant_elems": deq,
            "t_dram_ms": dram / (HW["hbm_gbs"] * 1e9 * HW["hbm_eff"]) * 1e3,
            "t_mma_ms": flops / (HW["bf16_tflops"] * 1e12) * 1e3,
            "t_dequant_ms": deq * 8 / (HW["alu_tops"] * 1e12) * 1e3}


def kv_bytes_per_ctx_token(crec: int, indexer_bytes: int = INDEXER_ROW_BYTES) -> float:
    """Bottom-up KV bytes per context token per rank for the compressed pools
    and the indexer-K pools (4 kv-source layers: ratios 2,2,2,1 -> 2.5 states
    per token). The SWA pools are ~constant per sequence and omitted."""
    states = sum(1.0 / COMPRESS_RATIOS[i] for i in KV_SOURCE_LAYERS)
    return states * (ROW_BYTES[crec] + indexer_bytes)
