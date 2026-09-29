# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A4 decode GEMM (M <= 64) — launch plan (M tiling + split-K
policy), reference, CPU twin, on-silicon oracle + bench.

Kernel: kernels-oxide/nvfp4_gemm (cuda-oxide -> PTX 8.7 .target sm_120a ->
ptxas 13.0 -> sm_120a SASS), block-scaled FP4 tensor-core mma
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
.f32.e2m1.e2m1.f32.ue4m3`. Host op: `_native.nvfp4_gemm_cuda`.

vLLM conventions reproduced here (dossier qwen38-27b-kernels.md §7):
  quant  (vllm scaled_fp4_quant, nvfp4_utils.cuh:262-287):
         sf = e4m3(amax16 * g / 6);  q = e2m1_rne(x * g / sf)
  weight uint8 [N, K/2], element 2i in the LOW nibble; scales e4m3 [N, K/16]
         swizzled 128x4 (nvfp4_utils.py:13-53, `swizzle_sf` below)
  gemm   y = alpha * sum(q_a sf_a q_b sf_b),  alpha = 1 / (g_x * g_w)

Launch plan (`plan`, mirrored by the host op src/nvfp4_gemm_oxide.rs):
  M tiling  tiles = ceil(M/16) in 1..4 -> entry nvfp4_gemm_t{tiles}; each
            warp loads a k64 weight fragment + scales once and runs `tiles`
            mmas on it (register accumulators), so weight bytes stay at the
            roofline for every M <= 64.
  split-K   see `splits_for`; partials f32 [splits, M, N] reduced in fixed
            split order by nvfp4_splitk_reduce (deterministic, graph-safe:
            the grid depends on (N, K, M) only, workspace preallocated).
  fused     ONE launch per linear (entries nvfp4_gemm_f{tiles}, plan
            `flags`): FUSE_QUANT = the bf16 quant runs in the GEMM prologue
            (each CTA quantizes rows < M over its split's K range into smem
            with the separate kernel's own quant function; CTA (0, split)
            also writes aq/asf so the layer-oracle readback still covers
            every row/block); FUSE_REDUCE = "last CTA fixes up" split-K (per
            column tile atomic ticket, the last arrival sums the partials in
            fixed split order 0..S-1 -> bit-identical to nvfp4_splitk_reduce;
            the ticket's atom.inc wraps the counter to 0: graph-replay safe).
            Old 3-launch path (nvfp4_quant_act + nvfp4_gemm_t + reduce):
            SUFFIX_NVFP4_GEMM_FUSED=0; =reduce fuses only the split-K reduce.
            Quant fusion also falls back above FUSED_QUANT_MAX_M or when its
            smem exceeds QUANT_SMEM_MAX (redundant per-CTA quant work grows
            with M; see `bench` old-vs-fused rows).

CLI (in-pod, SM120 + oxide bundle):
  python -m suffix_hybrid.kernels.nvfp4_gemm oracle   # numerics vs torch ref + vLLM op
  python -m suffix_hybrid.kernels.nvfp4_gemm bench    # us: ours vs vLLM (quant+gemm, CUDA graphs)
  python -m suffix_hybrid.kernels.nvfp4_gemm sweep    # bench + best split count per case
Boot: SUFFIX_NVFP4_GEMM_SPIKE=oracle|bench|both|sweep (sitecustomize, child process).
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np

MARKER = "[suffix nvfp4-gemm]"
FAMILY = "nvfp4_gemm"
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
# qwen3.8-27b decode projections (N, K): the spike target is the largest.
SHAPES = {
    "mlp_gate_up": (34816, 5120),
    "gdn_in_proj_qkvz": (16384, 5120),
    "mlp_down": (5120, 17408),
    "attn_qkv": (14336, 5120),
    "gdn_out_proj": (5120, 6144),
    "attn_o": (5120, 6144),
}
SPIKE_SHAPE = "mlp_gate_up"
# gemma-4-26b-a4b-nvfp4 dense linears (hidden 2816; sliding attn 16 q / 8 kv
# heads x 256; full attn 16 q x 512 + 2 kv x 512 with k_eq_v; dense MLP
# 2112). MoE experts (128 x 704) run FusedMoE, not this kernel family.
GEMMA_SHAPES = {
    "gemma_sliding_qkv": (8192, 2816),
    "gemma_full_qkv": (9216, 2816),
    "gemma_sliding_o": (2816, 4096),
    "gemma_full_o": (2816, 8192),
    "gemma_mlp_gate_up": (4224, 2816),
    "gemma_mlp_down": (2816, 2112),
}
# qwen3.8-flash-next-nvfp4 per-rank dense linears at TP=2 (hidden 2560; GDN
# 16 k / 48 v heads x 128; QSA 24 q (+24 gate) / 2 kv heads x 256; shared
# expert 640; PLE ple_embed_dim 2560, hc_count 4; HC lora rank 320) from
# vLLM v0.30.0 vllm/models/qwen4_exp/nvidia/{model,qsa,ple_layer,mtp,
# hyperconnection}.py + the HF config. QSA qkv = (24 q + 24 gate + 1 k +
# 1 v) x 256 = 6656 per rank (2 kv heads / TP2 -> 1). GDN out_proj and QSA
# o_proj share (2560, 3072). HyperConnection projections are replicated
# (disable_tp / ReplicatedLinear): inject = 320 lora + 4 hc + 12 pad rows.
QWEN_FLASH_SHAPES = {
    "flash_gdn_in_proj_qkvz": (8192, 2560),
    "flash_gdn_in_proj_ba": (48, 2560),
    "flash_o_proj": (2560, 3072),  # GDN out_proj + QSA o_proj
    "flash_qsa_qkv": (6656, 2560),
    "flash_shared_gate_up": (640, 2560),
    "flash_shared_down": (2560, 320),
    "flash_ple_kv_proj": (12800, 2560),
    "flash_mtp_fc": (1280, 2560),  # fc_embedding / fc_hidden (gather_output)
    "flash_hc_down_inject": (336, 10240),
    "flash_hc_down": (320, 10240),
    "flash_hc_up": (10240, 320),
}
ALL_SHAPES = {**SHAPES, **GEMMA_SHAPES, **QWEN_FLASH_SHAPES}

# ---------------------------------------------------------------------------
# launch plan: M tiling + split-K policy (host op validates, never re-derives)
# ---------------------------------------------------------------------------
MAX_M = 64
MAX_SPLITS = 64  # host op limit (src/nvfp4_gemm_oxide.rs MAX_SPLITS)
SMS = 188  # RTX PRO 6000 Blackwell (Server / Max-Q): 188 SMs
TARGET_CTAS = 4 * SMS  # ~4 resident 128-thread CTAs per SM
MIN_KPS = 2  # >= 2 k64 steps per split: the 2-stage load pipeline has depth
PARTIAL_DIV = 128  # splits <= K / (128 * tiles): f32 partial traffic
# (write + reduce read = 8 B x 16*tiles x N per split) <= N*K bytes ~ 1.8x
# the weight bytes, and it is L2-resident
# Host-op ABI (.param counts), checked against the cubin manifest at load.
PARAMS = {"nvfp4_quant_act": 7, "nvfp4_splitk_reduce": 7,
          **{f"nvfp4_gemm_t{t}": 15 for t in (1, 2, 3, 4)},
          **{f"nvfp4_gemm_f{t}": 21 for t in (1, 2, 3, 4)}}
# Single-launch plan (kernel FUSE_QUANT / FUSE_REDUCE; host op fused_layout).
FUSED_ENV = "SUFFIX_NVFP4_GEMM_FUSED"  # "0" = old 3-launch path, "reduce"
FUSE_QUANT, FUSE_REDUCE = 1, 2
STAGE_FLOATS = 4096  # split-K fixup staging (ch + 1) * M * 32 f32 <= 16 KB
FUSED_QUANT_MAX_M = 64  # ponytail: static; bench old-vs-fused rows retune it
# Silicon (qwen TP2, 2026-09-29): single launch wins only at M=1 (~2 us per
# linear); from M=5 the last-CTA split-K fixup is a serial tail (HC down
# 336x10240 S=54: 10 -> 22 us at M=5, 60 us at M=16). Fuse only up to this M.
FUSED_MAX_M_ENV = "SUFFIX_NVFP4_GEMM_FUSED_MAX_M"


def fused_max_m() -> int:
    import os
    return int(os.environ.get(FUSED_MAX_M_ENV, "1") or 1)
QUANT_SMEM_MAX = 20 * 1024  # prologue A+scales: keeps >= 4 CTAs/SM (TARGET_CTAS)


def tiles_for(m: int) -> int:
    return -(-m // 16)


def splits_for(n: int, k: int, m: int) -> int:
    """Split-K factor for an (N, K) GEMM at M rows (depends on the M bucket
    tiles = ceil(M/16) only). Enough CTAs for ~4 per SM (ceil(N/32) CTAs
    per split), capped by: MAX_SPLITS, >= MIN_KPS k64 steps per split, and
    the partial-traffic budget K / (PARTIAL_DIV * tiles). Wide-N projections
    stay at 1 (no reduce kernel); narrow-N / deep-K ones (HC down 336 x
    10240: 11 CTAs) split deep. ponytail: static heuristic; the in-pod
    `sweep` prints the measured best split per case to retune it."""
    ctas = -(-n // 32)
    want = -(-TARGET_CTAS // ctas)
    cap = min(MAX_SPLITS, (k // 64) // MIN_KPS, k // (PARTIAL_DIV * tiles_for(m)))
    return max(1, min(want, cap))


def fused_mode() -> str:
    """SUFFIX_NVFP4_GEMM_FUSED: "" / "1" -> "all" (default), "0" -> "off"
    (the old 3-launch path, exactly), "reduce" -> fuse the split-K reduce
    only (quant stays a separate launch). Loud on anything else."""
    raw = os.environ.get(FUSED_ENV, "").strip().lower()
    modes = {"": "all", "1": "all", "0": "off", "reduce": "reduce"}
    if raw not in modes:
        raise ValueError(f"{FUSED_ENV}={raw!r}: want 0, 1 or reduce")
    return modes[raw]


def quant_smem(m: int, kps: int) -> int:
    """Prologue smem of FUSE_QUANT: A rows (kps*32 + 16 B: odd multiple of
    16 -> conflict-free fragment loads) + scale rows (4*(kps|1) B)."""
    return m * (kps * 32 + 16) + m * 4 * (kps | 1)


def fused_layout(m: int, kps: int, splits: int, flags: int) -> tuple:
    """(dynamic smem bytes, fixup chunk ch) of an nvfp4_gemm_f launch —
    mirrors the host op's `fused_layout` (src/nvfp4_gemm_oxide.rs)."""
    q = quant_smem(m, kps) if flags & FUSE_QUANT else 0
    ch = min(max(STAGE_FLOATS // (32 * m) - 1, 1), splits)
    r = (ch + 1) * m * 32 * 4 if flags & FUSE_REDUCE and splits > 1 else 0
    return 16 + max(q, r), ch


def plan(n: int, k: int, m: int, splits: int | None = None, fused: str | bool | None = None,
         prequant: bool = False) -> dict:
    """What the host op launches for (N, K, M): GEMM entry, grid, k64 steps
    per split, launched splits (no empty trailing split), and the fusion
    flags (`fused`: None = SUFFIX_NVFP4_GEMM_FUSED, True/"all", "reduce",
    False/"off"; `prequant` = vLLM-quantized input: nothing to quantize)."""
    tiles = tiles_for(m)
    steps = k // 64
    s = splits_for(n, k, m) if splits is None else splits
    kps = -(-steps // s)
    s = -(-steps // kps)
    grid = (-(-n // 32), s)
    mode = fused_mode() if fused is None else {True: "all", False: "off"}.get(fused, fused)
    if fused is None and m > fused_max_m():
        mode = "off"
    fq = (mode == "all" and not prequant and m <= FUSED_QUANT_MAX_M
          and quant_smem(m, kps) <= QUANT_SMEM_MAX)
    fr = mode != "off" and s > 1
    flags = FUSE_QUANT * fq | FUSE_REDUCE * fr
    smem, ch = fused_layout(m, kps, s, flags)
    return dict(tiles=tiles, splits=s, kps=kps, grid=grid, ctas=grid[0] * grid[1],
                entry=f"nvfp4_gemm_{'f' if flags else 't'}{tiles}",
                partial=(s * m * n if s > 1 else 0), fused_quant=fq, fused_reduce=fr,
                flags=flags, smem=smem if flags else 0, ch=ch,
                launches=1 + (not fq and not prequant) + (s > 1 and not fr))


def plan_str(p: dict) -> str:
    fz = ("q" if p["fused_quant"] else "") + ("r" if p["fused_reduce"] else "")
    return (f"tiles={p['tiles']} splits={p['splits']} kps={p['kps']} "
            f"ctas={p['ctas']} fused={fz or '-'} launches={p['launches']}")


def partial_elems(n: int, k: int, max_m: int = MAX_M) -> int:
    """f32 split-K workspace for every M <= max_m (worst case at each M
    bucket's top row count)."""
    ms = {min(16 * t, max_m) for t in range(1, tiles_for(max_m) + 1)}
    return max(plan(n, k, m)["partial"] for m in ms)


# ---------------------------------------------------------------------------
# reference quantization (numpy, fp32) — identical rounding to the kernel
# ---------------------------------------------------------------------------
def e2m1_code(v):
    """|v| -> e2m1 magnitude code 0..7, RNE (ties to even code), saturating."""
    a = np.minimum(np.abs(np.asarray(v, dtype=np.float32)), np.float32(6.0))
    gt = (a[..., None] > _MID).sum(-1)
    tie_up = ((a[..., None] == _MID) & (np.arange(7) % 2 == 1)).sum(-1)
    return (gt + tie_up).astype(np.uint8)


def e4m3(v):
    """f32 -> e4m3fn (RNE, satfinite) as (bits uint8, value f32), via torch."""
    import torch
    t = torch.from_numpy(np.minimum(np.asarray(v, dtype=np.float32), 448.0))
    f8 = t.to(torch.float8_e4m3fn)
    return f8.view(torch.uint8).numpy(), f8.float().numpy()


def quantize(x, g):
    """x [R, K] f32 -> (packed uint8 [R, K/2] low-nibble-even, sf bits
    uint8 [R, K/16], sf values f32 [R, K/16]) — vLLM scaled_fp4_quant math."""
    x = np.asarray(x, dtype=np.float32)
    r, k = x.shape
    blk = x.reshape(r, k // 16, 16)
    amax = np.abs(blk).max(-1)
    sf_bits, sf = e4m3(amax * (np.float32(g) / np.float32(6.0)))
    inv = np.where(sf == 0, np.float32(0), np.float32(g) / np.where(sf == 0, 1, sf))
    scaled = (blk * inv[..., None]).astype(np.float32)
    code = e2m1_code(scaled) | np.where(scaled < 0, 8, 0).astype(np.uint8)
    code = code.reshape(r, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, sf_bits, sf


def dequant(packed, sf):
    """(packed [R, K/2], sf values [R, K/16]) -> f32 [R, K] (without 1/g)."""
    lo = packed & 0xF
    hi = packed >> 4
    code = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    val = E2M1[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    return (val.reshape(val.shape[0], -1, 16) * sf[..., None]).reshape(val.shape[0], -1)


def swizzle_sf(sf_bits):
    """vLLM swizzle_blockscale on uint8 [R, C] -> flat uint8 (padded)."""
    r, c = sf_bits.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    pad = np.zeros((rp, cp), np.uint8)
    pad[:r, :c] = sf_bits
    return pad.reshape(rp // 128, 4, 32, cp // 4, 4).transpose(0, 3, 2, 1, 4).reshape(-1)


def sf_offset(row, kb, kb_pad):
    """Kernel's `sf_offset` (must equal swizzle_sf's placement)."""
    atom = (row // 128) * (kb_pad // 4) + kb // 4
    return atom * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


@functools.lru_cache(maxsize=1)
def _weights(n, k, seed=0):
    """Quantized random weight in vLLM layout (cached: the oracle / bench
    walk M for one shape at a time)."""
    import torch
    rng = np.random.default_rng(seed)
    w = torch.from_numpy((rng.standard_normal((n, k)) * 0.05).astype(np.float32)).bfloat16()
    wf = w.float().numpy()
    g_w = np.float32(448.0 * 6.0 / np.abs(wf).max())
    w_packed, w_sf_bits, w_sf = quantize(wf, g_w)
    return dict(w_bf16=w, g_w=g_w, w_packed=w_packed, w_sf_bits=w_sf_bits, w_sf=w_sf,
                w_sf_swz=swizzle_sf(w_sf_bits), w_deq=(w_packed, dequant(w_packed, w_sf)))


def make_problem(m, n, k, seed=0):
    """Random bf16-representable x [m,k] (seeded by `seed`), quantized weight
    in vLLM layout (seeded per shape)."""
    rng = np.random.default_rng(1000 + seed)
    import torch
    x = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float32)).bfloat16()
    p = dict(_weights(n, k))
    g_x = np.float32(448.0 * 6.0 / np.abs(x.float().numpy()).max())
    p.update(x=x, g_x=g_x, alpha=np.float32(1.0 / (g_x * p["g_w"])))
    return p


def gemm_ref(p):
    """Exact fp32 result of the quantized problem (what the mma computes)."""
    xq, _, xsf = quantize(p["x"].float().numpy(), p["g_x"])
    a = dequant(xq, xsf)
    src, b = p.get("w_deq", (None, None))
    if src is not p["w_packed"]:  # caller swapped the weight: recompute
        b = dequant(p["w_packed"], p["w_sf"])
    return (a @ b.T * p["alpha"]).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU twin of nvfp4_gemm_t{tiles}'s addressing + the mma fragment layouts
# ---------------------------------------------------------------------------
def kernel_twin(p, mode=0, splits=1):
    """Replays nvfp4_gemm_t{tiles}'s per-lane u32 loads from flat buffers with
    the mxf4nvf4 m16n8k64 fragment/scale layouts (cute MMA_Traits
    SM120_16x8x64_TN_VS): CTA/warp column ownership (warps past N exit),
    one weight fragment per k64 step shared by `tiles` m16 A tiles, its
    split-K ranges, the [splits, M, N] partial layout and
    nvfp4_splitk_reduce's fixed-order sum; returns the fp32 [M, N] result.
    mode 0: our quant buffers ([16*tiles, K/2], row-major scales, a_rows =
    16*tiles, zero rows past M); mode 1: vLLM's pre-quantized activation
    ([M, K/2], 128x4-swizzled scales, a_rows = M). Any addressing bug
    (fragment rows/k, tile offsets, scale bytes, swizzles, split ranges,
    N tails) shows up here."""
    xf = p["x"].float().numpy()
    m, k = xf.shape
    n = p["w_packed"].shape[0]
    pl = plan(n, k, m, splits)
    tiles, kps, splits = pl["tiles"], pl["kps"], pl["splits"]
    xq, xsf_bits, _ = quantize(xf, p["g_x"])
    kh, nkb = k // 2, k // 16
    kb_pad = -(-nkb // 4) * 4
    if mode == 0:
        rows = 16 * tiles
        aq = np.zeros((rows, kh), np.uint8)
        asf_bits = np.zeros((rows, nkb), np.uint8)
        aq[:m], asf_bits[:m] = xq, xsf_bits
        a_rows, a_stride = rows, kh
        aq, asf = aq.reshape(-1), asf_bits.reshape(-1)
        sfa_off = lambda row, kb: row * nkb + kb
    else:
        aq, asf = xq.reshape(-1), swizzle_sf(xsf_bits)
        a_rows, a_stride = m, kh
        sfa_off = lambda row, kb: sf_offset(row, kb, kb_pad)
    w, wsf = p["w_packed"].reshape(-1), p["w_sf_swz"]
    f8 = lambda b: e4m3_bits_to_f32(np.asarray(b, np.uint8))
    partial = np.full((splits, m, n), np.nan, np.float32)

    def u32(buf, off, ok=True):
        return buf[off:off + 4] if ok else np.zeros(4, np.uint8)

    def nib(bytes4, j):
        byte = bytes4[j // 2]
        c = (byte >> 4) if j % 2 else (byte & 0xF)
        return E2M1[c & 7] * (-1.0 if c & 8 else 1.0)

    warps = [(cta * 4 + wp) * 8 for cta in range(pl["grid"][0]) for wp in range(4)]
    for sp in range(splits):
        ks = range(sp * kps * 64, min(k, (sp + 1) * kps * 64), 64)
        for n0 in warps:
            if n0 >= n:
                continue  # warp-uniform exit (N tail)
            acc = np.zeros((tiles, 16, 8), np.float64)
            for k0 in ks:
                off = k0 // 2
                B = np.full((8, 64), np.nan)
                SB = np.full((8, 4), np.nan)
                for lane in range(32):  # weight fragment: once per step
                    g, t = lane // 4, lane % 4
                    o = off + 4 * t
                    for r, reg in enumerate([u32(w, (n0 + g) * kh + o),
                                             u32(w, (n0 + g) * kh + o + 16)]):
                        for j in range(8):
                            _put(B, g, 8 * t + j + 32 * r, nib(reg, j))
                    sfb = u32(wsf, sf_offset(n0 + g, k0 // 16, kb_pad))
                    for b in range(4):
                        _put(SB, g, b, f8(sfb[b]))
                assert not np.isnan(B).any() and not np.isnan(SB).any()
                Bs = B * np.repeat(SB, 16, axis=1)
                for i in range(tiles):  # the tiles' A fragments
                    A = np.full((16, 64), np.nan)
                    SA = np.full((16, 4), np.nan)
                    for lane in range(32):
                        g, t = lane // 4, lane % 4
                        o = off + 4 * t
                        r0, r1 = 16 * i + g, 16 * i + g + 8
                        regs_a = [u32(aq, r0 * a_stride + o, r0 < a_rows),
                                  u32(aq, r1 * a_stride + o, r1 < a_rows),
                                  u32(aq, r0 * a_stride + o + 16, r0 < a_rows),
                                  u32(aq, r1 * a_stride + o + 16, r1 < a_rows)]
                        for r, reg in enumerate(regs_a):
                            for j in range(8):
                                _put(A, g + 8 * (r & 1), 8 * t + j + 32 * (r >> 1), nib(reg, j))
                        sr = 16 * i + 8 * (lane & 1) + g
                        sfa = u32(asf, sfa_off(sr, k0 // 16), sr < a_rows)
                        for b in range(4):
                            _put(SA, sr - 16 * i, b, f8(sfa[b]))
                    assert not np.isnan(A).any() and not np.isnan(SA).any()
                    acc[i] += (A * np.repeat(SA, 16, axis=1)) @ Bs.T
            for i in range(tiles):
                for row in range(16 * i, min(m, 16 * i + 16)):
                    partial[sp, row, n0:n0 + 8] = acc[i, row - 16 * i].astype(np.float32)
    out = np.zeros((m, n), np.float32)
    for sp in range(splits):  # nvfp4_splitk_reduce: fixed split order
        out += partial[sp]
    assert not np.isnan(out).any(), "output column/row never written"
    return (out * p["alpha"]).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU twin of the single-launch path (nvfp4_gemm_f{tiles}) vs the old one
# ---------------------------------------------------------------------------
def bf16_bits(x):
    """Kernel `f32_to_bf16` (RNE on the bits) -> uint16."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _u32(buf, off):
    """Little-endian u32 at byte offsets `off` (any shape) of uint8 `buf`."""
    b = buf.astype(np.uint32)
    return b[off] | b[off + 1] << 8 | b[off + 2] << 16 | b[off + 3] << 24


def quant_act_twin(x, g, tiles):
    """nvfp4_quant_act: aq [16*tiles, K/2], asf [16*tiles, K/16] (flat),
    rows >= M zero (`quantize` = the kernel's quant_block math)."""
    m, k = x.shape
    aq = np.zeros((16 * tiles, k // 2), np.uint8)
    asf = np.zeros((16 * tiles, k // 16), np.uint8)
    aq[:m], asf[:m] = quantize(x, g)[:2]
    return aq.reshape(-1), asf.reshape(-1)


def prologue_twin(x, g, kps, split, flags, splits, aq=None, asf=None):
    """gemm_fused's FUSE_QUANT prologue of one CTA: blocks i = (r, j) of
    rows < M over the split's K range -> quant_block -> smem A at 16 + r*astr
    + 8j, scales at 16 + m*astr + r*sstr + j; CTA x == 0 (aq/asf given) also
    writes the codes/scales to the separate kernel's aq/asf layout."""
    m, k = x.shape
    kb0, kb1 = split * kps * 4, min(k // 16, (split + 1) * kps * 4)
    astr, sstr = kps * 32 + 16, 4 * (kps | 1)
    smem = np.full(fused_layout(m, kps, splits, flags)[0], 0xAB, np.uint8)  # garbage
    q, sf = quantize(x[:, 16 * kb0:16 * kb1], g)[:2]
    for i in range(m * (kb1 - kb0)):
        r, j = divmod(i, kb1 - kb0)
        smem[16 + r * astr + 8 * j:16 + r * astr + 8 * j + 8] = q[r, 8 * j:8 * j + 8]
        smem[16 + m * astr + r * sstr + j] = sf[r, j]
        if aq is not None:
            kb = kb0 + j
            aq[r * (k // 2) + 8 * kb:r * (k // 2) + 8 * kb + 8] = q[r, 8 * j:8 * j + 8]
            asf[r * (k // 16) + kb] = sf[r, j]
    return smem, dict(aq=smem[16:], a_stride=astr, a_rows=m, k_base=16 * kb0,
                      asf=smem[16 + m * astr:], swz=False, sf_stride=sstr)


def a_operands(src, k0s, tiles, kb_pad):
    """load_a for every lane / tile / k64 step in `k0s` (ASrc semantics):
    -> (A u32 [steps, tiles, 32, 4], SFA u32 [steps, tiles, 32])."""
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    k0 = np.asarray(k0s)[:, None, None]
    tile = np.arange(tiles)[None, :, None]
    shape = (len(k0s), tiles, 32)
    off = np.broadcast_to((k0 - src["k_base"]) // 2 + 4 * t, shape)
    a = np.zeros((len(k0s), tiles, 32, 4), np.uint32)
    for reg, (dr, dk) in enumerate([(0, 0), (8, 0), (0, 16), (8, 16)]):
        r = np.broadcast_to(16 * tile + g + dr, shape)
        ok = r < src["a_rows"]
        o = np.where(ok, r * src["a_stride"] + off + dk, 0)
        a[..., reg] = np.where(ok, _u32(src["aq"], o), 0)
    sr = np.broadcast_to(16 * tile + 8 * (lane & 1) + g, shape)
    kb = np.broadcast_to(k0 // 16, shape)
    ok = sr < src["a_rows"]
    so = (sf_offset(sr, kb, kb_pad) if src["swz"]
          else sr * src["sf_stride"] + kb - src["k_base"] // 16)
    sfa = np.where(ok, _u32(src["asf"], np.where(ok, so, 0)), 0)
    return a, sfa


def splitk_reduce_twin(partial, alpha, raw=False):
    """nvfp4_splitk_reduce: bf16(alpha * (((0 + p0) + p1) + ...)), f32
    (raw: the f32 value before the bf16 rounding)."""
    acc = np.zeros(partial.shape[1:], np.float32)
    for sp in range(partial.shape[0]):
        acc = acc + partial[sp]
    acc = acc * np.float32(alpha)
    return acc if raw else bf16_bits(acc)


def fixup_twin(partial, alpha, n, ch, order, counters, out):
    """FUSE_REDUCE: CTAs (x, split) arrive in `order`; each draws a ticket
    (atom.inc: t = c; c = 0 if t >= S-1 else t+1) after its partial is
    visible; the drawer of S-1 stages ch splits at a time (the kernel's
    [ch][M*32] smem layout, cols >= N zero) and sums them into the running
    sums in split order 0..S-1, then writes bf16(acc*alpha) of its 32-col
    tile (f32 `out`: the f32 value before the bf16 rounding). Fails loudly
    if a fixup would read a partial not yet written."""
    sp_n, m, _ = partial.shape
    tile = m * 32
    written = set()
    for x, sp in order:
        written.add((x, sp))
        t = int(counters[x])
        counters[x] = 0 if t >= sp_n - 1 else t + 1
        if t != sp_n - 1:
            continue
        missing = [s_ for s_ in range(sp_n) if (x, s_) not in written]
        if missing:
            raise AssertionError(f"tile {x}: fixup before splits {missing} wrote partials")
        c0 = 32 * x
        col = c0 + np.arange(tile) % 32
        row = np.arange(tile) // 32
        valid = col < n
        acc_s = np.zeros(tile, np.float32)
        s0 = 0
        while s0 < sp_n:
            cn = min(ch, sp_n - s0)
            stg = np.zeros(cn * tile, np.float32)
            i = np.arange(cn * tile)
            ok = valid[i % tile]
            stg[ok] = partial[s0 + i[ok] // tile, row[i[ok] % tile], col[i[ok] % tile]]
            for s_ in range(cn):
                acc_s = acc_s + stg[s_ * tile:(s_ + 1) * tile]
            s0 += cn
        v = acc_s[valid] * np.float32(alpha)
        out[row[valid], col[valid]] = v if out.dtype == np.float32 else bf16_bits(v)
    return out


def fused_twin(p, mode=0, splits=None, fused="all", seed=0, counters=None):
    """Old path (quant kernel -> gemm_t -> reduce) vs the single-launch
    path of the SAME plan split count: asserts every A operand (u32
    fragment regs + scale words, every split / k64 step / m16 tile / lane)
    and the aq/asf readback rows < M bit-identical, then reduces fp32
    partials of those operands both ways (the fused fixup with CTAs
    arriving in a seeded random order through `counters`) and asserts the
    bf16 outputs bit-identical. Returns (out bf16 bits [M, N], plan,
    counters). mode 0 = bf16 route, 1 = prequant (vLLM swizzled scales)."""
    xf = p["x"].float().numpy()
    m, k = xf.shape
    n = p["w_packed"].shape[0]
    pl = plan(n, k, m, splits, fused, prequant=mode == 1)
    tiles, kps, sp_n = pl["tiles"], pl["kps"], pl["splits"]
    nkb = k // 16
    kb_pad = -(-nkb // 4) * 4
    if mode == 0:
        aq, asf = quant_act_twin(xf, p["g_x"], tiles)
        old = dict(aq=aq, a_stride=k // 2, a_rows=16 * tiles, k_base=0, asf=asf,
                   swz=False, sf_stride=nkb)
    else:
        xq, xsf_bits, _ = quantize(xf, p["g_x"])
        old = dict(aq=xq.reshape(-1), a_stride=k // 2, a_rows=m, k_base=0,
                   asf=swizzle_sf(xsf_bits), swz=True, sf_stride=nkb)
    ws_aq = np.full(16 * tiles * k // 2, 0xCD, np.uint8)  # stale workspace
    ws_asf = np.full(16 * tiles * nkb, 0xCD, np.uint8)
    xs, ws_sf = quantize(xf, p["g_x"])[:2]
    xdeq = dequant(xs, e4m3_bits_to_f32(ws_sf))
    wdeq = dequant(p["w_packed"], p["w_sf"])
    partial = np.zeros((sp_n, m, n), np.float32)
    for sp in range(sp_n):
        k0s = range(sp * kps * 64, min(k, (sp + 1) * kps * 64), 64)
        src = old
        if pl["fused_quant"]:
            _, src = prologue_twin(xf, p["g_x"], kps, sp, pl["flags"], sp_n, ws_aq, ws_asf)
        for got, want in zip(a_operands(src, k0s, tiles, kb_pad),
                             a_operands(old, k0s, tiles, kb_pad)):
            np.testing.assert_array_equal(got, want, err_msg=f"split {sp} A operands")
        ks = slice(k0s[0], k0s[-1] + 64)
        partial[sp] = (xdeq[:, ks].astype(np.float64) @ wdeq[:, ks].T).astype(np.float32)
    if pl["fused_quant"]:  # layer-oracle readback: every row < M, every block
        np.testing.assert_array_equal(ws_aq[:m * k // 2], old["aq"][:m * k // 2])
        np.testing.assert_array_equal(ws_asf[:m * nkb], old["asf"][:m * nkb])
    alpha = p["alpha"]
    want = splitk_reduce_twin(partial, alpha)
    if not pl["fused_reduce"]:
        return want, pl, counters
    ctas = -(-n // 32)
    counters = np.zeros(ctas, np.int64) if counters is None else counters
    rng = np.random.default_rng(seed)
    order = [(x, sp) for x in range(ctas) for sp in range(sp_n)]
    order = [order[i] for i in rng.permutation(len(order))]
    # f32 sums compared too (bf16 rounding hides most order changes), also on
    # random partials: NVFP4 products of short mantissas often sum exactly
    for part in (partial, rng.standard_normal(partial.shape).astype(np.float32)):
        raw = fixup_twin(part, alpha, n, pl["ch"], order, counters.copy(),
                         np.full((m, n), np.nan, np.float32))
        np.testing.assert_array_equal(raw, splitk_reduce_twin(part, alpha, raw=True))
    got = fixup_twin(partial, alpha, n, pl["ch"], order, counters,
                     np.full((m, n), 0xFFFF, np.uint16))
    np.testing.assert_array_equal(got, want, err_msg="fused fixup vs splitk_reduce")
    return got, pl, counters


def _put(arr, i, j, v):
    if not np.isnan(arr[i, j]) and arr[i, j] != v:
        raise AssertionError(f"fragment conflict at {(i, j)}: {arr[i, j]} vs {v}")
    arr[i, j] = v


def e4m3_bits_to_f32(b):
    import torch
    return torch.from_numpy(np.atleast_1d(b).astype(np.uint8)).view(
        torch.float8_e4m3fn).float().numpy().reshape(np.shape(b))


def unswizzle_sf(sf_swz, rows: int, nkb: int):
    """torch: vLLM 128x4-swizzled scale buffer (any shape / e4m3 or uint8)
    -> uint8 [rows, nkb] row-major (inverse of `swizzle_sf`)."""
    import torch
    rp, cp = -(-rows // 128) * 128, -(-nkb // 4) * 4
    flat = sf_swz.reshape(-1).view(torch.uint8)[:rp * cp]
    return (flat.reshape(rp // 128, cp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
            .reshape(rp, cp)[:rows, :nkb])


def dequant_torch(packed, sf_u8):
    """torch f64 exact dequant: packed uint8 [R, K/2] (low nibble = even k),
    row-major e4m3 scale bits uint8 [R, K/16] -> [R, K] (without 1/g)."""
    import torch
    lut = torch.tensor(np.concatenate([E2M1, -E2M1]), dtype=torch.float64,
                       device=packed.device)
    r = packed.shape[0]
    code = torch.stack([packed & 15, packed >> 4], -1).reshape(r, -1).long()
    sf = sf_u8.contiguous().view(torch.float8_e4m3fn).double()
    return (lut[code].reshape(r, sf.shape[1], 16) * sf[..., None]).reshape(r, -1)


def exact_ref(xq, xsf, w, wsf, alpha: float, n: int, chunk_elems: int = 1 << 24):
    """torch f64 alpha * deq(xq) @ deq(w[:n]).T from vLLM-layout tensors
    (xq [M, K/2], w [>=N, K/2], both scale buffers 128x4-swizzled); weight
    rows dequantized in chunks (load-time memory bound)."""
    import torch
    m, k = xq.shape[0], xq.shape[1] * 2
    a = dequant_torch(xq.contiguous(), unswizzle_sf(xsf, m, k // 16))
    wsf_u8 = unswizzle_sf(wsf, n, k // 16)
    out = torch.empty(m, n, dtype=torch.float64, device=xq.device)
    step = max(1, chunk_elems // k)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        out[:, r0:r1] = a @ dequant_torch(w[r0:r1], wsf_u8[r0:r1]).T
    return out * alpha


# ---------------------------------------------------------------------------
# on-silicon oracle + bench
# ---------------------------------------------------------------------------
def _dev_problem(m, n, k, dev, seed=0, max_splits=None):
    import torch
    p = make_problem(m, n, k, seed)
    rows = 16 * tiles_for(m)
    s = plan(n, k, m)["splits"] if max_splits is None else max_splits
    t = dict(
        x=p["x"].to(dev),
        w=torch.from_numpy(p["w_packed"]).to(dev),
        w_sf=torch.from_numpy(p["w_sf_swz"]).to(dev).view(torch.float8_e4m3fn),
        aq=torch.empty(rows, k // 2, dtype=torch.uint8, device=dev),
        asf=torch.empty(rows, k // 16, dtype=torch.uint8, device=dev),
        partial=torch.empty(max(1, s * m * n), dtype=torch.float32, device=dev),
        out=torch.empty(m, n, dtype=torch.bfloat16, device=dev),
        counters=torch.zeros(-(-n // 32), dtype=torch.int32, device=dev),
        g=torch.tensor([p["g_x"]], dtype=torch.float32, device=dev),
        alpha=torch.tensor([p["alpha"]], dtype=torch.float32, device=dev),
        g_f=float(p["g_x"]), alpha_f=float(p["alpha"]),
    )
    t["w_sf_2d"] = t["w_sf"].view(-(-n // 128) * 128, -1)
    # FlashInfer CUTLASS needs N % 32: pad rows like vLLM's
    # pad_nvfp4_weight_for_cutlass (the swizzled scales already are).
    t["w_fi"] = torch.nn.functional.pad(t["w"], (0, 0, 0, -n % 32)).contiguous()
    return p, t


def _ours(native, t, stream, splits=None, fused=None):
    n, k, m = t["w"].shape[0], t["x"].shape[1], t["x"].shape[0]
    pl = plan(n, k, m, splits, fused)
    native.nvfp4_gemm_cuda(t["x"], t["w"], t["w_sf"], t["aq"], t["asf"], t["partial"],
                           t["out"], t["g_f"], t["alpha_f"], pl["splits"], stream,
                           pl["flags"], t["counters"])
    return t["out"]


def _ours_q(native, t, xq, xsf, stream, splits=None, fused=None):
    n, k, m = t["w"].shape[0], t["x"].shape[1], t["x"].shape[0]
    pl = plan(n, k, m, splits, fused, prequant=True)
    native.nvfp4_gemm_q_cuda(xq, xsf, t["w"], t["w_sf"], t["partial"], t["out"],
                             t["alpha_f"], pl["splits"], stream, pl["flags"], t["counters"])
    return t["out"]


def fused_vs_old(run, aq, asf, counters, fused_quant: bool) -> str | None:
    """Single-launch vs old 3-launch path on silicon: outputs bit-identical,
    (fused_quant) the prologue's aq/asf readback == the quant kernel's
    (`aq` / `asf`: views of the workspace rows < M), and a second (replayed)
    fused call bit-identical with every ticket counter back at 0.
    `run(fused)` launches ours and returns its output. None = OK."""
    import torch
    old = run(False).clone()
    want = aq.clone(), asf.clone()
    aq.fill_(0xCD)
    asf.fill_(0xCD)
    for rep in range(2):
        got = run(True)
        if not torch.equal(got.view(torch.int16), old.view(torch.int16)):
            bad = int((got.view(torch.int16) != old.view(torch.int16)).sum())
            return f"fused out != old out ({bad} elements, call {rep})"
        if int(counters.abs().sum()) != 0:
            return f"ticket counters not reset after call {rep}"
    if fused_quant and not (torch.equal(aq, want[0]) and torch.equal(asf, want[1])):
        return "fused quant aq/asf readback != nvfp4_quant_act"
    aq.copy_(want[0])
    asf.copy_(want[1])
    return None


def _vllm(t, n):
    """vLLM's own NVFP4 linear ops on the same tensors: scaled_fp4_quant +
    FlashInfer CUTLASS SM120 mm (the default backend), falling back to
    vLLM's cutlass_scaled_fp4_mm."""
    import torch
    from vllm import _custom_ops as ops
    xq, xsf = ops.scaled_fp4_quant(t["x"], t["g"])
    try:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
        y = flashinfer_scaled_fp4_mm(xq, t["w_fi"], xsf, t["w_sf_2d"], t["alpha"],
                                     torch.bfloat16, backend="cutlass")
    except Exception:  # FlashInfer absent: vLLM's own CUTLASS kernel
        y = ops.cutlass_scaled_fp4_mm(xq, t["w_fi"], xsf, t["w_sf_2d"], t["alpha"],
                                      torch.bfloat16)
    return y[:, :n]


def _native_ready():
    import torch
    from suffix_hybrid import oxide_kernels
    native = oxide_kernels.native()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("NVFP4 GEMM cubin is sm_120a SASS (cc 12.x only)")
    oxide_kernels.ensure_loaded(FAMILY, params=PARAMS)  # ABI + sha256 + cuModuleLoadData
    return native


def oracle_ok(rel_vs_ref: float, rel_vs_vllm: float, vllm_rel_vs_ref: float) -> bool:
    """Pass iff we agree with vLLM's own NVFP4 GEMM (<= 2e-2) AND our error
    vs the exact fp32 reference is within 10 % of vLLM's own (floor 1e-2):
    the fp32 reference is stricter than vLLM itself meets (bf16 output
    rounding at large K*|x| — silicon 87fdbc80: gate_up prequant M=16
    rel_vs_ref 1.27e-2 with rel_vs_vllm 0)."""
    # Triangle bound vs vLLM (silicon 2026-09-29, qwen gate_up M=16: ours ==
    # spec exactly, vLLM 2.3e-2 away: its scaled_fp4_quant uses rcp.approx).
    return (rel_vs_ref <= max(1e-2, 1.1 * vllm_rel_vs_ref)
            and rel_vs_vllm <= max(2e-2, 1.1 * (rel_vs_ref + vllm_rel_vs_ref)))


ORACLE_MS = (1, 5, 16, 17, 40, 64)
BENCH_MS = (1, 5, 8, 16, 24, 40, 64)


def oracle(ms=ORACLE_MS, shapes=("mlp_gate_up", "mlp_down", "gdn_in_proj_qkvz", "attn_o",
                                 "gemma_mlp_down", *QWEN_FLASH_SHAPES)):
    """Both entry paths (bf16 x -> our quant; vLLM-prequantized x with
    swizzled scales, i.e. the fused SiLU*mul / RMSNorm quant route) with the
    production M tiling + split-K plan, vs the exact reference and vs vLLM's
    op; plus the single-launch path bit-identical to the old 3-launch one
    (`fused_vs_old`: outputs, aq/asf readback, counter reset on replay)."""
    import torch
    from vllm import _custom_ops as ops
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev).cuda_stream
    lines = []
    for name in shapes:
        n, k = ALL_SHAPES[name]
        for m in ms:
            p, t = _dev_problem(m, n, k, dev, seed=m)
            ref = torch.from_numpy(gemm_ref(p)).to(dev)
            ref16 = ref.bfloat16().float()
            vl = _vllm(t, n).float()
            vl_rel = float((vl - ref16).norm() / ref16.norm())  # vLLM's own error
            xq, xsf = ops.scaled_fp4_quant(t["x"], t["g"])
            for path in ("bf16", "prequant"):
                run = ((lambda f: _ours(native, t, stream, fused=f)) if path == "bf16"
                       else (lambda f: _ours_q(native, t, xq, xsf, stream, fused=f)))
                why = fused_vs_old(run, t["aq"][:m], t["asf"][:m], t["counters"],
                                   path == "bf16" and plan(n, k, m, fused=True)["fused_quant"])
                if why is not None:
                    raise RuntimeError(f"{MARKER} NVFP4-GEMM ORACLE FAIL: {name} {path} M={m} "
                                       f"N={n} K={k}: {why}")
                ours = run(None).float()  # the production plan
                rel = float((ours - ref16).norm() / ref16.norm())
                cos_v = float(torch.nn.functional.cosine_similarity(
                    ours.flatten(), ref.flatten(), dim=0))
                rel_v = float((ours - vl).norm() / vl.norm())
                ok = oracle_ok(rel, rel_v, vl_rel)
                lines.append(
                    f"{name} {path} M={m} N={n} K={k} "
                    f"{plan_str(plan(n, k, m, prequant=path != 'bf16'))} fused==old bits: "
                    f"rel_vs_ref={rel:.2e} cos={cos_v:.6f} rel_vs_vllm={rel_v:.2e} "
                    f"vllm_rel_vs_ref={vl_rel:.2e} {'OK' if ok else 'FAIL'}")
                if not ok:
                    raise RuntimeError(f"{MARKER} NVFP4-GEMM ORACLE FAIL: {lines[-1]}")
    for ln in lines:
        print(f"{MARKER} {ln}", file=sys.stderr, flush=True)
    return f"{MARKER} NVFP4-GEMM ORACLE PASS ({len(lines)} cases, sm_120a mxf4nvf4 mma)"


def _graph_us(fn, dev, iters):
    import torch
    s = torch.cuda.Stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        fn(s)  # warm (FlashInfer tactic selection happens here)
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn(s)
    torch.cuda.current_stream(dev).wait_stream(s)
    g.replay()
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize(dev)
    return e0.elapsed_time(e1) * 1000.0 / iters


def route_from_bench(res: dict, ms=BENCH_MS) -> dict:
    """{(N, K): largest benched M such that ours won at every benched M' <=
    it} from bench results {(N, K, M): (vllm_us, ours_us)}; 0 = never."""
    out = {}
    for (n, k) in dict.fromkeys((n, k) for n, k, _ in res):
        best = 0
        for m in sorted(ms):
            if (n, k, m) not in res:
                continue
            v, o = res[(n, k, m)]
            if o > v:
                break
            best = m
        out[(n, k)] = best
    return out


def bench(ms=BENCH_MS, shapes=tuple(ALL_SHAPES), iters=200, sweep=False):
    """us per call (quant + gemm), each path captured in one CUDA graph, vs
    vLLM's FlashInfer CUTLASS (quant + gemm); per (shape, M) also the old
    3-launch path vs the single-launch one (+ reduce-only fusion where the
    plan fuses the quant), and per-M totals. sweep: also time split counts
    {1, 2, 4, ..., 64} and print the best (heuristic retuning). Ends with the
    SUFFIX_NVFP4_GEMM_ROUTE string that keeps every losing (shape, M) on
    FlashInfer."""
    native = _native_ready()
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    res, fres = {}, {}
    for name in shapes:
        n, k = ALL_SHAPES[name]
        if any((n, k) == key[:2] for key in res):
            continue  # same (N, K) under another name
        for m in ms:
            pl = plan(n, k, m)
            cands = sorted({pl["splits"], *(s for s in (1, 2, 4, 8, 16, 32, 64)
                                             if s <= k // 64)}) if sweep else [pl["splits"]]
            p, t = _dev_problem(m, n, k, dev, seed=1, max_splits=max(cands))
            t_v = _graph_us(lambda s: _vllm(t, n), dev, iters)
            t_o = {c: _graph_us(lambda s, c=c: _ours(native, t, s.cuda_stream, c), dev, iters)
                   for c in cands}
            ours = t_o[pl["splits"]]
            res[(n, k, m)] = (t_v, ours)
            # old 3-launch path vs single launch (and reduce-only fusion when
            # the plan also fuses the quant: retunes FUSED_QUANT_MAX_M)
            fz = {f: _graph_us(lambda s, f=f: _ours(native, t, s.cuda_stream, fused=f), dev, iters)
                  for f in (False, True, *(("reduce",) if plan(n, k, m, fused=True)["fused_quant"]
                                           else ()))}
            fres[(n, k, m)] = fz
            roof = (n * k // 2 + n * k // 16) / 1.79e12 * 1e6
            best = min(t_o, key=t_o.get)
            extra = (f"; sweep best splits={plan(n, k, m, best)['splits']} "
                     f"{t_o[best]:.1f} us" if sweep else "")
            print(f"{MARKER} bench {name} M={m} N={n} K={k} {plan_str(pl)}: vllm "
                  f"{t_v:.1f} us, ours {ours:.1f} us (x{ours / t_v:.2f} "
                  f"{'WIN' if ours <= t_v else 'LOSE'}), weight-roofline {roof:.1f} us"
                  f"{extra}", file=sys.stderr, flush=True)
            print(f"{MARKER} bench fused {name} M={m} N={n} K={k}: old 3-launch "
                  f"{fz[False]:.1f} us ({plan(n, k, m, fused=False)['launches']} launches), "
                  f"fused {fz[True]:.1f} us ({plan(n, k, m, fused=True)['launches']} launch) "
                  f"delta {fz[True] - fz[False]:+.1f} us"
                  + (f", reduce-only {fz['reduce']:.1f} us" if "reduce" in fz else ""),
                  file=sys.stderr, flush=True)
    for m in ms:
        tot = [sum(fres[key][f] for key in fres if key[2] == m) for f in (False, True)]
        print(f"{MARKER} bench fused total M={m} over {sum(key[2] == m for key in fres)} shapes: "
              f"old {tot[0]:.1f} us, fused {tot[1]:.1f} us ({tot[1] - tot[0]:+.1f} us)",
              file=sys.stderr, flush=True)
    route = route_from_bench(res, ms)
    lose = ",".join(f"{n}x{k}:{mm}" for (n, k), mm in route.items() if mm < max(ms))
    print(f"{MARKER} bench route: ours wins up to M per (N x K): "
          + " ".join(f"{n}x{k}<={mm}" for (n, k), mm in route.items()),
          file=sys.stderr, flush=True)
    print(f"{MARKER} bench suggested SUFFIX_NVFP4_GEMM_ROUTE={lose or '(none: ours wins all)'}",
          file=sys.stderr, flush=True)
    return res


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "both"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("bench", "both", "sweep"):
            bench(sweep=mode == "sweep")
        return 0
    except Exception as exc:
        print(f"{MARKER} NVFP4-GEMM {mode.upper()} FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
