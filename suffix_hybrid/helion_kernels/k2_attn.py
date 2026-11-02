# SPDX-License-Identifier: Apache-2.0
"""K2-NVFP4 on Helion: our SM120 NVFP4-KV paged decode / spec-verify attention
(port of kernels-oxide/k2_nvfp4_attn, host op src/nvfp4_attn_oxide.rs, lane
sm120/nvfp4_kv_patch/own_attn.py). Knob ``SUFFIX_HELION_K2_ATTN`` =
oxide (default; this module is never imported) | helion | auto.

ONE fused kernel replaces both oxide entries (nvfp4_attn_partial +
nvfp4_attn_merge, register variants w1/w2/w3). Program = (request b, kv head
h, KV split s); grid = (B, HKV, NS) — independent of seq_lens and of the q
lengths, so a captured CUDA graph replays for any KV lengths (like oxide).

  * rows = request b's q tokens x the G q heads of kv head h, token-major
    (row r -> token r // G, head h*G + r % G), tiled by BR (tunable); ragged
    q lengths come from qo_indptr on the device, rows past them are tile
    masked, so every KV byte a program streams feeds all its rows;
  * KV: the request's causal(/window) range [lo, kv_len) in BN-token tiles
    (tunable, BN divides the page, tiles page-aligned: one block_table read
    per tile), cut into NS (tunable) splits of >= 2 tiles;
  * K/V dequantized in registers: e2m1 nibble -> f16 bits (exact), x e4m3
    block scale as f16 (exact: every e2m1 x e4m3 product is an f16). The head
    dim is split even/odd (low/high nibble of byte j = dims 2j / 2j+1):
    S = Qe Klo^T + Qo Khi^T, O = [P Vlo | P Vhi] interleaved — no shuffles;
  * f16 Q/K/P/V, fp32 accumulate, exp2 online softmax (-inf safe); each
    split stores its normalized partial (bf16) + lse (fp32, log2 domain);
  * the LAST split program of (b, h) to arrive (acq_rel atomic counter) does
    the LSE merge, x v_scale, bf16 store, and resets the counter to 0 (graph
    replay safe). The last program also zeroes graph-padding rows past
    qo_indptr[B] (oxide merge semantics).

Numerics = own_attn_ref.kernel_twin (f16 operands, bf16 partials, fp32
softmax) up to split boundaries; gated by the oracle's COS/REL (0.9995 / 2e-2).

Serving never autotunes: configs ship in configs/helion/k2_attn/<platform>.json
keyed by the case (head_dim, heads, kv_heads, page, window, width bucket,
numtokens = batch bucket). Context length is NOT a key: graphs are captured
per (batch, width) and the host never sees seq_lens (H22), so one config must
serve every context — the lab tunes on a realistic 2k..32k spread.
"""
from __future__ import annotations

import math

import torch

from suffix_hybrid import helion_kernels as hk

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU-only console/CI host: the lab ABI must still import
    helion = hl = None

ENV = "SUFFIX_HELION_K2_ATTN"
KERNEL = "k2_attn"
MARKER = "[suffix helion k2]"
LOG2E = 1.4426950408889634
WINDOW_FULL = 1 << 30           # window_left < 0 (full attention) in the body
BATCHES = (1, 8, 32)            # numtokens buckets (decode batch)
# Keep each supported width distinct. A width-4 win says nothing about the
# width-2/k1 or width-6/k5 shapes in a dynamic speculative schedule. Existing
# width-1/4 configs retain their keys; other widths stay on oxide until tuned.
WIDTHS = tuple(range(1, 9))
PAGE = 64                       # gemma pools' KV page size (oracle PAGE)
# Gemma 4 per-rank attention layouts beyond model_dims(): full-attention
# layers use global_head_dim = 2 x head_dim with num_global_key_value_heads =
# kv_heads / 4 (26B-A4B tp1: 16 q, 8/2 kv; 31B tp2: same per rank); sliding
# layers window 1024 (FlashInfer window_left 1023).
SLIDING_WINDOW_LEFT = 1023
# Gates of the on-silicon oracle (sm120/nvfp4_kv_patch/oracle.py).
COS_MIN = 0.9995
REL_MAX = 2e-2
# index_dtype: KV caches exceed 2^31 bytes. autotune_max_generations: LFBO full
# runs up to 20 generations at ~2 min each for b32 x 32k contexts, past the lab's
# 30 min case limit (run 20261017T172102Z-rtx-pro-6000-9236); searches converged
# by generation 7-10 in every run so far. LFBO's subsequent polish descent
# has its own rounds (not covered by max_generations): the 5090 d512/b1/w4
# experiment spent another ~10 minutes rechecking an unchanged incumbent.
# Budget the whole search, including polish, below the lab's 30-minute case
# timeout so the best-so-far config is checked and saved instead of lost.
# Serving binds shipped configs and never runs this search.
# The lab defaults to static shapes; serving uses dynamic shapes within
# each batch/width bucket. Override the lab default to tune that same code.
SETTINGS = {"index_dtype": torch.int64, "static_shapes": False, "autotune_max_generations": 10,
            "autotune_budget_seconds": 900}
MAX_COUNTERS = 1 << 16                   # B * HKV per launch


def _e2m1_f16(code):
    """e2m1 codes (int32 0..15) -> exact float16 (bit construction)."""
    m = code & 7
    bits = torch.where(m >= 2, (m << 9) + 0x3800, m * 0x3800) | ((code & 8) << 12)
    return bits.to(torch.int16).view(torch.float16)


def _k2_nvfp4_attn(q: torch.Tensor, kd: torch.Tensor, ks: torch.Tensor, vd: torch.Tensor,
                   vs: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor,
                   qo_indptr: torch.Tensor, out: torch.Tensor, counters: torch.Tensor,
                   kv_heads: int, page_size: int, page_bytes: int, window: int,
                   qk_scale_log2: float, v_scale: float) -> torch.Tensor:
    """q/out [T, HQ, D] bf16; kd/vd/ks/vs = body_args() row views of the
    paged cache (one 1-D row index per token: no scalar page indexer);
    counters int32 zeros (left zero)."""
    hq = hl.specialize(q.size(1))
    d = hl.specialize(q.size(2))
    sd = hl.specialize(kd.size(1))
    hl.specialize([kd.size(2), vd.size(1), vd.size(2), ks.size(1), vs.size(1)])
    hkv = hl.specialize(kv_heads)
    page = hl.specialize(page_size)
    g = hq // hkv
    dh = d // 2
    rows_d = page_bytes // dh   # data rows (D/2 bytes) per page
    rows_s = page_bytes // sd   # K-scale rows (D/16 bytes) per page
    batch = qo_indptr.size(0) - 1
    q3 = q.reshape(-1, dh, 2)
    out3 = out.view(-1, dh, 2)
    n_rows = q3.size(0)
    bn = hl.register_block_size(16, min(page & -page, 128))
    br = hl.register_block_size(16, 64)
    ns = hl.register_block_size(1, 128)
    o_part = torch.empty([ns, n_rows, dh, 2], dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty([ns, n_rows], dtype=torch.float32, device=q.device)
    for b, h, s in hl.grid([batch, hkv, ns]):
        qs0 = qo_indptr[b]
        q_len = qo_indptr[b + 1] - qs0
        kv_len = seq_lens[b]
        q0 = kv_len - q_len
        lo = torch.clamp(q0 - window, min=0)
        lo_al = (lo // bn) * bn
        n_t = (kv_len - lo_al + bn - 1) // bn
        per = torch.clamp((n_t + ns - 1) // ns, min=2) * bn
        t0 = lo_al + s * per
        t1 = torch.minimum(t0 + per, kv_len)
        for tile_r in hl.tile(0, q_len * g, block_size=br):
            r = tile_r.index
            row = (qs0 + r // g) * hq + h * g + r % g
            qpos = q0 + r // g
            qe, qo = hl.split(q3[row, :, :])
            qe = qe.to(torch.float16)
            qo = qo.to(torch.float16)
            m_i = hl.full([tile_r], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_r], dtype=torch.float32)
            acc_e = hl.zeros([tile_r, dh], dtype=torch.float32)
            acc_o = hl.zeros([tile_r, dh], dtype=torch.float32)
            for tile_n in hl.tile(t0, t1, block_size=bn):
                pg = block_table[b, tile_n.begin // page]
                kpos = tile_n.index
                tok = kpos % page
                rd = pg * rows_d + h * page + tok
                kb = kd[rd, :, :].to(torch.int32)
                ksc = ks[pg * rows_s + h * page + tok, :].to(torch.float16)[:, :, None]
                klo = (_e2m1_f16(kb & 15) * ksc).reshape(kb.size(0), dh)
                khi = (_e2m1_f16(kb >> 4) * ksc).reshape(kb.size(0), dh)
                sc = hl.dot(qe, klo.T, out_dtype=torch.float32)
                sc = hl.dot(qo, khi.T, acc=sc) * qk_scale_log2
                ok = ((kpos[None, :] <= qpos[:, None])
                      & (kpos[None, :] + window >= qpos[:, None])
                      & (kpos[None, :] < kv_len))
                sc = torch.where(ok, sc, float("-inf"))
                m_new = torch.maximum(m_i, torch.amax(sc, 1))
                m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
                p = torch.exp2(sc - m_safe[:, None])
                alpha = torch.exp2(m_i - m_safe)
                l_i = l_i * alpha + torch.sum(p, 1)
                m_i = m_new
                vb = vd[rd, :, :].to(torch.int32)
                rv = pg * page_bytes + h * (page * sd) + (tok // 4) * (4 * sd) + tok % 4
                vsc = vs[rv, :].to(torch.float16)[:, :, None]
                # stale tail tokens may hold e4m3 NaN codes: 0 * NaN poisons O
                vok = (kpos < kv_len)[:, None]
                vlo = torch.where(vok, (_e2m1_f16(vb & 15) * vsc).reshape(vb.size(0), dh), 0.0)
                vhi = torch.where(vok, (_e2m1_f16(vb >> 4) * vsc).reshape(vb.size(0), dh), 0.0)
                p16 = p.to(torch.float16)
                acc_e = hl.dot(p16, vlo, acc=acc_e * alpha[:, None])
                acc_o = hl.dot(p16, vhi, acc=acc_o * alpha[:, None])
            l_ok = l_i > 0.0
            l_one = torch.where(l_ok, l_i, 1.0)
            inv = torch.where(l_ok, 1.0 / l_one, 0.0)
            o_part[s, row, :, :] = hl.join(acc_e * inv[:, None], acc_o * inv[:, None]).to(torch.bfloat16)
            lse_part[s, row] = torch.where(l_ok, m_i + torch.log2(l_one), float("-inf"))
        old = hl.atomic_add(counters, [b * hkv + h], 1, sem="acq_rel")
        if old == ns - 1:
            # every split of (b, h) has stored its partial: merge them
            for tile_r in hl.tile(0, q_len * g, block_size=br):
                r = tile_r.index
                row = (qs0 + r // g) * hq + h * g + r % g
                lse_all = lse_part[:, row]
                mx = torch.amax(lse_all, 0)
                mx = torch.where(mx == float("-inf"), 0.0, mx)
                wsum = torch.sum(torch.exp2(lse_all - mx[None, :]), 0)
                acc = hl.zeros([tile_r, dh, 2], dtype=torch.float32)
                for tile_s in hl.tile(ns):
                    w = torch.exp2(lse_part[tile_s, row] - mx[None, :])
                    acc = acc + torch.sum(o_part[tile_s, row, :, :].to(torch.float32)
                                          * w[:, :, None, None], 0)
                w_ok = wsum > 0.0
                inv_w = torch.where(w_ok, v_scale / torch.where(w_ok, wsum, 1.0), 0.0)
                out3[row, :, :] = (acc * inv_w[:, None, None]).to(torch.bfloat16)
            counters[b * hkv + h] = 0
        if (b * hkv + h) * ns + s == batch * hkv * ns - 1:
            # CUDA-graph padding tokens past the last request: zero them
            for tile_z in hl.tile(qo_indptr[batch] * hq, n_rows, block_size=br):
                out3[tile_z, :, :] = hl.zeros([tile_z, dh, 2], dtype=torch.bfloat16)
    return out


# ---- host side ----------------------------------------------------------------
def body_args(q, k_data, k_sf, v_data, v_sf, block_table, seq_lens, qo_indptr, out,
              counters, window_left, qk_scale_log2, v_scale):
    """The oxide op's views (nvfp4_split_data_scale, HND: one page stride
    `pb`, (head, token, byte) dense per side) -> the body's args: row views
    with the page folded into ONE row index per token, so every gather has a
    single 1-D tensor indexer. Data row (pg, h, t) = pg*pb/(D/2) + h*PAGE + t;
    K-scale row = pg*pb/(D/16) + h*PAGE + t; V scales follow the store
    kernel's 4-token swizzle (t, g) -> (t//4)*4S + 4g + t%4 per (page, head):
    a byte-offset row view with column stride 4."""
    p, hkv, page, dh = k_data.shape
    sd = k_sf.shape[-1]
    pb = k_data.stride(0)
    for name, t, w in (("k_data", k_data, dh), ("v_data", v_data, dh),
                       ("k_sf", k_sf, sd), ("v_sf", v_sf, sd)):
        if (tuple(t.shape) != (p, hkv, page, w) or t.stride() != (pb, page * w, w, 1)):
            raise ValueError(f"{MARKER} {name}: need the HND nvfp4 view [{p}, {hkv}, {page}, {w}] "
                             f"with page stride {pb}, got {tuple(t.shape)} {t.stride()}")
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"{MARKER} q/out must be bf16 (out contiguous)")
    if page % 16 or dh != 8 * sd or pb % dh:
        raise ValueError(f"{MARKER} page {page} (need a multiple of 16), head_dim {2 * dh}, "
                         f"page stride {pb} (need a multiple of D/2)")

    def rows(t, n, size, stride):
        return torch.as_strided(t, (n, *size), stride, t.storage_offset())

    nd = (p - 1) * (pb // dh) + hkv * page
    kd = rows(k_data, nd, (sd, 8), (dh, 8, 1))
    vd = rows(v_data, nd, (sd, 8), (dh, 8, 1))
    ks = rows(k_sf, (p - 1) * (pb // sd) + hkv * page, (sd,), (sd, 1))
    vs = rows(v_sf, (p - 1) * pb + hkv * page * sd - 4 * sd + 4, (sd,), (1, 4))
    return (q.contiguous(), kd, ks, vd, vs, block_table, seq_lens, qo_indptr, out, counters,
            hkv, page, pb, WINDOW_FULL if window_left < 0 else int(window_left),
            float(qk_scale_log2), float(v_scale))


def natural_views(args):
    """body_args() row views -> the oxide op's (k_data, k_sf, v_data, v_sf)."""
    kd, ks, vd, vs, hkv, page, pb = (*args[1:5], *args[10:13])
    sd = kd.shape[1]
    dh = 8 * sd
    p = (kd.shape[0] - hkv * page) // (pb // dh) + 1

    def view(t, w):
        return torch.as_strided(t, (p, hkv, page, w), (pb, page * w, w, 1), t.storage_offset())

    return view(kd, dh), view(ks, sd), view(vd, dh), view(vs, sd)


def case(d, hq, hkv, page, window_left, batch, q_max) -> dict:
    """The tuned case a launch runs as (bucketed batch and q width)."""
    return {"head_dim": d, "heads": hq, "kv_heads": hkv, "page": page,
            "window": -1 if window_left < 0 else int(window_left),
            "width": int(q_max), "numtokens": hk.bucket(batch, BATCHES)}


_ROUTE: dict = {}
_FAST: dict = {}
_PREPARED: dict = {}
_COUNTERS: dict = {}


def kernel_for(c: dict):
    """Compiled Helion kernel for case `c`, or None (auto: oxide serves it).
    helion with no shipped config is a hard error (fail closed)."""
    key = hk.case_key(c)
    if key not in _ROUTE:
        mode = hk.backend(ENV)
        if hk.config(KERNEL, c) is None:
            if mode == "helion":
                raise RuntimeError(
                    f"{ENV}=helion but no tuned {KERNEL} config for {key} on {hk.platform()} "
                    f"(tune it in the autotune lab or use {ENV}=auto)")
            _ROUTE[key] = None
        else:
            _ROUTE[key] = hk.compiled(_k2_nvfp4_attn, KERNEL, c, **SETTINGS)
    return _ROUTE[key]


def counters(device) -> torch.Tensor:
    """Per-device split-arrival counters (zeros; every launch leaves them
    zero). Allocated in prepare(), before CUDA-graph capture."""
    dev = torch.device(device)
    buf = _COUNTERS.get(dev)
    if buf is None:
        buf = _COUNTERS[dev] = torch.zeros(MAX_COUNTERS, dtype=torch.int32, device=dev)
    return buf


def run(q, k_data, k_sf, v_data, v_sf, block_table, seq_lens, qo_indptr, out, q_max,
        window_left, sm_scale, v_scale, max_rows=48):
    """own_attn.run's signature (max_rows is oxide tile planning, unused).
    Returns `out`, or None when this shape stays on oxide (auto, no config)."""
    tokens, hq, d = q.shape
    _, hkv, page, _ = k_data.shape
    batch = qo_indptr.shape[0] - 1
    rkey = (d, hq, hkv, page, window_left, batch, q_max)
    kern = _FAST.get(rkey, _FAST)
    if kern is _FAST:  # first launch of this shape: one dict hit per call after
        kern = _FAST[rkey] = kernel_for(case(d, hq, hkv, page, window_left, batch, q_max))
    if kern is None:
        return None
    if tokens == 0 or batch <= 0:
        return out
    if batch * hkv > MAX_COUNTERS:
        raise ValueError(f"{MARKER} batch {batch} x {hkv} kv heads > {MAX_COUNTERS} counters")
    kern(*body_args(q, k_data, k_sf, v_data, v_sf, block_table, seq_lens, qo_indptr, out,
                    counters(q.device), window_left, sm_scale * LOG2E, v_scale))
    return out


def prepare(served, q_max, device) -> str:
    """Load-time: compile every Helion case of this layer shape (all batch and
    width buckets <= q_max) and gate it against the reference on a small
    random cache — before any CUDA-graph capture. auto lists what stays oxide."""
    d, hq, hkv, page, wl = served
    counters(device)
    key = (tuple(served), q_max, str(device))
    if key in _PREPARED:  # one builder per KV group asks for the same shape
        return _PREPARED[key]
    on, off = [], []
    for w in range(1, q_max + 1):
        for nb in BATCHES:
            c = case(d, hq, hkv, page, wl, nb, w)
            if kernel_for(c) is None:
                off.append(f"b{nb}w{w}")
                continue
            i = BATCHES.index(nb)
            kern = kernel_for(c)
            for batch in sorted({nb, BATCHES[i - 1] + 1 if i else 1}):  # bucket edges
                # Gemma4 uses 1.0; retain coverage of standard attention too.
                for sm_scale in (1 / math.sqrt(d), 1.0):
                    args = _problem(d, hq, hkv, page, wl, batch, w, 0, device,
                                    ctx=lambda j, n: 300 + 397 * j, pad=3,
                                    sm_scale=sm_scale)
                    ok, msg = check(kern(*args), reference(*args))
                    if not ok:
                        raise RuntimeError(f"{MARKER} load-time oracle FAILED for {c} batch {batch} "
                                           f"scale {sm_scale}: {msg}")
            on.append(f"b{nb}w{w}")
    _PREPARED[key] = (f"Helion K2 d{d} {hq}/{hkv} wl={wl}: helion {on or '-'}"
                      + (f", oxide {off}" if off else ""))
    return _PREPARED[key]


# ---- reference + problem builder (lab ABI, load-time oracle, CPU tests) --------
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _dequant(data, sf):
    """[n, HKV, PAGE, D/2] u8 x [n, HKV, PAGE, D/16] scales -> [n, HKV, PAGE, D] f32."""
    lut = torch.tensor(_E2M1, device=data.device)
    x = data.long()
    v = torch.stack((lut[x & 15], lut[x >> 4]), -1).flatten(-2)
    return v * sf.float().repeat_interleave(16, -1)


def reference(*args):
    """Exact eager answer of the body args: fp32 math over the dequantized
    cache BYTES (K scales linear, V scales de-swizzled per the store kernel),
    causal with q at the tail, optional window, GQA; padding rows zero."""
    q, bt, seq_lens, qo_indptr = args[0], args[5], args[6], args[7]
    window, qk_scale_log2, v_scale = args[13:16]
    k_data, k_sf, v_data, v_sf = natural_views(args)
    tokens, hq, d = q.shape
    _, hkv, page, sd = k_sf.shape
    g = hq // hkv
    res = torch.zeros(tokens, hq, d, dtype=torch.float32, device=q.device)
    qo, sl = qo_indptr.tolist(), seq_lens.tolist()
    t = torch.arange(page, device=q.device)[:, None]
    swz = ((t // 4) * 4 * sd + 4 * torch.arange(sd, device=q.device)[None, :] + t % 4).flatten()
    for b in range(len(qo) - 1):
        ql, n = qo[b + 1] - qo[b], sl[b]
        if ql <= 0 or n <= 0:
            continue
        pages = bt[b, :math.ceil(n / page)].long()
        vsf = v_sf[pages].flatten(2)[:, :, swz].view(-1, hkv, page, sd)
        k = _dequant(k_data[pages], k_sf[pages])
        v = _dequant(v_data[pages], vsf) * v_scale
        k = k.permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n].repeat_interleave(g, 1)
        v = v.permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n].repeat_interleave(g, 1)
        qpos = torch.arange(n - ql, n, device=q.device)[:, None]
        kpos = torch.arange(n, device=q.device)[None, :]
        ok = (kpos <= qpos) & (kpos + window >= qpos)
        logits = torch.einsum("qhd,khd->hqk", q[qo[b]:qo[b + 1]].float(), k) * (qk_scale_log2 / LOG2E)
        logits = logits.masked_fill(~ok, float("-inf"))
        res[qo[b]:qo[b + 1]] = torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)
    return res


def check(out, ref):
    """The oracle's gates (cos >= 0.9995, max rel <= 2e-2) on the request
    rows; padding rows must be exactly zero."""
    o, r = out.float(), ref.float()
    if not torch.isfinite(o).all():
        return False, "non-finite output"
    pad = (r == 0).flatten(1).all(1)
    if bool((o[pad] != 0).any()):
        return False, "padding rows not zeroed"
    o, r = o[~pad].flatten(), r[~pad].flatten()
    cos = float(o @ r / (o.norm() * r.norm()).clamp_min(1e-30))
    rel = float((o - r).abs().max() / r.abs().max().clamp_min(1e-30))
    return cos >= COS_MIN and rel <= REL_MAX, f"cos {cos:.6f} (>= {COS_MIN}), rel {rel:.2e} (<= {REL_MAX})"


def _ctx(i, n):
    """Lab context spread: 8k for one request, else geometric 2k..32k."""
    return 8192 if n == 1 else int(2048 * 16 ** (i / (n - 1)))


def _problem(d, hq, hkv, page, window_left, batch, width, seed, device, ctx=_ctx, pad=0,
             sm_scale=None):
    """Random paged NVFP4 cache in vLLM's HND byte layout (page = [K data |
    K sf | V data | V sf], head-major), shuffled page table, uniform q width,
    `pad` graph-padding tokens -> body args."""
    gen = torch.Generator().manual_seed(1000 + seed)
    dh, sd = d // 2, d // 16
    lens = [ctx(i, batch) + width for i in range(batch)]
    per = [math.ceil(n / page) for n in lens]
    num_pages = sum(per) + 1
    side = hkv * page * (dh + sd)
    raw = torch.randint(0, 256, (num_pages, 2 * side), dtype=torch.uint8, generator=gen)
    for off in (hkv * page * dh, side + hkv * page * dh):  # valid e4m3 ~0.03..16
        n = hkv * page * sd
        raw[:, off:off + n] = ((torch.randint(3, 11, (num_pages, n), generator=gen) << 3)
                               | torch.randint(0, 8, (num_pages, n), generator=gen)).to(torch.uint8)
    raw = raw.to(device)

    def view(off, w, dtype=torch.uint8):
        return raw[:, off:off + hkv * page * w].view(dtype).view(num_pages, hkv, page, w)

    k_data, v_data = view(0, dh), view(side, dh)
    k_sf = view(hkv * page * dh, sd, torch.float8_e4m3fn)
    v_sf = view(side + hkv * page * dh, sd, torch.float8_e4m3fn)
    ids = torch.randperm(num_pages - 1, generator=gen)
    bt = torch.zeros(batch, max(per), dtype=torch.int32)
    cur = 0
    for b, n in enumerate(per):
        bt[b, :n] = ids[cur:cur + n]
        cur += n
    tokens = batch * width + pad
    q = (torch.randn(tokens, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    out = torch.full((tokens, hq, d), float("nan"), dtype=torch.bfloat16, device=device)
    i32 = dict(dtype=torch.int32, device=device)
    qo = torch.arange(batch + 1, **i32) * width
    cnt = counters(device) if torch.device(device).type == "cuda" else torch.zeros(MAX_COUNTERS, **i32)
    sm_scale = 1 / math.sqrt(d) if sm_scale is None else float(sm_scale)
    return body_args(q, k_data, k_sf, v_data, v_sf, bt.to(device), torch.tensor(lens, **i32), qo,
                     out, cnt, window_left, LOG2E * sm_scale, 1.0)


# ---- lab ABI (inference-console api/autotune_kernels.py) ----------------------
def cases(model) -> list:
    """Gemma 4 per-rank layouts: sliding (head_dim, kv_heads, window 1023) and
    full attention (2 x head_dim, kv_heads / 4) x width x batch buckets."""
    if not str(model.get("architecture") or "").startswith("Gemma4"):
        return []
    hq, hkv, d = model.get("heads"), model.get("kv_heads"), model.get("head_dim")
    if not (hq and hkv and d):
        return []
    layers = ((d, hkv, SLIDING_WINDOW_LEFT), (2 * d, max(1, hkv // 4), -1))
    return [case(hd, hq, kv, PAGE, wl, nb, w)
            for hd, kv, wl in layers for w in WIDTHS for nb in BATCHES]


def working_set_bytes(args):
    """Lab-only upper bound of the distinct bytes touched by one call.

    ``body_args`` exposes four overlapping as_strided views of ONE KV
    storage. Their numel sum counts storage gaps repeatedly (especially V
    scales). Also, SWA reads only the tail window, regardless of the full
    context allocation. Count active physical pages once, plus small inputs,
    output and counters, so rotating-copy timing actually spills L2. Page
    rounding conservatively includes the partially touched boundary pages.
    This synchronizes metadata; never call it in serving/graph capture.
    """
    q, bt, sl, qo, out = args[0], args[5], args[6], args[7], args[8]
    hkv, page, window = args[10], args[11], args[13]
    lens, offsets = sl.tolist(), qo.tolist()
    active = set()
    for b, n in enumerate(lens):
        width = offsets[b + 1] - offsets[b]
        if width <= 0 or n <= 0:
            continue
        lo = max(0, n - width - window)
        active.update(bt[b, lo // page:math.ceil(n / page)].tolist())
    page_bytes = 2 * hkv * page * (q.shape[-1] // 2 + q.shape[-1] // 16)
    # Metadata is tiny; use allocated sizes as a conservative bound. The
    # large shared counter allocation is only indexed at [batch * Hkv].
    metadata = sum(t.numel() * t.element_size() for t in (bt, sl, qo))
    return (len(active) * page_bytes + metadata + len(lens) * hkv * 4
            + q.numel() * q.element_size() + out.numel() * out.element_size())


def make(c, seed):
    # cases() is Gemma4-only; its Q/K normalization uses scale=1.0.
    return _problem(c["head_dim"], c["heads"], c["kv_heads"], c["page"], c["window"],
                    c["numtokens"], c["width"], seed, "cuda", sm_scale=1.0)


def make_baseline(c, seed):
    """The same problem in the oxide op's argument layout."""
    args = make(c, seed)
    k_data, k_sf, v_data, v_sf = natural_views(args)
    return (args[0], k_data, k_sf, v_data, v_sf, *args[5:9], c["width"], c["window"],
            args[14] / LOG2E, args[15])


_OWN = []


def baseline(*args):
    """The oxide K2 op exactly as served (own_attn.prepare + run_oxide)."""
    if not _OWN:
        import importlib.util
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Bundle: runtime_bundle.py ships sm120/nvfp4_kv_patch as <root>/nvfp4_kv_patch;
        # a repo checkout keeps it under sm120/ (lab run 20261017T121746Z: FileNotFoundError).
        path = os.path.join(root, "nvfp4_kv_patch", "own_attn.py")
        if not os.path.isfile(path):
            path = os.path.join(root, "sm120", "nvfp4_kv_patch", "own_attn.py")
        spec = importlib.util.spec_from_file_location("_k2_own_attn_baseline", path)
        own = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(own)
        q, k_data = args[0], args[1]
        own.prepare(own.native(), (q.shape[2], q.shape[1], k_data.shape[1], k_data.shape[2],
                                   args[10]), (args[9],))
        _OWN.append(own)
    return _OWN[0].run_oxide(*args)


KERNELS = [
    {"name": KERNEL, "body": _k2_nvfp4_attn, "cases": cases, "make": make,
     "make_baseline": make_baseline, "reference": reference, "check": check,
     "working_set_bytes": working_set_bytes,
     "baseline": baseline, "baseline_name": "oxide k2_nvfp4_attn", "settings": SETTINGS,
     "replaces": [r"nvfp4_attn_partial", r"nvfp4_attn_merge"]},
]
