# SPDX-License-Identifier: Apache-2.0
"""CPU (numpy) references for the K2-NVFP4 kernel (src/nvfp4_attn_gpu.rs).

* ``reference``: independent float64 math — dequantize the cache BYTES with
  the store kernel's own swizzle formula (nvfp4_kv_cache_kernels.cu
  swizzle_scale_offset), then plain causal(/windowed) GQA softmax attention.
* ``kernel_twin``: statement-level transcription of the two tile kernels
  (tile loop over pages, low-nibble-first unpack, linear K scales, V scales
  through the kernel's 6-D view + permute, exp2 online softmax with the
  -inf guards, bf16 P / bf16 normalized partials, LSE merge, padded rows).
  The CPU tests pin twin == reference on the served shapes, so the tile
  algorithm is proven before silicon; the on-silicon oracle then pins
  kernel == reference.

Cache layout helper ``make_cache`` builds the exact HND byte image vLLM
writes: page = [K_data | K_sf | V_data | V_sf], head-major inside each part.
"""

import math

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3(byte):
    b = np.asarray(byte, dtype=np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6,
                   (1 + m / 8.0) * 2.0 ** (e - 7.0))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return s * val


def bf16(x):
    """Round float -> bf16 (nearest-even), returned as float64."""
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    out = u.astype(np.uint32).view(np.float32).astype(np.float64)
    return np.where(np.isnan(f), np.nan, out)


def swizzle_scale_offset(t, g, s):
    grp = s // 4
    return ((t // 4) * 4 + g // grp) * s + (g % grp) * 4 + t % 4


class Cache:
    """Byte image (P, 2*HKV, PAGE, D/2 + D/16) + the four split views."""

    def __init__(self, raw, hkv, page, d):
        self.raw, self.hkv, self.page, self.d = raw, hkv, page, d
        dh, sd = d // 2, d // 16
        flat = raw.reshape(raw.shape[0], -1)
        side = hkv * page * (dh + sd)

        def view(off, width):
            return flat[:, off:off + hkv * page * width].reshape(
                -1, hkv, page, width)
        self.k_data = view(0, dh)
        self.k_sf = view(hkv * page * dh, sd)
        self.v_data = view(side, dh)
        self.v_sf = view(side + hkv * page * dh, sd)
        if isinstance(raw, np.ndarray):  # views, never copies
            assert all(np.shares_memory(v, raw) for v in (
                self.k_data, self.k_sf, self.v_data, self.v_sf))


def make_cache(rng, num_pages, hkv, page, d):
    """Random NVFP4 data bytes; scale bytes are valid (non-NaN) e4m3."""
    raw = rng.integers(0, 256, size=(num_pages, 2 * hkv, page, d // 2 + d // 16),
                       dtype=np.uint8)
    c = Cache(raw, hkv, page, d)
    for sf in (c.k_sf, c.v_sf):
        # e4m3 exponent 3..10 (scales ~0.03..16), random sign-free mantissa
        sf[...] = (rng.integers(3, 11, size=sf.shape) << 3) | rng.integers(
            0, 8, size=sf.shape)
    return c


def _dequant(data, sf, swizzled):
    """(P,H,N,D/2) bytes + (P,H,N,D/16) scale bytes -> (P,H,N,D) float64."""
    p, h, n, dh = data.shape
    s = sf.shape[-1]
    vals = np.stack((E2M1[data & 0xF], E2M1[data >> 4]), -1).reshape(p, h, n, 2 * dh)
    sc = e4m3(sf)
    if swizzled:
        t = np.arange(n)[:, None]
        g = np.arange(s)[None, :]
        flat = sc.reshape(p, h, n * s)
        sc = flat[:, :, swizzle_scale_offset(t, g, s)]
    return vals * np.repeat(sc, 16, axis=-1)


def q_rows(q_len, tokens):
    """q_len: int (uniform, B = tokens // q_len) or per-request lengths
    (ragged). -> (lengths, offsets[B+1])."""
    if isinstance(q_len, (int, np.integer)):
        lens = [int(q_len)] * (tokens // int(q_len))
    else:
        lens = [int(x) for x in q_len]
    return lens, [0] + list(np.cumsum(lens, dtype=np.int64))


def reference(q, cache, block_table, seq_lens, q_len, sm_scale, k_scale=1.0,
              v_scale=1.0, window_left=-1):
    """q (T, HQ, D) float -> (T, HQ, D) float64; q_len int or per-request
    lengths (request b's rows start at the running sum)."""
    k = _dequant(cache.k_data, cache.k_sf, False) * k_scale
    v = _dequant(cache.v_data, cache.v_sf, True) * v_scale
    tokens, hq, d = q.shape
    hkv, page = cache.hkv, cache.page
    g = hq // hkv
    out = np.zeros((tokens, hq, d))
    lens, offs = q_rows(q_len, tokens)
    for b, ql in enumerate(lens):
        n = int(seq_lens[b])
        if n == 0:
            continue
        pages = block_table[b, :math.ceil(n / page)]
        kb = k[pages].transpose(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        vb = v[pages].transpose(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        for i in range(ql):
            pos = n - ql + i
            kk = np.arange(n)
            ok = kk <= pos
            if window_left >= 0:
                ok &= pos - kk <= window_left
            for hh in range(hq):
                logits = kb[:, hh // g] @ q[offs[b] + i, hh] * sm_scale
                logits = np.where(ok, logits, -np.inf)
                w = np.exp(logits - logits.max())
                out[offs[b] + i, hh] = (w / w.sum()) @ vb[:, hh // g]
    return out


def f16(x):
    """Round float -> f16 (nearest-even), returned as float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def split_tiles(n_t, rows, ns, slots, tn, min_tiles=2, cost_tokens=64):
    """Mirror of nvfp4_attn::split_tiles / the kernel's device choice."""
    c0 = -(-cost_tokens // tn)
    best, best_per = None, min_tiles
    for k in range(1, 5):
        s = ns if k == 4 else min(max(k * slots // rows, 1), ns)
        per = max(-(-n_t // s), min_tiles)
        cost = -(-(rows * -(-n_t // per)) // slots) * (per + c0)
        if best is None or cost < best:
            best, best_per = cost, per
    return best_per


def kernel_twin(q, cache, block_table, seq_lens, q_len, sm_scale, plan,
                k_scale=1.0, v_scale=1.0, window_left=-1):
    """Transcription of nvfp4_attn_partial + nvfp4_attn_merge (v2 layout:
    rows = q tokens x GQA heads packed token-major, TN-token tiles, f16
    Q/K/V/P operands, NS splits of >= min_tiles tiles, bf16 partials)."""
    tokens, hq, d = q.shape
    hkv, page = cache.hkv, cache.page
    g, qt_n, m_rows = hq // hkv, plan["qt"], plan["m"]
    tn, ns, nqt = plan["tn"], plan["ns"], plan["nqt"]
    min_tiles = plan.get("min_tiles", 1)
    sd, sg, tq = d // 16, d // 64, tn // 4
    lens, offs = q_rows(q_len, tokens)
    batch = len(lens)
    qk_scale_log2 = sm_scale * k_scale * math.log2(math.e)
    q4 = f16(bf16(q)).reshape(tokens, hkv, g, d)
    rows = batch * hkv * nqt
    o_part = np.zeros((rows, ns, m_rows, d))
    lse_part = np.full((rows, ns, m_rows), -np.inf)
    tiles_per_page = page // tn
    real = qt_n * g
    for r in range(rows):
        b, h, qt = r // (hkv * nqt), (r // nqt) % hkv, r % nqt
        ql = lens[b]
        if qt * qt_n >= ql:
            continue  # q tile past this request's rows: lse -inf, no store
        kv_len = int(seq_lens[b])
        q0 = kv_len - ql + qt * qt_n
        hi = max(min(q0 + qt_n, kv_len), 0)
        lo = max(q0 - window_left, 0) if window_left >= 0 else 0
        lo_t, hi_t = lo // tn, -(-hi // tn)
        n_t = max(hi_t - lo_t, 0)
        if "split_cost_tokens" in plan:  # kernel split_tiles (wave-aware)
            per = split_tiles(n_t, rows, ns, plan["slots"], tn, min_tiles,
                              plan["split_cost_tokens"])
        else:
            per = max(-(-n_t // ns), min_tiles)
        # rows token-major (i*G + g), padded rows / tokens past q_len = 0
        qm = np.zeros((m_rows, d))
        row_ok = np.zeros(m_rows, dtype=bool)
        for row in range(real):
            i = row // g
            if qt * qt_n + i < ql:
                qm[row] = q4[offs[b] + qt * qt_n + i, h, row % g]
                row_ok[row] = True
        qpos = (q0 + np.arange(m_rows) // g)[:, None]
        for s in range(ns):
            t0 = lo_t + s * per
            t1 = min(t0 + per, hi_t)
            if t0 >= t1:
                continue  # empty split: lse -inf, O never read
            m_i = np.full((m_rows, 1), -np.inf)
            l_i = np.zeros((m_rows, 1))
            acc = np.zeros((m_rows, d))
            for j in range(t0, t1):
                tok0 = j * tn
                pg = int(block_table[b, j // tiles_per_page])
                rows_ = slice((j % tiles_per_page) * tn, (j % tiles_per_page + 1) * tn)
                kb = cache.k_data[pg, h, rows_]
                kq = np.stack((E2M1[kb & 0xF], E2M1[kb >> 4]), -1).reshape(tn, d)
                ks = np.repeat(e4m3(cache.k_sf[pg, h, rows_]), 16, axis=-1)
                kt = kq * ks  # exact in f16
                with np.errstate(invalid="ignore"):
                    sc = (qm @ kt.T) * qk_scale_log2
                kpos = (tok0 + np.arange(tn))[None, :]
                ok = (kpos <= qpos) & (kpos < kv_len) & row_ok[:, None]
                if window_left >= 0:
                    ok &= kpos + window_left >= qpos
                sc = np.where(ok, sc, -np.inf)
                m_new = np.maximum(m_i, sc.max(1, keepdims=True))
                m_safe = np.where(m_new == -np.inf, 0.0, m_new)
                p = np.exp2(sc - m_safe)
                alpha = np.exp2(m_i - m_safe)
                l_i = l_i * alpha + p.sum(1, keepdims=True)
                m_i = m_new
                vb = cache.v_data[pg, h, rows_]
                vq = np.stack((E2M1[vb & 0xF], E2M1[vb >> 4]), -1).reshape(tn, d)
                vsf = e4m3(cache.v_sf[pg, h].reshape(page // 4, 4, sg, 4)[
                    (j % tiles_per_page) * tq:(j % tiles_per_page + 1) * tq])
                vs = np.repeat(vsf.transpose(0, 3, 1, 2).reshape(tn, sd), 16, -1)
                valid = (tok0 + np.arange(tn) < kv_len)[:, None]
                with np.errstate(invalid="ignore"):
                    vt = np.where(valid, vq * vs, 0.0)
                acc = f16(p) @ vt + acc * alpha
            l_safe = np.where(l_i == 0, 1.0, l_i)
            o_part[r, s] = bf16(np.where(l_i == 0, 0.0, acc / l_safe))
            with np.errstate(divide="ignore"):
                lse_part[r, s] = (m_i + np.log2(l_i))[:, 0]
    out = np.zeros((tokens, hq, d))
    out4 = out.reshape(tokens, hkv, g, d)
    for r in range(rows):
        b, h, qt = r // (hkv * nqt), (r // nqt) % hkv, r % nqt
        if qt * qt_n >= lens[b]:
            continue  # merge CTA returns before any store
        lse = lse_part[r]
        mx = lse.max(0, keepdims=True)
        mx = np.where(mx == -np.inf, 0.0, mx)
        w = np.exp2(lse - mx)
        wsum = w.sum(0)[:, None]
        inv = np.where(wsum == 0, 0.0, v_scale / np.where(wsum == 0, 1.0, wsum))
        o = bf16((o_part[r] * w[:, :, None]).sum(0) * inv)
        for row in range(real):
            i = row // g
            if qt * qt_n + i < lens[b]:
                out4[offs[b] + qt * qt_n + i, h, row % g] = o[row]
    return out
