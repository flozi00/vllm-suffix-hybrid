# SPDX-License-Identifier: Apache-2.0
"""CPU proof of the DSA indexer logits kernel (kernels-oxide/dsa_indexer) and
its launch plan (suffix_hybrid/kernels/dsa_indexer.py).

`twin()` transcribes the kernel: grid (splits, groups) from plan(), per-CTA
tile range, the per-tile page-dedup item walk (next_item), cp.async staging
into a garbage-filled smem ring (pitch 144 + scale region), Q A-fragments
loaded per lane, ldmatrix.x4 B fragments, the documented m16n8k32 e4m3
fragment layouts, the relu*w epilogue + butterfly and the masked stores —
against an f64 reference, on a NaN-poisoned output (nothing outside
[lo, hi) may be written).
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T, PITCH, D = 64, 144, 128
STAGE = T * PITCH + T * 4
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double().numpy()


# ---------------------------------------------------------------------------
# kernel twin
# ---------------------------------------------------------------------------
def f32(x):
    return np.float32(x)


def mma(c, a, b):
    """m16n8k32 e4m3 -> f32 for one warp. a [32][4] regs, b [32][2] regs
    (each reg = 4 byte values), c [32][4] f32. Fragment layouts (PTX ISA):
    a0 (g, 4t..), a1 (g+8, 4t..), a2 (g, 16+4t..), a3 (g+8, 16+4t..);
    b0 (k 4t.., n g), b1 (k 16+4t.., n g); c0 (g, 2t) c1 (g, 2t+1) c2 (g+8,
    2t) c3 (g+8, 2t+1)."""
    A = np.zeros((16, 32))
    B = np.zeros((32, 8))
    for ln in range(32):
        g, t = ln // 4, ln % 4
        A[g, 4 * t:4 * t + 4] = E4M3[a[ln][0]]
        A[g + 8, 4 * t:4 * t + 4] = E4M3[a[ln][1]]
        A[g, 16 + 4 * t:20 + 4 * t] = E4M3[a[ln][2]]
        A[g + 8, 16 + 4 * t:20 + 4 * t] = E4M3[a[ln][3]]
        B[4 * t:4 * t + 4, g] = E4M3[b[ln][0]]
        B[16 + 4 * t:20 + 4 * t, g] = E4M3[b[ln][1]]
    Dm = A @ B
    out = []
    for ln in range(32):
        g, t = ln // 4, ln % 4
        out.append([f32(c[ln][0] + Dm[g, 2 * t]), f32(c[ln][1] + Dm[g, 2 * t + 1]),
                    f32(c[ln][2] + Dm[g + 8, 2 * t]), f32(c[ln][3] + Dm[g + 8, 2 * t + 1])])
    return out


def ldmatrix_x4(smem, addr):
    """addr[l] = row address supplied by lane l (matrix l // 8, row l % 8);
    lane (g, t) gets 4 bytes at column 4t of row g of each matrix."""
    return [[smem[addr[8 * j + ln // 4] + 4 * (ln % 4): addr[8 * j + ln // 4] + 4 * (ln % 4) + 4]
             for j in range(4)] for ln in range(32)]


def twin(out, q, w, kvb, scb, bt, lo, hi, n_cols, group, slices, bt_div, splits, min_tiles, paged,
         ps_v, ps_s, pitch, seed=0):
    """Byte-level transcription of `dsa_logits`. q uint8 [rows, H*128]; kvb /
    scb flat uint8 buffers (value / scale base as the host passes them)."""
    rows, heads = q.shape[0], q.shape[1] // D
    rng = np.random.default_rng(seed)

    def window(r):
        if r >= rows:
            return 0, 0
        h = min(max(int(hi[r]), 0), n_cols)
        lw = max(int(lo[r]), 0) if lo is not None else 0
        return lw, h

    for gy in range(math.ceil(rows / group)):
        row0 = gy * group

        def active(k):
            m = 0
            for j in range(group):
                lw, h = window(row0 + j)
                if lw < h and lw < (k + 1) * T and h > k * T:
                    m |= 1 << j
            return m

        def page(j, k):
            return int(bt[(row0 + j) // bt_div, k]) if paged else k

        def next_item(k, served, k1):
            rem = active(k) & ~served if k < k1 else 0
            while rem == 0:
                k += 1
                if k >= k1:
                    return k, None, 0, 0
                served, rem = 0, active(k)
            lead = (rem & -rem).bit_length() - 1
            mem = 0
            for j in range(lead, group):
                if rem >> j & 1 and page(j, k) == page(lead, k):
                    mem |= 1 << j
            return k, lead, mem, served | mem

        wins = [window(row0 + j) for j in range(group)]
        live = [(a, b) for a, b in wins if a < b]
        if not live:
            continue
        glo, ghi = min(a for a, _ in live), max(b for _, b in live)
        t0, t1 = glo // T, -(-ghi // T)
        per = max(-(-(t1 - t0) // splits), min_tiles)
        for s in range(splits):
            k0 = t0 + s * per
            if k0 >= t1:
                continue
            k1 = min(k0 + per, t1)
            smem = rng.integers(0, 256, 2 * STAGE, dtype=np.int64)  # stale bytes (NaN codes too)

            def issue(st, k, lead):
                pg = page(lead, k)
                for i in range(T):
                    if paged or k * T + i < n_cols:
                        v = pg * ps_v + i * pitch
                        smem[st * STAGE + i * PITCH: st * STAGE + i * PITCH + D] = kvb[v:v + D]
                        smem[st * STAGE + T * PITCH + 4 * i: st * STAGE + T * PITCH + 4 * i + 4] = \
                            scb[pg * ps_s + 4 * i: pg * ps_s + 4 * i + 4]

            k, lead, mem, served = next_item(k0, 0, k1)
            if lead is None:
                continue
            issue(0, k, lead)
            st = 0
            while True:
                nk, nlead, nmem, nserved = next_item(k, served, k1)
                if nlead is not None:
                    issue(st ^ 1, nk, nlead)
                for warp in range(group * slices):
                    jr, sl = warp // slices, warp % slices
                    row = row0 + jr
                    if not mem >> jr & 1:
                        continue
                    rlo, rhi = window(row)
                    qrow = q[row]
                    wrow = w[row]
                    # per lane: m-tiles x k-steps of A regs + weights
                    qa, wv = [], []
                    for mt in range(heads // 16):
                        lanes, wl = [], []
                        for ln in range(32):
                            g, t = ln // 4, ln % 4
                            h0 = 16 * mt + g
                            at = lambda h, o: qrow[h * D + o: h * D + o + 4]  # noqa: E731
                            lanes.append([[at(h0, 32 * ks + 4 * t), at(h0 + 8, 32 * ks + 4 * t),
                                           at(h0, 32 * ks + 16 + 4 * t), at(h0 + 8, 32 * ks + 16 + 4 * t)]
                                          for ks in range(4)])
                            wl.append((f32(wrow[h0]), f32(wrow[h0 + 8])))
                        qa.append(lanes)
                        wv.append(wl)
                    for nt in range(sl, T // 8, slices):
                        n0 = k * T + nt * 8
                        if not (n0 < rhi and n0 + 8 > rlo):
                            continue
                        base = st * STAGE
                        addr = [base + (nt * 8 + ln % 8) * PITCH + 16 * (ln // 8) for ln in range(32)]
                        b0 = ldmatrix_x4(smem, addr)
                        b1 = ldmatrix_x4(smem, [x + 64 for x in addr])
                        s0 = [f32(0)] * 32
                        s1 = [f32(0)] * 32
                        for mt in range(heads // 16):
                            c = [[f32(0)] * 4 for _ in range(32)]
                            c = mma(c, [qa[mt][ln][0] for ln in range(32)], [(b0[ln][0], b0[ln][1]) for ln in range(32)])
                            c = mma(c, [qa[mt][ln][1] for ln in range(32)], [(b0[ln][2], b0[ln][3]) for ln in range(32)])
                            c = mma(c, [qa[mt][ln][2] for ln in range(32)], [(b1[ln][0], b1[ln][1]) for ln in range(32)])
                            c = mma(c, [qa[mt][ln][3] for ln in range(32)], [(b1[ln][2], b1[ln][3]) for ln in range(32)])
                            for ln in range(32):
                                w0, w1 = wv[mt][ln]
                                r = [max(x, f32(0)) for x in c[ln]]
                                s0[ln] = f32(f32(s0[ln] + f32(r[0] * w0)) + f32(r[2] * w1))
                                s1[ln] = f32(f32(s1[ln] + f32(r[1] * w0)) + f32(r[3] * w1))
                        for m in (4, 8, 16):
                            s0 = [f32(s0[ln] + s0[ln ^ m]) for ln in range(32)]
                            s1 = [f32(s1[ln] + s1[ln ^ m]) for ln in range(32)]
                        sc = smem[base + T * PITCH: base + T * PITCH + 4 * T].astype(np.uint8).view(np.float32)
                        for t in range(4):  # lanes g == 0
                            n, i = n0 + 2 * t, nt * 8 + 2 * t
                            if rlo <= n < rhi:
                                out[row, n] = f32(s0[t] * sc[i])
                            if rlo <= n + 1 < rhi:
                                out[row, n + 1] = f32(s1[t] * sc[i + 1])
                if nlead is None:
                    break
                k, mem, served = nk, nmem, nserved
                st ^= 1


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------
def decode_problem(heads, lens, next_n, flattened, shared_prefix=0, seed=0):
    """Requests with context lens `lens` over shuffled physical pages (the first
    `shared_prefix` pages of every request are the SAME physical pages)."""
    g = torch.Generator().manual_seed(seed)
    nblk = [-(-n // T) for n in lens]
    num_blocks = sum(nblk) + 2
    perm = torch.randperm(num_blocks, generator=g)
    bt = torch.full((len(lens), max(nblk) + 1), -1, dtype=torch.int32)  # -1 past the context
    o = 0
    for i, nb in enumerate(nblk):
        for b in range(nb):
            if b < shared_prefix:
                bt[i, b] = perm[b]
            else:
                bt[i, b] = perm[o + b]
        o += nb
    kf = (torch.randn((num_blocks * T, D), generator=g) * 0.7).to(torch.float8_e4m3fn)
    ksc = 2.0 ** torch.randint(-6, 0, (num_blocks * T,), generator=g).float()
    cache = di.build_paged_cache(kf, ksc, torch.arange(num_blocks, dtype=torch.int32), num_blocks)
    rows = len(lens) * next_n
    q = (torch.randn((rows, heads, D), generator=g) * 1.5).to(torch.float8_e4m3fn)
    w = torch.randn((rows, heads), generator=g) * 0.1
    cl = (torch.tensor(lens).unsqueeze(1) - next_n + 1 + torch.arange(next_n)).clamp_min(0).to(torch.int32)
    return dict(cache=cache, bt=bt, q=q, w=w, cl=cl, rows=rows, next_n=next_n, flattened=flattened)


def reference(pr, max_len):
    """f64 logits + |term| magnitudes over each row's window via the split-page
    gather (row r -> request r // next_n)."""
    rows = pr["rows"]
    ref = torch.full((rows, max_len), float("nan"), dtype=torch.float64)
    mag = torch.zeros((rows, max_len), dtype=torch.float64)
    cl = pr["cl"].reshape(-1)
    for r in range(rows):
        n = min(int(cl[r]), max_len)
        if n == 0:
            continue
        k, s = di.gather_paged(pr["cache"], pr["bt"][r // pr["next_n"]], n)
        ref[r, :n], mag[r, :n] = di.ref_logits(pr["q"][r], pr["w"][r], k, s)
    return ref, mag


def run_twin_decode(pr, max_len, group=None):
    rows, nn = pr["rows"], pr["next_n"]
    if pr["flattened"]:
        bt, bt_div, grp = pr["bt"].repeat_interleave(nn, 0), 1, group or nn
    else:
        bt, bt_div, grp = pr["bt"], nn, group or nn
    p = di.plan(rows, grp, max_len, n_sms=3, min_tiles=1)
    out = np.full((rows, max_len), np.nan, dtype=np.float32)
    flat = pr["cache"].reshape(-1).numpy().astype(np.int64)
    ps = T * 132
    twin(out, pr["q"].view(torch.uint8).reshape(rows, -1).numpy().astype(np.int64),
         pr["w"].numpy(), flat, flat[T * D:], bt.numpy(), None, pr["cl"].reshape(-1).numpy(),
         max_len, p["group"], p["slices"], bt_div, p["splits"], p["min_tiles"], True, ps, ps, D)
    return out


def check(out, ref, mag):
    inside = ~torch.isnan(ref)
    o = torch.from_numpy(out).double()
    assert torch.isnan(o[~inside]).all(), "wrote outside [lo, hi)"
    err = ((o[inside] - ref[inside]).abs() / (mag[inside] + 1e-30)).max()
    assert err <= di.TOL, float(err)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("heads,lens,next_n,flat,group,shared", [
    (32, [150], 1, False, None, 0),            # plain decode, 1 row, 4 warps/row
    (32, [200, 70], 6, True, None, 0),         # vLLM SM120 flattened MTP (k=5)
    (64, [130, 190], 2, False, None, 0),       # native [B, 2], bt row = r // 2
    (32, [100, 140, 90], 5, True, 6, 0),       # group 6 straddles 5-row requests
    (32, [160, 160], 3, True, 6, 2),           # shared prefix pages dedup across requests
    (32, [7, 2], 6, True, None, 0),            # rows with cl 0 (clamped) + sub-tile windows
])
def test_twin_decode_matches_f64(heads, lens, next_n, flat, group, shared):
    pr = decode_problem(heads, lens, next_n, flat, shared_prefix=shared, seed=len(lens) + next_n)
    max_len = 300
    out = run_twin_decode(pr, max_len, group)
    check(out, *reference(pr, max_len))


def test_twin_decode_clamps_to_max_model_len():
    pr = decode_problem(32, [250], 2, False, seed=3)
    out = run_twin_decode(pr, 190)  # contexts longer than the logits width
    ref, mag = reference(pr, 190)
    check(out, ref, mag)


def test_twin_prefill_windows_match_f64():
    g = torch.Generator().manual_seed(5)
    m, n, heads = 11, 150, 64
    q = (torch.randn((m, heads, D), generator=g)).to(torch.float8_e4m3fn)
    kv = (torch.randn((n, D), generator=g) * 0.5).to(torch.float8_e4m3fn)
    ksc = 2.0 ** torch.randint(-6, 0, (n,), generator=g).float()
    w = torch.randn((m, heads), generator=g) * 0.1
    ks = torch.tensor([0] * 5 + [60] * 6, dtype=torch.int32)             # two packed requests
    ke = torch.tensor([56 + i for i in range(5)] + [145 + i for i in range(6)], dtype=torch.int32)
    ke[-1] = n
    p = di.plan(m, di.PREFILL_GROUP, n, n_sms=2, min_tiles=1)
    out = np.full((m, n), np.nan, dtype=np.float32)
    twin(out, q.view(torch.uint8).reshape(m, -1).numpy().astype(np.int64), w.numpy(),
         kv.view(torch.uint8).reshape(-1).numpy().astype(np.int64),
         ksc.view(torch.uint8).numpy().astype(np.int64), None, ks.numpy(), ke.numpy(), n,
         p["group"], p["slices"], 1, p["splits"], p["min_tiles"], False, T * D, T * 4, D)
    ref = torch.full((m, n), float("nan"), dtype=torch.float64)
    mag = torch.zeros((m, n), dtype=torch.float64)
    for r in range(m):
        a, b = int(ks[r]), int(ke[r])
        ref[r, a:b], mag[r, a:b] = di.ref_logits(q[r], w[r], kv[a:b], ksc[a:b])
    check(out, ref, mag)


def test_paged_split_layout_roundtrip():
    """build_paged_cache writes vLLM's split page (64*128 values, then 64
    f32 scales) — the byte offsets of indexer_k_quant_and_cache_kernel — and
    gather_paged reads tokens back through a scattered block table."""
    g = torch.Generator().manual_seed(1)
    k = (torch.randn((3 * T, D), generator=g)).to(torch.float8_e4m3fn)
    s = torch.rand(3 * T, generator=g) + 0.5
    pages = torch.tensor([2, 0, 1], dtype=torch.int32)
    cache = di.build_paged_cache(k, s, pages, 3)
    flat = cache.reshape(3, -1)
    t = 70  # logical block 1 -> physical page 0, slot 6
    assert torch.equal(flat[0, 6 * D:7 * D], k[t].view(torch.uint8))
    assert flat[0, T * D + 24:T * D + 28].view(torch.float32).item() == s[t].item()
    kk, ss = di.gather_paged(cache, pages, 3 * T - 5)
    assert torch.equal(kk.view(torch.uint8), k[:3 * T - 5].view(torch.uint8))
    assert torch.equal(ss, s[:3 * T - 5])


def test_head_weight_reduction_reference():
    """ref_logits == the contract formula written out per head."""
    g = torch.Generator().manual_seed(2)
    q = torch.randn((32, D), generator=g).to(torch.float8_e4m3fn)
    k = torch.randn((9, D), generator=g).to(torch.float8_e4m3fn)
    w = torch.randn(32, generator=g)
    s = torch.rand(9, generator=g)
    val, mag = di.ref_logits(q, w, k, s)
    for n in range(9):
        want = sum(max(float(q[h].double() @ k[n].double()), 0.0) * float(w[h]) for h in range(32)) * float(s[n])
        assert abs(float(val[n]) - want) <= 1e-9 * float(mag[n])


def test_plan_and_groups(monkeypatch):
    assert [di.slices_for(g) for g in (1, 2, 3, 4, 6, 12)] == [4, 2, 2, 1, 1, 1]
    p = di.plan(rows=6, group=6, n_cols=202752, n_sms=188)
    assert p["grid"] == (376, 1) and p["threads"] == 192 and p["min_tiles"] == 2
    p = di.plan(rows=192, group=6, n_cols=202752, n_sms=188)
    assert p["grid"] == (12, 32)
    assert di.plan(rows=1, group=1, n_cols=100, n_sms=188)["grid"] == (2, 1)  # <= tiles
    assert di.plan(rows=5, group=40, n_cols=64, n_sms=1)["group"] == di.MAX_WARPS
    for g in range(1, 13):
        assert 32 * g * di.slices_for(g) <= 384  # launch_bounds(384)
    monkeypatch.delenv(di.GROUP_ENV, raising=False)
    monkeypatch.setitem(di._state, "cfg_group", None)
    assert di.decode_group(2) == 2 and di.decode_group(6) == 6
    assert di.decode_group(1) == 1  # no vLLM config here
    monkeypatch.setitem(di._state, "cfg_group", 6)
    assert di.decode_group(1) == 6
    monkeypatch.setenv(di.GROUP_ENV, "4")
    assert di.decode_group(1) == 4


def test_mtp_rows_from_config():
    class S:
        num_speculative_tokens = 5
        parallel_drafting = False

    class C:
        speculative_config = S()

    assert di.mtp_rows(C()) == 6
    S.parallel_drafting = True
    assert di.mtp_rows(C()) == 11
    assert di.mtp_rows(None) is None


def test_eligibility_reasons():
    f8 = torch.float8_e4m3fn
    q = torch.zeros((2, 6, 32, D), dtype=f8)
    kv = torch.zeros((4, T, 1, 132), dtype=torch.uint8)
    w = torch.zeros((12, 32))
    bt = torch.zeros((2, 4), dtype=torch.int32)
    assert di.paged_ineligible(q, kv, w, bt, None, None) is None
    assert "MXFP4" in di.paged_ineligible(q, kv, w, bt, torch.zeros(1), None)
    assert "indices" in di.paged_ineligible(q, kv, w, bt, None, torch.zeros(1))
    assert "H 32|64" in di.paged_ineligible(torch.zeros((2, 6, 16, D), dtype=f8), kv, w, bt, None, None)
    assert "64, 132" in di.paged_ineligible(q, torch.zeros((4, 32, 1, 132), dtype=torch.uint8), w, bt, None, None)
    kvc = torch.zeros((9, D), dtype=f8)
    assert di.mqa_ineligible(q[0], kvc, torch.zeros(9), w[:6], None) is None
    assert "k_scale" in di.mqa_ineligible(q[0], kvc, torch.zeros(8), w[:6], None)


def test_shim_wrappers_route(monkeypatch):
    calls = []
    monkeypatch.setattr(di, "paged_logits", lambda *a, **k: calls.append("ours") or "ours")
    monkeypatch.setattr(di, "mqa_logits", lambda *a, **k: calls.append("ours-mqa") or "ours-mqa")
    paged = di.wrap_paged(lambda *a: "triton")
    mqa = di.wrap_mqa(lambda *a: "triton-mqa")
    f8 = torch.float8_e4m3fn
    q = torch.zeros((2, 1, 32, D), dtype=f8)
    kv = torch.zeros((4, T, 1, 132), dtype=torch.uint8)
    w = torch.zeros((2, 32))
    bt = torch.zeros((2, 4), dtype=torch.int32)
    cl = torch.ones((2, 1), dtype=torch.int32)
    assert paged((q, None), kv, w, cl, bt, None, 100, False, None) == "ours"
    assert paged((q, None), kv, w, cl, bt, None, 100, False, torch.zeros(1)) == "triton"
    kvc = torch.zeros((9, D), dtype=f8)
    assert mqa((q[:, 0], None), (kvc, torch.zeros(9)), w, cl[:, 0], cl[:, 0], False) == "ours-mqa"
    assert mqa((q[:, 0].to(torch.uint8), torch.zeros(1)), (kvc, torch.zeros(9)), w, cl[:, 0], cl[:, 0], False) == "triton-mqa"


def test_topk_set_helpers():
    vals = torch.tensor([5.0, 1.0, 3.0, 3.0, 0.5])
    assert di.vllm_set_ok([0, 2], vals, 2) and di.vllm_set_ok([0, 3], vals, 2)  # tie at the k-th
    assert not di.vllm_set_ok([0, 1], vals, 2)
    assert not di.vllm_set_ok([0, 7], vals, 2)       # index past the window (poison read)
    assert not di.vllm_set_ok([0, 0], vals, 2)
    assert di.vllm_set_ok([0, 1, 2, 3, 4, -1, -1], vals, 7)  # short row: 0..n-1, -1 fill
    a = torch.tensor([4.0, 3.0, 2.0, 2.0 + 1e-6, 0.0])
    b = torch.tensor([4.0, 3.0, 2.0 + 1e-6, 2.0, 0.0])
    ok, nd = di.topk_agree(a, b, a.double(), 3, 1e-5)
    assert ok and nd == 2
    ok, _ = di.topk_agree(a, torch.tensor([0.0, 3.0, 2.0, 2.0, 4.0]), a.double(), 3, 1e-5)
    assert not ok


def _kernel_params(src, name):
    src = re.sub(r"//[^\n]*", "", src)
    body = re.search(rf"pub unsafe fn {name}\((.*?)\)\s*\{{", src, re.S).group(1)
    kinds = []
    for prm in filter(None, (x.strip() for x in body.split(","))):
        ty = prm.split(":", 1)[1].strip()
        kinds.append("Ptr" if ty.startswith("*") else {"u32": "U32", "i32": "I32", "f32": "F32"}[ty])
    return kinds


def test_launch_args_match_kernel_params():
    dev = (ROOT / "kernels-oxide/dsa_indexer/src/main.rs").read_text()
    host = re.sub(r"//[^\n]*", "", (ROOT / "src/dsa_indexer_oxide.rs").read_text())
    body = re.search(r"let args = vec!\[(.*?)\];", host, re.S).group(1)
    got = [tok or "U32" for tok in re.findall(r"Arg::(\w+)\(|\bu\(", body)]
    want = _kernel_params(dev, "dsa_logits")
    assert got == want
    assert di.PARAMS == {"dsa_logits": len(want)}
    # kernel constants == host op + plan constants
    for name, val in (("T", T), ("PITCH", PITCH), ("MAX_G", di.MAX_WARPS)):
        assert re.search(rf"const {name}: u32 = {val};", dev), name
    assert "launch_bounds(384)" in dev and 32 * di.MAX_WARPS == 384


def test_shim_wires_the_gate():
    src = (ROOT / "sm120/deep_gemm_shim/__init__.py").read_text()
    assert '"SUFFIX_SM120_DSA_INDEXER"' in src and "wrap_paged(fp8_fp4_paged_mqa_logits)" in src
    assert di.GATE == "SUFFIX_SM120_DSA_INDEXER"
    site = (ROOT / "sitecustomize.py").read_text()
    assert '"dsa_indexer_oracle": (["-m", "suffix_hybrid.kernels.dsa_indexer", "oracle"], {})' in site
    assert '"dsa_indexer_bench": (["-m", "suffix_hybrid.kernels.dsa_indexer", "bench"], {})' in site
