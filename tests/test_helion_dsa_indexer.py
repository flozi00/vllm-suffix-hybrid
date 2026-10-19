# SPDX-License-Identifier: Apache-2.0
"""Helion DSA indexer port (suffix_hybrid/helion_kernels/dsa_indexer.py).

Without Helion: the lab ABI (cases / KERNELS), the serving routing, and both
bodies executed by a minimal eager stand-in for helion.language (tiles =
index tensors, the ref-mode semantics) against the lane's f64 spec on a
-inf-filled output (nothing outside [lo, hi) may be written). With Helion:
the same problems through helion.kernel in HELION_INTERPRET=1 eager mode.
"""
import itertools
import os
import re
import types

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")  # before anything imports helion
torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_capture as cap  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

T, D = 64, 128
NEG = float("-inf")


# ---------------------------------------------------------------------------
# problems (CPU, small)
# ---------------------------------------------------------------------------
def build_cache(kf, ksc, page, pad):
    """uint8 [blocks, page, 132] split pages (token t of page p at row
    p * page + t of kf) living in a backing whose page stride is
    page * 132 + pad bytes (DeepSeek-V4.1 packs the indexer page beside the
    MLA pages of a block); pad bytes are garbage."""
    nb = kf.shape[0] // page
    stride = page * 132 + pad
    back = torch.randint(0, 256, (nb * stride,), dtype=torch.uint8)
    pages = back.view(nb, stride)
    pages[:, :page * D] = kf.view(torch.uint8).reshape(nb, page * D)
    pages[:, page * D:page * 132] = ksc.view(torch.uint8).reshape(nb, page * 4)
    return back.as_strided((nb, page, 132), (stride, 132, 1))


def cache_of(kv, page, prow):
    """The uint8 [blocks, page, 132] cache behind cache_views' kv view."""
    s = prow * 128
    return kv.view(torch.uint8).as_strided(((kv.numel() - page * 132) // s + 1, page, 132), (s, 132, 1))


def decode_problem(heads, lens, next_n, n_cols, seed=0, page=T, pad=0):
    """vLLM SM120 flattened decode: rows of a request carry copies of its
    block-table row, row j of a request sees L - next_n + 1 + j tokens."""
    g = torch.Generator().manual_seed(seed)
    nblk = [-(-n // page) for n in lens]
    num_blocks = sum(nblk) + 2
    perm = torch.randperm(num_blocks, generator=g)
    bt = torch.full((len(lens), max(nblk) + 1), -1, dtype=torch.int32)
    o = 0
    for i, nb in enumerate(nblk):
        bt[i, :nb] = perm[o:o + nb].to(torch.int32)
        o += nb
    kf = (torch.randn((num_blocks * page, D), generator=g) * 0.7).to(torch.float8_e4m3fn)
    ksc = 2.0 ** torch.randint(-6, 0, (num_blocks * page,), generator=g).float()
    cache = build_cache(kf, ksc, page, pad)
    rows = len(lens) * next_n
    q = (torch.randn((rows, heads, D), generator=g) * 1.5).to(torch.float8_e4m3fn)
    w = torch.randn((rows, heads), generator=g) * 0.1
    cl = (torch.tensor(lens).unsqueeze(1) - next_n + 1 + torch.arange(next_n)).clamp_min(0).to(torch.int32)
    pr = {"rows": rows, "heads": heads, "cache": cache.unsqueeze(2), "q": q.view(rows, 1, heads, D),
          "w": w, "cl": cl.reshape(rows, 1), "bt": bt.repeat_interleave(next_n, 0), "next_n": next_n}
    return hdi.decode_args(pr, n_cols, n_sms=2)


def prefill_problem(heads=64, m=11, n=150, splits=3, compressed=True):
    return (*hdi.prefill_problem(heads, m, n, "cpu", seed=5, compressed=compressed), splits)


def cand_problem(heads, lens, next_n, ncand=16, bsz=8, page=T, pad=0, seed=0):
    """decode_problem + vLLM-ordered candidate blocks per row (cand_problem)."""
    q, w, kv, ks, bt, cl, _, pg, prow, sp, _ = decode_problem(heads, lens, next_n, 1, seed, page, pad)
    cand = hdi.cand_problem({"cl": cl}, ncand, bsz, seed)
    out = torch.full((cl.shape[0], ncand * bsz), NEG)
    return q, w, kv, ks, bt, cl, cand, out, pg, prow, sp, 3


# ---------------------------------------------------------------------------
# eager stand-in for helion.language (ref-mode semantics, tunable block sizes)
# ---------------------------------------------------------------------------
class _Tile(torch.Tensor):
    pass


def _tile(b, e):
    t = torch.arange(b, e).as_subclass(_Tile)
    t.begin, t.index = b, torch.arange(b, e)
    return t


def _load(t, index, extra_mask=None, eviction_policy=None):
    """hl.load: masked lanes read nothing (0) and never index `t` (an
    out-of-range index on a live lane raises, like a fault)."""
    ix = [i.as_subclass(torch.Tensor) if isinstance(i, torch.Tensor) else i for i in index]
    if extra_mask is None:
        return t[tuple(ix)]
    m = extra_mask.as_subclass(torch.Tensor)
    safe = tuple(torch.where(m, i, torch.zeros_like(i)) if isinstance(i, torch.Tensor) else i for i in ix)
    got = t[safe]
    return torch.where(m, got, torch.zeros_like(got))


def _atomic_add(t, index, v):
    old = t[index[0]].clone()
    t[index[0]] = old + v
    return old


def make_stub(block, tunables):
    def tile(begin, end=None, block_size=None):
        if end is None:
            begin, end = [0] * len(begin) if isinstance(begin, list) else 0, begin
        multi = isinstance(end, list)
        bs = block_size if isinstance(block_size, list) else [block_size]
        dims = []
        for b, e, s in zip(*(x if multi else [x] for x in (begin, end)), bs):
            b, e = int(b), int(e)
            s = int(s) if s is not None else max(1, e - b)
            dims.append([_tile(i, min(i + s, e)) for i in range(b, e, s)])
        for combo in itertools.product(*dims):
            yield combo if multi else combo[0]

    def grid(ends):
        return itertools.product(*(range(int(e)) for e in ends)) if isinstance(ends, list) \
            else range(int(ends))

    def store(t, index, val, extra_mask=None):
        ix = [i.index if "begin" in getattr(i, "__dict__", {}) else i for i in index]
        ix = [i.as_subclass(torch.Tensor) if isinstance(i, torch.Tensor) else i for i in ix]
        if len(ix) == 2 and all(isinstance(i, torch.Tensor) for i in ix):
            ix = [ix[0][:, None], ix[1][None, :]]
        ix = tuple(ix)
        val = val.as_subclass(torch.Tensor).to(t.dtype)
        if extra_mask is None:
            t[ix] = val
            return
        # masked lanes store nothing (they may alias live ones, like Triton's)
        *ixb, m, v = torch.broadcast_tensors(*[torch.as_tensor(i) for i in ix],
                                             extra_mask.as_subclass(torch.Tensor),
                                             val.broadcast_to(t[ix].shape))
        t[tuple(i[m] for i in ixb)] = v[m]

    return types.SimpleNamespace(
        tile=tile, grid=grid, store=store, specialize=int,
        register_block_size=lambda lo, hi: max(lo, min(hi, block)),
        register_tunable=lambda name, frag: tunables[name],
        dot=lambda a, b, out_dtype=None: a.float() @ b.float(),
        zeros=lambda shape, dtype: torch.zeros([len(s) for s in shape], dtype=dtype),
        static_range=lambda a, b=None: range(a) if b is None else range(a, b),
        atomic_add=_atomic_add, load=_load)


def run_stub(monkeypatch, body, args, block=32, min_tiles=1, split_div=1):
    monkeypatch.setattr(hdi, "hl", make_stub(block, {"min_tiles": min_tiles, "split_div": split_div}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    res = body(*args)
    return tuple(r.as_subclass(torch.Tensor) for r in res) if isinstance(res, tuple) \
        else res.as_subclass(torch.Tensor)


# ---------------------------------------------------------------------------
# bodies vs the f64 spec (stand-in)
# ---------------------------------------------------------------------------
DECODE_PROBLEMS = [  # (heads, lens, next_n, n_cols)
    (32, [150], 1, 300),
    (32, [200, 70], 6, 300),     # flattened MTP k=5
    (64, [130, 190], 2, 300),
    (32, [7, 2], 6, 300),        # rows with cl 0 + sub-tile windows
    (32, [250], 2, 190),         # contexts longer than the logits width (clamped)
]


@pytest.mark.parametrize("heads,lens,next_n,n_cols", DECODE_PROBLEMS)
@pytest.mark.parametrize("block,min_tiles,split_div", [(32, 1, 1), (16, 2, 2), (512, 4, 1)])
def test_decode_body_matches_f64(monkeypatch, heads, lens, next_n, n_cols, block, min_tiles, split_div):
    args = decode_problem(heads, lens, next_n, n_cols, seed=len(lens) + next_n)
    ref = hdi.decode_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_decode_logits, args, block, min_tiles, split_div)
    ok, msg = hdi.check(out, ref)
    assert ok, msg


def group_problem(heads, lens, next_n, n_cols, g=6, seed=0, page=T, pad=0):
    """decode_problem in the row-group body's args; rows need not be a
    multiple of g (the last group is padded) and groups straddle requests
    whenever next_n != g."""
    args = decode_problem(heads, lens, next_n, n_cols, seed, page, pad)
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = args
    q4, w3, cl = hdi.group_pad(q, w, cl, g)
    return (q4, w3, kv, ks, bt, cl, out, pg, prow, sp, 3), args


GROUP_PROBLEMS = [  # (heads, lens, next_n, n_cols, page, pad)
    (32, [200, 70], 6, 300, T, 0),           # groups == requests (DSpark k=5)
    (32, [250, 90, 40], 4, 300, T, 0),       # groups straddle requests: different tables per group
    (32, [150], 5, 300, T, 0),               # 5 rows: padded to one group of 6
    (32, [7, 2, 130], 6, 300, T, 0),         # cl 0 rows + sub-tile windows
    (32, [250], 6, 190, T, 0),               # contexts longer than the logits width
    (32, [300, 130], 6, 400, 128, 128),      # V4.1 ratio-1 pages, packed stride
    (32, [100, 90], 6, 200, 32, 128),        # V4.1 ratio-2 pages at the block-64 hook (P = 32)
    (64, [130, 190], 2, 300, T, 0),
]


@pytest.mark.parametrize("prob", GROUP_PROBLEMS)
@pytest.mark.parametrize("block,min_tiles,split_div", [(32, 1, 1), (16, 2, 2), (256, 4, 1)])
def test_group_body_matches_f64(monkeypatch, prob, block, min_tiles, split_div):
    heads, lens, nn, n_cols, page, pad = prob
    gargs, args = group_problem(heads, lens, nn, n_cols, seed=len(lens) + nn, page=page, pad=pad)
    ref = hdi.decode_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_group_logits, gargs, block, min_tiles, split_div)
    ok, msg = hdi.check(out, ref)
    assert ok, msg
    assert out.shape[0] == args[0].shape[0]      # padding rows never allocate an output row


@pytest.mark.parametrize("page,pad", [(32, 128), (64, 0), (128, 128)])
@pytest.mark.parametrize("length", [1, 2, 6, 63, 64, 65, 127, 128, 129, 191, 192, 193])
def test_group_body_page_and_tile_edges(monkeypatch, length, page, pad):
    """ctx 1, one token past a page / tile boundary, exact multiples: the lengths
    where masks and block-table reads go wrong."""
    gargs, args = group_problem(32, [length], 6, 260, seed=length, page=page, pad=pad)
    ref = hdi.decode_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_group_logits, gargs, 16, 1, 1)
    ok, msg = hdi.check(out, ref)
    assert ok, msg


def test_group_body_all_equal_logits_and_ties(monkeypatch):
    """q = 0 makes every logit exactly 0 (all rows tie everywhere); equal K rows and equal
    q rows tie every position at one positive value: the kernel must write the exact value
    in the window and nothing outside, and the top-k contract accepts every tie order."""
    gargs, args = group_problem(32, [200, 90], 6, 300, seed=1)
    q4, w3, kv, ks, bt, cl, out, pg, prow, sp, sps = gargs
    zero = (torch.zeros_like(q4.view(torch.uint8)).view(torch.float8_e4m3fn), w3, kv, ks, bt, cl, out.clone(),
            pg, prow, sp, sps)
    res = run_stub(monkeypatch, hdi.dsa_group_logits, zero, 32, 1, 1)
    live = ~torch.isneginf(res)
    assert live.sum(1).tolist() == [min(int(c), 300) for c in cl[:res.shape[0]]]
    assert torch.equal(res[live], torch.zeros(int(live.sum())))
    for r in range(res.shape[0]):
        n = int(live[r].sum())
        if n:
            assert di.vllm_set_ok(torch.topk(res[r, :n], min(8, n)).indices.int().tolist() + [-1] * max(0, 8 - n),
                                  res[r, :n], 8)
    # equal K rows (one page replicated), identical heads: one positive value everywhere
    ones_q = torch.ones_like(q4.float()).to(torch.float8_e4m3fn)
    kvf = torch.ones_like(kv.float()).to(torch.float8_e4m3fn)
    ksf = torch.ones_like(ks)
    wf = torch.full_like(w3, 0.01)
    flat = (ones_q, wf, kvf, ksf, bt, cl, out.clone(), pg, prow, sp, sps)
    res = run_stub(monkeypatch, hdi.dsa_group_logits, flat, 32, 1, 1)
    live = ~torch.isneginf(res)
    # (the CPU stand-in's head sum may differ by an ulp between a full and a partial tile)
    assert torch.allclose(res[live], torch.full_like(res[live], 32 * 128 * 0.01), rtol=1e-6)


def test_group_body_at_the_524288_target(monkeypatch):
    """One 524288-token request x 6 DSpark rows (the d6 target): spot-checked against the
    f64 spec on edge columns + a random sample of every row, nothing written past cl."""
    n = 524288
    g = torch.Generator().manual_seed(11)
    pagec = 64
    args = decode_problem(32, [n], 6, n, seed=3)
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = args
    q4, w3, clp = hdi.group_pad(q, w, cl, 6)
    gargs = (q4, w3, kv, ks, bt, clp, out, pg, prow, sp, 3)
    res = run_stub(monkeypatch, hdi.dsa_group_logits, gargs, 256, 1, 1)
    for r in (0, 5):
        lim = int(cl[r])
        assert lim == n - 6 + 1 + r
        assert bool(torch.isneginf(res[r, lim:]).all()) and bool(torch.isfinite(res[r, :lim]).all())
        cols = torch.cat([torch.tensor([0, 1, 63, 64, 65, lim - 1, lim - 2, lim - 65, lim - 64]),
                          torch.randint(0, lim, (1500,), generator=g)]).unique()
        t = cols
        pgs = bt[r, t // pagec].long()
        k = kv[pgs * prow + t % pagec]
        s = ks[pgs * sp + pagec * 32 + t % pagec]
        want, mag = di.ref_logits(q[r], w[r], k, s)
        err = float(((res[r, cols].double() - want).abs() / (mag + 1e-30)).max())
        assert err <= di.TOL, (r, err)


def test_group_body_equals_per_row_body(monkeypatch):
    """Same logits as dsa_decode_logits to the bit (identical f32 reduction order
    per row: the tile dot is the same)."""
    gargs, args = group_problem(32, [220, 90], 6, 300, seed=2)
    a = run_stub(monkeypatch, hdi.dsa_decode_logits, args, 32, 1, 1)
    b = run_stub(monkeypatch, hdi.dsa_group_logits, gargs, 32, 1, 1)
    live = ~torch.isneginf(a)
    assert torch.equal(a[live], b[live]) and torch.equal(torch.isneginf(a), torch.isneginf(b))


def test_group_body_reads_only_each_rows_own_pages(monkeypatch):
    """A row whose context is shorter than its group-mates' must not read (or
    write) past its own length even when its block-table row ends early: the
    padding entries are poison (a page id far outside the cache; the stand-in
    raises on any unmasked lane that dereferences it)."""
    gargs, args = group_problem(32, [250, 40, 100], 4, 300, seed=4)   # groups of 6 straddle requests
    q4, w3, kv, ks, bt, cl, out, pg, prow, sp, sps = gargs
    bt = bt.clone()
    for r in range(bt.shape[0]):
        bt[r, -(-int(cl[r]) // T):] = 10 ** 6   # past the row's own table: a page far outside the cache
    ref = hdi.decode_reference(*args)
    res = run_stub(monkeypatch, hdi.dsa_group_logits, (q4, w3, kv, ks, bt, cl, out, pg, prow, sp, sps), 32, 1, 1)
    ok, msg = hdi.check(res, ref)
    assert ok, msg


@pytest.mark.parametrize("lens,nn", [([150], 5), ([200, 70], 6), ([120, 80, 60], 4)])
def test_group_serving_entry(monkeypatch, lens, nn):
    """hdi.group_logits end to end (warm-up of every shipped bucket included):
    rows not a multiple of G are padded with dead rows and nothing outside
    a row's window is written."""
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [32]} if kernel == hdi.GROUP else None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = decode_problem(32, lens, nn, 300, seed=6)
    cache = cache_of(kv, pg, prow)
    ref = hdi.decode_reference(q, w, kv, ks, bt, cl, out, pg, prow, sp, 2)
    res = hdi.group_logits(q, w, cache.unsqueeze(2), bt, cl, out, 300, 6, 100000)
    assert res is out
    ok, msg = hdi.check(out, ref)
    assert ok, msg




def select_problem(heads, lens, nn, n_cols, keep=10, seed=0, page=T, pad=0, mode="mid"):
    """A group problem + a bound per row: mid (halfway between the keep-th and
    (keep+1)-th largest logit, clear of f32 noise), all (-inf), none (+inf)."""
    gargs, args = group_problem(heads, lens, nn, n_cols, seed=seed, page=page, pad=pad)
    q4, w3, kv, ks, bt, cl, out, pg, prow, sp, splits = gargs
    val, mag = hdi.decode_reference(*args)
    rows = val.shape[0]
    thr = torch.empty(rows)
    for r in range(rows):
        v = val[r][~torch.isnan(val[r])].float()
        t = v.topk(min(keep + 1, max(v.numel(), 1))).values if v.numel() else torch.zeros(2)
        thr[r] = (t[-1] + t[-2]) / 2 if t.numel() > 1 else 0.0
    if mode == "all":
        thr.fill_(NEG)
    elif mode == "none":
        thr.fill_(float("inf"))
    comp_v = torch.full((rows, n_cols), float("nan"))
    comp_i = torch.full((rows, n_cols), -1, dtype=torch.int32)
    cnt = torch.zeros(rows, dtype=torch.int32)
    sargs = (q4, w3, kv, ks, bt, cl, thr, comp_v, comp_i, cnt, pg, prow, sp, splits)
    return sargs, hdi.select_reference(*sargs)


@pytest.mark.parametrize("prob", GROUP_PROBLEMS[:3] + GROUP_PROBLEMS[5:7])
@pytest.mark.parametrize("mode", ["mid", "all", "none"])
@pytest.mark.parametrize("block,min_tiles,split_div", [(32, 1, 1), (16, 2, 2)])
def test_select_body_writes_exactly_the_survivors(monkeypatch, prob, mode, block, min_tiles, split_div):
    heads, lens, nn, n_cols, page, pad = prob
    sargs, ref = select_problem(heads, lens, nn, n_cols, seed=len(lens) + nn, page=page, pad=pad, mode=mode)
    out = run_stub(monkeypatch, hdi.dsa_group_select, sargs, block, min_tiles, split_div)
    ok, msg = hdi.select_check(out, ref)
    assert ok, msg
    cnt = out[2]
    if mode == "none":
        assert int(cnt.sum()) == 0
    if mode == "all":
        assert cnt.tolist() == [int(torch.clamp(c, 0, n_cols)) for c in sargs[5][:cnt.numel()]]


def test_select_check_catches_each_failure(monkeypatch):
    sargs, ref = select_problem(32, [200, 70], 6, 300, seed=1)
    good = run_stub(monkeypatch, hdi.dsa_group_select, sargs)
    assert hdi.select_check(good, ref)[0]
    v, i, n = (t.clone() for t in good)
    empty = (torch.full_like(v, float("nan")), torch.full_like(i, -1), torch.zeros_like(n))
    ok, msg = hdi.select_check(empty, ref)           # survivors missing
    assert not ok and "missing" in msg
    bad_i = i.clone()
    bad_i[0, 1] = bad_i[0, 0]                        # a position twice
    assert "twice" in hdi.select_check((v, bad_i, n), ref)[1]
    bad_v = v.clone()
    bad_v[0, 0] += 1.0                               # a logit off the spec
    assert "off the spec" in hdi.select_check((bad_v, i, n), ref)[1]
    junk = v.clone()
    junk[0, int(n[0])] = 1.0                         # written past the cursor
    assert "past the cursor" in hdi.select_check((junk, i, n), ref)[1]
    extra_n = n.clone()
    extra_n[0] += 1                                  # the cursor claims a never-written slot
    assert not hdi.select_check((v, i, extra_n), ref)[0]


@pytest.mark.parametrize("prob", [GROUP_PROBLEMS[0], GROUP_PROBLEMS[1]])
def test_select_interpret(prob):
    heads, lens, nn, n_cols, page, pad = prob
    sargs, ref = select_problem(heads, lens, nn, n_cols, seed=8, page=page, pad=pad)
    ok, msg = hdi.select_check(_interpret(hdi.dsa_group_select, sargs), ref)
    assert ok, msg


def test_select_tune_check_ignores_the_atomic_race_order():
    v = torch.tensor([[1.0, 2.0, 3.0, float("nan")]])
    i = torch.tensor([[7, 3, 9, -1]], dtype=torch.int32)
    n = torch.tensor([3], dtype=torch.int32)
    hdi._select_tune_check((v, i, n), (v[:, [1, 2, 0, 3]], i[:, [1, 2, 0, 3]], n))
    with pytest.raises(AssertionError):
        hdi._select_tune_check((v, i, n), (v, i + 1, n))
    with pytest.raises(AssertionError):
        hdi._select_tune_check((v, i, n), (v, i, n + 1))

@pytest.mark.parametrize("page,pad", [(128, 0), (64, 256), (128, 128)])
def test_decode_body_free_page_layout(monkeypatch, page, pad):
    """DeepSeek-V4.1: 128-token pages (ratio-1 layers) and a page stride
    beside the MLA pages; the pad bytes are garbage the body must not read."""
    args = decode_problem(32, [300, 130], 6, 400, seed=3, page=page, pad=pad)
    assert args[7:10] == (page, (page * 132 + pad) // 128, (page * 132 + pad) // 4)
    ref = hdi.decode_reference(*args)
    ok, msg = hdi.check(run_stub(monkeypatch, hdi.dsa_decode_logits, args, 32, 2, 1), ref)
    assert ok, msg


@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize("block,min_tiles,split_div,splits", [(32, 1, 1, 3), (16, 1, 2, 7), (64, 4, 1, 2)])
def test_prefill_body_matches_f64(monkeypatch, block, min_tiles, split_div, splits, compressed):
    args = prefill_problem(splits=splits, compressed=compressed)
    assert args[7].shape[1] == (int((args[5] - args[4]).max()) if compressed else 150)
    ref = hdi.prefill_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_prefill_logits, args, block, min_tiles, split_div)
    ok, msg = hdi.check(out, ref)
    assert ok, msg


CAND_PROBLEMS = [  # (heads, lens, next_n, ncand, bsz, page, pad)
    (32, [300, 70], 6, 16, 8, T, 0),      # long rows (> C x B) + short ones (-1 padding)
    (32, [1000], 1, 32, 8, 128, 256),     # 128-token pages, strided
    (64, [9, 0, 200], 1, 8, 8, T, 0),     # a cl-0 row, a sub-block row
]


@pytest.mark.parametrize("prob", CAND_PROBLEMS)
@pytest.mark.parametrize("block,min_tiles,split_div", [(16, 1, 1), (32, 2, 2), (512, 1, 1)])
def test_cand_body_matches_f64(monkeypatch, prob, block, min_tiles, split_div):
    heads, lens, nn, ncand, bsz, page, pad = prob
    args = cand_problem(heads, lens, nn, ncand, bsz, page, pad, seed=len(lens))
    ref = hdi.cand_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_cand_logits, args, block, min_tiles, split_div)
    ok, msg = hdi.check(out, ref)
    assert ok, msg
    cl, lens_c = args[5], hdi.cand_lens(args[5], ncand, bsz)
    assert torch.isneginf(out[lens_c == 0]).all()
    assert int(lens_c.max()) <= ncand * bsz and bool((lens_c[cl > 0] > 0).all())


def _topk(logits, lens, k):
    """vLLM's decode top-k contract (persistent_topk): rows <= k long return
    0..len-1 without ranking, else the top-k of [0, len); -1 fill."""
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r, n in enumerate(lens.reshape(-1).tolist()):
        if n <= k:
            out[r, :n] = torch.arange(n, dtype=torch.int32)
        else:
            out[r] = torch.topk(logits[r, :n], k).indices.to(torch.int32)
    return out


def _vllm_mask(dense, cl, cand, bsz):
    """apply_candidate_mask (decode): keep [0, cl) columns whose block is a
    candidate, -inf everywhere else."""
    out = torch.full_like(dense, NEG)
    for r in range(dense.shape[0]):
        n = int(cl[r])
        keep = torch.zeros(dense.shape[1], dtype=torch.bool)
        for b in cand[r].tolist():
            if b >= 0:
                keep[b * bsz:(b + 1) * bsz] = True
        keep[n:] = False
        out[r, keep] = dense[r, keep]
    return out


@pytest.mark.parametrize("prob", CAND_PROBLEMS)
@pytest.mark.parametrize("k", [8, 64])
def test_fused_topk_equals_dense_mask(monkeypatch, prob, k):
    """vLLM's path (dense logits -> candidate mask -> top-k) and ours (fused
    candidate slots -> top-k over cand_lens -> cand_remap) select the same
    positions."""
    heads, lens, nn, ncand, bsz, page, pad = prob
    args = cand_problem(heads, lens, nn, ncand, bsz, page, pad, seed=11)
    q, w, kv, ks, bt, cl, cand, out, pg, prow, sp, _ = args
    width = max(lens) + 5
    val, _ = hdi.decode_reference(q, w, kv, ks, bt, cl, torch.empty((cl.shape[0], width)), pg, prow, sp, 2)
    dense = torch.where(torch.isnan(val), NEG, val).float()
    want = _topk(_vllm_mask(dense, cl, cand, bsz), cl, k)
    fused = run_stub(monkeypatch, hdi.dsa_cand_logits, args, 32)
    got = hdi.cand_remap(_topk(fused, hdi.cand_lens(cl, ncand, bsz), k), cand, cl, bsz)
    for r in range(cl.shape[0]):
        w_ = {i for i in want[r].tolist() if i >= 0 and dense[r, i] > NEG}
        g_ = {i for i in got[r].tolist() if i >= 0}
        assert g_ == w_, (r, sorted(g_ ^ w_))


def test_cand_remap_marks_dead_slots():
    cand = torch.tensor([[3, -1, 0]], dtype=torch.int32)
    idx = torch.tensor([[0, 5, 8, 16, 17, -1]], dtype=torch.int32)
    hdi.cand_remap(idx, cand, torch.tensor([30], dtype=torch.int32), 8)
    assert idx.tolist() == [[24, 29, -1, 0, 1, -1]]
    idx = torch.tensor([[7]], dtype=torch.int32)        # position 31 past cl 30
    assert hdi.cand_remap(idx, cand, torch.tensor([30], dtype=torch.int32), 8).tolist() == [[-1]]


def test_check_catches_writes_outside_and_errors():
    args = decode_problem(32, [100], 1, 200)
    val, mag = hdi.decode_reference(*args)
    out = torch.where(torch.isnan(val), NEG, val).float()
    assert hdi.check(out, (val, mag))[0]
    bad = out.clone()
    bad[0, 150] = 0.0
    assert not hdi.check(bad, (val, mag))[0]
    bad = out.clone()
    bad[0, 5] += 1.0
    assert not hdi.check(bad, (val, mag))[0]


def test_cache_views_address_the_split_page():
    _, _, kv, ks, bt, cl, _, page, prow, sp, _ = decode_problem(32, [130], 1, 200)
    assert (page, prow, sp) == (64, 66, 2112)
    cache = kv.view(torch.uint8).view(-1, T, 132)
    k, s = di.gather_paged(cache, bt[0], 130)
    k2, s2 = hdi.gather_views(kv, ks, page, prow, sp, bt[0], 130)
    assert torch.equal(k2.view(torch.uint8), k.view(torch.uint8)) and torch.equal(s2, s)


def test_storage_span_counts_the_page_stride_gaps():
    """The int32 / int64 index class follows the flat views' extent, not the element
    count of the strided cache view (a V4.1 page sits beside MLA pages)."""
    back = torch.zeros(10 * 8704 + 8448, dtype=torch.uint8)
    cache = back.as_strided((11, 64, 132), (8704, 132, 1))
    assert hdi.storage_span(cache) == 10 * 8704 + 64 * 132 == hdi.cache_views(cache)[0].numel()
    assert hdi.storage_span(cache.unsqueeze(2)) == hdi.storage_span(cache)
    assert cache.numel() < hdi.storage_span(cache)


def test_layout_reason():
    u8 = torch.uint8
    assert hdi.layout_reason(torch.zeros((4, 64, 1, 132), dtype=u8)) is None
    assert hdi.layout_reason(build_cache(torch.zeros((256, D), dtype=torch.float8_e4m3fn),
                                         torch.ones(256), 128, 384)) is None
    assert "P % 32" in hdi.layout_reason(torch.zeros((4, 16, 132), dtype=u8))
    assert "P % 32" in hdi.layout_reason(torch.zeros(4 * (64 * 132 + 4), dtype=u8)
                                         .as_strided((4, 64, 132), (64 * 132 + 4, 132, 1)))
    assert "uint8" in hdi.layout_reason(torch.zeros((4, 64, 136), dtype=u8))


# ---------------------------------------------------------------------------
# lab ABI
# ---------------------------------------------------------------------------
def test_kernels_abi():
    glm = {"architecture": "GlmMoeDsaForCausalLM", "name": "glm-5.3-nvfp4", "tp": 8}
    dsv41 = {"architecture": "DeepseekV41ForCausalLM", "name": "deepseek-v4.1-flash-1017-nvfp4"}
    names = [k["name"] for k in hdi.KERNELS]
    assert names == [hdi.DECODE, hdi.GROUP, hdi.SELECT, hdi.PREFILL, hdi.CANDK, hdi.DECODE + "_dg",
                     hdi.GROUP + "_dg"]
    K = {k["name"]: k for k in hdi.KERNELS}
    for k in hdi.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"])
        assert k["body"].__module__ == hdi.__name__ and not hasattr(k["body"], "bind")
        for key in ("make", "reference", "baseline", "check", "cases"):
            assert callable(k[key])
        assert k["baseline_name"].startswith(("oxide dsa_indexer", "vLLM DeepGEMM"))
        assert len(k["replaces"]) <= 16 and all(re.compile(r) for r in k["replaces"])
        for model in (glm, dsv41):
            cases = k["cases"](model)
            assert len(cases) <= 64 and all("numtokens" in c for c in cases)
            assert all(isinstance(v, (int, float, str, bool)) for c in cases for v in c.values())
            assert all(c["heads"] == 32 for c in cases)
        assert bool(k["cases"](glm)) == (k["name"] in (hdi.DECODE, hdi.GROUP, hdi.PREFILL))
        assert k["cases"](dsv41)
        assert k["cases"]({"architecture": "Qwen3ForCausalLM"}) == [] and k["cases"]({}) == []
    cand = K[hdi.CANDK]["cases"](dsv41)
    assert {(c["cand"], c["cbs"], c["ctx"]) for c in cand} == {(2048, 8, hdi.CAND_CTX)}
    assert [c["rows"] for c in K[hdi.DECODE + "_dg"]["cases"](dsv41)] == [1, 6, 48, 192]
    # DSpark block 5 verifies 1 + 5 rows per request: the MTP bucket is 6
    assert K[hdi.DECODE]["cases"](dsv41) == K[hdi.DECODE]["cases"](glm)
    assert K[hdi.PREFILL]["cases"]({"architecture": "DeepseekV32ForCausalLM"})[0]["heads"] == 64
    # row-group decode: G = 6, requests x context buckets up to the 524288 target
    grp = K[hdi.GROUP]["cases"](dsv41)
    assert {(c["ctx"], c["numtokens"]) for c in grp} == {
        (c, r) for c, rs in hdi.GROUP_REQS.items() for r in rs}
    assert max(c["ctx"] for c in grp) == 524288 and all(c["rows"] == c["numtokens"] * 6 for c in grp)
    assert {(c["ctx"], c["numtokens"]) for c in K[hdi.GROUP + "_dg"]["cases"](dsv41)} == {
        (32768, 1), (32768, 8), (524288, 1), (524288, 4)}
    assert K[hdi.GROUP + "_dg"]["cases"](glm) == [] and K[hdi.SELECT]["cases"](glm) == []
    assert {(c["ctx"], c["numtokens"]) for c in K[hdi.SELECT]["cases"](dsv41)} == set(hdi.SELECT_CASES)


def test_serving_buckets():
    assert hdi.serving_case(hdi.DECODE, 32, 5) == hdi.decode_case(32, 6)
    assert hdi.serving_case(hdi.DECODE, 32, 7)["rows"] == 12
    assert hdi.serving_case(hdi.DECODE, 32, 9999)["rows"] == hdi.DECODE_ROWS[-1]
    assert hdi.serving_case(hdi.PREFILL, 64, 300)["m"] == 1024
    assert hdi.splits_for(1, 1, 188) == 16 and hdi.splits_for(1, 202752, 188) == 1504
    # always in one Triton specialisation class (a multiple of 16), whatever the batch
    assert all(hdi.splits_for(u, c, 188) % 16 == 0 for u in range(1, 40) for c in (1, 300, 8192, 524288))


# ---------------------------------------------------------------------------
# serving routing
# ---------------------------------------------------------------------------
class _Native:
    def __init__(self):
        self.calls = 0

    def dsa_indexer_logits_cuda(self, *a):
        self.calls += 1


def _paged_call(entry):
    q, w, kv, ks, bt, cl, out, *_ = decode_problem(32, [100], 1, 200)
    cache = kv.view(torch.uint8).view(-1, T, 132)
    return entry(out, q, w, cache, None, bt, None, cl, 200, 1, 4, 1, 2, 2, 0)


def test_backend_routing(monkeypatch):
    nat = _Native()
    monkeypatch.setattr(di, "_native", lambda dev: nat)
    monkeypatch.delenv(di.HELION_ENV, raising=False)
    assert di._logits_op(torch.device("cpu")) == nat.dsa_indexer_logits_cuda
    monkeypatch.setenv(di.HELION_ENV, "auto")
    assert di._logits_op(torch.device("cpu")) is hdi.dsa_indexer_logits_cuda
    # auto, nothing shipped: oxide serves, untouched args
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: None)
    monkeypatch.setattr(hdi, "_WARM", set())
    _paged_call(hdi.dsa_indexer_logits_cuda)
    assert nat.calls == 1
    # helion, nothing shipped: fail closed at load
    monkeypatch.setenv(di.HELION_ENV, "helion")
    monkeypatch.setattr(hk, "platform", lambda device=0: "cpu_test")
    monkeypatch.setattr(hdi, "_WARM", set())
    with pytest.raises(RuntimeError, match="no tuned dsa_indexer_decode config"):
        _paged_call(hdi.dsa_indexer_logits_cuda)
    assert nat.calls == 1


def test_helion_entry_runs_the_body(monkeypatch):
    """helion backend with a config for every bucket: the entry warms every
    bucket, then serves through the body (stand-in) == f64."""
    monkeypatch.setenv(di.HELION_ENV, "helion")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: {"block_sizes": [32]})
    monkeypatch.setattr(hdi, "_WARM", set())
    stub = make_stub(32, {"min_tiles": 1, "split_div": 1})
    monkeypatch.setattr(hdi, "hl", stub)
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    warmed = []
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: warmed.append(case) or body)
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = decode_problem(32, [200, 70], 6, 300, page=128, pad=128)
    cache = cache_of(kv, pg, prow)
    hdi.dsa_indexer_logits_cuda(out, q, w, cache, None, bt, None, cl, 300, 6, 1, 1, 2, 2, 0)
    assert len(warmed) == len(hdi.DECODE_ROWS) + 1
    ok, msg = hdi.check(out, hdi.decode_reference(q, w, kv, ks, bt, cl, out, pg, prow, sp, 2))
    assert ok, msg


def test_prefill_and_cand_entries(monkeypatch):
    """The lane's compressed-prefill and fused-candidate entries run their
    bodies (stand-in) on vLLM's argument forms."""
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: {"block_sizes": [32]})
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    q, kv, ksc, w, lo, hi, off, out, _ = prefill_problem(heads=32)
    got = hdi.prefill_logits(q, kv, ksc, w, lo, hi, out, compressed=True)
    ok, msg = hdi.check(got, hdi.prefill_reference(q, kv, ksc, w, lo, hi, off, out, 2))
    assert ok, msg
    args = cand_problem(32, [300, 70], 6, page=128, pad=128)
    q, w, kv, ks, bt, cl, cand, out, pg, prow, sp, _ = args
    cache = cache_of(kv, pg, prow)
    hdi.cand_logits(q, w, cache.unsqueeze(2), bt, cl, cand, out)
    ok, msg = hdi.check(out, hdi.cand_reference(*args))
    assert ok, msg


def test_lane_names_the_knob():
    src = open(di.__file__).read()
    assert di.HELION_ENV == "SUFFIX_HELION_DSA_INDEXER" and "_logits_op(dev)(" in src
    assert "_native(dev).dsa_indexer_logits_cuda(" not in src.replace(
        "return _native(dev).dsa_indexer_logits_cuda", "")


# ---------------------------------------------------------------------------
# Helion eager (HELION_INTERPRET=1)
# ---------------------------------------------------------------------------
def _interpret(body, args):
    helion = pytest.importorskip("helion")
    if hdi.helion is None:
        pytest.skip("helion imported after the port module")
    kern = helion.kernel(body, config=helion.Config(min_tiles=1, split_div=1))
    return kern(*args)


@pytest.mark.parametrize("heads,lens,next_n,n_cols", DECODE_PROBLEMS)
def test_decode_interpret(heads, lens, next_n, n_cols):
    args = decode_problem(heads, lens, next_n, n_cols, seed=7)
    ref = hdi.decode_reference(*args)
    ok, msg = hdi.check(_interpret(hdi.dsa_decode_logits, args), ref)
    assert ok, msg


def test_prefill_interpret():
    args = prefill_problem()
    ref = hdi.prefill_reference(*args)
    ok, msg = hdi.check(_interpret(hdi.dsa_prefill_logits, args), ref)
    assert ok, msg


@pytest.mark.parametrize("prob", CAND_PROBLEMS[:2])
def test_cand_interpret(prob):
    heads, lens, nn, ncand, bsz, page, pad = prob
    args = cand_problem(heads, lens, nn, ncand, bsz, page, pad, seed=4)
    ok, msg = hdi.check(_interpret(hdi.dsa_cand_logits, args), hdi.cand_reference(*args))
    assert ok, msg


@pytest.mark.parametrize("page,pad", [(128, 128)])
def test_decode_interpret_free_pages(page, pad):
    args = decode_problem(32, [300, 70], 6, 400, seed=9, page=page, pad=pad)
    ok, msg = hdi.check(_interpret(hdi.dsa_decode_logits, args), hdi.decode_reference(*args))
    assert ok, msg


@pytest.mark.parametrize("prob", [GROUP_PROBLEMS[0], GROUP_PROBLEMS[1], GROUP_PROBLEMS[2],
                                  GROUP_PROBLEMS[5], GROUP_PROBLEMS[6]])
def test_group_interpret(prob):
    heads, lens, nn, n_cols, page, pad = prob
    gargs, args = group_problem(heads, lens, nn, n_cols, seed=8, page=page, pad=pad)
    ok, msg = hdi.check(_interpret(hdi.dsa_group_logits, gargs), hdi.decode_reference(*args))
    assert ok, msg


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("body,make", [
    (hdi.dsa_group_logits, lambda: group_problem(32, [200, 70], 6, 300)[0]),
    (hdi.dsa_group_select, lambda: select_problem(32, [200, 70], 6, 300)[0]),
    (hdi.dsa_decode_logits, lambda: decode_problem(32, [200, 70], 6, 300)),
    (hdi.dsa_prefill_logits, prefill_problem),
    (hdi.dsa_cand_logits, lambda: cand_problem(32, [300, 70], 6)),
])
def test_triton_codegen(body, make, static):
    """Helion's compiler front end accepts both bodies (device-side loop bounds,
    scalar row + page-table gather, head loop, SymInt clamp): Triton source
    for the default config, serving (dynamic) and lab (static) shapes."""
    helion = pytest.importorskip("helion")
    if hdi.helion is None:
        pytest.skip("helion imported after the port module")
    from helion.runtime.ref_mode import RefMode
    bound = helion.kernel(body, static_shapes=static, ref_mode=RefMode.OFF).bind(make())
    code = bound.to_triton_code(bound.config_spec.default_config())
    assert "tl.dot" in code and "tl.float8e4nv" in code


def test_tune_check_compares_only_the_windows():
    args = decode_problem(32, [100], 1, 200)
    val, _ = hdi.decode_reference(*args)
    base = torch.where(torch.isnan(val), NEG, val).float()
    junk = base.clone()
    junk[0, 150:] = 7.0                       # outside the window: ignored
    hdi._tune_check((junk, 3), (base, 3))
    junk[0, 5] += 1.0                         # inside: rejected
    with pytest.raises(AssertionError):
        hdi._tune_check(junk, base)
    assert all("autotune_baseline_fn" in k["settings"] for k in hdi.KERNELS)
    # every body is searched inside the bounded space (kernels/dsa_capture.search_filter)
    assert all(k["settings"].get("autotune_config_filter") is cap.search_filter for k in hdi.KERNELS)
