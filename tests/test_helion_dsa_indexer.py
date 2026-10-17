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
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

T, D = 64, 128
NEG = float("-inf")


# ---------------------------------------------------------------------------
# problems (CPU, small)
# ---------------------------------------------------------------------------
def decode_problem(heads, lens, next_n, n_cols, seed=0):
    """vLLM SM120 flattened decode: rows of a request carry copies of its
    block-table row, row j of a request sees L - next_n + 1 + j tokens."""
    g = torch.Generator().manual_seed(seed)
    nblk = [-(-n // T) for n in lens]
    num_blocks = sum(nblk) + 2
    perm = torch.randperm(num_blocks, generator=g)
    bt = torch.full((len(lens), max(nblk) + 1), -1, dtype=torch.int32)
    o = 0
    for i, nb in enumerate(nblk):
        bt[i, :nb] = perm[o:o + nb].to(torch.int32)
        o += nb
    kf = (torch.randn((num_blocks * T, D), generator=g) * 0.7).to(torch.float8_e4m3fn)
    ksc = 2.0 ** torch.randint(-6, 0, (num_blocks * T,), generator=g).float()
    cache = di.build_paged_cache(kf, ksc, torch.arange(num_blocks, dtype=torch.int32), num_blocks)
    rows = len(lens) * next_n
    q = (torch.randn((rows, heads, D), generator=g) * 1.5).to(torch.float8_e4m3fn)
    w = torch.randn((rows, heads), generator=g) * 0.1
    cl = (torch.tensor(lens).unsqueeze(1) - next_n + 1 + torch.arange(next_n)).clamp_min(0).to(torch.int32)
    pr = {"rows": rows, "heads": heads, "cache": cache.unsqueeze(2), "q": q.view(rows, 1, heads, D),
          "w": w, "cl": cl.reshape(rows, 1), "bt": bt.repeat_interleave(next_n, 0), "next_n": next_n}
    return hdi.decode_args(pr, n_cols, n_sms=2)


def prefill_problem(heads=64, m=11, n=150, splits=3):
    return (*hdi.prefill_problem(heads, m, n, "cpu", seed=5), splits)


# ---------------------------------------------------------------------------
# eager stand-in for helion.language (ref-mode semantics, tunable block sizes)
# ---------------------------------------------------------------------------
class _Tile(torch.Tensor):
    pass


def _tile(b, e):
    t = torch.arange(b, e).as_subclass(_Tile)
    t.begin, t.index = b, torch.arange(b, e)
    return t


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
        ix = [i.index if isinstance(i, _Tile) else i for i in index]
        if len(ix) == 2 and all(isinstance(i, torch.Tensor) for i in ix):
            ix = [ix[0][:, None], ix[1][None, :]]
        ix = tuple(ix)
        val = val.as_subclass(torch.Tensor).to(t.dtype)
        t[ix] = val if extra_mask is None else torch.where(extra_mask, val, t[ix])

    return types.SimpleNamespace(
        tile=tile, grid=grid, store=store, specialize=int,
        register_block_size=lambda lo, hi: max(lo, min(hi, block)),
        register_tunable=lambda name, frag: tunables[name],
        dot=lambda a, b, out_dtype=None: a.float() @ b.float(),
        zeros=lambda shape, dtype: torch.zeros([len(s) for s in shape], dtype=dtype))


def run_stub(monkeypatch, body, args, block=32, min_tiles=1, split_div=1):
    monkeypatch.setattr(hdi, "hl", make_stub(block, {"min_tiles": min_tiles, "split_div": split_div}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    return body(*args).as_subclass(torch.Tensor)


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


@pytest.mark.parametrize("block,min_tiles,split_div,splits", [(32, 1, 1, 3), (16, 1, 2, 7), (64, 4, 1, 2)])
def test_prefill_body_matches_f64(monkeypatch, block, min_tiles, split_div, splits):
    args = prefill_problem(splits=splits)
    ref = hdi.prefill_reference(*args)
    out = run_stub(monkeypatch, hdi.dsa_prefill_logits, args, block, min_tiles, split_div)
    ok, msg = hdi.check(out, ref)
    assert ok, msg


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
    _, _, kv, ks, bt, cl, _, _ = decode_problem(32, [130], 1, 200)
    cache = kv.view(torch.uint8).view(-1, T, 132)
    k, s = di.gather_paged(cache, bt[0], 130)
    t = torch.arange(130)
    pg = bt[0, t // T].long()
    assert torch.equal(kv[pg * 66 + t % T].view(torch.uint8), k.view(torch.uint8))
    assert torch.equal(ks[pg * 2112 + 2048 + t % T], s)


# ---------------------------------------------------------------------------
# lab ABI
# ---------------------------------------------------------------------------
def test_kernels_abi():
    glm = {"architecture": "GlmMoeDsaForCausalLM", "name": "glm-5.3-nvfp4", "tp": 8}
    for k in hdi.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"])
        assert k["body"].__module__ == hdi.__name__ and not hasattr(k["body"], "bind")
        for key in ("make", "reference", "baseline", "check", "cases"):
            assert callable(k[key])
        assert k["baseline_name"] == "oxide dsa_indexer"
        assert len(k["replaces"]) <= 16 and all(re.compile(r) for r in k["replaces"])
        cases = k["cases"](glm)
        assert 0 < len(cases) <= 64 and all("numtokens" in c for c in cases)
        assert all(isinstance(v, (int, float, str, bool)) for c in cases for v in c.values())
        assert all(c["heads"] == 32 for c in cases)
        assert k["cases"]({"architecture": "Qwen3ForCausalLM"}) == [] and k["cases"]({}) == []
    assert hdi.KERNELS[1]["cases"]({"architecture": "DeepseekV32ForCausalLM"})[0]["heads"] == 64
    assert any(re.search(r, "dsa_logits", re.I) for r in hdi.KERNELS[0]["replaces"])
    dec = {c["rows"]: c["numtokens"] for c in hdi.KERNELS[0]["cases"](glm)}
    assert dec[6] == 1 and dec[48] == 8 and dec[192] == 32 and dec[1] == 1


def test_serving_buckets():
    assert hdi.serving_case(hdi.DECODE, 32, 5) == hdi.decode_case(32, 6)
    assert hdi.serving_case(hdi.DECODE, 32, 7)["rows"] == 12
    assert hdi.serving_case(hdi.DECODE, 32, 9999)["rows"] == hdi.DECODE_ROWS[-1]
    assert hdi.serving_case(hdi.PREFILL, 64, 300)["m"] == 1024
    assert hdi.splits_for(1, 1, 188) == 2 and hdi.splits_for(1, 202752, 188) == 1504


# ---------------------------------------------------------------------------
# serving routing
# ---------------------------------------------------------------------------
class _Native:
    def __init__(self):
        self.calls = 0

    def dsa_indexer_logits_cuda(self, *a):
        self.calls += 1


def _paged_call(entry):
    q, w, kv, ks, bt, cl, out, _ = decode_problem(32, [100], 1, 200)
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
    q, w, kv, ks, bt, cl, out, _ = decode_problem(32, [200, 70], 6, 300)
    cache = kv.view(torch.uint8).view(-1, T, 132)
    hdi.dsa_indexer_logits_cuda(out, q, w, cache, None, bt, None, cl, 300, 6, 1, 1, 2, 2, 0)
    assert len(warmed) == len(hdi.DECODE_ROWS) + 1
    ok, msg = hdi.check(out, hdi.decode_reference(q, w, kv, ks, bt, cl, out, 2))
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


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("body,make", [
    (hdi.dsa_decode_logits, lambda: decode_problem(32, [200, 70], 6, 300)),
    (hdi.dsa_prefill_logits, prefill_problem),
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
