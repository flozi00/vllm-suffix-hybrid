# SPDX-License-Identifier: Apache-2.0
"""CPU tests for fp8_dense_patch/ours.py (SUFFIX_FP8_DENSE_BACKEND ours / best / ours_fp4).

The REAL Triton kernel source runs under a small torch interpreter of the triton.language
subset it uses (programs one at a time in a shuffled order, so the last-CTA split-K
reduction sees arbitrary arrival; out-of-storage offsets raise) against the exact
dequantized reference: every planned (tile, split) at several M, both weight kinds, a
ragged N, head-batched strided inputs. Plus planning, routing, banks, the JSON cache, and
convert_model end to end over fake vLLM through torch.ops.suffix_fp8d.linear."""
import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
sys.path.insert(0, str(REPO))
from fp8_dense_patch import runtime as R  # noqa: E402

# --- a torch interpreter for the triton.language subset ours.py uses ----------------
_RAW = {torch.float8_e4m3fn: torch.uint8}
_CTX = {}


class _Ptr:
    def __init__(self, buf, off, dt):
        self.buf, self.off, self.dt = buf, off, dt

    @property
    def dtype(self):
        return types.SimpleNamespace(element_ty=self.dt)

    def __add__(self, o):
        return _Ptr(self.buf, self.off + o, self.dt)

    __radd__ = __add__


def _ptr(t):
    raw = _RAW.get(t.dtype, t.dtype)
    st = t.untyped_storage()
    n = st.nbytes() // torch.empty(0, dtype=raw).element_size()
    return _Ptr(torch.empty(0, dtype=raw).set_(st, 0, (n,), (1,)), t.storage_offset(), t.dtype)


def _bcast(ptr, mask):
    off = ptr.off if torch.is_tensor(ptr.off) else torch.tensor(ptr.off)
    m = mask if torch.is_tensor(mask) else torch.tensor(True if mask is None else bool(mask))
    return torch.broadcast_tensors(off, m)


def _load(ptr, mask=None, other=0.0, cache_modifier=None, eviction_policy=None):
    assert other == 0
    off, m = _bcast(ptr, mask)
    raw = torch.zeros(off.shape, dtype=ptr.buf.dtype)
    raw[m] = ptr.buf[off[m]]
    return raw.view(ptr.dt) if ptr.dt in _RAW else raw


def _store(ptr, val, mask=None):
    assert val.dtype == ptr.dt, (val.dtype, ptr.dt)  # every store casts explicitly
    off, m = _bcast(ptr, mask)
    v = val.view(torch.uint8) if ptr.dt in _RAW else val
    ptr.buf[off[m]] = torch.broadcast_to(v, off.shape)[m]


def _atomic(op):
    def f(ptr, val, sem=None, scope=None):
        old = int(ptr.buf[ptr.off])
        ptr.buf[ptr.off] = op(old, val)
        return old
    return f


def _mm(fn):
    def f(a, b):
        if not torch.is_tensor(a) and not torch.is_tensor(b):
            return fn(torch.tensor(a), torch.tensor(b)).item()
        ref = a if torch.is_tensor(a) else b
        return fn(torch.as_tensor(a, dtype=ref.dtype), torch.as_tensor(b, dtype=ref.dtype))
    return f


class _Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *a, **k):  # a @jit helper called from a kernel
        return self.fn(*a, **k)

    def __getitem__(self, grid):
        def launch(*args, num_warps=4, num_stages=1, **const):
            g = tuple(grid) + (1,) * (3 - len(grid))
            args = [_ptr(a) if torch.is_tensor(a) else a for a in args]
            pids = [(i, j, l) for i in range(g[0]) for j in range(g[1]) for l in range(g[2])]
            random.Random(len(pids)).shuffle(pids)
            for pid in pids:
                _CTX["pid"], _CTX["grid"] = pid, g
                self.fn(*args, **const)
        return launch


def _fake_triton():
    tl = types.ModuleType("triton.language")
    tl.constexpr = type("constexpr", (), {})
    tl.float16, tl.bfloat16, tl.float32 = torch.float16, torch.bfloat16, torch.float32
    tl.int16, tl.int32, tl.float8e4nv = torch.int16, torch.int32, torch.float8_e4m3fn
    tl.program_id = lambda i: _CTX["pid"][i]
    tl.num_programs = lambda i: _CTX["grid"][i]
    tl.cdiv = lambda a, b: -(-a // b)
    tl.arange = lambda a, b: torch.arange(a, b)
    tl.zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    tl.minimum, tl.maximum = _mm(torch.minimum), _mm(torch.maximum)
    tl.load, tl.store = _load, _store
    tl.atomic_add = _atomic(lambda old, v: old + v)
    tl.atomic_xchg = _atomic(lambda old, v: v)
    tl.debug_barrier = lambda: None
    tl.range = lambda a, b, s=1, loop_unroll_factor=None: range(int(a), int(b), int(s))
    tl.static_range = lambda n: range(n)
    tl.dot = lambda a, b, acc=None: (a.float() @ b.float()) + (0 if acc is None else acc)
    tl.trans = lambda x: x.transpose(-2, -1)
    tl.reshape = lambda x, shape: x.reshape(shape)
    tl.split = lambda x: x.unbind(-1) if x.shape[-1] == 2 else pytest.fail("split needs 2")
    tl.join = lambda a, b: torch.stack((a, b), -1)
    tl.sum = lambda x, axis: x.sum(axis)
    tl.max = lambda x, axis: x.amax(axis)
    tl.abs, tl.where = torch.abs, torch.where
    tl.cast = lambda x, dt, bitcast=False, fp_downcast_rounding=None: (
        x.view(dt) if bitcast else x.to(dt))  # torch rounds to nearest even
    tr = types.ModuleType("triton")
    tr.language, tr.cdiv = tl, tl.cdiv
    tr.jit = lambda fn=None, **kw: _Kernel(fn) if fn is not None else _Kernel
    return tr, tl


@pytest.fixture(scope="module")
def O():
    tr, tl = _fake_triton()
    saved = {k: sys.modules.get(k) for k in ("triton", "triton.language")}
    sys.modules["triton"], sys.modules["triton.language"] = tr, tl
    try:
        spec = importlib.util.spec_from_file_location(
            "ours_under_fake_triton", REPO / "sm120/fp8_dense_patch/ours.py")
        mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # dataclasses
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert mod.triton is tr
    return mod


def fp4_quant(w, g=None):
    """CPU stand-in for vLLM scaled_fp4_quant (linear SF layout): packed u8 [N, K/2]
    (low nibble = even k), e4m3 [N, K/16], global g; w ~= e2m1 * sf / g."""
    n, k = w.shape
    wf = w.float()
    if g is None:
        g = torch.tensor([448.0 * 6.0], dtype=torch.float32) / wf.abs().amax().clamp_min(1e-12)
    blocks = wf.view(n, k // 16, 16)
    sf = (blocks.abs().amax(-1) * g / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    mag = (blocks / (sf.float() / g).clamp_min(1e-30)[..., None]).abs()
    idx = (mag[..., None] - torch.tensor(R.E2M1, dtype=torch.float32)).abs().argmin(-1)
    code = (idx | ((blocks < 0).long() << 3)).view(n, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8), sf, g


def _weights(O, kind, b, n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(b, n, k, generator=g) * 0.02).bfloat16()
    if kind == "w8":
        q, s = R.quantize_weight(w.reshape(b * n, k))
        wt = O.Weights.fp8(q.view(b, n, k), s.view(b, n)) if b > 1 else O.Weights.fp8(q, s)
    else:
        per = [fp4_quant(w[h]) for h in range(b)]
        if b == 1:
            wt = O.Weights.fp4(*per[0])
        else:
            wt = O.Weights.fp4(torch.stack([p[0] for p in per]), torch.stack([p[1] for p in per]),
                               torch.cat([p[2] for p in per]))
    return w, wt


def _x(b, m, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    if b > 1:  # vLLM's absorbed-MLA layout: [T, H, K] viewed as [H, T, K]
        return torch.randn(m, b, k, generator=g).bfloat16().transpose(0, 1)
    return torch.randn(m, k, generator=g).bfloat16()


# --- the kernel itself ----------------------------------------------------------------
@pytest.mark.parametrize("kind", ["w8", "w4"])
@pytest.mark.parametrize("b,n,k,ms", [(1, 96, 512, (1, 3, 6, 17, 40, 70)),
                                      (1, 80, 768, (2, 8, 33)),      # N % 32 != 0: masks
                                      (3, 48, 192, (1, 5, 24))])     # heads, K = 3 x 64
def test_every_planned_kernel_matches_the_exact_reference(O, kind, b, n, k, ms):
    _w, wt = _weights(O, kind, b, n, k)
    deq = O.dequant(wt)
    sem = torch.zeros(O.SEM_SLOTS, dtype=torch.int32)
    tried = set()
    for m in ms:
        x = _x(b, m, k)
        for plan in O.plans(kind, b, n, k, m, nsm=8):
            for p in (plan, O.Plan(plan.cfg, plan.kb_split, plan.n_split, True)) \
                    if plan.n_split > 1 else (plan,):
                y = O.run(x, wt, p, sem)
                assert y.shape == ((b, m, n) if b > 1 else (m, n)) and y.dtype == torch.bfloat16
                err = O.rel_err(y, O.reference(x, deq, p.cfg))
                assert err <= O.OURS_BOUND, (p.tag(), m, err)
                assert not sem.any(), "split-K counters must be rearmed"
                tried.add((p.cfg.mma, p.cfg.a8, p.cfg.xs, p.n_split > 1, p.two))
    want = {(False, False, False, True, False), (True, False, False, True, True),
            (True, False, False, False, False)}
    want |= ({(True, False, True, True, False)} if kind == "w4"
             else {(True, True, False, True, False)})
    assert want <= tried


def test_big_m_unsplit_tiles_and_empty_batch(O):
    _w, wt = _weights(O, "w4", 1, 64, 256)
    x = _x(1, 130, 256)
    for plan in O.plans("w4", 1, 64, 256, 130, nsm=8, big=True):
        assert plan.cfg.bm == 128 and plan.n_split == 1
        assert O.rel_err(O.run(x, wt, plan), O.reference(x, O.dequant(wt))) <= O.OURS_BOUND
    assert O.run(_x(1, 0, 256), wt, plan).shape == (0, 64)


def test_w4_decode_is_exact_e2m1_times_scale(O):
    # codes 0..15 in one block, scale 2, global 4: e2m1 * 2 / 4, low nibble first
    wq = torch.tensor([[(2 * i) | ((2 * i + 1) << 4) for i in range(8)]], dtype=torch.uint8)
    wt = O.Weights.fp4(wq, torch.tensor([[2.0]]).to(torch.float8_e4m3fn), torch.tensor([4.0]))
    want = [v / 2 for v in R.E2M1] + [-v / 2 for v in R.E2M1]
    assert O.dequant(wt)[0].tolist() == want
    x = torch.eye(16).bfloat16()
    for plan in O.plans("w4", 1, 1, 16, 16, nsm=8):
        assert O.run(x, wt, plan)[:, 0].float().tolist() == want, plan.tag()


# --- planning and routing -------------------------------------------------------------
def test_plans_cover_k_without_empty_splits_and_respect_caps(O):
    for kind, n, k, m in (("w8", 2624, 6144, 6), ("w4", 6144, 12288, 96), ("w8", 512, 6144, 32),
                          ("w4", 6144, 256, 1), ("w8", 512, 192, 8)):
        ps = O.plans(kind, 1, n, k, m)
        assert ps
        for p in ps:
            kb = k // p.cfg.bk
            assert k % p.cfg.bk == 0 and p.kb_split * (p.n_split - 1) < kb <= p.kb_split * p.n_split
            assert p.ws(1, m, n) <= O.WS_CAP
            assert not p.cfg.a8 or kind == "w8"
        cfgs = {p.cfg for p in ps}
        assert any(not c.mma for c in cfgs) == (m <= 8)
        assert all(c.bm == (1 << (m - 1).bit_length()) for c in cfgs if not c.mma)
    assert {c.bk for c in O.configs("w8", 192, 6)} == {64}
    assert {c.bk for c in O.configs("w4", 256, 1) if not c.mma} == {256}
    assert O.default_plan("w8", 1, 2624, 6144, 6).cfg.mma


def test_decide_and_route_lookup(O):
    p = O.Plan(O.Cfg(True, 16, 64, 128, 4, 4), 4, 3)
    table = {1: {"ours": [(3.0, p)], "base": {"cutlass": 5.0, "marlin": 2.0}},
             8: {"ours": [(4.0, p)], "base": {"cutlass": 3.5}},
             32: {"ours": [], "base": {"cutlass": 6.0}}}
    d = O.decide(table, {"cutlass"})
    assert d == {1: ("ours", p), 8: ("cutlass", None), 32: ("cutlass", None)}
    assert O.decide(table, {"cutlass", "marlin"})[1] == ("marlin", None)
    with pytest.raises(RuntimeError, match="no working kernel"):
        O.decide({4: {"ours": [], "base": {"cutlass": 1.0}}})
    r = O.Route((1, 8, 32), [d[1], d[8], d[32]], ("cutlass", None))
    assert [r.lookup(m)[0] for m in (1, 2, 8, 9, 32, 33, 9000)] == \
        ["ours", "cutlass", "cutlass", "cutlass", "cutlass", "cutlass", "cutlass"]
    assert r.ws(1, 100) == 3 * 1 * 100 and r.paths() == {"ours", "cutlass"}
    assert "M<=1 mma16x64x128w4s4/k3 3.0us" in r.line(table) and r.line().endswith("M>32 cutlass")


def test_counters_bind_streams_until_they_run_out(O, monkeypatch):
    for name in ("_SEMS", "_FREE"):
        monkeypatch.setattr(O, name, {})
    dev = torch.device("cpu")
    assert O.sem_for(dev, 11) is None  # nothing before load
    O.ensure_sems(dev)
    s1 = O.sem_for(dev, 11)
    assert s1 is O.sem_for(dev, 11) and s1.numel() == O.SEM_SLOTS and not s1.any()
    O.ensure_sems(dev)  # idempotent: bindings survive
    assert O.sem_for(dev, 11) is s1
    got = {id(O.sem_for(dev, s)) for s in range(11, 11 + O.N_SEMS)}
    assert len(got) == O.N_SEMS and O.sem_for(dev, 99) is None


def test_cache_round_trip(O, tmp_path, monkeypatch):
    monkeypatch.setenv(O.CACHE_ENV, str(tmp_path / "c.json"))
    p = O.Plan(O.Cfg(False, 8, 32, 256), 6, 4, True)
    table = {6: {"ours": [(2.5, p)], "base": {"cutlass": 4.0}}}
    O.cache_put("k", table)
    assert O.cache_get("k") == table and O.cache_get("other") is None


def test_budget_env(O, monkeypatch):
    monkeypatch.delenv(O.BUDGET_ENV, raising=False)
    assert O.budget_s() == O.BUDGET_S
    monkeypatch.setenv(O.BUDGET_ENV, "-3")
    with pytest.raises(ValueError):
        O.budget_s()


# --- convert_model end to end over fake vLLM -----------------------------------------
class FakeOps:
    @staticmethod
    def cutlass_scaled_mm_supports_fp8(cap):
        return cap == 120

    @staticmethod
    def scaled_fp8_quant(x, use_per_token_if_dynamic=False):
        s = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s

    @staticmethod
    def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
        assert b.stride(0) == 1  # the [N, K] FP8 weight read transposed: one copy
        out = (a.float() * scale_a.view(-1, 1)) @ (b.float() * scale_b.view(1, -1))
        return (out if bias is None else out + bias.float()).to(out_dtype)

    @staticmethod
    def scaled_fp4_quant(w, g, is_sf_swizzled_layout=True):
        assert not is_sf_swizzled_layout and g.dtype == torch.float32
        wq, sf, _g = fp4_quant(w, g)
        return wq, sf


class LinearBase(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k).bfloat16() * 0.02, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return nn.functional.linear(x, layer.weight, bias)


class FakeMarlin:
    @staticmethod
    def pack(q, s):
        return dict(marlin_w=q.clone(), marlin_s=s.clone(), marlin_ws=None)

    @staticmethod
    def linear(x2, mod):
        w = mod.marlin_w.float() * mod.marlin_s.float()[:, None]
        return (x2.float() @ w.t()).to(x2.dtype)


def _model():
    torch.manual_seed(0)
    root = nn.Module()
    root.model = nn.Module()
    layers = nn.ModuleDict()
    for i in ("0", "1"):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.fused_qkv_a_proj = LinearBase(96, 512)
        layer.self_attn.o_proj = LinearBase(64, 256)
        layers[i] = layer
    mtp = nn.Module()
    mtp.eh_proj = nn.Linear(512, 64, bias=False).bfloat16()
    layers["78"] = mtp
    root.model.layers = layers
    return root


@pytest.fixture
def fake_vllm(O, monkeypatch):
    lin = types.ModuleType("vllm.model_executor.layers.linear")
    lin.LinearBase, lin.UnquantizedLinearMethod = LinearBase, UnquantizedLinearMethod
    lin.LinearMethodBase = type("LinearMethodBase", (), {})
    monkeypatch.setitem(sys.modules, lin.__name__, lin)
    for name, v in (("_OPS", FakeOps), ("_METHOD_CLS", None), ("_SELFTESTED", set()),
                    ("_LAYERS", {}), ("_ROUTES", {}), ("_OURS", O), ("_NO_SEM", set()),
                    ("_capability", lambda dev: (12, 0)),
                    ("marlin_fp8_pack", FakeMarlin.pack), ("marlin_linear", FakeMarlin.linear)):
        monkeypatch.setattr(R, name, v)
    for name in ("_SEMS", "_FREE", "_DUMMY"):
        monkeypatch.setattr(O, name, {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda dev=None: types.SimpleNamespace(cuda_stream=7))
    calls = []

    def tune_route(wt, dev, extra, big, log=print):
        """No timing on CPU: split gemv for M <= 8, split mma to 32, then big (or an
        unsplit tile); `marlin` wins M = 1 when offered. The serving verify runs."""
        kind = "w4" if wt.w4 else "w8"
        calls.append((kind, wt.n, wt.k, tuple(sorted(extra)), big))

        def pick(m, want):
            ps = O.plans(kind, wt.b, wt.n, wt.k, m, nsm=8)
            return next((p for p in ps if want(p)), ps[0])

        choices = []
        for m in O.BUCKETS:
            if m == 1 and "marlin" in extra:
                choices.append(("marlin", None))
            elif m <= 32:
                choices.append(("ours", pick(m, lambda p: p.n_split > 1 and p.cfg.mma == (m > 8))))
            else:
                choices.append((big, None) if big else ("ours", pick(m, lambda p: p.n_split == 1)))
        bigc = (big, None) if big else ("ours", O.plans(kind, 1, wt.n, wt.k, O.BIG_M, 8, True)[0])
        route = O.Route(O.BUCKETS, choices, bigc)
        O.verify(route, wt, dev)
        return route

    monkeypatch.setattr(O, "tune_route", tune_route)
    return calls


@pytest.mark.parametrize("be", ["ours", "best", "ours_fp4"])
def test_convert_routes_inside_the_custom_op(fake_vllm, monkeypatch, capsys, be, O):
    monkeypatch.setenv(R.BACKEND_ENV, be)
    root = _model()
    mods = dict(root.named_modules())
    w = {n: m.weight.data.clone() for n, m in mods.items() if hasattr(m, "weight")}
    done = R.convert_model(root)
    assert [d[0] for d in done] == ["model.layers.0.self_attn.fused_qkv_a_proj",
                                    "model.layers.0.self_attn.o_proj",
                                    "model.layers.1.self_attn.fused_qkv_a_proj",
                                    "model.layers.1.self_attn.o_proj",
                                    "model.layers.78.eh_proj"]
    big = None if be == "ours_fp4" else "cutlass"
    extra = (("cutlass", "marlin") if be == "best" else ("cutlass",)) if big else ()
    kind = "w4" if be == "ours_fp4" else "w8"
    assert fake_vllm == [(kind, n, k, extra, big)  # timed once per shape
                         for n, k in ((96, 512), (64, 256), (64, 512))]
    err = capsys.readouterr().err
    assert "converted 5 linears" in err and "split-K counters 8 streams" in err
    qb = R.FP4_QUANT_BOUND if be == "ours_fp4" else R.QUANT_BOUND
    for name, _n, _k in done:
        mod = mods[name]
        assert mod._sfx_fp8d == be and R._LAYERS[mod._sfx_fp8d_key] is mod
        assert mod.weight.dtype == (torch.uint8 if be == "ours_fp4" else torch.float8_e4m3fn)
        assert hasattr(mod, "marlin_w") == (be == "best")  # best: Marlin wins M = 1 here
        for m in (1, 6, 20, 70):
            x = torch.randn(2, m, w[name].shape[1]).bfloat16()  # 3-D input
            y = mod(x)
            assert y.shape == (2, m, w[name].shape[0]) and y.dtype == torch.bfloat16
            assert R.rel_frob(y, nn.functional.linear(x, w[name])) < qb
    assert type(mods["model.layers.78.eh_proj"]) is R.Fp8DenseNNLinear


def test_stream_without_counters_runs_the_same_plan_two_stage(fake_vllm, monkeypatch, O,
                                                               capsys):
    monkeypatch.setenv(R.BACKEND_ENV, "ours")
    root = _model()
    o = root.model.layers["0"].self_attn.fused_qkv_a_proj
    w = o.weight.data.clone()
    R.convert_model(root)
    path, plan = o._sfx_ours.lookup(6)
    assert path == "ours" and plan.n_split > 1
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda dev=None: types.SimpleNamespace(cuda_stream=99))
    while O.sem_for(torch.device("cpu"), len(O._SEMS) + 1000) is not None:
        pass  # every other counter array goes to some other stream
    seen = []
    real = O.run
    monkeypatch.setattr(O, "run", lambda x, wt, p, sem=None: seen.append((p, sem)) or
                        real(x, wt, p, sem))
    x = torch.randn(6, 512).bfloat16()
    assert R.rel_frob(o(x), nn.functional.linear(x, w)) < R.QUANT_BOUND
    assert seen[-1] == (plan.two_stage(), None)
    assert "stream 0x63 found no free split-K counters" in capsys.readouterr().err


def test_oracle_fails_closed_on_a_wrong_kernel(fake_vllm, monkeypatch, O):
    monkeypatch.setenv(R.BACKEND_ENV, "ours")
    real = O.run
    monkeypatch.setattr(O, "run", lambda *a, **k: real(*a, **k) * 1.01)
    with pytest.raises(RuntimeError, match="(oracle FAILED|routed plan).*rel err"):
        R.convert_model(_model())


def test_backend_env_accepts_the_routed_backends(monkeypatch):
    for be in ("ours", "best", "ours_fp4"):
        monkeypatch.setenv(R.BACKEND_ENV, be)
        assert R.backend() == be and be in R.ROUTED


def test_gate_is_allowlisted_and_reads_the_tuner_tables(O):
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    assert ns["_BOOT_GATES"]["glm_dense_ours_bench"] == (
        ["-m", "suffix_hybrid.tools.glm_dense_ours"], {})
    from suffix_hybrid.tools import glm_dense_ours as G

    a = O.Plan(O.Cfg(True, 16, 64, 128), 4, 2)
    b = O.Plan(O.Cfg(True, 16, 64, 128, a8=True), 4, 2)
    t8 = {6: {"ours": [(2.0, b), (3.0, a)], "base": {"cutlass": 4.0, "bf16": 9.0}}}
    t4 = {6: {"ours": [(1.5, a)], "base": {}}}
    cols = G.columns(t8, t4, 6)
    assert cols == {"cutlass": (4.0, None), "bf16": (9.0, None), "ours8": (3.0, a),
                    "ours8a8": (2.0, b), "ours4": (1.5, a)}
    assert G.route_us(t8, 6, ("ours", b)) == 2.0 and G.route_us(t8, 6, ("cutlass", None)) == 4.0
    assert {s[0] for s in G.SHAPES} >= {"qkv_a", "q_b", "o_proj", "mtp.eh_proj"}


@pytest.mark.parametrize("be", ["ours", "best", "ours_fp4"])
def test_load_time_code_under_vllm_bf16_default_dtype(fake_vllm, monkeypatch, be, O):
    """vLLM runs process_weights_after_loading under torch.set_default_dtype(bfloat16):
    quantization, routing, verify and the oracle must keep fp32 where they need it."""
    monkeypatch.setenv(R.BACKEND_ENV, be)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        root = _model()
        done = R.convert_model(root)
        mods = dict(root.named_modules())
        for name, _n, k in done:
            wt = mods[name]._sfx_ours_w
            assert wt.s.dtype == torch.float32 and O.dequant(wt).dtype == torch.float32
            x = torch.randn(6, k, dtype=torch.bfloat16)
            path, plan = mods[name]._sfx_ours.lookup(6)
            assert O.reference(x, O.dequant(wt), plan.cfg if plan else None).dtype == torch.float32
            assert mods[name](x).dtype == torch.bfloat16
    finally:
        torch.set_default_dtype(prev)
