# SPDX-License-Identifier: Apache-2.0
"""Helion MXFP8 GEMM (DeepSeek-V4.1 FP8 linears on SM120): lab cases, spec
quant / dequant, serving plan + knob, the lane's selection / o_proj wrappers,
the K5 capture/replay machinery, and (with Helion installed) both bodies in
interpret mode vs the exact reference and lowered to Triton source by Helion's
real front end (no semaphore in the two-stage default)."""
import json
import os
import subprocess
import sys

os.environ["HELION_INTERPRET"] = "1"  # before any helion.kernel is built

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import mxfp8_gemm as hx  # noqa: E402
from suffix_hybrid.kernels import mxfp8_linear as lane  # noqa: E402

DSV41 = {"name": "deepseek-v4.1-flash-1017-nvfp4", "tp": 1, "quant_format": "nvfp4"}
K = {e["name"]: e for e in hx.KERNELS}


def test_kernels_abi_and_cases():
    assert list(K) == [hx.A16, hx.A8]
    for e in hx.KERNELS:
        assert e["body"] is hx.BODIES[e["name"]] and callable(e["check"]) and e["replaces"] == []
        assert "ignore_warnings" in e["settings"]  # the torch reduce is host code on purpose
        cases = e["cases"](DSV41)
        assert 0 < len(cases) <= 64 and len({hk.case_key(c) for c in cases}) == len(cases)
        for c in cases:
            assert c["numtokens"] in hx.BUCKETS and json.loads(hk.case_key(c)) == c
            assert c["k"] % 32 == 0 and set(c) <= {"g", "k", "n", "numtokens"}
        assert e["cases"]({"name": "gemma-4-26b-a4b-nvfp4"}) == []
    a16, a8 = K[hx.A16]["cases"](DSV41), K[hx.A8]["cases"](DSV41)
    # TP8 decode first: the 6-token DSpark step (bucket 8), wo_a (the known loser) leading
    assert a16[0] == {"g": 1, "k": 4096, "n": 1024, "numtokens": 8}
    assert all("g" not in c for c in a8)  # wo_a's vLLM path is bf16 x bf16: A16 only
    tp8 = {(c["n"], c["k"]) for c in a16[:28]}
    assert tp8 == {(1024, 4096), (1792, 5120), (5120, 15360), (4096, 1280), (5120, 1024),
                   (576, 5120), (5120, 288)}
    assert {(c["n"], c["k"]) for c in a16[28:]} == {(8192, 1280), (5120, 2048), (1152, 5120),
                                                    (5120, 576)}


def test_shapes_follow_the_tp_split():
    s8, s4 = hx.dsv41_shapes(8), hx.dsv41_shapes(4)
    assert s8["shared.gate_up"] == (576, 5120) and s8["shared.down"] == (5120, 288)
    assert s4["shared.gate_up"] == (1152, 5120) and s4["wo_b"] == (5120, 2048)
    assert s8["wq_b"] == s8["indexer.wq_b"] == (4096, 1280)
    assert all(k % 32 == 0 and n % 32 == 0 for n, k in s8.values())


def test_quant_spec_is_vllm_mxfp8():
    x = torch.randn(5, 96) * torch.logspace(-3, 3, 96)
    x[1] = 0.0
    x[2, :32] = 448.0  # amax / 448 an exact power of two: ceil(log2) must not round up
    q, s = hx.quant_spec(x.bfloat16())
    xb = x.bfloat16().float().view(5, 3, 32)
    sb = (torch.ceil(torch.log2(xb.abs().amax(-1).clamp(min=torch.finfo(torch.float32).tiny)
                                / 448.0)) + 127).clamp(0, 254)
    assert torch.equal(s, sb.to(torch.uint8)) and int(s[2, 0]) == 127 and int(s[1, 0]) == 0
    back = q.float().view(5, 3, 32) * torch.exp2(s.float() - 127)[:, :, None]
    assert (back - xb).abs().max() <= xb.abs().amax() * 2 ** -4  # e4m3: 3 mantissa bits
    assert q.float().abs().max() <= 448


def test_dequant_and_exact():
    w = (torch.randn(64, 64) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    ws = torch.randint(118, 124, (64, 2), dtype=torch.int32).to(torch.uint8)
    d = hx.dequant(w, ws)
    assert torch.equal(d[3, 33:40], w[3, 33:40].double() * 2.0 ** (int(ws[3, 1]) - 127))
    x = torch.randn(3, 64).bfloat16()
    assert torch.allclose(hx.exact(x, w, ws), x.double() @ d.T)
    ok, msg = hx.check(hx.exact(x, w, ws).bfloat16(), hx.exact(x, w, ws))
    assert ok, msg
    bad = hx.exact(x, w, ws).clone()
    bad[:, 7] += 1.0
    assert not hx.check(bad, hx.exact(x, w, ws))[0]


def test_e4m3_bit_decode_matches_torch():
    """The A16 body's dequant: e4m3 bits -> f16 bits (x 2^-8) -> * 2^(e - 119)."""
    b = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    b = b[(b & 127) != 127]  # NaN codes never occur in a checkpoint
    wb = b.to(torch.int16)
    hw = (((wb & 127) << 7) | ((wb & 128) << 8)).view(torch.float16)
    for e in (100, 120, 127, 140):
        sb = torch.tensor([(e + 8) << 23], dtype=torch.int32).view(torch.float32)
        got = (hw.float() * sb).bfloat16()
        want = (b.view(torch.float8_e4m3fn).float() * 2.0 ** (e - 127)).bfloat16()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), e


def test_a8_scale_bits_match_spec():
    """The A8 body's scale from f32 bits == ceil(log2(amax / 448)) + 127."""
    amax = torch.tensor([448.0, 449.0, 1.0, 3e-3, 7.5e5, 448.0 * 2 ** -20])
    yb = (amax / 448.0).view(torch.int32)
    s = torch.clamp((yb >> 23) + ((yb & 8388607) != 0).to(torch.int32), 0, 254)
    want = torch.ceil(torch.log2((amax / 448.0).double())) + 127
    assert torch.equal(s, want.to(torch.int32))
    inv = ((254 - s) << 23).view(torch.float32)
    assert torch.equal(inv.double(), torch.exp2(127.0 - want))


def test_validate_weight_fails_closed():
    w = torch.zeros(32, 64).to(torch.float8_e4m3fn)
    ws = torch.full((32, 2), 123, dtype=torch.uint8)
    hx.validate_weight(w, ws)  # ok
    hx.validate_weight(w, ws.fill_(hx.MAX_SCALE))  # the edge still decodes
    with pytest.raises(RuntimeError, match="e8m0 scale 247"):
        hx.validate_weight(w, ws.fill_(hx.MAX_SCALE + 1), "layers.3.attn.wo_a")
    bad = w.view(torch.uint8).clone()
    bad[5, 7] = 0xFF  # S.1111.111 = e4m3fn NaN
    with pytest.raises(RuntimeError, match="NaN code"):
        hx.validate_weight(bad.view(torch.float8_e4m3fn), ws.fill_(123))
    # the largest legal scale really decodes: 2^(246 - 127) * 448 stays finite in bf16
    top = hx.dequant(torch.full((1, 32), 448.0).to(torch.float8_e4m3fn),
                     torch.full((1, 1), hx.MAX_SCALE, dtype=torch.uint8))
    assert torch.isfinite(top.float()).all() and float(top.max()) < 3.0e38


def test_max_splits_warm_ms_describe():
    assert [hx.max_splits(n) for n in (576, 1024, 1792, 4096, 5120)] == [64, 32, 16, 8, 8]
    assert hx.warm_ms(192) == [1, 7, 8, 31, 32, 63, 64, 191, 192]
    assert hx.warm_ms(8) == [1, 7, 8]
    assert hx.warm_ms(40) == [1, 7, 8, 31, 32, 39, 40]
    plan = [None] + [("mxfp8_gemm", None, 4, 8)] * 8 + [None] * 24
    assert hx.describe(plan) == "M1-8:mxfp8_gemm(split<=4) M9-32:vllm"


@pytest.fixture
def shipped(monkeypatch):
    table = {}
    monkeypatch.setattr(hk, "config", lambda name, case, plat=None: table.get(
        (name, case["numtokens"], "g" in case)))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    monkeypatch.setattr(hk, "compiled", lambda body, name, case, **kw: (name, case["numtokens"]))
    return table


def test_prepare_auto_prefers_a16(shipped):
    shipped.update({(hx.A16, 8, False): {"split_k": 4}, (hx.A8, 8, False): {},
                    (hx.A8, 192, False): {"split_k": 2}})
    plan = hx.prepare(1792, 5120, 192, "auto")
    assert len(plan) == 193 and plan[0] is None
    assert plan[1][0] == plan[8][0] == hx.A16 and plan[8][2:] == (4, 8)
    assert all(plan[m] is None for m in range(9, 65))
    assert plan[65][0] == plan[192][0] == hx.A8 and plan[192][2:] == (2, 192)


def test_no_module_level_scratch_or_counters():
    """K5 / capture-safety regression guard: nothing sized lazily, shared between layers or
    streams, or bound at first use may come back into the serving path."""
    for gone in ("_SCRATCH", "_NEED", "_scratch"):
        assert not hasattr(hx, gone), gone
    for body in (hx.mxfp8_gemm_body, hx.mxfp8_gemm_a8_body):
        assert body.__code__.co_varnames[:body.__code__.co_argcount] == ("x", "w", "ws", "out", "partial")
    for body in (hx.mxfp8_gemm_sem_body, hx.mxfp8_gemm_a8_sem_body):
        assert body.__code__.co_varnames[:body.__code__.co_argcount][-1] == "counters"
    assert set(hx.BODIES) == {hx.A16, hx.A8} and set(hx.SEM_BODIES) == {hx.A16_SEM, hx.A8_SEM}


def test_prepare_and_register_refuse_the_semaphore_variant(shipped, monkeypatch):
    shipped.update({(hx.A16, 8, False): {"split_k": 4}})
    monkeypatch.setenv(hx.SEM_ENV, "1")
    with pytest.raises(RuntimeError, match="UNPROVEN under CUDA-graph capture"):
        hx.prepare(1792, 5120, 8, "auto")
    monkeypatch.setenv(lane.ENV, "auto")
    with pytest.raises(RuntimeError, match="never serves it"):
        lane.register()
    monkeypatch.setenv(hx.SEM_ENV, "2")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        hx.semaphore_on()
    monkeypatch.delenv(hx.SEM_ENV)
    assert not hx.semaphore_on() and hx.prepare(1792, 5120, 8, "auto")[8][0] == hx.A16


def test_semaphore_bodies_are_listed_only_under_the_knob():
    code = ("import sys; sys.path.insert(0, '.'); "
            "from suffix_hybrid.helion_kernels import mxfp8_gemm as hx; "
            "print([k['name'] for k in hx.KERNELS])")
    # No SUFFIX_* from other tests (a leaked SUFFIX_HYBRID_WRAP=1 arms sitecustomize's wrap).
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env["HELION_INTERPRET"] = "1"
    base = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert base.stdout.strip() == str([hx.A16, hx.A8]), base.stderr[-400:]
    on = subprocess.run([sys.executable, "-c", code], env={**env, hx.SEM_ENV: "1"},
                        capture_output=True, text=True)
    assert on.stdout.strip() == str([hx.A16, hx.A8, hx.A16_SEM, hx.A8_SEM]), on.stderr[-400:]


def test_prepare_bmm_is_a16_only_and_helion_fails_closed(shipped):
    shipped.update({(hx.A8, b, True): {} for b in hx.BUCKETS})
    assert not any(hx.prepare(1024, 4096, 192, "auto", bmm=True)[1:])
    shipped.update({(hx.A16, b, True): {} for b in (8, 32)})
    assert hx.prepare(1024, 4096, 32, "helion", bmm=True)[32][0] == hx.A16
    with pytest.raises(RuntimeError, match="bucket 64 on gpu"):
        hx.prepare(1024, 4096, 64, "helion", bmm=True)
    with pytest.raises(RuntimeError):
        hx.prepare(1024, 4096, 64, "oxide")


def test_lane_knob_and_max_m(monkeypatch):
    monkeypatch.delenv(lane.ENV, raising=False)
    assert lane.mode() == "oxide" and lane.register() is None  # off: nothing imported
    monkeypatch.setenv(lane.ENV, "nope")
    with pytest.raises(RuntimeError):
        lane.mode()
    monkeypatch.setenv(lane.MAX_M_ENV, "193")
    with pytest.raises(ValueError):
        lane.max_m_env()
    monkeypatch.setenv(lane.MAX_M_ENV, "64")
    assert lane.max_m_env() == 64


def test_selection_wrappers():
    class Stock:
        def __init__(self, config=None):
            self.config = config

    class Ours(Stock):
        pass

    class Layer:
        prefix = "model.layers.3.attn.wq_b"

    sel = lane._wrap_select(lambda spec, layer, rt, weight_shape=None: Stock("cfg"), Ours, Stock)
    assert type(sel(None, Layer(), None)) is Ours and sel(None, Layer(), None).config == "cfg"
    eng = Layer()
    eng.prefix = "model.layers.1.engram.wkv"
    assert type(sel(None, eng, None)) is Stock
    init = lane._wrap_init(lambda *, weight_shape, bmm_batch_size=None: Stock(bmm_batch_size),
                           Ours, Stock)
    assert type(init(weight_shape=(1, 1), bmm_batch_size=1)) is Ours
    assert type(init(weight_shape=(1, 1))) is Stock  # plain linears go through select


def test_o_proj_wrapper_passes_through_without_a_plan():
    calls = []
    wrapped = lane._wrap_o_proj(lambda *a, **kw: calls.append((a, kw)) or "orig", None)

    class WoA:
        pass

    assert wrapped("o", "pos", "cache", WoA(), "wo_b", n_groups=1) == "orig" and calls


# ---------------------------------------------------------------------------
# interpret mode (Helion installed, e.g. CI): bodies vs exact reference
# ---------------------------------------------------------------------------
CASES = [{"k": 96, "n": 48, "numtokens": 3},     # N tail at bn 32, K = 3 blocks
         {"k": 288, "n": 32, "numtokens": 17},   # shared.down's K (9 blocks), 2 m16 rows
         {"k": 64, "n": 16, "numtokens": 1}]


@pytest.fixture
def cpu_mm(monkeypatch):
    mm = torch.mm  # CPU aten::mm has no out_dtype (hl.dot's eager reference uses it)
    monkeypatch.setattr(torch, "mm", lambda a, b, *, out_dtype=None, **kw: (
        mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)))


def _ref(name):
    return hx.reference if name in (hx.A16, hx.A16_SEM) else hx.reference_a8


@pytest.mark.parametrize("name", list(hx.ALL_BODIES))
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"m{c['numtokens']}n{c['n']}k{c['k']}")
def test_body_interpret_vs_reference(name, case, cpu_mm):
    helion = pytest.importorskip("helion")
    sem = name in hx.SEM_BODIES
    p = hx.problem(case, 0, "cpu", sem=sem)
    args = hx.body_args(name, p)
    ref = _ref(name)(*args)
    for split in (1, 2, 4):
        kern = helion.kernel(hx.ALL_BODIES[name], config=helion.Config(split_k=split))
        for _ in range(2):  # a replay on the same buffers
            ok, msg = hx.check(kern(*args), ref)
            assert ok, f"{name} split_k={split}: {msg}"
            if sem:
                assert not p["counters"].any(), "split-K counters not reset"


@pytest.mark.parametrize("name", list(hx.BODIES))
def test_two_stage_is_deterministic_and_leaves_partial_alone_when_unsplit(name, cpu_mm):
    helion = pytest.importorskip("helion")
    p = hx.problem({"k": 288, "n": 32, "numtokens": 17}, 0, "cpu")
    args = hx.body_args(name, p)
    p["partial"].fill_(float("nan"))
    one = helion.kernel(hx.BODIES[name], config=helion.Config(split_k=1))
    one(*args)
    assert torch.isnan(p["partial"]).all(), "split_k=1 must not touch (or read) the partial"
    two = helion.kernel(hx.BODIES[name], config=helion.Config(split_k=4))
    y2 = two(*args).clone()
    assert hx.bits_equal(y2, two(*args))  # run-to-run bit-exact: fixed-order reduce, no atomics
    assert torch.isfinite(p["partial"][:3]).all()  # nkb = 9 -> kb_split 4 -> 3 splits written


def _adversarial(case, pattern, sem=False):
    """problem() with hostile scale / activation patterns (K3): per-block extremes, a single outlier
    block, subnormal-scale blocks, and activations spanning 60 binades across blocks."""
    p = hx.problem(case, 1, "cpu", sem=sem)
    n, k = p["n"], p["k"]
    nkb = k // 32
    if pattern == "alternating_extremes":  # 2^-122 / 2^3 blocks side by side in every row
        ws = torch.where(torch.arange(nkb) % 2 == 0, 5, 130).expand(n, nkb)
    elif pattern == "one_hot_block":  # a single huge-scale block per row, rest at the floor
        ws = torch.full((n, nkb), 1)
        ws[torch.arange(n), torch.arange(n) % nkb] = 140
    else:  # "binades": scales walk through 60 binades along K
        ws = (60 + torch.arange(nkb) * 2 % 120).expand(n, nkb)
    p["ws"] = ws.to(torch.uint8).contiguous()
    if pattern == "binades":
        p["x"] = (p["x"].float() * torch.exp2(torch.arange(nkb).repeat_interleave(32) % 60 - 30.0)
                  ).bfloat16()
    return p


@pytest.mark.parametrize("name", list(hx.ALL_BODIES))
@pytest.mark.parametrize("pattern", ["alternating_extremes", "one_hot_block", "binades"])
def test_body_interpret_adversarial_scales(name, pattern, cpu_mm):
    helion = pytest.importorskip("helion")
    sem = name in hx.SEM_BODIES
    p = _adversarial({"k": 160, "n": 40, "numtokens": 5}, pattern, sem)
    hx.validate_weight(p["w"], p["ws"])
    args = hx.body_args(name, p)
    ref = _ref(name)(*args)
    assert torch.isfinite(ref).all()
    for split in (1, 2, 4):
        kern = helion.kernel(hx.ALL_BODIES[name], config=helion.Config(split_k=split))
        ok, msg = hx.check(kern(*args), ref)
        assert ok, f"{pattern} split_k={split}: {msg}"
        if sem:
            assert not p["counters"].any()


# ---------------------------------------------------------------------------
# Helion's real compiler front end: Triton source. The two-stage default must contain no
# atomic, no barrier, no counter; the semaphore variant must (so the test tells them apart).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("name", list(hx.ALL_BODIES))
def test_triton_codegen_semaphore_free_by_default(name, static):
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    sem = name in hx.SEM_BODIES
    p = hx.problem({"k": 1280, "n": 256, "numtokens": 8}, 0, "cpu", sem=sem)
    bound = helion.kernel(hx.ALL_BODIES[name], static_shapes=static, ref_mode=RefMode.OFF,
                          ignore_warnings=hx._ignore()).bind(hx.body_args(name, p))
    for split in (1, 2, 4):
        code = bound.to_triton_code(helion.Config(split_k=split, block_sizes=[16, 32, 2]))
        assert ("tl.dot_scaled" in code) == (name in (hx.A8, hx.A8_SEM)), name
        if sem:
            assert "tl.atomic_add" in code and "acq_rel" in code and "counters" in code
        else:
            assert not any(t in code for t in ("atomic", "debug_barrier", "counters", "semaphore",
                                               "cooperative", "grid_sync")), code
            # the reduce is plain host code after the one kernel launch, in stream order
            assert code.count("_launcher(") == 1
            assert "out.copy_(partial[:n_split].sum(0))" in code
            assert code.index("_launcher(") < code.index("out.copy_(partial[:n_split]")


# ---------------------------------------------------------------------------
# apply / gemm / capture_check on CPU (EagerRuntime stands in for torch.cuda graphs)
# ---------------------------------------------------------------------------
def _entry_for(name, split):
    import helion
    kern = helion.kernel(hx.BODIES[name], config=helion.Config(split_k=split),
                         static_shapes=False, ignore_warnings=hx._ignore())
    return name, kern, split, 8


def _layer(n, k, groups, bmm, m, seed=0):
    ps = [hx.problem({"k": k, "n": n, "numtokens": m}, seed + i, "cpu") for i in range(groups)]
    w, ws = torch.cat([q["w"] for q in ps]), torch.cat([q["ws"] for q in ps])
    gen = torch.Generator().manual_seed(5 + seed)
    x = torch.randn(*((m, groups, k) if bmm else (m, k)), generator=gen).bfloat16()
    return w, ws, x


@pytest.mark.parametrize("name", [hx.A16, hx.A8])
@pytest.mark.parametrize("split", [1, 2, 4])
def test_apply_two_stage_plain_and_grouped(name, split, cpu_mm):
    pytest.importorskip("helion")
    groups = 1 if name == hx.A8 else 2
    w, ws, x = _layer(32, 288, groups, bmm=name == hx.A16, m=5)
    entry = _entry_for(name, split)
    out = hx.apply(x, w, ws, entry, 32, groups, name == hx.A16, poison_nan=True)
    assert out.shape == (5, groups * 32) and torch.isfinite(out.float()).all()
    for i in range(groups):
        xi = x[:, i, :] if name == hx.A16 else x
        if name == hx.A8:
            q, s = hx.quant_spec(xi)
            ref = hx.exact(hx.dequant(q, s, torch.float32), w[i * 32:(i + 1) * 32], ws[i * 32:(i + 1) * 32])
        else:
            ref = hx.exact(xi, w[i * 32:(i + 1) * 32], ws[i * 32:(i + 1) * 32])
        ok, msg = hx.check(out[:, i * 32:(i + 1) * 32], ref)
        assert ok, msg
    # a caller-provided strided out (row stride free) takes the same bits
    big = torch.zeros(5, groups * 32 + 7, dtype=torch.bfloat16)
    again = hx.apply(x, w, ws, entry, 32, groups, name == hx.A16, out=big[:, 3:3 + groups * 32])
    assert hx.bits_equal(again, out)


def test_poison_nan_exposes_an_unwritten_partial(cpu_mm):
    """A kernel that forgets a split must fail the exact check instead of reading stale bytes."""
    pytest.importorskip("helion")
    w, ws, x = _layer(32, 288, 1, False, 5)
    name, kern, split, b = _entry_for(hx.A16, 4)
    good = hx.apply(x, w, ws, (name, kern, split, b), 32, poison_nan=True)
    assert hx.check(good, hx.exact(x, w, ws))[0]

    def forgets_a_split(*args):
        y = kern(*args)
        args[4][1].fill_(float("nan"))  # split 1 never written = still the poison...
        args[3].copy_(args[4][:3].sum(0))  # ...and the reduce reads it
        return y
    bad = hx.apply(x, w, ws, (name, forgets_a_split, split, b), 32, poison_nan=True)
    assert not hx.check(bad, hx.exact(x, w, ws))[0]


def test_capture_check_passes_and_bites(cpu_mm):
    pytest.importorskip("helion")
    w, ws, x = _layer(32, 288, 1, False, 5)
    xs = [x, (x.float() * 0.5).bfloat16(), (x.float() * -2).bfloat16()]
    entry = _entry_for(hx.A16, 2)
    run = lambda xx: hx.apply(xx, w, ws, entry, 32, poison_nan=True)
    assert hx.capture_check(run, xs, hx.EagerRuntime, replays=7) == 7

    # state bound at first use / replay-variant results: the 4th call differs -> must be caught
    calls = []

    def drift(xx):
        out = run(xx)
        calls.append(1)
        if len(calls) > 4:
            out = out.clone()
            out[0, 0] = out[0, 0] + 1.0
        return out
    with pytest.raises(RuntimeError, match=r"CAPTURE/REPLAY FAIL N=32.*replay \d \(input \d\) differs"):
        hx.capture_check(drift, xs, hx.EagerRuntime, replays=4, what="N=32")

    # a replay that ignores the NEW data (stale pointer / baked-in input) is caught too
    class Stale(hx.EagerRuntime):
        @staticmethod
        def capture(fn):
            y = fn()
            return y, (lambda: None)
    with pytest.raises(RuntimeError, match="differs from the eager result"):
        hx.capture_check(run, xs, Stale)


# ---------------------------------------------------------------------------
# lab_main on CPU: the case loop, the composite check, the JSON records (GPU parts stubbed)
# ---------------------------------------------------------------------------
@pytest.fixture
def lab_cpu(monkeypatch, cpu_mm):
    import types
    pytest.importorskip("helion")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i=0: types.SimpleNamespace(L2_cache_size=1 << 20))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "fake gpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    real = hx.problem
    monkeypatch.setattr(hx, "problem", lambda case, seed, device="cpu", sem=False: real(case, seed, "cpu", sem))
    monkeypatch.setattr(hx, "_time_us", lambda fn, sets, iters=50: (2.0, "graph"))
    monkeypatch.setattr(hx, "_baseline_args", lambda case, seed: ("x", type("FakeKern", (), {})(), None))

    def no_marlin(case, seed):
        raise ImportError("no vllm here")
    monkeypatch.setattr(hx, "_marlin_args", no_marlin)
    monkeypatch.setattr(hk, "platform", lambda device=0: "fake_gpu")
    monkeypatch.delenv("HELION_AUTOTUNER", raising=False)


def _records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_lab_main_times_the_composite_and_skips_wo_a_for_a8(lab_cpu, capsys):
    cases = [{"k": 96, "n": 48, "numtokens": 3}, {"k": 288, "n": 32, "numtokens": 9, "g": 1}]
    hx.lab_main({"kernels": [hx.A16, hx.A8], "cases": cases, "effort": "none",
                 "config": {"split_k": 2, "block_sizes": [16, 32, 2]}})
    recs = _records(capsys)
    assert recs[0]["gpu"] == "fake gpu" and recs[0]["platform"] == "fake_gpu"
    got = [(r["kernel"], "g" in r["case"]) for r in recs[1:]]
    assert got == [(hx.A16, False), (hx.A16, True), (hx.A8, False)]  # wo_a is A16 only
    for r in recs[1:]:
        assert r["correct"] is True and "error" not in r, r
        assert r["speedup"] == 1.0 and r["timing"] == "graph" and r["capture_fail"] is False
        assert "marlin_error" in r and r["config"]["split_k"] == 2


def test_lab_main_flags_a_kernel_that_cannot_be_captured(lab_cpu, capsys, monkeypatch):
    monkeypatch.setattr(hx, "_time_us", lambda fn, sets, iters=50: (2.0, "eager"))
    hx.lab_main({"kernels": [hx.A16], "cases": [{"k": 96, "n": 48, "numtokens": 3}], "effort": "none"})
    rec = _records(capsys)[1]
    assert rec["capture_fail"] is True  # never a timing detail: the ship rule drops the row


def test_lab_main_semaphore_bodies_need_the_knob(lab_cpu, capsys, monkeypatch):
    monkeypatch.delenv(hx.SEM_ENV, raising=False)
    with pytest.raises(RuntimeError, match="UNPROVEN under CUDA-graph capture|unproven under"):
        hx.lab_main({"kernels": [hx.A16_SEM], "cases": []})
    with pytest.raises(ValueError, match="unknown kernels"):
        hx.lab_main({"kernels": ["nope"]})
    monkeypatch.setenv(hx.SEM_ENV, "1")
    hx.lab_main({"kernels": [hx.A16_SEM], "cases": [{"k": 96, "n": 48, "numtokens": 3}],
                 "effort": "none", "config": {"split_k": 2}})
    rec = _records(capsys)[1]
    assert rec["kernel"] == hx.A16_SEM and rec["correct"] is True
