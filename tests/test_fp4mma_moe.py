# SPDX-License-Identifier: Apache-2.0
"""The V4.1 routed-experts dot_scaled VARIANT (dsv41_moe_fp4mma.py) on CPU: (1) the bodies in Helion's
eager mode against lane B's f64 spec (kernels/nvfp4_moe.moe_ref: quantize x and h to NVFP4, swiglu clamp
`gate = min(gate, L)` / `up = clamp(up, -L, L)` before silu, f32 accumulate), balanced / skewed / hot /
off-rank expert loads, adversarial scale blocks, forced small tiles, BOTH scale-read modes; (2) the
contract of lane B's per-bucket seam (`nvfp4_moe.check_variant`) and the seam's own end-to-end path
(config-table row -> served_ms -> serving_op -> this module's run_plan) against lane B's dequant bodies at the
oracle tolerance, with the refusals; (3) the K5 hook inside run_plan, the lab ABI and the capture gate.

CPU eager mode proves indexing and numerics. Speed, registers and the native instruction are K4/K5 (silicon).
"""
import ast
import importlib.abc
import inspect
import sys
import types

import pytest
import torch

import suffix_hybrid
from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
from suffix_hybrid.helion_kernels import nvfp4_moe as lb
from suffix_hybrid.kernels import fp4mma_probe as probe
from suffix_hybrid.kernels import nvfp4_moe as nm

from test_fp4mma_probe import DECOMPOSED_PTX, NATIVE_PTX, FakeCK, FakeJit, fake_launcher  # noqa: E402,F401
from test_fp4mma_ref import cpu_mm, nvfp4_ref  # noqa: E402,F401

try:
    import helion
except ImportError:
    helion = None
needs_helion = pytest.mark.skipif(helion is None, reason="Helion not installed")
WARN = [helion.exc.TensorOperationInWrapper] if helion else []

DS = {"name": "deepseek-v4.1-flash-1017-nvfp4", "architecture": "DeepseekV41ForCausalLM",
      "tp": 8, "hidden": 5120, "moe_intermediate": 288, "experts": 384, "topk": 6, "quant_format": "nvfp4"}


def kernels(plan="front4", cfgs=None):
    """The plan's kernels as the seam builds them (ours from BODIES, lane B's combine from lane B's)."""
    cfgs = cfgs or {}
    return [helion.kernel(fe.BODIES.get(k) or lb.BODIES[k], config=helion.Config(**cfgs.get(k, {})),
                          static_shapes=False, ignore_warnings=WARN) for k in fe.PLANS[plan]]


def layer(e_local=6, e_global=12, base=3, h=128, i=128, topk=3, m=7, dist="balanced", seed=0, limit=10.0,
          gain=1.0, adversarial=False, dead=None):
    p = nm.make_problem(e_local, h, i, "silu", "cpu", seed, limit=limit, adversarial=adversarial)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed + 5, base, e_local, dist=dist, dead=dead)
    gen = torch.Generator().manual_seed(seed + 9)
    x = (torch.randn(m, h, generator=gen) * 2688.0 / p["a1g"] / 4.5 * gain).bfloat16()
    return p, x, ids, tw, base


def run(p, x, ids, tw, base, cfgs=None, plan="front4"):
    out = torch.full((x.shape[0], x.shape[1]), float("nan")).bfloat16()
    fe.run_plan(kernels(plan, cfgs), plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                p["g2"], out, p["a1g"], p["a2g"], p["act"], base, p["limit"])
    return out


def rel_vs_spec(out, p, x, ids, tw, base):
    lid = nm.to_local(ids, base, p["w13"].shape[0]).int()
    ref = nm.moe_ref(p, x, lid, tw)
    return float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30)), lid


@pytest.fixture(autouse=True)
def _leave_the_variant_cache_as_found():
    """lane B caches the imported variant module in `_VARIANTS`: no test of ours may leak that into another file's."""
    before = dict(lb._VARIANTS)
    yield
    lb._VARIANTS.clear()
    lb._VARIANTS.update(before)


@pytest.fixture(params=["words", "bytes"])
def scale_read(request, monkeypatch):
    """Both scale-read modes of fc1 / fc2 (docs/dsv41-fp4.md section 5): the same numbers either way."""
    monkeypatch.setattr(fe, "SCALE_READ", request.param)
    return request.param


@pytest.fixture
def small_tiles(monkeypatch):
    def set_(brow=16, bcol=2, bh=32, bk=32, bp=16, bq=16):
        for name, v in (("BROW_MAX", brow), ("BCOL_MAX", bcol), ("BH_MAX", bh), ("BK_MAX", bk),
                        ("BP_MAX", bp), ("BQ_MAX", bq)):
            monkeypatch.setattr(fe, name, v)
    return set_


# ---------------------------------------------------------------------------
# the route: the python spec == lane B's sort; the kernel route == the spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 5, 16, 33])
def test_route_ref_is_lane_bs_sort_in_expert_major_form(m):
    ids, _ = nm.rand_routing(m, 12, 3, "cpu", m, 3, 6, dist="skewed")
    spec = fe.route_ref(ids, 6, 3)
    lps, lseg, lseg_e = lb.route_ref(ids, 6, 3)  # lane B's per-position arrays, rebuilt from (counts, starts)
    seg, seg_e = fe.seg_from(*spec)
    assert torch.equal(spec[0], lps) and torch.equal(seg, lseg) and torch.equal(seg_e, lseg_e)
    n_local = int(spec[1].sum())  # local pairs first, ascending expert, ascending pair
    ek = (ids.reshape(-1).long() - 3)[spec[0][:n_local].long()]
    assert bool((ek[1:] >= ek[:-1]).all()) and bool((0 <= ek).all()) and bool((ek < 6).all())
    assert int(spec[2][0]) == 0 and torch.equal(spec[2], torch.cumsum(spec[1], 0) - spec[1])


@needs_helion
@pytest.mark.parametrize("m", [1, 7, 20])
def test_route_quant_kernel_vs_spec_and_quant_codes(m, nvfp4_ref, small_tiles):
    small_tiles(bp=16, bq=16)
    p, x, ids, tw, base = layer(m=m, seed=m)
    k = helion.kernel(fe.BODIES[fe.ROUTE_QUANT], config=helion.Config(), static_shapes=False)
    out = k(x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), 6, base)
    ref = fe.route_quant_ref(x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), 6, base)
    ok, msg = fe.check_route_quant(out, ref)
    assert ok, msg
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1].view(torch.uint8), ref[1].view(torch.uint8))


@needs_helion
def test_quant_of_hostile_activations_is_the_spec(nvfp4_ref):
    p, x, ids, tw, base = layer(m=9, adversarial=True)
    x = nm.adversarial_blocks(x.float(), torch.Generator().manual_seed(4)).bfloat16()
    k = helion.kernel(fe.BODIES[fe.ROUTE_QUANT], config=helion.Config(), static_shapes=False)
    got = k(x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), 6, base)
    want = fe.route_quant_ref(x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), 6, base)
    assert fe.check_route_quant(got, want)[0]
    assert (want[1].view(torch.uint8) == 0).any()  # zero blocks really occur


# ---------------------------------------------------------------------------
# the layer, end to end (the plan's four kernels), both scale-read modes
# ---------------------------------------------------------------------------
LOADS = [dict(m=1), dict(m=7), dict(m=16, dist="skewed"), dict(m=12, dist="hot"),
         dict(m=9, limit=0.8), dict(m=16, limit=0.8, dist="skewed", adversarial=True)]


@needs_helion
@pytest.mark.parametrize("kw", LOADS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_layer_vs_f64_spec(kw, nvfp4_ref, cpu_mm, scale_read):
    p, x, ids, tw, base = layer(**kw)
    out = run(p, x, ids, tw, base)
    rel, lid = rel_vs_spec(out, p, x, ids, tw, base)
    assert rel <= nm.REF_TOL and torch.isfinite(out.float()).all(), rel
    assert rel < 5e-3  # measured ~1-2e-3 (bf16 output rounding + requant flips)
    if kw.get("limit") == 0.8:
        assert nm.clamp_frac(p, x, lid) > 0.05  # the clamp really engages


@needs_helion
def test_layer_with_many_small_tiles_and_long_row_loops(nvfp4_ref, cpu_mm, small_tiles, scale_read):
    """brow 16, 32 I-columns per program, 32-byte K steps, tiny route tiles; the hot load puts every row of
    the batch on one expert (a 40-row segment = 3 row tiles)."""
    small_tiles()
    p, x, ids, tw, base = layer(m=40, dist="hot", limit=0.8, seed=2)
    out = run(p, x, ids, tw, base)
    rel, lid = rel_vs_spec(out, p, x, ids, tw, base)
    assert rel <= nm.REF_TOL
    assert int(fe.route_ref(ids, 6, base)[1].max()) > 16  # a segment longer than one row tile
    assert nm.clamp_frac(p, x, lid) > 0.05


@needs_helion
def test_off_rank_tokens_are_exact_zeros_and_all_off_rank_batches_work(nvfp4_ref, cpu_mm, scale_read):
    p, x, ids, tw, base = layer(m=8, dead=True)  # every token routes outside this rank
    out = run(p, x, ids, tw, base)
    assert torch.equal(out, torch.zeros_like(out))
    p, x, ids, tw, base = layer(m=10, seed=3)  # rows t % 3 == 1 are off-rank
    out = run(p, x, ids, tw, base)
    dead = (nm.to_local(ids, base, 6) < 0).all(1)
    assert bool(dead.any()) and bool((out[dead] == 0).all())


@needs_helion
def test_gate_clamp_is_one_sided_and_the_test_can_tell(nvfp4_ref, cpu_mm, monkeypatch, scale_read):
    """vLLM clamps the gate from ABOVE only. Drive the gate strongly negative on some rows (a symmetric
    clamp would flatten them) and require the one-sided result; then show a symmetric spec would NOT match."""
    p, x, ids, tw, base = layer(m=16, limit=0.8, seed=6, gain=3.0)
    out = run(p, x, ids, tw, base)
    rel, lid = rel_vs_spec(out, p, x, ids, tw, base)
    assert rel <= nm.REF_TOL
    gu = nm.gate_up_ref(p, x, lid)
    i = gu.shape[1] // 2
    vp = (lid.reshape(-1) >= 0)
    assert bool((gu[vp][:, i:] < -0.8).any()), "no strongly negative gate in this batch: test is vacuous"
    orig = nm.clamp_ref
    monkeypatch.setattr(nm, "clamp_ref", lambda g, limit: g if limit is None else torch.cat(
        [g[..., :g.shape[-1] // 2].clamp(-limit, limit), g[..., g.shape[-1] // 2:].clamp(-limit, limit)], -1))
    sym, _ = rel_vs_spec(out, p, x, ids, tw, base)
    monkeypatch.setattr(nm, "clamp_ref", orig)
    assert sym > 3 * rel and sym > nm.REF_TOL / 4  # the symmetric clamp is distinguishable


@needs_helion
def test_stage_references_judge_fc1_fc2_and_combine(nvfp4_ref, cpu_mm, small_tiles, scale_read):
    small_tiles()
    p, x, ids, tw, base = layer(m=12, limit=0.8, dist="skewed", seed=4)
    ks = kernels("front4")
    e = p["w13"].shape[0]
    xq, xs, ps, counts, starts = ks[0](x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), e, base)
    w13s, w2s = fe.sf_arg(p["w13_sf"]), fe.sf_arg(p["w2_sf"])
    args1 = (xq, fe.sf_arg(xs), ps, counts, starts, p["w13"], w13s, p["g1"], fe._f32(p["a2g"]),
             fe._div6(p["a2g"]), p["act"], ids.shape[1], 0.8)
    hq, hs_ = ks[1](*args1)
    ok, msg = fe.check_fc1((hq, hs_), fe.fc1_ref(*args1))
    assert ok, msg
    args2 = (hq, fe.sf_arg(hs_), ps, counts, starts, tw, p["w2"], w2s, p["g2"])
    y = ks[2](*args2)
    ok, msg = fe.check_fc2(y, fe.fc2_ref(*args2))
    assert ok, msg
    out = torch.empty(12, 128, dtype=torch.bfloat16)
    ks[3](y, ids, base, e, out)  # lane B's combine, unchanged interface
    assert lb._check_combine(out, lb._ref_combine(y, ids, base, e, out))[0]


def kernels_for(name):
    return helion.kernel(fe.BODIES[name], config=helion.Config(), static_shapes=False, ignore_warnings=WARN)


@needs_helion
def test_lab_make_inputs_feed_the_bodies(nvfp4_ref, cpu_mm, scale_read):
    """What the autotune runner does per kernel: out = body(*make(case, 0)); check(out, reference(*args))."""
    model = {"name": "deepseekv41-tiny", "architecture": "DeepseekV41ForCausalLM", "quant_format": "nvfp4",
             "experts": 96, "hidden": 128, "moe_intermediate": 128, "topk": 3, "tp": 8, "ep": 8,
             "experts_total": 96, "swiglu_limit": 10.0}
    cs = fe.cases(model)
    assert cs and all(c["experts"] == 12 and c["experts_global"] == 96 and c["limit"] == 10.0 for c in cs)
    case = {**cs[0], "numtokens": 6}
    for e in fe.KERNELS:
        args = e["make"](case, 0)
        out = kernels_for(e["name"])(*args)
        ok, msg = e["check"](out, e["reference"](*args))
        assert ok, (e["name"], msg)


def test_autotune_check_compares_only_what_a_consumer_reads():
    """fc1 outputs hold garbage in off-rank rows: two configs agreeing on the local rows must pass, one
    differing there must fail."""
    ps = torch.tensor([2, 0, 1, 3, 4, 0], dtype=torch.int32)  # pairs = 5; local pairs {2, 0, 1}
    counts = torch.tensor([2, 1], dtype=torch.int32)
    starts = torch.tensor([0, 2], dtype=torch.int32)
    fe.LAST_ARGS[fe.FC1] = (None, None, ps, counts, starts)
    hq = torch.randint(0, 255, (6, 8), dtype=torch.uint8)
    hs_ = torch.randint(0, 120, (6, 2), dtype=torch.uint8).view(torch.float8_e4m3fn)
    other = (hq.clone(), hs_.clone())
    other[0][3:] = 7  # garbage rows 3, 4, 5 differ: not read
    chk = fe.autotune_check(fe.FC1)
    chk((hq, hs_), other)
    other[0][1] ^= 0xFF  # a local row differs
    with pytest.raises(AssertionError, match="hq"):
        chk((hq, hs_), other)
    with pytest.raises(AssertionError, match="no check for kernel"):
        fe.autotune_check(lb.COMBINE)((hq,), (hq,))  # lane B's body is judged by lane B's check


# ---------------------------------------------------------------------------
# Helion's real front end -> Triton source (structure of what would be compiled)
# ---------------------------------------------------------------------------
BANNED = ("atomic", "debug_barrier", "counters", "semaphore", "cooperative", "grid_sync", "barrier")


def lowered(name, e=4, h=256, i=128, topk=3, m=8):
    from helion.runtime.ref_mode import RefMode
    p = nm.make_problem(e, h, i, "silu", "cpu", 0, limit=10.0)
    ids, tw = nm.rand_routing(m, 2 * e, topk, "cpu", 1, 2, e)
    x = torch.randn(m, h).bfloat16()
    pairs = m * topk
    xq = torch.zeros(m, h // 2, dtype=torch.uint8)
    xs = fe.sf_arg(torch.zeros(m, h // 16, dtype=torch.uint8))
    ps = torch.zeros(pairs + 1, dtype=torch.int32)
    counts = torch.zeros(e, dtype=torch.int32)
    hq = torch.zeros(pairs + 1, i // 2, dtype=torch.uint8)
    hs_ = fe.sf_arg(torch.zeros(pairs + 1, i // 16, dtype=torch.uint8))
    args = {
        fe.ROUTE_QUANT: (x, ids, 1.0, 1.0, e, 2),
        fe.FC1: (xq, xs, ps, counts, counts, p["w13"], fe.sf_arg(p["w13_sf"]), p["g1"], 1.0, 1.0, 0, topk, 10.0),
        fe.FC2: (hq, hs_, ps, counts, counts, tw, p["w2"], fe.sf_arg(p["w2_sf"]), p["g2"]),
    }[name]
    cfg = {fe.ROUTE_QUANT: dict(block_sizes=[16, 32, 2, 8]), fe.FC1: dict(block_sizes=[16, 2, 32]),
           fe.FC2: dict(block_sizes=[16, 32, 32])}[name]
    k = helion.kernel(fe.BODIES[name], ref_mode=RefMode.OFF, static_shapes=False, ignore_warnings=WARN)
    return k.bind(args).to_triton_code(helion.Config(**cfg, num_warps=4, num_stages=3))


@needs_helion
@pytest.mark.parametrize("name", list(fe.BODIES))
def test_triton_source_of_every_body(name, scale_read):
    code = lowered(name)
    assert not any(t in code for t in BANNED), [t for t in BANNED if t in code]
    assert code.count("_launcher(") == 1
    assert f"def _helion_{name}(" in code and fe.TRITON_NAMES.get(name, "_helion_" + name) == "_helion_" + name
    calls = [ln for ln in code.splitlines() if "tl.dot_scaled(" in ln and not ln.strip().startswith("#")]
    expect = {fe.FC1: 2, fe.FC2: 1}.get(name, 0)
    assert len(calls) == expect and all(c.count("'e2m1'") == 2 for c in calls), calls
    if name in (fe.FC1, fe.FC2) and scale_read == "words":  # K5 targets: int32 words unpacked in registers, bit-cast
        assert code.count("tl.float8e4nv, bitcast=True") == expect + 1 and "tl.join" in code
    if name in (fe.FC1, fe.FC2) and scale_read == "bytes":  # the scale tile is a plain e4m3 load: no join, no bitcast
        assert "tl.join" not in code and "bitcast=True" not in code
    if name == fe.ROUTE_QUANT:  # three roots in ONE launch: pair sort, expert scan, quant
        grid = [ln for ln in code.splitlines() if "_launcher(" in ln][0]
        assert grid.count("+") >= 2 and "e_count" in grid


# ---------------------------------------------------------------------------
# the scale-read knob
# ---------------------------------------------------------------------------
def test_scale_read_knob_defaults_to_words_and_is_loud(monkeypatch):
    monkeypatch.delenv(fe.SCALE_ENV, raising=False)
    assert fe._scale_read() == "words" and fe.SCALE_READ == "words"
    monkeypatch.setenv(fe.SCALE_ENV, "bytes")
    assert fe._scale_read() == "bytes"
    monkeypatch.setenv(fe.SCALE_ENV, "swizzled")
    with pytest.raises(RuntimeError, match="expected words or bytes"):
        fe._scale_read()


def test_sf_arg_is_the_same_bytes_in_either_mode(monkeypatch):
    sf = torch.randint(0, 120, (2, 128, 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    monkeypatch.setattr(fe, "SCALE_READ", "words")
    w = fe.sf_arg(sf)
    assert w.dtype == torch.int32 and w.numel() == sf.numel() // 4
    monkeypatch.setattr(fe, "SCALE_READ", "bytes")
    b = fe.sf_arg(sf)
    assert b.dtype == torch.float8_e4m3fn and b.shape == (sf.numel(),)
    assert torch.equal(b.view(torch.uint8), sf.reshape(-1).view(torch.uint8)) \
        and torch.equal(w.view(torch.uint8), b.view(torch.uint8))
    with pytest.raises(RuntimeError, match="contiguous"):
        fe.sf_arg(sf.transpose(1, 2))


# ---------------------------------------------------------------------------
# THE CONTRACT of lane B's per-bucket seam (nvfp4_moe.check_variant)
# ---------------------------------------------------------------------------
def test_module_honours_the_variant_contract():
    lb.check_variant("dot_scaled", fe)  # raises with the exact violation otherwise
    assert fe.VARIANT == lb.DOT_SCALED == "dot_scaled"
    assert set(fe.PLANS) == {"front4"} and set(fe.PLANS) <= set(lb.PLANS)
    assert fe.COMBINE == lb.COMBINE and fe.PLANS["front4"] == (fe.ROUTE_QUANT, fe.FC1, fe.FC2, lb.COMBINE)
    # every plan name has a body (ours, or lane B's for the shared combine) and the Triton names K5 reads
    assert all(k in fe.BODIES or k in lb.BODIES for names in fe.PLANS.values() for k in names)
    assert set(fe.BODIES) == {fe.ROUTE_QUANT, fe.FC1, fe.FC2} and not set(fe.BODIES) & set(lb.BODIES)
    assert {k: b.__name__ for k, b in fe.BODIES.items()} == {k: k for k in fe.BODIES}
    assert set(fe.TRITON_NAMES.values()) == {"_helion_" + k for k in fe.BODIES}
    # lab names: lane B's documented prefix, no collision with the dequant kernels or the fused shared ones
    names = [e["name"] for e in fe.KERNELS]
    assert names == [fe.ROUTE_QUANT, fe.FC1, fe.FC2] and all(n.startswith("dsv41_moe_") for n in names)
    assert not set(names) & (set(lb.BODIES) | set(hs.NEW_KERNELS))
    assert all(set(lb._KERNEL_KEYS) <= set(e) for e in fe.KERNELS)
    # run_plan: lane B's signature, name for name
    want = inspect.signature(lb.run_plan)
    assert list(inspect.signature(fe.run_plan).parameters) == list(want.parameters)
    assert inspect.signature(fe.run_plan).parameters["limit"].default is None
    # the module is what VARIANT_MODULES points at, and the seam imports it through check_variant
    assert lb.VARIANT_MODULES[lb.DOT_SCALED] == fe.__name__
    assert lb.variant_module(lb.DOT_SCALED) is fe and lb.DOT_SCALED in lb.available_variants()


def _bad(**over):
    mod = types.ModuleType("bad_dsv41_moe_fp4mma")
    mod.__dict__.update({k: v for k, v in vars(fe).items() if not k.startswith("__")})
    for k, v in over.items():
        setattr(mod, k, v)
    return mod


@pytest.mark.parametrize("over,needle", [
    ({"VARIANT": None}, "VARIANT None != 'dot_scaled'"),
    ({"PLANS": {}}, "non-empty dict"),
    ({"PLANS": {"fp4mma4": ("a",)}}, "non-empty dict"),                       # the old plan names are refused
    ({"BODIES": []}, "BODIES must be"),
    ({"BODIES": {}}, "no body in BODIES"),                                    # a plan kernel nobody implements
    ({"run_plan": None}, "run_plan("),
    ({"KERNELS": []}, "KERNELS (lab ABI list) is missing"),
    ({"KERNELS": [{"name": "dsv41_moe_fc1"}]}, "lacks ['body'"),
    ({"KERNELS": [{**dict.fromkeys(lb._KERNEL_KEYS, 1), "name": lb.FC1}]}, "collides with a dequant kernel name"),
])
def test_contract_violations_are_refused(over, needle):
    if "no body in BODIES" in needle:
        over = {"BODIES": {}, "PLANS": {"front4": ("dsv41_moe_nobody",)}}
    with pytest.raises(RuntimeError, match="breaks the seam") as exc:
        lb.check_variant("dot_scaled", _bad(**over))
    assert needle in str(exc.value), str(exc.value)


# ---------------------------------------------------------------------------
# the seam end to end on CPU: table row -> served_ms -> serving_op -> run_plan, vs lane B's dequant bodies
# ---------------------------------------------------------------------------
class _Forbid(importlib.abc.MetaPathFinder):
    """The V4.1 route is oxide-free (docs/dsv41-moe.md section 12): a variant must not import the oxide family."""

    def __init__(self):
        self.hits = []

    def find_spec(self, fullname, path, target=None):
        if fullname in ("suffix_hybrid.oxide_kernels", "suffix_hybrid._native") \
                or fullname.startswith("suffix_hybrid._native."):
            self.hits.append(fullname)
            raise ImportError(f"{fullname}: the DeepSeek-V4 route is oxide-free")
        return None


@pytest.fixture
def seam(monkeypatch):
    """Lane B's real seam over a tiny clamped EP layer: hk.config answers from `rows`, no oxide importable."""
    rows = {}
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: rows.get((kernel, case["numtokens"])))
    finder = _Forbid()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for name in ("suffix_hybrid.oxide_kernels", "suffix_hybrid._native"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(suffix_hybrid, "oxide_kernels", raising=False)
    monkeypatch.delattr(suffix_hybrid, "_native", raising=False)
    hk._COMPILED.clear()
    lb._kernels.cache_clear()
    yield rows
    hk._COMPILED.clear()
    lb._kernels.cache_clear()
    lb._VARIANTS.pop("dot_scaled", None)
    assert finder.hits == [], f"the oxide-free route imported {finder.hits}"


def ship(rows, names, buckets, cfg=None):
    rows.update({(k, b): dict(cfg or {}) for k in names for b in buckets})


def ws():
    return (None,) * 7


def op_args(p, x, ids, tw, out):
    return (x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"], *ws(), out,
            p["a1g"], p["a2g"], p["act"])


MODE_FOR = lambda m: 2 if m <= 8 else 1  # the lane's tune table: fused3 up to M 8, front4 above
E_LOC, E_GLO, BASE, LIM = 6, 12, 3, 0.8


def seam_rows(rows, variant_buckets, buckets=lb.CLAMP_BUCKETS):
    """Shipped configs for the dequant plans on every bucket, the dot_scaled plan on `variant_buckets`
    (our three kernels; the combine is lane B's own config), and the table rows selecting dot_scaled."""
    ship(rows, {k for names in lb.PLANS.values() for k in names}, buckets)
    ship(rows, fe.BODIES, variant_buckets)
    for b in variant_buckets:
        rows[(lb.VARIANT_KERNEL, b)] = {"body": "dot_scaled"}


@needs_helion
def test_seam_serves_a_dot_scaled_bucket_and_matches_the_dequant_bodies(seam, nvfp4_ref, cpu_mm, small_tiles,
                                                                       scale_read, monkeypatch):
    small_tiles()
    seam_rows(seam, variant_buckets=(32,))
    got = lb.served_ms("helion", E_LOC, 128, 128, 3, 0, 40, MODE_FOR, LIM, E_GLO)
    assert got[20] == (32, "front4", "dot_scaled") and got[32] == (32, "front4", "dot_scaled")
    assert got[17] == (32, "front4", "dot_scaled") and got[16] == (16, "front4", "dequant")
    assert got[7] == (8, "fused3", "dequant") and got[40] == (64, "front4", "dequant")  # 33..40 -> bucket 64

    seen = []
    real = fe.run_plan
    monkeypatch.setattr(fe, "run_plan", lambda kerns, plan, *a, **kw: (seen.append((plan, len(kerns))),
                                                                     real(kerns, plan, *a, **kw))[1])
    table_op = lb.serving_op(None, "helion", LIM, E_GLO)                         # the table decides
    dequant_op = lb.serving_op(None, "helion", LIM, E_GLO, variant="dequant")    # forced: lane B's bodies
    p, x, ids, tw, base = layer(e_local=E_LOC, e_global=E_GLO, base=BASE, m=20, limit=LIM, dist="skewed", seed=7)
    ds = torch.full((20, 128), float("nan")).bfloat16()
    dq = torch.full((20, 128), float("nan")).bfloat16()
    r = table_op(*op_args(p, x, ids, tw, ds), base, 1, 0, 0)
    assert r is ds and seen == [("front4", 4)]                                    # OUR run_plan served it
    dequant_op(*op_args(p, x, ids, tw, dq), base, 1, 0, 0)
    assert seen == [("front4", 4)]                                                # lane B's bodies served this one
    rel_ds, lid = rel_vs_spec(ds, p, x, ids, tw, base)
    rel_dq, _ = rel_vs_spec(dq, p, x, ids, tw, base)
    rel_between = float((ds.double() - dq.double()).norm() / dq.double().norm().clamp_min(1e-30))
    assert rel_ds <= nm.REF_TOL and rel_dq <= nm.REF_TOL and rel_between <= nm.REF_TOL, (rel_ds, rel_dq, rel_between)
    assert nm.clamp_frac(p, x, lid) > 0.05                                        # the clamp engaged in both
    # a bucket with no row stays on the dequant bodies through the same op
    p7, x7, ids7, tw7, _ = layer(e_local=E_LOC, e_global=E_GLO, base=BASE, m=7, limit=LIM, seed=8)
    out7 = torch.full((7, 128), float("nan")).bfloat16()
    table_op(*op_args(p7, x7, ids7, tw7, out7), base, 2, 0, 0)
    assert seen == [("front4", 4)] and rel_vs_spec(out7, p7, x7, ids7, tw7, base)[0] <= nm.REF_TOL


@needs_helion
def test_seam_refuses_the_buckets_this_module_cannot_serve(seam):
    # M <= 8 is fused3 in the tune table; the variant has no fused3 plan: refused by name, never guessed
    seam_rows(seam, variant_buckets=(8,))
    with pytest.raises(RuntimeError, match=r"body variant 'dot_scaled' has no 'fused3' plan \(it serves \['front4'\]\)"):
        lb.served_ms("helion", E_LOC, 128, 128, 3, 0, 12, MODE_FOR, LIM, E_GLO)
    # a row whose bucket lacks one of the plan's configs is a startup error naming variant and kernel
    seam.clear()
    seam_rows(seam, variant_buckets=(32,))
    del seam[(fe.FC2, 32)]
    with pytest.raises(RuntimeError, match=r"M<=32 front4/dot_scaled: dsv41_moe_fc2"):
        lb.served_ms("helion", E_LOC, 128, 128, 3, 0, 40, MODE_FOR, LIM, E_GLO)
    # lane B's combine is part of the dot_scaled plan: its missing config is named too
    seam.clear()
    seam_rows(seam, variant_buckets=(32,))
    del seam[(lb.COMBINE, 32)]
    with pytest.raises(RuntimeError, match=r"M<=32 front4/dot_scaled: nvfp4_moe_combine"):
        lb.served_ms("helion", E_LOC, 128, 128, 3, 0, 40, MODE_FOR, LIM, E_GLO)


@needs_helion
def test_a_module_that_violates_the_contract_is_refused_through_the_seam(seam, monkeypatch):
    seam_rows(seam, variant_buckets=(32,))
    for over, needle in (({"VARIANT": "other"}, "VARIANT 'other'"),
                         ({"PLANS": {"fp4mma4": (fe.FC1,)}}, "non-empty dict"),
                         ({"run_plan": None}, "run_plan(")):
        monkeypatch.setitem(sys.modules, lb.VARIANT_MODULES["dot_scaled"], _bad(**over))
        monkeypatch.delitem(lb._VARIANTS, "dot_scaled", raising=False)
        with pytest.raises(RuntimeError, match="breaks the seam") as exc:
            lb.served_ms("helion", E_LOC, 128, 128, 3, 0, 40, MODE_FOR, LIM, E_GLO)    # the first use at load
        assert needle in str(exc.value)
        with pytest.raises(RuntimeError, match="breaks the seam"):
            lb.available_variants()                                                      # and in the lab
    monkeypatch.delitem(lb._VARIANTS, "dot_scaled", raising=False)


def _tensors(m, h=128, i=128, e=6, topk=3):
    p = nm.make_problem(e, h, i, "silu", "cpu", 0, limit=LIM)
    ids, tw = nm.rand_routing(m, 12, topk, "cpu", 1, 3, e)
    return p, torch.zeros(m, h, dtype=torch.bfloat16), ids, tw, torch.zeros(m, h, dtype=torch.bfloat16)


def test_run_plan_refuses_what_it_cannot_serve_before_launching_anything():
    boom = [lambda *a: pytest.fail("a kernel was launched")] * 4
    p, x, ids, tw, out = _tensors(4)
    args = (p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"], out, p["a1g"], p["a2g"])
    with pytest.raises(RuntimeError, match="no 'fused3' plan"):
        fe.run_plan(boom, "fused3", x, ids, tw, *args, 0, 3, LIM)
    with pytest.raises(RuntimeError, match="silu .0. only"):
        fe.run_plan(boom, "front4", x, ids, tw, *args, 1, 3, LIM)
    p2, x2, ids2, tw2, out2 = _tensors(4, h=128, i=96)  # I % 64 != 0: lane B's dequant bodies take it, ours do not
    with pytest.raises(RuntimeError, match="I % 64"):
        fe.run_plan(boom, "front4", x2, ids2, tw2, p2["w13"], p2["w13_sf"], p2["g1"], p2["w2"], p2["w2_sf"],
                    p2["g2"], out2, p2["a1g"], p2["a2g"], 0, 3, LIM)


# ---------------------------------------------------------------------------
# K5 inside run_plan: armed before the first GPU launch, judged right after it, nothing on CPU tensors
# ---------------------------------------------------------------------------
@pytest.fixture
def k5_state(monkeypatch):
    monkeypatch.setattr(fe, "_K5", {"stack": False, "seen": False, "n": 0})
    yield fe._K5


def _spy_kernels(calls):
    """Stand-ins that only record their order and return what run_plan feeds on (scale buffers uint8)."""
    sf = torch.zeros(4, dtype=torch.uint8)
    outs = {"k0": lambda a: (a[0], sf, a[1], a[1], a[1]), "k1": lambda a: (a[0], sf), "k2": lambda a: a[0],
            "k3": lambda a: a[-1]}
    return [lambda *a, _n=n: (calls.append(_n), outs[_n](a))[1] for n in ("k0", "k1", "k2", "k3")]


def test_run_plan_arms_k5_before_the_first_launch_and_judges_after_the_last_only_on_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(fe, "_k5_arm", lambda: calls.append("arm"))
    monkeypatch.setattr(fe, "_k5_check", lambda: calls.append("check"))
    p, x, ids, tw, out = _tensors(4)
    args = (x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"], out, p["a1g"], p["a2g"], 0, 3, LIM)
    fe.run_plan(_spy_kernels(calls), "front4", *args)
    assert calls == ["k0", "k1", "k2", "k3"], "CPU tensors: nothing armed, nothing judged"
    calls.clear()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))     # a GPU call, as far as run_plan can tell
    fe.run_plan(_spy_kernels(calls), "front4", *args)
    assert calls == ["arm", "k0", "k1", "k2", "k3", "check"]
    # the real arm on a box that is not SM12x: refused by name BEFORE any kernel is launched
    calls.clear()
    monkeypatch.undo()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fe, "_K5", {"stack": False, "seen": False, "n": 0})
    with pytest.raises(probe.Fp4mmaError, match="no CUDA device"):
        fe.run_plan(_spy_kernels(calls), "front4", *args)
    assert calls == []


def test_k5_arm_is_the_stack_gate_then_a_watching_recorder(k5_state, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)               # whatever box this runs on
    with pytest.raises(probe.Fp4mmaError, match="cannot run"):
        fe._k5_arm()
    assert not k5_state["stack"]
    seen = []
    monkeypatch.setattr(probe, "require_native_stack", lambda: seen.append("stack"))
    monkeypatch.setattr(probe, "install_recorder", lambda watch=None: seen.append(("recorder", watch)))
    monkeypatch.setattr(probe, "logline", lambda s: seen.append(s))
    fe._k5_arm()
    fe._k5_arm()                                                                  # once
    assert seen[0] == "stack" and seen[1] == ("recorder", tuple(fe.TRITON_NAMES.values())) and len(seen) == 3
    assert k5_state["stack"]


def test_k5_check_judges_every_new_fc1_fc2_compile_and_refuses_a_decomposition(k5_state, fake_launcher):
    import helion.runtime as hrt
    probe.install_recorder(watch=tuple(fe.TRITON_NAMES.values()))
    # first call: both native kernels must have been recorded at all (no PTX = no assertion = refuse)
    with pytest.raises(probe.Fp4mmaError, match="no compiled kernel recorded for '_helion_dsv41_moe_fc1'"):
        fe._k5_check()
    for name, h in (("_helion_dsv41_moe_fc1", "a"), ("_helion_dsv41_moe_fc2", "b"),
                    ("_helion_dsv41_moe_route_quant", "r")):
        fake_launcher["ck"] = FakeCK(NATIVE_PTX if name.endswith(("fc1", "fc2")) else DECOMPOSED_PTX, h)
        hrt.default_launcher(FakeJit(name), (1,))
    fe._k5_check()                                                                # route_quant has no MMA: not judged
    assert k5_state["seen"] and k5_state["n"] == probe.record_count() == 3
    fe._k5_check()                                                                # nothing new: one len(), no work
    # lane B's shared-expert kernel shares our name's prefix: neither recorded nor judged
    fake_launcher["ck"] = FakeCK(DECOMPOSED_PTX, "sh")
    hrt.default_launcher(FakeJit("_helion_dsv41_moe_fc1_sh"), (1,))
    assert probe.record_count() == 3
    # a new fc2 specialization that Triton decomposed is refused at the launch that compiled it
    fake_launcher["ck"] = FakeCK(DECOMPOSED_PTX, "c")
    hrt.default_launcher(FakeJit("_helion_dsv41_moe_fc2"), (1,))
    with pytest.raises(probe.Fp4mmaError, match="decomposed tl.dot_scaled"):
        fe._k5_check()


def test_recorder_watch_and_exact_names(fake_launcher):
    import helion.runtime as hrt
    probe.install_recorder(watch=("_helion_a",))
    probe.install_recorder(watch=("_helion_b",))                                   # accumulates
    for name, h in (("_helion_a", "1"), ("_helion_a_2", "2"), ("_helion_a_sh", "3"), ("_helion_b", "4"),
                    ("_helion_c", "5")):
        fake_launcher["ck"] = FakeCK(NATIVE_PTX, h)
        hrt.default_launcher(FakeJit(name), (1,))
    assert sorted(r.name for r in probe.records()) == ["_helion_a", "_helion_a_2", "_helion_b"]
    assert probe.name_matches("_helion_a_12", "_helion_a") and not probe.name_matches("_helion_a_sh", "_helion_a")
    got = probe.assert_native(["_helion_a"], log=lambda s: None, only_new=True)
    assert [r.name for r in got] == ["_helion_a", "_helion_a_2"]                   # exact: not _helion_a_sh, not _b
    assert probe.assert_native(["_helion_a"], log=lambda s: None, only_new=True) == []
    assert probe.assert_native(["_helion_zzz"], log=lambda s: None, only_new=True, require=False) == []


# ---------------------------------------------------------------------------
# lab ABI, capture-safety audits, the CPU gate
# ---------------------------------------------------------------------------
def test_cases_are_lane_bs_front4_cases_of_the_layouts_these_bodies_take(monkeypatch):
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    want = [c for c in lb.cases(DS, lb.FC2)]
    assert want and {c["numtokens"] for c in want} == {6, 8, 16, 32, 64, 128, 192}
    for e in fe.KERNELS:
        got = e["cases"](DS)
        assert got == want and all(c in lb.cases(DS, lb.FC1) for c in got)         # rows and selection share keys
    monkeypatch.setenv("NVFP4_MOE_TP8_NOEP", "1")                                  # the I = 288 no-EP rank: not ours
    assert all(c["moe_intermediate"] == 2304 for e in fe.KERNELS for c in e["cases"](DS))
    assert fe.cases({**DS, "quant_format": "bf16"}) == []


def test_tuner_plan_gains_exactly_our_three_kernels_on_the_front4_buckets(monkeypatch):
    from suffix_hybrid.helion_kernels import nvfp4_moe_tune as nt
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    assert [k["name"] for k in nt.variant_kernels()] == [e["name"] for e in fe.KERNELS]
    names = [k for k, _ in nt.todo()]
    assert {k: names.count(k) for k in fe.BODIES} == {k: 7 for k in fe.BODIES}
    assert len(names) == 53 + 3 * 7          # lane B's 53 (dequant 34 + fused shared 19) + 21 of ours
    assert nt.todo()[0][0] == lb.ROUTE_QUANT  # the key M of lane B's own plan still goes first


def _calls(fn):
    out = set()
    for node in ast.walk(ast.parse(inspect.getsource(fn))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_served_paths_are_sync_free_and_shape_static():
    banned = {"item", "cpu", "tolist", "synchronize", "nonzero", "bincount", "unique", "masked_select",
              "Stream", "Event", "record", "atomic_add", "empty_cache"}
    for fn in (fe.run_plan, fe._k5_arm, fe._k5_check, fe.sf_arg):
        assert not (_calls(fn) & banned), (fn.__name__, _calls(fn) & banned)


def test_module_never_imports_the_oxide_stack_and_holds_no_state_across_launches():
    tree = ast.parse(inspect.getsource(fe))
    for node in ast.walk(tree):
        mods = [node.module] if isinstance(node, ast.ImportFrom) and node.module else \
            [a.name for a in node.names] if isinstance(node, ast.Import) else []
        assert not any("oxide" in m or m.endswith("_native") for m in mods), mods
    for name, body in fe.BODIES.items():
        text = inspect.getsource(body)
        assert not any(t in text for t in ("atomic", "barrier", "counters", "semaphore")), name
        allocs = {t.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assign)
                  if isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "empty"
                  for t in n.targets if isinstance(t, ast.Name)}
        assert allocs, name  # every buffer a body writes is allocated in the body


@needs_helion
def test_capture_replay_gate_tiny_moe_probes_pass_on_the_eager_runtime(nvfp4_ref, cpu_mm):
    res = probe.run_gate(rt=probe.EagerRuntime, dev="cpu", plist=fe.probes("tiny"),
                         builders={"moe": fe.build_probe}, replays=3, streams=1, emit=lambda s: None)
    assert res["verdict"] == "PASS", res["probes"]
    assert {r["label"] for r in res["probes"]} == {"tiny.front4.m7", "tiny.front4.m16"}


def test_probe_lists():
    quick = fe.probes("quick")
    assert [p.spec["m"] for p in quick] == [6, 16, 48, 192] and all(p.spec["e"] == 48 for p in quick)
    assert all(p.spec["plan"] == "front4" for p in quick + fe.probes("full") + fe.probes("tiny"))
    assert {p.spec["dists"][2] for p in quick} == {"hot"} and "dead" in quick[0].spec["dists"]
    assert [p.spec["m"] for p in fe.probes("full")] == [6, 8, 16, 32, 48, 64, 96, 128, 192, 256]
