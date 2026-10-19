# SPDX-License-Identifier: Apache-2.0
"""K1-K3 on CPU for the V4.1 routed-experts family on native FP4 MMA (dsv41_moe_fp4mma.py): the five bodies
in Helion's eager mode against lane B's f64 spec (kernels/nvfp4_moe.moe_ref: quantize x and h to NVFP4,
swiglu clamp `gate = min(gate, L)` / `up = clamp(up, -L, L)` before silu, f32 accumulate), the two route
implementations against a plain python sort, balanced / skewed / hot / off-rank expert loads, adversarial
scale blocks, forced small tiles, the seam (per-M routing, chunking), the lab ABI and the capture gate.

CPU eager mode proves indexing and numerics. Speed, registers and the native instruction are K4/K5 (silicon).
"""
import ast
import inspect
import json

import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
from suffix_hybrid.helion_kernels import nvfp4_moe as lb
from suffix_hybrid.kernels import fp4mma_probe as probe
from suffix_hybrid.kernels import nvfp4_moe as nm

from test_fp4mma_ref import cpu_mm, nvfp4_ref  # noqa: E402,F401

try:
    import helion
except ImportError:
    helion = None
needs_helion = pytest.mark.skipif(helion is None, reason="Helion not installed")


def kernels(plan, cfgs=None):
    cfgs = cfgs or {}
    return [helion.kernel(fe.BODIES[k], config=helion.Config(**cfgs.get(k, {})), static_shapes=False,
                          ignore_warnings=[helion.exc.TensorOperationInWrapper]) for k in fe.PLANS[plan]]


def layer(e_local=6, e_global=12, base=3, h=128, i=128, topk=3, m=7, dist="balanced", seed=0, limit=10.0,
          gain=1.0, adversarial=False, dead=None):
    p = nm.make_problem(e_local, h, i, "silu", "cpu", seed, limit=limit, adversarial=adversarial)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed + 5, base, e_local, dist=dist, dead=dead)
    gen = torch.Generator().manual_seed(seed + 9)
    x = (torch.randn(m, h, generator=gen) * 2688.0 / p["a1g"] / 4.5 * gain).bfloat16()
    return p, x, ids, tw, base


def run(plan, p, x, ids, tw, base, cfgs=None):
    out = torch.full((x.shape[0], x.shape[1]), float("nan")).bfloat16()
    fe.run_plan(kernels(plan, cfgs), plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                p["g2"], out, p["a1g"], p["a2g"], p["act"], base, p["limit"])
    return out


def rel_vs_spec(out, p, x, ids, tw, base):
    lid = nm.to_local(ids, base, p["w13"].shape[0]).int()
    ref = nm.moe_ref(p, x, lid, tw)
    return float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30)), lid


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
# the route: three implementations of one sort
# ---------------------------------------------------------------------------
def ids_with_offrank(m, topk, e_global, base, e_local, seed):
    ids, _ = nm.rand_routing(m, e_global, topk, "cpu", seed, base, e_local, dist="skewed")
    return ids


@pytest.mark.parametrize("m", [1, 5, 16, 33])
def test_route_torch_equals_the_python_spec_and_lane_b(m):
    ids = ids_with_offrank(m, 3, 12, 3, 6, m)
    spec = fe.route_ref(ids, 6, 3)
    got = fe.route_torch(ids, 6, 3)
    for a, b in zip(got, spec):
        assert torch.equal(a, b) and a.dtype == torch.int32
    lps, lseg, lseg_e = lb.route_ref(ids, 6, 3)  # lane B's per-position arrays, rebuilt from (counts, starts)
    seg, seg_e = fe.seg_from(*spec)
    assert torch.equal(spec[0], lps) and torch.equal(seg, lseg) and torch.equal(seg_e, lseg_e)
    # structure: local pairs first, ascending expert, ascending pair; counts sum = number of local pairs
    n_local = int(spec[1].sum())
    keys = spec[0][:n_local].long()
    flat = ids.reshape(-1).long() - 3
    ek = flat[keys]
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
    # the quant kernel alone (the prefill-range plan) is the same quantization
    q = helion.kernel(fe.BODIES[fe.QUANT], config=helion.Config(), static_shapes=False)
    qo = q(x, fe._f32(p["a1g"]), fe._div6(p["a1g"]))
    assert torch.equal(qo[0], out[0]) and torch.equal(qo[1].view(torch.uint8), out[1].view(torch.uint8))


@needs_helion
def test_quant_of_hostile_activations_is_the_spec(nvfp4_ref):
    p, x, ids, tw, base = layer(m=9, adversarial=True)
    x = nm.adversarial_blocks(x.float(), torch.Generator().manual_seed(4)).bfloat16()
    q = helion.kernel(fe.BODIES[fe.QUANT], config=helion.Config(), static_shapes=False)
    got = q(x, fe._f32(p["a1g"]), fe._div6(p["a1g"]))
    want = fe.quant_ref(x, fe._f32(p["a1g"]), fe._div6(p["a1g"]))
    assert fe.check_quant(got, want)[0]
    assert (want[1].view(torch.uint8) == 0).any()  # zero blocks really occur


# ---------------------------------------------------------------------------
# the layer, end to end, both plans
# ---------------------------------------------------------------------------
LOADS = [dict(m=1), dict(m=7), dict(m=16, dist="skewed"), dict(m=12, dist="hot"),
         dict(m=9, limit=0.8), dict(m=16, limit=0.8, dist="skewed", adversarial=True)]


@needs_helion
@pytest.mark.parametrize("plan", ["fp4mma4", "fp4mma4t"])
@pytest.mark.parametrize("kw", LOADS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_layer_vs_f64_spec(plan, kw, nvfp4_ref, cpu_mm, scale_read):
    p, x, ids, tw, base = layer(**kw)
    out = run(plan, p, x, ids, tw, base)
    rel, lid = rel_vs_spec(out, p, x, ids, tw, base)
    assert rel <= nm.REF_TOL and torch.isfinite(out.float()).all(), rel
    assert rel < 5e-3  # measured ~1-2e-3 (bf16 output rounding + requant flips)
    if kw.get("limit") == 0.8:
        assert nm.clamp_frac(p, x, lid) > 0.05  # the clamp really engages


@needs_helion
@pytest.mark.parametrize("plan", ["fp4mma4", "fp4mma4t"])
def test_layer_with_many_small_tiles_and_long_row_loops(plan, nvfp4_ref, cpu_mm, small_tiles, scale_read):
    """brow 16, 32 I-columns per program, 32-byte K steps, tiny route tiles; the hot load puts every row of
    the batch on one expert (a 40-row segment = 3 row tiles)."""
    small_tiles()
    p, x, ids, tw, base = layer(m=40, dist="hot", limit=0.8, seed=2)
    out = run(plan, p, x, ids, tw, base)
    rel, lid = rel_vs_spec(out, p, x, ids, tw, base)
    assert rel <= nm.REF_TOL
    counts = fe.route_ref(ids, 6, base)[1]
    assert int(counts.max()) > 16  # a segment longer than one row tile
    assert nm.clamp_frac(p, x, lid) > 0.05


@needs_helion
def test_off_rank_tokens_are_exact_zeros_and_all_off_rank_batches_work(nvfp4_ref, cpu_mm, scale_read):
    for plan in ("fp4mma4", "fp4mma4t"):
        p, x, ids, tw, base = layer(m=8, dead=True)  # every token routes outside this rank
        out = run(plan, p, x, ids, tw, base)
        assert torch.equal(out, torch.zeros_like(out))
        p, x, ids, tw, base = layer(m=10, seed=3)  # rows t % 3 == 1 are off-rank
        out = run(plan, p, x, ids, tw, base)
        lid = nm.to_local(ids, base, 6)
        dead = (lid < 0).all(1)
        assert bool(dead.any()) and bool((out[dead] == 0).all())


@needs_helion
def test_gate_clamp_is_one_sided_and_the_test_can_tell(nvfp4_ref, cpu_mm, monkeypatch, scale_read):
    """vLLM clamps the gate from ABOVE only. Drive the gate strongly negative on some rows (a symmetric
    clamp would flatten them) and require the one-sided result; then show a symmetric spec would NOT match."""
    p, x, ids, tw, base = layer(m=16, limit=0.8, seed=6, gain=3.0)
    out = run("fp4mma4", p, x, ids, tw, base)
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
    ks = kernels("fp4mma4")
    e = p["w13"].shape[0]
    xq, xs, ps, counts, starts = ks[0](x, ids, fe._f32(p["a1g"]), fe._div6(p["a1g"]), e, base)
    w13s, w2s = fe.sf_arg(p["w13_sf"]), fe.sf_arg(p["w2_sf"])
    args1 = (xq, fe.sf_arg(xs), ps, counts, starts, p["w13"], w13s, p["g1"], fe._f32(p["a2g"]),
             fe._div6(p["a2g"]), p["act"], ids.shape[1], 0.8)
    hq, hs = ks[1](*args1)
    ok, msg = fe.check_fc1((hq, hs), fe.fc1_ref(*args1))
    assert ok, msg
    args2 = (hq, fe.sf_arg(hs), ps, counts, starts, tw, p["w2"], w2s, p["g2"])
    y = ks[2](*args2)
    ok, msg = fe.check_fc2(y, fe.fc2_ref(*args2))
    assert ok, msg
    out = torch.empty(12, 128, dtype=torch.bfloat16)
    ks[3](y, ids, base, e, out)
    want = lb._ref_combine(y, ids, base, e, out)
    assert lb._check_combine(out, want)[0]


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
        if e["name"] == fe.QUANT:
            continue
        args = e["make"](case, 0)
        out = kernels_for(e["name"])(*args)
        ok, msg = e["check"](out, e["reference"](*args))
        assert ok, (e["name"], msg)
    q = next(x for x in fe.KERNELS if x["name"] == fe.QUANT)
    qa = q["make"](case, 0)
    assert q["check"](kernels_for(fe.QUANT)(*qa), q["reference"](*qa))[0]


def kernels_for(name):
    return helion.kernel(fe.BODIES[name], config=helion.Config(), static_shapes=False,
                         ignore_warnings=[helion.exc.TensorOperationInWrapper])


def test_autotune_check_compares_only_what_a_consumer_reads():
    """fc1 outputs hold garbage in off-rank rows: two configs agreeing on the local rows must pass, one
    differing there must fail."""
    ps = torch.tensor([2, 0, 1, 3, 4, 0], dtype=torch.int32)  # pairs = 5; local pairs {2, 0, 1}
    counts = torch.tensor([2, 1], dtype=torch.int32)
    starts = torch.tensor([0, 2], dtype=torch.int32)
    fe.LAST_ARGS[fe.FC1] = (None, None, ps, counts, starts)
    hq = torch.randint(0, 255, (6, 8), dtype=torch.uint8)
    hs = torch.randint(0, 120, (6, 2), dtype=torch.uint8).view(torch.float8_e4m3fn)
    other = (hq.clone(), hs.clone())
    other[0][3:] = 7  # garbage rows 3, 4, 5 differ: not read
    chk = fe.autotune_check(fe.FC1)
    chk((hq, hs), other)
    other[0][1] ^= 0xFF  # a local row differs
    with pytest.raises(AssertionError, match="hq"):
        chk((hq, hs), other)


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
    hs = fe.sf_arg(torch.zeros(pairs + 1, i // 16, dtype=torch.uint8))
    args = {
        fe.ROUTE_QUANT: (x, ids, 1.0, 1.0, e, 2), fe.QUANT: (x, 1.0, 1.0),
        fe.FC1: (xq, xs, ps, counts, counts, p["w13"], fe.sf_arg(p["w13_sf"]), p["g1"], 1.0, 1.0, 0, topk, 10.0),
        fe.FC2: (hq, hs, ps, counts, counts, tw, p["w2"], fe.sf_arg(p["w2_sf"]), p["g2"]),
        fe.COMBINE: (torch.zeros(pairs + 1, h), ids, 2, e, torch.zeros(m, h, dtype=torch.bfloat16)),
    }[name]
    cfg = {fe.ROUTE_QUANT: dict(block_sizes=[16, 32, 2, 8]), fe.QUANT: dict(block_sizes=[2, 8]),
           fe.FC1: dict(block_sizes=[16, 2, 32]), fe.FC2: dict(block_sizes=[16, 32, 32]),
           fe.COMBINE: dict(block_sizes=[2, 64])}[name]
    k = helion.kernel(fe.BODIES[name], ref_mode=RefMode.OFF, static_shapes=False,
                      ignore_warnings=[helion.exc.TensorOperationInWrapper])
    return k.bind(args).to_triton_code(helion.Config(**cfg, num_warps=4, num_stages=3))


@needs_helion
@pytest.mark.parametrize("name", list(fe.BODIES))
def test_triton_source_of_every_body(name, scale_read):
    code = lowered(name)
    assert not any(t in code for t in BANNED), [t for t in BANNED if t in code]
    assert code.count("_launcher(") == 1
    calls = [ln for ln in code.splitlines() if "tl.dot_scaled(" in ln and not ln.strip().startswith("#")]
    expect = {fe.FC1: 2, fe.FC2: 1}.get(name, 0)
    assert len(calls) == expect and all(c.count("'e2m1'") == 2 for c in calls), calls
    if name in (fe.FC1, fe.FC2) and scale_read == "words":  # K5 targets: int32 words unpacked in registers, bit-cast to e4m3
        assert code.count("tl.float8e4nv, bitcast=True") == expect + 1 and "tl.join" in code
    if name in (fe.FC1, fe.FC2) and scale_read == "bytes":  # the scale tile is a plain e4m3 load: no join, no bitcast
        assert "tl.join" not in code and "bitcast=True" not in code
    if name == fe.ROUTE_QUANT:  # three roots in ONE launch: pair sort, expert scan, quant
        grid = [ln for ln in code.splitlines() if "_launcher(" in ln][0]
        assert grid.count("+") >= 2 and "e_count" in grid
    if name == fe.COMBINE:
        assert "tl.range(0, tl.cast(topk" in code


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------
@pytest.fixture
def shipped(monkeypatch):
    cfgs = {}
    monkeypatch.setattr(hk, "config", lambda name, case, plat=None: cfgs.get((name, case["numtokens"])))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    return cfgs


def ship(cfgs, bucket, plan):
    for k in fe.PLANS[plan]:
        cfgs[(k, bucket)] = {}


def test_bucket_case_uses_lane_bs_schema_with_this_familys_buckets():
    c = fe.bucket_case(48, 5120, 2304, 6, 0, 150, 10.0, 384)
    assert c == {"experts": 48, "hidden": 5120, "moe_intermediate": 2304, "topk": 6, "act": "silu",
                 "numtokens": 192, "limit": 10.0, "experts_global": 384}
    assert fe.bucket_case(48, 5120, 2304, 6, 0, 1, 10.0, 384)["numtokens"] == 1
    assert fe.bucket_case(48, 5120, 2304, 6, 0, 300, 10.0, 384)["numtokens"] == 512
    assert fe.bucket_case(48, 5120, 2304, 6, 0, 1024, 10.0, 384)["numtokens"] == 1024
    assert fe.bucket_case(48, 5120, 2304, 6, 0, 3000, 10.0, 384)["numtokens"] == 4096
    assert [fe.plan_name(m) for m in (1, 512, 513, 1024)] == ["fp4mma4", "fp4mma4", "fp4mma4t", "fp4mma4t"]


def test_served_ms_serves_a_bucket_only_when_every_kernel_of_its_plan_is_shipped(shipped):
    ship(shipped, 8, "fp4mma4")
    ship(shipped, 32, "fp4mma4")
    shipped[(fe.FC2, 64)] = {}  # a partially shipped bucket serves nothing
    shipped[(fe.FC1, 64)] = {}
    got = fe.served_ms(48, 5120, 2304, 6, 0, 70, 10.0, 384)
    assert sorted(set(got.values())) == [(8, "fp4mma4"), (32, "fp4mma4")]
    assert sorted(got) == list(range(7, 9)) + list(range(17, 33))  # M 7..8 -> bucket 8, 17..32 -> bucket 32
    assert all(not (9 <= m <= 16 or m > 32) for m in got)
    with pytest.raises(RuntimeError, match="STRICT.*M<=1 fp4mma4"):
        fe.served_ms(48, 5120, 2304, 6, 0, 70, 10.0, 384, fail_closed=True)


def test_served_ms_chunks_prefill_and_needs_every_chunk_bucket(shipped, monkeypatch):
    ship(shipped, 1024, "fp4mma4t")
    monkeypatch.setattr(fe, "CHUNK_M", 1024)
    got = fe.served_ms(48, 5120, 2304, 6, 0, 2048 + 7, 10.0, 384)
    assert 1024 in got and 2048 in got
    assert (2048 + 7) not in got  # the 7-token remainder chunk (bucket 8) is unserved
    assert 1025 not in got  # remainder chunk of 1 token: bucket 1 unshipped
    ship(shipped, 1, "fp4mma4")
    assert 1025 in fe.served_ms(48, 5120, 2304, 6, 0, 1100, 10.0, 384)
    assert fe.chunks(1100, 1024) == [(0, 1024), (1024, 1100)] and fe.chunks(50, 1024) == [(0, 50)]


def test_chunk_size_from_the_y_budget(monkeypatch):
    monkeypatch.delenv(fe.Y_MB_ENV, raising=False)
    monkeypatch.setattr(fe, "CHUNK_M", None)
    # V4.1: y = tokens * 6 * 5120 * 4 B; 512 MB -> 4096 tokens (2 passes at an 8192-token prefill chunk)
    assert fe.chunk_size(6, 5120) == 4096
    monkeypatch.setenv(fe.Y_MB_ENV, "128")
    assert fe.chunk_size(6, 5120) == 1024
    monkeypatch.setenv(fe.Y_MB_ENV, "16")
    assert fe.chunk_size(6, 5120) == 256 and fe.chunk_size(8, 6144) == 256  # floor
    monkeypatch.setenv(fe.Y_MB_ENV, "lots")
    with pytest.raises(RuntimeError, match="MB >= 16"):
        fe.chunk_size(6, 5120)
    monkeypatch.setattr(fe, "CHUNK_M", 7)
    assert fe.chunk_size(6, 5120) == 7


def test_warm_ms_per_specialization_class():
    served = {m: (8, "fp4mma4") for m in range(1, 40)}
    assert fe.warm_ms(served) == [1, 2, 8, 16, 24, 32, 39]
    assert fe.warm_ms({}) == []


def test_eligibility():
    assert fe.eligible(5120, 2304, 48) is None
    assert "I % 64" in fe.eligible(5120, 288, 384)  # the kernel-only TP8-no-EP rank: lane B's, not ours
    assert fe.eligible(5120, 2304, 0) and fe.eligible(100, 2304, 8) and fe.eligible(5120, 64 - 16, 8)
    assert fe.eligible(2816, 704, 128) is None  # gemma-4 26B: I = 704 = 11 * 64


def fake_args(m, h=128, topk=3, e=6, i=128):
    x = torch.zeros(m, h, dtype=torch.bfloat16)
    ids = torch.zeros(m, topk, dtype=torch.int32)
    tw = torch.zeros(m, topk)
    w13 = torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8)
    return (x, ids, tw, w13, None, None, None, None, None, None, None, None, None, None, None, None,
            torch.zeros(m, h, dtype=torch.bfloat16), 1.0, 1.0, 0, 0, 1, 0, 0)


def test_serving_op_routes_per_m_and_chunks(monkeypatch):
    calls = []
    base = lambda *a: calls.append(("base", a[0].shape[0])) or "base"
    monkeypatch.setattr(fe, "plan_name", lambda n: "fp4mma4")
    monkeypatch.setattr(fe, "_kernels", lambda key, plan, untuned: (json.loads(key)["numtokens"], plan))
    monkeypatch.setattr(fe, "run_plan", lambda kerns, plan, x, *a: calls.append((kerns[0], x.shape[0])))
    monkeypatch.setattr(fe, "CHUNK_M", 8)
    served = {m: (hk.bucket(m, fe.BUCKETS), "fp4mma4") for m in range(1, 9)}
    op = fe.serving_op(base, 10.0, 12, served)
    assert op.backend == "fp4mma" and op.served is served
    # served M: the family's plan, at the bucket the lab tuned (M = 5 -> bucket 6)
    out = op(*fake_args(5))
    assert calls == [(6, 5)] and out.shape == (5, 128)
    # an M the table does not hold (and not chunkable into held lengths): the wrapped op, all args through
    calls.clear()
    del served[7]
    assert op(*fake_args(7)) == "base" and calls == [("base", 7)]
    # chunked: M = 19 = 8 + 8 + 3, every chunk length served -> three plan runs on token slices, no base call
    calls.clear()
    out = op(*fake_args(19))
    assert calls == [(8, 8), (8, 8), (6, 3)] and out.shape == (19, 128)
    # one unserved chunk length sends the WHOLE call to the wrapped op
    del served[3]
    calls.clear()
    assert op(*fake_args(19)) == "base" and calls == [("base", 19)]
    # untuned_ok (standalone oracle / bench): no table needed, every M runs the family
    calls.clear()
    free = fe.serving_op(base, 10.0, 12, {}, untuned_ok=True)
    free(*fake_args(3))
    assert calls == [(6, 3)]


def test_prepare_layer_gates(monkeypatch, shipped):
    base = object()
    monkeypatch.delenv(fe.ENV, raising=False)
    with pytest.raises(RuntimeError, match="unset/0"):
        fe.prepare_layer(base, 5120, 2304, 48, 6, 0, 192, 10.0, 384)
    monkeypatch.setenv(fe.ENV, "1")
    with pytest.raises(RuntimeError, match="not eligible.*I % 64"):
        fe.prepare_layer(base, 5120, 288, 384, 6, 0, 192, 10.0, 384)
    with pytest.raises(probe.Fp4mmaError, match="cannot run"):
        fe.prepare_layer(base, 5120, 2304, 48, 6, 0, 192, 10.0, 384)  # this box: no SM12x / Triton
    monkeypatch.setattr(probe, "require_native_stack", lambda: None)
    monkeypatch.setattr(probe, "install_recorder", lambda: True)
    op, served = fe.prepare_layer(base, 5120, 2304, 48, 6, 0, 192, 10.0, 384)
    assert op is base and served == {}  # nothing shipped: lane B's behaviour exactly
    monkeypatch.setenv(fe.STRICT_ENV, "1")
    with pytest.raises(RuntimeError, match="STRICT"):
        fe.prepare_layer(base, 5120, 2304, 48, 6, 0, 192, 10.0, 384)
    monkeypatch.delenv(fe.STRICT_ENV)
    ship(shipped, 8, "fp4mma4")
    op, served = fe.prepare_layer(base, 5120, 2304, 48, 6, 0, 192, 10.0, 384)
    assert op is not base and op.backend == "fp4mma" and sorted(served) == [7, 8]


def test_verify_served_asserts_the_two_dot_kernels(monkeypatch):
    seen = []
    monkeypatch.setattr(probe, "assert_native", lambda names, allow_local=True: seen.append(list(names)) or [])
    assert fe.verify_served({}) == [] and seen == []
    fe.verify_served({1: (1, "fp4mma4")})
    assert seen == [["_helion_dsv41_fp4mma_fc1", "_helion_dsv41_fp4mma_fc2"]]


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


def test_knob_loud_and_default_off(monkeypatch):
    monkeypatch.delenv(fe.ENV, raising=False)
    assert fe.enabled() is False and fe.strict() is False
    monkeypatch.setenv(fe.ENV, "2")
    with pytest.raises(RuntimeError, match="expected 0 or 1"):
        fe.enabled()


# ---------------------------------------------------------------------------
# capture-safety audits + the CPU gate
# ---------------------------------------------------------------------------
def _calls(fn):
    out = set()
    for node in ast.walk(ast.parse(inspect.getsource(fn))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_served_paths_are_sync_free_and_shape_static():
    banned = {"item", "cpu", "tolist", "synchronize", "nonzero", "bincount", "unique", "masked_select",
              "Stream", "Event", "record", "atomic_add"}
    for fn in (fe.route_torch, fe.run_plan, fe.serving_op, fe.verify_served, fe.served_ms, fe.chunks):
        assert not (_calls(fn) & banned), (fn.__name__, _calls(fn) & banned)
    # the op closure inside serving_op: audited through its source text
    assert not any(t in inspect.getsource(fe.serving_op) for t in (".item()", "synchronize", "nonzero"))


def test_bodies_hold_no_state_across_launches():
    for name, body in fe.BODIES.items():
        text = inspect.getsource(body)
        assert not any(t in text for t in ("atomic", "barrier", "counters", "semaphore")), name
        tree = ast.parse(text)
        # every buffer a body writes is allocated in the body (torch.empty) or is its `out`
        allocs = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  if isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "empty"
                  for t in n.targets if isinstance(t, ast.Name)}
        assert allocs or name == fe.COMBINE, name


@needs_helion
def test_capture_replay_gate_tiny_moe_probes_pass_on_the_eager_runtime(nvfp4_ref, cpu_mm):
    res = probe.run_gate(rt=probe.EagerRuntime, dev="cpu", plist=fe.probes("tiny"),
                         builders={"moe": fe.build_probe}, replays=3, streams=1, emit=lambda s: None)
    assert res["verdict"] == "PASS", res["probes"]
    assert {r["label"] for r in res["probes"]} == {"tiny.fp4mma4.m7", "tiny.fp4mma4t.m16"}


def test_probe_lists():
    quick = fe.probes("quick")
    assert [p.spec["m"] for p in quick] == [1, 6, 48, 192, 600] and all(p.spec["e"] == 48 for p in quick)
    assert all(p.spec["plan"] == ("fp4mma4" if p.spec["m"] <= fe.ROUTE_KERNEL_MAX_M else "fp4mma4t") for p in quick)
    assert {p.spec["dists"][2] for p in quick} == {"hot"} and "dead" in quick[0].spec["dists"]
    assert len(fe.probes("full")) == 15
