# SPDX-License-Identifier: Apache-2.0
"""Helion port of nvfp4_moe: lab ABI + serving decisions (pure python), and
every body in Helion's eager reference mode (HELION_INTERPRET=1) on tiny CPU
problems vs the lane's spec — proves indexing and math, not speed. Silicon:
the autotune lab gate (reference vs Helion) and the lane's layer oracle."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before Helion is imported

import re  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_moe as hm  # noqa: E402
from suffix_hybrid.kernels import nvfp4_moe as nm  # noqa: E402

GLM_TP8 = {"name": "glm-5.3-nvfp4", "architecture": "GlmMoeDsaForCausalLM", "tp": 8,
           "hidden": 6144, "moe_intermediate": 256, "experts": 256, "topk": 8,
           "quant_format": "nvfp4"}


DS = {"name": "deepseek-v4.1-flash-1017-nvfp4", "architecture": "DeepseekV41ForCausalLM",
      "tp": 8, "hidden": 5120, "moe_intermediate": 288, "experts": 384, "topk": 6,
      "quant_format": "nvfp4"}


def test_cases_deepseek_v4_ep8_default_tp8_noep_opt_in(monkeypatch):
    tp8 = {"experts": 384, "hidden": 5120, "moe_intermediate": 288, "topk": 6, "act": "silu",
           "limit": 10.0}
    ep = {**tp8, "experts": 48, "moe_intermediate": 2304, "experts_global": 384}
    nb = len(hm.CLAMP_BUCKETS)
    strip = lambda got: [{k: v for k, v in c.items() if k != "numtokens"} for c in got]
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    # the console's api/autotune.model_dims for a TP8 + EP8 target: experts = this rank's,
    # experts_total = the model's, moe_intermediate = whole, ep = 8 (same lab cases)
    ep8 = {**DS, "ep": 8, "experts": 48, "experts_total": 384, "moe_intermediate": 2304}
    for model in (DS, ep8, {**DS, "tp": 1, "moe_intermediate": 2304},
                  {**DS, "tp": 4, "moe_intermediate": 576}):
        got = hm.cases(model)
        assert strip(got) == [ep] * nb
        assert [c["numtokens"] for c in got] == list(hm.CLAMP_BUCKETS)
    monkeypatch.setenv("NVFP4_MOE_TP8_NOEP", "1")
    for model in (DS, {**DS, "tp": 1, "moe_intermediate": 2304}):
        got = hm.cases(model)
        assert strip(got) == [ep] * nb + [tp8] * nb
        assert [c["numtokens"] for c in got] == list(hm.CLAMP_BUCKETS) * 2
    # per-kernel: each plan's own buckets (fused3 <= 16, front4 >= 6)
    per = {k["name"]: k["cases"](DS) for k in hm.KERNELS}
    assert len(per[hm.FC1]) == len(per[hm.ROUTE_QUANT]) == 2 * len(hm.CLAMP_BUCKETS)
    assert {c["numtokens"] for c in per[hm.FC2_COMBINE]} == {1, 6, 8, 16}
    assert {c["numtokens"] for c in per[hm.FC2]} == {c["numtokens"] for c in per[hm.COMBINE]} \
        == {6, 8, 16, 32, 64, 128, 192}
    assert sum(len(v) for v in per.values()) == 68  # both layouts (env set above)
    # GLM keeps every bucket for every kernel (unchanged keys)
    assert all(len(k["cases"](GLM_TP8)) == len(hm.BUCKETS) for k in hm.KERNELS)
    # serving builds exactly the lab's keys for both layouts
    assert hm.lane_case(384, 5120, 288, 6, 0, 2, 10.0, 384) == {**tp8, "numtokens": 6}
    assert hm.lane_case(384, 5120, 288, 6, 0, 9, 10.0, 384) == {**tp8, "numtokens": 16}
    assert hm.lane_case(48, 5120, 2304, 6, 0, 150, 10.0, 384) == {**ep, "numtokens": 192}
    assert hm.oxide_can(hm.cases(GLM_TP8)[0]) and not hm.oxide_can(hm.cases(DS)[0])


def test_cases_per_model():
    got = hm.cases(GLM_TP8)
    assert [c["numtokens"] for c in got] == list(hm.BUCKETS) and len(got) <= 64
    assert got[0] == {"experts": 256, "hidden": 6144, "moe_intermediate": 256, "topk": 8,
                      "act": "silu", "numtokens": 1}
    gemma = {**GLM_TP8, "architecture": "Gemma4ForConditionalGeneration", "experts": 128,
             "hidden": 2816, "moe_intermediate": 704}
    assert hm.cases(gemma)[0]["act"] == "gelu_tanh"
    for bad in ({"quant_format": "bf16"}, {"experts": 512}, {"moe_intermediate": 96},
                {"moe_intermediate": 100}, {"topk": None}):
        assert hm.cases({**GLM_TP8, **bad}) == []
    # serving builds exactly the lab's key for a decode call
    assert hm.lane_case(256, 6144, 256, 8, 0, 5) == {**got[0], "numtokens": 6}


def test_kernels_abi():
    names = [k["name"] for k in hm.KERNELS]
    assert len(set(names)) == len(names) == 5
    assert set(names) == {n for plan in hm.PLANS.values() for n in plan}
    for k in hm.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"])
        assert k["body"].__module__ == hm.__name__ and k["body"].__name__ == k["name"]
        assert k["baseline_name"].startswith("oxide nvfp4_moe ")
        oxide = [e for e in nm.PARAMS if any(re.search(r, e) for r in k["replaces"])]
        assert len(oxide) == 1 and oxide[0] in k["baseline_name"]


def test_baseline_runs_the_prod_tune(monkeypatch):
    monkeypatch.delenv(nm.TUNE_ENV, raising=False)
    glm, label = hm.baseline_tune(hm.cases(GLM_TP8)[0])
    assert label == "glm prod tune" and nm.tune_for(1, glm) == (2, nm.tune_word(32, 255, 1))
    assert nm.tune_for(64, glm) == (1, nm.tune_word(8, 255, 0))
    gemma = {**hm.cases(GLM_TP8)[0], "experts": 128, "hidden": 2816, "act": "gelu_tanh"}
    assert hm.baseline_tune(gemma) == (nm.parse_tune(nm.DEFAULT_TUNE), "lane default tune")
    monkeypatch.setenv(nm.TUNE_ENV, "8:legacy6/0/0/0")
    assert hm.baseline_tune(hm.cases(GLM_TP8)[0]) == ([(8, 0, 0, 0, 0)], "env tune")


def _fake_tables(monkeypatch, have):
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"num_warps": 4} if (kernel, case["numtokens"]) in have else None)


def test_served_ms_knob(monkeypatch):
    mode_for = lambda m: 2 if m <= 1 else 1  # fused3 at M=1, front4 above
    every = {(k, b) for k in (hm.ROUTE_QUANT, hm.FC1, hm.FC2, hm.COMBINE, hm.FC2_COMBINE)
             for b in hm.BUCKETS}
    _fake_tables(monkeypatch, every - {(hm.FC2, 8)})
    args = (256, 6144, 256, 8, 0, 12, mode_for)
    assert hm.served_ms("oxide", *args) == {}
    auto = hm.served_ms("auto", *args)
    assert auto[1] == (1, "fused3", "dequant") and auto[12] == (12, "front4", "dequant")
    assert not set(auto) & {7, 8}  # front4 bucket 8 lacks fc2 -> oxide there
    with pytest.raises(RuntimeError, match="M<=8 front4/dequant: nvfp4_moe_fc2"):
        hm.served_ms("helion", *args)
    _fake_tables(monkeypatch, every)
    assert sorted(hm.served_ms("helion", *args)) == list(range(1, 13))


def test_serving_op_falls_back_to_oxide_per_shape(monkeypatch):
    _fake_tables(monkeypatch, set())
    seen = []

    class Native:
        def nvfp4_moe_cuda(self, *a):
            seen.append(a)

    x = torch.zeros(3, 128, dtype=torch.bfloat16)
    args = (x, torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 2), torch.zeros(4, 128, 64,
            dtype=torch.uint8)) + (None,) * 13 + (1.0, 1.0, 0, 0, 1, 0x10, 0)
    hm.serving_op(Native(), "auto")(*args)
    assert seen == [args]  # the oxide op, verbatim arguments
    with pytest.raises(RuntimeError, match="helion"):
        hm.serving_op(Native(), "helion")(*args)


def test_route_ref_matches_the_oxide_route_twin():
    ids = torch.tensor([[3, 0], [0, 7], [3, -1], [9, 3]])
    ps, seg, seg_e = hm.route_ref(ids, 8, 0)
    se, so, sc, pl = nm.route_twin(ids, 8)
    assert ps[:len(pl)].tolist() == pl  # same order: expert ascending, pair ascending
    assert [int(seg[o]) for o in so[:3]] == sc[:3] and int(seg.sum()) == len(pl)
    assert [int(seg_e[o]) for o in so[:3]] == se[:3]
    assert (seg_e[len(pl):8] == 8).all()  # off-rank pairs sort last, no segment


# ---------------------------------------------------------------------------
# Helion interpret mode
# ---------------------------------------------------------------------------
@pytest.fixture
def helion(monkeypatch):
    mod = pytest.importorskip("helion")
    mm = torch.mm

    def cpu_mm(a, b, *, out_dtype=None, **kw):  # CPU lacks mm.dtype; bf16 products exact in f32
        return mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)

    monkeypatch.setattr(torch, "mm", cpu_mm)
    return mod


def _case(m, e=6, hd=128, i=64, k=3, act="silu", limit=None, eg=None):
    c = {"experts": e, "hidden": hd, "moe_intermediate": i, "topk": k, "act": act,
         "numtokens": m}
    c.update({} if limit is None else {"limit": limit}, **({} if eg is None else
                                                            {"experts_global": eg}))
    return c


@pytest.mark.parametrize("case", [_case(5), _case(1, act="gelu_tanh"),
                                  _case(24, e=3, k=2, i=128, act="gelu_tanh"),
                                  # DeepSeek-V4-like: clamp engaging, I % 64 != 0, EP rank
                                  _case(7, e=4, k=3, i=48, limit=0.3, eg=16),
                                  _case(3, e=20, k=2, i=80, limit=0.3),
                                  # I = 288 (18 NVFP4 blocks: masked tails), EP of 32
                                  _case(6, e=4, k=6, i=288, limit=0.3, eg=32)])
@pytest.mark.parametrize("name", [k["name"] for k in hm.KERNELS])
def test_body_matches_reference(helion, case, name):
    kern = next(k for k in hm.KERNELS if k["name"] == name)
    args = kern["make"](case, 0)
    ref = kern["reference"](*args)
    ok, msg = kern["check"](helion.kernel(kern["body"])(*args), ref)
    assert ok, msg


@pytest.mark.parametrize("plan", sorted(hm.PLANS))
@pytest.mark.parametrize("local,e_global,base,topk,m,act,dead,idim,limit", [
    (4, 16, 8, 3, 7, "silu", None, 64, None),     # EP rank: off-rank pairs + all-off-rank rows
    (6, 6, 0, 2, 12, "gelu_tanh", None, 64, None),
    (4, 16, 8, 3, 5, "silu", True, 64, None),     # every token off-rank: exact zeros
    (4, 16, 8, 3, 7, "silu", None, 48, 0.3),      # DS-like EP: clamp, I % 64 != 0
    (20, 20, 0, 2, 6, "silu", None, 80, 0.3),     # DS-like no EP
    (6, 48, 24, 6, 9, "silu", None, 288, 0.3),    # DS EP8-like: top-6, I 288 tails
    (6, 48, 24, 6, 4, "silu", True, 288, 0.3)])   # ... every token off-rank
def test_plan_matches_spec(helion, plan, local, e_global, base, topk, m, act, dead, idim,
                           limit):
    p = nm.make_problem(local, 128, idim, act, seed=m, limit=limit)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local,
                              dead=dead)
    x = torch.randn(m, 128).bfloat16()
    bodies = {k["name"]: k["body"] for k in hm.KERNELS}
    kerns = [helion.kernel(bodies[n]) for n in hm.PLANS[plan]]
    out = torch.full((m, 128), float("nan")).bfloat16()
    hm.run_plan(kerns, plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                p["g2"], out, p["a1g"], p["a2g"], p["act"], base, limit)
    lid = nm.to_local(ids, base, local)
    ref = nm.moe_ref(p, x, lid, tw)
    if limit is not None and not dead:  # the clamp engages; the unclamped spec is NOT matched
        assert nm.clamp_frac(p, x, lid) > 0.02
        assert hm._rel(out.float(), nm.moe_ref({**p, "limit": None}, x, lid, tw)) > nm.REF_TOL
    dz = (lid < 0).all(1)
    assert bool((out[dz] == 0).all()) and torch.isfinite(out.float()).all()
    if not dead:
        assert nm.oracle_ok(hm._rel(out.float(), ref), 0.0, 0.0)  # <= REF_TOL vs f64 spec


def test_clamp_is_one_sided_on_the_gate(helion, monkeypatch):
    """The fc1 body clamps gate from above only (up both sides): a reference with a
    symmetric gate clamp must NOT match, the vLLM one must."""
    case = _case(6, e=4, k=3, i=48, limit=0.3, eg=16)
    kern = next(k for k in hm.KERNELS if k["name"] == hm.FC1)
    args = kern["make"](case, 0)
    out = kern["body"] and helion.kernel(kern["body"])(*args)
    ok, msg = kern["check"](out, kern["reference"](*args))
    assert ok, msg

    def symmetric(gu, limit):
        i = gu.shape[-1] // 2
        return torch.cat([gu[..., :i].clamp(-limit, limit), gu[..., i:].clamp(-limit, limit)], -1)

    monkeypatch.setattr(nm, "clamp_ref", symmetric)
    ok, _ = kern["check"](out, kern["reference"](*args))
    assert not ok


@pytest.mark.parametrize("dist", nm.ROUTING_DISTS)
@pytest.mark.parametrize("adversarial", [False, True])
def test_plan_under_load_and_scale_stress(helion, dist, adversarial):
    """front4 / fused3 on a DS EP8-like rank: skewed / hot expert loads (row loops of up to
    M rows) and adversarial block scales (zero, e4m3-subnormal, flushed, saturated,
    outlier blocks) in weights AND activations, clamp on."""
    local, e_global, base, topk, m, idim = 6, 48, 24, 6, 20, 288
    p = nm.make_problem(local, 128, idim, "silu", seed=11, limit=0.5, adversarial=adversarial)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local,
                              dist=dist)
    x = torch.randn(m, 128)
    if adversarial:
        x = nm.adversarial_blocks(x, torch.Generator().manual_seed(7))
    x = x.bfloat16()
    lid = nm.to_local(ids, base, local)
    ref = nm.moe_ref(p, x, lid, tw)
    for plan in sorted(hm.PLANS):
        bodies = {k["name"]: k["body"] for k in hm.KERNELS}
        kerns = [helion.kernel(bodies[n]) for n in hm.PLANS[plan]]
        out = torch.full((m, 128), float("nan")).bfloat16()
        hm.run_plan(kerns, plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"],
                    p["w2_sf"], p["g2"], out, p["a1g"], p["a2g"], p["act"], base, 0.5)
        assert torch.isfinite(out.float()).all()
        # measured 1.5e-3..1.7e-3 on every dist x {plain, adversarial} (CPU, 2026-10-19)
        assert hm._rel(out.float(), ref) <= nm.REF_TOL, (plan, hm._rel(out.float(), ref))


@pytest.mark.parametrize("static", [True, False])  # the lab binds static, serving compiles dynamic
@pytest.mark.parametrize("name", [k["name"] for k in hm.KERNELS])
def test_search_spaces_are_bounded_and_codegen_is_clean(helion, name, static):
    """K2: every tunable block size of every body has a literal [min, max] (no 5120-wide
    or 8192-row tiles for the autotuner to time out on), at the real DS EP8 shapes; the default,
    the all-min and the all-max config each generate valid, state-free Triton (the Helion front
    end through to code, no GPU)."""
    from helion.runtime.ref_mode import RefMode  # interpret mode has no config space
    # 2 local experts keep the CPU weight build short; H, I, topk, M are the real ones
    case = {"experts": 2, "hidden": 5120, "moe_intermediate": 2304, "topk": 6, "act": "silu",
            "limit": 10.0, "experts_global": 16, "numtokens": 192}
    cap = {hm.ROUTE_QUANT: 128, hm.FC1: 64, hm.FC2: 256, hm.COMBINE: 512, hm.FC2_COMBINE: 128}
    kern = next(k for k in hm.KERNELS if k["name"] == name)
    bound = helion.kernel(kern["body"], static_shapes=static, ref_mode=RefMode.OFF).bind(
        kern["make"](case, 0))
    sizes = list(bound.config_spec.block_sizes)
    assert sizes and all(sp.max_size <= cap[name] for sp in sizes), \
        (name, [(sp.min_size, sp.max_size) for sp in sizes])
    base = dict(bound.config_spec.default_config().config)
    for pick in (lambda sp: sp.min_size, lambda sp: sp.max_size, None):
        cfg = helion.Config(**(base if pick is None else
                               {**base, "block_sizes": [pick(sp) for sp in sizes]}))
        bound.config_spec.normalize(cfg)
        code = bound.to_triton_code(cfg)
        compile(code, name, "exec")
        for word in ("atomic", "barrier", "semaphore"):
            assert word not in code, (name, word)


def test_in_lab_tuner_plan_and_config_files(tmp_path, monkeypatch):
    """nvfp4_moe_tune (the DEV bundle's nvfp4_moe_bench gate): the todo list covers every
    (kernel, case) once, EP8 (prod) first, and the config files it writes load through
    helion_kernels' own table reader under the exact serving case keys."""
    from suffix_hybrid.helion_kernels import nvfp4_moe_tune as nt
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    plan = nt.todo()
    assert len(plan) == len({(k, hk.case_key(c)) for k, c in plan}) == 53  # 34 routed + 19 fused
    assert all("experts_global" in c for _, c in plan)          # EP8 only (prod)
    names = [k for k, _ in plan]
    assert {k: names.count(k) for k in set(names)} == {
        "nvfp4_moe_route_quant": 8, "nvfp4_moe_fc1": 8, "nvfp4_moe_fc2": 7, "nvfp4_moe_combine": 7,
        "nvfp4_moe_fc2_combine": 4, "dsv41_moe_fc1_sh": 8, "dsv41_moe_fc2_combine_sh": 4,
        "dsv41_moe_combine_sh": 7}
    assert all(("sh_inter" in c) == k.startswith("dsv41_moe_") for k, c in plan)
    monkeypatch.setenv("NVFP4_MOE_TP8_NOEP", "1")
    both = nt.todo()
    assert len(both) == 106 and both[:53] == plan               # EP8 (prod) first
    plan = both
    assert plan[0][0] == hm.ROUTE_QUANT and plan[0][1]["numtokens"] == 8  # key M first
    monkeypatch.setattr(nt, "ROOT", str(tmp_path))
    rows = [{"kernel": k, "case": c, "effort": "quick", "platform": "p", "status": "ok",
             "helion_us": 10.0 + i, "config": {"block_sizes": [i]}} for i, (k, c) in enumerate(plan)]
    rows.append({**rows[0], "helion_us": 5.0, "config": {"block_sizes": [99]}})   # better row wins
    rows.append({**rows[1], "helion_us": 1.0, "status": "incorrect"})              # never written
    nt.write_configs(rows)
    kern, case = plan[0]
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path / "configs"))
    hk._TABLES.clear()
    monkeypatch.setattr(hk, "platform", lambda: "p")
    assert hk.config(kern, case) == {"block_sizes": [99]}
    assert hk.config(plan[1][0], plan[1][1]) == {"block_sizes": [1]}
    hk._TABLES.clear()


def test_autotune_check_compares_only_what_a_consumer_reads(helion):
    """Helion's autotuner compares every candidate with the default config's output over the WHOLE
    tensor. Rows of off-rank pairs, the trash row P and the trash slot of the route arrays are
    garbage that differs between configs: the lab's accuracy check masks them (and still
    rejects a real difference)."""
    case = {"experts": 6, "hidden": 128, "moe_intermediate": 48, "topk": 3, "act": "silu",
            "numtokens": 20, "limit": 0.3, "experts_global": 24}
    kerns = {k["name"]: k for k in hm.KERNELS}
    assert all("autotune_baseline_accuracy_check_fn" in k["settings"] for k in hm.KERNELS)
    # route_quant: the trash slot differs, a routing entry does not
    k = kerns[hm.ROUTE_QUANT]
    args = k["make"](case, 0)
    want = helion.kernel(k["body"])(*args)
    got = tuple(t.clone() for t in want)
    check = k["settings"]["autotune_baseline_accuracy_check_fn"]
    for t in got[1:]:
        t[-1] += 7  # the racy trash slot P
    check(got, want)
    got[2][0] += 1
    with pytest.raises(AssertionError, match="routing differs"):
        check(got, want)
    # fc1: garbage in the rows no consumer reads is ignored, one flipped local row is not
    k = kerns[hm.FC1]
    args = k["make"](case, 0)
    want = helion.kernel(k["body"])(*args)
    ps, seg_e = args[1], args[3]
    local = ps[:-1][seg_e[:-1] < 6].long()
    off = torch.ones(want.shape[0], dtype=torch.bool)
    off[local] = False
    got = want.clone()
    got[off] = torch.nan
    check = k["settings"]["autotune_baseline_accuracy_check_fn"]
    check(got, want)
    got[local] = want[local] * 0 + 1.0
    with pytest.raises(AssertionError, match="values differ"):
        check(got, want)
    # the fused-shared fc1 additionally compares hs
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    k = next(x for x in hs.KERNELS if x["name"] == hs.FC1_SH)
    args = k["make"]({**case, "sh_inter": 64}, 0)
    want = helion.kernel(k["body"])(*args)
    check = k["settings"]["autotune_baseline_accuracy_check_fn"]
    check((want[0].clone(), want[1].clone()), want)
    with pytest.raises(AssertionError, match="rel"):
        check((want[0], want[1] * 1.1), want)
