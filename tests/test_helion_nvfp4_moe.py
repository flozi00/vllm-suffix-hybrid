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


def test_cases_per_model():
    got = hm.cases(GLM_TP8)
    assert [c["numtokens"] for c in got] == list(hm.BUCKETS) and len(got) <= 64
    assert got[0] == {"experts": 256, "hidden": 6144, "moe_intermediate": 256, "topk": 8,
                      "act": "silu", "numtokens": 1}
    gemma = {**GLM_TP8, "architecture": "Gemma4ForConditionalGeneration", "experts": 128,
             "hidden": 2816, "moe_intermediate": 704}
    assert hm.cases(gemma)[0]["act"] == "gelu_tanh"
    for bad in ({"quant_format": "bf16"}, {"experts": 512}, {"moe_intermediate": 96},
                {"topk": None}):
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
        st = k["settings"]  # lab-only: eager baseline, lab check as the tuner's check
        assert st["autotune_baseline_fn"] is k["reference"] and "autotune_log_level" in st


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
    assert auto[1] == (1, "fused3") and auto[12] == (12, "front4")
    assert not set(auto) & {7, 8}  # front4 bucket 8 lacks fc2 -> oxide there
    with pytest.raises(RuntimeError, match="M<=8 front4: nvfp4_moe_fc2"):
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


def _case(m, e=6, hd=128, i=64, k=3, act="silu"):
    return {"experts": e, "hidden": hd, "moe_intermediate": i, "topk": k, "act": act,
            "numtokens": m}


@pytest.mark.parametrize("case", [_case(5), _case(1, act="gelu_tanh"),
                                  _case(24, e=3, k=2, i=128, act="gelu_tanh"),
                                  _case(3, e=8, hd=2816, i=704, k=8, act="gelu_tanh")])  # gemma H, I
@pytest.mark.parametrize("name", [k["name"] for k in hm.KERNELS])
def test_body_matches_reference(helion, case, name):
    kern = next(k for k in hm.KERNELS if k["name"] == name)
    args = kern["make"](case, 0)
    ref = kern["reference"](*args)
    out = helion.kernel(kern["body"])(*args)
    ok, msg = kern["check"](out, ref)
    assert ok, msg
    kern["settings"]["autotune_baseline_accuracy_check_fn"](out, ref)  # the tuner accepts it


@pytest.mark.parametrize("plan", sorted(hm.PLANS))
@pytest.mark.parametrize("local,e_global,base,topk,m,act,dead", [
    (4, 16, 8, 3, 7, "silu", None),        # EP rank: off-rank pairs + all-off-rank rows
    (6, 6, 0, 2, 12, "gelu_tanh", None),
    (4, 16, 8, 3, 5, "silu", True)])       # every token off-rank: exact zeros
def test_plan_matches_spec(helion, plan, local, e_global, base, topk, m, act, dead):
    p = nm.make_problem(local, 128, 64, act, seed=m)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local,
                              dead=dead)
    x = torch.randn(m, 128).bfloat16()
    bodies = {k["name"]: k["body"] for k in hm.KERNELS}
    kerns = [helion.kernel(bodies[n]) for n in hm.PLANS[plan]]
    out = torch.full((m, 128), float("nan")).bfloat16()
    hm.run_plan(kerns, plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                p["g2"], out, p["a1g"], p["a2g"], p["act"], base)
    lid = nm.to_local(ids, base, local)
    ref = nm.moe_ref(p, x, lid, tw)
    dz = (lid < 0).all(1)
    assert bool((out[dz] == 0).all()) and torch.isfinite(out.float()).all()
    if not dead:
        assert nm.oracle_ok(hm._rel(out.float(), ref), 0.0, 0.0)  # <= REF_TOL vs f64 spec
