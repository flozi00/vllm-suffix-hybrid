# SPDX-License-Identifier: Apache-2.0
"""(1) The expert load of an EP8 rank, re-derived: M tokens x top-6 of 384 experts, every rank sees
all M tokens (allgather dispatch), so an expert gets M / 64 rows, not M / 8 (docs/dsv41-moe.md
section 8 carried an 8x error). (2) The body-variant seam to lane I's dot_scaled bodies
(helion_kernels/nvfp4_moe.py "body variants per M bucket"): contract check, per-bucket selection
through the config table, serving dispatch, the lab's per-bucket pick and the tuner's rows, with a
fake variant module standing in for suffix_hybrid.helion_kernels.dsv41_moe_fp4mma."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before Helion is imported

import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_moe as hm  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_moe_tune as nt  # noqa: E402
from suffix_hybrid.kernels import nvfp4_moe as nm  # noqa: E402

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "dsv41-moe.md"
DS = {"name": "deepseek-v4.1-flash-1017-nvfp4", "architecture": "DeepseekV41ForCausalLM",
      "tp": 8, "hidden": 5120, "moe_intermediate": 288, "experts": 384, "topk": 6,
      "quant_format": "nvfp4"}


# ---------------------------------------------------------------------------
# (1) expert load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,rows,hit,per_hit", [
    (1, 1 / 64, 0.75, 1.0), (192, 3.0, 45.7, 3.15), (64, 1.0, 30.5, 1.57)])
def test_expected_load_is_m_over_64(m, rows, hit, per_hit):
    e = nm.expected_load(m, 6, 384, 48)
    assert e["rows_per_expert"] == pytest.approx(rows)
    assert e["hit"] == pytest.approx(hit, abs=0.06)
    assert e["rows_per_hit"] == pytest.approx(per_hit, abs=0.02)
    assert e["local_pairs"] == pytest.approx(0.75 * m)        # 1/8 of the M x 6 pairs
    assert nm.expected_load(0, 6, 384, 48)["rows_per_hit"] == 0.0


def test_balanced_routing_matches_the_formula_and_not_the_old_m_over_8():
    """Monte Carlo over the lane's own balanced router: the mean load of a rank agrees with
    expected_load within 6 % (160 batches), 8x below the old 'avg rows/expert = M x 6 / 48'."""
    for m in (16, 64, 192):
        stats = [nm.load_stats(nm.rand_routing(m, 384, 6, "cpu", seed=s, base=0, local=48,
                                               dead=False)[0], 96, 48) for s in range(160)]
        pairs = sum(s["pairs"] for s in stats) / len(stats)
        hit = sum(s["hit"] for s in stats) / len(stats)
        e = nm.expected_load(m, 6, 384, 48)
        assert pairs == pytest.approx(e["local_pairs"], rel=0.06)
        assert hit == pytest.approx(e["hit"], rel=0.06)
        assert pairs / 48 == pytest.approx(m / 64, rel=0.06)
        assert pairs / 48 < 0.2 * (m * 6 / 48)  # the old column


def test_skewed_and_hot_stress_the_row_loop_balanced_does_not():
    m = 192
    heavy = {d: nm.simulated_load(m, d, seeds=8) for d in nm.ROUTING_DISTS}
    assert heavy["balanced"]["max_rows"] < 16            # ~9: one 16-row dot tile covers it
    assert 40 < heavy["skewed"]["max_rows"] < 130        # ~79: the top expert takes ~0.4 M rows
    assert heavy["hot"]["max_rows"] == m                 # one expert takes every row
    assert heavy["balanced"]["rows_per_hit"] < 4         # ~3.2


def test_doc_load_table_is_the_codes():
    """docs/dsv41-moe.md section 8 pastes `python -m suffix_hybrid.kernels.nvfp4_moe load`; its
    analytical table must be the code's numbers (the simulated one is checked for shape)."""
    text = DOC.read_text()
    want = nm.load_markdown(ms=nm.LOAD_MS).splitlines()
    first = [ln for ln in want[:2 + len(nm.LOAD_MS)]]
    for ln in first:
        assert ln in text, f"docs/dsv41-moe.md lacks the table row {ln!r} (regenerate section 8)"
    rows = re.findall(r"^\| (\d+) \| (balanced|skewed|hot) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) "
                      r"\| (\d+) \|$", text, re.M)
    assert {(int(m), d) for m, d, *_ in rows} == {(m, d) for m in nm.LOAD_MS
                                                  for d in nm.ROUTING_DISTS}
    bal192 = next(r for r in rows if r[0] == "192" and r[1] == "balanced")
    assert float(bal192[3]) == pytest.approx(3.15, abs=0.2) and int(bal192[5]) < 16


# ---------------------------------------------------------------------------
# (2) body variants: a fake dot_scaled module
# ---------------------------------------------------------------------------
def _body(*a):
    return None


def _fake_variant(calls, plans=None, name="dot_scaled", **over):
    plans = plans or {"front4": ("fake_rq", "fake_fc1", "fake_fc2", "fake_comb"),
                      "fused3": ("fake_rq", "fake_fc1", "fake_fc2c")}
    mod = types.ModuleType("fake_dsv41_moe_fp4mma")
    mod.VARIANT = name
    mod.PLANS = plans
    mod.BODIES = {k: _body for names in plans.values() for k in names}

    def run_plan(kerns, plan, x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, out, a1, a2,
                 act, id_base, limit=None):
        calls.append((name, plan, len(kerns), x.shape[0]))
        return out.fill_(7.0)

    mod.run_plan = run_plan
    cases = lambda model, kernel=None: [dict(c) for c in hm.cases(model, hm.FC1)]
    mod.KERNELS = [{"name": k, "body": _body, "cases": cases, "make": _body, "reference": _body,
                    "check": _body, "baseline": _body} for k in mod.BODIES]
    for k, v in over.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def fake_variant(monkeypatch):
    """lane I's module, faked and importable under its real path; `calls` records run_plan."""
    calls = []
    mod = _fake_variant(calls)
    monkeypatch.setitem(sys.modules, hm.VARIANT_MODULES[hm.DOT_SCALED], mod)
    monkeypatch.delitem(hm._VARIANTS, hm.DOT_SCALED, raising=False)
    yield mod, calls
    hm._VARIANTS.pop(hm.DOT_SCALED, None)  # never leak the fake into other tests' lab lists
    hm._kernels.cache_clear()


def _table(monkeypatch, rows):
    """hk.config over {(kernel, bucket): config}; the variant pseudo-kernel included."""
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        rows.get((kernel, case["numtokens"])))


def _tuned(buckets=hm.CLAMP_BUCKETS, variant_kernels=(), only=None):
    rows = {(k, b): {"num_warps": 4} for k in (*hm.PLANS["front4"], *hm.PLANS["fused3"])
            for b in buckets}
    rows.update({(k, b): {"num_warps": 4} for k in variant_kernels for b in (only or buckets)})
    return rows


def test_seam_contract_is_checked(fake_variant):
    mod, _ = fake_variant
    hm.check_variant("dot_scaled", mod)
    for over, needle in (
            ({"VARIANT": "other"}, "VARIANT 'other'"),
            ({"PLANS": {}}, "non-empty dict"),
            ({"PLANS": {"front9": ("a",)}}, "non-empty dict"),
            ({"BODIES": []}, "BODIES must be"),
            ({"run_plan": None}, "run_plan("),
            ({"KERNELS": []}, "KERNELS (lab ABI list) is missing"),
            ({"PLANS": {"front4": ("nobody",)}, "BODIES": {}}, "names 'nobody': no body")):
        bad = _fake_variant([], **over)
        with pytest.raises(RuntimeError, match="breaks the seam") as exc:
            hm.check_variant("dot_scaled", bad)
        assert needle in str(exc.value), (over, str(exc.value))
    bad = _fake_variant([])
    bad.KERNELS[0] = {"name": "x"}
    with pytest.raises(RuntimeError, match=r"lacks \['body'"):
        hm.check_variant("dot_scaled", bad)
    bad = _fake_variant([], plans={"front4": (hm.ROUTE_QUANT,)})  # reuse of a dequant body is fine...
    bad.BODIES = {}
    bad.KERNELS = [{**bad.KERNELS[0], "name": "fake_other"}]
    hm.check_variant("dot_scaled", bad)
    bad.KERNELS = [{**bad.KERNELS[0], "name": hm.FC1}]            # ... a lab kernel under its name is not
    with pytest.raises(RuntimeError, match="collides"):
        hm.check_variant("dot_scaled", bad)


class _FailingImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Makes importing `fullname` raise ModuleNotFoundError(name=missing)."""

    def __init__(self, fullname, missing):
        self.fullname, self.missing = fullname, missing

    def find_spec(self, fullname, path, target=None):
        if fullname == self.fullname:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named {self.missing!r}", name=self.missing)


def test_variant_module_import_failures_are_loud(monkeypatch):
    path = hm.VARIANT_MODULES[hm.DOT_SCALED]
    monkeypatch.delitem(sys.modules, path, raising=False)
    monkeypatch.delitem(hm._VARIANTS, hm.DOT_SCALED, raising=False)
    # lane I's module is not in the build (whatever the worktree holds): dequant only, and a
    # selection row naming it fails closed with a message that says so
    monkeypatch.setattr(sys, "meta_path", [_FailingImport(path, path), *sys.meta_path])
    assert hm.available_variants() == (hm.DEQUANT,)
    with pytest.raises(RuntimeError, match="not part of this build"):
        hm.variant_module(hm.DOT_SCALED)
    with pytest.raises(RuntimeError, match="unknown body variant 'fp8'"):
        hm.variant_module("fp8")
    # a broken dependency INSIDE the module is not 'absent': it surfaces as itself
    monkeypatch.setattr(sys, "meta_path", [_FailingImport(path, "lane_i_dependency"),
                                           *sys.meta_path[1:]])
    with pytest.raises(ModuleNotFoundError, match="lane_i_dependency"):
        hm.available_variants()


def test_variant_rows_and_default(monkeypatch, fake_variant):
    case = hm.lane_case(48, 5120, 2304, 6, 0, 40, 10.0, 384)
    _table(monkeypatch, {(hm.VARIANT_KERNEL, 64): {"body": "dot_scaled"}})
    assert hm.variant_for(case) == hm.DOT_SCALED and hm.variant_for({**case, "numtokens": 32}) \
        == hm.DEQUANT
    _table(monkeypatch, {(hm.VARIANT_KERNEL, 64): {"body": "dequant"}})
    assert hm.variant_for(case) == hm.DEQUANT
    for row, needle in (({"body": "fp8"}, "names body 'fp8'"), ({}, "names body None")):
        _table(monkeypatch, {(hm.VARIANT_KERNEL, 64): row})
        with pytest.raises(RuntimeError, match=needle):
            hm.variant_for(case)


def test_served_ms_picks_the_variant_per_bucket_and_names_gaps(monkeypatch, fake_variant):
    mod, _ = fake_variant
    names = sorted({k for p in mod.PLANS.values() for k in p})
    rows = _tuned(variant_kernels=names)
    rows[(hm.VARIANT_KERNEL, 32)] = {"body": "dot_scaled"}
    rows[(hm.VARIANT_KERNEL, 64)] = {"body": "dot_scaled"}
    _table(monkeypatch, rows)
    mode_for = lambda m: 2 if m <= 8 else 1
    got = hm.served_ms("helion", 48, 5120, 2304, 6, 0, 70, mode_for, 10.0, 384)
    assert got[1] == (1, "fused3", "dequant") and got[8] == (8, "fused3", "dequant")
    assert got[20] == (32, "front4", "dot_scaled") and got[33] == (64, "front4", "dot_scaled")
    assert got[65] == (128, "front4", "dequant")
    # a selected bucket without its variant's configs is a startup error naming variant + kernel
    for k in names:
        rows.pop((k, 64))
    _table(monkeypatch, rows)
    with pytest.raises(RuntimeError, match=r"M<=64 front4/dot_scaled: fake_rq, fake_fc1, fake_fc2, "
                                           r"fake_comb"):
        hm.served_ms("helion", 48, 5120, 2304, 6, 0, 70, mode_for, 10.0, 384)
    # a variant that lacks the plan the tune table asks for is refused, not guessed
    mod.PLANS.pop("front4")
    with pytest.raises(RuntimeError, match="has no 'front4' plan"):
        hm.served_ms("helion", 48, 5120, 2304, 6, 0, 70, mode_for, 10.0, 384)


def test_serving_op_runs_the_buckets_variant(monkeypatch, fake_variant, helion_calls):
    mod, calls = fake_variant
    names = sorted({k for p in mod.PLANS.values() for k in p})
    rows = _tuned(variant_kernels=names)
    rows[(hm.VARIANT_KERNEL, 32)] = {"body": "dot_scaled"}
    _table(monkeypatch, rows)
    op = hm.serving_op(None, "helion", 10.0, 384)
    e, h, i, k, m = 48, 128, 288, 6, 20
    w13 = torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8)
    args = lambda m: (torch.zeros(m, h, dtype=torch.bfloat16), torch.zeros(m, k, dtype=torch.int32),
                      torch.zeros(m, k), w13, None, None, None, None, None, *hm_ws(), torch.empty(
                          m, h, dtype=torch.bfloat16), 1.0, 1.0, 0, 0)
    op(*args(m), 1, 0, 0)                       # bucket 32 -> dot_scaled front4
    op(*args(7), 2, 0, 0)                       # bucket 8 -> dequant fused3
    assert calls == [("dot_scaled", "front4", 4, 20)]
    assert helion_calls == [("fused3", 3, 7)]
    forced = hm.serving_op(None, "helion", 10.0, 384, variant="dequant")
    forced(*args(m), 1, 0, 0)                   # forced variant ignores the table
    assert helion_calls[-1] == ("front4", 4, 20) and len(calls) == 1


def hm_ws():
    return (None,) * 7


@pytest.fixture
def helion_calls(monkeypatch):
    """The dequant variant's run_plan, recorded; kernels are never built (hk.compiled stubbed)."""
    seen = []

    def run_plan(kerns, plan, x, *rest, **kw):
        seen.append((plan, len(kerns), x.shape[0]))
        return rest[9]  # out

    monkeypatch.setattr(hm, "run_plan", run_plan)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case, **kw: (kernel, case["numtokens"]))
    hm._kernels.cache_clear()
    yield seen
    hm._kernels.cache_clear()


def test_missing_stage_configs_of_the_default_variant_stay_errors(monkeypatch):
    rows = _tuned()
    rows.pop((hm.FC2, 8))
    _table(monkeypatch, rows)
    op = hm.serving_op(None, "helion", 10.0, 384)
    e, h, i, k = 48, 128, 288, 6
    args = (torch.zeros(7, h, dtype=torch.bfloat16), torch.zeros(7, k, dtype=torch.int32),
            torch.zeros(7, k), torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8), None, None, None,
            None, None, *hm_ws(), torch.empty(7, h, dtype=torch.bfloat16), 1.0, 1.0, 0, 0, 1, 0, 0)
    with pytest.raises(RuntimeError, match="no tuned .*nvfp4_moe_fc2"):
        op(*args)
    # an auto op on the oxide-free route has no native to fall back to: loud, not AttributeError
    auto = hm.serving_op(None, "auto", 10.0, 384)
    with pytest.raises(RuntimeError, match="no oxide kernel to fall back to"):
        auto(*args)


# ---------------------------------------------------------------------------
# the lab: per-bucket pick, the variant table, the tuner's rows
# ---------------------------------------------------------------------------
def test_pick_variants_dequant_keeps_ties_and_the_best_wins():
    rows = {
        1: {"balanced": (10.0, {"dequant": 8.0, "dot_scaled": 9.0}),
            "skewed": (10.0, {"dequant": 8.5, "dot_scaled": 9.5})},
        32: {"balanced": (60.0, {"dequant": 50.0, "dot_scaled": 49.8}),         # within 1 %: dequant
             "skewed": (60.0, {"dequant": 51.0, "dot_scaled": 50.5})},
        64: {"balanced": (90.0, {"dequant": 80.0, "dot_scaled": 60.0}),         # dot_scaled
             "skewed": (95.0, {"dequant": 85.0, "dot_scaled": 70.0})},
        128: {"balanced": (100.0, {"dequant": 120.0, "dot_scaled": 110.0}),     # both lose to FI
              "skewed": (100.0, {"dequant": 130.0, "dot_scaled": 90.0})},       # (worst dist 110)
    }
    got = nm.pick_variants(rows)
    assert got[1][0] == "dequant" and got[1][3] is True
    assert got[32][0] == "dequant"
    assert got[64] == ("dot_scaled", 70.0, 95.0, True)
    assert got[128][0] == "dot_scaled" and got[128][3] is False
    text, sel = nm.variant_table("EP8", rows)
    assert sel == {64: "dot_scaled", 128: "dot_scaled"}
    assert "M=64" in text and "body=dot_scaled 70.0 us vs flashinfer 95.0 us (WIN)" in text
    assert "(LOSS)" in text
    text, sel = nm.variant_table("EP8", {1: rows[1]})
    assert sel == {} and "every bucket stays on the dequant bodies" in text
    # no FlashInfer baseline (TP8 without EP): the pick still happens, the verdict says so
    nan = float("nan")
    text, sel = nm.variant_table("TP8", {8: {"balanced": (nan, {"dequant": 5.0, "dot_scaled": 3.0}),
                                             "skewed": (nan, {"dequant": 5.0, "dot_scaled": 3.5})}})
    assert sel == {8: "dot_scaled"} and "(no baseline)" in text
    # a variant that has no time for one of the dists is not comparable and is left out
    only = {5: {"balanced": (9.0, {"dequant": 5.0, "dot_scaled": 4.0}),
                "skewed": (9.0, {"dequant": 5.0})}}
    assert nm.pick_variants(only)[5][0] == "dequant"


def test_tuner_tunes_and_ships_the_variants_rows(monkeypatch, tmp_path, fake_variant):
    mod, _ = fake_variant
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    assert [k["name"] for k in nt.variant_kernels()] == [k["name"] for k in mod.KERNELS]
    assert set(nt.kernels()) >= {k["name"] for k in mod.KERNELS} | {k["name"] for k in hm.KERNELS}
    plan = nt.todo()
    names = [k for k, _ in plan]
    fake = {k["name"] for k in mod.KERNELS}
    assert {k: names.count(k) for k in fake} == {k: 8 for k in fake}     # every bucket, like fc1
    assert len(plan) == 53 + 8 * len(fake)
    # the key bucket's whole set is contiguous: dequant, variant, fused-shared kernels of M=8
    m8 = [k for k, c in plan if c["numtokens"] == 8]
    assert m8.index(hm.FC1) < m8.index("fake_fc1") < m8.index(hs.FC1_SH)
    # rows: this bench's non-default picks replace the benched buckets' rows, others stay
    monkeypatch.setattr(nt, "ROOT", str(tmp_path))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    c = lambda b: hm.lane_case(48, 5120, 2304, 6, 0, b, 10.0, 384)
    row = lambda b, v: {"key": c(b), "config": {"body": v}}
    nt.write_variant_rows([row(64, "dot_scaled"), row(128, "dot_scaled")], [c(b) for b in (64, 128)])
    path = tmp_path / "configs" / hm.VARIANT_KERNEL / "gpu.json"
    assert [r["key"]["numtokens"] for r in json.loads(path.read_text())] == [64, 128]
    nt.write_variant_rows([row(32, "dot_scaled")], [c(b) for b in (32, 64)])   # 64 went back to dequant
    got = json.loads(path.read_text())
    assert [(r["key"]["numtokens"], r["config"]["body"]) for r in got] == \
        [(32, "dot_scaled"), (128, "dot_scaled")]
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path / "configs"))
    hk._TABLES.clear()
    assert hm.variant_for(c(32)) == "dot_scaled" and hm.variant_for(c(64)) == "dequant"
    hk._TABLES.clear()


def test_fused_shared_expert_refuses_a_non_dequant_bucket(monkeypatch, fake_variant):
    rows = _tuned(buckets=hm.CLAMP_BUCKETS)
    rows.update({(k, b): {} for k in hs.NEW_KERNELS for b in hm.CLAMP_BUCKETS})
    rows[(hm.VARIANT_KERNEL, 16)] = {"body": "dot_scaled"}
    _table(monkeypatch, rows)
    mode_for = lambda m: 2 if m <= 8 else 1
    hs.served_ms_sh(48, 5120, 2304, 6, 8, mode_for, 10.0, 384, 288)       # up to bucket 8: fine
    with pytest.raises(RuntimeError, match=r"M=9 \(bucket 16\) is served by the 'dot_scaled' body"):
        hs.served_ms_sh(48, 5120, 2304, 6, 12, mode_for, 10.0, 384, 288)
