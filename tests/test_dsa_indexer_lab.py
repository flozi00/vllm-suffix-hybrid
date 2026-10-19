# SPDX-License-Identifier: Apache-2.0
"""The silicon lab driver (kernels/dsa_indexer_lab.py): its task plan and its report
(winner table, decode-step totals, crossovers) are pure functions over stored rows,
proven here on synthetic rows; the GPU tasks themselves run in the lab."""
import os

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
pytest.importorskip("torch")
from suffix_hybrid.kernels import dsa_indexer_lab as lab  # noqa: E402


def row(kind, n, reqs, stock, **arms):
    return {"task": ["bench_layer", [kind, n, reqs]], "result": {"kind": kind, "n": n, "reqs": reqs, lab.STOCK: stock, **arms}}


def synthetic(win_from=32768):
    """Stock grows with n; ours is slower at small n and wins from `win_from` on."""
    rows = []
    for kind, n, reqs in lab.points():
        stock = 10.0 + n / 1000
        ours = stock * (0.4 if n >= win_from else 1.5)
        arms = {"row_us": ours * 1.2, "groupS_us": ours, "group_us": ours * 0.9, "oxide_us": ours * 1.1}
        if kind == "dense":
            arms["select_us"] = ours * 0.8
        if kind == "source":
            arms["groupS_candsel_us"] = ours * 0.7
        if kind == "consumer":
            arms["cand_fused_us"] = ours * 0.5
        rows.append(row(kind, n, reqs, stock, **arms))
    return rows


def test_points_cover_the_grid_and_skip_what_cannot_be_resident():
    pts = lab.points()
    assert ("dense", 262144, 1) in pts and ("dense", 524288, 1) not in pts      # dense layers scan ctx / 2
    assert ("source", 524288, 4) in pts and ("consumer", 524288, 32) not in pts  # 32 x 512k K tokens: skipped
    assert ("consumer", 131072, 32) in pts and ("consumer", 8192, 32) in pts
    assert all(reqs * n <= lab.MAX_TOKENS for _, n, reqs in pts)
    assert {k for k, _, _ in pts} == {"dense", "source", "consumer"}


def test_task_plan_orders_oracle_gate_tune_bench_report():
    plan = lab.tasks()
    names = [t[0] for t in plan]
    assert names[:2] == ["oracle", "gate"] and names[-1] == "report"
    tunes = [t[1][0] for t in plan if t[0] == "tune"]
    assert tunes[0] == "dsa_indexer_group" and set(tunes) == set(lab.ARCH_ORDER)
    # one tune task per case of every kernel (group: requests x context buckets up to 524288)
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    by_name = {k["name"]: k for k in hdi.KERNELS}
    for name in lab.ARCH_ORDER:
        assert tunes.count(name) == len(by_name[name]["cases"](lab.MODEL))
    assert max(c["ctx"] for c in by_name["dsa_indexer_group"]["cases"](lab.MODEL)) == 524288
    assert names.index("bench_layer") > max(i for i, n in enumerate(names) if n == "tune")
    pre = [t[1] for t in plan if t[0] == "bench_prefill"]
    assert pre == [list(p) for p in lab.PREFILL_POINTS] and all(m * n <= 134_217_728 * 2 for m, n in pre)
    assert [256, 524288] in pre                       # the deepest chunk vLLM's logits bound allows


def test_report_step_totals_crossovers_winners():
    rep = lab.report(synthetic())
    tot = {(t["ctx"], t["reqs"]): t for t in rep["step_totals"]}
    assert (524288, 1) in tot and (524288, 32) not in tot               # the 32 x 512k point was skipped
    t = tot[(524288, 1)]
    assert t["ours_us"] < t["stock_us"] and t["speedup"] > 1.5
    assert t["arms"] == {"dense": "select_us", "source": "groupS_candsel_us", "consumer": "cand_fused_us"}
    # stock total = 3 dense (ctx / 2) + source + 4 consumers
    rows = lab.index_rows(synthetic())
    want = 3 * rows[("dense", 262144, 1)][lab.STOCK] + rows[("source", 524288, 1)][lab.STOCK] \
        + 4 * rows[("consumer", 524288, 1)][lab.STOCK]
    assert t["stock_us"] == pytest.approx(want, abs=0.1)
    cx = rep["crossovers"]
    assert cx["ours<stock[consumer]@reqs1"] == 8192 and cx["cand_fused<dense_mask@reqs1"] == 8192
    assert cx["ours<stock[source]@reqs4"] == 32768 and cx["ours<stock[dense]@reqs1"] == 32768
    assert cx["group<row@reqs1"] == 4096 and cx["select<group@reqs1"] == 4096
    assert rep["winners"]["consumer/524288/1"][0].startswith("cand_fused_us")
    assert rep["winners"]["dense/4096/1"][0].startswith("stock_us")      # ours loses below the crossover
    assert rep["serving_policy_penalty"]["dense/8192/1"] == pytest.approx(1 / 0.9, abs=1e-3)
    assert set(rep["byte_model_roofline_us"]["524288/1"]) == {"stock_us", "ours_us"}


def test_report_ignores_arms_that_failed_and_missing_points():
    rows = synthetic()
    for r in rows:
        if r["task"][1] == ["consumer", 131072, 1]:
            r["result"]["cand_fused_us"] = "RuntimeError: arm failed"       # a failed arm is not a time
            r["result"]["groupS_us"] = "RuntimeError: arm failed"
    rows = [r for r in rows if r["task"][1] != ["dense", 32768, 1]]         # ctx 65536 needs dense at 32768
    rep = lab.report(rows)
    tot = {(t["ctx"], t["reqs"]): t for t in rep["step_totals"]}
    assert (65536, 1) not in tot and (131072, 1) in tot
    assert tot[(131072, 1)]["arms"]["consumer"] == "row_us"               # the arm that did run
    assert lab.best({"row_us": "boom"}, "dense") is None
