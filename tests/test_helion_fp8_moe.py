# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/fp8_moe: lab ABI, cases per model, knob /
per-bucket dispatch (pure python), and the body in Helion's eager reference
mode (HELION_INTERPRET=1) on tiny CPU problems vs the lane's f64 spec —
routing/compaction, EP off-rank zeros, both activation-scale modes."""
import os
import pathlib
import re
import sys

os.environ.setdefault("HELION_INTERPRET", "1")  # before anything imports helion

import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import fp8_moe as hf
from suffix_hybrid.kernels import fp8_moe as fm
from suffix_hybrid.kernels import nvfp4_moe as nm

ROOT = pathlib.Path(__file__).resolve().parents[1]
LFM = {"name": "lfm2.5-24b-a2b-fp8", "quant_format": "fp8_block", "block": [128, 128],
       "experts": 64, "topk": 4, "hidden": 2048, "moe_intermediate": 1536, "tp": 1}


def test_cases_per_model():
    got = hf.cases(LFM)
    assert [c["numtokens"] for c in got] == list(hf.BUCKETS) and len(got) <= 64
    assert got[0] == {"experts": 64, "hidden": 2048, "intermediate": 1536, "topk": 4,
                      "numtokens": 1}
    assert hf.cases({**LFM, "quant_format": "bf16"}) == []
    assert hf.cases({**LFM, "block": [1, 128]}) == []
    assert hf.cases({**LFM, "moe_intermediate": 160}) == []  # qwen3.8-flash-next TP4: I/rank 160
    assert hf.cases({**LFM, "experts": 512}) == []  # lane serves E <= 256 per rank
    assert hf.cases({**LFM, "experts": None}) == []  # dense FP8 model


def test_lab_abi():
    names = [k["name"] for k in hf.KERNELS]
    assert names == list(hf.BODIES) and len(set(names)) == 5
    for k in hf.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"])
        assert k["body"] is hf.BODIES[k["name"]] and k["body"].__module__ == hf.__name__
        assert not hasattr(k["body"], "bind")  # undecorated: the runner wraps it
        for key in ("cases", "make", "reference", "check", "baseline", "make_baseline"):
            assert callable(k[key]), key
        assert k["baseline_name"].startswith("oxide fp8_moe moe_")
    src = (ROOT / "kernels-oxide" / "fp8_moe" / "src" / "main.rs").read_text()
    entries = re.findall(r"#\[kernel\]\s*#\[launch_bounds\(\d+\)\]\s*pub unsafe fn (\w+)", src)
    assert len(entries) == 8
    owner = {e: [k["name"] for k in hf.KERNELS if any(re.search(r, e, re.I) for r in k["replaces"])]
             for e in entries}
    assert all(len(v) == 1 for v in owner.values()), owner  # each oxide launch replaced once


def test_cases_follow_the_plan():
    by = {k["name"]: [c["numtokens"] for c in k["cases"](LFM)] for k in hf.KERNELS}
    small = [b for b in hf.BUCKETS if b <= nm.FUSED_MAX_M]
    big = [b for b in hf.BUCKETS if b > nm.FUSED_MAX_M]
    assert by[hf.ROUTE_QUANT] == by[hf.FC1] == list(hf.BUCKETS)
    assert by[hf.FC2_COMBINE] == small and by[hf.FC2] == by[hf.COMBINE] == big


def test_case_key_matches_serving():
    cfg = {"E": 64, "H": 2048, "I": 1536, "topk": 4}
    assert hk.case_key(hf.layer_case(cfg, 8)) == hk.case_key(hf.cases(LFM)[3])


def test_route_ref_matches_oxide_route_twin():
    """The workspace route == moe_route's (nvfp4_moe.route_twin, the CPU twin
    pinned against the kernel source) on what consumers read."""
    ids, _ = nm.rand_routing(7, 8, 3, "cpu", seed=3, base=4, local=4)
    r = hf.route_ref(ids, 4, 4)
    se, so, sc, pl = nm.route_twin(ids, 4, 4)
    want = torch.zeros_like(r)
    want[:4], want[2 * 4:3 * 4] = -1, 0
    want[:len(se)], want[4:4 + len(so)], want[8:8 + len(sc)] = (torch.tensor(v) for v in (se, so, sc))
    want[12:12 + len(pl)] = torch.tensor(pl, dtype=torch.int32)
    assert hf.route_equal(r, want, 4)


class _Native:
    def __init__(self, skips=True):
        self.calls = []
        if skips:
            self.FP8_MOE_SKIP_SHIFT = hf.SKIP_SHIFT

    def fp8_moe_cuda(self, *args):
        self.calls.append(args)


TAIL = ("out", False, 0, 1, 123)  # out, ue8m0, id_base, mode, stream_ptr


def _serving(monkeypatch, tuned, mode, skips=True, max_m=32):
    """tuned(kernel, numtokens) -> bool: which configs are shipped."""
    cfg = {"E": 4, "H": 256, "I": 256, "topk": 2, "max_m": max_m, "fused": True}
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"num_warps": 4} if tuned(kernel, case["numtokens"]) else None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case, **kw: (kernel, case["numtokens"]))
    calls = []
    monkeypatch.setattr(hf, "run_launch", lambda name, kern, *a: calls.append((name, kern)))
    monkeypatch.setenv(fm.HELION_ENV, mode)
    native = _Native(skips)
    return cfg, calls, native


def _call(op, m, native, calls):
    """One op call with a real-sized workspace; -> [(backend, launch or mode word)]."""
    ws = fm.workspace("cpu", m, 2, 256, 256, 4)
    x = torch.zeros(m, 256, dtype=torch.bfloat16)
    ids = torch.zeros(m, 2, dtype=torch.int32)
    w13 = torch.zeros(4, 512, 256, dtype=torch.float8_e4m3fn)
    calls.clear()
    native.calls.clear()
    seq = []
    orig_h, orig_n = hf.run_launch, native.fp8_moe_cuda
    hf.run_launch = lambda name, kern, *a: seq.append(("helion", name))
    native.fp8_moe_cuda = lambda *a: seq.append(("oxide", a[-2]))
    try:
        op(x, ids, "tw", w13, "s13", "w2", "s2", *ws, *TAIL)
    finally:
        hf.run_launch, native.fp8_moe_cuda = orig_h, orig_n
    return seq


def test_auto_mixes_per_launch(monkeypatch):
    # fc1 tuned at buckets 1/2 only (the lab's 2026-10-17 wins), nothing else
    cfg, calls, native = _serving(monkeypatch, lambda k, b: k == hf.FC1 and b <= 2, "auto")
    op = hf.ServingOp(cfg, "auto", native)
    assert sorted(op.plans) == [1, 2] and op.warm_ms == [1, 2]
    skip = lambda *launches: sum(1 << i for i in launches) << hf.SKIP_SHIFT
    assert _call(op, 2, native, calls) == [
        ("oxide", 2 | skip(1, 2)), ("helion", hf.FC1), ("oxide", 2 | skip(0, 1))]
    # untuned M: the whole oxide op with the lane's own mode word
    assert _call(op, 5, native, calls) == [("oxide", 1)]


def test_front4_runs_of_oxide_launches_share_one_call(monkeypatch):
    cfg, calls, native = _serving(monkeypatch, lambda k, b: k == hf.COMBINE and b == 16, "auto")
    op = hf.ServingOp(cfg, "auto", native)
    skip = lambda *launches: sum(1 << i for i in launches) << hf.SKIP_SHIFT
    assert _call(op, 12, native, calls) == [("oxide", 1 | skip(3)), ("helion", hf.COMBINE)]


def test_host_without_skip_mask_mixes_nothing(monkeypatch):
    cfg, calls, native = _serving(monkeypatch, lambda k, b: k == hf.FC1 or (
        b == 16 and k in hf.PLANS["front4"]), "auto", skips=False)
    op = hf.ServingOp(cfg, "auto", native)
    assert sorted(op.plans) == list(range(9, 17)) and "mixing off" in op.summary
    assert _call(op, 12, native, calls) == [("helion", k) for k in hf.PLANS["front4"]]
    assert _call(op, 1, native, calls) == [("oxide", 1)]


def test_auto_without_configs_stays_oxide(monkeypatch):
    cfg, _, native = _serving(monkeypatch, lambda k, b: False, "auto")
    from suffix_hybrid import oxide_kernels
    monkeypatch.setattr(oxide_kernels, "native", lambda: native)
    assert fm.helion_op(cfg) is None


def test_helion_missing_launch_fails_at_load(monkeypatch):
    cfg, _, native = _serving(monkeypatch, lambda k, b: k != hf.FC2 or b != 32, "helion")
    with pytest.raises(RuntimeError, match=r"numtokens 32 front4: fp8_moe_fc2"):
        hf.ServingOp(cfg, "helion", native)
    cfg, calls, native = _serving(monkeypatch, lambda k, b: True, "helion")
    op = hf.ServingOp(cfg, "helion", native)
    assert len(op.plans) == 32 and all(all(op.plans[m][2]) for m in op.plans)
    assert {hk.bucket(m, hf.BUCKETS) for m in op.warm_ms} == set(hf.BUCKETS[:6])
    assert all(b == "helion" for b, _ in _call(op, 7, native, calls))


def test_oxide_never_imports_helion_module(monkeypatch):
    monkeypatch.delenv(fm.HELION_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "suffix_hybrid.helion_kernels.fp8_moe")
    assert fm.helion_op({"E": 4}) is None
    assert "suffix_hybrid.helion_kernels.fp8_moe" not in sys.modules
    monkeypatch.setenv(fm.HELION_ENV, "triton")
    with pytest.raises(RuntimeError):
        fm.helion_op({"E": 4})


def test_run_ours_routes_through_the_op():
    seen = []
    p = {"w13": 1, "w13_s": 2, "w2": 3, "w2_s": 4, "ue8m0": False, "op": lambda *a: seen.append(a)}
    out = fm.run_ours(None, p, "x", "ids", "tw", ("ws",) * 7, 9, out="out", mode=2)
    assert out == "out" and seen[0][:7] == ("x", "ids", "tw", 1, 2, 3, 4)
    assert seen[0][-4:] == (False, 0, 2, 9)


# ---------------------------------------------------------------------------
# the bodies in Helion's eager reference mode (CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("plan", ["front4", "fused3"])
@pytest.mark.parametrize("m,ue8m0,ep", [(1, False, False), (3, False, False), (5, True, False),
                                        (6, False, True), (4, True, True)])
def test_plan_interpret_matches_spec(plan, m, ue8m0, ep):
    """Every launch of a plan in eager reference mode over the oxide
    workspace, end to end vs the f64 spec; each stage bit-checked against the
    spec's quantization (x quant bit-equal, route equal)."""
    helion = pytest.importorskip("helion")
    e_local, hdim, idim, topk = 4, 256, 256, 2
    e_global, base = (8, 4) if ep else (4, 0)
    p = fm.make_problem(e_local, hdim, idim, seed=m, ue8m0=ue8m0)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=e_local)
    x = torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)).bfloat16()
    out = torch.full((m, hdim), float("nan"), dtype=torch.bfloat16)
    ws = fm.workspace("cpu", m, topk, hdim, idim, e_local)
    views = hf.ws_views(ws, m, topk, hdim, idim, e_local)
    x0 = x.clone()
    for name in hf.PLANS[plan]:
        hf.run_launch(name, helion.kernel(hf.BODIES[name]), x, ids, tw, p["w13"], p["w13_s"],
                      p["w2"], p["w2_s"], views, out, ue8m0, base)
        if name == hf.ROUTE_QUANT:
            aq_r, as_r = hf._xq_ref(x, ue8m0)
            assert hf._frac(views[0], aq_r) == 0 and torch.equal(views[1], as_r)
            assert hf.route_equal(views[5], hf.route_ref(ids, e_local, base), e_local)
    assert torch.equal(x, x0)
    lid = nm.to_local(ids, base, e_local)
    ok, msg = hf._check_end(out, fm.moe_ref(p, x, lid, tw))
    assert ok, msg
    dead = (lid < 0).all(1)
    assert bool((out[dead] == 0).all())
    if ep:
        assert bool(dead.any())  # rand_routing EP rows t % 3 == 1 route off-rank


@pytest.mark.parametrize("name", list(hf.BODIES))
def test_lab_entry_interpret(name):
    """Each KERNELS entry: make -> body vs reference through its own check
    (the lab gate), on a tiny CPU case."""
    helion = pytest.importorskip("helion")
    (k,) = [k for k in hf.KERNELS if k["name"] == name]
    m = 3 if name == hf.FC2_COMBINE else 12
    case = {"experts": 4, "hidden": 256, "intermediate": 256, "topk": 2, "numtokens": m}
    args = k["make"](case, 0)
    ok, msg = k["check"](helion.kernel(k["body"])(*args), k["reference"](*args))
    assert ok, msg
