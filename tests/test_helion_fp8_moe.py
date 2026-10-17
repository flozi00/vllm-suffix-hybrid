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
    (k,) = hf.KERNELS
    assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"]) and k["name"] == hf.KERNEL
    assert k["body"] is hf.fp8_moe and k["body"].__module__ == hf.__name__
    assert not hasattr(k["body"], "bind")  # undecorated: the runner wraps it
    for key in ("cases", "make", "reference", "check", "baseline", "make_baseline"):
        assert callable(k[key]), key
    assert k["baseline_name"] == "oxide fp8_moe"
    src = (ROOT / "kernels-oxide" / "fp8_moe" / "src" / "main.rs").read_text()
    entries = re.findall(r"#\[kernel\]\s*#\[launch_bounds\(\d+\)\]\s*pub unsafe fn (\w+)", src)
    assert len(entries) == 8
    rx = [re.compile(r, re.I) for r in k["replaces"]]
    assert all(any(r.search(e) for r in rx) for e in entries)
    assert not any(r.search(n) for r in rx for n in ("moe_fc1_kernel", "fused_moe_kernel"))


def test_case_key_matches_serving():
    cfg = {"E": 64, "H": 2048, "I": 1536, "topk": 4}
    assert hk.case_key(hf.layer_case(cfg, 8)) == hk.case_key(hf.cases(LFM)[3])


def test_warm_covers_every_served_bucket():
    for mm in (1, 3, 8, 32, 48, 64, 200):
        warm = hf.warm_ms(mm)
        assert 1 in warm and max(warm) <= mm
        assert {hk.bucket(m, hf.BUCKETS) for m in warm} == set(hf.served_buckets(mm))


def test_launch_args_contract():
    x = torch.zeros(3, 256, dtype=torch.bfloat16)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    w13 = torch.zeros(4, 512, 256, dtype=torch.float8_e4m3fn)
    a = hf.launch_args(x, ids, None, w13, None, None, None, None, False, 5)
    assert a[8:] == (5, 4, 0.0, fm.MIN_SCALE, 0.0)
    a = hf.launch_args(x[:1], ids[:1], None, w13, None, None, None, None, True, 0)
    assert a[8:] == (0, 2, fm.EPS, 0.0, 1.0)


class _Native:
    def __init__(self):
        self.calls = []

    def fp8_moe_cuda(self, *args):
        self.calls.append(args)


def _serving(monkeypatch, tuned, mode):
    cfg = {"E": 4, "H": 256, "I": 256, "topk": 2, "max_m": 32}
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [16]} if case["numtokens"] in tuned else None)
    calls = []
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case, **kw:
                        (lambda *a: calls.append((case["numtokens"], a))))
    native = _Native()
    from suffix_hybrid import oxide_kernels
    monkeypatch.setattr(oxide_kernels, "native", lambda: native)
    monkeypatch.setenv(fm.HELION_ENV, mode)
    return cfg, calls, native


def test_auto_dispatches_per_bucket(monkeypatch):
    cfg, calls, native = _serving(monkeypatch, {1, 8}, "auto")
    op = fm.helion_op(cfg)
    assert sorted(op.kernels) == [1, 8]
    x = torch.zeros(5, 256, dtype=torch.bfloat16)
    ids = torch.zeros(5, 2, dtype=torch.int32)
    w13 = torch.zeros(4, 512, 256, dtype=torch.float8_e4m3fn)
    tail = ("aq", "as", "inter", "hq", "hs", "y", "route", "out", False, 0, 2, 123)
    op(x, ids, "tw", w13, "s13", "w2", "s2", *tail)  # M=5 -> bucket 8: Helion
    assert calls[-1][0] == 8 and calls[-1][1][9] == 4  # slots = min(E, M*topk)
    x3, ids3 = x[:3], ids[:3]
    op(x3, ids3, "tw", w13, "s13", "w2", "s2", *tail)  # M=3 -> bucket 4: oxide, same args
    want = (x3, ids3, "tw", w13, "s13", "w2", "s2", *tail)
    assert len(native.calls) == 1 and all(a is b for a, b in zip(native.calls[0], want, strict=True))
    assert len(calls) == 1 and op.kernel(1) is not None and op.kernel(16) is None


def test_auto_without_configs_stays_oxide(monkeypatch):
    cfg, _, _ = _serving(monkeypatch, set(), "auto")
    assert fm.helion_op(cfg) is None


def test_helion_missing_bucket_fails_at_load(monkeypatch):
    cfg, _, _ = _serving(monkeypatch, {1, 2, 4, 8, 16}, "helion")  # 32 missing
    with pytest.raises(RuntimeError, match=r"no tuned fp8_moe config .*\[32\]"):
        fm.helion_op(cfg)
    cfg, _, native = _serving(monkeypatch, set(hf.BUCKETS), "helion")
    assert fm.helion_op(cfg).native is None  # helion: the oxide op is never called


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
# the body in Helion's eager reference mode (CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,ue8m0,ep", [(1, False, False), (3, False, False), (5, True, False),
                                        (6, False, True), (4, True, True)])
def test_body_interpret_matches_spec(m, ue8m0, ep):
    helion = pytest.importorskip("helion")
    e_local, hdim, idim, topk = 4, 256, 256, 2
    e_global, base = (8, 4) if ep else (4, 0)
    p = fm.make_problem(e_local, hdim, idim, seed=m, ue8m0=ue8m0)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=e_local)
    x = torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)).bfloat16()
    out = torch.full((m, hdim), float("nan"), dtype=torch.bfloat16)
    args = hf.launch_args(x, ids, tw, p["w13"], p["w13_s"], p["w2"], p["w2_s"], out, ue8m0, base)
    x0 = x.clone()
    got = helion.kernel(hf.fp8_moe)(*args)
    assert torch.equal(x, x0)
    lid = nm.to_local(ids, base, e_local)
    ref = fm.moe_ref(p, x, lid, tw)
    ok, msg = hf.check(got, ref)
    assert ok, msg
    dead = (lid < 0).all(1)
    assert bool((got[dead] == 0).all())
    if ep:
        assert bool(dead.any())  # rand_routing EP rows t % 3 == 1 route off-rank


def test_body_interpret_matches_lab_reference():
    helion = pytest.importorskip("helion")
    case = {"experts": 4, "hidden": 256, "intermediate": 256, "topk": 2, "numtokens": 3}
    p = fm.make_problem(4, 256, 256, seed=7)
    ids, tw = nm.rand_routing(3, 4, 2, "cpu", seed=7)
    x = torch.randn(3, 256).bfloat16()
    out = torch.empty(3, 256, dtype=torch.bfloat16)
    args = hf.launch_args(x, ids, tw, p["w13"], p["w13_s"], p["w2"], p["w2_s"], out, False, 0)
    assert case["numtokens"] == args[0].shape[0]
    ok, msg = hf.check(helion.kernel(hf.fp8_moe)(*args), hf.reference(*args))
    assert ok, msg
