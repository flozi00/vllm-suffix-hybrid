# SPDX-License-Identifier: Apache-2.0
"""dsv41_router: Helion gate GEMV + the resident (fused) negative control.

K3 on CPU: bodies (Helion ref-mode stand-in / real Helion interpret mode when importable) vs
the float64 reference, and that reference vs a literal torch transcription of vLLM's
dsv4_topk Triton kernel. Lowering: no atomics, no grid sync (the semaphore split-K fusion is
banned)."""
import json
import os
import re

os.environ.setdefault("HELION_INTERPRET", "1")

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_router as rt  # noqa: E402
from test_dsv41_stub import backends, lower_to_triton, run_body  # noqa: E402

IMG = rt.IMAGE_LO


@pytest.fixture(autouse=True)
def cpu_mm_out_dtype(monkeypatch):
    """Helion's ref-mode hl.dot calls torch.mm(out_dtype=f32), which CPU torch lacks."""
    orig = torch.mm

    def mm(a, b, out_dtype=None):
        if out_dtype is not None and a.dtype != out_dtype:
            return orig(a.to(out_dtype), b.to(out_dtype))
        return orig(a, b)
    monkeypatch.setattr(torch, "mm", mm)


def small(m, e=40, k=256, seed=0, pad=True, vl=True, scale=0.05, ties=False):
    x, w, bias, bvl, ids, padm = rt.make_problem(m, seed, "cpu", experts=e, hidden=k, with_pad=pad,
                                                  scale=scale)
    if ties:      # experts 3 and 7 identical -> exact score ties, lowest id must win
        w[7] = w[3]
        bias[7] = bias[3]
    return rt.resident_args(x, w, bias, bvl if vl else None, ids, padm if pad else None, 1.5,
                            IMG if vl else 0)


def run(monkeypatch, body, args, backend="stub", blocks=None):
    return run_body(monkeypatch, rt, body, args, blocks, backend)


def blocks_of(backend, *vals):
    return dict(enumerate(vals)) if backend == "stub" else {"block_sizes": list(vals)}


def dsv4_topk_literal(logits, bias, scale, pad=None, ids=None, bias_vl=None, lo=0):
    """vLLM _dsv4_topk_kernel, row by row, in f32."""
    out_w, out_i = [], []
    for r in range(logits.shape[0]):
        b = bias.clone()
        if bias_vl is not None and lo > 0 and lo <= int(ids[r]) < lo + 5:
            b = bias_vl.clone()
        lg = logits[r].float()
        weights = torch.sqrt(torch.where(lg > 20.0, lg, torch.log(1.0 + torch.exp(lg))))
        cur = weights + b
        cur = torch.where(cur == cur, cur, torch.tensor(-1e30))
        sw, si = [], []
        for _ in range(6):
            mx = cur.max()
            eid = int(torch.nonzero(cur == mx).min())
            sw.append(weights[eid])
            si.append(eid)
            cur[eid] = float("-inf")
        sw = torch.stack(sw)
        s = sw.sum()
        sw = sw * (scale / (s if s > 0 else 1.0))
        if pad is not None and bool(pad[r]):
            sw, si = torch.zeros(6), [-1] * 6
        out_w.append(sw)
        out_i.append(torch.tensor(si, dtype=torch.int32))
    return torch.stack(out_w), torch.stack(out_i)


def test_reference_matches_stock_kernel_formulas():
    a = small(9, seed=4)
    x, w, bias, bvl, ids, pad = a[:6]
    logits = (x.double() @ w.double().T).float()
    sw, si = dsv4_topk_literal(logits, bias, 1.5, pad != 0, ids, bvl, IMG)
    ok, msg = rt.judge((sw, si), rt.reference(*a))
    assert ok, msg
    rw, ri, _ = rt.reference(*a)
    assert (ri[-1] == -1).all() and (rw[-1] == 0).all()


# ---------------------------------------------------------------------------
# GEMV
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("m,e,k,bm,bn,bk", [(1, 40, 256, 16, 16, 32), (6, 40, 256, 16, 32, 64),
                                            (20, 40, 256, 32, 16, 256), (37, 384, 512, 16, 128, 128),
                                            (5, 24, 96, 16, 16, 32)])
def test_gemv_body(monkeypatch, backend, m, e, k, bm, bn, bk):
    x, w, *_ = rt.make_problem(m, m, "cpu", experts=e, hidden=k)
    got = run(monkeypatch, rt.dsv41_gate_gemv, (x, w), backend, blocks_of(backend, bm, bn, bk))
    ok, msg = rt.judge_logits(got, x, w)
    assert ok, msg
    assert got.shape == (m, e) and got.dtype == torch.float32


def test_gemv_is_deterministic_across_launches(monkeypatch):
    """All 8 TP ranks recompute the router: the kernel must give identical bits every launch."""
    x, w, *_ = rt.make_problem(7, 3, "cpu", experts=40, hidden=256)
    a = run(monkeypatch, rt.dsv41_gate_gemv, (x, w), blocks={0: 16, 1: 16, 2: 64})
    b = run(monkeypatch, rt.dsv41_gate_gemv, (x, w), blocks={0: 16, 1: 16, 2: 64})
    assert torch.equal(a, b)


def test_gemv_wrapper_checks_and_warm(monkeypatch):
    x, w, *_ = rt.make_problem(6, 0, "cpu", experts=40, hidden=256)
    with pytest.raises(ValueError, match="need bf16"):
        rt.gate_logits(x.float(), w)
    with pytest.raises(ValueError, match="contiguous"):
        rt.gate_logits(x[:, :128], w)
    monkeypatch.setattr(rt, "_WARM", set())
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, c: (lambda *a: run_body(
        monkeypatch, rt, body, a, {0: 16, 1: 16, 2: 64})))
    out = rt.gate_logits(x, w)
    assert rt.judge_logits(out, x, w)[0]
    monkeypatch.setattr(rt, "_WARM", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="first used while capturing"):
        rt.gate_logits(x, w)


# ---------------------------------------------------------------------------
# resident (fused) router
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("m,bt,bk", [(1, 16, 32), (5, 16, 64), (16, 16, 256), (37, 16, 64), (40, 32, 32)])
@pytest.mark.parametrize("pad,vl", [(True, True), (False, False)])
def test_resident_body(monkeypatch, backend, m, bt, bk, pad, vl):
    a = small(m, seed=m, pad=pad, vl=vl)
    got = run(monkeypatch, rt.dsv41_gate_topk_resident, a, backend, blocks_of(backend, bt, bk))
    ok, msg = rt.judge(got, rt.reference(*a))
    assert ok, msg
    assert got[0].dtype == torch.float32 and got[1].dtype == torch.int32
    assert got[0].shape == (m, 6) and got[1].shape == (m, 6)


def test_ties_pick_lowest_expert_id(monkeypatch):
    a = small(6, seed=1, ties=True, pad=False, vl=False)
    got = run(monkeypatch, rt.dsv41_gate_topk_resident, a, blocks={0: 16, 1: 32})
    for row in got[1].tolist():
        assert not (7 in row and 3 not in row), row     # 3 and 7 tie: 3 first, 7 only after it
        if 3 in row and 7 in row:
            assert row.index(3) < row.index(7)


def test_softplus_large_logit_branch(monkeypatch):
    a = small(4, seed=2, scale=0.5, pad=False, vl=False)     # logits >> 20 on several experts
    assert float((a[0].double() @ a[1].double().T).max()) > 20.0
    got = run(monkeypatch, rt.dsv41_gate_topk_resident, a, blocks={0: 16, 1: 64})
    ok, msg = rt.judge(got, rt.reference(*a))
    assert ok, msg
    assert float(got[0].sum(1).sub(1.5).abs().max()) < 1e-5     # renormalised to routed_scale


def test_full_v41_shape(monkeypatch):
    """E = 384 (padded to 512), K = 5120, image rows, a padding row."""
    x, w, bias, bvl, ids, pad = rt.make_problem(3, 5, "cpu")
    a = rt.resident_args(x, w, bias, bvl, ids, pad, 1.5, IMG)
    got = run(monkeypatch, rt.dsv41_gate_topk_resident, a, blocks={0: 16, 1: 128})
    ok, msg = rt.judge(got, rt.reference(*a))
    assert ok, msg
    assert got[1].max() < 384 and got[1][:-1].min() >= 0
    lg = run(monkeypatch, rt.dsv41_gate_gemv, (x, w), blocks={0: 16, 1: 128, 2: 256})
    assert rt.judge_logits(lg, x, w)[0]


def test_judge_catches_errors(monkeypatch):
    a = small(6, seed=3)
    got = run(monkeypatch, rt.dsv41_gate_topk_resident, a, blocks={0: 16, 1: 32})
    ref = rt.reference(*a)
    assert rt.judge(got, ref)[0]
    gw, gi = got[0].clone(), got[1].clone()
    gi[0, 0], gi[0, 1] = gi[0, 1].clone(), gi[0, 0].clone()
    assert not rt.judge((gw, gi), ref)[0]
    assert not rt.judge((gw * 1.001, got[1]), ref)[0]
    gi2 = got[1].clone()
    gi2[-1] = 3                      # padding row must be -1
    assert not rt.judge((got[0], gi2), ref)[0]
    lg = run(monkeypatch, rt.dsv41_gate_gemv, a[:2], blocks={0: 16, 1: 16, 2: 32})
    assert rt.judge_logits(lg, a[0], a[1])[0] and not rt.judge_logits(lg * 1.001, a[0], a[1])[0]


# ---------------------------------------------------------------------------
# lowering, search space, ABI
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", [1, 6, 192])
def test_bodies_lower_to_triton_without_cross_cta_sync(t):
    x, w, bias, bvl, ids, pad = rt.make_problem(t, 0, "cpu")
    c = rt.case(t)
    g = lower_to_triton(rt.dsv41_gate_gemv, (x, w), rt.default_config(c, rt.K_GEMV))
    r = lower_to_triton(rt.dsv41_gate_topk_resident,
                        rt.resident_args(x, w, bias, bvl, ids, pad, 1.5, IMG), rt.default_config(c, rt.K_RES))
    for code in (g, r):
        assert "tl.dot" in code
        for banned in ("atomic", "grid_barrier", "while ", "semaphore", "spin", "gdc_"):
            assert banned not in code, banned
    assert "tl.sqrt" in r


def test_search_space_is_the_documented_one():
    pytest.importorskip("helion")
    import helion
    from helion.runtime.ref_mode import RefMode
    x, w, bias, bvl, ids, pad = rt.make_problem(192, 0, "cpu")
    for name, body, args in ((rt.K_GEMV, rt.dsv41_gate_gemv, (x, w)),
                             (rt.K_RES, rt.dsv41_gate_topk_resident,
                              rt.resident_args(x, w, bias, bvl, ids, pad, 1.5, IMG))):
        bound = helion.kernel(body, static_shapes=False, ref_mode=RefMode.OFF).bind(tuple(args))
        got = [(b.min_size, b.max_size) for b in bound.config_spec.block_sizes]
        assert got == list(rt.SEARCH_SPACE[name].values()), name
        cfg = rt.default_config(rt.case(192), name)
        for (lo, hi), v in zip(got, cfg["block_sizes"]):
            assert lo <= v <= hi and v & (v - 1) == 0


def test_buckets_cases_abi():
    assert rt.serving_case(30)["numtokens"] == 48 and rt.serving_case(500)["numtokens"] == 192
    model = {"architecture": "DeepseekV41ForCausalLM", "hidden": 5120}
    got = rt.cases(model)
    assert [c["numtokens"] for c in got] == list(rt.BUCKETS)
    json.dumps(got)
    assert rt.cases({"architecture": "LlamaForCausalLM"}) == []
    assert [e["name"] for e in rt.KERNELS] == [rt.K_GEMV, rt.K_RES]
    for ent in rt.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"]) and ent["body"].__module__ == rt.__name__
        assert ent["body"].__qualname__ == ent["name"]
        assert len(ent["replaces"]) <= 16 and all(len(p) <= 300 for p in ent["replaces"])
    assert any(re.search(p, "_dsv4_topk_kernel", re.I) for p in rt.KERNELS[1]["replaces"])


def test_resident_args_validation():
    x, w, bias, bvl, ids, pad = rt.make_problem(2, 0, "cpu", experts=40, hidden=256)
    with pytest.raises(ValueError, match="need bf16"):
        rt.resident_args(x.float(), w, bias, None, None, None, 1.5)
    a = rt.resident_args(x, w, bias, None, None, None, 1.5, 0)
    assert a[8] == 64 and a[9] is False and a[10] is False       # e_pad of 40 experts, flags
