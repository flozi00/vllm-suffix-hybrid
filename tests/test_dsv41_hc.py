# SPDX-License-Identifier: Apache-2.0
"""dsv41_hc: the Helion mHC boundary (suffix_hybrid/helion_kernels/dsv41_hc.py).

K1/K3 on CPU: every body (Helion ref-mode stand-in, and real Helion in interpret mode when
importable) vs the exact float64 reference; that reference vs vLLM's own torch reference
formulas (copied below from vllm/model_executor/kernels/mhc/torch.py @ 39983a5e7); the
bodies lower to Triton with no atomics / barriers / spin loops; the search spaces are the
documented ones."""
import json
import os
import re

os.environ.setdefault("HELION_INTERPRET", "1")

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_common as cm  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_hc as hc  # noqa: E402
from test_dsv41_stub import backends, lower_to_triton, run_body  # noqa: E402

H = 96   # small hidden (the real one is 5120); 3 groups of 32, not a multiple of 64 on purpose


# ---- vLLM's torch reference (mhc_post_torch / mhc_pre_delayed_torch), fp32 ----
def vllm_post(x, residual, post_layer_mix, comb_res_mix):
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)


def vllm_pre_delayed(residual, fn, hc_scale, hc_base, rms_eps, pre_eps, sk_eps, post_mult,
                     repeat, pre_mix=None):
    hcm = residual.shape[1]
    x = residual.flatten(1).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hcm] * hc_scale[0] + hc_base[:hcm]) + pre_eps
    post = torch.sigmoid(mixes[:, hcm:2 * hcm] * hc_scale[1] + hc_base[hcm:2 * hcm]) * post_mult
    comb = mixes[:, 2 * hcm:].view(-1, hcm, hcm) * hc_scale[2] + hc_base[2 * hcm:].view(1, hcm, hcm)
    comb = torch.softmax(comb, dim=-1) + sk_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    layer_input = (residual[:, 0] if pre_mix is None else
                   (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype))
    return post, comb, layer_input, pre


def rmsnorm_bf16(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def problem(t, use_pre_in=True, seed=0, h=H, **consts):
    return hc.make_problem(t, h, seed, "cpu", use_pre_in, consts or None)


def run(monkeypatch, body, args, backend="stub", blocks=None):
    return run_body(monkeypatch, hc, body, args, blocks, backend)


def blocks_of(backend, *vals):
    return dict(enumerate(vals)) if backend == "stub" else {"block_sizes": list(vals)}


# ---------------------------------------------------------------------------
# the reference itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_pre_in", [True, False])
def test_reference_matches_vllm_formulas(use_pre_in):
    P = problem(5, use_pre_in, seed=3)
    ref = hc.reference_args(hc.k0_args(P))
    rc, post_o, comb_o, lin, pre_o = ref[:5]
    rc_v = vllm_post(P.x, P.res, P.post.view(-1, 4, 1), P.comb)
    assert (rc.float() - rc_v.float()).abs().max() <= 2.0 ** -7 * rc.float().abs().max()
    p, cb, li, pr = vllm_pre_delayed(rc, P.fn, P.scale, P.base, 1e-6, 1e-6, 1e-6, 2.0, 20, P.pre_in)
    torch.testing.assert_close(post_o, p.float().view(5, 4), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(comb_o, cb, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pre_o, pr, rtol=1e-4, atol=1e-6)
    assert float(hc._ulp_close(lin, rmsnorm_bf16(li, P.norm_w, 1e-6)).float().mean()) > 0.995


# ---------------------------------------------------------------------------
# K1: post + split-K partials
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("t,bt,bh,ns", [(1, 1, 32, 1), (5, 2, 32, 4), (6, 4, 64, 2), (17, 4, 32, 8),
                                        (7, 1, 32, 16), (3, 2, 64, 4)])
def test_post_partials_body(monkeypatch, backend, t, bt, bh, ns):
    P = problem(t, seed=t)
    args = hc.k1_args(P)
    got = run(monkeypatch, hc.dsv41_hc_post_partials, args, backend, blocks_of(backend, bt, bh, ns))
    ok, msg = hc.judge_post_partials(got, args)
    assert ok, msg
    ns_eff = got[1].shape[0]          # Helion's ref mode ignores the config: block sizes = their max
    assert got[1].shape == (ns_eff, t) and got[4].shape == (ns_eff, t, 4, 4)
    if backend == "stub":
        assert ns_eff == ns
        # a split past the end of the hidden dimension contributes exactly zero
        per = -(-(-(-H // ns)) // bh) * bh
        for s in range(ns):
            if s * per >= H:
                assert not got[1][s].any() and not got[4][s].any()


# ---------------------------------------------------------------------------
# K2: epilogue (+ aux, + MXFP8 quant)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("t,bt,bg", [(1, 1, 1), (5, 2, 2), (6, 8, 4), (17, 4, 1), (130, 8, 2)])
@pytest.mark.parametrize("use_pre_in,aux,quant", [(True, False, False), (False, True, True),
                                                   (True, True, True)])
def test_epilogue_body(monkeypatch, backend, t, bt, bg, use_pre_in, aux, quant):
    P = problem(t, use_pre_in, seed=t + 1)
    rc = hc.reference_residual(P.x, P.res, P.post, P.comb)
    parts = hc.split_partials(rc, P.fn, ns=3)
    args = hc.k2_args(P, (rc, *parts), aux, quant)
    got = run(monkeypatch, hc.dsv41_hc_epilogue, args, backend, blocks_of(backend, bt, bg))
    want = hc._ref_k2(*args)
    ok, msg = hc.judge_epilogue(got, want)
    assert ok, msg
    if quant:
        assert got[6].shape == (cm.mxfp8_scale_bytes(t, H),) and got[5].shape == (t, H)
        # the padding bytes of the swizzled scales are exactly zero (rows >= T, groups >= H/32)
        ref_q, ref_s = cm.mxfp8_quant_reference(got[2])
        assert torch.equal(got[6], ref_s) and torch.equal(got[5].view(torch.uint8), ref_q.view(torch.uint8))
    else:
        assert got[5].numel() == 0 and got[6].numel() == 0
    assert got[4].shape == ((t, H) if aux else (0, H))


@pytest.mark.parametrize("backend", backends())
def test_epilogue_quant_padding_groups(monkeypatch, backend):
    """hidden = 5 groups of 32: the scale tensor pads to 8 group columns; those and the
    padding rows must be zero, whatever the allocator handed out before."""
    P = problem(3, seed=9, h=160)
    rc = hc.reference_residual(P.x, P.res, P.post, P.comb)
    args = hc.k2_args(P, (rc, *hc.split_partials(rc, P.fn)), False, True)
    got = run(monkeypatch, hc.dsv41_hc_epilogue, args, backend, blocks_of(backend, 2, 2))
    ref_q, ref_s = cm.mxfp8_quant_reference(got[2])
    assert got[6].numel() == 128 * 8 and torch.equal(got[6], ref_s)


# ---------------------------------------------------------------------------
# K0: the resident (one launch) boundary, and split2 == resident up to f32 sums
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("t,bt,bh,bg", [(1, 1, 32, 1), (5, 2, 32, 2), (9, 4, 64, 4)])
@pytest.mark.parametrize("use_pre_in,aux,quant", [(True, False, False), (False, True, True)])
def test_resident_body(monkeypatch, backend, t, bt, bh, bg, use_pre_in, aux, quant):
    P = problem(t, use_pre_in, seed=2 * t)
    args = hc.k0_args(P, aux, quant)
    got = run(monkeypatch, hc.dsv41_hc_resident, args, backend, blocks_of(backend, bt, bh, bg))
    ok, msg = hc.judge_boundary(got, args)
    assert ok, msg


@pytest.mark.parametrize("backend", backends())
def test_split2_equals_resident(monkeypatch, backend):
    P = problem(6, seed=11)
    r = run(monkeypatch, hc.dsv41_hc_resident, hc.k0_args(P, True, True), backend,
            blocks_of(backend, 2, 32, 2))
    k1 = run(monkeypatch, hc.dsv41_hc_post_partials, hc.k1_args(P), backend, blocks_of(backend, 2, 32, 4))
    k2 = run(monkeypatch, hc.dsv41_hc_epilogue, hc.k2_args(P, k1, True, True), backend,
             blocks_of(backend, 2, 2))
    assert torch.equal(r[0], k1[0])                              # residual_cur: same bits
    for a, b in zip(r[1:], k2):                                  # the rest: f32 split-sum noise only
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        else:
            assert float((a.view(torch.uint8) != b.view(torch.uint8)).float().mean()
                         if a.dtype != torch.bfloat16 else (a != b).float().mean()) < 2e-3


def test_adversarial_inputs(monkeypatch):
    P = problem(4, seed=2)
    P.x[0] = 0
    P.res[0] = 0                                          # sum(rc^2) = 0 -> rsqrt(eps), zero block scales
    P.res[1] = (P.res[1].float() * 300).to(torch.bfloat16)
    P.post[2] = 0
    P.comb[2] = torch.eye(4)
    P.res[3] = -P.res[3].abs()
    args = hc.k0_args(P, True, True)
    got = run(monkeypatch, hc.dsv41_hc_resident, args, blocks={0: 2, 1: 32, 2: 2})
    ok, msg = hc.judge_boundary(got, args)
    assert ok, msg
    assert torch.equal(got[0][0], torch.zeros(4, H, dtype=torch.bfloat16))
    assert all(bool(t.float().isfinite().all()) for t in got[:6])
    k1 = run(monkeypatch, hc.dsv41_hc_post_partials, hc.k1_args(P), blocks={0: 2, 1: 32, 2: 2})
    assert hc.judge_post_partials(k1, hc.k1_args(P))[0]


@pytest.mark.parametrize("repeat", [1, 2, 5])
def test_sinkhorn_iteration_count_matters(monkeypatch, repeat):
    P = problem(4, seed=6, repeat=repeat)
    args = hc.k0_args(P)
    got = run(monkeypatch, hc.dsv41_hc_resident, args, blocks={0: 2, 1: 32, 2: 2})
    ok, msg = hc.judge_boundary(got, args)
    assert ok, msg
    if repeat == 1:
        assert (got[2].sum(1) - 1).abs().max() < 1e-5      # ends with the column normalisation
        assert (got[2].sum(2) - 1).abs().max() > 1e-4      # rows not yet balanced


def test_judges_catch_errors(monkeypatch):
    P = problem(3, seed=4)
    args = hc.k0_args(P, True, True)
    got = list(run(monkeypatch, hc.dsv41_hc_resident, args, blocks={0: 1, 1: 32, 2: 2}))
    assert hc.judge_boundary(tuple(got), args)[0]
    for idx, mutate in ((2, lambda t: t * 1.01), (3, lambda t: t + 0.5), (4, lambda t: t * 1.01),
                        (6, lambda t: (t.view(torch.uint8) ^ 1).view(torch.float8_e4m3fn)),
                        (7, lambda t: (t ^ 1))):
        bad = got[:]
        bad[idx] = mutate(bad[idx].clone())
        assert not hc.judge_boundary(tuple(bad), args)[0], idx
    bad = got[:]
    bad[0] = bad[0].clone()
    bad[0][0, 0, :8] += 1.0
    assert not hc.judge_boundary(tuple(bad), args)[0]


# ---------------------------------------------------------------------------
# serving wrapper: split2 and resident through hk.compiled (stand-in), flags, warm check
# ---------------------------------------------------------------------------
@pytest.fixture
def served(monkeypatch):
    blocks = {hc.K_POST: {0: 2, 1: 32, 2: 4}, hc.K_EPI: {0: 2, 1: 2}, hc.K_RES: {0: 2, 1: 32, 2: 2}}

    def compiled(body, kernel, case_):
        return lambda *a: run_body(monkeypatch, hc, body, a, blocks[kernel])
    monkeypatch.setattr(hk, "compiled", compiled)
    monkeypatch.setattr(hc, "_WARM", set())
    return monkeypatch


@pytest.mark.parametrize("boundary", ["split2", "resident"])
@pytest.mark.parametrize("aux,quant", [(False, False), (True, True)])
def test_serving_wrapper_is_a_drop_in(served, boundary, aux, quant):
    served.setenv(hc.BOUNDARY_ENV, boundary)
    P = problem(6, seed=5)
    out = hc.hc_post_pre(*hc.stock_call_args(P, aux), want_q=quant)
    t = 6
    assert len(out) == (8 if quant else 6)
    assert [tuple(o.shape) for o in out[:6]] == [(t, 4, H), (t, 4, 1), (t, 4, 4), (t, H), (t, 4),
                                                 (t, H) if aux else (0, H)]
    args = hc.k0_args(P, aux, quant)
    tail = out[6:] if quant else (torch.empty(0, H, dtype=torch.float8_e4m3fn), torch.empty(0, dtype=torch.uint8))
    got = (out[0], out[1].squeeze(-1), *out[2:6], *tail)
    ok, msg = hc.judge_boundary(got, args)
    assert ok, msg
    if quant:
        assert torch.equal(out[7], cm.mxfp8_quant_reference(out[3])[1])


def test_serving_refuses_bad_calls_and_unwarmed_capture(served):
    P = problem(6, seed=5)
    stock = list(hc.stock_call_args(P))
    stock[13] = None                                        # norm_weight
    with pytest.raises(RuntimeError, match="norm_weight is None"):
        hc.hc_post_pre(*stock)
    with pytest.raises(ValueError, match="shapes"):
        hc.hc_post_pre(P.x[:, :-8], *hc.stock_call_args(P)[1:])
    served.setenv(hc.BOUNDARY_ENV, "nope")
    with pytest.raises(RuntimeError, match="split2 \\| resident"):
        hc.hc_post_pre(*hc.stock_call_args(P))
    served.delenv(hc.BOUNDARY_ENV)
    served.setattr(torch.cuda, "is_available", lambda: True)
    served.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="first used while capturing"):
        hc.hc_post_pre(*hc.stock_call_args(P))
    served.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    hc.hc_post_pre(*hc.stock_call_args(P))                  # eager warm-up registers the variants
    served.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hc.hc_post_pre(*hc.stock_call_args(P))                  # now capturing is allowed


# ---------------------------------------------------------------------------
# Triton lowering (real Helion front end), search spaces, lab ABI
# ---------------------------------------------------------------------------
def _spec_bounds(bound):
    return [(b.min_size, b.max_size) for b in bound.config_spec.block_sizes]


@pytest.mark.parametrize("t", [6, 48, 192])
@pytest.mark.parametrize("aux,quant", [(False, False), (True, True)])
def test_bodies_lower_to_triton_without_cross_cta_sync(t, aux, quant):
    P = hc.make_problem(t, 5120, 0, "cpu")
    c = hc.case(t)
    rc = hc.reference_residual(P.x, P.res, P.post, P.comb)
    for body, name, args in ((hc.dsv41_hc_post_partials, hc.K_POST, hc.k1_args(P)),
                             (hc.dsv41_hc_epilogue, hc.K_EPI,
                              hc.k2_args(P, (rc, *hc.split_partials(rc, P.fn)), aux, quant)),
                             (hc.dsv41_hc_resident, hc.K_RES, hc.k0_args(P, aux, quant))):
        code = lower_to_triton(body, args, hc.default_config(c, name))
        for banned in ("atomic", "grid_barrier", "while ", "semaphore", "spin", "gdc_", "cluster"):
            assert banned not in code, (name, banned)       # MINED-LESSONS: no cross-CTA sync
        # Helion itself puts a CTA-level tl.debug_barrier() between a store and the re-read of the same
        # buffer (rc in the resident kernel; Helion 1.5.0 also in the epilogue): __syncthreads, not a grid
        # barrier (their count is a performance detail). The resident kernel must have at least one.
        assert name != hc.K_RES or code.count("tl.debug_barrier()") > 0, name
        assert ("tl.float8e4nv" in code) == (quant and name != hc.K_POST)


@pytest.mark.parametrize("name,body,make", [
    (hc.K_POST, hc.dsv41_hc_post_partials, lambda P, rc: hc.k1_args(P)),
    (hc.K_EPI, hc.dsv41_hc_epilogue, lambda P, rc: hc.k2_args(P, (rc, *hc.split_partials(rc, P.fn)))),
    (hc.K_RES, hc.dsv41_hc_resident, lambda P, rc: hc.k0_args(P))])
def test_search_space_is_the_documented_one(name, body, make):
    pytest.importorskip("helion")
    import helion
    from helion.runtime.ref_mode import RefMode
    P = hc.make_problem(192, 5120, 0, "cpu")
    rc = hc.reference_residual(P.x, P.res, P.post, P.comb)
    bound = helion.kernel(body, static_shapes=False, ref_mode=RefMode.OFF).bind(tuple(make(P, rc)))
    want = list(hc.SEARCH_SPACE[name].values())
    assert _spec_bounds(bound) == want
    for t in (6, 48, 192):
        cfg = hc.default_config(hc.case(t), name)
        for (lo, hi), v in zip(want, cfg["block_sizes"]):
            assert lo <= v <= hi and v & (v - 1) == 0


def test_buckets_cases_and_abi():
    assert hc.serving_case(1, 5120) == hc.case(1, 5120)
    assert hc.serving_case(30, 5120) == hc.case(48, 5120)          # c5 at DSpark k=5 -> bucket 48
    assert hc.serving_case(192, 5120)["numtokens"] == 192          # c32
    assert hc.serving_case(193, 5120)["numtokens"] == 192          # clamps; the hook sends it to stock
    model = {"architecture": "DeepseekV41ForCausalLM", "hidden": 5120, "tp": 8}
    got = hc.cases(model)
    assert [c["numtokens"] for c in got] == list(hc.BUCKETS) and len(got) <= 64
    json.dumps(got)
    assert hc.cases({"architecture": "Qwen3ForCausalLM"}) == [] and hc.cases({}) == []
    assert [e["name"] for e in hc.KERNELS] == [hc.K_POST, hc.K_EPI, hc.K_RES]
    for ent in hc.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"])
        assert ent["body"].__module__ == hc.__name__ and ent["body"].__qualname__ == ent["name"]
        for key in ("cases", "make", "reference", "baseline", "check"):
            assert callable(ent[key])
        assert 0 < len(ent["replaces"]) <= 16 and all(len(p) <= 300 for p in ent["replaces"])
    pats = hc.KERNELS[2]["replaces"]
    for name in ("mhc_pre_big_fuse_with_norm_tilelang", "mhc_post_tilelang",
                 "mhc_fused_post_pre_delayed_tilelang"):
        assert any(re.search(p, name, re.I) for p in pats), name
    assert not any(re.search(p, "fused_add_rms_norm_kernel", re.I) for p in pats)


def test_lab_entries_judge_their_own_outputs(monkeypatch):
    """The ABI's reference + check wired together on CPU (make() builds CUDA tensors)."""
    k1, k2, k0 = hc.KERNELS
    a1 = hc.k1_args(problem(3, seed=1))
    monkeypatch.setitem(hc._LAST, "k1", a1)
    got = run(monkeypatch, hc.dsv41_hc_post_partials, a1, blocks={0: 1, 1: 32, 2: 2})
    assert k1["check"](got, k1["reference"](*a1))[0]
    P = problem(4, seed=2)
    rc = hc.reference_residual(P.x, P.res, P.post, P.comb)
    a2 = hc.k2_args(P, (rc, *hc.split_partials(rc, P.fn)), True, True)
    got = run(monkeypatch, hc.dsv41_hc_epilogue, a2, blocks={0: 2, 1: 2})
    assert k2["check"](got, k2["reference"](*a2))[0]
    a0 = hc.k0_args(P, False, True)
    monkeypatch.setitem(hc._LAST, "k0", a0)
    got = run(monkeypatch, hc.dsv41_hc_resident, a0, blocks={0: 2, 1: 32, 2: 2})
    assert k0["check"](got, k0["reference"](*a0))[0]


def test_launch_count_derivation():
    """The derivation in docs/dsv41-fuse.md: 77 served boundaries per step (layer 0's, layer 1's and
    layer 14's attention pre use other stock entry points)."""
    boundaries = 2 * 40 - 3
    stock = lambda tokens: 2 if tokens <= 32 else 3     # mhc_fused_post_pre_split_config
    split2 = 2
    assert boundaries * (stock(30) - split2) == 0       # c<=5: parity (stock fuses post + GEMM)
    assert boundaries * (stock(36) - split2) == 77      # c>=6 (M=36..192): -1 launch per boundary
    assert boundaries * (stock(36) - 1) == 154          # resident, if it ever wins
    quant = 37 + 40                                     # attention input (37 layers) + shared-expert gate_up (40)
    swiglu = 40
    assert quant + swiglu == 117                        # launches the quant epilogue + SwiGLU remove at any c
