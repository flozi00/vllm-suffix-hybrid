# SPDX-License-Identifier: Apache-2.0
"""K3 on CPU: the Helion functions of dsv41_attn.py (partial + merge, row) in
Helion's eager reference mode (HELION_INTERPRET=1) on adversarial rows against
the exact float64 reference of dsv41_attn_ref.py -- both record pairs (today's
584/584 and the vLLM nvfp4_ds_mla 528/288), split counts 1/3/8/32, tile tails
(bk 16/32), heads 1/4/8/16, SWA width 128/136/192, ratio-0 layers (no
compressed pool), -1 holes, short and empty lists, duplicate slots, lens
shorter than the live-looking prefix, scale bytes 4 and 235, saturating and
tiny tiles, |q| ~ 2^15, a -inf / finite / dominating sink, unwritten NaN rows
behind the clamp target. They prove indexing and math, not speed.

Also (when Helion is installed): Helion's own compiler front end, run without a
GPU (RefMode.OFF + to_triton_code), must accept every body at every record pair,
each Helion function must lower to exactly ONE Triton kernel with no atomics /
barriers, two dots per tile loop, no f32 K tile; and the boot gate
dsv41_attn_capture_replay is exercised with the eager stand-in backend
(healthy route passes; stale replay, non-determinism, unwritten rows and a
wrong kernel are caught)."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before any helion import

import re  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_attn as D  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_attn_ref as R  # noqa: E402
from suffix_hybrid.kernels import dsv41_attn_capture as G  # noqa: E402


@pytest.fixture
def interp(monkeypatch):
    helion = pytest.importorskip("helion")
    if D.helion is None:
        pytest.skip("dsv41_attn imported before helion was installed")
    real = torch.mm

    def mm(a, b, *args, out_dtype=None, **kw):
        # CPU torch has no mm(out_dtype=): f32 math is the same exact-product,
        # f32-accumulate semantics hl.dot has on the GPU
        if out_dtype is None or a.is_cuda:
            return real(a, b, *args, **kw) if out_dtype is None else real(a, b, *args, out_dtype=out_dtype, **kw)
        return real(a.float(), b.float()).to(out_dtype)
    monkeypatch.setattr(torch, "mm", mm)
    monkeypatch.setattr(D, "_WARM", set())
    return helion


def route_of(interp, kernel, t, heads, swa, topk, recs, ns=4, bk=None, ctx="c512k"):
    cfg = {"block_sizes": [bk]} if bk else {}
    if kernel == D.PARTIAL:
        cfg["num_splits"] = ns
    kern = interp.kernel(D._BODY[kernel], config=interp.Config(**cfg), static_shapes=False, **D.SETTINGS)
    kernels = {D.PARTIAL: {}, D.ROW: {}}
    kernels[kernel][(hk.bucket(t, D.buckets(kernel)), D.case_ctx(t, topk, ctx))] = kern
    return D.new_route("helion", heads, swa, topk, recs, kernels, "t", torch.device("cpu"))


def run(interp, kernel, recs, t, heads=8, swa=128, topk=512, ns=4, bk=None, seed=1, **kw):
    prob = D.adversarial_problem(t, heads, swa, *recs, seed=seed, topk=topk or R.TOPK, with_cmp=topk > 0, **kw)
    route = route_of(interp, kernel, t, heads, swa, topk, recs, ns, bk)
    out = torch.zeros(t, heads, R.HD, dtype=torch.bfloat16)
    D.run_problem(route, prob, out)
    rows = prob["rows"]
    ok, msg = R.check(out[rows], prob["ref"][rows])
    assert ok, f"{kernel} recs {recs} T={t} H={heads} swa={swa} topk={topk} ns={ns} bk={bk}: {msg}"
    return out, prob


PAIRS = [(R.REC_V4, R.REC_V4), (R.REC_FP8, R.REC_FP4)]


@pytest.mark.parametrize("recs", PAIRS)
@pytest.mark.parametrize("ns,bk", [(1, None), (3, 16), (8, 32), (32, 16)])
def test_partial_plus_merge_match_reference(interp, recs, ns, bk):
    run(interp, D.PARTIAL, recs, 15, ns=ns, bk=bk)


@pytest.mark.parametrize("recs", PAIRS)
@pytest.mark.parametrize("bk", [None, 16, 32])
def test_row_body_matches_reference(interp, recs, bk):
    run(interp, D.ROW, recs, 15, bk=bk)


@pytest.mark.parametrize("kernel", [D.PARTIAL, D.ROW])
@pytest.mark.parametrize("heads", [1, 4, 16])
def test_head_counts(interp, kernel, heads):
    run(interp, kernel, PAIRS[0], 9, heads=heads, bk=16)


@pytest.mark.parametrize("kernel", [D.PARTIAL, D.ROW])
@pytest.mark.parametrize("recs,swa", [(PAIRS[0], 136), (PAIRS[0], 192), (PAIRS[1], 192)])
def test_ratio0_layers_and_noncausal_width(interp, kernel, recs, swa):
    """No compressed pool (cmp_len 0, a [T, 1] dummy list) and the DSpark draft layers' 192-wide window."""
    run(interp, kernel, (recs[0], recs[0] if recs[0] == R.REC_V4 else R.REC_FP8), 9, swa=swa, topk=0, bk=16)


def test_single_token_and_pairs(interp):
    for t in (1, 2, 5):
        run(interp, D.PARTIAL, PAIRS[0], t, ns=2)
        run(interp, D.ROW, PAIRS[1], t)


def test_prefill_sized_bodies_sampled(interp):
    run(interp, D.ROW, PAIRS[0], 300, sample=24, bk=64)
    run(interp, D.PARTIAL, PAIRS[1], 300, sample=24, bk=32, ns=2)


def test_partial_equals_row_and_every_split_count(interp):
    """Every split count and the one-kernel body agree up to the bf16 rounding of the split partials."""
    outs = [run(interp, D.PARTIAL, PAIRS[0], 14, ns=n, seed=3)[0] for n in (1, 2, 5, 16)]
    outs.append(run(interp, D.ROW, PAIRS[0], 14, seed=3)[0])
    for o in outs[1:]:
        d = (o.float() - outs[0].float()).norm(dim=-1) / outs[0].float().norm(dim=-1).clamp_min(1e-30)
        assert float(d.max()) < 1e-2


def test_determinism_bit_exact_across_repeats(interp):
    """No atomics, fixed reduction order: the same inputs give the same bits (the boot gate relies on it)."""
    for kernel in (D.PARTIAL, D.ROW):
        a, _ = run(interp, kernel, PAIRS[0], 12, seed=7, bk=16)
        b, _ = run(interp, kernel, PAIRS[0], 12, seed=7, bk=16)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_sink_variants(interp):
    """-inf sink == no sink (exactly the sinkless reference); a dominating sink shrinks outputs but stays finite."""
    prob = D.adversarial_problem(8, 4, 128, *PAIRS[0], seed=5)
    nosink = R.attention_ref(prob["q"], prob["swa"], prob["cmp"], prob["swa_idx"], prob["swa_len"],
                             prob["cmp_idx"], prob["cmp_len"], None)
    for kernel in (D.PARTIAL, D.ROW):
        route = route_of(interp, kernel, 8, 4, 128, 512, PAIRS[0], 4, 16)
        prob["sink"] = torch.full((4,), float("-inf"))
        out = torch.zeros(8, 4, R.HD, dtype=torch.bfloat16)
        D.run_problem(route, prob, out)
        assert R.check(out, nosink)[0]
        prob["sink"] = torch.full((4,), 30.0)
        out = torch.zeros(8, 4, R.HD, dtype=torch.bfloat16)
        D.run_problem(route, prob, out)
        small = R.attention_ref(prob["q"], prob["swa"], prob["cmp"], prob["swa_idx"], prob["swa_len"],
                                prob["cmp_idx"], prob["cmp_len"], prob["sink"])
        assert R.check(out, small)[0] and float(out.float().norm()) < float(nosink.norm())


def test_stale_nan_rows_cannot_poison_outputs(interp):
    """The kernels clamp -1 to slot 0 and zero its bytes: NaN scale/codes there (and in unwritten rows) must not reach P.V."""
    for recs in PAIRS:
        prob = D.adversarial_problem(10, 8, 128, *recs, seed=9)
        for pl in (prob["swa"], prob["cmp"]):
            flat, rec, page, stride = pl
            flat.view(-1, stride)[0, : page * R.ROW_BYTES[rec]] = 0xFF   # block 0 = NaN scales / e4m3 NaN codes / NaN rope
        for idx in (prob["swa_idx"], prob["cmp_idx"]):
            idx[idx.lt(32)] = -1                                # nothing live inside block 0 ...
        prob["ref"] = R.attention_ref(prob["q"], prob["swa"], prob["cmp"], prob["swa_idx"], prob["swa_len"],
                                      prob["cmp_idx"], prob["cmp_len"], prob["sink"])
        for kernel in (D.PARTIAL, D.ROW):                       # ... but -1 columns still gather slot 0
            route = route_of(interp, kernel, 10, 8, 128, 512, recs, 4, 16)
            out = torch.zeros(10, 8, R.HD, dtype=torch.bfloat16)
            D.run_problem(route, prob, out)
            ok, msg = R.check(out, prob["ref"])
            assert ok, msg


def test_oracle_has_teeth(interp):
    """A corrupted scale byte (after the reference was computed) and a wrong sink must fail the K3 gate."""
    prob = D.adversarial_problem(8, 8, 128, *PAIRS[0], seed=2)
    flat, rec, page, stride = prob["cmp"]
    live_slot = int(prob["cmp_idx"][0, 0])
    R.poke_scale(flat, rec, page, stride, live_slot, 0, 130)   # 8x on the first 64 dims of a live row
    route = route_of(interp, D.ROW, 8, 8, 128, 512, PAIRS[0], bk=16)
    out = torch.zeros(8, 8, R.HD, dtype=torch.bfloat16)
    D.run_problem(route, prob, out)
    assert not R.check(out, prob["ref"])[0]
    prob = D.adversarial_problem(8, 8, 128, *PAIRS[0], seed=2)
    prob["sink"] = prob["sink"] + 1.0
    D.run_problem(route, prob, out)
    assert not R.check(out, prob["ref"])[0]


def test_warm_and_verify_runs_every_kernel_and_fails_closed(interp):
    p = route_of(interp, D.PARTIAL, 6, 4, 128, 512, PAIRS[0], 4, 16)
    r = route_of(interp, D.ROW, 12, 4, 128, 512, PAIRS[0], bk=64)
    route = D.new_route("helion", 4, 128, 512, PAIRS[0], {D.PARTIAL: p["kernels"][D.PARTIAL], D.ROW: r["kernels"][D.ROW]},
                        "v4", torch.device("cpu"))
    D.warm_and_verify(route, torch.device("cpu"))
    assert torch.get_default_dtype() == torch.float32
    # T = 5 and 6 (bucket 6) and T = 9 and 12 (bucket 12): not 1, not a multiple of 16 -> ONE specialization each
    assert len([k for k in D._WARM if k[0] == D.PARTIAL]) == 1 and len([k for k in D._WARM if k[0] == D.ROW]) == 1
    assert ("merge", 4, False, False) in D._WARM
    bad = D.new_route("helion", 4, 128, 512, PAIRS[0], {D.PARTIAL: {}, D.ROW: {(12, "any"): lambda *a: a[12].fill_(float("nan"))}},
                      "v4", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="load-time oracle FAILED at T=12"):
        D.warm_and_verify(bad, torch.device("cpu"))


# ---- the boot gate dsv41_attn_capture_replay (eager stand-in backend) --------
def gate_route(interp, perturb=None):
    p = route_of(interp, D.PARTIAL, 6, 4, 128, 512, PAIRS[0], 4, 16)
    r = route_of(interp, D.ROW, 12, 4, 128, 512, PAIRS[0], bk=64)
    kernels = {D.PARTIAL: dict(p["kernels"][D.PARTIAL]), D.ROW: dict(r["kernels"][D.ROW])}
    if perturb:
        for name in kernels:
            kernels[name] = {k: perturb(name, v) for k, v in kernels[name].items()}
    route = D.new_route("helion", 4, 128, 512, PAIRS[0], kernels, "v4", torch.device("cpu"))
    return route


def test_gate_passes_a_healthy_route(interp):
    route = gate_route(interp)
    res = G.dsv41_attn_capture_replay(route, backend=G.EagerBackend(), trials=3, small=True)
    assert [(x["kernel"], x["bucket"], x["T"]) for x in res] == [
        (D.PARTIAL, 6, 5), (D.PARTIAL, 6, 6), (D.ROW, 12, 9), (D.ROW, 12, 12)]
    assert all(x["trials"] == 3 for x in res)


def test_gate_catches_a_graph_that_does_not_follow_the_inputs(interp):
    class Stale(G.EagerBackend):
        def capture(self, fn):
            fn()
            return lambda: None            # a replay that writes nothing (stale / wrong pointers)
    with pytest.raises(G.GateFailure, match="non-finite"):
        G.dsv41_attn_capture_replay(gate_route(interp), backend=Stale(), trials=2, small=True)


def test_gate_catches_state_that_differs_between_capture_and_eager(interp):
    count = {"n": 0}

    def noisy(name, kern):
        def wrapped(*a):
            res = kern(*a)
            count["n"] += 1
            if name == D.ROW and count["n"] % 3 == 0:    # every third call: a stale-state class of bug
                a[12].add_(1.0)
            return res
        return wrapped
    with pytest.raises(G.GateFailure, match="differs from eager"):
        G.dsv41_attn_capture_replay(gate_route(interp, noisy), backend=G.EagerBackend(), trials=3, small=True)


def test_gate_catches_unwritten_rows_and_wrong_kernels(interp):
    def skip_row(name, kern):
        def wrapped(*a):
            res = kern(*a)
            if name == D.ROW:
                a[12][0].fill_(float("nan"))              # a row the kernel "forgot"
            return res
        return wrapped
    with pytest.raises(G.GateFailure, match="non-finite"):
        G.dsv41_attn_capture_replay(gate_route(interp, skip_row), backend=G.EagerBackend(), trials=1, small=True)

    def wrong(name, kern):
        def wrapped(*a):
            res = kern(*a)
            if name == D.ROW:
                a[12].mul_(1.5)                           # deterministic but wrong
            return res
        return wrapped
    with pytest.raises(G.GateFailure, match="K3 gate"):
        G.dsv41_attn_capture_replay(gate_route(interp, wrong), backend=G.EagerBackend(), trials=1, small=True)


def test_gate_reports_capture_failures(interp):
    class Broken(G.EagerBackend):
        def capture(self, fn):
            raise RuntimeError("operation not permitted when stream is capturing")
    with pytest.raises(G.GateFailure, match="graph capture failed: RuntimeError"):
        G.dsv41_attn_capture_replay(gate_route(interp), backend=Broken(), trials=1, small=True)


def test_gate_replays_with_changing_inputs_and_ctx_lengths(interp):
    seen = []
    orig = G._load_inputs

    def spy(st, prob):
        seen.append((st["swa_len"].clone(), st["cmp_len"].clone(), st["q"].clone()))
        return orig(st, prob)
    import unittest.mock as mock
    with mock.patch.object(G, "_load_inputs", spy):
        G.dsv41_attn_capture_replay(gate_route(interp), backend=G.EagerBackend(), trials=3, small=True)
    # (kernel, T) cases x (trials - 1) refills; every refill brought different lens and a different q
    assert len(seen) == 4 * 2
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(seen, seen[1:]))
    assert any(not torch.equal(a[2], b[2]) for a, b in zip(seen, seen[1:]))


# ---- Helion's own compiler front end, no GPU --------------------------------
def body_args(kernel, t=6, h=8, swa=128, topk=512, recs=(0, 0), seed=1):
    prob = D.adversarial_problem(t, h, swa, *recs, seed=seed, topk=topk or 512, with_cmp=topk > 0, reference=False)
    dm = {"d16": torch.zeros(1, 1, dtype=torch.int16), "s": torch.zeros(1, 8, dtype=torch.uint8)}

    def rv(pl):
        flat, rec, page, stride = pl
        d, d16, s, bd, bs = D.row_views(flat, rec, stride, dm)
        return d, d16, s, page.bit_length() - 1, bd, bs, rec
    sd, sd16, ss, sshift, sbd, sbs, srec = rv(prob["swa"])
    cd, cd16, cs, cshift, cbd, cbs, crec = rv(prob["cmp"]) if prob["cmp"] is not None else (sd, sd16, ss, sshift, sbd, sbs, srec)
    pools = (prob["q"], sd, sd16, ss, cd, cd16, cs, prob["swa_idx"], prob["swa_len"], prob["cmp_idx"], prob["cmp_len"],
             prob["sink"])
    scalars = (sshift, sbd, sbs, cshift, cbd, cbs, srec, crec)
    if kernel == D.ROW:
        return (*pools, torch.empty(t, h, 512, dtype=torch.bfloat16), *scalars)
    return (*pools, *scalars)


def triton_code(body, args, **cfg):
    helion = pytest.importorskip("helion")
    ref_mode = pytest.importorskip("helion.runtime.ref_mode")
    kern = helion.kernel(body, static_shapes=False, ref_mode=ref_mode.RefMode.OFF, **D.SETTINGS)
    bound = kern.bind(args)
    return bound.to_triton_code(helion.Config(**cfg) if cfg else bound.config_spec.default_config()), bound


@pytest.mark.parametrize("recs,topk", [((0, 0), 512), ((1, 2), 512), ((0, 0), 0), ((1, 1), 0)])
@pytest.mark.parametrize("kernel", [D.PARTIAL, D.ROW])
def test_helion_compiles_each_body_to_one_triton_kernel(kernel, recs, topk):
    swa = 192 if topk == 0 else 128
    code, bound = triton_code(D._BODY[kernel], body_args(kernel, swa=swa, topk=topk, recs=recs))
    assert code.count("@triton.jit") == 1                    # one launch per Helion function: no hidden barrier stage
    # Helion 1.5.0 puts one CTA-level tl.debug_barrier() (__syncthreads) between a store and a re-read of the
    # same buffer; that is not cross-CTA sync (1.5.1 does not always emit it). Everything else stays banned.
    low = code.lower().replace("tl.debug_barrier()", "")
    assert not any(w in low for w in ("atomic", "barrier", "signal", "semaphore", "spin"))
    assert code.count("tl.dot(") == 4                        # (QK, PV) x (SWA loop, compressed loop)
    assert not re.search(r"tl\.cast\(k16[_0-9a-z]*, tl\.float32", code)    # no f32 [bk, 512] K tile
    assert "float8e4nv" in code or recs[0] != 0
    if recs[1] == 2:
        assert "tl.join" in code                              # e2m1 nibble pairs
    assert "_launcher(" in code and code.count("_launcher(") == 1


def test_helion_compiles_the_merge_to_one_triton_kernel():
    helion = pytest.importorskip("helion")
    args = (torch.zeros(6, 8, 4, 512, dtype=torch.bfloat16), torch.zeros(6, 8, 4), torch.zeros(8),
            torch.zeros(6, 8, 512, dtype=torch.bfloat16))
    code, bound = triton_code(D.dsv41_attn_merge, args, block_sizes=[512], num_warps=4, num_stages=1)
    assert code.count("@triton.jit") == 1 and "tl.dot(" not in code
    assert not any(w in code.lower() for w in ("atomic", "barrier", "signal", "semaphore"))
    assert helion.Config(block_sizes=[512], num_warps=4, num_stages=1) is not None


@pytest.mark.parametrize("cfg", [
    {"block_sizes": [16], "num_splits": 1, "num_warps": 4, "num_stages": 1},
    {"block_sizes": [64], "num_splits": 32, "num_warps": 8, "num_stages": 3},
    {"block_sizes": [32], "num_splits": 8, "num_warps": 4, "num_stages": 2}])
def test_helion_accepts_the_search_space_corners(cfg):
    code, _ = triton_code(D.dsv41_attn_partial, body_args(D.PARTIAL), **cfg)
    assert f"num_warps={cfg['num_warps']}" in code and f"num_stages={cfg['num_stages']}" in code
    assert f"_BLOCK_SIZE_0 = tl.constexpr({cfg['block_sizes'][0]})" in code
    code, _ = triton_code(D.dsv41_attn_row, body_args(D.ROW), block_sizes=cfg["block_sizes"], num_warps=cfg["num_warps"],
                          num_stages=cfg["num_stages"])
    assert f"num_warps={cfg['num_warps']}" in code


def test_helion_config_spec_exposes_the_search_axes():
    helion = pytest.importorskip("helion")
    _, bound = triton_code(D.dsv41_attn_partial, body_args(D.PARTIAL))
    flat = bound.config_spec._flat_fields()
    for axis in ("block_sizes", "num_splits", "num_warps", "num_stages", "loop_orders", "l2_groupings",
                 "load_eviction_policies", "pid_type"):
        assert axis in flat, axis
    assert len(bound.config_spec.block_sizes) == 1            # bk only: the merge is a separate function
    assert helion is not None


# ---- the lab ABI end to end on CPU (make -> body -> reference -> check) --------
@pytest.mark.parametrize("lab,topk,swa", [("v4", 512, 128), ("v41", 512, 128), ("v41", 0, 192), ("v4", 0, 128)])
@pytest.mark.parametrize("kernel", [D.PARTIAL, D.ROW])
def test_lab_make_body_reference_check_roundtrip(interp, monkeypatch, kernel, lab, topk, swa):
    """The exact tuple the autotune lab builds (random valid pools, 6-row sequences, 80% overlap, short rows, a
    padding row) goes through the Helion body and the lab's reference + check, with the lab's own arg plumbing."""
    monkeypatch.setenv("DSV41_ATTN_LAB_DEVICE", "cpu")
    case = D.case(7, 8, swa, topk, lab, "c16k" if topk else "any")
    spec = next(k for k in D.KERNELS if k["name"] == kernel)
    args = spec["make"](case, 0)
    assert len(args) == (21 if kernel == D.ROW else 20) and args[0].shape == (7, 8, R.HD)
    ref = spec["reference"](*args)
    cfg = {"block_sizes": [16], **({"num_splits": 4} if kernel == D.PARTIAL else {})}
    body = interp.kernel(D._BODY[kernel], config=interp.Config(**cfg), static_shapes=False, **D.SETTINGS)
    out = body(*args)
    if kernel == D.PARTIAL:
        assert isinstance(out, tuple) and out[0].shape == (7, 8, 4, R.HD) and out[1].shape == (7, 8, 4)
    ok, msg = spec["check"](out, ref)
    assert ok, msg
    live = ~torch.isnan(ref[:, 0, 0])
    assert bool(live.all()) and ref.shape == (7, 8, R.HD)
    assert bool((D._merged(out)[5] == 0).all())                 # the padding row (len 0) is exactly zero


def test_lab_reference_samples_big_buckets(monkeypatch):
    """Above SAMPLE_ROWS rows only a spaced sample (+ the padding row) is referenced; the rest is NaN and skipped."""
    monkeypatch.setenv("DSV41_ATTN_LAB_DEVICE", "cpu")
    spec = next(k for k in D.KERNELS if k["name"] == D.ROW)
    args = spec["make"](D.case(80, 8, 128, 0, "v4", "any"), 1)
    ref = spec["reference"](*args)
    live = ~torch.isnan(ref[:, 0, 0])
    assert 60 <= int(live.sum()) <= D.SAMPLE_ROWS + 1 and bool(live[5]) and bool(live[0]) and bool(live[79])


def test_cli_default_route_runs_the_gate_without_shipped_configs(interp):
    """The lab/sanitizer entry point (``--small`` without --shipped): one default config per body for every bucket."""
    route = G.default_route(4, 128, 512, PAIRS[0], "cpu", buckets={6, 12})
    assert sorted(route["kernels"][D.PARTIAL]) == [(6, "c512k"), (12, "c512k")]
    assert sorted(route["kernels"][D.ROW]) == [(6, "c512k"), (12, "c512k")]
    assert route["merge"] is not None and route["pages"] == (32, 32)
    D.warm_and_verify(route, torch.device("cpu"))
    res = G.dsv41_attn_capture_replay(route, backend=G.EagerBackend(), trials=2, small=True)
    assert {(x["kernel"], x["bucket"]) for x in res} == {(D.PARTIAL, 6), (D.PARTIAL, 12), (D.ROW, 6), (D.ROW, 12)}
    r64 = G.default_route(4, 128, 512, PAIRS[1], "cpu", buckets={6}, cmp_page=64)     # ratio-1 layers, V4.1 records
    assert r64["pages"] == (32, 64)
    G.dsv41_attn_capture_replay(r64, backend=G.EagerBackend(), trials=1, small=True, only_buckets={6})
