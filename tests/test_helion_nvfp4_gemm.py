# SPDX-License-Identifier: Apache-2.0
"""Helion port of the NVFP4 decode GEMM: lab cases, serving plan / knob
wiring, the oxide-signature twins, and (with Helion installed) every body in
interpret mode against the exact reference."""
import json
import os

os.environ["HELION_INTERPRET"] = "1"  # before any helion.kernel is built

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_gemm as hx  # noqa: E402
from suffix_hybrid.kernels import nvfp4_gemm as ng  # noqa: E402

QWEN27 = {"name": "models/qwen3.8-27b-fable-distill-nvfp4", "tp": 1, "quant_format": "nvfp4",
          "hidden": 5120, "vocab": 248320,
          "linears": [{"name": "qkv_proj", "n": 7168, "k": 5120},
                      {"name": "lm_head", "n": 248320, "k": 5120}]}
FLASH = {"name": "models/qwen3.8-flash-next-nvfp4", "tp": 2, "quant_format": "nvfp4",
         "hidden": 2560, "vocab": 75776, "linears": []}
GLM = {"name": "models/glm-5.3-nvfp4", "tp": 8, "quant_format": "nvfp4", "hidden": 6144,
       "vocab": 19360, "linears": [{"name": "gate_up_proj", "n": 3072, "k": 6144},
                                   {"name": "down_proj", "n": 6144, "k": 1536},
                                   {"name": "lm_head", "n": 19360, "k": 6144}]}
K = {e["name"]: e for e in hx.KERNELS}


def test_kernels_abi():
    assert list(K) == ["nvfp4_gemm_q", "nvfp4_quant_act", "nvfp4_gemm"]  # lab budget order
    for e in hx.KERNELS:
        assert e["body"] is hx.BODIES[e["name"]] and callable(e["check"])
        assert all(r.startswith("^nvfp4_") for r in e["replaces"])
        for c in e["cases"](QWEN27) + e["cases"](FLASH) + e["cases"](GLM):
            assert c["numtokens"] in hx.BUCKETS and json.loads(hk.case_key(c)) == c
            assert set(c) == ({"k", "numtokens"} if e["name"] == hx.QUANT else
                              {"n", "k", "numtokens"})
    assert "prod launch, pf" in K["nvfp4_gemm"]["baseline_name"]
    assert "prod launch, pf" in K["nvfp4_gemm_q"]["baseline_name"]
    # one owner of the oxide entry names (no double credit in the e2e estimate)
    assert [bool(e["replaces"]) for e in hx.KERNELS] == [False, False, True]


def test_cases_per_model():
    dense = list(dict.fromkeys(ng.SHAPES.values()))  # qwen 27B: the per-arch table
    assert len(K["nvfp4_gemm"]["cases"](QWEN27)) == len(dense) * len(hx.BUCKETS)
    q = K["nvfp4_gemm_q"]["cases"](QWEN27)
    assert len(q) == (len(dense) + 1) * len(hx.BUCKETS)  # split-K is a tunable of every case
    assert {"k": 5120, "n": 248320, "numtokens": 1} in q  # lm_head: prequant route only
    assert [c["numtokens"] for c in q[:len(dense) + 1]] == [1] * (len(dense) + 1)  # decode first
    assert all(c["n"] != 248320 for c in K["nvfp4_gemm"]["cases"](QWEN27))
    ks = {k for _, k in dense}
    assert K["nvfp4_quant_act"]["cases"](QWEN27) == [
        {"k": k, "numtokens": m} for m in hx.BUCKETS for k in dict.fromkeys(k for _, k in dense)]
    assert {c["k"] for c in K["nvfp4_quant_act"]["cases"](QWEN27)} == ks
    assert len(K["nvfp4_gemm_q"]["cases"](FLASH)) <= 64  # 11 TP2 shapes + lm_head
    assert {(c["n"], c["k"]) for c in K["nvfp4_gemm"]["cases"](GLM)} == {(3072, 6144), (6144, 1536)}
    assert (19360, 6144) in {(c["n"], c["k"]) for c in K["nvfp4_gemm_q"]["cases"](GLM)}
    for e in hx.KERNELS:
        assert e["cases"]({**QWEN27, "quant_format": "fp8_block"}) == []
    # a qwen 27B at TP2 is not the TP1 table: generic per-rank linears
    assert {(c["n"], c["k"]) for c in K["nvfp4_gemm"]["cases"]({**QWEN27, "tp": 2})} == {(7168, 5120)}


def test_inv_table_is_the_spec_division():
    g = 2688.0 / 3.7
    t = hx.inv_table(g)
    sf = ng.e4m3_bits_to_f32(np.arange(128, dtype=np.uint8))
    ok = (sf > 0) & np.isfinite(sf)
    np.testing.assert_array_equal(t[:128][ok], np.float32(g) / sf[ok])
    assert not t[128:].any() and t[0] == 0 and t[0x7F] == 0  # sign bit / zero / NaN
    assert hx._g_of(torch.from_numpy(t)) == float(np.float32(g))


def test_max_splits_and_warm_ms():
    assert [hx.max_splits(n) for n in (48, 336, 2560, 5120, 16384, 34816)] == [64, 64, 16, 8, 2, 2]
    assert hx.warm_ms(64) == [1, 7, 8, 15, 16, 31, 32, 63, 64]
    assert hx.warm_ms(16) == [1, 7, 8, 15, 16]
    assert hx.warm_ms(40) == [1, 7, 8, 15, 16, 31, 32, 39, 40]


def test_swizzle_matches_spec():
    sf = torch.randint(0, 120, (200, 12), dtype=torch.uint8)
    got = hx._swizzle(sf.view(torch.float8_e4m3fn)).view(torch.uint8).numpy()
    np.testing.assert_array_equal(got, ng.swizzle_sf(sf.numpy()))


@pytest.fixture
def shipped(monkeypatch):
    """Configs 'shipped' for some (kernel, bucket); compiled() -> a recorder."""
    table, calls = {}, []
    monkeypatch.setattr(hk, "config", lambda name, case, plat=None: table.get(
        (name, case["numtokens"])))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")

    def compiled(body, name, case, **kw):
        def kern(*args):
            calls.append((name, case["numtokens"], args))
        return kern
    monkeypatch.setattr(hk, "compiled", compiled)
    monkeypatch.setattr(hx, "_PLANS", {})
    return table, calls


def test_prepare_auto_per_bucket(shipped):
    table, _ = shipped
    table.update({("nvfp4_gemm_q", 1): {}, ("nvfp4_gemm_q", 16): {"split_k": 16}})
    plan = hx.prepare(336, 10240, True, 40, "auto")
    assert len(plan) == 41 and plan[0] is None
    assert plan[1][0] == "nvfp4_gemm_q" and plan[1][2] == 1 and len(plan[1][1]) == 1
    assert all(plan[m] is None for m in range(2, 9))  # bucket 8: oxide
    assert all(plan[m][2:] == (16, 16) for m in range(9, 17))
    assert all(plan[m] is None for m in range(17, 41))
    assert hx.partial_elems(336, [plan]) == 16 * 16 * 336
    assert hx.describe(plan) == ("M1:nvfp4_gemm_q M2-8:oxide M9-16:nvfp4_gemm_q(split<=16) "
                                 "M17-40:oxide")


def test_prepare_bf16_route_one_launch_then_two(shipped):
    table, _ = shipped
    table.update({("nvfp4_gemm", 1): {}, ("nvfp4_gemm_q", 1): {}, ("nvfp4_gemm_q", 8): {},
                  ("nvfp4_quant_act", 8): {}, ("nvfp4_gemm_q", 16): {}})
    plan = hx.prepare(5120, 6144, False, 16, "auto")
    assert plan[1][0] == "nvfp4_gemm" and len(plan[1][1]) == 1  # one launch wins at M=1
    assert plan[8][0] == "nvfp4_quant_act+nvfp4_gemm_q" and len(plan[8][1]) == 2
    assert plan[16] is None  # gemm_q alone is not the bf16 route: no quant config
    assert hx._NEED["xq"] >= 64 * 6144 // 2 and hx._NEED["xsf"] >= 128 * 6144 // 16


def test_prepare_helion_fails_closed(shipped):
    table, _ = shipped
    table.update({("nvfp4_gemm", b): {} for b in (1, 8)})
    assert hx.prepare(5120, 6144, False, 8, "helion")[8][0] == "nvfp4_gemm"
    with pytest.raises(RuntimeError, match="bucket 16 on gpu"):
        hx.prepare(5120, 6144, False, 16, "helion")
    with pytest.raises(RuntimeError):
        hx.prepare(5120, 6144, False, 16, "oxide")


def test_twins_pass_body_layouts(shipped):
    table, calls = shipped
    table.update({("nvfp4_gemm", 1): {}, ("nvfp4_gemm_q", 8): {"split_k": 4},
                  ("nvfp4_quant_act", 8): {}})
    m, n, k = 5, 64, 256
    hx.prepare(n, k, False, 8, "auto")
    hx.prepare(n, k, True, 8, "auto")
    out = torch.empty(m, n, dtype=torch.bfloat16)
    partial = torch.empty(4 * 8 * n)
    wsf = torch.zeros(128, k // 16, dtype=torch.uint8)  # uint8 accepted, viewed as e4m3
    w = torch.zeros(n, k // 2, dtype=torch.uint8)
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    g6 = float(np.float32(3.0) / np.float32(6.0))
    hx.nvfp4_gemm_cuda(x[:1], w, wsf, None, None, partial, out[:1], 3.0, 0.5, 1, 0, 0, None)
    name, b, args = calls[-1]  # M=1: the one-launch fused body
    assert (name, b) == ("nvfp4_gemm", 1)
    assert [a.shape for a in args[:3]] == [(1, k // 2, 2), (n, k // 2), (128 * k // 16,)]
    assert args[2].dtype == torch.float8_e4m3fn and args[4].shape == (1, 1, n)
    assert torch.equal(args[6], torch.from_numpy(hx.inv_table(3.0))) and args[7:] == (g6, 0.5)
    hx.nvfp4_gemm_cuda(x, w, wsf, None, None, partial, out, 3.0, 0.5, 1, 0, 0, None)
    (qn, _, qargs), (gn, _, gargs) = calls[-2:]  # M=5: quant once, then the prequant GEMM
    assert (qn, gn) == ("nvfp4_quant_act", "nvfp4_gemm_q")
    assert qargs[0].shape == (m, k // 16, 16) and qargs[1].shape == (m, k // 16, 8)
    assert qargs[2] is gargs[1] and qargs[1].data_ptr() == gargs[0].data_ptr()
    assert gargs[0].shape == (m, k // 2) and gargs[5].shape == (4, m, n) and gargs[7] == 0.5
    xq = torch.zeros(m, k // 2, dtype=torch.uint8)
    hx.nvfp4_gemm_q_cuda(xq, wsf, w, wsf, partial, out, 0.25, 1, 0)
    name, b, args = calls[-1]
    assert name == "nvfp4_gemm_q" and args[5].shape == (4, m, n) and args[7] == 0.25
    assert args[6].dtype == torch.int32 and args[6].numel() >= hx.counter_elems(n)
    assert not args[6].any()
    with pytest.raises(RuntimeError, match="no Helion plan"):
        hx.nvfp4_gemm_q_cuda(xq[:1].repeat(9, 1), wsf, w, wsf, partial, out, 0.25, 1, 0)


def test_lane_knob_is_the_runtime_backend(monkeypatch):
    from suffix_hybrid.kernels.nvfp4_linear import HELION_ENV
    assert HELION_ENV == hx.ENV
    monkeypatch.setenv(HELION_ENV, "cutlass")
    with pytest.raises(RuntimeError):
        hk.backend(HELION_ENV)


# ---------------------------------------------------------------------------
# interpret mode (Helion installed): each body vs the exact reference
# ---------------------------------------------------------------------------
CASES = [{"k": 192, "n": 160, "numtokens": 3},   # N crosses a 128-row scale atom, N tail
         {"k": 2816, "n": 48, "numtokens": 17},  # K step tail (176 blocks / 32), 2 m16 rows
         {"k": 128, "n": 32, "numtokens": 1}]


@pytest.mark.parametrize("name", list(hx.BODIES))
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"m{c['numtokens']}n{c['n']}k{c['k']}")
def test_body_interpret_vs_reference(name, case, monkeypatch):
    helion = pytest.importorskip("helion")
    mm = torch.mm  # CPU aten::mm has no out_dtype (hl.dot's eager reference uses it)
    monkeypatch.setattr(torch, "mm", lambda a, b, *, out_dtype=None, **kw: (
        mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)))
    if name == hx.QUANT:  # bit-exact vs the spec quant
        args = hx.body_args(name, hx.problem({"k": case["k"], "numtokens": case["numtokens"]},
                                             0, "cpu"))
        ok, msg = hx.check_quant(helion.kernel(hx.BODIES[name], autotune_effort="none")(*args),
                                 hx.reference_quant(*args))
        assert ok, msg
        return
    p = hx.problem(case, 0, "cpu", name == hx.GEMM_Q)
    args = hx.body_args(name, p)
    ref = (hx.reference_q if name == hx.GEMM_Q else hx.reference)(*args)
    for split in (1, 2, 4):
        kern = helion.kernel(hx.BODIES[name], config=helion.Config(split_k=split))
        for _ in range(2):  # a replay: the split-K tickets must be back at zero
            ok, msg = hx.check(kern(*args), ref)
            assert ok, f"split_k={split}: {msg}"
            assert not p["counters"].any(), "split-K counters not reset"


def test_tune_check_compares_only_bf16_out():
    out = torch.randn(4, 8).bfloat16()
    scratch = torch.zeros(2, 4, 8)
    hx._tune_check((out, scratch), (out.clone(), scratch + 5.0))  # fp32 scratch mismatch ignored
    with pytest.raises(AssertionError):
        hx._tune_check((out + 1, scratch), (out, scratch))
    for e in hx.KERNELS:
        assert (e.get("settings", {}).get("autotune_baseline_accuracy_check_fn") is hx._tune_check) \
            == (e["name"] != hx.QUANT)
