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
    assert list(K) == ["nvfp4_gemm", "nvfp4_gemm_splitk", "nvfp4_gemm_q", "nvfp4_gemm_q_splitk"]
    for e in hx.KERNELS:
        assert e["body"] is hx.BODIES[e["name"]] and callable(e["check"])
        assert all(r.startswith("^nvfp4_") for r in e["replaces"])
        assert "prod launch, pf" in e["baseline_name"]
        for c in e["cases"](QWEN27) + e["cases"](FLASH) + e["cases"](GLM):
            assert c["numtokens"] in hx.BUCKETS and set(c) == {"n", "k", "numtokens"}
            assert json.loads(hk.case_key(c)) == c
    # one owner of the oxide entry names (no double credit in the e2e estimate)
    assert [bool(e["replaces"]) for e in hx.KERNELS] == [True, False, False, False]


def test_cases_per_model():
    dense = list(dict.fromkeys(ng.SHAPES.values()))  # qwen 27B: the per-arch table
    assert len(K["nvfp4_gemm"]["cases"](QWEN27)) == len(dense) * len(hx.BUCKETS)
    q = K["nvfp4_gemm_q"]["cases"](QWEN27)
    assert len(q) == (len(dense) + 1) * len(hx.BUCKETS)
    assert {"k": 5120, "n": 248320, "numtokens": 1} in q  # lm_head: prequant route only
    assert [c["numtokens"] for c in q[:len(dense) + 1]] == [1] * (len(dense) + 1)  # decode first
    assert all(c["n"] != 248320 for c in K["nvfp4_gemm"]["cases"](QWEN27))
    # split-K only where the oxide plan splits: never gate_up (34816) or the lm_head
    sk = K["nvfp4_gemm_q_splitk"]["cases"](QWEN27)
    assert sk and all(ng.splits_for(c["n"], c["k"], c["numtokens"]) > 1 for c in sk)
    assert all(c["n"] not in (34816, 248320) for c in sk)
    assert len(K["nvfp4_gemm_q"]["cases"](FLASH)) <= 64  # 11 TP2 shapes + lm_head
    assert {(c["n"], c["k"]) for c in K["nvfp4_gemm"]["cases"](GLM)} == {(3072, 6144), (6144, 1536)}
    assert (19360, 6144) in {(c["n"], c["k"]) for c in K["nvfp4_gemm_q"]["cases"](GLM)}
    for e in hx.KERNELS:
        assert e["cases"]({**QWEN27, "quant_format": "fp8_block"}) == []
    # a qwen 27B at TP2 is not the TP1 table: generic per-rank linears
    assert {(c["n"], c["k"]) for c in K["nvfp4_gemm"]["cases"]({**QWEN27, "tp": 2})} == {(7168, 5120)}


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


def test_prepare_auto_prefers_single_pass_then_split_then_oxide(shipped):
    table, _ = shipped
    table.update({("nvfp4_gemm_q", 1): {}, ("nvfp4_gemm_q_splitk", 1): {"split_k": 8},
                  ("nvfp4_gemm_q_splitk", 16): {"split_k": 16}})
    plan = hx.prepare(336, 10240, True, 40, "auto")
    assert len(plan) == 41 and plan[0] is None
    assert plan[1][0] == "nvfp4_gemm_q" and plan[1][2] == 0
    assert all(plan[m] is None for m in range(2, 9))  # bucket 8: oxide
    assert all(plan[m][:1] + plan[m][2:] == ("nvfp4_gemm_q_splitk", 16, 16)
               for m in range(9, 17))
    assert all(plan[m] is None for m in range(17, 41))
    assert hx.partial_elems(336, [plan]) == 16 * 16 * 336
    assert hx.describe(plan) == ("M1:nvfp4_gemm_q M2-8:oxide M9-16:nvfp4_gemm_q_splitk(split<=16) "
                                 "M17-40:oxide")


def test_prepare_helion_fails_closed(shipped):
    table, _ = shipped
    table.update({("nvfp4_gemm", b): {} for b in (1, 8)})
    assert hx.prepare(5120, 6144, False, 8, "helion")[8][0] == "nvfp4_gemm"
    with pytest.raises(RuntimeError, match="numtokens\":16.*on gpu"):
        hx.prepare(5120, 6144, False, 16, "helion")
    with pytest.raises(RuntimeError):
        hx.prepare(5120, 6144, False, 16, "oxide")


def test_twins_pass_body_layouts(shipped):
    table, calls = shipped
    table.update({("nvfp4_gemm", 8): {}, ("nvfp4_gemm_q_splitk", 8): {"split_k": 4}})
    m, n, k = 5, 64, 256
    hx.prepare(n, k, False, 8, "auto")
    hx.prepare(n, k, True, 8, "auto")
    out = torch.empty(m, n, dtype=torch.bfloat16)
    partial = torch.empty(4 * 8 * n)
    wsf = torch.zeros(128, k // 16, dtype=torch.uint8)  # uint8 accepted, viewed as e4m3
    w = torch.zeros(n, k // 2, dtype=torch.uint8)
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    hx.nvfp4_gemm_cuda(x, w, wsf, None, None, partial, out, 3.0, 0.5, 1, 0, 0, None)
    name, b, args = calls[-1]
    assert (name, b) == ("nvfp4_gemm", 8)
    assert [a.shape for a in args[:3]] == [(m, k // 16, 16), (n, k // 16, 8), (128 * k // 16,)]
    assert args[2].dtype == torch.float8_e4m3fn and args[3] is out
    assert args[4:] == (3.0, float(np.float32(3.0) / np.float32(6.0)), 0.5)
    xq = torch.zeros(m, k // 2, dtype=torch.uint8)
    hx.nvfp4_gemm_q_cuda(xq, wsf, w, wsf, partial, out, 0.25, 1, 0)
    name, b, args = calls[-1]
    assert name == "nvfp4_gemm_q_splitk" and args[5].shape == (4, m, n) and args[7] == 0.25
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
    q, split = name.startswith("nvfp4_gemm_q"), name.endswith("_splitk")
    args = hx.body_args(hx.problem(case, 0, "cpu", q), split)
    ref = (hx.reference_q if q else hx.reference)(*args)
    kw = {"config": helion.Config(split_k=2)} if split else {"autotune_effort": "none"}
    kern = helion.kernel(hx.BODIES[name], **kw)
    for _ in range(2):  # a replay: the split-K tickets must be back at zero
        out = kern(*args)
        ok, msg = hx.check(out, ref)
        assert ok, msg
        if split:
            cnt = [a for a in args if torch.is_tensor(a) and a.dtype == torch.int32]
            assert len(cnt) == 1 and not cnt[0].any(), "split-K counters not reset"
