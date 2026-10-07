# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_NVFP4_GEMM vLLM wiring: gate, eligibility routing, workspace,
VLLM_DISABLED_KERNELS union, entry point. (vLLM itself is not importable on
the CPU test host; the kernel class is exercised by the in-pod layer oracle.)"""
import os
os.environ["SUFFIX_NVFP4_GEMM_FUSED_MAX_M"] = "64"  # these tests exercise fusion at every M
import configparser
import pathlib
import sys

import pytest

from suffix_hybrid.kernels import nvfp4_linear as nl

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv("SUFFIX_NVFP4_GEMM", raising=False)
    before = set(sys.modules)
    assert nl.register() is None
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_eligibility():
    assert nl.eligible(34816, 5120, 34816, 0) is None
    assert nl.eligible(5120, 17408, 5120, 0) is None
    assert nl.eligible(2816, 2112, 2816, 0) is None  # gemma down
    # FlashInfer pads N to % 32 with zero rows: we read the first N rows
    assert nl.eligible(336, 10240, 352, 0) is None  # qwen-flash HC down+inject
    assert nl.eligible(48, 2560, 64, 0) is None
    assert nl.eligible(10240, 320, 10240, 0) is None  # HC up, K = 320
    assert "padded" in nl.eligible(5120, 5184, 5120, 32)
    assert "% 16" in nl.eligible(5128, 5120, 5152, 0)
    assert "% 64" in nl.eligible(5120, 5152, 5120, 0)
    assert "rows" in nl.eligible(128, 5120, 96, 0)


def test_max_m_env(monkeypatch):
    monkeypatch.delenv(nl.MAX_M_ENV, raising=False)
    assert nl.max_m_env() == 64
    monkeypatch.setenv(nl.MAX_M_ENV, "16")
    assert nl.max_m_env() == 16
    for bad in ("15", "65", "x", "-1"):
        monkeypatch.setenv(nl.MAX_M_ENV, bad)
        with pytest.raises(ValueError, match="16..64"):
            nl.max_m_env()


def test_route_table_parsing(monkeypatch):
    assert nl.parse_route("") == {}
    assert nl.parse_route("2560 x 3072:16, 336X10240:0,") == {(2560, 3072): 16,
                                                             (336, 10240): 0}
    for bad in ("2560x3072", "2560:16", "axb:1", "2560x3072:65", "1x2x3:4"):
        with pytest.raises(ValueError, match=nl.ROUTE_ENV):
            nl.parse_route(bad)
    monkeypatch.setattr(nl, "DEFAULT_ROUTE", {(8192, 2560): 40, (640, 2560): 16})
    monkeypatch.setenv(nl.ROUTE_ENV, "8192x2560:24")
    route = nl.route_table()  # env overrides the default table entry-wise
    assert route == {(8192, 2560): 24, (640, 2560): 16}
    assert nl.layer_max_m(8192, 2560, 64, route) == 24
    assert nl.layer_max_m(640, 2560, 64, route) == 16
    assert nl.layer_max_m(2560, 3072, 32, route) == 32  # unlisted: the env cap


def test_disable_earlier_is_a_union(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "Foo")
    added = nl._disable_earlier(["A", "Foo", "B"])
    assert added == ["A", "B"]
    import os
    assert os.environ["VLLM_DISABLED_KERNELS"] == "Foo,A,B"


def test_workspace_grows_at_load_only(monkeypatch):
    from suffix_hybrid.kernels import nvfp4_gemm as ng
    monkeypatch.setattr(nl, "_state", dict(nl._state, ws={}))
    a = nl._workspace("cpu", 5120, 17408)
    assert a[0].numel() == 64 * 17408 // 2 and a[1].numel() == 64 * 17408 // 16
    assert a[2].numel() == ng.partial_elems(5120, 17408) >= 5 * 16 * 5120
    b = nl._workspace("cpu", 336, 10240)  # narrow N, deep split
    assert b[0].numel() == 64 * 17408 // 2
    assert b[2].numel() == max(a[2].numel(), ng.partial_elems(336, 10240))
    assert nl._workspace("cpu", 128, 128) is b  # no realloc when it fits
    for m in range(1, 65):  # every M fits the workspace of its layer
        assert ng.plan(336, 10240, m)["partial"] <= b[2].numel()
    c = nl._workspace("cpu", 34816, 5120, max_m=16)  # capped M: 1 tile
    assert c[2].numel() >= ng.partial_elems(34816, 5120, 16)
    # split-K ticket counters: one per column tile of the widest layer, zeroed
    assert c[3].dtype.is_floating_point is False and c[3].numel() == 34816 // 32
    assert not c[3].any() and a[3].numel() == 5120 // 32


def test_entry_point():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    assert cfg["vllm.general_plugins"]["suffix_nvfp4_gemm"] == \
        "suffix_hybrid.kernels.nvfp4_linear:register"


def test_selection_verdict_fails_loud_on_bf16_checkpoint():
    msg = nl.selection_verdict(0, None, [])
    assert "TARGET checkpoint is NOT NVFP4-quantized" in msg and "NOT SELECTED" in msg
    msg = nl.selection_verdict(0, "modelopt_fp4", ["A"])
    assert "modelopt_fp4" in msg and "VLLM_DISABLED_KERNELS=A" in msg
    assert nl.selection_verdict(3, "modelopt_fp4", []) is None


def test_verdict_gemma_moe_only_checkpoint():
    # silicon 02d97b8a gemma-spec-dev: target modelopt_fp4, MTP drafter bf16,
    # NVFP4 only in the 30 MoE layers, dense linears bf16
    census = {"linear:UnquantizedLinearMethod": 180, "moe:ModelOptNvFp4FusedMoE": 30}
    msg = nl.selection_verdict(0, "modelopt_fp4", [], census, draft_quant=None)
    assert "quantizes only MoE experts" in msg and "30 quantized FusedMoE" in msg
    assert "linear:UnquantizedLinearMethod x180" in msg


class _Cfg:
    def __init__(self, q):
        self.quantization = q


class _Spec:
    target_model_config = _Cfg("modelopt_fp4")
    draft_model_config = _Cfg(None)


class _Vcfg:
    model_config = _Cfg(None)  # the drafter's config is current at first forward
    speculative_config = _Spec()


def test_target_quantization_prefers_spec_target():
    assert nl.target_quantization(_Vcfg()) == ("modelopt_fp4", None)
    assert nl.target_quantization(None) == (None, None)


def test_layer_census_classifies_linear_and_moe():
    class UnquantizedLinearMethod: pass
    class ModelOptNvFp4FusedMoE: pass
    import torch

    class RowParallelLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = UnquantizedLinearMethod()

    class FusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = ModelOptNvFp4FusedMoE()
    objs = [RowParallelLinear(), RowParallelLinear(), FusedMoE(), object()]
    assert nl.layer_census(objs) == {"linear:UnquantizedLinearMethod": 2,
                                     "moe:ModelOptNvFp4FusedMoE": 1}


def test_first_forward_hook_raises_when_not_selected(monkeypatch):
    import torch
    monkeypatch.setattr(nl, "_state", dict(nl._state, instances=0, checked=False,
                                           shapes={}, hook=None))
    h = torch.nn.modules.module.register_module_forward_pre_hook(nl._first_forward_check)
    nl._state["hook"] = h
    with pytest.raises(RuntimeError, match="NOT SELECTED"):
        torch.nn.Linear(2, 2)(torch.zeros(1, 2))
    torch.nn.Linear(2, 2)(torch.zeros(1, 2))  # hook removed: no second raise


def test_first_forward_hook_logs_selection(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(nl, "_state", dict(nl._state, instances=4, checked=False,
                                           shapes={"5120x17408:ours": 64}, hook=None))
    nl._state["hook"] = torch.nn.modules.module.register_module_forward_pre_hook(
        nl._first_forward_check)
    torch.nn.Linear(2, 2)(torch.zeros(1, 2))
    err = capsys.readouterr().err
    assert "NVFP4-GEMM SELECTION" in err and "5120x17408:ours x64" in err


# --- M-dependent dispatch through torch.ops.suffix_nvfp4.linear ------------
import torch  # noqa: E402


class _StubKernel:
    """CPU stand-in for SuffixNvFp4LinearKernel: ours -> 1s, stock -> 2s."""

    def __init__(self):
        self.calls = []

    def _ours(self, layer, cfg, x2d, qa2d=None):
        m = (x2d if qa2d is None else qa2d[0]).shape[0]
        self.calls.append(("ours", m, qa2d is not None))
        return torch.ones(m, cfg["n"], dtype=torch.bfloat16)

    def _stock(self, layer, x2d, xsf=None):
        self.calls.append(("stock", x2d.shape[0], xsf is not None))
        return torch.full((x2d.shape[0], layer._sfx_nvfp4["n"]), 2.0, dtype=torch.bfloat16)


def _stub_layer(monkeypatch, n=48, max_m=64, name="model.layers.0.mlp.down_proj"):
    import types
    monkeypatch.setattr(nl, "_state", dict(nl._state, layers={}, captured={},
                                           graph_logged=set(), active_logged=True))
    kern, layer = _StubKernel(), types.SimpleNamespace(_sfx_nvfp4={"n": n, "max_m": max_m})
    layer._sfx_nvfp4["key"] = nl.register_layer(kern, layer, name)
    return kern, layer, nl.register_op()


def test_op_fake_shape_and_dtype():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    op = nl.register_op()
    with FakeTensorMode():
        for x, xsf in ((torch.empty(7, 256, dtype=torch.bfloat16), None),
                       (torch.empty(7, 128, dtype=torch.uint8), torch.empty(128, 4))):
            out = op(x, xsf, 48, "unused")
            assert out.shape == (7, 48) and out.dtype == torch.bfloat16


def test_op_routes_per_m_both_inputs(monkeypatch):
    import torch
    kern, layer, op = _stub_layer(monkeypatch, max_m=40)
    key = layer._sfx_nvfp4["key"]
    for m, want in ((1, 1.0), (40, 1.0), (41, 2.0), (200, 2.0), (0, None)):
        out = op(torch.zeros(m, 256, dtype=torch.bfloat16), None, 48, key)
        assert out.shape == (m, 48) and (m == 0 or float(out[0, 0]) == want)
        out = op(torch.zeros(m, 128, dtype=torch.uint8), torch.zeros(128, 4), 48, key)
        assert out.shape == (m, 48) and (m == 0 or float(out[0, 0]) == want)
    assert [c[0] for c in kern.calls] == ["ours"] * 4 + ["stock"] * 6
    assert [c[2] for c in kern.calls] == [False, True] * 5  # prequant reaches both paths


def test_register_layer_keys_are_unique_and_stable(monkeypatch):
    import types
    _, layer, _ = _stub_layer(monkeypatch)
    other = types.SimpleNamespace(_sfx_nvfp4=None)
    k2 = nl.register_layer(object(), other, "model.layers.0.mlp.down_proj")
    assert k2 == "model.layers.0.mlp.down_proj#1"
    assert nl.register_layer(object(), other, "x") == k2  # reload keeps the key


def _compiled(fn):
    import torch
    torch._dynamo.reset()
    # vLLM VLLM_COMPILE: fullgraph, dims marked dynamic, every guard skipped
    return torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager",
                         options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe})


def test_compiled_dispatch_is_not_baked(monkeypatch):
    """Trace at M=200 (vLLM traces at max_num_batched_tokens), then M=1: the
    op must still pick ours. The old Python branch, compiled the same way,
    bakes the stock path (the 2026-10-07 silicon finding) — asserted too, so
    this test keeps proving what it claims."""
    import torch
    kern, layer, op = _stub_layer(monkeypatch)
    cfg = layer._sfx_nvfp4

    def new(x):  # what SuffixNvFp4LinearKernel.apply_weights traces now
        return op(x.reshape(-1, x.shape[-1]), None, cfg["n"], cfg["key"]).view(
            *x.shape[:-1], cfg["n"]) * 3

    def old(x):  # the pre-fix Python branch
        rows = x.numel() // x.shape[-1]
        y = (kern._ours(layer, cfg, x) if rows <= cfg["max_m"]
             else kern._stock(layer, x))
        return y * 3

    for fn, want_small in ((new, "ours"), (old, "stock")):
        kern.calls.clear()
        c = _compiled(fn)
        big = torch.zeros(200, 256, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(big, 0)
        assert c(big).shape == (200, 48)
        small = c(torch.zeros(1, 256, dtype=torch.bfloat16))
        assert small.shape == (1, 48)
        assert float(small[0, 0]) == (3.0 if want_small == "ours" else 6.0), fn.__name__
        if fn is new:  # runtime calls: trace-time fake calls never hit the kernel
            assert [c[:2] for c in kern.calls] == [("stock", 200), ("ours", 1)]


def test_in_graph_log_once_per_captured_m(monkeypatch, capsys):
    import types
    import torch
    kern, layer, op = _stub_layer(monkeypatch, max_m=64)
    k2 = types.SimpleNamespace(_sfx_nvfp4={"n": 48, "max_m": 16})
    k2._sfx_nvfp4["key"] = nl.register_layer(kern, k2, "l1")
    monkeypatch.setattr(nl, "_capturing", lambda: True)
    x = torch.zeros(32, 256, dtype=torch.bfloat16)
    op(x, None, 48, layer._sfx_nvfp4["key"])  # M=32: only the max_m=64 layer is ours
    op(x, None, 48, k2._sfx_nvfp4["key"])     # stock: not counted
    op(x[:8], None, 48, layer._sfx_nvfp4["key"])  # M=8: 1 of 2 so far -> no line
    err = capsys.readouterr().err
    assert "NVFP4-GEMM in-graph: 1 layers captured on ours at M=32" in err
    assert "at M=8" not in err
    op(x[:8], None, 48, k2._sfx_nvfp4["key"])
    op(x[:8], None, 48, k2._sfx_nvfp4["key"])  # FULL + PIECEWISE re-capture: no repeat
    err = capsys.readouterr().err
    assert err.count("NVFP4-GEMM in-graph: 2 layers captured on ours at M=8") == 1


# --- load-time cold-L2 autoroute (timing mocked: the GPU path is in-pod) ---
def _times(fi, **ours):
    """{route: {"fi": {M: us}, "pf<p>": {M: us}}} for both routes."""
    ms = sorted(fi)
    t = {"fi": dict(fi), **{k: dict(zip(ms, v)) for k, v in ours.items()}}
    return {r: {c: dict(v) for c, v in t.items()} for r in nl.ROUTES}


def test_autoroute_ms_clips_to_cap():
    assert nl.autoroute_ms(64) == [1, 2, 4, 8, 16, 24, 32, 48, 64]
    assert nl.autoroute_ms(40) == [1, 2, 4, 8, 16, 24, 32, 40]
    assert nl.autoroute_ms(16) == [1, 2, 4, 8, 16]


def test_decide_prefix_margin_and_pf():
    fi = {1: 100.0, 2: 100.0, 4: 100.0, 8: 100.0}
    # pf0 wins to M=2 (98 at M=4 is inside the 3 % margin: not a win)
    t = _times(fi, pf0=[90.0, 96.0, 98.0, 80.0], pf8=[95.0, 99.0, 99.0, 99.0])
    assert nl.decide(t, 0.03) == {"bf16": (2, 0), "prequant": (2, 0)}
    assert nl.decide(t, 0.0)["bf16"] == (8, 0)  # contiguous prefix to the end
    # the faster lookahead (summed over M) is the one served and judged
    t = _times(fi, pf0=[200.0] * 4, pf8=[50.0, 60.0, 70.0, 101.0])
    assert nl.decide(t, 0.03)["bf16"] == (4, 8)
    # never winning at M=1 -> FlashInfer only, even if later M would win
    t = _times(fi, pf0=[150.0, 10.0, 10.0, 10.0])
    assert nl.decide(t, 0.03)["bf16"] == (0, 0)
    # routes decide independently (prequant skips our quant: can win alone)
    t = _times(fi, pf0=[150.0] * 4)
    t["prequant"]["pf0"] = {m: 50.0 for m in fi}
    assert nl.decide(t, 0.03) == {"bf16": (0, 0), "prequant": (8, 0)}


def test_margin_env(monkeypatch):
    monkeypatch.delenv(nl.MARGIN_ENV, raising=False)
    assert nl.margin_env() == 0.03
    monkeypatch.setenv(nl.MARGIN_ENV, "0.1")
    assert nl.margin_env() == 0.1
    for bad in ("x", "-0.1", "1.5"):
        monkeypatch.setenv(nl.MARGIN_ENV, bad)
        with pytest.raises(ValueError, match=nl.MARGIN_ENV):
            nl.margin_env()


def test_autoroute_default_on(monkeypatch):
    monkeypatch.delenv(nl.AUTOROUTE_ENV, raising=False)
    assert nl.autoroute_on()
    monkeypatch.setenv(nl.AUTOROUTE_ENV, "0")
    assert not nl.autoroute_on()


def test_cache_roundtrip_and_corrupt_file(monkeypatch, tmp_path):
    path = tmp_path / "sub" / "ar.json"
    monkeypatch.setenv(nl.CACHE_ENV, str(path))
    assert nl.cache_get("k") is None  # no file
    t = _times({1: 10.0, 8: 20.0}, pf0=[5.0, 6.0])
    nl.cache_put("k", t)
    nl.cache_put("k2", t)  # merge, not overwrite
    assert nl.cache_get("k") == t and nl.cache_get("k2") == t  # int M keys restored
    path.write_text("{not json")
    assert nl.cache_get("k") is None
    nl.cache_put("k3", t)  # a corrupt file is replaced, never fatal
    assert nl.cache_get("k3") == t


def test_autoroute_shape_times_once_then_caches(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(nl.CACHE_ENV, str(tmp_path / "ar.json"))
    monkeypatch.delenv("SUFFIX_NVFP4_GEMM_PF", raising=False)
    monkeypatch.setattr(nl, "_state", dict(nl._state, ar_s=0.0, sha="abc123"))
    calls = []

    def time_fn(ms, pfs):  # ours wins to M=8 on qwen-like cold numbers
        calls.append((tuple(ms), tuple(pfs)))
        fi = {m: 40.0 + m for m in ms}
        return {r: {"fi": fi, **{f"pf{p}": {m: (30.0 if m <= 8 else 90.0) - p for m in ms}
                                 for p in pfs}} for r in nl.ROUTES}

    dec = nl.autoroute_shape(34816, 5120, 64, "RTX 5090", time_fn)
    assert dec == {"bf16": (8, 8), "prequant": (8, 8)}
    assert calls == [((1, 2, 4, 8, 16, 24, 32, 48, 64), (0, 8))]
    err = capsys.readouterr().err
    assert ("AUTOROUTE 34816x5120: ours<=M8 bf16 (pf 8) / ours<=M8 prequant (pf 8) "
            "(ours/fi us bf16|prequant at M=1: 22.0/41.0|22.0/41.0, M=2:") in err
    assert "[timed" in err
    assert nl.autoroute_shape(34816, 5120, 64, "RTX 5090", time_fn) == dec
    assert len(calls) == 1 and "[cached]" in capsys.readouterr().err
    # another GPU / cubin / plan is another key: timed again
    nl.autoroute_shape(34816, 5120, 64, "RTX PRO 6000", time_fn)
    monkeypatch.setenv("SUFFIX_NVFP4_GEMM_PF", "0")  # pinned lookahead: one candidate
    nl.autoroute_shape(34816, 5120, 64, "RTX 5090", time_fn)
    assert calls[1:] == [(calls[0][0], (0, 8)), (calls[0][0], (0,))]


def test_autoroute_budget_sends_untimed_shapes_to_flashinfer(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(nl.CACHE_ENV, str(tmp_path / "ar.json"))
    monkeypatch.setattr(nl, "_state", dict(nl._state, ar_s=nl.AUTOROUTE_BUDGET_S + 1, sha=""))
    dec = nl.autoroute_shape(5120, 17408, 64, "g", lambda ms, pfs: pytest.fail("timed"))
    assert dec == {"bf16": (0, 0), "prequant": (0, 0)}
    assert "budget" in capsys.readouterr().err


def test_autoroute_summary(monkeypatch):
    monkeypatch.setattr(nl, "_state", dict(nl._state, ar_s=2.5, autoroute={
        (1, 1): {"bf16": (8, 0), "prequant": (16, 8)},
        (2, 2): {"bf16": (0, 0), "prequant": (0, 0)},
        (3, 3): {"bf16": (0, 0), "prequant": (4, 0)}}))
    line = nl.autoroute_summary()
    assert "3 shapes, ours serves 1 (bf16) / 2 (prequant), FlashInfer only 1" in line
    monkeypatch.setattr(nl, "_state", dict(nl._state, autoroute={}))
    assert nl.autoroute_summary() is None


def test_op_routes_by_input_route_max(monkeypatch):
    """bf16 and prequant inputs have their own autoroute max M."""
    kern, layer, op = _stub_layer(monkeypatch, max_m=0)
    layer._sfx_nvfp4["max_m_q"] = 8
    key = layer._sfx_nvfp4["key"]
    op(torch.zeros(4, 256, dtype=torch.bfloat16), None, 48, key)
    op(torch.zeros(4, 128, dtype=torch.uint8), torch.zeros(128, 4), 48, key)
    op(torch.zeros(9, 128, dtype=torch.uint8), torch.zeros(128, 4), 48, key)
    assert [c[0] for c in kern.calls] == ["stock", "ours", "stock"]
