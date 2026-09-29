# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_NVFP4_GEMM vLLM wiring: gate, eligibility routing, workspace,
VLLM_DISABLED_KERNELS union, entry point. (vLLM itself is not importable on
the CPU test host; the kernel class is exercised by the in-pod layer oracle.)"""
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
