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
    assert "padded" in nl.eligible(128, 5120, 96, 0)  # N padded beyond output
    assert "padded" in nl.eligible(5120, 5184, 5120, 32)
    assert "% 32" in nl.eligible(5136, 5120, 5136, 0)


def test_disable_earlier_is_a_union(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "Foo")
    added = nl._disable_earlier(["A", "Foo", "B"])
    assert added == ["A", "B"]
    import os
    assert os.environ["VLLM_DISABLED_KERNELS"] == "Foo,A,B"


def test_workspace_grows_at_load_only(monkeypatch):
    monkeypatch.setattr(nl, "_state", dict(nl._state, ws={}))
    a = nl._workspace("cpu", 5120, 17408)
    assert a[0].numel() == 16 * 17408 // 2 and a[2].numel() == 8 * 16 * 5120
    b = nl._workspace("cpu", 34816, 5120)  # wider N, shorter K
    assert b[0].numel() == 16 * 17408 // 2 and b[2].numel() == 8 * 16 * 34816
    assert nl._workspace("cpu", 128, 128) is b  # no realloc when it fits


def test_entry_point():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    assert cfg["vllm.general_plugins"]["suffix_nvfp4_gemm"] == \
        "suffix_hybrid.kernels.nvfp4_linear:register"


def test_selection_verdict_fails_loud_on_bf16_checkpoint():
    msg = nl.selection_verdict(0, None, [])
    assert "NOT NVFP4-quantized" in msg and "NOT SELECTED" in msg
    msg = nl.selection_verdict(0, "modelopt_fp4", ["A"])
    assert "modelopt_fp4" in msg and "VLLM_DISABLED_KERNELS=A" in msg
    assert nl.selection_verdict(3, "modelopt_fp4", []) is None


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
