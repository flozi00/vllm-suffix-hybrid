import os
import types
from pathlib import Path

import pytest

from suffix_hybrid import nccl_split as ns

FIX = Path(__file__).resolve().parents[1] / "sm120/tests/fixtures/vllm_0.30.0/cuda_communicator.py"


def test_patch_applies_to_pinned_source_and_routes_by_size():
    out = ns.patch_all_reduce(FIX.read_text())
    assert "_suffix_small_nccl" in out and out.count("pynccl_comm.all_reduce(input_)") == 1


def test_drift_is_refused():
    src = FIX.read_text().replace(ns.OLD, ns.OLD.replace("assert pynccl_comm", "assert  pynccl_comm"))
    with pytest.raises(ns.PatchDriftError):
        ns.patch_all_reduce(src)


def test_global_nccl_algo_is_refused(monkeypatch):
    monkeypatch.setenv(ns.ENV, "allreduce:tree")
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    with pytest.raises(ns.PatchDriftError):
        ns.apply(types.SimpleNamespace(__file__=str(FIX)))


def test_unset_is_inert(monkeypatch):
    monkeypatch.delenv(ns.ENV, raising=False)
    assert ns.apply(object()) is None
