# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_MOE_ROUTE wiring (no vLLM, AITER or GPU): a deferred router
job is consumed by aiter's sort exactly once (fused when the call is eligible, else the
router's own top-k and then the stock sort), a stale job fails loudly instead of routing on
stale ids, and the stage-1 quant stash only answers for the hidden_states it quantized.
The kernel's bits are checked on silicon: boot gate moe_route_bench."""
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch

if importlib.util.find_spec("vllm") is None:  # CPU CI: the module only needs @jit at import
    _jit = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)  # noqa: E731
    _tu = types.ModuleType("vllm.triton_utils")
    _tu.tl, _tu.triton = SimpleNamespace(), SimpleNamespace(jit=_jit)
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.triton_utils": _tu})
    try:
        from suffix_hybrid.kernels import moe_route_rocm as mr
    finally:  # no fake vllm for the rest of the session
        del sys.modules["vllm"], sys.modules["vllm.triton_utils"]
from suffix_hybrid.kernels import moe_route_rocm as mr  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setitem(mr._ORIG, "sort", lambda *a, **k: log.append("sort") or "stock sorted")
    monkeypatch.setitem(mr._ORIG, "quant", lambda *a, **k: log.append("quant") or "stock quant")
    monkeypatch.setattr(mr, "_route", lambda *a: log.append("route") or "fused sorted")
    mr._JOBS.clear()
    mr._QUANT.clear()
    yield log
    mr._JOBS.clear()
    mr._QUANT.clear()


def _job(stock=None, m=5):
    return mr.Job(torch.zeros(m, 513, dtype=torch.bfloat16), torch.zeros(m, 2560, dtype=torch.bfloat16),
                  torch.zeros(m, 11), torch.zeros(m, 11, dtype=torch.int32), stock)


def test_sort_consumes_the_job_fused_or_stock(calls):
    job = _job()
    mr.defer(job)
    assert mr.moe_sorting(job.ids, job.weights, 513, 2560, torch.bfloat16, 32, None, None, 0,
                          accumulate=False) == "fused sorted"
    assert calls == ["route"] and not mr._JOBS
    ran = []
    job = _job(stock=lambda: ran.append(1))
    mr.defer(job)  # an expert mask (EP) is not the contract the kernel reproduces
    assert mr.moe_sorting(job.ids, job.weights, 513, 2560, torch.bfloat16, 32,
                          torch.ones(513)) == "stock sorted"
    assert ran == [1] and calls[-1] == "sort" and not mr._JOBS
    assert mr.moe_sorting(job.ids, job.weights, 513, 2560, torch.bfloat16) == "stock sorted"
    assert ran == [1]  # no job: plain stock sort, no second top-k


def test_stale_job_fails_loudly(calls):
    mr.defer(_job())
    with pytest.raises(RuntimeError, match="never reached"):
        mr.defer(_job())
    other = _job()
    with pytest.raises(RuntimeError, match="not the router's"):
        mr.moe_sorting(other.ids, other.weights, 513, 2560, torch.bfloat16)


def test_quant_stash_answers_only_for_its_hidden(calls):
    hidden, sorted_ids = torch.randn(5, 2560), torch.zeros(64, dtype=torch.int32)
    kw = dict(sorted_ids=sorted_ids, num_valid_ids=None, token_num=5, topk=11, block_size=32)
    mr._QUANT[sorted_ids.data_ptr()] = (hidden, "a1", "a1s")
    assert mr.quant_moe_sort(hidden.clone(), **kw) == "stock quant"  # another tensor: stock
    mr._QUANT[sorted_ids.data_ptr()] = (hidden, "a1", "a1s")
    assert mr.quant_moe_sort(hidden, **kw) == ("a1", "a1s")
    assert mr.quant_moe_sort(hidden, **kw) == "stock quant"  # used once (stage 2 is stock)


def test_scale_layout_is_a_bijection():
    addr = mr._scale_addr(torch.arange(64)[:, None], torch.arange(80)[None, :], 80)
    assert torch.equal(addr.flatten().sort().values, torch.arange(64 * 80))
