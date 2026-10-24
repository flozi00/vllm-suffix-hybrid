# SPDX-License-Identifier: Apache-2.0
"""dsv41_swiglu: clamped SwiGLU + MXFP8 quantisation of the shared-expert activation."""
import json
import os
import re

os.environ.setdefault("HELION_INTERPRET", "1")

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_common as cm  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_swiglu as sw  # noqa: E402
from test_dsv41_stub import backends, lower_to_triton, run_body  # noqa: E402


def run(monkeypatch, gate_up, backend="stub", blocks=None, limit=sw.LIMIT):
    return run_body(monkeypatch, sw, sw.dsv41_swiglu_quant, (gate_up, limit), blocks, backend)


def blocks_of(backend, *vals):
    return dict(enumerate(vals)) if backend == "stub" else {"block_sizes": list(vals)}


def vllm_native(x, limit):
    """SiluAndMulWithClamp.forward_native (vllm/model_executor/layers/activation.py)."""
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=limit)
    up = torch.clamp(x[..., d:], min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


def test_reference_is_the_stock_activation_up_to_bf16_rounding():
    gu = sw.make_problem(37, 288, 1)
    want = vllm_native(gu.float(), sw.LIMIT)
    got = sw.swiglu_reference(gu).float()
    # the CUDA packed path rounds the silu to bf16 before the multiply (native computes in bf16
    # and rounds twice): both are within two bf16 ulps of the f32 answer
    assert (got - want).abs().max() <= 2 * 2.0 ** -7 * want.abs().max()
    assert float((gu.float()[:, :288] > sw.LIMIT).float().mean()) > 0.001        # the clamp engages


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("t,d,bt,bg", [(1, 288, 1, 4), (5, 288, 2, 8), (6, 64, 4, 1), (130, 288, 16, 2),
                                       (3, 256, 2, 2), (128, 96, 8, 4)])
def test_body_is_bit_exact_on_cpu(monkeypatch, backend, t, d, bt, bg):
    gu = sw.make_problem(t, d, seed=t + d)
    q, qs = run(monkeypatch, gu, backend, blocks_of(backend, bt, bg))
    ok, msg = sw.judge((q, qs), gu, exact=True)
    assert ok, msg
    assert q.shape == (t, d) and qs.shape == (cm.mxfp8_scale_bytes(t, d),)
    q_ref, qs_ref = sw.reference(gu)
    assert torch.equal(qs, qs_ref) and torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def test_padding_bytes_are_zero_over_dirty_scratch(monkeypatch):
    """d = 288 -> 9 groups, scales pad to 12 columns; M = 130 pads to 256 rows: every padding byte
    is rewritten by the kernel, never inherited from the allocator."""
    gu = sw.make_problem(130, 288, seed=3)
    real = torch.empty
    junk = lambda *a, **k: real(*a, **k).fill_(0x55) if k.get("dtype") == torch.uint8 else real(*a, **k)
    monkeypatch.setattr(torch, "empty", junk)
    q, qs = run(monkeypatch, gu, blocks={0: 4, 1: 2})
    assert sw.judge((q, qs), gu, exact=True)[0]


def test_judge_catches_errors(monkeypatch):
    gu = sw.make_problem(20, 288, seed=5)
    q, qs = run(monkeypatch, gu, blocks={0: 4, 1: 4})
    assert sw.judge((q, qs), gu, exact=True)[0]
    bad = (q.view(torch.uint8) ^ 1).view(torch.float8_e4m3fn)
    assert not sw.judge((bad, qs), gu, exact=True)[0]
    assert not sw.judge((q, qs ^ 1), gu, exact=True)[0]
    dirty = qs.clone()
    dirty[-1] = 7                                           # a padding byte
    assert not sw.judge((q, dirty), gu)[0]
    # tolerance mode (GPU): a handful of one-step differences pass, a systematic error does not
    few = q.view(torch.uint8).clone()
    few[0, :3] ^= 1
    assert sw.judge((few.view(torch.float8_e4m3fn), qs), gu)[0]
    assert not sw.judge((q, torch.roll(qs, 1)), gu)[0]


def test_wrapper_validation_and_warm(monkeypatch):
    gu = sw.make_problem(6, 288, seed=1)
    with pytest.raises(ValueError, match="need a contiguous bf16"):
        sw.swiglu_quant(gu.float())
    with pytest.raises(ValueError, match="multiple of 32"):
        sw.swiglu_quant(torch.zeros(2, 100, dtype=torch.bfloat16))
    monkeypatch.setattr(sw, "_WARM", set())
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, c: (lambda *a: run_body(
        monkeypatch, sw, body, a, {0: 2, 1: 4})))
    q, qs = sw.swiglu_quant(gu)
    assert sw.judge((q, qs), gu, exact=True)[0]
    monkeypatch.setattr(sw, "_WARM", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="first used while capturing"):
        sw.swiglu_quant(gu)


@pytest.mark.parametrize("t", [1, 6, 192])
def test_body_lowers_to_triton_without_cross_cta_sync(t):
    code = lower_to_triton(sw.dsv41_swiglu_quant, (sw.make_problem(t, 288, 0), 10.0),
                           sw.default_config(sw.case(t)))
    assert "tl.float8e4nv" in code and "libdevice.exp" in code      # accurate expf, like the CUDA op
    # a CTA-level tl.debug_barrier() (__syncthreads, emitted by Helion 1.5.0) is not cross-CTA sync
    stripped = code.replace("tl.debug_barrier()", "")
    for banned in ("atomic", "barrier", "while ", "semaphore", "spin", "gdc_"):
        assert banned not in stripped, banned


def test_search_space_abi_and_cases():
    pytest.importorskip("helion")
    import helion
    from helion.runtime.ref_mode import RefMode
    bound = helion.kernel(sw.dsv41_swiglu_quant, static_shapes=False, ref_mode=RefMode.OFF).bind(
        (sw.make_problem(192, 288, 0), 10.0))
    assert [(b.min_size, b.max_size) for b in bound.config_spec.block_sizes] == list(sw.SEARCH_SPACE[sw.KERNEL].values())
    model = {"architecture": "DeepseekV41ForCausalLM", "tp": 8}
    got = sw.cases(model)
    assert [c["numtokens"] for c in got] == list(sw.BUCKETS) and {c["d"] for c in got} == {288}
    json.dumps(got)
    assert sw.cases(dict(model, tp=5)) == [] and sw.cases({"architecture": "Qwen3ForCausalLM"}) == []
    (ent,) = sw.KERNELS
    assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"]) and ent["body"].__qualname__ == ent["name"]
    assert any(re.search(p, "act_and_mul_kernel", re.I) for p in ent["replaces"])
