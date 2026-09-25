"""suffix_hybrid.step_profiler: env parse, trace summary, fail-soft wrap."""
import types

import pytest

from suffix_hybrid import step_profiler as sp


def test_parse_env():
    assert sp.parse_env("40:300") == (40, 300)
    assert sp.parse_env("20") == (20, 50)
    for bad in (None, "", "x", "0:5", "5:-1", "1:2:3"):
        assert sp.parse_env(bad) is None


def test_categories():
    assert sp.category("nvfp4_attn_partial") == "attn K2 own (verify)"
    assert sp.category("flashinfer::BatchPrefillWithPagedKVCacheKernel<>"
                       ).startswith("attn FlashInfer")
    assert sp.category("flashinfer::cutlass_moe_gemm") == "MoE"
    assert sp.category("kernel_unified_attention_2d").startswith("attn other")
    assert sp.category("Memcpy DtoH (Device -> Pinned)") == "copies/memset"


def _x(name, cat, ts, dur, tid=1, **args):
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur,
            "tid": tid, "args": args}


def test_summarize_synthetic_trace():
    ev = [
        _x(sp.STEP, "user_annotation", 0, 1000),
        _x(sp.STEP, "user_annotation", 1200, 1000),
        _x("attn.build:FlashInferMetadataBuilder", "user_annotation",
           100, 300),
        _x("cudaStreamSynchronize", "cuda_runtime", 150, 50),
        _x("cudaGraphLaunch", "cuda_runtime", 500, 10),
        _x("nvfp4_attn_partial", "kernel", 510, 200, tid=7,
           **{"graph id": 3}),
        _x("fused_moe_kernel", "kernel", 600, 300, tid=8),  # overlaps
        _x("Memcpy DtoH (Device -> Pinned)", "gpu_memcpy", 950, 10, tid=7),
        _x("nvfp4_attn_partial", "kernel", 1300, 400, tid=7),
    ]
    lines = sp.summarize({"traceEvents": ev},
                         {"reqs": 2, "toks": 18, "emitted": 5,
                          "cg": sp.Counter({"target:FULLx1": 2})})
    text = "\n".join(lines)
    assert "2 steps" in text
    assert "5.0 emitted tokens/step (2.50 per req-step" in text
    # busy = union: [510,900) + [950,960) + [1300,1700) = 800 us
    assert "gpu busy 0.400" in text
    assert "attn.build:FlashInferMetadataBuilder / cudaStreamSynchronize " \
           "0.5 0.025" in text
    assert "D2H copies/step 0.5" in text
    assert "inside graphs seen: 1 of 3" in text
    assert "target:FULLx1: 2" in text
    assert "attn K2 own (verify) 0.300" in text


class _Sess(sp._Session):
    def __init__(self, fail_start=False):
        super().__init__(n=2, skip=1)
        self.fail_start, self.started, self.stopped = fail_start, 0, 0

    def start(self, engine):
        if self.fail_start:
            raise RuntimeError("boom")
        self.started += 1
        self.phase = "prof"

    def stop(self):
        self.stopped += 1
        self.phase = "done"


def test_wrap_step_lifecycle_and_passthrough():
    calls = []

    def orig(self, x):
        calls.append(x)
        return x != "idle"

    sess = _Sess()
    step = sp.wrap_step(orig, sess)
    eng = types.SimpleNamespace()
    assert step(eng, "a") is True           # warm step 1 (skip=1)
    assert sess.phase == "skip"
    assert step(eng, "idle") is False       # still skip... start happens
    assert sess.started == 1                # executed>=skip -> start
    assert step(eng, "b") is True
    assert step(eng, "c") is True           # 2 profiled executed steps
    assert sess.stopped == 1 and sess.phase == "done"
    assert step(eng, "d") is True
    assert calls == ["a", "idle", "b", "c", "d"]


def test_wrap_step_fail_soft():
    def orig(self):
        return True

    sess = _Sess(fail_start=True)
    step = sp.wrap_step(orig, sess)
    assert step(None) is True and step(None) is True and step(None) is True
    assert sess.phase == "done"

    def raising(self):
        raise ValueError("engine error must propagate")

    with pytest.raises(ValueError):
        sp.wrap_step(raising, _Sess())(None)


def test_patch_is_idempotent(monkeypatch):
    monkeypatch.setenv(sp.ENV, "3:0")

    class EngineCoreProc:
        def _process_engine_step(self):
            return True

    mod = types.SimpleNamespace(EngineCoreProc=EngineCoreProc)
    sp._patch(mod)
    first = EngineCoreProc._process_engine_step
    sp._patch(mod)
    assert EngineCoreProc._process_engine_step is first
    assert first.__wrapped__ is not None
