"""suffix_hybrid.step_profiler: env parse, trace summary, fail-soft wrap."""
import types

import pytest

from suffix_hybrid import step_profiler as sp


def test_parse_env_and_buckets():
    assert sp.parse_env("40:300") == (40, 300, 1)
    assert sp.parse_env("40:300:2") == (40, 300, 2)
    assert sp.parse_env("20") == (20, 50, 1)
    for bad in (None, "", "x", "0:5", "5:-1", "5:1:0", "1:2:3:4"):
        assert sp.parse_env(bad) is None
    assert [sp.bucket(r) for r in (0, 1, 2, 6, 7, 12, 13, 24, 25, 32)] == [
        None, "c1", "c2-6", "c2-6", "c7-12", "c7-12", "c13-24", "c13-24",
        "c25+", "c25+"]


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
    def __init__(self, fail_start=False, per=1):
        super().__init__(n=2, skip=2, per=per)
        self.fail_start, self.started = fail_start, []

    def start(self, engine, label="-"):
        if self.fail_start:
            raise RuntimeError("boom")
        self._reset_window()
        self.started.append(label)
        self.label = label
        self.phase = "prof"

    def stop(self):
        self.windows += 1
        self.done[self.label] += 1
        self.stable = 0
        self.phase = "wait"


def _eng(running):
    return types.SimpleNamespace(
        scheduler=types.SimpleNamespace(running=[0] * running))


def test_windows_follow_load_buckets():
    calls = []

    def orig(self, x):
        calls.append(x)
        return x != "idle"

    sess = _Sess()
    step = sp.wrap_step(orig, sess)
    c1, c8 = _eng(1), _eng(8)
    for x in "ab":                       # 2 steady c1 steps
        assert step(c1, x) is True
    assert sess.started == []
    assert step(c1, "idle") is False     # window starts here (not counted)
    assert sess.started == ["c1"] and sess.phase == "prof"
    step(c1, "p1"), step(c1, "p2")       # 2 executed -> stop
    assert sess.phase == "wait" and sess.done["c1"] == 1
    for x in "cdef":                     # c1 already done: no new window
        step(c1, x)
    assert sess.started == ["c1"]
    step(c8, "g"), step(_eng(7), "h")    # bucket c7-12 held 2 steps
    step(c8, "i")
    assert sess.started == ["c1", "c7-12"]
    assert calls == ["a", "b", "idle", "p1", "p2", "c", "d", "e", "f", "g",
                     "h", "i"]


def test_wrap_step_fail_soft():
    def orig(self):
        return True

    sess = _Sess(fail_start=True)
    step = sp.wrap_step(orig, sess)
    for _ in range(5):
        assert step(_eng(1)) is True
    assert sess.phase == "done"
    assert step(object()) is True        # no scheduler attr: still serves

    def raising(self):
        raise ValueError("engine error must propagate")

    with pytest.raises(ValueError):
        sp.wrap_step(raising, _Sess())(_eng(1))


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
