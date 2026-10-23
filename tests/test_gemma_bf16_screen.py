"""The fixed Gemma diagnostic cannot silently run the unrelated default shapes."""

from suffix_hybrid.kernels import bf16_skinny as b


def test_gemma_screen_selects_pinned_shapes(monkeypatch):
    calls = []
    monkeypatch.setattr(b, "bench", lambda **kw: calls.append(kw))
    assert b.main(["gemma4"]) == 0
    assert calls == [{"shapes": b.GEMMA4_SHAPES, "ms": (1, 8, 32), "layers": 30}]
    assert (2816, 2112) in b.GEMMA4_SHAPES
    assert (16384, 1024) in b.GEMMA4_SHAPES


def test_default_screen_preserved_and_failure_is_nonzero(monkeypatch):
    calls = []
    monkeypatch.setattr(b, "bench", lambda **kw: calls.append(kw))
    assert b.main([]) == 0 and calls == [{}]

    def failed(**kw):
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(b, "bench", failed)
    assert b.main(["gemma4"]) == 1
