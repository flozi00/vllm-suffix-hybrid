"""SUFFIX_KDA_LAZY: the anchors, the pin and the wiring of suffix_hybrid/kda_lazy.py against the audited vLLM 0.33.0 glm5next KDA layer source (CPU only; the kernel itself
is proven on a GPU by `python -m suffix_hybrid.kda_lazy oracle`)."""
import types
from pathlib import Path

import pytest

from suffix_hybrid import kda_lazy as kl

FIX = Path(__file__).resolve().parents[1] / "sm120/tests/fixtures/vllm_0.33.0/glm5next_common_kda.py"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(kl, "_STATE", {"layers": 0, "spec_len": 0, "engaged": 0, "orig": None, "ctx": {}})
    monkeypatch.delenv(kl.ENV, raising=False)


def test_pinned_source_is_accepted():
    assert kl.check_source(FIX.read_text()) == "0.33.0"


@pytest.mark.parametrize("anchor", ["ANCHOR_SPEC", "ANCHOR_SPEC_ARGS", "ANCHOR_SCATTER", "ANCHOR_PLAIN"])
def test_missing_or_duplicated_anchor_is_refused(anchor):
    src, old = FIX.read_text(), getattr(kl, anchor)
    with pytest.raises(kl.PatchDriftError):
        kl.check_source(src.replace(old, old.replace("\n", "  # x\n", 1)))        # the anchor text is gone
    with pytest.raises(kl.PatchDriftError):
        kl.check_source(src.replace(old, old + '            _twice = """\n' + old + '"""\n'))   # the same text again (valid Python)


def test_unparsable_source_is_refused():
    with pytest.raises(kl.PatchDriftError, match="does not parse"):
        kl.check_source(FIX.read_text() + "\ndef (:\n")


def test_any_other_change_of_forward_is_refused_by_the_pin():
    src = FIX.read_text()
    with pytest.raises(kl.PatchDriftError, match="matches no audited release"):
        kl.check_source(src.replace("safe_gate = self.kda_safe_gate", "safe_gate = self.kda_safe_gate  # changed"))


def test_change_outside_forward_is_fine():
    assert kl.check_source(FIX.read_text().replace("class Glm5NextLinearAttention(", "# a comment\nclass Glm5NextLinearAttention(")) == "0.33.0"


def test_unset_is_inert():
    mod = object()
    assert kl.apply(mod) is None


class _Scatter:
    def __init__(self):
        self.calls = []
        self.warm = []

    def __call__(self, state, last, idx):
        self.calls.append(idx)

    def register_warmup(self, **kw):
        self.warm.append(kw)


def _fake_module(mode="none", backend="triton"):
    calls = []

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            self.kda_prefill_backend = backend
            self.num_spec = 7

    mod = types.SimpleNamespace(__file__=str(FIX), Glm5NextLinearAttention=Layer, scatter_states=_Scatter(),
                                fused_recurrent_kda=lambda **kw: calls.append(kw) or "orig")
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode=mode))
    return mod, calls, cfg


def _spec_kw(**over):
    kw = dict(q=1, k=2, v=3, g=4, beta=5, initial_state="S", use_qk_l2norm_in_kernel=True, cu_seqlens="cu", ssm_state_indices=types.SimpleNamespace(dim=lambda: 2),
              num_accepted_tokens="acc", out="O", sigmoid_beta=True, a_log="A", g_bias="B", compute_gate=True, lower_bound=-5.0)
    kw.update(over)
    return kw


def test_apply_routes_spec_calls_and_passes_everything_else_through(monkeypatch):
    monkeypatch.setenv(kl.ENV, "1")
    mod, calls, cfg = _fake_module()
    kl.apply(mod)
    mod.Glm5NextLinearAttention(None, cfg, "model.layers.0.self_attn")
    assert kl._STATE["layers"] == 1 and kl._STATE["spec_len"] == 8
    # plain decode (1-D indices, no num_accepted_tokens) and an old-style call: the original
    assert mod.fused_recurrent_kda(**_spec_kw(num_accepted_tokens=None, ssm_state_indices=types.SimpleNamespace(dim=lambda: 1))) == "orig"
    assert len(calls) == 1
    # spec call: the lazy context runs it and the state is returned like the stock (out, final_state)
    seen = {}

    class Ctx:
        spec_len = 8

        def run(self, *a, **kw):
            seen["args"] = a
            seen["kw"] = kw
            return "lazy-out"

    state = types.SimpleNamespace(data_ptr=lambda: 1, shape=(2, 16, 128, 128), dtype="fp32")
    monkeypatch.setattr(kl, "_ctx", lambda st, create=True: Ctx())
    out, st = mod.fused_recurrent_kda(**_spec_kw(initial_state=state, ssm_state_indices=types.SimpleNamespace(dim=lambda: 2, shape=(4, 8))))
    assert out == "lazy-out" and st is state and seen["kw"] == {"out": "O"} and len(calls) == 1


def test_unexpected_spec_call_is_refused(monkeypatch):
    monkeypatch.setenv(kl.ENV, "1")
    mod, calls, cfg = _fake_module()
    kl.apply(mod)
    with pytest.raises(kl.PatchDriftError):
        mod.fused_recurrent_kda(**_spec_kw(scale=0.5))
    with pytest.raises(kl.PatchDriftError):
        mod.fused_recurrent_kda(**_spec_kw(compute_gate=False))
    with pytest.raises(kl.PatchDriftError):
        mod.fused_recurrent_kda(**_spec_kw(lower_bound=None))


def test_scatter_states_clears_the_written_blocks_and_keeps_its_attributes(monkeypatch):
    monkeypatch.setenv(kl.ENV, "1")
    mod, _, cfg = _fake_module()
    orig = mod.scatter_states
    kl.apply(mod)
    mod.Glm5NextLinearAttention(None, cfg, "p")
    cleared = []
    monkeypatch.setattr(kl, "_ctx", lambda st, create=True: types.SimpleNamespace(clear=cleared.append))
    mod.scatter_states.register_warmup(state_shape=(1,))
    assert orig.warm == [{"state_shape": (1,)}]
    mod.scatter_states("S", "L", "IDX")
    assert orig.calls == ["IDX"] and cleared == ["IDX"]


def test_layer_configuration_is_checked(monkeypatch):
    monkeypatch.setenv(kl.ENV, "1")
    mod, _, cfg = _fake_module(mode="align")
    kl.apply(mod)
    with pytest.raises(kl.PatchDriftError, match="mamba_cache_mode"):
        mod.Glm5NextLinearAttention(None, cfg, "p")
    mod, _, cfg = _fake_module(backend="flashkda")          # flashkda's prefill checkpoint blocks sit behind the state columns: accepted
    kl.apply(mod)
    mod.Glm5NextLinearAttention(None, cfg, "p")


def test_drifted_module_fails_the_hook_closed(monkeypatch, tmp_path):
    monkeypatch.setenv(kl.ENV, "1")
    bad = tmp_path / "kda.py"
    bad.write_text(FIX.read_text().replace(kl.ANCHOR_SCATTER, ""))
    mod, _, _ = _fake_module()
    mod.__file__ = str(bad)
    with pytest.raises(kl.PatchDriftError):
        kl.apply(mod)
