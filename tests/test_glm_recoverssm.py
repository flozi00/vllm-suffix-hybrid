"""SUFFIX_GLM_RECOVERSSM: pins, config enabling and wiring of suffix_hybrid/glm_recoverssm.py against the audited vLLM 0.33.0 sources (CPU only; the kernels are upstream's and are
checked on a GPU by `python -m suffix_hybrid.glm_recoverssm oracle`)."""
import types
from pathlib import Path

import pytest

from suffix_hybrid import glm_recoverssm as gr
from suffix_hybrid.kda_lazy import PatchDriftError

FIX = Path(__file__).resolve().parents[1] / "sm120/tests/fixtures/vllm_0.33.0"
GDN = FIX / "gdn_attn.py"
CFG = FIX / "config_vllm.py"
K3OPS = FIX / "k3_ops_recoverssm.py"
REAL_CHECK_K3_OPS = gr.check_k3_ops                      # before the autouse fixture stubs it
KDA = FIX / "glm5next_common_kda.py"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(gr, "_STATE", {"layers": [], "by_state": {}, "ctx": {}, "engaged": 0, "orig_frk": None, "ones": {}})
    monkeypatch.delenv(gr.ENV, raising=False)
    monkeypatch.setattr(gr, "check_k3_ops", lambda: "0.33.0")        # needs vllm; covered by test_k3_ops_pin below


def test_builder_pin_accepts_the_audited_source():
    assert gr.check_builder_source(GDN.read_text()) == "0.33.0"


def test_builder_drift_is_refused():
    src = GDN.read_text().replace("num_spec_decodes = spec_sequence_masks_cpu.sum().item()", "num_spec_decodes = int(spec_sequence_masks_cpu.sum())")
    with pytest.raises(PatchDriftError, match="matches no audited release"):
        gr.check_builder_source(src)
    with pytest.raises(PatchDriftError, match="does not parse"):
        gr.check_builder_source(GDN.read_text() + "\ndef (:\n")


def test_change_outside_build_is_fine():
    assert gr.check_builder_source(GDN.read_text().replace("class GDNAttentionMetadataBuilder(", "# note\nclass GDNAttentionMetadataBuilder(")) == "0.33.0"


def _vc(arch="Glm5NextForConditionalGeneration", spec=7, v2=True, pp=1, backend="MambaBackendEnum.TRITON", mode="align", sr=False):
    cc = types.SimpleNamespace(mamba_cache_mode=mode, use_replayssm=False, use_kda_recoverssm=False)
    return types.SimpleNamespace(model_config=types.SimpleNamespace(architecture=arch), cache_config=cc, num_speculative_tokens=spec, use_v2_model_runner=v2,
                                 parallel_config=types.SimpleNamespace(pipeline_parallel_size=pp),
                                 mamba_config=types.SimpleNamespace(backend=backend, enable_stochastic_rounding=sr))


def test_config_flags_are_set_for_glm_with_speculation():
    vc = _vc()
    assert gr.enable_config(vc) is True
    assert vc.cache_config.use_replayssm and vc.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("vc", [_vc(arch="Qwen3NextForCausalLM"), _vc(spec=0)])
def test_other_models_and_no_speculation_are_untouched(vc):
    assert gr.enable_config(vc) is False
    assert not vc.cache_config.use_replayssm and not vc.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("kw", [dict(v2=False), dict(pp=2), dict(backend="MambaBackendEnum.FLASHINFER"), dict(mode="all"), dict(sr=True)])
def test_unsupported_configurations_are_refused(kw):
    with pytest.raises(PatchDriftError):
        gr.enable_config(_vc(**kw))


def test_patch_args_applies_the_flags_after_vllms_own_validation():
    vc = _vc()

    class EngineArgs:
        def create_engine_config(self, usage_context=None, headless=False):
            return vc

    mod = types.SimpleNamespace(EngineArgs=EngineArgs)
    gr.patch_args(mod)
    assert mod.EngineArgs().create_engine_config() is vc and vc.cache_config.use_kda_recoverssm


def _fake_kda_module(tmp_path, backend="triton"):
    calls = []

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            self.kda_prefill_backend = backend
            self.local_num_heads, self.head_dim = 16, 128

    mod = types.SimpleNamespace(__file__=str(KDA), Glm5NextLinearAttention=Layer, fused_recurrent_kda=lambda **kw: calls.append(kw) or "orig")
    return mod, calls


def test_kda_patch_requires_the_flags(tmp_path):
    mod, _ = _fake_kda_module(tmp_path)
    gr.patch_kda(mod)
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(use_replayssm=False, use_kda_recoverssm=False))
    with pytest.raises(PatchDriftError, match="flags are not set"):
        mod.Glm5NextLinearAttention(None, cfg, "p")


def test_non_spec_calls_go_to_the_original_and_odd_spec_calls_are_refused(tmp_path):
    mod, calls = _fake_kda_module(tmp_path)
    gr.patch_kda(mod)
    one_d = types.SimpleNamespace(dim=lambda: 1)
    assert mod.fused_recurrent_kda(num_accepted_tokens=None, ssm_state_indices=one_d) == "orig" and len(calls) == 1
    two_d = types.SimpleNamespace(dim=lambda: 2)
    with pytest.raises(PatchDriftError):
        mod.fused_recurrent_kda(num_accepted_tokens="acc", ssm_state_indices=two_d, scale=1.0)


def test_kda_patch_refuses_a_drifted_layer_source(tmp_path):
    bad = tmp_path / "kda.py"
    bad.write_text(KDA.read_text().replace("safe_gate = self.kda_safe_gate", "safe_gate = self.kda_safe_gate  # changed"))
    mod, _ = _fake_kda_module(tmp_path)
    mod.__file__ = str(bad)
    with pytest.raises(PatchDriftError):
        gr.patch_kda(mod)


def test_unset_hook_is_inert():
    assert gr.install_post_import_hook() is None


def test_validator_whitelist_gets_the_glm_architectures_and_the_module_still_parses():
    import ast
    out = gr.rewrite_config_source(CFG.read_text())
    ast.parse(out)
    assert out.count(gr.NEW_KDA_ARCHS) == 1 and out.count(gr.OLD_KDA_ARCHS) == 0
    assert out.replace(gr.NEW_KDA_ARCHS, gr.OLD_KDA_ARCHS) == CFG.read_text()      # that one tuple is the only change


def test_validator_drift_is_refused():
    src = CFG.read_text().replace("use_mamba_replayssm_spec = (", "use_mamba_replayssm_spec = (  # changed")
    with pytest.raises(PatchDriftError, match="matches no audited release"):
        gr.rewrite_config_source(src)
    with pytest.raises(PatchDriftError):
        gr.rewrite_config_source(CFG.read_text().replace(gr.OLD_KDA_ARCHS, gr.OLD_KDA_ARCHS + gr.OLD_KDA_ARCHS.replace("kda_architectures", "kda_architectures2")))


def test_supports_replayssm_is_true_for_glm_and_unchanged_otherwise():
    class ModelConfig:
        def __init__(self, arch, registry):
            self.architecture, self._registry = arch, registry

        @property
        def supports_replayssm(self):
            return self._registry

    mod = types.SimpleNamespace(ModelConfig=ModelConfig)
    gr.patch_model_config(mod)
    assert ModelConfig("Glm5NextForConditionalGeneration", False).supports_replayssm is True
    assert ModelConfig("NemotronHForCausalLM", True).supports_replayssm is True
    assert ModelConfig("Qwen3NextForCausalLM", False).supports_replayssm is False
    with pytest.raises(PatchDriftError, match="not a property"):
        gr.patch_model_config(types.SimpleNamespace(ModelConfig=type("M", (), {"supports_replayssm": True})))


def test_update_block_table_gives_each_kv_group_its_own_context_block_table_and_width(monkeypatch):
    """KV groups 2..N get a dataclasses.replace() copy of group 1's metadata from update_block_table: it must carry THIS group's commit context and block table (the commit writes the
    final / boundary state by block id) and the state rows must keep the width of the verify window."""
    import dataclasses

    @dataclasses.dataclass
    class Md:
        spec_state_indices_tensor: object = None
        recoverssm_commit: object = None
        recoverssm_context: object = None

    class FakeIdx:
        def __init__(self, width):
            self.width = width

        def __getitem__(self, key):
            return FakeIdx(1)

        def expand(self, a, w):
            return ("expanded", w)

    class Builder:
        num_spec = 3
        use_spec_decode = True

        def build(self, *a, **kw):
            raise AssertionError

        def update_block_table(self, metadata, blk_table, slot_mapping):
            return dataclasses.replace(metadata, spec_state_indices_tensor=FakeIdx(1))

    mod = types.SimpleNamespace(__file__=str(GDN), GDNAttentionMetadataBuilder=Builder)
    gr.patch_builder(mod)
    group1 = Md(spec_state_indices_tensor=FakeIdx(4), recoverssm_commit=("req", ("table-of-group-1", "computed", 2304)), recoverssm_context="ctx-1")
    monkeypatch.setattr(gr, "_commit_context", lambda builder: "ctx-of-this-group")
    md = Builder().update_block_table(group1, "table-of-this-group", None)
    assert md.recoverssm_context == "ctx-of-this-group"
    assert md.recoverssm_commit == ("req", ("table-of-this-group", "computed", 2304))
    assert md.spec_state_indices_tensor == ("expanded", 4)
    # a metadata without a commit plan (non-spec step, other model) passes through untouched
    other = Md(spec_state_indices_tensor=FakeIdx(1))
    assert Builder().update_block_table(other, "t", None).spec_state_indices_tensor.width == 1


def test_k3_ops_pin(monkeypatch, tmp_path):
    real = REAL_CHECK_K3_OPS
    import importlib.util as iu
    monkeypatch.setattr(gr, "check_k3_ops", real)
    monkeypatch.setattr(iu, "find_spec", lambda name: types.SimpleNamespace(origin=str(K3OPS)))
    assert gr.check_k3_ops() == "0.33.0"
    bad = tmp_path / "recoverssm.py"
    bad.write_text(K3OPS.read_text().replace("commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)", "commit_len = query_len"))
    monkeypatch.setattr(iu, "find_spec", lambda name: types.SimpleNamespace(origin=str(bad)))
    with pytest.raises(PatchDriftError, match="matches no audited release"):
        gr.check_k3_ops()
