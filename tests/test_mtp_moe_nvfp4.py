"""SUFFIX_MTP_MOE_NVFP4: on-load NVFP4 conversion of the MTP layer's BF16 routed experts (CPU)."""
import types

import pytest
import torch

from suffix_hybrid import mtp_moe_nvfp4 as mm


def _stream():
    g = torch.Generator().manual_seed(0)
    w = lambda r, c, s=1.0: (torch.randn(r, c, generator=g) * s).to(torch.bfloat16)  # noqa: E731
    base = "model.layers.78.mlp.experts"
    yield "model.layers.77.mlp.experts.0.gate_proj.weight", torch.zeros(4, 8, dtype=torch.uint8)
    yield f"{base}.0.gate_proj.weight", w(4, 32)
    yield f"{base}.1.down_proj.weight", w(32, 4)
    yield f"{base}.1.gate_proj.weight", w(4, 32)
    yield f"{base}.0.up_proj.weight", w(4, 32, 3.0)   # larger amax: the pair's shared scale
    yield f"{base}.0.down_proj.weight", w(32, 4)
    yield f"{base}.1.up_proj.weight", w(4, 32)
    yield "model.layers.78.eh_proj.weight", w(8, 16)


def _quant(t, gscale):
    return torch.zeros(t.shape[0], t.shape[1] // 2, dtype=torch.uint8), \
        torch.zeros(t.shape[0], t.shape[1] // 16, dtype=torch.float8_e4m3fn)


def _reset():
    mm._state.update(keep={}, scales=None, tensors=0, saved=0, switched={})


def _convert(stream, n_experts=4, local=lambda e: True):
    calls = []

    def scales():
        calls.append(1)
        return 0.5, 0.25

    _reset()
    out = list(mm.convert(stream, 78, n_experts, scales, local, quant=_quant,
                          check=lambda *a: None, device=torch.device("cpu")))
    return out, calls


def test_convert_emits_the_nvfp4_set_with_one_gate_up_scale():
    out, calls = _convert(_stream())
    names = [n for n, _ in out]
    d = dict(out)
    assert calls == [1]
    assert names[0].startswith("model.layers.77.") and "model.layers.78.eh_proj.weight" in names
    for e in (0, 1):
        for p in ("gate_proj", "up_proj", "down_proj"):
            b = f"model.layers.78.mlp.experts.{e}.{p}"
            assert d[f"{b}.weight"].dtype == torch.uint8 and d[f"{b}.weight_scale"].dtype == torch.float8_e4m3fn
            assert d[f"{b}.weight_scale_2"].dim() == 0
            assert float(d[f"{b}.input_scale"]) == (0.25 if p == "down_proj" else 0.5)
        b = f"model.layers.78.mlp.experts.{e}"
        assert torch.equal(d[f"{b}.gate_proj.weight_scale_2"], d[f"{b}.up_proj.weight_scale_2"])
    # the up half of expert 0 had the larger amax: it sets the pair's scale (gate alone is smaller)
    g0 = float(d["model.layers.78.mlp.experts.0.gate_proj.weight_scale_2"])
    d0 = float(d["model.layers.78.mlp.experts.0.down_proj.weight_scale_2"])
    assert g0 > d0
    # experts never streamed (EP weight filter) still get the global input scales
    for e in (2, 3):
        assert float(d[f"model.layers.78.mlp.experts.{e}.gate_proj.input_scale"]) == 0.5
        assert float(d[f"model.layers.78.mlp.experts.{e}.down_proj.input_scale"]) == 0.25
        assert f"model.layers.78.mlp.experts.{e}.gate_proj.weight" not in d
    assert sorted(mm._state["keep"][78]) == [0, 1] and len(mm._state["keep"][78][0]) == 3


def test_convert_keeps_only_local_references_and_refuses_an_unpaired_half():
    _convert(_stream(), local=lambda e: e == 1)
    assert sorted(mm._state["keep"][78]) == [1]
    bad = [("model.layers.78.mlp.experts.0.gate_proj.weight", torch.ones(4, 32, dtype=torch.bfloat16))]
    with pytest.raises(RuntimeError, match="never paired"):
        _convert(iter(bad))


def _vllm_replica():
    """vLLM nightly 0.31.1rc1.dev273 (the g10 pool) on the draft's construction path, reduced to
    what decides the MTP experts' quant method (paths under vllm/):
      quantization: overrides probed modelopt_fp4 before modelopt_mixed (config/model.py:1384);
        ModelOptNvFp4Config claims any quant_algo with NVFP4/FP4 (layers/quantization/
        modelopt.py:768) -> our served copy runs quantization=modelopt_fp4 (run-g10 log)
      ModelOptQuantConfigBase.get_quant_method, inherited by ModelOptNvFp4Config: excluded ->
        None, RoutedExperts -> FusedMoEMethodCls(self) (modelopt.py:183-254)
      ModelOptMixedPrecisionConfig.get_quant_method: excluded or not in quantized_layers ->
        None (modelopt.py:1796-1849)
      RoutedExperts.__init__ -> _get_quant_method -> resolve_quant_method -> get_quant_method
        (layer, prefix); None -> UnquantizedFusedMoEMethod (fused_moe/routed_experts.py:159,
        221-235; quantization/base_config.py:292)
      prefixes: DeepseekV32MTP "model" -> MultiTokenPredictor f"{p}.layers.{idx}", idx from
        num_hidden_layers -> mtp_block (same prefix) -> DeepseekV2MoE f"{p}.mlp" ->
        FusedMoEFactory f"{p}.experts" -> MoERunner(.routed_experts) (models/deepseek_v32/
        nvidia/mtp.py:64-100,171-185,274-280; nvidia/model.py:58; deepseek_v2.py:374;
        fused_moe/layer.py:227,388; runner/moe_runner.py:272), all inside set_current_vllm_config
        (target config) (model_loader/utils.py:90-95)."""
    from fnmatch import fnmatch

    class UnquantizedFusedMoEMethod:
        pass

    class ModelOptNvFp4FusedMoE:
        nvfp4_backend = types.SimpleNamespace(name="FLASHINFER_CUTLASS")

        def __init__(self, quant_config, moe_config):
            self.quant_config, self.moe_config = quant_config, moe_config

        def process_weights_after_loading(self, layer):
            pass

    class RoutedExperts:
        def __init__(self, prefix, quant_config):
            self.layer_name, self.moe_config = prefix, "moe-config"
            qm = quant_config.get_quant_method(self, prefix)
            self.quant_method = UnquantizedFusedMoEMethod() if qm is None else qm

        def _map_global_expert_id_to_local_expert_id(self, e):
            return e

    class ModelOptQuantConfigBase:
        FusedMoEMethodCls = None

        def __init__(self, exclude_modules):
            self.exclude_modules = exclude_modules

        def is_layer_excluded(self, prefix):
            return any(fnmatch(prefix, p) for p in self.exclude_modules)

        def get_quant_method(self, layer, prefix):
            if self.is_layer_excluded(prefix):
                return None
            if isinstance(layer, RoutedExperts):
                return self.FusedMoEMethodCls(quant_config=self, moe_config=layer.moe_config)
            return None

    class ModelOptNvFp4Config(ModelOptQuantConfigBase):
        FusedMoEMethodCls = ModelOptNvFp4FusedMoE

    class ModelOptFp8MoEMethod(ModelOptNvFp4FusedMoE):
        pass

    class ModelOptFp8Config(ModelOptQuantConfigBase):
        FusedMoEMethodCls = ModelOptFp8MoEMethod

    class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
        def __init__(self, exclude_modules, quantized_layers):
            super().__init__(exclude_modules)
            self.quantized_layers = quantized_layers
            self.nvfp4_config = ModelOptNvFp4Config([])

        def get_quant_method(self, layer, prefix):
            if self.is_layer_excluded(prefix):
                return None
            if isinstance(layer, RoutedExperts) and self.quantized_layers.get(prefix) == "NVFP4":
                return ModelOptNvFp4FusedMoE(quant_config=self.nvfp4_config, moe_config=layer.moe_config)
            return None

    def quant_config(algo, ignore):
        if "NVFP4" in algo or "FP4" in algo:
            return ModelOptNvFp4Config(ignore)
        assert algo == "MIXED_PRECISION"
        return ModelOptMixedPrecisionConfig(ignore, {f"model.layers.{i}.mlp.experts": "NVFP4"
                                                     for i in range(3, 78)})

    class MoERunner:
        def __init__(self, prefix, qc):
            self.routed_experts = RoutedExperts(prefix, qc)

    class DeepseekV32MTP:
        def __init__(self, vllm_config, prefix=""):
            self.config = cfg = vllm_config.model_config.hf_config
            pre = f"{prefix}.model" if prefix else "model"
            first = cfg.num_hidden_layers
            block = lambda p: types.SimpleNamespace(  # noqa: E731
                mlp=types.SimpleNamespace(experts=MoERunner(f"{p}.mlp.experts", vllm_config.quant_config)))
            self.model = types.SimpleNamespace(layers={
                str(i): types.SimpleNamespace(mtp_block=block(f"{pre}.layers.{i}"))
                for i in range(first, first + cfg.num_nextn_predict_layers)})

        def load_weights(self, weights):
            return {n for n, _ in weights}

    modelopt = types.SimpleNamespace(**{c.__name__: c for c in (
        ModelOptQuantConfigBase, ModelOptFp8Config, ModelOptNvFp4Config, ModelOptMixedPrecisionConfig,
        RoutedExperts, ModelOptNvFp4FusedMoE)})
    return types.SimpleNamespace(modelopt=modelopt, quant_config=quant_config, RoutedExperts=RoutedExperts,
                                 Unquantized=UnquantizedFusedMoEMethod,
                                 mtp=types.SimpleNamespace(DeepseekV32MTP=DeepseekV32MTP))


def _target(qc):
    hf = types.SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1, n_routed_experts=4)
    return types.SimpleNamespace(quant_config=qc, model_config=types.SimpleNamespace(
        model="/models/glm-5.3-nvfp4", hf_config=hf))


def _construct(monkeypatch, v, qc):
    """Target's last MoE layer + the draft, each under the current (target) vLLM config."""
    current = {}
    monkeypatch.setattr(mm, "_model_config", lambda: current["vllm"].model_config)
    current["vllm"] = _target(qc)
    main = v.RoutedExperts("model.layers.77.mlp.experts", qc)
    draft = v.mtp.DeepseekV32MTP(current["vllm"])
    current.clear()  # load_weights runs outside the context, like vLLM's loader
    return main, draft


@pytest.mark.parametrize("algo, ignore", [
    ("NVFP4", ["model.layers.78*"]),            # /models/glm-5.3-nvfp4 as served: modelopt_fp4
    ("MIXED_PRECISION", ["model.layers.78*"]),  # HF's current export: modelopt_mixed
    ("MIXED_PRECISION", []),                    # MTP layer just not in quantized_layers
])
def test_real_draft_construction_path_switches_the_mtp_experts(monkeypatch, capsys, algo, ignore):
    assert mm.is_mtp_experts("model.layers.78.mlp.experts", 78)
    assert not mm.is_mtp_experts("model.layers.77.mlp.experts", 78)
    assert not mm.is_mtp_experts("model.layers.78.mlp.gate", 78)
    v = _vllm_replica()
    mm.patch_quant(v.modelopt)
    mm.patch_mtp(v.mtp)
    qc = v.quant_config(algo, ignore)
    nv = getattr(qc, "nvfp4_config", qc)
    _reset()
    main, draft = _construct(monkeypatch, v, qc)
    assert type(main.quant_method) is v.modelopt.ModelOptNvFp4FusedMoE and main.quant_method.quant_config is nv
    ex = draft.model.layers["78"].mtp_block.mlp.experts.routed_experts
    assert ex.layer_name == "model.layers.78.mlp.experts"
    assert type(ex.quant_method) is v.modelopt.ModelOptNvFp4FusedMoE and ex.quant_method.quant_config is nv
    assert mm._state["switched"] == {78: "/models/glm-5.3-nvfp4"}
    assert (f"model.layers.78.mlp.experts: {type(qc).__name__} leaves it BF16 (excluded={bool(ignore)}) "
            "-> ModelOptNvFp4FusedMoE") in capsys.readouterr().err
    monkeypatch.setattr(mm, "input_scales", lambda path, layer: (0.5, 0.25))
    real = mm.convert
    monkeypatch.setattr(mm, "convert", lambda *a: real(*a, quant=_quant, check=lambda *x: None,
                                                        device=torch.device("cpu")))
    names = draft.load_weights(_stream())
    assert mm._state["tensors"] == 6 and "model.layers.78.mlp.experts.0.gate_proj.weight_scale_2" in names
    assert "model.layers.78.mlp.experts.3.down_proj.input_scale" in names


def test_gate_on_refuses_to_start_when_nothing_is_switched_or_converted(monkeypatch, capsys):
    v = _vllm_replica()
    mm.patch_quant(v.modelopt)
    mm.patch_mtp(v.mtp)
    _reset()
    # a ModelOpt config without an NVFP4 config (FP8): refused while the draft is built
    with pytest.raises(RuntimeError, match="no NVFP4 config"):
        _construct(monkeypatch, v, v.modelopt.ModelOptFp8Config(["model.layers.78*"]))
    # a draft whose quant config is not ModelOpt: nothing switched, refused at load
    _reset()
    _, draft = _construct(monkeypatch, v, types.SimpleNamespace(get_quant_method=lambda layer, prefix: None))
    assert type(draft.model.layers["78"].mtp_block.mlp.experts.routed_experts.quant_method) is v.Unquantized
    with pytest.raises(RuntimeError, match="no MTP routed-expert layer was switched"):
        draft.load_weights(_stream())
    assert "FAIL: SUFFIX_MTP_MOE_NVFP4=1 but no MTP routed-expert layer" in capsys.readouterr().err
    # switched, but the stream has no BF16 MTP expert weight
    mm._state["switched"][78] = "/models/glm-5.3-nvfp4"
    monkeypatch.setattr(mm, "input_scales", lambda path, layer: (0.5, 0.25))
    with pytest.raises(RuntimeError, match="no BF16 MTP expert weight to convert"):
        draft.load_weights(iter([("model.layers.78.eh_proj.weight", torch.ones(2, 2))]))


def test_input_scales_take_the_max_of_the_last_quantized_layer(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    t = {f"model.layers.77.mlp.experts.{e}.{p}.input_scale": torch.tensor(float(v))
         for e, vs in ((0, (1, 2, 3)), (1, (4, 1, 5))) for p, v in zip(("gate_proj", "up_proj",
                                                                         "down_proj"), vs)}
    st.save_file(t, str(tmp_path / "a.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {%s}}' % ", ".join(f'"{n}": "a.safetensors"' for n in t))
    assert mm.input_scales(str(tmp_path), 77) == (4.0, 5.0)


def _layer(tp=2, rank=1, n=4, hdim=64, inter=32):
    g = torch.Generator().manual_seed(1)
    full = {e: {"gate_proj": torch.randn(inter, hdim, generator=g).to(torch.bfloat16),
                "up_proj": torch.randn(inter, hdim, generator=g).to(torch.bfloat16),
                "down_proj": torch.randn(hdim, inter, generator=g).to(torch.bfloat16)} for e in range(n)}
    pc = types.SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer = types.SimpleNamespace(moe_config=types.SimpleNamespace(moe_parallel_config=pc),
                                  w13_weight=torch.zeros(1), global_num_experts=n, top_k=3,
                                  _map_global_expert_id_to_local_expert_id=lambda e: e)
    return layer, full


def _shard_forward(full, swap=False):
    def forward(layer, x, tw, ids):
        pc = layer.moe_config.moe_parallel_config
        out = torch.zeros(x.shape, dtype=torch.float32)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                w1, w3, w2 = (full[e][p].float() for p in ("gate_proj", "up_proj", "down_proj"))
                ip = w1.shape[0] // pc.tp_size
                sl = slice(pc.tp_rank * ip, (pc.tp_rank + 1) * ip)
                a, b = (w3, w1) if swap else (w1, w3)
                xf = x[t:t + 1].float()
                h = torch.nn.functional.silu(xf @ a[sl].t()) * (xf @ b[sl].t())
                out[t] += float(tw[t, j]) * (h @ w2[:, sl].t())[0]
        return out
    return forward


def test_oracle_matches_the_tp_shard_and_catches_a_gate_up_swap():
    layer, full = _layer()
    mm._state["keep"] = {78: {0: full[0], 1: full[1]}}
    assert mm.oracle(layer, 78, 0.01, 0.01, forward=_shard_forward(full)) < 1e-3
    mm._state["keep"] = {78: {0: full[0], 1: full[1]}}
    with pytest.raises(RuntimeError, match="LAYER ORACLE FAIL"):
        mm.oracle(layer, 78, 0.01, 0.01, forward=_shard_forward(full, swap=True))


def test_oracle_under_vllms_bf16_default_dtype():
    """g11b boot: vLLM loads under torch.set_default_dtype(bf16); the oracle's random
    activations must stay fp32 against the fp32 reference weights."""
    layer, full = _layer()
    mm._state["keep"] = {78: {0: full[0], 1: full[1]}}
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        assert mm.oracle(layer, 78, 0.01, 0.01, forward=_shard_forward(full)) < 1e-3
    finally:
        torch.set_default_dtype(prev)
