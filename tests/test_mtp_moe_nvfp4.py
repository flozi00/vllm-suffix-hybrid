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


def _convert(stream, n_experts=4, local=lambda e: True):
    seen = {}
    calls = []

    def quant(t, gscale):
        seen.setdefault(id(t), float(gscale))
        return torch.zeros(t.shape[0], t.shape[1] // 2, dtype=torch.uint8), \
            torch.zeros(t.shape[0], t.shape[1] // 16, dtype=torch.float8_e4m3fn)

    def scales():
        calls.append(1)
        return 0.5, 0.25

    mm._state.update(keep={}, scales=None, tensors=0, saved=0)
    out = list(mm.convert(stream, 78, n_experts, scales, local, quant=quant,
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


def test_mtp_prefix_and_quant_method_only_for_excluded_mtp_experts(monkeypatch):
    assert mm.is_mtp_experts("model.layers.78.mlp.experts", 78)
    assert not mm.is_mtp_experts("model.layers.77.mlp.experts", 78)
    assert not mm.is_mtp_experts("model.layers.78.mlp.gate", 78)

    class RoutedExperts:
        moe_config = "cfg"

    class Nvfp4:
        nvfp4_backend = types.SimpleNamespace(name="FLASHINFER_CUTLASS")

        def __init__(self, quant_config, moe_config):
            self.args = (quant_config, moe_config)

        def process_weights_after_loading(self, layer):
            pass

    class Mixed:
        nvfp4_config = "nvfp4-sub"

        def is_layer_excluded(self, prefix):
            return prefix.startswith("model.layers.78")

        def get_quant_method(self, layer, prefix):
            return "stock"

    module = types.SimpleNamespace(ModelOptMixedPrecisionConfig=Mixed, RoutedExperts=RoutedExperts,
                                   ModelOptNvFp4FusedMoE=Nvfp4)
    monkeypatch.setattr(mm, "_num_hidden_layers", lambda: 78)
    monkeypatch.setattr(mm, "_model_config", lambda: types.SimpleNamespace(model="/models/x"))
    mm._state["switched"].clear()
    mm.patch_quant(module)
    qm = Mixed().get_quant_method(RoutedExperts(), "model.layers.78.mlp.experts")
    assert isinstance(qm, Nvfp4) and qm.args == ("nvfp4-sub", "cfg")
    assert mm._state["switched"] == {78: "/models/x"}
    assert Mixed().get_quant_method(RoutedExperts(), "model.layers.77.mlp.experts") == "stock"
    assert Mixed().get_quant_method(object(), "model.layers.78.mlp.experts") == "stock"


def test_input_scales_take_the_max_of_the_last_quantized_layer(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    t = {f"model.layers.77.mlp.experts.{e}.{p}.input_scale": torch.tensor(float(v))
         for e, vs in ((0, (1, 2, 3)), (1, (4, 1, 5))) for p, v in zip(("gate_proj", "up_proj",
                                                                         "down_proj"), vs)}
    st.save_file(t, str(tmp_path / "a.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {%s}}' % ", ".join(f'"{n}": "a.safetensors"' for n in t))
    assert mm.input_scales(str(tmp_path), 77) == (4.0, 5.0)


def test_load_weights_is_stock_when_no_mtp_layer_was_switched():
    got = []

    class MTP:
        def load_weights(self, weights):
            got.append(list(weights))
            return "loaded"

    mm._state["switched"].clear()
    mm.patch_mtp(types.SimpleNamespace(DeepseekV32MTP=MTP))
    stream = [("model.layers.78.mlp.experts.0.gate_proj.weight", torch.ones(4, 32, dtype=torch.bfloat16))]
    assert MTP().load_weights(iter(stream)) == "loaded" and got[0][0][0] == stream[0][0]


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
