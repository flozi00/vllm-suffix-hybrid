# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 layer wiring (qwen_gdn._make_layer_cls) on CPU against stub vLLM
modules: per-layer contract by backend, the routing flag, and which path each
batch shape takes — non-spec decode, MTP verify, stock — for the flat and the
interleaved layout at TP>1. Kernels are recorded, not run (the bodies have
their own interpret tests; the oracles are GPU-only)."""
import dataclasses
import sys
import types

import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import kgdn1
from suffix_hybrid.kernels import qwen_gdn

CALLS = []


@dataclasses.dataclass
class GDNAttentionMetadata:
    num_prefills: int = 0
    num_decodes: int = 0
    num_actual_tokens: int = 0
    num_spec_decodes: int = 0
    spec_sequence_masks: object = None
    non_spec_state_indices_tensor: object = None
    spec_state_indices_tensor: object = None
    spec_query_start_loc: object = None
    num_accepted_tokens: object = None


class _Norm:
    activation, norm_before_gate, group_size, bias = "silu", True, None, None

    def __init__(self, v):
        self.weight = torch.ones(v)


class QwenGatedDeltaNetAttention:
    """The slice of the vLLM 0.32.0 layer the lane touches."""

    def __init__(self, tp=1, H=16, HV=48, K=128, V=128, interleaved=False, num_spec=0):
        self.tp_size, self.num_k_heads, self.num_v_heads = tp, H, HV
        self.head_k_dim, self.head_v_dim = K, V
        self.key_dim, self.value_dim = H * K, HV * V
        self.gqa_interleaved_layout, self.num_spec = interleaved, num_spec
        self.prefix, self.layer_norm_epsilon, self.activation = "m.layers.0.linear_attn", 1e-6, "silu"
        self.norm = _Norm(V)
        self.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
        self.A_log, self.dt_bias = torch.zeros(HV // tp), torch.zeros(HV // tp)
        ch = (2 * H * K + HV * V) // tp
        self.conv1d = types.SimpleNamespace(weight=torch.zeros(ch, 1, 4), bias=None)
        self.kv_cache = (torch.zeros(9, ch, 3), torch.zeros(9, HV // tp, V, K))
        self.enable_fused_gdn_spec_decode = False

    def get_state_dtype(self):
        return torch.bfloat16, torch.float32

    def forward_cuda(self, hidden_states):
        if self.enable_fused_gdn_spec_decode:
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(hidden_states)

    def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
        CALLS.append(("stock_packed", mixed_qkvz.shape, ba.shape))

    def _forward_core_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out):
        CALLS.append(("stock_norm", mixed_qkv, b, a, output_gate))


FWD = {"md": None}


def _conv(x, conv_state, w, bias, act, conv_state_indices=None, **kw):
    CALLS.append(("conv", x, conv_state_indices, kw))
    return x


class _Native:
    def gdn_decode_fused_cuda(self, *args):
        CALLS.append(("oxide", args))


@pytest.fixture
def stub_vllm(monkeypatch, tmp_path):
    mods = {
        "vllm": {}, "vllm.forward_context": {"get_forward_context":
                                             lambda: types.SimpleNamespace(attn_metadata=FWD["md"])},
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn":
            {"QwenGatedDeltaNetAttention": QwenGatedDeltaNetAttention},
        "vllm.model_executor.layers.mamba.mamba_utils": {"is_conv_state_dim_first": lambda: True},
        "vllm.model_executor.layers.mamba.ops.causal_conv1d": {"causal_conv1d_update": _conv},
        "vllm.v1.attention.backends.gdn_attn": {"GDNAttentionMetadata": GDNAttentionMetadata},
    }
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: types.SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(qwen_gdn, "load_prebuilt", lambda *a: CALLS.append(("sass", a)) or "sass")
    monkeypatch.setattr(qwen_gdn, "stock_decode_fused", lambda *a: CALLS.append(("stock_dec", a)))
    monkeypatch.setattr(qwen_gdn, "_state", dict(qwen_gdn._state))
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path))  # no decode configs shipped
    monkeypatch.setattr(hk, "_TABLES", {})
    monkeypatch.setattr(hk, "platform", lambda device=0: "testgpu")
    CALLS.clear()
    FWD["md"] = None

    def build(mode, verify_buckets=(), **kw):
        monkeypatch.setenv(qwen_gdn.HELION_ENV, mode)
        monkeypatch.setattr(kgdn1, "verify_op", lambda h, hv, k, v, w: (
            (lambda *a: CALLS.append(("verify", a))) if verify_buckets and w else None,
            tuple(verify_buckets) if w else ()))
        return qwen_gdn._make_layer_cls(_Native())(**kw)
    return build


def _md(**kw):
    return {"m.layers.0.linear_attn": GDNAttentionMetadata(**kw)}


def _run(layer, T):
    H, HV, K, V = layer._kgdn_dims
    qkvz = torch.randn(T, 2 * H * K + 2 * HV * V).to(torch.bfloat16)
    ba = torch.randn(T, 2 * HV).to(torch.bfloat16)
    out = torch.zeros(T, HV, V, dtype=torch.bfloat16)
    CALLS.clear()
    layer._forward_core_fused_norm_packed(qkvz, ba, out)
    return qkvz, ba, out


def test_oxide_mode_keeps_tp1_contract(stub_vllm):
    with pytest.raises(ValueError, match="tp_size=2"):
        stub_vllm("oxide", tp=2)
    with pytest.raises(ValueError, match="interleaved"):
        stub_vllm("oxide", interleaved=True)
    lay = stub_vllm("oxide", num_spec=4)
    assert ("sass", (16, 48, 128)) in CALLS and lay.enable_fused_gdn_spec_decode
    FWD["md"] = _md(num_decodes=3, num_actual_tokens=4,
                    non_spec_state_indices_tensor=torch.tensor([1, 2, 3, 0]))
    _run(lay, 4)
    conv, oxide = CALLS
    assert conv[0] == "conv" and oxide[0] == "oxide" and oxide[1][-1] == 7  # stream ptr
    assert oxide[1][9] == 16 and oxide[1][1].shape == (4, 48, 128)
    FWD["md"] = _md(num_spec_decodes=2, num_actual_tokens=10, spec_sequence_masks=torch.ones(2))
    _run(lay, 10)
    assert [c[0] for c in CALLS] == ["stock_packed"]   # verify stays vLLM's under oxide


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_auto_tp_flat_decode_and_verify(stub_vllm, tp):
    lay = stub_vllm("auto", verify_buckets=(8,), tp=tp, H=16, HV=32, num_spec=4)
    H, HV = 16 // tp, 32 // tp
    assert lay._kgdn_dims == (H, HV, 128, 128) and not any(c[0] == "sass" for c in CALLS)
    FWD["md"] = _md(num_decodes=2, num_actual_tokens=2,
                    non_spec_state_indices_tensor=torch.tensor([4, 5, 0]))
    qkvz, ba, _ = _run(lay, 2)
    conv, dec = CALLS
    assert dec[0] == "stock_dec"                      # no config shipped: vLLM's FLA chain
    assert conv[1].shape == (2, 2 * H * 128 + HV * 128) and conv[1].data_ptr() == qkvz.data_ptr()
    assert dec[1][1].shape == (2, HV, 128) and dec[1][2].data_ptr() == ba.data_ptr()
    assert dec[1][9] == H
    idx = torch.arange(1, 16).view(3, 5).int()
    md = dict(num_actual_tokens=12, spec_sequence_masks=torch.ones(3),
              spec_state_indices_tensor=idx, spec_query_start_loc=torch.tensor([0, 5, 9, 12, 12]),
              num_accepted_tokens=torch.tensor([1, 3, 2, 1]))
    FWD["md"] = _md(num_spec_decodes=3, **md)
    _run(lay, 12)
    conv, ver = CALLS
    assert ver[0] == "verify" and conv[3]["max_query_len"] == 5
    assert torch.equal(conv[2], idx[:, 0]) and conv[3]["query_start_loc"].tolist() == [0, 5, 9, 12]
    assert torch.equal(ver[1][7], idx) and ver[1][9].tolist() == [1, 3, 2] and ver[1][11] == H
    FWD["md"] = _md(num_spec_decodes=40, **md)       # bucket 128: not shipped -> vLLM stock
    _run(lay, 12)
    assert [c[0] for c in CALLS] == ["stock_packed"]
    FWD["md"] = _md(num_spec_decodes=3, num_prefills=1, **md)
    _run(lay, 12)
    assert [c[0] for c in CALLS] == ["stock_packed"]


def test_auto_interleaved(stub_vllm):
    lay = stub_vllm("auto", verify_buckets=(1, 8), tp=2, H=16, HV=32, interleaved=True,
                    num_spec=2)
    H, HV, G = 8, 16, 2
    FWD["md"] = _md(num_decodes=3, num_actual_tokens=3,
                    non_spec_state_indices_tensor=torch.tensor([1, 2, 3]))
    qkvz, ba, _ = _run(lay, 3)
    conv, dec = CALLS
    want_qkv, want_z, want_ba = qwen_gdn.deinterleave(qkvz, ba, H, G, 128, 128)
    assert torch.equal(conv[1], want_qkv) and dec[0] == "stock_dec"
    assert dec[1][1].shape == (3, H, G, 128) and torch.equal(dec[1][1], want_z)
    assert torch.equal(dec[1][2], want_ba)
    FWD["md"] = _md(num_prefills=1, num_actual_tokens=3)
    qkvz, ba, _ = _run(lay, 3)
    want_qkv, want_z, want_ba = qwen_gdn.deinterleave(qkvz, ba, H, G, 128, 128)
    (stock,) = CALLS
    assert stock[0] == "stock_norm" and torch.equal(stock[1], want_qkv)
    assert torch.equal(stock[2], want_ba[:, :HV]) and torch.equal(stock[3], want_ba[:, HV:])
    assert stock[4].shape == (3, HV, 128) and torch.equal(stock[4], want_z.reshape(3, HV, 128))
    FWD["md"] = _md(num_spec_decodes=1, num_actual_tokens=3, spec_sequence_masks=torch.ones(1),
                    spec_state_indices_tensor=torch.tensor([[1, 2]]).int(),
                    spec_query_start_loc=torch.tensor([0, 2]),
                    num_accepted_tokens=torch.tensor([2]))
    _run(lay, 3)
    assert [c[0] for c in CALLS] == ["conv", "verify"]


def test_profile_run_runs_oracles_then_stock(stub_vllm, monkeypatch):
    seen = []
    monkeypatch.setattr(qwen_gdn, "run_gpu_oracle", lambda *a, **k: seen.append(("dec", k)) or
                        {"out": 0.0, "state": 0.0})
    monkeypatch.setattr(qwen_gdn, "run_gpu_verify_oracle", lambda *a, **k: seen.append(
        ("ver", a[5], a[8])) or {"out": 0.0, "state": 0.0})
    lay = stub_vllm("auto", verify_buckets=(8,), tp=2, interleaved=True, num_spec=4)
    _run(lay, 2)
    assert seen[0][0] == "dec" and seen[0][1]["interleaved"] and seen[0][1]["row_stride"] is None
    assert seen[1] == ("ver", 5, (8,))
    assert [c[0] for c in CALLS] == ["stock_norm"]
    _run(lay, 2)
    assert len(seen) == 2                              # once per process
