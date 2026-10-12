# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_GDN_DEFER wiring (no vLLM, AITER or GPU): forward_spec
and the generic-path hook hand gdn_defer the unshifted slot ids, the strided b / a views
of ba, the spec rows' seq_lens and the align block size. The kernel itself is checked on
silicon against AITER over multi-step acceptance: boot gate gdn_defer_bench."""
import importlib.util
import sys
import types
from types import SimpleNamespace

import torch

from suffix_hybrid import rocm_patches as rp

if importlib.util.find_spec("vllm") is None:  # CPU CI: the kernel module only needs @jit
    _jit = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)  # noqa: E731
    _tu = types.ModuleType("vllm.triton_utils")
    _tu.tl, _tu.triton = SimpleNamespace(), SimpleNamespace(jit=_jit)
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.triton_utils": _tu})
    try:
        from suffix_hybrid.kernels import gdn_defer_rocm
    finally:  # no fake vllm for the rest of the session
        del sys.modules["vllm"], sys.modules["vllm.triton_utils"]
from suffix_hybrid.kernels import gdn_defer_rocm, gdn_mtp_rocm  # noqa: E402


def test_forward_spec_and_generic_hook_call_gdn_defer(monkeypatch):
    calls = []
    monkeypatch.setattr(gdn_defer_rocm, "gdn_defer", lambda *a: calls.append(a))
    monkeypatch.setattr(gdn_mtp_rocm, "_DEFER", True)
    monkeypatch.setattr(gdn_mtp_rocm, "_Q", SimpleNamespace(
        is_conv_state_dim_first=lambda: True,
        causal_conv1d_update=lambda x, *a, **kw: x))
    hv, qkv_dim = 2, 14
    layer = SimpleNamespace(
        gqa_interleaved_layout=False, key_dim=4, value_dim=6, tp_size=1, head_k_dim=2,
        head_v_dim=3, num_k_heads=2, num_v_heads=hv, activation="silu",
        A_log=torch.zeros(hv), dt_bias=torch.zeros(hv),
        conv1d=SimpleNamespace(weight=torch.randn(qkv_dim, 1, 4), bias=None),
        kv_cache=(torch.randn(9, qkv_dim, 4), torch.randn(9, hv, 3, 2)))
    idx = torch.tensor([[3, 1, 7], [5, 2, 4]], dtype=torch.int32)
    md = SimpleNamespace(spec_sequence_masks=torch.ones(2, dtype=torch.bool), num_prefills=0,
                         num_decodes=0, num_spec_decodes=2, num_actual_tokens=6,
                         spec_state_indices_tensor=idx,
                         num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32),
                         spec_query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
                         suffix_spec_seq_lens=torch.tensor([40, 9], dtype=torch.int32),
                         suffix_zone=1664)
    qkvz, ba, out = torch.randn(8, qkv_dim + 6), torch.randn(8, 2 * hv), torch.ones(8, hv, 3)
    assert gdn_mtp_rocm.forward_spec(layer, qkvz, ba, out, md) is True
    (qkv, a, b, A_log, dt, state, cu, sidx, acc, seq, o, H, K, V, zone), = calls
    assert qkv.shape == (6, qkv_dim) and qkv.data_ptr() == qkvz.data_ptr()
    assert b.data_ptr() == ba.data_ptr() and torch.equal(b, ba[:6, :hv])  # views, no copy
    assert torch.equal(a, ba[:6, hv:]) and a.stride(0) == 2 * hv
    assert state is layer.kv_cache[1] and sidx is idx  # unshifted: NULL = 0 skipped in-kernel
    assert acc is md.num_accepted_tokens and seq is md.suffix_spec_seq_lens and o is out
    assert torch.equal(cu, md.spec_query_start_loc) and (H, K, V, zone) == (2, 2, 3, 1664)
    assert not out[6:].any() and bool((out[:6] == 1).all())

    calls.clear()
    mq, sa, sb = torch.randn(6, qkv_dim), torch.randn(6, hv), torch.randn(6, hv)
    res = gdn_defer_rocm.defer_spec(layer, mq, sa, sb, layer.kv_cache[1], md)
    assert res.shape == (1, 6, hv, 3)
    (qkv, a, b, *_, o, H, K, V, zone), = calls
    assert qkv is mq and a is sa and b is sb and o.shape == (6, hv, 3) and zone == 1664


def test_generic_path_rewrite():
    meta, ctor, core, _ = rp.PATCHES["SUFFIX_ROCM_GDN_DEFER"]
    src = rp.patch_source(core, "def f(self, spec_sequence_masks):\n" + core.old)
    assert "_suffix_gdn_defer_spec(" in src and "fused_sigmoid" not in src
    compile(src, "<t>", "exec")
    assert core.after == "suffix_hybrid.kernels.gdn_defer_rocm:install"
    assert "suffix_spec_seq_lens" in meta.new and "spec_req_idx" in ctor.new


def test_forward_mixed_slices_verify_prefix_and_prefill_suffix(monkeypatch):
    """SUFFIX_ROCM_GDN_MIXED: verify rows [0, nst) through conv update + gdn_defer, prefill
    rows [nst, n) through vLLM's conv_fn / post-conv / chunk calls on slices, outputs
    placed by row range, stale tail zeroed; a non-prefix layout falls back to stock."""
    hv, K, V, qkv_dim, H = 2, 2, 3, 14, 2
    calls = {}

    def defer(qkv, a, b, A_log, dt, state, cu, idx, acc, seq, out, *rest):
        calls["defer"] = (qkv, a, b, cu, seq, rest)
        out[: qkv.shape[0]] = 5.0

    def chunk(**kw):  # like FLA: the output goes straight into core_attn_out
        calls["chunk"] = kw
        kw["core_attn_out"].fill_(7.0)
        return kw["core_attn_out"].view(1, -1, hv, V), torch.full(kw["initial_state"].shape, 3.0)

    monkeypatch.setattr(gdn_defer_rocm, "gdn_defer", defer)
    monkeypatch.setattr(gdn_mtp_rocm, "_MIXED", True)
    monkeypatch.setattr(gdn_mtp_rocm, "_PREFILL_MAX", 256)

    def prefill(x, a, b, A_log, dt, state, cu, slots, has, out, *rest):
        calls["prefill"] = (x, a, b, cu, slots, has, rest)
        out[:] = 9.0

    monkeypatch.setattr(gdn_defer_rocm, "gdn_prefill", prefill)
    monkeypatch.setattr(gdn_mtp_rocm, "_Q", SimpleNamespace(
        is_conv_state_dim_first=lambda: True,
        causal_conv1d_update=lambda x, *a, **kw: calls.setdefault("upd", (x, kw)) and x,
        causal_conv1d_fn=lambda x, *a, **kw: calls.setdefault("fn", (x, kw)) and x,
        fused_post_conv_prep=lambda **kw: calls.setdefault("post", kw) and (
            torch.zeros(kw["conv_output"].shape[0], H, K), torch.zeros(kw["conv_output"].shape[0], H, K),
            torch.zeros(kw["conv_output"].shape[0], hv, V), torch.zeros(kw["conv_output"].shape[0], hv),
            torch.zeros(kw["conv_output"].shape[0], hv))))
    layer = SimpleNamespace(
        gqa_interleaved_layout=False, key_dim=4, value_dim=6, tp_size=1, head_k_dim=K,
        head_v_dim=V, num_k_heads=H, num_v_heads=hv, activation="silu",
        A_log=torch.zeros(hv), dt_bias=torch.zeros(hv), chunk_gated_delta_rule=chunk,
        conv1d=SimpleNamespace(weight=torch.randn(qkv_dim, 1, 4), bias=None),
        kv_cache=(torch.randn(12, qkv_dim, 4), torch.randn(12, hv, V, K)))
    pre_idx = torch.tensor([8, 9])
    md = SimpleNamespace(
        spec_sequence_masks=torch.tensor([True, True, False, False]),
        spec_sequence_masks_cpu=torch.tensor([True, True, False, False]),
        num_prefills=2, num_decodes=0, num_spec_decodes=2, num_spec_decode_tokens=10,
        num_prefill_tokens=7, num_actual_tokens=17,
        spec_state_indices_tensor=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 10, 11, 0]], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 5, 10], dtype=torch.int32),
        suffix_spec_seq_lens=torch.tensor([40, 9], dtype=torch.int32), suffix_zone=1664,
        has_initial_state=torch.tensor([True, False]),
        non_spec_state_indices_tensor=pre_idx, non_spec_query_start_loc=torch.tensor([0, 3, 7]),
        prefill_state_indices=pre_idx, prefill_has_initial_state=torch.tensor([True, False]),
        prefill_query_start_loc=torch.tensor([0, 3, 7]), chunk_indices="ci", chunk_offsets="co",
        aiter_prefill_metadata=None, suffix_max_prefill=4096)
    qkvz, ba, out = torch.randn(20, qkv_dim + 6), torch.randn(20, 2 * hv), torch.ones(20, hv, V)
    before = layer.kv_cache[1][8].clone()
    assert gdn_mtp_rocm.forward_spec(layer, qkvz, ba, out, md) is True
    upd_x, upd_kw = calls["upd"]
    assert upd_x.shape == (10, qkv_dim) and upd_x.data_ptr() == qkvz.data_ptr()
    assert torch.equal(upd_kw["conv_state_indices"], torch.tensor([1, 6], dtype=torch.int32))
    qkv, a, b, cu, seq, rest = calls["defer"]
    assert torch.equal(a, ba[:10, hv:]) and torch.equal(b, ba[:10, :hv]) and torch.equal(cu, md.spec_query_start_loc)
    fn_x, fn_kw = calls["fn"]
    assert fn_x.shape == (qkv_dim, 7) and torch.equal(fn_x.t(), qkvz[10:17, :qkv_dim])
    assert fn_kw["cache_indices"] is pre_idx and fn_kw["metadata"] is md
    assert torch.equal(calls["post"]["a"], ba[10:17, hv:]) and torch.equal(calls["post"]["b"], ba[10:17, :hv])
    init = calls["chunk"]["initial_state"]
    assert torch.equal(init[0], before) and not init[1].any()  # no prior state -> zeros
    assert calls["chunk"]["cu_seqlens"] is md.prefill_query_start_loc
    co = calls["chunk"]["core_attn_out"]  # rows [nst, n) of the layer output, flat
    assert co.data_ptr() == out[10].data_ptr() and co.numel() == 7 * hv * V
    assert bool((layer.kv_cache[1][pre_idx] == 3.0).all())
    assert bool((out[:10] == 5.0).all()) and bool((out[10:17] == 7.0).all()) and not out[17:].any()

    # Short prefill chunks: one recurrent launch over the conv output, rows [nst, n).
    md.suffix_max_prefill, out[:] = 4, 1.0
    assert gdn_mtp_rocm.forward_spec(layer, qkvz, ba, out, md) is True
    x, a, b, cu, slots, has, rest = calls["prefill"]
    assert x.shape == (7, qkv_dim) and torch.equal(a, ba[10:17, hv:]) and torch.equal(b, ba[10:17, :hv])
    assert cu is md.prefill_query_start_loc and slots is pre_idx and has is md.prefill_has_initial_state
    assert bool((out[:10] == 5.0).all()) and bool((out[10:17] == 9.0).all()) and not out[17:].any()

    md.spec_sequence_masks_cpu = torch.tensor([True, False, True, False])  # not a verify prefix
    assert gdn_mtp_rocm.forward_spec(layer, qkvz, ba, out, md) is False
