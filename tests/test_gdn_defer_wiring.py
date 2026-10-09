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
    meta, ctor, core = rp.PATCHES["SUFFIX_ROCM_GDN_DEFER"]
    src = rp.patch_source(core, "def f(self, spec_sequence_masks):\n" + core.old)
    assert "_suffix_gdn_defer_spec(" in src and "fused_sigmoid" not in src
    compile(src, "<t>", "exec")
    assert core.after == "suffix_hybrid.kernels.gdn_defer_rocm:install"
    assert "suffix_spec_seq_lens" in meta.new and "spec_req_idx" in ctor.new
