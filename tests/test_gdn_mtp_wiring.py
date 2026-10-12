# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_GDN_MTP wiring (no vLLM, AITER or GPU).

A fake module carries both vLLM anchors verbatim. Stubs stand in for vLLM's conv
update and AITER's delta rule with each kernel's slot semantics (FLA skips state
slots <= 0, AITER < 0), so forward_spec's index shift is checked against the
stock contract. Silicon check: boot gate gdn_mtp_bench prints MATCH per case.
"""
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from suffix_hybrid import rocm_patches as rp
from suffix_hybrid.kernels import gdn_mtp_rocm

CORE, ZVIEW = rp.PATCHES["SUFFIX_ROCM_GDN_MTP"]
FAKE = (
    "class L:\n"
    "    def forward_hip(self, projected_states_qkvz, z, core_attn_out, hidden_states=None):\n"
    "        if True:\n" + ZVIEW.old +
    "    def _output_projection(self, core_attn_out, z):\n"
    "        return z\n"
    "    def forward_cuda(self, hidden_states):\n"
    "        return None\n"
    "    def _forward_core_rocm(self, qkvz, ba, z_out, core_attn_out, attn_metadata):\n"
    + CORE.old +
    "        return 'stock'\n")


def toy_rule(state, idx, nacc, cu, x, skip, keep):
    """Slot walk of the FLA/AITER kernels with a toy recurrence."""
    for i in range(len(cu) - 1):
        bos, eos = int(cu[i]), int(cu[i + 1])
        if eos == bos:
            continue
        s = int(idx[i, int(nacc[i]) - 1])
        if skip(s):
            continue
        h = state[s].clone()
        for t in range(eos - bos):
            h = 0.5 * h + x[bos + t]
            if keep(int(idx[i, t])):
                state[int(idx[i, t])] = h


def test_anchors_rewrite_and_fail_closed():
    out = FAKE
    for p in (CORE, ZVIEW):
        out = rp.patch_source(p, out)
        assert out.count(p.new) == 1
    compile(out, "<t>", "exec")
    for p in (CORE, ZVIEW):
        with pytest.raises(RuntimeError, match="drifted"):
            rp.patch_source(p, FAKE.replace(p.old, p.old.replace("        ", "    ", 1)))
        with pytest.raises(RuntimeError, match="drifted"):
            rp.patch_source(p, FAKE + FAKE)


def test_patched_module_dispatches_and_keeps_the_state_contract(tmp_path, monkeypatch):
    (tmp_path / "fake_gdn.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_GDN_MTP",
                        [p._replace(target="fake_gdn") for p in (CORE, ZVIEW)])
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv("SUFFIX_ROCM_GDN_MTP", "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rp.install_post_import_hook()
    try:
        mod = importlib.import_module("fake_gdn")
    finally:
        sys.modules.pop("fake_gdn", None)
    assert mod._suffix_gdn_mtp is gdn_mtp_rocm.forward_spec and gdn_mtp_rocm._Q is mod

    # Toy GDN: 2 qk heads x 2, 2 v heads x 3 -> qkv 14 + z 6 columns; 3 tokens per verify.
    hv, qkv_dim, win, pool = 2, 14, 3, 10
    layer = mod.L()
    layer.__dict__.update(
        gqa_interleaved_layout=False, key_dim=4, value_dim=6, tp_size=1, head_k_dim=2,
        head_v_dim=3, activation="silu", A_log=torch.zeros(hv), dt_bias=torch.zeros(hv),
        conv1d=SimpleNamespace(weight=torch.randn(qkv_dim, 1, 4), bias=None),
        kv_cache=(torch.randn(pool, qkv_dim, 4), torch.randn(pool, 1)))

    # forward_hip: the output gate reads z straight from qkvz (flat layout only).
    qkvz, z = torch.randn(14, qkv_dim + 6), torch.empty(14, hv, 3)
    gate = layer.forward_hip(qkvz, z, None)
    assert gate.data_ptr() == qkvz[:, qkv_dim:].data_ptr()
    assert torch.equal(gate.reshape(14, 6), qkvz[:, qkv_dim:])
    layer.gqa_interleaved_layout = True
    assert layer.forward_hip(qkvz, z, None) is z
    layer.gqa_interleaved_layout = False

    # seq0: NULL write slot at t=1; seq1: NULL read slot (skipped); seq2: acc 1;
    # seq3: graph padding (T=0, NULL slots). 12 padded tokens + a 2-row eager tail.
    idx = torch.tensor([[3, 0, 7], [5, 2, 0], [4, 6, 8], [0, 0, 0]], dtype=torch.int32)
    nacc = torch.tensor([3, 3, 1, 1], dtype=torch.int32)
    cu = torch.tensor([0, 3, 6, 9, 9], dtype=torch.int32)
    md = SimpleNamespace(spec_sequence_masks=torch.ones(4, dtype=torch.bool), num_prefills=0,
                         num_decodes=0, num_spec_decodes=4, num_actual_tokens=12,
                         spec_state_indices_tensor=idx, num_accepted_tokens=nacc,
                         spec_query_start_loc=cu)
    ba, out = torch.randn(14, 2 * hv), torch.ones(14, hv, 3)
    convs, rules = [], []
    ctx = SimpleNamespace()
    mod.is_conv_state_dim_first = lambda: True
    mod.get_forward_context = lambda: ctx
    mod.causal_conv1d_update = lambda x, *a, **kw: convs.append((x, a, kw)) or x
    mod.gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule = (
        lambda **kw: rules.append(kw) or toy_rule(
            kw["initial_state"], kw["ssm_state_indices"], kw["num_accepted_tokens"],
            kw["cu_seqlens"], kw["qkv"][:, :1], lambda s: s < 0, lambda d: d >= 0))
    ref = layer.kv_cache[1].clone()
    toy_rule(ref, idx, nacc, cu, qkvz[:12, :1], lambda s: s <= 0, lambda d: d > 0)

    assert layer._forward_core_rocm(qkvz, ba, z, out, md) is None  # handled, no fall-through
    assert torch.equal(layer.kv_cache[1], ref)  # same slots written and skipped as stock
    x, (conv_state, weight, *_), kw = convs[0]
    assert x.shape == (12, qkv_dim) and x.data_ptr() == qkvz.data_ptr()
    assert weight.shape == (qkv_dim, 4) and conv_state is layer.kv_cache[0]
    assert torch.equal(kw["conv_state_indices"], idx[:, 0]) and kw["max_query_len"] == win
    assert kw["num_accepted_tokens"] is nacc and kw["query_start_loc"] is cu
    rule = rules[0]
    assert rule["qkv"] is x and rule["core_attn_out"] is out and torch.equal(rule["cu_seqlens"], cu)
    assert rule["b"].is_contiguous() and torch.equal(rule["b"], ba[:12, :hv])
    assert rule["a"].is_contiguous() and torch.equal(rule["a"], ba[:12, hv:])
    assert not out[12:].any() and bool((out[:12] == 1).all())  # tail zeroed, rows < n kernel-owned

    # One index shift per forward context, shared by every layer.
    layer._forward_core_rocm(qkvz, ba, z, out, md)
    assert rules[1]["ssm_state_indices"] is rule["ssm_state_indices"]
    assert torch.equal(rule["ssm_state_indices"], idx - 1)
    ctx = SimpleNamespace()
    layer._forward_core_rocm(qkvz, ba, z, out, md)
    assert rules[2]["ssm_state_indices"] is not rule["ssm_state_indices"]

    # Mixed or non-spec batches and the interleaved layout stay on vLLM's path.
    for patch in ({"spec_sequence_masks": None}, {"num_prefills": 1}, {"num_decodes": 1}):
        mixed = SimpleNamespace(**{**vars(md), **patch})
        assert layer._forward_core_rocm(qkvz, ba, z, out, mixed) == "stock"
    layer.gqa_interleaved_layout = True
    assert layer._forward_core_rocm(qkvz, ba, z, out, md) == "stock" and not out.any()
    assert len(rules) == 3


def test_sitecustomize_arms_gate_and_boot_gate():
    src = (Path(__file__).resolve().parents[1] / "sitecustomize.py").read_text()
    assert "SUFFIX_ROCM_GDN_MTP" in re.search(r"for _g in \(([^)]*)\)", src).group(1)
    assert '"gdn_mtp_bench": (["-m", "suffix_hybrid.kernels.gdn_mtp_rocm"], {})' in src
