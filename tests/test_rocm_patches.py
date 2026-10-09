# SPDX-License-Identifier: Apache-2.0
"""CPU test for suffix_hybrid.rocm_patches (no vLLM, no GPU).

Silicon check: an MI350P pod with both gates logs one "[suffix rocm-patch]
ACTIVE" line per patch; Qwen3.8-Flash-Next MXFP4 TP1 GSM8K is ~0.96 (unpatched
pad ~0.74), and a >384-token prefill step no longer logs "Pcie atomics not
enabled" on worker-09.
"""
import importlib
import sys

import pytest

from suffix_hybrid import rocm_patches as rp

PAD = rp.PATCHES["SUFFIX_ROCM_AITER_PAD"]
TOPK = rp.PATCHES["SUFFIX_ROCM_QSA_TOPK_ROWS"]
MQA = rp.PATCHES["SUFFIX_ROCM_QSA_MQA"]
FAKE = {
    "fake_aiter_moe": (
        "def pads(hidden_pad, intermediate_pad, moe_config, activation):\n"
        "    if True:\n"
        "        if activation != 'situ':\n" + PAD.old +
        "    return hidden_pad, intermediate_pad\n"),
    "fake_qsa_ops": (
        "_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024\n"
        "def chunk(columns):\n" + TOPK.old +
        "    return rows_per_chunk\n"
        "def qsa_mqa_paged():\n"
        "    return 'stock'\n"),
    "fake_qsa_kernel": "def install(module):\n    module.qsa_mqa_paged = lambda: 'suffix'\n",
}


def test_patch_sources_and_drift():
    for patch, src in ((PAD, FAKE["fake_aiter_moe"]), (TOPK, FAKE["fake_qsa_ops"])):
        out = rp.patch_source(patch, src)
        assert patch.old not in out and patch.new in out
        compile(out, "<t>", "exec")
        with pytest.raises(RuntimeError, match="drifted"):
            rp.patch_source(patch, out)


def test_hook_rewrites_every_enabled_target(tmp_path, monkeypatch):
    for name, src in FAKE.items():
        (tmp_path / f"{name}.py").write_text(src)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_AITER_PAD", PAD._replace(target="fake_aiter_moe"))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_QSA_TOPK_ROWS", TOPK._replace(target="fake_qsa_ops"))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_QSA_MQA",
                        MQA._replace(target="fake_qsa_ops", after="fake_qsa_kernel:install"))
    for gate, p in rp.PATCHES.items():  # only the gates retargeted to a fake above
        if p.target in FAKE:
            monkeypatch.setenv(gate, "1")
        else:
            monkeypatch.delenv(gate, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rp.install_post_import_hook()
    try:
        moe = importlib.import_module("fake_aiter_moe")
        cfg = type("C", (), {"tp_size": 1})()
        # Unpatched: (0, 768) for raw (100, 384); patched passes raw through.
        assert moe.pads(100, 384, cfg, "silu") == (100, 384)
        qsa = importlib.import_module("fake_qsa_ops")
        assert qsa.chunk(1024) == 384 and qsa.chunk(1 << 20) == 32  # cap, workspace bound kept
        assert qsa.qsa_mqa_paged() == "suffix"  # after hook ran on the rewritten module
        assert getattr(moe, rp._MARK) and getattr(qsa, rp._MARK)
        assert not any(getattr(f, rp._MARK, False) for f in sys.meta_path)  # finder retired
    finally:
        for name in FAKE:
            sys.modules.pop(name, None)


def test_gate_off_is_inert(monkeypatch):
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    assert rp.install_post_import_hook() is False
    assert MQA.after == "suffix_hybrid.kernels.qsa_mqa_rocm:install" and not MQA.old
