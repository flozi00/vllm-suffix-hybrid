# SPDX-License-Identifier: Apache-2.0
"""CPU test for suffix_hybrid.rocm_aiter_pad (no vLLM, no GPU).

Silicon check: an MI350P pod with SUFFIX_ROCM_AITER_PAD=1 logs
"[suffix rocm-aiter-pad] ACTIVE" and Qwen3.8-Flash-Next MXFP4 TP1 GSM8K
recovers to ~0.96 (unpatched ~0.74).
"""
import importlib
import sys

import pytest

from suffix_hybrid import rocm_aiter_pad as rap

FAKE = (
    "def pads(hidden_pad, intermediate_pad, moe_config, activation):\n"
    "    if True:\n"
    "        if activation != 'situ':\n" + rap.OLD +
    "    return hidden_pad, intermediate_pad\n"
)


def test_patch_source_and_drift():
    out = rap.patch_source(FAKE)
    assert rap.OLD not in out and rap.NEW in out
    compile(out, "<t>", "exec")
    with pytest.raises(RuntimeError, match="drifted"):
        rap.patch_source(out)


def test_hook_rewrites_before_first_exec(tmp_path, monkeypatch):
    (tmp_path / "fake_aiter_moe.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(rap, "TARGET", "fake_aiter_moe")
    monkeypatch.setenv(rap.GATE, "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rap.install_post_import_hook()
    mod = importlib.import_module("fake_aiter_moe")
    try:
        cfg = type("C", (), {"tp_size": 1})()
        # Unpatched: (0, 768) for raw (100, 384); patched passes raw through.
        assert mod.pads(100, 384, cfg, "silu") == (100, 384)
        assert getattr(mod, rap._MARK)
    finally:
        sys.modules.pop("fake_aiter_moe", None)


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(rap.GATE, raising=False)
    assert rap.install_post_import_hook() is False
