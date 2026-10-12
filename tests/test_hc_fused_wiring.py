# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC_FUSE wiring in suffix_hybrid/rocm_patches.py
(no vLLM, no GPU). Both anchors occur once in vLLM 81198e97
vllm/models/qwen4_exp/amd/hyperconnection.py; the kernel's numerics are checked
on silicon by `python -m suffix_hybrid.kernels.hc_fused_rocm` (boot gate
hc_fuse_bench).
"""
import importlib
import sys

import pytest

from suffix_hybrid import rocm_patches as rp

MIX, CAM = rp.PATCHES["SUFFIX_ROCM_HC_FUSE"]
# GatedResidual with vLLM's mix / combine_and_mix tails (the anchors) over
# stand-in ops that return what they were called with.
FAKE_HC = (
    "def hc_silu(x, hc):\n    return ('silu', x, hc)\n"
    "def hc_gate_mix(xn, gate, hc):\n    return ('mix', xn, gate, hc)\n"
    "class Up:\n    weight = 'W_up'\n    def __call__(self, x):\n        return ('up', x)\n"
    "class GatedResidual:\n"
    "    hc_count = 4\n"
    "    input_mix_weight_up = Up()\n"
    "    def mix(self, hidden_states, lora, xn, injection):\n" + MIX.old +
    "        self, hidden_states, lora, xn, injection):\n" + CAM.old +
    "        self, hidden_states, block_output, injection):\n"
    "        return 'combine'\n")
KERNEL = "def install(module):\n    module.hc_up_gate_mix = lambda *a: ('fused',) + a\n"
STOCK = ("h", ("mix", "xn", ("up", ("silu", "lora", 4)), 4), "inj")
FUSED = ("h", ("fused", "lora", "W_up", "xn", 4), "inj")


def _calls(ns):
    gr = ns["GatedResidual"]()
    return gr.mix("h", "lora", "xn", "inj"), gr.combine_and_mix("h", "lora", "xn", "inj")


def test_anchors_rewrite_both_tails_once():
    stock: dict = {}
    exec(compile(FAKE_HC, "<stock>", "exec"), stock)
    assert _calls(stock) == (STOCK, STOCK)
    out = FAKE_HC
    for patch in (MIX, CAM):
        out = rp.patch_source(patch, out)
    fused: dict = {"hc_up_gate_mix": lambda *a: ("fused",) + a}
    exec(compile(out, "<fused>", "exec"), fused)
    assert _calls(fused) == (FUSED, FUSED)
    for patch in (MIX, CAM):
        with pytest.raises(RuntimeError, match="drifted"):  # already rewritten: 0 hits
            rp.patch_source(patch, out)
    with pytest.raises(RuntimeError, match="drifted"):  # mix tail without its def: 2 hits
        rp.patch_source(CAM, FAKE_HC.replace("def combine_and_mix(", "def combine("))


def test_hook_applies_both_patches_with_one_gate(tmp_path, monkeypatch, capsys):
    (tmp_path / "fake_hc.py").write_text(FAKE_HC)
    (tmp_path / "fake_hc_kernel.py").write_text(KERNEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_HC_FUSE", (
        MIX._replace(target="fake_hc", after="fake_hc_kernel:install"),
        CAM._replace(target="fake_hc")))
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv("SUFFIX_ROCM_HC_FUSE", "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rp.install_post_import_hook()
    try:
        hc = importlib.import_module("fake_hc")
        assert _calls(vars(hc)) == (FUSED, FUSED)
        assert getattr(hc, rp._MARK)
        assert not any(getattr(f, rp._MARK, False) for f in sys.meta_path)  # finder retired
        assert capsys.readouterr().err.count("[suffix rocm-patch] ACTIVE: HC silu") == 2
    finally:
        for name in ("fake_hc", "fake_hc_kernel"):
            sys.modules.pop(name, None)
    assert MIX.after == "suffix_hybrid.kernels.hc_fused_rocm:install" and not CAM.after
