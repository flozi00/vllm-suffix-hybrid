# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC_BIG wiring in suffix_hybrid/rocm_patches.py (no vLLM, no
GPU): its two anchors apply alone and next to HC_FUSE / HC_DOWN in any order, and the
patched mix / combine_and_mix return through hc_big_mix / hc_big_combine_and_mix before any
stock op runs. Fake module and stubs: test_hc_down_wiring's (vLLM 81198e97 verbatim methods);
the kernels are checked on silicon by `python -m suffix_hybrid.kernels.hc_big_rocm` (boot
gate hc_big_bench)."""
import types

import pytest
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(  # CI runs pytest --import-mode=importlib
    "test_hc_down_wiring", pathlib.Path(__file__).with_name("test_hc_down_wiring.py"))
_down = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_down)
FAKE_HC, KERNELS, _calls = _down.FAKE_HC, _down.KERNELS, _down._calls

from suffix_hybrid import rocm_patches as rp

BIG = "SUFFIX_ROCM_HC_BIG"
BIG_KERNELS = (
    "def install_big(module):\n"
    "    module.hc_big_mix = lambda self, h: ('big-mix', self.use_combine, h)\n"
    "    module.hc_big_combine_and_mix = lambda self, h, o, i: ('big-cam', self.use_combine, h, o, i)\n")


@pytest.mark.parametrize("gates", [(BIG,), (BIG, "SUFFIX_ROCM_HC_FUSE", "SUFFIX_ROCM_HC_DOWN"),
                                   ("SUFFIX_ROCM_HC_DOWN", "SUFFIX_ROCM_HC_FUSE", BIG)])
def test_big_returns_first_in_any_order(gates):
    src = FAKE_HC
    for gate in gates:
        for patch in rp.PATCHES[gate]:
            src = rp.patch_source(patch, src)  # raises unless the anchor occurs exactly once
    hc, kernels = types.ModuleType("hc"), {}
    exec(compile(src, "<hc>", "exec"), vars(hc))
    exec(KERNELS + BIG_KERNELS, kernels)
    for name in ("install_fuse", "install_down", "install_big"):
        kernels[name](hc)
    assert _calls(vars(hc)) == [("big-mix", True, "h"), ("big-mix", False, "h"),
                                ("big-cam", True, "h", "out", "inj"),
                                ("big-cam", False, "h", "out", "inj")]
    assert rp.PATCHES[BIG][0].after == "suffix_hybrid.kernels.hc_big_rocm:install"
