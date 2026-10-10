# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC3 wiring in suffix_hybrid/rocm_patches.py (no vLLM, no GPU):
its two anchors apply alone and next to HC_FUSE / HC_DOWN / HC_BIG, and with HC_BIG on, the
enabled-gate order (PATCHES order, as the import hook applies them) puts HC3's return first.
Fake module and stubs: test_hc_down_wiring's (vLLM 81198e97 verbatim methods); the kernels
are checked on silicon by `python -m suffix_hybrid.kernels.hc3_rocm` (boot gate hc3_bench)."""
import importlib.util
import pathlib
import types

import pytest

from suffix_hybrid import rocm_patches as rp


def _sibling(name):  # works under CI's --import-mode=importlib (no tests dir on sys.path)
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_down = _sibling("test_hc_down_wiring")
FAKE_HC, KERNELS, _calls = _down.FAKE_HC, _down.KERNELS, _down._calls
BIG_KERNELS = (
    "def install_big(module):\n"
    "    module.hc_big_mix = lambda self, h: ('big-mix', self.use_combine, h)\n"
    "    module.hc_big_combine_and_mix = lambda self, h, o, i: ('big-cam', self.use_combine, h, o, i)\n")

HC3 = "SUFFIX_ROCM_HC3"
HC3_KERNELS = (
    "def install_hc3(module):\n"
    "    module.hc3_mix = lambda self, h: ('hc3-mix', self.use_combine, h)\n"
    "    module.hc3_combine_and_mix = lambda self, h, o, i: ('hc3-cam', self.use_combine, h, o, i)\n")


@pytest.mark.parametrize("gates", [(HC3,), (HC3, "SUFFIX_ROCM_HC_FUSE", "SUFFIX_ROCM_HC_DOWN"),
                                   ("SUFFIX_ROCM_HC_FUSE", "SUFFIX_ROCM_HC_DOWN",
                                    "SUFFIX_ROCM_HC_BIG", HC3)])
def test_hc3_returns_first(gates):
    src = FAKE_HC
    for gate in (g for g in rp.PATCHES if g in gates):  # rp.enabled()'s order
        for patch in rp.PATCHES[gate]:
            src = rp.patch_source(patch, src)  # raises unless the anchor occurs exactly once
    hc, kernels = types.ModuleType("hc"), {}
    exec(compile(src, "<hc>", "exec"), vars(hc))
    exec(KERNELS + BIG_KERNELS + HC3_KERNELS, kernels)
    for name in ("install_fuse", "install_down", "install_big", "install_hc3"):
        kernels[name](hc)
    assert _calls(vars(hc)) == [("hc3-mix", True, "h"), ("hc3-mix", False, "h"),
                                ("hc3-cam", True, "h", "out", "inj"),
                                ("hc3-cam", False, "h", "out", "inj")]
    assert rp.PATCHES[HC3][0].after == "suffix_hybrid.kernels.hc3_rocm:install"
    assert list(rp.PATCHES).index(HC3) > list(rp.PATCHES).index("SUFFIX_ROCM_HC_BIG")
