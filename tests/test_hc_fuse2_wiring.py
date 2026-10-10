# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC_FUSE2 wiring in suffix_hybrid/rocm_patches.py (no vLLM, no
GPU): alone and with HC_FUSE / HC_DOWN in any order, mix and combine_and_mix return through
hc_fuse2_mix / hc_fuse2_combine_and_mix with their arguments. Same fake hyperconnection
module as test_hc_down_wiring (vLLM 81198e97 methods verbatim). The kernels' numerics are
checked on silicon by `python -m suffix_hybrid.kernels.hc_fuse2_rocm` (boot gate
hc_fuse2_bench)."""
import importlib.util
import pathlib
import types

import pytest

from suffix_hybrid import rocm_patches as rp

_spec = importlib.util.spec_from_file_location(
    "hc_down_wiring", pathlib.Path(__file__).with_name("test_hc_down_wiring.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)
FUSE2 = "SUFFIX_ROCM_HC_FUSE2"
KERNELS = hd.KERNELS + (
    "def install_fuse2(module):\n"
    "    module.hc_fuse2_mix = lambda self, h: ('mix2', self.use_combine, h)\n"
    "    module.hc_fuse2_combine_and_mix = (\n"
    "        lambda self, h, b, i: ('cam2', self.use_combine, h, b, i))\n")


@pytest.mark.parametrize("gates", [(FUSE2,), (hd.FUSE, hd.DOWN, FUSE2), (FUSE2, hd.FUSE, hd.DOWN),
                                   (hd.DOWN, FUSE2, hd.FUSE)])
def test_returns_through_fuse2_with_hc_fuse_down_in_any_order(gates):
    src = hd.FAKE_HC
    for gate in gates:
        for patch in rp.PATCHES[gate]:
            src = rp.patch_source(patch, src)  # raises unless the anchor occurs exactly once
    hc, kernels = types.ModuleType("hc"), {}
    exec(compile(src, "<hc>", "exec"), vars(hc))
    exec(KERNELS, kernels)
    for name in ("install_fuse", "install_down", "install_fuse2"):
        kernels[name](hc)
    assert hd._calls(vars(hc)) == [("mix2", True, "h"), ("mix2", False, "h"),
                                   ("cam2", True, "h", "out", "inj"),
                                   ("cam2", False, "h", "out", "inj")]
    # combine() (PLE layer, pipeline-parallel tails) stays stock
    assert hc.GatedResidual(True).combine("h", "out", "inj") == "combined"


def test_install_hook_named():
    mix, cam = rp.PATCHES[FUSE2]
    assert mix.after == "suffix_hybrid.kernels.hc_fuse2_rocm:install" and not cam.after
