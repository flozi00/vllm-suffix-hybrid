# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC_WQ wiring in suffix_hybrid/rocm_patches.py (no vLLM, no GPU):
the gate takes mxfp4 / fp8 (not 1), anything else refuses to start, and it only installs the
after hook (no source anchor), so it composes with every other HC gate. The kernels are
checked on silicon by `python -m suffix_hybrid.kernels.hc_wq_rocm` (boot gate hc_wq_bench)."""
import pytest

from suffix_hybrid import rocm_patches as rp

WQ = "SUFFIX_ROCM_HC_WQ"


@pytest.fixture
def clean_env(monkeypatch):
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    return monkeypatch


@pytest.mark.parametrize("value", ["mxfp4", "fp8"])
def test_values_enable_the_hook(clean_env, value):
    clean_env.setenv(WQ, value)
    todo = rp.enabled()
    assert todo == {rp.PATCHES[WQ].target: [rp.PATCHES[WQ]]}
    assert rp.PATCHES[WQ].after == "suffix_hybrid.kernels.hc_wq_rocm:install"
    assert not rp.PATCHES[WQ].old  # no anchor: composes with HC_FUSE / HC_DOWN / HC_BIG


@pytest.mark.parametrize("value", ["1", "nvfp4", "MXFP4"])
def test_other_values_refuse(clean_env, value):
    clean_env.setenv(WQ, value)
    with pytest.raises(SystemExit, match=WQ):
        rp.enabled()


@pytest.mark.parametrize("value", ["", "0"])
def test_off(clean_env, value):
    clean_env.setenv(WQ, value)
    assert rp.enabled() == {}


def test_composes_with_hc_gates(clean_env):
    for gate in ("SUFFIX_ROCM_HC_FUSE", "SUFFIX_ROCM_HC_DOWN", "SUFFIX_ROCM_HC_BIG"):
        clean_env.setenv(gate, "1")
    clean_env.setenv(WQ, "mxfp4")
    hc = rp.enabled()[rp.PATCHES[WQ].target]
    assert rp.PATCHES[WQ] in hc and len(hc) == 7  # 2 + 2 + 2 anchors + the hook
