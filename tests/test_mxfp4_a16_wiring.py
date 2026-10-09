# SPDX-License-Identifier: Apache-2.0
"""CPU test for SUFFIX_ROCM_MXFP4_A16 wiring (no vLLM, no AITER, no GPU).

Silicon check: boot gate mxfp4_a16_bench (python -m suffix_hybrid.kernels.mxfp4_a16_rocm)
exits 0 on the MI350P; a pod with the gate logs "[suffix rocm-patch] ACTIVE: dense
MXFP4 small M via AITER gemm_a16wfp4" in every process that imports vLLM's mxfp4 kernels.
"""
import importlib
import sys
import types

import pytest
import torch

from suffix_hybrid import rocm_patches as rp
from suffix_hybrid.kernels import mxfp4_a16_rocm as a16

A16 = rp.PATCHES["SUFFIX_ROCM_MXFP4_A16"]
# vLLM 81198e97 kernels/linear/mxfp4/aiter.py, trimmed: same nesting, and the ASM branch
# has the same `if x_scales is None:` one level deeper, so the anchor must still match once.
FAKE = (
    "def dynamic_mxfp4_quant(x):\n"
    "    return 'xq', 'xs'\n"
    "if True:  # vLLM: if is_aiter_found_and_supported():\n"
    "    def gemm_with_dynamic_quant(x, weight, weight_scale, rocm_use_aiter_fp4_asm_gemm=False,\n"
    "                                out_dtype=None, x_scales=None):\n"
    "        M = x.shape[0]\n"
    "        if rocm_use_aiter_fp4_asm_gemm:\n"
    "            if M <= 64:\n"
    "                if x_scales is None:\n"
    "                    # use hip quant kernel for performance\n"
    "                    x_q, x_s = 'hip', 'hip'\n"
    "            return 'asm'\n"
    "        else:\n" + A16.old +
    "            else:\n"
    "                x_q = x\n"
    "                x_s = x_scales\n"
    "            return 'stock', x_q\n")


def test_anchor_once_and_drift():
    out = rp.patch_source(A16, FAKE)
    assert A16.old not in out and A16.new in out
    compile(out, "<t>", "exec")
    with pytest.raises(RuntimeError, match="drifted"):
        rp.patch_source(A16, out)


def test_hook_routes_small_m_only(tmp_path, monkeypatch):
    (tmp_path / "fake_mxfp4_aiter.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_MXFP4_A16", A16._replace(target="fake_mxfp4_aiter"))
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv("SUFFIX_ROCM_MXFP4_A16", "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    seen = []
    monkeypatch.setattr(a16, "gemm_a16", lambda x, w, s, dt: seen.append((w, s, dt)) or (
        "a16" if x.shape[0] <= 2 else None))
    assert rp.install_post_import_hook()
    try:
        mod = importlib.import_module("fake_mxfp4_aiter")
        op = mod.gemm_with_dynamic_quant
        x1, x3 = torch.zeros(1, 64), torch.zeros(3, 64)
        assert op(x1, "w", "s", False, "bf16") == "a16"
        assert op(x3, "w", "s", False, "bf16") == ("stock", "xq")  # above max_m: stock quant
        assert op(x1, "w", "s", False, "bf16", x_scales="xs")[1] is x1  # pre-quantized: stock
        assert op(x1, "w", "s", True, "bf16") == "asm"
        assert seen == [("w", "s", "bf16")] * 2
        assert getattr(mod, rp._MARK)
    finally:
        sys.modules.pop("fake_mxfp4_aiter", None)


@pytest.fixture
def fake_aiter(monkeypatch):
    """aiter.ops.triton.gemm_a16wfp4 stand-in: AITER's config / split choice set per test."""
    m = types.ModuleType("aiter.ops.triton.gemm_a16wfp4")
    m.cfg, m.split, m.calls = {"BLOCK_SIZE_K": 512, "NUM_KSPLIT": 1}, None, []  # gfx950 DEFAULT
    m._get_config = lambda M, N, K: (dict(m.cfg), False)
    m.get_splitk = lambda K, bk, ks: m.split
    m.gemm_a16wfp4 = lambda *args: m.calls.append(args) or "y"
    monkeypatch.setitem(sys.modules, m.__name__, m)
    a16._config.cache_clear()
    yield m
    a16._config.cache_clear()


def test_config_is_even_k(fake_aiter):
    assert a16._config(8, 16384, 2560)["BLOCK_SIZE_K"] == 512
    assert a16._config(8, 2560, 6144)["BLOCK_SIZE_K"] == 512
    assert a16._config(8, 2560, 640) == {"BLOCK_SIZE_K": 128, "NUM_KSPLIT": 1}  # 640 = 5 x 128
    # A tuned split-K is kept when AITER's own split is even (get_splitk(3072, 512, 4)) ...
    fake_aiter.cfg, fake_aiter.split = {"BLOCK_SIZE_K": 512, "NUM_KSPLIT": 4}, (1536, 512, 4)
    assert a16._config(4, 2560, 6144)["NUM_KSPLIT"] == 4
    # ... else it becomes one EVEN_K launch (get_splitk(320, 512, 4) = (1024, 512, 1)).
    fake_aiter.split = (1024, 512, 1)
    assert a16._config(4, 2560, 640) == {"BLOCK_SIZE_K": 128, "NUM_KSPLIT": 1}


def test_gemm_a16_args(fake_aiter):
    x = torch.zeros(5, 640, dtype=torch.bfloat16)
    w = torch.zeros(2560, 320, dtype=torch.uint8)  # [N, K/2]
    ws = torch.zeros(20, 2560, dtype=torch.uint8)  # vLLM non-ASM layout [K/32, N]
    assert a16.gemm_a16(x, w, ws, torch.bfloat16, max_m=4) is None
    assert a16.gemm_a16(x[:, :32], w[:, :16], ws[:1], torch.bfloat16) is None  # K % 64
    assert a16.gemm_a16(x, w, ws, torch.bfloat16, max_m=5) == "y"
    xa, wa, wsa, atomic, dtype, y, cfg = fake_aiter.calls[-1]
    assert xa is x and wa is w and atomic is False and dtype is torch.bfloat16 and y is None
    assert wsa.shape == (2560, 20) and wsa.stride() == (1, 2560)  # weight_scale.T, as stock
    assert cfg == {"BLOCK_SIZE_K": 128, "NUM_KSPLIT": 1}
