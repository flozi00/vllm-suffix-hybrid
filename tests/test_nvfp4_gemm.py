# SPDX-License-Identifier: Apache-2.0
"""NVFP4 decode-GEMM spike: CPU contract tests (reference quant/dequant, vLLM
swizzle vs the kernel's sf_offset, and the CPU twin of the kernel's fragment
addressing vs the exact reference). GPU oracle/bench run in-pod only."""
import pathlib
import re

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import nvfp4_gemm as ng

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_e2m1_rne_ties_and_saturation():
    v = np.array([0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, 100.0])
    assert ng.E2M1[ng.e2m1_code(v)].tolist() == [0, 0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6, 6]


def test_quant_dequant_roundtrip_error_bound():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    g = np.float32(2688 / np.abs(x).max())
    q, _, sf = ng.quantize(x, g)
    xr = ng.dequant(q, sf) / g
    blk_amax = np.abs(x).reshape(3, -1, 16).max(-1)
    err = np.abs(xr - x).reshape(3, -1, 16).max(-1)
    assert (err <= blk_amax * 0.26 + 1e-6).all()  # half a step of the e2m1 grid


def test_low_nibble_is_even_element():
    x = np.zeros((1, 16), np.float32)
    x[0, 0], x[0, 1] = 6.0, -0.5
    q, _, sf = ng.quantize(x, np.float32(1.0))
    assert q[0, 0] & 0xF == 7 and q[0, 0] >> 4 == 0x8 | 1


def test_swizzle_matches_kernel_sf_offset():
    rng = np.random.default_rng(1)
    r, c = 200, 36
    bits = rng.integers(0, 255, (r, c), dtype=np.uint8)
    flat = ng.swizzle_sf(bits)
    kb_pad = -(-c // 4) * 4
    for row in range(r):
        for kb in range(c):
            assert flat[ng.sf_offset(row, kb, kb_pad)] == bits[row, kb]


@pytest.mark.parametrize("m", [1, 5, 16])
def test_kernel_twin_matches_reference(m):
    p = ng.make_problem(m, 16, 192, seed=m)
    ref = ng.gemm_ref(p)
    twin = ng.kernel_twin(p)
    np.testing.assert_allclose(twin, ref, rtol=1e-5, atol=1e-5)


def test_reference_tracks_bf16_matmul():
    p = ng.make_problem(4, 64, 512, seed=3)
    ref = ng.gemm_ref(p)
    full = (p["x"].float() @ p["w_bf16"].float().T).numpy()
    rel = np.linalg.norm(ref - full) / np.linalg.norm(full)
    assert rel < 0.25  # W4A4 quantization noise (~0.1-0.15 on gaussians)
    p["w_packed"] = p["w_packed"][::-1].copy()  # a layout bug looks like this
    bad = np.linalg.norm(ng.gemm_ref(p) - full) / np.linalg.norm(full)
    assert bad > 1.0


def test_kernel_source_uses_nvf4_block_scale_mma_and_sm120a():
    src = (ROOT / "kernels-oxide" / "nvfp4_gemm" / "src" / "main.rs").read_text()
    assert "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3" in src
    var = (ROOT / "kernels-oxide" / "nvfp4_gemm" / "oxide-variants.json").read_text()
    assert '"arch": "sm_120a"' in var
    entries = re.findall(r"pub unsafe fn (\w+)\(", src)
    assert entries == ["nvfp4_quant_act", "nvfp4_gemm_m16"]
