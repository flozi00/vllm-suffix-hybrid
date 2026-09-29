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


@pytest.mark.parametrize("m,mode,splits", [(1, 0, 1), (5, 0, 2), (16, 0, 3),
                                           (1, 1, 1), (5, 1, 3), (16, 1, 2)])
def test_kernel_twin_matches_reference(m, mode, splits):
    # mode 0: our quant buffers; mode 1: vLLM's pre-quantized activation with
    # swizzled scales (fused SiLU*mul quant route); splits: split-K + reduce
    p = ng.make_problem(m, 16, 192, seed=m)
    ref = ng.gemm_ref(p)
    twin = ng.kernel_twin(p, mode=mode, splits=splits)
    np.testing.assert_allclose(twin, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m,n,mode,splits", [(17, 32, 0, 1), (17, 48, 1, 2), (40, 16, 0, 3),
                                             (40, 48, 1, 1), (63, 32, 1, 3), (64, 16, 0, 2),
                                             (63, 48, 0, None)])
def test_kernel_twin_multi_tile_and_n_tail(m, n, mode, splits):
    # tiles 2..4 sharing each weight fragment; M not a multiple of 16 (rows
    # past M: zero rows in mode 0, predicated loads in mode 1); N = 16 / 48
    # leave a CTA's trailing warps without columns (N % 32 != 0)
    p = ng.make_problem(m, n, 192, seed=m + n)
    twin = ng.kernel_twin(p, mode=mode, splits=splits)
    np.testing.assert_allclose(twin, ng.gemm_ref(p), rtol=1e-5, atol=1e-5)


def test_plan_tiles_and_split_ranges():
    assert [ng.plan(8192, 2560, m)["entry"] for m in (1, 16, 17, 32, 33, 48, 49, 64)] == \
        ["nvfp4_gemm_t1"] * 2 + ["nvfp4_gemm_t2"] * 2 + ["nvfp4_gemm_t3"] * 2 + \
        ["nvfp4_gemm_t4"] * 2
    for (n, k) in ng.ALL_SHAPES.values():
        for m in range(1, ng.MAX_M + 1):
            pl = ng.plan(n, k, m)
            steps = k // 64
            assert 1 <= pl["splits"] <= min(ng.MAX_SPLITS, steps)
            assert pl["kps"] * (pl["splits"] - 1) < steps <= pl["kps"] * pl["splits"]
            assert pl["splits"] == 1 or pl["kps"] >= ng.MIN_KPS
            assert pl["grid"] == (-(-n // 32), pl["splits"])
            # graph safety: the plan is a function of the M bucket only
            assert pl == dict(ng.plan(n, k, 16 * pl["tiles"]), partial=pl["partial"])
            assert pl["partial"] <= ng.partial_elems(n, k)


def test_split_policy():
    s = lambda n, k, m=16: ng.plan(n, k, m)["splits"]
    # qwen3.8-27b / gemma (M <= 16): the silicon-validated policy (gemma down 7 -> 9)
    assert s(34816, 5120) == 1 and s(16384, 5120) == 2
    assert s(5120, 17408) == 5 and s(5120, 6144) == 5
    assert s(2816, 2112) == 9  # 88 CTAs -> 792
    # qwen-flash TP2: narrow-N / deep-K HC projections split deep ...
    assert s(336, 10240, 1) == 54 and s(320, 10240, 5) == 54  # 11/10 CTAs -> ~560
    assert s(336, 10240, 64) == 20  # partial budget K / (128 * 4 tiles)
    assert s(2560, 3072, 5) == 10 and s(2560, 3072, 64) == 6
    assert s(2560, 320, 5) == 2 and s(10240, 320, 40) == 1
    # ... wide-N ones stay (almost) whole
    assert s(8192, 2560) == 3 and s(12800, 2560) == 2 and s(6656, 2560) == 4
    assert s(48, 2560, 1) == 20 and s(48, 2560, 64) == 5  # GDN in_proj_ba: 2 CTAs


def test_unswizzle_and_torch_exact_ref_match_numpy():
    p = ng.make_problem(40, 48, 256, seed=2)
    xq, xsf_bits, _ = ng.quantize(p["x"].float().numpy(), p["g_x"])
    xsf = torch.from_numpy(ng.swizzle_sf(xsf_bits))
    assert torch.equal(ng.unswizzle_sf(xsf, 40, 16), torch.from_numpy(xsf_bits))
    # FlashInfer-padded weight rows (48 -> 64) are ignored past N
    w = torch.from_numpy(np.pad(p["w_packed"], ((0, 16), (0, 0)), constant_values=0x77))
    ref = ng.exact_ref(torch.from_numpy(xq), xsf, w, torch.from_numpy(p["w_sf_swz"]),
                       float(p["alpha"]), 48, chunk_elems=256 * 20)
    np.testing.assert_allclose(ref.numpy(), ng.gemm_ref(p), rtol=1e-5, atol=1e-5)


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
    assert entries == ["nvfp4_quant_act", "nvfp4_gemm_t1", "nvfp4_gemm_t2", "nvfp4_gemm_t3",
                       "nvfp4_gemm_t4", "nvfp4_splitk_reduce"]
    assert sorted(entries) == sorted(ng.PARAMS)
    host = (ROOT / "src" / "nvfp4_gemm_oxide.rs").read_text()
    assert all(f'"{e}"' in host for e in entries)


def test_oracle_gate_is_relative_to_vllm():
    # silicon 87fdbc80: bit-identical to vLLM, vLLM itself 1.27e-2 from ref
    assert ng.oracle_ok(1.27e-2, 0.0, 1.27e-2)
    assert ng.oracle_ok(9e-3, 1e-3, 1e-3)          # floor 1e-2
    assert not ng.oracle_ok(1.5e-2, 1e-3, 1.27e-2)  # > 1.1 x vLLM's error
    assert not ng.oracle_ok(5e-3, 3e-2, 5e-3)       # disagrees with vLLM
