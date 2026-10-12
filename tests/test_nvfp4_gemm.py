# SPDX-License-Identifier: Apache-2.0
"""NVFP4 decode-GEMM spike: CPU contract tests (reference quant/dequant, vLLM
swizzle vs the kernel's sf_offset, and the CPU twin of the kernel's fragment
addressing vs the exact reference). GPU oracle/bench run in-pod only."""
import os
os.environ["SUFFIX_NVFP4_GEMM_FUSED_MAX_M"] = "64"  # these tests exercise fusion at every M
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
    ms = (1, 16, 17, 32, 33, 48, 49, 64)
    assert [ng.plan(8192, 2560, m, fused=False)["entry"] for m in ms] == \
        ["nvfp4_gemm_t1"] * 2 + ["nvfp4_gemm_t2"] * 2 + ["nvfp4_gemm_t3"] * 2 + \
        ["nvfp4_gemm_t4"] * 2
    shape = ("tiles", "splits", "kps", "grid", "ctas")
    for (n, k) in ng.ALL_SHAPES.values():
        for m in range(1, ng.MAX_M + 1):
            pl = ng.plan(n, k, m)
            steps = k // 64
            assert 1 <= pl["splits"] <= min(ng.MAX_SPLITS, steps)
            assert pl["kps"] * (pl["splits"] - 1) < steps <= pl["kps"] * pl["splits"]
            assert pl["splits"] == 1 or pl["kps"] >= ng.MIN_KPS
            assert pl["grid"] == (-(-n // 32), pl["splits"])
            # graph safety: the launch grid is a function of the M bucket only
            top = ng.plan(n, k, 16 * pl["tiles"])
            assert all(pl[f] == top[f] for f in shape)
            assert pl["partial"] <= ng.partial_elems(n, k)
            # the host op re-derives kps from the launched splits: fixed point
            assert ng.plan(n, k, m, pl["splits"])["kps"] == pl["kps"]
            # fused defaults: one launch whenever the quant prologue fits
            assert pl["fused_reduce"] == (pl["splits"] > 1)
            assert pl["launches"] == 1 + (not pl["fused_quant"])
            assert pl["smem"] <= 16 + max(ng.QUANT_SMEM_MAX, ng.STAGE_FLOATS * 4)
            old = ng.plan(n, k, m, fused=False)
            assert old["flags"] == 0 and old["entry"] == f"nvfp4_gemm_t{pl['tiles']}"
            assert old["launches"] == 2 + (pl["splits"] > 1)
            q = ng.plan(n, k, m, prequant=True)
            assert not q["fused_quant"] and q["launches"] == 1


def test_fused_env(monkeypatch):
    monkeypatch.delenv(ng.FUSED_ENV, raising=False)
    assert ng.plan(336, 10240, 1)["flags"] == ng.FUSE_QUANT | ng.FUSE_REDUCE
    monkeypatch.setenv(ng.FUSED_ENV, "0")  # the old 3-launch path, exactly
    pl = ng.plan(336, 10240, 1)
    assert pl["flags"] == 0 and pl["entry"] == "nvfp4_gemm_t1" and pl["launches"] == 3
    monkeypatch.setenv(ng.FUSED_ENV, "reduce")
    pl = ng.plan(336, 10240, 1)
    assert pl["flags"] == ng.FUSE_REDUCE and pl["launches"] == 2
    monkeypatch.setenv(ng.FUSED_ENV, "yes")
    with pytest.raises(ValueError, match=ng.FUSED_ENV):
        ng.plan(336, 10240, 1)


def test_fused_layout():
    # same numbers as src/nvfp4_gemm_oxide.rs fused_layout_matches_python_plan
    both = ng.FUSE_QUANT | ng.FUSE_REDUCE
    assert ng.fused_layout(1, 3, 54, ng.FUSE_REDUCE) == (16 + 55 * 32 * 4, 54)
    assert ng.fused_layout(1, 3, 54, both) == (16 + 55 * 32 * 4, 54)
    assert ng.fused_layout(16, 14, 3, both) == (16 + 16 * (14 * 32 + 16) + 16 * 4 * 15, 3)
    assert ng.fused_layout(64, 8, 6, ng.FUSE_REDUCE) == (16 + 2 * 64 * 32 * 4, 1)
    assert ng.fused_layout(5, 40, 1, ng.FUSE_QUANT) == (16 + 5 * (40 * 32 + 16) + 5 * 4 * 41, 1)
    assert ng.fused_layout(17, 5, 2, 0) == (16, 2)
    # qwen-flash decode: quant fused at M <= 16 everywhere; the widest
    # (ple_kv 12800 x 2560, 2 splits) falls back above the smem budget
    for (n, k) in ng.QWEN_FLASH_SHAPES.values():
        assert all(ng.plan(n, k, m)["fused_quant"] for m in (1, 5, 16))
    assert not ng.plan(12800, 2560, 64)["fused_quant"]
    assert not ng.plan(34816, 5120, 64)["fused_quant"]  # 27b gate_up: 185 KB


def test_smem_rows_are_bank_conflict_free():
    # the 8 fragment rows g of a warp (4 lanes t each) hit 32 distinct banks;
    # the 16 scale rows of one k64 step hit 16 distinct banks
    for kps in range(1, 90):
        astr, sw = kps * 32 + 16, kps | 1
        banks = {((g * astr + 4 * t) // 4) % 32 for g in range(8) for t in range(4)}
        assert len(banks) == 32
        assert len({(sr * sw) % 32 for sr in range(16)}) == 16


FUSED_MS = (1, 5, 16, 17, 40, 64)
FUSED_SPLITS = (1, 2, 3, 10, 54)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("splits", FUSED_SPLITS)
@pytest.mark.parametrize("m", FUSED_MS)
def test_fused_twin_bit_identical_to_old_path(m, splits, mode):
    # K = 2 k64 steps per split (exactly `splits` launched), N = 48: the last
    # column tile has 16 valid columns and two warps without columns
    n, k = 48, 128 * splits
    p = ng.make_problem(m, n, k, seed=m * 7 + splits)
    out, pl, _ = ng.fused_twin(p, mode=mode, splits=splits, seed=m + splits)
    assert pl["splits"] == splits and pl["fused_reduce"] == (splits > 1)
    assert pl["fused_quant"] == (mode == 0)  # every case fits the smem budget
    assert pl["launches"] == 1
    as_f32 = lambda b: (b.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_allclose(as_f32(out), ng.gemm_ref(p), rtol=1e-2, atol=1e-3)  # sanity


def test_fused_counters_survive_graph_replay():
    # replays reuse the SAME counters with different arrival orders: each
    # launch leaves them at 0 and produces identical bits
    p = ng.make_problem(17, 80, 128 * 10, seed=4)
    counters = np.zeros(3, np.int64)
    outs = []
    for replay in range(4):
        out, pl, counters = ng.fused_twin(p, splits=10, seed=replay, counters=counters)
        assert pl["fused_reduce"] and not counters.any()
        outs.append(out)
    assert all(np.array_equal(o, outs[0]) for o in outs)


def test_fixup_twin_detects_stale_counter():
    # a counter left non-zero (no wrap) makes some CTA fix up early: caught
    rng = np.random.default_rng(0)
    partial = rng.standard_normal((5, 3, 32)).astype(np.float32)
    order = [(0, s) for s in range(5)]
    with pytest.raises(AssertionError, match="fixup before"):
        ng.fixup_twin(partial, 1.0, 32, 2, order, np.array([1]), np.zeros((3, 32), np.uint16))
    c = np.zeros(1, np.int64)
    out = ng.fixup_twin(partial, 1.0, 32, 2, order[::-1], c, np.zeros((3, 32), np.uint16))
    assert not c.any()
    np.testing.assert_array_equal(out, ng.splitk_reduce_twin(partial, 1.0))


def test_fixup_order_matters_so_bit_identity_is_meaningful():
    # partials where a different summation order changes the f32 result
    partial = np.array([[[1e8]], [[1.0]], [[-1e8]], [[1.0]]], np.float32).repeat(32, 2)
    fixed = ng.splitk_reduce_twin(partial, 1.0)
    swapped = ng.splitk_reduce_twin(partial[[0, 2, 1, 3]], 1.0)
    assert not np.array_equal(fixed, swapped)
    out = ng.fixup_twin(partial, 1.0, 32, 3, [(0, s) for s in (3, 1, 0, 2)],
                        np.zeros(1, np.int64), np.zeros((1, 32), np.uint16))
    np.testing.assert_array_equal(out, fixed)


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
                       "nvfp4_gemm_t4", "nvfp4_gemm_f1", "nvfp4_gemm_f2", "nvfp4_gemm_f3",
                       "nvfp4_gemm_f4", "nvfp4_splitk_reduce"]
    # one quant function for the separate kernel and the fused prologue
    assert src.count("quant_block(") == 3  # definition + 2 call sites
    assert "atom.acq_rel.gpu.global.inc.u32" in src and "fence.acq_rel.gpu" in src
    assert sorted(entries) == sorted(ng.PARAMS)
    host = (ROOT / "src" / "nvfp4_gemm_oxide.rs").read_text()
    assert all(f'"{e}"' in host for e in entries)


def test_oracle_gate_is_relative_to_vllm():
    # silicon 87fdbc80: bit-identical to vLLM, vLLM itself 1.27e-2 from ref
    assert ng.oracle_ok(1.27e-2, 0.0, 1.27e-2)
    assert ng.oracle_ok(9e-3, 1e-3, 1e-3)          # floor 1e-2
    assert not ng.oracle_ok(1.5e-2, 1e-3, 1.27e-2)  # > 1.1 x vLLM's error
    assert not ng.oracle_ok(5e-3, 3e-2, 5e-3)       # disagrees with vLLM


def test_oracle_ok_triangle_when_vllm_quant_deviates():
    from suffix_hybrid.kernels.nvfp4_gemm import oracle_ok
    # silicon 2026-09-29 qwen gate_up M=16: ours == spec, vLLM 2.3e-2 off -> pass
    assert oracle_ok(0.0, 2.30e-2, 2.30e-2)
    # ours off the spec while vLLM is exact -> fail
    assert not oracle_ok(2.0e-2, 2.0e-2, 0.0)


def test_fused_default_only_m1(monkeypatch):
    monkeypatch.delenv("SUFFIX_NVFP4_GEMM_FUSED_MAX_M", raising=False)
    assert ng.plan(336, 10240, 1)["flags"]          # M=1: single launch
    assert not ng.plan(336, 10240, 5)["flags"]      # M>=5: old path (serial fixup tail on silicon)


# --- L2 prefetch lookahead (flag bits 8..15 of the nvfp4_gemm_f entries) ---
def test_pf_plan_flags_and_env(monkeypatch):
    monkeypatch.delenv(ng.PF_ENV, raising=False)
    monkeypatch.delenv("SUFFIX_NVFP4_GEMM_FUSED_MAX_M", raising=False)
    assert ng.pf_env() is None
    base = ng.plan(34816, 5120, 8)
    assert base["flags"] == 0 and base["entry"] == "nvfp4_gemm_t1" and base["pf"] == 0
    p = ng.plan(34816, 5120, 8, pf=8)  # pf without fusion: f entry, same work
    assert p["flags"] == 8 << ng.PF_SHIFT and p["entry"] == "nvfp4_gemm_f1"
    assert (p["launches"], p["splits"], p["smem"]) == (base["launches"], base["splits"], 16)
    p1 = ng.plan(5120, 17408, 1, pf=8)  # M=1: fused bits kept under the pf bits
    assert p1["flags"] & 0xFF == ng.plan(5120, 17408, 1)["flags"] and p1["flags"] >> 8 == 8
    assert p1["smem"] == ng.plan(5120, 17408, 1)["smem"]
    monkeypatch.setenv(ng.PF_ENV, "4")
    assert ng.pf_env() == 4 and ng.plan(34816, 5120, 8)["flags"] == 4 << 8
    for bad in ("x", "256", "-1"):
        monkeypatch.setenv(ng.PF_ENV, bad)
        with pytest.raises(ValueError, match=ng.PF_ENV):
            ng.plan(34816, 5120, 8)
    with pytest.raises(ValueError):
        ng.plan(34816, 5120, 8, pf=256)


def test_pf_host_op_accepts_only_known_bits():
    host = (ROOT / "src" / "nvfp4_gemm_oxide.rs").read_text()
    assert "pub const PF_MASK: u32 = 0xFF << 8;" in host
    assert "fused & !(FUSE_QUANT | FUSE_REDUCE | PF_MASK) != 0" in host
    src = (ROOT / "kernels-oxide" / "nvfp4_gemm" / "src" / "main.rs").read_text()
    assert f"pub const PF_SHIFT: u32 = {ng.PF_SHIFT};" in src
    assert "let pf = (flags >> PF_SHIFT) & 0xFF;" in src


def test_pf_only_steers_prefetches():
    """Static half of "any pf is bit-identical" (the layer oracle checks it
    on silicon too): in the kernel source `pf` only reaches the prefetch
    helpers, which only compute addresses and issue prefetch.global.L2 —
    no load, no store, no other asm (cf. nvfp4_moe's tunables test)."""
    src = (ROOT / "kernels-oxide" / "nvfp4_gemm" / "src" / "main.rs").read_text()
    allowed = [r"\s*pf: u32,", r"\s*let pf = \(flags >> PF_SHIFT\) & 0xFF;",
               r"\s*if pf == 0 \{", r"\s*let e = if s0 \+ 1 \+ pf < s1 \{ s0 \+ 1 \+ pf \} else \{ s1 \};",
               r"\s*pf_head\(pf, k_begin / 64, k_end / 64, lane, w_row, wsf, n0, kb_pad\);",
               r"\s*if pf != 0 && k0 / 64 \+ pf < k_end / 64 \{",
               r"\s*pf_step\(k0 / 64 \+ pf, lane, w_row, wsf, n0, src.kb_pad\);",
               r"\s*fn pf_head\(pf: u32, .*",
               r".*k_loop::<MT>\(k_begin, k_end, lane, src, w_row, wsf, n0, b0, pf\)"]
    for ln in src.splitlines():
        code = ln.split("//")[0]
        if re.search(r"\bpf\b", code):
            assert any(re.fullmatch(a, code.rstrip()) for a in allowed), ln
    for fn in ("pf_l2", "pf_step", "pf_head"):
        body = re.search(rf"fn {fn}\(.*?\n    \}}\n", src, re.S).group(0)
        assert "*" not in body.replace("*const", "").replace("*mut", "").replace(" * ", ""), fn
        asm = re.findall(r'ptx_asm!\(\s*"([^"]*)"', body)
        assert asm in ([], ["prefetch.global.L2 [%0];"]), (fn, asm)


def _gemm_pf_stream(pf, s0, s1, rows, sf_line):
    """pf_head + k_loop's pf_step for one warp as written: [(iteration,
    address)], iteration -1 = pf_head, s = the loop iteration loading step
    s (s0 < s < s1). rows = the 8 rows' byte addresses (lane/4 = g)."""
    out = []
    if pf == 0:
        return out
    e = min(s0 + 1 + pf, s1)
    for a in rows:
        for t in range(4):
            out += [(-1, a + 32 * sp) for sp in range(s0 + t, e, 4)
                    if sp == s0 or (a + 32 * sp) % 128 == 0]
    out += [(-1, sf_line(sp)) for lane in range(32) for sp in range(s0 + lane, e, 32)]
    for s in range(s0 + 1, s1):
        if s + pf < s1:
            out += [(s, a + 32 * (s + pf)) for a in rows if (a + 32 * (s + pf)) % 128 == 0]
            out.append((s, sf_line(s + pf)))
    return out


@pytest.mark.parametrize("n,k", [(34816, 5120), (5120, 17408), (2816, 2112), (10240, 320),
                                 (336, 10240)])
@pytest.mark.parametrize("pf", [1, 4, 8, 64])
def test_pf_covers_every_weight_and_scale_line_in_bounds(n, k, pf):
    """Every 128 B line a warp's mma loads touch in its split (weight rows,
    rows K/2 apart and not always 128-aligned, + swizzled scale words) is
    prefetched before the loop iteration that loads it, and no prefetch
    address leaves the weight / scale tensors."""
    base, sfbase = 1 << 30, 1 << 40  # torch allocations are >= 256 B aligned
    kh, steps = k // 2, k // 64
    kb_pad, rows_pad = -(-(k // 16) // 4) * 4, -(-n // 128) * 128
    for m in (1, 17):
        pl = ng.plan(n, k, m)
        for split in sorted({0, pl["splits"] - 1}):
            s0, s1 = split * pl["kps"], min(steps, (split + 1) * pl["kps"])
            for n0 in sorted({0, 8 * (n // 16), n - 8}):
                rows = [base + (n0 + g) * kh for g in range(8)]
                got = _gemm_pf_stream(pf, s0, s1, rows,
                                      lambda sp: sfbase + int(ng.sf_offset(n0, 4 * sp, kb_pad)))
                want = {(a + 32 * sp + o) // 128 for a in rows for sp in range(s0, s1)
                        for o in (0, 31)}
                want |= {(sfbase + int(ng.sf_offset(r, 4 * sp + j, kb_pad))) // 128
                         for sp in range(s0, s1) for r in range(n0, n0 + 8) for j in range(4)}
                lines = {a // 128 for _, a in got}
                assert want <= lines, sorted(want - lines)[:4]
                for it, a in got:
                    in_w = base <= a < base + n * kh
                    assert in_w or sfbase <= a < sfbase + rows_pad * kb_pad
                    if in_w:  # issued before the iteration loading its step
                        sp = ((a - base) % kh) // 32
                        assert it < max(sp, s0 + 1)
    assert _gemm_pf_stream(0, 0, 80, [base], lambda sp: base) == []
