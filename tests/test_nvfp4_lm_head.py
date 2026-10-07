# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_NVFP4_LMHEAD: gate, eligibility, entry point, and the hot path
(dynamic activation scale + NVFP4 screen + exact top-R rescore) on CPU twins
of vLLM's scaled_fp4_quant and our GEMM. (vLLM is not importable here; the
head class runs under the in-pod load oracle / boot gate.)"""
import configparser
import pathlib
import sys

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import nvfp4_gemm as ng
from suffix_hybrid.kernels import nvfp4_lm_head as lh

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(lh.GATE, raising=False)
    before = set(sys.modules)
    assert lh.register() is None
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_entry_point():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    assert cfg["vllm.general_plugins"]["suffix_nvfp4_lm_head"] == \
        "suffix_hybrid.kernels.nvfp4_lm_head:register"


def test_our_three_heads_are_eligible():
    for n, k in lh.SHAPES.values():
        assert lh.eligible(n, k, "torch.bfloat16") is None
    assert "% 32" in lh.eligible(151936 + 8, 5120, "torch.bfloat16")
    assert "bf16 only" in lh.eligible(248320, 5120, "torch.float16")


def _twins(w_packed, w_sf_swz, k):
    """CPU stand-ins with the device ops' contracts: quant -> (uint8
    [M, K/2], 128x4-swizzled e4m3 bits); gemm -> bf16 acc * alpha."""
    nkb = k // 16

    def quant(x2, g):
        xq, bits, _ = ng.quantize(x2.float().numpy(), float(g))
        return torch.from_numpy(xq), torch.from_numpy(ng.swizzle_sf(bits))

    def gemm(xq, xsf, alpha):
        m = xq.shape[0]
        xbits = xsf.numpy()[lh.sf_index(np.arange(m), nkb)]
        n = w_packed.shape[0]
        wbits = w_sf_swz[lh.sf_index(np.arange(n), nkb)]
        return torch.from_numpy(
            lh.ref_logits(xq.numpy(), xbits, w_packed, wbits, 1.0, 1.0 / alpha)).bfloat16()
    return quant, gemm


def test_sampled_row_scale_gather_matches_swizzle():
    p = ng.make_problem(3, 256, 320, seed=4)  # K/16 = 20 -> padded to 20
    rows = np.array([0, 7, 128, 255])
    got = p["w_sf_swz"][lh.sf_index(rows, 320 // 16)]
    assert np.array_equal(got, p["w_sf_bits"][rows])


def test_screen_equals_exact_quantized_reference():
    """Dynamic per-call g_x on device + alpha = 1/g_w + in-place 1/g_x ==
    gemm_ref (alpha = 1/(g_x g_w)); RESCORE=0 isolates the NVFP4 screen."""
    for m in (1, 5, 16):
        p = ng.make_problem(m, 512, 256, seed=m)  # make_problem: g = 2688/amax
        quant, gemm = _twins(p["w_packed"], p["w_sf_swz"], 256)
        orig = lh.RESCORE
        lh.RESCORE = 0
        try:
            got = lh.logits_nvfp4(p["x"], p["w_bf16"], float(p["g_w"]), quant, gemm).float()
        finally:
            lh.RESCORE = orig
        ref = torch.from_numpy(ng.gemm_ref(p))
        assert float((got - ref).norm() / ref.norm()) < 1e-2


def _lm_problem(n=4096, k=512, m=16, peak=0.0, seed=0):
    rng = np.random.default_rng(seed)
    w = torch.from_numpy((rng.standard_normal((n, k)) * 0.02).astype(np.float32)).bfloat16()
    x = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float32))
    wt = w.float()[rng.integers(0, n, m)]
    x = (x + peak * wt / wt.norm(dim=1, keepdim=True) * np.sqrt(k)).bfloat16()
    g_w = lh.FP4_RANGE / float(w.float().abs().max())
    wq, bits, _ = ng.quantize(w.float().numpy(), g_w)
    return w, x, g_w, _twins(wq, ng.swizzle_sf(bits), k)


def test_rescore_restores_stock_greedy_where_w4a4_alone_flips():
    # random hidden states = worst case (near-tied top logits)
    w, x, g_w, (quant, gemm) = _lm_problem()
    stock = x @ w.t()
    orig = lh.RESCORE
    lh.RESCORE = 0
    try:
        raw = lh.logits_nvfp4(x, w, g_w, quant, gemm)
    finally:
        lh.RESCORE = orig
    fid = lh.fidelity(stock, raw)
    assert 0.08 < fid["rel"] < 0.2 and fid["top1"] < 0.95  # W4A4 noise is real
    got = lh.logits_nvfp4(x, w, g_w, quant, gemm)
    assert lh.greedy_ok(stock, got) and lh.fidelity(stock, got)["top1"] == 1.0
    top = raw.float().topk(lh.RESCORE, -1).indices  # the screen's candidates
    torch.testing.assert_close(got.gather(1, top).float(), stock.gather(1, top).float(),
                               rtol=2 ** -7, atol=1e-3)  # bf16 rows, not W4A4


def test_greedy_ok_tolerates_only_bf16_ties():
    ref = torch.tensor([[10.0, 9.99, 0.0], [1.0, 3.0, 2.0]])
    assert lh.greedy_ok(ref, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))  # tie
    assert not lh.greedy_ok(ref, torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))


def test_load_oracle_runs_for_every_m(monkeypatch):
    """load_oracle checks M in (1, 4, MAX_M); its screen-only mask must be per
    input row (a [1, R] mask crashed both gemma dev pods at M=4, 2026-09-28)."""
    n, k = 1024, 256
    w, _x, g_w, (quant, gemm) = _lm_problem(n=n, k=k, m=1)
    wq, bits, _ = ng.quantize(w.float().numpy(), g_w)
    st = {"n": n, "k": k, "g_w": g_w, "wq": torch.from_numpy(wq),
          "wsf": torch.from_numpy(ng.swizzle_sf(bits))}

    def run(_st, x):
        return lh.logits_nvfp4(x, w, g_w, quant, gemm)

    # the CPU gemm twin rounds its output to bf16 (the device kernel does not):
    # this test is about per-row mask shapes, not the plumbing tolerance
    monkeypatch.setattr(lh, "PLUMB_REL", 5e-2)
    res = lh.load_oracle(st, run, quant, lambda x: x @ w.t(), n_rows=128)
    assert {"m1", "m4", f"m{lh.MAX_M}"} <= set(res) and res["plumb_rel"] <= lh.PLUMB_REL


def test_max_m_env(monkeypatch):
    monkeypatch.delenv(lh.MAX_M_ENV, raising=False)
    assert lh.max_m_env() == 64
    monkeypatch.setenv(lh.MAX_M_ENV, "16")
    assert lh.max_m_env() == 16
    for bad in ("15", "65", "x"):
        monkeypatch.setenv(lh.MAX_M_ENV, bad)
        with pytest.raises(ValueError, match="16..64"):
            lh.max_m_env()


def test_graph_replay_path_matches_eager(monkeypatch):
    """replay(): static x rows in, static out rows back, per-M graph; a CPU
    stand-in 'graph' re-runs the eager hot path on the static slices (what
    the captured graph does on device) and must equal the eager call."""
    n, k, max_m = 1024, 256, 8
    w, _x, g_w, (quant, gemm) = _lm_problem(n=n, k=k, m=1)

    class G:
        def __init__(self, m):
            self.m = m

        def replay(self):
            st["gout"][:self.m].copy_(lh.logits_nvfp4(st["gx"][:self.m], w, g_w, quant, gemm))

    st = {"gx": torch.zeros(max_m, k, dtype=torch.bfloat16),
          "gout": torch.empty(max_m, n, dtype=torch.bfloat16),
          "graphs": [None] + [G(m) for m in range(1, max_m + 1)]}
    for m in (1, 3, max_m):
        x = _lm_problem(n=n, k=k, m=m, seed=m)[1]
        got = lh.replay(st, x)
        assert got.shape == (m, n) and got.data_ptr() == st["gout"].data_ptr()
        assert torch.equal(got, lh.logits_nvfp4(x, w, g_w, quant, gemm))
    monkeypatch.delenv(lh.GRAPH_ENV, raising=False)
    assert lh.graph_on()
    monkeypatch.setenv(lh.GRAPH_ENV, "0")
    assert not lh.graph_on()
