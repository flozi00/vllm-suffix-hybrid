# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-experts decode kernel: CPU contract tests (reference quant vs
the dense spike's, swizzle, deterministic routing, a flat-buffer twin of the
kernel's addressing vs the reference, oracle gate, eligibility, fail-loud
verdict, gate/entry point, host<->kernel ABI). Silicon: `python -m
suffix_hybrid.kernels.nvfp4_moe oracle|bench` + the per-layer load oracle."""
import configparser
import pathlib
import re
import sys

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import nvfp4_gemm as ng
from suffix_hybrid.kernels import nvfp4_moe as nm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_quant_codes_match_dense_reference():
    x = torch.randn(5, 256) * 3
    g = 2688.0 / float(x.abs().max())
    q, sfb = nm.quant_codes(x, g)
    q_np, sfb_np, _ = ng.quantize(x.numpy(), np.float32(g))
    assert np.array_equal(q.numpy(), q_np) and np.array_equal(sfb.numpy(), sfb_np)


def test_swizzle_matches_vllm_layout_and_inverts():
    bits = torch.randint(0, 255, (200, 36), dtype=torch.uint8)
    flat = nm.swizzle(bits)
    assert np.array_equal(flat.numpy(), ng.swizzle_sf(bits.numpy()))
    assert torch.equal(nm.unswizzle(flat, 200, 36), bits)


def test_route_twin_is_deterministic_and_sorted():
    ids = torch.tensor([[3, 0], [0, 7], [3, -1]])
    se, so, sc, pl = nm.route_twin(ids, 8)
    assert se == [0, 3, 7, -1, -1, -1]
    assert so[:3] == [0, 2, 4] and sc[:3] == [2, 2, 1]
    assert pl == [1, 2, 0, 4, 3]  # pairs ascending within each expert


def kernel_twin(p, x, ids, tw):
    """Replays kernels-oxide/nvfp4_moe from FLAT buffers with the kernel's
    own offsets: route slots, aq/asf row-major at token = pair / topk, expert
    bases e*2I*H/2 and e*r128(2I)*r4(H/16), up row j / gate row I+j,
    swizzled sf via sf_offset, inter/hq at row = pair, fc2 epilogue
    g2*topk_w, combine in ascending k over valid ids."""
    e_count, two_i, kh1 = p["w13"].shape
    hdim, idim = kh1 * 2, two_i // 2
    m, topk = ids.shape
    se, so, sc, pl = nm.route_twin(ids, e_count)
    f8 = lambda b: torch.from_numpy(np.asarray(b, np.uint8)).view(torch.float8_e4m3fn).float()

    def gemm_rows(aq, asf, a_rows, k, w, w_sf, e, rows_out, n_rows_total):
        kh, nkb = k // 2, k // 16
        kb_pad, rows_pad = -(-nkb // 4) * 4, -(-n_rows_total // 128) * 128
        wf, sff = w.reshape(-1).numpy(), w_sf.reshape(-1).view(torch.uint8).numpy()
        rows = np.asarray(rows_out)
        wb = wf[e * n_rows_total * kh + rows[:, None] * kh + np.arange(kh)[None]]
        kbs = np.arange(nkb)
        wsf = sff[e * rows_pad * kb_pad + ng.sf_offset(rows[:, None], kbs[None], kb_pad)]
        wd = nm.dequant(torch.from_numpy(wb), f8(wsf))
        ab = aq.reshape(-1).numpy()[np.asarray(a_rows)[:, None] * kh + np.arange(kh)[None]]
        asb = asf.reshape(-1).numpy()[np.asarray(a_rows)[:, None] * nkb + kbs[None]]
        return nm.dequant(torch.from_numpy(ab), f8(asb)) @ wd.T

    aq, asf = nm.quant_codes(x.float(), p["a1g"])
    inter = torch.full((m * topk, idim), float("nan"))
    for s, e in enumerate(se):
        if e < 0:
            continue
        for r0 in range(0, sc[s], 16):  # 16-row mma chunks
            pairs = pl[so[s] + r0: so[s] + min(sc[s], r0 + 16)]
            a_rows = [q // topk for q in pairs]
            up = gemm_rows(aq, asf, a_rows, hdim, p["w13"], p["w13_sf"], e,
                           list(range(idim)), two_i)
            gate = gemm_rows(aq, asf, a_rows, hdim, p["w13"], p["w13_sf"], e,
                             list(range(idim, two_i)), two_i)
            a = p["g1"][e]
            inter[pairs] = nm.act_ref(a * gate, p["act"]) * (a * up)
    valid = [(0 <= int(v) < e_count) for v in ids.reshape(-1)]
    hq, hsf = nm.quant_codes(torch.nan_to_num(inter), p["a2g"])
    y = torch.full((m * topk, hdim), float("nan"))
    for s, e in enumerate(se):
        if e < 0:
            continue
        for r0 in range(0, sc[s], 16):
            pairs = pl[so[s] + r0: so[s] + min(sc[s], r0 + 16)]
            c = gemm_rows(hq, hsf, pairs, idim, p["w2"], p["w2_sf"], e,
                          list(range(hdim)), hdim)
            y[pairs] = c * (p["g2"][e] * tw.reshape(-1)[pairs])[:, None]
    out = torch.zeros(m, hdim)
    for t in range(m):
        for k in range(topk):
            if valid[t * topk + k]:
                out[t] += y[t * topk + k]
    return out


@pytest.mark.parametrize("act,m,e_count,topk", [("gelu_tanh", 24, 2, 2),
                                                ("silu", 5, 6, 3),
                                                ("gelu_tanh", 1, 8, 4)])
def test_kernel_twin_matches_reference(act, m, e_count, topk):
    p = nm.make_problem(e_count, 128, 64, act, seed=m)
    ids, tw = nm.rand_routing(m, e_count, topk, "cpu", seed=m)
    x = torch.randn(m, 128).bfloat16()
    ref = nm.moe_ref(p, x, ids, tw)
    twin = kernel_twin(p, x, ids, tw)
    assert torch.isfinite(ref).all() and ref.abs().max() > 0
    torch.testing.assert_close(twin, ref, rtol=1e-4, atol=1e-4)


def test_invalid_expert_ids_are_skipped():
    p = nm.make_problem(4, 64, 64, "silu", seed=3)
    ids = torch.tensor([[0, 2], [3, -1]])
    tw = torch.tensor([[0.5, 0.5], [1.0, 9.0]])
    x = torch.randn(2, 64).bfloat16()
    ref = nm.moe_ref(p, x, ids, tw)
    torch.testing.assert_close(kernel_twin(p, x, ids, tw), ref, rtol=1e-4, atol=1e-4)
    only = nm.moe_ref(p, x, ids.clamp(min=0), torch.tensor([[0.5, 0.5], [1.0, 0.0]]))
    torch.testing.assert_close(ref, only)  # id -1 contributes nothing


def test_reference_approximates_dense_bf16_moe():
    """Quantized reference vs the unquantized expert math (catches layout
    mistakes such as swapped up/gate halves or a wrong alpha)."""
    torch.manual_seed(0)
    e_count, hdim, idim, topk, m = 4, 128, 64, 2, 6
    p = nm.make_problem(e_count, hdim, idim, "gelu_tanh", seed=7)
    ids, tw = nm.rand_routing(m, e_count, topk, "cpu", seed=1)
    x = torch.randn(m, hdim).bfloat16()
    want = torch.zeros(m, hdim)
    for t in range(m):
        for k in range(topk):
            e = int(ids[t, k])
            w13 = nm._expert_w(p["w13"], p["w13_sf"], e) * p["g1"][e] * p["a1g"]
            w2 = nm._expert_w(p["w2"], p["w2_sf"], e) * p["g2"][e] * p["a2g"]
            gu = x[t].float() @ w13.T
            h = nm.act_ref(gu[idim:], 1) * gu[:idim]
            want[t] += tw[t, k] * (h @ w2.T)
    got = nm.moe_ref(p, x, ids, tw)
    rel = float((got - want).norm() / want.norm())
    assert rel < 0.3, rel  # 2x activation FP4 quant ~0.17; swapped up/gate ~0.7


def test_oracle_gate_is_relative_to_flashinfer():
    assert nm.oracle_ok(5e-3, 1e-2, 5e-3)
    assert nm.oracle_ok(3e-2, 5e-2, 3e-2)  # both far from ref, equally
    assert not nm.oracle_ok(3e-2, 1e-2, 5e-3)  # we are worse than FlashInfer
    assert not nm.oracle_ok(5e-3, 9e-2, 5e-3)  # we disagree with FlashInfer


GOOD = dict(quant_dtype="nvfp4", backend="FLASHINFER_CUTLASS", scale_swizzled=True,
            act="gelu_tanh", clamp_limit=None, swiglu_alpha=None,
            router_weight_on_input=False, bias=False, tp=1, ep=1, dp=1,
            expert_map=False, E=128, H=2816, I=704, sf_exact=True, gscale_shared=True)


@pytest.mark.parametrize("change,needle", [
    ({}, None),
    ({"act": "silu"}, None),
    ({"quant_dtype": None}, "not NVFP4"),
    ({"backend": "FLASHINFER_TRTLLM"}, "backend"),
    ({"act": "swigluoai"}, "activation"),
    ({"clamp_limit": 7.0}, "clamp"),
    ({"router_weight_on_input": True}, "router_weight"),
    ({"tp": 2}, "single GPU"),
    ({"expert_map": True}, "expert_map"),
    ({"I": 720}, "shape"),
    ({"E": 512}, "shape"),
    ({"sf_exact": False}, "swizzled layout"),
    ({"gscale_shared": False}, "gscales"),
])
def test_eligibility(change, needle):
    why = nm.eligibility({**GOOD, **change})
    assert (why is None) if needle is None else (needle in why)


def test_verdict_fails_loud():
    assert nm.verdict(30, 30, {}) is None
    assert "no RoutedExperts" in nm.verdict(0, 0, {})
    assert nm.verdict(31, 0, {}) == ""  # weights not processed yet
    msg = nm.verdict(30, 0, {"NvFp4 MoE backend FLASHINFER_TRTLLM": 30})
    assert "NOT ENGAGED" in msg and "TRTLLM" in msg


def test_workspace_sizes_and_growth():
    ws = nm.workspace("cpu", 32, 8, 2816, 704, 128)
    p = 256
    assert [t.numel() for t in ws] == [32 * 1408, 32 * 176, p * 704, p * 352, p * 44,
                                       p * 2816, 3 * 128 + p]
    assert nm.workspace("cpu", 1, 8, 2816, 704, 128, ws) is ws  # no realloc


def test_max_m_knob(monkeypatch):
    monkeypatch.delenv(nm.MAX_M_ENV, raising=False)
    assert nm.max_m() == 32
    monkeypatch.setenv(nm.MAX_M_ENV, "65")
    with pytest.raises(ValueError):
        nm.max_m()


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(nm.GATE, raising=False)
    before = set(sys.modules)
    assert nm.register() is None
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_entry_point():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    assert cfg["vllm.general_plugins"]["suffix_nvfp4_moe"] == \
        "suffix_hybrid.kernels.nvfp4_moe:register"


def test_host_launch_args_match_kernel_abi():
    src = (ROOT / "kernels-oxide" / "nvfp4_moe" / "src" / "main.rs").read_text()
    host = (ROOT / "src" / "nvfp4_moe_oxide.rs").read_text()
    params = {}
    for name, sig in re.findall(r"pub unsafe fn (moe_\w+)\((.*?)\)\s*\{", src, re.S):
        params[name] = len([a for a in sig.split(",") if a.strip()])
    assert set(params) == {"moe_route", "moe_quant_rows", "moe_fc1", "moe_fc2", "moe_combine"}
    for arr, kern in [("route_args", "moe_route"), ("qx_args", "moe_quant_rows"),
                      ("qh_args", "moe_quant_rows"), ("fc1_args", "moe_fc1"),
                      ("fc2_args", "moe_fc2"), ("comb_args", "moe_combine")]:
        body = re.search(rf"let {arr} = \[(.*?)\];", host, re.S).group(1)
        n = len([a for a in re.split(r",\s*\n", body) if a.strip()])
        assert n == params[kern], (arr, n, params[kern])
        assert f'f("{kern}")' in host
    assert '"arch": "sm_120a"' in (ROOT / "kernels-oxide" / "nvfp4_moe"
                                   / "oxide-variants.json").read_text()
