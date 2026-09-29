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


def kernel_twin(p, x, ids, tw, base=0):
    """Replays kernels-oxide/nvfp4_moe from FLAT buffers with the kernel's
    own offsets: route slots, aq/asf row-major at token = pair / topk, expert
    bases e*2I*H/2 and e*r128(2I)*r4(H/16), up row j / gate row I+j,
    swizzled sf via sf_offset, inter/hq at row = pair, fc2 epilogue
    g2*topk_w, combine in ascending k over valid ids."""
    e_count, two_i, kh1 = p["w13"].shape
    hdim, idim = kh1 * 2, two_i // 2
    m, topk = ids.shape
    se, so, sc, pl = nm.route_twin(ids, e_count, base)
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
    valid = [(0 <= int(v) - base < e_count) for v in ids.reshape(-1)]
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
    torch.testing.assert_close(twin, ref.float(), rtol=1e-4, atol=1e-4)


def test_invalid_expert_ids_are_skipped():
    p = nm.make_problem(4, 64, 64, "silu", seed=3)
    ids = torch.tensor([[0, 2], [3, -1]])
    tw = torch.tensor([[0.5, 0.5], [1.0, 9.0]])
    x = torch.randn(2, 64).bfloat16()
    ref = nm.moe_ref(p, x, ids, tw)
    torch.testing.assert_close(kernel_twin(p, x, ids, tw), ref.float(), rtol=1e-4, atol=1e-4)
    only = nm.moe_ref(p, x, ids.clamp(min=0), torch.tensor([[0.5, 0.5], [1.0, 0.0]]))
    torch.testing.assert_close(ref, only)  # id -1 contributes nothing


def test_ep_twin_global_ids_match_local_reference_and_zero_offrank_tokens():
    """EP rank 2 of 4 (16 global experts, 4 local, id_base 8): the kernel twin
    fed GLOBAL ids == reference on to_local ids; rows routed only off-rank
    (t % 3 == 1) are exactly zero; an all-off-rank batch is all zero."""
    local, e_global, base = 4, 16, 8
    p = nm.make_problem(local, 128, 64, "silu", seed=5)
    ids, tw = nm.rand_routing(7, e_global, 3, "cpu", seed=2, base=base, local=local)
    lid = nm.to_local(ids, base, local)
    assert (lid[1::3] < 0).all() and (lid >= 0).any()
    assert (ids[lid >= 0] - base == lid[lid >= 0]).all()
    x = torch.randn(7, 128).bfloat16()
    ref = nm.moe_ref(p, x, lid, tw)
    twin = kernel_twin(p, x, ids, tw, base)
    torch.testing.assert_close(twin, ref.float(), rtol=1e-4, atol=1e-4)
    dead = (lid < 0).all(1)
    assert dead.any() and (twin[dead] == 0).all() and (ref[dead] == 0).all()
    ids, tw = nm.rand_routing(5, e_global, 3, "cpu", seed=3, base=base, local=local, dead=True)
    assert (nm.to_local(ids, base, local) < 0).all()
    assert (kernel_twin(p, x[:5], ids, tw, base) == 0).all()
    assert nm.route_twin(ids, local, base)[0] == [-1] * local


def test_ep_base_accepts_only_flashinfers_linear_map():
    lin = [-1] * 8 + [0, 1, 2, 3] + [-1] * 4  # rank 2 of 4, 16 experts
    assert nm.ep_base(lin, 2, 4, 16, 4) == 8
    assert nm.ep_base(None, 0, 1, 16, 16) == 0
    assert nm.ep_base(lin, 1, 4, 16, 4) is None  # map / rank mismatch
    rr = [(g // 4 if g % 4 == 2 else -1) for g in range(16)]  # round_robin
    assert nm.ep_base(rr, 2, 4, 16, 4) is None
    assert nm.ep_base(None, 0, 4, 16, 4) is None  # EP without a map
    glm = [g - 160 if 160 <= g < 192 else -1 for g in range(256)]
    assert nm.ep_base(glm, 5, 8, 256, 32) == 160


def test_judge_enforces_offrank_zero_contract():
    ref = torch.randn(3, 8)
    lid = torch.tensor([[0, 1], [-1, -1], [2, -1]])
    ref[1] = 0
    ok, _ = nm.judge(ref.clone(), ref.clone(), ref, lid)
    assert ok
    bad = ref.clone()
    bad[1, 0] = 1e-6  # off-rank token not exactly zero (ours)
    assert not nm.judge(bad, ref.clone(), ref, lid)[0]
    assert not nm.judge(ref.clone(), bad, ref, lid)[0]  # stock broke the contract
    fi_nan = ref.clone()
    fi_nan[1] = float("nan")  # FlashInfer never wrote the row
    assert not nm.judge(ref.clone(), fi_nan, ref, lid)[0]


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


def test_oracle_gate_is_absolute_vs_spec_and_triangle_vs_flashinfer():
    assert nm.oracle_ok(2e-3, 3e-2, 3e-2)  # FlashInfer far from spec: not our problem
    assert not nm.oracle_ok(1e-2, 3e-2, 3e-2)  # tanh.approx-class drift fails
    assert not nm.oracle_ok(2e-3, 9e-2, 3e-2)  # we disagree beyond both errors
    assert nm.oracle_ok(4e-3, 1.9e-2, 1e-3)  # floor 2e-2


def test_gelu_tanh_sigmoid_form_matches_kernel_constant():
    """Kernel: gelu_tanh(x) = x / (1 + 2^-(C*(x + 0.044715x^3))), C =
    2*sqrt(2/pi)*log2(e), == 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))."""
    src = (ROOT / "kernels-oxide" / "nvfp4_moe" / "src" / "main.rs").read_text()
    assert '"tanh.approx' not in src  # no tanh.approx asm
    c = float(re.search(r"if kind == 1 \{ ([0-9._]+) \*", src).group(1).replace("_", ""))
    x = torch.linspace(-12, 12, 4001, dtype=torch.float64)
    got = x / (1 + torch.exp2(-c * (x + 0.044715 * x ** 3)))
    torch.testing.assert_close(got, nm.act_ref(x, 1), rtol=1e-7, atol=1e-9)


def _ws_from_spec(p, x, ids, tw):
    """The workspace a spec-exact kernel would leave (f32 intermediates)."""
    m, topk = ids.shape
    e_count, hdim, ih = p["w2"].shape
    ws = nm.workspace("cpu", m, topk, hdim, ih * 2, e_count)
    aq, asf = nm.quant_codes(x.float(), p["a1g"])
    inter = nm.fc1_ref(p, x, ids).float()
    hq, hsf = nm.quant_codes(inter, p["a2g"])
    hd = nm.qdq(inter, p["a2g"]).double()
    y = torch.zeros(m * topk, hdim, dtype=torch.float64)
    for e, pr in nm._groups(ids, e_count).items():
        y[pr] = (float(p["g2"][e]) * (hd[pr] @ nm._expert_w(p["w2"], p["w2_sf"], e).double().T)
                 * tw.reshape(-1)[pr, None].double())
    y = y.float()
    for t, v in zip(ws, (aq, asf, inter, hq, hsf, y)):
        t[:v.numel()] = v.reshape(-1)
    acc = torch.zeros(m, hdim)
    for k in range(topk):
        acc = torch.where((ids[:, k] >= 0)[:, None], acc + y.view(m, topk, hdim)[:, k], acc)
    return ws, acc.bfloat16()


def test_stages_pin_drift_to_the_stage_that_made_it():
    p = nm.make_problem(4, 128, 64, "gelu_tanh", seed=4)
    ids, tw = nm.rand_routing(6, 4, 2, "cpu", seed=4)
    ids[1, 1] = -1  # off-rank pair: its rows are garbage and must be ignored
    x = torch.randn(6, 128).bfloat16()
    ws, out = _ws_from_spec(p, x, ids, tw)
    ws[2].view(12, 64)[3] = float("nan")  # inter row of pair 3 (= token 1, k 1)
    ok, msg = nm.stages(p, x, ids, tw, ws, out)
    assert ok, msg
    for idx, stage, bump in [(2, "fc1", lambda t: t.mul_(1 + 2 ** -11)),
                             (5, "fc2", lambda t: t.mul_(1 + 1e-3)),
                             (0, "xq", lambda t: t[:4].fill_(0x77))]:
        bad = list(ws)
        bad[idx] = ws[idx].clone()
        bump(bad[idx])
        ok, msg = nm.stages(p, x, ids, tw, bad, out)
        assert not ok and msg.split(f" {stage}=")[1].split()[0] != "0.0e+00", msg


def test_fi_emulation_is_farther_from_spec_than_ours():
    """The accuracy anatomy in nvfp4_moe.py, pinned: FlashInfer's bf16
    intermediates alone cost more than REF_TOL; our f32 path does not."""
    p = nm.make_problem(8, 512, 256, "gelu_tanh", seed=9)
    ids, tw = nm.rand_routing(8, 8, 4, "cpu", seed=9)
    x = torch.randn(8, 512).bfloat16()
    ref = nm.moe_ref(p, x, ids, tw)
    ours = kernel_twin(p, x, ids, tw).bfloat16()
    emu = nm.moe_ref(p, x, ids, tw, fi=True).bfloat16()
    assert nm._rel(ours, ref) < nm.REF_TOL < nm._rel(emu, ref)


GOOD = dict(quant_dtype="nvfp4", backend="FLASHINFER_CUTLASS", scale_swizzled=True,
            act="gelu_tanh", clamp_limit=None, swiglu_alpha=None,
            router_weight_on_input=False, bias=False, tp=1, ep=1, dp=1, all2all=False,
            eplb=False, mk_shared_overlap=False, expert_map=False, ep_base=0,
            E=128, H=2816, I=704, sf_exact=True, gscale_shared=True)
GLM_EP = dict(GOOD, act="silu", ep=8, expert_map=True, ep_base=160, E=32, H=6144, I=2048)
GLM_TP8 = dict(GOOD, act="silu", tp=8, E=256, H=6144, I=256)


@pytest.mark.parametrize("change,needle", [
    ({}, None),
    ({"act": "silu"}, None),
    ({"quant_dtype": None}, "not NVFP4"),
    ({"backend": "FLASHINFER_TRTLLM"}, "backend"),
    ({"act": "swigluoai"}, "activation"),
    ({"clamp_limit": 7.0}, "clamp"),
    ({"router_weight_on_input": True}, "router_weight"),
    ({"tp": 2}, None),  # TP-sharded intermediate: partial sums, vLLM all-reduces
    ({"dp": 2}, "DP/all2all"),
    ({"all2all": True}, "DP/all2all"),
    ({"eplb": True}, "EPLB"),
    ({"mk_shared_overlap": True}, "shared experts"),
    ({"tp": 2, "ep": 2, "expert_map": True, "ep_base": 0}, "not both"),
    ({"expert_map": True, "ep_base": None}, "linear"),
    ({"ep": 8, "ep_base": None}, "linear"),
    ({"I": 720}, "shape"),
    ({"E": 512}, "shape"),
    ({"sf_exact": False}, "swizzled layout"),
    ({"gscale_shared": False}, "gscales"),
])
def test_eligibility(change, needle):
    why = nm.eligibility({**GOOD, **change})
    assert (why is None) if needle is None else (needle in why)


@pytest.mark.parametrize("change,needle", [
    ({}, None),
    ({"ep_base": None, "expert_map": True}, "linear"),  # round_robin placement
    ({"eplb": True}, "EPLB"),
    ({"I": 2048 // 3}, "shape"),
    # GLM MTP draft layer: experts in the modelopt ignore list -> bf16
    ({"quant_dtype": None}, "not NVFP4"),
])
@pytest.mark.parametrize("glm", [GLM_EP, GLM_TP8])
def test_eligibility_glm(glm, change, needle):
    why = nm.eligibility({**glm, **change})
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
    monkeypatch.setenv(nm.MAX_M_ENV, "257")
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
    assert set(params) == {"moe_route", "moe_quant_rows", "moe_fc1", "moe_fc2", "moe_combine",
                           "moe_route_quant", "moe_fc1_quant", "moe_fc2_combine"}
    assert params == nm.PARAMS  # the loader's stale-cubin check uses these
    for arr, kern in [("route_args", "moe_route"), ("qx_args", "moe_quant_rows"),
                      ("qh_args", "moe_quant_rows"), ("fc1_args", "moe_fc1"),
                      ("fc2_args", "moe_fc2"), ("comb_args", "moe_combine"),
                      ("rq_args", "moe_route_quant"), ("fq_args", "moe_fc1_quant"),
                      ("fc_args", "moe_fc2_combine")]:
        body = re.search(rf"let {arr} = \[(.*?)\];", host, re.S).group(1)
        n = len([a for a in re.split(r",\s*\n", body) if a.strip()])
        assert n == params[kern], (arr, n, params[kern])
        assert f'("{kern}", ' in host and f"&{arr})" in host
    # dynamic smem the fused kernels index: fc1 tile 16 x 32 f32, ys topk x 8 f32
    assert '"moe_fc1_quant", ((i / 32) as u32, slots as u32), 128, 16 * 32 * 4,' in host
    # route_par: 16 i32 warp totals + u16 ids padded to 8 pairs, 256 threads
    assert "let route_par_smem = 64 + 2 * p.div_ceil(8) * 8;" in host
    assert '("moe_route_quant", (1 + (m * h / 16).div_ceil(256) as u32, 1), 256, route_par_smem' in host
    assert "const RT: u32 = 256;" in src and "let ids = unsafe { $sh.add(16) } as *mut u16;" in src
    assert "32 * nw, (k * 8 * 4) as u32" in host and "let nw = k.min(16)" in host
    assert "#[launch_bounds(512)]\n    pub unsafe fn moe_fc2_combine(" in src
    assert '"arch": "sm_120a"' in (ROOT / "kernels-oxide" / "nvfp4_moe"
                                   / "oxide-variants.json").read_text()


# ---------------------------------------------------------------------------
# Lane-level emulator of kernels-oxide/nvfp4_moe: moe_route (per-thread
# counting + tid-0 prefix), moe_fc1 / moe_fc2 transcribed per lane (g, t,
# sfa_row, ok0/ok1/oks masks, load_a / load_b byte offsets, the 3-stage
# register pipeline exactly as written) over the PTX m16n8k64
# mxf4nvf4.scale_vec::4X fragment layout (A: a0/a1 rows g/g+8 k 8t.., a2/a3
# k 32+8t..; B: b0/b1 col g; scale A row r from lane 4r (r<8) / 4(r-8)+1,
# scale B col n from lane 4n (thread-id selectors 0); C: rows g/g+8 cols
# 2t,2t+1), and moe_combine. Workspace starts NaN so any read of an unwritten
# row or a wrong lane->row mapping poisons the output.
# ---------------------------------------------------------------------------
_LANE = np.arange(32)
_LG, _LT = _LANE // 4, _LANE % 4
_LUT = np.array(nm._E2M1 + tuple(-v for v in nm._E2M1))
_F8 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double().numpy()


def _ld32(buf, off, ok):
    off = np.where(ok, off, 0).astype(np.int64)
    assert (off[ok] % 4 == 0).all() and (off[ok] + 4 <= buf.size).all(), "misaligned/OOB ld32"
    v = sum(buf[off + i].astype(np.uint64) << np.uint64(8 * i) for i in range(4))
    return np.where(ok, v, 0).astype(np.uint64)


def _nib(v, j):
    return ((v >> np.uint64(4 * j)) & np.uint64(0xF)).astype(np.int64)


def _mma(c, a, b):
    """c (32,4) f32, a (32,5) [a0..a3, sfa], b (32,3) [b0, b1, sfb] -> (32,4)."""
    A, B = np.zeros((16, 64)), np.zeros((8, 64))
    sa, sb = np.zeros((16, 4)), np.zeros((8, 4))
    for r in range(4):
        for j in range(8):
            A[_LG + 8 * (r & 1), 8 * _LT + 32 * (r >> 1) + j] = _LUT[_nib(a[:, r], j)]
    for r in range(2):
        for j in range(8):
            B[_LG, 8 * _LT + 32 * r + j] = _LUT[_nib(b[:, r], j)]
    for i in range(4):
        byte = lambda v: ((v >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.int64)
        sa[_LG[_LT == 0], i] = _F8[byte(a[_LT == 0, 4])]
        sa[_LG[_LT == 1] + 8, i] = _F8[byte(a[_LT == 1, 4])]
        sb[_LG[_LT == 0], i] = _F8[byte(b[_LT == 0, 2])]
    d = (A * np.repeat(sa, 16, 1)) @ (B * np.repeat(sb, 16, 1)).T
    cm = np.zeros((16, 8))
    for q in range(4):
        cm[_LG + 8 * (q >> 1), 2 * _LT + (q & 1)] = c[:, q]
    d = (d + cm).astype(np.float32)
    return np.stack([d[_LG + 8 * (q >> 1), 2 * _LT + (q & 1)] for q in range(4)], 1)


def _act32(x, kind):
    x = x.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        z = (np.float32(2.3022082) * (x + np.float32(0.044715) * x * x * x) if kind == 1
             else x * np.float32(1.442695))
        return (x / (np.float32(1) + np.exp2(-z))).astype(np.float32)


def _route_emu(ids_flat, id_base, e_count, slots):
    loc = ids_flat.astype(np.int64) - id_base
    cnt = np.array([(loc == tid).sum() for tid in range(e_count)], np.int32)
    se, so, sc = np.full(slots, -1), np.zeros(slots, np.int64), np.zeros(slots, np.int64)
    offs, off, s = np.zeros(e_count, np.int64), 0, 0
    for e in range(e_count):
        offs[e] = off
        if cnt[e] > 0 and s < slots:
            se[s], so[s], sc[s] = e, off, cnt[e]
            s += 1
        off += cnt[e]
    pl = np.full(len(loc) + 1, -7, np.int64)  # sentinel: must never be read
    for tid in range(e_count):
        w = offs[tid]
        for p in range(len(loc)):
            if loc[p] == tid:
                pl[w] = p
                w += 1
    return se, so, sc, pl


def _route_par_emu(ids_flat, id_base, e_count, slots, sentinel=-7):
    """route_par (moe_route_quant block 0) per thread as written: u16 id
    staging padded to 8 pairs, thread e counting 4 words (8 ids) per step,
    shfl.up warp scans of (cnt, cnt > 0), 8 warp totals, slot / empty-slot
    writes, the pair walk. -> (se, so, sc, pl, ws_route [3E + P]) with the
    host's slot_expert / slot_off / slot_cnt / pair_list layout, unwritten
    words = sentinel."""
    rt, pairs = 256, len(ids_flat)
    np8 = -(-pairs // 8) * 8
    loc = ids_flat.astype(np.int64) - id_base
    ids16 = np.full(np8, 0xFFFF, np.int64)
    for p in range(pairs):
        if 0 <= loc[p] < e_count:
            ids16[p] = loc[p]
    words = ids16[0::2] | (ids16[1::2] << 16)
    cnt = np.zeros(rt, np.int64)
    for tid in range(min(e_count, rt)):
        for w in range(0, np8 // 2, 4):
            for j in range(4):
                cnt[tid] += int(words[w + j] & 0xFFFF == tid) + int(words[w + j] >> 16 == tid)
    act = (cnt > 0).astype(np.int64)
    lane, warp = np.arange(rt) % 32, np.arange(rt) // 32
    c, a = cnt.copy(), act.copy()
    d = 1
    while d < 32:
        src = np.where(lane >= d, np.arange(rt) - d, np.arange(rt))  # shfl.up
        yc, ya = c[src], a[src]
        c, a = np.where(lane >= d, c + yc, c), np.where(lane >= d, a + ya, a)
        d *= 2
    wc, wa = c[31::32], a[31::32]
    off = np.array([wc[:w].sum() for w in warp]) + c - cnt
    slot = np.array([wa[:w].sum() for w in warp]) + a - act
    na = wa.sum()
    ws = np.full(3 * e_count + pairs, sentinel, np.int64)
    for tid in range(rt):
        if tid < e_count and cnt[tid] > 0 and slot[tid] < slots:
            ws[slot[tid]], ws[e_count + slot[tid]], ws[2 * e_count + slot[tid]] = tid, off[tid], cnt[tid]
        if tid < slots and tid >= na:
            ws[tid], ws[2 * e_count + tid] = -1, 0
    for tid in range(min(e_count, rt)):
        if cnt[tid] > 0:
            o = off[tid]
            for w in range(0, np8 // 2, 4):
                for j in range(8):
                    if (words[w + j // 2] >> (16 * (j % 2))) & 0xFFFF == tid:
                        ws[3 * e_count + o] = 2 * w + j
                        o += 1
    pl = np.full(pairs + 1, -7, np.int64)
    pl[:pairs] = ws[3 * e_count:]
    return ws[:slots], ws[e_count:e_count + slots], ws[2 * e_count:2 * e_count + slots], pl, ws


def _route_cta_ws(ids_flat, id_base, e_count, slots, sentinel=-7):
    """route_cta's ws_route words as written (empty slots' slot_off and the
    pair_list tail beyond the local pairs are never stored)."""
    se, so, sc, pl = _route_emu(ids_flat, id_base, e_count, slots)
    ws = np.full(3 * e_count + len(ids_flat), sentinel, np.int64)
    ws[:slots], ws[2 * e_count:2 * e_count + slots] = se, sc
    act = se >= 0
    ws[e_count:e_count + slots][act] = so[act]
    n = int(sc.sum())
    ws[3 * e_count:3 * e_count + n] = pl[:n]
    return ws


@pytest.mark.parametrize("seed,pairs,e_count,base,span", [
    (0, 1, 256, 0, 256), (1, 10, 256, 256, 512), (2, 7, 5, 5, 15), (3, 640, 256, 0, 512),
    (4, 2560, 256, 256, 512), (5, 333, 255, 0, 255), (6, 64, 32, 160, 256), (7, 13, 1, 0, 3),
    (8, 512, 128, 0, 128), (9, 40, 256, 0, 256)])
def test_route_par_matches_route_cta_word_for_word(seed, pairs, e_count, base, span):
    """moe_route_quant's parallel route == moe_route's route_cta on every
    ws_route word (incl. what neither writes): EP off-rank ids, -1, pair
    counts off the 8-pad, experts > 32 per warp, P < E and P >> E."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(-1, span, pairs)
    slots = min(e_count, pairs)
    par = _route_par_emu(ids, base, e_count, slots)
    assert np.array_equal(par[4], _route_cta_ws(ids, base, e_count, slots))
    se, so, sc, pl = nm.route_twin(torch.from_numpy(ids), e_count, base)
    assert list(par[0]) == se and list(par[2]) == sc and list(par[3][:len(pl)]) == pl


def _pf_stream(la, steps, rows, sf_line):
    """pf_head + the dot loops' lookahead for one weight stream of one
    warp, as written: [(iteration, address)] (iteration -1 = pf_head, s =
    dot loop iteration s >= 2); rows = the 8 row byte addresses (row g),
    sf_line(sp) = the warp's first row's scale address of step sp. pf_head
    requests the same (row, step) lines as pf_step would, only spread over
    the lanes (step t mod 4 by lane (g, t), scale step lane mod 32)."""
    out = []

    def step(it, sp, first):
        for a in rows:
            if first or (a + 32 * sp) % 128 == 0:
                out.append((it, a + 32 * sp))
        out.append((it, sf_line(sp)))
    if la:
        n0 = min(2 + la, steps)
        for a in rows:  # lane (g, t): steps t, t + 4, ..
            for t in range(4):
                out += [(-1, a + 32 * sp) for sp in range(t, n0, 4)
                        if sp == 0 or (a + 32 * sp) % 128 == 0]
        out += [(-1, sf_line(sp)) for lane in range(32) for sp in range(lane, n0, 32)]
        for s in range(2, steps):
            if s + la < steps:
                step(s, s + la, False)
    return out


@pytest.mark.parametrize("hdim,idim", [(2560, 640), (2816, 704), (6144, 2048), (6144, 256),
                                       (128, 192)])
@pytest.mark.parametrize("la", [1, 8, 16, 32, nm.LA_FULL])
def test_prefetch_covers_every_weight_and_scale_line_in_bounds(hdim, idim, la):
    """Every 128 B line an fc1 / fc2 warp's mma loads touch (weight rows +
    swizzled scale words) is prefetched exactly when la > 0, before the
    loop iteration that loads it, and no prefetch leaves the expert's
    weight / scale slab (fc2 rows are I/2 = 320 / 352 B: not 128-aligned)."""
    rng = np.random.default_rng(la + hdim)
    base = 256 * 1000  # torch allocations are >= 256 B aligned
    for k, nrows in ((hdim, 2 * idim), (idim, hdim)):  # fc1 (w13), fc2 (w2)
        kh, steps = k // 2, k // 64
        kb_pad, rows_pad = -(-(k // 16) // 4) * 4, -(-nrows // 128) * 128
        e = int(rng.integers(0, 4))
        slab, sfslab = base + e * nrows * kh, (1 << 40) + e * rows_pad * kb_pad
        for r0 in sorted({0, 8 * int(rng.integers(0, nrows // 8)), nrows - 8}):
            rows = [slab + (r0 + g) * kh for g in range(8)]
            got = _pf_stream(la, steps, rows,
                             lambda sp: sfslab + int(ng.sf_offset(r0, 4 * sp, kb_pad)))
            lines = {a // 128 for _, a in got}
            want = {(a + o) // 128 for a in rows for o in range(0, kh, 32)}
            want |= {(sfslab + int(ng.sf_offset(r, 4 * sp + j, kb_pad))) // 128
                     for sp in range(steps) for r in range(r0, r0 + 8) for j in range(4)}
            assert lines == want
            for it, a in got:
                in_w = slab <= a < slab + nrows * kh
                assert in_w or sfslab <= a < sfslab + rows_pad * kb_pad
                sp = ((a - slab) % kh) // 32 if in_w else None
                assert sp is None or it < max(sp, 2)  # before the iteration loading it
    assert _pf_stream(0, 40, [base], lambda sp: base) == []


@pytest.mark.parametrize("hdim,idim,nx", [(2560, 640, 20), (2816, 704, 22), (6144, 2048, 64),
                                          (6144, 256, 8), (128, 64, 2)])
def test_pf_w2_prefetches_the_experts_w2_exactly_once(hdim, idim, nx):
    """pf_w2 as written: the nx fc1 CTAs of an expert (128 threads) cover
    every 128 B line of its w2 rows and w2 scales exactly once, nothing
    outside them."""
    for bytes_ in (hdim * idim // 2, -(-hdim // 128) * 128 * (-(-(idim // 16) // 4) * 4)):
        assert bytes_ % 128 == 0
        lines = bytes_ // 128
        chunk = -(-lines // nx)
        got = [ln for bx in range(nx) for tid in range(128)
               for ln in range(bx * chunk + tid, min((bx + 1) * chunk, lines), 128)]
        assert sorted(got) == list(range(lines))


def _warp_dot(aq, asf, w, wsf, e, rows_b, r0, r1, rs, ok0, ok1, oks, kdim, nrows_w):
    """One warp's fc1_dot / fc2_dot: A rows r0/r1 (lanes' rows g / g+8 of
    aq, masked by ok0/ok1), scale row rs (masked by oks), weight rows
    rows_b[i] of expert e -> one (32, 4) f32 accumulator per weight row set,
    3-stage register pipeline exactly as written."""
    kh, nkb = kdim // 2, kdim // 16
    kb_pad, rows_pad = -(-nkb // 4) * 4, -(-nrows_w // 128) * 128

    def la(k0):
        o = k0 // 2 + 4 * _LT
        return np.stack([_ld32(aq, r0 * kh + o, ok0), _ld32(aq, r1 * kh + o, ok1),
                         _ld32(aq, r0 * kh + o + 16, ok0), _ld32(aq, r1 * kh + o + 16, ok1),
                         _ld32(asf, rs * nkb + k0 // 16, oks)], 1)

    def lb(k0, row):
        o = k0 // 2 + 4 * _LT
        base = e * nrows_w * kh + row * kh
        on = np.ones(32, bool)
        sfo = e * rows_pad * kb_pad + ng.sf_offset(row, k0 // 16, kb_pad)
        return np.stack([_ld32(w, base + o, on), _ld32(w, base + o + 16, on),
                         _ld32(wsf, sfo, on)], 1)

    steps = kdim // 64
    ld = lambda k0: (la(k0), [lb(k0, r) for r in rows_b])
    acc = [np.zeros((32, 4), np.float32) for _ in rows_b]
    s0 = ld(0)
    s1 = ld(64) if steps > 1 else s0
    s = 2
    while s < steps:
        s2 = ld(s * 64)
        acc = [_mma(c, s0[0], b) for c, b in zip(acc, s0[1])]
        s0, s1 = s1, s2
        s += 1
    acc = [_mma(c, s0[0], b) for c, b in zip(acc, s0[1])]
    if steps > 1:
        acc = [_mma(c, s1[0], b) for c, b in zip(acc, s1[1])]
    return acc


def _gemm_emu(out, aq, asf, w, wsf, owner, se, so, sc, pl, a_row_div, kdim, ndim, nrows_w,
              col_off, epi, sink=None):
    """Shared fc1/fc2 CTA/warp/lane walk (col_off: fc1 gate rows = idim + j).
    owner[p] = local expert of pair p: a CTA may only write its own pairs
    (a masked-row slip into the next expert's rows is a race on silicon).
    sink (moe_fc1_quant): instead of storing rows of `out`, the 4 warps'
    epilogue values of each 16-row chunk land in a 16x32 tile (row g / g+8,
    col warp*8 + 2t + dc, masked rows included) handed to
    sink(e, off, cnt, r0, bx, tile) after the chunk (the kernel's barrier)."""
    sfa_row = 8 * (_LANE & 1) + _LANE // 4
    for slot, e in enumerate(se):
        if e < 0:
            continue
        off, cnt = so[slot], sc[slot]
        for bx in range(ndim // 32):
            for r0 in range(0, cnt, 16):
                tile = np.full((16, 32), np.nan, np.float32)
                for warp in range(4):
                    j0 = (bx * 4 + warp) * 8
                    if j0 >= ndim:
                        continue
                    rows_b = [j0 + _LG + co for co in col_off]
                    ok0, ok1, oks = r0 + _LG < cnt, r0 + _LG + 8 < cnt, r0 + sfa_row < cnt
                    p0 = np.where(ok0, pl[np.where(ok0, off + r0 + _LG, 0)], 0)
                    p1 = np.where(ok1, pl[np.where(ok1, off + r0 + _LG + 8, 0)], 0)
                    ps = np.where(oks, pl[np.where(oks, off + r0 + sfa_row, 0)], 0)
                    assert (p0[ok0] >= 0).all() and (p1[ok1] >= 0).all() and (ps[oks] >= 0).all()
                    acc = _warp_dot(aq, asf, w, wsf, e, rows_b, p0 // a_row_div, p1 // a_row_div,
                                    ps // a_row_div, ok0, ok1, oks, kdim, nrows_w)
                    col = j0 + 2 * _LT
                    for ok, pp, q, rr in ((ok0, p0, 0, _LG), (ok1, p1, 2, _LG + 8)):
                        for dc in range(2):
                            v = epi(e, pp, [c[:, q + dc] for c in acc])
                            if sink is not None:
                                tile[rr, warp * 8 + 2 * _LT + dc] = v
                                continue
                            assert (owner[pp[ok]] == e).all(), "wrote another expert's row"
                            out[pp[ok], col[ok] + dc] = v[ok]
                if sink is not None:
                    sink(e, off, cnt, r0, bx, tile)


def _fc1_quant_emu(p, aq, asf, loc, se, so, sc, pl, topk, hdim, idim):
    """moe_fc1_quant: per CTA chunk, 32 threads = (row tid/2, block tid%2)
    quantize the smem tile (quant16 == nm.quant_codes) into their pair's
    hq / hsf row; returns (hq [P, I/2], hsf [P, I/16]) with untouched rows
    left at the 0xAB sentinel."""
    pairs = len(loc)
    hq = np.full((pairs, idim // 2), 0xAB, np.uint8)
    hsf = np.full((pairs, idim // 16), 0xAB, np.uint8)
    u8 = lambda t: t.reshape(-1).view(torch.uint8).numpy()
    g1 = p["g1"].numpy()

    def epi1(e, pp, c):  # c = [up, gate]
        a = np.float32(g1[e])
        return _act32(a * c[1], p["act"]) * (a * c[0])

    def sink(e, off, cnt, r0, bx, tile):
        for tid in range(32):
            row, blk = tid // 2, tid % 2
            if r0 + row < cnt:
                pp = pl[off + r0 + row]
                assert loc[pp] == e, "quantized another expert's row"
                col = bx * 32 + blk * 16
                q, sf = nm.quant_codes(torch.from_numpy(tile[row:row + 1, blk * 16:blk * 16 + 16].copy()),
                                       p["a2g"])
                hq[pp, col // 2:col // 2 + 8] = q.numpy()[0]
                hsf[pp, col // 16] = sf.numpy()[0, 0]

    _gemm_emu(None, aq, asf, u8(p["w13"]), u8(p["w13_sf"]), loc, se, so, sc, pl, topk,
              hdim, idim, 2 * idim, (0, idim), epi1, sink)
    return hq, hsf


def _fc2_combine_emu(p, hq, hsf, loc, e_count, tw, m, topk, hdim, idim):
    """moe_fc2_combine: CTA (token, 8 H cols), warp k = top-k slot k; a
    local pair sits alone in mma row 0 (ok0 = g < 1, ok1 = 0, oks =
    sfa_row < 1, all rows of the fragment walk = that pair); lanes g == 0
    park y = c * (g2 * tw) in ys[k]; 8 threads sum ys over local k
    ascending (f32) -> bf16."""
    u8 = lambda t: t.reshape(-1).view(torch.uint8).numpy()
    w2, w2sf = u8(p["w2"]), u8(p["w2_sf"])
    g2, twf = p["g2"].numpy(), tw.reshape(-1).numpy().astype(np.float32)
    sfa_row = 8 * (_LANE & 1) + _LANE // 4
    ok0, ok1, oks = _LG < 1, np.zeros(32, bool), sfa_row < 1
    out = np.zeros((m, hdim), np.float32)
    local = lambda pp: 0 <= loc[pp] < e_count
    for tok in range(m):
        for bx in range(hdim // 8):
            h0 = bx * 8
            ys = np.full((topk, 8), np.nan, np.float32)
            for k in range(topk):
                pp = tok * topk + k
                if not local(pp):
                    continue
                e = loc[pp]
                rows = np.full(32, pp)
                c = _warp_dot(hq, hsf, w2, w2sf, e, [h0 + _LG], rows, rows, rows, ok0, ok1, oks,
                              idim, hdim)[0]
                s = np.float32(g2[e]) * twf[pp]
                for dc in range(2):
                    ys[k, 2 * _LT[ok0] + dc] = c[ok0, dc] * s
            for j in range(8):
                acc = np.float32(0.0)
                for k in range(topk):
                    if local(tok * topk + k):
                        acc = np.float32(acc + ys[k, j])
                out[tok, h0 + j] = acc
    return out


def kernel_lane_emu_fused(p, x, ids, tw, base=0):
    """The fused 3-launch plan: moe_route_quant (route_par + quant16 of x:
    the same codes as moe_quant_rows by construction), moe_fc1_quant,
    moe_fc2_combine. -> (hq, hsf, out f32 before the bf16 store)."""
    m, topk = ids.shape
    e_count, two_i, kh1 = p["w13"].shape
    hdim, idim = kh1 * 2, two_i // 2
    ids_flat = ids.reshape(-1).numpy()
    se, so, sc, pl = _route_par_emu(ids_flat, base, e_count, min(e_count, m * topk))[:4]
    aq_t, asf_t = nm.quant_codes(x.float(), p["a1g"])
    loc = ids_flat.astype(np.int64) - base
    hq, hsf = _fc1_quant_emu(p, aq_t.reshape(-1).numpy(), asf_t.reshape(-1).numpy(), loc, se, so,
                             sc, pl, topk, hdim, idim)
    out = _fc2_combine_emu(p, hq.reshape(-1), hsf.reshape(-1), loc, e_count, tw, m, topk, hdim, idim)
    return hq, hsf, out


def kernel_lane_emu_front(p, x, ids, tw, base=0):
    """The front 4-launch plan: moe_route_quant (route_par + x quant),
    moe_fc1_quant, expert-major moe_fc2 (y rows of LOCAL pairs only: hq rows
    of off-rank pairs are never written by moe_fc1_quant), moe_combine.
    -> (hq, hsf, out f32 before the bf16 store)."""
    m, topk = ids.shape
    e_count, two_i, kh1 = p["w13"].shape
    hdim, idim = kh1 * 2, two_i // 2
    ids_flat = ids.reshape(-1).numpy()
    se, so, sc, pl = _route_par_emu(ids_flat, base, e_count, min(e_count, m * topk))[:4]
    aq_t, asf_t = nm.quant_codes(x.float(), p["a1g"])
    loc = ids_flat.astype(np.int64) - base
    hq, hsf = _fc1_quant_emu(p, aq_t.reshape(-1).numpy(), asf_t.reshape(-1).numpy(), loc, se, so,
                             sc, pl, topk, hdim, idim)
    u8 = lambda t: t.reshape(-1).view(torch.uint8).numpy()
    g2, twf = p["g2"].numpy(), tw.reshape(-1).numpy().astype(np.float32)
    y = np.full((m * topk, hdim), np.nan, np.float32)
    _gemm_emu(y, hq.reshape(-1), hsf.reshape(-1), u8(p["w2"]), u8(p["w2_sf"]), loc, se, so, sc, pl,
              1, idim, hdim, hdim, (0,), lambda e, pp, c: c[0] * (np.float32(g2[e]) * twf[pp]))
    acc = np.zeros((m, hdim), np.float32)
    for k in range(topk):
        v = (loc[k::topk] >= 0) & (loc[k::topk] < e_count)
        acc[v] += y[k::topk][v]
    return hq, hsf, acc


def kernel_lane_emu(p, x, ids, tw, base=0):
    """(ws tuple as nm.workspace, out bf16) from the lane-level emulation."""
    m, topk = ids.shape
    e_count, two_i, kh1 = p["w13"].shape
    hdim, idim = kh1 * 2, two_i // 2
    pairs = m * topk
    slots = min(e_count, pairs)
    ids_flat = ids.reshape(-1).numpy()
    se, so, sc, pl = _route_emu(ids_flat, base, e_count, slots)
    aq_t, asf_t = nm.quant_codes(x.float(), p["a1g"])
    aq, asf = aq_t.reshape(-1).numpy(), asf_t.reshape(-1).numpy()
    u8 = lambda t: t.reshape(-1).view(torch.uint8).numpy()
    inter = np.full((pairs, idim), np.nan, np.float32)
    g1, g2 = p["g1"].numpy(), p["g2"].numpy()
    twf = tw.reshape(-1).numpy().astype(np.float32)

    def epi1(e, pp, c):  # c = [up, gate]
        a = np.float32(g1[e])
        return _act32(a * c[1], p["act"]) * (a * c[0])

    loc = ids_flat.astype(np.int64) - base
    _gemm_emu(inter, aq, asf, u8(p["w13"]), u8(p["w13_sf"]), loc, se, so, sc, pl, topk,
              hdim, idim, two_i, (0, idim), epi1)
    with np.errstate(invalid="ignore"):
        hq_t, hsf_t = nm.quant_codes(torch.from_numpy(inter), p["a2g"])
    hq, hsf = hq_t.reshape(-1).numpy(), hsf_t.reshape(-1).numpy()
    y = np.full((pairs, hdim), np.nan, np.float32)

    def epi2(e, pp, c):
        return c[0] * (np.float32(g2[e]) * twf[pp])

    _gemm_emu(y, hq, hsf, u8(p["w2"]), u8(p["w2_sf"]), loc, se, so, sc, pl, 1,
              idim, hdim, hdim, (0,), epi2)
    acc = np.zeros((m, hdim), np.float32)
    for k in range(topk):
        v = (loc[k::topk] >= 0) & (loc[k::topk] < e_count)
        acc[v] += y[k::topk][v]
    out = torch.from_numpy(acc).bfloat16()
    ws = nm.workspace("cpu", m, topk, hdim, idim, e_count)
    for t, v in zip(ws, (aq, asf, inter, hq, hsf, y)):
        t[:v.size] = torch.from_numpy(np.ascontiguousarray(v).reshape(-1))
    kernel_lane_emu.acc = acc  # f32 sum before the bf16 store (fused-order test)
    return ws, out


@pytest.mark.parametrize("act,m,local,e_global,topk,hdim,idim,dead", [
    ("gelu_tanh", 1, 4, 4, 2, 256, 192, None),   # fc1 4 steps, fc2 3 steps
    ("gelu_tanh", 17, 2, 2, 2, 256, 64, None),   # 17 rows/expert: 2 chunks, fc2 1 step
    ("silu", 33, 3, 3, 3, 128, 128, None),       # odd M, odd E, 2-step both
    ("gelu_tanh", 64, 8, 8, 4, 128, 192, None),  # 32 rows/expert avg, masked tails
    ("silu", 9, 4, 16, 3, 128, 64, None),        # EP rank 2/4: t%3==1 off-rank
    ("gelu_tanh", 5, 4, 16, 2, 256, 64, True),   # EP: every token off-rank
    ("silu", 23, 2, 8, 2, 192, 128, None),       # EP rank 3/4, 3-step fc1
])
def test_lane_emulator_matches_f64_spec(act, m, local, e_global, topk, hdim, idim, dead):
    ep_rank = 2 if e_global == 16 else (3 if e_global == 8 and local == 2 else 0)
    base = ep_rank * local
    p = nm.make_problem(local, hdim, idim, act, seed=m)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local,
                              dead=dead)
    x = (torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)) * 1.3).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw, base)
    lid = nm.to_local(ids, base, local)
    ok, msg = nm.stages(p, x, lid, tw, ws, out)
    assert ok, msg
    ref = nm.moe_ref(p, x, lid, tw)
    assert torch.isfinite(out.float()).all()
    dead_rows = (lid < 0).all(1)
    assert (out[dead_rows] == 0).all()
    if dead:
        assert dead_rows.all()
    else:
        assert nm._rel(out.float(), ref) <= nm.REF_TOL, nm._rel(out.float(), ref)


def test_act_ex2_form_is_finite_and_signed_right_at_extremes():
    x = np.array([-3e38, -1e13, -200, -60, -11, -5, -1e-30, -0.0, 0.0, 1e-30, 5, 11, 60,
                  1e13, 3e38], np.float32)
    for kind in (0, 1):
        got = _act32(x, kind)
        ref = nm.act_ref(torch.from_numpy(x).double(), kind).numpy()
        assert np.isfinite(got).all(), (kind, got)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-30)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 47, 63, 64])
def test_lane_emulator_m_sweep_ep(m):
    """M sweep on an EP rank (5 local of 15 global, id_base 5, int64 ids):
    chunk tails at every residue, off-rank rows exact 0, all stages in tol."""
    local, e_global, base, topk = 5, 15, 5, 3
    p = nm.make_problem(local, 128, 128, "gelu_tanh" if m % 2 else "silu", seed=100 + m)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local)
    ids = ids.long()
    x = torch.randn(m, 128, generator=torch.Generator().manual_seed(m)).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw, base)
    lid = nm.to_local(ids, base, local)
    ok, msg = nm.stages(p, x, lid, tw, ws, out)
    assert ok, msg
    assert (out[(lid < 0).all(1)] == 0).all() and torch.isfinite(out.float()).all()
    assert nm._rel(out.float(), nm.moe_ref(p, x, lid, tw)) <= nm.REF_TOL


@pytest.mark.parametrize("m", [1, 2, 5, 8, 10, 16, 32, 64])
def test_fused_plan_is_bit_identical_to_legacy(m):
    """Fused 3-launch plan (route_par, moe_fc1_quant's in-CTA h quant from
    the smem tile, moe_fc2_combine's token-major fc2 + k-ascending smem sum)
    and front 4-launch plan (route_par, moe_fc1_quant, moe_fc2, moe_combine)
    == the legacy 6-launch plan (route_cta) BIT FOR BIT on an EP rank (5 local of 10 global,
    top-4, rows t % 3 == 1 routed only off-rank, int64 ids; >= 3 local
    terms per token somewhere, so the k order is observable): the f32 sum,
    the bf16 output and every local pair's hq/hsf row. Then `stages` on the
    workspace run_plans leaves (legacy inter + y, fused hq/hsf/out) passes."""
    local, e_global, base, topk, hdim, idim = 5, 10, 5, 4, 128, 128
    p = nm.make_problem(local, hdim, idim, "silu" if m % 2 else "gelu_tanh", seed=300 + m)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=10 + m, base=base, local=local)
    ids = ids.long()
    x = torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw, base)
    acc = kernel_lane_emu.acc
    hq, hsf, facc = kernel_lane_emu_fused(p, x, ids, tw, base)
    lid = nm.to_local(ids, base, local)
    vp = (lid.reshape(-1) >= 0).numpy()
    assert vp.any() and (~vp).any()
    if m >= 8:
        assert ((lid >= 0).sum(1) >= 3).any()
    pairs = m * topk
    assert np.array_equal(facc.view(np.uint32), acc.view(np.uint32))
    fhq, fhsf, front = kernel_lane_emu_front(p, x, ids, tw, base)
    assert np.array_equal(front.view(np.uint32), acc.view(np.uint32))
    assert np.array_equal(fhq, hq) and np.array_equal(fhsf, hsf)
    assert torch.equal(torch.from_numpy(facc).bfloat16().view(torch.int16), out.view(torch.int16))
    assert (out[(lid < 0).all(1)] == 0).all()
    leg_hq = ws[3][:pairs * idim // 2].view(pairs, -1).numpy()
    leg_hsf = ws[4][:pairs * idim // 16].view(pairs, -1).numpy()
    assert np.array_equal(hq[vp], leg_hq[vp]) and np.array_equal(hsf[vp], leg_hsf[vp])
    assert (hq[~vp] == 0xAB).all() and (hsf[~vp] == 0xAB).all()  # off-rank rows never written
    mixed = list(ws)
    mixed[3] = ws[3].clone()
    mixed[4] = ws[4].clone()
    mixed[3][:hq.size] = torch.from_numpy(hq.reshape(-1))
    mixed[4][:hsf.size] = torch.from_numpy(hsf.reshape(-1))
    ok, msg = nm.stages(p, x, lid, tw, mixed, torch.from_numpy(facc).bfloat16())
    assert ok, msg


def test_run_plans_flags_any_plan_that_differs():
    """run_plans (oracle glue): identical plans pass; one flipped bit in a
    fused plan's out / local hq row fails; off-rank hq rows are ignored."""
    local, topk, hdim, idim, m = 2, 2, 64, 64, 3
    lid = torch.tensor([[0, -1], [-1, -1], [1, 0]])
    ws = list(nm.workspace("cpu", m, topk, hdim, idim, local))
    for t in ws:
        t.zero_()
    out = torch.zeros(m, hdim, dtype=torch.bfloat16)

    def run(mode, poke=None):
        ws[3][2 * idim // 2] = 7 if mode == 2 and poke == "offrank" else 0  # pair 2 row
        ws[3][0] = 1 if mode == 2 and poke == "hq" else 0  # pair 0 = local
        o = out.clone()
        if mode == 1 and poke == "out":
            o[2, 5] = 1.0
        return o

    lens = ((hdim // 2, hdim // 16), (idim // 2, idim // 16))
    assert nm.run_plans(run, ws, lid, *lens)[1]
    assert nm.run_plans(lambda md: run(md, "offrank"), ws, lid, *lens)[1]
    for poke, needle in (("hq", "fused3.hq"), ("out", "front4.out")):
        _, ok, msg = nm.run_plans(lambda md: run(md, poke), ws, lid, *lens)
        assert not ok and needle in msg, msg


def test_launch_mode(monkeypatch):
    monkeypatch.delenv(nm.FUSED_ENV, raising=False)
    assert nm.fused_on()
    assert [nm.launch_mode(m) for m in (1, nm.FUSED_MAX_M, nm.FUSED_MAX_M + 1)] == [2, 2, 1]
    monkeypatch.setenv(nm.FUSED_ENV, "0")
    assert not nm.fused_on() and nm.launch_mode(1, nm.fused_on()) == 0


def test_tunables_only_steer_prefetches():
    """Static half of the "every tune is bit-identical" proof (the lane
    emulator is the numeric half): in the kernel source the lookahead /
    pf2 knobs (kernel param `la`, dot param `pf`, `pf2`) only reach
    prefetch helpers, and those helpers only compute addresses and issue
    `prefetch.global.L2` — no load, no store, no other asm. So a tune value
    cannot change any loaded byte or any f32 op."""
    src = (ROOT / "kernels-oxide" / "nvfp4_moe" / "src" / "main.rs").read_text()
    allowed = [r"\s*(la|pf|pf2): u32,", r"\s*pf_head\(la, .*\);", r"\s*if la == 0 \{",
               r"\s*let n0 = if 2 \+ la < steps \{ 2 \+ la \} else \{ steps \};",
               r"\s*if pf != 0 && s \+ pf < steps \{", r"\s*pf_step\(s \+ pf, false, .*\);",
               r"\s*if pf2 != 0 \{", r".*fc[12]_dot\(.*",
               r"\s*(kb_pad, hdim, )?lane, la\);", r"\s*la\);",
               r"\s*fn pf_head\(la: u32, .*"]
    for ln in src.splitlines():
        code = ln.split("//")[0]
        if re.search(r"\b(pf|pf2)\b|\bla\b(?!\()", code) and "let la = |" not in code:
            assert any(re.fullmatch(a, code.rstrip()) for a in allowed), ln
    for fn in ("pf_l2", "pf_step", "pf_head", "pf_w2"):
        body = re.search(rf"fn {fn}\(.*?\n    \}}\n", src, re.S).group(0)
        # no dereference: every `*` left is a pointer type or a product
        assert "*" not in body.replace("*const", "").replace("*mut", "").replace(" * ", ""), fn
        asm = re.findall(r'ptx_asm!\(\s*"([^"]*)"', body)
        assert asm in ([], ["prefetch.global.L2 [%0];"]), (fn, asm)


def test_tune_table_parse_select_and_env(monkeypatch):
    t = nm.parse_tune(nm.DEFAULT_TUNE)
    assert nm.format_tune(t) == nm.DEFAULT_TUNE and t[-1][0] >= 64
    tab = nm.parse_tune("8:fused3/255/255/0, 16:legacy6/8/0/1,64:1/16/255/0")
    assert tab == [(8, 2, 255, 255, 0), (16, 0, 8, 0, 1), (64, 1, 16, 255, 0)]
    pick = lambda m: nm.tune_for(m, tab)
    assert pick(1) == pick(8) == (2, 255 | 255 << 8)
    assert pick(9) == (0, 8 | 1 << 16) and pick(64) == pick(200) == (1, 16 | 255 << 8)
    assert nm.tune_for(5, tab, fused=False) == (0, 0)  # SUFFIX_MOE_FUSED=0: verbatim legacy
    for bad in ("", "8:fused4/0/0/0", "8:fused3/256/0/0", "8:fused3/0/0/2", "16:front4/0/0/0,8:front4/0/0/0",
                "0:front4/0/0/0", "8:front4/0/0"):
        with pytest.raises(ValueError):
            nm.parse_tune(bad)
    monkeypatch.setenv(nm.TUNE_ENV, "64:legacy6/0/0/0")
    assert nm.tune_for(3) == (0, 0)
    monkeypatch.delenv(nm.TUNE_ENV)
    assert nm.tune_table() == t


def test_tune_word_matches_host_decode():
    host = (ROOT / "src" / "nvfp4_moe_oxide.rs").read_text()
    assert ("let (la1, la2, pf2, skip) = (tune & 0xFF, (tune >> 8) & 0xFF, (tune >> 16) & 1, "
            "tune >> 24);") in host
    assert "if tune & 0xC0FE_0000 != 0 {" in host  # bits 17-23, 30-31 refused
    w = nm.tune_word(7, 200, 1, 0x3F)
    assert (w & 0xFF, w >> 8 & 0xFF, w >> 16 & 1, w >> 24) == (7, 200, 1, 0x3F)
    assert w & 0xC0FE0000 == 0 and nm.ORACLE_TUNE == 0x1FFFF
    assert ".filter(|(n, _)| skip >> n & 1 == 0)" in host
    # fc1 kernels get (la1, pf2, w2, w2_sf) last, fc2 kernels la2 last
    for arr, tail in (("fc1_args", "Arg::U32(la1),\n        Arg::U32(pf2),\n        Arg::Ptr(w2.ptr),"
                                   "\n        Arg::Ptr(w2s.ptr),\n    ];"),
                      ("fq_args", "Arg::U32(la1),\n        Arg::U32(pf2),\n        Arg::Ptr(w2.ptr),"
                                  "\n        Arg::Ptr(w2s.ptr),\n    ];"),
                      ("fc2_args", "Arg::U32(la2),\n    ];"), ("fc_args", "Arg::U32(la2),\n    ];")):
        assert re.search(rf"let {arr} = \[.*?\];", host, re.S).group(0).endswith(tail), arr


def test_sweep_verdict_merges_buckets_and_picks_max_m():
    f3, f4 = (2, 255, 255, 0), (1, 16, 255, 1)
    rows = [(1, f3, 10, 40), (8, f3, 50, 120), (16, f4, 150, 190), (24, f4, 250, 240),
            (32, f4, 280, 300), (64, (0, 0, 0, 0), 600, 500)]
    table, mx, lose = nm.sweep_verdict(rows)
    assert table == "8:fused3/255/255/0,32:front4/16/255/1,64:legacy6/0/0/0"
    assert mx == 32 and lose == [24]
    assert nm.parse_tune(table)  # paste-able
    assert nm.sweep_verdict([(1, f3, 50, 40)])[1:] == (None, [])
    cands = nm.sweep_candidates()
    assert (0, 0, 0, 0) in cands and len(set(cands)) == len(cands) == 60
