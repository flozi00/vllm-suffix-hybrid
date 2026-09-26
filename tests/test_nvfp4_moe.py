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


def _gemm_emu(out, aq, asf, w, wsf, owner, se, so, sc, pl, a_row_div, kdim, ndim, nrows_w,
              col_off, epi):
    """Shared fc1/fc2 CTA/warp/lane walk (col_off: fc1 gate rows = idim + j).
    owner[p] = local expert of pair p: a CTA may only write its own pairs
    (a masked-row slip into the next expert's rows is a race on silicon)."""
    kh, nkb = kdim // 2, kdim // 16
    kb_pad, rows_pad = -(-nkb // 4) * 4, -(-nrows_w // 128) * 128
    sfa_row = 8 * (_LANE & 1) + _LANE // 4
    for slot, e in enumerate(se):
        if e < 0:
            continue
        off, cnt = so[slot], sc[slot]
        for bx in range(ndim // 32):
            for warp in range(4):
                j0 = (bx * 4 + warp) * 8
                if j0 >= ndim:
                    continue
                rows_b = [j0 + _LG + co for co in col_off]
                for r0 in range(0, cnt, 16):
                    ok0, ok1, oks = r0 + _LG < cnt, r0 + _LG + 8 < cnt, r0 + sfa_row < cnt
                    p0 = np.where(ok0, pl[np.where(ok0, off + r0 + _LG, 0)], 0)
                    p1 = np.where(ok1, pl[np.where(ok1, off + r0 + _LG + 8, 0)], 0)
                    ps = np.where(oks, pl[np.where(oks, off + r0 + sfa_row, 0)], 0)
                    assert (p0[ok0] >= 0).all() and (p1[ok1] >= 0).all() and (ps[oks] >= 0).all()

                    def la(k0):
                        o = k0 // 2 + 4 * _LT
                        return np.stack([_ld32(aq, (p0 // a_row_div) * kh + o, ok0),
                                         _ld32(aq, (p1 // a_row_div) * kh + o, ok1),
                                         _ld32(aq, (p0 // a_row_div) * kh + o + 16, ok0),
                                         _ld32(aq, (p1 // a_row_div) * kh + o + 16, ok1),
                                         _ld32(asf, (ps // a_row_div) * nkb + k0 // 16, oks)], 1)

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
                    col = j0 + 2 * _LT
                    for ok, pp, q in ((ok0, p0, 0), (ok1, p1, 2)):
                        for dc in range(2):
                            assert (owner[pp[ok]] == e).all(), "wrote another expert's row"
                            v = epi(e, pp, [c[:, q + dc] for c in acc])
                            out[pp[ok], col[ok] + dc] = v[ok]


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
