# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scaled routed-experts decode kernel: CPU contract tests (quant
spec vs a literal transcription of vLLM's kernels, UE8M0 rounding, weight
layout / gate-up order vs dense math, lane-level emulation of the kernel's
m16n8k32 fragments + 16 B K permutation vs the f64 spec, EP zero contract,
eligibility, scale resolution, layer info, shared OOT arming, verdict,
host<->kernel ABI). Silicon: `python -m suffix_hybrid.kernels.fp8_moe
oracle|bench` + the per-layer load oracle."""
import configparser
import pathlib
import re
import sys
import types

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import fp8_moe as fm
from suffix_hybrid.kernels import nvfp4_moe as nm

ROOT = pathlib.Path(__file__).resolve().parents[1]
KSRC = (ROOT / "kernels-oxide" / "fp8_moe" / "src" / "main.rs").read_text()
HOST = (ROOT / "src" / "fp8_moe_oxide.rs").read_text()


# ---------------------------------------------------------------------------
# quant spec
# ---------------------------------------------------------------------------
def _vllm_group_quant(x, eps, smin, ue8m0):
    """Literal transcription of vLLM v0.30 per_token_group_quant.cu
    (ComputeGroupScale/QuantizeGroup) and fused_silu_mul_block_quant.cu,
    element by element in f32."""
    xs = x.numpy().astype(np.float32)
    r, k = xs.shape
    q = np.zeros((r, k), np.float32)
    s_out = np.zeros((r, k // 128), np.float32)
    for i in range(r):
        for g in range(k // 128):
            blk = xs[i, g * 128:(g + 1) * 128]
            amax = np.float32(max(np.float32(eps), np.abs(blk).max()))
            s = np.float32(amax / np.float32(448.0))
            s = np.float32(max(s, np.float32(smin)))
            if ue8m0:
                s = np.float32(2.0 ** np.ceil(np.log2(np.float64(max(s, np.float32(1e-10))))))
            s_out[i, g] = s
            q[i, g * 128:(g + 1) * 128] = np.clip(blk / s, -448, 448)
    q8 = torch.from_numpy(q).to(torch.float8_e4m3fn).view(torch.uint8)
    return q8, torch.from_numpy(s_out)


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("stage", ["x", "h"])
def test_quant_matches_vllm_kernels(ue8m0, stage):
    x = torch.randn(5, 384) * torch.tensor([1e-3, 1.0, 30.0]).repeat_interleave(128)
    x[2, 128:256] = 0.0  # all-zero group: eps / floor branch
    x[3, :128] = 1e-14   # tiny group: eps vs 1/(448*512) floor differ
    q, s = fm.quant_fp8(x, ue8m0, stage)
    eps, smin = fm.quant_params(ue8m0, stage)
    q_ref, s_ref = _vllm_group_quant(x, eps, smin, ue8m0)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)
    if stage == "h" and not ue8m0:
        assert float(s[3, 0]) == pytest.approx(1 / (448 * 512))  # fused silu floor
    else:
        assert float(s[2, 1]) == pytest.approx((1e-10 / 448) if not ue8m0 else 2.0 ** -33)


def test_quant_params_constants_match_host():
    assert fm.quant_params(False, "x") == (1e-10, 0.0)
    assert fm.quant_params(True, "h") == (1e-10, 0.0)
    assert fm.quant_params(False, "h") == (0.0, 1 / (448 * 512))
    assert "const EPS: f32 = 1e-10;" in HOST
    assert "const MIN_SCALE: f32 = 1.0 / (448.0 * 512.0);" in HOST
    assert "if ue8m0 { (EPS, 0.0) } else { (0.0, MIN_SCALE) }" in HOST


def test_pow2_ceil_exact_and_kernel_bit_form():
    s = torch.tensor([1.0, 1.0000001, 0.75, 2 ** -33, 3e-10, 448.0, 1e-10], dtype=torch.float32)
    got = fm.pow2_ceil(s)
    want = torch.tensor([2.0 ** np.ceil(np.log2(float(v))) for v in s.double()], dtype=torch.float32)
    assert torch.equal(got, want)
    # kernel: bits with a zero mantissa are already powers of two, else
    # (exponent bits) + one exponent step
    b = s.view(torch.int32)
    kern = torch.where((b & 0x7FFFFF) == 0, b, (b & 0x7F800000) + 0x800000).view(torch.float32)
    assert torch.equal(kern, got)
    assert "(b & 0x7f80_0000) + 0x80_0000" in KSRC


def test_quant_roundtrip_error_bound():
    x = torch.randn(4, 256) * 3
    rel = (fm.qdq(x) - x).abs() / x.abs().amax(-1, keepdim=True)
    assert float(rel.max()) <= 2 ** -4 * 1.01  # e4m3: half-ulp <= 2^-4 of the group max


# ---------------------------------------------------------------------------
# reference math / layout
# ---------------------------------------------------------------------------
def test_reference_approximates_dense_bf16_moe_gate_first():
    """Quantized spec vs unquantized math with w13 = [gate; up] (vLLM
    silu_and_mul: first half is the gate). Swapped halves fail hard."""
    e_count, hdim, idim, topk, m = 4, 256, 128, 2, 6
    p = fm.make_problem(e_count, hdim, idim, seed=7)
    ids, tw = nm.rand_routing(m, e_count, topk, "cpu", seed=1)
    x = torch.randn(m, hdim).bfloat16()

    def dense(swap):
        out = torch.zeros(m, hdim, dtype=torch.float64)
        for t in range(m):
            for k in range(topk):
                e = int(ids[t, k])
                gu = x[t].double() @ fm.expert_w(p["w13"], p["w13_s"], e).double().T
                g, u = (gu[idim:], gu[:idim]) if swap else (gu[:idim], gu[idim:])
                out[t] += tw[t, k] * (fm.silu(g) * u @ fm.expert_w(p["w2"], p["w2_s"], e).double().T)
        return out

    got = fm.moe_ref(p, x, ids, tw)
    assert nm._rel(got, dense(False)) < 0.06  # two e4m3 activation quants
    assert nm._rel(got, dense(True)) > 0.5


def test_block_scales_are_per_128x128_block():
    w = torch.randn(256, 384)
    w[:128, 128:256] *= 100  # one hot block must not leak into its neighbours
    q, s = fm.block_quant(w)
    assert s.shape == (2, 3) and float(s[0, 1]) > 50 * float(s[1, 1])
    wd = fm.expert_w(q[None], s[None], 0)
    assert nm._rel(wd, w) < 0.03


def test_ue8m0_problem_has_power_of_two_scales():
    p = fm.make_problem(2, 128, 128, seed=3, ue8m0=True)
    for s in (p["w13_s"], p["w2_s"]):
        assert torch.equal(fm.pow2_ceil(s), s)


def test_vllm_emulation_is_close_but_not_the_spec():
    p = fm.make_problem(8, 512, 256, seed=9)
    ids, tw = nm.rand_routing(8, 8, 4, "cpu", seed=9)
    x = torch.randn(8, 512).bfloat16()
    ref = fm.moe_ref(p, x, ids, tw)
    emu = fm.moe_ref(p, x, ids, tw, vllm=True).bfloat16()
    r = nm._rel(emu, ref)
    assert 1e-3 < r < 2e-2, r  # bf16 fc1 output + bf16 y: vLLM's own distance


def test_oracle_gate():
    assert fm.oracle_ok(1.5e-3, 6e-3, 5e-3)
    assert not fm.oracle_ok(4e-3, 6e-3, 5e-3)  # beyond REF_TOL
    assert not fm.oracle_ok(1e-3, 3e-2, 5e-3)  # disagree beyond both errors
    assert fm.oracle_ok(2e-3, 9e-3, 1e-3)  # floor 1e-2


def test_judge_enforces_offrank_zero_contract():
    ref = torch.randn(3, 8)
    lid = torch.tensor([[0, 1], [-1, -1], [2, -1]])
    ref[1] = 0
    assert fm.judge(ref.clone(), ref.clone(), ref, lid)[0]
    bad = ref.clone()
    bad[1, 0] = 1e-6
    assert not fm.judge(bad, ref.clone(), ref, lid)[0]
    assert not fm.judge(ref.clone(), bad, ref, lid)[0]


# ---------------------------------------------------------------------------
# Lane-level emulator of kernels-oxide/fp8_moe moe_fc1 / moe_fc2: per lane
# (g, t) the 32-byte loads at kb*128 + 32t (ld32b), mma i on words (2i,
# 2i+1) exactly as block_dot, the PTX m16n8k32 8-bit fragment layout (A: a0
# row g k 4t.., a1 row g+8 k 4t.., a2 row g k 16+4t.., a3 row g+8 k 16+4t..;
# B: b0 k 4t.. col g, b1 k 16+4t.. col g; C rows g/g+8 cols 2t,2t+1), the
# per-block rescale (dot * a_s * w_s), silu / topk_w epilogues, masked rows,
# route from nvfp4's twin (same kernel source) and the fixed-order combine.
# Workspace starts NaN so a wrong lane->row map poisons the output.
# ---------------------------------------------------------------------------
_LANE = np.arange(32)
_LG, _LT = _LANE // 4, _LANE % 4
_F8 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double().numpy()


def _words(buf, off, ok):
    """ld32b: 8 little-endian u32 words from bytes [off, off + 32) per lane."""
    off = np.where(ok, off, 0).astype(np.int64)
    assert (off[ok] % 16 == 0).all() and (off[ok] + 32 <= buf.size).all(), "misaligned/OOB"
    w = np.stack([sum(buf[off + 4 * j + i].astype(np.uint64) << np.uint64(8 * i)
                      for i in range(4)) for j in range(8)], 1)
    return np.where(ok[:, None], w, 0).astype(np.uint64)


def _bytes(v):
    return [((v >> np.uint64(8 * j)) & np.uint64(0xFF)).astype(np.int64) for j in range(4)]


def _mma(c, a, b):
    """c (32,4) f32; a = [a0..a3] (32,) u32 each; b = [b0, b1] -> (32,4)."""
    A, B = np.zeros((16, 32)), np.zeros((8, 32))
    for r in range(4):
        for j, byte in enumerate(_bytes(a[r])):
            A[_LG + 8 * (r & 1), 4 * _LT + j + 16 * (r >> 1)] = _F8[byte]
    for r in range(2):
        for j, byte in enumerate(_bytes(b[r])):
            B[_LG, 4 * _LT + j + 16 * r] = _F8[byte]
    cm = np.zeros((16, 8))
    for q in range(4):
        cm[_LG + 8 * (q >> 1), 2 * _LT + (q & 1)] = c[:, q]
    d = (A @ B.T + cm).astype(np.float32)
    return np.stack([d[_LG + 8 * (q >> 1), 2 * _LT + (q & 1)] for q in range(4)], 1)


def _block_dot(x0, x1, b):
    c = np.zeros((32, 4), np.float32)
    for i in range(4):
        c = _mma(c, [x0[:, 2 * i], x1[:, 2 * i], x0[:, 2 * i + 1], x1[:, 2 * i + 1]],
                 [b[:, 2 * i], b[:, 2 * i + 1]])
    return c


def _warp_dot(aq, a_s, w, w_s, e, col_offs, j0, t0, t1, ok0, ok1, kdim, nrows_w):
    """One warp's fc1_dot / fc2_dot: A rows t0/t1 (rows g / g+8, masked by
    ok0/ok1), per 128-K block rescale acc += dot * a_s * w_s, weight rows
    co + j0 + g of expert e for each co in col_offs -> one (32, 4) f32
    accumulator per co."""
    nkb = kdim // 128
    s_rows = nrows_w // 128
    accs = [np.zeros((32, 4), np.float32) for _ in col_offs]
    for kb in range(nkb):
        lo = kb * 128 + 32 * _LT
        x0 = _words(aq, t0 * kdim + lo, ok0)
        x1 = _words(aq, t1 * kdim + lo, ok1)
        sa0 = np.where(ok0, a_s[np.where(ok0, t0 * nkb + kb, 0)], 0).astype(np.float32)
        sa1 = np.where(ok1, a_s[np.where(ok1, t1 * nkb + kb, 0)], 0).astype(np.float32)
        for ci, co in enumerate(col_offs):
            row = co + j0 + _LG
            b = _words(w, e * nrows_w * kdim + row * kdim + lo, np.ones(32, bool))
            sw = np.float32(w_s[e * s_rows * nkb + ((co + j0) // 128) * nkb + kb])
            blk = _block_dot(x0, x1, b)
            sa = np.stack([sa0, sa0, sa1, sa1], 1)
            accs[ci] = (accs[ci] + blk * sa * sw).astype(np.float32)
    return accs


def _gemm_emu(out, aq, a_s, w, w_s, owner, se, so, sc, pl, a_row_div, kdim, ndim, nrows_w,
              col_offs, epi, warps=4, sink=None):
    """fc1 (col_offs (0, I): gate, up) / fc2 (col_offs (0,)) CTA/warp/lane
    walk, CTA = `warps` x 8 columns. sink (moe_fc1_quant, warps=16): the
    warps' epilogue values of each 16-row chunk land in a 16 x (8*warps)
    tile (row g / g+8, col warp*8 + 2t + dc) handed to sink(e, off, cnt,
    r0, bx, tile) after the chunk (the kernel's barrier)."""
    for slot, e in enumerate(se):
        if e < 0:
            continue
        off, cnt = so[slot], sc[slot]
        for bx in range(ndim // (8 * warps)):
            for r0 in range(0, cnt, 16):
                tile = np.full((16, 8 * warps), np.nan, np.float32)
                for warp in range(warps):
                    j0 = (bx * warps + warp) * 8
                    if j0 >= ndim:
                        continue
                    ok0, ok1 = r0 + _LG < cnt, r0 + _LG + 8 < cnt
                    p0 = np.where(ok0, pl[np.where(ok0, off + r0 + _LG, 0)], 0)
                    p1 = np.where(ok1, pl[np.where(ok1, off + r0 + _LG + 8, 0)], 0)
                    accs = _warp_dot(aq, a_s, w, w_s, e, col_offs, j0, p0 // a_row_div,
                                     p1 // a_row_div, ok0, ok1, kdim, nrows_w)
                    col = j0 + 2 * _LT
                    for ok, pp, q, rr in ((ok0, p0, 0, _LG), (ok1, p1, 2, _LG + 8)):
                        for dc in range(2):
                            v = epi(pp, [a[:, q + dc] for a in accs])
                            if sink is not None:
                                tile[rr, warp * 8 + 2 * _LT + dc] = v
                                continue
                            assert (owner[pp[ok]] == e).all(), "wrote another expert's row"
                            out[pp[ok], col[ok] + dc] = v[ok]
                if sink is not None:
                    sink(e, off, cnt, r0, bx, tile)


def _silu32(x):
    x = x.astype(np.float32)
    with np.errstate(over="ignore"):
        return (x / (np.float32(1) + np.exp2(-x * np.float32(1.442695)))).astype(np.float32)


def kernel_lane_emu(p, x, ids, tw, base=0):
    """(ws tuple as fm.workspace, out bf16) from the lane-level emulation."""
    m, topk = ids.shape
    e_count, two_i, hdim = p["w13"].shape
    idim, pairs = two_i // 2, m * topk
    ids_flat = ids.reshape(-1).numpy()
    se, so, sc, pl = nm.route_twin(ids, e_count, base)
    pl = np.asarray(pl + [-7], np.int64)  # sentinel: never read
    loc = ids_flat.astype(np.int64) - base
    u8 = lambda t: t.reshape(-1).contiguous().view(torch.uint8).numpy()
    aq_t, as_t = fm.quant_fp8(x.float(), p["ue8m0"])
    inter = np.full((pairs, idim), np.nan, np.float32)
    _gemm_emu(inter, u8(aq_t), as_t.reshape(-1).numpy(), u8(p["w13"]), p["w13_s"].reshape(-1).numpy(),
              loc, se, so, sc, pl, topk, hdim, idim, two_i, (0, idim),
              lambda pp, c: _silu32(c[0]) * c[1])
    with np.errstate(invalid="ignore"):
        hq_t, hs_t = fm.quant_fp8(torch.from_numpy(inter), p["ue8m0"], "h")
    y = np.full((pairs, hdim), np.nan, np.float32)
    twf = tw.reshape(-1).numpy().astype(np.float32)
    _gemm_emu(y, u8(hq_t), hs_t.reshape(-1).numpy(), u8(p["w2"]), p["w2_s"].reshape(-1).numpy(),
              loc, se, so, sc, pl, 1, idim, hdim, hdim, (0,), lambda pp, c: c[0] * twf[pp])
    acc = np.zeros((m, hdim), np.float32)
    for k in range(topk):
        v = (loc[k::topk] >= 0) & (loc[k::topk] < e_count)
        acc[v] += y[k::topk][v]
    ws = fm.workspace("cpu", m, topk, hdim, idim, e_count)
    for t, v in zip(ws, (u8(aq_t), as_t, inter, u8(hq_t), hs_t, y)):
        v = torch.as_tensor(np.ascontiguousarray(v)).reshape(-1)
        t[:v.numel()] = v
    kernel_lane_emu.acc = acc  # f32 sum before the bf16 store (fused-order test)
    return ws, torch.from_numpy(acc).bfloat16()


def kernel_lane_emu_fused(p, x, ids, tw, base=0):
    """The fused 3-launch plan: moe_route_quant (route + quant_group of x:
    moe_quant_rows' code), moe_fc1_quant (CTA = 128 cols of I = 16 warps;
    warp w quantizes chunk row w of the 16x128 smem tile, quant_group ==
    fm.quant_fp8 "h"), moe_fc2_combine (token-major, pair alone in mma row
    0, k-ascending f32 sum). -> (hq [P, I], h_s [P, I/128], out f32)."""
    m, topk = ids.shape
    e_count, two_i, hdim = p["w13"].shape
    idim, pairs = two_i // 2, m * topk
    ids_flat = ids.reshape(-1).numpy()
    se, so, sc, pl = nm.route_twin(ids, e_count, base)
    pl = np.asarray(pl + [-7], np.int64)
    loc = ids_flat.astype(np.int64) - base
    u8 = lambda t: t.reshape(-1).contiguous().view(torch.uint8).numpy()
    aq_t, as_t = fm.quant_fp8(x.float(), p["ue8m0"])
    hq = np.full((pairs, idim), 0xAB, np.uint8)
    hs = np.full((pairs, idim // 128), np.nan, np.float32)

    def sink(e, off, cnt, r0, bx, tile):
        for w in range(16):
            if r0 + w < cnt:
                pp = pl[off + r0 + w]
                assert loc[pp] == e, "quantized another expert's row"
                q, s_ = fm.quant_fp8(torch.from_numpy(tile[w:w + 1].copy()), p["ue8m0"], "h")
                hq[pp, bx * 128:(bx + 1) * 128] = q.numpy()[0]
                hs[pp, bx] = s_.numpy()[0, 0]

    _gemm_emu(None, u8(aq_t), as_t.reshape(-1).numpy(), u8(p["w13"]), p["w13_s"].reshape(-1).numpy(),
              loc, se, so, sc, pl, topk, hdim, idim, two_i, (0, idim),
              lambda pp, c: _silu32(c[0]) * c[1], warps=16, sink=sink)
    w2, w2_s = u8(p["w2"]), p["w2_s"].reshape(-1).numpy()
    twf = tw.reshape(-1).numpy().astype(np.float32)
    ok0, ok1 = _LG < 1, np.zeros(32, bool)
    local = lambda pp: 0 <= loc[pp] < e_count
    out = np.zeros((m, hdim), np.float32)
    for tok in range(m):
        for bx in range(hdim // 8):
            h0 = bx * 8
            ys = np.full((topk, 8), np.nan, np.float32)
            for k in range(topk):
                pp = tok * topk + k
                if not local(pp):
                    continue
                rows = np.full(32, pp)
                c = _warp_dot(hq.reshape(-1), hs.reshape(-1), w2, w2_s, loc[pp], (0,), h0, rows, rows,
                              ok0, ok1, idim, hdim)[0]
                for dc in range(2):
                    ys[k, 2 * _LT[ok0] + dc] = c[ok0, dc] * twf[pp]
            for j in range(8):
                acc = np.float32(0.0)
                for k in range(topk):
                    if local(tok * topk + k):
                        acc = np.float32(acc + ys[k, j])
                out[tok, h0 + j] = acc
    return hq, hs, out


@pytest.mark.parametrize("m,local,e_global,topk,hdim,idim,ue8m0,dead", [
    (1, 4, 4, 2, 256, 128, False, None),     # 1 row: 15 masked mma rows
    (17, 2, 2, 2, 128, 256, False, None),    # 17 rows/expert: 2 chunks
    (9, 4, 16, 3, 256, 128, True, None),     # EP rank 2/4, UE8M0 scales
    (5, 4, 16, 2, 128, 128, False, True),    # EP: every token off-rank
    (23, 3, 6, 3, 384, 256, False, None),    # EP rank 1/2, 3-block fc1, 2-block fc2
])
def test_lane_emulator_matches_f64_spec(m, local, e_global, topk, hdim, idim, ue8m0, dead):
    ep_rank = {16: 2, 6: 1}.get(e_global, 0)
    base = ep_rank * local
    p = fm.make_problem(local, hdim, idim, seed=m, ue8m0=ue8m0)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local, dead=dead)
    x = torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw, base)
    lid = nm.to_local(ids, base, local)
    ok, msg = fm.stages(p, x, lid, tw, ws, out)
    assert ok, msg
    dead_rows = (lid < 0).all(1)
    assert (out[dead_rows] == 0).all() and torch.isfinite(out.float()).all()
    if dead:
        assert dead_rows.all()
    else:
        r = nm._rel(out.float(), fm.moe_ref(p, x, lid, tw))
        assert r <= fm.REF_TOL, r


def test_stages_pin_drift_to_the_stage_that_made_it():
    p = fm.make_problem(4, 256, 128, seed=4)
    ids, tw = nm.rand_routing(6, 4, 2, "cpu", seed=4)
    x = torch.randn(6, 256).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw)
    assert fm.stages(p, x, ids, tw, ws, out)[0]
    for idx, stage, bump in [(2, "fc1", lambda t: t.mul_(1 + 2 ** -11)),
                             (5, "fc2", lambda t: t.mul_(1 + 1e-3)),
                             (1, "xq", lambda t: t[:1].mul_(2.0)),
                             (4, "hq", lambda t: t[:1].mul_(2.0))]:
        bad = list(ws)
        bad[idx] = ws[idx].clone()
        bump(bad[idx])
        ok, msg = fm.stages(p, x, ids, tw, bad, out)
        assert not ok and msg.split(f" {stage}=")[1].split()[0] != "0.0e+00", msg


@pytest.mark.parametrize("m", [1, 2, 5, 8, 16, 32])
def test_fused_plan_is_bit_identical_to_legacy(m):
    """Fused 3-launch plan == legacy 6-launch plan BIT FOR BIT (f32 sum,
    bf16 out, every local pair's hq / h_s row) on an EP rank (4 local of 8
    global, top-4, rows t % 3 == 1 off-rank, 2-block fc1, 2-block fc2,
    UE8M0 on odd M); `stages` on run_plans' mixed workspace passes."""
    local, e_global, base, topk, hdim, idim = 4, 8, 4, 4, 256, 256
    p = fm.make_problem(local, hdim, idim, seed=500 + m, ue8m0=bool(m % 2))
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=10 + m, base=base, local=local)
    x = torch.randn(m, hdim, generator=torch.Generator().manual_seed(m)).bfloat16()
    ws, out = kernel_lane_emu(p, x, ids, tw, base)
    acc = kernel_lane_emu.acc
    hq, hs, facc = kernel_lane_emu_fused(p, x, ids, tw, base)
    lid = nm.to_local(ids, base, local)
    vp = (lid.reshape(-1) >= 0).numpy()
    assert vp.any() and (~vp).any()
    if m >= 8:
        assert ((lid >= 0).sum(1) >= 3).any()  # k order observable
    pairs = m * topk
    assert np.array_equal(facc.view(np.uint32), acc.view(np.uint32))
    assert torch.equal(torch.from_numpy(facc).bfloat16().view(torch.int16), out.view(torch.int16))
    assert (out[(lid < 0).all(1)] == 0).all()
    leg_hq = ws[3][:pairs * idim].view(pairs, -1).numpy()
    leg_hs = ws[4][:pairs * idim // 128].view(pairs, -1).numpy()
    assert np.array_equal(hq[vp], leg_hq[vp])
    assert np.array_equal(hs[vp].view(np.uint32), leg_hs[vp].view(np.uint32))
    assert (hq[~vp] == 0xAB).all()  # off-rank rows never written
    mixed = list(ws)
    mixed[3], mixed[4] = ws[3].clone(), ws[4].clone()
    mixed[3][:hq.size] = torch.from_numpy(hq.reshape(-1))
    mixed[4][:hs.size] = torch.from_numpy(hs.reshape(-1))
    ok, msg = fm.stages(p, x, lid, tw, mixed, torch.from_numpy(facc).bfloat16())
    assert ok, msg


def test_k_permutation_is_a_bijection_per_block():
    """Lane t word w (bytes 32t + 4w..) -> (mma i = w // 2, logical k =
    4t + 16 (w % 2) ..): every physical k of a 128-block lands on exactly
    one (mma, logical k), identically for A and B."""
    seen = set()
    for t in range(4):
        for w in range(8):
            for j in range(4):
                seen.add((w // 2, 4 * t + 16 * (w % 2) + j))
    assert len(seen) == 128 and {i for i, _ in seen} == {0, 1, 2, 3}
    assert "c = mma(c, r0[2 * i], r1[2 * i], r0[2 * i + 1], r1[2 * i + 1], b[2 * i], b[2 * i + 1]);" in KSRC


# ---------------------------------------------------------------------------
# vLLM wiring (pure parts)
# ---------------------------------------------------------------------------
GOOD = dict(quant_dtype="fp8", block_shape=[128, 128], backend="DEEPGEMM", act="silu",
            clamp_limit=None, swiglu_alpha=None, router_weight_on_input=False, bias=False,
            tp=1, ep=2, dp=1, all2all=False, eplb=False, mk_shared_overlap=False,
            expert_map=True, ep_base=256, E=256, H=2560, I=640, w_exact=True, scales=None)


@pytest.mark.parametrize("change,needle", [
    ({}, None),
    ({"backend": "TRITON"}, None),
    ({"ep": 1, "expert_map": False, "ep_base": 0, "E": 256}, None),
    ({"tp": 2, "ep": 1, "expert_map": False, "ep_base": 0, "I": 256}, None),  # TP shard
    ({"quant_dtype": "nvfp4"}, "not FP8"),
    ({"block_shape": [64, 128]}, "block_shape"),
    ({"block_shape": []}, "block_shape"),  # per-tensor FP8
    ({"backend": "FLASHINFER_CUTLASS"}, "backend"),
    ({"backend": "MARLIN"}, "backend"),
    ({"act": "gelu"}, "activation"),
    ({"clamp_limit": 7.0}, "clamp"),
    ({"router_weight_on_input": True}, "router_weight"),
    ({"bias": True}, "biases"),
    ({"dp": 2}, "DP/all2all"),
    ({"all2all": True}, "DP/all2all"),
    ({"eplb": True}, "EPLB"),
    ({"mk_shared_overlap": True}, "shared experts"),
    ({"tp": 2}, "not both"),
    ({"ep_base": None}, "linear"),
    ({"I": 704}, "shape"),
    ({"E": 512}, "shape"),
    ({"w_exact": False}, "contiguous e4m3"),
    ({"scales": "weight scales x"}, "weight scales"),
])
def test_eligibility(change, needle):
    why = fm.eligibility({**GOOD, **change})
    assert (why is None) if needle is None else (needle in why), why


def test_resolve_scale_prefers_logical_post_then_pre():
    shape = (4, 10, 20)
    pre = torch.rand(shape)
    post = pre.permute(0, 2, 1).contiguous().permute(0, 2, 1)  # MN-major, same values
    t, why = fm.resolve_scale(post, pre, shape)
    assert why is None and t.is_contiguous() and torch.equal(t, pre)
    packed = torch.zeros(4, 20, 3, dtype=torch.int32)  # DeepGEMM UE8M0 pack
    t, why = fm.resolve_scale(packed, pre, shape)
    assert why is None and t is pre
    t, why = fm.resolve_scale(packed, None, shape)
    assert t is None and "not resolvable" in why


def _fake_layer(ep_rank=1, e_local=4, e_global=8, packed=False):
    f8 = torch.float8_e4m3fn
    h, i = 256, 128
    w13 = torch.zeros(e_local, 2 * i, h, dtype=f8)
    w2 = torch.zeros(e_local, h, i, dtype=f8)
    s13 = torch.rand(e_local, 2 * i // 128, h // 128)
    s2 = torch.rand(e_local, h // 128, i // 128)
    em = torch.full((e_global,), -1, dtype=torch.int32)
    em[ep_rank * e_local:(ep_rank + 1) * e_local] = torch.arange(e_local, dtype=torch.int32)
    qc = types.SimpleNamespace(quant_dtype=f8, block_shape=[128, 128], gemm1_clamp_limit=None,
                               gemm1_alpha=None, w1_bias=None,
                               w1_scale=torch.zeros(1, dtype=torch.int32) if packed else s13,
                               w2_scale=torch.zeros(1, dtype=torch.int32) if packed else s2)
    qm = types.SimpleNamespace(moe_quant_config=qc,
                               fp8_backend=types.SimpleNamespace(name="DEEPGEMM"))
    pc = types.SimpleNamespace(ep_rank=ep_rank, use_all2all_kernels=False, enable_eplb=False)
    mc = types.SimpleNamespace(tp_size=1, ep_size=e_global // e_local, dp_size=1,
                               moe_parallel_config=pc)
    layer = types.SimpleNamespace(quant_method=qm, moe_config=mc, w13_weight=w13, w2_weight=w2,
                                  activation=types.SimpleNamespace(value="silu"),
                                  apply_router_weight_on_input=False, expert_map=em,
                                  global_num_experts=e_global, top_k=2, layer_name="mtp.x")
    return layer, {"w13": s13, "w2": s2}


@pytest.mark.parametrize("packed", [False, True])
def test_layer_info_on_a_vllm_shaped_layer(packed):
    layer, pre = _fake_layer(packed=packed)
    info, sc = fm.layer_info(layer, pre)
    assert fm.eligibility(info) is None, info
    assert info["ep_base"] == 4 and (info["E"], info["H"], info["I"]) == (4, 256, 128)
    assert torch.equal(sc[0], pre["w13"]) and torch.equal(sc[1], pre["w2"])
    assert fm.is_fp8(layer)
    state = {"seen": 0, "ours": 0, "stock": {}, "oracle": []}
    layer.quant_method.fp8_backend.name = "MARLIN"
    assert fm.prepare(layer, pre, None, state) is None
    assert state["seen"] == 1 and list(state["stock"]) == [fm.eligibility(fm.layer_info(layer, pre)[0])]


def test_verdict_fails_loud():
    assert fm.verdict(1, 1, {}) is None
    assert "no FP8 block-quantized" in fm.verdict(0, 0, {})
    msg = fm.verdict(1, 0, {"Fp8 MoE backend MARLIN": 1})
    assert "NOT ENGAGED" in msg and "MARLIN" in msg


def test_workspace_sizes_and_growth():
    ws = fm.workspace("cpu", 32, 10, 2560, 640, 256)
    p = 320
    assert [t.numel() for t in ws] == [32 * 2560, 32 * 20, p * 640, p * 640, p * 5,
                                       p * 2560, 3 * 256 + p]
    assert fm.workspace("cpu", 1, 10, 2560, 640, 256, ws) is ws


def test_max_m_knob(monkeypatch):
    monkeypatch.delenv(fm.MAX_M_ENV, raising=False)
    assert fm.max_m() == 32
    monkeypatch.setenv(fm.MAX_M_ENV, "0")
    with pytest.raises(ValueError):
        fm.max_m()


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(fm.GATE, raising=False)
    before = set(sys.modules)
    assert fm.register() is None
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_both_gates_share_one_registration(monkeypatch):
    """SUFFIX_FP8_MOE=1 re-uses nvfp4_moe's armed OOT class: once armed
    (by either gate) the other register() returns the same state without
    touching vLLM, so register_oot runs exactly once."""
    monkeypatch.setenv(fm.GATE, "1")
    monkeypatch.setenv(nm.GATE, "1")
    monkeypatch.setitem(nm._state, "armed", True)
    before = set(sys.modules)
    assert fm.register() is nm._state and nm.register() is nm._state
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)
    src = (ROOT / "suffix_hybrid" / "kernels" / "nvfp4_moe.py").read_text()
    assert src.count("register_oot(") == 1


def test_entry_point():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    assert cfg["vllm.general_plugins"]["suffix_fp8_moe"] == "suffix_hybrid.kernels.fp8_moe:register"


def test_boot_gates_registered():
    src = (ROOT / "sitecustomize.py").read_text()
    assert '"fp8_moe_oracle": (["-m", "suffix_hybrid.kernels.fp8_moe", "oracle"], {})' in src
    assert '"fp8_moe_bench": (["-m", "suffix_hybrid.kernels.fp8_moe", "bench"], {})' in src


def test_host_launch_args_match_kernel_abi():
    params = {}
    for name, sig in re.findall(r"pub unsafe fn (moe_\w+)\((.*?)\)\s*\{", KSRC, re.S):
        params[name] = len([a for a in sig.split(",") if a.strip()])
    assert set(params) == {"moe_route", "moe_quant_rows", "moe_fc1", "moe_fc2", "moe_combine",
                           "moe_route_quant", "moe_fc1_quant", "moe_fc2_combine"}
    for arr, kern in [("route_args", "moe_route"), ("qx_args", "moe_quant_rows"),
                      ("qh_args", "moe_quant_rows"), ("fc1_args", "moe_fc1"),
                      ("fc2_args", "moe_fc2"), ("comb_args", "moe_combine"),
                      ("rq_args", "moe_route_quant"), ("fq_args", "moe_fc1_quant"),
                      ("fc_args", "moe_fc2_combine")]:
        body = re.search(rf"let {arr} = \[(.*?)\];", HOST, re.S).group(1)
        n = len([a for a in re.split(r",\s*\n", body) if a.strip()])
        assert n == params[kern], (arr, n, params[kern])
        assert f'("{kern}", ' in HOST and f"&{arr})" in HOST
    assert '"arch": "sm_120a"' in (ROOT / "kernels-oxide" / "fp8_moe" / "oxide-variants.json").read_text()
    assert 'name = "fp8_moe"' in (ROOT / "kernels-oxide" / "fp8_moe" / "Cargo.toml").read_text()
    # fused fc1: 16 warps (whole 128-col h quant group), 16x128 f32 smem tile
    assert '"moe_fc1_quant", ((i / 128) as u32, slots as u32), 512, 16 * 128 * 4,' in HOST
    assert "#[launch_bounds(512)]\n    pub unsafe fn moe_fc1_quant(" in KSRC
    assert "32 * nw, (k * 8 * 4) as u32" in HOST


def test_route_and_combine_are_nvfp4s_verified_kernels():
    """moe_route / moe_combine / moe_fc2_combine's k-ascending sum and the
    route_cta macro are byte-identical to nvfp4_moe's (silicon-verified
    there), so nm.route_twin is their spec here too."""
    nv = (ROOT / "kernels-oxide" / "nvfp4_moe" / "src" / "main.rs").read_text()
    body = lambda s, n: re.search(rf"pub unsafe fn {n}\(.*?\n    \}}\n", s, re.S).group(0)
    for n in ("moe_route", "moe_combine"):
        assert body(KSRC, n) == body(nv, n), n
    macro = lambda s: re.search(r"macro_rules! route_cta \{.*?\n\}\n", s, re.S).group(0)
    assert macro(KSRC) == macro(nv)
    tail = lambda s: re.search(r"        thread::sync_threads\(\);\n        if tid < 8 \{.*?\n    \}\n",
                               s[s.index("pub unsafe fn moe_fc2_combine("):], re.S).group(0)
    assert tail(KSRC) == tail(nv)


def test_first_forward_hook_decides_each_gated_family(monkeypatch):
    monkeypatch.setenv(fm.GATE, "1")
    monkeypatch.delenv(nm.GATE, raising=False)
    monkeypatch.setitem(nm._state, "hook", None)
    for fp8, raises in (({"seen": 0, "ours": 0, "stock": {}}, "no FP8 block-quantized"),
                        ({"seen": 1, "ours": 0, "stock": {"x": 1}}, "every FP8 MoE layer"),
                        ({"seen": 1, "ours": 1, "stock": {}}, None)):
        monkeypatch.setitem(nm._state, "checked", False)
        monkeypatch.setitem(nm._state, "fp8", {**fp8, "oracle": []})
        if raises:
            with pytest.raises(RuntimeError, match=raises):
                nm._first_forward_check(None, ())
        else:
            nm._first_forward_check(None, ())
        assert nm._state["checked"]
    # both gates: an NVFP4 family still undecided ("") keeps waiting
    monkeypatch.setenv(nm.GATE, "1")
    monkeypatch.setitem(nm._state, "checked", False)
    monkeypatch.setitem(nm._state, "instances", 3)
    monkeypatch.setitem(nm._state, "layers_ours", 0)
    monkeypatch.setitem(nm._state, "stock", {})
    nm._first_forward_check(None, ())
    assert not nm._state["checked"]
