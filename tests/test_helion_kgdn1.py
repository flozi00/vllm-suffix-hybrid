# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 Helion port (suffix_hybrid/helion_kernels/kgdn1.py): lab ABI, cases,
the torch references vs the lane's oracle reference, backend routing, and —
with Helion installed — the decode and MTP-verify bodies in interpret mode on
CPU (every TP rank shape of the Qwen3.5 / Qwen3-Next layouts, the interleaved
layout) plus Triton codegen (the frontend proof; speed and SASS are the lab
gate's job)."""
import json
import os
import pathlib
import re
import subprocess
import sys

os.environ.setdefault("HELION_INTERPRET", "1")  # before any helion import

import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import kgdn1
from suffix_hybrid.kernels import qwen_gdn

ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN27 = {"name": "qwen3.8-27b-fable-distill-nvfp4", "architecture": "Qwen3_5ForConditionalGeneration",
          "hidden": 5120, "tp": 1}


def _args(T, H, HV, K, seed=0, idx=None, act=0):
    args = list(kgdn1.make(kgdn1.case(T, H, HV, K, K), seed, device="cpu"))
    if idx is not None:
        args[10].copy_(torch.tensor(idx, dtype=torch.int32))
    args[-1] = act
    return args


# ---------------------------------------------------------------------------
# lab ABI + cases
# ---------------------------------------------------------------------------
def test_kernels_abi():
    dec, ver, dec_vt, ver_vt = kgdn1.KERNELS
    for vt, full in ((dec_vt, dec), (ver_vt, ver)):  # same cases / reference / baseline / check
        assert {k: v for k, v in vt.items() if k not in ("name", "body", "settings")} == {
            k: v for k, v in full.items() if k not in ("name", "body")}
    for ent, name, body in ((dec, kgdn1.KERNEL, "kgdn1_decode"),
                            (ver, kgdn1.VERIFY_KERNEL, "kgdn1_verify"),
                            (dec_vt, kgdn1.KERNEL_VT, "kgdn1_decode_vt"),
                            (ver_vt, kgdn1.VERIFY_KERNEL_VT, "kgdn1_verify_vt")):
        assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"]) and ent["name"] == name
        assert ent["body"].__qualname__ == body  # module-level, undecorated
        for key in ("cases", "make", "make_baseline", "reference", "baseline", "check"):
            assert callable(ent[key])
        pats = ent["replaces"]
        assert 0 < len(pats) <= 16 and all(len(p) <= 300 for p in pats)
        assert not any(re.search(p, "fused_add_rms_norm_kernel", re.I) for p in pats)
    for name in ("kgdn1_decode_h16_hv48_k128",
                 "fused_recurrent_gated_delta_rule_packed_decode_kernel",
                 "layer_norm_fwd_kernel"):
        assert any(re.search(p, name, re.I) for p in dec["replaces"]), name
    for name in ("void gdn_decode_post_conv_mtp_kernel<float, 3, false>",
                 "fused_sigmoid_gating_delta_rule_update_kernel"):
        assert any(re.search(p, name, re.I) for p in ver["replaces"]), name


def test_cases_per_rank_layouts():
    got = kgdn1.cases(QWEN27)
    assert [c["numtokens"] for c in got] == list(kgdn1.BUCKETS) and len(got) <= 64
    assert {(c["h"], c["hv"], c["k"], c["v"]) for c in got} == {(16, 48, 128, 128)}
    assert {(c["h"], c["hv"]) for c in kgdn1.cases(dict(QWEN27, tp=2))} == {(8, 24)}
    nxt = {"architecture": "Qwen3NextForCausalLM", "hidden": 2048, "tp": 4}
    assert {(c["h"], c["hv"]) for c in kgdn1.cases(nxt)} == {(4, 8)}
    assert kgdn1.cases(dict(QWEN27, tp=3)) == []                  # heads do not split
    assert kgdn1.cases(dict(QWEN27, hidden=4096)) == []           # unknown size: no guess
    assert kgdn1.cases({"architecture": "LlamaForCausalLM", "hidden": 5120, "tp": 1}) == []
    json.dumps(got)  # flat JSON scalars
    # no tp in the dict: every TP of 1/2/4/8 whose heads divide
    assert sorted({(c["h"], c["hv"]) for c in kgdn1.cases(dict(QWEN27, tp=None))}) == [
        (2, 6), (4, 12), (8, 24), (16, 48)]


# Qwen4Exp (vllm/models/qwen4_exp): dims come from config.json's linear_* fields
# (no GDN_DIMS row). The head counts here are test values, not the checkpoint's.
QWEN4EXP = {"name": "qwen4-exp", "architecture": "Qwen4ExpForConditionalGeneration",
            "hidden": 2560, "tp": 2, "spec_k": 4, "linear_num_key_heads": 16,
            "linear_num_value_heads": 32, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128}


def test_cases_from_linear_fields():
    assert {(c["h"], c["hv"], c["k"], c["v"]) for c in kgdn1.cases(QWEN4EXP)} == {
        (8, 16, 128, 128)}
    tps = {(c["h"], c["hv"]) for c in kgdn1.cases(dict(QWEN4EXP, tp=None))}
    assert tps == {(16, 32), (8, 16), (4, 8), (2, 4)}
    odd = dict(QWEN4EXP, tp=None, linear_num_key_heads=6, linear_num_value_heads=12)
    assert {(c["h"], c["hv"]) for c in kgdn1.cases(odd)} == {(6, 12), (3, 6)}  # 4, 8 skipped
    # fields win over the table; K != V passes through
    kv = dict(QWEN27, linear_num_key_heads=16, linear_num_value_heads=32,
              linear_key_head_dim=128, linear_value_head_dim=64)
    assert {(c["h"], c["hv"], c["k"], c["v"]) for c in kgdn1.cases(kv)} == {(16, 32, 128, 64)}


def test_verify_cases():
    got = kgdn1.verify_cases(QWEN4EXP)
    assert [c["numtokens"] for c in got] == [m * 5 for m in kgdn1.BUCKETS]
    assert {(c["h"], c["hv"], c["w"]) for c in got} == {(8, 16, 5)}
    assert kgdn1.verify_cases(dict(QWEN4EXP, spec_k=0)) == []
    assert kgdn1.verify_cases(dict(QWEN27)) == []          # no spec_k: no verify cases
    assert hk.case_key(got[0]) == hk.case_key(kgdn1.verify_case(1, 8, 16, 128, 128, 5))
    json.dumps(got)


# ---------------------------------------------------------------------------
# reference == the lane's oracle reference (gdn_decode_fused_torch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("T,H,HV,K,idx", [(1, 1, 1, 8, None), (4, 2, 6, 16, [3, 0, 1, 4]),
                                          (2, 16, 48, 128, None)])
def test_reference_matches_lane_reference(T, H, HV, K, idx, act):
    args = _args(T, H, HV, K, seed=T + K, idx=idx, act=act)
    lane = list(kgdn1.problem(kgdn1.case(T, H, HV, K, K), T + K, device="cpu"))
    if idx is not None:
        lane[7].copy_(torch.tensor(idx, dtype=torch.int32))
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    ref_out, ref_state = kgdn1.reference(*args)
    for a, b in zip(args, before):  # reference must not mutate its args
        assert not isinstance(a, torch.Tensor) or torch.equal(a, b)
    st = lane[6].clone()
    want = qwen_gdn.gdn_decode_fused_torch(*lane[:6], st, lane[7], H, K ** -0.5, 1e-6, act)
    torch.testing.assert_close(ref_out.float().view(T, HV, K),
                               want.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_state.reshape(st.shape), st, rtol=2e-6, atol=2e-6)
    ok, msg = kgdn1.check((ref_out, ref_state), (ref_out, ref_state))
    assert ok, msg
    worse = ref_state.clone()
    worse.view(-1)[0] += 1e-3
    assert not kgdn1.check((ref_out, worse), (ref_out, ref_state))[0]


def test_body_args_are_views():
    lane = kgdn1.problem(kgdn1.case(3, 2, 6, 16, 16), 0, device="cpu")
    q, k, v, b, a, z, a_log, dt, w, state, idx, out = kgdn1.body_args(*lane[:10])
    assert q.data_ptr() == lane[0].data_ptr() and z.data_ptr() == lane[1].data_ptr()
    assert state.data_ptr() == lane[6].data_ptr() and state.stride(0) == lane[6].stride(0)
    assert out.data_ptr() == lane[8].data_ptr() and a.data_ptr() == b.data_ptr() + 6 * 2
    assert v.shape == (3, 2, 3, 16) and state.shape == (4, 2, 3, 16, 16)


# ---------------------------------------------------------------------------
# backend knob routing (no GPU: config table + platform stubbed)
# ---------------------------------------------------------------------------
class _Native:
    def __init__(self):
        self.calls = []

    def gdn_decode_fused_cuda(self, *args):
        self.calls.append(args[0].shape[0])


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path))
    monkeypatch.setattr(hk, "_TABLES", {})
    monkeypatch.setattr(hk, "platform", lambda device=0: "testgpu")

    def ship(*buckets):
        (tmp_path / kgdn1.KERNEL).mkdir(exist_ok=True)
        rows = [{"key": hk.case_key(kgdn1.case(m, 16, 48, 128, 128)), "config": {"num_warps": 8}}
                for m in buckets]
        (tmp_path / kgdn1.KERNEL / "testgpu.json").write_text(json.dumps(rows))
    return ship


def test_oxide_default_never_imports_helion():
    code = ("import sys\nfrom suffix_hybrid.kernels import qwen_gdn\n"
            "class N:\n    gdn_decode_fused_cuda = print\n"
            "op, b = qwen_gdn.decode_op(N, 16, 48, 128)\n"
            "assert op is print and b == ()\n"
            "assert 'suffix_hybrid.helion_kernels.kgdn1' not in sys.modules\n"
            "assert 'helion' not in sys.modules\n")
    # No SUFFIX_* from other tests (a leaked SUFFIX_HYBRID_WRAP=1 arms sitecustomize's wrap).
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)


def test_helion_without_config_fails_at_load(table, monkeypatch):
    monkeypatch.setenv(kgdn1.ENV, "helion")
    table(1, 8, 32)  # 128 missing
    with pytest.raises(RuntimeError, match="no tuned kgdn1_decode config"):
        qwen_gdn.decode_op(_Native(), 16, 48, 128)


def test_auto_routes_per_bucket(table, monkeypatch):
    monkeypatch.setenv(kgdn1.ENV, "auto")
    native = _Native()
    op, buckets = qwen_gdn.decode_op(native, 16, 48, 128)
    assert op == native.gdn_decode_fused_cuda and buckets == ()  # nothing shipped: oxide
    monkeypatch.setattr(hk, "_TABLES", {})
    table(8)
    helion_calls = []
    monkeypatch.setattr(kgdn1, "gdn_decode_fused_helion",
                        lambda *a: helion_calls.append(a[0].shape[0]))
    op, buckets = qwen_gdn.decode_op(native, 16, 48, 128)
    assert buckets == (8,)
    for T in (1, 5, 8, 9):
        x = torch.empty(T, 1)
        op(x, *([None] * 8), 16, 0.1, 1e-6, 0, 0)
    assert helion_calls == [5, 8] and native.calls == [1, 9]


def test_int32_pool_guard():
    m = torch.device("meta")
    state = torch.empty(3000, 48, 128, 128, device=m)  # slot offsets > 2^31
    with pytest.raises(RuntimeError, match="2\\^31"):
        kgdn1.gdn_decode_fused_helion(torch.empty(1, 10240, device=m), None, None, None, None,
                                      None, state, None, None, 16, 0.1, 1e-6, 0)


# ---------------------------------------------------------------------------
# Helion: interpret mode (eager torch) + Triton codegen of the body
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("T,H,HV,K,idx", [(1, 1, 1, 8, None), (5, 2, 6, 16, [3, 0, -1, 1, 5])])
def test_body_interpret_matches_reference(T, H, HV, K, idx, act):
    helion = pytest.importorskip("helion")
    args = _args(T, H, HV, K, seed=3, idx=idx, act=act)
    ref = kgdn1.reference(*args)
    got = helion.kernel(kgdn1.kgdn1_decode)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    if idx is not None:
        for t, s in enumerate(idx):
            if s <= 0:
                assert torch.count_nonzero(got[0][t]) == 0


@pytest.mark.parametrize("static", [True, False])
def test_body_lowers_to_triton(static):
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    args = _args(3, 16, 48, 128)
    bound = helion.kernel(kgdn1.kgdn1_decode, static_shapes=static,
                          ref_mode=RefMode.OFF).bind(tuple(args))
    code = bound.to_triton_code(bound.config_spec.default_config())
    assert code.count("tl.load(state +") == 1 and code.count("tl.store(state +") == 1  # one pass


# ---------------------------------------------------------------------------
# TP-divided rank shapes, interleaved layout, MTP verify
# ---------------------------------------------------------------------------
def _rank(H, HV, tp):
    return H // tp, HV // tp


RANKS = [(*_rank(H, HV, tp), tp) for H, HV in ((16, 48), (16, 32)) for tp in (1, 2, 4, 8)]


def _lane_decode(T, H, HV, K, seed, idx=None, interleaved=False, act=0):
    inp = qwen_gdn.make_inputs(T, H, HV, K, slots=T + 2, seed=seed, idx=idx, page_pad=8,
                               interleaved=interleaved)
    out = torch.zeros(T, HV, K, dtype=torch.bfloat16)
    lane = [inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], out]
    return lane, [*kgdn1.body_args(*lane, H), K ** -0.5, 1e-6, act]


@pytest.mark.parametrize("H,HV,tp", RANKS)
def test_body_interpret_tp_rank_shapes(H, HV, tp):
    helion = pytest.importorskip("helion")
    lane, args = _lane_decode(3, H, HV, 128, seed=tp, idx=[2, 0, 1])
    ref = kgdn1.reference(*args)
    want_state = lane[6].clone()
    want = qwen_gdn.gdn_decode_fused_torch(*lane[:6], want_state, lane[7], H, 128 ** -0.5,
                                           1e-6, 0)
    torch.testing.assert_close(ref[0].float().view(3, HV, 128),
                               want.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
    got = helion.kernel(kgdn1.kgdn1_decode)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    torch.testing.assert_close(lane[6], want_state, rtol=1e-5, atol=1e-6)  # in place, padded pages


def _fix_qkv_ordering(mixed_qkvz, mixed_ba, H, G, K, V):
    """Transcription of vLLM's fix_query_key_value_ordering + forward_cuda's
    rearrange/cat (qwen_gdn_linear_attn.py, 0.30.0 == 0.32.0), per rank."""
    T = mixed_qkvz.shape[0]
    x = mixed_qkvz.view(T, H, 2 * K + 2 * G * V)
    y = mixed_ba.view(T, H, 2 * G)
    q, k, v, z = torch.split(x, [K, K, G * V, G * V], dim=2)
    b, a = torch.split(y, [G, G], dim=2)
    v = v.reshape(T, -1, V)
    z = z.reshape(T, -1, V)
    mixed_qkv = torch.cat((q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)), dim=-1)
    return mixed_qkv, z, b.reshape(T, H * G), a.reshape(T, H * G)


@pytest.mark.parametrize("H,G,K,V", [(2, 3, 16, 16), (4, 2, 8, 16), (1, 1, 8, 8)])
def test_deinterleave_matches_vllm_ordering(H, G, K, V):
    T = 3
    qkvz = torch.randn(T, H * (2 * K + 2 * G * V)).to(torch.bfloat16)
    ba = torch.randn(T, 2 * H * G).to(torch.bfloat16)
    qkv, z, ba_p = qwen_gdn.deinterleave(qkvz, ba, H, G, K, V)
    want_qkv, want_z, b, a = _fix_qkv_ordering(qkvz, ba, H, G, K, V)
    assert torch.equal(qkv, want_qkv) and qkv.is_contiguous()
    assert torch.equal(z.reshape(T, H * G, V), want_z)
    assert z.data_ptr() == qkvz.data_ptr() + (2 * K + G * V) * 2  # a view, no copy
    assert torch.equal(ba_p[:, :H * G], b) and torch.equal(ba_p[:, H * G:], a)


@pytest.mark.parametrize("H,HV,tp", [RANKS[1], RANKS[6]])
def test_body_interpret_interleaved(H, HV, tp):
    helion = pytest.importorskip("helion")
    lane, args = _lane_decode(4, H, HV, 128, seed=11, idx=[3, 1, 0, 5], interleaved=True)
    assert lane[1].dim() == 4 and args[5].data_ptr() == lane[1].data_ptr()  # z: strided view
    want_state = lane[6].clone()
    want = qwen_gdn.gdn_decode_fused_torch(*lane[:6], want_state, lane[7], H, 128 ** -0.5,
                                           1e-6, 0)
    ref = kgdn1.reference(*args)
    got = helion.kernel(kgdn1.kgdn1_decode)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    torch.testing.assert_close(got[0].float().view(4, HV, 128),
                               want.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)


def _verify_inputs(N, W, H, HV, K, seed=0, **kw):
    inp = qwen_gdn.make_verify_inputs(N, W, H, HV, K, seed=seed, page_pad=8, **kw)
    T = inp["mixed_qkv"].shape[0]
    out = torch.zeros(T, HV, K, dtype=torch.bfloat16)
    return [inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], inp["cu"], inp["acc"], out]


def _verify_by_decode_steps(lane, H, act=0):
    """Independent answer: vLLM's spec semantics as consecutive single-token
    steps of the lane's decode reference (gdn_decode_fused_torch)."""
    qkv, z, ba, a_log, dt, w, state, idx, cu, acc, out = lane
    N, W = idx.shape
    HV, V, K = state.shape[1], state.shape[2], state.shape[3]
    want_state, want_out = state.clone(), out.clone().float()
    for n in range(N):
        bos, tn, na = int(cu[n]), int(cu[n + 1] - cu[n]), int(acc[n])
        src = int(idx[n, na - 1]) if 1 <= na <= W else 0
        if src <= 0:
            want_out[bos:bos + tn] = 0
            continue
        tmp = torch.stack([state[0], state[src]]).clone()
        for i in range(tn):
            r = slice(bos + i, bos + i + 1)
            zr = z[r] if z.dim() == 3 else z[r].reshape(1, HV, V)
            want_out[r] = qwen_gdn.gdn_decode_fused_torch(
                qkv[r], zr, ba[r], a_log, dt, w, tmp, torch.tensor([1]), H, K ** -0.5, 1e-6,
                act)
            if int(idx[n, i]) > 0:
                want_state[int(idx[n, i])] = tmp[1]
    return want_out, want_state


VERIFY_CASES = [
    # (N, W, H, HV, K, lens, accepted, pad_tokens)
    (3, 5, 2, 6, 16, [5, 2, 4], [1, 5, 3], 0),
    (4, 3, 1, 2, 8, [3, 0, 1, 3], [2, 1, 1, 4], 2),   # padded seq, acc out of range, pad rows
    (2, 2, 2, 4, 16, [2, 2], [2, 1], 0),
]


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("N,W,H,HV,K,lens,acc,pad", VERIFY_CASES)
def test_verify_reference_is_decode_steps(N, W, H, HV, K, lens, acc, pad, act):
    lane = _verify_inputs(N, W, H, HV, K, seed=N, lens=lens, accepted=acc, pad_tokens=pad)
    lane[7][0, 1] = 0  # a NULL destination: that position's state is dropped
    before = [t.clone() for t in lane]
    ref_out, ref_state = kgdn1.verify_reference(*kgdn1.verify_body_args(*lane, H),
                                                K ** -0.5, 1e-6, act)
    for t, b in zip(lane, before):  # reference must not mutate its args
        assert torch.equal(t, b)
    want_out, want_state = _verify_by_decode_steps(lane, H, act)
    torch.testing.assert_close(ref_out.float().reshape(want_out.shape),
                               want_out.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_state.reshape(want_state.shape), want_state,
                               rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("N,W,H,HV,K,lens,acc,pad", VERIFY_CASES)
def test_verify_body_interpret_matches_reference(N, W, H, HV, K, lens, acc, pad, act):
    helion = pytest.importorskip("helion")
    lane = _verify_inputs(N, W, H, HV, K, seed=N + 7, lens=lens, accepted=acc, pad_tokens=pad)
    lane[7][N - 1, 0] = 0
    args = [*kgdn1.verify_body_args(*lane, H), K ** -0.5, 1e-6, act]
    ref = kgdn1.verify_reference(*args)
    got = helion.kernel(kgdn1.kgdn1_verify)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    for n in range(N):  # untouched slots: no state IO outside the sequence's row
        if lens[n] == 0 or not 1 <= acc[n] <= W:
            assert torch.count_nonzero(got[0][int(lane[8][n]):int(lane[8][n + 1])]) == 0


@pytest.mark.parametrize("H,HV,tp", RANKS)
def test_verify_body_interpret_tp_rank_shapes(H, HV, tp):
    helion = pytest.importorskip("helion")
    lane = _verify_inputs(2, 5, H, HV, 128, seed=tp, lens=[5, 3], accepted=[2, 5])
    args = [*kgdn1.verify_body_args(*lane, H), 128 ** -0.5, 1e-6, 0]
    ref = kgdn1.verify_reference(*args)
    got = helion.kernel(kgdn1.kgdn1_verify)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg


def test_verify_body_interpret_interleaved():
    helion = pytest.importorskip("helion")
    lane = _verify_inputs(3, 5, 4, 8, 32, seed=5, lens=[5, 1, 3], accepted=[3, 1, 5],
                          interleaved=True)
    assert lane[1].dim() == 4
    args = [*kgdn1.verify_body_args(*lane, 4), 32 ** -0.5, 1e-6, 0]
    ref = kgdn1.verify_reference(*args)
    want_out, _ = _verify_by_decode_steps(lane, 4)
    got = helion.kernel(kgdn1.kgdn1_verify)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    torch.testing.assert_close(got[0].float().reshape(want_out.shape),
                               want_out.to(torch.bfloat16).float(), rtol=2e-2, atol=3e-2)


def test_verify_lab_problem_shapes():
    c = kgdn1.verify_case(2, 2, 6, 16, 16, 3)
    args = kgdn1.verify_make(c, 0, device="cpu")
    assert args[10].shape == (2, 3) and args[0].shape == (6, 2, 16)  # state_idx, q
    assert args[9].stride(0) == 6 * 16 * 16 + 4096        # vLLM's padded state pages
    ok, msg = kgdn1.check(kgdn1.verify_reference(*args), kgdn1.verify_reference(*args))
    assert ok, msg


@pytest.mark.parametrize("static", [True, False])
def test_verify_body_lowers_to_triton(static):
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    lane = _verify_inputs(3, 5, 8, 24, 128, seed=1, lens=[5, 5, 5])
    args = [*kgdn1.verify_body_args(*lane, 8), 128 ** -0.5, 1e-6, 0]
    bound = helion.kernel(kgdn1.kgdn1_verify, static_shapes=static,
                          ref_mode=RefMode.OFF).bind(tuple(args))
    code = bound.to_triton_code(bound.config_spec.default_config())
    # one state read per sequence program, one store per (unrolled) token position
    assert code.count("tl.load(state +") == 1 and code.count("tl.store(state +") == 5


# ---------------------------------------------------------------------------
# routing on shapes the oxide cubin cannot run (TP>1, interleaved, verify)
# ---------------------------------------------------------------------------
def test_auto_non_oxide_shape_falls_back_to_stock(table, monkeypatch):
    monkeypatch.setenv(kgdn1.ENV, "auto")
    stock = []
    monkeypatch.setattr(qwen_gdn, "stock_decode_fused", lambda *a: stock.append(len(a)))
    native = _Native()
    op, buckets = qwen_gdn.decode_op(native, 8, 24, 128)          # TP2 rank of the 27B
    assert buckets == ()
    op(torch.empty(2, 1), *([None] * 8), 8, 0.1, 1e-6, 0, 0)
    assert stock == [13] and native.calls == []                  # stream_ptr dropped
    op, _ = qwen_gdn.decode_op(native, 16, 48, 128, layout="interleaved")
    op(torch.empty(1, 1), *([None] * 8), 16, 0.1, 1e-6, 0, 0)    # oxide cannot read 4-D z
    assert native.calls == [] and len(stock) == 2


def test_helion_non_oxide_shape_needs_config(table, monkeypatch):
    monkeypatch.setenv(kgdn1.ENV, "helion")
    with pytest.raises(RuntimeError, match="no tuned kgdn1_decode config"):
        qwen_gdn.decode_op(_Native(), 8, 24, 128)


def test_verify_op_routing(table, tmp_path, monkeypatch):
    monkeypatch.setenv(kgdn1.ENV, "oxide")
    assert kgdn1.verify_op(8, 24, 128, 128, 5) == (None, ())
    monkeypatch.setenv(kgdn1.ENV, "auto")
    assert kgdn1.verify_op(8, 24, 128, 128, 5) == (None, ())     # nothing shipped: vLLM stock
    assert kgdn1.verify_op(8, 24, 128, 128, 0) == (None, ())     # no spec decode
    (tmp_path / kgdn1.VERIFY_KERNEL).mkdir()
    rows = [{"key": kgdn1.verify_case(m, 8, 24, 128, 128, 5), "config": {"num_warps": 4}}
            for m in (8, 32)]
    (tmp_path / kgdn1.VERIFY_KERNEL / "testgpu.json").write_text(json.dumps(rows))
    monkeypatch.setattr(hk, "_TABLES", {})
    op, buckets = kgdn1.verify_op(8, 24, 128, 128, 5)
    assert op is kgdn1.gdn_verify_fused_helion and buckets == (8, 32)
    assert kgdn1.verify_op(8, 24, 128, 128, 3) == (None, ())     # other k: other cases
    monkeypatch.setenv(kgdn1.ENV, "helion")
    with pytest.raises(RuntimeError, match="no tuned kgdn1_verify config"):
        kgdn1.verify_op(8, 24, 128, 128, 5)                      # buckets 1, 128 missing


@pytest.mark.parametrize("interleaved", [False, True])
def test_verify_oracle_cpu(interleaved):
    """The lane's boot verify oracle (minus the CUDA-graph round) with the
    interpret-mode body as the op: its inputs, reference and gate agree."""
    helion = pytest.importorskip("helion")
    kern = helion.kernel(kgdn1.kgdn1_verify)

    def op(*a):
        *lane, H, scale, eps, act = a
        kern(*kgdn1.verify_body_args(*lane, H), scale, eps, act)

    worst = qwen_gdn.run_gpu_verify_oracle(op, 2, 4, 16, 16, 3, 0, 1e-6, (1, 8), interleaved,
                                           device=torch.device("cpu"))
    assert worst["out"] < 0.1 and worst["state"] < 1e-4

    def broken(*a):
        op(*a)
        a[10][0] += 1.0  # corrupt one output row

    with pytest.raises(RuntimeError, match="verify oracle FAIL|K-GDN1 oracle FAIL"):
        qwen_gdn.run_gpu_verify_oracle(broken, 2, 4, 16, 16, 3, 0, 1e-6, (8,), interleaved,
                                       device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# V-tiled bodies (kgdn1_decode_vt / kgdn1_verify_vt + the vt_norm launch)
# ---------------------------------------------------------------------------
V_BLOCKS = (16, 32, 64)


def _vt(body, v_block):
    helion = pytest.importorskip("helion")
    return helion.kernel(body, config=helion.Config(v_block=v_block),
                         ignore_warnings=kgdn1._ignore())


def _clone(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


@pytest.mark.parametrize("v_block", V_BLOCKS)
@pytest.mark.parametrize("H,HV,tp", RANKS[:4])            # (16, 48, 128) at tp 1/2/4/8
@pytest.mark.parametrize("interleaved", [False, True])
def test_decode_vt_interpret_tp_rank_shapes(H, HV, tp, interleaved, v_block):
    lane, args = _lane_decode(3, H, HV, 128, seed=tp, idx=[2, 0, 1], interleaved=interleaved)
    ref = kgdn1.reference(*args)
    full = pytest.importorskip("helion").kernel(kgdn1.kgdn1_decode)(*_clone(args))
    got = _vt(kgdn1.kgdn1_decode_vt, v_block)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    assert torch.count_nonzero(got[0][1]) == 0                    # NULL slot: zero row
    torch.testing.assert_close(got[0], full[0], rtol=0, atol=8e-3)  # ~1 bf16 ulp of the body
    torch.testing.assert_close(got[1], full[1], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("v_block", V_BLOCKS)
def test_decode_vt_interpret_small(v_block, act):
    """V < v_block (masked partial tile), K != V, sigmoid gate."""
    args = list(kgdn1.make(kgdn1.case(5, 2, 6, 16, 16), 3, device="cpu"))
    args[10].copy_(torch.tensor([3, 0, -1, 1, 5], dtype=torch.int32))
    args[-1] = act
    ok, msg = kgdn1.check(_vt(kgdn1.kgdn1_decode_vt, v_block)(*_clone(args)), kgdn1.reference(*args))
    assert ok, msg
    lane, kv = _lane_decode(2, 2, 4, 32, seed=1, idx=[1, 2])
    inp = qwen_gdn.make_inputs(2, 2, 4, 32, slots=4, seed=2, idx=[2, 1], page_pad=8, V=64)
    out = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    kv = [*kgdn1.body_args(inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
                           inp["norm_w"], inp["state"], inp["state_idx"], out, 2),
          32 ** -0.5, 1e-6, act]
    ok, msg = kgdn1.check(_vt(kgdn1.kgdn1_decode_vt, v_block)(*_clone(kv)), kgdn1.reference(*kv))
    assert ok, msg


def _rand_verify(N, W, H, HV, K, seed, interleaved=False, pad=0):
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(0, W + 1, (N,), generator=gen).tolist()
    acc = torch.randint(1, W + 2, (N,), generator=gen).tolist()  # W + 1: out of range
    lane = _verify_inputs(N, W, H, HV, K, seed=seed, lens=lens, accepted=acc,
                          interleaved=interleaved, pad_tokens=pad)
    lane[10].fill_(0.5)  # rows no sequence covers must keep their values
    return lane


@pytest.mark.parametrize("v_block", V_BLOCKS)
@pytest.mark.parametrize("H,HV,tp", RANKS[:4])
@pytest.mark.parametrize("interleaved", [False, True])
def test_verify_vt_interpret_tp_rank_shapes(H, HV, tp, interleaved, v_block):
    lane = _rand_verify(3, 5, H, HV, 128, seed=10 * tp + v_block, interleaved=interleaved, pad=2)
    lane[7][2, 0] = 0  # a NULL destination
    args = [*kgdn1.verify_body_args(*lane, H), 128 ** -0.5, 1e-6, 0]
    ref = kgdn1.verify_reference(*args)
    full = pytest.importorskip("helion").kernel(kgdn1.kgdn1_verify)(*_clone(args))
    got = _vt(kgdn1.kgdn1_verify_vt, v_block)(*args)
    ok, msg = kgdn1.check(got, ref)
    assert ok, msg
    torch.testing.assert_close(got[0], full[0], rtol=0, atol=8e-3)
    torch.testing.assert_close(got[1], full[1], rtol=1e-6, atol=1e-7)
    assert torch.all(got[0][-2:] == 0.5)                         # pad rows untouched


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("N,W,H,HV,K,lens,acc,pad", VERIFY_CASES)
def test_verify_vt_interpret_edge_cases(N, W, H, HV, K, lens, acc, pad, act):
    lane = _verify_inputs(N, W, H, HV, K, seed=N + 7, lens=lens, accepted=acc, pad_tokens=pad)
    lane[7][N - 1, 0] = 0
    lane[10].fill_(-1.0)
    args = [*kgdn1.verify_body_args(*lane, H), K ** -0.5, 1e-6, act]
    for v_block in V_BLOCKS:
        ok, msg = kgdn1.check(_vt(kgdn1.kgdn1_verify_vt, v_block)(*_clone(args)),
                              kgdn1.verify_reference(*args))
        assert ok, (v_block, msg)


@pytest.mark.parametrize("v_block", V_BLOCKS)
def test_verify_vt_random_accepted_qwen4exp_tp2(v_block):
    """Today's measured case: per-rank H=8, HV=24, K=V=128, w=5; random lens/accepted."""
    for seed in range(3):
        lane = _rand_verify(4, 5, 8, 24, 128, seed=seed)
        args = [*kgdn1.verify_body_args(*lane, 8), 128 ** -0.5, 1e-6, 0]
        ok, msg = kgdn1.check(_vt(kgdn1.kgdn1_verify_vt, v_block)(*_clone(args)),
                              kgdn1.verify_reference(*args))
        assert ok, (seed, msg)


@pytest.mark.parametrize("static", [True, False])
def test_vt_bodies_lower_to_triton(static):
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode

    def code(body, args, v_block=32):
        bound = helion.kernel(body, static_shapes=static, ref_mode=RefMode.OFF,
                              ignore_warnings=kgdn1._ignore()).bind(tuple(args))
        cfg = bound.config_spec.default_config()
        assert cfg["v_block"] == 32
        return bound.to_triton_code(helion.Config(**{**cfg.config, "v_block": v_block}))

    src = code(kgdn1.kgdn1_decode_vt, _args(3, 8, 24, 128))
    assert src.count("tl.load(state +") == 1 and src.count("tl.store(state +") == 1
    assert "(128 + _BLOCK_SIZE_3 - 1) // _BLOCK_SIZE_3" in src       # V tiles in the grid
    assert "torch.ops.suffix_kgdn1.vt_norm(o, z, norm_w, out, None, None" in src
    lane = _verify_inputs(3, 5, 8, 24, 128, seed=1, lens=[5, 5, 5])
    src = code(kgdn1.kgdn1_verify_vt, [*kgdn1.verify_body_args(*lane, 8), 128 ** -0.5, 1e-6, 0],
               v_block=16)
    assert src.count("tl.load(state +") == 1 and src.count("tl.store(state +") == 5
    assert "torch.ops.suffix_kgdn1.vt_norm(o, z, norm_w, out, cu_lo, cu_hi" in src
    # the norm launches lower at their fixed config
    o = torch.zeros(3, 8, 3, 128)
    z = torch.zeros(3, 8, 3, 128, dtype=torch.bfloat16)
    for body, extra, o_ in ((kgdn1.kgdn1_vt_norm, (), o),
                            (kgdn1.kgdn1_vt_norm_seq, (lane[8][:-1], lane[8][1:]),
                             o.view(3, 1, 8, 3, 128).expand(3, 5, 8, 3, 128).contiguous())):
        b = helion.kernel(body, config=helion.Config(**kgdn1.NORM_CONFIG), static_shapes=static,
                          ref_mode=RefMode.OFF).bind((o_, z, torch.ones(128), z.clone(), *extra,
                                                      1e-6, 0))
        assert "tl.store(out +" in b.to_triton_code(helion.Config(**kgdn1.NORM_CONFIG))


def _ship(tmp_path, kernel, cases):
    (tmp_path / kernel).mkdir(exist_ok=True)
    rows = [{"key": c, "config": {"num_warps": 4}} for c in cases]
    (tmp_path / kernel / "testgpu.json").write_text(json.dumps(rows))
    hk._TABLES.clear()


def test_pick_rule(table, tmp_path, monkeypatch):
    """Full tile where shipped, else V-tiled, else the baseline (per bucket)."""
    monkeypatch.setenv(kgdn1.ENV, "auto")
    dec = lambda m: kgdn1.case(m, 8, 24, 128, 128)  # noqa: E731
    _ship(tmp_path, kgdn1.KERNEL, [dec(8), dec(32)])
    _ship(tmp_path, kgdn1.KERNEL_VT, [dec(1), dec(8)])
    assert kgdn1.served(8, 24, 128, 128) == {1: kgdn1.KERNEL_VT, 8: kgdn1.KERNEL,
                                              32: kgdn1.KERNEL}
    calls = []
    monkeypatch.setattr(hk, "compiled", lambda body, name, c, **kw: (
        lambda *a: calls.append((body.__name__, name, c["numtokens"]))))
    monkeypatch.setattr(kgdn1, "body_args", lambda *a: ())
    monkeypatch.setattr(kgdn1, "verify_body_args", lambda *a: ())
    native = _Native()
    op, buckets = qwen_gdn.decode_op(native, 8, 24, 128)
    assert buckets == (1, 8, 32)
    for T in (1, 7, 20):
        op(torch.empty(T, 1), *([None] * 5), torch.empty(4, 24, 128, 128, device="meta"),
           None, None, 8, 0.1, 1e-6, 0, 0)
    assert calls == [("kgdn1_decode_vt", kgdn1.KERNEL_VT, 1), ("kgdn1_decode", kgdn1.KERNEL, 8),
                     ("kgdn1_decode", kgdn1.KERNEL, 32)]
    # helion mode: the V-tiled row satisfies the bucket; 128 is still missing
    monkeypatch.setenv(kgdn1.ENV, "helion")
    with pytest.raises(RuntimeError, match="no tuned kgdn1_decode config"):
        qwen_gdn.decode_op(native, 8, 24, 128)
    _ship(tmp_path, kgdn1.KERNEL_VT, [dec(1), dec(8), dec(128)])
    assert qwen_gdn.decode_op(native, 8, 24, 128)[1] == (1, 8, 32, 128)
    # verify: V-tiled rows alone route the buckets they cover
    monkeypatch.setenv(kgdn1.ENV, "auto")
    ver = lambda m: kgdn1.verify_case(m, 8, 24, 128, 128, 5)  # noqa: E731
    _ship(tmp_path, kgdn1.VERIFY_KERNEL_VT, [ver(1), ver(8)])
    op, vb = kgdn1.verify_op(8, 24, 128, 128, 5)
    assert op is kgdn1.gdn_verify_fused_helion and vb == (1, 8)
    assert kgdn1.served(8, 24, 128, 128, 5) == {1: kgdn1.VERIFY_KERNEL_VT,
                                                 8: kgdn1.VERIFY_KERNEL_VT}
    calls.clear()
    kgdn1.gdn_verify_fused_helion(None, *([None] * 5), torch.empty(4, 24, 128, 128, device="meta"),
                                  torch.zeros(1, 5, dtype=torch.int32), None, None, None, 8,
                                  0.1, 1e-6, 0)
    assert calls == [("kgdn1_verify_vt", kgdn1.VERIFY_KERNEL_VT, 5)]


def test_no_vt_rows_is_todays_routing(table, monkeypatch):
    """Only full-tile rows shipped: the same buckets, bodies and configs as before."""
    monkeypatch.setenv(kgdn1.ENV, "auto")
    table(1, 32)
    assert kgdn1.served(16, 48, 128, 128) == {1: kgdn1.KERNEL, 32: kgdn1.KERNEL}
    assert qwen_gdn.decode_op(_Native(), 16, 48, 128)[1] == (1, 32)


@pytest.mark.parametrize("interleaved", [False, True])
def test_verify_oracle_cpu_vt(interleaved):
    """The boot verify oracle with the V-tiled body as the op (CPU, no graph round)."""
    kern = _vt(kgdn1.kgdn1_verify_vt, 16)

    def op(*a):
        *lane, H, scale, eps, act = a
        kern(*kgdn1.verify_body_args(*lane, H), scale, eps, act)

    worst = qwen_gdn.run_gpu_verify_oracle(op, 2, 4, 32, 32, 5, 0, 1e-6, (1, 8), interleaved,
                                           device=torch.device("cpu"))
    assert worst["out"] < 0.1 and worst["state"] < 1e-4
