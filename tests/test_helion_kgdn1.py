# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 Helion port (suffix_hybrid/helion_kernels/kgdn1.py): lab ABI, cases,
the torch reference vs the lane's oracle reference, backend routing, and —
with Helion installed — the body in interpret mode on CPU plus Triton codegen
(the frontend proof; speed and SASS are the lab gate's job)."""
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
    (ent,) = kgdn1.KERNELS
    assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"]) and ent["name"] == kgdn1.KERNEL
    assert ent["body"].__qualname__ == "kgdn1_decode"  # module-level, undecorated
    for key in ("cases", "make", "make_baseline", "reference", "baseline", "check"):
        assert callable(ent[key])
    pats = ent["replaces"]
    assert 0 < len(pats) <= 16 and all(len(p) <= 300 for p in pats)
    for name in ("kgdn1_decode_h16_hv48_k128",
                 "fused_recurrent_gated_delta_rule_packed_decode_kernel",
                 "layer_norm_fwd_kernel"):
        assert any(re.search(p, name, re.I) for p in pats), name
    assert not any(re.search(p, "fused_add_rms_norm_kernel", re.I) for p in pats)


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
