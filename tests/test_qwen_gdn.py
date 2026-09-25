# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 (qwen3.8-27b fused GDN decode step) — CPU contract tests.

Two independent implementations of the same vLLM 0.30.0 semantics must agree:
  * the Rust CPU reference `_native.gdn_decode_fused_ref` (src/qwen_gdn.rs,
    the GPU kernel's oracle), and
  * the torch reference `gdn_decode_fused_torch` (suffix_hybrid/kernels/
    qwen_gdn.py, a transcription of FLA packed decode + RMSNormGated).
Plus the fail-loud wiring contract (gate off = inert; gate on without the
GPU op / off SM120 = hard error) and the entry-point plumbing.
The on-GPU identity + CUDA-graph check runs only with CUDA + a
`qwen-gdn-kernels` wheel (skipped otherwise, loudly).
"""
import configparser
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

from suffix_hybrid import _native
from suffix_hybrid.kernels import qwen_gdn

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_both(T, H, HV, K, slots, act, seed, idx=None, eps=1e-6):
    inp = qwen_gdn.make_inputs(T, H, HV, K, slots, seed=seed, idx=idx)
    f32 = {k: (v.float() if v.is_floating_point() else v) for k, v in inp.items()}
    st_torch = f32["state"].clone()
    ref = qwen_gdn.gdn_decode_fused_torch(
        f32["mixed_qkv"], f32["z"], f32["ba"], f32["a_log"], f32["dt_bias"],
        f32["norm_w"], st_torch, f32["state_idx"], H, K ** -0.5, eps, act)
    st_rust = np.ascontiguousarray(f32["state"].numpy().copy())
    out = _native.gdn_decode_fused_ref(
        np.ascontiguousarray(f32["mixed_qkv"].numpy()),
        np.ascontiguousarray(f32["z"].numpy()),
        f32["ba"].numpy(), f32["a_log"].numpy(), f32["dt_bias"].numpy(),
        f32["norm_w"].numpy(), st_rust, f32["state_idx"].numpy(), H, K ** -0.5,
        eps, act)
    return ref, torch.from_numpy(out), st_torch, torch.from_numpy(st_rust), f32


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("T,H,HV,K", [(1, 1, 1, 8), (4, 2, 6, 16), (5, 4, 12, 32)])
def test_rust_reference_matches_torch_reference(T, H, HV, K, act):
    ref, out, st_t, st_r, _ = _run_both(T, H, HV, K, slots=T + 3, act=act, seed=T * 7 + K)
    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(st_r, st_t, rtol=2e-5, atol=2e-6)


def test_serving_shape_qwen38_27b():
    # The real layer shape: H=16 key heads, HV=48 value heads, K=V=128.
    ref, out, st_t, st_r, _ = _run_both(2, 16, 48, 128, slots=4, act=0, seed=1)
    torch.testing.assert_close(out, ref, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(st_r, st_t, rtol=5e-5, atol=5e-6)


def test_null_slots_zero_rows_and_untouched_state():
    ref, out, st_t, st_r, f32 = _run_both(4, 2, 4, 16, slots=6, act=0, seed=3,
                                          idx=[0, 3, -1, 5])
    for t in (0, 2):
        assert torch.count_nonzero(out[t]) == 0
        assert torch.count_nonzero(ref[t]) == 0
    untouched = [0, 1, 2, 4]
    torch.testing.assert_close(st_r[untouched], f32["state"][untouched], rtol=0, atol=0)
    assert not torch.equal(st_r[3], f32["state"][3])


def test_decode_recurrence_over_steps():
    # 6 consecutive steps on the same slot: the recurrence must keep agreeing
    # (errors compound through the state if any op differs).
    H, HV, K = 2, 4, 16
    st_t = st_r = None
    for step in range(6):
        inp = qwen_gdn.make_inputs(1, H, HV, K, slots=3, seed=100 + step, idx=[2])
        f = {k: (v.float() if v.is_floating_point() else v) for k, v in inp.items()}
        if st_t is None:
            st_t = f["state"].clone()
            st_r = f["state"].numpy().copy()
        ref = qwen_gdn.gdn_decode_fused_torch(
            f["mixed_qkv"], f["z"], f["ba"], f["a_log"], f["dt_bias"], f["norm_w"],
            st_t, f["state_idx"], H, K ** -0.5, 1e-6, 0)
        out = _native.gdn_decode_fused_ref(
            np.ascontiguousarray(f["mixed_qkv"].numpy()),
            np.ascontiguousarray(f["z"].numpy()), f["ba"].numpy(),
            f["a_log"].numpy(), f["dt_bias"].numpy(), f["norm_w"].numpy(), st_r,
            f["state_idx"].numpy(), H, K ** -0.5, 1e-6, 0)
        torch.testing.assert_close(torch.from_numpy(out), ref, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(torch.from_numpy(st_r), st_t, rtol=5e-5, atol=5e-6)


def test_rust_reference_rejects_bad_contract():
    inp = qwen_gdn.make_inputs(2, 2, 5, 16, 3, seed=0)  # HV % H != 0
    f = {k: (v.float() if v.is_floating_point() else v) for k, v in inp.items()}
    with pytest.raises(ValueError, match="multiple of H"):
        _native.gdn_decode_fused_ref(
            np.ascontiguousarray(f["mixed_qkv"].numpy()),
            np.ascontiguousarray(f["z"].numpy()), f["ba"].numpy(),
            f["a_log"].numpy(), f["dt_bias"].numpy(), f["norm_w"].numpy(),
            f["state"].numpy().copy(), f["state_idx"].numpy(), 2, 0.25, 1e-6, 0)
    inp = qwen_gdn.make_inputs(1, 1, 1, 8, 2, seed=0, idx=[7])  # slot >= S
    f = {k: (v.float() if v.is_floating_point() else v) for k, v in inp.items()}
    with pytest.raises(ValueError, match="out of range"):
        _native.gdn_decode_fused_ref(
            np.ascontiguousarray(f["mixed_qkv"].numpy()),
            np.ascontiguousarray(f["z"].numpy()), f["ba"].numpy(),
            f["a_log"].numpy(), f["dt_bias"].numpy(), f["norm_w"].numpy(),
            f["state"].numpy().copy(), f["state_idx"].numpy(), 1, 0.25, 1e-6, 0)


def test_bf16_rounding_of_o_is_within_numerics_gate():
    # Stock chain rounds o to bf16 before the norm; K-GDN1 does not. The
    # difference must sit inside the oracle tolerance the GPU gate uses.
    inp = qwen_gdn.make_inputs(3, 16, 48, 128, 5, seed=11)
    a = qwen_gdn.gdn_decode_fused_torch(
        inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
        inp["norm_w"], inp["state"].clone(), inp["state_idx"], 16, 128 ** -0.5,
        1e-6, 0, round_o_bf16=False)
    b = qwen_gdn.gdn_decode_fused_torch(
        inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
        inp["norm_w"], inp["state"].clone(), inp["state_idx"], 16, 128 ** -0.5,
        1e-6, 0, round_o_bf16=True)
    a16 = a.to(torch.bfloat16).float()  # the kernel stores bf16
    assert bool(((a16 - b).abs() <= qwen_gdn.OUT_ATOL + qwen_gdn.OUT_RTOL * b.abs()).all())


# ---------------------------------------------------------------------------
# wiring: gate, fail-loud, entry point
# ---------------------------------------------------------------------------
def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv("SUFFIX_QWEN_GDN", raising=False)
    before = set(sys.modules)
    assert qwen_gdn.register() is None
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_gate_on_without_gpu_op_fails_loud(monkeypatch):
    monkeypatch.setenv("SUFFIX_QWEN_GDN", "1")
    monkeypatch.setattr(qwen_gdn, "_state", dict(qwen_gdn._state, armed=False))
    if getattr(_native, "QWEN_GDN_BACKEND", "none") == "oxide":
        pytest.skip("wheel carries the oxide GPU op; covered by the GPU test")
    with pytest.raises(RuntimeError, match="oxide-kernels"):
        qwen_gdn.register()


def test_sm120_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    with pytest.raises(RuntimeError, match="SM family 120"):
        qwen_gdn.check_sm120()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    assert qwen_gdn.check_sm120() == (12, 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device"):
        qwen_gdn.check_sm120()


class _Norm:
    activation = "silu"
    norm_before_gate = True
    group_size = None
    bias = None


class _Layer:
    tp_size = 1
    gqa_interleaved_layout = False
    head_k_dim = head_v_dim = 128
    num_k_heads, num_v_heads = 16, 48
    norm = _Norm()

    class model_config:
        dtype = torch.bfloat16

    def get_state_dtype(self):
        return torch.bfloat16, torch.float32


def test_layer_contract():
    assert qwen_gdn.layer_contract_violations(_Layer()) == []
    bad = _Layer()
    bad.tp_size = 2
    bad.get_state_dtype = lambda: (torch.bfloat16, torch.bfloat16)
    why = qwen_gdn.layer_contract_violations(bad)
    assert any("tp_size" in w for w in why) and any("state dtype" in w for w in why)


def test_entry_point_dist_info():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "suffix_qwen_gdn_ep-1.0.dist-info" / "entry_points.txt")
    target = cfg["vllm.general_plugins"]["suffix_qwen_gdn"]
    mod, fn = target.split(":")
    assert mod == "suffix_hybrid.kernels.qwen_gdn" and fn == "register"
    assert callable(getattr(qwen_gdn, fn))


def test_padded_page_state_matches_contiguous():
    # vLLM's mamba state is an as_strided view with a padded page stride.
    a = qwen_gdn.make_inputs(3, 2, 4, 16, 5, seed=4)
    b = qwen_gdn.make_inputs(3, 2, 4, 16, 5, seed=4, page_pad=96)
    assert b["state"].stride(0) == 4 * 16 * 16 + 96
    torch.testing.assert_close(b["state"], a["state"], rtol=0, atol=0)
    args = lambda i: (i["mixed_qkv"], i["z"], i["ba"], i["a_log"], i["dt_bias"],
                      i["norm_w"], i["state"], i["state_idx"], 2, 0.25, 1e-6, 0)
    torch.testing.assert_close(qwen_gdn.gdn_decode_fused_torch(*args(b)),
                               qwen_gdn.gdn_decode_fused_torch(*args(a)))
    torch.testing.assert_close(b["state"], a["state"])


# ---------------------------------------------------------------------------
# bundle-prebuilt cubins: manifest contract (fail closed, no JIT)
# ---------------------------------------------------------------------------
def _fake_bundle(tmp, corrupt=False, version="13.2"):
    import hashlib
    import json
    cub = b"\x7fELF-fake-cubin"
    (tmp / "a.cubin").write_bytes(cub)
    man = {"kernel": "gdn_decode_fused_k1", "arch": "sm_120",
           "tileiras_version": "Cuda compilation tools\nBuild 13.4",
           "bytecode_version": version,
           "entries": [{"file": "a.cubin", "h": 16, "hv": 48, "k": 128, "act": 0,
                        "t_div": 1, "s_div": 16, "bc_sha256": "0" * 64,
                        "sha256": hashlib.sha256(b"x" if corrupt else cub).hexdigest(),
                        "bytes": len(cub)}]}
    (tmp / "manifest.json").write_text(json.dumps(man))
    return man


def _oxide_bundle(tmp, with_kgdn1=True):
    import json
    kernels = [{"name": "probe", "file": "probe.cubin", "sha256": "0" * 64,
                "ptx_isa": "8.7", "entries": ["probe_add"]}]
    if with_kgdn1:
        kernels.append({"name": "kgdn1", "file": "kgdn1.cubin", "sha256": "ab" * 32,
                        "ptx_isa": "8.7", "entries": ["kgdn1_decode_h16_hv48_k128"]})
    (tmp / "manifest.json").write_text(json.dumps(
        {"arch": "sm_120", "ptxas_version": "13.0.88", "kernels": kernels}))


def test_oxide_manifest_without_kgdn1_fails_closed(tmp_path, monkeypatch):
    from suffix_hybrid import oxide_kernels
    _oxide_bundle(tmp_path, with_kgdn1=False)
    monkeypatch.setenv("SUFFIX_OXIDE_CUBINS", str(tmp_path))
    monkeypatch.setattr(oxide_kernels, "_MANIFEST", None)
    with pytest.raises(RuntimeError, match="no 'kgdn1'"):
        qwen_gdn.oxide_manifest_entry()


def test_oxide_manifest_kgdn1_entry(tmp_path, monkeypatch):
    from suffix_hybrid import oxide_kernels
    _oxide_bundle(tmp_path)
    monkeypatch.setenv("SUFFIX_OXIDE_CUBINS", str(tmp_path))
    monkeypatch.setattr(oxide_kernels, "_MANIFEST", None)
    ent = qwen_gdn.oxide_manifest_entry()
    assert ent["entries"] == [qwen_gdn_oxide_entry()]
    assert ent["ptxas_version"] == "13.0.88"


def qwen_gdn_oxide_entry():
    from suffix_hybrid.kernels import qwen_gdn_oxide
    return qwen_gdn_oxide.interface()["entry"]


def test_load_prebuilt_refuses_other_shapes():
    with pytest.raises(ValueError, match="built for"):
        qwen_gdn.load_prebuilt(16, 32, 128)


def test_runtime_bundle_ships_qgdn_cubins(tmp_path):
    import importlib.util
    import json
    import zipfile
    _fake_bundle(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "rb_qgdn", ROOT / "scripts" / "runtime_bundle.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    wheel = tmp_path / "p.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("suffix_hybrid/__init__.py", "")
        z.writestr("suffix_hybrid/_native.abi3.so", b"x")
    out = tmp_path / "runtime"
    rb.bundle(wheel, out, "f" * 40, qgdn_cubins=tmp_path)
    build = json.loads((out / "BUILD.json").read_text())["sha256"]
    assert "suffix_hybrid/qgdn_cubins/manifest.json" in build
    assert "suffix_hybrid/qgdn_cubins/a.cubin" in build
    # the pod-side default location resolves to exactly this directory
    assert (out / "suffix_hybrid" / "qgdn_cubins" / "a.cubin").is_file()
    bad = tmp_path / "bad"
    bad.mkdir()
    _fake_bundle(bad, corrupt=True)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        rb.bundle(wheel, tmp_path / "r2", "f" * 40, qgdn_cubins=bad)


# ---------------------------------------------------------------------------
# GPU: the real kernel (SM120 + oxide-kernels wheel + bundle SASS only)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.skipif(getattr(_native, "QWEN_GDN_BACKEND", "none") != "oxide",
                    reason="wheel built without cargo feature oxide-kernels")
def test_gpu_kernel_oracle_and_graph_replay():
    qwen_gdn.check_sm120()
    assert "prebuilt SASS loaded, ptxas 13.0" in qwen_gdn.load_prebuilt(16, 48, 128)
    for act in (0, 1):
        worst = qwen_gdn.run_gpu_oracle(_native, 16, 48, 128, act, 1e-6)
        assert worst["out"] < 1.0
