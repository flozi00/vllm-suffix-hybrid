# SPDX-License-Identifier: Apache-2.0
"""Helion K2-NVFP4 port (suffix_hybrid/helion_kernels/k2_attn.py), CPU only.

* pure python: lab cases for Gemma 4 layouts, bucketing, view contract,
  exact e2m1 -> f16 bits, the V-scale swizzle row formula, the torch
  reference == own_attn_ref.reference (independent float64 numpy), knob
  routing (oxide default, auto -> oxide without a config, helion fails
  closed);
* interpret mode (HELION_INTERPRET=1, skipped without Helion): the body vs
  the reference on tiny paged caches — decode and verify widths, sliding
  window, ragged q lengths with an empty request, graph-padding rows, stale
  NaN scale codes past kv_len.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sm120"))

from nvfp4_kv_patch import own_attn  # noqa: E402
from nvfp4_kv_patch import own_attn_ref as ref  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import k2_attn as K  # noqa: E402

GEMMA26 = {"name": "gemma-4-26b-a4b-nvfp4", "architecture": "Gemma4ForConditionalGeneration",
           "tp": 1, "heads": 16, "kv_heads": 8, "head_dim": 256}
GEMMA31_TP2 = {**GEMMA26, "name": "gemma-4-31b-it-nvfp4", "tp": 2}


def _problem(d, hq, hkv, page, wl, batch, width, pad=0, seed=0):
    return K._problem(d, hq, hkv, page, wl, batch, width, seed, "cpu",
                      ctx=lambda i, n: 37 + 61 * i, pad=pad)


def _with_qo(args, lens):
    qo = torch.tensor([0, *np.cumsum(lens)], dtype=torch.int32)
    return (*args[:7], qo, *args[8:])


def test_cases_cover_gemma4_layouts_only():
    for model in (GEMMA26, GEMMA31_TP2):
        cs = K.cases(model)
        assert len(cs) == 12 <= 64
        assert {(c["head_dim"], c["kv_heads"], c["window"]) for c in cs} == {
            (256, 8, 1023), (512, 2, -1)}
        assert {(c["width"], c["numtokens"]) for c in cs} == {
            (w, b) for w in K.WIDTHS for b in K.BATCHES}
        assert all(c["heads"] == 16 and c["page"] == 64 for c in cs)
    assert K.cases({**GEMMA26, "architecture": "Qwen3ForCausalLM"}) == []
    assert K.cases({"architecture": "Gemma4ForConditionalGeneration"}) == []
    entry, = K.KERNELS
    assert entry["name"] == K.KERNEL and entry["baseline_name"] == "oxide k2_nvfp4_attn"
    assert entry["replaces"] == [r"nvfp4_attn_partial", r"nvfp4_attn_merge"]


def test_serving_case_buckets_batch_and_width():
    c = K.case(512, 16, 2, 64, -1, 5, 3)
    assert (c["numtokens"], c["width"], c["window"]) == (8, 4, -1)
    assert K.case(256, 16, 8, 64, 1023, 40, 9)["numtokens"] == 32
    assert K.case(256, 16, 8, 64, 1023, 1, 1)["width"] == 1
    assert K.cases(GEMMA26)[0] == K.case(256, 16, 8, 64, 1023, 1, 1)


def test_e2m1_f16_bits_are_exact():
    codes = torch.arange(16, dtype=torch.int32)
    got = K._e2m1_f16(codes).float().tolist()
    assert got == list(ref.E2M1) and math.copysign(1, got[8]) == -1


def test_v_scale_rows_follow_the_store_swizzle():
    sd, page = 32, 64
    for t in range(page):
        base = (t // 4) * 4 * sd + t % 4
        assert [base + 4 * g for g in range(sd)] == [
            ref.swizzle_scale_offset(t, g, sd) for g in range(sd)]


def test_body_args_round_trip_and_contract():
    args = _problem(128, 4, 2, 16, -1, 2, 1)
    kd, ks, vd, vs = K.natural_views(args)
    assert kd.stride(0) == args[12] and tuple(ks.shape) == (kd.shape[0], 2, 16, 8)
    again = K.body_args(args[0], kd, ks, vd, vs, *args[5:10], -1, 1.0, 1.0)
    for a, b in zip(again[1:5], args[1:5]):
        assert a.shape == b.shape and a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
    with pytest.raises(ValueError, match="HND nvfp4 view"):
        K.body_args(args[0], kd.contiguous(), ks, vd, vs, *args[5:10], -1, 1.0, 1.0)


@pytest.mark.parametrize("d,hq,hkv,page,wl,lens", [
    (128, 4, 2, 16, -1, (1, 1, 1)),
    (256, 4, 1, 32, 20, (3, 0, 1, 2)),
    (512, 8, 2, 16, -1, (4,)),
])
def test_reference_matches_numpy_float64(d, hq, hkv, page, wl, lens):
    args = _with_qo(_problem(d, hq, hkv, page, wl, len(lens), max(lens), pad=2), lens)
    k_data, pb = K.natural_views(args)[0], args[12]
    p = k_data.shape[0]
    raw = torch.as_strided(k_data, (p, pb), (pb, 1), k_data.storage_offset())
    cache = ref.Cache(raw.numpy().reshape(p, 2 * hkv, page, d // 2 + d // 16), hkv, page, d)
    q = args[0][:sum(lens)].float().numpy().astype(np.float64)
    want = ref.reference(q, cache, args[5].numpy(), args[6].numpy(), list(lens),
                         1 / math.sqrt(d), window_left=wl)
    got = K.reference(*args)
    assert np.allclose(got[:sum(lens)].numpy(), want, atol=1e-4, rtol=1e-4)
    assert bool((got[sum(lens):] == 0).all())


def test_knob_routing(monkeypatch, tmp_path):
    monkeypatch.delenv(K.ENV, raising=False)
    assert own_attn.helion_port() is None  # oxide default: Helion never imported
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path))
    monkeypatch.setattr(hk, "_TABLES", {})
    monkeypatch.setattr(hk, "platform", lambda device=0: "cpu_test")
    monkeypatch.setattr(K, "_ROUTE", {})
    monkeypatch.setattr(K, "_FAST", {})
    args = _problem(128, 4, 2, 16, -1, 2, 1)
    oxide_args = (args[0], *K.natural_views(args), *args[5:9], 1, -1, 0.1, 1.0)
    monkeypatch.setenv(K.ENV, "auto")
    assert own_attn.helion_port() is K
    assert K.run(*oxide_args) is None  # no shipped config -> oxide serves it
    monkeypatch.setattr(K, "_ROUTE", {})
    monkeypatch.setattr(K, "_FAST", {})
    monkeypatch.setenv(K.ENV, "helion")
    with pytest.raises(RuntimeError, match="no tuned k2_attn config"):
        K.run(*oxide_args)


def test_check_gates_padding_and_accuracy():
    ref_out = torch.zeros(3, 2, 4)
    ref_out[0] = 1.0
    ok, _ = K.check(ref_out.to(torch.bfloat16), ref_out)
    assert ok
    bad = ref_out.clone()
    bad[2] = 0.5
    assert not K.check(bad, ref_out)[0]
    bad = ref_out.clone()
    bad[0, 0, 0] = 1.5
    assert not K.check(bad, ref_out)[0]


# ---- interpret mode -----------------------------------------------------------
@pytest.fixture(scope="module")
def body():
    os.environ["HELION_INTERPRET"] = "1"
    helion = pytest.importorskip("helion")
    # CPU torch has no aten::mm.dtype (hl.dot's eager reference): fp32
    # accumulate then cast, which is what tl.dot(out_dtype=f32) does.
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("mm.dtype", lambda a, b, out_dtype: (a.float() @ b.float()).to(out_dtype), "CPU")
    yield helion.kernel(K._k2_nvfp4_attn, **K.SETTINGS)
    del lib


@pytest.mark.parametrize("d,hq,hkv,page,wl,lens,pad", [
    (128, 4, 2, 16, -1, (1, 1, 1), 0),        # decode, G 2
    (128, 8, 1, 16, 20, (3, 3), 2),           # verify, G 8, window, padding
    (256, 4, 2, 32, -1, (4,), 0),             # hd256 verify
    (512, 8, 2, 16, -1, (1, 4, 0, 2), 3),     # hd512 ragged + empty request
    (256, 4, 2, 16, 33, (2, 0, 1), 1),        # window on ragged rows
])
def test_body_matches_reference(body, d, hq, hkv, page, wl, lens, pad):
    args = _with_qo(_problem(d, hq, hkv, page, wl, len(lens), max(lens), pad=pad), lens)
    want = K.reference(*args)
    ok, msg = K.check(body(*args), want)
    assert ok, msg
    assert int(args[9].abs().sum()) == 0  # counters left zero


def test_body_ignores_stale_nan_scales_past_kv_len(body):
    d, hq, hkv, page = 128, 4, 2, 16
    args = _problem(d, hq, hkv, page, -1, 2, 2)
    kd, ks, vd, vs = K.natural_views(args)
    for b, n in enumerate(args[6].tolist()):
        if n % page:
            last = int(args[5][b, (n - 1) // page])
            start = n % page
            ks.view(torch.uint8)[last, :, start:] = 0x7F
            g4 = -(-start // 4) * 4
            vs.view(torch.uint8)[last].flatten(1)[:, g4 * (d // 16):] = 0x7F
            vd[last, :, start:] = 0xFF
    want = K.reference(*args)
    ok, msg = K.check(body(*args), want)
    assert ok, msg
