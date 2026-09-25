# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 cuda-oxide port: CPU twin vs the torch reference + interface lock.

The twin (suffix_hybrid/kernels/qwen_gdn_oxide.py) transcribes the SIMT
kernel (kernels-oxide/kgdn1/src/main.rs) incl. its lane layout and xor
reduction order; here it must reproduce vLLM's semantics (torch reference)
at the qwen3.8-27b serving shape, with vLLM's padded state pages, NULL slots,
both gate activations and T up to 16. The interface test pins the PTX entry
name + parameter order between the Rust kernel and interface.json (what the
host launcher / manifest consume).
"""
import pathlib
import re

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import qwen_gdn
from suffix_hybrid.kernels import qwen_gdn_oxide as ox

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels-oxide" / "kgdn1" / "src" / "main.rs"


def _bits(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)


def _run(T, idx, act, page_pad=512, seed=0):
    H, HV, K = ox.H, ox.HV, ox.K
    slots = 8
    inp = qwen_gdn.make_inputs(T, H, HV, K, slots, seed=seed, idx=idx, page_pad=page_pad)
    # reference (vLLM semantics, fp32 o into the norm) on a dense copy
    st_ref = inp["state"].clone().contiguous()
    ref = None if max(idx) >= slots else qwen_gdn.gdn_decode_fused_torch(
        inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
        inp["norm_w"], st_ref, inp["state_idx"], H, K ** -0.5, 1e-6, act)
    # twin on flat buffers with the kernel's strides
    qkvz = inp["mixed_qkv"]
    row_stride = qkvz.stride(0)
    flat_qkvz = _bits(torch.as_strided(qkvz, (T, row_stride), (row_stride, 1)))
    z_off = inp["z"].storage_offset() - qkvz.storage_offset()
    page = inp["state"].stride(0)
    state_flat = torch.as_strided(inp["state"], (slots * page,), (1,)).numpy().copy()
    out = np.zeros(T * HV * K, dtype=np.uint16)
    ox.kgdn1_twin(out, flat_qkvz, flat_qkvz[z_off:], _bits(inp["ba"]),
                  inp["a_log"].numpy(), inp["dt_bias"].numpy(), inp["norm_w"].numpy(),
                  state_flat, inp["state_idx"].numpy(), T, slots, row_stride,
                  row_stride, page, act, K ** -0.5, 1e-6)
    got = torch.from_numpy(ox.bf16_to_f32(out).reshape(T, HV, K).copy())
    st_got = torch.from_numpy(state_flat.reshape(slots, page)[:, :HV * K * K].reshape(
        slots, HV, K, K).copy())
    return ref, got, st_ref, st_got


def _gate(ref, got, st_ref, st_got):
    ref16 = ref.to(torch.bfloat16).float()
    bad = (got - ref16).abs() > (qwen_gdn.OUT_ATOL + qwen_gdn.OUT_RTOL * ref16.abs())
    assert not bool(bad.any()), f"{int(bad.sum())} out elems off, max {(got - ref16).abs().max()}"
    torch.testing.assert_close(st_got, st_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("act", [0, 1])
def test_twin_matches_reference_serving_shape(act):
    _gate(*_run(3, [2, 5, 1], act, seed=act))


def test_twin_t16_null_slots_and_padded_pages():
    idx = [1, 0, 2, 3, -1, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]  # distinct live slots
    ref, got, st_ref, st_got = _run(16, idx, 0, page_pad=4096, seed=7)
    _gate(ref, got, st_ref, st_got)
    for t, s in enumerate(idx):
        if s <= 0:
            assert torch.count_nonzero(got[t]) == 0


def test_twin_traps_on_bad_slot():
    with pytest.raises(ox.Trap):
        _run(1, [9], 0)


def test_bf16_rne_matches_torch():
    x = torch.cat([torch.randn(4096) * 10, torch.tensor(
        [0.0, -0.0, 1e-40, 3.3895e38, float("inf"), -float("inf"), 1.00390625, 1.01171875])])
    want = x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    np.testing.assert_array_equal(ox.f32_to_bf16(x.numpy()), want)
    np.testing.assert_array_equal(ox.bf16_to_f32(want), x.to(torch.bfloat16).float().numpy())


def test_interface_json_matches_kernel_signature():
    iface = ox.interface()
    src = SRC.read_text()
    m = re.search(r"pub unsafe fn (\w+)\((.*?)\)\s*\{", src, re.S)
    assert m and m.group(1) == iface["entry"]
    names = [p.split(":")[0].strip() for p in m.group(2).split(",") if ":" in p]
    assert names == [p["name"] for p in iface["params"]]
    assert iface["block"] == [256, 1, 1] and "#[launch_bounds(256, 4)]" in src
    assert (iface["shape"]["H"], iface["shape"]["HV"], iface["shape"]["K"]) == (ox.H, ox.HV, ox.K)
