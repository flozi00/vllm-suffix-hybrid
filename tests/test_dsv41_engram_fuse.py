# SPDX-License-Identifier: Apache-2.0
"""dsv41_engram (helion_kernels/dsv41_engram.py): packed-row gather + post-exchange
dequant, bit-exact against the stock PACKED_FP4 decode (transcribed)."""
import json
import os
import re

os.environ.setdefault("HELION_INTERPRET", "1")

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid.helion_kernels import dsv41_engram as eg  # noqa: E402
from test_dsv41_stub import backends, run_body  # noqa: E402


def gather_args(t=5, rows=64, local_heads=3, head_start=3, total=24, seed=0, vocab=(64, 128)):
    d, s = eg.make_table(rows, seed)
    g = torch.Generator().manual_seed(seed + 9)
    ids = torch.randint(0, 3 * rows, (t, total), generator=g, dtype=torch.int32)
    for tok, val in ((0, vocab[0]), (1, vocab[1] - 1), (2, vocab[1])):   # first / last owned, one past
        if tok < t and head_start < total:
            ids[tok, head_start] = val
    return ids, d, s, vocab[0], vocab[1], head_start, local_heads, total


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("t,lh,hs", [(1, 3, 0), (5, 3, 3), (7, 3, 21), (4, 5, 20), (6, 3, 18)])
@pytest.mark.parametrize("br", [16, 32])
def test_gather_matches_reference(monkeypatch, backend, t, lh, hs, br):
    a = gather_args(t, local_heads=lh, head_start=hs)
    blocks = {0: br} if backend == "stub" else {"block_sizes": [br]}
    got = run_body(monkeypatch, eg, eg.dsv41_engram_gather_packed, a, blocks, backend)
    ref = eg.gather_reference(*a)
    assert torch.equal(got, ref)
    if hs + lh > 24:
        assert not got[:, 24 - hs:].any(), "padded heads of an uneven shard stay zero"
    assert got.shape == (t, lh, eg.REC)


def test_gather_ownership_edges():
    a = gather_args(4, head_start=0)
    ids, d, s = a[:3]
    ref = eg.gather_reference(*a)
    assert torch.equal(ref[0, 0, :eg.DATA], d[0]) and torch.equal(ref[1, 0, eg.DATA:], s[63])
    assert not ref[2, 0].any()


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("r,br", [(1, 1), (9, 4), (24, 8)])
def test_dequant_bit_exact(monkeypatch, backend, r, br):
    d, s = eg.make_table(r, seed=r)
    packed = torch.cat([d, s], 1).contiguous()
    blocks = {0: br} if backend == "stub" else {"block_sizes": [br]}
    got = run_body(monkeypatch, eg, eg.dsv41_engram_dequant_packed, (packed,), blocks, backend)
    ref = eg.dequant_reference(packed)
    assert got.dtype == torch.bfloat16 and got.shape == (r, eg.DIM)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_dequant_covers_every_code_and_edge_scales(monkeypatch):
    """all 256 byte values (both nibbles), scale bytes 0 / 1 / 127 / 254 / 255 (0 -> +0.0 and
    255 -> inf exactly as the stock bitcast)."""
    data = torch.cat([torch.arange(128, dtype=torch.uint8), torch.arange(128, 256, dtype=torch.uint8)]).view(2, 128)
    sc = torch.tensor([[0, 1, 127, 254, 255, 100, 130, 126], [127] * 8], dtype=torch.uint8)
    packed = torch.cat([data, sc], 1)
    got = run_body(pytest.MonkeyPatch(), eg, eg.dsv41_engram_dequant_packed, (packed,), {0: 2})
    ref = eg.dequant_reference(packed)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    # spot values: byte 0x21 = lo nibble 1 (0.5), hi nibble 2 (1.0) at scale 2^0
    one = torch.cat([torch.full((1, 128), 0x21, dtype=torch.uint8), torch.full((1, 8), 127, dtype=torch.uint8)], 1)
    out = run_body(pytest.MonkeyPatch(), eg, eg.dsv41_engram_dequant_packed, (one,), {0: 1})
    assert out[0, 0] == 0.5 and out[0, 1] == 1.0
    neg = torch.cat([torch.full((1, 128), 0xF7, dtype=torch.uint8), torch.full((1, 8), 128, dtype=torch.uint8)], 1)
    out = run_body(pytest.MonkeyPatch(), eg, eg.dsv41_engram_dequant_packed, (neg,), {0: 1})
    assert out[0, 0] == 12.0 and out[0, 1] == -12.0     # lo 0x7 = +6, hi 0xF = -6, scale 2^1


def test_roundtrip_gather_then_dequant_equals_stock_lookup(monkeypatch):
    """gather + all-gather + dequant == the stock lookup's bf16 rows (per head)."""
    a = gather_args(6, rows=64, local_heads=3, head_start=0, vocab=(0, 64), seed=3)
    a[0].clamp_(0, 63)
    packed = run_body(monkeypatch, eg, eg.dsv41_engram_gather_packed, a, {0: 16})
    rows = run_body(monkeypatch, eg, eg.dsv41_engram_dequant_packed,
                    (packed.reshape(-1, eg.REC),), {0: 4}).view(6, 3, eg.DIM)
    d, s = a[1], a[2]
    for t in range(6):
        for h in range(3):
            i = int(a[0][t, h])
            want = eg.dequant_reference(torch.cat([d[i], s[i]]).view(1, -1))[0]
            assert torch.equal(rows[t, h].view(torch.int16), want.view(torch.int16))


def test_wire_bytes_derivation():
    assert eg.REC == 136 and eg.wire_bytes(1, packed=False) == 24 * 512
    ratio = eg.wire_bytes(192, packed=False) / eg.wire_bytes(192)
    assert abs(ratio - 512 / 136) < 1e-9 and 3.7 < ratio < 3.8
    assert eg.wire_bytes(192, packed=False) == 2359296       # 2.36 MB at c32 (192 tokens)


def test_cases_and_abi():
    model = {"architecture": "DeepseekV41ForCausalLM", "tp": 8}
    g = eg.cases_gather(model)
    assert [c["numtokens"] for c in g] == list(eg.BUCKETS) and {c["local_heads"] for c in g} == {3}
    assert {c["local_heads"] for c in eg.cases_gather(dict(model, tp=5))} == {5}
    assert {c["local_heads"] for c in eg.cases_dequant(model)} == {24}
    json.dumps(g)
    assert eg.cases_gather({"architecture": "Qwen3ForCausalLM"}) == []
    for ent in eg.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", ent["name"]) and ent["body"].__module__ == eg.__name__
        assert len(ent["replaces"]) <= 16
    args = eg.make_gather(eg.case(4), 0, device="cpu")
    ref = eg.gather_reference(*args)
    assert ref.shape == (4, 3, eg.REC) and ref.any(-1).all(), "the lab problem must exercise owned rows"
    (packed,) = eg.make_dequant(eg.case(2, 24), 0, device="cpu")
    assert eg.dequant_reference(packed).shape == (48, eg.DIM)


def test_bodies_lower_to_triton():
    from test_dsv41_stub import lower_to_triton
    g = lower_to_triton(eg.dsv41_engram_gather_packed, eg.make_gather(eg.case(4), 0, device="cpu"),
                        eg.default_config({}))
    assert "tl.load(w_data" in g and "tl.store(pk_d" in g and "tl.store(pk_s" in g
    (p,) = eg.make_dequant(eg.case(2, 24), 0, device="cpu")
    d = lower_to_triton(eg.dsv41_engram_dequant_packed, (p,), eg.default_config({}))
    assert "tl.join" in d
