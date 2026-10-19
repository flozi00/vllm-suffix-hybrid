# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 sparse-MLA attention, CPU-only: the K1 record contract
(byte-level), the row views the kernels address the pools through, the exact
reference (sink / empty / duplicates / lens semantics), FlashInfer layout
interop, the cost + capacity model, the lab ABI, the SUFFIX_HELION_DSV41_ATTN
knob (partial + merge + row routing, context classes), the capture-safety
rules (no barriers / atomics, no allocation on the call path, no compile
inside a capture) and the K5 hook (anchors, shim dispatch).
The Helion bodies and the boot gate run in tests/test_dsv41_attn_helion.py."""
import ast
import math
import re
import types
from pathlib import Path

import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_attn as D
from suffix_hybrid.helion_kernels import dsv41_attn_lab as L
from suffix_hybrid.helion_kernels import dsv41_attn_ref as R
from suffix_hybrid.kernels import dsv41_attn as K
from suffix_hybrid.kernels import dsv41_attn_capture as G

REPO = Path(__file__).resolve().parents[1]


def rows(n, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, R.HD, generator=g) * scale).bfloat16()


def pool(rec, page, x, pad=0, fill=0):
    flat, stride = R.make_pool(rec, page, x, pad=pad, fill=fill)
    return flat, rec, page, stride


# ---- K1: record bytes -------------------------------------------------------
def test_block_strides_are_whole_granules():
    assert R.block_stride(R.REC_V4, 32) == 33 * 576 == 19008          # vLLM: round_up(32 * 584, 576)
    assert R.block_stride(R.REC_V4, 64) == 65 * 576 == 37440
    assert R.block_stride(R.REC_FP8, 32) == 32 * 528 and R.block_stride(R.REC_FP4, 32) == 32 * 288
    assert R.block_stride(R.REC_V4, 32, 2) == 35 * 576
    for rec in (R.REC_V4, R.REC_FP8, R.REC_FP4):
        for page in (16, 32, 64):
            assert R.block_stride(rec, page) % R.STRIDE_GRANULE[rec] == 0
            assert R.block_stride(rec, page) >= R.block_bytes(rec, page)


def test_v4_record_bytes():
    """584 B/token: [page*576 data | page*8 scales | pad]; 448 fp8 + 128 B bf16 RoPE; 7 UE8M0 + 1 zero."""
    page = 32
    x = rows(70, 1)
    x[37, :64] = 1.0                                   # amax 1 -> exponent ceil(log2(1/448)) = -8 -> byte 119
    flat, stride = R.make_pool(R.REC_V4, page, x, pad=1, fill=0xAB)
    assert stride == 34 * 576 and flat.numel() == 3 * stride
    slot = 37                                          # block 1, token 5
    base = 1 * stride
    data, scale = base + 5 * 576, base + page * 576 + 5 * 8
    assert flat[scale].item() == 127 - 8 and flat[scale + 7].item() == 0
    assert bytes(flat[data + 448: data + 576].tolist()) == x[slot, 448:].contiguous().view(torch.uint8).numpy().tobytes()
    assert flat[data].view(torch.float8_e4m3fn).float().item() == 1.0 * 2.0 ** 8   # 1.0 / 2^-8 = 256
    assert bool((flat[base + page * 584: base + stride] == 0xAB).all())            # the pad keeps the fill


def test_fp8_record_bytes():
    """528 B/token interleaved: 512 fp8 then 16 UE8M0 (one per 32 dims, RoPE included)."""
    page = 32
    x = rows(40, 2)
    x[35, 32:64] = 2.0 ** 10
    flat, stride = R.make_pool(R.REC_FP8, page, x)
    base = 1 * stride + 3 * 528                        # slot 35 = block 1 token 3
    assert flat[base + 512 + 1].item() == 127 + math.ceil(math.log2(2.0 ** 10 / 448))
    assert flat[base + 32].view(torch.float8_e4m3fn).float().item() == 2.0 ** 10 / 2.0 ** math.ceil(math.log2(2.0 ** 10 / 448))


def test_fp4_record_bytes_and_nibble_order():
    """288 B/token interleaved: 256 e2m1 pairs (even dim = low nibble) then 32 e4m3 per 16 dims."""
    x = torch.zeros(1, R.HD)
    x[0, 0], x[0, 1] = 3.0, -6.0                       # tile 0 amax 6 -> scale 1.0
    flat, stride = R.make_pool(R.REC_FP4, 32, x.bfloat16())
    b0 = int(flat[0])
    assert b0 & 15 == 5 and b0 >> 4 == 8 + 7           # 3.0 = code 5; -6 = sign + code 7
    assert flat[256].view(torch.float8_e4m3fn).float().item() == 1.0
    assert flat[257].view(torch.float8_e4m3fn).float().item() == 2.0 ** -9      # all-zero tile: floor 2^-9
    d = R.dequant_rows(flat, R.REC_FP4, 32, stride, torch.tensor([0]))
    assert d[0, 0].item() == 3.0 and d[0, 1].item() == -6.0


def test_e2m1_round_to_nearest_even_and_saturate():
    v = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 6.0, 100.0, -0.25, -5.01, -1e9])
    mag = R._e2m1_mag(R.e2m1_code(v)).tolist()
    assert mag == [0, 0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, -0.0, -6.0, -6.0]


@pytest.mark.parametrize("rec,bound", [(R.REC_V4, 0.04), (R.REC_FP8, 0.04), (R.REC_FP4, 0.12)])
def test_quantization_error_budget(rec, bound):
    """The cache is lossy by design; the reader must be exact on whatever bytes are there."""
    x = rows(64, 3)
    flat, stride = R.make_pool(rec, 32, x)
    d = R.dequant_rows(flat, rec, 32, stride, torch.arange(64))
    err = ((d - x.double()).norm(dim=-1) / x.double().norm(dim=-1)).max().item()
    assert err < bound, err


def test_footer_conversion_matches_flashinfer_layout():
    """FlashInfer 0.7.2 packs V4.1 records as page footers (data of the whole page, then the scales)."""
    x = rows(70, 4)
    for rec, dw in ((R.REC_FP8, 512), (R.REC_FP4, 256)):
        flat, stride = R.make_pool(rec, 32, x, pad=1)
        foot = R.to_flashinfer_footer(flat, rec, 32, stride)
        for slot in (0, 5, 33, 69):
            b, r = divmod(slot, 32)
            inter = flat[b * stride + r * R.ROW_BYTES[rec]: b * stride + (r + 1) * R.ROW_BYTES[rec]]
            data = foot[b * stride + r * dw: b * stride + (r + 1) * dw]
            sw = R.ROW_BYTES[rec] - dw
            sc = foot[b * stride + 32 * dw + r * sw: b * stride + 32 * dw + (r + 1) * sw]
            assert torch.equal(data, inter[:dw]) and torch.equal(sc, inter[dw:])
        assert torch.equal(foot.view(-1, stride)[:, 32 * R.ROW_BYTES[rec]:], flat.view(-1, stride)[:, 32 * R.ROW_BYTES[rec]:])
    flat, stride = R.make_pool(R.REC_V4, 32, x)
    assert R.to_flashinfer_footer(flat, R.REC_V4, 32, stride) is flat  # V4 584 IS FlashInfer's DSV4 layout


# ---- the row views the kernels address (K1 at the call boundary) -------------
def vllm_cache(rec, page, x, pad=1):
    flat, rec, page, stride = pool(rec, page, x, pad)
    return D.vllm_view_of(flat, rec, page, stride), flat, stride


DUMMIES = {"d16": torch.zeros(1, 1, dtype=torch.int16), "s": torch.zeros(1, 8, dtype=torch.uint8)}


def test_row_views_address_the_records():
    x = rows(100, 5)
    for rec in (R.REC_V4, R.REC_FP8, R.REC_FP4):
        cache, flat, stride = vllm_cache(rec, 32, x)
        d, d16, s, shift, bd, bs, got_rec, got_flat = D.pool_args(cache, "t", DUMMIES)
        assert got_rec == rec and shift == 5 and d.data_ptr() == cache.data_ptr()   # views, never copies
        assert D.pool_args(cache, "t", DUMMIES)[0] is d                              # cached: stable tensors
        assert torch.equal(D.flat_of(d, s, rec), got_flat) and got_flat.data_ptr() == flat.data_ptr()
        for slot in (0, 7, 31, 32, 63, 99):
            blk, r = divmod(slot, 32)
            w = R.ROW_BYTES[rec] if rec != R.REC_V4 else 576
            row = d[blk * bd + r]
            if rec == R.REC_V4:
                assert bd == stride // 576 and bs == stride // 8
                assert torch.equal(row, flat[blk * stride + r * 576: blk * stride + (r + 1) * 576])
                assert torch.equal(s[blk * bs + (72 << shift) + r], flat[blk * stride + 32 * 576 + r * 8: blk * stride + 32 * 576 + (r + 1) * 8])
                assert torch.equal(d16[blk * bd + r, 224:].view(torch.bfloat16), x[slot, 448:])   # RoPE as 64 int16
            else:
                assert bd == stride // w and torch.equal(row, flat[blk * stride + r * w: blk * stride + (r + 1) * w])


def test_pool_view_geometry_and_errors():
    x = rows(100, 6)
    cache, flat, stride = vllm_cache(R.REC_V4, 32, x)
    f, rec, pg, st = D.pool_view(cache, "t")
    assert (rec, pg, st) == (R.REC_V4, 32, stride) and f.numel() == (cache.shape[0] - 1) * stride + 32 * 584
    assert D.pool_view(cache.squeeze(2), "t")[3] == stride                      # 3-D form too
    for bad in (flat.as_strided((4, 32, 1, 300), (stride, 300, 300, 1)),         # unknown record
                flat.as_strided((4, 24, 1, 584), (stride, 584, 584, 1)),         # 24-token page
                flat.as_strided((4, 32, 1, 584), (32 * 584 - 8, 584, 584, 1)),   # overlapping blocks
                flat.as_strided((4, 32, 1, 584), (stride - 8, 584, 584, 1)),     # stride not a whole row (8 B off)
                flat.as_strided((3, 32, 1, 584), (stride + 1, 584, 584, 1))):               # 1 B off a whole row
        with pytest.raises(RuntimeError, match="dsv41_attn"):
            D.pool_view(bad, "t")


# ---- K3: the reference ------------------------------------------------------
def naive(q, pool_rows, sink, scale=R.SCALE):
    """Independent formulation: softmax over [scores..., sink] with a zero value for the sink."""
    s = scale * (q.double() @ pool_rows.T)
    full = torch.cat([s, sink.double()[:, None]], 1)
    p = torch.softmax(full, 1)
    return p[:, :-1] @ pool_rows


def test_reference_sink_and_listing_semantics():
    x = rows(60, 5)
    swa = pool(R.REC_V4, 32, x)
    flat, rec, page, stride = swa
    q = rows(3 * 4, 6).reshape(3, 4, R.HD)
    sink = torch.tensor([-math.inf, 0.0, 2.0, -1.0])
    idx = torch.tensor([[5, 7, 9, -1, 11, 7], [3, -1, -1, -1, -1, -1], [1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    ln = torch.tensor([6, 6, 3], dtype=torch.int32)    # row 2: len 3 ignores slots 4..6
    out = R.attention_ref(q, swa, None, idx, ln, sink=sink)
    deq = lambda sl: R.dequant_rows(flat, rec, page, stride, torch.tensor(sl))   # noqa: E731
    assert torch.allclose(out[0], naive(q[0], deq([5, 7, 9, 11, 7]), sink), atol=1e-12)   # -1 skipped, 7 counted twice
    assert torch.allclose(out[1], naive(q[1], deq([3]), sink), atol=1e-12)
    assert torch.allclose(out[2], naive(q[2], deq([1, 2, 3]), sink), atol=1e-12)
    # FlashMLA convention: with-sink output = no-sink output * sigmoid(lse - sink)
    nos = R.attention_ref(q[:1], swa, None, idx[:1], ln[:1])
    lse = torch.logsumexp(R.SCALE * (q[0].double() @ deq([5, 7, 9, 11, 7]).T), 1)
    gate = torch.sigmoid(lse - sink.double())
    assert torch.allclose(out[0, 1:], nos[0, 1:] * gate[1:, None], atol=1e-12)
    assert torch.equal(out[0, 0], nos[0, 0])           # -inf sink = no sink


def test_reference_two_lists_share_one_softmax():
    swa, cmp = pool(R.REC_V4, 32, rows(40, 8)), pool(R.REC_FP4, 32, rows(40, 9))
    q = rows(4, 10).reshape(1, 4, R.HD)
    si, ci = torch.tensor([[1, 2, 3]], dtype=torch.int32), torch.tensor([[4, 5]], dtype=torch.int32)
    ln = torch.tensor([3], dtype=torch.int32)
    out = R.attention_ref(q, swa, cmp, si, ln, ci, torch.tensor([2], dtype=torch.int32), sink=None)
    k = torch.cat([R.dequant_rows(swa[0], R.REC_V4, 32, swa[3], torch.tensor([1, 2, 3])),
                   R.dequant_rows(cmp[0], R.REC_FP4, 32, cmp[3], torch.tensor([4, 5]))])
    assert torch.allclose(out[0], naive(q[0], k, torch.full((4,), -math.inf)), atol=1e-12)


def test_reference_empty_rows_are_exactly_zero_even_with_sink():
    x = rows(40, 7)
    p = pool(R.REC_FP8, 32, x)
    q = rows(8, 8).reshape(2, 4, R.HD)
    idx = torch.full((2, 5), -1, dtype=torch.int32)
    idx[1, :3] = 4
    out = R.attention_ref(q, p, None, idx, torch.tensor([5, 0], dtype=torch.int32), sink=torch.zeros(4))
    assert bool((out == 0).all())                      # all -1, and live-looking slots past len 0


def test_check_is_the_reader_gate():
    ref = torch.randn(3, 2, R.HD, dtype=torch.float64)
    ref[1] = 0
    out = ref.bfloat16().clone()
    assert R.check(out, ref)[0]
    bad = out.clone(); bad[1, 0, 0] = 1e-3
    assert not R.check(bad, ref)[0]                    # empty row must be exactly 0
    bad = out.clone(); bad[0, 1, 3] = float("nan")
    assert not R.check(bad, ref)[0]
    assert not R.check(out * 1.05, ref)[0]             # 5e-2 > 2e-2


def test_merge_ref_is_the_logsumexp_of_the_partials():
    """The maths dsv41_attn_merge implements, against an independent single softmax."""
    g = torch.Generator().manual_seed(3)
    t, h, s, cols = 3, 4, 3, 4
    sc = torch.randn(t, h, s, cols, generator=g, dtype=torch.float64) * 2        # logits (already scaled), natural log
    v = torch.randn(t, h, s, cols, 512, generator=g, dtype=torch.float64)
    sink = torch.tensor([-math.inf, 0.0, 1.5, -2.0], dtype=torch.float64)
    sc[0, :, 1] = -math.inf                                                         # a split with nothing live
    sc[2] = -math.inf                                                               # token 2: nothing live at all
    w = torch.softmax(sc, -1).nan_to_num(0.0)
    o_part = (w[..., None] * v).sum(3)                                              # normalized per-split output
    lse2 = torch.logsumexp(sc, -1) * R.LOG2E                                        # base-2 lse; -inf for empty splits
    got = D.merge_ref(o_part.bfloat16(), lse2.float(), sink.float()).double()
    flat_sc = sc.reshape(t, h, s * cols)
    flat_v = v.reshape(t, h, s * cols, 512)
    full = torch.cat([flat_sc, sink[None, :, None].expand(t, h, 1)], -1)
    p = torch.softmax(full, -1).nan_to_num(0.0)[..., :-1]
    want = (p[..., None] * flat_v).sum(2)
    live = want.abs().sum(-1) > 0
    assert torch.allclose(got[live], want[live], rtol=2e-2, atol=2e-2)
    assert bool((got[2] == 0).all())                                                 # empty token: exact 0, even with a sink


# ---- model ------------------------------------------------------------------
def test_cost_and_capacity_model_numbers():
    assert R.COMPRESS_RATIOS.count(2) == 18 and R.COMPRESS_RATIOS.count(1) == 20 and R.COMPRESS_RATIOS.count(0) == 2
    per_row = R.decode_bytes_per_row(R.REC_V4, R.REC_V4)
    assert per_row == 38 * (128 + 512) * 584 + 2 * 128 * 584 + 3 * 133 * 584 == 14_585_400
    nv = R.decode_bytes_per_row(R.REC_FP8, R.REC_FP4)
    assert nv == 38 * (128 * 528 + 512 * 288) + 2 * 128 * 528 + 3 * 133 * 528 == 8_517_360
    m = R.decode_step_model(overlap=1.0)
    assert m["dram_bytes"] == 32 * per_row and m["l2_bytes"] == 192 * per_row
    assert R.decode_step_model(overlap=0.0)["dram_bytes"] == 192 * per_row
    assert R.kv_bytes_per_ctx_token(R.REC_V4) == 2.5 * (584 + 132) == 1790
    assert R.kv_bytes_per_ctx_token(R.REC_FP4) == 2.5 * (288 + 132) == 1050
    assert R.kv_bytes_per_ctx_token(R.REC_FP4, 68) == 890


def test_overlap_stats():
    idx = torch.arange(24, dtype=torch.int32).reshape(6, 4).repeat(2, 1)   # rows 0..5 distinct, 6..11 same
    same = torch.arange(4, dtype=torch.int32).repeat(12, 1)
    assert R.overlap_stats(same)["union_over_total"] == pytest.approx(1 / 6)
    assert R.overlap_stats(idx)["union_over_total"] == pytest.approx(1.0)
    assert R.overlap_stats(idx)["groups"] == 2


# ---- lab ABI ----------------------------------------------------------------
MODEL = {"name": "glm", "architecture": "DeepseekV41ForCausalLM", "heads": 8, "tp": 8}


def test_lab_abi():
    assert [k["name"] for k in D.KERNELS] == [D.PARTIAL, D.ROW]
    expect = {D.PARTIAL: 2 * (14 * 3 + 3 + 17 + 17), D.ROW: 2 * (14 * 3 + 6 + 20 + 20)}
    for k in D.KERNELS:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"])
        for key in ("body", "cases", "make", "reference", "baseline", "check"):
            assert callable(k[key]), key
        assert k["settings"]["autotune_baseline_fn"] is not None and k["settings"]["index_dtype"] is torch.int64
        cs = k["cases"](MODEL)
        assert len(cs) == expect[k["name"]] and len({hk.case_key(c) for c in cs}) == len(cs)
        assert all(set(c) == {"numtokens", "heads", "swa", "topk", "rec", "ctx"} for c in cs)
        assert {(c["swa"], c["topk"]) for c in cs} == set(D.LAYER_SHAPES)
        # context classes only on decode buckets of compressed layers
        assert {c["ctx"] for c in cs if c["topk"] and c["numtokens"] <= 192} == {"c16k", "c128k", "c512k"}
        assert {c["ctx"] for c in cs if not c["topk"] or c["numtokens"] > 192} == {"any"}
        assert k["cases"]({**MODEL, "architecture": "GlmMoeDsaForCausalLM"}) == []
        assert k["cases"]({**MODEL, "heads": None}) == []
    assert D.DECODE_BUCKETS[-1] == 192 and D.PARTIAL_BUCKETS[-1] == 1024 and D.ROW_BUCKETS[-1] == 8192
    assert 6 in D.DECODE_BUCKETS and set(D.ROW_BUCKETS) >= {1 << k for k in range(8, 14)}
    # the merge stage is NOT a lab entry: a fixed-config kernel shipped in code
    assert D.MERGE not in [k["name"] for k in D.KERNELS]


def test_lab_plan_is_cuda_free(capsys):
    assert L.main(["--plan", "--buckets", "6,256", "--shapes", "128:512", "--ctx", "c512k"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4                              # partial: 6 (c512k), 256 (any); row: 6 (c512k), 256 (any)
    cases = L.plan_cases(L.parse(["--kernels", "row", "--buckets", "6,256", "--shapes", "128:512,192:0",
                                  "--rec", "v41", "--ctx", "c16k,c512k"]))
    assert [(c["numtokens"], c["ctx"], c["topk"]) for _, c in cases] == [
        (6, "c16k", 512), (6, "c512k", 512), (6, "any", 0), (256, "any", 512), (256, "any", 0)]
    assert all(c["rec"] == "v41" for _, c in cases)
    assert L.to_config(D.PARTIAL, {"bk": 32, "num_warps": 4, "num_stages": 2, "num_splits": 8}) == \
        {"block_sizes": [32], "num_warps": 4, "num_stages": 2, "num_splits": 8}
    assert L.to_config(D.ROW, {"bk": 16, "num_warps": 8, "num_stages": 1}) == \
        {"block_sizes": [16], "num_warps": 8, "num_stages": 1}
    assert len(L.configs_for(D.PARTIAL, True)) == 6 * 3 * 2 * 3 and L.configs_for(D.ROW, False) == [None]


def test_lab_overlap_probe_cli(tmp_path, capsys):
    idx = torch.arange(24, dtype=torch.int32).reshape(6, 4).repeat(2, 1)
    f = tmp_path / "topk.pt"
    torch.save(idx, f)
    assert L.main(["--overlap", str(f)]) == 0
    out = capsys.readouterr().out
    assert '"rows": 12' in out and '"groups": 2' in out and '"union_over_total": 1.0' in out


def test_check_skips_unreferenced_rows_and_merges_partials():
    ref = torch.full((4, 2, R.HD), float("nan"), dtype=torch.float64)
    ref[1] = torch.randn(2, R.HD, dtype=torch.float64)
    out = torch.randn(4, 2, R.HD).bfloat16()
    out[1] = ref[1].bfloat16()
    assert D._check(out, ref)[0]                        # rows 0, 2, 3 are NaN in ref: not compared
    out[1, 0, 0] += 1.0
    assert not D._check(out, ref)[0]
    # the partial body returns (o_part, lse_part, sink): _check merges it before comparing
    o = torch.zeros(1, 2, 1, R.HD)
    o[0, :, 0] = ref[1].float()
    ok, _ = D._check((o.bfloat16(), torch.zeros(1, 2, 1), torch.full((2,), -math.inf)), ref[1:2])
    assert ok


def test_tune_check_vs_flashinfer():
    ref = torch.randn(3, 2, R.HD)
    ref[1] = 0
    D._tune_check(ref.bfloat16(), ref)
    D._tune_check((ref * 1.05).bfloat16(), ref)         # fp8 q/P baseline noise is tolerated
    for bad in (ref * 1.3, ref + (ref == 0) * 1e-3):
        with pytest.raises(AssertionError):
            D._tune_check(bad.bfloat16(), ref)


# ---- capture-safety rules ----------------------------------------------------
FORBIDDEN = ("barrier", "atomic", "semaphore", "spin", "cooperative", "persistent")


def test_kernels_have_no_cross_cta_synchronisation():
    """Owner rule: no grid barrier (hl.barrier -> persistent kernel + in-kernel barrier: deadlocked on
    silicon), no atomics / semaphores / last-arriver merge. Stage dependencies are separate launches."""
    src = REPO / "suffix_hybrid/helion_kernels/dsv41_attn.py"
    tree = ast.parse(src.read_text())
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
             {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not {name for name in called if any(f in name.lower() for f in FORBIDDEN)}, called
    for fn in tree.body:                               # the Helion bodies themselves
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("dsv41_attn_"):
            tops = [n for n in fn.body if isinstance(n, ast.For)]
            assert len(tops) == 1, (fn.name, len(tops))   # ONE kernel per Helion function: stages are functions
    assert "pid_type" not in src.read_text()           # no persistent pid type forced anywhere


def test_split_path_is_two_helion_functions():
    assert D.dsv41_attn_partial.__name__ == "dsv41_attn_partial" and D.dsv41_attn_merge.__name__ == "dsv41_attn_merge"
    assert D._BODY == {D.PARTIAL: D.dsv41_attn_partial, D.ROW: D.dsv41_attn_row}
    # the partial function allocates its own scratch (torch's allocator) and returns it
    src = (REPO / "suffix_hybrid/helion_kernels/dsv41_attn.py").read_text()
    assert "o_part = torch.empty(" in src and "return o_part, lse_part, sink" in src


# ---- serving: knob + routing ------------------------------------------------
@pytest.fixture
def knob(monkeypatch):
    """Configs shipped for `tuned[kernel]` buckets (any context class); compiled() -> recording fakes."""
    calls, verified, tuned, parts = [], [], {D.PARTIAL: set(), D.ROW: set()}, {}
    monkeypatch.setattr(hk, "config", lambda k, c, plat=None: (
        {"num_splits": 4} if c["numtokens"] in tuned[k] else None))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")

    def compiled(body, kernel, case, **kw):
        assert body is D._BODY[kernel] and kw == D.SETTINGS

        def fake(*a):
            calls.append((kernel, case["numtokens"], case["ctx"], a))
            if kernel == D.PARTIAL:
                t, h = a[0].shape[:2]
                if (t, h) not in parts:                      # (cached: the allocation test patches torch.zeros)
                    parts[(t, h)] = (torch.zeros(t, h, 4, R.HD, dtype=torch.bfloat16), torch.zeros(t, h, 4))
                return (*parts[(t, h)], a[11])
            return a[12]
        return fake
    monkeypatch.setattr(hk, "compiled", compiled)
    monkeypatch.setattr(D, "merge_kernel", lambda: (lambda *a: calls.append(("merge", a[1].shape[2], a)) or a[3]))
    monkeypatch.setattr(D, "warm_and_verify", lambda route, dev: verified.append(
        {k: sorted(v) for k, v in route["kernels"].items()}))
    monkeypatch.setattr(D, "_VERIFIED", set())
    monkeypatch.setattr(D, "_WARM", set())
    monkeypatch.setattr(D, "helion", D.helion or object())
    monkeypatch.delenv(D.KNOB, raising=False)
    monkeypatch.delenv(D.ROUTE_ENV, raising=False)
    return types.SimpleNamespace(calls=calls, verified=verified, tuned=tuned, mp=monkeypatch)


def views_of(t, swa_w=128, topk=512, rec=R.REC_V4, page=32, h=8):
    stride = R.block_stride(rec, page, 1)
    raw = torch.zeros(4 * stride, dtype=torch.uint8)
    pl = raw.as_strided((4, page, 1, R.ROW_BYTES[rec]), (stride, R.ROW_BYTES[rec], R.ROW_BYTES[rec], 1))
    q = torch.zeros(t, h, R.HD, dtype=torch.bfloat16)
    return dict(q=q, swa_cache=pl, swa_idx=torch.zeros(t, 1, swa_w, dtype=torch.int32),
                swa_len=torch.zeros(t, dtype=torch.int32),
                cmp_cache=pl if topk else None,
                cmp_idx=torch.zeros(t, 1, topk, dtype=torch.int32) if topk else None,
                cmp_len=torch.zeros(t, dtype=torch.int32) if topk else None,
                sink=torch.zeros(h), out=torch.empty_like(q))


def test_knob_oxide_auto_helion(knob):
    cpu = torch.device("cpu")
    assert D.load(8, 128, 512, 0, 0, cpu) is None                      # default oxide: nothing compiled
    knob.mp.setenv(D.KNOB, "auto")
    assert D.load(8, 128, 512, 0, 0, cpu) is None and knob.verified == []   # nothing tuned
    knob.tuned[D.PARTIAL].update({6, 32})
    knob.tuned[D.ROW].update({256, 8192})
    route = D.load(8, 128, 512, 0, 0, cpu)
    assert sorted({b for b, _ in route["kernels"][D.PARTIAL]}) == [6, 32]
    assert sorted({b for b, _ in route["kernels"][D.ROW]}) == [256, 8192]
    assert {c for b, c in route["kernels"][D.PARTIAL]} == {"c16k", "c128k", "c512k"}   # decode: per context class
    assert {c for b, c in route["kernels"][D.ROW]} == {"any"}
    assert len(knob.verified) == 1 and route["merge"] is not None
    D.load(8, 128, 512, 0, 0, cpu)
    assert len(knob.verified) == 1                                     # once per device+geometry, not per layer
    # dispatch: T=5 -> bucket 6 partial+merge (merge after, the partials' S passed on)
    kw = views_of(5)
    assert D.attend(route, **kw) is kw["out"]
    kern, b, cx, a = knob.calls[0]
    st = R.block_stride(R.REC_V4, 32, 1)
    assert (kern, b, cx) == (D.PARTIAL, 6, "c512k") and len(a) == 20 and a[11] is kw["sink"]
    assert a[12:] == (5, st // 576, st // 8, 5, st // 576, st // 8, 0, 0)
    assert knob.calls[1][0] == "merge" and knob.calls[1][1] == 4       # the fake partial wrote S = 4 splits
    # context classes: a c16k step uses the c16k config of the same bucket
    assert D.attend(route, **views_of(6), ctx="c16k") is not None and knob.calls[-2][2] == "c16k"
    # T=200 > 192 -> the row body at bucket 256 (no merge)
    n = len(knob.calls)
    assert D.attend(route, **views_of(200)) is not None
    assert knob.calls[n][:3] == (D.ROW, 256, "any") and len(knob.calls) == n + 1 and len(knob.calls[n][3]) == 21
    assert D.attend(route, **views_of(300)) is None                    # bucket 512 untuned -> FlashInfer
    assert D.attend(route, **views_of(3)) is None                      # bucket 4 untuned (auto)
    assert D.attend(route, **views_of(6, h=4)) is None                 # a head count the route was not loaded for
    for bad in (dict(swa_len=torch.zeros(5, dtype=torch.int64)), dict(q=torch.zeros(5, 8, R.HD, dtype=torch.float16)),
                dict(sink=torch.zeros(8, dtype=torch.bfloat16))):
        v = {**views_of(5), **bad}
        v["out"] = torch.empty_like(v["q"])
        with pytest.raises(RuntimeError, match="must be bf16"):
            D.attend(route, **v)
    knob.mp.setenv(D.KNOB, "helion")
    with pytest.raises(RuntimeError, match="no tuned dsv41_attn_partial/dsv41_attn_row config"):
        D.load(8, 128, 512, 0, 0, cpu)
    knob.tuned[D.PARTIAL].update(D.PARTIAL_BUCKETS)
    knob.tuned[D.ROW].update(D.ROW_BUCKETS)
    knob.mp.setattr(D, "_VERIFIED", set())
    route = D.load(8, 128, 512, 0, 0, cpu)
    assert D.attend(route, **views_of(3)) is not None
    with pytest.raises(RuntimeError, match="helion: no dsv41 attention kernel"):
        D.attend(route, **views_of(6, h=4))
    knob.mp.setenv(D.KNOB, "triton")
    with pytest.raises(RuntimeError, match="expected oxide"):
        D.load(8, 128, 512, 0, 0, cpu)


def test_pick_context_class_fallback(knob):
    cpu = torch.device("cpu")
    knob.mp.setenv(D.KNOB, "auto")
    have = {"cls": ("c16k", "c512k")}
    knob.mp.setattr(hk, "config", lambda k, c, plat=None: {"num_splits": 4} if (
        c["numtokens"] == 6 and k == D.PARTIAL and c["ctx"] in have["cls"]) else None)
    for cls in (("c16k", "c512k"), ("c128k",), ("c16k",)):
        have["cls"] = cls
        knob.mp.setattr(D, "_VERIFIED", set())
        route = D.load(8, 128, 512, 0, 0, cpu)
        assert all(D.pick(route, 6, cx) is not None for cx in ("c16k", "c128k", "c512k"))   # always SOME tuned class
    # preference: same class, then LARGER (more splits are the safe side), then smaller
    have["cls"] = ("c16k", "c512k")
    knob.mp.setattr(D, "_VERIFIED", set())
    route = D.load(8, 128, 512, 0, 0, cpu)
    cfg = {cx: route["kernels"][D.PARTIAL][(6, cx)] for cx in ("c16k", "c512k")}
    assert D.pick(route, 6, "c128k")[1] is cfg["c512k"] and D.pick(route, 6, "c16k")[1] is cfg["c16k"]
    assert D.pick(route, 7, "c512k") is None                            # untuned bucket
    assert D.ctx_class(None) == "c512k" and D.ctx_class(100) == "c16k" and D.ctx_class(16384) == "c16k"
    assert D.ctx_class(16385) == "c128k" and D.ctx_class(524288) == "c512k"
    assert D.case_ctx(6, 512, "c16k") == "c16k" and D.case_ctx(6, 0, "c16k") == "any"
    assert D.case_ctx(256, 512, "c16k") == "any"


def test_route_row_and_geometry_checks(knob):
    cpu = torch.device("cpu")
    knob.mp.setenv(D.KNOB, "helion")
    knob.tuned[D.ROW].update(D.ROW_BUCKETS)                            # only the one-kernel body tuned
    r0 = D.load(8, 128, 512, 0, 0, cpu)                                 # fine: row covers every decode bucket too
    assert D.pick(r0, 6)[0] == D.ROW and r0["merge"] is None
    knob.tuned[D.ROW].discard(6)
    with pytest.raises(RuntimeError, match=r"no tuned .* T-buckets \[6\]"):
        D.load(8, 128, 512, 0, 0, cpu)                                  # a bucket with neither body: startup error
    knob.tuned[D.ROW].add(6)
    knob.mp.setenv(D.ROUTE_ENV, "row")
    knob.tuned[D.PARTIAL].update(D.PARTIAL_BUCKETS)
    knob.mp.setattr(D, "_VERIFIED", set())
    route = D.load(8, 128, 512, 0, 0, cpu)                              # row forced: the partial path never looked at
    assert route["kernels"][D.PARTIAL] == {} and route["merge"] is None
    D.attend(route, **views_of(6))
    assert knob.calls[-1][0] == D.ROW
    for bad in (dict(topk=0), dict(swa_w=192), dict(rec=R.REC_FP8), dict(page=64)):
        with pytest.raises(RuntimeError, match="differ from the loaded route"):
            D.attend(route, **views_of(6, **bad))
    # the ratio-1 layers' compressed pool has 64-token blocks: a route loaded for it takes them (SWA stays 32)
    knob.mp.setattr(D, "_VERIFIED", set())
    r64 = D.load(8, 128, 512, 0, 0, cpu, 64)
    assert r64["pages"] == (32, 64)
    v = views_of(6)
    v["cmp_cache"] = views_of(6, page=64)["cmp_cache"]
    assert D.attend(r64, **v) is v["out"]
    with pytest.raises(RuntimeError, match="differ from the loaded route"):
        D.attend(r64, **views_of(6))                                    # both pools 32: the compressed page is wrong
    knob.mp.setenv(D.ROUTE_ENV, "split")
    with pytest.raises(RuntimeError, match="expected auto | row"):
        D.load(8, 128, 512, 0, 0, cpu)


def test_ratio0_and_draft_routes_have_their_own_config_keys(knob):
    seen = []
    knob.mp.setattr(hk, "config", lambda k, c, plat=None: seen.append(c) or None)
    knob.mp.setenv(D.KNOB, "auto")
    for sw, tk in D.LAYER_SHAPES:
        D.load(8, sw, tk, 0, 0, torch.device("cpu"))
    assert {(c["swa"], c["topk"], c["rec"]) for c in seen} == {(128, 512, "v4"), (128, 0, "v4"), (192, 0, "v4")}
    assert {c["ctx"] for c in seen if c["topk"] and c["numtokens"] <= 192} == {"c16k", "c128k", "c512k"}
    with pytest.raises(RuntimeError, match="unsupported record pair"):
        D.rec_label(R.REC_V4, R.REC_FP4)
    assert D.rec_label(R.REC_FP8, R.REC_FP4) == "v41" and D.rec_label(R.REC_FP8, R.REC_FP8, 0) == "v41"


def test_warm_sizes_and_specialization_keys():
    # T values reaching every Triton specialization (T == 1, T % 16 == 0, other) of a bucket
    assert D.warm_sizes(D.ROW, 1) == [1] and D.warm_sizes(D.ROW, 6) == [5, 6] and D.warm_sizes(D.ROW, 4) == [3, 4]
    assert D.warm_sizes(D.ROW, 32) == [25, 32] and D.warm_sizes(D.ROW, 96) == [65, 80, 96]
    assert D.warm_sizes(D.ROW, 256) == [193, 208, 256] and D.warm_sizes(D.PARTIAL, 1024) == [513, 528, 1024]
    for name in (D.PARTIAL, D.ROW):
        for b in D.buckets(name):
            ws = D.warm_sizes(name, b)
            assert b in ws and all(1 <= t <= b for t in ws)
            lo = D.buckets(name)[D.buckets(name).index(b) - 1] if D.buckets(name).index(b) else 0
            assert all(lo < t <= b for t in ws)
            variants = {(t == 1, t % 16 == 0) for t in range(lo + 1, b + 1)}
            assert variants <= {(t == 1, t % 16 == 0) for t in ws}, (name, b)   # every variant the bucket can see
    a = (5, 33, 4752, 5, 65, 4752 * 2, 0, 0)
    k = D.spec_key(D.ROW, 6, a)
    assert D.spec_key(D.ROW, 7, a) == k and D.spec_key(D.ROW, 16, a) != k and D.spec_key(D.ROW, 1, a) != k
    assert D.spec_key(D.ROW, 6, (5, 33 + 16, 4752, 5, 65, 4752 * 2, 0, 0)) == k       # same %16 class
    assert D.spec_key(D.ROW, 6, (5, 32, 4752, 5, 65, 4752 * 2, 0, 0)) != k            # bd % 16 flips


def test_no_compile_inside_a_capture_and_no_call_path_allocation(knob):
    cpu = torch.device("cpu")
    knob.mp.setenv(D.KNOB, "auto")
    knob.tuned[D.PARTIAL].add(6)
    knob.tuned[D.ROW].add(256)
    route = D.load(8, 128, 512, 0, 0, cpu)
    kw = views_of(6)
    knob.mp.setattr(D, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="capture reached the Triton specialization"):
        D.attend(route, **kw)                                          # never compiled eagerly: refused loudly
    knob.mp.setattr(D, "capturing", lambda: False)
    assert D.attend(route, **kw) is kw["out"]                          # first (eager) call warms the specialization
    knob.mp.setattr(D, "capturing", lambda: True)
    assert D.attend(route, **kw) is kw["out"]                          # now legal inside a capture
    # ratio-0 layers: the dummy lists come from load-time buffers, nothing is allocated per call
    knob.mp.setattr(D, "capturing", lambda: False)
    knob.tuned[D.PARTIAL].add(6)
    r0 = D.load(8, 128, 0, 0, 0, cpu)
    kw0 = views_of(6, topk=0)

    def boom(*a, **k):
        raise AssertionError("allocation on the attend() call path")
    for name in ("full", "zeros", "ones", "empty", "tensor", "arange", "empty_like", "zeros_like"):
        knob.mp.setattr(torch, name, boom)
    assert D.attend(r0, **kw0) is kw0["out"]
    assert r0["neg"].shape == (D.ROW_BUCKETS[-1], 1) and r0["zero"].shape == (D.ROW_BUCKETS[-1],)


# ---- K5: anchors + shim -----------------------------------------------------
DECODE_CALL = '''        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )
'''
PREFILL_CALL = '''            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
'''
MQA = "        self._forward_sparse_impl(\n            q=q,\n"


def test_anchors_match_pinned_source_and_refuse_drift():
    K.check_source("_forward_decode", DECODE_CALL)
    K.check_source("_forward_prefill", PREFILL_CALL)
    K.check_source("forward_mqa", MQA)
    for name, text in (("_forward_decode", DECODE_CALL), ("_forward_prefill", PREFILL_CALL)):
        for drift in (text.replace('kv_layout="NHD"', 'kv_layout="HND"'),
                      text.replace("sinks=self.attn_sink", "sinks=None"),
                      text + text,                                         # a second call site
                      text.replace("swa_topk_lens", "swa_lens")):
            with pytest.raises(K.AnchorDrift):
                K.check_source(name, drift)
    with pytest.raises(K.AnchorDrift):
        K.check_source("forward_mqa", "self._forward_other(")


def test_shim_dispatch_stats_and_fail_closed(monkeypatch):
    fallback = []
    orig = lambda **kw: fallback.append(kw) or "flashinfer"                # noqa: E731
    shim = K.make_shim(orig)
    layer = types.SimpleNamespace(prefix="model.layers.7.attn")
    route = {"mode": "auto", "topk": 512, "kernels": {D.PARTIAL: {(6, "c512k"): object()}, D.ROW: {}}}
    state = {"serve": True, "ctx": []}
    monkeypatch.setattr(K, "route_for", lambda *a: route)
    monkeypatch.setattr(D, "attend", lambda *a: state["ctx"].append(a[-1]) or (a[-2] if state["serve"] else None))
    monkeypatch.setattr(D, "pick", lambda route, t, ctx="c512k": (D.PARTIAL, object()))
    v = views_of(5)
    kw = dict(query=v["q"], swa_kv_cache=v["swa_cache"], workspace_buffer=None, sparse_indices=v["swa_idx"],
              compressed_kv_cache=v["cmp_cache"], out=v["out"], bmm1_scale=R.SCALE, sinks=v["sink"],
              kv_layout="NHD", swa_topk_lens=v["swa_len"], extra_sparse_indices=v["cmp_idx"],
              extra_sparse_topk_lens=v["cmp_len"])
    K.STATS.__init__()
    with pytest.raises(RuntimeError, match="layer ctx MISSING"):
        shim(**kw)                                                          # outside forward_mqa
    K._CUR.layer, K._CUR.max_seq_len = layer, 300_000
    try:
        assert shim(**kw) is v["out"] and fallback == [] and state["ctx"] == ["c512k"]   # ctx class from the host int
        K._CUR.max_seq_len = 9000
        assert shim(**kw) is v["out"] and state["ctx"][-1] == "c16k"
        assert K.STATS.helion[("model.layers.7.attn", 6)] == 2
        state["serve"] = False                                             # auto: untuned bucket -> FlashInfer, counted
        assert shim(**kw) == "flashinfer" and len(fallback) == 1
        assert K.STATS.fallback[("model.layers.7.attn", 6)] == 1
        for bad in ({"kv_layout": "HND"}, {"bmm1_scale": 0.1}, {"swa_topk_lens": None}, {"cum_seq_lens_q": None},
                    {"sparse_indices_are_storage_offsets": True}):
            with pytest.raises(RuntimeError, match="unexpected FlashInfer call form"):
                shim(**{**kw, **bad})
        assert len(fallback) == 1                                           # refusals never reached FlashInfer
    finally:
        K._CUR.layer = None
        K._CUR.max_seq_len = None


def test_shim_never_falls_back_to_flashinfer_for_v41_records(monkeypatch):
    """FlashInfer reads V4.1 records in its footer layout, vLLM writes them interleaved: no fallback, ever."""
    fallback = []
    shim = K.make_shim(lambda **kw: fallback.append(kw) or "flashinfer")
    monkeypatch.setattr(K, "route_for", lambda *a: {"mode": "auto", "topk": 512, "kernels": {}})
    monkeypatch.setattr(D, "attend", lambda *a: None)             # auto mode, nothing tuned for this bucket
    for rec in (R.REC_FP8,):
        v = views_of(5, rec=rec)
        kw = dict(query=v["q"], swa_kv_cache=v["swa_cache"], workspace_buffer=None, sparse_indices=v["swa_idx"],
                  compressed_kv_cache=v["cmp_cache"], out=v["out"], bmm1_scale=R.SCALE, sinks=v["sink"],
                  kv_layout="NHD", swa_topk_lens=v["swa_len"], extra_sparse_indices=v["cmp_idx"],
                  extra_sparse_topk_lens=v["cmp_len"])
        K._CUR.layer, K._CUR.max_seq_len = types.SimpleNamespace(prefix="l"), None
        try:
            with pytest.raises(RuntimeError, match="FlashInfer cannot read vLLM's V4.1 interleaved records"):
                shim(**kw)
        finally:
            K._CUR.layer = None
    assert fallback == []


def test_route_for_refuses_to_load_inside_a_capture(monkeypatch):
    monkeypatch.setattr(K, "_ROUTES", {})
    monkeypatch.setattr(D, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="reached a CUDA graph capture without a loaded route"):
        K.route_for(8, 192, 0, 0, 0, torch.device("cpu"))


def test_route_for_runs_the_boot_gate_once_and_can_skip_it(monkeypatch):
    seen = []
    monkeypatch.setattr(K, "_ROUTES", {})
    monkeypatch.setattr(D, "capturing", lambda: False)
    monkeypatch.setattr(D, "load", lambda *a: {"label": "x"})
    monkeypatch.setattr(G, "dsv41_attn_capture_replay", lambda route, dev, **kw: seen.append(route))
    dev = torch.device("cpu")
    r = K.route_for(8, 128, 512, 0, 0, dev)
    assert K.route_for(8, 128, 512, 0, 0, dev) is r and len(seen) == 1       # once per geometry
    monkeypatch.setenv(K.GATE_ENV, "0")
    K.route_for(8, 192, 0, 0, 0, dev)
    assert len(seen) == 1                                                      # SUFFIX_DSV41_ATTN_GATE=0 skips it
    monkeypatch.setattr(D, "load", lambda *a: None)                            # nothing tuned in auto: no gate, no route
    monkeypatch.delenv(K.GATE_ENV)
    assert K.route_for(4, 128, 512, 0, 0, dev) is None and len(seen) == 1


def test_hook_composes_with_the_block64_hook_on_the_same_module(tmp_path, monkeypatch):
    """Two post-import finders armed for ONE module (block64 + this hook) both fire, in order, on a single import."""
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(REPO / "sm120"))
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "fake_dsv41_target.py").write_text("ORDER = ['module body']\n")
    nv = importlib.import_module("nvfp4_kv_patch")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    sys.modules.pop("fake_dsv41_target", None)
    fired = []
    for tag in ("block64", "attn"):                      # armed in sitecustomize order: the later one is in front
        sys.meta_path.insert(0, nv._PostImportFinder(
            "fake_dsv41_target", lambda m, tag=tag: (fired.append(tag), m.ORDER.append(tag))))
    mod = importlib.import_module("fake_dsv41_target")
    assert sorted(fired) == ["attn", "block64"] and set(mod.ORDER) == {"module body", "attn", "block64"}
    sys.modules.pop("fake_dsv41_target", None)


def test_install_hook_is_inert_unless_the_knob_is_set(monkeypatch):
    monkeypatch.delenv(D.KNOB, raising=False)
    assert K.install_post_import_hook() is False                          # oxide default: nothing armed
    monkeypatch.setenv(D.KNOB, "triton")
    with pytest.raises(RuntimeError, match="expected oxide"):
        K.install_post_import_hook()


def test_capture_cli_needs_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    assert G.main(["--small"]) == 2
