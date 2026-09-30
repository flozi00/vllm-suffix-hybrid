# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_HC_FUSED_QUANT on CPU: the numeric contract (torch spec quant ==
nvfp4_gemm.quantize bits), per-op CPU twins of the fused Triton kernels
(program partition, block masks, swizzled stores, padded-row zeroing) vs
stock-op reference -> numpy spec quant -> swizzle_sf, and the wiring
(eligibility reasons, fused GatedResidual data flow, pending block-consumer
swap) on fake modules. The Triton kernels themselves run in the pod oracle."""
import sys
import types

import numpy as np
import pytest
import torch

from suffix_hybrid.kernels import hc_fused_quant as hq
from suffix_hybrid.kernels import nvfp4_gemm as ng


def _np_ref(y, g):
    """Independent path: numpy spec quant + numpy swizzle, padded rows zero."""
    m, k = y.shape
    q, sfb, _ = ng.quantize(y.float().numpy(), np.float32(g))
    return q, ng.swizzle_sf(sfb).reshape(-1, k // 16)[:hq.round_up(m, 128)]


def _assert_q(q, sf, y, g):
    q_ref, sf_ref = _np_ref(y, g)
    np.testing.assert_array_equal(q.numpy(), q_ref)
    np.testing.assert_array_equal(sf.numpy(), sf_ref)  # incl. zeroed padded rows


def _edge(m, k, seed=0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=gen) * scale
    x[0] = 0  # all-zero row
    if m > 2:
        x[2, :16] = torch.tensor([-0.0] * 16)  # negative zeros
        x[2, 16:32] *= 1e3  # a hot block
    return x.bfloat16()


@pytest.mark.parametrize("g", [2688 / 5.0, 2688 / 0.01, 1.0, 6.0, 0.37])
def test_spec_quant_is_nvfp4_gemm_quantize(g):
    y = _edge(7, 256, seed=int(g * 7) % 97, scale=3.0)
    q, sf = hq.spec_quant(y, g)
    q_ref, sfb, _ = ng.quantize(y.float().numpy(), np.float32(g))
    np.testing.assert_array_equal(q.numpy(), q_ref)
    np.testing.assert_array_equal(sf.numpy(), sfb)


def test_spec_quant_ties_and_saturation():
    # g = 6: g/6 = 1 -> sf = e4m3(amax); amax 1.0625 is an e4m3 tie -> 1.0 (even)
    y = torch.zeros(1, 32)
    y[0, :16] = torch.tensor([1.0625, 0.25 * 1.0, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                              -0.75, -1.75, -3.5, 0.0, -0.0, 0.5, 6.0, 0.1])
    y[0, 16:] = 1e4  # saturates: sf clamps at 448, codes at 6
    q, sf = hq.spec_quant(y.bfloat16(), 6.0)
    q_ref, sfb, _ = ng.quantize(y.bfloat16().float().numpy(), np.float32(6.0))
    np.testing.assert_array_equal(q.numpy(), q_ref)
    np.testing.assert_array_equal(sf.numpy(), sfb)
    assert sf[0, 1] == 0x7E  # 448
    assert (q[0, 8:] == 0x77).all()  # +6 | +6 << 4


def test_sf_offset_matches_swizzle():
    rows, nkb = 300, 20
    bits = np.arange(rows * nkb, dtype=np.uint32).reshape(rows, nkb) % 251 + 1
    flat = ng.swizzle_sf(bits.astype(np.uint8))
    r, kb = np.meshgrid(np.arange(rows), np.arange(nkb), indexing="ij")
    np.testing.assert_array_equal(flat[hq.sf_offset(r, kb, nkb)], bits.astype(np.uint8))
    np.testing.assert_array_equal(hq.sf_offset(r, kb, nkb), ng.sf_offset(r, kb, nkb))


def test_quant_store_rounds_to_bf16_first():
    # 0.75 - 2^-9 is a bf16 tie -> 0.75 (even) -> e2m1 tie -> code 2; the
    # unrounded fp32 value would give code 1: the store must see the bf16 value
    tile = torch.full((16,), 0.75 - 2.0 ** -9)
    tile[0] = 6.0
    q = torch.full((32,), 0xAB, dtype=torch.uint8)  # one row, K = 64
    sf = torch.full((128 * 4,), 0xAB, dtype=torch.uint8)
    hq.twin_quant_store(q, sf, tile, 0, 1, 0, 16, 6.0, 64)
    want, _, _ = ng.quantize(tile.bfloat16().float().numpy()[None], np.float32(6.0))
    np.testing.assert_array_equal(q[:8].numpy(), want[0])
    raw, _, _ = ng.quantize(tile.numpy()[None], np.float32(6.0))
    assert not np.array_equal(raw[0], want[0])
    assert q[1] & 0xF == 2  # element 2 (low nibble of byte 1) = 1.0


def test_pad_rows_are_covered_exactly_once():
    for m in list(range(1, 300)) + [2700, 8192]:
        rm, pr = hq.round_up(m, 128), hq.pad_pr(m)
        rows = [m + r + i * m for r in range(m) for i in range(pr) if m + r + i * m < rm]
        assert sorted(rows) == list(range(m, rm)), m
    assert (hq.pad_pr(1), hq.pad_pr(5), hq.pad_pr(64), hq.pad_pr(160), hq.pad_pr(2700)) == \
        (128, 32, 1, 1, 1)


H, NH, R = 128, 4, 64
D = H * NH


@pytest.mark.parametrize("m", [1, 5, 130])
def test_combine_norm_q_twin(m):
    gen = torch.Generator().manual_seed(m)
    res, blk = _edge(m, D, seed=m), _edge(m, H, seed=m + 1)
    inj = torch.randn(m, NH, generator=gen).bfloat16()
    for w in (torch.randn(D, generator=gen) * 0.1, torch.randn(H, generator=gen) * 0.1):
        for i in (inj, None):
            for g in (2688 / 60.0, 2688 / 0.5):
                out, y, q, sf = hq.twin_combine_norm_q(res, blk, i, w.bfloat16(), 1e-6, NH, g)
                o_ref, y_ref = hq.ref_combine_norm(res, blk, i, w.bfloat16(), 1e-6, NH)
                assert torch.equal(out, o_ref) and torch.equal(y, y_ref)
                _assert_q(q, sf, y_ref, g)


def test_combine_norm_twin_tile_is_the_kernels():
    # hidden 2560 -> NUM_TILES 5 -> pad 8 -> 4096-element program tile
    y = _edge(2, 4 * 2560, scale=2.0)
    hd = 2560
    q, sf = hq.twin_quantize(y, 40.0, hd, hq._np2(-(-hd // 512)) * 512)
    _assert_q(q, sf, y, 40.0)


@pytest.mark.parametrize("m", [1, 5, 130])
def test_grouped_norm_q_twin(m):
    x = _edge(m, D, seed=m, scale=4.0)
    w = (torch.randn(D) * 0.1).bfloat16()
    y, q, sf = hq.twin_grouped_norm_q(x, w, 1e-6, NH, 2688 / 50.0)
    assert torch.equal(y, hq.ref_grouped_norm(x, w, 1e-6, NH))
    _assert_q(q, sf, y, 2688 / 50.0)


@pytest.mark.parametrize("m", [1, 5, 130])
def test_silu_q_twin_strided_input(m):
    buf = _edge(m, R + 16, seed=m, scale=20.0)
    lora = buf[:, :R]  # the down-output split view
    for g in (2688 / 10.0, 2688 / 0.2):
        q, sf = hq.twin_silu_q(lora, NH, g)
        _assert_q(q, sf, hq.ref_silu(lora, NH), g)


@pytest.mark.parametrize("m", [1, 5, 130])
def test_gate_mix_q_twin_two_scales(m):
    x, gate = _edge(m, 4 * 2560, seed=m, scale=5.0), _edge(m, 4 * 2560, seed=m + 3)
    y, bufs = hq.twin_gate_mix_q(x, gate, 4, [2688 / 30.0, 2688 / 0.3])
    assert torch.equal(y, hq.ref_gate_mix(x, gate, 4))
    for g, (q, sf) in zip([2688 / 30.0, 2688 / 0.3], bufs):
        _assert_q(q, sf, y, g)


# ---------------------------------------------------------------------------
# wiring on fake modules
# ---------------------------------------------------------------------------
KEY = "kNvfp4Dynamic"


class FakeQA:
    def __init__(self, data, scale, orig_dtype, orig_shape, quant_key):
        self.data, self.scale, self.orig_dtype = data, scale, orig_dtype
        self.orig_shape, self.quant_key = orig_shape, quant_key


class FakeMethod:
    """Exact W(bf16) x dequant(spec quant of x): the fused input must give
    the unfused result bit-for-bit."""

    def __init__(self):
        self.qa = self.bf16 = 0

    def apply(self, layer, x, bias=None):
        k, g = layer.input_size_per_partition, float(layer.input_global_scale_inv)
        if isinstance(x, FakeQA):
            assert x.quant_key == KEY and x.orig_dtype == torch.bfloat16
            self.qa += 1
            rows = x.data.shape[0]
            q, sf = x.data, ng.unswizzle_sf(x.scale, rows, k // 16)
            shape = x.orig_shape
        else:
            self.bf16 += 1
            if x.numel() == 0:
                return x.new_zeros(*x.shape[:-1], layer.output_size_per_partition)
            q, sf = hq.spec_quant(x.reshape(-1, k), g)
            shape = x.shape
        a = ng.dequant_torch(q, sf) / g
        return (a @ layer.w.double().T).float().reshape(*shape[:-1], -1).bfloat16()


class FakeLin(torch.nn.Module):
    def __init__(self, n, k, g, key=KEY, seed=0):
        super().__init__()
        self.input_size_per_partition, self.output_size_per_partition = k, n
        self.weight = torch.zeros(n, k // 2, dtype=torch.uint8)
        self.input_global_scale_inv = torch.tensor([g])
        self._input_quant_key = key
        self.bias = None
        self.w = torch.randn(n, k, generator=torch.Generator().manual_seed(seed)) * 0.05
        self.quant_method = FakeMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


class GatedResidual(torch.nn.Module):
    """vLLM 0.30.0 GatedResidual's mix / combine_and_mix, stock ops = refs."""

    def __init__(self, use_combine=True, seed=0):
        super().__init__()
        self.hc_count, self.hidden_size, self.lora_rank = NH, H, R
        self.use_combine = use_combine
        self.config = types.SimpleNamespace(rms_norm_eps=1e-6)
        self.hc_norm = torch.nn.Module()
        self.hc_norm.weight = torch.nn.Parameter((torch.randn(D) * 0.1).bfloat16())
        self.pad_size = (-(R + NH)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = FakeLin(R + NH + self.pad_size, D,
                                                              2688 / 30.0, seed=seed)
        else:
            self.input_mix_weight_down = FakeLin(R, D, 2688 / 30.0, seed=seed)
        self.input_mix_weight_up = FakeLin(D, R, 2688 / 2.0, seed=seed + 1)

    def _rest(self, xn):
        if self.use_combine:
            d = self.input_mix_weight_down_block_inject(xn)
            lora, inj, _ = d.split([R, NH, self.pad_size], dim=-1)
        else:
            lora, inj = self.input_mix_weight_down(xn), None
        gate = self.input_mix_weight_up(hq.ref_silu(lora, NH))
        return hq.ref_gate_mix(xn, gate, NH), inj

    def mix(self, hs):
        return (hs, *self._rest(hq.ref_grouped_norm(hs, self.hc_norm.weight, 1e-6, NH)))

    def combine_and_mix(self, hs, blk, inj):
        hs, xn = hq.ref_combine_norm(hs, blk, inj, self.hc_norm.weight, 1e-6, NH)
        return (hs, *self._rest(xn))


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_hyper_connection = GatedResidual(seed=1)
        self.mlp_hyper_connection = GatedResidual(seed=2)
        self.linear_attn = torch.nn.Module()
        self.linear_attn.in_proj_qkvz = FakeLin(96, H, 2688 / 8.0, seed=3)
        self.linear_attn.in_proj_ba = FakeLin(16, H, 2688 / 9.0, seed=4)
        self.linear_attn.out = FakeLin(H, 96, 2688 / 9.0, key="other", seed=5)
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert = torch.nn.Module()
        self.mlp.shared_expert.gate_up_proj = FakeLin(64, H, 2688 / 7.0, seed=6)

    def forward(self, hs, blk, inj):
        hs, bi, inj = self.attn_hyper_connection.combine_and_mix(hs, blk, inj)
        a = self.linear_attn.in_proj_qkvz(bi)
        b = self.linear_attn.in_proj_ba(bi.view(-1, H))  # a view: still matched
        attn_out = self.linear_attn.out(a)
        hs, bi, inj = self.mlp_hyper_connection.combine_and_mix(hs, attn_out, inj)
        mlp_out = torch.cat([self.mlp.shared_expert.gate_up_proj(bi), bi], -1)[:, :H]
        return hs, mlp_out, inj, a, b


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.hyper_connection_mixer = GatedResidual(use_combine=False, seed=9)

    def forward(self, x):
        hs, bi, inj = self.layers[0].attn_hyper_connection.mix(x)  # first-layer mix path
        outs = [bi]
        blk = bi
        for layer in self.layers:
            hs, blk, inj, a, b = layer(hs, blk, inj)
            outs += [hs, blk, inj, a, b]
        outs += list(self.hyper_connection_mixer.combine_and_mix(hs, blk, inj)[:2])
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    ops = types.SimpleNamespace(
        hc_combine_norm=hq.ref_combine_norm, grouped_gemma_rmsnorm=hq.ref_grouped_norm,
        hc_silu=hq.ref_silu, hc_gate_mix=hq.ref_gate_mix,
        combine_norm_q=hq.twin_combine_norm_q, grouped_norm_q=hq.twin_grouped_norm_q,
        silu_q=hq.twin_silu_q, gate_mix_q=hq.twin_gate_mix_q)
    monkeypatch.setattr(hq, "_V", types.SimpleNamespace(QA=FakeQA, KEY=KEY, ops=ops))
    monkeypatch.setattr(hq, "_state", {"missed": set(), "pairs": {}})


def _methods(model):
    return [m.quant_method for m in model.modules() if isinstance(m, FakeLin)]


def test_plan_and_fused_forward_equals_unfused(fake_vllm):
    torch.manual_seed(0)
    model = Model()
    x = (torch.randn(5, D) * 2).bfloat16()
    ref = model(x)
    plans, skipped = hq.plan_model(model, KEY)
    assert len(plans) == 5
    assert hq.pair_counts(plans) == {"A norm->down": 5, "B silu->up": 5,
                                     "C gate_mix->block": 6, "C gate_mix launches": 4}
    attn = next(p for p in plans if p.name == "layers.0.attn_hyper_connection")
    assert len(attn.groups) == 2  # in_proj_qkvz / in_proj_ba: two static scales
    assert skipped == []
    for m in _methods(model):
        m.qa = m.bf16 = 0
    for p in plans:
        hq.install(p)
    got = model(x)
    for a, b in zip(got, ref):
        assert (a is None and b is None) or torch.equal(a, b)
    by = {n: m.quant_method for n, m in model.named_modules() if isinstance(m, FakeLin)}
    # every HC projection + block consumer took the prequantized input ...
    for n, qm in by.items():
        if n.endswith(".out"):
            assert qm.qa == 0 and qm.bf16 == 1
        elif n.startswith("layers.0.attn_hyper_connection"):
            assert qm.qa == 2 and qm.bf16 == 0  # mix + combine_and_mix
        else:
            assert qm.qa >= 1 and qm.bf16 == 0, n
    assert not hq._state["missed"]


def test_eligibility_reasons(fake_vllm):
    model = Model()
    l0 = model.layers[0]
    l0.linear_attn.in_proj_ba._input_quant_key = "fp8"
    l0.mlp_hyper_connection.input_mix_weight_up.weight = torch.zeros(D, R // 2 + 8,
                                                                     dtype=torch.uint8)
    l0.attn_hyper_connection.input_mix_weight_down_block_inject.bias = torch.zeros(1)
    l1 = model.layers[1]
    l1.linear_attn.in_proj_ba.input_global_scale_inv = torch.tensor([3.0])
    l1.linear_attn.in_proj_qkvz.input_global_scale_inv = torch.tensor([4.0])
    l1.mlp.shared_expert.gate_up_proj.requires_unquantized_input = True
    # a third distinct scale at l1's attn boundary
    l1.self_attn = torch.nn.Module()
    l1.self_attn.qkv_proj = FakeLin(32, H, 5.0)
    plans, skipped = hq.plan_model(model, KEY)
    text = "\n".join(skipped)
    assert "layers.0.linear_attn.in_proj_ba C gate_mix->block: input quant key fp8" in text
    assert "layers.0.mlp_hyper_connection B silu->up: K-padded" in text
    assert "layers.0.attn_hyper_connection A norm->down: bias" in text
    assert "layers.1.self_attn.qkv_proj C gate_mix->block: more than 2 distinct" in text
    assert "layers.1.mlp.shared_expert.gate_up_proj C gate_mix->block: input quant key None" in text
    assert hq.consumer_reason(None, 64, KEY) == "absent"
    assert "% 64" in hq.consumer_reason(FakeLin(16, 96, 1.0), 96, KEY)
    assert "!= producer width" in hq.consumer_reason(FakeLin(16, 64, 1.0), 128, KEY)
    assert "not a vLLM" in hq.producer_reason(torch.nn.Module())


def test_pending_miss_runs_unfused_and_logs_once(fake_vllm, capsys):
    model = Model()
    plans, _ = hq.plan_model(model, KEY)
    for p in plans:
        hq.install(p)
    lin = model.layers[0].linear_attn.in_proj_qkvz
    y = torch.randn(3, H).bfloat16()
    lin._sfx_hcq_pending = (y, None, None, None)
    other = y.clone()
    lin(other)
    lin._sfx_hcq_pending = (y, None, None, None)
    lin(other)
    assert lin.quant_method.bf16 == 2 and lin.quant_method.qa == 0
    assert "_sfx_hcq_pending" not in lin.__dict__  # consumed
    assert capsys.readouterr().err.count("MISS") == 1
    hq.install(plans[0])  # idempotent: apply is wrapped once
    assert not isinstance(lin.quant_method.apply.args[0], type(lin.quant_method.apply))


def test_empty_batch_takes_the_stock_path(fake_vllm):
    model = Model()
    for p in hq.plan_model(model, KEY)[0]:
        hq.install(p)
    hc = model.layers[0].attn_hyper_connection
    hs, bi, inj = hc.mix(torch.zeros(0, D).bfloat16())
    assert bi.shape == (0, H)


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(hq.GATE, raising=False)
    before = set(sys.modules)
    assert hq.wire(object()) == []
    assert not any(m.startswith("vllm") for m in set(sys.modules) - before)


def test_converter_calls_wire_only_with_the_gate(monkeypatch):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "sm120"))
    try:
        import fp8_dense_patch as fdp
        from fp8_dense_patch import nvfp4
    finally:
        sys.path.pop(0)
    calls = []
    monkeypatch.setattr(nvfp4, "convert_model", lambda m: calls.append("convert"))
    monkeypatch.setattr(hq, "wire", lambda m: calls.append("wire"))
    monkeypatch.setenv("SUFFIX_NVFP4_DENSE", "1")
    monkeypatch.delenv("SUFFIX_FP8_DENSE", raising=False)
    monkeypatch.delenv("SUFFIX_ACT_AMAX_RECORD", raising=False)
    monkeypatch.delenv(hq.GATE, raising=False)
    fdp._suffix_fp8_dense_convert(object())
    assert calls == ["convert"]
    monkeypatch.setenv(hq.GATE, "1")
    fdp._suffix_fp8_dense_convert(object())
    assert calls == ["convert", "convert", "wire"]


def test_boot_gates_allowlisted():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    for mode in ("oracle", "bench"):
        argv, env = ns["_BOOT_GATES"][f"hc_fused_quant_{mode}"]
        assert argv == ["-m", "suffix_hybrid.kernels.hc_fused_quant", mode] and env == {}
