# SPDX-License-Identifier: Apache-2.0
"""TP-sharded NVFP4 decode MoE in Triton (suffix_hybrid/kernels/nvfp4_moe_tp.py): CPU contract
tests. The @triton.jit kernels themselves run through tests/triton_shim.py (plain Python per
program on torch CPU tensors, bounds-checked pointers, poisoned masked lanes): every plan vs the
f64 spec and the oxide CPU twin, quant / dequant bits vs vLLM's spec, the in-kernel route's pair
list, multi-pass hot experts, the backend switch and table, the fail-closed load path. Silicon:
boot gate glm_moe_tp_kernel_bench + the per-layer load oracle."""
import importlib
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import triton_shim  # noqa: E402
from test_nvfp4_moe import kernel_twin  # noqa: E402

from suffix_hybrid.kernels import nvfp4_moe as nm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
E, H, I, TOPK = 16, 512, 64, 4


@pytest.fixture(scope="module")
def tp():
    with triton_shim.installed():
        sys.modules.pop("suffix_hybrid.kernels.nvfp4_moe_tp", None)
        mod = importlib.import_module("suffix_hybrid.kernels.nvfp4_moe_tp")
        assert isinstance(mod._fc1_fma, triton_shim.JITFunction)
        yield mod
    sys.modules.pop("suffix_hybrid.kernels.nvfp4_moe_tp", None)


@pytest.fixture(autouse=True)
def fresh_state(tp):
    tp._ST.update(ws={}, table=None, pdl=None, layers=0, logged=False)
    yield


@pytest.fixture(scope="module")
def prob():
    p = nm.make_problem(E, H, I, "silu", seed=3)
    p.update(topk=TOPK, E_global=E, id_base=0, max_m=8)
    return p


@pytest.fixture(scope="module")
def prob_sh(prob):
    """+ a fused shared slot (nvfp4_moe.quant_shared, SUFFIX_NVFP4_MOE_FUSE_SHARED): its own
    tensors, g1 != g1u (c = 1/4) and g2 carrying 1/rsf (2.5)."""
    gen = torch.Generator().manual_seed(11)
    w_gu = (torch.randn(2 * I, H, generator=gen, dtype=torch.float32) * 0.05).bfloat16()
    w_dn = (torch.randn(H, I, generator=gen, dtype=torch.float32) * 0.05).bfloat16()
    return dict(prob, shared=nm.quant_shared(w_gu, w_dn, prob["a1g"], prob["a2g"], 0.25, 2.5))


def _state(tp, p):
    return tp.layer_state(p, dict(E=E, H=H, I=I, ep=1))


def _batch(m, seed, hot=False):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, H, generator=gen, dtype=torch.float32) * 1.3).bfloat16()
    ids, tw = nm.rand_routing(m, E, TOPK, "cpu", seed=seed)
    if hot:
        ids = ids[:1].expand(m, -1).contiguous()
    return x, ids.int().contiguous(), tw.float().contiguous()


def test_quant_bits_match_vllm_spec_including_ties(tp):
    g = 6.0  # g / 6 == 1: a block whose amax is 6 has sf = 6 and inv = 1, so s == x exactly
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    ties = torch.cat([mids, -mids]).reshape(1, 16)
    rnd = torch.randn(7, 16) * torch.logspace(-3, 2, 7)[:, None]
    blocks = torch.cat([ties, rnd, torch.zeros(1, 16)])
    for gg in (g, 2688.0 / 4.5, 1 / 0.0013718378031626344):
        s, q, sf, sf8 = tp._quant(blocks.reshape(1, -1, 16), gg, triton_shim.div_rn(gg, 6.0), 2)
        code = tp._code(q) | torch.where(s < 0, 8, 0).int()
        packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8).reshape(len(blocks), 8)
        want_q, want_sf = nm.quant_codes(blocks, gg)
        assert torch.equal(packed, want_q), gg
        assert torch.equal(sf8.view(torch.uint8).reshape(-1), want_sf.reshape(-1)), gg
        assert torch.equal(sf.reshape(-1), want_sf.view(torch.float8_e4m3fn).float().reshape(-1))


def test_x_quant_dequant_is_the_spec_qdq_split_even_odd(tp):
    x = torch.randn(4, 256) * 2
    g = 2688.0 / float(x.abs().max())
    g6 = triton_shim.div_rn(g, 6.0)
    want = nm.qdq(x, g)
    even, odd = tp._xq_rows(x, g, g6, 4, 256)
    assert torch.equal(even, want[:, 0::2]) and torch.equal(odd, want[:, 1::2])
    e1, o1 = tp._xq_row(x[2], g, g6, 256)
    assert torch.equal(e1, want[2, 0::2]) and torch.equal(o1, want[2, 1::2])
    assert torch.equal(want.half().float(), want)  # exact as fp16 MMA operands


def test_fp4_dequant_every_byte_and_scale_exact(tp):
    b = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(2, 128)
    sf = torch.tensor([[0x01, 0x08, 0x38, 0x7E] * 4] * 2, dtype=torch.uint8)  # e4m3: min sub, min norm, 1, 448
    want = nm.dequant(b, sf.view(torch.float8_e4m3fn).float())
    lo, hi = tp._deq32(b, sf, 2, 16)
    assert torch.equal(lo, want[:, 0::2]) and torch.equal(hi, want[:, 1::2])
    lo16, hi16 = tp._deq16(b, sf, 2, 16)
    assert lo16.dtype == torch.float16 and torch.equal(lo16.float(), want[:, 0::2])
    assert torch.equal(hi16.float(), want[:, 1::2])
    nz = want[want != 0].abs()
    assert float(nz.min()) >= 2.0 ** -14  # every MMA operand value is fp16-normal


def test_swizzled_scale_offsets_match_the_reference_layout(tp):
    rows, kb, kb_pad = torch.arange(300).int(), torch.arange(48).int(), 48
    off = tp._sf_row(rows, kb_pad)[:, None] + tp._sf_col(kb)[None, :]
    bits = (torch.arange(300 * 48) % 251).to(torch.uint8).reshape(300, 48)
    flat = nm.swizzle(bits)
    assert torch.equal(flat[off.long()], bits)


@pytest.mark.parametrize("plan", ["f1", "f1s", "f1n", "f2", "m64", "m128", "m128w"])
@pytest.mark.parametrize("m,hot", [(1, False), (6, False), (5, True)])
def test_plan_matches_f64_spec_and_oxide_twin(tp, prob, plan, m, hot):
    st = _state(tp, prob)
    x, ids, tw = _batch(m, 40 + m, hot)
    out = tp.run(st, x, ids, tw, plan)
    again = tp.run(st, x, ids, tw, plan)
    assert out.dtype == torch.bfloat16 and out.shape == (m, H)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))  # deterministic
    ref = nm.moe_ref(prob, x, ids, tw)
    twin = kernel_twin(prob, x, ids, tw)
    r_ref, r_twin, twin_ref = nm._rel(out.float(), ref), nm._rel(out.float(), twin), nm._rel(twin, ref)
    assert r_ref <= nm.REF_TOL, (r_ref, r_twin)
    assert r_twin <= 1.05 * (r_ref + twin_ref) + 1e-6


@pytest.mark.parametrize("plan", ["f1", "f1s", "f1n", "f2", "m64", "m128", "m128w"])
@pytest.mark.parametrize("m,hot", [(2, False), (6, True)])
def test_plan_runs_the_fused_shared_slot(tp, prob, prob_sh, plan, m, hot):
    st = _state(tp, prob_sh)
    assert st["sh"] is not None and st["sh"]["g1"] != st["sh"]["g1u"]
    x, ids, tw = _batch(m, 60 + m, hot)
    out = tp.run(st, x, ids, tw, plan)
    assert torch.equal(out.view(torch.int16), tp.run(st, x, ids, tw, plan).view(torch.int16))
    ref = nm.moe_ref(prob_sh, x, ids, tw)  # routed k sum + the shared slot (shared_ref)
    shared = nm.shared_ref(prob_sh, x)
    assert nm._rel(shared, ref) > 0.05  # the slot is a real part of the output
    assert nm._rel(out.float(), ref) <= nm.REF_TOL
    routed_only = tp.run(_state(tp, prob), x, ids, tw, plan).float()
    assert nm._rel(out.float() - routed_only, shared) < 0.05


def test_pair_list_is_route_twin_order(tp, prob):
    st = _state(tp, prob)
    x, ids, tw = _batch(9, 77)
    tp.run(st, x, ids, tw, "f1")
    _, _, _, pl = nm.route_twin(ids, E)
    assert tp._ST["ws"][x.device]["pairs"][:len(pl)].tolist() == pl


def test_plans_agree_with_each_other(tp, prob):
    st = _state(tp, prob)
    x, ids, tw = _batch(6, 5)
    outs = {p: tp.run(st, x, ids, tw, p).float() for p in tp.PLANS}
    base = outs.pop("f1")
    for p, o in outs.items():
        assert nm._rel(o, base) < 2 * nm.REF_TOL, p


def test_route_skips_invalid_ids_like_oxide(tp, prob):
    st = _state(tp, prob)
    x, ids, tw = _batch(4, 9)
    ids[1, 2] = -1
    ids[3, 0] = E + 3
    out = tp.run(st, x, ids, tw, "f2").float()
    ref = nm.moe_ref(prob, x, ids, tw)  # invalid ids contribute exactly nothing in the spec
    assert nm._rel(out, ref) <= nm.REF_TOL


def test_table_parse_merge_and_lookup(tp):
    t = tp.parse_table(" 2:f1, 8:f2,64:oxide ,192:m128")
    assert t == [(2, "f1"), (8, "f2"), (64, "oxide"), (192, "m128")]
    assert tp.format_table(t) == "2:f1,8:f2,64:oxide,192:m128"
    plans = tp.by_m(t, 200)
    assert plans[1] == plans[2] == "f1" and plans[3] == plans[8] == "f2"
    assert plans[9] is None and plans[64] is None and plans[65] == "m128" and plans[200] == "m128"
    assert tp.merge_table([(1, "f1"), (2, "f1"), (4, "f2"), (6, "f1")]) == [(2, "f1"), (4, "f2"), (6, "f1")]
    for bad in ("", "8:nope", "8:f1,4:f2", "0:f1", "x:f1", "300:f1"):
        with pytest.raises(ValueError):
            tp.parse_table(bad)


def test_impl_env(tp, monkeypatch):
    monkeypatch.delenv(tp.IMPL_ENV, raising=False)
    assert tp.impl() == "oxide"
    for v in ("tp", "auto", "oxide"):
        monkeypatch.setenv(tp.IMPL_ENV, v)
        assert tp.impl() == v
    monkeypatch.setenv(tp.IMPL_ENV, "fast")
    with pytest.raises(ValueError):
        tp.impl()


@pytest.mark.parametrize("change,needle", [
    ({"cfg": {"kind": "fp8"}}, "FP8"),
    ({"cfg": {"shared": {"w13": torch.empty(512, 3072), "w2": torch.empty(6144, 1024)}}},
     "fused shared slot"),
    ({"info": {"ep": 8}}, "expert parallel"),
    ({"cfg": {"id_base": 32}}, "expert parallel"),
    ({"info": {"E": 48}, "cfg": {"E_global": 48}}, "E=48"),
    ({"info": {"I": 96}}, "I=96"),
    ({"info": {"H": 2816}}, "H=2816"),
    ({"cfg": {"topk": 20}}, "top-20"),
    ({"cfg": {"act": 7}}, "activation"),
])
def test_eligibility(tp, change, needle):
    cfg = dict(topk=8, act=0, id_base=0, E_global=256)
    info = dict(E=256, H=6144, I=256, ep=1)
    assert tp.eligibility(cfg, info) is None
    ok_shared = {"w13": torch.empty(512, 3072), "w2": torch.empty(6144, 128)}
    assert tp.eligibility(dict(cfg, shared=ok_shared), info) is None  # TP8 shared slot fits
    cfg.update(change.get("cfg", {}))
    info.update(change.get("info", {}))
    assert needle in tp.eligibility(cfg, info)


def test_roofline_bytes_and_cpu_import(tp):
    import subprocess
    assert tp.roofline_bytes(1, 6144, 256) == 2654208.0  # w13 + w2 + e4m3 scales of one expert
    r = subprocess.run([sys.executable, "-c", "import suffix_hybrid.kernels.nvfp4_moe_tp as m; "
                        "assert m.triton is None and m.impl() == 'oxide'"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")})
    assert r.returncode == 0, r.stderr


def _cfg(prob, **kw):
    cfg = dict(prob)
    cfg.update(kw)
    return cfg


def test_prepare_is_inert_when_oxide(tp, prob, monkeypatch):
    monkeypatch.delenv(tp.IMPL_ENV, raising=False)
    cfg = _cfg(prob)
    tp.prepare(SimpleNamespace(layer_name="l0"), cfg, dict(E=E, H=H, I=I, ep=1), None)
    assert "tp_impl" not in cfg and tp._ST["table"] is None


def test_prepare_tp_with_pasted_table_dispatches_per_m(tp, prob, monkeypatch):
    monkeypatch.setenv(tp.IMPL_ENV, "auto")
    monkeypatch.setenv(tp.TABLE_ENV, "2:f1,6:oxide,8:m64")
    oxide_calls = []

    def oxide(x, ids, tw):
        oxide_calls.append(x.shape[0])
        return kernel_twin(prob, x, ids, tw).bfloat16()

    cfg = _cfg(prob)
    tp.prepare(SimpleNamespace(layer_name="model.layers.3.mlp.experts"), cfg,
               dict(E=E, H=H, I=I, ep=1), oxide)
    layer = cfg["tp_impl"]
    assert layer.plans[1:] == ["f1", "f1", None, None, None, None, "m64", "m64"]
    assert oxide_calls  # the oracle compared against the oxide path
    x, ids, tw = _batch(4, 1)
    assert layer(x, ids, tw) is None  # M=4 belongs to oxide
    x, ids, tw = _batch(2, 1)
    out = layer(x, ids.long(), tw)  # int64 ids are narrowed, not recompiled
    assert nm._rel(out.float(), nm.moe_ref(prob, x, ids, tw)) <= nm.REF_TOL


def test_prepare_under_vllms_bf16_default_dtype(tp, prob_sh, monkeypatch):
    # vLLM loads weights under torch.set_default_dtype(bf16): every load-time tensor must be explicit
    monkeypatch.setenv(tp.IMPL_ENV, "tp")
    monkeypatch.setenv(tp.TABLE_ENV, "4:f1,8:m128")
    cfg = _cfg(prob_sh)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        tp.prepare(SimpleNamespace(layer_name="model.layers.5.mlp.experts"), cfg,
                   dict(E=E, H=H, I=I, ep=1), lambda x, i, w: nm.moe_ref(prob_sh, x, i, w).bfloat16())
        x, ids, tw = tp._batch(cfg["tp_impl"].st, 3, torch.device("cpu"), 5)
        assert x.dtype == torch.bfloat16 and tw.dtype == torch.float32
        out = cfg["tp_impl"](x, ids, tw)
    finally:
        torch.set_default_dtype(old)
    assert nm._rel(out.float(), nm.moe_ref(prob_sh, x, ids, tw)) <= nm.REF_TOL


def test_prepare_tp_refuses_oxide_buckets_and_ineligible_layers(tp, prob, monkeypatch):
    monkeypatch.setenv(tp.IMPL_ENV, "tp")
    monkeypatch.setenv(tp.TABLE_ENV, "4:f1,8:oxide")
    with pytest.raises(RuntimeError, match="use auto"):
        tp.prepare(SimpleNamespace(layer_name="l0"), _cfg(prob), dict(E=E, H=H, I=I, ep=1), None)
    tp._ST["table"] = None
    with pytest.raises(RuntimeError, match="expert parallel"):
        tp.prepare(SimpleNamespace(layer_name="l0"), _cfg(prob, id_base=16),
                   dict(E=E, H=H, I=I, ep=2), None)


def test_load_oracle_fails_closed(tp, prob, monkeypatch):
    monkeypatch.setenv(tp.IMPL_ENV, "tp")
    monkeypatch.setenv(tp.TABLE_ENV, "8:f2")
    real = tp.run
    tp._ST["table"] = [(8, "f2")]  # validated earlier; now the kernel drifts on this layer
    monkeypatch.setattr(tp, "run", lambda *a, **k: real(*a, **k) * 1.01)
    with pytest.raises(RuntimeError, match="LAYER ORACLE FAIL"):
        tp.prepare(SimpleNamespace(layer_name="l9"), _cfg(prob), dict(E=E, H=H, I=I, ep=1),
                   lambda x, i, w: kernel_twin(prob, x, i, w).bfloat16())


def test_validation_rejects_a_bad_plan_and_auto_falls_back(tp, prob, monkeypatch):
    monkeypatch.setenv(tp.IMPL_ENV, "auto")
    monkeypatch.delenv(tp.TABLE_ENV, raising=False)
    real = tp.run
    monkeypatch.setattr(tp, "run", lambda st, x, i, w, plan, **k: real(st, x, i, w, plan, **k) * 2)
    cfg = _cfg(prob)
    tp.prepare(SimpleNamespace(layer_name="l0"), cfg, dict(E=E, H=H, I=I, ep=1), None)
    assert tp._ST["table"] == [(256, "oxide")] and all(p is None for p in cfg["tp_impl"].plans)


def test_boot_gate_and_hook_are_wired():
    src = (ROOT / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    assert ns["_BOOT_GATES"]["glm_moe_tp_kernel_bench"] == (
        ["-m", "suffix_hybrid.kernels.nvfp4_moe_tp", "bench"], {})
    moe = (ROOT / "suffix_hybrid/kernels/nvfp4_moe.py").read_text()
    assert "nvfp4_moe_tp.prepare(self, cfg, info," in moe
    assert 'cfg.get("tp_impl")' in moe
