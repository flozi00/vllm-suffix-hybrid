# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_DSV41_MOE_FUSE_SHARED on CPU: the MXFP8 layout contract, the three fused Helion bodies in
Helion's eager reference mode (HELION_INTERPRET=1) against the f64 spec (stage level, plan level,
balanced / skewed / hot expert loads, adversarial scales), their generated Triton code (static and
dynamic shapes, default / all-min / all-max configs: stateless, one launch with a shared-first
program-id region), bounded search spaces, the lab ABI, eligibility / knobs / the serving state,
the layer oracle + capture-replay gate logic on fakes, and the serving-path AST audit. Silicon:
`python -m suffix_hybrid.kernels.dsv41_moe_shared oracle|moe_capture_replay|bench`."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before Helion is imported

import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_moe as hm  # noqa: E402
from suffix_hybrid.kernels import dsv41_moe_shared as glue  # noqa: E402
from suffix_hybrid.kernels import nvfp4_moe as nm  # noqa: E402

DS = {"name": "deepseek-v4.1-flash-1017-nvfp4", "architecture": "DeepseekV41ForCausalLM",
      "tp": 8, "hidden": 5120, "moe_intermediate": 288, "experts": 384, "topk": 6,
      "quant_format": "nvfp4"}
EP8 = {"experts": 48, "hidden": 5120, "moe_intermediate": 2304, "topk": 6, "act": "silu",
       "limit": 10.0, "experts_global": 384, "sh_inter": 288}


def _swizzle_mxfp8_scale(sf, M, K):
    """vllm/model_executor/layers/quantization/utils/mxfp8_utils.py swizzle_mxfp8_scale at
    39983a5e7, verbatim."""
    scaling_vector_size = 32
    factor = scaling_vector_size * 4
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4
    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device)
    sf_padded[:M, :scale_cols] = sf
    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_swizzled = sf_reshaped.transpose(1, 3)
    return sf_swizzled.contiguous().view(-1)


# ---------------------------------------------------------------------------
# the layout contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(576, 5120), (5120, 288), (200, 160), (128, 128)])
def test_swizzle_is_vllms_mxfp8_layout(n, k):
    s = torch.randint(1, 247, (n, k // 32), dtype=torch.uint8)
    assert torch.equal(hs.swizzle_scales(s), _swizzle_mxfp8_scale(s, n, k))
    assert torch.equal(hs.unswizzle_scales(hs.swizzle_scales(s), n, k // 32), s)


@pytest.mark.parametrize("rows32", [False, True])
def test_quantize_dequantize(rows32):
    w = torch.randn(64, 128) * 0.03
    q, s = hs.quantize_mxfp8(w, rows32=rows32)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8 and s.shape == (64, 4)
    assert int(s.min()) >= hs.S_MIN and int(s.max()) <= hs.S_MAX
    d = hs.dequant_mxfp8(q, s)
    assert float((d - w.double()).abs().max()) <= float(w.abs().max()) * 2 ** -4
    if rows32:  # the checkpoint's 32x32 blocks: a scale byte is constant over 32 rows
        assert bool((s.view(2, 32, 4) == s.view(2, 32, 4)[:, :1]).all())


def test_adversarial_mxfp8_hits_the_scale_corners():
    gen = torch.Generator().manual_seed(1)
    q, s = hs.quantize_mxfp8(torch.randn(256, 512) * 0.02)
    hs.adversarial_mxfp8(q, s, gen)
    assert int(s.min()) == hs.S_MIN and int(s.max()) == hs.S_MAX
    u8 = q.view(torch.uint8)
    assert bool(((u8 & 0x7F) != 0x7F).all())  # no e4m3 NaN code anywhere
    d = hs.dequant_mxfp8(q, s)
    assert torch.isfinite(d).all()
    assert int(((u8 & 0x7F) == 0x7E).sum()) > 100  # +-448 outliers (e4m3 max) in their blocks
    zero_blocks = (u8.view(256, 16, 32) == 0).all(-1)
    assert int(zero_blocks.sum()) > 100
    glue.validate_weights(q, hs.swizzle_scales(s), "t")  # the load check accepts it
    bad = q.clone()
    bad.view(torch.uint8)[0, 0] = 0x7F
    with pytest.raises(RuntimeError, match="NaN"):
        glue.validate_weights(bad, hs.swizzle_scales(s), "t")
    s2 = s.clone()
    s2[0, 0] = 255
    with pytest.raises(RuntimeError, match="scale byte"):
        glue.validate_weights(q, hs.swizzle_scales(s2), "t")


# ---------------------------------------------------------------------------
# bodies: Helion eager reference mode vs the f64 spec
# ---------------------------------------------------------------------------
@pytest.fixture
def helion(monkeypatch):
    mod = pytest.importorskip("helion")
    mm = torch.mm

    def cpu_mm(a, b, *, out_dtype=None, **kw):  # CPU lacks mm.dtype; the products are exact in f32
        return mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)

    monkeypatch.setattr(torch, "mm", cpu_mm)
    return mod


def _case(m, e=6, hd=128, i=48, k=3, limit=0.3, eg=24, isz=64):
    return {"experts": e, "hidden": hd, "moe_intermediate": i, "topk": k, "act": "silu",
            "numtokens": m, "limit": limit, "experts_global": eg, "sh_inter": isz}


@pytest.mark.parametrize("hd,isz", [(128, 64), (512, 160)])  # 2nd: several 128-row / k-tile
@pytest.mark.parametrize("seed", [0, 1])  # balanced / skewed expert loads  # scale tiles, padded
@pytest.mark.parametrize("m", [5, 20])  # scale columns (5 of 8), a partial last column block
@pytest.mark.parametrize("name", [k["name"] for k in hs.KERNELS])
def test_body_matches_spec(helion, name, m, seed, hd, isz):
    kern = next(k for k in hs.KERNELS if k["name"] == name)
    case = _case(m, hd=hd, isz=isz)
    args = kern["make"](case, seed)
    out = helion.kernel(kern["body"])(*args)
    ok, msg = kern["check"](out, kern["reference"](*args))
    assert ok, msg


def _plan_inputs(m, dist, adversarial, local=6, e_global=48, base=24, topk=6, idim=288,
                 hdim=128, isz=64, limit=0.5, dead=None):
    p = nm.make_problem(local, hdim, idim, "silu", seed=11, limit=limit, adversarial=adversarial)
    sp = hs.make_shared_problem(hdim, isz, seed=3, limit=limit, wstd=0.05, adversarial=adversarial)
    ids, tw = nm.rand_routing(m, e_global, topk, "cpu", seed=m, base=base, local=local, dist=dist,
                              dead=dead)
    x = torch.randn(m, hdim)
    if adversarial:
        x = nm.adversarial_blocks(x, torch.Generator().manual_seed(7))
    return p, sp, ids, tw, x.bfloat16(), base, limit


def _run(helion, plan, p, sp, ids, tw, x, base, limit):
    kerns = [helion.kernel(hs.BODIES_SH[k]) for k in hs.PLANS_SH[plan]]
    out = torch.full((x.shape[0], x.shape[1]), float("nan")).bfloat16()
    sh = hs.sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])
    return hs.run_plan_sh(kerns, plan, x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"],
                          p["w2_sf"], p["g2"], out, p["a1g"], p["a2g"], base, limit, sh)


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("dist", nm.ROUTING_DISTS)
@pytest.mark.parametrize("plan", ["front4", "fused3"])
def test_plan_routed_plus_shared_vs_spec(helion, plan, dist, adversarial):
    """out = bf16(routed + shared) vs the f64 spec of both, under every expert-load shape and
    the adversarial block-scale patterns (zero / min-scale / max-scale / outlier blocks)."""
    p, sp, ids, tw, x, base, limit = _plan_inputs(20, dist, adversarial)
    lid = nm.to_local(ids, base, 6)
    out = _run(helion, plan, p, sp, ids, tw, x, base, limit)
    ref = nm.moe_ref(p, x, lid, tw) + hs.shared_ref(sp, x)
    assert torch.isfinite(out.float()).all()
    assert hm._rel(out.float(), ref) <= nm.REF_TOL
    # the shared expert is not noise (>= 3 % of the output; its own tight bounds are the
    # shared-alone test and the stage-level hs / S checks)
    assert hm._rel(out.float(), nm.moe_ref(p, x, lid, tw)) > 0.03


@pytest.mark.parametrize("plan", ["front4", "fused3"])
def test_shared_partial_alone_when_no_token_is_local(helion, plan):
    """Every token routed to other ranks: the layer returns exactly the shared partial (the
    all-reduce then sums S over the ranks like stock's shared_output)."""
    p, sp, ids, tw, x, base, limit = _plan_inputs(12, "balanced", False, dead=True)
    out = _run(helion, plan, p, sp, ids, tw, x, base, limit)
    assert hm._rel(out.float(), hs.shared_ref(sp, x)) <= glue.SHARED_REF_TOL


def test_plans_agree_and_are_deterministic_and_stateless(helion):
    pa = _plan_inputs(9, "skewed", False)
    pb = _plan_inputs(5, "hot", True)
    a1 = _run(helion, "front4", *pa)
    b1 = _run(helion, "front4", *pb)
    a2 = _run(helion, "front4", *pa)
    assert torch.equal(a1, a2)  # B in between changed nothing: no carried state
    f3 = _run(helion, "fused3", *pa)
    assert torch.equal(f3, _run(helion, "fused3", *pa))
    assert hm._rel(a1.float(), f3.float()) <= 3e-3  # two summation orders, one bf16 rounding
    assert torch.isfinite(b1.float()).all()


def test_combine_order_vs_vllm_shared_plus_routed(helion):
    """vLLM's runner: result = bf16(shared_bf16 + fused_bf16) (routed scaling already folded into
    the top-k weights; runner rsf 1.0). Ours rounds the f32 sum once: within ~1 bf16 ulp."""
    p, sp, ids, tw, x, base, limit = _plan_inputs(16, "balanced", False)
    lid = nm.to_local(ids, base, 6)
    ours = _run(helion, "front4", p, sp, ids, tw, x, base, limit).float()
    stock = (nm.moe_ref(p, x, lid, tw).bfloat16()
             + hs.shared_ref(sp, x).bfloat16()).float()
    assert hm._rel(ours, stock) <= 4e-3


# ---------------------------------------------------------------------------
# generated Triton code (Helion front end, no GPU): stateless, one launch, bounded
# ---------------------------------------------------------------------------
def _bind(helion, name, static, case=None, m=8):
    from helion.runtime.ref_mode import RefMode  # interpret mode has no config space
    case = case or {**EP8, "experts": 2, "experts_global": 16, "numtokens": m}
    kern = next(k for k in hs.KERNELS if k["name"] == name)
    args = kern["make"](case, 0)
    return helion.kernel(kern["body"], static_shapes=static, ref_mode=RefMode.OFF).bind(args)


@pytest.fixture(scope="module")
def real_shape_kernels():
    """The three bodies bound at the real V4.1 EP8 shapes (2 local experts keep the CPU weight
    build short; H, I, I_sh, topk are the real ones)."""
    pytest.importorskip("helion")
    return {}


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("name", [k["name"] for k in hs.KERNELS])
def test_search_spaces_are_bounded_and_codegen_is_clean(helion, real_shape_kernels, name, static):
    bound = real_shape_kernels.setdefault((name, static), _bind(helion, name, static))
    specs = list(bound.config_spec.block_sizes)
    assert [(sp.min_size, sp.max_size) for sp in specs] == list(hs.BOUNDS[name].values())
    assert max(sp.max_size for sp in specs) <= hs.CAPS[name]
    base = dict(bound.config_spec.default_config().config)
    for pick in (lambda sp: sp.min_size, lambda sp: sp.max_size, None):
        cfg = helion.Config(**(base if pick is None else
                               {**base, "block_sizes": [pick(sp) for sp in specs]}))
        bound.config_spec.normalize(cfg)
        code = bound.to_triton_code(cfg)
        compile(code, name, "exec")
        # stateless by construction: no atomics, barriers, semaphores or other cross-program state
        for word in ("atomic", "barrier", "semaphore", "gdc_wait", "tl.debug"):
            assert word not in code, (name, word)


def test_shared_region_is_first_and_in_the_same_launch(helion, real_shape_kernels):
    bound = real_shape_kernels.setdefault((hs.FC1_SH, False), _bind(helion, hs.FC1_SH, False))
    code = bound.to_triton_code(bound.config_spec.default_config())
    assert code.count("tl.program_id(0)") == 1 and code.count("@triton.jit") == 1
    kernel = code.split("def dsv41_moe_fc1_sh(")[0]  # the @triton.jit function, not the host launcher
    assert kernel.index("if pid_shared <") < kernel.index("else:")
    head, tail = kernel.split("else:", 1)
    assert "s_acc_g" in head and "s_acc_g" not in tail  # shared GEMM: the lowest program ids
    assert re.search(r"\bacc_g_1\b", tail) and not re.search(r"\bacc_g_1\b", head)  # routed fc1 behind


# ---------------------------------------------------------------------------
# lab ABI + serving decisions
# ---------------------------------------------------------------------------
def test_cases_and_serving_keys(monkeypatch):
    monkeypatch.delenv("NVFP4_MOE_TP8_NOEP", raising=False)
    ep8 = {**DS, "ep": 8, "experts": 48, "experts_total": 384, "moe_intermediate": 2304}
    for model in (DS, ep8):
        per = {k["name"]: k["cases"](model) for k in hs.KERNELS}
        assert {c["numtokens"] for c in per[hs.FC1_SH]} == set(hm.CLAMP_BUCKETS)
        assert {c["numtokens"] for c in per[hs.FC2_COMBINE_SH]} == {1, 6, 8, 16}
        assert {c["numtokens"] for c in per[hs.COMBINE_SH]} == {6, 8, 16, 32, 64, 128, 192}
        assert all(c["sh_inter"] == 288 and c["experts_global"] == 384 for v in per.values()
                   for c in v)
    assert hs.lane_case_sh(48, 5120, 2304, 6, 150, 10.0, 384, 288) == {**EP8, "numtokens": 192}
    assert hs.lane_case_sh(48, 5120, 2304, 6, 2, 10.0, 384, 288)["numtokens"] == 6
    assert hs.base_case(hs.lane_case_sh(48, 5120, 2304, 6, 2, 10.0, 384, 288)) \
        == hm.lane_case(48, 5120, 2304, 6, 0, 2, 10.0, 384)
    monkeypatch.setenv("NVFP4_MOE_TP8_NOEP", "1")  # the no-EP rank: same shared shard
    assert {c["sh_inter"] for v in [k["cases"](DS) for k in hs.KERNELS] for c in v} == {288}
    assert hs.cases({**DS, "quant_format": "bf16"}) == [] and hs.cases({**DS, "architecture":
                                                                        "GlmMoeDsaForCausalLM"}) == []


def test_served_ms_names_every_gap(monkeypatch):
    mode_for = lambda m: 2 if m <= 8 else 1
    have = {(hs.hm.ROUTE_QUANT, 1), (hs.FC1_SH, 1), (hs.FC2_COMBINE_SH, 1)}

    def config(kernel, case, plat=None):
        return {} if (kernel, case["numtokens"]) in have else None

    monkeypatch.setattr(hk, "config", config)
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    got = hs.served_ms_sh(48, 5120, 2304, 6, 1, mode_for, 10.0, 384, 288)
    assert got == {1: (1, "fused3")}
    with pytest.raises(RuntimeError) as exc:
        hs.served_ms_sh(48, 5120, 2304, 6, 12, mode_for, 10.0, 384, 288)
    msg = str(exc.value)
    assert "M<=6 fused3" in msg and "M<=16 front4" in msg and hs.COMBINE_SH in msg
    assert "SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M" in msg


def test_fused_plan_serving_call_and_warm_guard(helion, monkeypatch):
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: {})
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    hs._kernels_sh.cache_clear()
    p, sp, ids, tw, x, base, limit = _plan_inputs(7, "balanced", False)
    sh = hs.sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])
    mode_for = lambda m: 2 if m <= 8 else 1
    plan = hs.FusedPlan(6, 128, 288, 6, limit, 48, 64, 12, mode_for)
    assert plan.served[7] == (8, "fused3") and plan.served[12] == (16, "front4")
    args = lambda xx, ii, tt: (xx, ii, tt, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"],
                               p["g2"], p["a1g"], p["a2g"], base, sh)
    # a compile inside a capture is refused; outside a capture the first call warms that M
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="M=7 was not warmed before CUDA-graph capture"):
        plan(*args(x, ids, tw))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out = plan(*args(x, ids, tw))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert torch.equal(plan(*args(x, ids, tw)), out)  # warmed: allowed in a capture, same bits
    _, _, ids6, tw6, x6, _, _ = _plan_inputs(6, "balanced", False)  # same bucket, other M: Triton
    with pytest.raises(RuntimeError, match="M=6 was not warmed"):  # specializes on sizes
        plan(*args(x6, ids6, tw6))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run_m(m):
        _, _, ids_, tw_, x_, _, _ = _plan_inputs(m, "balanced", False)
        plan(*args(x_, ids_, tw_))

    assert plan.warm(run_m) == 4  # buckets 1, 6, 8 (fused3) and 16 (front4)


# ---------------------------------------------------------------------------
# knobs, eligibility, facts
# ---------------------------------------------------------------------------
def test_knobs_fail_closed(monkeypatch):
    monkeypatch.delenv(glue.KNOB, raising=False)
    monkeypatch.delenv(glue.MAX_M_ENV, raising=False)
    assert glue.fuse_on() is False
    for off in ("", "0"):
        monkeypatch.setenv(glue.KNOB, off)
        assert glue.fuse_on() is False
    monkeypatch.setenv(glue.KNOB, "1")
    assert glue.fuse_on() is True
    monkeypatch.setenv(glue.KNOB, "yes")
    with pytest.raises(RuntimeError, match="expected 0 or 1"):
        glue.fuse_on()
    monkeypatch.setenv(glue.KNOB, "1")
    assert glue.fuse_max_m(32) == 8 and glue.fuse_max_m(4) == 4  # default, capped by the lane
    monkeypatch.setenv(glue.MAX_M_ENV, "24")
    assert glue.fuse_max_m(32) == 24
    for bad, why in (("0", "1..192"), ("193", "1..192"), ("x", "1..192"), ("48", "exceeds")):
        monkeypatch.setenv(glue.MAX_M_ENV, bad)
        with pytest.raises(RuntimeError, match=why):
            glue.fuse_max_m(32)
    glue.check_env(True, "auto")
    for gate, backend, why in ((False, "auto", "SUFFIX_NVFP4_MOE=1"),
                               (True, "oxide", "SUFFIX_HELION_NVFP4_MOE")):
        with pytest.raises(RuntimeError, match=why):
            glue.check_env(gate, backend)
    monkeypatch.setenv(glue.KNOB, "0")
    glue.check_env(False, "oxide")  # off: nothing is checked


GOOD = dict(runner=True, shared=True, dp=1, mlp=True, kernel="FlashInferCutlassMxfp8LinearKernel",
            wdtype="float8_e4m3fn", sdtype="uint8", H=5120, I=288, routed_H=5120, limit=10.0,
            routed_limit=10.0, rsf=1.0, gu_shape=(576, 5120), dn_shape=(5120, 288),
            sgu_numel=640 * 160, sd_numel=5120 * 12)


@pytest.mark.parametrize("change,needle", [
    ({}, None),
    ({"kernel": "SuffixMxfp8LinearKernel"}, None),  # lane D's subclass keeps the layout
    ({"sdtype": "float8_e8m0fnu"}, None),
    ({"runner": False}, "runner hook"),
    ({"shared": False}, "no shared expert"),
    ({"shared_gate": True}, "gated"),
    ({"transform": True}, "transform"),
    ({"sp": True}, "sequence parallel"),
    ({"reduced": True}, "reduces its own output"),
    ({"mk_overlap": True}, "overlapped"),
    ({"bias": True}, "bias"),
    ({"reduce_results": True}, "all-reduces by itself"),
    ({"dp": 2}, "data parallel"),
    ({"rsf": 1.5}, "routed_scaling_factor"),
    ({"mlp": False}, "SiluAndMulWithClamp"),
    ({"kernel": "EmulationMxfp8LinearKernel"}, "MXFP8 layout"),
    ({"wdtype": "bfloat16"}, "float8_e4m3fn"),
    ({"sdtype": "float32"}, "e8m0"),
    ({"I": 290}, "whole 32-blocks"),
    ({"routed_H": 4096}, "hidden"),
    ({"gu_shape": (288, 5120)}, "weight shapes"),
    ({"sgu_numel": 576 * 160}, "F8_128x4"),
    ({"limit": 7.0}, "swiglu_limit"),
])
def test_shared_eligibility(change, needle):
    why = glue.shared_eligibility({**GOOD, **change})
    assert (why is None) if needle is None else (needle in why), why


class SiluAndMulWithClamp:
    def __init__(self, limit):
        self.swiglu_limit, self.alpha, self.beta = limit, 1.0, 0.0


class _Mlp:
    """vLLM's DeepseekV4MLP as the lane reads it: gate_up_proj / down_proj linears whose weight
    and weight_scale are the post-processed MXFP8 tensors, SiluAndMulWithClamp, callable."""

    def __init__(self, sp, kernel, limit, fn):
        lin = lambda w, s: types.SimpleNamespace(
            weight=types.SimpleNamespace(data=w, dtype=w.dtype, shape=w.shape, dim=lambda: 2),
            weight_scale=types.SimpleNamespace(data=s, dtype=s.dtype, numel=s.numel), bias=None,
            reduce_results=False,
            quant_method=types.SimpleNamespace(kernel=type(kernel, (), {})()))
        self.gate_up_proj = lin(sp["wgu"], sp["sgu"])
        self.down_proj = lin(sp["wd"], sp["sd"])
        self.act_fn = SiluAndMulWithClamp(limit)
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


class _FakeShared:
    """vLLM's SharedExperts as the lane drives it (maybe_forward_async / forward / wait / output)."""

    def __init__(self, mlp, aux):
        self._layer, self.aux, self._out, self.calls = mlp, aux, None, []

    def maybe_forward_async(self, x):
        self.calls.append("async" if self.aux else "no-async")
        if self.aux:
            self._out = self._layer(x)
        return self.aux

    def forward(self, x, order):
        self.calls.append(f"inline:{order}")
        self._out = self._layer(x)

    def wait(self):
        self.calls.append("wait")

    @property
    def output(self):
        out, self._out = self._out, None
        assert out is not None
        return out


def _fake_runner(sp, kernel="FlashInferCutlassMxfp8LinearKernel", limit=10.0, aux=True, fn=None,
                 **over):
    mlp = _Mlp(sp, kernel, limit, fn or (lambda x: x))
    r = types.SimpleNamespace(
        _shared_experts=_FakeShared(mlp, aux), moe_config=types.SimpleNamespace(
            is_sequence_parallel=False), shared_expert_gate=None, routed_input_transform=None,
        routed_output_transform=None, _fused_output_is_reduced=False, routed_scaling_factor=1.0,
        _select_forward=lambda: "moe_forward", _forward_entry="moe_forward_shared")
    for k, v in over.items():
        setattr(r, k, v)
    return r, mlp


def test_collect_facts_reads_the_post_processed_layout():
    sp = hs.make_shared_problem(5120, 288, seed=0)
    runner, mlp = _fake_runner(sp)
    info = {"H": 5120, "clamp_limit": 10.0, "dp": 1}
    facts, got = glue.collect_facts(runner, info)
    assert got is mlp and glue.shared_eligibility(facts) is None
    assert facts["gu_shape"] == (576, 5120) and facts["I"] == 288 and facts["limit"] == 10.0
    assert facts["sgu_numel"]() == 640 * 160 if callable(facts["sgu_numel"]) else True
    for over, needle in (({"routed_scaling_factor": 1.5}, "routed_scaling_factor"),
                         ({"_fused_output_is_reduced": True}, "reduces its own output"),
                         ({"shared_expert_gate": object()}, "gated")):
        r2, _ = _fake_runner(sp, **over)
        assert needle in glue.shared_eligibility(glue.collect_facts(r2, info)[0])
    r3, _ = _fake_runner(sp, kernel="SomethingElse")
    assert "MXFP8 layout" in glue.shared_eligibility(glue.collect_facts(r3, info)[0])
    r4, _ = _fake_runner(sp, limit=7.0)
    assert "swiglu_limit" in glue.shared_eligibility(glue.collect_facts(r4, info)[0])
    assert glue.collect_facts(None, info)[0]["runner"] is False


# ---------------------------------------------------------------------------
# capture safety: the AST audit and the capture-replay logic
# ---------------------------------------------------------------------------
def test_serving_path_has_no_host_sync_and_creates_no_streams():
    fns = glue.serving_functions()
    assert len(fns) >= 8
    for fn in fns:
        assert glue.audit_function(fn) == [], fn.__qualname__
    # the audit has teeth
    def bad(x):
        torch.cuda.synchronize()
        return x.sum().item() + torch.cuda.Stream()
    assert {w for _, w in glue.audit_function(bad)} >= {".synchronize", ".item", ".Stream"}


def _cpu_batch(cfg, hdim):
    return glue.batch_factory(cfg, hdim, torch.device("cpu"))


def _lane_cfg(p, e_global, base, topk):
    return dict(p, E_global=e_global, topk=topk, id_base=base)


def test_capture_replay_logic_on_cpu(helion):
    p, sp, ids, tw, x, base, limit = _plan_inputs(4, "balanced", False)
    cfg = _lane_cfg(p, 48, base, 6)
    sh = hs.sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])

    def run(x_, ids_, tw_):
        m = x_.shape[0]
        plan = "fused3" if m <= 8 else "front4"
        kerns = [helion.kernel(hs.BODIES_SH[k]) for k in hs.PLANS_SH[plan]]
        out = torch.empty(m, x_.shape[1], dtype=torch.bfloat16)
        return hs.run_plan_sh(kerns, plan, x_, ids_, tw_, p["w13"], p["w13_sf"], p["g1"], p["w2"],
                              p["w2_sf"], p["g2"], out, p["a1g"], p["a2g"], base, limit, sh)

    line = glue.capture_replay(run, _cpu_batch(cfg, 128), (1, 5, 11), rounds=2, cpu_graph=True)
    assert "PASS" in line and "3 graphs" in line and "CPU stand-in" in line
    calls = {"n": 0}

    def flaky(x_, ids_, tw_):  # depends on how often it ran: neither deterministic nor replayable
        calls["n"] += 1
        return run(x_, ids_, tw_) * (1.0 + 0.01 * (calls["n"] % 3))

    with pytest.raises(RuntimeError, match="moe_capture_replay FAIL"):
        glue.capture_replay(flaky, _cpu_batch(cfg, 128), (1, 5), rounds=2, cpu_graph=True)
    assert glue.gate_ms(6) == (1, 2, 6)


# ---------------------------------------------------------------------------
# prepare_layer / layer oracle / the serving state, on fakes
# ---------------------------------------------------------------------------
def _fake_vllm(monkeypatch):
    """vllm.model_executor.layers.fused_moe.runner.shared_experts.SharedExpertsOrder."""
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.runner",
             "vllm.model_executor.layers.fused_moe.runner.shared_experts"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    sys.modules[names[-1]].SharedExpertsOrder = types.SimpleNamespace(NO_OVERLAP=1)


def _stock_mlp(sp):
    """What vLLM's module returns: the shared MLP with MXFP8-quantized activations (W8A8)."""
    def q(t):
        qv, s = hs.quantize_mxfp8(t.float())
        return hs.dequant_mxfp8(qv, s, torch.float32)

    def mlp(x):
        gu = (q(x) @ sp["deq_gu"].float().T).bfloat16().float()
        i = sp["isz"]
        gate = gu[:, :i].clamp(max=sp["limit"])
        up = gu[:, i:].clamp(-sp["limit"], sp["limit"])
        h = (gate * torch.sigmoid(gate) * up).bfloat16()
        return (q(h) @ sp["deq_d"].float().T).bfloat16()
    return mlp


def _prepared(helion, monkeypatch, aux, kernel="FlashInferCutlassMxfp8LinearKernel",
              times=(10.0, 12.0, 11.0, 13.0)):
    """times: what the (fake) graph timer returns for fused M=1, split M=1, fused M=4, split M=4."""
    _fake_vllm(monkeypatch)
    queue = list(times)
    timed = []

    def graph_us(fn, dev, iters, flush=None):  # CPU stand-in for nvfp4_moe._graph_us
        fn(None)  # the function under test really runs
        timed.append(1)
        return queue.pop(0)

    monkeypatch.setattr(nm, "_graph_us", graph_us)
    monkeypatch.setitem(nm._state, "shared_fused", 0)
    monkeypatch.setattr(hk, "config", lambda kernel_, case, plat=None: {})
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    hs._kernels_sh.cache_clear()
    monkeypatch.setenv(glue.MAX_M_ENV, "4")
    hdim, idim, isz, local, eg, base, topk, limit = 128, 288, 64, 6, 48, 24, 6, 10.0
    p = nm.make_problem(local, hdim, idim, "silu", seed=11, limit=limit)
    sp = hs.make_shared_problem(hdim, isz, seed=3, limit=limit, wstd=0.05)
    cfg = dict(_lane_cfg(p, eg, base, topk), max_m=8, limit=limit, H=hdim,
               tune=nm.parse_tune(nm.DEFAULT_TUNE), fused=True)
    info = {"E": local, "H": hdim, "I": idim, "clamp_limit": limit, "dp": 1}
    runner, mlp = _fake_runner(sp, kernel=kernel, limit=limit, aux=aux, fn=_stock_mlp(sp))
    layer = types.SimpleNamespace(layer_name="model.layers.3.ffn.experts",
                                  w13_weight=types.SimpleNamespace(device=torch.device("cpu")),
                                  _sfx_runner=lambda: runner)

    def routed(x_, tw_, ids_):  # the lane's routed-only path (bf16 of the exact routed spec)
        return nm.moe_ref(cfg, x_, nm.to_local(ids_, base, local), tw_).bfloat16()

    return layer, cfg, info, runner, mlp, routed, sp, (hdim, local, base)


def test_long_pole_verdict_and_tolerance_knob(monkeypatch):
    rows = {1: (10.0, 12.0), 8: (21.0, 20.0)}
    assert glue.long_pole_verdict(rows, 1.05) is None
    why = glue.long_pole_verdict(rows, 1.02)
    assert "long pole at M=8" in why and "21.0 us" in why and glue.MAX_M_ENV in why
    monkeypatch.delenv(glue.POLE_TOL_ENV, raising=False)
    assert glue.pole_tol() == 1.05
    monkeypatch.setenv(glue.POLE_TOL_ENV, "1.2")
    assert glue.pole_tol() == 1.2
    for bad in ("0.9", "x"):
        monkeypatch.setenv(glue.POLE_TOL_ENV, bad)
        with pytest.raises(RuntimeError, match=">= 1.0"):
            glue.pole_tol()


def test_prepare_layer_refuses_when_fused_is_the_long_pole(helion, monkeypatch):
    layer, cfg, info, runner, mlp, routed, sp, _ = _prepared(
        helion, monkeypatch, True, times=(10.0, 12.0, 30.0, 12.0))  # fused 2.5x slower at M=4
    with pytest.raises(RuntimeError, match=r"LONG POLE GATE FAIL.*long pole at M=4"):
        glue.prepare_layer(layer, cfg, info, routed, routed)
    assert runner._shared_experts is not None and runner._forward_entry == "moe_forward_shared"


@pytest.mark.parametrize("aux", [True, False])
def test_prepare_layer_oracle_gate_and_split_path(helion, monkeypatch, aux):
    layer, cfg, info, runner, mlp, routed, sp, (hdim, local, base) = _prepared(
        helion, monkeypatch, aux)
    shared = runner._shared_experts
    parent = lambda x_, tw_, ids_: routed(x_, tw_, ids_)  # FlashInfer stand-in (same spec)
    state = glue.prepare_layer(layer, cfg, info, routed, parent)
    assert runner._shared_experts is None and runner._forward_entry == "moe_forward"
    assert state.fuse_max == 4 and nm._state["shared_fused"] >= 1
    assert state.plan.served[4] == (6, "fused3") and state.obj is shared
    make = glue.batch_factory(cfg, hdim, torch.device("cpu"))
    # fused: the f64 spec of routed + shared, and the shared module is never called
    shared.calls.clear()
    x, ids, tw = make(3, 5, "balanced")
    out = state.forward(routed, cfg, x, tw, ids)
    ref = nm.moe_ref(cfg, x, nm.to_local(ids, base, local), tw) + hs.shared_ref(sp, x)
    assert hm._rel(out.float(), ref) <= nm.REF_TOL and shared.calls == []
    # above the ceiling: routed-only + the module, exactly the runner's `shared + routed`
    x, ids, tw = make(6, 9, "skewed")
    got = state.forward(routed, cfg, x, tw, ids)
    assert torch.equal(got, routed(x, tw, ids) + mlp(x))
    assert shared.calls == (["async", "wait"] if aux else ["no-async", "inline:1"])
    # M outside the fused shape contract (non-bf16 / wrong hidden) takes the split path too
    assert not state.fused(x.float()) and not state.fused(torch.zeros(2, hdim + 32).bfloat16())
    assert not state.fused(torch.zeros(5, hdim).bfloat16()) and state.fused(x[:4])
    # the first fused layer timed fused vs split at M=1 and at the ceiling; the next one does not
    assert nm._state["shared_fused"] == 1
    layer2, cfg2, info2, runner2, mlp2, routed2, sp2, _ = _prepared(helion, monkeypatch, aux)
    nm._state["shared_fused"] = 1
    monkeypatch.setattr(nm, "_graph_us", lambda *a, **k: pytest.fail("only the first layer is timed"))
    glue.prepare_layer(layer2, cfg2, info2, routed2, routed2)


def test_prepare_layer_fails_closed(helion, monkeypatch):
    layer, cfg, info, runner, mlp, routed, sp, _ = _prepared(
        helion, monkeypatch, True, kernel="EmulationMxfp8LinearKernel")
    with pytest.raises(RuntimeError, match=r"cannot be fused: shared linears run on Emulation"):
        glue.prepare_layer(layer, cfg, info, routed, routed)
    assert runner._shared_experts is not None  # nothing was detached
    layer.layer_name = "x"
    layer._sfx_runner = None
    with pytest.raises(RuntimeError, match="runner hook did not fire"):
        glue.prepare_layer(layer, cfg, info, routed, routed)
    # a layer whose fused output is wrong never leaves the runner: break the routed spec
    layer, cfg, info, runner, mlp, routed, sp, _ = _prepared(helion, monkeypatch, True)
    broken = lambda x_, tw_, ids_: routed(x_, tw_, ids_) + 1.0
    with pytest.raises(RuntimeError, match="LAYER ORACLE FAIL"):
        glue.prepare_layer(layer, cfg, info, routed, broken)
    assert runner._shared_experts is not None and runner._forward_entry == "moe_forward_shared"
    # a missing tuned config is a startup error (no silent unfused M below the ceiling)
    layer, cfg, info, runner, mlp, routed, sp, _ = _prepared(helion, monkeypatch, True)
    monkeypatch.setattr(hk, "config", lambda kernel_, case, plat=None: None)
    with pytest.raises(RuntimeError, match="no tuned config"):
        glue.prepare_layer(layer, cfg, info, routed, routed)


# ---------------------------------------------------------------------------
# K4 harness pieces that need no GPU
# ---------------------------------------------------------------------------
def test_fuse_table_ceiling_is_the_unbroken_prefix():
    rows = {1: {"balanced": (10.0, 14.0, 8.0), "skewed": (10.0, 13.0, 8.0)},
            6: {"balanced": (30.0, 33.0, 25.0), "skewed": (31.0, 33.0, 25.0)},
            12: {"balanced": (60.0, 59.0, 50.0), "skewed": (58.0, 59.0, 50.0)},   # balanced loses
            24: {"balanced": (90.0, 120.0, 70.0), "skewed": (90.0, 120.0, 70.0)}}  # wins again
    text, best = glue.fuse_table("t", rows)
    assert best == 6 and "M=12" in text and "LOSS" in text
    assert f"{glue.MAX_M_ENV}=6" in text and "shared cost +2.0 vs +6.0" in text
    text, best = glue.fuse_table("t", {1: {"balanced": (10.0, 9.0, 5.0)}})
    assert best is None and f"{glue.KNOB}=0" in text


def test_stock_shared_baseline_needs_vllm_and_says_so():
    sp = hs.make_shared_problem(128, 64, seed=0)
    with pytest.raises(RuntimeError, match="needs vLLM \\+ FlashInfer"):
        glue.stock_shared(sp)


def test_bench_plan_covers_the_k4_grid():
    plan = glue._plan(nm.DS_EP, glue.BENCH_MS)
    assert plan.dims == (48, 5120, 2304, 6, 10.0, 384, 288) and plan.untuned_ok
    got = {m: plan.served[m] for m in glue.BENCH_MS}
    assert got == {1: (1, "fused3"), 6: (6, "fused3"), 12: (16, "front4"), 24: (32, "front4"),
                   48: (64, "front4"), 96: (128, "front4"), 192: (192, "front4")}
    assert hs.sh_inter_of({"tp": 8}, {"moe_intermediate": 2304, "experts_global": 384}) == 288


@pytest.mark.parametrize("name", [k["name"] for k in hs.KERNELS])
def test_unfused_lab_baseline_computes_the_same_math(helion, monkeypatch, name):
    """The runner's baseline arm (base stage kernel + torch bf16 shared GEMMs) agrees with the
    spec to bf16 level: it is the right thing to beat, and it pins the baseline's own indexing."""
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    hs._base_stage.cache_clear()
    kern = next(k for k in hs.KERNELS if k["name"] == name)
    case = _case(12, hd=512, isz=160)
    got = kern["baseline"](*kern["make_baseline"](case, 0))
    ref = kern["reference"](*kern["make"](case, 0))
    if name == hs.FC1_SH:
        assert hm._rel(got[1].float(), ref[1].float()) <= 1e-2
        valid = ref[0][1]
        assert hm._frac(got[0][valid].float(), ref[0][0][valid].float()) <= 5e-3
    else:
        assert hm._rel(got.float(), ref) <= nm.REF_TOL


# ---------------------------------------------------------------------------
# the lane's dispatch (SuffixRoutedExperts.forward_modular) on a fake vLLM base class
# ---------------------------------------------------------------------------
def _lane_class(monkeypatch):
    from suffix_hybrid import oxide_kernels
    mod = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")

    class RoutedExperts:  # vLLM's base: forward_modular is FlashInfer's path
        parent_calls = []

        def __init__(self, *a, **kw):
            self.quant_method = types.SimpleNamespace(process_weights_after_loading=lambda l: None)

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                            shared_experts_input=None):
            RoutedExperts.parent_calls.append(x.shape[0])
            return torch.full((x.shape[0], x.shape[1]), 5.0).bfloat16()

    mod.RoutedExperts = RoutedExperts
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(oxide_kernels, "native", lambda: types.SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: types.SimpleNamespace(
        cuda_stream=0))
    return nm._make_layer_cls(), RoutedExperts


def test_forward_modular_dispatch(monkeypatch):
    cls, base = _lane_class(monkeypatch)
    layer = cls()
    cfg = dict(max_m=8, H=16, tune=None, fused=True, id_base=0,
               op=lambda x, ids, tw, *rest: rest[6].copy_(torch.full_like(rest[6], 2.0)))  # rest[6] = out
    monkeypatch.setitem(nm._state["ws"], torch.device("cpu"), ())
    monkeypatch.setitem(nm._state, "active_logged", True)
    layer._sfx_moe = None
    x = lambda m: torch.zeros(m, 16, dtype=torch.bfloat16)
    ids, tw = torch.zeros(3, 2, dtype=torch.int32), torch.ones(3, 2)
    # lane off for this layer (no cfg): vLLM's path, shared untouched
    assert float(layer.forward_modular(x(3), tw, ids)[0, 0]) == 5.0
    layer._sfx_moe = cfg
    cfg.update(w2=torch.zeros(1, 16, 8, dtype=torch.uint8), a1g=1.0, a2g=1.0, act=0, w13=None,
               w13_sf=None, g1=None, w2_sf=None, g2=None)
    assert float(layer.forward_modular(x(3), tw, ids)[0, 0]) == 2.0       # ours
    assert float(layer.forward_modular(x(9), tw, ids)[0, 0]) == 5.0       # above max_m: vLLM's

    class Plan:
        calls = []

        def __call__(self, xx, ii, tt, *rest):
            Plan.calls.append(xx.shape[0])
            return torch.full((xx.shape[0], xx.shape[1]), 7.0).bfloat16()

    class Obj:
        def __init__(self):
            self.log, self.out = [], None

        def maybe_forward_async(self, xx):
            self.log.append("async")
            self.out = torch.full((xx.shape[0], xx.shape[1]), 1.0).bfloat16()
            return True

        def wait(self):
            self.log.append("wait")

        @property
        def output(self):
            return self.out

    obj = Obj()
    cfg["dsv41_shared"] = glue.SharedState(plan=Plan(), sh=(), obj=obj, no_overlap=1, mlp=None,
                                           hdim=16, isz=32, limit=10.0, fuse_max=4, name="l")
    assert float(layer.forward_modular(x(3), tw, ids)[0, 0]) == 7.0       # fused, plan called
    assert Plan.calls == [3] and obj.log == []
    assert float(layer.forward_modular(x(6), tw, ids)[0, 0]) == 3.0       # ours 2 + shared 1
    assert obj.log == ["async", "wait"] and Plan.calls == [3]
    obj.log.clear()
    assert float(layer.forward_modular(x(9), tw, ids)[0, 0]) == 6.0       # vLLM's 5 + shared 1
    assert base.parent_calls[-1] == 9 and obj.log == ["async", "wait"]


def test_register_refuses_the_knob_without_the_lane(monkeypatch):
    monkeypatch.setenv(glue.KNOB, "1")
    monkeypatch.delenv(nm.GATE, raising=False)
    with pytest.raises(RuntimeError, match="needs SUFFIX_NVFP4_MOE=1"):
        nm.register()
    monkeypatch.setenv(glue.KNOB, "0")
    assert nm.register() is None  # off: the previous plugin
