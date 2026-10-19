# SPDX-License-Identifier: Apache-2.0
"""K1-K3 on CPU for the native-FP4-MMA dense GEMM family (nvfp4_gemm_fp4mma.py): the four bodies in
Helion's eager mode against an f64 exact reference, with adversarial scales, K / N / M tails, forced small
tiles, split-K; Helion's real front end lowering every body to Triton source (no semaphore, one launch, the
host reduce after it); the serving plan, knob and twins; capture-safety audits; the capture/replay gate.

Eager mode proves INDEXING and NUMERICS. It does not prove speed, register pressure, or that Triton lowers
the dot to the native instruction (the P0 probe and K5 do; tests/test_fp4mma_ptx.py runs the first on Linux CI).
"""
import ast
import inspect
import json
import os

import numpy as np
import pytest
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
from suffix_hybrid.kernels import fp4mma_probe as probe
from suffix_hybrid.kernels import nvfp4_gemm as ng

from test_fp4mma_ref import adversarial_scales, cpu_mm, nvfp4_ref  # noqa: E402,F401

try:
    import helion
    import helion.language as hl
except ImportError:
    helion = hl = None
needs_helion = pytest.mark.skipif(helion is None, reason="Helion not installed")

KINDS = [fm.WORDS, fm.BYTES, fm.LIN, fm.CVT]
ref_of = lambda kind: fm.reference_lin if kind == fm.LIN else fm.reference_q


def kernel(kind, **cfg):
    return helion.kernel(fm.BODIES[kind], config=helion.Config(**cfg), static_shapes=False,
                         ignore_warnings=fm._ignore())


def tight(out, ref, rel=3e-5):
    """bf16 output rounding only: every element within 2^-8 |ref| (+ f32 accumulate noise)."""
    o, r = out.double(), ref.double()
    bad = ((o - r).abs() > 2.0 ** -8 * r.abs() + rel * r.abs().max()).sum()
    return int(bad)


@pytest.fixture
def small_tiles(monkeypatch):
    """Force many small tiles in eager mode (register_block_size returns the MAX there)."""
    def set_(bm=16, bn=32, bk=32):
        monkeypatch.setattr(fm, "BM_MIN", 16)
        monkeypatch.setattr(fm, "BM_MAX", bm)
        monkeypatch.setattr(fm, "BN_MIN", 16)
        monkeypatch.setattr(fm, "BN_MAX", bn)
        monkeypatch.setattr(fm, "BK_MIN", 32)
        monkeypatch.setattr(fm, "BK_MAX", bk)
    return set_


# ---------------------------------------------------------------------------
# K3: numerics, every body, every tail
# ---------------------------------------------------------------------------
M_SET = (1, 2, 15, 16, 17, 64)


@needs_helion
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", M_SET)
def test_body_vs_exact_all_m_tails(kind, m, nvfp4_ref, cpu_mm, small_tiles):
    """M in {1, 2, 15, 16, 17, 64} x N tail (48 = 32 + 16) x K tail (640 B-steps of 64: 5 of 128 bytes
    + a 64-byte one)."""
    small_tiles(16, 32, 64)
    p = fm.problem({"k": 640, "n": 48, "numtokens": m}, 3, "cpu", kind)
    args = fm.body_args(kind, p)
    out = kernel(kind, split_k=1)(*args)
    assert fm.check(out, ref_of(kind)(*args))[0]
    assert tight(out, ref_of(kind)(*args)) == 0


@needs_helion
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("tiles", [None, (16, 32, 32), (32, 64, 64)])
def test_body_splits_and_tiles(kind, split, tiles, nvfp4_ref, cpu_mm, small_tiles):
    if tiles:
        small_tiles(*tiles)
    p = fm.problem({"k": 2816, "n": 160, "numtokens": 17}, 5, "cpu", kind)
    args = fm.body_args(kind, p)
    ref = ref_of(kind)(*args)
    out = kernel(kind, split_k=split)(*args)
    assert tight(out, ref) == 0, f"{kind} split {split} tiles {tiles}"
    # the f32 partials sum to the f64 product (no bf16 rounding in the way)
    n_split = -(-1408 // (-(-1408 // (split * 32)) * 32)) if False else None
    if split > 1:
        part = p["partial"]
        k_split = 32 * (1 << (-(-1408 // (split * 32)) - 1).bit_length())
        n_split = -(-1408 // k_split)
        assert n_split > 1
        tot = part[:n_split].double().sum(0)
        assert float((tot - ref).norm() / ref.norm()) < 5e-6
        assert torch.isfinite(part[:n_split]).all()


@needs_helion
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [127, 129, 192])
def test_m_above_one_scale_atom(kind, m, nvfp4_ref, cpu_mm, small_tiles):
    """M > 128: the activation scales span several 128-row swizzle blocks ((row >> 7) * row-block stride)."""
    small_tiles(64, 32, 64)
    p = fm.problem({"k": 256, "n": 40, "numtokens": m}, 6, "cpu", kind)
    args = fm.body_args(kind, p)
    out = kernel(kind, split_k=2)(*args)
    assert tight(out, ref_of(kind)(*args)) == 0


@needs_helion
@pytest.mark.parametrize("kind", KINDS)
def test_adversarial_scales_zero_subnormal_saturated_outlier(kind, nvfp4_ref, cpu_mm, small_tiles):
    """Zero blocks, e4m3 subnormal scales, the smallest normal, saturated 448, mid values -- weights AND
    activations -- plus a single-element outlier row. Exact f64 reference, bf16 rounding is the only error."""
    small_tiles(16, 32, 32)
    m, n, k = 9, 40, 384
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 2, "cpu", kind)
    nkb = k // 16
    xs = adversarial_scales(m, nkb, 21)
    ws = adversarial_scales(n, nkb, 22)
    if kind == fm.LIN:
        p["xsf"], p["wsf"] = fm.as_words(xs), fm.as_words(ws)
    else:
        sx = torch.from_numpy(ng.swizzle_sf(xs.numpy()))
        sw = torch.from_numpy(ng.swizzle_sf(ws.numpy()))
        if kind == fm.BYTES:
            p["xsf"], p["wsf"] = sx.view(torch.float8_e4m3fn), sw.view(torch.float8_e4m3fn)
        else:
            p["xsf"], p["wsf"] = fm.as_words(sx), fm.as_words(sw)
    p["xq"][0, :] = 0x77  # a row of saturated +6.0 codes
    p["xq"][1, 3] = 0x0F  # a single nonzero element among zeros
    p["xq"][1, :3] = 0
    p["xq"][1, 4:] = 0
    args = fm.body_args(kind, p)
    ref = ref_of(kind)(*args)
    assert torch.isfinite(ref).all()
    out = kernel(kind, split_k=2)(*args)
    assert tight(out, ref) == 0
    # some blocks really are subnormal / zero / saturated (the test would pass vacuously otherwise)
    flat = torch.cat([xs.reshape(-1), ws.reshape(-1)])
    assert (flat == 0).any() and ((flat > 0) & (flat < 8)).any() and (flat == 0x7E).any()


@needs_helion
def test_low_nibble_first_order_is_what_the_body_reads(nvfp4_ref, cpu_mm):
    """One 16-block, K = 64: weights nibbles walk 1..8 (lo = even k); a wrong nibble order or a swapped
    pair changes the dot product by a known amount."""
    k, n, m = 64, 16, 1
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 0, "cpu", fm.WORDS)
    p["xq"].zero_()
    p["w"].zero_()
    p["xq"][0, 0] = 0x21  # k0 = 0.5 (code 1), k1 = 1.0 (code 2)
    p["w"][:, 0] = 0x43  # k0 = 1.5 (code 3), k1 = 2.0 (code 4)
    one = np.full((128, 4), 0x38, np.uint8)  # scale 1.0 everywhere
    p["xsf"] = fm.as_words(torch.from_numpy(ng.swizzle_sf(one)))
    p["wsf"] = fm.as_words(torch.from_numpy(ng.swizzle_sf(one)))
    p["alpha"] = 1.0
    out = kernel(fm.WORDS, split_k=1)(*fm.body_args(fm.WORDS, p))
    assert float(out[0, 0]) == 0.5 * 1.5 + 1.0 * 2.0  # 2.75: lo*lo + hi*hi (a swap gives 0.5*2 + 1*1.5 = 2.5)


@needs_helion
@pytest.mark.parametrize("kind", [fm.WORDS, fm.CVT])
def test_deterministic_bit_for_bit(kind, nvfp4_ref, cpu_mm):
    p = fm.problem({"k": 640, "n": 64, "numtokens": 17}, 4, "cpu", kind)
    args = fm.body_args(kind, p)
    k4 = kernel(kind, split_k=4)
    a = k4(*args).clone()
    assert torch.equal(a.view(torch.int16), k4(*args).view(torch.int16))  # fixed-order reduce, no atomics


@needs_helion
def test_unsplit_leaves_the_partial_alone_and_split_writes_only_its_own(nvfp4_ref, cpu_mm):
    p = fm.problem({"k": 320, "n": 32, "numtokens": 17}, 0, "cpu", fm.WORDS)
    p["partial"].fill_(float("nan"))
    kernel(fm.WORDS, split_k=1)(*fm.body_args(fm.WORDS, p))
    assert torch.isnan(p["partial"]).all()  # split_k = 1 never touches (or reads) the partial
    kernel(fm.WORDS, split_k=4)(*fm.body_args(fm.WORDS, p))
    # kh = 160 bytes -> k_split = 32 * next_pow2(cdiv(160, 4 * 32)) = 64 -> 3 splits written
    assert torch.isfinite(p["partial"][:3]).all() and torch.isnan(p["partial"][3:]).all()


@needs_helion
def test_k_below_one_tile_and_n_below_one_tile(nvfp4_ref, cpu_mm):
    for kind in (fm.WORDS, fm.BYTES):
        p = fm.problem({"k": 64, "n": 16, "numtokens": 3}, 1, "cpu", kind)
        args = fm.body_args(kind, p)
        out = kernel(kind, split_k=1)(*args)
        assert tight(out, ref_of(kind)(*args)) == 0


# ---------------------------------------------------------------------------
# Helion's real front end: Triton source (no GPU, no Triton): structure of what would be compiled
# ---------------------------------------------------------------------------
BANNED = ("atomic", "debug_barrier", "counters", "semaphore", "cooperative", "grid_sync", "barrier")


def lowered(kind, split, bs=(16, 64, 128), static=False, k=5120, n=256, m=16):
    from helion.runtime.ref_mode import RefMode
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 0, "cpu", kind)
    kern = helion.kernel(fm.BODIES[kind], static_shapes=static, ref_mode=RefMode.OFF,
                         ignore_warnings=fm._ignore())
    return kern.bind(fm.body_args(kind, p)).to_triton_code(
        helion.Config(split_k=split, block_sizes=list(bs), num_warps=4, num_stages=3))


@needs_helion
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("kind", KINDS)
def test_triton_source_is_semaphore_free_two_stage_and_native(kind, split, static):
    code = lowered(kind, split, static=static)
    assert not any(t in code for t in BANNED), [t for t in BANNED if t in code]
    assert code.count("_launcher(") == 1
    assert "out.copy_(partial[:n_split].sum(0))" in code  # the stage-2 reduce is host code after the launch
    assert code.index("_launcher(") < code.index("out.copy_(partial[:n_split]")
    if kind == fm.CVT:
        assert "tl.dot_scaled" not in code and "tl.dot(" in code
        assert code.count("tl.inline_asm_elementwise") == 2 and "cvt.rn.f16x2.e2m1x2" in code
        return
    call = [ln for ln in code.splitlines() if "tl.dot_scaled(" in ln and not ln.strip().startswith("#")]
    assert len(call) == 1 and call[0].count("'e2m1'") == 2, call
    assert "acc=" in call[0] and "out_dtype" not in call[0]  # f32 accumulator through the loop
    if kind == fm.BYTES:  # the scale operand is a plain e4m3 pointer gather (no join / bitcast)
        assert "tl.join" not in code and "bitcast" not in code
    else:  # int32 words, unpacked in registers and bit-cast to e4m3
        assert code.count("tl.join(") == 6 and code.count("tl.float8e4nv, bitcast=True") == 2
        assert "tl.uint8" in code


@needs_helion
def test_triton_source_mask_and_clamp_discipline_for_tails():
    """K / M tails: data loads masked to zero, the scale gathers clamped into the buffer (never OOB)."""
    code = lowered(fm.WORDS, 1, k=2816, n=160, m=17)
    loads = [ln.strip() for ln in code.splitlines() if "tl.load(" in ln]
    assert any("tl.load(xq" in ln and "mask_0" in ln and "mask_2" in ln and "other=0" in ln for ln in loads), loads
    assert any("tl.load(w +" in ln and "mask_2" in ln for ln in loads), loads
    assert "tl.minimum(" in code  # the quad / row clamps


@needs_helion
def test_bodies_are_module_level_literal_only_and_never_import_the_oxide_stack():
    # oxide-free: V4.1 has no oxide in the loop (no import of suffix_hybrid.oxide_kernels / ._native anywhere)
    for mod in (fm, probe):
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [(node.module or "") + "." + a.name for a in node.names] if isinstance(node, ast.ImportFrom)
                     else [])
            assert not any("oxide_kernels" in n or n.endswith("._native") or n.endswith("import _native")
                           for n in names), (mod.__name__, names)
    for kind, body in fm.BODIES.items():
        text = inspect.getsource(body)
        assert not any(t in text for t in ("atomic", "barrier", "counters"))


# ---------------------------------------------------------------------------
# knob, plan, twins
# ---------------------------------------------------------------------------
@pytest.fixture
def table(monkeypatch):
    """Shipped configs 'present' for (kernel, bucket); compiled() records the call. The CPU box has
    no Triton / SM12x: the stack gate is stubbed (it has its own tests)."""
    cfgs, calls = {}, []
    monkeypatch.setenv(fm.ENV, "1")
    monkeypatch.setattr(hk, "config", lambda name, case, plat=None: cfgs.get((name, case["numtokens"])))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")

    def compiled(body, name, case, **kw):
        def kern(*args):
            calls.append((name, case["numtokens"], args))
            return args[4] if name != fm.QUANT else None
        return kern
    monkeypatch.setattr(hk, "compiled", compiled)
    monkeypatch.setattr(probe, "require_native_stack", lambda: None)
    monkeypatch.setattr(probe, "install_recorder", lambda watch=None: True)
    monkeypatch.setattr(fm, "_PLANS", {})
    return cfgs, calls


def test_knob_default_off_and_loud(monkeypatch):
    monkeypatch.delenv(fm.ENV, raising=False)
    assert fm.enabled() is False
    monkeypatch.setenv(fm.ENV, "1")
    assert fm.enabled() is True
    monkeypatch.setenv(fm.ENV, "yes")
    with pytest.raises(RuntimeError, match="expected 0 or 1"):
        fm.enabled()
    monkeypatch.delenv(fm.ENV)
    with pytest.raises(RuntimeError, match="unset/0"):
        fm.prepare(64, 128, True, 8, "auto")  # the lane must not import us with the knob off


def test_prepare_auto_is_per_bucket_and_prefers_words(table):
    cfgs, _ = table
    cfgs.update({(fm.WORDS, 1): {}, (fm.BYTES, 1): {}, (fm.WORDS, 16): {"split_k": 4},
                 (fm.CVT, 64): {"split_k": 2}})
    plan = fm.prepare(5120, 2816, True, 70, "auto")
    assert len(plan) == 71 and plan[0] is None
    assert plan[1][0] == fm.WORDS and plan[1][2] == 1
    assert all(plan[m] is None for m in range(2, 9))  # bucket 8: nothing shipped -> the incumbent
    assert all(plan[m][0] == fm.WORDS and plan[m][2] == 4 for m in range(9, 17))
    assert all(plan[m][0] == fm.CVT for m in range(33, 65))
    assert all(plan[m] is None for m in range(65, 71))
    assert fm.describe(plan) == ("M1:nvfp4_gemm_q_fp4mma M2-8:incumbent M9-16:nvfp4_gemm_q_fp4mma(split<=4) "
                                 "M17-32:incumbent M33-64:nvfp4_gemm_q_fp4cvt(split<=2) M65-70:incumbent")
    assert fm.partial_elems(5120, [plan]) == 0  # the family allocates its partial per call


def test_prepare_helion_and_strict_fail_closed(table, monkeypatch):
    cfgs, _ = table
    cfgs.update({(fm.WORDS, b): {} for b in (1, 8)})
    assert fm.prepare(5120, 2816, True, 8, "helion")[8][0] == fm.WORDS
    with pytest.raises(RuntimeError, match="bucket 16 on gpu"):
        fm.prepare(5120, 2816, True, 16, "helion")
    monkeypatch.setenv(fm.STRICT_ENV, "1")
    with pytest.raises(RuntimeError, match="STRICT"):
        fm.prepare(5120, 2816, True, 16, "auto")  # auto + STRICT=1 is helion
    with pytest.raises(RuntimeError, match="never reaches the family"):
        fm.prepare(5120, 2816, True, 8, "oxide")
    with pytest.raises(RuntimeError, match="K=100"):
        fm.prepare(64, 100, True, 8, "auto")  # K % 64


def test_bf16_route_refuses_m_above_one_scale_atom(table):
    cfgs, _ = table
    cfgs.update({(fm.WORDS, b): {} for b in fm.BUCKETS})
    with pytest.raises(RuntimeError, match="capped at M <= 128"):
        fm.prepare(5120, 2816, False, 129, "auto")
    assert fm.prepare(5120, 2816, True, 192, "auto")[192][0] == fm.WORDS  # prequant: no cap below MAX_M


def test_prepare_bf16_route_needs_the_quant_config_too(table):
    cfgs, _ = table
    cfgs.update({(fm.WORDS, 1): {}, (fm.WORDS, 8): {}, (fm.QUANT, 8): {}})
    plan = fm.prepare(5120, 6144, False, 8, "auto")
    assert plan[1] is None  # gemm shipped, quant not: the bf16 route needs both
    assert plan[8][0] == f"{fm.QUANT}+{fm.WORDS}" and len(plan[8][1]) == 2


def test_warm_ms_covers_every_triton_specialization():
    assert fm.warm_ms(192) == [1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 191, 192]
    assert fm.warm_ms(16) == [1, 7, 8, 15, 16]


@needs_helion
def test_twin_prequant_passes_words_a_per_call_partial_and_no_counters(table):
    cfgs, calls = table
    cfgs.update({(fm.WORDS, 8): {"split_k": 4}, (fm.BYTES, 1): {}})
    m, n, k = 5, 64, 256
    fm.prepare(n, k, True, 8, "auto")
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 0, "cpu", fm.WORDS)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    lane_partial = torch.empty(4 * 8 * n)
    lane_counters = torch.zeros(7, dtype=torch.int32)
    fm.nvfp4_gemm_q_cuda(p["xq"], p["xsf_swz"], p["w"], p["wsf_swz"], lane_partial, out, 0.25, 4, 0, 0,
                         lane_counters)
    name, bucket, args = calls[-1]
    assert (name, bucket) == (fm.WORDS, 8)
    xq, xs, w, ws, o, partial, alpha = args
    assert xs.dtype == ws.dtype == torch.int32 and xs.data_ptr() == p["xsf_swz"].data_ptr()
    assert partial.shape == (4, m, n) and partial.data_ptr() != lane_partial.data_ptr()
    assert alpha == 0.25 and o is out and not lane_counters.any()
    # M = 1: the bytes body, an unsplit config -> the (1, 1, 1) stub, scales as e4m3 bytes
    fm.nvfp4_gemm_q_cuda(p["xq"][:1], p["xsf_swz"], p["w"], p["wsf_swz"], lane_partial, out[:1], 0.5, 1, 0)
    name, _, args = calls[-1]
    assert name == fm.BYTES and args[1].dtype == torch.float8_e4m3fn and args[5].shape == (1, 1, 1)
    with pytest.raises(RuntimeError, match="no fp4mma plan"):
        fm.nvfp4_gemm_q_cuda(p["xq"][:3].repeat(3, 1), p["xsf_swz"], p["w"], p["wsf_swz"], lane_partial,
                             out, 0.5, 1, 0)  # M = 9: bucket 16, nothing shipped


@needs_helion
def test_twin_bf16_route_quantizes_into_per_call_buffers(table):
    cfgs, calls = table
    cfgs.update({(fm.WORDS, 8): {}, (fm.QUANT, 8): {}})
    m, n, k = 5, 64, 256
    fm.prepare(n, k, False, 8, "auto")
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 0, "cpu", fm.WORDS)
    x = torch.randn(m, k).bfloat16()
    out = torch.empty(m, n, dtype=torch.bfloat16)
    aq, asf = torch.zeros(1), torch.zeros(1)
    fm.nvfp4_gemm_cuda(x, p["w"], p["wsf_swz"], aq, asf, torch.empty(1), out, 3.0, 0.5, 1, 0)
    (qn, _, qargs), (gn, _, gargs) = calls[-2:]
    assert (qn, gn) == (fm.QUANT, fm.WORDS)
    assert qargs[0].shape == (m, k // 16, 16) and qargs[1].shape == (m, k // 16, 8)
    assert qargs[2].shape == (128 * (k // 16),) and qargs[2].dtype == torch.float8_e4m3fn
    assert gargs[0].shape == (m, k // 2) and gargs[0].data_ptr() == qargs[1].data_ptr()
    assert aq.numel() == 1 and asf.numel() == 1  # the lane's aq / asf are NOT written


@needs_helion
@pytest.mark.parametrize("split", [1, 2])
def test_twin_end_to_end_prequant_and_bf16_route(split, monkeypatch, nvfp4_ref, cpu_mm, small_tiles):
    """Real (eager) kernels through the twins: the lane's call -> bf16 out, vs the exact product."""
    small_tiles(16, 32, 64)
    monkeypatch.setenv(fm.ENV, "1")
    monkeypatch.setattr(probe, "require_native_stack", lambda: None)
    monkeypatch.setattr(probe, "install_recorder", lambda watch=None: True)
    monkeypatch.setattr(hk, "platform", lambda device=0: "cpu")
    old = fm._old()
    real = {fm.WORDS: kernel(fm.WORDS, split_k=split),
            fm.QUANT: helion.kernel(old.BODIES[fm.QUANT], config=helion.Config(), static_shapes=False)}
    monkeypatch.setattr(hk, "config", lambda name, case, plat=None: {"split_k": split})
    monkeypatch.setattr(hk, "compiled", lambda body, name, case, **kw: real[name])
    monkeypatch.setattr(fm, "_PLANS", {})
    m, n, k = 11, 96, 384
    fm.prepare(n, k, True, 16, "auto")
    fm.prepare(n, k, False, 16, "auto")
    p = fm.problem({"k": k, "n": n, "numtokens": m}, 3, "cpu", fm.WORDS)
    out = torch.zeros(m, n, dtype=torch.bfloat16)
    fm.nvfp4_gemm_q_cuda(p["xq"], p["xsf_swz"], p["w"], p["wsf_swz"], torch.empty(1), out, p["alpha"], 1, 0)
    assert tight(out, ng.exact_ref(p["xq"], p["xsf_swz"], p["w"], p["wsf_swz"], p["alpha"], n)) == 0
    # bf16 route: x -> spec quant -> exact product
    pb = old.problem({"k": k, "n": n, "numtokens": m}, 4, "cpu", False)
    out2 = torch.zeros(m, n, dtype=torch.bfloat16)
    fm.nvfp4_gemm_cuda(pb["x"], pb["w"], pb["wsf"], None, None, torch.empty(1), out2, pb["g"], pb["alpha"], 1, 0)
    assert tight(out2, old.reference(pb["x"].view(m, k // 2, 2), pb["w"], pb["wsf"], pb["out"], pb["partial"],
                                     pb["counters"], pb["inv"], pb["g6"], pb["alpha"])) == 0


# ---------------------------------------------------------------------------
# capture-safety audits (the standing rule: no atomics / barriers / semaphores / counters, scratch from
# torch's allocator, nothing created on first use, no host sync in the served paths)
# ---------------------------------------------------------------------------
SERVED_FUNCS = ("gemm", "quant_gemm", "nvfp4_gemm_q_cuda", "nvfp4_gemm_cuda")


def _calls(fn):
    out = []
    for node in ast.walk(ast.parse(inspect.getsource(fn))):
        if isinstance(node, ast.Call):
            f = node.func
            out.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_served_paths_have_no_host_sync_no_streams_no_atomics():
    banned = {"item", "cpu", "tolist", "synchronize", "Stream", "Event", "record", "wait_stream",
              "atomic_add", "atomic_cas", "zeros", "full"}  # zeros / full: a fill that is not the poison
    for name in SERVED_FUNCS:
        calls = set(_calls(getattr(fm, name)))
        assert not (calls & banned), (name, calls & banned)


def test_module_holds_no_device_scratch():
    """No module-level tensors / persistent buffers (the glm g14 suspect class): the plan is data only."""
    for k, v in vars(fm).items():
        assert not isinstance(v, torch.Tensor), k
    assert not hasattr(fm, "_SCRATCH") and not hasattr(fm, "_NEED")
    assert fm.partial_elems(4096, []) == 0


def test_bodies_write_no_state_other_than_out_and_partial():
    for body in fm.BODIES.values():
        tree = ast.parse(inspect.getsource(body))
        stores = {ast.unparse(t.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Subscript)}
        assert stores <= {"out", "partial"}, (body.__name__, stores)


# sabotaged body: forgets to store split 1's partial (a kernel that "worked" only because torch.empty
# happened to hold the right bytes) -- module level for Helion's source lookup
if helion is not None:
    def skip_one_split_body(xq: torch.Tensor, xsf: torch.Tensor, w: torch.Tensor, wsf: torch.Tensor,
                            out: torch.Tensor, partial: torch.Tensor, alpha: float) -> torch.Tensor:
        m = xq.size(0)
        n = hl.specialize(w.size(0))
        kh = hl.specialize(w.size(1))
        cp4 = (kh // 8 + 3) // 4
        rw = cp4 * 128
        cap = hl.specialize(partial.size(0))
        split_k = hl.register_tunable("split_k", helion.autotuner.PowerOfTwoFragment(1, cap, 1))
        k_split = helion.next_power_of_2(helion.cdiv(kh, split_k * 32)) * 32
        n_split = helion.cdiv(kh, k_split)
        for tile_m, tile_n, tile_s in hl.tile([m, n, kh], block_size=[16, 32, k_split]):
            rm = torch.clamp(tile_m.index, max=m - 1)[:, None]
            rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
            ra = (rm >> 7) * rw + (rm & 31) * 4 + ((rm >> 5) & 3)
            rb = (rn >> 7) * rw + (rn & 31) * 4 + ((rn >> 5) & 3)
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=32):
                q = torch.clamp(torch.amin(tile_k.index.reshape(-1, 32), dim=1) >> 5, max=cp4 - 1)[None, :] * 128
                wa = xsf[ra + q]
                wb = wsf[rb + q]
                sa = hl.join(hl.join(wa & 255, (wa >> 16) & 255), hl.join((wa >> 8) & 255, (wa >> 24) & 255))
                sa = sa.reshape(sa.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
                sb = sb.reshape(sb.size(0), -1).to(torch.uint8).view(torch.float8_e4m3fn)
                acc = hl.dot_scaled(xq[tile_m, tile_k], sa, "e2m1", w[tile_n, tile_k].T, sb, "e2m1", acc=acc)
            if tile_s.id != 1:
                partial[tile_s.id, tile_m, tile_n] = acc * alpha
        out.copy_(partial[:n_split].sum(0))
        return out


@needs_helion
def test_poisoned_partial_exposes_an_unwritten_split(nvfp4_ref, cpu_mm):
    p = fm.problem({"k": 256, "n": 32, "numtokens": 5}, 0, "cpu", fm.WORDS)
    kern = helion.kernel(skip_one_split_body, config=helion.Config(split_k=2), static_shapes=False,
                         ignore_warnings=fm._ignore())
    entry = (fm.WORDS, (kern,), 2, 5)
    out = torch.empty(5, 32, dtype=torch.bfloat16)
    got = fm.gemm(p["xq"], p["xsf_swz"], p["w"], p["wsf_swz"], out, entry, p["alpha"], poison_nan=True)
    ok, msg = fm.check(got, fm.reference_q(*fm.body_args(fm.WORDS, p)))
    assert not ok and "non-finite" in msg  # the poison makes the forgotten split visible


# ---------------------------------------------------------------------------
# the capture/replay gate, CPU twin
# ---------------------------------------------------------------------------
@needs_helion
def test_capture_replay_gate_tiny_tier_passes_on_the_eager_runtime(nvfp4_ref, cpu_mm):
    res = probe.run_gate(rt=probe.EagerRuntime, dev="cpu", plist=fm.probes("tiny"),
                         builders={"gemm": fm.build_probe}, replays=3, streams=1, emit=lambda s: None)
    assert res["verdict"] == "PASS", [r for r in res["probes"] if r["status"] != "PASS"]
    assert {r["label"] for r in res["probes"]} == {p.label for p in fm.probes("tiny")}
    assert all(r["detail"].endswith("guards intact") for r in res["probes"])


@needs_helion
def test_capture_replay_gate_fails_a_kernel_with_an_unwritten_split(nvfp4_ref, cpu_mm, monkeypatch):
    monkeypatch.setitem(fm.BODIES, fm.WORDS, skip_one_split_body)  # build_probe binds BODIES[body]
    res = probe.run_gate(rt=probe.EagerRuntime, dev="cpu", plist=[fm.probes("tiny")[1]],
                         builders={"gemm": fm.build_probe}, replays=2, streams=1, emit=lambda s: None)
    assert res["verdict"] == "FAIL" and "non-finite" in res["probes"][0]["detail"]


# ---------------------------------------------------------------------------
# lab ABI
# ---------------------------------------------------------------------------
QWEN27 = {"name": "models/qwen3.8-27b-fable-distill-nvfp4", "tp": 1, "quant_format": "nvfp4",
          "hidden": 5120, "vocab": 248320, "linears": []}
DSV41 = {"name": "deepseek-v4.1-flash-1017-nvfp4", "tp": 8, "quant_format": "nvfp4", "hidden": 5120,
         "vocab": 16160, "linears": [{"name": "qkv_proj", "n": 4096, "k": 5120}]}


def test_kernels_abi_and_cases():
    names = [e["name"] for e in fm.KERNELS]
    assert names == [fm.WORDS, fm.BYTES, fm.CVT, fm.LIN] and len(set(names)) == 4
    assert all(1 <= len(n) <= 40 and n == n.lower() for n in names)
    for e in fm.KERNELS:
        assert e["body"] is fm.BODIES[e["name"]] and callable(e["check"]) and e["replaces"] == []
        assert "ignore_warnings" in e["settings"]
        for c in e["cases"](QWEN27) + e["cases"](DSV41):
            assert c["numtokens"] in fm.BUCKETS and json.loads(hk.case_key(c)) == c
            assert set(c) == {"k", "n", "numtokens"}
        assert e["cases"]({**QWEN27, "quant_format": "fp8_block"}) == []
    dense = list(dict.fromkeys(ng.SHAPES.values()))
    q = fm.KERNELS[0]["cases"](QWEN27)
    assert {(c["n"], c["k"]) for c in q} == set(dense) | {(248320, 5120)}
    assert [c["numtokens"] for c in q[:len(dense) + 1]] == [1] * (len(dense) + 1)  # decode first
    # V4.1: NO NVFP4 dense layer (attention / shared experts are MXFP8): the lm_head only, per TP8 rank
    assert {(c["n"], c["k"]) for c in fm.KERNELS[0]["cases"](DSV41)} == {fm.V41_LM_HEAD}
    assert len(fm.KERNELS[0]["cases"](DSV41)) == len(fm.BUCKETS) <= 64


def test_prefill_cases_are_opt_in(monkeypatch):
    monkeypatch.setenv("FP4MMA_PREFILL_CASES", "1")
    assert {c["numtokens"] for c in fm.KERNELS[0]["cases"](DSV41)} == set(fm.BUCKETS) | set(fm.PREFILL_BUCKETS)


@needs_helion
@pytest.mark.parametrize("kind", KINDS)
def test_lab_roundtrip_make_run_reference_check(kind, nvfp4_ref, cpu_mm):
    """What the console runner does: out = tuned(*make(case, 0)) must pass check(out, reference(*args))."""
    e = next(x for x in fm.KERNELS if x["name"] == kind)
    args = e["make"]({"k": 192, "n": 48, "numtokens": 5}, 0)
    out = kernel(kind, split_k=2)(*args)
    ok, msg = e["check"](out, e["reference"](*args))
    assert ok, msg
    # out / partial are ZERO-initialized (the autotuner probes which args a baseline run mutates)
    assert fm.problem({"k": 192, "n": 48, "numtokens": 5}, 0, "cpu", kind)["partial"].eq(0).all()


def test_v41_shape_table_matches_the_checkpoint():
    assert fm.V41_LM_HEAD == (129280 // 8, 5120) and fm.V41_LM_HEAD[0] % 32 == 0 and fm.V41_LM_HEAD[1] % 64 == 0


def test_verify_served_asserts_native_for_dot_bodies_and_cvt_for_the_control(monkeypatch):
    seen = {}

    def fake_native(names, allow_local=True):
        seen["native"] = list(names)
        return []

    def fake_contains(names, tokens, what):
        seen["cvt"] = (list(names), list(tokens))
        return []

    monkeypatch.setattr(probe, "assert_native", fake_native)
    monkeypatch.setattr(probe, "assert_contains", fake_contains)
    plan = [None, (fm.WORDS, (None,), 1, 1), (f"{fm.QUANT}+{fm.BYTES}", (None, None), 2, 8),
            (fm.CVT, (None,), 1, 16), None]
    fm.verify_served([plan, [None]])
    assert seen["native"] == sorted([fm.TRITON_NAMES[fm.WORDS], fm.TRITON_NAMES[fm.BYTES]])
    assert seen["cvt"] == ([fm.TRITON_NAMES[fm.CVT]], ["cvt.rn.f16x2.e2m1x2"])
    seen.clear()
    assert fm.verify_served([[None, None]]) == [] and seen == {}  # nothing planned: nothing to assert
