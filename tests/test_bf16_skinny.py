# SPDX-License-Identifier: Apache-2.0
"""BF16 skinny GEMM, Helion mode (helion_kernels/bf16_skinny.py + kernels/bf16_skinny.py):
tuner cases from the Qwen4Exp model dicts, the body in interpret mode vs the exact
reference (splits, N = 1, unaligned M), the bf16-GEMM error bound, Triton codegen and the
generated HOST code (one launch, then the stage-2 reduce, no atomics), the per-bucket plan,
and the serving lane on CPU (stubbed vLLM linear classes, EagerRuntime for graph capture):
knob off = no wiring, auto without configs = F.linear, armed = op path + K5 log, the oracle
refuses a wrong kernel, the first-forward check fails closed."""
import json
import os
import sys
import types

os.environ["HELION_INTERPRET"] = "1"  # before any helion.kernel is built

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import bf16_skinny as hb  # noqa: E402
from suffix_hybrid.kernels import bf16_skinny as lane  # noqa: E402

QWEN = {"architecture": "Qwen4ExpForConditionalGeneration", "hidden": 2560,
        "experts_global": 512, "experts": 256, "shared_expert_intermediate": 640}


@pytest.fixture
def cpu_mm(monkeypatch):
    mm = torch.mm  # CPU aten::mm has no out_dtype (hl.dot's eager reference uses it)
    monkeypatch.setattr(torch, "mm", lambda a, b, *, out_dtype=None, **kw: (
        mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)))


def test_cases_qwen_router_and_shared_gate_only():
    cs = hb.cases(QWEN)
    assert {(c["n"], c["k"]) for c in cs} == {(512, 2560), (1, 2560)}
    assert sorted({c["numtokens"] for c in cs}) == list(hb.BUCKETS)
    assert all(json.loads(hk.case_key(c)) == c for c in cs)
    assert hb.cases({**QWEN, "architecture": "Qwen4ExpMTP"}) == cs  # the drafter: same shapes
    assert hb.cases({**QWEN, "shared_expert_intermediate": None}) == [
        c for c in cs if c["n"] == 512]
    assert hb.cases({**QWEN, "architecture": "DeepseekV32ForCausalLM"}) == []  # fp32 router
    assert hb.cases({"architecture": "Qwen3ForCausalLM", "hidden": 4096}) == []


def test_tuner_model_dicts_feed_the_family():
    from suffix_hybrid.tools import helion_tune as ht
    assert "bf16_skinny" in ht.FAMILIES
    from tests.test_helion_tune import CONFIG, NAME
    got = ht.collect(hb.KERNELS[0], ht.layouts(CONFIG, [2], "on", NAME))
    assert {(c["n"], c["k"]) for c, _ in got} == {(512, 2560), (1, 2560)}  # replicated router
    assert len(got) == 2 * len(hb.BUCKETS)
    assert all(set(lab) == {"tp2-ep", "tp2-ep-mtp"} for _, lab in got)


def test_max_splits_and_plan_helpers():
    assert hb.max_splits(512, 2560) == 32 and hb.max_splits(1, 2560) == 32
    assert hb.max_splits(1, 96) == 2 and hb.max_splits(8192, 2560) == 4
    e8, e32 = ("k8", 4, 8), ("k32", 1, 32)
    plan = [None] + [None] * 4 + [e8] * 4 + [None] * 8 + [e32] * 16 + [None] * 8  # max_m 40
    assert hb.describe(plan) == ("M1-4:F.linear M5-8:helion(b8,split 4) M9-16:F.linear "
                                 "M17-32:helion(b32) M33-40:F.linear")
    assert hb.oracle_ms(plan) == [5, 7, 8, 17, 25, 32]


def test_bound_is_tight_and_bites():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 2560, generator=g).bfloat16()
    w = (torch.randn(512, 2560, generator=g) * 0.02).bfloat16()
    ref = hb.exact(x, w)
    y = ref.float().bfloat16()  # fp32 sum (any order) + one rounding: inside
    assert hb.check_exact(y, x, w, ref)[0] and hb.check(y, ref)[0]
    assert hb.check_exact(torch.nn.functional.linear(x, w), x, w, ref)[0]
    bad = y.clone()
    bad[3, 100] = bad[3, 100].float() * (1 + 2 ** -6)  # two bf16 ulps off on one element
    assert not hb.check_exact(bad, x, w, ref)[0]
    half = y.clone()
    half[:, 7] = (ref[:, 7] / 2).bfloat16()  # a lost split
    assert not hb.check(half, ref)[0] and not hb.check_exact(half, x, w, ref)[0]
    assert not hb.check_exact(torch.full_like(y, float("nan")), x, w, ref)[0]


# ---------------------------------------------------------------------------
# interpret mode: the body vs the exact reference
# ---------------------------------------------------------------------------
CASES = [{"k": 2560, "n": 512, "numtokens": 5},   # router, c1 MTP verify
         {"k": 2560, "n": 1, "numtokens": 17},    # shared_expert_gate, M past one m16 tile
         {"k": 96, "n": 48, "numtokens": 1},      # N tail at bn 32, K tail
         {"k": 320, "n": 40, "numtokens": 33}]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"m{c['numtokens']}n{c['n']}k{c['k']}")
def test_body_interpret_vs_reference(case, cpu_mm):
    helion = pytest.importorskip("helion")
    x, w, out, partial = hb.problem(case, 0, "cpu")
    ref = hb.reference(x, w, out, partial)
    for split in (1, 2, 4):
        kern = helion.kernel(hb.bf16_skinny_body, config=helion.Config(split_k=split),
                             ignore_warnings=hb._ignore())
        for _ in range(2):  # a replay on the same buffers
            y = kern(x, w, out, partial)
            ok, msg = hb.check_exact(y, x, w, ref)
            assert ok, f"split_k={split}: {msg}"
            assert hb.check(y, ref)[0]


def test_two_stage_is_deterministic_and_leaves_partial_alone_when_unsplit(cpu_mm):
    helion = pytest.importorskip("helion")
    x, w, out, partial = hb.problem({"k": 320, "n": 40, "numtokens": 17}, 0, "cpu")
    partial.fill_(float("nan"))
    helion.kernel(hb.bf16_skinny_body, config=helion.Config(split_k=1),
                  ignore_warnings=hb._ignore())(x, w, out, partial)
    assert torch.isnan(partial).all(), "split_k=1 must not touch (or read) the partial"
    two = helion.kernel(hb.bf16_skinny_body, config=helion.Config(split_k=4),
                        ignore_warnings=hb._ignore())
    y = two(x, w, out, partial).clone()
    assert torch.equal(y.view(torch.int16), two(x, w, out, partial).view(torch.int16))
    assert torch.isfinite(partial[:3]).all()  # K 320 -> k_split 128 -> 3 splits written


# ---------------------------------------------------------------------------
# Helion's real front end: Triton source + the generated host code run on CPU
# ---------------------------------------------------------------------------
def _host_run(args, config, static):
    """tests/test_helion_kgdn1.py::_host_run: bind + Triton codegen at `config`, exec the
    generated module with triton stubbed, call its HOST function with a recording
    launcher (the kernel never runs; the host code's own torch work does, on CPU)."""
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    bound = helion.kernel(hb.bf16_skinny_body, static_shapes=static, ref_mode=RefMode.OFF,
                          ignore_warnings=hb._ignore()).bind(tuple(args))
    src = bound.to_triton_code(helion.Config(**{**bound.config_spec.default_config().config,
                                                **config}))
    tl = types.SimpleNamespace(constexpr=lambda v=None: v)
    ns = {"triton": types.SimpleNamespace(jit=lambda f: f, language=tl), "tl": tl,
          "libdevice": None, "__name__": "bf16_skinny_host_run"}
    exec(compile("\n".join(ln for ln in src.splitlines() if not ln.startswith(
        ("import triton", "from torch._inductor.runtime.triton_compat"))), "bf16_skinny", "exec"),
        ns)
    launches = []
    res = ns["bf16_skinny_body"](*args, _launcher=lambda k, grid, *a, **kw: launches.append(
        (k.__name__, grid)))
    return src, res, launches


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("case", [{"k": 2560, "n": 512, "numtokens": 1},
                                  {"k": 2560, "n": 1, "numtokens": 160},
                                  {"k": 2560, "n": 512, "numtokens": 192}],
                         ids=lambda c: f"m{c['numtokens']}n{c['n']}")
def test_host_code_one_launch_then_reduce(case, static):
    for split in (1, 4):
        args = hb.problem(case, 0, "cpu")
        x, w, out, partial = args
        partial.fill_(0.5)  # the host reduce must sum exactly the first n_split slices
        out.zero_()
        src, res, launches = _host_run(args, {"split_k": split, "block_sizes": [16, 32, 64]},
                                       static)
        n_split = -(-2560 // (1 << (-(-2560 // split) - 1).bit_length()))
        tiles = -(-case["numtokens"] // 16) * -(-case["n"] // 32) * n_split
        assert launches == [("_helion_bf16_skinny_body", (tiles,))]
        assert res is out
        assert not any(t in src for t in ("atomic", "debug_barrier", "semaphore", "torch.ops"))
        assert src.count("_launcher(") == 1
        if split > 1:
            assert src.index("_launcher(") < src.index("out.copy_(partial[:n_split].sum(0))")
            assert torch.all(out.float() == 0.5 * n_split)  # stage 2 ran on the host
        else:
            assert not out.any()  # no host-side work at split 1


# ---------------------------------------------------------------------------
# serving plan + lane (CPU: stubbed vLLM classes, interpret-mode kernels)
# ---------------------------------------------------------------------------
@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """configs/helion/bf16_skinny/<platform>.json with buckets 8 (split 2) and 32 (split 1)
    for N 40 K 320, platform pinned (no GPU)."""
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path))
    monkeypatch.setattr(hk, "platform", lambda device=0: "cpu_test")
    monkeypatch.setattr(hk, "_TABLES", {})
    monkeypatch.setattr(hk, "_COMPILED", {})
    d = tmp_path / hb.KERNEL
    d.mkdir()
    rows = [{"key": hb.case_of(40, 320, 8), "config": {"split_k": 2}},
            {"key": hb.case_of(40, 320, 32), "config": {"split_k": 1}}]
    (d / "cpu_test.json").write_text(json.dumps(rows))
    return rows


def test_prepare_auto_and_helion(shipped):
    plan = hb.prepare(40, 320, 40, "auto")
    assert len(plan) == 41 and plan[0] is None
    assert all(plan[m] is None for m in (1, 2, 3, 4, 9, 16, 33, 40))
    assert all(plan[m][1:] == (2, 8) for m in range(5, 9))
    assert all(plan[m][1:] == (1, 32) for m in range(17, 33))
    assert hb.describe(plan) == ("M1-4:F.linear M5-8:helion(b8,split 2) M9-16:F.linear "
                                 "M17-32:helion(b32) M33-40:F.linear")
    with pytest.raises(RuntimeError, match="no tuned bf16_skinny config.*M bucket 1"):
        hb.prepare(40, 320, 40, "helion")
    with pytest.raises(RuntimeError, match="lane is off"):
        hb.prepare(40, 320, 40, "oxide")
    assert hb.prepare(512, 2560, 192, "auto") == [None] * 193  # nothing shipped: F.linear


def test_run_matches_reference(shipped, cpu_mm):
    pytest.importorskip("helion")
    plan = hb.prepare(40, 320, 32, "auto")
    for m in (5, 7, 8, 17, 32):
        x, w, _, _ = hb.problem(hb.case_of(40, 320, m), 3, "cpu")
        ok, msg = hb.check_exact(hb.run(x, w, plan[m], poison_nan=True), x, w)
        assert ok, (m, msg)


class _Unq:
    def __init__(self):
        self.apply = self._apply

    def _apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer):
        layer.post_loaded = True


class _Linear(torch.nn.Module):
    def __init__(self, prefix, n, k, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.prefix = prefix
        self.weight = torch.nn.Parameter((torch.randn(n, k, generator=g) * 0.02).bfloat16(),
                                         requires_grad=False)
        self.bias = None
        self.quant_method = _Unq()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


@pytest.fixture
def lane_env(monkeypatch, shipped):
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod, linear.LinearBase = _Unq, _Linear
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear)
    monkeypatch.setattr(_Unq, "process_weights_after_loading",
                        _Unq.process_weights_after_loading)  # restored after register()
    monkeypatch.setattr(lane, "_helion", {"layers": {}, "matched": 0, "captured": set(),
                                          "hook": None, "checked": False})
    monkeypatch.setattr(lane, "_state", {"armed": False, "layers": 0, "oracle": [],
                                         "launch_logged": False})
    for k in (lane.GATE, lane.HELION_ENV, lane.LAYERS_ENV, lane.MAX_M_ENV):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch
    h = lane._helion.get("hook")
    if h is not None:
        h.remove()


def _eager_rt():
    from suffix_hybrid.helion_kernels.mxfp8_gemm import EagerRuntime
    return EagerRuntime


def test_knob_off_or_triton_mode_registers_nothing(lane_env):
    post = _Unq.process_weights_after_loading
    assert lane.register() is None  # gate off
    lane_env.setenv(lane.GATE, "1")
    assert lane.register() is None  # gate on, SUFFIX_HELION_BF16_SKINNY unset = Triton lane
    assert _Unq.process_weights_after_loading is post and lane._helion["hook"] is None
    lane_env.setenv(lane.HELION_ENV, "bogus")
    with pytest.raises(RuntimeError, match="expected oxide"):
        lane.register()


def test_max_m_cap_follows_the_mode(lane_env):
    lane_env.setenv(lane.MAX_M_ENV, "160")
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        lane.max_m()
    lane_env.setenv(lane.HELION_ENV, "auto")
    assert lane.max_m() == 160
    lane_env.setenv(lane.MAX_M_ENV, "193")
    with pytest.raises(ValueError, match=r"\[1, 192\]"):
        lane.max_m()


def test_register_arms_matching_layers_through_the_post_load_hook(lane_env, cpu_mm, capsys):
    pytest.importorskip("helion")
    lane_env.setenv(lane.GATE, "1")
    lane_env.setenv(lane.HELION_ENV, "auto")
    lane_env.setenv(lane.MAX_M_ENV, "40")
    lane_env.setenv(lane.LAYERS_ENV, "*.mlp.gate,*.mlp.shared_expert_gate")
    real_install = lane.install_helion_layer  # CPU: EagerRuntime stands in for CUDA graphs
    lane_env.setattr(lane, "install_helion_layer",
                     lambda name, mod: real_install(name, mod, rt=_eager_rt()))
    assert lane.register() is lane._state
    gate = _Linear("model.layers.3.mlp.gate", 40, 320, seed=1)
    other = _Linear("model.layers.3.mlp.gate_up_proj", 40, 320, seed=2)
    big = _Linear("model.layers.3.mlp.shared_expert_gate", 1, 320, seed=3)  # nothing shipped
    for mod in (gate, other, big):
        mod.quant_method.process_weights_after_loading(mod)
        assert mod.post_loaded  # vLLM's own post-load still runs first
    assert list(lane._helion["layers"]) == ["model.layers.3.mlp.gate"]
    assert lane._helion["matched"] == 2 and other.quant_method.apply == other.quant_method._apply
    err = capsys.readouterr().err
    assert ("model.layers.3.mlp.gate N=40 K=320: M1-4:F.linear M5-8:helion(b8,split 2) "
            "M9-16:F.linear M17-32:helion(b32) M33-40:F.linear, M>40:F.linear "
            "(LAYER ORACLE PASS") in err
    assert "shared_expert_gate N=1 K=320: stays on F.linear (no shipped" in err
    # serving: the op path for M with a plan entry, the original apply elsewhere -> same math
    calls = []
    lane_env.setattr(hb, "run", lambda x, w, e, **kw: calls.append(x.shape[0]) or
                     torch.nn.functional.linear(x, w))
    for m in (1, 5, 8, 17, 40, 64):
        x = torch.randn(m, 320).bfloat16()
        assert torch.equal(gate(x), torch.nn.functional.linear(x, gate.weight))
    assert calls == [5, 8, 17]
    # a second post-load pass over the same layer does not re-arm it
    gate.quant_method.process_weights_after_loading(gate)
    assert lane._helion["matched"] == 2
    # first module forward: ACTIVE summary, hook removed
    torch.nn.Linear(2, 2)(torch.zeros(1, 2))
    assert "BF16-SKINNY ACTIVE: 1 of 2 matched layers on Helion" in capsys.readouterr().err
    assert lane._helion["hook"] is None and lane._helion["checked"]


def test_oracle_refuses_a_wrong_kernel(lane_env, cpu_mm):
    pytest.importorskip("helion")
    lane_env.setenv(lane.HELION_ENV, "auto")
    real_run = hb.run
    lane_env.setattr(hb, "run", lambda x, w, e, **kw: real_run(x, w, e, **kw) * 1.01)
    mod = _Linear("model.layers.0.mlp.gate", 40, 320)
    with pytest.raises(RuntimeError, match="HELION ORACLE FAIL model.layers.0.mlp.gate"):
        lane.install_helion_layer("model.layers.0.mlp.gate", mod, rt=_eager_rt())
    assert mod.quant_method.apply == mod.quant_method._apply  # never wrapped


def test_capture_replay_check_bites(lane_env, cpu_mm):
    pytest.importorskip("helion")
    lane_env.setenv(lane.HELION_ENV, "auto")
    state = {"n": 0}
    real_run = hb.run

    def drifting(x, w, e, **kw):  # right once (eager), wrong on "replay": a stale-buffer bug
        state["n"] += 1
        y = real_run(x, w, e, **kw)
        return y if state["n"] <= 7 else y * 0  # 3 oracle Ms, 3 eager, 1 capture
    lane_env.setattr(hb, "run", drifting)
    plan = hb.prepare(40, 320, 8, "auto")
    w = _Linear("g", 40, 320).weight.data
    with pytest.raises(RuntimeError, match="CAPTURE/REPLAY FAIL"):
        lane.helion_oracle("g", w, plan, _eager_rt())


def test_first_forward_fails_closed(lane_env):
    lane_env.setenv(lane.GATE, "1")
    lane_env.setenv(lane.HELION_ENV, "auto")
    lane_env.setenv(lane.LAYERS_ENV, "*.mlp.gate")
    lane.register()
    mod = _Linear("model.layers.0.self_attn.o_proj", 8, 32)
    mod.quant_method.process_weights_after_loading(mod)  # a model loads, nothing matches
    with pytest.raises(RuntimeError, match="no loaded layer matched"):
        torch.nn.Linear(2, 2)(torch.zeros(1, 2))


def test_first_forward_refuses_a_swapped_quant_method(lane_env):
    lane_env.setenv(lane.HELION_ENV, "auto")
    mod = _Linear("model.layers.0.mlp.gate", 40, 320)
    lane._helion["matched"] = 1
    lane._helion["layers"]["model.layers.0.mlp.gate"] = (mod, mod.quant_method, None,
                                                         mod.weight.data, [None])
    mod.quant_method = _Unq()  # a later post-load pass converted the layer
    lane._arm_first_forward_check()
    with pytest.raises(RuntimeError, match="replaced after arming"):
        torch.nn.Linear(2, 2)(torch.zeros(1, 2))


def test_op_routes_by_plan_and_has_a_fake(lane_env):
    op = lane.register_op()
    mod = _Linear("l", 4, 32)
    sentinel = torch.full((3, 4), 7.0).bfloat16()
    lane._helion["layers"]["l"] = (mod, mod.quant_method, lambda layer, x, b: sentinel,
                                   mod.weight.data, [None, None, None, None])
    x = torch.randn(3, 32).bfloat16()
    assert op(x, 4, "l") is not None and torch.equal(op(x, 4, "l"), sentinel)  # no entry
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as fm:
        y = op(fm.from_tensor(x), 4, "l")
    assert tuple(y.shape) == (3, 4) and y.dtype == torch.bfloat16
