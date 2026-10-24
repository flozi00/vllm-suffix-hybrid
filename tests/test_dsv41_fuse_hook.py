# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid/kernels/dsv41_fuse_hook.py: gates, anchors, routing, K5 accounting, and the
hooks end to end on CPU (the real Helion bodies through the ref-mode stand-in)."""
import hashlib
import os
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_common as cm  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_hc as hc  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_router as rt  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_swiglu as sw  # noqa: E402
from suffix_hybrid.kernels import dsv41_fuse_hook as fh  # noqa: E402
from test_dsv41_stub import run_body  # noqa: E402

ENVS = ("SUFFIX_DSV41_FUSE", fh.QUANT_ENV, fh.HC_ENV, fh.ROUTER_ENV, fh.SWIGLU_ENV, hc.BOUNDARY_ENV)
H = 96


class QA:
    """stand-in for vllm QuantizedActivation"""

    def __init__(self, data, scale, orig_dtype, orig_shape, quant_key):
        self.data, self.scale, self.orig_dtype, self.orig_shape, self.quant_key = (
            data, scale, orig_dtype, orig_shape, quant_key)


@pytest.fixture
def env(monkeypatch):
    for k in ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(fh, "_STATE", fh._fresh_state())
    fh._STATE["checked_sm"] = True
    monkeypatch.setattr(fh.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(fh, "_vllm_qa", lambda: (QA, "KEY_MXFP8"))
    monkeypatch.setattr(fh, "_can_take_mxfp8", lambda lin: getattr(lin, "mxfp8", True))
    monkeypatch.setattr(hc, "_WARM", set())
    monkeypatch.setattr(rt, "_WARM", set())
    monkeypatch.setattr(sw, "_WARM", set())
    return monkeypatch


def ship(monkeypatch, buckets=None):
    """configs shipped for `buckets` (None: all); every Helion kernel runs through the ref stand-in."""
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: (
        {"num_warps": 4} if buckets is None or case["numtokens"] in buckets else None))
    monkeypatch.setattr(hk, "platform", lambda device=0: "testgpu")
    blocks = {hc.K_POST: {0: 2, 1: 32, 2: 4}, hc.K_EPI: {0: 2, 1: 2}, hc.K_RES: {0: 2, 1: 32, 2: 2},
              rt.K_GEMV: {0: 16, 1: 32, 2: 64}, sw.KERNEL: {0: 2, 1: 4}}

    def compiled(body, kernel, case_):
        return lambda *a: run_body(monkeypatch, sys.modules[body.__module__], body, a, blocks[kernel])
    monkeypatch.setattr(hk, "compiled", compiled)

    orig = torch.mm      # ref-mode hl.dot calls torch.mm(out_dtype=f32), which CPU torch lacks
    monkeypatch.setattr(torch, "mm", lambda a, b, out_dtype=None: orig(
        *(t.to(out_dtype) for t in (a, b))) if out_dtype is not None and a.dtype != out_dtype else orig(a, b))


def target(monkeypatch, tmp_path, name, text):
    f = tmp_path / (name.rsplit(".", 1)[-1] + ".py")
    f.write_text(text)
    monkeypatch.setitem(fh.SHA256, name, hashlib.sha256(text.encode()).hexdigest())
    return types.SimpleNamespace(__file__=str(f), __name__=name)


HC_SRC = "def mhc_shifted_post_pre(...):\n    return mhc_fused_post_pre_delayed_tilelang(x)\n"


def hc_module(monkeypatch, tmp_path, stock=None):
    mod = target(monkeypatch, tmp_path, fh.HC_TARGET, HC_SRC)
    mod.mhc_fused_post_pre_delayed_tilelang = stock or (lambda *a, **k: ("stock", a[0].shape[0]))
    return mod


# ---------------------------------------------------------------------------
# arming
# ---------------------------------------------------------------------------
def test_gate_off_and_oxide_are_inert(env):
    assert fh.install_post_import_hook() is False
    env.setenv("SUFFIX_DSV41_FUSE", "1")
    assert fh.install_post_import_hook() is False                  # every knob defaults to oxide
    env.setenv(fh.HC_ENV, "bogus")
    with pytest.raises(SystemExit, match="expected oxide"):
        fh.install_post_import_hook()
    env.setenv(fh.HC_ENV, "oxide")
    env.setenv(fh.QUANT_ENV, "1")
    with pytest.raises(SystemExit, match="needs SUFFIX_HELION_DSV41_HC"):
        fh.install_post_import_hook()


def test_plan_targets_follow_the_knobs(env):
    assert fh.plan_from_env({})["targets"] == []
    p = fh.plan_from_env({fh.HC_ENV: "auto"})
    assert p["targets"] == [fh.HC_TARGET] and not p["quant"]
    p = fh.plan_from_env({fh.HC_ENV: "helion", fh.QUANT_ENV: "1", fh.ROUTER_ENV: "auto"})
    assert p["targets"] == [fh.HC_TARGET, fh.ATTN_TARGET, fh.MLP_TARGET, fh.GATE_TARGET] and p["quant"]
    p = fh.plan_from_env({fh.SWIGLU_ENV: "auto"})
    assert p["targets"] == [fh.MLP_TARGET]


def test_armed_inserts_finders_and_refuses_early_import(env):
    finders = []
    fake = types.ModuleType("nvfp4_kv_patch")
    fake._PostImportFinder = lambda t, cb: finders.append((t, cb)) or types.SimpleNamespace()
    env.setitem(sys.modules, "nvfp4_kv_patch", fake)
    env.setattr(sys, "meta_path", list(sys.meta_path))
    env.setenv("SUFFIX_DSV41_FUSE", "1")
    env.setenv(fh.HC_ENV, "auto")
    env.setenv(fh.ROUTER_ENV, "auto")
    assert fh.install_post_import_hook() is True
    assert [t for t, _ in finders] == [fh.HC_TARGET, fh.GATE_TARGET]
    env.setitem(sys.modules, fh.GATE_TARGET, types.ModuleType(fh.GATE_TARGET))
    with pytest.raises(SystemExit, match="imported before the hook armed"):
        fh.install_post_import_hook()


def test_drifted_source_fails_closed_for_every_target(env, tmp_path):
    for name, apply in ((fh.HC_TARGET, lambda m: fh.apply_hc(m, "auto")), (fh.ATTN_TARGET, fh.apply_attn),
                        (fh.MLP_TARGET, lambda m: fh.apply_mlp(m, "auto", False)),
                        (fh.GATE_TARGET, lambda m: fh.apply_gate(m, "auto"))):
        mod = types.SimpleNamespace(__file__=str(tmp_path / "x.py"), __name__=name)
        (tmp_path / "x.py").write_text("x = 1\n")
        with pytest.raises(RuntimeError, match="re-check the hooked call sites"):
            apply(mod)
    mod = target(env, tmp_path, fh.HC_TARGET, "x = 1\n")
    mod.mhc_fused_post_pre_delayed_tilelang = lambda *a: None
    with pytest.raises(RuntimeError, match="call site is gone"):
        fh.apply_hc(mod, "auto")
    plan = fh.plan_from_env({fh.HC_ENV: "auto"})
    with pytest.raises(SystemExit, match="FAILED"):
        fh._callback_for(fh.HC_TARGET, plan)(types.SimpleNamespace(
            __file__=str(tmp_path / "x.py"), __name__=fh.HC_TARGET))


# ---------------------------------------------------------------------------
# hc
# ---------------------------------------------------------------------------
def call_hc(wrapper, t, aux=False, seed=1, h=H, **over):
    P = hc.make_problem(t, h, seed, "cpu")
    args = list(hc.stock_call_args(P, aux))
    for k, v in over.items():
        args[int(k[1:])] = v
    return P, wrapper(*args[:12], pre_mix=args[12], norm_weight=args[13], norm_eps=args[14],
                      capture_aux=args[15])


def test_hc_routes_by_mode_bucket_and_tokens(env, tmp_path):
    ship(env, buckets=(6, 48))
    mod = hc_module(env, tmp_path)
    fh.apply_hc(mod, "auto")
    w = mod.mhc_fused_post_pre_delayed_tilelang
    P, out = call_hc(w, 6)
    assert len(out) == 6 and out[0].shape == (6, 4, H)                  # served by Helion (bucket 6)
    _, out = call_hc(w, 30)                                              # bucket 48 shipped
    assert out[0].shape == (30, 4, H)
    assert call_hc(w, 12)[1] == ("stock", 12)                            # bucket 12 not shipped (auto)
    assert call_hc(w, 500)[1] == ("stock", 500)                          # prefill chunk
    assert fh._STATE["hc"]["served"] == {6: 1, 48: 1}
    assert fh._STATE["hc"]["stock"] == {"no-config": 1, "tokens>192": 1}
    assert "hc[auto] helion by bucket" in fh.summary()


def test_hc_helion_mode_missing_config_is_an_error(env, tmp_path):
    ship(env, buckets=(6,))
    mod = hc_module(env, tmp_path)
    fh.apply_hc(mod, "helion")
    w = mod.mhc_fused_post_pre_delayed_tilelang
    assert len(call_hc(w, 5)[1]) == 6
    with pytest.raises(RuntimeError, match="no tuned dsv41_hc_post_partials/dsv41_hc_epilogue config"):
        call_hc(w, 12)
    assert call_hc(w, 193)[1] == ("stock", 193)                         # beyond the buckets: stock, counted


def test_hc_output_matches_the_reference_both_boundaries(env, tmp_path):
    ship(env)
    for boundary in ("split2", "resident"):
        env.setenv(hc.BOUNDARY_ENV, boundary)
        mod = hc_module(env, tmp_path)
        fh.apply_hc(mod, "helion")
        P, out = call_hc(mod.mhc_fused_post_pre_delayed_tilelang, 6, aux=True)
        tail = (torch.empty(0, H, dtype=torch.float8_e4m3fn), torch.empty(0, dtype=torch.uint8))
        ok, msg = hc.judge_boundary((out[0], out[1].squeeze(-1), *out[2:6], *tail), hc.k0_args(P, True, False))
        assert ok, msg


def test_hc_unexpected_calls_raise_instead_of_falling_back(env, tmp_path):
    ship(env)
    mod = hc_module(env, tmp_path)
    fh.apply_hc(mod, "helion")
    w = mod.mhc_fused_post_pre_delayed_tilelang
    with pytest.raises(RuntimeError, match="unexpected call"):
        call_hc(w, 6, a13=None)                                          # norm_weight
    P = hc.make_problem(6, H, 1, "cpu")
    args = list(hc.stock_call_args(P))
    args[0] = args[0].float()
    with pytest.raises(RuntimeError, match="unexpected call"):
        w(*args[:12], pre_mix=args[12], norm_weight=args[13], norm_eps=args[14], capture_aux=False)
    args = list(hc.stock_call_args(P))
    args[1] = torch.zeros(6, 2, H, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="unexpected call"):
        w(*args[:12], pre_mix=args[12], norm_weight=args[13], norm_eps=args[14], capture_aux=False)


def test_sm120_check_refuses_other_devices(env):
    fh._STATE["checked_sm"] = False
    env.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    with pytest.raises(RuntimeError, match="SM120-only"):
        fh._check_sm120()
    env.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    fh._check_sm120()
    assert fh._STATE["checked_sm"] is True


# ---------------------------------------------------------------------------
# quant: HC emits the MXFP8 activation, attention + shared experts consume it
# ---------------------------------------------------------------------------
class FakeLinear:
    def __init__(self, mxfp8=True):
        self.mxfp8, self.seen = mxfp8, []

    def __call__(self, x):
        self.seen.append(x)
        if isinstance(x, QA):
            return torch.zeros(x.orig_shape[0], 7), None
        return torch.zeros(x.shape[0], 7), None


def attn_module(monkeypatch, tmp_path):
    mod = target(monkeypatch, tmp_path, fh.ATTN_TARGET, "class DeepseekV4Attention: ...\n")

    class DeepseekV4Attention:
        shard_fused_wqa_wkv = False

        def __init__(self, mxfp8=True):
            self.fused_wqa_wkv = FakeLinear(mxfp8)
            self.stock_calls = 0

        def _fused_wqa_wkv_gemm(self, hidden_states):
            self.stock_calls += 1
            return torch.full((hidden_states.shape[0], 7), -1.0)
    mod.DeepseekV4Attention = DeepseekV4Attention
    return mod


def test_quant_attaches_and_attention_consumes(env, tmp_path):
    ship(env)
    env.setenv(hc.BOUNDARY_ENV, "split2")
    fh._STATE["quant"] = True
    hmod = hc_module(env, tmp_path)
    fh.apply_hc(hmod, "helion")
    amod = attn_module(env, tmp_path)
    fh.apply_attn(amod)
    P, out = call_hc(hmod.mhc_fused_post_pre_delayed_tilelang, 6, h=H)
    assert len(out) == 6
    qa = getattr(out[3], fh.ATTR)
    assert isinstance(qa, QA) and qa.quant_key == "KEY_MXFP8" and qa.orig_shape == out[3].shape
    q_ref, qs_ref = cm.mxfp8_quant_reference(out[3])
    assert torch.equal(qa.data.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(qa.scale, qs_ref)
    layer = amod.DeepseekV4Attention()
    got = layer._fused_wqa_wkv_gemm(out[3])
    assert layer.stock_calls == 0 and layer.fused_wqa_wkv.seen == [qa] and got.shape == (6, 7)
    # a plain tensor (layer 0 / Engram layers / > 192 tokens) takes the stock path
    layer._fused_wqa_wkv_gemm(torch.zeros(6, H, dtype=torch.bfloat16))
    assert layer.stock_calls == 1
    assert fh._STATE["q"]["attn_hit"] == 1 and fh._STATE["q"]["attn_miss"] == 1


def test_quant_that_never_arrives_is_an_error(env, tmp_path):
    amod = attn_module(env, tmp_path)
    fh.apply_attn(amod)
    layer = amod.DeepseekV4Attention()
    fh._STATE["q"]["emitted"] = fh.FIRST_QUANT_CHECK
    with pytest.raises(RuntimeError, match="lost its attribute"):
        layer._fused_wqa_wkv_gemm(torch.zeros(6, H, dtype=torch.bfloat16))


def test_attention_refuses_a_linear_that_cannot_take_mxfp8(env, tmp_path):
    amod = attn_module(env, tmp_path)
    fh.apply_attn(amod)
    layer = amod.DeepseekV4Attention(mxfp8=False)
    x = torch.zeros(6, H, dtype=torch.bfloat16)
    setattr(x, fh.ATTR, QA(None, None, torch.bfloat16, x.shape, "KEY_MXFP8"))
    with pytest.raises(RuntimeError, match="cannot take an MXFP8 activation"):
        layer._fused_wqa_wkv_gemm(x)


def test_attention_token_sharded_path_stays_stock(env, tmp_path):
    amod = attn_module(env, tmp_path)
    fh.apply_attn(amod)
    layer = amod.DeepseekV4Attention()
    layer.shard_fused_wqa_wkv = True
    x = torch.zeros(fh.ATTN_MIN_SHARD_TOKENS, 8, dtype=torch.bfloat16)
    setattr(x, fh.ATTR, QA(None, None, torch.bfloat16, x.shape, "KEY_MXFP8"))
    layer._fused_wqa_wkv_gemm(x)
    assert layer.stock_calls == 1 and not layer.fused_wqa_wkv.seen


# ---------------------------------------------------------------------------
# mlp: shared-expert prequant + SwiGLU + quant
# ---------------------------------------------------------------------------
class FakeAct:
    swiglu_limit, alpha, beta = 10.0, 1.0, 0.0

    def __call__(self, gate_up):
        return sw.swiglu_reference(gate_up)


FakeAct.__name__ = "SiluAndMulWithClamp"


def mlp_module(monkeypatch, tmp_path, d=64):
    mod = target(monkeypatch, tmp_path, fh.MLP_TARGET, "class DeepseekV4MLP: ...\n")

    class Gate:
        mxfp8 = True

        def __init__(self):
            self.seen = []

        def __call__(self, x):
            self.seen.append(x)
            return sw.make_problem(x.orig_shape[0] if isinstance(x, QA) else x.shape[0], d, 3), None

    class Down(FakeLinear):
        pass

    class DeepseekV4MLP:
        def __init__(self):
            self.gate_up_proj, self.down_proj, self.act_fn = Gate(), Down(), FakeAct()

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            out, _ = self.down_proj(self.act_fn(gate_up))
            return out
    mod.DeepseekV4MLP = DeepseekV4MLP
    return mod


def test_mlp_swiglu_quant_feeds_down_proj(env, tmp_path):
    ship(env)
    mod = mlp_module(env, tmp_path)
    fh.apply_mlp(mod, "helion", quant_on=False)
    mlp = mod.DeepseekV4MLP()
    x = torch.zeros(6, H, dtype=torch.bfloat16)
    out = mlp.forward(x)
    (qa,) = mlp.down_proj.seen
    assert isinstance(qa, QA) and qa.quant_key == "KEY_MXFP8" and tuple(qa.orig_shape) == (6, 64)
    gu = sw.make_problem(6, 64, 3)
    q_ref, qs_ref = sw.reference(gu)
    assert torch.equal(qa.data.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(qa.scale, qs_ref)
    assert out.shape == (6, 7) and fh._STATE["swiglu"]["served"] == {6: 1}


def test_mlp_stock_paths_and_refusals(env, tmp_path):
    ship(env, buckets=(6,))
    mod = mlp_module(env, tmp_path)
    fh.apply_mlp(mod, "auto", quant_on=False)
    mlp = mod.DeepseekV4MLP()
    mlp.forward(torch.zeros(12, H, dtype=torch.bfloat16))               # bucket 12 not shipped
    mlp.forward(torch.zeros(300, H, dtype=torch.bfloat16))              # prefill
    assert fh._STATE["swiglu"]["stock"] == {"no-config": 1, "tokens>192": 1}
    assert not any(isinstance(s, QA) for s in mlp.down_proj.seen)       # the stock act output went in
    mlp.act_fn = types.SimpleNamespace(swiglu_limit=10.0)               # not SiluAndMulWithClamp
    with pytest.raises(RuntimeError, match="unexpected shared-expert activation"):
        mlp.forward(torch.zeros(6, H, dtype=torch.bfloat16))
    mlp = mod.DeepseekV4MLP()
    mlp.down_proj.mxfp8 = False
    with pytest.raises(RuntimeError, match="down_proj cannot take an MXFP8 activation"):
        mlp.forward(torch.zeros(6, H, dtype=torch.bfloat16))


def test_mlp_ffn_prequant_reaches_gate_up(env, tmp_path):
    ship(env)
    mod = mlp_module(env, tmp_path)
    fh.apply_mlp(mod, "oxide", quant_on=True)
    mlp = mod.DeepseekV4MLP()
    x = torch.zeros(6, H, dtype=torch.bfloat16)
    qa = QA(None, None, torch.bfloat16, x.shape, "KEY_MXFP8")
    setattr(x, fh.ATTR, qa)
    mlp.forward(x)
    assert mlp.gate_up_proj.seen == [qa] and fh._STATE["q"]["ffn_hit"] == 1
    mlp.forward(torch.zeros(6, H, dtype=torch.bfloat16))
    assert fh._STATE["q"]["ffn_miss"] == 1 and not isinstance(mlp.gate_up_proj.seen[-1], QA)
    bad = mod.DeepseekV4MLP()
    bad.gate_up_proj.mxfp8 = False
    with pytest.raises(RuntimeError, match="gate_up cannot take an MXFP8 activation"):
        bad.forward(x)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def gate_module(monkeypatch, tmp_path, shape=(384, 5120)):
    mod = target(monkeypatch, tmp_path, fh.GATE_TARGET, "class GateLinear: ...\n")

    class GateLinear:
        def __init__(self, w, out_dtype=torch.float32):
            self.weight, self.out_dtype = w, out_dtype

        def _return(self, out):
            return out, None

        def forward(self, x):
            return ("stock", x.shape[0]), None
    mod.GateLinear = GateLinear
    return mod


def test_gate_gemv_replaces_only_the_v41_router(env, tmp_path):
    ship(env, buckets=(6,))
    mod = gate_module(env, tmp_path)
    fh.apply_gate(mod, "auto")
    x, w, *_ = rt.make_problem(6, 2, "cpu")
    g = mod.GateLinear(w)
    out, bias = g.forward(x)
    assert bias is None and rt.judge_logits(out, x, w)[0]
    assert fh._STATE["router"]["served"] == {6: 1}
    assert g.forward(rt.make_problem(12, 2, "cpu")[0])[0] == ("stock", 12)         # not shipped
    assert g.forward(torch.zeros(300, 5120, dtype=torch.bfloat16))[0] == ("stock", 300)
    other = mod.GateLinear(torch.zeros(128, 5120, dtype=torch.bfloat16))           # the DSpark gate
    assert other.forward(x)[0] == ("stock", 6)
    with pytest.raises(RuntimeError, match="unexpected gate call"):
        mod.GateLinear(w, out_dtype=torch.bfloat16).forward(x)
    with pytest.raises(RuntimeError, match="unexpected gate call"):
        g.forward(x.float())


def test_gate_helion_mode_missing_config_is_an_error(env, tmp_path):
    ship(env, buckets=(6,))
    mod = gate_module(env, tmp_path)
    fh.apply_gate(mod, "helion")
    x, w, *_ = rt.make_problem(12, 2, "cpu")
    with pytest.raises(RuntimeError, match="no tuned dsv41_gate_gemv config"):
        mod.GateLinear(w).forward(x)


# ---------------------------------------------------------------------------
def test_oxide_default_never_imports_helion_modules():
    code = ("import sys\nfrom suffix_hybrid.kernels import dsv41_fuse_hook as h\n"
            "assert h.install_post_import_hook() is False\n"
            "assert 'helion' not in sys.modules\n"
            "assert not any(m.startswith('suffix_hybrid.helion_kernels.dsv41') for m in sys.modules)\n")
    envv = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    envv["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))
    subprocess.run([sys.executable, "-c", code], check=True, env=envv)
