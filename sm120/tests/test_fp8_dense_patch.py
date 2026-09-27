# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sm120/fp8_dense_patch: pinned fixture + anchor replay, drift
refusal, allowlist/deny matching, the e4m3 quantizer vs a numpy reference,
and convert_model over fake vLLM modules with CPU stand-ins for the two
kernels (asserting the cutlass operand layout)."""
import ast
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
import fp8_dense_patch as P  # noqa: E402
from fp8_dense_patch import oracle as O  # noqa: E402
from fp8_dense_patch import runtime as R  # noqa: E402
from nvfp4_kv_patch import _PostImportFinder  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures/vllm_0.30.0" / P.FIXTURE
FIX_SHA = "aa398e74ee835537403e60186f2e9a81e8741a4b35fec68b47df6537716a45b7"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.delenv(P.LAYERS_ENV, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


# --- anchor / drift ---------------------------------------------------------
def test_fixture_is_pinned():
    assert hashlib.sha256(FIX.read_bytes()).hexdigest() == FIX_SHA


def test_replay_is_exact_and_twice_is_drift():
    src = FIX.read_text()
    new, applied = P.patch_source(src)
    assert applied == ["convert_after_post_load"]
    _n, old, rep, _c = P.EDITS[0]
    assert new.replace(rep, old) == src
    with pytest.raises(P.PatchDriftError):
        P.patch_source(new)


def test_convert_runs_last_in_process_weights_after_loading():
    new, _ = P.patch_source(FIX.read_text())
    fn = next(n for n in ast.walk(ast.parse(new)) if isinstance(n, ast.FunctionDef)
              and n.name == "process_weights_after_loading")
    body = ast.unparse(fn)
    i = body.index("_suffix_fp8_dense_convert(model)")
    # after the per-layer finalize and the MLA W_UK/W_UV absorb (kv_b_proj BF16)
    assert body.index("quant_method.process_weights_after_loading(module)") < i
    assert body.index("module.process_weights_after_loading(model_config.dtype)") < i
    assert body.index("model.process_weights_after_loading()") < i


def test_drift_missing_and_duplicate_anchor():
    src = FIX.read_text()
    old = P.EDITS[0][1]
    with pytest.raises(P.PatchDriftError, match="expected 1, found 0"):
        P.patch_source(src.replace(old, old.replace("\n", " \n", 1)))
    with pytest.raises(P.PatchDriftError, match="expected 1, found 2"):
        P.patch_source(src + "\n" + old)


def _fake_target(tmp_path, src):
    f = tmp_path / "utils.py"
    f.write_text(src)
    m = types.ModuleType(P.TARGET_MODULE)
    m.__file__ = str(f)
    return m


def test_apply_refuses_drift_version_and_ondisk_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    with pytest.raises(P.PatchDriftError, match="convert_after_post_load"):
        P.apply(_fake_target(tmp_path, "def process_weights_after_loading(): pass\n"))
    with pytest.raises(P.PatchDriftError, match="already carries"):
        P.apply(_fake_target(tmp_path, P.patch_source(FIX.read_text())[0]))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.31.0"))
    with pytest.raises(P.PatchDriftError, match="pinned 0.30.0"):
        P.apply(_fake_target(tmp_path, FIX.read_text()))
    with pytest.raises(SystemExit, match="installation FAILED"):
        P._hook_callback(_fake_target(tmp_path, FIX.read_text()))


def test_gate_off_is_inert(tmp_path):
    assert P.apply(_fake_target(tmp_path, "garbage")) is False
    assert P.install_post_import_hook() is False
    assert not any(isinstance(f, _PostImportFinder) and f.target == P.TARGET_MODULE
                   for f in sys.meta_path)


def test_hook_arms_once_and_refuses_live_module(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.delitem(sys.modules, P.TARGET_MODULE, raising=False)
    assert P.install_post_import_hook() and P.install_post_import_hook()
    armed = [f for f in sys.meta_path if isinstance(f, _PostImportFinder)
             and f.target == P.TARGET_MODULE]
    assert len(armed) == 1 and sys.meta_path[0] is armed[0]
    monkeypatch.setitem(sys.modules, P.TARGET_MODULE, types.ModuleType("x"))
    with pytest.raises(SystemExit, match="imported before the hook armed"):
        P.install_post_import_hook()


# --- allowlist ----------------------------------------------------------------
GLM_NAMES = {
    "model.layers.0.self_attn.fused_qkv_a_proj": True,
    "model.layers.0.self_attn.q_b_proj": True,
    "model.layers.0.self_attn.o_proj": True,
    "model.layers.0.self_attn.kv_b_proj": False,
    "model.layers.0.self_attn.indexer.wq_b": True,
    "model.layers.0.self_attn.indexer.wk_weights_proj": False,
    "model.layers.0.mlp.gate_up_proj": True,
    "model.layers.0.mlp.down_proj": True,
    "model.layers.5.mlp.gate": False,
    "model.layers.5.mlp.shared_experts.gate_up_proj": True,
    "model.layers.5.mlp.shared_experts.down_proj": True,
    "model.layers.78.eh_proj": True,
    "model.layers.78.mtp_block.self_attn.o_proj": True,
    "lm_head": False,
    "model.layers.78.shared_head.head": False,
}


def test_default_allowlist_on_glm_names():
    assert {n: P.selected(n) for n in GLM_NAMES} == GLM_NAMES


def test_env_allowlist_narrows_and_deny_always_wins(monkeypatch):
    monkeypatch.setenv(P.LAYERS_ENV, " *.o_proj , *.eh_proj,")
    assert P.layer_patterns() == ("*.o_proj", "*.eh_proj")
    assert [n for n in GLM_NAMES if P.selected(n)] == [
        "model.layers.0.self_attn.o_proj", "model.layers.78.eh_proj",
        "model.layers.78.mtp_block.self_attn.o_proj"]
    monkeypatch.setenv(P.LAYERS_ENV, "*")
    assert not any(P.selected(n) for n, v in GLM_NAMES.items() if not v)


# --- quantizer numerics vs numpy ------------------------------------------------
def _e4m3_grid():
    """All finite non-negative float8_e4m3fn values, with their code's LSB."""
    vals = []
    for code in range(0x7F):  # 0x7F = NaN
        e, m = code >> 3, code & 7
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        vals.append((v, code & 1))
    return np.array([v for v, _ in vals]), np.array([o for _, o in vals])


def np_e4m3(x):
    """Round-to-nearest-even onto the e4m3fn grid (saturating at 448)."""
    grid, odd = _e4m3_grid()
    a = np.minimum(np.abs(x), 448.0)
    hi = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
    lo = hi - 1
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(odd[lo] == 0, lo, hi)))
    return np.sign(x) * grid[pick]


def np_quantize_weight(w):
    """Same fp32 arithmetic as the torch path; e4m3 rounding in numpy."""
    w = w.astype(np.float32)
    s = np.maximum(np.abs(w).max(axis=1), np.float32(1e-12)) / np.float32(448.0)
    return np_e4m3((w / s[:, None]).astype(np.float64)), s


def test_quantizer_matches_numpy_reference():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(64, 256, generator=g) * 0.02)
    w[3] = 0.0                    # all-zero row: finite, exact zero
    w[5, 7] = 3.0                 # outlier row
    w = w.bfloat16()
    q, s = R.quantize_weight(w)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    assert q.shape == (64, 256) and s.shape == (64,)
    nq, ns = np_quantize_weight(w.float().numpy())
    np.testing.assert_array_equal(s.numpy(), ns)
    np.testing.assert_array_equal(q.float().numpy(), nq.astype(np.float32))
    assert float(q.float().abs().amax(1)[[0, 5]].min()) == 448.0  # full range used
    deq = q.float() * s[:, None]
    assert R.rel_frob(deq, w.float()) < 0.03  # e4m3 weight-only floor


# --- convert_model over fake vLLM ------------------------------------------------
class FakeOps:
    calls = 0

    @staticmethod
    def cutlass_scaled_mm_supports_fp8(cap):
        return cap == 120

    @staticmethod
    def scaled_fp8_quant(x, use_per_token_if_dynamic=False):
        assert use_per_token_if_dynamic and x.dim() == 2 and x.stride(-1) == 1
        s = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s

    @classmethod
    def cutlass_scaled_mm(cls, a, b, scale_a, scale_b, out_dtype, bias=None):
        # the real SM120 kernel's operand contract
        m, k = a.shape
        n = b.shape[1]
        assert a.dtype == b.dtype == torch.float8_e4m3fn and b.shape[0] == k
        assert b.stride(0) == 1, "B must be column-major [K, N]"
        assert k % 16 == 0 and n % 16 == 0
        assert scale_a.numel() == m and scale_b.numel() == n and scale_b.is_contiguous()
        assert scale_a.dtype == scale_b.dtype == torch.float32
        cls.calls += 1
        out = (a.float() * scale_a.view(-1, 1)) @ (b.float() * scale_b.view(1, -1))
        return (out if bias is None else out + bias.float()).to(out_dtype)


class LinearBase(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k).bfloat16() * 0.02, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return nn.functional.linear(x, layer.weight, bias)


class LinearMethodBase:
    pass


class OtherMethod(UnquantizedLinearMethod):  # e.g. GLM52LowLatencyLinearMethod
    pass


@pytest.fixture
def fake_vllm(monkeypatch):
    lin = types.ModuleType("vllm.model_executor.layers.linear")
    lin.LinearBase, lin.UnquantizedLinearMethod = LinearBase, UnquantizedLinearMethod
    lin.LinearMethodBase = LinearMethodBase
    monkeypatch.setitem(sys.modules, lin.__name__, lin)
    monkeypatch.setattr(R, "_OPS", FakeOps)
    monkeypatch.setattr(R, "_METHOD_CLS", None)
    monkeypatch.setattr(R, "_SELFTESTED", set())
    monkeypatch.setattr(R, "_capability", lambda dev: (12, 0))


def _glm_like():
    torch.manual_seed(0)
    root = nn.Module()
    root.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.fused_qkv_a_proj = LinearBase(64, 96)
    layer.self_attn.o_proj = LinearBase(96, 32)
    layer.self_attn.kv_b_proj = LinearBase(64, 32)          # denied
    layer.self_attn.odd = LinearBase(40, 32)                # not allowlisted
    layer.mlp = nn.Module()
    layer.mlp.gate = LinearBase(16, 96)                     # denied (router)
    layer.mlp.shared_experts = nn.Module()
    layer.mlp.shared_experts.down_proj = LinearBase(96, 24)  # K % 16 -> skipped
    layer.mlp.shared_experts.gate_up_proj = LinearBase(32, 96)
    layer.mlp.shared_experts.gate_up_proj.quant_method = OtherMethod()  # not stock
    root.model.layers = nn.ModuleDict({"0": layer})
    mtp = nn.Module()
    mtp.eh_proj = nn.Linear(64, 96, bias=False).bfloat16()
    root.model.layers["78"] = mtp
    return root


def test_convert_model_converts_exactly_the_allowlisted_stock_linears(fake_vllm, capsys):
    root = _glm_like()
    mods = dict(root.named_modules())
    x = {n: torch.randn(6, m.weight.shape[1]).bfloat16() for n, m in mods.items()
         if hasattr(m, "weight")}
    ref = {n: nn.functional.linear(x[n], m.weight) for n, m in mods.items() if n in x}
    done = R.convert_model(root)
    assert [d[0] for d in done] == ["model.layers.0.self_attn.fused_qkv_a_proj",
                                    "model.layers.0.self_attn.o_proj",
                                    "model.layers.78.eh_proj"]
    assert "[suffix fp8-dense] converted 3 linears" in capsys.readouterr().err
    for name, _n, _k in done:
        m = mods[name]
        assert m.weight.dtype == torch.float8_e4m3fn and m.weight_scale.shape == (_n,)
        out = m(x[name])
        assert out.shape == ref[name].shape and out.dtype == torch.bfloat16
        assert R.rel_frob(out, ref[name]) < R.QUANT_BOUND
        assert R.rel_frob(out, R.reference(x[name], m.weight, m.weight_scale)) < R.KERNEL_BOUND
    assert type(mods["model.layers.78.eh_proj"]) is R.Fp8DenseNNLinear
    assert isinstance(mods["model.layers.0.self_attn.o_proj"].quant_method, LinearMethodBase)
    for kept in ("kv_b_proj", "odd"):
        assert mods[f"model.layers.0.self_attn.{kept}"].weight.dtype == torch.bfloat16
    assert mods["model.layers.0.mlp.gate"].weight.dtype == torch.bfloat16
    assert mods["model.layers.0.mlp.shared_experts.down_proj"].weight.dtype == torch.bfloat16
    assert mods["model.layers.0.mlp.shared_experts.gate_up_proj"].weight.dtype == torch.bfloat16


def test_convert_model_3d_input_and_selftest_once_per_shape(fake_vllm):
    root = nn.Module()
    root.a, root.b = nn.Module(), nn.Module()
    root.a.self_attn, root.b.self_attn = nn.Module(), nn.Module()
    root.a.self_attn.o_proj = LinearBase(32, 64)
    root.b.self_attn.o_proj = LinearBase(32, 64)
    FakeOps.calls = 0
    R.convert_model(root)
    assert FakeOps.calls == 1  # one self-test for the shared (32, 64) shape
    y = root.a.self_attn.o_proj(torch.randn(2, 3, 64).bfloat16())
    assert y.shape == (2, 3, 32)


def test_convert_model_inert_off_sm120(fake_vllm, monkeypatch, capsys):
    monkeypatch.setattr(R, "_capability", lambda dev: (9, 0))
    root = _glm_like()
    assert R.convert_model(root) == []
    assert "present but inert" in capsys.readouterr().err
    assert root.model.layers["0"].self_attn.o_proj.weight.dtype == torch.bfloat16


def test_selftest_fails_closed_on_a_broken_kernel(fake_vllm, monkeypatch):
    class Broken(FakeOps):
        @classmethod
        def cutlass_scaled_mm(cls, *a, **k):
            return super().cutlass_scaled_mm(*a, **k) * 1.1

    monkeypatch.setattr(R, "_OPS", Broken)
    with pytest.raises(RuntimeError, match="self-test FAILED"):
        R.convert_model(_glm_like())


def test_oracle_inventory_shapes_are_cutlass_aligned():
    assert all(n % 16 == 0 and k % 16 == 0 for _l, n, k, _c in O.INVENTORY)
    names = {row[0] for row in O.INVENTORY}
    assert {"fused_qkv_a_proj", "o_proj", "mtp.eh_proj"} <= names
