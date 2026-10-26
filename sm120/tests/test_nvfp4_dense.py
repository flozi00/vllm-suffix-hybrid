# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the NVFP4 mode of sm120/fp8_dense_patch: e2m1 packing and
block-scale math vs an independent numpy reference, the checkpoint layout
handed to (fake) vLLM ModelOptLinearMethod, Gemma 4 allowlist / deny, the
proven activation bounds, gate / dispatch / drift, fail-closed self-test."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
sys.path.insert(0, str(REPO))
import fp8_dense_patch as P  # noqa: E402
from fp8_dense_patch import nvfp4 as N  # noqa: E402
from fp8_dense_patch import nvfp4_oracle as O  # noqa: E402
from fp8_dense_patch import runtime as R  # noqa: E402
from nvfp4_kv_patch import _PostImportFinder  # noqa: E402
from suffix_hybrid.kernels import nvfp4_gemm as ng  # noqa: E402
from suffix_hybrid.tools import quantize_nvfp4 as QT  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures/vllm_0.30.0" / P.FIXTURE


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (P.GATE_ENV, P.LAYERS_ENV, P.NVFP4_GATE_ENV, P.NVFP4_LAYERS_ENV, N.ACT_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


# --- independent numpy reference ---------------------------------------------
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])


def np_e2m1_code(v):
    """RNE onto the e2m1 grid (ties to the even code), saturating; sign bit 3."""
    a = np.minimum(np.abs(v), 6.0)
    d = np.abs(a[..., None] - E2M1)
    lo = d.argmin(-1)                      # first minimum = lower neighbour
    s = np.sort(d, -1)
    tie = s[..., 0] == s[..., 1]
    code = np.where(tie & (lo % 2 == 1), lo + 1, lo).astype(np.uint8)
    return code | np.where(v < 0, 8, 0).astype(np.uint8)


def np_e4m3(x):
    """RNE onto the float8_e4m3fn grid, saturating at 448 (x >= 0)."""
    grid = np.array([(m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
                     for e in range(16) for m in range(8)][:127])
    a = np.minimum(x, 448.0)
    hi = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
    lo = hi - 1
    dlo, dhi = a - grid[lo], grid[hi] - a
    return grid[np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))]


def np_nvfp4(w):
    """bf16 [N, K] -> (packed uint8 [N, K/2] low nibble = even k, e4m3 block
    scale values [N, K/16], weight_scale_2 = amax / 2688)."""
    w = w.astype(np.float32)
    n, k = w.shape
    amax = float(np.abs(w).max())
    g = float(np.float32(N.FP4_RANGE / amax))  # vLLM passes an fp32 global
    blk = w.reshape(n, k // 16, 16)
    sf = np_e4m3((np.abs(blk).max(-1) * np.float32(g / 6.0)).astype(np.float32).astype(np.float64))
    sf32 = sf.astype(np.float32)
    inv = np.where(sf32 == 0, np.float32(0), np.float32(g) / np.where(sf32 == 0, 1, sf32))
    code = np_e2m1_code((blk * inv[..., None]).astype(np.float32)).reshape(n, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), sf32, amax / N.FP4_RANGE


def assert_codes_match(q, rq):
    """Packed e2m1 codes equal except exact-midpoint ties, which depend on how
    the implementation rounds g / scale (torch: reciprocal * g; the CUDA
    kernel: rcp.approx): those may land on the neighbouring code only."""
    q = np.asarray(q)
    for a, b in ((q & 0xF, rq & 0xF), (q >> 4, rq >> 4)):
        bad = a != b
        assert bad.mean() < 5e-3, bad.mean()
        assert np.all((a[bad] & 8) == (b[bad] & 8))                     # same sign
        assert np.all(np.abs((a[bad] & 7).astype(int) - (b[bad] & 7)) == 1)  # neighbour


def np_dequant(packed, sf, ws2):
    lo, hi = packed & 0xF, packed >> 4
    code = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    val = E2M1[code & 7] * np.where(code & 8, -1.0, 1.0)
    n, k = val.shape
    return (val.reshape(n, k // 16, 16) * sf[..., None]).reshape(n, k) * ws2


# --- fake vLLM -------------------------------------------------------------------
class FakeOps:
    @staticmethod
    def scaled_fp4_quant(w, g, is_sf_swizzled_layout=True):
        """vLLM contract: bf16 [R, K], scalar fp32 global -> uint8 [R, K/2],
        e4m3 [R, K/16] (linear layout when not swizzled). Math: the repo's
        torch quantizer (the self-test's reference)."""
        assert not is_sf_swizzled_layout and w.dtype == torch.bfloat16
        assert g.dtype == torch.float32 and g.numel() == 1 and w.is_contiguous()
        q, sf, _ = QT.quantize_weight(w, N.FP4_RANGE / float(g))
        return q, sf


class LinearBase(nn.Module):
    def __init__(self, n, k, partitions=None, scale=0.02):
        super().__init__()
        self.weight = nn.Parameter((torch.randn(n, k) * scale).bfloat16(), requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()
        self.input_size = self.input_size_per_partition = k
        self.output_size = n
        self.output_partition_sizes = partitions or [n]
        self.params_dtype = torch.bfloat16
        self.weight_loader_v2 = lambda *a, **kw: None

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return nn.functional.linear(x, layer.weight, bias)


class FakeModelOptNvFp4Method:
    """KNvfp4Static + KNvfp4Dynamic registration / process + a kernel that
    swizzles the block scales (vLLM swizzle_blockscale layout) and computes the
    exact W4A4 result from the swizzled tensors."""
    built = []

    def __init__(self, cfg, algo, prefix):
        assert algo == "NVFP4" and cfg.group_size == 16
        self.prefix = prefix
        FakeModelOptNvFp4Method.built.append(self)

    def create_weights(self, layer, isp, ops, input_size, output_size, params_dtype,
                       weight_loader=None):
        assert weight_loader is not None and params_dtype == torch.bfloat16
        n, parts = sum(ops), len(ops)
        P_ = lambda t: nn.Parameter(t, requires_grad=False)  # noqa: E731
        layer.register_parameter("weight", P_(torch.empty(n, isp // 2, dtype=torch.uint8)))
        layer.register_parameter("weight_scale", P_(
            torch.full((n, isp // 16), float("nan")).to(torch.float8_e4m3fn)))
        layer.register_parameter("weight_scale_2", P_(torch.empty(parts)))
        layer.register_parameter("input_scale", P_(torch.empty(parts)))

    def process_weights_after_loading(self, layer):
        assert not torch.isnan(layer.weight_scale.float()).any(), "scale never loaded"
        self.ckpt = {k: getattr(layer, k).data.clone()
                     for k in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
        layer.weight_global_scale = layer.weight_scale_2.max()
        layer.input_global_scale = layer.input_scale.max()
        layer.input_global_scale_inv = 1.0 / layer.input_global_scale
        layer.alpha = layer.input_global_scale * layer.weight_global_scale
        del layer.weight_scale_2, layer.input_scale
        n, kb = layer.weight_scale.shape
        self.kb = kb
        swz = ng.swizzle_sf(layer.weight_scale.data.view(torch.uint8).numpy())
        layer.weight_scale = nn.Parameter(torch.from_numpy(swz).view(torch.float8_e4m3fn),
                                          requires_grad=False)

    def apply(self, layer, x, bias=None):
        n = layer.weight.shape[0]
        kb_pad = -(-self.kb // 4) * 4
        rows, kbs = np.meshgrid(np.arange(n), np.arange(self.kb), indexing="ij")
        sf_bits = layer.weight_scale.data.view(torch.uint8).numpy()[
            ng.sf_offset(rows, kbs, kb_pad)]
        sf = torch.from_numpy(sf_bits).view(torch.float8_e4m3fn)
        x2 = x.reshape(-1, x.shape[-1]).float()
        xq, xsf, xs2 = QT.quantize_weight(x2, float(layer.input_global_scale) * N.FP4_RANGE)
        wd = QT.dequantize_weight(layer.weight.data, sf, float(layer.weight_global_scale))
        out = QT.dequantize_weight(xq, xsf, float(xs2)) @ wd.t()
        out = out if bias is None else out + bias.float()
        return out.to(x.dtype).view(*x.shape[:-1], n)


class RMSNorm(nn.Module):
    def __init__(self, d, has_weight=True, outlier=None):
        super().__init__()
        self.has_weight = has_weight
        if has_weight:
            w = 1 + 0.1 * torch.randn(d)
            if outlier:
                w[3] = outlier
            self.weight = nn.Parameter(w.bfloat16(), requires_grad=False)


@pytest.fixture
def fake_vllm(monkeypatch):
    lin = types.ModuleType("vllm.model_executor.layers.linear")
    lin.LinearBase, lin.UnquantizedLinearMethod = LinearBase, UnquantizedLinearMethod
    monkeypatch.setitem(sys.modules, lin.__name__, lin)
    monkeypatch.setattr(N, "_OPS", FakeOps)
    monkeypatch.setattr(N, "_BUILD", FakeModelOptNvFp4Method)
    monkeypatch.setattr(N, "_SELFTESTED", set())
    monkeypatch.setattr(R, "_capability", lambda dev: (12, 0))
    FakeModelOptNvFp4Method.built = []


H, HEADS, HD, KV, INTER = 64, 4, 16, 2, 48


def _decoder(kv_shared=False):
    layer = nn.Module()
    layer.input_layernorm = RMSNorm(H, outlier=7.0)
    layer.pre_feedforward_layernorm = RMSNorm(H)
    attn = layer.self_attn = nn.Module()
    attn.head_dim = HD
    if kv_shared:
        attn.q_proj = LinearBase(HEADS * HD, H)
    else:
        attn.qkv_proj = LinearBase((HEADS + 2 * KV) * HD, H, [HEADS * HD, KV * HD, KV * HD])
        attn.v_norm = RMSNorm(HD, has_weight=False)
    attn.o_proj = LinearBase(H, HEADS * HD)
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = LinearBase(2 * INTER, H, [INTER, INTER])
    layer.mlp.down_proj = LinearBase(H, INTER)
    layer.router = nn.Module()
    layer.router.proj = LinearBase(16, H)
    return layer


def _gemma_like():
    torch.manual_seed(0)
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = nn.ModuleDict({"0": _decoder(), "1": _decoder(True)})
    root.vision_tower = nn.Module()
    root.vision_tower.layers = nn.ModuleDict({"0": _decoder()})
    root.embed_vision = nn.Module()
    root.embed_vision.embedding_projection = LinearBase(H, 32)
    return root


L0 = "language_model.model.layers.0"
L1 = "language_model.model.layers.1"


# --- quantizer numerics vs numpy ------------------------------------------------
def test_weight_quant_packing_and_block_scales_match_numpy():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(48, 256, generator=g) * 0.02
    w[2] = 0.0                      # all-zero row: zero scales, zero codes
    w[5, 9] = 1.5                   # outlier -> sets the global scale
    w[7, :16] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2) * 0.05
    w = w.bfloat16()
    old, N._OPS = N._OPS, FakeOps
    try:
        q, sf, ws2 = N.quantize_weight(w)
    finally:
        N._OPS = old
    rq, rsf, rws2 = np_nvfp4(w.float().numpy())
    assert q.dtype == torch.uint8 and q.shape == (48, 128)
    assert sf.dtype == torch.float8_e4m3fn and sf.shape == (48, 16)
    assert_codes_match(q.numpy(), rq)
    np.testing.assert_array_equal(sf.float().numpy(), rsf)
    assert ws2 == pytest.approx(rws2, rel=1e-7)
    assert not q[2].any() and not sf[2].float().any()
    deq = np_dequant(q.numpy(), sf.float().numpy(), ws2)
    rel = np.linalg.norm(deq - w.float().numpy()) / np.linalg.norm(w.float().numpy())
    assert rel < 0.12  # e2m1 weight-only floor ~9.5 % on Gaussian rows


def test_e2m1_ties_go_to_even_and_nibble_order():
    v = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.25, -9.0])
    assert [E2M1[c & 7] * (-1 if c & 8 else 1) for c in np_e2m1_code(v)] == \
        [0, 1, 1, 2, 2, 4, 4, 6, -0.0, -6]
    torch_codes = QT.e2m1_codes(torch.tensor(v, dtype=torch.float32)).numpy()
    np.testing.assert_array_equal(torch_codes, np_e2m1_code(v))
    w = torch.zeros(1, 16).bfloat16()
    w[0, 0], w[0, 1] = 6.0, -3.0  # even k -> low nibble
    q, _sf, _ = QT.quantize_weight(w, 6.0 * 448.0 / 448.0)
    assert int(q[0, 0]) == (0x7 | (0xD << 4))


# --- allowlist ----------------------------------------------------------------------
GEMMA_NAMES = {
    f"{L0}.self_attn.qkv_proj": True,
    f"{L1}.self_attn.q_proj": True,
    f"{L0}.self_attn.o_proj": True,
    f"{L0}.mlp.gate_up_proj": True,
    f"{L0}.mlp.down_proj": False,  # opt-in only (loose bound -> garbage on 26B)
    f"{L0}.router.proj": False,
    f"{L0}.moe.experts": False,
    "language_model.model.layers.0.per_layer_input_gate": False,
    "language_model.lm_head": False,
    "vision_tower.encoder.layers.0.self_attn.qkv_proj": False,
    "vision_tower.encoder.layers.0.mlp.down_proj": False,
    "embed_vision.embedding_projection": False,
    "audio_tower.layers.0.self_attn.o_proj": False,
}


def test_default_nvfp4_allowlist_on_gemma4_names():
    pats = P.nvfp4_layer_patterns()
    assert pats == P.NVFP4_DEFAULT_LAYERS
    assert {n: P.selected(n, pats) for n in GEMMA_NAMES} == GEMMA_NAMES


def test_env_allowlist_is_separate_from_fp8_and_deny_wins(monkeypatch):
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, "*.o_proj")
    monkeypatch.setenv(P.LAYERS_ENV, "*.qkv_proj")
    assert P.nvfp4_layer_patterns() == ("*.o_proj",)
    assert P.layer_patterns() == ("*.qkv_proj",)
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, "*")
    assert not any(P.selected(n, P.nvfp4_layer_patterns())
                   for n, v in GEMMA_NAMES.items()
                   if not v and "per_layer" not in n and "moe" not in n
                   and n != f"{L0}.mlp.down_proj")  # default-off, not denied


# --- activation bounds -----------------------------------------------------------------
def test_rmsnorm_bound_is_proven_and_attained():
    root = _gemma_like()
    mods = dict(root.named_modules())
    a, rule = N.act_bound(f"{L0}.self_attn.qkv_proj", mods)
    w = mods[f"{L0}.input_layernorm"].weight.float()
    assert rule == "input_layernorm" and a == pytest.approx(1.01 * math.sqrt(H) * 7.0)
    x = torch.zeros(H)
    x[3] = 5.0                               # all energy on the outlier channel
    y = x / torch.sqrt((x * x).mean() + 1e-6) * w
    assert float(y.abs().max()) <= a and float(y.abs().max()) > 0.98 * a / 1.01
    for _ in range(200):
        x = torch.randn(H) * torch.rand(H) ** 4
        y = x / torch.sqrt((x * x).mean() + 1e-6) * w
        assert float(y.abs().max()) <= a
    assert N.act_bound(f"{L0}.mlp.gate_up_proj", mods)[1] == "pre_feedforward_layernorm"


def test_o_proj_bound_needs_unscaled_v_norm():
    mods = dict(_gemma_like().named_modules())
    assert N.act_bound(f"{L0}.self_attn.o_proj", mods) == (1.01 * math.sqrt(HD), "v_norm")
    v = torch.randn(10, HD) * 30
    v = v / torch.sqrt((v * v).mean(-1, keepdim=True))      # v_norm, no scale
    o = torch.softmax(torch.randn(10), 0) @ v              # convex combination
    assert float(o.abs().max()) <= math.sqrt(HD)
    assert N.act_bound(f"{L1}.self_attn.o_proj", mods) is None  # no v_norm: stays bf16


def test_down_proj_cauchy_schwarz_bound_is_tight_for_the_worst_input():
    mods = dict(_gemma_like().named_modules())
    a_bound, rule = N.act_bound(f"{L0}.mlp.down_proj", mods)
    assert rule.startswith("gate_up")
    gu = mods[f"{L0}.mlp.gate_up_proj"].weight.float()
    r = 1.01 * math.sqrt(H) * float(mods[f"{L0}.pre_feedforward_layernorm"].weight.float().abs().max())
    best = 0.0
    for j in range(INTER):  # worst unit direction of y -> g_j u_j = y^T sym(a b^T) y
        a, b = gu[j].double(), gu[INTER + j].double()
        m = (torch.outer(a, b) + torch.outer(b, a)) / 2
        ev, vec = torch.linalg.eigh(m)
        for e, v in ((ev[0], vec[:, 0]), (ev[-1], vec[:, -1])):
            y = v * r
            g, u = float(a @ y), float(b @ y)
            gelu = g * 0.5 * (1 + math.tanh(0.7978845608 * (g + 0.044715 * g ** 3)))
            assert abs(gelu * u) <= a_bound and abs(g * u) <= a_bound
            best = max(best, abs(g * u))
    assert best > 0.97 * a_bound / 1.02   # the bilinear part is attained


def test_env_override_wins_and_is_validated(monkeypatch):
    mods = dict(_gemma_like().named_modules())
    monkeypatch.setenv(N.ACT_ENV, "*.mlp.down_proj=2048, *.o_proj=5")
    assert N.act_bound(f"{L0}.mlp.down_proj", mods) == (2048.0, "env")
    assert N.act_bound(f"{L1}.self_attn.o_proj", mods) == (5.0, "env")
    monkeypatch.setenv(N.ACT_ENV, "*.o_proj=-1")
    with pytest.raises(ValueError, match="bad entry"):
        N.act_bound(f"{L0}.self_attn.o_proj", mods)


# --- conversion over fake vLLM ------------------------------------------------------------
def test_convert_model_builds_modelopt_checkpoint_layers(fake_vllm, capsys, monkeypatch):
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, ",".join(P.NVFP4_DEFAULT_LAYERS + ("*language_model*.mlp.down_proj",)))  # down_proj is opt-in
    root = _gemma_like()
    mods = dict(root.named_modules())
    bf16 = {n: m.weight.data.clone() for n, m in mods.items() if isinstance(m, LinearBase)}
    done = N.convert_model(root)
    names = [d[0] for d in done]
    assert names == [f"{L0}.self_attn.qkv_proj", f"{L0}.self_attn.o_proj",
                     f"{L0}.mlp.gate_up_proj", f"{L0}.mlp.down_proj",
                     f"{L1}.self_attn.q_proj", f"{L1}.mlp.gate_up_proj",
                     f"{L1}.mlp.down_proj"]
    err = capsys.readouterr().err
    assert "converted 7 linears to NVFP4 W4A4" in err and "no activation bound" in err
    assert f"{L1}.self_attn.o_proj" in err  # skipped, named
    by_prefix = {m.prefix: m for m in FakeModelOptNvFp4Method.built}
    for name, n, k, amax, _rule in done:
        mod, ck = mods[name], by_prefix[name].ckpt
        assert mod.quant_method is by_prefix[name]
        rq, rsf, rws2 = np_nvfp4(bf16[name].float().numpy())
        assert_codes_match(ck["weight"].numpy(), rq)
        np.testing.assert_array_equal(ck["weight_scale"].float().numpy(), rsf)
        parts = len(mod.output_partition_sizes)
        assert ck["weight_scale_2"].shape == (parts,)
        assert torch.all(ck["weight_scale_2"] == ck["weight_scale_2"][0])  # one global
        assert float(ck["weight_scale_2"][0]) == pytest.approx(rws2, rel=1e-6)
        assert torch.allclose(ck["input_scale"], torch.full((parts,), amax / 2688.0))
        x = (torch.randn(3, 2, k) * amax / 20).bfloat16()
        out = mod(x)
        assert out.shape == (3, 2, n) and out.dtype == torch.bfloat16
        ref = nn.functional.linear(x, bf16[name])
        assert N.rel_frob(out, ref) < N.QUANT_BOUND
    for kept in (f"{L0}.router.proj", "vision_tower.layers.0.self_attn.qkv_proj",
                 "vision_tower.layers.0.mlp.down_proj", "embed_vision.embedding_projection",
                 f"{L1}.self_attn.o_proj"):
        assert mods[kept].weight.dtype == torch.bfloat16
        assert type(mods[kept].quant_method) is UnquantizedLinearMethod


def test_selftest_once_per_shape_and_fails_closed(fake_vllm, monkeypatch):
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, ",".join(P.NVFP4_DEFAULT_LAYERS + ("*language_model*.mlp.down_proj",)))  # down_proj is opt-in
    calls = []
    real = N._selftest
    monkeypatch.setattr(N, "_selftest", lambda *a: (calls.append(a[0]), real(*a)))
    N.convert_model(_gemma_like())
    assert len(calls) == len(set(calls)) == 4  # 7 layers, 4 distinct (N, K)

    class Broken(FakeModelOptNvFp4Method):
        def apply(self, layer, x, bias=None):
            return super().apply(layer, x, bias) * 1.05

    monkeypatch.setattr(N, "_BUILD", Broken)
    monkeypatch.setattr(N, "_SELFTESTED", set())
    with pytest.raises(RuntimeError, match="self-test FAILED"):
        N.convert_model(_gemma_like())


def test_convert_model_inert_off_sm120(fake_vllm, monkeypatch, capsys):
    monkeypatch.setattr(R, "_capability", lambda dev: (10, 0))
    root = _gemma_like()
    assert N.convert_model(root) == []
    assert "present but inert" in capsys.readouterr().err


def test_nvfp4_runs_before_fp8_and_fp8_skips_its_layers(fake_vllm, monkeypatch):
    order = []
    monkeypatch.setattr(N, "convert_model", lambda m: order.append("nvfp4"))
    monkeypatch.setattr(R, "convert_model", lambda m: order.append("fp8"))
    monkeypatch.setattr(P, "route_converted_hc", lambda m: order.append("hc"))
    monkeypatch.setenv(P.NVFP4_GATE_ENV, "1")
    P._suffix_fp8_dense_convert(None)
    assert order == ["nvfp4", "hc"]
    monkeypatch.setenv(P.GATE_ENV, "1")
    P._suffix_fp8_dense_convert(None)
    assert order == ["nvfp4", "hc", "nvfp4", "fp8", "hc"]


# --- gate / hook / drift --------------------------------------------------------------------
def _fake_target(tmp_path, src):
    f = tmp_path / "utils.py"
    f.write_text(src)
    m = types.ModuleType(P.TARGET_MODULE)
    m.__file__ = str(f)
    return m


def test_nvfp4_gate_alone_arms_the_shared_anchor(tmp_path, monkeypatch):
    assert not P.gate_enabled()
    assert P.apply(_fake_target(tmp_path, "garbage")) is False
    monkeypatch.setenv(P.NVFP4_GATE_ENV, "1")
    assert P.gate_enabled() and not P.fp8_enabled()
    monkeypatch.delitem(sys.modules, P.TARGET_MODULE, raising=False)
    assert P.install_post_import_hook()
    assert any(isinstance(f, _PostImportFinder) and f.target == P.TARGET_MODULE
               for f in sys.meta_path)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    with pytest.raises(P.PatchDriftError, match="convert_after_post_load"):
        P.apply(_fake_target(tmp_path, "def process_weights_after_loading(): pass\n"))
    mod = _fake_target(tmp_path, FIX.read_text())
    # the real rewrite exec's the module source: only check the pure transform
    new, applied = P.patch_source(FIX.read_text())
    assert applied == ["convert_after_post_load"] and "_suffix_fp8_dense_convert(model)" in new
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.31.0"))
    with pytest.raises(P.PatchDriftError, match="pinned 0.30.0"):
        P.apply(mod)


# --- oracle inventory -------------------------------------------------------------------------
def test_oracle_shapes_follow_the_hf_configs():
    def shapes(hidden, heads, kv, gkv, inter, tp):
        h, k, gk = heads // tp, max(1, kv // tp), max(1, gkv // tp)
        return {"qkv_proj.sliding": ((h + 2 * k) * 256, hidden),
                "qkv_proj.full": ((h + 2 * gk) * 512, hidden),
                "o_proj.sliding": (hidden, h * 256), "o_proj.full": (hidden, h * 512),
                "mlp.gate_up_proj": (2 * inter // tp, hidden),
                "mlp.down_proj": (hidden, inter // tp)}
    want = {"g26": shapes(2816, 16, 8, 2, 2112, 1), "g31tp2": shapes(5376, 32, 16, 4, 21504, 2)}
    for model, layer, n, k, _calls in O.INVENTORY:
        assert want[model][layer] == (n, k), (model, layer)
        assert n % 16 == 0 and k % 16 == 0
    assert sum(c for m, l, *_x, c in O.INVENTORY if m == "g26" and l.startswith("qkv")) == 30
    assert sum(c for m, l, *_x, c in O.INVENTORY if m == "g31tp2" and l.startswith("qkv")) == 60


def test_oracle_layer_stub_goes_through_convert_layer(fake_vllm):
    w = (torch.randn(64, 128) * 0.02).bfloat16()
    m, (q, sf, ws2) = O._layer(w.clone(), 256.0)
    assert isinstance(m.quant_method, FakeModelOptNvFp4Method)
    assert N.layer_act_amax(m) == pytest.approx(256.0, rel=1e-6)
    x = (torch.randn(4, 128) * 12).bfloat16()
    out = m.quant_method.apply(m, x).float()
    assert N.rel_frob(out, N.reference(x, q, sf, ws2, N.layer_act_amax(m))) < N.KERNEL_BOUND
    assert N.rel_frob(out, nn.functional.linear(x, w).float()) < N.QUANT_BOUND
    assert O._wq_mismatch(w, q, sf) == 0.0  # fake ops = the same torch quantizer


# --- vLLM 0.32.0 Qwen4Exp seams (fixtures: verbatim v0.30.0 / v0.32.0 sources) ---------------
FIXTURES = Path(__file__).resolve().parent / "fixtures"
V032_SHA = {
    "model_loader_utils.py": "3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6",
    "qwen4_exp_hyperconnection.py": "3cf42672bbf21b97a070fc8901ddb4eacfa66db4e05913187e5431f9ee905601",
    "qwen4_exp_qsa.py": "1eee006d61000de48aa783037dc2474202413310f9368fcd41c2713e1a41d95c",
}
V030_SHA = {
    "qwen4_exp_hyperconnection.py": "2a15d6b22fbe1def4d2bcd85568269d1157c5e1c298f0c6af43c1fb9797489ba",
    "qwen4_exp_qsa.py": "9ac72b0e5f02341704a2f5a49bf2f4a6f141d2e241e9db07ea5918592bac90ea",
}


def _fix(version, name):
    return (FIXTURES / f"vllm_{version}" / name).read_text()


def test_v032_fixtures_are_pinned_upstream_copies():
    import hashlib
    for version, pins in (("0.32.0", V032_SHA), ("0.30.0", V030_SHA)):
        for name, sha in pins.items():
            assert hashlib.sha256(_fix(version, name).encode()).hexdigest() == sha, (version, name)


@pytest.mark.parametrize("version", ["0.30.0", "0.32.0"])
def test_anchor_replays_on_both_pins(version):
    new, applied = P.patch_source(_fix(version, P.FIXTURE))
    assert applied == ["convert_after_post_load"] and "_suffix_fp8_dense_convert(model)" in new
    assert version.startswith(P.SUPPORTED_VLLM) and "0.32.0rc1.dev329+g39983a5e7".startswith(
        P.SUPPORTED_VLLM)
    assert not "0.33.0".startswith(P.SUPPORTED_VLLM)


def test_upstream_seams_the_guards_key_on():
    """0.32 HC: _use_hc_down_silu (set at __init__ from the BF16 weight dtype)
    gates a direct read of input_mix_weight_down_block_inject.weight; 0.32 QSA:
    fuse_indexer_projection swaps qkv_proj for the packed QKV+indexer linear.
    0.30 has neither (both guards are no-ops there)."""
    hc32, qsa32 = _fix("0.32.0", "qwen4_exp_hyperconnection.py"), _fix("0.32.0", "qwen4_exp_qsa.py")
    assert "self._use_hc_down_silu = (" in hc32 and "weight.dtype == torch.bfloat16" in hc32
    assert ("self._use_hc_down_silu\n" in hc32 and "hc_down_silu(\n                xn,\n"
            "                self.input_mix_weight_down_block_inject.weight," in hc32)
    assert "self.fuse_indexer_projection = vllm_config.lora_config is None" in qsa32
    assert "self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(" in qsa32
    assert "del self.indexer.index_qk_proj" in qsa32
    hc30, qsa30 = _fix("0.30.0", "qwen4_exp_hyperconnection.py"), _fix("0.30.0", "qwen4_exp_qsa.py")
    assert "hc_down_silu" not in hc30 and "fuse_indexer_projection" not in qsa30
    for src in (hc30, hc32):  # the glob leaf names the pool env lists are unchanged
        for leaf in ("input_mix_weight_down_block_inject", "input_mix_weight_down",
                     "input_mix_weight_up"):
            assert f'maybe_prefix(prefix, "{leaf}")' in src


def _qsa_like():
    torch.manual_seed(0)
    root = nn.Module()
    lm = root.language_model = nn.Module()
    lm.model = nn.Module()
    layer = _decoder()
    layer.self_attn.fuse_indexer_projection = True  # vLLM 0.32 QSA, no LoRA
    lm.model.layers = nn.ModuleDict({"0": layer})
    return root


def test_fused_qsa_indexer_qkv_is_refused_unless_accepted(fake_vllm, monkeypatch):
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, "*language_model*self_attn.qkv_proj")
    with pytest.raises(P.PatchDriftError, match="indexer"):
        N.convert_model(_qsa_like())
    root = _qsa_like()
    root.language_model.model.layers["0"].self_attn.fuse_indexer_projection = False  # 0.30 / LoRA
    assert [d[0] for d in N.convert_model(root)] == [f"{L0}.self_attn.qkv_proj"]
    monkeypatch.setenv(P.QSA_INDEXER_ENV, "1")
    assert [d[0] for d in N.convert_model(_qsa_like())] == [f"{L0}.self_attn.qkv_proj"]


def test_fp8_mode_refuses_the_fused_qsa_indexer_too(monkeypatch):
    lin = types.ModuleType("vllm.model_executor.layers.linear")
    lin.LinearBase, lin.UnquantizedLinearMethod = LinearBase, UnquantizedLinearMethod
    lin.LinearMethodBase = object
    monkeypatch.setitem(sys.modules, lin.__name__, lin)
    monkeypatch.setattr(R, "_capability", lambda dev: (12, 0))
    monkeypatch.setattr(R, "_OPS", object())
    monkeypatch.setattr(R, "_METHOD_CLS", None)
    with pytest.raises(P.PatchDriftError, match="indexer"):
        R.convert_model(_qsa_like(), patterns=("*self_attn.qkv_proj",))


class _HC(nn.Module):
    """vLLM 0.32 GatedResidual (use_combine=True) attributes the guard reads."""

    def __init__(self):
        super().__init__()
        self.input_mix_weight_down_block_inject = LinearBase(336, 64)
        self.input_mix_weight_up = LinearBase(64, 32)
        self._use_hc_down_silu = True


def test_converted_hc_down_inject_leaves_the_bf16_fast_path(fake_vllm, monkeypatch, capsys):
    monkeypatch.setenv(P.NVFP4_GATE_ENV, "1")
    monkeypatch.setenv(P.NVFP4_LAYERS_ENV, "*attn_hyper_connection.input_mix_weight_down_block_inject")
    monkeypatch.setenv(N.ACT_ENV, "*input_mix_weight_down_block_inject=8.0")
    root = nn.Module()
    root.layers = nn.ModuleDict({"0": nn.Module(), "1": nn.Module()})
    root.layers["0"].attn_hyper_connection = _HC()
    root.layers["0"].mlp_hyper_connection = _HC()  # not allowlisted: stays BF16 + fast path
    root.layers["1"].attn_hyper_connection = _HC()
    root.layers["1"].attn_hyper_connection._use_hc_down_silu = False  # vLLM said no already
    P._suffix_fp8_dense_convert(root)
    hc0 = root.layers["0"].attn_hyper_connection
    assert hc0.input_mix_weight_down_block_inject.weight.dtype == torch.uint8
    assert hc0._use_hc_down_silu is False
    assert root.layers["0"].mlp_hyper_connection._use_hc_down_silu is True
    assert root.layers["1"].attn_hyper_connection._use_hc_down_silu is False
    assert "1 HyperConnection down+inject projections converted" in capsys.readouterr().err
    assert P.route_converted_hc(root) == []  # idempotent
