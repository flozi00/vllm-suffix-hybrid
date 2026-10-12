# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_ACT_QUANT_FUSE wiring (no vLLM, AITER or GPU): the rewritten
GDN _output_projection and QSA forward tail (vLLM 81198e97 text) take the fused op only
for an eligible layer, the QSA gate is read from the fused QK kernel's copy or from the
gate halves of q_gate in qkv, and a layer that is not served by vLLM's non-ASM AITER MXFP4
kernel at TP 1 stays stock. Silicon check: boot gate act_quant_bench."""
import importlib.util
import sys
import types
from types import SimpleNamespace

import torch

from suffix_hybrid import rocm_patches as rp

if importlib.util.find_spec("vllm") is None:  # CPU CI: the module only needs @jit + the op registry
    _jit = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)  # noqa: E731
    _tu = types.ModuleType("vllm.triton_utils")
    _tu.tl, _tu.triton = SimpleNamespace(constexpr=None), SimpleNamespace(jit=_jit)
    _ut = types.ModuleType("vllm.utils.torch_utils")
    _ut.direct_register_custom_op = lambda **k: None
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.triton_utils": _tu,
                        "vllm.utils": types.ModuleType("vllm.utils"), "vllm.utils.torch_utils": _ut})
    try:
        from suffix_hybrid.kernels import act_quant_rocm as aq
    finally:
        for k in ("vllm", "vllm.triton_utils", "vllm.utils", "vllm.utils.torch_utils"):
            del sys.modules[k]
from suffix_hybrid.kernels import act_quant_rocm as aq  # noqa: E402

GDN, QSA = rp.PATCHES["SUFFIX_ROCM_ACT_QUANT_FUSE"]


def test_gdn_output_projection_dispatch():
    src = rp.patch_source(GDN, "def _output_projection(self, core_attn_out, z):\n" + GDN.old)
    calls = []
    mod = {"_suffix_aq_gdn_ok": lambda layer: layer.ok,
           "_suffix_aq_gdn_out": lambda layer, x, z: calls.append((x, z)) or "fused"}
    exec(compile(src, "fake_gdn", "exec"), mod)
    flat = SimpleNamespace(flatten=lambda d: ("flat", d))
    layer = SimpleNamespace(ok=True, norm=lambda x, z: flat, out_proj=lambda x: (("proj", x), None))
    assert mod["_output_projection"](layer, "x", "z") == "fused" and calls == [("x", "z")]
    layer.ok = False
    assert mod["_output_projection"](layer, "x", "z") == ("proj", ("flat", -2))
    assert GDN.after == "suffix_hybrid.kernels.act_quant_rocm:install_gdn"


def test_qsa_tail_dispatch():
    src = rp.patch_source(QSA, "def tail(self, attn_output, num_tokens, gate, qkv):\n" + QSA.old)
    calls = []
    mod = {"torch": torch, "_suffix_aq_qsa_ok": lambda layer: layer.ok,
           "_suffix_aq_qsa_out": lambda layer, f, g, qkv: calls.append((f, g, qkv)) or "fused"}
    exec(compile(src, "fake_qsa", "exec"), mod)
    a, g = torch.ones(2, 4), torch.zeros(2, 4)
    layer = SimpleNamespace(ok=True, o_proj=lambda x: (x * 3, None))
    assert mod["tail"](layer, a, 2, g, "qkv") == "fused" and calls[0][1:] == (g, "qkv")
    assert torch.equal(mod["tail"](layer, a, 2, None, "qkv"), a * 3)  # no gate: stock
    layer.ok = False
    assert torch.equal(mod["tail"](layer, a, 2, g, "qkv"), a * torch.sigmoid(g) * 3)
    assert QSA.after == "suffix_hybrid.kernels.act_quant_rocm:install_qsa"


class _Kernel:
    def __init__(self, asm=False, dtype=torch.bfloat16):
        self.use_asm_gemm, self.out_dtype = asm, dtype


def _linear(kernel=None, tp=1, bias=None, wdtype=torch.uint8):
    return SimpleNamespace(scheme=SimpleNamespace(ocp_mx_linear=kernel or _Kernel()), tp_size=tp,
                           bias=bias, weight=torch.zeros(2, 2, dtype=wdtype),
                           weight_scale=torch.zeros(2, 2, dtype=torch.uint8))


def test_eligibility_and_gate_source(monkeypatch):
    monkeypatch.setattr(aq, "_AITER_KERNEL", _Kernel)
    assert aq._linear_ok(_linear())
    for bad in (_linear(_Kernel(asm=True)), _linear(tp=2), _linear(bias=torch.zeros(2)),
                _linear(wdtype=torch.bfloat16), _linear(_Kernel(dtype=torch.float16)),
                SimpleNamespace(scheme=None, tp_size=1, bias=None, weight=torch.zeros(1))):
        assert not aq._linear_ok(bad)
    norm = SimpleNamespace(activation="sigmoid", norm_before_gate=True, group_size=None,
                           weight="nw", eps=1e-6)
    gdn = SimpleNamespace(norm=norm, head_v_dim=128, out_proj=_linear())
    assert aq.gdn_ok(gdn)
    for k, v in (("activation", "silu"), ("norm_before_gate", False), ("group_size", 64)):
        assert not aq.gdn_ok(SimpleNamespace(norm=SimpleNamespace(**{**vars(norm), k: v}),
                                             head_v_dim=128, out_proj=_linear()))
    seen = []
    ops = SimpleNamespace(suffix_aq_gdn_out=lambda *a: seen.append(a) or "g",
                          suffix_aq_qsa_out=lambda *a: seen.append(a) or "q")
    monkeypatch.setattr(aq, "torch", SimpleNamespace(ops=SimpleNamespace(vllm=ops),
                                                     bfloat16=torch.bfloat16, uint8=torch.uint8))
    assert aq.gdn_out(gdn, "x", "z") == "g"
    assert seen.pop()[:4] == ("x", "z", "nw", 1e-6)
    qkv = torch.arange(2 * 192).view(2, 192)  # 2 heads x (q 32 | gate 32) | k | v
    for fused, (src, g_head) in ((True, ("gate", 32)), (False, (qkv[:, 32:], 64))):
        qsa = SimpleNamespace(head_dim=32, attn_output_gate=True, o_proj=_linear(),
                              use_fused_qk_norm_rope_gate=fused)
        assert aq.qsa_ok(qsa) and aq.qsa_out(qsa, "flat", "gate", qkv) == "q"
        args = seen.pop()
        assert args[0] == "flat" and args[2:4] == (g_head, 32)
        assert args[1] == src if fused else torch.equal(args[1], src)
    assert not aq.qsa_ok(SimpleNamespace(head_dim=32, attn_output_gate=False, o_proj=_linear()))
    assert not aq.qsa_ok(SimpleNamespace(head_dim=48, attn_output_gate=True, o_proj=_linear()))
