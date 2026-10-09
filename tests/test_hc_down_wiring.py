# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_HC_DOWN wiring in suffix_hybrid/rocm_patches.py (no vLLM,
no GPU), alone and together with SUFFIX_ROCM_HC_FUSE in either order. The fake module
carries GatedResidual.mix / combine_and_mix / combine verbatim from vLLM 81198e97
vllm/models/qwen4_exp/amd/hyperconnection.py (lines 127-194) over stand-in ops; the
kernel's numerics are checked on silicon by `python -m suffix_hybrid.kernels.hc_down_rocm`
(boot gate hc_down_bench).
"""
import importlib
import pathlib
import sys
import types

import pytest

from suffix_hybrid import rocm_patches as rp

FUSE, DOWN = "SUFFIX_ROCM_HC_FUSE", "SUFFIX_ROCM_HC_DOWN"
DOWN_MIX, DOWN_CAM = rp.PATCHES[DOWN]
STUBS = '''from __future__ import annotations


class Tag(tuple):  # every tensor; .split tags the lora / injection / pad columns
    def split(self, sizes, dim):
        return [Tag((part,) + self) for part in ("lora", "inj", "pad")]


class Lin:  # a stock vLLM Linear
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return Tag(("gemm", self.weight, x))


def grouped_gemma_rmsnorm(x, w, eps, hc):
    return "xn"


def hc_combine_norm(residual, block_output, injection, w, eps, hc):
    return "h2", "xn"


def hc_silu(x, hc):
    return ("silu", x)


def hc_gate_mix(xn, gate, hc):
    return ("gmix", xn, gate)


def hc_combine(*args):
    return "combined"


class GatedResidual:
    config = type("Cfg", (), {"rms_norm_eps": 1e-6})()
    hc_norm = Lin("W_norm")
    hc_count, lora_rank, pad_size = 4, 320, 12
    input_mix_weight_down_block_inject = Lin("W_dbi")
    input_mix_weight_down = Lin("W_d")
    input_mix_weight_up = Lin("W_up")

    def __init__(self, use_combine):
        self.use_combine = use_combine

'''
METHODS = '''    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)
'''
FAKE_HC = STUBS + METHODS
KERNELS = (
    "def install_fuse(module):\n"
    "    module.hc_up_gate_mix = lambda lora, w, xn, hc: ('fused', lora, w, xn, hc)\n"
    "def install_down(module):\n"
    "    module.hc_down = lambda x, w: module.Tag(('down', w, x))\n")


def _calls(ns):
    """mix / combine_and_mix of a combining site and of a final mixer."""
    gr = ns["GatedResidual"]
    return ([gr(use_combine).mix("h") for use_combine in (True, False)]
            + [gr(use_combine).combine_and_mix("h", "out", "inj") for use_combine in (True, False)])


def _expected(gates):
    g = "down" if DOWN in gates else "gemm"
    out = []
    for hidden in ("h", "h2"):  # mix keeps hidden_states, combine_and_mix returns the combine
        for lora, inj in ((("lora", g, "W_dbi", "xn"), ("inj", g, "W_dbi", "xn")),
                          ((g, "W_d", "xn"), None)):
            block = (("fused", lora, "W_up", "xn", 4) if FUSE in gates
                     else ("gmix", "xn", ("gemm", "W_up", ("silu", lora))))
            out.append((hidden, block, inj))
    return out


@pytest.mark.parametrize("gates", [(), (DOWN,), (FUSE, DOWN), (DOWN, FUSE)])
def test_anchors_alone_and_with_hc_fuse_in_either_order(gates):
    src = FAKE_HC
    for gate in gates:
        for patch in rp.PATCHES[gate]:
            src = rp.patch_source(patch, src)  # raises unless the anchor occurs exactly once
    hc, kernels = types.ModuleType("hc"), {}
    exec(compile(src, "<hc>", "exec"), vars(hc))
    exec(KERNELS, kernels)
    kernels["install_fuse"](hc)
    kernels["install_down"](hc)
    assert _calls(vars(hc)) == _expected(gates)
    if DOWN in gates:
        for patch in (DOWN_MIX, DOWN_CAM):
            with pytest.raises(RuntimeError, match="drifted"):  # already rewritten: 0 hits
                rp.patch_source(patch, src)


def test_anchor_drift_fails_closed():
    # combine_and_mix's norm taking hidden_states right before the weight -> 2 mix hits
    twin = FAKE_HC.replace("prev_injection,\n            self.hc_norm.weight",
                           "hidden_states,\n            self.hc_norm.weight")
    with pytest.raises(RuntimeError, match="found 2x"):
        rp.patch_source(DOWN_MIX, twin)


def test_hook_applies_both_gates(tmp_path, monkeypatch, capsys):
    (tmp_path / "fake_hc.py").write_text(FAKE_HC)
    (tmp_path / "fake_hc_kernels.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(tmp_path))
    for gate, hook in ((FUSE, "fake_hc_kernels:install_fuse"),
                       (DOWN, "fake_hc_kernels:install_down")):
        mix, cam = rp.PATCHES[gate]
        monkeypatch.setitem(rp.PATCHES, gate, (mix._replace(target="fake_hc", after=hook),
                                               cam._replace(target="fake_hc")))
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv(FUSE, "1")
    monkeypatch.setenv(DOWN, "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rp.install_post_import_hook()
    try:
        hc = importlib.import_module("fake_hc")
        assert _calls(vars(hc)) == _expected((FUSE, DOWN))
        assert getattr(hc, rp._MARK)
        err = capsys.readouterr().err
        assert err.count("ACTIVE: HC down projection") == 2 and err.count("ACTIVE: HC silu") == 2
    finally:
        for name in ("fake_hc", "fake_hc_kernels"):
            sys.modules.pop(name, None)
    assert DOWN_MIX.after == "suffix_hybrid.kernels.hc_down_rocm:install" and not DOWN_CAM.after


def test_sitecustomize_arms_every_rocm_gate():
    # A gate missing from sitecustomize's tuple never installs the hook: stock, silently.
    src = (pathlib.Path(__file__).resolve().parents[1] / "sitecustomize.py").read_text()
    arm = src[:src.index("from suffix_hybrid.rocm_patches import install_post_import_hook")]
    arm = arm[arm.rindex("if any("):]  # the gate check that imports the hook
    assert [g for g in rp.PATCHES if f'"{g}"' not in arm] == []
