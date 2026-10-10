# SPDX-License-Identifier: Apache-2.0
"""CPU wiring tests for suffix_hybrid.kernels.mxfp4_lm_head (no vLLM, no GPU).

Silicon check: boot gate mxfp4_lmhead_bench on an MI350P prints fidelity and
us/call vs the stock head; a pod with SUFFIX_MXFP4_LMHEAD=1 logs LOAD ORACLE
PASS, GRAPHS and ACTIVE.
"""
import pytest

from suffix_hybrid.kernels import mxfp4_lm_head as m


def test_gate_off_is_inert(monkeypatch):
    monkeypatch.delenv(m.GATE, raising=False)
    assert m.register() is None


def test_max_m_env(monkeypatch):
    monkeypatch.delenv(m.MAX_M_ENV, raising=False)
    assert m.max_m_env() == m.MAX_M
    monkeypatch.setenv(m.MAX_M_ENV, "4")
    assert m.max_m_env() == 4
    monkeypatch.setenv(m.MAX_M_ENV, "257")
    with pytest.raises(ValueError):
        m.max_m_env()


def test_eligible():
    assert m.eligible(248320, 2560, "torch.bfloat16") is None
    assert "bf16" in m.eligible(248320, 2560, "torch.float16")
    assert "% 32" in m.eligible(248320, 2550, "torch.bfloat16")
