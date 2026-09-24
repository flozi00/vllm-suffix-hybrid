# SPDX-License-Identifier: Apache-2.0
"""Regression tests: env-gate parse harmonization (review aa5accb1 #2).

The three `!= "0"` gates (SUFFIX_HYBRID_D2H_PIPE, SUFFIX_HYBRID_W0_FASTPATH,
SUFFIX_HYBRID_PIPE_EARLY) used to treat ANY non-"0" value -- including
"false", "no", "off" -- as ON. An operator setting
SUFFIX_HYBRID_D2H_PIPE=false on a live pod ARMED the pipeline; the same
value on W0_FASTPATH kept the fast path on. These tests pin the
harmonized _env_flag polarity: only 1/true/yes/on arm; 0/false/no/off/
empty/unknown disarm. CPU-only (SIM arm / State inspection).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from suffix_hybrid import wrap_v2

# ---- unit: _env_flag polarity -------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "yes", "on",
                                  " TRUE ", "Yes", "On", "1"])
def test_env_flag_on_values(monkeypatch, val):
    monkeypatch.setenv("SUFFIX_HYBRID_TEST_GATE", val)
    assert wrap_v2._env_flag("SUFFIX_HYBRID_TEST_GATE") is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "",
                                  " FALSE ", "No", "Off", "2", "junk"])
def test_env_flag_off_values(monkeypatch, val):
    monkeypatch.setenv("SUFFIX_HYBRID_TEST_GATE", val)
    assert wrap_v2._env_flag("SUFFIX_HYBRID_TEST_GATE") is False


def test_env_flag_default_when_unset(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_TEST_GATE", raising=False)
    assert wrap_v2._env_flag("SUFFIX_HYBRID_TEST_GATE", "1") is True
    assert wrap_v2._env_flag("SUFFIX_HYBRID_TEST_GATE", "0") is False


# ---- integration: the three gates inside the wrapper --------------------

K = 4
NROWS = 4
NCOLS = 64


class FakeTotals:
    def __init__(self, totals):
        self.t = torch.tensor(list(totals), dtype=torch.int64)

    def __getitem__(self, idx):
        return self.t[idx]


class Speculator:
    def propose(self, *args, **kwargs):
        raise AssertionError("suffix-only replaces propose; never called")

    def __init__(self, k=K, max_reqs=NROWS):
        self.draft_tokens = torch.zeros((max_reqs, k), dtype=torch.int64)
        self.max_num_reqs = max_reqs
        self.max_model_len = None
        self.num_speculative_steps = k
        self.draft_logits = None


class TP:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, value, src=0):
        return value


def make_env(monkeypatch):
    for var in ("SUFFIX_HYBRID_SUFFIX_ONLY", "SUFFIX_HYBRID_W0_FASTPATH",
                "SUFFIX_HYBRID_D2H_PIPE", "SUFFIX_HYBRID_D2H_PIPE_SIM",
                "SUFFIX_HYBRID_PIPE_EARLY", "SUFFIX_HYBRID_TRACE",
                "SUFFIX_HYBRID_LOG_INTERVAL", "SUFFIX_HYBRID_SUFFIX_MIN",
                "SUFFIX_HYBRID_UNIFORM_K"):
        monkeypatch.delenv(var, raising=False)


def wrap_sim(monkeypatch, pipe_val, sim="1", early=None, fast=None):
    """Wrap with the CPU-only SIM pipeline arm; returns wrapped proposer."""
    from suffix_hybrid._native import HybridMixer
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE", pipe_val)
    monkeypatch.setenv("SUFFIX_HYBRID_D2H_PIPE_SIM", sim)
    if early is not None:
        monkeypatch.setenv("SUFFIX_HYBRID_PIPE_EARLY", early)
    if fast is not None:
        monkeypatch.setenv("SUFFIX_HYBRID_W0_FASTPATH", fast)
    ats_np = np.zeros((NROWS, NCOLS), dtype=np.int32)
    ats_np[2, :12] = 101
    totals = FakeTotals([3, 3, 12, 6])
    states = NS(total_len=NS(gpu=totals),
                all_token_ids=NS(_uva_buf=NS(np=ats_np)),
                req_id_to_index={"a": 2, "b": 3})
    runner = NS(req_states=states, vllm_config=NS(
        scheduler_config=NS(async_scheduling=False,
                            max_concurrent_batches=1,
                            num_speculative_tokens=K)))
    spec = Speculator()
    mixer = HybridMixer(K, 512)
    wrapped = wrap_v2._suffix_only_wrap(runner, spec, mixer, TP(), K)
    return wrapped, runner


@pytest.mark.parametrize("gate_val", ["false", "FALSE", "no", "off", ""])
def test_d2h_pipe_false_variant_never_arms(monkeypatch, gate_val):
    # SUFFIX_HYBRID_D2H_PIPE=<false-ish> must leave the pipe OFF
    # (review #2: the old != "0" parse ARMED the pipeline on these).
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, gate_val)
    pipe = getattr(wrapped, "_suffix_pipe", None)
    assert pipe is None or pipe["on"] is False, (
        f"SUFFIX_HYBRID_D2H_PIPE={gate_val!r} must NOT arm the pipe")


@pytest.mark.parametrize("gate_val", ["1", "true", "yes", "on"])
def test_d2h_pipe_on_values_arm_sim(monkeypatch, gate_val):
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, gate_val)
    pipe = getattr(wrapped, "_suffix_pipe", None)
    assert pipe is not None and pipe["on"] is True


@pytest.mark.parametrize("gate_val", ["false", "FALSE", "no", "off"])
def test_w0_fastpath_false_variant_disarms(monkeypatch, gate_val):
    # W0_FASTPATH=<false-ish> must turn the fast path OFF (old parse
    # kept it ON -- breaking the "OFF = bit-identical stock" contract).
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, "0", sim="0", fast=gate_val)
    assert wrapped._suffix_w0fp["enabled"] is False, (
        f"SUFFIX_HYBRID_W0_FASTPATH={gate_val!r} must disable the arm")


def test_w0_fastpath_default_and_one_stay_on(monkeypatch):
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, "0", sim="0")  # unset = ON
    assert wrapped._suffix_w0fp["enabled"] is True
    wrapped, runner = wrap_sim(monkeypatch, "0", sim="0", fast="1")
    assert wrapped._suffix_w0fp["enabled"] is True


@pytest.mark.parametrize("gate_val", ["false", "FALSE", "no", "off"])
def test_pipe_early_false_variant_stays_option_a(monkeypatch, gate_val):
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, "1", sim="1", early=gate_val)
    pipe = wrapped._suffix_pipe
    # Pipe arms (SIM), but the EARLY entry-record arm must stay OFF.
    assert pipe["on"] is True and pipe["early"] is False, (
        f"SUFFIX_HYBRID_PIPE_EARLY={gate_val!r} must not arm EARLY")


def test_pipe_early_on_values_arm_early(monkeypatch):
    make_env(monkeypatch)
    wrapped, runner = wrap_sim(monkeypatch, "1", sim="1", early="true")
    assert wrapped._suffix_pipe["early"] is True