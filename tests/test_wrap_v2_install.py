"""Install-path and fixed-width semantics tests for the V2 wire-up.

CPU-only: fakes replace the CUDA runner/speculator; the compiled Rust mixer
is real. These pin the wrapper's contract (passthrough, degrade, fixed-width
row write, capability gate), not GPU throughput.
"""
from types import SimpleNamespace as NS
import pytest
import torch
from suffix_hybrid import wrap_v2
from suffix_hybrid._native import HybridMixer


class TP:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, value, src=0):
        return value


def _fixture(native):
    states = NS(total_len=NS(gpu=torch.tensor([8, 2])),
                all_token_ids=NS(gpu=torch.tensor([list(range(1, 33)),
                                                   [700] * 32])))
    runner = NS(req_states=states)
    batch = NS(req_ids=['a'], num_reqs=1, idx_mapping=torch.tensor([0]))

    def invoke(wrapped, mixer_calls, num_sampled=torch.tensor([1]),
               num_rejected=torch.tensor([0]), **kw):
        return wrapped(batch, {}, {}, torch.zeros(1), None,
                       num_sampled, num_rejected,
                       torch.zeros(2), torch.zeros(1, 2),
                       torch.tensor([0.8, 0.8]), torch.ones(2), **kw)

    return runner, batch, invoke


def test_dummy_and_profile_pass_through_native_untouched():
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class Boom:
        def mix_numpy(self, *a, **k):
            raise AssertionError("mixer must not run on dummy/profile calls")

    calls = []

    def original(*a, **k):
        calls.append(1)
        return native

    wrapped = wrap_v2._wrap_propose(runner, original, Boom(), TP())
    for kw in ({"dummy_run": True}, {"is_profile": True},
               {"skip_attn_for_dummy_run": True}):
        got = invoke(wrapped, calls, **kw)
        assert got is native
    assert len(calls) == 3


def test_short_mixed_row_keeps_native_tail_fixed_width():
    # A mixed proposal shorter than K must NOT shorten the row: slots past
    # the suffix tail retain the native draft token (never -1 padding).
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class Short:
        def mix_numpy(self, *a, **k):
            return [[9, 10]]

    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    Short(), TP())
    got = invoke(wrapped, None)
    assert got.shape == native.shape
    assert got.tolist() == [[9, 10, 11, 12, 99]]


def test_overlong_mixed_row_degrades_to_native():
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class Long:
        def mix_numpy(self, *a, **k):
            return [[9, 10, 11, 12, 99, 100]]

    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    Long(), TP())
    got = invoke(wrapped, None)
    assert got.tolist() == native.tolist()


def test_mixer_raise_degrades_to_native_not_propagates():
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class Boom:
        def mix_numpy(self, *a, **k):
            raise RuntimeError("mixer exploded")

    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    Boom(), TP())
    got = invoke(wrapped, None)
    assert got.tolist() == native.tolist()


def test_non_owner_rank_skips_mixer_and_broadcasts_native():
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class Boom:
        def mix_numpy(self, *a, **k):
            raise AssertionError("non-owner rank must not mix")

    class Slave(TP):
        rank_in_group = 1

    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    Boom(), Slave())
    got = invoke(wrapped, None)
    assert got.tolist() == native.tolist()


def test_real_mixer_round_trip_stays_fixed_width():
    # End-to-end with the compiled mixer: rows never exceed K and the
    # second proposal (learned split) keeps the [1, K] shape.
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = HybridMixer(5, 32)
    mixer.suffix_cache.add_sequence(list(range(1, 30)))
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    first = invoke(wrapped, None)
    assert first.shape == native.shape
    second = invoke(wrapped, None)
    assert second.shape == native.shape
    assert second.dtype == native.dtype


def test_capability_gate_rejects_renamed_propose():
    class Bad:
        __name__ = "Bad"

        def propose(self, input_batch, attn_metadata):  # missing params
            pass

    with pytest.raises(RuntimeError, match="missing params"):
        wrap_v2._check_speculator_capability(Bad)


def test_capability_gate_accepts_full_shape():
    class Good:
        __name__ = "Good"

        def propose(self, input_batch, attn_metadata, slot_mappings,
                    last_hidden_states, aux_hidden_states, num_sampled,
                    num_rejected, last_sampled, next_prefill_tokens,
                    temperature, seeds, dp_sync=None, dummy_run=False,
                    skip_attn_for_dummy_run=False, mm_inputs=None,
                    is_profile=False):
            pass

    assert callable(wrap_v2._check_speculator_capability(Good))


def test_install_v2_disabled_without_env(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_WRAP", raising=False)
    assert wrap_v2.install_v2() is False


def test_install_v2_absent_module_returns_false(monkeypatch):
    # The test venv has no vllm: install_v2 must report "not V2" (False) so
    # sitecustomize falls through to the V1 hook, never raise.
    monkeypatch.setenv("SUFFIX_HYBRID_WRAP", "1")
    assert wrap_v2.install_v2() is False
