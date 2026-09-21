"""CPU contract tests: real Torch tensors and the compiled Rust mixer.

The vLLM runner/model doubles replace unavailable CUDA forwards only. These
are not GPU acceptance or distributed-performance claims.
"""
from types import SimpleNamespace as NS
import pytest
import torch
from suffix_hybrid import _native

# n=2 comes from tests/conftest.py (process-wide for the test suite).
def _mixer(k, max_len):
    return _native.HybridMixer(k, max_len)


class TP:
    rank_in_group = 0
    world_size = 1

    def all_reduce(self, value):
        return value

    def broadcast(self, value, src=0):
        return value


def fixture():
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    states = NS(total_len=NS(gpu=torch.tensor([8, 2])),
                all_token_ids=NS(gpu=torch.tensor([list(range(1, 33)), [700] * 32])))
    runner = NS(req_states=states, speculator=NS(draft_logits=None))
    batch = NS(req_ids=['a'], num_reqs=1, idx_mapping=torch.tensor([0]))
    mixer = _mixer(5, 32)
    mixer.suffix_cache.add_sequence(list(range(1, 30)))
    assert mixer.suffix_cache.stats()['index_n'] == 2
    calls = []

    def original(*args, **kwargs):
        calls.append('native')
        return native

    def invoke(wrapped, **kw):
        return wrapped(batch, {}, {}, torch.zeros(1), None,
                       kw.pop('num_sampled', torch.tensor([1])),
                       kw.pop('num_rejected', torch.tensor([0])),
                       torch.zeros(2), torch.zeros(1, 2),
                       torch.tensor([0.8, 0.8]), torch.ones(2), **kw)

    return runner, batch, mixer, native, original, invoke, calls


def test_native_then_full_width_mix_stochastic_target():
    from suffix_hybrid import wrap_v2
    runner, batch, mixer, native, original, invoke, calls = fixture()
    wrapped = wrap_v2._wrap_propose(runner, original, mixer, TP())
    got = invoke(wrapped)
    assert calls == ['native']
    # First proposal for the row is native-only by design (no per-row
    # evidence yet); the learned split binds from the second proposal on.
    assert got.tolist() == [[9, 10, 11, 12, 99]]
    assert native.tolist() == [[9, 10, 11, 12, 99]]
    assert got.shape == native.shape and got.dtype == native.dtype
    got2 = invoke(wrapped)
    assert got2.tolist() == [[9, 10, 11, 12, 13]]
    assert mixer.get_stats()['suffix_proposed'] == 1


def test_feedback_follows_request_ids_after_slot_reorder():
    from suffix_hybrid import wrap_v2
    runner, batch, mixer, native, _, invoke, _ = fixture()
    batch.req_ids, batch.num_reqs = ['a', 'b'], 2
    batch.idx_mapping = torch.tensor([0, 1])
    original = lambda *a, **kw: native.repeat(2, 1)
    wrapped = wrap_v2._wrap_propose(runner, original, mixer, TP())
    invoke(wrapped, num_sampled=torch.tensor([1, 1]), num_rejected=torch.tensor([0, 0]))
    batch.req_ids, batch.idx_mapping = ['b', 'a'], torch.tensor([1, 0])
    # Row 'a' is accepted 4-of-5 (rejected at the tail); the row id follows
    # the slot reorder, so the feedback trains row a, not slot 0. Its next
    # proposal carries a genuinely non-native tail ([..,13] for [..,99]).
    got2 = invoke(wrapped, num_sampled=torch.tensor([6, 5]), num_rejected=torch.tensor([0, 1]))
    assert got2.tolist()[1] == [9, 10, 11, 12, 13]
    stats = mixer.get_stats()
    assert stats['native_accepted'] == 9
    assert stats['suffix_accepted'] == 0
    assert stats['suffix_tested'] == 0
    # Row a's own EWMA learned the rejection: its native prefix shrank.
    assert mixer.last_native_counts()[1] == 4


def test_truncated_success_and_zero_sample_are_censored():
    from suffix_hybrid import wrap_v2
    for sampled, rejected in [(3, 0), (0, 6), (1, 0)]:
        runner, batch, mixer, native, original, invoke, _ = fixture()
        wrapped = wrap_v2._wrap_propose(runner, original, mixer, TP())
        invoke(wrapped)
        invoke(wrapped, num_sampled=torch.tensor([sampled]), num_rejected=torch.tensor([rejected]))
        assert mixer.get_stats()['native_tested'] == 0
        assert mixer.get_stats()['suffix_tested'] == 0
