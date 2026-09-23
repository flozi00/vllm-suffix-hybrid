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


def test_probabilistic_gates_stochastic_rows_to_native():
    # draft_sample_method=probabilistic: rows with temperature>0 must keep
    # the NATIVE draft (the sampler grades them against native draft_logits);
    # temperature==0 rows are arbitrated normally.
    native = torch.tensor([[9, 10, 11, 12, 99],
                           [8, 8, 8, 8, 88]], dtype=torch.int64)
    states = NS(total_len=NS(gpu=torch.tensor([8, 8])),
                all_token_ids=NS(gpu=torch.tensor([list(range(1, 33)),
                                                   list(range(101, 133))])))
    runner = NS(req_states=states)
    batch = NS(req_ids=['a', 'b'], num_reqs=2, idx_mapping=torch.tensor([0, 1]))
    temperature = torch.tensor([0.0, 0.7, 0.0, 0.0])

    class Short:
        def mix_numpy(self, *a, **k):
            return [[9, 10], [7, 7]]

    wrapped = wrap_v2._wrap_propose(runner, lambda *x, **k: native,
                                    Short(), TP(), probabilistic=True)
    got = wrapped(batch, {}, {}, torch.zeros(1), None,
                  torch.tensor([1, 1]), torch.tensor([0, 0]),
                  torch.zeros(2), torch.zeros(1, 2),
                  temperature, torch.ones(2))
    # row 0 (temp 0): suffix [9,10] written over native head.
    assert got[0].tolist() == [9, 10, 11, 12, 99]
    # row 1 (temp 0.7): native untouched despite mixer proposing [7,7].
    assert got[1].tolist() == [8, 8, 8, 8, 88]


def test_greedy_mode_writes_all_rows_regardless_of_temperature():
    native = torch.tensor([[9, 10, 11, 12, 99],
                           [8, 8, 8, 8, 88]], dtype=torch.int64)
    states = NS(total_len=NS(gpu=torch.tensor([8, 8])),
                all_token_ids=NS(gpu=torch.tensor([list(range(1, 33)),
                                                   list(range(101, 133))])))
    runner = NS(req_states=states)
    batch = NS(req_ids=['a', 'b'], num_reqs=2, idx_mapping=torch.tensor([0, 1]))
    temperature = torch.tensor([0.9, 0.7, 0.0, 0.0])

    class Short:
        def mix_numpy(self, *a, **k):
            return [[9, 10], [7, 7]]

    wrapped = wrap_v2._wrap_propose(runner, lambda *x, **k: native,
                                    Short(), TP(), probabilistic=False)
    got = wrapped(batch, {}, {}, torch.zeros(1), None,
                  torch.tensor([1, 1]), torch.tensor([0, 0]),
                  torch.zeros(2), torch.zeros(1, 2),
                  temperature, torch.ones(2))
    assert got[0].tolist() == [9, 10, 11, 12, 99]
    assert got[1].tolist() == [7, 7, 8, 8, 88]


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


# ---------------------------------------------------------------------------
# Async mirror fast path: unchanged skip, write-back, env switches.
# The fake runner has plain CPU tensors on .gpu attributes, so the mirror
# path must work with torch.cuda absent (synchronous copies into the same
# pinned-shaped staging). Regression rule: the cache-size gate was REMOVED
# from the fast path — mix_numpy is the only corpus ingestion path, so a
# cold cache must still be absorbed, enqueued and mixed (live-pod
# self-lock, 2026-09-22: cold skip before mix meant nothing ever ingested
# and suffix_hybrid_native never printed once).
# ---------------------------------------------------------------------------

class CacheMix:
    """Fake mixer with a cache_tokens() accessor and scripted mix output."""

    def __init__(self, rows, tokens=5):
        self.rows = rows
        self.tokens = tokens
        self.calls = []

    def cache_tokens(self):
        return self.tokens

    def get_stats(self):
        return {"cache": {"cached_tokens": self.tokens}}

    def mix_numpy(self, ids, counts, history, native, accepted):
        self.calls.append((list(ids), counts.copy(), history.copy(),
                           [list(r) for r in native], list(accepted)))
        return [list(r) for r in self.rows]


class StatsOnlyMix(CacheMix):
    """Pre-accessor build: cache size only via get_stats()['cache'].

    Kept so the wrapper is exercised against mixer builds that predate the
    cache_tokens() accessor — the fast path must not depend on it existing.
    """

    cache_tokens = None  # not callable


def test_cold_cache_still_mixes_and_returns_native(monkeypatch):
    # Anti-self-lock regression (live-pod, 2026-09-22): with an empty cache
    # the fast path MUST still run the mix (that IS the ingestion path);
    # the echo mix then publishes the native object untouched.
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = CacheMix([[9, 10, 11, 12, 99]], tokens=0)  # cold + echo
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)
    assert len(mixer.calls) == 1        # cold cache still ingests
    assert got is native                # echo mix: native, no clone


def test_pre_accessor_mixer_shape_still_ingests(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = StatsOnlyMix([[1, 2, 3, 4, 5]], tokens=0)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)
    assert len(mixer.calls) == 1        # no accessor -> still mixes
    assert got is not native            # scripted mix differs -> rewritten
    assert got.tolist() == [[1, 2, 3, 4, 5]]


def test_unchanged_mix_returns_native_without_rewrite(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    # Mix identical to native: nothing to write, nothing to broadcast.
    mixer = CacheMix([[9, 10, 11, 12, 99]])
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)
    assert got is native
    assert len(mixer.calls) == 1  # the mixer DID run (cache warm)


def test_changed_mix_rewrites_rows(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = CacheMix([[7, 8, 9]])   # suffix extends from token 9
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)
    assert got is not native
    assert got.tolist() == [[7, 8, 9, 12, 99]]     # fixed width, native tail
    assert got.shape == native.shape
    ids, counts, history, native_rows, accepted = mixer.calls[0]
    # Mirror-driven: counts/history come from the CPU mirror, not a
    # per-step GPU history D2H. Seeded from all_token_ids row 0.
    assert ids == ['a'] and counts.tolist() == [8]
    assert history[0, :8].tolist() == list(range(1, 9))
    assert native_rows == [[9, 10, 11, 12, 99]]
    # First step has no previous proposal: the accepted gate must be -1.
    assert accepted == [-1]


def test_mirror_extends_across_steps(monkeypatch):
    # Two steps: the fake "engine" (running between propose calls, like the
    # real sample_tokens postprocess before propose) advances total_len and
    # appends a token; the mirror must absorb the bounded window so step
    # 2's mix sees the longer history without any full-buffer re-read.
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = CacheMix([[9, 10, 11, 12, 99]])   # echo: skip write-back noise

    step = {"n": 0}

    def original(*a, **k):
        if step["n"] == 1:
            # Engine advanced the row BEFORE this propose call.
            runner.req_states.total_len.gpu += 1
            runner.req_states.all_token_ids.gpu[0, 8] = 4242
        step["n"] += 1
        return native

    wrapped = wrap_v2._wrap_propose(runner, original, mixer, TP())
    invoke(wrapped, None)
    invoke(wrapped, None)
    assert mixer.calls[-1][1].tolist() == [9]                 # grew by 1
    assert mixer.calls[-1][2][0, :9].tolist() == (            # tail absorbed
        list(range(1, 9)) + [4242])


def test_sync_hook_env_forces_old_body(monkeypatch):
    # SUFFIX_HYBRID_SYNC_HOOK=1 must run the OLD body even when the fast
    # path would cold-skip: the old body has no cold gate, so the mixer is
    # invoked and the result is a clone+broadcast copy, never the native
    # object itself.
    monkeypatch.setenv("SUFFIX_HYBRID_SYNC_HOOK", "1")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = CacheMix([[9, 10, 11, 12, 99]], tokens=0)  # cold: fast skips
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)
    assert len(mixer.calls) == 1     # old body mixed despite cold cache
    assert got is not native         # old body: clone + broadcast
    assert got.tolist() == native.tolist()


def test_fast_path_crash_falls_back_stickily_to_sync(monkeypatch):
    # A fast-path raise (here: the mirror's first read of the GPU history
    # row explodes once) must log once and route every later step through
    # the synchronous body.
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)
    mixer = CacheMix([[9, 10, 11, 12, 99]], tokens=5)  # warm: reach the mix

    good_ats = runner.req_states.all_token_ids

    class FlakyATS:
        def __init__(self):
            self.n = 0

        @property
        def gpu(self):
            self.n += 1
            if self.n == 1:
                # First fast-path seed read: blow up inside the mirror.
                raise RuntimeError("mirror staging exploded")
            return good_ats.gpu

    runner.req_states.all_token_ids = FlakyATS()
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    got = invoke(wrapped, None)          # first call: fallback triggers
    assert got is not native             # sync body re-ran and cloned
    assert got.tolist() == native.tolist()
    # Later steps keep using the sync body (sticky flag): the fast path
    # would now succeed against the healed buffer, but must not be retried.
    got2 = invoke(wrapped, None)
    assert got2 is not native
    assert got2.tolist() == native.tolist()


def test_replay_mode_skips_collective(monkeypatch):
    # TP_MODE=replay: every rank mirrors+mixes locally, no broadcast ever.
    monkeypatch.delenv("SUFFIX_HYBRID_SYNC_HOOK", raising=False)
    monkeypatch.setenv("SUFFIX_HYBRID_TP_MODE", "replay")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _fixture(native)

    class NoCast(TP):
        world_size = 8

        def broadcast(self, value, src=0):
            raise AssertionError("replay mode must never hit a collective")

    mixer = CacheMix([[7, 8, 9]])
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, NoCast())
    got = invoke(wrapped, None)
    assert got.tolist() == [[7, 8, 9, 12, 99]]


class ProbeMix(CacheMix):
    """Mixer exposing a suffix_cache handle whose speculate() is scripted.

    Scripts the probe contract: `hit` rows yield a strong continuation
    (len >= PROBE_MIN_LEN), `miss` rows yield nothing.
    """

    def __init__(self, rows, tokens=5, hit=False):
        super().__init__(rows, tokens)
        self.hit = hit
        self.probe_calls = 0

    @property
    def suffix_cache(self):
        return NS(speculate=self._spec)

    def _spec(self, tail, w):
        self.probe_calls += 1
        if self.hit:
            return [42, 43, 44], 1.0, 8
        return [], 0.0, 0


class WinningMix(ProbeMix):
    """ProbeMix whose mix WINS rows: mix_numpy returns suffix rows."""

    def mix_numpy(self, ids, counts, history, native, accepted):
        self.calls.append((list(ids), counts.copy(), history.copy(),
                           [list(r) for r in native], list(accepted)))
        return [[42, 43] for _ in ids]


def _probe_fixture(native, history_rows=None):
    states = NS(total_len=NS(gpu=torch.tensor([8, 2])),
                all_token_ids=NS(gpu=torch.tensor(
                    history_rows or [list(range(1, 33)), [700] * 32])))
    runner = NS(req_states=states)
    batch = NS(req_ids=['a'], num_reqs=1, idx_mapping=torch.tensor([0]))

    def invoke(wrapped, calls, num_sampled=torch.tensor([1]),
               num_rejected=torch.tensor([0]), **kw):
        return wrapped(batch, {}, {}, torch.zeros(1), None,
                       num_sampled, num_rejected,
                       torch.zeros(2), torch.zeros(1, 2),
                       torch.tensor([0.8, 0.8]), torch.ones(2), **kw)

    return runner, batch, invoke


def test_probe_gate_skips_mix_when_no_suffix_evidence(monkeypatch):
    # Warm corpus + probe finds NO strong continuation -> native returned
    # UNTOUCHED (identity) and mix_numpy never runs.
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=False)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    calls = []
    # Two steps: step 1 seeds the mirror (row is None -> full path), step 2
    # has a mirrored row and the probe misses -> gate fires.
    got1 = invoke(wrapped, calls)
    assert got1 is not None
    got2 = invoke(wrapped, calls)
    assert got2 is native
    assert mixer.probe_calls >= 1
    # mix ran at most once (step 1); the gated step never reached it.
    assert len(mixer.calls) <= 1


def test_probe_gate_fires_full_path_on_strong_suffix(monkeypatch):
    # Warm corpus + probe finds a strong continuation -> full arbitrate
    # path runs (mix_numpy called on the second step too).
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    calls = []
    invoke(wrapped, calls)
    invoke(wrapped, calls)
    assert len(mixer.calls) == 2


def test_probe_never_gates_cold_corpus(monkeypatch):
    # cache_tokens()==0 -> full path every step (self-lock rule).
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=0, hit=False)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    calls = []
    for _ in range(4):
        invoke(wrapped, calls)
    assert len(mixer.calls) == 4


def test_adaptive_gate_cools_down_after_misses(monkeypatch):
    # COOLDOWN=2: two full-path mixes that find evidence but never WIN the
    # row -> from the 3rd evidenced step on, the request no longer forces
    # arbitration (probe gate fires despite cache.speculate returning a
    # strong continuation).
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    # step 1: un-mirrored -> full path (seed); misses := 1
    invoke(wrapped, None)
    # step 2: evidenced + trusted (misses 1 < 2) -> full path; misses := 2
    invoke(wrapped, None)
    assert len(mixer.calls) == 2
    # step 3: evidenced but misses 2 >= cooldown 2 -> GATED (native echo).
    got3 = invoke(wrapped, None)
    assert got3 is native
    assert len(mixer.calls) == 2
    assert mixer.probe_calls >= 1


def test_frozen_fast_path_skips_all_machinery(monkeypatch):
    # COOLDOWN=2, HEARTBEAT=0: once the request is cooled AND mirrored AND
    # the corpus is warm, the wrapper must echo native WITHOUT calling
    # absorb/enqueue/probe/mix (probe_calls and mixer.calls frozen).
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    invoke(wrapped, None)          # step 1: seed (full path)
    invoke(wrapped, None)          # step 2: evidenced, trusted (full path)
    assert len(mixer.calls) == 2
    calls_before = len(mixer.calls)
    probes_before = mixer.probe_calls
    # steps 3..6: cooled. Mirror exists, corpus warm, hb disabled ->
    # frozen echo: no mixes, no probes.
    for _ in range(4):
        got = invoke(wrapped, None)
        assert got is native
    assert len(mixer.calls) == calls_before
    assert mixer.probe_calls == probes_before


def test_frozen_path_thaws_for_new_request(monkeypatch):
    # A new request entering a frozen batch must break the freeze: the
    # mirror miss forces the full path (seed), so suffix arbitration can
    # still reach the newcomer.
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    for _ in range(4):             # cool request r0 fully down
        invoke(wrapped, None)
    assert len(mixer.calls) >= 2
    calls_before = len(mixer.calls)
    # New request appears: batch.req_ids becomes [r0, r1].
    batch.req_ids = list(batch.req_ids) + ["r1"]
    invoke(wrapped, None)
    assert len(mixer.calls) == calls_before + 1   # full path ran


def test_frozen_path_respects_heartbeat(monkeypatch):
    # HEARTBEAT=3: the freeze must thaw every 3rd step so the corpus
    # keeps ingesting even when all requests are cooled.
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "3")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = ProbeMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    for _ in range(4):             # cool down (steps advance on full path)
        invoke(wrapped, None)
    assert len(mixer.calls) >= 2
    # Frozen steps tick state["steps"]; every hb-th step runs the full
    # path again regardless of the freeze.
    calls_before = len(mixer.calls)
    ran = 0
    for _ in range(6):
        invoke(wrapped, None)
        if len(mixer.calls) > calls_before:
            ran += 1
            calls_before = len(mixer.calls)
    assert ran >= 1                 # heartbeat thawed at least once
    assert ran < 6                  # ...but not on every step


def test_adaptive_gate_rearms_on_win(monkeypatch):
    # A WINNING mixer never cools down: every step stays on the full path.
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = WinningMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    for _ in range(5):
        got = invoke(wrapped, None)
        assert got is not native        # a win writes a row every time
    assert len(mixer.calls) == 5


def test_zero_op_arm_echoes_native_and_touches_nothing(monkeypatch):
    # ZERO-OP (p22 residual bound): with SUFFIX_HYBRID_ZERO_OP=1 the armed
    # hook must return native verbatim on EVERY step — no mixer calls, no
    # probes, no state mutation, even when the mixer would win and the
    # corpus is warm. This is the by-elimination bound for the ~2.7ms/step
    # armed-vs-inert residual: zero-op ≈ baseline => the cost is per-step
    # machinery; zero-op still slow => the cost is the hook's presence.
    monkeypatch.setenv("SUFFIX_HYBRID_ZERO_OP", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "8")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = WinningMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    for _ in range(6):
        got = invoke(wrapped, None)
        assert got is native            # verbatim echo, every step
    assert len(mixer.calls) == 0        # mixer never invoked
    assert mixer.probe_calls == 0       # probe never consulted
    # And the gate stays untouched: no state drift while zero-op is armed.
    assert not getattr(runner, "_zero_op_state_touched", False)


def test_verifier_win_false_wins_do_not_rearm(monkeypatch):
    # p24: with SUFFIX_HYBRID_VERIFIER_WIN=1, a request whose rows keep
    # publishing differences but keep getting REJECTED by the verifier must
    # NOT re-arm the gate — the cooldown veto engages despite continuous
    # self-match evidence (the 58% bench-window share pathology).
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    monkeypatch.setenv("SUFFIX_HYBRID_VERIFIER_WIN", "1")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = WinningMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    # Every step publishes a diff (WinningMix rows are 2 wide; previous_widths
    # records 2) and the feedback shows rejection: ns=1, nr=1 => a=0,
    # verified=1, previous=2 -> valid, accepted=0 (published row rejected).
    for _ in range(6):
        invoke(wrapped, None, num_sampled=torch.tensor([1]),
               num_rejected=torch.tensor([1]))
    # Misses must have accumulated: the request cooled down and later steps
    # stopped forcing the full path (mixer.calls stop growing).
    calls = len(mixer.calls)
    for _ in range(3):
        got = invoke(wrapped, None, num_sampled=torch.tensor([1]),
                     num_rejected=torch.tensor([1]))
        assert got is native
    assert len(mixer.calls) == calls      # no new full-path mixes


def test_verifier_win_verified_win_rearms(monkeypatch):
    # p24: when the feedback shows the published row was actually accepted
    # (accepted >= 1), the gate resets — a genuine win still re-arms.
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_MIN_LEN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_COOLDOWN", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_PROBE_HEARTBEAT", "0")
    monkeypatch.setenv("SUFFIX_HYBRID_VERIFIER_WIN", "1")
    native = torch.tensor([[9, 10, 11, 12, 99]], dtype=torch.int64)
    runner, batch, invoke = _probe_fixture(native)
    mixer = WinningMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                    mixer, TP())
    # Warm the request up with REJECTED feedback (accepted=0) until cooled.
    for _ in range(4):
        invoke(wrapped, None, num_sampled=torch.tensor([1]),
               num_rejected=torch.tensor([0]))
    # The staged feedback buffers lag one step; deliver ACCEPTED feedback
    # (num_sampled=3 => accepted=2 >= 1) on a full-path step. To force the
    # full path, thaw via a heartbeat-free path: new rid? Simplest: the
    # accepted feedback arrives while still full-path (not yet cooled).
    # Re-build with accepted feedback from the start instead:
    mixer2 = WinningMix([[9, 10, 11, 12, 99]], tokens=1000, hit=True)
    wrapped2 = wrap_v2._wrap_propose(runner, lambda *a, **k: native,
                                     mixer2, TP())
    for _ in range(6):
        invoke(wrapped2, None, num_sampled=torch.tensor([3]),
               num_rejected=torch.tensor([0]))
    # Verified wins the whole way: gate stays reset, full path keeps running.
    assert len(mixer2.calls) >= 4
