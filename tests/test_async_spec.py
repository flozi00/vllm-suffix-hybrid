# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_HYBRID_ASYNC=1: async-scheduling-compatible suffix/hybrid drafting.

CPU-only. The Rust V2SuffixProposer is real; CUDA buffers are CPU tensors
and CUDA events are fakes that check the lag-1 wait discipline. Pins:
  * gate off: install_v2 + sched_sync behave exactly as before
  * gate on: propose makes no host sync (only the previous step's event is
    synchronized; staging reads only; every staging copy is non_blocking)
  * draft buffer contents for hit / miss / misaligned / offset rows
  * the SchedulerOutput trim (copy, never mutate; dummy runs untouched)
  * hybrid row select + temperature gate + broadcast discipline
  * sched_sync no longer forces sync when the gate is on
"""
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from suffix_hybrid import async_spec, sched_sync, wrap_v2
from suffix_hybrid._native import V2SuffixProposer

K = 4
LC = 2 * K + 1
ROWS = 4
COLS = 64
S = list(range(1000, 1040))          # corpus passage (40 tokens)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for var in ("SUFFIX_HYBRID_ASYNC", "SUFFIX_HYBRID_SUFFIX_ONLY",
                "SUFFIX_HYBRID_ASYNC_TRIM", "SUFFIX_HYBRID_ASYNC_HYB_MIN",
                "SUFFIX_HYBRID_TP_MODE", "SUFFIX_HYBRID_SUFFIX_MIN",
                "SUFFIX_HYBRID_LOG_INTERVAL", "SUFFIX_HYBRID_SYNC_SCHED",
                "SUFFIX_HYBRID_D2H_PIPE", "SUFFIX_HYBRID_UNIFORM_K",
                "SUFFIX_HYBRID_EWMA_WIDTH", "SUFFIX_HYBRID_TRACE",
                "SUFFIX_HYBRID_TRACE_VERIFY", "SUFFIX_HYBRID_TRACE_VERIFY2"):
        monkeypatch.delenv(var, raising=False)


class Ev:
    """Fake CUDA event: synchronize() is legal only on an event recorded
    by an EARLIER propose call (lag-1 discipline, invariant I1)."""
    log = []

    def __init__(self):
        self.recorded_at = None

    def record(self, stream=None):
        self.recorded_at = Ev.call

    def synchronize(self):
        assert self.recorded_at is not None, "sync on a never-recorded event"
        assert self.recorded_at < Ev.call, "sync on THIS step's event"
        Ev.log.append((Ev.call, self.recorded_at))


Ev.call = 0


class World:
    """Fake runner state: UVA token rows + totals the 'GPU' advances."""

    def __init__(self):
        self.ats = np.zeros((ROWS, COLS), dtype=np.int32)
        self.tot = torch.zeros(ROWS, dtype=torch.int32)
        self.runner = NS(req_states=NS(
            total_len=NS(gpu=self.tot),
            all_token_ids=NS(_uva_buf=NS(np=self.ats),
                             gpu=torch.from_numpy(self.ats)),
            req_id_to_index={}))

    def put(self, row, toks):
        t = int(self.tot[row])
        self.ats[row, t:t + len(toks)] = toks
        self.tot[row] = t + len(toks)


class Spec:
    def __init__(self, native=None):
        self.draft_tokens = torch.zeros((ROWS, K), dtype=torch.int64)
        self.max_model_len = COLS
        self.num_speculative_steps = K
        self.draft_logits = None
        self._native = native

    def propose(self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature,
                seeds, dp_sync=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        n = int(input_batch.num_reqs)
        out = self.draft_tokens[:n]
        out.copy_(self._native[:n])
        return out


def batch(rows):
    ids = [r for r, _ in rows]
    idx = np.array([i for _, i in rows], dtype=np.int64)
    return NS(req_ids=ids, num_reqs=len(ids), idx_mapping_np=idx,
              idx_mapping=torch.from_numpy(idx))


def call(d, b, temperature=None, **kw):
    Ev.call += 1
    n = b.num_reqs
    t = temperature if temperature is not None else torch.zeros(ROWS)
    return d.propose(b, {}, {}, torch.zeros(1), None, torch.ones(n),
                     torch.zeros(n), torch.zeros(ROWS, 1), torch.zeros(1, 2),
                     t, torch.ones(ROWS), **kw)


def drafter(world, native=None, group=None, probabilistic=False,
            hyb_min=None):
    proposer = V2SuffixProposer(LC, COLS, 1, False, False)
    proposer.suffix_cache.add_sequence(S)
    spec = Spec(native)
    nat = None
    if native is not None:
        nat = type(spec).propose.__get__(spec, type(spec))
    return async_spec.AsyncSuffixDrafter(
        world.runner, spec, proposer, K, native=nat, group=group,
        probabilistic=probabilistic, hyb_min=hyb_min, make_event=Ev)


# ---------------------------------------------------------------------------
# device resolve + host widths
# ---------------------------------------------------------------------------

def test_hit_miss_offset_and_misaligned_rows():
    w = World()
    w.put(2, S[:10])                      # 'a' follows the corpus
    w.put(3, [500, 501, 502])             # 'b' never matches
    d = drafter(w)
    b = batch([("a", 2), ("b", 3)])

    # step 1: no lag snapshot yet -> no candidates, zero drafts, no widths
    out = call(d, b)
    assert out.tolist() == [[0] * K, [0] * K]
    assert d.widths == {}

    # GPU step: 'a' samples S[10] (o=1), 'b' samples 600
    w.put(2, [S[10]])
    w.put(3, [600])
    out = call(d, b)
    assert out[0].tolist() == S[11:11 + K]          # C[o:o+K], aligned
    assert out[1].tolist() == [0] * K               # miss row: filler
    assert d.widths == {"a": K}                     # 'b' absent -> width 0

    # GPU step: all K drafts accepted + bonus -> o = K+1 = 5
    w.put(2, S[11:11 + K + 1])
    w.put(3, [601])
    out = call(d, b)
    # lag-1 anchor p=11, C = S[11:20] (len 9), o=5 -> C[5:9] = S[16:20]
    assert out[0].tolist() == S[16:20]

    # GPU step: target diverges (accepted 0, sampled 7) -> misaligned
    w.put(2, [7])
    w.put(3, [602])
    out = call(d, b)
    assert out[0].tolist() == [0] * K               # C[0] != 7 -> filler
    assert d.widths["a"] == K       # host only knows the upper bound (waste)


def test_short_candidate_real_positions_only():
    # Candidate shorter than o+K: positions past len(C) are filler.
    w = World()
    w.put(2, S[30:37])               # passage tail: few tokens left in corpus
    d = drafter(w)
    b = batch([("a", 2)])
    call(d, b)
    w.put(2, [S[37]])
    out = call(d, b)
    # anchor 7 tokens S[30:37]; C = S[37:40] (corpus ends), o=1 -> S[38:40]
    assert out[0].tolist() == [S[38], S[39], 0, 0]
    assert d.widths == {"a": 2}


def test_row_reindexed_or_new_gets_no_candidate():
    w = World()
    w.put(2, S[:10])
    w.put(1, S[:10])
    d = drafter(w)
    call(d, batch([("a", 2)]))
    w.put(2, [S[10]])
    w.put(1, [S[10]])
    # 'a' moved to row 1 (preempt/resume), 'c' is new: neither may draft.
    out = call(d, batch([("a", 1), ("c", 2)]))
    assert out.tolist() == [[0] * K, [0] * K]
    assert d.widths == {}


def test_dummy_and_profile_zero_and_skip_state():
    w = World()
    w.put(2, S[:10])
    d = drafter(w)
    d.spec.draft_tokens.fill_(9)
    for kw in ({"dummy_run": True}, {"is_profile": True},
               {"skip_attn_for_dummy_run": True}):
        out = call(d, batch([("a", 2)]), **kw)
        assert out.tolist() == [[0] * K]
    assert d.pending is None and d.step == 0


def test_exception_degrades_retracts_widths_and_reseeds(capsys):
    w = World()
    w.put(2, S[:10])
    d = drafter(w)
    b = batch([("a", 2)])
    call(d, b)
    w.put(2, [S[10]])
    call(d, b)
    assert d.widths == {"a": K}

    class Boom:
        def propose_suffix_only(self, *a):
            raise RuntimeError("rust down")

    d.proposer = Boom()
    out = call(d, b)
    assert out.tolist() == [[0] * K]
    assert d.widths == {} and d.pending is None
    assert d.stats["degraded"] == 1
    assert "async passthrough" in capsys.readouterr().err
    # I2: the degraded step still recorded its event (its snapshot copy is
    # in flight), and the next step waits on it before reusing staging.
    Ev.log.clear()
    call(d, b)
    assert [c - r for c, r in Ev.log] == [1]


# ---------------------------------------------------------------------------
# no host sync in the propose path (I1) + staging discipline (I2)
# ---------------------------------------------------------------------------

def test_no_host_sync_in_propose(monkeypatch):
    w = World()
    w.put(2, S[:10])
    w.put(3, [500, 501, 502])
    d = drafter(w)
    b = batch([("a", 2), ("b", 3)])
    staging = {t.untyped_storage().data_ptr() for t in d.snap + d.pack}

    def staged(t):
        return t.untyped_storage().data_ptr() in staging

    def boom(name):
        def f(*a, **k):
            raise AssertionError(f"host sync: {name}")
        return f

    monkeypatch.setattr(torch.cuda, "synchronize", boom("cuda.synchronize"))
    monkeypatch.setattr(torch.cuda.Stream, "synchronize",
                        boom("Stream.synchronize"))
    monkeypatch.setattr(torch.cuda.Event, "synchronize",
                        boom("cuda.Event.synchronize"))
    for name in ("item", "tolist", "cpu", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom(f"Tensor.{name}"))
    real_numpy, real_copy = torch.Tensor.numpy, torch.Tensor.copy_
    copies = []

    def numpy_(self, *a, **k):   # host reads only from pinned staging
        assert staged(self), "numpy() on a device tensor"
        return real_numpy(self, *a, **k)

    def copy_(self, src, non_blocking=False):
        if staged(self) or staged(src):
            assert non_blocking, "blocking copy through staging"
            copies.append("staging")
        return real_copy(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "numpy", numpy_)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    Ev.log.clear()
    for tok in ([S[10]], S[11:16], [7]):
        call(d, b)
        w.ats[2, int(real_numpy(w.tot)[2]):][:len(tok)] = tok
        w.tot[2:3].add_(len(tok))
    call(d, b)
    monkeypatch.undo()
    # every propose after the first waited exactly once, on the previous
    # call's event (Ev asserts recorded_at < current call)
    assert [c - r for c, r in Ev.log] == [1, 1, 1]
    # per step: snapshot D2H + counter D2H + candidate H2D
    assert len(copies) == 3 * 4


def test_staging_slots_alternate_by_step_parity():
    w = World()
    w.put(2, S[:10])
    d = drafter(w)
    b = batch([("a", 2)])
    slots = []
    for _ in range(4):
        call(d, b)
        slots.append(d.pending[0])
        w.put(2, [S[int(w.tot[2])]])
    assert slots == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# SchedulerOutput trim
# ---------------------------------------------------------------------------

def _so(spec, num, structured=False):
    return NS(scheduled_spec_decode_tokens=spec, num_scheduled_tokens=num,
              total_num_scheduled_tokens=sum(num.values()),
              has_structured_output_requests=structured)


def test_trim_copies_and_cuts_to_host_widths():
    d = drafter(World())
    d.widths = {"a": 2, "c": 9}
    so = _so({"a": [-1] * K, "b": [-1] * K, "c": [-1] * K},
             {"a": 5, "b": 5, "c": 5, "p": 7})
    got = d.trim(so)
    assert got is not so
    assert got.scheduled_spec_decode_tokens == {"a": [-1, -1],
                                                "c": [-1] * K}
    assert got.num_scheduled_tokens == {"a": 3, "b": 1, "c": 5, "p": 7}
    assert got.total_num_scheduled_tokens == 16
    # the scheduler's object is untouched (UniProc shares it)
    assert so.scheduled_spec_decode_tokens["b"] == [-1] * K
    assert so.num_scheduled_tokens["a"] == 5
    assert so.total_num_scheduled_tokens == 22
    assert d.stats["trimmed"] == 6


def test_trim_noop_cases():
    d = drafter(World())
    d.widths = {"a": K}
    full = _so({"a": [-1] * K}, {"a": 5})
    assert d.trim(full) is full
    none = _so({}, {"a": 1})
    assert d.trim(none) is none
    d.widths = {}
    structured = _so({"a": [-1] * K}, {"a": 5}, structured=True)
    assert d.trim(structured) is structured


# ---------------------------------------------------------------------------
# hybrid mode
# ---------------------------------------------------------------------------

def test_hybrid_selects_full_width_suffix_rows_else_native():
    w = World()
    w.put(2, S[:10])
    w.put(3, [500, 501, 502])
    native = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]] + [[0] * K] * 2)
    d = drafter(w, native=native)
    b = batch([("a", 2), ("b", 3)])
    out = call(d, b)
    assert out.tolist() == native[:2].tolist()       # no lag yet: native
    w.put(2, [S[10]])
    w.put(3, [600])
    out = call(d, b)
    assert out[0].tolist() == S[11:11 + K]            # full suffix row
    assert out[1].tolist() == [5, 6, 7, 8]            # miss: native


def test_hybrid_probabilistic_keeps_stochastic_rows_native():
    w = World()
    w.put(2, S[:10])
    w.put(3, S[:10])
    native = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]] + [[0] * K] * 2)
    d = drafter(w, native=native, probabilistic=True)
    b = batch([("a", 2), ("b", 3)])
    temp = torch.tensor([0.0, 0.0, 0.0, 0.7])        # row 3 ('b') stochastic
    call(d, b, temperature=temp)
    w.put(2, [S[10]])
    w.put(3, [S[10]])
    out = call(d, b, temperature=temp)
    assert out[0].tolist() == S[11:11 + K]
    assert out[1].tolist() == [5, 6, 7, 8]


def test_hybrid_short_suffix_below_threshold_stays_native():
    w = World()
    w.put(2, S[30:37])
    native = torch.tensor([[1, 2, 3, 4]] + [[0] * K] * 3)
    d = drafter(w, native=native)                    # hyb_min default K
    b = batch([("a", 2)])
    call(d, b)
    w.put(2, [S[37]])
    assert call(d, b)[0].tolist() == [1, 2, 3, 4]    # only 2 real drafts
    d2 = drafter(World(), native=native, hyb_min=2)
    assert d2.hyb_min == 2


def test_hybrid_broadcast_mode_every_rank_broadcasts():
    native = torch.tensor([[1, 2, 3, 4]] + [[0] * K] * 3)

    class Group:
        world_size = 2

        def __init__(self, rank):
            self.rank_in_group = rank
            self.calls = []

        def broadcast(self, t, src=0):
            self.calls.append((src, t.tolist()))
            return t

    for rank in (0, 1):
        w = World()
        w.put(2, S[:10])
        g = Group(rank)
        d = drafter(w, native=native, group=g)

        class NoRust:
            def propose_suffix_only(self, *a):
                raise AssertionError("non-owner rank must not look up")

        if rank:
            d.proposer = NoRust()
        b = batch([("a", 2)])
        call(d, b)
        w.put(2, [S[10]])
        call(d, b)
        call(d, b, dummy_run=True)          # capture: no collective
        assert len(g.calls) == 2 and all(src == 0 for src, _ in g.calls)


# ---------------------------------------------------------------------------
# install path: gate off = unchanged, gate on = async drafter + trim
# ---------------------------------------------------------------------------

def _fake_vllm(monkeypatch):
    class DraftModelSpeculator:
        pass

    class MTPSpeculator(DraftModelSpeculator, Spec):
        def __init__(self):
            Spec.__init__(self, torch.zeros((ROWS, K), dtype=torch.int64))

    class GPUModelRunner:
        def __init__(self):
            w = World()
            self.req_states = w.runner.req_states
            self.speculator = MTPSpeculator()
            self.draft_tokens_handler = NS()
            self.rejection_sampler = None
            self.vllm_config = NS(
                scheduler_config=NS(async_scheduling=True),
                parallel_config=NS(pipeline_parallel_size=1))
            self.seen = []

        def load_model(self):
            return "loaded"

        def execute_model(self, scheduler_output, *a, **kw):
            self.seen.append(scheduler_output)
            return None

    tp = NS(world_size=1, rank_in_group=0, broadcast=lambda t, src=0: t)
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
        "vllm.v1.worker.gpu.model_runner": types.ModuleType("mr"),
        "vllm.v1.worker.gpu.spec_decode": types.ModuleType("sd"),
        "vllm.v1.worker.gpu.spec_decode.speculator": types.ModuleType("sp"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.parallel_state": types.ModuleType("ps"),
    }
    mods["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = GPUModelRunner
    mods["vllm.v1.worker.gpu.spec_decode.speculator"]\
        .DraftModelSpeculator = DraftModelSpeculator
    mods["vllm.distributed.parallel_state"].get_tp_group = lambda: tp
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("SUFFIX_HYBRID_WRAP", "1")
    assert wrap_v2.install_v2() is True
    return GPUModelRunner


def test_gate_off_install_is_the_sync_suffix_only_wrap(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_SUFFIX_ONLY", "1")
    runner = _fake_vllm(monkeypatch)()
    original_exec = runner.execute_model
    assert runner.load_model() == "loaded"
    p = runner.speculator.propose
    assert getattr(p, "_suffix_proposer", None) is not None   # sync wrap
    assert not hasattr(p, "_suffix_async")
    assert "get_draft_tokens" in vars(runner.draft_tokens_handler)
    assert runner.execute_model == original_exec              # no trim hook


def test_gate_on_install_async_suffix_only_with_trim(monkeypatch, capsys):
    monkeypatch.setenv("SUFFIX_HYBRID_SUFFIX_ONLY", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_ASYNC", "1")
    runner = _fake_vllm(monkeypatch)()
    runner.load_model()
    d = runner.speculator.propose._suffix_async
    assert d.native is None and d.group is None
    # warm-start seam still finds the (async) proposer's cache
    assert runner.speculator.propose._suffix_proposer is d.proposer
    assert hasattr(d.proposer, "suffix_cache")
    assert "get_draft_tokens" not in vars(runner.draft_tokens_handler)
    assert "ASYNC installed mode=suffix-only" in capsys.readouterr().err
    d.widths = {"a": 1}
    so = _so({"a": [-1] * K}, {"a": 5})
    runner.execute_model(so)
    assert runner.seen[-1].num_scheduled_tokens == {"a": 2}
    runner.execute_model(so, dummy_run=True)                  # untouched
    assert runner.seen[-1] is so


def test_gate_on_hybrid_and_trim_off(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_ASYNC", "1")
    runner = _fake_vllm(monkeypatch)()
    exec_before = runner.execute_model
    runner.load_model()
    d = runner.speculator.propose._suffix_async
    assert d.native is not None                     # hybrid: native first
    assert runner.execute_model == exec_before      # hybrid never trims

    monkeypatch.setenv("SUFFIX_HYBRID_SUFFIX_ONLY", "1")
    monkeypatch.setenv("SUFFIX_HYBRID_ASYNC_TRIM", "0")
    runner2 = _fake_vllm(monkeypatch)()
    exec2 = runner2.execute_model
    runner2.load_model()
    assert runner2.execute_model == exec2


# ---------------------------------------------------------------------------
# sched_sync: the async gate no longer forces sync
# ---------------------------------------------------------------------------

def _engine_args_module():
    mod = types.ModuleType("vllm.engine.arg_utils")

    class EngineArgs:
        async_scheduling = None
        speculative_config = None

        def create_engine_config(self):
            # config/vllm.py:1438-1471: None resolves to True unless
            # disable_padded_drafter_batch=True turns async off.
            a = self.async_scheduling
            if a is None:
                a = not (self.speculative_config or {}).get(
                    "disable_padded_drafter_batch", False)
            return NS(scheduler_config=NS(async_scheduling=a),
                      spec=self.speculative_config)

    mod.EngineArgs = EngineArgs
    return mod


def test_sched_sync_gate_off_still_forces_sync(capsys):
    mod = _engine_args_module()
    sched_sync._patch(mod)
    args = mod.EngineArgs()
    args.speculative_config = {"method": "mtp"}
    assert args.create_engine_config().scheduler_config.async_scheduling \
        is False
    assert "FORCE APPLIED" in capsys.readouterr().err


def test_sched_sync_gate_on_keeps_async_and_drops_padded_flag(monkeypatch,
                                                              capsys):
    monkeypatch.setenv("SUFFIX_HYBRID_ASYNC", "1")
    mod = _engine_args_module()
    sched_sync._patch(mod)
    args = mod.EngineArgs()
    spec = {"method": "mtp", "disable_padded_drafter_batch": True}
    args.speculative_config = spec
    cfg = args.create_engine_config()
    assert cfg.scheduler_config.async_scheduling is True
    assert cfg.spec["disable_padded_drafter_batch"] is False
    assert spec["disable_padded_drafter_batch"] is True   # caller's dict
    err = capsys.readouterr().err
    assert "sync NOT forced" in err and "FORCE APPLIED" not in err
    # an explicit operator --no-async-scheduling is respected
    args2 = mod.EngineArgs()
    args2.async_scheduling = False
    assert args2.create_engine_config().scheduler_config.async_scheduling \
        is False
