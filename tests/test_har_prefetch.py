import os
import re
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from suffix_hybrid import har_prefetch as hp
from suffix_hybrid import host_ar as H

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ rows / spans
def test_rows_are_aligned_inside_their_span_chunked_and_cut_at_the_budget():
    spans = [(1000, 300_000), (5 << 20, 100), (7 << 20, 64 << 10)]  # 2nd: below MIN_BYTES
    rows = hp.rows_for(spans, budget=320_000, chunk=128 << 10)
    assert all(a % 16 == 0 and n % 16 == 0 and 0 < n <= 128 << 10 for a, n in rows)
    assert all(any(s <= a and a + n <= s + sz for s, sz in spans) for a, n in rows)
    assert rows[0][0] == 1008 and sum(n for a, n in rows) <= 320_000
    first = [r for r in rows if r[0] < 5 << 20]
    assert sum(n for _, n in first) == (1000 + 300_000) // 16 * 16 - 1008  # whole span, shrunk
    assert [a for a, _ in rows] == sorted(a for a, _ in rows)  # use order kept
    assert hp.rows_for(spans, budget=0) == []


def test_tensor_span_covers_views_and_stays_inside_the_storage():
    base = torch.zeros(64, 128, dtype=torch.float32)
    assert hp.tensor_span(base) == (base.data_ptr(), base.numel() * 4)
    t = base.t()  # transposed view: same bytes
    assert hp.tensor_span(t) == (base.data_ptr(), base.numel() * 4)
    s = base[8:16, :64]  # strided slice: first to last element it touches
    assert hp.tensor_span(s) == (s.data_ptr(), ((8 - 1) * 128 + 64) * 4)
    assert hp.tensor_span(torch.empty(0, dtype=torch.float32)) is None


def test_table_is_int64_padded_with_skipped_rows_under_bf16_default_dtype():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # vLLM loads models under this (silicon pitfall)
    try:
        rows, cap = hp.make_table([(4096, 1 << 20), (1 << 30, 4096)], "cpu")
        assert rows.dtype == torch.int64 and rows.shape == (cap, 2) and cap == 16
        assert rows[0].tolist() == [4096, 1 << 20] and rows[1].tolist() == [1 << 30, 4096]
        assert rows[2:].abs().sum() == 0
        assert hp.make_table(list(zip(range(16, 16 * 18, 16), [16] * 17)), "cpu")[1] == 32
    finally:
        torch.set_default_dtype(old)


def test_selftest_tensors_have_explicit_dtypes_under_bf16_default(monkeypatch):
    seen = []

    class FakeHar:
        device, rank, q8_min, l2pf, cpu_group = torch.device("cpu"), 0, 256 << 10, None, None

        def all_reduce(self, x, pf=None, **kw):
            seen.append((x.dtype, None if pf is None else (pf[0].dtype, pf[1]), kw))
            return x.clone()

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, **kw: None)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        assert hp.selftest(FakeHar(), mb=1) == "two-shot/one-shot/int8"
    finally:
        torch.set_default_dtype(old)
    assert {s[0] for s in seen} == {torch.bfloat16}
    assert [s[1] is None for s in seen] == [True, False] * 3
    assert all(s[1][0] == torch.int64 and s[1][1] % 16 == 0 for s in seen[1::2])
    assert seen[-1][2] == {"oneshot": False, "q8": True}


def test_selftest_refuses_a_mismatch(monkeypatch):
    class Bad:
        device, rank, q8_min, l2pf, cpu_group = torch.device("cpu"), 0, 1 << 62, None, None

        def all_reduce(self, x, pf=None, **kw):
            return x + (pf is not None)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, **kw: None)
    with pytest.raises(RuntimeError, match="not bit-identical"):
        hp.selftest(Bad(), mb=1)


# ------------------------------------------------------------------ recorder
class Attn(nn.Module):
    """MLA-shaped: q_b_proj, absorbed W_UK_T / W_UV attributes, o_proj."""

    def __init__(self):
        super().__init__()
        self.fused_qkv_a_proj = nn.Linear(64, 48, bias=False)
        self.q_b_proj = nn.Linear(32, 64, bias=False)
        self.o_proj = nn.Linear(64, 64, bias=False)
        self.W_UK_T = torch.randn(4, 16, 32, dtype=torch.float32)
        self.W_UV = torch.randn(4, 32, 16, dtype=torch.float32)

    def forward(self, x):
        q = self.q_b_proj(self.fused_qkv_a_proj(x)[:, :32])
        return self.o_proj(q)


class Layer(nn.Module):
    def __init__(self, har):
        super().__init__()
        self.har, self.attn = har, Attn()
        self.gate = nn.Linear(64, 16, bias=False)
        self.up = nn.Linear(64, 64, bias=False)

    def forward(self, x, first):
        if not first:
            x = self.har(x)
        x = self.har(self.attn(x))
        self.gate(x)
        return self.up(x)


class Root(nn.Module):
    def __init__(self, har, n=3):
        super().__init__()
        self.layers = nn.ModuleList(Layer(har) for _ in range(n))
        self.order = None

    def forward(self, input_ids, positions):
        x = torch.zeros(positions.shape[-1], 64, dtype=torch.float32)
        for i in (self.order or range(len(self.layers))):
            x = self.layers[i](x, i == 0)
        return self.layers[0].har(x)


@pytest.fixture
def rec(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(hp, "watched", lambda m: isinstance(m, nn.Linear))
    monkeypatch.setattr(hp, "MIN_BYTES", 16)
    r = hp.Recorder(budget=1 << 30)
    r.device = torch.device("cpu")
    got = []

    def har(x):
        got.append(r.on_ar())
        return x

    r.got = got
    return r, har


def _window_spans(rows):
    return [tuple(v) for v in rows.tolist() if v[1]]


def _span(t):
    a, n = hp.tensor_span(t)
    return a, n


def test_first_forward_records_then_every_all_reduce_prefetches_its_next_window(rec):
    r, har = rec
    root = Root(har)
    assert r.attach_root("target", root) == 3 * 5
    ids, pos = torch.zeros(6, dtype=torch.int64), torch.arange(6)
    root(ids, pos)
    assert r.got == [None] * 6 and len(r.schedules[(id(root), 6)]) == 6  # 1 + 2 + 2 + final
    r.got.clear()
    root(ids, positions=pos)
    assert all(g is not None for g in r.got[:-1]) and r.got[-1] is None  # final AR: empty window
    L0, L1 = root.layers[0], root.layers[1]
    # after layer 0's post-attention AR: its gate + up; after layer 1's input AR: layer 1's
    # attention window in use order (qkv_a, q_b, W_UK_T after it, W_UV before o_proj, o_proj)
    want0 = [_span(L0.gate.weight), _span(L0.up.weight)]
    want1 = [_span(L1.attn.fused_qkv_a_proj.weight), _span(L1.attn.q_b_proj.weight),
             _span(L1.attn.W_UK_T), _span(L1.attn.W_UV), _span(L1.attn.o_proj.weight)]
    for got, want in ((r.got[0], want0), (r.got[1], want1)):
        spans = _window_spans(got[0])
        assert [s[0] for s in spans] == [-(-a // 16) * 16 for a, _ in want]


def test_prefill_sized_forwards_are_neither_recorded_nor_prefetched(rec):
    r, har = rec
    root = Root(har)
    r.attach_root("target", root)
    big = torch.arange(hp.MAX_TOKENS + 1)
    root(torch.zeros_like(big), big)
    root(torch.zeros_like(big), big)
    assert r.schedules == {} and set(r.got) == {None}
    assert r.attach_root("target", root) == 0  # load_model again: no second set of hooks


def test_changed_schedule_in_capture_fails_closed_for_that_key_after_capture(rec, monkeypatch):
    r, har = rec
    root = Root(har)
    r.attach_root("target", root)
    ids, pos = torch.zeros(6, dtype=torch.int64), torch.arange(6)
    root(ids, pos)
    rows = [t[0] for t in r.tables.values() if t[0][:, 1].any()]
    assert rows
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    root.order = [0, 2, 1]
    root(ids, pos)  # captured forward records and compares: different windows
    key = (id(root), 6)
    assert key in r.unstable and r.zero and all(t[:, 1].any() for t in rows)  # deferred in capture
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    r.got.clear()
    root.order = None
    root(ids, pos)
    assert set(r.got) == {None} and not any(t[:, 1].any() for t in rows)


def test_disable_zeroes_every_table_and_stops_prefetch(rec):
    r, har = rec
    root = Root(har)
    r.attach_root("target", root)
    ids, pos = torch.zeros(6, dtype=torch.int64), torch.arange(6)
    root(ids, pos)
    r.disable("test")
    r.got.clear()
    root(ids, pos)
    assert set(r.got) == {None} and not any(t[0][:, 1].any() for t in r.tables.values())


# ------------------------------------------------------------------ host_ar wiring
def test_all_reduce_asks_the_recorder_once_per_call_and_passes_rows(monkeypatch):
    calls = []

    class K:
        def __getitem__(self, grid):
            return lambda *a, **kw: calls.append((grid, a, kw))

    monkeypatch.setattr(H, "_har_kernel", K(), raising=False)
    monkeypatch.setattr(H, "_har_q8_kernel", K(), raising=False)
    har = object.__new__(H.HostAllReduce)
    har.oneshot_max, har.q8_min, har.q8_group, har.round, har.max_blocks = 48 << 10, 256 << 10, 128, 1 << 21, 64
    har.per_program, har.rank, har.world, har.timeout, har.lean = 1024, 0, 8, 1, True
    har.ctr, har.data, har.red, har.flags, har.arr = (torch.zeros(1, dtype=torch.int32),) * 5
    har.data_q = har.data_s = har.red_q = har.red_s = har.ctr
    rows = torch.zeros(16, 2, dtype=torch.int64)
    asks = []
    har.l2pf = lambda: asks.append(1) or (rows, 16)
    har.all_reduce(torch.ones(6 * 6144, dtype=torch.bfloat16))
    har.all_reduce(torch.ones(48 * 6144, dtype=torch.bfloat16))  # int8 transport
    har.l2pf = lambda: asks.append(1)
    har.all_reduce(torch.ones(6144, dtype=torch.bfloat16))
    assert len(asks) == 3
    (_, a0, k0), (_, a1, k1), (_, a2, k2) = calls
    assert k0["PF"] and a0[-2] is rows and a0[-1] == 16 and not k0["ONESHOT"]
    assert k1["PF"] and a1[-2] is rows and "G" in k1
    assert not k2["PF"] and a2[-1] == 0 and k2["ONESHOT"]


def test_prefetch_only_in_the_waiting_branch_and_first_barrier_of_a_round():
    src = (REPO / "suffix_hybrid/host_ar.py").read_text()
    body = src[src.index("    def _barrier("):src.index("    @triton.jit\n    def _har_kernel")]
    head, waiting = body.split("        else:\n", 1)
    assert "_l2_prefetch" not in head and "_l2_prefetch(pf_ptr, pf_cap)" in waiting
    assert waiting.index("_l2_prefetch") < waiting.index("while done == 0")
    tails = []
    for m in re.finditer(r"(?<!def )_barrier\(", src):  # call sites: text up to the closing paren
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        tails.append(" ".join(src[m.end():i - 1].split())[-36:])
    assert sum(t.endswith("rnd == 0, PF") for t in tails) == 3  # one-shot, two-shot 1st, int8 1st
    assert sum(t.endswith("False, False") for t in tails) == 3  # second barriers + all-gather
    asm = src[src.index("def _l2_prefetch"):src.index("def _pf_kernel")]
    assert "cp.async.bulk.prefetch.L2.global [pfa], pfn;" in asm and "@pfq setp.ne.u64 pfq, pfs, 0;" in asm
    assert "st." not in asm  # prefetch never writes


# ------------------------------------------------------------------ install / gates
def test_runner_patch_fails_closed_without_har_or_model(monkeypatch):
    class GPUModelRunner:
        def load_model(self):
            self.model = None

    mod = types.SimpleNamespace(GPUModelRunner=GPUModelRunner)
    hp._patch_runner(mod)
    hp._patch_runner(mod)  # idempotent
    monkeypatch.setattr(hp, "RECORDER", None)
    with pytest.raises(SystemExit, match="no host all-reduce"):
        GPUModelRunner().load_model()
    r = hp.Recorder(1 << 20)
    r.hars = 1
    monkeypatch.setattr(hp, "RECORDER", r)
    with pytest.raises(SystemExit, match="no model"):
        GPUModelRunner().load_model()


def test_gate_off_is_inert_and_boot_gate_is_allowlisted(tmp_path):
    import subprocess
    import sys

    src = (REPO / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["har_l2_prefetch_bench"] == (["-m", "suffix_hybrid.har_prefetch", "bench"], {})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SUFFIX_", "NCCL_"))}
    env.update(PYTHONPATH=str(REPO))
    probe = ("import sitecustomize, sys; print(sum(type(f).__module__ == 'suffix_hybrid.har_prefetch'"
             " for f in sys.meta_path))")
    off = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, cwd=tmp_path)
    on = subprocess.run([sys.executable, "-c", probe], env={**env, hp.GATE_ENV: "1"},
                        capture_output=True, text=True, cwd=tmp_path)
    assert off.stdout.strip() == "0" and on.stdout.strip() == "1", (off.stderr, on.stderr)
