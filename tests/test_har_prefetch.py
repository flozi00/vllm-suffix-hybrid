"""SUFFIX_HAR_L2_PREFETCH on DeepSeek-V4.1-Flash shapes (port of the GLM lane, glm-int 5efcfd8).

CPU only: rows/spans math, the recorder on a V4.1-shaped fake model (two all-reduces per layer, wo_a read
without a module call, hc_*_fn as decoder-layer parameters), the host_ar wiring and the fail-closed paths.
The Triton kernels and cp.async.bulk.prefetch need silicon (boot gate har_l2_prefetch_bench_dsv41).
"""
import os
import re
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
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
            seen.append((x.dtype, None if pf is None else (pf[0].dtype, pf[1]), kw, x.numel()))
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
    assert [s[3] for s in seen[::2]] == [6 * 5120, 5120, 48 * 5120]  # V4.1 token x hidden sizes


def test_selftest_refuses_a_mismatch(monkeypatch):
    class Bad:
        device, rank, q8_min, l2pf, cpu_group = torch.device("cpu"), 0, 1 << 62, None, None

        def all_reduce(self, x, pf=None, **kw):
            return x + (pf is not None)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, **kw: None)
    with pytest.raises(RuntimeError, match="not bit-identical"):
        hp.selftest(Bad(), mb=1)


# ------------------------------------------------------------------ recorder on a V4.1-shaped model
class Attn(nn.Module):
    """DeepseekV4Attention-shaped: wo_a is read through .weight (deep_gemm_fp8_o_proj), never called;
    wo_b is a RowParallelLinear whose output is all-reduced inside the module."""

    def __init__(self, har):
        super().__init__()
        self.har = har
        self.fused_wqa_wkv = nn.Linear(64, 48, bias=False)
        self.wq_b = nn.Linear(32, 64, bias=False)
        self.wo_a = nn.Linear(64, 32, bias=False)
        self.wo_b = nn.Linear(32, 64, bias=False)

    def forward(self, x):
        q = self.wq_b(self.fused_wqa_wkv(x)[:, :32])
        z = F.linear(q, self.wo_a.weight)
        return self.har(self.wo_b(z))


class Gate(nn.Linear):
    """GateLinear-shaped: a hash layer also carries a vocab x topk table that must not enter a window."""

    def __init__(self, hash_layer=False):
        super().__init__(64, 16, bias=False)
        self.tid2eid = nn.Parameter(torch.zeros(512, 6, dtype=torch.int32), requires_grad=False) if hash_layer else None


class Ffn(nn.Module):
    def __init__(self, har, hash_layer=False):
        super().__init__()
        self.har = har
        self.gate = Gate(hash_layer)
        self.shared_up = nn.Linear(64, 64, bias=False)
        self.shared_down = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        self.gate(x)
        return self.har(self.shared_down(self.shared_up(x)))


class Layer(nn.Module):
    def __init__(self, har, hash_layer=False):
        super().__init__()
        self.attn, self.ffn = Attn(har), Ffn(har, hash_layer)
        self.hc_attn_fn = nn.Parameter(torch.randn(6, 64), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.randn(6, 64), requires_grad=False)

    def forward(self, x):
        return self.ffn(self.attn(x))


class Root(nn.Module):
    def __init__(self, har, n=3):
        super().__init__()
        self.layers = nn.ModuleList(Layer(har, hash_layer=(i == 0)) for i in range(n))
        self.order = None

    def forward(self, input_ids, positions):
        x = torch.zeros(positions.shape[-1], 64, dtype=torch.float32)  # embedding reduce is outside
        for i in (self.order or range(len(self.layers))):
            x = self.layers[i](x)
        return x


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


def _starts(window):
    return [s[0] for s in _window_spans(window[0])]


def _aligned(tensors):
    return [-(-hp.tensor_span(t)[0] // 16) * 16 for t in tensors]


def test_first_forward_records_then_every_all_reduce_prefetches_its_next_window(rec):
    r, har = rec
    root = Root(har)
    # 3 x (fused_wqa_wkv, wq_b, wo_a-as-extra-carrier is not a hook of its own: wo_b carries it) + gate/up/down
    # hooks: attn(extras) + 4 linears + ffn(extras) + 3 linears per layer (wo_a is a Linear too)
    assert r.attach_root("target", root) == 3 * (1 + 4 + 1 + 3)
    ids, pos = torch.zeros(6, dtype=torch.int64), torch.arange(6)
    root(ids, pos)
    assert r.got == [None] * 6 and len(r.schedules[(id(root), 6)]) == 6  # 2 all-reduces per layer
    r.got.clear()
    root(ids, positions=pos)
    assert all(g is not None for g in r.got[:-1]) and r.got[-1] is None  # final AR: empty window
    L0, L1 = root.layers[0], root.layers[1]
    # after layer 0's attention all-reduce: its ffn window = hc_ffn_fn, then the router, then the shared expert;
    # the hash layer's tid2eid table is not part of it
    want_ffn0 = [L0.hc_ffn_fn, L0.ffn.gate.weight, L0.ffn.shared_up.weight, L0.ffn.shared_down.weight]
    assert _starts(r.got[0]) == _aligned(want_ffn0)
    # after layer 0's ffn all-reduce: layer 1's attention window in use order -- hc_attn_fn, fused_wqa_wkv,
    # wq_b, then wo_a (an extra before wo_b: it is never called) and wo_b
    want_attn1 = [L1.hc_attn_fn, L1.attn.fused_wqa_wkv.weight, L1.attn.wq_b.weight,
                  L1.attn.wo_a.weight, L1.attn.wo_b.weight]
    assert _starts(r.got[1]) == _aligned(want_attn1)
    # layer 1's ffn window follows its attention all-reduce, the next attention window its ffn all-reduce
    assert _starts(r.got[2]) == _aligned([L1.hc_ffn_fn, L1.ffn.gate.weight, L1.ffn.shared_up.weight,
                                          L1.ffn.shared_down.weight])
    assert _starts(r.got[3])[0] == _aligned([root.layers[2].hc_attn_fn])[0]


def test_hash_routing_table_is_not_prefetched_and_other_params_are(rec):
    r, har = rec
    root = Root(har)
    gate = root.layers[0].ffn.gate
    assert gate.tid2eid is not None and "tid2eid" in hp.SKIP_PARAMS
    assert gate.tid2eid.data_ptr() not in {s[0] for g in r.schedules.values() for w in g for s in w}
    r.attach_root("target", root)
    root(torch.zeros(6, dtype=torch.int64), torch.arange(6))
    spans = {s for sig in r.schedules[(id(root), 6)] for s in sig}
    assert hp.tensor_span(gate.tid2eid) not in spans and hp.tensor_span(gate.weight) in spans


def test_extras_name_the_wo_a_and_hc_reads_of_the_real_module_layout():
    attn = Attn(lambda x: x)
    layer = Layer(lambda x: x)
    got = hp._extras(attn)
    assert [(sub is attn.wo_b, when) for sub, when, _ in got] == [(True, "before")]
    assert got[0][2] is attn.wo_a.weight
    names = [(sub, t) for sub, _, t in hp._extras(layer)]
    assert names == [(layer.attn, layer.hc_attn_fn), (layer.ffn, layer.hc_ffn_fn)]
    assert hp._extras(nn.Linear(4, 4)) == []   # nothing else is special


def test_a_model_without_linear_modules_refuses_to_arm(rec):
    r, _ = rec

    class Empty(nn.Module):
        def forward(self, input_ids, positions):
            return positions

    with pytest.raises(SystemExit, match="nothing to prefetch"):
        r.attach_root("target", Empty())


def test_schedule_line_is_bounded_and_rank0_only(rec, capsys):
    r, _ = rec
    root = Root(lambda x: x)
    r.roots[id(root)] = "target"
    sig = lambda *n: tuple((i * 4096, v) for i, v in enumerate(n))  # noqa: E731
    r.device = torch.device("cuda", 1)
    r._log_schedule((id(root), 6), [sig(1 << 20)])
    assert capsys.readouterr().err == ""
    r.device = torch.device("cuda", 0)
    draft = Root(lambda x: x)
    r.roots[id(draft)] = "draft"
    for k in range(hp.SCHEDULE_LOGS + 3):
        r._log_schedule((id(root), 6 + k), [sig(3 << 20, 1 << 20), (), sig(40 << 20)])
    r._log_schedule((id(draft), 5), [sig(1 << 20)])   # the draft root still gets its own lines
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == hp.SCHEDULE_LOGS + 1
    assert "schedule target @ 6 tokens: 3 windows after an all-reduce, 2 non-empty" in lines[0]
    assert lines[-1].startswith("[suffix har-l2pf] schedule draft @ 5 tokens: 1 windows")


def test_a_key_first_seen_inside_a_capture_is_reported_as_late(rec, monkeypatch, capsys):
    r, har = rec
    r.device = torch.device("cuda", 0)
    root = Root(har)
    r.attach_root("target", root)
    monkeypatch.setattr(hp, "tensor_span", lambda t: None)   # the fake CPU tensors are not on cuda:0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    root(torch.zeros(6, dtype=torch.int64), torch.arange(6))
    err = capsys.readouterr().err
    assert r.late == 1 and "first record of target @ 6 tokens happened inside CUDA-graph capture" in err
    assert "WITHOUT prefetch" in err and set(r.got) == {None}   # the captured forward asked for no rows


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


# ------------------------------------------------------------------ bench shapes
def test_windows_fit_the_default_budget_and_the_proxies_keep_the_bytes():
    attn, ffn = hp.window_bytes(hp.ATTN), hp.window_bytes(hp.FFN)
    assert 28 << 20 < attn < 32 << 20                  # the attention window fits the 32 MiB default
    assert 9 << 20 < ffn < 11 << 20
    assert hp.budget_bytes() == 32 << 20
    g = torch.Generator().manual_seed(0)
    rnd = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    for name, n, k, fmt in hp.ATTN + hp.FFN:
        w = hp.proxy_weight(n, k, fmt, rnd)
        model_bytes = n * k * {"f32": 4, "bf16": 2, "p8": 1}[fmt] + (n * k // 32 if fmt == "p8" else 0)
        assert w.numel() * w.element_size() == pytest.approx(model_bytes, rel=0.04), name


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
    har.all_reduce(torch.ones(6 * 5120, dtype=torch.bfloat16))
    har.all_reduce(torch.ones(48 * 5120, dtype=torch.bfloat16))  # int8 transport
    har.l2pf = lambda: asks.append(1)
    har.all_reduce(torch.ones(5120, dtype=torch.bfloat16))
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


def test_make_comm_attaches_the_prefetch_only_with_the_gate(monkeypatch):
    from suffix_hybrid import nccl_split as ns

    attached = []

    class FakeHost:
        def __init__(self, *a, **kw):
            attached.append("built")

    monkeypatch.setattr("suffix_hybrid.host_ar.HostAllReduce", FakeHost)
    monkeypatch.setattr(hp, "attach", lambda har: attached.append("attach"))
    monkeypatch.delenv(hp.GATE_ENV, raising=False)
    ns.make_comm("har", None, "cpu")
    monkeypatch.setenv(hp.GATE_ENV, "1")
    ns.make_comm("har", None, "cpu")
    assert attached == ["built", "built", "attach"]


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
    assert g["_BOOT_GATES"]["har_l2_prefetch_bench_dsv41"] == (["-m", "suffix_hybrid.har_prefetch", "bench"], {})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SUFFIX_", "NCCL_"))}
    env.update(PYTHONPATH=str(REPO))
    probe = ("import sitecustomize, sys; print(sum(type(f).__module__ == 'suffix_hybrid.har_prefetch'"
             " for f in sys.meta_path))")
    off = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, cwd=tmp_path)
    on = subprocess.run([sys.executable, "-c", probe], env={**env, hp.GATE_ENV: "1"},
                        capture_output=True, text=True, cwd=tmp_path)
    assert off.stdout.strip() == "0" and on.stdout.strip() == "1", (off.stderr, on.stderr)
