"""TP micro-batch overlap (suffix_hybrid/tp_overlap.py) on CPU.

Proves, with FakeStreams (deferred launches run in a random order that keeps
only per-stream FIFO + event waits, i.e. what a GPU guarantees):
  1. the pipelined forward is bit-identical to the unsplit forward and to the
     split-sequential one on a shrunk DeepSeek-V3.2-like TP stack, single
     process and at TP=2 over gloo (ranks interleave differently);
  2. every overlapped collective is on the comm stream in the same order on
     all ranks (and only the embedding all-reduce stays outside);
  3. the fallback thresholds / refusals; dropping any wait is detected.
"""
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sm120"))  # hisparse_mtp_patch.oracle.index_leaders

from suffix_hybrid import tp_overlap as tpo  # noqa: E402
from suffix_hybrid import tp_overlap_bench as tb  # noqa: E402

FIX = REPO / "sm120/tests/fixtures/vllm_0.30.0/deepseek_v32_model.py"


def double(t):  # all-reduce of two identical partials, one process
    return t * 2


def double_into(x, out):
    out.copy_(x * 2)


def _model(**kw):
    c = tb.toy_config(**kw)
    return c, tb.ToyModel(c, 0, 1, double)


# ------------------------------------------------------------- split planner
def test_plan_split_is_gated_and_falls_back(monkeypatch):
    monkeypatch.delenv(tpo.ENV, raising=False)
    monkeypatch.delenv(tpo.MIN_TOKENS_ENV, raising=False)
    assert tpo.plan_split([6] * 32, 6) is None  # default off
    monkeypatch.setenv(tpo.ENV, "1")
    assert tpo.plan_split([6] * 32, 6) == 96                 # c32 MTP k=5: 16 + 16 requests
    assert tpo.plan_split([6] * 9, 6) == 30                  # odd: ub0 gets the extra request
    assert tpo.plan_split([6] * 7, 6) is None                # 42 < 48 tokens
    assert tpo.plan_split([6] * 8, 6) == 24
    assert tpo.plan_split([6] * 7 + [1], 6) is None          # ragged verify
    assert tpo.plan_split([6] * 20 + [512], 6) is None       # prefill chunk in the batch
    assert tpo.plan_split([96], 96) is None                  # one request never splits
    assert tpo.plan_split([1] * 64, 1) == 32                 # plain decode
    assert tpo.plan_split([6] * 4, 6, threshold=12) == 12
    monkeypatch.setenv(tpo.MIN_TOKENS_ENV, "192")
    assert tpo.plan_split([6] * 31, 6) is None and tpo.plan_split([6] * 32, 6) == 96


# ---------------------------------------------------------- drift / refusals
def test_anchors_match_pinned_vllm_source_and_drift_is_refused():
    src = FIX.read_text()
    tpo.check_source(src)
    with pytest.raises(tpo.DriftError, match="self.mlp"):
        tpo.check_source(src.replace("self.mlp(hidden_states)", "self.mlp(hidden_states, x)"))


def test_refusals():
    _, m = _model()
    assert tpo.glm_refusal(m, 2) is None
    assert "size 1" in tpo.glm_refusal(m, 1)
    m.use_sequence_parallel = True
    assert "sequence-parallel" in tpo.glm_refusal(m, 2)
    m.use_sequence_parallel = False
    m.end_layer = 2
    assert "pipeline" in tpo.glm_refusal(m, 2)
    m.end_layer, m.aux_hidden_state_layers = len(m.layers), (1,)
    assert "EAGLE3" in tpo.glm_refusal(m, 2)
    m.aux_hidden_state_layers = ()
    HiSparseMLAIndexGroup = type("HiSparseMLAIndexGroup", (), {})
    m.layers[1].self_attn.impl = types.SimpleNamespace(index_group=HiSparseMLAIndexGroup())
    assert "HiSparse" in tpo.glm_refusal(m, 2)


def test_cross_layer_buffers_find_topk_and_index_group_workspaces():
    _, m = _model()
    ws = torch.zeros(9, 8, dtype=torch.int32)
    group = types.SimpleNamespace(physical_topk_indices=ws, valid_topk_counts=torch.zeros(9))
    m.layers[0].self_attn.impl = types.SimpleNamespace(index_group=group)
    m.layers[2].self_attn.impl = types.SimpleNamespace(index_group=group)  # a follower
    bufs = tpo.cross_layer_buffers(m)
    assert bufs[0] is m.topk_indices_buffer and bufs[1] is ws and len(bufs) == 3


# ------------------------------------------------ bit-identity, one process
@pytest.mark.parametrize("joint", [False, True], ids=["split", "hybrid"])
@pytest.mark.parametrize("n", [12, 48, 96])
def test_overlap_bit_identical_under_random_interleavings(n, joint):
    c, m = _model()
    ids, pos, reqs = tb.toy_batch(c, n, seed=n)
    split = (n // 6 + 1) // 2 * 6
    with torch.inference_mode():
        full = tb.run_full(m, ids, pos, reqs)
        assert torch.equal(tb.run_split_sequential(m, ids, pos, reqs, split), full)
        for seed in range(12):
            out = tb.run_overlap(m, ids, pos, reqs, split, tpo.FakeStreams(seed), double_into,
                                 joint_mlp=joint)
            assert torch.equal(out, full), seed
    assert m.topk_indices_buffer.storage_offset() == 0 and m.topk_indices_buffer.shape[0] == 96


def test_indexshare_rows_hazard_is_real_and_rows_from_fixes_it():
    c, m = _model()  # index_leaders(4, 2, 2) = [T, T, F, T]: layer 2 re-reads layer 1's rows
    ids, pos, reqs = tb.toy_batch(c, 48, seed=3)
    with torch.inference_mode():
        full = tb.run_full(m, ids, pos, reqs)
        bad = tb.run_overlap(m, ids, pos, reqs, 24, tpo.FakeStreams(0), double_into, rebase=False)
    assert not torch.equal(bad, full)
    buf = m.topk_indices_buffer
    with tpo.rows_from([buf], 24):
        assert buf.shape[0] == 72 and buf.storage_offset() == 24 * buf.stride(0)
    assert buf.shape[0] == 96 and buf.storage_offset() == 0
    with pytest.raises(ValueError):
        with tpo.rows_from([buf], 97):
            pass
    assert buf.shape[0] == 96


def test_toy_forward_context_must_follow_the_micro_batch():
    c, m = _model()
    ids, pos, reqs = tb.toy_batch(c, 48, seed=1)
    with pytest.raises(RuntimeError, match="forward context"), torch.inference_mode():
        tpo.glm_forward(m, ids, pos, 24, tpo.FakeStreams(0), double_into)  # no enter()


class DropWait(tpo.FakeStreams):
    """FakeStreams that silently drops waits of one kind: the harness must see it."""

    def __init__(self, seed, stream, kind):
        super().__init__(seed)
        self.drop = (stream, kind)

    def wait(self, stream, ev):
        if (stream, ev[0][0]) != self.drop:
            super().wait(stream, ev)


@pytest.mark.parametrize("stream,kind", [("compute", "done"), ("comm", "ready")])
def test_every_wait_is_load_bearing(stream, kind):
    c, m = _model()
    ids, pos, reqs = tb.toy_batch(c, 48, seed=2)
    wrong = 0  # what a GPU would do silently: read an unreduced / unfilled buffer
    with torch.inference_mode():
        full = tb.run_full(m, ids, pos, reqs)
        for seed in range(12):
            try:
                out = tb.run_overlap(m, ids, pos, reqs, 24, DropWait(seed, stream, kind),
                                     double_into)
                wrong += not torch.equal(out, full)
            except KeyError:  # comm ran before the very first partial existed
                pass
    assert wrong >= 5, wrong


@pytest.mark.parametrize("joint", [False, True], ids=["split", "hybrid"])
def test_collective_log_is_comm_stream_program_order(joint):
    c, m = _model()
    ids, pos, reqs = tb.toy_batch(c, 48, seed=4)
    logs = []
    with torch.inference_mode():
        for seed in range(4):
            log = []
            tb.run_overlap(m, ids, pos, reqs, 24, tpo.FakeStreams(seed), double_into, log=log,
                           joint_mlp=joint)
            logs.append(log)
    want = [(k, ub, (24, c["hidden"])) for k in range(2 * c["layers"]) for ub in (0, 1)]
    assert all(log == want for log in logs)


# ------------------------------------------------------- TP=2 over gloo
def _child(rank, world, path):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{path}/pg", rank=rank, world_size=world)
    try:
        sync = []

        def ar(t):
            sync.append(tuple(t.shape))
            t = t.clone()
            dist.all_reduce(t)
            return t

        def ar_into(x, out):
            out.copy_(x)
            dist.all_reduce(out)

        c = tb.toy_config()
        m = tb.ToyModel(c, rank, world, ar)  # rank-local shards: a real TP=2 stack
        res = {"full": [], "seq": [], "ovl": [], "logs": [], "sync_in_ovl": []}
        with torch.inference_mode():
            for n in (48, 96):
                ids, pos, reqs = tb.toy_batch(c, n, seed=n)
                split = (n // 6 + 1) // 2 * 6
                res["full"].append(tb.run_full(m, ids, pos, reqs))
                res["seq"].append(tb.run_split_sequential(m, ids, pos, reqs, split))
                for s in range(4):  # different interleavings on the two ranks
                    log, n_sync = [], len(sync)
                    res["ovl"].append(tb.run_overlap(m, ids, pos, reqs, split,
                                                     tpo.FakeStreams(100 * rank + s), ar_into, log,
                                                     joint_mlp=s % 2 == 1))
                    res["logs"].append(log)
                    res["sync_in_ovl"].append(sync[n_sync:])
        torch.save(res, f"{path}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_tp2_gloo_bit_identical_and_same_collective_order(tmp_path, monkeypatch):
    import torch.multiprocessing as mp

    for key in [k for k in os.environ if k.startswith("SUFFIX_")]:
        monkeypatch.delenv(key)  # spawned children run sitecustomize
    mp.spawn(_child, args=(2, str(tmp_path)), nprocs=2, join=True)
    rs = [torch.load(tmp_path / f"r{r}.pt", weights_only=False) for r in range(2)]
    hidden = tb.toy_config()["hidden"]
    for r in rs:
        for i, full in enumerate(r["full"]):
            assert torch.equal(r["seq"][i], full)
            assert all(torch.equal(o, full) for o in r["ovl"][4 * i:4 * i + 4])
        # only the unsplit embedding all-reduce runs outside the comm stream
        assert all(s == [(n, hidden)] for s, n in zip(r["sync_in_ovl"], [48] * 4 + [96] * 4))
    assert torch.equal(rs[0]["full"][0], rs[1]["full"][0])  # own shards, same reduced output
    assert rs[0]["logs"] == rs[1]["logs"]
    assert rs[0]["logs"][0] == [(k, ub, (24, hidden)) for k in range(8) for ub in (0, 1)]


# ------------------------------------------------------------- plumbing
def test_pynccl_allreduce_uses_the_band_communicator_and_refuses_qar(monkeypatch):
    calls = []

    class Comm:
        disabled = False

        def __init__(self, name):
            self.name = name

        def all_reduce(self, x, out):
            calls.append(self.name)
            out.copy_(x)
            return out

    base, band = Comm("base"), Comm("band")
    dc = types.SimpleNamespace(pynccl_comm=base,
                               _suffix_pick=lambda x, d: band if x.numel() > 4 else d)
    monkeypatch.delenv("SUFFIX_NCCL_QAR", raising=False)
    ar = tpo.pynccl_allreduce(dc)
    ar(torch.ones(2), torch.empty(2))
    ar(torch.ones(8), torch.empty(8))
    assert calls == ["base", "band"]
    monkeypatch.setenv("SUFFIX_NCCL_QAR", "int8")
    with pytest.raises(ValueError, match="QAR"):
        tpo.pynccl_allreduce(dc)


def test_vllm_enter_swaps_forward_context_and_rows(monkeypatch):
    fc = types.ModuleType("vllm.forward_context")
    fc._forward_context = "full"
    vllm = types.ModuleType("vllm")
    vllm.forward_context = fc
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", fc)
    buf = torch.arange(10).view(5, 2)
    enter = tpo.vllm_enter(["ctx0", "ctx1"], [buf], (0, 3))
    with enter(1):
        assert fc._forward_context == "ctx1" and buf[0].tolist() == [6, 7]
    assert fc._forward_context == "full" and buf.shape == (5, 2)


def test_bench_cli_skips_without_gpus_and_reports(capsys):
    assert tb.main(["oracle"]) == 0 or torch.cuda.is_available()
    row = {"n": 48, "split": 24, "split_vs_unsplit": 0.01, "hybrid_vs_unsplit": 0.02,
           "checks": {"split-eager": True, "hybrid-replay": True}}
    bad = dict(row, checks={"split-eager": True, "hybrid-replay": False})
    assert tb._report(2, "oracle", [{"rank": 0, "rows": [row]}, {"rank": 1, "rows": [row]}], 4)
    assert not tb._report(2, "oracle", [{"rank": 0, "rows": [row]}, {"rank": 1, "rows": [bad]}], 4)
    t = {"n": 192, "base": 4.0, "seq": 4.6, "split": 3.2, "hybrid": 3.0, "compute": 2.2,
         "comm": 1.8}
    assert tb._report(8, "bench", [{"rank": 0, "rows": [t]}], 4)
    out = capsys.readouterr().out
    assert "ORACLE tp=2 n=48 split=24 PASS" in out and "FAIL hybrid-replay" in out
    assert "best hybrid saves 1.000 (25.0 %)" in out and "unsplit 78.0 hybrid 58.5" in out


def test_boot_gates_are_allowlisted():
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    for mode in ("oracle", "bench"):
        argv, env = ns["_BOOT_GATES"][f"tp_overlap_{mode}"]
        assert argv == ["-m", "suffix_hybrid.tp_overlap_bench", mode] and env == {}
    assert ns["_BOOT_GATES"]["tp_overlap_bench_ring_simple"][0][-2:] == ["--spec",
                                                                       "allreduce:ring/Simple"]


def test_bench_moe_graph_path_matches_the_exact_path():
    """The GPU bench's fixed-capacity MoE (static shapes, capturable) computes
    the same function as the exact path when nothing overflows (cap == n)."""
    c = tb.toy_config()
    torch.manual_seed(0)
    x = torch.randn(4, c["hidden"])
    for rank in (0, 1):
        exact = tb.MoE(c, rank, 2, 7, "cpu", torch.float32, True)
        fast = tb.MoE(c, rank, 2, 7, "cpu", torch.float32, False)
        torch.testing.assert_close(fast(x), exact(x), rtol=1e-5, atol=1e-5)
    m = tb.ToyModel(c, 0, 1, double, exact=False)  # whole fast path runs, overlap == unsplit-ish
    ids, pos, reqs = tb.toy_batch(c, 48, seed=5)
    with torch.inference_mode():
        a = tb.run_split_sequential(m, ids, pos, reqs, 24)
        b = tb.run_overlap(m, ids, pos, reqs, 24, tpo.FakeStreams(1), double_into)
    assert torch.equal(a, b)
