# SPDX-License-Identifier: Apache-2.0
"""Plumbing of the boot gate dsa_indexer_capture_replay (kernels/dsa_indexer_gate.py)
on CPU: the CUDA graph is replaced by a stand-in runner that (like the real
capture) forbids any launch whose specialisation class was not warmed, and
replays by re-running the captured call into the captured output tensors.
What the real gate adds on silicon -- real capture on a side stream, real replay,
compute-sanitizer -- is the lab's job; this proves the schedule (every input
rewritten in place, edge lengths, rows that are not a multiple of G), the
warm-up-only-before-capture contract, and that a stale replay is caught."""
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_capture as cap  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer_gate as gate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import test_helion_dsa_indexer as th  # noqa: E402

CPU = torch.device("cpu")
SMALL = ((1, 6, 300, 64, 0), (3, 5, 200, 128, 128), (8, 6, 150, 32, 128))


class FakeRunner:
    """Capture = run the call once with the capture guard on (an unwarmed launch is
    refused, exactly as in a real capture); replay = re-run into the same output tensors."""

    def __init__(self, monkeypatch, stale=False):
        self.mp, self.stale = monkeypatch, stale

    def capture(self, run):
        self.mp.setattr(cap, "capturing", lambda: True)
        try:
            out = run()
        finally:
            self.mp.setattr(cap, "capturing", lambda: False)
        return run, out

    def replay(self, run):
        if self.stale:
            return                       # a graph that did not see the new inputs
        new = run()
        graph_out = self.last_out
        for k, v in new.items():
            graph_out[k].copy_(v)

    def sync(self):
        pass


def arm(monkeypatch):
    shipped = {hdi.DECODE, hdi.GROUP, hdi.CANDK}
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None: {"block_sizes": [32]} if kernel in shipped else None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", th.make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    monkeypatch.setattr(di, "_sms", lambda dev: 4)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: types.SimpleNamespace(cuda_stream=0))
    monkeypatch.setenv(di.HELION_ENV, "helion")
    monkeypatch.setenv(di.GROUP_ENV, "6")
    cap.reset()


def run(monkeypatch, runner=None, cases=SMALL, trials=3, entries=None):
    arm(monkeypatch)
    runner = runner or FakeRunner(monkeypatch)
    # the stand-in's replay needs the captured outputs: remember them at capture
    cap_orig = runner.capture

    def capture(run_):
        g, out = cap_orig(run_)
        runner.last_out = out
        return g, out
    runner.capture = capture
    return gate.run_gate(runner, CPU, cases, trials, entries or gate.ENTRIES[:3], log=lambda m: None)


def test_every_decode_entry_replays_bit_exact_after_a_warm_up_only(monkeypatch):
    lines = run(monkeypatch)
    names = [ln.split()[1] for ln in lines]
    assert names == ["row", "group", "cand"] * len(SMALL)
    assert all("bit-exact vs eager" in ln and ln.endswith("OK") for ln in lines)
    assert any("helion row-group" in ln for ln in lines)          # the group body ran in the capture
    assert any("fused candidates" in ln and "no vLLM" in ln for ln in lines)


def test_the_capture_refuses_what_the_warm_up_missed(monkeypatch):
    arm(monkeypatch)
    monkeypatch.setattr(hdi, "warm_all", lambda *a, **k: None)   # a warm-up that warms nothing
    runner = FakeRunner(monkeypatch)
    with pytest.raises(RuntimeError, match="inside (a )?CUDA-graph capture"):
        gate.run_gate(runner, CPU, SMALL[:1], 2, gate.ENTRIES[:1], log=lambda m: None)


def test_a_replay_that_ignores_the_new_inputs_fails_the_gate(monkeypatch):
    runner = FakeRunner(monkeypatch, stale=True)
    with pytest.raises(RuntimeError, match="replay != eager"):
        run(monkeypatch, runner, SMALL[:1], 2, gate.ENTRIES[:1])


def test_refill_rewrites_every_input_in_place_with_edge_lengths():
    rig = gate.Rig(CPU, 4, 6, 300, 64, 0, seed=1)
    ptrs = [t.data_ptr() for t in (rig.q, rig.w, rig.cl, rig.bt, rig.cand, rig.back, rig.logits)]
    seen_lens, snaps = set(), []
    for trial in range(8):
        rig.refill(trial)
        snaps.append((rig.q.clone(), rig.cl.clone(), rig.bt.clone(), rig.back.clone()))
        seen_lens.update(rig.lens.tolist())
        assert int(rig.lens.max()) <= rig.capacity
        # block tables point only at valid, disjoint pages
        for i, n in enumerate(rig.lens.tolist()):
            nbk = min(-(-n // 64), rig.width)
            assert (rig.bt[i * 6, :nbk] < rig.nb - 4).all()
        used = [p for i in range(4) for p in rig.bt[i * 6, :min(-(-int(rig.lens[i]) // 64), rig.width)].tolist()]
        assert len(used) == len(set(used))
    assert ptrs == [t.data_ptr() for t in (rig.q, rig.w, rig.cl, rig.bt, rig.cand, rig.back, rig.logits)]
    assert any(not torch.equal(a[i], b[i]) for a, b in zip(snaps, snaps[1:]) for i in range(4))   # inputs change
    assert seen_lens & set(gate.EDGE) and max(seen_lens) >= rig.n_cols     # edges and the logits width
    assert rig.capacity > rig.n_cols                                       # tables cover past the width


def test_pad_bytes_between_pages_are_garbage_the_kernels_must_not_read():
    rig = gate.Rig(CPU, 1, 6, 200, 128, 128, seed=2)
    rig.refill(0)
    pages = rig.back.view(rig.nb, rig.stride)
    assert int(pages[:, 128 * 132:].to(torch.int32).sum()) > 0           # not zeros


def test_same_bits_treats_nan_patterns_as_equal_and_flags_one_ulp():
    a = torch.tensor([1.0, float("nan"), 3.0])
    mask = torch.tensor([True, True, True])
    assert gate.same_bits(a, a.clone(), mask)
    b = a.clone()
    b[2] = torch.nextafter(b[2], torch.tensor(4.0))
    assert not gate.same_bits(a, b, mask)
    assert gate.same_bits(a, b, torch.tensor([True, True, False]))
    assert gate.same_sets(torch.tensor([[3, 1, -1]]), torch.tensor([[1, 3, -1]]))
    assert not gate.same_sets(torch.tensor([[3, 1, -1]]), torch.tensor([[1, 2, -1]]))


def test_cli_entry_points_exist():
    assert callable(gate.main)
    src = Path(di.__file__).read_text()
    assert 'mode == "capture_replay"' in src and "dsa_indexer_gate" in src
