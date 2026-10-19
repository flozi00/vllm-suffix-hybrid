# SPDX-License-Identifier: Apache-2.0
"""The mxfp8_capture_replay boot gate (suffix_hybrid/kernels/dsv41_dense_capture_replay.py) and the
lane-D lab driver (suffix_hybrid/kernels/dsv41_dense_lab.py) on CPU: the gate's whole logic runs against
Helion interpret-mode kernels with the EagerRuntime standing in for torch.cuda graphs; the CUDA-only
parts (real capture, streams, sanitizer) are the silicon run's."""
import os

os.environ["HELION_INTERPRET"] = "1"  # before any helion.kernel is built

import pytest  # noqa: E402
import torch  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import mxfp8_gemm as hx  # noqa: E402
from suffix_hybrid.kernels import dsv41_dense_capture_replay as gate  # noqa: E402
from suffix_hybrid.kernels import dsv41_dense_lab as lab  # noqa: E402
from suffix_hybrid.kernels import mxfp8_linear as lane  # noqa: E402


@pytest.fixture
def cpu_mm(monkeypatch):
    mm = torch.mm  # CPU aten::mm has no out_dtype (hl.dot's eager reference uses it)
    monkeypatch.setattr(torch, "mm", lambda a, b, *, out_dtype=None, **kw: (
        mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b, **kw)))


def run(**kw):
    lines = []
    res = gate.run_gate(rt=hx.EagerRuntime, dev="cpu", replays=4, streams=0, emit=lines.append, **kw)
    return res, lines


# ------------------------------------------------------------------------------------ the gate
def test_probe_lists():
    quick = gate.probes("quick")
    assert len(quick) == 9 and len({p.label for p in quick}) == 9
    assert {p.kernel for p in quick} == {hx.A16, hx.A8}
    assert any(p.bmm and p.groups == 2 for p in quick)           # strided wo_a groups (TP4)
    assert any(p.split == 1 for p in quick) and any(p.split >= 8 for p in quick)
    assert any(p.m == 192 for p in quick) and any(p.k == 15360 for p in quick)
    assert all(p.k % 32 == 0 and p.n % 32 == 0 for p in quick)
    full = gate.probes("full")
    assert len(full) == 12 + 7 * 4 * 3 * 2 and len({p.label for p in full}) == len(full)
    assert all(not (p.bmm and p.kernel == hx.A8) for p in full)  # wo_a is A16 only
    sem = gate.probes("quick", "semaphore")
    assert {p.kernel for p in sem} == {hx.A16_SEM, hx.A8_SEM}
    cfg = gate.Probe("x", hx.A16, 64, 160, 5, 4).config()
    assert cfg["split_k"] == 4 and cfg["block_sizes"] == [16, 32, 2]
    assert gate.Probe("x", hx.A16, 64, 160, 64, 2).config()["block_sizes"][0] == 64


def test_shipped_probes_follow_the_config_tables(monkeypatch):
    tables = {hx.A16: {hk.case_key({"k": 5120, "n": 576, "numtokens": 8}): {"split_k": 8, "block_sizes": [16, 16, 4]},
                       hk.case_key({"k": 4096, "n": 1024, "numtokens": 8, "g": 1}): {"split_k": 2}},
              hx.A8: {}}
    monkeypatch.setattr(hk, "table", lambda kernel, plat=None: tables[kernel])
    ps = gate.shipped_probes()
    assert [(p.n, p.k, p.m, p.split, p.bmm, p.source) for p in ps] == [
        (576, 5120, 8, 8, False, "shipped"), (1024, 4096, 8, 2, True, "shipped")]
    assert ps[0].config()["block_sizes"] == [16, 16, 4]  # the shipped config wins over the defaults
    res, lines = run(tier="tiny", plist=[], variant="two_stage")
    assert res["verdict"] == "NOT RUN"  # nothing ran: not a pass


@pytest.mark.parametrize("variant", ["two_stage", "semaphore"])
def test_gate_probe_configs_lower_to_triton(variant):
    """K2-lite: Helion's real compiler front end accepts every quick-tier probe config on the real
    V4.1 per-rank shapes (block sizes, split_k, num_warps / num_stages) and, for the two-stage
    variant, emits a kernel with no atomic / barrier / counter."""
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    for pr in gate.probes("quick", variant):
        sem = pr.kernel in hx.SEM_BODIES
        wdt = torch.uint8 if pr.kernel in (hx.A16, hx.A16_SEM) else torch.float8_e4m3fn
        args = [torch.empty(pr.m, pr.k // 32, 32, dtype=torch.bfloat16),
                torch.empty(pr.n, pr.k // 32, 32, dtype=wdt),
                torch.empty(pr.n, pr.k // 32, dtype=torch.uint8),
                torch.empty(pr.m, pr.n, dtype=torch.bfloat16),
                torch.empty((pr.split, pr.m, pr.n) if pr.split > 1 else (1, 1, 1))]
        if sem:
            args.append(torch.zeros(hx.counter_elems(pr.n), dtype=torch.int32))
        bound = helion.kernel(hx.ALL_BODIES[pr.kernel], static_shapes=False, ref_mode=RefMode.OFF,
                              ignore_warnings=hx._ignore()).bind(tuple(args))
        code = bound.to_triton_code(helion.Config(**pr.config()))
        assert "num_warps=4, num_stages=2" in code, pr.label
        assert ("atomic_add" in code) == sem, pr.label


def test_gate_tiny_passes_on_cpu(cpu_mm):
    pytest.importorskip("helion")
    res, lines = run(plist=gate.probes("tiny"))
    assert res["verdict"] == "PASS", lines
    assert [r["status"] for r in res["probes"]] == ["PASS"] * 4
    assert all("bit-exact" in r["detail"] and "guards intact" in r["detail"] for r in res["probes"])
    assert any("MXFP8-CAPTURE-REPLAY PASS variant=two_stage probes 4 pass / 0 fail" in s for s in lines)
    # the label is printed BEFORE the probe runs (a crash must name the culprit)
    starts = [i for i, s in enumerate(lines) if "PROBE tiny.split2: mxfp8_gemm N=64" in s]
    ends = [i for i, s in enumerate(lines) if "PROBE tiny.split2: PASS" in s]
    assert starts and ends and starts[0] < ends[0]


def test_gate_only_filter_and_eager_mode(cpu_mm):
    pytest.importorskip("helion")
    res, _ = run(plist=gate.probes("tiny"), only="bmm")
    assert [r["label"] for r in res["probes"]] == ["tiny.bmm2"] and res["verdict"] == "PASS"
    lines = []
    res = gate.run_gate(plist=gate.probes("tiny")[:1], rt=hx.EagerRuntime, dev="cpu", capture=False,
                        emit=lines.append)
    assert res["probes"][0]["detail"] == "eager only (--no-capture)"


def test_gate_fails_on_replay_drift(cpu_mm, monkeypatch):
    pytest.importorskip("helion")
    real, calls = hx.apply, []

    def drifting(*a, **kw):
        out = real(*a, **kw)
        calls.append(1)
        if len(calls) > 4:  # eager (3) + the capture call are fine, then state "drifts"
            out = out.clone()
            out[0, 0] += 1.0
        return out
    monkeypatch.setattr(hx, "apply", drifting)
    res, lines = run(plist=gate.probes("tiny")[:1])
    assert res["verdict"] == "FAIL" and "CAPTURE/REPLAY FAIL" in res["probes"][0]["detail"], lines


def test_gate_fails_on_a_store_past_the_output(cpu_mm, monkeypatch):
    pytest.importorskip("helion")
    real = hx.apply

    def overrun(x, w, ws, entry, n, groups=1, bmm=False, out=None, **kw):
        y = real(x, w, ws, entry, n, groups, bmm, out, **kw)
        if out is not None:  # the guarded run: scribble one element past the output view
            torch.as_strided(out, (out.numel() + 1,), (1,))[-1] = 1.0
        return y
    monkeypatch.setattr(hx, "apply", overrun)
    res, _ = run(plist=gate.probes("tiny")[:1])
    assert res["verdict"] == "FAIL" and "GUARD FAIL" in res["probes"][0]["detail"]


def test_gate_fails_on_an_unwritten_element(cpu_mm, monkeypatch):
    pytest.importorskip("helion")
    real = hx.apply

    def skips_a_tile(*a, **kw):
        y = real(*a, **kw)
        y = y.clone()
        y[0, 0] = float("nan")  # what a poisoned, never-written output element looks like
        return y
    monkeypatch.setattr(hx, "apply", skips_a_tile)
    res, _ = run(plist=gate.probes("tiny")[:1])
    assert res["verdict"] == "FAIL" and "non-finite" in res["probes"][0]["detail"]


def test_gate_compile_failure_is_not_run_not_fail(monkeypatch):
    def boom(pr, dev):
        raise RuntimeError("dot_scaled unsupported on this arch")
    monkeypatch.setattr(gate, "_build", boom)
    res, _ = run(plist=gate.probes("tiny"))
    assert res["verdict"] == "NOT RUN" and "dot_scaled" in res["probes"][0]["detail"]


def test_semaphore_variant_needs_the_knob(cpu_mm, monkeypatch):
    pytest.importorskip("helion")
    monkeypatch.delenv(hx.SEM_ENV, raising=False)
    res, lines = run(plist=gate.probes("tiny", "semaphore"), variant="semaphore")
    assert res["verdict"] == "NOT RUN" and hx.SEM_ENV in res["reason"] and "UNPROVEN" in res["reason"]
    monkeypatch.setenv(hx.SEM_ENV, "1")
    res, lines = run(plist=gate.probes("tiny", "semaphore"), variant="semaphore")
    assert res["verdict"] == "PASS", lines  # the old bodies still compute the right thing; capture is the open question
    assert all(r["kernel"] in hx.SEM_BODIES for r in res["probes"])


def test_cli_without_cuda_is_not_run(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gate.main(["--quick"]) == 2
    assert "NOT RUN: no CUDA device" in capsys.readouterr().out
    assert lane.main(["capture_replay"]) == 2          # python -m ...mxfp8_linear capture_replay
    assert lane.main([]) == 2 and "usage" in capsys.readouterr().err


def test_boot_gate_entries_parse_with_the_gates_own_parser(monkeypatch, capsys):
    """The two sitecustomize `_BOOT_GATES` entries the main session pastes (BOOT_GATES): argv =
    -m <importable module> <subcommand> <flags the gate accepts>; the _sem entry carries its knob."""
    import importlib.util
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert set(gate.BOOT_GATES) == {"mxfp8_capture_replay", "mxfp8_capture_replay_sem"}
    for name, (argv, extra) in gate.BOOT_GATES.items():
        assert argv[0] == "-m" and importlib.util.find_spec(argv[1]) is not None, name
        assert argv[2] == "capture_replay"
        args = gate.parser().parse_args(argv[3:])
        assert args.variant == ("semaphore" if name.endswith("_sem") else "two_stage")
        assert (extra == {hx.SEM_ENV: "1"}) == name.endswith("_sem")
        assert lane.main(argv[2:]) == 2  # runs through the module dispatch; no CUDA here -> NOT RUN
    capsys.readouterr()


# ---------------------------------------------------------------------------------- lab driver
def _fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "fake gpu")


def test_lab_runs_the_markov_bench_first(monkeypatch):
    _fake_cuda(monkeypatch)
    order = []
    monkeypatch.setattr(lab, "markov_bench", lambda *a, **k: order.append("markov"))
    monkeypatch.setattr(lab, "capture_stage", lambda *a, **k: order.append("capture") or {"verdict": "PASS"})
    monkeypatch.setattr(lab, "cases_stage", lambda *a, **k: order.append("cases"))
    assert lab.STAGES == ("markov", "capture", "cases")
    assert lab.main(["--effort", "none"]) == 0 and order == ["markov", "capture", "cases"]
    order.clear()
    assert lab.main(["--skip", "markov"]) == 0 and order == ["capture", "cases"]
    with pytest.raises(SystemExit):
        lab.main(["--skip", "nope"])


def test_lab_stops_after_a_capture_failure(monkeypatch):
    _fake_cuda(monkeypatch)
    order = []
    monkeypatch.setattr(lab, "markov_bench", lambda *a, **k: order.append("markov"))
    monkeypatch.setattr(lab, "capture_stage", lambda *a, **k: order.append("capture") or {"verdict": "FAIL"})
    monkeypatch.setattr(lab, "cases_stage", lambda *a, **k: order.append("cases"))
    assert lab.main([]) == 1 and order == ["markov", "capture"]  # no timing of an unsafe lane
    order.clear()
    assert lab.main(["--keep-going"]) == 0 and order == ["markov", "capture", "cases"]


def test_lab_cases_stage_plans(monkeypatch, capsys):
    plans = []
    monkeypatch.setattr(hx, "lab_main", lambda plan: plans.append(plan))
    lab.cases_stage("quick", 99.0)
    assert [p["kernels"] for p in plans] == [[hx.A16], [hx.A8]]
    base = hx._cases({"name": "deepseek-v4.1"}, hx.A16)
    assert plans[0]["cases"] == base + lab.MARKOV_CASES and len(base) > 0
    assert "cases" not in plans[1] and plans[0]["effort"] == plans[1]["effort"] == "quick"
    assert plans[0]["budget_s"] == 99.0
    plans.clear()
    lab.cases_stage("none", 5.0, markov_cases=False)
    assert "cases" not in plans[0]
    capsys.readouterr()


def test_markov_bench_on_cpu():
    recs = []
    out = lab.markov_bench(rows=(1, 4), iters=2, device="cpu", n=512, k=16, chain=3, flush_mb=1,
                           emit=recs.append)
    assert out == recs and [r["rows"] for r in recs] == [1, 4]
    for r in recs:
        assert r["stage"] == "markov" and r["chain_calls"] == 3 and r["w2_mb"] == round(512 * 16 * 2 / 1e6, 1)
        assert r["warm_us_per_call"] > 0 and r["cold_us_per_call"] >= 0
        assert isinstance(r["l2_resident"], bool) and r["hint"]
    assert lab.MARKOV_CASES[0] == {"k": 256, "n": 129280, "numtokens": 8}
