# SPDX-License-Identifier: Apache-2.0
"""The K3/K4 harness (suffix_hybrid/kernels/dsv41_bench.py) end to end on CPU: `oracle` for every
family through Helion's interpret mode (SUFFIX_DSV41_FUSE_LAB_DEVICE=cpu stand-in), and the boot-gate
registry. Skipped without Helion."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("helion")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_common as cm  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_engram as eg  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_hc as hc  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_router as rt  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_swiglu as sw  # noqa: E402
from suffix_hybrid.kernels import dsv41_bench as bench  # noqa: E402

FAMILIES = [hc, rt, sw, eg]


@pytest.fixture
def cpu_lab(monkeypatch):
    monkeypatch.setattr(cm, "DEVICE", "cpu")
    monkeypatch.setattr(hk, "platform", lambda device=0: "cpu_stand_in")
    monkeypatch.setattr(hk, "_TABLES", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    orig = torch.mm
    monkeypatch.setattr(torch, "mm", lambda a, b, out_dtype=None: orig(
        *(t.to(out_dtype) for t in (a, b))) if out_dtype is not None and a.dtype != out_dtype else orig(a, b))


@pytest.mark.parametrize("module", FAMILIES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_oracle_cli_passes_on_cpu(module, cpu_lab, capsys):
    status = bench.main(module, ["oracle", "--buckets", "6"])
    out = capsys.readouterr().out
    assert status == 0, out
    assert out.count("PASS") >= len(module.KERNELS) and "FAIL" not in out, out


def test_oracle_cli_reports_a_failing_kernel(cpu_lab, capsys, monkeypatch):
    ent = dict(sw.KERNELS[0])
    ent["check"] = lambda got, ref: (False, "forced")
    monkeypatch.setattr(sw, "KERNELS", [ent])
    assert bench.main(sw, ["oracle", "--buckets", "6"]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("module", FAMILIES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_kernel_has_a_capture_replay_boot_gate(module):
    names = {e["name"] for e in module.KERNELS}
    assert set(module.BOOT_GATES) == {f"{n}_capture_replay" for n in names}
    for fn_name, gate in module.BOOT_GATES.items():
        assert getattr(module, fn_name) is gate and callable(gate)


@pytest.mark.parametrize("module", FAMILIES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_default_configs_stay_inside_the_search_space(module):
    space = getattr(module, "SEARCH_SPACE", None)
    if space is None:
        return
    for name, dims in space.items():
        for t in module.BUCKETS:
            case_ = module.case(t) if module is not hc else module.case(t, 5120)
            cfg = module.default_config(case_, name) if module in (hc, rt) else module.default_config(case_)
            for (lo, hi), v in zip(dims.values(), cfg["block_sizes"]):
                assert lo <= v <= hi and v & (v - 1) == 0, (name, t, cfg)


class FakeRunner:
    """capture = run once and remember the outputs; replay = run again into them (CPU stand-in)."""

    def capture(self, fn):
        self.fn, self.out = fn, fn()
        self.flat = cm._flat(self.out)
        return self.out

    def replay(self):
        for o, n in zip(self.flat, cm._flat(self.fn())):
            o.copy_(n)


@pytest.mark.parametrize("module", FAMILIES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_boot_gates_run_end_to_end_on_cpu(module, cpu_lab, capsys):
    """Every <kernel>_capture_replay runs its real kernel (Helion interpret mode) through the gate
    logic with the CPU stand-in graph runner: the argument plumbing of the gate itself is exercised."""
    for name, gate in module.BOOT_GATES.items():
        assert gate(tokens=6, runner_factory=FakeRunner), name
    err = capsys.readouterr().err
    for name in module.BOOT_GATES:
        assert f"{name} PASS" in err, err


def test_hc_bench_boundary_runs_on_cpu(cpu_lab, monkeypatch, capsys):
    """K4's whole-boundary comparison (split2 vs resident vs the stock function) end to end, with the
    CUDA-graph timer and the stock TileLang call replaced by CPU stand-ins."""
    monkeypatch.setattr(bench, "graph_us", lambda fn, iters=0, **k: (fn(), 10.0)[1])
    monkeypatch.setattr(hc, "baseline", lambda *a: None)
    assert hc.bench_boundary(tokens=(6,), hidden=160, iters=1) == 0
    out = capsys.readouterr().out
    assert "split2" in out and "resident" in out and "FAIL" not in out, out


def test_router_bench_boundary_runs_on_cpu(cpu_lab, monkeypatch, capsys):
    import sys
    import types
    from test_dsv41_router import dsv4_topk_literal
    fake = types.ModuleType("vllm.model_executor.layers.fused_moe.router.dsv4_topk")

    def dsv4_topk(logits, bias, indices_dtype, scale, is_padding=None, input_ids=None, bias_vl=None,
                  image_sentinel_lo=0):
        return dsv4_topk_literal(logits, bias, scale, is_padding, input_ids, bias_vl, image_sentinel_lo)
    fake.dsv4_topk = dsv4_topk
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.router"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(bench, "graph_us", lambda fn, iters=0, **k: (fn(), 10.0)[1])
    monkeypatch.setattr(rt, "_resident_baseline", lambda *a: None)
    assert rt.bench_boundary(tokens=(6,), iters=1) == 0
    out = capsys.readouterr().out
    assert "unfused" in out and "resident" in out and "FAIL" not in out, out
