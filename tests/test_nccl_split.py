import math
import os
import types
from pathlib import Path

import pytest
import torch

from suffix_hybrid import nccl_qar as qar
from suffix_hybrid import nccl_split as ns

FIX = Path(__file__).resolve().parents[1] / "sm120/tests/fixtures/vllm_0.30.0/cuda_communicator.py"
K, M = 1 << 10, 1 << 20


# ------------------------------------------------------------ anchor rewrite
def test_patch_applies_to_pinned_source_and_routes_by_size():
    out = ns.patch_all_reduce(FIX.read_text())
    assert "_suffix_pick" in out and out.count("pynccl_comm.all_reduce(input_)") == 1


def test_drift_is_refused():
    src = FIX.read_text().replace(ns.OLD, ns.OLD.replace("assert pynccl_comm", "assert  pynccl_comm"))
    with pytest.raises(ns.PatchDriftError):
        ns.patch_all_reduce(src)


def test_global_nccl_algo_is_refused(monkeypatch):
    monkeypatch.setenv(ns.ENV, "allreduce:tree")
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    with pytest.raises(ns.PatchDriftError):
        ns.apply(types.SimpleNamespace(__file__=str(FIX)))


def test_unset_is_inert(monkeypatch):
    for k in (ns.ENV, ns.BANDS_ENV, ns.AUTOTUNE_ENV, ns.QAR_ENV):
        monkeypatch.delenv(k, raising=False)
    assert ns.apply(object()) is None


class FakeComm:
    def __init__(self, name):
        self.name, self.calls, self.destroyed = name, [], False

    def all_reduce(self, x):
        self.calls.append(x.numel() * x.element_size())
        return x * 8

    def destroy(self):
        self.destroyed = True


def _fake_module():
    """The pinned cuda_communicator's names all_reduce needs, around a fake CudaCommunicator."""
    class CudaCommunicator:
        def __init__(self, unique_name="tp:0", world_size=8):
            self.unique_name, self.world_size, self.rank = unique_name, world_size, 0
            self.cpu_group, self.device = None, "cpu"
            self.pynccl_comm = FakeComm("default")
            self.pynccl_comm.disabled, self.pynccl_comm.world_size = False, world_size
            self.fi_ar_comm = self.qr_comm = self.fi_pcie_ipc_ar_comm = None
            self.aiter_ar_comm = self.ca_comm = self.symm_mem_comm = None
            self.use_aiter_allreduce = False

    return types.SimpleNamespace(__file__=str(FIX), torch=torch, CudaCommunicator=CudaCommunicator,
                                 should_nccl_symm_mem_allreduce=lambda *a: False)


def test_apply_routes_real_patched_all_reduce_by_bytes(monkeypatch):
    for k in (ns.ENV, ns.AUTOTUNE_ENV, ns.QAR_ENV, "NCCL_ALGO", "NCCL_PROTO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ns.BANDS_ENV, "256K-4M:allreduce:ring/Simple,4M-16M:allreduce:tree")
    made = {}
    monkeypatch.setattr(ns, "make_comm", lambda spec, g, d: made.setdefault(spec, FakeComm(spec)))
    mod = _fake_module()
    ns.apply(mod)
    cc = mod.CudaCommunicator()
    for nbytes in (48 * K, 256 * K, 4 * M - 2, 4 * M, 16 * M):
        cc.all_reduce(torch.ones(nbytes // 2, dtype=torch.bfloat16))
    assert cc.pynccl_comm.calls == [48 * K, 16 * M]
    assert made["allreduce:ring/Simple"].calls == [256 * K, 4 * M - 2]
    assert made["allreduce:tree"].calls == [4 * M]
    single = mod.CudaCommunicator(world_size=1)
    assert single._suffix_pick is None


def test_legacy_small_envs_are_one_inclusive_band(monkeypatch):
    for k in (ns.BANDS_ENV, ns.AUTOTUNE_ENV, ns.QAR_ENV, "NCCL_ALGO", "NCCL_PROTO", ns.BYTES_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ns.ENV, "allreduce:tree")
    monkeypatch.setenv(ns.PROTO_ENV, "Simple")
    monkeypatch.setenv(ns.MIN_ENV, str(384 * K))
    bands, auto, mode = ns.config()
    assert bands == [(384 * K, 2 * M + 1, "allreduce:tree/Simple")] and not auto and mode == ""
    monkeypatch.setenv(ns.BANDS_ENV, "0-1M:default")
    with pytest.raises(ns.PatchDriftError):
        ns.config()


def test_bad_qar_mode_is_refused(monkeypatch):
    monkeypatch.setenv(ns.QAR_ENV, "int4")
    with pytest.raises(ns.PatchDriftError):
        ns.config()


def test_qar_band_and_dtype_in_pick():
    q, a, d = FakeComm("qar"), FakeComm("a"), FakeComm("d")
    pick = ns.make_pick([(0, 8 * M, a)], q, 4 * M, 64 * M)
    assert pick(torch.empty(M, dtype=torch.bfloat16), d) is a          # 2 MiB
    assert pick(torch.empty(2 * M, dtype=torch.bfloat16), d) is q      # 4 MiB
    assert pick(torch.empty(M, dtype=torch.float32), d) is a           # 4 MiB fp32: no QAR
    assert pick(torch.empty(32 * M, dtype=torch.bfloat16), d) is d     # 64 MiB: past QAR, no band


# ---------------------------------------------------------------- band specs
def test_parse_bands_roundtrip_and_errors():
    text = "0-256K:default,256K-1.5M:allreduce:ring/Simple,4M-inf:allreduce:tree/LL"
    bands = ns.parse_bands(text)
    assert bands[1] == (256 * K, 1536 * K, "allreduce:ring/Simple") and bands[2][1] == ns.INF
    assert ns.parse_bands(ns.bands_str(bands)) == bands
    assert ns.spec_env("allreduce:ring/Simple") == {"NCCL_ALGO": "allreduce:ring", "NCCL_PROTO": "Simple"}
    assert ns.parse_size("256KiB") == ns.parse_size("256K") == 256 * K
    for bad in ("0-1M", "1M-1M:default", "0-2M:default,1M-3M:default", "0-1M:/Simple"):
        with pytest.raises(ValueError):
            ns.parse_bands(bad)


def test_autotune_sizes_cover_decode_and_prefill():
    s = ns.autotune_sizes(6144)
    assert s[0] == 6144 * 2 and 16 * K in s and s[-1] == 128 * M and s == sorted(set(s))
    assert {6 * 6144 * 2, 192 * 6144 * 2, 8192 * 6144 * 2} <= set(s)


def test_build_bands_from_worker06_table():
    # 2026-09-29 worker-06 measurements (us): default comm vs NCCL_ALGO=allreduce:tree comm.
    sizes = [48 * K, 192 * K, 384 * K, 768 * K, 1536 * K, 3 * M, 6 * M]
    table = {"default": [42, 113, 280, 784, 1595, 534, 1056],
             "allreduce:tree": [49, 132, math.inf, 338, 608, 846, 1286]}
    bands = ns.build_bands(sizes, table)
    lo, hi = ns._cross(384 * K, 768 * K), ns._cross(1536 * K, 3 * M)
    assert bands == [(0, lo, "default"), (lo, hi, "allreduce:tree"), (hi, ns.INF, "default")]
    assert 384 * K < lo < 768 * K and abs(lo - math.sqrt(384 * K * 768 * K)) <= 512


def test_build_bands_margin_and_unmeasured_default():
    sizes = [1 * M, 2 * M, 4 * M]
    table = {"default": [100, 100, math.inf], "x": [97, 50, 1]}
    assert ns.build_bands(sizes, table) == [(0, ns._cross(M, 2 * M), "default"),
                                            (ns._cross(M, 2 * M), ns._cross(2 * M, 4 * M), "x"),
                                            (ns._cross(2 * M, 4 * M), ns.INF, "default")]


# ------------------------------------------------------ autotune over gloo (4 ranks)
SIZES = [16 * K, 64 * K, 256 * K, 1 * M, 4 * M, 16 * M]
BASE = {"default": [30, 40, 100, 800, 500, 2000], "A": [35, 45, 90, 300, 700, 2100],
        "B": [20, 30, 60, 250, 300, 900], "C": [1, 1, 1, 1, 1, 1],
        "D": [300, 400, 1000, 8000, 5000, 20000]}


def _autotune_child(rank, world, path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{path}/pg", rank=rank, world_size=world)
    try:
        made, calls = [], []

        def make_comm(spec):
            if spec == "C" and rank == 2:
                raise RuntimeError("ncclInvalidUsage")
            made.append(FakeComm(spec))
            return made[-1]

        def measure(comm, nbytes):
            i = SIZES.index(nbytes)
            calls.append((comm.name, i))
            if comm.name == "B" and i == 3 and rank == 1:
                raise RuntimeError("wrong all-reduce result")
            return BASE[comm.name][i] * (1 + 0.01 * rank)  # ranks disagree locally

        default = FakeComm("default")
        table, comms, errors = ns.autotune(None, default, list(BASE), SIZES, make_comm, measure)
        tick = [0]

        def clock():  # every rank's clock runs at a different speed
            tick[0] += 1
            return tick[0] * (1 + rank)

        cut, _, _ = ns.autotune(None, default, ["default", "A"], SIZES, make_comm,
                                lambda c, n: BASE[c.name][SIZES.index(n)], budget_s=12, clock=clock)
        torch.save(dict(table=table, comms=sorted(comms), errors=sorted(errors), calls=calls,
                        destroyed=[c.name for c in made if c.destroyed], cut=cut,
                        bands=ns.build_bands(SIZES, table)), f"{path}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def _spawn(fn, world, tmp_path, monkeypatch):
    import torch.multiprocessing as mp
    for key in [k for k in os.environ if k.startswith("SUFFIX_")]:
        monkeypatch.delenv(key)  # spawned children run sitecustomize
    mp.spawn(fn, args=(world, str(tmp_path)), nprocs=world, join=True)
    return [torch.load(tmp_path / f"r{r}.pt", weights_only=False) for r in range(world)]


def test_autotune_identical_decision_and_failure_fallback(tmp_path, monkeypatch):
    rs = _spawn(_autotune_child, 4, tmp_path, monkeypatch)
    for r in rs[1:]:
        assert r["table"] == rs[0]["table"] and r["bands"] == rs[0]["bands"] and r["cut"] == rs[0]["cut"]
    t = rs[0]["table"]
    assert t["A"][2] == pytest.approx(90 * 1.03)                 # MAX over ranks
    assert all(math.isinf(v) for v in t["B"])                    # errored on rank 1 only -> never picked
    assert all(math.isinf(v) for v in t["C"]) and "C" not in rs[0]["comms"]  # init failed on rank 2
    assert rs[0]["errors"] == ["B", "C"] and rs[1]["errors"] == ["B", "C"]
    assert "C" in rs[0]["destroyed"] and "C" not in [n for n, _ in rs[2]["calls"]]
    assert [i for n, i in rs[0]["calls"] if n == "D"] == [0]     # pruned after 10x the best
    assert rs[0]["bands"] == [(0, 128 * K, "default"), (128 * K, 2 * M, "A"), (2 * M, ns.INF, "default")]
    cut = rs[0]["cut"]  # budget hit after the same cell on every rank, rest unmeasured
    assert math.isfinite(cut["default"][0]) and math.isinf(cut["default"][-1])


# ------------------------------------------------------------------ QAR numerics
def _data(kind, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g)
    if kind == "heavy":
        x = x * torch.where(torch.rand(n, generator=g) < 0.01, 100.0, 1.0)
    if kind == "zeros":
        x[: 3 * qar.BLOCK] = 0
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("mode", ["int8", "fp8"])
@pytest.mark.parametrize("kind", ["normal", "heavy", "zeros"])
@pytest.mark.parametrize("n", [8 * 128 * 6, 6144 * 3 + 77])
def test_qar_within_analytic_bound_and_deterministic(mode, kind, n):
    xs = [_data(kind, n, s) for s in range(8)]
    y = qar.simulate(xs, mode)
    exact = torch.stack([x.float() for x in xs]).sum(0)
    assert y.dtype == torch.bfloat16 and y.shape == xs[0].shape
    assert bool(((y.float() - exact).abs() <= qar.error_bound(xs, mode)).all())
    assert torch.equal(y, qar.simulate(xs, mode))
    if kind == "zeros":
        assert not y[: 3 * qar.BLOCK].isnan().any()


def test_int8_pack_roundtrip_error_is_half_step():
    x = torch.randn(4, 4 * qar.BLOCK) * 3
    back = qar._unpack(qar._pack(x, "int8"), x.shape[1], "int8")
    step = x.view(4, 4, qar.BLOCK).abs().amax(-1, keepdim=True) / 127
    assert bool(((back - x).view(4, 4, qar.BLOCK).abs() <= step / 2 * 1.0001).all())


def test_accuracy_report_runs():
    lines = qar.accuracy_report(world=4, n=6144 * 2)
    assert len(lines) == 3 and all("int8 rel-l2" in line for line in lines)


def _qar_child(rank, world, path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{path}/pg", rank=rank, world_size=world)
    try:
        def a2a(send):
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send)
            return recv

        def ag(mine):
            parts = [torch.empty_like(mine) for _ in range(world)]
            dist.all_gather(parts, mine)
            return torch.cat(parts)

        x = _data("heavy", 6144 * 5 + 3, rank).view(-1, 1)  # padding path
        torch.save({m: qar.qar_all_reduce(x, world, a2a, ag, m) for m in qar.QMAX}, f"{path}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_qar_over_gloo_matches_simulation_bitwise_on_every_rank(tmp_path, monkeypatch):
    rs = _spawn(_qar_child, 4, tmp_path, monkeypatch)
    xs = [_data("heavy", 6144 * 5 + 3, r).view(-1, 1) for r in range(4)]
    for mode in qar.QMAX:
        ref = qar.simulate(xs, mode)
        assert all(torch.equal(r[mode], ref) for r in rs)


# ------------------------------------------------------- bench + boot wiring
def test_matrix_print_path(capsys):
    from suffix_hybrid.tools import ar_bench

    sizes = [48 * K, 1536 * K]
    table = {"default": [42.0, 1595.0], "allreduce:ring/Simple": [50.0, 300.0]}
    qrow = {"int8": {"us": 200.0, "err": 9e-3, "sum": 1.0}, "fp8": {"us": 210.0, "err": 3e-2, "sum": 2.0}}
    res = [{"matrix": {"sizes": sizes, "table": table, "errors": {}}, "qar": [qrow, qrow]}] * 2
    ar_bench._print_matrix(res)
    out = capsys.readouterr().out
    assert "best ring/Simple (5.32x vs default)" in out and "qar-int8" in out and "ranks-identical True" in out
    assert "SUFFIX_NCCL_BANDS=0-272K:default,272K-inf:allreduce:ring/Simple" in out


def test_boot_gates_and_sitecustomize_arm_the_hook(tmp_path):
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["allreduce_matrix_bench"][0][:3] == ["-m", "suffix_hybrid.tools.ar_bench", "--matrix"]
    assert "--qar" in g["_BOOT_GATES"]["allreduce_qar_bench"][0]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SUFFIX_", "NCCL_"))}
    env.update(PYTHONPATH=str(repo), SUFFIX_NCCL_AUTOTUNE="1")
    r = subprocess.run([sys.executable, "-c", "import sitecustomize, sys; "
                        "print(any(type(f).__name__ == '_Finder' for f in sys.meta_path))"],
                       env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.stdout.strip() == "True", r.stderr


def test_autotune_in_setup_keeps_only_used_comms_and_degrades_on_failure(monkeypatch):
    for k in (ns.ENV, ns.BANDS_ENV, ns.QAR_ENV, "NCCL_ALGO", "NCCL_PROTO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ns.AUTOTUNE_ENV, "1")
    monkeypatch.setenv("SUFFIX_NCCL_AUTOTUNE_HIDDEN", "6144")
    made = {}

    def tune(cpu_group, device, default, sizes, cands, budget, iters):
        assert cands[0] == "default" and len(cands) == len(ns.CANDIDATES) and 12 * 6144 * 2 in sizes
        comms = {"default": default, **{c: made.setdefault(c, FakeComm(c)) for c in cands[1:]}}
        table = {c: [100.0] * len(sizes) for c in cands}
        table["allreduce:ring/Simple"] = [50.0 if 1 * M <= s <= 2 * M else 200.0 for s in sizes]
        return table, comms, {}

    monkeypatch.setattr(ns, "tune", tune)
    mod = _fake_module()
    ns.apply(mod)
    cc = mod.CudaCommunicator()
    assert [c for c, v in made.items() if not v.destroyed] == ["allreduce:ring/Simple"]
    cc.all_reduce(torch.ones(768 * K, dtype=torch.bfloat16))  # 1.5 MiB
    assert made["allreduce:ring/Simple"].calls == [1536 * K]
    assert mod.CudaCommunicator(unique_name="ep:0")._suffix_pick is None  # autotune: tp only
    monkeypatch.setattr(ns, "tune", lambda *a: 1 / 0)
    assert mod.CudaCommunicator()._suffix_pick is None  # failure -> static bands (none here)
