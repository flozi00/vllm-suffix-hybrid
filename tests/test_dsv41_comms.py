"""DeepSeek-V4.1-Flash comms lane (docs/dsv41-comms.md): everything here runs on CPU.

Message sizes, host all-reduce routing at hidden 5120, the estimated band table, the preset,
the nccl_split anchors against the exact vLLM pin (39983a5e), and the k(c) hypothesis table.
"""
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from suffix_hybrid import host_ar
from suffix_hybrid import nccl_split as ns
from suffix_hybrid.mtp_tune import dspark_k as dk
from suffix_hybrid.tools import spec_k_policy as P

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "tests/fixtures/vllm_39983a5e/cuda_communicator.py"  # vllm@39983a5e, verbatim
HIDDEN, VOCAB, WORLD = 5120, 129280, 8
DECODE_TOKENS = tuple(6 * c for c in (1, 2, 4, 8, 16, 32))  # verify rows at k=5
K = 1 << 10


# ------------------------------------------------------------- message sizes
def test_every_v41_message_is_host_ar_legal_and_round_counts():
    for tokens in (1, 2, 3, 4, 5, 6, 12, 24, 48, 96, 192, 8192, 16384):
        n = tokens * HIDDEN
        assert n % 8 == 0 and n % host_ar.Q8_GROUP == 0   # supports() and int8 groups
    assert (host_ar.ROUND_BYTES // 2) % (WORLD * host_ar.BLOCK) == 0
    assert host_ar.rounds(192 * HIDDEN) == 1                # c32 k=5 decode: one round
    assert host_ar.rounds(8192 * HIDDEN) == 20               # --max-num-batched-tokens 8192: 80 MiB
    assert host_ar.rounds(16384 * HIDDEN) == 40              # 160 MiB
    assert VOCAB % WORLD == 0 and (VOCAB // WORLD) % 8 == 0  # logits shard rows gather as int16


def test_route_policy_at_hidden_5120():
    one_max = 48 * K
    kinds = {t: host_ar.route(t * HIDDEN, one_max, 1 << 62) for t in (1, 4, 5, 6, 48, 192, 8192)}
    # 4 tokens = 40 KiB one-shot; the 5-token DSpark pass (50 KiB) and the 6-token verify are two-shot
    assert kinds == {1: "one", 4: "one", 5: "two", 6: "two", 48: "two", 192: "two", 8192: "two"}
    assert host_ar.route(6 * HIDDEN, 64 * K, 1 << 62) == "one"          # the 64K arm
    # int8 transport only from its threshold, never one-shot, needs 128-element groups
    assert host_ar.route(48 * HIDDEN, one_max, 256 * K) == "q8"
    assert host_ar.route(6 * HIDDEN, one_max, 256 * K) == "two"
    assert host_ar.route(4 * HIDDEN, one_max, 0) == "one"
    assert host_ar.route(48 * HIDDEN + 8, one_max, 0) == "two"           # not a multiple of 128
    assert host_ar.route(6 * HIDDEN, one_max, 1 << 62, oneshot=False) == "two"
    assert host_ar.route(6 * HIDDEN, one_max, 1 << 62, oneshot=True) == "one"


# -------------------------------------------------- estimated size/band table
def test_estimates_reproduce_the_glm_bench_exactly_at_its_own_sizes():
    for kind, pts in ns.GLM_BENCH.items():
        for tokens, us in pts.items():
            assert ns.estimate_us(kind, tokens * ns.GLM_HIDDEN * 2) == pytest.approx(us)
    for nbytes, us in ns.NCCL_DEFAULT_US.items():
        assert ns.estimate_us("default", nbytes) == pytest.approx(us)


def test_estimates_scale_by_bytes_and_are_monotone():
    rows = ns.estimate_table(HIDDEN)
    assert [r["tokens"] for r in rows] == list(ns.TOKENS) and rows[1]["bytes"] == 6 * HIDDEN * 2
    for kind in ("nccl", "har", "q8"):
        ys = [r[kind] for r in rows if r[kind] is not None]
        assert ys == sorted(ys)
    # hidden 5120 is 5/6 of the bytes of hidden 6144 at the same tokens -> faster, never slower
    glm = ns.estimate_table(ns.GLM_HIDDEN)
    assert all(a["har"] <= b["har"] + 1e-9 for a, b in zip(rows, glm))
    assert rows[0]["q8"] is None and rows[ns.TOKENS.index(96)]["q8"] is not None


def test_v41_bands_are_host_ar_everywhere_and_the_preset_says_so():
    assert ns.bands_str(ns.estimate_bands(HIDDEN)) == "0-inf:har"
    assert ns.bands_str(ns.estimate_bands(ns.GLM_HIDDEN)) == "0-inf:har"     # = glm53-sm120
    # host all-reduce beats the best NCCL algorithm at every estimated V4.1 size (GLM measured the same)
    assert all(r["har"] < r["nccl"] for r in ns.estimate_table(HIDDEN))
    ns.parse_bands(ns.bands_str(ns.estimate_bands(HIDDEN)))


def test_table_cli_prints_the_estimate_and_the_env(capsys):
    assert ns._table_main(["table", "--hidden", "5120"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ESTIMATE") and "SUFFIX_NCCL_BANDS=0-inf:har" in out


# -------------------------------------------------------------------- preset
def _env_of(preset, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SUFFIX_", "NCCL_"))}
    env.update(PYTHONPATH=f"{ROOT}{os.pathsep}{ROOT / 'sm120'}", **({"SUFFIX_PRESET": preset} if preset else {}),
               **(extra or {}))
    code = ("import os; print('|'.join(f'{k}={os.environ.get(k)}' for k in ("
            "'SUFFIX_NCCL_BANDS','SUFFIX_HAR_ONESHOT_MAX_BYTES','SUFFIX_HAR_ALLGATHER',"
            "'SUFFIX_HAR_Q8_MIN_BYTES','SUFFIX_DSV41_SM120_BLOCK64','SUFFIX_DSV41_ENGRAM_MXFP4')))")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return dict(kv.split("=", 1) for kv in r.stdout.splitlines()[-1].split("|"))


def test_comms_preset_is_dsv41_plus_host_collectives_and_never_lossy():
    got = _env_of("dsv41-sm120-comms")
    assert got == {"SUFFIX_NCCL_BANDS": "0-inf:har", "SUFFIX_HAR_ONESHOT_MAX_BYTES": "48K",
                   "SUFFIX_HAR_ALLGATHER": "1", "SUFFIX_HAR_Q8_MIN_BYTES": "None",
                   "SUFFIX_DSV41_SM120_BLOCK64": "1", "SUFFIX_DSV41_ENGRAM_MXFP4": "1"}


def test_plain_dsv41_preset_is_unchanged_and_explicit_env_wins():
    plain = _env_of("dsv41-sm120")
    assert plain["SUFFIX_NCCL_BANDS"] == "None" and plain["SUFFIX_HAR_ALLGATHER"] == "None"
    assert plain["SUFFIX_DSV41_SM120_BLOCK64"] == "1"
    assert _env_of("dsv41-sm120-comms", {"SUFFIX_HAR_ONESHOT_MAX_BYTES": "64K"})["SUFFIX_HAR_ONESHOT_MAX_BYTES"] == "64K"


def test_dsv41_boot_gates_are_registered():
    src = (ROOT / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["host_ar_bench_dsv41"][0] == ["-m", "suffix_hybrid.host_ar", "bench", "--hidden", "5120",
                                                          "--vocab", "129280"]
    assert g["_BOOT_GATES"]["allreduce_matrix_bench_dsv41"][0][-2:] == ["--hidden", "5120"]


# ----------------------------------------------- anchors vs the exact vLLM pin
class FakeComm:
    def __init__(self, name):
        self.name, self.calls = name, []

    def all_reduce(self, x):
        self.calls.append(x.numel() * x.element_size())
        return x * 8


def _pinned_module():
    class CudaCommunicator:
        def __init__(self, unique_name="tp:0", world_size=8):
            self.unique_name, self.world_size, self.rank = unique_name, world_size, 0
            self.cpu_group, self.device = None, "cpu"
            self.pynccl_comm = FakeComm("default")
            self.pynccl_comm.disabled, self.pynccl_comm.world_size = False, world_size
            self.fi_ar_comm = self.qr_comm = self.fi_pcie_ipc_ar_comm = None
            self.aiter_ar_comm = self.ca_comm = self.symm_mem_comm = None
            self.use_aiter_allreduce = False

    return types.SimpleNamespace(
        __file__=str(PIN), torch=torch, CudaCommunicator=CudaCommunicator,
        envs=types.SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        current_platform=types.SimpleNamespace(is_rocm=lambda: False),
        should_nccl_symm_mem_allreduce=lambda *a: False, should_nccl_symm_mem_ag_rs=lambda: False)


def test_pin_has_both_anchors_exactly_once():
    text = PIN.read_text()
    assert "Using %s all-reduce backends (in dispatch order)" in text   # the dispatch the boot log printed
    assert "_suffix_pick" in ns.patch_all_reduce(text)
    assert "_suffix_hag" in ns.patch_all_gather(text)


def test_pin_routes_by_bytes_and_host_gathers_with_the_comms_env(monkeypatch):
    for k in (ns.ENV, ns.AUTOTUNE_ENV, ns.QAR_ENV, "NCCL_ALGO", "NCCL_PROTO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ns.BANDS_ENV, "0-inf:har")
    monkeypatch.setenv(ns.HAR_AG_ENV, "1")
    made = {}

    def make(spec, g, d):
        c = made.setdefault(spec, FakeComm(spec))
        c.supports = lambda x: x.dtype == torch.bfloat16 and x.numel() % 8 == 0
        c.supports_gather = lambda x: x.element_size() in (2, 4)
        c.all_gather_into = lambda out, x: out.copy_(x.repeat(8, *([1] * (x.dim() - 1))))
        return c

    monkeypatch.setattr(ns, "make_comm", make)
    mod = _pinned_module()
    ns.apply(mod)
    cc = mod.CudaCommunicator()
    for tokens in (1, 6, 192):
        cc.all_reduce(torch.ones(tokens, HIDDEN, dtype=torch.bfloat16))
    cc.all_reduce(torch.ones(6, HIDDEN, dtype=torch.float32))   # not bf16 -> the default communicator
    assert made["har"].calls == [t * HIDDEN * 2 for t in (1, 6, 192)]
    assert cc.pynccl_comm.calls == [6 * HIDDEN * 4]
    eng = cc.all_gather(torch.arange(6 * 256, dtype=torch.bfloat16).view(6, 1, 256), dim=1)  # Engram heads
    assert eng.shape == (6, 8, 256)
    logits = cc.all_gather(torch.zeros(6, VOCAB // WORLD, dtype=torch.bfloat16), dim=-1)
    assert logits.shape == (6, VOCAB)


def test_route_log_names_each_band_and_each_fallback_once():
    lines = []
    host = FakeComm("har")
    host.supports = lambda x: x.dtype == torch.bfloat16
    pick = ns.make_pick([(0, ns.INF, host)], log=lines.append)
    default = FakeComm("default")
    for tokens in (1, 6, 192, 6):
        assert pick(torch.empty(tokens, HIDDEN, dtype=torch.bfloat16), default) is host
    for _ in range(3):
        assert pick(torch.empty(6, HIDDEN, dtype=torch.float32), default) is default
    assert len(lines) == 2
    assert lines[0].startswith("route [0, inf) -> FakeComm") and "bfloat16" in lines[0]
    assert lines[1].startswith("FALLBACK to the default communicator") and "float32" in lines[1]
    quiet = ns.make_pick([(0, ns.INF, host)])   # log=None (ranks > 0): no output, same routing
    assert quiet(torch.empty(8, dtype=torch.bfloat16), default) is host


# ------------------------------------------------------------ k(c) hypothesis
def test_bringup_rates_reproduce_the_logged_mean_acceptance_length():
    assert P.expected_tokens(dk.BRINGUP_RATES, 5) == pytest.approx(4.31, abs=0.005)   # boot5.log:2472
    assert sum(dk.BRINGUP_RATES) == pytest.approx(129 / 39, abs=0.005)               # 129 accepted / 39 steps
    assert all(a > b for a, b in zip(dk.BRINGUP_RATES, dk.BRINGUP_RATES[1:]))


@pytest.mark.parametrize("rates", [dk.BRINGUP_RATES, dk.ORGANIC_HYP])
@pytest.mark.parametrize("cost", list(dk.COST))
def test_schedules_are_valid_vllm_dynamic_sd_schedules(rates, cost):
    sched = dk.schedule(rates, cost)
    assert sched[0][0] == 1 and sched[-1][1] == dk.MAX_NUM_SEQS    # utils.py: starts at 1, covers max_num_seqs
    for (s0, e0, _), (s1, _, _) in zip(sched, sched[1:]):
        assert s0 <= e0 and s1 == e0 + 1
    assert all(1 <= k <= dk.KMAX for _, _, k in sched)
    assert dk.schedule_json(rates, cost).startswith("[[1,")


def test_k_follows_acceptance_not_the_other_way():
    hi, lo = dk.schedule(dk.BRINGUP_RATES), dk.schedule(dk.ORGANIC_HYP)
    kat = lambda s, b: next(k for a, e, k in s if a <= b <= e)  # noqa: E731
    assert kat(hi, 1) == 5                                            # high acceptance: full block at c1
    assert all(kat(hi, b) >= 4 for b in range(1, 33))                # ... and never below 4 anywhere
    assert all(kat(lo, b) <= kat(hi, b) for b in range(1, 33))       # lower acceptance never asks for more
    assert kat(lo, 8) <= 3 and kat(lo, 16) <= 3
    row = dk.speedup_row(dk.BRINGUP_RATES, 8)
    assert row[0] == 1.0 and row[4] == pytest.approx(row[5], abs=0.02)   # k=4 vs 5 is a ~1 % call at c8


def test_cost_model_sums_to_one_unit_step():
    for cm in dk.COST.values():
        assert math.isclose(cm.lat + cm.w_dense + cm.w_exp, 1.0)
