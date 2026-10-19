"""DeepSeek-V4.1-Flash layout study (suffix_hybrid/comm_layout.py, docs/dsv41-comms.md section 6). CPU only."""
import pytest

from suffix_hybrid import comm_layout as L

TP8, TP4, TP2, DP8 = (8, 1), (4, 2), (2, 4), (1, 8)


def test_dense_stream_matches_the_lane_d_inventory_and_grows_with_replication():
    assert L.stream_mb(*TP8) == pytest.approx(2621, abs=1)       # docs/dsv41-dense.md section 2 total
    assert L.stream_mb(*DP8) == pytest.approx(12372, abs=2)       # sharded tensors x8, replicated unchanged
    assert L.stream_mb(*TP8) < L.stream_mb(*TP4) < L.stream_mb(*TP2) < L.stream_mb(*DP8)
    # sequence parallelism (tp > 1 and dp > 1) replicates the shared expert on the TP group
    assert L.stream_mb(4, 2) > L.stream_mb(4, 1)
    assert L.stream_mb(1, 8, wo_a_fp8=True) == pytest.approx(12372 - (361 - 186) * 8, abs=2)
    # per step at 1357 GB/s: 1.9 ms (TP8) vs 9.1 ms (DP8)
    assert L.step(*TP8, 1)["weights_ms"] == pytest.approx(1.93, abs=0.02)
    assert L.step(*DP8, 1)["weights_ms"] == pytest.approx(9.12, abs=0.05)


def test_extra_weights_and_aggregate_kv_capacity():
    assert L.extra_weights_gib(*TP8) == 0.0
    assert L.extra_weights_gib(*DP8) == pytest.approx(9.5, abs=0.2)
    per, tokens = L.kv_capacity(*DP8)
    base, base_tokens = L.kv_capacity(*TP8)
    assert base == 40.0 and base_tokens == pytest.approx(11.7e6, rel=0.02)   # boot5: 10.87M tokens at 37.17 GiB
    assert per == pytest.approx(30.5, abs=0.3)
    assert 5.5 < tokens / base_tokens < 6.5                                  # ~6x, not 8x: the replicas' weights
    # a 524288-token request needs ~1.8 GiB of KV: one replica's pool is never the limit
    assert 524288 / L.KV_TOKENS_PER_GIB < 2.0 < per


def test_tp8_comm_reproduces_the_section_3_per_step_estimate():
    s = L.step(*TP8, 1)
    assert s["comm_ms"] == pytest.approx(1.9, abs=0.15)      # 81 + 7 all-reduces ~1.66 ms + logits + Engram
    assert L.step(*TP8, 32)["comm_ms"] == pytest.approx(25.3, abs=1.0)


def test_dp_attention_loses_at_low_concurrency_and_wins_only_from_the_high_teens():
    for tp, dp in (TP4, TP2, DP8):
        assert L.delta_ms(tp, dp, 1) > 2.0 and L.delta_ms(tp, dp, 8) > 1.0, L.name(tp, dp)
        assert L.delta_ms(tp, dp, 32) < -4.0, L.name(tp, dp)
    assert L.delta_ms(*DP8, 1) == pytest.approx(9.3, abs=0.6)
    assert L.delta_ms(*DP8, 32) == pytest.approx(-7.8, abs=0.8)
    be = {L.name(tp, dp): L.break_even(tp, dp) for tp, dp in (TP4, TP2, DP8)}
    assert 14 <= be["TP4xDP2+EP8"] <= be["TP2xDP4+EP8"] <= be["DP8+EP8"] <= 22
    # the lane D FP8 wo_a does not rescue c1 / c8
    assert L.delta_ms(*DP8, 1, wo_a_fp8=True) > 7 and L.delta_ms(*DP8, 8, wo_a_fp8=True) > 2


def test_c1_loss_survives_a_tp8_dense_stream_far_below_its_byte_floor():
    # lane D: small-N CUTLASS tiles run below the roofline; even 3x the floor leaves DP8 slower at c1
    deltas = [L.delta_ms(*DP8, 1, small_gemm_x=x) for x in (1.0, 1.5, 2.0, 3.0)]
    assert deltas == sorted(deltas, reverse=True) and deltas[-1] > 5.0
    assert L.delta_ms(*DP8, 8, small_gemm_x=3.0) < L.delta_ms(*DP8, 8) < 4.5     # c8 only approaches a tie
    assert L.step(*TP8, 1, small_gemm_x=2.0)["weights_ms"] == pytest.approx(2 * L.step(*TP8, 1)["weights_ms"])
    assert L.step(*DP8, 1, small_gemm_x=2.0)["weights_ms"] == L.step(*DP8, 1)["weights_ms"]


def test_slower_reduce_scatter_than_modelled_moves_break_even_up_or_away():
    slow = (L.ar_host, lambda b: 2 * L.rs_nccl(b), lambda b: 2 * L.ag_nccl(b))   # silicon says RS / AG cost twice
    assert L.delta_ms(*DP8, 32, costs=slow) > L.delta_ms(*DP8, 32)
    assert (L.break_even(*DP8, costs=slow) or 99) > L.break_even(*DP8)
    fast = (L.ar_host, lambda b: 0.5 * L.rs_nccl(b), lambda b: 0.5 * L.ag_nccl(b))
    assert L.break_even(*DP8, costs=fast) < L.break_even(*DP8)


def test_the_c32_go_no_go_rule_of_the_doc():
    # docs/dsv41-comms.md 11.6: DP8 beats TP8 at c32 while its layer (RS + AG at 192 tokens) stays <= ~420 us
    for factor, wins in ((1.7, True), (1.9, False)):
        costs = (L.ar_host, lambda b, f=factor: f * L.rs_nccl(b), lambda b, f=factor: f * L.ag_nccl(b))
        assert (L.delta_ms(*DP8, 32, costs=costs) < 0) is wins
        assert (L.layer_us(1, 8, 24, costs) <= 422) is wins
    assert L.layer_us(1, 8, 24) == pytest.approx(237, abs=2) and L.layer_us(8, 1, 192) == pytest.approx(541, abs=2)


def test_dp_pads_every_replica_to_the_busiest_one():
    # c = 1 and c = 8 both put (at most) one request on every replica: the MoE sees 8 x 6 padded rows either way
    assert L.step(*DP8, 1)["comm_ms"] == pytest.approx(L.step(*DP8, 8)["comm_ms"], rel=1e-9)
    assert L.step(*DP8, 9)["comm_ms"] > L.step(*DP8, 8)["comm_ms"]               # 2 requests on one replica


def test_table_and_bench_gate_are_wired(capsys):
    assert L.main(["table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ESTIMATE") and "DP8+EP8" in out and "break-even c" in out
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["dp_ag_rs_bench_dsv41"] == (["-m", "suffix_hybrid.comm_layout", "bench"], {})
    assert L.TOTAL_TOKENS == (48, 96, 192) and all(t % L.GPUS == 0 for t in L.TOTAL_TOKENS)
