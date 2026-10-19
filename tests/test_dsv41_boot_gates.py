"""Every V4.1 boot gate in sitecustomize._BOOT_GATES must point at a module that exists in this tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ("mxfp8_capture_replay", "mxfp8_capture_replay_sem", "fp4mma_capture_replay", "fp4mma_capture_replay_bytes", "moe_capture_replay", "dsv41_moe_shared_bench",
         "dsa_indexer_capture_replay", "dsv41_attn_capture_replay", "dsv41_hc_capture_replay",
         "dsv41_router_capture_replay", "dsv41_swiglu_capture_replay", "dsv41_engram_capture_replay",
         "host_ar_bench_dsv41", "allreduce_matrix_bench_dsv41", "har_l2_prefetch_bench_dsv41",
         "dp_ag_rs_bench_dsv41")


def test_v41_gates_point_at_real_modules():
    src = (ROOT / "sitecustomize.py").read_text()
    ns: dict = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    for name in GATES:
        argv, env = ns["_BOOT_GATES"][name]
        assert argv[0] == "-m", name
        assert (ROOT / (argv[1].replace(".", "/") + ".py")).is_file(), f"{name}: {argv[1]} not in tree"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), name
