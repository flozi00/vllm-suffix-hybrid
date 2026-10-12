# SPDX-License-Identifier: Apache-2.0
"""spec_k_policy on the acceptance measured from live prod logs 2026-09-26."""
from suffix_hybrid.tools import spec_k_policy as P

# Verbatim Rust-frontend line from the prod `glm` pod.
LINE = ("(RustFrontend pid=1230) INFO 09-26 07:00:59 [log_stats.rs:302] SpecDecoding "
        "metrics: Mean acceptance length: 3.04, Accepted throughput: 5.10 tokens/s, "
        "Drafted throughput: 12.50 tokens/s, Accepted: 51 tokens, Drafted: 125 tokens, "
        "Per-position acceptance rate: 0.800, 0.520, 0.400, 0.280, 0.040, "
        "Avg Draft acceptance rate: 40.8%\n")
TINY = ("SpecDecoding metrics: Mean acceptance length: 1.00, Accepted: 0 tokens, "
        "Drafted: 10 tokens, Per-position acceptance rate: 0.000, 0.000, 0.000, 0.000, "
        "0.000, Avg Draft acceptance rate: 0.0%\n")

GLM_ORGANIC = [0.767, 0.499, 0.313, 0.185, 0.108]   # 08-14h, ~900 steps
GLM_BURST = [0.875, 0.722, 0.581, 0.458, 0.357]     # 07h, ~22.8k steps
QWEN_FLASH = [0.888, 0.786, 0.713, 0.641]           # k=4, ~980k steps


def test_parse_and_step_weighting():
    w = P.parse_windows(LINE + TINY)
    assert [s for s, _ in w] == [25.0, 2.0]
    agg = P.aggregate(w)
    # MAL identity: 1 + sum(rates) == 1 + accepted/steps over both windows
    assert abs(sum(agg) - 51 / 27) < 1e-9
    assert abs(agg[0] - 0.8 * 25 / 27) < 1e-9


def test_extend_is_exact_below_measured_and_geometric_above():
    e = P.extend(QWEN_FLASH, 6)
    assert e[:4] == QWEN_FLASH
    r = (0.713 / 0.786 + 0.641 / 0.713) / 2
    assert abs(e[5] / e[4] - r) < 1e-12 and e[4] < e[3]


def test_optimal_k_tracks_acceptance():
    glm = P.PRESETS["glm"]
    k_org = P.best_k(glm, P.extend(GLM_ORGANIC, 8), 1, 8)
    k_burst = P.best_k(glm, P.extend(GLM_BURST, 8), 1, 8)
    assert k_org < k_burst
    assert k_burst == 5            # prod k=5 is right for the high-acceptance mix
    assert 2 <= k_org <= 4         # but too long for organic chat traffic
    # qwen-flash-next runs k=4 with ~0.89 conditional acceptance: go longer
    assert P.best_k(P.PRESETS["qwen-flash"], P.extend(QWEN_FLASH, 10), 1, 10) >= 6


def test_moe_verify_grows_with_k_dense_does_not_at_c1():
    moe, dense = P.PRESETS["gemma"], P.PRESETS["qwen27b"]
    assert moe.verify(1, 8) > 1.3 * moe.verify(1, 0)        # expert union
    assert dense.verify(1, 8) == dense.verify(1, 0)          # bandwidth-bound
    assert abs(moe.verify(1, 0) - 1) < 1e-9 and abs(dense.verify(1, 0) - 1) < 1e-9


def test_v2_fixed_drafter_makes_verify_trim_worse_than_static_k():
    glm, e = P.PRESETS["glm"], P.extend(GLM_ORGANIC, 8)
    static3 = P.speedup(glm, e, 1, 3)
    trim3_of5 = P.speedup(glm, e, 1, 3, draft_k=5)
    assert trim3_of5 < static3


def test_schedule_is_valid_vllm_dynamic_sd_schedule():
    for name, rates in (("glm", GLM_ORGANIC), ("qwen-flash", QWEN_FLASH)):
        sched = P.schedule(P.PRESETS[name], P.extend(rates, 8), 32, 8)
        # vllm/v1/spec_decode/dynamic/utils.py rules
        assert sched[0][0] == 1 and sched[-1][1] == 32
        for (s0, e0, k0), (s1, _, _) in zip(sched, sched[1:]):
            assert s0 <= e0 and s1 == e0 + 1
        assert all(0 <= k <= 8 for _, _, k in sched)
