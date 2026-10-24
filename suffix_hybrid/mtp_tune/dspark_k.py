# SPDX-License-Identifier: Apache-2.0
"""k(c) HYPOTHESIS for DeepSeek-V4.1-Flash + DSpark (block 5) on 8 x RTX PRO 6000. CPU only.

Reuses suffix_hybrid.tools.spec_k_policy (same acceptance + cost model as the GLM schedule
[[1,2,5],[3,5,3],[6,32,2]]); only the inputs are new. Everything below is an ESTIMATE until
a step profile of the live pool replaces the cost constants:

  rates   per-position UNCONDITIONAL acceptance from the bring-up log (boot5.log line 2472,
          one window: 39 steps, 195 drafted, 129 accepted, mean length 4.31). 39 steps is a
          smoke sample: the standard error of position 1 alone is ~0.05, and the prompt mix is
          unknown. ORGANIC_HYP scales it by the GLM organic/burst ratio of the same positions
          (test_spec_k_policy.py) to show what a lower-acceptance mix does to the table.
  cost    unit = one plain 1-token decode step at batch 1.
          lat 0.55 / dense 0.30 / experts 0.15: 81 all-reduces + 43 eager attention regions
          (breakable graphs) + mHC/sampler launches dominate c1 (GLM g4 profile: 25-30% of a
          step is all-reduce incl. rank waiting); 384 experts top-6, EP8.
          DSpark drafts ALL 5 positions in ONE parallel pass (3 decoder layers + head), and the
          V2 runner ignores the per-step K in the drafter (dflash/speculator.py: the parameter
          is unused), so the draft cost does not depend on K: spec_k_policy charges
          draft_k * draft, hence draft=0.022 with draft_k=5 -> ~0.11 unit at B=1 (3 of 43
          layers + logits gather + Markov head), growing past B~32 (160 query tokens).
  schedule only trims the VERIFY width (scheduler: num_spec_tokens_to_schedule).

    python -m suffix_hybrid.mtp_tune.dspark_k
"""
from __future__ import annotations

import json
import sys

from suffix_hybrid.tools import spec_k_policy as P

BRINGUP_RATES = [0.872, 0.769, 0.641, 0.590, 0.436]
_GLM_ORGANIC = [0.767, 0.499, 0.313, 0.185, 0.108]
_GLM_BURST = [0.875, 0.722, 0.581, 0.458, 0.357]
ORGANIC_HYP = [round(m * o / b, 3) for m, o, b in zip(BRINGUP_RATES, _GLM_ORGANIC, _GLM_BURST)]
KMAX, MAX_NUM_SEQS = 5, 32

COST = {  # name -> CostModel; "central" is the one the table below quotes
    "central": P.CostModel(lat=0.55, w_dense=0.30, w_exp=0.15, n_experts=384, topk=6,
                           draft=0.022, draft_ridge=32.0),
    "latency-bound": P.CostModel(lat=0.70, w_dense=0.20, w_exp=0.10, n_experts=384, topk=6,
                                 draft=0.022, draft_ridge=32.0),
    "bandwidth-bound": P.CostModel(lat=0.40, w_dense=0.35, w_exp=0.25, n_experts=384, topk=6,
                                   draft=0.022, draft_ridge=32.0),
}


def schedule(rates, cost: str = "central"):
    """[(first batch, last batch, k)] for num_speculative_tokens_per_batch_size."""
    return P.schedule(COST[cost], rates, MAX_NUM_SEQS, KMAX, v2=True)


def schedule_json(rates, cost: str = "central") -> str:
    """The value of --speculative-config.num_speculative_tokens_per_batch_size."""
    return json.dumps([list(r) for r in schedule(rates, cost)], separators=(",", ":"))


def speedup_row(rates, batch: int, cost: str = "central") -> dict[int, float]:
    """Modelled decode speedup vs no speculation, per verified k, drafter always 5 wide."""
    return {k: round(P.speedup(COST[cost], rates, batch, k, KMAX), 3) for k in range(KMAX + 1)}


def main() -> int:
    for name, rates in (("bring-up", BRINGUP_RATES), ("organic-hyp", ORGANIC_HYP)):
        print(f"== {name}: rates {rates}  E[tokens/step] by k: "
              f"{[round(P.expected_tokens(rates, k), 2) for k in range(KMAX + 1)]}")
        for cost in COST:
            print(f"   {cost:16s} {schedule_json(rates, cost)}")
        print("   central speedup(k) by batch:")
        for b in (1, 2, 4, 8, 16, 32):
            row = speedup_row(rates, b)
            print(f"     c={b:<2d} {row}  best k={max(row, key=lambda k: (row[k], -k))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
