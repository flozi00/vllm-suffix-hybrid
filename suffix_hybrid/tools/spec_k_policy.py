# SPDX-License-Identifier: Apache-2.0
"""Pick num_speculative_tokens (k) per batch size from MEASURED acceptance.

Offline, stdlib only, no GPU. Input: vLLM "SpecDecoding metrics" log lines
(python loggers.py or the Rust frontend log_stats.rs, same text). Output: the
optimal fixed k and a ``num_speculative_tokens_per_batch_size`` schedule that
vLLM 0.30.0 accepts as-is (config/speculative.py field; validated by
v1/spec_decode/dynamic/utils.py; V2 runner captures one full graph per k,
worker/gpu/cudagraph_utils.py:268-284).

    python -m suffix_hybrid.tools.spec_k_policy --model glm pod.log [...]

Acceptance model: per-position UNCONDITIONAL rates a_i (what vLLM logs),
step-weighted over windows (drafted/k = steps). Chain drafting => a_i for
i <= k does not depend on k, so truncating k is exact; positions beyond the
measured k are extrapolated geometrically with the mean of the last two
conditional ratios (flagged in the output).

Cost model (per step, unit = one plain 1-token decode step at batch 1):
    verify(B, k) = lat + max(bytes, compute)
      bytes   = w_dense + w_exp * distinct_experts(N) / topk   (N = B*(k+1))
      compute = (w_dense + w_exp) * N / ridge
    draft(B, k) = k * draft * max(1, B / draft_ridge)
MoE verify is NOT flat in k: every extra verified token routes to its own
experts, so at low batch the bytes grow with the expert union (``rho``
discounts co-routing of neighbouring tokens). ponytail: presets are
engineering estimates (bytes from checkpoint sizes, lat from c=1 tok/s);
replace with step-profiler ms per (B, k) when a GPU run is available.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

_LINE = re.compile(
    r"Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+?), Avg")


def parse_windows(text: str) -> list[tuple[float, list[float]]]:
    """-> [(steps, per-position unconditional rates)] per logged window."""
    out = []
    for m in _LINE.finditer(text):
        rates = [float(x) for x in m.group(2).split(",")]
        steps = int(m.group(1)) / len(rates)
        if steps > 0:
            out.append((steps, rates))
    return out


def aggregate(windows) -> list[float]:
    """Step-weighted per-position rates (a window of 2 steps must not weigh
    like a window of 2000)."""
    if not windows:
        raise ValueError("no SpecDecoding metrics windows")
    k = len(windows[0][1])
    windows = [w for w in windows if len(w[1]) == k]  # k changed mid-log: keep first
    total = sum(s for s, _ in windows)
    return [sum(s * r[i] for s, r in windows) / total for i in range(k)]


def extend(rates: list[float], kmax: int) -> list[float]:
    """Geometric tail beyond the measured k from the last conditional ratios."""
    rates = list(rates)
    if len(rates) >= kmax:
        return rates[:kmax]
    cond = [rates[0]] + [rates[i] / rates[i - 1] if rates[i - 1] > 0 else 0.0
                         for i in range(1, len(rates))]
    r = sum(cond[-2:]) / len(cond[-2:])
    while len(rates) < kmax:
        rates.append(rates[-1] * r)
    return rates


def expected_tokens(rates: list[float], k: int) -> float:
    """Tokens emitted per step with k drafts = 1 (bonus/correction) + sum a_i."""
    return 1.0 + sum(rates[:k])


@dataclass(frozen=True)
class CostModel:
    lat: float            # fixed latency floor (launch / TP collectives)
    w_dense: float        # non-expert weight + KV bytes time
    w_exp: float = 0.0    # routed-expert bytes time for ONE token
    n_experts: int = 0
    topk: int = 0
    rho: float = 0.7      # co-routing discount for extra tokens (1 = independent)
    ridge: float = 140.0  # tokens/step where compute time == bytes time
    draft: float = 0.05   # one drafter step at batch 1
    draft_ridge: float = 64.0

    def verify(self, b: int, k: int) -> float:
        n = b * (k + 1)
        exp = 0.0
        if self.w_exp:
            n_eff = 1 + (n - 1) * self.rho
            p = self.topk / self.n_experts
            exp = self.w_exp * self.n_experts * (1 - (1 - p) ** n_eff) / self.topk
        bytes_t = self.w_dense + exp
        compute = (self.w_dense + self.w_exp) * n / self.ridge
        return self.lat + max(bytes_t, compute)

    def step(self, b: int, k: int, draft_k: int | None = None) -> float:
        """draft_k: drafter steps actually run. The V2 runner's drafter loop
        is captured at num_speculative_tokens and ignores the per-step K of a
        batch-size schedule (worker/gpu/spec_decode/speculator.py), so a
        schedule only trims VERIFY width there: pass draft_k=kmax."""
        d = k if draft_k is None else (draft_k if k else 0)
        return self.verify(b, k) + d * self.draft * max(1.0, b / self.draft_ridge)


# ponytail: estimates, see module docstring. Sum lat+w_dense+w_exp == 1.
PRESETS = {
    # GLM-5.3 NVFP4 (experts only), TP8+EP over PCIe: latency-bound at c=1;
    # 256 experts top-8; MTP = 1 MoE layer + 154k-vocab head (TP-sharded).
    "glm": CostModel(lat=0.60, w_dense=0.29, w_exp=0.11, n_experts=256, topk=8,
                     draft=0.05),
    # gemma-4-26B-A4B NVFP4 experts + bf16 dense, TP1, 226 tok/s c=1;
    # assistant drafter 4 layers x 1024 + centroid head (~0.25 GB read/step).
    "gemma": CostModel(lat=0.52, w_dense=0.38, w_exp=0.10, n_experts=128, topk=8,
                       draft=0.08),
    # qwen3.8-27b NVFP4 dense+GDN, TP1, 66.9 tok/s c=1 (~15 ms, ~16 GB/step);
    # MTP = 1 bf16 layer (0.7 GB) + bf16 248k lm_head (2.5 GB) => ~0.14/step.
    # GDN verify is a per-token recurrence => lower ridge.
    "qwen27b": CostModel(lat=0.40, w_dense=0.60, ridge=64.0, draft=0.14),
    # qwen3.8-flash-next NVFP4, TP2, 512 experts top-10 + shared.
    "qwen-flash": CostModel(lat=0.60, w_dense=0.25, w_exp=0.15, n_experts=512,
                            topk=10, draft=0.06),
}


def speedup(cm: CostModel, rates: list[float], b: int, k: int,
            draft_k: int | None = None) -> float:
    """Decode throughput at batch b with k drafts vs no speculation."""
    return expected_tokens(rates, k) * cm.verify(b, 0) / cm.step(b, k, draft_k)


def best_k(cm: CostModel, rates: list[float], b: int, kmax: int,
           draft_k: int | None = None) -> int:
    return max(range(kmax + 1),
               key=lambda k: (speedup(cm, rates, b, k, draft_k), -k))


def schedule(cm: CostModel, rates: list[float], max_batch: int, kmax: int,
             v2: bool = True):
    """-> [(start, end, k)] for num_speculative_tokens_per_batch_size, with
    num_speculative_tokens = kmax (v2: drafter always runs kmax steps)."""
    out: list[list[int]] = []
    for b in range(1, max_batch + 1):
        k = best_k(cm, rates, b, kmax, kmax if v2 else None)
        if out and out[-1][2] == k:
            out[-1][1] = b
        else:
            out.append([b, b, k])
    return [tuple(x) for x in out]


def report(name: str, cm: CostModel, rates: list[float], measured_k: int,
           kmax: int, max_batch: int) -> dict:
    ext = extend(rates, kmax)
    rows = {b: {k: round(speedup(cm, ext, b, k), 3) for k in range(kmax + 1)}
            for b in (1, 4, 16, max_batch)}
    return {
        "model": name,
        "rates": [round(r, 3) for r in ext],
        "extrapolated_from": measured_k,
        "E_tokens_per_step": {k: round(expected_tokens(ext, k), 2)
                              for k in range(kmax + 1)},
        "best_k": {b: best_k(cm, ext, b, kmax) for b in rows},
        "speedup": rows,
        "schedule": schedule(cm, ext, max_batch, kmax),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--rates", help="comma list instead of logs (e.g. .8,.6,.4)")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    a = ap.parse_args(argv)
    if a.rates:
        rates = [float(x) for x in a.rates.split(",")]
    else:
        text = "".join(open(f, errors="replace").read() for f in a.logs)
        rates = aggregate(parse_windows(text))
    print(json.dumps(report(a.model, PRESETS[a.model], rates, len(rates),
                            a.kmax, a.max_num_seqs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
