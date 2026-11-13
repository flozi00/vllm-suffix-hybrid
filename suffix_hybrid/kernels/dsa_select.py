# SPDX-License-Identifier: Apache-2.0
"""Fused indexer top-k without materialising the logits: warm-start threshold
select (DESIGN + exact CPU reference; NOT wired into serving, see "Status").

Why. After the row-group body reads each request's K once, the decode indexer
layer's remaining cost is per ROW: the dense fp32 logits row is written once
(4 B / position) and read again by the radix top-k (~3 passes, ~12 B / position),
96 B/position per request at G = 6 rows against 132 B/position of K. At 524288
positions that is the 2 MB x rows logits write plus the top-k reads.

Algorithm (exact, two cheap passes + the existing top-k):
  1. subset  P_r = the row's previous top-k positions (this layer, last step) plus
     EXTRA evenly spaced window positions (m = k + EXTRA, default 2k). Any set of
     >= k window positions is valid; stale ones just loosen the bound. Score only
     P_r (m gathered K rows: 135 KB/row, the fused candidate kernel with block
     size 1).
  2. thr_r = the k-th largest logit of P_r (positions outside the window score
     -inf). It is the k-th largest of a SUBSET of the window, and the k-th largest
     of a subset never exceeds the k-th largest of the whole window, so
     thr_r <= tau_r = the true k-th largest => every true top-k element has
     logit >= tau_r >= thr_r.  Exact for ANY P_r; its quality only sets how many
     survive. (With m = k this is the min of P_r, which ONE stale member drags
     down to the bulk of the row: the extra sample is what keeps the bound tight
     when the warm set has drifted; tests/test_dsa_select.py shows both.)
  3. the scan writes only survivors (logit >= thr_r) packed to the front of a
     [rows, W] buffer behind a per-row cursor (helion_kernels.dsa_group_select),
     so the dense row is never written. M_r = survivors >= k whenever the
     window has >= k positions (P_r itself survives).
  4. vLLM's own persistent top-k runs on the packed row with per-row length M_r
     (rows with M_r <= k return slots 0..M_r-1), and a table lookup maps slots
     back to positions (dsa_candidates.cand_remap with block size 1).

Tie semantics: identical to the stock path up to ties at the k-th value
(persistent_topk resolves them arbitrarily); the SET of values above the k-th
is identical. The survivor ORDER is the programs' atomic race, so tie
resolution may differ run to run, like any other exact-set top-k here.

f32 slack: thr is computed from a different tile schedule than the scan, so the
same element can differ by an f32 ulp-class amount (the lane's TOL = 1e-5 of the
sum of |terms|); WARM_SLACK widens thr by that, which can only ADD survivors.

Cost (bytes per position, G rows of one request, ESTIMATE, docs/dsv41-indexer.md):
  dense:  132/G (K, shared) + 4 (write) + ~12 (top-k reads)        = 38 at G = 6
  select: 132/G + (M/W) * 8 (survivors) + ~0 + 135 KB/row warm gather = 22 at G = 6
so 1.7x fewer bytes per dense layer after row sharing, for ~5 more launches
(warm gather, min, cursor zero, scan, top-k, remap: 6 vs 2-3). Break-even needs
the lab (K4); the prev-top-k hit rate (M/W) is a property of the trained model
and only a serving A/B can measure it.

Status: the reference below and the Helion compaction body are CPU-proven
(tests/test_dsa_select.py, tests/test_helion_dsa_indexer.py); the lab measures
the composite (kernels/dsa_indexer_lab.py, task bench_select). Wiring needs a
per-layer prev-top-k store keyed by row slot (any row mapping is correct, only
the bound's quality suffers) and a patched get_indexer_topk like
dsa_candidates; it waits for the lab's byte/launch break-even.
"""
from __future__ import annotations

import torch

K = 512             # DeepSeek-V4.1 index_topk
EXTRA = 512         # evenly spaced window positions added to the previous top-k
WARM_SLACK = 1e-5   # relative f32 slack on the bound (= the lane's TOL class)


def warm_positions(prev: torch.Tensor, cl: torch.Tensor, extra: int = EXTRA) -> torch.Tensor:
    """[rows, k + extra] int64 positions to score, sorted ascending with every
    repeat replaced by -1 (dead): the previous top-k (-1 = empty slot) and
    `extra` evenly spaced positions of each row's window. DISTINCT positions are
    what makes the k-th largest a bound: the same position counted twice (a
    previous top-k member that is also a sample point, or a short window sampled
    more than once) could lift the k-th largest of the multiset above the true
    one. Sorted order also makes the K gather page-sequential."""
    j = torch.arange(extra, dtype=torch.float64, device=cl.device) + 0.5
    even = (j.reshape(1, -1) * cl.reshape(-1, 1).double() / max(extra, 1)).long()
    pos = torch.cat([prev.long(), even], 1).sort(1).values
    dup = torch.zeros_like(pos, dtype=torch.bool)
    dup[:, 1:] = (pos[:, 1:] == pos[:, :-1]) & (pos[:, 1:] >= 0)
    return torch.where(dup, torch.full_like(pos, -1), pos)


def warm_threshold(sub_logits: torch.Tensor, k: int = K, slack: float = WARM_SLACK) -> torch.Tensor:
    """Lower bound of each row's k-th largest logit from the logits of the warm
    subset ([rows, m >= k] f32, -inf where a position is outside the window):
    its k-th largest (= its minimum when m == k), widened by `slack` x the
    largest finite |logit|. -inf (fewer than k valid positions) stays -inf: the
    whole window survives."""
    kth = sub_logits.topk(k, dim=1).values[:, -1]
    mag = torch.where(torch.isfinite(sub_logits), sub_logits.abs(), torch.zeros_like(sub_logits)).amax(1)
    return kth - slack * mag


def warm_logits(logits: torch.Tensor, cl: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Reference twin of the warm gather: logits[r, pos[r, j]] for 0 <= pos < cl[r],
    else -inf. logits [rows, W] f32 (dense reference), cl [rows], pos int [rows, m]."""
    ok = (pos >= 0) & (pos < cl.reshape(-1, 1))
    got = logits.gather(1, pos.clamp_min(0).long().clamp_max(logits.shape[1] - 1))
    return torch.where(ok, got, torch.full_like(got, float("-inf")))


def compact(logits: torch.Tensor, cl: torch.Tensor, thr: torch.Tensor, g: torch.Generator | None = None):
    """Reference twin of dsa_group_select: survivors (position < cl and logit >=
    thr) packed to the front of (comp_v, comp_i) in a RANDOM order (the atomic
    race), count per row. NaN-poisoned beyond the count like an unwritten buffer."""
    rows, width = logits.shape
    comp_v = torch.full((rows, width), float("nan"))
    comp_i = torch.full((rows, width), -1, dtype=torch.int32)
    cnt = torch.zeros(rows, dtype=torch.int32)
    for r in range(rows):
        n = min(int(cl[r]), width)
        pos = torch.nonzero(logits[r, :n] >= thr[r]).reshape(-1)
        pos = pos[torch.randperm(pos.numel(), generator=g)]
        comp_v[r, :pos.numel()] = logits[r, pos]
        comp_i[r, :pos.numel()] = pos.to(torch.int32)
        cnt[r] = pos.numel()
    return comp_v, comp_i, cnt


def persistent_topk(logits: torch.Tensor, lens: torch.Tensor, k: int = K) -> torch.Tensor:
    """vLLM's decode top-k contract (persistent_topk): rows of length <= k return
    0..len-1 without ranking, else the top-k of [0, len); int32, -1 filled. Reads
    nothing past `lens`."""
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r, n in enumerate(lens.reshape(-1).tolist()):
        n = max(0, min(n, logits.shape[1]))
        out[r, :min(n, k)] = (torch.arange(n) if n <= k else logits[r, :n].topk(k).indices).to(torch.int32)[:k]
    return out


def remap(slots: torch.Tensor, comp_i: torch.Tensor, cl: torch.Tensor) -> torch.Tensor:
    """slot ids -> positions (dsa_candidates.cand_remap with block size 1)."""
    s = slots.long()
    pos = comp_i.gather(1, s.clamp_min(0)).long()
    keep = (s >= 0) & (pos >= 0) & (pos < cl.reshape(-1, 1))
    return torch.where(keep, pos, torch.full_like(pos, -1)).to(torch.int32)


def select(logits: torch.Tensor, cl: torch.Tensor, prev: torch.Tensor, k: int = K,
           g: torch.Generator | None = None, extra: int = EXTRA):
    """The whole pipeline on dense reference logits: (topk positions int32
    [rows, k], survivors per row). Compare with the stock path via vllm_set_ok."""
    thr = warm_threshold(warm_logits(logits, cl, warm_positions(prev, cl, extra)), k)
    comp_v, comp_i, cnt = compact(logits, cl, thr, g)
    return remap(persistent_topk(comp_v, cnt, k), comp_i, cl), cnt
