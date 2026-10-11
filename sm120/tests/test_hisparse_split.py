# SPDX-License-Identifier: Apache-2.0
"""Residency split (hisparse_mtp_patch, 0.31.1 nightly, MTP k > 3 on SM120): the
pass arithmetic, the nightly anchors, and a model of hisparse_resolve_residency's
LRU phases (hisparse_kernels.cu:276-747) proving that resolving a request's
rows in passes never evicts a slot an earlier pass of the same step resolved."""
import random
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "sm120")]
import hisparse_mtp_patch as HS  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def resolve(rows, lru, dgi):
    """One kernel launch for one request: rows = per-row host ids (unique per row).
    Returns per-row slots. Phases: union by first (row, k) reference; hits = slots
    holding a union id; misses take evictables oldest-first in first-reference
    order; LRU <- remaining evictables, miss slots, hits (phase 5)."""
    first = {}
    for r, ids in enumerate(rows):
        for i, g in enumerate(ids):
            first.setdefault(g, (r, i))
    hits = [s for s in lru if dgi[s] in first]
    evict = [s for s in lru if dgi[s] not in first]
    slot_of = {dgi[s]: s for s in hits}
    misses = sorted((g for g in first if g not in slot_of), key=first.get)
    assert len(misses) <= len(evict), "union above the hot buffer"
    for m, g in enumerate(misses):
        dgi[evict[m]] = g
        slot_of[g] = evict[m]
    lru[:] = evict[len(misses):] + evict[:len(misses)] + hits
    return [[slot_of[g] for g in ids] for ids in rows]


@pytest.mark.parametrize("q,r", [(6, 4), (6, 2), (5, 3), (4, 1)])
def test_passes_never_evict_an_earlier_pass_of_the_step(q, r):
    rnd = random.Random(q * 10 + r)
    top_k, hot = 8, (q + 1) * 8  # upstream default device_buffer_size = (q + 1) * top_k
    lru, dgi = list(range(hot)), [-1] * hot
    for _step in range(300):
        base = rnd.sample(range(64), top_k)
        rows = []
        for _ in range(q):  # MTP rows: mostly the same positions, some new
            ids = list(dict.fromkeys(g if rnd.random() < 0.7 else rnd.randrange(64) for g in base))
            ids += [g for g in rnd.sample(range(64), top_k * 2) if g not in ids][:top_k - len(ids)]
            rows.append(ids)
        slots = []
        for p in range(0, q, r):  # the split: one launch per pass of r rows
            slots += resolve(rows[p:p + r], lru, dgi)
        for ids, ss in zip(rows, slots):  # what attention reads after every pass
            assert [dgi[s] for s in ss] == ids


def test_pass_rows_for_glm_on_sm120():
    t, hot = 2048, 7 * 2048
    assert HS.residency_smem_bytes(6 * t, hot) == 132388  # the nightly's refusal
    assert HS.residency_pass_rows(6, t, hot, 101376) == 4  # -> 2 passes
    assert HS.residency_pass_rows(4, t, 5 * 2048, 101376) == 4  # k=3 fits: upstream
    assert HS.residency_pass_rows(6, t, hot, 1 << 20) == 6
    assert HS.residency_pass_rows(1, 1 << 20, hot, 101376) == 0


def test_pass_of_each_row_follows_its_request_run():
    ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32)
    rows = torch.arange(ids.numel(), dtype=torch.int32)
    got = HS.residency_passes(ids, rows, 4, 2).tolist()
    assert got == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_split_anchors_nightly_only():
    assert HS.check_split_source((FIX / "vllm_0.31.1rc1.dev273/hisparse_runtime.py").read_text())
    assert not HS.check_split_source((FIX / "vllm_0.30.0/hisparse_runtime.py").read_text())
    drifted = (FIX / "vllm_0.31.1rc1.dev273/hisparse_runtime.py").read_text().replace(
        "        shared = group.shared_topk\n", "        shared = group.topk\n")
    with pytest.raises(HS.PatchDriftError, match="resolve_shared"):
        HS.check_split_source(drifted)


def test_oracle_is_a_boot_gate_and_fails_closed_before_serving():
    src = (REPO / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["hisparse_resolver_oracle"][0] == ["-m", "hisparse_mtp_patch.split_oracle"]
    block = src[src.index('"SUFFIX_SM120_HISPARSE_SPLIT_ORACLE_DONE")'):]
    assert "hisparse_mtp_patch.split_oracle" in block and "_rc not in (0, 3)" in block
    assert "raise SystemExit" in block[:block.index("# NOTE: the WARM-START")]
