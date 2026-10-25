# SPDX-License-Identifier: Apache-2.0
"""Warm-start threshold select (suffix_hybrid/kernels/dsa_select.py): the whole
pipeline against the stock top-k contract on adversarial logits, and the one
property its exactness rests on (the bound never exceeds the k-th largest)."""
import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402
from suffix_hybrid.kernels import dsa_select as ds  # noqa: E402

NEG = float("-inf")


def stock(logits, cl, k):
    return ds.persistent_topk(logits, cl, k)


def assert_same_set(logits, cl, prev, k, g=None):
    got, cnt = ds.select(logits, cl, prev, k, g)
    want = stock(logits, cl, k)
    for r in range(logits.shape[0]):
        n = int(cl[r])
        row = logits[r, :n]
        assert di.vllm_set_ok(got[r].tolist(), row, k), (r, n)
        # same values above the k-th, same count
        assert sorted(x for x in got[r].tolist() if x >= 0) != [] or n == 0
        gv = sorted(row[[i for i in got[r].tolist() if i >= 0]].tolist())
        wv = sorted(row[[i for i in want[r].tolist() if i >= 0]].tolist())
        assert gv == wv, r
        assert int(cnt[r]) >= min(n, k)
    return cnt


def prev_from(logits, cl, k, overlap, seed):
    """A previous top-k with `overlap` of the current top-k kept and the rest
    random window positions (a drifting model)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int64)
    for r in range(logits.shape[0]):
        n = int(cl[r])
        if not n:
            continue
        top = logits[r, :n].topk(min(k, n)).indices
        keep = top[torch.randperm(top.numel(), generator=g)[:int(len(top) * overlap)]]
        rest = torch.randperm(n, generator=g)[:k - keep.numel()]
        out[r, :keep.numel() + rest.numel()] = torch.cat([keep, rest])[:k]
    return out


@pytest.mark.parametrize("overlap", [1.0, 0.9, 0.5, 0.0])
def test_select_equals_stock_topk(overlap):
    g = torch.Generator().manual_seed(1)
    k, w = 32, 3000
    logits = torch.randn((5, w), generator=g)
    cl = torch.tensor([3000, 2000, 700, 33, 32])
    prev = prev_from(logits, cl, k, overlap, 4)
    cnt = assert_same_set(logits, cl, prev, k, g)
    # even a stale warm set prunes: the extra sample keeps the bound near tau
    assert int(cnt[0]) < 0.5 * 3000


def test_bound_never_exceeds_the_kth_largest_for_any_subset():
    g = torch.Generator().manual_seed(9)
    k = 16
    for trial in range(40):
        logits = torch.randn((1, 500), generator=g) * (trial + 1)
        cl = torch.tensor([500])
        prev = torch.randperm(500, generator=g)[:k].reshape(1, k)
        thr = ds.warm_threshold(ds.warm_logits(logits, cl, ds.warm_positions(prev, cl, 16)), k)
        tau = logits[0].topk(k).values[-1]
        assert float(thr[0]) <= float(tau)
        # inclusive: the true top-k all survive
        assert bool((logits[0, logits[0].topk(k).indices] >= thr[0]).all())


def test_dead_or_stale_warm_positions_keep_the_whole_window():
    logits = torch.randn((2, 200))
    cl = torch.tensor([200, 150])
    prev = torch.randint(0, 200, (2, 16))
    prev[0, 3] = -1            # an empty slot
    prev[1, 5] = 190           # a position past this row's context (stale slot from another request)
    # m == k: one dead member leaves fewer than k valid positions -> the whole window survives
    thr = ds.warm_threshold(ds.warm_logits(logits, cl, prev), 16)
    assert torch.isneginf(thr).all()
    got, cnt = ds.select(logits, cl, prev, 16, extra=0)
    assert cnt.tolist() == [200, 150]                       # dense, still exact
    got, cnt = ds.select(logits, cl, prev, 16)              # with the sample: bound recovers, still exact
    assert (cnt < torch.tensor([200, 150])).all()
    for r in range(2):
        assert di.vllm_set_ok(got[r].tolist(), logits[r, :int(cl[r])], 16)


@pytest.mark.parametrize("case", ["all_equal", "all_zero_relu", "boundary_ties", "two_levels"])
def test_adversarial_ties(case):
    g = torch.Generator().manual_seed(5)
    k, w = 24, 1500
    if case == "all_equal":
        logits = torch.full((3, w), 0.75)
    elif case == "all_zero_relu":                     # relu clipped almost everywhere: tau = 0
        logits = torch.zeros((3, w))
        logits[:, :10] = torch.rand((3, 10), generator=g) + 1
    elif case == "boundary_ties":                     # 100 positions tie exactly at the k-th value
        logits = torch.rand((3, w), generator=g)
        logits[:, 50:150] = 2.0
        logits[:, :k - 10] = 3.0
    else:
        logits = torch.where(torch.rand((3, w), generator=g) > 0.5, 1.0, 2.0)
    cl = torch.tensor([w, w - 7, 900])
    for overlap in (1.0, 0.3):
        prev = prev_from(logits, cl, k, overlap, 2)
        assert_same_set(logits, cl, prev, k, g)


@pytest.mark.parametrize("n", [0, 1, 7, 24, 25, 26])
def test_tiny_and_short_contexts(n):
    k = 24
    logits = torch.randn((1, 64))
    cl = torch.tensor([n])
    prev = torch.randint(0, max(n, 1), (1, k))
    got, cnt = ds.select(logits, cl, prev, k)
    if n <= k:
        # rows <= k long: every position, no ranking
        assert sorted(x for x in got[0].tolist() if x >= 0) == list(range(n))
        assert int(cnt[0]) == n
    else:
        assert di.vllm_set_ok(got[0].tolist(), logits[0, :n], k)


def test_target_context_524288():
    g = torch.Generator().manual_seed(3)
    k, w = 512, 524288
    logits = torch.randn((2, w), generator=g)
    cl = torch.tensor([w, 400000])
    prev = prev_from(logits, cl, k, 0.9, 6)
    cnt = assert_same_set(logits, cl, prev, k, g)
    assert int(cnt[0]) < 0.25 * w                    # a stale-ish warm start still prunes the row


def test_remap_marks_padding_and_stale_slots():
    comp_i = torch.tensor([[40, 7, 99, 3]], dtype=torch.int32)
    slots = torch.tensor([[2, 0, -1, 3]], dtype=torch.int32)
    assert ds.remap(slots, comp_i, torch.tensor([50])).tolist() == [[-1, 40, -1, 3]]


def test_min_bound_collapses_on_one_stale_member_the_extra_sample_fixes_it():
    g = torch.Generator().manual_seed(2)
    k, w = 64, 20000
    logits = torch.randn((1, w), generator=g)
    cl = torch.tensor([w])
    prev = prev_from(logits, cl, k, 0.9, 1)                 # 10 % random stale members
    _, dense = ds.select(logits, cl, prev, k, extra=0)
    _, tight = ds.select(logits, cl, prev, k, extra=k)
    assert int(dense[0]) > 0.7 * w and int(tight[0]) < 0.2 * w


def test_repeated_positions_are_not_counted_twice():
    """A previous top-k member that is also a sample point (or a short window
    sampled repeatedly) must not lift the bound above the true k-th largest."""
    g = torch.Generator().manual_seed(1)
    k = 32
    logits = torch.randn((4, 3000), generator=g)
    cl = torch.tensor([3000, 3000, 20, 40])
    top = logits[0].topk(k).indices
    prev = torch.stack([top, top, torch.arange(k) % 20, torch.arange(k)])      # exact top-k, repeats
    pos = ds.warm_positions(prev, cl, 512)
    for r in range(4):
        live = pos[r][pos[r] >= 0]
        assert live.numel() == live.unique().numel()
    live0 = [x for x in pos[0].tolist() if x >= 0]
    assert live0 == sorted(live0)
    thr = ds.warm_threshold(ds.warm_logits(logits, cl, pos), k)
    for r in range(4):
        n = int(cl[r])
        tau = logits[r, :n].topk(k).values[-1] if n >= k else float("-inf")
        assert float(thr[r]) <= float(tau), r
