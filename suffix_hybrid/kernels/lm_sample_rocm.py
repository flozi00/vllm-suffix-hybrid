# SPDX-License-Identifier: Apache-2.0
"""lm_head candidates and top-k / top-p sampling masks on the MI350P (ROCm), rows split
across programs.

1. row_topk(y, k): the k largest entries of each bf16 row in torch.topk's order (ATen's
   bf16 radix key, ties to the lowest index), in two launches: grid (rows, 4096-wide
   chunks) picks each chunk's k by a 16-step bisection on the key, then one program per
   row picks the k of the chunks' candidates the same way. mxfp4_lm_head's rescore uses
   it instead of y.topk(64) (ATen's multi-kernel radix select).

2. SUFFIX_ROCM_TOPK_TOPP=1: vLLM's apply_top_k_top_p_triton (Qrita, vllm/v1/sample/ops/
   topk_topp_triton.py @81198e97) for batches with a top-k. Stock runs one program per
   row, and that program sweeps the 248320-wide row three times around its pivot
   search; at c1 that is 5 programs on 128 CUs. Here:
     _qrita_gather_kernel  grid (rows, splits): stock's zeroth pass (block-0 statistics,
                           the outlier pivot) and its first pass on the split's tiles:
                           max, finite min, finite count, the outliers (value + index)
                           into a per-split segment;
     _qrita_search_kernel  stock's grid and row loop: merges the splits, compacts the
                           segments into stock's per-program buffer in index order, runs
                           stock's top-k search, top-p and standalone top-p code VERBATIM
                           and stores the sixth pass's decision;
     _qrita_mask_kernel    grid (rows, splits): the sixth pass on the split's tiles.
                           Boundary ties are kept up to the search's last kept index; a
                           row the parallel form cannot reproduce gets stock's sixth pass
                           in one program.
   Same masks as stock by construction: the statistics and every float sum run stock's
   code with stock's tile shapes and 8 warps on the same buffer layout, while max, min,
   counts and the in-order compaction are exact in any order. Rows without a top-k
   keep stock's paths: the split top-p pipeline (batch <= 64) runs after these kernels as
   in stock, the standalone top-p of larger batches runs verbatim in the search kernel.
   install() pins the source of every stock function this mirrors (sha256) and keeps
   stock when the image's vLLM differs.

    python -m suffix_hybrid.kernels.lm_sample_rocm lmhead     # boot gate lm_sample_lmhead
    python -m suffix_hybrid.kernels.lm_sample_rocm topktopp   # boot gate lm_sample_topk_topp
"""
import hashlib
import sys

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix lm-sample]"
TARGET = "vllm.v1.sample.ops.topk_topp_triton"
ROW_TOPK_CHUNK = 4096
QRITA_BLOCK = 8192  # stock's BLOCK_SIZE / BLOCK_SIZE_TRUNC / num_warps on GPUs (_topk_topp)
QRITA_BLOCK_TRUNC = 4096
QRITA_WARPS = 8
COMPACT_ELEMS = 16384  # compaction tile: COMPACT positions x SPLITS_PAD segments
# sha256 of each stock function's source (decorators included) at vLLM 81198e97.
PINS = {
    "_update_min_larger_stats": "6765e3dc17394384cdf7b4900fbf07130de8326f564141da6e7e246c71d3f7bc",
    "_topk_topp_kernel": "93860dd70695262636d7a57d2cfaf2f9de023812c503720b466c8a8bf756eb98",
    "_topk_topp": "af00705111315d413550613a1d76afe403cb09b5639e3814b63e632e5bf577c4",
    "apply_top_k_top_p_triton": "eda8949f42671d5f10ae366c8d037118dea25c7dc8fdac79ed68e08fbc2da405",
}
_STATE: dict = {}


# ---------------------------------------------------------------------------
# 1. lm_head candidates
# ---------------------------------------------------------------------------
@triton.jit
def _topk_key(x):
    """ATen TopKTypeConfig<BFloat16>::convert: bits ^ 0xffff with the sign bit set, else
    ^ 0x8000; NaN -> 0xffff. Ascending key = torch.topk's order."""
    b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key = tl.where(b >= 0x8000, b ^ 0xFFFF, b ^ 0x8000)
    return tl.where(x == x, key, 0xFFFF)


@triton.jit
def _kth_key(key, K: tl.constexpr):
    """Largest t with count(key >= t) >= K (the K-th largest key; 0 when fewer than K
    keys are >= 0, i.e. take every valid one). Invalid entries carry -1."""
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), 65536, tl.int32)
    for _ in tl.static_range(16):
        mid = (lo + hi) // 2
        enough = tl.sum((key >= mid).to(tl.int32)) >= K
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid)
    return lo


@triton.jit
def _take_k(key, K: tl.constexpr):
    """Mask of the K largest keys, ties at the K-th key to the lowest positions."""
    t = _kth_key(key, K)
    above = key > t
    tie = key == t
    return above | (tie & (tl.cumsum(tie.to(tl.int32), axis=0)
                           <= K - tl.sum(above.to(tl.int32))))


@triton.jit(do_not_specialize=["N"])
def _row_topk_chunk_kernel(Y, CKEY, CIDX, N, stride_y, K: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    ok = offs < N
    x = tl.load(Y + row.to(tl.int64) * stride_y + offs, mask=ok, other=0.0)
    key = tl.where(ok, _topk_key(x), -1)
    take = _take_k(key, K)
    pos = tl.cumsum(take.to(tl.int32), axis=0) - 1
    base = (row * tl.num_programs(1) + chunk) * K
    tl.store(CKEY + base + pos, key, mask=take)
    tl.store(CIDX + base + pos, offs, mask=take)
    slot = tl.arange(0, K)  # a chunk with fewer than K entries: its free slots never win
    tl.store(CKEY + base + slot, -1, mask=slot >= tl.sum(take.to(tl.int32)))


@triton.jit
def _row_topk_merge_kernel(CKEY, CIDX, OUT, CANDS, K: tl.constexpr, C_PAD: tl.constexpr):
    # Candidates sit chunk by chunk, each chunk's in index order: position order is
    # index order, so _take_k's tie rule is torch.topk's.
    row = tl.program_id(0)
    j = tl.arange(0, C_PAD)
    ok = j < CANDS
    key = tl.load(CKEY + row * CANDS + j, mask=ok, other=-1)
    idx = tl.load(CIDX + row * CANDS + j, mask=ok, other=0)
    take = _take_k(key, K)
    pos = tl.cumsum(take.to(tl.int32), axis=0) - 1
    tl.store(OUT + row * K + pos, idx.to(tl.int64), mask=take)


def row_topk(y: torch.Tensor, k: int) -> torch.Tensor:
    """Indices [M, k] (int64, ascending) of the k largest entries of each row of bf16 y,
    the same set as y.topk(k).indices. HIP-graph safe; k a power of two <= row length."""
    m, n = y.shape
    if (y.dtype != torch.bfloat16 or y.stride(1) != 1 or k & (k - 1) or not 0 < k <= n):
        raise ValueError(f"{MARK} row_topk: unsupported y {tuple(y.shape)} {y.dtype} k={k}")
    chunks = triton.cdiv(n, ROW_TOPK_CHUNK)
    ckey = torch.empty((m, chunks, k), dtype=torch.int32, device=y.device)
    cidx = torch.empty_like(ckey)
    out = torch.empty((m, k), dtype=torch.int64, device=y.device)
    if m:
        _row_topk_chunk_kernel[(m, chunks)](y, ckey, cidx, n, y.stride(0), K=k,
                                            CHUNK=ROW_TOPK_CHUNK, num_warps=4)
        _row_topk_merge_kernel[(m,)](ckey, cidx, out, chunks * k, K=k,
                                     C_PAD=triton.next_power_of_2(chunks * k), num_warps=4)
    return out


# ---------------------------------------------------------------------------
# 2. Qrita top-k / top-p, rows split across programs
# ---------------------------------------------------------------------------
@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  → replace both
      - tile min == running min → accumulate count
      - tile min > running min  → keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _qrita_gather_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    PERCENTILE_TO_STD_TABLE,
    K,
    SMAX,
    SMIN,
    SNFIN,
    SNOUT,
    OPIVOT,
    SCR_VAL,
    SCR_IDX,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
    SPLIT_CAP: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    k = tl.load(K + row_id)
    if k < VOCAB_SIZE:
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        SEG = row_id * SPLITS + split
        max_logit = -float("inf")
        min_logit = float("inf")
        # ---- stock's zeroth pass, verbatim (every split recomputes it) ----
        # Zeroth pass: Compute avg and std from a sample block
        offs = tl.arange(0, BLOCK_SIZE)
        mask_n = offs < VOCAB_SIZE
        logits_blk0 = tl.load(
            LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
        )
        # Exclude -inf values (e.g. from grammar bitmasks) from
        # statistics to avoid NaN in pivot computation.
        finite_mask = (logits_blk0 > -float("inf")) & mask_n
        num_finite = tl.sum(finite_mask)
        finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
        avg_logit = tl.where(
            num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
        )
        sq_avg_logit = tl.where(
            num_finite > 0,
            tl.sum(finite_logits * finite_logits) / num_finite,
            0.0,
        )
        std_logit = tl.sqrt(
            tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
        )

        # Calculate outlier pivot t for Gaussian sigma-truncation
        percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
        percentile = tl.minimum(percentile, 199)
        sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
        sigma = sigma + tl.abs(sigma) * -0.15
        outlier_pivot = avg_logit + std_logit * sigma
        num_outliers = tl.zeros((), dtype=tl.uint32)

        # ---- stock's first pass on this split's tiles; outliers into the segment ----
        # First pass: compute max and min logits and gather outliers
        num_finite_total = tl.zeros((), dtype=tl.uint32)
        for i in range(split * TILES_PER_SPLIT,
                       tl.minimum((split + 1) * TILES_PER_SPLIT, NUM_TILES)):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < VOCAB_SIZE
            logits_blk = tl.load(
                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
            )

            max_logit = tl.maximum(max_logit, tl.max(logits_blk))
            # Exclude -inf from min to keep binary search bounds
            # finite (avoids NaN pivots).
            finite_blk_mask = logits_blk > -float("inf")
            finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
            min_logit = tl.minimum(min_logit, tl.min(finite_blk))
            num_finite_total += tl.sum(finite_blk_mask & mask_n)

            outlier_mask = (logits_blk > outlier_pivot) & mask_n
            cumulative_pos = tl.cast(
                tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
            )
            num_outliers += tl.sum(outlier_mask)
            write_pos = tl.where(outlier_mask, cumulative_pos, -1)
            seg_mask = outlier_mask & (write_pos < SPLIT_CAP)
            tl.store(SCR_VAL + SEG * SPLIT_CAP + write_pos, logits_blk, mask=seg_mask)
            tl.store(SCR_IDX + SEG * SPLIT_CAP + write_pos, offs_n, mask=seg_mask)

        tl.store(SMAX + SEG, max_logit)
        tl.store(SMIN + SEG, min_logit)
        tl.store(SNFIN + SEG, num_finite_total)
        tl.store(SNOUT + SEG, num_outliers)
        if split == 0:
            tl.store(OPIVOT + row_id, outlier_pivot)


@triton.jit(do_not_specialize_on_alignment=["BATCH_SIZE"])
def _qrita_search_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    BATCH_SIZE,
    SMAX,
    SMIN,
    SNFIN,
    SNOUT,
    OPIVOT,
    SCR_VAL,
    SCR_IDX,
    IDXBUF,
    DEC_F,
    DEC_I,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
    SPLIT_COVERS_PONLY: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    SPLIT_CAP: tl.constexpr,
    COMPACT: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        # suffix: what the mask kernel may do in parallel for this row
        IDX_ROW = IDXBUF + pid * (SPLITS * SPLIT_CAP)
        par_ok = tl.zeros((), dtype=tl.int32)
        nout = tl.zeros((), dtype=tl.int32)
        opivot = tl.full((), float("inf"), tl.float32)

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # suffix: the zeroth / first pass ran split (_qrita_gather_kernel);
                # merge its per-split results (max / min / counts: exact in any order).
                sp = tl.arange(0, SPLITS_PAD)
                sp_ok = sp < SPLITS
                seg = row_id * SPLITS + sp
                max_logit = tl.max(tl.load(SMAX + seg, mask=sp_ok, other=-float("inf")))
                min_logit = tl.min(tl.load(SMIN + seg, mask=sp_ok, other=float("inf")))
                cnts = tl.load(SNOUT + seg, mask=sp_ok, other=0)
                num_finite_total = tl.zeros((), dtype=tl.uint32) + tl.sum(
                    tl.load(SNFIN + seg, mask=sp_ok, other=0)
                ).to(tl.uint32)
                num_outliers = tl.zeros((), dtype=tl.uint32) + tl.sum(cnts).to(tl.uint32)
                outlier_pivot = tl.load(OPIVOT + row_id)
                if tl.max(cnts) > SPLIT_CAP:
                    # a segment overflowed: stock's zeroth / first pass, verbatim
                    max_logit = -float("inf")
                    min_logit = float("inf")
                    # Zeroth pass: Compute avg and std from a sample block
                    offs = tl.arange(0, BLOCK_SIZE)
                    mask_n = offs < VOCAB_SIZE
                    logits_blk0 = tl.load(
                        LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                    )
                    # Exclude -inf values (e.g. from grammar bitmasks) from
                    # statistics to avoid NaN in pivot computation.
                    finite_mask = (logits_blk0 > -float("inf")) & mask_n
                    num_finite = tl.sum(finite_mask)
                    finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                    avg_logit = tl.where(
                        num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                    )
                    sq_avg_logit = tl.where(
                        num_finite > 0,
                        tl.sum(finite_logits * finite_logits) / num_finite,
                        0.0,
                    )
                    std_logit = tl.sqrt(
                        tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                    )

                    # Calculate outlier pivot t for Gaussian sigma-truncation
                    percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                    percentile = tl.minimum(percentile, 199)
                    sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                    sigma = sigma + tl.abs(sigma) * -0.15
                    outlier_pivot = avg_logit + std_logit * sigma
                    num_outliers = tl.zeros((), dtype=tl.uint32)

                    # First pass: compute max and min logits and gather outliers
                    num_finite_total = tl.zeros((), dtype=tl.uint32)
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE
                        logits_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )

                        max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                        # Exclude -inf from min to keep binary search bounds
                        # finite (avoids NaN pivots).
                        finite_blk_mask = logits_blk > -float("inf")
                        finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                        min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                        num_finite_total += tl.sum(finite_blk_mask & mask_n)

                        outlier_mask = (logits_blk > outlier_pivot) & mask_n
                        cumulative_pos = tl.cast(
                            tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                        )
                        num_outliers += tl.sum(outlier_mask)
                        write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                        tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

                else:
                    # segments -> BUFFER_ROW in index order (+ each outlier's index)
                    n_out = tl.sum(cnts)
                    ends = tl.cumsum(cnts, axis=0)
                    starts = ends - cnts
                    for q0 in range(0, n_out, COMPACT):
                        q = q0 + tl.arange(0, COMPACT)
                        q_ok = q < n_out
                        sg = tl.sum((q[:, None] >= ends[None, :]).to(tl.int32), axis=1)
                        st = tl.sum(tl.where(sg[:, None] == sp[None, :], starts[None, :], 0),
                                    axis=1)
                        src = (row_id * SPLITS + sg) * SPLIT_CAP + (q - st)
                        tl.store(BUFFER_ROW + q, tl.load(SCR_VAL + src, mask=q_ok), mask=q_ok)
                        tl.store(IDX_ROW + q, tl.load(SCR_IDX + src, mask=q_ok), mask=q_ok)
                    tl.debug_barrier()
                    nout = n_out
                    par_ok = (num_outliers > k).to(tl.int32)
                    opivot = outlier_pivot

                # ---- stock, verbatim from here to its sixth pass ----
                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Second passes: Ternary search for pivot
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.uint32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                if num_outliers > k:
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute k_pivots_num,
                        # min_larger, and num_min_larger together to avoid
                        # a second data scan. See _update_min_larger_stats
                        # for the tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                else:
                    # If top-k outlier gathering failed, search whole logit space
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass over full vocab (same approach
                        # as the buffer path above).
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Third pass: Calculate exp logits and sum, gather outliers
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling for Top-k
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = (
                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
                                    )
                                    duplicate_count = (
                                        tl.cumsum(duplicate_mask) + num_kept
                                    )
                                    duplicate_keep_mask = (
                                        duplicate_count <= num_keep
                                    ) & duplicate_mask
                                    duplicate_remove_mask = (
                                        duplicate_mask & ~duplicate_keep_mask
                                    )
                                    outlier_mask = outlier_mask & (
                                        ~duplicate_remove_mask
                                    )
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed,
                            # retry gathering using top-k pivot
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling for Top-k
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                    tl.int32,
                                )
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(
                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                                )

                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0

                        # Fifth passes: Search for p_pivot
                        found_pivot = 0
                        while found_pivot == 0:
                            p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                            # Single fused pass: compute p_pivots_sum,
                            # min_larger, and num_min_larger together.
                            # See _update_min_larger_stats for the
                            # tile-level merge logic.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                above_0 = probs_blk > p_pivot_0
                                p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                                min_larger_0, num_min_larger_0 = (
                                    _update_min_larger_stats(
                                        probs_blk,
                                        above_0,
                                        min_larger_0,
                                        num_min_larger_0,
                                        1.0,
                                    )
                                )

                            # Check if the pivot satisfies termination condition
                            if p_pivots_sum_0 >= p and (
                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                            ):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # Update range
                            if p_pivots_sum_0 > p:
                                min_range = p_pivot_0
                            elif p_pivots_sum_0 < p:
                                max_range = p_pivot_0

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                p_pivot = (max_range + min_range) / 2.0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                        duplicate_logit = (
                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        )
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast(
                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
                        )
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        fp_topk = final_pivot  # suffix: before stock's standalone top-p
        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            # When the split top-p pipeline co-runs (mixed top-k + top-p
            # batches), it covers p-only rows (k >= VOCAB_SIZE); skip those
            # here. Rows whose top-k was a no-op because they have <= k
            # finite logits (e.g. grammar masks) are NOT covered by the
            # split pipeline and must get standalone top-p here.
            run_standalone = True
            if SPLIT_COVERS_PONLY and TOPK_ENABLED:
                run_standalone = tl.load(K + row_id) < VOCAB_SIZE
            p = tl.load(P + row_id)
            if run_standalone and p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # suffix: the sixth pass's decision for _qrita_mask_kernel (stock's skip rule).
        # mode 1: keep > final_pivot; 2: and boundary ties up to index dup_last (the
        # num_keep-th tie in index order, from the buffer's indices); 3: stock's sixth
        # pass in one program (ties that need not sit in the buffer).
        mode = tl.zeros((), dtype=tl.int32)
        dup_last = tl.full((), -1, tl.int32)
        if final_pivot < max_logit:
            if final_pivot != -float("inf"):
                mode = tl.full((), 1, tl.int32)
                if num_keep < num_duplicate_logit:
                    mode = tl.full((), 3, tl.int32)
                    # every value within 1e-9 of duplicate_logit is an outlier (in the
                    # buffer) when duplicate_logit clears the outlier pivot by a margin
                    if ((par_ok == 1) & (fp_topk != -float("inf"))
                            & (duplicate_logit > opivot + tl.abs(opivot) * 1e-6 + 1e-6)):
                        mode = tl.full((), 2, tl.int32)
                        nk = num_keep.to(tl.int32)
                        seen = tl.zeros((), dtype=tl.int32)
                        for q0 in range(0, nout, BLOCK_SIZE_TRUNC):
                            q = q0 + tl.arange(0, BLOCK_SIZE_TRUNC)
                            q_ok = q < nout
                            idx = tl.load(IDX_ROW + q, mask=q_ok, other=0)
                            v = tl.load(LOGITS_ROW + idx, mask=q_ok, other=-float("inf"))
                            dup = (tl.abs(v - duplicate_logit) < 1e-9) & q_ok
                            rank = tl.cumsum(dup.to(tl.int32), axis=0) + seen
                            dup_last = tl.maximum(
                                dup_last, tl.max(tl.where(dup & (rank == nk), idx, -1)))
                            seen += tl.sum(dup.to(tl.int32))
        tl.store(DEC_F + row_id * 2, final_pivot)
        tl.store(DEC_F + row_id * 2 + 1, duplicate_logit)
        tl.store(DEC_I + row_id * 4, mode)
        tl.store(DEC_I + row_id * 4 + 1, dup_last)
        tl.store(DEC_I + row_id * 4 + 2, num_keep.to(tl.int32, bitcast=True))
        tl.store(DEC_I + row_id * 4 + 3, num_duplicate_logit.to(tl.int32, bitcast=True))


@triton.jit
def _qrita_mask_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    DEC_F,
    DEC_I,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    mode = tl.load(DEC_I + row_id * 4)
    if mode != 0:
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        final_pivot = tl.load(DEC_F + row_id * 2)
        duplicate_logit = tl.load(DEC_F + row_id * 2 + 1)
        if mode == 3:
            if split == 0:
                num_keep = tl.load(DEC_I + row_id * 4 + 2).to(tl.uint32, bitcast=True)
                num_duplicate_logit = tl.load(DEC_I + row_id * 4 + 3).to(
                    tl.uint32, bitcast=True)
                num_kept = tl.zeros((), dtype=tl.uint32)
                # ---- stock's sixth pass, verbatim ----
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    keep_mask = (logits_blk > final_pivot) & mask_n

                    # Duplicate logit handling
                    if num_keep < num_duplicate_logit:
                        duplicate_mask = (
                            tl.abs(logits_blk - duplicate_logit) < 1e-9
                        ) & mask_n
                        duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                        duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                        duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                        num_kept += tl.sum(duplicate_keep_mask)
                        keep_mask = keep_mask & (~duplicate_remove_mask)

                    logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)
        else:
            dup_last = tl.load(DEC_I + row_id * 4 + 1)
            for i in range(split * TILES_PER_SPLIT,
                           tl.minimum((split + 1) * TILES_PER_SPLIT, NUM_TILES)):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                keep_mask = (logits_blk > final_pivot) & mask_n
                if mode == 2:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    keep_mask = keep_mask & ((~duplicate_mask) | (offs_n <= dup_last))
                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def _split_plan(batch: int, vocab: int, num_sm: int):
    """(splits, tiles per split, segment capacity): ~2 programs per CU, an outlier
    segment of 1/8 of the split (a fuller split takes the verbatim fallback)."""
    tiles = triton.cdiv(vocab, QRITA_BLOCK)
    tps = triton.cdiv(tiles, max(1, min(tiles, triton.cdiv(2 * num_sm, batch))))
    return triton.cdiv(tiles, tps), tps, triton.next_power_of_2(tps * QRITA_BLOCK // 8)


def _qrita_split(logits, k, p, mask_value, num_sm, split_covers_ponly, modes=None):
    """The three launches on fp32 logits [B, V] (row stride free, unit column stride),
    int32 k [B] (top-k on), fp32 p [B] or None. Masks logits in place."""
    mod = sys.modules[TARGET]
    batch, vocab = logits.shape
    dev = logits.device
    programs = min(num_sm, batch)
    buf_key = (dev, logits.dtype, vocab)  # stock's per-program buffer and tables
    buffer = mod._TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < programs:
        buffer = logits.new_empty((min(mod.next_power_of_2(programs), num_sm), vocab))
        mod._TRITON_BUFFER_CACHE[buf_key] = buffer
    tables = mod._TRITON_TABLE_CACHE.get(dev)
    if tables is None:
        with mod.gpu_sync_allowed():
            tables = (logits.new_tensor(mod._NORMAL_CDF_TO_SIGMA_TABLE),
                      logits.new_tensor(mod._PERCENTILE_TO_STD_TABLE))
        mod._TRITON_TABLE_CACHE[dev] = tables
    normal_cdf_to_sigma_table, percentile_to_std_table = tables
    splits, tps, cap = _split_plan(batch, vocab, num_sm)
    f32 = dict(dtype=torch.float32, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    smax, smin = torch.empty((batch, splits), **f32), torch.empty((batch, splits), **f32)
    snfin, snout = torch.empty((batch, splits), **i32), torch.empty((batch, splits), **i32)
    opivot = torch.empty((batch,), **f32)
    scr_val = torch.empty((batch, splits, cap), **f32)
    scr_idx = torch.empty((batch, splits, cap), **i32)
    idxbuf = torch.empty((programs, splits * cap), **i32)
    dec_f, dec_i = torch.empty((batch, 2), **f32), torch.empty((batch, 4), **i32)
    _qrita_gather_kernel[(batch, splits)](
        logits, logits.stride(0), percentile_to_std_table, k, smax, smin, snfin, snout,
        opivot, scr_val, scr_idx, VOCAB_SIZE=vocab, BLOCK_SIZE=QRITA_BLOCK, SPLITS=splits,
        TILES_PER_SPLIT=tps, SPLIT_CAP=cap, num_warps=QRITA_WARPS)
    _qrita_search_kernel[(programs,)](
        logits, logits.stride(0), buffer[:programs], percentile_to_std_table,
        normal_cdf_to_sigma_table, k, p if p is not None else logits, batch, smax, smin,
        snfin, snout, opivot, scr_val, scr_idx, idxbuf, dec_f, dec_i, VOCAB_SIZE=vocab,
        BLOCK_SIZE=QRITA_BLOCK, BLOCK_SIZE_TRUNC=QRITA_BLOCK_TRUNC, TOPK_ENABLED=True,
        TOPP_ENABLED=p is not None, SPLIT_COVERS_PONLY=split_covers_ponly, SPLITS=splits,
        SPLITS_PAD=triton.next_power_of_2(splits), SPLIT_CAP=cap,
        COMPACT=min(4096, COMPACT_ELEMS // triton.next_power_of_2(splits)),
        num_warps=QRITA_WARPS)
    _qrita_mask_kernel[(batch, splits)](
        logits, logits.stride(0), dec_f, dec_i, VOCAB_SIZE=vocab, MASK_VALUE=mask_value,
        BLOCK_SIZE=QRITA_BLOCK, TILES_PER_SPLIT=tps, num_warps=QRITA_WARPS)
    if modes is not None:
        modes.append(dec_i[:, 0])


def apply_top_k_top_p_triton(logits, k, p, mask_value=float("-inf")):
    """Drop-in for vLLM's apply_top_k_top_p_triton: stock's argument handling, the split
    kernels when a top-k is on, stock otherwise."""
    return _apply(logits, k, p, mask_value)


def _apply(logits, k, p, mask_value, modes=None):
    if (k is None or logits.ndim != 2 or logits.dtype != torch.float32
            or logits.device.type != "cuda" or logits.shape[0] == 0):
        return _STATE["stock"](logits, k, p, mask_value)
    mod = sys.modules[TARGET]
    batch = logits.shape[0]
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    assert k.ndim == 1 and k.shape[0] == batch
    k_ptr = k.to(torch.int32)
    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch
        p = p.to(torch.float32)
    num_sm = mod.num_compute_units(logits.device.index)
    use_split = p is not None and batch <= mod._SPLIT_MAX_BATCH
    _qrita_split(logits, k_ptr, p, mask_value, num_sm, use_split, modes)
    if use_split:  # stock: p-only rows of small batches
        mod._apply_topp_split(logits, k_ptr, p, mask_value, num_sm)
    return logits


def source_pins(src: str) -> dict:
    """sha256 per pinned function of a module source (decorators included)."""
    import ast

    lines = src.splitlines(keepends=True)
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in PINS:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            out[node.name] = hashlib.sha256(
                "".join(lines[start - 1:node.end_lineno]).encode()).hexdigest()
    return out


def install(module) -> None:
    """rocm_patches `after` hook for vllm.v1.sample.ops.topk_topp_triton."""
    import inspect

    got = source_pins(inspect.getsource(module))
    drift = sorted(name for name, digest in PINS.items() if got.get(name) != digest)
    if drift:
        print(f"{MARK} stock top-k / top-p kept: {drift} differ from vLLM 81198e97",
              file=sys.stderr, flush=True)
        return
    _STATE["stock"] = module.apply_top_k_top_p_triton
    module.apply_top_k_top_p_triton = apply_top_k_top_p_triton


def _graph_us(fn, iters: int = 20):
    """us per call of fn() replayed from one HIP graph, and the graphed output."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters, out


def _same_set(a, b) -> bool:
    return torch.equal(a.sort(dim=1).values, b.sort(dim=1).values)


def _oracle_lmhead() -> int:
    """row_topk == torch.topk's index set on adversarial bf16 rows; then the MXFP4 head
    with row_topk vs with y.topk (the shipped path), bitwise, M 1..256, greedy vs the
    bf16 head, graph replay == eager, graphed us/call."""
    from suffix_hybrid.kernels import mxfp4_lm_head as lmh

    prepare, run, _ = lmh._ops()
    n, kdim = lmh.SHAPE
    dev, r = "cuda", lmh.RESCORE
    gen = torch.Generator(device=dev).manual_seed(0)
    failed = False

    def rows(kind, m):
        if kind == "ties":  # 17 distinct values: thousands of ties at the 64th
            return (torch.randint(-8, 9, (m, n), generator=gen, device=dev) / 4).bfloat16()
        y = torch.randn(m, n, generator=gen, device=dev)
        if kind == "boundary-ties":  # the 64th value repeated in every chunk
            kth = y.topk(r, dim=-1).values[:, -1:]
            pos = torch.randint(0, n, (m, 200), generator=gen, device=dev)
            y.scatter_(1, pos, kth.expand(-1, 200))
        elif kind == "negative":
            y = -y.abs() - 1
        elif kind == "signed-zero":  # few positives, +0.0 / -0.0 decide the rest
            y = torch.zeros(m, n, device=dev)
            y[:, 1::2] = -0.0
            y.scatter_(1, torch.randint(0, n, (m, 20), generator=gen, device=dev), 1.0)
        elif kind == "inf-nan":
            y[:, 7::4099] = float("inf")
            y[:, 11::5003] = float("-inf")
            y[:, 13::7001] = float("nan")
        return y.bfloat16()

    for kind in ("randn", "ties", "boundary-ties", "negative", "signed-zero", "inf-nan"):
        bad = []
        for m in (1, 5, 64, 160, 256):
            y = rows(kind, m)
            if not _same_set(row_topk(y, r), y.topk(r, dim=-1).indices):
                bad.append(m)
        failed |= bool(bad)
        print(f"{MARK} lmhead row_topk({r}) vs torch.topk set, {kind}: "
              f"{'MATCH' if not bad else f'MISMATCH at M={bad}'}", flush=True)

    w = torch.randn(n, kdim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    st = prepare(w, 256)

    def ref_topk(y, k):
        return y.topk(k, dim=-1).indices

    for m in (1, 2, 4, 5, 8, 16, 32, 40, 64, 96, 128, 160, 192, 256):
        x = torch.randn(m, kdim, generator=gen, device=dev, dtype=torch.bfloat16)
        new, old = run(st, x), run(st, x, topk=ref_topk)
        ok = torch.equal(new, old) and lmh.greedy_ok(torch.nn.functional.linear(x, w), new)
        out = torch.empty_like(new)
        new_us, graphed = _graph_us(lambda: run(st, x, out))
        ok &= torch.equal(graphed, new)
        old_us, _ = _graph_us(lambda: run(st, x, out, topk=ref_topk))
        bf16_us, _ = _graph_us(lambda: torch.nn.functional.linear(x, w))
        torch_us, _ = _graph_us(lambda: new.topk(r, dim=-1).indices)
        ours_us, _ = _graph_us(lambda: row_topk(new, r))
        failed |= not ok
        print(f"{MARK} lmhead M={m}: {'MATCH' if ok else 'MISMATCH'} (logits bitwise vs the "
              f"y.topk path, greedy == bf16, graph == eager) | graphed us: bf16 {bf16_us:.1f} "
              f"| mxfp4 + y.topk {old_us:.1f} -> mxfp4 + row_topk {new_us:.1f} | top-{r} alone: "
              f"torch {torch_us:.1f} -> row_topk {ours_us:.1f}", flush=True)
    return 1 if failed else 0


def _lm_like(b, vocab, gen, temperature):
    """LM-like logits: a N(0, 2.5^2) body and a heavy head (300 tokens 6..20 above it),
    rounded to bf16 as the lm_head returns them, / temperature in fp32."""
    z = torch.randn(b, vocab, generator=gen, device="cuda") * 2.5
    head = torch.randint(0, vocab, (b, 300), generator=gen, device="cuda")
    z.scatter_add_(1, head, torch.rand(b, 300, generator=gen, device="cuda") * 14 + 6)
    return z.bfloat16().float() / temperature


def _oracle_topk_topp() -> int:
    """Masked logits bitwise vs vLLM's apply_top_k_top_p_triton (stock module, this pod's
    vLLM) at batch 1..256 over LM-like logits, ties, -inf masks, rows with fewer finite
    values than k, constant rows and batches mixing top-k / p-only / k=V rows; graph
    replay == eager; us/call stock vs split (eager, the sampler is not graphed)."""
    import importlib
    import inspect

    mod = importlib.import_module(TARGET)
    got = source_pins(inspect.getsource(mod))
    drift = sorted(n for n, d in PINS.items() if got.get(n) != d)
    print(f"{MARK} topktopp source pins: {'OK' if not drift else f'DRIFT {drift}'}", flush=True)
    stock = mod.apply_top_k_top_p_triton
    _STATE["stock"] = stock
    dev, vocab = "cuda", 248320
    gen = torch.Generator(device=dev).manual_seed(0)
    num_sm = mod.num_compute_units(0)
    failed = bool(drift)

    def case(name, b):
        i32 = dict(dtype=torch.int32, device=dev)
        if name == "k20-p0.95-t1":
            return _lm_like(b, vocab, gen, 1.0), torch.full((b,), 20, **i32), torch.full((b,), 0.95, device=dev)
        if name == "k20-p0.8-t0.7":
            return _lm_like(b, vocab, gen, 0.7), torch.full((b,), 20, **i32), torch.full((b,), 0.8, device=dev)
        if name == "k20-p1":
            return _lm_like(b, vocab, gen, 1.0), torch.full((b,), 20, **i32), torch.ones(b, device=dev)
        if name == "k20-nop":
            return _lm_like(b, vocab, gen, 0.2), torch.full((b,), 20, **i32), None
        if name == "k1-p0.95":
            return _lm_like(b, vocab, gen, 1.0), torch.ones(b, **i32), torch.full((b,), 0.95, device=dev)
        if name == "k64-p0.95-t0.2":
            return _lm_like(b, vocab, gen, 0.2), torch.full((b,), 64, **i32), torch.full((b,), 0.95, device=dev)
        if name == "k1000-p0.9":
            return _lm_like(b, vocab, gen, 1.0), torch.full((b,), 1000, **i32), torch.full((b,), 0.9, device=dev)
        if name == "ties":
            x = (torch.randn(b, vocab, generator=gen, device=dev) * 8).round() / 4
            return x, torch.full((b,), 20, **i32), torch.full((b,), 0.95, device=dev)
        # mixed: per-row k in {20, V, 1, 64}, p in {0.95, 1.0, 0.8}, grammar-style -inf
        # rows, a row with 10 finite values (< k), a constant row
        x = _lm_like(b, vocab, gen, 1.0)
        kk = torch.tensor([20, vocab, 1, 64], **i32).repeat(b // 4 + 1)[:b]
        pp = torch.tensor([0.95, 1.0, 0.8], device=dev).repeat(b // 3 + 1)[:b]
        x[1::5, ::2] = float("-inf")
        if b > 2:
            x[2, 10:] = float("-inf")  # 10 finite values, k 20
            kk[2] = 20
        if b > 3:
            x[3] = 1.5
        return x, kk, pp

    names = ("k20-p0.95-t1", "k20-p0.8-t0.7", "k20-p1", "k20-nop", "k1-p0.95",
             "k64-p0.95-t0.2", "k1000-p0.9", "ties", "mixed")
    for b in (1, 2, 5, 8, 16, 32, 40, 64, 65, 128, 160, 256):
        line = []
        for name in names:
            x, kk, pp = case(name, b)
            ref = stock(x.clone(), kk, pp)
            modes = []
            new = _apply(x.clone(), kk, pp, float("-inf"), modes)
            rows_eq = (ref == new).all(dim=1)
            ok = bool(rows_eq.all())
            failed |= not ok
            hist = torch.bincount(modes[0].long(), minlength=4).tolist() if modes else []
            line.append(f"{name} {'ok' if ok else f'MISMATCH {int((~rows_eq).sum())} rows'}"
                        f" m{hist}")
        x, kk, pp = case("k20-p0.95-t1", b)
        work = x.clone()

        def fill_and_split():
            work.copy_(x)
            return apply_top_k_top_p_triton(work, kk, pp)

        eager = fill_and_split().clone()
        _, graphed = _graph_us(fill_and_split, iters=3)
        graph_ok = torch.equal(graphed, eager)
        failed |= not graph_ok

        def eager_us(fn, iters=20):
            for _ in range(3):
                work.copy_(x)
                fn(work, kk, pp)
            torch.cuda.synchronize()
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(iters):
                work.copy_(x)
                fn(work, kk, pp)
            e.record()
            torch.cuda.synchronize()
            return a.elapsed_time(e) * 1e3 / iters

        copy_us = eager_us(lambda *a: None)
        stock_us = eager_us(stock) - copy_us
        new_us = eager_us(apply_top_k_top_p_triton) - copy_us
        print(f"{MARK} topktopp B={b} splits {_split_plan(b, vocab, num_sm)[0]}: "
              f"{' | '.join(line)} | graph {'== eager' if graph_ok else 'MISMATCH'} | "
              f"k20 p0.95 us: stock {stock_us:.1f} -> split {new_us:.1f}", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    which = (argv if argv is not None else sys.argv[1:]) or ["lmhead", "topktopp"]
    rc = 0
    if "lmhead" in which:
        rc |= _oracle_lmhead()
    if "topktopp" in which:
        rc |= _oracle_topk_topp()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
