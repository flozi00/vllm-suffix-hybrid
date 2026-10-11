# SPDX-License-Identifier: Apache-2.0
"""QSA indexer scores over visible columns only (SUFFIX_ROCM_QSA_MQA=1, ROCm).

vLLM's AMD qsa_mqa_paged launches rows x capacity/32 programs, capacity being the
page-table width for max_model_len (65536 compressed columns at 262144), and each
program runs the head loop and stores -inf even when all its columns lie past
the row's visible length. On the MI350P at 32 concurrent MTP-4 requests that
kernel took 16.1 of 62.9 ms per step. Here a fixed (rows, NPROG) grid, safe for
graph replay, strides over the row's visible column blocks (read from device
memory) and loads each key block once instead of once per head; the math per
column is unchanged. Columns past `visible` are left unwritten: the only
consumer, top_k_per_row_decode, reads [0, visible) and does not read logits at
all for rows of <= top-k columns.

    python -m suffix_hybrid.kernels.qsa_mqa_rocm   # GPU oracle vs vLLM + us/call (boot gate qsa_mqa_bench)
"""
from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton

NPROG = 16
BLOCK_N = 32


@triton.jit
def _qsa_mqa_visible_kernel(
    q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr,
    sequence_lengths_ptr, visible_blocks_ptr, logits_ptr,
    stride_q_row, stride_q_head, stride_q_dim,
    stride_cache_block, stride_cache_token, stride_cache_dim,
    stride_table_req, stride_table_page, stride_logits_row,
    num_columns, num_pages, num_requests, score_divisor,
    PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr, NPROG: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row)
    request_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(sequence_lengths_ptr + safe_request, mask=request_ok, other=0)
    visible = tl.minimum((query_position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO)
    if lane == 0:
        tl.store(visible_blocks_ptr + row, visible)
    end = tl.minimum(visible, num_columns)
    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < HEAD_DIM
    for block in range(lane, tl.cdiv(end, BLOCK_N), NPROG):
        columns = block * BLOCK_N + tl.arange(0, BLOCK_N)
        logical_page = columns // PAGE_SIZE
        valid = (columns < end) & request_ok & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            page_table_ptr + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid, other=-1)
        valid &= (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        keys = tl.load(
            k_cache_ptr + tl.maximum(physical_page, 0).to(tl.int64)[:, None] * stride_cache_block
            + (columns % PAGE_SIZE)[:, None] * stride_cache_token + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & dim_ok[None, :], other=0.0).to(tl.float32)
        score = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for head in tl.static_range(0, NUM_HEADS):
            query = tl.load(q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
                            mask=dim_ok, other=0.0).to(tl.float32)
            score += tl.maximum(tl.sum(keys * query[None, :], axis=1), 0.0)
        score /= score_divisor
        tl.store(logits_ptr + row * stride_logits_row + columns,
                 tl.where(valid, score, -float("inf")), mask=columns < num_columns)


def qsa_mqa_paged(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
                  compress_ratio, num_columns=None, score_scale=None):
    """Drop-in for vllm.models.qwen4_exp.amd.ops.qsa.qsa_mqa_paged (same signature/outputs
    on [0, visible) of every row)."""
    rows = q.shape[0]
    if (q.ndim != 3 or k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.shape[3] != q.shape[2]
            or page_table.ndim != 2 or token_to_req.shape != (rows,)
            or query_positions.shape != (rows,) or sequence_lengths.shape != (page_table.shape[0],)
            or compress_ratio <= 0):
        raise ValueError("[suffix qsa-mqa] unexpected QSA scoring shapes")
    divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(rows, dtype=torch.int32, device=q.device)
    if rows and columns:
        _qsa_mqa_visible_kernel[(rows, NPROG)](
            q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
            visible_blocks, logits,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
            page_table.stride(0), page_table.stride(1), logits.stride(0),
            columns, k_cache.shape[0], page_table.shape[0], float(divisor),
            PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2], BLOCK_N=BLOCK_N,
            BLOCK_D=triton.next_power_of_2(q.shape[2]), COMPRESS_RATIO=compress_ratio,
            NPROG=NPROG, num_warps=4)
    return logits, visible_blocks


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.ops.qsa."""
    module.qsa_mqa_paged = qsa_mqa_paged


def _time_us(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def main() -> int:
    """Oracle on silicon: logits on [0, visible) and visible_blocks must equal vLLM's kernel.
    Shapes of qwen3.8-flash-next (indexer 4 heads x 128, compress 4, 262144 ctx = 65536 columns)."""
    from vllm.models.qwen4_exp.amd.ops import qsa as stock

    dev, heads, dim, page, width, ratio, pages = "cuda", 4, 128, 64, 1024, 4, 8192
    torch.manual_seed(0)
    k_cache = torch.randn(pages, page, 1, dim, device=dev, dtype=torch.bfloat16)
    failed = False
    # name, rows, requests, (seq_lo, seq_hi), prefill rows (positions run up to seq) vs decode rows
    for name, rows, reqs, (lo, hi), prefill in (("decode c32 x mtp5", 160, 32, (64, 2400), False),
                                                  ("decode long 40k", 20, 4, (30000, 41000), False),
                                                  ("prefill chunk 384", 384, 1, (8192, 8192), True)):
        seq = torch.randint(lo, hi + 1, (reqs,), device=dev, dtype=torch.int32)
        table = torch.randint(0, pages, (reqs, width), device=dev, dtype=torch.int32)
        idx = torch.arange(rows, device=dev)
        tok2req = (idx * reqs // rows).to(torch.int32)
        tok2req[-3:] = -1  # graph-padding rows
        last = seq[tok2req.clamp(min=0).long()]
        pos = (last - rows + idx if prefill else last - 1 - idx % 5).clamp(min=0).to(torch.int32)
        q = torch.randn(rows, heads, dim, device=dev, dtype=torch.bfloat16)
        args = (q, k_cache, table, tok2req, pos, seq, ratio)
        ref, ref_vis = stock.qsa_mqa_paged(*args)
        out, vis = qsa_mqa_paged(*args)
        ok = torch.equal(ref_vis, vis)
        worst = 0.0
        for r in range(rows):
            v = max(int(vis[r]), 0)
            if v:
                worst = max(worst, (out[r, :v] - ref[r, :v]).abs().max().item())
                ok &= torch.allclose(out[r, :v], ref[r, :v], rtol=1e-5, atol=1e-6)
        failed |= not ok
        print(f"[suffix qsa-mqa] {name}: rows {rows} cols {ref.shape[1]} "
              f"{'MATCH' if ok else 'MISMATCH'} (max abs diff {worst:.2e}) | stock "
              f"{_time_us(lambda: stock.qsa_mqa_paged(*args)):.1f} us -> "
              f"{_time_us(lambda: qsa_mqa_paged(*args)):.1f} us", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
