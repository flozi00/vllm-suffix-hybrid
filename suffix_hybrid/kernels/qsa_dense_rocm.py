# SPDX-License-Identifier: Apache-2.0
"""QSA attention with a dense path for short requests (SUFFIX_ROCM_QSA_DENSE=1, ROCm).

vLLM's AMD Qwen4Exp QSA layers (12 of Qwen3.8-Flash-Next's 48, plus the MTP layer)
attend over a per-row selection of token_topk + compress - 1 slots (2048 + 3). For a
row at position p with p + 1 <= token_topk that selection is every token 0..p: all
(p + 1) // compress complete blocks fit the block top-k, and expand_qsa_block_indices
adds the open group's tail. vLLM's sparse kernel still gathers it token by token once
per query row, so the five MTP-4 verify rows of a request read the same K/V five
times: 179 us per layer at c32 (7.7% of the step) for ~20 MB of distinct K/V.

The dense kernel runs a request with seq_len <= token_topk and at most
DENSE_BLOCK_M // group query rows (5 x 12 heads = 60 of 64) as one block over its
context in token order: K/V tiles straight from the page table, causal mask by
position, each K/V byte read once per request and KV head. Every other request (long
context, prefill chunks) keeps vLLM's sparse kernel, copied here with a row skip for
the dense requests and SUFFIX_ROCM_QSA_SPARSE_SKIP's all-padding tile skip. Both write
the same split partials and share vLLM's merge kernel. MTP steps that reuse the step-0
selection (indexer.skip_topk) attend to that selection, not to 0..p: the caller passes
dense=False and the stock path runs unchanged.

Numerics: the same fp32 online softmax over the same tokens; the fp32 accumulation
order differs (token order and tile cuts vs the top-k order), so dense outputs can
differ by bf16 roundings. Sparse rows are bit-identical to vLLM's.

    python -m suffix_hybrid.kernels.qsa_dense_rocm   # GPU oracle + us/call (boot gate qsa_dense_bench)
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix qsa-dense]"
DENSE_BLOCK_M = 64  # query rows x group heads of one dense program
DENSE_CONFIG = (64, 4, 1)  # BLOCK_N, num_warps, num_stages (boot gate qsa_dense_bench sweeps)


@triton.jit
def _dense_request(request, seq_lens_ptr, query_start_ptr, num_dense_requests,
                   GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, TOKEN_TOPK: tl.constexpr):
    """(q_start, q_len, seq_len, eligible) of a request; one formula for both kernels."""
    safe = tl.minimum(tl.maximum(request, 0), num_dense_requests - 1)
    q_start = tl.load(query_start_ptr + safe)
    q_len = tl.load(query_start_ptr + safe + 1) - q_start
    seq_len = tl.load(seq_lens_ptr + safe)
    eligible = ((request >= 0) & (request < num_dense_requests) & (q_len >= 1)
                & (q_len * GROUP_SIZE <= BLOCK_M) & (seq_len >= q_len) & (seq_len <= TOKEN_TOPK))
    return q_start, q_len, seq_len, eligible


@triton.jit
def _qsa_dense_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, seq_lens_ptr, query_start_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr,
    stride_q_row, stride_q_head,
    stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head,
    stride_table_req, stride_output_row, stride_output_head,
    num_rows, num_cache_blocks, num_dense_requests,
    TOKEN_TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    q_start, q_len, seq_len, eligible = _dense_request(
        request, seq_lens_ptr, query_start_ptr, num_dense_requests,
        GROUP_SIZE, BLOCK_M, TOKEN_TOPK)
    if eligible == 0:
        return

    # Program row m = query q_index of the request x head m % GROUP_SIZE of this KV head.
    m = tl.arange(0, BLOCK_M)
    q_index = m // GROUP_SIZE
    head = kv_head * GROUP_SIZE + m % GROUP_SIZE
    row = q_start + q_index
    row_ok = q_index < q_len
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr + row[:, None] * stride_q_row + head[:, None] * stride_q_head
        + dim_offsets[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    position = seq_len - q_len + q_index  # the row's selection: tokens 0..position

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    num_tiles = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, NUM_SPLITS)
    tile_start = split_id * tiles_per_split
    tile_end = tl.minimum(tile_start + tiles_per_split, num_tiles)
    for tile in range(tile_start, tile_end):
        token = tile * BLOCK_N + column_offsets
        logical_page = token // PAGE_SIZE
        page_offset = token % PAGE_SIZE
        valid = (token < seq_len) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        visible = valid[None, :] & (token[None, :] <= position[:, None])
        scores = tl.where(visible, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(visible, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr + row[:, None] * stride_output_row + head[:, None] * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=row_ok[:, None],
        )
    else:
        partial_lse = tl.where(
            has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        slot = (split_id * num_rows + row) * NUM_QUERY_HEADS + head
        tl.store(partial_output_ptr + slot[:, None] * HEAD_DIM + dim_offsets[None, :],
                 normalized_output, mask=row_ok[:, None])
        tl.store(partial_lse_ptr + slot, partial_lse, mask=row_ok)


@triton.jit
def _qsa_sparse_rows_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, indices_ptr, block_table_ptr, token_to_req_ptr,
    seq_lens_ptr, query_start_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr,
    stride_q_row, stride_q_head,
    stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head,
    stride_indices_row, stride_table_req, stride_output_row, stride_output_head,
    num_rows, num_cache_blocks, num_requests, num_dense_requests,
    TOPK: tl.constexpr, TOKEN_TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DENSE_BLOCK_M: tl.constexpr,
) -> None:
    # vllm/models/qwen4_exp/amd/ops/qsa.py @81198e97 _qsa_sparse_paged_gqa_splitk_kernel,
    # plus: rows of dense requests return at once (_qsa_dense_kernel writes them), and a
    # tile whose slots are all -1 is skipped (a no-op in the online softmax).
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    _, _, _, dense = _dense_request(request, seq_lens_ptr, query_start_ptr, num_dense_requests,
                                    GROUP_SIZE, DENSE_BLOCK_M, TOKEN_TOPK)
    if dense:
        return
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
        if tl.max(logical_token, axis=0) >= 0:
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_offset = safe_token % PAGE_SIZE
            valid = ((request >= 0) & (request < num_requests) & (logical_token >= 0)
                     & (logical_page < PAGE_TABLE_WIDTH))
            physical_page = tl.load(
                block_table_ptr + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            keys = tl.load(
                k_cache_ptr + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scores = tl.dot(query, keys)
            scores *= softmax_scale_log2
            scores = tl.where(valid[None, :], scores, -1.0e20)
            next_max = tl.maximum(max_value, tl.max(scores, axis=1))
            alpha = tl.math.exp2(max_value - next_max)
            probabilities = tl.where(
                valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
            accumulator = tl.dot(
                probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        tl.store(
            partial_output_ptr
            + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
               + head_offsets[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


def qsa_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
                  seq_lens, query_start_loc, token_topk, dense=True, dense_config=None):
    """Drop-in for vLLM's qsa_sparse_paged_attention(q, k, v, indices, block_table,
    token_to_req, out) given the main attention metadata's seq_lens / query_start_loc."""
    from vllm.models.qwen4_exp.amd.ops import qsa as stock

    group = q.shape[1] // k_cache.shape[2]
    if (not dense or not q.shape[0] or group > DENSE_BLOCK_M or seq_lens is None
            or query_start_loc is None or seq_lens.dtype != torch.int32
            or query_start_loc.dtype != torch.int32 or not seq_lens.shape[0]
            or query_start_loc.shape != (seq_lens.shape[0] + 1,)):
        return stock.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)
    assert q.dtype == k_cache.dtype == v_cache.dtype == out.dtype == torch.bfloat16
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == out.stride(2) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == token_to_req.stride(0) == 1
    assert seq_lens.is_contiguous() and query_start_loc.is_contiguous()

    # vLLM's split profile (qsa_sparse_paged_attention): the sparse rows keep their exact
    # schedule (bit-identical), the dense requests split their context the same number
    # of ways so both share the partial buffers and one merge.
    block_m = triton.next_power_of_2(group)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    num_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)
    if num_splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32,
                                  device=q.device)
    common = dict(TOKEN_TOPK=token_topk, PAGE_SIZE=k_cache.shape[1],
                  PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group,
                  HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits)
    strides = (q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
               v_cache.stride(0), v_cache.stride(1), v_cache.stride(2))
    _qsa_sparse_rows_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, seq_lens,
        query_start_loc, partial_output, partial_lse, out, *strides,
        logical_indices.stride(0), block_table.stride(0), out.stride(0), out.stride(1),
        q.shape[0], k_cache.shape[0], block_table.shape[0], seq_lens.shape[0],
        TOPK=logical_indices.shape[1], NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n,
        DENSE_BLOCK_M=DENSE_BLOCK_M, num_warps=partial_warps, num_stages=1, **common)
    dense_n, dense_warps, dense_stages = dense_config or DENSE_CONFIG
    _qsa_dense_kernel[(seq_lens.shape[0], k_cache.shape[2], num_splits)](
        q, k_cache, v_cache, block_table, seq_lens, query_start_loc,
        partial_output, partial_lse, out, *strides,
        block_table.stride(0), out.stride(0), out.stride(1),
        q.shape[0], k_cache.shape[0], seq_lens.shape[0],
        BLOCK_M=DENSE_BLOCK_M, BLOCK_N=dense_n, num_warps=dense_warps,
        num_stages=dense_stages, **common)
    if num_splits > 1:
        stock._qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1)
    return out


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.qsa, whose rewritten
    forward_qsa calls _suffix_qsa_attention."""
    module._suffix_qsa_attention = qsa_attention


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next QSA shapes (24 query / 2 KV heads x 256,
    1664-token pages, token top-k 2048, compress 4, indexer 4 heads x 128): selections from
    vLLM's own qsa_select_paged_tokens; vLLM's sparse attention vs qsa_attention. Dense
    requests: the selection must be exactly 0..p and the outputs within the bf16 rounding
    bound; sparse rows bitwise; graph replay with new lengths, tables and indices."""
    from vllm.models.qwen4_exp.amd.ops import qsa as stock

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    dev, bf16, i32 = "cuda", torch.bfloat16, torch.int32
    heads, kv_heads, dim, page, topk, ratio, iheads, idim = 24, 2, 256, 1664, 2048, 4, 4, 128
    width, cpage = topk + ratio - 1, page // ratio
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    def rand(lo, hi, n):
        return torch.randint(lo, hi + 1, (n,), generator=gen).tolist()

    def build(reqs, pad_reqs=0, layers=1):
        """reqs: [(seq_len, q_len)]; pad_reqs graph-padding requests (q_len 5, seq_len 0)
        and two trailing rows mapped to no request."""
        lens = [s for s, _ in reqs] + [0] * pad_reqs
        qlens = [q for _, q in reqs] + [5] * pad_reqs
        n_req, rows = len(lens), sum(qlens) + 2
        qsl = torch.zeros(n_req + 1, dtype=i32)
        qsl[1:] = torch.tensor(qlens).cumsum(0)
        tok2req = torch.full((rows,), -1, dtype=i32)
        pos = torch.zeros(rows, dtype=i32)
        for r in range(n_req):
            a, b = int(qsl[r]), int(qsl[r + 1])
            tok2req[a:b] = r
            pos[a:b] = (torch.arange(b - a) + lens[r] - qlens[r]).clamp(min=0)
        pages_of = [max(1, -(-s // page)) for s in lens]
        n_pages = sum(pages_of) + 3
        perm = torch.randperm(n_pages, generator=gen)
        table = torch.full((n_req, max(pages_of) + 1), -1, dtype=i32)
        at = 0
        for r, n in enumerate(pages_of):
            table[r, :n] = perm[at:at + n]
            at += n
        cw = max(1, -(-max(lens) // ratio) // cpage + 1)
        ctable = torch.randint(0, 64, (n_req, cw), generator=gen, dtype=i32)
        ck = torch.randn(64, cpage, 1, idim, generator=gen).to(bf16)
        qi = torch.randn(rows, iheads, idim, generator=gen).to(bf16)
        st = dict(lens=lens, qlens=qlens, qsl=qsl.to(dev), tok2req=tok2req.to(dev),
                  pos=pos.to(dev), seq=torch.tensor(lens, dtype=i32).to(dev), table=table.to(dev),
                  q=torch.randn(rows, heads, dim, generator=gen).to(bf16).to(dev))
        st["sel"] = stock.qsa_select_paged_tokens(
            qi.to(dev), ck.to(dev), ctable.to(dev), st["tok2req"], st["pos"], st["seq"], topk, ratio)
        # The main cache as forward_qsa sees it: [blocks, kv_heads, page, K|V] transposed
        # to [blocks, page, kv_heads, dim] views of one interleaved tensor.
        st["kv"] = [torch.randn(n_pages, kv_heads, page, 2 * dim, device=dev, dtype=bf16)
                    for _ in range(layers)]
        return st

    def kv_views(raw):
        k, v = raw.transpose(1, 2).split(dim, dim=-1)
        return k, v

    def dense_rows(st):
        mask = torch.zeros(st["q"].shape[0], dtype=torch.bool)
        for r, (s, ql) in enumerate(zip(st["lens"], st["qlens"])):
            if 1 <= ql and ql * (heads // kv_heads) <= DENSE_BLOCK_M and ql <= s <= topk:
                a = int(st["qsl"][r])
                mask[a:a + ql] = True
        return mask.to(dev)

    def run_stock(st, layer=0, out=None):
        k, v = kv_views(st["kv"][layer])
        out = torch.zeros_like(st["q"]) if out is None else out
        return stock.qsa_sparse_paged_attention(st["q"], k, v, st["sel"], st["table"],
                                                st["tok2req"], out)

    def run_new(st, layer=0, out=None, cfg=None):
        k, v = kv_views(st["kv"][layer])
        out = torch.zeros_like(st["q"]) if out is None else out
        return qsa_attention(st["q"], k, v, st["sel"], st["table"], st["tok2req"], out,
                             st["seq"], st["qsl"], topk, dense_config=cfg)

    def compare(st, ref, got):
        dmask = dense_rows(st)
        sel_ok = True  # the premise: a dense row's selection is exactly tokens 0..p
        for row in dmask.nonzero().flatten().tolist():
            s = st["sel"][row]
            s = s[s >= 0].sort().values
            sel_ok &= torch.equal(s, torch.arange(int(st["pos"][row]) + 1, device=dev, dtype=i32))
        sparse_ok = torch.equal(ref[~dmask], got[~dmask])
        r, g = ref[dmask].float(), got[dmask].float()
        diff = (r - g).abs()
        tol = 2**-6 * r.abs() + 2**-9  # ~2 bf16 ulps of the output + P roundings
        worst = (diff / tol).max().item() if diff.numel() else 0.0
        exact = (r == g).float().mean().item() if diff.numel() else 1.0
        ok = sel_ok and sparse_ok and worst <= 1 and bool(torch.isfinite(got).all())
        return ok, (f"{'MATCH' if ok else 'MISMATCH'} dense rows {int(dmask.sum())}/"
                    f"{dmask.numel()} (selection==0..p {sel_ok}, max abs diff "
                    f"{diff.max().item() if diff.numel() else 0:.2e}, worst {worst:.2f} of tol, "
                    f"bit-exact {100 * exact:.2f}%) sparse rows bitwise {sparse_ok}")

    failed = False
    layers = 12  # distinct K/V per call: the 12 QSA layers of a step
    cases = (
        ("c1 x mtp5", [(310, 5)], 0, True),
        ("c8 x mtp5", [(s, 5) for s in rand(100, 600, 8)], 0, True),
        ("c32 x mtp5 + 4 graph-padding reqs", [(s, 5) for s in rand(100, 2000, 32)], 4, True),
        ("budget edges", [(2046, 5), (2047, 5), (2048, 5), (2049, 5), (2052, 5), (3000, 5),
                          (5, 5), (1, 1), (4, 5)], 0, False),
        ("long ctx 40k", [(40000, 5)] * 4, 0, False),
        ("prefill chunk + decodes", [(600, 300), (500, 5), (2048, 5), (40, 40)], 0, False),
        ("q_len 1 rows", [(s, 1) for s in rand(100, 600, 32)], 0, False),
    )
    for name, reqs, pad, timed in cases:
        st = build(reqs, pad, layers if timed else 1)
        ok, msg = compare(st, run_stock(st), run_new(st))
        failed |= not ok
        line = f"{MARK} {name}: {msg}"
        if timed:
            outs = [torch.zeros_like(st["q"]) for _ in range(layers)]
            t_stock = _graph_us(lambda i: run_stock(st, i, outs[i]), layers)[0]
            t_new = _graph_us(lambda i: run_new(st, i, outs[i]), layers)[0]
            line += f" | graphed stock {t_stock:.1f} us -> dense {t_new:.1f} us"
            sweep = []
            for cfg in ((32, 4, 1), (64, 4, 1), (64, 4, 2), (64, 8, 1), (128, 4, 1), (128, 8, 1)):
                try:
                    good = compare(st, run_stock(st), run_new(st, cfg=cfg))[0]
                    us = _graph_us(lambda i: run_new(st, i, outs[i], cfg), layers)[0]
                    sweep.append((us, f"{cfg[0]}/{cfg[1]}w/{cfg[2]}s {us:.1f}"
                                      + ("" if good else " MISMATCH")))
                    failed |= not good
                except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                    sweep.append((float("inf"), f"{cfg} {type(exc).__name__}"))
            line += f" | sweep BLOCK_N/warps/stages us: {' | '.join(s for _, s in sweep)}"
        print(line, flush=True)

    # Graph safety: capture on the c32 layout, then replay after rewriting lengths (some
    # requests long, some short), tables and selections in place.
    st = build([(s, 5) for s in rand(100, 2000, 32)], 4)
    k, v = kv_views(st["kv"][0])
    out = torch.zeros_like(st["q"])
    run_new(st, out=out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_new(st, out=out)
    st2 = build([(s, 5) for s in rand(100, 2000, 24)] + [(s, 5) for s in rand(2049, 4000, 8)], 4)
    if st2["table"].shape[1] > st["table"].shape[1] or st2["kv"][0].shape != st["kv"][0].shape:
        st2["table"] = st2["table"][:, : st["table"].shape[1]]  # keep the captured shapes
    for key in ("qsl", "tok2req", "pos", "seq", "sel", "q"):
        st[key].copy_(st2[key])
    st["table"].fill_(-1)
    st["table"][:, : st2["table"].shape[1]].copy_(st2["table"].clamp(max=st["kv"][0].shape[0] - 1))
    st["lens"], st["qlens"] = st2["lens"], st2["qlens"]
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    ok, msg = compare(st, run_stock(st), out)
    failed |= not ok
    print(f"{MARK} graph replay (captured 32 short, replayed 24 short + 8 long): {msg}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
