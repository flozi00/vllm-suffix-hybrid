# SPDX-License-Identifier: Apache-2.0
"""Triton MQA-logits fallback for SM120 (cc 12.0) GPUs.

Implements the three kernels the sparse attention indexer needs from
DeepGEMM, matching the wrapper contract of ``vllm/utils/deep_gemm.py``
(vLLM 0.30) for the FP8 path:

``fp8_fp4_mqa_logits(q=(qv, None), kv=(kv, k_scale), weights, ks, ke,
clean_logits)``
    qv  [M, H, D] float8_e4m3fn, kv [N, D] float8_e4m3fn, k_scale [N] fp32,
    weights [M, H] fp32 (the Q scale, softmax_scale and head_scale are all
    folded in upstream by ``fused_indexer_q_rope_quant``), ks/ke [M] int32
    per-row KV window. Returns [M, N] fp32 with
    ``logits[m, n] = sum_h relu(dot(q[m, h], kv[n])) * weights[m, h] *
    k_scale[n]``; with ``clean_logits=True`` columns outside ``[ks[m],
    ke[m])`` are forced to ``-inf`` (when False the real kernel leaves them
    undefined — we write the unmasked score, a safe superset: every
    consumer restricts its read range to the window).

``fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
schedule_metadata, max_model_len, clean_logits, indices)``
    FP8 indexer K-cache layout — the non-obvious one, read out of
    ``indexer_k_quant_and_cache_kernel`` in ``cache_kernels.cu``: the
    allocation is ``[num_blocks, block_size, D + D/quant_block*4]`` uint8 but
    the BYTES ARE NOT interleaved per token. Each block is a split layout:
    ``block_size*D`` fp8 values first, then ``block_size*4`` fp32 scales at
    ``block_offset*head_dim`` strides (per token) inside the values region.
    ``schedule_metadata`` is accepted and ignored (it exists only to
    distribute SM90/SM100 CTA work).

``get_paged_mqa_logits_metadata(context_lens, block_size, num_slots,
indices)`` returns the trivially-shaped ``[num_slots + 1, 2]`` int32 tensor
vLLM sizes its buffers with; our paged kernel ignores the contents.

Phase 1 scope: FP8 values only, ``quant_block_size == head_dim`` (one fp32
scale per K token — the value every DeepSeek-style indexer in vLLM 0.30
ships). MXFP4 inputs raise ``NotImplementedError`` with guidance instead of
silently misquoting.

Precision note: fp8e4m3 values (4 significant bits) are exactly
representable in bf16, and the SM120 tensor-core bf16 FMA accumulates in
fp32, so the dot products here are exact up to accumulation order — same
guarantee the CPU reference kernel (``fp8_paged_mqa_logits_cpu``) gives.
"""

import torch
import triton
import triton.language as tl

_FP8 = torch.float8_e4m3fn


def _require_fp8(name: str, t: torch.Tensor) -> None:
    if t.dtype != _FP8:
        raise NotImplementedError(
            f"sm120 deep_gemm shim: {name} has dtype {t.dtype}; the Triton "
            "fallback serves the FP8 (float8_e4m3fn) path only. The MXFP4 "
            "packed path (use_fp4_cache=True / q_scale != None) needs a "
            "native SM120 DeepGEMM build — disable the fp4 indexer cache "
            "(run without --use-fp4-cache / MXFP4 quantization) on SM120 "
            "until phase 2 lands."
        )


@triton.jit
def _mqa_logits_kernel(
    q_ptr, kv_ptr, k_scale_ptr, w_ptr, ks_ptr, ke_ptr, out_ptr,
    M, N,
    stride_qm, stride_qh,
    stride_kn,
    stride_wn,
    stride_om,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLEAN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < M
    n_ok = offs_n < N

    # K tile: [BLOCK_N, D] fp8 -> bf16 (exact); scales: [BLOCK_N] fp32.
    kv = tl.load(kv_ptr + offs_n[:, None] * stride_kn + offs_d[None, :],
                 mask=n_ok[:, None], other=0.0).to(tl.bfloat16)
    k_scale = tl.load(k_scale_ptr + offs_n, mask=n_ok, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.static_range(H):
        q = tl.load(q_ptr + offs_m[:, None] * stride_qm + h * stride_qh
                    + offs_d[None, :],
                    mask=m_ok[:, None], other=0.0).to(tl.bfloat16)
        # bf16 x bf16 tensor-core dot, fp32 accumulator: exact products of
        # fp8-exact inputs.
        dot = tl.dot(q, tl.trans(kv), out_dtype=tl.float32)
        w = tl.load(w_ptr + offs_m * stride_wn + h, mask=m_ok, other=0.0)
        acc += tl.maximum(dot, 0.0) * w[:, None]

    logits = acc * k_scale[None, :]
    if CLEAN:
        ks = tl.load(ks_ptr + offs_m, mask=m_ok, other=0)
        ke = tl.load(ke_ptr + offs_m, mask=m_ok, other=0)
        in_window = (offs_n[None, :] >= ks[:, None]) & \
                    (offs_n[None, :] < ke[:, None])
        logits = tl.where(in_window, logits, float("-inf"))

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], logits,
             mask=m_ok[:, None] & n_ok[None, :])


def fp8_fp4_mqa_logits(
    q: tuple,
    kv: tuple,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Triton stand-in for deep_gemm.fp8_fp4_mqa_logits (FP8 path only)."""
    qv, q_scale = q
    kv_v, kv_scale = kv
    if q_scale is not None:
        raise NotImplementedError(
            "sm120 deep_gemm shim: MXFP4 Q (q_scale is not None) is not "
            "supported by the Triton fallback in phase 1 — see "
            "sm120/README.md; run the FP8 indexer cache path on SM120."
        )
    _require_fp8("q values", qv)
    _require_fp8("kv values", kv_v)
    if qv.dim() != 3 or kv_v.dim() != 2 or kv_v.shape[1] != qv.shape[2]:
        raise ValueError(
            f"sm120 mqa logits: expected q [M,H,D] and kv [N,D==qD], got "
            f"{tuple(qv.shape)} / {tuple(kv_v.shape)}"
        )
    M, H, D = qv.shape
    N = kv_v.shape[0]
    if kv_scale.shape[0] != N or kv_scale.numel() != N:
        raise NotImplementedError(
            "sm120 mqa logits: expected exactly one fp32 scale per K token "
            "(quant_block_size == head_dim); got kv_scale shape "
            f"{tuple(kv_scale.shape)} for N={N}."
        )
    if weights.shape != (M, H) or weights.dtype != torch.float32:
        raise ValueError(
            f"sm120 mqa logits: weights must be fp32 [M={M}, H={H}], got "
            f"{tuple(weights.shape)}/{weights.dtype}"
        )
    if M == 0 or N == 0:
        return torch.empty((M, N), dtype=torch.float32, device=qv.device)
    if D != triton.next_power_of_2(D) or H != triton.next_power_of_2(H):
        _bad_dim(max(D, H))

    logits = torch.empty((M, N), dtype=torch.float32, device=qv.device)
    BLOCK_M, BLOCK_N = 16, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mqa_logits_kernel[grid](
        qv, kv_v, kv_scale, weights, cu_seqlen_ks, cu_seqlen_ke, logits,
        M, N,
        qv.stride(0), qv.stride(1),
        kv_v.stride(0),
        weights.stride(0),
        logits.stride(0),
        H=H, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CLEAN=bool(clean_logits),
        num_warps=4,
    )
    return logits


def _bad_dim(d: int):
    raise NotImplementedError(
        f"sm120 mqa logits: head_dim/head-count {d} is not a power of two; "
        "the Triton fallback covers the DeepSeek indexer shapes (D=128, "
        "H in {8,16,32,64}). A native SM120 DeepGEMM is needed otherwise."
    )


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr, cache_ptr, w_ptr, cl_ptr, bt_ptr, out_ptr,
    rows, max_len,
    next_n,
    stride_qm, stride_qh,
    stride_s0,           # byte stride per physical page (uint8 view)
    stride_bt,           # block_tables row stride
    stride_om,
    stride_wn,           # weights row stride
    scale_region_bytes,  # block_size * D: start of the fp32 scale region
    block_size,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLEAN: tl.constexpr,
):
    r = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_ok = offs_n < max_len

    b = r // next_n           # batch slot owning this Q row
    cl = tl.load(cl_ptr + r)  # per-row effective context length
    ok = n_ok & (offs_n < cl)

    # Physical page for each K position via the block table (split-byte
    # layout: values at phys*s0 + within*D, fp32 scale bytes at
    # phys*s0 + scale_region + within*4).
    logical = offs_n // block_size
    within = offs_n % block_size
    phys = tl.load(bt_ptr + b * stride_bt + logical, mask=ok, other=0)
    base = phys.to(tl.int64) * stride_s0 + within.to(tl.int64) * D
    kv_u8 = tl.load(cache_ptr + base[:, None] + offs_d[None, :],
                    mask=ok[:, None], other=0)
    # Bytes are float8_e4m3fn payloads: bitcast, never numeric convert.
    kv = kv_u8.to(tl.float8e4nv, bitcast=True)
    # Rebuild the fp32 per-K scale from its 4 little-endian bytes.
    sb = cache_ptr + phys.to(tl.int64) * stride_s0 + scale_region_bytes \
        + within.to(tl.int64) * 4
    s_u32 = (tl.load(sb + 0, mask=ok, other=0).to(tl.uint32)
             | (tl.load(sb + 1, mask=ok, other=0).to(tl.uint32) << 8)
             | (tl.load(sb + 2, mask=ok, other=0).to(tl.uint32) << 16)
             | (tl.load(sb + 3, mask=ok, other=0).to(tl.uint32) << 24))
    k_scale = s_u32.to(tl.float32, bitcast=True)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for h in tl.static_range(H):
        q = tl.load(q_ptr + r * stride_qm + h * stride_qh + offs_d).to(
            tl.bfloat16)
        # [D] x [BLOCK_N, D]: elementwise multiply + reduce stands in for
        # the rank-1 dot (one Q row per program).
        prod = tl.sum(kv.to(tl.float32) * q.to(tl.float32)[None, :], axis=1)
        w = tl.load(w_ptr + r * stride_wn + h)
        acc += tl.maximum(prod, 0.0) * w

    logits = acc * k_scale
    if CLEAN:
        logits = tl.where(ok, logits, float("-inf"))
    tl.store(out_ptr + r * stride_om + offs_n, logits, mask=n_ok)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_size: int,
    num_sms: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Trivially-valid schedule tensor.

    vLLM (``utils/deep_gemm.get_paged_mqa_logits_metadata``) already reduces
    ``num_sms`` to schedule slots before calling the backend, and asserts the
    returned tensor is shaped ``[slots + 1, 2]``. Our paged kernel is
    data-driven (grid over rows x logit columns) and ignores the contents,
    so we only have to satisfy the shape/device contract.
    """
    return torch.zeros((int(num_sms) + 1, 2), dtype=torch.int32,
                       device=context_lens.device)


def fp8_fp4_paged_mqa_logits(
    q: tuple,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = False,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton stand-in for deep_gemm.fp8_fp4_paged_mqa_logits (FP8 only).

    Reads the paged indexer K-cache through block_tables + context_lens with
    the split values/scale byte layout of
    ``indexer_k_quant_and_cache_kernel``; ``schedule_metadata`` is ignored.
    """
    qv, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "sm120 deep_gemm shim: MXFP4 Q is not supported by the Triton "
            "paged fallback in phase 1 — see sm120/README.md."
        )
    _require_fp8("q values", qv)
    if kv_cache.dtype != torch.uint8:
        raise NotImplementedError(
            f"sm120 paged mqa logits: expected the uint8 indexer K-cache, "
            f"got {kv_cache.dtype}."
        )
    if indices is not None:
        raise NotImplementedError(
            "sm120 paged mqa logits: varlen `indices` remapping is not "
            "implemented in the Triton fallback (phase 1)."
        )
    # sap.py hands the cache through kv_cache_as_quant_view -> 4D
    # [num_blocks, block_size, 1, D+4]; normalise to [num_blocks,
    # block_size, page_bytes].
    if kv_cache.dim() == 4:
        assert kv_cache.shape[2] == 1
        kv_cache = kv_cache.squeeze(2)
    num_blocks, block_size, page_bytes = kv_cache.shape
    # Q arrives as [B, next_n, H, D] on the paged path (sap.py).
    if qv.dim() == 3:
        qv = qv.unsqueeze(1)
    B, next_n, H, D = qv.shape
    rows = B * next_n
    if D != triton.next_power_of_2(D) or H != triton.next_power_of_2(H):
        _bad_dim(max(D, H))
    if context_lens.dim() == 2:
        # (B, next_n) per-row effective lens -> flat per-Q-row lens.
        cl = context_lens.reshape(-1).contiguous()
        if cl.shape[0] != rows:
            raise ValueError(
                f"sm120 paged mqa logits: context_lens has "
                f"{cl.shape[0]} rows, q has {rows}"
            )
    else:
        # 1D per-request lens replicated across next_n (deep_gemm
        # semantics: every Q row of a request sees the same window).
        if context_lens.shape[0] != B:
            raise ValueError(
                f"sm120 paged mqa logits: 1D context_lens has "
                f"{context_lens.shape[0]} entries, q batch is {B}"
            )
        cl = context_lens.repeat_interleave(next_n).contiguous()
    if weights.dtype != torch.float32 or weights.shape[1] != H:
        raise ValueError(
            f"sm120 paged mqa logits: weights must be fp32 [rows={rows}, "
            f"H={H}], got {tuple(weights.shape)}/{weights.dtype}"
        )
    if block_tables.dim() == 1:
        block_tables = block_tables.unsqueeze(-1)
    # The fp8 values occupy the first block_size*D bytes of each page and
    # the fp32 scales the following block_size*4 bytes (verified against
    # indexer_k_quant_and_cache_kernel's offset maths).
    scale_region = block_size * D
    if page_bytes - D != 4:
        raise NotImplementedError(
            f"sm120 paged mqa logits: page width {page_bytes} for D={D} does "
            "not match the FP8 indexer cache contract (head_dim fp8 values + "
            "one fp32 scale per token, i.e. D + D/128*4 with "
            "quant_block_size == 128). Multi-scale-per-token caches need a "
            "dedicated fallback."
        )
    if rows == 0 or max_model_len == 0:
        return torch.empty((rows, max_model_len), dtype=torch.float32,
                           device=qv.device)
    # One row per Q token; reshape normalises non-contiguous pack_seq views
    # so stride(1)/stride(2) address (row, head) unambiguously.
    qv = qv.reshape(rows, H, D).contiguous()

    out = torch.empty((rows, max_model_len), dtype=torch.float32,
                      device=qv.device)
    BLOCK_N = 128
    grid = (rows, triton.cdiv(max_model_len, BLOCK_N))
    _paged_mqa_logits_kernel[grid](
        qv, kv_cache, weights, cl, block_tables, out,
        rows, max_model_len,
        next_n,
        qv.stride(0), qv.stride(1),   # [rows, H, D]: per row, per head
        kv_cache.stride(0),
        block_tables.stride(0),
        out.stride(0),
        weights.stride(0),
        scale_region,
        block_size,
        H=H, D=triton.next_power_of_2(D), BLOCK_N=BLOCK_N,
        CLEAN=bool(clean_logits),
        num_warps=4,
    )
    return out
