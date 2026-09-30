# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of SUFFIX_HC_FUSED_QUANT (see hc_fused_quant.py): vLLM
0.30.0 vllm/models/qwen4_exp/nvidia/ops/hc.py kernel bodies VERBATIM + the
NVFP4 epilogue `_quant_store` (spec quant, IEEE div, 128x4 swizzled scales,
padded scale rows zeroed). Module-level jit functions (Triton resolves
callees through the defining module's globals); imported lazily, pod only.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def _sf_off(row, kb, KB_PAD: tl.constexpr):
    atom = (row // 128) * (KB_PAD // 4) + kb // 4
    return atom * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


@triton.jit
def _reload(y_ptr, row_off, col0, valid_len, NB: tl.constexpr):
    """[NB, 16] fp32 of the bf16 row segment this CTA just stored. The
    barrier makes the stores visible CTA-wide; reading memory (not the
    registers) keeps the stock computation's layouts, hence its reduction
    order and bf16 bits, untouched by the epilogue."""
    tl.debug_barrier()
    b = tl.arange(0, NB)
    e = tl.arange(0, 16)
    ok = (b * 16 < valid_len)[:, None]
    return tl.load(y_ptr + row_off + col0 + b[:, None] * 16 + e[None, :], mask=ok,
                   other=0.0).to(tl.float32)


@triton.jit
def _quant_store(yb, row, M, RM, col0, valid_len, q_ptr, sf_ptr, G, G6,
                 KH: tl.constexpr, KB_PAD: tl.constexpr, NB: tl.constexpr,
                 PR: tl.constexpr):
    # yb: fp32 [NB, 16] (bf16 values) = row elements col0 .. col0 + NB*16.
    amax = tl.max(tl.abs(yb), axis=1)
    sf8 = tl.minimum(amax * G6, 448.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    sff = sf8.to(tl.float32)
    gv = tl.zeros([NB], tl.float32) + G
    inv = tl.where(sff == 0.0, 0.0, tl.math.div_rn(gv, tl.where(sff == 0.0, 1.0, sff)))
    s = yb * inv[:, None]
    a = tl.minimum(tl.abs(s), 6.0)
    code = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32))
    code = code + tl.where(s < 0.0, 8, 0)
    lo, hi = tl.split(tl.reshape(code, (NB, 8, 2)))
    byte = (lo | (hi << 4)).to(tl.uint8)
    b = tl.arange(0, NB)
    valid = b * 16 < valid_len
    kb = col0 // 16 + b
    j = tl.arange(0, 8)
    tl.store(q_ptr + row * KH + kb[:, None] * 8 + j[None, :], byte, mask=valid[:, None])
    tl.store(sf_ptr + _sf_off(row, kb, KB_PAD), sf8.to(tl.uint8, bitcast=True), mask=valid)
    # padded scale rows [M, RM) of this program's k-blocks -> 0, one masked
    # 2D store of u32 words (4 k-blocks of a row are contiguous; programs own
    # whole 4-block groups): rows M + row + i*M, i < PR = pow2 >= ceil((RM-M)/M)
    wd = tl.arange(0, NB // 4)
    pr = M + row + tl.arange(0, PR) * M
    ok = (pr < RM)[:, None] & (wd * 64 < valid_len)[None, :]
    woff = _sf_off(pr[:, None], col0 // 16 + wd[None, :] * 4, KB_PAD) // 4
    tl.store(sf_ptr.to(tl.pointer_type(tl.int32)) + woff,
             tl.zeros([PR, NB // 4], tl.int32), mask=ok)


# --- stock vLLM 0.30.0 ops/hc.py kernel bodies + the quant epilogue ---


@triton.jit
def _grouped_gemma_rmsnorm_q_kernel(
    x_ptr, w_ptr, y_ptr, q_ptr, sf_ptr, stride_x, stride_y, M, RM, G, G6,
    DIM: tl.constexpr, NUM_GROUPS: tl.constexpr, W_SHARED: tl.constexpr,
    EPS: tl.constexpr, KH: tl.constexpr, KB_PAD: tl.constexpr,
    PR: tl.constexpr, launch_pdl: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)
    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = pid // NUM_GROUPS
    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    w_offs = offs_g if W_SHARED else offsets
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)
    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offsets, y, mask)
    yb = _reload(y_ptr, row * stride_y, group_id * GROUP_DIM, GROUP_DIM, BLOCK_SIZE // 16)
    _quant_store(yb, row, M, RM, group_id * GROUP_DIM, GROUP_DIM, q_ptr, sf_ptr,
                 G, G6, KH, KB_PAD, BLOCK_SIZE // 16, PR)


@triton.jit
def _hc_silu_q_kernel(
    x_ptr, q_ptr, sf_ptr, stride_x, M, RM, G, G6, DIM: tl.constexpr,
    HC: tl.constexpr, KH: tl.constexpr, KB_PAD: tl.constexpr,
    PR: tl.constexpr, launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    # the stock op's bf16 output, quantized in registers (elementwise: no
    # reduction whose order a layout change could perturb)
    yb = tl.reshape(y.to(tl.bfloat16).to(tl.float32), (BLOCK_SIZE // 16, 16))
    _quant_store(yb, row, M, RM, 0, DIM, q_ptr, sf_ptr, G, G6, KH, KB_PAD,
                 BLOCK_SIZE // 16, PR)


@triton.jit
def _hc_gate_mix_q_kernel(
    x_ptr, g_ptr, y_ptr, q_ptr, sf_ptr, q2_ptr, sf2_ptr, stride_x, stride_g,
    stride_y, M, RM, G, G6, G2, G62, DIM: tl.constexpr, HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NQ: tl.constexpr, KH: tl.constexpr,
    KB_PAD: tl.constexpr, PR: tl.constexpr, launch_pdl: tl.constexpr,
) -> None:
    HC_DIM: tl.constexpr = DIM // HC
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HC_DIM
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask)
    yb = tl.reshape(acc.to(y_ptr.dtype.element_ty).to(tl.float32), (BLOCK_SIZE // 16, 16))
    col0 = tile_id * BLOCK_SIZE  # elementwise over the HC streams: layout-free
    _quant_store(yb, row, M, RM, col0, HC_DIM - col0, q_ptr, sf_ptr, G, G6, KH,
                 KB_PAD, BLOCK_SIZE // 16, PR)
    if NQ == 2:
        _quant_store(yb, row, M, RM, col0, HC_DIM - col0, q2_ptr, sf2_ptr, G2, G62,
                     KH, KB_PAD, BLOCK_SIZE // 16, PR)


@triton.jit
def _hc_combine_norm_q_kernel(
    block_ptr, res_ptr, inj_ptr, w_ptr, out_ptr, y_ptr, q_ptr, sf_ptr,
    stride_block, stride_res, stride_inj, stride_out, stride_y, M, RM, G, G6,
    HC_DIM: tl.constexpr, HC: tl.constexpr, W_SHARED: tl.constexpr,
    EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr, KH: tl.constexpr,
    KB_PAD: tl.constexpr, PR: tl.constexpr, launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)
    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    w_offs = offs_inner if W_SHARED else offs
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)
    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    if inj_ptr is not None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)
    yb = _reload(y_ptr, row * stride_y, stream * HC_DIM, HC_DIM,
                 NUM_TILES_PAD * BLOCK_SIZE // 16)
    _quant_store(yb, row, M, RM, stream * HC_DIM, HC_DIM, q_ptr, sf_ptr, G, G6, KH,
                 KB_PAD, NUM_TILES_PAD * BLOCK_SIZE // 16, PR)
