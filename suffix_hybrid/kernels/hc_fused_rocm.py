# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp HC silu + up GEMM + gate mix in one kernel (SUFFIX_ROCM_HC_FUSE=1, ROCm).

Each HyperConnection site of vLLM's AMD Qwen4Exp (GatedResidual.mix /
combine_and_mix; 109 per MTP-4 decode step of Qwen3.8-Flash-Next) ends in three
launches: _hc_silu [M, 320] -> BF16 up GEMM 10240x320 writing the gate
[M, 10240] -> _hc_gate_mix. hc_up_gate_mix computes block_input [M, 2560] from
lora, the up weight and xn in one launch and never writes the gate.

Numerics are vllm/models/qwen4_exp/amd/ops/hc.py's: silu(lora / HC) in fp32
rounded to bf16, the GEMM accumulated in fp32 and rounded to bf16,
sigmoid(gate) * xn summed over the HC streams in fp32, / HC, rounded to bf16.
Only fp32 summation orders differ (MFMA K order; the stream sum is a tree).

Grid (row blocks of BLOCK_M, column groups): a program computes the silu of its
rows once (K = 256 + 64, kept in registers), then loops over its column
blocks; one block is a dot over BLOCK_N columns of all HC streams (HC x 16 = 64
MFMA columns over the 4 warps) plus the gate-mix epilogue. Every column
group recomputes its rows' silu, so _config trades parallelism (128 CUs at
small M) against that redundancy (large M).

    python -m suffix_hybrid.kernels.hc_fused_rocm   # GPU oracle + us/call (boot gate hc_fuse_bench)
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_N = 16  # columns per stream in one block: HC * BLOCK_N dot columns
TARGET_PROGRAMS = 128  # one program per CU (MI350P): no CU runs two silu prologues
# MI350P oracle 2026-10-09 (graphed us/call vs stock silu + hipBLASLt + gate mix):
# M<=16 12 -> 6.7, M=40 12.0 -> 10.6, M=160 15.4 -> 16.7..19.2, M=1024 44.6 -> 51.3.
# Above MAX_M (c32 verify, prefill) the stock three launches stay faster.
MAX_M = int(os.environ.get("SUFFIX_ROCM_HC_FUSE_MAX_M", "64"))


@triton.jit(do_not_specialize=["M", "col_blocks"])
def _hc_up_gate_mix_kernel(
    lora_ptr, w_ptr, xn_ptr, out_ptr, M, col_blocks,
    stride_lora, stride_w, stride_xn, stride_out,
    D: tl.constexpr, HC: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    k0 = tl.arange(0, K0)
    k1 = K0 + tl.arange(0, K1)
    # _hc_silu_kernel: silu(lora / HC) in fp32, rounded to the activation dtype.
    lora_rows = lora_ptr + rows[:, None] * stride_lora
    x = tl.load(lora_rows + k0[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
    a0 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
    x = tl.load(lora_rows + k1[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
    a1 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
    # Dot column j = column j % BLOCK_N of the block in stream j // BLOCK_N, i.e.
    # row s * D + n of W_up and column s * D + n of xn.
    j = tl.arange(0, HC * BLOCK_N)
    stream_col = (j // BLOCK_N) * D + j % BLOCK_N
    cols = tl.arange(0, BLOCK_N)
    for cb in range(col_blocks):
        n0 = (tl.program_id(1) * col_blocks + cb) * BLOCK_N
        w_cols = w_ptr + (n0 + stream_col)[None, :] * stride_w
        acc = tl.dot(a0, tl.load(w_cols + k0[:, None]))
        acc = tl.dot(a1, tl.load(w_cols + k1[:, None]), acc)
        # The up GEMM's output is bf16; _hc_gate_mix_kernel then averages
        # sigmoid(gate) * xn over the streams in fp32.
        gate = acc.to(lora_ptr.dtype.element_ty).to(tl.float32)
        xn = tl.load(xn_ptr + rows[:, None] * stride_xn + (n0 + stream_col)[None, :],
                     mask=row_ok, other=0.0)
        mixed = tl.sigmoid(gate) * xn.to(tl.float32)
        y = tl.sum(tl.reshape(mixed, (BLOCK_M, HC, BLOCK_N)), axis=1) / HC
        tl.store(out_ptr + rows[:, None] * stride_out + (n0 + cols)[None, :], y, mask=row_ok)


@triton.jit(do_not_specialize=["M"])
def _hc_up_gate_mix_ws_kernel(
    lora_ptr, w_ptr, xn_ptr, out_ptr, M,
    stride_lora, stride_w, stride_xn, stride_out,
    D: tl.constexpr, HC: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ROW_GROUPS: tl.constexpr,
):
    # Weight-stationary variant: program (c, r) loads the HC x BLOCK_N weight rows of
    # column block c once and streams row blocks r, r + ROW_GROUPS, ... through them
    # (each row block's silu is recomputed per column block). Same ops in the same
    # order as _hc_up_gate_mix_kernel per output element.
    j = tl.arange(0, HC * BLOCK_N)
    stream_col = (j // BLOCK_N) * D + j % BLOCK_N
    n0 = tl.program_id(0) * BLOCK_N
    k0 = tl.arange(0, K0)
    k1 = K0 + tl.arange(0, K1)
    w_cols = w_ptr + (n0 + stream_col)[None, :] * stride_w
    w0 = tl.load(w_cols + k0[:, None])
    w1 = tl.load(w_cols + k1[:, None])
    cols = tl.arange(0, BLOCK_N)
    for rb in range(tl.program_id(1), tl.cdiv(M, BLOCK_M), ROW_GROUPS):
        rows = rb * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = (rows < M)[:, None]
        lora_rows = lora_ptr + rows[:, None] * stride_lora
        x = tl.load(lora_rows + k0[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
        a0 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
        x = tl.load(lora_rows + k1[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
        a1 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
        acc = tl.dot(a0, w0)
        acc = tl.dot(a1, w1, acc)
        gate = acc.to(lora_ptr.dtype.element_ty).to(tl.float32)
        xn = tl.load(xn_ptr + rows[:, None] * stride_xn + (n0 + stream_col)[None, :],
                     mask=row_ok, other=0.0)
        mixed = tl.sigmoid(gate) * xn.to(tl.float32)
        y = tl.sum(tl.reshape(mixed, (BLOCK_M, HC, BLOCK_N)), axis=1) / HC
        tl.store(out_ptr + rows[:, None] * stride_out + (n0 + cols)[None, :], y, mask=row_ok)


def _config(m: int, n_blocks: int) -> tuple[int, int]:
    """(BLOCK_M, column groups) for m rows: the most groups that keep the grid
    at <= TARGET_PROGRAMS programs (fewer groups = less redundant silu)."""
    bm = 16 if m <= 64 else 32 if m <= 256 else 64
    row_blocks = triton.cdiv(m, bm)
    return bm, max((g for g in range(1, n_blocks + 1)
                    if n_blocks % g == 0 and row_blocks * g <= TARGET_PROGRAMS), default=1)


def _launch(lora, w_up, xn, hc_count, config=None):
    m, k = lora.shape
    n = w_up.shape[0]
    d = n // hc_count
    k0 = 1 << ((k - 1).bit_length() - 1)  # K = K0 + K1, powers of two (320 = 256 + 64)
    k1 = k - k0
    if (w_up.shape[1] != k or xn.shape != (m, n) or d * hc_count != n or d % BLOCK_N
            or k1 < 16 or k1 & (k1 - 1) or hc_count & (hc_count - 1)
            or lora.stride(1) != 1 or w_up.stride(1) != 1 or xn.stride(1) != 1
            or not lora.dtype == w_up.dtype == xn.dtype):
        raise ValueError(f"[suffix hc-fuse] unsupported HC shapes: lora {tuple(lora.shape)} "
                         f"w_up {tuple(w_up.shape)} xn {tuple(xn.shape)} hc_count {hc_count}")
    out = xn.new_empty((m, d))
    if m and config and config[0] == "ws":
        _, bm, row_groups, block_n = config
        _hc_up_gate_mix_ws_kernel[(d // block_n, row_groups)](
            lora, w_up, xn, out, m, lora.stride(0), w_up.stride(0), xn.stride(0), out.stride(0),
            D=d, HC=hc_count, K0=k0, K1=k1, BLOCK_M=bm, BLOCK_N=block_n, ROW_GROUPS=row_groups,
            num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    elif m:
        bm, groups = config or _config(m, d // BLOCK_N)
        _hc_up_gate_mix_kernel[(triton.cdiv(m, bm), groups)](
            lora, w_up, xn, out, m, d // BLOCK_N // groups,
            lora.stride(0), w_up.stride(0), xn.stride(0), out.stride(0),
            D=d, HC=hc_count, K0=k0, K1=k1, BLOCK_M=bm, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
            matrix_instr_nonkdim=16)  # 16x16 MFMA: 4 warps tile 64 columns at every BLOCK_M
    return out


def _hc_up_gate_mix(lora: torch.Tensor, w_up: torch.Tensor, xn: torch.Tensor,
                    hc_count: int) -> torch.Tensor:
    if lora.shape[0] > MAX_M:
        from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu
        return hc_gate_mix(xn, torch.nn.functional.linear(hc_silu(lora, hc_count), w_up), hc_count)
    return _launch(lora, w_up, xn, hc_count)


def _hc_up_gate_mix_fake(lora: torch.Tensor, w_up: torch.Tensor, xn: torch.Tensor,
                         hc_count: int) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc_count))


direct_register_custom_op(op_name="suffix_hc_up_gate_mix", op_func=_hc_up_gate_mix,
                          fake_impl=_hc_up_gate_mix_fake)


def hc_up_gate_mix(lora, w_up, xn, hc_count):
    """Drop-in for hc_gate_mix(xn, F.linear(hc_silu(lora, hc_count), w_up), hc_count)."""
    return torch.ops.vllm.suffix_hc_up_gate_mix(lora, w_up, xn, hc_count)


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection,
    whose rewritten mix / combine_and_mix tails call hc_up_gate_mix."""
    module.hc_up_gate_mix = hc_up_gate_mix


def _graph_us(fn, reps: int, iters: int = 10):
    """us per call of fn(i), replayed from one HIP graph of fn(0..reps-1) (i picks
    the weight copy), and fn(0)'s output from that graph."""
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(i) for i in range(reps)]
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (iters * reps), outs[0]


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC shape (hidden 2560, hc 4,
    lowrank 320): vLLM's hc_silu -> F.linear -> hc_gate_mix vs the fused kernel
    (default config, then every config of the sweep), us/call from HIP graphs over
    48 weight copies (cold weights, as in the model). M=1024 covers prefill."""
    from vllm.models.qwen4_exp.amd.ops import hc as stock

    linear = torch.nn.functional.linear
    dev, d, hc, k, copies = "cuda", 2560, 4, 320, 48
    n_blocks = d // BLOCK_N
    torch.manual_seed(0)
    ws = [(torch.randn(hc * d, k, device=dev) * 0.02).to(torch.bfloat16) for _ in range(copies)]
    w = ws[0]
    w_down = (torch.randn(k + 16, hc * d, device=dev) * 0.02).to(torch.bfloat16)
    failed = False
    for m in (1, 5, 8, 16, 40, 64, 160, 256, 1024):
        # xn as the grouped Gemma RMSNorm leaves it; lora = the [:, :320] view of the
        # merged down+inject GEMM output (row stride 336), as in combine_and_mix.
        xn = (torch.randn(m, hc * d, device=dev)
              * (1 + 0.1 * torch.randn(hc * d, device=dev))).to(torch.bfloat16)
        lora = linear(xn, w_down)[:, :k]
        silu = stock.hc_silu(lora, hc)
        gate = linear(silu, w)
        ref = stock.hc_gate_mix(xn, gate, hc)
        # Per-element bound from the stock bf16 roundings: the output's own (1 ulp
        # <= 2^-7 |ref|) plus, per stream, a gate landing on a neighbouring bf16
        # (2 ulp <= 2^-6 |gate|) or moved by the fp32 summation-order bound
        # (K 2^-23 sum|silu w|), through sigmoid' <= 1/4 and the mean over hc;
        # + fp32 slack of the stream sum.
        x = xn.float().abs().view(m, hc, d)
        dg = 2**-6 * gate.float().abs() + k * 2**-23 * linear(silu.float().abs(), w.float().abs())
        tol = (2**-7 * ref.float().abs() + (x * dg.view(m, hc, d)).sum(1) / (4 * hc)
               + 2**-22 * x.sum(1))

        def verdict(out):
            # Within the bound everywhere and >= 99 % bit-identical: misplaced
            # roundings stay inside the bound but drop that to ~92-96 % (CPU
            # emulation; correct order changes ~0.003 %).
            diff = (out.float() - ref.float()).abs()
            worst = (diff / tol).max().item()
            exact = (out == ref).float().mean().item()
            return worst <= 1 and exact >= 0.99, diff.max().item(), worst, exact

        out = _launch(lora, w, xn, hc)
        ok, diff, worst, exact = verdict(out)
        ok &= torch.equal(out, _launch(lora.contiguous(), w, xn, hc))  # final mixer: row stride 320
        stock_us, _ = _graph_us(lambda i: stock.hc_gate_mix(
            xn, linear(stock.hc_silu(lora, hc), ws[i]), hc), copies)
        fused_us, graphed = _graph_us(lambda i: _launch(lora, ws[i], xn, hc), copies)
        ok &= torch.equal(graphed, out)
        failed |= not ok
        bm, groups = _config(m, n_blocks)
        print(f"[suffix hc-fuse] M={m}: max abs diff {diff:.2e} rel "
              f"{diff / ref.float().abs().max().item():.2e} (worst {worst:.2f} of tol, "
              f"bit-exact {100 * exact:.3f}%) {'MATCH' if ok else 'MISMATCH'} | stock "
              f"{stock_us:.1f} us -> fused {fused_us:.1f} us [BMxNG {bm}x{groups}]", flush=True)
        configs = [(bm, g) for bm in (16, 32, 64, 128)
                   if bm <= max(16, triton.next_power_of_2(m))
                   for g in range(1, n_blocks + 1)
                   if n_blocks % g == 0 and 32 <= triton.cdiv(m, bm) * g <= 320]
        configs += [("ws", bm, rg, bn) for bn in (16, 32) for bm in (16, 32, 64)
                    if bm <= max(16, triton.next_power_of_2(m))
                    for rg in (1, 2, 4) if rg <= triton.cdiv(m, bm)]
        sweep = []  # (us, label) per config: BMxNG row-stationary, ws BM/RG/BN weight-stationary
        for cfg in configs:
            label = f"{cfg[0]}x{cfg[1]}" if cfg[0] != "ws" else f"ws{cfg[1]}/{cfg[2]}/{cfg[3]}"
            try:
                good = verdict(_launch(lora, w, xn, hc, cfg))[0]
                us = _graph_us(lambda i: _launch(lora, ws[i], xn, hc, cfg), copies)[0]
                sweep.append((us, f"{label} {us:.1f}{'' if good else ' MISMATCH'}"))
            except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                good = False
                sweep.append((float("inf"), f"{label} {type(exc).__name__}"))
            failed |= not good
        print(f"[suffix hc-fuse] M={m} sweep us: {' | '.join(s for _, s in sweep)} "
              f"-> best {min(sweep)[1]}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
