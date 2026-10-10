# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp HC down projection as a deterministic split-K GEMM (SUFFIX_ROCM_HC_DOWN=1, ROCm).

Each HyperConnection site of vLLM's AMD Qwen4Exp (109 per MTP-4 decode step of
Qwen3.8-Flash-Next) projects xn [M, 10240] onto the merged down+inject weight [336,
10240] (the final mixers: down only, [320, 10240]). hipBLASLt runs it at decode M as a
split-K GEMM plus a PostGSU reduce (M=5 ~7 + ~4 us, M=40 ~15 us) for a 6.9 MB weight
read (~2 us at HBM bandwidth).

Here: grid (N tiles, SPLIT, M tiles). A program reads one [BLOCK_N, K / SPLIT] weight
slab (default 64 x 256: one 32 KB load, 240 programs, so every slab is in flight at
once on the 128 CUs), dots it with the matching xn slab on the MFMA units and writes
fp32 partials [SPLIT, M, N]; a second launch sums the SPLIT partials of each output in
one load and rounds once to bf16. SPLIT = 1 stores bf16 directly (one launch, N / BLOCK_N
programs; oracle sweep only).

Numerics: bf16 products accumulated in fp32 (MFMA), fp32 partials, an fp32 sum, one RNE
rounding to bf16: the stock GEMM's contract, only the fp32 summation order differs.
Deterministic: split, tiles and the reduce tree are fixed per (M, N, K); no atomics.
HIP-graph safe: grids and the partials buffer follow from shapes; no host sync.

    python -m suffix_hybrid.kernels.hc_down_rocm   # GPU oracle + us/call (boot gate hc_down_bench)
"""
import os
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

TARGET_PROGRAMS = 256  # 2 per CU on the 128-CU MI350P
# SUFFIX_ROCM_HC_DOWN_FUSED=1: the last split program of each output tile sums the partials
# (fixed order) and stores bf16, instead of the second launch (one launch per HC site).
_FUSED = os.environ.get("SUFFIX_ROCM_HC_DOWN_FUSED", "").strip() == "1"
_COUNTERS: dict = {}  # device -> int32 tile counters, zero between launches
# Above MAX_M (c32 verify, prefill) vLLM's rocm_unquantized_gemm stays in charge.
MAX_M = int(os.environ.get("SUFFIX_ROCM_HC_DOWN_MAX_M", "64"))


@triton.jit(do_not_specialize=["M"])
def _hc_down_kernel(
    x_ptr, w_ptr, out_ptr, M, N, stride_x, stride_w,
    K_SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # out[s] = x[:, s K_SPLIT:(s + 1) K_SPLIT] @ w[:, same].T for s = program_id(1): the fp32
    # partials, or y itself when SPLIT == 1 (the store then rounds to bf16).
    rows = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.program_id(1) * K_SPLIT + tl.arange(0, BLOCK_K)
    row_ok = (rows < M)[:, None]
    col_ok = (cols < N)[None, :]
    x_ptrs = x_ptr + rows[:, None] * stride_x + ks[None, :]
    w_ptrs = w_ptr + cols[None, :] * stride_w + ks[:, None]  # [BLOCK_K, BLOCK_N] view of w.T
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K_SPLIT // BLOCK_K):
        acc = tl.dot(tl.load(x_ptrs, mask=row_ok, other=0.0),
                     tl.load(w_ptrs, mask=col_ok, other=0.0), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    out = out_ptr + (tl.program_id(1) * M + rows)[:, None] * N + cols[None, :]
    tl.store(out, acc, mask=row_ok & col_ok)


@triton.jit(do_not_specialize=["M"])
def _hc_down_fused_kernel(
    x_ptr, w_ptr, p_ptr, y_ptr, cnt_ptr, M, N, stride_x, stride_w,
    SPLIT: tl.constexpr, K_SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SG: tl.constexpr,
):
    # _hc_down_kernel's partial, then a per-tile arrival counter (acq_rel: the partials
    # stored before it are visible to whoever arrives last). The last of the SPLIT programs
    # sums all partials of the tile in split order, SG at a time (fixed order: the result
    # does not depend on who arrives last), rounds once to bf16 and resets the counter.
    rows = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.program_id(1) * K_SPLIT + tl.arange(0, BLOCK_K)
    row_ok = (rows < M)[:, None]
    col_ok = (cols < N)[None, :]
    x_ptrs = x_ptr + rows[:, None] * stride_x + ks[None, :]
    w_ptrs = w_ptr + cols[None, :] * stride_w + ks[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K_SPLIT // BLOCK_K):
        acc = tl.dot(tl.load(x_ptrs, mask=row_ok, other=0.0),
                     tl.load(w_ptrs, mask=col_ok, other=0.0), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    off = rows[:, None] * N + cols[None, :]
    ok = row_ok & col_ok
    tl.store(p_ptr + tl.program_id(1) * M * N + off, acc, mask=ok)
    tile = tl.program_id(2) * tl.num_programs(0) + tl.program_id(0)
    if tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu") == SPLIT - 1:
        tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g = tl.arange(0, SG)
        for s0 in range(0, SPLIT, SG):
            part = tl.load(p_ptr + (s0 + g)[:, None, None] * M * N + off[None, :, :],
                           mask=((s0 + g) < SPLIT)[:, None, None] & ok[None, :, :], other=0.0)
            tot += tl.sum(part, axis=0)
        tl.store(y_ptr + off, tot.to(y_ptr.dtype.element_ty), mask=ok)
        tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit  # MN = M x N stays a multiple of 16 (vector loads), one compile for every M
def _hc_down_reduce_kernel(p_ptr, y_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    # y.flat = bf16(sum over s of p[s].flat): all SPLIT partials in one load (one memory
    # round trip, not SPLIT), summed by a fixed-order tree.
    S_PAD: tl.constexpr = triton.next_power_of_2(SPLIT)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, S_PAD)
    ok = offs < MN
    p = tl.load(p_ptr + s[:, None] * MN + offs[None, :], mask=(s < SPLIT)[:, None] & ok[None, :],
                other=0.0)
    tl.store(y_ptr + offs, tl.sum(p, axis=0), mask=ok)


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT, num_warps): one M tile up to 64 rows, 64-column
    slabs, the largest split of K into 128-wide slabs that keeps the grid at <=
    TARGET_PROGRAMS (K = 10240, M <= 64: 40 splits, 6 x 40 = 240 programs; M = 160: 10).
    MI350P sweep 2026-10-09, graphed us: M<=40 6.1..7.3 (256-wide 6.3..7.6), M=160 11.6 (13.5)."""
    bm = min(64, max(16, triton.next_power_of_2(m)))
    bn, bk = 64, 128
    tiles, kb = triton.cdiv(n, bn) * triton.cdiv(m, bm), k // bk
    split = max((s for s in range(1, kb + 1) if kb % s == 0 and tiles * s <= TARGET_PROGRAMS),
                default=1)
    return bm, bn, bk, split, 4


def _launch(x, w, config=None, fused=None):
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, split, warps = config or _config(m, n, k)
    if (w.shape[1] != k or k % (split * bk) or x.stride(1) != 1 or w.stride(1) != 1
            or x.dtype != w.dtype):
        raise ValueError(f"[suffix hc-down] unsupported: x {tuple(x.shape)} {x.dtype} w "
                         f"{tuple(w.shape)} {w.dtype}, K split {split} x {bk}")
    y = x.new_empty((m, n))
    out = y if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    if split > 1 and (_FUSED if fused is None else fused):
        tiles = triton.cdiv(n, bn) * triton.cdiv(m, bm)
        cnt = _COUNTERS.get(x.device)
        if cnt is None or cnt.numel() < tiles:
            cnt = _COUNTERS[x.device] = torch.zeros(max(256, tiles), dtype=torch.int32,
                                                    device=x.device)
        _hc_down_fused_kernel[(triton.cdiv(n, bn), split, triton.cdiv(m, bm))](
            x, w, out, y, cnt, m, n, x.stride(0), w.stride(0),
            SPLIT=split, K_SPLIT=k // split, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            SG=min(8, triton.next_power_of_2(split)), num_warps=warps, num_stages=2,
            matrix_instr_nonkdim=16)
        return y
    _hc_down_kernel[(triton.cdiv(n, bn), split, triton.cdiv(m, bm))](
        x, w, out, m, n, x.stride(0), w.stride(0),
        K_SPLIT=k // split, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16)
    if split > 1:
        block = max(64, 8192 // triton.next_power_of_2(split))  # [S_PAD, BLOCK] = 8K fp32
        _hc_down_reduce_kernel[(triton.cdiv(m * n, block),)](
            out, y, m * n, SPLIT=split, BLOCK=block, num_warps=4)
    return y


def _hc_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Inside the custom op, so the M branch runs per call (eager / capture), not at trace.
    if 0 < x.shape[0] <= MAX_M:
        return _launch(x, weight)
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight)  # UnquantizedLinearMethod's GEMM


def _hc_down_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(op_name="suffix_hc_down", op_func=_hc_down, fake_impl=_hc_down_fake)


def hc_down(x, weight):
    """Drop-in for the HC down projection layer call (bias-free x @ weight.T)."""
    return torch.ops.vllm.suffix_hc_down(x, weight)


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection, whose
    rewritten mix / combine_and_mix call hc_down for both down projections."""
    module.hc_down = hc_down


def _sweep(m: int, n: int, k: int):
    """Oracle configs: SPLIT = 1 at 16-column slabs (one launch, 21 programs at N = 336)
    and every split with 96..640 programs for a few slab shapes."""
    bm = min(64, max(16, triton.next_power_of_2(m)))
    for bn, bk, warps in ((16, 1024, 8), (32, 256, 4), (64, 128, 4), (64, 256, 4),
                          (64, 512, 4), (128, 256, 4)):
        kb = k // bk
        for split in (s for s in range(1, kb + 1) if kb % s == 0):
            programs = triton.cdiv(n, bn) * split * triton.cdiv(m, bm)
            if (split == 1 and bn == 16) or 96 <= programs <= 640:
                yield bm, bn, bk, split, warps


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC down shapes (K = 4 x 2560; N = 336
    merged down+inject, N = 320 final mixers): the kernel (default config, then the sweep
    at N = 336) vs torch.nn.functional.linear (hipBLASLt), us/call from HIP graphs over 48
    weight copies (cold weights, as in the model; 330 MB > the MALL), reduce included.
    M = 160 (c32 verify, above MAX_M) says whether raising SUFFIX_ROCM_HC_DOWN_MAX_M pays."""
    import vllm.model_executor.layers.utils  # noqa: F401 - registers vllm::rocm_unquantized_gemm

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    linear = torch.nn.functional.linear
    dev, k, copies = "cuda", 10240, 48
    torch.manual_seed(0)
    failed = False
    for n in (336, 320):
        ws = [(torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16) for _ in range(copies)]
        w = ws[0]
        for m in (1, 5, 8, 16, 40, 64, 160):
            x = torch.randn(m, k, device=dev).to(torch.bfloat16)
            ref = linear(x, w)
            # Both sides round an fp32 sum of the same bf16 products once (RNE): they differ
            # by <= 1 bf16 ulp (<= 2^-7 |ref|) when the two sums straddle a rounding midpoint,
            # plus the fp32 order difference, bounded per order by lambda sqrt(K) 2^-24
            # sum|x w| (Higham-Mary, lambda = 2): 2^-22 sqrt(K) sum|x w| for two orders. CPU
            # emulation (GSU16 sequential vs 40 splits + tree) uses <= 0.1 % of that term.
            tol = (2**-7 * ref.float().abs()
                   + 2**-22 * k**0.5 * linear(x.float().abs(), w.float().abs()))

            def verdict(out):
                # Within tol everywhere and >= 99 % bit-identical to hipBLASLt (emulated: >=
                # 99.99 %). Emulated bugs: bf16 partials 2.4x tol / 57 %, a lost split 179x.
                diff = (out.float() - ref.float()).abs()
                worst = (diff / tol).max().item()
                exact = (out == ref).float().mean().item()
                return worst <= 1 and exact >= 0.99, diff.max().item(), worst, exact

            out = _launch(x, w, fused=False)
            ok, diff, worst, exact = verdict(out)
            ok &= torch.equal(out, _launch(x, w, fused=False))  # run-to-run bitwise
            stock_us, _ = _graph_us(lambda i: linear(x, ws[i]), copies)
            ours_us, graphed = _graph_us(lambda i: _launch(x, ws[i], fused=False), copies)
            ok &= torch.equal(graphed, out)  # graph replay == eager, bitwise
            fz = _launch(x, w, fused=True)  # SUFFIX_ROCM_HC_DOWN_FUSED: last program reduces
            fok, fdiff, fworst, fexact = verdict(fz)
            fok &= torch.equal(fz, _launch(x, w, fused=True))
            fused_us, fgraphed = _graph_us(lambda i: _launch(x, ws[i], fused=True), copies)
            fok &= torch.equal(fgraphed, fz)
            failed |= not fok
            # The op the rewrite calls: ours up to MAX_M, vLLM's stock dispatch above.
            ok &= torch.equal(_hc_down(x, w), (fz if _FUSED else out) if m <= MAX_M
                              else torch.ops.vllm.rocm_unquantized_gemm(x, w))
            failed |= not ok
            bm, bn, bk, split, warps = _config(m, n, k)
            print(f"[suffix hc-down] N={n} M={m}: max abs diff {diff:.2e} rel "
                  f"{diff / ref.float().abs().max().item():.2e} (worst {worst:.2f} of tol, "
                  f"bit-exact {100 * exact:.3f}%) {'MATCH' if ok else 'MISMATCH'} | hipBLASLt "
                  f"{stock_us:.1f} us -> ours {ours_us:.1f} us [{bn}x{bk}/{split}w{warps}] | "
                  f"fused {fused_us:.1f} us (worst {fworst:.2f}, bit-exact {100 * fexact:.3f}%, "
                  f"{'MATCH' if fok else 'MISMATCH'})"
                  + ("" if m <= MAX_M else f" (above MAX_M={MAX_M}: stock in serving)"),
                  flush=True)
            if n != 336 or "--no-sweep" in sys.argv:
                continue
            sweep = []  # (us, label) per config; one that does not build (LDS) is a datum
            for cfg in _sweep(m, n, k):
                label = f"{cfg[1]}x{cfg[2]}/{cfg[3]}w{cfg[4]}"
                try:
                    good = verdict(_launch(x, w, cfg))[0]
                    us = _graph_us(lambda i: _launch(x, ws[i], cfg), copies)[0]
                except Exception as exc:  # noqa: BLE001
                    sweep.append((float("inf"), f"{label} {type(exc).__name__}"))
                    continue
                failed |= not good
                sweep.append((us, f"{label} {us:.1f}{'' if good else ' MISMATCH'}"))
            print(f"[suffix hc-down] M={m} sweep BNxBK/SPLITwWARPS us: "
                  f"{' | '.join(s for _, s in sweep)} -> best {min(sweep)[1]}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
