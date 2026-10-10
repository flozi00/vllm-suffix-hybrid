# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp HC sites at small M in three launches (SUFFIX_ROCM_HC3=1, ROCm).

A HyperConnection site at decode M is four launches today (k34b c1 profile, ~106 sites a
step, 1.96 of 10.19 ms GPU): vLLM's combine_norm, hc_down's split-K pair, hc_up_gate_mix.
The norm sits in front of the down GEMM only through a per-(row, stream) scalar:
xn_s = h_s * r_s * (1 + w_s) with r_s = rsqrt(mean(h_s^2) + eps), so
lora = sum_s r_s * ((h_s * (1 + w_s)) @ W_down_s^T). At 0 < M <= SUFFIX_ROCM_HC3_MAX_M:

  K1 _hc3_down_kernel, grid (N tiles, SPLIT): vLLM's combine on the fly (h = bf16(hidden
     + block_out * 2 sigmoid(injection / HC)), written for its K slice by N tile 0), fp32
     partials of (h * (1 + w)) @ W_down^T (the fp32 operand split into two bf16 MFMA
     operands, A_SPLIT 2, or rounded once, A_SPLIT 1) and, from N tile 0, fp32 partial
     sums of h^2 (a split never crosses a stream);
  K2 _hc3_reduce_kernel, grid (rows, N blocks): r_s from the sum-of-squares partials, the
     down output (lora + injection) = bf16(sum_s r_s * sum of stream s's partials); r [M, HC]
     for K3;
  K3 _hc3_up_gate_mix_kernel: hc_up_gate_mix, with xn rebuilt on the fly from h, r and the
     norm weight exactly as vLLM's combine_norm rounds it (y = h * r; y += y * w; bf16).

mix() (no combine: h = hidden_states) and the final mixers (down 320, no injection) are the
same kernels. Above MAX_M the ops run the site as served without this gate (vLLM's norm,
then HC_BIG's tail when SUFFIX_ROCM_HC_BIG=1, else HC_DOWN's + HC_FUSE's kernels).

Numerics differ from vLLM's chain on purpose: xn is not rounded to bf16 before the down
GEMM (with A_SPLIT 2 the GEMM sees h * (1 + w) to ~2^-17), r_s and the split / stream sums
run in other fp32 orders. hidden_states are vLLM's combine bit for bit; K3's xn equals
vLLM's whenever r_s does. The oracle bounds block_input / injection against an fp64 chain
next to vLLM's chain and today's path.

    python -m suffix_hybrid.kernels.hc3_rocm   # GPU oracle + us/site (boot gate hc3_bench)
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from suffix_hybrid.kernels import hc_big_rocm, hc_down_rocm, hc_fused_rocm

MAX_M = int(os.environ.get("SUFFIX_ROCM_HC3_MAX_M", "16"))
# (up to M, K1 (BLOCK_N, BLOCK_K, SPLIT, warps, A_SPLIT)): unmeasured defaults (hc_down's
# 64x128/40 is its best at M <= 16); SUFFIX_ROCM_HC3_CFG="16:64/128/40/4/2,..." overrides.
CFG = ((16, (64, 128, 40, 4, 2)), (64, (64, 128, 40, 4, 2)))
REDUCE_BLOCK_N = 128
MARK = "[suffix hc3]"
_LAST = {}  # the last compiled kernel per launch (the oracle's isa stats)
if os.environ.get("SUFFIX_ROCM_HC3_CFG", "").strip():
    CFG = tuple(sorted((int(top), tuple(map(int, cfg.split("/"))))
                       for top, cfg in (e.split(":") for e in
                                        os.environ["SUFFIX_ROCM_HC3_CFG"].split(","))))


def _cfg(m: int):
    return next((c for top, c in CFG if m <= top), CFG[-1][1])


@triton.jit(do_not_specialize=["M"])
def _hc3_down_kernel(
    res_ptr, blk_ptr, inj_ptr, nw_ptr, w_ptr, hid_ptr, part_ptr, ss_ptr, M, N,
    stride_res, stride_blk, stride_inj, stride_hid, stride_w,
    D: tl.constexpr, HC: tl.constexpr, COMBINE: tl.constexpr, W_SHARED: tl.constexpr,
    K_SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    A_SPLIT: tl.constexpr,
):
    split = tl.program_id(1)
    k_lo = split * K_SPLIT
    stream = k_lo // D  # K_SPLIT divides D: one stream per program
    rows = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = (cols < N)[None, :]
    first = tl.program_id(0) == 0  # N tile 0 writes h and the sums of squares
    res_rows = res_ptr + rows[:, None] * stride_res
    blk_rows = blk_ptr + rows[:, None] * stride_blk
    if COMBINE:  # vLLM's hc_combine_norm: 2 sigmoid(logit / HC) of this row's stream
        inj = tl.load(inj_ptr + rows * stride_inj + stream, mask=rows < M, other=0.0)
        inj = (2.0 * tl.sigmoid(inj.to(tl.float32) / HC))[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(k_lo, k_lo + K_SPLIT, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        h = tl.load(res_rows + k[None, :], mask=row_ok, other=0.0)
        if COMBINE:  # rounded as materialized, before anything else reads it
            b = tl.load(blk_rows + (k - stream * D)[None, :], mask=row_ok, other=0.0)
            h = (h.to(tl.float32) + b.to(tl.float32) * inj).to(res_ptr.dtype.element_ty)
            tl.store(hid_ptr + rows[:, None] * stride_hid + k[None, :], h, mask=row_ok & first)
        hf = h.to(tl.float32)
        ss += tl.sum(hf * hf, axis=1)
        if W_SHARED:
            nw = tl.load(nw_ptr + (k - stream * D))
        else:
            nw = tl.load(nw_ptr + k)
        a = hf + hf * nw.to(tl.float32)[None, :]  # h * (1 + w), as vLLM's y += y * w
        w = tl.load(w_ptr + cols[None, :] * stride_w + k[:, None], mask=col_ok, other=0.0)
        a_hi = a.to(w_ptr.dtype.element_ty)
        acc = tl.dot(a_hi, w, acc)
        if A_SPLIT == 2:  # the fp32 operand as hi + lo bf16 parts
            acc = tl.dot((a - a_hi.to(tl.float32)).to(w_ptr.dtype.element_ty), w, acc)
    tl.store(part_ptr + (split * M + rows)[:, None] * N + cols[None, :], acc,
             mask=row_ok & col_ok)
    tl.store(ss_ptr + split * M + rows, ss, mask=(rows < M) & first)


@triton.jit
def _hc3_reduce_kernel(part_ptr, ss_ptr, y_ptr, r_ptr, M, N, stride_y,
                       SPLIT: tl.constexpr, HC: tl.constexpr, D: tl.constexpr,
                       EPS: tl.constexpr, BLOCK_N: tl.constexpr):
    # Row `row`, columns of block program_id(1): y = bf16(sum_s r_s * sum of stream s's
    # partials), r_s = rsqrt(sum of stream s's h^2 partials / D + EPS) (vLLM's formula).
    S_PAD: tl.constexpr = triton.next_power_of_2(SPLIT)
    QPS: tl.constexpr = SPLIT // HC  # splits per stream, stream-major
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < N
    s = tl.arange(0, S_PAD)
    s_ok = s < SPLIT
    ss = tl.load(ss_ptr + s * M + row, mask=s_ok, other=0.0)
    p = tl.load(part_ptr + (s[:, None] * M + row) * N + cols[None, :],
                mask=s_ok[:, None] & col_ok[None, :], other=0.0)
    y = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for st in tl.static_range(HC):
        grp = (s // QPS == st) & s_ok
        r = tl.rsqrt(tl.sum(tl.where(grp, ss, 0.0)) / D + EPS)
        y += r * tl.sum(tl.where(grp[:, None], p, 0.0), axis=0)
        if tl.program_id(1) == 0:
            tl.store(r_ptr + row * HC + st, r)
    tl.store(y_ptr + row * stride_y + cols, y, mask=col_ok)


@triton.jit(do_not_specialize=["M", "col_blocks"])
def _hc3_up_gate_mix_kernel(
    lora_ptr, w_ptr, hid_ptr, r_ptr, nw_ptr, out_ptr, M, col_blocks,
    stride_lora, stride_w, stride_hid, stride_out,
    D: tl.constexpr, HC: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
    W_SHARED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # hc_fused_rocm._hc_up_gate_mix_kernel with xn = bf16(h * r * (1 + w)) built in place.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    k0 = tl.arange(0, K0)
    k1 = K0 + tl.arange(0, K1)
    lora_rows = lora_ptr + rows[:, None] * stride_lora
    x = tl.load(lora_rows + k0[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
    a0 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
    x = tl.load(lora_rows + k1[None, :], mask=row_ok, other=0.0).to(tl.float32) / HC
    a1 = (x * tl.sigmoid(x)).to(lora_ptr.dtype.element_ty)
    j = tl.arange(0, HC * BLOCK_N)
    stream_col = (j // BLOCK_N) * D + j % BLOCK_N
    r = tl.load(r_ptr + rows[:, None] * HC + (j // BLOCK_N)[None, :], mask=row_ok, other=0.0)
    cols = tl.arange(0, BLOCK_N)
    for cb in range(col_blocks):
        n0 = (tl.program_id(1) * col_blocks + cb) * BLOCK_N
        w_cols = w_ptr + (n0 + stream_col)[None, :] * stride_w
        acc = tl.dot(a0, tl.load(w_cols + k0[:, None]))
        acc = tl.dot(a1, tl.load(w_cols + k1[:, None]), acc)
        gate = acc.to(lora_ptr.dtype.element_ty).to(tl.float32)
        h = tl.load(hid_ptr + rows[:, None] * stride_hid + (n0 + stream_col)[None, :],
                    mask=row_ok, other=0.0)
        if W_SHARED:
            nw = tl.load(nw_ptr + n0 + j % BLOCK_N)
        else:
            nw = tl.load(nw_ptr + n0 + stream_col)
        xv = h.to(tl.float32) * r  # vLLM's combine_norm: y = out * rrms; y += y * w
        xv += xv * nw.to(tl.float32)[None, :]
        xn = xv.to(lora_ptr.dtype.element_ty)
        mixed = tl.sigmoid(gate) * xn.to(tl.float32)
        y = tl.sum(tl.reshape(mixed, (BLOCK_M, HC, BLOCK_N)), axis=1) / HC
        tl.store(out_ptr + rows[:, None] * stride_out + (n0 + cols)[None, :], y, mask=row_ok)


def site3(hidden, block, inj, norm_w, w_down, w_up, eps, hc, cfg=None):
    """(hidden_states [M, HC*D] (hidden itself for mix), block_input [M, D], down [M, N]
    (lora + injection), r [M, HC]) in K1 + K2 + K3; block is None for mix()."""
    m, kk = hidden.shape
    n, rank = w_down.shape[0], w_up.shape[1]
    d = kk // hc
    bn, bk, split, warps, a_split = cfg or _cfg(m)
    ks = kk // split if split else 0
    combine = block is not None
    if (not 0 < m <= 64 or d * hc != kk or split % hc or not ks or d % ks or ks % bk
            or w_down.shape[1] != kk or w_up.shape[0] != kk or n < rank or rank & 15
            or norm_w.numel() not in (d, kk) or not norm_w.is_contiguous()
            or hidden.stride(1) != 1 or w_down.stride(1) != 1 or w_up.stride(1) != 1
            or not hidden.dtype == w_down.dtype == w_up.dtype
            or combine and (block.shape != (m, d) or block.stride(1) != 1
                            or inj.shape != (m, hc) or inj.stride(1) != 1)):
        raise ValueError(f"{MARK} unsupported site: hidden {tuple(hidden.shape)} w_down "
                         f"{tuple(w_down.shape)} w_up {tuple(w_up.shape)} hc {hc} cfg "
                         f"{(bn, bk, split, warps, a_split)}")
    dev = hidden.device
    hid = torch.empty_like(hidden) if combine else hidden
    part = torch.empty((split, m, n), dtype=torch.float32, device=dev)
    ss = torch.empty((split, m), dtype=torch.float32, device=dev)
    bm = min(64, max(16, triton.next_power_of_2(m)))
    _LAST["down"] = _hc3_down_kernel[(triton.cdiv(n, bn), split, triton.cdiv(m, bm))](
        hidden, block if combine else hidden, inj if combine else hidden, norm_w, w_down, hid,
        part, ss, m, n, hidden.stride(0), block.stride(0) if combine else 0,
        inj.stride(0) if combine else 0, hid.stride(0), w_down.stride(0),
        D=d, HC=hc, COMBINE=combine, W_SHARED=norm_w.numel() == d, K_SPLIT=ks,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, A_SPLIT=a_split,
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16)
    down = hidden.new_empty((m, n))
    r = torch.empty((m, hc), dtype=torch.float32, device=dev)
    _LAST["reduce"] = _hc3_reduce_kernel[(m, triton.cdiv(n, REDUCE_BLOCK_N))](
        part, ss, down, r, m, n, down.stride(0), SPLIT=split, HC=hc, D=d, EPS=eps,
        BLOCK_N=REDUCE_BLOCK_N, num_warps=4)
    k0 = 1 << ((rank - 1).bit_length() - 1)  # 320 = 256 + 64, as hc_fused_rocm splits it
    out = hidden.new_empty((m, d))
    bm3, groups = hc_fused_rocm._config(m, d // hc_fused_rocm.BLOCK_N)
    _LAST["up"] = _hc3_up_gate_mix_kernel[(triton.cdiv(m, bm3), groups)](
        down, w_up, hid, r, norm_w, out, m, d // hc_fused_rocm.BLOCK_N // groups,
        down.stride(0), w_up.stride(0), hid.stride(0), out.stride(0),
        D=d, HC=hc, K0=k0, K1=rank - k0, W_SHARED=norm_w.numel() == d, BLOCK_M=bm3,
        BLOCK_N=hc_fused_rocm.BLOCK_N, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    return hid, out, down, r


def _served(hidden, block, inj, norm_w, w_down, w_up, eps, hc, rank):
    """The site as served without this gate: vLLM's norm, then HC_BIG's tail (its own M
    branches) when SUFFIX_ROCM_HC_BIG=1, else HC_DOWN's split-K + HC_FUSE's up path."""
    from vllm.models.qwen4_exp.amd.ops.hc import grouped_gemma_rmsnorm, hc_combine_norm

    if block is None:
        hid, xn = hidden, grouped_gemma_rmsnorm(hidden, norm_w, eps, hc)
    else:
        hid, xn = hc_combine_norm(hidden, block, inj, norm_w, eps, hc)
    if os.environ.get("SUFFIX_ROCM_HC_BIG", "").strip() == "1":
        out, down = hc_big_rocm._hc_tail(xn, w_down, w_up, hc, rank)
    else:
        down = hc_down_rocm._hc_down(xn, w_down)
        out = hc_fused_rocm._hc_up_gate_mix(down[:, :rank], w_up, xn, hc)
    return hid, out, down


def _hc3_combine_and_mix(hidden: torch.Tensor, block: torch.Tensor, inj: torch.Tensor,
                         norm_w: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor,
                         eps: float, hc_count: int,
                         lora_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inside the custom op, so the M branch runs per call (eager / capture), not at trace.
    if 0 < hidden.shape[0] <= MAX_M:
        return site3(hidden, block, inj, norm_w, w_down, w_up, eps, hc_count)[:3]
    return _served(hidden, block, inj, norm_w, w_down, w_up, eps, hc_count, lora_rank)


def _hc3_combine_and_mix_fake(hidden: torch.Tensor, block: torch.Tensor, inj: torch.Tensor,
                              norm_w: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor,
                              eps: float, hc_count: int,
                              lora_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = hidden.shape[0]
    return (hidden.new_empty(hidden.shape), hidden.new_empty((m, hidden.shape[1] // hc_count)),
            hidden.new_empty((m, w_down.shape[0])))


def _hc3_mix(hidden: torch.Tensor, norm_w: torch.Tensor, w_down: torch.Tensor,
             w_up: torch.Tensor, eps: float, hc_count: int,
             lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if 0 < hidden.shape[0] <= MAX_M:
        return site3(hidden, None, None, norm_w, w_down, w_up, eps, hc_count)[1:3]
    return _served(hidden, None, None, norm_w, w_down, w_up, eps, hc_count, lora_rank)[1:]


def _hc3_mix_fake(hidden: torch.Tensor, norm_w: torch.Tensor, w_down: torch.Tensor,
                  w_up: torch.Tensor, eps: float, hc_count: int,
                  lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    m = hidden.shape[0]
    return (hidden.new_empty((m, hidden.shape[1] // hc_count)),
            hidden.new_empty((m, w_down.shape[0])))


direct_register_custom_op(op_name="suffix_hc3_combine_and_mix", op_func=_hc3_combine_and_mix,
                          fake_impl=_hc3_combine_and_mix_fake)
direct_register_custom_op(op_name="suffix_hc3_mix", op_func=_hc3_mix, fake_impl=_hc3_mix_fake)


def _down_weight(self):
    return (self.input_mix_weight_down_block_inject if self.use_combine
            else self.input_mix_weight_down).weight


def _injection(self, down):
    return down[:, self.lora_rank:self.lora_rank + self.hc_count] if self.use_combine else None


def hc3_mix(self, hidden_states):
    """GatedResidual.mix: (hidden_states, block_input, injection | None)."""
    block_input, down = torch.ops.vllm.suffix_hc3_mix(
        hidden_states, self.hc_norm.weight, _down_weight(self), self.input_mix_weight_up.weight,
        self.config.rms_norm_eps, self.hc_count, self.lora_rank)
    return hidden_states, block_input, _injection(self, down)


def hc3_combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
    """GatedResidual.combine_and_mix: (combined hidden_states, block_input, injection | None)."""
    hidden_states, block_input, down = torch.ops.vllm.suffix_hc3_combine_and_mix(
        hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,
        _down_weight(self), self.input_mix_weight_up.weight, self.config.rms_norm_eps,
        self.hc_count, self.lora_rank)
    return hidden_states, block_input, _injection(self, down)


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection, whose
    mix / combine_and_mix now return through hc3_mix / hc3_combine_and_mix."""
    module.hc3_mix = hc3_mix
    module.hc3_combine_and_mix = hc3_combine_and_mix


def _sweep(m):
    """K1 configs (BLOCK_N, BLOCK_K, SPLIT, warps, A_SPLIT)."""
    for bn, warps in ((64, 4), (32, 2)):
        for split in (20, 40, 80):
            for bk in (128, 256):
                if 10240 // split % bk == 0:
                    for a_split in (2, 1):
                        yield bn, bk, split, warps, a_split


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC shape (hidden 2560, hc 4, lowrank 320,
    per-branch norm): per site kind (combine_and_mix N 336, mix N 336, final mixer N 320) and
    M, vLLM's stock chain vs today's serving path (vLLM's norm + HC_DOWN split-K + HC_FUSE,
    HC_BIG's tail above 16) vs this one, each against an fp64 chain from the same combined
    hidden_states; hidden_states bitwise, run-to-run bitwise, the ops' dispatch and the
    patched method bodies, graph replay == eager; graphed us per whole site over 24 weight
    copies (cold weights); then a K1 config sweep (+ isa) and a config line."""
    from types import SimpleNamespace

    import vllm.model_executor.layers.utils  # noqa: F401 - registers vllm::rocm_unquantized_gemm
    from vllm.models.qwen4_exp.amd.ops import hc as stock

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    linear = torch.nn.functional.linear
    os.environ["SUFFIX_ROCM_HC_BIG"] = "1"  # the preset's tail above MAX_M (boot gates strip it)
    dev, bf16 = "cuda", torch.bfloat16
    d, hc, rank, copies, eps = 2560, 4, 320, 24, 1e-6
    big = hc * d
    torch.manual_seed(0)
    w_up = [(torch.randn(big, rank, device=dev) * 0.05).to(bf16) for _ in range(copies)]
    w_dn = {n: [(torch.randn(n, big, device=dev) * 0.01).to(bf16) for _ in range(copies)]
            for n in (336, 320)}
    norm_w = (0.2 * torch.randn(big, device=dev)).to(bf16)
    best = {}

    def err(x, ref):
        diff = (x.double() - ref).abs()
        return (diff.max().item() / ref.abs().max().item(),
                diff.mean().item() / ref.abs().mean().item())

    def check(kind, n, m) -> bool:
        """One site kind at M: True when it failed."""
        hidden = torch.randn(m, big, device=dev).to(bf16)
        prev = (0.5 * torch.randn(m, d, device=dev)).to(bf16)
        inj = torch.randn(m, 336, device=dev).to(bf16)[:, rank:rank + hc]  # a strided view
        mix = kind == "mix"

        def site(path, i, cfg=None):
            """(hidden_states, block_input, injection | None) of one path."""
            wd, wu = w_dn[n][i], w_up[i]
            if path == "new":
                hid, blk, down, _ = site3(hidden, None if mix else prev, None if mix else inj,
                                          norm_w, wd, wu, eps, hc, cfg)
            elif path == "op":
                if mix:
                    blk, down = torch.ops.vllm.suffix_hc3_mix(hidden, norm_w, wd, wu, eps, hc,
                                                              rank)
                    hid = hidden
                else:
                    hid, blk, down = torch.ops.vllm.suffix_hc3_combine_and_mix(
                        hidden, prev, inj, norm_w, wd, wu, eps, hc, rank)
            else:
                if mix:
                    hid, xn = hidden, stock.grouped_gemma_rmsnorm(hidden, norm_w, eps, hc)
                else:
                    hid, xn = stock.hc_combine_norm(hidden, prev, inj, norm_w, eps, hc)
                if path == "stock":
                    down = linear(xn, wd)
                    blk = stock.hc_gate_mix(xn, linear(stock.hc_silu(down[:, :rank], hc), wu), hc)
                elif m <= 16:  # served today: HC_DOWN split-K (MAX_M 256) + HC_FUSE
                    down = hc_down_rocm._launch(xn, wd)
                    blk = hc_fused_rocm._hc_up_gate_mix(down[:, :rank], wu, xn, hc)
                else:  # served today above 16: HC_BIG's tail
                    blk, down = hc_big_rocm._hc_tail(xn, wd, wu, hc, rank)
            return hid, blk, (down[:, rank:rank + hc] if n == 336 else None)

        hid_s, blk_s, inj_s = site("stock", 0)
        _, blk_c, inj_c = site("current", 0)
        hid_n, blk_n, inj_n = site("new", 0)
        ok = torch.equal(hid_n, hid_s)  # vLLM's combine, bit for bit
        h64 = hid_s.double().view(m, hc, d)
        x64 = (h64 / (h64.pow(2).mean(-1, keepdim=True) + eps).sqrt()
               * (1 + norm_w.double().view(hc, d))).view(m, big)
        down64 = x64 @ w_dn[n][0].double().T
        s64 = down64[:, :rank] / hc * torch.sigmoid(down64[:, :rank] / hc)
        blk64 = (torch.sigmoid(s64 @ w_up[0].double().T) * x64).view(m, hc, d).sum(1) / hc
        e = {p: err(b, blk64) for p, b in (("new", blk_n), ("cur", blk_c), ("stock", blk_s))}
        ok &= all(e["new"][t] <= 1.25 * max(e["cur"][t], e["stock"][t]) + 1e-9 for t in (0, 1))
        line = (f"hidden bitwise {torch.equal(hid_n, hid_s)} | block_input rel err max/mean new "
                f"{e['new'][0]:.1e}/{e['new'][1]:.1e} cur {e['cur'][0]:.1e}/{e['cur'][1]:.1e} "
                f"stock {e['stock'][0]:.1e}/{e['stock'][1]:.1e}")
        if n == 336:
            i64 = down64[:, rank:rank + hc]
            ei = {p: err(b, i64) for p, b in (("new", inj_n), ("cur", inj_c), ("stock", inj_s))}
            ok &= all(ei["new"][t] <= 1.25 * max(ei["cur"][t], ei["stock"][t]) + 1e-9
                      for t in (0, 1))
            line += (f" | injection max/mean new {ei['new'][0]:.1e}/{ei['new'][1]:.1e} cur "
                     f"{ei['cur'][0]:.1e}/{ei['cur'][1]:.1e} stock {ei['stock'][0]:.1e}/"
                     f"{ei['stock'][1]:.1e}")
        again = site("new", 0)
        ok &= torch.equal(again[1], blk_n) and torch.equal(again[0], hid_n)  # run-to-run
        op = site("op", 0)
        want = (hid_n, blk_n, inj_n) if m <= MAX_M else site("current", 0)
        ok &= torch.equal(op[0], want[0]) and torch.equal(op[1], want[1])
        ok &= n == 320 or torch.equal(op[2], want[2])
        # The patched method bodies on a stand-in GatedResidual, as the model calls them.
        lin = [SimpleNamespace(weight=w) for w in (norm_w, w_dn[n][0], w_up[0])]
        gr = SimpleNamespace(use_combine=n == 336, hc_count=hc, lora_rank=rank, hc_norm=lin[0],
                             config=SimpleNamespace(rms_norm_eps=eps), input_mix_weight_up=lin[2],
                             input_mix_weight_down_block_inject=lin[1], input_mix_weight_down=lin[1])
        h_w, blk_w, inj_w = hc3_mix(gr, hidden) if mix else hc3_combine_and_mix(gr, hidden, prev, inj)
        ok &= torch.equal(h_w, op[0]) and torch.equal(blk_w, op[1])
        ok &= (inj_w is None) if n == 320 else torch.equal(inj_w, op[2])
        cur_us, _ = _graph_us(lambda i: site("current", i)[1], copies)
        new_us, graphed = _graph_us(lambda i: site("new", i)[1], copies)
        ok &= torch.equal(graphed, blk_n)  # graph replay == eager
        print(f"{MARK} {kind} N={n} M={m}: {'MATCH' if ok else 'MISMATCH'} {line} | site us "
              f"current {cur_us:.1f} -> new {new_us:.1f} [{hc_big_rocm._fmt(_cfg(m))}]"
              + ("" if m <= MAX_M else f" (above MAX_M={MAX_M}: current path in serving)"),
              flush=True)
        if kind != "combine_and_mix" or m not in (1, 5, 8, 16):
            return not ok
        rows = []
        for c in _sweep(m):
            try:
                blk = site("new", 0, c)[1]
                good = all(err(blk, blk64)[t] <= 1.25 * max(e["cur"][t], e["stock"][t]) + 1e-9
                           for t in (0, 1))
                us = _graph_us(lambda i: site("new", i, c)[1], copies)[0]
            except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                rows.append((float("inf"), f"{hc_big_rocm._fmt(c)} {type(exc).__name__}"))
                continue
            rows.append((us if good else float("inf"), f"{hc_big_rocm._fmt(c)} {us:.1f} "
                         f"{hc_big_rocm._isa(_LAST.get('down'))}{'' if good else ' BAD'}"))
        top = min(rows)
        best[m] = top[1].split()[0]
        print(f"{MARK} M={m} K1 sweep BN/BK/SPLIT/WARPS/A_SPLIT us/site (current {cur_us:.1f}; "
              f"reduce {hc_big_rocm._isa(_LAST.get('reduce'))}, up "
              f"{hc_big_rocm._isa(_LAST.get('up'))}): {' | '.join(r for _, r in rows)} -> best "
              f"{top[1]}", flush=True)
        return not ok

    failed = False
    for kind, n in (("combine_and_mix", 336), ("mix", 336), ("final", 320)):
        for m in (1, 2, 3, 5, 8, 10, 15, 16, 17, 32, 40, 64):
            try:
                failed |= check(kind, n, m)
            except Exception as exc:  # noqa: BLE001 - e.g. a default config that does not build
                failed = True
                print(f"{MARK} {kind} N={n} M={m}: ERROR {type(exc).__name__}: {str(exc)[:300]}",
                      flush=True)
    for name in ("down", "reduce", "up"):
        print(f"{MARK} isa {name}: {hc_big_rocm._isa(_LAST.get(name))}", flush=True)
    if best:
        print(f"{MARK} SUFFIX_ROCM_HC3_CFG=" + ",".join(f"{m}:{c}" for m, c in sorted(best.items())),
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
