# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp HyperConnection site in two launches (SUFFIX_ROCM_HC_FUSE2=1, ROCm).

With HC_FUSE + HC_DOWN, a GatedResidual.combine_and_mix site is four launches:
hc_combine_norm, hc_down's split-K GEMM, its reduce, and hc_up_gate_mix (~18.7 us per
site at c1 in the k17 profile, 109 sites per MTP-4 decode step). At M <= MAX_M, one
custom op per site (suffix_hc_site) runs instead:

  K1 _hc_norm_down_kernel, grid (N tiles, SPLIT). Each program recomputes the rms of
     its stream for every row from the combined bf16 values (hc_combine_norm's math;
     M x 2560 x 2 bf16 reads, L2-hot after the first program), then streams its K
     slice: combine -> Gemma RMSNorm -> bf16 xn -> MFMA dot with its [BLOCK_N,
     K / SPLIT] slab of the down(+inject) weight -> fp32 partials [SPLIT, M, N]. The
     N-tile-0 programs also write the combined hidden_states and xn.
  K2 _hc_reduce_up_gate_mix_kernel: hc_up_gate_mix with hc_down_reduce as prologue
     (every program sums the SPLIT partials of its rows and rounds lora to bf16);
     program column 0 writes the injection columns.

A config with SPLIT > SPLIT_FUSED keeps the reduce as its own launch (hc_down's
reduce, then hc_up_gate_mix): 3 launches. mix() (first layer, after the PLE combine,
MTP) is the same op without the combine (grouped_gemma_rmsnorm's math); the final
mixers (no injection, down 320) are the same op too. Above MAX_M (c32 verify, prefill)
the op runs the HC_FUSE + HC_DOWN path, each with its own MAX_M.

Numerics: the stock rounding points (combined hidden bf16, xn bf16, lora / injection
bf16 after an fp32 sum, silu bf16, gate bf16, block_input bf16). fp32 orders that
differ from the 4-launch path: the rms sum (2D tile instead of hc_combine_norm's
per-row tile), the split-K sum (sequential in K2 instead of hc_down_reduce's tree).
Deterministic: fixed grids and orders, no atomics, no cross-program sync (device-scope
acq_rel costs 4-8x a launch on the MI350P's per-XCD L2s, see hc_down_rocm).

    python -m suffix_hybrid.kernels.hc_fuse2_rocm   # GPU oracle + us/site (boot gate hc_fuse2_bench)
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from suffix_hybrid.kernels import hc_down_rocm as down
from suffix_hybrid.kernels import hc_fused_rocm as fused

MAX_M = int(os.environ.get("SUFFIX_ROCM_HC_FUSE2_MAX_M", "16"))
SPLIT_FUSED = 16  # K2 sums at most this many partials (unrolled loads); more -> 3 launches
# (BLOCK_N, BLOCK_K, SPLIT, num_warps) per M bound, smallest bound >= M wins.
# SUFFIX_ROCM_HC_FUSE2_CFG="M:BN/BK/SPLIT/WARPS,..." overrides (oracle sweep labels).
# Unmeasured defaults (hc_down's sweep: 96..240 programs stream the 6.9 MB weight fastest).
CONFIGS = {16: (64, 128, 16, 4), 64: (64, 128, 40, 4)}
for _item in filter(None, os.environ.get("SUFFIX_ROCM_HC_FUSE2_CFG", "").split(",")):
    _m, _cfg = _item.split(":")
    CONFIGS[int(_m)] = tuple(int(v) for v in _cfg.split("/"))


@triton.jit(do_not_specialize=["M"])
def _hc_norm_down_kernel(
    res_ptr, blk_ptr, inj_ptr, nw_ptr, w_ptr, hid_ptr, xn_ptr, part_ptr, M, N,
    stride_res, stride_blk, stride_inj, stride_hid, stride_xn, stride_w,
    D: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr, COMBINE: tl.constexpr,
    W_SHARED: tl.constexpr, K_SPLIT: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, RMS_BLOCK: tl.constexpr,
):
    k_lo = tl.program_id(1) * K_SPLIT
    stream = k_lo // D  # K_SPLIT divides D: one stream per program
    rows = tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    res_rows = res_ptr + rows[:, None] * stride_res
    blk_rows = blk_ptr + rows[:, None] * stride_blk
    if COMBINE:  # hc_combine_norm: this row's injection weight for this stream
        inj = tl.load(inj_ptr + rows * stride_inj + stream, mask=rows < M, other=0.0)
        inj = (2.0 * tl.sigmoid(inj.to(tl.float32) / HC))[:, None]
    # Pass 1: rms of (row, stream) over the combined values, rounded as materialized.
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d0 in range(0, D, RMS_BLOCK):
        d = d0 + tl.arange(0, RMS_BLOCK)
        x = tl.load(res_rows + (stream * D + d)[None, :], mask=row_ok, other=0.0)
        if COMBINE:
            b = tl.load(blk_rows + d[None, :], mask=row_ok, other=0.0)
            x = (x.to(tl.float32) + b.to(tl.float32) * inj).to(res_ptr.dtype.element_ty)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)
    rrms = tl.rsqrt(sum_sq / D + EPS)[:, None]
    # Pass 2: this program's K slice -> bf16 xn -> partial dot with its weight slab.
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = (cols < N)[None, :]
    first = tl.program_id(0) == 0  # one N tile writes hidden_states / xn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_lo + K_SPLIT, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        d = k - stream * D
        x = tl.load(res_rows + k[None, :], mask=row_ok, other=0.0)
        if COMBINE:
            b = tl.load(blk_rows + d[None, :], mask=row_ok, other=0.0)
            x = (x.to(tl.float32) + b.to(tl.float32) * inj).to(res_ptr.dtype.element_ty)
            tl.store(hid_ptr + rows[:, None] * stride_hid + k[None, :], x, mask=row_ok & first)
        if W_SHARED:
            nw = tl.load(nw_ptr + d)
        else:
            nw = tl.load(nw_ptr + k)
        # Gemma's (1 + w) affine as the stock kernels write it (one FMA).
        y = x.to(tl.float32) * rrms
        y += y * nw.to(tl.float32)[None, :]
        xn = y.to(res_ptr.dtype.element_ty)
        tl.store(xn_ptr + rows[:, None] * stride_xn + k[None, :], xn, mask=row_ok & first)
        w = tl.load(w_ptr + cols[None, :] * stride_w + k[:, None], mask=col_ok, other=0.0)
        acc = tl.dot(xn, w, acc)
    tl.store(part_ptr + (tl.program_id(1) * M + rows)[:, None] * N + cols[None, :], acc,
             mask=row_ok & col_ok)


@triton.jit(do_not_specialize=["M", "col_blocks"])
def _hc_reduce_up_gate_mix_kernel(
    part_ptr, w_ptr, xn_ptr, out_ptr, inj_ptr, M, col_blocks,
    stride_w, stride_xn, stride_out, stride_inj,
    N_DOWN: tl.constexpr, SPLIT: tl.constexpr, D: tl.constexpr, HC: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, INJ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    k0 = tl.arange(0, K0)
    k1 = K0 + tl.arange(0, K1)
    # hc_down_reduce: fp32 sum of the split partials, one RNE rounding to bf16 (lora).
    p_rows = part_ptr + rows[:, None] * N_DOWN
    s0 = tl.zeros((BLOCK_M, K0), dtype=tl.float32)
    s1 = tl.zeros((BLOCK_M, K1), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        s0 += tl.load(p_rows + s * M * N_DOWN + k0[None, :], mask=row_ok, other=0.0)
        s1 += tl.load(p_rows + s * M * N_DOWN + k1[None, :], mask=row_ok, other=0.0)
    if INJ > 0:
        if tl.program_id(1) == 0:
            ki = tl.arange(0, 16)
            inj_ok = row_ok & (ki < INJ)[None, :]
            si = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                si += tl.load(p_rows + s * M * N_DOWN + (K0 + K1 + ki)[None, :], mask=inj_ok,
                              other=0.0)
            tl.store(inj_ptr + rows[:, None] * stride_inj + ki[None, :], si, mask=inj_ok)
    # From here hc_fused_rocm._hc_up_gate_mix_kernel on that bf16 lora.
    x = s0.to(out_ptr.dtype.element_ty).to(tl.float32) / HC
    a0 = (x * tl.sigmoid(x)).to(out_ptr.dtype.element_ty)
    x = s1.to(out_ptr.dtype.element_ty).to(tl.float32) / HC
    a1 = (x * tl.sigmoid(x)).to(out_ptr.dtype.element_ty)
    j = tl.arange(0, HC * BLOCK_N)
    stream_col = (j // BLOCK_N) * D + j % BLOCK_N
    cols = tl.arange(0, BLOCK_N)
    for cb in range(col_blocks):
        n0 = (tl.program_id(1) * col_blocks + cb) * BLOCK_N
        w_cols = w_ptr + (n0 + stream_col)[None, :] * stride_w
        acc = tl.dot(a0, tl.load(w_cols + k0[:, None]))
        acc = tl.dot(a1, tl.load(w_cols + k1[:, None]), acc)
        gate = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
        xn = tl.load(xn_ptr + rows[:, None] * stride_xn + (n0 + stream_col)[None, :],
                     mask=row_ok, other=0.0)
        mixed = tl.sigmoid(gate) * xn.to(tl.float32)
        y = tl.sum(tl.reshape(mixed, (BLOCK_M, HC, BLOCK_N)), axis=1) / HC
        tl.store(out_ptr + rows[:, None] * stride_out + (n0 + cols)[None, :], y, mask=row_ok)


def _config(m: int):
    return CONFIGS[min((b for b in CONFIGS if b >= m), default=max(CONFIGS))]


def _site(hidden, block, inj, norm_w, w_down, w_up, eps, hc, config=None):
    """K1 (+ hc_down's reduce if SPLIT > SPLIT_FUSED) + K2. Returns (hidden_states or an
    empty placeholder for mix(), xn, block_input, [M, N] buffer whose columns
    lora_rank.. hold the injection)."""
    m, kk = hidden.shape
    n, r = w_down.shape[0], w_up.shape[1]
    d = kk // hc
    bn, bk, split, warps = config or _config(m)
    ks = kk // split
    bm = min(64, max(16, triton.next_power_of_2(m)))
    combine = block is not None
    if (m > 64 or d * hc != kk or w_down.shape[1] != kk or w_up.shape[0] != kk or kk % split
            or d % ks or ks % bk or n < r or r & 15 or norm_w.numel() not in (d, kk)
            or not norm_w.is_contiguous() or hidden.stride(1) != 1 or w_down.stride(1) != 1
            or w_up.stride(1) != 1 or not hidden.dtype == w_down.dtype == w_up.dtype
            or combine and (block.shape != (m, d) or block.stride(1) != 1
                            or inj.shape != (m, hc) or inj.stride(1) != 1)):
        raise ValueError(f"[suffix hc-fuse2] unsupported HC site: hidden {tuple(hidden.shape)} "
                         f"w_down {tuple(w_down.shape)} w_up {tuple(w_up.shape)} hc {hc} "
                         f"config {(bn, bk, split, warps)}")
    hid = torch.empty_like(hidden) if combine else hidden.new_empty((0,))
    xn = torch.empty_like(hidden)
    part = torch.empty((split, m, n), dtype=torch.float32, device=hidden.device)
    _hc_norm_down_kernel[(triton.cdiv(n, bn), split)](
        hidden, block if combine else hidden, inj if combine else hidden, norm_w, w_down,
        hid if combine else hidden, xn, part, m, n,
        hidden.stride(0), block.stride(0) if combine else 0, inj.stride(0) if combine else 0,
        hid.stride(0) if combine else 0, xn.stride(0), w_down.stride(0),
        D=d, HC=hc, EPS=eps, COMBINE=combine, W_SHARED=norm_w.numel() == d, K_SPLIT=ks,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, RMS_BLOCK=min(512, 2048 * warps // bm),
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16)
    if split > SPLIT_FUSED:  # 3 launches: hc_down's reduce writes lora + injection
        y = hidden.new_empty((m, n))
        block_sz = max(64, 8192 // triton.next_power_of_2(split))
        down._hc_down_reduce_kernel[(triton.cdiv(m * n, block_sz),)](
            part, y, m * n, SPLIT=split, BLOCK=block_sz, num_warps=4)
        return hid, xn, fused._launch(y[:, :r], w_up, xn, hc), y
    k0 = 1 << ((r - 1).bit_length() - 1)  # 320 = 256 + 64, as hc_fused_rocm splits it
    y = hidden.new_empty((m, n))  # only the injection columns are written
    out = hidden.new_empty((m, d))
    bm2, groups = fused._config(m, d // fused.BLOCK_N)
    _hc_reduce_up_gate_mix_kernel[(triton.cdiv(m, bm2), groups)](
        part, w_up, xn, out, y[:, r:] if n > r else y, m, d // fused.BLOCK_N // groups,
        w_up.stride(0), xn.stride(0), out.stride(0), y.stride(0),
        N_DOWN=n, SPLIT=split, D=d, HC=hc, K0=k0, K1=r - k0, INJ=hc if n > r else 0,
        BLOCK_M=bm2, BLOCK_N=fused.BLOCK_N, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    return hid, xn, out, y


def _current(hidden, block, inj, norm_w, w_down, w_up, eps, hc):
    """The HC_FUSE + HC_DOWN path (rewritten mix / combine_and_mix), same outputs as _site."""
    from vllm.models.qwen4_exp.amd.ops.hc import grouped_gemma_rmsnorm, hc_combine_norm

    if block is None:
        hid, xn = hidden.new_empty((0,)), grouped_gemma_rmsnorm(hidden, norm_w, eps, hc)
    else:
        hid, xn = hc_combine_norm(hidden, block, inj, norm_w, eps, hc)
    y = down._hc_down(xn, w_down)
    return hid, xn, fused._hc_up_gate_mix(y[:, :w_up.shape[1]], w_up, xn, hc), y


def _hc_site(hidden: torch.Tensor, block: torch.Tensor | None, inj: torch.Tensor | None,
             norm_w: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, eps: float,
             hc_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inside the custom op, so the M branch runs per call (eager / capture), not at trace.
    run = _site if 0 < hidden.shape[0] <= MAX_M else _current
    hid, _, out, y = run(hidden, block, inj, norm_w, w_down, w_up, eps, hc_count)
    return hid, out, y


def _hc_site_fake(hidden: torch.Tensor, block: torch.Tensor | None, inj: torch.Tensor | None,
                  norm_w: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, eps: float,
                  hc_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = hidden.shape[0]
    return (hidden.new_empty(hidden.shape if block is not None else (0,)),
            hidden.new_empty((m, hidden.shape[1] // hc_count)),
            hidden.new_empty((m, w_down.shape[0])))


direct_register_custom_op(op_name="suffix_hc_site", op_func=_hc_site, fake_impl=_hc_site_fake)


def _down_weight(self):
    return (self.input_mix_weight_down_block_inject if self.use_combine
            else self.input_mix_weight_down).weight


def hc_fuse2_mix(self, hidden_states):
    """GatedResidual.mix: (hidden_states, block_input, injection | None)."""
    _, block_input, y = torch.ops.vllm.suffix_hc_site(
        hidden_states, None, None, self.hc_norm.weight, _down_weight(self),
        self.input_mix_weight_up.weight, self.config.rms_norm_eps, self.hc_count)
    injection = y[:, self.lora_rank:self.lora_rank + self.hc_count] if self.use_combine else None
    return hidden_states, block_input, injection


def hc_fuse2_combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
    """GatedResidual.combine_and_mix: (combined hidden_states, block_input, injection | None)."""
    hidden_states, block_input, y = torch.ops.vllm.suffix_hc_site(
        hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,
        _down_weight(self), self.input_mix_weight_up.weight, self.config.rms_norm_eps,
        self.hc_count)
    injection = y[:, self.lora_rank:self.lora_rank + self.hc_count] if self.use_combine else None
    return hidden_states, block_input, injection


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection, whose
    mix / combine_and_mix return through these."""
    module.hc_fuse2_mix = hc_fuse2_mix
    module.hc_fuse2_combine_and_mix = hc_fuse2_combine_and_mix


def _isa_stats(fn) -> list:
    """AMDGCN facts of every compiled variant of a Triton kernel: registers, spills, LDS,
    occupancy, instruction mix."""
    import re
    from collections import Counter

    rows = []
    for entry in getattr(fn, "device_caches", {}).values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for key, ck in getattr(cache, "items", lambda: [])():
            asm = (getattr(ck, "asm", {}) or {}).get("amdgcn", "")
            if not asm:
                continue

            def grab(tag):
                m = re.search(rf"; {tag}: (\d+)", asm)
                return m.group(1) if m else "?"

            ins = [ln.split()[0] for ln in asm.splitlines()
                   if ln.startswith("\t") and ln.strip() and ln.strip()[0] not in ";."]
            cls = Counter("mfma" if "mfma" in i else "lds" if i.startswith("ds_")
                          else "mem" if i.startswith(("global_", "buffer_"))
                          else "valu" if i.startswith("v_") else "salu" if i.startswith("s_")
                          else "other" for i in ins)
            rows.append(f"vgpr {grab('NumVgprs')} agpr {grab('NumAgprs')} scratch "
                        f"{grab('ScratchSize')} lds {grab('LDSByteSize')} occ {grab('Occupancy')} "
                        f"instrs {len(ins)} " + " ".join(f"{k} {v}" for k, v in cls.most_common())
                        + f" | {str(key)[-90:]}")
    return rows


def _sweep_configs():
    """(BN, BK, SPLIT, WARPS): K2-fused splits (<= SPLIT_FUSED) and 3-launch ones."""
    for bn, warps in ((16, 1), (32, 2), (64, 4)):
        for split in (4, 8, 16, 20, 40):
            for bk in (128, 256, 512):
                ks = 10240 // split
                if ks % bk == 0 and ks // bk <= 10 and (split <= SPLIT_FUSED or bn == 64):
                    yield bn, bk, split, warps


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC site (hidden 2560, hc 4, lowrank 320,
    down+inject 336 / final mixer 320, per-branch norm weights): the fused site vs the
    current HC_FUSE + HC_DOWN path vs vLLM's stock chain, each against an fp64 chain from
    the same combined bf16 hidden_states. Then us/site from HIP graphs over weight copies
    (cold weights, as in the model) and a config sweep."""
    import vllm.model_executor.layers.utils  # noqa: F401 - registers vllm::rocm_unquantized_gemm
    from vllm.models.qwen4_exp.amd.ops import hc as stock

    linear = torch.nn.functional.linear
    dev, d, hc, r, eps, copies = "cuda", 2560, 4, 320, 1e-6, 24
    kk = hc * d
    torch.manual_seed(0)
    nw = (0.1 * torch.randn(kk, device=dev)).to(torch.bfloat16)
    wus = [(torch.randn(kk, r, device=dev) * 0.02).to(torch.bfloat16) for _ in range(copies)]
    wds = {n: [(torch.randn(n, kk, device=dev) * 0.01).to(torch.bfloat16)
               for _ in range(copies)] for n in (336, 320)}
    failed = False

    def stock_chain(hidden, block, inj, w_down, w_up):
        if block is None:
            hid, xn = hidden.new_empty((0,)), stock.grouped_gemma_rmsnorm(hidden, nw, eps, hc)
        else:
            hid, xn = stock.hc_combine_norm(hidden, block, inj, nw, eps, hc)
        y = linear(xn, w_down)
        return hid, xn, stock.hc_gate_mix(xn, linear(stock.hc_silu(y[:, :r], hc), w_up), hc), y

    for n, combine in ((336, True), (336, False), (320, True)):
        site = "combine_and_mix" if combine else "mix"
        for m in (1, 2, 5, 8, 16, 32, 40, 64):
            hidden = torch.randn(m, kk, device=dev).to(torch.bfloat16)
            block = (2 * torch.randn(m, d, device=dev)).to(torch.bfloat16) if combine else None
            inj = (torch.randn(m, 336, device=dev).to(torch.bfloat16)[:, r:r + hc]
                   if combine else None)
            args = (hidden, block, inj, nw)
            wd, wu = wds[n][0], wus[0]
            ref_hid, ref_xn, ref_out, ref_y = stock_chain(*args[:3], wd, wu)
            cur = _current(*args, wd, wu, eps, hc)
            new = _site(*args, wd, wu, eps, hc)
            # fp64 chain from the combined bf16 hidden_states (every path materializes it).
            c = (ref_hid if combine else hidden).double().view(m, hc, d)
            w64 = nw.double().view(hc, d)
            x64 = (c / (c.pow(2).mean(-1, keepdim=True) + eps).sqrt() * (1 + w64)).view(m, kk)
            y64 = x64 @ wd.double().T
            s64 = y64[:, :r] / hc
            g64 = (s64 * torch.sigmoid(s64)) @ wu.double().T
            o64 = (torch.sigmoid(g64) * x64).view(m, hc, d).mean(1)

            def err(out, ref):
                diff = (out.double() - ref).abs()
                return (diff.max().item() / ref.abs().max().item(),
                        diff.mean().item() / ref.abs().mean().item())

            ok = torch.equal(new[0], cur[0])  # combined hidden_states: same math, bitwise
            line = []
            for name, idx, ref, cols in (("block_input", 2, o64, slice(None)),
                                         ("injection", 3, y64, slice(r, r + hc))):
                if name == "injection" and n == r:
                    continue
                e_new, e_cur, e_stock = (err(p[idx][:, cols], ref[:, cols])
                                         for p in (new, cur, (ref_hid, ref_xn, ref_out, ref_y)))
                exact = (new[idx][:, cols] == cur[idx][:, cols]).float().mean().item()
                # As close to fp64 as the 4-launch path or stock (max and mean), >= 99 % bitwise.
                good = (e_new[0] <= 1.25 * max(e_cur[0], e_stock[0]) + 1e-6
                        and e_new[1] <= 1.25 * max(e_cur[1], e_stock[1]) + 1e-7 and exact >= 0.99)
                ok &= good
                line.append(f"{name} rel err max/mean new {e_new[0]:.1e}/{e_new[1]:.1e} cur "
                            f"{e_cur[0]:.1e}/{e_cur[1]:.1e} stock {e_stock[0]:.1e}/{e_stock[1]:.1e} "
                            f"bitwise vs cur {100 * exact:.2f}%")
            xn_exact = (new[1] == cur[1]).float().mean().item()
            ok &= xn_exact >= 0.999
            again = _site(*args, wd, wu, eps, hc)  # run-to-run (y holds only the injection)
            ok &= all(torch.equal(a, b) for a, b in zip(again[:3], new[:3]))
            ok &= n == r or torch.equal(again[3][:, r:r + hc], new[3][:, r:r + hc])
            cur_us, _ = fused._graph_us(lambda i: _current(*args, wds[n][i], wus[i], eps, hc)[2],
                                        copies)
            new_us, graphed = fused._graph_us(lambda i: _site(*args, wds[n][i], wus[i], eps, hc)[2],
                                              copies)
            ok &= torch.equal(graphed, new[2])  # graph replay == eager, bitwise
            op = torch.ops.vllm.suffix_hc_site(*args, wd, wu, eps, hc)  # serving dispatch
            want = new if m <= MAX_M else cur
            ok &= torch.equal(op[1], want[2]) and (not combine or torch.equal(op[0], want[0]))
            failed |= not ok
            print(f"[suffix hc-fuse2] {site} N={n} M={m}: {'MATCH' if ok else 'MISMATCH'} hidden "
                  f"bitwise, xn bitwise vs cur {100 * xn_exact:.3f}% | {' | '.join(line)} | "
                  f"current {cur_us:.1f} us -> fused {new_us:.1f} us "
                  f"[{'/'.join(map(str, _config(m)))}]"
                  + ("" if m <= MAX_M else f" (above MAX_M={MAX_M}: current path in serving)"),
                  flush=True)
    for fn in (_hc_norm_down_kernel, _hc_reduce_up_gate_mix_kernel):
        for row in _isa_stats(fn):
            print(f"[suffix hc-fuse2] isa {fn.__name__}: {row}", flush=True)

    best = {}
    for m in (1, 5, 8, 16, 40, 64):  # combine_and_mix, N 336: the 96 per-layer sites
        hidden = torch.randn(m, kk, device=dev).to(torch.bfloat16)
        block = (2 * torch.randn(m, d, device=dev)).to(torch.bfloat16)
        inj = torch.randn(m, 336, device=dev).to(torch.bfloat16)[:, r:r + hc]
        args = (hidden, block, inj, nw)
        cur_out = _current(*args, wds[336][0], wus[0], eps, hc)[2]
        cur_us = fused._graph_us(lambda i: _current(*args, wds[336][i], wus[i], eps, hc)[2],
                                 copies)[0]
        sweep = []
        for cfg in _sweep_configs():
            label = "/".join(map(str, cfg))
            try:
                out = _site(*args, wds[336][0], wus[0], eps, hc, cfg)[2]
                good = (out == cur_out).float().mean().item() >= 0.99
                us = fused._graph_us(lambda i: _site(*args, wds[336][i], wus[i], eps, hc, cfg)[2],
                                     copies)[0]
            except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                sweep.append((float("inf"), f"{label} {type(exc).__name__}"))
                continue
            failed |= not good
            sweep.append((us, f"{label} {us:.1f}{'' if good else ' MISMATCH'}"))
        best[m] = min(sweep)
        print(f"[suffix hc-fuse2] M={m} sweep BN/BK/SPLIT/WARPS us (current {cur_us:.1f}): "
              f"{' | '.join(s for _, s in sweep)} -> best {best[m][1]}", flush=True)
    print("[suffix hc-fuse2] SUFFIX_ROCM_HC_FUSE2_CFG=" + ",".join(
        f"{m}:{b[1].split()[0]}" for m, b in best.items()), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
