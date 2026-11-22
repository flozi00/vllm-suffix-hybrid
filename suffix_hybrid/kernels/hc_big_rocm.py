# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp HC sites at decode-batch M in four launches (SUFFIX_ROCM_HC_BIG=1, ROCm).

At the c32 MTP-4 verify (M = 160) a HyperConnection site is six launches (k29 serving):
vLLM's combine_norm 5.7 us, hc_down's split-K pair 10.0 + 4.1, hc_silu 4.1, hipBLASLt's up
GEMM writing the [M, 10240] gate 11.3, hc_gate_mix 4.6: ~40 us, 96 sites a step. For
SUFFIX_ROCM_HC_BIG_MIN_M < M <= SUFFIX_ROCM_HC_BIG_MAX_M (default 16 < M <= 256):

  combine_norm / grouped norm (vLLM's) -> hc_down's split-K partials at a large-M config
  -> one launch sums them, rounds to bf16 and applies silu(. / HC) on the lora columns
  (the injection columns stay the plain sum, as hc_down_reduce leaves them) -> the up
  GEMM + gate mix on a 2D (row block, BLOCK_N outputs) grid; the gate is never stored.

At M <= MIN_M the site runs HC_DOWN's + HC_FUSE's kernels, above MAX_M vLLM's stock chain,
both exactly as served without this gate (the custom op branches per call, so CUDA graphs
and torch.compile see one op). The patch returns from mix / combine_and_mix at their top,
so it composes with HC_FUSE / HC_DOWN in any order (their rewritten bodies go unreached).

Numerics: vLLM ops/hc.py's rounding points (lora bf16, silu(lora / HC) bf16, gate bf16,
sigmoid(gate) * xn summed in fp32 / HC -> bf16). fp32 orders differ (split count, MFMA K
order, a tree over the 4 streams), so the oracle bounds the outputs against an fp64 chain
next to the stock and today's paths.

    python -m suffix_hybrid.kernels.hc_big_rocm   # GPU oracle + us/site (boot gate hc_big_bench)
"""
import os
import re

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from suffix_hybrid.kernels import hc_down_rocm, hc_fused_rocm
from suffix_hybrid.kernels.hc_wq_rocm import mx4_quant, wq_args

MIN_M = int(os.environ.get("SUFFIX_ROCM_HC_BIG_MIN_M", "16"))
MAX_M = int(os.environ.get("SUFFIX_ROCM_HC_BIG_MAX_M", "256"))
# (up to M, down (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT, warps), up (BLOCK_M, BLOCK_N, warps,
# rows_first)): the hc_big_bench sweep's picks on the MI350P (gate6, 2026-10-10; down /
# up us at M 160: 11.6 / 11.0 vs today's pair 11.4 / up path 15.9).
# SUFFIX_ROCM_HC_BIG_CFG="32:32/64/128/40/4:32/16/4/1,..." overrides this table.
CFG = ((32, (32, 64, 128, 40, 4), (32, 16, 4, 1)),
       (64, (32, 64, 128, 20, 4), (32, 16, 4, 0)),
       (128, (64, 64, 128, 20, 4), (64, 16, 4, 0)),
       (256, (64, 64, 128, 10, 4), (32, 16, 4, 0)))
MARK = "[suffix hc-big]"
_LAST = {}  # the last compiled kernel per launch kind (the oracle's isa stats)


def parse_cfg(spec: str):
    rows = []
    for entry in spec.split(","):
        top, down, up = entry.split(":")
        rows.append((int(top), tuple(map(int, down.split("/"))), tuple(map(int, up.split("/")))))
    return tuple(sorted(rows))


if os.environ.get("SUFFIX_ROCM_HC_BIG_CFG", "").strip():
    CFG = parse_cfg(os.environ["SUFFIX_ROCM_HC_BIG_CFG"])


def _cfg(m: int):
    return next(((d, u) for top, d, u in CFG if m <= top), CFG[-1][1:])


@triton.jit  # MN = M x N stays a multiple of 16 (vector loads), one compile for every M
def _hc_reduce_silu_kernel(p_ptr, y_ptr, MN, N, R, SPLIT: tl.constexpr, HC: tl.constexpr,
                           BLOCK: tl.constexpr):
    # hc_down_reduce's sum (all SPLIT partials in one load, the same tree), rounded to bf16;
    # columns < R (lora) then go through hc_silu: bf16(silu(lora / HC)), computed in fp32.
    S_PAD: tl.constexpr = triton.next_power_of_2(SPLIT)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, S_PAD)
    ok = offs < MN
    p = tl.load(p_ptr + s[:, None] * MN + offs[None, :], mask=(s < SPLIT)[:, None] & ok[None, :],
                other=0.0)
    y = tl.sum(p, axis=0).to(y_ptr.dtype.element_ty)
    x = y.to(tl.float32) / HC
    y = tl.where(offs % N < R, (x * tl.sigmoid(x)).to(y_ptr.dtype.element_ty), y)
    tl.store(y_ptr + offs, y, mask=ok)


@triton.jit(do_not_specialize=["M"])
def _hc_up_mix_kernel(
    a_ptr, w_ptr, xn_ptr, out_ptr, M, stride_a, stride_w, stride_xn, stride_out, s_ptr, stride_s,
    D: tl.constexpr, HC: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_FIRST: tl.constexpr, WQ: tl.constexpr,
):
    # One [BLOCK_M rows, BLOCK_N outputs] tile: a = silu(lora / HC) (bf16) times the HC x
    # BLOCK_N up-weight rows of those outputs (dot column j = stream j // BLOCK_N, column
    # j % BLOCK_N), the bf16 gate, then hc_gate_mix: sigmoid(gate) * xn over the streams / HC.
    if ROWS_FIRST:  # the row blocks of one weight tile are consecutive programs
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    else:
        pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = (rows < M)[:, None]
    k0 = tl.arange(0, K0)
    k1 = K0 + tl.arange(0, K1)
    j = tl.arange(0, HC * BLOCK_N)
    cols = pid_n * BLOCK_N + (j // BLOCK_N) * D + j % BLOCK_N
    a_rows = a_ptr + rows[:, None] * stride_a
    w_cols = w_ptr + cols[None, :] * stride_w
    if WQ == 1:  # hc_wq_rocm MXFP4: a quantized per 32 here (its tail padded to 128 columns)
        k1p = K0 + tl.arange(0, 128)
        a0q, a0s = mx4_quant(tl.load(a_rows + k0[None, :], mask=row_ok, other=0.0).to(tl.float32),
                             K0, BLOCK_M)
        a1q, a1s = mx4_quant(tl.load(a_rows + k1p[None, :], mask=row_ok & (k1p < K0 + K1)[None, :],
                                     other=0.0).to(tl.float32), 128, BLOCK_M)
        sc_rows = s_ptr + cols[:, None] * stride_s
        acc = tl.dot_scaled(a0q, a0s, "e2m1", tl.load(w_cols + tl.arange(0, K0 // 2)[:, None]),
                            tl.load(sc_rows + tl.arange(0, K0 // 32)[None, :]), "e2m1")
        acc = tl.dot_scaled(a1q, a1s, "e2m1", tl.load(w_cols + (K0 // 2 + tl.arange(0, 64))[:, None]),
                            tl.load(sc_rows + (K0 // 32 + tl.arange(0, 4))[None, :]), "e2m1", acc)
    elif WQ == 2:  # hc_wq_rocm FP8: e4m3 widened to bf16, fp32 scale per output column
        acc = tl.dot(tl.load(a_rows + k0[None, :], mask=row_ok, other=0.0),
                     tl.load(w_cols + k0[:, None]).to(tl.bfloat16))
        acc = tl.dot(tl.load(a_rows + k1[None, :], mask=row_ok, other=0.0),
                     tl.load(w_cols + k1[:, None]).to(tl.bfloat16), acc)
        acc = acc * tl.load(s_ptr + cols)[None, :]
    else:
        acc = tl.dot(tl.load(a_rows + k0[None, :], mask=row_ok, other=0.0), tl.load(w_cols + k0[:, None]))
        acc = tl.dot(tl.load(a_rows + k1[None, :], mask=row_ok, other=0.0), tl.load(w_cols + k1[:, None]),
                     acc)
    gate = acc.to(a_ptr.dtype.element_ty).to(tl.float32)
    xn = tl.load(xn_ptr + rows[:, None] * stride_xn + cols[None, :], mask=row_ok, other=0.0)
    mixed = tl.sigmoid(gate) * xn.to(tl.float32)
    y = tl.sum(tl.reshape(mixed, (BLOCK_M, HC, BLOCK_N)), axis=1) / HC
    tl.store(out_ptr + rows[:, None] * stride_out + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :],
             y, mask=row_ok)


def down_silu(x, w, hc_count, rank, cfg, wq=None):
    """[M, N] bf16: x @ w.T as hc_down's split-K partials at cfg (on the SUFFIX_ROCM_HC_WQ
    copy of w when there is one), then one launch for their sum (bf16) with silu(. / HC)
    applied to the lora columns (< rank)."""
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, split, warps = cfg
    wt, st, stride_s, mode = wq_args(w, wq)
    if (split < 2 or w.shape[1] != k or k % (split * bk) or x.stride(1) != 1
            or w.stride(1) != 1 or x.dtype != w.dtype or mode == 1 and bk % 128):
        raise ValueError(f"{MARK} unsupported down: x {tuple(x.shape)} w {tuple(w.shape)} "
                         f"cfg {cfg} WQ {mode}")
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _LAST["down"] = hc_down_rocm._hc_down_kernel[(triton.cdiv(n, bn), split, triton.cdiv(m, bm))](
        x, wt, part, m, n, x.stride(0), wt.stride(0), st, stride_s,
        K_SPLIT=k // split, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WQ=mode,
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16)
    y = x.new_empty((m, n))
    block = max(64, 8192 // triton.next_power_of_2(split))  # hc_down_reduce's tiling
    _LAST["reduce"] = _hc_reduce_silu_kernel[(triton.cdiv(m * n, block),)](
        part, y, m * n, n, rank, SPLIT=split, HC=hc_count, BLOCK=block, num_warps=4)
    return y


def up_mix(a, w_up, xn, hc_count, cfg, wq=None):
    """block_input [M, D] from a = silu(lora / HC): up GEMM + hc_gate_mix, no gate tensor
    (on the SUFFIX_ROCM_HC_WQ copy of w_up when there is one)."""
    m, k = a.shape
    d = w_up.shape[0] // hc_count
    bm, bn, warps, rows_first = cfg
    k0 = 1 << ((k - 1).bit_length() - 1)  # K = K0 + K1, powers of two (320 = 256 + 64)
    k1 = k - k0
    wt, st, stride_s, mode = wq_args(w_up, wq)
    if (w_up.shape[1] != k or xn.shape != (m, w_up.shape[0]) or d % bn or k1 < 16
            or k1 & (k1 - 1) or a.stride(1) != 1 or w_up.stride(1) != 1 or xn.stride(1) != 1
            or mode == 1 and (k1 > 128 or k0 % 128 or wt.shape[1] != (k0 + 128) // 2)):
        raise ValueError(f"{MARK} unsupported up: a {tuple(a.shape)} w_up {tuple(w_up.shape)} "
                         f"xn {tuple(xn.shape)} cfg {cfg} WQ {mode}")
    out = xn.new_empty((m, d))
    if m:
        rb = triton.cdiv(m, bm)
        _LAST["up"] = _hc_up_mix_kernel[(rb, d // bn) if rows_first else (d // bn, rb)](
            a, wt, xn, out, m, a.stride(0), wt.stride(0), xn.stride(0), out.stride(0), st,
            stride_s, D=d, HC=hc_count, K0=k0, K1=k1, BLOCK_M=bm, BLOCK_N=bn,
            ROWS_FIRST=bool(rows_first), WQ=mode,
            num_warps=warps, num_stages=1, matrix_instr_nonkdim=16)
    return out


def _hc_tail(xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, hc_count: int,
             lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(block_input [M, D], down [M, N]): only down's injection columns (>= lora_rank) are
    for the caller; its lora columns hold silu(lora / HC) or lora depending on the path."""
    m = xn.shape[0]
    if MIN_M < m <= MAX_M:
        down_cfg, up_cfg = _cfg(m)
        down = down_silu(xn, w_down, hc_count, lora_rank, down_cfg)
        return up_mix(down[:, :lora_rank], w_up, xn, hc_count, up_cfg), down
    # As served without this gate: HC_DOWN's split-K up to its MAX_M (else hipBLASLt),
    # HC_FUSE's fused silu + up + gate mix up to its MAX_M (else the stock three ops).
    down = hc_down_rocm._hc_down(xn, w_down)
    return hc_fused_rocm._hc_up_gate_mix(down[:, :lora_rank], w_up, xn, hc_count), down


def _hc_tail_fake(xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, hc_count: int,
                  lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (xn.new_empty((xn.shape[0], w_up.shape[0] // hc_count)),
            xn.new_empty((xn.shape[0], w_down.shape[0])))


direct_register_custom_op(op_name="suffix_hc_tail", op_func=_hc_tail, fake_impl=_hc_tail_fake)

_VLLM = {}  # vLLM's grouped_gemma_rmsnorm / hc_combine_norm, bound by install()


def _tail(self, xn):
    w_down = (self.input_mix_weight_down_block_inject if self.use_combine
              else self.input_mix_weight_down).weight
    block_input, down = torch.ops.vllm.suffix_hc_tail(
        xn, w_down, self.input_mix_weight_up.weight, self.hc_count, self.lora_rank)
    injection = (down[:, self.lora_rank:self.lora_rank + self.hc_count] if self.use_combine
                 else None)
    return block_input, injection


def hc_big_mix(self, hidden_states):
    """GatedResidual.mix: vLLM's grouped norm, then suffix_hc_tail."""
    xn = _VLLM["norm"](hidden_states, self.hc_norm.weight, self.config.rms_norm_eps,
                       self.hc_count)
    return (hidden_states, *_tail(self, xn))


def hc_big_combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
    """GatedResidual.combine_and_mix: vLLM's combine_norm, then suffix_hc_tail."""
    hidden_states, xn = _VLLM["combine_norm"](
        hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,
        self.config.rms_norm_eps, self.hc_count)
    return (hidden_states, *_tail(self, xn))


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection, whose
    mix / combine_and_mix now return through hc_big_mix / hc_big_combine_and_mix."""
    _VLLM.update(norm=module.grouped_gemma_rmsnorm, combine_norm=module.hc_combine_norm)
    module.hc_big_mix = hc_big_mix
    module.hc_big_combine_and_mix = hc_big_combine_and_mix


def _isa(kernel) -> str:
    asm = (getattr(kernel, "asm", None) or {}).get("amdgcn", "")

    def grab(tag):
        found = re.search(rf"; {tag}: (\d+)", asm)
        return found.group(1) if found else "?"

    return (f"v{grab('NumVgprs')}+a{grab('NumAgprs')}/o{grab('Occupancy')}"
            + ("" if grab("ScratchSize") in ("0", "?") else f"/spill{grab('ScratchSize')}"))


def _fmt(cfg) -> str:
    return "/".join(map(str, cfg))


def _sweep_down(m):
    bms = [b for b in (32, 64, 128, 256) if b <= max(32, triton.next_power_of_2(m))
           and 3 * b >= m]
    for bm in bms:
        w = 4 if bm <= 64 else 8
        yield from ((bm, 64, 128, 10, w), (bm, 64, 128, 20, w), (bm, 64, 128, 40, w),
                    (bm, 32, 128, 40, w), (bm, 128, 128, 20, 8), (bm, 64, 256, 20, w))


def _sweep_up(m):
    for bm in (b for b in (16, 32, 64, 128, 256) if b <= max(16, triton.next_power_of_2(m))):
        for bn in (16, 32):
            for w in ((4,) if bm <= 32 else (4, 8)):
                yield bm, bn, w, 1
        if bm in (32, 64):
            yield bm, 16, 4, 0


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC shape (hidden 2560, hc 4, lowrank 320,
    per-branch norm): per site kind (combine_and_mix N 336, mix N 336, final mixer N 320) and
    M, vLLM's stock chain vs today's serving path (HC_DOWN split-K at MAX_M 256 + HC_FUSE) vs
    this one against an fp64 chain from the same xn; run-to-run bitwise, the op's dispatch,
    the patched method bodies, graph replay == eager; graphed us per whole site over 24
    weight copies (cold weights); then down / up config sweeps (+ isa) and a config line."""
    from types import SimpleNamespace

    import vllm.model_executor.layers.utils  # noqa: F401 - registers vllm::rocm_unquantized_gemm
    from vllm.models.qwen4_exp.amd.ops import hc as stock

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    linear = torch.nn.functional.linear
    _VLLM.update(norm=stock.grouped_gemma_rmsnorm, combine_norm=stock.hc_combine_norm)
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
        return diff.max().item() / ref.abs().max().item(), diff.mean().item() / ref.abs().mean().item()

    def check(kind, n, m) -> bool:
        """One site kind at M: True when it failed."""
        hidden = torch.randn(m, big, device=dev).to(bf16)
        prev = (0.5 * torch.randn(m, d, device=dev)).to(bf16)
        inj = torch.randn(m, 336, device=dev).to(bf16)[:, rank:rank + hc]  # a strided view

        def site(path, i, cfg=None):
            if kind == "mix":
                xn = stock.grouped_gemma_rmsnorm(hidden, norm_w, eps, hc)
            else:
                xn = stock.hc_combine_norm(hidden, prev, inj, norm_w, eps, hc)[1]
            wd, wu = w_dn[n][i], w_up[i]
            if path == "stock":
                down = linear(xn, wd)
                blk = stock.hc_gate_mix(xn, linear(stock.hc_silu(down[:, :rank], hc), wu), hc)
            elif path == "current":  # serving: SUFFIX_ROCM_HC_DOWN_MAX_M=256, HC_FUSE MAX_M 64
                down = hc_down_rocm._launch(xn, wd)
                blk = hc_fused_rocm._hc_up_gate_mix(down[:, :rank], wu, xn, hc)
            elif path == "op":
                blk, down = torch.ops.vllm.suffix_hc_tail(xn, wd, wu, hc, rank)
            else:
                dc, uc = cfg or _cfg(m)
                down = down_silu(xn, wd, hc, rank, dc)
                blk = up_mix(down[:, :rank], wu, xn, hc, uc)
            return xn, blk, (down[:, rank:rank + hc] if n == 336 else None)

        xn, blk_s, inj_s = site("stock", 0)
        _, blk_c, inj_c = site("current", 0)
        _, blk_n, inj_n = site("new", 0)
        x64 = xn.double()
        lora64 = x64 @ w_dn[n][0].double().T
        s64 = lora64[:, :rank] / hc * torch.sigmoid(lora64[:, :rank] / hc)
        blk64 = (torch.sigmoid(s64 @ w_up[0].double().T) * x64).view(m, hc, d).sum(1) / hc
        e = {p: err(b, blk64) for p, b in (("new", blk_n), ("cur", blk_c), ("stock", blk_s))}
        ok = all(e["new"][t] <= 1.25 * max(e["cur"][t], e["stock"][t]) + 1e-9 for t in (0, 1))
        line = (f"block_input rel err max/mean new {e['new'][0]:.1e}/{e['new'][1]:.1e} cur "
                f"{e['cur'][0]:.1e}/{e['cur'][1]:.1e} stock {e['stock'][0]:.1e}/"
                f"{e['stock'][1]:.1e} bitwise vs cur {100 * (blk_n == blk_c).float().mean():.2f}%")
        if n == 336:
            i64 = lora64[:, rank:rank + hc]
            ei = {p: err(b, i64) for p, b in (("new", inj_n), ("cur", inj_c), ("stock", inj_s))}
            ok &= all(ei["new"][t] <= 1.25 * max(ei["cur"][t], ei["stock"][t]) + 1e-9
                      for t in (0, 1))
            line += (f" | injection max new {ei['new'][0]:.1e} cur {ei['cur'][0]:.1e} stock "
                     f"{ei['stock'][0]:.1e} bitwise vs cur {100 * (inj_n == inj_c).float().mean():.2f}%")
        ok &= torch.equal(site("new", 0)[1], blk_n)  # run-to-run bitwise
        op = site("op", 0)
        ok &= torch.equal(op[1], blk_n if MIN_M < m <= MAX_M else blk_c)  # the op's dispatch
        # The patched method bodies (hc_big_mix / hc_big_combine_and_mix) on a stand-in
        # GatedResidual: norm + op + the injection slice, as the model calls them.
        lin = [SimpleNamespace(weight=w) for w in (norm_w, w_dn[n][0], w_up[0])]
        gr = SimpleNamespace(use_combine=n == 336, hc_count=hc, lora_rank=rank, hc_norm=lin[0],
                             config=SimpleNamespace(rms_norm_eps=eps), input_mix_weight_up=lin[2],
                             input_mix_weight_down_block_inject=lin[1], input_mix_weight_down=lin[1])
        h_w, blk_w, inj_w = (hc_big_mix(gr, hidden) if kind == "mix"
                             else hc_big_combine_and_mix(gr, hidden, prev, inj))
        h_ref = hidden if kind == "mix" else stock.hc_combine_norm(hidden, prev, inj, norm_w, eps, hc)[0]
        ok &= torch.equal(h_w, h_ref) and torch.equal(blk_w, op[1])
        ok &= (inj_w is None) if n == 320 else torch.equal(inj_w, op[2])
        cur_us, _ = _graph_us(lambda i: site("current", i)[1], copies)
        new_us, graphed = _graph_us(lambda i: site("new", i)[1], copies)
        ok &= torch.equal(graphed, blk_n)  # graph replay == eager
        dc, uc = _cfg(m)
        print(f"{MARK} {kind} N={n} M={m}: {'MATCH' if ok else 'MISMATCH'} {line} | site us "
              f"current {cur_us:.1f} -> new {new_us:.1f} [{_fmt(dc)}:{_fmt(uc)}]"
              + ("" if MIN_M < m <= MAX_M else f" (M <= MIN_M={MIN_M}: current path in serving)"),
              flush=True)
        if kind != "combine_and_mix" or m not in (32, 64, 128, 160):
            return not ok
        # Sweeps: the down (+ reduce/silu) and the up stage alone, each config checked against
        # the default config's output (fp32 order only: a few bf16 ulps at most).
        ref_down = down_silu(xn, w_dn[n][0], hc, rank, dc)
        ref_up = up_mix(ref_down[:, :rank], w_up[0], xn, hc, uc)
        for stage, configs, ref, run in (
                ("down", list(_sweep_down(m)), ref_down,
                 lambda i, c: down_silu(xn, w_dn[n][i], hc, rank, c)),
                ("up", list(_sweep_up(m)), ref_up,
                 lambda i, c: up_mix(ref_down[:, :rank], w_up[i], xn, hc, c))):
            cur = (_graph_us(lambda i: hc_down_rocm._launch(xn, w_dn[n][i]), copies)[0]
                   if stage == "down" else
                   _graph_us(lambda i: hc_fused_rocm._hc_up_gate_mix(
                       ref_down[:, :rank], w_up[i], xn, hc), copies)[0])
            rows = []
            for c in configs:
                try:
                    out = run(0, c)
                    good = ((out.float() - ref.float()).abs().max()
                            <= 2**-6 * ref.float().abs().max()).item()
                    us = _graph_us(lambda i: run(i, c), copies)[0]
                    tag = _isa(_LAST.get(stage))
                except Exception as exc:  # noqa: BLE001 - a config that does not build is a datum
                    rows.append((float("inf"), f"{_fmt(c)} {type(exc).__name__}"))
                    continue
                ok &= good
                rows.append((us if good else float("inf"),
                             f"{_fmt(c)} {us:.1f} {tag}{'' if good else ' BAD'}"))
            top = min(rows)
            best[(stage, m)] = top[1].split()[0]
            print(f"{MARK} M={m} {stage} sweep us (today's "
                  f"{'hc_down pair' if stage == 'down' else 'up path'} {cur:.1f}; reduce+silu "
                  f"{_isa(_LAST.get('reduce'))}): {' | '.join(r for _, r in rows)} -> best "
                  f"{top[1]}", flush=True)
        return not ok

    failed = False
    for kind, n in (("combine_and_mix", 336), ("mix", 336), ("final", 320)):
        for m in (17, 32, 64, 96, 128, 160, 192, 256):
            try:
                failed |= check(kind, n, m)
            except Exception as exc:  # noqa: BLE001 - e.g. a default config that does not build
                failed = True
                print(f"{MARK} {kind} N={n} M={m}: ERROR {type(exc).__name__}: {str(exc)[:300]}",
                      flush=True)
    if all(("down", m) in best and ("up", m) in best for m in (32, 64, 128, 160)):
        print(f"{MARK} SUFFIX_ROCM_HC_BIG_CFG=" + ",".join(
            f"{top}:{best[('down', m)]}:{best[('up', m)]}"
            for top, m in ((32, 32), (64, 64), (128, 128), (256, 160))), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
