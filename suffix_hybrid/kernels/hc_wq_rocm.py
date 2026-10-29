# SPDX-License-Identifier: Apache-2.0
"""MXFP4 / FP8 weights for the Qwen4Exp HC projections (SUFFIX_ROCM_HC_WQ=mxfp4|fp8, ROCm).

vLLM builds every GatedResidual's projections with quant_config=None: per site the merged
down(+inject) weight [336 or 320, 10240] and the up weight [10240, 320] stay BF16, 13.5 MB
a site, ~1.3 GB an MTP-4 decode step on ~106 sites. With this gate the HC Linear modules'
quant method also keeps a quantized copy of the weight (built in
process_weights_after_loading, registered by the weight's data pointer), and the HC
kernels we serve run their GEMMs on it:

  mxfp4  W4A4 on the gfx950 scaled MFMA (tl.dot_scaled e2m1 x e2m1): the weight is AITER's
         dynamic_mxfp4_quant (e2m1 pairs + e8m0 per 32 along K, K zero-padded to a multiple
         of 128, the scaled MFMA's K), so a checkpoint could ship it as is; the activation
         (xn for the down GEMM, silu(lora / HC) rounded to bf16 for the up GEMM) goes
         through the same quant op inside the GEMM kernel (copied below, checked bitwise
         against dynamic_mxfp4_quant by the oracle). 3.6 MB a site.
  fp8    the fallback: W8A16, e4m3 (OCP, gfx950) with one fp32 scale per output channel
         (amax / 448); the kernel widens the weight to bf16 (exact) and scales the
         accumulator's columns. 6.8 MB a site.

Kernels with a WQ constexpr (0 = BF16 as before, 1 = mxfp4, 2 = fp8): hc_down_rocm's
split-K down GEMM (HC_DOWN, M <= its MAX_M; HC_BIG's down), hc_fused_rocm's up + gate mix
(HC_FUSE, M <= 64), hc_big_rocm's up + mix (HC_BIG, 16 < M <= 256). Larger M (prefill) and
vLLM's stock chain keep the BF16 weight.

    python -m suffix_hybrid.kernels.hc_wq_rocm   # GPU oracle + us/site (boot gate hc_wq_bench)
"""
import os
import sys
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix hc-wq]"
GATE = "SUFFIX_ROCM_HC_WQ"
MODES = {"mxfp4": 1, "fp8": 2}
MODE = MODES.get(os.environ.get(GATE, "").strip().lower(), 0)
K_ALIGN = 128  # MXFP4 K padding: the scaled MFMA's K (16x16x128)
FP8_MAX = 448.0  # e4m3fn
_REG: dict = {}  # weight data_ptr -> QWeight


class QWeight(NamedTuple):
    mode: int        # 1 mxfp4, 2 fp8
    q: torch.Tensor  # mxfp4: [N, Kp / 2] uint8 (e2m1 pairs); fp8: [N, K] float8_e4m3fn
    s: torch.Tensor  # mxfp4: [N, Kp / 32] uint8 (e8m0); fp8: [N] fp32 per output channel
    k: int           # logical K


def quantize(w: torch.Tensor, mode: int) -> QWeight:
    """The quantized copy of a bf16 weight [N, K] (rows = output channels)."""
    n, k = w.shape
    if mode == 1:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant  # importing aiter initializes HIP

        kp = triton.cdiv(k, K_ALIGN) * K_ALIGN
        wp = torch.nn.functional.pad(w, (0, kp - k)) if kp != k else w
        q, s = dynamic_mxfp4_quant(wp.contiguous())
        return QWeight(1, q.contiguous(), s.contiguous(), k)
    if mode == 2:
        s = w.abs().amax(dim=1).float().clamp_min(1e-30) / FP8_MAX
        return QWeight(2, (w.float() / s[:, None]).clamp(-FP8_MAX, FP8_MAX)
                       .to(torch.float8_e4m3fn).contiguous(), s.contiguous(), k)
    raise ValueError(f"{MARK} unknown mode {mode}")


def lookup(w: torch.Tensor):
    """The registered quantized copy of an HC weight, or None (gate off / not an HC weight)."""
    return _REG.get(w.data_ptr()) if MODE else None


def wq_args(w: torch.Tensor, wq=None):
    """(weight, scale, scale row stride, WQ constexpr) for an HC kernel: the bf16 weight, or
    its quantized copy (wq given, else the registered one)."""
    q = wq if wq is not None else lookup(w)
    if q is None:
        return w, w, 0, 0
    return q.q, q.s, q.s.stride(0) if q.mode == 1 else 0, q.mode


# ---- AITER's MXFP4 activation quant (aiter/ops/triton/_triton_kernels/quant/quant.py @
# v0.1.24.post1, MIT: _mxfp4_scale_from_amax, _mxfp4_pack_bits, _mxfp4_quant_op with
# SCALING_MODE 0 and the bit-arithmetic pack, i.e. what dynamic_mxfp4_quant runs). Copied
# because importing aiter at module import initializes HIP.
@triton.jit
def _mx4_scale_from_amax(amax):
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    return bs_e8m0, quant_scale


@triton.jit
def _mx4_pack_bits(qx, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   MXFP4_QUANT_BLOCK_SIZE: tl.constexpr):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)
    denorm_exp: tl.constexpr = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    return x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)


@triton.jit
def mx4_quant(x, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """fp32 x [BLOCK_SIZE_M, BLOCK_SIZE_N] -> (e2m1 pairs [M, N / 2] uint8, e8m0 [M, N / 32])."""
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // 32
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 32)
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    bs_e8m0, quant_scale = _mx4_scale_from_amax(amax)
    x_fp4 = _mx4_pack_bits(x * quant_scale, BLOCK_SIZE_N, BLOCK_SIZE_M, 32)
    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _mx4_quant_probe_kernel(x_ptr, q_ptr, s_ptr, stride_x, BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    # Oracle only: one [BLOCK_M, BLOCK_N] tile through mx4_quant, row-major outputs.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    q, s = mx4_quant(tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :]).to(tl.float32),
                     BLOCK_N, BLOCK_M)
    nq = tl.num_programs(1) * (BLOCK_N // 2)
    ns = tl.num_programs(1) * (BLOCK_N // 32)
    tl.store(q_ptr + rows[:, None] * nq + (tl.program_id(1) * (BLOCK_N // 2)
                                           + tl.arange(0, BLOCK_N // 2))[None, :], q)
    tl.store(s_ptr + rows[:, None] * ns + (tl.program_id(1) * (BLOCK_N // 32)
                                           + tl.arange(0, BLOCK_N // 32))[None, :], s)


_HC_LINEARS = ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up")


def install(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.hyperconnection: every
    GatedResidual's HC Linear modules get a quant method that also registers the quantized
    copy of the weight once vLLM has loaded it."""
    if not MODE:
        raise SystemExit(f"{MARK} {GATE}={os.environ.get(GATE)!r}: use mxfp4 or fp8")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class HcWqLinearMethod(UnquantizedLinearMethod):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def process_weights_after_loading(self, layer):
            self.inner.process_weights_after_loading(layer)
            _REG[layer.weight.data_ptr()] = quantize(layer.weight.data, MODE)
            if len(_REG) == 1:
                print(f"{MARK} ACTIVE: HC projections on {os.environ[GATE]} (first weight "
                      f"{tuple(layer.weight.shape)})", file=sys.stderr, flush=True)

        def apply(self, layer, x, bias=None):
            return self.inner.apply(layer, x, bias)

    cls = module.GatedResidual
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        for name in _HC_LINEARS:
            lin = getattr(self, name, None)
            qm = getattr(lin, "quant_method", None)
            if isinstance(qm, UnquantizedLinearMethod) and not isinstance(qm, HcWqLinearMethod):
                lin.quant_method = HcWqLinearMethod(qm)

    cls.__init__ = __init__


def _isa(fn) -> str:
    """vgpr/agpr/occupancy/spills of the compiled variants of a Triton kernel."""
    import re

    out = []
    for entry in getattr(fn, "device_caches", {}).values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for ck in getattr(cache, "values", lambda: [])():
            asm = (getattr(ck, "asm", None) or {}).get("amdgcn", "")

            def grab(tag):
                found = re.search(rf"; {tag}: (\d+)", asm)
                return found.group(1) if found else "?"

            out.append(f"v{grab('NumVgprs')}+a{grab('NumAgprs')}/o{grab('Occupancy')}"
                       + ("" if grab("ScratchSize") in ("0", "?") else
                          f"/spill{grab('ScratchSize')}"))
    return " ".join(sorted(set(out))) or "-"


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_mxfp4(q: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
    """fp64 [N, k] from e2m1 pairs [N, Kp / 2] and e8m0 [N, Kp / 32] (low nibble first)."""
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float64,
                         device=q.device)
    nib = torch.stack(((q & 0xF).long(), (q >> 4).long()), dim=-1).flatten(1)
    scale = torch.exp2(s.double() - 127).repeat_interleave(32, dim=1)
    return (table[nib] * scale)[:, :k]


def main() -> int:
    """Oracle on silicon at the Qwen3.8-Flash-Next HC shape (hidden 2560, hc 4, lowrank 320,
    per-branch norm). (1) mx4_quant == AITER's dynamic_mxfp4_quant, bitwise. (2) Each GEMM
    path vs a reference on the same quantized operands (fp64), i.e. the dot_scaled / fp8
    plumbing. (3) Per site kind (combine_and_mix N 336, mix N 336, final mixer N 320) and M:
    block_input / injection error vs an fp64 chain on the bf16 weights for mxfp4, fp8, the
    bf16 kernels and vLLM's stock chain; graph replay == eager; graphed us per site (norm +
    down + up; HC_DOWN + HC_FUSE kernels up to 16, HC_BIG's above) over 24 weight copies
    (cold weights) for bf16 / mxfp4 / fp8; isa of the quantized kernels."""
    import vllm.model_executor.layers.utils  # noqa: F401 - registers vllm::rocm_unquantized_gemm
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from vllm.models.qwen4_exp.amd.ops import hc as stock

    from suffix_hybrid.kernels import hc_big_rocm, hc_down_rocm, hc_fused_rocm

    linear = torch.nn.functional.linear
    dev, bf16 = "cuda", torch.bfloat16
    d, hc, rank, copies, eps = 2560, 4, 320, 24, 1e-6
    big = hc * d
    torch.manual_seed(0)
    failed = False

    # (1) the in-kernel activation quant vs AITER's
    for m in (1, 5, 64):
        x = (torch.randn(m, big, device=dev) * torch.rand(m, 1, device=dev) * 4).to(bf16)
        q, s = dynamic_mxfp4_quant(x)
        bm = max(16, triton.next_power_of_2(m))
        xp = torch.zeros(bm, big, device=dev, dtype=bf16)
        xp[:m] = x
        q2 = torch.empty(bm, big // 2, device=dev, dtype=torch.uint8)
        s2 = torch.empty(bm, big // 32, device=dev, dtype=torch.uint8)
        _mx4_quant_probe_kernel[(1, big // 256)](xp, q2, s2, xp.stride(0), BLOCK_M=bm,
                                                  BLOCK_N=256, num_warps=4)
        ok = torch.equal(q2[:m], q) and torch.equal(s2[:m], s)
        failed |= not ok
        print(f"{MARK} act quant M={m}: mx4_quant vs AITER dynamic_mxfp4_quant "
              f"{'bitwise' if ok else 'MISMATCH'}", flush=True)

    w_up = [(torch.randn(big, rank, device=dev) * 0.05).to(bf16) for _ in range(copies)]
    w_dn = {n: [(torch.randn(n, big, device=dev) * 0.01).to(bf16) for _ in range(copies)]
            for n in (336, 320)}
    norm_w = (0.2 * torch.randn(big, device=dev)).to(bf16)
    q_up = {mode: [quantize(w, mode) for w in w_up] for mode in (1, 2)}
    q_dn = {(n, mode): [quantize(w, mode) for w in w_dn[n]] for n in (336, 320) for mode in (1, 2)}

    # (2) GEMM plumbing: the kernels vs fp64 on the same quantized operands
    for m in (5, 40, 160):
        xn = (torch.randn(m, big, device=dev)).to(bf16)
        lora = (torch.randn(m, rank, device=dev) * 2).to(bf16)
        for mode, name in ((1, "mxfp4"), (2, "fp8")):
            qd, qu = q_dn[(336, mode)][0], q_up[mode][0]
            if mode == 1:
                xq, xs = dynamic_mxfp4_quant(xn)
                xd = dequant_mxfp4(xq, xs, big)
                wd = dequant_mxfp4(qd.q, qd.s, big)
                silu = (lora.float() / hc * torch.sigmoid(lora.float() / hc)).to(bf16)
                aq, as_ = dynamic_mxfp4_quant(torch.nn.functional.pad(silu, (0, 64)))
                ad = dequant_mxfp4(aq, as_, rank)
                wu = dequant_mxfp4(qu.q, qu.s, rank)
            else:
                xd, wd = xn.double(), qd.q.double() * qd.s.double()[:, None]
                silu = (lora.float() / hc * torch.sigmoid(lora.float() / hc)).to(bf16)
                ad, wu = silu.double(), qu.q.double() * qu.s.double()[:, None]
            down = hc_down_rocm._launch(xn, w_dn[336][0], wq=qd)
            ref = xd @ wd.T
            e_down = ((down.double() - ref).abs().max() / ref.abs().max()).item()
            gate_ref = ad @ wu.T  # the up GEMM through the fused kernel's gate mix
            blk = (hc_fused_rocm._launch(lora, w_up[0], xn, hc, wq=qu) if m <= 64 else
                   hc_big_rocm.up_mix(silu, w_up[0], xn, hc, hc_big_rocm._cfg(m)[1], wq=qu))
            blk_ref = (torch.sigmoid(gate_ref.to(bf16).double()) * xn.double()).view(
                m, hc, d).sum(1) / hc
            e_up = ((blk.double() - blk_ref).abs().max() / blk_ref.abs().max()).item()
            ok = e_down < 2**-7 and e_up < 2**-6  # bf16 output rounding + fp32 order
            failed |= not ok
            print(f"{MARK} gemm {name} M={m}: down vs fp64 on the quantized operands rel "
                  f"{e_down:.1e}, up+mix rel {e_up:.1e} {'OK' if ok else 'BAD'}", flush=True)

    def err(x, ref):
        diff = (x.double() - ref).abs()
        return (diff.max().item() / ref.abs().max().item(),
                diff.mean().item() / ref.abs().mean().item())

    # (3) whole sites
    for kind, n in (("combine_and_mix", 336), ("mix", 336), ("final", 320)):
        for m in (1, 5, 8, 16, 40, 64, 160, 256):
            hidden = torch.randn(m, big, device=dev).to(bf16)
            prev = (0.5 * torch.randn(m, d, device=dev)).to(bf16)
            inj = torch.randn(m, 336, device=dev).to(bf16)[:, rank:rank + hc]

            def site(path, i):
                """(block_input, injection | None) of one path; weights copy i."""
                if kind == "mix":
                    xn = stock.grouped_gemma_rmsnorm(hidden, norm_w, eps, hc)
                else:
                    xn = stock.hc_combine_norm(hidden, prev, inj, norm_w, eps, hc)[1]
                wd, wu = w_dn[n][i], w_up[i]
                if path == "stock":
                    down = linear(xn, wd)
                    blk = stock.hc_gate_mix(xn, linear(stock.hc_silu(down[:, :rank], hc), wu), hc)
                else:
                    mode = {"bf16": 0, "mxfp4": 1, "fp8": 2}[path]
                    qd = q_dn[(n, mode)][i] if mode else None
                    qu = q_up[mode][i] if mode else None
                    if m <= 16:  # HC_DOWN + HC_FUSE
                        down = hc_down_rocm._launch(xn, wd, wq=qd)
                        blk = hc_fused_rocm._launch(down[:, :rank], wu, xn, hc, wq=qu)
                    else:  # HC_BIG
                        dc, uc = hc_big_rocm._cfg(m)
                        down = hc_big_rocm.down_silu(xn, wd, hc, rank, dc, wq=qd)
                        blk = hc_big_rocm.up_mix(down[:, :rank], wu, xn, hc, uc, wq=qu)
                return blk, (down[:, rank:rank + hc] if n == 336 else None), xn

            outs = {p: site(p, 0) for p in ("stock", "bf16", "mxfp4", "fp8")}
            x64 = outs["stock"][2].double()
            down64 = x64 @ w_dn[n][0].double().T
            s64 = down64[:, :rank] / hc * torch.sigmoid(down64[:, :rank] / hc)
            blk64 = (torch.sigmoid(s64 @ w_up[0].double().T) * x64).view(m, hc, d).sum(1) / hc
            line = []
            for p in ("mxfp4", "fp8", "bf16", "stock"):
                eb = err(outs[p][0], blk64)
                text = f"{p} blk {eb[0]:.1e}/{eb[1]:.1e}"
                if n == 336:
                    ei = err(outs[p][1], down64[:, rank:rank + hc])
                    text += f" inj {ei[0]:.1e}/{ei[1]:.1e}"
                line.append(text)
            ok = True
            us = {}
            for p in ("bf16", "mxfp4", "fp8"):
                t, graphed = hc_fused_rocm._graph_us(lambda i: site(p, i)[0], copies)
                us[p] = t
                ok &= torch.equal(graphed, outs[p][0])  # graph replay == eager
                ok &= bool(torch.isfinite(outs[p][0]).all())
            failed |= not ok
            print(f"{MARK} {kind} N={n} M={m}: {'OK' if ok else 'BAD'} (graph == eager, finite) "
                  f"| rel err vs fp64 max/mean: {' | '.join(line)} | site us bf16 {us['bf16']:.1f}"
                  f" mxfp4 {us['mxfp4']:.1f} fp8 {us['fp8']:.1f}", flush=True)
    for fn in (hc_down_rocm._hc_down_kernel, hc_fused_rocm._hc_up_gate_mix_kernel,
               hc_big_rocm._hc_up_mix_kernel):
        print(f"{MARK} isa {fn.__name__}: {_isa(fn)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
