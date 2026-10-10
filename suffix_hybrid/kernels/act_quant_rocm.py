# SPDX-License-Identifier: Apache-2.0
"""MXFP4 activation quant inside its producer kernel (SUFFIX_ROCM_ACT_QUANT_FUSE=1, ROCm).

vLLM's AiterMxfp4LinearKernel (non-ASM path) runs a dense W4A4 linear as AITER
dynamic_mxfp4_quant + gemm_afp4wfp4: one ~4 us quant launch after the producer of every
activation (92 per MTP-4 step at c1). Two producers quantize their own output here, so
the sites go from producer + quant + GEMM to fused kernel + GEMM:

* GDN output: RMSNormGated(core_attn_out, z) (per 128-wide head, norm before a sigmoid
  gate) -> out_proj, 36 per step;
* QSA output: attn_output * sigmoid(gate) -> o_proj, 16 per step.

Each site is one custom op. Its Triton kernel computes the producer's bf16 result the
way inductor compiles vLLM's native code (RMSNormGated.forward_static; x * sigmoid(gate)):
fp32 math in the same op order, libdevice rsqrt, one bf16 rounding. That bf16 result
goes through AITER's own _mxfp4_quant_op, the code dynamic_mxfp4_quant runs ("even" E8M0
scale, RNE E2M1, 32-column groups), into dynamic_mxfp4_quant's layouts; then
gemm_afp4wfp4 with the linear's weights exactly as vLLM's gemm_with_dynamic_quant calls
it. The QSA gate is read where vLLM's split leaves it (the gate halves of q_gate in qkv,
or the fused QK kernel's gate copy), so inductor needs no copy for the op's input.

    python -m suffix_hybrid.kernels.act_quant_rocm   # GPU oracle + us/call (boot gate act_quant_bench)
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MARK = "[suffix act-quant]"
GDN_CONFIG = (16, 4, True)  # BLOCK_R (head rows), num_warps, libdevice rsqrt
QSA_CONFIG = (256, False)   # BLOCK_N, sigmoid rounded to bf16 before the multiply

_mxfp4_quant_op = None  # AITER's, bound on first use: importing aiter initializes HIP
_AITER_KERNEL = None    # vLLM's AiterMxfp4LinearKernel class, bound by install_*


def _aiter():
    global _mxfp4_quant_op
    if _mxfp4_quant_op is None:
        from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op as op
        _mxfp4_quant_op = op
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    return gemm_afp4wfp4


@triton.jit(do_not_specialize=["R"])
def _gdn_norm_quant_kernel(x_ptr, z_ptr, w_ptr, q_ptr, s_ptr, y_ptr, R, eps,
                           stride_xm, stride_xh, stride_zm, stride_zh, stride_qm, stride_sm,
                           stride_sn, HV: tl.constexpr, D: tl.constexpr, BLOCK_R: tl.constexpr,
                           LIBDEVICE_RSQRT: tl.constexpr, STORE_Y: tl.constexpr):
    # Rows r = (token r // HV, head r % HV) of D values: RMSNormGated.forward_static with
    # norm_before_gate and a sigmoid gate, then AITER's MXFP4 quant of its bf16 rounding.
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r // HV
    h = r % HV
    ok = (r < R)[:, None]
    d = tl.arange(0, D)
    x = tl.load(x_ptr + m[:, None] * stride_xm + h[:, None] * stride_xh + d[None, :], mask=ok,
                other=0.0).to(tl.float32)
    z = tl.load(z_ptr + m[:, None] * stride_zm + h[:, None] * stride_zh + d[None, :], mask=ok,
                other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / D + eps
    if LIBDEVICE_RSQRT:  # inductor's torch.rsqrt
        rs = tl.extra.hip.libdevice.rsqrt(var)
    else:
        rs = tl.rsqrt(var)
    y = x * rs[:, None] * tl.load(w_ptr + d).to(tl.float32)[None, :] * tl.sigmoid(z)
    y = y.to(tl.bfloat16)
    if STORE_Y:  # oracle only: the bf16 producer result
        tl.store(y_ptr + r[:, None] * D + d[None, :], y, mask=ok)
    q, s = _mxfp4_quant_op(y.to(tl.float32), D, BLOCK_R, 32)
    tl.store(q_ptr + m[:, None] * stride_qm + h[:, None] * (D // 2)
             + tl.arange(0, D // 2)[None, :], q, mask=ok)
    tl.store(s_ptr + m[:, None] * stride_sm
             + (h[:, None] * (D // 32) + tl.arange(0, D // 32)[None, :]) * stride_sn, s, mask=ok)


@triton.jit(do_not_specialize=["M"])
def _gate_quant_kernel(x_ptr, g_ptr, q_ptr, s_ptr, y_ptr, M, stride_xm, stride_gm, stride_qm,
                       stride_sm, stride_sn, G_HEAD: tl.constexpr, HD: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ROUND_SIG: tl.constexpr,
                       STORE_Y: tl.constexpr):
    # x * sigmoid(gate), gate column c of head c // HD at g_ptr + (c // HD) * G_HEAD + c % HD,
    # then AITER's MXFP4 quant of its bf16 rounding.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = (rows < M)[:, None]
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=ok, other=0.0)
    g = tl.load(g_ptr + rows[:, None] * stride_gm + (cols // HD * G_HEAD + cols % HD)[None, :],
                mask=ok, other=0.0)
    sg = tl.sigmoid(g.to(tl.float32))
    if ROUND_SIG:  # eager torch: sigmoid's own bf16 output
        sg = sg.to(tl.bfloat16).to(tl.float32)
    y = (x.to(tl.float32) * sg).to(tl.bfloat16)
    if STORE_Y:
        tl.store(y_ptr + rows[:, None] * (BLOCK_N * tl.num_programs(1)) + cols[None, :], y,
                 mask=ok)
    q, s = _mxfp4_quant_op(y.to(tl.float32), BLOCK_N, BLOCK_M, 32)
    tl.store(q_ptr + rows[:, None] * stride_qm
             + (tl.program_id(1) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :], q,
             mask=ok)
    tl.store(s_ptr + rows[:, None] * stride_sm
             + (tl.program_id(1) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32))[None, :]
             * stride_sn, s, mask=ok)


def _bufs(m: int, k: int, dev):
    """dynamic_mxfp4_quant's outputs: x_fp4 [M, K/2] row-major, E8M0 [M, K/32] column-major."""
    return (torch.empty((m, k // 2), dtype=torch.uint8, device=dev),
            torch.empty((k // 32, m), dtype=torch.uint8, device=dev).T)


def gdn_quant(x, z, norm_w, eps, config=None, y=None):
    """MXFP4 (x_fp4, scales) of RMSNormGated(x, z) for x / z [M, HV, D] (z row-strided)."""
    _aiter()
    m, hv, d = x.shape
    br, warps, libdevice_rsqrt = config or GDN_CONFIG
    assert x.stride(2) == z.stride(2) == 1 and z.shape == x.shape and d % 32 == 0
    assert d & (d - 1) == 0 and norm_w.shape == (d,)
    q, s = _bufs(m, hv * d, x.device)
    if m:
        _gdn_norm_quant_kernel[(triton.cdiv(m * hv, br),)](
            x, z, norm_w, q, s, q if y is None else y, m * hv, eps, x.stride(0), x.stride(1),
            z.stride(0), z.stride(1), q.stride(0), s.stride(0), s.stride(1), HV=hv, D=d,
            BLOCK_R=br, LIBDEVICE_RSQRT=libdevice_rsqrt, STORE_Y=y is not None,
            num_warps=warps)
    return q, s


def gate_quant(x, g, g_head, hd, config=None, y=None):
    """MXFP4 (x_fp4, scales) of x * sigmoid(gate) for x [M, K] and a gate whose head h
    (HD columns) starts at g[:, h * g_head]."""
    _aiter()
    m, k = x.shape
    bn, round_sig = config or QSA_CONFIG
    assert x.stride(1) == g.stride(1) == 1 and k % bn == 0 and bn % 32 == 0
    q, s = _bufs(m, k, x.device)
    if m:
        bm = min(8, triton.next_power_of_2(m))
        _gate_quant_kernel[(triton.cdiv(m, bm), k // bn)](
            x, g, q, s, q if y is None else y, m, x.stride(0), g.stride(0), q.stride(0),
            s.stride(0), s.stride(1), G_HEAD=g_head, HD=hd, BLOCK_M=bm, BLOCK_N=bn,
            ROUND_SIG=round_sig, STORE_Y=y is not None, num_warps=1 if bm * bn <= 512 else 4)
    return q, s


def _gemm(q, s, weight, weight_scale):
    # gemm_with_dynamic_quant's non-ASM call: weight [N, K/2], weight_scale [K/32, N] as
    # vLLM keeps it after process_weights_after_loading (passed .T).
    y = torch.empty((q.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=q.device)
    if q.shape[0]:
        _aiter()(q, weight, s, weight_scale.T, torch.bfloat16, y)
    return y


def _gdn_out(x: torch.Tensor, z: torch.Tensor, norm_w: torch.Tensor, eps: float,
             weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return _gemm(*gdn_quant(x, z, norm_w, eps), weight, weight_scale)


def _qsa_out(x: torch.Tensor, g: torch.Tensor, g_head: int, hd: int, weight: torch.Tensor,
             weight_scale: torch.Tensor) -> torch.Tensor:
    return _gemm(*gate_quant(x, g, g_head, hd), weight, weight_scale)


def _gdn_out_fake(x: torch.Tensor, z: torch.Tensor, norm_w: torch.Tensor, eps: float,
                  weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)


def _qsa_out_fake(x: torch.Tensor, g: torch.Tensor, g_head: int, hd: int, weight: torch.Tensor,
                  weight_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(op_name="suffix_aq_gdn_out", op_func=_gdn_out, fake_impl=_gdn_out_fake)
direct_register_custom_op(op_name="suffix_aq_qsa_out", op_func=_qsa_out, fake_impl=_qsa_out_fake)


def _linear_ok(lin) -> bool:
    """A RowParallelLinear served by vLLM's non-ASM AITER MXFP4 kernel at TP 1, no bias."""
    k = getattr(getattr(lin, "scheme", None), "ocp_mx_linear", None)
    return (isinstance(k, _AITER_KERNEL) and not k.use_asm_gemm
            and k.out_dtype == torch.bfloat16 and lin.tp_size == 1
            and getattr(lin, "bias", None) is None and lin.weight.dtype == torch.uint8)


def gdn_ok(layer) -> bool:
    n, d = layer.norm, layer.head_v_dim
    return (n.activation == "sigmoid" and n.norm_before_gate and n.group_size is None
            and getattr(n, "bias", None) is None and d % 32 == 0 and d & (d - 1) == 0
            and _linear_ok(layer.out_proj))


def gdn_out(layer, core_attn_out, z):
    """_output_projection: self.out_proj(self.norm(core_attn_out, z).flatten(-2))."""
    return torch.ops.vllm.suffix_aq_gdn_out(core_attn_out, z, layer.norm.weight, layer.norm.eps,
                                            layer.out_proj.weight, layer.out_proj.weight_scale)


def qsa_ok(layer) -> bool:
    return layer.attn_output_gate and layer.head_dim % 32 == 0 and _linear_ok(layer.o_proj)


def qsa_out(layer, flat_output, gate, qkv):
    """self.o_proj(flat_output * torch.sigmoid(gate)), the gate read where the split left
    it: the fused QK kernel's own [T, H * hd] copy, else the gate halves of q_gate in qkv."""
    hd = layer.head_dim
    if layer.use_fused_qk_norm_rope_gate:
        g, g_head = gate, hd
    else:
        g, g_head = qkv[:, hd:], 2 * hd
    return torch.ops.vllm.suffix_aq_qsa_out(flat_output, g, g_head, hd, layer.o_proj.weight,
                                            layer.o_proj.weight_scale)


def _bind_kernel_class():
    global _AITER_KERNEL
    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    _AITER_KERNEL = AiterMxfp4LinearKernel


def install_gdn(module) -> None:
    """rocm_patches `after` hook for qwen_gdn_linear_attn (_output_projection)."""
    _bind_kernel_class()
    module._suffix_aq_gdn_ok, module._suffix_aq_gdn_out = gdn_ok, gdn_out


def install_qsa(module) -> None:
    """rocm_patches `after` hook for vllm.models.qwen4_exp.amd.qsa (forward's tail)."""
    _bind_kernel_class()
    module._suffix_aq_qsa_ok, module._suffix_aq_qsa_out = qsa_ok, qsa_out


def main() -> int:
    """Oracle on silicon at Qwen3.8-Flash-Next shapes (GDN 48 x 128 -> out_proj 2560, QSA
    24 x 256 -> o_proj 2560), M 1..1024. Stock = what serving runs: vLLM's native producer
    compiled by inductor (dynamic shapes), dynamic_mxfp4_quant, gemm_afp4wfp4. Per M and
    site: the fused kernel's bf16 producer result and its x_fp4 / E8M0 bytes vs stock's,
    the op's output vs stock bitwise, a graph replay with new inputs, graphed us/call
    stock vs fused (24 weight copies, cold), and the alternative numerics knob."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from vllm.model_executor.layers.layernorm import RMSNormGated

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    dev, bf16 = "cuda", torch.bfloat16
    HV, D, NH, HD, N, eps, copies = 48, 128, 24, 256, 2560, 1e-6, 24
    K = HV * D
    torch.manual_seed(0)
    weights = []
    for _ in range(copies):  # vLLM's non-ASM layout: weight [N, K/2], weight_scale [K/32, N]
        wq, ws = dynamic_mxfp4_quant((torch.randn(N, K, device=dev) * 0.02).to(bf16))
        weights.append((wq, ws.T.contiguous()))
    norm_w = (1 + 0.1 * torch.randn(D, device=dev)).to(bf16)
    norm_c = torch.compile(lambda x, z: RMSNormGated.forward_static(
        x, z, norm_w, eps, bf16, None, True, "sigmoid"), dynamic=True)
    qkz = HV * D * 2 + 2 * 16 * D + K  # a qkvz-like row: z is its last HV * D columns

    def gate_of(qkv, m):  # Qwen3NextAttention._project_qkv_gate's eager split
        _, gate = torch.chunk(qkv[:, :NH * HD * 2].view(m, NH, -1), 2, dim=-1)
        return gate.reshape(m, -1)

    gate_c = torch.compile(lambda a, qkv: a * torch.sigmoid(gate_of(qkv, a.shape[0])),
                           dynamic=True)
    gate_cc = torch.compile(lambda a, g: a * torch.sigmoid(g), dynamic=True)

    def share(a, b):
        return (a == b).float().mean().item() * 100

    failed = False
    for m in (1, 2, 5, 8, 16, 40, 64, 160, 256, 1024):
        # GDN: core_attn_out [m, HV, D], z a strided view into a qkvz-like buffer
        x = (0.3 * torch.randn(m, HV, D, device=dev)).to(bf16)
        zbuf = torch.randn(m, qkz, device=dev).to(bf16)
        z = zbuf[:, qkz - K:].view(m, HV, D)
        rows = {}
        for name, fn in (("gdn", lambda: norm_c(x, z).view(m, K)),
                         ("gdn-eager", lambda: RMSNormGated.forward_static(
                             x, z, norm_w, eps, bf16, None, True, "sigmoid").view(m, K))):
            rows[name] = fn()
        y_new = torch.empty(m * HV, D, device=dev, dtype=bf16)
        q_new, s_new = gdn_quant(x, z, norm_w, eps, y=y_new)
        q_ref, s_ref = dynamic_mxfp4_quant(rows["gdn"])
        q_own, s_own = dynamic_mxfp4_quant(y_new.view(m, K))
        alt = gdn_quant(x, z, norm_w, eps, (GDN_CONFIG[0], GDN_CONFIG[1], not GDN_CONFIG[2]))
        out_new = _gdn_out(x, z, norm_w, eps, *weights[0])
        out_ref = _gemm(q_ref, s_ref, *weights[0])
        exact = torch.equal(q_new, q_ref) and torch.equal(s_new, s_ref) and torch.equal(
            out_new, out_ref)
        own = torch.equal(q_new, q_own) and torch.equal(s_new, s_own)
        # graph replay with new inputs
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = _gdn_out(x, z, norm_w, eps, *weights[0])
        x.copy_((0.3 * torch.randn(m, HV, D, device=dev)).to(bf16))
        zbuf.copy_(torch.randn(m, qkz, device=dev).to(bf16))
        g.replay()
        replay = torch.equal(out_g, _gdn_out(x, z, norm_w, eps, *weights[0]))
        t_ref = _graph_us(lambda i: _gemm(*dynamic_mxfp4_quant(norm_c(x, z).view(m, K)),
                                          *weights[i % copies]), copies)[0]
        t_new = _graph_us(lambda i: _gdn_out(x, z, norm_w, eps, *weights[i % copies]),
                          copies)[0]
        failed |= not (exact and own and replay)
        print(f"{MARK} gdn M={m}: bf16 norm vs inductor {share(y_new.view(m, K), rows['gdn']):.3f}% "
              f"(vs eager {share(y_new.view(m, K), rows['gdn-eager']):.3f}%); codes vs "
              f"dynamic_mxfp4_quant(inductor norm) q {share(q_new, q_ref):.3f}% s "
              f"{share(s_new, s_ref):.3f}%, vs quant of own bf16 {'bitwise' if own else 'DIFFER'}; "
              f"alt rsqrt q {share(alt[0], q_ref):.3f}%; out_proj vs stock "
              f"{'bitwise' if torch.equal(out_new, out_ref) else 'max abs %.3g' % (out_new.float() - out_ref.float()).abs().max().item()}; "
              f"graph replay {'ok' if replay else 'DIFFERS'} | graphed us stock {t_ref:.1f} "
              f"fused {t_new:.1f} -> {'MATCH' if exact and own and replay else 'MISMATCH'}",
              flush=True)

        # QSA: attn_output [m, NH * HD]; the gate from qkv (eager split) or contiguous (fused QK)
        a = (0.5 * torch.randn(m, NH * HD, device=dev)).to(bf16)
        qkv = torch.randn(m, NH * HD * 2 + 2 * 2 * HD, device=dev).to(bf16)
        gc = torch.randn(m, NH * HD, device=dev).to(bf16)
        for label, ref_fn, gsrc, g_head in (
                ("gate in qkv", lambda: gate_c(a, qkv), lambda: qkv[:, HD:], 2 * HD),
                ("gate contiguous", lambda: gate_cc(a, gc), lambda: gc, HD)):
            ref = ref_fn()
            y_new = torch.empty(m, NH * HD, device=dev, dtype=bf16)
            q_new, s_new = gate_quant(a, gsrc(), g_head, HD, y=y_new)
            q_ref, s_ref = dynamic_mxfp4_quant(ref)
            q_own, s_own = dynamic_mxfp4_quant(y_new)
            alt = gate_quant(a, gsrc(), g_head, HD, (QSA_CONFIG[0], not QSA_CONFIG[1]))
            out_new = _qsa_out(a, gsrc(), g_head, HD, *weights[0])
            out_ref = _gemm(q_ref, s_ref, *weights[0])
            exact = torch.equal(q_new, q_ref) and torch.equal(s_new, s_ref) and torch.equal(
                out_new, out_ref)
            own = torch.equal(q_new, q_own) and torch.equal(s_new, s_own)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out_g = _qsa_out(a, gsrc(), g_head, HD, *weights[0])
            a.copy_((0.5 * torch.randn(m, NH * HD, device=dev)).to(bf16))
            qkv.copy_(torch.randn(m, qkv.shape[1], device=dev).to(bf16))
            gc.copy_(torch.randn(m, NH * HD, device=dev).to(bf16))
            g.replay()
            replay = torch.equal(out_g, _qsa_out(a, gsrc(), g_head, HD, *weights[0]))
            t_ref = _graph_us(lambda i: _gemm(*dynamic_mxfp4_quant(ref_fn()),
                                              *weights[i % copies]), copies)[0]
            t_new = _graph_us(lambda i: _qsa_out(a, gsrc(), g_head, HD, *weights[i % copies]),
                              copies)[0]
            failed |= not (exact and own and replay)
            print(f"{MARK} qsa M={m} ({label}): bf16 gate product vs inductor "
                  f"{share(y_new, ref):.3f}%; codes vs dynamic_mxfp4_quant(inductor) q "
                  f"{share(q_new, q_ref):.3f}% s {share(s_new, s_ref):.3f}%, vs quant of own bf16 "
                  f"{'bitwise' if own else 'DIFFER'}; alt sigmoid rounding q "
                  f"{share(alt[0], q_ref):.3f}%; o_proj vs stock "
                  f"{'bitwise' if torch.equal(out_new, out_ref) else 'max abs %.3g' % (out_new.float() - out_ref.float()).abs().max().item()}; "
                  f"graph replay {'ok' if replay else 'DIFFERS'} | graphed us stock {t_ref:.1f} "
                  f"fused {t_new:.1f} -> {'MATCH' if exact and own and replay else 'MISMATCH'}",
                  flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
