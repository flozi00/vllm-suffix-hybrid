# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A4 decode-GEMM capability spike (qwen3.8-27b) — reference, CPU
twin, on-silicon oracle + bench.

Kernel: kernels-oxide/nvfp4_gemm (cuda-oxide -> PTX 8.7 .target sm_120a ->
ptxas 13.0 -> sm_120a SASS), block-scaled FP4 tensor-core mma
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
.f32.e2m1.e2m1.f32.ue4m3`. Host op: `_native.nvfp4_gemm_cuda`.

vLLM conventions reproduced here (dossier qwen38-27b-kernels.md §7):
  quant  (vllm scaled_fp4_quant, nvfp4_utils.cuh:262-287):
         sf = e4m3(amax16 * g / 6);  q = e2m1_rne(x * g / sf)
  weight uint8 [N, K/2], element 2i in the LOW nibble; scales e4m3 [N, K/16]
         swizzled 128x4 (nvfp4_utils.py:13-53, `swizzle_sf` below)
  gemm   y = alpha * sum(q_a sf_a q_b sf_b),  alpha = 1 / (g_x * g_w)

CLI (in-pod, SM120 + oxide bundle):
  python -m suffix_hybrid.kernels.nvfp4_gemm oracle   # numerics vs torch ref + vLLM op
  python -m suffix_hybrid.kernels.nvfp4_gemm bench    # us: ours vs vLLM (quant+gemm, CUDA graphs)
Boot: SUFFIX_NVFP4_GEMM_SPIKE=oracle|bench|both (sitecustomize, child process).
"""
from __future__ import annotations

import sys

import numpy as np

MARKER = "[suffix nvfp4-gemm]"
FAMILY = "nvfp4_gemm"
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
# qwen3.8-27b decode projections (N, K): the spike target is the largest.
SHAPES = {
    "mlp_gate_up": (34816, 5120),
    "gdn_in_proj_qkvz": (16384, 5120),
    "mlp_down": (5120, 17408),
    "attn_qkv": (14336, 5120),
    "gdn_out_proj": (5120, 6144),
    "attn_o": (5120, 6144),
}
SPIKE_SHAPE = "mlp_gate_up"


# ---------------------------------------------------------------------------
# reference quantization (numpy, fp32) — identical rounding to the kernel
# ---------------------------------------------------------------------------
def e2m1_code(v):
    """|v| -> e2m1 magnitude code 0..7, RNE (ties to even code), saturating."""
    a = np.minimum(np.abs(np.asarray(v, dtype=np.float32)), np.float32(6.0))
    gt = (a[..., None] > _MID).sum(-1)
    tie_up = ((a[..., None] == _MID) & (np.arange(7) % 2 == 1)).sum(-1)
    return (gt + tie_up).astype(np.uint8)


def e4m3(v):
    """f32 -> e4m3fn (RNE, satfinite) as (bits uint8, value f32), via torch."""
    import torch
    t = torch.from_numpy(np.minimum(np.asarray(v, dtype=np.float32), 448.0))
    f8 = t.to(torch.float8_e4m3fn)
    return f8.view(torch.uint8).numpy(), f8.float().numpy()


def quantize(x, g):
    """x [R, K] f32 -> (packed uint8 [R, K/2] low-nibble-even, sf bits
    uint8 [R, K/16], sf values f32 [R, K/16]) — vLLM scaled_fp4_quant math."""
    x = np.asarray(x, dtype=np.float32)
    r, k = x.shape
    blk = x.reshape(r, k // 16, 16)
    amax = np.abs(blk).max(-1)
    sf_bits, sf = e4m3(amax * (np.float32(g) / np.float32(6.0)))
    inv = np.where(sf == 0, np.float32(0), np.float32(g) / np.where(sf == 0, 1, sf))
    scaled = (blk * inv[..., None]).astype(np.float32)
    code = e2m1_code(scaled) | np.where(scaled < 0, 8, 0).astype(np.uint8)
    code = code.reshape(r, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, sf_bits, sf


def dequant(packed, sf):
    """(packed [R, K/2], sf values [R, K/16]) -> f32 [R, K] (without 1/g)."""
    lo = packed & 0xF
    hi = packed >> 4
    code = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    val = E2M1[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    return (val.reshape(val.shape[0], -1, 16) * sf[..., None]).reshape(val.shape[0], -1)


def swizzle_sf(sf_bits):
    """vLLM swizzle_blockscale on uint8 [R, C] -> flat uint8 (padded)."""
    r, c = sf_bits.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    pad = np.zeros((rp, cp), np.uint8)
    pad[:r, :c] = sf_bits
    return pad.reshape(rp // 128, 4, 32, cp // 4, 4).transpose(0, 3, 2, 1, 4).reshape(-1)


def sf_offset(row, kb, kb_pad):
    """Kernel's `sf_offset` (must equal swizzle_sf's placement)."""
    atom = (row // 128) * (kb_pad // 4) + kb // 4
    return atom * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


def make_problem(m, n, k, seed=0):
    """Random bf16-representable x [m,k], quantized weight in vLLM layout."""
    rng = np.random.default_rng(seed)
    import torch
    x = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float32)).bfloat16()
    w = torch.from_numpy((rng.standard_normal((n, k)) * 0.05).astype(np.float32)).bfloat16()
    xf, wf = x.float().numpy(), w.float().numpy()
    g_x = np.float32(448.0 * 6.0 / np.abs(xf).max())
    g_w = np.float32(448.0 * 6.0 / np.abs(wf).max())
    w_packed, w_sf_bits, w_sf = quantize(wf, g_w)
    return dict(x=x, w_bf16=w, g_x=g_x, g_w=g_w, alpha=np.float32(1.0 / (g_x * g_w)),
                w_packed=w_packed, w_sf_bits=w_sf_bits, w_sf=w_sf,
                w_sf_swz=swizzle_sf(w_sf_bits))


def gemm_ref(p):
    """Exact fp32 result of the quantized problem (what the mma computes)."""
    xq, _, xsf = quantize(p["x"].float().numpy(), p["g_x"])
    a = dequant(xq, xsf)
    b = dequant(p["w_packed"], p["w_sf"])
    return (a @ b.T * p["alpha"]).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU twin of nvfp4_gemm_m16's addressing + the mma fragment layouts
# ---------------------------------------------------------------------------
def kernel_twin(p):
    """Replays the kernel's per-lane u32 loads from flat buffers and the
    mxf4nvf4 m16n8k64 fragment/scale layouts (cute MMA_Traits
    SM120_16x8x64_TN_VS), returning the fp32 [M, N] result. Any addressing
    bug in the kernel (fragment rows/k, scale bytes, swizzle) shows up here."""
    xf = p["x"].float().numpy()
    m, k = xf.shape
    n = p["w_packed"].shape[0]
    aq = np.zeros((16, k // 2), np.uint8)
    asf_bits = np.zeros((16, k // 16), np.uint8)
    xq, xsf_bits, _ = quantize(xf, p["g_x"])
    aq[:m], asf_bits[:m] = xq, xsf_bits
    aq, asf = aq.reshape(-1), asf_bits.reshape(-1)
    w, wsf = p["w_packed"].reshape(-1), p["w_sf_swz"]
    kh, nkb = k // 2, k // 16
    kb_pad = -(-nkb // 4) * 4
    f8 = lambda b: e4m3_bits_to_f32(np.asarray(b, np.uint8))
    out = np.zeros((m, n), np.float32)

    def u32(buf, off):
        return buf[off:off + 4]

    def nib(bytes4, j):
        byte = bytes4[j // 2]
        c = (byte >> 4) if j % 2 else (byte & 0xF)
        return E2M1[c & 7] * (-1.0 if c & 8 else 1.0)

    for n0 in range(0, n, 8):
        acc = np.zeros((16, 8), np.float64)
        for k0 in range(0, k, 64):
            A = np.full((16, 64), np.nan)
            B = np.full((8, 64), np.nan)
            SA = np.full((16, 4), np.nan)
            SB = np.full((8, 4), np.nan)
            for lane in range(32):
                g, t = lane // 4, lane % 4
                off = k0 // 2 + 4 * t
                regs_a = [u32(aq, g * kh + off), u32(aq, (g + 8) * kh + off),
                          u32(aq, g * kh + off + 16), u32(aq, (g + 8) * kh + off + 16)]
                for r, reg in enumerate(regs_a):
                    for j in range(8):
                        mm, kk = g + 8 * (r & 1), 8 * t + j + 32 * (r >> 1)
                        _put(A, mm, kk, nib(reg, j))
                regs_b = [u32(w, (n0 + g) * kh + off), u32(w, (n0 + g) * kh + off + 16)]
                for r, reg in enumerate(regs_b):
                    for j in range(8):
                        _put(B, g, 8 * t + j + 32 * r, nib(reg, j))
                sfa_row = 8 * (lane & 1) + lane // 4
                sfa = u32(asf, sfa_row * nkb + k0 // 16)
                sfb = u32(wsf, sf_offset(n0 + g, k0 // 16, kb_pad))
                for b in range(4):
                    _put(SA, sfa_row, b, f8(sfa[b]))
                    _put(SB, lane // 4, b, f8(sfb[b]))
            assert not np.isnan(A).any() and not np.isnan(B).any()
            assert not np.isnan(SA).any() and not np.isnan(SB).any()
            As = A * np.repeat(SA, 16, axis=1)
            Bs = B * np.repeat(SB, 16, axis=1)
            acc += As @ Bs.T
        out[:, n0:n0 + 8] = (acc[:m] * p["alpha"]).astype(np.float32)
    return out


def _put(arr, i, j, v):
    if not np.isnan(arr[i, j]) and arr[i, j] != v:
        raise AssertionError(f"fragment conflict at {(i, j)}: {arr[i, j]} vs {v}")
    arr[i, j] = v


def e4m3_bits_to_f32(b):
    import torch
    return torch.from_numpy(np.atleast_1d(b).astype(np.uint8)).view(
        torch.float8_e4m3fn).float().numpy().reshape(np.shape(b))


# ---------------------------------------------------------------------------
# on-silicon oracle + bench
# ---------------------------------------------------------------------------
def _dev_problem(m, n, k, dev, seed=0):
    import torch
    p = make_problem(m, n, k, seed)
    t = dict(
        x=p["x"].to(dev),
        w=torch.from_numpy(p["w_packed"]).to(dev),
        w_sf=torch.from_numpy(p["w_sf_swz"]).to(dev).view(torch.float8_e4m3fn),
        aq=torch.empty(16, k // 2, dtype=torch.uint8, device=dev),
        asf=torch.empty(16, k // 16, dtype=torch.uint8, device=dev),
        out=torch.empty(m, n, dtype=torch.bfloat16, device=dev),
        g=torch.tensor([p["g_x"]], dtype=torch.float32, device=dev),
        alpha=torch.tensor([p["alpha"]], dtype=torch.float32, device=dev),
    )
    t["w_sf_2d"] = t["w_sf"].view(-(-n // 128) * 128, -1)
    return p, t


def _ours(native, t, stream):
    native.nvfp4_gemm_cuda(t["x"], t["w"], t["w_sf"], t["aq"], t["asf"], t["out"],
                           t["g_f"], t["alpha_f"], stream)
    return t["out"]


def _vllm(t, n):
    """vLLM's own NVFP4 linear ops on the same tensors: scaled_fp4_quant +
    FlashInfer CUTLASS SM120 mm (the default backend), falling back to
    vLLM's cutlass_scaled_fp4_mm."""
    import torch
    from vllm import _custom_ops as ops
    xq, xsf = ops.scaled_fp4_quant(t["x"], t["g"])
    try:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
        y = flashinfer_scaled_fp4_mm(xq, t["w"], xsf, t["w_sf_2d"], t["alpha"],
                                     torch.bfloat16, backend="cutlass")
    except Exception:  # FlashInfer absent: vLLM's own CUTLASS kernel
        y = ops.cutlass_scaled_fp4_mm(xq, t["w"], xsf, t["w_sf_2d"], t["alpha"],
                                      torch.bfloat16)
    return y[:, :n]


def _native_ready():
    import torch
    from suffix_hybrid import oxide_kernels
    native = oxide_kernels.native()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("NVFP4 spike cubin is sm_120a SASS (cc 12.x only)")
    oxide_kernels.ensure_loaded(FAMILY)  # sha256 + cuModuleLoadData
    return native


def oracle(ms=(1, 4, 16), shapes=("mlp_gate_up", "mlp_down", "gdn_in_proj_qkvz")):
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(dev).cuda_stream
    lines = []
    for name in shapes:
        n, k = SHAPES[name]
        for m in ms:
            p, t = _dev_problem(m, n, k, dev, seed=m)
            t["g_f"], t["alpha_f"] = float(p["g_x"]), float(p["alpha"])
            ours = _ours(native, t, stream).float()
            ref = torch.from_numpy(gemm_ref(p)).to(dev)
            ref16 = ref.bfloat16().float()
            rel = float((ours - ref16).norm() / ref16.norm())
            cos_v = float(torch.nn.functional.cosine_similarity(
                ours.flatten(), ref.flatten(), dim=0))
            vl = _vllm(t, n).float()
            rel_v = float((ours - vl).norm() / vl.norm())
            ok = rel <= 1e-2 and cos_v >= 0.9999 and rel_v <= 2e-2
            lines.append(f"{name} M={m} N={n} K={k}: rel_vs_ref={rel:.2e} cos={cos_v:.6f} "
                         f"rel_vs_vllm={rel_v:.2e} {'OK' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"{MARKER} NVFP4-GEMM ORACLE FAIL: {lines[-1]}")
    for ln in lines:
        print(f"{MARKER} {ln}", file=sys.stderr, flush=True)
    return f"{MARKER} NVFP4-GEMM ORACLE PASS ({len(lines)} cases, sm_120a mxf4nvf4 mma)"


def bench(ms=(1, 4, 8, 16), shapes=tuple(SHAPES), iters=200):
    """us per call (quant + gemm), each path captured in one CUDA graph."""
    import torch
    native = _native_ready()
    dev = torch.device("cuda", torch.cuda.current_device())
    res = {}
    for name in shapes:
        n, k = SHAPES[name]
        for m in ms:
            p, t = _dev_problem(m, n, k, dev, seed=1)
            t["g_f"], t["alpha_f"] = float(p["g_x"]), float(p["alpha"])
            times = []
            for which in ("vllm", "ours"):
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    fn = (lambda: _vllm(t, n)) if which == "vllm" else \
                        (lambda: _ours(native, t, s.cuda_stream))
                    fn()  # warm (FlashInfer tactic selection happens here)
                    torch.cuda.synchronize(dev)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s):
                        fn()
                torch.cuda.current_stream(dev).wait_stream(s)
                g.replay()
                torch.cuda.synchronize(dev)
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(iters):
                    g.replay()
                e1.record()
                torch.cuda.synchronize(dev)
                times.append(e0.elapsed_time(e1) * 1000.0 / iters)
            bytes_w = n * k // 2 + n * k // 16
            roof = bytes_w / 1.79e12 * 1e6
            res[(name, m)] = tuple(times)
            print(f"{MARKER} bench {name} M={m} N={n} K={k}: vllm {times[0]:.1f} us, "
                  f"ours {times[1]:.1f} us (x{times[1] / times[0]:.2f}), "
                  f"weight-roofline {roof:.1f} us", file=sys.stderr, flush=True)
    return res


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "both"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("bench", "both"):
            bench()
        return 0
    except Exception as exc:
        print(f"{MARKER} NVFP4-GEMM {mode.upper()} FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
