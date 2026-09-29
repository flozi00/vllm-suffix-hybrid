// SPDX-License-Identifier: Apache-2.0
//! NVFP4 W4A4 decode GEMM (M <= 64: decode rows x MTP verify tokens).
//!
//! Proves the cuda-oxide -> PTX 8.7 (.target sm_120a) -> ptxas 13.0 chain can
//! drive SM120's block-scaled FP4 tensor cores:
//!
//!   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//!            .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//!
//! (cuda-device ec4aa479 only wraps kind::mxf8f6f4 / ue8m0 — MX, block 32 —
//! so the NVFP4 form (block 16, e4m3 scales) is emitted through `ptx_asm!`,
//! the exact operand shape CUTLASS uses: cute/arch/mma_sm120.hpp
//! SM120_16x8x64_TN_VS<e2m1, e2m1, f32, ue4m3, 16>.) The instruction is
//! arch-specific: PTX must say `.target sm_120a` (ptxas rejects it for plain
//! sm_120), the cubin is sm_120a SASS, which the driver loads on cc 12.0.
//!
//! y[M, N] = alpha * sum_k deq(xq[m, k]) * deq(w[n, k])      (bf16 out)
//! with vLLM's NVFP4 conventions (dossier qwen38-27b-kernels.md §7):
//!   w     [N, K/2]  u8, two e2m1 per byte, LOW nibble = even k
//!   w_sf  e4m3 [N, K/16], CUTLASS 128x4-swizzled (see `sf_offset`)
//!   x     bf16 [M, K]  -> quantized here by `nvfp4_quant_act`
//!   alpha = 1 / (x_global_scale * w_global_scale)
//!
//! Entries (interface.json):
//!   nvfp4_quant_act  grid (K/16/128 rounded up, 16*MT), block 128: one
//!                    thread per (row, 16-block): amax -> e4m3 scale -> e2m1
//!                    codes; rows >= M are written as zeros (the mma always
//!                    runs m16 tiles). Output: aq [16*MT, K/2] u8, asf
//!                    [16*MT, K/16] u8 (row major, NOT swizzled — our layout).
//!   nvfp4_gemm_t{MT} MT = ceil(M/16) in 1..4; grid (ceil(N/32), splits),
//!                    block 128 (4 warps x n8 tiles): each warp owns 8 output
//!                    columns and streams its K range in k64 steps; the
//!                    weight fragment + scales of a step are loaded ONCE and
//!                    feed MT mmas (one per m16 activation tile), so weight
//!                    bytes stay at the roofline independent of M.
//!   nvfp4_splitk_reduce  fixed-order sum over the split partials (fp32
//!                    workspace [splits, M, N]) -> bf16: deterministic.
//! Fragment maps (cute MMA_Traits<SM120_16x8x64_TN_VS>, g = lane/4, t = lane%4;
//! A/SFA rows of m16 tile i are offset by 16*i):
//!   A reg r: row g + 8*(r&1), k = 8t..8t+7 (+32 for r>=2)   -> u32 of aq row
//!   B reg r: col g,           k = 8t..8t+7 (+32 for r=1)    -> u32 of w row
//!   SFA:     row 8*(lane&1) + lane/4, 4 bytes = k-blocks 0..3 of the k64 step
//!   SFB:     col lane/4, 4 bytes = k-blocks 0..3
//!   C:       c0,c1 row g cols 2t,2t+1; c2,c3 row g+8.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::{kernel, launch_bounds, ptx_asm, thread};

    const WARPS: u32 = 4;

    /// `for $i in 0..$mt` fully unrolled (MT <= 4): every tile index is a
    /// constant, so the per-tile arrays stay in registers (a rolled loop
    /// over the accumulators spilled them to .local for MT >= 3).
    macro_rules! unroll_mt {
        ($mt:expr, $i:ident, $body:block) => {{
            { const $i: usize = 0; if $i < $mt $body }
            { const $i: usize = 1; if $i < $mt $body }
            { const $i: usize = 2; if $i < $mt $body }
            { const $i: usize = 3; if $i < $mt $body }
        }};
    }

    #[inline(always)]
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    #[inline(always)]
    fn f32_to_bf16(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    /// f32 -> e4m3fn byte (cvt.rn.satfinite; both halves get the same value,
    /// so the x2 packing order does not matter).
    #[inline(always)]
    fn to_e4m3(x: f32) -> u32 {
        let r: u32;
        unsafe {
            ptx_asm!(
                "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %1; cvt.u32.u16 %0, t; }",
                out("=r") r,
                in("f") x,
                options(register_only),
            );
        }
        r & 0xff
    }

    /// e4m3fn byte -> f32 (exact).
    #[inline(always)]
    fn e4m3_to_f32(b: u32) -> f32 {
        let r: f32;
        let h = (b | (b << 8)) as u16;
        unsafe {
            ptx_asm!(
                "{ .reg .b32 t; .reg .b16 lo, hi; cvt.rn.f16x2.e4m3x2 t, %1; mov.b32 {lo, hi}, t; cvt.f32.f16 %0, lo; }",
                out("=f") r,
                in("h") h,
                options(register_only),
            );
        }
        r
    }

    /// f32 -> e2m1 code (RNE, saturating at 6), sign in bit 3.
    #[inline(always)]
    fn to_e2m1(x: f32) -> u32 {
        let r: u32;
        unsafe {
            ptx_asm!(
                "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %1; cvt.u32.u8 %0, t; }",
                out("=r") r,
                in("f") x,
                options(register_only),
            );
        }
        r & 0xf
    }

    /// Byte offset of scale (row, k-block) in CUTLASS's 128x4 swizzled SF
    /// layout (vLLM swizzle_blockscale): 128-row x 4-col atoms of 512 B,
    /// atoms row-major over (row/128, kb/4); inside an atom
    /// (row%32)*16 + ((row/32)%4)*4 + kb%4.
    #[inline(always)]
    fn sf_offset(row: u32, kb: u32, kb_padded: u32) -> usize {
        let atom = (row / 128) * (kb_padded / 4) + kb / 4;
        (atom * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4) as usize
    }

    /// One thread per (row, 16-element block) of the activation.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn nvfp4_quant_act(
        x: *const u16,
        aq: *mut u8,
        asf: *mut u8,
        m: u32,
        k: u32,
        x_stride: u32,
        gscale: f32,
    ) {
        let kb = thread::blockIdx_x() * 128 + thread::threadIdx_x();
        let row = thread::blockIdx_y(); // 0..16
        let nkb = k / 16;
        if kb >= nkb {
            return;
        }
        let q = unsafe { aq.add((row * (k / 2) + kb * 8) as usize) };
        if row >= m {
            let mut i = 0;
            while i < 8 {
                unsafe { *q.add(i) = 0 };
                i += 1;
            }
            unsafe { *asf.add((row * nkb + kb) as usize) = 0 };
            return;
        }
        let src = unsafe { x.add((row * x_stride + kb * 16) as usize) };
        // Two passes over the 32-byte block (L1-resident) instead of a
        // register array (a dynamically indexed [f32; 16] spills to .local).
        let mut amax = 0.0f32;
        let mut i = 0;
        while i < 16 {
            let v = bf16_to_f32(unsafe { *src.add(i) });
            let a = if v < 0.0 { -v } else { v };
            if a > amax {
                amax = a;
            }
            i += 1;
        }
        // vLLM scaled_fp4_quant: sf = e4m3(amax / 6 * g); x_q = e2m1(x * g / sf)
        let sf = to_e4m3(amax * (gscale / 6.0));
        let sf_f = e4m3_to_f32(sf);
        let inv = if sf_f == 0.0 { 0.0 } else { gscale / sf_f };
        let mut j = 0;
        while j < 8 {
            let lo = to_e2m1(bf16_to_f32(unsafe { *src.add(2 * j) }) * inv);
            let hi = to_e2m1(bf16_to_f32(unsafe { *src.add(2 * j + 1) }) * inv);
            unsafe { *q.add(j) = (lo | (hi << 4)) as u8 };
            j += 1;
        }
        unsafe { *asf.add((row * nkb + kb) as usize) = sf as u8 };
    }

    #[inline(always)]
    fn ld32(p: *const u8, off: usize) -> u32 {
        unsafe { *(p.add(off) as *const u32) }
    }

    /// One k64 step's operands for this lane: the weight (B) fragment and
    /// its scales are loaded ONCE and shared by the MT m16 activation tiles.
    #[derive(Clone, Copy)]
    struct Step<const MT: usize> {
        a: [[u32; 4]; MT],
        sfa: [u32; MT],
        b: [u32; 2],
        sfb: u32,
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn load_step<const MT: usize>(
        k0: u32,
        lane: u32,
        aq: *const u8,
        a_stride: u32,
        a_rows: u32,
        w_row: *const u8,
        asf: *const u8,
        wsf: *const u8,
        asf_mode: u32,
        nkb: u32,
        kb_pad: u32,
        n0: u32,
    ) -> Step<MT> {
        let g = lane / 4;
        let t = lane % 4;
        let off = (k0 / 2 + 4 * t) as usize;
        let kb = k0 / 16;
        let mut s = Step {
            a: [[0u32; 4]; MT],
            sfa: [0u32; MT],
            b: [ld32(w_row, off), ld32(w_row, off + 16)],
            sfb: ld32(wsf, sf_offset(n0 + g, kb, kb_pad)),
        };
        unroll_mt!(MT, I, {
            let r0 = 16 * I as u32 + g;
            let r1 = r0 + 8;
            if r0 < a_rows {
                let p = (r0 * a_stride) as usize + off;
                s.a[I][0] = ld32(aq, p);
                s.a[I][2] = ld32(aq, p + 16);
            }
            if r1 < a_rows {
                let p = (r1 * a_stride) as usize + off;
                s.a[I][1] = ld32(aq, p);
                s.a[I][3] = ld32(aq, p + 16);
            }
            let sr = 16 * I as u32 + 8 * (lane & 1) + g;
            if sr < a_rows {
                s.sfa[I] = if asf_mode == 0 {
                    ld32(asf, (sr * nkb + kb) as usize)
                } else {
                    ld32(asf, sf_offset(sr, kb, kb_pad))
                };
            }
        });
        s
    }

    #[inline(always)]
    fn mma(c: [f32; 4], a: [u32; 4], b: [u32; 2], sfa: u32, sfb: u32) -> [f32; 4] {
        let zero: u16 = 0;
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};",
                out("=f") d0,
                out("=f") d1,
                out("=f") d2,
                out("=f") d3,
                in("r") a[0],
                in("r") a[1],
                in("r") a[2],
                in("r") a[3],
                in("r") b[0],
                in("r") b[1],
                in("f") c[0],
                in("f") c[1],
                in("f") c[2],
                in("f") c[3],
                in("r") sfa,
                in("h") zero,
                in("h") zero,
                in("r") sfb,
                in("h") zero,
                in("h") zero,
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    #[inline(always)]
    fn mma_all<const MT: usize>(mut c: [[f32; 4]; MT], s: Step<MT>) -> [[f32; 4]; MT] {
        unroll_mt!(MT, I, {
            c[I] = mma(c[I], s.a[I], s.b, s.sfa[I], s.sfb);
        });
        c
    }

    /// y[m, n] = alpha * (A_q x W_q^T) for M <= 16*MT: one warp per 8
    /// columns (CTA = 4 warps = 32 columns, grid.x = ceil(N/32); warps past
    /// N exit, so N % 8 == 0 suffices), MT m16 tiles per warp sharing each
    /// k64 weight fragment (register-resident accumulators), split-K over
    /// blockIdx.y (`kps` k64 steps per split). partial == null: write bf16
    /// `out` (alpha applied); else fp32 partials partial[(split*m + row) * n
    /// + col] for nvfp4_splitk_reduce. A: `a_rows` rows of `a_stride` bytes
    /// (rows >= a_rows read as 0); asf_mode 0 = row-major [a_rows, K/16]
    /// (our quant), 1 = vLLM's 128x4-swizzled activation scales (fused
    /// SiLU*mul / RMSNorm quant).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_body<const MT: usize>(
        out: *mut u16,
        partial: *mut f32,
        aq: *const u8,
        asf: *const u8,
        w: *const u8,
        wsf: *const u8,
        m: u32,
        n: u32,
        k: u32,
        out_stride: u32,
        a_stride: u32,
        a_rows: u32,
        asf_mode: u32,
        kps: u32,
        alpha: f32,
    ) {
        let tid = thread::threadIdx_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let n0 = (thread::blockIdx_x() * WARPS + warp) * 8;
        if n0 >= n {
            return; // warp-uniform
        }
        let split = thread::blockIdx_y();
        let k_begin = split * kps * 64;
        let mut k_end = k_begin + kps * 64;
        if k_end > k {
            k_end = k;
        }
        let nkb = k / 16;
        let kb_pad = (nkb + 3) / 4 * 4;
        let w_row = unsafe { w.add(((n0 + g) * (k / 2)) as usize) };
        let mut c = [[0.0f32; 4]; MT];
        if k_begin < k_end {
            // 2-stage software pipeline: next step's loads are in flight
            // while the current step's MT mmas issue.
            let mut cur = load_step::<MT>(
                k_begin, lane, aq, a_stride, a_rows, w_row, asf, wsf, asf_mode, nkb, kb_pad, n0,
            );
            let mut k0 = k_begin + 64;
            while k0 < k_end {
                let nxt = load_step::<MT>(
                    k0, lane, aq, a_stride, a_rows, w_row, asf, wsf, asf_mode, nkb, kb_pad, n0,
                );
                c = mma_all(c, cur);
                cur = nxt;
                k0 += 64;
            }
            c = mma_all(c, cur);
        }
        let col = n0 + 2 * t;
        let bf16 = partial.is_null();
        let store = |row: u32, v0: f32, v1: f32| {
            if row < m {
                if bf16 {
                    let v = f32_to_bf16(v0 * alpha) | (f32_to_bf16(v1 * alpha) << 16);
                    unsafe { *(out.add((row * out_stride + col) as usize) as *mut u32) = v };
                } else {
                    let p = unsafe { partial.add(((split * m + row) * n + col) as usize) };
                    unsafe {
                        *p = v0;
                        *p.add(1) = v1;
                    }
                }
            }
        };
        unroll_mt!(MT, I, {
            let row = 16 * I as u32 + g;
            store(row, c[I][0], c[I][1]);
            store(row + 8, c[I][2], c[I][3]);
        });
    }

    // One entry per m16-tile count (M <= 16, 32, 48, 64); the host picks it
    // from M, so the launch shape is a function of M only (graph-safe).
    #[kernel]
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn nvfp4_gemm_t1(
        out: *mut u16, partial: *mut f32, aq: *const u8, asf: *const u8, w: *const u8,
        wsf: *const u8, m: u32, n: u32, k: u32, out_stride: u32, a_stride: u32, a_rows: u32,
        asf_mode: u32, kps: u32, alpha: f32,
    ) {
        unsafe {
            gemm_body::<1>(out, partial, aq, asf, w, wsf, m, n, k, out_stride, a_stride, a_rows,
                           asf_mode, kps, alpha)
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn nvfp4_gemm_t2(
        out: *mut u16, partial: *mut f32, aq: *const u8, asf: *const u8, w: *const u8,
        wsf: *const u8, m: u32, n: u32, k: u32, out_stride: u32, a_stride: u32, a_rows: u32,
        asf_mode: u32, kps: u32, alpha: f32,
    ) {
        unsafe {
            gemm_body::<2>(out, partial, aq, asf, w, wsf, m, n, k, out_stride, a_stride, a_rows,
                           asf_mode, kps, alpha)
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn nvfp4_gemm_t3(
        out: *mut u16, partial: *mut f32, aq: *const u8, asf: *const u8, w: *const u8,
        wsf: *const u8, m: u32, n: u32, k: u32, out_stride: u32, a_stride: u32, a_rows: u32,
        asf_mode: u32, kps: u32, alpha: f32,
    ) {
        unsafe {
            gemm_body::<3>(out, partial, aq, asf, w, wsf, m, n, k, out_stride, a_stride, a_rows,
                           asf_mode, kps, alpha)
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn nvfp4_gemm_t4(
        out: *mut u16, partial: *mut f32, aq: *const u8, asf: *const u8, w: *const u8,
        wsf: *const u8, m: u32, n: u32, k: u32, out_stride: u32, a_stride: u32, a_rows: u32,
        asf_mode: u32, kps: u32, alpha: f32,
    ) {
        unsafe {
            gemm_body::<4>(out, partial, aq, asf, w, wsf, m, n, k, out_stride, a_stride, a_rows,
                           asf_mode, kps, alpha)
        }
    }

    /// out[row, col] = bf16(alpha * sum_s partial[(s*m + row) * n + col]),
    /// one thread per (row < m, col); fixed summation order (deterministic).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn nvfp4_splitk_reduce(
        out: *mut u16,
        partial: *const f32,
        m: u32,
        n: u32,
        splits: u32,
        out_stride: u32,
        alpha: f32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= m * n {
            return;
        }
        let row = i / n;
        let col = i % n;
        let mut acc = 0.0f32;
        let mut s = 0;
        while s < splits {
            acc += unsafe { *partial.add(((s * m + row) * n + col) as usize) };
            s += 1;
        }
        unsafe { *out.add((row * out_stride + col) as usize) = f32_to_bf16(acc * alpha) as u16 };
    }
}

fn main() {
    // Build-only crate: scripts/oxide_build.py -> PTX (.target sm_120a) ->
    // ptxas 13.0 -> sm_120a cubin; the plugin launches it (src/nvfp4_gemm_oxide.rs).
}
