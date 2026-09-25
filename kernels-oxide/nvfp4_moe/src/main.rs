// SPDX-License-Identifier: Apache-2.0
//! NVFP4 W4A4 MoE decode kernels (gemma-4-26b-a4b-nvfp4 routed experts):
//! expert-grouped m16 block-scaled FP4 mma (sm_120a,
//! `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...ue4m3`).
//!
//! Replaces FlashInfer `cutlass_fused_moe` (128-row tiles) for decode-sized
//! batches, where each expert sees only ~1-16 routed rows. Six stream-ordered
//! launches, all grids static in M (CUDA-graph capturable):
//!   1 moe_route       topk_ids -> active experts (ascending id) + per-slot
//!                     row lists (ascending pair id): deterministic
//!   2 moe_quant_rows  x bf16 [M, H] -> NVFP4 (a1 global scale)
//!   3 moe_fc1         per (slot, 32-col tile of I): up + gate tiles share
//!                     the A fragments; h = act(alpha*gate) * alpha*up, f32
//!   4 moe_quant_rows  h f32 [P, I] -> NVFP4 (a2 global scale)
//!   5 moe_fc2         per (slot, 32-col tile of H): y[p] = g2_alpha * w_p * acc
//!   6 moe_combine     out[t] = sum_k y[t*topk + k] (fixed k order) -> bf16
//! Layouts = vLLM FLASHINFER_CUTLASS after process_weights_after_loading
//! (quantization/utils/flashinfer_fp4_moe.py:309-420): w13 uint8
//! [E, 2I, H/2] ordered [w3(up); w1(gate)]; w13_sf e4m3 per-expert 128x4
//! swizzled [E, round128(2I), round4(H/16)]; w2 [E, H, I/2]; w2_sf
//! [E, round128(H), round4(I/16)]; g1/g2 alphas f32 [E]; one activation
//! global scale per layer. Pair p = token * topk + k.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

    const WARPS: u32 = 4;

    #[inline(always)]
    fn ld32(p: *const u8, off: usize) -> u32 {
        unsafe { *(p.add(off) as *const u32) }
    }

    #[inline(always)]
    fn f32_to_bf16(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    #[inline(always)]
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

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

    #[inline(always)]
    fn ex2(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    #[inline(always)]
    fn tanh(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("tanh.approx.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    /// act_and_mul activation: 0 = silu, 1 = gelu_tanh (gemma-4).
    #[inline(always)]
    fn act(x: f32, kind: u32) -> f32 {
        if kind == 1 {
            let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + tanh(u))
        } else {
            x / (1.0 + ex2(-x * 1.442_695))
        }
    }

    /// CUTLASS 128x4 swizzled scale byte offset (vLLM swizzle_blockscale).
    #[inline(always)]
    fn sf_offset(row: u32, kb: u32, kb_pad: u32) -> usize {
        let atom = (row / 128) * (kb_pad / 4) + kb / 4;
        (atom * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4) as usize
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn mma(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32, sfa: u32, sfb: u32) -> [f32; 4] {
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
                in("r") b0,
                in("r") b1,
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

    /// Single CTA of 256 threads (E <= 256). Dynamic smem: 2*E i32.
    /// slot_expert[s] = s-th active expert (ascending id) or -1;
    /// slot_off / slot_cnt: its rows in pair_list (ascending pair id).
    /// Pairs with ids outside [0, E) are ignored (combine skips them).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_route(
        topk_ids: *const i32,
        ids_i64: u32,
        pairs: u32,
        num_experts: u32,
        max_slots: u32,
        slot_expert: *mut i32,
        slot_off: *mut i32,
        slot_cnt: *mut i32,
        pair_list: *mut i32,
    ) {
        let tid = thread::threadIdx_x();
        let sh: *mut i32 = DynamicSharedArray::<i32>::get();
        let id_of = |p: u32| -> i32 {
            if ids_i64 != 0 {
                unsafe { *(topk_ids as *const i64).add(p as usize) as i32 }
            } else {
                unsafe { *topk_ids.add(p as usize) }
            }
        };
        let mut cnt = 0i32;
        if tid < num_experts {
            let mut p = 0;
            while p < pairs {
                if id_of(p) == tid as i32 {
                    cnt += 1;
                }
                p += 1;
            }
            unsafe { *sh.add(tid as usize) = cnt };
        }
        thread::sync_threads();
        if tid == 0 {
            let mut off = 0i32;
            let mut s = 0u32;
            let mut e = 0;
            while e < num_experts {
                let c = unsafe { *sh.add(e as usize) };
                unsafe { *sh.add((num_experts + e) as usize) = off };
                if c > 0 && s < max_slots {
                    unsafe {
                        *slot_expert.add(s as usize) = e as i32;
                        *slot_off.add(s as usize) = off;
                        *slot_cnt.add(s as usize) = c;
                    }
                    s += 1;
                }
                off += c;
                e += 1;
            }
            while s < max_slots {
                unsafe {
                    *slot_expert.add(s as usize) = -1;
                    *slot_cnt.add(s as usize) = 0;
                }
                s += 1;
            }
        }
        thread::sync_threads();
        if tid < num_experts && cnt > 0 {
            let mut w = unsafe { *sh.add((num_experts + tid) as usize) };
            let mut p = 0;
            while p < pairs {
                if id_of(p) == tid as i32 {
                    unsafe { *pair_list.add(w as usize) = p as i32 };
                    w += 1;
                }
                p += 1;
            }
        }
    }

    /// One thread per (row, 16-block): vLLM scaled_fp4_quant math
    /// (sf = e4m3(amax16 * g / 6), q = e2m1_rne(x * g / sf)), row-major out
    /// q [rows, K/2], sf [rows, K/16]. Input bf16 (x_f32 == 0) or f32.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_quant_rows(
        x: *const u8,
        x_f32: u32,
        q: *mut u8,
        sf: *mut u8,
        rows: u32,
        k: u32,
        x_stride: u32,
        gscale: f32,
    ) {
        let kb = thread::blockIdx_x() * 128 + thread::threadIdx_x();
        let row = thread::blockIdx_y();
        let nkb = k / 16;
        if kb >= nkb || row >= rows {
            return;
        }
        let base = (row * x_stride + kb * 16) as usize;
        let load = |i: usize| -> f32 {
            if x_f32 != 0 {
                unsafe { *(x as *const f32).add(base + i) }
            } else {
                bf16_to_f32(unsafe { *(x as *const u16).add(base + i) })
            }
        };
        let mut amax = 0.0f32;
        let mut i = 0;
        while i < 16 {
            let v = load(i);
            let a = if v < 0.0 { -v } else { v };
            if a > amax {
                amax = a;
            }
            i += 1;
        }
        let s = to_e4m3(amax * (gscale / 6.0));
        let s_f = e4m3_to_f32(s);
        let inv = if s_f == 0.0 { 0.0 } else { gscale / s_f };
        let qo = unsafe { q.add((row * (k / 2) + kb * 8) as usize) };
        let mut j = 0;
        while j < 8 {
            let lo = to_e2m1(load(2 * j) * inv);
            let hi = to_e2m1(load(2 * j + 1) * inv);
            unsafe { *qo.add(j) = (lo | (hi << 4)) as u8 };
            j += 1;
        }
        unsafe { *sf.add((row * nkb + kb) as usize) = s as u8 };
    }

    /// fc1 + act_and_mul. grid (I/32, slots), 4 warps x 8 columns of I.
    /// inter[p, j] = act(g1[e] * gate_j) * g1[e] * up_j   (f32 [P, I]).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_fc1(
        inter: *mut f32,
        aq: *const u8,
        asf: *const u8,
        w13: *const u8,
        w13_sf: *const u8,
        g1_alpha: *const f32,
        slot_expert: *const i32,
        slot_off: *const i32,
        slot_cnt: *const i32,
        pair_list: *const i32,
        topk: u32,
        hdim: u32,
        idim: u32,
        act_kind: u32,
    ) {
        let slot = thread::blockIdx_y() as usize;
        let e = unsafe { *slot_expert.add(slot) };
        if e < 0 {
            return; // CTA-uniform
        }
        let e = e as u32;
        let off = unsafe { *slot_off.add(slot) } as u32;
        let cnt = unsafe { *slot_cnt.add(slot) } as u32;
        let tid = thread::threadIdx_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let j0 = (thread::blockIdx_x() * WARPS + warp) * 8;
        if j0 >= idim {
            return; // warp-uniform
        }
        let kh = hdim / 2;
        let nkb = hdim / 16;
        let kb_pad = (nkb + 3) / 4 * 4;
        let rows_pad = (2 * idim + 127) / 128 * 128;
        let w_e = unsafe { w13.add(e as usize * (2 * idim * kh) as usize) };
        let sf_e = unsafe { w13_sf.add(e as usize * (rows_pad * kb_pad) as usize) };
        let up_row = j0 + g;
        let gate_row = idim + j0 + g;
        let w_up = unsafe { w_e.add((up_row * kh) as usize) };
        let w_gate = unsafe { w_e.add((gate_row * kh) as usize) };
        let alpha = unsafe { *g1_alpha.add(e as usize) };
        let sfa_row = 8 * (lane & 1) + lane / 4;
        let mut r0 = 0;
        while r0 < cnt {
            // rows of this 16-row chunk -> token rows of aq
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let oks = r0 + sfa_row < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let ps = if oks { (unsafe { *pair_list.add((off + r0 + sfa_row) as usize) }) as u32 } else { 0 };
            let a_row0 = unsafe { aq.add(((p0 / topk) * kh) as usize) };
            let a_row1 = unsafe { aq.add(((p1 / topk) * kh) as usize) };
            let sfa_base = unsafe { asf.add(((ps / topk) * nkb) as usize) };
            let mut cu = [0.0f32; 4];
            let mut cg = [0.0f32; 4];
            let mut k0 = 0;
            while k0 < hdim {
                let o = (k0 / 2 + 4 * t) as usize;
                let kb = k0 / 16;
                let a = [
                    if ok0 { ld32(a_row0, o) } else { 0 },
                    if ok1 { ld32(a_row1, o) } else { 0 },
                    if ok0 { ld32(a_row0, o + 16) } else { 0 },
                    if ok1 { ld32(a_row1, o + 16) } else { 0 },
                ];
                let sfa = if oks { ld32(sfa_base, kb as usize) } else { 0 };
                cu = mma(
                    cu,
                    a,
                    ld32(w_up, o),
                    ld32(w_up, o + 16),
                    sfa,
                    ld32(sf_e, sf_offset(up_row, kb, kb_pad)),
                );
                cg = mma(
                    cg,
                    a,
                    ld32(w_gate, o),
                    ld32(w_gate, o + 16),
                    sfa,
                    ld32(sf_e, sf_offset(gate_row, kb, kb_pad)),
                );
                k0 += 64;
            }
            let col = (j0 + 2 * t) as usize;
            if ok0 {
                let dst = unsafe { inter.add(p0 as usize * idim as usize + col) };
                unsafe {
                    *dst = act(alpha * cg[0], act_kind) * (alpha * cu[0]);
                    *dst.add(1) = act(alpha * cg[1], act_kind) * (alpha * cu[1]);
                }
            }
            if ok1 {
                let dst = unsafe { inter.add(p1 as usize * idim as usize + col) };
                unsafe {
                    *dst = act(alpha * cg[2], act_kind) * (alpha * cu[2]);
                    *dst.add(1) = act(alpha * cg[3], act_kind) * (alpha * cu[3]);
                }
            }
            r0 += 16;
        }
    }

    /// fc2. grid (H/32, slots). y[p, h] = g2[e] * topk_w[p] * (hq[p] . w2[e, h]).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_fc2(
        y: *mut f32,
        hq: *const u8,
        hsf: *const u8,
        w2: *const u8,
        w2_sf: *const u8,
        g2_alpha: *const f32,
        topk_w: *const f32,
        slot_expert: *const i32,
        slot_off: *const i32,
        slot_cnt: *const i32,
        pair_list: *const i32,
        hdim: u32,
        idim: u32,
    ) {
        let slot = thread::blockIdx_y() as usize;
        let e = unsafe { *slot_expert.add(slot) };
        if e < 0 {
            return;
        }
        let e = e as u32;
        let off = unsafe { *slot_off.add(slot) } as u32;
        let cnt = unsafe { *slot_cnt.add(slot) } as u32;
        let tid = thread::threadIdx_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let h0 = (thread::blockIdx_x() * WARPS + warp) * 8;
        if h0 >= hdim {
            return;
        }
        let kh = idim / 2;
        let nkb = idim / 16;
        let kb_pad = (nkb + 3) / 4 * 4;
        let rows_pad = (hdim + 127) / 128 * 128;
        let w_row = unsafe { w2.add(e as usize * (hdim * kh) as usize + ((h0 + g) * kh) as usize) };
        let sf_e = unsafe { w2_sf.add(e as usize * (rows_pad * kb_pad) as usize) };
        let alpha = unsafe { *g2_alpha.add(e as usize) };
        let sfa_row = 8 * (lane & 1) + lane / 4;
        let mut r0 = 0;
        while r0 < cnt {
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let oks = r0 + sfa_row < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let ps = if oks { (unsafe { *pair_list.add((off + r0 + sfa_row) as usize) }) as u32 } else { 0 };
            let a_row0 = unsafe { hq.add((p0 * kh) as usize) };
            let a_row1 = unsafe { hq.add((p1 * kh) as usize) };
            let sfa_base = unsafe { hsf.add((ps * nkb) as usize) };
            let mut c = [0.0f32; 4];
            let mut k0 = 0;
            while k0 < idim {
                let o = (k0 / 2 + 4 * t) as usize;
                let kb = k0 / 16;
                let a = [
                    if ok0 { ld32(a_row0, o) } else { 0 },
                    if ok1 { ld32(a_row1, o) } else { 0 },
                    if ok0 { ld32(a_row0, o + 16) } else { 0 },
                    if ok1 { ld32(a_row1, o + 16) } else { 0 },
                ];
                let sfa = if oks { ld32(sfa_base, kb as usize) } else { 0 };
                c = mma(
                    c,
                    a,
                    ld32(w_row, o),
                    ld32(w_row, o + 16),
                    sfa,
                    ld32(sf_e, sf_offset(h0 + g, kb, kb_pad)),
                );
                k0 += 64;
            }
            let col = (h0 + 2 * t) as usize;
            if ok0 {
                let s = alpha * unsafe { *topk_w.add(p0 as usize) };
                let dst = unsafe { y.add(p0 as usize * hdim as usize + col) };
                unsafe {
                    *dst = c[0] * s;
                    *dst.add(1) = c[1] * s;
                }
            }
            if ok1 {
                let s = alpha * unsafe { *topk_w.add(p1 as usize) };
                let dst = unsafe { y.add(p1 as usize * hdim as usize + col) };
                unsafe {
                    *dst = c[2] * s;
                    *dst.add(1) = c[3] * s;
                }
            }
            r0 += 16;
        }
    }

    /// out[t, h] = bf16(sum_k y[t*topk + k, h]) over valid expert ids, k in
    /// ascending order (deterministic). One thread per (t, h).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_combine(
        out: *mut u16,
        y: *const f32,
        topk_ids: *const i32,
        ids_i64: u32,
        m: u32,
        topk: u32,
        hdim: u32,
        num_experts: u32,
        out_stride: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= m * hdim {
            return;
        }
        let tok = i / hdim;
        let h = i % hdim;
        let mut acc = 0.0f32;
        let mut k = 0;
        while k < topk {
            let p = tok * topk + k;
            let id = if ids_i64 != 0 {
                unsafe { *(topk_ids as *const i64).add(p as usize) as i32 }
            } else {
                unsafe { *topk_ids.add(p as usize) }
            };
            if id >= 0 && (id as u32) < num_experts {
                acc += unsafe { *y.add((p * hdim + h) as usize) };
            }
            k += 1;
        }
        unsafe { *out.add((tok * out_stride + h) as usize) = f32_to_bf16(acc) as u16 };
    }
}

fn main() {
    // Build-only crate: scripts/oxide_build.py -> PTX (.target sm_120a) ->
    // ptxas 13.0 -> sm_120a cubin; the plugin launches it (src/nvfp4_moe_oxide.rs).
}
