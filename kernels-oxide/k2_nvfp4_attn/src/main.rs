// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4 on the cuda-oxide (SIMT) track: split-KV paged decode /
//! spec-verify attention over vLLM's HND NVFP4 KV cache, bf16 `mma.sync`
//! (m16n8k16, sm_80+ instruction, runs on sm_120) for QK^T and PV.
//!
//! Same algorithm, plan and numerics contract as the (retired) cutile
//! kernels — sm120/nvfp4_kv_patch/own_attn_ref.py `kernel_twin` is the CPU
//! transcription; src/nvfp4_attn.rs `plan` the launch plan (TN = 16).
//!
//! ABI (all params are raw pointers or 32-bit scalars — one `.param` each,
//! launched by src/nvfp4_attn_oxide.rs with cuLaunchKernel):
//!
//! nvfp4_attn_partial  grid (R, NS, 1), block (128, 1, 1), dyn smem =
//!   partial_smem_bytes(M, D)
//! nvfp4_attn_merge    grid (R, 1, 1),  block (128, 1, 1), dyn smem =
//!   NS*M*4 + M*4
//!
//! CTA = 4 warps. Per 16-token KV tile: all threads dequantize K and V
//! (e2m1 x e4m3, exact in bf16; V block scales de-swizzled with the store
//! kernel's swizzle_scale_offset) into shared memory; warp w computes the
//! partial S over its quarter of the head dim, the partials are summed in
//! the online-softmax step (one thread per row, exp2 domain, -inf safe),
//! then warp w accumulates O for its quarter of the head-dim columns in
//! registers (16 fixed mma slots: (M/16)*(D/32) <= 16 by the plan budget).
use cuda_device::{DynamicSharedArray, cuda_module, kernel, ptx_asm, thread};

#[cuda_module]
mod kernels {
    use super::*;

    const WARPS: u32 = 4;
    const TN: u32 = 16;
    const PAD: u32 = 8; // bf16 row padding (bank spread)
    const SLOTS: usize = 16;

    #[inline(always)]
    fn ex2(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    #[inline(always)]
    fn lg2(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("lg2.approx.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    /// f32 -> bf16 bits, round-to-nearest-even (NaN not expected here).
    #[inline(always)]
    fn bf16_bits(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    #[inline(always)]
    fn bf16_to_f32(b: u32) -> f32 {
        f32::from_bits(b << 16)
    }

    /// e4m3fn byte -> f32 (NaN code -> NaN; masked out by the caller).
    #[inline(always)]
    fn e4m3(b: u32) -> f32 {
        let e = (b >> 3) & 0xF;
        let m = b & 7;
        let mag = if e == 0 {
            (m as f32) * (1.0 / 512.0) // m/8 * 2^-6
        } else if e == 15 && m == 7 {
            f32::NAN
        } else {
            f32::from_bits(((e + 120) << 23) | (m << 20))
        };
        if b & 0x80 != 0 { -mag } else { mag }
    }

    /// e2m1 nibble -> f32.
    #[inline(always)]
    fn e2m1(n: u32) -> f32 {
        let c = n & 7;
        let mag = if c < 2 {
            (c as f32) * 0.5
        } else {
            // c = 2..7: 1, 1.5, 2, 3, 4, 6  (exponent c>>1, mantissa c&1)
            f32::from_bits(((126 + (c >> 1)) << 23) | ((c & 1) << 22))
        };
        if n & 8 != 0 { -mag } else { mag }
    }

    #[inline(always)]
    fn swizzle_scale_offset(t: u32, g: u32, s: u32) -> u32 {
        let grp = s / 4;
        ((t / 4) * 4 + g / grp) * s + (g % grp) * 4 + t % 4
    }

    #[inline(always)]
    unsafe fn ld_u32(p: *const u16, idx: u32) -> u32 {
        unsafe { *(p.add(idx as usize) as *const u32) }
    }

    #[kernel]
    pub fn nvfp4_attn_partial(
        q: *const u16,
        k_data: *const u8,
        k_sf: *const u8,
        v_data: *const u8,
        v_sf: *const u8,
        block_table: *const i32,
        seq_lens: *const i32,
        o_part: *mut u16,
        lse_part: *mut f32,
        q_tok_stride: u32,
        page_bytes: u32,
        bt_stride: u32,
        hkv: u32,
        g_heads: u32,
        gp: u32,
        qt_n: u32,
        m_rows: u32,
        d: u32,
        nqt: u32,
        q_len: u32,
        page_size: u32,
        ns: u32,
        window_left: i32,
        qk_scale_log2: f32,
    ) {
        let tid = thread::threadIdx_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let gq = lane / 4;
        let t4 = lane % 4;
        let r = thread::blockIdx_x();
        let s = thread::blockIdx_y();
        let b = r / (hkv * nqt);
        let h = (r / nqt) % hkv;
        let qt = r % nqt;
        let dh = d / 2;
        let sd = d / 16;
        let row_s = d + PAD; // smem row stride (bf16 elems)

        // ---- shared memory carve-up ------------------------------------
        let qs: *mut u16 = DynamicSharedArray::<u16>::get();
        let ks: *mut u16 = unsafe { qs.add((m_rows * row_s) as usize) };
        let vs: *mut u16 = unsafe { ks.add((TN * row_s) as usize) };
        let sp: *mut f32 = unsafe { vs.add((TN * row_s) as usize) as *mut f32 };
        let ps: *mut u16 = unsafe { sp.add((WARPS * m_rows * TN) as usize) as *mut u16 };
        let st: *mut f32 = unsafe { ps.add((m_rows * TN) as usize) as *mut f32 };
        // st: [m_i | l_i | alpha] x m_rows

        let kv_len = unsafe { *seq_lens.add(b as usize) };
        let q0 = kv_len - q_len as i32 + (qt * qt_n) as i32;
        let mut hi = q0 + qt_n as i32;
        if hi > kv_len {
            hi = kv_len;
        }
        if hi < 0 {
            hi = 0;
        }
        let mut lo = 0i32;
        if window_left >= 0 {
            lo = q0 - window_left;
            if lo < 0 {
                lo = 0;
            }
        }
        let lo_t = (lo as u32) / TN;
        let hi_t = (hi as u32).div_ceil(TN);
        let n_t = if hi_t > lo_t { hi_t - lo_t } else { 0 };
        let per = n_t.div_ceil(ns);
        let t0 = lo_t + s * per;
        let mut t1 = t0 + per;
        if t1 > hi_t {
            t1 = hi_t;
        }

        // ---- Q tile [M, D] -> smem (padded rows/heads/tokens = 0) --------
        let mut idx = tid;
        while idx < m_rows * d {
            let row = idx / d;
            let c = idx % d;
            let i = row / gp;
            let gg = row % gp;
            let tok = qt * qt_n + i;
            let v = if tok < q_len && gg < g_heads {
                unsafe {
                    *q.add(
                        ((b * q_len + tok) * q_tok_stride + (h * g_heads + gg) * d + c) as usize,
                    )
                }
            } else {
                0u16
            };
            unsafe { *qs.add((row * row_s + c) as usize) = v };
            idx += WARPS * 32;
        }
        if tid < m_rows {
            unsafe {
                *st.add(tid as usize) = f32::NEG_INFINITY;
                *st.add((m_rows + tid) as usize) = 0.0;
            }
        }

        // O accumulators: slot k -> (m-tile k / ntw, n-tile k % ntw) of this
        // warp's column quarter.
        let ntw = d / 32; // n8 tiles per warp quarter
        let used = (m_rows / 16) * ntw;
        let col0 = warp * (d / 4);
        let mut acc = [[0.0f32; 4]; SLOTS];

        let tiles_per_page = page_size / TN;
        let mut j = t0;
        while j < t1 {
            thread::sync_threads();
            let tok0 = j * TN;
            let page = unsafe {
                *block_table.add((b * bt_stride + j / tiles_per_page) as usize)
            } as u64;
            let tip0 = (j % tiles_per_page) * TN;
            let base = page * page_bytes as u64;
            let head_d = (h * page_size) as u64;

            // ---- dequant K and V tile -> smem (8 elements per unit) -----
            let units = TN * d / 8;
            let mut u = tid;
            while u < units {
                let tt = u / (d / 8);
                let c8 = u % (d / 8);
                let tip = tip0 + tt;
                let valid = ((tok0 + tt) as i32) < kv_len;
                let off = base + (head_d + tip as u64) * dh as u64 + (c8 * 4) as u64;
                let kw = unsafe { *(k_data.add(off as usize) as *const u32) };
                let vw = unsafe { *(v_data.add(off as usize) as *const u32) };
                let g16 = c8 / 2;
                let ksc = e4m3(unsafe {
                    *k_sf.add((base + (head_d + tip as u64) * sd as u64 + g16 as u64) as usize)
                } as u32);
                let vsc = e4m3(unsafe {
                    *v_sf.add(
                        (base + head_d * sd as u64 + swizzle_scale_offset(tip, g16, sd) as u64)
                            as usize,
                    )
                } as u32);
                let mut e = 0u32;
                while e < 8 {
                    let kv = if valid { e2m1(kw >> (4 * e)) * ksc } else { 0.0 };
                    let vv = if valid { e2m1(vw >> (4 * e)) * vsc } else { 0.0 };
                    unsafe {
                        *ks.add((tt * row_s + c8 * 8 + e) as usize) = bf16_bits(kv) as u16;
                        *vs.add((tt * row_s + c8 * 8 + e) as usize) = bf16_bits(vv) as u16;
                    }
                    e += 1;
                }
                u += WARPS * 32;
            }
            thread::sync_threads();

            // ---- partial S = Q K^T over this warp's k quarter -------------
            let kq0 = warp * (d / 4);
            let mut mt = 0u32;
            while mt < m_rows / 16 {
                let mut nt = 0u32;
                while nt < 2 {
                    let mut c = [0.0f32; 4];
                    let mut k0 = kq0;
                    while k0 < kq0 + d / 4 {
                        let ra = (mt * 16 + gq) * row_s + k0 + 2 * t4;
                        let rb = (mt * 16 + gq + 8) * row_s + k0 + 2 * t4;
                        let a = unsafe {
                            [ld_u32(qs, ra), ld_u32(qs, rb), ld_u32(qs, ra + 8), ld_u32(qs, rb + 8)]
                        };
                        let kr = (nt * 8 + gq) * row_s + k0 + 2 * t4;
                        let bb = unsafe { [ld_u32(ks, kr), ld_u32(ks, kr + 8)] };
                        c = unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c, a, bb) };
                        k0 += 16;
                    }
                    let base_s = warp * m_rows * TN;
                    let r0 = mt * 16 + gq;
                    let cc = nt * 8 + 2 * t4;
                    unsafe {
                        *sp.add((base_s + r0 * TN + cc) as usize) = c[0];
                        *sp.add((base_s + r0 * TN + cc + 1) as usize) = c[1];
                        *sp.add((base_s + (r0 + 8) * TN + cc) as usize) = c[2];
                        *sp.add((base_s + (r0 + 8) * TN + cc + 1) as usize) = c[3];
                    }
                    nt += 1;
                }
                mt += 1;
            }
            thread::sync_threads();

            // ---- online softmax, one thread per row ----------------------
            if tid < m_rows {
                let row = tid;
                let qpos = q0 + (row / gp) as i32;
                let m_old = unsafe { *st.add(row as usize) };
                let mut sc = [0.0f32; 16];
                let mut mx = f32::NEG_INFINITY;
                let mut c = 0u32;
                while c < TN {
                    let kpos = (tok0 + c) as i32;
                    let mut ok = kpos <= qpos && kpos < kv_len;
                    if window_left >= 0 && kpos + window_left < qpos {
                        ok = false;
                    }
                    let mut v = f32::NEG_INFINITY;
                    if ok {
                        let mut acc_s = 0.0f32;
                        let mut w = 0u32;
                        while w < WARPS {
                            acc_s += unsafe { *sp.add((w * m_rows * TN + row * TN + c) as usize) };
                            w += 1;
                        }
                        v = acc_s * qk_scale_log2;
                    }
                    sc[c as usize] = v;
                    if v > mx {
                        mx = v;
                    }
                    c += 1;
                }
                let m_new = if m_old > mx { m_old } else { mx };
                let m_safe = if m_new == f32::NEG_INFINITY { 0.0 } else { m_new };
                let alpha = ex2(m_old - m_safe);
                let mut psum = 0.0f32;
                let mut c = 0u32;
                while c < TN {
                    let p = ex2(sc[c as usize] - m_safe);
                    psum += p;
                    unsafe { *ps.add((row * TN + c) as usize) = bf16_bits(p) as u16 };
                    c += 1;
                }
                unsafe {
                    *st.add(row as usize) = m_new;
                    let l = *st.add((m_rows + row) as usize);
                    *st.add((m_rows + row) as usize) = l * alpha + psum;
                    *st.add((2 * m_rows + row) as usize) = alpha;
                }
            }
            thread::sync_threads();

            // ---- O[:, warp cols] = O * alpha + P V --------------------------
            let mut k = 0usize;
            while k < SLOTS {
                if (k as u32) < used {
                    let mt = k as u32 / ntw;
                    let nt = k as u32 % ntw;
                    let r0 = mt * 16 + gq;
                    let a0 = unsafe { *st.add((2 * m_rows + r0) as usize) };
                    let a8 = unsafe { *st.add((2 * m_rows + r0 + 8) as usize) };
                    let c = acc[k];
                    let c = [c[0] * a0, c[1] * a0, c[2] * a8, c[3] * a8];
                    let pa = r0 * TN + 2 * t4;
                    let pb = (r0 + 8) * TN + 2 * t4;
                    let a = unsafe {
                        [ld_u32(ps, pa), ld_u32(ps, pb), ld_u32(ps, pa + 8), ld_u32(ps, pb + 8)]
                    };
                    let col = col0 + nt * 8 + gq;
                    let v = |tok: u32| unsafe { *vs.add((tok * row_s + col) as usize) as u32 };
                    let bb = [
                        v(2 * t4) | (v(2 * t4 + 1) << 16),
                        v(2 * t4 + 8) | (v(2 * t4 + 9) << 16),
                    ];
                    acc[k] = unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c, a, bb) };
                }
                k += 1;
            }
            j += 1;
        }
        thread::sync_threads();

        // ---- normalized partial O (bf16) + lse (log2) ----------------------
        let slot_base = (r * ns + s) * m_rows;
        let mut k = 0usize;
        while k < SLOTS {
            if (k as u32) < used {
                let mt = k as u32 / ntw;
                let nt = k as u32 % ntw;
                let r0 = mt * 16 + gq;
                let l0 = unsafe { *st.add((m_rows + r0) as usize) };
                let l8 = unsafe { *st.add((m_rows + r0 + 8) as usize) };
                let i0 = if l0 == 0.0 { 0.0 } else { 1.0 / l0 };
                let i8 = if l8 == 0.0 { 0.0 } else { 1.0 / l8 };
                let col = col0 + nt * 8 + 2 * t4;
                let c = acc[k];
                unsafe {
                    let o0 = o_part.add(((slot_base + r0) * d + col) as usize) as *mut u32;
                    *o0 = bf16_bits(c[0] * i0) | (bf16_bits(c[1] * i0) << 16);
                    let o8 = o_part.add(((slot_base + r0 + 8) * d + col) as usize) as *mut u32;
                    *o8 = bf16_bits(c[2] * i8) | (bf16_bits(c[3] * i8) << 16);
                }
            }
            k += 1;
        }
        if tid < m_rows {
            let m = unsafe { *st.add(tid as usize) };
            let l = unsafe { *st.add((m_rows + tid) as usize) };
            let lse = if l == 0.0 { f32::NEG_INFINITY } else { m + lg2(l) };
            unsafe { *lse_part.add((slot_base + tid) as usize) = lse };
        }
    }

    /// LSE merge of the NS partials for launch row r; v_scale; bf16 store
    /// into out [B*q_len, HQ, D] (padded q-token / head rows skipped).
    #[kernel]
    pub fn nvfp4_attn_merge(
        out: *mut u16,
        o_part: *const u16,
        lse_part: *const f32,
        hkv: u32,
        g_heads: u32,
        gp: u32,
        qt_n: u32,
        m_rows: u32,
        d: u32,
        nqt: u32,
        q_len: u32,
        ns: u32,
        v_scale: f32,
    ) {
        let tid = thread::threadIdx_x();
        let r = thread::blockIdx_x();
        let b = r / (hkv * nqt);
        let h = (r / nqt) % hkv;
        let qt = r % nqt;
        let hq = hkv * g_heads;
        let wts: *mut f32 = DynamicSharedArray::<f32>::get(); // [ns][M]
        let inv: *mut f32 = unsafe { wts.add((ns * m_rows) as usize) }; // [M]
        if tid < m_rows {
            let mut mx = f32::NEG_INFINITY;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add(((r * ns + s) * m_rows + tid) as usize) };
                if v > mx {
                    mx = v;
                }
                s += 1;
            }
            let mx = if mx == f32::NEG_INFINITY { 0.0 } else { mx };
            let mut sum = 0.0f32;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add(((r * ns + s) * m_rows + tid) as usize) };
                let w = ex2(v - mx);
                unsafe { *wts.add((s * m_rows + tid) as usize) = w };
                sum += w;
                s += 1;
            }
            unsafe { *inv.add(tid as usize) = if sum == 0.0 { 0.0 } else { v_scale / sum } };
        }
        thread::sync_threads();
        let mut idx = tid;
        while idx < m_rows * d {
            let row = idx / d;
            let c = idx % d;
            let i = row / gp;
            let gg = row % gp;
            let tok = qt * qt_n + i;
            if tok < q_len && gg < g_heads {
                let mut acc = 0.0f32;
                let mut s = 0u32;
                while s < ns {
                    let w = unsafe { *wts.add((s * m_rows + row) as usize) };
                    if w != 0.0 {
                        let o = unsafe {
                            *o_part.add((((r * ns + s) * m_rows + row) * d + c) as usize)
                        };
                        acc += w * bf16_to_f32(o as u32);
                    }
                    s += 1;
                }
                let v = acc * unsafe { *inv.add(row as usize) };
                unsafe {
                    *out.add((((b * q_len + tok) * hq + h * g_heads + gg) * d + c) as usize) =
                        bf16_bits(v) as u16;
                }
            }
            idx += 128;
        }
    }
}

fn main() {}
