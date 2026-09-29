// SPDX-License-Identifier: Apache-2.0
//! FP8 block-scaled MoE decode kernels (qwen3.8-flash-next MTP draft routed
//! experts: e4m3 weights with 128x128 f32 block scales, dynamic per-token
//! 1x128-group activation scales), sm_120a
//! `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
//!
//! Replaces vLLM's TritonOrDeepGemmExperts (generic Triton on SM120) for
//! decode-sized batches. Same stages and launch plans as
//! kernels-oxide/nvfp4_moe (legacy 6: moe_route, moe_quant_rows, moe_fc1,
//! moe_quant_rows, moe_fc2, moe_combine; front 4: moe_route_quant,
//! moe_fc1_quant, moe_fc2, moe_combine; fused 3: moe_route_quant,
//! moe_fc1_quant, moe_fc2_combine), all grids static in M (CUDA-graph
//! capturable), all bit-identical:
//!   route       topk_ids -> active local experts (ascending id) + per-slot
//!               row lists (ascending pair id): deterministic
//!   quant x     x bf16 [M, H] -> e4m3 + f32 scale per (row, 128 cols)
//!   fc1         gate + up tiles share the A fragments; per 128-K block the
//!               f32 mma partial is rescaled by a_scale[row, kb] *
//!               w_scale[n/128, kb]; h = silu(gate) * up, f32 [P, I]
//!   quant h     h f32 [P, I] -> e4m3 + f32 scale per (pair, 128 cols); the
//!               fused fc1 owns a whole 128-col group per CTA to do it
//!               in-CTA (moe_fc1_quant)
//!   fc2         y[p] = topk_w[p] * acc
//!   combine     out[t] = sum_k y[t*topk + k] (fixed k order) -> bf16
//! Layouts = vLLM Fp8MoEMethod (block quant) for the TRITON / DEEPGEMM
//! backends: w13 e4m3 [E, 2I, H] rows [w1 (gate); w3 (up)] (silu_and_mul
//! takes gate = first half); w13 scale f32 [E, 2I/128, H/128]; w2 e4m3
//! [E, H, I]; w2 scale f32 [E, H/128, I/128]. Pair p = token * topk + k.
//!
//! K permutation (16-byte weight loads): within one 128-K block lane t loads
//! bytes [32t, 32t + 32) of its A rows and B row (two 16 B loads each) =
//! eight u32 words w0..w7; mma i (0..3) takes words (2i, 2i+1) as its
//! (k 4t.., k 16+4t..) register pair. A and B use the same map, which is a
//! bijection of the block's 128 k onto 4 x 32 mma k, so the block dot
//! product is unchanged and every scale stays per 128-K block.

use cuda_device::cuda_module;

/// moe_route body (a macro, not a fn: the dynamic-smem pointer is an
/// addrspace(3) value that cuda-oxide cannot pass to a generic-pointer
/// parameter). One CTA of 256 threads (E <= 256), `sh` = 2*E i32 of dynamic
/// smem. slot_expert[s] = s-th active expert (ascending id) or -1; slot_off
/// / slot_cnt: its rows in pair_list (ascending pair id). Pairs whose local
/// id falls outside [0, E) are ignored (the combine skips them, so a token
/// with no local expert is exactly 0).
macro_rules! route_cta {
    ($tid:expr, $sh:expr, $topk_ids:expr, $ids_i64:expr, $id_base:expr, $pairs:expr,
     $num_experts:expr, $max_slots:expr, $slot_expert:expr, $slot_off:expr, $slot_cnt:expr,
     $pair_list:expr) => {{
        let id_of = |p: u32| -> i32 { local_id($topk_ids, $ids_i64, $id_base, p) };
        let mut cnt = 0i32;
        if $tid < $num_experts {
            let mut p = 0;
            while p < $pairs {
                if id_of(p) == $tid as i32 {
                    cnt += 1;
                }
                p += 1;
            }
            unsafe { *$sh.add($tid as usize) = cnt };
        }
        cuda_device::thread::sync_threads();
        if $tid == 0 {
            let mut off = 0i32;
            let mut s = 0u32;
            let mut e = 0;
            while e < $num_experts {
                let c = unsafe { *$sh.add(e as usize) };
                unsafe { *$sh.add(($num_experts + e) as usize) = off };
                if c > 0 && s < $max_slots {
                    unsafe {
                        *$slot_expert.add(s as usize) = e as i32;
                        *$slot_off.add(s as usize) = off;
                        *$slot_cnt.add(s as usize) = c;
                    }
                    s += 1;
                }
                off += c;
                e += 1;
            }
            while s < $max_slots {
                unsafe {
                    *$slot_expert.add(s as usize) = -1;
                    *$slot_cnt.add(s as usize) = 0;
                }
                s += 1;
            }
        }
        cuda_device::thread::sync_threads();
        if $tid < $num_experts && cnt > 0 {
            let mut w = unsafe { *$sh.add(($num_experts + $tid) as usize) };
            let mut p = 0;
            while p < $pairs {
                if id_of(p) == $tid as i32 {
                    unsafe { *$pair_list.add(w as usize) = p as i32 };
                    w += 1;
                }
                p += 1;
            }
        }
    }};
}

#[cuda_module]
pub mod kernels {
    use cuda_device::warp::shuffle_xor_f32_sync;
    use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

    const WARPS: u32 = 4;
    const FULL: u32 = 0xffff_ffff;
    const FP8_MAX: f32 = 448.0;

    #[inline(always)]
    fn f32_to_bf16(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    #[inline(always)]
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    /// IEEE round-to-nearest division (vLLM's CUDA quant kernels use `/`).
    #[inline(always)]
    fn div_rn(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("div.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
        }
        r
    }

    /// Two f32 -> two e4m3 bytes (RNE; inputs pre-clamped to +-448):
    /// low byte = lo, high byte = hi.
    #[inline(always)]
    fn e4m3x2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe {
            ptx_asm!(
                "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t; }",
                out("=r") r,
                in("f") hi,
                in("f") lo,
                options(register_only),
            );
        }
        r & 0xffff
    }

    #[inline(always)]
    fn ex2(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    /// silu(x) = x / (1 + 2^(-x log2 e)); ex2.approx rel err ~2^-22.
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + ex2(-x * 1.442_695))
    }

    /// Smallest power of two >= s (s normal, > 0): exact
    /// exp2(ceil(log2(s))), vLLM's UE8M0 activation scale.
    #[inline(always)]
    fn pow2_ceil(s: f32) -> f32 {
        let b = s.to_bits();
        if b & 0x7f_ffff == 0 { s } else { f32::from_bits((b & 0x7f80_0000) + 0x80_0000) }
    }

    #[inline(always)]
    fn clamp8(x: f32) -> f32 {
        if x > FP8_MAX {
            FP8_MAX
        } else if x < -FP8_MAX {
            -FP8_MAX
        } else {
            x
        }
    }

    /// 16-byte read-only global load (weights).
    #[inline(always)]
    unsafe fn ld_nc_v4(p: *const u8) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        unsafe {
            ptx_asm!(
                "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];",
                out("=r") a,
                out("=r") b,
                out("=r") c,
                out("=r") d,
                in("l") p as u64,
            );
        }
        [a, b, c, d]
    }

    /// 16-byte global load (activations written by the previous launch).
    #[inline(always)]
    unsafe fn ld_v4(p: *const u8) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        unsafe {
            ptx_asm!(
                "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
                out("=r") a,
                out("=r") b,
                out("=r") c,
                out("=r") d,
                in("l") p as u64,
            );
        }
        [a, b, c, d]
    }

    /// 32 bytes [off, off + 32) as 8 words (two 16 B loads).
    #[inline(always)]
    fn ld32b(p: *const u8, off: usize, nc: bool) -> [u32; 8] {
        let (x, y) = unsafe {
            if nc {
                (ld_nc_v4(p.add(off)), ld_nc_v4(p.add(off + 16)))
            } else {
                (ld_v4(p.add(off)), ld_v4(p.add(off + 16)))
            }
        };
        [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
    }

    #[inline(always)]
    fn mma(c: [f32; 4], a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
                out("=f") d0,
                out("=f") d1,
                out("=f") d2,
                out("=f") d3,
                in("r") a0,
                in("r") a1,
                in("r") a2,
                in("r") a3,
                in("r") b0,
                in("r") b1,
                in("f") c[0],
                in("f") c[1],
                in("f") c[2],
                in("f") c[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    /// One 128-K block: rows (g, g+8) words `r0`/`r1`, weight words `b`.
    #[inline(always)]
    fn block_dot(r0: &[u32; 8], r1: &[u32; 8], b: &[u32; 8]) -> [f32; 4] {
        let mut c = [0.0f32; 4];
        let mut i = 0;
        while i < 4 {
            c = mma(c, r0[2 * i], r1[2 * i], r0[2 * i + 1], r1[2 * i + 1], b[2 * i], b[2 * i + 1]);
            i += 1;
        }
        c
    }

    /// acc += blk * a_scale(row) * w_scale  (Triton order: dot * a_s * b_s).
    #[inline(always)]
    fn rescale(acc: &mut [f32; 4], blk: [f32; 4], sa0: f32, sa1: f32, sw: f32) {
        acc[0] += blk[0] * sa0 * sw;
        acc[1] += blk[1] * sa0 * sw;
        acc[2] += blk[2] * sa1 * sw;
        acc[3] += blk[3] * sa1 * sw;
    }

    /// Local expert id of pair p: topk_ids[p] - id_base (vLLM's linear
    /// expert_map; outside [0, E) = another rank's expert or -1).
    #[inline(always)]
    fn local_id(topk_ids: *const i32, ids_i64: u32, id_base: u32, p: u32) -> i32 {
        let raw = if ids_i64 != 0 {
            unsafe { *(topk_ids as *const i64).add(p as usize) as i32 }
        } else {
            unsafe { *topk_ids.add(p as usize) }
        };
        raw.wrapping_sub(id_base as i32)
    }

    /// vLLM per-token-group FP8 quant of one 128-col group held by a warp,
    /// 4 columns per lane (lane l owns cols 4l..4l+4; csrc
    /// per_token_group_quant.cu / fused_silu_mul_block_quant.cu):
    ///   amax = max(eps, max|x|); s = max(amax / 448, smin);
    ///   ue8m0: s = 2^ceil(log2(max(s, 1e-10)));  q = e4m3(clamp(x / s))
    /// -> (4 e4m3 bytes as one little-endian word, s). All 32 lanes must call
    /// it (butterfly max). Used by every FP8 activation quant (x, h, fused).
    #[inline(always)]
    fn quant_group(v: [f32; 4], eps: f32, smin: f32, ue8m0: u32) -> (u32, f32) {
        let mut amax = eps;
        let mut i = 0;
        while i < 4 {
            let a = if v[i] < 0.0 { -v[i] } else { v[i] };
            if a > amax {
                amax = a;
            }
            i += 1;
        }
        let mut m = 16;
        while m > 0 {
            let o = shuffle_xor_f32_sync(FULL, amax, m);
            if o > amax {
                amax = o;
            }
            m /= 2;
        }
        let mut s = div_rn(amax, FP8_MAX);
        if s < smin {
            s = smin;
        }
        if ue8m0 != 0 {
            s = pow2_ceil(if s < 1e-10 { 1e-10 } else { s });
        }
        let lo = e4m3x2(clamp8(div_rn(v[0], s)), clamp8(div_rn(v[1], s)));
        let hi = e4m3x2(clamp8(div_rn(v[2], s)), clamp8(div_rn(v[3], s)));
        (lo | (hi << 16), s)
    }

    /// fc1 gate/up products of one 16-row chunk for one warp (8 cols): rows
    /// (g, g+8) = tokens t0/t1 of aq, per 128-K block acc += dot * a_s[tok,
    /// kb] * w_s[e, row/128, kb]. 2-deep register pipeline: block kb+1's
    /// operands AND scales are in flight while block kb's 8 mma issue
    /// (latency-bound loop). ponytail: the last step re-loads block nkb-1
    /// (L1/L2 hit, no DRAM) instead of branching around the prefetch.
    /// -> (gate, up).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fc1_dot(
        lo: usize,
        a_row0: *const u8,
        a_row1: *const u8,
        ok0: bool,
        ok1: bool,
        t0: u32,
        t1: u32,
        a_s: *const f32,
        w_gate: *const u8,
        w_up: *const u8,
        s_gate: *const f32,
        s_up: *const f32,
        nkb: u32,
    ) -> ([f32; 4], [f32; 4]) {
        let la = |kb: u32, ok: bool, row: *const u8| -> [u32; 8] {
            if ok { ld32b(row, (kb * 128) as usize + lo, false) } else { [0; 8] }
        };
        let lw = |kb: u32, row: *const u8| -> [u32; 8] { ld32b(row, (kb * 128) as usize + lo, true) };
        let sa = |kb: u32, ok: bool, tok: u32| -> f32 {
            if ok { unsafe { *a_s.add((tok * nkb + kb) as usize) } } else { 0.0 }
        };
        let ls = |kb: u32| -> [f32; 4] {
            unsafe { [sa(kb, ok0, t0), sa(kb, ok1, t1), *s_gate.add(kb as usize), *s_up.add(kb as usize)] }
        };
        let mut cg = [0.0f32; 4];
        let mut cu = [0.0f32; 4];
        let (mut x0, mut x1, mut bg, mut bu, mut sc) =
            (la(0, ok0, a_row0), la(0, ok1, a_row1), lw(0, w_gate), lw(0, w_up), ls(0));
        let mut kb = 0;
        while kb < nkb {
            let nk = if kb + 1 < nkb { kb + 1 } else { kb };
            let (y0, y1, ng_, nu, ns) = (la(nk, ok0, a_row0), la(nk, ok1, a_row1), lw(nk, w_gate), lw(nk, w_up), ls(nk));
            rescale(&mut cg, block_dot(&x0, &x1, &bg), sc[0], sc[1], sc[2]);
            rescale(&mut cu, block_dot(&x0, &x1, &bu), sc[0], sc[1], sc[3]);
            (x0, x1, bg, bu, sc) = (y0, y1, ng_, nu, ns);
            kb += 1;
        }
        (cg, cu)
    }

    /// fc2 product of one 16-row chunk for one warp (8 cols of H): rows
    /// (g, g+8) = pairs p0/p1 of hq, same block rescale and pipeline as
    /// fc1_dot.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fc2_dot(
        lo: usize,
        a_row0: *const u8,
        a_row1: *const u8,
        ok0: bool,
        ok1: bool,
        p0: u32,
        p1: u32,
        h_s: *const f32,
        w_row: *const u8,
        s_row: *const f32,
        nkb: u32,
    ) -> [f32; 4] {
        let la = |kb: u32, ok: bool, row: *const u8| -> [u32; 8] {
            if ok { ld32b(row, (kb * 128) as usize + lo, false) } else { [0; 8] }
        };
        let sa = |kb: u32, ok: bool, p: u32| -> f32 {
            if ok { unsafe { *h_s.add((p * nkb + kb) as usize) } } else { 0.0 }
        };
        let ls = |kb: u32| -> [f32; 3] { [sa(kb, ok0, p0), sa(kb, ok1, p1), unsafe { *s_row.add(kb as usize) }] };
        let lw = |kb: u32| -> [u32; 8] { ld32b(w_row, (kb * 128) as usize + lo, true) };
        let mut c = [0.0f32; 4];
        let (mut x0, mut x1, mut b, mut sc) = (la(0, ok0, a_row0), la(0, ok1, a_row1), lw(0), ls(0));
        let mut kb = 0;
        while kb < nkb {
            let nk = if kb + 1 < nkb { kb + 1 } else { kb };
            let (y0, y1, nb, ns) = (la(nk, ok0, a_row0), la(nk, ok1, a_row1), lw(nk), ls(nk));
            rescale(&mut c, block_dot(&x0, &x1, &b), sc[0], sc[1], sc[2]);
            (x0, x1, b, sc) = (y0, y1, nb, ns);
            kb += 1;
        }
        c
    }

    /// Element i of a bf16 (x_f32 == 0) or f32 global row at `base`.
    #[inline(always)]
    fn load_row(x: *const u8, x_f32: u32, base: usize, i: usize) -> f32 {
        if x_f32 != 0 {
            unsafe { *(x as *const f32).add(base + i) }
        } else {
            bf16_to_f32(unsafe { *(x as *const u16).add(base + i) })
        }
    }

    /// [legacy launch 1] single CTA of 256 threads, dynamic smem 2*E i32:
    /// route_cta (identical to nvfp4_moe's).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_route(
        topk_ids: *const i32,
        ids_i64: u32,
        id_base: u32,
        pairs: u32,
        num_experts: u32,
        max_slots: u32,
        slot_expert: *mut i32,
        slot_off: *mut i32,
        slot_cnt: *mut i32,
        pair_list: *mut i32,
    ) {
        let sh: *mut i32 = DynamicSharedArray::<i32>::get();
        let tid = thread::threadIdx_x();
        route_cta!(
            tid,
            sh,
            topk_ids,
            ids_i64,
            id_base,
            pairs,
            num_experts,
            max_slots,
            slot_expert,
            slot_off,
            slot_cnt,
            pair_list
        );
    }

    /// [legacy launches 2, 4] one warp per (row, 128-col group), 4 columns
    /// per lane: quant_group. Row-major out q [rows, K], s f32 [rows,
    /// K/128]. Input bf16 (x_f32 == 0) or f32.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_quant_rows(
        x: *const u8,
        x_f32: u32,
        q: *mut u8,
        sc: *mut f32,
        rows: u32,
        k: u32,
        x_stride: u32,
        eps: f32,
        smin: f32,
        ue8m0: u32,
    ) {
        let lane = thread::threadIdx_x() % 32;
        let grp = thread::blockIdx_x() * WARPS + thread::threadIdx_x() / 32;
        let row = thread::blockIdx_y();
        let ng = k / 128;
        if grp >= ng || row >= rows {
            return; // warp-uniform
        }
        let base = (row * x_stride + grp * 128 + 4 * lane) as usize;
        let ld = |i: usize| load_row(x, x_f32, base, i);
        let (word, s) = quant_group([ld(0), ld(1), ld(2), ld(3)], eps, smin, ue8m0);
        unsafe { *(q.add((row * k + grp * 128 + 4 * lane) as usize) as *mut u32) = word };
        if lane == 0 {
            unsafe { *sc.add((row * ng + grp) as usize) = s };
        }
    }

    /// [fused launch 1] route + quant x in ONE launch: block 0 = route_cta
    /// (dynamic smem 2*E i32), blocks 1.. = 8 warps each quantizing one
    /// (token, 128-col group) of x (quant_group, eps 1e-10, no floor).
    /// Independent (quant reads only x). grid 1 + ceil(M*H/128 / 8), 256
    /// threads.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_route_quant(
        topk_ids: *const i32,
        ids_i64: u32,
        id_base: u32,
        pairs: u32,
        num_experts: u32,
        max_slots: u32,
        slot_expert: *mut i32,
        slot_off: *mut i32,
        slot_cnt: *mut i32,
        pair_list: *mut i32,
        x: *const u8,
        aq: *mut u8,
        a_s: *mut f32,
        m: u32,
        hdim: u32,
        x_stride: u32,
        eps: f32,
        smin: f32,
        ue8m0: u32,
    ) {
        let sh: *mut i32 = DynamicSharedArray::<i32>::get();
        let tid = thread::threadIdx_x();
        let b = thread::blockIdx_x();
        if b == 0 {
            route_cta!(
                tid,
                sh,
                topk_ids,
                ids_i64,
                id_base,
                pairs,
                num_experts,
                max_slots,
                slot_expert,
                slot_off,
                slot_cnt,
                pair_list
            );
            return; // CTA-uniform
        }
        let lane = tid % 32;
        let ng = hdim / 128;
        let w = (b - 1) * 8 + tid / 32;
        let row = w / ng;
        let grp = w % ng;
        if row >= m {
            return; // warp-uniform
        }
        let base = (row * x_stride + grp * 128 + 4 * lane) as usize;
        let ld = |i: usize| load_row(x, 0, base, i);
        let (word, s) = quant_group([ld(0), ld(1), ld(2), ld(3)], eps, smin, ue8m0);
        unsafe { *(aq.add((row * hdim + grp * 128 + 4 * lane) as usize) as *mut u32) = word };
        if lane == 0 {
            unsafe { *a_s.add((row * ng + grp) as usize) = s };
        }
    }

    /// [legacy launch 3] fc1 + silu act_and_mul. grid (I/32, slots), 4
    /// warps x 8 columns of I. inter[p, j] = silu(gate_j) * up_j (f32 [P,
    /// I]); gate = w13 row j, up = w13 row I + j.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_fc1(
        inter: *mut f32,
        aq: *const u8,
        a_s: *const f32,
        w13: *const u8,
        w13_s: *const f32,
        slot_expert: *const i32,
        slot_off: *const i32,
        slot_cnt: *const i32,
        pair_list: *const i32,
        topk: u32,
        hdim: u32,
        idim: u32,
    ) {
        let slot = thread::blockIdx_y() as usize;
        let e = unsafe { *slot_expert.add(slot) };
        if e < 0 {
            return; // CTA-uniform
        }
        let e = e as usize;
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
        let nkb = hdim / 128;
        let w_e = unsafe { w13.add(e * (2 * idim * hdim) as usize) };
        let w_gate = unsafe { w_e.add(((j0 + g) * hdim) as usize) };
        let w_up = unsafe { w_e.add(((idim + j0 + g) * hdim) as usize) };
        let s_e = unsafe { w13_s.add(e * ((2 * idim / 128) * nkb) as usize) };
        let s_gate = unsafe { s_e.add(((j0 / 128) * nkb) as usize) };
        let s_up = unsafe { s_e.add((((idim + j0) / 128) * nkb) as usize) };
        let lo = (32 * t) as usize;
        let mut r0 = 0;
        while r0 < cnt {
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let t0 = p0 / topk;
            let t1 = p1 / topk;
            let a_row0 = unsafe { aq.add((t0 * hdim) as usize) };
            let a_row1 = unsafe { aq.add((t1 * hdim) as usize) };
            let (cg, cu) = fc1_dot(lo, a_row0, a_row1, ok0, ok1, t0, t1, a_s, w_gate, w_up, s_gate, s_up, nkb);
            let col = (j0 + 2 * t) as usize;
            if ok0 {
                let dst = unsafe { inter.add(p0 as usize * idim as usize + col) };
                unsafe {
                    *dst = silu(cg[0]) * cu[0];
                    *dst.add(1) = silu(cg[1]) * cu[1];
                }
            }
            if ok1 {
                let dst = unsafe { inter.add(p1 as usize * idim as usize + col) };
                unsafe {
                    *dst = silu(cg[2]) * cu[2];
                    *dst.add(1) = silu(cg[3]) * cu[3];
                }
            }
            r0 += 16;
        }
    }

    /// [fused launch 2] fc1 + silu act_and_mul + FP8 group quant of h.
    /// The h quant group is 128 columns, so this CTA owns a whole 128-col
    /// tile of I: grid (I/128, slots), 16 warps x 8 columns (512 threads),
    /// each warp running moe_fc1's exact per-warp code (same 16 B weight
    /// loads, same fc1_dot; the grid holds the same warps as moe_fc1's, 4x
    /// fewer / 4x larger CTAs; launch_bounds(512) caps regs at 128/thread).
    /// Chosen over a cross-CTA amax (impossible without a grid sync) and
    /// over 4 n-tiles per warp (4x the per-warp serial chain, ~2x regs).
    /// Per 16-row chunk the warps park h (the exact f32 value moe_fc1
    /// stores to `inter`) in a 16x128 f32 smem tile (dynamic smem 8 KB);
    /// after a barrier warp w quantizes chunk row w exactly like
    /// moe_quant_rows (lane = 4 cols, quant_group) and writes its pair's
    /// hq / h_s row: no `inter` round trip, no quant launch, bit-identical.
    /// Host guarantees I % 128 == 0 (no idle warp; all reach the barriers).
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn moe_fc1_quant(
        hq: *mut u8,
        h_s: *mut f32,
        aq: *const u8,
        a_s: *const f32,
        w13: *const u8,
        w13_s: *const f32,
        slot_expert: *const i32,
        slot_off: *const i32,
        slot_cnt: *const i32,
        pair_list: *const i32,
        topk: u32,
        hdim: u32,
        idim: u32,
        eps: f32,
        smin: f32,
        ue8m0: u32,
    ) {
        let tile: *mut f32 = DynamicSharedArray::<f32>::get();
        let slot = thread::blockIdx_y() as usize;
        let e = unsafe { *slot_expert.add(slot) };
        if e < 0 {
            return; // CTA-uniform
        }
        let e = e as usize;
        let off = unsafe { *slot_off.add(slot) } as u32;
        let cnt = unsafe { *slot_cnt.add(slot) } as u32;
        let tid = thread::threadIdx_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let grp = thread::blockIdx_x();
        let j0 = grp * 128 + warp * 8;
        let nkb = hdim / 128;
        let w_e = unsafe { w13.add(e * (2 * idim * hdim) as usize) };
        let w_gate = unsafe { w_e.add(((j0 + g) * hdim) as usize) };
        let w_up = unsafe { w_e.add(((idim + j0 + g) * hdim) as usize) };
        let s_e = unsafe { w13_s.add(e * ((2 * idim / 128) * nkb) as usize) };
        let s_gate = unsafe { s_e.add(((j0 / 128) * nkb) as usize) };
        let s_up = unsafe { s_e.add((((idim + j0) / 128) * nkb) as usize) };
        let lo = (32 * t) as usize;
        let tcol = (warp * 8 + 2 * t) as usize;
        let ng = idim / 128;
        let mut r0 = 0;
        while r0 < cnt {
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let t0 = p0 / topk;
            let t1 = p1 / topk;
            let a_row0 = unsafe { aq.add((t0 * hdim) as usize) };
            let a_row1 = unsafe { aq.add((t1 * hdim) as usize) };
            let (cg, cu) = fc1_dot(lo, a_row0, a_row1, ok0, ok1, t0, t1, a_s, w_gate, w_up, s_gate, s_up, nkb);
            // masked rows hold finite garbage (0-fragments); never quantized out
            unsafe {
                *tile.add(g as usize * 128 + tcol) = silu(cg[0]) * cu[0];
                *tile.add(g as usize * 128 + tcol + 1) = silu(cg[1]) * cu[1];
                *tile.add((g + 8) as usize * 128 + tcol) = silu(cg[2]) * cu[2];
                *tile.add((g + 8) as usize * 128 + tcol + 1) = silu(cg[3]) * cu[3];
            }
            thread::sync_threads();
            if r0 + warp < cnt {
                // warp-uniform: chunk row `warp`
                let p = (unsafe { *pair_list.add((off + r0 + warp) as usize) }) as u32;
                let b = (warp * 128 + 4 * lane) as usize;
                let v = unsafe { [*tile.add(b), *tile.add(b + 1), *tile.add(b + 2), *tile.add(b + 3)] };
                let (word, s) = quant_group(v, eps, smin, ue8m0);
                unsafe { *(hq.add((p * idim + grp * 128 + 4 * lane) as usize) as *mut u32) = word };
                if lane == 0 {
                    unsafe { *h_s.add((p * ng + grp) as usize) = s };
                }
            }
            thread::sync_threads(); // tile reused by the next chunk
            r0 += 16;
        }
    }

    /// [legacy launch 5, fused path for M > FUSED_MAX_M] fc2. grid (H/32,
    /// slots). y[p, h] = topk_w[p] * sum_kb dot_kb(hq[p], w2[e, h]) *
    /// h_s[p, kb] * w2_s[e, h/128, kb].
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_fc2(
        y: *mut f32,
        hq: *const u8,
        h_s: *const f32,
        w2: *const u8,
        w2_s: *const f32,
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
        let e = e as usize;
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
        let nkb = idim / 128;
        let w_row = unsafe { w2.add(e * (hdim * idim) as usize + ((h0 + g) * idim) as usize) };
        let s_row = unsafe { w2_s.add(e * ((hdim / 128) * nkb) as usize + ((h0 / 128) * nkb) as usize) };
        let lo = (32 * t) as usize;
        let mut r0 = 0;
        while r0 < cnt {
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let a_row0 = unsafe { hq.add((p0 * idim) as usize) };
            let a_row1 = unsafe { hq.add((p1 * idim) as usize) };
            let c = fc2_dot(lo, a_row0, a_row1, ok0, ok1, p0, p1, h_s, w_row, s_row, nkb);
            let col = (h0 + 2 * t) as usize;
            if ok0 {
                let s = unsafe { *topk_w.add(p0 as usize) };
                let dst = unsafe { y.add(p0 as usize * hdim as usize + col) };
                unsafe {
                    *dst = c[0] * s;
                    *dst.add(1) = c[1] * s;
                }
            }
            if ok1 {
                let s = unsafe { *topk_w.add(p1 as usize) };
                let dst = unsafe { y.add(p1 as usize * hdim as usize + col) };
                unsafe {
                    *dst = c[2] * s;
                    *dst.add(1) = c[3] * s;
                }
            }
            r0 += 16;
        }
    }

    /// [legacy launch 6, fused path for M > FUSED_MAX_M] out[t, h] =
    /// bf16(sum_k y[t*topk + k, h]) over local expert ids, k ascending
    /// (deterministic); 0 for a token with no local expert. One thread per
    /// (t, h). = nvfp4_moe's.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_combine(
        out: *mut u16,
        y: *const f32,
        topk_ids: *const i32,
        ids_i64: u32,
        id_base: u32,
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
            let id = local_id(topk_ids, ids_i64, id_base, p);
            if id >= 0 && (id as u32) < num_experts {
                acc += unsafe { *y.add((p * hdim + h) as usize) };
            }
            k += 1;
        }
        unsafe { *out.add((tok * out_stride + h) as usize) = f32_to_bf16(acc) as u16 };
    }

    /// [fused launch 3, M <= FUSED_MAX_M] fc2 + combine, TOKEN-MAJOR and
    /// deterministic (the nvfp4_moe design): grid (H/8, M), one CTA per
    /// (token, 8 cols of H), one warp per top-k slot (k = warp, warp + nw,
    /// ..; host launches nw = min(topk, 16) warps). A local pair's warp runs
    /// moe_fc2's fc2_dot with the pair alone in mma row 0 (lanes g == 0 load
    /// it; an mma output element depends only on its own A row) and parks
    /// y = c * topk_w[p] (moe_fc2's f32 value) in smem ys[k][8] (dynamic
    /// smem topk * 32 B); after one barrier 8 threads sum ys over k
    /// ASCENDING for local k (moe_combine's loop) and store bf16 once. A
    /// token with no local expert -> exact 0.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn moe_fc2_combine(
        out: *mut u16,
        hq: *const u8,
        h_s: *const f32,
        w2: *const u8,
        w2_s: *const f32,
        topk_w: *const f32,
        topk_ids: *const i32,
        ids_i64: u32,
        id_base: u32,
        topk: u32,
        hdim: u32,
        idim: u32,
        num_experts: u32,
        out_stride: u32,
    ) {
        let ys: *mut f32 = DynamicSharedArray::<f32>::get();
        let tok = thread::blockIdx_y();
        let h0 = thread::blockIdx_x() * 8;
        let tid = thread::threadIdx_x();
        let nw = thread::blockDim_x() / 32;
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let nkb = idim / 128;
        let lo = (32 * t) as usize;
        let ok0 = g < 1; // moe_fc2's mask for a 1-row chunk
        let mut k = warp;
        while k < topk {
            let p = tok * topk + k;
            let id = local_id(topk_ids, ids_i64, id_base, p);
            if id >= 0 && (id as u32) < num_experts {
                // warp-uniform
                let e = id as usize;
                let w_row = unsafe { w2.add(e * (hdim * idim) as usize + ((h0 + g) * idim) as usize) };
                let s_row = unsafe { w2_s.add(e * ((hdim / 128) * nkb) as usize + ((h0 / 128) * nkb) as usize) };
                let a_row = unsafe { hq.add((p * idim) as usize) };
                let c = fc2_dot(lo, a_row, a_row, ok0, false, p, p, h_s, w_row, s_row, nkb);
                if ok0 {
                    let s = unsafe { *topk_w.add(p as usize) };
                    unsafe {
                        *ys.add((k * 8 + 2 * t) as usize) = c[0] * s;
                        *ys.add((k * 8 + 2 * t + 1) as usize) = c[1] * s;
                    }
                }
            }
            k += nw;
        }
        thread::sync_threads();
        if tid < 8 {
            let mut acc = 0.0f32;
            let mut k = 0;
            while k < topk {
                let id = local_id(topk_ids, ids_i64, id_base, tok * topk + k);
                if id >= 0 && (id as u32) < num_experts {
                    acc += unsafe { *ys.add((k * 8 + tid) as usize) };
                }
                k += 1;
            }
            unsafe { *out.add((tok * out_stride + h0 + tid) as usize) = f32_to_bf16(acc) as u16 };
        }
    }
}

fn main() {
    // Build-only crate: scripts/oxide_build.py -> PTX (.target sm_120a) ->
    // ptxas 13.0 -> sm_120a cubin; the plugin launches it (src/fp8_moe_oxide.rs).
}
