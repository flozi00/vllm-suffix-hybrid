// SPDX-License-Identifier: Apache-2.0
//! NVFP4 W4A4 MoE decode kernels (gemma-4-26b-a4b-nvfp4 routed experts):
//! expert-grouped m16 block-scaled FP4 mma (sm_120a,
//! `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...ue4m3`).
//!
//! Replaces FlashInfer `cutlass_fused_moe` (128-row tiles) for decode-sized
//! batches, where each expert sees only ~1-16 routed rows. Stages:
//!   route       topk_ids -> active experts (ascending id) + per-slot row
//!               lists (ascending pair id): deterministic
//!   quant x     x bf16 [M, H] -> NVFP4 (a1 global scale)
//!   fc1         per (slot, 32-col tile of I): up + gate tiles share the A
//!               fragments; h = act(alpha*gate) * alpha*up, f32
//!   quant h     h f32 [P, I] -> NVFP4 (a2 global scale)
//!   fc2         y[p] = g2_alpha * w_p * acc
//!   combine     out[t] = sum_k y[t*topk + k] (fixed k order) -> bf16
//! Launch plans (host `mode`, all grids static in M -> CUDA-graph
//! capturable, all bit-identical: same f32 ops per output element):
//!   legacy 6    moe_route, moe_quant_rows, moe_fc1, moe_quant_rows,
//!               moe_fc2, moe_combine
//!   front 4     moe_route_quant, moe_fc1_quant, moe_fc2, moe_combine
//!   fused 3     moe_route_quant, moe_fc1_quant, moe_fc2_combine (decode M)
//! Decode is launch/latency bound (weights of ~5-10 experts per token), so
//! each launch removed is the win; see the fused kernels' docs.
//! Mid-M tunables (host `tune` word, all bit-identical: they change only
//! WHEN bytes move, never which bytes or which f32 ops):
//!   la1 / la2   L2 prefetch lookahead of the fc1 / fc2 weight rows + their
//!               scale lines in k64 steps (0 = off, >= K/64 = the whole row
//!               up front): `prefetch.global.L2` per 128 B line, so DRAM sees
//!               whole lines per row well ahead of the 32 B-per-row mma
//!               loads instead of scattered 32 B sector misses
//!   pf2         fc1 CTAs prefetch their share of the expert's w2 (+ scales)
//!               into L2 after their k loop, for the fc2 launch
//! moe_route_quant routes with `route_par` (same outputs as `route_cta`).
//! Layouts = vLLM FLASHINFER_CUTLASS after process_weights_after_loading
//! (quantization/utils/flashinfer_fp4_moe.py:309-420): w13 uint8
//! [E, 2I, H/2] ordered [w3(up); w1(gate)]; w13_sf e4m3 per-expert 128x4
//! swizzled [E, round128(2I), round4(H/16)]; w2 [E, H, I/2]; w2_sf
//! [E, round128(H), round4(I/16)]; g1/g2 alphas f32 [E]; one activation
//! global scale per layer. Pair p = token * topk + k.

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

/// moe_route_quant's routing: the SAME outputs as route_cta, byte for byte
/// (slot_expert / slot_off / slot_cnt / pair_list; empty slots' slot_off
/// untouched), in O(P/256 + P/2) smem steps per thread instead of
/// route_cta's two O(P) walks over global ids plus a 256-step serial tid-0
/// scan. One CTA of RT = 256 threads (E <= 256). `sh` = dynamic smem: 16 i32
/// warp totals, then every pair's local id as u16 (0xFFFF = not on this
/// rank), padded to a multiple of 8 pairs (host: 64 + 2 * round8(P) B).
/// Thread e counts / walks expert e 8 ids (4 independent u32 smem loads)
/// per iteration; the offset and
/// slot prefix sums are warp shuffle scans + 8 warp totals (integer adds:
/// exact in any order, so deterministic).
macro_rules! route_par {
    ($tid:expr, $sh:expr, $topk_ids:expr, $ids_i64:expr, $id_base:expr, $pairs:expr,
     $num_experts:expr, $max_slots:expr, $slot_expert:expr, $slot_off:expr, $slot_cnt:expr,
     $pair_list:expr) => {{
        let tid: u32 = $tid;
        let ne: u32 = $num_experts;
        let np8 = ($pairs + 7) / 8 * 8;
        let ids = unsafe { $sh.add(16) } as *mut u16;
        let mut p = tid;
        while p < np8 {
            let v = if p < $pairs { local_id($topk_ids, $ids_i64, $id_base, p) } else { -1 };
            let u = if v >= 0 && (v as u32) < ne { v as u32 } else { 0xFFFF };
            unsafe { *ids.add(p as usize) = u as u16 };
            p += RT;
        }
        cuda_device::thread::sync_threads();
        let words = unsafe { $sh.add(16) } as *const u32;
        let nw = np8 / 2;
        let mut cnt = 0u32;
        if tid < ne {
            let mut w = 0;
            while w < nw {
                let x = unsafe { [*words.add(w as usize), *words.add(w as usize + 1),
                                  *words.add(w as usize + 2), *words.add(w as usize + 3)] };
                let mut j = 0;
                while j < 4 {
                    cnt += ((x[j] & 0xFFFF) == tid) as u32 + ((x[j] >> 16) == tid) as u32;
                    j += 1;
                }
                w += 4;
            }
        }
        let act = (cnt > 0) as u32;
        let lane = tid % 32;
        let warp = tid / 32;
        let (mut c, mut a) = (cnt, act);
        let mut d = 1u32;
        while d < 32 {
            let yc = cuda_device::warp::shuffle_up_sync(0xFFFF_FFFF, c, d);
            let ya = cuda_device::warp::shuffle_up_sync(0xFFFF_FFFF, a, d);
            if lane >= d {
                c += yc;
                a += ya;
            }
            d *= 2;
        }
        if lane == 31 {
            unsafe {
                *$sh.add(warp as usize) = c as i32;
                *$sh.add(8 + warp as usize) = a as i32;
            }
        }
        cuda_device::thread::sync_threads();
        let (mut bc, mut ba, mut na) = (0u32, 0u32, 0u32);
        let mut w = 0u32;
        while w < RT / 32 {
            let wc = unsafe { *$sh.add(w as usize) } as u32;
            let wa = unsafe { *$sh.add(8 + w as usize) } as u32;
            if w < warp {
                bc += wc;
                ba += wa;
            }
            na += wa;
            w += 1;
        }
        let off = bc + c - cnt; // exclusive prefix of the counts (route_cta's `off`)
        let slot = ba + a - act; // exclusive prefix of the active flags (route_cta's `s`)
        if tid < ne && cnt > 0 && slot < $max_slots {
            unsafe {
                *$slot_expert.add(slot as usize) = tid as i32;
                *$slot_off.add(slot as usize) = off as i32;
                *$slot_cnt.add(slot as usize) = cnt as i32;
            }
        }
        if tid < $max_slots && tid >= na {
            unsafe {
                *$slot_expert.add(tid as usize) = -1;
                *$slot_cnt.add(tid as usize) = 0;
            }
        }
        if tid < ne && cnt > 0 {
            let mut o = off;
            let mut w = 0;
            while w < nw {
                let x = unsafe { [*words.add(w as usize), *words.add(w as usize + 1),
                                  *words.add(w as usize + 2), *words.add(w as usize + 3)] };
                let mut j = 0;
                while j < 8 {
                    // id of pair 2w + j: low half = even pair (little endian u16s)
                    if (x[j / 2] >> (16 * (j % 2))) & 0xFFFF == tid {
                        unsafe { *$pair_list.add(o as usize) = (2 * w + j as u32) as i32 };
                        o += 1;
                    }
                    j += 1;
                }
                w += 4;
            }
        }
    }};
}

/// NVFP4 quant of ONE 16-element block, vLLM scaled_fp4_quant math:
/// sf = e4m3(amax16 * (g / 6)), q = e2m1_rne(x * (g / sf)) (IEEE div).
/// `|i| load` = element i (bf16/f32 global row, or the fused fc1's f32 smem
/// tile; expanded in place — a closure capturing an smem pointer does not
/// compile in cuda-oxide); 8 packed bytes (low nibble = even element) to
/// qo, the scale to sfo. Used by every NVFP4 activation quant (x, h, fused).
macro_rules! quant16 {
    (|$i:ident| $ld:expr, $gscale:expr, $qo:expr, $sfo:expr $(,)?) => {{
        let (gscale, qo, sfo): (f32, *mut u8, *mut u8) = ($gscale, $qo, $sfo);
        let mut amax = 0.0f32;
        let mut n = 0usize;
        while n < 16 {
            let v: f32 = {
                let $i = n;
                $ld
            };
            let a = if v < 0.0 { -v } else { v };
            if a > amax {
                amax = a;
            }
            n += 1;
        }
        let s = to_e4m3(amax * (gscale / 6.0));
        let s_f = e4m3_to_f32(s);
        let inv = if s_f == 0.0 { 0.0 } else { gscale / s_f };
        let mut j = 0usize;
        while j < 8 {
            let lo = to_e2m1({
                let $i = 2 * j;
                $ld
            } * inv);
            let hi = to_e2m1({
                let $i = 2 * j + 1;
                $ld
            } * inv);
            unsafe { *qo.add(j) = (lo | (hi << 4)) as u8 };
            j += 1;
        }
        unsafe { *sfo = s as u8 };
    }};
}

#[cuda_module]
pub mod kernels {
    use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

    const WARPS: u32 = 4;
    /// moe_route_quant block size (route_par's warp-total layout).
    const RT: u32 = 256;

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

    /// act_and_mul activation: 0 = silu, 1 = gelu_tanh (gemma-4).
    /// gelu_tanh = 0.5x(1 + tanh u) = x * sigmoid(2u), u = sqrt(2/pi)(x +
    /// 0.044715x^3), evaluated with ex2 (rel err ~2^-22). NOT tanh.approx.f32
    /// (rel err 2^-11): the intermediate is re-quantized to FP4 right after,
    /// and a 2^-11 perturbation flips enough e2m1 codes / e4m3 block scales
    /// to cost 3e-3..1.7e-2 end-to-end (CPU emulation, see nvfp4_moe.py).
    #[inline(always)]
    fn act(x: f32, kind: u32) -> f32 {
        let z = if kind == 1 { 2.302_208_2 * (x + 0.044_715 * x * x * x) } else { x * 1.442_695 };
        x / (1.0 + ex2(-z))
    }

    /// One k64 step of an A (routed rows) fragment: 4 regs + 4 row scales.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn load_a(
        k0: u32,
        t: u32,
        a_row0: *const u8,
        a_row1: *const u8,
        ok0: bool,
        ok1: bool,
        sfa_base: *const u8,
        oks: bool,
    ) -> [u32; 5] {
        let o = (k0 / 2 + 4 * t) as usize;
        [
            if ok0 { ld32(a_row0, o) } else { 0 },
            if ok1 { ld32(a_row1, o) } else { 0 },
            if ok0 { ld32(a_row0, o + 16) } else { 0 },
            if ok1 { ld32(a_row1, o + 16) } else { 0 },
            if oks { ld32(sfa_base, (k0 / 16) as usize) } else { 0 },
        ]
    }

    /// One k64 step of a B (weight row) fragment: 2 regs + swizzled scales.
    #[inline(always)]
    fn load_b(k0: u32, t: u32, w_row: *const u8, sf_e: *const u8, row: u32, kb_pad: u32) -> [u32; 3] {
        let o = (k0 / 2 + 4 * t) as usize;
        [ld32(w_row, o), ld32(w_row, o + 16), ld32(sf_e, sf_offset(row, k0 / 16, kb_pad))]
    }

    /// CUTLASS 128x4 swizzled scale byte offset (vLLM swizzle_blockscale).
    #[inline(always)]
    fn sf_offset(row: u32, kb: u32, kb_pad: u32) -> usize {
        let atom = (row / 128) * (kb_pad / 4) + kb / 4;
        (atom * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4) as usize
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn mma(c: [f32; 4], a: [u32; 5], b: [u32; 3]) -> [f32; 4] {
        let (b0, b1, sfa, sfb) = (b[0], b[1], a[4], b[2]);
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

    /// Element i of a bf16 (x_f32 == 0) or f32 global row at `base`.
    #[inline(always)]
    fn load_row(x: *const u8, x_f32: u32, base: usize, i: usize) -> f32 {
        if x_f32 != 0 {
            unsafe { *(x as *const f32).add(base + i) }
        } else {
            bf16_to_f32(unsafe { *(x as *const u16).add(base + i) })
        }
    }

    /// L2 prefetch of the line holding `p` (a hint: never faults, never
    /// changes data; the kernels only pass addresses inside their tensors).
    #[inline(always)]
    fn pf_l2(p: *const u8) {
        unsafe {
            ptx_asm!("prefetch.global.L2 [%0];", in("l") p as u64);
        }
    }

    /// Prefetch k64 step `sp` of one weight stream of a warp (8 rows, one
    /// per g): lanes t == `wt` the 128 B line of their row that step `sp`
    /// opens (or any line when `first`: the window's first step), lane `sl`
    /// (a g == 0 lane) the step's 128 B scale line — the 8 rows' swizzled
    /// scale words of one k64 step share one line (rows 8-aligned). Every
    /// line the mma loads touch is requested once, when its first 32 B
    /// span enters the lookahead window.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pf_step(sp: u32, first: bool, t: u32, lane: u32, wt: u32, sl: u32, w_row: *const u8,
               sf_e: *const u8, row: u32, kb_pad: u32) {
        if t == wt {
            let a = unsafe { w_row.add((32 * sp) as usize) };
            if first || (a as u64) % 128 == 0 {
                pf_l2(a);
            }
        }
        if lane == sl {
            pf_l2(unsafe { sf_e.add(sf_offset(row, 4 * sp, kb_pad)) });
        }
    }

    /// Lookahead prologue: steps [0, min(2 + la, steps)) of one stream
    /// (the dot loops prefetch step s + la at iteration s >= 2), the same
    /// lines pf_step would request, spread over the whole warp: row g's
    /// steps sp = t (mod 4) by lane (g, t), the scale lines (all 8 rows'
    /// words of a step, row0 = the warp's first row) sp = lane (mod 32).
    /// la = 0: off.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pf_head(la: u32, steps: u32, t: u32, lane: u32, w_row: *const u8, sf_e: *const u8,
               row0: u32, kb_pad: u32) {
        if la == 0 {
            return;
        }
        let n0 = if 2 + la < steps { 2 + la } else { steps };
        let mut sp = t;
        while sp < n0 {
            let a = unsafe { w_row.add((32 * sp) as usize) };
            if sp == 0 || (a as u64) % 128 == 0 {
                pf_l2(a);
            }
            sp += 4;
        }
        let mut sp = lane;
        while sp < n0 {
            pf_l2(unsafe { sf_e.add(sf_offset(row0, 4 * sp, kb_pad)) });
            sp += 32;
        }
    }

    /// pf2: CTA blockIdx.x of gridDim.x prefetches its contiguous share of
    /// expert e's w2 rows [H, I/2] and w2 scales [r128(H), r4(I/16)] into L2
    /// (all 128 B multiples), for the fc2 launch that follows.
    #[inline(always)]
    fn pf_w2(w2: *const u8, w2_sf: *const u8, e: u32, hdim: u32, idim: u32) {
        let nx = thread::gridDim_x();
        let bx = thread::blockIdx_x();
        let slab = hdim * (idim / 2);
        let sfslab = (hdim + 127) / 128 * 128 * ((idim / 16 + 3) / 4 * 4);
        let mut part = 0;
        while part < 2 {
            let (base, bytes) = if part == 0 {
                (unsafe { w2.add(e as usize * slab as usize) }, slab)
            } else {
                (unsafe { w2_sf.add(e as usize * sfslab as usize) }, sfslab)
            };
            let lines = bytes / 128;
            let chunk = (lines + nx - 1) / nx;
            let end = if (bx + 1) * chunk < lines { (bx + 1) * chunk } else { lines };
            let mut l = bx * chunk + thread::threadIdx_x();
            while l < end {
                pf_l2(unsafe { base.add((l * 128) as usize) });
                l += thread::blockDim_x();
            }
            part += 1;
        }
    }

    /// fc1 up/gate dot products of one 16-row chunk for one warp (8 cols):
    /// 3-stage register pipeline, steps s+1, s+2 in flight while step s's
    /// mma issue (the loop is latency-bound, not BW-bound). pf != 0: at
    /// iteration s also L2-prefetch step s + pf of both weight streams
    /// (pf_step; the caller ran pf_head). -> (up, gate).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fc1_dot(
        t: u32,
        a_row0: *const u8,
        a_row1: *const u8,
        ok0: bool,
        ok1: bool,
        sfa_base: *const u8,
        oks: bool,
        w_up: *const u8,
        w_gate: *const u8,
        sf_e: *const u8,
        up_row: u32,
        gate_row: u32,
        kb_pad: u32,
        hdim: u32,
        lane: u32,
        pf: u32,
    ) -> ([f32; 4], [f32; 4]) {
        let mut cu = [0.0f32; 4];
        let mut cg = [0.0f32; 4];
        let la = |k0: u32| load_a(k0, t, a_row0, a_row1, ok0, ok1, sfa_base, oks);
        let lu = |k0: u32| load_b(k0, t, w_up, sf_e, up_row, kb_pad);
        let lg = |k0: u32| load_b(k0, t, w_gate, sf_e, gate_row, kb_pad);
        let steps = hdim / 64;
        let (mut a0, mut u0, mut q0) = (la(0), lu(0), lg(0));
        let (mut a1, mut u1, mut q1) = if steps > 1 { (la(64), lu(64), lg(64)) } else { (a0, u0, q0) };
        let mut s = 2;
        while s < steps {
            let (a2, u2, q2) = (la(s * 64), lu(s * 64), lg(s * 64));
            if pf != 0 && s + pf < steps {
                pf_step(s + pf, false, t, lane, 0, 2, w_up, sf_e, up_row, kb_pad);
                pf_step(s + pf, false, t, lane, 1, 3, w_gate, sf_e, gate_row, kb_pad);
            }
            cu = mma(cu, a0, u0);
            cg = mma(cg, a0, q0);
            (a0, u0, q0) = (a1, u1, q1);
            (a1, u1, q1) = (a2, u2, q2);
            s += 1;
        }
        cu = mma(cu, a0, u0);
        cg = mma(cg, a0, q0);
        if steps > 1 {
            cu = mma(cu, a1, u1);
            cg = mma(cg, a1, q1);
        }
        (cu, cg)
    }

    /// fc2 dot product of one 16-row chunk for one warp (8 cols of H), same
    /// 3-stage pipeline and la prefetch as fc1_dot.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fc2_dot(
        t: u32,
        a_row0: *const u8,
        a_row1: *const u8,
        ok0: bool,
        ok1: bool,
        sfa_base: *const u8,
        oks: bool,
        w_row: *const u8,
        sf_e: *const u8,
        row: u32,
        kb_pad: u32,
        idim: u32,
        lane: u32,
        pf: u32,
    ) -> [f32; 4] {
        let mut c = [0.0f32; 4];
        let la = |k0: u32| load_a(k0, t, a_row0, a_row1, ok0, ok1, sfa_base, oks);
        let lb = |k0: u32| load_b(k0, t, w_row, sf_e, row, kb_pad);
        let steps = idim / 64;
        let (mut a0, mut b0) = (la(0), lb(0));
        let (mut a1, mut b1) = if steps > 1 { (la(64), lb(64)) } else { (a0, b0) };
        let mut s = 2;
        while s < steps {
            let (a2, b2) = (la(s * 64), lb(s * 64));
            if pf != 0 && s + pf < steps {
                pf_step(s + pf, false, t, lane, 0, 1, w_row, sf_e, row, kb_pad);
            }
            c = mma(c, a0, b0);
            (a0, b0) = (a1, b1);
            (a1, b1) = (a2, b2);
            s += 1;
        }
        c = mma(c, a0, b0);
        if steps > 1 {
            c = mma(c, a1, b1);
        }
        c
    }

    /// [legacy launch 1] single CTA of 256 threads, dynamic smem 2*E i32:
    /// route_cta.
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

    /// [legacy launches 2, 4] one thread per (row, 16-block): quant16,
    /// row-major out q [rows, K/2], sf [rows, K/16]. Input bf16 (x_f32 ==
    /// 0) or f32.
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
        quant16!(
            |i| load_row(x, x_f32, base, i),
            gscale,
            unsafe { q.add((row * (k / 2) + kb * 8) as usize) },
            unsafe { sf.add((row * nkb + kb) as usize) },
        );
    }

    /// [fused launch 1] route + quant x in ONE launch: block 0 = route_par
    /// (dynamic smem 64 + 2*round2(P) B), blocks 1.. = one thread per (token, 16-block)
    /// of x -> aq/asf (quant16, a1 gscale). Independent (quant reads only
    /// x), so no inter-block dependency. grid 1 + ceil(M*H/16 / 256), 256
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
        asf: *mut u8,
        m: u32,
        hdim: u32,
        x_stride: u32,
        gscale: f32,
    ) {
        let sh: *mut i32 = DynamicSharedArray::<i32>::get();
        let tid = thread::threadIdx_x();
        let b = thread::blockIdx_x();
        if b == 0 {
            route_par!(
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
        let nkb = hdim / 16;
        let i = (b - 1) * 256 + tid;
        let row = i / nkb;
        let kb = i % nkb;
        if row >= m {
            return;
        }
        let base = (row * x_stride + kb * 16) as usize;
        quant16!(
            |i| load_row(x, 0, base, i),
            gscale,
            unsafe { aq.add((row * (hdim / 2) + kb * 8) as usize) },
            unsafe { asf.add((row * nkb + kb) as usize) },
        );
    }

    /// [legacy launch 3] fc1 + act_and_mul. grid (I/32, slots), 4 warps x 8
    /// columns of I. inter[p, j] = act(g1[e] * gate_j) * g1[e] * up_j (f32
    /// [P, I]). la: weight-row prefetch lookahead (pf_head as soon as the
    /// expert is known, then the dot loop); pf2 != 0: pf_w2 after the loop.
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
        la: u32,
        pf2: u32,
        w2: *const u8,
        w2_sf: *const u8,
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
        pf_head(la, hdim / 64, t, lane, w_up, sf_e, j0, kb_pad);
        pf_head(la, hdim / 64, t, lane, w_gate, sf_e, idim + j0, kb_pad);
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
            let (cu, cg) =
                fc1_dot(t, a_row0, a_row1, ok0, ok1, sfa_base, oks, w_up, w_gate, sf_e, up_row, gate_row, kb_pad, hdim, lane, la);
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
        if pf2 != 0 {
            pf_w2(w2, w2_sf, e, hdim, idim);
        }
    }

    /// [fused launch 2] fc1 + act_and_mul + NVFP4 quant of h (a2 gscale).
    /// Same grid / warps / dot products as moe_fc1 (grid (I/32, slots), 4
    /// warps x 8 columns; host guarantees I % 32 == 0, so no warp idles and
    /// every thread reaches the barriers). The CTA's 32 columns are two whole
    /// 16-col NVFP4 blocks, so per 16-row chunk the warps park h (the exact
    /// f32 value moe_fc1 stores to `inter`) in a 16x32 f32 smem tile
    /// (dynamic smem 2 KB), then 32 threads = (row, block) run quant16 on it
    /// and write hq / hsf rows of their pair directly: no f32 `inter` round
    /// trip, no quant launch, bit-identical codes. la / pf2 as moe_fc1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn moe_fc1_quant(
        hq: *mut u8,
        hsf: *mut u8,
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
        a2_gscale: f32,
        la: u32,
        pf2: u32,
        w2: *const u8,
        w2_sf: *const u8,
    ) {
        let tile: *mut f32 = DynamicSharedArray::<f32>::get();
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
        let c0 = thread::blockIdx_x() * (WARPS * 8);
        let j0 = c0 + warp * 8;
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
        pf_head(la, hdim / 64, t, lane, w_up, sf_e, j0, kb_pad);
        pf_head(la, hdim / 64, t, lane, w_gate, sf_e, idim + j0, kb_pad);
        let sfa_row = 8 * (lane & 1) + lane / 4;
        let tcol = (warp * 8 + 2 * t) as usize;
        let mut r0 = 0;
        while r0 < cnt {
            let ok0 = r0 + g < cnt;
            let ok1 = r0 + g + 8 < cnt;
            let oks = r0 + sfa_row < cnt;
            let p0 = if ok0 { (unsafe { *pair_list.add((off + r0 + g) as usize) }) as u32 } else { 0 };
            let p1 = if ok1 { (unsafe { *pair_list.add((off + r0 + g + 8) as usize) }) as u32 } else { 0 };
            let ps = if oks { (unsafe { *pair_list.add((off + r0 + sfa_row) as usize) }) as u32 } else { 0 };
            let a_row0 = unsafe { aq.add(((p0 / topk) * kh) as usize) };
            let a_row1 = unsafe { aq.add(((p1 / topk) * kh) as usize) };
            let sfa_base = unsafe { asf.add(((ps / topk) * nkb) as usize) };
            let (cu, cg) =
                fc1_dot(t, a_row0, a_row1, ok0, ok1, sfa_base, oks, w_up, w_gate, sf_e, up_row, gate_row, kb_pad, hdim, lane, la);
            // masked rows hold finite garbage (0-fragments); never quantized out
            unsafe {
                *tile.add(g as usize * 32 + tcol) = act(alpha * cg[0], act_kind) * (alpha * cu[0]);
                *tile.add(g as usize * 32 + tcol + 1) = act(alpha * cg[1], act_kind) * (alpha * cu[1]);
                *tile.add((g + 8) as usize * 32 + tcol) = act(alpha * cg[2], act_kind) * (alpha * cu[2]);
                *tile.add((g + 8) as usize * 32 + tcol + 1) = act(alpha * cg[3], act_kind) * (alpha * cu[3]);
            }
            thread::sync_threads();
            let row = tid / 2;
            let blk = tid % 2;
            if tid < 32 && r0 + row < cnt {
                let p = (unsafe { *pair_list.add((off + r0 + row) as usize) }) as u32;
                let col = c0 + blk * 16;
                let base = (row * 32 + blk * 16) as usize;
                quant16!(
                    |i| unsafe { *tile.add(base + i) },
                    a2_gscale,
                    unsafe { hq.add((p * (idim / 2) + col / 2) as usize) },
                    unsafe { hsf.add((p * (idim / 16) + col / 16) as usize) },
                );
            }
            thread::sync_threads(); // tile reused by the next chunk
            r0 += 16;
        }
        if pf2 != 0 {
            pf_w2(w2, w2_sf, e, hdim, idim);
        }
    }

    /// [legacy launch 5, front launch 3] fc2. grid (H/32,
    /// slots). y[p, h] = g2[e] * topk_w[p] * (hq[p] . w2[e, h]). la: w2 row
    /// prefetch lookahead (as moe_fc1).
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
        la: u32,
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
        pf_head(la, idim / 64, t, lane, w_row, sf_e, h0, kb_pad);
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
            let c = fc2_dot(t, a_row0, a_row1, ok0, ok1, sfa_base, oks, w_row, sf_e, h0 + g, kb_pad, idim, lane,
                            la);
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

    /// [legacy launch 6, front launch 4] out[t, h] =
    /// bf16(sum_k y[t*topk + k, h]) over local expert ids, k ascending
    /// (deterministic); 0 for a token with no local expert. One thread per
    /// (t, h).
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

    /// [fused launch 3] fc2 + combine (la as moe_fc2), TOKEN-MAJOR and
    /// deterministic. grid (H/8, M): one CTA per (token, 8 cols of H), one
    /// warp per top-k slot (k = warp, warp + nw, ..; host launches nw =
    /// min(topk, 16) warps). A local pair's warp runs exactly moe_fc2's
    /// fragment walk with the pair alone in mma row 0 (lanes g == 0; an
    /// mma output element depends only on its own A row) and parks
    /// y = c * (g2[e] * topk_w[p]) — moe_fc2's f32 value — in smem
    /// ys[k][8] (dynamic smem topk * 32 B). After one barrier, 8 threads
    /// sum ys over k ASCENDING for local k (moe_combine's loop) and store
    /// bf16 once: no y workspace, no combine launch, bit-identical output.
    /// A token with no local expert sums nothing -> exact 0 (EP contract).
    /// Tokens sharing an expert each re-read its w2 rows (mostly L2 hits
    /// within one launch): fine at decode M, which is why the host keeps
    /// the expert-major moe_fc2 + moe_combine as the per-M alternative (TUNE).
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn moe_fc2_combine(
        out: *mut u16,
        hq: *const u8,
        hsf: *const u8,
        w2: *const u8,
        w2_sf: *const u8,
        g2_alpha: *const f32,
        topk_w: *const f32,
        topk_ids: *const i32,
        ids_i64: u32,
        id_base: u32,
        topk: u32,
        hdim: u32,
        idim: u32,
        num_experts: u32,
        out_stride: u32,
        la: u32,
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
        let kh = idim / 2;
        let nkb = idim / 16;
        let kb_pad = (nkb + 3) / 4 * 4;
        let rows_pad = (hdim + 127) / 128 * 128;
        let sfa_row = 8 * (lane & 1) + lane / 4;
        // moe_fc2's masks for a 1-row chunk (cnt = 1, r0 = 0)
        let ok0 = g < 1;
        let oks = sfa_row < 1;
        let mut k = warp;
        while k < topk {
            let p = tok * topk + k;
            let id = local_id(topk_ids, ids_i64, id_base, p);
            if id >= 0 && (id as u32) < num_experts {
                // warp-uniform
                let e = id as u32;
                let w_row = unsafe { w2.add(e as usize * (hdim * kh) as usize + ((h0 + g) * kh) as usize) };
                let sf_e = unsafe { w2_sf.add(e as usize * (rows_pad * kb_pad) as usize) };
                let a_row = unsafe { hq.add((p * kh) as usize) };
                let sfa_base = unsafe { hsf.add((p * nkb) as usize) };
                pf_head(la, idim / 64, t, lane, w_row, sf_e, h0, kb_pad);
                let c = fc2_dot(t, a_row, a_row, ok0, false, sfa_base, oks, w_row, sf_e, h0 + g, kb_pad, idim,
                                lane, la);
                if ok0 {
                    let s = unsafe { *g2_alpha.add(e as usize) } * unsafe { *topk_w.add(p as usize) };
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
    // ptxas 13.0 -> sm_120a cubin; the plugin launches it (src/nvfp4_moe_oxide.rs).
}
