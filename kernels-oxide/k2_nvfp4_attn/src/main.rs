// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4 v2 (performance layout) on the cuda-oxide track: split-KV paged
//! decode / spec-verify attention over vLLM's HND NVFP4 KV cache with f16
//! `mma.sync.m16n8k16` (sm_80+ ISA; PTX .target sm_120, ptxas 13.0 SASS).
//!
//! Plan: src/nvfp4_attn.rs. CPU transcription: own_attn_ref.kernel_twin.
//!
//! One CTA (8 warps) = one (request, kv head, q-token tile, KV split). Its M
//! rows are ALL q tokens x GQA heads of the tile packed token-major
//! (row = i*G + g): decode q_len 1 (G rows) and MTP verify q_len 9 (9G rows)
//! stream every KV byte ONCE per CTA and feed it to all rows, so verify costs
//! ~ decode instead of ~9x. Ragged batches: request b's q rows are
//! qo_indptr[b]..qo_indptr[b+1] (<= the planned q_max); q tiles past a
//! request's length exit at once, rows past it are masked and not stored.
//!
//! Per TN-token KV tile:
//!   * cp.async 16 B copies of the raw packed K/V data + block scales into a
//!     2-stage shared-memory ring (next tile in flight while this one runs;
//!     data rows padded by 16 B so fragment loads are bank-conflict-free);
//!   * S = Q K^T: warp w owns KV n-tile w % (TN/8); K fragments are
//!     dequantized in registers straight from the raw bytes (8 e2m1 nibbles
//!     -> 4 f16x2 via two `prmt` LUT lookups, x e4m3 scale as f16x2 — exact:
//!     every e2m1 x e4m3 product is an f16); the head dim is permuted so a
//!     lane's two k-steps come from ONE aligned u32; Q (f16, same
//!     permutation) comes from smem via ldmatrix.x4, shared by its m-tiles;
//!   * online softmax (exp2 domain, -inf safe) across warps via small row
//!     max/sum exchanges; P (f16) to smem;
//!   * O += P V: warp w owns a 64-column group of the head dim for its
//!     m-tiles, V fragments dequantized in registers (two tokens' nibbles
//!     interleaved with bit ops, V scales de-swizzled per token): 8 n-tiles
//!     per k-step from one u32 per token.
//! Accumulators live in fully unrolled fixed-size arrays (no local memory).
//!
//! e4m3 -> f16 is a bit shift that yields value * 2^-8 exactly (subnormals
//! included); the host folds 2^8 into qk_scale_log2 and v_scale.
use cuda_device::{DynamicSharedArray, cuda_module, kernel, launch_bounds, ptx_asm, thread};

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::async_copy::{
        cp_async_cg_16, cp_async_commit_group, cp_async_wait_all,
    };
    use cuda_device::convert::cvt_f16x2_f32;
    use cuda_device::prmt::prmt;
    use cuda_device::warp::shuffle_xor_f32_sync;
    use cuda_device::wmma::{ldmatrix_x4, mma_m16n8k16_f32_f16};

    const WARPS: u32 = 8;
    // Build variant (cargo feature, one cubin each — oxide-variants.json):
    // MTW = O m-tile slots per warp (accumulator registers: 32 * MTW),
    // MTS = S m-tile slots per warp. The host picks the smallest variant
    // that covers the plan; w1 is built with -maxrregcount=128 (2 CTAs/SM).
    #[cfg(feature = "w1")]
    const MTW: usize = 1;
    #[cfg(feature = "w1")]
    const MTS: usize = 2;
    #[cfg(feature = "w2")]
    const MTW: usize = 2;
    #[cfg(feature = "w2")]
    const MTS: usize = 3;
    #[cfg(not(any(feature = "w1", feature = "w2")))]
    const MTW: usize = 3;
    #[cfg(not(any(feature = "w1", feature = "w2")))]
    const MTS: usize = 3;
    const THREADS: u32 = 256;
    const FULL: u32 = 0xFFFF_FFFF;
    const NEG_INF: f32 = f32::NEG_INFINITY;
    const MIN_TILES_PER_SPLIT: u32 = 2;
    /// KV tokens a split should cover (fixed per-split cost: Q load, pipeline
    /// fill, partial-O write + merge read) — relaxed when fewer tokens per
    /// split are needed to keep one wave of CTAs busy.
    const MIN_SPLIT_TOKENS: u32 = 512;

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

    #[inline(always)]
    fn hmul2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe {
            ptx_asm!(
                "mul.rn.f16x2 %0, %1, %2;",
                out("=r") r,
                in("r") a,
                in("r") b,
                options(register_only)
            );
        }
        r
    }

    /// bf16 bits of an f32, round-to-nearest-even.
    #[inline(always)]
    fn bf16_bits(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    #[inline(always)]
    fn bf16_to_f32(b: u32) -> f32 {
        f32::from_bits(b << 16)
    }

    /// e4m3fn byte -> f16 bits of value * 2^-8 (exact incl. subnormals).
    #[inline(always)]
    fn e4m3_f16(b: u32) -> u32 {
        ((b & 0x7F) << 7) | ((b & 0x80) << 8)
    }

    /// 8 e2m1 nibbles (n0 = low nibble) -> 4 f16x2 words
    /// [(n0,n1), (n2,n3), (n4,n5), (n6,n7)], first element in the low half.
    #[inline(always)]
    fn nib8(w: u32) -> [u32; 4] {
        // f16 high bytes of |e2m1| = {0, .5, 1, 1.5, 2, 3, 4, 6}
        let mags = w & 0x7777_7777;
        let lb = prmt(0x3E3C_3800, 0x4644_4240, mags);
        let hb = prmt(0x3E3C_3800, 0x4644_4240, mags >> 16);
        let s = w & 0x8888_8888;
        [
            prmt(lb, 0, 0x1404) | ((s << 12) & 0x8000) | ((s << 24) & 0x8000_0000),
            prmt(lb, 0, 0x3424) | ((s << 4) & 0x8000) | ((s << 16) & 0x8000_0000),
            prmt(hb, 0, 0x1404) | ((s >> 4) & 0x8000) | ((s << 8) & 0x8000_0000),
            prmt(hb, 0, 0x3424) | ((s >> 12) & 0x8000) | (s & 0x8000_0000),
        ]
    }

    #[inline(always)]
    fn swizzle_scale_offset(t: u32, g: u32, s: u32) -> u32 {
        let grp = s / 4;
        ((t / 4) * 4 + g / grp) * s + (g % grp) * 4 + t % 4
    }

    /// O rescale factor of one row for this tile: exp2(m_old - m_new), m_new
    /// = max(m_old, every n-tile's row max) — the same value the stats thread
    /// commits for the next tile.
    #[inline(always)]
    fn row_alpha(m_run: *const f32, red_max: *const f32, row: u32, ntc: u32, m_rows: u32) -> f32 {
        let m_old = unsafe { *m_run.add(row as usize) };
        let mut m_new = m_old;
        let mut n = 0u32;
        while n < ntc {
            let v = unsafe { *red_max.add((n * m_rows + row) as usize) };
            if v > m_new {
                m_new = v;
            }
            n += 1;
        }
        let m_safe = if m_new == NEG_INF { 0.0 } else { m_new };
        ex2(m_old - m_safe)
    }

    /// Concurrent CTA slots: SMs (%nsmid) x CTAs per SM of this variant
    /// (the register-capped w1 cubin runs 2 per SM).
    #[inline(always)]
    fn sm_slots() -> u32 {
        let nsm: u32;
        unsafe {
            ptx_asm!("mov.u32 %0, %%nsmid;", out("=r") nsm, options(register_only));
        }
        nsm * if MTW == 1 { 2 } else { 1 }
    }

    /// 16-byte read-only global load (4 x u32).
    #[inline(always)]
    unsafe fn ld_nc_v4(p: *const u16) -> [u32; 4] {
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

    /// Two bf16 (low half first) -> f16x2 (RN).
    #[inline(always)]
    fn bf16x2_to_f16x2(w: u32) -> u32 {
        cvt_f16x2_f32(bf16_to_f32(w & 0xFFFF), bf16_to_f32(w >> 16))
    }

    /// Physical head-dim index of logical k position `kl` (16-wide k-steps;
    /// lane t of k-step pair p owns physical dims 32p + 8t .. 32p + 8t + 7).
    #[inline(always)]
    fn perm_k(kl: u32) -> u32 {
        let s = kl / 16;
        let c = kl % 16;
        (s / 2) * 32 + ((c % 8) / 2) * 8 + (s % 2) * 4 + (c / 8) * 2 + c % 2
    }

    /// cp.async one TN-token tile into a stage: [Kd TN*DH | Ks TN*SD |
    /// Vd TN*DH | Vs TN*SD] — four contiguous runs of the page (the V scale
    /// swizzle stays inside 4-token groups, TN is a multiple of 16).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn issue_tile(
        dst: *mut u8,
        k_data: *const u8,
        k_sf: *const u8,
        v_data: *const u8,
        v_sf: *const u8,
        page_off: u64,
        head_tok: u64,
        tn: u32,
        dh: u32,
        sd: u32,
        tid: u32,
    ) {
        unsafe {
            let nd = tn * dh / 16;
            let nsf = tn * sd / 16;
            // data rows land padded to dh + 16 bytes (bank-conflict-free
            // fragment loads); scale runs stay dense.
            let kp = dh + 16;
            let rc = dh / 16; // 16 B chunks per data row
            let mut c = tid;
            while c < 2 * (nd + nsf) {
                let (src, dst_off) = if c < nd {
                    (
                        k_data.add((page_off + head_tok * dh as u64 + (c * 16) as u64) as usize),
                        (c / rc) * kp + (c % rc) * 16,
                    )
                } else if c < nd + nsf {
                    let i = c - nd;
                    (
                        k_sf.add((page_off + head_tok * sd as u64 + (i * 16) as u64) as usize),
                        tn * kp + i * 16,
                    )
                } else if c < 2 * nd + nsf {
                    let i = c - nd - nsf;
                    (
                        v_data.add((page_off + head_tok * dh as u64 + (i * 16) as u64) as usize),
                        tn * (kp + sd) + (i / rc) * kp + (i % rc) * 16,
                    )
                } else {
                    let i = c - 2 * nd - nsf;
                    (
                        v_sf.add((page_off + head_tok * sd as u64 + (i * 16) as u64) as usize),
                        tn * (2 * kp + sd) + i * 16,
                    )
                };
                cp_async_cg_16(dst.add(dst_off as usize) as *mut u32, src as *const u32);
                c += THREADS;
            }
            cp_async_commit_group();
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn nvfp4_attn_partial(
        q: *const u16,
        k_data: *const u8,
        k_sf: *const u8,
        v_data: *const u8,
        v_sf: *const u8,
        block_table: *const i32,
        seq_lens: *const i32,
        qo_indptr: *const i32,
        o_part: *mut u16,
        lse_part: *mut f32,
        q_tok_stride: u32,
        page_bytes: u32,
        bt_stride: u32,
        hkv: u32,
        g_heads: u32,
        qt_n: u32,
        m_rows: u32,
        d: u32,
        nqt: u32,
        page_size: u32,
        ns: u32,
        tn: u32,
        window_left: i32,
        qk_scale_log2: f32,
    ) {
        let tid = thread::threadIdx_x();
        let w = tid / 32;
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
        let qrow = d + 8; // f16 elements per Q smem row
        let prow = tn + 8; // f16 elements per P smem row
        let ntc = tn / 8;
        let mtiles = m_rows / 16;
        let real_rows = qt_n * g_heads;
        let slot_base = (r * ns + s) * m_rows;

        // ---- this request's q rows; tiles past them have no work -----------
        let qs0 = unsafe { *qo_indptr.add(b as usize) };
        let qe = unsafe { *qo_indptr.add(b as usize + 1) };
        let q_len = if qe > qs0 { (qe - qs0) as u32 } else { 0 };
        let qs0 = qs0 as u32;
        if qt * qt_n >= q_len {
            if tid < m_rows {
                unsafe { *lse_part.add((slot_base + tid) as usize) = NEG_INF };
            }
            return;
        }

        // ---- this CTA's KV range [t0, t1) in TN-token tiles -----------------
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
        let lo_t = (lo as u32) / tn;
        let hi_t = (hi as u32).div_ceil(tn);
        let n_t = if hi_t > lo_t { hi_t - lo_t } else { 0 };
        // Tiles per split: the plan's ns, but never finer than
        // MIN_SPLIT_TOKENS unless that many splits are needed to fill one
        // wave (rows x splits >= SM slots); launched splits past the
        // request's need exit at once (-inf lse, skipped by the merge).
        let mut per = n_t.div_ceil(ns);
        if per < MIN_TILES_PER_SPLIT {
            per = MIN_TILES_PER_SPLIT;
        }
        let wave = (n_t * thread::gridDim_x()).div_ceil(sm_slots());
        let tok_floor = MIN_SPLIT_TOKENS.div_ceil(tn);
        let floor = if wave < tok_floor { wave } else { tok_floor };
        if per < floor {
            per = floor;
        }
        let t0 = lo_t + s * per;
        let mut t1 = t0 + per;
        if t1 > hi_t {
            t1 = hi_t;
        }
        if t0 >= t1 {
            // Empty split: weight 0 in the merge, its O is never read.
            if tid < m_rows {
                unsafe { *lse_part.add((slot_base + tid) as usize) = NEG_INF };
            }
            return;
        }

        // ---- shared memory carve-up (all offsets 16-byte aligned) -----------
        let qs: *mut u16 = DynamicSharedArray::<u16>::get();
        let kp = dh + 16; // padded smem row of packed K/V data
        let stage_bytes = tn * (kp + sd) * 2;
        let kv0: *mut u8 = unsafe { qs.add((m_rows * qrow) as usize) as *mut u8 };
        let ps: *mut u16 = unsafe { kv0.add((2 * stage_bytes) as usize) as *mut u16 };
        let red_max: *mut f32 = unsafe { ps.add((m_rows * prow) as usize) as *mut f32 };
        let red_sum: *mut f32 = unsafe { red_max.add((ntc * m_rows) as usize) };
        // running max double-buffered by tile parity (tile j reads m_runs[j%2]
        // and its stats thread writes m_runs[(j+1)%2]), so no barrier sits
        // between the stats update and the O phase.
        let m_run0: *mut f32 = unsafe { red_sum.add((ntc * m_rows) as usize) };
        let l_run: *mut f32 = unsafe { m_run0.add(m_rows as usize) };
        let m_run1: *mut f32 = unsafe { l_run.add(m_rows as usize) };

        let tile_src = |j: u32| -> (u64, u64) {
            let tok0 = j * tn;
            let pg = unsafe { *block_table.add((b * bt_stride + tok0 / page_size) as usize) } as u64;
            (pg * page_bytes as u64, (h * page_size + tok0 % page_size) as u64)
        };

        // ---- first tile in flight, then Q -> smem (f16, permuted k) ----------
        let (po, ht) = tile_src(t0);
        unsafe { issue_tile(kv0, k_data, k_sf, v_data, v_sf, po, ht, tn, dh, sd, tid) };
        // Q -> smem as f16 in logical (permuted) k order. 16-byte loads of 8
        // physical dims, QB chunks in flight per thread; physical offset
        // o = 8a + 4b + 2e + f inside a 32-dim block P sits at logical
        // k = 16(2P + b) + 8e + 2a + f (inverse of perm_k), so each (f=0,1)
        // pair stays adjacent: 4 u32 smem stores per chunk.
        const QB: u32 = 4;
        let nchunk = m_rows * d / 8;
        let mut base = 0u32;
        while base < nchunk {
            let mut w = [[0u32; 4]; QB as usize];
            let mut u = 0usize;
            #[unroll]
            while u < QB as usize {
                let c = base + tid + u as u32 * THREADS;
                if c < nchunk {
                    let row = c / (d / 8);
                    let i = row / g_heads;
                    let gg = row % g_heads;
                    let tok = qt * qt_n + i;
                    if row < real_rows && tok < q_len {
                        let off = (qs0 + tok) * q_tok_stride + (h * g_heads + gg) * d + (c % (d / 8)) * 8;
                        w[u] = unsafe { ld_nc_v4(q.add(off as usize)) };
                    }
                }
                u += 1;
            }
            let mut u = 0usize;
            #[unroll]
            while u < QB as usize {
                let c = base + tid + u as u32 * THREADS;
                if c < nchunk {
                    let row = c / (d / 8);
                    let ph = (c % (d / 8)) * 8; // physical dim of element 0
                    let blk = ph / 32;
                    let a = (ph % 32) / 8;
                    let mut k = 0usize;
                    #[unroll]
                    while k < 4 {
                        // element pair k: o = 8a + 2k (+f) -> b = k / 2, e = k % 2
                        let kl = 16 * (2 * blk + k as u32 / 2) + 8 * (k as u32 % 2) + 2 * a;
                        unsafe {
                            *(qs.add((row * qrow + kl) as usize) as *mut u32) = bf16x2_to_f16x2(w[u][k]);
                        }
                        k += 1;
                    }
                }
                u += 1;
            }
            base += QB * THREADS;
        }
        if tid < m_rows {
            unsafe {
                *m_run0.add(tid as usize) = NEG_INF;
                *l_run.add(tid as usize) = 0.0;
            }
        }

        // warp roles
        let nt = w % ntc; // S phase: KV n-tile
        let s_grp = w / ntc; // S phase: m-tile group
        let s_wps = WARPS / ntc;
        let jg = d / 64; // PV phase: 64-column groups
        let cj = w % jg;
        let o_grp = w / jg;
        let o_wpj = WARPS / jg;
        let arow = (lane % 8) + 8 * ((lane / 8) % 2); // ldmatrix row of this lane
        let acol = 8 * (lane / 16);
        let gblk = 4 * cj + gq / 2; // V scale block of this lane's dims

        let mut acc_o = [[[0.0f32; 4]; 8]; MTW];

        let mut j = t0;
        while j < t1 {
            let st = (j - t0) % 2;
            let m_run = if st == 0 { m_run0 } else { m_run1 };
            let m_next = if st == 0 { m_run1 } else { m_run0 };
            // tile j landed (the only group in flight); the barrier also
            // retires tile j-1's O phase, freeing stage st^1 for tile j+1.
            unsafe { cp_async_wait_all() };
            thread::sync_threads();
            if j + 1 < t1 {
                let (po, ht) = tile_src(j + 1);
                let dst = unsafe { kv0.add(((st ^ 1) * stage_bytes) as usize) };
                unsafe { issue_tile(dst, k_data, k_sf, v_data, v_sf, po, ht, tn, dh, sd, tid) };
            }
            let tok0 = j * tn;
            let kd = unsafe { kv0.add((st * stage_bytes) as usize) };
            let ksf = unsafe { kd.add((tn * kp) as usize) };
            let vd = unsafe { kd.add((tn * (kp + sd)) as usize) };
            let vsf = unsafe { kd.add((tn * (2 * kp + sd)) as usize) };

            // ---- S = Q K^T for (this warp's m-tiles) x (n-tile nt) ----------
            let mut sacc = [[0.0f32; 4]; MTS];
            let mut sacc2 = [[0.0f32; 4]; MTS]; // 2nd chain: halves mma dependency depth
            let krow = nt * 8 + gq;
            let mut p = 0u32;
            while p < d / 32 {
                let wk = unsafe { *(kd.add((krow * kp + p * 16 + 4 * t4) as usize) as *const u32) };
                let sc = e4m3_f16(unsafe { *ksf.add((krow * sd + 2 * p + t4 / 2) as usize) } as u32);
                let sc2 = sc | (sc << 16);
                let f = nib8(wk);
                let ba = [hmul2(f[0], sc2), hmul2(f[1], sc2)];
                let bb = [hmul2(f[2], sc2), hmul2(f[3], sc2)];
                let mut slot = 0usize;
                #[unroll]
                while slot < MTS {
                    let mt = s_grp + slot as u32 * s_wps;
                    if mt < mtiles {
                        let rowp = (mt * 16 + arow) * qrow + 32 * p + acol;
                        let aa = unsafe { ldmatrix_x4(qs.add(rowp as usize) as *const u32) };
                        let ab = unsafe { ldmatrix_x4(qs.add((rowp + 16) as usize) as *const u32) };
                        sacc[slot] = unsafe { mma_m16n8k16_f32_f16(sacc[slot], aa, ba) };
                        sacc2[slot] = unsafe { mma_m16n8k16_f32_f16(sacc2[slot], ab, bb) };
                    }
                    slot += 1;
                }
                p += 1;
            }

            let mut slot = 0usize;
            #[unroll]
            while slot < MTS {
                let mut e = 0usize;
                #[unroll]
                while e < 4 {
                    sacc[slot][e] += sacc2[slot][e];
                    e += 1;
                }
                slot += 1;
            }

            // ---- scale + mask + per-warp row max ------------------------------
            let kp0 = (tok0 + nt * 8 + 2 * t4) as i32;
            let mut slot = 0usize;
            #[unroll]
            while slot < MTS {
                let mt = s_grp + slot as u32 * s_wps;
                if mt < mtiles {
                    let mut half = 0usize;
                    #[unroll]
                    while half < 2 {
                        let row = mt * 16 + gq + 8 * half as u32;
                        let i = row / g_heads;
                        let valid = row < real_rows && qt * qt_n + i < q_len;
                        let qpos = q0 + i as i32;
                        let mut e = 0usize;
                        #[unroll]
                        while e < 2 {
                            let kp = kp0 + e as i32;
                            let mut ok = valid && kp <= qpos && kp < kv_len;
                            if window_left >= 0 && kp + window_left < qpos {
                                ok = false;
                            }
                            let v = sacc[slot][2 * half + e];
                            sacc[slot][2 * half + e] = if ok { v * qk_scale_log2 } else { NEG_INF };
                            e += 1;
                        }
                        let a = sacc[slot][2 * half];
                        let c = sacc[slot][2 * half + 1];
                        let mut mx = if a > c { a } else { c };
                        let o1 = shuffle_xor_f32_sync(FULL, mx, 1);
                        if o1 > mx {
                            mx = o1;
                        }
                        let o2 = shuffle_xor_f32_sync(FULL, mx, 2);
                        if o2 > mx {
                            mx = o2;
                        }
                        if t4 == 0 {
                            unsafe { *red_max.add((nt * m_rows + row) as usize) = mx };
                        }
                        half += 1;
                    }
                }
                slot += 1;
            }
            thread::sync_threads();

            // ---- P = exp2(S - m_new) -> smem (f16), per-warp row sums ----------
            let mut slot = 0usize;
            #[unroll]
            while slot < MTS {
                let mt = s_grp + slot as u32 * s_wps;
                if mt < mtiles {
                    let mut half = 0usize;
                    #[unroll]
                    while half < 2 {
                        let row = mt * 16 + gq + 8 * half as u32;
                        let mut m_new = unsafe { *m_run.add(row as usize) };
                        let mut n = 0u32;
                        while n < ntc {
                            let v = unsafe { *red_max.add((n * m_rows + row) as usize) };
                            if v > m_new {
                                m_new = v;
                            }
                            n += 1;
                        }
                        let m_safe = if m_new == NEG_INF { 0.0 } else { m_new };
                        let p0 = ex2(sacc[slot][2 * half] - m_safe);
                        let p1 = ex2(sacc[slot][2 * half + 1] - m_safe);
                        unsafe {
                            *(ps.add((row * prow + nt * 8 + 2 * t4) as usize) as *mut u32) =
                                cvt_f16x2_f32(p0, p1);
                        }
                        let mut sum = p0 + p1;
                        sum += shuffle_xor_f32_sync(FULL, sum, 1);
                        sum += shuffle_xor_f32_sync(FULL, sum, 2);
                        if t4 == 0 {
                            unsafe { *red_sum.add((nt * m_rows + row) as usize) = sum };
                        }
                        half += 1;
                    }
                }
                slot += 1;
            }
            thread::sync_threads();

            // ---- running stats: one thread per row ------------------------------
            if tid < m_rows {
                let row = tid;
                let m_old = unsafe { *m_run.add(row as usize) };
                let mut m_new = m_old;
                let mut psum = 0.0f32;
                let mut n = 0u32;
                while n < ntc {
                    let v = unsafe { *red_max.add((n * m_rows + row) as usize) };
                    if v > m_new {
                        m_new = v;
                    }
                    psum += unsafe { *red_sum.add((n * m_rows + row) as usize) };
                    n += 1;
                }
                let m_safe = if m_new == NEG_INF { 0.0 } else { m_new };
                let alpha = ex2(m_old - m_safe);
                unsafe {
                    *m_next.add(row as usize) = m_new;
                    let l = *l_run.add(row as usize);
                    *l_run.add(row as usize) = l * alpha + psum;
                }
            }

            // ---- O[:, 64-col group cj] = O * alpha + P V ----------------------
            let mut slot = 0usize;
            #[unroll]
            while slot < MTW {
                let mt = o_grp + slot as u32 * o_wpj;
                if mt < mtiles {
                    let a0 = row_alpha(m_run, red_max, mt * 16 + gq, ntc, m_rows);
                    let a8 = row_alpha(m_run, red_max, mt * 16 + gq + 8, ntc, m_rows);
                    let mut jn = 0usize;
                    #[unroll]
                    while jn < 8 {
                        acc_o[slot][jn][0] *= a0;
                        acc_o[slot][jn][1] *= a0;
                        acc_o[slot][jn][2] *= a8;
                        acc_o[slot][jn][3] *= a8;
                        jn += 1;
                    }
                }
                slot += 1;
            }
            let mut ks = 0u32;
            while ks < tn / 16 {
                // V fragments: tokens (2t, 2t+1) -> b0, (2t+8, 2t+9) -> b1; dims
                // 64cj + 8gq + 0..7 = one u32 per token; n-tile jn = dim offset.
                let mut b0 = [0u32; 8];
                let mut b1 = [0u32; 8];
                let mut pr = 0usize;
                #[unroll]
                while pr < 2 {
                    let ta = ks * 16 + 2 * t4 + 8 * pr as u32;
                    let tb = ta + 1;
                    let va = ((tok0 + ta) as i32) < kv_len;
                    let vb = ((tok0 + tb) as i32) < kv_len;
                    // Stale tail tokens may hold e4m3 NaN codes: zero data AND
                    // scale (P is 0 there, but 0 * NaN would poison O).
                    let wa = if va {
                        unsafe { *(vd.add((ta * kp + cj * 32 + 4 * gq) as usize) as *const u32) }
                    } else {
                        0
                    };
                    let wb = if vb {
                        unsafe { *(vd.add((tb * kp + cj * 32 + 4 * gq) as usize) as *const u32) }
                    } else {
                        0
                    };
                    let sa = if va {
                        e4m3_f16(unsafe { *vsf.add(swizzle_scale_offset(ta, gblk, sd) as usize) } as u32)
                    } else {
                        0
                    };
                    let sb = if vb {
                        e4m3_f16(unsafe { *vsf.add(swizzle_scale_offset(tb, gblk, sd) as usize) } as u32)
                    } else {
                        0
                    };
                    let sc2 = sa | (sb << 16);
                    // byte k of lo = (a dim 2k, b dim 2k); of hi = (a 2k+1, b 2k+1)
                    let lo = (wa & 0x0F0F_0F0F) | ((wb & 0x0F0F_0F0F) << 4);
                    let hi = ((wa >> 4) & 0x0F0F_0F0F) | (wb & 0xF0F0_F0F0);
                    let fl = nib8(lo);
                    let fh = nib8(hi);
                    let mut k = 0usize;
                    #[unroll]
                    while k < 4 {
                        let even = hmul2(fl[k], sc2);
                        let odd = hmul2(fh[k], sc2);
                        if pr == 0 {
                            b0[2 * k] = even;
                            b0[2 * k + 1] = odd;
                        } else {
                            b1[2 * k] = even;
                            b1[2 * k + 1] = odd;
                        }
                        k += 1;
                    }
                    pr += 1;
                }
                let pcol = ks * 16 + acol;
                let mut slot = 0usize;
                #[unroll]
                while slot < MTW {
                    let mt = o_grp + slot as u32 * o_wpj;
                    if mt < mtiles {
                        let a = unsafe {
                            ldmatrix_x4(
                                ps.add(((mt * 16 + arow) * prow + pcol) as usize) as *const u32,
                            )
                        };
                        let mut jn = 0usize;
                        #[unroll]
                        while jn < 8 {
                            acc_o[slot][jn] =
                                unsafe { mma_m16n8k16_f32_f16(acc_o[slot][jn], a, [b0[jn], b1[jn]]) };
                            jn += 1;
                        }
                    }
                    slot += 1;
                }
                ks += 1;
            }
            j += 1;
        }

        thread::sync_threads(); // last tile's stats (l_run, m) visible
        let m_fin = if (t1 - t0) % 2 == 0 { m_run0 } else { m_run1 };

        // ---- normalized partial O (bf16, 2 x 16 B per row) + lse --------------
        let mut slot = 0usize;
        #[unroll]
        while slot < MTW {
            let mt = o_grp + slot as u32 * o_wpj;
            if mt < mtiles {
                let mut half = 0usize;
                #[unroll]
                while half < 2 {
                    let row = mt * 16 + gq + 8 * half as u32;
                    let l = unsafe { *l_run.add(row as usize) };
                    let inv = if l == 0.0 { 0.0 } else { 1.0 / l };
                    let dst = unsafe {
                        o_part.add(((slot_base + row) * d + 64 * cj + 16 * t4) as usize) as *mut u32
                    };
                    let mut k = 0usize;
                    #[unroll]
                    while k < 4 {
                        // n-tile jn covers dims 64cj + 8c + jn; this lane's C
                        // cols 2t, 2t+1 -> dims 64cj + 16t + jn and + 8.
                        let e0 = acc_o[slot][2 * k][2 * half] * inv;
                        let e1 = acc_o[slot][2 * k + 1][2 * half] * inv;
                        let o0 = acc_o[slot][2 * k][2 * half + 1] * inv;
                        let o1 = acc_o[slot][2 * k + 1][2 * half + 1] * inv;
                        unsafe {
                            *dst.add(k) = bf16_bits(e0) | (bf16_bits(e1) << 16);
                            *dst.add(4 + k) = bf16_bits(o0) | (bf16_bits(o1) << 16);
                        }
                        k += 1;
                    }
                    half += 1;
                }
            }
            slot += 1;
        }
        if tid < m_rows {
            let m = unsafe { *m_fin.add(tid as usize) };
            let l = unsafe { *l_run.add(tid as usize) };
            let lse = if l == 0.0 { NEG_INF } else { m + lg2(l) };
            unsafe { *lse_part.add((slot_base + tid) as usize) = lse };
        }
    }

    /// Weighted sum over splits s = s0, s0+step, ... of one row's 8 dims.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn merge_item(
        o_part: *const u16,
        wts: *const f32,
        r: u32,
        ns: u32,
        m_rows: u32,
        d: u32,
        row: u32,
        col: u32,
        s0: u32,
        step: u32,
    ) -> [f32; 8] {
        unsafe {
            let mut acc = [0.0f32; 8];
            let mut s = s0;
            while s < ns {
                let wgt = *wts.add((s * m_rows + row) as usize);
                if wgt != 0.0 {
                    let src = o_part.add((((r * ns + s) * m_rows + row) * d + col) as usize) as *const u32;
                    let mut k = 0usize;
                    while k < 4 {
                        let v = *src.add(k);
                        acc[2 * k] += wgt * bf16_to_f32(v & 0xFFFF);
                        acc[2 * k + 1] += wgt * bf16_to_f32(v >> 16);
                        k += 1;
                    }
                }
                s += step;
            }
            acc
        }
    }

    /// LSE merge of the NS partials for launch row r and head-dim chunk
    /// blockIdx.y (64 dims); v_scale; bf16 store into out [T, HQ, D] at the
    /// request's qo_indptr rows (rows past its q length are not stored).
    /// Work item = (row, 8 dims); with M*8 <= 128 items two threads split the
    /// splits of one item and combine through smem.
    #[kernel]
    pub fn nvfp4_attn_merge(
        out: *mut u16,
        o_part: *const u16,
        lse_part: *const f32,
        qo_indptr: *const i32,
        t_rows: u32,
        hkv: u32,
        g_heads: u32,
        qt_n: u32,
        m_rows: u32,
        d: u32,
        nqt: u32,
        ns: u32,
        v_scale: f32,
    ) {
        let tid = thread::threadIdx_x();
        let r = thread::blockIdx_x();
        let dc = thread::blockIdx_y();
        let b = r / (hkv * nqt);
        let h = (r / nqt) % hkv;
        let qt = r % nqt;
        let hq = hkv * g_heads;
        if r + 1 == thread::gridDim_x() {
            // Padding tokens past the last request (CUDA-graph batches) are
            // owned by no request: zero them (this 64-dim chunk, all heads)
            // so downstream layers never read stale memory.
            let nb = thread::gridDim_x() / (hkv * nqt);
            let tail = unsafe { *qo_indptr.add(nb as usize) } as u32;
            let mut item = tid;
            while tail + item / (hq * 8) < t_rows {
                let tok = tail + item / (hq * 8);
                let hh = (item / 8) % hq;
                let col = 64 * dc + 8 * (item % 8);
                let dst = unsafe { out.add(((tok * hq + hh) * d + col) as usize) as *mut u32 };
                let mut k = 0usize;
                while k < 4 {
                    unsafe { *dst.add(k) = 0 };
                    k += 1;
                }
                item += THREADS;
            }
        }
        let qs0 = unsafe { *qo_indptr.add(b as usize) };
        let qe = unsafe { *qo_indptr.add(b as usize + 1) };
        let q_len = if qe > qs0 { (qe - qs0) as u32 } else { 0 };
        let qs0 = qs0 as u32;
        if qt * qt_n >= q_len {
            return; // no rows of this request in this q tile
        }
        let wts: *mut f32 = DynamicSharedArray::<f32>::get(); // [ns][M]
        let inv: *mut f32 = unsafe { wts.add((ns * m_rows) as usize) }; // [M]
        let part: *mut f32 = unsafe { inv.add(m_rows as usize) }; // [M*8][8]
        let base = r * ns * m_rows;
        if tid < m_rows {
            let mut mx = NEG_INF;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add((base + s * m_rows + tid) as usize) };
                if v > mx {
                    mx = v;
                }
                s += 1;
            }
            let mx = if mx == NEG_INF { 0.0 } else { mx };
            let mut sum = 0.0f32;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add((base + s * m_rows + tid) as usize) };
                let wgt = ex2(v - mx);
                unsafe { *wts.add((s * m_rows + tid) as usize) = wgt };
                sum += wgt;
                s += 1;
            }
            unsafe { *inv.add(tid as usize) = if sum == 0.0 { 0.0 } else { v_scale / sum } };
        }
        thread::sync_threads();
        let items = m_rows * 8;
        let store = |item: u32, acc: [f32; 8]| {
            let row = item / 8;
            let col = 64 * dc + 8 * (item % 8);
            let i = row / g_heads;
            let gg = row % g_heads;
            let tok = qt * qt_n + i;
            if row < qt_n * g_heads && tok < q_len {
                let sc = unsafe { *inv.add(row as usize) };
                let dst = unsafe {
                    out.add((((qs0 + tok) * hq + h * g_heads + gg) * d + col) as usize) as *mut u32
                };
                let mut k = 0usize;
                #[unroll]
                while k < 4 {
                    unsafe {
                        *dst.add(k) = bf16_bits(acc[2 * k] * sc) | (bf16_bits(acc[2 * k + 1] * sc) << 16);
                    }
                    k += 1;
                }
            }
        };
        if 2 * items <= THREADS {
            // one item per thread pair: halves of the splits, combine in smem
            let item = tid % items;
            let grp = tid / items;
            let active = tid < 2 * items;
            let mut acc = [0.0f32; 8];
            if active {
                acc = unsafe {
                    merge_item(o_part, wts, r, ns, m_rows, d, item / 8, 64 * dc + 8 * (item % 8), grp, 2)
                };
                if grp == 1 {
                    let mut k = 0usize;
                    #[unroll]
                    while k < 8 {
                        unsafe { *part.add((item * 8) as usize + k) = acc[k] };
                        k += 1;
                    }
                }
            }
            thread::sync_threads();
            if active && grp == 0 {
                let mut k = 0usize;
                #[unroll]
                while k < 8 {
                    acc[k] += unsafe { *part.add((item * 8) as usize + k) };
                    k += 1;
                }
                store(item, acc);
            }
        } else {
            let mut item = tid;
            while item < items {
                let acc = unsafe {
                    merge_item(o_part, wts, r, ns, m_rows, d, item / 8, 64 * dc + 8 * (item % 8), 0, 1)
                };
                store(item, acc);
                item += THREADS;
            }
        }
    }
}

fn main() {}
