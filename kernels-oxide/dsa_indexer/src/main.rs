// SPDX-License-Identifier: Apache-2.0
//! DeepSeek-style sparse-attention (DSA) indexer logits on SM120 (GLM-5.3,
//! DeepSeek-V3.2): the FP8 contract of deep_gemm `fp8_fp4_paged_mqa_logits`
//! (decode, paged cache) and `fp8_fp4_mqa_logits` (prefill, contiguous K)
//! with sm_120a `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
//!
//!   logits[r, n] = sum_h relu(dot(q[r, h], k[n])) * w[r, h] * k_scale[n]
//!
//! for lo[r] <= n < hi[r] (decode: lo 0, hi context_lens[r]; prefill: ks,
//! ke). Every other column is left UNWRITTEN: vLLM 0.30 always passes
//! clean_logits=False and its top-k kernels (persistent_topk,
//! top_k_per_row_decode / _prefill) only read [0, cl) / [ks, ke). The host
//! pre-fills -inf when a caller asks for clean_logits.
//!
//! One CTA = one GROUP of G consecutive Q rows x a contiguous range of 64-token
//! K tiles (grid (splits, ceil(rows / G))). Warp w owns row w / S of the group
//! and the n-tiles (8 tokens) nt == w % S (mod S) of each tile; its Q row (H x
//! 128 e4m3 = 2 or 4 m-tiles x 4 k-steps of A fragments, 64 regs) and head
//! weights stay in registers for the whole CTA. K tiles stream through a
//! 2-stage cp.async ring (64 x 144 B pitch — conflict-free ldmatrix — + 64 f32
//! scales); B fragments come from ldmatrix.x4 (natural k order, same as the A
//! fragments loaded from global).
//!
//! Row grouping is decided PER TILE from the pages themselves: a tile is
//! staged once per DISTINCT physical page among the group's rows active in it
//! ("item" = (tile, leader row, member rows sharing the leader's page)).
//! vLLM on SM120 flattens MTP decode (next_n > 2) into single-token rows that
//! each carry a COPY of their request's block-table row, so G consecutive rows
//! of one request (row j sees cl = L - next_n + 1 + j) resolve to one item per
//! tile and K is read once for all of them; a group that straddles requests
//! simply yields one item per request (correct for any G; shared prefix pages
//! even dedup across requests). Contiguous (prefill) K has one "page" per tile.
//!
//! Epilogue per n-tile: C lane (g, t) holds heads (16mt + g, 16mt + g + 8) x
//! tokens (2t, 2t + 1); s = sum over its heads of relu(c) * w (in-register over
//! m-tiles, then butterfly over g), times the token's k_scale, stored by lanes
//! g == 0. f32 accumulation throughout; e4m3 x e4m3 products are exact, only
//! the summation order differs from the Triton fallback.
//!
//! The pure-Python twin of this schedule (items, fragments, reduction order)
//! lives in tests/test_dsa_indexer.py; launch plan: suffix_hybrid/kernels/
//! dsa_indexer.py; host op: src/dsa_indexer_oxide.rs.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::async_copy::{cp_async_ca_4, cp_async_cg_16, cp_async_commit_group, cp_async_wait_all};
    use cuda_device::warp::shuffle_xor_f32_sync;
    use cuda_device::wmma::ldmatrix_x4;
    use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

    const FULL: u32 = 0xffff_ffff;
    /// K tile = one vLLM indexer page (block_size 64, host-enforced).
    pub const T: u32 = 64;
    const D: u32 = 128;
    /// smem row pitch: 128 data + 16 pad -> ldmatrix rows hit distinct banks.
    pub const PITCH: u32 = 144;
    pub const STAGE: u32 = T * PITCH + T * 4;
    /// max rows per group (warps = G * S <= 12, launch_bounds(384)).
    pub const MAX_G: u32 = 12;

    #[inline(always)]
    fn mma(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
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
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    #[inline(always)]
    fn relu(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    /// One 16-head m-tile of a Q row: A fragments of the 4 k-steps (natural k
    /// order: a0 = (head h0, k 4t..), a1 = (h0 + 8, 4t..), a2 = (h0, 16 + 4t..),
    /// a3 = (h0 + 8, 16 + 4t..) of each 32-byte k-step) + the two head weights.
    #[derive(Clone, Copy)]
    struct QTile {
        a: [[u32; 4]; 4],
        w0: f32,
        w1: f32,
    }

    #[inline(always)]
    fn load_qtile(qrow: *const u8, wrow: *const f32, h0: u32, t: u32, live: bool) -> QTile {
        if !live {
            return QTile { a: [[0; 4]; 4], w0: 0.0, w1: 0.0 };
        }
        let at = |h: u32, off: u32| -> u32 { unsafe { *(qrow.add((h * D + off) as usize) as *const u32) } };
        let ks = |s: u32| -> [u32; 4] {
            let o = 32 * s + 4 * t;
            [at(h0, o), at(h0 + 8, o), at(h0, o + 16), at(h0 + 8, o + 16)]
        };
        QTile {
            a: [ks(0), ks(1), ks(2), ks(3)],
            w0: unsafe { *wrow.add(h0 as usize) },
            w1: unsafe { *wrow.add((h0 + 8) as usize) },
        }
    }

    /// s += sum over this lane's two heads of relu(q . k) * w for tokens
    /// (2t, 2t + 1) of the n-tile whose B fragments are b0 (k-steps 0, 1) and
    /// b1 (k-steps 2, 3).
    #[inline(always)]
    fn tile_sum(qt: &QTile, b0: [u32; 4], b1: [u32; 4], s: (f32, f32)) -> (f32, f32) {
        let mut c = [0.0f32; 4];
        c = mma(c, qt.a[0], b0[0], b0[1]);
        c = mma(c, qt.a[1], b0[2], b0[3]);
        c = mma(c, qt.a[2], b1[0], b1[1]);
        c = mma(c, qt.a[3], b1[2], b1[3]);
        (s.0 + relu(c[0]) * qt.w0 + relu(c[2]) * qt.w1, s.1 + relu(c[1]) * qt.w0 + relu(c[3]) * qt.w1)
    }

    /// Row window [lo, hi) clamped to [0, n_cols); (0, 0) for rows past `rows`.
    #[inline(always)]
    fn window(row: u32, rows: u32, lo: *const i32, hi: *const i32, has_lo: u32, n_cols: u32) -> (u32, u32) {
        if row >= rows {
            return (0, 0);
        }
        let h = unsafe { *hi.add(row as usize) };
        let h = if h < 0 { 0 } else if h as u32 > n_cols { n_cols } else { h as u32 };
        let l = if has_lo != 0 { unsafe { *lo.add(row as usize) } } else { 0 };
        let l = if l < 0 { 0 } else { l as u32 };
        (l, h)
    }

    /// Kernel arguments shared by the helpers (plain struct: every field is a
    /// kernel parameter, so it lives in registers / param space).
    #[derive(Clone, Copy)]
    struct P {
        bt: *const i32,
        lo: *const i32,
        hi: *const i32,
        rows: u32,
        group: u32,
        n_cols: u32,
        paged: u32,
        bt_div: u32,
        bt_stride: u32,
        has_lo: u32,
        row0: u32,
    }

    /// Rows of the group whose window intersects tile k (bit j = row0 + j).
    #[inline(always)]
    fn active(p: P, k: u32) -> u32 {
        let mut m = 0u32;
        let mut j = 0;
        while j < p.group {
            let (l, h) = window(p.row0 + j, p.rows, p.lo, p.hi, p.has_lo, p.n_cols);
            if l < h && l < (k + 1) * T && h > k * T {
                m |= 1 << j;
            }
            j += 1;
        }
        m
    }

    /// Physical page of tile k for group row j (contiguous K: the tile).
    #[inline(always)]
    fn page(p: P, j: u32, k: u32) -> i64 {
        if p.paged != 0 {
            let r = (p.row0 + j) / p.bt_div;
            (unsafe { *p.bt.add((r as u64 * p.bt_stride as u64 + k as u64) as usize) }) as i64
        } else {
            k as i64
        }
    }

    /// Next work item after (k, served) inside tiles [k, k1): -> (k, leader,
    /// members, served') or leader == u32::MAX when done. `served` = rows of
    /// tile k already covered by earlier items.
    #[inline(always)]
    fn next_item(p: P, mut k: u32, mut served: u32, k1: u32) -> (u32, u32, u32, u32) {
        let mut rem = if k < k1 { active(p, k) & !served } else { 0 };
        while rem == 0 {
            k += 1;
            if k >= k1 {
                return (k, u32::MAX, 0, 0);
            }
            served = 0;
            rem = active(p, k);
        }
        let lead = rem.trailing_zeros();
        let pl = page(p, lead, k);
        let mut mem = 0u32;
        let mut j = lead;
        while j < p.group {
            if (rem >> j) & 1 != 0 && page(p, j, k) == pl {
                mem |= 1 << j;
            }
            j += 1;
        }
        (k, lead, mem, served | mem)
    }

    /// cp.async tile k of `leader`'s page into a stage: 64 token rows x 8
    /// 16 B chunks (pitch 144) + 64 f32 scales. Contiguous K copies only
    /// tokens < n_cols (the tensor end); a page is always whole.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn issue(
        dst: *mut u8,
        p: P,
        kv: *const u8,
        sc: *const u8,
        k: u32,
        lead: u32,
        ps_v: u32,
        ps_s: u32,
        pitch: u32,
        tid: u32,
        nthr: u32,
    ) {
        let pg = page(p, lead, k) as u64;
        let vbase = unsafe { kv.add((pg * ps_v as u64) as usize) };
        let sbase = unsafe { sc.add((pg * ps_s as u64) as usize) };
        let mut c = tid;
        while c < T * 9 {
            let (i, part) = if c < T * 8 { (c / 8, c % 8) } else { (c - T * 8, 8) };
            if p.paged != 0 || k * T + i < p.n_cols {
                unsafe {
                    if part < 8 {
                        cp_async_cg_16(
                            dst.add((i * PITCH + part * 16) as usize) as *mut u32,
                            vbase.add((i as u64 * pitch as u64 + part as u64 * 16) as usize) as *const u32,
                        );
                    } else {
                        cp_async_ca_4(
                            dst.add((T * PITCH + i * 4) as usize) as *mut u32,
                            sbase.add((i * 4) as usize) as *const u32,
                        );
                    }
                }
            }
            c += nthr;
        }
        unsafe { cp_async_commit_group() };
    }

    /// Indexer logits, decode (paged) and prefill (contiguous) — module doc.
    /// grid (splits, ceil(rows / group)), block 32 * group * slices, dynamic
    /// smem 2 * STAGE.
    #[kernel]
    #[launch_bounds(384)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsa_logits(
        out: *mut f32,   // [rows, out_stride] f32
        q: *const u8,    // [rows, heads, 128] e4m3, contiguous
        w: *const f32,   // [rows, w_stride] f32 head weights
        kv: *const u8,   // paged: cache pages; contiguous: K [N, pitch] e4m3
        sc: *const u8,   // paged: kv + 64*128 (page scale region); contiguous: k_scale [N] f32
        bt: *const i32,  // block table (paged only)
        lo: *const i32,  // per-row window start (has_lo) — prefill ks
        hi: *const i32,  // per-row window end — decode context_lens, prefill ke
        rows: u32,
        group: u32,      // G rows per CTA (<= MAX_G)
        slices: u32,     // S warps per row
        heads: u32,      // 32 or 64
        n_cols: u32,     // logits width (max_model_len) / contiguous N
        out_stride: u32,
        w_stride: u32,
        paged: u32,
        bt_div: u32,     // block-table row = row / bt_div (native next_n; flattened 1)
        bt_stride: u32,
        ps_v: u32,       // bytes per page (values) / per 64-token tile of contiguous K
        ps_s: u32,       // bytes per page (scales) / 256
        pitch: u32,      // bytes per K token row (128 paged)
        splits: u32,
        min_tiles: u32,  // minimum tiles per CTA (fewer, longer CTAs at short contexts)
        has_lo: u32,
    ) {
        let smem: *mut u8 = DynamicSharedArray::<u32>::get() as *mut u8;
        let tid = thread::threadIdx_x();
        let nthr = thread::blockDim_x();
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let s = thread::blockIdx_x();
        let row0 = thread::blockIdx_y() * group;
        let p = P { bt, lo, hi, rows, group, n_cols, paged, bt_div, bt_stride, has_lo, row0 };

        // ---- this CTA's tile range [k0, k1) of the group's window union ----
        let mut glo = u32::MAX;
        let mut ghi = 0u32;
        let mut j = 0;
        while j < group {
            let (l, h) = window(row0 + j, rows, lo, hi, has_lo, n_cols);
            if l < h {
                if l < glo {
                    glo = l;
                }
                if h > ghi {
                    ghi = h;
                }
            }
            j += 1;
        }
        if ghi == 0 {
            return; // CTA-uniform: no live row
        }
        let t0 = glo / T;
        let t1 = ghi.div_ceil(T);
        let mut per = (t1 - t0).div_ceil(splits);
        if per < min_tiles {
            per = min_tiles;
        }
        let k0 = t0 + s * per;
        if k0 >= t1 {
            return; // CTA-uniform
        }
        let k1 = if k0 + per < t1 { k0 + per } else { t1 };

        // ---- first item in flight; meanwhile this warp's Q / w -> regs ----
        let (mut k, lead, mut mem, mut served) = next_item(p, k0, 0, k1);
        if lead == u32::MAX {
            return; // CTA-uniform (window union gap)
        }
        unsafe { issue(smem, p, kv, sc, k, lead, ps_v, ps_s, pitch, tid, nthr) };

        let jr = warp / slices; // group row of this warp
        let slice = warp % slices;
        let row = row0 + jr;
        let (rlo, rhi) = window(row, rows, lo, hi, has_lo, n_cols);
        // m-tiles as separate values (a [m-tile] array indexed in a loop
        // lands in local memory); m-tiles 2, 3 exist only for heads == 64
        let live = row < rows;
        let qrow = unsafe { q.add((row as u64 * heads as u64 * D as u64) as usize) };
        let wrow = unsafe { w.add((row as u64 * w_stride as u64) as usize) };
        let q0 = load_qtile(qrow, wrow, g, t, live);
        let q1 = load_qtile(qrow, wrow, 16 + g, t, live);
        let wide = heads > 32;
        let q2 = load_qtile(qrow, wrow, 32 + g, t, live && wide);
        let q3 = load_qtile(qrow, wrow, 48 + g, t, live && wide);

        let mut st = 0u32;
        loop {
            let (nk, nlead, nmem, nserved) = next_item(p, k, served, k1);
            unsafe { cp_async_wait_all() };
            thread::sync_threads();
            if nlead != u32::MAX {
                let dst = unsafe { smem.add(((st ^ 1) * STAGE) as usize) };
                unsafe { issue(dst, p, kv, sc, nk, nlead, ps_v, ps_s, pitch, tid, nthr) };
            }
            let stage = unsafe { smem.add((st * STAGE) as usize) };
            if (mem >> jr) & 1 != 0 {
                // warp-uniform: this warp's row reads this item's page
                let mut nt = slice;
                while nt < T / 8 {
                    let n0 = k * T + nt * 8;
                    if n0 < rhi && n0 + 8 > rlo {
                        let rp = unsafe { stage.add(((nt * 8 + lane % 8) * PITCH + 16 * (lane / 8)) as usize) };
                        let b0 = unsafe { ldmatrix_x4(rp as *const u32) }; // k-steps 0, 1
                        let b1 = unsafe { ldmatrix_x4(rp.add(64) as *const u32) }; // k-steps 2, 3
                        let (mut s0, mut s1) = tile_sum(&q0, b0, b1, (0.0, 0.0));
                        (s0, s1) = tile_sum(&q1, b0, b1, (s0, s1));
                        if wide {
                            (s0, s1) = tile_sum(&q2, b0, b1, (s0, s1));
                            (s0, s1) = tile_sum(&q3, b0, b1, (s0, s1));
                        }
                        let mut m = 4;
                        while m < 32 {
                            s0 += shuffle_xor_f32_sync(FULL, s0, m);
                            s1 += shuffle_xor_f32_sync(FULL, s1, m);
                            m *= 2;
                        }
                        if g == 0 {
                            let sc_s = unsafe { stage.add((T * PITCH) as usize) as *const f32 };
                            let orow = unsafe { out.add((row as u64 * out_stride as u64) as usize) };
                            let n = n0 + 2 * t;
                            let i = nt * 8 + 2 * t;
                            if n >= rlo && n < rhi {
                                unsafe { *orow.add(n as usize) = s0 * *sc_s.add(i as usize) };
                            }
                            if n + 1 >= rlo && n + 1 < rhi {
                                unsafe { *orow.add((n + 1) as usize) = s1 * *sc_s.add((i + 1) as usize) };
                            }
                        }
                    }
                    nt += slices;
                }
            }
            if nlead == u32::MAX {
                break; // CTA-uniform
            }
            (k, mem, served) = (nk, nmem, nserved);
            st ^= 1;
        }
    }
}

fn main() {
    // Build-only crate: scripts/oxide_build.py -> PTX (.target sm_120a) ->
    // ptxas 13.0 -> sm_120a cubin; the plugin launches it (src/dsa_indexer_oxide.rs).
}
