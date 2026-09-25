// SPDX-License-Identifier: Apache-2.0
//! NVFP4 DS-MLA reader-cache kernels on the cuda-oxide track: sparse-MLA
//! decode (top-k gathered rows) over vLLM's `nvfp4_ds_mla` reader cache with
//! warp `mma.sync.m16n8k16` f32/f16. PTX .target sm_120a (the writer's
//! cvt.rn.satfinite.e2m1x2.f32 is arch-specific), ptxas 13.0 SASS; no
//! tcgen05/UMMA.
//!
//! Plan: src/nvfp4_ds_mla.rs (pure Rust, CPU-testable). Host op:
//! src/nvfp4_ds_mla_oxide.rs. vLLM patch + oracle: sm120/nvfp4_ds_mla_patch/.
//!
//! Row layout (pinned domain contract — vLLM `nvfp4_ds_mla` reader cache,
//! rows [num_blocks, block_size, 352] uint8, per token):
//!   [0,256)   512-dim NoPE latent, e2m1 packed pairs, LOW nibble = even dim
//!   [256,320) 64 RoPE dims as raw e4m3 BYTES, one per dim, unscaled
//!   [320,352) 32 latent SF bytes, e4m3, one per 16 dims, stored through the
//!             byte permutation s -> 8*(s&3) + (s>>2) (bijective on 0..31)
//! Quantization (single global scale 1.0; the trtllm bmm1/bmm2 scales are
//! applied OUTSIDE as floats): sf = e4m3(max(amax16/6, 2^-9)),
//! dequant x = e2m1_val * sf; RoPE dequant x = e4m3 byte (no SF).
//!
//! Kernels (k2-family shape; the split domain is the CAPACITY axis — the
//! host plan picks ns from (T, HQ, C, SMs) only, c_per_split a multiple of 64):
//!   * nvfp4_ds_mla_quant_store (writer): one CTA per token, 64 threads —
//!     warp 0 owns the 32 16-dim latent blocks (one lane each: amax -> SF
//!     byte, permuted store, 8 e2m1-packed data bytes), warp 1 lanes 0..3
//!     own the 4 RoPE 16-dim groups (16 raw e4m3 bytes each), mirroring
//!     vLLM's concat_and_cache_nvfp4_ds_mla_kernel (grid tokens, block 64).
//!   * nvfp4_ds_mla_attn_partial: one CTA (8 warps) = one (q token, q-head
//!     tile of 8 heads, capacity split of 64 gathered rows). q is the
//!     packed 576-dim query [T, HQ, 576] bf16 (512 NoPE + 64 RoPE, the
//!     exact tensor the patch's call site hands us). S = q_nope * k_latent^T
//!     (e2m1 x SF dequantized in registers, k2 recipe) + q_rope * k_pe^T
//!     (raw e4m3 bytes -> f16x2), online softmax in exp2 domain, O = P * V
//!     with V = the SAME gathered rows' latent 512 dims. Gathered rows
//!     stream through a 2-stage cp.async 16 B ring (352 B = 22 chunks per
//!     row, row pitch 368 B); -1 sentinel slots / rows past topk_len are
//!     zero-staged AND masked to S = -inf (stale cache bytes can hold NaN
//!     e4m3 SF codes: zeroing kills 0*NaN poisoning, masking kills the
//!     spurious P = exp2(0)).
//!   * nvfp4_ds_mla_attn_merge: LSE merge over the ceil(topk/64) splits
//!     for one (token, head) output row — 512 dims, 256 threads.
//!
//! Numerics: every e2m1 x e4m3 product is exact in f16 (2 x 3 mantissa
//! bits); S and P*V accumulate in f32. e4m3 -> f16 is a bit shift yielding
//! value * 2^-8 exactly (subnormals included); the host folds 2^8 into
//! qk_scale_log2 AND v_scale (both operands carry the same 2^-8).
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

    const THREADS: u32 = 256;
    const FULL: u32 = 0xFFFF_FFFF;
    const NEG_INF: f32 = f32::NEG_INFINITY;

    // ---- pinned row layout (see module doc; plan module mirrors these) ----
    pub const DIM: u32 = 512; // kv_lora_rank: latent (and value) dims
    pub const PE_DIM: u32 = 64; // RoPE dims (raw e4m3 bytes)
    const DATA_BYTES: u32 = DIM / 2; // 256: e2m1-packed latent
    const PE_BASE: u32 = DATA_BYTES; // 256: raw e4m3 RoPE bytes
    const SF_BASE: u32 = PE_BASE + PE_DIM; // 320: latent SF (permuted)
    const SF_BYTES: u32 = DIM / 16; // 32
    pub const ROW_BYTES: u32 = SF_BASE + SF_BYTES; // 352
    /// smem stage row pitch (bytes; 23 x 16 B — 352 data + 16 pad).
    pub const ROW_PITCH: u32 = 368;
    const ROW_CHUNKS: u32 = ROW_BYTES / 16; // 22

    /// SF byte permutation inside the 32-byte region (bijective 0..31).
    #[inline(always)]
    fn sf_perm(s: u32) -> u32 {
        8 * (s & 3) + (s >> 2)
    }

    /// Byte offset of cache row `slot` in a [blocks, block_size, 352] view
    /// whose block stride may exceed block_size * 352 (flashinfer semantics:
    /// page = slot / block_size, entry = slot % block_size).
    #[inline(always)]
    fn row_offset(slot: u64, block_size: u32, block_stride: u32) -> usize {
        let bs = block_size as u64;
        ((slot / bs) * block_stride as u64 + (slot % bs) * ROW_BYTES as u64) as usize
    }

    /// e4m3fn byte -> f16 bits of value * 2^-8 (exact incl. subnormals).
    #[inline(always)]
    fn e4m3_f16(b: u32) -> u32 {
        ((b & 0x7F) << 7) | ((b & 0x80) << 8)
    }

    /// f32 -> e4m3fn byte (cvt.rn.satfinite; both halves get the same value,
    /// so the x2 packing order does not matter). [nvfp4_gemm verbatim]
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

    /// e4m3fn byte -> f32 (exact). [nvfp4_gemm verbatim]
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

    /// f32 -> e2m1 code (RNE, saturating at 6), sign in bit 3. [nvfp4_gemm]
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

    /// bf16 bits of an f32, round-to-nearest-even. [k2 verbatim]
    #[inline(always)]
    fn bf16_bits(x: f32) -> u32 {
        let u = x.to_bits();
        (u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16
    }

    #[inline(always)]
    fn bf16_to_f32(b: u32) -> f32 {
        f32::from_bits(b << 16)
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
    fn lg2(x: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!("lg2.approx.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
        }
        r
    }

    /// 8 e2m1 nibbles (n0 = low nibble) -> 4 f16x2 words
    /// [(n0,n1), (n2,n3), (n4,n5), (n6,n7)], first element in the low half.
    /// [k2 verbatim]
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

    /// Two e4m3 bytes (low = dim 2j, high = dim 2j+1) -> f16x2, both scaled
    /// by 2^-8 (folded into the host-side scale like every other operand).
    #[inline(always)]
    fn e4m3x2_f16x2(w: u32) -> u32 {
        e4m3_f16(w & 0xFF) | (e4m3_f16(w >> 8) << 16)
    }

    // =====================================================================
    // Writer: kv_c [T, 512] bf16 + k_pe [T, 64] bf16 -> reader-cache rows
    // [num_slots, 352] u8 via slot_mapping [T] i64 (negative slot = skip,
    // padding token). One CTA per token, 64 threads, no smem, no barrier:
    //   warp 0: 32 lanes, one 16-dim latent SF block each — amax ->
    //     sf = e4m3(max(amax/6, 2^-9)) -> permuted SF byte -> 8 data bytes
    //     (e2m1 pairs, LOW nibble = even dim, scaled by 1/sf).
    //   warp 1: lanes 0..3, one 16-dim RoPE group each -> 16 raw e4m3
    //     bytes (unscaled, cvt.rn.satfinite); lanes 4..31 idle.
    // Global scale is 1.0 (contract); bmm1/bmm2 scales are applied outside.
    // =====================================================================
    #[kernel]
    #[launch_bounds(64)]
    pub fn nvfp4_ds_mla_quant_store(
        kv_c: *const u16,      // [T, 512] bf16
        k_pe: *const u16,      // [T, 64] bf16
        slot_mapping: *const i64, // [T]
        rows: *mut u8,         // [num_slots, 352]
        num_tokens: u32,
        kv_stride: u32,
        pe_stride: u32,
        block_size: u32,   // cache rows per block (tensor shape[1])
        block_stride: u32, // bytes between blocks (tensor stride(0); may be
                           // padded, e.g. HiSparse hot views)
    ) {
        let tok = thread::blockIdx_x();
        if tok >= num_tokens {
            return;
        }
        let slot = unsafe { *slot_mapping.add(tok as usize) };
        if slot < 0 {
            return; // padding token: no cache row
        }
        let tid = thread::threadIdx_x();
        let dst = unsafe { rows.add(row_offset(slot as u64, block_size, block_stride)) };
        let blk = tid & 31; // latent SF block (warp 0) / RoPE group (warp 1)
        if tid < 32 {
            // ---- latent: one lane owns SF block `blk` and its 16 dims ----
            let kc = unsafe { kv_c.add((tok * kv_stride + blk * 16) as usize) };
            let mut amax = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let v = bf16_to_f32(unsafe { *kc.add(i) } as u32);
                let a = if v < 0.0 { -v } else { v };
                if a > amax {
                    amax = a;
                }
                i += 1;
            }
            let mut q = amax * (1.0 / 6.0);
            if q < (1.0 / 512.0) {
                q = 1.0 / 512.0; // 2^-9 scale floor (trtllm recipe floor)
            }
            let sf = to_e4m3(q);
            let sf_f = e4m3_to_f32(sf);
            let inv = if sf_f == 0.0 { 0.0 } else { 1.0 / sf_f };
            unsafe { *dst.add((SF_BASE + sf_perm(blk)) as usize) = sf as u8 };
            let d = unsafe { dst.add((blk * 8) as usize) };
            let mut j = 0usize;
            while j < 8 {
                let lo = to_e2m1(bf16_to_f32(unsafe { *kc.add(2 * j) } as u32) * inv);
                let hi = to_e2m1(bf16_to_f32(unsafe { *kc.add(2 * j + 1) } as u32) * inv);
                unsafe { *d.add(j) = (lo | (hi << 4)) as u8 };
                j += 1;
            }
        } else if blk < PE_DIM / 16 {
            // ---- RoPE: one lane owns 16 raw e4m3 dims --------------------
            let kp = unsafe { k_pe.add((tok * pe_stride + blk * 16) as usize) };
            let d = unsafe { dst.add((PE_BASE + blk * 16) as usize) };
            let mut j = 0usize;
            while j < 16 {
                let b = to_e4m3(bf16_to_f32(unsafe { *kp.add(j) } as u32));
                unsafe { *d.add(j) = b as u8 };
                j += 1;
            }
        }
    }

    // =====================================================================
    // Attention partial. One CTA = (q token t, q-head tile ht of TH heads,
    // capacity split s); grid (T*HQT, NS). M = 16 staged q rows = TH(8) real
    // heads + 8 zero mirror rows so the m16 mma shape stays integral
    // (mirror rows are never stored, their stats are per-row and unused).
    // S phase: warp w owns the w-th 8-row n-tile of the gathered-row tile;
    // PV phase: warp w owns the w-th 64-dim group of the 512-dim output.
    // Split domain = CAPACITY axis (never seq lens): grid steps / per-split
    // row count come from the plan (function of T, HQ, C ONLY), so a
    // captured CUDA graph replays for any live capacity fill state.
    // =====================================================================
    const TH: u32 = 16; // staged q rows per CTA (8 real + 8 zero mirror)
    const TN: u32 = 64; // gathered rows per smem stage (LN == TN)

    /// cp.async one TN-row window of gathered rows into a stage. Copies
    /// `ln` rows (the CTA's tile split may end mid-stage); rows past `ln`
    /// up to TN are ZEROED (16 B chunks) so stale smem can never leak into
    /// P (masked S = -inf) or O (data and SF both zero — 0 * NaN safety).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn issue_tile(
        dst: *mut u8,
        rows: *const u8,
        capacity: *const i32,
        cap_base: u64, // = t * cap_stride
        cap_len: u32,
        c0: u32, // first logical capacity row of the tile
        ln: u32, // live rows in the tile (<= TN)
        tid: u32,
        block_size: u32,
        block_stride: u32,
    ) {
        unsafe {
            let mut c = tid;
            while c < TN * ROW_CHUNKS {
                let row = c / ROW_CHUNKS;
                let chunk = c % ROW_CHUNKS;
                let live = row < ln;
                let cc = c0 + row;
                let slot = if live && cc < cap_len {
                    *capacity.add((cap_base + cc as u64) as usize)
                } else {
                    -1
                };
                let d = dst.add((row * ROW_PITCH + chunk * 16) as usize) as *mut u32;
                if slot >= 0 {
                    let src = rows
                        .add(row_offset(slot as u64, block_size, block_stride) + (chunk * 16) as usize);
                    cp_async_cg_16(d, src as *const u32);
                } else {
                    *d = 0;
                    *d.add(1) = 0;
                    *d.add(2) = 0;
                    *d.add(3) = 0;
                }
                c += THREADS;
            }
            cp_async_commit_group();
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_ds_mla_attn_partial(
        q: *const u16,            // packed query [T, HQ, 576] bf16 (512 nope | 64 rope)
        rows: *const u8,          // reader cache [num_slots, 352] (flattened)
        topk_indices: *const i32, // [T, C] physical token-slot ids; -1 masks
        o_part: *mut u16,         // [T, HQ, NS, 512] bf16
        lse_part: *mut f32,       // [T, HQ, NS] f32
        topk_stride: u32,         // topk_indices token stride (i32 elements)
        topk_len: u32,            // active capacity width (cols past it are
                                  // expected -1; -1 columns masked regardless)
        c_per_split: u32,         // capacity rows per split (multiple of TN;
                                  // host plan, src/nvfp4_ds_mla.rs)
        ns: u32,                  // launched splits = ceil(C / c_per_split)
        q_stride: u32,            // q token stride (HQ * 576 elements)
        num_tokens: u32,          // padded token count T (grid rows past the
                                  // live count exit with -inf lse; graph-safe)
        hq: u32,                  // query heads (GLM 5.3: 64; runtime u32)
        hqt: u32,                 // q-head tiles = ceil(HQ / 8)
        block_size: u32,          // cache rows per block (kv shape[1])
        block_stride: u32,        // bytes between cache blocks (kv stride(0))
        qk_scale_log2: f32,       // sm_scale * log2(e) * 2^8 (SF shift undo;
                                  // bmm1_scale = self.scale, bmm2_scale = 1.0)
    ) {
        let tid = thread::threadIdx_x();
           let w = tid / 32;
           let lane = tid % 32;
           let gq = lane / 4;
           let t4 = lane % 4;
           let r = thread::blockIdx_x();
           let s = thread::blockIdx_y();
           let ht = r % hqt;
           let t = r / hqt;
           let h0 = ht * 8; // first head of this tile
        // lse slot of CTA row pos (splits past the need / padding tokens
        // write -inf so the merge skips them; heads past HQ do not exist).
        let empty_exit = |head: u32| {
            // NOTE: no `t < num_tokens` guard here — padding-CTA rows
            // (t >= num_tokens, launched to keep graph grids rectangular)
            // are exactly the rows that must exit with -inf lse.
            if h0 + head < hq {
                let idx = ((t * hq + h0 + head) * ns + s) as usize;
                unsafe { *lse_part.add(idx) = NEG_INF };
            }
        };
        if t >= num_tokens {
            if tid < 8 {
                empty_exit(tid);
            }
            return;
        }

        // ---- this CTA's capacity range [c0, c1) ---------------------------
        let c0 = s * c_per_split;
        let mut c1 = c0 + c_per_split;
        if c1 > topk_len {
            c1 = topk_len;
        }
        if c0 >= c1 {
            if tid < 8 {
                empty_exit(tid);
            }
            return;
        }

        // ---- shared memory carve-up (all offsets 16-byte aligned) --------
        // Q f16 [16][576+8... use 16 rows x 576] | 2 x KV stage TN x 368 B
        // | P f16 [16][TN+8] | red_max + red_sum [TN/8][16] | m, l, m' [16]
        let qs: *mut u16 = DynamicSharedArray::<u16>::get();
        let qrow = DIM + PE_DIM + 8; // 584 f16 per staged Q row
        let stage_bytes = TN * ROW_PITCH;
        let kv0: *mut u8 = unsafe { qs.add((TH as usize * qrow as usize) as usize) as *mut u8 };
        let ps: *mut u16 = unsafe { kv0.add((2 * stage_bytes) as usize) as *mut u16 };
        let prow = TN + 8;
        let ntc = TN / 8; // 8 n-tiles of 8 gathered rows
        let m_rows: u32 = TH; // 16
        let red_max: *mut f32 =
            unsafe { ps.add((m_rows as usize * prow as usize) as usize) as *mut f32 };
        let red_sum: *mut f32 = unsafe { red_max.add((ntc * m_rows) as usize) };
        let m_run0: *mut f32 = unsafe { red_sum.add((ntc * m_rows) as usize) };
        let l_run: *mut f32 = unsafe { m_run0.add(m_rows as usize) };
        let m_run1: *mut f32 = unsafe { l_run.add(m_rows as usize) };

        // ---- first tile in flight; meanwhile Q -> smem (f16) -------------
        unsafe {
            issue_tile(kv0, rows, topk_indices, (t * topk_stride) as u64, topk_len, c0, TN.min(c1 - c0), tid, block_size, block_stride);
        }
        // Q staging: 16 rows x 576 f16 as bf16->f16 (mirror rows 8..15
        // zero; heads past hq zero). One u32 word (2 f16) per work item,
        // 9216 B total. LOGICAL-K PERMUTATION (k2 precedent): the B
        // fragments dequant dims 32p+8t4+{0..7} (byte 16p+4t4 of the
        // pinned 352 B row) and hand them to the mma at hardware k
        // positions {2t4,2t4+1,2t4+8,2t4+9}(+16) — with a straight
        // dim-major staging the dot product scrambles the head dim.
        // Physical dim d = 32*blk + 8a + 4b + 2e + f therefore goes to
        // smem COLUMN 32*blk + 16b + 8e + 2a + f; (f=0,1) pairs stay
        // adjacent so each u32 word stays one store.
        let words = TH * (DIM + PE_DIM) / 2; // 4608
        let mut wi = tid;
        while wi < words {
            let row = wi / ((DIM + PE_DIM) / 2); // head-mirror row 0..15
            let off = wi % ((DIM + PE_DIM) / 2); // f16 pair offset in row
            let valid = row < 8 && h0 + row < hq;
            let d0 = off * 2; // first physical dim of this pair (even)
            let o = d0 % 32;
            let (ka, kb, ke) = (o / 8, (o % 8) / 4, (o % 4) / 2);
            let col = 32 * (d0 / 32) + 16 * kb + 8 * ke + 2 * ka;
            let v: u32 = if valid {
                let p = unsafe {
                    q.add((t * q_stride + (h0 + row) * (DIM + PE_DIM) + d0) as usize)
                };
                let a = bf16_to_f32(unsafe { *p.add(0) } as u32);
                let b = bf16_to_f32(unsafe { *p.add(1) } as u32);
                cvt_f16x2_f32(a, b)
            } else {
                0
            };
            unsafe { *(qs.add((row * qrow + col) as usize) as *mut u32) = v };
            wi += THREADS;
        }
        if tid < m_rows {
            unsafe {
                *m_run0.add(tid as usize) = NEG_INF;
                *l_run.add(tid as usize) = 0.0;
            }
        }

        // warp roles (S phase: n-tile w; PV phase: 64-col group w)
        let nt = w;
        let arow = (lane % 8) + 8 * ((lane / 8) % 2); // ldmatrix row of this lane
        let acol = 8 * (lane / 16);

        let mut acc_o = [[[0.0f32; 4]; 8]; 1];

        let mut j = c0;
        while j < c1 {
            let st = ((j - c0) / TN) % 2; // ring stage of this tile
            let m_run = if st == 0 { m_run0 } else { m_run1 };
            let m_next = if st == 0 { m_run1 } else { m_run0 };
            unsafe { cp_async_wait_all() };
            thread::sync_threads();
            if j + TN < c1 {
                let dst = unsafe { kv0.add(((st ^ 1) * stage_bytes) as usize) };
                unsafe {
                    issue_tile(dst, rows, topk_indices, (t * topk_stride) as u64, topk_len, j + TN, TN.min(c1 - j - TN), tid, block_size, block_stride);
                }
            }
            let kd = unsafe { kv0.add((st * stage_bytes) as usize) };

            // ---- S = S0 (latent, 16 k-steps) + S1 (RoPE, 2 k-steps) -------
            // Warp w owns gathered rows nt*8..nt*8+7 of the tile (n-tile);
            // each lane gq owns row krow, dims k = 32p + 8*t4 (+16) step.
            let krow = nt * 8 + gq;
            let cc_krow = j + krow; // this lane's logical capacity index
            let krow_live = cc_krow < c1
                && unsafe { *topk_indices.add((t * topk_stride + cc_krow) as usize) } >= 0;
            let mut sacc = [0.0f32; 4];
            let mut p = 0u32;
            while p < DIM / 32 {
                let wk = unsafe {
                    *(kd.add((krow * ROW_PITCH + p * 16 + 4 * t4) as usize) as *const u32)
                };
                let sfb = sf_perm(2 * p + t4 / 2);
                let sc = e4m3_f16(unsafe {
                    *kd.add((krow * ROW_PITCH + SF_BASE + sfb) as usize) as u32
                });
                let sc2 = sc | (sc << 16);
                let f = nib8(wk);
                let ba = [hmul2(f[0], sc2), hmul2(f[1], sc2)];
                let bb = [hmul2(f[2], sc2), hmul2(f[3], sc2)];
                let rowp = arow * qrow + 32 * p + acol;
                let aa = unsafe { ldmatrix_x4(qs.add(rowp as usize) as *const u32) };
                let ab = unsafe { ldmatrix_x4(qs.add((rowp + 16) as usize) as *const u32) };
                sacc = unsafe { mma_m16n8k16_f32_f16(sacc, aa, ba) };
                sacc = unsafe { mma_m16n8k16_f32_f16(sacc, ab, bb) };
                p += 1;
            }
            let mut pr = 0u32;
            while pr < PE_DIM / 32 {
                // raw e4m3 RoPE bytes: 32 dims = 32 B; lane t4 owns bytes
                // 8t4..8t4+7 (dims 32pr + 8t4 .. +7) as two u32 words.
                let w0 = unsafe {
                    *(kd.add((krow * ROW_PITCH + PE_BASE + 32 * pr + 8 * t4) as usize)
                        as *const u32)
                };
                let w1 = unsafe {
                    *(kd.add((krow * ROW_PITCH + PE_BASE + 32 * pr + 8 * t4 + 4) as usize)
                        as *const u32)
                };
                let ba = [e4m3x2_f16x2(w0 & 0xFFFF), e4m3x2_f16x2(w0 >> 16)];
                let bb = [e4m3x2_f16x2(w1 & 0xFFFF), e4m3x2_f16x2(w1 >> 16)];
                let rowp = arow * qrow + DIM + 32 * pr + acol;
                let aa = unsafe { ldmatrix_x4(qs.add(rowp as usize) as *const u32) };
                let ab = unsafe { ldmatrix_x4(qs.add((rowp + 16) as usize) as *const u32) };
                sacc = unsafe { mma_m16n8k16_f32_f16(sacc, aa, ba) };
                sacc = unsafe { mma_m16n8k16_f32_f16(sacc, ab, bb) };
                pr += 1;
            }

            // ---- scale + validity mask + per-warp row max ----------------
            // Lane gq holds C rows gq, gq+8 x cols 2t4, 2t4+1 of this
            // n-tile. Rows 8..15 are zero-mirror q rows; their stats are
            // computed but never stored — masked to -inf anyway for l==0.
            let mut half = 0u32;
            while half < 2 {
                let row = gq + 8 * half;
                let valid = krow_live && row < 8 && h0 + row < hq;
                let mut e = 0usize;
                while e < 2 {
                    let v = sacc[2 * half as usize + e];
                    sacc[2 * half as usize + e] = if valid {
                        v * qk_scale_log2
                    } else {
                        NEG_INF
                    };
                    e += 1;
                }
                let a = sacc[2 * half as usize];
                let c = sacc[2 * half as usize + 1];
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
            thread::sync_threads();

            // ---- P = exp2(S - m_new) -> smem (f16), per-warp row sums -----
            let mut half = 0u32;
            while half < 2 {
                let row = gq + 8 * half;
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
                let p0 = ex2(sacc[2 * half as usize] - m_safe);
                let p1 = ex2(sacc[2 * half as usize + 1] - m_safe);
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
            thread::sync_threads();

            // ---- running stats: one thread per row ------------------------
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

            // ---- O[:, 64-col group w] = O * alpha + P V --------------------
            // V = THIS tile's gathered rows, latent dims 64w..64w+63
            // (bytes 32w..32w+32 of the row); warp w owns the whole group.
            // Row alpha for this tile (m_new over all n-tiles vs m_old):
            let alpha_row = |row: u32| -> f32 {
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
            };
            // rescale this lane's two O rows (gq, gq+8) by their alphas
            let a0 = alpha_row(gq);
            let a8 = alpha_row(gq + 8);
            let mut jn = 0usize;
            while jn < 8 {
                acc_o[0][jn][0] *= a0;
                acc_o[0][jn][1] *= a0;
                acc_o[0][jn][2] *= a8;
                acc_o[0][jn][3] *= a8;
                jn += 1;
            }
            // PV fragments: SF byte of this lane's dims is scale block
            // 4w + gq/2 of the row -> permuted SF position.
            let gblk = sf_perm(4 * w + gq / 2);
            let mut ks = 0u32;
            while ks < TN / 16 {
                // V fragments: rows (2t, 2t+1) -> b0, (2t+8, 2t+9) -> b1;
                // dim jn of the 8-col n-tile covers dims 64w + 8c + jn.
                let mut b0 = [0u32; 8];
                let mut b1 = [0u32; 8];
                let mut pr2 = 0u32;
                while pr2 < 2 {
                    let ta = ks * 16 + 2 * t4 + 8 * pr2;
                    let tb = ta + 1;
                    let wa = unsafe {
                        *(kd.add((ta * ROW_PITCH + 32 * w + 4 * gq) as usize) as *const u32)
                    };
                    let wb = unsafe {
                        *(kd.add((tb * ROW_PITCH + 32 * w + 4 * gq) as usize) as *const u32)
                    };
                    let sa = e4m3_f16(unsafe {
                        *kd.add((ta * ROW_PITCH + SF_BASE + gblk) as usize) as u32
                    });
                    let sb = e4m3_f16(unsafe {
                        *kd.add((tb * ROW_PITCH + SF_BASE + gblk) as usize) as u32
                    });
                    let sc2 = sa | (sb << 16);
                    // byte k of lo = (row a dim 2k, row b dim 2k); hi = odd.
                    let lo = (wa & 0x0F0F_0F0F) | ((wb & 0x0F0F_0F0F) << 4);
                    let hi = ((wa >> 4) & 0x0F0F_0F0F) | (wb & 0xF0F0_F0F0);
                    let fl = nib8(lo);
                    let fh = nib8(hi);
                    let mut k = 0usize;
                    while k < 4 {
                        let even = hmul2(fl[k], sc2);
                        let odd = hmul2(fh[k], sc2);
                        if pr2 == 0 {
                            b0[2 * k] = even;
                            b0[2 * k + 1] = odd;
                        } else {
                            b1[2 * k] = even;
                            b1[2 * k + 1] = odd;
                        }
                        k += 1;
                    }
                    pr2 += 1;
                }
                let pcol = ks * 16 + acol;
                let a = unsafe { ldmatrix_x4(ps.add((arow * prow + pcol) as usize) as *const u32) };
                let mut jn = 0usize;
                while jn < 8 {
                    acc_o[0][jn] =
                        unsafe { mma_m16n8k16_f32_f16(acc_o[0][jn], a, [b0[jn], b1[jn]]) };
                    jn += 1;
                }
                ks += 1;
            }
            j += TN;
        }

        thread::sync_threads(); // last tile's stats (l_run, m) visible
        let m_fin = if (c1 - c0).div_ceil(TN) % 2 == 0 { m_run0 } else { m_run1 };

        // ---- normalized partial O (bf16) + lse ----------------------------
        // lane gq of warp w stores C rows gq, gq+8 (dims 64w + 16t4, +8).
        let mut half = 0u32;
        while half < 2 {
            let row = gq + 8 * half;
            if row < 8 && h0 + row < hq {
                let l = unsafe { *l_run.add(row as usize) };
                let inv = if l == 0.0 { 0.0 } else { 1.0 / l };
                let out_row = (t * hq + h0 + row) * ns + s;
                // dims are 64w + 16t4 + jn (and +8 on the second store) — one packed
                // u32 (2 x bf16) per k, offsets in u32 elements.
                let dst = unsafe {
                    o_part.add((out_row * DIM + 64 * w + 16 * t4) as usize) as *mut u32
                };
                let mut k = 0usize;
                while k < 4 {
                    // n-tile jn covers dims 64w + 8c + jn; this lane's C
                    // cols 2t, 2t+1 -> dims 64w + 16t + jn and + 8.
                    let e0 = acc_o[0][2 * k][2 * half as usize] * inv;
                    let e1 = acc_o[0][2 * k + 1][2 * half as usize] * inv;
                    let o0 = acc_o[0][2 * k][2 * half as usize + 1] * inv;
                    let o1 = acc_o[0][2 * k + 1][2 * half as usize + 1] * inv;
                    unsafe {
                        *dst.add(k) = bf16_bits(e0) | (bf16_bits(e1) << 16);
                        *dst.add(4 + k) = bf16_bits(o0) | (bf16_bits(o1) << 16);
                    }
                    k += 1;
                }
            }
            if tid < 32 && w == 0 {
                // lse for this CTA's 16 staged rows (mirror rows unused);
                // all 32 lanes of warp 0 participate: lanes 0..15 write
                // gq 0..3 and lanes 16..31 write gq 4..7 (lanes gq*4+0..3
                // inside one half each compute the identical m/l from smem).
                if half == 0 && gq < 8 && h0 + gq < hq {
                    let m = unsafe { *m_fin.add(gq as usize) };
                    let l = unsafe { *l_run.add(gq as usize) };
                    let lse = if l == 0.0 { NEG_INF } else { m + lg2(l) };
                    unsafe {
                        *lse_part.add(((t * hq + h0 + gq) * ns + s) as usize) = lse
                    };
                }
            }
            half += 1;
        }
    }

    // =====================================================================
    // Merge: out[r, 0:512] = v_scale * (sum_s w_s * o_part[r, s, :]) /
    // sum_s w_s with w_s = exp2(lse[r, s] - max_s lse[r, :]). One CTA per
    // (token, head) row r, 256 threads over 512 dims (2 dims per thread).
    // All -inf splits => weight 0 everywhere => zero row (softmax of an
    // entirely masked capacity; the host layer decides what that means).
    // =====================================================================
    #[kernel]
    #[launch_bounds(256)]
    pub fn nvfp4_ds_mla_attn_merge(
        out: *mut u16,       // [T, HQ, 512] bf16
        o_part: *const u16,   // [T, HQ, NS, 512] bf16
        lse_part: *const f32, // [T, HQ, NS]
        ns: u32,
        v_scale: f32,
    ) {
        let tid = thread::threadIdx_x();
        let r = thread::blockIdx_x();
        let wts: *mut f32 = DynamicSharedArray::<f32>::get(); // [ns]
        let inv: *mut f32 = unsafe { wts.add(ns as usize) };
        if tid == 0 {
            let mut mx = NEG_INF;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add((r * ns + s) as usize) };
                if v > mx {
                    mx = v;
                }
                s += 1;
            }
            let mx = if mx == NEG_INF { 0.0 } else { mx };
            let mut sum = 0.0f32;
            let mut s = 0u32;
            while s < ns {
                let v = unsafe { *lse_part.add((r * ns + s) as usize) };
                let wgt = ex2(v - mx);
                unsafe { *wts.add(s as usize) = wgt };
                sum += wgt;
                s += 1;
            }
            unsafe { *inv.add(0) = if sum == 0.0 { 0.0 } else { v_scale / sum } };
        }
        thread::sync_threads();
        let sc = unsafe { *inv.add(0) };
        // 2 dims per thread: dim pair 2/2+1 of u32 word `wd` at word tid + k*256
        let mut wd = tid;
        while wd < DIM / 2 {
            let d0 = 2 * wd;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut s = 0u32;
            while s < ns {
                let wgt = unsafe { *wts.add(s as usize) };
                if wgt != 0.0 {
                    let v = unsafe {
                        *(o_part.add(((r * ns + s) * DIM + d0) as usize) as *const u32)
                    };
                    acc0 += wgt * bf16_to_f32(v & 0xFFFF);
                    acc1 += wgt * bf16_to_f32(v >> 16);
                }
                s += 1;
            }
            unsafe {
                *(out.add((r * DIM + d0) as usize) as *mut u32) =
                    bf16_bits(acc0 * sc) | (bf16_bits(acc1 * sc) << 16);
            }
            wd += THREADS;
        }
    }
}

fn main() {
    // Build-only crate: scripts/oxide_build.py -> PTX (.target sm_120a) ->
    // ptxas 13.0 -> sm_120a cubin; launched by src/nvfp4_ds_mla_oxide.rs.
}