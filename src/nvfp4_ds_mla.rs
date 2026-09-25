// SPDX-License-Identifier: Apache-2.0
//! NVFP4 DS-MLA: launch planning + layout contract for our SM120 sparse-MLA
//! reader-cache attention (kernels kernels-oxide/nvfp4_ds_mla, host op
//! src/nvfp4_ds_mla_oxide.rs). Always built: pure integer/float math,
//! unit-tested on any host (`nvfp4_ds_mla_plan` is exposed to Python so
//! the adapter sizes the workspace with the exact numbers the kernels run
//! with).
//!
//! Layout contract (pinned; kernels-oxide/nvfp4_ds_mla/README.md mirrors):
//!   reader-cache rows [num_slots, 352] uint8, per token:
//!     [0,256)   512-dim NoPE latent, e2m1 pairs, LOW nibble = even dim
//!     [256,320) 64 RoPE dims as raw UNSCALED e4m3 bytes
//!     [320,352) 32 latent scale-factor bytes, e4m3, one per 16 dims,
//!               stored through byte permutation s -> 8*(s&3)+(s>>2)
//!   sf = e4m3(max(amax16/6, 2^-9)); dequant x = e2m1 * sf (global scale
//!   1.0; the trtllm bmm1_scale/bmm2_scale stay outside as f32).
//!
//! Attention (BMM1 of sparse MLA decode): q = concat(q_nope [T,HQ,512],
//! q_rope [T,HQ,64]); the top-k `capacity` [T, C] i32 gives per token the
//! PHYSICAL SLOT ids of the gathered rows (the row sources). S = q . k^T
//! (NoPE latent + RoPE), out = softmax(S * sm_scale) . V with V = the same
//! rows' dequantized 512-dim latent (the latent IS the value).
//!
//! Tile plan (mirror of the K2 family; the split domain is the CAPACITY
//! axis, NEVER the capacity fill state, so the grid depends only on
//! (T, HQ, C) and a captured CUDA graph replays correctly):
//!   one CTA = (q token, q-head tile of 8 heads, capacity split);
//!   M = 16 staged q rows (8 real + 8 zero mirror rows keeps the m16 mma
//!   shape integral; mirror rows are masked to -inf and never stored);
//!   TN = 64 gathered rows per 2-stage cp.async smem ring (row pitch
//!   368 B, 22 x 16 B chunks per 352 B row);
//!   NS splits cover C: the partial CTA holds 69 KB of smem, so ONE CTA
//!   runs per SM; ns = the fewest splits that fill one wave of `num_sms`
//!   CTAs (ceil(num_sms / (T * HQT)), capped at ceil(C / TN)), and
//!   c_per_split = TN * ceil(ceil(C / TN) / ns). Chosen from (T, HQ, C,
//!   num_sms) ONLY (CUDA-graph stable). Large T (prefill chunks) collapse to
//!   ns = 1, which bounds the o_part workspace at T * HQ * 1 KiB.

pub const DIM: usize = 512; // kv_lora_rank / latent (and value) dims
pub const PE_DIM: usize = 64; // RoPE dims (raw e4m3)
pub const ROW_BYTES: usize = 352;
pub const ROW_PITCH: usize = 368; // smem stage row pitch (23 x 16 B)
/// e4m3 -> f16 in-kernel yields scale * 2^-8 (exact bit shift); the host
/// folds the inverse into qk_scale_log2 and v_scale.
pub const SCALE_FIX: f32 = 256.0;
/// SM120 opt-in dynamic shared memory per block.
pub const SMEM_LIMIT: usize = 99 * 1024;
/// SM120 shared memory per SM (2 CTAs only if both fit, minus 1 KB each).
pub const SMEM_PER_SM: usize = 100 * 1024;
/// Threads per CTA of both attention kernels (8 warps).
pub const THREADS: usize = 256;
/// Gathered rows per smem stage (kernel TN; kernel mirror).
pub const TN: usize = 64;
pub const MAX_SPLITS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsMlaPlan {
    /// q-head tiles (CTAs per token): ceil(HQ / 8).
    pub hqt: usize,
    /// capacity rows per split (multiple of TN).
    pub c_per_split: usize,
    /// launched splits: ceil(C / c_per_split) clamped to the wave target.
    pub ns: usize,
    /// partial grid rows R = T * HQT.
    pub rows: usize,
    /// merge grid = (T * HQ, 1, 1).
    pub merge_rows: usize,
}

/// sf byte permutation inside the 32-byte SF region (kernel mirror).
pub fn sf_perm(s: usize) -> usize {
    debug_assert!(s < 32);
    8 * (s & 3) + (s >> 2)
}

/// Dynamic shared memory of nvfp4_ds_mla_attn_partial (kernel mirror):
/// Q f16 [16][584] | 2 x stage of TN x ROW_PITCH bytes | P f16 [16][TN+8]
/// | red_max + red_sum [TN/8][16] f32 | m, l, m' [16] f32 x3.
pub fn partial_smem_bytes() -> usize {
    16 * (DIM + PE_DIM + 8) * 2
        + 2 * TN * ROW_PITCH
        + 16 * (TN + 8) * 2
        + 2 * (TN / 8) * 16 * 4
        + 3 * 16 * 4
}

/// Dynamic shared memory of nvfp4_ds_mla_attn_merge: weights [ns] + scale
/// [1] f32.
pub fn merge_smem_bytes(ns: usize) -> usize {
    (ns + 1) * 4
}

/// Capacity rows per split implied by a workspace of `ns` splits (the host
/// op derives it from o_part's shape, so plan and launch cannot disagree).
pub fn split_rows(capacity: usize, ns: usize) -> usize {
    TN * capacity.div_ceil(TN).div_ceil(ns.max(1))
}

/// Plan from (T, HQ, C, num_sms) ONLY (CUDA-graph stable).
pub fn plan(
    tokens: usize,
    hq: usize,
    capacity: usize,
    num_sms: usize,
) -> Result<DsMlaPlan, String> {
    if tokens == 0 || hq == 0 {
        return Err("tokens and hq must be positive".into());
    }
    if num_sms == 0 {
        return Err("num_sms must be positive".into());
    }
    if capacity == 0 {
        return Err("capacity must be positive (top-k C)".into());
    }
    if capacity > u32::MAX as usize {
        return Err("capacity exceeds u32".into());
    }
    let hqt = hq.div_ceil(8);
    let rows = tokens * hqt;
    // One partial CTA per SM (smem-bound): fill one wave with the fewest
    // splits; every split adds o_part traffic and merge work.
    let tiles = capacity.div_ceil(TN);
    let want = num_sms.div_ceil(rows).clamp(1, tiles.min(MAX_SPLITS));
    let c_per_split = split_rows(capacity, want);
    let ns = capacity.div_ceil(c_per_split);
    Ok(DsMlaPlan {
        hqt,
        c_per_split,
        ns,
        rows,
        merge_rows: tokens * hq,
    })
}

// ---------------------------------------------------------------------
// CPU numerics twin: pure-Rust reference of the row quantization
// round-trip (e2m1 / e4m3 codes as bit patterns, NO torch), pinning the
// writer's exact behavior so the GPU kernel can be diffed against it.
// ---------------------------------------------------------------------

/// e4m3fn byte -> f32 (bit-exact decode; mirror of the kernel's cvt chain).
pub fn e4m3_to_f32(b: u8) -> f32 {
    // e4m3fn: exp bias 7, denorm at exp==0, no inf; 0x7F/0xFF = NaN.
    if b & 0x7F == 0x7F {
        return f32::NAN;
    }
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as i32;
    let val = if exp == 0 {
        (man as f32) * 2.0f32.powi(-9) // denormal: man * 2^(1-7-3)
    } else {
        ((man + 8) as f32) * 2.0f32.powi(exp - 7 - 3)
    };
    sign * val
}

/// e2m1 nibble -> f32: {0, .5, 1, 1.5, 2, 3, 4, 6}, bit 3 = sign.
pub fn e2m1_to_f32(c: u8) -> f32 {
    const MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAGS[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// f32 -> e2m1 nibble (RNE, clamp to +-6), mirror of cvt.rn.satfinite.
pub fn f32_to_e2m1(x: f32) -> u8 {
    const MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = x.abs();
    let sign = if x < 0.0 || (x == 0.0 && x.is_sign_negative()) {
        8u8
    } else {
        0
    };
    // satfinite clamps |x| > 6 to 6 (nearest grid point); RNE ties go to
    // the even code (e.g. 0.25 -> 0, 5.0 -> 4.0).
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &m) in MAGS.iter().enumerate() {
        let e = (a - m).abs();
        if e < best_err || (e == best_err && i % 2 == 0) {
            best_err = e;
            best = i as u8;
        }
    }
    sign | best
}

/// f32 -> e4m3fn byte (RNE, saturating to +-448), mirror of
/// cvt.rn.satfinite.e4m3x2.
pub fn f32_to_e4m3(x: f32) -> u8 {
    let a = x.abs();
    let sign = if x < 0.0 || (x == 0.0 && x.is_sign_negative()) {
        0x80u8
    } else {
        0
    };
    if a.is_nan() {
        return 0x7F; // e4m3fn NaN
    }
    if a == 0.0 {
        return sign;
    }
    if a >= 448.0 {
        return sign | 0x7E; // satfinite: max finite
    }
    // find exp: largest e with 2^(e-7) <= a
    if a < 2.0f32.powi(-6) {
        // denormal: a = man/8 * 2^-6
        let man = (a / 2.0f32.powi(-9)).round_ties_even() as u8;
        return sign | man; // exp == 0 (man == 8 is exactly the min normal)
    }
    // normal
    let mut e = -6i32;
    while (2.0f32.powi(e + 1)) <= a {
        e += 1;
    }
    // e in -6..=8
    let mut exp = (e + 7) as u8;
    let frac = a / 2.0f32.powi(e) - 1.0;
    let mut man = (frac * 8.0).round_ties_even() as u8;
    if man == 8 {
        exp += 1;
        man = 0;
    }
    sign | (exp << 3) | man
}

/// One 16-dim latent block -> (sf byte, 8 packed data bytes), the writer's
/// per-lane contract: sf = e4m3(max(amax/6, 2^-9)); byte j packs dims
/// 2j (LOW nibble) and 2j+1 (HIGH nibble) as e2m1(x / sf).
pub fn quant_block(x: &[f32; 16]) -> (u8, [u8; 8]) {
    let mut amax = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let q = amax / 6.0;
    let q = if q < 2.0f32.powi(-9) { 2.0f32.powi(-9) } else { q };
    let sf = f32_to_e4m3(q);
    let sf_f = e4m3_to_f32(sf);
    let inv = if sf_f == 0.0 { 0.0 } else { 1.0 / sf_f };
    let mut data = [0u8; 8];
    for j in 0..8 {
        let lo = f32_to_e2m1(x[2 * j] * inv);
        let hi = f32_to_e2m1(x[2 * j + 1] * inv);
        data[j] = lo | (hi << 4);
    }
    (sf, data)
}

/// Dequant one stored row (352 B) -> (latent [512], rope [64]) f32.
pub fn dequant_row(row: &[u8]) -> ([f32; 512], [f32; 64]) {
    let mut latent = [0.0f32; 512];
    for blk in 0..32 {
        let sfb = sf_perm(blk);
        let sf = e4m3_to_f32(row[320 + sfb]);
        for j in 0..8 {
            let byte = row[blk * 8 + j];
            latent[blk * 16 + 2 * j] = e2m1_to_f32(byte & 0xF) * sf;
            latent[blk * 16 + 2 * j + 1] = e2m1_to_f32(byte >> 4) * sf;
        }
    }
    let mut rope = [0.0f32; 64];
    for d in 0..64 {
        rope[d] = e4m3_to_f32(row[256 + d]);
    }
    (latent, rope)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMS: usize = 188; // RTX PRO 6000 Blackwell

    #[test]
    fn sf_perm_is_bijective() {
        let mut seen = [false; 32];
        for s in 0..32 {
            let p = sf_perm(s);
            assert!(p < 32, "sf_perm({s}) = {p} out of range");
            assert!(!seen[p], "sf_perm not injective at {s}");
            seen[p] = true;
        }
        // pinned spot values of the trtllm recipe
        assert_eq!((sf_perm(0), sf_perm(1), sf_perm(4), sf_perm(31)), (0, 8, 1, 31));
    }

    #[test]
    fn smem_fits_the_budget() {
        let s = partial_smem_bytes();
        assert!(s <= SMEM_LIMIT, "partial smem {s} > opt-in {SMEM_LIMIT}");
        assert!(s + 1024 <= SMEM_PER_SM); // 1 CTA/SM even with w2/w3
        assert!(merge_smem_bytes(MAX_SPLITS) <= SMEM_LIMIT);
    }

    #[test]
    fn plan_covers_capacity_and_stays_graph_stable() {
        for &(t, hq, c) in &[
            (1, 32, 2048),  // GLM 5.3 nominal
            (1, 64, 2048),  // GLM 5.3 wide heads
            (8, 64, 4096),  // batched, capacity max
            (32, 32, 512),  // deep batch, short top-k
            (256, 16, 1024),
            (1, 8, 64),     // tiny
        ] {
            let p = plan(t, hq, c, SMS).unwrap();
            assert_eq!(p.hqt, hq.div_ceil(8));
            assert_eq!(p.rows, t * p.hqt);
            // every launched split covers its share; splits past C exit.
            assert!(p.ns >= 1 && p.ns <= MAX_SPLITS);
            assert_eq!(p.c_per_split % TN, 0, "c_per_split must tile by TN");
            assert_eq!(split_rows(c, p.ns), p.c_per_split, "host re-derivation");
            // no empty split: the last one starts inside C
            assert!((p.ns - 1) * p.c_per_split < c);
            // NS alone covers C unless clamped by the wave target — then
            // c_per_split grows so ns * c_per_split >= C ALWAYS holds.
            assert!(
                p.ns * p.c_per_split >= c,
                "coverage: ns {} * c/s {} >= c {} (t={t} hq={hq})",
                p.ns,
                p.c_per_split,
                c
            );
            assert_eq!(p.merge_rows, t * hq);
            // deterministic: same inputs, same plan (CUDA-graph replay).
            assert_eq!(plan(t, hq, c, SMS).unwrap(), p);
        }
    }

    #[test]
    fn plan_glm53_shapes() {
        // GLM 5.3 per-rank HQ at TP=8 is 64/8 = 8; topk 2048; 188 SMs.
        let p = plan(1, 8, 2048, SMS).unwrap();
        assert_eq!((p.hqt, p.ns, p.c_per_split), (1, 32, 64));
        let p = plan(6, 8, 2048, SMS).unwrap(); // MTP k=5 verify
        assert_eq!((p.ns, p.c_per_split), (32, 64));
        let p = plan(32, 8, 2048, SMS).unwrap();
        assert_eq!((p.ns, p.c_per_split), (6, 384)); // multi-tile splits
        let p = plan(8192, 8, 2048, SMS).unwrap(); // prefill chunk
        assert_eq!((p.ns, p.c_per_split), (1, 2048));
        let p = plan(1, 64, 2048, SMS).unwrap(); // TP=1
        assert_eq!((p.hqt, p.ns, p.c_per_split), (8, 16, 128));
    }

    #[test]
    fn plan_rejects_out_of_contract() {
        assert!(plan(0, 32, 2048, SMS).is_err());
        assert!(plan(1, 0, 2048, SMS).is_err());
        assert!(plan(1, 32, 0, SMS).is_err());
        assert!(plan(1, 32, 2048, 0).is_err());
    }

    // ---- CPU numerics twin: quant round-trip error bounds ---------------

    #[test]
    fn e2m1_e4m3_codes_round_trip() {
        // e4m3 code decode/encode agreement on its own grid.
        for b in [0u8, 1, 8, 0x40, 0x41, 0x7C, 0x7E] {
            let v = e4m3_to_f32(b);
            assert_eq!(f32_to_e4m3(v), b & 0x7F | (b & 0x80), "grid b={b:02x}");
        }
        // e2m1 likewise, incl. sign.
        for c in 0u8..15 {
            let v = e2m1_to_f32(c);
            assert_eq!(f32_to_e2m1(v), c, "grid c={c}");
        }
        // 0 and saturation clamps.
        assert_eq!(f32_to_e2m1(100.0), 7);
        assert_eq!(f32_to_e2m1(-100.0), 0xF);
        assert_eq!(f32_to_e2m1(0.25), 0); // ties -> even code
        assert_eq!(f32_to_e2m1(0.75), 2);
        assert_eq!(f32_to_e2m1(2.5), 4);
        assert_eq!(f32_to_e2m1(5.0), 6);
        assert_eq!(f32_to_e2m1(5.01), 7);
        assert_eq!(f32_to_e4m3(1e30), 0x7E);
        assert_eq!(f32_to_e4m3(420.0), 0x7D); // 416, not saturated
        assert_eq!(f32_to_e4m3(440.0), 0x7E);
    }

    #[test]
    fn latent_block_quant_relative_error_bound() {
        // 16-dim block round trip: |x - dq(x)| <= max(sf/2, |x| * 1/16):
        // e2m1 grid relative spacing 1/8 around 4..6 gives <= 3 bits of
        // relative error, and the SF grid multiplies that by <= 1+1/8.
        for seed in 0..64u32 {
            let mut x = [0.0f32; 16];
            let mut s = seed.wrapping_mul(2654435761).rotate_left(13);
            for v in x.iter_mut() {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                *v = ((s >> 8) % 4096) as f32 / 4095.0 * 12.0 - 6.0;
            }
            let (sf, data) = quant_block(&x);
            let sfd = e4m3_to_f32(sf);
            for j in 0..8 {
                let byte = data[j];
                let d0 = e2m1_to_f32(byte & 0xF) * sfd;
                let d1 = e2m1_to_f32(byte >> 4) * sfd;
                for (orig, dq) in [(x[2 * j], d0), (x[2 * j + 1], d1)] {
                    let err = (orig - dq).abs();
                    // e2m1 grid: |mag - x| <= 0.5 near the bottom of the
                    // scale (values just below 0.5*sf round to 0 or .5*sf),
                    // worst relative error 1/4 higher on the 4..6 span.
                    let bound = sfd * 0.5 + orig.abs() * (1.0 / 4.0);
                    assert!(
                        err <= bound,
                        "seed {seed}: |{orig} - {dq}| = {err} > {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn row_round_trip_full_352_bytes() {
        // end-to-end: 16 blocks of latent + 64 RoPE bytes, packed into a
        // 352 B row with the permuted SF region, then dequantized through
        // dequant_row (the merge of writer + reader contracts).
        let mut x = [0.0f32; 512];
        let mut pe = [0.0f32; 64];
        let mut s = 0xDEADBEEFu32;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) % 2000) as f32 / 999.0 * 4.0 - 2.0;
        }
        for v in pe.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) % 2000) as f32 / 999.0 * 0.5 - 0.25;
        }
        let mut row = [0u8; 352];
        for blk in 0..32 {
            let mut xb = [0.0f32; 16];
            xb.copy_from_slice(&x[blk * 16..blk * 16 + 16]);
            let (sf, data) = quant_block(&xb);
            row[blk * 8..blk * 8 + 8].copy_from_slice(&data);
            row[320 + sf_perm(blk)] = sf;
        }
        for d in 0..64 {
            row[256 + d] = f32_to_e4m3(pe[d]);
        }
        let (lat, rope) = dequant_row(&row);
        for d in 0..512 {
            // block amax -> sf <= amax/6; the dequant grid's absolute
            // resolution is sf/2 and its relative error <= 1/4 (the same
            // bound as latent_block_quant_relative_error_bound, with the
            // SF's own e4m3 rounding <= 1/16 rel folded into the fudge).
            let sfb = sf_perm(d / 16);
            let sf = e4m3_to_f32(row[320 + sfb]);
            assert!(
                (lat[d] - x[d]).abs() <= sf * 0.55 + x[d].abs() * 0.32 + 1e-6,
                "dim {d}: {} vs {}",
                lat[d],
                x[d]
            );
        }
        for d in 0..64 {
            // raw e4m3 storage: one rounding to the e4m3 grid (+-1/16 rel).
            assert!(
                (rope[d] - pe[d]).abs() <= pe[d].abs() * 0.07 + 1e-6,
                "rope dim {d}"
            );
        }
    }
}