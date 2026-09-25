// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4: launch planning + layout contract for our SM120 NVFP4-KV paged
//! decode / spec-verify attention (kernel kernels-oxide/k2_nvfp4_attn, host
//! op src/nvfp4_attn_oxide.rs). Always built: pure integer math, unit-tested
//! on any host and exposed to Python (`nvfp4_attn_plan`) so the adapter sizes
//! the split-KV workspace with the exact numbers the kernel runs with.
//!
//! Layout contract (vLLM 0.30.0 nvfp4_kv_cache_kernels.cu, HND pages; split
//! views = vllm.utils.torch_utils.nvfp4_split_data_scale):
//!   k_data/v_data [P, HKV, PAGE, D/2]  u8, e2m1 pairs, LOW nibble = even dim
//!   k_sf          [P, HKV, PAGE, D/16] e4m3, linear (token, block)
//!   v_sf          same bytes, 4-token swizzled per (page, head):
//!                 logical (t, g) at ((t/4)*4 + g/(S/4))*S + (g%(S/4))*4 + t%4
//!   value = e2m1 * e4m3 * global scale.
//!
//! Tile plan (v2, performance layout):
//!   rows of one CTA = QT q-tokens x G heads of ONE kv head, packed
//!      token-major (row = i*G + g) — every KV byte a CTA streams feeds all
//!      of them (MTP verify q_len 9 x G 8 = 72 rows share the KV reads).
//!   M = round16(QT*G) <= MAX_ROWS (O accumulator / Q smem budget); QT is
//!      the balanced split of q_len into NQT = ceil(q_len*G / MAX_ROWS) CTAs.
//!   TN KV tokens per pipelined smem stage: largest of 64/32/16 dividing the
//!      page and fitting the 99 KB opt-in smem with a 2-stage cp.async ring.
//!   NS split-KV partitions = ceil(num_sms / R): one wave of CTAs, a
//!      function of the launch rows ONLY (never of seq_lens) so the grid is
//!      CUDA-graph replay-stable; each CTA derives its chunk from seq_lens on
//!      the device (>= MIN_TILES_PER_SPLIT tiles, so short contexts use fewer
//!      splits and less partial traffic).

pub const MAX_ROWS: usize = 48;
pub const MAX_SPLITS: usize = 256;
pub const MAX_Q_LEN: usize = 16;
pub const MIN_TILES_PER_SPLIT: usize = 2;
pub const HEAD_DIMS: [usize; 3] = [128, 256, 512];
/// SM120 opt-in dynamic shared memory per block.
pub const SMEM_LIMIT: usize = 99 * 1024;
/// Threads per CTA of both kernels (8 warps).
pub const THREADS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnPlan {
    pub d: usize,
    pub g: usize,
    pub qt: usize,
    pub nqt: usize,
    pub m: usize,
    pub tn: usize,
    pub ns: usize,
    /// Launch rows R = batch * hkv * nqt (partial grid = (R, NS, 1), merge
    /// grid = (R, D/64, 1)).
    pub rows: usize,
}

pub fn round16(x: usize) -> usize {
    x.div_ceil(16) * 16
}

/// Dynamic shared memory of nvfp4_attn_partial (mirror of its carve-up):
/// Q f16 [M][D+8] | 2 x KV stage (per side TN rows of D/2+16 padded data
/// bytes + TN x D/16 scales) | P f16 [M][TN+8] | row max + row sum
/// [TN/8][M] f32 | m, l, alpha [M] f32.
pub fn partial_smem_bytes(m: usize, d: usize, tn: usize) -> usize {
    m * (d + 8) * 2
        + 2 * tn * (d / 2 + 16 + d / 16) * 2
        + m * (tn + 8) * 2
        + 2 * (tn / 8) * m * 4
        + 3 * m * 4
}

/// Dynamic shared memory of nvfp4_attn_merge: weights [NS][M] + scale [M] +
/// 2-way split-reduction buffer [M*8 items][8] f32.
pub fn merge_smem_bytes(ns: usize, m: usize) -> usize {
    (ns * m + m) * 4 + m * 8 * 8 * 4
}

pub fn plan(
    batch: usize,
    q_len: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    page_size: usize,
    num_sms: usize,
) -> Result<AttnPlan, String> {
    if !HEAD_DIMS.contains(&d) {
        return Err(format!(
            "head_dim {d} unsupported (need one of {HEAD_DIMS:?})"
        ));
    }
    if hkv == 0 || hq == 0 || hq % hkv != 0 {
        return Err(format!(
            "num_q_heads {hq} must be a positive multiple of num_kv_heads {hkv}"
        ));
    }
    if q_len == 0 || q_len > MAX_Q_LEN {
        return Err(format!("q_len {q_len} outside 1..={MAX_Q_LEN}"));
    }
    if batch == 0 || num_sms == 0 {
        return Err("batch and num_sms must be positive".into());
    }
    let g = hq / hkv;
    if g > MAX_ROWS {
        return Err(format!("GQA group {g} exceeds {MAX_ROWS} rows per CTA"));
    }
    let mut nqt = (q_len * g).div_ceil(MAX_ROWS);
    let mut qt = q_len.div_ceil(nqt);
    while round16(qt * g) > MAX_ROWS {
        nqt += 1;
        qt = q_len.div_ceil(nqt);
    }
    let nqt = q_len.div_ceil(qt);
    let m = round16(qt * g);
    let tn = [64, 32, 16]
        .into_iter()
        .find(|&t| page_size % t == 0 && partial_smem_bytes(m, d, t) <= SMEM_LIMIT)
        .ok_or_else(|| {
            format!(
                "page_size {page_size} has no KV tile in {{64, 32, 16}} fitting smem (M={m}, D={d})"
            )
        })?;
    let rows = batch * hkv * nqt;
    let ns = num_sms.div_ceil(rows).clamp(1, MAX_SPLITS);
    Ok(AttnPlan {
        d,
        g,
        qt,
        nqt,
        m,
        tn,
        ns,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMS: usize = 188;

    #[test]
    fn gemma_full_attn_hd512() {
        // 16 q / 2 kv heads, page 64 (live), decode and MTP k=8 verify.
        let p = plan(1, 1, 16, 2, 512, 64, SMS).unwrap();
        assert_eq!((p.g, p.qt, p.m, p.nqt, p.tn), (8, 1, 16, 1, 64));
        assert_eq!((p.rows, p.ns), (2, 94));
        // q_len 9: 72 rows -> 2 CTAs of 5 + 4 tokens (40 -> M 48), TN 32.
        let v = plan(1, 9, 16, 2, 512, 64, SMS).unwrap();
        assert_eq!(
            (v.qt, v.nqt, v.m, v.tn, v.rows, v.ns),
            (5, 2, 48, 32, 4, 47)
        );
    }

    #[test]
    fn gemma_swa_hd256_and_qwen() {
        let s = plan(1, 9, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s.g, s.qt, s.nqt, s.m, s.tn), (2, 9, 1, 32, 64));
        let s1 = plan(32, 1, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s1.qt, s1.m, s1.rows, s1.ns), (1, 16, 256, 1));
        // qwen3.8-27b: 24/4 heads (G=6), page 2816 = 64*44.
        let q = plan(8, 1, 24, 4, 256, 2816, SMS).unwrap();
        assert_eq!((q.g, q.qt, q.m, q.tn), (6, 1, 16, 64));
    }

    #[test]
    fn invariants_hold_over_the_served_grid() {
        for &(hq, hkv, d) in &[(16, 2, 512), (16, 8, 256), (24, 4, 256), (8, 8, 128)] {
            for page in [16, 32, 64, 2816] {
                for q_len in 1..=MAX_Q_LEN {
                    for batch in [1, 2, 7, 32, 256] {
                        let p = plan(batch, q_len, hq, hkv, d, page, SMS).unwrap();
                        assert!(p.m % 16 == 0 && p.m <= MAX_ROWS && p.qt * p.g <= p.m);
                        assert!(page % p.tn == 0 && p.tn % 16 == 0);
                        assert!(partial_smem_bytes(p.m, d, p.tn) <= SMEM_LIMIT, "{p:?}");
                        assert!(merge_smem_bytes(p.ns, p.m) <= SMEM_LIMIT, "{p:?}");
                        assert!(p.qt * p.nqt >= q_len && p.qt * (p.nqt - 1) < q_len);
                        assert!(p.ns >= 1 && p.ns <= MAX_SPLITS);
                        assert!(p.rows * p.ns >= SMS.min(p.rows * MAX_SPLITS));
                    }
                }
            }
        }
        assert_eq!((round16(1), round16(16), round16(17)), (16, 16, 32));
    }

    #[test]
    fn rejects_out_of_contract() {
        assert!(plan(1, 1, 16, 2, 576, 64, SMS).is_err());
        assert!(plan(1, 17, 16, 2, 512, 64, SMS).is_err());
        assert!(plan(1, 1, 15, 2, 512, 64, SMS).is_err());
        assert!(plan(1, 1, 16, 2, 512, 24, SMS).is_err()); // no tile divides 24
        assert!(plan(1, 1, 128, 2, 512, 64, SMS).is_err()); // G=64 > 48 rows
    }
}
