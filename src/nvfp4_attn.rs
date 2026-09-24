// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4: launch planning + layout contract for our SM120 NVFP4-KV paged
//! decode / spec-verify attention kernel (kernel + host op:
//! src/nvfp4_attn_gpu.rs, feature `nvfp4-attn-kernels`). Always built: the
//! plan is pure integer math, unit-tested on any host and exposed to Python
//! (`nvfp4_attn_plan`) so the adapter sizes the split-KV workspace with the
//! exact numbers the kernel is specialized on.
//!
//! Layout contract (vLLM 0.30.0 nvfp4_kv_cache_kernels.cu, HND pages; split
//! views = vllm.utils.torch_utils.nvfp4_split_data_scale):
//!   k_data/v_data [P, HKV, PAGE, D/2]  u8, e2m1 pairs, LOW nibble = even dim
//!   k_sf          [P, HKV, PAGE, D/16] e4m3, linear (token, block)
//!   v_sf          same bytes, 4-token swizzled per (page, head):
//!                 logical (t, g) at ((t/4)*4 + g/(S/4))*S + (g%(S/4))*4 + t%4
//!                 => viewed as [P, HKV, PAGE/4, 4 (g/SG), SG (g%SG), 4 (t%4)]
//!   value = e2m1 * e4m3 * global scale (k_scale folded into the logit scale,
//!   v_scale applied once in the merge).
//!
//! Tile plan (all tile dims powers of two — a Tile IR requirement):
//!   GP = next_pow2(G)          query heads of one KV group (G = HQ/HKV)
//!   QT q tokens per CTA, M = QT*GP rows, M*D <= ACC_ELEMS (f32 accumulator
//!      budget per CTA) and M >= 16 where the budget allows (mma row tile)
//!   NQT = ceil(q_len/QT) CTAs per (request, kv head)
//!   TN KV tokens per tile: TN*D <= KV_TILE_ELEMS, TN | page_size, 4 | TN
//!   NS split-KV partitions: a function of the launch rows ONLY (never of
//!      seq_lens) so the grid is CUDA-graph replay-stable; each CTA derives
//!      its chunk from seq_lens on the device.
//!   DT merge D-chunk: NS*M*DT <= MERGE_ELEMS.

pub const ACC_ELEMS: usize = 8192;
pub const KV_TILE_ELEMS: usize = 8192;
pub const MERGE_ELEMS: usize = 8192;
pub const MAX_SPLITS: usize = 64;
pub const MAX_Q_LEN: usize = 16;
pub const HEAD_DIMS: [usize; 3] = [128, 256, 512];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnPlan {
    pub d: usize,
    pub g: usize,
    pub gp: usize,
    pub qt: usize,
    pub nqt: usize,
    pub m: usize,
    pub tn: usize,
    pub ns: usize,
    pub dt: usize,
    /// Launch rows R = batch * hkv * nqt (partial grid = (R, NS, 1), merge
    /// grid = (R, D/DT, 1)).
    pub rows: usize,
}

fn floor_pow2(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        1 << (usize::BITS - 1 - x.leading_zeros())
    }
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
    if page_size == 0 || page_size % 4 != 0 {
        return Err(format!(
            "page_size {page_size} must be a positive multiple of 4 (V-SF groups)"
        ));
    }
    if batch == 0 || num_sms == 0 {
        return Err("batch and num_sms must be positive".into());
    }
    let g = hq / hkv;
    let gp = g.next_power_of_two();
    if gp * d > ACC_ELEMS {
        return Err(format!(
            "GQA group {g} x head_dim {d} exceeds the per-CTA accumulator budget"
        ));
    }
    let qt_max = ACC_ELEMS / (d * gp);
    let qt = q_len.next_power_of_two().max(16 / gp.min(16)).min(qt_max);
    let nqt = q_len.div_ceil(qt);
    let m = qt * gp;
    let mut tn = floor_pow2(KV_TILE_ELEMS / d);
    while page_size % tn != 0 {
        tn /= 2;
    }
    if tn < 4 {
        return Err(format!(
            "page_size {page_size} has no power-of-two tile divisor >= 4"
        ));
    }
    let rows = batch * hkv * nqt;
    let ns = (2 * num_sms)
        .div_ceil(rows)
        .next_power_of_two()
        .clamp(1, MAX_SPLITS);
    let dt = floor_pow2(MERGE_ELEMS / (ns * m)).clamp(1, d);
    Ok(AttnPlan {
        d,
        g,
        gp,
        qt,
        nqt,
        m,
        tn,
        ns,
        dt,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMS: usize = 188;

    #[test]
    fn gemma_full_attn_hd512() {
        // 16 q / 2 kv heads, decode and MTP k=8 verify.
        let p = plan(1, 1, 16, 2, 512, 16, SMS).unwrap();
        assert_eq!((p.gp, p.qt, p.m, p.nqt, p.tn), (8, 2, 16, 1, 16));
        assert_eq!((p.rows, p.ns), (2, 64));
        assert!(p.ns * p.m * p.dt <= MERGE_ELEMS);
        let v = plan(4, 9, 16, 2, 512, 16, SMS).unwrap();
        assert_eq!((v.qt, v.nqt, v.m, v.rows), (2, 5, 16, 40));
        assert_eq!(v.ns, 16); // ceil(376/40)=10 -> 16
    }

    #[test]
    fn gemma_swa_hd256_and_qwen() {
        let s = plan(1, 9, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s.g, s.gp, s.qt, s.nqt, s.m, s.tn), (2, 2, 16, 1, 32, 32));
        let s1 = plan(32, 1, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s1.qt, s1.m, s1.rows, s1.ns), (8, 16, 256, 2));
        // qwen3.8-27b: 24/4 heads (G=6 -> GP=8), page 2816 = 16*176.
        let q = plan(8, 1, 24, 4, 256, 2816, SMS).unwrap();
        assert_eq!((q.g, q.gp, q.qt, q.m, q.tn), (6, 8, 2, 16, 32));
        assert_eq!(2816 % q.tn, 0);
    }

    #[test]
    fn invariants_hold_over_the_served_grid() {
        for &(hq, hkv, d) in &[(16, 2, 512), (16, 8, 256), (24, 4, 256), (8, 8, 128)] {
            for page in [16, 32, 64, 2816] {
                for q_len in 1..=MAX_Q_LEN {
                    for batch in [1, 2, 7, 32, 256] {
                        let p = plan(batch, q_len, hq, hkv, d, page, SMS).unwrap();
                        assert!(p.m * d <= ACC_ELEMS);
                        assert!(p.tn * d <= KV_TILE_ELEMS && page % p.tn == 0 && p.tn % 4 == 0);
                        assert!(p.qt * p.nqt >= q_len && p.qt * (p.nqt - 1) < q_len);
                        assert!(p.ns.is_power_of_two() && p.ns <= MAX_SPLITS);
                        assert!(p.dt.is_power_of_two() && d % p.dt == 0);
                        assert!(p.ns * p.m * p.dt <= MERGE_ELEMS.max(p.ns * p.m));
                        assert!(p.gp >= p.g && p.gp.is_power_of_two());
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_contract() {
        assert!(plan(1, 1, 16, 2, 576, 16, SMS).is_err());
        assert!(plan(1, 17, 16, 2, 512, 16, SMS).is_err());
        assert!(plan(1, 1, 15, 2, 512, 16, SMS).is_err());
        assert!(plan(1, 1, 16, 2, 512, 18, SMS).is_err());
        assert!(plan(1, 1, 64, 2, 512, 16, SMS).is_err()); // G=32 x 512 > budget
    }
}
