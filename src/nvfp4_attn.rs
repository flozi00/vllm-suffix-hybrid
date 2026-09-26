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
//!   NS split-KV partitions (grid cap) = choose_splits: a function of the
//!      launch rows ONLY (never of seq_lens) so the grid is CUDA-graph
//!      replay-stable. Each CTA derives its request's split from seq_lens on
//!      the device (split_tiles: wave-aware, <= NS); launched splits past the
//!      need exit at once.

pub const MAX_ROWS: usize = 48;
pub const MAX_SPLITS: usize = 256;
pub const MAX_Q_LEN: usize = 16;
pub const MIN_TILES_PER_SPLIT: usize = 2;
/// Kernel mirror (SPLIT_COST_TOKENS): fixed cost of one split (Q load,
/// pipeline fill, partial-O write + merge read) in KV-token equivalents.
// ponytail: one calibration knob for all shapes; tune from oracle --bench
// (K2 hd512/hd256 b8..32 rows) if the silicon curve says otherwise.
pub const SPLIT_COST_TOKENS: usize = 64;
pub const HEAD_DIMS: [usize; 3] = [128, 256, 512];
/// SM120 opt-in dynamic shared memory per block.
pub const SMEM_LIMIT: usize = 99 * 1024;
/// Threads per CTA of both kernels (8 warps).
pub const THREADS: usize = 256;
/// SM120 shared memory per SM (2 CTAs only if both fit, minus 1 KB each).
pub const SMEM_PER_SM: usize = 100 * 1024;
/// S m-tile slots per warp of register variant w (kernel MTS).
pub const S_SLOTS: [usize; 4] = [0, 2, 3, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnPlan {
    pub d: usize,
    pub g: usize,
    pub qt: usize,
    pub nqt: usize,
    pub m: usize,
    pub tn: usize,
    pub ns: usize,
    /// Register variant w (cubin k2_nvfp4_attn_w{w}): O m-tile slots per
    /// warp; w1 is register-capped for 2 CTAs/SM.
    pub wv: usize,
    /// CTAs per SM the plan sizes the grid for (1 or 2).
    pub cps: usize,
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
    plan_rows(batch, q_len, hq, hkv, d, page_size, num_sms, MAX_ROWS)
}

/// `plan` with the CTA row budget capped at `max_rows` (16..=MAX_ROWS,
/// multiple of 16). A small cap suits ragged batches dominated by q_len-1
/// rows: each row needs one light CTA (w1, 2 CTAs/SM) instead of a
/// q_max-sized tile; wider rows take ceil(q_len / QT) CTAs.
#[allow(clippy::too_many_arguments)]
pub fn plan_rows(
    batch: usize,
    q_len: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    page_size: usize,
    num_sms: usize,
    max_rows: usize,
) -> Result<AttnPlan, String> {
    if !(16..=MAX_ROWS).contains(&max_rows) || max_rows % 16 != 0 {
        return Err(format!("max_rows {max_rows} not in 16..={MAX_ROWS} step 16"));
    }
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
    if round16(g) > max_rows {
        return Err(format!("GQA group {g} exceeds {max_rows} rows per CTA"));
    }
    let mut nqt = (q_len * g).div_ceil(max_rows);
    let mut qt = q_len.div_ceil(nqt);
    while round16(qt * g) > max_rows {
        nqt += 1;
        qt = q_len.div_ceil(nqt);
    }
    let nqt = q_len.div_ceil(qt);
    let m = round16(qt * g);
    // Register variant: O m-tile slots per warp = m-tiles / warps sharing
    // one 64-column group.
    let o_wpj = 8 / (d / 64);
    let wv = (m / 16).div_ceil(o_wpj);
    if wv > 3 {
        return Err(format!("M={m} D={d} needs {wv} O slots per warp (max 3)"));
    }
    let pick_tn = |cps: usize| {
        [64, 32, 16].into_iter().find(|&t| {
            let s_wps = 8 / (t / 8);
            page_size % t == 0
                && (m / 16).div_ceil(s_wps) <= S_SLOTS[wv]
                && partial_smem_bytes(m, d, t) <= SMEM_LIMIT.min(SMEM_PER_SM / cps - 1024)
        })
    };
    let (tn, cps) = match (wv == 1).then(|| pick_tn(2)).flatten() {
        Some(t) => (t, 2),
        None => (
            pick_tn(1).ok_or_else(|| {
                format!(
                    "page_size {page_size} has no KV tile in {{64, 32, 16}} fitting smem (M={m}, D={d})"
                )
            })?,
            1,
        ),
    };
    let rows = batch * hkv * nqt;
    let ns = choose_splits(rows, num_sms * cps);
    Ok(AttnPlan {
        d,
        g,
        qt,
        nqt,
        m,
        tn,
        ns,
        wv,
        cps,
        rows,
    })
}

/// Split count: minimize waves/ns (time per unit of work with R*ns equal
/// CTAs over `slots` concurrent slots); among splits within 3% of the best,
/// the smallest (fewer partials). One-wave fill for few rows, wave-tail
/// balancing for many (b >= 8 verify: R >= slots).
pub fn choose_splits(rows: usize, slots: usize) -> usize {
    let cost = |ns: usize| (rows * ns).div_ceil(slots) as f64 / ns as f64;
    let best = (1..=MAX_SPLITS).map(cost).fold(f64::INFINITY, f64::min);
    (1..=MAX_SPLITS)
        .find(|&ns| cost(ns) <= best * 1.03)
        .unwrap_or(1)
}

/// Tiles per split for one request (kernel mirror of the device-side
/// choice; `n_t` = its KV tiles, `rows` = launch rows, `slots` = SMs x CTAs
/// per SM). Candidates: the most splits fitting 1, 2 and 3 full waves of
/// `rows` CTAs, and the grid cap `ns`; cost = waves x (tiles + per-split
/// cost), first minimum wins (fewest splits). One wave never spills into a
/// partial second one (ds-MLA lesson: 192 CTAs on 188 SMs cost 21%).
pub fn split_tiles(n_t: usize, rows: usize, ns: usize, slots: usize, tn: usize) -> usize {
    let c0 = SPLIT_COST_TOKENS.div_ceil(tn);
    let (mut best, mut best_per) = (usize::MAX, MIN_TILES_PER_SPLIT);
    for k in 1..=4 {
        let s = if k == 4 { ns } else { (k * slots / rows).clamp(1, ns) };
        let per = n_t.div_ceil(s).max(MIN_TILES_PER_SPLIT);
        let cost = (rows * n_t.div_ceil(per)).div_ceil(slots) * (per + c0);
        if cost < best {
            (best, best_per) = (cost, per);
        }
    }
    best_per
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMS: usize = 188;

    #[test]
    fn capped_rows_for_q1_dominated_ragged() {
        // hd512 g8, q_max 9 under a 16-row cap: QT 2 -> 5 q tiles of M 16,
        // light w1 variant at 2 CTAs/SM (q_len-1 rows use one tile).
        let p = plan_rows(8, 9, 16, 2, 512, 64, SMS, 16).unwrap();
        assert_eq!((p.qt, p.nqt, p.m, p.wv, p.cps), (2, 5, 16, 1, 2));
        // hd256 g2: 2 balanced tiles (QT 5, M 16).
        let s = plan_rows(8, 9, 16, 8, 256, 64, SMS, 16).unwrap();
        assert_eq!((s.qt, s.nqt, s.m, s.wv), (5, 2, 16, 1));
        assert_eq!(plan_rows(8, 9, 16, 2, 512, 64, SMS, 48), plan(8, 9, 16, 2, 512, 64, SMS));
        assert!(plan_rows(1, 1, 48, 1, 512, 64, SMS, 32).is_err()); // g 48 > cap
        assert!(plan_rows(1, 1, 16, 2, 512, 64, SMS, 24).is_err());
    }

    #[test]
    fn gemma_full_attn_hd512() {
        // 16 q / 2 kv heads, page 64 (live), decode and MTP k=8 verify.
        // decode: 8 rows -> M 16, w1 (2 CTAs/SM -> TN 16 to fit), one wave.
        let p = plan(1, 1, 16, 2, 512, 64, SMS).unwrap();
        assert_eq!(
            (p.g, p.qt, p.m, p.nqt, p.wv, p.cps, p.tn),
            (8, 1, 16, 1, 1, 2, 16)
        );
        assert!(p.rows * p.ns * 100 >= SMS * p.cps * 97 && p.rows * p.ns <= SMS * p.cps);
        // q_len 9: 72 rows -> 2 CTAs of 5 + 4 tokens (40 -> M 48), w3, TN 32.
        let v = plan(1, 9, 16, 2, 512, 64, SMS).unwrap();
        assert_eq!(
            (v.qt, v.nqt, v.m, v.tn, v.rows, v.wv, v.cps, v.ns),
            (5, 2, 48, 32, 4, 3, 1, 46)
        );
        // b32 verify: 128 rows on 188 slots -> tail-balanced split (not 2)
        let b = plan(32, 9, 16, 2, 512, 64, SMS).unwrap();
        assert_eq!((b.rows, b.ns), (128, 10));
    }

    #[test]
    fn gemma_swa_hd256_and_qwen() {
        // SWA verify: 18 rows -> M 32, w1 (2 CTAs/SM) with TN 32 smem.
        let s = plan(1, 9, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!(
            (s.g, s.qt, s.nqt, s.m, s.wv, s.cps, s.tn),
            (2, 9, 1, 32, 1, 2, 32)
        );
        let s32 = plan(32, 9, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s32.rows, s32.ns), (256, 10));
        let s1 = plan(32, 1, 16, 8, 256, 64, SMS).unwrap();
        assert_eq!((s1.qt, s1.m, s1.rows), (1, 16, 256));
        // qwen3.8-27b: 24/4 heads (G=6), page 2816 = 64*44.
        let q = plan(8, 1, 24, 4, 256, 2816, SMS).unwrap();
        assert_eq!((q.g, q.qt, q.m, q.wv, q.tn), (6, 1, 16, 1, 32));
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
                        assert!(p.cps * (partial_smem_bytes(p.m, d, p.tn) + 1024) <= SMEM_PER_SM);
                        assert!((1..=3).contains(&p.wv) && (p.cps == 1 || p.wv == 1));
                        assert!((p.m / 16).div_ceil(8 / (p.tn / 8)) <= S_SLOTS[p.wv]);
                        assert!(merge_smem_bytes(p.ns, p.m) <= SMEM_LIMIT, "{p:?}");
                        assert!(p.qt * p.nqt >= q_len && p.qt * (p.nqt - 1) < q_len);
                        assert!(p.ns >= 1 && p.ns <= MAX_SPLITS);
                        assert!(p.rows * p.ns >= (SMS * p.cps).min(p.rows * MAX_SPLITS) * 2 / 3);
                    }
                }
            }
        }
        assert_eq!((round16(1), round16(16), round16(17)), (16, 16, 32));
    }

    /// (active CTAs, waves) of one request shape's device split choice.
    fn launched(p: &AttnPlan, n_t: usize, per: usize) -> (usize, usize) {
        let ctas = p.rows * n_t.div_ceil(per);
        (ctas, ctas.div_ceil(SMS * p.cps))
    }

    /// KV tiles of one q tile: full attention over kv, or a sliding window.
    fn tiles(kv: usize, win: Option<usize>, tn: usize) -> usize {
        match win {
            None => kv.div_ceil(tn),
            Some(w) => {
                let lo = (kv - 1).saturating_sub(w);
                kv.div_ceil(tn) - lo / tn
            }
        }
    }

    #[test]
    fn split_choice_is_wave_aware_on_gemma_shapes() {
        // gemma-4 26B-A4B per GPU: hd512 16/2 full attn, hd256 16/8 SWA.
        for &(hkv, d, win) in &[(2, 512, None), (8, 256, Some(1023usize)), (8, 256, Some(511))] {
            for batch in [1, 2, 4, 8, 16, 32] {
                for q_len in 1..=9 {
                    for kv in [4096, 8192, 16384, 32768, 65536, 131072] {
                        let p = plan(batch, q_len, 16, hkv, d, 64, SMS).unwrap();
                        let slots = SMS * p.cps;
                        let n_t = tiles(kv, win, p.tn);
                        let per = split_tiles(n_t, p.rows, p.ns, slots, p.tn);
                        let eff = n_t.div_ceil(per);
                        assert!(per >= MIN_TILES_PER_SPLIT && eff <= p.ns, "{p:?} {n_t}");
                        // one-wave grids stay one wave
                        if p.rows * p.ns <= slots {
                            assert!(p.rows * eff <= slots, "{p:?} {n_t} {per}");
                        }
                        // never worse (same model) than the old floor rule
                        let c0 = SPLIT_COST_TOKENS.div_ceil(p.tn);
                        let fill = (n_t * p.rows).div_ceil(slots);
                        let old = n_t.div_ceil(p.ns).max(2).max(fill.min(512 / p.tn));
                        let cost = |x: usize| launched(&p, n_t, x).1 * (x + c0);
                        assert!(cost(per) <= cost(old), "{p:?} n_t {n_t}: {per} vs {old}");
                    }
                }
            }
        }
    }

    #[test]
    fn split_choice_fixes_partial_waves() {
        // hd512 b8 MTP verify at 4k: the old floor (16 tiles) launched 288
        // CTAs on 188 SMs (1.5 waves); now one full wave.
        let p = plan(8, 9, 16, 2, 512, 64, SMS).unwrap();
        let n_t = tiles(4096 + 9, None, p.tn);
        let per = split_tiles(n_t, p.rows, p.ns, SMS * p.cps, p.tn);
        assert_eq!(launched(&p, n_t, 16), (288, 2));
        assert_eq!(launched(&p, n_t, per).1, 1);
        // hd512 b16 verify at 4k (the c13-24 prod sink): 4 waves -> 2.
        let p = plan(16, 9, 16, 2, 512, 64, SMS).unwrap();
        let per = split_tiles(n_t, p.rows, p.ns, SMS * p.cps, p.tn);
        assert_eq!(launched(&p, n_t, 16).1, 4);
        assert_eq!(launched(&p, n_t, per).1, 2);
        // hd256 SWA (window 1024) b8 verify: 384 CTAs on 376 slots -> one wave.
        let s = plan(8, 9, 16, 8, 256, 64, SMS).unwrap();
        let w = tiles(32768, Some(1023), s.tn);
        let per = split_tiles(w, s.rows, s.ns, SMS * s.cps, s.tn);
        assert_eq!(launched(&s, w, 6), (384, 2));
        assert_eq!(launched(&s, w, per).1, 1);
        // long context, many rows: the grid cap ns still wins (no change).
        let b = plan(32, 9, 16, 2, 512, 64, SMS).unwrap();
        let n = tiles(131072, None, b.tn);
        assert_eq!(n.div_ceil(split_tiles(n, b.rows, b.ns, SMS, b.tn)), b.ns);
    }

    /// The kernel's device `split_tiles`, compiled on the host: `device!`
    /// both defines it and keeps its tokens, which the test below matches
    /// against kernels-oxide/k2_nvfp4_attn/src/main.rs (whitespace and
    /// comments aside), so this IS the device code, in debug-build u32
    /// arithmetic (an overflow panics).
    macro_rules! device {
        ($($t:tt)*) => {
            const DEVICE_SRC: &str = stringify!($($t)*);
            $($t)*
        };
    }
    mod dev {
        pub const MIN_TILES_PER_SPLIT: u32 = 2;
        pub const SPLIT_COST_TOKENS: u32 = 64;
        device! {
            pub fn split_tiles(n_t: u32, rows: u32, ns: u32, slots: u32, tn: u32) -> u32 {
                let c0 = SPLIT_COST_TOKENS.div_ceil(tn);
                let mut best = u32::MAX;
                let mut best_per = MIN_TILES_PER_SPLIT;
                let mut k = 1u32;
                while k <= 4 {
                    let mut s = if k == 4 { ns } else { k * slots / rows };
                    if s > ns {
                        s = ns;
                    }
                    if s < 1 {
                        s = 1;
                    }
                    let mut per = n_t.div_ceil(s);
                    if per < MIN_TILES_PER_SPLIT {
                        per = MIN_TILES_PER_SPLIT;
                    }
                    let cost = (rows * n_t.div_ceil(per)).div_ceil(slots) * (per + c0);
                    if cost < best {
                        best = cost;
                        best_per = per;
                    }
                    k += 1;
                }
                best_per
            }
        }
        pub fn src() -> &'static str {
            DEVICE_SRC
        }
    }

    fn squash(s: &str) -> String {
        s.lines()
            .map(|l| l.split("//").next().unwrap())
            .collect::<String>()
            .replace("pub ", "")
            .split_whitespace()
            .collect()
    }

    #[test]
    fn device_split_tiles_copy_is_the_kernel_source() {
        let k = include_str!("../kernels-oxide/k2_nvfp4_attn/src/main.rs");
        let start = k.find("fn split_tiles(").expect("kernel split_tiles");
        let end = start + k[start..].find("\n    }\n").expect("fn end") + 6;
        assert_eq!(squash(&k[start..end]), squash(dev::src()));
        for c in [
            format!(
                "const MIN_TILES_PER_SPLIT: u32 = {};",
                dev::MIN_TILES_PER_SPLIT
            ),
            format!("const SPLIT_COST_TOKENS: u32 = {};", dev::SPLIT_COST_TOKENS),
        ] {
            assert!(k.contains(&c), "kernel lacks {c}");
        }
        assert_eq!(dev::MIN_TILES_PER_SPLIT as usize, MIN_TILES_PER_SPLIT);
        assert_eq!(dev::SPLIT_COST_TOKENS as usize, SPLIT_COST_TOKENS);
    }

    /// Host == device over plan-realistic and arbitrary (rows, ns, slots)
    /// grids, plus the kernel's split invariants: every tile of [0, n_t) in
    /// exactly one non-empty split, all active splits < ns, per >= 2, and a
    /// chosen 1..3-wave candidate never spills past its wave count.
    #[test]
    fn split_tiles_host_matches_device_and_covers_every_tile() {
        let mut n_ts: Vec<usize> = (0..=300).collect();
        for kv in [
            511, 512, 513, 1023, 1024, 1025, 4097, 8191, 32769, 65535, 131073, 262143, 262144,
            262145,
        ] {
            for tn in [16, 32, 64] {
                n_ts.push(kv / tn);
                n_ts.push(kv.div_ceil(tn));
            }
        }
        let check = |n_t: usize, rows: usize, ns: usize, slots: usize, tn: usize| {
            let per = split_tiles(n_t, rows, ns, slots, tn);
            let dper =
                dev::split_tiles(n_t as u32, rows as u32, ns as u32, slots as u32, tn as u32);
            assert_eq!(
                per, dper as usize,
                "n_t {n_t} rows {rows} ns {ns} slots {slots} tn {tn}"
            );
            assert!(per >= MIN_TILES_PER_SPLIT);
            // kernel: split s covers [s*per, min(s*per+per, n_t)), s < ns
            let mut covered = 0;
            for s in 0..ns {
                let (t0, t1) = (s * per, (s * per + per).min(n_t));
                if t0 < t1 {
                    assert_eq!(t0, covered, "gap/overlap");
                    covered = t1;
                }
            }
            assert_eq!(covered, n_t, "tiles past split ns-1 dropped");
            let eff = n_t.div_ceil(per);
            assert!(eff <= ns && (n_t == 0 || eff >= 1));
            // a <= 1-wave grid stays one wave
            if rows * ns <= slots {
                assert!(rows * eff <= slots);
            }
        };
        let mut plans = 0;
        for batch in [1, 2, 3, 4, 7, 8, 16, 31, 32, 64] {
            for q_len in 1..=9 {
                for &(hq, hkv) in &[(16, 2), (16, 8), (32, 8), (8, 1), (64, 8)] {
                    for d in [256, 512] {
                        for page in [16, 64] {
                            for sms in [1, 2, 7, 94, 170, 188] {
                                for max_rows in [16, 48] {
                                    let Ok(p) =
                                        plan_rows(batch, q_len, hq, hkv, d, page, sms, max_rows)
                                    else {
                                        continue;
                                    };
                                    plans += 1;
                                    for &n_t in &n_ts {
                                        check(n_t, p.rows, p.ns, sms * p.cps, p.tn);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(plans > 1000, "{plans}");
        // arbitrary (rows, ns, slots): slots 1..376, the MAX_SPLITS cap
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |m: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % m as u64) as usize
        };
        for _ in 0..200_000 {
            let (rows, ns, slots) = (1 + rnd(4096), 1 + rnd(MAX_SPLITS), 1 + rnd(376));
            let tn = [16, 32, 64][rnd(3)];
            check(rnd(262_145 / 16 + 2), rows, ns, slots, tn);
        }
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
