// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4: our SM120 paged decode / spec-verify attention over vLLM's NVFP4
//! KV cache, as two cutile-rs (Tile track) kernels:
//!
//!   1. `nvfp4_attn_partial` grid (R, NS, 1): one CTA per (request, kv head,
//!      q-token tile, KV split). Loads the M = QT*GP query rows of one GQA
//!      group ONCE, then streams its KV chunk in TN-token tiles straight from
//!      the packed pages: e2m1 unpack + e4m3 block-scale multiply (K linear,
//!      V de-swizzled by a 6-D view + permute — no gather), bf16 tensor-core
//!      mma for S = Q K^T and O += P V, online softmax in the exp2 domain.
//!      Every dequantized KV element feeds all M rows (8x reuse at gemma hd512
//!      decode, 72x-class at MTP verify) — the FA2 route re-dequantizes per
//!      q tile and stages FP4 in zero-padded 128b smem slots.
//!   2. `nvfp4_attn_merge` grid (R, D/DT, 1): log-sum-exp merge of the NS
//!      normalized bf16 partials, v_scale, bf16 store into the (possibly
//!      padded-tile) output rows — out-of-bounds rows are masked by Tile IR.
//!
//! Plan/contract: src/nvfp4_attn.rs. Numerics contract: exact dequant (e2m1 x
//! e4m3 is exact in bf16), fp32 logits/softmax/accumulate, bf16 P for the PV
//! mma — the same class as FlashInfer FA2 (oracle gate: cos >= 0.9995,
//! rel <= 2e-2 vs the fp32 dequant reference).
//!
//! Host contract (same as K-GDN1): torch owns all memory (borrowed pointers),
//! launch on torch's CURRENT stream with `async_on` (never `sync_on`: that
//! synchronizes and is illegal under CUDA-graph capture), GIL released, panic
//! boundary `crate::guard_py`, every dtype/shape/stride/device mismatch a
//! hard error. Grid depends only on (batch, q_len, heads) — never on seq_lens
//! — so a captured graph replays correctly for any KV lengths.
//!
//! Built only with cargo feature `nvfp4-attn-kernels` (CUDA 13.x headers at
//! build time; pods are driver-only). GPU-less compile gate:
//!   CUDA_TOOLKIT_PATH=<dir with include/cuda.h,curand.h> \
//!     cargo test --features nvfp4-attn-kernels nvfp4_attn
//! lowers both kernels to Tile IR bytecode for sm_120 at every served shape.

#[cfg(feature = "nvfp4-attn-kernels")]
pub use device::*;

#[cfg(feature = "nvfp4-attn-kernels")]
mod device {
    use crate::nvfp4_attn::{plan, AttnPlan};
    use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn, Device, Stream};
    use cutile::half::bf16;
    use cutile::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use std::sync::{Arc, Mutex};

    #[cutile::module]
    pub mod nvfp4_attn_kernels {
        use cutile::core::*;

        /// Split-KV partial attention for one (request b, kv head h, q-token
        /// tile qt, split s). Writes normalized O (bf16) and lse (log2
        /// domain, f32; -inf for an empty split) for its M rows.
        #[cutile::entry()]
        pub unsafe fn nvfp4_attn_partial<
            const D: i32,
            const DH: i32,
            const SD: i32,
            const SG: i32,
            const GP: i32,
            const QT: i32,
            const M: i32,
            const TN: i32,
            const TQ: i32,
            const NS: i32,
        >(
            q: &Tensor<bf16, { [-1, -1, -1, -1, -1] }>,
            k_data: &Tensor<f4e2m1fnx2, { [-1, -1, -1, -1] }>,
            k_sf: &Tensor<f8e4m3fn, { [-1, -1, -1, -1] }>,
            v_data: &Tensor<f4e2m1fnx2, { [-1, -1, -1, -1] }>,
            v_sf: &Tensor<f8e4m3fn, { [-1, -1, -1, -1, -1, -1] }>,
            block_table: &Tensor<i32, { [-1, -1] }>,
            seq_lens: &Tensor<i32, { [-1] }>,
            o_part: &Tensor<bf16, { [-1, -1, -1, -1] }>,
            lse_part: &Tensor<f32, { [-1, -1, -1] }>,
            hkv: i32,
            nqt: i32,
            q_len: i32,
            page_size: i32,
            window_left: i32,
            qk_scale_log2: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let r: i32 = pid.0;
            let s: i32 = pid.1;
            let b: i32 = r / (hkv * nqt);
            let h: i32 = (r / nqt) % hkv;
            let qt: i32 = r % nqt;

            let sl_part: Partition<i32, { [1] }> = seq_lens.partition(shape![1]);
            let sl_tile: Tile<i32, { [1] }> = sl_part.load([b]);
            let kv_len: i32 = tile_to_scalar(sl_tile.reshape(shape![]));

            // Query positions: token i of this request sits at kv_len - q_len + i
            // (vLLM writes the step's KV before attention). This CTA owns
            // tokens [qt*QT, qt*QT+QT); padded tokens (>= q_len) are never
            // stored by the merge.
            let q0: i32 = kv_len - q_len + qt * QT;
            let hi: i32 = max(min(q0 + QT, kv_len), 0i32);
            let mut lo: i32 = 0i32;
            if window_left >= 0i32 {
                lo = max(q0 - window_left, 0i32);
            }
            let lo_t: i32 = lo / TN;
            let hi_t: i32 = ceil_div(hi, TN);
            let n_t: i32 = max(hi_t - lo_t, 0i32);
            let per: i32 = ceil_div(n_t, NS);
            let t0: i32 = lo_t + s * per;
            let t1: i32 = min(t0 + per, hi_t);

            let q_p: Partition<bf16, { [1, QT, 1, GP, D] }> = q.partition(shape![1, QT, 1, GP, D]);
            let q5: Tile<bf16, { [1, QT, 1, GP, D] }> = q_p.load([b, qt, h, 0i32, 0i32]);
            let qm: Tile<bf16, { [M, D] }> = q5.reshape(shape![M, D]);

            // Per-row query position [M, TN] (row = token * GP + head).
            let qi: Tile<i32, { [QT] }> = iota(shape![QT]);
            let qpos: Tile<i32, { [QT] }> = qi + broadcast_scalar(q0, shape![QT]);
            let qpos: Tile<i32, { [M, TN] }> = qpos
                .reshape(shape![QT, 1])
                .broadcast(shape![QT, GP])
                .reshape(shape![M, 1])
                .broadcast(shape![M, TN]);
            let kn: Tile<i32, { [TN] }> = iota(shape![TN]);
            let kv_len_mn: Tile<i32, { [M, TN] }> = broadcast_scalar(kv_len, shape![M, TN]);
            let kv_len_n: Tile<i32, { [TN] }> = broadcast_scalar(kv_len, shape![TN]);
            let wl_mn: Tile<i32, { [M, TN] }> = broadcast_scalar(window_left, shape![M, TN]);
            let scale_mn: Tile<f32, { [M, TN] }> = broadcast_scalar(qk_scale_log2, shape![M, TN]);
            let neg_inf_mn: Tile<f32, { [M, TN] }> = constant(f32::NEG_INFINITY, shape![M, TN]);
            let neg_inf_m: Tile<f32, { [M, 1] }> = constant(f32::NEG_INFINITY, shape![M, 1]);
            let zero_m: Tile<f32, { [M, 1] }> = constant(0.0f32, shape![M, 1]);
            let zero_nd: Tile<f32, { [TN, D] }> = constant(0.0f32, shape![TN, D]);
            let zero_mn: Tile<f32, { [M, TN] }> = constant(0.0f32, shape![M, TN]);

            let mut m_i: Tile<f32, { [M, 1] }> = constant(f32::NEG_INFINITY, shape![M, 1]);
            let mut l_i: Tile<f32, { [M, 1] }> = constant(0.0f32, shape![M, 1]);
            let mut acc: Tile<f32, { [M, D] }> = constant(0.0f32, shape![M, D]);

            let bt_p: Partition<i32, { [1, 1] }> = block_table.partition(shape![1, 1]);
            let kd_p: Partition<f4e2m1fnx2, { [1, 1, TN, DH] }> =
                k_data.partition(shape![1, 1, TN, DH]);
            let ks_p: Partition<f8e4m3fn, { [1, 1, TN, SD] }> =
                k_sf.partition(shape![1, 1, TN, SD]);
            let vd_p: Partition<f4e2m1fnx2, { [1, 1, TN, DH] }> =
                v_data.partition(shape![1, 1, TN, DH]);
            let vs_p: Partition<f8e4m3fn, { [1, 1, TQ, 4, SG, 4] }> =
                v_sf.partition(shape![1, 1, TQ, 4, SG, 4]);
            let tiles_per_page: i32 = page_size / TN;

            for j in t0..t1 {
                let tok0: i32 = j * TN;
                let page_slot: i32 = j / tiles_per_page;
                let tip: i32 = j % tiles_per_page;
                let pg_tile: Tile<i32, { [1, 1] }> = bt_p.load([b, page_slot]);
                let page: i32 = tile_to_scalar(pg_tile.reshape(shape![]));

                // ---- K: unpack e2m1, x e4m3 block scale (linear) -> bf16 --
                let kb: Tile<f4e2m1fnx2, { [1, 1, TN, DH] }> = kd_p.load([page, h, tip, 0i32]);
                let kq: Tile<f4e2m1fn, { [TN, D] }> =
                    kb.reshape(shape![TN, DH]).unpack(shape![TN, D]);
                let kf: Tile<f32, { [TN, D] }> = convert_tile(kq);
                let ks8: Tile<f8e4m3fn, { [1, 1, TN, SD] }> = ks_p.load([page, h, tip, 0i32]);
                let ks: Tile<f32, { [1, 1, TN, SD] }> = convert_tile(ks8);
                let ks: Tile<f32, { [TN, D] }> = ks
                    .reshape(shape![TN, SD, 1])
                    .broadcast(shape![TN, SD, 16])
                    .reshape(shape![TN, D]);
                let kbf: Tile<bf16, { [TN, D] }> = convert_tile(kf * ks);
                let kt: Tile<bf16, { [D, TN] }> = kbf.transpose();

                // ---- S = Q K^T (fp32), mask, online softmax (log2 domain) ----
                let sc: Tile<f32, { [M, TN] }> = mma(qm, kt, zero_mn);
                let sc: Tile<f32, { [M, TN] }> = sc * scale_mn;
                let kpos_n: Tile<i32, { [TN] }> = kn + broadcast_scalar(tok0, shape![TN]);
                let kpos: Tile<i32, { [M, TN] }> =
                    kpos_n.reshape(shape![1, TN]).broadcast(shape![M, TN]);
                let mut ok: Tile<bool, { [M, TN] }> = le_tile(kpos, qpos);
                ok = ok & lt_tile(kpos, kv_len_mn);
                if window_left >= 0i32 {
                    ok = ok & ge_tile(kpos + wl_mn, qpos);
                }
                let sc: Tile<f32, { [M, TN] }> = select(ok, sc, neg_inf_mn);
                let mx: Tile<f32, { [M] }> = reduce_max(sc, 1i32);
                let m_new: Tile<f32, { [M, 1] }> = max_tile(m_i, mx.reshape(shape![M, 1]));
                // A row with nothing visible yet keeps m = -inf: subtract 0
                // instead (exp2(-inf) = 0, never NaN from -inf - -inf).
                let m_safe: Tile<f32, { [M, 1] }> =
                    select(eq_tile(m_new, neg_inf_m), zero_m, m_new);
                let p: Tile<f32, { [M, TN] }> =
                    exp2(sc - m_safe.broadcast(shape![M, TN]), ftz::Disabled);
                let alpha: Tile<f32, { [M, 1] }> = exp2(m_i - m_safe, ftz::Disabled);
                let ps: Tile<f32, { [M] }> = reduce_sum(p, 1i32);
                l_i = l_i * alpha + ps.reshape(shape![M, 1]);
                m_i = m_new;

                // ---- V: unpack, de-swizzled scales, zero rows >= kv_len ----
                let vb: Tile<f4e2m1fnx2, { [1, 1, TN, DH] }> = vd_p.load([page, h, tip, 0i32]);
                let vq: Tile<f4e2m1fn, { [TN, D] }> =
                    vb.reshape(shape![TN, DH]).unpack(shape![TN, D]);
                let vf: Tile<f32, { [TN, D] }> = convert_tile(vq);
                let vs8: Tile<f8e4m3fn, { [1, 1, TQ, 4, SG, 4] }> =
                    vs_p.load([page, h, tip, 0i32, 0i32, 0i32]);
                let vs: Tile<f32, { [1, 1, TQ, 4, SG, 4] }> = convert_tile(vs8);
                // [tq, a, b, t4] -> [tq, t4, a, b] = token-major [TN, SD]
                let vs: Tile<f32, { [TQ, 4, 4, SG] }> =
                    permute(vs.reshape(shape![TQ, 4, SG, 4]), const_array![0, 3, 1, 2]);
                let vs: Tile<f32, { [TN, D] }> = vs
                    .reshape(shape![TN, SD, 1])
                    .broadcast(shape![TN, SD, 16])
                    .reshape(shape![TN, D]);
                let vvalid: Tile<bool, { [TN, D] }> = lt_tile(kpos_n, kv_len_n)
                    .reshape(shape![TN, 1])
                    .broadcast(shape![TN, D]);
                // Tail tokens past kv_len may hold stale/NaN-pattern bytes;
                // P is 0 there but 0 * NaN is NaN, so zero V itself.
                let vbf: Tile<bf16, { [TN, D] }> = convert_tile(select(vvalid, vf * vs, zero_nd));

                let pb: Tile<bf16, { [M, TN] }> = convert_tile(p);
                acc = mma(pb, vbf, acc * alpha.broadcast(shape![M, D]));
            }

            // Normalized partial (bf16) + lse (log2 domain). Empty split:
            // l = 0 -> O = 0, lse = -inf (merge weight 0, never NaN).
            let one_m: Tile<f32, { [M, 1] }> = constant(1.0f32, shape![M, 1]);
            let l_safe: Tile<f32, { [M, 1] }> = select(eq_tile(l_i, zero_m), one_m, l_i);
            let o: Tile<f32, { [M, D] }> = true_div(acc, l_safe.broadcast(shape![M, D]));
            let ob: Tile<bf16, { [M, D] }> = convert_tile(o);
            let lse: Tile<f32, { [M, 1] }> = m_i + log2(l_i);
            let mut o_view: PartitionMut<bf16, { [1, 1, M, D] }> =
                unsafe { o_part.partition_full_mut(shape![1, 1, M, D]) };
            o_view.store(ob.reshape(shape![1, 1, M, D]), [r, s, 0i32, 0i32]);
            let mut l_view: PartitionMut<f32, { [1, 1, M] }> =
                unsafe { lse_part.partition_full_mut(shape![1, 1, M]) };
            l_view.store(lse.reshape(shape![1, 1, M]), [r, s, 0i32]);
        }

        /// LSE merge of the NS partials for launch row r, D-chunk dc; applies
        /// v_scale and stores bf16 into out [B, q_len, HKV, G, D] (padded
        /// q-token / head rows of the tile fall outside the view: masked).
        #[cutile::entry()]
        pub unsafe fn nvfp4_attn_merge<
            const GP: i32,
            const QT: i32,
            const M: i32,
            const NS: i32,
            const DT: i32,
        >(
            out: &Tensor<bf16, { [-1, -1, -1, -1, -1] }>,
            o_part: &Tensor<bf16, { [-1, -1, -1, -1] }>,
            lse_part: &Tensor<f32, { [-1, -1, -1] }>,
            hkv: i32,
            nqt: i32,
            v_scale: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let r: i32 = pid.0;
            let dc: i32 = pid.1;
            let b: i32 = r / (hkv * nqt);
            let h: i32 = (r / nqt) % hkv;
            let qt: i32 = r % nqt;

            let l_p: Partition<f32, { [1, NS, M] }> = lse_part.partition(shape![1, NS, M]);
            let l3: Tile<f32, { [1, NS, M] }> = l_p.load([r, 0i32, 0i32]);
            let lse: Tile<f32, { [NS, M] }> = l3.reshape(shape![NS, M]);
            let mx: Tile<f32, { [M] }> = reduce_max(lse, 0i32);
            let mx: Tile<f32, { [1, M] }> = mx.reshape(shape![1, M]);
            let neg_inf: Tile<f32, { [1, M] }> = constant(f32::NEG_INFINITY, shape![1, M]);
            let zero: Tile<f32, { [1, M] }> = constant(0.0f32, shape![1, M]);
            let mx: Tile<f32, { [1, M] }> = select(eq_tile(mx, neg_inf), zero, mx);
            let w: Tile<f32, { [NS, M] }> = exp2(lse - mx.broadcast(shape![NS, M]), ftz::Disabled);
            let wsum: Tile<f32, { [M] }> = reduce_sum(w, 0i32);
            let wsum: Tile<f32, { [M, 1] }> = wsum.reshape(shape![M, 1]);
            let zero_m: Tile<f32, { [M, 1] }> = constant(0.0f32, shape![M, 1]);
            let one_m: Tile<f32, { [M, 1] }> = constant(1.0f32, shape![M, 1]);
            let wsum: Tile<f32, { [M, 1] }> = select(eq_tile(wsum, zero_m), one_m, wsum);
            let inv: Tile<f32, { [M, 1] }> = broadcast_scalar(v_scale, shape![M, 1]) / wsum;

            let o_p: Partition<bf16, { [1, NS, M, DT] }> = o_part.partition(shape![1, NS, M, DT]);
            let o4: Tile<bf16, { [1, NS, M, DT] }> = o_p.load([r, 0i32, 0i32, dc]);
            let o3: Tile<f32, { [NS, M, DT] }> = convert_tile(o4.reshape(shape![NS, M, DT]));
            let wb: Tile<f32, { [NS, M, DT] }> =
                w.reshape(shape![NS, M, 1]).broadcast(shape![NS, M, DT]);
            let o_red: Tile<f32, { [M, DT] }> = reduce_sum(o3 * wb, 0i32);
            let o_sum: Tile<f32, { [M, DT] }> = o_red.reshape(shape![M, DT]);
            let inv_b: Tile<f32, { [M, DT] }> = inv.broadcast(shape![M, DT]);
            let o_fin: Tile<f32, { [M, DT] }> = o_sum * inv_b;
            let ob: Tile<bf16, { [M, DT] }> = convert_tile(o_fin);
            let mut out_view: PartitionMut<bf16, { [1, QT, 1, GP, DT] }> =
                unsafe { out.partition_full_mut(shape![1, QT, 1, GP, DT]) };
            out_view.store(ob.reshape(shape![1, QT, 1, GP, DT]), [b, qt, h, 0i32, dc]);
        }
    }
    use nvfp4_attn_kernels::{nvfp4_attn_merge, nvfp4_attn_partial};

    // =====================================================================
    // torch-tensor introspection (duck-typed; no torch bindings).
    // =====================================================================
    struct TInfo {
        ptr: u64,
        shape: Vec<usize>,
        stride: Vec<usize>,
        dtype: String,
        device: usize,
    }

    fn tinfo(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<TInfo> {
        let dev = obj.getattr("device")?;
        let typ: String = dev.getattr("type")?.extract()?;
        if typ != "cuda" {
            return Err(PyValueError::new_err(format!(
                "{name} must be a CUDA tensor (got {typ:?}); K2-NVFP4 never runs elsewhere"
            )));
        }
        let stride: Vec<isize> = obj.call_method0("stride")?.extract()?;
        if stride.iter().any(|s| *s < 0) {
            return Err(PyValueError::new_err(format!("{name}: negative stride")));
        }
        Ok(TInfo {
            ptr: obj.call_method0("data_ptr")?.extract()?,
            shape: obj.call_method0("size")?.extract()?,
            stride: stride.into_iter().map(|s| s as usize).collect(),
            dtype: obj.getattr("dtype")?.str()?.to_string(),
            device: dev
                .getattr("index")?
                .extract::<Option<usize>>()?
                .unwrap_or(0),
        })
    }

    fn need(name: &str, i: &TInfo, dtype: &str, shape: &[usize]) -> PyResult<()> {
        if i.dtype != dtype {
            return Err(PyValueError::new_err(format!(
                "{name}: dtype {} != {dtype}",
                i.dtype
            )));
        }
        if i.shape != shape {
            return Err(PyValueError::new_err(format!(
                "{name}: shape {:?} != {shape:?}",
                i.shape
            )));
        }
        if i.stride.last().copied().unwrap_or(1) != 1 {
            return Err(PyValueError::new_err(format!(
                "{name}: last dim must be contiguous"
            )));
        }
        if i.stride.iter().any(|s| *s > i32::MAX as usize) {
            return Err(PyValueError::new_err(format!("{name}: stride exceeds i32")));
        }
        Ok(())
    }

    // Primary-context device handles, one per ordinal: Device::new retains
    // the PRIMARY context (the one torch uses) — never a second context.
    static DEVICES: Mutex<Vec<(usize, Arc<Device>)>> = Mutex::new(Vec::new());

    fn device(ordinal: usize) -> Result<Arc<Device>, String> {
        let mut g = DEVICES.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, d)) = g.iter().find(|(o, _)| *o == ordinal) {
            return Ok(d.clone());
        }
        let d = Device::new(ordinal).map_err(|e| format!("Device::new({ordinal}): {e:?}"))?;
        g.push((ordinal, d.clone()));
        Ok(d)
    }

    fn i32s(v: &[usize]) -> Vec<i32> {
        v.iter().map(|x| *x as i32).collect()
    }

    /// Our NVFP4 paged decode / spec-verify attention (K2-NVFP4).
    ///
    /// q        [B*q_len, HQ, D] bf16 (row stride free, heads/dims dense)
    /// k_data   [P, HKV, PAGE, D/2] uint8     } nvfp4_split_data_scale views
    /// k_sf     [P, HKV, PAGE, D/16] e4m3     } of the HND cache (any page
    /// v_data   [P, HKV, PAGE, D/2] uint8     } stride; (head, token, byte)
    /// v_sf     [P, HKV, PAGE, D/16] e4m3     } dense within a page side)
    /// block_table [B, max_pages] int32, seq_lens [B] int32 (incl. the step)
    /// out      [B*q_len, HQ, D] bf16, contiguous — fully written
    /// o_part   [R, NS, M, D] bf16, lse_part [R, NS, M] f32: workspace sized
    ///          from `nvfp4_attn_plan` (same args) — scratch, any contents.
    /// window_left: -1 = full attention, else FlashInfer semantics.
    /// qk_scale_log2 = sm_scale * k_scale * log2(e); v_scale = global V scale.
    #[pyfunction]
    #[pyo3(signature = (q, k_data, k_sf, v_data, v_sf, block_table, seq_lens, out, o_part, lse_part, q_len, window_left, qk_scale_log2, v_scale, num_sms, stream_ptr))]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_paged_attn_cuda<'py>(
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k_data: &Bound<'py, PyAny>,
        k_sf: &Bound<'py, PyAny>,
        v_data: &Bound<'py, PyAny>,
        v_sf: &Bound<'py, PyAny>,
        block_table: &Bound<'py, PyAny>,
        seq_lens: &Bound<'py, PyAny>,
        out: &Bound<'py, PyAny>,
        o_part: &Bound<'py, PyAny>,
        lse_part: &Bound<'py, PyAny>,
        q_len: usize,
        window_left: i32,
        qk_scale_log2: f32,
        v_scale: f32,
        num_sms: usize,
        stream_ptr: usize,
    ) -> PyResult<()> {
        let qi = tinfo("q", q)?;
        let kd = tinfo("k_data", k_data)?;
        let ks = tinfo("k_sf", k_sf)?;
        let vd = tinfo("v_data", v_data)?;
        let vs = tinfo("v_sf", v_sf)?;
        let bt = tinfo("block_table", block_table)?;
        let sl = tinfo("seq_lens", seq_lens)?;
        let oo = tinfo("out", out)?;
        let op = tinfo("o_part", o_part)?;
        let lp = tinfo("lse_part", lse_part)?;
        if qi.shape.len() != 3 || kd.shape.len() != 4 || bt.shape.len() != 2 {
            return Err(PyValueError::new_err(
                "q must be 3-D, k/v views 4-D, block_table 2-D",
            ));
        }
        let (tokens, hq, d) = (qi.shape[0], qi.shape[1], qi.shape[2]);
        let (pages, hkv, page_size) = (kd.shape[0], kd.shape[1], kd.shape[2]);
        if q_len == 0 || tokens % q_len != 0 {
            return Err(PyValueError::new_err(format!(
                "q rows {tokens} not a multiple of q_len {q_len} (uniform batches only)"
            )));
        }
        let batch = tokens / q_len;
        if batch == 0 {
            return Ok(());
        }
        let p: AttnPlan =
            plan(batch, q_len, hq, hkv, d, page_size, num_sms).map_err(PyValueError::new_err)?;
        let (dh, sd) = (d / 2, d / 16);
        need("q", &qi, "torch.bfloat16", &[tokens, hq, d])?;
        if qi.stride[1] != d {
            return Err(PyValueError::new_err(
                "q: heads must be dense (stride[1] == D)",
            ));
        }
        for (n, i, w, dt) in [
            ("k_data", &kd, dh, "torch.uint8"),
            ("v_data", &vd, dh, "torch.uint8"),
            ("k_sf", &ks, sd, "torch.float8_e4m3fn"),
            ("v_sf", &vs, sd, "torch.float8_e4m3fn"),
        ] {
            need(n, i, dt, &[pages, hkv, page_size, w])?;
            if i.stride[2] != w || i.stride[1] != page_size * w {
                return Err(PyValueError::new_err(format!(
                    "{n}: (head, token, byte) must be dense within a page side \
                     (HND nvfp4 layout), got strides {:?}",
                    i.stride
                )));
            }
        }
        need("block_table", &bt, "torch.int32", &[batch, bt.shape[1]])?;
        need("seq_lens", &sl, "torch.int32", &[batch])?;
        need("out", &oo, "torch.bfloat16", &[tokens, hq, d])?;
        if oo.stride != [hq * d, d, 1] {
            return Err(PyValueError::new_err("out must be contiguous"));
        }
        need("o_part", &op, "torch.bfloat16", &[p.rows, p.ns, p.m, d])?;
        need("lse_part", &lp, "torch.float32", &[p.rows, p.ns, p.m])?;
        for (n, i) in [
            ("k_data", &kd),
            ("k_sf", &ks),
            ("v_data", &vd),
            ("v_sf", &vs),
            ("block_table", &bt),
            ("seq_lens", &sl),
            ("out", &oo),
            ("o_part", &op),
            ("lse_part", &lp),
        ] {
            if i.device != qi.device {
                return Err(PyValueError::new_err(format!(
                    "{n} on cuda:{} != cuda:{}",
                    i.device, qi.device
                )));
            }
        }
        if stream_ptr == 0 {
            return Err(PyValueError::new_err(
                "stream_ptr must be torch's current CUDA stream",
            ));
        }
        let dims = Dims {
            batch,
            q_len,
            hq,
            hkv,
            d,
            pages,
            page_size,
        };
        py.detach(move || {
            crate::guard_py("nvfp4_paged_attn_cuda", move || {
                launch(
                    &p,
                    &dims,
                    &qi,
                    &kd,
                    &ks,
                    &vd,
                    &vs,
                    &bt,
                    &sl,
                    &oo,
                    &op,
                    &lp,
                    window_left,
                    qk_scale_log2,
                    v_scale,
                    stream_ptr,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("K2-NVFP4 launch failed: {e}")))
            })
        })
    }

    struct Dims {
        batch: usize,
        q_len: usize,
        hq: usize,
        hkv: usize,
        d: usize,
        pages: usize,
        page_size: usize,
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        p: &AttnPlan,
        x: &Dims,
        qi: &TInfo,
        kd: &TInfo,
        ks: &TInfo,
        vd: &TInfo,
        vs: &TInfo,
        bt: &TInfo,
        sl: &TInfo,
        oo: &TInfo,
        op: &TInfo,
        lp: &TInfo,
        window_left: i32,
        qk_scale_log2: f32,
        v_scale: f32,
        stream_ptr: usize,
    ) -> Result<(), String> {
        let device = device(qi.device)?;
        // SAFETY: stream_ptr is torch's live current stream on this device;
        // borrowed (never destroyed by us).
        let stream: Arc<Stream> =
            unsafe { Stream::borrow_raw(stream_ptr as *mut std::ffi::c_void, &device) };
        let ord = qi.device;
        let (g, sd, sg) = (x.hq / x.hkv, x.d / 16, x.d / 64);
        let s = |v: &[usize]| i32s(v);
        // SAFETY (borrow_raw_parts): every (ptr, shape, strides) triple below
        // is a validated view of a live torch tensor (checked in the caller)
        // re-expressed with more dims over the same bytes; torch keeps the
        // memory alive past these stream-ordered launches.
        let q5 = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                qi.ptr,
                ord,
                s(&[x.batch, x.q_len, x.hkv, g, x.d]),
                s(&[x.q_len * qi.stride[0], qi.stride[0], g * x.d, x.d, 1]),
            )
        };
        let out5 = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                oo.ptr,
                ord,
                s(&[x.batch, x.q_len, x.hkv, g, x.d]),
                s(&[x.q_len * x.hq * x.d, x.hq * x.d, g * x.d, x.d, 1]),
            )
        };
        let kd4 = unsafe {
            Tensor::<f4e2m1fnx2>::borrow_raw_parts(kd.ptr, ord, s(&kd.shape), s(&kd.stride))
        };
        let vd4 = unsafe {
            Tensor::<f4e2m1fnx2>::borrow_raw_parts(vd.ptr, ord, s(&vd.shape), s(&vd.stride))
        };
        let ks4 = unsafe {
            Tensor::<f8e4m3fn>::borrow_raw_parts(ks.ptr, ord, s(&ks.shape), s(&ks.stride))
        };
        // V scales: [P, H, N, S] -> [P, H, N/4, 4 (g/SG), SG (g%SG), 4 (t%4)]
        // (the store kernel's swizzle_scale_offset, as a strided view).
        let vs6 = unsafe {
            Tensor::<f8e4m3fn>::borrow_raw_parts(
                vs.ptr,
                ord,
                s(&[x.pages, x.hkv, x.page_size / 4, 4, sg, 4]),
                s(&[vs.stride[0], vs.stride[1], 4 * sd, sd, 4, 1]),
            )
        };
        let bt2 =
            unsafe { Tensor::<i32>::borrow_raw_parts(bt.ptr, ord, s(&bt.shape), s(&bt.stride)) };
        let sl1 =
            unsafe { Tensor::<i32>::borrow_raw_parts(sl.ptr, ord, s(&sl.shape), s(&sl.stride)) };
        let op4 =
            unsafe { Tensor::<bf16>::borrow_raw_parts(op.ptr, ord, s(&op.shape), s(&op.stride)) };
        let lp3 =
            unsafe { Tensor::<f32>::borrow_raw_parts(lp.ptr, ord, s(&lp.shape), s(&lp.stride)) };

        let pg = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        // SAFETY (unsafe entries): the partial kernel writes o_part/lse_part
        // only at its own [r, s] slot (grid == (rows, ns)); the merge writes
        // only out rows of its own (b, qt, h, dc) tile. No two CTAs alias.
        let partial = unsafe {
            nvfp4_attn_partial(
                &q5,
                &kd4,
                &ks4,
                &vd4,
                &vs6,
                &bt2,
                &sl1,
                &op4,
                &lp3,
                x.hkv as i32,
                p.nqt as i32,
                x.q_len as i32,
                x.page_size as i32,
                window_left,
                qk_scale_log2,
            )
        }
        .generics(pg(&[
            x.d,
            x.d / 2,
            sd,
            sg,
            p.gp,
            p.qt,
            p.m,
            p.tn,
            p.tn / 4,
            p.ns,
        ]))
        .grid((p.rows as u32, p.ns as u32, 1));
        let merge =
            unsafe { nvfp4_attn_merge(&out5, &op4, &lp3, x.hkv as i32, p.nqt as i32, v_scale) }
                .generics(pg(&[p.gp, p.qt, p.m, p.ns, p.dt]))
                .grid((p.rows as u32, (x.d / p.dt) as u32, 1));
        // SAFETY (async_on): outputs are torch-owned and only read by later
        // work on the same stream; the merge is ordered after the partial on
        // that stream; no host access before a torch sync.
        unsafe { partial.async_on(&stream) }.map_err(|e| format!("partial: {e:?}"))?;
        unsafe { merge.async_on(&stream) }.map_err(|e| format!("merge: {e:?}"))?;
        Ok(())
    }

    /// Point cutile's JIT at a prebuilt cubin store (bundle-shipped) so a pod
    /// never runs `tileiras`. Returns the store root.
    #[pyfunction]
    pub fn nvfp4_attn_enable_jit_store(dir: String) -> PyResult<String> {
        let store = cutile::jit_cache::FileSystemJitStore::new(&dir)
            .map_err(|e| PyRuntimeError::new_err(format!("jit store {dir}: {e}")))?;
        let root = store.root().display().to_string();
        cutile::jit_cache::enable(Arc::new(store));
        Ok(root)
    }

    /// (backend_compiles, disk_hits) since process start — the startup
    /// assertion requires backend_compiles == 0 on pods (no tileiras there).
    #[pyfunction]
    pub fn nvfp4_attn_jit_stats() -> (u64, u64) {
        (
            cutile::jit_cache::jit_backend_compile_count(),
            cutile::jit_cache::jit_disk_hit_count(),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::nvfp4_attn_kernels;
        use crate::nvfp4_attn::plan;
        use cutile::compile_api::KernelCompiler;

        fn dump(name: &str, ir: &str, bc: &[u8]) {
            if let Some(dir) = std::env::var_os("NVFP4_ATTN_DUMP_IR") {
                let dir = std::path::Path::new(&dir);
                std::fs::write(dir.join(format!("{name}.mlir")), ir).expect("dump IR");
                std::fs::write(dir.join(format!("{name}.bc")), bc).expect("dump bytecode");
            }
        }

        /// Both kernels -> Tile IR -> bytecode for sm_120 at every served
        /// shape (gemma-4 hd512 16/2 + SWA hd256 16/8, qwen3.8-27b hd256
        /// 24/4; decode q_len 1 and MTP verify q_len 9). No GPU/driver.
        #[test]
        fn compiles_to_tile_ir_for_sm120() {
            if std::env::var_os("CUTILE_BYTECODE_VERSION").is_none() {
                std::env::set_var("CUTILE_BYTECODE_VERSION", "13.2");
            }
            // (hq, hkv, d, page, q_len, batch)
            let shapes = [
                (16, 2, 512, 16, 1, 1),
                (16, 2, 512, 16, 9, 4),
                (16, 8, 256, 16, 1, 8),
                (16, 8, 256, 16, 9, 4),
                (24, 4, 256, 2816, 1, 8),
            ];
            for (hq, hkv, d, page, q_len, batch) in shapes {
                let p = plan(batch, q_len, hq, hkv, d, page, 188).unwrap();
                let (dh, sd, sg) = (d / 2, d / 16, d / 64);
                let tag = format!("hd{d}_{hq}_{hkv}_q{q_len}_b{batch}");
                let pages = 64usize;
                let page_bytes = 2 * hkv * page * (dh + sd);
                let g = hq / hkv;
                let gen = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
                let st = |v: &[usize]| v.iter().map(|x| *x as i32).collect::<Vec<_>>();
                let q_row = 3 * hq * d; // qkv-slice row stride, as served
                let _ = pages;
                let partial = KernelCompiler::new(
                    nvfp4_attn_kernels::__module_ast_self,
                    "nvfp4_attn_kernels",
                    "nvfp4_attn_partial",
                )
                .generics(gen(&[d, dh, sd, sg, p.gp, p.qt, p.m, p.tn, p.tn / 4, p.ns]))
                .strides(&[
                    ("q", &st(&[q_len * q_row, q_row, g * d, d, 1])),
                    ("k_data", &st(&[page_bytes, page * dh, dh, 1])),
                    ("k_sf", &st(&[page_bytes, page * sd, sd, 1])),
                    ("v_data", &st(&[page_bytes, page * dh, dh, 1])),
                    ("v_sf", &st(&[page_bytes, page * sd, 4 * sd, sd, 4, 1])),
                    ("block_table", &st(&[512, 1])),
                    ("seq_lens", &st(&[1])),
                    ("o_part", &st(&[p.ns * p.m * d, p.m * d, d, 1])),
                    ("lse_part", &st(&[p.ns * p.m, p.m, 1])),
                ])
                .target("sm_120")
                .compile()
                .unwrap_or_else(|e| panic!("partial {tag} must lower to Tile IR: {e:?}"));
                let ir = partial.ir_text();
                assert!(ir.contains("mma"), "{tag}: expected tensor-core mma in IR");
                let bc = partial.bytecode().expect("bytecode");
                assert_eq!(&bc[..8], &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]);
                dump(&format!("partial_{tag}"), &ir, &bc);

                let merge = KernelCompiler::new(
                    nvfp4_attn_kernels::__module_ast_self,
                    "nvfp4_attn_kernels",
                    "nvfp4_attn_merge",
                )
                .generics(gen(&[p.gp, p.qt, p.m, p.ns, p.dt]))
                .strides(&[
                    ("out", &st(&[q_len * hq * d, hq * d, g * d, d, 1])),
                    ("o_part", &st(&[p.ns * p.m * d, p.m * d, d, 1])),
                    ("lse_part", &st(&[p.ns * p.m, p.m, 1])),
                ])
                .target("sm_120")
                .compile()
                .unwrap_or_else(|e| panic!("merge {tag} must lower to Tile IR: {e:?}"));
                let ir = merge.ir_text();
                let bc = merge.bytecode().expect("bytecode");
                dump(&format!("merge_{tag}"), &ir, &bc);
            }
        }
    }
}
