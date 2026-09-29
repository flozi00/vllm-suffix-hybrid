// SPDX-License-Identifier: Apache-2.0
//! NVFP4 MoE decode host op (feature `oxide-kernels`): launches
//! kernels-oxide/nvfp4_moe (sm_120a SASS, ptxas 13.0) on torch's stream.
//! All grids a function of M only (no device->host reads). `mode`:
//!   0 legacy, 6 launches: route, quant(x), fc1+act_and_mul, quant(h), fc2,
//!     combine
//!   1 fused front, 4: route+quant(x), fc1+act_and_mul+quant(h), fc2, combine
//!   2 fused, 3: route+quant(x), fc1+act_and_mul+quant(h), token-major
//!     fc2+combine (decode M; Python picks the plan per M bucket, TUNE table)
//! All three are bit-identical (same f32 ops per output element).
//! `tune` (Python packs it per M bucket, nvfp4_moe.py tune_for): bits 0-7
//! la1, 8-15 la2 (L2 prefetch lookahead of the fc1 / fc2 weight rows in k64
//! steps, 0 = off), bit 16 pf2 (fc1 prefetches w2 for fc2), bits 24-29 skip
//! mask (launch i of the plan is skipped when bit 24+i is set: the sweep's
//! per-launch timing only; serving never sets it). Every tune value is
//! bit-identical to tune 0: prefetches are hints, the loads are unchanged.
//! Oracle/bench + vLLM wiring: suffix_hybrid/kernels/nvfp4_moe.py.

use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "nvfp4_moe";

pub(crate) struct T {
    pub(crate) ptr: u64,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) dtype: String,
    pub(crate) device: usize,
}

pub(crate) fn info(name: &str, o: &Bound<'_, PyAny>) -> PyResult<T> {
    let dev = o.getattr("device")?;
    let typ: String = dev.getattr("type")?.extract()?;
    if typ != "cuda" {
        return Err(PyValueError::new_err(format!(
            "{name} must be a CUDA tensor"
        )));
    }
    let stride: Vec<isize> = o.call_method0("stride")?.extract()?;
    if stride.iter().any(|s| *s < 0) {
        return Err(PyValueError::new_err(format!("{name}: negative stride")));
    }
    Ok(T {
        ptr: o.call_method0("data_ptr")?.extract()?,
        shape: o.call_method0("size")?.extract()?,
        stride: stride.into_iter().map(|s| s as usize).collect(),
        dtype: o.getattr("dtype")?.str()?.to_string(),
        device: dev
            .getattr("index")?
            .extract::<Option<usize>>()?
            .unwrap_or(0),
    })
}

pub(crate) fn contiguous(t: &T) -> bool {
    let mut want = 1;
    for (d, s) in t.shape.iter().zip(&t.stride).rev() {
        if *d > 1 && *s != want {
            return false;
        }
        want *= d;
    }
    true
}

pub(crate) fn need(
    name: &str,
    t: &T,
    dtypes: &[&str],
    shape: Option<&[usize]>,
    min_elems: usize,
) -> PyResult<()> {
    if !dtypes.contains(&t.dtype.as_str()) {
        return Err(PyValueError::new_err(format!(
            "{name}: dtype {} not in {dtypes:?}",
            t.dtype
        )));
    }
    if let Some(s) = shape {
        if t.shape != s {
            return Err(PyValueError::new_err(format!(
                "{name}: shape {:?} != {s:?}",
                t.shape
            )));
        }
    }
    if t.shape.iter().product::<usize>() < min_elems {
        return Err(PyValueError::new_err(format!(
            "{name}: needs >= {min_elems} elements"
        )));
    }
    if !contiguous(t) {
        return Err(PyValueError::new_err(format!("{name} must be contiguous")));
    }
    Ok(())
}

/// Routed-expert forward for M decode tokens. Shapes (E experts, H hidden,
/// I intermediate, K top-k, P = M*K):
///   x bf16 [M, H]; topk_ids int32/int64 [M, K]; topk_w f32 [M, K]
///   w13 uint8 [E, 2I, H/2] ([up; gate]); w13_sf e4m3/uint8 [E, r128(2I), r4(H/16)]
///   w2 uint8 [E, H, I/2]; w2_sf [E, r128(H), r4(I/16)]; g1, g2 f32 [E]
///   ws_* preallocated scratch; out bf16 [M, H]
/// a1_gscale / a2_gscale: the layer's single activation quant multipliers.
/// id_base: expert parallel offset (vLLM linear expert_map: ep_rank * E);
/// topk_ids are GLOBAL ids, pairs outside [id_base, id_base + E) contribute
/// nothing on this rank (vLLM all-reduces the partial outputs afterwards).
/// mode: launch plan (module doc); ws_inter is written only by mode 0,
/// ws_y only by modes 0 and 1. tune: module doc.
#[pyfunction]
#[pyo3(signature = (x, topk_ids, topk_w, w13, w13_sf, g1, w2, w2_sf, g2, ws_aq, ws_asf, ws_inter, ws_hq, ws_hsf, ws_y, ws_route, out, a1_gscale, a2_gscale, act, id_base, mode, tune, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_moe_cuda<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    topk_ids: &Bound<'py, PyAny>,
    topk_w: &Bound<'py, PyAny>,
    w13: &Bound<'py, PyAny>,
    w13_sf: &Bound<'py, PyAny>,
    g1: &Bound<'py, PyAny>,
    w2: &Bound<'py, PyAny>,
    w2_sf: &Bound<'py, PyAny>,
    g2: &Bound<'py, PyAny>,
    ws_aq: &Bound<'py, PyAny>,
    ws_asf: &Bound<'py, PyAny>,
    ws_inter: &Bound<'py, PyAny>,
    ws_hq: &Bound<'py, PyAny>,
    ws_hsf: &Bound<'py, PyAny>,
    ws_y: &Bound<'py, PyAny>,
    ws_route: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    a1_gscale: f32,
    a2_gscale: f32,
    act: u32,
    id_base: u32,
    mode: u32,
    tune: u32,
    stream_ptr: usize,
) -> PyResult<()> {
    if mode > 2 {
        return Err(PyValueError::new_err(format!("mode {mode} not in 0..=2")));
    }
    if tune & 0xC0FE_0000 != 0 {
        return Err(PyValueError::new_err(format!("tune {tune:#x}: unknown bits")));
    }
    let (la1, la2, pf2, skip) = (tune & 0xFF, (tune >> 8) & 0xFF, (tune >> 16) & 1, tune >> 24);
    let x = info("x", x)?;
    let ids = info("topk_ids", topk_ids)?;
    let tw = info("topk_w", topk_w)?;
    let w13 = info("w13", w13)?;
    let w13s = info("w13_sf", w13_sf)?;
    let g1 = info("g1", g1)?;
    let w2 = info("w2", w2)?;
    let w2s = info("w2_sf", w2_sf)?;
    let g2 = info("g2", g2)?;
    let aq = info("ws_aq", ws_aq)?;
    let asf = info("ws_asf", ws_asf)?;
    let inter = info("ws_inter", ws_inter)?;
    let hq = info("ws_hq", ws_hq)?;
    let hsf = info("ws_hsf", ws_hsf)?;
    let y = info("ws_y", ws_y)?;
    let route = info("ws_route", ws_route)?;
    let out = info("out", out)?;
    if x.shape.len() != 2 || ids.shape.len() != 2 || w13.shape.len() != 3 || w2.shape.len() != 3 {
        return Err(PyValueError::new_err("x/topk_ids 2-D, w13/w2 3-D required"));
    }
    let (m, h) = (x.shape[0], x.shape[1]);
    let (e, two_i) = (w13.shape[0], w13.shape[1]);
    let i = two_i / 2;
    let k = ids.shape[1];
    let p = m * k;
    if m == 0 {
        return Ok(());
    }
    // I % 32: moe_fc1_quant's CTA = 32 cols of I = two whole NVFP4 blocks
    if e > 256 || h % 64 != 0 || i % 64 != 0 || k == 0 {
        return Err(PyValueError::new_err(format!(
            "unsupported MoE shape E={e} H={h} I={i} topk={k} (E<=256, H,I % 64 == 0, topk >= 1)"
        )));
    }
    need("x", &x, &["torch.bfloat16"], Some(&[m, h]), 0)?;
    need(
        "topk_ids",
        &ids,
        &["torch.int32", "torch.int64"],
        Some(&[m, k]),
        0,
    )?;
    need("topk_w", &tw, &["torch.float32"], Some(&[m, k]), 0)?;
    need("w13", &w13, &["torch.uint8"], Some(&[e, 2 * i, h / 2]), 0)?;
    let (r13, c13) = ((2 * i).div_ceil(128) * 128, (h / 16).div_ceil(4) * 4);
    need(
        "w13_sf",
        &w13s,
        &["torch.float8_e4m3fn", "torch.uint8"],
        None,
        e * r13 * c13,
    )?;
    need("w2", &w2, &["torch.uint8"], Some(&[e, h, i / 2]), 0)?;
    let (r2, c2) = (h.div_ceil(128) * 128, (i / 16).div_ceil(4) * 4);
    need(
        "w2_sf",
        &w2s,
        &["torch.float8_e4m3fn", "torch.uint8"],
        None,
        e * r2 * c2,
    )?;
    if w13s.shape.iter().product::<usize>() != e * r13 * c13
        || w2s.shape.iter().product::<usize>() != e * r2 * c2
    {
        return Err(PyValueError::new_err(
            "w13_sf / w2_sf are not the per-expert padded swizzled layout",
        ));
    }
    need("g1", &g1, &["torch.float32"], Some(&[e]), 0)?;
    need("g2", &g2, &["torch.float32"], Some(&[e]), 0)?;
    need("ws_aq", &aq, &["torch.uint8"], None, m * h / 2)?;
    need("ws_asf", &asf, &["torch.uint8"], None, m * h / 16)?;
    need("ws_inter", &inter, &["torch.float32"], None, p * i)?;
    need("ws_hq", &hq, &["torch.uint8"], None, p * i / 2)?;
    need("ws_hsf", &hsf, &["torch.uint8"], None, p * i / 16)?;
    need("ws_y", &y, &["torch.float32"], None, p * h)?;
    need("ws_route", &route, &["torch.int32"], None, 3 * e + p)?;
    need("out", &out, &["torch.bfloat16"], Some(&[m, h]), 0)?;
    for (nm, t) in [
        ("topk_ids", &ids),
        ("topk_w", &tw),
        ("w13", &w13),
        ("w13_sf", &w13s),
        ("g1", &g1),
        ("w2", &w2),
        ("w2_sf", &w2s),
        ("g2", &g2),
        ("ws_aq", &aq),
        ("ws_asf", &asf),
        ("ws_inter", &inter),
        ("ws_hq", &hq),
        ("ws_hsf", &hsf),
        ("ws_y", &y),
        ("ws_route", &route),
        ("out", &out),
    ] {
        if t.device != x.device {
            return Err(PyValueError::new_err(format!("{nm} on another device")));
        }
    }
    let ids_i64 = u32::from(ids.dtype == "torch.int64");
    // route_par smem: 16 warp totals + u16 ids padded to 8 pairs
    let route_par_smem = 64 + 2 * p.div_ceil(8) * 8;
    if route_par_smem > crate::oxide::MAX_DYN_SMEM as usize {
        return Err(PyValueError::new_err(format!("M*topk={p} pairs exceed the route smem")));
    }
    let slots = e.min(p);
    let r = route.ptr;
    let (slot_expert, slot_off, slot_cnt, pair_list) =
        (r, r + 4 * e as u64, r + 8 * e as u64, r + 12 * e as u64);
    let u = |v: usize| Arg::U32(v as u32);
    let route_args = [
        Arg::Ptr(ids.ptr),
        Arg::U32(ids_i64),
        Arg::U32(id_base),
        u(p),
        u(e),
        u(slots),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
    ];
    let qx_args = [
        Arg::Ptr(x.ptr),
        Arg::U32(0),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        u(m),
        u(h),
        u(x.stride[0]),
        Arg::F32(a1_gscale),
    ];
    let fc1_args = [
        Arg::Ptr(inter.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        Arg::Ptr(w13.ptr),
        Arg::Ptr(w13s.ptr),
        Arg::Ptr(g1.ptr),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        u(k),
        u(h),
        u(i),
        Arg::U32(act),
        Arg::U32(la1),
        Arg::U32(pf2),
        Arg::Ptr(w2.ptr),
        Arg::Ptr(w2s.ptr),
    ];
    let qh_args = [
        Arg::Ptr(inter.ptr),
        Arg::U32(1),
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hsf.ptr),
        u(p),
        u(i),
        u(i),
        Arg::F32(a2_gscale),
    ];
    let fc2_args = [
        Arg::Ptr(y.ptr),
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hsf.ptr),
        Arg::Ptr(w2.ptr),
        Arg::Ptr(w2s.ptr),
        Arg::Ptr(g2.ptr),
        Arg::Ptr(tw.ptr),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        u(h),
        u(i),
        Arg::U32(la2),
    ];
    let comb_args = [
        Arg::Ptr(out.ptr),
        Arg::Ptr(y.ptr),
        Arg::Ptr(ids.ptr),
        Arg::U32(ids_i64),
        Arg::U32(id_base),
        u(m),
        u(k),
        u(h),
        u(e),
        u(out.stride[0]),
    ];
    // fused launch plan (modes 1, 2)
    let rq_args = [
        Arg::Ptr(ids.ptr),
        Arg::U32(ids_i64),
        Arg::U32(id_base),
        u(p),
        u(e),
        u(slots),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        Arg::Ptr(x.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        u(m),
        u(h),
        u(x.stride[0]),
        Arg::F32(a1_gscale),
    ];
    let fq_args = [
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hsf.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        Arg::Ptr(w13.ptr),
        Arg::Ptr(w13s.ptr),
        Arg::Ptr(g1.ptr),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        u(k),
        u(h),
        u(i),
        Arg::U32(act),
        Arg::F32(a2_gscale),
        Arg::U32(la1),
        Arg::U32(pf2),
        Arg::Ptr(w2.ptr),
        Arg::Ptr(w2s.ptr),
    ];
    let fc_args = [
        Arg::Ptr(out.ptr),
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hsf.ptr),
        Arg::Ptr(w2.ptr),
        Arg::Ptr(w2s.ptr),
        Arg::Ptr(g2.ptr),
        Arg::Ptr(tw.ptr),
        Arg::Ptr(ids.ptr),
        Arg::U32(ids_i64),
        Arg::U32(id_base),
        u(k),
        u(h),
        u(i),
        u(e),
        u(out.stride[0]),
        Arg::U32(la2),
    ];
    let route_smem = (2 * e * 4) as u32;
    let mm = m as u32;
    // (entry, grid, threads, dynamic smem, args)
    let legacy: [(&str, (u32, u32), u32, u32, &[Arg]); 6] = [
        ("moe_route", (1, 1), 256, route_smem, &route_args),
        ("moe_quant_rows", ((h / 16).div_ceil(128) as u32, mm), 128, 0, &qx_args),
        ("moe_fc1", ((i / 32) as u32, slots as u32), 128, 0, &fc1_args),
        ("moe_quant_rows", ((i / 16).div_ceil(128) as u32, p as u32), 128, 0, &qh_args),
        ("moe_fc2", ((h / 32) as u32, slots as u32), 128, 0, &fc2_args),
        ("moe_combine", ((m * h).div_ceil(256) as u32, 1), 256, 0, &comb_args),
    ];
    let front: [(&str, (u32, u32), u32, u32, &[Arg]); 2] = [
        ("moe_route_quant", (1 + (m * h / 16).div_ceil(256) as u32, 1), 256, route_par_smem as u32, &rq_args),
        ("moe_fc1_quant", ((i / 32) as u32, slots as u32), 128, 16 * 32 * 4, &fq_args),
    ];
    let tail: Vec<(&str, (u32, u32), u32, u32, &[Arg])> = if mode == 2 {
        let nw = k.min(16) as u32;
        vec![("moe_fc2_combine", ((h / 8) as u32, mm), 32 * nw, (k * 8 * 4) as u32, &fc_args)]
    } else {
        legacy[4..].to_vec()
    };
    let plan: Vec<(&str, (u32, u32), u32, u32, &[Arg])> =
        if mode == 0 { legacy.to_vec() } else { front.iter().cloned().chain(tail).collect() };
    let plan: Vec<_> = plan
        .into_iter()
        .enumerate()
        .filter(|(n, _)| skip >> n & 1 == 0)
        .map(|(_, l)| l)
        .collect();
    let ordinal = x.device;
    py.detach(move || {
        crate::guard_py("nvfp4_moe_cuda", move || {
            let e_ = |e: String| PyRuntimeError::new_err(format!("nvfp4 moe launch: {e}"));
            for (entry, grid, threads, smem, args) in plan {
                let f = function(FAMILY, entry, ordinal).map_err(PyRuntimeError::new_err)?;
                launch(f, (grid.0, grid.1, 1), (threads, 1, 1), smem, stream_ptr, args).map_err(e_)?;
            }
            Ok(())
        })
    })
}

/// Per-entry resource facts of the loaded nvfp4_moe cubin, for the sweep's
/// occupancy report: [(entry, regs/thread, local bytes/thread, max resident
/// CTAs per SM at `threads` threads + `smem` dynamic bytes)] for each
/// (entry, threads, smem) asked.
#[pyfunction]
pub fn nvfp4_moe_attrs(
    entries: Vec<(String, u32, u32)>,
    device_ordinal: usize,
) -> PyResult<Vec<(String, i32, i32, i32)>> {
    use cuda_core::sys;
    let mut out = Vec::new();
    for (entry, threads, smem) in entries {
        let f = function(FAMILY, &entry, device_ordinal).map_err(PyRuntimeError::new_err)?
            as sys::CUfunction;
        let (mut regs, mut local, mut ctas) = (0i32, 0i32, 0i32);
        // SAFETY: `f` is a live CUfunction of a loaded module.
        unsafe {
            sys::cuFuncGetAttribute(&mut regs, sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS, f);
            sys::cuFuncGetAttribute(
                &mut local,
                sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                f,
            );
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut ctas, f, threads as i32, smem as usize);
        }
        out.push((entry, regs, local, ctas));
    }
    Ok(out)
}
