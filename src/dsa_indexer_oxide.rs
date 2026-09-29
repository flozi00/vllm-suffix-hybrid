// SPDX-License-Identifier: Apache-2.0
//! DSA indexer logits host op (feature `oxide-kernels`): launches
//! kernels-oxide/dsa_indexer `dsa_logits` (sm_120a SASS, ptxas 13.0) on
//! torch's stream. Grid (splits, ceil(rows / group)) — a function of host
//! shapes only (no device reads): CUDA-graph capturable. Launch plan, shape
//! gating and the deep_gemm shim wiring: suffix_hybrid/kernels/dsa_indexer.py.

use crate::nvfp4_moe_oxide::{info, T};
use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "dsa_indexer";
/// = kernels-oxide/dsa_indexer T / PITCH / STAGE / MAX_G.
const TILE: usize = 64;
const PITCH: usize = 144;
const STAGE: usize = TILE * PITCH + TILE * 4;
const MAX_WARPS: usize = 12;
const D: usize = 128;

fn bad(msg: String) -> PyErr {
    PyValueError::new_err(format!("dsa_indexer: {msg}"))
}

fn want(name: &str, t: &T, dtypes: &[&str], dims: usize) -> PyResult<()> {
    if !dtypes.contains(&t.dtype.as_str()) || t.shape.len() != dims {
        return Err(bad(format!(
            "{name}: {} {:?} (want {dtypes:?}, {dims}-D)",
            t.dtype, t.shape
        )));
    }
    if dims > 0 && t.shape[dims - 1] > 1 && t.stride[dims - 1] != 1 {
        return Err(bad(format!("{name}: last dim must be contiguous")));
    }
    Ok(())
}

/// logits[r, n] for lo[r] <= n < hi[r] (module doc of the kernel crate).
///   out f32 [rows, >= n_cols] (row stride free); q e4m3 [rows, H, 128]
///   contiguous (H 32 | 64); weights f32 [>= rows, H]
///   paged (block_tables given): kv uint8 [num_blocks, 64, 132] contiguous
///     (split page: 64*128 e4m3 values, then 64 f32 scales), block_tables
///     int32 [*, >= blocks] with row = r / bt_div, k_scale None
///   contiguous: kv e4m3/uint8 [N, 128] (row stride % 16 == 0), k_scale f32
///     [N], n_cols == N
///   hi int32 [rows] (decode context_lens flat / prefill ke); lo int32 [rows]
///   or None (= 0).
#[pyfunction]
#[pyo3(signature = (out, q, weights, kv, k_scale, block_tables, lo, hi, n_cols, group, slices, bt_div, splits, min_tiles, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn dsa_indexer_logits_cuda<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    kv: &Bound<'py, PyAny>,
    k_scale: Option<&Bound<'py, PyAny>>,
    block_tables: Option<&Bound<'py, PyAny>>,
    lo: Option<&Bound<'py, PyAny>>,
    hi: &Bound<'py, PyAny>,
    n_cols: u32,
    group: u32,
    slices: u32,
    bt_div: u32,
    splits: u32,
    min_tiles: u32,
    stream_ptr: usize,
) -> PyResult<()> {
    let out = info("out", out)?;
    let q = info("q", q)?;
    let w = info("weights", weights)?;
    let kv = info("kv", kv)?;
    let hi = info("hi", hi)?;
    let f8 = ["torch.float8_e4m3fn", "torch.uint8"];
    want("q", &q, &f8, 3)?;
    let (rows, heads) = (q.shape[0], q.shape[1]);
    if q.shape[2] != D || (heads != 32 && heads != 64) || q.stride != [heads * D, D, 1] {
        return Err(bad(format!("q {:?} strides {:?}: need contiguous [rows, 32|64, 128]", q.shape, q.stride)));
    }
    want("out", &out, &["torch.float32"], 2)?;
    want("weights", &w, &["torch.float32"], 2)?;
    want("hi", &hi, &["torch.int32"], 1)?;
    if out.shape[0] != rows || out.shape[1] < n_cols as usize || w.shape[0] < rows || w.shape[1] != heads || hi.shape[0] != rows {
        return Err(bad(format!(
            "rows {rows}: out {:?} (n_cols {n_cols}), weights {:?}, hi {:?}",
            out.shape, w.shape, hi.shape
        )));
    }
    let lo = lo.map(|t| info("lo", t)).transpose()?;
    if let Some(l) = &lo {
        want("lo", l, &["torch.int32"], 1)?;
        if l.shape[0] != rows {
            return Err(bad(format!("lo {:?} != rows {rows}", l.shape)));
        }
    }
    let nw = (group * slices) as usize;
    if group == 0 || slices == 0 || nw > MAX_WARPS || splits == 0 || bt_div == 0 {
        return Err(bad(format!("group {group} x slices {slices} (<= {MAX_WARPS} warps), splits {splits}, bt_div {bt_div}")));
    }
    // (sc, bt, paged, bt_stride, ps_v, ps_s, pitch)
    let (sc, bt, paged, bt_stride, ps_v, ps_s, pitch) = match block_tables {
        Some(b) => {
            let b = info("block_tables", b)?;
            want("block_tables", &b, &["torch.int32"], 2)?;
            want("kv", &kv, &["torch.uint8"], 3)?;
            if kv.shape[1] != TILE || kv.shape[2] != D + 4 || kv.stride != [TILE * (D + 4), D + 4, 1] {
                return Err(bad(format!("paged kv {:?} strides {:?}: need contiguous [blocks, 64, 132]", kv.shape, kv.stride)));
            }
            if b.device != q.device || k_scale.is_some() {
                return Err(bad("paged: block_tables on another device / k_scale given".into()));
            }
            let ps = (TILE * (D + 4)) as u32;
            (kv.ptr + (TILE * D) as u64, b.ptr, 1u32, b.stride[0] as u32, ps, ps, D as u32)
        }
        None => {
            let s = info("k_scale", k_scale.ok_or_else(|| bad("contiguous kv needs k_scale".into()))?)?;
            want("k_scale", &s, &["torch.float32"], 1)?;
            want("kv", &kv, &f8, 2)?;
            let n = kv.shape[0];
            let pitch = kv.stride[0];
            if kv.shape[1] != D || s.shape[0] != n || n_cols as usize != n || pitch % 16 != 0 || s.device != q.device {
                return Err(bad(format!("contiguous kv {:?} stride {pitch}, k_scale {:?}, n_cols {n_cols}", kv.shape, s.shape)));
            }
            (s.ptr, 0u64, 0u32, 0u32, (TILE * pitch) as u32, (TILE * 4) as u32, pitch as u32)
        }
    };
    for (nm, t) in [("out", &out), ("weights", &w), ("kv", &kv), ("hi", &hi)] {
        if t.device != q.device {
            return Err(bad(format!("{nm} on another device")));
        }
    }
    if lo.as_ref().is_some_and(|l| l.device != q.device) {
        return Err(bad("lo on another device".into()));
    }
    // cp.async 16 B / 4 B sources, u32 Q fragment loads
    if kv.ptr % 16 != 0 || sc % 4 != 0 || q.ptr % 4 != 0 {
        return Err(bad("kv not 16-byte / k_scale or q not 4-byte aligned".into()));
    }
    if rows == 0 || n_cols == 0 {
        return Ok(());
    }
    let groups = rows.div_ceil(group as usize) as u32;
    let u = |v: usize| Arg::U32(v as u32);
    let args = vec![
        Arg::Ptr(out.ptr),
        Arg::Ptr(q.ptr),
        Arg::Ptr(w.ptr),
        Arg::Ptr(kv.ptr),
        Arg::Ptr(sc),
        Arg::Ptr(bt),
        Arg::Ptr(lo.as_ref().map_or(0, |l| l.ptr)),
        Arg::Ptr(hi.ptr),
        u(rows),
        Arg::U32(group),
        Arg::U32(slices),
        u(heads),
        Arg::U32(n_cols),
        u(out.stride[0]),
        u(w.stride[0]),
        Arg::U32(paged),
        Arg::U32(bt_div),
        Arg::U32(bt_stride),
        Arg::U32(ps_v),
        Arg::U32(ps_s),
        Arg::U32(pitch),
        Arg::U32(splits),
        Arg::U32(min_tiles),
        Arg::U32(u32::from(lo.is_some())),
    ];
    let ordinal = q.device;
    py.detach(move || {
        crate::guard_py("dsa_indexer_logits_cuda", move || {
            let f = function(FAMILY, "dsa_logits", ordinal).map_err(PyRuntimeError::new_err)?;
            launch(f, (splits, groups, 1), (32 * nw as u32, 1, 1), (2 * STAGE) as u32, stream_ptr, &args)
                .map_err(|e| PyRuntimeError::new_err(format!("dsa_indexer launch: {e}")))
        })
    })
}
