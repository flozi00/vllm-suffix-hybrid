// SPDX-License-Identifier: Apache-2.0
//! NVFP4 W4A4 decode GEMM, host side (feature `oxide-kernels`): launches
//! kernels-oxide/nvfp4_gemm (sm_120a SASS from ptxas 13.0; block-scaled FP4
//! mma kind::mxf4nvf4) on torch's stream for M <= 64. The split-K policy
//! lives in Python (suffix_hybrid/kernels/nvfp4_gemm.py `plan`, CPU-tested,
//! printed by the oracle/bench); this side validates it and launches.

use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "nvfp4_gemm";
const QUANT: &str = "nvfp4_quant_act";
/// GEMM entry per m16-tile count (tiles = ceil(M/16)).
const GEMM: [&str; 4] = ["nvfp4_gemm_t1", "nvfp4_gemm_t2", "nvfp4_gemm_t3", "nvfp4_gemm_t4"];
const REDUCE: &str = "nvfp4_splitk_reduce";
/// Single-launch entries (quant prologue and/or last-CTA split-K fixup).
const GEMM_F: [&str; 4] = ["nvfp4_gemm_f1", "nvfp4_gemm_f2", "nvfp4_gemm_f3", "nvfp4_gemm_f4"];
/// `fused` flag bits (kernel FUSE_QUANT / FUSE_REDUCE).
pub const FUSE_QUANT: u32 = 1;
pub const FUSE_REDUCE: u32 = 2;
/// Split-K fixup staging budget in floats (ch splits x M x 32 + the M x 32
/// running sums <= this): see `fused_layout`.
pub const STAGE_FLOATS: usize = 4096;

struct T {
    ptr: u64,
    shape: Vec<usize>,
    stride: Vec<usize>,
    dtype: String,
    device: usize,
}

fn info(name: &str, o: &Bound<'_, PyAny>) -> PyResult<T> {
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

fn need(name: &str, t: &T, dtype: &str, shape: &[usize], contiguous: bool) -> PyResult<()> {
    if t.dtype != dtype || t.shape != shape {
        return Err(PyValueError::new_err(format!(
            "{name}: got {} {:?}, need {dtype} {shape:?}",
            t.dtype, t.shape
        )));
    }
    let mut want = 1;
    for (d, s) in t.shape.iter().zip(&t.stride).rev() {
        if contiguous && *d > 1 && *s != want {
            return Err(PyValueError::new_err(format!("{name} must be contiguous")));
        }
        want *= d;
    }
    if t.stride.last().copied().unwrap_or(1) != 1 {
        return Err(PyValueError::new_err(format!(
            "{name}: last dim must be dense"
        )));
    }
    Ok(())
}

fn u32_of(name: &str, v: usize) -> PyResult<u32> {
    u32::try_from(v).map_err(|_| PyValueError::new_err(format!("{name}={v} exceeds u32")))
}

pub const MAX_M: usize = 64;
pub const MAX_SPLITS: usize = 64;

/// m16 tiles for M rows (the GEMM entry and quant grid depend on M only).
pub fn tiles_for(m: usize) -> usize {
    m.div_ceil(16)
}

/// (k64 steps per split, splits actually launched): no empty trailing split.
pub fn split_ranges(k: usize, splits: usize) -> (usize, usize) {
    let steps = k / 64;
    let kps = steps.div_ceil(splits);
    (kps, steps.div_ceil(kps))
}

/// Dynamic smem bytes and fixup chunk `ch` (splits staged per round) of an
/// nvfp4_gemm_f launch (mirrors nvfp4_gemm.py `fused_layout`): 16 B ticket
/// word + max(quant prologue A = m * (kps*32 + 16) + m * 4*(kps|1) scale
/// bytes, fixup staging (ch + 1) * m * 32 f32).
pub fn fused_layout(m: usize, kps: usize, splits: usize, flags: u32) -> (usize, usize) {
    let q = if flags & FUSE_QUANT != 0 { m * (kps * 32 + 16) + m * 4 * (kps | 1) } else { 0 };
    let ch = (STAGE_FLOATS / (32 * m)).saturating_sub(1).clamp(1, splits);
    let r = if flags & FUSE_REDUCE != 0 && splits > 1 { (ch + 1) * m * 32 * 4 } else { 0 };
    (16 + q.max(r), ch)
}

struct Gemm {
    a_ptr: u64,
    a_stride: usize,
    a_rows: usize,
    asf_ptr: u64,
    asf_mode: u32,
}

#[allow(clippy::too_many_arguments)]
fn validate_w(
    w: &T,
    wsf: &T,
    out: &T,
    partial: &T,
    m: usize,
    k: usize,
    splits: usize,
) -> PyResult<usize> {
    let n = w.shape[0];
    if !(1..=MAX_M).contains(&m) || k == 0 || k % 64 != 0 || n == 0 || n % 16 != 0 {
        return Err(PyValueError::new_err(format!(
            "NVFP4 decode GEMM needs 1 <= M <= {MAX_M}, K % 64 == 0, N % 16 == 0 (M={m} K={k} N={n})"
        )));
    }
    need("w", w, "torch.uint8", &[n, k / 2], true)?;
    let sf_rows = n.div_ceil(128) * 128;
    let sf_cols = (k / 16).div_ceil(4) * 4;
    if wsf.shape.iter().product::<usize>() != sf_rows * sf_cols
        || !(wsf.dtype == "torch.float8_e4m3fn" || wsf.dtype == "torch.uint8")
    {
        return Err(PyValueError::new_err(format!(
            "w_sf must hold the swizzled [{sf_rows}, {sf_cols}] e4m3 scales (got {} {:?})",
            wsf.dtype, wsf.shape
        )));
    }
    need("out", out, "torch.bfloat16", &[m, n], false)?;
    if !(1..=MAX_SPLITS).contains(&splits) || splits > k / 64 {
        return Err(PyValueError::new_err(format!(
            "splits={splits} out of range"
        )));
    }
    let (_, launched) = split_ranges(k, splits);
    let need_part = if launched > 1 { launched * m * n } else { 0 };
    if partial.dtype != "torch.float32" || partial.shape.iter().product::<usize>() < need_part {
        return Err(PyValueError::new_err(format!(
            "partial workspace must be float32 with >= {need_part} elements (splits x M x N)"
        )));
    }
    Ok(n)
}

/// bf16-route extras for the fused entries: x, its row stride, global scale.
struct Fused {
    flags: u32,
    x: (u64, usize, f32),
    counters: u64,
}

#[allow(clippy::too_many_arguments)]
fn run(
    py: Python<'_>,
    quant: Option<[Arg; 7]>,
    g: Gemm,
    w: &T,
    wsf: &T,
    out: &T,
    partial: &T,
    m: usize,
    n: usize,
    k: usize,
    splits: usize,
    alpha: f32,
    fz: Fused,
    stream_ptr: usize,
    ordinal: usize,
) -> PyResult<()> {
    let (kps, splits) = split_ranges(k, splits);
    let tiles = tiles_for(m);
    let part_ptr = if splits > 1 { partial.ptr } else { 0 };
    let (smem, ch) = fused_layout(m, kps, splits, fz.flags);
    if fz.flags != 0 && smem > crate::oxide::MAX_DYN_SMEM as usize {
        return Err(PyValueError::new_err(format!(
            "fused NVFP4 GEMM needs {smem} B smem (> {}): plan must fall back",
            crate::oxide::MAX_DYN_SMEM
        )));
    }
    let mut gemm_args = vec![
        Arg::Ptr(out.ptr),
        Arg::Ptr(part_ptr),
        Arg::Ptr(g.a_ptr),
        Arg::Ptr(g.asf_ptr),
        Arg::Ptr(w.ptr),
        Arg::Ptr(wsf.ptr),
        Arg::U32(m as u32),
        Arg::U32(u32_of("n", n)?),
        Arg::U32(u32_of("k", k)?),
        Arg::U32(u32_of("out_stride", out.stride[0])?),
        Arg::U32(u32_of("a_stride", g.a_stride)?),
        Arg::U32(g.a_rows as u32),
        Arg::U32(g.asf_mode),
        Arg::U32(kps as u32),
        Arg::F32(alpha),
    ];
    if fz.flags != 0 {
        gemm_args.extend([
            Arg::Ptr(fz.x.0),
            Arg::U32(u32_of("x_stride", fz.x.1)?),
            Arg::F32(fz.x.2),
            Arg::Ptr(fz.counters),
            Arg::U32(fz.flags),
            Arg::U32(ch as u32),
        ]);
    }
    let red_args = [
        Arg::Ptr(out.ptr),
        Arg::Ptr(partial.ptr),
        Arg::U32(m as u32),
        Arg::U32(n as u32),
        Arg::U32(splits as u32),
        Arg::U32(out.stride[0] as u32),
        Arg::F32(alpha),
    ];
    let quant = if fz.flags & FUSE_QUANT != 0 { None } else { quant };
    let reduce = splits > 1 && fz.flags & FUSE_REDUCE == 0;
    let entry = if fz.flags != 0 { GEMM_F[tiles - 1] } else { GEMM[tiles - 1] };
    let smem = if fz.flags != 0 { smem as u32 } else { 0 };
    let qgrid = ((k / 16).div_ceil(128) as u32, (16 * tiles) as u32, 1);
    let ggrid = (n.div_ceil(32) as u32, splits as u32, 1);
    let rgrid = ((m * n).div_ceil(256) as u32, 1, 1);
    py.detach(move || {
        crate::guard_py("nvfp4_gemm_cuda", move || {
            let e = |e: String| PyRuntimeError::new_err(format!("nvfp4 gemm launch: {e}"));
            if let Some(qa) = &quant {
                let q = function(FAMILY, QUANT, ordinal).map_err(PyRuntimeError::new_err)?;
                launch(q, qgrid, (128, 1, 1), 0, stream_ptr, qa).map_err(e)?;
            }
            let gm = function(FAMILY, entry, ordinal).map_err(PyRuntimeError::new_err)?;
            launch(gm, ggrid, (128, 1, 1), smem, stream_ptr, &gemm_args).map_err(e)?;
            if reduce {
                let r = function(FAMILY, REDUCE, ordinal).map_err(PyRuntimeError::new_err)?;
                launch(r, rgrid, (256, 1, 1), 0, stream_ptr, &red_args).map_err(e)?;
            }
            Ok(())
        })
    })
}

/// `fused` flags + `counters` (int32 >= ceil(N/32), zeroed at load, left
/// zeroed by every launch) -> the kernel's extras; loud on misuse.
fn fused_args(
    fused: u32,
    counters: Option<&Bound<'_, PyAny>>,
    n: usize,
    splits: usize,
    k: usize,
    device: usize,
    x: (u64, usize, f32),
) -> PyResult<Fused> {
    if fused & !(FUSE_QUANT | FUSE_REDUCE) != 0 {
        return Err(PyValueError::new_err(format!("fused={fused}: unknown flag bits")));
    }
    let mut cptr = 0;
    if fused & FUSE_REDUCE != 0 && split_ranges(k, splits).1 > 1 {
        let c = counters
            .ok_or_else(|| PyValueError::new_err("fused split-K reduce needs `counters`"))
            .and_then(|c| info("counters", c))?;
        if c.dtype != "torch.int32" || c.shape.iter().product::<usize>() < n.div_ceil(32) {
            return Err(PyValueError::new_err(format!(
                "counters must be int32 with >= ceil(N/32) = {} elements (zeroed)",
                n.div_ceil(32)
            )));
        }
        if c.device != device {
            return Err(PyValueError::new_err("counters on another device"));
        }
        cptr = c.ptr;
    }
    Ok(Fused { flags: fused, x, counters: cptr })
}

/// y = alpha * nvfp4(x) @ w^T for M <= 64 (decode), bf16 activation path:
/// our quant (`aq` uint8 >= [16*tiles, K/2], `asf` uint8 >= [16*tiles, K/16]
/// scratch, tiles = ceil(M/16)) + GEMM (+ split-K reduce into `partial`
/// float32 >= splits*M*N). x bf16 [M, K]
/// (row stride allowed); w uint8 [N, K/2] (vLLM packed e2m1, low nibble =
/// even k); w_sf swizzled 128x4 e4m3; out bf16 [M, N]. gscale = x global
/// scale (vLLM input_global_scale_inv), alpha = vLLM layer.alpha. All
/// buffers torch-owned and preallocated: no allocation, no host sync
/// (CUDA-graph capturable).
#[pyfunction]
#[pyo3(signature = (x, w, w_sf, aq, asf, partial, out, gscale, alpha, splits, stream_ptr,
                    fused=0, counters=None))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_cuda<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    w_sf: &Bound<'py, PyAny>,
    aq: &Bound<'py, PyAny>,
    asf: &Bound<'py, PyAny>,
    partial: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    gscale: f32,
    alpha: f32,
    splits: usize,
    stream_ptr: usize,
    fused: u32,
    counters: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let x = info("x", x)?;
    let w = info("w", w)?;
    let wsf = info("w_sf", w_sf)?;
    let aq = info("aq", aq)?;
    let asf = info("asf", asf)?;
    let partial = info("partial", partial)?;
    let out = info("out", out)?;
    if x.shape.len() != 2 || w.shape.len() != 2 {
        return Err(PyValueError::new_err("x and w must be 2-D"));
    }
    let (m, k) = (x.shape[0], x.shape[1]);
    let n = validate_w(&w, &wsf, &out, &partial, m, k, splits)?;
    need("x", &x, "torch.bfloat16", &[m, k], false)?;
    let rows = 16 * tiles_for(m);
    if aq.dtype != "torch.uint8" || aq.shape.iter().product::<usize>() < rows * k / 2 {
        return Err(PyValueError::new_err(format!(
            "aq must be uint8 with >= {rows}*K/2 bytes"
        )));
    }
    if asf.dtype != "torch.uint8" || asf.shape.iter().product::<usize>() < rows * k / 16 {
        return Err(PyValueError::new_err(format!(
            "asf must be uint8 with >= {rows}*K/16 bytes"
        )));
    }
    for (nm, t) in [
        ("w", &w),
        ("w_sf", &wsf),
        ("aq", &aq),
        ("asf", &asf),
        ("partial", &partial),
        ("out", &out),
    ] {
        if t.device != x.device {
            return Err(PyValueError::new_err(format!("{nm} on another device")));
        }
    }
    let quant = [
        Arg::Ptr(x.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        Arg::U32(m as u32),
        Arg::U32(u32_of("k", k)?),
        Arg::U32(u32_of("x_stride", x.stride[0])?),
        Arg::F32(gscale),
    ];
    let g = Gemm {
        a_ptr: aq.ptr,
        a_stride: k / 2,
        a_rows: rows,
        asf_ptr: asf.ptr,
        asf_mode: 0,
    };
    let fz = fused_args(fused, counters, n, splits, k, x.device, (x.ptr, x.stride[0], gscale))?;
    run(
        py,
        Some(quant),
        g,
        &w,
        &wsf,
        &out,
        &partial,
        m,
        n,
        k,
        splits,
        alpha,
        fz,
        stream_ptr,
        x.device,
    )
}

/// Same GEMM on an activation vLLM already quantized (fused SiLU*mul /
/// RMSNorm + NVFP4 quant): xq uint8 [M, K/2], xsf e4m3/uint8 swizzled 128x4
/// ([round128(M), round4(K/16)]). `fused` may only carry FUSE_REDUCE here
/// (the activation is already quantized).
#[pyfunction]
#[pyo3(signature = (xq, xsf, w, w_sf, partial, out, alpha, splits, stream_ptr, fused=0,
                    counters=None))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_q_cuda<'py>(
    py: Python<'py>,
    xq: &Bound<'py, PyAny>,
    xsf: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    w_sf: &Bound<'py, PyAny>,
    partial: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    alpha: f32,
    splits: usize,
    stream_ptr: usize,
    fused: u32,
    counters: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    if fused & FUSE_QUANT != 0 {
        return Err(PyValueError::new_err("nvfp4_gemm_q_cuda: FUSE_QUANT needs the bf16 route"));
    }
    let xq = info("xq", xq)?;
    let xsf = info("xsf", xsf)?;
    let w = info("w", w)?;
    let wsf = info("w_sf", w_sf)?;
    let partial = info("partial", partial)?;
    let out = info("out", out)?;
    if xq.shape.len() != 2 || w.shape.len() != 2 {
        return Err(PyValueError::new_err("xq and w must be 2-D"));
    }
    let (m, k) = (xq.shape[0], xq.shape[1] * 2);
    let n = validate_w(&w, &wsf, &out, &partial, m, k, splits)?;
    need("xq", &xq, "torch.uint8", &[m, k / 2], false)?;
    let sf_rows = m.div_ceil(128) * 128;
    let sf_cols = (k / 16).div_ceil(4) * 4;
    if xsf.shape.iter().product::<usize>() < sf_rows * sf_cols
        || !(xsf.dtype == "torch.float8_e4m3fn" || xsf.dtype == "torch.uint8")
    {
        return Err(PyValueError::new_err(format!(
            "xsf must hold swizzled [{sf_rows}, {sf_cols}] e4m3 scales (got {} {:?})",
            xsf.dtype, xsf.shape
        )));
    }
    for (nm, t) in [
        ("xsf", &xsf),
        ("w", &w),
        ("w_sf", &wsf),
        ("partial", &partial),
        ("out", &out),
    ] {
        if t.device != xq.device {
            return Err(PyValueError::new_err(format!("{nm} on another device")));
        }
    }
    let g = Gemm {
        a_ptr: xq.ptr,
        a_stride: xq.stride[0],
        a_rows: m,
        asf_ptr: xsf.ptr,
        asf_mode: 1,
    };
    let fz = fused_args(fused, counters, n, splits, k, xq.device, (0, 0, 0.0))?;
    run(
        py, None, g, &w, &wsf, &out, &partial, m, n, k, splits, alpha, fz, stream_ptr, xq.device,
    )
}

#[cfg(test)]
mod tests {
    use super::{fused_layout, split_ranges, tiles_for, FUSE_QUANT, FUSE_REDUCE, GEMM};

    #[test]
    fn tiles_pick_the_entry_from_m_only() {
        let pick = |m: usize| GEMM[tiles_for(m) - 1];
        assert_eq!(pick(1), "nvfp4_gemm_t1");
        assert_eq!(pick(16), "nvfp4_gemm_t1");
        assert_eq!(pick(17), "nvfp4_gemm_t2");
        assert_eq!(pick(40), "nvfp4_gemm_t3");
        assert_eq!(pick(64), "nvfp4_gemm_t4");
    }

    #[test]
    fn split_ranges_drop_empty_trailing_splits() {
        assert_eq!(split_ranges(10240, 69), (3, 54)); // hc down: 160 steps
        assert_eq!(split_ranges(3072, 10), (5, 10));
        assert_eq!(split_ranges(320, 2), (3, 2));
        assert_eq!(split_ranges(128, 1), (2, 1));
    }

    #[test]
    fn fused_layout_matches_python_plan() {
        // same numbers as tests/test_nvfp4_gemm.py::test_fused_layout
        let both = FUSE_QUANT | FUSE_REDUCE;
        assert_eq!(fused_layout(1, 3, 54, FUSE_REDUCE), (16 + 55 * 32 * 4, 54));
        assert_eq!(fused_layout(1, 3, 54, both), (16 + 55 * 32 * 4, 54));
        assert_eq!(fused_layout(16, 14, 3, both), (16 + 16 * (14 * 32 + 16) + 16 * 4 * 15, 3));
        assert_eq!(fused_layout(64, 8, 6, FUSE_REDUCE), (16 + 2 * 64 * 32 * 4, 1));
        assert_eq!(fused_layout(5, 40, 1, FUSE_QUANT), (16 + 5 * (40 * 32 + 16) + 5 * 4 * 41, 1));
        assert_eq!(fused_layout(17, 5, 2, 0), (16, 2));
    }
}
