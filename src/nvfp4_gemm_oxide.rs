// SPDX-License-Identifier: Apache-2.0
//! NVFP4 W4A4 decode-GEMM capability spike, host side (feature
//! `oxide-kernels`): launches kernels-oxide/nvfp4_gemm (sm_120a SASS from
//! ptxas 13.0; block-scaled FP4 mma kind::mxf4nvf4) on torch's stream.
//! Oracle/bench: suffix_hybrid/kernels/nvfp4_gemm.py.

use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "nvfp4_gemm";
const QUANT: &str = "nvfp4_quant_act";
const GEMM: &str = "nvfp4_gemm_m16";

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

/// y = alpha * nvfp4(x) @ w^T for M <= 16 (decode). x bf16 [M, K] (row
/// stride allowed); w uint8 [N, K/2] (vLLM packed e2m1, low nibble = even
/// k); w_sf float8_e4m3fn/uint8 swizzled 128x4 of padded [round128(N),
/// round4(K/16)]; aq uint8 [16, K/2], asf uint8 [16, K/16] scratch; out bf16
/// [M, N]. gscale = x global scale (vLLM input_global_scale), alpha = 1 /
/// (gscale * w_global_scale). N % 32 == 0, K % 64 == 0.
#[pyfunction]
#[pyo3(signature = (x, w, w_sf, aq, asf, out, gscale, alpha, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_cuda<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    w_sf: &Bound<'py, PyAny>,
    aq: &Bound<'py, PyAny>,
    asf: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    gscale: f32,
    alpha: f32,
    stream_ptr: usize,
) -> PyResult<()> {
    let x = info("x", x)?;
    let w = info("w", w)?;
    let wsf = info("w_sf", w_sf)?;
    let aq = info("aq", aq)?;
    let asf = info("asf", asf)?;
    let out = info("out", out)?;
    if x.shape.len() != 2 || w.shape.len() != 2 {
        return Err(PyValueError::new_err("x and w must be 2-D"));
    }
    let (m, k) = (x.shape[0], x.shape[1]);
    let n = w.shape[0];
    if !(1..=16).contains(&m) || k % 64 != 0 || n % 32 != 0 {
        return Err(PyValueError::new_err(format!(
            "spike supports 1 <= M <= 16, K % 64 == 0, N % 32 == 0 (got M={m} K={k} N={n})"
        )));
    }
    need("x", &x, "torch.bfloat16", &[m, k], false)?;
    need("w", &w, "torch.uint8", &[n, k / 2], true)?;
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
    need("aq", &aq, "torch.uint8", &[16, k / 2], true)?;
    need("asf", &asf, "torch.uint8", &[16, k / 16], true)?;
    need("out", &out, "torch.bfloat16", &[m, n], false)?;
    for (nm, t) in [
        ("w", &w),
        ("w_sf", &wsf),
        ("aq", &aq),
        ("asf", &asf),
        ("out", &out),
    ] {
        if t.device != x.device {
            return Err(PyValueError::new_err(format!("{nm} on another device")));
        }
    }
    let quant_args = [
        Arg::Ptr(x.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        Arg::U32(m as u32),
        Arg::U32(u32_of("k", k)?),
        Arg::U32(u32_of("x_stride", x.stride[0])?),
        Arg::F32(gscale),
    ];
    let gemm_args = [
        Arg::Ptr(out.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(asf.ptr),
        Arg::Ptr(w.ptr),
        Arg::Ptr(wsf.ptr),
        Arg::U32(m as u32),
        Arg::U32(u32_of("n", n)?),
        Arg::U32(u32_of("k", k)?),
        Arg::U32(u32_of("out_stride", out.stride[0])?),
        Arg::F32(alpha),
    ];
    let qgrid = ((k / 16).div_ceil(128) as u32, 16, 1);
    let ggrid = ((n / 32) as u32, 1, 1);
    let ordinal = x.device;
    py.detach(move || {
        crate::guard_py("nvfp4_gemm_cuda", move || {
            let q = function(FAMILY, QUANT, ordinal).map_err(PyRuntimeError::new_err)?;
            let g = function(FAMILY, GEMM, ordinal).map_err(PyRuntimeError::new_err)?;
            launch(q, qgrid, (128, 1, 1), 0, stream_ptr, &quant_args)
                .and_then(|_| launch(g, ggrid, (128, 1, 1), 0, stream_ptr, &gemm_args))
                .map_err(|e| PyRuntimeError::new_err(format!("nvfp4 gemm launch: {e}")))
        })
    })
}
