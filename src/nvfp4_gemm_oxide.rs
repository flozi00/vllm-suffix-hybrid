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
const REDUCE: &str = "nvfp4_splitk_reduce";

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

/// SM count of the RTX PRO 6000 (Server / Max-Q): the split-K policy
/// targets ~4 resident CTAs per SM.
pub const SMS: usize = 188;
pub const MAX_SPLITS: usize = 8;

/// Split-K factor for an (N, K) decode GEMM: enough CTAs (N/32 per split)
/// to cover ~4 waves of the 188 SMs, capped at 8 and at K/64 steps. Wide-N
/// projections (gate_up, qkvz, qkv) stay at 1 (no reduction kernel).
pub fn splits_for(n: usize, k: usize) -> usize {
    let ctas = n / 32;
    let want = (4 * SMS).div_ceil(ctas.max(1));
    want.clamp(1, MAX_SPLITS).min(k / 64)
}

#[pyfunction]
pub fn nvfp4_gemm_splits(n: usize, k: usize) -> usize {
    splits_for(n, k)
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
    if !(1..=16).contains(&m) || k % 64 != 0 || n % 32 != 0 {
        return Err(PyValueError::new_err(format!(
            "NVFP4 decode GEMM needs 1 <= M <= 16, K % 64 == 0, N % 32 == 0 (M={m} K={k} N={n})"
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
    if partial.dtype != "torch.float32" || partial.shape.iter().product::<usize>() < splits * 16 * n
    {
        return Err(PyValueError::new_err(format!(
            "partial workspace must be float32 with >= {} elements",
            splits * 16 * n
        )));
    }
    Ok(n)
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
    stream_ptr: usize,
    ordinal: usize,
) -> PyResult<()> {
    let steps = k / 64;
    let kps = steps.div_ceil(splits);
    let splits = steps.div_ceil(kps); // no empty trailing split
    let part_ptr = if splits > 1 { partial.ptr } else { 0 };
    let gemm_args = [
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
    let red_args = [
        Arg::Ptr(out.ptr),
        Arg::Ptr(partial.ptr),
        Arg::U32(m as u32),
        Arg::U32(n as u32),
        Arg::U32(splits as u32),
        Arg::U32(out.stride[0] as u32),
        Arg::F32(alpha),
    ];
    let qgrid = ((k / 16).div_ceil(128) as u32, 16, 1);
    let ggrid = ((n / 32) as u32, splits as u32, 1);
    let rgrid = ((m * n).div_ceil(256) as u32, 1, 1);
    py.detach(move || {
        crate::guard_py("nvfp4_gemm_cuda", move || {
            let e = |e: String| PyRuntimeError::new_err(format!("nvfp4 gemm launch: {e}"));
            if let Some(qa) = &quant {
                let q = function(FAMILY, QUANT, ordinal).map_err(PyRuntimeError::new_err)?;
                launch(q, qgrid, (128, 1, 1), 0, stream_ptr, qa).map_err(e)?;
            }
            let gm = function(FAMILY, GEMM, ordinal).map_err(PyRuntimeError::new_err)?;
            launch(gm, ggrid, (128, 1, 1), 0, stream_ptr, &gemm_args).map_err(e)?;
            if splits > 1 {
                let r = function(FAMILY, REDUCE, ordinal).map_err(PyRuntimeError::new_err)?;
                launch(r, rgrid, (256, 1, 1), 0, stream_ptr, &red_args).map_err(e)?;
            }
            Ok(())
        })
    })
}

/// y = alpha * nvfp4(x) @ w^T for M <= 16 (decode), bf16 activation path:
/// our quant (`aq` uint8 [16, K/2], `asf` uint8 [16, K/16] scratch) + GEMM
/// (+ split-K reduce into `partial` float32 >= splits*16*N). x bf16 [M, K]
/// (row stride allowed); w uint8 [N, K/2] (vLLM packed e2m1, low nibble =
/// even k); w_sf swizzled 128x4 e4m3; out bf16 [M, N]. gscale = x global
/// scale (vLLM input_global_scale_inv), alpha = vLLM layer.alpha. All
/// buffers torch-owned and preallocated: no allocation, no host sync
/// (CUDA-graph capturable).
#[pyfunction]
#[pyo3(signature = (x, w, w_sf, aq, asf, partial, out, gscale, alpha, splits, stream_ptr))]
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
    if aq.dtype != "torch.uint8" || aq.shape.iter().product::<usize>() < 16 * k / 2 {
        return Err(PyValueError::new_err(
            "aq must be uint8 with >= 16*K/2 bytes",
        ));
    }
    if asf.dtype != "torch.uint8" || asf.shape.iter().product::<usize>() < 16 * k / 16 {
        return Err(PyValueError::new_err(
            "asf must be uint8 with >= 16*K/16 bytes",
        ));
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
        a_rows: 16,
        asf_ptr: asf.ptr,
        asf_mode: 0,
    };
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
        stream_ptr,
        x.device,
    )
}

/// Same GEMM on an activation vLLM already quantized (fused SiLU*mul /
/// RMSNorm + NVFP4 quant): xq uint8 [M, K/2], xsf e4m3/uint8 swizzled 128x4
/// ([round128(M), round4(K/16)]).
#[pyfunction]
#[pyo3(signature = (xq, xsf, w, w_sf, partial, out, alpha, splits, stream_ptr))]
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
) -> PyResult<()> {
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
    run(
        py, None, g, &w, &wsf, &out, &partial, m, n, k, splits, alpha, stream_ptr, xq.device,
    )
}

#[cfg(test)]
mod tests {
    use super::splits_for;

    #[test]
    fn split_policy_targets_narrow_n_only() {
        assert_eq!(splits_for(34816, 5120), 1); // gate_up: 1088 CTAs
        assert_eq!(splits_for(16384, 5120), 2); // qkvz: 512 CTAs -> 1024
        assert_eq!(splits_for(5120, 17408), 5); // down: 160 -> 800
        assert_eq!(splits_for(5120, 6144), 5); // o / gdn out
        assert_eq!(splits_for(2816, 2112), 8); // gemma down: 88 CTAs
        assert!(splits_for(64, 64) == 1); // K/64 cap
    }
}
