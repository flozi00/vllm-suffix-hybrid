// SPDX-License-Identifier: Apache-2.0
//! K-GDN1 host op on the cuda-oxide track (feature `oxide-kernels`):
//! launches kernels-oxide/kgdn1 (sm_120 SASS from ptxas 13.0, loaded by
//! suffix_hybrid.oxide_kernels.ensure_loaded("kgdn1")) on torch's stream.
//! Same Python signature and contract as the retired cutile op
//! (src/qwen_gdn_gpu.rs), so suffix_hybrid/kernels/qwen_gdn.py is unchanged
//! at the call site. Layout contract: src/qwen_gdn.rs; kernel ABI:
//! kernels-oxide/kgdn1/interface.json.

use crate::oxide::{function, launch, Arg};
use crate::qwen_gdn::{GdnDims, ACT_SIGMOID, ACT_SILU};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "kgdn1";
pub const ENTRY: &str = "kgdn1_decode_h16_hv48_k128";
/// The shape the shipped cubin is compiled for (interface.json).
pub const SHAPE: (usize, usize, usize) = (16, 48, 128);
const BLOCK: u32 = 256;
const SMEM_BYTES: u32 = 129 * 4;

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
            "{name} must be a CUDA tensor (got {typ:?}); K-GDN1 never runs elsewhere"
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
    if i.dtype != dtype || i.shape != shape {
        return Err(PyValueError::new_err(format!(
            "{name}: got {} {:?}, need {dtype} {shape:?}",
            i.dtype, i.shape
        )));
    }
    if i.stride.last().copied().unwrap_or(1) != 1 {
        return Err(PyValueError::new_err(format!(
            "{name}: last dim must be contiguous"
        )));
    }
    Ok(())
}

fn u32_of(name: &str, v: usize) -> PyResult<u32> {
    u32::try_from(v).map_err(|_| PyValueError::new_err(format!("{name}={v} exceeds u32")))
}

/// Fused GDN decode step (K-GDN1). `state` [S, HV, V, K] fp32 is updated in
/// place (slot stride may be vLLM's padded page); `out` [T, HV, V] bf16
/// (contiguous) is fully written (zero rows for slot <= 0). `mixed_qkv`
/// [T, 2HK+HV*V] and `z` [T, HV, V] bf16 may be row-strided views.
/// `stream_ptr` = torch.cuda.current_stream().cuda_stream (0 = legacy default).
#[pyfunction]
#[pyo3(signature = (mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads, scale, norm_eps, act, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_fused_cuda<'py>(
    py: Python<'py>,
    mixed_qkv: &Bound<'py, PyAny>,
    z: &Bound<'py, PyAny>,
    ba: &Bound<'py, PyAny>,
    a_log: &Bound<'py, PyAny>,
    dt_bias: &Bound<'py, PyAny>,
    norm_w: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    state_idx: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    num_k_heads: usize,
    scale: f32,
    norm_eps: f32,
    act: i32,
    stream_ptr: usize,
) -> PyResult<()> {
    let st = tinfo("state", state)?;
    let mq = tinfo("mixed_qkv", mixed_qkv)?;
    if st.shape.len() != 4 || mq.shape.len() != 2 {
        return Err(PyValueError::new_err("state must be 4-D and mixed_qkv 2-D"));
    }
    let d = GdnDims {
        t: mq.shape[0],
        h: num_k_heads,
        hv: st.shape[1],
        v: st.shape[2],
        k: st.shape[3],
        slots: st.shape[0],
    };
    d.validate().map_err(PyValueError::new_err)?;
    if (d.h, d.hv, d.k) != SHAPE {
        return Err(PyValueError::new_err(format!(
            "K-GDN1 cubin is compiled for (H, HV, K) = {SHAPE:?}, got ({}, {}, {})",
            d.h, d.hv, d.k
        )));
    }
    if act != ACT_SILU && act != ACT_SIGMOID {
        return Err(PyValueError::new_err(format!(
            "unknown activation code {act}"
        )));
    }
    let zz = tinfo("z", z)?;
    let bb = tinfo("ba", ba)?;
    let al = tinfo("a_log", a_log)?;
    let dt = tinfo("dt_bias", dt_bias)?;
    let nw = tinfo("norm_w", norm_w)?;
    let si = tinfo("state_idx", state_idx)?;
    let oo = tinfo("out", out)?;
    need("mixed_qkv", &mq, "torch.bfloat16", &[d.t, d.qkv_width()])?;
    need("z", &zz, "torch.bfloat16", &[d.t, d.hv, d.v])?;
    if zz.stride[1] != d.v {
        return Err(PyValueError::new_err(
            "z: head dim must be dense (stride[1] == V)",
        ));
    }
    need("ba", &bb, "torch.bfloat16", &[d.t, 2 * d.hv])?;
    if bb.stride[0] != 2 * d.hv {
        return Err(PyValueError::new_err("ba must be contiguous"));
    }
    need("a_log", &al, "torch.float32", &[d.hv])?;
    need("dt_bias", &dt, "torch.float32", &[d.hv])?;
    need("norm_w", &nw, "torch.float32", &[d.v])?;
    need("state", &st, "torch.float32", &[d.slots, d.hv, d.v, d.k])?;
    if st.stride[1..] != [d.v * d.k, d.k, 1] || st.stride[0] < d.hv * d.v * d.k {
        return Err(PyValueError::new_err(format!(
            "state strides {:?}: per-slot [HV, V, K] block must be dense",
            st.stride
        )));
    }
    need("state_idx", &si, "torch.int32", &[d.t])?;
    need("out", &oo, "torch.bfloat16", &[d.t, d.hv, d.v])?;
    if oo.stride != [d.hv * d.v, d.v, 1] {
        return Err(PyValueError::new_err("out must be contiguous"));
    }
    for (n, i) in [
        ("z", &zz),
        ("ba", &bb),
        ("a_log", &al),
        ("dt_bias", &dt),
        ("norm_w", &nw),
        ("state", &st),
        ("state_idx", &si),
        ("out", &oo),
    ] {
        if i.device != mq.device {
            return Err(PyValueError::new_err(format!(
                "{n} on cuda:{} != cuda:{}",
                i.device, mq.device
            )));
        }
    }
    if d.t == 0 {
        return Ok(());
    }
    if d.t > 65535 {
        return Err(PyValueError::new_err("T exceeds grid.y (65535)"));
    }
    let args = [
        Arg::Ptr(oo.ptr),
        Arg::Ptr(mq.ptr),
        Arg::Ptr(zz.ptr),
        Arg::Ptr(bb.ptr),
        Arg::Ptr(al.ptr),
        Arg::Ptr(dt.ptr),
        Arg::Ptr(nw.ptr),
        Arg::Ptr(st.ptr),
        Arg::Ptr(si.ptr),
        Arg::U32(u32_of("slots", d.slots)?),
        Arg::U32(u32_of("qkv_stride", mq.stride[0])?),
        Arg::U32(u32_of("z_stride", zz.stride[0])?),
        Arg::U32(u32_of("state_stride", st.stride[0])?),
        Arg::U32(act as u32),
        Arg::F32(scale),
        Arg::F32(norm_eps),
    ];
    let grid = (d.hv as u32, d.t as u32, 1);
    let ordinal = mq.device;
    py.detach(move || {
        crate::guard_py("gdn_decode_fused_cuda", move || {
            let f = function(FAMILY, ENTRY, ordinal).map_err(PyRuntimeError::new_err)?;
            launch(f, grid, (BLOCK, 1, 1), SMEM_BYTES, stream_ptr, &args)
                .map_err(|e| PyRuntimeError::new_err(format!("K-GDN1 launch failed: {e}")))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host constants must match the interface file CI/loader use.
    #[test]
    fn host_constants_match_interface_json() {
        let j = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels-oxide/kgdn1/interface.json"
        ))
        .unwrap();
        assert!(j.contains(&format!("\"entry\": \"{ENTRY}\"")));
        assert!(j.contains(&format!("\"block\": [{BLOCK}, 1, 1]")));
        assert!(j.contains(&format!("\"dynamic_smem_bytes\": {SMEM_BYTES}")));
        assert!(j.contains("\"H\": 16, \"HV\": 48, \"K\": 128"));
        // 16 params, in the launch order above
        assert_eq!(j.matches("\"name\":").count(), 16);
    }
}
