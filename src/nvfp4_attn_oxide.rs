// SPDX-License-Identifier: Apache-2.0
//! K2-NVFP4 host op on the cuda-oxide track (feature `oxide-kernels`):
//! validates the torch views, then launches the two kernels of
//! kernels-oxide/k2_nvfp4_attn (sm_120 SASS from ptxas 13.0, loaded by
//! suffix_hybrid.oxide_kernels) on torch's current stream. Python API is the
//! same as the retired cutile op, so sm120/nvfp4_kv_patch/own_attn.py (and
//! the H19/H20/H21/H13 patch logic, oracle and CPU twin) are unchanged.
//!
//! Grid depends only on (batch, q_len, heads) — never on seq_lens — so a
//! captured CUDA graph replays correctly for any KV lengths.

use crate::nvfp4_attn::{merge_smem_bytes, partial_smem_bytes, plan, round16, AttnPlan, THREADS};
use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "k2_nvfp4_attn";
/// e4m3 -> f16 in-kernel yields scale * 2^-8 (exact bit shift); undone here.
const SCALE_FIX: f32 = 256.0;

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

/// Our NVFP4 paged decode / spec-verify attention (K2-NVFP4).
///
/// q        [B*q_len, HQ, D] bf16 (row stride free, heads/dims dense)
/// k_data   [P, HKV, PAGE, D/2] uint8     } nvfp4_split_data_scale views
/// k_sf     [P, HKV, PAGE, D/16] e4m3     } of the HND cache (one shared
/// v_data   [P, HKV, PAGE, D/2] uint8     } page stride; (head, token,
/// v_sf     [P, HKV, PAGE, D/16] e4m3     } byte) dense within a side)
/// block_table [>=B, W] int32 (row-contiguous), seq_lens [>=B] int32
/// meta     [16] int32 (kept for API stability; unused on this track)
/// out      [B*q_len, HQ, D] bf16, contiguous — fully written
/// o_part   [round16(R), NS, M, D] bf16, lse_part [round16(R), NS, M] f32
/// window_left: -1 = full attention, else FlashInfer semantics.
/// qk_scale_log2 = sm_scale * k_scale * log2(e); v_scale = global V scale.
#[pyfunction]
#[pyo3(signature = (q, k_data, k_sf, v_data, v_sf, block_table, meta, seq_lens, out, o_part, lse_part, q_len, window_left, qk_scale_log2, v_scale, num_sms, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_paged_attn_cuda<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    k_data: &Bound<'py, PyAny>,
    k_sf: &Bound<'py, PyAny>,
    v_data: &Bound<'py, PyAny>,
    v_sf: &Bound<'py, PyAny>,
    block_table: &Bound<'py, PyAny>,
    meta: &Bound<'py, PyAny>,
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
    let _ = meta;
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
    if qi.shape.len() != 3 || kd.shape.len() != 4 || bt.shape.len() != 2 || sl.shape.len() != 1 {
        return Err(PyValueError::new_err(
            "q must be 3-D, k/v views 4-D, block_table 2-D, seq_lens 1-D",
        ));
    }
    let (tokens, hq, d) = (qi.shape[0], qi.shape[1], qi.shape[2]);
    let (pages, hkv, page) = (kd.shape[0], kd.shape[1], kd.shape[2]);
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
        plan(batch, q_len, hq, hkv, d, page, num_sms).map_err(PyValueError::new_err)?;
    let (dh, sd) = (d / 2, d / 16);
    need("q", &qi, "torch.bfloat16", &[tokens, hq, d])?;
    if qi.stride[1] != d {
        return Err(PyValueError::new_err(
            "q: heads must be dense (stride[1] == D)",
        ));
    }
    let page_bytes = kd.stride[0];
    for (n, i, w, dt) in [
        ("k_data", &kd, dh, "torch.uint8"),
        ("v_data", &vd, dh, "torch.uint8"),
        ("k_sf", &ks, sd, "torch.float8_e4m3fn"),
        ("v_sf", &vs, sd, "torch.float8_e4m3fn"),
    ] {
        need(n, i, dt, &[pages, hkv, page, w])?;
        if i.stride[2] != w || i.stride[1] != page * w || i.stride[0] != page_bytes {
            return Err(PyValueError::new_err(format!(
                "{n}: (head, token, byte) must be dense within a page side and all four \
                 views share one page stride (HND nvfp4 layout), got strides {:?}",
                i.stride
            )));
        }
    }
    // cp.async 16 B copies of every page run: all four view bases and the
    // page stride must be 16-byte aligned (torch allocations + HND views are).
    if [kd.ptr, ks.ptr, vd.ptr, vs.ptr].iter().any(|p| p % 16 != 0) || page_bytes % 16 != 0 {
        return Err(PyValueError::new_err(
            "k/v data and scale views must be 16-byte aligned (cp.async)",
        ));
    }
    if page_bytes > u32::MAX as usize {
        return Err(PyValueError::new_err("page stride exceeds u32"));
    }
    if bt.shape[0] < batch || bt.stride[1] != 1 || sl.shape[0] < batch {
        return Err(PyValueError::new_err(
            "block_table [>=B, W] (row-contiguous) and seq_lens [>=B] required",
        ));
    }
    need("block_table", &bt, "torch.int32", &bt.shape.clone())?;
    need("seq_lens", &sl, "torch.int32", &sl.shape.clone())?;
    need("out", &oo, "torch.bfloat16", &[tokens, hq, d])?;
    if oo.stride != [hq * d, d, 1] {
        return Err(PyValueError::new_err("out must be contiguous"));
    }
    let rows16 = round16(p.rows);
    need("o_part", &op, "torch.bfloat16", &[rows16, p.ns, p.m, d])?;
    need("lse_part", &lp, "torch.float32", &[rows16, p.ns, p.m])?;
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
    let ord = qi.device;
    let u = |x: usize| Arg::U32(x as u32);
    let partial_args = vec![
        Arg::Ptr(qi.ptr),
        Arg::Ptr(kd.ptr),
        Arg::Ptr(ks.ptr),
        Arg::Ptr(vd.ptr),
        Arg::Ptr(vs.ptr),
        Arg::Ptr(bt.ptr),
        Arg::Ptr(sl.ptr),
        Arg::Ptr(op.ptr),
        Arg::Ptr(lp.ptr),
        u(qi.stride[0]),
        u(page_bytes),
        u(bt.stride[0]),
        u(hkv),
        u(p.g),
        u(p.qt),
        u(p.m),
        u(d),
        u(p.nqt),
        u(q_len),
        u(page),
        u(p.ns),
        u(p.tn),
        Arg::I32(window_left),
        Arg::F32(qk_scale_log2 * SCALE_FIX),
    ];
    let merge_args = vec![
        Arg::Ptr(oo.ptr),
        Arg::Ptr(op.ptr),
        Arg::Ptr(lp.ptr),
        u(hkv),
        u(p.g),
        u(p.qt),
        u(p.m),
        u(d),
        u(p.nqt),
        u(q_len),
        u(p.ns),
        Arg::F32(v_scale * SCALE_FIX),
    ];
    let (rows, ns) = (p.rows as u32, p.ns as u32);
    let psmem = partial_smem_bytes(p.m, d, p.tn) as u32;
    let msmem = merge_smem_bytes(p.ns, p.m) as u32;
    let block = (THREADS as u32, 1, 1);
    py.detach(move || {
        crate::guard_py("nvfp4_paged_attn_cuda", move || {
            let run = || -> Result<(), String> {
                let fp = function(FAMILY, "nvfp4_attn_partial", ord)?;
                let fm = function(FAMILY, "nvfp4_attn_merge", ord)?;
                launch(fp, (rows, ns, 1), block, psmem, stream_ptr, &partial_args)?;
                launch(
                    fm,
                    (rows, (d / 64) as u32, 1),
                    block,
                    msmem,
                    stream_ptr,
                    &merge_args,
                )
            };
            run().map_err(|e| PyRuntimeError::new_err(format!("K2-NVFP4 launch failed: {e}")))
        })
    })
}
