// SPDX-License-Identifier: Apache-2.0
//! NVFP4 DS-MLA host ops on the cuda-oxide track (feature `oxide-kernels`):
//! validate the torch views, then launch the kernels of
//! kernels-oxide/nvfp4_ds_mla (sm_120a SASS from ptxas 13.0, loaded by
//! suffix_hybrid.oxide_kernels) on torch's current stream.
//!
//! Decode replaces flashinfer's `trtllm_batch_decode_with_kv_cache_mla` SM120
//! sparse call in FlashInferMLASparseSM120Impl._run_mqa_kernel (same
//! semantics, dossier dsmla-kernel-contracts-glm53.md §4):
//!   q            [T, HQ, 576] or [T, 1, HQ, 576] bf16 (512 NoPE + 64 RoPE)
//!   kv_cache     uint8 [num_blocks, block_size, 352]; stride(0) may exceed
//!                block_size * 352 (HiSparse hot views) — slot s lives at
//!                (s / block_size) * stride(0) + (s % block_size) * 352
//!   block_tables [T, C] or [T, 1, C] int32 physical token slots (-1 masks)
//!   out          [T, HQ, 512] or [T, 1, HQ, 512] bf16 (bmm2_scale == 1)
//!   o_part [T, HQ, NS, 512] bf16 + lse_part [T, HQ, NS] f32 — NS from the
//!                plan (`nvfp4_ds_mla_plan`); c_per_split is re-derived from
//!                NS here, so the launch always matches the workspace.
//! Grid = f(T, HQ, C, NS) only: CUDA-graph replay safe.

use crate::nvfp4_ds_mla::{
    DIM, PE_DIM, ROW_BYTES, SCALE_FIX, THREADS, check_launch, merge_smem_bytes, partial_smem_bytes,
    split_rows,
};
use crate::oxide::{Arg, function, launch};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "nvfp4_ds_mla";

struct TInfo {
    ptr: u64,
    shape: Vec<usize>,
    stride: Vec<usize>,
    device: usize,
}

fn err<T>(msg: String) -> PyResult<T> {
    Err(PyValueError::new_err(format!("NVFP4-DSMLA: {msg}")))
}

fn tinfo(name: &str, obj: &Bound<'_, PyAny>, dtype: &str) -> PyResult<TInfo> {
    let dev = obj.getattr("device")?;
    let typ: String = dev.getattr("type")?.extract()?;
    if typ != "cuda" {
        return err(format!("{name} must be a CUDA tensor (got {typ})"));
    }
    let got = obj.getattr("dtype")?.str()?.to_string();
    if got != dtype {
        return err(format!("{name}: dtype {got} != {dtype}"));
    }
    let stride: Vec<isize> = obj.call_method0("stride")?.extract()?;
    if stride.iter().any(|s| *s < 0) {
        return err(format!("{name}: negative stride"));
    }
    Ok(TInfo {
        ptr: obj.call_method0("data_ptr")?.extract()?,
        shape: obj.call_method0("size")?.extract()?,
        stride: stride.into_iter().map(|s| s as usize).collect(),
        device: dev.getattr("index")?.extract::<Option<usize>>()?.unwrap_or(0),
    })
}

/// Dense (row-major contiguous) up to size-1 dims, which never move an
/// offset (unsqueeze gives them arbitrary strides).
fn dense(name: &str, i: &TInfo) -> PyResult<()> {
    let mut want = 1usize;
    for (d, s) in i.shape.iter().zip(i.stride.iter()).rev() {
        if *d != 1 && *s != want {
            return err(format!("{name}: non-dense stride {:?} for {:?}", i.stride, i.shape));
        }
        want *= d;
    }
    Ok(())
}

/// Drop the wrapper's singleton axis 1 of a rank-`rank` view ([T, 1, ...]).
fn squeeze1(shape: &[usize], rank: usize) -> Vec<usize> {
    if shape.len() == rank && shape[1] == 1 {
        [&shape[..1], &shape[2..]].concat()
    } else {
        shape.to_vec()
    }
}

/// kv cache [blocks, block_size, 352] uint8 -> (block_size, block_stride B).
fn kv_geometry(kv: &TInfo) -> PyResult<(usize, usize)> {
    if kv.shape.len() != 3 || kv.shape[2] != ROW_BYTES {
        return err(format!("kv_cache must be uint8 [blocks, block_size, {ROW_BYTES}], got {:?}", kv.shape));
    }
    let (bs, bstride) = (kv.shape[1], kv.stride[0]);
    if kv.stride[2] != 1 || (bs > 1 && kv.stride[1] != ROW_BYTES) {
        return err(format!("kv_cache rows must be dense 352 B, strides {:?}", kv.stride));
    }
    if bs == 0 || kv.ptr % 16 != 0 || bstride % 16 != 0 || bstride > u32::MAX as usize {
        return err(format!("kv_cache: 16-byte aligned base/block stride needed (stride {bstride})"));
    }
    if kv.shape[0] > 1 && bstride < bs * ROW_BYTES {
        return err(format!("kv_cache: block stride {bstride} overlaps rows"));
    }
    Ok((bs, bstride))
}

/// NVFP4 sparse-MLA decode (see module doc). `sm_scale` = bmm1_scale.
#[pyfunction]
#[pyo3(signature = (q, kv_cache, block_tables, out, o_part, lse_part, sm_scale, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_ds_mla_decode_cuda<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    kv_cache: &Bound<'py, PyAny>,
    block_tables: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    o_part: &Bound<'py, PyAny>,
    lse_part: &Bound<'py, PyAny>,
    sm_scale: f32,
    stream_ptr: usize,
) -> PyResult<()> {
    let qi = tinfo("q", q, "torch.bfloat16")?;
    let kv = tinfo("kv_cache", kv_cache, "torch.uint8")?;
    let bt = tinfo("block_tables", block_tables, "torch.int32")?;
    let oo = tinfo("out", out, "torch.bfloat16")?;
    let op = tinfo("o_part", o_part, "torch.bfloat16")?;
    let lp = tinfo("lse_part", lse_part, "torch.float32")?;
    let (tokens, hq) = match squeeze1(&qi.shape, 4)[..] {
        [t, h, d] if d == DIM + PE_DIM => (t, h),
        _ => return err(format!("q must be [T, HQ, 576] or [T, 1, HQ, 576], got {:?}", qi.shape)),
    };
    if tokens == 0 {
        return Ok(());
    }
    if hq == 0 || qi.ptr % 16 != 0 {
        return err("q: HQ >= 1 and a 16-byte aligned base needed".into());
    }
    dense("q", &qi)?;
    if squeeze1(&oo.shape, 4) != [tokens, hq, DIM] {
        return err(format!("out must be [T, HQ, 512] (T={tokens}, HQ={hq}), got {:?}", oo.shape));
    }
    dense("out", &oo)?;
    let (bs, bstride) = kv_geometry(&kv)?;
    let cap = match squeeze1(&bt.shape, 3)[..] {
        [t, c] if t == tokens => c,
        _ => return err(format!("block_tables must be [T, C] or [T, 1, C] (T={tokens}), got {:?}", bt.shape)),
    };
    dense("block_tables", &bt)?;
    if cap == 0 || cap > u32::MAX as usize {
        return err(format!("capacity {cap} outside 1..=u32::MAX"));
    }
    let ns = match op.shape[..] {
        [t, h, n, d] if t == tokens && h == hq && d == DIM && n >= 1 => n,
        _ => return err(format!("o_part must be [T, HQ, NS, 512], got {:?}", op.shape)),
    };
    check_launch(tokens, hq, cap, ns).or_else(err)?;
    let c_per_split = split_rows(cap, ns);
    if cap.div_ceil(c_per_split) != ns {
        return err(format!("NS={ns} is not a plan split count for capacity {cap}"));
    }
    if lp.shape != [tokens, hq, ns] {
        return err(format!("lse_part must be [T, HQ, NS]={:?}, got {:?}", [tokens, hq, ns], lp.shape));
    }
    dense("o_part", &op)?;
    dense("lse_part", &lp)?;
    for (n, i) in [("kv_cache", &kv), ("block_tables", &bt), ("out", &oo), ("o_part", &op), ("lse_part", &lp)] {
        if i.device != qi.device {
            return err(format!("{n} on cuda:{} != q on cuda:{}", i.device, qi.device));
        }
    }
    let hqt = hq.div_ceil(8);
    let u = |x: usize| Arg::U32(x as u32);
    // exp2 domain (log2 e) + undo the in-kernel e4m3->f16 2^-8 shift on K;
    // the merge undoes it on V (bmm2_scale == 1).
    let partial_args = vec![
        Arg::Ptr(qi.ptr),
        Arg::Ptr(kv.ptr),
        Arg::Ptr(bt.ptr),
        Arg::Ptr(op.ptr),
        Arg::Ptr(lp.ptr),
        u(cap), // topk_stride (dense rows)
        u(cap), // topk_len: seq_lens=None -> every column active
        u(c_per_split),
        u(ns),
        u(hq * (DIM + PE_DIM)),
        u(tokens),
        u(hq),
        u(hqt),
        u(bs),
        u(bstride),
        Arg::F32(sm_scale * std::f32::consts::LOG2_E * SCALE_FIX),
    ];
    let merge_args = vec![Arg::Ptr(oo.ptr), Arg::Ptr(op.ptr), Arg::Ptr(lp.ptr), u(ns), Arg::F32(SCALE_FIX)];
    let ord = qi.device;
    let grid_p = ((tokens * hqt) as u32, ns as u32, 1);
    let grid_m = ((tokens * hq) as u32, 1, 1);
    let (psmem, msmem) = (partial_smem_bytes() as u32, merge_smem_bytes(ns) as u32);
    py.detach(move || {
        crate::guard_py("nvfp4_ds_mla_decode_cuda", move || {
            let run = || -> Result<(), String> {
                let fp = function(FAMILY, "nvfp4_ds_mla_attn_partial", ord)?;
                let fm = function(FAMILY, "nvfp4_ds_mla_attn_merge", ord)?;
                launch(fp, grid_p, (THREADS as u32, 1, 1), psmem, stream_ptr, &partial_args)?;
                launch(fm, grid_m, (THREADS as u32, 1, 1), msmem, stream_ptr, &merge_args)
            };
            run().map_err(|e| PyRuntimeError::new_err(format!("NVFP4-DSMLA launch failed: {e}")))
        })
    })
}

/// Writer: kv_c [T, 512] + k_pe [T, 64] bf16 (last dim contiguous, any row
/// stride) -> 352 B rows of kv_cache [blocks, block_size, 352] at
/// slot_mapping [>=T] int64 (negative = skip). Replaces concat_and_cache_mla
/// for nvfp4_ds_mla on SM120 (its C++ dispatch is SM100-gated twice).
#[pyfunction]
#[pyo3(signature = (kv_c, k_pe, kv_cache, slot_mapping, stream_ptr))]
pub fn nvfp4_ds_mla_quant_store_cuda<'py>(
    py: Python<'py>,
    kv_c: &Bound<'py, PyAny>,
    k_pe: &Bound<'py, PyAny>,
    kv_cache: &Bound<'py, PyAny>,
    slot_mapping: &Bound<'py, PyAny>,
    stream_ptr: usize,
) -> PyResult<()> {
    let kc = tinfo("kv_c", kv_c, "torch.bfloat16")?;
    let kp = tinfo("k_pe", k_pe, "torch.bfloat16")?;
    let kv = tinfo("kv_cache", kv_cache, "torch.uint8")?;
    let sm = tinfo("slot_mapping", slot_mapping, "torch.int64")?;
    let rows = kc.shape.first().copied().unwrap_or(0);
    if kc.shape != [rows, DIM] || kp.shape != [rows, PE_DIM] {
        return err(format!("kv_c {:?} / k_pe {:?}: need [T, 512] / [T, 64]", kc.shape, kp.shape));
    }
    if kc.stride[1] != 1 || kp.stride[1] != 1 || sm.shape.len() != 1 || sm.stride[0] != 1 {
        return err("kv_c/k_pe last dim and slot_mapping must be contiguous".into());
    }
    let (bs, bstride) = kv_geometry(&kv)?;
    for (n, i) in [("k_pe", &kp), ("slot_mapping", &sm), ("kv_cache", &kv)] {
        if i.device != kc.device {
            return err(format!("{n} on cuda:{} != kv_c on cuda:{}", i.device, kc.device));
        }
    }
    let tokens = rows.min(sm.shape[0]);
    if tokens == 0 {
        return Ok(());
    }
    let u = |x: usize| Arg::U32(x as u32);
    let args = vec![
        Arg::Ptr(kc.ptr),
        Arg::Ptr(kp.ptr),
        Arg::Ptr(sm.ptr),
        Arg::Ptr(kv.ptr),
        u(tokens),
        u(kc.stride[0]),
        u(kp.stride[0]),
        u(bs),
        u(bstride),
    ];
    let ord = kc.device;
    py.detach(move || {
        crate::guard_py("nvfp4_ds_mla_quant_store_cuda", move || {
            let run = || -> Result<(), String> {
                let f = function(FAMILY, "nvfp4_ds_mla_quant_store", ord)?;
                launch(f, (tokens as u32, 1, 1), (64, 1, 1), 0, stream_ptr, &args)
            };
            run().map_err(|e| PyRuntimeError::new_err(format!("NVFP4-DSMLA writer launch failed: {e}")))
        })
    })
}
