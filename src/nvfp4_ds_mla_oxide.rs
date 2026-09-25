// SPDX-License-Identifier: Apache-2.0
//! NVFP4 DS-MLA host op on the cuda-oxide track (feature `oxide-kernels`):
//! validates the torch views, then launches the three kernels of
//! kernels-oxide/nvfp4_ds_mla (sm_120 SASS from ptxas 13.0, loaded by
//! suffix_hybrid.oxide_kernels) on torch's current stream.
//!
//! ABI mirrors the call site it replaces (dossier
//! dsmla-kernel-contracts-glm53.md §4/§6): our op is a drop-in for
//! flashinfer's `trtllm_batch_decode_with_kv_cache_mla` SM120 sparse path
//! as invoked by FlashInferMLASparseSM120Impl._run_mqa_kernel —
//!   q           [T, 1, HQ, 576] bf16 (packed 512 NoPE + 64 RoPE)
//!   kv_cache    uint8 flat [num_blocks, block_size, 352] (3-D view)
//!   block_tables= topk_indices_physical [T, 1, C] int32 PHYSICAL token
//!               slot ids (-1 masks a row; NOT page indices)
//!   seq_lens    = None  -> every column active (we take topk_len = C)
//!   sm_scale    = bmm1_scale (python float); bmm2_scale == 1.0 (output
//!               NOT rescaled — nothing folded into v_scale beyond the
//!               SF e4m3->f16 2^-8 shift undo)
//!   out         [T, 1, HQ, 512] bf16
//! No workspace_buffer / max_seq_len needed: o_part [T, HQ, NS, 512] and
//! lse_part [T, HQ, NS] are sliced out of the same shared workspace the
//! impl already owns (or freshly passed — 394 MiB buffer is overkill for
//! topk=2048: T*64*32*(512*2B + 4B) ≈ 2.1 MiB/token).
//!
//! Grid depends only on (T_padded, HQ, C) — never on live fill state or
//! token counts (padding tokens read -1 indices / exit -inf) — so a
//! captured CUDA graph replays correctly.

use crate::nvfp4_ds_mla::{
    merge_smem_bytes, partial_smem_bytes, plan, DsMlaPlan, DIM, PE_DIM, ROW_BYTES, SCALE_FIX,
    THREADS,
};
use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "nvfp4_ds_mla";

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
            "{name} must be a CUDA tensor (got {typ:?}); NVFP4-DSMLA never runs elsewhere"
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
    // Dense strides required: every non-degenerate dim must be exactly the
    // product of the trailing dims (the kernels address o_part/lse_part/
    // kv_cache FLAT — a non-dense stride on an interior dim, e.g. an
    // ns-slice carved out of an [T,H,max_ns,512] workspace, would be
    // silently mis-addressed). Size-1 dims are exempt: torch's unsqueeze
    // gives them the next dim's stride, and they never move an offset.
    let mut want = 1usize;
    for (d, s) in i.shape.iter().zip(i.stride.iter()).rev() {
        if *d != 1 && *s != want {
            return Err(PyValueError::new_err(format!(
                "{name}: non-dense stride {:?} for shape {:?} — contiguous tensors only",
                i.stride, i.shape
            )));
        }
        want *= d.max(1);
    }
    Ok(())
}

/// NVFP4 sparse-MLA decode (DS-MLA BMM1+BMM2 over the nvfp4 reader cache).
///
/// q            [T, 1, HQ, 576] bf16 — packed query (512 NoPE + 64 RoPE);
///              the exact tensor _run_mqa_kernel hands the flashinfer
///              wrapper (leading T, heads dense, 576 contiguous)
/// kv_cache     uint8 [num_blocks, block_size, 352] (flat 3-D view of the
///              reader cache; block_size inferred from the shape; slots
///              addressed as flat [slot * 352])
/// block_tables [T, 1, C] int32 physical token-slot ids (topk indices,
///              converted logical->physical upstream; -1 masks a row)
/// out          [T, 1, HQ, 512] bf16 (bmm2_scale == 1.0: no rescale)
/// o_part       [T, HQ, NS, 512] bf16; lse_part [T, HQ, NS] f32 —
///              NS = ceil(C / 64) (flashinfer split contract; slice these
///              from the impl's workspace like mid_out / mid_lse)
/// sm_scale     python float (bmm1_scale; static for CUDA-graph replay)
/// num_sms      device SM count (sanity only — grid is (T*ceil(HQ/8), NS))
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
    let qi = tinfo("q", q)?;
    let kv = tinfo("kv_cache", kv_cache)?;
    let bt = tinfo("block_tables", block_tables)?;
    let oo = tinfo("out", out)?;
    let op = tinfo("o_part", o_part)?;
    let lp = tinfo("lse_part", lse_part)?;
    // q: accept [T, 1, HQ, 576] (wrapper shape) or [T, HQ, 576].
    let (tokens, hq) = match qi.shape[..] {
        [t, 1, h, 576] => (*t, *h),
        [t, h, 576] => (*t, *h),
        _ => {
            return Err(PyValueError::new_err(
                "q must be [T, 1, HQ, 576] or [T, HQ, 576] bf16 (packed 512 NoPE + 64 RoPE)",
            ))
        }
    };
    if tokens == 0 || hq == 0 || hq % 8 != 0 {
        return Err(PyValueError::new_err(format!(
            "bad q shape [tokens={tokens}, heads={hq}]: heads must be a positive multiple of 8"
        )));
    }
    need("q", &qi, "torch.bfloat16", &qi.shape.clone())?;
    // heads/dims dense: token stride must be HQ*576 (both [T,1,HQ,576] and
    // [T,HQ,576] pack every token contiguously).
    let q_stride = qi.stride[0];
    if q_stride != hq * 576 || qi.ptr % 16 != 0 {
        return Err(PyValueError::new_err(
            "q: token stride must be HQ*576 (dense heads) and 16-byte aligned",
        ));
    }
    // kv_cache: flat uint8 [num_blocks, block_size, 352] (or [slots, 352]).
    if kv.shape.len() < 2 || kv.shape.last().copied().unwrap_or(0) != ROW_BYTES {
        return Err(PyValueError::new_err(format!(
            "kv_cache must be a flat uint8 view with last dim {ROW_BYTES} (nvfp4_ds_mla rows), got {:?}",
            kv.shape
        )));
    }
    need("kv_cache", &kv, "torch.uint8", &kv.shape.clone())?;
    if kv.stride.last() != Some(&1) || kv.ptr % 16 != 0 {
        return Err(PyValueError::new_err(
            "kv_cache: last dim contiguous and 16-byte aligned (cp.async)",
        ));
    }
    // block_tables: [T, 1, C] (or [T, C]) int32 physical slot ids.
    let cap = match bt.shape[..] {
        [t, 1, c] if *t >= tokens => *c,
        [t, c] if *t >= tokens => *c,
        _ => {
            return Err(PyValueError::new_err(
                "block_tables must be [T, >=T, 1, C] int32 physical token-slot ids",
            ))
        }
    };
    need("block_tables", &bt, "torch.int32", &bt.shape.clone())?;
    let bt_stride = if bt.shape.len() == 3 { bt.stride[0] } else { bt.stride[0] };
    if cap == 0 || cap > u32::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "block_tables capacity {cap} outside 1..=u32::MAX"
        )));
    }
    need("out", &oo, "torch.bfloat16", &oo.shape.clone())?;
    if oo.shape != qi.shape[..qi.shape.len() - 1].to_vec() + [DIM] {
        return Err(PyValueError::new_err(
            "out must match q's leading dims with last dim 512",
        ));
    }
    if oo.stride.last() != Some(&1) || oo.stride != [hq * DIM, DIM, 1] && oo.shape.len() == 3 {
        // accept [T, HQ, 512] contiguous
        if !(oo.shape.len() == 4 && oo.stride == [hq * DIM, DIM, DIM, 1]
            || oo.shape.len() == 3 && oo.stride == [hq * DIM, DIM, 1])
        {
            return Err(PyValueError::new_err("out must be contiguous"));
        }
    }
    // Plan: grid purely of (T, HQ, C) — CUDA-graph stable regardless of
    // live fill state; -1 indices and padding tokens handle the rest in
    // the kernel (dossier §6).
    let p: DsMlaPlan = plan(tokens, hq, cap, 148 /* SM120 SM count, sanity only */)
        .map_err(PyValueError::new_err)?;
    need("o_part", &op, "torch.bfloat16", &[tokens, hq, p.ns, DIM])?;
    need("lse_part", &lp, "torch.float32", &[tokens, hq, p.ns])?;
    for (n, i) in [
        ("q", &qi),
        ("kv_cache", &kv),
        ("block_tables", &bt),
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
    let _ = PE_DIM; // 576 = DIM + PE_DIM asserted by the q shape check
    let ord = qi.device;
    let u = |x: usize| Arg::U32(x as u32);
    // bmm1_scale = sm_scale (python float); exp2 domain needs log2(e); the
    // SF e4m3->f16 shift leaves a x2^-8 on K (and V) — undo both here.
    let qk_scale_log2 = sm_scale * std::f32::consts::LOG2_E * SCALE_FIX;
    let v_scale = SCALE_FIX; // bmm2_scale == 1.0 * shift undo
    let partial_args = vec![
        Arg::Ptr(qi.ptr),
        Arg::Ptr(kv.ptr),
        Arg::Ptr(bt.ptr),
        Arg::Ptr(op.ptr),
        Arg::Ptr(lp.ptr),
        u(bt_stride),
        u(cap), // topk_len: seq_lens=None -> all columns active
        u(p.c_per_split),
        u(p.ns),
        u(q_stride),
        u(tokens),
        u(hq),
        u(p.hqt),
        Arg::F32(qk_scale_log2),
    ];
    let merge_args = vec![
        Arg::Ptr(oo.ptr),
        Arg::Ptr(op.ptr),
        Arg::Ptr(lp.ptr),
        u(p.ns),
        Arg::F32(v_scale),
    ];
    let (rows, ns) = (p.rows as u32, p.ns as u32);
    let psmem = partial_smem_bytes() as u32;
    let msmem = merge_smem_bytes(p.ns) as u32;
    let block = (THREADS as u32, 1, 1);
    py.detach(move || {
        crate::guard_py("nvfp4_ds_mla_decode_cuda", move || {
            let run = || -> Result<(), String> {
                let fp = function(FAMILY, "nvfp4_ds_mla_attn_partial", ord)?;
                let fm = function(FAMILY, "nvfp4_ds_mla_attn_merge", ord)?;
                launch(fp, (rows, ns, 1), block, psmem, stream_ptr, &partial_args)?;
                launch(fm, (p.merge_rows as u32, 1, 1), block, msmem, stream_ptr, &merge_args)
            };
            run().map_err(|e| {
                PyRuntimeError::new_err(format!("NVFP4-DSMLA launch failed: {e}"))
            })
        })
    })
}

/// Writer: kv_c [T, 512] bf16 + k_pe [T, 64] bf16 -> reader-cache rows
/// [slots, 352] uint8 via slot_mapping [T] i64 (-1 = skip). Replaces
/// torch.ops._C_cache_ops.concat_and_cache_mla for kv_cache_dtype
/// nvfp4_ds_mla on SM120 (dossier §1/§3: the C++ dispatcher is both
/// build-gated (FP4_SM100_ARCHS) and runtime-gated major==10).
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
    let kc = tinfo("kv_c", kv_c)?;
    let kp = tinfo("k_pe", k_pe)?;
    let kv = tinfo("kv_cache", kv_cache)?;
    let sm = tinfo("slot_mapping", slot_mapping)?;
    if kc.shape.len() != 2 || kp.shape.len() != 2 {
        return Err(PyValueError::new_err(
            "kv_c must be [T, 512] and k_pe [T, 64] bf16",
        ));
    }
    let tokens = kc.shape[0];
    if kc.shape != [tokens, DIM] || kp.shape != [tokens, PE_DIM] {
        return Err(PyValueError::new_err(format!(
            "kv_c {:?} / k_pe {:?}: need [T, 512] / [T, 64] (kv_lora_rank 512, pe 64)",
            kc.shape, kp.shape
        )));
    }
    need("kv_c", &kc, "torch.bfloat16", &[tokens, DIM])?;
    need("k_pe", &kp, "torch.bfloat16", &[tokens, PE_DIM])?;
    need("slot_mapping", &sm, "torch.int64", &sm.shape.clone())?;
    if sm.shape[0] < tokens || sm.stride.last() != Some(&1) {
        return Err(PyValueError::new_err(
            "slot_mapping must be [>=T] int64, contiguous",
        ));
    }
    if kv.shape.last().copied() != Some(ROW_BYTES) || kv.shape.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "kv_cache must have last dim {ROW_BYTES} (nvfp4_ds_mla rows)"
        )));
    }
    need("kv_cache", &kv, "torch.uint8", &kv.shape.clone())?;
    if kv.stride.last() != Some(&1) {
        return Err(PyValueError::new_err("kv_cache: last dim contiguous"));
    }
    for (n, i) in [("k_pe", &kp), ("slot_mapping", &sm), ("kv_cache", &kv)] {
        if i.device != kc.device {
            return Err(PyValueError::new_err(format!(
                "{n} on cuda:{} != cuda:{}",
                i.device, kc.device
            )));
        }
    }
    if tokens == 0 {
        return Ok(());
    }
    let ord = kc.device;
    let u = |x: usize| Arg::U32(x as u32);
    let args = vec![
        Arg::Ptr(kc.ptr),
        Arg::Ptr(kp.ptr),
        Arg::Ptr(sm.ptr),
        Arg::Ptr(kv.ptr),
        u(tokens),
        u(kc.stride[0]),
        u(kp.stride[0]),
    ];
    let grid = (tokens as u32, 1, 1);
    py.detach(move || {
        crate::guard_py("nvfp4_ds_mla_quant_store_cuda", move || {
            let run = || -> Result<(), String> {
                let f = function(FAMILY, "nvfp4_ds_mla_quant_store", ord)?;
                // writer: 64 threads (mirror of the SM100 nvfp4 writer's
                // warp0-lantent / warp1-rope geometry)
                launch(f, grid, (64, 1, 1), 64 * 4, stream_ptr, &args)
            };
            run().map_err(|e| {
                PyRuntimeError::new_err(format!("NVFP4-DSMLA writer launch failed: {e}"))
            })
        })
    })
}