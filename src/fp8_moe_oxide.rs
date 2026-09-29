// SPDX-License-Identifier: Apache-2.0
//! FP8 block-scaled MoE decode host op (feature `oxide-kernels`): launches
//! kernels-oxide/fp8_moe (sm_120a SASS, ptxas 13.0) on torch's stream.
//! Six launches, all grids a function of M only (no device->host reads):
//! route, quant(x), fc1+silu_and_mul, quant(h), fc2, combine.
//! Oracle/bench + vLLM wiring: suffix_hybrid/kernels/fp8_moe.py.

use crate::nvfp4_moe_oxide::{info, need};
use crate::oxide::{function, launch, Arg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub const FAMILY: &str = "fp8_moe";

/// Activation quant constants = vLLM v0.30 (fp8_moe.py `quant_params`):
/// x (per_token_group_quant_fp8): eps 1e-10 on amax, no scale floor;
/// h non-UE8M0 (silu_and_mul_per_block_quant): no eps, scale floor
/// 1/(448*512); h UE8M0 (silu_and_mul + per_token_group_quant_fp8): as x.
const EPS: f32 = 1e-10;
const MIN_SCALE: f32 = 1.0 / (448.0 * 512.0);

/// Routed-expert forward for M decode tokens. Shapes (E experts, H hidden,
/// I intermediate, K top-k, P = M*K):
///   x bf16 [M, H]; topk_ids int32/int64 [M, K]; topk_w f32 [M, K]
///   w13 e4m3/uint8 [E, 2I, H] ([gate; up]); w13_s f32 [E, 2I/128, H/128]
///   w2 e4m3/uint8 [E, H, I]; w2_s f32 [E, H/128, I/128]
///   ws_* preallocated scratch; out bf16 [M, H]
/// ue8m0: activation scales rounded up to powers of two (vLLM with DeepGEMM
/// E8M0 on). id_base: expert parallel offset (vLLM linear expert_map:
/// ep_rank * E); topk_ids are GLOBAL ids, pairs outside [id_base,
/// id_base + E) contribute nothing on this rank.
#[pyfunction]
#[pyo3(signature = (x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as, ws_inter, ws_hq, ws_hs, ws_y, ws_route, out, ue8m0, id_base, stream_ptr))]
#[allow(clippy::too_many_arguments)]
pub fn fp8_moe_cuda<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    topk_ids: &Bound<'py, PyAny>,
    topk_w: &Bound<'py, PyAny>,
    w13: &Bound<'py, PyAny>,
    w13_s: &Bound<'py, PyAny>,
    w2: &Bound<'py, PyAny>,
    w2_s: &Bound<'py, PyAny>,
    ws_aq: &Bound<'py, PyAny>,
    ws_as: &Bound<'py, PyAny>,
    ws_inter: &Bound<'py, PyAny>,
    ws_hq: &Bound<'py, PyAny>,
    ws_hs: &Bound<'py, PyAny>,
    ws_y: &Bound<'py, PyAny>,
    ws_route: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    ue8m0: bool,
    id_base: u32,
    stream_ptr: usize,
) -> PyResult<()> {
    let x = info("x", x)?;
    let ids = info("topk_ids", topk_ids)?;
    let tw = info("topk_w", topk_w)?;
    let w13 = info("w13", w13)?;
    let w13s = info("w13_s", w13_s)?;
    let w2 = info("w2", w2)?;
    let w2s = info("w2_s", w2_s)?;
    let aq = info("ws_aq", ws_aq)?;
    let a_s = info("ws_as", ws_as)?;
    let inter = info("ws_inter", ws_inter)?;
    let hq = info("ws_hq", ws_hq)?;
    let hs = info("ws_hs", ws_hs)?;
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
    if e == 0 || e > 256 || h % 128 != 0 || i % 128 != 0 || two_i != 2 * i {
        return Err(PyValueError::new_err(format!(
            "unsupported MoE shape E={e} H={h} I={i} (E<=256, H,I % 128 == 0)"
        )));
    }
    let f8 = ["torch.float8_e4m3fn", "torch.uint8"];
    need("x", &x, &["torch.bfloat16"], Some(&[m, h]), 0)?;
    need("topk_ids", &ids, &["torch.int32", "torch.int64"], Some(&[m, k]), 0)?;
    need("topk_w", &tw, &["torch.float32"], Some(&[m, k]), 0)?;
    need("w13", &w13, &f8, Some(&[e, 2 * i, h]), 0)?;
    need("w13_s", &w13s, &["torch.float32"], Some(&[e, 2 * i / 128, h / 128]), 0)?;
    need("w2", &w2, &f8, Some(&[e, h, i]), 0)?;
    need("w2_s", &w2s, &["torch.float32"], Some(&[e, h / 128, i / 128]), 0)?;
    need("ws_aq", &aq, &["torch.uint8"], None, m * h)?;
    need("ws_as", &a_s, &["torch.float32"], None, m * h / 128)?;
    need("ws_inter", &inter, &["torch.float32"], None, p * i)?;
    need("ws_hq", &hq, &["torch.uint8"], None, p * i)?;
    need("ws_hs", &hs, &["torch.float32"], None, p * i / 128)?;
    need("ws_y", &y, &["torch.float32"], None, p * h)?;
    need("ws_route", &route, &["torch.int32"], None, 3 * e + p)?;
    need("out", &out, &["torch.bfloat16"], Some(&[m, h]), 0)?;
    for (nm, t) in [
        ("topk_ids", &ids),
        ("topk_w", &tw),
        ("w13", &w13),
        ("w13_s", &w13s),
        ("w2", &w2),
        ("w2_s", &w2s),
        ("ws_aq", &aq),
        ("ws_as", &a_s),
        ("ws_inter", &inter),
        ("ws_hq", &hq),
        ("ws_hs", &hs),
        ("ws_y", &y),
        ("ws_route", &route),
        ("out", &out),
    ] {
        if t.device != x.device {
            return Err(PyValueError::new_err(format!("{nm} on another device")));
        }
    }
    // the kernels read weight / quantized-activation rows with 16 B loads
    for (nm, t) in [("w13", &w13), ("w2", &w2), ("ws_aq", &aq), ("ws_hq", &hq)] {
        if t.ptr % 16 != 0 {
            return Err(PyValueError::new_err(format!("{nm} not 16-byte aligned")));
        }
    }
    let ids_i64 = u32::from(ids.dtype == "torch.int64");
    let slots = e.min(p);
    let r = route.ptr;
    let (slot_expert, slot_off, slot_cnt, pair_list) =
        (r, r + 4 * e as u64, r + 8 * e as u64, r + 12 * e as u64);
    let u = |v: usize| Arg::U32(v as u32);
    let (h_eps, h_min) = if ue8m0 { (EPS, 0.0) } else { (0.0, MIN_SCALE) };
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
        Arg::Ptr(a_s.ptr),
        u(m),
        u(h),
        u(x.stride[0]),
        Arg::F32(EPS),
        Arg::F32(0.0),
        Arg::U32(u32::from(ue8m0)),
    ];
    let fc1_args = [
        Arg::Ptr(inter.ptr),
        Arg::Ptr(aq.ptr),
        Arg::Ptr(a_s.ptr),
        Arg::Ptr(w13.ptr),
        Arg::Ptr(w13s.ptr),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        u(k),
        u(h),
        u(i),
    ];
    let qh_args = [
        Arg::Ptr(inter.ptr),
        Arg::U32(1),
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hs.ptr),
        u(p),
        u(i),
        u(i),
        Arg::F32(h_eps),
        Arg::F32(h_min),
        Arg::U32(u32::from(ue8m0)),
    ];
    let fc2_args = [
        Arg::Ptr(y.ptr),
        Arg::Ptr(hq.ptr),
        Arg::Ptr(hs.ptr),
        Arg::Ptr(w2.ptr),
        Arg::Ptr(w2s.ptr),
        Arg::Ptr(tw.ptr),
        Arg::Ptr(slot_expert),
        Arg::Ptr(slot_off),
        Arg::Ptr(slot_cnt),
        Arg::Ptr(pair_list),
        u(h),
        u(i),
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
    // quant: one warp per 128-col group, 4 groups per 128-thread CTA
    let grids = [
        (1u32, 1u32),
        ((h / 128).div_ceil(4) as u32, m as u32),
        ((i / 32) as u32, slots as u32),
        ((i / 128).div_ceil(4) as u32, p as u32),
        ((h / 32) as u32, slots as u32),
        ((m * h).div_ceil(256) as u32, 1),
    ];
    let ordinal = x.device;
    let route_smem = (2 * e * 4) as u32;
    py.detach(move || {
        crate::guard_py("fp8_moe_cuda", move || {
            let f = |entry: &str| function(FAMILY, entry, ordinal).map_err(PyRuntimeError::new_err);
            let e_ = |e: String| PyRuntimeError::new_err(format!("fp8 moe launch: {e}"));
            let steps: [(&str, usize, u32, u32, &[Arg]); 6] = [
                ("moe_route", 0, 256, route_smem, &route_args),
                ("moe_quant_rows", 1, 128, 0, &qx_args),
                ("moe_fc1", 2, 128, 0, &fc1_args),
                ("moe_quant_rows", 3, 128, 0, &qh_args),
                ("moe_fc2", 4, 128, 0, &fc2_args),
                ("moe_combine", 5, 256, 0, &comb_args),
            ];
            for (entry, gi, threads, smem, args) in steps {
                launch(
                    f(entry)?,
                    (grids[gi].0, grids[gi].1, 1),
                    (threads, 1, 1),
                    smem,
                    stream_ptr,
                    args,
                )
                .map_err(e_)?;
            }
            Ok(())
        })
    })
}
