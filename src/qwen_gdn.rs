// SPDX-License-Identifier: Apache-2.0
//! K-GDN1: fused Gated-DeltaNet decode step for Qwen3.5-class hybrid models
//! (qwen3.8-27b-fable-distill) — shared semantic contract + CPU reference.
//!
//! Replaces, for the NON-spec packed decode path of vLLM 0.30.0
//! (`qwen_gdn_linear_attn.py:_forward_core_decode_non_spec` + the trailing
//! `_rms_norm_gated_cuda`), the chain
//!   `a.contiguous()`, `b.contiguous()`                      (2 copy kernels)
//!   FLA `fused_recurrent_gated_delta_rule_packed_decode`    (Triton)
//!   FLA `layer_norm_fwd` (RMSNormGated, norm_before_gate)   (Triton)
//! with ONE kernel per GDN layer (GPU twin: src/qwen_gdn_gpu.rs, cargo
//! feature `qwen-gdn-kernels`). `causal_conv1d_update` stays stock (its
//! output is this op's `mixed_qkv` input).
//!
//! Layout contract (TP=1, Qwen3.5 non-interleaved layout; every dim checked):
//!   mixed_qkv  [T, 2*H*K + HV*V]  post-conv activations; row = [q(H*K) | k(H*K) | v(HV*V)]
//!   z          [T, HV, V]         output gate (the `z` half of in_proj_qkvz)
//!   ba         [T, 2*HV]          row = [b(HV) | a(HV)]  (split_ba: ba.chunk(2) -> b, a)
//!   a_log      [HV]  dt_bias [HV]  norm_w [V]   (fp32)
//!   state      [S, HV, V, K]      recurrent state, fp32, updated IN PLACE at slot state_idx[t]
//!   state_idx  [T] i32            slot per token; slot <= 0 = NULL_BLOCK -> out row = 0, state untouched
//!   out        [T, HV, V]
//! Per (t, hv), h = hv / (HV / H):
//!   q, k  = l2norm(q_h), l2norm(k_h)   (x / sqrt(sum x^2 + 1e-6)); q *= scale (= K^-0.5)
//!   g     = -exp(a_log) * softplus(a + dt_bias)   (softplus(x) = x if x > 20)
//!   beta  = sigmoid(b)
//!   S     = S * exp(g)
//!   v'    = beta * (v - S k)
//!   S     = S + v' k^T
//!   o     = S q
//!   out   = rmsnorm(o) * norm_w * act(z)   (rstd over V, eps = rms_norm_eps;
//!           act = silu (z*sigmoid(z)) or sigmoid)
//! This is statement-for-statement vLLM's math (fused_recurrent.py:309-332,
//! layernorm_guard.py:119-172). One deliberate deviation, owned by the
//! numerics gate: the norm reads `o` in fp32 instead of the bf16-rounded
//! `core_attn_out` the stock two-kernel chain round-trips through.

use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Output-gate activation, as encoded for the GPU kernel's `ACT` generic.
pub const ACT_SILU: i32 = 0;
pub const ACT_SIGMOID: i32 = 1;
pub const L2_EPS: f32 = 1e-6;
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// Shape contract shared by the CPU reference and the GPU host wrapper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnDims {
    pub t: usize,
    pub h: usize,
    pub hv: usize,
    pub k: usize,
    pub v: usize,
    pub slots: usize,
}

impl GdnDims {
    pub fn qkv_width(&self) -> usize {
        2 * self.h * self.k + self.hv * self.v
    }
    /// Every precondition the kernel relies on. Hard errors, never clamps.
    pub fn validate(&self) -> Result<(), String> {
        let GdnDims { h, hv, k, v, .. } = *self;
        if h == 0 || hv == 0 || k == 0 || v == 0 {
            return Err(format!("degenerate GDN dims {self:?}"));
        }
        if hv % h != 0 {
            return Err(format!("HV={hv} must be a multiple of H={h}"));
        }
        // GPU twin addresses q/k/v with one [1, K] column partition.
        if k != v {
            return Err(format!("K-GDN1 requires K == V (got K={k}, V={v})"));
        }
        if !k.is_power_of_two() || k > 256 {
            return Err(format!("K-GDN1 requires power-of-two K <= 256 (got {k})"));
        }
        Ok(())
    }
}

fn softplus(x: f32) -> f32 {
    if x <= SOFTPLUS_THRESHOLD {
        (1.0 + x.exp()).ln()
    } else {
        x
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Pure-Rust reference of one fused decode step. `state` is updated in
/// place exactly like the kernel; returns `out` [T*HV*V] (fp32).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_step_ref(
    d: GdnDims,
    mixed_qkv: &[f32],
    z: &[f32],
    ba: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_w: &[f32],
    state: &mut [f32],
    state_idx: &[i32],
    scale: f32,
    norm_eps: f32,
    act: i32,
) -> Result<Vec<f32>, String> {
    d.validate()?;
    let GdnDims {
        t,
        h,
        hv,
        k,
        v,
        slots,
    } = d;
    let w = d.qkv_width();
    let checks = [
        ("mixed_qkv", mixed_qkv.len(), t * w),
        ("z", z.len(), t * hv * v),
        ("ba", ba.len(), t * 2 * hv),
        ("a_log", a_log.len(), hv),
        ("dt_bias", dt_bias.len(), hv),
        ("norm_w", norm_w.len(), v),
        ("state", state.len(), slots * hv * v * k),
        ("state_idx", state_idx.len(), t),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(format!("{name}: expected {want} elements, got {got}"));
        }
    }
    if act != ACT_SILU && act != ACT_SIGMOID {
        return Err(format!("unknown output-gate activation code {act}"));
    }
    let group = hv / h;
    let mut out = vec![0f32; t * hv * v];
    let mut q = vec![0f32; k];
    let mut kk = vec![0f32; k];
    let mut vv = vec![0f32; v];
    let mut o = vec![0f32; v];
    for ti in 0..t {
        let slot = state_idx[ti];
        if slot <= 0 {
            continue; // NULL_BLOCK: zero output, state untouched
        }
        let slot = slot as usize;
        if slot >= slots {
            return Err(format!(
                "state_idx[{ti}]={slot} out of range (slots={slots})"
            ));
        }
        let row = &mixed_qkv[ti * w..(ti + 1) * w];
        for hvi in 0..hv {
            let hi = hvi / group;
            q.copy_from_slice(&row[hi * k..(hi + 1) * k]);
            kk.copy_from_slice(&row[h * k + hi * k..h * k + (hi + 1) * k]);
            vv.copy_from_slice(&row[2 * h * k + hvi * v..2 * h * k + (hvi + 1) * v]);
            let qn = 1.0 / (q.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
            let kn = 1.0 / (kk.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
            q.iter_mut().for_each(|x| *x *= qn * scale);
            kk.iter_mut().for_each(|x| *x *= kn);

            let b = ba[ti * 2 * hv + hvi];
            let a = ba[ti * 2 * hv + hv + hvi];
            let g = -a_log[hvi].exp() * softplus(a + dt_bias[hvi]);
            let decay = g.exp();
            let beta = sigmoid(b);

            let base = (slot * hv + hvi) * v * k;
            let s = &mut state[base..base + v * k];
            for vi in 0..v {
                let srow = &mut s[vi * k..(vi + 1) * k];
                let mut kv = 0f32;
                for (sx, kx) in srow.iter_mut().zip(&kk) {
                    *sx *= decay;
                    kv += *sx * kx;
                }
                let dv = beta * (vv[vi] - kv);
                let mut acc = 0f32;
                for ((sx, kx), qx) in srow.iter_mut().zip(&kk).zip(&q) {
                    *sx += dv * kx;
                    acc += *sx * qx;
                }
                o[vi] = acc;
            }
            let var = o.iter().map(|x| x * x).sum::<f32>() / v as f32;
            let rstd = 1.0 / (var + norm_eps).sqrt();
            let zrow = &z[(ti * hv + hvi) * v..(ti * hv + hvi + 1) * v];
            let orow = &mut out[(ti * hv + hvi) * v..(ti * hv + hvi + 1) * v];
            for vi in 0..v {
                let gate = if act == ACT_SILU {
                    zrow[vi] * sigmoid(zrow[vi])
                } else {
                    sigmoid(zrow[vi])
                };
                orow[vi] = o[vi] * rstd * norm_w[vi] * gate;
            }
        }
    }
    Ok(out)
}

/// Python entry: CPU reference / oracle of the K-GDN1 GPU kernel.
/// All float inputs fp32 NumPy; `state` [S, HV, V, K] is mutated in place.
/// Returns `out` [T, HV, V] fp32.
#[pyfunction]
#[pyo3(signature = (mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, num_k_heads, scale, norm_eps, act))]
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_fused_ref<'py>(
    py: Python<'py>,
    mixed_qkv: PyReadonlyArray2<'py, f32>,
    z: PyReadonlyArray3<'py, f32>,
    ba: PyReadonlyArray2<'py, f32>,
    a_log: PyReadonlyArray1<'py, f32>,
    dt_bias: PyReadonlyArray1<'py, f32>,
    norm_w: PyReadonlyArray1<'py, f32>,
    mut state: PyReadwriteArray4<'py, f32>,
    state_idx: PyReadonlyArray1<'py, i32>,
    num_k_heads: usize,
    scale: f32,
    norm_eps: f32,
    act: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let sshape = state.shape().to_vec();
    let zshape = z.shape().to_vec();
    let d = GdnDims {
        t: mixed_qkv.shape()[0],
        h: num_k_heads,
        hv: sshape[1],
        k: sshape[3],
        v: sshape[2],
        slots: sshape[0],
    };
    if zshape != [d.t, d.hv, d.v] {
        return Err(PyValueError::new_err(format!(
            "z must be [T, HV, V] = {:?}, got {zshape:?}",
            [d.t, d.hv, d.v]
        )));
    }
    let nc = |name: &str| PyValueError::new_err(format!("{name} must be C-contiguous"));
    let mq = mixed_qkv.as_slice().map_err(|_| nc("mixed_qkv"))?;
    let zz = z.as_slice().map_err(|_| nc("z"))?;
    let bb = ba.as_slice().map_err(|_| nc("ba"))?;
    let al = a_log.as_slice().map_err(|_| nc("a_log"))?;
    let dt = dt_bias.as_slice().map_err(|_| nc("dt_bias"))?;
    let nw = norm_w.as_slice().map_err(|_| nc("norm_w"))?;
    let si = state_idx
        .as_slice()
        .map_err(|_| PyValueError::new_err("state_idx must be C-contiguous"))?;
    let st = state
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let out = crate::guard_py("gdn_decode_fused_ref", || {
        gdn_decode_step_ref(d, mq, zz, bb, al, dt, nw, st, si, scale, norm_eps, act)
            .map_err(PyValueError::new_err)
    })?;
    let arr = out.into_pyarray(py);
    arr.reshape([d.t, d.hv, d.v])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn dims() -> GdnDims {
        GdnDims {
            t: 3,
            h: 2,
            hv: 6,
            k: 8,
            v: 8,
            slots: 5,
        }
    }

    #[test]
    fn contract_rejects_bad_dims() {
        let mut d = dims();
        d.hv = 5;
        assert!(d.validate().is_err());
        let mut d = dims();
        d.v = 16;
        assert!(d.validate().is_err());
        assert!(dims().validate().is_ok());
    }

    #[test]
    fn null_slot_zero_output_and_state_untouched() {
        let d = dims();
        let mut s = 7u64;
        let mq: Vec<f32> = (0..d.t * d.qkv_width()).map(|_| lcg(&mut s)).collect();
        let z: Vec<f32> = (0..d.t * d.hv * d.v).map(|_| lcg(&mut s)).collect();
        let ba: Vec<f32> = (0..d.t * 2 * d.hv).map(|_| lcg(&mut s)).collect();
        let al = vec![0.1f32; d.hv];
        let dt = vec![0.2f32; d.hv];
        let nw = vec![1.0f32; d.v];
        let mut st: Vec<f32> = (0..d.slots * d.hv * d.v * d.k)
            .map(|_| lcg(&mut s))
            .collect();
        let before = st.clone();
        let idx = vec![0i32, -1, 0];
        let out = gdn_decode_step_ref(
            d, &mq, &z, &ba, &al, &dt, &nw, &mut st, &idx, 0.35, 1e-6, ACT_SILU,
        )
        .unwrap();
        assert!(out.iter().all(|x| *x == 0.0));
        assert_eq!(st, before);
    }

    #[test]
    fn zero_state_zero_decay_rank_one() {
        // With S0 = 0 the step is closed-form: S1 = beta * v k^T and
        // o = beta * v * (k.q); check one head by hand.
        let d = GdnDims {
            t: 1,
            h: 1,
            hv: 1,
            k: 4,
            v: 4,
            slots: 2,
        };
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let mq: Vec<f32> = q.iter().chain(&k).chain(&v).copied().collect();
        let z = vec![100.0f32; 4]; // silu(100) ~= 100
        let ba = vec![0.0f32, 0.0]; // beta = 0.5
        let mut st = vec![0f32; 2 * 16];
        let out = gdn_decode_step_ref(
            d,
            &mq,
            &z,
            &ba,
            &[0.0],
            &[0.0],
            &[1.0; 4],
            &mut st,
            &[1],
            1.0,
            0.0,
            ACT_SILU,
        )
        .unwrap();
        // o = 0.5 * v ; rms(o) = 0.5*sqrt(7.5) ; out = o/rms * 100
        let rms = 0.5f32 * 7.5f32.sqrt();
        for i in 0..4 {
            let want = 0.5 * v[i] / rms * 100.0;
            assert!((out[i] - want).abs() < 1e-3, "{i}: {} vs {want}", out[i]);
            assert!((st[16 + i * 4] - 0.5 * v[i]).abs() < 1e-6);
        }
    }
}
