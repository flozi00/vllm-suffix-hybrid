// SPDX-License-Identifier: Apache-2.0
//! K-GDN1 (qwen3.8-27b fused Gated-DeltaNet decode step) — cuda-oxide SIMT
//! port. Target: sm_120 SASS from ptxas 13.0.x (PTX ISA pinned to the CUDA
//! 13.0 floor); the node driver (580.x) has no PTX JIT, so only the cubin
//! ships. Semantics/layout contract: plugin repo src/qwen_gdn.rs (CPU oracle),
//! numpy SIMT twin: suffix_hybrid/kernels/qwen_gdn_oxide.py (transcribes THIS
//! file statement for statement, incl. the reduction order).
//!
//! ## Interface (what the host plugin launches) — kept in sync with
//! ## kernels-oxide/kgdn1/interface.json
//! PTX entry `kgdn1_decode_h16_hv48_k128`, grid = (HV=48, T, 1),
//! block = (256, 1, 1), dynamic smem = 0. Params, in order (all 8-byte
//! pointers are device addresses of torch-owned memory):
//!   0 out        *mut u16   bf16 [T, HV, V] contiguous (fully written)
//!   1 mixed_qkv  *const u16 bf16 [T, 2HK+HV*V], row stride `qkv_stride`
//!   2 z          *const u16 bf16 [T, HV, V], row stride `z_stride`, head stride V
//!   3 ba         *const u16 bf16 [T, 2HV] contiguous, row = [b | a]
//!   4 a_log      *const f32 [HV]
//!   5 dt_bias    *const f32 [HV]
//!   6 norm_w     *const f32 [V]
//!   7 state      *mut f32   [S, HV, V, K]; slot stride `state_stride` (vLLM's
//!                            padded as_strided page), inner [HV,V,K] dense
//!   8 state_idx  *const i32 [T]  (slot <= 0 = NULL block: zero row, no state IO)
//!   9 slots        u32  S (slot >= S traps: never silently wrong)
//!  10 qkv_stride   u32  elements
//!  11 z_stride     u32  elements
//!  12 state_stride u32  elements
//!  13 act          u32  0 = silu gate, 1 = sigmoid gate
//!  14 scale        f32  K^-0.5
//!  15 norm_eps     f32  rms_norm_eps
//!
//! ## Design (V-chunked SIMT, no tensor cores)
//! One CTA per (value head hv, token t); 8 warps, warp w owns value rows
//! [16w, 16w+16) (BV = 16). Lane l owns the K-slice {l, l+32, l+64, l+96}:
//! every state row is read and written as 128 consecutive f32 across the warp
//! (fully coalesced 512 B), and a thread holds only 4 state values at a time
//! (the cutile whole-tile build hit REG:254). Per row:
//!   s   = decay * S[v, :]
//!   pk  = warp_sum(s . k), pq = warp_sum(s . q)        (xor butterflies)
//!   dv  = beta * (v[v] - pk);  S[v, :] = s + dv * k    (store back)
//!   o_v = pq + dv * (k . q)                            (= S_new[v,:] . q)
//! o goes to shared memory; warp 0 reduces sum(o^2); every thread < 128 then
//! writes out[v] = o_v * rstd * w[v] * gate(z[v]) as bf16 (RNE).
//! Estimate: ~40-56 regs/thread, 0 spills; 256 thr/CTA -> 4-6 CTAs/SM
//! (48-warp SM limit); smem 516 B/CTA. UNVERIFIED until ptxas -v in CI.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::float::{ex2_approx_f32, lg2_approx_f32};
    use cuda_device::{SharedArray, debug, kernel, launch_bounds, thread, warp};

    pub const H: u32 = 16;
    pub const HV: u32 = 48;
    pub const K: u32 = 128; // == V
    pub const WARP_ROWS: u32 = 16; // BV
    const LOG2E: f32 = core::f32::consts::LOG2_E;
    const LN2: f32 = core::f32::consts::LN_2;

    #[inline(always)]
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    /// Round-to-nearest-even f32 -> bf16 (torch `.to(torch.bfloat16)`).
    #[inline(always)]
    fn f32_to_bf16(x: f32) -> u16 {
        let b = x.to_bits();
        if (b & 0x7fff_ffff) > 0x7f80_0000 {
            return 0x7fc0; // NaN
        }
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    #[inline(always)]
    fn exp_approx(x: f32) -> f32 {
        ex2_approx_f32(x * LOG2E)
    }

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp_approx(-x))
    }

    /// Xor butterfly: every lane ends with the warp sum (fixed order 16..1).
    #[inline(always)]
    fn warp_sum(mut v: f32) -> f32 {
        v = v + warp::shuffle_xor_f32(v, 16);
        v = v + warp::shuffle_xor_f32(v, 8);
        v = v + warp::shuffle_xor_f32(v, 4);
        v = v + warp::shuffle_xor_f32(v, 2);
        v = v + warp::shuffle_xor_f32(v, 1);
        v
    }

    #[kernel]
    #[launch_bounds(256, 4)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kgdn1_decode_h16_hv48_k128(
        out: *mut u16,
        mixed_qkv: *const u16,
        z: *const u16,
        ba: *const u16,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const f32,
        state: *mut f32,
        state_idx: *const i32,
        slots: u32,
        qkv_stride: u32,
        z_stride: u32,
        state_stride: u32,
        act: u32,
        scale: f32,
        norm_eps: f32,
    ) {
        static mut O: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut RSTD: SharedArray<f32, 1> = SharedArray::UNINIT;

        let hv = thread::blockIdx_x();
        let t = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        let w = tid / 32;
        let lane = tid % 32;
        let out_row = unsafe { out.add(((t * HV + hv) * K) as usize) };

        let slot = unsafe { *state_idx.add(t as usize) };
        if slot <= 0 {
            // NULL block: zero output row, state untouched (CTA-uniform exit).
            if tid < K {
                unsafe { *out_row.add(tid as usize) = 0 };
            }
            return;
        }
        if slot as u32 >= slots {
            debug::trap(); // malformed slot: fail loud, never write elsewhere
        }

        // ---- q / k slices (bf16 -> f32), L2 norm, scale -----------------
        let h = hv / (HV / H);
        let row = unsafe { mixed_qkv.add((t * qkv_stride) as usize) };
        let mut q = [0.0f32; 4];
        let mut k = [0.0f32; 4];
        let mut j = 0;
        while j < 4 {
            let c = lane + 32 * j as u32;
            q[j] = bf16_to_f32(unsafe { *row.add((h * K + c) as usize) });
            k[j] = bf16_to_f32(unsafe { *row.add((H * K + h * K + c) as usize) });
            j += 1;
        }
        let qss = warp_sum(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        let kss = warp_sum(k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3]);
        let qn = scale / (qss + 1e-6).sqrt();
        let kn = 1.0 / (kss + 1e-6).sqrt();
        let mut j = 0;
        while j < 4 {
            q[j] *= qn;
            k[j] *= kn;
            j += 1;
        }
        let kq = warp_sum(k[0] * q[0] + k[1] * q[1] + k[2] * q[2] + k[3] * q[3]);

        // ---- gating scalars (per CTA; every thread computes) -------------
        let brow = unsafe { ba.add((t * 2 * HV) as usize) };
        let b = bf16_to_f32(unsafe { *brow.add(hv as usize) });
        let a = bf16_to_f32(unsafe { *brow.add((HV + hv) as usize) });
        let x = a + unsafe { *dt_bias.add(hv as usize) };
        let softplus = if x <= 20.0 { lg2_approx_f32(1.0 + exp_approx(x)) * LN2 } else { x };
        let decay = exp_approx(-exp_approx(unsafe { *a_log.add(hv as usize) }) * softplus);
        let beta = sigmoid(b);

        // ---- delta rule over this warp's 16 value rows --------------------
        let vrow = unsafe { row.add((2 * H * K + hv * K) as usize) };
        // usize math: slot * page stride exceeds u32 on large KV pools.
        let s_head = unsafe {
            state.add(slot as usize * state_stride as usize + (hv * K * K) as usize)
        };
        let mut r = 0;
        while r < WARP_ROWS {
            let v = w * WARP_ROWS + r;
            let p = unsafe { s_head.add((v * K) as usize) };
            let mut s = [0.0f32; 4];
            let mut j = 0;
            while j < 4 {
                s[j] = unsafe { *p.add((lane + 32 * j as u32) as usize) } * decay;
                j += 1;
            }
            let pk = warp_sum(s[0] * k[0] + s[1] * k[1] + s[2] * k[2] + s[3] * k[3]);
            let pq = warp_sum(s[0] * q[0] + s[1] * q[1] + s[2] * q[2] + s[3] * q[3]);
            let dv = beta * (bf16_to_f32(unsafe { *vrow.add(v as usize) }) - pk);
            let mut j = 0;
            while j < 4 {
                unsafe { *p.add((lane + 32 * j as u32) as usize) = s[j] + dv * k[j] };
                j += 1;
            }
            if lane == 0 {
                unsafe { O[v as usize] = pq + dv * kq };
            }
            r += 1;
        }
        thread::sync_threads();

        // ---- RMSNormGated epilogue (norm_before_gate) ---------------------
        if w == 0 {
            let mut ss = 0.0f32;
            let mut j = 0;
            while j < 4 {
                let o = unsafe { O[(lane + 32 * j) as usize] };
                ss += o * o;
                j += 1;
            }
            ss = warp_sum(ss);
            if lane == 0 {
                unsafe { RSTD[0] = 1.0 / (ss * (1.0 / K as f32) + norm_eps).sqrt() };
            }
        }
        thread::sync_threads();
        if tid < K {
            let zv = bf16_to_f32(unsafe { *z.add((t * z_stride + hv * K + tid) as usize) });
            let sg = sigmoid(zv);
            let gate = if act == 0 { zv * sg } else { sg };
            let y = unsafe { O[tid as usize] * RSTD[0] } * unsafe { *norm_w.add(tid as usize) } * gate;
            unsafe { *out_row.add(tid as usize) = f32_to_bf16(y) };
        }
    }
}

fn main() {
    // Build-only crate: `cargo oxide build kgdn1 --arch sm_120` emits the PTX
    // that CI assembles with ptxas 13.0.88; the plugin launches the cubin.
}
