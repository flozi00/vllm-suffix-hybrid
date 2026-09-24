// SPDX-License-Identifier: Apache-2.0
//! K-GDN1 GPU op: fused Gated-DeltaNet decode step (qwen3.8-27b) as ONE
//! cutile-rs tile kernel per GDN layer. Semantics + layout contract:
//! src/qwen_gdn.rs (the CPU reference is the oracle this kernel must match).
//!
//! Built only with cargo feature `qwen-gdn-kernels` (CUDA 13.x headers at
//! build time; pods are driver-only). Type-check / Tile IR lowering is
//! reproducible on a GPU-less host with just the CUDA headers:
//!   CUDA_TOOLKIT_PATH=<dir with include/cuda.h,curand.h> \
//!     cargo test --features qwen-gdn-kernels qwen_gdn_gpu
//! (compile-only `KernelCompiler` test at the bottom: kernel -> Tile IR ->
//! bytecode for sm_120; `tileiras` (bytecode -> cubin) is CI/pod-side).
//!
//! ## Kernel shape
//! Grid = (T, HV, 1), inferred from the `out` partition [1, 1, V] over
//! [T, HV, V]: one tile block per (token, value head), holding the whole
//! [V, K] fp32 state tile (the vLLM CUDA MTP kernel's grid choice too,
//! csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu:400 — the full V row
//! is what lets the RMSNormGated epilogue fuse). The stock FLA Triton decode
//! kernel runs grid (V/32, T*HV) with num_warps=1 (fused_recurrent.py:441-484)
//! — 1 warp/SM at batch 1 — and needs a second Triton kernel for the norm.
//!
//! ## Host contract (fail loud, never silent-fallback)
//! Torch owns every buffer; we borrow data_ptrs (`borrow_raw_parts`), launch
//! on torch's CURRENT stream (`Stream::borrow_raw`) with `async_on` — NOT
//! `sync_on`, which calls cuStreamSynchronize after every launch
//! (cuda-async-0.3.1/src/device_operation.rs:428-436): a host stall per layer
//! and illegal inside CUDA-graph capture. GIL released; panic boundary via
//! `crate::guard_py`. Any dtype/shape/stride/device mismatch is a hard error.

#[cfg(feature = "qwen-gdn-kernels")]
pub use device::*;

#[cfg(feature = "qwen-gdn-kernels")]
mod device {
    use crate::qwen_gdn::{GdnDims, ACT_SIGMOID, ACT_SILU};
    use cutile::cuda_core::{Device, Stream};
    use cutile::half::bf16;
    use cutile::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use std::sync::{Arc, Mutex};

    #[cutile::module]
    pub mod qwen_gdn_kernels {
        use cutile::core::*;

        /// One fused GDN decode step for token `t`, value head `hv`.
        /// H/HV: key/value head counts, K: head dim (== V), ACT: 0 silu, 1 sigmoid.
        #[cutile::entry()]
        pub unsafe fn gdn_decode_fused_k1<
            const H: i32,
            const HV: i32,
            const K: i32,
            const ACT: i32,
        >(
            out: &mut Tensor<bf16, { [1, 1, K] }>,
            mixed_qkv: &Tensor<bf16, { [-1, -1] }>,
            z: &Tensor<bf16, { [-1, -1, -1] }>,
            ba: &Tensor<bf16, { [-1, -1] }>,
            a_log: &Tensor<f32, { [-1] }>,
            dt_bias: &Tensor<f32, { [-1] }>,
            norm_w: &Tensor<f32, { [-1] }>,
            state: &Tensor<f32, { [-1, -1, -1, -1] }>,
            state_idx: &Tensor<i32, { [-1] }>,
            scale: f32,
            norm_eps: f32,
            inv_v: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let t: i32 = pid.0;
            let hv: i32 = pid.1;
            let h: i32 = hv / (HV / H);

            let idx_part: Partition<i32, { [1] }> = state_idx.partition(shape![1]);
            let slot_tile: Tile<i32, { [1] }> = idx_part.load([t]);
            let slot: i32 = tile_to_scalar(slot_tile.reshape(shape![]));

            if slot > 0i32 {
                // ---- q/k/v (bf16 -> f32), q/k L2 norm, q scale --------------
                let qkv_part: Partition<bf16, { [1, K] }> = mixed_qkv.partition(shape![1, K]);
                let q: Tile<f32, { [1, K] }> = convert_tile(qkv_part.load([t, h]));
                let k: Tile<f32, { [1, K] }> = convert_tile(qkv_part.load([t, H + h]));
                let v: Tile<f32, { [1, K] }> = convert_tile(qkv_part.load([t, 2i32 * H + hv]));

                let l2_eps: Tile<f32, { [1, 1] }> = constant(1e-6f32, shape![1, 1]);
                let q_ss: Tile<f32, { [1] }> = reduce_sum(q * q, 1i32);
                let k_ss: Tile<f32, { [1] }> = reduce_sum(k * k, 1i32);
                let q_inv: Tile<f32, { [1, 1] }> =
                    rsqrt(q_ss.reshape(shape![1, 1]) + l2_eps, ftz::Disabled);
                let k_inv: Tile<f32, { [1, 1] }> =
                    rsqrt(k_ss.reshape(shape![1, 1]) + l2_eps, ftz::Disabled);
                let scale_t: Tile<f32, { [1, K] }> = broadcast_scalar(scale, shape![1, K]);
                let q: Tile<f32, { [1, K] }> = q * q_inv.broadcast(shape![1, K]) * scale_t;
                let k: Tile<f32, { [1, K] }> = k * k_inv.broadcast(shape![1, K]);

                // ---- gating: g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
                let ba_part: Partition<bf16, { [1, 1] }> = ba.partition(shape![1, 1]);
                let b: Tile<f32, { [1, 1] }> = convert_tile(ba_part.load([t, hv]));
                let a: Tile<f32, { [1, 1] }> = convert_tile(ba_part.load([t, HV + hv]));
                let alog_part: Partition<f32, { [1] }> = a_log.partition(shape![1]);
                let dtb_part: Partition<f32, { [1] }> = dt_bias.partition(shape![1]);
                let alog: Tile<f32, { [1] }> = alog_part.load([hv]);
                let dtb: Tile<f32, { [1] }> = dtb_part.load([hv]);
                let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
                let x: Tile<f32, { [1, 1] }> = a + dtb.reshape(shape![1, 1]);
                let thresh: Tile<f32, { [1, 1] }> = constant(20.0f32, shape![1, 1]);
                let small: Tile<bool, { [1, 1] }> = le_tile(x, thresh);
                let softplus: Tile<f32, { [1, 1] }> = select(small, log(one + exp(x)), x);
                let g: Tile<f32, { [1, 1] }> = negf(exp(alog.reshape(shape![1, 1]))) * softplus;
                let decay: Tile<f32, { [1, 1] }> = exp(g);
                let beta: Tile<f32, { [1, 1] }> = true_div(one, one + exp(negf(b)));

                // ---- delta-rule update on the [V, K] state tile, in place ----
                let mut s_view: PartitionMut<f32, { [1, 1, K, K] }> =
                    unsafe { state.partition_full_mut(shape![1, 1, K, K]) };
                // Trap (device assert -> loud CUDA error) on slot >= S BEFORE
                // the unchecked load; the store below is checked as well.
                check_partition_access_mut(&s_view, [slot, hv, 0i32, 0i32]);
                let s0: Tile<f32, { [1, 1, K, K] }> = unsafe { s_view.load([slot, hv, 0i32, 0i32]) };
                let s: Tile<f32, { [K, K] }> =
                    s0.reshape(shape![K, K]) * decay.broadcast(shape![K, K]);
                let kb: Tile<f32, { [K, K] }> = k.broadcast(shape![K, K]);
                let kv: Tile<f32, { [K] }> = reduce_sum(s * kb, 1i32);
                let dv: Tile<f32, { [1, K] }> =
                    (v - kv.reshape(shape![1, K])) * beta.broadcast(shape![1, K]);
                let s: Tile<f32, { [K, K] }> =
                    s + dv.reshape(shape![K, 1]).broadcast(shape![K, K]) * kb;
                let o: Tile<f32, { [K] }> = reduce_sum(s * q.broadcast(shape![K, K]), 1i32);
                s_view.store(s.reshape(shape![1, 1, K, K]), [slot, hv, 0i32, 0i32]);

                // ---- RMSNormGated epilogue (norm_before_gate=True) ------------
                let o: Tile<f32, { [1, K] }> = o.reshape(shape![1, K]);
                let var: Tile<f32, { [1] }> = reduce_sum(o * o, 1i32);
                let var: Tile<f32, { [1, 1] }> =
                    var.reshape(shape![1, 1]) * broadcast_scalar(inv_v, shape![1, 1]);
                let rstd: Tile<f32, { [1, 1] }> =
                    rsqrt(var + broadcast_scalar(norm_eps, shape![1, 1]), ftz::Disabled);
                let w_part: Partition<f32, { [K] }> = norm_w.partition(shape![K]);
                let w: Tile<f32, { [K] }> = w_part.load([0i32]);
                let z_part: Partition<bf16, { [1, 1, K] }> = z.partition(shape![1, 1, K]);
                let zt: Tile<f32, { [1, 1, K] }> = convert_tile(z_part.load([t, hv, 0i32]));
                let zt: Tile<f32, { [1, K] }> = zt.reshape(shape![1, K]);
                let one_k: Tile<f32, { [1, K] }> = constant(1.0f32, shape![1, K]);
                let sig: Tile<f32, { [1, K] }> = true_div(one_k, one_k + exp(negf(zt)));
                let mut gate: Tile<f32, { [1, K] }> = sig;
                if ACT == 0i32 {
                    gate = zt * sig;
                }
                let y: Tile<f32, { [1, K] }> =
                    o * rstd.broadcast(shape![1, K]) * w.reshape(shape![1, K]) * gate;
                let y: Tile<bf16, { [1, K] }> = convert_tile(y);
                out.store(y.reshape(shape![1, 1, K]));
            } else {
                // NULL_BLOCK slot (<= 0): zero output row, state untouched —
                // same contract as fused_recurrent.py:293-297.
                let zero: Tile<f32, { [1, 1, K] }> = constant(0.0f32, shape![1, 1, K]);
                let zero: Tile<bf16, { [1, 1, K] }> = convert_tile(zero);
                out.store(zero);
            }
        }
    }
    use qwen_gdn_kernels::gdn_decode_fused_k1;

    // =====================================================================
    // torch-tensor introspection (duck-typed; no torch bindings).
    // =====================================================================
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
            device: dev.getattr("index")?.extract::<Option<usize>>()?.unwrap_or(0),
        })
    }

    fn need(name: &str, i: &TInfo, dtype: &str, shape: &[usize]) -> PyResult<()> {
        if i.dtype != dtype {
            return Err(PyValueError::new_err(format!("{name}: dtype {} != {dtype}", i.dtype)));
        }
        if i.shape != shape {
            return Err(PyValueError::new_err(format!(
                "{name}: shape {:?} != {shape:?}",
                i.shape
            )));
        }
        if i.stride.last().copied().unwrap_or(1) != 1 {
            return Err(PyValueError::new_err(format!("{name}: last dim must be contiguous")));
        }
        Ok(())
    }

    fn contiguous(name: &str, i: &TInfo) -> PyResult<()> {
        let mut want = 1usize;
        for (d, s) in i.shape.iter().zip(&i.stride).rev() {
            if *d > 1 && *s != want {
                return Err(PyValueError::new_err(format!("{name} must be contiguous")));
            }
            want *= *d;
        }
        Ok(())
    }

    // Primary-context device handles, one per ordinal: Device::new retains
    // the PRIMARY context (the one torch uses) — never a second context.
    static DEVICES: Mutex<Vec<(usize, Arc<Device>)>> = Mutex::new(Vec::new());

    fn device(ordinal: usize) -> Result<Arc<Device>, String> {
        let mut g = DEVICES.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, d)) = g.iter().find(|(o, _)| *o == ordinal) {
            return Ok(d.clone());
        }
        let d = Device::new(ordinal).map_err(|e| format!("Device::new({ordinal}): {e:?}"))?;
        g.push((ordinal, d.clone()));
        Ok(d)
    }

    fn i32s(v: &[usize]) -> Vec<i32> {
        v.iter().map(|x| *x as i32).collect()
    }

    /// Fused GDN decode step (K-GDN1). All args torch CUDA tensors except the
    /// scalars. `state` [S, HV, V, K] fp32 is updated in place; `out`
    /// [T, HV, V] bf16 (contiguous) is fully written (zero rows for slot<=0).
    /// `mixed_qkv` [T, 2HK+HV*V] bf16 and `z` [T, HV, V] bf16 may have a row
    /// stride (views into in_proj_qkvz output); inner dims must be dense.
    /// `stream_ptr` = torch.cuda.current_stream().cuda_stream.
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
        if act != ACT_SILU && act != ACT_SIGMOID {
            return Err(PyValueError::new_err(format!("unknown activation code {act}")));
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
            return Err(PyValueError::new_err("z: head dim must be dense (stride[1] == V)"));
        }
        need("ba", &bb, "torch.bfloat16", &[d.t, 2 * d.hv])?;
        need("a_log", &al, "torch.float32", &[d.hv])?;
        need("dt_bias", &dt, "torch.float32", &[d.hv])?;
        need("norm_w", &nw, "torch.float32", &[d.v])?;
        need("state", &st, "torch.float32", &[d.slots, d.hv, d.v, d.k])?;
        contiguous("state", &st)?;
        need("state_idx", &si, "torch.int32", &[d.t])?;
        need("out", &oo, "torch.bfloat16", &[d.t, d.hv, d.v])?;
        contiguous("out", &oo)?;
        let dev = mq.device;
        for (n, i) in [("z", &zz), ("ba", &bb), ("a_log", &al), ("dt_bias", &dt),
                       ("norm_w", &nw), ("state", &st), ("state_idx", &si), ("out", &oo)] {
            if i.device != dev {
                return Err(PyValueError::new_err(format!("{n} on cuda:{} != cuda:{dev}", i.device)));
            }
        }
        if stream_ptr == 0 {
            return Err(PyValueError::new_err("stream_ptr must be torch's current CUDA stream"));
        }
        if d.t == 0 {
            return Ok(());
        }
        if d.t > i32::MAX as usize || d.slots > i32::MAX as usize {
            return Err(PyValueError::new_err("T / slots exceed i32 range"));
        }

        py.detach(move || {
            crate::guard_py("gdn_decode_fused_cuda", move || {
                launch(d, &mq, &zz, &bb, &al, &dt, &nw, &st, &si, &oo, scale, norm_eps, act, stream_ptr)
                    .map_err(|e| PyRuntimeError::new_err(format!("K-GDN1 launch failed: {e}")))
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        d: GdnDims,
        mq: &TInfo,
        zz: &TInfo,
        bb: &TInfo,
        al: &TInfo,
        dt: &TInfo,
        nw: &TInfo,
        st: &TInfo,
        si: &TInfo,
        oo: &TInfo,
        scale: f32,
        norm_eps: f32,
        act: i32,
        stream_ptr: usize,
    ) -> Result<(), String> {
        let device = device(mq.device)?;
        // SAFETY: stream_ptr is torch's live current stream on this device;
        // borrowed (never destroyed by us).
        let stream: Arc<Stream> =
            unsafe { Stream::borrow_raw(stream_ptr as *mut std::ffi::c_void, &device) };
        let ord = mq.device;
        // SAFETY (borrow_raw_parts): every pointer/shape/stride triple was
        // validated above against the tensor it came from; torch keeps the
        // memory alive past this launch (caller holds the tensors, all work
        // is stream-ordered on `stream`).
        let t = |i: &TInfo| (i.ptr, i32s(&i.shape), i32s(&i.stride));
        let (p, s, r) = t(mq);
        let mq_t = unsafe { Tensor::<bf16>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(zz);
        let z_t = unsafe { Tensor::<bf16>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(bb);
        let ba_t = unsafe { Tensor::<bf16>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(al);
        let al_t = unsafe { Tensor::<f32>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(dt);
        let dt_t = unsafe { Tensor::<f32>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(nw);
        let nw_t = unsafe { Tensor::<f32>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(st);
        let st_t = unsafe { Tensor::<f32>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(si);
        let si_t = unsafe { Tensor::<i32>::borrow_raw_parts(p, ord, s, r) };
        let (p, s, r) = t(oo);
        let out_t = unsafe { Tensor::<bf16>::borrow_raw_parts(p, ord, s, r) };
        let out_part = out_t.partition([1, 1, d.v]);

        let generics = vec![
            d.h.to_string(),
            d.hv.to_string(),
            d.k.to_string(),
            act.to_string(),
        ];
        // SAFETY (unsafe entry): the kernel writes `state` through
        // partition_full_mut at [state_idx[t], hv]; distinct decode tokens
        // own distinct state slots (vLLM mamba slot allocator), slot <= 0 is
        // never written, and the store is bounds-checked against S.
        let op = unsafe {
            gdn_decode_fused_k1(
                out_part, &mq_t, &z_t, &ba_t, &al_t, &dt_t, &nw_t, &st_t, &si_t,
                scale, norm_eps, 1.0f32 / d.v as f32,
            )
        }
        .generics(generics);
        // SAFETY (async_on): outputs are torch-owned and only read by later
        // work on the same stream; no host access before a torch sync.
        unsafe { op.async_on(&stream) }.map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    /// Point cutile's JIT at a prebuilt cubin store (bundle-shipped) so a pod
    /// never runs `tileiras`. Returns the store root.
    #[pyfunction]
    pub fn qwen_gdn_enable_jit_store(dir: String) -> PyResult<String> {
        let store = cutile::jit_cache::FileSystemJitStore::new(&dir)
            .map_err(|e| PyRuntimeError::new_err(format!("jit store {dir}: {e}")))?;
        let root = store.root().display().to_string();
        cutile::jit_cache::enable(Arc::new(store));
        Ok(root)
    }

    /// (backend_compiles, disk_hits) since process start — the startup
    /// assertion requires backend_compiles == 0 on pods (no tileiras there).
    #[pyfunction]
    pub fn qwen_gdn_jit_stats() -> (u64, u64) {
        (
            cutile::jit_cache::jit_backend_compile_count(),
            cutile::jit_cache::jit_disk_hit_count(),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::qwen_gdn_kernels;
        use cutile::compile_api::KernelCompiler;

        /// Kernel -> Tile IR -> bytecode for sm_120 at the qwen3.8-27b shape
        /// (H=16, HV=48, K=V=128, silu), with the pod's real strides
        /// (in_proj_qkvz row = 2*16*128 + 2*48*128 = 16384). No GPU/driver.
        #[test]
        fn compiles_to_tile_ir_for_sm120() {
            // Pin the bytecode version so serialization needs no `tileiras`
            // probe (13.2 = the cuTile floor; the CI toolkit may override).
            if std::env::var_os("CUTILE_BYTECODE_VERSION").is_none() {
                std::env::set_var("CUTILE_BYTECODE_VERSION", "13.2");
            }
            for act in ["0", "1"] {
                let artifacts = KernelCompiler::new(
                    qwen_gdn_kernels::__module_ast_self,
                    "qwen_gdn_kernels",
                    "gdn_decode_fused_k1",
                )
                .generics(vec!["16".into(), "48".into(), "128".into(), act.into()])
                .strides(&[
                    ("out", &[48 * 128, 128, 1]),
                    ("mixed_qkv", &[16384, 1]),
                    ("z", &[16384, 128, 1]),
                    ("ba", &[96, 1]),
                    ("a_log", &[1]),
                    ("dt_bias", &[1]),
                    ("norm_w", &[1]),
                    ("state", &[48 * 128 * 128, 128 * 128, 128, 1]),
                    ("state_idx", &[1]),
                ])
                .target("sm_120")
                .compile()
                .expect("K-GDN1 must lower to Tile IR");
                let ir = artifacts.ir_text();
                assert!(ir.contains("reduce"), "expected reductions in IR");
                assert!(ir.contains("store_view_tko"), "expected state/out stores in IR");
                if let Some(dir) = std::env::var_os("QWEN_GDN_DUMP_IR") {
                    let p = std::path::Path::new(&dir).join(format!("k_gdn1_act{act}.mlir"));
                    std::fs::write(p, &ir).expect("dump IR");
                }
                let bc = artifacts.bytecode().expect("bytecode");
                assert_eq!(&bc[..8], &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]);
            }
        }
    }
}
