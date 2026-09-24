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
    use cutile::compile_api::KernelCompiler;
    use cutile::cuda_core::{Device, Stream};
    use cutile::cutile_compiler::cuda_tile_runtime_utils::{
        get_gpu_name, run_tileiras, serialize_tile_ir_bytecode, tileiras_fingerprint,
        TileirasOptions,
    };
    use cutile::cutile_compiler::jit_cache::l2_key;
    use cutile::cutile_compiler::specialization::{
        compute_spec, max_pow2_divisor, SpecializationBits,
    };
    use cutile::half::bf16;
    use cutile::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use pyo3::types::PyBytes;
    use sha2::{Digest, Sha256};
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
                let s0: Tile<f32, { [1, 1, K, K] }> =
                    unsafe { s_view.load([slot, hv, 0i32, 0i32]) };
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
                    var.reshape(shape![1, 1]) * scalar_to_tile(inv_v).reshape(shape![1, 1]);
                let rstd: Tile<f32, { [1, 1] }> = rsqrt(
                    var + scalar_to_tile(norm_eps).reshape(shape![1, 1]),
                    ftz::Disabled,
                );
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
            device: dev
                .getattr("index")?
                .extract::<Option<usize>>()?
                .unwrap_or(0),
        })
    }

    fn need(name: &str, i: &TInfo, dtype: &str, shape: &[usize]) -> PyResult<()> {
        if i.dtype != dtype {
            return Err(PyValueError::new_err(format!(
                "{name}: dtype {} != {dtype}",
                i.dtype
            )));
        }
        if i.shape != shape {
            return Err(PyValueError::new_err(format!(
                "{name}: shape {:?} != {shape:?}",
                i.shape
            )));
        }
        if i.stride.last().copied().unwrap_or(1) != 1 {
            return Err(PyValueError::new_err(format!(
                "{name}: last dim must be contiguous"
            )));
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
        need("a_log", &al, "torch.float32", &[d.hv])?;
        need("dt_bias", &dt, "torch.float32", &[d.hv])?;
        need("norm_w", &nw, "torch.float32", &[d.v])?;
        need("state", &st, "torch.float32", &[d.slots, d.hv, d.v, d.k])?;
        // vLLM views the mamba state through as_strided with PADDED page
        // strides (v1/worker/mamba_utils.py:354-357): only the per-slot block
        // must be dense; stride[0] is the page stride.
        if st.stride[1..] != [d.v * d.k, d.k, 1] || st.stride[0] < d.hv * d.v * d.k {
            return Err(PyValueError::new_err(format!(
                "state strides {:?}: per-slot [HV, V, K] block must be dense",
                st.stride
            )));
        }
        need("state_idx", &si, "torch.int32", &[d.t])?;
        need("out", &oo, "torch.bfloat16", &[d.t, d.hv, d.v])?;
        contiguous("out", &oo)?;
        let dev = mq.device;
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
            if i.device != dev {
                return Err(PyValueError::new_err(format!(
                    "{n} on cuda:{} != cuda:{dev}",
                    i.device
                )));
            }
        }
        // stream_ptr == 0 is torch's legacy default stream (the NULL CUstream),
        // valid for cuLaunchKernel and exactly what torch's default stream is;
        // vLLM runs init + the warmup oracle there (2026-09-24 pods rejected it).
        // Graph capture always uses a non-default stream, so capture is unaffected.
        if d.t == 0 {
            return Ok(());
        }
        if d.t > i32::MAX as usize || d.slots > i32::MAX as usize {
            return Err(PyValueError::new_err("T / slots exceed i32 range"));
        }

        py.detach(move || {
            crate::guard_py("gdn_decode_fused_cuda", move || {
                launch(
                    d, &mq, &zz, &bb, &al, &dt, &nw, &st, &si, &oo, scale, norm_eps, act,
                    stream_ptr,
                )
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
        let metas = [
            Meta::of("out", oo),
            Meta::of("mixed_qkv", mq),
            Meta::of("z", zz),
            Meta::of("ba", bb),
            Meta::of("a_log", al),
            Meta::of("dt_bias", dt),
            Meta::of("norm_w", nw),
            Meta::of("state", st),
            Meta::of("state_idx", si),
        ];
        require_prebuilt(d, act, &metas)?;
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
                out_part,
                &mq_t,
                &z_t,
                &ba_t,
                &al_t,
                &dt_t,
                &nw_t,
                &st_t,
                &si_t,
                scale,
                norm_eps,
                1.0f32 / d.v as f32,
            )
        }
        .generics(generics);
        // SAFETY (async_on): outputs are torch-owned and only read by later
        // work on the same stream; no host access before a torch sync.
        // A cutile fallthrough to tileiras (absent on pods) becomes OUR
        // classified error (store miss vs driver-rejected cubin).
        let before = crate::cubin_store::jit_snapshot();
        unsafe { op.async_on(&stream) }.map_err(|e| {
            crate::cubin_store::explain_launch_failure("K-GDN1", before, format!("{e:?}"))
        })?;
        Ok(())
    }

    // =====================================================================
    // Prebuilt-cubin contract (no JIT on pods).
    //
    // cutile specializes a kernel on power-of-two divisibility (clamped to
    // 16) of every tensor's shape, strides and base pointer
    // (cutile-compiler specialization.rs:152) plus the stride-is-1 hints the
    // generated launcher passes (cutile-macro kernel_launcher_generator.rs:
    // 1782-1806). For K-GDN1 at a fixed model shape only two of those vary in
    // serving: T (batch rows) and S (state slots). A variant is therefore
    // (act, div(T), div(S)); CI prebuilds every variant for sm_120 with the
    // offline `tileiras`, the pod rebuilds the (pure-Rust, GPU-free) Tile IR
    // bytecode for each, checks its sha256 against the manifest, and serves
    // the CI cubin from an in-memory JitStore under the pod's own L2 key.
    // A launch whose specialization is not installed is refused BEFORE the
    // cutile launcher could fall through to `tileiras`.
    // =====================================================================
    pub const DIVS: [i32; 5] = [1, 2, 4, 8, 16];

    #[derive(Clone)]
    pub struct Meta {
        name: &'static str,
        ptr: u64,
        shape: Vec<i32>,
        strides: Vec<i32>,
    }

    impl Meta {
        fn of(name: &'static str, i: &TInfo) -> Self {
            Meta {
                name,
                ptr: i.ptr,
                shape: i32s(&i.shape),
                strides: i32s(&i.stride),
            }
        }
        fn new(name: &'static str, shape: &[usize], strides: &[usize]) -> Self {
            // ptr 0 = maximally aligned (divisor 16), what torch allocations
            // and the vLLM views K-GDN1 receives provide.
            Meta {
                name,
                ptr: 0,
                shape: i32s(shape),
                strides: i32s(strides),
            }
        }
    }

    type Specs = (Vec<(String, Vec<i32>)>, Vec<(String, SpecializationBits)>);

    /// Exactly what the generated launcher passes: stride hints (1 / -1) and
    /// `compute_spec` of each tensor, in parameter order.
    fn specs_of(metas: &[Meta]) -> Specs {
        let strides = metas
            .iter()
            .map(|m| {
                let h = m
                    .strides
                    .iter()
                    .map(|s| if *s == 1 { 1 } else { -1 })
                    .collect();
                (m.name.to_string(), h)
            })
            .collect();
        let specs = metas
            .iter()
            .map(|m| {
                (
                    m.name.to_string(),
                    compute_spec(m.ptr, &m.shape, &m.strides, 0),
                )
            })
            .collect();
        (strides, specs)
    }

    /// The serving layout at representative T / S (their divisibility is all
    /// that reaches the kernel key).
    fn variant_metas(d: GdnDims) -> Vec<Meta> {
        let GdnDims {
            t, h, hv, k, slots, ..
        } = d;
        let row = 2 * h * k + 2 * hv * k; // in_proj_qkvz width: [q|k|v|z]
        vec![
            Meta::new("out", &[t, hv, k], &[hv * k, k, 1]),
            Meta::new("mixed_qkv", &[t, 2 * h * k + hv * k], &[row, 1]),
            Meta::new("z", &[t, hv, k], &[row, k, 1]),
            Meta::new("ba", &[t, 2 * hv], &[2 * hv, 1]),
            Meta::new("a_log", &[hv], &[1]),
            Meta::new("dt_bias", &[hv], &[1]),
            Meta::new("norm_w", &[k], &[1]),
            Meta::new("state", &[slots, hv, k, k], &[hv * k * k, k * k, k, 1]),
            Meta::new("state_idx", &[t], &[1]),
        ]
    }

    fn div(x: usize) -> i32 {
        max_pow2_divisor(x as i32)
    }

    fn variant_dims(
        h: usize,
        hv: usize,
        k: usize,
        t_div: i32,
        s_div: i32,
    ) -> Result<GdnDims, String> {
        if !DIVS.contains(&t_div) || !DIVS.contains(&s_div) {
            return Err(format!("variant divisors must be in {DIVS:?}"));
        }
        let d = GdnDims {
            t: t_div as usize,
            h,
            hv,
            k,
            v: k,
            slots: s_div as usize,
        };
        d.validate()?;
        Ok(d)
    }

    /// Tile IR bytecode, its version, and this process's L2 JIT-cache key
    /// (the one the cutile launcher will look up) for one variant. GPU-free.
    fn variant_bytecode(
        d: GdnDims,
        act: i32,
        gpu_name: &str,
    ) -> Result<(Vec<u8>, String, String), String> {
        let (strides, specs) = specs_of(&variant_metas(d));
        let sref: Vec<(&str, &[i32])> = strides
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let pref: Vec<(&str, SpecializationBits)> =
            specs.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
        let art = KernelCompiler::new(
            qwen_gdn_kernels::__module_ast_self,
            "qwen_gdn_kernels",
            "gdn_decode_fused_k1",
        )
        .generics(vec![
            d.h.to_string(),
            d.hv.to_string(),
            d.k.to_string(),
            act.to_string(),
        ])
        .strides(&sref)
        .spec_args(&pref)
        .target(gpu_name)
        .compile()
        .map_err(|e| format!("K-GDN1 Tile IR compile: {e}"))?;
        let (bc, ver) = serialize_tile_ir_bytecode(art.module())
            .map_err(|e| format!("K-GDN1 bytecode: {e}"))?;
        let key = l2_key(
            &bc,
            ver,
            gpu_name,
            &TileirasOptions::default(),
            tileiras_fingerprint(),
        );
        Ok((bc, format!("{}.{}", ver.major, ver.minor), key))
    }

    fn sha256_hex(b: &[u8]) -> String {
        Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    }

    /// Installed variants: (h, hv, k, act, t_div, s_div).
    static INSTALLED: Mutex<Vec<(usize, usize, usize, i32, i32, i32)>> = Mutex::new(Vec::new());

    fn require_prebuilt(d: GdnDims, act: i32, metas: &[Meta]) -> Result<(), String> {
        let (td, sd) = (div(d.t), div(d.slots));
        let want = (d.h, d.hv, d.k, act, td, sd);
        if !INSTALLED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains(&want)
        {
            return Err(format!(
                "no prebuilt K-GDN1 cubin for (H={}, HV={}, K={}, act={act}, div(T)={td}, \
                 div(S)={sd}); pods never JIT — the bundle manifest must carry it",
                d.h, d.hv, d.k
            ));
        }
        let rep = variant_metas(GdnDims {
            t: td as usize,
            slots: sd as usize,
            ..d
        });
        let (got, want) = (specs_of(metas), specs_of(&rep));
        if format!("{got:?}") != format!("{want:?}") {
            return Err(format!(
                "K-GDN1 launch specialization differs from the prebuilt variant \
                 (strides/alignment): got {got:?} want {want:?}"
            ));
        }
        Ok(())
    }

    /// CI + debugging: (bytecode, bytecode_version, sha256_hex) of a variant.
    #[pyfunction]
    pub fn qwen_gdn_variant_bytecode<'py>(
        py: Python<'py>,
        h: usize,
        hv: usize,
        k: usize,
        act: i32,
        t_div: i32,
        s_div: i32,
        gpu_name: &str,
    ) -> PyResult<(Bound<'py, PyBytes>, String, String)> {
        let d = variant_dims(h, hv, k, t_div, s_div).map_err(PyValueError::new_err)?;
        let (bc, ver, _) = variant_bytecode(d, act, gpu_name).map_err(PyRuntimeError::new_err)?;
        let sha = sha256_hex(&bc);
        Ok((PyBytes::new(py, &bc), ver, sha))
    }

    /// CI only: bytecode -> cubin with the offline `tileiras` (no GPU).
    #[pyfunction]
    pub fn qwen_gdn_compile_cubin<'py>(
        py: Python<'py>,
        bytecode: &[u8],
        gpu_name: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cubin = run_tileiras(bytecode, gpu_name, &TileirasOptions::default())
            .map_err(|e| PyRuntimeError::new_err(format!("tileiras: {e}")))?;
        Ok(PyBytes::new(py, &cubin))
    }

    /// `sm_XY` name cutile keys cubins by for this device.
    #[pyfunction]
    pub fn qwen_gdn_gpu_name(ordinal: usize) -> String {
        get_gpu_name(ordinal)
    }

    /// Pod startup: rebuild the variant's bytecode, require its sha256 to
    /// equal the manifest's, and serve `cubin` for it from the in-memory
    /// JitStore under this process's own L2 key. Returns the key.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    pub fn qwen_gdn_install_cubin(
        h: usize,
        hv: usize,
        k: usize,
        act: i32,
        t_div: i32,
        s_div: i32,
        gpu_name: &str,
        bc_sha256: &str,
        cubin: &[u8],
    ) -> PyResult<String> {
        let d = variant_dims(h, hv, k, t_div, s_div).map_err(PyValueError::new_err)?;
        let (bc, _, key) = variant_bytecode(d, act, gpu_name).map_err(PyRuntimeError::new_err)?;
        let got = sha256_hex(&bc);
        if got != bc_sha256 {
            return Err(PyRuntimeError::new_err(format!(
                "K-GDN1 manifest mismatch: pod bytecode sha256 {got} != bundle {bc_sha256} \
                 (act={act}, div(T)={t_div}, div(S)={s_div}; check CUTILE_BYTECODE_VERSION)"
            )));
        }
        // Shared process-global store (K2-NVFP4 installs into it too).
        crate::cubin_store::install(&key, &bc, gpu_name, cubin).map_err(PyRuntimeError::new_err)?;
        let mut inst = INSTALLED.lock().unwrap_or_else(|p| p.into_inner());
        let v = (h, hv, k, act, t_div, s_div);
        if !inst.contains(&v) {
            inst.push(v);
        }
        Ok(key)
    }

    /// (backend_compiles, disk_hits) since process start. The K-GDN1 startup
    /// assertion requires backend_compiles == 0 (no JIT on pods).
    #[pyfunction]
    pub fn qwen_gdn_jit_stats() -> (u64, u64) {
        (
            cutile::jit_cache::jit_backend_compile_count(),
            cutile::jit_cache::jit_disk_hit_count(),
        )
    }

    // =====================================================================
    // tileiras construct probes (CI diagnostics). tileiras 13.4.92 rejects
    // the K-GDN1 program with only "failed to compile Tile IR program"; these
    // minimal kernels each isolate ONE construct K-GDN1 uses, so a CI run
    // that compiles all of them pinpoints the rejected one
    // (scripts/tileiras_diag.py prints tileiras' raw output per probe).
    // =====================================================================
    #[cutile::module]
    pub mod qgdn_probes {
        use cutile::core::*;

        /// Big register tile: [R, 128] f32 load + row reduce (K-GDN1: R=128).
        #[cutile::entry()]
        pub fn p_tile_reduce<const R: i32>(
            out: &mut Tensor<f32, { [R] }>,
            x: &Tensor<f32, { [-1, -1] }>,
        ) {
            let part: Partition<f32, { [R, 128] }> = x.partition(shape![R, 128]);
            let t: Tile<f32, { [R, 128] }> = part.load([0i32, 0i32]);
            let r: Tile<f32, { [R] }> = reduce_sum(t * t, 1i32);
            out.store(r);
        }

        /// Dynamic-slot in-place update through partition_full_mut (+ asserts).
        #[cutile::entry()]
        pub unsafe fn p_full_mut_dyn(
            out: &mut Tensor<f32, { [1] }>,
            state: &Tensor<f32, { [-1, -1, -1] }>,
            idx: &Tensor<i32, { [-1] }>,
        ) {
            let ip: Partition<i32, { [1] }> = idx.partition(shape![1]);
            let it: Tile<i32, { [1] }> = ip.load([0i32]);
            let slot: i32 = tile_to_scalar(it.reshape(shape![]));
            let mut v: PartitionMut<f32, { [1, 32, 32] }> =
                unsafe { state.partition_full_mut(shape![1, 32, 32]) };
            check_partition_access_mut(&v, [slot, 0i32, 0i32]);
            let s: Tile<f32, { [1, 32, 32] }> = unsafe { v.load([slot, 0i32, 0i32]) };
            let s2: Tile<f32, { [1, 32, 32] }> = s * s;
            v.store(s2, [slot, 0i32, 0i32]);
            let z: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
            out.store(z);
        }

        /// Stores to the block output in BOTH branches of a data-dependent if.
        #[cutile::entry()]
        pub fn p_if_else_store(out: &mut Tensor<f32, { [1, 32] }>, idx: &Tensor<i32, { [-1] }>) {
            let ip: Partition<i32, { [1] }> = idx.partition(shape![1]);
            let it: Tile<i32, { [1] }> = ip.load([0i32]);
            let c: i32 = tile_to_scalar(it.reshape(shape![]));
            if c > 0i32 {
                let a: Tile<f32, { [1, 32] }> = constant(1.0f32, shape![1, 32]);
                out.store(a);
            } else {
                let b: Tile<f32, { [1, 32] }> = constant(0.0f32, shape![1, 32]);
                out.store(b);
            }
        }

        /// bf16 load -> f32 math -> bf16 store (ftof both ways).
        #[cutile::entry()]
        pub fn p_bf16_io(out: &mut Tensor<bf16, { [1, 128] }>, x: &Tensor<bf16, { [-1, -1] }>) {
            let part: Partition<bf16, { [1, 128] }> = x.partition(shape![1, 128]);
            let t: Tile<f32, { [1, 128] }> = convert_tile(part.load([0i32, 0i32]));
            let y: Tile<bf16, { [1, 128] }> = convert_tile(t * t);
            out.store(y);
        }

        /// [1,1] scalar-tile math: exp/log/select/negf/true_div/rsqrt.
        #[cutile::entry()]
        pub fn p_scalar_math(out: &mut Tensor<f32, { [1, 1] }>, x: &Tensor<f32, { [-1, -1] }>) {
            let part: Partition<f32, { [1, 1] }> = x.partition(shape![1, 1]);
            let a: Tile<f32, { [1, 1] }> = part.load([0i32, 0i32]);
            let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
            let th: Tile<f32, { [1, 1] }> = constant(20.0f32, shape![1, 1]);
            let sp: Tile<f32, { [1, 1] }> = select(le_tile(a, th), log(one + exp(a)), a);
            let sg: Tile<f32, { [1, 1] }> = true_div(one, one + exp(negf(sp)));
            let r: Tile<f32, { [1, 1] }> = rsqrt(sg + one, ftz::Disabled);
            out.store(r);
        }

        /// 4-D [1,1,R,R] partition load + reshape + row/col broadcasts + reduce
        /// (the K-GDN1 state-tile data path without the in-place store).
        #[cutile::entry()]
        pub fn p_4d_outer<const R: i32>(
            out: &mut Tensor<f32, { [R] }>,
            st: &Tensor<f32, { [-1, -1, -1, -1] }>,
            v: &Tensor<f32, { [-1, -1] }>,
        ) {
            let sp: Partition<f32, { [1, 1, R, R] }> = st.partition(shape![1, 1, R, R]);
            let s0: Tile<f32, { [1, 1, R, R] }> = sp.load([0i32, 0i32, 0i32, 0i32]);
            let s: Tile<f32, { [R, R] }> = s0.reshape(shape![R, R]);
            let vp: Partition<f32, { [1, R] }> = v.partition(shape![1, R]);
            let k: Tile<f32, { [1, R] }> = vp.load([0i32, 0i32]);
            let kb: Tile<f32, { [R, R] }> = k.broadcast(shape![R, R]);
            let col: Tile<f32, { [R] }> = reduce_sum(s * kb, 1i32);
            let cb: Tile<f32, { [R, R] }> = col.reshape(shape![R, 1]).broadcast(shape![R, R]);
            let o: Tile<f32, { [R] }> = reduce_sum((s + cb * kb) * kb, 1i32);
            out.store(o);
        }

        /// Runtime scalar broadcast to [1,1] (reshape + same-shape broadcast).
        #[cutile::entry()]
        pub fn p_scalar_broadcast(out: &mut Tensor<f32, { [1, 1] }>, s: f32) {
            let b: Tile<f32, { [1, 1] }> = broadcast_scalar(s, shape![1, 1]);
            out.store(b);
        }
    }

    const PROBE_ENTRIES: &[&str] = &[
        "p_full_mut_dyn",
        "p_if_else_store",
        "p_bf16_io",
        "p_scalar_math",
        "p_scalar_broadcast",
    ];

    /// CI diagnostics: [(probe_name, tile_ir_bytecode)] — construct probes,
    /// tile-size sweep, and K-GDN1 at K = 32/64/128 (the real kernel).
    #[pyfunction]
    pub fn qwen_gdn_probe_bytecodes<'py>(
        py: Python<'py>,
        gpu_name: &str,
    ) -> PyResult<Vec<(String, Bound<'py, PyBytes>)>> {
        let mut out = Vec::new();
        let e =
            |x: cutile::cutile_compiler::error::JITError| PyRuntimeError::new_err(x.to_string());
        let mut push = |name: String, art: cutile::compile_api::CompileArtifacts| -> PyResult<()> {
            let (bc, _) = serialize_tile_ir_bytecode(art.module()).map_err(e)?;
            out.push((name, PyBytes::new(py, &bc)));
            Ok(())
        };
        for r in [8, 32, 128] {
            let art = KernelCompiler::new(
                qgdn_probes::__module_ast_self,
                "qgdn_probes",
                "p_tile_reduce",
            )
            .generics(vec![r.to_string()])
            .strides(&[("out", &[1]), ("x", &[-1, 1])])
            .target(gpu_name)
            .compile()
            .map_err(e)?;
            push(format!("p_tile_reduce_R{r}x128"), art)?;
        }
        for r in [32, 128] {
            let art =
                KernelCompiler::new(qgdn_probes::__module_ast_self, "qgdn_probes", "p_4d_outer")
                    .generics(vec![r.to_string()])
                    .strides(&[("out", &[1]), ("st", &[-1, -1, -1, 1]), ("v", &[-1, 1])])
                    .target(gpu_name)
                    .compile()
                    .map_err(e)?;
            push(format!("p_4d_outer_R{r}"), art)?;
        }
        for name in PROBE_ENTRIES {
            let strides: Vec<(&str, &[i32])> = match *name {
                "p_full_mut_dyn" => vec![("out", &[1]), ("state", &[-1, -1, 1]), ("idx", &[1])],
                "p_if_else_store" => vec![("out", &[-1, 1]), ("idx", &[1])],
                "p_bf16_io" | "p_scalar_math" => vec![("out", &[-1, 1]), ("x", &[-1, 1])],
                _ => vec![("out", &[-1, 1])],
            };
            let art = KernelCompiler::new(qgdn_probes::__module_ast_self, "qgdn_probes", name)
                .strides(&strides)
                .target(gpu_name)
                .compile()
                .map_err(e)?;
            push(name.to_string(), art)?;
        }
        drop(push);
        for k in [32usize, 64, 128] {
            let d = variant_dims(16, 48, k, 1, 16).map_err(PyValueError::new_err)?;
            let (bc, _, _) =
                variant_bytecode(d, ACT_SILU, gpu_name).map_err(PyRuntimeError::new_err)?;
            out.push((format!("k_gdn1_K{k}"), PyBytes::new(py, &bc)));
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn pin_bc() {
            if std::env::var_os("CUTILE_BYTECODE_VERSION").is_none() {
                std::env::set_var("CUTILE_BYTECODE_VERSION", "13.3");
            }
        }

        /// Every serving variant at the qwen3.8-27b shape (H=16, HV=48,
        /// K=V=128) lowers to Tile IR bytecode for sm_120. No GPU/driver.
        #[test]
        fn all_variants_lower_to_tile_ir_for_sm120() {
            pin_bc();
            let mut shas = std::collections::HashSet::new();
            for act in [ACT_SILU, ACT_SIGMOID] {
                for t in DIVS {
                    for s in [1, 16] {
                        let d = variant_dims(16, 48, 128, t, s).unwrap();
                        let (bc, ver, key) = variant_bytecode(d, act, "sm_120").unwrap();
                        assert_eq!(key.len(), 64);
                        assert_eq!(&bc[..8], &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]);
                        assert_eq!(ver, std::env::var("CUTILE_BYTECODE_VERSION").unwrap());
                        shas.insert(sha256_hex(&bc));
                        if let Some(dir) = std::env::var_os("QWEN_GDN_DUMP_IR") {
                            let p = std::path::Path::new(&dir)
                                .join(format!("k_gdn1_act{act}_t{t}_s{s}.bc"));
                            std::fs::write(p, &bc).unwrap();
                        }
                    }
                }
            }
            // divisibility really reaches the key (else one variant would do)
            assert!(shas.len() > 2, "variants collapsed: {}", shas.len());
            // and compilation is deterministic
            let d = variant_dims(16, 48, 128, 4, 8).unwrap();
            assert_eq!(
                variant_bytecode(d, 0, "sm_120").unwrap().0,
                variant_bytecode(d, 0, "sm_120").unwrap().0
            );
        }

        #[test]
        fn tileiras_probes_lower_to_bytecode() {
            pin_bc();
            pyo3::Python::initialize();
            pyo3::Python::attach(|py| {
                let v = qwen_gdn_probe_bytecodes(py, "sm_120").unwrap();
                assert_eq!(v.len(), 3 + 2 + PROBE_ENTRIES.len() + 3);
            });
        }

        #[test]
        fn launch_spec_check_matches_variants_and_refuses_the_rest() {
            let d = GdnDims {
                t: 12,
                h: 16,
                hv: 48,
                k: 128,
                v: 128,
                slots: 40,
            };
            let mut metas = variant_metas(d);
            // not installed yet -> refused
            assert!(require_prebuilt(d, 0, &metas)
                .unwrap_err()
                .contains("no prebuilt"));
            INSTALLED.lock().unwrap().push((16, 48, 128, 0, 4, 8));
            // real sizes 12 / 40 have div 4 / 8 -> same specialization
            require_prebuilt(d, 0, &metas).unwrap();
            // a padded vLLM page stride (div 16) keeps the variant
            metas[7].strides[0] = 48 * 128 * 128 + 16 * 1024;
            require_prebuilt(d, 0, &metas).unwrap();
            // a misaligned pointer is a different specialization -> refused
            metas[2].ptr = 2;
            assert!(require_prebuilt(d, 0, &metas)
                .unwrap_err()
                .contains("differs"));
        }
    }
}
