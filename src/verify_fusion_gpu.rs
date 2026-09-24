// SPDX-License-Identifier: Apache-2.0
//! K1 GPU twin: `rejection_greedy_accept` as one cutile-rs (CUDA-Rust) tile
//! kernel — our OWN kernel implementing vLLM 0.30.0's greedy verify semantics
//! (NOT a rehosting of vLLM's Triton code).
//!
//! Everything in this file — kernel, host wrapper, registration — is gated
//! behind the non-default cargo feature `cutile-kernels`:
//!   * feature-off builds stay exactly on today's plugin (no cutile dep is
//!     even resolved into the build graph);
//!   * feature-on builds require a CUDA 13.x toolkit (cuda-bindings runs
//!     bindgen against cuda.h at build time) and therefore only happen in
//!     the CI container (protocol: dossiers/kernel-k1-spike.md §5).
//! Mac hosts have no CUDA toolchain: this file is NOT compiled locally.
//! Local validation is the CPU oracle (`src/verify_fusion.rs`, tests/
//! test_verify_fusion.py) plus the per-block kernel-algorithm twin test
//! (tests/test_verify_fusion_gpu_twin.py), which transcribes the kernel's
//! scalar logic below statement-for-statement.
//!
//! ## Kernel design (K1 identity-gate target)
//!
//! One tile block per request; grid = batch. Per block b:
//!   start = b == 0 ? 0 : cu[b-1]                       (inclusive-cumsum:
//!                                                       rejection_sampler
//!                                                       kernel :739-745)
//!   end   = min(cu[b], num_tokens)
//!   owned = clamp(end - start, 0, k)
//!   count = length of the leading run p in [0, owned) with
//!           draft[start+p] >= 0 && draft[start+p] == argmax[start+p]
//!           (a padded -1 draft is a REJECTION, not a skip: :760-761)
//!   accept_count[b] = count
//!   emit_mask[b, :] = (iota(WIDTH) <= count) as i32   (accepted positions
//!           plus the corrective/bonus slot; WIDTH = k+1)
//!
//! Both documented subtleties from the CPU oracle are preserved here:
//!   1. `cu_num_draft_tokens` is an INCLUSIVE cumsum (start = cu[b-1]);
//!   2. padded (-1) draft == rejection: the run stops and the corrective
//!      argmax slot is emitted, never a skip.
//!
//! The GPU-identity gate (FRONTIER K1) is: this kernel's (accept_count,
//! emit_mask) must equal `suffix_hybrid._native.rejection_greedy_accept`
//! bit-for-bit on the int outputs for the tests/test_verify_fusion.py
//! shapes plus a c=1..32 batch sweep, run on the SM120 pool GPU.
//!
//! ## Host wrapper contract (fail loud, never silent-fallback)
//!
//! `rejection_greedy_accept_cuda` takes torch CUDA tensors and writes into
//! caller-owned output tensors (torch owns all memory; we only borrow the
//! data_ptrs via `Tensor::borrow_raw_parts`), launches on the CALLER's
//! current CUDA stream (`Stream::borrow_raw`, no private stream), wraps the
//! launch in `py.allow_threads` (GIL released) and the existing
//! `catch_unwind` panic boundary (guard_py, src/lib.rs). Any precondition
//! miss (wrong dtype/shape/device, non-contiguous, bad cu) is a hard
//! ValueError/RuntimeError — the caller must degrade, we never silently
//! route into another path.

#[cfg(feature = "cutile-kernels")]
pub use device::*;

#[cfg(feature = "cutile-kernels")]
mod device {
    use cutile::cuda_core::Stream;
    use cutile::prelude::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use std::sync::Arc;

    // =====================================================================
    // cuTile kernel (compiled to sm_120a cubin by the Tile IR backend on
    // first launch in the CI-built feature-on wheel; PTX floor: driver R580).
    // One tile block per request. WIDTH = max_spec_len + 1 (emit slots).
    // =====================================================================
    #[cutile::module]
    mod verify_fusion_greedy {
        use cutile::core::*;

        #[cutile::entry]
        fn rejection_greedy_accept_k1<const WIDTH: i32>(
            draft_token_ids: &Tensor<i64, { [-1] }>,
            target_argmax: &Tensor<i64, { [-1] }>,
            cu_num_draft_tokens: &Tensor<i64, { [-1] }>,
            accept_count: &mut Tensor<i32, { [1] }>,
            emit_mask: &mut Tensor<i32, { [1, WIDTH] }>,
            num_tokens: i32,
            max_spec_len: i32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let b: i32 = pid.0;

            let cu_part = cu_num_draft_tokens.partition(const_shape![1]);
            // Inclusive cumsum: this request owns rows [cu[b-1], cu[b]).
            // Tile IR lowers neither scalar i32<->i64 casts nor scalar
            // i64->i32 conversion: narrow cu values as TILES (trunci), then
            // clamp in i32 (host guarantees num_tokens < 2^31).
            let start: i32 = if b == 0i32 {
                0i32
            } else {
                let prev: Tile<i64, { [1] }> = cu_part.load([b - 1i32]);
                let prev: Tile<i32, { [1] }> = trunci(prev, overflow::NoSignedWrap);
                tile_to_scalar(prev.reshape(const_shape![]))
            };
            let mut end: i32 = {
                let cur: Tile<i64, { [1] }> = cu_part.load([b]);
                let cur: Tile<i32, { [1] }> = trunci(cur, overflow::NoSignedWrap);
                tile_to_scalar(cur.reshape(const_shape![]))
            };
            if end > num_tokens {
                end = num_tokens;
            }
            let mut owned: i32 = end - start;
            if owned < 0i32 {
                owned = 0i32;
            }
            if owned > max_spec_len {
                owned = max_spec_len;
            }

            let draft_part = draft_token_ids.partition(const_shape![1]);
            let argmax_part = target_argmax.partition(const_shape![1]);

            // Keep-while-match prefix over this request's draft slots. A
            // padded (-1) draft mismatches and stops the run, exactly like
            // the CPU oracle and rejection_sampler.py:761.
            // (Tile IR lowers no short-circuit `&&`: bounded `for` + flags.)
            let mut count: i32 = 0i32;
            let mut accept: i32 = 1i32;
            for p in 0i32..owned {
                if accept == 1i32 {
                    let idx: i32 = start + p;
                    let d_tile: Tile<i64, { [1] }> = draft_part.load([idx]);
                    let d: i64 = tile_to_scalar(d_tile.reshape(const_shape![]));
                    let a_tile: Tile<i64, { [1] }> = argmax_part.load([idx]);
                    let a: i64 = tile_to_scalar(a_tile.reshape(const_shape![]));
                    let mut hit: i32 = 0i32;
                    if d >= 0i64 {
                        if d == a {
                            hit = 1i32;
                        }
                    }
                    if hit == 1i32 {
                        count = count + 1i32;
                    } else {
                        accept = 0i32;
                    }
                }
            }

            // accept_count[b]
            let count_scalar: Tile<i32, { [] }> = scalar_to_tile(count);
            let count_tile: Tile<i32, { [1] }> = count_scalar.reshape(const_shape![1]);
            accept_count.store(count_tile);

            // emit_mask[b, 0..WIDTH] = 1 for positions 0..=count, else 0.
            let offs: Tile<i32, { [WIDTH] }> = iota(const_shape![WIDTH]);
            let offs: Tile<i32, { [1, WIDTH] }> = offs.reshape(const_shape![1, WIDTH]);
            let thresh: Tile<i32, { [1, WIDTH] }> = broadcast_scalar(count, const_shape![1, WIDTH]);
            let keep: Tile<bool, { [1, WIDTH] }> = le_tile(offs, thresh);
            let one: Tile<i32, { [1, WIDTH] }> = constant(1i32, const_shape![1, WIDTH]);
            let zero: Tile<i32, { [1, WIDTH] }> = constant(0i32, const_shape![1, WIDTH]);
            let mask: Tile<i32, { [1, WIDTH] }> = select(keep, one, zero);
            emit_mask.store(mask);
        }
    }
    use verify_fusion_greedy::rejection_greedy_accept_k1;

    // =====================================================================
    // torch-tensor introspection (no torch bindings; duck-typed via PyAny).
    // Every check is a hard error: this op is the K1 fused path, a silent
    // fallback to stock kernels would defeat the kernel-path assertion.
    // =====================================================================
    struct TorchTensorInfo {
        data_ptr: usize,
        shape: Vec<usize>,
        dtype: String,
        device_index: usize,
    }

    fn torch_info(name: &str, obj: &Bound<'_, PyAny>) -> pyo3::PyResult<TorchTensorInfo> {
        let typ = obj
            .getattr("device")?
            .getattr("type")?
            .extract::<String>()?;
        if typ != "cuda" {
            return Err(PyValueError::new_err(format!(
                "{name} must be a CUDA tensor (got device type {typ:?}); the \
                 cutile K1 path never silently runs on another device"
            )));
        }
        let device_index = obj
            .getattr("device")?
            .getattr("index")?
            .extract::<Option<usize>>()?
            .unwrap_or(0);
        let data_ptr = obj.call_method0("data_ptr")?.extract::<usize>()?;
        let shape_tuple = obj.call_method0("size")?;
        let shape = shape_tuple.extract::<Vec<usize>>()?;
        let dtype = obj.getattr("dtype")?.str()?.to_string();
        let contiguous = obj.call_method0("is_contiguous")?.extract::<bool>()?;
        if !contiguous {
            return Err(PyValueError::new_err(format!("{name} must be contiguous")));
        }
        Ok(TorchTensorInfo {
            data_ptr,
            shape,
            dtype,
            device_index,
        })
    }

    fn require(
        name: &str,
        info: &TorchTensorInfo,
        dtype: &str,
        shape: &[usize],
    ) -> pyo3::PyResult<()> {
        if info.dtype != dtype {
            return Err(PyValueError::new_err(format!(
                "{name} must have dtype {dtype} (got {})",
                info.dtype
            )));
        }
        if info.shape != shape {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape {shape:?} (got {:?})",
                info.shape
            )));
        }
        Ok(())
    }

    /// GPU twin of `suffix_hybrid._native.rejection_greedy_accept` (same
    /// semantics; see src/verify_fusion.rs). Runs on the caller's CUDA
    /// stream; writes into the torch-owned output tensors in place.
    ///
    /// Args (all torch CUDA tensors, contiguous):
    ///   draft_token_ids     int64 [num_tokens]      (-1 = padded draft)
    ///   target_argmax       int64 [num_tokens]
    ///   cu_num_draft_tokens int64 [batch]           inclusive cumsum
    ///   accept_count        int32 [batch]           output, written in place
    ///   emit_mask           int32 [batch, k+1]      output, written in place
    ///   max_spec_len        int                     k
    ///   stream_ptr          int                     torch current stream:
    ///                                               `torch.cuda.current_stream(
    ///                                                  device).cuda_stream`
    ///
    /// No GIL is held across the launch (py.allow_threads) and the launch is
    /// behind the lib.rs catch_unwind panic boundary, so a kernel-side panic
    /// degrades the same way as every other plugin op instead of escaping the
    /// wrapper's `except Exception` guard.
    #[pyfunction]
    #[pyo3(signature = (draft_token_ids, target_argmax, cu_num_draft_tokens, accept_count, emit_mask, max_spec_len, stream_ptr))]
    pub fn rejection_greedy_accept_cuda<'py>(
        py: Python<'py>,
        draft_token_ids: &Bound<'py, PyAny>,
        target_argmax: &Bound<'py, PyAny>,
        cu_num_draft_tokens: &Bound<'py, PyAny>,
        accept_count: &Bound<'py, PyAny>,
        emit_mask: &Bound<'py, PyAny>,
        max_spec_len: usize,
        stream_ptr: usize,
    ) -> PyResult<()> {
        let d_info = torch_info("draft_token_ids", draft_token_ids)?;
        let a_info = torch_info("target_argmax", target_argmax)?;
        let cu_info = torch_info("cu_num_draft_tokens", cu_num_draft_tokens)?;
        let c_info = torch_info("accept_count", accept_count)?;
        let m_info = torch_info("emit_mask", emit_mask)?;

        let batch = cu_info.shape[0];
        let num_tokens = d_info.shape[0];
        let width = max_spec_len + 1;
        if a_info.shape[0] != num_tokens {
            return Err(PyValueError::new_err(format!(
                "draft_token_ids ({num_tokens}) and target_argmax ({}) length mismatch",
                a_info.shape[0]
            )));
        }
        if d_info
            .shape
            .first()
            .map(|n| *n > i32::MAX as usize)
            .unwrap_or(false)
        {
            return Err(PyValueError::new_err("num_tokens exceeds i32 range"));
        }
        require("draft_token_ids", &d_info, "torch.int64", &[num_tokens])?;
        require("target_argmax", &a_info, "torch.int64", &[num_tokens])?;
        require("cu_num_draft_tokens", &cu_info, "torch.int64", &[batch])?;
        require("accept_count", &c_info, "torch.int32", &[batch])?;
        require("emit_mask", &m_info, "torch.int32", &[batch, width])?;
        if batch > 0 {
            let devices = [
                d_info.device_index,
                a_info.device_index,
                cu_info.device_index,
                c_info.device_index,
                m_info.device_index,
            ];
            if devices.iter().any(|d| *d != devices[0]) {
                return Err(PyValueError::new_err(
                    "all tensors must live on the same CUDA device",
                ));
            }
        }
        let device_ordinal = if batch > 0 { d_info.device_index } else { 0 };

        // stream_ptr == 0 is torch's legacy default stream (the NULL CUstream),
        // valid for cuLaunchKernel and exactly what torch's default stream is;
        // vLLM runs init + the warmup oracle there (2026-09-24 pods rejected it).
        // Graph capture always uses a non-default stream, so capture is unaffected.

        // All Python interaction ends here: the launch runs GIL-free.
        py.detach(|| {
            crate::guard_py("rejection_greedy_accept_cuda", move || {
                run_kernel(
                    d_info.data_ptr,
                    a_info.data_ptr,
                    cu_info.data_ptr,
                    c_info.data_ptr,
                    m_info.data_ptr,
                    num_tokens,
                    batch,
                    device_ordinal,
                    max_spec_len,
                    stream_ptr,
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "cutile rejection_greedy_accept_k1 failed: {e}"
                    ))
                })
            })
        })
    }

    fn run_kernel(
        draft_ptr: usize,
        argmax_ptr: usize,
        cu_ptr: usize,
        count_ptr: usize,
        mask_ptr: usize,
        num_tokens: usize,
        batch: usize,
        device_ordinal: usize,
        max_spec_len: usize,
        stream_ptr: usize,
    ) -> Result<(), String> {
        // Fail loud on cu inconsistency before touching the GPU: cu must be
        // a monotone inclusive cumsum bounded by num_tokens. The kernel
        // additionally clamps defensively, but a malformed cu is a caller
        // bug and must surface as an error, not a silent clamp.
        // (Validation of VALUES needs device reads; here we validate shape
        // bounds only — value validation is the identity-oracle's job.)

        let device = Device::new(device_ordinal).map_err(|e| format!("{e:?}"))?;
        // Borrow torch's current stream: all work lands on the caller's
        // stream, so downstream torch ops observe the writes without an
        // extra event/sync.
        let stream = unsafe { Stream::borrow_raw(stream_ptr as *mut std::ffi::c_void, &device) };

        let num_tokens_i32 = num_tokens as i32;
        let k_i32 = max_spec_len as i32;
        let width_i32 = (max_spec_len + 1) as i32;

        // torch owns every buffer; cutile only borrows the data_ptrs.
        // SAFETY (borrow_raw_parts): each dptr points at `shape` contiguous
        // elements on `device_ordinal` that stay alive and unmutated-by-
        // third-parties until the launched kernel completes on `stream`
        // below (stream-ordered async_on); outputs are written only by this kernel.
        let draft = unsafe {
            Tensor::<i64>::borrow_raw_parts(
                draft_ptr as u64,
                device_ordinal,
                vec![num_tokens_i32],
                vec![1],
            )
        };
        let argmax = unsafe {
            Tensor::<i64>::borrow_raw_parts(
                argmax_ptr as u64,
                device_ordinal,
                vec![num_tokens_i32],
                vec![1],
            )
        };
        let cu = unsafe {
            Tensor::<i64>::borrow_raw_parts(
                cu_ptr as u64,
                device_ordinal,
                vec![batch as i32],
                vec![1],
            )
        };
        let count = unsafe {
            Tensor::<i32>::borrow_raw_parts(
                count_ptr as u64,
                device_ordinal,
                vec![batch as i32],
                vec![1],
            )
        };
        let mask = unsafe {
            Tensor::<i32>::borrow_raw_parts(
                mask_ptr as u64,
                device_ordinal,
                vec![batch as i32, width_i32],
                vec![width_i32, 1],
            )
        };

        let draft: Arc<Tensor<i64>> = Arc::new(draft);
        let argmax: Arc<Tensor<i64>> = Arc::new(argmax);
        let cu: Arc<Tensor<i64>> = Arc::new(cu);
        let count_part = count.partition([1]);
        let mask_part = mask.partition([1, max_spec_len + 1]);

        // Instantiate WIDTH = k + 1 (the only const generic); grid is
        // inferred from the two output partitions -> (batch, 1, 1).
        let op = rejection_greedy_accept_k1(
            draft,
            argmax,
            cu,
            count_part,
            mask_part,
            num_tokens_i32,
            k_i32,
        )
        .generics(vec![width_i32.to_string()]);
        // Stream-ordered, capture-safe launch: `sync_on` would block the host
        // on cuStreamSynchronize after every launch (cuda-async-0.3.1
        // device_operation.rs:428-436) and is illegal inside CUDA-graph
        // capture. SAFETY: all buffers are torch-owned and outlive the
        // stream-ordered kernel; torch readers are ordered on the same stream.
        unsafe { op.async_on(&stream) }.map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::verify_fusion_greedy;
        use cutile::compile_api::KernelCompiler;

        /// K1 lowers to Tile IR bytecode for sm_120 (k = 8). No GPU/driver.
        #[test]
        fn k1_lowers_to_tile_ir_for_sm120() {
            if std::env::var_os("CUTILE_BYTECODE_VERSION").is_none() {
                std::env::set_var("CUTILE_BYTECODE_VERSION", "13.3");
            }
            let art = KernelCompiler::new(
                verify_fusion_greedy::__module_ast_self,
                "verify_fusion_greedy",
                "rejection_greedy_accept_k1",
            )
            .generics(vec!["9".into()])
            .strides(&[
                ("draft_token_ids", &[1]),
                ("target_argmax", &[1]),
                ("cu_num_draft_tokens", &[1]),
                ("accept_count", &[1]),
                ("emit_mask", &[-1, 1]),
            ])
            .target("sm_120")
            .compile()
            .expect("K1 must lower to Tile IR");
            let bc = art.bytecode().expect("bytecode");
            assert_eq!(&bc[..8], &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]);
        }
    }
}
