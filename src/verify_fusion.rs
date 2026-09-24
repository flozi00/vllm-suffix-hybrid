// SPDX-License-Identifier: Apache-2.0
//! K1 capability spike: fused spec-decode verification (greedy).
//!
//! One minimal fused op: `rejection_greedy_accept` — the batched
//! draft-token compare + accept-mask construction that vLLM 0.30.0 splits
//! across `target_logits.argmax(dim=-1)` (rejection_sampler.py:468) and the
//! per-request row loop in `rejection_greedy_sample_kernel`
//! (rejection_sampler.py:719-773). On SM120 the same source shape compiles
//! via cutile-rs (elementwise + row-reduction over the draft axis, no tensor
//! cores needed — Grout precedents `add_rms_norm_f16`,
//! `GROUT_FUSED_LM_HEAD_ARGMAX` in cuda-rust-research/cutile-rs.md); this
//! host module is the shared semantic contract + CPU oracle:
//!   - the GPU kernel (cutile-rs, non-default cargo feature
//!     `cutile-kernels`, CI-built cubin shipped in the runtime bundle) must
//!     reproduce `rejection_greedy_accept` exactly on the int outputs
//!     (identity gate below);
//!   - this CPU path serves the torch-reference test on hosts without CUDA
//!     and documents the semantics the startup kernel-path assertion checks.
//!
//! Semantics (mirror of vLLM `rejection_sample`, all-greedy, non-synthetic;
//! rejection_sampler.py:407-517, kernel :719-773):
//!   draft_token_ids      i64 [num_tokens]   flattened; -1 = padded draft
//!   target_argmax        i64 [num_tokens]   argmax over each target-model
//!                                           logits row (vocab axis)
//!   cu_num_draft_tokens  i64 [batch]        exclusive cumsum of each
//!                                           request's num_draft_tokens
//!   max_spec_len         int                k (draft length)
//!   ->
//!   accept_count i32 [batch]: per request, length of the leading
//!       draft-position run with draft == argmax (padding always mismatches).
//!   emit_mask i32 [batch, max_spec_len+1]: emit_mask[b, p] = 1 for
//!       p < accept_count (accepted draft positions) and p == accept_count
//!       (the token replacing the first rejection, or the bonus token when
//!       every draft is accepted — both consume exactly one extra slot of
//!       vLLM's `output_token_ids`), else 0.
//! Reconstructing tokens as `argmax_or_draft` at masked positions reproduces
//! vLLM's greedy `output_token_ids` (rejection_sampler.py:760-773) exactly;
//! `parse_output` drops the remaining PLACEHOLDER (-1) columns, which our
//! mask marks 0.
//!
//! Inputs are plain per-request NumPy buffers (the mixer.rs convention); no
//! CUDA types appear here so the module builds on any host.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (draft_token_ids, target_argmax, cu_num_draft_tokens, max_spec_len))]
pub fn rejection_greedy_accept<'py>(
    py: Python<'py>,
    draft_token_ids: &Bound<'py, PyAny>,
    target_argmax: &Bound<'py, PyAny>,
    cu_num_draft_tokens: &Bound<'py, PyAny>,
    max_spec_len: usize,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<i32>>)> {
    use pyo3::types::PyAny;
    let to_vec = |name: &str, arg: &Bound<'py, PyAny>| -> PyResult<Vec<i64>> {
            let owned = arg
                .extract::<PyReadonlyArray1<'_, i64>>()
                .map_err(|_| {
                    PyValueError::new_err(format!("{name} must be a contiguous i64 1D array"))
                })?;
            let arr = owned.as_array();
            match arr.as_slice() {
                Some(s) => Ok(s.to_vec()),
                None => Err(PyValueError::new_err(format!("{name} is not contiguous"))),
            }
        };
    let draft = to_vec("draft_token_ids", draft_token_ids)?;
    let argmax = to_vec("target_argmax", target_argmax)?;
    let cu = to_vec("cu_num_draft_tokens", cu_num_draft_tokens)?;
    if draft.len() != argmax.len() {
        return Err(PyValueError::new_err(format!(
            "draft_token_ids ({}) and target_argmax ({}) length mismatch",
            draft.len(),
            argmax.len()
        )));
    }
    let batch = cu.len();
    let num_tokens = draft.len();
    let width = max_spec_len + 1;

    let mut accept_count = vec![0i32; batch];
    let mut emit_mask = vec![0i32; batch * width];
    for b in 0..batch {
        let start = if b == 0 { 0 } else { cu[b - 1] as usize };
        let end = (cu[b] as usize).min(num_tokens);
        let owned = (end.saturating_sub(start)).min(max_spec_len);
        let mut count = 0usize;
        for p in 0..owned {
            let idx = start + p;
            // Padded drafts (-1) and any mismatch stop the accept run, per
            // rejection_sampler.py:761 plus the -1 pad rule in the kernel.
            if draft[idx] >= 0 && draft[idx] == argmax[idx] {
                count += 1;
            } else {
                break;
            }
        }
        accept_count[b] = count as i32;
        for p in 0..=count.min(max_spec_len) {
            emit_mask[b * width + p] = 1;
        }
    }
    let masks: Vec<Vec<i32>> = (0..batch)
        .map(|b| emit_mask[b * width..(b + 1) * width].to_vec())
        .collect();
    Ok((
        accept_count.into_pyarray(py),
        PyArray2::from_vec2(py, &masks)?,
    ))
}