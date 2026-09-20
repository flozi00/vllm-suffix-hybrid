// SPDX-License-Identifier: Apache-2.0
//! One coherent chain: a native prefix followed by a conditional suffix tail.
//! The verifier's accepted-prefix lengths train a discounted split-bandit.
use super::*;
use pyo3::{exceptions::PyValueError, types::PyDict};
use std::collections::HashSet;

struct Previous {
    context: Vec<i64>,
    native: usize,
    length: usize,
    arm: usize,
}

#[pyclass(module = "suffix_hybrid._native")]
pub struct HybridMixer {
    cache: SuffixCache,
    previous: HashMap<String, Previous>,
    k: usize,
    max_model_len: usize,
    calls: u64,
    decisions: u64,
    initial: usize,
    rewards: Vec<f64>,
    trials: Vec<f64>,
    last_native: Vec<usize>,
    native_tested: u64,
    native_accepted: u64,
    suffix_tested: u64,
    suffix_accepted: u64,
    native_proposed: u64,
    suffix_proposed: u64,
    elapsed_ns: u64,
}

impl HybridMixer {
    fn choose(&self) -> usize {
        if self.decisions == 0 {
            return self.initial;
        }
        // Periodic exploration prevents the initial 80/20 prior locking in.
        if self.decisions % 8 == 0 {
            return (self.decisions as usize / 8) % self.k + 1;
        }
        (1..=self.k)
            .max_by(|&a, &b| {
                let score = |n: usize| (self.rewards[n] + 0.5) / (self.trials[n] + 1.0);
                score(a)
                    .total_cmp(&score(b))
                    .then_with(|| b.abs_diff(self.initial).cmp(&a.abs_diff(self.initial)))
            })
            .unwrap_or(1)
    }
}

#[pymethods]
impl HybridMixer {
    #[new]
    fn new(num_speculative_tokens: usize, max_model_len: usize) -> PyResult<Self> {
        if !(1..=64).contains(&num_speculative_tokens) || !(1..=16_777_216).contains(&max_model_len)
        {
            return Err(PyValueError::new_err(
                "hybrid budget must be 1..64 and context 1..16777216",
            ));
        }
        let k = num_speculative_tokens;
        Ok(Self {
            cache: SuffixCache::new(),
            previous: HashMap::new(),
            k,
            max_model_len,
            calls: 0,
            decisions: 0,
            initial: ((k as f64 * 0.8).round() as usize).clamp(1, k.saturating_sub(1).max(1)),
            rewards: vec![0.0; k + 1],
            trials: vec![0.0; k + 1],
            last_native: vec![],
            native_tested: 0,
            native_accepted: 0,
            suffix_tested: 0,
            suffix_accepted: 0,
            native_proposed: 0,
            suffix_proposed: 0,
            elapsed_ns: 0,
        })
    }
    #[getter]
    fn suffix_cache(&self) -> SuffixCache {
        self.cache.clone()
    }

    #[pyo3(signature = (request_ids, contexts, native_drafts, accepted_lengths=None, step_time_ns=None))]
    fn mix(
        &mut self,
        request_ids: Vec<String>,
        contexts: Vec<Vec<i64>>,
        native_drafts: Vec<Vec<i64>>,
        accepted_lengths: Option<Vec<i64>>,
        step_time_ns: Option<u64>,
    ) -> PyResult<Vec<Vec<i64>>> {
        let start = std::time::Instant::now();
        let n = request_ids.len();
        if n > 4096
            || contexts.len() != n
            || native_drafts.len() != n
            || accepted_lengths.as_ref().is_some_and(|v| v.len() != n)
            || request_ids.iter().collect::<HashSet<_>>().len() != n
            || contexts.iter().any(|c| c.len() > self.max_model_len)
            || native_drafts
                .iter()
                .any(|d| d.len() > self.k || d.iter().any(|t| *t < 0))
        {
            return Err(PyValueError::new_err(
                "invalid hybrid batch dimensions, IDs, token IDs or budget",
            ));
        }
        if let Some(feedback) = &accepted_lengths {
            for (id, &accepted) in request_ids.iter().zip(feedback) {
                if accepted < -1
                    || self
                        .previous
                        .get(id)
                        .is_some_and(|p| accepted > p.length as i64)
                {
                    return Err(PyValueError::new_err(
                        "accepted length exceeds previous proposal",
                    ));
                }
            }
        }
        if let Some(feedback) = &accepted_lengths {
            for (id, &accepted) in request_ids.iter().zip(feedback) {
                if accepted < 0 {
                    continue;
                }
                if let Some(p) = self.previous.get(id) {
                    let accepted = accepted as usize;
                    // Exactly one rejected position is tested; the rest are censored.
                    let tested = (accepted + 1).min(p.length);
                    self.native_tested += tested.min(p.native) as u64;
                    self.native_accepted += accepted.min(p.native) as u64;
                    self.suffix_tested += tested.saturating_sub(p.native) as u64;
                    self.suffix_accepted += accepted.saturating_sub(p.native) as u64;
                    if p.length > 0 {
                        // Forget stale traffic gradually. Native full-draft compute is
                        // currently constant across splits, so initial objective is
                        // accepted tokens per slot; caller can supply real step cost.
                        let reward = if let Some(ns) = step_time_ns.filter(|ns| *ns > 0) {
                            ((accepted + 1) as f64 * 1_000_000.0 / ns as f64).min(1.0)
                        } else {
                            accepted as f64 / p.length as f64
                        };
                        self.rewards[p.arm] = self.rewards[p.arm] * 0.98 + reward;
                        self.trials[p.arm] = self.trials[p.arm] * 0.98 + 1.0;
                    }
                }
            }
        }
        let ids: HashSet<&str> = request_ids.iter().map(String::as_str).collect();
        let gone: Vec<String> = self
            .previous
            .keys()
            .filter(|id| !ids.contains(id.as_str()))
            .cloned()
            .collect();
        for id in gone {
            let previous = self.previous.remove(&id).unwrap();
            self.cache.inner.lock().unwrap().add(previous.context);
        }
        self.last_native.clear();
        let mut result = Vec::with_capacity(n);
        for ((id, context), native) in request_ids.into_iter().zip(contexts).zip(native_drafts) {
            let cap = self
                .k
                .min(self.max_model_len.saturating_sub(context.len()))
                .min(native.len());
            if let Some(previous) = self.previous.get(&id) {
                if !context.starts_with(&previous.context) {
                    // Stable IDs supplied by the runner; reset/reuse still cannot join
                    // unrelated histories into a spurious cached sequence.
                    self.cache
                        .inner
                        .lock()
                        .unwrap()
                        .add(previous.context.clone());
                }
            }
            let arm = self.choose();
            let prefix = arm.min(cap);
            let mut conditioned = context.clone();
            conditioned.extend_from_slice(&native[..prefix]);
            let (suffix, _, _) = self
                .cache
                .inner
                .lock()
                .unwrap()
                .speculate(&conditioned, cap - prefix);
            let (draft, native_count) = if suffix.is_empty() {
                (native[..cap].to_vec(), cap)
            } else {
                let mut combined = native[..prefix].to_vec();
                combined.extend(suffix);
                // Do NOT append old native tokens after a changed suffix; they
                // were conditioned on a different prefix.
                (combined, prefix)
            };
            self.native_proposed += native_count as u64;
            self.suffix_proposed += draft.len().saturating_sub(native_count) as u64;
            self.last_native.push(native_count);
            self.previous.insert(
                id,
                Previous {
                    context,
                    native: native_count,
                    length: draft.len(),
                    arm: if native_count == draft.len() {
                        self.k
                    } else {
                        arm
                    },
                },
            );
            result.push(draft);
            self.decisions += 1;
        }
        self.calls += 1;
        self.elapsed_ns = self
            .elapsed_ns
            .saturating_add(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        Ok(result)
    }
    /// Read authoritative CPU buffers in Rust, without per-token Python objects.
    #[pyo3(signature = (request_ids, num_tokens_no_spec, token_ids_cpu, native_drafts, accepted_lengths=None, step_time_ns=None))]
    fn mix_numpy(
        &mut self,
        request_ids: Vec<String>,
        num_tokens_no_spec: &Bound<'_, PyAny>,
        token_ids_cpu: &Bound<'_, PyAny>,
        native_drafts: Vec<Vec<i64>>,
        accepted_lengths: Option<Vec<i64>>,
        step_time_ns: Option<u64>,
    ) -> PyResult<Vec<Vec<i64>>> {
        use numpy::{PyReadonlyArray1, PyReadonlyArray2};
        let counts: Vec<i64> =
            if let Ok(a) = num_tokens_no_spec.extract::<PyReadonlyArray1<'_, i32>>() {
                a.as_array().iter().map(|&x| x as i64).collect()
            } else if let Ok(a) = num_tokens_no_spec.extract::<PyReadonlyArray1<'_, i64>>() {
                a.as_array().iter().copied().collect()
            } else {
                return Err(PyValueError::new_err("counts must be int32/int64 NumPy"));
            };
        let n = request_ids.len();
        if counts.len() < n {
            return Err(PyValueError::new_err("missing counts"));
        }
        let read = |a: numpy::ndarray::ArrayView2<'_, i64>| -> PyResult<Vec<Vec<i64>>> {
            if a.nrows() < n
                || counts
                    .iter()
                    .take(n)
                    .any(|&c| c < 0 || c as usize > a.ncols() || c as usize > self.max_model_len)
            {
                return Err(PyValueError::new_err("invalid row bounds"));
            }
            Ok((0..n)
                .map(|i| a.row(i).iter().take(counts[i] as usize).copied().collect())
                .collect())
        };
        let contexts = if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i64>>() {
            read(a.as_array())?
        } else if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i32>>() {
            let a = a.as_array();
            if a.nrows() < n
                || counts
                    .iter()
                    .take(n)
                    .any(|&c| c < 0 || c as usize > a.ncols() || c as usize > self.max_model_len)
            {
                return Err(PyValueError::new_err("invalid row bounds"));
            }
            (0..n)
                .map(|i| {
                    a.row(i)
                        .iter()
                        .take(counts[i] as usize)
                        .map(|&x| x as i64)
                        .collect()
                })
                .collect()
        } else {
            return Err(PyValueError::new_err("tokens must be int32/int64 NumPy"));
        };
        self.mix(
            request_ids,
            contexts,
            native_drafts,
            accepted_lengths,
            step_time_ns,
        )
    }
    fn last_native_counts(&self) -> Vec<usize> {
        self.last_native.clone()
    }
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("calls", self.calls)?;
        d.set_item("native_tested", self.native_tested)?;
        d.set_item("native_accepted", self.native_accepted)?;
        d.set_item("suffix_tested", self.suffix_tested)?;
        d.set_item("suffix_accepted", self.suffix_accepted)?;
        d.set_item("native_proposed", self.native_proposed)?;
        d.set_item("suffix_proposed", self.suffix_proposed)?;
        d.set_item("native_mix_time_ns", self.elapsed_ns)?;
        d.set_item("split_trials", self.trials.clone())?;
        d.set_item("split_rewards", self.rewards.clone())?;
        d.set_item("next_native_budget", self.choose())?;
        d.set_item("cache", self.cache.stats())?;
        Ok(d)
    }
}
