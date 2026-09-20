// SPDX-License-Identifier: Apache-2.0
use super::*;
use numpy::{ndarray::ArrayView2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, types::PyDict};
use std::time::Instant;

#[pyclass(module = "suffix_hybrid._native")]
pub struct Engine {
    cache: SuffixCache,
    rows: Vec<Vec<i64>>,
    k: usize,
    max_model_len: usize,
    max_rows: usize,
    ngram_min: usize,
    ngram_max: usize,
    use_ngram: bool,
    calls: u64,
    proposed: u64,
    suffix_proposals: u64,
    ngram_proposals: u64,
    match_sum: u64,
    native_ns: u64,
    total_ns: u64,
    log_interval: u64,
    stats_interval: u64,
}

fn ngram(context: &[i64], lo: usize, hi: usize, cap: usize) -> (Vec<i64>, f64) {
    if cap == 0 {
        return (vec![], 0.0);
    }
    for n in (lo..=hi.min(context.len().saturating_sub(1))).rev() {
        let tail = &context[context.len() - n..];
        // Latest preceding occurrence, never the suffix itself.
        for pos in (0..context.len() - n).rev() {
            if &context[pos..pos + n] == tail {
                let end = (pos + n + cap).min(context.len());
                return (context[pos + n..end].to_vec(), n as f64 / (n + 1) as f64);
            }
        }
    }
    (vec![], 0.0)
}

impl Engine {
    fn run<T: Copy + Into<i64>>(
        &mut self,
        sampled: &[Vec<i64>],
        counts: &[i64],
        tokens: ArrayView2<'_, T>,
    ) -> PyResult<Vec<Vec<i64>>> {
        let size = sampled.len();
        if size > self.max_rows || counts.len() < size || tokens.nrows() < size {
            return Err(PyValueError::new_err(
                "batch dimensions exceed buffers or MAX_ACTIVE_ROWS",
            ));
        }
        for &n in counts.iter().take(size) {
            if n < 0 || n as usize > tokens.ncols() || n as usize > self.max_model_len {
                return Err(PyValueError::new_err(
                    "num_tokens_no_spec outside buffer/max_model_len",
                ));
            }
        }
        // Counts validate bounds only, never request identity. Every reused snapshot
        // must equal the complete authoritative prefix (including on reordered rows).
        let mut old: Vec<Option<Vec<i64>>> = std::mem::take(&mut self.rows)
            .into_iter()
            .map(Some)
            .collect();
        let mut current = Vec::with_capacity(size);
        for (i, &n) in counts.iter().take(size).enumerate() {
            let n = n as usize;
            let row = tokens.row(i);
            let matches = |v: &Vec<i64>| {
                !v.is_empty()
                    && v.len() <= n
                    && v.iter().zip(row.iter()).all(|(&a, &b)| a == b.into())
            };
            let same = old.get(i).and_then(Option::as_ref).is_some_and(matches);
            let matched = if same {
                Some(i)
            } else {
                old.iter().position(|v| v.as_ref().is_some_and(matches))
            };
            let mut context = matched.and_then(|j| old[j].take()).unwrap_or_default();
            // Copy only the validated new portion, not the padded allocation.
            context.extend(
                row.iter()
                    .skip(context.len())
                    .take(n - context.len())
                    .map(|&x| x.into()),
            );
            current.push(context);
        }
        let mut cache = self.cache.inner.lock().unwrap();
        for previous in old.into_iter().flatten() {
            cache.add(previous);
        }
        let mut result = Vec::with_capacity(size);
        for (i, context) in current.iter().enumerate() {
            let cap = self.k.min(self.max_model_len.saturating_sub(context.len()));
            if sampled[i].is_empty() || cap == 0 {
                result.push(vec![]);
                continue;
            }
            let (suffix, score, matched) = cache.speculate(context, cap);
            let (ng, ng_score) = if self.use_ngram {
                ngram(context, self.ngram_min, self.ngram_max, cap)
            } else {
                (vec![], 0.0)
            };
            // Empirical confidence is a heuristic, NOT measured acceptance.
            let draft = if !suffix.is_empty()
                && score * suffix.len() as f64 >= ng_score * ng.len() as f64
            {
                self.suffix_proposals += 1;
                self.match_sum += matched as u64;
                suffix
            } else {
                if !ng.is_empty() {
                    self.ngram_proposals += 1;
                }
                ng
            };
            self.proposed += draft.len() as u64;
            result.push(draft);
        }
        self.rows = current;
        self.calls += 1;
        Ok(result)
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (num_speculative_tokens=8, max_model_len=32768))]
    fn new(num_speculative_tokens: usize, max_model_len: usize) -> PyResult<Self> {
        if max_model_len == 0 || max_model_len > 16_777_216 {
            return Err(PyValueError::new_err(
                "max_model_len must be in 1..16777216",
            ));
        }
        let ngram_min = env_usize("NGRAM_MIN", 4, 1, 128);
        let override_k = env_usize("MAX_SPEC_TOKENS", 0, 0, 4096);
        Ok(Self {
            cache: SuffixCache::new(),
            rows: vec![],
            k: if override_k == 0 {
                num_speculative_tokens.min(4096)
            } else {
                override_k
            },
            max_model_len,
            max_rows: env_usize("MAX_ACTIVE_ROWS", 4096, 1, 65536),
            ngram_min,
            ngram_max: env_usize("NGRAM_MAX", 16, ngram_min, 128),
            use_ngram: env_usize("USE_NGRAM", 1, 0, 1) == 1,
            calls: 0,
            proposed: 0,
            suffix_proposals: 0,
            ngram_proposals: 0,
            match_sum: 0,
            native_ns: 0,
            total_ns: 0,
            log_interval: env_usize("LOG_INTERVAL", 100, 1, 1_000_000) as u64,
            stats_interval: env_usize("STATS_INTERVAL", 100, 1, 1_000_000) as u64,
        })
    }
    #[getter]
    fn suffix_cache(&self) -> SuffixCache {
        self.cache.clone()
    }
    #[getter]
    fn num_speculative_tokens(&self) -> usize {
        self.k
    }
    #[getter]
    fn use_ngram(&self) -> bool {
        self.use_ngram
    }
    fn propose(
        &mut self,
        sampled_token_ids: Vec<Vec<i64>>,
        num_tokens_no_spec: &Bound<'_, PyAny>,
        token_ids_cpu: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<i64>>> {
        let started = Instant::now();
        let counts: Vec<i64> =
            if let Ok(a) = num_tokens_no_spec.extract::<PyReadonlyArray1<'_, i32>>() {
                a.as_array().iter().map(|&n| n as i64).collect()
            } else if let Ok(a) = num_tokens_no_spec.extract::<PyReadonlyArray1<'_, i64>>() {
                a.as_array().iter().copied().collect()
            } else {
                return Err(PyValueError::new_err(
                    "num_tokens_no_spec must be a 1D int32/int64 NumPy array",
                ));
            };
        let result = if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i32>>() {
            self.run(&sampled_token_ids, &counts, a.as_array())
        } else if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i64>>() {
            self.run(&sampled_token_ids, &counts, a.as_array())
        } else {
            Err(PyValueError::new_err(
                "token_ids_cpu must be a 2D int32/int64 NumPy array",
            ))
        };
        self.native_ns = self
            .native_ns
            .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        result
    }
    /// Return telemetry scheduling decisions so no proposer bookkeeping lives in Python.
    fn record_total(&mut self, elapsed_ns: u64) -> (bool, bool) {
        self.total_ns = self.total_ns.saturating_add(elapsed_ns);
        (
            self.calls % self.log_interval == 0,
            self.calls % self.stats_interval == 0,
        )
    }
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("version", "0.2.0-rust-v1")?;
        d.set_item("calls", self.calls)?;
        d.set_item("proposed", self.proposed)?;
        d.set_item("accepted", py.None())?;
        d.set_item("acceptance_rate", py.None())?;
        d.set_item("acceptance_source", "vllm_metrics_only")?;
        d.set_item("suffix_proposals", self.suffix_proposals)?;
        d.set_item("ngram_proposals", self.ngram_proposals)?;
        d.set_item(
            "avg_match_len",
            self.match_sum as f64 / self.suffix_proposals.max(1) as f64,
        )?;
        d.set_item("active_requests", self.rows.len())?;
        d.set_item(
            "active_context_tokens",
            self.rows.iter().map(Vec::len).sum::<usize>(),
        )?;
        d.set_item("native_time_ns", self.native_ns)?;
        d.set_item("total_proposer_time_ns", self.total_ns)?;
        d.set_item("cache", self.cache.stats())?;
        Ok(d)
    }
}
