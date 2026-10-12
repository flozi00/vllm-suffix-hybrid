// SPDX-License-Identifier: Apache-2.0
//! One coherent chain: a native prefix followed by a conditional suffix tail.
//! The verifier's accepted-prefix lengths train a discounted split-bandit.
use super::*;
use numpy::ndarray::ArrayView2;
use pyo3::{exceptions::PyValueError, types::PyDict};
use std::collections::HashSet;

struct Previous {
    context: Vec<i64>,
    native: usize,
    length: usize,
    arm: usize,
    // Per-request expected acceptance, updated from this row's own feedback.
    accept_estimate: f64,
    // Per-request suffix quality: EWMA of accepted SUFFIX tokens on rows
    // that carried a non-native tail (None = no tail feedback yet, the
    // tail gets one trial). Gates replace-splices, never the budget.
    suffix_estimate: Option<f64>,
    // Whether the last proposal for this row carried a suffix tail.
    had_suffix: bool,
    // SPLIT=diverge: the first native/suffix disagreement of the last
    // proposal, resolved against the verified token next step.
    div: Option<Divergence>,
}

/// First position where the suffix candidate disagrees with the native
/// draft. Next step's context delta holds the TRUE token at `j` whenever
/// verification reached it (accepted >= j), whichever source was
/// published, so both sources are graded on the same event: a
/// counterfactual signal that needs no exploration.
#[derive(Clone, Copy)]
struct Divergence {
    j: usize,
    suffix_tok: i64,
    native_tok: i64,
    bucket: usize,
}

/// Evidence buckets for divergence trust: global-cache match length
/// (<16, <32, >=32) and in-request n-gram order (<8, <16, >=16).
const DIV_BUCKETS: usize = 6;

fn div_bucket(self_lookup: bool, matched: usize) -> usize {
    let tier = if matched < 16 { 0 } else if matched < 32 { 1 } else { 2 };
    tier + if self_lookup { 3 } else { 0 }
}

/// How this row's context relates to the tracked one. The list API (`mix`)
/// proves it with an exact full-prefix compare; `mix_numpy` proves it from
/// head+tail boundary windows and appends only the delta, so a long context
/// is copied once per request, not once per decode step (the O(rows x ctx)
/// memcpy per step was the dominant mixer cost at high concurrency).
/// Row classifier: given (row index, tracked context), return how the row
/// relates to the tracked context and the delta tokens to append.
type Decide<'a> = dyn Fn(usize, Option<&Vec<i64>>) -> (Continuity, Vec<i64>) + 'a;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Continuity {
    Fresh,
    Continuing,
    Reset,
}

/// Boundary windows proving a NumPy row extends the tracked context without
/// re-reading it whole. 64 tokens at each end on a >=32k vocabulary makes an
/// accidental match between unrelated requests negligible; a false positive
/// only pollutes one corpus sequence (drafts are always verified by the
/// target model), it cannot crash or corrupt serving state.
const BOUNDARY: usize = 64;

fn boundary_matches(tracked: &[i64], row: &[i64]) -> bool {
    let l = tracked.len();
    if row.len() < l {
        return false;
    }
    let w = BOUNDARY.min(l);
    tracked[..w] == row[..w] && tracked[l - w..] == row[l - w..l]
}

#[pyclass(module = "suffix_hybrid._native")]
pub struct HybridMixer {
    cache: SuffixCache,
    previous: Map<String, Previous>,
    k: usize,
    max_model_len: usize,
    calls: u64,
    decisions: u64,
    // Seed at FULL width: a row with no history has no evidence
    // against any slot, and the first proposal is native-only
    // anyway. Seeding below full width only shrinks the window the
    // EWMA learns from on step 0.
    initial: usize,
    last_native: Vec<usize>,
    native_tested: u64,
    native_accepted: u64,
    suffix_tested: u64,
    suffix_accepted: u64,
    native_proposed: u64,
    suffix_proposed: u64,
    elapsed_ns: u64,
    // SUFFIX_HYBRID_SPLIT=diverge (default off = legacy split-bandit).
    diverge: bool,
    div_threshold: f64,
    self_lo: usize,
    self_hi: usize,
    self_window: usize,
    // Discounted [suffix right, native right] counts per bucket.
    div_wins: [[f64; 2]; DIV_BUCKETS],
    div_observed: u64,
    div_suffix_published: u64,
    self_hits: u64,
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
            previous: map(),
            k,
            max_model_len,
            calls: 0,
            decisions: 0,
            initial: k,
            last_native: vec![],
            native_tested: 0,
            native_accepted: 0,
            suffix_tested: 0,
            suffix_accepted: 0,
            native_proposed: 0,
            suffix_proposed: 0,
            elapsed_ns: 0,
            diverge: std::env::var("SUFFIX_HYBRID_SPLIT").is_ok_and(|v| v.trim() == "diverge"),
            div_threshold: env_float("DIVERGE_THRESHOLD", 0.5).clamp(0.0, 1.0),
            self_lo: env_usize("SELF_NGRAM_MIN", 4, 1, 128),
            self_hi: env_usize("SELF_NGRAM_MAX", 16, 1, 128),
            // Bounded by default: this scan is on the decode critical path.
            self_window: env_usize("SELF_NGRAM_WINDOW", 8192, 0, 16_777_216),
            div_wins: [[0.0; 2]; DIV_BUCKETS],
            div_observed: 0,
            div_suffix_published: 0,
            self_hits: 0,
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
        // Panic boundary: a Rust panic must reach wrap.py's `except
        // Exception` native-fallback, never pyo3's BaseException-shaped
        // PanicException that would kill EngineCore (see lib.rs guard_py).
        super::guard_py("mix", || {
            self.validate(&request_ids, &native_drafts, accepted_lengths.as_deref())?;
            if contexts.len() != request_ids.len()
                || contexts.iter().any(|c| c.len() > self.max_model_len)
            {
                return Err(PyValueError::new_err(
                    "invalid hybrid batch dimensions, IDs, token IDs or budget",
                ));
            }
            // List API: the caller hands the full authoritative context, so
            // continuity is an exact prefix test (unchanged contract).
            let decide = |i: usize, tracked: Option<&Vec<i64>>| -> (Continuity, Vec<i64>) {
                let ctx = &contexts[i];
                match tracked {
                    None => (Continuity::Fresh, ctx.clone()),
                    Some(t) if ctx.starts_with(t.as_slice()) => {
                        (Continuity::Continuing, ctx[t.len()..].to_vec())
                    }
                    Some(_) => (Continuity::Reset, ctx.clone()),
                }
            };
            self.mix_core(
                request_ids,
                &native_drafts,
                accepted_lengths.as_deref(),
                step_time_ns,
                &decide,
            )
        })
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
        super::guard_py("mix_numpy", || {
            use numpy::{PyReadonlyArray1, PyReadonlyArray2};
            self.validate(&request_ids, &native_drafts, accepted_lengths.as_deref())?;
            let n = request_ids.len();
            let counts_i64 = num_tokens_no_spec
                .extract::<PyReadonlyArray1<'_, i64>>()
                .ok();
            let counts_i32 = if counts_i64.is_none() {
                num_tokens_no_spec
                    .extract::<PyReadonlyArray1<'_, i32>>()
                    .ok()
            } else {
                None
            };
            // Widened scratch for the i32 path (and non-contiguous i64); the
            // common contiguous-i64 path reads the NumPy view inline with zero
            // allocs. No unsafe: guards outlive the whole function body.
            let counts_widened: Vec<i64> = if let Some(ref a) = counts_i32 {
                let view = a.as_array();
                if view.len() < n {
                    return Err(PyValueError::new_err("missing counts"));
                }
                view.iter().map(|&x| x as i64).collect()
            } else if let Some(ref a) = counts_i64 {
                let view = a.as_array();
                if view.len() < n {
                    return Err(PyValueError::new_err("missing counts"));
                }
                match view.as_slice() {
                    Some(_) => Vec::new(), // contiguous: count_at reads the view
                    None => view.iter().copied().collect(),
                }
            } else {
                return Err(PyValueError::new_err("counts must be int32/int64 NumPy"));
            };
            let use_scratch = !counts_widened.is_empty()
                || counts_i64
                    .as_ref()
                    .is_some_and(|a| a.as_array().as_slice().is_none());
            let count_at = |i: usize| -> i64 {
                if use_scratch {
                    counts_widened[i]
                } else {
                    counts_i64.as_ref().unwrap().as_array()[i]
                }
            };
            // Row accessor per dtype. The live vLLM buffer is int64 and
            // C-contiguous: rows are zero-copy slices of one flat buffer and
            // the incremental path reads only boundary windows + the delta.
            // The i32 path materializes rows eagerly (correct, not the fast
            // path); vLLM's token_ids_cpu is int64.
            if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i64>>() {
                let a = a.as_array();
                if a.nrows() < n
                    || (0..n).any(|i| {
                        let c = count_at(i);
                        c < 0 || c as usize > a.ncols() || c as usize > self.max_model_len
                    })
                {
                    return Err(PyValueError::new_err("invalid row bounds"));
                }
                match a.as_slice() {
                    Some(flat) => {
                        let ncols = a.ncols();
                        let decide =
                            |i: usize, tracked: Option<&Vec<i64>>| -> (Continuity, Vec<i64>) {
                                let count = count_at(i) as usize;
                                let row = &flat[i * ncols..i * ncols + count];
                                match tracked {
                                    None => (Continuity::Fresh, row.to_vec()),
                                    Some(t) if boundary_matches(t, row) => {
                                        (Continuity::Continuing, row[t.len()..].to_vec())
                                    }
                                    Some(_) => (Continuity::Reset, row.to_vec()),
                                }
                            };
                        self.mix_core(
                            request_ids,
                            &native_drafts,
                            accepted_lengths.as_deref(),
                            step_time_ns,
                            &decide,
                        )
                    }
                    None => {
                        let decide =
                            |i: usize, tracked: Option<&Vec<i64>>| -> (Continuity, Vec<i64>) {
                                let count = count_at(i) as usize;
                                let row: Vec<i64> = a.row(i).iter().take(count).copied().collect();
                                match tracked {
                                    None => (Continuity::Fresh, row),
                                    Some(t) if row.len() >= t.len() && row[..t.len()] == t[..] => {
                                        (Continuity::Continuing, row[t.len()..].to_vec())
                                    }
                                    Some(_) => (Continuity::Reset, row),
                                }
                            };
                        self.mix_core(
                            request_ids,
                            &native_drafts,
                            accepted_lengths.as_deref(),
                            step_time_ns,
                            &decide,
                        )
                    }
                }
            } else if let Ok(a) = token_ids_cpu.extract::<PyReadonlyArray2<'_, i32>>() {
                let a = a.as_array();
                if a.nrows() < n
                    || (0..n).any(|i| {
                        let c = count_at(i);
                        c < 0 || c as usize > a.ncols() || c as usize > self.max_model_len
                    })
                {
                    return Err(PyValueError::new_err("invalid row bounds"));
                }
                let decide = |i: usize, tracked: Option<&Vec<i64>>| -> (Continuity, Vec<i64>) {
                    let count = count_at(i) as usize;
                    let row: Vec<i64> = a.row(i).iter().take(count).map(|&x| x as i64).collect();
                    match tracked {
                        None => (Continuity::Fresh, row),
                        Some(t) if row.len() >= t.len() && row[..t.len()] == t[..] => {
                            (Continuity::Continuing, row[t.len()..].to_vec())
                        }
                        Some(_) => (Continuity::Reset, row),
                    }
                };
                self.mix_core(
                    request_ids,
                    &native_drafts,
                    accepted_lengths.as_deref(),
                    step_time_ns,
                    &decide,
                )
            } else {
                Err(PyValueError::new_err("tokens must be int32/int64 NumPy"))
            }
        })
    }
    fn last_native_counts(&self) -> Vec<usize> {
        self.last_native.clone()
    }
    /// Cached-token count of the shared suffix cache. `tokens == 0` is
    /// exactly the `fed == false` predicate in `mix_core`, so a Python
    /// wrapper that sees 0 here can skip all per-step mixer work and
    /// publish the native drafts verbatim with provably identical output
    /// to the native-echo path (no cache to speculate from).
    fn cache_tokens(&self) -> usize {
        super::lock_cache(&self.cache.inner).tokens
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
        let est_mean = if self.previous.is_empty() {
            0.0
        } else {
            self.previous
                .values()
                .map(|p| p.accept_estimate)
                .sum::<f64>()
                / self.previous.len() as f64
        };
        d.set_item("mean_accept_estimate", est_mean)?;
        d.set_item("active_tracked", self.previous.len())?;
        d.set_item("split_mode", if self.diverge { "diverge" } else { "legacy" })?;
        d.set_item("div_observed", self.div_observed)?;
        d.set_item("div_suffix_published", self.div_suffix_published)?;
        d.set_item("self_hits", self.self_hits)?;
        d.set_item("div_trust", self.div_wins.iter().map(|w| Self::trust(w)).collect::<Vec<_>>())?;
        d.set_item("cache", self.cache.stats())?;
        Ok(d)
    }
}

impl HybridMixer {
    /// P(suffix right | the two sources disagree), Laplace prior: an
    /// unobserved bucket sits at exactly 0.5, so native wins the tie.
    fn trust(w: &[f64; 2]) -> f64 {
        (w[0] + 1.0) / (w[0] + w[1] + 2.0)
    }

    /// SPLIT=diverge per-row body. Suffix candidate is conditioned on the
    /// VERIFIED context only (global cache, else an in-request n-gram scan
    /// = prompt lookup over prompt + generation). If it agrees with native
    /// on every slot, native is published. At the first disagreement `j`
    /// the bucket's learned trust picks the source: suffix -> the draft is
    /// the suffix over its own length (slots < j are identical anyway),
    /// native filler past it keeps the published width == cap, so V2's
    /// fixed-K verify still yields feedback. The decision never reads
    /// native tokens beyond `j`: the same draft is an overlay of the
    /// suffix onto the native buffer, which a sync-free GPU merge can do.
    #[allow(clippy::too_many_arguments)]
    fn diverge_row(
        &mut self,
        cache: &mut Cache,
        id: String,
        i: usize,
        native: &[i64],
        accepted: i64,
        decide: &Decide<'_>,
    ) -> (Vec<i64>, usize) {
        use std::collections::hash_map::Entry;
        let slot = self.previous.entry(id);
        let tracked = match &slot {
            Entry::Occupied(o) => Some(&o.get().context),
            Entry::Vacant(_) => None,
        };
        let (continuity, delta) = decide(i, tracked);
        let p = match slot {
            Entry::Occupied(o) => {
                let p = o.into_mut();
                if continuity == Continuity::Continuing {
                    if let Some(d) = p.div.take() {
                        // delta[..a] = accepted drafts, delta[a] = bonus:
                        // the true token at j is known iff j <= a.
                        if accepted >= d.j as i64 && delta.len() > d.j {
                            let truth = delta[d.j];
                            let w = &mut self.div_wins[d.bucket];
                            w[0] *= 0.999;
                            w[1] *= 0.999;
                            w[0] += f64::from(u8::from(truth == d.suffix_tok));
                            w[1] += f64::from(u8::from(truth == d.native_tok));
                            self.div_observed += 1;
                        }
                    }
                    p.context.extend_from_slice(&delta);
                } else {
                    cache.add(std::mem::take(&mut p.context));
                    p.context = delta;
                    p.div = None;
                }
                p
            }
            Entry::Vacant(v) => v.insert(Previous {
                context: delta,
                native: 0,
                length: 0,
                arm: 0,
                accept_estimate: self.initial as f64,
                suffix_estimate: None,
                had_suffix: false,
                div: None,
            }),
        };
        let ctx = &p.context;
        let cap = self
            .k
            .min(self.max_model_len.saturating_sub(ctx.len()))
            .min(native.len());
        let (mut cand, mut matched) = {
            let (s, _, m) = cache.speculate(ctx, cap);
            (s, m)
        };
        let mut self_lookup = false;
        if cand.len() < cap && self.self_hi >= self.self_lo {
            let (s, score) = super::engine::ngram(ctx, self.self_lo, self.self_hi, cap, self.self_window);
            // ngram scores order n as n/(n+1).
            let order = (score / (1.0 - score)).round() as usize;
            if s.len() > cand.len() || (!s.is_empty() && order > matched) {
                cand = s;
                matched = order;
                self_lookup = true;
                self.self_hits += 1;
            }
        }
        cand.truncate(cap);
        let mut draft = native[..cap].to_vec();
        let mut native_count = cap;
        p.div = None;
        if let Some(j) = (0..cand.len()).find(|&t| cand[t] != native[t]) {
            let bucket = div_bucket(self_lookup, matched);
            p.div = Some(Divergence {
                j,
                suffix_tok: cand[j],
                native_tok: native[j],
                bucket,
            });
            if Self::trust(&self.div_wins[bucket]) > self.div_threshold {
                draft[..cand.len()].copy_from_slice(&cand);
                native_count = j;
                self.div_suffix_published += 1;
                self.suffix_proposed += (cand.len() - j) as u64;
            }
        }
        self.native_proposed += native_count as u64;
        p.native = native_count;
        p.length = draft.len();
        p.arm = cap;
        p.had_suffix = native_count < cap;
        (draft, native_count)
    }

    /// Shape validation shared by both entry points; runs BEFORE any state
    /// mutation so a rejected call leaves the mixer untouched (contract test).
    fn validate(
        &self,
        request_ids: &[String],
        native_drafts: &[Vec<i64>],
        accepted_lengths: Option<&[i64]>,
    ) -> PyResult<()> {
        let n = request_ids.len();
        // Duplicate-ID set doubles as the eviction allowlist below: build it
        // once, pre-sized, instead of hashing every ID twice per call.
        let mut id_set: HashSet<&str> = HashSet::with_capacity(n.max(1) * 2);
        id_set.extend(request_ids.iter().map(String::as_str));
        if n > 4096
            || native_drafts.len() != n
            || accepted_lengths.is_some_and(|v| v.len() != n)
            || id_set.len() != n
            || native_drafts
                .iter()
                .any(|d| d.len() > self.k || d.iter().any(|t| *t < 0))
        {
            return Err(PyValueError::new_err(
                "invalid hybrid batch dimensions, IDs, token IDs or budget",
            ));
        }
        if let Some(feedback) = &accepted_lengths {
            for (id, &accepted) in request_ids.iter().zip(*feedback) {
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
        Ok(())
    }

    /// The shared per-step body. `decide(i, tracked)` classifies row i
    /// against the tracked context and returns ONLY the delta to append
    /// (full context for Fresh/Reset rows), so the hot Continuing path is
    /// O(new tokens), never O(context).
    fn mix_core(
        &mut self,
        request_ids: Vec<String>,
        native_drafts: &[Vec<i64>],
        accepted_lengths: Option<&[i64]>,
        step_time_ns: Option<u64>,
        decide: &Decide<'_>,
    ) -> PyResult<Vec<Vec<i64>>> {
        let start = std::time::Instant::now();
        let n = request_ids.len();
        if let Some(feedback) = &accepted_lengths {
            for (id, &accepted) in request_ids.iter().zip(*feedback) {
                if accepted < 0 {
                    continue;
                }
                if let Some(p) = self.previous.get_mut(id) {
                    let accepted = accepted as usize;
                    // Exactly one rejected position is tested; the rest are censored.
                    let tested = (accepted + 1).min(p.length);
                    self.native_tested += tested.min(p.native) as u64;
                    self.native_accepted += accepted.min(p.native) as u64;
                    self.suffix_tested += tested.saturating_sub(p.native) as u64;
                    self.suffix_accepted += accepted.saturating_sub(p.native) as u64;
                    if p.length > 0 {
                        // Per-request EWMA of accepted tokens: this row's own
                        // history only. Censored tails must not train the
                        // estimate down (the tail was never verified), so a
                        // fully-accepted row keeps its budget instead of
                        // shrinking toward the observed prefix.
                        let _ = step_time_ns;
                        if accepted >= p.length {
                            // Fully accepted: jump straight back to the
                            // published width. Creeping up by halves keeps
                            // a healthy row one slot short for a dozen
                            // steps, donating a verified slot to
                            // speculation that earned nothing.
                            p.accept_estimate = p.length as f64;
                        } else {
                            p.accept_estimate = 0.7 * p.accept_estimate + 0.3 * accepted as f64;
                        }
                        // Suffix quality: accepted suffix tokens on proposals that
                        // carried a NON-NATIVE tail. Training on had_suffix
                        // rows (which include pure-continuation publishes)
                        // would teach the gate that doing nothing is failure.
                        if p.had_suffix && accepted.min(p.native) < p.native {
                            let suffix_got = accepted.saturating_sub(p.native) as f64;
                            p.suffix_estimate = Some(match p.suffix_estimate {
                                Some(e) => 0.7 * e + 0.3 * suffix_got,
                                None => suffix_got,
                            });
                        }
                    }
                }
            }
        }
        let mut id_set: HashSet<&str> = HashSet::with_capacity(n.max(1) * 2);
        id_set.extend(request_ids.iter().map(String::as_str));
        let ids: &HashSet<&str> = &id_set;
        let gone: Vec<String> = self
            .previous
            .keys()
            .filter(|id| !ids.contains(id.as_str()))
            .cloned()
            .collect();
        {
            let mut cache = super::lock_cache(&self.cache.inner);
            for id in gone {
                let previous = self.previous.remove(&id).unwrap();
                cache.add(previous.context);
            }
        }
        self.last_native.clear();
        self.last_native
            .reserve(n.saturating_sub(self.last_native.capacity()));
        let mut result = Vec::with_capacity(n);
        // One lock acquisition per mix() call: the cache's fed flag and every
        // speculate() below previously re-locked the mutex per row (2n+1
        // locks). A single guard covers validation-free reads; the evictions
        // above already dropped their borrows, so this cannot deadlock.
        // NOTE: SuffixCache wraps Arc<Mutex<Cache>> shared with Python, so a
        // per-row lock was also a per-row contention point with add_sequence
        // callers on other threads; holding one guard shortens that window.
        // Own Arc handle so the guard does not borrow `self` (diverge_row
        // needs &mut self while the one per-call lock is held).
        let cache_handle = self.cache.inner.clone();
        let mut cache = super::lock_cache(&cache_handle);
        // Reuses `conditioned` scratch across rows: the speculate window is
        // the last `depth` tokens plus the kept native prefix, pre-sized once
        // and truncated per row instead of cloned per row.
        let mut conditioned = Vec::with_capacity(cache.cfg.depth + self.k);
        // Copy the fields `pick` needs so it can run while a row's HashMap
        // entry borrow is held (one hash per row on the common path).
        let (initial, k) = (self.initial, self.k);
        // Per-request draft budget: expected accepted tokens (EWMA of this
        // row's own feedback, seeded at `initial`) rounded UP to a length in
        // 1..=k. A row that accepts 2 of 5 gets ~2-3 next, never 5 on repeat
        // failure. Ceil (not round) so a fully-accepted row converges back to
        // full width instead of hovering one slot below it forever.
        let pick = |estimate: Option<f64>| -> usize {
            let est = estimate.unwrap_or(initial as f64);
            (est.ceil() as usize).clamp(1, k)
        };
        for (i, (id, native)) in request_ids.into_iter().zip(native_drafts).enumerate() {
            if self.diverge {
                let accepted = accepted_lengths.map_or(-1, |a| a[i]);
                let (draft, native_count) =
                    self.diverge_row(&mut cache, id, i, native, accepted, decide);
                self.last_native.push(native_count);
                result.push(draft);
                self.decisions += 1;
                continue;
            }
            // ONE entry lookup per row: continuity, estimate, freshness,
            // gate and the final insert all read it (was: get + contains_key
            // + entry = three hashes/row).
            use std::collections::hash_map::Entry;
            let mut slot = self.previous.entry(id);
            let tracked = match &slot {
                Entry::Occupied(o) => Some(&o.get().context),
                Entry::Vacant(_) => None,
            };
            let (continuity, delta) = decide(i, tracked);
            // Reset: a different request reusing the slot. Finalize the old
            // history into the corpus BEFORE this row's speculate (original
            // ordering: the new request may legitimately continue the old
            // one's cached context) and clear the slot's context so the
            // conditioned window below never reads stale tokens.
            if continuity == Continuity::Reset {
                match &mut slot {
                    Entry::Occupied(o) => {
                        let old = std::mem::take(&mut o.get_mut().context);
                        cache.add(old);
                    }
                    Entry::Vacant(_) => {}
                }
            }
            let seen = matches!(&slot, Entry::Occupied(_));
            // A Reset row is a different request reusing the slot: like the
            // original remove+reinsert shape, it publishes native-only and
            // seeds fresh estimates (no per-row evidence for the new request).
            let fresh_row = matches!(&slot, Entry::Vacant(_)) || continuity == Continuity::Reset;
            // Effective context length after this step, without materializing
            // the merged context on the Continuing path.
            let context_len = match (&slot, continuity) {
                (Entry::Occupied(o), Continuity::Continuing) => o.get().context.len() + delta.len(),
                _ => delta.len(),
            };
            let cap = self
                .k
                .min(self.max_model_len.saturating_sub(context_len))
                .min(native.len());
            // A row whose context no longer extends the tracked one is a
            // different request reusing the slot: finalize the old history
            // and reset the estimate.
            let mut estimate = None;
            if continuity == Continuity::Continuing {
                estimate = match &slot {
                    Entry::Occupied(o) => Some(o.get().accept_estimate),
                    Entry::Vacant(_) => None,
                };
            } else if matches!(&slot, Entry::Occupied(_)) {
                // Reset: the old context goes to the corpus below (kept in
                // place until then so the borrow above stays simple).
                estimate = None;
            }
            // Dynamic budget: expected accepted tokens for THIS row, capped
            // by what the native drafter actually produced. When the suffix
            // cache has never been fed (fresh process, no accepted context
            // yet) the native draft passes through untouched: shortening the
            // publish without any suffix evidence only shrinks the window
            // the EWMA learns from.
            let fed = cache.tokens > 0;
            let budget = if fed { pick(estimate).min(cap) } else { cap };
            // Arbitrate per row: the native prefix fills the first `prefix`
            // slots; the suffix cache is asked for a tail conditioned on it
            // and may REPLACE native tokens the row has not earned. The
            // published draft always has exactly `budget` slots: every slot
            // is either kept native evidence or suffix evidence, never an
            // unconditioned native tail.
            // Suffix tail must EARN its slots from the SECOND proposal on:
            // the first proposal for a row is always native-only (the cache
            // has no per-row evidence yet). From then on the prefix is the
            // learned per-row budget, floored at cap-1 -- EXCEPT a row at
            // full budget keeps the whole native draft: there is no unearned
            // tail to replace, and the suffix tail can only append past it.
            let floor = cap.saturating_sub(1).max(1);
            let prefix = if seen && continuity != Continuity::Reset {
                let b = budget.min(cap);
                if b >= cap {
                    cap
                } else {
                    b.max(floor)
                }
            } else {
                cap
            };
            // Condition the lookup on context + the native prefix WITHOUT
            // the last kept token: the lookup key is the window ENDING at
            // the divergence point, so including a token the cache has never
            // seen after this context poisons the key and yields nothing.
            // Dropping one token costs nothing (the native draft still
            // supplies it) and lets a real cache continuation surface.
            // speculate() only ever reads the LAST `depth` tokens of this
            // key, so take that window from the delta (plus the tracked tail
            // when the delta is shorter) — never the whole context.
            conditioned.clear();
            let depth = cache.cfg.depth;
            if continuity == Continuity::Continuing {
                if let Entry::Occupied(o) = &slot {
                    let p = o.get();
                    let take = depth.saturating_sub(delta.len()).min(p.context.len());
                    conditioned.extend_from_slice(&p.context[p.context.len() - take..]);
                }
            }
            let dskip = delta.len().saturating_sub(depth.max(0));
            conditioned.extend_from_slice(&delta[dskip..]);
            conditioned.extend_from_slice(&native[..prefix.saturating_sub(1)]);
            let (suffix, _, _) = cache.speculate(&conditioned, cap);
            // Full native width is preserved whenever the suffix cache has
            // nothing to contribute, and for fresh rows proposing the
            // first time: shortening the publish without per-row evidence
            // only shrinks the window the EWMA learns from. The budget
            // binds the NATIVE PREFIX the tail is conditioned on (the
            // learned per-row split), not the published width.
            let gate = match &slot {
                Entry::Occupied(o) => o.get().suffix_estimate,
                Entry::Vacant(_) => None,
            };
            let (draft, native_count, had_suffix) = if suffix.is_empty() || fresh_row {
                (native[..cap].to_vec(), cap, false)
            } else {
                // The lookup was conditioned on prefix-1 tokens, so the tail
                // CONTINUES the kept native prefix -- position i of the tail
                // verifies against position (prefix-1)+i of the full draft.
                // cache tail only while the tail earns its slots for THIS
                // row (suffix_estimate >= 1, or no tail feedback yet: the
                // tail gets one trial). A tail the row keeps rejecting is
                // demoted to APPEND: it may only fill slots past the native
                // draft, never overwrite verified positions. Pure-extension
                // publishes (native_count == cap) train neither gate nor
                // estimate: they are no-ops, not evidence of failure.
                let mut combined = native[..prefix].to_vec();
                let keep_prefix = prefix.saturating_sub(1);
                let tail = suffix;
                let dropped = prefix.saturating_sub(keep_prefix);
                let replaces =
                    prefix > keep_prefix && tail.len() >= dropped && gate.is_none_or(|e| e >= 1.0);
                if replaces {
                    combined.truncate(keep_prefix);
                    combined.extend(tail.into_iter());
                } else {
                    // Gate closed (or short tail): extension only, from the
                    // full native prefix. Never overwrite verified slots.
                    combined = native[..prefix].to_vec();
                    let room = cap.saturating_sub(combined.len());
                    combined.extend(tail.into_iter().take(room));
                }
                let keep = combined.len().min(cap);
                combined.truncate(keep);
                // Slots where the tail merely continues the native draft are
                // still native evidence: count the leading run of the tail
                // that matches the native tokens it replaced.
                let mut native_count = keep_prefix.min(keep);
                while native_count < keep
                    && native_count < native.len()
                    && combined[native_count] == native[native_count]
                {
                    native_count += 1;
                }
                let had = combined.len() > native_count;
                (combined, native_count, had)
            };
            self.native_proposed += native_count as u64;
            self.suffix_proposed += draft.len().saturating_sub(native_count) as u64;
            self.last_native.push(native_count);
            // Per-row estimates persist across proposals; the feedback pass
            // above already updated them in place. Fresh rows seed the
            // acceptance EWMA at full width and leave the suffix gate open
            // (None = the tail gets one trial). The `slot` entry above
            // already holds the borrow, so reuse it instead of re-hashing.
            let seed = estimate.is_none();
            let (accept_estimate, suffix_estimate) = match &slot {
                Entry::Occupied(o) if !seed => {
                    let p = o.get();
                    (p.accept_estimate, p.suffix_estimate)
                }
                _ => (self.initial as f64, None),
            };
            match slot {
                Entry::Occupied(mut o) => {
                    let p = o.get_mut();
                    if continuity == Continuity::Continuing {
                        // Hot path: append only the new tokens.
                        p.context.extend_from_slice(&delta);
                    } else {
                        // Reset: the old history is finalized into the
                        // corpus; the reused slot starts the new request.
                        let old = std::mem::take(&mut p.context);
                        cache.add(old);
                        p.context = delta;
                    }
                    p.native = native_count;
                    p.length = draft.len();
                    p.arm = budget;
                    p.accept_estimate = accept_estimate;
                    p.suffix_estimate = suffix_estimate;
                    p.had_suffix = had_suffix;
                }
                Entry::Vacant(v) => {
                    v.insert(Previous {
                        context: delta,
                        native: native_count,
                        length: draft.len(),
                        arm: budget,
                        accept_estimate,
                        suffix_estimate,
                        had_suffix,
                        div: None,
                    });
                }
            }
            // NLL ends the entry borrow here; `result`/`self.decisions` below
            // touch disjoint state.
            result.push(draft);
            self.decisions += 1;
        }
        self.calls += 1;
        self.elapsed_ns = self
            .elapsed_ns
            .saturating_add(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        Ok(result)
    }
}

// ===================== V2 suffix-only proposer =====================
// Drafter-free speculative decoding for the V2 runner (gemma lane, wake
// #6-7): the wrapper never calls the native drafter forward, so a step
// costs one target forward of width (drafts + 1) — the 10x step cost of
// MTP is gone. ALL per-step algorithm work lives here in Rust (language
// contract, hard rule #3): the Python adapter gathers batch-order request
// ids, row indices, and total lengths (pure vLLM-contract wiring), then
// hands ONE zero-copy view of vLLM's host-resident token buffer
// (RequestState.all_token_ids — UVA pinned memory, int32) to a single
// call. Row reads, continuity proof, cache lookups, ingestion on
// departure, and width tracking are all Rust.
//
// State:
//   mirror: per-request FULL token history. Fresh rows copy the whole row
//     once (same cost class as engine.rs / mix_numpy Fresh); continuing
//     rows prove continuity with head+tail boundary windows and append
//     only the delta (<= K+1 per decode step).
//   widths: TRUE draft width published per request last step; the
//     adapter's get_draft_tokens patch reads this table (ragged rows:
//     width-0 -> plain 1-token decode at 1x cost).
#[pyclass(module = "suffix_hybrid._native")]
pub struct V2SuffixProposer {
    cache: SuffixCache,
    mirror: Map<String, Vec<i64>>,
    widths: Map<String, usize>,
    k: usize,
    max_model_len: usize,
    min_len: usize,
    /// Uniform CUDA-graph pallet mode: publish width=k for all rows
    /// (miss rows padded with -1 placeholders), so every decode step
    /// matches the compiled (k+1)-query uniform-decode CUDA graph.
    uniform_k: bool,
    steps: u64,
    hits: u64,
    hit_tokens: u64,
    ingested: u64,
    resets: u64,
    /// Rows published at width=k but padded with placeholders beyond
    /// the real suffix (uniform mode misses) — the pallet overhead
    /// meter (sum over steps of row counts, not tokens).
    padded: u64,
    /// Acceptance-EWMA draft-width gate (SUFFIX_HYBRID_EWMA_WIDTH,
    /// default OFF): per-request EWMA of accepted draft tokens,
    /// publishing w = ceil(EWMA).clamp(1, k) instead of the full
    /// matched suffix width. Port of mix_core's accept_estimate/pick
    /// (verify-econ dossier C.#1): at c1-warm acceptance ~0.024 the
    /// ungated path pays a (k+1)-wide verify forward per step and
    /// rejects ~97.6% of it; the gate narrows the publish toward what
    /// the row actually wins (mirror-length delta - 1 is the per-step
    /// accepted count), width clamp floor 1 / ceiling k.
    width_gate: bool,
    /// Per-request gate state; separate from `mirror` so the OFF arm's
    /// memory profile is unchanged (empty map costs nothing).
    gate: Map<String, GateRow>,
    elapsed_ns: u64,
}

struct SuffixRowOut {
    tokens: Vec<i64>,
    /// PUBLISHED width (what the scheduler schedules): true width in
    /// ragged mode; k for every row in uniform mode (CUDA-graph pallet).
    width: usize,
    /// TRUE matched-suffix length (stats + uniform padding split point).
    real_w: usize,
}

/// Acceptance-EWMA width-gate state, per request (mirrors the hybrid
/// path's `Previous` accept_estimate EWMA in mix_core). `ewma` tracks
/// the number of draft tokens THIS row actually got accepted per
/// proposal; `last_width` is the width the gate last published for
/// this row, needed (a) to interpret the next step's mirror delta as
/// an accept count (feedback) and (b) to censor retractions.
struct GateRow {
    ewma: f64,
    last_width: usize,
}

impl V2SuffixProposer {
    /// Core per-batch algorithm, generic over the buffer dtype (live
    /// all_token_ids is int32; int64 accepted for tests).
    fn run<T: Copy + Into<i64>>(
        &mut self,
        request_ids: &[String],
        indices: &[usize],
        tokens: ArrayView2<'_, T>,
        totals: &dyn Fn(usize) -> i64,
    ) -> PyResult<Vec<SuffixRowOut>> {
        let n = request_ids.len();
        if indices.len() < n {
            return Err(PyValueError::new_err(
                "suffix-only batch dimension mismatch",
            ));
        }
        let ncols = tokens.ncols();
        // 1) Ingestion: rows that left the batch donate their FULL tracked
        // history (prompt + generation) to the shared corpus — the ONLY
        // ingestion path in this mode. Echo traffic then hits from the
        // first decode steps of the next request repeating the passage.
        let live: HashSet<&str> = request_ids.iter().map(|s| s.as_str()).collect();
        let gone: Vec<String> = self
            .mirror
            .keys()
            .filter(|rid| !live.contains(rid.as_str()))
            .cloned()
            .collect();
        for rid in gone {
            if let Some(row) = self.mirror.remove(&rid) {
                self.widths.remove(&rid);
                self.gate.remove(&rid);
                if row.len() > 8 {
                    lock_cache(&self.cache.inner).add(row);
                    self.ingested += 1;
                }
            }
        }
        // 2) Per-row: continuity proof, incremental extend, cache lookup.
        let k = self.k;
        let min_len = self.min_len;
        let mut padded_self: usize = 0;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let rid = &request_ids[i];
            let total = totals(i);
            if total < 0 || total as usize > ncols || total as usize > self.max_model_len {
                return Err(PyValueError::new_err("suffix-only row bounds"));
            }
            let total = total as usize;
            let row = tokens.row(indices[i]);
            let tracked = self.mirror.get(rid);
            let tracked_len = tracked.map(|t| t.len());
            // Boundary proof on the VIEW (no full copy to prove identity):
            // the tracked prefix's head and the tokens where it ends must
            // match. Same 64-token windows as mix_numpy's boundary check.
            let cont = match tracked {
                Some(t) if total >= t.len() => {
                    let l = t.len();
                    let w = BOUNDARY.min(l);
                    (t[..w]
                        .iter()
                        .zip(row.iter().take(w))
                        .all(|(&a, &b)| a == b.into()))
                        && (t[l - w..]
                            .iter()
                            .zip(row.iter().skip(l - w).take(w))
                            .all(|(&a, &b)| a == b.into()))
                }
                _ => false,
            };
            // Extend the tracked mirror IN PLACE: cloning it to append the
            // delta was an O(context) copy per row per step.
            if cont {
                let t = self.mirror.get_mut(rid).unwrap();
                let l = t.len();
                t.extend(row.iter().skip(l).take(total - l).map(|&x| x.into()));
            } else {
                if tracked.is_some() {
                    self.resets += 1;
                }
                self.mirror
                    .insert(rid.clone(), row.iter().take(total).map(|&x| x.into()).collect());
            }
            let tokens_row = &self.mirror[rid];
            let pad = tokens_row.first().copied().unwrap_or(0);
            // Cache lookup: the continuation starts at the true next token
            // (the buffer read happens AFTER postprocess_sampled wrote the
            // last sampled token — the mirror is EXACT, no shift needed).
            let (suffix, _score, _matched) =
                lock_cache(&self.cache.inner).speculate(tokens_row, k);
            let (mut tokens, mut real_w) = if !suffix.is_empty() && suffix.len() >= min_len {
                let w = suffix.len().min(k);
                (suffix[..w].to_vec(), w)
            } else {
                (vec![], 0)
            };

            // ---- Acceptance-EWMA width gate (SUFFIX_HYBRID_EWMA_WIDTH).
            // Port of mix_core's accept_estimate/pick (verify-econ
            // dossier C.#1). The engine verifies LAST step's published
            // width by construction: the next authoritative total
            // exceeds the previous mirror by exactly (accepted drafts
            // + 1 new sampled token) on a continuing row, so
            //     accepted = delta_since_last_view - 1
            // with delta = total - prev_mirror_len. Mislabels are
            // censored: feedback only trains when (a) continuity
            // proofs held this step, (b) delta >= 1 (a real verify
            // step elapsed), (c) last_width > 0 (drafts were actually
            // published AND not retracted by clear_widths, which
            // censors by zeroing last_width), and accepted is capped
            // at last_width. Fully-accepted rows jump the EWMA back
            // to the published width (mix_core's "don't hover one
            // slot short" rule); partials decay by 0.7/0.3 like the
            // hybrid path. Published width = ceil(EWMA).clamp(1, k)
            // via shortening the matched suffix, floor 1: a row that
            // still hits keeps at least one speculative position, so
            // the scheduled step remains width-(>=2) and the
            // num_speculative_tokens contract the engine sees
            // (per-row widths from widths_table) is unchanged — this
            // is the same truncation that already happens for
            // ragged short suffixes.
            if self.width_gate {
                let mut gate_w = k; // fresh row: no evidence, keep width
                let mut ewma = None;
                if let Some(g) = self.gate.get_mut(rid) {
                    if cont && g.last_width > 0 {
                        if let Some(l_prev) = tracked_len {
                            let delta = total - l_prev;
                            if delta >= 1 {
                                let a = (delta - 1).min(g.last_width);
                                g.ewma = if a >= g.last_width {
                                    // Fully accepted (RIGHT-censored: the
                                    // engine took every published draft; its
                                    // true capacity is unknown). mix_core
                                    // jumps the EWMA to p.length == cap;
                                    // here the publish was gate-narrowed so
                                    // the jump target is unknown -> creep up
                                    // one width per fully-accepted step.
                                    // Any partial acceptance below decays
                                    // the estimate right back.
                                    (g.last_width + 1) as f64
                                } else {
                                    0.7 * g.ewma + 0.3 * a as f64
                                };
                            }
                        }
                    }
                    gate_w = (g.ewma.ceil() as usize).clamp(1, k);
                    ewma = Some(g.ewma);
                }
                if gate_w < real_w {
                    tokens.truncate(gate_w);
                    real_w = gate_w;
                }
                // Record the width THIS step publishes (uniform mode
                // pads up to k afterwards; feedback censors at
                // out_row.width == k, which is exactly what the
                // engine verifies there). No entry yet -> insert with
                // the full-width seed (no evidence against width).
                let last_width = if self.uniform_k { k } else { real_w };
                match self.gate.get_mut(rid) {
                    Some(g) => g.last_width = last_width,
                    None => {
                        self.gate.insert(
                            rid.clone(),
                            GateRow {
                                ewma: ewma.unwrap_or(k as f64),
                                last_width,
                            },
                        );
                    }
                }
            }
            // Uniform-k pallet (uniform_k mode): publish width=k for
            // EVERY row — hit rows carry the real suffix, miss rows are
            // padded so every step matches the compiled (k+1)-query
            // uniform-decode CUDA graph. PAD VALIDITY IS A HARD CONTRACT
            // (gemma wake #10 crash): v0.30.0 forwards draft token IDs
            // through the target model's embedding during verification,
            // so an out-of-range ID (-1) trips a device-side
            // vectorized_gather assert and kills EngineCore. Pad with a
            // token that is ALWAYS in-vocab: the row's own first context
            // token (col 0 of the tracked history; every tracked row has
            // >= 1 token). Greedy verification is self-correcting
            // (rejected on mismatch) — validity is the only requirement,
            // plausibility is irrelevant.
            let out_row = if self.uniform_k {
                let mut t = tokens;
                while t.len() < k {
                    t.push(pad);
                }
                SuffixRowOut {
                    tokens: t,
                    width: k,
                    real_w,
                }
            } else {
                SuffixRowOut {
                    tokens,
                    width: real_w,
                    real_w,
                }
            };
            if out_row.real_w > 0 {
                self.hits += 1;
                self.hit_tokens += out_row.real_w as u64;
            }
            self.widths.insert(rid.clone(), out_row.width);
            padded_self += usize::from(self.uniform_k && out_row.real_w < out_row.width);
            out.push(out_row);
        }
        self.steps += 1;
        self.padded = self.padded.saturating_add(padded_self as u64);
        Ok(out)
    }
}

#[pymethods]
impl V2SuffixProposer {
    #[new]
    #[pyo3(signature = (num_speculative_tokens, max_model_len, min_len=1, uniform_k=false, width_gate=false))]
    fn new(
        num_speculative_tokens: usize,
        max_model_len: usize,
        min_len: usize,
        uniform_k: bool,
        width_gate: bool,
    ) -> PyResult<Self> {
        if !(1..=64).contains(&num_speculative_tokens) || !(1..=16_777_216).contains(&max_model_len)
        {
            return Err(PyValueError::new_err(
                "suffix-only k must be 1..64 and context 1..16777216",
            ));
        }
        Ok(Self {
            cache: SuffixCache::new(),
            mirror: map(),
            widths: map(),
            k: num_speculative_tokens,
            max_model_len,
            min_len: min_len.max(1),
            uniform_k,
            steps: 0,
            hits: 0,
            hit_tokens: 0,
            ingested: 0,
            resets: 0,
            padded: 0,
            width_gate,
            gate: map(),
            elapsed_ns: 0,
        })
    }

    /// Batch entry from the adapter. All state transitions live here.
    ///
    /// - `request_ids`: batch-order engine request ids (list[str])
    /// - `indices`:     batch-order req-state row indices (int64 1D NumPy
    ///                  or list), from RequestState.req_id_to_index
    /// - `totals`:      batch-order authoritative total lengths (int64 1D
    ///                  NumPy), read after the blocking D2H that fences
    ///                  the UVA buffer writes
    /// - `tokens`:      ZERO-COPY view of the full host-resident
    ///                  [max_num_reqs, max_model_len] token buffer
    ///                  (int32 live; int64 accepted)
    /// Returns (packed [rows, k] int64 drafts, [rows] int64 true widths).
    #[pyo3(signature = (request_ids, indices, totals, tokens))]
    fn propose_suffix_only(
        &mut self,
        py: Python<'_>,
        request_ids: Vec<String>,
        indices: &Bound<'_, PyAny>,
        totals: &Bound<'_, PyAny>,
        tokens: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
        let started = Instant::now();
        // Panic boundary guards the algorithm; numpy packing runs after
        // with the held GIL token (no unsafe re-acquisition).
        let rows = super::guard_py("propose_suffix_only", || {
            let totals = totals
                .extract::<PyReadonlyArray1<'_, i64>>()
                .map_err(|_| PyValueError::new_err("totals must be int64 1D NumPy"))?;
            let n = request_ids.len();
            if totals.as_array().len() < n {
                return Err(PyValueError::new_err("totals shorter than request ids"));
            }
            let t_vec: Vec<i64> = match totals.as_array().as_slice() {
                Some(_) => Vec::new(),
                None => totals.as_array().iter().copied().collect(),
            };
            let t_at = move |i: usize| -> i64 {
                if t_vec.is_empty() {
                    totals.as_array()[i]
                } else {
                    t_vec[i]
                }
            };
            let idx: Vec<i64> = indices
                .extract::<Vec<i64>>()
                .map_err(|_| PyValueError::new_err("indices must be ints"))?;
            if idx.len() < n {
                return Err(PyValueError::new_err("indices shorter than request ids"));
            }
            let idx_usize: Vec<usize> = idx.iter().map(|&x| x.max(0) as usize).collect();
            if let Ok(a) = tokens.extract::<PyReadonlyArray2<'_, i32>>() {
                self.run(&request_ids, &idx_usize, a.as_array(), &t_at)
            } else if let Ok(a) = tokens.extract::<PyReadonlyArray2<'_, i64>>() {
                self.run(&request_ids, &idx_usize, a.as_array(), &t_at)
            } else {
                Err(PyValueError::new_err("tokens must be int32/int64 2D NumPy"))
            }
        })?;
        // Pack [n, k] drafts and [n] widths into numpy (contiguous,
        // C-order; tiny: n x k <= 32 x 64). The 2-D shape is a hard
        // contract with the adapter's out.copy_() — a flat n*k vector
        // broadcast-fails for any n > 1 (mixer.rs flat-packing bug,
        // gemma lane wake #8).
        let n = rows.len();
        let k = self.k;
        let mut packed: Vec<i64> = vec![0; n * k];
        let mut widths_np: Vec<i64> = vec![0; n];
        for (i, r) in rows.iter().enumerate() {
            widths_np[i] = r.width as i64;
            for (j, &tok) in r.tokens.iter().enumerate() {
                if j < k {
                    packed[i * k + j] = tok;
                }
            }
        }
        self.elapsed_ns = self
            .elapsed_ns
            .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        let packed_np = numpy::ndarray::Array2::<i64>::from_shape_vec((n, k), packed)
            .map_err(|e| PyValueError::new_err(format!("suffix-only pack shape {n}x{k}: {e}")))?
            .into_pyarray(py);
        let widths_arr = widths_np.into_pyarray(py);
        Ok((
            packed_np.into_any().unbind(),
            widths_arr.into_any().unbind(),
        ))
    }

    /// True width published per request id at the last propose call. The
    /// adapter's get_draft_tokens patch consumes this for ragged rows.
    #[getter]
    fn widths_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (rid, &w) in &self.widths {
            d.set_item(rid, w)?;
        }
        Ok(d)
    }

    /// Shared-handle view of the live suffix cache (WARM-START seam,
    /// SUFFIX_HYBRID_WARMSTART): SuffixCache wraps the SAME
    /// Arc<Mutex<Cache>> the proposer speculates/ingests against, so
    /// `.suffix_cache.add_sequence(tokens)` here primes the exact cache
    /// the next propose call looks up — identical to the proven offline
    /// bench/scale_bench.py mixer.suffix_cache path. Mirror of the
    /// HybridMixer getter (same name, same handle semantics).
    #[getter]
    fn suffix_cache(&self) -> SuffixCache {
        self.cache.clone()
    }

    /// Retract every published width (all rows -> miss). The adapter calls
    /// this on its exception path: run() may have already inserted widths
    /// for this batch while the packed upload failed, and the scheduler
    /// would otherwise verify zeroed drafts at phantom widths. The width
    /// gate censors with the same signal: a retracted width was never
    /// verified, so its last_width goes to 0 and the next step's mirror
    /// delta cannot be misread as an accept count for it.
    fn clear_widths(&mut self) {
        for (rid, w) in self.widths.iter_mut() {
            *w = 0;
            if let Some(g) = self.gate.get_mut(rid) {
                g.last_width = 0;
            }
        }
    }

    /// Acceptance-EWMA width-gate diagnostics: per-rid EWMA and last
    /// published width (gate arm only; empty when the gate is off).
    #[getter]
    fn gate_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (rid, g) in &self.gate {
            d.set_item(rid, (g.ewma, g.last_width))?;
        }
        Ok(d)
    }

    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("steps", self.steps)?;
        d.set_item("hits", self.hits)?;
        d.set_item("hit_tokens", self.hit_tokens)?;
        d.set_item("ingested", self.ingested)?;
        d.set_item("resets", self.resets)?;
        d.set_item("padded", self.padded)?;
        d.set_item("uniform_k", self.uniform_k)?;
        d.set_item("width_gate", self.width_gate)?;
        d.set_item("active", self.mirror.len())?;
        d.set_item("elapsed_ns", self.elapsed_ns)?;
        d.set_item("cache", self.cache.stats())?;
        Ok(d)
    }
}
