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
    // Per-request expected acceptance, updated from this row's own feedback.
    accept_estimate: f64,
    // Per-request suffix quality: EWMA of accepted SUFFIX tokens on rows
    // that carried a non-native tail (None = no tail feedback yet, the
    // tail gets one trial). Gates replace-splices, never the budget.
    suffix_estimate: Option<f64>,
    // Whether the last proposal for this row carried a suffix tail.
    had_suffix: bool,
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
        d.set_item("cache", self.cache.stats())?;
        Ok(d)
    }
}

impl HybridMixer {
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
        let mut cache = super::lock_cache(&self.cache.inner);
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
