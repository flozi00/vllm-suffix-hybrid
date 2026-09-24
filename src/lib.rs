// SPDX-License-Identifier: Apache-2.0
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// FxHash: the index maps are hit once per n-gram position on every add and
/// once per speculate; SipHash (std default) costs more than the bucket work
/// it guards. Keys are token IDs and short token windows — not adversarial
/// input (they come from the engine's own CPU buffers), so a non-cryptographic
/// hasher is the right trade.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}
const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
type Map<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

fn map<K, V>() -> Map<K, V> {
    Map::with_hasher(BuildHasherDefault::default())
}

/// Poison-safe lock: a panic anywhere while the cache lock is held would
/// otherwise poison the mutex permanently, and every later `lock().unwrap()`
/// panics too — inside a `#[pymethods]` fn that surfaces as pyo3's
/// PanicException, which subclasses BaseException and therefore ESCAPES
/// wrap.py's `except Exception` degrade-to-native guard, killing EngineCore
/// on every subsequent step (the 2026-09-21 GLM cascade shape). Drafts are
/// advisory — the target model verifies every token — so recovering the
/// guard from a poisoned mutex is strictly safer than propagating.
fn lock_cache(cache: &Arc<Mutex<Cache>>) -> MutexGuard<'_, Cache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run hot-path Rust work with a panic boundary. A panic inside `#[pymethods]`
/// surfaces as pyo3's PanicException, which subclasses BaseException and
/// therefore ESCAPES wrap.py's `except Exception` degrade-to-native guard —
/// the exact EngineCore-killer class from the 2026-09-21 GLM crash. Converting
/// a panic to a regular RuntimeError routes it into the wrapper's native
/// fallback instead: a lost draft costs acceptance, a dead engine costs the
/// pod. State touched mid-panic is plain HashMaps/Vecs (no invariants a later
/// call can violate), so recovering is safe.
fn guard_py<T, F: FnOnce() -> PyResult<T>>(what: &str, f: F) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "suffix_hybrid {what} panicked; caller must degrade to native drafts"
        ))
    })?
}

fn env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(format!("SUFFIX_HYBRID_{name}"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}
fn env_float(name: &str, default: f64) -> f64 {
    std::env::var(format!("SUFFIX_HYBRID_{name}"))
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|x| x.is_finite())
        .unwrap_or(default)
}
#[derive(Clone)]
pub(crate) struct Config {
    n: usize,
    depth: usize,
    sequences: usize,
    pub tokens: usize,
    positions: usize,
    candidates: usize,
    support: usize,
    factor: f64,
    offset: f64,
    prompts: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            n: env_usize("INDEX_N", 8, 1, 128),
            depth: env_usize("MAX_TREE_DEPTH", 64, 1, 4096),
            sequences: env_usize("MAX_CACHED_REQUESTS", 512, 0, 100_000),
            tokens: env_usize("MAX_CACHED_TOKENS", 1_048_576, 0, 100_000_000),
            positions: env_usize("MAX_POSITIONS_PER_NGRAM", 64, 1, 4096),
            candidates: env_usize("MAX_CANDIDATES", 32, 1, 4096),
            support: env_usize("MIN_TOKEN_COUNT", 1, 1, 4096),
            factor: env_float("MAX_SPEC_FACTOR", 1.0).max(0.0),
            offset: env_float("MAX_SPEC_OFFSET", 0.0),
            prompts: env_usize("INDEX_PROMPTS", 0, 0, 1) == 1,
        }
    }
}
pub struct Cache {
    pub(crate) cfg: Config,
    sequences: Map<u64, Vec<i64>>,
    fifo: VecDeque<u64>,
    index: Map<Vec<i64>, VecDeque<(u64, usize)>>,
    next: u64,
    tokens: usize,
}
impl Cache {
    fn new(cfg: Config) -> Self {
        Self {
            cfg,
            sequences: map(),
            fifo: VecDeque::new(),
            index: map(),
            next: 0,
            tokens: 0,
        }
    }
    fn evict(&mut self) {
        if let Some(id) = self.fifo.pop_front() {
            let seq = self.sequences.remove(&id).unwrap();
            self.tokens -= seq.len();
            // Eager cleanup: neither dead positions nor dead unique keys accumulate.
            for key in seq.windows(self.cfg.n) {
                if let Some(bucket) = self.index.get_mut(key) {
                    bucket.retain(|&(si, _)| si != id);
                    if bucket.is_empty() {
                        self.index.remove(key);
                    }
                }
            }
        }
    }
    fn add(&mut self, mut tokens: Vec<i64>) {
        if self.cfg.sequences == 0 || self.cfg.tokens <= self.cfg.n {
            return;
        }
        if tokens.len() > self.cfg.tokens {
            tokens = tokens.split_off(tokens.len() - self.cfg.tokens);
        }
        // No shrink_to_fit here: it reallocs and copies the whole sequence on
        // EVERY add (O(context) per finished request), and the bounded token
        // budget already caps total retention. The slack is one Vec per cached
        // sequence, bounded by MAX_CACHED_REQUESTS.
        if tokens.len() <= self.cfg.n {
            return;
        }
        while self.fifo.len() >= self.cfg.sequences || self.tokens + tokens.len() > self.cfg.tokens
        {
            self.evict();
        }
        let id = self.next;
        self.next += 1;
        for (pos, key) in tokens.windows(self.cfg.n).enumerate() {
            let bucket = self.index.entry(key.to_vec()).or_default();
            bucket.push_back((id, pos));
            if bucket.len() > self.cfg.positions {
                bucket.pop_front();
            }
        }
        self.tokens += tokens.len();
        self.sequences.insert(id, tokens);
        self.fifo.push_back(id);
    }
    fn speculate(&self, context: &[i64], max_tokens: usize) -> (Vec<i64>, f64, usize) {
        let window = &context[context.len().saturating_sub(self.cfg.depth)..];
        let n = self.cfg.n;
        if max_tokens == 0 || window.len() < n {
            return (vec![], 0.0, 0);
        }
        let Some(bucket) = self.index.get(&window[window.len() - n..]) else {
            return (vec![], 0.0, 0);
        };
        let mut best = 0;
        let mut ends = Vec::new();
        for &(id, pos) in bucket.iter().rev().take(self.cfg.candidates) {
            let seq = &self.sequences[&id];
            let mut matched = n;
            while matched < window.len()
                && matched < pos + n
                && seq[pos + n - matched - 1] == window[window.len() - matched - 1]
            {
                matched += 1;
            }
            if matched > best {
                best = matched;
                ends.clear();
            }
            if matched == best {
                ends.push((id, pos + n));
            }
        }
        let cap =
            max_tokens.min((self.cfg.factor * best as f64 + self.cfg.offset).max(0.0) as usize);
        let mut result = Vec::new();
        let mut score = 1.0;
        for step in 0..cap {
            let mut counts: Map<i64, (usize, u64, usize)> = map();
            for &(id, pos) in &ends {
                if let Some(&tok) = self.sequences[&id].get(pos + step) {
                    let item = counts.entry(tok).or_insert((0, id, pos));
                    item.0 += 1;
                    if (id, pos) > (item.1, item.2) {
                        item.1 = id;
                        item.2 = pos;
                    }
                }
            }
            let Some((&tok, &(support, _, _))) =
                counts.iter().max_by_key(|(tok, rank)| (**rank, **tok))
            else {
                break;
            };
            if support < self.cfg.support {
                break;
            }
            score *= support as f64 / ends.len() as f64;
            result.push(tok);
            ends.retain(|&(id, pos)| self.sequences[&id].get(pos + step) == Some(&tok));
        }
        if result.is_empty() {
            score = 0.0;
        }
        (result, score, best)
    }
    fn stats(&self) -> HashMap<String, usize> {
        HashMap::from([
            ("num_sequences".into(), self.sequences.len()),
            ("num_index_keys".into(), self.index.len()),
            (
                "index_positions".into(),
                self.index.values().map(|v| v.len()).sum(),
            ),
            ("cached_tokens".into(), self.tokens),
            (
                "allocated_token_capacity".into(),
                self.sequences.values().map(Vec::capacity).sum(),
            ),
            ("index_n".into(), self.cfg.n),
            ("max_cached_requests".into(), self.cfg.sequences),
            ("max_cached_tokens".into(), self.cfg.tokens),
        ])
    }
}
#[pyclass(module = "suffix_hybrid._native", skip_from_py_object)]
#[derive(Clone)]
struct SuffixCache {
    inner: Arc<Mutex<Cache>>,
}
#[pymethods]
impl SuffixCache {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Cache::new(Config::default()))),
        }
    }
    fn add_sequence(&self, tokens: Vec<i64>) {
        lock_cache(&self.inner).add(tokens);
    }
    fn add_prompt(&self, tokens: Vec<i64>) {
        let mut cache = lock_cache(&self.inner);
        if cache.cfg.prompts {
            cache.add(tokens);
        }
    }
    fn speculate(&self, context_tokens: Vec<i64>, max_tokens: isize) -> (Vec<i64>, f64, usize) {
        lock_cache(&self.inner).speculate(&context_tokens, max_tokens.max(0) as usize)
    }
    /// Test seam: pin the n-gram order for deterministic unit tests,
    /// independent of process env and pytest file order. Production code
    /// never calls this (n comes from SUFFIX_HYBRID_INDEX_N, default 8).
    fn set_test_n(&self, n: usize) {
        lock_cache(&self.inner).cfg.n = n.clamp(1, 128);
    }
    fn stats(&self) -> HashMap<String, usize> {
        lock_cache(&self.inner).stats()
    }
}
#[cfg(any(feature = "qwen-gdn-kernels", feature = "nvfp4-attn-kernels"))]
mod cubin_store;
mod engine;
mod mixer;
mod nvfp4_attn;
#[cfg(feature = "nvfp4-attn-kernels")]
mod nvfp4_attn_gpu;
mod qwen_gdn;
#[cfg(feature = "qwen-gdn-kernels")]
mod qwen_gdn_gpu;
mod verify_fusion;
#[cfg(feature = "cutile-kernels")]
mod verify_fusion_gpu;

/// K2-NVFP4 launch plan (src/nvfp4_attn.rs) as a dict; ValueError when the
/// shape is outside the kernel contract (the adapter must refuse, loudly).
#[pyfunction]
#[pyo3(signature = (batch, q_len, num_q_heads, num_kv_heads, head_dim, page_size, num_sms))]
fn nvfp4_attn_plan(
    batch: usize,
    q_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    num_sms: usize,
) -> PyResult<HashMap<String, usize>> {
    let p = nvfp4_attn::plan(
        batch,
        q_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        num_sms,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(HashMap::from([
        ("g".to_string(), p.g),
        ("gp".to_string(), p.gp),
        ("qt".to_string(), p.qt),
        ("nqt".to_string(), p.nqt),
        ("m".to_string(), p.m),
        ("tn".to_string(), p.tn),
        ("ns".to_string(), p.ns),
        ("dt".to_string(), p.dt),
        ("rows".to_string(), p.rows),
    ]))
}

/// Startup self-check shared by every prebuilt-cubin lane: load each
/// installed cubin through the CUDA driver; a toolchain/driver skew surfaces
/// here with the driver's error instead of as a tileiras JIT attempt on the
/// first launch. Returns the number of cubins loaded.
#[cfg(any(feature = "qwen-gdn-kernels", feature = "nvfp4-attn-kernels"))]
#[pyfunction]
fn cubin_store_driver_check(py: Python<'_>, device_ordinal: usize) -> PyResult<usize> {
    py.detach(|| {
        guard_py("cubin_store_driver_check", || {
            cubin_store::driver_load_all(device_ordinal)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        })
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SuffixCache>()?;
    m.add_class::<engine::Engine>()?;
    m.add_class::<mixer::HybridMixer>()?;
    m.add_class::<mixer::V2SuffixProposer>()?;
    m.add_function(wrap_pyfunction!(verify_fusion::rejection_greedy_accept, m)?)?;
    // K1 GPU twin: present only in feature-on (CI CUDA-toolkitted) builds.
    // Its ABSENCE is part of the startup kernel-path assertion: a pod that
    // needs the fused path and finds this import missing must fail loud,
    // never silently use a fallback verify path.
    #[cfg(feature = "cutile-kernels")]
    m.add_function(wrap_pyfunction!(
        verify_fusion_gpu::rejection_greedy_accept_cuda,
        m
    )?)?;
    // K-GDN1 (qwen3.8-27b GDN decode): CPU reference always; the cutile GPU
    // op only in `qwen-gdn-kernels` builds. HAS_QWEN_GDN_CUDA is what the
    // SUFFIX_QWEN_GDN=1 startup assertion reads (absent op = hard error).
    m.add_function(wrap_pyfunction!(qwen_gdn::gdn_decode_fused_ref, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::gdn_decode_fused_cuda, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(
        qwen_gdn_gpu::qwen_gdn_variant_bytecode,
        m
    )?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::qwen_gdn_compile_cubin, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::qwen_gdn_gpu_name, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::qwen_gdn_install_cubin, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::qwen_gdn_probe_bytecodes, m)?)?;
    #[cfg(feature = "qwen-gdn-kernels")]
    m.add_function(wrap_pyfunction!(qwen_gdn_gpu::qwen_gdn_jit_stats, m)?)?;
    m.add("HAS_QWEN_GDN_CUDA", cfg!(feature = "qwen-gdn-kernels"))?;
    // K2-NVFP4 attention: the plan is always present (CPU, sizes the split-KV
    // workspace); the CUDA op only in feature-on builds. HAS_NVFP4_ATTN_CUDA
    // is the startup kernel-path assertion's probe — an armed pod without it
    // refuses to start, it never silently keeps the FA2 decode route.
    m.add_function(wrap_pyfunction!(nvfp4_attn_plan, m)?)?;
    m.add("HAS_NVFP4_ATTN_CUDA", cfg!(feature = "nvfp4-attn-kernels"))?;
    #[cfg(feature = "nvfp4-attn-kernels")]
    {
        m.add_function(wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_paged_attn_cuda, m)?)?;
        for f in [
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_split_set, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_variant_bytecode, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_compile_cubin, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_gpu_name, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_install_cubin, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_allow_jit, m)?,
            wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_selfcheck, m)?,
        ] {
            m.add_function(f)?;
        }
        m.add_function(wrap_pyfunction!(nvfp4_attn_gpu::nvfp4_attn_jit_stats, m)?)?;
    }
    #[cfg(any(feature = "qwen-gdn-kernels", feature = "nvfp4-attn-kernels"))]
    m.add_function(wrap_pyfunction!(cubin_store_driver_check, m)?)?;
    m.add("VERSION", "0.2.0-rust-v1")?;
    Ok(())
}
