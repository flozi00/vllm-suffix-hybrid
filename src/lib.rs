// SPDX-License-Identifier: Apache-2.0
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

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
struct Config {
    n: usize,
    depth: usize,
    sequences: usize,
    tokens: usize,
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
struct Cache {
    cfg: Config,
    sequences: HashMap<u64, Vec<i64>>,
    fifo: VecDeque<u64>,
    index: HashMap<Vec<i64>, VecDeque<(u64, usize)>>,
    next: u64,
    tokens: usize,
}
impl Cache {
    fn new(cfg: Config) -> Self {
        Self {
            cfg,
            sequences: HashMap::new(),
            fifo: VecDeque::new(),
            index: HashMap::new(),
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
        tokens.shrink_to_fit();
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
            let mut counts: HashMap<i64, (usize, u64, usize)> = HashMap::new();
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
#[pyclass(module = "suffix_hybrid._native")]
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
        self.inner.lock().unwrap().add(tokens);
    }
    fn add_prompt(&self, tokens: Vec<i64>) {
        let mut cache = self.inner.lock().unwrap();
        if cache.cfg.prompts {
            cache.add(tokens);
        }
    }
    fn speculate(&self, context_tokens: Vec<i64>, max_tokens: isize) -> (Vec<i64>, f64, usize) {
        self.inner
            .lock()
            .unwrap()
            .speculate(&context_tokens, max_tokens.max(0) as usize)
    }
    fn stats(&self) -> HashMap<String, usize> {
        self.inner.lock().unwrap().stats()
    }
}
mod engine;
mod mixer;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SuffixCache>()?;
    m.add_class::<engine::Engine>()?;
    m.add_class::<mixer::HybridMixer>()?;
    m.add("VERSION", "0.2.0-rust-v1")?;
    Ok(())
}
