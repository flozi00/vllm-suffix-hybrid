// SPDX-License-Identifier: Apache-2.0
//! Multi-model router + GPU swap scheduler (dossiers/multimodel-serving.md).
//! N `vllm serve` children listen on 127.0.0.1 and share one GPU through vLLM
//! sleep mode. This router owns the public port: it routes by body `model`
//! (Host header fallback), merges /v1/models and /metrics, and its swapper
//! thread decides which child holds the GPU. std + serde_json only.
//!
//! Lifecycle: `multimodel_serve(config_json)` blocks (GIL released) until
//! `multimodel_shutdown()` or a fatal router error (Err -> RuntimeError). The
//! launcher (suffix_hybrid/multimodel.py) owns child processes; a failed
//! sleep/wake marks the model Failed (503 + /health 503) and the launcher's
//! supervisor restarts the pod. Nothing here retries silently.
use pyo3::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

const MAX_BODY: usize = 512 << 20; // d3 media arrives base64 inline
const MAX_HEAD: usize = 64 << 10;
const HEALTH_PERIOD: Duration = Duration::from_secs(1);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
// Consecutive failed child /health checks before the router reports 503 (and
// the launcher restarts the pod): one slow reply must not reboot every model.
const HEALTH_MISSES: u32 = 3;
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);
// Per-read timeout on /sleep and /wake_up: offloading tens of GiB to pinned
// host RAM takes seconds; ten minutes of silence is a dead child.
const SWAP_TIMEOUT: Duration = Duration::from_secs(600);
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(300);
const PUBLIC_EXACT: &[&str] = &[
    "/pooling", "/classify", "/score", "/rerank", "/tokenize", "/detokenize", "/invocations",
];
const HOP_BY_HOP: &[&str] = &[
    "connection", "keep-alive", "proxy-connection", "transfer-encoding", "content-length",
    "te", "trailer", "upgrade", "expect",
];

// ---------------------------------------------------------------- config ---

struct Model {
    name: String,
    port: u16,
    util: f64,
    hostnames: Vec<String>,
}
struct Config {
    budget: f64,
    level: u64,
    slice: Duration,
    models: Vec<Model>,
    awake: Vec<bool>,
    listen: String,
}

/// The launcher validated the full contract; this re-checks only what the
/// router itself relies on (fail closed if the two ever disagree). Unknown
/// fields (argv, env, boot_timeout_s) belong to the launcher.
fn parse_config(text: &str) -> Result<Config, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("config: {e}"))?;
    let num = |v: &Value, k: &str| {
        v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("config: {k} must be a number"))
    };
    let budget = num(&v, "gpu_budget")?;
    if !(budget > 0.0 && budget <= 0.95) {
        return Err(format!("config: gpu_budget {budget} outside (0, 0.95]"));
    }
    let level = v.get("sleep_level").and_then(Value::as_u64).unwrap_or(0);
    if !(1..=2).contains(&level) {
        return Err("config: sleep_level must be 1 or 2".into());
    }
    let slice = Duration::from_millis(v.get("slice_ms").and_then(Value::as_u64).ok_or("config: slice_ms must be a non-negative integer")?);
    let listen = v.get("listen").and_then(Value::as_str).ok_or("config: listen must be a string")?.to_owned();
    let raw = v.get("models").and_then(Value::as_array).ok_or("config: models must be a list")?;
    if raw.len() < 2 {
        return Err("config: multimodel serving needs at least 2 models".into());
    }
    let (mut models, mut awake, mut names) = (Vec::new(), Vec::new(), HashSet::new());
    for m in raw {
        let name = m.get("served_as").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("config: served_as must be a non-empty string")?;
        if !names.insert(name) {
            return Err(format!("config: duplicate served_as {name}"));
        }
        let port = m.get("port").and_then(Value::as_u64).filter(|p| (8100..=8199).contains(p)).ok_or_else(|| format!("config: {name}: port must be in 8100-8199"))?;
        let util = num(m, "gpu_memory_utilization")?;
        if !(util > 0.0 && util <= budget) {
            return Err(format!("config: {name}: gpu_memory_utilization {util} outside (0, gpu_budget]"));
        }
        let hostnames = match m.get("hostnames") {
            None => Vec::new(),
            Some(h) => h.as_array().and_then(|a| a.iter().map(|x| x.as_str().map(str::to_ascii_lowercase)).collect()).ok_or_else(|| format!("config: {name}: hostnames must be a list of strings"))?,
        };
        awake.push(m.get("awake").and_then(Value::as_bool).ok_or_else(|| format!("config: {name}: awake (boot state) must be a bool"))?);
        models.push(Model { name: name.to_owned(), port: port as u16, util, hostnames });
    }
    if models.iter().map(|m| m.port).collect::<HashSet<_>>().len() != models.len() {
        return Err("config: duplicate port".into());
    }
    let held: f64 = models.iter().zip(&awake).filter(|(_, a)| **a).map(|(m, _)| m.util).sum();
    if held > budget + 1e-9 {
        return Err(format!("config: boot state holds {held} > gpu_budget {budget}"));
    }
    Ok(Config { budget, level, slice, models, awake, listen })
}

// ------------------------------------------------------------- scheduler ---

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Asleep,
    Waking,
    Awake,
    Draining,
    Sleeping,
    Failed,
}
struct Slot {
    state: State,
    util: f64,
    inflight: usize,
    waiters: VecDeque<(u64, Instant)>,
    last_used: Instant,
    healthy: bool,
    wait_sum: f64,
    wait_count: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Io {
    Sleep(usize),
    Wake(usize),
}
#[derive(Debug, PartialEq)]
enum Step {
    Io(Io),
    Idle,
    Until(Instant),
}
#[derive(Debug, PartialEq)]
enum Enter {
    Now,
    Queue,
    Fail,
}

/// Pure, socket-free swap state machine. The swapper thread calls `next`,
/// performs the returned Io without the lock, then reports it with `done`.
struct Sched {
    slots: Vec<Slot>,
    budget: f64,
    slice: Duration,
    ticket: u64,
    swap: Option<(Instant, Option<usize>)>, // (started, first victim)
    swaps: BTreeMap<(Option<usize>, usize), u64>,
    swap_sum: f64,
    swap_count: u64,
}
impl Sched {
    fn new(utils: &[f64], awake: &[bool], budget: f64, slice: Duration, now: Instant) -> Self {
        let slots = utils.iter().zip(awake).map(|(&util, &a)| Slot {
            state: if a { State::Awake } else { State::Asleep },
            util, inflight: 0, waiters: VecDeque::new(), last_used: now,
            healthy: true, wait_sum: 0.0, wait_count: 0,
        }).collect();
        Sched { slots, budget, slice, ticket: 0, swap: None, swaps: BTreeMap::new(), swap_sum: 0.0, swap_count: 0 }
    }
    fn busy(&self, i: usize) -> bool {
        self.slots[i].inflight > 0 || !self.slots[i].waiters.is_empty()
    }
    /// GPU share held by every model not fully asleep. Failed counts: its
    /// memory state is unknown, so it is never handed to another model.
    fn held(&self) -> f64 {
        self.slots.iter().filter(|s| s.state != State::Asleep).map(|s| s.util).sum()
    }
    fn next(&mut self, now: Instant) -> Step {
        if let Some(i) = self.slots.iter().position(|s| s.state == State::Draining) {
            if self.slots[i].inflight > 0 {
                return Step::Idle;
            }
            self.slots[i].state = State::Sleeping;
            return Step::Io(Io::Sleep(i));
        }
        let Some((c, since)) = self.slots.iter().enumerate()
            .filter(|(_, s)| s.state == State::Asleep && !s.waiters.is_empty())
            .map(|(i, s)| (i, s.waiters[0].1))
            .min_by_key(|&(_, t)| t)
        else {
            return Step::Idle;
        };
        if self.slots[c].util <= self.budget - self.held() + 1e-9 {
            self.slots[c].state = State::Waking;
            self.swap.get_or_insert((now, None));
            return Step::Io(Io::Wake(c));
        }
        // Victim: idle Awake models first, then least recently used.
        let Some(v) = (0..self.slots.len())
            .filter(|&i| self.slots[i].state == State::Awake)
            .min_by_key(|&i| (self.busy(i), self.slots[i].last_used))
        else {
            return Step::Idle; // memory held by Waking/Sleeping/Failed only
        };
        if self.busy(v) && now < since + self.slice {
            return Step::Until(since + self.slice);
        }
        self.slots[v].state = State::Draining;
        self.swap.get_or_insert((now, Some(v)));
        if self.slots[v].inflight > 0 {
            return Step::Idle;
        }
        self.slots[v].state = State::Sleeping;
        Step::Io(Io::Sleep(v))
    }
    fn done(&mut self, io: Io, ok: bool, now: Instant) {
        match io {
            Io::Sleep(i) => self.slots[i].state = if ok { State::Asleep } else { State::Failed },
            Io::Wake(i) => {
                self.slots[i].state = if ok { State::Awake } else { State::Failed };
                self.slots[i].last_used = now;
                if let (true, Some((t0, from))) = (ok, self.swap) {
                    *self.swaps.entry((from, i)).or_default() += 1;
                    self.swap_sum += (now - t0).as_secs_f64();
                    self.swap_count += 1;
                }
            }
        }
        if !ok || matches!(io, Io::Wake(_)) {
            self.swap = None;
        }
    }
    fn enter(&self, i: usize) -> Enter {
        match self.slots[i].state {
            State::Awake => Enter::Now,
            State::Failed => Enter::Fail,
            _ => Enter::Queue,
        }
    }
    fn enqueue(&mut self, i: usize, now: Instant) -> u64 {
        self.ticket += 1;
        self.slots[i].waiters.push_back((self.ticket, now));
        self.ticket
    }
    fn leave(&mut self, i: usize, ticket: u64) {
        self.slots[i].waiters.retain(|&(t, _)| t != ticket);
    }
    fn admit(&mut self, i: usize, since: Instant, now: Instant) {
        let s = &mut self.slots[i];
        s.inflight += 1;
        s.last_used = now;
        s.wait_sum += (now - since).as_secs_f64();
        s.wait_count += 1;
    }
    fn release(&mut self, i: usize, now: Instant) {
        self.slots[i].inflight -= 1;
        self.slots[i].last_used = now;
    }
    fn healthy(&self) -> bool {
        self.slots.iter().all(|s| s.healthy && s.state != State::Failed)
    }
    fn render(&self, names: &[&str]) -> String {
        let label = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let mut o = String::from("# HELP suffix_mm_swaps_total GPU swaps (from = first model put to sleep, none when the woken model fit).\n# TYPE suffix_mm_swaps_total counter\n");
        for (&(from, to), n) in &self.swaps {
            o += &format!("suffix_mm_swaps_total{{from=\"{}\",to=\"{}\"}} {n}\n", from.map_or("none".into(), |f| label(names[f])), label(names[to]));
        }
        o += &format!("# HELP suffix_mm_swap_seconds Swap decision to woken model awake (drain + sleep + wake).\n# TYPE suffix_mm_swap_seconds summary\nsuffix_mm_swap_seconds_sum {}\nsuffix_mm_swap_seconds_count {}\n", self.swap_sum, self.swap_count);
        o += "# HELP suffix_mm_wait_seconds Time requests queued for their model to hold the GPU.\n# TYPE suffix_mm_wait_seconds summary\n";
        for (s, n) in self.slots.iter().zip(names) {
            o += &format!("suffix_mm_wait_seconds_sum{{model=\"{0}\"}} {1}\nsuffix_mm_wait_seconds_count{{model=\"{0}\"}} {2}\n", label(n), s.wait_sum, s.wait_count);
        }
        type Gauge = fn(&Slot) -> usize;
        let gauges: [(&str, &str, Gauge); 3] = [
            ("queue_depth", "Requests waiting for their model to wake.", |s| s.waiters.len()),
            ("awake", "1 while the model holds the GPU (Awake or Draining).", |s| matches!(s.state, State::Awake | State::Draining) as usize),
            ("inflight", "Requests forwarded and not yet finished.", |s| s.inflight),
        ];
        for (g, help, f) in gauges {
            o += &format!("# HELP suffix_mm_{g} {help}\n# TYPE suffix_mm_{g} gauge\n");
            for (s, n) in self.slots.iter().zip(names) {
                o += &format!("suffix_mm_{g}{{model=\"{}\"}} {}\n", label(n), f(s));
            }
        }
        o
    }
}

// ------------------------------------------------------------------ http ---

struct Head {
    start: String,
    headers: Vec<(String, String)>,
}
impl Head {
    fn get(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}
type HttpErr = (u16, String);

fn line<R: BufRead>(r: &mut R, budget: &mut usize) -> Result<Vec<u8>, HttpErr> {
    let mut buf = Vec::new();
    r.by_ref().take(*budget as u64 + 1).read_until(b'\n', &mut buf).map_err(|e| (400, format!("read: {e}")))?;
    if buf.len() > *budget {
        return Err((431, "header section too large".into()));
    }
    *budget -= buf.len();
    Ok(buf)
}
/// Start line + headers. Ok(None) = peer closed before sending anything.
fn read_head<R: BufRead>(r: &mut R) -> Result<Option<Head>, HttpErr> {
    let mut budget = MAX_HEAD;
    let first = line(r, &mut budget)?;
    if first.is_empty() {
        return Ok(None);
    }
    let start = String::from_utf8(first).map_err(|_| (400, "non-UTF-8 start line".to_string()))?.trim_end().to_owned();
    let mut headers = Vec::new();
    loop {
        let l = line(r, &mut budget)?;
        if l.is_empty() {
            return Err((400, "connection closed inside headers".into()));
        }
        let l = String::from_utf8(l).map_err(|_| (400, "non-UTF-8 header".to_string()))?;
        let l = l.trim_end_matches(['\r', '\n']);
        if l.is_empty() {
            return Ok(Some(Head { start, headers }));
        }
        let (k, v) = l.split_once(':').ok_or((400, format!("malformed header: {l}")))?;
        headers.push((k.trim().to_owned(), v.trim().to_owned()));
    }
}
/// Body per Content-Length or chunked framing, capped. `to_eof` reads an
/// unframed body to EOF (responses only).
fn read_body<R: BufRead>(r: &mut R, h: &Head, cap: usize, to_eof: bool) -> Result<Vec<u8>, HttpErr> {
    let lens: Vec<&str> = h.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.as_str()).collect();
    let io = |e: io::Error| (400, format!("body: {e}"));
    let too_big = || (413, format!("body exceeds {cap} bytes"));
    if let Some(te) = h.get("transfer-encoding") {
        if !lens.is_empty() {
            return Err((400, "both Content-Length and Transfer-Encoding".into()));
        }
        if !te.eq_ignore_ascii_case("chunked") {
            return Err((501, format!("unsupported Transfer-Encoding: {te}")));
        }
        let mut body = Vec::new();
        loop {
            let mut budget = MAX_HEAD;
            let size_line = String::from_utf8_lossy(&line(r, &mut budget)?).into_owned();
            let hex = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(hex, 16).map_err(|_| (400, format!("bad chunk size: {hex:?}")))?;
            if size == 0 {
                loop {
                    let t = line(r, &mut budget)?;
                    if t.is_empty() || t == b"\r\n" || t == b"\n" {
                        return Ok(body);
                    }
                }
            }
            if body.len() + size > cap {
                return Err(too_big());
            }
            let at = body.len();
            body.resize(at + size, 0);
            r.read_exact(&mut body[at..]).map_err(io)?;
            let mut crlf = [0u8; 2];
            r.read_exact(&mut crlf).map_err(io)?;
            if &crlf != b"\r\n" {
                return Err((400, "chunk not terminated by CRLF".into()));
            }
        }
    }
    match lens.as_slice() {
        [] if to_eof => {
            let mut body = Vec::new();
            r.take(cap as u64 + 1).read_to_end(&mut body).map_err(io)?;
            if body.len() > cap { Err(too_big()) } else { Ok(body) }
        }
        [] => Ok(Vec::new()),
        [n] => {
            let n: usize = n.parse().map_err(|_| (400, format!("bad Content-Length: {n}")))?;
            if n > cap {
                return Err(too_big());
            }
            let mut body = vec![0u8; n];
            r.read_exact(&mut body).map_err(io)?;
            Ok(body)
        }
        _ => Err((400, "multiple Content-Length headers".into())),
    }
}

/// Paths forwarded to a child. Everything else (dev endpoints: /sleep,
/// /wake_up, /collective_rpc, /reset_*, ...) is a 404 and never forwarded.
/// Encoded or dot segments are refused outright: no public path needs them.
fn public(path: &str) -> bool {
    !path.contains('%') && !path.contains("..") && (path.starts_with("/v1/") || PUBLIC_EXACT.contains(&path))
}

// Minimal top-level JSON object walk: finds `model` without materialising a
// 512 MiB body as a serde Value. Malformed JSON -> None (vLLM rejects it).
fn ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        i += 1;
    }
    i
}
fn string_end(b: &[u8], i: usize) -> Option<usize> {
    if b.get(i) != Some(&b'"') {
        return None;
    }
    let mut j = i + 1;
    loop {
        match *b.get(j)? {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
}
fn value_end(b: &[u8], i: usize) -> Option<usize> {
    match *b.get(i)? {
        b'"' => string_end(b, i),
        b'{' | b'[' => {
            let (mut depth, mut j) = (0usize, i);
            loop {
                match *b.get(j)? {
                    b'"' => {
                        j = string_end(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
        }
        _ => {
            let j = i + b[i..].iter().position(|c| matches!(c, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')).unwrap_or(b.len() - i);
            (j > i).then_some(j)
        }
    }
}
/// Top-level `model` of a JSON object body: None = absent/null/not JSON,
/// Some(Err) = present but not a string. Last duplicate wins, like json.loads.
fn body_model(b: &[u8]) -> Option<Result<String, ()>> {
    let mut i = ws(b, 0);
    if b.get(i) != Some(&b'{') {
        return None;
    }
    i = ws(b, i + 1);
    if b.get(i) == Some(&b'}') {
        return None;
    }
    let mut found = None;
    loop {
        let ke = string_end(b, i)?;
        let key: String = serde_json::from_slice(&b[i..ke]).ok()?;
        i = ws(b, ke);
        if b.get(i) != Some(&b':') {
            return None;
        }
        let vs = ws(b, i + 1);
        let ve = value_end(b, vs)?;
        if key == "model" {
            found = match serde_json::from_slice::<Value>(&b[vs..ve]) {
                Ok(Value::String(s)) => Some(Ok(s)),
                Ok(Value::Null) => None,
                _ => Some(Err(())),
            };
        }
        i = ws(b, ve);
        match b.get(i) {
            Some(b',') => i = ws(b, i + 1),
            Some(b'}') => return found,
            _ => return None,
        }
    }
}
fn resolve(models: &[Model], body: &[u8], host: Option<&str>) -> Result<usize, HttpErr> {
    match body_model(body) {
        Some(Ok(name)) => models.iter().position(|m| m.name == name).ok_or_else(|| (404, format!("model {name:?} is not served here"))),
        Some(Err(())) => Err((400, "\"model\" must be a string".into())),
        None => {
            let host = host.map(|h| h.rsplit_once(':').filter(|(_, p)| p.bytes().all(|c| c.is_ascii_digit())).map_or(h, |(h, _)| h).to_ascii_lowercase());
            host.and_then(|h| models.iter().position(|m| m.hostnames.contains(&h)))
                .ok_or_else(|| (400, "request names no model (body \"model\") and its Host matches no model hostname".into()))
        }
    }
}

/// Children's Prometheus text merged: one HELP/TYPE per family, samples
/// appended in child order. vLLM samples carry model_name labels; an exact
/// duplicate series (unlabelled python_/process_ collectors) keeps the first
/// child's value so the exposition stays valid.
// ponytail: dedupe drops the other children's process_* series; label them per child if anyone needs them.
fn merge_metrics(texts: &[String]) -> String {
    struct Fam<'a> {
        help: Option<&'a str>,
        kind: Option<&'a str>,
        samples: Vec<&'a str>,
    }
    let (mut order, mut fams, mut seen) = (Vec::new(), HashMap::new(), HashSet::new());
    for text in texts {
        let mut cur: Option<&str> = None;
        for l in text.lines() {
            let meta = l.strip_prefix("# HELP ").map(|r| (r, true)).or_else(|| l.strip_prefix("# TYPE ").map(|r| (r, false)));
            let fam_name = if let Some((rest, help)) = meta {
                let name = rest.split_whitespace().next().unwrap_or("");
                cur = Some(name);
                let f = fams.entry(name).or_insert_with(|| { order.push(name); Fam { help: None, kind: None, samples: Vec::new() } });
                let slot = if help { &mut f.help } else { &mut f.kind };
                slot.get_or_insert(l);
                continue;
            } else if l.starts_with('#') || l.trim().is_empty() {
                continue;
            } else {
                let name = l.split(['{', ' ']).next().unwrap_or("");
                cur.filter(|c| name.starts_with(c)).unwrap_or(name)
            };
            let series = l.rfind('}').map_or_else(|| l.split(' ').next().unwrap_or(l), |e| &l[..=e]);
            if seen.insert(series) {
                fams.entry(fam_name).or_insert_with(|| { order.push(fam_name); Fam { help: None, kind: None, samples: Vec::new() } }).samples.push(l);
            }
        }
    }
    let mut o = String::new();
    for name in order {
        let f = &fams[name];
        for l in f.help.iter().chain(&f.kind).chain(&f.samples) {
            o += l;
            o.push('\n');
        }
    }
    o
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "",
    }
}
/// Client writer that remembers whether any response byte went out, so a
/// panic or upstream failure only sends an error status on a clean socket.
struct Out<'a> {
    s: &'a TcpStream,
    sent: bool,
}
impl Write for Out<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.sent = true;
        (&*self.s).write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        (&*self.s).flush()
    }
}
fn respond(out: &mut Out, code: u16, ctype: &str, body: &[u8]) -> io::Result<()> {
    let mut msg = format!("HTTP/1.1 {code} {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reason(code), body.len()).into_bytes();
    msg.extend_from_slice(body);
    out.write_all(&msg)
}
fn respond_err(out: &mut Out, (code, msg): HttpErr) -> io::Result<()> {
    respond(out, code, "application/json", json!({"error": {"message": msg, "code": code}}).to_string().as_bytes())
}

/// One request to a child on 127.0.0.1 (health, dev endpoints, aggregation).
fn fetch(port: u16, method: &str, path: &str, timeout: Duration) -> Result<(u16, Vec<u8>), String> {
    let s = TcpStream::connect_timeout(&SocketAddr::from(([127, 0, 0, 1], port)), timeout).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(timeout)).and(s.set_write_timeout(Some(timeout))).map_err(|e| e.to_string())?;
    let _ = s.set_nodelay(true);
    (&s).write_all(format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).map_err(|e| e.to_string())?;
    let mut r = BufReader::new(&s);
    let head = read_head(&mut r).map_err(|e| e.1)?.ok_or("closed before responding")?;
    let code = head.start.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or_else(|| format!("bad status line: {}", head.start))?;
    Ok((code, read_body(&mut r, &head, MAX_BODY, true).map_err(|e| e.1)?))
}

// ---------------------------------------------------------------- router ---

struct Shared {
    cfg: Config,
    sched: Mutex<Sched>,
    cv: Condvar,
    stop: AtomicBool,
    fatal: Mutex<Option<String>>,
    addr: SocketAddr,
}
impl Shared {
    /// Poison-safe: a panicking connection thread must not wedge the router.
    fn lock(&self) -> MutexGuard<'_, Sched> {
        self.sched.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn names(&self) -> Vec<&str> {
        self.cfg.models.iter().map(|m| m.name.as_str()).collect()
    }
    fn shutdown(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.cv.notify_all();
        let ip = if self.addr.ip().is_unspecified() { [127, 0, 0, 1].into() } else { self.addr.ip() };
        let _ = TcpStream::connect_timeout(&SocketAddr::new(ip, self.addr.port()), Duration::from_secs(1));
    }
    fn die(&self, msg: String) {
        eprintln!("SUFFIX_MULTIMODEL FATAL {msg}");
        self.fatal.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert(msg);
        self.shutdown();
    }
}

/// Holds one inflight slot; dropping it (normal end, client disconnect,
/// panic unwind) releases the slot and wakes the swapper.
struct Lease<'a>(&'a Shared, usize);
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.lock().release(self.1, Instant::now());
        self.0.cv.notify_all();
    }
}
fn acquire(sh: &Shared, i: usize) -> Result<Lease<'_>, String> {
    let t0 = Instant::now();
    let mut s = sh.lock();
    let failed = || format!("model {} failed to swap; the pod restarts", sh.cfg.models[i].name);
    match s.enter(i) {
        Enter::Now => {
            s.admit(i, t0, t0);
            return Ok(Lease(sh, i));
        }
        Enter::Fail => return Err(failed()),
        Enter::Queue => {}
    }
    let ticket = s.enqueue(i, t0);
    sh.cv.notify_all();
    // ponytail: a client that disconnects while queued is only noticed when
    // its turn comes (the write fails); poll the socket if queues get long.
    loop {
        s = sh.cv.wait(s).unwrap_or_else(|p| p.into_inner());
        let verdict = if sh.stop.load(Ordering::SeqCst) { Enter::Fail } else { s.enter(i) };
        match verdict {
            Enter::Now => {
                s.leave(i, ticket);
                s.admit(i, t0, Instant::now());
                return Ok(Lease(sh, i));
            }
            Enter::Fail => {
                s.leave(i, ticket);
                return Err(failed());
            }
            Enter::Queue => {}
        }
    }
}

fn swapper(sh: &Shared) {
    let mut s = sh.lock();
    while !sh.stop.load(Ordering::SeqCst) {
        let now = Instant::now();
        match s.next(now) {
            Step::Io(io) => {
                drop(s);
                let (i, verb) = match io { Io::Sleep(i) => (i, "sleep"), Io::Wake(i) => (i, "wake") };
                let m = &sh.cfg.models[i];
                let path = match io { Io::Sleep(_) => format!("/sleep?level={}&mode=wait", sh.cfg.level), Io::Wake(_) => "/wake_up".into() };
                let t = Instant::now();
                let result = fetch(m.port, "POST", &path, SWAP_TIMEOUT).and_then(|(code, body)| {
                    let awake = || serde_json::from_slice::<Value>(&body).ok().and_then(|v| v["status"].as_str().map(|s| s == "awake")).unwrap_or(false);
                    match (code, io) {
                        (200, Io::Sleep(_)) => Ok(()),
                        (200, Io::Wake(_)) if awake() => Ok(()),
                        _ => Err(format!("HTTP {code}: {}", String::from_utf8_lossy(&body))),
                    }
                });
                match &result {
                    Ok(()) => eprintln!("SUFFIX_MULTIMODEL SWAP {verb} {} ok in {:.2}s", m.name, t.elapsed().as_secs_f64()),
                    Err(e) => eprintln!("SUFFIX_MULTIMODEL FAILED {verb} {}: {e} (model marked Failed, /health 503)", m.name),
                }
                s = sh.lock();
                s.done(io, result.is_ok(), Instant::now());
                sh.cv.notify_all();
            }
            Step::Idle => s = sh.cv.wait_timeout(s, HEALTH_PERIOD).unwrap_or_else(|p| p.into_inner()).0,
            Step::Until(t) => s = sh.cv.wait_timeout(s, t.saturating_duration_since(now)).unwrap_or_else(|p| p.into_inner()).0,
        }
    }
}

fn health(sh: &Shared) {
    let mut misses = vec![0u32; sh.cfg.models.len()];
    while !sh.stop.load(Ordering::SeqCst) {
        for (i, m) in sh.cfg.models.iter().enumerate() {
            let answered = matches!(fetch(m.port, "GET", "/health", HEALTH_TIMEOUT), Ok((200, _)));
            misses[i] = if answered { 0 } else { misses[i] + 1 };
            let ok = misses[i] < HEALTH_MISSES;
            let mut s = sh.lock();
            if s.slots[i].healthy != ok {
                eprintln!("SUFFIX_MULTIMODEL HEALTH {} {}", m.name, if ok { "ok" } else { "FAILED" });
            }
            s.slots[i].healthy = ok;
        }
        thread::sleep(HEALTH_PERIOD);
    }
}

fn aggregate(sh: &Shared, path: &str) -> Result<Vec<Vec<u8>>, HttpErr> {
    sh.cfg.models.iter().map(|m| match fetch(m.port, "GET", path, FETCH_TIMEOUT) {
        Ok((200, body)) => Ok(body),
        Ok((code, _)) => Err((502, format!("{} {path}: HTTP {code}", m.name))),
        Err(e) => Err((502, format!("{} {path}: {e}", m.name))),
    }).collect()
}

fn handle(sh: &Shared, stream: &TcpStream, out: &mut Out) -> io::Result<()> {
    let _ = stream.set_nodelay(true);
    stream.set_read_timeout(Some(CLIENT_READ_TIMEOUT))?;
    let mut r = BufReader::new(stream);
    let head = match read_head(&mut r) {
        Ok(Some(h)) => h,
        Ok(None) => return Ok(()),
        Err(e) => return respond_err(out, e),
    };
    let mut parts = head.start.split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let path = target.split('?').next().unwrap_or("");
    match (method, path) {
        ("GET", "/health") => {
            let ok = sh.lock().healthy();
            respond(out, if ok { 200 } else { 503 }, "text/plain", b"")
        }
        ("GET", "/v1/models") => match aggregate(sh, "/v1/models") {
            Ok(bodies) => {
                let mut data = Vec::new();
                for (b, m) in bodies.iter().zip(&sh.cfg.models) {
                    match serde_json::from_slice::<Value>(b).ok().and_then(|v| v["data"].as_array().cloned()) {
                        Some(d) => data.extend(d),
                        None => return respond_err(out, (502, format!("{} /v1/models: no data list", m.name))),
                    }
                }
                respond(out, 200, "application/json", json!({"object": "list", "data": data}).to_string().as_bytes())
            }
            Err(e) => respond_err(out, e),
        },
        ("GET", "/metrics") => match aggregate(sh, "/metrics") {
            Ok(bodies) => {
                let texts: Vec<String> = bodies.into_iter().map(|b| String::from_utf8_lossy(&b).into_owned()).collect();
                let text = merge_metrics(&texts) + &sh.lock().render(&sh.names());
                respond(out, 200, "text/plain; version=0.0.4; charset=utf-8", text.as_bytes())
            }
            Err(e) => respond_err(out, e),
        },
        ("POST", p) if public(p) => {
            if head.get("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) {
                (&*stream).write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
            }
            let body = match read_body(&mut r, &head, MAX_BODY, false) {
                Ok(b) => b,
                Err(e) => return respond_err(out, e),
            };
            let _ = stream.set_read_timeout(None);
            match resolve(&sh.cfg.models, &body, head.get("host")) {
                Ok(i) => proxy(sh, i, method, target, &head, &body, out),
                Err(e) => respond_err(out, e),
            }
        }
        _ => respond_err(out, (404, format!("{method} {path} is not served by the multimodel router"))),
    }
}

/// Forward one request to child `i` and stream its response back byte for
/// byte until upstream EOF (SSE included). Dropping `up` on a client write
/// error closes the upstream socket, which makes vLLM abort the request.
// ponytail: a non-streaming request whose client vanished runs to completion
// upstream (the disconnect is seen on the first write back).
fn proxy(sh: &Shared, i: usize, method: &str, target: &str, head: &Head, body: &[u8], out: &mut Out) -> io::Result<()> {
    let _lease = match acquire(sh, i) {
        Ok(l) => l,
        Err(e) => return respond_err(out, (503, e)),
    };
    let up = match TcpStream::connect(("127.0.0.1", sh.cfg.models[i].port)) {
        Ok(s) => s,
        Err(e) => return respond_err(out, (502, format!("{}: {e}", sh.cfg.models[i].name))),
    };
    let _ = up.set_nodelay(true);
    let mut req = format!("{method} {target} HTTP/1.1\r\n");
    for (k, v) in &head.headers {
        if !HOP_BY_HOP.iter().any(|h| k.eq_ignore_ascii_case(h)) {
            req += &format!("{k}: {v}\r\n");
        }
    }
    req += &format!("Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let sent = (&up).write_all(req.as_bytes()).and((&up).write_all(body));
    let mut buf = vec![0u8; 64 << 10];
    let copy = || -> io::Result<()> {
        sent?;
        loop {
            let n = (&up).read(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            out.write_all(&buf[..n])?;
        }
    };
    match copy() {
        Err(e) if !out.sent => respond_err(out, (502, format!("{}: {e}", sh.cfg.models[i].name))),
        r => r,
    }
}

fn connection(sh: Arc<Shared>, stream: TcpStream) {
    let mut out = Out { s: &stream, sent: false };
    match catch_unwind(AssertUnwindSafe(|| handle(&sh, &stream, &mut out))) {
        Ok(_) => {} // client I/O errors end the connection; nothing to report
        Err(_) => {
            eprintln!("SUFFIX_MULTIMODEL connection handler panicked");
            if !out.sent {
                let _ = respond_err(&mut out, (500, "multimodel router panicked".into()));
            }
        }
    }
}

static RUNNING: Mutex<Option<Arc<Shared>>> = Mutex::new(None);
struct Running;
impl Drop for Running {
    fn drop(&mut self) {
        *RUNNING.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

fn spawn_critical(sh: &Arc<Shared>, name: &'static str, f: fn(&Shared)) -> Result<(), String> {
    let sh = sh.clone();
    thread::Builder::new().name(format!("mm-{name}")).spawn(move || {
        if catch_unwind(AssertUnwindSafe(|| f(&sh))).is_err() {
            sh.die(format!("{name} thread panicked"));
        }
    }).map(|_| ()).map_err(|e| format!("spawn {name}: {e}"))
}

fn serve(text: &str) -> Result<(), String> {
    let cfg = parse_config(text)?;
    let listener = TcpListener::bind(&cfg.listen).map_err(|e| format!("bind {}: {e}", cfg.listen))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let utils: Vec<f64> = cfg.models.iter().map(|m| m.util).collect();
    let sched = Sched::new(&utils, &cfg.awake, cfg.budget, cfg.slice, Instant::now());
    let sh = Arc::new(Shared { cfg, sched: Mutex::new(sched), cv: Condvar::new(), stop: AtomicBool::new(false), fatal: Mutex::new(None), addr });
    {
        let mut running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        if running.is_some() {
            return Err("a multimodel router is already running in this process".into());
        }
        *running = Some(sh.clone());
    }
    let _running = Running;
    spawn_critical(&sh, "swapper", swapper)?;
    spawn_critical(&sh, "health", health)?;
    eprintln!("SUFFIX_MULTIMODEL ROUTER listening {addr} models={}", sh.names().join(","));
    // ponytail: one thread per connection, unbounded; a pool/semaphore if
    // client fan-in ever reaches thousands of concurrent connections.
    for conn in listener.incoming() {
        if sh.stop.load(Ordering::SeqCst) {
            break;
        }
        match conn {
            Ok(stream) => {
                let sh = sh.clone();
                if let Err(e) = thread::Builder::new().name("mm-conn".into()).spawn(move || connection(sh, stream)) {
                    eprintln!("SUFFIX_MULTIMODEL spawn connection thread: {e}");
                }
            }
            Err(e) => eprintln!("SUFFIX_MULTIMODEL accept: {e}"),
        }
    }
    sh.stop.store(true, Ordering::SeqCst);
    sh.cv.notify_all();
    let fatal = sh.fatal.lock().unwrap_or_else(|p| p.into_inner()).take();
    fatal.map_or(Ok(()), Err)
}

/// Run the router until `multimodel_shutdown()` or a fatal error. Blocks
/// with the GIL released; a router error or panic raises RuntimeError.
#[pyfunction]
pub fn multimodel_serve(py: Python<'_>, config_json: &str) -> PyResult<()> {
    py.detach(|| catch_unwind(|| serve(config_json)))
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SUFFIX_MULTIMODEL router panicked"))?
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("SUFFIX_MULTIMODEL router: {e}")))
}

/// Stop the running router: `multimodel_serve` returns Ok. False when no
/// router runs in this process.
#[pyfunction]
pub fn multimodel_shutdown() -> bool {
    let sh = RUNNING.lock().unwrap_or_else(|p| p.into_inner()).clone();
    sh.map(|sh| sh.shutdown()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    const SLICE: Duration = Duration::from_millis(2000);

    fn two(now: Instant, utils: [f64; 2]) -> Sched {
        Sched::new(&utils, &[true, false], 0.92, SLICE, now)
    }
    #[test]
    fn wakes_without_sleeping_when_it_fits() {
        let t = Instant::now();
        let mut s = two(t, [0.4, 0.4]);
        assert_eq!(s.next(t), Step::Idle);
        assert_eq!(s.enter(1), Enter::Queue);
        s.enqueue(1, t);
        assert_eq!(s.next(t), Step::Io(Io::Wake(1)));
        assert_eq!(s.slots[1].state, State::Waking);
        s.done(Io::Wake(1), true, t);
        assert_eq!((s.slots[0].state, s.slots[1].state), (State::Awake, State::Awake));
        assert_eq!(s.swaps[&(None, 1)], 1);
    }
    #[test]
    fn victim_idle_first_then_lru() {
        let t = Instant::now();
        let later = t + Duration::from_millis(5);
        let mut s = Sched::new(&[0.45, 0.45, 0.45], &[true, true, false], 0.92, SLICE, t);
        s.admit(0, t, t); // model 0 busy, used before model 1
        s.slots[1].last_used = later;
        s.enqueue(2, later);
        assert_eq!(s.next(later), Step::Io(Io::Sleep(1)), "idle model goes first even if more recent");
        let mut s = Sched::new(&[0.45, 0.45, 0.45], &[true, true, false], 0.92, SLICE, t);
        s.slots[0].last_used = later;
        s.enqueue(2, later);
        assert_eq!(s.next(later), Step::Io(Io::Sleep(1)), "both idle: least recently used");
    }
    #[test]
    fn busy_model_keeps_gpu_for_a_slice_then_drains() {
        let t = Instant::now();
        let mut s = two(t, [0.8, 0.8]);
        s.admit(0, t, t);
        s.enqueue(1, t);
        assert_eq!(s.next(t + Duration::from_millis(10)), Step::Until(t + SLICE));
        assert_eq!(s.next(t + SLICE), Step::Idle);
        assert_eq!(s.slots[0].state, State::Draining);
        assert_eq!(s.enter(0), Enter::Queue, "Draining queues new requests");
        s.release(0, t + SLICE);
        let t2 = t + SLICE + Duration::from_millis(1);
        assert_eq!(s.next(t2), Step::Io(Io::Sleep(0)));
        s.done(Io::Sleep(0), true, t2);
        assert_eq!(s.next(t2), Step::Io(Io::Wake(1)), "A sleeps before B wakes");
        s.done(Io::Wake(1), true, t2 + Duration::from_secs(1));
        assert_eq!(s.swaps[&(Some(0), 1)], 1);
        assert!((s.swap_sum - 1.001).abs() < 1e-6);
        assert_eq!(s.enter(1), Enter::Now);
        assert_eq!(s.enter(0), Enter::Queue);
    }
    #[test]
    fn failure_marks_failed_and_unhealthy() {
        let t = Instant::now();
        let mut s = two(t, [0.8, 0.8]);
        s.enqueue(1, t);
        assert_eq!(s.next(t), Step::Io(Io::Sleep(0)));
        s.done(Io::Sleep(0), true, t);
        assert_eq!(s.next(t), Step::Io(Io::Wake(1)));
        s.done(Io::Wake(1), false, t);
        assert_eq!(s.enter(1), Enter::Fail);
        assert!(!s.healthy());
        assert!(s.swaps.is_empty());
        s.enqueue(0, t);
        assert_eq!(s.next(t), Step::Idle, "a Failed model's memory is never handed out");
    }

    fn head(raw: &str) -> (Head, &[u8]) {
        let mut r = raw.as_bytes();
        let h = read_head(&mut r).unwrap().unwrap();
        (h, r)
    }
    #[test]
    fn parses_content_length_chunked_and_cap() {
        let (h, mut rest) = head("POST /v1/x HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\nhello");
        assert_eq!(h.start, "POST /v1/x HTTP/1.1");
        assert_eq!(read_body(&mut rest, &h, 10, false).unwrap(), b"hello");
        let (h, mut rest) = head("POST / HTTP/1.1\r\ntransfer-encoding: Chunked\r\n\r\n5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\nX-T: 1\r\n\r\n");
        assert_eq!(read_body(&mut rest, &h, 11, false).unwrap(), b"hello world");
        let (h, mut rest) = head("POST / HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n");
        assert_eq!(read_body(&mut rest, &h, 10, false).unwrap_err().0, 413);
        let (h, mut rest) = head("POST / HTTP/1.1\r\nContent-Length: 11\r\n\r\nhello world");
        assert_eq!(read_body(&mut rest, &h, 10, false).unwrap_err().0, 413);
        let (h, mut rest) = head("POST / HTTP/1.1\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\nx");
        assert_eq!(read_body(&mut rest, &h, 10, false).unwrap_err().0, 400);
        let (h, mut rest) = head("POST / HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort");
        assert_eq!(read_body(&mut rest, &h, 10, false).unwrap_err().0, 400);
        assert!(read_head(&mut &b""[..]).unwrap().is_none());
        assert_eq!(read_head(&mut &b"GET / HTTP/1.1\r\nnocolon\r\n\r\n"[..]).err().unwrap().0, 400);
        let huge = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(read_head(&mut huge.as_bytes()).err().unwrap().0, 431);
    }

    fn models() -> Vec<Model> {
        ["d3", "gemma"].iter().enumerate().map(|(i, n)| Model { name: n.to_string(), port: 8101 + i as u16, util: 0.4, hostnames: vec![format!("{n}.pl-ai.net")] }).collect()
    }
    #[test]
    fn resolves_body_model_then_host() {
        let m = models();
        assert_eq!(resolve(&m, br#"{"messages":[{"model":"d3","x":"}\"{"}],"model":"gemma"}"#, None), Ok(1));
        assert_eq!(resolve(&m, br#" { "model" : "d3" , "a": [1, {"b": null}], "c": 1e5 } "#, Some("gemma.pl-ai.net")), Ok(0));
        assert_eq!(resolve(&m, br#"{"model":"d3","model":"gemma"}"#, None), Ok(1), "last duplicate wins");
        assert_eq!(resolve(&m, br#"{"model":"d3"}"#, None), Ok(0));
        assert_eq!(resolve(&m, br#"{"input":"x"}"#, Some("GEMMA.pl-ai.net:443")), Ok(1));
        assert_eq!(resolve(&m, b"not json", Some("d3.pl-ai.net")), Ok(0));
        assert_eq!(resolve(&m, br#"{"model":null}"#, Some("d3.pl-ai.net")), Ok(0));
        assert_eq!(resolve(&m, br#"{"model":"other"}"#, Some("d3.pl-ai.net")).unwrap_err().0, 404);
        assert_eq!(resolve(&m, br#"{"model":5}"#, None).unwrap_err().0, 400);
        assert_eq!(resolve(&m, br#"{"input":"x"}"#, Some("elsewhere")).unwrap_err().0, 400);
        assert_eq!(resolve(&m, br#"{"input":"x"}"#, None).unwrap_err().0, 400);
    }
    #[test]
    fn refuses_dev_and_odd_paths() {
        for p in ["/v1/chat/completions", "/v1/systemone", "/pooling", "/classify", "/tokenize", "/invocations"] {
            assert!(public(p), "{p}");
        }
        for p in ["/sleep", "/wake_up", "/is_sleeping", "/collective_rpc", "/reset_prefix_cache", "/metrics",
                  "/v1/../sleep", "/v1/%2e%2e/sleep", "/%73leep", "/v1", "/poolingx", "http://x/v1/chat"] {
            assert!(!public(p), "{p}");
        }
    }
    #[test]
    fn merges_metrics_one_header_per_family() {
        let a = "# HELP vllm:x X\n# TYPE vllm:x counter\nvllm:x_total{model_name=\"d3\"} 1\n# HELP h H\n# TYPE h histogram\nh_bucket{le=\"1\",model_name=\"d3\"} 2\nh_count{model_name=\"d3\"} 2\n# HELP process_cpu P\n# TYPE process_cpu counter\nprocess_cpu 5\n".to_string();
        let b = "# HELP vllm:x X\n# TYPE vllm:x counter\nvllm:x_total{model_name=\"gemma\"} 3\n# HELP process_cpu P\n# TYPE process_cpu counter\nprocess_cpu 9\n# EOF\n".to_string();
        let m = merge_metrics(&[a, b]);
        assert_eq!(m.matches("# HELP vllm:x ").count(), 1);
        assert_eq!(m.matches("# TYPE vllm:x ").count(), 1);
        let lines: Vec<&str> = m.lines().collect();
        let at = |l: &str| lines.iter().position(|x| *x == l).unwrap();
        assert!(at("# TYPE vllm:x counter") < at("vllm:x_total{model_name=\"d3\"} 1"));
        assert_eq!(at("vllm:x_total{model_name=\"d3\"} 1") + 1, at("vllm:x_total{model_name=\"gemma\"} 3"));
        assert!(at("# TYPE h histogram") < at("h_count{model_name=\"d3\"} 2"));
        assert!(m.contains("process_cpu 5\n") && !m.contains("process_cpu 9"));
        assert!(!m.contains("EOF"));
        let s = Sched::new(&[0.4, 0.4], &[true, false], 0.92, SLICE, Instant::now());
        let r = s.render(&["d3", "gemma"]);
        assert!(r.contains("suffix_mm_awake{model=\"d3\"} 1\n") && r.contains("suffix_mm_awake{model=\"gemma\"} 0\n"));
        assert!(r.contains("# TYPE suffix_mm_swap_seconds summary") && r.contains("suffix_mm_queue_depth{model=\"gemma\"} 0"));
    }
    #[test]
    fn config_refusals() {
        let ok = r#"{"listen":"127.0.0.1:0","gpu_budget":0.92,"sleep_level":1,"slice_ms":2000,"models":[
            {"served_as":"a","port":8101,"gpu_memory_utilization":0.8,"awake":false},
            {"served_as":"b","port":8102,"gpu_memory_utilization":0.8,"awake":true,"hostnames":["B.x"]}]}"#;
        let c = parse_config(ok).unwrap();
        assert_eq!(c.models[1].hostnames, vec!["b.x"]);
        for (from, to) in [("\"awake\":false", "\"awake\":true"), ("8102", "8101"), ("\"b\"", "\"a\""), ("0.92", "0.97"),
                           ("\"sleep_level\":1", "\"sleep_level\":3"), ("8101", "8200"), ("0.8,\"awake\":false", "0.93,\"awake\":false"),
                           ("\"awake\":false", "\"x\":1")] {
            assert!(parse_config(&ok.replacen(from, to, 1)).is_err(), "{from} -> {to}");
        }
        let one = r#"{"listen":"x","gpu_budget":0.9,"sleep_level":1,"slice_ms":1,"models":[{"served_as":"a","port":8101,"gpu_memory_utilization":0.8,"awake":true}]}"#;
        assert!(parse_config(one).err().unwrap().contains("at least 2"));
    }
}
