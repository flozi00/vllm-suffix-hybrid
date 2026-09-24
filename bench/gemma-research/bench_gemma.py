#!/usr/bin/env python3
"""bench_gemma — gemma-spec-dev golden-rule bench driver (stdlib only).

Pool: gemma-spec-dev (vLLM v0.30.0, Gemma-4-26B-A4B NVFP4 MoE, SM120 RTX PRO
6000 96GB, 128k ctx, max-num-seqs 32) at http://gemma-spec-dev.pl-ai.net/v1.
Companion runbook: RUNBOOK-gemma-spec-dev.md. Ledger rows go to
bench/gemma-research/results-gemma.jsonl (golden_rule.py-compatible superset).

Design notes:
- Streaming (SSE) parsing measures TTFT and decode tok/s over the generation
  window only (after the first content token), i.e. excludes prefill time.
- Modes: --prompt-len chat | long (long = ~100k-token prefill estimate via a
  repeated filler paragraph; tokenizer-agnostic, char count logged; gen 8192).
  --corpus unique (per-run distinct prompts; spec-dec worst case) | reuse
  (deterministic same seed prompt set within a stage so the suffix cache hits).
- The multimodal probe synthesizes a tiny PNG in-memory (zlib+struct): no
  shelling out, no image libs.
- DRY-RUN LEDGER POLICY: --dry-run writes ONE ledger row flagged
  "dry_run": true (payloads built, nothing sent). To skip the row entirely,
  point --ledger at /dev/null.
"""
import argparse
import base64
import io
import json
import math
import random
import re
import socket
import ssl
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

DEFAULTS = dict(
    base_url="https://gemma-spec-dev.pl-ai.net",
    api_path="/v1/chat/completions",
    ledger="bench/gemma-research/results-gemma.jsonl",
    model="gemma-spec-dev",
)

VISION_B64_PNG = None  # module-level cache for the synthesized probe image


# ---------------------------------------------------------------- payload gen

def make_png_b64():
    """Synthesize a tiny valid PNG (zlib+struct only, no subprocess/libs)."""
    # 1 32x32 RGB chunk at +32 alpha. Header + IHDR + IDAT + IEND.
    w, h = 32, 32
    raw = b""
    for y in range(h):
        for x in range(w):
            # simple gradient so the tensor is not a constant plane
            raw += bytes(((x * 8) & 255, (y * 8) & 255, ((x + y) * 4) & 255))
    scan = b""
    for y in range(h):
        scan += b"\x00" + raw[y * w * 3:(y + 1) * w * 3]
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I",
            zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(scan, 6))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode("ascii")


def chat_prompt(run_seed, idx, unique):
    """Deterministic prompt; unique mode varies per run, reuse does not."""
    topics = ["kernel routing", "KV cache sizing", "flashinfer tactic choice",
              "speculative decoding", "rotary embeddings", "MoE routing bias"]
    rnd = random.Random(0 if not unique else run_seed)
    topic = topics[(idx + (rnd.randrange(6) if unique else 0)) % len(topics)]
    body = (f"Explain {topic} for a production inference engine, step by"
            f" step, precisely and at length."
            f"{' Variation ' + str(run_seed) + '.' if unique else ''}")
    return [{"role": "user", "content": body}]


FILLER = (
    "The quick brown fox jumps over the lazy dog while the engine schedules "
    "its next batch of tokens. "
)


def long_prompt_messages(target_chars):
    """~100k-token prefill estimate: ~4 chars/token heuristic, filler repeated.

    Tokenizer-agnostic by design: only the char count is claimed, and it is
    logged in every row (prompt_chars) and printed.
    """
    n = max(1, math.ceil(target_chars / len(FILLER)))
    text = FILLER * n
    return n, [{"role": "user", "content":
        text + "\n\nSummarize the passage above in detail, then continue "
               "the story at length."}]


def build_payload(args, messages, gen, image_b64=None, ignore_eos=False):
    content = messages
    if image_b64:
        # multimodal: image part first, then the text
        text = messages[0]["content"]
        content = [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + image_b64}},
            {"type": "text", "text": text},
        ]
    return {
        "model": args.model,
        "messages": [{"role": "user",
                      "content": (content if image_b64 else
                                  messages[0]["content"])}],
        "max_tokens": gen,
        "stream": True,
        "temperature": 0.0,  # greedy + spec-dec consistent (runbook arm B/C)
        # token-exact history capture (WARM-START dump round-trip): the
        # server echoes prompt_token_ids in the first chunk and per-chunk
        # delta token_ids (vLLM return_token_ids, v0.30.0) — byte-exact
        # ids without a tokenizer on the driver side.
        **({"return_token_ids": True} if getattr(
            args, "dump_histories", None) else {}),
        **({"ignore_eos": True} if ignore_eos else {}),
    }


# ---------------------------------------------------------------- SSE client

def stream_one(url, payload, timeout):
    """One SSE request. Returns dict(ttft, decode_tps, ntokens, t_total) —
    decode tok/s uses the generation window (after TTFT), excluding prefill.
    When payload has return_token_ids, also returns prompt_ids/token_ids
    (token-exact history for the WARM-START dump)."""
    want_ids = bool(payload.get("return_token_ids"))
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    tokens = 0
    last_t = None
    prompt_ids = None
    out_ids = []
    buf = b""
    open_resp = None
    try:
        open_resp = urllib.request.urlopen(req, timeout=timeout)
        # Iterate LINES, not fixed-size chunks: through the pl-ai.net gateway
        # the SSE stream flushes per-event and a chunked read() can coalesce
        # the whole response into one piece, collapsing the generation window
        # to ~0 andproducing garbage tok/s. raw iteration yields as data arrives.
        for raw in open_resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            delta = None
            choices = chunk.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if want_ids:
                    ch = choices[0]
                    if prompt_ids is None:
                        prompt_ids = chunk.get("prompt_token_ids")
                        if prompt_ids is None and ch.get("token_ids") is None:
                            pass
                    ch_ids = ch.get("token_ids")
                    if ch_ids is not None:
                        out_ids.extend(int(t) for t in ch_ids)
            if not delta:
                try:
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens") and tokens == 0:
                        tokens = usage["completion_tokens"]
                        if ttft is None:
                            ttft = time.monotonic() - t0
                except Exception:
                    pass
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - t0
            tokens += 1
            last_t = now
    finally:
        if open_resp is not None:
            try:
                open_resp.close()
            except Exception:
                pass
    t_total = time.monotonic() - t0
    # count content tokens: we incremented per content delta; if the server
    # only reported usage, tokens came from there (fallback below).
    # NOTE: last_t/ttft math — last_t is an ABSOLUTE monotonic timestamp,
    # ttft is RELATIVE to t0. The generation window is (last_t - t0) - ttft,
    # i.e. last-arrival minus first-arrival, both relative to t0.
    gen_window = ((last_t - t0) - ttft) if (ttft is not None and last_t and
                                            (last_t - t0) > ttft) else None
    return {
        "ttft": ttft,
        "decode_tps": (tokens / gen_window) if (gen_window and tokens) else None,
        "n_tokens": tokens,
        "t_total": t_total,
        # token-exact history (present only when return_token_ids requested)
        "prompt_ids": prompt_ids,
        "token_ids": out_ids,
    }


def post_nonstream(url, payload, timeout):
    """Non-stream POST used for the multimodal parity probe (mm_probe_ok).

    The pl-ai.net gateway streams every completion response, even without
    "stream": true in the body — so a plain json.loads(r.read()) dies with
    "Expecting value". Parse tolerantly: try JSON first, fall back to SSE
    chunk accumulation."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    # plain JSON object?
    raw_l = raw.lstrip()
    if raw_l.startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if data is not None:
            choices = data.get("choices") or []
            ok = bool(choices and (choices[0].get("message") or {}).get("content"))
            return {"ok": ok, "status": r.status}
    # SSE fallback: accumulate content deltas
    text = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk_data = line[5:].strip()
        if chunk_data == "[DONE]":
            continue
        try:
            chunk = json.loads(chunk_data)
        except ValueError:
            continue
        choices = chunk.get("choices") or []
        if choices:
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                text.append(delta)
    ok = bool("".join(text).strip())
    return {"ok": ok, "status": r.status}


SPEC_COUNTER_RE = {
    name: re.compile(
        r"^" + re.escape(name) + r"\{[^}]*\}\s+([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    )
    for name in (
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
}


def parse_spec_counters(metrics_text):
    """Extract the three spec-decode counters from /metrics text.

    Tolerant: matches bare OR 'vllm:'-prefixed spellings and returns
    {name: float} for whatever is present (missing -> absent).
    """
    out = {}
    if not metrics_text:
        return out
    for name, rx in SPEC_COUNTER_RE.items():
        m = rx.search(metrics_text)
        if m:
            out[name] = float(m.group(1))
        else:
            bare = name.split(":", 1)[1]
            m = re.search(
                r"^" + re.escape(bare) + r"\{[^}]*\}\s+([0-9.eE+-]+)\s*$",
                metrics_text, re.MULTILINE)
            if m:
                out[name] = float(m.group(1))
    return out


def spec_metrics_delta(before_text, after_text):
    """spec_metrics dict from PRE/POST /metrics texts (fail soft -> nulls)."""
    b = parse_spec_counters(before_text)
    a = parse_spec_counters(after_text)
    nd = a.get("vllm:spec_decode_num_drafts_total")
    dtok = a.get("vllm:spec_decode_num_draft_tokens_total")
    atok = a.get("vllm:spec_decode_num_accepted_tokens_total")
    if nd is not None:
        nd = nd - b.get("vllm:spec_decode_num_drafts_total", 0.0)
    if dtok is not None:
        dtok = dtok - b.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    if atok is not None:
        atok = atok - b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return {
        "num_drafts": nd,
        "draft_tokens": dtok,
        "accepted_tokens": atok,
        "accept_ratio": (atok / dtok) if (atok is not None
                                          and dtok) else None,
        "avg_draft_len": (dtok / nd) if (dtok is not None
                                         and nd) else None,
    }


def scrape_metrics(url, timeout):
    """Tolerant GET of a /metrics text page (404/no-server -> None)."""
    try:
        with urllib.request.urlopen(url, timeout=min(timeout, 15)) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


# ---------------------------------------------------------------- warm-start

def hist_file_sequences(path):
    """Read a JSONL history file -> list of lists of token ids.

    One JSON object per line, each one of:
      {"tokens": [ids...]}                    (canonical, our dump format)
      [ids...]                                (bare array)
      {"prompt": [ids...], "completion": [ids...]}  (concatenated)
    The canonical source is --dump-histories OUT (this driver), whose rows
    are exactly a finished request's full prompt+completion token history —
    the round-trip that mirrors the offline replay's warm regime.
    """
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: not valid JSON: {exc}") from exc
            if isinstance(obj, list):
                sequences.append([int(t) for t in obj])
            elif isinstance(obj, dict):
                if isinstance(obj.get("tokens"), list):
                    sequences.append([int(t) for t in obj["tokens"]])
                elif isinstance(obj.get("prompt"), list):
                    seq = list(obj["prompt"])
                    if isinstance(obj.get("completion"), list):
                        seq += list(obj["completion"])
                    sequences.append([int(t) for t in seq])
                else:
                    raise ValueError(
                        f"{path}:{lineno}: object needs 'tokens' or "
                        "'prompt'/'completion'")
            else:
                raise ValueError(f"{path}:{lineno}: unsupported row type")
    return sequences


def warm_start_http(base_url, api_path, sequences, timeout,
                    model=None, fail_hard=False):
    """POST histories to the env-gated debug endpoint; fail loud.

    The pool's EPP gateway body-validates every POST against its completions
    validator before forwarding: a bare {"sequences": ...} body is REJECTED
    ("must have prompt field"). A body carrying "model" (the pool's served
    model) and a "prompt" carrier string ALONGSIDE "sequences" passes and is
    forwarded with the original path intact — the route handler reads only
    "sequences" and ignores the extras.

    Returns the server's {"ingested": N}, or None (WARNING) when the
    endpoint 404s and fail_hard is False (SUFFIX_HYBRID_WARMSTART unset or
    VLLM_PLUGINS not naming the plugin server-side — the leg degrades to a
    cold run, which is what W1's f_draft comparison needs to see). With
    fail_hard=True a 404 FAILS the driver (non-zero exit + clear message):
    a missing route must never be mistaken for warm success.
    """
    url = base_url.rstrip("/") + "/debug/warm-start"
    body_dict = {"sequences": sequences,
                 "prompt": "warmstart-carrier"}
    if model:
        body_dict["model"] = model
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if fail_hard and exc.code == 404:
            raise SystemExit(
                f"WARM-START FAILED: server replied 404 for {url} — the "
                f"/debug/warm-start route is NOT registered. Check that the "
                f"pod env has VLLM_PLUGINS naming 'suffix_hybrid_warmstart' "
                f"AND SUFFIX_HYBRID_WARMSTART=1 (see the "
                f"'registered (endpoint_plugins)' banner in the pod log).")
        print(f"WARM-START: server replied {exc.code} for {url} "
              f"(endpoint gated by SUFFIX_HYBRID_WARMSTART=1 on the pod?)")
        return None
    except Exception as exc:
        print(f"WARM-START: {url} unreachable ({exc}) — leg runs cold")
        return None


# ---------------------------------------------------------------- stats utils

def spread(vals):
    """Stats over a list of floats (None-filtered). NOTE: *_min/*_max are the
    extremes over the values given to this call — for run rows that is the
    per-request population of ONE run (confound visibility, NOT a run spread);
    cross-run cell spread lives in the cell_summary row's runs array."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None,
                "stdev": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": (statistics.stdev(vals) if len(vals) >= 2 else 0.0),
        "n": len(vals),
    }


def pstats(prefix, d):
    if d.get("mean") is None:
        return {}
    return {
        prefix + "_mean": d["mean"], prefix + "_median": d["median"],
        prefix + "_min": d["min"], prefix + "_max": d["max"],
        prefix + "_stddev": d["stdev"],
    }


def target_chars(args):
    # ~100k tokens at ~4 chars/token heuristic (long mode)
    return 400_000 if args.prompt_len == "long" else 1_200


# ---------------------------------------------------------------- main bench

def run_stage(args, url, conc, run_idx, mm_probe):
    """One run at one concurrency. Returns per-request results + mm_probe_ok."""
    if args.corpus == "reuse":
        messages = [chat_prompt(0, i, unique=False) for i in range(conc)]
    else:
        messages = [chat_prompt(run_idx, i, unique=True) for i in range(conc)]
    if args.prompt_len == "long":
        target = target_chars(args)
        n_rep, msgs = long_prompt_messages(target)
        messages = msgs * conc
        print(f"  long prefill: {n_rep} filler reps, "
              f"~{len(msgs[0]['content'])} chars (~100k tokens est)")
    gen = args.gen_tokens if args.gen_tokens else (
        8192 if args.prompt_len == "long" else 256)
    payloads = [build_payload(args, m, gen, ignore_eos=getattr(args, "ignore_eos", False)) for m in messages]
    results = [None] * conc
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            results[i] = stream_one(url, payloads[i], args.timeout)
        except Exception as exc:
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    if errors:
        raise RuntimeError("; ".join(errors[:3]))
    mm_ok = None
    if mm_probe:
        try:
            mm_payload = build_payload(args, messages[0], 16,
                                       image_b64=make_png_b64(),
                                       ignore_eos=getattr(args, "ignore_eos", False))
            mm_ok = post_nonstream(url, mm_payload, args.timeout)["ok"]
        except Exception as exc:
            print(f"  mm probe failed: {exc}")
            mm_ok = False
    _content = messages[0][0]["content"] if (messages and messages[0]) else ""
    pchars = len(str(_content))
    return results, wall, mm_ok, pchars


def dump_histories(path, args, results, conc, run_idx):
    """Append this run's finished-request histories as JSONL.

    One row per finished request: {"tokens": prompt+completion} — the
    exact full token history the SUFFIX-ONLY cache ingests at departure,
    so a later --warm-start leg re-feeds byte-identical history rows.
    """
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            if not r:
                continue
            if r.get("prompt_ids") is None and not r.get("token_ids"):
                continue   # server did not honor return_token_ids
            hist = list(r.get("prompt_ids") or []) + list(r.get("token_ids") or [])
            if not hist:
                continue
            f.write(json.dumps({
                "tokens": [int(t) for t in hist],
                "run": run_idx, "corpus": args.corpus,
            }, sort_keys=True) + "\n")
            n += 1
    return n


def warmup(args, url):
    payload = build_payload(args, [{"role": "user", "content": "warmup ping"}], 8,
                            ignore_eos=getattr(args, "ignore_eos", False))
    try:
        stream_one(url, payload, args.timeout)
    except Exception as exc:
        print(f"warmup failed (continuing): {exc}")


def append_ledger(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="gemma-spec-dev golden-rule bench driver")
    ap.add_argument("--base-url", default=DEFAULTS["base_url"])
    ap.add_argument("--api-path", default=DEFAULTS["api_path"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32",
                    help="comma list of concurrency stages")
    ap.add_argument("--corpus", choices=["unique", "reuse"], default="unique",
                    help="unique = per-run distinct prompts (spec-dec worst "
                         "case); reuse = deterministic seed set (suffix-cache "
                         "friendly)")
    ap.add_argument("--prompt-len", choices=["chat", "long"], default="chat",
                    help="long = ~100k-token prefill estimate + 8192 gen")
    ap.add_argument("--gen-tokens", type=int, default=None,
                    help="default 256 chat / 8192 long")
    ap.add_argument("--runs", type=int, default=3,
                    help="runs per stage (n>=3 for ledger numbers)")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="depth-pinned: send ignore_eos:true (native-depth-anomaly audit)")
    ap.add_argument("--mm-probe", action=argparse.BooleanOptionalAction,
                    default=True, help="multimodal parity probe per stage")
    ap.add_argument("--mm-image", default=None,
                    help="path to a PNG to use instead of the synthesized one")
    ap.add_argument("--metrics-url", default=None,
                    help="optional /metrics scrape before/after each stage "
                         "(tolerant to 404)")
    ap.add_argument("--warm-start", default=None, metavar="FILE",
                    help="JSONL corpus histories to POST to the pod's "
                         "env-gated POST /debug/warm-start BEFORE the "
                         "measurement legs (one JSON row per sequence; "
                         "produce with --dump-histories). Requires "
                         "SUFFIX_HYBRID_WARMSTART=1 on the pool")
    ap.add_argument("--dump-histories", default=None, metavar="OUT",
                    help="run the legs normally and append each finished "
                         "request's full token history (prompt+completion "
                         "ids, via return_token_ids) as JSONL — the "
                         "round-trip corpus for a later --warm-start leg")
    ap.add_argument("--ledger", default=DEFAULTS["ledger"],
                    help="JSONL ledger (relative to repo root; /dev/null to "
                         "skip dry-run row)")
    ap.add_argument("--arm", default="unlabeled", help="arm name for ledger")
    ap.add_argument("--meta", action="append", default=[],
                    metavar="KEY=VALUE", help="repeatable metadata pairs")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate args, build all payloads, write ONE "
                         "dry_run ledger row (policy: row IS written; use "
                         "--ledger /dev/null to skip), print plan; pool NOT "
                         "required")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-request timeout seconds (long mode needs more)")
    args = ap.parse_args(argv)

    # -------- validate
    concs = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    if not concs or min(concs) < 1:
        ap.error("--concurrencies must be a comma list of positive ints")
    if args.runs < 1:
        ap.error("--runs must be >= 1")
    meta = {}
    for kv in args.meta:
        if "=" not in kv:
            ap.error(f"--meta expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        meta[k] = v
    if args.mm_image:
        with open(args.mm_image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    else:
        image_b64 = make_png_b64()
    gen = args.gen_tokens if args.gen_tokens else (
        8192 if args.prompt_len == "long" else 256)
    url = args.base_url.rstrip("/") + args.api_path

    # -------- dry-run plan (never contacts the pool for the plan itself)
    if args.dry_run:
        print("=== DRY RUN PLAN ===")
        print(f"arm={args.arm} baseUrl={url} model={args.model}")
        print(f"corpus={args.corpus} prompt_len={args.prompt_len} "
              f"gen_tokens={gen} runs={args.runs}")
        print(f"stages={concs}")
        plan_rows = []
        for c in concs:
            for r in range(args.runs):
                plan_rows.append((c, r))
        print(f"planned runs: {len(plan_rows)} "
              f"(combinations above), 1 warmup request per stage")
        msgs = [chat_prompt(0, i, unique=False) for i in range(max(concs))]
        sample_payload = build_payload(args, msgs[0], gen,
                                          ignore_eos=getattr(args, "ignore_eos", False))
        print(f"sample chat payload keys: {sorted(sample_payload)}")
        n_rep, lp = long_prompt_messages(target_chars(args))
        print(f"long prompt: {n_rep} filler reps, {len(lp[0]['content'])} chars")
        mm_payload = build_payload(args, msgs[0], 16, image_b64=image_b64,
                                      ignore_eos=getattr(args, "ignore_eos", False))
        print(f"mm probe payload: image_url data URI "
              f"({len(image_b64)} b64 chars), stream=false")
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "arm": args.arm, "pod": meta.get("pod"),
            "concurrency": concs, "corpus": args.corpus,
            "prompt_len": args.prompt_len, "gen_tokens": gen,
            "request_count": 0, "model": args.model,
            "base_url": args.base_url, "meta": meta, "dry_run": True,
            "note": "DRY RUN: args validated, all payload shapes built "
                    "(chat, long, mm image); pool not contacted.",
        }
        try:
            append_ledger(args.ledger, row)
        except OSError as exc:
            print(f"ledger write skipped ({exc}); pass --ledger /tmp/…")
        else:
            if args.ledger != "/dev/null":
                print(f"dry-run ledger row appended: {args.ledger}")
        return 0

    # -------- live
    print(f"target: {url}")
    warm_meta = {}
    if getattr(args, "warm_start", None):
        try:
            sequences = hist_file_sequences(args.warm_start)
        except (OSError, ValueError) as exc:
            ap.error(f"--warm-start: {exc}")
        if not sequences:
            ap.error(f"--warm-start: {args.warm_start} contains no sequences")
        print(f"WARM-START: posting {len(sequences)} histories to "
              f"{args.base_url.rstrip('/')}/debug/warm-start")
        # fail_hard: --warm-start was REQUESTED, so a 404 (route not
        # registered) must fail the run loudly, never read as warm success.
        resp = warm_start_http(args.base_url, args.api_path, sequences,
                               args.timeout, model=args.model,
                               fail_hard=True)
        if resp is None:
            print("WARM-START: proceeding COLD (see warning above) — the "
                  "run's ledger rows carry warm_started=false")
        else:
            ing = int(resp.get("ingested", -1))
            print(f"WARM-START: server ingested {ing} sequences")
            warm_meta = {"warm_started": True, "warm_sequences": len(sequences),
                         "warm_ingested": ing}
    meta.update(warm_meta)
    for c in concs:
        print(f"-- concurrency {c}: warmup + {args.runs} runs")
        warmup(args, url)
        m_prev = scrape_metrics(args.metrics_url, args.timeout) \
            if args.metrics_url else None
        per_run = []
        for r in range(args.runs):
            results, wall, mm_ok, pchars = run_stage(
                args, url, c, r, mm_probe=args.mm_probe)
            if getattr(args, "dump_histories", None):
                ndump = dump_histories(args.dump_histories, args, results,
                                       c, r)
                print(f"   histories dumped: {ndump} rows -> "
                      f"{args.dump_histories}")
            m_after = scrape_metrics(args.metrics_url, args.timeout) \
                if args.metrics_url else None
            tok_s_list = [x["decode_tps"] for x in results if x]
            ttft_list = [x["ttft"] for x in results if x]
            ntok_list = [x["n_tokens"] for x in results if x]
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "arm": args.arm, "pod": meta.get("pod"),
                "concurrency": c, "corpus": args.corpus,
                "prompt_len": args.prompt_len, "gen_tokens": gen,
                "prompt_chars": pchars, "run_index": r,
                "request_count": len(results),
                "ttft": pstats("ttft", spread(ttft_list)),
                "decode_tps": pstats("decode_tps", spread(tok_s_list)),
                "gen_tokens_actual": {
                    "sum": (sum(ntok_list) if ntok_list else None),
                    "min": (min(ntok_list) if ntok_list else None),
                    "max": (max(ntok_list) if ntok_list else None),
                },
                "stage_wall_s": wall, "model": args.model,
                "base_url": args.base_url, "meta": meta,
                "mm_probe_ok": mm_ok, "dry_run": False,
            }
            # NOTE: a run row's spread() is over per-request values of ONE run.
            # Cross-run aggregation is done via the cell_summary row appended
            # after all runs of the cell: it carries each run's means in
            # `runs`, so mean-hides-spike over 3-run cells is impossible.
            if args.metrics_url:
                row["spec_metrics"] = spec_metrics_delta(m_prev, m_after)
                m_prev = m_after
            append_ledger(args.ledger, row)
            per_run.append(row)
            agg = spread(tok_s_list)
            print(f"   run {r + 1}: tok/s mean="
                  f"{agg['mean']:.1f} median={agg['median']:.1f} "
                  f"(min {agg['min']:.1f} "
                  f"max {agg['max']:.1f} std {agg['stdev']:.1f}) "
                  f"ttft mean={spread(ttft_list)['mean']:.3f}s "
                  f"mm_probe_ok={mm_ok}")
        if per_run:
            runs = [{
                "run_index": row["run_index"],
                "decode_tps_mean": row["decode_tps"].get("decode_tps_mean"),
                "ttft_mean": row["ttft"].get("ttft_mean"),
                "gen_tokens_sum": row["gen_tokens_actual"]["sum"],
            } for row in per_run]
            cell_means = [r["decode_tps_mean"] for r in runs
                          if r["decode_tps_mean"] is not None]
            cell_summary = {
                "row_type": "cell_summary",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "arm": args.arm, "pod": meta.get("pod"),
                "concurrency": c, "corpus": args.corpus,
                "prompt_len": args.prompt_len, "gen_tokens": gen,
                "runs": runs,
                "decode_tps_mean_of_runs": (statistics.fmean(cell_means)
                                            if cell_means else None),
                "decode_tps_median_of_runs": (statistics.median(cell_means)
                                              if cell_means else None),
                "model": args.model, "meta": meta, "dry_run": False,
            }
            append_ledger(args.ledger, cell_summary)
        if args.metrics_url:
            note = "scraped" if m_prev else "unavailable (404?)"
            print(f"   metrics: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())