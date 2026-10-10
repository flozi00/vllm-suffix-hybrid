#!/usr/bin/env python3
"""abbench.py — owner A/B gate metrics for one arm (stdlib only).

Per concurrency c: c closed-loop streams for --seconds, unique chat prompts,
fixed --gen-tokens (ignore_eos), temperature --temperature (prod default 0.2).
Tokens counted from delta token_ids (spec decode emits several per chunk).
Prints one JSON line per cell: aggregate tok/s, per-stream decode tok/s
(median, p10), TTFT p50/max, ITL p99 (chunk gap / tokens in chunk), n requests.

  abbench.py --url https://x.pl-ai.net --model x --arm NAME [--conc 1,8,32]
"""
import argparse, json, random, socket, ssl, statistics, subprocess, sys, threading, time, urllib.request
from collections import Counter

_gai = socket.getaddrinfo
def _gai_fallback(host, *a, **kw):  # macOS negative-caches fresh dev hostnames
    try:
        return _gai(host, *a, **kw)
    except socket.gaierror:
        out = subprocess.run(["host", host], capture_output=True, text=True).stdout
        ip = next((l.split()[-1] for l in out.splitlines() if "has address" in l), None)
        if not ip:
            raise
        return _gai(ip, *a, **kw)
socket.getaddrinfo = _gai_fallback
CTX = ssl._create_unverified_context()
TOPICS = ["volcanoes", "the history of tea", "sorting algorithms", "Roman roads", "coral reefs",
          "the printing press", "black holes", "sourdough baking", "the Silk Road", "photosynthesis",
          "jazz improvisation", "glaciers", "the immune system", "medieval castles", "solar panels"]


PASSAGE = None


def one(url, model, gen, temp, seed):
    r = random.Random(seed)
    prompt = (f"[{seed}] Write a detailed essay about {r.choice(TOPICS)} and "
              f"{r.choice(TOPICS)}, with concrete examples. ({r.random():.6f})")
    if PASSAGE:  # edit-style traffic: the answer mostly copies the prompt (suffix-decoding friendly)
        prompt = (f"[{seed}] Return the following text unchanged except replace every occurrence of "
                  f"'river' with 'stream'. Output only the text.\n\n{PASSAGE}")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen, "ignore_eos": True, "stream": True,
            "temperature": temp, "return_token_ids": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    ttft = last = None
    n = 0
    gaps = []
    with urllib.request.urlopen(req, context=CTX, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ch = (json.loads(line[5:]).get("choices") or [{}])[0]
            k = len(ch.get("token_ids") or ())
            if not k:
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - t0
            elif last is not None:
                gaps += [(now - last) / k] * k
            last = now
            n += k
    dec = (n - 1) / (last - t0 - ttft) if n > 1 and last - t0 > ttft else None
    return {"ttft": ttft, "n": n, "decode": dec, "gaps": gaps, "t0": t0, "t1": last}


def cell(url, model, c, seconds, gen, temp):
    res, lock = [], threading.Lock()
    stop = time.monotonic() + seconds
    def worker(i):
        j = 0
        while time.monotonic() < stop:
            try:
                r = one(url, model, gen, temp, hash((i, j, time.time())) & 0xFFFFFFF)
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            with lock:
                res.append(r)
            j += 1
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    t_start = time.monotonic()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.monotonic() - t_start
    ok = [r for r in res if "error" not in r and r["n"]]
    errs = len(res) - len(ok)
    if errs:  # what failed: client exceptions, or responses without tokens
        why = Counter(r.get("error", "no tokens")[:160] for r in res if r not in ok)
        print(f"conc {c}: {errs} errors: {dict(why.most_common(3))}", file=sys.stderr, flush=True)
    toks = sum(r["n"] for r in ok)
    dec = sorted(r["decode"] for r in ok if r["decode"])
    ttfts = sorted(r["ttft"] for r in ok)
    gaps = sorted(g for r in ok for g in r["gaps"])
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    return {"conc": c, "requests": len(ok), "errors": errs, "agg_tps": toks / wall,
            "stream_tps_med": statistics.median(dec) if dec else None, "stream_tps_p10": q(dec, 0.10),
            "ttft_p50": q(ttfts, 0.5), "ttft_max": ttfts[-1] if ttfts else None,
            "itl_p99_ms": q(gaps, 0.99) * 1e3 if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--conc", default="1,8,32")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--gen-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--mode", choices=["essay", "edit"], default="essay")
    a = ap.parse_args()
    global PASSAGE
    if a.mode == "edit":
        PASSAGE = " ".join(f"Sentence {i}: the river near village {i % 7} carried {i * 3} boats past "
                           f"the mill while the miller counted sacks of grain." for i in range(40))
    one(a.url, a.model, 32, a.temperature, 0)  # warmup
    for c in [int(x) for x in a.conc.split(",")]:
        row = {"arm": a.arm, "model": a.model, "temperature": a.temperature,
               "gen_tokens": a.gen_tokens, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **cell(a.url, a.model, c, a.seconds, a.gen_tokens, a.temperature)}
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if a.ledger:
            with open(a.ledger, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
