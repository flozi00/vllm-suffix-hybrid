#!/usr/bin/env python3
"""qgate.py — owner quality gate for one arm (stdlib only).

  GSM8K   first --gsm N test questions (greedy, chat, 'answer after ####'),
          accuracy = last number in the reply == reference.
  tools   5 tool-call prompts; PASS when the model emits the right function
          with parseable JSON arguments containing the expected key.
  needle  a passphrase buried at ~45k tokens of filler; must be recalled.
  image   (--image) tiny in-memory PNG, red|blue halves; reply must name both.
  junk    every reply checked for degenerate repetition (same 8-gram >= 6x).

  qgate.py --url https://x.pl-ai.net --model x --arm NAME [--gsm 200] [--image]
GSM8K data: ~/.cache/ovh/gsm8k_test.jsonl (openai/grade-school-math test.jsonl).
"""
import argparse, base64, concurrent.futures as cf, json, os, re, socket, ssl, struct, subprocess, time, urllib.request, zlib

_gai = socket.getaddrinfo
def _gai_fallback(host, *a, **kw):
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


def chat(url, model, messages, max_tokens=1024, tools=None, temperature=0.0):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, context=CTX, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]


def junk(text):
    w = (text or "").split()
    grams = {}
    for i in range(len(w) - 7):
        g = tuple(w[i:i + 8])
        grams[g] = grams.get(g, 0) + 1
    return max(grams.values(), default=0) >= 6


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", (s or "").replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def gsm(url, model, n):
    rows = [json.loads(l) for l in open(os.path.expanduser("~/.cache/ovh/gsm8k_test.jsonl"))][:n]
    def one(r):
        msg = chat(url, model, [{"role": "user", "content": r["question"] +
                   "\nSolve step by step, then give the final answer after '####'."}], 2048)
        text = msg.get("content") or ""
        ans = text.split("####")[-1] if "####" in text else text
        ref = r["answer"].split("####")[-1].strip().replace(",", "")
        got = num(ans)
        ok = got is not None and abs(float(got) - float(ref)) < 1e-6
        return ok, junk(text)
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, rows))
    return sum(o for o, _ in res) / len(res), sum(j for _, j in res)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount",
          "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "to": {"type": "string"}},
                         "required": ["amount", "to"]}}}]
TOOL_CASES = [("What's the weather in Berlin right now?", "get_weather", "city"),
              ("Wie ist das Wetter in München?", "get_weather", "city"),
              ("Convert 120 euros to US dollars.", "convert_currency", "amount"),
              ("Is it raining in Tokyo?", "get_weather", "city"),
              ("How much is 50 GBP in JPY?", "convert_currency", "to")]


def tools(url, model):
    ok = 0
    for q, fn, key in TOOL_CASES:
        try:
            m = chat(url, model, [{"role": "user", "content": q}], 1024, TOOLS)
            tc = (m.get("tool_calls") or [{}])[0].get("function", {})
            ok += tc.get("name") == fn and key in json.loads(tc.get("arguments") or "{}")
        except Exception:  # noqa: BLE001
            pass
    return ok


def needle(url, model, approx_tokens=45000):
    filler = ("The quiet river carried autumn leaves past the old mill while farmers counted "
              "their harvest and children played near the stone bridge. ")
    reps = approx_tokens // 28
    secret = "PURPLE-ORCHID-7731"
    parts = [filler] * reps
    parts.insert(reps // 2, f" Remember this passphrase: {secret}. ")
    m = chat(url, model, [{"role": "user", "content": "".join(parts) +
             "\n\nWhat is the passphrase mentioned in the text above? Reply with it only."}], 512)
    return secret in ((m.get("content") or "") + (m.get("reasoning") or m.get("reasoning_content") or ""))


def png_red_blue(w=64, h=32):
    rows = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w))
                    for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def image(url, model):
    b64 = base64.b64encode(png_red_blue()).decode()
    m = chat(url, model, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Which two colors does this image show? Answer briefly."}]}], 512)
    t = (m.get("content") or "").lower()
    return "red" in t and "blue" in t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gsm", type=int, default=200)
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--needle-tokens", type=int, default=45000)
    ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    t0 = time.time()
    acc, junky = gsm(a.url, a.model, a.gsm)
    row = {"arm": a.arm, "model": a.model, "gsm8k": round(acc, 4), "gsm_n": a.gsm, "gsm_junk": junky,
           "tools": f"{tools(a.url, a.model)}/5", "needle": needle(a.url, a.model, a.needle_tokens)}
    if a.image:
        row["image"] = image(a.url, a.model)
    row["secs"] = round(time.time() - t0)
    print(json.dumps(row), flush=True)
    if a.ledger:
        with open(a.ledger, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
