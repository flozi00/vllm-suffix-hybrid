#!/usr/bin/env python3
"""Greedy-decode quality probe: fixed prompts, temperature=0.

Speculative decoding (MTP) is lossless by construction — verified tokens
follow the target model's greedy distribution exactly. Therefore outputs
MUST be byte-identical across kernel/spec configs. Any divergence signals a
real numerics/quality regression (e.g. a bad kernel backend).

Usage:
  python3 bench/quality_probe.py --endpoint URL --model M --label TAG
Writes bench/quality/<label>.json and, when a reference exists, diffs
greedy outputs against bench/quality/reference.json.
"""
import argparse, hashlib, json, os, ssl, sys, time, urllib.request

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number using memoization. Only code.",
    "A train leaves Munich at 10:00 going 120 km/h. Another leaves Nuremberg 30 minutes later going 160 km/h. The distance is 170 km. When do they meet? Show your reasoning.",
    "Translate to German: 'The quick brown fox jumps over the lazy dog.' Then explain the grammar case of each noun.",
    "List the first 12 prime numbers, then compute their sum. Verify your sum by adding in pairs.",
    "Refactor this function to remove the nested loop: for i in range(n):\n  for j in range(n):\n    if a[i]+a[j]==t: return (i,j)",
    "What is 17 * 23? Then divide the result by 7 and round to two decimals.",
    "Summarize the plot of a story about a lighthouse keeper who finds a message in a bottle, in exactly three bullet points.",
]

CTX = ssl._create_unverified_context()

def chat(endpoint, model, prompt, max_tokens=256):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
        d = json.load(r)
    ch = d["choices"][0]
    text = ch["message"]["content"]
    if text is None:
        # reasoning-only truncation: content=None when max_tokens cut off before answer
        text = "<NONE finish=%s>" % ch.get("finish_reason")
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality")
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        text = chat(args.endpoint, args.model, p)
        key = f"p{i}"
        results[key] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "chars": len(text),
            "secs": round(time.time() - t0, 2),
            "text": text,
        }
        r = results[key]
        print(f"{key}: {r['sha256']} ({r['chars']} chars, {r['secs']}s)", flush=True)

    path = os.path.join(outdir, f"{args.label}.json")
    json.dump(results, open(path, "w"), indent=2)

    ref_path = os.path.join(outdir, "reference.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        diffs = [k for k in results if ref.get(k, {}).get("sha256") != results[k]["sha256"]]
        if diffs:
            print(f"QUALITY-DIVERGENCE vs reference: {diffs}")
            for k in diffs:
                print(f"--- {k} reference:\n{ref[k]['text'][:400]}")
                print(f"--- {k} {args.label}:\n{results[k]['text'][:400]}")
            sys.exit(2)
        print(f"QUALITY-IDENTICAL to reference ({len(results)} prompts)")
    else:
        import shutil
        shutil.copy(path, ref_path)
        print(f"stored as reference ({len(results)} prompts)")

if __name__ == "__main__":
    main()
