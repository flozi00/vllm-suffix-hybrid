# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 SFT bring-up on ONE card: the dsv41_slice engine (one rank's dims, full depth, dummy
weights, prod serving flags) with SUFFIX_SFT=1, driven through the plugin's own ``suffix_sft`` RPCs
exactly like /v1/sft/train does: install, served-vs-trained parity, gradient reach, a memorization
run, commit, served NLL after. stdout is ONE JSON object; logs go to stderr.

    python3 -m suffix_hybrid.tools.dsv41_sft --opts '{"slice": {"k": 0}, "sft": {"steps": 8}}'

opts.slice = dsv41_slice opts (validated by its schema; default here: k=0, conc [1]); opts.sft below.
Exit codes: 0 ok, 2 bad opts, 3 engine refused, 5 an SFT step failed, 7 environment, 1 unexpected.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

from suffix_hybrid.tools import dsv41_slice as S

SFT_SCHEMA = {   # name: (default, lo, hi)
    "rank": (16, 1, 256), "seq_len": (512, 16, 65536), "examples": (2, 1, 64), "steps": (8, 0, 1000),
    "lr": (1e-3, 1e-7, 1.0), "offload_gib": (2.0, 0.0, 64.0), "min_free_gib": (1.0, 0.0, 64.0),
    "parity_window": (1024, 16, 65536), "mem_trace": (False, None, None), "host_ckpt": (False, None, None),
    "targets": ("", None, None), "lora_plus": (1.0, 1.0, 64.0), "diag": (False, None, None),
    "profile": (False, None, None), "timing": (False, None, None),
}
SLICE_DEFAULTS = {"k": 0, "conc": [1], "gpu_memory_utilization": 0.8, "max_model_len": 65536}


def log(msg):
    print(f"[dsv41-sft] {msg}", file=sys.stderr, flush=True)


def parse(raw):
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - {"slice", "sft"}:
        raise S.SliceError("opts: an object with keys 'slice' and/or 'sft'", 2)
    o = S.parse_opts({**SLICE_DEFAULTS, **(raw.get("slice") or {})})
    sft = {}
    for k, v in (raw.get("sft") or {}).items():
        if k not in SFT_SCHEMA:
            raise S.SliceError(f"opts.sft: unknown key {k}; valid {sorted(SFT_SCHEMA)}", 2)
        d, lo, hi = SFT_SCHEMA[k]
        if type(v) is not type(d) and not (isinstance(d, float) and isinstance(v, int) and not isinstance(v, bool)):
            raise S.SliceError(f"opts.sft.{k}: {type(d).__name__} expected", 2)
        if lo is not None and not lo <= v <= hi:
            raise S.SliceError(f"opts.sft.{k}: in [{lo}, {hi}]", 2)
        if k == "targets" and not all(c.isalnum() or c in "._*,?" for c in v):
            raise S.SliceError("opts.sft.targets: glob characters only", 2)
        sft[k] = v
    return o, {k: sft.get(k, d) for k, (d, _, _) in SFT_SCHEMA.items()}


def sft_env(s):
    env = {"SUFFIX_SFT": "1", "SUFFIX_SFT_RANK": str(s["rank"]), "SUFFIX_SFT_OFFLOAD_GIB": str(s["offload_gib"]),
           "SUFFIX_SFT_MIN_FREE_GIB": str(s["min_free_gib"]), "SUFFIX_SFT_MAX_TOKENS": str(max(4096, s["seq_len"])),
           "SUFFIX_SFT_MEM_TRACE": str(int(s["mem_trace"])), "SUFFIX_SFT_HOST_CKPT": str(int(s["host_ckpt"])),
           "SUFFIX_SFT_LORA_PLUS": str(s["lora_plus"])}
    if s["targets"]:
        env["SUFFIX_SFT_TARGETS"] = s["targets"]
    if s["diag"]:
        env["SUFFIX_SFT_DIAG"] = "1"
    return env


def _instrument():
    """Synchronized wall timers (inclusive, seconds) around the twin's pieces and the frozen-op
    autograd Functions; in-process engine, so the worker runs these very module objects."""
    import collections
    import functools

    import torch
    from suffix_hybrid.mtp_tune import train as T
    from suffix_hybrid.sft import dsv41 as D
    times = collections.defaultdict(float)

    def wrap(fn, key):
        @functools.wraps(fn)
        def inner(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                torch.cuda.synchronize()
                times[key] += time.perf_counter() - t
        return inner
    for name in ("attention", "moe", "_mhc_pre", "_mhc_post", "engram", "engram_kv", "_select", "_compress",
                 "_attend", "_index_k"):
        setattr(D, name, wrap(getattr(D, name), name))
    for cls in (T.FrozenExperts, T.FrozenLinear):
        for m in ("forward", "backward"):
            setattr(cls, m, staticmethod(wrap(getattr(cls, m), f"{cls.__name__}.{m}")))
    return times


def run(o, s, doc):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    plan = S.build_plan(o, doc["model_dir"])
    if o["shared_expert_tp"]:
        S.apply_shared_expert_patch()
    t0 = time.time()
    try:
        llm = LLM(**plan["engine_kwargs"])
    except BaseException as exc:  # noqa: BLE001
        doc["engine_error"] = f"{type(exc).__name__}: {exc}"
        doc["traceback"] = traceback.format_exc()[-6000:]
        return 3
    doc["boot_s"] = round(time.time() - t0, 1)
    free, total = torch.cuda.mem_get_info()
    doc["after_boot_free_gib"] = round(free / 2**30, 2)

    def rpc(op, payload=None):
        r = llm.collective_rpc("suffix_sft", args=(op, payload or {}))[0]
        if isinstance(r, dict) and r.get("error"):
            raise RuntimeError(f"{op}: {r['error']}")
        return r

    def served(ids, mask):
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
        lp = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].prompt_logprobs
        return [-lp[i][ids[i]].logprob for i in range(1, len(ids)) if mask[i]]

    def parity(ids, mask, tag):
        w = s["parity_window"]
        ids, mask = (ids[:w], mask[:w]) if any(mask[1:w]) else (ids[-w:], mask[-w:])
        a = served(ids, mask)
        r = rpc("eval", dict(ids=ids, mask=mask, seg_lens=[len(ids)], per_token=True))
        b = r["nll"]
        d = [x - y for x, y in zip(a, b)]
        out = dict(served=round(sum(a) / len(a), 5), trained=round(sum(b) / len(b), 5),
                   delta=round(abs(sum(d) / len(d)), 5), tok_max=round(max(abs(x) for x in d), 4), n=len(d))
        log(f"parity {tag}: {out}")
        return out

    doc["info"] = rpc("info")
    vocab = json.load(open(os.path.join(doc["model_dir"], "config.json")))["text_config"]["vocab_size"]
    rng = random.Random(o["seed"])
    data = []
    for _ in range(s["examples"]):
        ids = [rng.randrange(3, vocab) for _ in range(s["seq_len"])]
        mask = [0] * (s["seq_len"] // 2) + [1] * (s["seq_len"] - s["seq_len"] // 2)
        data.append((ids, mask))
    doc["parity_before"] = parity(*data[0], "before")
    if s["diag"]:   # sensitivity: does the parity see each component? (corr of per-token NLL patterns)
        ids, mask = data[0][0][:s["parity_window"]], data[0][1][:s["parity_window"]]
        a = served(ids, mask)
        variants = {"base": {}, "no_attn": {"dsv4_no_attn": 1}, "no_moe": {"dsv4_no_moe": 1},
                    "no_engram": {"dsv4_no_engram": 1}, "no_comp": {"dsv4_no_comp": 1},
                    "no_kvqdq": {"dsv4_no_kvqdq": 1}}
        r = rpc("diag", dict(ids=ids, mask=mask, seg_lens=[len(ids)], variants=variants))

        def corr(x, y):
            mx, my = sum(x) / len(x), sum(y) / len(y)
            cx = [v - mx for v in x]
            cy = [v - my for v in y]
            den = (sum(v * v for v in cx) * sum(v * v for v in cy)) ** 0.5
            return sum(p * q for p, q in zip(cx, cy)) / den if den else float("nan")
        sd = (sum((v - sum(a) / len(a)) ** 2 for v in a) / len(a)) ** 0.5
        doc["diag"] = {"served_sd": round(sd, 6), **{k: dict(
            corr=round(corr(a, v), 6), mean_delta=round(abs(sum(a) - sum(v)) / len(a), 6),
            max_delta=round(max(abs(p - q) for p, q in zip(a, v)), 5)) for k, v in r["nll"].items()}}
        log(f"diag {doc['diag']}")
    if s["steps"]:
        rpc("begin", dict(lr=s["lr"]))
        doc["steps"] = []
        for step in range(s["steps"]):
            norm = sum(sum(m[1:]) for _, m in data)
            loss, micro = 0.0, []
            for i, (ids, mask) in enumerate(data):
                r = rpc("micro", dict(ids=ids, mask=mask, seg_lens=[len(ids)], norm=norm,
                                      check=(step == 0 and i == 0)))
                loss += r["loss_sum"]
                micro.append({k: r.get(k) for k in ("ms", "fwd_ms", "bwd_ms", "peak_gib", "fwd_peak_gib",
                                                     "free_gib", "host_pinned_gib", "tokens")})
            if s["timing"] and step == 1:
                times = _instrument()
                t0 = time.perf_counter()
                rpc("micro", dict(ids=data[0][0], mask=data[0][1], seg_lens=[len(data[0][0])], norm=1e9))
                times["total"] = time.perf_counter() - t0
                doc["timing_s"] = {k: round(v, 3) for k, v in sorted(times.items(), key=lambda kv: -kv[1])}
                log(f"timing {doc['timing_s']}")
            g = rpc("step")
            row = dict(step=step, loss=round(loss / norm, 5), grad_norm=round(g["grad_norm"], 5),
                       step_ms=g["ms"], micro=micro)
            log(f"step {row}")
            doc["steps"].append(row)
        rpc("commit")
        llm.reset_prefix_cache()
        doc["parity_after"] = parity(*data[0], "after")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dsv41_sft")
    ap.add_argument("--opts", default="{}")
    args = ap.parse_args(argv)
    try:
        o, s = parse(json.loads(args.opts))
    except (S.SliceError, json.JSONDecodeError) as exc:
        print(f"dsv41-sft: {exc}", file=sys.stderr)
        return 2
    env = S.env_plan(o)
    env.update(sft_env(s))
    S.maybe_reexec(env)
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)   # library prints go to stderr; stdout is the JSON
    cfg, _ = S.slice_config(o)
    model_dir, _ = S.write_model_dir(o["work_dir"], cfg)
    doc = {"schema": "dsv41-sft/1", "opts": {"slice": o, "sft": s}, "model_dir": model_dir}
    try:
        code = run(o, s, doc)
    except Exception as exc:  # noqa: BLE001
        doc["error"] = f"{type(exc).__name__}: {exc}"
        doc["traceback"] = traceback.format_exc()[-6000:]
        code = 5
    doc.update(ok=code == 0, exit=code)
    out.write(json.dumps(doc, default=str) + "\n")
    out.flush()
    return code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
