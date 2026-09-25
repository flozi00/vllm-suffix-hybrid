# SPDX-License-Identifier: Apache-2.0
"""Single-GPU silicon oracle for the SM120 HiSparse+MTP patch (no GLM).

    python -m hisparse_mtp_patch.oracle [--k 3 5] [--prompt-len 4096] ...

Model: upstream tests/v1/e2e/general/test_hisparse.py recipe — DeepSeek-V3.2
config (config.json only; ``load_format=dummy`` = seeded random weights, no
weight download) shrunk via hf_overrides, MTP draft, HiSparseConnector,
FULL_AND_PIECEWISE graphs. Differences: index_topk stays 2048 (the SM120
backend rejects anything else) so prompts > 2048 tokens make top-k sparse,
and max_num_seqs=1 keeps every run's batch shapes identical.

Per k (prod GLM runs k=5), three child processes (fresh CUDA context each):
  ref     HiSparse off, MTP            (non-HiSparse spec-as-decode path)
  stock   HiSparse on, MTP, gate off   (verify tokens -> prefill staging)
  patched HiSparse on, MTP, SUFFIX_SM120_HISPARSE_MTP=1 (hot-buffer decode)
Each generates greedily: target, 4 pressure prompts (spill target's pages
to host), target again (restore from host through the hot buffer).

A child that crashes is REPORTED (its last error line) and the others still
run: a stock crash is evidence that stock HiSparse+MTP is broken (vLLM 0.30.0:
swap-row capacity off-by-one, fixed by the patch), never a reason to skip the
patched run.

PASS (per k) needs: patched completed, is patched + on
FLASHINFER_MLA_SPARSE_SM120, took multi-token decode batches during
generation (HISPARSE-MTP-DECODE counter delta > 0), spilled (> 0), patched
target before == after spill, and patched == ref token-for-token on every
prompt. patched != ref but == a completed stock is INCONCLUSIVE (exit 2:
upstream HiSparse on/off not bitwise, not a patch fault); a crashed ref is
INCONCLUSIVE too. A completed stock that reports multi-token decode = FAIL
(gate leaked). Overall exit: any FAIL -> 1, else any INCONCLUSIVE -> 2.

Caveat: dummy weights make decode near-bigram (streams fall into 2-cycles
after the first, prompt-dependent token), so token parity mostly proves the
prefill/staging path + no crash; the "distinct" counts in the summary line
say how much the decode path could have shown.
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

MARK = "[suffix sm120-hisparse-mtp] ORACLE"
RESULT = "ORACLE-RESULT "


def _prompts(n_tokens: int):
    rng = random.Random(0)
    return [[rng.randrange(100, 30000) for _ in range(n_tokens)]
            for _ in range(5)]


def child(mode: str, a) -> dict:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    def shrink(config):
        over = {"num_hidden_layers": a.layers, "hidden_size": 256,
                "intermediate_size": 512, "num_attention_heads": 8,
                "num_key_value_heads": 1, "n_routed_experts": 8,
                "num_experts_per_tok": 2, "index_topk": 2048}
        if any("MTP" in arch for arch in config.architectures):
            over.pop("num_hidden_layers")
        config.update(over)
        return config

    hisparse = mode != "ref"
    kw = {}
    if hisparse:
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="HiSparseConnector", kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": a.host_gib})
        if a.gpu_blocks:
            kw["num_gpu_blocks_override"] = a.gpu_blocks
    llm = LLM(
        a.model, load_format="dummy", seed=0, hf_overrides=shrink,
        skip_tokenizer_init=True, kv_cache_dtype=a.kv_cache_dtype,
        block_size=64, max_model_len=a.prompt_len + a.max_tokens + 64,
        max_num_seqs=1, max_num_batched_tokens=a.batched_tokens,
        enable_prefix_caching=True, enable_chunked_prefill=True,
        gpu_memory_utilization=a.gpu_util,
        speculative_config={"method": "mtp", "num_speculative_tokens": a.k},
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE",
                            "cudagraph_capture_sizes": [1, 2, 4, 8]},
        **kw)

    import vllm.model_executor.layers.attention.sparse_mla_attention as sma

    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    cfg = getattr(runner, "vllm_config", None) or llm.llm_engine.vllm_config
    impls = sorted({type(getattr(layer, "impl", None)).__name__
                    for layer in cfg.compilation_config
                    .static_forward_context.values()
                    if getattr(layer, "impl", None) is not None})
    stats = getattr(sma, "_SUFFIX_HISPARSE_MTP_STATS", None)
    base = stats["multi_token_decode_builds"] if stats else 0

    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                        ignore_eos=True)

    def gen(p):
        return list(llm.generate([TokensPrompt(prompt_token_ids=p)], sp,
                                 use_tqdm=False)[0].outputs[0].token_ids)

    prompts = _prompts(a.prompt_len)
    out = {"target_1": gen(prompts[0])}
    for i, p in enumerate(prompts[1:]):
        out[f"pressure_{i}"] = gen(p)
    out["target_2"] = gen(prompts[0])

    spills = None
    if hisparse:
        from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
        spills = get_hisparse_coordinator(
            core.scheduler.kv_cache_manager).next_spill_id
    return {
        "mode": mode, "impls": impls, "outputs": out, "spills": spills,
        "patched": getattr(sma, "__suffix_hisparse_mtp_revision__", None),
        "mtp_decode_builds": (stats["multi_token_decode_builds"] - base)
        if stats else 0,
        "max_decode_query_len": stats["max_decode_query_len"] if stats else 0,
    }


CHILD_TIMEOUT = int(os.environ.get("SUFFIX_HISPARSE_ORACLE_CHILD_TIMEOUT", "1200"))


def _run_child(mode: str, argv) -> dict:
    """Never raises on a child failure: returns {"mode", "error"} instead."""
    env = dict(os.environ)
    env.pop("SUFFIX_SM120_HISPARSE_MTP", None)
    if mode == "patched":
        env["SUFFIX_SM120_HISPARSE_MTP"] = "1"
    # Own process group: a crashed child can leave vLLM engine-core / worker
    # processes holding the GPU; the group is killed after EVERY child so the
    # next one starts clean. Output is streamed live (a hang stays visible)
    # and captured for the verdict; CHILD_TIMEOUT bounds a hung child.
    import signal
    import threading
    proc = subprocess.Popen(
        [sys.executable, "-m", "hisparse_mtp_patch.oracle", "--child", mode]
        + argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    out, err = [], []

    def pump(src, sink, buf):
        for line in src:
            buf.append(line)
            sink.write(line)
            sink.flush()
    pumps = [threading.Thread(target=pump, args=(proc.stdout, sys.stdout, out), daemon=True),
             threading.Thread(target=pump, args=(proc.stderr, sys.stderr, err), daemon=True)]
    for t in pumps:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    for t in pumps:
        t.join(timeout=10)
    for line in out:
        if line.startswith(RESULT):
            return json.loads(line[len(RESULT):])
    if timed_out:
        return {"mode": mode, "error": f"TIMEOUT after {CHILD_TIMEOUT}s (hung)"}
    errs = [ln.strip() for ln in err if "Error" in ln]
    return {"mode": mode, "error": f"exit {proc.returncode}: "
            + (errs[-1] if errs else "no result line")}


def _diff(a: dict, b: dict) -> list:
    bad = []
    for key in a:
        x, y = a[key], b.get(key)
        if x != y:
            pos = next((i for i, (p, q) in enumerate(zip(x, y or [])) if p != q),
                       min(len(x), len(y or [])))
            bad.append(f"{key}@{pos}")
    return bad


def verdict(ref: dict, stock: dict, patched: dict) -> tuple[int, list]:
    """Pure: (exit code, reasons). 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
    A crashed stock is only a note; it never blocks judging patched."""
    note = [f"stock crashed ({stock['error']})"] if "error" in stock else []
    if not note and "error" not in ref and _diff(ref["outputs"], stock["outputs"]):
        note = [f"stock != ref at {_diff(ref['outputs'], stock['outputs'])}"]
    if "error" in patched:
        return 1, [f"patched crashed ({patched['error']})"] + note
    if "error" in ref:
        return 2, [f"ref crashed ({ref['error']}): no reference"] + note
    fail = []
    if not patched["patched"]:
        fail.append("patch not active in the patched run")
    if "FlashInferMLASparseSM120Impl" not in patched["impls"]:
        fail.append(f"SM120 sparse-MLA backend not selected: {patched['impls']}")
    if patched["mtp_decode_builds"] <= 0:
        fail.append("no multi-token HiSparse decode batch during generation")
    stock_ok = "error" not in stock
    if stock_ok and stock["mtp_decode_builds"] != 0:
        fail.append("stock run reports multi-token decode (gate leaked)")
    if not patched["spills"]:
        fail.append("no HiSparse spill: host path not exercised "
                    "(lower --gpu-blocks / raise --prompt-len)")
    po = patched["outputs"]
    if po["target_1"] != po["target_2"]:
        fail.append("patched target changed after spill/restore")
    d = _diff(ref["outputs"], po)
    if d and stock_ok and not _diff(stock["outputs"], po):
        fail.append(f"INCONCLUSIVE: patched == stock but both != ref at {d} "
                    "(upstream HiSparse on/off not bitwise)")
    elif d:
        fail.append(f"patched != ref at {d}")
    if fail:
        rc = 2 if all(f.startswith("INCONCLUSIVE") for f in fail) else 1
        return rc, fail + note
    return 0, note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=("ref", "stock", "patched"))
    # Vendored DeepSeek-V3.2 config.json (public, MIT): load_format=dummy +
    # skip_tokenizer_init need nothing else, so the pod needs no HF access.
    ap.add_argument("--model", default=str(Path(__file__).with_name("deepseek_v32")))
    ap.add_argument("--k", type=int, nargs="+", default=[3],
                    help="num_speculative_tokens; several = one sweep each")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-blocks", type=int, default=0,
                    help="num_gpu_blocks_override for HiSparse runs (0=auto)")
    ap.add_argument("--host-gib", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)
    if a.child:
        a.k = a.k[0]
        print(RESULT + json.dumps(child(a.child, a)), flush=True)
        return 0
    codes = []
    for k in a.k:
        runs = {m: _run_child(m, argv + ["--k", str(k)])  # last --k wins
                for m in ("ref", "stock", "patched")}
        for m, r in runs.items():
            if "error" in r:
                print(f"{MARK} k={k} {m}: CRASHED {r['error']}", flush=True)
                continue
            distinct = sorted(len(set(v)) for v in r["outputs"].values())
            print(f"{MARK} k={k} {m}: impls={r['impls']} spills={r['spills']} "
                  f"mtp_decode_builds={r['mtp_decode_builds']} "
                  f"max_q_len={r['max_decode_query_len']} "
                  f"distinct={distinct}", flush=True)
        rc, why = verdict(runs["ref"], runs["stock"], runs["patched"])
        codes.append(rc)
        print(f"{MARK} k={k} {['PASS', 'FAIL', 'INCONCLUSIVE'][rc]}"
              + (": " + "; ".join(why) if why else ""), flush=True)
    return 1 if 1 in codes else max(codes)


if __name__ == "__main__":
    sys.exit(main())
