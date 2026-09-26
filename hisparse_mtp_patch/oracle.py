# SPDX-License-Identifier: Apache-2.0
"""Silicon oracle for the SM120 HiSparse+MTP patch (no GLM); 1 GPU, or
--tp N GPUs (N>1 = the shared /dev/shm host pool prod's TP=8 uses: every
rank cudaHostRegisters one mmap; stock crashed there, patched serializes).

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

--multi (boot gate glm_stack_multi_oracle): what prod does and
max_num_seqs=1 cannot. 8 layers with IndexShare (index_topk_freq=4,
index_skip_topk_offset=3 -> indexer layers 0,1,2,6 + MTP; followers 3,4,5,7),
index_share_for_mtp_iteration (MTP draft steps >0 reuse step 0's top-k),
max_num_seqs=4, max_num_batched_tokens from --batched-tokens (256: chunked
prefill mixed with decodes), ONE generate() of 8 prompts (multi_prompts()),
spill-forcing num_gpu_blocks_override from the real HiSparse group layout
(hisparse_hot_groups(), mirrors vllm/v1/hisparse/layout.py). Children: ref
(HiSparse off), patched (SUFFIX_SM120_HISPARSE_MTP=1), prefetch (patched +
SUFFIX_SM120_HISPARSE_PREFETCH=1). PASS needs patched == ref and prefetch ==
patched token-for-token, multi-token decode builds, spills, and the prefetch
child's leader actually prefetching follower rows (verdict_multi()).
Combines with --tp N (worker stats via collective_rpc, rank 0).

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
import time
import sys

MARK = "[suffix sm120-hisparse-mtp] ORACLE"
RESULT = "ORACLE-RESULT "


def _prompts(n_tokens: int):
    rng = random.Random(0)
    return [[rng.randrange(100, 30000) for _ in range(n_tokens)]
            for _ in range(5)]


# fp8_ds_mla / nvfp4_ds_mla bytes per token (mla_attention.py:1361); the
# DeepSeek-V3.2 indexer row is 128 fp8 + 4 B scale (deepseek_v2.py:724).
SOURCE_ROW = {"fp8_ds_mla": 656, "nvfp4_ds_mla": 352}
INDEXER_ROW = 132


def index_leaders(layers: int, freq: int = 1, offset: int = 2, mtp: int = 1) -> list:
    """Per attention layer (backbone then MTP): does it own an indexer?
    deepseek_v2.py:1136-1160 (IndexCache); MTP layers always do."""
    return [max(i - offset + 1, 0) % freq == 0 for i in range(layers)] + [True] * mtp


def hisparse_hot_groups(layers: int, freq: int = 1, offset: int = 2,
                        kv_dtype: str = "fp8_ds_mla", mtp: int = 1) -> list:
    """Layer ids per HiSparse hot group, as vllm/v1/hisparse/layout.py packs
    them: units start at index-group leaders, a unit joins the current group
    while the group's page sum stays <= the summed indexer page. KV groups
    total = 1 host source + 1 indexer + resident + hot = 2 + 2 * len(result)
    (silicon, 2 layers + MTP: 8)."""
    lead = index_leaders(layers, freq, offset, mtp)
    source, indexer = SOURCE_ROW.get(kv_dtype, 656), INDEXER_ROW * sum(lead)  # x block_size
    units = []
    for i, is_leader in enumerate(lead):
        if is_leader or not units:
            units.append([])
        units[-1].append(i)
    groups, cur = [], []
    for unit in units:
        if cur and (len(cur) + len(unit)) * source > indexer:
            groups.append(cur)
            cur = []
        cur = cur + unit
    return groups + [cur]


def multi_gpu_blocks(k: int, hot_groups: int, max_model_len: int,
                     block: int = 64, top_k: int = 2048) -> int:
    """Spill-forcing pool: one request's hot regions (device_buffer_size =
    (k+2)*top_k per hot group, runtime.py:122) + two max-length requests'
    indexer + resident pages. The coordinator's transition watermark is the
    hot cost (coordinator.py:158), so requests start reading from host once
    ~2 long prompts are resident; concurrent host readers then compete for
    hot regions (queue / preempt + restore from host)."""
    hot = hot_groups * -(-(k + 2) * top_k // block)
    return hot + 2 * (1 + hot_groups) * -(-max_model_len // block) + 64


def multi_prompts() -> list:
    """(name, prompt ids, max_tokens) for the ONE --multi generate(): long
    prompts with staggered max_tokens (decodes mix with 256-token prefill
    chunks), the target's first 2048 tokens re-used after pressure (prefix hit
    on spilled pages) with 3/40-token tails, a pure 25-token prefill, and the
    target again (full prefix hit, restore through the hot buffer)."""
    rng = random.Random(1)

    def toks(n):
        return [rng.randrange(100, 30000) for _ in range(n)]
    target = toks(4096)
    return [("target", target, 64), ("long_3000", toks(3000), 48),
            ("long_2500", toks(2500), 40), ("short_700", toks(700), 8),
            ("share_tail3", target[:2048] + toks(3), 32),
            ("share_tail40", target[:2048] + toks(40), 24),
            ("prefill_25", toks(25), 1), ("target_again", target, 16)]


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
        if a.multi:  # IndexShare followers (deepseek_v2.py:1136-1160)
            over.update(index_topk_freq=a.index_freq,
                        index_skip_topk_offset=a.index_offset)
        if any("MTP" in arch for arch in config.architectures):
            over.pop("num_hidden_layers")
        config.update(over)
        return config

    hisparse = mode != "ref"
    kw = {}
    prompts = multi_prompts() if a.multi else None
    mml = (max(len(p) + n for _, p, n in prompts) if a.multi
           else a.prompt_len + a.max_tokens) + 64
    groups = hisparse_hot_groups(a.layers, a.index_freq if a.multi else 1,
                                 a.index_offset if a.multi else 2,
                                 a.kv_cache_dtype)
    if hisparse:
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="HiSparseConnector", kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": a.host_gib})
        blocks = a.gpu_blocks
        if blocks < 0 and a.multi:
            blocks = multi_gpu_blocks(a.k, len(groups), mml)
        elif blocks < 0:
            # Auto: room for the per-request hot regions + indexer + one
            # request's pages, but not for all prompts resident -> spills
            # happen and admission still progresses. G = layers + MTP layer.
            g = a.layers + 1
            mml = a.prompt_len + a.max_tokens + 64
            # silicon (2 layers + MTP): 8 KV groups of 64-token blocks; one
            # request = hot regions of g groups + its pages in ~3g groups.
            blocks = (g * -(-(a.k + 2) * 2048 // 64) + 3 * g * -(-mml // 64)
                      + 64)
        if blocks:
            kw["num_gpu_blocks_override"] = blocks
            print(f"[suffix sm120-hisparse-mtp] ORACLE {mode}: num_gpu_blocks_override={blocks}"
                  f" (predicted {2 + 2 * len(groups)} KV groups, hot groups {groups})",
                  flush=True)
    spec = {"method": "mtp", "num_speculative_tokens": a.k}
    if a.multi:
        spec["index_share_for_mtp_iteration"] = True  # config/speculative.py:1555
    llm = LLM(
        a.model, load_format="dummy", seed=0, hf_overrides=shrink,
        skip_tokenizer_init=True, kv_cache_dtype=a.kv_cache_dtype,
        block_size=64, max_model_len=mml,
        max_num_seqs=4 if a.multi else 1, max_num_batched_tokens=a.batched_tokens,
        enable_prefix_caching=True, enable_chunked_prefill=True,
        gpu_memory_utilization=a.gpu_util, speculative_config=spec,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE",
                            "cudagraph_capture_sizes": [1, 2, 4, 8]},
        tensor_parallel_size=a.tp,
        # tp>1: "mp" workers = the only executor with the shared HiSparse
        # host pool (runtime.py use_shared_hisparse_host_pool).
        distributed_executor_backend="mp" if a.tp > 1 else None,
        **kw)

    core = llm.llm_engine.engine_core.engine_core
    # Worker-side state via RPC: with tp>1 the workers are other processes.
    base = llm.collective_rpc(_worker_probe)[0]

    layout = None
    if hisparse:
        kvc = getattr(core.scheduler, "kv_cache_config", None)
        if kvc is not None:
            layout = {"groups": len(kvc.kv_cache_groups),
                      "predicted": 2 + 2 * len(groups),
                      "num_blocks": kvc.num_blocks,
                      "host_blocks": getattr(kvc, "hisparse_host_num_blocks", None)}
            print(f"{MARK} {mode}: HiSparse HMA layout {json.dumps(layout)}", flush=True)
    if a.multi:
        out, steps = _generate_multi(llm, core, mode, prompts)
        spills = None
        if hisparse:
            from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
            spills = get_hisparse_coordinator(
                core.scheduler.kv_cache_manager).next_spill_id
        probes = llm.collective_rpc(_worker_probe)
        end, pf0 = probes[0], base["prefetch"] or {}
        return {
            "mode": mode, "tp": a.tp, "impls": end["impls"], "outputs": out,
            "spills": spills, "patched": end["patched"],
            "mtp_decode_builds": end["builds"] - base["builds"],
            "max_decode_query_len": end["max_q"],
            "shared_host_pool": [p["shared_host_pool"] for p in probes],
            "steps": steps, "layout": layout,
            "prefetch": {key: v - pf0.get(key, 0) for key, v in end["prefetch"].items()}
            if end["prefetch"] else None,
            "peak_reserved_gib": max(p["peak_reserved_gib"] for p in probes),
        }

    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                        ignore_eos=True)

    def gen(p):
        return list(llm.generate([TokensPrompt(prompt_token_ids=p)], sp,
                                 use_tqdm=False)[0].outputs[0].token_ids)

    prompts = _prompts(a.prompt_len)
    def step(name, p):
        print(f"{MARK} {mode}: {name} start", flush=True)  # crash forensics
        out[name] = gen(p)
        print(f"[suffix sm120-hisparse-mtp] ORACLE {mode}: {name} done "
              f"({len(out[name])} tokens, {time.monotonic() - t0:.1f}s)", flush=True)

    out = {}
    t0 = time.monotonic()
    step("target_1", prompts[0])
    for i, p in enumerate(prompts[1:]):
        step(f"pressure_{i}", p)
    step("target_2", prompts[0])

    spills = None
    if hisparse:
        from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
        spills = get_hisparse_coordinator(
            core.scheduler.kv_cache_manager).next_spill_id
    probes = llm.collective_rpc(_worker_probe)
    end = probes[0]
    return {
        "mode": mode, "tp": a.tp, "impls": end["impls"], "outputs": out,
        "spills": spills, "patched": end["patched"],
        "mtp_decode_builds": end["builds"] - base["builds"],
        "max_decode_query_len": end["max_q"],
        "shared_host_pool": [p["shared_host_pool"] for p in probes],
    }


def _worker_probe(worker) -> dict:
    """Runs inside each worker (collective_rpc); self-contained imports."""
    import vllm.model_executor.layers.attention.sparse_mla_attention as sma

    runner = worker.model_runner
    cfg = getattr(runner, "vllm_config", None) or worker.vllm_config
    impls = sorted({type(getattr(layer, "impl", None)).__name__
                    for layer in cfg.compilation_config
                    .static_forward_context.values()
                    if getattr(layer, "impl", None) is not None})
    stats = getattr(sma, "_SUFFIX_HISPARSE_MTP_STATS", None) or {}
    kvc = getattr(runner, "kv_cache_config", None)
    import torch
    import vllm.v1.hisparse.runtime as hrt
    pf = getattr(hrt, "_SUFFIX_HISPARSE_PREFETCH_STATS", None)
    return {"prefetch": dict(pf) if pf else None,
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "impls": impls,
            "patched": getattr(sma, "__suffix_hisparse_mtp_revision__", None),
            "builds": stats.get("multi_token_decode_builds", 0),
            "max_q": stats.get("max_decode_query_len", 0),
            "shared_host_pool": getattr(kvc, "hisparse_shared_host_pool", None)}


def _generate_multi(llm, core, mode: str, prompts: list):
    """ONE generate() of every prompt; the engine step is wrapped for live
    batch lines (the last "start" line names the crash stage) and per-request
    finish lines. Returns ({name: tokens}, step counters)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sched, engine = core.scheduler, llm.llm_engine
    orig_step = engine.step
    steps = {"steps": 0, "mixed": 0, "max_running": 0}
    t0 = time.monotonic()

    def step():
        run = list(sched.running)
        pf = sum(r.num_computed_tokens < r.num_prompt_tokens for r in run)
        steps["steps"] += 1
        steps["mixed"] += bool(pf and len(run) - pf)
        steps["max_running"] = max(steps["max_running"], len(run))
        print(f"{MARK} {mode}: step{steps['steps']}(run={len(run)},pf={pf},"
              f"wait={len(sched.waiting)}) start", flush=True)
        outs = orig_step()
        for o in outs:
            if getattr(o, "finished", False):
                print(f"{MARK} {mode}: request {o.request_id} done "
                      f"({len(o.outputs[0].token_ids)} tokens, "
                      f"{time.monotonic() - t0:.1f}s)", flush=True)
        return outs

    engine.step = step
    print(f"{MARK} {mode}: generate start ({len(prompts)} prompts: "
          + ", ".join(f"{n}={len(p)}+{m}" for n, p, m in prompts) + ")", flush=True)
    res = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for _, p, _ in prompts],
        [SamplingParams(temperature=0.0, max_tokens=m, ignore_eos=True)
         for *_, m in prompts], use_tqdm=False)
    engine.step = orig_step
    out = {n: list(r.outputs[0].token_ids) for (n, *_), r in zip(prompts, res)}
    print(f"{MARK} {mode}: generate done ({steps['steps']} steps, "
          f"{steps['mixed']} mixed prefill+decode, {time.monotonic() - t0:.1f}s)",
          flush=True)
    return out, steps


def _child_timeout(argv) -> int:
    return int(os.environ.get("SUFFIX_HISPARSE_ORACLE_CHILD_TIMEOUT",
                              "1200" if "--multi" in argv else "600"))



def _run_child(mode: str, argv) -> dict:
    """Never raises on a child failure: returns {"mode", "error"} instead."""
    env = dict(os.environ)
    env.pop("SUFFIX_SM120_HISPARSE_MTP", None)
    env.pop("SUFFIX_SM120_HISPARSE_PREFETCH", None)
    if mode in ("patched", "prefetch"):
        env["SUFFIX_SM120_HISPARSE_MTP"] = "1"
    if mode == "prefetch":
        env["SUFFIX_SM120_HISPARSE_PREFETCH"] = "1"
    timeout = _child_timeout(argv)
    # Own process group: a crashed child can leave vLLM engine-core / worker
    # processes holding the GPU; the group is killed after EVERY child so the
    # next one starts clean. Output is streamed live (a hang stays visible)
    # and captured for the verdict; _child_timeout bounds a hung child.
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
        proc.wait(timeout=timeout)
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
    started = [ln.split(": ")[-1][:-len(" start")] for ln in map(str.strip, out)
               if ln.startswith(f"{MARK} {mode}: ") and ln.endswith(" start")]
    during = f" during {started[-1]}" if started else ""
    if timed_out:
        return {"mode": mode, "error": f"TIMEOUT after {timeout}s{during} (hung)"}
    errs = [ln.strip() for ln in err if "Error" in ln]
    return {"mode": mode, "error": f"exit {proc.returncode}{during}: "
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
    if patched.get("tp", 1) > 1 and not all(patched.get("shared_host_pool") or [False]):
        fail.append(f"tp={patched['tp']} but the shared HiSparse host pool was "
                    f"not used on every rank: {patched.get('shared_host_pool')}")
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


def verdict_multi(ref: dict, patched: dict, prefetch: dict) -> tuple[int, list]:
    """Pure: (exit code, reasons) for --multi. 0 PASS, 1 FAIL, 2 INCONCLUSIVE
    (ref crashed: no reference)."""
    fail = [f"{r['mode']} crashed ({r['error']})"
            for r in (patched, prefetch) if "error" in r]
    for r in (patched, prefetch):
        if "error" in r:
            continue
        m = r["mode"]
        if not r["patched"]:
            fail.append(f"{m}: patch not active")
        if "FlashInferMLASparseSM120Impl" not in r["impls"]:
            fail.append(f"{m}: SM120 sparse-MLA backend not selected: {r['impls']}")
        if r["mtp_decode_builds"] <= 0:
            fail.append(f"{m}: no multi-token HiSparse decode batch")
        if not r["spills"]:
            fail.append(f"{m}: no HiSparse spill")
        if r.get("tp", 1) > 1 and not all(r.get("shared_host_pool") or [False]):
            fail.append(f"{m}: tp={r['tp']} but the shared HiSparse host pool "
                        f"was not used on every rank: {r.get('shared_host_pool')}")
        o = r["outputs"]
        if o["target_again"] != o["target"][:len(o["target_again"])]:
            fail.append(f"{m}: target changed after spill/restore")
        if "error" not in ref and _diff(ref["outputs"], o):
            fail.append(f"{m} != ref at {_diff(ref['outputs'], o)}")
    if "error" not in prefetch:
        if not (prefetch.get("prefetch") or {}).get("follower_prefetched"):
            fail.append("prefetch: leader never prefetched follower rows")
        if "error" not in patched and _diff(patched["outputs"], prefetch["outputs"]):
            fail.append("prefetch != patched at "
                        f"{_diff(patched['outputs'], prefetch['outputs'])}")
    if fail:
        return 1, fail
    if "error" in ref:
        return 2, [f"ref crashed ({ref['error']}): no reference"]
    return 0, []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=("ref", "stock", "patched", "prefetch"))
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
                    help="num_gpu_blocks_override for HiSparse runs (0=vLLM sizing, -1=spill-forcing auto)")
    ap.add_argument("--host-gib", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size; >1 exercises the shared (mmap) "
                         "HiSparse host pool, i.e. prod's TP=8 registration path")
    ap.add_argument("--multi", action="store_true",
                    help="max_num_seqs=4 IndexShare stress (see module doc)")
    ap.add_argument("--index-freq", type=int, default=4)
    ap.add_argument("--index-offset", type=int, default=3)
    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)
    if a.child:
        a.k = a.k[0]
        # Hang forensics: all thread stacks to stderr every 240 s (the engine
        # core runs in-process here, so this shows scheduler/HiSparse frames).
        import faulthandler
        faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
        print(RESULT + json.dumps(child(a.child, a)), flush=True)
        return 0
    codes = []
    modes = ("ref", "patched", "prefetch") if a.multi else ("ref", "stock", "patched")
    for k in a.k:
        runs = {m: _run_child(m, argv + ["--k", str(k)])  # last --k wins
                for m in modes}
        for m, r in runs.items():
            if "error" in r:
                print(f"{MARK} k={k} {m}: CRASHED {r['error']}", flush=True)
                continue
            distinct = sorted(len(set(v)) for v in r["outputs"].values())
            extra = "".join(f" {key}={json.dumps(r[key])}" for key in (
                "steps", "layout", "prefetch", "peak_reserved_gib") if key in r)
            print(f"{MARK} k={k} {m}: impls={r['impls']} spills={r['spills']} "
                  f"mtp_decode_builds={r['mtp_decode_builds']} "
                  f"max_q_len={r['max_decode_query_len']} "
                  f"distinct={distinct}{extra}", flush=True)
        rc, why = (verdict_multi(*runs.values()) if a.multi
                   else verdict(runs["ref"], runs["stock"], runs["patched"]))
        codes.append(rc)
        print(f"{MARK} k={k} {['PASS', 'FAIL', 'INCONCLUSIVE'][rc]}"
              + (": " + "; ".join(why) if why else ""), flush=True)
    return 1 if 1 in codes else max(codes)


if __name__ == "__main__":
    sys.exit(main())
