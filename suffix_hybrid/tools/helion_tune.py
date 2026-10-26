# SPDX-License-Identifier: Apache-2.0
"""In-pod Helion autotuner: the deleted console Autotune Lab's per-case loop on ONE GPU.

    python -m suffix_hybrid.tools.helion_tune --config <path | s3://bucket/prefix[/config.json] | JSON>
        [--tp 1,2,4,8] [--ep auto|on|off] [--families kgdn1,nvfp4_moe,fp8_moe,nvfp4_gemm,nvfp4_gemm_fp4mma]
        [--budget-s 600] [--effort full|quick] [--kernels a,b] [--numtokens 1,8]
        [--deadline-s 19800] [--min-speedup 1.0] [--name NAME] [--mtp-quant auto|fp8_block|nvfp4|off]
        [--cases-only]

Pod gate: SUFFIX_HELION_TUNE="<these args>" (sitecustomize.py) runs it before vLLM starts and then
holds the pod for log harvest.

Model dicts: per (tp, ep) the per-rank dims every family's cases(model) reads, from config.json
(text_config when present). EP on: experts = experts // tp of experts_global, whole
moe_intermediate; off: every expert at moe_intermediate // tp. --ep auto = both for an MoE model
at tp > 1, off otherwise. A checkpoint with MTP layers adds one draft dict per layout (architecture
<Arch>MTP, the MTP experts' quant format, no GDN, no dense linears). Cases are deduplicated per
kernel across layouts and run smallest numtokens first.

Per case (one child process per family, so a poisoned CUDA context costs only its family):
autotune (HELION_AUTOTUNER = LFBOTreeSearch when scikit-learn imports, else PatternSearch; an
explicit env value wins; HELION_AUTOTUNE_BUDGET_SECONDS = --budget-s; SIGALRM kill at
budget + max(300, budget // 2)), correctness on fresh inputs vs the kernel's `reference` (its
`check`, else atol/rtol): incorrect is never timed and never shipped. Then cold-L2 timing (CUDA
graph replay over rotating input copies, both arms eager if either cannot be captured) of the
new config, the kernel's `baseline`, and the config already shipped for this case if any.

Log lines (stdout):
  [suffix helion-tune] REC <json>                     model / case / env / record / merge / summary
  [suffix helion-tune] CONFIG <kernel> <platform> <json>   only where rows changed: the shipped
      rows + new WINS (correct, speedup > --min-speedup; a shipped key is replaced only when the
      new config timed faster than the shipped one in the same harness), sorted by case key.
      File text = json.dumps(rows, indent=2) + "\\n" at configs/helion/<kernel>/<platform>.json
  [suffix helion-tune] DONE
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from suffix_hybrid import helion_kernels as hk

PREFIX = "[suffix helion-tune]"
FAMILIES = ("kgdn1", "nvfp4_moe", "fp8_moe", "nvfp4_gemm", "nvfp4_gemm_fp4mma")
_NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_PLATFORM_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
# Rotating input copies (the lab's): spill L2 so both arms read cold weights, bounded.
L2_SPILL_FACTOR = 3
MAX_COPIES = 256
MAX_COPY_BYTES = 6 << 30
TIMED_ITERS = 20


def emit(kind: str, **row) -> None:
    print(f"{PREFIX} REC {json.dumps({'kind': kind, **row}, default=str)}", flush=True)


# ------------------------------------------------------------------ config -> model dicts --
def load_config(spec: str) -> dict:
    spec = spec.strip()
    if spec.startswith("{"):
        return json.loads(spec)
    if spec.startswith("s3://"):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise SystemExit(f"{PREFIX} --config {spec}: boto3 is not importable ({exc})")
        bucket, _, key = spec[5:].partition("/")
        if not key.endswith(".json"):
            key = key.rstrip("/") + "/config.json"
        # path-style: Garage has no virtual-host names (the console's download script)
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                          config=Config(s3={"addressing_style": "path"},
                                        retries={"max_attempts": 5}))
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if os.path.isdir(spec):
        spec = os.path.join(spec, "config.json")
    with open(spec) as f:
        return json.load(f)


def model_name(spec: str, config: dict) -> str:
    """The checkpoint dir name (nvfp4_gemm keys per-arch shape tables on it)."""
    spec = spec.strip()
    if spec.startswith("{"):
        return str(config.get("_name_or_path") or "").rstrip("/").split("/")[-1]
    spec = spec.rstrip("/")
    if spec.endswith(".json"):
        spec = spec.rsplit("/", 1)[0] if "/" in spec else ""
    return spec.split("/")[-1]


def _int(v):
    try:
        return int(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def _quant_config(config: dict) -> dict:
    return config.get("quantization_config") or (config.get("text_config") or {}).get(
        "quantization_config") or {}


def _block(qc) -> list | None:
    """The first weight_block_size anywhere in a quantization config (sub)tree."""
    if isinstance(qc, dict):
        if isinstance(qc.get("weight_block_size"), list):
            return [int(x) for x in qc["weight_block_size"]]
        qc = list(qc.values())
    if isinstance(qc, list):
        for v in qc:
            b = _block(v)
            if b:
                return b
    return None


def quant_format(qc: dict) -> str:
    blob = json.dumps(qc).lower()
    if "fp4" in blob:
        return "nvfp4"
    if "fp8" in blob:
        return "fp8_block" if _block(qc) else "fp8"
    return str(qc.get("quant_method") or "bf16")


def mtp_quant(qc: dict, main: str) -> str | None:
    """Quant format of the MTP layers: entries whose key names an mtp module (fp8 before fp4,
    the MoE experts are what fp8_moe tunes), None when the mtp modules are only listed as
    ignored/excluded (bf16: nothing here tunes them), else the checkpoint's own format."""
    hits, ignored = [], False

    def walk(o):
        nonlocal ignored
        if isinstance(o, dict):
            for k, v in o.items():
                if "mtp" in str(k).lower():
                    hits.append(json.dumps(v).lower())
                walk(v)
        elif isinstance(o, list):
            for v in o:
                if isinstance(v, str) and "mtp" in v.lower():
                    ignored = True
                walk(v)

    walk(qc)
    blob = " ".join(hits)
    if "fp8" in blob:
        return "fp8_block"
    if "fp4" in blob:
        return "nvfp4"
    return None if ignored else main


def _div(a, b, what):
    if a is None:
        return None
    if a % b:
        raise ValueError(f"{what}={a} is not divisible by tp={b}")
    return a // b


def model_dict(config: dict, tp: int, ep: bool, name: str = "") -> dict:
    """Per-TP-rank dims (the deleted console model_dims + the GDN, shared-expert and gated-
    attention linears). EP: this rank's experts of experts_global, whole moe_intermediate."""
    text = config.get("text_config") or config

    def g(*keys):
        return next((_int(text.get(k)) for k in keys if _int(text.get(k)) is not None), None)

    hidden, heads = g("hidden_size"), g("num_attention_heads")
    kv = g("num_key_value_heads") or heads
    head_dim = g("head_dim") or (hidden // heads if hidden and heads else None)
    inter, moe_inter = g("intermediate_size"), g("moe_intermediate_size")
    experts = g("num_experts", "num_local_experts", "n_routed_experts")
    topk = g("num_experts_per_tok", "moe_topk", "top_k_experts")
    shared = g("shared_expert_intermediate_size")
    vocab = g("vocab_size") or _int(config.get("vocab_size"))
    nk, nv = g("linear_num_key_heads"), g("linear_num_value_heads")
    dk, dv = g("linear_key_head_dim"), g("linear_value_head_dim")
    if experts and not moe_inter:
        moe_inter, inter = inter, None  # Mixtral-style: every MLP is an expert
    ep_size = tp if ep and experts and tp > 1 else 1
    r_heads = _div(heads, tp, "num_attention_heads")
    r_kv = max(1, kv // tp) if kv else None  # vLLM replicates KV heads past tp
    linears = []
    if nk and nv and dk and dv and hidden:  # Gated DeltaNet layers
        linears += [{"name": "gdn_in_proj_qkvz", "n": _div(2 * nk * dk + 2 * nv * dv, tp, "gdn qkvz"),
                     "k": hidden},
                    {"name": "gdn_in_proj_ba", "n": _div(2 * nv, tp, "linear_num_value_heads"),
                     "k": hidden},
                    {"name": "gdn_out_proj", "n": hidden, "k": _div(nv * dv, tp, "gdn v")}]
    if hidden and r_heads and r_kv and head_dim and not text.get("kv_lora_rank"):
        # ponytail: hybrid GDN checkpoints (Qwen3-Next / Qwen4Exp) gate attention unless the
        # config says otherwise (vLLM default); other families need attn_output_gate set
        gated = text.get("attn_output_gate", bool(nk))
        q = r_heads * head_dim * (2 if gated else 1)
        linears += [{"name": "qkv_proj", "n": q + 2 * r_kv * head_dim, "k": hidden},
                    {"name": "o_proj", "n": hidden, "k": r_heads * head_dim}]
    dense = inter and (not experts or text.get("mlp_only_layers")
                       or (g("first_k_dense_replace") or 0) > 0)
    if hidden and dense:
        linears += [{"name": "gate_up_proj", "n": _div(2 * inter, tp, "intermediate_size"), "k": hidden},
                    {"name": "down_proj", "n": hidden, "k": _div(inter, tp, "intermediate_size")}]
    if hidden and experts and shared:
        linears += [{"name": "shared_gate_up", "n": _div(2 * shared, tp, "shared_expert"), "k": hidden},
                    {"name": "shared_down", "n": hidden, "k": _div(shared, tp, "shared_expert")}]
    if hidden and vocab:
        linears.append({"name": "lm_head", "n": math.ceil(vocab / tp), "k": hidden})
    qc = _quant_config(config)
    fmt = quant_format(qc)
    return {
        "name": name, "architecture": (config.get("architectures") or [None])[0],
        "tp": tp, "ep": ep_size, "hidden": hidden,
        "intermediate": _div(inter, tp, "intermediate_size") if dense else None,
        "moe_intermediate": (moe_inter if ep_size > 1 else _div(moe_inter, tp, "moe_intermediate_size"))
        if moe_inter else None,
        "experts": _div(experts, ep_size, "num_experts") if experts else None,
        # experts_total: the deleted console's key (nvfp4_moe's DeepSeek branch reads it)
        "experts_global": experts, "experts_total": experts, "topk": topk,
        "heads": r_heads, "kv_heads": r_kv, "head_dim": head_dim,
        "vocab": math.ceil(vocab / tp) if vocab else None,
        "rms_eps": text.get("rms_norm_eps") or 1e-6,
        "quant": qc.get("quant_method"), "quant_format": fmt,
        "block": _block(qc) if fmt == "fp8_block" else None,
        "linears": linears,
        # raw (whole-model) GDN dims: kgdn1.cases divides by tp itself
        "linear_num_key_heads": nk, "linear_num_value_heads": nv,
        "linear_key_head_dim": dk, "linear_value_head_dim": dv,
        "swiglu_limit": text.get("swiglu_limit"),
    }


def mtp_dict(config: dict, main: dict, mtp_q: str) -> dict | None:
    text = config.get("text_config") or config
    layers = _int(text.get("mtp_num_hidden_layers")) or _int(text.get("num_nextn_predict_layers"))
    if not layers or mtp_q == "off":
        return None
    qc = _quant_config(config)
    fmt = mtp_quant(qc, main["quant_format"]) if mtp_q == "auto" else mtp_q
    if fmt is None:
        return None
    arch = re.sub(r"For(CausalLM|ConditionalGeneration)$", "MTP", str(main["architecture"] or ""))
    # ponytail: an FP8 MTP without weight_block_size is taken as 128x128 (the Qwen3.8-Flash-Next
    # checkpoint, kernels/fp8_moe.py doc); read the real block when a checkpoint differs
    block = (_block(qc) or [128, 128]) if fmt == "fp8_block" else None
    return {**main, "architecture": arch, "quant_format": fmt, "block": block, "linears": [],
            "intermediate": None, "linear_num_key_heads": None, "linear_num_value_heads": None,
            "linear_key_head_dim": None, "linear_value_head_dim": None}


def layouts(config: dict, tps, ep_mode: str, name: str = "", mtp_q: str = "auto") -> list:
    """[(label, model dict)] per (tp, ep), plus the MTP draft dict of each."""
    text = config.get("text_config") or config
    moe = any(_int(text.get(k)) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    out = []
    for tp in tps:
        eps = {"on": [True], "off": [False]}.get(ep_mode, [False, True])
        for ep in eps if (moe and tp > 1) else [False]:
            label = f"tp{tp}" + ("-ep" if ep else "")
            try:
                m = model_dict(config, tp, ep, name)
            except ValueError as exc:
                emit("model", layout=label, error=str(exc))
                continue
            out.append((label, m))
            d = mtp_dict(config, m, mtp_q)
            if d:
                out.append((label + "-mtp", d))
    return out


def collect(kernel: dict, models, kernels=None, numtokens=None) -> list:
    """[(case, [layout labels])] of one KERNELS entry, deduplicated, smallest numtokens first."""
    if kernels and kernel["name"] not in kernels:
        return []
    seen = {}
    for label, model in models:
        for case in kernel["cases"](model):
            if numtokens and case.get("numtokens") not in numtokens:
                continue
            seen.setdefault(hk.case_key(case), (case, []))[1].append(label)
    return sorted(seen.values(), key=lambda cl: cl[0].get("numtokens") or 0)


# ------------------------------------------------------------------ merge (parent, no torch) --
def wins(rec: dict, min_speedup: float = 1.0) -> bool:
    return (rec.get("status") == "ok" and rec.get("correct") is True and bool(rec.get("config"))
            and (rec.get("speedup") or 0) > min_speedup)


def merge(shipped: list, recs: list, min_speedup: float = 1.0):
    """(rows, added, replaced): the shipped [{"key", "config"}] rows plus winning records of
    ONE (kernel, platform). A shipped key is replaced only when the record timed faster than
    the shipped config (shipped_us; None = the shipped config could not run/was not timed)."""
    rows = {r["key"] if isinstance(r["key"], str) else hk.case_key(r["key"]): r for r in shipped}
    best = {}
    for rec in recs:
        if wins(rec, min_speedup):
            k = hk.case_key(rec["case"])
            if k not in best or rec["helion_us"] < best[k]["helion_us"]:
                best[k] = rec
    added = replaced = 0
    for k, rec in best.items():
        if k in rows:
            if rec.get("shipped_us") is not None and rec["helion_us"] >= rec["shipped_us"]:
                continue
            replaced += 1
        else:
            added += 1
        rows[k] = {"key": rec["case"], "config": rec["config"]}
    return [rows[k] for k in sorted(rows)], added, replaced


def shipped_rows(kernel: str, platform: str) -> list:
    path = os.path.join(hk.CONFIG_ROOT, kernel, platform + ".json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ GPU child --
class CaseTimeout(BaseException):
    """BaseException: Helion's autotuner catches Exception around every benchmark."""


def _alarm(signum, frame):
    raise CaseTimeout()


def default_autotuner() -> str:
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401 - what LFBO imports
    except Exception:  # noqa: BLE001
        return "PatternSearch"
    return "LFBOTreeSearch"


def kill_after(budget_s: int) -> int:
    return int(budget_s) + max(300, int(budget_s) // 2)


def _env() -> dict:
    import torch
    from importlib.metadata import version
    env = {"platform": hk.platform(), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "l2_bytes": getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)}
    for pkg in ("triton", "helion", "vllm"):
        try:
            env[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            env[pkg] = None
    try:
        import helion
        env["helion_loaded"] = getattr(helion, "__version__", None)
        env["helion_path"] = os.path.dirname(helion.__file__)
    except Exception:  # noqa: BLE001
        pass
    env["autotuner"] = os.environ.get("HELION_AUTOTUNER")
    env["budget_s"] = os.environ.get("HELION_AUTOTUNE_BUDGET_SECONDS")
    return env


def _nbytes(args) -> int:
    seen, total = set(), 0
    for a in args:
        if hasattr(a, "untyped_storage"):
            s = a.untyped_storage()
            if (str(a.device), s.data_ptr()) not in seen:
                seen.add((str(a.device), s.data_ptr()))
                total += s.nbytes()
    return total


def copies_for(touched: int, allocated: int, l2_bytes: int) -> int:
    if touched <= 0:
        return 1
    want = math.ceil(L2_SPILL_FACTOR * l2_bytes / touched)
    return max(1, min(MAX_COPIES, want, MAX_COPY_BYTES // max(1, allocated)))


def _time_us(fn, sets, graph: bool) -> float:
    import torch
    for args in sets:
        fn(*args)
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
            for args in sets:
                fn(*args)
        g.replay()
        torch.cuda.synchronize()
        step = g.replay
    else:
        def step():
            for args in sets:
                fn(*args)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        step()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / TIMED_ITERS / len(sets)


def _time_arms(arms):
    """([us per arm], method): every arm graph-replayed, or every arm eager when any one
    cannot be captured (a graph-vs-eager pair fakes wins at decode sizes)."""
    import torch
    try:
        return [_time_us(f, s, True) for f, s in arms], "graph"
    except Exception:  # noqa: BLE001
        torch.cuda.synchronize()
        return [_time_us(f, s, False) for f, s in arms], "eager"


def _check(kernel, out, ref):
    import torch
    if kernel.get("check"):
        return kernel["check"](out, ref)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    if len(outs) != len(refs):
        return False, f"kernel returned {len(outs)} outputs, reference {len(refs)}"
    worst = 0.0
    for o, r in zip(outs, refs):
        try:
            torch.testing.assert_close(o, r, atol=kernel.get("atol", 1e-2),
                                       rtol=kernel.get("rtol", 1e-2), check_dtype=False)
        except AssertionError as exc:
            return False, str(exc).splitlines()[0][:300]
        worst = max(worst, (o.float() - r.float()).abs().max().item())
    return True, f"max abs err {worst:.3g}"


def run_case(kernel: dict, case: dict, effort: str, l2_bytes: int) -> dict:
    import helion
    import torch
    rec = {"status": "ok"}
    settings = {"static_shapes": True, **(kernel.get("settings") or {})}
    t0 = time.time()
    args = kernel["make"](case, 0)
    tuner = helion.kernel(kernel["body"], autotune_effort=effort, **settings)
    config = tuner.autotune(args, force=True)
    rec["autotune_s"] = round(time.time() - t0, 1)
    rec["config"] = json.loads(config.to_json())
    best = helion.kernel(kernel["body"], config=config, **settings)
    fresh = kernel["make"](case, 0)  # autotune runs may have mutated `args` in place
    ref = kernel["reference"](*fresh)
    ok, msg = _check(kernel, best(*fresh), ref)
    rec["correct"], rec["check"] = bool(ok), msg
    del fresh, ref
    if not ok:
        rec["status"] = "incorrect"
        return rec
    est = kernel.get("working_set_bytes")
    touched, allocated = (est or _nbytes)(args), _nbytes(args)
    copies = copies_for(touched, allocated, l2_bytes)
    sets = [args] + [kernel["make"](case, s) for s in range(1, copies)]
    make_base = kernel.get("make_baseline") or kernel["make"]
    base_sets = [make_base(case, s) for s in range(copies)]
    arms = [(best, sets), (kernel["baseline"], base_sets)]
    old = hk.config(kernel["name"], case)
    if old is not None:
        try:
            shipped = helion.kernel(kernel["body"], config=helion.Config(**old), **settings)
            shipped(*sets[0])
            arms.append((shipped, sets))
        except Exception as exc:  # noqa: BLE001 - e.g. a knob this Helion does not know
            rec["shipped_error"] = f"{type(exc).__name__}: {exc}"[:500]
            if "CUDA" in str(exc) or "illegal" in str(exc):
                raise
    us, rec["timing"] = _time_arms(arms)
    rec["helion_us"], rec["baseline_us"] = round(us[0], 3), round(us[1], 3)
    rec["shipped_us"] = round(us[2], 3) if len(us) > 2 else None
    rec["speedup"] = round(us[1] / us[0], 4) if us[0] else None
    rec["copies"], rec["working_set_bytes"] = copies, touched
    del sets, base_sets, arms
    torch.cuda.empty_cache()
    return rec


def child(task: dict) -> int:
    family = task["family"]
    os.environ.setdefault("HELION_AUTOTUNER", default_autotuner())  # before the module's setdefault
    os.environ["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(task["budget_s"])
    base = {"family": family}
    try:
        mod = importlib.import_module(f"suffix_hybrid.helion_kernels.{family}")
        env = _env()
    except Exception as exc:  # noqa: BLE001
        emit("record", **base, status="error", error=f"family import failed: {exc}"[:2000])
        return 0
    emit("env", **base, **env)
    if getattr(mod, "helion", None) is None:
        emit("record", **base, status="error", error="helion is not importable in this pod")
        return 0
    signal.signal(signal.SIGALRM, _alarm)
    kill_s = kill_after(task["budget_s"])
    models = [tuple(m) for m in task["models"]]
    for kernel in mod.KERNELS:
        try:
            todo = collect(kernel, models, task.get("kernels"), task.get("numtokens"))
        except Exception as exc:  # noqa: BLE001
            emit("record", **base, kernel=kernel["name"], status="error",
                 error=f"cases() failed: {exc}"[:2000])
            continue
        for case, labels in todo:
            row = {**base, "kernel": kernel["name"], "case": case, "case_key": hk.case_key(case),
                   "layouts": labels, "platform": env["platform"], "effort": task["effort"],
                   "baseline": kernel.get("baseline_name")}
            if time.time() + kill_s > task["deadline"]:
                emit("record", **row, status="skipped", error="--deadline-s reached")
                continue
            signal.alarm(kill_s)
            try:
                emit("record", **row, **run_case(kernel, case, task["effort"], env["l2_bytes"]))
            except CaseTimeout:
                emit("record", **row, status="error", error=f"case killed after {kill_s}s")
                return 0  # autotune workers may still hold the GPU: end the family
            except Exception as exc:  # noqa: BLE001
                text = f"{type(exc).__name__}: {exc}"
                where = "".join(f" [at {os.path.basename(f.filename)}:{f.lineno} {f.name}]"
                                for f in traceback.extract_tb(exc.__traceback__)[-1:])
                emit("record", **row, status="error", error=(text + where)[:2000])
                if "CUDA" in text or "illegal" in text:
                    emit("record", **base, status="error",
                         error="CUDA error poisons the context: family stopped")
                    return 0
            finally:
                signal.alarm(0)
    return 0


# ------------------------------------------------------------------ parent --
def _run_family(task: dict) -> list:
    """Run one family's child, echo its stdout, return its record rows. A child silent past
    its SIGALRM kill (stuck in a CUDA sync) is killed with its process group."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(task, f)
    proc = subprocess.Popen([sys.executable, "-m", "suffix_hybrid.tools.helion_tune",
                             "--child", f.name], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout] + [lines.put(None)],
                     daemon=True).start()
    recs, silent_s = [], kill_after(task["budget_s"]) + 180
    tag = f"{PREFIX} REC "
    while True:
        try:
            line = lines.get(timeout=silent_s)
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            emit("record", family=task["family"], status="error",
                 error=f"child silent for {silent_s}s: killed")
            break
        if line is None:
            break
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith(tag):
            try:
                recs.append(json.loads(line[len(tag):]))
            except ValueError:
                pass
    if proc.wait():
        emit("record", family=task["family"], status="error", error=f"child exit {proc.returncode}")
    os.unlink(f.name)
    return recs


def gpu_streams(value: str) -> list:
    """SUFFIX_HELION_TUNE value -> [(gpu index or None, argv)]. ' ;; ' separates
    independent arg sets, one per GPU: stream i runs with CUDA_VISIBLE_DEVICES=i
    (a 2-GPU pool tunes two families at once). One set = no pinning."""
    import shlex
    sets = [shlex.split(part) for part in value.split(";;") if part.strip()]
    if len(sets) == 1:
        return [(None, sets[0])]
    return [(i, argv) for i, argv in enumerate(sets)]


def _csv(text, cast=str):
    return [cast(x) for x in str(text).split(",") if x.strip()] if text else None


def raise_nofile() -> tuple:
    """Soft RLIMIT_NOFILE -> hard. Helion's autotuner leaks descriptors when
    candidate benchmark subprocesses crash (illegal-address configs); at the
    container default 1024 a family died mid-run with EMFILE (2026-10-26)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else max(soft, 1 << 20)
    if want > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--config")
    p.add_argument("--tp", default="1")
    p.add_argument("--ep", default="auto", choices=("auto", "on", "off"))
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--kernels", help="only these KERNELS names")
    p.add_argument("--numtokens", help="only these numtokens buckets")
    p.add_argument("--budget-s", type=int, default=600, help="Helion search budget per case")
    p.add_argument("--effort", default="full", choices=("full", "quick"))
    p.add_argument("--deadline-s", type=int, default=19800,
                   help="whole-run wall budget: later cases are skipped (startup probe: 6 h)")
    p.add_argument("--min-speedup", type=float, default=1.0)
    p.add_argument("--name", help="checkpoint name (default: from --config)")
    p.add_argument("--mtp-quant", default="auto", choices=("auto", "fp8_block", "nvfp4", "off"))
    p.add_argument("--cases-only", action="store_true", help="list the cases, no GPU")
    p.add_argument("--spec-k", type=int, default=0,
                   help="serving MTP k (kgdn1_verify cases need it; 0 = no verify cases)")
    a = p.parse_args(argv)
    raise_nofile()
    if a.child:
        with open(a.child) as f:
            return child(json.load(f))
    if not a.config:
        p.error("--config is required")
    families = _csv(a.families)
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        p.error(f"unknown families {bad} (known: {','.join(FAMILIES)})")
    t0 = time.time()
    config = load_config(a.config)
    models = layouts(config, _csv(a.tp, int), a.ep, a.name or model_name(a.config, config),
                     a.mtp_quant)
    for label, m in models:
        m["spec_k"] = a.spec_k
        emit("model", layout=label, model=m)
    kernels, numtokens = _csv(a.kernels), _csv(a.numtokens, int)
    recs = []
    for family in families:
        if a.cases_only:
            mod = importlib.import_module(f"suffix_hybrid.helion_kernels.{family}")
            for kernel in mod.KERNELS:
                todo = collect(kernel, models, kernels, numtokens)
                for case, labels in todo:
                    emit("case", family=family, kernel=kernel["name"], case=case, layouts=labels)
                emit("count", family=family, kernel=kernel["name"], cases=len(todo))
            continue
        recs += _run_family({"family": family, "models": models, "budget_s": a.budget_s,
                             "effort": a.effort, "deadline": t0 + a.deadline_s,
                             "kernels": kernels, "numtokens": numtokens})
    done = [r for r in recs if r.get("kind") == "record" and r.get("case") is not None]
    groups = {}
    for r in done:
        groups.setdefault((r["kernel"], r["platform"]), []).append(r)
    for (kernel, plat), rs in sorted(groups.items()):
        if not (_NAME_RE.match(str(kernel)) and _PLATFORM_RE.match(str(plat))):
            continue
        old = shipped_rows(kernel, plat)
        rows, added, replaced = merge(old, rs, a.min_speedup)
        emit("merge", kernel=kernel, platform=plat, shipped=len(old), added=added,
             replaced=replaced, rows=len(rows))
        if added or replaced:
            print(f"{PREFIX} CONFIG {kernel} {plat} {json.dumps(rows, separators=(',', ':'))}",
                  flush=True)
    status = {}
    for r in done:
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
    emit("summary", records=len(done), status=status,
         wins=sum(wins(r, a.min_speedup) for r in done), wall_s=round(time.time() - t0, 1))
    print(f"{PREFIX} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
