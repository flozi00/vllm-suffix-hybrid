# SPDX-License-Identifier: Apache-2.0
"""The bake-off of the native-FP4-MMA family (lane I): ONE command per question, lab-ready for one RTX
PRO 6000 and one RTX 5090 (the same script on both; the platform name is in every record). NOTHING in this
file has run on silicon: the pure parts (case lists, winner table, config shipping) are tested on CPU.

    python -m suffix_hybrid.kernels.fp4mma_bench dense --screen 21 --tune quick --out /tmp/fp4
    python -m suffix_hybrid.kernels.fp4mma_bench dense --screen dsv41 --tune full --out /tmp/fp4
    python -m suffix_hybrid.kernels.fp4mma_bench report /tmp/fp4        # re-render tables from the records
    python -m suffix_hybrid.kernels.fp4mma_bench selftest               # CPU

DENSE arms (same NVFP4 operands, same exact f64 reference, every arm checked before it is timed; a
wrong arm is reported WRONG and never wins):
  fp4mma_words / _bytes / _lin   native FP4 MMA, the three scale-access patterns (nvfp4_gemm_fp4mma.py)
  fp4cvt                         hardware cvt.rn.f16x2.e2m1x2 decode + f16 MMA (the "ALU vs MMA" control)
  alu_decode                     the current Helion body (nvfp4_gemm.py nvfp4_gemm_q_body: bit-placement
                                 decode in ALU, bf16 MMA)
  oxide                          kernels-oxide nvfp4_gemm at its production launch (where the bundle has the cubin)
  flashinfer                     vLLM's FlashInfer CUTLASS FP4 GEMM (the stock kernel; V4.1 has no other)
  triton_pr38338                 the raw-Triton kernel of vLLM#38338 (linear scales, 9 tile configs): the
                                 control that separates "Triton dot_scaled" from "what Helion generates"
  scaled_mm                      torch.nn.functional.scaled_mm / torch._scaled_mm NVFP4 (cuBLASLt; unverified on SM120:
                                 the record says "runs" or the exception)
Timing = the lane's autoroute method (kernels/nvfp4_gemm._graph_us): one CUDA graph per arm, the L2
flushed (a 2x-L2 scratch read) before EVERY replay, the median of --iters replays; the Helion arms
include their stage-2 reduce and their per-call allocations (what serves). Records are JSON lines
(<out>/dense.jsonl); the WINNER TABLE prints per (shape, M); a case where the best fp4mma arm beats the
stock arms (flashinfer, oxide) by >= --margin (default 3 %, the autoroute's) is a SHIPPING ROW, written to
<out>/configs/<kernel>/<platform>.json = [{"key": case, "config": {...}}], the exact convention of
docs/helion-kernels.md (copy the files into suffix_hybrid/configs/helion/ for the winners only; the
pull keeps one body per case: the fastest).

MOE: not here. The routed experts are a body VARIANT of lane B's seam (docs/dsv41-moe.md section 11), so
the K4 for them is lane B's gate bench (helion_kernels/nvfp4_moe_tune.py): FlashInfer and EVERY body variant
per bucket, balanced and skewed, cold L2, the VARIANT TABLE and the `dsv41_moe_variant` rows it writes. This
file benches the dense GEMM arms only.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time

MARKER = "[suffix fp4mma-bench]"
DEFAULT_MARGIN = 0.03
SCREEN_FABLE_NAMES = ("mlp_gate_up", "gdn_in_proj_qkvz", "mlp_down", "attn_qkv", "gdn_out_proj", "attn_o")
SCREEN_MS = (1, 16, 64)
DSV41_MS = (1, 6, 16, 32, 64, 96, 128, 192)
V41_LM_HEAD = (16160, 5120)
QWEN_LM_HEAD = (248320, 5120)
ARMS = ("fp4mma_words", "fp4mma_bytes", "fp4mma_lin", "fp4cvt", "alu_decode", "oxide", "flashinfer",
        "triton_pr38338", "scaled_mm")
STOCK_ARMS = ("flashinfer", "oxide")
FP4MMA_ARMS = {"fp4mma_words": "nvfp4_gemm_q_fp4mma", "fp4mma_bytes": "nvfp4_gemm_q_fp4mma_b",
               "fp4mma_lin": "nvfp4_gemm_q_fp4mma_lin", "fp4cvt": "nvfp4_gemm_q_fp4cvt"}


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# case lists (pure)
# ---------------------------------------------------------------------------
def screen_cases(name: str) -> list:
    """[(label, n, k, m)]. 21 = the dossier's quick screen: GEMM_Q on the fable (qwen3.8-27b) shapes -- its
    6 dense linears + the lm_head -- x M {1, 16, 64}; dsv41 = the V4.1 lm_head at the decode / verify Ms;
    gemma = gemma-4 26B's dense linears x M {1, 16, 64}."""
    from suffix_hybrid.kernels import nvfp4_gemm as ng
    if name == "21":
        rows = [(nm, *ng.SHAPES[nm]) for nm in SCREEN_FABLE_NAMES] + [("lm_head", *QWEN_LM_HEAD)]
        return [(f"{lab}.m{m}", n, k, m) for lab, n, k in rows for m in SCREEN_MS]
    if name == "dsv41":
        return [(f"v41_lm_head.m{m}", *V41_LM_HEAD, m) for m in DSV41_MS]
    if name == "gemma":
        return [(f"{lab}.m{m}", n, k, m) for lab, (n, k) in ng.GEMMA_SHAPES.items() for m in SCREEN_MS]
    if name == "qwen-flash":
        # Qwen4Exp (qwen3.8-flash-next) per-rank TP2 dense shapes + the 0.32 QSA qkv (indexer rows
        # fused: 7296) + the TP2 lm_head; serving M with MTP k=4 at c1/c8/c32 ~ 5/40/160 -> 8/64/192
        rows = list(ng.QWEN_FLASH_SHAPES.items()) + [("flash_qsa_qkv_v032", (7296, 2560)),
                                                     ("flash_lm_head_tp2", (124160, 2560))]
        return [(f"{lab}.m{m}", n, k, m) for lab, (n, k) in rows for m in (1, 8, 64, 192)]
    if name == "all":
        return screen_cases("21") + screen_cases("dsv41") + screen_cases("gemma")
    raise ValueError(f"unknown screen {name!r}: want 21 | dsv41 | gemma | qwen-flash | all")


def case_dict(n: int, k: int, m: int) -> dict:
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    from suffix_hybrid import helion_kernels as hk
    return {"k": k, "n": n, "numtokens": hk.bucket(m, fm.BUCKETS)}


# ---------------------------------------------------------------------------
# the winner table + shipping rows (pure: tested on CPU with canned records)
# ---------------------------------------------------------------------------
def best(records: list, arms) -> dict | None:
    ok = [r for r in records if r["arm"] in arms and r.get("correct") and r.get("us")]
    return min(ok, key=lambda r: r["us"]) if ok else None


def winners(records: list, margin: float = DEFAULT_MARGIN) -> list:
    """One row per (label): the per-arm times, the stock arm to beat (best of STOCK_ARMS), the best fp4mma
    arm, the verdict WIN | LOSS | TIE | NO-STOCK | NO-ARM and the speedup (stock / fp4mma)."""
    rows = []
    for label in dict.fromkeys(r["label"] for r in records):
        rs = [r for r in records if r["label"] == label]
        stock = best(rs, STOCK_ARMS)
        mine = best(rs, FP4MMA_ARMS)
        overall = best(rs, ARMS)
        row = {"label": label, "n": rs[0]["n"], "k": rs[0]["k"], "m": rs[0]["m"],
               "us": {r["arm"]: (round(r["us"], 2) if r.get("correct") and r.get("us") else None)
                      for r in rs},
               "wrong": sorted(r["arm"] for r in rs if r.get("correct") is False),
               "errors": {r["arm"]: r["error"] for r in rs if r.get("error")},
               "stock": stock["arm"] if stock else None, "mine": mine["arm"] if mine else None,
               "winner": overall["arm"] if overall else None}
        if mine is None:
            row["verdict"], row["speedup"] = "NO-ARM", None
        elif stock is None:
            row["verdict"], row["speedup"] = "NO-STOCK", None
        else:
            sp = stock["us"] / mine["us"]
            row["speedup"] = round(sp, 3)
            row["verdict"] = "WIN" if sp >= 1.0 + margin else ("LOSS" if sp < 1.0 else "TIE")
        rows.append(row)
    return rows


def table(rows: list, platform: str = "") -> str:
    arms = [a for a in ARMS if any(r["us"].get(a) for r in rows)]
    head = f"{'case':28s} " + " ".join(f"{a[:13]:>13s}" for a in arms) + "  verdict (stock/fp4mma)"
    lines = [f"WINNER TABLE {platform}  (us, cold L2, CUDA graph, median)", head]
    for r in rows:
        cells = " ".join(f"{(str(r['us'].get(a)) if r['us'].get(a) else '-'):>13s}" for a in arms)
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else ""
        lines.append(f"{r['label']:28s} {cells}  {r['verdict']} {sp} [{r['mine'] or '-'} vs {r['stock'] or '-'}]"
                     + (f" WRONG: {','.join(r['wrong'])}" if r["wrong"] else ""))
    wins = sum(1 for r in rows if r["verdict"] == "WIN")
    lines.append(f"{wins}/{len(rows)} cases won by >= margin")
    return "\n".join(lines)


def shipping_rows(records: list, margin: float = DEFAULT_MARGIN) -> dict:
    """{kernel: [{"key": case, "config": cfg}]} for the cases a fp4mma arm wins by >= margin: ONE row per
    (kernel, case key); several M of the same bucket must ALL win for the bucket to ship (a bucket
    is served for every M of it), and the config shipped is the one tuned at the bucket's top M."""
    out, by_case = {}, {}
    for row in winners(records, margin):
        rs = [r for r in records if r["label"] == row["label"] and r["arm"] == row["mine"]]
        if not rs or row["verdict"] != "WIN":
            by_case.setdefault(json.dumps(case_dict(row["n"], row["k"], row["m"]), sort_keys=True), []).append(False)
            continue
        key = json.dumps(case_dict(row["n"], row["k"], row["m"]), sort_keys=True)
        by_case.setdefault(key, []).append(rs[0])
    for key, vals in by_case.items():
        if not vals or not all(vals):
            continue
        rec = max(vals, key=lambda r: r["m"])  # the config tuned at the bucket's largest M
        kernel = FP4MMA_ARMS[rec["arm"]]
        if rec.get("config") is None:
            continue
        out.setdefault(kernel, []).append({"key": json.loads(key), "config": rec["config"]})
    return out


def write_configs(out_dir: str, platform: str, shipped: dict) -> list:
    paths = []
    for kernel, rows in shipped.items():
        d = os.path.join(out_dir, "configs", kernel)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, platform + ".json")
        with open(path, "w") as f:
            json.dump(sorted(rows, key=lambda r: json.dumps(r["key"], sort_keys=True)), f, indent=1)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# device side
# ---------------------------------------------------------------------------
def _graph_us(fn, dev, iters, flush):
    from suffix_hybrid.kernels.nvfp4_gemm import _graph_us
    return _graph_us(fn, dev, iters, flush)


@functools.lru_cache(maxsize=None)
def toolchain() -> dict:
    """What compiled the kernels of this run, in every record: the lab (Triton 3.7.1) vs serving (3.8.0)
    skew and the ptxas that assembled them decide whether a timing transfers (dossier 3.7 T3)."""
    import torch
    out = {"torch": torch.__version__}
    try:
        import triton
        from triton.backends.nvidia.compiler import get_ptxas
        cc = torch.cuda.get_device_capability()
        path = get_ptxas(cc[0] * 10 + cc[1]).path
        ver = subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
        out.update(triton=triton.__version__, ptxas=ver[-1] if ver else "?", ptxas_path=path)
    except Exception as exc:  # noqa: BLE001 — the record says why instead of hiding it
        out["triton_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import helion
        out["helion"] = getattr(helion, "__version__", "?")
    except ImportError:
        out["helion"] = None
    return out


def _flusher(dev):
    import torch
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0) or 128 << 20)
    buf = torch.zeros(max(2 * l2, 256 << 20) // 4, dtype=torch.float32, device=dev)
    return (lambda: buf.sum()), buf


def tune_body(body, args, effort: str, settings: dict):
    """Autotune a Helion body on `args` (effort quick | full); the winning config as a dict. The
    served form is dynamic-shape, so the tuning is too (one config per bucket across its Ms)."""
    import helion
    if effort == "none":
        return None
    try:
        import sklearn.ensemble  # noqa: F401 — what LFBO imports
        os.environ.setdefault("HELION_AUTOTUNER", "LFBOTreeSearch")
    except Exception:  # noqa: BLE001
        os.environ.setdefault("HELION_AUTOTUNER", "PatternSearch")
    tuner = helion.kernel(body, autotune_effort=effort, static_shapes=False, **settings)
    cfg = tuner.autotune(args, force=True)
    return json.loads(cfg.to_json()) if hasattr(cfg, "to_json") else dict(cfg)


def dense_arm(arm: str, n: int, k: int, m: int, dev, tune: str, iters: int, flush, seed: int = 0) -> dict:
    """Build, check and time one arm. Returns the record (never raises)."""
    from suffix_hybrid import helion_kernels as hk
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    rec = {"arm": arm, "n": n, "k": k, "m": m, "platform": hk.platform(), "toolchain": toolchain()}
    t0 = time.time()
    try:
        case = {"k": k, "n": n, "numtokens": m}
        if arm in FP4MMA_ARMS:
            kind = FP4MMA_ARMS[arm]
            p = fm.problem(case, seed, dev, kind)
            args = fm.body_args(kind, p)
            ref = (fm.reference_lin if kind == fm.LIN else fm.reference_q)(*args)
            cfg = _shipped_or_default(kind, case, m)
            tuned = None
            if tune != "none":
                tuned = tune_body(fm.BODIES[kind], args, tune, {"ignore_warnings": fm._ignore()})
            rec["config"], rec["config_source"] = (tuned, "tuned") if tuned else (cfg[0], cfg[1])
            fn, out_of = _helion_runner(kind, p, rec["config"])
        elif arm == "alu_decode":
            fn, out_of, ref, rec["config"] = _alu_runner(case, seed, dev, tune)
        elif arm == "oxide":
            fn, out_of, ref = _oxide_runner(case, seed, dev)
        elif arm == "flashinfer":
            fn, out_of, ref = _flashinfer_runner(case, seed, dev)
        elif arm == "triton_pr38338":
            fn, out_of, ref = _triton_pr_runner(case, seed, dev)
        elif arm == "scaled_mm":
            fn, out_of, ref, rec["recipe"] = _scaled_mm_runner(case, seed, dev)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        out = out_of(fn)
        ok, msg = fm.check(out, ref)
        rec["correct"], rec["check"] = ok, msg
        if ok:
            rec["us"] = _graph_us(fn, dev, iters, flush)
            rec["gbps"] = round(n * k * 0.5625 / (rec["us"] * 1e-6) / 1e9, 1)
    except Exception as exc:  # noqa: BLE001 — an arm that cannot run is a result
        rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _shipped_or_default(kind, case, m):
    from suffix_hybrid import helion_kernels as hk
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    bucketed = {**case, "numtokens": hk.bucket(m, fm.BUCKETS)}
    cfg = hk.config(kind, bucketed)
    return (cfg, "shipped") if cfg is not None else (fm.probe_config(m, 1), "default")


def _helion_runner(kind, p, cfg):
    import helion
    import torch
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    kern = helion.kernel(fm.BODIES[kind], config=helion.Config(**cfg), static_shapes=False,
                         ignore_warnings=fm._ignore())
    split = int(cfg.get("split_k", 1))
    entry = (kind, (kern,), split, p["m"])
    alpha = p["alpha"]
    xq, xsf, w, wsf = p["xq"], p["xsf"], p["w"], p["wsf"]

    def fn(_stream=None):
        out = torch.empty(p["m"], p["n"], dtype=torch.bfloat16, device=xq.device)
        return fm.gemm(xq, xsf, w, wsf, out, entry, alpha)

    return fn, (lambda f: f(None))


def _alu_runner(case, seed, dev, tune):
    import helion
    from suffix_hybrid import helion_kernels as hk
    from suffix_hybrid.helion_kernels import nvfp4_gemm as old
    p = old.problem(case, seed, dev, True)
    args = old.body_args(old.GEMM_Q, p)
    ref = old.reference_q(*args)
    cfg = hk.config(old.GEMM_Q, case)
    kern = helion.kernel(old.BODIES[old.GEMM_Q], config=helion.Config(**(cfg or {})), static_shapes=False)
    return (lambda _s=None: kern(*args)), (lambda f: f(None)), ref, cfg


def _oxide_runner(case, seed, dev):
    from suffix_hybrid.helion_kernels import nvfp4_gemm as old
    args = old._baseline_args(case, seed, True)
    p = old.problem(case, seed, dev, True)
    ref = old.reference_q(p["xq"], p["xsf"], p["w"], p["wsf"], p["out"], p["partial"], p["counters"], p["alpha"])
    return (lambda _s=None: old._baseline_q(*args)), (lambda f: f(None)), ref


def _flashinfer_runner(case, seed, dev):
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    args = fm._fi_args(case, seed)
    p = fm.problem(case, seed, dev, fm.WORDS)
    ref = fm.reference_q(*fm.body_args(fm.WORDS, p))
    return (lambda _s=None: fm._fi_baseline(*args)), (lambda f: f(None)), ref


_PR_KERNEL_SRC = '''
import triton
import triton.language as tl


@triton.jit
def pr38338_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, alpha, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   stride_a_scale_m, stride_b_scale_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   VEC_SIZE: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    # vLLM PR #38338 benchmarks/kernels/triton_nvfp4_gemm.py (Apache-2.0, vLLM project), EVEN_K path:
    # tl.dot_scaled on e2m1 operands with linear e4m3 block-16 scales.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    SCALE_K: tl.constexpr = BLOCK_K // VEC_SIZE
    offs_scale_k = tl.arange(0, SCALE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * stride_a_scale_m + offs_scale_k[None, :]
    b_scale_ptrs = b_scale_ptr + offs_n[:, None] * stride_b_scale_n + offs_scale_k[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        scale_a = tl.load(a_scale_ptrs)
        scale_b = tl.load(b_scale_ptrs)
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator,
                                    lhs_k_pack=True, rhs_k_pack=True)
        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        a_scale_ptrs += SCALE_K
        b_scale_ptrs += SCALE_K
    c = (accumulator * alpha).to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None].to(tl.int64) * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
'''
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, warps, stages): the 9 configs vLLM#38338 shipped (5090 sweep)
PR_CONFIGS = ((256, 128, 128, 32, 8, 2), (128, 128, 128, 32, 8, 3), (64, 256, 256, 1, 8, 3),
              (64, 128, 256, 8, 4, 4), (64, 64, 256, 8, 4, 4), (32, 64, 256, 8, 4, 5),
              (16, 64, 256, 1, 4, 4), (16, 64, 256, 1, 8, 3), (16, 32, 256, 8, 4, 4))


def pr_configs_for(m: int, n: int, k: int) -> list:
    """The PR's pruning: BLOCK_M <= next_pow2(M) (at least 16), BLOCK_N <= next_pow2(N), BLOCK_K | K."""
    mcap = max(16, 1 << max(0, (m - 1).bit_length()))
    return [c for c in PR_CONFIGS if c[0] <= mcap and c[1] <= max(16, n) and k % c[2] == 0] or [PR_CONFIGS[-1]]


def _triton_pr_runner(case, seed, dev):
    import importlib.util
    import tempfile
    import torch
    import triton
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    d = tempfile.mkdtemp(prefix="pr38338_")
    path = os.path.join(d, "pr38338_kernel.py")
    with open(path, "w") as f:
        f.write(_PR_KERNEL_SRC)
    spec = importlib.util.spec_from_file_location("pr38338_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = fm.problem(case, seed, dev, fm.LIN)  # linear scales, the PR's layout
    m, n, k = p["m"], p["n"], p["k"]
    xs = fm._u8(p["xsf"]).view(torch.float8_e4m3fn).reshape(m, k // 16)
    ws = fm._u8(p["wsf"]).view(torch.float8_e4m3fn).reshape(n, k // 16)
    ref = fm.reference_lin(*fm.body_args(fm.LIN, p))
    best_cfg = {}

    def launch(cfg):
        bm, bn, bk, gm, warps, stages = cfg
        out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        mod.pr38338_kernel[grid](p["xq"], p["w"], out, xs, ws, float(p["alpha"]), m, n, k,
                                 p["xq"].stride(0), p["xq"].stride(1), p["w"].stride(1), p["w"].stride(0),
                                 out.stride(0), out.stride(1), xs.stride(0), ws.stride(0),
                                 BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, VEC_SIZE=16, GROUP_SIZE_M=gm,
                                 num_warps=warps, num_stages=stages)
        return out

    # pick the fastest of the PR's pruned configs with plain warm timing (its autotuner's job)
    cands = pr_configs_for(m, n, k)
    if len(cands) > 1:
        best_t = None
        for cfg in cands:
            try:
                launch(cfg)
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(True), torch.cuda.Event(True)
                t0[0].record()
                for _ in range(5):
                    launch(cfg)
                t0[1].record()
                torch.cuda.synchronize()
                t = t0[0].elapsed_time(t0[1])
            except Exception:  # noqa: BLE001
                continue
            if best_t is None or t < best_t:
                best_t, best_cfg["cfg"] = t, cfg
    best_cfg.setdefault("cfg", cands[0])
    return (lambda _s=None: launch(best_cfg["cfg"])), (lambda f: f(None)), ref


def _scaled_mm_runner(case, seed, dev):
    """NVFP4 through torch (cuBLASLt): F.scaled_mm with the two-level recipe, else torch._scaled_mm with
    the block scales only (the global alpha applied after). Whichever form runs is recorded; if none,
    the exception is the finding."""
    import torch
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    p = fm.problem(case, seed, dev, fm.WORDS)
    ref = fm.reference_q(*fm.body_args(fm.WORDS, p))
    a = p["xq"].view(torch.float4_e2m1fn_x2)
    b = p["w"].view(torch.float4_e2m1fn_x2).t()
    sa = p["xsf_swz"].reshape(-1).view(torch.float8_e4m3fn)
    sb = p["wsf_swz"].reshape(-1).view(torch.float8_e4m3fn)
    ga = torch.tensor(p["alpha"], dtype=torch.float32, device=dev)
    gb = torch.tensor(1.0, dtype=torch.float32, device=dev)
    forms = []
    try:
        from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
        forms.append(("F.scaled_mm two-level", lambda: scaled_mm(
            a, b, scale_a=[sa, ga], scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            scale_b=[sb, gb], scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
            swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE], output_dtype=torch.bfloat16)))
    except ImportError:
        pass
    forms.append(("torch._scaled_mm block scales * alpha", lambda: (torch._scaled_mm(
        a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16).float() * p["alpha"]).bfloat16()))
    errors = []
    for name, f in forms:
        try:
            f()
            torch.cuda.synchronize()
            return (lambda _s=None, f=f: f()), (lambda fn: fn(None)), ref, name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
    raise RuntimeError("scaled_mm NVFP4 did not run on this GPU: " + " | ".join(errors))


def run_dense(args) -> int:
    import torch
    from suffix_hybrid import helion_kernels as hk
    if not torch.cuda.is_available():
        _log("DENSE NOT RUN: no CUDA device")
        return 2
    from suffix_hybrid.kernels import fp4mma_probe
    fp4mma_probe.require_native_stack()
    fp4mma_probe.install_recorder()
    dev = torch.device("cuda", torch.cuda.current_device())
    plat = hk.platform()
    arms = [a for a in ARMS if not args.arms or a in args.arms.split(",")]
    cases = screen_cases(args.screen)
    if args.m:
        keep = {int(x) for x in args.m.split(",")}
        cases = [c for c in cases if c[3] in keep]
    flush, buf = _flusher(dev)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dense.jsonl")
    done, recs, end = {}, [], time.time() + args.budget_s
    _log(f"dense: {len(cases)} cases x {len(arms)} arms on {torch.cuda.get_device_name(dev)} ({plat}), "
         f"tune={args.tune}, iters={args.iters}, margin={args.margin}")
    _log(f"toolchain {toolchain()}")
    with open(path, "a") as out:
        for label, n, k, m in cases:
            for arm in arms:
                if time.time() > end:
                    _log("budget spent: stopping")
                    break
                key = (arm, n, k, m)
                if key in done:  # the same shape under another name (attn_o == gdn_out_proj)
                    rec = {**done[key], "label": label}
                else:
                    rec = dense_arm(arm, n, k, m, dev, args.tune, args.iters, flush)
                    rec["label"] = label
                    done[key] = rec
                    torch.cuda.empty_cache()
                recs.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                _log(f"{label} {arm}: " + (f"{rec['us']:.1f} us" if rec.get("us") else
                                           rec.get("error") or f"WRONG {rec.get('check')}"))
    return finish_dense(recs, args.out, plat, args.margin)


def finish_dense(recs, out_dir, plat, margin) -> int:
    rows = winners(recs, margin)
    print(table(rows, plat))
    shipped = shipping_rows(recs, margin)
    for p in write_configs(out_dir, plat, shipped):
        print(f"{MARKER} wrote {p}")
        with open(p) as f:  # in-pod runs: the log is the only way out
            print(f"{MARKER} CONFIG {os.path.relpath(p, out_dir)} {json.dumps(json.load(f))}", flush=True)
    with open(os.path.join(out_dir, "dense-winners.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 0


def selftest() -> int:
    """CPU: case lists, winner logic, shipping rows (bucket rule), the PR config pruning, max-M line."""
    assert len(screen_cases("21")) == 21 and screen_cases("21")[0][1:] == (34816, 5120, 1)
    assert {c[3] for c in screen_cases("dsv41")} == set(DSV41_MS)
    recs = [
        {"arm": "flashinfer", "label": "a.m1", "n": 64, "k": 128, "m": 1, "correct": True, "us": 10.0},
        {"arm": "oxide", "label": "a.m1", "n": 64, "k": 128, "m": 1, "correct": True, "us": 12.0},
        {"arm": "fp4mma_words", "label": "a.m1", "n": 64, "k": 128, "m": 1, "correct": True, "us": 8.0,
         "config": {"split_k": 1}},
        {"arm": "fp4cvt", "label": "a.m1", "n": 64, "k": 128, "m": 1, "correct": False, "us": 1.0},
        {"arm": "flashinfer", "label": "a.m16", "n": 64, "k": 128, "m": 16, "correct": True, "us": 10.0},
        {"arm": "fp4mma_words", "label": "a.m16", "n": 64, "k": 128, "m": 16, "correct": True, "us": 10.1,
         "config": {"split_k": 2}},
    ]
    w = winners(recs)
    assert w[0]["verdict"] == "WIN" and w[0]["speedup"] == 1.25 and w[0]["wrong"] == ["fp4cvt"]
    assert w[0]["winner"] == "fp4mma_words" and w[1]["verdict"] == "LOSS"  # 10.0 / 10.1 < 1.0
    ship = shipping_rows(recs)
    # bucket 1 (M=1) wins alone -> ships; bucket 16 (M=16) did not win -> no row
    assert [r["key"]["numtokens"] for r in ship["nvfp4_gemm_q_fp4mma"]] == [1], ship
    assert "WINNER TABLE" in table(w, "x")
    assert pr_configs_for(1, 4096, 5120)[0][0] == 16 and all(c[0] <= 64 for c in pr_configs_for(40, 4096, 5120))
    print("fp4mma_bench selftest OK")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fp4mma_bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dense")
    d.add_argument("--screen", default="21", help="21 | dsv41 | gemma | qwen-flash | all")
    d.add_argument("--arms", default="", help="comma list (default all): " + ",".join(ARMS))
    d.add_argument("--m", default="", help="restrict to these Ms")
    d.add_argument("--tune", default="none", choices=("none", "quick", "full"))
    d.add_argument("--iters", type=int, default=15)
    d.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    d.add_argument("--budget-s", type=float, default=3000.0)
    d.add_argument("--out", default="/tmp/fp4mma")
    rp = sub.add_parser("report")
    rp.add_argument("dir")
    rp.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    sub.add_parser("selftest")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "dense":
        from suffix_hybrid import helion_kernels as hk
        hk.ensure_vendor()  # standalone (in-pod tune stream): no serving lane put _vendor on sys.path
        return run_dense(a)
    if a.cmd == "report":
        recs = [json.loads(ln) for ln in open(os.path.join(a.dir, "dense.jsonl"))]
        plat = (recs[0].get("platform") if recs else "") or ""
        return finish_dense(recs, a.dir, plat, a.margin)
    parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
