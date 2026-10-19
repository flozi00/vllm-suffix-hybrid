# SPDX-License-Identifier: Apache-2.0
"""Silicon driver for the DSA indexer lane on DeepSeek-V4.1 (ONE SM120 GPU, a
kernel lab with this plugin mounted and vLLM 39983a5e7 installed).

One run produces, for the eight V4.1 indexer layers (2/8/14 dense at ctx/2, 20 the
candidate source, 24/28/32/36 consumers), at ctx 8192 .. 524288 and 1 / 4 / 32
requests of 1 + 5 DSpark rows:
  * a GPU oracle of every new body (row-group, survivor select, remap, block
    selection) against the f64 spec, adversarial rows included;
  * full-effort Helion autotune of every serving bucket (the autotune runner's
    recipe: static shapes, oxide/Triton accuracy baseline, check() vs the f64
    spec), group kernel first;
  * per-layer-kind timings of the STOCK path (vendored DeepGEMM paged logits ->
    vLLM's candidate mask / block selection -> persistent_topk, exactly what
    vLLM 39983a5e7 runs on SM120) against ours (oxide row group, Helion per row,
    Helion row group, fused candidates, valid-block selection, warm-start
    survivor select), cold L2 per replay;
  * the report: winner table, the stock-vs-ours decode-step total (3 x dense at
    ctx/2 + source + 4 x consumer) and the crossovers (first ctx where each
    lever wins), next to the byte model of kernels/dsa_indexer_cost.py.

Resumable: progress lives in STATE; each call works through TASKS until its
budget, then prints this call's results as JSON lines ("[dsa-lab] {...}"); once
everything ran, further calls print the report and page through the stored
results.

  python -m suffix_hybrid.kernels.dsa_indexer_lab [budget_s]
"""
from __future__ import annotations

import json
import os
import sys
import time

STATE = os.environ.get("SUFFIX_DSA_LAB_STATE", "/tmp/dsa_indexer_lab.json")
TAG = "[dsa-lab]"
HEADS = 32
G = 6                                   # 1 + 5 DSpark rows per request
TOPK = 512                              # DeepSeek-V4.1 index_topk
MML = 524288                            # logits width of a ratio-1 layer (d6-spec.json target)
CAND, CBS = 2048, 8                     # candidate_topk_blocks / candidate_block_size
POS_GRID = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)   # index positions per request
REQS = (1, 4, 32)                       # requests of G rows (rows 6 / 24 / 192)
MAX_TOKENS = 4 * 1024 * 1024            # skip points holding more than this many K tokens
# (Q rows, K window) of the prefill benchmark: vLLM bounds a chunk's logits to
# VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (M x N <= 134M f32), so deep contexts run few rows
PREFILL_POINTS = ((8192, 8192), (8192, 32768), (1024, 131072), (256, 524288))
MODEL = {"architecture": "DeepseekV41ForCausalLM"}
ARCH_ORDER = ("dsa_indexer_group", "dsa_indexer_cand", "dsa_indexer_select", "dsa_indexer_decode",
              "dsa_indexer_prefill")


def points():
    """[(kind, positions, requests)] the layer benchmarks cover. Dense layers
    (ratio 2) scan ctx/2 positions, source / consumer layers ctx; the step
    totals need dense at ctx/2 for every ctx of the grid."""
    out = []
    for kind, grid in (("dense", POS_GRID[:-1]), ("source", POS_GRID[1:]), ("consumer", POS_GRID[1:])):
        for n in grid:
            for reqs in REQS:
                if reqs * n <= MAX_TOKENS:
                    out.append((kind, n, reqs))
    return out


def tasks():
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    by_name = {k["name"]: k for k in hdi.KERNELS}
    out = [["oracle", None], ["gate", None]]
    for name in ARCH_ORDER:
        out += [["tune", [name, i]] for i in range(len(by_name[name]["cases"](MODEL)))]
    out += [["bench_layer", list(p)] for p in points()]
    out += [["bench_prefill", list(mn)] for mn in PREFILL_POINTS]
    out += [["report", None]]
    return out


def _load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"done": 0, "results": [], "configs": {}, "cursor": 0}


def _save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def _emit(row):
    print(f"{TAG} {json.dumps(row, separators=(',', ':'))}", flush=True)


# ---------------------------------------------------------------------------
# the report: pure functions over the stored bench rows (CPU-tested)
# ---------------------------------------------------------------------------
# Arms that can serve V4.1 decode (deployable): oxide only reads contiguous 64-token pages (V4.1's
# packed / 32- and 128-token pages are Helion-only) so it is a reference arm, never a candidate.
# groupS = the row-group body under the SERVING policy: the longest-context bucket's config at every
# context (a decode graph is captured once); group = the config tuned for this very context.
OURS = {"dense": ("row_us", "groupS_us", "select_us"),
        "source": ("row_us", "groupS_us", "row_candsel_us", "groupS_candsel_us"),
        "consumer": ("row_us", "groupS_us", "cand_fused_us")}
REFERENCE = {"dense": ("oxide_us", "group_us"), "source": ("oxide_us", "group_us", "group_candsel_us"),
             "consumer": ("oxide_us", "group_us")}
STOCK = "stock_us"


def _num(row, key):
    v = row.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def best(row, kind):
    """(arm, us) of the fastest of OURS[kind] that ran; None when none did."""
    got = [(k, _num(row, k)) for k in OURS[kind] if _num(row, k) is not None]
    return min(got, key=lambda kv: kv[1]) if got else None


def index_rows(results):
    """{(kind, positions, requests): bench row} from stored bench_layer results."""
    out = {}
    for r in results:
        if r.get("task", [None])[0] == "bench_layer" and isinstance(r.get("result"), dict):
            kind, n, reqs = r["task"][1]
            out[(kind, n, reqs)] = r["result"]
    return out


def step_totals(rows):
    """[{ctx, reqs, stock_us, ours_us, speedup, arms}]: one decode step's eight
    indexer layers = 3 dense (ctx/2 positions) + source + 4 consumers, from the
    measured layer rows; a ctx is listed when every needed layer row has a stock
    time and an ours time."""
    out = []
    for reqs in REQS:
        for ctx in POS_GRID[1:]:
            need = [("dense", ctx // 2, reqs, 3), ("source", ctx, reqs, 1), ("consumer", ctx, reqs, 4)]
            stock = ours = 0.0
            arms = {}
            for kind, n, rq, mult in need:
                row = rows.get((kind, n, rq))
                b = best(row, kind) if row else None
                if row is None or _num(row, STOCK) is None or b is None:
                    break
                stock += mult * row[STOCK]
                ours += mult * b[1]
                arms[kind] = b[0]
            else:
                out.append({"ctx": ctx, "reqs": reqs, "stock_us": round(stock, 1), "ours_us": round(ours, 1),
                            "speedup": round(stock / ours, 3), "arms": arms})
    return out


def crossovers(rows):
    """First position count where each lever beats its alternative, per request
    count: the candidate fusion vs the dense path (consumer), the warm select vs
    the dense group path (dense), the row group vs the Helion per-row body (any
    kind), and ours-best vs stock. None: never won on the measured grid."""
    levers = {"cand_fused<dense_mask": ("consumer", "cand_fused_us", "groupS_us"),
              "select<group": ("dense", "select_us", "groupS_us"),
              "group<row": ("dense", "groupS_us", "row_us"),
              "group<oxide": ("dense", "groupS_us", "oxide_us"),
              "candsel<stock_select": ("source", "groupS_candsel_us", "groupS_us")}
    out = {}
    for reqs in REQS:
        for name, (kind, a, b) in levers.items():
            hit = None
            for n in POS_GRID:
                row = rows.get((kind, n, reqs))
                if row and _num(row, a) is not None and _num(row, b) is not None and row[a] < row[b]:
                    hit = n
                    break
            out[f"{name}@reqs{reqs}"] = hit
        for kind in OURS:
            hit = None
            for n in POS_GRID:
                row = rows.get((kind, n, reqs))
                bb = best(row, kind) if row else None
                if bb and _num(row, STOCK) is not None and bb[1] < row[STOCK]:
                    hit = n
                    break
            out[f"ours<stock[{kind}]@reqs{reqs}"] = hit
    return out


def winners(rows):
    """{kind/positions/reqs: [arm ranking]} fastest first, stock included."""
    out = {}
    for (kind, n, reqs), row in sorted(rows.items()):
        arms = [(k, _num(row, k)) for k in (STOCK,) + OURS[kind] + REFERENCE[kind] if _num(row, k) is not None]
        out[f"{kind}/{n}/{reqs}"] = [f"{k}={v}" for k, v in sorted(arms, key=lambda kv: kv[1])]
    return out


def policy_penalty(rows):
    """groupS_us / group_us per point: what serving the longest bucket's config at
    every context costs against a config per context (1.0 = nothing)."""
    out = {}
    for (kind, n, reqs), row in sorted(rows.items()):
        a, b = _num(row, "groupS_us"), _num(row, "group_us")
        if a and b:
            out[f"{kind}/{n}/{reqs}"] = round(a / b, 3)
    return out


def report(results):
    rows = index_rows(results)
    from suffix_hybrid.kernels import dsa_indexer_cost as cost
    model = {f"{ctx}/{reqs}": {"stock_us": round(cost.step_cost(ctx, reqs, G, False)["us"], 1),
                               "ours_us": round(cost.step_cost(ctx, reqs, G, True)["us"], 1)}
             for ctx in POS_GRID[1:] for reqs in REQS}
    return {"step_totals": step_totals(rows), "crossovers": crossovers(rows), "winners": winners(rows),
            "serving_policy_penalty": policy_penalty(rows), "byte_model_roofline_us": model}


# ---------------------------------------------------------------------------
# helpers (GPU)
# ---------------------------------------------------------------------------
def _dev():
    import torch
    return torch.device("cuda", torch.cuda.current_device())


def _timer():
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    dev = _dev()
    scrub = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

    def us(fn, iters=20):
        try:
            return round(nm._graph_us(lambda s: fn(), dev, iters, lambda: scrub.fill_(1)), 2)
        except Exception as exc:  # noqa: BLE001 - an arm that cannot run is a result
            torch.cuda.synchronize()
            return f"{type(exc).__name__}: {str(exc)[:160]}"
    return us


def _cfg(st, kernel, case):
    from suffix_hybrid import helion_kernels as hk
    return st["configs"].get(kernel, {}).get(hk.case_key(case)) or hk.config(kernel, case)


def _any_cfg(st, kernel):
    """Some config of `kernel` (tuned in this run, else shipped): timing arms of
    shapes that were not tuned still need a schedule, and the config is shape-agnostic."""
    from suffix_hybrid import helion_kernels as hk
    got = list(st["configs"].get(kernel, {}).values())
    return got[-1] if got else (next(iter(hk.table(kernel).values()), None))


def _kern(body, cfg):
    import helion
    if cfg is None:
        return helion.kernel(body, autotune_effort="none", static_shapes=False)
    return helion.kernel(body, config=helion.Config(**cfg), static_shapes=False)


def _strided_cache(kf, ksc, page, pad):
    """uint8 [blocks, page, 132] split pages at stride page * 132 + pad (GPU)."""
    import torch
    nb = kf.shape[0] // page
    stride = page * 132 + pad
    back = torch.randint(0, 256, (nb * stride,), dtype=torch.uint8, device=kf.device)
    pages = back.view(nb, stride)
    pages[:, :page * 128] = kf.view(torch.uint8).reshape(nb, page * 128)
    pages[:, page * 128:page * 132] = ksc.view(torch.uint8).reshape(nb, page * 4)
    return back.as_strided((nb, page, 132), (stride, 132, 1))


def _decode_pr(reqs, ctx, seed=7):
    from suffix_hybrid.kernels import dsa_indexer as di
    return di.make_decode(_dev(), HEADS, reqs, G, ctx, True, MML, seed=seed, exact=True)


def _rel(a, b, cl):
    """max |a - b| / max |b| over each row's [0, cl) window."""
    worst = 0.0
    for r in range(a.shape[0]):
        n = int(cl[r])
        if n:
            d = (a[r, :n] - b[r, :n]).abs().max() / b[r, :n].abs().max().clamp_min(1e-30)
            worst = max(worst, float(d))
    return worst


class _Topk:
    """vLLM's persistent_topk (the SM12x choice for k = 512 / 2048) without the engine."""
    def __init__(self, dev):
        import torch
        import vllm._custom_ops  # noqa: F401
        self.ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)

    def __call__(self, logits, lens, nn, out, k, bound):
        import torch
        torch.ops._C.persistent_topk(logits, lens.reshape(-1, 1).to(torch.int32), out, self.ws, k, bound)


def _hash_rows(idx):
    return [tuple(sorted(x for x in row if x >= 0)) for row in idx.tolist()]


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------
def t_oracle(st, _):
    import helion
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_candidates as dc
    from suffix_hybrid.kernels import dsa_indexer as di
    from suffix_hybrid.kernels import dsa_select as ds
    dev, res = _dev(), {}
    n_sms = di._sms(dev)

    def verdict(ok, msg):
        return f"{'OK' if ok else 'FAIL'} {msg}"

    try:
        res["lane_oxide_triton"] = di.oracle()[-60:]
    except Exception as exc:  # noqa: BLE001
        res["lane_oxide_triton"] = f"FAIL {type(exc).__name__}: {str(exc)[:300]}"
    # Triton fallback, nightly contracts: compressed prefill
    args = hdi.prefill_problem(HEADS, 300, 5000, dev)
    res["triton_prefill_compressed"] = verdict(*hdi.check(hdi._prefill_tune_baseline(*args, 0),
                                                           hdi.prefill_reference(*args, 0)))
    g = torch.Generator(device=dev).manual_seed(3)
    for page, pad in ((128, 0), (128, 256), (64, 512), (32, 128), (64, 0)):
        nb = 90
        kf = (torch.randn((nb * page, 128), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        ksc = 2.0 ** torch.randint(-8, -3, (nb * page,), generator=g, device=dev).float()
        cache = _strided_cache(kf, ksc, page, pad)
        rows, n_cols = 14, 5000          # 14 rows: two groups of 6 + a padded third
        bt = torch.randperm(nb, generator=g, device=dev)[:40].to(torch.int32).repeat(rows, 1)
        # rows of a group belong to DIFFERENT requests half the time: shuffle tables
        bt[1::3] = torch.randperm(nb, generator=g, device=dev)[:40].to(torch.int32)
        cl = torch.randint(1, min(40 * page, n_cols), (rows,), generator=g, device=dev).to(torch.int32)
        cl[3] = 0
        q = (torch.randn((rows, HEADS, 128), generator=g, device=dev) * 2).to(torch.float8_e4m3fn)
        w = torch.randn((rows, HEADS), generator=g, device=dev) * 0.02
        kv, ks, pg, prow, sp = hdi.cache_views(cache)
        out = torch.full((rows, n_cols), float("-inf"), device=dev)
        ref = hdi.decode_reference(q, w, kv, ks, bt, cl, out, pg, prow, sp, 2)
        tri = di._fallback_mod().fp8_fp4_paged_mqa_logits(
            (q.view(rows, 1, HEADS, 128), None), cache.unsqueeze(2), w, cl.view(rows, 1), bt, None, n_cols)
        col = torch.arange(n_cols, device=dev)
        tri = torch.where(col[None, :] < cl[:, None], tri, float("-inf"))
        tag = f"p{page}_pad{pad}"
        res[f"triton_paged_{tag}"] = verdict(*hdi.check(tri, ref))
        splits = hdi.splits_for(rows, n_cols, n_sms)
        kern = helion.kernel(hdi.dsa_decode_logits, autotune_effort="none", static_shapes=False)
        res[f"helion_row_default_{tag}"] = verdict(*hdi.check(
            kern(q, w, kv, ks, bt, cl, out.clone(), pg, prow, sp, splits), ref))
        q4, w3, clp = hdi.group_pad(q, w, cl, G)
        kern = helion.kernel(hdi.dsa_group_logits, autotune_effort="none", static_shapes=False)
        res[f"helion_group_default_{tag}"] = verdict(*hdi.check(
            kern(q4, w3, kv, ks, bt, clp, out.clone(), pg, prow, sp, hdi.splits_for(q4.shape[0], n_cols, n_sms)), ref))
        # survivor select: bound = the 12th largest logit of each row (mid) / all / none
        val, mag = ref
        for mode in ("mid", "all", "none"):
            thr = torch.empty(rows, device=dev)
            for r in range(rows):
                v = val[r][~torch.isnan(val[r])].float()
                t = v.topk(min(13, max(v.numel(), 1))).values if v.numel() else torch.zeros(2, device=dev)
                thr[r] = (t[-1] + t[-2]) / 2 if t.numel() > 1 else 0.0
            thr = {"mid": thr, "all": torch.full_like(thr, float("-inf")),
                   "none": torch.full_like(thr, float("inf"))}[mode]
            cv = torch.full((rows, n_cols), float("nan"), device=dev)
            ci = torch.full((rows, n_cols), -1, dtype=torch.int32, device=dev)
            cn = torch.zeros(rows, dtype=torch.int32, device=dev)
            kern = helion.kernel(hdi.dsa_group_select, autotune_effort="none", static_shapes=False)
            got = kern(q4, w3, kv, ks, bt, clp, thr, cv, ci, cn, pg, prow, sp,
                       hdi.splits_for(q4.shape[0], n_cols, n_sms))
            res[f"helion_select_{mode}_{tag}"] = verdict(*hdi.select_check(got, (val, mag, thr)))
        pr = {"cl": cl}
        cand = hdi.cand_problem(pr, 64, 8, seed=page + pad)
        cout = torch.full((rows, 512), float("-inf"), device=dev)
        cargs = (q, w, kv, ks, bt, cl, cand, cout, pg, prow, sp, hdi.splits_for(rows, 512, n_sms))
        kern = helion.kernel(hdi.dsa_cand_logits, autotune_effort="none", static_shapes=False)
        res[f"helion_cand_default_{tag}"] = verdict(*hdi.check(kern(*cargs), hdi.cand_reference(*cargs)))
    args = hdi.prefill_problem(HEADS, 300, 5000, dev)
    kern = helion.kernel(hdi.dsa_prefill_logits, autotune_effort="none", static_shapes=False)
    res["helion_prefill_compressed_default"] = verdict(*hdi.check(kern(*args, 4), hdi.prefill_reference(*args, 4)))

    # the Triton table remap == its torch twin (positions, dead slots, stale positions)
    idx = torch.randint(-1, 3000, (40, 512), generator=g, device=dev, dtype=torch.int32)
    tab = torch.randint(-1, 5000, (40, 3000), generator=g, device=dev, dtype=torch.int32)
    cl40 = torch.randint(0, 5000, (40,), generator=g, device=dev, dtype=torch.int32)
    want = ds.remap(idx.cpu(), tab.cpu(), cl40.cpu())
    got = dc.cand_remap(idx.clone(), tab, cl40, 1).cpu()
    res["triton_remap_table"] = verdict(bool(torch.equal(want, got)), "slots -> positions")
    cand8 = torch.randint(-1, 400, (40, 64), generator=g, device=dev, dtype=torch.int32)
    idx8 = torch.randint(-1, 512, (40, 128), generator=g, device=dev, dtype=torch.int32)
    want = hdi.cand_remap(idx8.clone(), cand8, cl40, 8)
    got = dc.cand_remap(idx8.clone(), cand8, cl40, 8)
    res["triton_remap_blocks"] = verdict(bool(torch.equal(want, got)), "slots -> block positions")

    # valid-block selection vs vLLM's stock select_candidate_blocks (set equality, ties aside)
    try:
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
        topk = _Topk(dev)
        fast = dc.select_blocks_fn(cb, lambda backend: topk, cb.select_candidate_blocks)
        rows_, width = 24, 524288
        lg = torch.randn((rows_, width), generator=g, device=dev)
        ke = torch.cat([torch.randint(1, width, (rows_ - 4,), generator=g, device=dev, dtype=torch.int32),
                        torch.tensor([width, 8, 16400, 1], dtype=torch.int32, device=dev)])
        a = torch.full((rows_, CAND), -9, dtype=torch.int32, device=dev)
        b = a.clone()
        cb.select_candidate_blocks(lg, None, ke, CAND, CBS, a)
        fast(lg, None, ke, CAND, CBS, b)
        torch.cuda.synchronize()
        same = _hash_rows(a.cpu()) == _hash_rows(b.cpu())
        res["candsel_vs_stock"] = verdict(same, f"{rows_} rows, width {width}")
    except Exception as exc:  # noqa: BLE001
        res["candsel_vs_stock"] = f"FAIL {type(exc).__name__}: {str(exc)[:200]}"
    return res


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------
def t_tune(st, spec):
    """The autotune runner's _run_case, in the lab: spec = [kernel, case index]."""
    import helion
    import torch
    from suffix_hybrid import helion_kernels as hk
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    name, i = spec
    k = next(x for x in hdi.KERNELS if x["name"] == name)
    case = k["cases"](MODEL)[i]
    args = k["make"](case, 0)
    ref = k["reference"](*args)
    settings = {"static_shapes": True, **(k.get("settings") or {})}
    t0 = time.time()
    cfg = helion.kernel(k["body"], autotune_effort="full", **settings).autotune(args, force=True)
    best_k = helion.kernel(k["body"], config=cfg, **settings)
    ok, msg = k["check"](best_k(*args), ref)
    del ref
    rec = {"kernel": name, "case": case, "autotune_s": round(time.time() - t0), "correct": ok,
           "check": msg, "config": json.loads(cfg.to_json())}
    if ok:
        us = _timer()
        bargs = (k.get("make_baseline") or k["make"])(case, 0)
        rec["helion_us"] = us(lambda: best_k(*args))
        rec["baseline_us"] = us(lambda: k["baseline"](*bargs))
        if isinstance(rec["helion_us"], float) and isinstance(rec["baseline_us"], float):
            rec["speedup"] = round(rec["baseline_us"] / rec["helion_us"], 3)
        st["configs"].setdefault(name, {})[hk.case_key(case)] = rec["config"]
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# per-layer-kind benchmark
# ---------------------------------------------------------------------------
def t_bench_layer(st, spec):
    """One (kind, index positions n, requests) point: the stock layer path and
    every arm of ours, in microseconds (cold L2, CUDA graph, each composite one
    graph). dense layers are the ratio-2 indexers (n = ctx / 2, logits width
    MML / 2), source / consumer the ratio-1 ones (n = ctx, width MML)."""
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_candidates as dc
    from suffix_hybrid.kernels import dsa_indexer as di
    kind, n, reqs = spec
    dev, us = _dev(), _timer()
    width = MML // 2 if kind == "dense" else MML
    pr = _decode_pr(reqs, n)
    rows, n_sms = pr["rows"], di._sms(dev)
    r = {"kind": kind, "n": n, "reqs": reqs, "rows": rows, "width": width}
    row_args = hdi.decode_args(pr, width, n_sms)
    q, w, kv, ks, bt, cl = row_args[:6]
    cl1 = pr["cl"].reshape(-1)
    nn = pr["next_n"]
    topk = _Topk(dev)
    idx = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
    ox = torch.full((rows, width), float("-inf"), device=dev)
    gcase = hdi.group_serving_case(HEADS, rows, G, n)                  # the context's own bucket
    gcase_s = hdi.group_serving_case(HEADS, rows, G, None)             # serving: the longest bucket
    gcfg = _cfg(st, hdi.GROUP, gcase) or _any_cfg(st, hdi.GROUP)
    gcfg_s = _cfg(st, hdi.GROUP, gcase_s) or _any_cfg(st, hdi.GROUP)
    rcfg = _cfg(st, hdi.DECODE, hdi.serving_case(hdi.DECODE, HEADS, rows)) or _any_cfg(st, hdi.DECODE)
    gargs = hdi.group_args(row_args, G)
    gargs_s = hdi.group_args(hdi.decode_args(pr, width, n_sms), G)
    gkern, gkern_s, rkern = (_kern(hdi.dsa_group_logits, gcfg), _kern(hdi.dsa_group_logits, gcfg_s),
                             _kern(hdi.dsa_decode_logits, rcfg))
    # (logits launcher, the buffer it fills): every top-k / mask below runs on the arm's own output
    fams = {"oxide": (lambda: hdi.decode_baseline(ox, pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], nn), ox),
            "row": (lambda: rkern(*row_args), row_args[6]),
            "group": (lambda: gkern(*gargs), gargs[6]),
            "groupS": (lambda: gkern_s(*gargs_s), gargs_s[6])}
    for arm, (fn, _) in fams.items():
        r[f"{arm}_logits_us"] = us(fn)
    r["group_vs_oxide_rel"] = _rel(gargs[6], ox, cl1)
    r["topk_us"] = us(lambda: topk(ox, cl1, 1, idx, TOPK, n))
    try:
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
    except ImportError:
        cb = None
    cand = hdi.cand_problem(pr, CAND, CBS, seed=n)
    out_c = torch.empty((rows, CAND), dtype=torch.int32, device=dev)
    fast = dc.select_blocks_fn(cb, lambda backend: topk, cb.select_candidate_blocks) if cb is not None else None
    extra, extra_fast = None, None
    if cb is not None and kind == "consumer":
        extra = lambda buf: cb.apply_candidate_mask(buf, None, cl1, cand, CBS)  # noqa: E731
    elif cb is not None and kind == "source":
        extra = lambda buf: cb.select_candidate_blocks(buf, None, cl1, CAND, CBS, out_c)  # noqa: E731
        extra_fast = lambda buf: fast(buf, None, cl1, CAND, CBS, out_c)  # noqa: E731

    def composite(logits_fn, buf, ex):
        def run():
            logits_fn()
            if ex is not None:
                ex(buf)
            topk(buf, cl1, 1, idx, TOPK, n)
        return run
    # ---- stock: vendored DeepGEMM logits (its own output tensor) -> mask / block selection -> top-k
    try:
        dga = hdi.decode_make_dg_from(pr, n_sms)

        def stock():
            lg = hdi.decode_baseline_dg(*dga)
            if extra is not None:
                extra(lg)
            topk(lg, cl1, 1, idx, TOPK, n)
        r["dg_logits_us"] = us(lambda: hdi.decode_baseline_dg(*dga))
        r[STOCK] = us(stock, 10)
        r["stock_parts"] = {"logits_us": r["dg_logits_us"], "topk_us": r["topk_us"],
                            "mask_or_select_us": us(lambda: extra(ox), 10) if extra else 0.0}
    except Exception as exc:  # noqa: BLE001
        r["dg_logits_us"] = r[STOCK] = f"{type(exc).__name__}: {str(exc)[:160]}"
    # ---- ours: every logits arm with the layer kind's selection, one graph each
    for arm, (fn, buf) in fams.items():
        r[f"{arm}_us"] = us(composite(fn, buf, extra), 10)
        if extra_fast is not None:
            r[f"{arm}_candsel_us"] = us(composite(fn, buf, extra_fast), 10)
    if kind == "source" and cb is not None:
        r["select_blocks_stock_us"] = us(lambda: extra(ox), 10)
        r["select_blocks_fast_us"] = us(lambda: extra_fast(ox), 10)
    if kind == "consumer":
        ccase = hdi.serving_case(hdi.CANDK, HEADS, rows, ncand=CAND, bsz=CBS)
        ckern = _kern(hdi.dsa_cand_logits, _cfg(st, hdi.CANDK, ccase) or _any_cfg(st, hdi.CANDK))
        cout = torch.full((rows, CAND * CBS), float("-inf"), device=dev)
        cargs = (q, w, kv, ks, bt, cl, cand, cout, *row_args[7:10], hdi.splits_for(rows, CAND * CBS, n_sms))
        lens = hdi.cand_lens(cl1, CAND, CBS).to(torch.int32)
        idx_c = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
        bound = min(CAND * CBS, n + CBS)

        def fused():
            ckern(*cargs)
            topk(cout, lens, 1, idx_c, TOPK, bound)
            dc.cand_remap(idx_c, cand, cl1, CBS)
        r["cand_logits_us"] = us(lambda: ckern(*cargs))
        r["cand_fused_us"] = us(fused, 10)
        try:   # same selected sets as the dense path?
            composite(fams["groupS"][0], fams["groupS"][1], extra)()
            want = _hash_rows(idx.cpu())
            fused()
            r["fused_topk_rows_differ"] = sum(a != b for a, b in zip(want, _hash_rows(idx_c.cpu())))
        except Exception as exc:  # noqa: BLE001
            r["fused_topk_rows_differ"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if kind == "dense":
        r.update(_bench_select(st, pr, gargs, gargs[6], cl1, n, topk, idx, us))
    torch.cuda.empty_cache()
    return r


def _bench_select(st, pr, gargs, dense, cl1, n, topk, idx, us):
    """Warm-start survivor select (kernels/dsa_select.py) vs the dense group path
    at several previous-top-k qualities: the K4 number for that design."""
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_candidates as dc
    from suffix_hybrid.kernels import dsa_indexer as di
    from suffix_hybrid.kernels import dsa_select as ds
    dev = dense.device
    rows, width = dense.shape
    q4, w3, kv, ks, bt, cl, _, page, prow, sp, splits = gargs
    out = {}
    gkern = _kern(hdi.dsa_group_logits, _cfg(st, hdi.GROUP, hdi.group_serving_case(HEADS, rows, G, n))
                  or _any_cfg(st, hdi.GROUP))
    gkern(*gargs)                                      # dense logits of the current step
    logits = dense
    skern = _kern(hdi.dsa_group_select, _cfg(st, hdi.SELECT, hdi.select_case(HEADS, pr["rows"] // G, n))
                  or _any_cfg(st, hdi.SELECT))
    ckern = _kern(hdi.dsa_cand_logits, _any_cfg(st, hdi.CANDK))
    out["select_dense_us"] = us(lambda: (gkern(*gargs), topk(dense, cl1, 1, idx, TOPK, n)), 10)
    best = None
    for stale in (0.0, 0.1, 0.5):
        g = torch.Generator(device=dev).manual_seed(int(stale * 100) + 1)
        prev = torch.full((rows, ds.K), -1, dtype=torch.int64, device=dev)
        for r in range(rows):
            m = min(int(cl1[r]), width)
            top = logits[r, :m].topk(min(ds.K, m)).indices
            drop = int(top.numel() * stale)
            prev[r, :top.numel()] = torch.cat([top[drop:], torch.randint(0, m, (drop,), generator=g, device=dev)])[:top.numel()]
        pos = ds.warm_positions(prev, cl1.to(torch.int64))
        m_ = pos.shape[1]
        wout = torch.full((rows, m_), float("-inf"), device=dev)
        cand32 = pos.to(torch.int32).contiguous()
        wargs = (q4.reshape(rows, HEADS, 128), w3.reshape(rows, HEADS), kv, ks, bt, cl, cand32, wout,
                 page, prow, sp, hdi.splits_for(rows, m_, di._sms(dev)))
        cv = torch.full((rows, width), float("nan"), device=dev)
        ci = torch.full((rows, width), -1, dtype=torch.int32, device=dev)
        cn = torch.zeros(rows, dtype=torch.int32, device=dev)
        idx2 = torch.empty_like(idx)

        def pipeline():
            wout.fill_(float("-inf"))
            ckern(*wargs)                                          # warm gather (candidate body, block size 1)
            thr = ds.warm_threshold(wout, ds.K)                    # k-th largest + slack
            cn.zero_()
            skern(q4, w3, kv, ks, bt, cl, thr, cv, ci, cn, page, prow, sp, splits)
            topk(cv, cn, 1, idx2, TOPK, n)
            dc.cand_remap(idx2, ci, cl1, 1)
        tag = f"select_stale{int(stale * 100)}_us"
        out[tag] = us(pipeline, 10)
        try:
            pipeline()
            torch.cuda.synchronize()
            gkern(*gargs)
            topk(dense, cl1, 1, idx, TOPK, n)
            out[f"select_stale{int(stale * 100)}_rows_differ"] = sum(
                a != b for a, b in zip(_hash_rows(idx.cpu()), _hash_rows(idx2.cpu())))
            out[f"select_stale{int(stale * 100)}_survivors_mean"] = round(float(cn.double().mean()), 1)
        except Exception as exc:  # noqa: BLE001
            out[f"select_stale{int(stale * 100)}_rows_differ"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        if isinstance(out[tag], float) and (best is None or out[tag] < best):
            best = out[tag]
    out["select_us"] = round(best, 2) if best is not None else None
    return out


def t_bench_prefill(st, spec):
    """m Q rows of one request whose K window ends at n (row i sees
    [0, n - m + 1 + i)): oxide (absolute columns) / Helion compressed /
    Triton compressed / vLLM DeepGEMM."""
    import inspect
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_indexer as di
    m, n = spec
    dev, us = _dev(), _timer()
    g = torch.Generator(device=dev).manual_seed(n)
    q = (torch.randn((m, HEADS, 128), generator=g, device=dev) * 2).to(torch.float8_e4m3fn)
    kv = (torch.randn((n, 128), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ksc = 2.0 ** torch.randint(-8, -3, (n,), generator=g, device=dev).float()
    w = torch.randn((m, HEADS), generator=g, device=dev) * 0.02
    lo = torch.zeros(m, dtype=torch.int32, device=dev)
    hi = (n - m + 1 + torch.arange(m, device=dev)).to(torch.int32)
    r = {"m": m, "n": n}
    ab = torch.full((m, n), float("-inf"), device=dev)
    r["oxide_us"] = us(lambda: hdi.prefill_baseline(q, kv, ksc, w, lo, hi, ab, 0), iters=5)
    fb = di._fallback_mod()
    r["triton_us"] = us(lambda: fb.fp8_fp4_mqa_logits((q, None), (kv, ksc), w, lo, hi, n), iters=3)
    cfg = _cfg(st, hdi.PREFILL, hdi.serving_case(hdi.PREFILL, HEADS, m))
    if cfg:
        out = torch.full((m, n), float("-inf"), device=dev)
        kern = _kern(hdi.dsa_prefill_logits, cfg)
        pargs = (q, kv, ksc, w, lo, hi, lo, out, hdi.splits_for(-(-m // 16), n, di._sms(dev)))
        r["helion_us"] = us(lambda: kern(*pargs), iters=5)
        r["helion_vs_oxide_rel"] = _rel(out, ab, hi)
    try:
        from vllm.utils import deep_gemm as vdg
        fn = vdg.fp8_fp4_mqa_logits
        last = list(inspect.signature(fn).parameters)[-1]
        extra = n if last == "max_seqlen_k" else False
        r["deepgemm_us"] = us(lambda: fn((q, None), (kv, ksc), w, lo, hi, extra), iters=5)
    except Exception as exc:  # noqa: BLE001
        r["deepgemm_us"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    torch.cuda.empty_cache()
    return r


def t_gate(st, _):
    """Boot gate dsa_indexer_capture_replay with the bounded default Helion configs
    (nothing is shipped yet), plain and -- when the tool exists -- under
    compute-sanitizer memcheck (--quick). Subprocesses: the gate patches the config
    lookup globally and must not leak into the tasks after it."""
    import shutil
    import subprocess
    cmd = [sys.executable, "-m", "suffix_hybrid.kernels.dsa_indexer", "capture_replay", "--default-configs"]
    res = {}
    for name, argv, timeout in (("default_configs", cmd, 3600),
                                ("memcheck_quick", ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
                                                    *cmd, "--quick"], 7200)):
        if name == "memcheck_quick" and shutil.which("compute-sanitizer") is None:
            res[name] = "skipped: compute-sanitizer not installed in this image"
            continue
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            res[name] = {"rc": p.returncode, "tail": (p.stderr or p.stdout)[-1500:]}
        except Exception as exc:  # noqa: BLE001
            res[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def t_report(st, _):
    return report(st["results"])


RUN = {"oracle": t_oracle, "gate": t_gate, "tune": t_tune, "bench_layer": t_bench_layer,
       "bench_prefill": t_bench_prefill, "report": t_report}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    budget = float(argv[0]) if argv else 1100.0
    try:  # Helion's LFBO search when scikit-learn loads, like the autotune runner
        import sklearn  # noqa: F401
        os.environ.setdefault("HELION_AUTOTUNER", "LFBOTreeSearch")
    except Exception:  # noqa: BLE001
        os.environ.setdefault("HELION_AUTOTUNER", "PatternSearch")
    from suffix_hybrid import helion_kernels as hk
    hk.ensure_vendor()
    st, todo, t0 = _load(), tasks(), time.time()
    _emit({"env": {"vllm": __import__("vllm").__version__, "helion": getattr(
        __import__("helion"), "__version__", "?"), "done": st["done"], "of": len(todo)}})
    while st["done"] < len(todo) and time.time() - t0 < budget:
        name, spec = todo[st["done"]]
        t1 = time.time()
        try:
            res = RUN[name](st, spec)
        except Exception as exc:  # noqa: BLE001 - recorded, the queue moves on
            import traceback
            res = {"error": f"{type(exc).__name__}: {exc}"[:600],
                   "where": traceback.format_exc()[-600:]}
        row = {"task": [name, spec], "s": round(time.time() - t1), "result": res}
        st["results"].append(row)
        st["done"] += 1
        _save(st)
        _emit(row)
    if st["done"] >= len(todo):  # everything ran: the report again, then page through the stored results
        _emit({"report": report(st["results"])})
        c = st.get("cursor", 0)
        for row in st["results"][c:c + 3]:
            _emit({"replay": c, **row})
            c += 1
        st["cursor"] = c % max(1, len(st["results"]))
        _save(st)
    _emit({"progress": f"{st['done']}/{len(todo)}", "elapsed_s": round(time.time() - t0)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
