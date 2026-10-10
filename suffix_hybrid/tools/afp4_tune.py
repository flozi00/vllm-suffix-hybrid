# SPDX-License-Identifier: Apache-2.0
"""Tune AITER's Triton MXFP4 GEMM for this card (boot gate afp4_tune; SUFFIX_ROCM_AFP4_CONFIGS).

vLLM's dense MXFP4 linears (non-ASM AITER path) run dynamic_mxfp4_quant + gemm_afp4wfp4 (+ a
split-K reduce). AITER picks the GEMM config from JSONs keyed by arch + (N, K) + M bucket, with no
CU count, and ships none of Qwen3.8-Flash-Next's shapes: the 128-CU MI350P runs the gfx950
DEFAULT.json tiles tuned on the 256-CU MI355X (c8 verify M=40: 32x32x512, ~20 us/call for a
2-6 us weight read). This times candidate configs the way vLLM calls the op (quant + GEMM +
reduce, one HIP graph over ~1 GiB of weight copies: every weight read is cold), checks each
against AITER's own config (bf16 bound, graph replay == eager) and prints per (N, K) a table of
AITER's config vs the best (us per M) and the AITER-format JSON between
"[suffix afp4-tune] JSON BEGIN N=.. K=.." and "[suffix afp4-tune] JSON END": a top-level
M_BOUNDS (AITER's loader honours it) with one M_LEQ_<M> per tuned M, then AITER's own entries
above the largest tuned M, so prefill keeps AITER's tiles. A bucket keeps AITER's config unless
the best is >= 3% faster on a 20-replay re-check. Ship it as
suffix_hybrid/configs/afp4/GEMM-AFP4WFP4-N=<N>-K=<K>.json and set SUFFIX_ROCM_AFP4_CONFIGS=1;
boot gate afp4_tune_shipped reruns this with the gate on ("json" marks the shipped baseline).
With SUFFIX_ROCM_MXFP4_A16=1, M <= its MAX_M runs gemm_a16wfp4 instead of these buckets.

Search per (N, K, M). Stage A: BLOCK_M tied to M x BLOCK_N 16..256 x BLOCK_K 256..1024 x
NUM_KSPLIT 1..8 (only splits AITER's get_splitk keeps) x GROUP_SIZE_M 1/4 (several M tiles),
nonkdim 16, the other knobs from AITER's config; pruned to EVEN_K, >= CUs/2 workgroups, split-K
only under 2 waves of tiles and with fp32 partials <= half the weight bytes. Stage B: one-knob
variations of the best two (num_warps 2/4/8, waves_per_eu 1/2/4, cache_modifier, nonkdim 32,
num_stages 1..3).
~1400 + ~500 candidates for 3 shapes x 10 Ms; each stage compiles in a process pool first
(Triton's disk cache), then this process times. --minutes bounds the whole run.

    python -m suffix_hybrid.tools.afp4_tune [--minutes 24] [--jobs 16] [--ms 5,40]
                                            [--shapes 2560x6144]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import time

MARK = "[suffix afp4-tune]"
# (N, K) of the live dense MXFP4 linears; out_proj (GDN) and o_proj (attention) share 2560x6144.
SHAPES = ((16384, 2560, "in_proj_qkvz"), (2560, 6144, "out_proj/o_proj"), (13312, 2560, "qkv_proj"))
MS = (1, 5, 8, 16, 32, 40, 64, 128, 160, 256)  # MTP-4 decode: c1/c8/c32 drafts + verifies
HOT = (40, 5, 160, 8, 1, 32)  # timed first: what the clock cuts last
KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages",
        "waves_per_eu", "matrix_instr_nonkdim", "cache_modifier", "NUM_KSPLIT")  # AITER entry keys
WIN = 0.97  # a candidate replaces AITER's config only when >= 3% faster on the re-check


def say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def stage_a(m: int, n: int, k: int, base: dict, cus: int, get_splitk) -> list[dict]:
    """Tile/split grid for one (M, N, K logical) around AITER's config `base`."""
    p2 = 1 << (m - 1).bit_length()
    bms = sorted({max(4, p2), 16}) if m <= 16 else [p2 // 2, p2]
    out = []
    for bm, bn, bk, ks in itertools.product(bms, (16, 32, 64, 128, 256), (256, 512, 1024),
                                            (1, 2, 3, 4, 5, 6, 8)):
        tiles = -(-m // bm) * -(-n // bn)
        if ks == 1:
            if k % bk:  # K tail: masked loads (not EVEN_K)
                continue
        elif (get_splitk(k // 2, bk, ks)[1:] != (bk, ks)  # AITER would run another split
              or tiles >= 2 * cus or ks * m * 16 > k):  # enough tiles / partials too big
            continue
        if ks * tiles < cus // 2:
            continue
        for g in (1, 4) if ks == 1 and m > bm else (1,):
            out.append(dict(base, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                            GROUP_SIZE_M=g, NUM_KSPLIT=ks, matrix_instr_nonkdim=16,
                            num_stages=1 if k <= ks * bk else max(2, base["num_stages"])))
    return out


def stage_b(c: dict, k: int) -> list[dict]:
    """One-knob variations of a stage-A winner."""
    knobs = {"num_warps": (2, 4, 8), "waves_per_eu": (1, 2, 4), "cache_modifier": (None, ".cg"),
             "matrix_instr_nonkdim": (16, 32) if min(c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"]) >= 32
             else (16,),
             "num_stages": (1, 2, 3) if k > c["NUM_KSPLIT"] * c["BLOCK_SIZE_K"] else (1,)}
    return [dict(c, **{key: v}) for key, vals in knobs.items() for v in vals if v != c[key]]


def to_json(best: dict, src: dict, bounds) -> dict:
    """AITER GEMM-AFP4WFP4-N=..-K=.. content: M_LEQ_<M> per tuned M (`best`: M -> config), then
    the entries of `src` (AITER's own file for the shape) above the largest tuned M."""
    top = max(best)
    doc = {"M_BOUNDS": sorted(set(best) | set(src.get("M_BOUNDS", bounds)))}
    doc.update((f"M_LEQ_{m}", {key: best[m][key] for key in KEYS}) for m in sorted(best))
    doc.update((key, v) for key, v in src.items()
               if key != "M_BOUNDS" and not (key.startswith("M_LEQ_") and int(key[6:]) <= top))
    return doc


def _dumps(doc: dict) -> str:  # one bucket per line
    return "{\n" + ",\n".join(f"  {json.dumps(key)}: {json.dumps(v)}"
                               for key, v in doc.items()) + "\n}"


def _fmt(c: dict) -> str:
    return (f"{c['BLOCK_SIZE_M']}x{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']} ks{c['NUM_KSPLIT']} "
            f"g{c['GROUP_SIZE_M']} w{c['num_warps']} s{c['num_stages']} e{c['waves_per_eu']} "
            f"i{c['matrix_instr_nonkdim']} {c['cache_modifier'] or '-'}")


def _key(c: dict) -> str:
    return json.dumps(c, sort_keys=True)


def _spec(k: int, m: int, c: dict) -> tuple:
    """What Triton compiles the GEMM for here (every N and stride is a multiple of 16)."""
    return k, m == 1, m % 16 == 0, _key(c)


_DATA: dict = {}


def _data(n: int, k: int, m: int):
    """x [M, K] bf16, w [N, K/2] e2m1 pairs, ws [K/32, N] e8m0: vLLM's non-ASM layouts as loaded."""
    import torch
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    if ("w", n, k) not in _DATA:
        g = torch.Generator(device="cuda").manual_seed(n + k)
        w, ws = dynamic_mxfp4_quant(
            torch.randn(n, k, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02)
        _DATA["w", n, k] = w, ws.T.contiguous()  # process_weights_after_loading: [K/32, N]
    if ("x", m, k) not in _DATA:  # per-32-group scales 2^-6..2^6: many E8M0 exponents
        g = torch.Generator(device="cuda").manual_seed(m * k + 1)
        _DATA["x", m, k] = (torch.randn(m, k // 32, 32, generator=g, device="cuda") * torch.exp2(
            torch.randint(-6, 7, (m, k // 32, 1), generator=g, device="cuda").float())
                            ).reshape(m, k).to(torch.bfloat16)
    return (_DATA["x", m, k], *_DATA["w", n, k])


def _run(x, w, ws, cfg):
    """vLLM 81198e97 gemm_with_dynamic_quant, non-ASM branch; cfg None = AITER's own config."""
    import torch
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    x_q, x_s = dynamic_mxfp4_quant(x)
    y = torch.empty(x_q.shape[0], w.shape[0], device=x_q.device, dtype=torch.bfloat16)
    gemm_afp4wfp4(x_q, w, x_s, ws.T, torch.bfloat16, y, cfg)
    return y


def _warm(jobs, deadline: float = float("inf")) -> list:
    """Run each (n, k, m, cfg) once, so Triton compiles it into its disk cache (pool worker).
    Error text or None per job; stops at the deadline (shorter list)."""
    import torch

    out = []
    for n, k, m, cfg in jobs:
        if time.monotonic() > deadline:
            break
        try:
            _run(*_data(n, k, m), cfg)
            torch.cuda.synchronize()
            out.append(None)
        except Exception as e:  # noqa: BLE001 - OutOfResources / compile error: candidate skipped
            out.append(f"{type(e).__name__}: {e}".split("\n")[0][:120])
    return out


def _warm_all(pool, jobs, deadline: float) -> dict:
    """Compile `jobs` (one per Triton specialization, in order) in the pool: spec -> error/None."""
    todo: dict = {}
    for n, k, m, c in jobs:
        todo.setdefault(_spec(k, m, c), (n, k, m, c))
    reps = list(todo.values())
    chunks = [reps[i::256] for i in range(256)] if pool else [reps]  # strided: hot Ms first in each
    try:
        runs = list(pool.map(_warm, chunks, itertools.repeat(deadline))) if pool else [
            _warm(reps, deadline)]
    except Exception as e:  # noqa: BLE001 - e.g. a broken pool: compile here instead
        say(f"compile pool failed ({e!r}); compiling in this process")
        chunks, runs = [reps], [_warm(reps, deadline)]
    return {_spec(*job[1:]): err
            for chunk, errs in zip(chunks, runs) for job, err in zip(chunk, errs)}


def _close(out, ref) -> bool:
    """Same quantized operands, other fp32 summation order: within one bf16 ulp + a floor."""
    import torch

    out, ref = out.float(), ref.float()
    tol = 2 ** -7 * torch.maximum(out.abs(), ref.abs()) + 2 ** -12 * ref.abs().max()
    return bool(torch.isfinite(out).all()) and bool(((out - ref).abs() <= tol).all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=24.0, help="wall budget of the whole run")
    ap.add_argument("--jobs", type=int, default=min(16, len(os.sched_getaffinity(0))),
                    help="compile worker processes (<= 1: compile in this process)")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--shapes", default=",".join(f"{n}x{k}" for n, k, _ in SHAPES), help="NxK,...")
    a = ap.parse_args()
    t0 = time.monotonic()

    def until(frac: float) -> float:
        return t0 + frac * a.minutes * 60

    import torch
    import triton
    from aiter.ops.triton.gemm_afp4wfp4 import _get_config, get_splitk
    from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
    from aiter.ops.triton.utils.gemm_config_utils import STANDARD_M_BOUNDS

    from suffix_hybrid.kernels.mxfp4_a16_rocm import _graph_us

    names = {(n, k): name for n, k, name in SHAPES}
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    ms = sorted({int(m) for m in a.ms.split(",")})
    pairs = [(n, k, m) for m in sorted(ms, key=lambda m: HOT.index(m) if m in HOT else len(HOT))
             for n, k in shapes]
    props = torch.cuda.get_device_properties(0)
    cus = props.multi_processor_count
    say(f"{props.gcnArchName} {cus} CUs, triton {triton.__version__}; N x K {shapes}; M {ms}; "
        f"{a.jobs} compile workers; budget {a.minutes:g} min")
    pool = None
    if a.jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn"))

    base, tuned, meas, bad = {}, {}, {}, collections.Counter()
    for n, k, m in pairs:
        base[n, k, m], tuned[n, k, m] = _get_config(m, n, k // 2)
        meas[n, k, m] = {}
    copies = {}  # ~1 GiB of each weight: no read hits L2 / MALL, as in a decode step
    for n, k in shapes:
        _, w, ws = _data(n, k, 1)
        copies[n, k] = [(w.clone(), ws.clone())
                        for _ in range(max(1, min(512, (1 << 30) // w.numel())))]
    refs, errors, late = {}, collections.Counter(), [0]

    def measure(p, cfgs, status, stop):
        n, k, m = p
        x, wc = _data(n, k, m)[0], copies[n, k]
        if p not in refs:
            refs[p] = _run(x, *wc[0], None)  # AITER's own config, as vLLM calls it
        for c in cfgs:
            if _key(c) in meas[p]:
                continue
            if time.monotonic() > stop:
                late[0] += 1
                continue
            err = status.get(_spec(k, m, c), "not compiled (clock)")
            if err is None:
                try:
                    eager = _run(x, *wc[0], c)
                    us, replayed = _graph_us(lambda x, w, s: _run(x, w, s, c), x, wc)
                    err = None if _close(eager, refs[p]) and torch.equal(replayed, eager) else (
                        "output differs from AITER's config or replay != eager")
                except Exception as e:  # noqa: BLE001 - launch failure: candidate skipped
                    err = f"{type(e).__name__}: {e}".split("\n")[0][:120]
            if err is None:
                meas[p][_key(c)] = us
            else:
                bad[p] += 1
                errors[err] += 1

    for stage, warm_frac, time_frac in (("A", 0.35, 0.6), ("B", 0.75, 0.92)):
        if stage == "A":
            cand = {p: [base[p]] + stage_a(p[2], p[0], p[1], base[p], cus, get_splitk)
                    for p in pairs}
        else:
            cand = {p: [c for key in sorted(meas[p], key=meas[p].get)[:2]
                        for c in stage_b(json.loads(key), p[1])] for p in pairs}
        jobs = [(*p, c) for p in pairs for c in cand[p]]
        status = _warm_all(pool, jobs, until(warm_frac))
        say(f"stage {stage}: {len(jobs)} candidates, {len(status)} compiled "
            f"({sum(e is not None for e in status.values())} failed) at "
            f"{(time.monotonic() - t0) / 60:.1f} min")
        for p in pairs:
            measure(p, cand[p], status, until(time_frac))
        say(f"stage {stage}: {sum(map(len, meas.values()))} timed at "
            f"{(time.monotonic() - t0) / 60:.1f} min")
    if pool:
        pool.shutdown(cancel_futures=True)
    for err, cnt in errors.most_common(6):
        say(f"skipped {cnt}x: {err}")
    if late[0]:
        say(f"{late[0]} candidates not timed: clock (--minutes {a.minutes:g})")

    cfg_dir = resolve_config_dir("gemm", "GEMM-AFP4WFP4")
    for n, k in shapes:
        say(f"N={n} K={k} {names.get((n, k), '')}: us/call (quant + GEMM + reduce, HIP graph, "
            "cold weights, 20 replays); json = AITER config came from a JSON for this N/K")
        say(f"{'M':>5} | {'AITER config':<40} {'us':>6} | {'best':<35} {'us':>6} | "
            "speedup | ok/failed")
        best = {}
        for m in ms:
            p = (n, k, m)
            x, wc = _data(n, k, m)[0], copies[n, k]
            c = json.loads(min(meas[p], key=meas[p].get, default=_key(base[p])))
            try:
                t_def = _graph_us(lambda x, w, s: _run(x, w, s, None), x, wc, reps=20)[0]
                t = t_def if c == base[p] else _graph_us(
                    lambda x, w, s: _run(x, w, s, c), x, wc, reps=20)[0]
            except Exception as e:  # noqa: BLE001 - keep AITER's config, still print the rest
                say(f"M={m}: re-check failed ({type(e).__name__}: {e}); keeping AITER's config")
                t_def = t = float("nan")
            best[m], t = (c, t) if t <= WIN * t_def else (base[p], t_def)
            say(f"{m:5d} | {_fmt(base[p]):<35} {'json' if tuned[p] else '':4} {t_def:6.1f} | "
                f"{_fmt(best[m]):<35} {t:6.1f} | {t_def / t:6.2f}x | {len(meas[p])}/{bad[p]}")
        src = load_config_json(f"{cfg_dir}/GEMM-AFP4WFP4-N={n}-K={k}.json", required=False
                               ) or load_config_json(f"{cfg_dir}/DEFAULT.json")
        say(f"JSON BEGIN N={n} K={k} (suffix_hybrid/configs/afp4/GEMM-AFP4WFP4-N={n}-K={k}.json)")
        print(_dumps(to_json(best, src, STANDARD_M_BOUNDS)), flush=True)
        say("JSON END")
    say(f"done in {(time.monotonic() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
