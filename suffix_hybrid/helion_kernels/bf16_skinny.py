# SPDX-License-Identifier: Apache-2.0
"""Helion BF16 skinny GEMM with deterministic two-stage split-K, for the small-N /
large-K BF16 linears cuBLAS serves badly at decode M (kernels/bf16_skinny.py is
the serving lane, knob SUFFIX_HELION_BF16_SKINNY).

    y[M, N] = bf16(sum_k x[m, k] * w[n, k])   x bf16 [M, K], w bf16 [N, K] (rows contiguous)

Target: Qwen3-Next / Qwen4Exp MoE blocks on SM120 (qwen-flash TP2+EP, vLLM 0.32.0):
  mlp.gate               GateLinear(hidden -> experts), replicated, out_dtype None on SM120
                         -> tier 6 = ReplicatedLinear -> UnquantizedLinearMethod.apply -> F.linear
                         (N 512, K 2560: cuBLAS wmma 16x16 tiles, ~32 CTAs on 188 SMs)
  mlp.shared_expert_gate ReplicatedLinear(hidden -> 1, quant_config=None) -> F.linear -> cuBLAS
                         gemv + splitKreduce (N 1, K 2560: two launches for 5 KB of weight)
The MTP drafter's MoE layer has the same two (Qwen4ExpMTP builds Qwen4ExpDecoderLayer).

Precision contract = cuBLAS bf16 GEMM: bf16 x bf16 products (exact in fp32), fp32
accumulation, ONE bf16 rounding of the fp32 total. Split-K is TWO-STAGE and capture-safe
(the mxfp8_gemm pattern): stage 1 (the Helion kernel) stores each split's fp32 partial
to partial[split, M, N]; stage 2 is a host-side torch reduce after it on the same stream
(partial[:n_split].sum(0) in a fixed order, copied into out = the one rounding). The
partial comes from torch's stream-ordered allocator per call (graph-private under
capture): no atomics, no semaphore, no counters, no module-level scratch. split_k = 1
writes out directly and never touches the partial. Not batch invariant (the M bucket
picks the config), exactly like cuBLAS.

KERNELS (tools/helion_tune.py ABI): one entry, baseline F.linear (= cuBLAS, what
UnquantizedLinearMethod runs on CUDA). Serving binds the shipped config of the M bucket
(helion_kernels.compiled / bucket / config); a bucket without one stays on F.linear.
"""
from __future__ import annotations

import re

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import ensure_vendor

ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no Helion: cases, plans and references still import
    helion = hl = PowerOfTwoFragment = None

KERNEL = "bf16_skinny"
MARKER = "[suffix bf16-skinny helion]"
# Decode M per rank: MTP k=4 verify = 5 x concurrency (5 / 40 / 160 at c1 / c8 / c32),
# drafter steps 1 x concurrency; 192 = the lane's M cap.
BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192)
MAX_M = BUCKETS[-1]
SPLIT_COLS = 4 * 188 * 32  # ~4 CTAs per SM at 32 columns (mxfp8_gemm.SPLIT_COLS)
# Qwen MoE blocks with a BF16 router (GateLinear tier 6 on SM120) + shared_expert_gate
_QWEN_MOE = re.compile(r"Qwen(2Moe|3Next|3_5Moe|4Exp)")


# ---------------------------------------------------------------------------
# Helion body (literals only inside; tests run it in interpret mode against the
# exact reference and lower it to Triton with Helion's real front end).
# ---------------------------------------------------------------------------
def bf16_skinny_body(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor,
                     partial: torch.Tensor) -> torch.Tensor:
    """x bf16 [M, K], w bf16 [N, K], out bf16 [M, N]; partial f32 [cap, M, N] (cap =
    the largest split_k allowed; serving allocates the tuned split per call, or
    [1, 1, 1] when it does not split: never read then). Writes and returns out."""
    m = x.size(0)
    n = hl.specialize(w.size(0))
    k = hl.specialize(w.size(1))
    cap = hl.specialize(partial.size(0))
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, cap, 1))
    k_split = helion.next_power_of_2(helion.cdiv(k, split_k))
    bm = hl.register_block_size(16, 256)
    bn = hl.register_block_size(16, 128)
    bk = hl.register_block_size(32, 256)
    n_split = helion.cdiv(k, k_split)
    for tile_m, tile_n, tile_s in hl.tile([m, n, k], block_size=[bm, bn, k_split]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin, tile_s.end, block_size=bk):
            acc = hl.dot(x[tile_m, tile_k], w[tile_n, tile_k].T, acc=acc)
        if n_split == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            partial[tile_s.id, tile_m, tile_n] = acc
    if n_split > 1:
        # stage 2: a separate launch on the same stream (stream order is the only
        # synchronisation), fp32 sum over the splits in a fixed order, one bf16 rounding
        out.copy_(partial[:n_split].sum(0))
    return out


def _ignore() -> list:
    """Helion warns that the stage-2 torch reduce is not fused into the kernel: intended."""
    return [helion.exc.TensorOperationInWrapper] if helion is not None else []


def max_splits(n: int, k: int) -> int:
    """Split-K cap: ~4 CTAs per SM over (N / 32) x split, a power of two in [2, 64],
    and no split shorter than 64 elements of K."""
    want = -(-SPLIT_COLS // max(1, n))
    cap = min(64, max(2, 1 << (want - 1).bit_length()))
    while cap > 2 and k // cap < 64:
        cap //= 2
    return cap


# ---------------------------------------------------------------------------
# cases (the tuner's per-rank model dicts) + serving plan
# ---------------------------------------------------------------------------
def case_of(n: int, k: int, b: int) -> dict:
    return {"k": k, "n": n, "numtokens": b}


def shapes(model: dict) -> list:
    """[(N, K)] of the BF16 router linears this family serves for a model dict: the router
    gate (replicated: ALL experts on every rank, EP or not) and, with a Qwen shared expert,
    its N = 1 shared_expert_gate."""
    h, e = model.get("hidden"), model.get("experts_global")
    if not (h and e and _QWEN_MOE.search(str(model.get("architecture") or ""))):
        return []
    return [(e, h)] + ([(1, h)] if model.get("shared_expert_intermediate") else [])


def cases(model: dict) -> list:
    return [case_of(n, k, b) for b in BUCKETS for n, k in shapes(model)]


def prepare(n: int, k: int, max_m: int, mode: str) -> list:
    """[None] + per M = 1..max_m: (compiled kernel, split_k, bucket) or None (F.linear
    serves). mode "auto": Helion where a config is shipped; "helion": a bucket without
    one is a startup error."""
    if mode not in ("auto", "helion"):
        raise RuntimeError(f"{MARKER} prepare: mode {mode!r} (the Helion lane is off)")
    if not 1 <= max_m <= MAX_M:
        raise RuntimeError(f"{MARKER} prepare: max_m {max_m} outside [1, {MAX_M}]")
    per = {}
    for b in sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)}):
        case = case_of(n, k, b)
        cfg = hk.config(KERNEL, case)
        if cfg is None:
            if mode == "helion":
                raise RuntimeError(
                    f"{MARKER} SUFFIX_HELION_BF16_SKINNY=helion: no tuned {KERNEL} config for "
                    f"N={n} K={k} M bucket {b} on {hk.platform()} (tune it with "
                    f"suffix_hybrid.tools.helion_tune --families {KERNEL} and ship "
                    f"configs/helion/{KERNEL}/<platform>.json, or set auto)")
            per[b] = None
            continue
        per[b] = (hk.compiled(bf16_skinny_body, KERNEL, case, ignore_warnings=_ignore()),
                  int(cfg.get("split_k", 1)), b)
    return [None] + [per[hk.bucket(m, BUCKETS)] for m in range(1, max_m + 1)]


def run(x: torch.Tensor, w: torch.Tensor, entry, poison_nan: bool = False) -> torch.Tensor:
    """The lane's GPU work for one call: out bf16 [M, N] = x @ w.T via a plan entry. The
    only allocations are out and the fp32 partial (torch's stream-ordered allocator,
    graph-private under capture). poison_nan (checks only): NaN-fill both first, so a
    tile or split a kernel forgets to write fails the exact check."""
    kern, split, _ = entry
    m, n = x.shape[0], w.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    partial = torch.empty((split, m, n) if split > 1 else (1, 1, 1), dtype=torch.float32,
                          device=x.device)
    if poison_nan:
        out.fill_(float("nan"))
        partial.fill_(float("nan"))
    return kern(x, w, out, partial)


def describe(plan: list) -> str:
    """'M1:helion(split 4) M2-8:F.linear ...' (K5 load log)."""
    runs, last = [], object()
    for m, e in enumerate(plan[1:], 1):
        name = (f"helion(b{e[2]}" + (f",split {e[1]})" if e[1] > 1 else ")")) if e else "F.linear"
        if name == last:
            runs[-1][1] = m
        else:
            runs.append([m, m, name])
            last = name
    return " ".join(f"M{a}{'' if a == b else f'-{b}'}:{s}" for a, b, s in runs)


def oracle_ms(plan: list) -> list:
    """Ms that check (and JIT-compile) every specialization a plan can launch: per served
    bucket its first and last row count and an odd one between (Triton specializes M == 1
    and M % 16 == 0; every bucket >= 16 is a multiple of 16)."""
    out, lo = set(), 1
    for m in range(1, len(plan)):
        e = plan[m]
        if m == len(plan) - 1 or plan[m + 1] is not e:
            if e is not None:
                out |= {lo, m, (lo + m) // 2 | 1 if lo < m else m}
            lo = m + 1
    return sorted(x for x in out if 1 <= x < len(plan))


# ---------------------------------------------------------------------------
# reference + checks (pure torch: CPU tests, the lane's load oracle, the tuner)
# ---------------------------------------------------------------------------
def exact(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.double() @ w.double().T


def bound(x: torch.Tensor, w: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-element error bound of ANY fp32-accumulated bf16 GEMM with one final bf16
    rounding: half an ulp of bf16 (2^-8 relative) on top of the fp32 summation error
    (K * 2^-24 * sum_k |x||w|, any order or split), doubled for the rounding of that."""
    k = x.shape[-1]
    s = x.double().abs() @ w.double().abs().T
    return 2.0 ** -8 * ref.abs() + 2.0 * k * 2.0 ** -24 * s


def check_exact(y, x, w, ref=None):
    """(ok, msg): y within `bound` of the exact product everywhere, finite."""
    ref = exact(x, w) if ref is None else ref
    yf = y.double()
    if not bool(torch.isfinite(yf).all()):
        return False, "non-finite output"
    err = (yf - ref).abs()
    lim = bound(x, w, ref)
    bad = int((err > lim).sum())
    worst = float((err / lim.clamp_min(1e-300)).max()) if err.numel() else 0.0
    return bad == 0, f"{bad} of {err.numel()} past the bf16 GEMM bound (worst {worst:.2f} of it)"


def check(out, ref):
    """Tuner check (out, exact ref) without the operands: every element within 2^-8 |ref|
    + 2^-10 max|ref| (one bf16 rounding plus a generous fp32-accumulation floor). A missing
    split, tile or column is off by ~|ref|: caught."""
    o, r = out.double(), ref.double()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite output"
    err = (o - r).abs()
    bad = int((err > 2.0 ** -8 * r.abs() + 2.0 ** -10 * r.abs().max()).sum())
    rel = float(err.norm() / r.norm().clamp_min(1e-300))
    return bad == 0, f"rel {rel:.2e}, {bad} elements past 2^-8 |ref| + 2^-10 max|ref|"


# ---------------------------------------------------------------------------
# tuner entry (tools/helion_tune.py ABI)
# ---------------------------------------------------------------------------
def problem(case: dict, seed: int, device="cuda") -> tuple:
    """(x, w, out, partial): router-like operands, x ~ N(0, 1) (post-RMSNorm hidden),
    w ~ N(0, 0.02^2); out zeros and partial zeros (Helion's autotuner compares args
    before / after the baseline run: NaN garbage would pin them as mutated)."""
    m, n, k = case["numtokens"], case["n"], case["k"]
    gen = torch.Generator(device=device).manual_seed(1000 + 7919 * seed + n + 31 * k)
    x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=gen, device=device) * 0.02).to(torch.bfloat16)
    return (x, w, torch.zeros(m, n, dtype=torch.bfloat16, device=device),
            torch.zeros(max_splits(n, k), m, n, dtype=torch.float32, device=device))


def reference(x, w, out, partial):
    return exact(x, w)


def baseline(x, w, out, partial):
    return torch.nn.functional.linear(x, w)


def working_set(args) -> int:
    """Bytes one call really touches (the zeroed partial cap is not read at split 1)."""
    x, w = args[0], args[1]
    return (x.numel() + w.numel() + x.shape[0] * w.shape[0]) * 2


KERNELS = [{
    "name": KERNEL, "body": bf16_skinny_body, "cases": cases,
    "make": problem, "reference": reference, "check": check, "baseline": baseline,
    "working_set_bytes": working_set, "settings": {"ignore_warnings": _ignore()},
    "baseline_name": "F.linear (cuBLAS bf16 GEMM / gemv+splitKreduce, UnquantizedLinearMethod)",
    "replaces": ["cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8",
                 "gemvx", "splitKreduce"],
}]
