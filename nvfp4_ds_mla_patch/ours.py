# SPDX-License-Identifier: Apache-2.0
"""Our Triton sparse-MLA decode over the nvfp4_ds_mla reader cache (SM120),
SUFFIX_SM120_DSMLA_ATTN=ours|auto|oxide (default oxide = the cuda-oxide pair).

Same contract as the oxide kernels (q [T, HQ, 576] bf16, 352 B rows, top-k
[T, C] physical slots with -1 masked, out [T, HQ, 512] bf16, softmax(S *
scale) . latent, an all-masked token is 0). What it does differently:

* Work follows the live list, not the capacity: a program (token, 8/16-head
  tile, split s) takes split s of [0, L), L = last live column + 1, in whole
  TN-row tiles (DSA prefix lists of short contexts cost len rows, not 2048).
* One kernel: splits store bf16 partials + lse (log2) and the LAST arriving
  split of a (token, head tile) merges them in split order (acq_rel gpu-scope
  counter after a CTA barrier, the counter rearmed by that split): no merge
  launch, bitwise the same whoever arrives last.
* No spills by construction (rev 2): S^T = K q^T over 64-dim chunks, so K is
  the A operand of S^T and the B operand of PV in its natural row-major
  layout (no transposes of K; q^T chunks are loop-invariant shared memory,
  P is transposed once per tile), 8 output accumulators of [16, 64], raw
  e2m1/e4m3 bytes staged through shared memory and dequantized after the
  operand load (cvt.rn.f16x2.e2m1x2, portable bit-move fallback), the KV
  gathers pipelined by Triton as cp.async (16 B aligned rows: the alignment
  hint makes them vector loads) plus an L2 prefetch PFD tiles ahead.
* Even/odd dim split: byte j of a row holds dims 2j (low nibble) and 2j + 1
  (high): K_lo / K_hi planes against q's even / odd dims. Every product is
  exact in f16 (e2m1 x e4m3 <= 6 significant bits); K and V carry 2^-8,
  folded into the logit scale and the normalization. q gets the oxide
  per-(token, head) power-of-two prescale (amax -> [2^14, 2^15)).

Configs (TN rows per tile, warps, num_stages, L2 prefetch distance, planned
CTAs per SM -> splits): CONFIGS. The boot oracle (``python -m
nvfp4_ds_mla_patch.ours oracle``, once per pod via sitecustomize, fail closed
for ``ours``) checks every config against the f64 reference and the oxide
kernel, times each per (T, short / long lists) cold-L2 graphed, and writes
TABLE_PATH: per T the fastest config, ours vs oxide, registers / spills.
``auto`` takes ours where it is faster on short lists and not slower (2%) on
long ones; ``ours`` everywhere. ``bench`` = boot gate dsmla_attn_bench.
"""

import json
import math
import os
import sys

MARK = "[suffix dsmla-ours]"
MODE_ENV = "SUFFIX_SM120_DSMLA_ATTN"
TABLE_ENV = "SUFFIX_SM120_DSMLA_TABLE"
TABLE_PATH = "/tmp/suffix_dsmla_ours.json"
ROW = 352
DIM, PE = 512, 64
M_ROWS = 16      # q rows per program (tl.dot minimum): HB heads, padded
NS_MAX = 16      # splits per (token, head tile): bounds the last split's merge reads
NSP = 16         # merge lse vector (>= NS_MAX)
SEM_SLOTS = 1 << 16  # merge counters per device: T * HQT (larger calls run NS = 1)
FLUSH_BYTES = 256 << 20  # bench: > the 128 MB L2, zeroed before every replay
LOG2E = 1.4426950408889634
SCALE_FIX = 256.0
# name -> TN rows/tile, warps, num_stages (2: cp.async gathers), L2 prefetch distance in
# tiles, CTAs per SM the split plan assumes (shared memory: t64 ~58 KB, t32 ~38 KB, s1 ~22 KB)
CONFIGS = {
    "t32": {"tn": 32, "warps": 4, "stages": 2, "pfd": 2, "occ": 2},
    "t64": {"tn": 64, "warps": 8, "stages": 2, "pfd": 2, "occ": 1},
    "t32s1": {"tn": 32, "warps": 4, "stages": 1, "pfd": 2, "occ": 4},
    "t64s1": {"tn": 64, "warps": 8, "stages": 1, "pfd": 2, "occ": 3},
}
DEFAULT = "t32"


def mode() -> str:
    m = os.environ.get(MODE_ENV, "").strip().lower() or "oxide"
    if m not in ("ours", "auto", "oxide"):
        raise ValueError(f"{MODE_ENV}={m!r}: expected ours | auto | oxide")
    return m


def q_group(q_len: int) -> int:
    """Decode q-len -> the patch's group hint (rev 1 unioned a request's verify tokens;
    rev 2 runs per token and ignores it). Kept: the patch records it per call."""
    return max(g for g in (1, 2, 3, 4) if q_len % g == 0) if q_len > 0 else 1


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def plan(t: int, hq: int, c: int, num_sms: int, cfg: str = DEFAULT) -> dict:
    """Launch plan from (T, HQ, C, SMs, config) only (CUDA-graph stable): one wave of
    T * HQT * NS programs at the config's CTAs per SM, NS <= NS_MAX and <= the tiles."""
    if t <= 0 or hq <= 0 or c <= 0 or num_sms <= 0:
        raise ValueError(f"bad shape T={t} HQ={hq} C={c} SMs={num_sms}")
    k = CONFIGS[cfg]
    hb = 8 if hq <= 8 else 16
    hqt = -(-hq // hb)
    ns = max(1, min(NS_MAX, -(-c // k["tn"]), num_sms * k["occ"] // (t * hqt)))
    if t * hqt > SEM_SLOTS:
        ns = 1
    return {"hb": hb, "hqt": hqt, "ns": ns, "m": M_ROWS, "cp": _pow2(c), "cfg": cfg, **k}


# ---------------------------------------------------------------------------
# Triton kernel (ours_triton.py, imported on first use: needs triton)
# ---------------------------------------------------------------------------
_JIT = None


def _kernel():
    global _JIT
    if _JIT is None:
        from . import ours_triton as k

        _JIT = k._fwd
    return _JIT


# compile config the boot probe settled (table key "asm"): PTX e2m1x2 -> f16x2
# dequant when it is bitwise == the portable bit-move path. The interpreter has
# no inline asm (and no prefetch).
CFG = {"asm": True}
KERNELS = {}  # (config, HQ, CP, asm) -> last compiled _fwd (regs / spills / smem)
_SEMS = {}    # device index -> int32 [SEM_SLOTS] merge counters (zero between launches)


def use_asm() -> bool:
    return CFG["asm"] and not _interpret()


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "") == "1"


def _sems(dev):
    """This device's merge counters: zeroed once, rearmed by every merging split. Allocated
    on the first eager call (vLLM's profile / warm-up run precedes graph capture); calls on
    one stream only (vLLM's attention and the MTP drafter share the forward stream)."""
    import torch

    key = dev.index if dev.type == "cuda" else -1
    sem = _SEMS.get(key)
    if sem is None:
        if dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{MARK} first call inside CUDA graph capture: no merge "
                               "counters yet (an eager call must come first)")
        sem = _SEMS[key] = torch.zeros(SEM_SLOTS, dtype=torch.int32, device=dev)
    return sem


def config_for(t: int) -> str:
    """The table's fastest config at the nearest measured T (log scale), else DEFAULT."""
    rows = [r for r in ((_TABLE or {}).get("rows") or []) if r.get("cfg") in CONFIGS]
    if not rows:
        return DEFAULT
    return min(rows, key=lambda r: (abs(math.log(r["T"] / t)), -r["T"]))["cfg"]


def decode(q, kv_cache, topk, output, scale, group, num_sms, asm=None, cfg=None):
    """Our sparse-MLA decode: fills and returns output [T, HQ, 512] bf16. `group` (the
    patch's q-len hint) is unused since rev 2 (per-token programs)."""
    import torch

    t, hq = q.shape[0], q.shape[1]
    if t == 0:
        return output
    kv = kv_cache.view(torch.uint8)
    if (q.stride(2) != 1 or output.stride(2) != 1 or kv.dim() != 3 or kv.stride(1) != ROW
            or kv.stride(2) != 1):
        raise ValueError(f"{MARK} layout: q stride {q.stride()}, out stride "
                         f"{output.stride()}, kv {tuple(kv.shape)} stride {kv.stride()}")
    topk = topk.contiguous()
    c = topk.shape[1]
    p = plan(t, hq, c, num_sms, cfg or config_for(t))
    if p["ns"] > 1:
        opart = q.new_empty((t, hq, p["ns"], DIM), dtype=torch.bfloat16)
        lse = q.new_empty((t, hq, p["ns"]), dtype=torch.float32)
        sem = _sems(q.device)
    else:  # NS == 1 writes output; typed dummies keep one specialization
        opart, lse, sem = output, q.new_empty((1,), dtype=torch.float32), topk
    use = use_asm() if asm is None else asm
    k = _kernel()[(t * p["hqt"], p["ns"])](
        q, kv, topk, output, opart, lse, sem,
        t, c, p["ns"], topk.stride(0), q.stride(0), q.stride(1), output.stride(0),
        output.stride(1), kv.shape[1], kv.stride(0), float(scale) * LOG2E * SCALE_FIX,
        HQ=hq, HB=p["hb"], HQT=p["hqt"], M=p["m"], TN=p["tn"], CP=p["cp"], NSP=NSP,
        PFD=0 if _interpret() else p["pfd"], ASM=use, INTERPRET=_interpret(),
        num_warps=p["warps"],
        num_stages=p["stages"])
    KERNELS[(p["cfg"], hq, p["cp"], use)] = k
    return output


# ---------------------------------------------------------------------------
# Selection table (written by the boot oracle, read by the workers)
# ---------------------------------------------------------------------------
_TABLE = None


def table_path() -> str:
    return os.environ.get(TABLE_ENV, "").strip() or TABLE_PATH


def load_table(path=None):
    global _TABLE
    with open(path or table_path()) as f:
        _TABLE = json.load(f)
    return _TABLE


_MODE = "oxide"
_AUTO = {}


def init() -> str:
    """Worker side (impl init, eager): mode + table. ours: the table must
    exist and pass, else raise (fail closed); auto: a missing or failed
    table keeps the oxide kernels everywhere. Returns a summary line."""
    global _MODE, _TABLE
    _MODE, _TABLE = mode(), None
    _AUTO.clear()
    if _MODE == "oxide":
        return "oxide"
    path = table_path()
    try:
        tbl = load_table(path)
    except (OSError, ValueError) as exc:
        if _MODE == "ours":
            raise RuntimeError(
                f"{MARK} {MODE_ENV}=ours but no oracle table {path} ({exc}); sitecustomize "
                "runs `python -m nvfp4_ds_mla_patch.ours oracle` before serving") from exc
        return f"auto: no table {path} ({type(exc).__name__}), oxide everywhere"
    if not tbl.get("pass"):
        if _MODE == "ours":
            raise RuntimeError(f"{MARK} {MODE_ENV}=ours but the boot oracle FAILED: "
                               f"{tbl.get('failed')}")
        return f"auto: boot oracle failed {tbl.get('failed')}, oxide everywhere"
    CFG.update(asm=bool(tbl.get("asm", True)))
    rows = tbl.get("rows", [])
    return (f"{_MODE}: oracle pass, {CFG}, ours faster at "
            f"{sum(bool(r['ours']) for r in rows)}/{len(rows)} benched shapes")


def use_ours(t: int, group: int = 1) -> bool:
    """ours: always (init proved the oracle passed); auto: the table's verdict
    at the nearest measured T (log scale); else oxide."""
    if _MODE == "oxide" or _TABLE is None or not _TABLE.get("pass"):
        return False
    if _MODE == "ours":
        return True
    if t not in _AUTO:
        rows = _TABLE.get("rows", [])
        _AUTO[t] = bool(rows) and bool(min(
            rows, key=lambda r: (abs(math.log(r["T"] / t)), -r["T"]))["ours"])
    return _AUTO[t]


# ---------------------------------------------------------------------------
# torch emulation of the kernel algorithm (CPU tests; same plan / chunks /
# f16 operands / per-tile online softmax / bf16 partials / split-order merge)
# ---------------------------------------------------------------------------
def _f16_rows(rows):
    """[N, 352] u8 -> (k_lo, k_hi [N, 256], k_pe [N, 64]) f32 = value * 2^-8."""
    import torch

    from .oracle import E2M1, e4m3_float

    lut = torch.tensor(E2M1, dtype=torch.float32, device=rows.device)
    b = rows[:, :256].long()
    blk = torch.arange(256, device=rows.device) // 8
    sf = e4m3_float(rows[:, 320 + 8 * (blk & 3) + (blk >> 2)])
    k_lo = lut[b & 15] * sf / SCALE_FIX
    k_hi = lut[b >> 4] * sf / SCALE_FIX
    return k_lo, k_hi, e4m3_float(rows[:, 256:320]) / SCALE_FIX


def union_ok(topk, t0, t_last) -> bool:
    """Every token's column is -1 or the last token's slot (list property, tests)."""
    ref = topk[t_last]
    lst = topk[t0:t_last]
    return bool(((lst < 0) | (lst == ref)).all())


def live_len(lst) -> int:
    idx = (lst >= 0).nonzero()
    return int(idx[-1]) + 1 if idx.numel() else 0


def emulate(q, kv_cache, topk, scale, num_sms, cfg=DEFAULT):
    """Torch mirror of decode() for one config."""
    import torch

    t, hq = q.shape[0], q.shape[1]
    kv = kv_cache.view(torch.uint8)
    bs, bstride = kv.shape[1], kv.stride(0)
    flat = kv.as_strided((kv.shape[0], bs, ROW), (bstride, ROW, 1))
    p = plan(t, hq, topk.shape[1], num_sms, cfg)
    ns, tn = p["ns"], p["tn"]
    opart = torch.zeros(t, hq, ns, DIM, dtype=torch.float32)
    lse = torch.full((t, hq, ns), float("-inf"), dtype=torch.float32)
    qf = q.float()
    amax = qf.abs().amax(-1)
    e = (amax.view(torch.int32) >> 23) & 0xFF
    k = (141 - e).clamp(-114, 126)
    up = torch.pow(2.0, k.double()).float()
    q16 = (qf * up[..., None]).half().float()
    rsc = float(scale) * LOG2E * SCALE_FIX * torch.pow(2.0, -k.double()).float()
    for tk in range(t):
        L = live_len(topk[tk])
        chunk = tn * -(-(-(-L // tn)) // ns)  # TN * ceil(ceil(L / TN) / NS)
        for s in range(ns):
            lo, hi = s * chunk, min(s * chunk + chunk, L)
            qq, rs = q16[tk], rsc[tk]
            m = torch.full((hq,), float("-inf"), dtype=torch.float32)
            lsum = torch.zeros(hq, dtype=torch.float32)
            acc = torch.zeros(hq, DIM, dtype=torch.float32)
            for n0 in range(lo, hi, tn):
                cols = torch.arange(n0, min(n0 + tn, hi))
                slot = topk[tk, cols].long()
                rv = slot >= 0
                sl = slot.clamp_min(0)
                r = flat[sl // bs, sl % bs].clone()
                r[~rv] = 0
                k_lo, k_hi, k_pe = _f16_rows(r)
                kk = torch.cat([torch.stack([k_lo, k_hi], -1).view(-1, DIM), k_pe], -1)
                x = torch.where(rv[None, :], (qq @ kk.T) * rs[:, None], float("-inf"))
                m_new = torch.maximum(m, x.amax(1))
                m_use = torch.where(m_new == float("-inf"), 0.0, m_new)
                alpha = torch.exp2(m - m_use)
                pr = torch.exp2(x - m_use[:, None])
                lsum = lsum * alpha + pr.sum(1)
                acc = acc * alpha[:, None] + pr.half().float() @ kk[:, :DIM]
                m = m_new
            has = lsum > 0
            o = torch.where(has[:, None], acc * (SCALE_FIX / lsum.clamp_min(1e-38))[:, None], 0.0)
            opart[tk, :, s] = o.bfloat16().float()
            lse[tk, :, s] = torch.where(has, m + torch.log2(lsum.clamp_min(1e-38)),
                                        float("-inf"))
    if ns == 1:
        return opart[:, :, 0].bfloat16()
    mx = lse.amax(-1, keepdim=True)
    mx = torch.where(mx == float("-inf"), 0.0, mx)
    w = torch.exp2(lse - mx)
    den = w.sum(-1, keepdim=True)
    out = torch.where(den > 0, (w[..., None] * opart).sum(2) / den.clamp_min(1e-38), 0.0)
    return out.bfloat16()


# ---------------------------------------------------------------------------
# realistic lists: short-context prefixes and long-context top-k sets
# ---------------------------------------------------------------------------
def make_lists(n_req, q_len, nslots, gen, regime, topk=2048, ctx=(200, 1800)):
    """[n_req * q_len, topk] int32 physical slots (CPU). Request r owns a
    random page table over the slot pool (64-slot pages). 'short': context
    L_r in ctx, token j sees positions 0..L_r - q_len + j (DSA's 0..len-1
    then -1). 'long': 4 x topk positions of context; each token takes a
    2048-subset sharing ~90% with its neighbors, shuffled (top-k order)."""
    import torch

    pages = nslots // 64
    rows = []
    for _r in range(n_req):
        if regime == "short":
            L = int(torch.randint(ctx[0], ctx[1] + 1, (1,), generator=gen))
            npg = -(-L // 64)
            tbl = torch.randperm(pages, generator=gen)[:npg]
            pos2slot = (tbl[:, None] * 64 + torch.arange(64)).view(-1)[:L]
            for j in range(q_len):
                n = L - q_len + 1 + j
                row = torch.full((topk,), -1, dtype=torch.int32)
                row[:n] = pos2slot[:n].int()
                rows.append(row)
        else:
            L = min(4 * topk, pages * 64)
            tbl = torch.randperm(pages, generator=gen)[:-(-L // 64)]
            pos2slot = (tbl[:, None] * 64 + torch.arange(64)).view(-1)[:L]
            base = torch.randperm(L, generator=gen)[:topk]
            for _j in range(q_len):
                cur = base.clone()
                swap = torch.randperm(topk, generator=gen)[:topk // 10]
                pool = torch.randperm(L, generator=gen)
                fresh = pool[~torch.isin(pool, cur)][:swap.numel()]
                cur[swap] = fresh
                rows.append(pos2slot[cur[torch.randperm(topk, generator=gen)]].int())
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# GPU oracle + bench (boot)
# ---------------------------------------------------------------------------
class Triton:
    """oracle.Ours with the decode routed to our kernel at a fixed config."""

    def __init__(self, base, cfg=DEFAULT):
        self.base, self.cfg, self.P, self.impl = base, cfg, base.P, base.impl

    def write(self, *a):
        return self.base.write(*a)

    def decode(self, q, cache, topk, out=None):
        if out is None:
            out = q.new_empty(q.shape[0], q.shape[1], DIM)
        return decode(q, cache, topk, out, self.impl.scale, 1, self.impl._nvfp4_num_sms,
                      cfg=self.cfg)


def _oxide(base, q, cache, topk, out):
    import torch

    n = base.P._native()
    t, h = q.shape[0], q.shape[1]
    ns = n.nvfp4_ds_mla_plan(t, h, topk.shape[-1], base.impl._nvfp4_num_sms)["ns"]
    o_part = q.new_empty((t, h, ns, DIM), dtype=torch.bfloat16)
    lse = q.new_empty((t, h, ns), dtype=torch.float32)
    n.nvfp4_ds_mla_decode_cuda(q, cache.view(torch.uint8), topk.contiguous(), out, o_part,
                               lse, float(base.impl.scale),
                               torch.cuda.current_stream(q.device).cuda_stream)
    return out


def _cache(base, dev, nslots, gen):
    import torch

    from .oracle import make_kv

    cache = torch.zeros((nslots // 64, 64, ROW), dtype=torch.uint8, device=dev)
    for lo in range(0, nslots, 1 << 16):
        kv_c, k_pe = make_kv(min(1 << 16, nslots - lo), dev, gen, special=False)
        base.write(kv_c, k_pe, cache, torch.arange(lo, lo + kv_c.shape[0], device=dev))
    return cache


# (q_len, n_req) shapes: c1 verify (k=5) / drafts, c8 / c32 verify (k=2), k=3 batches,
# prefill-sized calls
ORACLE_SHAPES = [(1, 1), (6, 1), (1, 8), (3, 8), (4, 4), (3, 32), (3, 64), (4, 48)]
BENCH_SHAPES = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (6, 1), (6, 2),
                (4, 3), (4, 4), (4, 5), (3, 6), (3, 8), (3, 12), (3, 16), (3, 24),
                (3, 32), (4, 64), (4, 256), (4, 1024), (4, 2048)]


def probe(base, dev, cache, nslots, gen):
    """Settle CFG["asm"] (the PTX dequant only if bitwise == portable on DEFAULT) and the
    configs that compile, run and match the f64 reference; raises when none does."""
    import torch

    from .oracle import SCALE, attention_ref, dequant_rows_ref, row_rel_l2

    lat, rope = (x.double() for x in dequant_rows_ref(cache.view(-1, ROW)))
    cases = []
    for q_len, n_req, regime in ((6, 1, "short"), (6, 1, "long"), (1, 1, "short"),
                                 (3, 4, "long"), (1, 32, "short")):
        topk = make_lists(n_req, q_len, nslots, gen, regime).to(dev)
        cases.append((topk, torch.randn(topk.shape[0], 8, DIM + PE, generator=gen,
                                        dtype=torch.float32).bfloat16().to(dev)))

    def run(cfg, asm):
        outs = [decode(q, cache, tk, q.new_empty(q.shape[0], 8, DIM), SCALE, 1,
                       base.impl._nvfp4_num_sms, asm=asm, cfg=cfg) for tk, q in cases]
        torch.cuda.synchronize()
        return outs

    ok_cfgs, err = [], None
    for cfg in CONFIGS:
        try:
            outs = run(cfg, False)
            worst = max(row_rel_l2(o, attention_ref(q.double(), lat, rope, tk, SCALE),
                                   (tk >= 0).any(-1)[:, None].expand(-1, 8))
                        for o, (tk, q) in zip(outs, cases))
        except Exception as exc:  # noqa: BLE001 - e.g. OutOfResources: drop the config
            err = exc
            print(f"{MARK} config {cfg}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"{MARK} config {cfg}: max row rel-L2 vs f64 {worst:.2e}", flush=True)
        if worst <= 2e-2:
            ok_cfgs.append(cfg)
        if cfg == DEFAULT and cfg in ok_cfgs:
            try:
                CFG["asm"] = all(torch.equal(a, b) for a, b in zip(run(cfg, True), outs))
                if not CFG["asm"]:
                    print(f"{MARK} asm dequant != portable: portable path", flush=True)
            except Exception as exc:  # noqa: BLE001 - the portable path is the fallback
                print(f"{MARK} asm dequant unusable, portable path: {type(exc).__name__}: "
                      f"{exc}", flush=True)
                CFG["asm"] = False
    if DEFAULT not in ok_cfgs:
        raise RuntimeError(f"default config {DEFAULT} fails (ok: {ok_cfgs}): {err}")
    return {"asm": CFG["asm"], "configs": ok_cfgs}


def run_oracle(base, dev, report, nslots=1 << 18):
    """Gates (fail closed); returns the configs that passed the probe."""
    import torch

    from .oracle import (SCALE, attention_ref, dequant_rows_ref, gate_adversarial,
                         gate_reader, rel_l2, row_rel_l2, Stock)

    gen = torch.Generator().manual_seed(0)
    cache = _cache(base, dev, nslots, torch.Generator(device=dev).manual_seed(1))
    lat, rope = (x.double() for x in dequant_rows_ref(cache.view(-1, ROW)))
    pr = probe(base, dev, cache, nslots, gen)
    print(f"{MARK} config {pr}", flush=True)
    for q_len, n_req in ORACLE_SHAPES:
        for regime in ("short", "long"):
            topk = make_lists(n_req, q_len, nslots, gen, regime).to(dev)
            t = topk.shape[0]
            if t > 1:
                topk[t // 2, 5::7] = -1  # holes
                topk[-1] = -1 if regime == "long" else topk[-1]  # an all -1 token
            q = (torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32)
                 * 2).bfloat16().to(dev)
            ours = decode(q, cache, topk, q.new_empty(t, 8, DIM), SCALE, 1,
                          base.impl._nvfp4_num_sms)
            ox = _oxide(base, q, cache, topk, q.new_empty(t, 8, DIM))
            torch.cuda.synchronize()
            ref = attention_ref(q.double(), lat, rope, topk, SCALE)
            valid = (topk >= 0).any(-1)
            keep = valid[:, None].expand(t, 8)
            e_ref = row_rel_l2(ours, ref, keep)
            e_ox = rel_l2(ours[valid], ox[valid]) if valid.any() else 0.0
            zero = bool((ours[~valid] == 0).all())
            report(f"ours_{regime}_q{q_len}x{n_req}",
                   e_ref <= 2e-2 and e_ox <= 2e-2 and zero and bool(torch.isfinite(ours).all()),
                   f"T={t} ns={plan(t, 8, 2048, base.impl._nvfp4_num_sms)['ns']} max row rel-L2 "
                   f"vs f64 {e_ref:.2e}, vs oxide {e_ox:.2e}, all -1 token zero {zero}")
    # the existing reader / adversarial gates through our kernel; the reader gate also
    # needs vLLM's stock FlashInfer fp8_ds_mla path
    try:
        stock = Stock(dev)
    except Exception as exc:  # noqa: BLE001 - ours == f64 ref and oxide above
        stock = None
        print(f"{MARK} stock FlashInfer path unavailable, reader gate skipped: "
              f"{type(exc).__name__}: {exc}", flush=True)
    ours = Triton(base)
    if stock is not None:
        gate_reader(ours, stock, dev, gen, report)
    gate_adversarial(ours, dev, gen, report)
    # CUDA graph: capture once, replay with new lists == eager (every config: the merge
    # counters must come back rearmed)
    t = 24
    q = torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32).bfloat16().to(dev)
    for cfg in pr["configs"]:
        topk = make_lists(8, 3, nslots, gen, "short").to(dev)
        out = q.new_empty(t, 8, DIM)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            decode(q, cache, topk, out, SCALE, 1, base.impl._nvfp4_num_sms, cfg=cfg)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            decode(q, cache, topk, out, SCALE, 1, base.impl._nvfp4_num_sms, cfg=cfg)
        same = True
        for regime in ("long", "short", "long"):
            topk.copy_(make_lists(8, 3, nslots, gen, regime).to(dev))
            gr.replay()
            eager = decode(q, cache, topk, q.new_empty(t, 8, DIM), SCALE, 1,
                           base.impl._nvfp4_num_sms, cfg=cfg)
            torch.cuda.synchronize()
            same = same and torch.equal(out, eager)
        report(f"ours_cuda_graph_replay_{cfg}", same,
               "graph replays with new lists (long, short, long) bitwise == eager")
    for line in kernel_lines():
        print(f"{MARK} {line}", flush=True)
    return pr["configs"]


def kernel_lines():
    return [f"kernel {cfg} HQ={hq} CP={cp} asm={asm}: {k.n_regs} regs, {k.n_spills} spills, "
            f"{k.metadata.shared} B smem, {k.metadata.num_warps} warps"
            for (cfg, hq, cp, asm), k in sorted(KERNELS.items())]


def _time(fn, flush, reps):
    """Graphed GPU us of one call, L2 flushed before each replay."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    tot = 0.0
    for _ in range(reps):
        flush.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        tot += e0.elapsed_time(e1) * 1e3
    return round(tot / reps, 2)


def run_bench(base, dev, reps=20, nslots=1 << 20, configs=None):
    """Per T: every config on short (prefix) and long (top-k) lists vs oxide; the row keeps
    the config with the smallest worst ratio to oxide over both regimes."""
    import torch

    from .oracle import SCALE

    configs = configs or list(CONFIGS)
    gen = torch.Generator().manual_seed(2)
    cache = _cache(base, dev, nslots, torch.Generator(device=dev).manual_seed(3))
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    sms = base.impl._nvfp4_num_sms
    rows = []
    for q_len, n_req in BENCH_SHAPES:
        t = q_len * n_req
        q = torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32).bfloat16().to(dev)
        out = q.new_empty(t, 8, DIM)
        r = {"T": t, "q_len": q_len}
        lists = {g: make_lists(n_req, q_len, nslots, gen, g).to(dev) for g in ("short", "long")}
        for regime, topk in lists.items():
            r[f"oxide_{regime}"] = _time(lambda: _oxide(base, q, cache, topk, out), flush, reps)
            for cfg in configs:
                r[f"{cfg}_{regime}"] = _time(
                    lambda: decode(q, cache, topk, out, SCALE, 1, sms, cfg=cfg), flush, reps)
        best = min(configs, key=lambda c: (max(r[f"{c}_short"] / r["oxide_short"],
                                               r[f"{c}_long"] / r["oxide_long"]),
                                           r[f"{c}_short"]))
        r.update(cfg=best, ns=plan(t, 8, 2048, sms, best)["ns"], ours_short=r[f"{best}_short"],
                 ours_long=r[f"{best}_long"])
        r["ours"] = (r["ours_short"] <= r["oxide_short"]
                     and r["ours_long"] <= 1.02 * r["oxide_long"])
        rows.append(r)
        per = " ".join(f"{c} {r[f'{c}_short']:.1f}/{r[f'{c}_long']:.1f}" for c in configs)
        print(f"{MARK} bench T={t:5d} (q_len {q_len} x {n_req}): oxide short/long "
              f"{r['oxide_short']:.1f}/{r['oxide_long']:.1f} us | {per} -> "
              f"{'ours ' + best if r['ours'] else 'oxide'} (ns {r['ns']})", flush=True)
    return rows


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="our sparse-MLA decode: oracle + bench")
    ap.add_argument("what", choices=["oracle", "bench"],
                    help="oracle: gates + bench + table (fail closed); bench: same, "
                         "never fails (boot gate evidence)")
    ap.add_argument("--table", default=table_path())
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        import torch

        torch.set_default_dtype(torch.float32)  # every helper's implicit floats
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            print(f"{MARK} NOT RUN: needs an SM120 (cc 12.x) GPU", flush=True)
            return 2
        from .oracle import Ours

        dev = torch.device("cuda", torch.cuda.current_device())
        base = Ours(dev.index)
    except Exception as exc:  # noqa: BLE001 - the verdict is the evidence
        print(f"{MARK} NOT RUN: {type(exc).__name__}: {exc}", flush=True)
        return 2
    results = []

    def report(name, ok, detail):
        results.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"{MARK} {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    configs = []
    try:
        configs = run_oracle(base, dev, report)
    except Exception as exc:  # noqa: BLE001
        report("oracle", False, f"raised {type(exc).__name__}: {exc}")
    ok = bool(results) and all(r["pass"] for r in results)
    rows = []
    if ok:
        try:
            rows = run_bench(base, dev, args.reps, configs=configs)
        except Exception as exc:  # noqa: BLE001
            print(f"{MARK} bench raised {type(exc).__name__}: {exc}", flush=True)
            ok = args.what == "bench"
    table = {"pass": ok, "rows": rows, "asm": CFG["asm"], "configs": configs,
             "kernels": kernel_lines(), "failed": [r["name"] for r in results if not r["pass"]]}
    tmp = f"{args.table}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(table, f)
    os.replace(tmp, args.table)
    print(f"{MARK} ORACLE {'PASS' if ok else 'FAIL'} "
          f"({sum(r['pass'] for r in results)}/{len(results)} gates), table {args.table}: "
          f"ours at {sum(r['ours'] for r in rows)}/{len(rows)} shapes", flush=True)
    return 0 if ok or args.what == "bench" else 1


if __name__ == "__main__":
    sys.exit(main())
