# SPDX-License-Identifier: Apache-2.0
"""Our Triton sparse-MLA decode over the nvfp4_ds_mla reader cache (SM120),
SUFFIX_SM120_DSMLA_ATTN=ours|auto|oxide (default oxide = the cuda-oxide pair).

Same contract as the oxide kernels (q [T, HQ, 576] bf16, 352 B rows, top-k
[T, C] physical slots with -1 masked, out [T, HQ, 512] bf16, softmax(S *
scale) . latent, an all-masked token is 0). What it does differently:

* Work follows the live list, not the capacity. A program (token group,
  split) splits [0, L) of its list, L = last live column + 1, into NS
  chunks of whole TN-row tiles. Short contexts (DSA lists are 0..len-1 then
  -1) cost len rows, not 2048; programs past the work write lse = -inf.
* Union of a request's MTP verify tokens. A group = G consecutive tokens (G
  from the decode q-len, impl._suffix_q_group). When every token's list is
  -1 or equal to the group's last list column by column (the short-context
  case: prefixes of one position list), each row tile is dequantized ONCE
  for all G x 8 q rows (M = 16/32) with per-row live masks; otherwise the
  group runs per token (M = 16). Correct for any grouping: the check
  decides, G is only a performance hint.
* Even/odd dim split: K_lo / K_hi are the low / high nibble planes, q is
  split the same way, so no in-register interleave. e2m1 -> f16 by PTX
  cvt.rn.f16x2.e2m1x2 (portable fallback: a bit move giving value * 2^-14,
  SF pre-scaled by 2^14), SF and RoPE e4m3 -> f16 by a bit move (* 2^-8):
  every product exact in f16, K and V carry 2^-8 (folded into the logit
  scale and the partial normalization). q gets the oxide per-(token, head)
  power-of-two prescale (amax -> [2^14, 2^15)) before f16.
* NS == 1 writes the output directly; else bf16 partials + lse (log2) and
  a merge kernel (weights exp2(lse - max), all -inf -> 0 row).

Selection: the boot oracle (``python -m nvfp4_ds_mla_patch.ours oracle``,
run once per pod by sitecustomize, fail closed for ``ours``) checks ours vs
the f64 reference and the oxide kernel and writes TABLE_PATH: per (G, T)
graphed cold-L2 us of ours and oxide on short-context (prefix) and
long-context (top-k) lists; ``auto`` takes ours where it is faster on short
lists and not slower (2% noise) on long ones, so no regime regresses;
``ours`` forces it everywhere. ``bench`` = the boot gate dsmla_attn_bench
(evidence, same table).
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
TN = 32          # list rows per tile
MAX_NS = 256     # splits (merge NSP)
WAVE_C = 3       # fixed per-program cost in tiles (plan model)
FLUSH_BYTES = 256 << 20  # bench: > the 128 MB L2, zeroed before every replay
LOG2E = 1.4426950408889634
SCALE_FIX = 256.0


def mode() -> str:
    m = os.environ.get(MODE_ENV, "").strip().lower() or "oxide"
    if m not in ("ours", "auto", "oxide"):
        raise ValueError(f"{MODE_ENV}={m!r}: expected ours | auto | oxide")
    return m


def q_group(q_len: int) -> int:
    """Union group size for a uniform decode q-len: the largest divisor <= 4
    (M = 8G rows <= 32), so groups never straddle requests."""
    return max(g for g in (1, 2, 3, 4) if q_len % g == 0) if q_len > 0 else 1


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def plan(t: int, hq: int, c: int, group: int, num_sms: int) -> dict:
    """Launch plan from (T, HQ, C, G, SMs) only (CUDA-graph stable).
    NS = argmin waves * (G * k + WAVE_C) over canonical splits (k tiles per
    split; the full-list per-token worst case), ties keep fewer splits."""
    if t <= 0 or hq <= 0 or c <= 0 or num_sms <= 0:
        raise ValueError(f"bad shape T={t} HQ={hq} C={c} SMs={num_sms}")
    hb = 8 if hq <= 8 else 16
    g = max(1, min(group, 32 // hb))
    hqt = -(-hq // hb)
    groups = -(-t // g)
    tiles = -(-c // TN)
    best = None
    for ns in range(1, min(tiles, MAX_NS) + 1):
        k = -(-tiles // ns)
        if -(-tiles // k) != ns:
            continue
        cost = -(-(groups * hqt * ns) // num_sms) * (g * k + WAVE_C)
        if best is None or cost < best[0]:
            best = (cost, ns)
    m = max(16, _pow2(g * hb))
    return {"hb": hb, "g": g, "hqt": hqt, "groups": groups, "ns": best[1],
            "m": m, "cp": _pow2(c), "warps": 8 if m > 16 else 4}


# ---------------------------------------------------------------------------
# Triton kernels (ours_triton.py, imported on first use: needs triton)
# ---------------------------------------------------------------------------
_JIT = None


def _kernels():
    global _JIT
    if _JIT is None:
        from . import ours_triton as k

        _JIT = (k._fwd, k._merge)
    return _JIT


# compile config the boot probe settled (table keys "asm", "stages"): PTX
# e2m1x2 -> f16x2 dequant when it is bitwise == the portable bit-move path;
# num_stages 1 when 2 does not fit shared memory. The interpreter has no asm.
CFG = {"asm": True, "stages": 2}
KERNELS = {}  # (G, M, HQ, CP, asm) -> last compiled _fwd (regs / spills / smem)


def use_asm() -> bool:
    return CFG["asm"] and os.environ.get("TRITON_INTERPRET", "") != "1"


def decode(q, kv_cache, topk, output, scale, group, num_sms, stats=None, asm=None):
    """Our sparse-MLA decode: fills and returns output [T, HQ, 512] bf16."""
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
    p = plan(t, hq, c, group, num_sms)
    fwd, merge = _kernels()
    if p["ns"] > 1:
        opart = q.new_empty((t, hq, p["ns"], DIM), dtype=torch.bfloat16)
        lse = q.new_empty((t, hq, p["ns"]), dtype=torch.float32)
    else:  # NS == 1 writes output; typed dummies keep one specialization
        opart, lse = output, q.new_empty((1,), dtype=torch.float32)
    use = use_asm() if asm is None else asm
    k = fwd[(p["groups"] * p["hqt"], p["ns"])](
        q, kv, topk, output, opart, lse, stats if stats is not None else topk,
        t, c, p["ns"], topk.stride(0), q.stride(0), q.stride(1), output.stride(0),
        output.stride(1), kv.shape[1], kv.stride(0), float(scale) * LOG2E * SCALE_FIX,
        HQ=hq, HB=p["hb"], HQT=p["hqt"], G=p["g"], M=p["m"], TN=TN, CP=p["cp"],
        STATS=stats is not None, ASM=use, num_warps=p["warps"], num_stages=CFG["stages"])
    KERNELS[(p["g"], p["m"], hq, p["cp"], use)] = k
    if p["ns"] > 1:
        merge[(t * hq,)](output, opart, lse, p["ns"], output.stride(0), output.stride(1),
                         HQ=hq, NSP=MAX_NS, num_warps=4)
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
    CFG.update(asm=bool(tbl.get("asm", True)), stages=int(tbl.get("stages", 2)))
    rows = tbl.get("rows", [])
    return (f"{_MODE}: oracle pass, {CFG}, ours faster at "
            f"{sum(bool(r['ours']) for r in rows)}/{len(rows)} benched shapes")


def use_ours(t: int, group: int) -> bool:
    """ours: always (init proved the oracle passed); auto: the table's verdict
    at the nearest measured T (log scale) of the same G; else oxide."""
    if _MODE == "oxide" or _TABLE is None or not _TABLE.get("pass"):
        return False
    if _MODE == "ours":
        return True
    key = (t, group)
    if key not in _AUTO:
        rows = [r for r in _TABLE.get("rows", []) if r["G"] == group]
        _AUTO[key] = bool(rows) and bool(min(
            rows, key=lambda r: (abs(math.log(r["T"] / t)), -r["T"]))["ours"])
    return _AUTO[key]


# ---------------------------------------------------------------------------
# torch emulation of the kernel algorithm (CPU tests; same plan / chunks /
# union rule / f16 operands / per-tile online softmax / bf16 partials)
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
    """Union rule: every token's column is -1 or the last token's slot."""
    ref = topk[t_last]
    lst = topk[t0:t_last]
    return bool(((lst < 0) | (lst == ref)).all())


def live_len(lst) -> int:
    idx = (lst >= 0).nonzero()
    return int(idx[-1]) + 1 if idx.numel() else 0


def emulate(q, kv_cache, topk, scale, group, num_sms, union=None):
    """Torch mirror of decode(); union (list, one bool per group) records the
    path each group took."""
    import torch

    t, hq = q.shape[0], q.shape[1]
    kv = kv_cache.view(torch.uint8)
    bs, bstride = kv.shape[1], kv.stride(0)
    flat = kv.as_strided((kv.shape[0], bs, ROW), (bstride, ROW, 1))
    p = plan(t, hq, topk.shape[1], group, num_sms)
    ns = p["ns"]
    opart = torch.zeros(t, hq, ns, DIM, dtype=torch.float32)
    lse = torch.full((t, hq, ns), float("-inf"), dtype=torch.float32)
    qf = q.float()
    amax = qf.abs().amax(-1)
    e = (amax.view(torch.int32) >> 23) & 0xFF
    k = (141 - e).clamp(-114, 126)
    up = torch.pow(2.0, k.double()).float()
    q16 = (qf * up[..., None]).half().float()
    rsc = float(scale) * LOG2E * SCALE_FIX * torch.pow(2.0, -k.double()).float()

    def run(toks, ref_tok, L, h0, s):
        heads = [h for h in range(h0, min(h0 + p["hb"], hq))]
        chunk = TN * -(-(-(-L // TN)) // ns)  # TN * ceil(ceil(L / TN) / ns)
        lo, hi = s * chunk, min(s * chunk + chunk, L)
        rows = [(tk, h) for tk in toks for h in heads]
        qq = torch.stack([q16[tk, h] for tk, h in rows])
        rs = torch.stack([rsc[tk, h] for tk, h in rows])
        m = torch.full((len(rows),), float("-inf"), dtype=torch.float32)
        lsum = torch.zeros(len(rows), dtype=torch.float32)
        acc = torch.zeros(len(rows), DIM, dtype=torch.float32)
        for n0 in range(lo, hi, TN):
            cols = torch.arange(n0, min(n0 + TN, hi))
            slot = topk[ref_tok, cols].long()
            rv = slot >= 0
            sl = slot.clamp_min(0)
            r = flat[sl // bs, sl % bs].clone()
            r[~rv] = 0
            k_lo, k_hi, k_pe = _f16_rows(r)
            kk = torch.cat([torch.stack([k_lo, k_hi], -1).view(-1, DIM), k_pe], -1)
            live = torch.stack([topk[tk, cols] >= 0 for tk, _ in rows])
            x = torch.where(live, (qq @ kk.T) * rs[:, None], float("-inf"))
            m_new = torch.maximum(m, x.amax(1))
            m_use = torch.where(m_new == float("-inf"), 0.0, m_new)
            alpha = torch.exp2(m - m_use)
            pr = torch.exp2(x - m_use[:, None])
            lsum = lsum * alpha + pr.sum(1)
            acc = acc * alpha[:, None] + pr.half().float() @ kk[:, :DIM]
            m = m_new
        has = lsum > 0
        o = torch.where(has[:, None], acc * (SCALE_FIX / lsum.clamp_min(1e-38))[:, None], 0.0)
        for i, (tk, h) in enumerate(rows):
            opart[tk, h, s] = o[i].bfloat16().float()
            lse[tk, h, s] = m[i] + torch.log2(lsum[i]) if has[i] else float("-inf")

    for gi in range(p["groups"]):
        t0 = gi * p["g"]
        t_last = min(t0 + p["g"], t) - 1
        ok = union_ok(topk, t0, t_last)
        if union is not None:
            union.append(ok)
        for ht in range(p["hqt"]):
            for s in range(ns):
                if ok:
                    run(range(t0, t_last + 1), t_last, live_len(topk[t_last]), ht * p["hb"], s)
                else:
                    for tk in range(t0, t_last + 1):
                        run([tk], tk, live_len(topk[tk]), ht * p["hb"], s)
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
    """oracle.Ours with the decode routed to our kernel at a fixed G."""

    def __init__(self, base, group=1):
        self.base, self.group, self.P, self.impl = base, group, base.P, base.impl

    def write(self, *a):
        return self.base.write(*a)

    def decode(self, q, cache, topk, out=None, stats=None):
        if out is None:
            out = q.new_empty(q.shape[0], q.shape[1], DIM)
        return decode(q, cache, topk, out, self.impl.scale, self.group,
                      self.impl._nvfp4_num_sms, stats=stats)


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


# (q_len, n_req[, G]) shapes: c1 verify (k=5) / drafts, c8 / c32 verify (k=2),
# k=3 batches, prefill-sized calls (G 4: consecutive prefill tokens), and G
# not matching the q-len (groups straddle requests: a graph captured at
# another draft length) -> those groups must take the per-token path
ORACLE_SHAPES = [(1, 1), (6, 1), (1, 8), (3, 8), (4, 4), (3, 32), (3, 64), (4, 48),
                 (3, 8, 4), (6, 2, 4)]
BENCH_SHAPES = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (6, 1), (6, 2),
                (4, 3), (4, 4), (4, 5), (3, 6), (3, 8), (3, 12), (3, 16), (3, 24),
                (3, 32), (4, 64), (4, 256), (4, 1024), (4, 2048)]


def probe(base, dev, cache, nslots, gen):
    """Settle CFG: num_stages 2 (else 1) where the portable path compiles and
    runs; the PTX dequant only if it is bitwise == portable (G 1, 3, 4; union
    and per-token paths). Raises when nothing compiles."""
    import torch

    from .oracle import SCALE

    cases = []
    for q_len, regime in ((6, "short"), (6, "long"), (1, "short"), (4, "long")):
        topk = make_lists(2, q_len, nslots, gen, regime).to(dev)
        cases.append((q_len, topk, torch.randn(topk.shape[0], 8, DIM + PE, generator=gen,
                                                dtype=torch.float32).bfloat16().to(dev)))

    def run(asm):
        outs = [decode(q, cache, tk, q.new_empty(q.shape[0], 8, DIM), SCALE, q_group(ql),
                       base.impl._nvfp4_num_sms, asm=asm) for ql, tk, q in cases]
        torch.cuda.synchronize()
        return outs

    err = None
    for stages in (2, 1):
        CFG["stages"] = stages
        try:
            ref = run(False)
        except Exception as exc:  # noqa: BLE001 - e.g. OutOfResources: fewer stages
            err = exc
            print(f"{MARK} num_stages {stages}: {type(exc).__name__}: {exc}", flush=True)
            continue
        try:
            CFG["asm"] = all(torch.equal(a, b) for a, b in zip(run(True), ref))
            if not CFG["asm"]:
                print(f"{MARK} asm dequant != portable: portable path", flush=True)
        except Exception as exc:  # noqa: BLE001 - the portable path is the fallback
            print(f"{MARK} asm dequant unusable, portable path: {type(exc).__name__}: {exc}",
                  flush=True)
            CFG["asm"] = False
        return dict(CFG)
    raise RuntimeError(f"our kernel does not run at num_stages 2 or 1: {err}")


def run_oracle(base, dev, report, nslots=1 << 18):
    import torch

    from .oracle import (SCALE, attention_ref, dequant_rows_ref, gate_adversarial,
                         gate_reader, rel_l2, row_rel_l2, Stock)

    gen = torch.Generator().manual_seed(0)
    cache = _cache(base, dev, nslots, torch.Generator(device=dev).manual_seed(1))
    lat, rope = (x.double() for x in dequant_rows_ref(cache.view(-1, ROW)))
    print(f"{MARK} config {probe(base, dev, cache, nslots, gen)}", flush=True)
    for q_len, n_req, *g in ORACLE_SHAPES:
        g = g[0] if g else q_group(q_len)
        for regime in ("short", "long"):
            topk = make_lists(n_req, q_len, nslots, gen, regime).to(dev)
            t = topk.shape[0]
            if t > 1:
                topk[t // 2, 5::7] = -1  # holes: breaks the union of that group
                topk[-1] = -1 if regime == "long" else topk[-1]  # an all -1 token
            q = (torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32)
                 * 2).bfloat16().to(dev)
            stats = torch.full((-(-t // g),), -1, dtype=torch.int32, device=dev)
            ours = decode(q, cache, topk, q.new_empty(t, 8, DIM), SCALE, g,
                          base.impl._nvfp4_num_sms, stats=stats)
            ox = _oxide(base, q, cache, topk, q.new_empty(t, 8, DIM))
            torch.cuda.synchronize()
            ref = attention_ref(q.double(), lat, rope, topk, SCALE)
            valid = (topk >= 0).any(-1)
            keep = valid[:, None].expand(t, 8)
            want = [union_ok(topk.cpu(), g0, min(g0 + g, t) - 1) for g0 in range(0, t, g)]
            e_ref = row_rel_l2(ours, ref, keep)
            e_ox = rel_l2(ours[valid], ox[valid]) if valid.any() else 0.0
            zero = bool((ours[~valid] == 0).all())
            path = stats.tolist() == [int(w) for w in want]
            report(f"ours_{regime}_q{q_len}x{n_req}_G{g}",
                   e_ref <= 2e-2 and e_ox <= 2e-2 and zero and path
                   and bool(torch.isfinite(ours).all()),
                   f"T={t} max row rel-L2 vs f64 {e_ref:.2e}, vs oxide {e_ox:.2e}, all -1 "
                   f"token zero {zero}, union groups {sum(want)}/{len(want)} as expected "
                   f"{path}")
    # the existing reader / adversarial gates through our kernel (G 1 and 3);
    # the reader gate also needs vLLM's stock FlashInfer fp8_ds_mla path
    try:
        stock = Stock(dev)
    except Exception as exc:  # noqa: BLE001 - ours == f64 ref and oxide above
        stock = None
        print(f"{MARK} stock FlashInfer path unavailable, reader gate skipped: "
              f"{type(exc).__name__}: {exc}", flush=True)
    for g in (1, 3):
        ours = Triton(base, g)
        if stock is not None:
            gate_reader(ours, stock, dev, gen, report)
        gate_adversarial(ours, dev, gen, report)
    # CUDA graph: capture once, replay with new lists == eager
    t, g = 24, 3
    q = torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32).bfloat16().to(dev)
    topk = make_lists(8, 3, nslots, gen, "short").to(dev)
    out = q.new_empty(t, 8, DIM)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        decode(q, cache, topk, out, SCALE, g, base.impl._nvfp4_num_sms)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        decode(q, cache, topk, out, SCALE, g, base.impl._nvfp4_num_sms)
    for regime in ("long", "short"):
        topk.copy_(make_lists(8, 3, nslots, gen, regime).to(dev))
        gr.replay()
        eager = decode(q, cache, topk, q.new_empty(t, 8, DIM), SCALE, g,
                       base.impl._nvfp4_num_sms)
        torch.cuda.synchronize()
        report(f"ours_cuda_graph_replay_{regime}", torch.equal(out, eager),
               "graph replay with new lists bitwise == eager")
    for (g, m, hq, cp, asm), k in sorted(KERNELS.items()):
        print(f"{MARK} kernel G={g} M={m} HQ={hq} CP={cp} asm={asm}: {k.n_regs} regs, "
              f"{k.n_spills} spills, {k.metadata.shared} B smem, {k.metadata.num_warps} warps",
              flush=True)


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


def run_bench(base, dev, reps=20, nslots=1 << 20):
    import torch

    from .oracle import SCALE

    gen = torch.Generator().manual_seed(2)
    cache = _cache(base, dev, nslots, torch.Generator(device=dev).manual_seed(3))
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    sms = base.impl._nvfp4_num_sms
    rows = []
    for q_len, n_req in BENCH_SHAPES:
        t, g = q_len * n_req, q_group(q_len)
        q = torch.randn(t, 8, DIM + PE, generator=gen, dtype=torch.float32).bfloat16().to(dev)
        out = q.new_empty(t, 8, DIM)
        r = {"T": t, "G": g, "q_len": q_len, "ns": plan(t, 8, 2048, g, sms)["ns"]}
        for regime in ("short", "long"):
            topk = make_lists(n_req, q_len, nslots, gen, regime).to(dev)
            r[f"ours_{regime}"] = _time(lambda: decode(q, cache, topk, out, SCALE, g, sms),
                                        flush, reps)
            r[f"oxide_{regime}"] = _time(lambda: _oxide(base, q, cache, topk, out),
                                         flush, reps)
        r["ours"] = (r["ours_short"] <= r["oxide_short"]
                     and r["ours_long"] <= 1.02 * r["oxide_long"])
        rows.append(r)
        print(f"{MARK} bench T={t:5d} G={g} (q_len {q_len} x {n_req}) ns={r['ns']:3d}: short "
              f"ours {r['ours_short']:8.2f} / oxide {r['oxide_short']:8.2f} us, long ours "
              f"{r['ours_long']:8.2f} / oxide {r['oxide_long']:8.2f} us -> "
              f"{'ours' if r['ours'] else 'oxide'}", flush=True)
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

    try:
        run_oracle(base, dev, report)
    except Exception as exc:  # noqa: BLE001
        report("oracle", False, f"raised {type(exc).__name__}: {exc}")
    ok = bool(results) and all(r["pass"] for r in results)
    rows = []
    if ok:
        try:
            rows = run_bench(base, dev, args.reps)
        except Exception as exc:  # noqa: BLE001
            print(f"{MARK} bench raised {type(exc).__name__}: {exc}", flush=True)
            ok = args.what == "bench"
    table = {"pass": ok, "rows": rows, "asm": CFG["asm"], "stages": CFG["stages"],
             "failed": [r["name"] for r in results if not r["pass"]]}
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
