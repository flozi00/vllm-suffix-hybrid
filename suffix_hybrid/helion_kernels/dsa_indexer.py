# SPDX-License-Identifier: Apache-2.0
"""DSA indexer logits on Helion (GLM-5.3 / DeepSeek-V3.2 lightning indexer):
port of kernels-oxide/dsa_indexer (PTX entry ``dsa_logits``), the FP8 contract
of deep_gemm ``fp8_fp4_paged_mqa_logits`` (decode) and ``fp8_fp4_mqa_logits``
(prefill):

    logits[r, n] = sum_h relu(dot(q[r, h], k[n])) * w[r, h] * k_scale[n]

for lo[r] <= n < hi[r] (decode: lo 0, hi = context_lens[r] clamped to the
logits width; prefill: [ks, ke) clamped to [0, N)). Every other column is left
UNWRITTEN, exactly like the oxide kernel: vLLM's top-k kernels read only the
window (the lane's oracle proves it on a NaN-poisoned buffer). q / k e4m3,
hl.dot with f32 accumulation (e4m3 x e4m3 products are exact; only the f32
summation order differs from oxide), relu, x w, sum over heads, x k_scale —
the oxide order. Semantics, layouts and the oracle are the lane's
(suffix_hybrid/kernels/dsa_indexer.py).

Paged layout: the cache is uint8 [blocks, 64, 132] split pages (64 x 128 e4m3
values, then 64 f32 scales). The bodies take two VIEWS of that storage:
``kv`` e4m3 [blocks * 66, 128] (token t of page p = row 66 p + t; rows 64, 65
of a page are its scale bytes) and ``ks`` f32 [blocks * 2112] (scale of token
t of page p = 2112 p + 2048 + t). Every token is gathered through its own
block-table entry, so a KV tile may straddle pages and block_n is free.

Design: one program = one decode ROW (decode) / a block of Q rows (prefill)
x one split of that row's window; the split range comes from the window
itself on device (``per = max(ceil(tiles / eff_splits), min_tiles)`` 64-token
tiles), so the grid is a function of host shapes only (CUDA-graph safe) and
short contexts use fewer, longer programs like oxide's min_tiles. Tunables:
block_n (KV tokens per step), block_m (prefill Q rows), ``min_tiles``,
``split_div`` (fewer effective splits than the host cap), num_warps /
num_stages (the gathered K loads pipeline), pid_type / loop order of the
(row, split) grid, eviction policy per load.
ponytail: decode reads K once per ROW (MTP rows of one request share pages
through L2, not through one staged tile like oxide's page-dedup groups) — rows
with different block tables cannot share a 2-D dot without a data-dependent
dedup. Upgrade path if MTP shapes lose in the lab: a row-group body with a
batched 3-D dot over per-row gathers.

Serving: ``dsa_indexer_logits_cuda`` has the oxide op's exact signature; the
lane calls it instead of ``_native.dsa_indexer_logits_cuda`` when
SUFFIX_HELION_DSA_INDEXER != oxide (routed per rows bucket; auto -> oxide
where no config is shipped).
"""
from __future__ import annotations

import json

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.kernels import dsa_indexer as di

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no-Helion host: module stays importable (lab ABI, tests)
    helion = hl = PowerOfTwoFragment = None

DECODE = "dsa_indexer_decode"
PREFILL = "dsa_indexer_prefill"
DECODE_ROWS = (1, 6, 12, 24, 48, 96, 192, 384)  # rows (= requests x (1 + MTP k)) the lab tunes
PREFILL_M = (256, 1024, 4096, 8192)             # Q rows of a prefill chunk
MTP_ROWS = 6       # GLM-5.3 prod MTP k=5: vLLM SM120 flattens each request into 1 + k rows
DECODE_CTX = 32768  # context the decode buckets are tuned at (U[ctx/2, ctx] per request)
PAGE = 64           # = vLLM indexer block_size (= oxide T)
INT32_MAX = 2 ** 31 - 1

# Indexer dims are not in the console's model_dims() (it has the MLA heads):
# index_n_heads per architecture, mirroring the checkpoints' config.json
# (index_head_dim 128, index block 64, index_topk 2048 for both). The indexer
# is replicated, not TP-split: every rank runs all index heads.
INDEX_HEADS = {"GlmMoeDsaForCausalLM": 32, "DeepseekV32ForCausalLM": 64}


# ---------------------------------------------------------------------------
# Helion bodies (module-level, undecorated: the lab and compiled() wrap them)
# ---------------------------------------------------------------------------
def dsa_decode_logits(q: torch.Tensor, w: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor,
                      bt: torch.Tensor, cl: torch.Tensor, out: torch.Tensor,
                      splits: int) -> torch.Tensor:
    """Paged decode. q e4m3 [rows, H, 128]; w f32 [rows, H]; kv / ks the cache
    views (module doc); bt int32 [rows, W] (row r's own block-table row);
    cl int32 [rows]; out f32 [rows, n_cols] (written on [0, min(cl, n_cols)))."""
    rows = q.size(0)
    hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(w.size(1))
    hl.specialize(kv.size(1))
    n_cols = out.size(1)
    block_n = hl.register_block_size(16, 512)
    min_tiles = hl.register_tunable("min_tiles", PowerOfTwoFragment(1, 16, 2))
    split_div = hl.register_tunable("split_div", PowerOfTwoFragment(1, 16, 1))
    for r, s in hl.grid([rows, splits]):
        eff = (splits + split_div - 1) // split_div  # programs s >= eff find k0 >= hi
        hi = torch.clamp(cl[r], min=0, max=n_cols)
        tiles = (hi + 63) // 64
        per = torch.clamp((tiles + eff - 1) // eff, min=min_tiles) * 64
        k0 = s * per
        k1 = torch.minimum(k0 + per, hi)
        qr = q[r, :, :]                      # [H, 128] e4m3, once per program
        wr = w[r, :]
        for tile_n in hl.tile(k0, k1, block_size=block_n):
            n = tile_n.index
            pg = bt[r, n // 64]              # physical page of each token
            slot = n % 64
            kt = kv[pg * 66 + slot, :]       # [block_n, 128] e4m3 gather
            sc = ks[pg * 2112 + 2048 + slot]
            acc = hl.dot(qr, kt.T, out_dtype=torch.float32)  # [H, block_n] f32
            lg = torch.sum(torch.relu(acc) * wr[:, None], 0) * sc
            hl.store(out, [r, tile_n], lg)
    return out


def dsa_prefill_logits(q: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor, w: torch.Tensor,
                       lo: torch.Tensor, hi: torch.Tensor, out: torch.Tensor,
                       splits: int) -> torch.Tensor:
    """Prefill (contiguous K). q e4m3 [M, H, 128]; kv e4m3 [N, 128] (row stride
    free); ks f32 [N]; w f32 [M, H]; lo / hi int32 [M]; out f32 [M, N]
    (written on [max(lo, 0), min(hi, N)) per row)."""
    m = q.size(0)
    heads = hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(w.size(1))
    hl.specialize(kv.size(1))
    n = kv.size(0)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    min_tiles = hl.register_tunable("min_tiles", PowerOfTwoFragment(1, 16, 2))
    split_div = hl.register_tunable("split_div", PowerOfTwoFragment(1, 16, 1))
    for tile_m, tile_s in hl.tile([m, splits], block_size=[block_m, 1]):
        eff = (splits + split_div - 1) // split_div
        kl = torch.clamp(lo[tile_m], min=0)
        kh = torch.clamp(hi[tile_m], min=0, max=n)
        # window union of the live rows (rows past M load 0/0 = empty)
        glo = torch.amin(kl + (kh <= kl).to(torch.int32) * n, 0)
        ghi = torch.amax(kh, 0)
        tiles = (ghi - glo + 63) // 64
        per = torch.clamp((tiles + eff - 1) // eff, min=min_tiles) * 64
        k0 = glo + tile_s.begin * per
        k1 = torch.minimum(k0 + per, ghi)
        for tile_n in hl.tile(k0, k1, block_size=block_n):
            kt = kv[tile_n, :]               # [block_n, 128] e4m3
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for h in hl.grid(heads):
                d = hl.dot(q[tile_m, h, :], kt.T, out_dtype=torch.float32)
                acc = acc + torch.relu(d) * w[tile_m, h][:, None]
            val = acc * ks[tile_n][None, :]
            col = tile_n.index
            inside = (col[None, :] >= kl[:, None]) & (col[None, :] < kh[:, None])
            hl.store(out, [tile_m, tile_n], val, extra_mask=inside)
    return out


# ---------------------------------------------------------------------------
# host helpers shared by serving, the lab and the tests
# ---------------------------------------------------------------------------
def cache_views(cache):
    """(kv e4m3 [blocks * 66, 128], ks f32 [blocks * 2112]) of a contiguous
    uint8 [blocks, 64, (1,) 132] split-page cache (same storage)."""
    return (cache.view(torch.float8_e4m3fn).view(-1, 128),
            cache.reshape(-1).view(torch.float32))


def splits_for(units: int, cols: int, n_sms: int) -> int:
    """Host split cap: ~8 programs per SM over `units` row programs, at most one
    per 64-token tile, >= 2 (never a 0/1-specialized int). The body narrows it
    per row (split_div, min_tiles)."""
    return max(2, min(-(-cols // PAGE), -(-8 * n_sms // max(1, units))))


def decode_case(heads: int, rows: int) -> dict:
    reqs = rows // MTP_ROWS if rows % MTP_ROWS == 0 else rows
    return {"heads": heads, "rows": rows, "ctx": DECODE_CTX, "numtokens": reqs}


def prefill_case(heads: int, m: int) -> dict:
    return {"heads": heads, "m": m, "n": max(4 * m, 8192), "numtokens": m}


_CASE = {DECODE: (decode_case, DECODE_ROWS), PREFILL: (prefill_case, PREFILL_M)}
_BODY = {DECODE: dsa_decode_logits, PREFILL: dsa_prefill_logits}


def serving_case(kernel: str, heads: int, rows: int) -> dict:
    make_case, buckets = _CASE[kernel]
    return make_case(heads, hk.bucket(rows, buckets))


# ---------------------------------------------------------------------------
# serving entry
# ---------------------------------------------------------------------------
_WARM: set = set()


def _warm(kernel, heads, dev, kv, n_cols, mode):
    """Once per (kernel, heads, int64 cache): fail closed on missing configs
    (helion), then compile EVERY shipped bucket on tiny inputs with the
    serving size classes (rows 1 | >= 2; the real paged cache, so its int32 /
    int64 index class matches) — before any CUDA-graph capture."""
    big = kv is not None and kv.numel() > INT32_MAX
    key = (kernel, heads, big)
    if key in _WARM:
        return
    make_case, buckets = _CASE[kernel]
    cases = [make_case(heads, b) for b in buckets]
    missing = [c for c in cases if hk.config(kernel, c) is None]
    if mode == "helion" and missing:
        raise RuntimeError(f"{hk.MARKER} {di.HELION_ENV}=helion: no tuned {kernel} config on "
                           f"{hk.platform(dev.index or 0)} for "
                           + ", ".join(hk.case_key(c) for c in missing)
                           + " (tune in the autotune lab, or use auto)")
    if dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{hk.MARKER} {kernel}: first call inside CUDA-graph capture "
                           "(Helion must compile its buckets in an eager call first)")
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    f8 = torch.float8_e4m3fn
    for b, c in zip(buckets, cases):
        if hk.config(kernel, c) is None:
            continue
        rows = 1 if b == 1 else 2
        kern = hk.compiled(_BODY[kernel], kernel, c)
        q = torch.zeros((rows, heads, 128), dtype=f8, device=dev)
        w = torch.zeros((rows, heads), dtype=torch.float32, device=dev)
        one = torch.ones(rows, dtype=torch.int32, device=dev)
        if kernel == DECODE:
            bt = torch.zeros((rows, 2), dtype=torch.int32, device=dev)
            out = torch.empty((rows, n_cols), dtype=torch.float32, device=dev)
            kern(q, w, *cache_views(kv), bt, one, out, splits_for(rows, n_cols, n_sms))
        else:
            kc = torch.zeros((128, 128), dtype=f8, device=dev)
            ksc = torch.zeros(128, dtype=torch.float32, device=dev)
            out = torch.empty((rows, 128), dtype=torch.float32, device=dev)
            kern(q, kc, ksc, w, one - 1, one, out, splits_for(1, 128, n_sms))
    on = [b for b, c in zip(buckets, cases) if c not in missing]
    hk.log(f"{kernel} H={heads}: helion buckets {on}"
           + (f", oxide buckets {[b for b, c in zip(buckets, cases) if c in missing]}" if missing else ""))
    _WARM.add(key)


def dsa_indexer_logits_cuda(out, q, weights, kv, k_scale, block_tables, lo, hi, n_cols, group,
                            slices, bt_div, splits, min_tiles, stream_ptr):
    """Drop-in for ``_native.dsa_indexer_logits_cuda`` (same signature and
    contract, src/dsa_indexer_oxide.rs): paged when block_tables is given.
    group / slices / splits / min_tiles are the oxide plan — passed through
    on the oxide fallback, ignored by Helion (its config owns the schedule).
    Runs on torch's current stream (= stream_ptr in the lane)."""
    paged = block_tables is not None
    kernel = DECODE if paged else PREFILL
    if q.dim() != 3 or tuple(q.shape[1:]) not in ((32, 128), (64, 128)) or not q.is_contiguous():
        raise ValueError(f"dsa_indexer: q {tuple(q.shape)}: need contiguous [rows, 32|64, 128]")
    if paged and (kv.dtype != torch.uint8 or kv.shape[1] != PAGE or kv.shape[-1] != 132
                  or not kv.is_contiguous()):
        raise ValueError(f"dsa_indexer: paged kv {tuple(kv.shape)} {kv.dtype}: need contiguous "
                         "uint8 [blocks, 64, 132]")
    rows, heads = q.shape[0], q.shape[1]
    if not paged and (k_scale is None or kv.shape[0] != n_cols):
        raise ValueError(f"dsa_indexer: contiguous kv {tuple(kv.shape)} needs k_scale and n_cols == N")
    mode = di._backend()
    dev = q.device
    _warm(kernel, heads, dev, kv if paged else None, n_cols, mode)
    case = serving_case(kernel, heads, rows)
    if hk.config(kernel, case) is None:  # auto: oxide won (or was never beaten) here
        return di._native(dev).dsa_indexer_logits_cuda(
            out, q, weights, kv, k_scale, block_tables, lo, hi, n_cols, group, slices, bt_div,
            splits, min_tiles, stream_ptr)
    if rows == 0 or n_cols == 0:
        return None
    kern = hk.compiled(_BODY[kernel], kernel, case)
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    if paged:
        bt = block_tables if bt_div == 1 else block_tables.repeat_interleave(bt_div, 0)
        kern(q, weights[:rows], *cache_views(kv), bt, hi, out[:, :n_cols],
             splits_for(rows, n_cols, n_sms))
    else:
        if lo is None:
            lo = torch.zeros(rows, dtype=torch.int32, device=dev)
        kern(q, kv.view(torch.float8_e4m3fn), k_scale, weights[:rows], lo, hi, out[:, :n_cols],
             splits_for(-(-rows // 16), n_cols, n_sms))
    return None


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py) + the oracle's spec
# ---------------------------------------------------------------------------
def decode_args(pr, n_cols, n_sms):
    """Body args of a lane problem (di.make_decode, flattened layout)."""
    rows, heads = pr["rows"], pr["heads"]
    kv, ks = cache_views(pr["cache"])
    bt = pr["bt"] if pr["bt"].shape[0] == rows else pr["bt"].repeat_interleave(pr["next_n"], 0)
    out = torch.full((rows, n_cols), float("-inf"), dtype=torch.float32, device=kv.device)
    return (pr["q"].reshape(rows, heads, 128), pr["w"][:rows], kv, ks, bt.contiguous(),
            pr["cl"].reshape(-1).to(torch.int32).contiguous(), out, splits_for(rows, n_cols, n_sms))


def _dev_sms():
    dev = torch.device("cuda", torch.cuda.current_device())
    return dev, di._sms(dev)


def _decode_problem(case, seed):
    dev, n_sms = _dev_sms()
    rows, reqs = case["rows"], case["numtokens"]
    pr = di.make_decode(dev, case["heads"], reqs, rows // reqs, case["ctx"], True,
                        di.MAX_MODEL_LEN, seed=1000 + seed)
    return pr, n_sms


def decode_make(case, seed):
    pr, n_sms = _decode_problem(case, seed)
    return decode_args(pr, di.MAX_MODEL_LEN, n_sms)


def decode_reference(q, w, kv, ks, bt, cl, out, splits):
    """f64 logits + |term| magnitudes (di.ref_logits through the split-page
    gather), NaN outside each row's [0, min(cl, n_cols)) window."""
    rows, n_cols = out.shape
    cache = kv.view(torch.uint8).view(-1, PAGE, 132)
    val = torch.full((rows, n_cols), float("nan"), dtype=torch.float64, device=out.device)
    mag = torch.zeros((rows, n_cols), dtype=torch.float64, device=out.device)
    for r in range(rows):
        n = min(max(int(cl[r]), 0), n_cols)
        if n:
            k, s = di.gather_paged(cache, bt[r], n)
            val[r, :n], mag[r, :n] = di.ref_logits(q[r], w[r], k, s)
    return val, mag


def decode_make_baseline(case, seed):
    """The same problem in the oxide layout + its MTP row group (oracle style)."""
    pr, n_sms = _decode_problem(case, seed)
    out = torch.full((pr["rows"], di.MAX_MODEL_LEN), float("-inf"), device=pr["q"].device)
    return (out, pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], case["rows"] // case["numtokens"])


def _oxide(out, q, w, kv, k_scale, bt, lo, hi, group):
    """The oxide kernel with the lane's launch plan (no backend routing)."""
    dev = q.device
    rows, n = out.shape
    p = di.plan(rows, group, n, di._sms(dev), di._min_tiles())
    di._native(dev).dsa_indexer_logits_cuda(
        out, q, w, kv, k_scale, bt, lo, hi, n, p["group"], p["slices"], 1, p["splits"],
        p["min_tiles"], torch.cuda.current_stream(dev).cuda_stream)
    return out


def decode_baseline(out, q, cache, w, cl, bt, group):
    return _oxide(out, q.reshape(out.shape[0], q.shape[-2], 128), w, cache.squeeze(2), None, bt,
                  None, cl.reshape(-1).to(torch.int32), group)


def prefill_problem(heads, m, n, dev, seed=0):
    """The oracle's prefill: two requests packed in K (request 0 = K [0, n/2)
    with the first m/2 rows, request 1 = the rest); row i of a request with L
    keys and Mq rows sees [a, a + L - Mq + 1 + i)."""
    g = torch.Generator(device="cpu").manual_seed(m + n + seed)
    q = (torch.randn((m, heads, 128), generator=g) * 2).to(torch.float8_e4m3fn).to(dev)
    kv = (torch.randn((n, 128), generator=g) * 0.5).to(torch.float8_e4m3fn).to(dev)
    ksc = (2.0 ** torch.randint(-8, -3, (n,), generator=g).float()).to(dev)
    w = (torch.randn((m, heads), generator=g) * 0.02).to(dev)
    half = m // 2
    ks_l, ke_l = [], []
    for a, ln, mq in ((0, n // 2, half), (n // 2, n - n // 2, m - half)):
        ks_l += [a] * mq
        ke_l += [a + ln - mq + 1 + i for i in range(mq)]
    lo = torch.tensor(ks_l, dtype=torch.int32, device=dev)
    hi = torch.tensor(ke_l, dtype=torch.int32, device=dev)
    out = torch.full((m, n), float("-inf"), dtype=torch.float32, device=dev)
    return q, kv, ksc, w, lo, hi, out


def prefill_make(case, seed):
    dev, n_sms = _dev_sms()
    args = prefill_problem(case["heads"], case["m"], case["n"], dev, seed)
    return (*args, splits_for(-(-case["m"] // 16), case["n"], n_sms))


def prefill_reference(q, kv, ks, w, lo, hi, out, splits):
    """f64 logits + magnitudes per row window [max(lo, 0), min(hi, N)), NaN elsewhere."""
    m, n = out.shape
    val = torch.full((m, n), float("nan"), dtype=torch.float64, device=out.device)
    mag = torch.zeros((m, n), dtype=torch.float64, device=out.device)
    for r in range(m):
        a, b = max(int(lo[r]), 0), min(int(hi[r]), n)
        if a < b:
            val[r, a:b], mag[r, a:b] = di.ref_logits(q[r], w[r], kv[a:b], ks[a:b])
    return val, mag


def prefill_baseline(q, kv, ks, w, lo, hi, out, splits):
    return _oxide(out, q, w, kv, ks, None, lo, hi, di.PREFILL_GROUP)


# Helion's autotuner checks every candidate against a baseline run over the
# WHOLE output; the windows are the only defined part (the kernel never writes
# elsewhere). Lab-only settings (serving never autotunes): the baseline is the
# oxide kernel on the body's own args, and the comparison is restricted to the
# columns the baseline wrote (finite), like the consumer. The lab's check()
# still proves the winner wrote nothing outside the windows.
def _decode_tune_baseline(q, w, kv, ks, bt, cl, out, splits):
    return _oxide(out, q, w, kv.view(torch.uint8).view(-1, PAGE, 132), None, bt, None, cl, 1)


def _prefill_tune_baseline(q, kv, ks, w, lo, hi, out, splits):
    return prefill_baseline(q, kv, ks, w, lo, hi, out, splits)


def _tune_check(actual, expected):
    """Autotuner accuracy check: float tensors compared where `expected` is
    finite (rtol 1e-3, atol 1e-3 x its RMS); raises AssertionError."""
    from torch.utils._pytree import tree_flatten
    for a, e in zip(tree_flatten(actual)[0], tree_flatten(expected)[0], strict=True):
        if isinstance(e, torch.Tensor) and e.dtype == torch.float32:
            m = torch.isfinite(e)
            ef = e[m]
            rms = float(ef.double().square().mean().sqrt()) if ef.numel() else 0.0
            torch.testing.assert_close(a[m], ef, rtol=1e-3, atol=1e-3 * max(rms, 1e-6))


def check(out, ref):
    """The consumer's view: only each row's window is read, so only it is
    compared (|out - f64| <= TOL * sum|terms|, finite); everything outside
    must still hold make()'s -inf (nothing written there)."""
    val, mag = ref
    inside = ~torch.isnan(val)
    if not bool(torch.isneginf(out[~inside]).all()):
        return False, "wrote outside the [lo, hi) windows"
    o = out[inside].double()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite logits inside the windows"
    err = float(((o - val[inside]).abs() / (mag[inside] + 1e-30)).max()) if o.numel() else 0.0
    return err <= di.TOL, f"max err/A {err:.2e} (tol {di.TOL:.0e})"


def _cases(kernel):
    def cases(model):
        heads = INDEX_HEADS.get((model or {}).get("architecture"))
        if heads is None:
            return []
        make_case, buckets = _CASE[kernel]
        return [make_case(heads, b) for b in buckets]
    return cases


KERNELS = [
    {"name": DECODE, "body": dsa_decode_logits, "cases": _cases(DECODE),
     "make": decode_make, "reference": decode_reference, "check": check,
     "make_baseline": decode_make_baseline, "baseline": decode_baseline,
     "baseline_name": "oxide dsa_indexer", "replaces": [r"dsa_logits"],
     "settings": {"autotune_baseline_fn": _decode_tune_baseline,
                  "autotune_baseline_accuracy_check_fn": _tune_check}},
    # the oxide entry is ONE kernel for decode and prefill: the decode-dominated
    # profile op is credited to the decode port only (no double count)
    {"name": PREFILL, "body": dsa_prefill_logits, "cases": _cases(PREFILL),
     "make": prefill_make, "reference": prefill_reference, "check": check,
     "baseline": prefill_baseline, "baseline_name": "oxide dsa_indexer", "replaces": [],
     "settings": {"autotune_baseline_fn": _prefill_tune_baseline,
                  "autotune_baseline_accuracy_check_fn": _tune_check}},
]


def oracle_lines():
    """Load-time gate for the Helion backend: every SHIPPED case of this GPU
    runs its compiled config on make(case) vs the f64 spec (fatal on mismatch)."""
    lines = []
    for k in KERNELS:
        for key in sorted(hk.table(k["name"])):
            case = json.loads(key)
            args = k["make"](case, 0)
            ok, msg = check(hk.compiled(k["body"], k["name"], case)(*args), k["reference"](*args))
            lines.append(f"helion {k['name']} {key}: {msg} {'OK' if ok else 'FAIL'}")
            di._log(lines[-1])
            if not ok:
                raise RuntimeError(f"{di.MARKER} DSA-INDEXER HELION ORACLE FAIL: {lines[-1]}")
            del args
            torch.cuda.empty_cache()
    return lines
