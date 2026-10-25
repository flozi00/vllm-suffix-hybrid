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

Paged layout: the cache is uint8 [blocks, P, 132] split pages (P x 128 e4m3
values, then P f32 scales; P = 64 for GLM / DeepSeek-V4.1 ratio-2 layers, 128
for V4.1's ratio-1 layers) at a free page stride S (V4.1 packs the indexer
page beside its MLA pages in one block). The bodies take two VIEWS of that
storage: ``kv`` e4m3 [*, 128] (token t of page p = row p * S/128 + t) and
``ks`` f32 [*] (scale of token t of page p = p * S/4 + 32 P + t); the ints
``page`` P, ``prow`` S/128, ``sp`` S/4 ride along (cache_views). Every token
is gathered through its own block-table entry, so a KV tile may straddle pages
and block_n is free.

DeepSeek-V4.1 two-level selection (``dsa_cand_logits``): consumer indexers
mask their logits to the candidate source's top 2048 blocks of 8 positions
before the row top-k. The fused body scores ONLY those 16384 positions and
writes COMPRESSED logits [rows, 2048 * 8] (slot j = position cand[j // 8] * 8
+ j % 8), so the dense pass, vLLM's full-width mask pass and the top-k over the
whole context all shrink to the candidate set (kernels/dsa_candidates.py
wires it into vLLM's sparse_attn_indexer and maps the top-k slots back).

Prefill: vLLM nightly (#61149, 2026-10-12) returns COMPRESSED logits [M, W]
(row i holds K positions [ks[i], ke[i]) from column 0); the body writes column
n - off[r] (off = ks: compressed; 0: the old absolute [M, N] contract).

Design: one program = one decode ROW (decode) / a block of Q rows (prefill)
x one split of that row's window; the split range comes from the window
itself on device (``per = max(ceil(tiles / eff_splits), min_tiles)`` 64-token
tiles), so the grid is a function of host shapes only (CUDA-graph safe) and
short contexts use fewer, longer programs like oxide's min_tiles. Tunables:
block_n (KV tokens per step), block_m (prefill Q rows), ``min_tiles``,
``split_div`` (fewer effective splits than the host cap), num_warps /
num_stages (the gathered K loads pipeline), pid_type / loop order of the
(row, split) grid, eviction policy per load.
The per-row body reads K once per ROW (the MTP rows of a request meet in L2).
``dsa_group_logits`` is the row-group body: one program sweeps the union window of
G consecutive rows and scores them against the same K tiles (each row still
addresses K through its own block-table row, so any grouping is exact; the 2nd..Gth
load of a tile is an L1 hit): K traffic / G at DSpark's 1 + 5 rows. Tuned per request
count and context bucket (up to 524288); serving uses the longest bucket's config
at every context, because a decode graph is captured once.

Graph discipline (kernels/dsa_capture.py): every launch goes through
``cap.launch`` (an unwarmed specialisation class cannot JIT inside a capture),
every body writes only columns of its own range (no barriers, no semaphores, no
cross-program merge), scratch comes from torch's allocator, and every body's
config space is bounded (rolled row loop, autotune_config_filter).

Serving: ``dsa_indexer_logits_cuda`` has the oxide op's exact signature; the
lane calls it instead of ``_native.dsa_indexer_logits_cuda`` when
SUFFIX_HELION_DSA_INDEXER != oxide (routed per rows bucket; auto -> oxide
where no config is shipped).
"""
from __future__ import annotations

import json

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.kernels import dsa_capture as cap
from suffix_hybrid.kernels import dsa_indexer as di
from suffix_hybrid.kernels import dsa_select as ds

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no-Helion host: module stays importable (lab ABI, tests)
    helion = hl = PowerOfTwoFragment = None

DECODE = "dsa_indexer_decode"
PREFILL = "dsa_indexer_prefill"
CANDK = "dsa_indexer_cand"
GROUP = "dsa_indexer_group"
SELECT = "dsa_indexer_select"
DECODE_ROWS = (1, 6, 12, 24, 48, 96, 192, 384)  # rows (= requests x (1 + MTP k)) the lab tunes
PREFILL_M = (256, 1024, 4096, 8192)             # Q rows of a prefill chunk
MTP_ROWS = 6       # GLM-5.3 prod MTP k=5: vLLM SM120 flattens each request into 1 + k rows
DECODE_CTX = 32768  # context the decode buckets are tuned at (U[ctx/2, ctx] per request)
PAGE = 64           # = vLLM indexer block_size (= oxide T)
# Row-group decode (GROUP): one program scores the G = 1 + k MTP rows of a request
# against ONE sweep of its K (DSpark k=5 -> G = 6). Tuned per request count and
# context bucket: the long bucket is the V4.1 target (524288), where the K scan
# is the layer's whole cost. Long contexts are KV-capacity bound, so only few
# requests are tuned there.
GROUP_G = MTP_ROWS
GROUP_REQS = {32768: (1, 2, 4, 8, 16, 32), 524288: (1, 2, 4)}
GROUP_CTX = tuple(sorted(GROUP_REQS))
INT32_MAX = 2 ** 31 - 1

# Indexer dims are not in the console's model_dims() (it has the MLA heads):
# index_n_heads per architecture, mirroring the checkpoints' config.json
# (index_head_dim 128; index_topk 2048 for GLM / DSv3.2, 512 for V4.1). The
# indexer is replicated, not TP-split: every rank runs all index heads.
INDEX_HEADS = {"GlmMoeDsaForCausalLM": 32, "DeepseekV32ForCausalLM": 64,
               "DeepseekV41ForCausalLM": 32}
# Two-level candidate selection (DeepSeek-V4.1 config.json: candidate_topk_blocks,
# candidate_block_size): the archs whose consumer indexers the fused body serves.
CAND = {"DeepseekV41ForCausalLM": (2048, 8)}
CAND_CTX = 65536    # > 2048 x 8: the candidate set, not the context, sizes the work


# ---------------------------------------------------------------------------
# Helion bodies (module-level, undecorated: the lab and compiled() wrap them)
# ---------------------------------------------------------------------------
def dsa_decode_logits(q: torch.Tensor, w: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor,
                      bt: torch.Tensor, cl: torch.Tensor, out: torch.Tensor, page: int,
                      prow: int, sp: int, splits: int) -> torch.Tensor:
    """Paged decode. q e4m3 [rows, H, 128]; w f32 [rows, H]; kv / ks / page /
    prow / sp the cache views (module doc); bt int32 [rows, W] (row r's own
    block-table row); cl int32 [rows]; out f32 [rows, n_cols] (written on
    [0, min(cl, n_cols)))."""
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
            pg = bt[r, n // page]            # physical page of each token
            slot = n % page
            kt = kv[pg * prow + slot, :]     # [block_n, 128] e4m3 gather
            sc = ks[pg * sp + page * 32 + slot]
            acc = hl.dot(qr, kt.T, out_dtype=torch.float32)  # [H, block_n] f32
            lg = torch.sum(torch.relu(acc) * wr[:, None], 0) * sc
            hl.store(out, [r, tile_n], lg)
    return out


def dsa_group_logits(q: torch.Tensor, w: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor,
                     bt: torch.Tensor, cl: torch.Tensor, out: torch.Tensor, page: int,
                     prow: int, sp: int, splits: int) -> torch.Tensor:
    """Row-group paged decode: dsa_decode_logits for G consecutive rows per
    program, K tile loaded once per row-group sweep. q e4m3 [R, G, H, 128]
    (R programs x G rows; row r * G + g; rows past out's are padding with
    cl 0); w f32 [R, G, H]; kv / ks / page / prow / sp the cache views; bt
    int32 [rows, W] (every row's own block-table row: groups need not
    coincide with requests, rows of one request just share pages); cl int32
    [R * G]; out f32 [rows, n_cols] (row written on [0, min(cl, n_cols)),
    nothing else).

    The program sweeps the UNION window of its rows; each row addresses K
    through its own block-table row and masks its own length. Rows of one
    request resolve to the same cache lines, so the 2nd..Gth load of a tile is
    an L1 hit and HBM sees the K tile once (K4 on silicon decides whether L1
    delivers: oxide's page-dedup group is the baseline). The row loop is a
    rolled device loop (not unrolled): the config space (one eviction /
    indexing entry per memory op) stays that of the per-row body and the
    compile stays short. Dead lanes (past the row's own length) issue no
    block-table / K read for the row: the table gather is masked, K comes from
    page 0, the store is masked. No cross-program ordering anywhere: every
    program writes only columns of its own range (graph-capture safe)."""
    reqs = q.size(0)
    g = hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(q.size(3))
    hl.specialize(w.size(2))
    hl.specialize(kv.size(1))
    n_cols = out.size(1)
    nrows = out.size(0)
    block_n = hl.register_block_size(16, 256)
    min_tiles = hl.register_tunable("min_tiles", PowerOfTwoFragment(1, 16, 2))
    split_div = hl.register_tunable("split_div", PowerOfTwoFragment(1, 16, 1))
    for r, s in hl.grid([reqs, splits]):
        eff = (splits + split_div - 1) // split_div
        r0 = r * g
        hi = torch.clamp(cl[r0], min=0, max=n_cols)
        for gi in hl.grid(g):
            hi = torch.maximum(hi, torch.clamp(cl[r0 + gi], min=0, max=n_cols))
        tiles = (hi + 63) // 64
        per = torch.clamp((tiles + eff - 1) // eff, min=min_tiles) * 64
        k0 = s * per
        k1 = torch.minimum(k0 + per, hi)
        for tile_n in hl.tile(k0, k1, block_size=block_n):
            n = tile_n.index
            slot = n % page
            for gi in hl.grid(g):
                lim = torch.clamp(cl[r0 + gi], min=0, max=n_cols)
                rr = min(r0 + gi, nrows - 1)     # padding rows: any real row (they are dead)
                live = n < lim
                pg = hl.load(bt, [rr, n // page], extra_mask=live)   # dead lanes: page 0, no table read
                kt = kv[pg * prow + slot, :]     # [block_n, 128] e4m3 gather
                sc = ks[pg * sp + page * 32 + slot]
                acc = hl.dot(q[r, gi, :, :], kt.T, out_dtype=torch.float32)  # [H, block_n]
                lg = torch.sum(torch.relu(acc) * w[r, gi, :][:, None], 0) * sc
                hl.store(out, [rr, tile_n], lg, extra_mask=live)
    return out


def dsa_group_select(q: torch.Tensor, w: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor,
                     bt: torch.Tensor, cl: torch.Tensor, thr: torch.Tensor, comp_v: torch.Tensor,
                     comp_i: torch.Tensor, cnt: torch.Tensor, page: int, prow: int, sp: int,
                     splits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LAB-ONLY EXPERIMENT (not wired, not capture-safe as written): dsa_group_logits
    with a threshold-compacting epilogue. Instead of the dense [rows, n_cols]
    logits only the survivors logit >= thr[row] of each row's window are
    written, packed to the front of comp_v / comp_i (logit, position) behind a
    per-row atomic cursor cnt (int32 [rows]; the caller zeroes it before every
    launch -- a memset node in a graph). The cursor is a lock-free slot claim
    (no program ever waits on another), but the packed ORDER is the programs'
    race, so replays are not bit-identical in order, and the owner's rule for
    graph-captured paths is two-stage partials + a separate merge: if the
    lab shows this select winning (K4), port it to per-tile counts -> scan ->
    scatter before any wiring. thr is a lower bound of the row's top-k
    threshold (kernels/dsa_select.py proves why a warm-start bound is exact);
    thr = -inf keeps the whole window. comp_* are [rows, n_cols]: a row cannot
    have more survivors than positions (the pos < n_cols guard also makes a
    stale cursor harmless). Shapes / views / padding rows as dsa_group_logits."""
    reqs = q.size(0)
    g = hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(q.size(3))
    hl.specialize(w.size(2))
    hl.specialize(kv.size(1))
    n_cols = comp_v.size(1)
    nrows = comp_v.size(0)
    block_n = hl.register_block_size(16, 256)
    min_tiles = hl.register_tunable("min_tiles", PowerOfTwoFragment(1, 16, 2))
    split_div = hl.register_tunable("split_div", PowerOfTwoFragment(1, 16, 1))
    for r, s in hl.grid([reqs, splits]):
        eff = (splits + split_div - 1) // split_div
        r0 = r * g
        hi = torch.clamp(cl[r0], min=0, max=n_cols)
        for gi in hl.grid(g):
            hi = torch.maximum(hi, torch.clamp(cl[r0 + gi], min=0, max=n_cols))
        tiles = (hi + 63) // 64
        per = torch.clamp((tiles + eff - 1) // eff, min=min_tiles) * 64
        k0 = s * per
        k1 = torch.minimum(k0 + per, hi)
        for tile_n in hl.tile(k0, k1, block_size=block_n):
            n = tile_n.index
            slot = n % page
            for gi in hl.grid(g):
                lim = torch.clamp(cl[r0 + gi], min=0, max=n_cols)
                rr = min(r0 + gi, nrows - 1)
                live = n < lim
                pg = hl.load(bt, [rr, n // page], extra_mask=live)
                kt = kv[pg * prow + slot, :]
                sc = ks[pg * sp + page * 32 + slot]
                acc = hl.dot(q[r, gi, :, :], kt.T, out_dtype=torch.float32)
                lg = torch.sum(torch.relu(acc) * w[r, gi, :][:, None], 0) * sc
                keep = live & (lg >= thr[rr])
                ones = keep.to(torch.int32)
                base = hl.atomic_add(cnt, [rr], torch.sum(ones, 0))
                pos = base + torch.cumsum(ones, 0) - 1
                keep = keep & (pos < n_cols)
                hl.store(comp_v, [rr, pos], lg, extra_mask=keep)
                hl.store(comp_i, [rr, pos], n.to(torch.int32), extra_mask=keep)
    return comp_v, comp_i, cnt


def dsa_prefill_logits(q: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor, w: torch.Tensor,
                       lo: torch.Tensor, hi: torch.Tensor, off: torch.Tensor, out: torch.Tensor,
                       splits: int) -> torch.Tensor:
    """Prefill (contiguous K). q e4m3 [M, H, 128]; kv e4m3 [N, 128] (row stride
    free); ks f32 [N]; w f32 [M, H]; lo / hi / off int32 [M]; out f32
    contiguous [M, W]: K position n of row r (max(lo, 0) <= n < min(hi, N))
    lands in column n - off[r] (off = lo: vLLM's compressed logits; 0: the
    absolute [M, N] layout), columns outside [0, W) are dropped."""
    m = q.size(0)
    heads = hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(w.size(1))
    hl.specialize(kv.size(1))
    n = kv.size(0)
    width = out.size(1)
    flat = out.view(-1)
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
            n_ = tile_n.index
            c = n_[None, :] - off[tile_m][:, None]
            inside = (n_[None, :] >= kl[:, None]) & (n_[None, :] < kh[:, None]) \
                & (c >= 0) & (c < width)
            hl.store(flat, [tile_m.index[:, None] * width + c], val, extra_mask=inside)
    return out


def dsa_cand_logits(q: torch.Tensor, w: torch.Tensor, kv: torch.Tensor, ks: torch.Tensor,
                    bt: torch.Tensor, cl: torch.Tensor, cand: torch.Tensor, out: torch.Tensor,
                    page: int, prow: int, sp: int, splits: int) -> torch.Tensor:
    """Fused two-level selection (decode consumers). q / w / kv / ks / bt / cl /
    page / prow / sp as dsa_decode_logits; cand int32 [rows, C] candidate block
    ids (-1 pad, valid ones first: vLLM's sorted top-k); out f32 [rows, C * B]:
    slot j of row r = logit of position p = cand[r, j // B] * B + j % B, -inf
    where the block is -1 or p >= cl[r]. Written on [0, min(ceil(cl / B), C) * B)
    — every slot that can hold a candidate (cand_lens)."""
    rows = q.size(0)
    hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    hl.specialize(w.size(1))
    hl.specialize(kv.size(1))
    ncand = hl.specialize(cand.size(1))
    bsz = hl.specialize(out.size(1)) // ncand
    block_n = hl.register_block_size(16, 512)
    min_tiles = hl.register_tunable("min_tiles", PowerOfTwoFragment(1, 16, 2))
    split_div = hl.register_tunable("split_div", PowerOfTwoFragment(1, 16, 1))
    for r, s in hl.grid([rows, splits]):
        eff = (splits + split_div - 1) // split_div
        hi = torch.clamp(cl[r], min=0)
        nslot = torch.clamp((hi + bsz - 1) // bsz, max=ncand) * bsz
        tiles = (nslot + 63) // 64
        per = torch.clamp((tiles + eff - 1) // eff, min=min_tiles) * 64
        k0 = s * per
        k1 = torch.minimum(k0 + per, nslot)
        qr = q[r, :, :]
        wr = w[r, :]
        for tile_j in hl.tile(k0, k1, block_size=block_n):
            j = tile_j.index
            blk = cand[r, j // bsz]
            pos = blk * bsz + j % bsz
            ok = (blk >= 0) & (pos < hi)
            pos = torch.where(ok, pos, 0)    # dead slots read token 0 (row has >= 1 page)
            pg = bt[r, pos // page]
            slot = pos % page
            kt = kv[pg * prow + slot, :]
            sc = ks[pg * sp + page * 32 + slot]
            acc = hl.dot(qr, kt.T, out_dtype=torch.float32)
            lg = torch.sum(torch.relu(acc) * wr[:, None], 0) * sc
            out[r, tile_j] = torch.where(ok, lg, float("-inf"))
    return out


# ---------------------------------------------------------------------------
# host helpers shared by serving, the lab and the tests
# ---------------------------------------------------------------------------
layout_reason = di.layout_reason   # torch-only; lives in the lane so the oxide backend never imports this module


def storage_span(cache) -> int:
    """Elements (bytes) the paged cache's storage spans, page-stride gaps included:
    what Helion's int32 / int64 index class is decided on (the flat views below)."""
    c = cache.squeeze(2) if cache.dim() == 4 else cache
    return (c.shape[0] - 1) * c.stride(0) + c.shape[1] * 132


def cache_views(cache):
    """(kv e4m3 [*, 128], ks f32 [*], page P, prow S / 128, sp S / 4) of a uint8
    [blocks, P, (1,) 132] split-page cache with page stride S bytes (same
    storage; the caller checked layout_reason)."""
    c = cache.squeeze(2) if cache.dim() == 4 else cache
    nb, page, s = c.shape[0], c.shape[1], c.stride(0)
    flat = c.as_strided(((nb - 1) * s + page * 132,), (1,))
    return flat.view(torch.float8_e4m3fn).view(-1, 128), flat.view(torch.float32), page, s // 128, s // 4


def gather_views(kv, ks, page, prow, sp, bt_row, n):
    """(K e4m3 [n, 128], scale f32 [n]) of a sequence's first n tokens through
    the cache views (the bodies' addressing, read back on the host)."""
    t = torch.arange(n, device=kv.device)
    pg = bt_row[t // page].long()
    return kv[pg * prow + t % page], ks[pg * sp + page * 32 + t % page]


def splits_for(units: int, cols: int, n_sms: int) -> int:
    """Host split cap: ~8 programs per SM over `units` row programs, at most one
    per 64-token tile, a multiple of 16 (>= 16: one Triton specialisation class
    for the grid width whatever the batch, see kernels/dsa_capture; surplus
    programs find an empty range and exit). The body narrows it per row
    (split_div, min_tiles)."""
    return -(-max(2, min(-(-cols // PAGE), -(-8 * n_sms // max(1, units)))) // 16) * 16


def _reqs(rows):
    return rows // MTP_ROWS if rows % MTP_ROWS == 0 else rows


FMT_V41 = "v41"   # case-key marker of DeepSeek-V4.1 geometry: its own configs, tuned against the STOCK path


def _tag(case: dict, fmt) -> dict:
    """GLM-5.3 keys stay exactly as shipped; V4.1 keys carry fmt so a config tuned
    against oxide (GLM) can never serve V4.1 and V4.1 configs only exist where they
    beat the stock path."""
    return {**case, "fmt": fmt} if fmt else case


def decode_case(heads: int, rows: int, fmt=None) -> dict:
    return _tag({"heads": heads, "rows": rows, "ctx": DECODE_CTX, "numtokens": _reqs(rows)}, fmt)


def prefill_case(heads: int, m: int, fmt=None) -> dict:
    return _tag({"heads": heads, "m": m, "n": max(4 * m, 8192), "numtokens": m}, fmt)


def cand_case(heads: int, rows: int, ncand: int = 2048, bsz: int = 8) -> dict:
    return {"heads": heads, "rows": rows, "ctx": CAND_CTX, "cand": ncand, "cbs": bsz,
            "numtokens": _reqs(rows)}


def group_case(heads: int, reqs: int, ctx: int, g: int = GROUP_G, fmt=None) -> dict:
    return _tag({"heads": heads, "rows": reqs * g, "group": g, "ctx": ctx, "numtokens": reqs}, fmt)


def ctx_bucket(ctx) -> int:
    """The tuned GROUP context bucket a decode of up to `ctx` tokens runs as
    (unknown ctx -> the longest bucket: a config tuned for 512k also serves
    short contexts, only slower than the short bucket's)."""
    return hk.bucket(ctx, GROUP_CTX) if ctx else GROUP_CTX[-1]


def group_serving_case(heads: int, rows: int, g: int, ctx, fmt=None) -> dict:
    cb = ctx_bucket(ctx)
    return group_case(heads, hk.bucket(-(-rows // g), GROUP_REQS[cb]), cb, g, fmt)


def has_group(heads: int, rows: int, g: int, ctx, fmt=None) -> bool:
    """A shipped row-group config for this decode (g must be the tuned G)."""
    return g == GROUP_G and rows > 0 \
        and hk.config(GROUP, group_serving_case(heads, rows, g, ctx, fmt)) is not None


def n_cols_for(case: dict) -> int:
    """Logits width of a lab problem: the lane's MAX_MODEL_LEN, or the context
    when that is longer (V4.1 serves 524288)."""
    return max(di.MAX_MODEL_LEN, case["ctx"])


_CASE = {DECODE: (decode_case, DECODE_ROWS), PREFILL: (prefill_case, PREFILL_M),
         CANDK: (cand_case, DECODE_ROWS)}
_BODY = {DECODE: dsa_decode_logits, PREFILL: dsa_prefill_logits, CANDK: dsa_cand_logits,
         GROUP: dsa_group_logits}


def all_cases(kernel: str, heads: int, **kw) -> list:
    """[(dummy-warm size, case)] of every tuned bucket of `kernel`."""
    if kernel == GROUP:
        g, fmt = kw.get("g", GROUP_G), kw.get("fmt")
        return [(r, group_case(heads, r, c, g, fmt)) for c in GROUP_CTX for r in GROUP_REQS[c]]
    make_case, buckets = _CASE[kernel]
    return [(b, make_case(heads, b, **kw)) for b in buckets]


def serving_case(kernel: str, heads: int, rows: int, **kw) -> dict:
    make_case, buckets = _CASE[kernel]
    return make_case(heads, hk.bucket(rows, buckets), **kw)


def has_config(kernel: str, heads: int, rows: int, **kw) -> bool:
    return hk.config(kernel, serving_case(kernel, heads, rows, **kw)) is not None


# ---------------------------------------------------------------------------
# serving entries
# ---------------------------------------------------------------------------
_WARM: set = set()
# (programs, rows) of the group warm-ups: Triton specialises the grid / row
# counts on == 1, % 16 == 0, other -- one pair per reachable class combination
# at G = 6 (rows 1 | 6 | 12 | 48 | 96; every vLLM capture size lands in one).
WARM_GROUPS = ((1, 1), (1, GROUP_G), (2, 2 * GROUP_G), (8, 8 * GROUP_G), (16, 16 * GROUP_G))
WARM_ROWS = (1, 2, 16)     # the same classes for the per-row bodies


def _like(real: int, cap_: int = 4096) -> int:
    """A small size in the Triton specialisation class of `real` (== 1, % 16, other)."""
    if real <= cap_:
        return real
    return 16 * (cap_ // 16) if real % 16 == 0 else 16 * (cap_ // 16) + 1


def _launch(kern, *args):
    return cap.launch(kern, *args)


def _warm(kernel, heads, dev, kv, n_cols, mode, bt_width=2, force=False, **kw):
    """Once per (kernel, heads, int64 cache, size classes, kw): fail closed on
    missing configs (helion; the GROUP kernel is an optimisation over DECODE and
    may be absent) and on configs outside the lane's bounded space, then compile
    EVERY shipped bucket on tiny inputs in every row-count class the serving
    shapes can land in (the real paged cache, so its int32 / int64 index class
    and alignment match) -- before any CUDA-graph capture. A later launch of a
    class this missed is refused inside a capture (cap.launch), never compiled.
    `force` (warm_all): run the dummy launches again even when this size class
    was warmed on another cache tensor (the capture guard keys on the tensors)."""
    big = kv is not None and storage_span(kv) > INT32_MAX
    key = (kernel, heads, big, cap._cls(n_cols), cap._cls(bt_width), tuple(sorted(kw.items())))
    if key in _WARM and not force:
        return
    pairs = all_cases(kernel, heads, **kw)
    buckets, cases = [b for b, _ in pairs], [c for _, c in pairs]
    missing = [c for c in cases if hk.config(kernel, c) is None]
    if mode == "helion" and missing and kernel != GROUP:
        raise RuntimeError(f"{hk.MARKER} {di.HELION_ENV}=helion: no tuned {kernel} config on "
                           f"{hk.platform(dev.index or 0)} for "
                           + ", ".join(hk.case_key(c) for c in missing)
                           + " (tune in the autotune lab, or use auto)")
    for c in cases:
        bad = cap.serving_violation(hk.config(kernel, c))
        if bad:
            raise RuntimeError(f"{hk.MARKER} shipped {kernel} config for {hk.case_key(c)}: {bad}")
    if cap.capturing():
        raise RuntimeError(f"{hk.MARKER} {kernel}: first call inside CUDA-graph capture "
                           "(Helion must compile its buckets in an eager call first)")
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    f8 = torch.float8_e4m3fn
    cols, width = _like(n_cols), _like(bt_width)
    for b, c in zip(buckets, cases):
        if hk.config(kernel, c) is None:
            continue
        kern = hk.compiled(_BODY[kernel], kernel, c)
        if kernel == GROUP:
            g = c["group"]
            kv_, ks_, page, prow, sp = cache_views(kv)
            for groups, rows in WARM_GROUPS:
                q = torch.zeros((groups, g, heads, 128), dtype=f8, device=dev)
                w = torch.zeros((groups, g, heads), dtype=torch.float32, device=dev)
                one = torch.ones(groups * g, dtype=torch.int32, device=dev)
                bt = torch.zeros((rows, width), dtype=torch.int32, device=dev)
                out = torch.empty((rows, cols), dtype=torch.float32, device=dev)
                _launch(kern, q, w, kv_, ks_, bt, one, out, page, prow, sp, splits_for(groups, n_cols, n_sms))
            continue
        for rows in WARM_ROWS if kernel != PREFILL else (1, 2):
            q = torch.zeros((rows, heads, 128), dtype=f8, device=dev)
            w = torch.zeros((rows, heads), dtype=torch.float32, device=dev)
            one = torch.ones(rows, dtype=torch.int32, device=dev)
            bt = torch.zeros((rows, width), dtype=torch.int32, device=dev)
            if kernel == DECODE:
                kv_, ks_, page, prow, sp = cache_views(kv)
                out = torch.empty((rows, cols), dtype=torch.float32, device=dev)
                _launch(kern, q, w, kv_, ks_, bt, one, out, page, prow, sp, splits_for(rows, n_cols, n_sms))
            elif kernel == CANDK:
                kv_, ks_, page, prow, sp = cache_views(kv)
                cand = torch.zeros((rows, c["cand"]), dtype=torch.int32, device=dev)
                out = torch.empty((rows, c["cand"] * c["cbs"]), dtype=torch.float32, device=dev)
                _launch(kern, q, w, kv_, ks_, bt, one, cand, out, page, prow, sp,
                        splits_for(rows, c["cand"] * c["cbs"], n_sms))
            else:
                kc = torch.zeros((128, 128), dtype=f8, device=dev)
                ksc = torch.zeros(128, dtype=torch.float32, device=dev)
                out = torch.empty((rows, 128), dtype=torch.float32, device=dev)
                _launch(kern, q, kc, ksc, w, one - 1, one, one - 1, out, splits_for(1, 128, n_sms))
    on = [b for b, c in zip(buckets, cases) if c not in missing]
    hk.log(f"{kernel} H={heads}: helion buckets {on}"
           + (f", other buckets {[b for b, c in zip(buckets, cases) if c in missing]}" if missing else ""))
    _WARM.add(key)


def warm_all(dev, cache, n_cols, bt_width, heads=32, g=GROUP_G, cand=None, v41=False):
    """Load-time warm-up of everything the lane can launch on a decode graph:
    the per-row and row-group bodies (and the fused-candidate body for
    cand = (ncand, bsz)) for this cache, logits width and block-table width
    (v41: the V4.1-keyed configs). Call it before the first capture (boot gate
    dsa_indexer_capture_replay does)."""
    fmt = {"fmt": FMT_V41} if v41 else {}
    _warm(DECODE, heads, dev, cache, n_cols, di._backend(), bt_width, force=True, **fmt)
    _warm(GROUP, heads, dev, cache, n_cols, "auto", bt_width, force=True, g=g, **fmt)
    if cand is not None:
        _warm(CANDK, heads, dev, cache, cand[0] * cand[1], "auto", bt_width, force=True,
              ncand=cand[0], bsz=cand[1])


def _check_q(q):
    if q.dim() != 3 or tuple(q.shape[1:]) not in ((32, 128), (64, 128)) or not q.is_contiguous():
        raise ValueError(f"dsa_indexer: q {tuple(q.shape)}: need contiguous [rows, 32|64, 128]")


def dsa_indexer_logits_cuda(out, q, weights, kv, k_scale, block_tables, lo, hi, n_cols, group,
                            slices, bt_div, splits, min_tiles, stream_ptr, *, v41=False):
    """Drop-in for ``_native.dsa_indexer_logits_cuda`` (same signature and
    contract, src/dsa_indexer_oxide.rs): paged when block_tables is given
    (any cache_views layout: the oxide fallback below only takes contiguous
    64-token pages, the lane routes other layouts here only with a config).
    group / slices / splits / min_tiles are the oxide plan — passed through
    on the oxide fallback, ignored by Helion (its config owns the schedule).
    v41: DeepSeek-V4.1 geometry -- its own (fmt-keyed) configs, and NO oxide
    fallback: a bucket without a config is an error here (the shim wrapper
    sends such calls to the stock path before getting here).
    Runs on torch's current stream (= stream_ptr in the lane)."""
    paged = block_tables is not None
    kernel = DECODE if paged else PREFILL
    _check_q(q)
    if paged and layout_reason(kv) is not None:
        raise ValueError(f"dsa_indexer: {layout_reason(kv)}")
    rows, heads = q.shape[0], q.shape[1]
    if not paged and (k_scale is None or kv.shape[0] != n_cols):
        raise ValueError(f"dsa_indexer: contiguous kv {tuple(kv.shape)} needs k_scale and n_cols == N")
    mode = di._backend()
    dev = q.device
    fmt = {"fmt": FMT_V41} if v41 else {}
    _warm(kernel, heads, dev, kv if paged else None, n_cols, mode,
          block_tables.shape[-1] if paged and block_tables.dim() > 1 else 2, **fmt)
    case = serving_case(kernel, heads, rows, **fmt)
    if hk.config(kernel, case) is None:  # auto: oxide won (or was never beaten) here
        if v41:
            raise RuntimeError(f"{hk.MARKER} {kernel}: no tuned config for {hk.case_key(case)} on "
                               f"{hk.platform(dev.index or 0)}; DeepSeek-V4.1 geometry has no oxide fallback "
                               "(oxide is retired for V4.1): ship the config or serve the stock path")
        return di._native(dev).dsa_indexer_logits_cuda(
            out, q, weights, kv, k_scale, block_tables, lo, hi, n_cols, group, slices, bt_div,
            splits, min_tiles, stream_ptr)
    if rows == 0 or n_cols == 0:
        return None
    kern = hk.compiled(_BODY[kernel], kernel, case)
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    if paged:
        bt = block_tables if bt_div == 1 else block_tables.repeat_interleave(bt_div, 0)
        kv_, ks_, page, prow, sp = cache_views(kv)
        _launch(kern, q, weights[:rows], kv_, ks_, bt, hi, out[:, :n_cols], page, prow, sp,
                splits_for(rows, n_cols, n_sms))
    else:
        if lo is None:
            lo = torch.zeros(rows, dtype=torch.int32, device=dev)
        _launch(kern, q, kv.view(torch.float8_e4m3fn), k_scale, weights[:rows], lo, hi,
                torch.zeros_like(lo), out, splits_for(-(-rows // 16), n_cols, n_sms))
    return None


def prefill_logits(q, kv, k_scale, w, lo, hi, out, compressed, v41=False):
    """vLLM's prefill contract on the Helion body (the lane checked has_config;
    v41: the V4.1-keyed configs):
    q e4m3 [M, H, 128], kv e4m3 [N, 128], k_scale f32 [N], w f32 [M, H], lo / hi
    int32 [M]; out contiguous f32 [M, W] — compressed (column n - lo) or
    absolute (W == N)."""
    _check_q(q)
    rows, heads = q.shape[0], q.shape[1]
    dev = q.device
    fmt = {"fmt": FMT_V41} if v41 else {}
    _warm(PREFILL, heads, dev, None, out.shape[1], di._backend(), **fmt)
    if rows == 0 or out.shape[1] == 0 or kv.shape[0] == 0:
        return out
    lo, hi = lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous()
    kern = hk.compiled(dsa_prefill_logits, PREFILL, serving_case(PREFILL, heads, rows, **fmt))
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    _launch(kern, q, kv, k_scale.reshape(-1).to(torch.float32).contiguous(), w[:rows], lo, hi,
            lo if compressed else torch.zeros_like(lo), out,
            splits_for(-(-rows // 16), kv.shape[0], n_sms))
    return out


def group_pad(q, w, cl, g):
    """Flat decode rows -> the group body's views: (q [R, G, H, 128], w
    [R, G, H], cl [R * G]) with R = ceil(rows / G); the last group is padded
    with dead rows (cl 0, q / w zero)."""
    rows, heads = q.shape[0], q.shape[1]
    reqs = -(-rows // g)
    pad = reqs * g - rows
    w, cl = w[:rows], cl.reshape(-1)
    if pad:
        q = torch.cat([q.view(torch.uint8), q.new_zeros((pad, heads, 128), dtype=torch.uint8)]) \
            .view(torch.float8_e4m3fn)
        w = torch.cat([w, w.new_zeros((pad, heads))])
        cl = torch.cat([cl, cl.new_zeros(pad)])
    return q.view(reqs, g, heads, 128), w.view(reqs, g, heads), cl


def group_logits(q, w, cache, bt, cl, out, n_cols, g, ctx=None, v41=False):
    """Row-group decode (the lane checked has_group and layout_reason): q e4m3
    [rows, H, 128] contiguous (rows = G x requests, flattened MTP layout), w f32
    [>= rows, H], cache uint8 [blocks, P, (1,) 132], bt int32 [rows, W] (a block
    table row per row), cl int32 [rows]; out f32 [rows, >= n_cols] (row written
    on [0, min(cl, n_cols))). rows need not be a multiple of g: the last group
    is padded with cl 0 rows. ctx: the tuned context bucket (None: the longest --
    a decode graph is captured once and replayed at every context, so serving
    never has a per-step context to bucket by). v41: the V4.1-keyed configs."""
    _check_q(q)
    rows, heads = q.shape[0], q.shape[1]
    dev = q.device
    fmt = FMT_V41 if v41 else None
    _warm(GROUP, heads, dev, cache, n_cols, "auto", bt.shape[-1], g=g, **({"fmt": fmt} if fmt else {}))
    if rows == 0 or n_cols == 0:
        return out
    kern = hk.compiled(dsa_group_logits, GROUP, group_serving_case(heads, rows, g, ctx, fmt))
    q4, w3, cl = group_pad(q, w, cl, g)   # dead rows read the last real block-table row
    kv, ks, page, prow, sp = cache_views(cache)
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    _launch(kern, q4, w3, kv, ks, bt, cl, out[:, :n_cols], page, prow, sp,
            splits_for(q4.shape[0], n_cols, n_sms))
    return out


def cand_logits(q, w, cache, bt, cl, cand, out):
    """Fused candidate logits (the lane checked has_config and layout_reason):
    q e4m3 [rows, H, 128], w f32 [rows, H], cache uint8 [blocks, P, (1,) 132],
    bt int32 [rows, *], cl int32 [rows], cand int32 [rows, C]; out f32
    [rows, C * B] (written on [0, cand_lens))."""
    _check_q(q)
    rows, heads = q.shape[0], q.shape[1]
    ncand, bsz = cand.shape[1], out.shape[1] // cand.shape[1]
    dev = q.device
    _warm(CANDK, heads, dev, cache, out.shape[1], "auto", bt.shape[-1], ncand=ncand, bsz=bsz)
    if rows == 0:
        return out
    kv, ks, page, prow, sp = cache_views(cache)
    kern = hk.compiled(dsa_cand_logits, CANDK, serving_case(CANDK, heads, rows, ncand=ncand, bsz=bsz))
    n_sms = di._sms(dev) if dev.type == "cuda" else 4
    _launch(kern, q, w[:rows], kv, ks, bt, cl, cand, out, page, prow, sp,
            splits_for(rows, out.shape[1], n_sms))
    return out


def cand_lens(cl, ncand, bsz):
    """Slots of each row the fused body writes = vLLM's valid candidate count x B."""
    return torch.clamp((cl + bsz - 1) // bsz, max=ncand) * bsz


def cand_remap(idx, cand, cl, bsz):
    """In place: top-k SLOT ids -> request-local positions (-1 where the slot
    is padding, a -1 block or past the context).
    ponytail: ~8 small torch ops per call; one Helion kernel if it shows."""
    s = idx.long()
    sc = s.clamp_min(0)
    blk = cand.gather(1, sc // bsz).long()
    pos = blk * bsz + sc % bsz
    keep = (s >= 0) & (blk >= 0) & (pos < cl.reshape(-1, 1))
    idx.copy_(torch.where(keep, pos, -1).to(idx.dtype))
    return idx


# ---------------------------------------------------------------------------
# lab ABI (inference-console api/autotune_kernels.py) + the oracle's spec
# ---------------------------------------------------------------------------
def decode_args(pr, n_cols, n_sms):
    """Body args of a lane problem (di.make_decode, flattened layout)."""
    rows, heads = pr["rows"], pr["heads"]
    kv, ks, page, prow, sp = cache_views(pr["cache"])
    bt = pr["bt"] if pr["bt"].shape[0] == rows else pr["bt"].repeat_interleave(pr["next_n"], 0)
    out = torch.full((rows, n_cols), float("-inf"), dtype=torch.float32, device=kv.device)
    return (pr["q"].reshape(rows, heads, 128), pr["w"][:rows], kv, ks, bt.contiguous(),
            pr["cl"].reshape(-1).to(torch.int32).contiguous(), out, page, prow, sp,
            splits_for(rows, n_cols, n_sms))


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
    return decode_args(pr, n_cols_for(case), n_sms)


def group_args(args, g):
    """decode_args -> the group body's args (q / w viewed [R, G, ...])."""
    q, w, kv, ks, bt, cl, out, page, prow, sp, splits = args
    q4, w3, cl = group_pad(q, w, cl, g)
    return (q4, w3, kv, ks, bt, cl, out, page, prow, sp,
            splits_for(q4.shape[0], out.shape[1], _dev_sms()[1]))


def group_make(case, seed):
    return group_args(decode_make(case, seed), case["group"])


def group_reference(q, w, kv, ks, bt, cl, out, page, prow, sp, splits):
    rows, heads = out.shape[0], q.shape[2]
    return decode_reference(q.reshape(-1, heads, 128)[:rows], w.reshape(-1, heads)[:rows], kv, ks, bt,
                            cl, out, page, prow, sp, splits)


def decode_reference(q, w, kv, ks, bt, cl, out, page, prow, sp, splits):
    """f64 logits + |term| magnitudes (di.ref_logits through the cache views),
    NaN outside each row's [0, min(cl, n_cols)) window."""
    rows, n_cols = out.shape
    val = torch.full((rows, n_cols), float("nan"), dtype=torch.float64, device=out.device)
    mag = torch.zeros((rows, n_cols), dtype=torch.float64, device=out.device)
    for r in range(rows):
        n = min(max(int(cl[r]), 0), n_cols)
        if n:
            k, s = gather_views(kv, ks, page, prow, sp, bt[r], n)
            val[r, :n], mag[r, :n] = di.ref_logits(q[r], w[r], k, s)
    return val, mag


def decode_make_baseline(case, seed):
    """The same problem in the oxide layout + its MTP row group (oracle style)."""
    pr, n_sms = _decode_problem(case, seed)
    out = torch.full((pr["rows"], n_cols_for(case)), float("-inf"), device=pr["q"].device)
    return (out, pr["q"], pr["cache"], pr["w"], pr["cl"], pr["bt"], case["rows"] // case["numtokens"])


SELECT_CASES = ((32768, 1), (32768, 8), (524288, 1), (524288, 4))   # (ctx, requests)
SELECT_STALE = 0.1   # fraction of the warm set that is stale (random window positions)


def select_case(heads: int, reqs: int, ctx: int, g: int = GROUP_G) -> dict:
    return group_case(heads, reqs, ctx, g, FMT_V41)    # V4.1-only experiment


def select_make(case, seed):
    """A group problem + the warm-start bound the serving pipeline would compute
    (dsa_select: previous top-k with SELECT_STALE random members + an even
    sample, k = 512), from the f64 spec's logits."""
    args = group_make(case, seed)
    q4, w3, kv, ks, bt, cl, out, page, prow, sp, splits = args
    val, _ = group_reference(*args)
    rows, n_cols = out.shape
    logits = torch.where(torch.isnan(val), torch.full_like(val, float("-inf")), val).float()
    g = torch.Generator(device=out.device).manual_seed(seed + 5)
    k = ds.K
    prev = torch.full((rows, k), -1, dtype=torch.int64, device=out.device)
    for r in range(rows):
        n = min(int(cl[r]), n_cols)
        if n:
            top = logits[r, :n].topk(min(k, n)).indices
            drop = int(top.numel() * SELECT_STALE)
            rest = torch.randint(0, n, (drop,), generator=g, device=out.device)
            prev[r, :top.numel()] = torch.cat([top[drop:], rest])[:top.numel()]
    thr = ds.warm_threshold(ds.warm_logits(logits, cl[:rows], ds.warm_positions(prev, cl[:rows])), k)
    dev = out.device
    return (q4, w3, kv, ks, bt, cl, thr, torch.full((rows, n_cols), float("nan"), device=dev),
            torch.full((rows, n_cols), -1, dtype=torch.int32, device=dev),
            torch.zeros(rows, dtype=torch.int32, device=dev), page, prow, sp, splits)


def select_reference(q, w, kv, ks, bt, cl, thr, comp_v, comp_i, cnt, page, prow, sp, splits):
    val, mag = group_reference(q, w, kv, ks, bt, cl, comp_v, page, prow, sp, splits)
    return val, mag, thr


def select_check(out, ref):
    """Survivor-set oracle: per row every window position whose spec logit is
    >= thr by more than the f32 noise (TOL x sum|terms|) is present, nothing
    below thr - noise is, no position twice, (idx, value) pairs agree with the
    spec, and nothing is written past the cursor."""
    comp_v, comp_i, cnt = out
    val, mag, thr = ref
    noise = di.TOL * mag
    for r in range(val.shape[0]):
        inside = ~torch.isnan(val[r])
        m = int(cnt[r])
        idx = comp_i[r, :m].long()
        if m > comp_v.shape[1] or (idx.numel() and (int(idx.min()) < 0 or not bool(inside[idx].all()))):
            return False, f"row {r}: survivor outside the window"
        if idx.unique().numel() != m:
            return False, f"row {r}: a position twice"
        t = float(thr[r])
        need = inside & (val[r] - noise[r] >= t)
        allow = inside & (val[r] + noise[r] >= t)
        have = torch.zeros_like(inside)
        have[idx] = True
        if bool((need & ~have).any()):
            return False, f"row {r}: a position above the bound is missing"
        if bool((have & ~allow).any()):
            return False, f"row {r}: a position below the bound survived"
        if m and float(((comp_v[r, :m].double() - val[r, idx]).abs() / (mag[r, idx] + 1e-30)).max()) > di.TOL:
            return False, f"row {r}: survivor logit off the spec"
        if not bool(torch.isnan(comp_v[r, m:]).all()):
            return False, f"row {r}: wrote past the cursor"
    return True, f"{int(cnt.sum())} survivors of {int((~torch.isnan(val)).sum())} window positions"


def select_make_baseline(case, seed):
    """The stock dense logits (vendored DeepGEMM) + a torch compaction (what the
    select replaces, minus the top-k)."""
    pr, n_sms = _decode_problem(case, seed)
    return (*decode_make_dg_from(pr, n_sms), select_make(case, seed)[6])


def select_baseline(q, cache, w, cl, bt, sched, thr):
    out = decode_baseline_dg(q, cache, w, cl, bt, sched)
    col = torch.arange(out.shape[1], device=out.device)
    keep = (out >= thr[:, None]) & (col[None, :] < cl.reshape(-1, 1))
    cnt = keep.sum(1, dtype=torch.int32)
    return torch.nonzero(keep), cnt


def _select_tune_baseline(q, w, kv, ks, bt, cl, thr, comp_v, comp_i, cnt, page, prow, sp, splits):
    """Dense stock logits + position-order compaction, in the select kernel's output form."""
    rows = comp_v.shape[0]
    dense = _stock_dense(q, w, kv, bt, cl, comp_v.shape[1], page, sp)
    keep = dense >= thr[:, None]
    n = keep.sum(1, dtype=torch.int32)
    pos = torch.cumsum(keep.to(torch.int32), 1) - 1
    rr = torch.arange(rows, device=comp_v.device)[:, None].expand_as(keep)
    out_v, out_i = torch.full_like(comp_v, float("nan")), torch.full_like(comp_i, -1)
    out_v[rr[keep], pos[keep]] = dense[keep]
    out_i[rr[keep], pos[keep]] = torch.arange(dense.shape[1], device=dense.device, dtype=torch.int32) \
        .expand_as(keep)[keep]
    return out_v, out_i, n


def _select_tune_check(actual, expected):
    """Autotuner accuracy check: survivors are packed in a race order, so compare
    per row the (position -> logit) maps and the counts (rtol 1e-3)."""
    (av, ai, an), (ev, ei, en) = actual, expected
    assert torch.equal(an, en), f"survivor counts {an.tolist()} vs {en.tolist()}"
    for r in range(av.shape[0]):
        m = int(en[r])
        a, e = ai[r, :m].long().sort(), ei[r, :m].long().sort()
        assert torch.equal(a.values, e.values), f"row {r}: survivor positions differ"
        torch.testing.assert_close(av[r, :m][a.indices], ev[r, :m][e.indices], rtol=1e-3, atol=1e-3)


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


def baseline_make(case, seed):
    """K4 baseline problem of a DECODE / GROUP case: V4.1 (fmt) -> the stock
    vendored-DeepGEMM call, GLM-5.3 -> oxide's page-dedup row group. The config
    pull then ships a V4.1 config only where Helion beat the STOCK path."""
    if case.get("fmt") == FMT_V41:
        return ("stock", *decode_make_dg(case, seed))
    return ("oxide", *decode_make_baseline(case, seed))


def baseline_run(kind, *a):
    return decode_baseline_dg(*a) if kind == "stock" else decode_baseline(*a)


def decode_make_dg(case, seed):
    """The problem in vLLM's DeepGEMM paged-MQA call (flattened next_n = 1)."""
    return decode_make_dg_from(*_decode_problem(case, seed))


def vendor_baseline_guard():
    """K4 integrity: a V4.1 baseline is the stock VENDORED DeepGEMM call. With the suffix shim's fallback
    engaged (SUFFIX_SM120=1) vLLM's wrappers still reach the vendor, but only while the shim's trio dispatches
    there (sm120_fallback.vendor_paged_ok) and the lane's own wrappers are not installed; on a pod where they
    would hit the Triton stand-in (GLM-era vLLM, SUFFIX_DSA_INDEXER_STOCK=shim) or our lane, the measured
    'baseline' would not be what boot5 ran and a won config would not beat A1: refuse."""
    try:
        import deep_gemm
    except ImportError:         # no shim on the path: vLLM's wrappers import the vendored module itself
        return
    if not (getattr(deep_gemm, "__suffix_shim__", False) and getattr(deep_gemm, "_use_fallback", False)):
        return
    why = "the lane's own wrappers are installed (SUFFIX_SM120_DSA_INDEXER=1)" if getattr(
        deep_gemm, "_dsa_indexer", False) else None
    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata"):
        applies = getattr(getattr(deep_gemm, name, None), "vendor_applies", None)
        if why is None and (applies is None or not applies(max_seqlen_k=1, clean_logits=False)):
            why = f"deep_gemm.{name} resolves to the shim's Triton stand-in"
    if why:
        raise RuntimeError(f"[suffix dsa-indexer] a V4.1 K4 baseline must be the vendored DeepGEMM, but {why}: "
                           "run the lab with the shim inert (no SUFFIX_SM120) or on a nightly pod whose vendor "
                           "runs the SM120 paged MQA (SUFFIX_DSA_INDEXER_STOCK unset)")


def decode_make_dg_from(pr, n_sms):
    vendor_baseline_guard()
    from vllm.utils import deep_gemm as vdg
    rows = pr["rows"]
    cl = pr["cl"].reshape(rows, 1).to(torch.int32).contiguous()
    bt = pr["bt"].contiguous()
    sched = vdg.get_paged_mqa_logits_metadata(cl, PAGE, n_sms)
    return (pr["q"].reshape(rows, 1, pr["heads"], 128), pr["cache"], pr["w"][:rows].contiguous(),
            cl, bt, sched)


def decode_baseline_dg(q, cache, w, cl, bt, sched):
    """vLLM's production SM120 path: the vendored DeepGEMM (vllm.utils.deep_gemm,
    whatever wrapper name this vLLM has); never our shim's fallback."""
    vendor_baseline_guard()
    from vllm.utils import deep_gemm as vdg
    import inspect
    fn = getattr(vdg, "fp8_fp4_paged_mqa_logits", None)
    if fn is None:
        return vdg.fp8_paged_mqa_logits(q, cache, w, cl, bt, sched,
                                        max(di.MAX_MODEL_LEN, bt.shape[1] * PAGE), False)
    kw = {"clean_logits": False} if "clean_logits" in inspect.signature(fn).parameters else {}
    return fn((q, None), cache, w, cl, bt, sched, max(di.MAX_MODEL_LEN, bt.shape[1] * PAGE), **kw)


def prefill_problem(heads, m, n, dev, seed=0, compressed=True):
    """The oracle's prefill: two requests packed in K (request 0 = K [0, n/2)
    with the first m/2 rows, request 1 = the rest); row i of a request with L
    keys and Mq rows sees [a, a + L - Mq + 1 + i). compressed: vLLM nightly's
    [M, max window] output, column n - lo (off = lo); else absolute [M, N]."""
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
    width = int((hi - lo).max()) if compressed else n
    out = torch.full((m, width), float("-inf"), dtype=torch.float32, device=dev)
    return q, kv, ksc, w, lo, hi, (lo.clone() if compressed else torch.zeros_like(lo)), out


def prefill_make(case, seed):
    dev, n_sms = _dev_sms()
    args = prefill_problem(case["heads"], case["m"], case["n"], dev, seed)
    return (*args, splits_for(-(-case["m"] // 16), case["n"], n_sms))


def prefill_make_baseline(case, seed):
    """The oxide kernel's (absolute-column) layout of the same problem."""
    dev, n_sms = _dev_sms()
    q, kv, ksc, w, lo, hi, _, out = prefill_problem(case["heads"], case["m"], case["n"], dev, seed,
                                                    compressed=False)
    return q, kv, ksc, w, lo, hi, out, 0


def prefill_reference(q, kv, ks, w, lo, hi, off, out, splits):
    """f64 logits + magnitudes: K position n of row r's window [max(lo, 0),
    min(hi, N)) at column n - off[r] (inside [0, W)), NaN elsewhere."""
    m, width = out.shape
    n = kv.shape[0]
    val = torch.full((m, width), float("nan"), dtype=torch.float64, device=out.device)
    mag = torch.zeros((m, width), dtype=torch.float64, device=out.device)
    for r in range(m):
        a, b = max(int(lo[r]), 0, int(off[r])), min(int(hi[r]), n, int(off[r]) + width)
        if a < b:
            o = int(off[r])
            val[r, a - o:b - o], mag[r, a - o:b - o] = di.ref_logits(q[r], w[r], kv[a:b], ks[a:b])
    return val, mag


def prefill_make_baseline_any(case, seed):
    """V4.1 (fmt): the stock vendored-DeepGEMM compressed prefill call; GLM: oxide."""
    if case.get("fmt") == FMT_V41:
        dev, _ = _dev_sms()
        q, kv, ksc, w, lo, hi, _, _ = prefill_problem(case["heads"], case["m"], case["n"], dev, seed,
                                                      compressed=True)
        return ("stock", q, kv, ksc, w, lo, hi, case["n"])
    return ("oxide", *prefill_make_baseline(case, seed))


def prefill_baseline_any(kind, *a):
    if kind == "stock":
        vendor_baseline_guard()
        from vllm.utils import deep_gemm as vdg
        q, kv, ksc, w, lo, hi, n = a
        return vdg.fp8_fp4_mqa_logits((q, None), (kv, ksc), w, lo, hi, n)
    return prefill_baseline(*a)


def prefill_baseline(q, kv, ks, w, lo, hi, out, splits):
    return _oxide(out, q, w, kv, ks, None, lo, hi, di.PREFILL_GROUP)


def cand_problem(pr, ncand, bsz, seed=0):
    """vLLM's candidate list per row: min(C, ceil(cl / B)) distinct valid block
    ids (the newest always in), shuffled, then -1 padding — the sorted top-k
    order select_candidate_blocks leaves (-inf scores last)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cl = pr["cl"].reshape(-1).cpu()
    cand = torch.full((cl.shape[0], ncand), -1, dtype=torch.int32)
    for r, n in enumerate(cl.tolist()):
        nb = -(-n // bsz)
        if not nb:
            continue
        pick = torch.randperm(nb - 1, generator=g)[:min(ncand, nb) - 1] if nb > 1 \
            else torch.empty(0, dtype=torch.long)
        ids = torch.cat([torch.tensor([nb - 1]), pick])
        cand[r, :ids.numel()] = ids[torch.randperm(ids.numel(), generator=g)].to(torch.int32)
    return cand.to(pr["cl"].device)


def cand_args(pr, ncand, bsz, n_sms, seed=0):
    q, w, kv, ks, bt, cl, _, page, prow, sp, _ = decode_args(pr, 1, n_sms)
    out = torch.full((pr["rows"], ncand * bsz), float("-inf"), dtype=torch.float32, device=kv.device)
    return (q, w, kv, ks, bt, cl, cand_problem(pr, ncand, bsz, seed), out, page, prow, sp,
            splits_for(pr["rows"], ncand * bsz, n_sms))


def _cand_problem(case, seed):
    dev, n_sms = _dev_sms()
    rows, reqs = case["rows"], case["numtokens"]
    pr = di.make_decode(dev, case["heads"], reqs, rows // reqs, case["ctx"], True,
                        di.MAX_MODEL_LEN, seed=2000 + seed)
    return pr, n_sms


def cand_make(case, seed):
    pr, n_sms = _cand_problem(case, seed)
    return cand_args(pr, case["cand"], case["cbs"], n_sms, seed)


def cand_reference(q, w, kv, ks, bt, cl, cand, out, page, prow, sp, splits):
    """f64 slot logits (-inf on dead slots) + magnitudes; NaN past cand_lens."""
    rows, width = out.shape
    bsz = width // cand.shape[1]
    val = torch.full((rows, width), float("nan"), dtype=torch.float64, device=out.device)
    mag = torch.zeros((rows, width), dtype=torch.float64, device=out.device)
    lens = cand_lens(cl.long(), cand.shape[1], bsz)
    for r in range(rows):
        n, ln = int(cl[r]), int(lens[r])
        if not ln:
            continue
        j = torch.arange(ln, device=out.device)
        blk = cand[r, j // bsz].long()
        pos = blk * bsz + j % bsz
        ok = (blk >= 0) & (pos < n)
        val[r, :ln] = float("-inf")
        if ok.any():
            k, s = gather_views(kv, ks, page, prow, sp, bt[r], n)
            v, a = di.ref_logits(q[r], w[r], k[pos[ok]], s[pos[ok]])
            val[r, j[ok]], mag[r, j[ok]] = v, a
    return val, mag


def cand_make_baseline(case, seed):
    """What the fused body replaces, on the same problem: the stock dense logits
    (vendored DeepGEMM) + vLLM's apply_candidate_mask over the full logits width
    (when this vLLM has it; else the dense pass alone -- a baseline that only
    gets cheaper)."""
    pr, n_sms = _cand_problem(case, seed)
    cand = cand_problem(pr, case["cand"], case["cbs"], seed)
    return (*decode_make_dg_from(pr, n_sms), cand, case["cbs"])


def cand_baseline(q, cache, w, cl, bt, sched, cand, bsz):
    out = decode_baseline_dg(q, cache, w, cl, bt, sched)
    try:
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    except ImportError:
        return out
    apply_candidate_mask(out, None, cl.reshape(-1), cand, bsz)
    return out


# Helion's autotuner checks every candidate against a baseline run over the
# WHOLE output; the windows are the only defined part (the kernel never writes
# elsewhere). Lab-only settings (serving never autotunes): the accuracy baseline
# is the shim's Triton stand-in of the DeepGEMM contract (sm120_fallback.triton_*:
# exact on every page layout, no oxide, never the vendor) on the body's own args, and the
# comparison is restricted to the columns the baseline wrote (finite), like the
# consumer. The lab's check() still proves the winner wrote nothing outside the
# windows.
def _cache_of(kv, page, sp):
    """The uint8 [blocks, page, 132] split-page cache behind cache_views' kv view
    (contiguous or with a packed page stride)."""
    s = sp * 4
    flat = kv.view(torch.uint8).reshape(-1)
    return flat.as_strided(((flat.numel() - page * 132) // s + 1, page, 132), (s, 132, 1))


def _stock_dense(q, w, kv, bt, cl, n_cols, page, sp):
    """Dense logits [rows, n_cols] (-inf outside every row's window) from the shim's
    Triton stand-in of the DeepGEMM contract (sm120_fallback.triton_paged_mqa_logits,
    never the vendored kernel): the accuracy baseline of every decode tune case, GLM
    and V4.1 alike -- exact on any page layout, and no oxide."""
    rows = bt.shape[0]
    heads = q.shape[-2]
    cl = cl.reshape(-1)[:rows]            # (group bodies carry padding rows past `rows`)
    res = di._fallback_mod().triton_paged_mqa_logits(
        (q.reshape(-1, heads, 128)[:rows].reshape(rows, 1, heads, 128), None),
        _cache_of(kv, page, sp).unsqueeze(2), w.reshape(-1, heads)[:rows], cl.reshape(rows, 1), bt, None, n_cols)
    col = torch.arange(n_cols, device=res.device)
    return torch.where(col[None, :] < torch.clamp(cl, 0, n_cols)[:, None], res, float("-inf"))


def _decode_tune_baseline(q, w, kv, ks, bt, cl, out, page, prow, sp, splits):
    return _stock_dense(q, w, kv, bt, cl, out.shape[1], page, sp)


def _group_tune_baseline(q, w, kv, ks, bt, cl, out, page, prow, sp, splits):
    return _stock_dense(q, w, kv, bt, cl, out.shape[1], page, sp)


def _prefill_tune_baseline(q, kv, ks, w, lo, hi, off, out, splits):
    """The Triton fallback's compressed logits, cleaned past each window."""
    res = di._fallback_mod().triton_mqa_logits((q, None), (kv, ks), w, lo, hi, out.shape[1])
    col = torch.arange(out.shape[1], device=out.device)
    return torch.where(col[None, :] < (hi - lo)[:, None], res, float("-inf"))


def _cand_tune_baseline(q, w, kv, ks, bt, cl, cand, out, page, prow, sp, splits):
    dense = _stock_dense(q, w, kv, bt, cl, di.MAX_MODEL_LEN, page, sp)
    bsz = out.shape[1] // cand.shape[1]
    j = torch.arange(out.shape[1], device=q.device)
    blk = cand[:, j // bsz].long()
    pos = blk * bsz + j % bsz
    ok = (blk >= 0) & (pos < cl.reshape(-1, 1)) \
        & (j[None, :] < cand_lens(cl.long(), cand.shape[1], bsz)[:, None])
    return torch.where(ok, dense.gather(1, pos.clamp_min(0)), float("-inf"))


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
    compared (|out - f64| <= TOL * sum|terms|, finite; -inf exactly where the
    reference says -inf: dead candidate slots); everything outside must still
    hold make()'s -inf (nothing written there)."""
    val, mag = ref
    inside = ~torch.isnan(val)
    if not bool(torch.isneginf(out[~inside]).all()):
        return False, "wrote outside the [lo, hi) windows"
    dead = torch.isneginf(val)
    if not bool(torch.isneginf(out[dead]).all()):
        return False, "a dead candidate slot is not -inf"
    live = inside & ~dead
    o = out[live].double()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite logits inside the windows"
    err = float(((o - val[live]).abs() / (mag[live] + 1e-30)).max()) if o.numel() else 0.0
    return err <= di.TOL, f"max err/A {err:.2e} (tol {di.TOL:.0e})"


def _cases(kernel, archs=None, rows=None):
    def cases(model):
        arch = (model or {}).get("architecture")
        heads = INDEX_HEADS.get(arch)
        if heads is None or (archs is not None and arch not in archs):
            return []
        make_case, buckets = _CASE[kernel]
        kw = {}
        if kernel == CANDK:
            kw = dict(zip(("ncand", "bsz"), CAND[arch]))
        elif arch in CAND:   # DeepSeek-V4.1: its own keys, baselines are the stock path
            kw = {"fmt": FMT_V41}
        return [make_case(heads, b, **kw) for b in buckets if rows is None or b in rows]
    return cases


def _group_cases(only=None, archs=None):
    def cases(model):
        arch = (model or {}).get("architecture")
        heads = INDEX_HEADS.get(arch)
        if heads is None or (archs is not None and arch not in archs):
            return []
        fmt = FMT_V41 if arch in CAND else None
        return [c for _, c in all_cases(GROUP, heads, fmt=fmt) if only is None or (c["ctx"], c["numtokens"]) in only]
    return cases


_SETTINGS = {"autotune_baseline_accuracy_check_fn": _tune_check, **cap.SEARCH_SETTINGS}
KERNELS = [
    {"name": DECODE, "body": dsa_decode_logits, "cases": _cases(DECODE),
     "make": decode_make, "reference": decode_reference, "check": check,
     "make_baseline": baseline_make, "baseline": baseline_run,
     "baseline_name": "oxide dsa_indexer (GLM-5.3) | vLLM DeepGEMM paged MQA (V4.1)",
     "replaces": [r"dsa_logits"],
     "settings": {"autotune_baseline_fn": _decode_tune_baseline, **_SETTINGS}},
    # MTP rows of a request scored in one K sweep (V4.1 DSpark: G = 6), per
    # request count and context bucket up to 524288; baseline = the stock
    # DeepGEMM call on V4.1 (oxide's page-dedup group on GLM-5.3)
    {"name": GROUP, "body": dsa_group_logits, "cases": _group_cases(),
     "make": group_make, "reference": group_reference, "check": check,
     "make_baseline": baseline_make, "baseline": baseline_run,
     "baseline_name": "oxide dsa_indexer MTP row group (GLM-5.3) | vLLM DeepGEMM paged MQA (V4.1)", "replaces": [],
     "settings": {"autotune_baseline_fn": _group_tune_baseline, **_SETTINGS}},
    # fused top-k without dense logits: the group scan writes only the survivors of a
    # warm-start bound (kernels/dsa_select.py); baseline = the stock dense logits + a
    # torch compaction. Lab only: not wired into serving yet (no config is read).
    {"name": SELECT, "body": dsa_group_select, "cases": _group_cases(set(SELECT_CASES), CAND),
     "make": select_make, "reference": select_reference, "check": select_check,
     "make_baseline": select_make_baseline, "baseline": select_baseline,
     "baseline_name": "vLLM DeepGEMM dense logits + torch compaction", "replaces": [],
     "settings": {"autotune_baseline_fn": _select_tune_baseline,
                  "autotune_baseline_accuracy_check_fn": _select_tune_check, **cap.SEARCH_SETTINGS}},
    # the oxide entry is ONE kernel for decode and prefill: the decode-dominated
    # profile op is credited to the decode port only (no double count)
    {"name": PREFILL, "body": dsa_prefill_logits, "cases": _cases(PREFILL),
     "make": prefill_make, "reference": prefill_reference, "check": check,
     "make_baseline": prefill_make_baseline_any, "baseline": prefill_baseline_any,
     "baseline_name": "oxide dsa_indexer absolute columns (GLM-5.3) | vLLM DeepGEMM compressed (V4.1)",
     "replaces": [],
     "settings": {"autotune_baseline_fn": _prefill_tune_baseline, **_SETTINGS}},
    # DeepSeek-V4.1 consumer indexers: fused candidate scoring vs dense + mask
    {"name": CANDK, "body": dsa_cand_logits, "cases": _cases(CANDK, CAND),
     "make": cand_make, "reference": cand_reference, "check": check,
     "make_baseline": cand_make_baseline, "baseline": cand_baseline,
     "baseline_name": "vLLM DeepGEMM dense logits + apply_candidate_mask",
     "replaces": [r"_mask_candidates_kernel", r"_candidate_flags_kernel"],
     "settings": {"autotune_baseline_fn": _cand_tune_baseline, **_SETTINGS}},
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
