# SPDX-License-Identifier: Apache-2.0
"""row_topk(y, k): the k largest entries of each bf16 row in torch.topk's order (ATen's
bf16 radix key, ties to the lowest index) for mxfp4_lm_head's exact rescore, in two
launches: grid (rows, 4096-wide chunks) picks each chunk's k by a 16-step bisection on
the key, then one program per row picks the k of the chunks' candidates the same way.
Replaces y.topk(64) (ATen's multi-kernel radix select): MI350P gate6 graphed us for the
whole MXFP4 head + rescore, M 1 / 5 / 32 / 64: 191.7 / 247.9 / 356.3 / 523.1 with y.topk
-> 155.6 / 174.3 / 299.4 / 422.0, logits bitwise. At M >= 96 the BF16 head stays faster
(M 160: 537 vs 710 us), so SUFFIX_MXFP4_LMHEAD_MAX_M stays at 64.

Tried and dropped (gate6): the vLLM Qrita top-k / top-p sampler with rows split across
programs (statistics / mask passes in parallel, stock's pivot search verbatim): same
masks, but B 1 / 5 / 160 = 171 / 253 / 466 us stock -> 231 / 234 / 349 us; the search
loop is the floor.

    python -m suffix_hybrid.kernels.lm_sample_rocm   # boot gate lm_sample_lmhead
"""
import sys

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix lm-sample]"
ROW_TOPK_CHUNK = 4096


# ---------------------------------------------------------------------------
@triton.jit
def _topk_key(x):
    """ATen TopKTypeConfig<BFloat16>::convert: bits ^ 0xffff with the sign bit set, else
    ^ 0x8000; NaN -> 0xffff. Ascending key = torch.topk's order."""
    b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key = tl.where(b >= 0x8000, b ^ 0xFFFF, b ^ 0x8000)
    return tl.where(x == x, key, 0xFFFF)


@triton.jit
def _kth_key(key, K: tl.constexpr):
    """Largest t with count(key >= t) >= K (the K-th largest key; 0 when fewer than K
    keys are >= 0, i.e. take every valid one). Invalid entries carry -1."""
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), 65536, tl.int32)
    for _ in tl.static_range(16):
        mid = (lo + hi) // 2
        enough = tl.sum((key >= mid).to(tl.int32)) >= K
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid)
    return lo


@triton.jit
def _take_k(key, K: tl.constexpr):
    """Mask of the K largest keys, ties at the K-th key to the lowest positions."""
    t = _kth_key(key, K)
    above = key > t
    tie = key == t
    return above | (tie & (tl.cumsum(tie.to(tl.int32), axis=0)
                           <= K - tl.sum(above.to(tl.int32))))


@triton.jit(do_not_specialize=["N"])
def _row_topk_chunk_kernel(Y, CKEY, CIDX, N, stride_y, K: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    ok = offs < N
    x = tl.load(Y + row.to(tl.int64) * stride_y + offs, mask=ok, other=0.0)
    key = tl.where(ok, _topk_key(x), -1)
    take = _take_k(key, K)
    pos = tl.cumsum(take.to(tl.int32), axis=0) - 1
    base = (row * tl.num_programs(1) + chunk) * K
    tl.store(CKEY + base + pos, key, mask=take)
    tl.store(CIDX + base + pos, offs, mask=take)
    slot = tl.arange(0, K)  # a chunk with fewer than K entries: its free slots never win
    tl.store(CKEY + base + slot, -1, mask=slot >= tl.sum(take.to(tl.int32)))


@triton.jit
def _row_topk_merge_kernel(CKEY, CIDX, OUT, CANDS, K: tl.constexpr, C_PAD: tl.constexpr):
    # Candidates sit chunk by chunk, each chunk's in index order: position order is
    # index order, so _take_k's tie rule is torch.topk's.
    row = tl.program_id(0)
    j = tl.arange(0, C_PAD)
    ok = j < CANDS
    key = tl.load(CKEY + row * CANDS + j, mask=ok, other=-1)
    idx = tl.load(CIDX + row * CANDS + j, mask=ok, other=0)
    take = _take_k(key, K)
    pos = tl.cumsum(take.to(tl.int32), axis=0) - 1
    tl.store(OUT + row * K + pos, idx.to(tl.int64), mask=take)


def row_topk(y: torch.Tensor, k: int) -> torch.Tensor:
    """Indices [M, k] (int64, ascending) of the k largest entries of each row of bf16 y,
    the same set as y.topk(k).indices. HIP-graph safe; k a power of two <= row length."""
    m, n = y.shape
    if (y.dtype != torch.bfloat16 or y.stride(1) != 1 or k & (k - 1) or not 0 < k <= n):
        raise ValueError(f"{MARK} row_topk: unsupported y {tuple(y.shape)} {y.dtype} k={k}")
    chunks = triton.cdiv(n, ROW_TOPK_CHUNK)
    ckey = torch.empty((m, chunks, k), dtype=torch.int32, device=y.device)
    cidx = torch.empty_like(ckey)
    out = torch.empty((m, k), dtype=torch.int64, device=y.device)
    if m:
        _row_topk_chunk_kernel[(m, chunks)](y, ckey, cidx, n, y.stride(0), K=k,
                                            CHUNK=ROW_TOPK_CHUNK, num_warps=4)
        _row_topk_merge_kernel[(m,)](ckey, cidx, out, chunks * k, K=k,
                                     C_PAD=triton.next_power_of_2(chunks * k), num_warps=4)
    return out


def _graph_us(fn, iters: int = 20):
    """us per call of fn() replayed from one HIP graph, and the graphed output."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters, out


def _same_set(a, b) -> bool:
    return torch.equal(a.sort(dim=1).values, b.sort(dim=1).values)


def _oracle_lmhead() -> int:
    """row_topk == torch.topk's index set on adversarial bf16 rows; then the MXFP4 head
    with row_topk vs with y.topk (the shipped path), bitwise, M 1..256, greedy vs the
    bf16 head, graph replay == eager, graphed us/call."""
    from suffix_hybrid.kernels import mxfp4_lm_head as lmh

    prepare, run, _ = lmh._ops()
    n, kdim = lmh.SHAPE
    dev, r = "cuda", lmh.RESCORE
    gen = torch.Generator(device=dev).manual_seed(0)
    failed = False

    def rows(kind, m):
        if kind == "ties":  # 17 distinct values: thousands of ties at the 64th
            return (torch.randint(-8, 9, (m, n), generator=gen, device=dev) / 4).bfloat16()
        y = torch.randn(m, n, generator=gen, device=dev)
        if kind == "boundary-ties":  # the 64th value repeated in every chunk
            kth = y.topk(r, dim=-1).values[:, -1:]
            pos = torch.randint(0, n, (m, 200), generator=gen, device=dev)
            y.scatter_(1, pos, kth.expand(-1, 200))
        elif kind == "negative":
            y = -y.abs() - 1
        elif kind == "inf-nan":
            y[:, 7::4099] = float("inf")
            y[:, 11::5003] = float("-inf")
            y[:, 13::7001] = float("nan")
        return y.bfloat16()

    # No signed-zero rows: ATen keys +0 above -0, row_topk ties them (gate6: one set
    # MISMATCH at M=1); real logits never put +-0 at the 64th place.
    for kind in ("randn", "ties", "boundary-ties", "negative", "inf-nan"):
        bad = []
        for m in (1, 5, 64, 160, 256):
            y = rows(kind, m)
            if not _same_set(row_topk(y, r), y.topk(r, dim=-1).indices):
                bad.append(m)
        failed |= bool(bad)
        print(f"{MARK} lmhead row_topk({r}) vs torch.topk set, {kind}: "
              f"{'MATCH' if not bad else f'MISMATCH at M={bad}'}", flush=True)

    w = torch.randn(n, kdim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    st = prepare(w, 256)

    def ref_topk(y, k):
        return y.topk(k, dim=-1).indices

    for m in (1, 2, 4, 5, 8, 16, 32, 40, 64, 96, 128, 160, 192, 256):
        x = torch.randn(m, kdim, generator=gen, device=dev, dtype=torch.bfloat16)
        new, old = run(st, x), run(st, x, topk=ref_topk)
        ok = torch.equal(new, old) and lmh.greedy_ok(torch.nn.functional.linear(x, w), new)
        out = torch.empty_like(new)
        new_us, graphed = _graph_us(lambda: run(st, x, out))
        ok &= torch.equal(graphed, new)
        old_us, _ = _graph_us(lambda: run(st, x, out, topk=ref_topk))
        bf16_us, _ = _graph_us(lambda: torch.nn.functional.linear(x, w))
        torch_us, _ = _graph_us(lambda: new.topk(r, dim=-1).indices)
        ours_us, _ = _graph_us(lambda: row_topk(new, r))
        failed |= not ok
        print(f"{MARK} lmhead M={m}: {'MATCH' if ok else 'MISMATCH'} (logits bitwise vs the "
              f"y.topk path, greedy == bf16, graph == eager) | graphed us: bf16 {bf16_us:.1f} "
              f"| mxfp4 + y.topk {old_us:.1f} -> mxfp4 + row_topk {new_us:.1f} | top-{r} alone: "
              f"torch {torch_us:.1f} -> row_topk {ours_us:.1f}", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    return _oracle_lmhead()


if __name__ == "__main__":
    raise SystemExit(main())
