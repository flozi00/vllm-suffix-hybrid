# SPDX-License-Identifier: Apache-2.0
"""On-GPU oracle for the HiSparse residency split (fail closed at boot).

    python -m hisparse_mtp_patch.split_oracle   # exit 0 PASS, 1 FAIL, 3 not applicable

Real HiSparseRuntime objects (vLLM's own hisparse_resolve_residency /
hisparse_gather_compact ops, our split wrappers installed) resolve several
decode steps of synthetic MTP rows (2048 positions each, ~70 % shared between a
request's rows, masked -1 entries, LRU churn across steps). After every step each
row's attention indices must read EXACTLY the host rows of its positions
(-1 stays -1):
  A  k=3 (4 rows, fits in one pass): upstream single pass vs a forced 2-pass
     split on identical inputs, both byte-exact;
  B  k=5 (6 rows, needs 132,388 B smem on stock): the split chosen from this
     GPU's shared memory, byte-exact;
  C  B's step captured in one CUDA graph, replayed with new top-k, byte-exact.
Not applicable (exit 3): no CUDA / not SM120 / vLLM without the union resolver
(0.30.0)."""
import sys
from pathlib import Path

TAG = "[suffix sm120-hisparse-mtp] split oracle"
TOP_K, BS, ROW = 2048, 64, 64  # top-k, source/hot block size, row bytes
CONTEXT = 6000


def _runtime(R, q, reqs, dev, hot=None, union=None, swap_rows=None):
    import torch

    hot = hot or (q + 1) * TOP_K
    cfg = R.ResolvedHiSparseConfig(top_k=TOP_K, device_buffer_size=hot,
                                   max_union_rows=union or q * TOP_K)
    rt = R.HiSparseRuntime(cfg, reqs, ROW, torch.uint8, dev,
                           max_swap_rows=swap_rows or reqs * q + 1)
    hot_blocks = hot // BS
    raw = torch.zeros(reqs * hot * ROW, dtype=torch.uint8, device=dev)
    table = torch.arange(reqs * hot_blocks, dtype=torch.int32, device=dev).view(reqs, hot_blocks)
    rt.bind_hot_cache(raw, byte_offset=0, block_stride=BS * ROW, num_blocks=reqs * hot_blocks,
                      block_size=BS, block_table=table)
    rt.request_state_indices = torch.arange(reqs, dtype=torch.int32, device=dev)
    return rt, raw


def _host(reqs, dev):
    """Pinned host pool whose row g holds g; source block table: request j's
    context block b -> host block 1 + j * blocks + b (block 0 = null)."""
    import torch

    blocks = -(-CONTEXT // BS)
    rows = (1 + reqs * blocks) * BS
    host = torch.empty((rows, ROW), dtype=torch.uint8, pin_memory=True)
    host.view(torch.int32).copy_(torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, ROW // 4))
    bt = (1 + torch.arange(reqs * blocks, dtype=torch.int32).view(reqs, blocks)).to(dev)
    return host, bt


def _topk(gen, q, reqs, dev):
    """[reqs * q, TOP_K] request-relative positions: per request a base set, each
    row keeping ~70 % of it; unique within a row; ~1 % masked (-1)."""
    import torch

    out = []
    for _ in range(reqs):
        base = torch.randperm(CONTEXT, generator=gen)[:TOP_K]
        for _r in range(q):
            keep = base[torch.rand(TOP_K, generator=gen) < 0.7]
            rest = torch.randperm(CONTEXT, generator=gen)
            rest = rest[~torch.isin(rest, keep)][: TOP_K - keep.numel()]
            row = torch.cat([keep, rest])[torch.randperm(TOP_K, generator=gen)]
            row[torch.rand(TOP_K, generator=gen) < 0.01] = -1
            out.append(row)
    return torch.stack(out).to(torch.int32).to(dev)


def _step(rt, ids, bt, topk):
    rt.begin_forward()
    rows = rt._step_rows(topk.shape[0])
    rt._resolve_residency(resident=None, req_id_per_token=ids, block_table=bt,
                          topk_indices=topk, block_size=BS, shared_rows=rows)
    rt._swap_rows(rows)
    return rt.index_group.shared_topk.physical_topk_indices[rows]


def _check(raw, host, bt, ids, topk, phys) -> bool:
    """Every (row, k): the hot row read at its attention index == host row of its position."""
    import torch

    hot_rows = raw.view(-1, ROW).view(torch.int32)[:, 0]
    pos = topk.long()
    blk = pos.clamp(min=0) // BS
    g = bt.long()[ids.long()[:, None], blk] * BS + pos.clamp(min=0) % BS
    want = torch.where(pos >= 0, g, torch.full_like(g, -1))
    got = torch.where(phys >= 0, hot_rows[phys.clamp(min=0).long()].long(), torch.full_like(g, -1))
    return bool(torch.equal(got, want))


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            print(f"{TAG}: not applicable (no SM120 GPU)", flush=True)
            return 3
        from vllm.platforms import current_platform

        assert current_platform.is_cuda()  # loads vllm._C_stable_libtorch: torch.ops._C_cache_ops
        from vllm.v1.hisparse import runtime as R

        import hisparse_mtp_patch as HS

        if not HS.check_split_source(Path(R.__file__).read_text()):
            print(f"{TAG}: not applicable (no union resolver in this vLLM)", flush=True)
            return 3
        HS.install_split(R)
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
        gen = torch.Generator().manual_seed(1234)
        reqs, ok, lines = 3, True, []

        # A: k=3, upstream single pass vs forced 2-pass split, identical inputs.
        host, bt = _host(reqs, dev)
        (up, up_raw), (sp, sp_raw) = _runtime(R, 4, reqs, dev), _runtime(R, 4, reqs, dev)
        for rt in (up, sp):
            rt.bind_source_cache(host)
        assert not getattr(up.index_group, "_suffix_pass_rows", 0), "k=3 must fit one pass"
        HS.enable_split(R, sp, 2, 4, TOP_K)
        ids = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(4)
        good = 0
        for _ in range(12):
            topk = _topk(gen, 4, reqs, dev)
            a, b = _step(up, ids, bt, topk), _step(sp, ids, bt, topk)
            torch.cuda.synchronize()
            good += _check(up_raw, host, bt, ids, topk, a) and _check(sp_raw, host, bt, ids, topk, b)
        ok &= good == 12
        lines.append(f"A k=3 upstream vs forced 2-pass split: {good}/12 steps byte-exact")

        # B: k=5, the split this GPU's shared memory needs.
        rt, raw = _runtime(R, 6, reqs, dev)
        rt.bind_source_cache(host)
        r = getattr(rt.index_group, "_suffix_pass_rows", 0)
        ids6 = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(6)
        good = 0
        for _ in range(12):
            topk = _topk(gen, 6, reqs, dev)
            phys = _step(rt, ids6, bt, topk)
            torch.cuda.synchronize()
            good += _check(raw, host, bt, ids6, topk, phys)
        ok &= good == 12 and r > 0
        lines.append(f"B k=5 split into passes of {r}: {good}/12 steps byte-exact")

        # C: the k=5 step in one CUDA graph, replayed with new top-k.
        rt, raw = _runtime(R, 6, reqs, dev)
        rt.bind_source_cache(host)
        topk = _topk(gen, 6, reqs, dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _step(rt, ids6, bt, topk)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            phys = _step(rt, ids6, bt, topk)
        good = 0
        for _ in range(8):
            topk.copy_(_topk(gen, 6, reqs, dev))
            g.replay()
            torch.cuda.synchronize()
            good += _check(raw, host, bt, ids6, topk, phys)
        ok &= good == 8
        lines.append(f"C k=5 split in a CUDA graph: {good}/8 replays byte-exact")
    except Exception as exc:  # noqa: BLE001 - any error is a FAIL: refuse to serve
        import traceback

        print(f"{TAG}: ERROR {exc!r}\n{traceback.format_exc()[-1500:]}", flush=True)
        return 1
    for line in lines:
        print(f"{TAG}: {line}", flush=True)
    print(f"{TAG}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
