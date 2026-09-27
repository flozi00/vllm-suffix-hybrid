# SPDX-License-Identifier: Apache-2.0
"""All-reduce microbench for PCIe-only multi-GPU nodes (boot gate allreduce_bench).

vLLM 0.30.0 disables every fast all-reduce on 8x RTX PRO 6000 (PCIe, SM120):
custom AR (">2 PCIe-only GPUs"), FlashInfer AR (world_size 8), SymmMem (cc 12.0).
Prod glm decode then spends ~31 % of each step in NCCL RING_LL all-reduces
(172 per step, ~61 us each; worker profile 2026-09-27). This measures, on all
visible GPUs, bf16 all-reduce latency at decode-sized messages for NCCL and for
vLLM's CustomAllreduce forced past the NVLink check (its own P2P test still
runs), and checks both give the same sums. Prints one "[suffix ar-bench]" line
per size plus the P2P access matrix. Evidence only; exit 0 unless it crashed.

    python -m suffix_hybrid.tools.ar_bench [--sizes-kib 12,48,96,192,384,768,1536] [--iters 200]
"""
from __future__ import annotations

import argparse
import os
import sys

MARK = "[suffix ar-bench]"


def _worker(rank: int, world: int, port: int, sizes, iters: int, q) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    out = {"rank": rank, "rows": [], "ca": None}
    ca = None
    try:
        from vllm.platforms import current_platform
        from vllm.distributed.device_communicators import custom_all_reduce as car

        current_platform.is_fully_connected = lambda ids: True  # bypass the NVLink-only gate
        ca = car.CustomAllreduce(group=cpu, device=torch.device("cuda", rank),
                                 max_size=max(sizes) * 1024 + 4096)
        out["ca"] = "disabled" if ca.disabled else "enabled"
    except Exception as exc:  # noqa: BLE001
        out["ca"] = f"error: {exc!r}"[:300]
        ca = None

    def timed(fn, t):
        for _ in range(20):
            fn(t)
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(t)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / iters  # us

    for kib in sizes:
        n = kib * 1024 // 2
        g = torch.Generator(device="cuda").manual_seed(1234 + rank)
        x = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
        ref = x.clone()
        dist.all_reduce(ref)
        row = {"kib": kib, "nccl_us": timed(lambda t: dist.all_reduce(t), x.clone())}
        if ca is not None and not ca.disabled and ca.should_custom_ar(x):
            got = ca.custom_all_reduce(x.clone())
            row["ca_max_abs_err"] = float((got.float() - ref.float()).abs().max())
            row["ca_us"] = timed(lambda t: ca.custom_all_reduce(t), x.clone())
        out["rows"].append(row)
    if rank == 0:
        out["p2p"] = [[int(i == j or torch.cuda.can_device_access_peer(i, j))
                       for j in range(world)] for i in range(world)]
    q.put(out)
    dist.barrier(group=cpu)
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-kib", default="12,48,96,192,384,768,1536")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world < 2:
        print(f"{MARK} SKIP: {world} GPU(s)", flush=True)
        return 0
    sizes = [int(s) for s in a.sizes_kib.split(",")]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_worker, args=(r, world, port, sizes, a.iters, q))
             for r in range(world)]
    [p.start() for p in procs]
    res = [q.get(timeout=900) for _ in procs]
    [p.join(timeout=120) for p in procs]
    res.sort(key=lambda r: r["rank"])
    print(f"{MARK} world={world} NCCL_ALGO={os.environ.get('NCCL_ALGO', 'auto')} "
          f"NCCL_PROTO={os.environ.get('NCCL_PROTO', 'auto')} custom-AR: "
          f"{sorted({r['ca'] for r in res})}", flush=True)
    if "p2p" in res[0]:
        print(f"{MARK} p2p access matrix rows: {res[0]['p2p']}", flush=True)
    for i, kib in enumerate(sizes):
        rows = [r["rows"][i] for r in res]
        nccl = max(r["nccl_us"] for r in rows)
        line = f"{MARK} {kib:5d} KiB bf16: nccl {nccl:8.1f} us"
        if all("ca_us" in r for r in rows):
            ca = max(r["ca_us"] for r in rows)
            err = max(r["ca_max_abs_err"] for r in rows)
            line += f" | custom-AR {ca:8.1f} us ({nccl / ca:4.2f}x) max|err| {err:.3g}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
