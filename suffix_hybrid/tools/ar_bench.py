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
        [--matrix] [--qar] [--hidden 6144]

--matrix: every nccl_split.CANDIDATES communicator x the autotune sizes, timed
by the SAME code as SUFFIX_NCCL_AUTOTUNE (exact-sum check, median, MAX over
ranks, no budget) + the bands autotune would pick. --qar (implies --matrix):
nccl_qar int8/fp8 compressed all-reduce vs the best NCCL candidate per size,
rel-l2 error vs an fp32 NCCL sum, and whether all ranks got identical bytes.
"""
from __future__ import annotations

import argparse
import os
import sys

MARK = "[suffix ar-bench]"


def _matrix(cpu, rank, hidden, qar, out) -> None:
    import torch

    from suffix_hybrid import nccl_split as ns
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    dev = torch.device("cuda", rank)
    auto = PyNcclCommunicator(group=cpu, device=dev)
    sizes = ns.autotune_sizes(hidden)
    table, comms, errors = ns.tune(cpu, dev, auto, sizes, budget_s=1e9)
    for spec, comm in comms.items():
        if spec != "default":
            comm.destroy()
    out["matrix"] = {"sizes": sizes, "table": table, "errors": errors}
    if not qar:
        return
    from suffix_hybrid.nccl_qar import PyncclQar

    rows = []
    for nbytes in sizes:
        g = torch.Generator(device="cuda").manual_seed(4321 + rank)
        x = torch.randn(nbytes // 2, device="cuda", dtype=torch.bfloat16, generator=g)
        exact = auto.all_reduce(x.float())
        row = {"bytes": nbytes}
        for mode in ("int8", "fp8"):
            qc = PyncclQar(auto, mode)
            y = qc.all_reduce(x)
            row[mode] = {"err": float((y.float() - exact).norm() / exact.norm()),
                         "sum": float(y.view(torch.int16).double().sum()),
                         "us": ns.time_us(lambda: qc.all_reduce(x), 7 if nbytes <= 8 << 20 else 3)}
            del y
        rows.append(row)
    out["qar"] = rows


def _worker(rank: int, world: int, port: int, sizes, iters: int, q, split: str = "",
            matrix: bool = False, qar: bool = False, hidden: int = 6144) -> None:
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
    if split:
        # Two vLLM PyNccl communicators, the second built while NCCL_ALGO is set
        # (suffix_hybrid/nccl_split.py relies on NCCL reading it per comm init).
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        dev = torch.device("cuda", rank)
        auto = PyNcclCommunicator(group=cpu, device=dev)
        algo, _, proto = split.partition("/")  # e.g. "allreduce:ring/Simple"
        os.environ["NCCL_ALGO"] = algo
        if proto:
            os.environ["NCCL_PROTO"] = proto
        try:
            small = PyNcclCommunicator(group=cpu, device=dev)
        finally:
            os.environ.pop("NCCL_ALGO", None)
            os.environ.pop("NCCL_PROTO", None)
        for kib in sizes + [256, 384, 512, 1024, 2048, 3072, 6144, 12288, 24576]:
            x = torch.randn(kib * 512, device="cuda", dtype=torch.bfloat16)
            a_us = timed(lambda t: auto.all_reduce(t), x.clone())
            s_us = timed(lambda t: small.all_reduce(t), x.clone())
            out["rows"].append({"kib": kib, "split": True, "auto_us": a_us, "small_us": s_us})
    if matrix or qar:
        _matrix(cpu, rank, hidden, qar, out)
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
    ap.add_argument("--split", default="", help="NCCL_ALGO[/NCCL_PROTO] for a second PyNccl comm, "
                    "e.g. allreduce:tree or allreduce:ring/Simple")
    ap.add_argument("--matrix", action="store_true", help="all autotune candidates x sizes")
    ap.add_argument("--qar", action="store_true", help="compressed all-reduce vs best NCCL (implies --matrix)")
    ap.add_argument("--hidden", type=int, default=6144, help="model hidden size for the token sizes")
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
    procs = [ctx.Process(target=_worker, args=(r, world, port, sizes, a.iters, q, a.split,
                                                  a.matrix, a.qar, a.hidden))
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
    for i in range(len(res[0]["rows"])):
        if res[0]["rows"][i].get("split"):
            rows = [r["rows"][i] for r in res]
            au, sm = max(r["auto_us"] for r in rows), max(r["small_us"] for r in rows)
            print(f"{MARK} split {rows[0]['kib']:6d} KiB: default-comm {au:8.1f} us | "
                  f"NCCL_ALGO={a.split} comm {sm:8.1f} us ({au / sm:4.2f}x)", flush=True)
            continue
        kib = sizes[i]
        rows = [r["rows"][i] for r in res]
        nccl = max(r["nccl_us"] for r in rows)
        line = f"{MARK} {kib:5d} KiB bf16: nccl {nccl:8.1f} us"
        if all("ca_us" in r for r in rows):
            ca = max(r["ca_us"] for r in rows)
            err = max(r["ca_max_abs_err"] for r in rows)
            line += f" | custom-AR {ca:8.1f} us ({nccl / ca:4.2f}x) max|err| {err:.3g}"
        print(line, flush=True)
    if "matrix" in res[0]:
        _print_matrix(res)
    return 0


def _print_matrix(res) -> None:
    from suffix_hybrid import nccl_split as ns

    m = res[0]["matrix"]  # already MAX-reduced over ranks inside tune()
    sizes, table = m["sizes"], m["table"]
    short = {c: c.removeprefix("allreduce:") for c in table}
    for i, nbytes in enumerate(sizes):
        cells = " | ".join(f"{short[c]} {table[c][i]:8.1f}" for c in table)
        best = min(table, key=lambda c: table[c][i])
        line = (f"{MARK} matrix {ns.fmt_size(nbytes):>6s}: {cells} | best {short[best]} "
                f"({table['default'][i] / table[best][i]:4.2f}x vs default)")
        if "qar" in res[0]:
            rows = [r["qar"][i] for r in res]
            for mode in ("int8", "fp8"):
                us = max(r[mode]["us"] for r in rows)
                same = len({r[mode]["sum"] for r in rows}) == 1
                line += (f" | qar-{mode} {us:8.1f} us ({table[best][i] / us:4.2f}x vs best) "
                         f"rel-l2 {rows[0][mode]['err']:.2e} ranks-identical {same}")
        print(line, flush=True)
    print(f"{MARK} matrix failed: {m['errors'] or '-'}", flush=True)
    print(f"{MARK} matrix bands (margin 5 %): SUFFIX_NCCL_BANDS={ns.bands_str(ns.build_bands(sizes, table))}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
