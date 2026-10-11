# SPDX-License-Identifier: Apache-2.0
"""Host-memory all-reduce for PCIe nodes without GPU P2P (nccl_split spec "har").

worker-06 (8x RTX PRO 6000, no P2P) sends NCCL through host memory with ring
algorithms: 14 dependent hops per all-reduce, ~44 us graphed at 72 KiB (GLM
c1) and ~0.4 ms at 2.3 MiB (c32) for the best NCCL algorithm. Here every
rank's kernel writes its bf16 partial into ONE shared, pinned /dev/shm buffer
that all GPUs map (zero-copy over PCIe, ~50 GB/s each way per GPU) and reads
the other partials straight from it:

  two-shot: write partial -> flag -> rank r sums chunk r of all partials in
    fp32 (rank order) -> bf16 chunk to the output and to host -> flag -> copy
    the other ranks' chunks from host into the output.
  one-shot: write partial -> flag -> every rank sums every partial in rank
    order (identical bits on every rank).

Either way the result is the fp32 rank-order sum rounded once (NCCL's ring
rounds to bf16 at every hop), bit-identical on all ranks.

Synchronization is per program: program b of every rank owns the same slice
of every chunk, publishes the round's epoch for it (bar.sync, then
st.release.sys, as vLLM's custom all-reduce does) and polls only the eight
flags of slice b (one 32 B line, ld.acquire.sys) -- no grid-wide barrier, no
PCIe atomics. Epochs come from a device counter the kernel advances itself
(CUDA-graph safe); buffers alternate by epoch parity (program b rewrites a
slice only after program b of every rank passed the round that read it).
Messages above one round (ROUND_BYTES per rank) loop rounds in the kernel.
A program that waits longer than TIMEOUT_SPINS polls traps: a mismatched
collective fails loudly instead of hanging the pod.

Page placement: each rank first-touches the slots it writes from a CPU of its
GPU's NUMA node (pages stay put once pinned), so the writes stay socket-local
and both sockets' memory controllers serve the reads.

    python -m suffix_hybrid.host_ar bench   # boot gate host_ar_bench: exactness + graphed us
"""
from __future__ import annotations

import ctypes
import mmap
import os
import sys
import uuid

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU CI: the Python side stays importable
    triton = tl = None

MARK = "[suffix har]"
ROUND_BYTES = 4 << 20    # one round's partial per rank
MAXB = 128               # programs per launch (flag columns)
FLAG_BYTES = 64 << 10    # 2 parities x 2 phases x MAXB x world int32
BLOCK = 1024             # bf16 per program step (16 B per thread at 4 warps)
ELEMS_PER_PROGRAM = 2048  # grid = clamp(first round / this, 1, max_blocks)
TIMEOUT_SPINS = 1 << 27
WORLDS = (2, 4, 8)

if triton is not None:

    @triton.jit
    def _st_release(ptr, val, SYS: tl.constexpr):
        if SYS:
            tl.inline_asm_elementwise("st.release.sys.global.u32 [$1], $2;\nmov.u32 $0, 0;",
                                      "=r,l,r", [ptr, val], dtype=tl.uint32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise("st.release.gpu.global.u32 [$1], $2;\nmov.u32 $0, 0;",
                                      "=r,l,r", [ptr, val], dtype=tl.uint32, is_pure=False, pack=1)

    @triton.jit
    def _ld_acquire(ptrs, SYS: tl.constexpr):
        if SYS:
            v = tl.inline_asm_elementwise("ld.acquire.sys.global.u32 $0, [$1];", "=r,l", [ptrs],
                                          dtype=tl.uint32, is_pure=False, pack=1)
        else:
            v = tl.inline_asm_elementwise("ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", [ptrs],
                                          dtype=tl.uint32, is_pure=False, pack=1)
        return v

    @triton.jit
    def _trap():
        tl.inline_asm_elementwise("trap;\nmov.u32 $0, 0;", "=r", [], dtype=tl.uint32,
                                  is_pure=False, pack=1)

    @triton.jit
    def _publish_wait(fp, epoch, RANK: tl.constexpr, WORLD: tl.constexpr, TIMEOUT: tl.constexpr):
        """Every thread's host stores happen before this program's flag; then wait for the
        same program of every other rank."""
        tl.debug_barrier()
        _st_release(fp + RANK, epoch, True)
        ranks = tl.arange(0, WORLD)
        done = tl.zeros([], dtype=tl.int32)
        spins = tl.zeros([], dtype=tl.int32)
        while done == 0:
            f = _ld_acquire(fp + ranks, True)
            done = tl.min(((f == epoch) | (ranks == RANK)).to(tl.int32), axis=0)
            spins += 1
            if spins > TIMEOUT:
                _trap()

    @triton.jit
    def _har_kernel(x_ptr, out_ptr, data_ptr, red_ptr, flag_ptr, ctr_ptr, arr_ptr, n,
                    RANK: tl.constexpr, WORLD: tl.constexpr, ROUND: tl.constexpr,
                    MAXB: tl.constexpr, ONESHOT: tl.constexpr, BLOCK: tl.constexpr,
                    TIMEOUT: tl.constexpr):
        RCH: tl.constexpr = ROUND // WORLD
        b = tl.program_id(0)
        nb = tl.num_programs(0)
        ar = tl.arange(0, BLOCK)
        ctr0 = tl.load(ctr_ptr).to(tl.uint32)
        nrounds = tl.cdiv(n, ROUND)
        for rnd in range(0, nrounds):
            epoch = ctr0 + (rnd + 1).to(tl.uint32)
            par = (epoch & 1).to(tl.int32)
            base = rnd * ROUND
            rn = tl.multiple_of(tl.minimum(n - base, ROUND), 8)
            dat = data_ptr + par * (WORLD * ROUND)
            f1 = flag_ptr + ((par * 2) * MAXB + b) * WORLD
            if ONESHOT:
                sub = tl.cdiv(tl.cdiv(rn, nb), 8) * 8
                lo = tl.multiple_of(b * sub, 8)
                hi = tl.multiple_of(tl.minimum(lo + sub, rn), 8)
                for s in range(lo, hi, BLOCK):
                    o = s + ar
                    m = o < hi
                    v = tl.load(x_ptr + base + o, mask=m)
                    tl.store(dat + RANK * ROUND + o, v, mask=m, cache_modifier=".wt")
                _publish_wait(f1, epoch, RANK, WORLD, TIMEOUT)
                for s in range(lo, hi, BLOCK):
                    o = s + ar
                    m = o < hi
                    acc = tl.zeros([BLOCK], dtype=tl.float32)
                    for j in tl.static_range(WORLD):
                        if j == RANK:
                            v = tl.load(x_ptr + base + o, mask=m, other=0.0)
                        else:
                            v = tl.load(dat + j * ROUND + o, mask=m, other=0.0, cache_modifier=".cv")
                        acc += v.to(tl.float32)
                    tl.store(out_ptr + base + o, acc.to(out_ptr.dtype.element_ty), mask=m)
            else:
                red = red_ptr + par * (WORLD * RCH)
                ch = tl.cdiv(tl.cdiv(rn, WORLD), 8) * 8
                sub = tl.cdiv(tl.cdiv(ch, nb), 8) * 8
                s0 = tl.multiple_of(b * sub, 8)
                s1 = tl.multiple_of(tl.minimum(s0 + sub, ch), 8)
                for c in tl.static_range(WORLD):  # phase 1: own partial of the others' chunks
                    if c != RANK:
                        lo = tl.multiple_of(c * ch + s0, 8)
                        hi = tl.multiple_of(tl.minimum(c * ch + s1, rn), 8)
                        for s in range(lo, hi, BLOCK):
                            o = s + ar
                            m = o < hi
                            v = tl.load(x_ptr + base + o, mask=m)
                            tl.store(dat + RANK * ROUND + o, v, mask=m, cache_modifier=".wt")
                _publish_wait(f1, epoch, RANK, WORLD, TIMEOUT)
                lo = tl.multiple_of(RANK * ch + s0, 8)  # phase 2: reduce own chunk
                hi = tl.multiple_of(tl.minimum(RANK * ch + s1, rn), 8)
                for s in range(lo, hi, BLOCK):
                    o = s + ar
                    m = o < hi
                    acc = tl.zeros([BLOCK], dtype=tl.float32)
                    for j in tl.static_range(WORLD):
                        if j == RANK:
                            v = tl.load(x_ptr + base + o, mask=m, other=0.0)
                        else:
                            v = tl.load(dat + j * ROUND + o, mask=m, other=0.0, cache_modifier=".cv")
                        acc += v.to(tl.float32)
                    r = acc.to(out_ptr.dtype.element_ty)
                    tl.store(out_ptr + base + o, r, mask=m)
                    tl.store(red + RANK * RCH + (o - RANK * ch), r, mask=m, cache_modifier=".wt")
                _publish_wait(f1 + MAXB * WORLD, epoch, RANK, WORLD, TIMEOUT)
                for c in tl.static_range(WORLD):  # phase 3: the others' reduced chunks
                    if c != RANK:
                        lo = tl.multiple_of(c * ch + s0, 8)
                        hi = tl.multiple_of(tl.minimum(c * ch + s1, rn), 8)
                        for s in range(lo, hi, BLOCK):
                            o = s + ar
                            m = o < hi
                            v = tl.load(red + c * RCH + (o - c * ch), mask=m, cache_modifier=".cv")
                            tl.store(out_ptr + base + o, v, mask=m)
        # Every program read ctr0 above; the last one to finish advances the epoch counter.
        old = tl.atomic_add(arr_ptr, 1, sem="acq_rel", scope="gpu")
        if old == nb - 1:
            tl.store(arr_ptr, 0)
            tl.store(ctr_ptr, (ctr0 + nrounds.to(tl.uint32)).to(tl.int32))

    @triton.jit
    def _copy_kernel(src, dst, n, CV: tl.constexpr, WT: tl.constexpr, BLOCK: tl.constexpr):
        """Bandwidth probe: dst[:n] = src[:n] (host-mapped either side)."""
        for s in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
            o = s + tl.arange(0, BLOCK)
            m = o < n
            if CV:
                v = tl.load(src + o, mask=m, cache_modifier=".cv")
            else:
                v = tl.load(src + o, mask=m)
            if WT:
                tl.store(dst + o, v, mask=m, cache_modifier=".wt")
            else:
                tl.store(dst + o, v, mask=m)


class _Ptr:
    """data_ptr()/dtype view of a mapped host address for Triton launches (Triton resolves
    it to the device pointer with cuPointerGetAttribute)."""

    def __init__(self, addr: int, dtype):
        self.addr, self.dtype = addr, dtype

    def data_ptr(self) -> int:
        return self.addr


def gpu_numa_node(device) -> int:
    """NUMA node of the GPU's PCI device (-1 if unknown or a single node)."""
    import torch

    p = torch.cuda.get_device_properties(device)
    try:
        bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        return int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
    except (AttributeError, OSError, ValueError):
        return -1


def node_cpus(node: int) -> set[int]:
    out: set[int] = set()
    try:
        for part in open(f"/sys/devices/system/node/node{node}/cpulist").read().strip().split(","):
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    except (OSError, ValueError):
        pass
    return out


def numa_pages(addr: int) -> str:
    """'N0=.. N1=..' page counts of the mapping that starts at addr (/proc/self/numa_maps)."""
    try:
        for line in open("/proc/self/numa_maps"):
            if int(line.split()[0], 16) == addr:
                return " ".join(t for t in line.split() if t[:1] == "N" and "=" in t)
    except (OSError, ValueError):
        pass
    return "?"


class HostAllReduce:
    """bf16 all_reduce over the ranks of a gloo `cpu_group`, every rank on this node.

    oneshot_max_bytes: messages up to this size use one-shot, larger two-shot.
    numa: "owner" (each rank first-touches the slots it writes on its GPU's node) or
    "rank0" (rank 0 touches everything: one node serves all traffic).
    """

    def __init__(self, cpu_group, device, oneshot_max_bytes: int = 0, max_blocks: int = 64,
                 round_bytes: int = ROUND_BYTES, timeout_spins: int = TIMEOUT_SPINS,
                 numa: str = "owner", elems_per_program: int = ELEMS_PER_PROGRAM):
        import torch
        import torch.distributed as dist

        self.disabled = True
        self.rank, self.world = dist.get_rank(cpu_group), dist.get_world_size(cpu_group)
        if self.world not in WORLDS:
            raise ValueError(f"host all-reduce supports world {WORLDS}, not {self.world}")
        self.device = torch.device(device)
        self.round = round_bytes // 2
        if self.round % (self.world * BLOCK):
            raise ValueError(f"round_bytes must be a multiple of {2 * self.world * BLOCK}")
        self.oneshot_max, self.timeout = oneshot_max_bytes, timeout_spins
        self.max_blocks, self.per_program = min(max_blocks, MAXB), elems_per_program
        rch = self.round // self.world
        data_bytes = 2 * self.world * self.round * 2
        self.size = FLAG_BYTES + data_bytes + 2 * self.world * rch * 2
        name = [f"/dev/shm/suffix-har-{uuid.uuid4().hex}" if self.rank == 0 else None]
        if self.rank == 0:
            fd = os.open(name[0], os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, self.size)  # zero-filled: every flag starts at epoch 0
            os.close(fd)
        dist.broadcast_object_list(name, src=dist.get_global_rank(cpu_group, 0), group=cpu_group)
        fd = os.open(name[0], os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        dist.barrier(group=cpu_group)
        if self.rank == 0:
            os.unlink(name[0])  # the mappings keep it alive; nothing left behind on exit
        self._cbuf = ctypes.c_char.from_buffer(self._mm)
        self.addr = ctypes.addressof(self._cbuf)
        # First touch decides the node of every page (pinned pages never migrate).
        self.node = gpu_numa_node(self.device)
        own = [(FLAG_BYTES + (p * self.world + self.rank) * self.round * 2, self.round * 2)
               for p in (0, 1)]
        own += [(FLAG_BYTES + data_bytes + (p * self.world + self.rank) * rch * 2, rch * 2)
                for p in (0, 1)]
        if numa == "rank0":
            own = [(0, self.size)] if self.rank == 0 else []
        elif self.rank == 0:
            own.append((0, FLAG_BYTES))
        self._touch(own)
        dist.barrier(group=cpu_group)
        err = 0
        for r in range(self.world):  # one rank at a time (concurrent pinning crashed HiSparse at TP=8)
            if r == self.rank:
                err = int(torch.cuda.cudart().cudaHostRegister(self.addr, self.size, 3))  # Portable|Mapped
            dist.barrier(group=cpu_group)
        bad = torch.tensor([float(err != 0)])
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=cpu_group)
        if bad.item():
            raise RuntimeError(f"cudaHostRegister failed (rank {self.rank}: error {err})")
        self.pages = numa_pages(self.addr)
        self.flags = _Ptr(self.addr, torch.int32)
        self.data = _Ptr(self.addr + FLAG_BYTES, torch.bfloat16)
        self.red = _Ptr(self.addr + FLAG_BYTES + data_bytes, torch.bfloat16)
        self.ctr = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.arr = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.disabled = False

    def _touch(self, regions) -> None:
        cpus = node_cpus(self.node) if self.node >= 0 else set()
        cur = os.sched_getaffinity(0)
        if cpus & cur:
            os.sched_setaffinity(0, cpus & cur)
        try:
            for off, size in regions:
                ctypes.memset(self.addr + off, 0, size)
        finally:
            os.sched_setaffinity(0, cur)

    def supports(self, x) -> bool:
        import torch

        return (not self.disabled and x.dtype == torch.bfloat16 and x.is_cuda
                and x.is_contiguous() and x.numel() % 8 == 0)

    def grid(self, n: int) -> int:
        return max(1, min(self.max_blocks, -(-min(n, self.round) // self.per_program)))

    def all_reduce(self, x, oneshot: bool | None = None, grid: int | None = None):
        import torch

        out = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return out
        if oneshot is None:
            oneshot = n * 2 <= self.oneshot_max
        _har_kernel[(grid or self.grid(n),)](
            x, out, self.data, self.red, self.flags, self.ctr, self.arr, n,
            RANK=self.rank, WORLD=self.world, ROUND=self.round, MAXB=MAXB,
            ONESHOT=bool(oneshot), BLOCK=BLOCK, TIMEOUT=self.timeout, num_warps=4)
        return out

    def destroy(self) -> None:
        import torch

        if self.disabled:
            return
        self.disabled = True
        torch.cuda.synchronize(self.device)
        torch.cuda.cudart().cudaHostUnregister(self.addr)
        del self._cbuf
        self._mm.close()


# ------------------------------------------------------------------ boot gate
def _inputs(rank: int, world: int, n: int, device):
    """Every rank regenerates all ranks' inputs (seeded) -> the exact reference needs no comm."""
    import torch

    xs = []
    for r in range(world):
        g = torch.Generator(device=device).manual_seed(1000 * n + r)
        xs.append(torch.randn(n, device=device, generator=g).to(torch.bfloat16))
    acc = xs[0].float()
    for t in xs[1:]:
        acc = acc + t.float()  # rank order, fp32: the kernel's exact arithmetic
    return xs[rank], acc.to(torch.bfloat16)


# (name, HostAllReduce kwargs, token counts, grids to sweep (None = policy))
VARIANTS = (
    ("owner-4M", {}, (1, 6, 12, 24, 48, 96, 192, 384, 1024, 8192), (None,)),
    ("owner-4M-grid", {}, (6, 48, 192), (4, 8, 16, 32, 96)),
    ("rank0-4M", {"numa": "rank0"}, (6, 48, 192, 1024, 8192), (None,)),
    ("owner-1M", {"round_bytes": 1 << 20}, (48, 192, 1024, 8192), (None,)),
)


def _bench_worker(rank, world, port, q, hidden) -> None:
    import torch
    import torch.distributed as dist

    from suffix_hybrid import nccl_split as ns

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=dev)
    cpu = dist.new_group(backend="gloo")
    out = {"rank": rank, "rows": [], "probe": {}, "err": None, "node": gpu_numa_node(dev),
           "pages": {}}
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        nccl = PyNcclCommunicator(group=cpu, device=dev)
        nccl_us = {}
        for vname, kw, toks, grids in VARIANTS:
            har = HostAllReduce(cpu, dev, timeout_spins=1 << 23, **kw)
            out["pages"][vname] = har.pages
            if vname == "owner-4M":  # PCIe probes on this rank's own slot
                nb = 4 << 20
                src = torch.randn(nb // 2, device=dev).to(torch.bfloat16)
                dst = torch.empty_like(src)
                mine = _Ptr(har.data.addr + rank * har.round * 2, torch.bfloat16)
                other = _Ptr(har.data.addr + ((rank + 1) % world) * har.round * 2, torch.bfloat16)
                for label, fn in (
                        ("write", lambda: _copy_kernel[(128,)](src, mine, nb // 2, CV=False, WT=True,
                                                               BLOCK=BLOCK, num_warps=4)),
                        ("read", lambda: _copy_kernel[(128,)](other, dst, nb // 2, CV=True, WT=False,
                                                              BLOCK=BLOCK, num_warps=4))):
                    for scope in ("alone", "all"):
                        dist.barrier(group=cpu)
                        if scope == "alone" and rank != 0:
                            dist.barrier(group=cpu)
                            continue
                        out["probe"][f"{label}-{scope}"] = nb / ns.graph_time_us(fn, 10, 5) / 1e3
                        if scope == "alone":
                            dist.barrier(group=cpu)
            for t in toks:
                n = t * hidden
                x, ref = _inputs(rank, world, n, dev)
                for grid in grids:
                    row = {"v": vname, "tok": t, "grid": grid or har.grid(n)}
                    for mode, one in (("two", False), ("one", True)):
                        if one and t > 1024:
                            continue
                        y = har.all_reduce(x, oneshot=one, grid=grid)
                        torch.cuda.synchronize()
                        row[f"{mode}_exact"] = bool(torch.equal(y, ref))
                        row[f"{mode}_us"] = ns.graph_time_us(lambda: har.all_reduce(x, oneshot=one, grid=grid))
                    if t not in nccl_us:
                        y = nccl.all_reduce(x)
                        torch.cuda.synchronize()
                        nccl_us[t] = ns.graph_time_us(lambda: nccl.all_reduce(x))
                    row["nccl_us"] = nccl_us[t]
                    out["rows"].append(row)
                del x, ref
            har.destroy()
    except Exception:  # noqa: BLE001
        import traceback

        out["err"] = traceback.format_exc()[-1500:]
    q.put(out)
    try:
        dist.barrier(group=cpu)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def bench(hidden: int = 6144) -> int:
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world not in WORLDS:
        print(f"{MARK} SKIP: {world} GPU(s)", flush=True)
        return 0
    nodes = sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d[:4] == "node"
                   and d[4:].isdigit()) if os.path.isdir("/sys/devices/system/node") else []
    print(f"{MARK} host: {os.cpu_count()} cpus, numa nodes "
          + ", ".join(f"{nd}: {len(node_cpus(nd))} cpus" for nd in nodes), flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29600 + os.getpid() % 1000
    procs = [ctx.Process(target=_bench_worker, args=(r, world, port, q, hidden)) for r in range(world)]
    [p.start() for p in procs]
    res = []
    try:
        for _ in procs:
            res.append(q.get(timeout=1500))
    except Exception as exc:  # noqa: BLE001
        print(f"{MARK} FAIL: {len(res)}/{world} ranks reported ({exc!r})", flush=True)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    errs = [r for r in res if r["err"]]
    for r in errs[:2]:
        print(f"{MARK} rank {r['rank']} ERROR: {r['err']}", flush=True)
    if errs or len(res) < world:
        return 1
    res.sort(key=lambda r: r["rank"])
    print(f"{MARK} GPU numa nodes by rank: {[r['node'] for r in res]}; pages "
          + " | ".join(f"{v}: {p}" for v, p in res[0]["pages"].items()), flush=True)
    pr = {k: [r["probe"][k] for r in res if k in r["probe"]] for k in res[0]["probe"]}
    print(f"{MARK} PCIe GB/s (4 MiB, 128 programs): "
          + " | ".join(f"{k} min {min(v):.1f} max {max(v):.1f}" for k, v in pr.items()), flush=True)
    ok = True
    for i, row in enumerate(res[0]["rows"]):
        rows = [r["rows"][i] for r in res]
        cell = {k: max(rr[k] for rr in rows) for k in ("two_us", "one_us", "nccl_us") if k in row}
        exact = all(rr[k] for rr in rows for k in ("two_exact", "one_exact") if k in rr)
        ok &= exact
        best = min(v for k, v in cell.items() if k != "nccl_us")
        one = f"{cell['one_us']:8.1f}" if "one_us" in cell else "       -"
        print(f"{MARK} {row['v']:13s} {row['tok']:5d} tok ({row['tok'] * hidden * 2 / 1024:8.1f} KiB) "
              f"grid {row['grid']:3d}: two-shot {cell['two_us']:8.1f} us | one-shot {one} us | nccl "
              f"{cell['nccl_us']:8.1f} us ({cell['nccl_us'] / best:4.2f}x) | exact {exact}", flush=True)
    print(f"{MARK} {'PASS' if ok else 'FAIL'}: every rank bit-exact vs the fp32 rank-order sum", flush=True)
    return 0 if ok else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench"])
    ap.add_argument("--hidden", type=int, default=6144)
    a = ap.parse_args()
    return bench(a.hidden)


if __name__ == "__main__":
    sys.exit(main())
