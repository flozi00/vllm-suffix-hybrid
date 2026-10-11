# SPDX-License-Identifier: Apache-2.0
"""Host-memory all-reduce for PCIe nodes without GPU P2P (nccl_split spec "har").

worker-06 (8x RTX PRO 6000, no P2P) sends NCCL through host memory with ring
algorithms: 14 dependent hops per all-reduce, ~61 us at 74 KiB (GLM c1) and
~0.5 ms at 2.3 MiB (c32), ~40 / ~60 % of a GLM decode step. Here every rank's
kernel writes its bf16 partial into ONE shared, pinned /dev/shm buffer that
all GPUs map (zero-copy over PCIe) and reads the other partials straight from
it:

  two-shot: write partial -> barrier -> rank r sums chunk r of all partials
    in fp32 (rank order) -> bf16 chunk to the output and to host -> barrier ->
    copy the other ranks' chunks from host into the output.
  one-shot: write partial -> barrier -> every rank sums every partial in rank
    order (identical bits on every rank).

Either way the result is the fp32 rank-order sum rounded once (NCCL's ring
rounds to bf16 at every hop), bit-identical on all ranks.

Barrier: each block fences its host stores (fence.acq_rel.sys) and arrives on
a device counter; the last block of the rank publishes the round's epoch
(st.release.sys), polls every rank's flag (ld.acquire.sys) and releases the
other blocks through a device word. Plain loads/stores only: no PCIe atomics.
Epochs come from a device counter the kernel advances itself (CUDA-graph
safe); buffers alternate by epoch parity (a buffer is rewritten only after
every rank passed the barrier of the round that read it). Messages above one
round (ROUND_BYTES per rank) loop rounds inside the kernel. A rank that waits
longer than TIMEOUT_SPINS polls traps: a mismatched collective fails loudly
instead of hanging the pod.

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
ROUND_BYTES = 4 << 20   # one round's partial per rank
FLAG_STRIDE = 16        # int32 per flag slot (one 64 B line per rank)
BLOCK = 1024            # bf16 per program step (16 B per thread at 4 warps)
TIMEOUT_SPINS = 1 << 27
WORLDS = (2, 4, 8)

if triton is not None:

    @triton.jit
    def _fence_sys():
        tl.inline_asm_elementwise("fence.acq_rel.sys;\nmov.u32 $0, 0;", "=r", [],
                                  dtype=tl.uint32, is_pure=False, pack=1)

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
    def _barrier(flags, arr_ptr, go_ptr, epoch, ctr_ptr, ctr_new, final,
                 RANK: tl.constexpr, WORLD: tl.constexpr, FLAG_STRIDE: tl.constexpr,
                 TIMEOUT: tl.constexpr):
        _fence_sys()
        tl.debug_barrier()
        old = tl.atomic_add(arr_ptr, 1, sem="acq_rel", scope="gpu")
        if old == tl.num_programs(0) - 1:
            # Last block of this rank: every block fenced its host stores before arriving.
            tl.atomic_xchg(arr_ptr, 0, sem="relaxed", scope="gpu")
            if final:
                tl.store(ctr_ptr, ctr_new)
            _st_release(flags + RANK * FLAG_STRIDE, epoch, True)
            fp = flags + tl.arange(0, WORLD) * FLAG_STRIDE
            done = tl.zeros([], dtype=tl.int32)
            spins = tl.zeros([], dtype=tl.int32)
            while done == 0:
                done = tl.min((_ld_acquire(fp, True) == epoch).to(tl.int32), axis=0)
                spins += 1
                if spins > TIMEOUT:
                    _trap()
            _st_release(go_ptr, epoch, False)
        else:
            done = tl.zeros([], dtype=tl.int32)
            spins = tl.zeros([], dtype=tl.int32)
            while done == 0:
                done = (_ld_acquire(go_ptr, False) == epoch).to(tl.int32)
                spins += 1
                if spins > TIMEOUT:
                    _trap()
        _fence_sys()

    @triton.jit
    def _har_kernel(x_ptr, out_ptr, data_ptr, red_ptr, flag_ptr, ctr_ptr, arr_ptr, n,
                    RANK: tl.constexpr, WORLD: tl.constexpr, ROUND: tl.constexpr,
                    FLAG_STRIDE: tl.constexpr, ONESHOT: tl.constexpr, BLOCK: tl.constexpr,
                    TIMEOUT: tl.constexpr):
        RCH: tl.constexpr = ROUND // WORLD
        pid = tl.program_id(0)
        step = tl.num_programs(0) * BLOCK
        ar = tl.arange(0, BLOCK)
        ctr0 = tl.load(ctr_ptr).to(tl.uint32)
        nrounds = tl.cdiv(n, ROUND)
        ctr_new = (ctr0 + nrounds.to(tl.uint32)).to(tl.int32)
        for rnd in range(0, nrounds):
            epoch = ctr0 + (rnd + 1).to(tl.uint32)
            par = (epoch & 1).to(tl.int32)
            base = rnd * ROUND
            rn = tl.minimum(n - base, ROUND)
            final = rnd == nrounds - 1
            ch = tl.cdiv(tl.cdiv(rn, WORLD), 8) * 8
            lo = RANK * ch
            hi = tl.minimum(lo + ch, rn)
            dat = data_ptr + par * (WORLD * ROUND)
            red = red_ptr + par * (WORLD * RCH)
            flags = flag_ptr + par * (2 * WORLD * FLAG_STRIDE)
            # phase 1: own partial -> host (two-shot skips the chunk only this rank reads)
            for s in range(pid * BLOCK, rn, step):
                o = s + ar
                m = o < rn
                if not ONESHOT:
                    m = m & ((o < lo) | (o >= hi))
                v = tl.load(x_ptr + base + o, mask=m)
                tl.store(dat + RANK * ROUND + o, v, mask=m, cache_modifier=".wt")
            if ONESHOT:
                _barrier(flags, arr_ptr, arr_ptr + 2, epoch, ctr_ptr, ctr_new, final,
                         RANK, WORLD, FLAG_STRIDE, TIMEOUT)
                for s in range(pid * BLOCK, rn, step):
                    o = s + ar
                    m = o < rn
                    acc = tl.zeros([BLOCK], dtype=tl.float32)
                    for j in tl.static_range(WORLD):
                        if j == RANK:
                            v = tl.load(x_ptr + base + o, mask=m, other=0.0)
                        else:
                            v = tl.load(dat + j * ROUND + o, mask=m, other=0.0, cache_modifier=".cv")
                        acc += v.to(tl.float32)
                    tl.store(out_ptr + base + o, acc.to(out_ptr.dtype.element_ty), mask=m)
            else:
                _barrier(flags, arr_ptr, arr_ptr + 2, epoch, ctr_ptr, ctr_new, False,
                         RANK, WORLD, FLAG_STRIDE, TIMEOUT)
                for s in range(lo + pid * BLOCK, hi, step):
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
                    tl.store(red + RANK * RCH + (o - lo), r, mask=m, cache_modifier=".wt")
                _barrier(flags + WORLD * FLAG_STRIDE, arr_ptr + 1, arr_ptr + 3, epoch, ctr_ptr,
                         ctr_new, final, RANK, WORLD, FLAG_STRIDE, TIMEOUT)
                for j in tl.static_range(WORLD):
                    if j != RANK:
                        jl = j * ch
                        jh = tl.minimum(jl + ch, rn)
                        for s in range(jl + pid * BLOCK, jh, step):
                            o = s + ar
                            m = o < jh
                            v = tl.load(red + j * RCH + (o - jl), mask=m, cache_modifier=".cv")
                            tl.store(out_ptr + base + o, v, mask=m)

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


class HostAllReduce:
    """bf16 all_reduce over the ranks of a gloo `cpu_group`, every rank on this node.

    oneshot_max_bytes: messages up to this size use one-shot, larger two-shot.
    """

    def __init__(self, cpu_group, device, oneshot_max_bytes: int = 0, max_blocks: int = 64,
                 round_bytes: int = ROUND_BYTES, timeout_spins: int = TIMEOUT_SPINS):
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
        self.oneshot_max, self.max_blocks, self.timeout = oneshot_max_bytes, max_blocks, timeout_spins
        flag_bytes = 4096
        data_bytes = 2 * self.world * self.round * 2
        red_bytes = 2 * self.world * (self.round // self.world) * 2
        self.size = flag_bytes + data_bytes + red_bytes
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
        err = 0
        for r in range(self.world):  # one rank at a time (concurrent pinning crashed HiSparse at TP=8)
            if r == self.rank:
                err = int(torch.cuda.cudart().cudaHostRegister(self.addr, self.size, 3))  # Portable|Mapped
            dist.barrier(group=cpu_group)
        bad = torch.tensor([float(err != 0)])
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=cpu_group)
        if bad.item():
            raise RuntimeError(f"cudaHostRegister failed (rank {self.rank}: error {err})")
        self.flags = _Ptr(self.addr, torch.int32)
        self.data = _Ptr(self.addr + flag_bytes, torch.bfloat16)
        self.red = _Ptr(self.addr + flag_bytes + data_bytes, torch.bfloat16)
        self.ctr = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.arr = torch.zeros(4, dtype=torch.int32, device=self.device)  # arrive x2, go x2
        self.disabled = False

    def supports(self, x) -> bool:
        import torch

        return (not self.disabled and x.dtype == torch.bfloat16 and x.is_cuda
                and x.is_contiguous())

    def grid(self, n: int) -> int:
        return max(1, min(self.max_blocks, -(-min(n, self.round) // BLOCK)))

    def all_reduce(self, x, oneshot: bool | None = None):
        import torch

        out = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return out
        if oneshot is None:
            oneshot = n * 2 <= self.oneshot_max
        _har_kernel[(self.grid(n),)](
            x, out, self.data, self.red, self.flags, self.ctr, self.arr, n,
            RANK=self.rank, WORLD=self.world, ROUND=self.round, FLAG_STRIDE=FLAG_STRIDE,
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


def _bench_worker(rank, world, port, sizes, q, oneshot_max, blocks) -> None:
    import torch
    import torch.distributed as dist

    from suffix_hybrid import nccl_split as ns

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=dev)
    cpu = dist.new_group(backend="gloo")
    out = {"rank": rank, "rows": [], "probe": {}, "err": None}
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        nccl = PyNcclCommunicator(group=cpu, device=dev)
        har = HostAllReduce(cpu, dev, oneshot_max_bytes=oneshot_max, max_blocks=blocks,
                            timeout_spins=1 << 23)
        # PCIe probes on this rank's own slot: GPU -> host writes, host -> GPU reads, alone
        # (rank 0 only) and with every rank at once.
        nb = 4 << 20
        src = torch.randn(nb // 2, device=dev).to(torch.bfloat16)
        dst = torch.empty_like(src)
        mine = _Ptr(har.data.addr + rank * har.round * 2, torch.bfloat16)
        other = _Ptr(har.data.addr + ((rank + 1) % world) * har.round * 2, torch.bfloat16)
        for label, fn in (("write", lambda: _copy_kernel[(128,)](src, mine, nb // 2, CV=False, WT=True,
                                                                  BLOCK=BLOCK, num_warps=4)),
                          ("read", lambda: _copy_kernel[(128,)](other, dst, nb // 2, CV=True, WT=False,
                                                                 BLOCK=BLOCK, num_warps=4))):
            for scope in ("alone", "all"):
                dist.barrier(group=cpu)
                if scope == "alone" and rank != 0:
                    dist.barrier(group=cpu)
                    continue
                us = ns.graph_time_us(fn, reps=10, replays=5)
                out["probe"][f"{label}-{scope}"] = nb / us / 1e3  # GB/s
                if scope == "alone":
                    dist.barrier(group=cpu)
        for nbytes in sizes:
            n = nbytes // 2
            x, ref = _inputs(rank, world, n, dev)
            row = {"bytes": nbytes}
            for mode, one in (("two", False), ("one", True)):
                y = har.all_reduce(x, oneshot=one)
                torch.cuda.synchronize()
                row[f"{mode}_exact"] = bool(torch.equal(y, ref))
                row[f"{mode}_maxdiff"] = float((y.float() - ref.float()).abs().max())
                row[f"{mode}_us"] = ns.graph_time_us(lambda: har.all_reduce(x, oneshot=one))
            y = nccl.all_reduce(x)
            torch.cuda.synchronize()
            row["nccl_rel"] = float((y.float() - ref.float()).norm() / ref.float().norm())
            row["nccl_us"] = ns.graph_time_us(lambda: nccl.all_reduce(x))
            row["eager_two_us"] = ns.time_us(lambda: har.all_reduce(x, oneshot=False), 7)
            out["rows"].append(row)
        har.destroy()
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["err"] = traceback.format_exc()[-1500:]
    q.put(out)
    try:
        dist.barrier(group=cpu)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def bench(sizes, oneshot_max: int = 0, blocks: int = 64) -> int:
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world not in WORLDS:
        print(f"{MARK} SKIP: {world} GPU(s)", flush=True)
        return 0
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29600 + os.getpid() % 1000
    procs = [ctx.Process(target=_bench_worker, args=(r, world, port, sizes, q, oneshot_max, blocks))
             for r in range(world)]
    [p.start() for p in procs]
    res = []
    try:
        for _ in procs:
            res.append(q.get(timeout=900))
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
    pr = {k: [r["probe"][k] for r in res if k in r["probe"]] for k in res[0]["probe"]}
    print(f"{MARK} world={world} blocks={blocks} PCIe GB/s (4 MiB, 128 programs): "
          + " | ".join(f"{k} min {min(v):.1f} max {max(v):.1f}" for k, v in pr.items()), flush=True)
    ok = True
    for i, nbytes in enumerate(sizes):
        rows = [r["rows"][i] for r in res]
        cell = {k: max(rr[k] for rr in rows) for k in ("two_us", "one_us", "nccl_us", "eager_two_us")}
        exact = {m: all(rr[f"{m}_exact"] for rr in rows) for m in ("two", "one")}
        ok &= all(exact.values())
        best = min(cell["two_us"], cell["one_us"])
        print(f"{MARK} {nbytes / 1024:9.1f} KiB ({nbytes // 12288:5d} tok@6144): two-shot "
              f"{cell['two_us']:8.1f} us | one-shot {cell['one_us']:8.1f} us | nccl {cell['nccl_us']:8.1f} us "
              f"({cell['nccl_us'] / best:4.2f}x) | eager two {cell['eager_two_us']:8.1f} | exact two "
              f"{exact['two']} one {exact['one']} (max|d| {max(rr['two_maxdiff'] for rr in rows):.3g}/"
              f"{max(rr['one_maxdiff'] for rr in rows):.3g}) nccl rel-l2 {max(rr['nccl_rel'] for rr in rows):.2e}",
              flush=True)
    print(f"{MARK} {'PASS' if ok else 'FAIL'}: every rank bit-exact vs the fp32 rank-order sum", flush=True)
    return 0 if ok else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench"])
    ap.add_argument("--tokens", default="1,6,12,24,48,96,192,384,1024,4096,8192")
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--blocks", type=int, default=64)
    a = ap.parse_args()
    sizes = [t * a.hidden * 2 for t in map(int, a.tokens.split(","))]
    return bench(sizes, blocks=a.blocks)


if __name__ == "__main__":
    sys.exit(main())
