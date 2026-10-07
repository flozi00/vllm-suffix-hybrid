#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-call HOST time of the per-step Python paths trimmed on cpu-diet.

  cpu    -- no GPU needed. lm_head: the eager hot path (logits_nvfp4, ~15
            torch ops) vs the graph path (copy_ + replay + slice) on META
            tensors: pure dispatcher/Python cost, no kernels, no launches.
            own_attn: the old per-call q-lens torch ops vs the per-step
            numpy cache (18 calls/step = 6 KV groups x 3 asks).
  gpu    -- in-pod (sm_120 + oxide bundle + vLLM): real NVFP4 head at the
            gemma shape (262144 x 2816), host us/call of run() (eager) vs
            replay() (graph), measured as wall time of N back-to-back calls
            with NO sync inside the window (the GPU queue absorbs the work,
            so this is the CPU cost the engine thread pays), plus device us.
  embed  -- in-pod: splits the gemma4 text-embed path (V2 encoder_runner
            get_inputs_embeds, text-only decode, 1 token) into its host ops,
            GPU idle vs busy, to localise the 0.26 ms VocabParallelEmbedding
            self time seen in the GIL profile.

  python bench/host_path_bench.py cpu
  python bench/host_path_bench.py gpu [--m 1 4 16 64] [--iters 2000]
  python bench/host_path_bench.py embed [--iters 2000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _us(fn, iters, sync=None):
    for _ in range(min(50, iters)):
        fn()
    if sync:
        sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    host = (time.perf_counter() - t) * 1e6 / iters
    if sync:
        sync()
    return host


def cpu(iters):
    import torch

    from suffix_hybrid.kernels import nvfp4_lm_head as lh
    n, k = 1024, 64
    dev = torch.device("cpu")
    w = torch.zeros(n, k, dtype=torch.bfloat16, device=dev)

    def quant(x2, g):
        return (torch.empty(x2.shape[0], k // 2, dtype=torch.uint8, device=dev),
                torch.empty(x2.shape[0] * k // 16, dtype=torch.uint8, device=dev))

    def gemm(xq, xsf, alpha):
        return torch.zeros(xq.shape[0], n, dtype=torch.bfloat16, device=dev)

    class G:
        def replay(self):
            pass

    for m in (1, 4, 64):
        x = torch.ones(m, k, dtype=torch.bfloat16, device=dev)
        st = {"gx": torch.empty(64, k, dtype=torch.bfloat16, device=dev),
              "gout": torch.empty(64, n, dtype=torch.bfloat16, device=dev),
              "graphs": [None] + [G() for _ in range(64)]}
        a = _us(lambda: lh.logits_nvfp4(x, w, 1.0, quant, gemm), iters)
        b = _us(lambda: lh.replay(st, x), iters)
        print(f"lm_head M={m:2d} cpu dispatch: eager {a:7.1f} us/call -> graph path "
              f"{b:5.1f} us/call (+ cudaGraphLaunch on device builds)")

    q = torch.tensor([0, 1], dtype=torch.int32)

    def old():  # pre-cpu-diet: _takes x2 + _decode (2 more max().item + tolist)
        for _ in range(3):
            ql = q[1:2] - q[:1]
            int(ql.max().item())
        ql.tolist()

    from sm120.tests.test_nvfp4_own_attn import _helper_ns, own_attn
    ns = _helper_ns(_nvfp4_own_attn=own_attn)
    ql = ns["_nvfp4_own_attn_qlens"]
    a = _us(old, iters) * 6
    b = _us(lambda: [ql(q, 1) for _ in range(18)], iters)

    def fresh():  # worst case: a new step object every call (cache miss)
        ql(q.clone(), 1)
    c = _us(fresh, iters) + 17 * b / 18
    print(f"own_attn q-lens per step (6 groups, c=1): old {a:6.1f} us -> cached "
          f"{b:5.1f} us (first-miss incl. {c:5.1f} us)")


def gpu(ms, iters):
    import torch

    from suffix_hybrid import oxide_kernels
    from suffix_hybrid.kernels import nvfp4_lm_head as lh
    from suffix_hybrid.kernels.nvfp4_gemm import PARAMS
    native = oxide_kernels.native()
    oxide_kernels.ensure_loaded(lh.FAMILY, params=PARAMS)
    prepare, _quant, run, capture = lh._device_ops(native)
    dev = torch.device("cuda", torch.cuda.current_device())
    n, k = lh.SHAPES["gemma"]
    st = prepare(lh._synthetic(n, k, dev))
    t0 = time.perf_counter()
    capture(st)
    print(f"capture M=1..{st['max_m']} + bit-identity check: "
          f"{time.perf_counter() - t0:.2f} s, pool "
          f"{torch.cuda.memory_reserved(dev) / 2**20:.0f} MiB reserved")
    sync = torch.cuda.synchronize
    for m in ms:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        assert torch.equal(run(st, x), lh.replay(st, x))
        he = _us(lambda: run(st, x), iters, sync)
        hg = _us(lambda: lh.replay(st, x), iters, sync)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(200):
            lh.replay(st, x)
        e1.record()
        sync()
        print(f"lm_head M={m:2d}: host eager {he:6.1f} us/call -> graph {hg:5.1f} "
              f"us/call (saved {he - hg:6.1f}); device graph "
              f"{e0.elapsed_time(e1) * 5:.1f} us/call")


def embed(iters):
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda", torch.cuda.current_device())
    n, k = 262144, 2816
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    norm = torch.tensor(k ** 0.5, device=dev, dtype=torch.bfloat16)
    ids = torch.zeros(1024, dtype=torch.int32, device=dev)[:1]
    buf = torch.empty(1024, k, device=dev, dtype=torch.bfloat16)
    pin = torch.zeros(1, dtype=torch.bool, pin_memory=True)
    big = torch.randn(8192, 8192, device=dev)
    sync = torch.cuda.synchronize
    ops = {
        "ids.long()": lambda: ids.long(),
        "F.embedding(int64)": lambda: F.embedding(ids.long(), w),
        "F.embedding(int32)": lambda: F.embedding(ids, w),
        "emb * normalizer": lambda: F.embedding(ids, w) * norm,
        "buf[:1] = x": lambda: buf.__setitem__(slice(0, 1), w[:1]),
        "pinned zeros(1)": lambda: torch.zeros(1, dtype=torch.bool, pin_memory=True),
        "mask H2D + masked_fill": lambda: ids.masked_fill(pin.to(dev, non_blocking=True), 0),
        "full text path": lambda: buf.__setitem__(slice(0, 1), F.embedding(
            ids.masked_fill(torch.zeros(1, dtype=torch.bool, pin_memory=True).to(
                dev, non_blocking=True), 0).long(), w) * norm),
    }
    tiny = torch.zeros(1, device=dev)

    def behind(fn, flood, rounds=100):
        """host us of ONE fn() call with a long kernel queued ahead and,
        with flood, ~1500 tiny launches behind it (a FULL-graph decode step
        of a 30-layer MoE is ~1-2k kernel nodes): if the launch queue is
        full, cudaLaunchKernel blocks and the wait lands in fn's frame."""
        tot = 0.0
        for _ in range(rounds):
            sync()
            big @ big
            for _ in range(1500 if flood else 0):
                tiny.add_(0)
            t = time.perf_counter()
            fn()
            tot += time.perf_counter() - t
        sync()
        return tot * 1e6 / rounds

    for name, fn in ops.items():
        idle = _us(fn, iters, sync)
        print(f"embed {name:24s}: host {idle:6.1f} us idle, {behind(fn, False):6.1f} "
              f"behind a long kernel, {behind(fn, True):7.1f} behind a full queue")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cpu", "gpu", "embed"))
    ap.add_argument("--m", type=int, nargs="*", default=[1, 4, 16, 64])
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args(argv)
    {"cpu": lambda: cpu(a.iters), "gpu": lambda: gpu(a.m, a.iters),
     "embed": lambda: embed(a.iters)}[a.mode]()


if __name__ == "__main__":
    main()
