# SPDX-License-Identifier: Apache-2.0
"""Compressed all-reduce prototype (SUFFIX_NCCL_QAR=int8|fp8, default OFF).

bf16 all-reduce as quantized reduce-scatter + all-gather over the default
PyNccl communicator, ~half the bytes of a bf16 ring (1 B + 4/128 B scale per
element each way instead of 2 B):
  (a) per-128-element block quantization (int8 amax/127 or fp8-e4m3 amax/448,
      fp32 scale), rank r's chunk j packed as one uint8 row [q | scales],
  (b) all-to-all of the rows (grouped ncclSend/ncclRecv),
  (c) dequantize the W received rows of its own chunk, fp32 sum in rank order,
  (d) requantize that partial sum and all-gather it,
  (e) dequantize everything to the input dtype.
Every rank dequantizes the SAME gathered bytes -> bit-identical outputs on all
ranks, deterministic run to run (fixed shapes, no atomics). Static shapes and
torch-allocator buffers only -> CUDA-graph safe (during capture the graph's
private pool serves the temporaries).

QUALITY RISK: error per element <= sum_r amax(block_r)/254 + amax(block_sum)/254
(int8; see error_bound) -- i.e. relative to the BLOCK max, not to the element.
Heavy-tailed activations (outlier channels) lose small values next to an
outlier. Never enable without the pod bench (boot gate allreduce_qar_bench)
AND a GSM8K / tool-call quality gate on the target model.

    python -m suffix_hybrid.nccl_qar   # CPU accuracy report (random + heavy-tailed)
"""
from __future__ import annotations

import torch

BLOCK = 128
QMAX = {"int8": 127.0, "fp8": 448.0}


def _qdtype(mode: str):
    return torch.int8 if mode == "int8" else torch.float8_e4m3fn


def _pack(x: torch.Tensor, mode: str) -> torch.Tensor:
    """fp32 [R, C] (C % BLOCK == 0) -> uint8 [R, C + 4 * C // BLOCK]."""
    r, c = x.shape
    blk = x.view(r, c // BLOCK, BLOCK)
    scale = blk.abs().amax(-1, keepdim=True).clamp_min(1e-30) / QMAX[mode]
    v = blk / scale
    q = (v.round().clamp(-127, 127) if mode == "int8" else v.clamp(-448, 448)).to(_qdtype(mode))
    return torch.cat([q.view(r, c).view(torch.uint8),
                      scale.view(r, -1).contiguous().view(torch.uint8)], 1)


def _unpack(buf: torch.Tensor, c: int, mode: str) -> torch.Tensor:
    """uint8 [R, L] -> fp32 [R, C]."""
    r = buf.shape[0]
    q = buf[:, :c].contiguous().view(_qdtype(mode)).float().view(r, c // BLOCK, BLOCK)
    scale = buf[:, c:].contiguous().view(torch.float32).view(r, c // BLOCK, 1)
    return (q * scale).view(r, c)


def _split(x: torch.Tensor, world: int) -> tuple[torch.Tensor, int]:
    flat = x.reshape(-1).float()
    n = flat.numel()
    pad = -n % (world * BLOCK)
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.view(world, -1), n


def qar_all_reduce(x: torch.Tensor, world: int, a2a, ag, mode: str = "int8") -> torch.Tensor:
    """a2a(uint8 [W, L]) -> [W, L] (row j from rank j); ag(uint8 [1, L]) -> [W, L]."""
    rows, n = _split(x, world)
    c = rows.shape[1]
    part = _unpack(a2a(_pack(rows, mode)), c, mode).sum(0, keepdim=True)
    out = _unpack(ag(_pack(part, mode)), c, mode)
    return out.view(-1)[:n].to(x.dtype).view(x.shape)


def simulate(xs: list[torch.Tensor], mode: str = "int8") -> torch.Tensor:
    """Single-process reference of qar_all_reduce over len(xs) ranks."""
    w = len(xs)
    packed = [_pack(_split(x, w)[0], mode) for x in xs]
    c = _split(xs[0], w)[0].shape[1]
    parts = [_pack(_unpack(torch.stack([p[j] for p in packed]), c, mode).sum(0, keepdim=True), mode)
             for j in range(w)]
    out = _unpack(torch.cat(parts), c, mode)
    return out.view(-1)[:xs[0].numel()].to(xs[0].dtype).view(xs[0].shape)


def error_bound(xs: list[torch.Tensor], mode: str = "int8") -> torch.Tensor:
    """Per-element analytic bound on |qar - exact fp32 sum| (flattened, fp32)."""
    w = len(xs)
    rows = [_split(x, w)[0].reshape(-1, BLOCK) for x in xs]
    exact = torch.stack(rows).sum(0)
    if mode == "int8":
        e1 = sum(r.abs().amax(-1, keepdim=True) / 254 for r in rows)
        e2 = (exact.abs() + e1).amax(-1, keepdim=True) / 254
    else:  # e4m3: 3 mantissa bits (rel 2^-4), subnormal step 2^-9 of scale
        e1 = sum(torch.maximum(r.abs() / 16, r.abs().amax(-1, keepdim=True) / 448 / 1024) for r in rows)
        e2 = torch.maximum((exact.abs() + e1) / 16, (exact.abs() + e1).amax(-1, keepdim=True) / 448 / 1024)
    out_rel = 2.0 ** -8 if xs[0].dtype == torch.bfloat16 else 2.0 ** -11
    bound = e1 + e2 + (exact.abs() + e1 + e2) * out_rel + 1e-6 * exact.abs().amax()
    return bound.view(-1)[:xs[0].numel()]


class PyncclQar:
    """all_reduce(x) over a vLLM PyNcclCommunicator (the default, auto-algo one)."""

    def __init__(self, comm, mode: str):
        assert mode in QMAX, mode
        self.comm, self.mode = comm, mode

    def _a2a(self, send: torch.Tensor) -> torch.Tensor:
        recv = torch.empty_like(send)
        me = self.comm.rank
        recv[me].copy_(send[me])
        self.comm.group_start()
        for p in range(self.comm.world_size):
            if p != me:
                self.comm.send(send[p], p)
                self.comm.recv(recv[p], p)
        self.comm.group_end()
        return recv

    def _ag(self, mine: torch.Tensor) -> torch.Tensor:
        out = mine.new_empty(self.comm.world_size, mine.shape[1])
        self.comm.all_gather(out, mine)
        return out

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return qar_all_reduce(x, self.comm.world_size, self._a2a, self._ag, self.mode)


def accuracy_report(world: int = 8, n: int = 6144 * 48, seed: int = 0) -> list[str]:
    """Relative error vs exact fp32 sum; bf16 ring-style sum as the baseline."""
    data = {
        "normal": lambda: torch.randn(n),
        "student-t(3)": lambda: torch.distributions.StudentT(3.0).sample((n,)),
        "outlier-ch(x100)": lambda: torch.randn(n) * torch.where(
            torch.arange(n) % 6144 < 8, 100.0, 1.0),
    }
    lines = []
    for name, gen in data.items():
        torch.manual_seed(seed)
        xs = [gen().to(torch.bfloat16) for _ in range(world)]
        exact = torch.stack([x.float() for x in xs]).sum(0)
        ring = xs[0].clone()
        for x in xs[1:]:
            ring = (ring.float() + x.float()).to(torch.bfloat16)
        cols = [f"{name:16s}"]
        for label, y in [("bf16-seq", ring)] + [(m, simulate(xs, m)) for m in QMAX]:
            d = (y.float() - exact)
            cols.append(f"{label} rel-l2 {float(d.norm() / exact.norm()):.2e} "
                        f"max/amax {float(d.abs().max() / exact.abs().max()):.2e}")
            if label in QMAX:
                ratio = float((d.abs() / error_bound(xs, label)).max())
                cols[-1] += f" err/bound {ratio:.2f}"
        lines.append(" | ".join(cols))
    return lines


if __name__ == "__main__":
    for line in accuracy_report():
        print(f"[suffix nccl-qar] {line}", flush=True)
