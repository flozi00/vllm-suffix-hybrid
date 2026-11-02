# SPDX-License-Identifier: Apache-2.0
"""GPU oracle for SUFFIX_ROCM_QSA_SPARSE_SKIP (boot gate qsa_sparse_bench).

The gate (suffix_hybrid/rocm_patches.py) wraps the tile body of vLLM's AMD
_qsa_sparse_paged_gqa_splitk_kernel in `if tl.max(logical_token) >= 0`. The
selection is token_topk + compress - 1 = 2051 slots per row, padded with -1
for every context shorter than the budget, and stock runs the page lookup, the
K/V loads, both dots and the softmax update on all of them. A tile with no
valid slot is a no-op there (scores -1e20 -> probabilities 0, alpha
exp2(0) = 1), so skipping it is exact.

    python -m suffix_hybrid.kernels.qsa_sparse_rocm   # stock vs patched: bitwise + us/call
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import torch

MARK = "[suffix qsa-sparse]"
TARGET = "vllm.models.qwen4_exp.amd.ops.qsa"


def _patched_module():
    """A second copy of the QSA ops module with the gate's rewrite applied."""
    from suffix_hybrid import rocm_patches as rp

    spec = importlib.util.find_spec(TARGET)
    src = rp.patch_source(rp.PATCHES["SUFFIX_ROCM_QSA_SPARSE_SKIP"], spec.loader.get_source(TARGET))
    mod = types.ModuleType(TARGET + "_suffix_sparse_skip")
    mod.__file__ = spec.origin
    mod.__package__ = TARGET.rsplit(".", 1)[0]
    exec(compile(src, spec.origin, "exec"), mod.__dict__)
    return mod


def _time_us(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def main() -> int:
    stock = importlib.import_module(TARGET)
    new = _patched_module()
    # qwen3.8-flash QSA layer: 24 query heads, 2 KV heads x 256, selection 2048 + 4 - 1.
    dev, heads, kv_heads, dim, page, width, pages, sel = "cuda", 24, 2, 256, 64, 4096, 8192, 2051
    torch.manual_seed(0)
    k_cache = torch.randn(pages, page, kv_heads, dim, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(pages, page, kv_heads, dim, device=dev, dtype=torch.bfloat16)
    failed = False
    for name, rows, reqs, lo, hi in (("c1 x mtp5", 5, 1, 120, 480), ("c8 x mtp5", 40, 8, 100, 600),
                                     ("c32 x mtp5", 160, 32, 100, 600),
                                     ("long ctx (full budget)", 16, 4, 3000, 3000),
                                     ("prefill chunk", 384, 1, 64, 2051)):
        valid = torch.randint(lo, hi + 1, (rows,), device=dev).clamp(max=sel)
        cols = torch.arange(sel, device=dev)[None, :]
        tokens = torch.randint(0, width * page, (rows, sel), device=dev)
        idx = torch.where(cols < valid[:, None], tokens, -1).to(torch.int32)
        table = torch.randint(0, pages, (reqs, width), device=dev, dtype=torch.int32)
        tok2req = (torch.arange(rows, device=dev) * reqs // rows).to(torch.int32)
        tok2req[-2:] = -1  # graph-padding rows
        q = torch.randn(rows, heads, dim, device=dev, dtype=torch.bfloat16)
        args = (q, k_cache, v_cache, idx, table, tok2req)
        ref = stock.qsa_sparse_paged_attention(*args)
        out = new.qsa_sparse_paged_attention(*args)
        ok = torch.equal(ref, out)
        failed |= not ok
        print(f"{MARK} {name}: rows {rows} valid {int(valid.min())}..{int(valid.max())} of {sel} "
              f"{'MATCH (bitwise)' if ok else 'MISMATCH'} max abs diff "
              f"{(ref.float() - out.float()).abs().max().item():.2e} | stock "
              f"{_time_us(lambda: stock.qsa_sparse_paged_attention(*args)):.1f} us -> skip "
              f"{_time_us(lambda: new.qsa_sparse_paged_attention(*args)):.1f} us", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
