# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture discipline of the DSA indexer lane (owner rules, silicon history).

V4.1 decode runs the indexer INSIDE the full CUDA graph (vLLM's
``eager_break_during_capture`` only breaks it in the piecewise prefill graphs),
so every kernel of the lane is launched from a capture. The rules, enforced
here instead of hoped for:

* No JIT inside a capture. Helion hands Triton runtime ints (sizes, strides,
  page parameters) that Triton specialises on (== 1, % 16 == 0, else), and
  tensor pointers on 16-byte alignment: a launch whose specialisation class was
  never executed eagerly would compile (and load a module) mid-capture.
  ``launch`` records the classes of every eager launch and REFUSES the first
  launch of an unseen class while a capture is open (RuntimeError naming the
  fix: warm before capture, ``dsa_indexer_capture_replay`` boot gate).
* Scratch only from torch's allocator (private pool of the capture, stable
  pointers across replays) or load-time persistent buffers: this lane keeps no
  first-use GPU state; nothing here or in the lane allocates anywhere else.
* No host sync, no cross-program ordering in a captured path: no grid
  barriers, no split-K semaphores, no last-arriving-CTA merges (silicon:
  barriers deadlock, atomic merges fault in capture). The serving kernels
  write disjoint columns of their own range; anything that must combine
  partial results across programs is two-stage (partials, then a separate merge
  kernel). The one atomic cursor in the lane (dsa_group_select) is lab-only.
* Bounded search space: ``search_filter`` clamps every autotune candidate
  (no persistent grids, no warp specialisation, <= 8 warps, <= 4 stages) so LFBO
  never compiles a persistent/specialised variant; ``serving_violation`` refuses
  a shipped config outside the part of the space the filter admits that matters
  for capture safety.
"""
from __future__ import annotations

import torch

MARKER = "[suffix dsa-capture]"
MAX_WARPS = 8
MAX_STAGES = 4
_SEEN: set = set()


def capturing() -> bool:
    """A CUDA-graph capture is open on the current stream (False without CUDA)."""
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _cls(v: int) -> int:
    """Triton's integer specialisation class: 1 (== 1), 16 (% 16 == 0), 0 (other);
    2 for a zero size (Helion treats 0/1 sizes as static)."""
    return 2 if v == 0 else 1 if v == 1 else 16 if v % 16 == 0 else 0


def spec_key(*args) -> tuple:
    """Everything of a launch's arguments Helion / Triton specialise a binary on:
    per tensor the dtype, rank, size / stride classes, 16-byte pointer alignment
    and the int32/int64 index class (numel > 2^31); per int its class."""
    key = []
    for a in args:
        if isinstance(a, torch.Tensor):
            key.append((str(a.dtype), a.dim(), tuple(_cls(d) for d in a.shape),
                        tuple(_cls(s) for s in a.stride()), a.data_ptr() % 16 == 0,
                        a.numel() > 2 ** 31 - 1))
        elif isinstance(a, bool):
            key.append(a)
        elif isinstance(a, int):
            key.append(_cls(a))
        elif isinstance(a, str):
            key.append(a)
        else:
            key.append(type(a).__name__)
    return tuple(key)


def seen(kern, *args) -> bool:
    return (id(kern), spec_key(*args)) in _SEEN


def launch(kern, *args):
    """``kern(*args)`` under the capture contract (module doc)."""
    key = (id(kern), spec_key(*args))
    if key not in _SEEN and capturing():
        raise RuntimeError(
            f"{MARKER} the first launch of {getattr(kern, '__name__', type(kern).__name__)} with this "
            "argument specialisation (sizes / strides / alignment classes) happens inside a CUDA-graph "
            "capture, where Triton would JIT-compile; warm every shape class eagerly before the capture "
            "(dsa_indexer.warm_all / the dsa_indexer_capture_replay boot gate)")
    out = kern(*args)
    _SEEN.add(key)
    return out


def reset() -> None:
    _SEEN.clear()


def serving_violation(cfg: dict | None) -> str | None:
    """Why a shipped Helion config may not serve this lane (None: it may)."""
    if not cfg:
        return None
    if str(cfg.get("pid_type", "flat")).startswith("persistent"):
        return f"pid_type {cfg['pid_type']!r} (persistent grids need co-resident programs: deadlock risk)"
    if any(x is True for x in cfg.get("range_warp_specializes", [])):
        return "warp-specialised loop (not admitted in this lane)"
    if int(cfg.get("num_warps", 4)) > 32:
        return f"num_warps {cfg['num_warps']}"
    return None


def search_filter(config):
    """Helion ``autotune_config_filter``: every autotune candidate is clamped to
    the bounded space before it is compiled or timed (module doc)."""
    import helion
    c = dict(config.config if hasattr(config, "config") else config)
    if str(c.get("pid_type", "flat")).startswith("persistent"):
        c["pid_type"] = "flat"
    if "range_warp_specializes" in c:
        c["range_warp_specializes"] = [None if x is None else False for x in c["range_warp_specializes"]]
    if "num_warps" in c:
        c["num_warps"] = min(int(c["num_warps"]), MAX_WARPS)
    if "num_stages" in c:
        c["num_stages"] = min(int(c["num_stages"]), MAX_STAGES)
    return helion.Config(**c)


# helion.kernel(...) kwargs every body of the lane's KERNELS carries (lab + autotune runner)
SEARCH_SETTINGS = {"autotune_config_filter": search_filter}
