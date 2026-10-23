# SPDX-License-Identifier: Apache-2.0
"""Helion ports of the plugin's cuda-oxide kernels — shared runtime.

Helion -> Triton -> Triton's bundled ptxas -> SASS: loads on the driver-580
nodes with no PTX JIT, like every vLLM Triton kernel. Serving NEVER autotunes:
configs are tuned on idle GPUs by the inference-console autotune lab and
shipped here, in exactly the files its runner writes:

    suffix_hybrid/configs/helion/<kernel>/<platform>.json = [{"key": case, "config": {...}}]

  <kernel>    a KERNELS entry name of a port module (e.g. nvfp4_moe_fc1)
  <platform>  vLLM's canonical GPU name (vllm.kernels.helion.utils), e.g.
              nvidia_geforce_rtx_5090
  case        the flat case dict (a case_key string is accepted too)

Every port module (suffix_hybrid/helion_kernels/<family>.py) is ALSO an
autotune-lab kernel module: a module-level ``KERNELS`` list in the console's
ABI (inference-console api/autotune_kernels.py docstring) whose baseline is
the oxide kernel it replaces. The lab mounts this plugin and runs a
two-line shim: ``sys.path.append("/plugins"); from suffix_hybrid.helion_kernels.<family> import KERNELS``.

Helion itself is vendored in the bundle under /plugins/_vendor (lowest
sys.path priority: an image that ships Helion wins).
"""
from __future__ import annotations

import json
import os
import re
import sys

MARKER = "[suffix helion]"
CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "helion")
BUCKET = "numtokens"


def log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


def ensure_vendor() -> None:
    """Make the bundle's vendored Helion importable (appended: image packages win)."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "_vendor")
    if os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)
    # Helion's default LFBO search needs scikit-learn; serving never searches,
    # and an in-pod tune must not die on the import. The env still wins.
    os.environ.setdefault("HELION_AUTOTUNER", "PatternSearch")


def case_key(case: dict) -> str:
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def platform(device: int = 0) -> str:
    """The autotune runner's platform name (same function, same fallback)."""
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name
        return get_canonical_gpu_name(device)
    except Exception:  # noqa: BLE001 — older vLLM: the same normalisation by hand
        import torch
        return re.sub(r"[\s/-]+", "_", torch.cuda.get_device_name(device).lower())


_TABLES: dict = {}


def table(kernel: str, plat: str | None = None) -> dict:
    """{case_key: config} shipped for `kernel` on this GPU ({} if none)."""
    plat = plat or platform()
    key = (kernel, plat)
    if key not in _TABLES:
        path = os.path.join(CONFIG_ROOT, kernel, plat + ".json")
        rows = []
        if os.path.isfile(path):
            with open(path) as f:
                rows = json.load(f)
        # The runner writes "key" as the case object (vLLM ConfigManager form).
        _TABLES[key] = {r["key"] if isinstance(r["key"], str) else case_key(r["key"]): r["config"]
                        for r in rows}
    return _TABLES[key]


def bucket(m: int, buckets) -> int:
    """The tuned case a serving M runs as: the smallest bucket >= m, else the
    largest (a port's cases() lists its buckets; the lab tunes exactly those)."""
    above = [b for b in buckets if b >= m]
    return min(above) if above else max(buckets)


def config(kernel: str, case: dict, plat: str | None = None) -> dict | None:
    """The shipped config for exactly `case` (None: not tuned, or the oxide
    kernel won there — the config pull ships only lab WINS)."""
    return table(kernel, plat).get(case_key(case))


def backend(env: str) -> str:
    """A lane's backend knob: oxide (default) | helion (every shape; a missing
    config is a startup error) | auto (Helion where a shipped config exists =
    where it beat oxide in the lab, oxide elsewhere)."""
    value = os.environ.get(env, "oxide").strip().lower() or "oxide"
    if value not in ("oxide", "helion", "auto"):
        raise RuntimeError(f"{env}={value!r}: expected oxide | helion | auto")
    return value


_COMPILED: dict = {}


def compiled(body, kernel: str, case: dict, **settings):
    """The serving Helion kernel for exactly `case` (already bucketed): `body`
    bound to its shipped config, cached per (kernel, case). KeyError when none
    is shipped. Dynamic shapes unless the caller passes static_shapes=True
    (one tuned config serves every M of its bucket)."""
    cfg = config(kernel, case)
    if cfg is None:
        raise KeyError(f"{MARKER} no tuned {kernel} config for {case_key(case)} on {platform()} "
                       f"(tune it in the autotune lab, ship configs/helion/{kernel}/<platform>.json)")
    key = (kernel, case_key(case))
    kern = _COMPILED.get(key)
    if kern is None:
        ensure_vendor()
        import helion
        settings.setdefault("static_shapes", False)
        kern = _COMPILED[key] = helion.kernel(body, config=helion.Config(**cfg), **settings)
    return kern
