# SPDX-License-Identifier: Apache-2.0
"""FlashInfer autotune cache seed + harvest (SM120 MoE tactic persistence).

vLLM keeps FlashInfer autotune results at
    <root>/flashinfer_autotune_cache/<jit_ws>/<sha256(config_hash)>/autotune_configs.json
with root = VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR or ~/.cache/vllm — the
CONTAINER filesystem, so a pod restart loses the (tens of minutes, SM120 JIT)
trtllm::fused_moe::gemm1/gemm2 sweep and every cold boot pays it again under
the engine-ready timeout. This module closes the loop with no kubectl/exec
access, only the pod log as egress:

  SEED (SUFFIX_FICACHE=seed|both): a lazy import hook wraps vLLM's
    resolve_flashinfer_autotune_file(); when the target file is missing and a
    bundled seed matches the hash directory (/plugins/ficache/seeds/<hash>.json.gz),
    the seed is written into place before warmup runs. Config-hash keyed:
    a mismatched config never applies (the sweep simply runs as today), so a
    seed can never corrupt serving — worst case is today's behaviour.

  HARVEST (SUFFIX_FICACHE=dump|both): a daemon thread rescans the cache roots
    every 30 s; whenever an autotune_configs.json changes (mtime_ns+size), it
    prints ONE log line  SUFFIX_FICACHE_DUMP <hash_dir> <gzip+b64>  that the
    console log reader can collect. The sweep's final save is the payload we
    bake into the next bundle revision.

Both halves are inert unless SUFFIX_FICACHE names them, and every failure
path degrades to stock vLLM behaviour (logged, never fatal): serving never
depended on this module.
"""
from __future__ import annotations

import atexit
import base64
import gzip
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

_TARGET_MODULE = "vllm.model_executor.warmup.flashinfer_autotune_cache"
_DUMP_PREFIX = "SUFFIX_FICACHE_DUMP"
_SEEDS_DIR = Path("/plugins/ficache/seeds")


def _log(msg: str) -> None:
    print(f"suffix ficache: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# SEED: import hook + resolve wrapper
# --------------------------------------------------------------------------

class _SeedModuleImporter(importlib.abc.Loader):
    """Wraps the real loader so our patch applies at first import of the
    warmup cache module — before kernel_warmup's from-import can bind the
    unpatched symbol."""

    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        try:
            _patch_resolve(module)
        except Exception as exc:  # degrade: stock behaviour
            _log(f"seed patch failed (stock cache path stays in use): {exc}")


class _SeedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE or getattr(self, "_reentry", False):
            return None
        self._reentry = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._reentry = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _SeedModuleImporter(spec.loader)
        return spec


def _patch_resolve(module) -> None:
    real = module.resolve_flashinfer_autotune_file

    def resolve_flashinfer_autotune_file(runner):
        path = real(runner)
        try:
            _seed_if_missing(path)
        except Exception as exc:  # seeding is best-effort, never fatal
            _log(f"seed step skipped for {path}: {exc}")
        return path

    setattr(module, "resolve_flashinfer_autotune_file", resolve_flashinfer_autotune_file)
    # kernel_warmup does `from ... import resolve_flashinfer_autotune_file`;
    # if it already imported, rebind its copy too. (Order normally guarantees
    # kernel_warmup imports AFTER this patch, but cover both.)
    kw = sys.modules.get("vllm.model_executor.warmup.kernel_warmup")
    if kw is not None and hasattr(kw, "resolve_flashinfer_autotune_file"):
        setattr(kw, "resolve_flashinfer_autotune_file", resolve_flashinfer_autotune_file)
    _log(f"seed hook installed on {_TARGET_MODULE}")


def _seed_if_missing(path: Path) -> None:
    if path.exists():
        return
    seed = _SEEDS_DIR / f"{path.parent.name}.json.gz"
    if not seed.is_file():
        return
    data = gzip.decompress(seed.read_bytes())
    # parse before writing: a corrupt seed must not poison the cache dir —
    # the sweep simply runs as if no seed existed.
    import json

    try:
        json.loads(data)
    except ValueError as exc:
        _log(f"seed {seed.name} is not valid JSON, ignoring: {exc}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".seedtmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    _log(f"seeded {len(data)} bytes into {path} from {seed.name}")


# --------------------------------------------------------------------------
# HARVEST: periodic scan + single-line log dumps
# --------------------------------------------------------------------------

def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", "")
    if override:
        roots.append(Path(override).expanduser())
    vllm_cache = os.environ.get("VLLM_CACHE_ROOT", "") or str(
        Path.home() / ".cache" / "vllm"
    )
    roots.append(Path(vllm_cache) / "flashinfer_autotune_cache")
    return roots


def encode_dump(path: Path) -> str:
    payload = base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode()
    return f"{_DUMP_PREFIX} {path.parent.name} {payload}"


def _scan_once(seen: dict[Path, tuple[int, int]]) -> None:
    for root in _cache_roots():
        if not root.is_dir():
            continue
        for cfg in root.glob("*/*/autotune_configs.json"):
            try:
                st = cfg.stat()
                key = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
            if seen.get(cfg) == key:
                continue
            seen[cfg] = key
            try:
                line = encode_dump(cfg)
            except OSError as exc:
                _log(f"dump skipped for {cfg}: {exc}")
                continue
            # stdout: matches how the console's log reader collects lines
            # (stderr is reserved for our own _log chatter).
            print(line, flush=True)


def _harvest_loop(interval_s: float) -> None:
    seen: dict[Path, tuple[int, int]] = {}
    while True:
        try:
            _scan_once(seen)
        except Exception as exc:  # never let the thread die on a bad scan
            _log(f"harvest scan failed (retrying): {exc}")
        time.sleep(interval_s)


def install(mode: str) -> None:
    """Called from sitecustomize with SUFFIX_FICACHE in {seed, dump, both}."""
    if mode in ("seed", "both"):
        sys.meta_path.insert(0, _SeedFinder())
        mod = sys.modules.get(_TARGET_MODULE)
        if mod is not None:  # already imported (unexpected) — patch directly
            try:
                _patch_resolve(mod)
            except Exception as exc:
                _log(f"seed patch failed: {exc}")
    if mode in ("dump", "both"):
        t = threading.Thread(
            target=_harvest_loop, args=(30.0,), daemon=True, name="ficache-harvest"
        )
        t.start()
        try:
            atexit.register(_scan_once, {})  # final sweep before exit
        except Exception:
            pass
        _log("harvest thread running (30 s cadence, marker: " + _DUMP_PREFIX + ")")
