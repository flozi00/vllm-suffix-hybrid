# SPDX-License-Identifier: Apache-2.0
"""WARM-START seam: pre-fill the live suffix cache with corpus histories.

Why (dossier hit-rate-offline-replay.md, H1 CONFIRMED): the SUFFIX-ONLY
ingestion path donates a request's history to the cache only at DEPARTURE
(src/mixer.rs run() gone-row bookkeeping), so a single-pass bench leg can
never hit — cold measured 0.0% on every corpus. Feeding the same legs'
histories BEFORE measurement reproduces the warm regime the offline replay
proved at 99.6% chat hit-rate. This module exposes that lever on the LIVE
pool, over the exact same `SuffixCache.add_sequence` pyo3 seam that
bench/scale_bench.py uses offline.

Mechanics (vLLM v0.30.0, all seams read from that source):

* WORKER RPC: `_probe_worker` is passed to EngineClient.collective_rpc_async
  as a CALLABLE. The API-server → EngineCore hop encodes it with
  serial_utils' enc_hook FunctionType branch (cloudpickle; requires
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 env); the executor's
  _execute_worker_rpc then re-imports it IN THE WORKER PROCESS
  (multiproc_executor.py: `partial(cloudpickle.loads(method), self.worker)`)
  — the same process where wrap_v2's load_model installed
  `propose._suffix_proposer` (the handle wrap_v2 exposes at line
  "propose._suffix_proposer = proposer").

* HTTP ROUTE: vLLM v0.30.0 has a native endpoint-plugin surface
  (vllm.plugins.endpoint_plugins) but it requires a REGISTERED entry
  point; this bundle is mounted as bare files at /plugins (no installed
  distribution), so instead we use the sched_sync.py v3 loader-proxy
  pattern: a sys.meta_path finder wraps vllm.entrypoints.launchers.app's
  exec so `build_app` gains `POST /debug/warm-start` the moment it is
  imported. No vLLM import happens in THIS module at import time — the
  pytest venv has no vllm, and the worker process must never pay for the
  API server's route.

Gating (default OFF must change NOTHING):
  SUFFIX_HYBRID_WARMSTART=1 arms BOTH parts. Without the flag the hook is
  never armed: no finder on sys.meta_path, no route object, no RPC from
  the serving path. sitecustomize arms it in the API-server process only
  (where build_app is imported) — see sitecustomize.py warm-start gate.
"""
import os
import sys

_route_marker = "_suffix_hybrid_warmstart_route"


def _armed() -> bool:
    return os.environ.get("SUFFIX_HYBRID_WARMSTART", "").strip() == "1"


# ---------------------------------------------------------------------------
# WORKER-SIDE (runs inside the engine worker process via collective_rpc)
# ---------------------------------------------------------------------------

def fill(suffix_proposer, sequences):
    """Add each sequence to the suffix cache via the LIVE proposer handle.

    Identical to bench/scale_bench.py's offline priming: the proposer
    exposes `suffix_cache` (V2SuffixProposer #[getter] suffix_cache,
    src/mixer.rs — a Clone handle to the SAME Arc<Mutex<Cache>> the next
    propose call speculates against), and `add_sequence(Vec<i64>)`
    indexes every overlapping n-gram of the donated history. Returns the
    number of sequences submitted for ingestion.
    """
    cache = getattr(suffix_proposer, "suffix_cache", None)
    if cache is None:
        raise RuntimeError(
            "suffix-hybrid warm-start: proposer handle lacks suffix_cache "
            "(V2SuffixProposer getter missing — bundle .so older than "
            "src/mixer.rs?)")
    n = 0
    for seq in sequences:
        cache.add_sequence(list(seq))
        n += 1
    return n


def _probe_worker(worker, sequences):
    """collective_rpc callable body; runs in the WORKER process.

    Reaches the wrapped speculator the same way wrap_v2's install path
    does: worker.model_runner.speculator.propose._suffix_proposer (the
    SUFFIX-ONLY arm's test handle, wrap_v2.py "propose._suffix_proposer
    = proposer"). Returns the ingested count for THIS worker.
    """
    runner = getattr(worker, "model_runner", None)
    speculator = getattr(runner, "speculator", None)
    propose = getattr(speculator, "propose", None)
    proposer = getattr(propose, "_suffix_proposer", None)
    if proposer is None:
        raise RuntimeError(
            "suffix-hybrid warm-start: no _suffix_proposer on the "
            "speculator's propose wrap in this worker (SUFFIX-ONLY wrap "
            "not installed, or k<=0/plain native arm?)")
    return fill(proposer, sequences)


# ---------------------------------------------------------------------------
# API-SERVER ROUTE (imports are lazy: nothing here runs without vLLM +
# fastapi, i.e. only inside the built app of an armed server process)
# ---------------------------------------------------------------------------

async def warm_start(raw_request):
    """POST /debug/warm-start body {"sequences": [[ids...], ...]}.

    Returns {"ingested": N}. There is NO auth beyond the gateway (dev pool
    only, SUFFIX_HYBRID_WARMSTART=1 gated at the process level).
    """
    from fastapi.responses import JSONResponse
    engine = getattr(raw_request.app.state, "engine_client", None)
    if engine is None:
        return JSONResponse(status_code=503,
                            content={"error": "no engine client"})
    try:
        body = await raw_request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})
    sequences = (body or {}).get("sequences")
    if not isinstance(sequences, list) or any(
            not isinstance(s, list) for s in sequences):
        return JSONResponse(status_code=400,
                            content={"error": "sequences must be a list "
                                              "of token-id lists"})
    results = await engine.collective_rpc_async(
        _probe_worker, args=(sequences,))
    # collective_rpc returns one result per rank; TP>1 broadcast mode mixes
    # on rank 0 only but every rank executes the RPC, so every rank has
    # submitted the same sequences to its own pod-local cache — the reply
    # list is per-rank ingestion counts; the dev pool is TP=1 (one entry).
    results = results if isinstance(results, list) else [results]
    ingested = sum(int(r) for r in results)
    print(f"suffix_hybrid WARM-START ingested {ingested}/{len(sequences)} "
          f"sequences", file=sys.stderr, flush=True)
    return {"ingested": ingested}


def attach_warmstart_route(app):
    """Register the warm-start route on the built FastAPI app.

    Called from the wrapped build_app (install_post_import_hook). Uses
    add_api_route with a raw Request handler so the route module needs
    NO pydantic model (keeps the surface minimal and avoids any
    request-schema drift across vLLM versions).
    """
    app.add_api_route("/debug/warm-start", warm_start,
                      methods=["POST"], name="suffix_hybrid_warm_start")
    return app


# ---------------------------------------------------------------------------
# POST-IMPORT HOOK (sched_sync.py v3 loader-proxy pattern)
# ---------------------------------------------------------------------------

def _wrap_launchers_app(module):
    """Called right after vllm.entrypoints.launchers.app exec's."""
    import functools

    original = module.build_app
    if getattr(original, _route_marker, False):
        return

    @functools.wraps(original)
    def build_app(*args, **kwargs):
        app = original(*args, **kwargs)
        try:
            attach_warmstart_route(app)
            print("suffix_hybrid WARM-START: POST /debug/warm-start "
                  "registered (SUFFIX_HYBRID_WARMSTART=1)",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            # Never block serving: the bench driver sees 404 and reports.
            print(f"suffix_hybrid WARM-START route attach FAILED (serving "
                  f"continues without warm-start): {exc}",
                  file=sys.stderr, flush=True)
        return app

    setattr(build_app, _route_marker, True)
    module.build_app = build_app


def install_post_import_hook() -> None:
    """Arm the loader-wrap finder. Idempotent, never raises, NO-OP when
    SUFFIX_HYBRID_WARMSTART is unset — the flag-off process never even
    builds the finder object graph.
    """
    import importlib.util

    if not _armed():
        return
    target = "vllm.entrypoints.launchers.app"

    class _LoaderProxy:
        """Exec the real module exactly once; wrap it the instant it ends."""

        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            return self._real.create_module(spec)

        def exec_module(self, module):
            self._real.exec_module(module)
            try:
                _wrap_launchers_app(module)
            except Exception as exc:
                print(f"suffix_hybrid WARM-START post-import wrap failed: "
                      f"{exc}", file=sys.stderr, flush=True)

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target:
                return None
            # Locate the REAL spec; drop ourselves to avoid re-entry.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            # Re-entrant import of an already-loaded module: stand down.
            if getattr(sys.modules.get(fullname), "__spec__", None) \
                    is not None and fullname in sys.modules:
                return None
            spec.loader = _LoaderProxy(spec.loader)
            return spec

    for f in sys.meta_path:
        if isinstance(f, _Finder):
            return
    sys.meta_path.insert(0, _Finder())
    print("suffix_hybrid WARM-START: build_app post-import hook armed",
          file=sys.stderr, flush=True)