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

* HTTP ROUTE: attached through vLLM v0.30.0's NATIVE endpoint-plugin seam
  (vllm.plugins.endpoint_plugins) by the repo-root module
  suffix_hybrid_warmstart_ep.py (EndpointPlugin protocol: name /
  required_tasks=None / attach_router(app) / init_state(engine_client,
  state, args)); it is discovered via the dist-info dir
  suffix_hybrid_warmstart_ep-1.0.dist-info/ entry point
  [vllm.endpoint_plugins] suffix_hybrid_warmstart =
  suffix_hybrid_warmstart_ep:register. The loader (load_endpoint_plugins in
  vllm/plugins/__init__.py) is STRICT opt-in: it is not called at all unless
  env VLLM_PLUGINS names the plugin, and the plugin's attach_router further
  self-gates on SUFFIX_HYBRID_WARMSTART=1. The old sys.meta_path
  loader-proxy build_app wrap was REMOVED: on the live pod it armed 3x but
  never attached (an earlier importer had already put
  vllm.entrypoints.launchers.app in sys.modules, short-circuiting find_spec).
  No vLLM import happens in THIS module at import time — the pytest venv has
  no vllm, and the worker process must never pay for the API server's route.

* EPP GATEWAY CARRIER KEYS: the pool's gateway (EPP) body-validates every
  POST before forwarding it to the FastAPI app against its completions
  validator; a bare {"sequences": ...} body is REJECTED ("must have prompt
  field"). A body carrying "model" (the pool's served model) and "prompt"
  (a carrier string) alongside "sequences" PASSES and is forwarded with the
  original path intact. The bench driver (bench_gemma.py --warm-start)
  merges those carrier keys into its POST body; the handler reads ONLY
  "sequences" and ignores the extras.

Gating (default OFF must change NOTHING):
  TWO keys must be on for the route to exist: env VLLM_PLUGINS (naming
  suffix_hybrid_warmstart; unset => vLLM never even runs the endpoint-plugin
  loader, so the plugin module is never imported by the server) AND
  SUFFIX_HYBRID_WARMSTART=1 (the plugin's attach_router self-gate/belt).
  Without them: no route object, no RPC from the serving path — and
  sitecustomize no longer touches warm-start at all.
"""
import sys


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
# API-SERVER ROUTE — attached by the suffix_hybrid_warmstart_ep
# EndpointPlugin (vllm.endpoint_plugins seam); nothing here runs without vLLM
# + fastapi, i.e. only inside the built app of an armed server process.

# FastAPI injects the Request object ONLY for a parameter annotated with a
# Request subclass — a bare name (any name) is treated as a query param and
# yields 422. The annotation must therefore sit in the SIGNATURE; fastapi is
# imported eagerly-if-present (the pytest venv may lack it — then the
# annotation is None and the function is simply never routed).
try:
    from fastapi import Request as _FastAPIRequest
except Exception:                # pragma: no cover - venv without fastapi
    _FastAPIRequest = None


async def warm_start(raw_request: _FastAPIRequest):
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
    """Register the warm-start route on a FastAPI app.

    Called by the suffix_hybrid_warmstart_ep EndpointPlugin's attach_router
    (vllm.endpoint_plugins seam). Uses add_api_route with a raw Request
    handler so the route module needs NO pydantic model (keeps the surface
    minimal and avoids any request-schema drift across vLLM versions).
    """
    app.add_api_route("/debug/warm-start", warm_start,
                      methods=["POST"], name="suffix_hybrid_warm_start")
    return app