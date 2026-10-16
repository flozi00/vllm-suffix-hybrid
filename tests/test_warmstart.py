# SPDX-License-Identifier: Apache-2.0
"""WARM-START seam tests (offline; no vLLM, no server, no GPU).

Covers (dossier hit-rate-offline-replay.md → live lever):
- V2SuffixProposer.suffix_cache getter returns the LIVE cache handle:
  add_sequence through it primes the exact cache the next propose call
  speculates against (the bench/scale_bench.py path, in-process).
- _probe_worker reaches the proposer through the worker object graph
  (worker.model_runner.speculator.propose._suffix_proposer) and ingests.
- warm_start HTTP handler shapes (503 no-engine, 400 malformed, TP-rank
  summing) — see test_warmstart_ep.py for the vllm.endpoint_plugins seam
  tests (route attach via the EndpointPlugin, VLLM_PLUGINS loader gating).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suffix_hybrid import warmstart
from suffix_hybrid._native import V2SuffixProposer

# exact seam strings (vLLM 0.30.0 pinned tarball: vllm/plugins/__init__.py)
EP_GROUP = "vllm.endpoint_plugins"
EP_NAME = "suffix_hybrid_warmstart"
EP_ENV_VAR = "VLLM_PLUGINS"


def _propose(p, seq, rid="a"):
    tok = np.array([seq], dtype=np.int32)
    return p.propose_suffix_only(
        request_ids=[rid], indices=[0],
        totals=np.array([len(seq)], dtype=np.int64), tokens=tok)


# ---------------------------------------------------------------- live handle

def test_suffix_cache_getter_primes_the_live_proposer():
    p = V2SuffixProposer(4, 512)
    _propose(p, [1, 2, 3, 4, 5, 6, 7, 8])          # cold: no hit possible
    assert p.get_stats()["hits"] == 0
    # warm-start via the EXACT seam the endpoint uses
    n = warmstart.fill(p, [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    assert n == 1
    packed, widths = _propose(p, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert widths[0] > 0            # the next propose HIT the warm cache
    assert p.get_stats()["hits"] == 1
    assert p.suffix_cache.stats()["num_sequences"] == 1


def test_suffix_cache_getter_shares_state_not_a_copy():
    p = V2SuffixProposer(4, 512)
    a = p.suffix_cache
    b = p.suffix_cache
    a.add_sequence([5, 6, 7, 8, 9, 10, 11, 12])
    # same Arc<Mutex<Cache>>: both handles see the donation
    assert a.stats()["num_sequences"] == 1
    assert b.stats()["num_sequences"] == 1
    assert p.suffix_cache.stats()["cached_tokens"] == 8


def test_fill_rejects_missing_handle_loudly():
    with pytest.raises(RuntimeError, match="suffix_cache"):
        warmstart.fill(NS(), [[1, 2]])


# ---------------------------------------------------------------- probe path

def test_probe_worker_walks_worker_graph():
    p = V2SuffixProposer(4, 512)
    fake_worker = NS(model_runner=NS(speculator=NS(propose=NS(
        _suffix_proposer=p))))
    seqs = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    assert warmstart._probe_worker(fake_worker, seqs) == 1
    assert p.suffix_cache.stats()["num_sequences"] == 1


def test_probe_worker_fails_loud_without_wrap():
    fake_worker = NS(model_runner=NS(speculator=NS(propose=NS())))
    with pytest.raises(RuntimeError, match="_suffix_proposer"):
        warmstart._probe_worker(fake_worker, [[1, 2]])


# ---------------------------------------------------------------- env gating

def test_handler_shapes():
    """Pure-logic assertions on the route handler without fastapi installed:
    - rejects when engine_client missing (503 path via JSONResponse mock)
    - rejects malformed bodies.
    Uses a duck-typed request. NOTE: needs fastapi responses import; if
    fastapi is absent (test venv), skip — the shape is pinned by the
    live test on the pool."""
    pytest.importorskip("fastapi")

    class Req:
        def __init__(self, state, body):
            self.app = NS(state=state)
            self._body = body

        async def json(self):
            if isinstance(self._body, Exception):
                raise self._body
            return self._body

    import asyncio

    # no engine client -> 503
    out = asyncio.run(warmstart.warm_start(Req(NS(engine_client=None), {})))
    assert out.status_code == 503

    # malformed sequences -> 400
    calls = []

    class FakeEngine:
        async def collective_rpc_async(self, *a, **k):
            calls.append((a, k))
            return [0]

    state = NS(engine_client=FakeEngine())
    out = asyncio.run(warmstart.warm_start(
        Req(state, {"sequences": "nope"})))
    assert out.status_code == 400
    assert not calls                      # never reached the engine


def test_handler_ingests_and_sums_tp_ranks():
    pytest.importorskip("fastapi")
    import asyncio

    seen = {}

    class FakeEngine:
        async def collective_rpc_async(self, method, args=(), **kw):
            seen["method"] = method
            seen["args"] = args
            return [2, 2]                # TP=2 broadcast: both ranks report

    class Req:
        app = NS(state=NS(engine_client=FakeEngine()))

        async def json(self):
            return {"sequences": [[1, 2, 3], [4, 5, 6]]}

    out = asyncio.run(warmstart.warm_start(Req()))
    assert seen["method"] is warmstart._probe_worker
    assert out == {"ingested": 4}
