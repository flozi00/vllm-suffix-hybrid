# SPDX-License-Identifier: Apache-2.0
"""WARM-START seam tests (offline; no vLLM, no server, no GPU).

Covers (dossier hit-rate-offline-replay.md → live lever):
- V2SuffixProposer.suffix_cache getter returns the LIVE cache handle:
  add_sequence through it primes the exact cache the next propose call
  speculates against (the bench/scale_bench.py path, in-process).
- _probe_worker reaches the proposer through the worker object graph
  (worker.model_runner.speculator.propose._suffix_proposer) and ingests.
- suffix_hybrid.warmstart.install_post_import_hook is STRICTLY env-gated:
  no finder, no route, nothing armed with SUFFIX_HYBRID_WARMSTART unset.
- The route attaches to a FastAPI-like app only through the wrapped
  build_app; flag-off build_app is untouched.
- bench_gemma.py: --warm-start/--dump-histories flag parsing, hist-file
  reader (tokens / bare-array / prompt+completion rows), dump format.
"""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suffix_hybrid import warmstart
from suffix_hybrid._native import V2SuffixProposer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench", "gemma-research", "bench_gemma.py")


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_gemma_warm", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

@pytest.fixture
def warm_env(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_WARMSTART", "1")
    yield
    # _Finder removal handled by test; env restored by monkeypatch


class _FakeFinder:
    pass


def test_hook_inert_when_flag_unset(monkeypatch):
    monkeypatch.delenv("SUFFIX_HYBRID_WARMSTART", raising=False)
    before = list(sys.meta_path)
    warmstart.install_post_import_hook()
    assert sys.meta_path == before            # nothing armed at all


def test_hook_arms_finder_when_flag_set(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_WARMSTART", "1")
    warmstart.install_post_import_hook()
    finder = sys.meta_path[0]
    try:
        assert finder.__class__.__name__ == "_Finder"
        # idempotent: second install adds nothing
        warmstart.install_post_import_hook()
        n_finders = sum(1 for f in sys.meta_path
                        if f.__class__ is finder.__class__)
        assert n_finders == 1
    finally:
        sys.meta_path.remove(finder)


def test_wrap_marks_build_app_and_attaches_route(monkeypatch):
    monkeypatch.setenv("SUFFIX_HYBRID_WARMSTART", "1")

    class FakeApp:
        def __init__(self):
            self.routes = []

        def add_api_route(self, path, endpoint, methods=None, name=None):
            self.routes.append((path, endpoint, tuple(methods or ()), name))

    calls = {"orig": 0}

    def build_app(*a, **k):
        calls["orig"] += 1
        return FakeApp()

    module = NS(build_app=build_app)
    warmstart._wrap_launchers_app(module)
    # wrapped but not called yet; original untouched behavior on call
    app = module.build_app()
    assert calls["orig"] == 1
    assert len(app.routes) == 1
    path, endpoint, methods, name = app.routes[0]
    assert path == "/debug/warm-start"
    assert "POST" in methods
    assert endpoint is warmstart.warm_start


def test_wrapped_build_app_survives_route_failure(monkeypatch, capsys):
    """Route attach failure must NEVER block serving."""
    def boom(app):
        raise RuntimeError("fastapi exploded")
    monkeypatch.setattr(warmstart, "attach_warmstart_route", boom)

    def build_app():
        return "APP"

    module = NS(build_app=build_app)
    warmstart._wrap_launchers_app(module)
    assert module.build_app() == "APP"      # serving continues
    assert "FAILED" in capsys.readouterr().err


def test_warm_start_handler_shapes():
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


# ---------------------------------------------------------------- bench flags

def test_bench_hist_reader_and_flags(tmp_path):
    bg = _load_bench()
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({"tokens": [1, 2, 3]}) + "\n"
                 + json.dumps([9, 8, 7]) + "\n"
                 + json.dumps({"prompt": [1], "completion": [2, 3]}) + "\n")
    assert bg.hist_file_sequences(str(p)) == [[1, 2, 3], [9, 8, 7], [1, 2, 3]]
    # flags parse
    rc = bg.main(["--dry-run", "--ledger", "/dev/null", "--warm-start",
                  str(p), "--dump-histories", str(tmp_path / "o.jsonl")])
    assert rc == 0


def test_bench_hist_reader_rejects_garbage(tmp_path):
    bg = _load_bench()
    p = tmp_path / "bad.jsonl"
    p.write_text("{nope\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        bg.hist_file_sequences(str(p))


def test_bench_payload_gains_return_token_ids_only_when_dumping(tmp_path):
    bg = _load_bench()
    plain = NS(dump_histories=None, ignore_eos=False, model="m")
    dumping = NS(dump_histories="out.jsonl", ignore_eos=False, model="m")
    msgs = [{"role": "user", "content": "hi"}]
    assert "return_token_ids" not in bg.build_payload(plain, msgs, 8)
    assert bg.build_payload(dumping, msgs, 8)["return_token_ids"] is True


def test_bench_dump_histories_format(tmp_path):
    bg = _load_bench()
    args = NS(corpus="reuse")
    out = tmp_path / "hist.jsonl"
    results = [
        {"prompt_ids": [1, 2, 3], "token_ids": [4, 5]},
        {"prompt_ids": None, "token_ids": [6]},   # promptless row
        {"prompt_ids": None, "token_ids": []},    # empty -> skipped
        None,                                      # failed request
    ]
    n = bg.dump_histories(str(out), args, results, 8, 0)
    assert n == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["tokens"] == [1, 2, 3, 4, 5]
    assert rows[0]["corpus"] == "reuse"
    assert rows[1]["tokens"] == [6]


def test_bench_warm_start_http_404_degrades(capsys):
    bg = _load_bench()
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", None, None)

    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        out = bg.warm_start_http("http://x", "/v1", [[1]], 5)
    finally:
        urllib.request.urlopen = orig
    assert out is None
    assert "404" in capsys.readouterr().out