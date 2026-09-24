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
- bench_gemma.py: --warm-start/--dump-histories flag parsing, hist-file
  reader (tokens / bare-array / prompt+completion rows), dump format.
"""
import importlib.util
import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suffix_hybrid import warmstart
from suffix_hybrid._native import V2SuffixProposer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench", "gemma-research", "bench_gemma.py")

# exact seam strings (vLLM 0.30.0 pinned tarball: vllm/plugins/__init__.py)
EP_GROUP = "vllm.endpoint_plugins"
EP_NAME = "suffix_hybrid_warmstart"
EP_ENV_VAR = "VLLM_PLUGINS"


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
    # default posture: 404 -> WARNING + cold leg, exit code untouched
    bg = _load_bench()
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", None, None)

    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        out = bg.warm_start_http("http://x", "/v1", [[1]], 5, model="m")
    finally:
        urllib.request.urlopen = orig
    assert out is None
    assert "404" in capsys.readouterr().out


def test_bench_warm_start_http_body_carries_epp_keys(capsys):
    """EPP gateway carrier keys: the POST body must carry "model" + "prompt"
    alongside "sequences", or the gateway rejects it before the FastAPI app
    ever sees it. Captures the POSTed body."""
    bg = _load_bench()
    import urllib.request
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["method"] = req.get_method()

        class R:
            def read(self):
                return b'{"ingested": 1}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R()

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        out = bg.warm_start_http("http://x/", "/v1", [[1, 2], [3]], 5,
                                 model="pool-model-123")
    finally:
        urllib.request.urlopen = orig
    assert out == {"ingested": 1}
    assert seen["url"] == "http://x/debug/warm-start"
    assert seen["method"] == "POST"
    assert seen["body"]["sequences"] == [[1, 2], [3]]
    assert seen["body"]["model"] == "pool-model-123"
    assert isinstance(seen["body"]["prompt"], str)   # carrier passes the EPP


def test_bench_warm_start_http_404_fails_hard_when_requested(capsys):
    # --warm-start mode calls with fail_hard=True: a missing route must
    # NEVER be mistaken for warm success.
    bg = _load_bench()
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", None, None)

    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        with pytest.raises(SystemExit, match="NOT registered"):
            bg.warm_start_http("http://x", "/v1", [[1]], 5, model="m",
                               fail_hard=True)
    finally:
        urllib.request.urlopen = orig


def test_bench_warm_start_http_other_codes_fail_hard_message(capsys):
    # non-404 codes still degrade (not a missing-route situation)
    bg = _load_bench()
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 503, "Service Unavailable", None, None)

    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        out = bg.warm_start_http("http://x", "/v1", [[1]], 5, model="m",
                                 fail_hard=True)
    finally:
        urllib.request.urlopen = orig
    assert out is None
    assert "503" in capsys.readouterr().out