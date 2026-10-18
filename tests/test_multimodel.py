"""SUFFIX_MULTIMODEL gates (dossiers/multimodel-serving.md): launcher config
refusals on CPU, and the real Rust router against two fake vLLM children."""
import http.client
import json
import socket
import statistics
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import suffix_hybrid
except ModuleNotFoundError as error:
    if error.name != "suffix_hybrid":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suffix_hybrid import multimodel

try:
    from suffix_hybrid import _native
except ImportError:
    _native = None
ROUTER = getattr(_native, "multimodel_serve", None) is not None


def model(name, port, **changes):
    return {"served_as": name, "port": port, "gpu_memory_utilization": 0.8,
            "hostnames": [f"{name}.test"], "argv": [f"/models/{name}", "--max-model-len", "4096"],
            "env": {"SUFFIX_PPLX_DECIDER": "1"}} | changes


def config(**changes):
    return {"gpu_budget": 0.92, "models": [model("a", 8101), model("b", 8102)]} | changes


class ConfigTests(unittest.TestCase):
    def test_accepts_contract_example_and_fills_defaults(self):
        value = multimodel.validate(config())
        self.assertEqual((value["listen"], value["sleep_level"], value["slice_ms"]),
                         ("0.0.0.0:8000", 1, 2000))
        command = multimodel.child_command(value["models"][0])
        self.assertEqual(command[1:5], ["-m", "vllm.entrypoints.cli.main", "serve", "/models/a"])
        self.assertEqual(command[-9:], ["--host", "127.0.0.1", "--port", "8101",
                                        "--served-model-name", "a",
                                        "--gpu-memory-utilization", "0.8", "--enable-sleep-mode"])
        env = multimodel.child_env(value["models"][0] | {"env": {"VLLM_SERVER_DEV_MODE": "0", "X": "1"}},
                                   {"POD": "1", "X": "0"})
        self.assertEqual(env, {"POD": "1", "X": "1", "VLLM_SERVER_DEV_MODE": "1"})

    def test_refusals(self):
        bad_models = [
            [model("a", 8101)],
            [model("a", 8101), model("a", 8102)],
            [model("a", 8101), model("b", 8101)],
            [model("a", 8101), model("b", 8200)],
            [model("a", 8101), model("b", True)],
            [model("a", 8101), model("b", 8102, gpu_memory_utilization=0.93)],
            [model("a", 8101), model("b", 8102, gpu_memory_utilization=0)],
            [model("a", 8101), model("b", 8102, argv=[])],
            [model("a", 8101), model("b", 8102, argv=["--max-model-len", "1"])],
            [model("a", 8101), model("b", 8102, env={"X": 1})],
            [model("a", 8101), model("b", 8102, extra=1)],
        ] + [[model("a", 8101), model("b", 8102, argv=["/m", flag])] for flag in (
            "--port", "--port=9000", "--host", "--served-model-name=x", "--served_model_name",
            "--gpu-memory-utilization", "--gpu-memory-util=0.5", "--enable-sleep-mode",
            "--no-enable-sleep-mode", "--enable_sleep_mode")]
        cases = [config(models=m) for m in bad_models] + [
            config(gpu_budget=0.96), config(gpu_budget=True), config(sleep_level=3),
            config(slice_ms=-1), config(slice_ms=1.5), config(listen=8000), config(extra=1), []]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                multimodel.validate(json.loads(json.dumps(case)))

    def test_refuses_to_run_without_config(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(multimodel.main(), 1)
        with patch.dict(os.environ, {multimodel.ENV: json.dumps(config(sleep_level=0))}):
            self.assertEqual(multimodel.main(), 1)


# ---------------------------------------------------------- fake children ---

class Children:
    """Shared event log so swap order and 'at most one awake' are observable."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events, self.awake, self.max_awake, self.forwarded_asleep = [], set(), 0, 0

    def mark(self, name, event):
        with self.lock:
            self.events.append((name, event))
            (self.awake.add if event == "wake" else self.awake.discard)(name)
            self.max_awake = max(self.max_awake, len(self.awake))


SSE = [b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
       b'data: {"choices":[{"delta":{"content":"llo \xc3\xa9"}}]}\n\n', b"data: [DONE]\n\n"]


def fake_child(children, name, port, awake, fail_wake=False):
    if awake:
        children.mark(name, "wake")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # close after every response, like Connection: close

        def log_message(self, *args):
            pass

        def reply(self, code, body, ctype="application/json"):
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, b"", "text/plain")
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
            elif self.path == "/metrics":
                self.reply(200, (f"# HELP vllm:num_requests_running Running.\n"
                                 f"# TYPE vllm:num_requests_running gauge\n"
                                 f'vllm:num_requests_running{{model_name="{name}"}} 0.0\n'
                                 f"# HELP process_cpu_seconds_total CPU.\n"
                                 f"# TYPE process_cpu_seconds_total counter\n"
                                 f"process_cpu_seconds_total 1.0\n").encode(), "text/plain")
            elif self.path == "/is_sleeping":
                self.reply(200, {"is_sleeping": name not in children.awake})
            else:
                self.reply(404, {"detail": "Not Found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/sleep"):
                children.mark(name, "sleep")
                self.reply(200, {"status": "sleeping", "level": 1})
                return
            if self.path == "/wake_up":
                if fail_wake:
                    self.reply(500, {"detail": "cuda oom"})
                    return
                children.mark(name, "wake")
                self.reply(200, {"status": "awake", "tags": None})
                return
            if name not in children.awake:
                children.forwarded_asleep += 1
                self.reply(500, {"detail": f"{name} is asleep"})
            elif self.path == "/v1/chat/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in SSE:
                    self.wfile.write(event)
                    self.wfile.flush()
                    time.sleep(0.05)
            elif self.path == "/v1/systemone":
                self.reply(200, {"model": name, "host": self.headers.get("Host"),
                                 "bytes": len(body)})
            else:
                self.reply(404, {"detail": "Not Found"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def free_ports(count, low=8100, high=8199):
    ports = []
    for port in range(low, high + 1):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        ports.append(port)
        if len(ports) == count:
            return ports
    raise unittest.SkipTest("no free ports in 8100-8199")


def request(port, method, path, body=None, headers=None, raw=False):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    connection.request(method, path, body=data, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload if raw else (json.loads(payload) if payload.startswith(b"{") else payload)


class Router:
    def __init__(self, children, specs, slice_ms=200):
        *child_ports, self.port = free_ports(len(specs) + 1)
        self.servers = [fake_child(children, name, port, awake, fail)
                        for (name, awake, fail), port in zip(specs, child_ports)]
        self.ports = dict(zip([s[0] for s in specs], child_ports))
        value = multimodel.validate({
            "listen": f"127.0.0.1:{self.port}", "gpu_budget": 0.92, "slice_ms": slice_ms,
            "models": [model(name, port) | {"argv": ["/m"]} for (name, _, _), port in zip(specs, child_ports)]})
        for spec, entry in zip(specs, value["models"]):
            entry["awake"] = spec[1]
        self.error = None
        self.thread = threading.Thread(target=self.run, args=(json.dumps(value),), daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self.error is None:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError(f"router did not come up: {self.error}")

    def run(self, value):
        try:
            _native.multimodel_serve(value)
        except Exception as error:  # surfaced through stop()
            self.error = error

    def stop(self):
        stopped = _native.multimodel_shutdown()
        self.thread.join(10)
        for server in self.servers:
            server.shutdown()
            server.server_close()
        if not stopped or self.thread.is_alive() or self.error:
            raise RuntimeError(f"router did not stop cleanly: {self.error}")


@unittest.skipUnless(ROUTER, "built _native lacks multimodel_serve; rebuild with maturin first")
class RouterIntegration(unittest.TestCase):
    def test_routing_swaps_streaming_and_overhead(self):
        children = Children()
        router = Router(children, [("a", False, False), ("b", True, False)])
        try:
            self.check(router, children)
        finally:
            router.stop()

    def check(self, router, children):
        port = router.port
        self.assertEqual(request(port, "GET", "/health")[0], 200)
        status, listed = request(port, "GET", "/v1/models")
        self.assertEqual((status, sorted(m["id"] for m in listed["data"])), (200, ["a", "b"]))
        status, metrics = request(port, "GET", "/metrics")
        text = metrics.decode()
        self.assertEqual(status, 200)
        self.assertEqual(text.count("# TYPE vllm:num_requests_running "), 1)
        self.assertIn('vllm:num_requests_running{model_name="a"} 0.0', text)
        self.assertIn('vllm:num_requests_running{model_name="b"} 0.0', text)
        self.assertEqual(text.count("process_cpu_seconds_total 1.0"), 1)
        self.assertIn('suffix_mm_awake{model="b"} 1', text)

        for method, path in (("POST", "/sleep"), ("POST", "/sleep?level=1"), ("POST", "/wake_up"),
                             ("GET", "/is_sleeping"), ("POST", "/collective_rpc"),
                             ("POST", "/reset_prefix_cache"), ("POST", "/v1/../sleep"),
                             ("GET", "/v1/chat/completions")):
            self.assertEqual(request(port, method, path, b"{}")[0], 404, path)
        self.assertEqual(children.events, [("b", "wake")], "dev paths must never be forwarded")

        status, answer = request(port, "POST", "/v1/systemone", {"model": "b", "state": {}})
        self.assertEqual((status, answer["model"]), (200, "b"))
        self.assertEqual(children.events, [("b", "wake")], "an awake model needs no swap")
        # Host fallback to a: budgets 0.8 + 0.8 > 0.92, so b must sleep first.
        status, answer = request(port, "POST", "/v1/systemone", {"state": {}},
                                 {"Host": "A.test:443"})
        self.assertEqual((status, answer["model"], answer["host"]), (200, "a", "A.test:443"))
        self.assertEqual(children.events, [("b", "wake"), ("b", "sleep"), ("a", "wake")])
        self.assertEqual((children.max_awake, children.forwarded_asleep), (1, 0))

        self.assertEqual(request(port, "POST", "/v1/systemone", {"state": {}})[0], 400)
        self.assertEqual(request(port, "POST", "/v1/systemone", {"state": {}},
                                 {"Host": "nobody.test"})[0], 400)
        self.assertEqual(request(port, "POST", "/v1/systemone", {"model": "zzz"})[0], 404)

        chat = {"model": "a", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        start = time.monotonic()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request("POST", "/v1/chat/completions", json.dumps(chat).encode())
        response = connection.getresponse()
        first = response.read1(65536)
        first_at = time.monotonic() - start
        routed = first + response.read()
        total = time.monotonic() - start
        connection.close()
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        self.assertLess(first_at, total - 0.05, "SSE must stream, not buffer")
        direct = request(router.ports["a"], "POST", "/v1/chat/completions", chat, raw=True)[1]
        self.assertEqual(routed, direct)
        self.assertEqual(routed, b"".join(SSE))

        # Chunked request body reaches the child re-framed with Content-Length.
        body = json.dumps({"model": "a", "pad": "x" * 100_000}).encode()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request("POST", "/v1/systemone", iter([body[:7], body[7:]]),
                           {"Transfer-Encoding": "chunked"}, encode_chunked=True)
        response = connection.getresponse()
        self.assertEqual((response.status, json.loads(response.read())["bytes"]), (200, len(body)))
        connection.close()

        # Concurrent clients of both models: never two awake, never forwarded asleep.
        results = []
        def client(name):
            results.append(request(port, "POST", "/v1/systemone", {"model": name})[1]["model"])
        threads = [threading.Thread(target=client, args=(n,)) for n in "abab"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(sorted(results), ["a", "a", "b", "b"])
        self.assertEqual((children.max_awake, children.forwarded_asleep), (1, 0))
        events = children.events
        for before, after in zip(events, events[1:]):
            if after[1] == "wake":
                self.assertEqual(before[1], "sleep", f"woke without a prior sleep: {events}")

        # P2: router overhead vs direct-to-child p50 on localhost.
        awake = next(iter(children.awake))
        def p50(target):
            samples = []
            for _ in range(300):
                start = time.perf_counter()
                request(target, "POST", "/v1/systemone", {"model": awake})
                samples.append(time.perf_counter() - start)
            return statistics.median(samples)
        direct_p50, routed_p50 = p50(router.ports[awake]), p50(port)
        overhead_ms = (routed_p50 - direct_p50) * 1e3
        print(f"\nSUFFIX_MULTIMODEL router overhead p50: {overhead_ms:.3f} ms "
              f"(direct {direct_p50 * 1e3:.3f} ms, routed {routed_p50 * 1e3:.3f} ms)")
        self.assertLessEqual(overhead_ms, 1.0)

        status, metrics = request(port, "GET", "/metrics")
        self.assertIn('suffix_mm_swaps_total{from="b",to="a"}', metrics.decode())
        self.assertIn('suffix_mm_inflight{model="a"} 0', metrics.decode())

    def test_failed_wake_is_503_and_unhealthy(self):
        children = Children()
        router = Router(children, [("c", True, False), ("d", False, True)])
        try:
            status, answer = request(router.port, "POST", "/v1/chat/completions", {"model": "d"})
            self.assertEqual(status, 503)
            self.assertIn("failed", answer["error"]["message"])
            self.assertEqual(request(router.port, "GET", "/health")[0], 503)
            self.assertEqual(request(router.port, "POST", "/v1/systemone", {"model": "d"})[0], 503)
            self.assertEqual(children.events, [("c", "wake"), ("c", "sleep")])
        finally:
            router.stop()


if __name__ == "__main__":
    unittest.main()
