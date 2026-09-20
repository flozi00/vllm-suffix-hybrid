"""Synthetic local fixtures only; these are not benchmark results."""
import hashlib
import io
import json
import unittest

import compare


class StreamTests(unittest.TestCase):
    def test_sse_reasoning_ttft_usage_and_rates(self):
        frames = [
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [], "usage": {"completion_tokens": 9, "prompt_tokens": 20}},
        ]
        wire = b': keepalive\r\n\r\n' + b''.join(
            ('data: ' + json.dumps(f) + '\r\n\r\n').encode() for f in frames
        ) + b'data: [DONE]\n\n'
        ticks = iter([11.0, 12.0, 16.0, 18.0, 19.0])
        result = compare.consume_stream(io.BytesIO(wire), 10.0, lambda: next(ticks))
        self.assertEqual(result['ttft_s'], 2.0)
        self.assertEqual(result['latency_s'], 9.0)
        self.assertEqual(result['completion_tokens'], 9)
        self.assertEqual(result['decode_tok_s'], 2.0)  # (9-1)/(16-12), not full latency
        self.assertEqual(result['content_sha256'], hashlib.sha256(b'answer').hexdigest())
        self.assertEqual(result['reasoning_sha256'], hashlib.sha256(b'think').hexdigest())


class ValidationTests(unittest.TestCase):
    def test_missing_usage_and_truncated_stream_are_errors(self):
        for wire in [b'data: [DONE]\n\n', b'data: {"usage":{"completion_tokens":1}}\n\n']:
            with self.assertRaises(ValueError):
                compare.consume_stream(io.BytesIO(wire), 0, lambda: 1)

    def test_single_chunk_has_no_decode_rate(self):
        wire = b'data: {"choices":[{"delta":{"content":"abc"}}],"usage":{"completion_tokens":8}}\n\ndata: [DONE]\n\n'
        result = compare.consume_stream(io.BytesIO(wire), 0, lambda: 1)
        self.assertIsNone(result['decode_tok_s'])

    def test_multiline_sse(self):
        self.assertEqual(list(compare.sse_events(io.BytesIO(b': hi\ndata: one\ndata: two\n\n'))), ['one\ntwo'])

    def test_aggregate_uses_wall_time_and_excludes_errors(self):
        result = compare.summarize([
            {'completion_tokens': 10, 'ttft_s': 1, 'latency_s': 5, 'decode_tok_s': 3},
            {'error': 'failed'},
            {'completion_tokens': 20, 'ttft_s': 3, 'latency_s': 7, 'decode_tok_s': None},
        ], 10)
        self.assertEqual(result['aggregate_tok_s'], 3)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['ttft_s']['p50'], 2)
        self.assertEqual(result['decode_tok_s']['count'], 1)


class LocalHTTPTests(unittest.TestCase):
    def test_cli_persists_real_local_stream_and_metrics(self):
        import http.server
        import pathlib
        import tempfile
        import threading
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'vllm:spec_decode_num_draft_tokens_total 42\n')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(body)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if len(seen) == 2:
                    self.wfile.write(b'data: {"error":"synthetic failure"}\n\n')
                    return
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\ndata: {"usage":{"completion_tokens":4,"prompt_tokens":8},"choices":[]}\n\ndata: [DONE]\n\n')

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            scratch = pathlib.Path.home() / '.hermes' / 'cache' / 'scratch'
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(scratch)) as directory:
                output = pathlib.Path(directory) / 'artifact.json'
                import contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    code = compare.main(['--endpoint', 'http://127.0.0.1:%s/v1' % server.server_port,
                                         '--model', 'synthetic-test', '--num-prompts', '2',
                                         '--concurrency', '1', '--max-tokens', '4', '--label', 'fixture',
                                         '--output', str(output)])
                artifact = json.loads(output.read_text())
                self.assertEqual(code, 1)
                self.assertEqual(artifact['summary']['errors'], 1)
                self.assertEqual(len(artifact['requests']), 2)
                self.assertIn('vllm:spec_decode', artifact['metrics']['before']['raw'])
                self.assertIn('vllm:spec_decode', artifact['metrics']['after']['raw'])
                self.assertEqual(artifact['label'], 'fixture')
                for payload in seen:
                    self.assertEqual(payload['temperature'], 0)
                    self.assertEqual(payload['min_p'], 0)
                    self.assertEqual(payload['top_p'], 1)
                    self.assertTrue(payload['ignore_eos'])
                    self.assertTrue(payload['stream_options']['include_usage'])
                    self.assertEqual(payload['max_tokens'], 4)
                    self.assertIn('seed', payload)
                from bench import make_prompt
                self.assertEqual(seen[0]['messages'][0]['content'], make_prompt(0))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
