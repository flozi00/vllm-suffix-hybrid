#!/usr/bin/env python3
"""Stdlib streaming benchmark; Python 3.9+. See --help for CLI.

TTFT is client-observed first nonempty content/reasoning delta. Decode rate
is (server completion_tokens - 1)/(last content time - first content time).
SSE chunks may contain multiple tokens (especially speculative decoding):
this is an observed decode-rate estimate, NOT per-token or engine-step latency.
Usage/DONE transport delay is excluded from decode time, included in latency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

from bench import make_prompt


def summarize(results, wall_s):
    ok = [r for r in results if 'error' not in r]
    summary = {'requests': len(results), 'successful': len(ok),
               'errors': len(results) - len(ok), 'wall_s': wall_s,
               'completion_tokens': sum(r['completion_tokens'] for r in ok)}
    summary['aggregate_tok_s'] = summary['completion_tokens'] / wall_s if wall_s > 0 else None
    for name in ('ttft_s', 'latency_s', 'decode_tok_s'):
        values = sorted(r[name] for r in ok if r.get(name) is not None)
        def percentile(p):
            if not values:
                return None
            pos = (len(values) - 1) * p
            low = int(pos)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (pos - low)
        summary[name] = {'count': len(values), 'mean': statistics.mean(values) if values else None,
                         'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99)}
    return summary


def run_one(args, index, headers):
    prompt = make_prompt(args.prompt_seed + index)
    record = {'index': index, 'prompt_seed': args.prompt_seed + index,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    payload = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': args.max_tokens, 'temperature': 0, 'seed': args.seed,
               'min_p': 0, 'top_p': 1, 'ignore_eos': True, 'stream': True,
               'stream_options': {'include_usage': True}}
    req = urllib.request.Request(args.endpoint.rstrip('/') + '/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            record.update(consume_stream(response, started))
        if record['completion_tokens'] != args.max_tokens:
            record['error'] = 'fixed output budget mismatch: expected %s, received %s' % (
                args.max_tokens, record['completion_tokens'])
    except Exception as exc:
        record['error'] = '%s: %s' % (type(exc).__name__, exc)
        record['latency_s'] = time.perf_counter() - started
        if isinstance(exc, urllib.error.HTTPError):
            record['http_status'] = exc.code
            record['response_body'] = exc.read(8192).decode('utf-8', errors='replace')
    return record


def metric_snapshot(url, timeout, headers):
    result = {'url': url, 'captured_at': datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            result['raw'] = response.read().decode('utf-8')
    except Exception as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True, help='OpenAI API base, e.g. http://localhost:8000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--num-prompts', '--prompts', type=int, default=64)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--label', default='run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--seed', type=int, default=0, help='Identical sampling seed on every request')
    parser.add_argument('--prompt-seed', type=int, default=0, help='First existing bench.make_prompt seed')
    parser.add_argument('--metrics-url', help='Default: endpoint origin /metrics; override for proxies')
    parser.add_argument('--metadata', type=Path, help='Optional JSON environment metadata (GPU, image, revision, etc.)')
    args = parser.parse_args(argv)
    if min(args.num_prompts, args.concurrency, args.max_tokens) < 1 or args.timeout <= 0:
        parser.error('prompts, concurrency, max-tokens and timeout must be positive')
    metadata = json.loads(args.metadata.read_text()) if args.metadata else None
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    metrics_url = args.metrics_url or urllib.parse.urljoin(args.endpoint, '/metrics')
    # Do not forward credentials to a different metrics origin.
    metrics_headers = {}
    if urllib.parse.urlsplit(metrics_url)[:2] == urllib.parse.urlsplit(args.endpoint)[:2]:
        metrics_headers = {k: v for k, v in headers.items() if k == 'Authorization'}
    before = metric_snapshot(metrics_url, args.timeout, metrics_headers)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(lambda i: run_one(args, i, headers), range(args.num_prompts)))
    wall_s = time.perf_counter() - started
    after = metric_snapshot(metrics_url, args.timeout, metrics_headers)
    artifact = {
        'schema_version': 1, 'label': args.label, 'started_at': started_at,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'sampling': {'temperature': 0, 'seed': args.seed, 'min_p': 0, 'top_p': 1, 'ignore_eos': True},
        'environment': {'client_python': platform.python_version(), 'client_platform': platform.platform(),
                        'provided': metadata},
        'measurement_notes': [
            'TTFT: request start to first nonempty content or reasoning delta; excludes executor queue time.',
            'Decode tok/s: (server completion_tokens-1)/(last-first content/reasoning time).',
            'Decode rate is an estimate: SSE chunks can contain multiple tokens; not engine step latency.',
            'Single-content-chunk decode rate is null; usage/DONE delay excluded from decode duration.',
            'Aggregate: successful fixed-budget completion tokens / workload wall seconds, including failed-request time.',
            'Percentiles use linear interpolation; raw metrics are server-global, not isolated to this client.',
            'No warmup or retries; run identical workloads sequentially on the same otherwise idle GPU.',
        ],
        'metrics': {'before': before, 'after': after}, 'requests': results,
        'summary': summarize(results, wall_s),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'label': args.label, 'output': str(args.output), **artifact['summary']}, allow_nan=False))
    return 1 if artifact['summary']['errors'] else 0


def sse_events(stream):
    """Parse SSE data fields, including comments, CRLF and multiline events."""
    data = []
    for raw in stream:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            value = line[5:]
            data.append(value[1:] if value.startswith(' ') else value)
    if data:
        yield '\n'.join(data)


def consume_stream(stream, started, clock=time.perf_counter):
    content, reasoning = [], []
    first = last = None
    usage = None
    done = False
    for event in sse_events(stream):
        now = clock()
        if event == '[DONE]':
            done = True
            break
        chunk = json.loads(event)
        if chunk.get('error'):
            raise ValueError('server error: ' + json.dumps(chunk['error']))
        if chunk.get('usage') is not None:
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            text = delta.get('content') or ''
            thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
            if text or thought:
                if first is None:
                    first = now
                last = now
                content.append(text)
                reasoning.append(thought)
    if not done:
        raise ValueError('stream ended without [DONE]')
    tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        raise ValueError('missing or invalid authoritative completion_tokens usage')
    decode = last - first if first is not None else None
    return {
        'usage': usage, 'completion_tokens': tokens,
        'ttft_s': first - started if first is not None else None,
        'latency_s': now - started,
        'decode_duration_s': decode,
        'decode_tok_s': (tokens - 1) / decode if tokens > 1 and decode else None,
        'content_sha256': hashlib.sha256(''.join(content).encode()).hexdigest(),
        'reasoning_sha256': hashlib.sha256(''.join(reasoning).encode()).hexdigest(),
    }


if __name__ == '__main__':
    raise SystemExit(main())
