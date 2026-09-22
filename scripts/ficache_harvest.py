#!/usr/bin/env python3
"""Harvest SUFFIX_FICACHE_DUMP log lines into bundle seed files.

Usage: python ficache_harvest.py <log-file>... <out-dir>

The ficache dump daemon (SUFFIX_FICACHE=dump) prints one line per changed
autotune_configs.json:  SUFFIX_FICACHE_DUMP <config-hash> <gzip+b64>.
Console get_logs is the only egress from the pods, so we reconstruct the
seeds from captured log text. Last dump per hash wins (daemon re-dumps on
every change). Output: <out-dir>/<hash>.json.gz — drop into
sm120/ficache/seeds/ and rebuild the runtime bundle.
"""
import base64
import gzip
import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"SUFFIX_FICACHE_DUMP ([0-9a-f]{64}) ([A-Za-z0-9+/=]+)")


def harvest(log_paths):
    latest = {}
    for p in log_paths:
        text = Path(p).read_text(errors="replace")
        for h, b64 in LINE.findall(text):
            try:
                data = gzip.decompress(base64.b64decode(b64))
                json.loads(data)  # validate: seeds must be valid JSON
                latest[h] = data
            except Exception as exc:
                print(f"skip corrupt dump frame for {h[:12]}: {exc}",
                      file=sys.stderr)
    return latest


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    *logs, out = argv[1:]
    seeds = harvest(logs)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, data in sorted(seeds.items()):
        dest = out_dir / f"{h}.json.gz"
        dest.write_bytes(gzip.compress(data, mtime=0))
        n = len(json.loads(data).get("tactics", {}))
        print(f"wrote {dest.name} ({len(data)} bytes, {n} tuned ops)")
    if not seeds:
        print("no SUFFIX_FICACHE_DUMP frames found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
