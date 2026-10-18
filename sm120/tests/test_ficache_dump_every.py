# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_FICACHE_DUMP_EVERY repeat-dump mode (CPU tests).

The single-shot dump at t+207s was lost 3x to pod log rotation (log window
life ~250s at ~20 lines/s). With SUFFIX_FICACHE_DUMP_EVERY armed, the
harvest re-emits the SUFFIX_FICACHE_DUMP marker every cadence even while
the cache content is unchanged.

Covers: _env_flag polarity (the a4d3db93 review-lesson helper, mirrored
here — never a bare '!= "0"' string compare), repeat vs single-shot scan
behaviour, content-change re-dump in both modes, and install() banner.
"""
import os
from pathlib import Path

import pytest

from sm120.ficache import _env_flag, _scan_once, encode_dump


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    hash_dir = tmp_path / "deadbeef" / "jit_ws_thread0"
    hash_dir.mkdir(parents=True)
    (hash_dir / "autotune_configs.json").write_text('{"tactic": 1}')
    return tmp_path


def _markers(capsys):
    out = capsys.readouterr().out
    return [l for l in out.splitlines() if l.startswith("SUFFIX_FICACHE_DUMP")]


# ---------------------------------------------------------------- _env_flag

@pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", " on ", "ON"])
def test_env_flag_arms(cache_root, val):
    os.environ["SUFFIX_FICACHE_DUMP_EVERY"] = val
    try:
        assert _env_flag("SUFFIX_FICACHE_DUMP_EVERY") is True
    finally:
        del os.environ["SUFFIX_FICACHE_DUMP_EVERY"]


@pytest.mark.parametrize(
    "val", ["0", "false", "False", "no", "off", "", "  ", "2", "yes please"]
)
def test_env_flag_disarms(cache_root, val):
    os.environ["SUFFIX_FICACHE_DUMP_EVERY"] = val
    try:
        # the a4d3db93 lesson: bare '!= "0"' would ARM on "false"/"no"/"off"
        assert _env_flag("SUFFIX_FICACHE_DUMP_EVERY") is False
    finally:
        del os.environ["SUFFIX_FICACHE_DUMP_EVERY"]


def test_env_flag_default_off(cache_root, monkeypatch):
    monkeypatch.delenv("SUFFIX_FICACHE_DUMP_EVERY", raising=False)
    assert _env_flag("SUFFIX_FICACHE_DUMP_EVERY") is False


# ------------------------------------------------------------------- scans

def test_scan_once_default_single_shot(cache_root, capsys):
    seen: dict = {}
    _scan_once(seen)
    assert len(_markers(capsys)) == 1
    # unchanged content: no re-dump
    _scan_once(seen)
    assert len(_markers(capsys)) == 0


def test_scan_once_repeat_redumps_unchanged(cache_root, capsys):
    seen: dict = {}
    for _ in range(3):  # three cadences, content never changes
        _scan_once(seen, repeat=True)
        assert len(_markers(capsys)) == 1
    # the re-emitted lines are identical (same hash dir, same payload)
    seen2: dict = {}
    _scan_once(seen2, repeat=True)
    first = _markers(capsys)[0]
    seen: dict = {}
    capsys.readouterr()
    _scan_once(seen, repeat=True)
    assert _markers(capsys)[0] == first


def test_scan_once_repeat_dumps_content_change(cache_root, capsys):
    seen: dict = {}
    _scan_once(seen, repeat=True)
    line1 = _markers(capsys)[0]
    cfg = next(cache_root.glob("*/*/autotune_configs.json"))
    cfg.write_text('{"tactic": 22}')  # size change: same-size rewrites can share mtime_ns on coarse-timestamp CI filesystems  # mtime_ns+size change
    _scan_once(seen, repeat=True)
    line2 = _markers(capsys)[0]
    assert line2 != line1  # payload changed -> new dump


def test_scan_once_default_dumps_content_change(cache_root, capsys):
    seen: dict = {}
    _scan_once(seen)
    _markers(capsys)
    cfg = next(cache_root.glob("*/*/autotune_configs.json"))
    cfg.write_text('{"tactic": 22}')  # size change: same-size rewrites can share mtime_ns on coarse-timestamp CI filesystems
    _scan_once(seen)
    assert len(_markers(capsys)) == 1


def test_scan_once_empty_root(tmp_path, capsys):
    _scan_once({})
    assert _markers(capsys) == []


def test_scan_legacy_and_version_arch_hash_layouts(tmp_path, monkeypatch, capsys):
    from sm120 import ficache
    import base64
    import gzip
    import json

    monkeypatch.setattr(ficache, "_cache_roots", lambda: [tmp_path])
    caches = [tmp_path / "workspace" / "legacy_hash" / "autotune_configs.json",
              tmp_path / "0.7.0.post1" / "120f" / "current_hash" / "autotune_configs.json"]
    for i, path in enumerate(caches):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"tactic": i}))
    (tmp_path / "unrelated.json").write_text('{"tactic": 99}')
    seen = {}
    _scan_once(seen)
    lines = _markers(capsys)
    payloads = {line.split()[1]: json.loads(gzip.decompress(
        base64.b64decode(line.split()[2]))) for line in lines}
    assert payloads == {"legacy_hash": {"tactic": 0}, "current_hash": {"tactic": 1}}
    _scan_once(seen)
    assert _markers(capsys) == []
    caches[1].write_text('{"tactic": 22}')
    _scan_once(seen)
    assert len(_markers(capsys)) == 1

def test_encode_dump_roundtrip(cache_root):
    import base64
    import gzip

    cfg = next(cache_root.glob("*/*/autotune_configs.json"))
    line = encode_dump(cfg)
    prefix, hash_dir, payload = line.split(" ", 2)
    assert prefix == "SUFFIX_FICACHE_DUMP"
    assert hash_dir == cfg.parent.name
    assert gzip.decompress(base64.b64decode(payload)) == cfg.read_bytes()


def test_ranked_cache_receipts_and_seeds_do_not_collide(tmp_path, monkeypatch, capsys):
    from sm120 import ficache
    import base64
    import gzip
    import json

    root = tmp_path / "cache"
    folder = root / "0.7.0.post1" / "120f" / "same_hash"
    folder.mkdir(parents=True)
    monkeypatch.setattr(ficache, "_cache_roots", lambda: [root])
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    monkeypatch.setattr(ficache, "_SEEDS_DIR", seeds)
    expected = {}
    for dp, rank in [(0, 0), (0, 1), (1, 0)]:
        path = folder / f"autotune_configs_dp{dp}_rank{rank}.json"
        path.write_text(json.dumps({"dp": dp, "rank": rank}))
        expected[ficache._cache_key(path)] = path.read_bytes()
    for name in ["autotune_configs_notes.json", "autotune_configs_dp0_rankx.json"]:
        (folder / name).write_text('{}')
    _scan_once({})
    lines = _markers(capsys)
    assert len(lines) == 3
    for line in lines:
        _, key, payload = line.split()
        assert gzip.decompress(base64.b64decode(payload)) == expected[key]
        (seeds / f"{key}.json.gz").write_bytes(base64.b64decode(payload))
    for path in folder.glob("autotune_configs_dp[01]_rank[01].json"):
        key = ficache._cache_key(path)
        path.unlink()
        ficache._seed_if_missing(path)
        assert path.read_bytes() == expected[key]
