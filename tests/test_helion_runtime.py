# SPDX-License-Identifier: Apache-2.0
"""Helion runtime: the runner's config file format, exact lookup, buckets, knob."""
import json

import pytest

from suffix_hybrid import helion_kernels as hk


def test_config_lookup(tmp_path, monkeypatch):
    rows = [{"key": {"n": 64, "numtokens": 1}, "config": {"m": 1}},  # the runner's form
            {"key": hk.case_key({"n": 64, "numtokens": 32}), "config": {"m": 32}}]
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "gpu.json").write_text(json.dumps(rows))
    monkeypatch.setattr(hk, "CONFIG_ROOT", str(tmp_path))
    monkeypatch.setattr(hk, "_TABLES", {})
    assert hk.config("k", {"numtokens": 32, "n": 64}, "gpu") == {"m": 32}
    assert hk.config("k", {"numtokens": 1, "n": 64}, "gpu") == {"m": 1}
    assert hk.config("k", {"n": 64, "numtokens": 8}, "gpu") is None  # oxide won / untuned
    assert hk.config("missing", {"numtokens": 1}, "gpu") is None
    assert [hk.bucket(m, (1, 8, 32)) for m in (1, 2, 8, 9, 99)] == [1, 8, 8, 32, 32]


def test_backend_knob(monkeypatch):
    monkeypatch.delenv("X_BACKEND", raising=False)
    assert hk.backend("X_BACKEND") == "oxide"
    monkeypatch.setenv("X_BACKEND", "Auto")
    assert hk.backend("X_BACKEND") == "auto"
    monkeypatch.setenv("X_BACKEND", "cutlass")
    with pytest.raises(RuntimeError):
        hk.backend("X_BACKEND")
