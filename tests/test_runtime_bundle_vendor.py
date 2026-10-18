# SPDX-License-Identifier: Apache-2.0
"""runtime_bundle --vendor-dir: the vendored tree lands in _vendor/, hashed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendor_dir_bundled_and_hashed(tmp_path, monkeypatch):
    rb = _load("runtime_bundle", REPO / "scripts" / "runtime_bundle.py")
    wheel = tmp_path / "plugin.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("suffix_hybrid/__init__.py", "")
        archive.writestr("suffix_hybrid/_native.abi3.so", b"x")
        archive.writestr("suffix_hybrid/configs/helion/k/gpu.json", "[]")
        archive.write(REPO / "suffix_hybrid" / "helion_kernels" / "__init__.py",
                      "suffix_hybrid/helion_kernels/__init__.py")
    vendor = tmp_path / "vendor"
    (vendor / "helion" / "__pycache__").mkdir(parents=True)
    (vendor / "helion" / "__init__.py").write_text("VERSION = 1\n")
    (vendor / "helion" / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"pyc")
    (vendor / "helion-1.5.0.dist-info").mkdir()
    (vendor / "helion-1.5.0.dist-info" / "METADATA").write_text("Name: helion\n")
    out = tmp_path / "runtime"
    rb.bundle(wheel, out, "f" * 40, vendor_dir=vendor)
    hashes = json.loads((out / "BUILD.json").read_text())["sha256"]
    for rel in ("_vendor/helion/__init__.py", "_vendor/helion-1.5.0.dist-info/METADATA"):
        assert hashes[rel] == hashlib.sha256((out / rel).read_bytes()).hexdigest()
    assert "suffix_hybrid/configs/helion/k/gpu.json" in hashes
    assert not any("__pycache__" in rel for rel in hashes)
    # the bundled runtime finds the bundled vendor dir (appended: image wins)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HELION_AUTOTUNER", "PatternSearch")
    _load("bundled_hk", out / "suffix_hybrid" / "helion_kernels" / "__init__.py").ensure_vendor()
    assert sys.path[-1] == str(out / "_vendor")
