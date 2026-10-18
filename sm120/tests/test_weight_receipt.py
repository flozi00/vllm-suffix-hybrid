# SPDX-License-Identifier: Apache-2.0
"""Load-time diagnostic hashes must detect one-bit changes without mutation."""
import base64
import ast
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp8_dense_patch as P
from fp8_dense_patch import weight_receipt as R


def test_tensor_bytes_noncontiguous_and_scalar():
    x = torch.arange(24, dtype=torch.int32).reshape(4, 6).t()
    before = x.clone()
    want = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    assert R.tensor_digest(x) == want
    assert torch.equal(x, before)
    assert R.tensor_digest(torch.tensor(7, dtype=torch.int32)) == hashlib.sha256(
        torch.tensor([7], dtype=torch.int32).numpy().tobytes()).hexdigest()
    with pytest.raises(ValueError, match="meta"):
        R.tensor_digest(torch.empty(3, device="meta"))


def test_receipt_roundtrip_detects_weight_and_buffer_changes(capsys):
    model = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    model.register_buffer("scale", torch.tensor([1.0]))
    original = R.collect(model, "loaded")
    assert {r["kind"] for r in original["tensors"]} == {"parameter", "buffer"}
    R.emit(model, "loaded")
    marker, encoded = capsys.readouterr().err.strip().split()
    assert marker == R.MARKER
    assert json.loads(gzip.decompress(base64.b64decode(encoded))) == original
    assert R.collect(model, "loaded") == original
    model.scale[0] = 2.0
    changed = R.collect(model, "loaded")
    assert changed["sha256"] != original["sha256"]
    with torch.no_grad():
        model.weight.view(torch.int16)[0, 0] ^= 1
    assert R.collect(model, "loaded")["sha256"] != changed["sha256"]


def test_unloaded_tensor_names_the_failing_phase_and_parameter():
    model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(RuntimeError, match="before_load: parameter bias"):
        R.collect(model, "before_load")


def test_gate_off_does_no_receipt_work(monkeypatch):
    monkeypatch.delenv("SUFFIX_WEIGHT_RECEIPT", raising=False)
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.delenv(P.NVFP4_GATE_ENV, raising=False)
    monkeypatch.delenv("SUFFIX_ACT_AMAX_RECORD", raising=False)
    monkeypatch.setattr(R, "emit", lambda *_: pytest.fail("receipt ran with gate off"))
    assert not P.gate_enabled()
    P._suffix_fp8_dense_convert(torch.nn.Linear(2, 2))


def test_gate_records_both_load_phases(monkeypatch):
    monkeypatch.setenv("SUFFIX_WEIGHT_RECEIPT", "1")
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.delenv(P.NVFP4_GATE_ENV, raising=False)
    monkeypatch.delenv("SUFFIX_ACT_AMAX_RECORD", raising=False)
    seen = []
    monkeypatch.setattr(R, "emit", lambda model, phase: seen.append(phase))
    assert P.gate_enabled()
    source = Path(__file__).resolve().parents[2] / "sitecustomize.py"
    arm = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.If)
               and "SUFFIX_WEIGHT_RECEIPT" in ast.unparse(node.test))
    assert eval(compile(ast.Expression(arm.test), "<receipt-arm>", "eval"), {"os": os})
    P._suffix_fp8_dense_convert(torch.nn.Linear(2, 2))
    assert seen == ["before_dense_conversion", "after_dense_conversion"]
