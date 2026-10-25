# SPDX-License-Identifier: Apache-2.0
"""Opt-in load-time tensor hashes; no hooks or work in serving forwards.

SUFFIX_WEIGHT_RECEIPT=1 records parameters and buffers before/after the
dense conversion. Full tensor bytes are hashed on CPU; values are never
logged. Receipts distinguish checkpoint/loading changes from execution
changes when two deterministic model versions disagree.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sys

MARKER = "SUFFIX_WEIGHT_RECEIPT"


def tensor_digest(tensor) -> str:
    import torch

    if tensor.device.type == "meta":
        raise ValueError("cannot fingerprint an unloaded meta tensor")
    # Transfer before making a contiguous view: never allocate a second
    # complete weight on the GPU. memoryview avoids another CPU byte copy.
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def collect(model, phase: str) -> dict:
    rows = []
    for kind, entries in (("parameter", model.named_parameters()),
                          ("buffer", model.named_buffers())):
        for name, tensor in sorted(entries):
            try:
                digest = tensor_digest(tensor)
            except Exception as exc:
                raise RuntimeError(f"weight receipt failed during {phase}: {kind} {name}") from exc
            rows.append({"kind": kind, "name": name,
                         "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "stride": list(tensor.stride()),
                         "bytes": tensor.numel() * tensor.element_size(),
                         "sha256": digest})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": 1, "phase": phase, "model_class": type(model).__name__,
            "tensors": rows, "total_bytes": sum(r["bytes"] for r in rows),
            "sha256": hashlib.sha256(payload).hexdigest()}


def emit(model, phase: str) -> None:
    receipt = collect(model, phase)
    blob = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.b64encode(gzip.compress(blob, mtime=0)).decode()
    print(f"{MARKER} {encoded}", file=sys.stderr, flush=True)
