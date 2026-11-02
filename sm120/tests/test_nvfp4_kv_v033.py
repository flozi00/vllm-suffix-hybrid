# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream replay gates for the vLLM 0.33.0 K2 adapter.

0.33.0 keeps every anchor of the 0.32.0 adapter except
``own_attn_spec_as_decode``: upstream inserted
``self.use_fa2_native_decode = _use_fa2_native_decode()`` and ORs it into
``supports_spec_as_decode``. That helper is ``not has_device_capability(90)``,
so it is already True on SM120 — but the anchor is the only place that sets
``self.use_own_nvfp4_attn``, so it is re-anchored rather than dropped.
"""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sm120"))
import fp8_dense_patch as dense
import nvfp4_kv_patch as patch
from nvfp4_kv_patch import v032

FIXTURE = Path(__file__).parent / "fixtures/vllm_0.33.0"
V033 = "0.33.0"


def backend_src():
    return (FIXTURE / "flashinfer_backend.py").read_text()


def test_pinned_fixture_hashes():
    expected = {
        "flashinfer_backend.py":
            "5dbc8ee6bf006a1a",
        "model_loader_utils.py":
            "f81091610299bd1a",
        "gather_batch_req_state.py.txt":
            "18da780f3c3ffdea",
    }
    for name, prefix in expected.items():
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert digest.startswith(prefix), name


def test_adapter_table_accepts_the_shipped_pair():
    assert (V033, "0.7.2.post1") in patch.NATIVE_ADAPTERS
    assert ("0.32.0", "0.7.0.post1") in patch.NATIVE_ADAPTERS
    # A 0.33.0 image with the OLD flashinfer must not take the native path.
    assert (V033, "0.7.0.post1") not in patch.NATIVE_ADAPTERS


def test_v033_replay_applies_every_anchor():
    out, applied = v032.patch_backend_source(backend_src(), patch, V033)
    assert len(applied) == 19
    assert "own_attn_spec_as_decode" in applied
    # The own-attn lane is armed by this anchor and by nothing else.
    assert "self.use_own_nvfp4_attn = _nvfp4_own_attn_gate(self)" in out
    assert "or self.use_own_nvfp4_attn" in out
    compile(out, "<patched_v033>", "exec")


def test_v032_anchor_set_does_not_apply_to_v033():
    """The version dispatch is load-bearing, not cosmetic."""
    with pytest.raises(patch.PatchDriftError, match="own_attn_spec_as_decode"):
        v032.patch_backend_source(backend_src(), patch, "0.32.0")


def test_v033_replay_is_not_idempotent():
    out, _ = v032.patch_backend_source(backend_src(), patch, V033)
    with pytest.raises(patch.PatchDriftError):
        v032.patch_backend_source(out, patch, V033)


def test_gather_guard_hash_is_unchanged_in_v033():
    """The hash-pinned prefill graph guard must still install on 0.33.0."""
    src = (FIXTURE / "gather_batch_req_state.py.txt").read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == (
        "18da780f3c3ffdea5e38b6d663083792796b92e8bfe2159ae16b76d0af9db7ca"
    )


def test_dense_loader_hook_applies_on_v033():
    assert V033 in dense.SUPPORTED_VLLM
    assert dense.patch_source(
        (FIXTURE / "model_loader_utils.py").read_text())[1] == [
            "convert_after_post_load"]
