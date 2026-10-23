# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream replay and cache-format regression gates for vLLM 0.32."""
import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sm120"))
import fp8_dense_patch as dense
import nvfp4_kv_patch as patch
from nvfp4_kv_patch import v032

FIXTURE = Path(__file__).parent / "fixtures/vllm_0.32.0"


def transformed():
    return v032.patch_backend_source((FIXTURE / "flashinfer_backend.py").read_text(), patch)[0]


def test_pinned_sources_and_dense_loader_hook():
    expected = {
        "flashinfer_backend.py": "343b949d62db7fc03f6ca0c682e1ec1cb60125461069b081ac1554b7a05214e9",
        "model_loader_utils.py": "3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6",
    }
    for name, sha in expected.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == sha
    assert dense.patch_source((FIXTURE / "model_loader_utils.py").read_text())[1] == [
        "convert_after_post_load"]


def test_fail_closed_on_source_drift():
    src = (FIXTURE / "flashinfer_backend.py").read_text()
    with pytest.raises(patch.PatchDriftError, match="v032_keep_cpp_cache_writer"):
        v032.patch_backend_source(src.replace("            if self.nvfp4_fa2:\n", ""), patch)
    with pytest.raises(patch.PatchDriftError):
        v032.patch_backend_source(transformed(), patch)


@pytest.mark.parametrize("our_format", [True, False])
def test_writer_matches_reader_format(our_format):
    # Execute the actual transformed upstream method. Swizzled K2 pages must
    # use the C++ writer; other native FA2 instances keep the linear writer.
    cls = next(n for n in ast.parse(transformed()).body
               if isinstance(n, ast.ClassDef) and n.name == "FlashInferImpl")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "do_kv_cache_update")
    fn.decorator_list = []
    calls = []
    fake_torch = SimpleNamespace(Tensor=torch.Tensor, nn=torch.nn, ops=SimpleNamespace(
        _C_cache_ops=SimpleNamespace(reshape_and_cache_flash=lambda *a: calls.append(("cpp", a)))))
    ns = {"torch": fake_torch,
          "_nvfp4_kv_views": lambda cache, heads: ("data", "scales", "HND"),
          "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping":
              lambda *a, **kw: calls.append(("linear", a))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<writer>", "exec"), ns)
    obj = SimpleNamespace(kv_sharing_target_layer_name=None, nvfp4_fa2=True,
                          use_fa2_nvfp4_kv=our_format, is_kvcache_nvfp4=True,
                          num_kv_heads=2, cache_dtype="nvfp4")
    layer = SimpleNamespace(_k_scale=1, _v_scale=2)
    cache = torch.empty(3, 4, 16, 72, dtype=torch.uint8)
    ns["do_kv_cache_update"](obj, layer, "key", "value", cache, "slots")
    assert len(calls) == 1 and calls[0][0] == ("cpp" if our_format else "linear")
    if our_format:
        assert calls[0][1][2].shape == calls[0][1][3].shape == (3, 16, 2, 72)


def test_mm_and_k2_reuse_existing_gated_helpers():
    src = transformed()
    compile(src, "<patched_v032>", "exec")
    assert src.count("def _nvfp4_kv_mm_resplit") == 1
    assert src.count("def _nvfp4_kv_mm_prefill_mask") == 1
    # Prefill masks and short image extends survive; K2 remains independent
    # of host sequence lengths and runs through the actual FIDecode seam.
    assert "_mm_split = _nvfp4_kv_mm_resplit(common_attn_metadata, num_decodes)" in src
    assert "attn_metadata.decode = _own_decode" in src
    assert "prefill_wrapper._custom_mask_buf" in src


def test_new_flashinfer_custom_variant_condition_keeps_mask_gate():
    old = "if self._custom_mask_buf is not None:\n            mask_mode = MaskMode.CUSTOM.value"
    new = "if self._custom_mask_buf is not None or self._variant_owns_mask:\n            mask_mode = MaskMode.CUSTOM.value"
    rest = "\nself._mask_indptr_buf = mask_indptr.to(\nself._mask_indptr_buf,"
    files = {"prefill.py": new + rest}
    assert not patch.probe_flashinfer_headers(files)["mm_wrapper_mask_buffers"]
    assert v032.probe_headers(files, patch.probe_flashinfer_headers)["mm_wrapper_mask_buffers"]
    # Removing either custom dispatch or the mask indptr must fail closed.
    for broken in [(new + rest).replace(new, ""), (new + rest).replace(rest, "")]:
        assert not v032.probe_headers({"prefill.py": broken}, patch.probe_flashinfer_headers)[
            "mm_wrapper_mask_buffers"]


def test_full_graph_guard_preserves_decode_and_dummy_but_excludes_every_prefill():
    from collections import namedtuple
    State = namedtuple("State", "has_prefill decode_graph_eligible prefill_runs_as_decode_np")
    outputs = [(None, 4), (State(False, True, None), 4),
               (State(True, True, [True]), 4), (State(True, False, [False]), None)]
    for result in outputs:
        calls = []
        def gather(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        got = v032.guard_gather_state(gather)("runner", dummy_run=False)
        assert calls == [(("runner",), {"dummy_run": False})]
        state, uniform = result
        if state is None or not state.has_prefill:
            assert got == result and got[0] is state
        else:
            assert not got[0].decode_graph_eligible and got[1] is None
            assert got[0].prefill_runs_as_decode_np is state.prefill_runs_as_decode_np


def test_graph_guard_checks_real_upstream_method_identity(monkeypatch):
    import inspect
    import types
    ns = {"__name__": "pinned_gather"}
    path = FIXTURE / "gather_batch_req_state.py.txt"
    exec(compile(path.read_text(), str(path), "exec"), ns)
    cls = ns["GPUModelRunner"]
    source = inspect.getsource(cls.gather_batch_req_state).rstrip() + "\n"
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "18da780f3c3ffdea5e38b6d663083792796b92e8bfe2159ae16b76d0af9db7ca")
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                 "vllm.v1.worker.gpu.model_runner"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = cls
    assert v032.install_prefill_graph_guard()
    wrapped = cls.gather_batch_req_state
    assert v032.install_prefill_graph_guard() and cls.gather_batch_req_state is wrapped
    cls.gather_batch_req_state = lambda *a, **kw: None
    assert not v032.install_prefill_graph_guard()
