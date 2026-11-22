# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream replay gates for vLLM 0.34.0 (+ FlashInfer 0.7.3).

- flashinfer backend: the 0.33.0 anchor set applies 19/19 unchanged
  (``own_attn_spec_as_decode`` keeps its 0.33.0 re-anchor).
- gather_batch_req_state: byte-identical (sha256 18da780f...).
- model_loader utils: the attention + model-level post-load hooks moved into
  ``finalize_model_weights``; the dense conversion is re-anchored after it.
- cudagraph_utils: uniform decode graphs are keyed by query length
  (``decode_max_reqs`` -> ``_decode_descs``); the K2 every-width graphs are
  re-ported onto that table and replayed here through the REAL 0.34.0
  ``_init_candidates`` / ``dispatch`` methods.
"""
import __future__
import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sm120")]
import fp8_dense_patch as dense  # noqa: E402
import nvfp4_kv_patch as patch  # noqa: E402
from nvfp4_kv_patch import own_attn, v032  # noqa: E402
from suffix_hybrid import rocm_patches as rp  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/vllm_0.34.0"
V034 = "0.34.0"


def text(name):
    return (FIXTURE / name).read_text()


def test_pinned_fixture_hashes():
    expected = {
        "flashinfer_backend.py":
            "be0934b213f0d05d40b8a8520a8cd71965fb2fc71edd7c770dd4487c0ee42d83",
        "model_loader_utils.py":
            "593685e149d001e9ee16b03f7008f6682fb6479a6ce0af0cd147b7afec8c5c0f",
        "gather_batch_req_state.py.txt":
            "18da780f3c3ffdea5e38b6d663083792796b92e8bfe2159ae16b76d0af9db7ca",
        "flashinfer_cutlass_moe.py":
            "8524cb10017008920b92b7ab7670e4a187967c5f971185b2bc21cf99a4113e81",
        "gpu_cudagraph_utils.py":
            "e2f3ad9e831fa261291a04062f15d611fc8e1d2dbefa58786fb5b1158c90ecfc",
        "config_compilation.py":
            "53c277c61273721cdab1c4f5b0954bdcd26765e9c26bec0b1e018f185c7f21cc",
        "lfm2_moe.py":
            "712d9ac8d98e121e323b06739f6c35eb7f6a41267d2e2b0cb9a1d29d2034e0b1",
    }
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in FIXTURE.iterdir()} == expected


# ---- NVFP4-KV backend adapter ---------------------------------------------

def test_adapter_table_accepts_the_shipped_pair():
    assert (V034, "0.7.3") in patch.NATIVE_ADAPTERS
    # A 0.34.0 image with an older FlashInfer must not take the native path.
    assert (V034, "0.7.2.post1") not in patch.NATIVE_ADAPTERS
    assert ("0.33.0", "0.7.3") not in patch.NATIVE_ADAPTERS


def test_v034_replay_applies_every_anchor():
    out, applied = v032.patch_backend_source(text("flashinfer_backend.py"), patch, V034)
    assert len(applied) == 19
    assert "own_attn_spec_as_decode" in applied
    # The own-attn lane is armed by this anchor and by nothing else.
    assert "self.use_own_nvfp4_attn = _nvfp4_own_attn_gate(self)" in out
    assert "or self.use_own_nvfp4_attn" in out
    compile(out, "<patched_v034>", "exec")


def test_v032_anchor_set_does_not_apply_to_v034():
    with pytest.raises(patch.PatchDriftError, match="own_attn_spec_as_decode"):
        v032.patch_backend_source(text("flashinfer_backend.py"), patch, "0.32.0")


def test_v034_replay_is_not_idempotent():
    out, _ = v032.patch_backend_source(text("flashinfer_backend.py"), patch, V034)
    with pytest.raises(patch.PatchDriftError):
        v032.patch_backend_source(out, patch, V034)


def test_gather_guard_hash_is_unchanged_in_v034():
    src = text("gather_batch_req_state.py.txt")
    assert hashlib.sha256(src.encode()).hexdigest() == (
        "18da780f3c3ffdea5e38b6d663083792796b92e8bfe2159ae16b76d0af9db7ca")


def _anchor_func():
    if decode_graph_eligible and is_uniform_query_len(  # noqa: F821
            1, 1, 1):
        return 1


@pytest.mark.parametrize("ver,armed", [("0.34.0", True), ("0.34.0+cu130", True),
                                       ("0.33.0", True), ("0.35.0", False)])
def test_uniform_graph_tier_admits_v034_only_by_pin(monkeypatch, ver, armed):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    fake = ModuleType("vllm")
    fake.__version__ = ver
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(v032, "install_prefill_graph_guard", lambda: True)
    assert own_attn.uniform_decode_excludes_prefill(_anchor_func) is armed


# ---- dense post-load conversion --------------------------------------------

def test_dense_loader_hook_reanchored_on_v034():
    assert V034 in dense.SUPPORTED_VLLM
    src = text("model_loader_utils.py")
    with pytest.raises(dense.PatchDriftError, match="convert_after_post_load"):
        dense.patch_source(src)  # the 0.30..0.33 anchor is gone
    out, applied = dense.patch_source(src, V034)
    assert applied == ["convert_after_finalize_model_weights"]
    tree = ast.parse(out)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "process_weights_after_loading")
    calls = [n.value.func.id for n in fn.body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name)]
    # The conversion runs right after every module's own post-load hook
    # (attention layers + the model-level one, which finalize runs last).
    i = calls.index("finalize_model_weights")
    assert calls[i + 1] == "_suffix_fp8_dense_convert"
    with pytest.raises(dense.PatchDriftError):
        dense.patch_source(out, V034)


# ---- rocm_patches gates the pools set on 0.34.0 --------------------------

@pytest.mark.parametrize("gate,fixture", [
    ("SUFFIX_NVFP4_MOE_ORDERED_FINALIZE", "flashinfer_cutlass_moe.py"),
    ("SUFFIX_SHORTCONV_FP8", "lfm2_moe.py"),
])
def test_source_patch_anchors_apply_on_v034(gate, fixture):
    p = rp.PATCHES[gate]
    assert text(fixture).count(p.old) == 1
    out = rp.patch_source(p, text(fixture))
    assert out.count(p.new) == 1
    compile(out, f"<{gate}>", "exec")


def test_shortconv_hook_is_the_root_post_load_hook_on_v034():
    """0.34.0 calls the model-level hook from finalize_model_weights: every
    module's hook, children first, so the CausalLM root (ours) runs last."""
    src = text("model_loader_utils.py")
    assert "    finalize_model_weights(model, target_device)\n" in src
    out = rp.patch_source(rp.PATCHES["SUFFIX_SHORTCONV_FP8"], text("lfm2_moe.py"))
    cls = next(n for n in ast.parse(out).body
               if isinstance(n, ast.ClassDef) and n.name == "Lfm2MoeForCausalLM")
    assert [f.name for f in cls.body if isinstance(f, ast.FunctionDef)
            and f.name == "process_weights_after_loading"] == [
                "process_weights_after_loading"]


# ---- K2 every-width FULL graphs on the 0.34.0 graph manager ----------------

def _pick(tree, names):
    return [n for n in tree.body if getattr(n, "name", None) in names]


def _real_manager():
    """CudaGraphManager with the REAL 0.34.0 _init_candidates / dispatch /
    helpers (exec'd from the pinned fixtures); __init__'s attribute setup is
    mirrored and its order checked separately."""
    cc = ast.parse(text("config_compilation.py"))
    cg = ast.parse(text("gpu_cudagraph_utils.py"))
    ns = {}
    import bisect
    import enum
    import math
    from collections import defaultdict
    from dataclasses import dataclass, replace
    from itertools import groupby, product
    ns.update(bisect=bisect, enum=enum, math=math, defaultdict=defaultdict,
              dataclass=dataclass, replace=replace, groupby=groupby, product=product,
              cdiv=lambda a, b: -(-a // b))
    mod = ast.Module(body=_pick(cc, {"CUDAGraphMode", "decode_cudagraph_capture_shapes",
                                     "decode_cudagraph_capture_max_tokens"})
                     + _pick(cg, {"BatchExecutionDescriptor", "_is_compatible"}),
                     type_ignores=[])
    lazy = __future__.annotations.compiler_flag  # upstream: postponed annotations
    exec(compile(mod, "<pinned 0.34.0>", "exec", flags=lazy, dont_inherit=True), ns)
    mgr_cls = next(n for n in cg.body if isinstance(n, ast.ClassDef)
                   and n.name == "CudaGraphManager")
    meths = [f for f in mgr_cls.body if isinstance(f, ast.FunctionDef) and f.name in {
        "_init_candidates", "dispatch", "_resolve_effective_loras", "_maybe_ubatch_twin"}]
    body = ast.Module(body=[ast.ClassDef(
        name="Mgr", bases=[], keywords=[], body=meths, decorator_list=[],
        type_params=[])], type_ignores=[])
    ast.fix_missing_locations(body)
    exec(compile(body, "<pinned 0.34.0 CudaGraphManager>", "exec", flags=lazy,
                 dont_inherit=True), ns)
    return ns["Mgr"], ns["CUDAGraphMode"]


def _make(cls, mode, dq, *, max_reqs=8, sizes=(1, 2, 4, 8), max_tokens=64, spec=None):
    m = cls.__new__(cls)
    m.vllm_config = SimpleNamespace(
        speculative_config=spec, uniform_decode_query_len=dq,
        cache_config=SimpleNamespace(use_kda_recoverssm=False))
    m.compilation_config = SimpleNamespace(
        prefill_cudagraph_capture_token_sizes=[1, 2, 4, 8, 16, 32, 64],
        decode_cudagraph_capture_request_sizes=list(sizes),
        max_cudagraph_capture_size=max_tokens,
        max_decode_cudagraph_capture_tokens=max_tokens)
    m.decode_request_sizes = None
    m.max_num_reqs = max_reqs
    m.cudagraph_mode = mode
    # __init__: {q: min(max_reqs, max_num_reqs)} sorted; dq = max(keys)
    m.decode_max_reqs = {dq: max_reqs}
    m.decode_query_len = max(m.decode_max_reqs)
    m.varlen_decode = False
    m.ubatch_runner = None
    m.lora_capture_cases = [0]
    m._lora_dispatch_map, m._max_lora_case = {}, 0
    m._graphs_captured = True
    m._candidates, m._decode_descs, m._capture_descs = {}, {}, {}
    m._init_candidates()
    return m


def test_v034_manager_sets_decode_lens_before_building_candidates():
    """The port mutates decode_max_reqs inside the _init_candidates wrapper;
    that is only sound if __init__ fills it first and builds once after."""
    cg = ast.parse(text("gpu_cudagraph_utils.py"))
    cls = next(n for n in cg.body if isinstance(n, ast.ClassDef)
               and n.name == "CudaGraphManager")
    init = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
    src = ast.unparse(init)
    assert src.index("self.decode_max_reqs = {") < src.index(
        "self.decode_query_len = max(self.decode_max_reqs)") < src.index(
        "self._init_candidates()")
    assert src.count("self._init_candidates()") == 1
    ic = ast.unparse(next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                          and f.name == "_init_candidates"))
    assert "self.decode_max_reqs" in ic and "self._decode_descs[decode_key] = table" in ic


def test_v034_every_width_graphs_on_the_real_builder():
    Mgr, Mode = _real_manager()
    mode = Mode.FULL_AND_PIECEWISE

    class Ours(Mgr):
        pass

    assert own_attn.wrap_init_candidates(Ours, Mode, V034)
    assert own_attn.wrap_init_candidates(Ours, Mode, V034)  # idempotent
    stock, mgr = _make(Mgr, mode, 4), _make(Ours, mode, 4)
    assert mgr.decode_query_len == stock.decode_query_len == 4
    assert sorted(k[0] for k in stock._decode_descs) == [4]
    assert sorted(k[0] for k in mgr._decode_descs) == [1, 2, 3, 4]
    assert mgr._decode_descs[(4, 0, 1)] == stock._decode_descs[(4, 0, 1)]
    full = mgr._capture_descs[Mode.FULL]
    assert set(stock._capture_descs[Mode.FULL]) <= set(full)
    assert len(full) == len(set(full))
    assert mgr._capture_descs[Mode.PIECEWISE] == stock._capture_descs[Mode.PIECEWISE]
    assert mgr._candidates == stock._candidates  # token-keyed table untouched
    # stock: a uniform q_len-1 batch had no FULL graph; ours replays one
    assert stock.dispatch(3, 3, 1, 0).cg_mode == Mode.PIECEWISE
    for reqs, q in ((1, 1), (3, 1), (8, 1), (2, 2), (5, 3), (8, 4)):
        d = mgr.dispatch(reqs, reqs * q, q, 0)
        assert d.cg_mode == Mode.FULL and d.uniform_token_count == q, (reqs, q)
        assert d.num_reqs >= reqs and d.num_tokens == d.num_reqs * q
    # mixed (non-uniform) batches keep the stock pick
    assert mgr.dispatch(3, 5, None, 0) == stock.dispatch(3, 5, None, 0)


def test_v034_every_width_inert_cases():
    Mgr, Mode = _real_manager()

    class Ours(Mgr):
        pass

    own_attn.wrap_init_candidates(Ours, Mode, V034)
    # no spec decode (q = 1), piecewise-only decode, dynamic spec: unchanged
    assert sorted(k[0] for k in _make(Ours, Mode.FULL_AND_PIECEWISE, 1)._decode_descs) == [1]
    pw = _make(Ours, Mode.PIECEWISE, 4)
    assert pw._decode_descs == _make(Mgr, Mode.PIECEWISE, 4)._decode_descs
    dyn = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: True)
    assert sorted(k[0] for k in _make(Ours, Mode.FULL_AND_PIECEWISE, 4,
                                     spec=dyn)._decode_descs) == [4]


def test_v034_wrapper_fails_loud_on_a_pre_034_manager():
    """A 0.34.0 pin against a manager without decode_max_reqs is a startup
    error, never a silent 1+k-only graph set."""
    Mgr, Mode = _real_manager()

    class Old(Mgr):
        pass

    own_attn.wrap_init_candidates(Old, Mode, V034)
    m = Old.__new__(Old)
    m.vllm_config = SimpleNamespace(speculative_config=None)
    m.cudagraph_mode, m.decode_query_len, m.varlen_decode = Mode.FULL_AND_PIECEWISE, 4, False
    with pytest.raises(AttributeError, match="decode_max_reqs"):
        m._init_candidates()
