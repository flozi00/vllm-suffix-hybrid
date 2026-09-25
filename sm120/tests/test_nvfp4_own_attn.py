# SPDX-License-Identifier: Apache-2.0
"""CPU tests for K2-NVFP4 (our SM120 NVFP4-KV decode/spec-verify kernel).

* numpy kernel twin (tile algorithm of src/nvfp4_attn_gpu.rs) == independent
  float64 reference on the served shapes (gemma-4 hd512 16/2, SWA hd256 16/8
  with window, qwen3.8-27b hd256 24/4), decode q_len 1 and MTP verify q_len 9,
  ragged/padded batches, NaN-pattern stale tail bytes, split-count invariance;
* the numpy reference == the on-silicon oracle's torch reference (so the
  oracle and these tests judge against the same math);
* patch anchors H19/H20 + helper gate semantics (default OFF, fail closed);
* GPU test (skips without CUDA + feature build).
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))

from nvfp4_kv_patch import own_attn  # noqa: E402
from nvfp4_kv_patch import own_attn_ref as ref  # noqa: E402

_native = pytest.importorskip("suffix_hybrid._native")
if not hasattr(_native, "nvfp4_attn_plan"):
    pytest.skip("native module predates K2-NVFP4 (rebuild suffix_hybrid._native)",
                allow_module_level=True)


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "nvfp4_kv_patch_own_t", REPO / "sm120" / "nvfp4_kv_patch" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = _load_patch()
FIXTURE = REPO / "sm120" / "tests" / "fixtures" / "vllm_0.30.0" / "flashinfer_backend.py"


def _case(rng, d, hq, hkv, page, q_len, kv_lens, nan_tail=False):
    pages_per = [max(1, math.ceil(n / page)) for n in kv_lens]
    num_pages = sum(pages_per) + 2
    cache = ref.make_cache(rng, num_pages, hkv, page, d)
    perm = rng.permutation(num_pages)
    bt = np.zeros((len(kv_lens), max(pages_per) + 1), dtype=np.int64)
    cur = 0
    for b, p in enumerate(pages_per):
        bt[b, :p] = perm[cur:cur + p]
        cur += p
        n = kv_lens[b]
        if nan_tail and n % page:
            last = bt[b, (n - 1) // page]
            for sf in (cache.k_sf, cache.v_sf):
                # stale rows past kv_len: e4m3 NaN code (0x7F). V is
                # swizzled, so poison the whole tail group region generously
                # only where it cannot hit live tokens: rows >= next 4-group.
                start = n % page
                if sf is cache.k_sf:
                    sf[last, :, start:] = 0x7F
                else:
                    g4 = -(-start // 4) * 4
                    flat = sf[last].reshape(hkv, -1)
                    flat[:, g4 * (d // 16):] = 0x7F
            cache.v_data[last, :, n % page:] = 0xFF
    # bf16 queries, as served (and as the on-silicon oracle feeds them)
    q = ref.bf16(rng.standard_normal((len(kv_lens) * q_len, hq, d)))
    return cache, bt, np.array(kv_lens), q


def _cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# (d, hq, hkv, page, q_len, kv_lens, window_left)
SHAPES = [
    ("gemma_hd512_decode", 512, 16, 2, 16, 1, (1, 17, 300), -1),
    ("gemma_hd512_verify_k8", 512, 16, 2, 16, 9, (9, 40, 333), -1),
    ("gemma_swa_hd256_verify_win", 256, 16, 8, 16, 9, (9, 200, 129), 63),
    ("gemma_swa_hd256_decode_p64", 256, 16, 8, 64, 1, (70, 250), 127),
    ("qwen_hd256_g6_decode", 256, 24, 4, 32, 1, (5, 100), -1),
]


@pytest.mark.parametrize("name,d,hq,hkv,page,q_len,kv_lens,wl", SHAPES,
                         ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("sms", [1, 188])
def test_twin_matches_reference(name, d, hq, hkv, page, q_len, kv_lens, wl,
                                sms):
    rng = np.random.default_rng(len(name) * 7 + sms)
    cache, bt, sl, q = _case(rng, d, hq, hkv, page, q_len, kv_lens,
                             nan_tail=True)
    plan = _native.nvfp4_attn_plan(len(kv_lens), q_len, hq, hkv, d, page, sms)
    sm = 1.0 / math.sqrt(d)
    want = ref.reference(q, cache, bt, sl, q_len, sm, 0.5, 2.0, wl)
    got = ref.kernel_twin(q, cache, bt, sl, q_len, sm, plan, 0.5, 2.0, wl)
    assert np.isfinite(got).all()
    # FA2-class: on-silicon FA2 measured cos >= 0.99999, rel 0.0025-0.0055
    # (mtrace/gemma-spec-dev-nvfp4-oracle-4d650e38.log); oracle gate 2e-2.
    assert _cos(got, want) >= 0.99995, (name, _cos(got, want))
    assert _rel(got, want) <= 8e-3, (name, _rel(got, want))


def test_padded_rows_are_zero_and_split_invariant():
    rng = np.random.default_rng(3)
    cache, bt, sl, q = _case(rng, 512, 16, 2, 16, 1, (0, 50, 0))
    outs = []
    for sms in (1, 188):
        plan = _native.nvfp4_attn_plan(3, 1, 16, 2, 512, 16, sms)
        outs.append(ref.kernel_twin(q, cache, bt, sl, 1, 0.05, plan))
    assert outs[0][0].max() == 0 and outs[0][2].max() == 0
    assert _cos(outs[0], outs[1]) > 0.9999


def test_numpy_reference_matches_oracle_torch_reference():
    torch = pytest.importorskip("torch")
    from nvfp4_kv_patch.oracle import _ref_attn, dequant_side

    rng = np.random.default_rng(11)
    d, hkv, page = 256, 2, 16
    cache, bt, sl, q = _case(rng, d, 8, hkv, page, 3, (40,))
    t = torch.from_numpy
    k = dequant_side(t(cache.k_data), t(cache.k_sf.copy()).view(torch.float8_e4m3fn),
                     False, 1.0).double()
    v = dequant_side(t(cache.v_data), t(cache.v_sf.copy()).view(torch.float8_e4m3fn),
                     True, 1.0).double()
    pages = t(bt[0, :3])
    kd = k[pages].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:40]
    vd = v[pages].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:40]
    want = _ref_attn(t(q).float(), kd.float(), vd.float(), 0.1).numpy()
    got = ref.reference(q, cache, bt, sl, 3, 0.1)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=2e-3)  # fp32 torch
    # e4m3 decoder agrees with torch's float8_e4m3fn on every byte
    allb = torch.arange(256, dtype=torch.uint8)
    np.testing.assert_array_equal(
        np.nan_to_num(ref.e4m3(np.arange(256)), nan=123.0),
        np.nan_to_num(allb.view(torch.float8_e4m3fn).double().numpy(), nan=123.0))


# ---------------------------------------------------------------------------
# Patch anchors + gates
# ---------------------------------------------------------------------------

def test_h19_h20_anchor_replay():
    new, applied = PATCH.patch_backend_source(FIXTURE.read_text())
    assert "own_attn_spec_as_decode" in applied
    assert "own_attn_decode_wrapper" in applied
    assert "self.use_own_nvfp4_attn = _nvfp4_own_attn_gate(self)" in new
    assert "or self.use_own_nvfp4_attn\n" in new
    assert new.count("def _nvfp4_own_attn_gate") == 1
    # our decode branch precedes the stock fa2 plan branch
    assert new.index("_nvfp4_own_attn_decode(\n") < new.index(
        "pure_decode = num_prefills == 0")
    # forward() is untouched: the stock decode_wrapper.run call remains
    assert new.count("decode_wrapper.run(") == 2


class _FIDecode:
    def __init__(self, wrapper):
        self.wrapper = wrapper


def _helper_ns(**extra):
    from types import SimpleNamespace

    import logging

    log = logging.getLogger("t")
    log.warning_once = log.warning
    ns = {"FIDecode": _FIDecode, "FlashInferImpl": None, "logger": log,
          "get_per_layer_parameters": lambda *a: None,
          "infer_global_hyperparameters": lambda _p: SimpleNamespace(
              window_left=-1, logits_soft_cap=None)}
    ns.update(extra)
    exec(compile(PATCH._OWN_ATTN_HELPER_SRC, "<own>", "exec"), ns)
    return ns


def test_gate_default_off_and_fail_closed(monkeypatch):
    monkeypatch.delenv(own_attn.ENV, raising=False)
    ns = _helper_ns()
    assert ns["_nvfp4_own_attn_gate"](object()) is False
    monkeypatch.setenv(own_attn.ENV, "1")
    with pytest.raises(RuntimeError, match="not injected"):
        ns["_nvfp4_own_attn_gate"](object())

    class B:
        use_fa2_nvfp4_kv = True
        vllm_config, layer_names = None, []

    if not getattr(_native, "HAS_NVFP4_ATTN_CUDA", False):
        with pytest.raises(RuntimeError, match="nvfp4-attn-kernels"):
            _helper_ns(_nvfp4_own_attn=own_attn)["_nvfp4_own_attn_gate"](B())
    B.use_fa2_nvfp4_kv = False
    assert own_attn.builder_gate(B(), -1, None) is False


def test_decode_metadata_q_len_rules():
    torch = pytest.importorskip("torch")
    ns = _helper_ns(_nvfp4_own_attn=own_attn)

    class B:
        num_qo_heads, num_kv_heads, head_dim, page_size = 16, 2, 512, 16
        window_left, sm_scale, logits_soft_cap = -1, 1.0, None

    bt = torch.zeros(4, 8, dtype=torch.int32)
    sl = torch.ones(4, dtype=torch.int32)
    mk = ns["_nvfp4_own_attn_decode"]
    w = mk(B, bt, sl, torch.tensor([0, 9, 18, 27, 36]), 4).wrapper
    assert w.q_len == 9 and w.block_table.shape[0] == 4
    w = mk(B, bt, sl, torch.tensor([0, 9, 18, 18, 18]), 4).wrapper  # cg pad
    assert w.q_len == 9
    with pytest.raises(RuntimeError, match="uniform"):
        mk(B, bt, sl, torch.tensor([0, 9, 10, 19, 28]), 4)
    B.logits_soft_cap = 30.0
    with pytest.raises(ValueError, match="soft-capping"):
        mk(B, bt, sl, torch.tensor([0, 1, 2, 3, 4]), 4)


def test_wrapper_rejects_unsupported_run_args():
    w = own_attn.DecodeWrapper(None, None, 1, 16, 2, 512, 16, -1, 1.0, None)
    with pytest.raises(ValueError, match="sinks"):
        w.run(None, (None, None), sinks=object(), out=object(),
              kv_cache_sf=(None, None))
    with pytest.raises(ValueError, match="kv_cache_sf"):
        w.run(None, (None, None), out=object())


def test_max_decode_q_len_mirrors_vllm_threshold():
    class S:
        num_speculative_tokens, parallel_drafting = 8, False

    class C:
        speculative_config = S

    assert own_attn.max_decode_q_len(C) == 9
    S.parallel_drafting = True
    assert own_attn.max_decode_q_len(C) == 17
    C.speculative_config = None
    assert own_attn.max_decode_q_len(C) == 1


# ---------------------------------------------------------------------------
# GPU (skips cleanly without CUDA + the feature build)
# ---------------------------------------------------------------------------

def test_gpu_kernel_matches_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(
            _native, "HAS_NVFP4_ATTN_CUDA", False):
        pytest.skip("needs CUDA + suffix_hybrid built with nvfp4-attn-kernels")
    rng = np.random.default_rng(5)
    d, hq, hkv, page, q_len = 512, 16, 2, 16, 9
    cache, bt, sl, q = _case(rng, d, hq, hkv, page, q_len, (9, 40, 333))
    dev = "cuda"
    raw = torch.from_numpy(cache.raw).to(dev)
    c2 = ref.Cache(raw, hkv, page, d)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=dev)
    own_attn.run(torch.from_numpy(q).to(dev, torch.bfloat16), c2.k_data,
                 c2.k_sf.view(torch.float8_e4m3fn), c2.v_data,
                 c2.v_sf.view(torch.float8_e4m3fn),
                 torch.from_numpy(bt).to(dev, torch.int32),
                 torch.from_numpy(sl).to(dev, torch.int32), out, q_len, -1,
                 1 / math.sqrt(d), 1.0)
    want = ref.reference(ref.bf16(q), cache, bt, sl, q_len, 1 / math.sqrt(d))
    got = out.float().cpu().numpy()
    assert _cos(got, want) >= 0.9995 and _rel(got, want) <= 2e-2


# ---------------------------------------------------------------------------
# UNIFORM_BATCH graphs (H13) + the multimodal guard
# ---------------------------------------------------------------------------

def test_h13_uniform_batch_only_behind_own_attn_and_h21():
    new, applied = PATCH.patch_backend_source(FIXTURE.read_text())
    i = new.index('"""Get the cudagraph support level for FlashInfer attention."""')
    block = new[i:i + 700]
    assert block.index("_nvfp4_own_attn_graphs_ok()") < block.index(
        "return AttentionCGSupport.UNIFORM_BATCH") < block.index(
        "return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE")
    assert "own_attn_no_paged_indices" in applied
    assert 'and not getattr(self, "use_own_nvfp4_attn", False)\n        )' in new


def _uniform_ok(has_prefill_rule):
    def get_uniform_decode_token_count(num_reqs, num_tokens, max_query_len,
                                       has_prefill):
        pass
    src = ("def get_uniform_decode_token_count(n, t, q, has_prefill):\n"
           + ("    if not has_prefill and is_uniform(n, t, q):\n"
              if has_prefill_rule else "    if is_uniform(n, t, q):\n")
           + "        return q\n    return None\n")
    ns = {}
    exec(compile(src, "<fake_utils>", "exec"), ns)
    import linecache
    linecache.cache["<fake_utils>"] = (len(src), None, src.splitlines(True),
                                       "<fake_utils>")
    return ns["get_uniform_decode_token_count"]


def test_mm_guard_requires_v2_prefill_exclusion(monkeypatch):
    """A uniform-shaped batch holding a still-prefilling (image) chunk must
    never replay a causal FULL graph: UNIFORM_BATCH is only claimed when the
    runner provably excludes prefilling batches from uniform-decode."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    assert own_attn.uniform_decode_excludes_prefill(_uniform_ok(True))
    assert not own_attn.uniform_decode_excludes_prefill(_uniform_ok(False))
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert not own_attn.uniform_decode_excludes_prefill(_uniform_ok(True))
    # pinned vLLM 0.30.0 text, if the source tree is around
    real = Path("/tmp/vllm-src-kv/vllm/v1/worker/utils.py")
    if real.is_file():
        assert "if not has_prefill and is_uniform_query_len(" in real.read_text()


def test_graphs_ok_helper_gates(monkeypatch):
    monkeypatch.delenv(own_attn.ENV, raising=False)
    ns = _helper_ns(_nvfp4_own_attn=own_attn)
    assert ns["_nvfp4_own_attn_graphs_ok"]() is False  # K2 not armed
    monkeypatch.setenv(own_attn.ENV, "1")
    monkeypatch.setattr(own_attn, "uniform_decode_excludes_prefill",
                        lambda: False)
    assert ns["_nvfp4_own_attn_graphs_ok"]() is False  # invariant missing
    monkeypatch.setattr(own_attn, "uniform_decode_excludes_prefill",
                        lambda: True)
    assert ns["_nvfp4_own_attn_graphs_ok"]() is True
