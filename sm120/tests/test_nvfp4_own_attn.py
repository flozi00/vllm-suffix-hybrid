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
    # our decode branch precedes the stock fa2 plan branch and falls through
    # to it when the helper returns None (q_len 1)
    assert new.index("_nvfp4_own_attn_decode(\n") < new.index(
        "pure_decode = num_prefills == 0")
    assert ") is not None:\n                # decode rows (ragged ok) -> K2" in new
    compile(new, "<patched>", "exec")
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


def test_decode_metadata_routing_rules():
    torch = pytest.importorskip("torch")
    ns = _helper_ns(_nvfp4_own_attn=own_attn)

    class B:
        num_qo_heads, num_kv_heads, head_dim, page_size = 16, 2, 512, 16
        window_left, sm_scale, logits_soft_cap = -1, 1.0, None

    n = 12
    bt = torch.zeros(n, 8, dtype=torch.int32)
    sl = torch.ones(n, dtype=torch.int32)
    mk = ns["_nvfp4_own_attn_decode"]

    def qo(*lens):
        return torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)

    q = qo(9, 9, 9, 9)
    w = mk(B, bt, sl, q, q, 4).wrapper
    assert w.q_max == 9 and w.block_table.shape[0] == 4
    assert w.qo_indptr.shape[0] == 5
    w = mk(B, bt, sl, qo(9, 9, 0, 0), qo(9, 9, 0, 0), 4).wrapper  # cg pad
    assert w.q_max == 9
    # ragged widths (suffix drafts, misses as q_len 1) -> K2, q_max bound
    w = mk(B, bt, sl, qo(1, 9, 3, 1), qo(1, 9, 3, 1), 4).wrapper
    assert w.q_max == 9 and w.seq_lens.shape[0] == 4 and w.max_rows == 48
    w = mk(B, bt, sl, qo(1, 1, 3, 1), qo(1, 1, 3, 1), 4).wrapper
    assert w.q_max == 3 and w.max_rows == 16  # miss-dominated: light tiles
    assert own_attn.row_budget([9, 9, 0]) == 48 and own_attn.row_budget([]) == 48
    # plain q_len-1 batches: K2 up to K2_Q1_MAX_BATCH, fa2 beyond
    k = own_attn.K2_Q1_MAX_BATCH
    ones = qo(*([1] * k))
    assert mk(B, bt, sl, ones, ones, k).wrapper.q_max == 1
    wide = qo(*([1] * (k + 1)))
    assert mk(B, bt, sl, wide, wide, k + 1) is None
    takes = ns["_nvfp4_own_attn_takes"]
    assert takes(wide, k + 1) is False and takes(qo(1, 2), 2) is True
    assert takes(torch.tensor([0]), 0) is False
    B.logits_soft_cap = 30.0
    with pytest.raises(ValueError, match="soft-capping"):
        mk(B, bt, sl, qo(2, 2), qo(2, 2), 2)


@pytest.mark.parametrize("sms", [1, 188])
def test_twin_ragged_matches_reference(sms):
    """Ragged q lengths (suffix drafts: misses q_len 1, partial and full
    verifies) through the qo_indptr indexing, plus a padding request."""
    rng = np.random.default_rng(21 + sms)
    for d, hq, hkv, page, wl in ((512, 16, 2, 16, -1), (256, 16, 8, 64, 63)):
        lens = [1, 9, 3, 1, 0]
        kv = (70, 130, 9, 1, 0)
        cache, bt, sl, _ = _case(rng, d, hq, hkv, page, 1, kv)
        q = ref.bf16(rng.standard_normal((sum(lens), hq, d)))
        sm = 1.0 / math.sqrt(d)
        want = ref.reference(q, cache, bt, sl, lens, sm, 0.5, 2.0, wl)
        for cap in (16, 48):
            plan = _native.nvfp4_attn_plan(len(lens), max(lens), hq, hkv, d,
                                           page, sms, cap)
            assert plan["m"] <= cap
            got = ref.kernel_twin(q, cache, bt, sl, lens, sm, plan, 0.5, 2.0,
                                  wl)
            assert _cos(got, want) >= 0.99995 and _rel(got, want) <= 8e-3
        # each request equals its own uniform run (rows are independent)
        off = 0
        for b, ql in enumerate(lens[:-1]):
            one = ref.reference(q[off:off + ql], cache, bt[b:b + 1],
                                sl[b:b + 1], ql, sm, 0.5, 2.0, wl)
            assert np.allclose(want[off:off + ql], one)
            off += ql


def test_exact_seq_lens_helper(monkeypatch):
    torch = pytest.importorskip("torch")
    import contextlib
    from types import SimpleNamespace as NS

    ns = _helper_ns(_nvfp4_own_attn=own_attn, torch=torch,
                    gpu_sync_allowed=contextlib.nullcontext)
    ns["logger"].info = lambda *a: None
    exact = ns["_nvfp4_exact_seq_lens_cpu"]
    ub = torch.tensor([5, 9, 0], dtype=torch.int32)
    cm = NS(seq_lens=ub.clone(), seq_lens_cpu_upper_bound=ub)
    b = NS(use_dcp=False, vllm_config=NS(
        scheduler_config=NS(async_scheduling=False)))
    assert exact(b, cm) is ub
    b.vllm_config.scheduler_config.async_scheduling = True
    assert exact(b, cm) is None  # async: upper bound is only a bound
    b.vllm_config.scheduler_config.async_scheduling = False
    assert exact(b, NS(seq_lens=ub, seq_lens_cpu_upper_bound=None)) is None
    b.use_dcp = True
    assert exact(b, cm) is None
    b.use_dcp = False
    bad = NS(seq_lens=ub + 1, seq_lens_cpu_upper_bound=ub)
    with pytest.raises(RuntimeError, match="refusing"):
        exact(b, bad)  # sampled check fails closed
    ns["_NVFP4_SEQ_LENS_CHECKS"][0] = 100
    assert exact(b, bad) is ub  # unsampled build: trusts the contract


def test_h23_h24_anchors():
    new, applied = PATCH.patch_backend_source(FIXTURE.read_text())
    assert {"exact_seq_lens_no_sync", "own_attn_ragged_split"} <= set(applied)
    i = new.index("seq_lens_cpu = common_attn_metadata.seq_lens.cpu()")
    block = new[i - 400:i]
    assert "_nvfp4_exact_seq_lens_cpu(self, common_attn_metadata)" in block
    assert 'getattr(self, "use_fa2_nvfp4_kv", False)' in block
    assert "if seq_lens_cpu is None:" in block
    assert ('require_uniform=not (\n                        self.use_xqa\n'
            '                        or getattr(self, "use_own_nvfp4_attn", False)),'
            in new)
    compile(new, "<patched>", "exec")


def test_wrapper_rejects_unsupported_run_args():
    w = own_attn.DecodeWrapper(None, None, None, 1, 16, 2, 512, 16, -1, 1.0,
                               None, 16)
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
                 torch.from_numpy(sl).to(dev, torch.int32),
                 torch.arange(4, dtype=torch.int32, device=dev) * q_len, out,
                 q_len, -1, 1 / math.sqrt(d), 1.0)
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
    assert ('and not (getattr(self, "use_own_nvfp4_attn", False)\n'
            '                     and _nvfp4_own_attn_takes(qo_indptr_cpu, num_decodes))'
            in new)


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


def test_h22_skips_seq_lens_sync_only_for_k2_only_batches():
    new, applied = PATCH.patch_backend_source(FIXTURE.read_text())
    assert "own_attn_no_seq_lens_sync" in applied
    i = new.index("needs_seq_lens_cpu = self.use_dcp or use_cascade")
    block = new[i:i + 500]
    for cond in ("num_prefills == 0", "not self.use_dcp", "not use_cascade",
                 'getattr(self, "use_own_nvfp4_attn", False)',
                 "_nvfp4_own_attn_takes(qo_indptr_cpu, num_decodes)"):
        assert cond in block, cond
    # decided before the seq_lens.cpu() read it guards
    assert i < new.index("seq_lens_cpu = common_attn_metadata.seq_lens.cpu()")


class _Mode:
    """Stand-in for vLLM CUDAGraphMode: FULL_AND_PIECEWISE = (FULL, PW)."""
    FULL, PIECEWISE = "FULL", "PIECEWISE"

    def __init__(self, decode):
        self.decode = decode

    def __bool__(self):
        return True

    def separate_routine(self):
        return True

    def decode_mode(self):
        return self.decode


class _Desc:
    def __init__(self, cg_mode, num_tokens, num_reqs=None, u=None):
        self.cg_mode, self.num_tokens = cg_mode, num_tokens
        self.num_reqs, self.uniform_token_count = num_reqs, u

    def _k(self):
        return (self.cg_mode, self.num_tokens, self.num_reqs,
                self.uniform_token_count)

    def __eq__(self, o):
        return self._k() == o._k()

    def __hash__(self):
        return hash(self._k())


class _Mgr:
    """Mirrors vllm 0.30.0 CudaGraphManager._init_candidates (separate
    decode routine, non-varlen, non-dynamic spec)."""
    sizes = [1, 2, 4, 8, 9, 16, 18, 24, 32]
    max_num_reqs = 4

    def __init__(self, dq, decode=_Mode.FULL):
        from types import SimpleNamespace
        self.vllm_config = SimpleNamespace(speculative_config=None)
        self.cudagraph_mode = _Mode(decode)
        self.decode_query_len, self.varlen_decode = dq, False
        self._capture_descs, self._candidates = {}, {}
        self._init_candidates()

    def _init_candidates(self):
        from collections import defaultdict
        from itertools import groupby
        by = defaultdict(list)
        u = self.decode_query_len
        for n in self.sizes:
            r = -(-n // u) * u
            if r <= self.max_num_reqs * u:
                d = _Desc("FULL", r, r // u, u)
                if d not in by["FULL"]:
                    by["FULL"].append(d)
            by["PIECEWISE"].append(_Desc("PIECEWISE", n))
        for m, ds in by.items():
            ds.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[m] = ds
        for m in ("FULL", "PIECEWISE"):
            start = 0
            for n, grp in groupby(tuple(reversed(by.get(m, []))),
                                  lambda d: d.num_tokens):
                grp = list(grp)
                for i in range(start, n + 1):
                    self._candidates.setdefault(i, []).extend(grp)
                start = n + 1

    def dispatch(self, n, u):
        for d in self._candidates.get(n, []):
            if d.uniform_token_count in (None, u) and d.num_tokens >= n:
                return d
        return None


def test_uniform_graph_lens_adds_every_width(monkeypatch):
    class M(_Mgr):
        pass

    assert own_attn.wrap_init_candidates(M, _Mode)
    assert own_attn.wrap_init_candidates(M, _Mode)  # idempotent
    stock, mgr = _Mgr(3), M(3)
    assert mgr.decode_query_len == 3
    full = mgr._capture_descs["FULL"]
    assert {d.uniform_token_count for d in full} == {1, 2, 3}
    assert len(full) == len(set(full))
    assert [d.num_tokens for d in full] == sorted(
        (d.num_tokens for d in full), reverse=True)
    assert set(stock._capture_descs["FULL"]) <= set(full)
    assert mgr._capture_descs["PIECEWISE"] == stock._capture_descs["PIECEWISE"]
    # stock only had FULL for u=3; now q_len 1 and 2 batches replay FULL too
    assert stock.dispatch(2, 1).cg_mode == "PIECEWISE"
    for n, u in ((1, 1), (3, 1), (4, 1), (2, 2), (6, 2), (3, 3), (9, 3)):
        d = mgr.dispatch(n, u)
        assert d.cg_mode == "FULL" and d.uniform_token_count == u, (n, u)
        assert d.num_tokens >= n and d.num_tokens % u == 0
    # mixed (non-uniform) batches keep the stock piecewise pick
    assert mgr.dispatch(5, None).cg_mode == "PIECEWISE"
    for key, lst in mgr._candidates.items():
        modes = [d.cg_mode for d in lst]
        assert modes == sorted(modes), key  # FULL entries before PIECEWISE


def test_uniform_graph_lens_inert_cases():
    class M(_Mgr):
        pass

    own_attn.wrap_init_candidates(M, _Mode)
    assert {d.uniform_token_count for d in M(1)._capture_descs["FULL"]} == {1}
    pw = M(3, decode="PIECEWISE")
    base = _Mgr(3, decode="PIECEWISE")
    assert pw._capture_descs["FULL"] == base._capture_descs["FULL"]


def test_uniform_graph_lens_kill_switch(monkeypatch):
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV_GRAPH_ALL_WIDTHS", "0")
    assert own_attn.install_uniform_graph_lens() is False
