# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sm120/nvfp4_ds_mla_patch: pinned fixtures, transform replay
(byte-exact inverse), drift -> PatchDriftError, gate-off inertness, hook
arming (+ composition with hisparse_mtp_patch), the rewritten impl exercised
against stub vLLM modules, the HiSparse row-width fork, the oracle's torch
references, and the plan module through _native (when the wheel is built)."""
import ast
import hashlib
import sys
import types
import typing
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
import hisparse_mtp_patch as HS  # noqa: E402
import nvfp4_ds_mla_patch as P  # noqa: E402
from nvfp4_ds_mla_patch import oracle as O  # noqa: E402
from nvfp4_kv_patch import _PostImportFinder  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures/vllm_0.30.0"
FIXTURE_SHA256 = {  # vllm-0.30.0 wheel == sdist copies
    "flashinfer_mla_sparse.py":
        "2116d965b2067134d8afa5878d4d16c0149b061258f50c4106d993aa13b073cc",
    "flashinfer_mla_sparse_sm120.py":
        "102ca08793d567f95598eefeb34c3f6ec50b3b9d704f162d1402b97b12b5777b",
    "index_group.py":
        "c24e028c3c160a9bb92e043da5b5b34c805e84745b7c4b8da6fb8702a0ddd641",
}


def fixture(module):
    return (FIX / P.TARGETS[module][0]).read_text()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.delenv(HS.GATE_ENV, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


def test_fixtures_are_pinned():
    assert {f for f, _ in P.TARGETS.values()} == set(FIXTURE_SHA256)
    for name, sha in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize("module", list(P.TARGETS))
def test_replay_is_exact(module):
    src = fixture(module)
    new, applied = P.patch_source(module, src)
    edits = P.TARGETS[module][1]
    assert applied == [e[0] for e in edits]
    for _n, _old, rep, _c in edits:
        assert new.count(rep) == 1
    back = new
    for _n, old, rep, _c in reversed(edits):
        back = back.replace(rep, old)
    assert back == src  # nothing outside the hunks changed
    with pytest.raises(P.PatchDriftError):
        P.patch_source(module, new)  # twice = drift


@pytest.mark.parametrize("module,name", [(m, e[0]) for m, (_f, es) in P.TARGETS.items()
                                         for e in es])
def test_drift_missing_and_duplicate_anchor(module, name):
    src = fixture(module)
    old = next(e[1] for e in P.TARGETS[module][1] if e[0] == name)
    with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 0"):
        P.patch_source(module, src.replace(old, old.replace("\n", " \n", 1)))
    with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 2"):
        P.patch_source(module, src + "\n" + old)


def _class_node(src, cls):
    return next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.ClassDef) and n.name == cls)


def test_backend_accepts_nvfp4_only_on_sm120_class():
    new, _ = P.patch_source(P.BACKEND_MODULE, fixture(P.BACKEND_MODULE))
    for cls, want in (("FlashInferMLASparseSM120Backend", True),
                      ("FlashInferMLASparseTRTLLMBackend", False)):
        seg = ast.get_source_segment(new, _class_node(new, cls))
        assert ('"nvfp4_ds_mla"' in seg) is want, cls
    seg = ast.get_source_segment(new, _class_node(new, "FlashInferMLASparseSM120Backend"))
    assert seg.count('"nvfp4_ds_mla"') == 2  # support list + supports_combination


def test_hisparse_row_width_fork():
    import torch

    new, _ = P.patch_source(P.INDEX_GROUP_MODULE, fixture(P.INDEX_GROUP_MODULE))
    new_text = P.INDEX_GROUP_EDITS[0][2]
    assert new_text in new
    src = ("class X:\n    def f(self, kv_cache_dtype, head_size):\n" + new_text
           + "            kv_dtype = torch.uint8\n        else:\n"
           "            row_width = head_size\n            kv_dtype = None\n"
           "        return row_width, kv_dtype\n")
    ns = {"torch": torch, "FP8_DS_MLA_ROW_BYTES": 656}
    exec(src, ns)
    f = ns["X"]().f
    assert f("nvfp4_ds_mla", 576) == (352, torch.uint8)
    assert f("fp8_ds_mla", 576) == (656, torch.uint8)
    assert f("auto", 576) == (576, None)


def test_gate_off_is_inert():
    before = list(sys.meta_path)
    mod = types.ModuleType(P.IMPL_MODULE)
    snap = dict(mod.__dict__)
    assert P.install_post_import_hook() is False
    assert P.apply(mod) is False
    assert sys.meta_path == before and mod.__dict__ == snap


def test_gate_on_not_sm120_is_inert(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: False)
    mod = types.ModuleType(P.IMPL_MODULE)
    assert P.apply(mod) is False and not hasattr(mod, P.MARKER_ATTR)


def test_hooks_arm_all_targets_and_compose_with_hisparse_mtp(monkeypatch):
    for t in list(P.TARGETS) + [HS.TARGET_MODULE, HS.DEPENDENT_MODULE]:
        monkeypatch.delitem(sys.modules, t, raising=False)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setenv(HS.GATE_ENV, "1")
    assert HS.install_post_import_hook() is True
    assert P.install_post_import_hook() is True
    assert P.install_post_import_hook() is True  # idempotent
    finders = [f for f in sys.meta_path if isinstance(f, _PostImportFinder)]
    assert sys.meta_path[:len(finders)] == finders  # all at the front
    targets = [f.target for f in finders]
    assert sorted(targets) == sorted(list(P.TARGETS) + [HS.TARGET_MODULE])
    assert HS.TARGET_MODULE not in P.TARGETS  # disjoint files


def test_late_arm_fails_closed(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setitem(sys.modules, P.INDEX_GROUP_MODULE, types.ModuleType("x"))
    with pytest.raises(SystemExit, match="imported before"):
        P.install_post_import_hook()


def test_apply_refuses_other_vllm(monkeypatch, tmp_path):
    path = tmp_path / "m.py"
    path.write_text(fixture(P.IMPL_MODULE))
    mod = types.ModuleType(P.IMPL_MODULE)
    mod.__file__ = str(path)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.31.0"))
    with pytest.raises(P.PatchDriftError, match="0.31.0"):
        P.apply(mod)
    with pytest.raises(SystemExit, match="FAILED"):
        P._hook_callback(mod)


# ---- the rewritten impl, executed against stub vLLM modules ---------------
class _Base(typing.Generic[typing.TypeVar("M")]):
    def __init__(self, num_heads, head_size, scale, *a, **kw):
        self.num_heads, self.scale = num_heads, scale
        self.kv_lora_rank, self.qk_rope_head_dim, self.qk_nope_head_dim = 512, 64, 192
        self.topk_indices_buffer = kw.get("topk_indices_buffer")

    def do_kv_cache_update(self, *args):
        CALLS.append(("base_writer", args[4]))


CALLS = []


def _stub_vllm(monkeypatch):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    mod("vllm", __version__="0.30.0")
    mod("vllm.model_executor.layers.attention.sparse_mla_attention",
        SparseMLACommonImpl=_Base)
    mod("vllm.v1.attention.backend", AttentionLayer=object,
        AttentionType=types.SimpleNamespace(DECODER="decoder"))
    mod("vllm.v1.attention.backends.mla.flashinfer_mla_sparse",
        FlashInferMLASparseMetadata=object,
        _get_workspace_buffer=lambda d: CALLS.append(("fi_workspace",)))
    mod("vllm.v1.attention.backends.mla.index_group", HiSparseMLAIndexGroup=type("H", (), {}))
    mod("vllm.v1.attention.backends.mla.sparse_utils",
        triton_convert_req_index_to_global_index=None)
    mod("vllm.config", get_current_vllm_config=lambda: types.SimpleNamespace(
        model_config=None))
    mod("vllm.utils.flashinfer", has_flashinfer_sparse_mla_sm120=lambda: True,
        flashinfer_trtllm_batch_decode_with_kv_cache_mla=lambda **kw: (
            CALLS.append(("flashinfer",)) or kw["out"]))


def _patched_impl_module(monkeypatch, tmp_path):
    _stub_vllm(monkeypatch)
    path = tmp_path / "flashinfer_mla_sparse_sm120.py"
    path.write_text(fixture(P.IMPL_MODULE))
    m = types.ModuleType(P.IMPL_MODULE)
    m.__file__ = str(path)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    assert P.apply(m) is True and P.apply(m) is True  # idempotent
    for k in P._HELPERS:  # record instead of touching CUDA
        monkeypatch.setitem(m.__dict__, k, lambda *a, _k=k: CALLS.append((_k,)) or "ours")
    return m


def _impl(m, dtype):
    return m.FlashInferMLASparseSM120Impl(
        8, 576, 0.0625, 1, None, None, dtype, None, "decoder", None,
        topk_indices_buffer=object())


def test_patched_impl_routes(monkeypatch, tmp_path):
    import torch

    m = _patched_impl_module(monkeypatch, tmp_path)
    CALLS.clear()
    impl = _impl(m, "nvfp4_ds_mla")
    assert impl._use_nvfp4_ds_mla and CALLS == [("_suffix_nvfp4_ds_mla_init",)]
    q, topk = torch.zeros(3, 8, 576, dtype=torch.bfloat16), torch.zeros(3, 2048)
    CALLS.clear()
    assert impl._run_mqa_kernel(q, None, topk) == "ours"
    impl.do_kv_cache_update(None, None, None, None, "nvfp4_ds_mla", None)
    assert CALLS == [("_suffix_nvfp4_ds_mla_decode",), ("_suffix_nvfp4_ds_mla_write",)]
    # fp8_ds_mla keeps the stock flashinfer + C++ writer routes
    CALLS.clear()
    impl = _impl(m, "fp8_ds_mla")
    assert impl._use_nvfp4_ds_mla is False
    impl._run_mqa_kernel(q, torch.zeros(4, 64, 656, dtype=torch.uint8), topk)
    impl.do_kv_cache_update(None, None, None, None, "fp8_ds_mla", None)
    assert CALLS == [("fi_workspace",), ("flashinfer",), ("base_writer", "fp8_ds_mla")]
    with pytest.raises(NotImplementedError, match="fp8_ds_mla"):
        _impl(m, "auto")
    import linecache
    fname = m.FlashInferMLASparseSM120Impl.do_kv_cache_update.__code__.co_filename
    assert P.HELPER_TAG in "".join(linecache.getlines(fname))


def _fake_native(monkeypatch, calls):
    n = types.SimpleNamespace(
        HAS_NVFP4_DSMLA_CUDA=True,
        nvfp4_ds_mla_plan=lambda t, h, c, sms: {"ns": 7},
        nvfp4_ds_mla_decode_cuda=lambda *a: calls.append(("decode", a)),
        nvfp4_ds_mla_quant_store_cuda=lambda *a: calls.append(("write", a)))
    pkg = types.ModuleType("suffix_hybrid")
    pkg._native = n
    monkeypatch.setitem(sys.modules, "suffix_hybrid", pkg)
    monkeypatch.setitem(sys.modules, "suffix_hybrid._native", n)
    import torch

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda d=None: types.SimpleNamespace(cuda_stream=42))
    return n


def test_helpers_shape_the_workspace_and_args(monkeypatch):
    import torch

    calls = []
    n = _fake_native(monkeypatch, calls)
    impl = types.SimpleNamespace(scale=0.0625, _nvfp4_num_sms=188)
    q = torch.zeros(6, 8, 576, dtype=torch.bfloat16)
    out = torch.empty(6, 8, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 64, 352, dtype=torch.uint8)
    topk = torch.zeros(2048, 6, dtype=torch.int32).t()  # non-contiguous view
    assert P._suffix_nvfp4_ds_mla_decode(impl, q, kv, topk, out) is out
    (_, a), = calls
    assert a[2].is_contiguous() and a[3] is out
    assert a[4].shape == (6, 8, 7, 512) and a[4].dtype == torch.bfloat16
    assert a[5].shape == (6, 8, 7) and a[5].dtype == torch.float32
    assert a[6] == 0.0625 and a[7] == 42
    calls.clear()
    assert P._suffix_nvfp4_ds_mla_decode(impl, q[:0], kv, topk[:0], out[:0]).numel() == 0
    assert calls == []
    P._suffix_nvfp4_ds_mla_write(torch.zeros(5, 512), torch.zeros(5, 1, 64), kv,
                                 torch.zeros(5, 1, dtype=torch.int64))
    (_, a), = calls
    assert a[1].shape == (5, 64) and a[3].shape == (5,) and a[4] == 42
    calls.clear()
    P._suffix_nvfp4_ds_mla_write(None, None, torch.zeros(0, dtype=torch.uint8), None)
    assert calls == []
    n.HAS_NVFP4_DSMLA_CUDA = False
    with pytest.raises(RuntimeError, match="oxide-kernels"):
        P._native()


# ---- oracle references (CPU) ----------------------------------------------
def _np_quant_block(x):
    """Independent numpy-f32 transcription of the writer lane: nearest-grid
    search with ties to the even code (no threshold table)."""
    import numpy as np
    import torch

    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    x = np.asarray(x, np.float32)
    q = np.float32(max(np.float32(np.abs(x).max()) * np.float32(1 / 6), np.float32(2 ** -9)))
    sf = int(O.e4m3_bytes(torch.tensor([q]))[0])
    sff = np.float32(float(O.e4m3_float(torch.tensor([sf], dtype=torch.uint8))[0]))
    inv = np.float32(1) / sff
    codes = []
    for v in x:
        a = abs(np.float32(v * inv))
        d = np.abs(grid - a)
        best = [i for i in range(8) if d[i] == d.min()]
        c = best[0] if len(best) == 1 else [i for i in best if i % 2 == 0][0]
        codes.append(c | (8 if np.signbit(v) else 0))
    return sf, [codes[2 * j] | (codes[2 * j + 1] << 4) for j in range(8)]


def test_quant_rows_ref_matches_independent_transcription():
    import torch

    g = torch.Generator().manual_seed(3)
    kv = (torch.randn(6, 512, generator=g) * torch.exp(torch.randn(512, generator=g))).bfloat16()
    kv[0] = 0
    kv[1, :16] = 5000.0
    kv[2, :16] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
                               -0.25, -0.75, 7.0, 0.0, 1.0, 1.0, 1.0, 1.0]).bfloat16()
    pe = torch.randn(6, 64, generator=g).bfloat16()
    rows = O.quant_rows_ref(kv, pe)
    for t in range(6):
        for b in range(32):
            sf, data = _np_quant_block(kv[t, 16 * b:16 * b + 16].float().numpy())
            assert int(rows[t, 320 + O.SF_PERM[b]]) == sf, (t, b)
            assert rows[t, 8 * b:8 * b + 8].tolist() == data, (t, b)
    assert int(rows[0, 320]) == 0x01  # 2^-9 floor
    assert int(rows[1, 320]) == 0x7E  # saturated sf (448)
    assert rows[:, 256:320].equal(pe.float().to(torch.float8_e4m3fn).view(torch.uint8))


def test_e2m1_ties_and_e4m3_saturation():
    import torch

    x = torch.tensor([0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 100.0, -0.0, -3.0])
    assert O.e2m1_codes(x).tolist() == [0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 8, 13]
    assert O.e4m3_bytes(torch.tensor([420.0, 440.0, 1e6, -1e6])).tolist() == [
        0x7D, 0x7E, 0x7E, 0xFE]


def test_dequant_round_trip_and_sf_permutation():
    import torch

    g = torch.Generator().manual_seed(4)
    kv = torch.randn(32, 512, generator=g).bfloat16()
    pe = torch.randn(32, 64, generator=g).bfloat16()
    lat, rope = O.dequant_rows_ref(O.quant_rows_ref(kv, pe))
    assert O.rel_l2(lat, kv.float()) < 0.15 and O.rel_l2(rope, pe.float()) < 0.05
    # moving one block's SF byte to a wrong (unpermuted) slot must break it
    rows = O.quant_rows_ref(kv, pe)
    bad = rows.clone()
    bad[:, 320:352] = rows[:, [320 + p for p in O.SF_PERM]]  # stored in block order
    lat_bad, _ = O.dequant_rows_ref(bad)
    assert O.rel_l2(lat_bad, kv.float()) > 0.3
    assert O.canon(torch.tensor([[0x88] + [0] * 351], dtype=torch.uint8))[0, 0] == 0


def test_attention_ref_masking_semantics():
    import torch

    g = torch.Generator().manual_seed(5)
    lat, rope = torch.randn(100, 512, generator=g), torch.randn(100, 64, generator=g)
    q = torch.randn(3, 4, 576, generator=g)
    topk = torch.full((3, 16), -1, dtype=torch.int32)
    topk[0, :5] = torch.tensor([3, 7, 11, 50, 99])
    topk[1, 7] = 42  # single valid row behind holes
    out = O.attention_ref(q, lat, rope, topk, 0.0625)
    assert torch.equal(out[2], torch.zeros(4, 512))  # all masked -> 0
    torch.testing.assert_close(out[1], lat[42].expand(4, 512))
    perm = topk[0, torch.randperm(16, generator=g)]
    torch.testing.assert_close(
        O.attention_ref(q[:1], lat, rope, perm[None], 0.0625), out[:1])


def test_dequant_fp8_rows_layout():
    import torch

    row = torch.zeros(1, 656, dtype=torch.uint8)
    row[0, :512] = torch.full((512,), 1.5).to(torch.float8_e4m3fn).view(torch.uint8)
    row[0, 512:528] = torch.tensor([1.0, 2.0, 4.0, 8.0]).view(torch.uint8)
    row[0, 528:656] = torch.arange(64, dtype=torch.bfloat16).view(torch.uint8)
    lat, rope = O.dequant_fp8_rows(row)
    assert lat[0, 0] == 1.5 and lat[0, 128] == 3.0 and lat[0, 511] == 12.0
    assert torch.equal(rope[0], torch.arange(64.0))


def test_plan_via_native_when_built():
    try:
        from suffix_hybrid import _native
    except ImportError:
        pytest.skip("suffix_hybrid._native not built")
    if not hasattr(_native, "nvfp4_ds_mla_plan"):
        pytest.skip("wheel predates nvfp4_ds_mla")
    p = _native.nvfp4_ds_mla_plan
    assert (p(1, 8, 2048, 188)["ns"], p(6, 8, 2048, 188)["ns"]) == (32, 32)
    assert (p(32, 8, 2048, 188)["ns"], p(32, 8, 2048, 188)["c_per_split"]) == (6, 384)
    assert p(8192, 8, 2048, 188)["ns"] == 1  # prefill: bounded workspace
    assert p(1, 8, 2048, 188)["partial_smem_bytes"] == 69312
    with pytest.raises(ValueError):
        p(0, 8, 2048, 188)


# ---- oracle harness dry run: GPU kernels emulated by the references -------
class _FakeGraph:
    def __init__(self):
        self.fn = None

    def replay(self):
        self.fn()


class _EmuOurs:
    capturing = None

    def write(self, kv_c, k_pe, cache, slots):
        rows = O.quant_rows_ref(kv_c, k_pe)
        bs = cache.shape[1]
        for i, s in enumerate(slots.tolist()[:rows.shape[0]]):
            if s >= 0:
                cache[s // bs, s % bs] = rows[i]

    def decode(self, q, cache, topk, out=None):
        import torch

        if out is None:
            out = q.new_empty(q.shape[0], q.shape[1], O.DIM)

        def run():
            lat, rope = O.dequant_rows_ref(cache.reshape(-1, O.ROW))
            out.copy_(O.attention_ref(q.float(), lat, rope, topk, O.SCALE))
        if _EmuOurs.capturing is not None:
            _EmuOurs.capturing.fn = run
        else:
            run()
        return out if isinstance(out, torch.Tensor) else None


class _EmuStock(_EmuOurs):
    def write(self, kv_c, k_pe, cache, slots):
        import torch

        x = kv_c.float().view(-1, 4, 128)
        scale = torch.clamp(x.abs().amax(-1) / 448.0, min=1e-30)
        rows = torch.zeros(kv_c.shape[0], O.FP8_ROW, dtype=torch.uint8)
        rows[:, :512] = O.e4m3_bytes((x / scale[..., None]).view(-1, 512))
        rows[:, 512:528] = scale.contiguous().view(torch.uint8)
        rows[:, 528:] = k_pe.contiguous().view(torch.uint8)
        bs = cache.shape[1]
        for i, s in enumerate(slots.tolist()):
            if s >= 0:
                cache[s // bs, s % bs] = rows[i]

    def decode(self, q, cache, topk, out=None):
        lat, rope = O.dequant_fp8_rows(cache.reshape(-1, O.FP8_ROW))
        return O.attention_ref(q.float(), lat, rope, topk, O.SCALE).bfloat16()


def test_oracle_gates_dry_run_on_cpu(monkeypatch):
    """The oracle harness itself (data, indexing, masks, metrics, verdicts)
    runs end to end with exact emulations in place of the CUDA kernels —
    so a silicon run can only fail on kernel numerics, not harness bugs."""
    import contextlib

    import torch

    def fused_norm_rope(*a, slot_mapping, mla_kv_cache, kv_c_out, k_pe_out, **kw):
        kv_c_out.copy_(a[4])
        k_pe_out.copy_(a[7])
        _EmuOurs().write(kv_c_out, k_pe_out, mla_kv_cache, slot_mapping)

    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v32.common.kernels",
                        types.SimpleNamespace(fused_norm_rope=fused_norm_rope))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    stream = types.SimpleNamespace(wait_stream=lambda s: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)

    @contextlib.contextmanager
    def graph(g):
        _EmuOurs.capturing = g
        yield
        _EmuOurs.capturing = None
    monkeypatch.setattr(torch.cuda, "graph", graph)
    results = []

    def report(name, ok, detail):
        results.append((name, ok, detail))

    gen = torch.Generator().manual_seed(0)
    O.gate_writer(_EmuOurs(), "cpu", gen, report)
    O.gate_reader(_EmuOurs(), _EmuStock(), "cpu", gen, report)
    names = [r[0] for r in results]
    assert len(results) == 16 and "reader_cuda_graph_replay" in names
    print(*results, sep="\n")
    assert all(ok for _n, ok, _d in results), [r for r in results if not r[1]]
