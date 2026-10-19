# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 geometry (32 index heads x 128, top-512, block 64) is oxide-free.

Owner direction 2026-10-19: the oxide Rust CUDA kernels are retired and V4.1 must not touch
them. Oxide's dsa_indexer is the GLM-5.3 format and cannot read V4.1's pages anyway. On the
V4.1 route (shim wrappers -> routing -> Helion bodies / the stock fallback):

  * ``auto``   -> Helion where a V4.1-keyed (fmt=v41) config is shipped, else the shim's
                  fallback (the vendored DeepGEMM where it runs, tests/test_dsa_vendor_fallback.py;
                  here: a fallback that cannot say = the Triton stand-in, label STOCK) -- never
                  oxide, and never a GLM-5.3 config (keys differ);
  * ``helion`` -> every bucket needs its config, a missing one is a startup error;
  * ``oxide`` (also the default when the knob is unset) -> refused, naming why;
  * the native extension is never imported (the whole route runs with the import blocked);
  * K5 reports the fallback per layer, loudly, and STRICT refuses it.
GLM-5.3 geometry (top-2048) keeps every previous behaviour.
"""
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_capture as cap  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import test_helion_dsa_indexer as th  # noqa: E402

F8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setitem(di._state, "layers", {})
    monkeypatch.setitem(di._state, "stock", [])
    monkeypatch.setitem(di._state, "vendor", [])
    monkeypatch.setitem(di._state, "geom", None)
    for var in (di.HELION_ENV, di.GEOMETRY_ENV, di.STRICT_ENV, di.CTX_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(di.GROUP_ENV, "6")
    cap.reset()


@pytest.fixture
def no_oxide(monkeypatch):
    """The native extension and its Python loader cannot be imported at all, and the lane's own
    loader is poisoned: any touch of oxide on the route under test fails loudly."""
    import suffix_hybrid
    monkeypatch.setitem(sys.modules, "suffix_hybrid.oxide_kernels", None)
    monkeypatch.setitem(sys.modules, "suffix_hybrid._native", None)
    monkeypatch.delattr(suffix_hybrid, "oxide_kernels", raising=False)
    monkeypatch.delattr(suffix_hybrid, "_native", raising=False)

    def boom(dev):
        raise AssertionError("the oxide _native extension was touched on the V4.1 route")
    monkeypatch.setattr(di, "_native", boom)
    with pytest.raises(ImportError):
        from suffix_hybrid import oxide_kernels  # noqa: F401 - the block really blocks


def arm(monkeypatch, shipped=(), keys="v41"):
    """Helion bodies on the CPU stand-in; hk.config ships a config for `shipped` kernels whose case
    key is V4.1-keyed (keys="v41"), GLM-keyed (keys="glm") or both."""
    def config(kernel, case, plat=None):
        if kernel not in shipped:
            return None
        has_fmt = "fmt" in case
        return {"block_sizes": [32]} if keys == "both" or (keys == "v41") == has_fmt else None
    monkeypatch.setattr(hk, "config", config)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    monkeypatch.setattr(hk, "platform", lambda device=0, plat=None: "test_gpu")   # error texts name the GPU
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", th.make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    monkeypatch.setattr(di, "_sms", lambda dev: 4)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: types.SimpleNamespace(cuda_stream=0))


def problem(nn=6, lens=(200,), n_cols=300, page=64, pad=0):
    """The shim-level call arguments of one flattened decode (rows = requests x nn)."""
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = th.decode_problem(32, list(lens), nn, n_cols, seed=4, page=page, pad=pad)
    rows = q.shape[0]
    cache = th.cache_of(kv, pg, prow).unsqueeze(2)
    ref = hdi.decode_reference(q, w, kv, ks, bt, cl, out, pg, prow, sp, 2)
    return (q.view(rows, 1, 32, 128), None), cache, w, cl.view(rows, 1), bt, None, n_cols, ref


STOCK_RAN = "stock-ran"


def window_ok(out, ref):
    """out agrees with the f64 spec on every row's window (the kernels leave the rest unwritten)."""
    val, mag = ref
    inside = ~torch.isnan(val)
    got = out[:, :val.shape[1]][inside].double()
    return bool(torch.isfinite(got).all()) and \
        float(((got - val[inside]).abs() / (mag[inside] + 1e-30)).max()) <= di.TOL


def stock(*a, **k):
    return STOCK_RAN


# ---------------------------------------------------------------------------
# geometry detection
# ---------------------------------------------------------------------------
def test_v41_geometry_from_the_indexer_frame_the_config_or_the_pin(monkeypatch):
    assert not di.v41_geometry()                                       # nothing known: not V4.1
    assert di.v41_geometry(32, 128, {"topk_tokens": 512})
    assert not di.v41_geometry(32, 128, {"topk_tokens": 2048})         # GLM-5.3 / DSv3.2
    assert not di.v41_geometry(64, 128, {"topk_tokens": 512}) and not di.v41_geometry(32, 64, {"topk_tokens": 512})
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    assert di.v41_geometry() and di.v41_geometry(32, 128, {"topk_tokens": 2048})   # the pin wins
    assert not di.v41_geometry(64, 128)                                # a 64-head indexer is never V4.1
    monkeypatch.setenv(di.GEOMETRY_ENV, "glm")
    assert not di.v41_geometry(32, 128, {"topk_tokens": 512})
    monkeypatch.delenv(di.GEOMETRY_ENV)

    class Cfg:
        model_config = types.SimpleNamespace(architectures=["DeepseekV41ForCausalLM"],
                                             hf_config=types.SimpleNamespace(index_topk=512))
    fake = types.ModuleType("vllm.config")
    fake.get_current_vllm_config_or_none = lambda: Cfg()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", fake)
    assert di._state["geom"] is False                                  # (no vLLM was importable: cached "not V4.1")
    monkeypatch.setitem(di._state, "geom", None)
    assert di.v41_geometry()                                           # from the vLLM config
    monkeypatch.setitem(di._state, "geom", None)
    Cfg.model_config = types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"],
                                             hf_config=types.SimpleNamespace(index_topk=2048))
    assert not di.v41_geometry()


# ---------------------------------------------------------------------------
# oxide is refused for V4.1 geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", [None, "oxide"])
def test_the_oxide_backend_is_refused_everywhere_on_the_v41_route(monkeypatch, backend, no_oxide):
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    if backend:
        monkeypatch.setenv(di.HELION_ENV, backend)          # explicit, or unset = the default "oxide"
    q, cache, w, cl, bt, sched, n_cols, _ = problem()
    mqa_args = ((q[0][:, 0], None), (torch.zeros((9, 128), dtype=F8), torch.zeros(9)), torch.zeros((6, 32)),
                torch.zeros(6, dtype=torch.int32), torch.ones(6, dtype=torch.int32))
    calls = [
        lambda: di.wrap_paged(stock)(q, cache, w, cl, bt, sched, n_cols),
        lambda: di.wrap_mqa(stock)(*mqa_args, 5),
        lambda: di.wrap_mqa(stock)(*mqa_args),
        lambda: di.paged_route(q[0], cache),
        lambda: di.paged_logits(q[0], cache, w, cl, bt, n_cols),
        lambda: di.mqa_logits(q[0][:, 0], torch.zeros((9, 128), dtype=F8), torch.zeros(9), torch.zeros((6, 32)),
                              torch.zeros(6, dtype=torch.int32), torch.ones(6, dtype=torch.int32)),
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as e:
            call()
        msg = str(e.value)
        assert "oxide" in msg and "GLM-5.3 format" in msg and "retired for V4.1" in msg
        assert "SUFFIX_HELION_DSA_INDEXER=helion" in msg and "auto" in msg      # what to set instead
    assert di.stock_fallbacks() == []                                           # refused, not "fallen back"


def test_glm_geometry_keeps_the_oxide_backend(monkeypatch):
    """Not V4.1 (top-2048 / pinned glm): the previous routing, oxide included."""
    monkeypatch.setenv(di.GEOMETRY_ENV, "glm")
    q, cache, w, cl, bt, sched, n_cols, _ = problem(page=64)
    assert di.paged_route(q[0], cache) == "oxide"                       # default backend, contiguous 64-page layout
    monkeypatch.setenv(di.HELION_ENV, "auto")
    arm(monkeypatch)                                                    # auto, no config shipped
    assert di.paged_route(q[0], cache) == "oxide"                       # ... oxide where no config is shipped (GLM)
    p128 = torch.zeros((4, 128, 1, 132), dtype=torch.uint8)
    assert di.paged_route(q[0], p128) == di.STOCK                       # a layout oxide cannot read: stock


# ---------------------------------------------------------------------------
# auto: V4.1 configs or the stock path, never oxide
# ---------------------------------------------------------------------------
def test_auto_without_v41_configs_serves_the_stock_path_even_when_glm_configs_exist(monkeypatch, no_oxide, capsys):
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    monkeypatch.setenv(di.HELION_ENV, "auto")
    arm(monkeypatch, {hdi.DECODE, hdi.GROUP}, keys="glm")               # shipped for GLM-5.3 keys only
    q, cache, w, cl, bt, sched, n_cols, _ = problem()
    assert di.paged_route(q[0], cache) == di.STOCK
    assert di.wrap_paged(stock)(q, cache, w, cl, bt, sched, n_cols) == STOCK_RAN
    (layer, kind, why), = di.stock_fallbacks()
    assert (layer, kind) == ("?", "decode") and "V4.1 geometry" in why and "oxide is retired" in why
    err = capsys.readouterr().err
    assert "STOCK FALLBACK layer ?: decode path stock" in err and "not a Helion kernel" in err
    assert di.vendor_fallbacks() == []                                  # (this fallback cannot say it is the vendor)
    # nothing of the Helion lane launched (no GLM config leaked into V4.1)
    assert not hdi._WARM


def test_auto_with_v41_configs_runs_the_helion_group_or_per_row_body(monkeypatch, no_oxide):
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    monkeypatch.setenv(di.HELION_ENV, "auto")
    q, cache, w, cl, bt, sched, n_cols, ref = problem()
    for shipped, route in (({hdi.DECODE, hdi.GROUP}, "helion-group"), ({hdi.DECODE}, "helion")):
        arm(monkeypatch, shipped, keys="v41")
        monkeypatch.setitem(di._state, "layers", {})
        assert di.paged_route(q[0], cache) == route
        out = di.wrap_paged(stock)(q, cache, w, cl, bt, sched, n_cols)
        assert out is not STOCK_RAN
        assert window_ok(out, ref)
        assert di.layer_paths() == {"?": [("decode", route)]} and di.stock_fallbacks() == []


def test_helion_mode_with_a_missing_v41_config_is_a_startup_error_not_a_fallback(monkeypatch, no_oxide):
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    monkeypatch.setenv(di.HELION_ENV, "helion")
    arm(monkeypatch, {hdi.DECODE, hdi.GROUP}, keys="glm")               # GLM configs exist, V4.1 ones do not
    q, cache, w, cl, bt, sched, n_cols, _ = problem()
    with pytest.raises(RuntimeError, match="no tuned dsa_indexer_decode config"):
        di.wrap_paged(stock)(q, cache, w, cl, bt, sched, n_cols)
    assert di.stock_fallbacks() == []
    # shipped for V4.1: serves
    arm(monkeypatch, {hdi.DECODE}, keys="v41")
    out = di.wrap_paged(stock)(q, cache, w, cl, bt, sched, n_cols)
    assert out is not STOCK_RAN


def test_the_v41_decode_entry_has_no_oxide_fallback(monkeypatch, no_oxide):
    """hdi.dsa_indexer_logits_cuda(v41=True): a bucket without a V4.1 config raises (GLM-5.3 configs
    do not match V4.1 keys); GLM geometry keeps its oxide fallback."""
    monkeypatch.setenv(di.HELION_ENV, "auto")
    arm(monkeypatch, {hdi.DECODE}, keys="glm")
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = th.decode_problem(32, [100], 1, 200)
    cache = th.cache_of(kv, pg, prow)
    call = lambda **kw: hdi.dsa_indexer_logits_cuda(  # noqa: E731
        out, q, w, cache, None, bt, None, cl, 200, 1, 4, 1, 2, 2, 0, **kw)
    with pytest.raises(RuntimeError, match="no oxide fallback"):
        call(v41=True)
    call()                                                             # the GLM-5.3 key IS shipped for GLM: Helion
    arm(monkeypatch, set())
    with pytest.raises(AssertionError, match="oxide _native extension was touched"):
        call()                                                         # GLM, nothing shipped: its oxide fallback
    arm(monkeypatch, {hdi.DECODE}, keys="v41")
    call(v41=True)                                                     # the V4.1 key is shipped: Helion runs


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------
def test_v41_prefill_is_helion_or_stock_never_oxide(monkeypatch, no_oxide):
    monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
    got = []
    monkeypatch.setattr(hdi, "prefill_logits", lambda *a, **k: got.append(k) or "helion")
    q, kvc, w = torch.zeros((2, 32, 128), dtype=F8), torch.zeros((9, 128), dtype=F8), torch.zeros((2, 32))
    one = torch.ones(2, dtype=torch.int32)
    mqa = di.wrap_mqa(stock)
    call = lambda *x: mqa((q, None), (kvc, torch.zeros(9)), w, one - 1, one, *x)  # noqa: E731
    monkeypatch.setenv(di.HELION_ENV, "auto")
    arm(monkeypatch, set(), keys="v41")
    assert call(7) == STOCK_RAN and not got                              # compressed contract, no V4.1 config
    arm(monkeypatch, {hdi.PREFILL}, keys="glm")
    assert call(7) == STOCK_RAN and not got                              # a GLM-5.3 prefill config is not V4.1's
    arm(monkeypatch, {hdi.PREFILL}, keys="v41")
    assert call(7) == "helion" and got == [{"compressed": True, "v41": True}]
    monkeypatch.setenv(di.HELION_ENV, "helion")
    assert call(7) == "helion"
    # the absolute-column contract (bool 6th argument) is the oxide-era one: stock for V4.1
    assert call(False) == STOCK_RAN and call() == STOCK_RAN
    paths = di.layer_paths()["?"]
    assert ("prefill", "helion-compressed") in paths and ("prefill", di.STOCK) in paths


# ---------------------------------------------------------------------------
# K5 / STRICT
# ---------------------------------------------------------------------------
def test_k5_reports_the_stock_fallback_per_layer_and_strict_refuses_it(monkeypatch, no_oxide, capsys):
    monkeypatch.setenv(di.HELION_ENV, "auto")
    arm(monkeypatch)                                                    # nothing shipped
    q, cache, w, cl, bt, sched, n_cols, _ = problem()
    paged = di.wrap_paged(stock)

    def sparse_attn_indexer(k_cache_prefix, topk_tokens):              # vLLM's frame: the layer name, top-k
        return paged(q, cache, w, cl, bt, sched, n_cols)
    for layer in (2, 8):
        assert sparse_attn_indexer(f"model.layers.{layer}.attn.indexer.k_cache", 512) == STOCK_RAN
    assert sparse_attn_indexer("model.layers.8.attn.indexer.k_cache", 512) == STOCK_RAN      # logged once per layer
    err = capsys.readouterr().err
    for layer in (2, 8):
        assert err.count(f"STOCK FALLBACK layer model.layers.{layer}.attn.indexer.k_cache: decode path stock") == 1
    assert [x[0] for x in di.stock_fallbacks()] == [f"model.layers.{n}.attn.indexer.k_cache" for n in (2, 8)]
    monkeypatch.setenv(di.STRICT_ENV, "1")
    monkeypatch.setitem(di._state, "layers", {})
    with pytest.raises(RuntimeError, match=r"STRICT=1: layer model.layers.20.attn.indexer.k_cache decode would "
                                           r"fall back to the stock Triton stand-in"):
        sparse_attn_indexer("model.layers.20.attn.indexer.k_cache", 512)


def test_glm_frame_keeps_oxide_and_is_not_reported_as_v41(monkeypatch):
    seen = []
    monkeypatch.setattr(di, "paged_logits", lambda *a, **k: seen.append(k.get("v41")) or "ours")
    q, cache, w, cl, bt, sched, n_cols, _ = problem()
    paged = di.wrap_paged(stock)

    def sparse_attn_indexer(k_cache_prefix, topk_tokens):
        return paged(q, cache, w, cl, bt, sched, n_cols)
    assert sparse_attn_indexer("model.layers.3.attn.indexer.k_cache", 2048) == "ours"   # oxide backend: untouched
    assert seen == [False] and di.stock_fallbacks() == []
