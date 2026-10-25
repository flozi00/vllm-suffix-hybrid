# SPDX-License-Identifier: Apache-2.0
"""A bucket the plugin does not serve behaves exactly like the baseline (A1).

Turning the lane on (SUFFIX_SM120=1) replaces the deep_gemm shim's MQA-logits trio. On a pod whose
vendored DeepGEMM runs the SM120 paged MQA at block 64 (vLLM nightly >= 39983a5e7, the path boot5.log
ran) the replacement must be the vendor itself, verbatim: an unserved bucket then costs what A1 costs.
The Triton stand-in stays for the pods where the vendor cannot (GLM-era vLLM 0.30, no vendor).

Everything here runs on a CPU box: the real shim package is loaded into a fake vLLM / vendor world and
the calls are followed by identity. Silicon still has to confirm that the vendor's kernel accepts the
call (docs/dsv41-indexer.md, "Fallback = baseline").
"""
import ast
import importlib.machinery
import importlib.util
import os
import re
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
torch = pytest.importorskip("torch")
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:suffix deep_gemm shim")

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "sm120" / "deep_gemm_shim"
FIXTURE = Path(__file__).parent / "fixtures" / "vllm_39983a5e7" / "deep_gemm.py.txt"
TRIO = ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata")
F8 = torch.float8_e4m3fn
ENVS = ("SUFFIX_SM120", "SUFFIX_SM120_DSA_INDEXER", "SUFFIX_DSA_INDEXER_STOCK", di.HELION_ENV, di.GEOMETRY_ENV,
        di.STRICT_ENV, di.CTX_ENV)


def nightly_surface():
    """Every DeepGEMM symbol vLLM's wrapper resolves from its `_dg` (the shim): read from the anchored source."""
    text = FIXTURE.read_text()
    names = set(re.findall(r'getattr\(\s*(?:_dg|mod),\s*"(\w+)"', text)) | set(re.findall(r"\bdg\.(\w+)\(", text))
    names.discard("__name__")
    return sorted(names)


def _fake(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class World:
    """A fake vLLM + vendored DeepGEMM and the real shim package loaded into it."""

    def __init__(self):
        self.calls = []            # what the VENDOR was called with: (name, args, kwargs)
        self.triton_calls = []     # what the Triton stand-in was called with
        self.page_size_asks = 0
        self.vendor = self.shim = None

    def vendor_fn(self, name, out):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return out
        fn.__name__ = name
        return fn

    def build_vendor(self, surface, drop=()):
        vendor = _fake("vllm.third_party.deep_gemm")
        outs = {"fp8_fp4_mqa_logits": "vendor-mqa", "fp8_fp4_paged_mqa_logits": "vendor-paged",
                "get_paged_mqa_logits_metadata": "vendor-schedule"}
        for name in list(surface) + ["brand_new_vendor_symbol"]:
            if name not in drop:
                setattr(vendor, name, self.vendor_fn(name, outs.get(name, f"vendor-{name}")))
        return vendor

    def same(self, call, name, *args, **kwargs):
        """The vendor was called as `name(*args, **kwargs)` with exactly these objects (identity)."""
        got_name, got_args, got_kwargs = call
        return got_name == name and len(got_args) == len(args) and all(a is b for a, b in zip(got_args, args)) \
            and list(got_kwargs) == list(kwargs) and all(got_kwargs[k] is v for k, v in kwargs.items())


@pytest.fixture
def world(monkeypatch):
    saved = {k: sys.modules.get(k) for k in ("deep_gemm", "deep_gemm.sm120_fallback")}
    for var in ENVS:
        monkeypatch.delenv(var, raising=False)
    for key in ("layers", "stock", "vendor"):
        monkeypatch.setitem(di._state, key, {} if key == "layers" else [])
    monkeypatch.setitem(di._state, "geom", None)

    def load(*, gate=True, sm=(12, 0), vendor=True, declares=True, page_sizes=(64,), drop=(), dsa=False,
             stock_env=None, surface=None):
        w = World()
        for k in saved:
            sys.modules.pop(k, None)
        tri = _fake("triton", jit=lambda fn=None, **kw: fn if fn is not None else (lambda f: f),
                    cdiv=lambda a, b: -(-a // b), next_power_of_2=lambda n: 1 << max(0, (n - 1).bit_length()))
        tri.language = _fake("triton.language", constexpr=object)
        monkeypatch.setitem(sys.modules, "triton", tri)
        monkeypatch.setitem(sys.modules, "triton.language", tri.language)
        if vendor:
            w.vendor = w.build_vendor(surface or nightly_surface(), drop)
            vdg = _fake("vllm.utils.deep_gemm")
            if declares:   # vLLM nightly: utils.deep_gemm.get_paged_mqa_page_sizes(); GLM-era vLLM: no such function

                def get_paged_mqa_page_sizes():
                    w.page_size_asks += 1
                    return page_sizes
                vdg.get_paged_mqa_page_sizes = get_paged_mqa_page_sizes
            utils = _fake("vllm.utils", deep_gemm=vdg)
            tp = _fake("vllm.third_party", deep_gemm=w.vendor)
            for name, mod in (("vllm", _fake("vllm", third_party=tp, utils=utils)), ("vllm.third_party", tp),
                              ("vllm.third_party.deep_gemm", w.vendor), ("vllm.utils", utils),
                              ("vllm.utils.deep_gemm", vdg)):
                monkeypatch.setitem(sys.modules, name, mod)
        else:
            for name in ("vllm.third_party.deep_gemm", "vllm.utils.deep_gemm"):
                monkeypatch.setitem(sys.modules, name, None)       # import blocked: no vendor
            monkeypatch.setitem(sys.modules, "vllm", _fake("vllm"))
        if gate:
            monkeypatch.setenv("SUFFIX_SM120", "1")
        if dsa:
            monkeypatch.setenv("SUFFIX_SM120_DSA_INDEXER", "1")
            monkeypatch.setenv(di.GEOMETRY_ENV, "v41")
        if stock_env:
            monkeypatch.setenv("SUFFIX_DSA_INDEXER_STOCK", stock_env)
        spec = importlib.util.spec_from_file_location("deep_gemm", SHIM / "__init__.py",
                                                      submodule_search_locations=[str(SHIM)])
        w.shim = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "deep_gemm", w.shim)
        real = torch.cuda.is_available, torch.cuda.get_device_capability
        torch.cuda.is_available, torch.cuda.get_device_capability = (lambda: True), (lambda *a, **k: sm)
        try:   # the shim decides SM120 at import; nothing after that may believe in a GPU on this CPU box
            spec.loader.exec_module(w.shim)
        finally:
            torch.cuda.is_available, torch.cuda.get_device_capability = real
        fb = sys.modules.get("deep_gemm.sm120_fallback")
        if fb is not None:   # the Triton stand-in is recorded, never run (it needs a GPU)
            for name, out in (("triton_mqa_logits", "triton-mqa"), ("triton_paged_mqa_logits", "triton-paged"),
                              ("triton_paged_mqa_metadata", "triton-schedule")):
                monkeypatch.setattr(fb, name, lambda *a, _n=name, _o=out, **k: w.triton_calls.append((_n, a, k)) or _o)
        return w

    yield load
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def objs(n):
    return tuple(object() for _ in range(n))


# ---------------------------------------------------------------------------
# the trio is the only replaced surface
# ---------------------------------------------------------------------------
def test_the_trio_is_the_only_replaced_surface(world):
    """SUFFIX_SM120=1 on SM120 must not change any other DeepGEMM symbol: fp8 gemms, the draft head's
    DeepGemmFP4Experts grouped MXFP4 gemm, layout helpers, set_pdl / num_sms ... are the vendor's own objects."""
    w = world()
    surface = nightly_surface()
    assert set(TRIO) <= set(surface) and len(surface) >= 24                    # the anchor really lists the nightly's names
    assert {"m_grouped_fp8_fp4_gemm_nt_contiguous", "fp8_gemm_nt", "transform_sf_into_required_layout",
            "get_mn_major_tma_aligned_tensor", "pack_ue8m0_to_int", "set_pdl", "get_num_sms"} <= set(surface)
    for name in surface:
        mine, theirs = getattr(w.shim, name), getattr(w.vendor, name)
        assert (mine is not theirs) == (name in TRIO), name
    vendor_public = {n for n in dir(w.vendor) if not n.startswith("__")}
    assert "brand_new_vendor_symbol" in vendor_public                          # ... and a symbol nobody has heard of
    assert {n for n in vendor_public if getattr(w.shim, n) is not getattr(w.vendor, n)} == set(TRIO)
    assert {n for n in dir(w.shim) if not n.startswith("_")} - vendor_public <= {
        "importlib", "os", "sys", "warnings", "sm120_fallback"}                 # nothing else is added
    fb = sys.modules["deep_gemm.sm120_fallback"]
    for name in TRIO:
        assert getattr(w.shim, name) is getattr(fb, name)                      # the dispatchers, not the Triton kernels


def test_the_indexer_wrappers_add_nothing_but_the_paged_and_prefill_entries(world):
    w = world(dsa=True)
    vendor_public = {n for n in dir(w.vendor) if not n.startswith("__")}
    assert {n for n in vendor_public if getattr(w.shim, n) is not getattr(w.vendor, n)} == set(TRIO)
    fb = sys.modules["deep_gemm.sm120_fallback"]
    assert w.shim.get_paged_mqa_logits_metadata is fb.get_paged_mqa_logits_metadata     # the schedule: never wrapped
    assert w.shim.fp8_fp4_paged_mqa_logits is not fb.fp8_fp4_paged_mqa_logits           # wrapped by the lane
    assert w.shim.fp8_fp4_mqa_logits is not fb.fp8_fp4_mqa_logits


@pytest.mark.parametrize("gate, sm", [(False, (12, 0)), (True, (9, 0)), (True, (10, 0))])
def test_the_shim_is_inert_without_the_gate_or_off_sm120(world, gate, sm):
    w = world(gate=gate, sm=sm)
    for name in nightly_surface():
        assert getattr(w.shim, name) is getattr(w.vendor, name), name
    assert "deep_gemm.sm120_fallback" not in sys.modules                       # the stand-in is not even imported


# ---------------------------------------------------------------------------
# the trio resolves to the vendor, verbatim
# ---------------------------------------------------------------------------
def test_unserved_paged_calls_reach_the_vendor_verbatim(world):
    w = world()
    q, kv, wt, cl, bt, sched, idx = objs(7)
    assert w.shim.get_paged_mqa_logits_metadata(cl, 64, 188) == "vendor-schedule"
    assert w.same(w.calls[-1], "get_paged_mqa_logits_metadata", cl, 64, 188)
    assert w.shim.get_paged_mqa_logits_metadata(cl, 64, 188, indices=idx) == "vendor-schedule"
    assert w.same(w.calls[-1], "get_paged_mqa_logits_metadata", cl, 64, 188, indices=idx)
    # vLLM's wrapper: seven positional arguments, `indices` only when it is not None
    qt = (q, None)
    assert w.shim.fp8_fp4_paged_mqa_logits(qt, kv, wt, cl, bt, sched, 524288) == "vendor-paged"
    assert w.same(w.calls[-1], "fp8_fp4_paged_mqa_logits", qt, kv, wt, cl, bt, sched, 524288)
    w.shim.fp8_fp4_paged_mqa_logits(qt, kv, wt, cl, bt, sched, 524288, indices=idx)
    assert w.same(w.calls[-1], "fp8_fp4_paged_mqa_logits", qt, kv, wt, cl, bt, sched, 524288, indices=idx)
    assert not w.triton_calls


def test_unserved_prefill_reaches_the_vendor_with_the_nightly_contract(world):
    w = world()
    qt, kvt, wt, ks, ke = objs(5)
    assert w.shim.fp8_fp4_mqa_logits(qt, kvt, wt, ks, ke, 4096) == "vendor-mqa"        # vLLM nightly: int max_seqlen_k
    assert w.same(w.calls[-1], "fp8_fp4_mqa_logits", qt, kvt, wt, ks, ke, 4096)
    assert len(w.calls) == 1 and not w.triton_calls
    # the OLD contract (bool / missing 6th argument, clean_logits) is not the vendor's: the stand-in answers it
    for call in ((), (False,), (True,), (None,)):
        assert w.shim.fp8_fp4_mqa_logits(qt, kvt, wt, ks, ke, *call) == "triton-mqa"
    assert w.shim.fp8_fp4_mqa_logits(qt, kvt, wt, ks, ke, 4096, clean_logits=True) == "triton-mqa"
    assert len(w.calls) == 1 and len(w.triton_calls) == 5


def test_mxfp4_inputs_reach_the_vendor_untouched(world):
    """The draft head / MXFP4 paths: the stand-in refuses MXFP4 (NotImplementedError); the baseline's vendor
    takes it. Vendor-first hands it over without looking at a single tensor."""
    w = world()
    qt, kvt = (objs(1)[0], objs(1)[0]), (objs(1)[0], objs(1)[0])      # (packed values, scales) x (q, kv)
    ks, ke, wt = objs(3)
    assert w.shim.fp8_fp4_mqa_logits(qt, kvt, wt, ks, ke, 128) == "vendor-mqa"
    assert w.same(w.calls[-1], "fp8_fp4_mqa_logits", qt, kvt, wt, ks, ke, 128) and not w.triton_calls


# ---------------------------------------------------------------------------
# the Triton stand-in serves only where the vendor cannot
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("why, kw", [
    ("no vendored DeepGEMM", dict(vendor=False)),
    ("GLM-era vLLM: no page-size declaration", dict(declares=False)),
    ("vLLM declares only 32-token pages", dict(page_sizes=(32,))),
    ("the vendor lacks one of the trio", dict(drop=("get_paged_mqa_logits_metadata",))),
    ("forced by SUFFIX_DSA_INDEXER_STOCK=shim", dict(stock_env="shim")),
])
def test_the_stand_in_serves_only_where_the_vendor_cannot(world, why, kw):
    w = world(**kw)
    cl, bt = objs(2)
    for fn, args, name in ((w.shim.fp8_fp4_paged_mqa_logits, ((1, None), 2, 3, cl, bt, 5, 6), "triton_paged_mqa_logits"),
                           (w.shim.get_paged_mqa_logits_metadata, (cl, 64, 188), "triton_paged_mqa_metadata"),
                           (w.shim.fp8_fp4_mqa_logits, ((1, None), (2, 3), 4, 5, 6, 7), "triton_mqa_logits")):
        fn(*args)
        assert w.triton_calls[-1][0] == name, why
    assert w.calls == []                                                       # the vendor was never called
    for fn in (w.shim.fp8_fp4_paged_mqa_logits, w.shim.fp8_fp4_mqa_logits, w.shim.get_paged_mqa_logits_metadata):
        assert fn.vendor_applies() is False


def test_clean_logits_paged_calls_are_the_old_contract(world):
    w = world()
    args = ((1, None), 2, 3, objs(1)[0], objs(1)[0], 5, 6)
    assert w.shim.fp8_fp4_paged_mqa_logits(*args, clean_logits=True) == "triton-paged"
    assert w.shim.fp8_fp4_paged_mqa_logits.vendor_applies(clean_logits=True) is False
    assert w.shim.fp8_fp4_paged_mqa_logits.vendor_applies(clean_logits=False) is True
    assert w.calls == []


def test_vendor_mode_fails_closed_and_a_typo_is_loud(world):
    w = world(vendor=False, stock_env="vendor")
    with pytest.raises(RuntimeError, match="SUFFIX_DSA_INDEXER_STOCK=vendor but the vendored DeepGEMM cannot"):
        w.shim.get_paged_mqa_logits_metadata(objs(1)[0], 64, 188)
    with pytest.raises(RuntimeError, match="cannot run the SM120 paged MQA"):
        w.shim.fp8_fp4_paged_mqa_logits.vendor_applies()                     # on every call: nothing is cached
    w = world(stock_env="vendor")                                              # the vendor can: it just runs
    assert w.shim.get_paged_mqa_logits_metadata(objs(1)[0], 64, 188) == "vendor-schedule"
    w = world(stock_env="vendr")
    with pytest.raises(ValueError, match="expected vendor | shim | unset"):
        w.shim.get_paged_mqa_logits_metadata(objs(1)[0], 64, 188)


def test_vendor_applies_is_what_the_dispatchers_do(world):
    """K5 labels come from `vendor_applies`: it must agree with the call that follows, case by case."""
    w = world()
    qt, kvt, wt, ks, ke, cl, bt = objs(7)
    for call in ({}, {"clean_logits": True}):
        before = len(w.calls)
        w.shim.fp8_fp4_paged_mqa_logits(qt, kvt, wt, cl, bt, 5, 6, **call)
        assert (len(w.calls) > before) == w.shim.fp8_fp4_paged_mqa_logits.vendor_applies(**call)
    for sixth, kw in ((4096, {}), (4096, {"clean_logits": True}), (False, {}), (None, {}), (True, {})):
        before = len(w.calls)
        w.shim.fp8_fp4_mqa_logits(qt, kvt, wt, ks, ke, sixth, **kw)
        assert (len(w.calls) > before) == w.shim.fp8_fp4_mqa_logits.vendor_applies(max_seqlen_k=sixth, **kw), (sixth, kw)


def test_the_probe_runs_once_and_is_pure_python(world):
    """The first trio call (eager: the schedule metadata is built before any logits call and before a graph
    capture) decides; later calls, possibly inside a capture, only read a flag."""
    w = world()
    cl, bt = objs(2)
    for _ in range(3):
        w.shim.get_paged_mqa_logits_metadata(cl, 64, 188)
        w.shim.fp8_fp4_paged_mqa_logits((1, None), 2, 3, cl, bt, 5, 6)
        w.shim.fp8_fp4_mqa_logits((1, None), (2, 3), 4, 5, 6, 7)
    assert w.page_size_asks == 1 and len(w.calls) == 9
    tree = ast.parse((SHIM / "sm120_fallback.py").read_text())
    probe = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (
        "vendor_module", "_probe_vendor", "vendor_paged_ok")]
    assert len(probe) == 3
    used = {x.id for fn in probe for x in ast.walk(fn) if isinstance(x, ast.Name)} | \
           {x.attr for fn in probe for x in ast.walk(fn) if isinstance(x, ast.Attribute)}
    assert not used & {"torch", "cuda", "synchronize", "current_stream", "empty", "zeros"}      # no device, no allocation


def test_the_oracles_and_benches_name_the_stand_in_explicitly():
    """K3 oracles, K4 accuracy baselines and the lab's Triton arm must never reach the vendor through the
    vendor-first public names, whatever the pod's probe says."""
    for rel in ("suffix_hybrid/kernels/dsa_indexer.py", "suffix_hybrid/kernels/dsa_indexer_lab.py",
                "suffix_hybrid/helion_kernels/dsa_indexer.py"):
        src = (ROOT / rel).read_text()
        assert not re.search(r"(?:_fallback_mod\(\)|\bfb)\.fp8_fp4_|(?:_fallback_mod\(\)|\bfb)\.get_paged_mqa", src), rel
        assert "triton_paged_mqa_logits" in src or "triton_mqa_logits" in src, rel


# ---------------------------------------------------------------------------
# the lane's wrappers (SUFFIX_SM120_DSA_INDEXER=1): V4.1 geometry, no config -> the vendor, loudly (K5)
# ---------------------------------------------------------------------------
@pytest.fixture
def lane(world, monkeypatch):
    pytest.importorskip("helion")
    sys.path.insert(0, str(Path(__file__).parent))
    import test_dsa_v41_oxide_free as v41
    from suffix_hybrid.kernels import dsa_capture as cap
    cap.reset()
    monkeypatch.setenv(di.GROUP_ENV, "6")
    import suffix_hybrid
    monkeypatch.setitem(sys.modules, "suffix_hybrid.oxide_kernels", None)     # the whole route runs without oxide
    monkeypatch.setitem(sys.modules, "suffix_hybrid._native", None)
    monkeypatch.delattr(suffix_hybrid, "oxide_kernels", raising=False)
    monkeypatch.delattr(suffix_hybrid, "_native", raising=False)

    def boom(dev):
        raise AssertionError("the oxide _native extension was touched on the vendor-fallback route")
    monkeypatch.setattr(di, "_native", boom)

    def make(**kw):
        w = world(dsa=True, **kw)
        v41.arm(monkeypatch)                                                   # Helion lane armed, nothing shipped
        return w
    return make


def paged_args(rows=6, **over):
    q = (torch.zeros((rows, 1, 32, 128), dtype=F8), over.pop("q_scale", None))
    return (q, torch.zeros((4, 64, 1, 132), dtype=torch.uint8), torch.zeros((rows, 32)),
            torch.ones((rows, 1), dtype=torch.int32), torch.zeros((rows, 4), dtype=torch.int32), "sched", 256)


def in_layer(layer, fn, *args):
    def sparse_attn_indexer(k_cache_prefix, topk_tokens):                    # vLLM's frame: layer name, top-k
        return fn(*args)
    return sparse_attn_indexer(f"model.layers.{layer}.attn.indexer.k_cache", 512)


def test_v41_auto_without_a_config_delegates_decode_to_the_vendor_and_k5_says_so(lane, monkeypatch, capsys):
    monkeypatch.setenv(di.HELION_ENV, "auto")
    w = lane()
    args = paged_args()
    assert in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, *args) == "vendor-paged"
    assert w.same(w.calls[-1], "fp8_fp4_paged_mqa_logits", *args)              # verbatim: same objects, no extra argument
    assert not w.triton_calls
    err = capsys.readouterr().err
    assert ("VENDOR FALLBACK layer model.layers.8.attn.indexer.k_cache: decode path vendor (V4.1 geometry: no "
            "shipped Helion config for this rows bucket") in err and "serves it verbatim" in err
    assert "STOCK FALLBACK" not in err
    assert [x[:2] for x in di.vendor_fallbacks()] == [("model.layers.8.attn.indexer.k_cache", "decode")]
    assert di.stock_fallbacks() == []
    assert di.layer_paths() == {"model.layers.8.attn.indexer.k_cache": [("decode", "vendor")]}
    in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, *args)                        # logged once per layer, served every time
    assert capsys.readouterr().err.count("VENDOR FALLBACK") == 0 and len(w.calls) == 2


def test_v41_auto_without_a_config_delegates_prefill_to_the_vendor(lane, monkeypatch, capsys):
    monkeypatch.setenv(di.HELION_ENV, "auto")
    w = lane()
    q = (torch.zeros((2, 32, 128), dtype=F8), None)
    kv = (torch.zeros((9, 128), dtype=F8), torch.zeros(9))
    rest = (torch.zeros((2, 32)), torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32), 7)
    assert in_layer(2, w.shim.fp8_fp4_mqa_logits, q, kv, *rest) == "vendor-mqa"
    assert w.same(w.calls[-1], "fp8_fp4_mqa_logits", q, kv, *rest)
    assert "VENDOR FALLBACK layer model.layers.2.attn.indexer.k_cache: prefill path vendor" in capsys.readouterr().err
    assert di.layer_paths() == {"model.layers.2.attn.indexer.k_cache": [("prefill", "vendor")]}


def test_a_bucket_the_lane_serves_never_touches_the_vendor_or_its_probe(lane, monkeypatch):
    """The vendor-first fallback costs nothing where a Helion config serves the bucket: no vendor call, no
    schedule, not even the probe (K5 asks `vendor_applies` only when a call falls back)."""
    import test_dsa_v41_oxide_free as v41
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    monkeypatch.setenv(di.HELION_ENV, "auto")
    w = lane()
    v41.arm(monkeypatch, {hdi.DECODE, hdi.GROUP}, keys="v41")                   # shipped for V4.1
    q, cache, wt, cl, bt, sched, n_cols, ref = v41.problem()
    out = in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, q, cache, wt, cl, bt, sched, n_cols)
    assert isinstance(out, torch.Tensor) and v41.window_ok(out, ref)
    assert w.calls == [] and w.triton_calls == [] and w.page_size_asks == 0
    assert di.layer_paths() == {"model.layers.8.attn.indexer.k_cache": [("decode", "helion-group")]}
    assert di.vendor_fallbacks() == [] and di.stock_fallbacks() == []


def test_ineligible_v41_shapes_are_the_baselines_too(lane, monkeypatch, capsys):
    """MXFP4 queries, varlen indices ...: outside every kernel of the lane, so the vendor answers, as in A1."""
    monkeypatch.setenv(di.HELION_ENV, "auto")
    w = lane()
    args = paged_args(q_scale=torch.zeros((6, 1, 32), dtype=torch.int32))
    assert in_layer(14, w.shim.fp8_fp4_paged_mqa_logits, *args) == "vendor-paged"
    assert w.same(w.calls[-1], "fp8_fp4_paged_mqa_logits", *args)
    assert "decode path vendor (non-FP8 (MXFP4) query)" in capsys.readouterr().err


def test_strict_refuses_the_vendor_fallback_before_the_vendor_runs(lane, monkeypatch):
    monkeypatch.setenv(di.HELION_ENV, "auto")
    monkeypatch.setenv(di.STRICT_ENV, "1")
    w = lane()
    with pytest.raises(RuntimeError, match=r"STRICT=1: layer model.layers.20.attn.indexer.k_cache decode would "
                                           r"fall back to the vendored DeepGEMM"):
        in_layer(20, w.shim.fp8_fp4_paged_mqa_logits, *paged_args())
    q = (torch.zeros((2, 32, 128), dtype=F8), None)
    kv = (torch.zeros((9, 128), dtype=F8), torch.zeros(9))
    with pytest.raises(RuntimeError, match="prefill would fall back to the vendored DeepGEMM"):
        in_layer(20, w.shim.fp8_fp4_mqa_logits, q, kv, torch.zeros((2, 32)), torch.zeros(2, dtype=torch.int32),
                 torch.ones(2, dtype=torch.int32), 7)
    assert w.calls == [] and w.triton_calls == []


def test_oxide_stays_refused_for_v41_and_the_vendor_is_not_touched(lane, monkeypatch):
    for mode in (None, "oxide"):                                               # unset == oxide
        if mode:
            monkeypatch.setenv(di.HELION_ENV, mode)
        else:
            monkeypatch.delenv(di.HELION_ENV, raising=False)
        w = lane()
        with pytest.raises(RuntimeError, match="retired for V4.1"):
            in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, *paged_args())
        assert w.calls == [] and w.triton_calls == []


def test_helion_mode_with_a_missing_config_is_a_startup_error_not_a_vendor_fallback(lane, monkeypatch):
    monkeypatch.setenv(di.HELION_ENV, "helion")
    w = lane()
    with pytest.raises(RuntimeError, match="no tuned dsa_indexer_decode config"):
        in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, *paged_args())
    assert w.calls == [] and di.vendor_fallbacks() == [] and di.stock_fallbacks() == []


def test_a_pod_whose_vendor_cannot_gets_the_stand_in_and_k5_calls_it_stock(lane, monkeypatch, capsys):
    monkeypatch.setenv(di.HELION_ENV, "auto")
    w = lane(declares=False)                                                   # GLM-era vLLM: no page-size declaration
    args = paged_args()
    assert in_layer(8, w.shim.fp8_fp4_paged_mqa_logits, *args) == "triton-paged"
    assert w.calls == [] and w.triton_calls[-1][0] == "triton_paged_mqa_logits"
    err = capsys.readouterr().err
    assert "STOCK FALLBACK layer model.layers.8.attn.indexer.k_cache: decode path stock" in err
    assert "Triton stand-in" in err and "VENDOR FALLBACK" not in err
    assert [x[:2] for x in di.stock_fallbacks()] == [("model.layers.8.attn.indexer.k_cache", "decode")]
    assert di.vendor_fallbacks() == []
    monkeypatch.setenv(di.STRICT_ENV, "1")
    with pytest.raises(RuntimeError, match="stock Triton stand-in"):
        in_layer(14, w.shim.fp8_fp4_paged_mqa_logits, *args)


def test_k5_label_comes_from_the_shim_fallback_and_defaults_to_stock():
    assert di.stock_label(lambda *a: None) == di.STOCK                         # a fallback that cannot say
    fn = lambda *a: None  # noqa: E731
    fn.vendor_applies = lambda **kw: kw.get("clean_logits") is False
    assert di.stock_label(fn, clean_logits=False) == di.VENDOR
    assert di.stock_label(fn, clean_logits=True) == di.STOCK


# ---------------------------------------------------------------------------
# K4: a V4.1 baseline is the vendored DeepGEMM, never the stand-in nor our own wrappers
# ---------------------------------------------------------------------------
@pytest.fixture
def guard():
    pytest.importorskip("helion")
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    return hdi.vendor_baseline_guard


def test_the_k4_baseline_guard_passes_only_on_the_vendor(world, guard):
    w = world()                                   # SUFFIX_SM120=1 on SM120, the vendor can: baseline == vendor
    assert guard() is None
    w = world(gate=False)                         # the shim inert
    assert guard() is None
    sys.modules["deep_gemm"] = None               # no shim on the path at all
    assert guard() is None


@pytest.mark.parametrize("kw, why", [
    (dict(declares=False), "deep_gemm.fp8_fp4_mqa_logits resolves to the shim's Triton stand-in"),
    (dict(vendor=False), "Triton stand-in"),
    (dict(stock_env="shim"), "Triton stand-in"),
    (dict(dsa=True), "the lane's own wrappers are installed"),
])
def test_the_k4_baseline_guard_refuses_a_baseline_that_is_not_the_vendor(world, guard, kw, why):
    world(**kw)
    with pytest.raises(RuntimeError, match="a V4.1 K4 baseline must be the vendored DeepGEMM") as e:
        guard()
    assert why in str(e.value)


def test_every_v41_baseline_entry_goes_through_the_guard(monkeypatch):
    pytest.importorskip("helion")
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi

    class Refused(Exception):
        pass

    def refuse():
        raise Refused
    monkeypatch.setattr(hdi, "vendor_baseline_guard", refuse)
    for call in (lambda: hdi.decode_make_dg_from({"rows": 1}, 4),
                 lambda: hdi.decode_baseline_dg(*objs(6)),
                 lambda: hdi.prefill_baseline_any("stock", *objs(7))):
        with pytest.raises(Refused):
            call()

