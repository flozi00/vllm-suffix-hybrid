# SPDX-License-Identifier: Apache-2.0
"""SUFFIX_FI_PLAN_HOSTFREE: the patched FlashInferMetadataBuilder.build()
never reads the device on the host (fa2 NVFP4 route, async scheduling, no
spec decode) and hands FlashInfer bit-identical plan arguments.

The REAL patched v0.30.0 backend source (pinned fixture + every anchor) is
exec'd against stub vllm/flashinfer modules; device tensors are a Tensor
subclass whose host reads (.cpu/.item/.tolist/.numpy/bool/int, D2H copy_/to)
raise, so any device->host transfer in build() fails the test.
"""

import ast
import builtins
import contextlib
import functools
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "sm120" / "tests" / "fixtures" / "vllm_0.30.0"


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "nvfp4_kv_patch_hostfree_t",
        REPO / "sm120" / "nvfp4_kv_patch" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = _load_patch()
OWN = PATCH.own_attn_module()


# --------------------------------------------------------------------------
# device-tensor detector
# --------------------------------------------------------------------------

class D2H(AssertionError):
    pass


STATE = {"allow": 0, "d2h": 0, "allowed_d2h": 0, "pins": 0}
PINNED = set()  # storage ptrs of persistent "pinned" buffers

_T = torch.Tensor
_HOST_READS = {_T.cpu, _T.item, _T.tolist, _T.numpy, _T.__bool__, _T.__int__,
               _T.__index__, _T.__float__}


def _d2h(what):
    if STATE["allow"]:
        STATE["allowed_d2h"] += 1
        return
    STATE["d2h"] += 1
    raise D2H(f"device->host read in build(): {what}")


class Dev(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        to_host = func is _T.cpu
        if func in _HOST_READS:
            _d2h(getattr(func, "__name__", func))
        elif func is _T.to:
            tgt = [a for a in args[1:] if isinstance(a, (str, torch.device))]
            tgt += [kwargs["device"]] if "device" in kwargs else []
            to_host = any(str(t).startswith("cpu") for t in tgt)
            if to_host:
                _d2h("to(cpu)")
        elif func is _T.copy_ and not isinstance(args[0], Dev):
            _d2h("host.copy_(device)")
        out = super().__torch_function__(func, types_, args, kwargs)
        return host(out) if to_host else out


def dev(t):
    return t.as_subclass(Dev)


def host(t):
    return t.as_subclass(torch.Tensor) if isinstance(t, Dev) else t


@pytest.fixture(autouse=True)
def _pin_counter(monkeypatch):
    def pin_memory(self, *a, **k):
        if self.untyped_storage().data_ptr() not in PINNED:
            STATE["pins"] += 1  # a fresh pinned allocation in real torch
        return self
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    for k in STATE:
        STATE[k] = 0
    PINNED.clear()  # freed buffers' addresses get reused across tests
    yield


# --------------------------------------------------------------------------
# stub vllm / flashinfer + exec of the real patched backend
# --------------------------------------------------------------------------

class _AnyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return cls

    def __getitem__(cls, k):
        return cls

    def __call__(cls, *a, **k):
        return cls


class Any(metaclass=_AnyMeta):
    pass


class _Generic:
    def __class_getitem__(cls, item):
        return cls


class CpuGpuBuffer:  # vllm/v1/utils.py:110 semantics
    def __init__(self, *size, dtype, device, pin_memory=False):
        self.cpu = torch.zeros(*size, dtype=dtype)
        self.gpu = dev(torch.zeros(*size, dtype=dtype))
        self.np = self.cpu.numpy()
        if pin_memory:
            PINNED.add(self.cpu.untyped_storage().data_ptr())

    def copy_to_gpu(self, n=None):
        cpu, gpu = self.cpu[:n], self.gpu[:n]
        return gpu.copy_(cpu.pin_memory() if PIN[0] else cpu, non_blocking=True)


PIN = [True]
CALLS = []


def _rec(kind, wrapper, kw):
    CALLS.append((kind, id(wrapper), {
        k: (host(v).clone() if isinstance(v, torch.Tensor) else v)
        for k, v in kw.items()}))


class FakeWrapper:
    def __init__(self, *a, use_cuda_graph=False, **k):
        self.is_cuda_graph_enabled = use_cuda_graph

    def plan(self, *a, **kw):
        _rec("plan", self, kw)


def fake_fast_decode_plan(wrapper, **kw):
    _rec("fast_decode_plan", wrapper, kw)


def get_seq_lens(kv_indptr, kv_last_page_len, page_size):  # flashinfer/page.py
    return (torch.clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size
            + kv_last_page_len)


@contextlib.contextmanager
def gpu_sync_allowed():
    STATE["allow"] += 1
    try:
        yield
    finally:
        STATE["allow"] -= 1


@functools.lru_cache(None)
def _split_fn():
    src = (FIX / "attention_backends_utils.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                and n.name == "split_decodes_and_prefills")
    ns = {"torch": torch, "CommonAttentionMetadata": object}
    exec(compile(ast.Module([node], []), "<split>", "exec"), ns)
    return ns["split_decodes_and_prefills"]


def _stub(name):
    over = {
        "flashinfer": dict(BatchDecodeWithPagedKVCacheWrapper=FakeWrapper,
                           BatchPrefillWithPagedKVCacheWrapper=FakeWrapper,
                           get_seq_lens=get_seq_lens),
        "flashinfer.decode": dict(fast_decode_plan=fake_fast_decode_plan),
        "vllm.utils.flashinfer": dict(
            use_trtllm_attention=lambda *a, **k: False,
            can_use_trtllm_attention=lambda *a, **k: False,
            pin_host_range_buf=lambda n: None),
        "vllm.utils.gpu_sync_debug": dict(gpu_sync_allowed=gpu_sync_allowed),
        "vllm.utils.torch_utils": dict(PIN_MEMORY=True),
        "vllm.v1.utils": dict(CpuGpuBuffer=CpuGpuBuffer),
        "vllm.v1.attention.backend": dict(AttentionMetadataBuilder=_Generic,
                                          AttentionBackend=_Generic,
                                          AttentionImpl=object),
        "vllm.v1.attention.backends.utils": dict(
            split_decodes_and_prefills=_split_fn(),
            get_flashinfer_layout_string=lambda layout: "HND"),
    }.get(name, {})
    m = types.ModuleType(name)
    m.__dict__.update(over)
    m.__getattr__ = lambda attr: Any
    return m


@pytest.fixture(scope="module")
def backend():
    src, _ = PATCH.patch_backend_source((FIX / "flashinfer_backend.py").read_text())
    real_import = builtins.__import__

    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in ("vllm", "flashinfer"):
            return _stub(name)
        return real_import(name, globals, locals, fromlist, level)

    ns = {"__name__": "fi_backend_hostfree_t",
          "__builtins__": dict(builtins.__dict__, __import__=imp)}
    exec(compile(src, "<patched flashinfer.py>", "exec"), ns)
    ns["_nvfp4_own_attn"] = OWN

    def copy_page_indices(grid):  # Triton kernel -> CPU twin (no host reads
        def run(out, bt, stride, indptr, BLOCK_SIZE):  # of the build path)
            o, b, p = host(out), host(bt), host(indptr)
            for r in range(grid[0]):
                s, e = int(p[r]), int(p[r + 1])
                o[s:e] = b[r, :e - s]
        return run
    ns["_copy_page_indices_kernel"] = type(
        "K", (), {"__getitem__": lambda self, g: copy_page_indices(g)})()
    return ns


PAGE = 16


def _builder(ns, *, hostfree, own, cudagraph, spec=False):
    os.environ["SUFFIX_FI_PLAN_HOSTFREE"] = "1" if hostfree else "0"
    try:
        B = ns["FlashInferMetadataBuilder"]
        b = object.__new__(B)
        b.__dict__.update(
            reorder_batch_threshold=1, use_xqa=False, nvfp4_mm_prefix=False,
            use_own_nvfp4_attn=own, page_size=PAGE,
            attention_config=NS(use_trtllm_attention=False),
            num_qo_heads=16, num_kv_heads=8, dcp_world_size=1,
            cache_dtype="nvfp4", q_data_type_prefill=torch.bfloat16,
            q_data_type_decode=torch.bfloat16, has_sinks=False,
            use_trtllm_decode_attention=False, use_dcp=False,
            global_hyperparameters=NS(has_same_window_lefts=True,
                                      has_same_all_params=True),
            use_fa2_nvfp4_kv=True, is_kvcache_nvfp4=True,
            vllm_config=NS(scheduler_config=NS(async_scheduling=True),
                           speculative_config=NS() if spec else None),
            enable_cuda_graph=cudagraph, _decode_cudagraph_max_bs=512,
            _decode_wrappers_cudagraph={}, _decode_wrapper=None,
            _prefill_wrapper=None, model_config=NS(dtype=torch.bfloat16),
            head_dim=256, sm_scale=0.0625, window_left=-1,
            logits_soft_cap=None, kv_cache_dtype=torch.uint8,
            decode_fixed_split_size=-1, prefill_fixed_split_size=-1,
            disable_split_kv=False, device=torch.device("cpu"),
            cache_config=NS(get_resolved_kv_cache_layout=lambda: "HND"),
            _workspace_buffer=torch.zeros(1, dtype=torch.uint8))
        # H25 exactly as the patched __init__ does it
        b._fi_hostfree = ns["_fi_plan_hostfree"](b)
        b._fi_plan_ev = None
        b.paged_kv_indptr = CpuGpuBuffer(
            65, dtype=torch.int32, device=b.device,
            pin_memory=b._fi_hostfree and ns["PIN_MEMORY"])
        b.paged_kv_indices = dev(torch.zeros(64 * 64, dtype=torch.int32))
        b.paged_kv_last_page_len = CpuGpuBuffer(
            64, dtype=torch.int32, device=b.device,
            pin_memory=b._fi_hostfree and ns["PIN_MEMORY"])
        return b
    finally:
        os.environ.pop("SUFFIX_FI_PLAN_HOSTFREE", None)


def _cm(q_lens, seq_lens):
    n = len(q_lens)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    bt = torch.arange(n * 64, dtype=torch.int32).reshape(n, 64) + 7
    return NS(num_reqs=n, num_actual_tokens=int(qsl[-1]), causal=True,
              max_query_len=max(q_lens), max_seq_len=max(seq_lens),
              seq_lens=dev(sl.clone()), seq_lens_cpu_upper_bound=sl,
              block_table_tensor=dev(bt), query_start_loc=dev(qsl.clone()),
              query_start_loc_cpu=qsl, slot_mapping=dev(torch.zeros(1)),
              mm_req_doc_ranges=None, is_prefilling=None)


def _run(ns, b, cm, steps=2):
    ns["_NVFP4_SEQ_LENS_CHECKS"][0] = 100  # past the sampled-check window
    CALLS.clear()
    out = [b.build(0, cm) for _ in range(steps)]
    return out, [(k, kw) for k, _w, kw in CALLS]


def _same(a, b):
    assert len(a) == len(b) and a, (len(a), len(b))
    for (ka, wa), (kb, wb) in zip(a, b):
        assert ka == kb and wa.keys() == wb.keys()
        for k in wa:
            if isinstance(wa[k], torch.Tensor):
                assert torch.equal(wa[k], wb[k]), k
            else:
                assert wa[k] == wb[k], k


def _lens(n):
    return [1 + 37 * i % 300 for i in range(n)]  # incl. page-aligned lens


@pytest.mark.parametrize("n", [1, 9, 32])
@pytest.mark.parametrize("own", [False, True])
@pytest.mark.parametrize("cudagraph", [False, True])
def test_decode_build_is_hostfree_and_plan_identical(backend, n, own, cudagraph):
    cm = _cm([1] * n, _lens(n))
    stock = _builder(backend, hostfree=False, own=own, cudagraph=cudagraph)
    STATE["allow"] = 0
    _, stock_calls = _run(backend, stock, cm)
    k2 = own and n <= OWN.K2_Q1_MAX_BATCH
    # the detector sees the stock blocking seq_lens.cpu() (one per build)
    assert STATE["allowed_d2h"] == (0 if k2 else 2)
    stock_pins = STATE["pins"]

    STATE.update(allowed_d2h=0, pins=0)
    hf = _builder(backend, hostfree=True, own=own, cudagraph=cudagraph)
    assert hf._fi_hostfree
    md, calls = _run(backend, hf, cm)
    assert STATE["d2h"] == 0 and STATE["allowed_d2h"] == 0
    assert STATE["pins"] == 0 and (k2 or stock_pins == 10)
    if k2:
        assert calls == [] and stock_calls == []
        assert type(md[0].decode.wrapper).__name__ == "DecodeWrapper"
        return
    kinds = [k for k, _ in calls]
    assert kinds == (["plan", "fast_decode_plan"] if cudagraph
                     else ["plan", "plan"])
    _same(stock_calls, calls)
    # FA2 decode plan inputs are the exact host seq_lens
    kw = calls[0][1]
    exp = cm.seq_lens_cpu_upper_bound
    got = get_seq_lens(kw.get("indptr", kw.get("indptr_cpu")),
                       kw.get("last_page_len", kw.get("last_page_len_cpu")), PAGE)
    assert torch.equal(got, exp)


def test_mixed_batch_hostfree_and_plan_identical(backend):
    q, s = [1] * 9 + [40, 7], _lens(9) + [40, 300]
    cm = _cm(q, s)
    _, stock_calls = _run(backend, _builder(backend, hostfree=False, own=False,
                                            cudagraph=True), cm)
    STATE.update(allowed_d2h=0, pins=0)
    _, calls = _run(backend, _builder(backend, hostfree=True, own=False,
                                      cudagraph=True), cm)
    assert STATE["d2h"] == 0 and STATE["allowed_d2h"] == 0
    assert len(calls) == 4  # prefill + decode plan per build
    _same(stock_calls, calls)


def test_spec_decode_keeps_the_stock_sync(backend):
    """Async SPEC decode: the host upper bound is optimistic on decode rows,
    so the FA2 plan keeps reading the device (inside gpu_sync_allowed)."""
    cm = _cm([1] * 9, _lens(9))
    b = _builder(backend, hostfree=True, own=False, cudagraph=True, spec=True)
    _run(backend, b, cm, steps=1)
    assert STATE["d2h"] == 0 and STATE["allowed_d2h"] == 1


def test_sampled_seq_lens_check_still_fails_closed(backend):
    cm = _cm([1] * 9, _lens(9))
    b = _builder(backend, hostfree=True, own=False, cudagraph=True)
    backend["_NVFP4_SEQ_LENS_CHECKS"][0] = 0
    b.build(0, cm)
    assert STATE["allowed_d2h"] == 1  # sampled check, explicitly allowed
    cm.seq_lens = dev(host(cm.seq_lens) + 1)
    with pytest.raises(RuntimeError, match="refusing"):
        b.build(0, cm)


def test_build_leaves_shared_host_metadata_untouched(backend):
    """The runner shares seq_lens_cpu_upper_bound / query_start_loc_cpu with
    every KV group's builder (GDN, other attention groups)."""
    cm = _cm([1] * 32, _lens(32))
    ub, qsl = cm.seq_lens_cpu_upper_bound.clone(), cm.query_start_loc_cpu.clone()
    b1 = _builder(backend, hostfree=True, own=False, cudagraph=True)
    b2 = _builder(backend, hostfree=True, own=False, cudagraph=False)
    _run(backend, b1, cm)
    _run(backend, b2, cm)
    assert torch.equal(cm.seq_lens_cpu_upper_bound, ub)
    assert torch.equal(cm.query_start_loc_cpu, qsl)


def test_gate_off_and_out_of_scope_builders(backend):
    assert not _builder(backend, hostfree=False, own=False, cudagraph=True)._fi_hostfree
    os.environ["SUFFIX_FI_PLAN_HOSTFREE"] = "1"
    try:
        f = backend["_fi_plan_hostfree"]
        assert f(NS(use_fa2_nvfp4_kv=True, use_dcp=False))
        assert not f(NS(use_fa2_nvfp4_kv=False, use_dcp=False))
        assert not f(NS(use_fa2_nvfp4_kv=True, use_dcp=True))
    finally:
        os.environ.pop("SUFFIX_FI_PLAN_HOSTFREE")


def test_build_fence_waits_previous_build_then_records():
    log = []

    class Ev:
        def synchronize(self):
            log.append("wait")

        def record(self):
            log.append("record")

    fake_torch = NS(cuda=NS(Event=Ev,
                            is_current_stream_capturing=lambda: cap[0]))
    cap = [False]
    ns = {"torch": fake_torch}
    exec(compile(PATCH._OWN_ATTN_HELPER_SRC, "<h>", "exec"), ns)

    def build(self, x):
        log.append("build")
        if x == "boom":
            raise ValueError(x)
        return x

    wrapped = ns["_fi_hostfree_build"](build)
    b = NS(_fi_hostfree=True, _fi_plan_ev=None, device=NS(type="cuda"))
    assert wrapped(b, 1) == 1
    assert log == ["build", "record"]
    log.clear()
    wrapped(b, 2)
    assert log == ["wait", "build", "record"]
    log.clear()
    with pytest.raises(ValueError):
        wrapped(b, "boom")
    assert log == ["wait", "build", "record"]
    log.clear()
    cap[0] = True  # graph capture: no host waits, no event records
    wrapped(b, 3)
    assert log == ["build"]
    log.clear()
    b._fi_hostfree = False  # gate off: stock build, untouched
    wrapped(b, 4)
    assert log == ["build"]


def test_hostfree_anchors():
    new, applied = PATCH.patch_backend_source((FIX / "flashinfer_backend.py").read_text())
    assert {"hostfree_pinned_buffers", "hostfree_no_kv_lens_pin",
            "hostfree_build_fence"} <= set(applied)
    assert new.count("pin_memory=self._fi_hostfree and PIN_MEMORY") == 2
    i = new.index("    build = _fi_hostfree_build(build)")
    assert new.rindex("    def build(\n", 0, i) > new.index(
        "class FlashInferMetadataBuilder")
