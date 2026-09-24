# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the SM120 NVFP4-KV patch (sm120/nvfp4_kv_patch).

The patch rewrites vllm/v1/attention/backends/flashinfer.py by exact-anchor
source replacement, so the ENTIRE mechanism is verifiable without a GPU:
we replay the pure transform against the pinned v0.30.0 fixture
(tests/fixtures/vllm_0.30.0/flashinfer_backend.py) and byte-assert every
hunk, plus the drift gate, gate logic, header probe (against vendored
flashinfer 0.6.18.post1 sources), JIT-flag handling, the post-import hook,
and end-to-end `apply()` against a synthetic fake vllm/flashinfer stack.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "nvfp4_kv_patch", REPO / "sm120" / "nvfp4_kv_patch" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCH = _load_patch_module()
BACKEND_FIXTURE = FIXTURES / "vllm_0.30.0" / "flashinfer_backend.py"
FI_FIXTURE = FIXTURES / "flashinfer_0.6.18.post1"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("SUFFIX_SM120_NVP4KV", "SUFFIX_SM120_NVP4KV_ALLOW_DRIFT",
                "FLASHINFER_EXTRA_CUDAFLAGS"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Source transform against the pinned fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched():
    src = BACKEND_FIXTURE.read_text()
    new, applied = PATCH.patch_backend_source(src)
    return src, new, applied


def test_fixture_is_pinned_vllm_030():
    src = BACKEND_FIXTURE.read_text()
    assert "def get_q_data_type(self, is_prefill: bool)" in src
    assert "def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None)" in src


def test_patch_applies_all_anchors(patched):
    src, new, applied = patched
    assert len(applied) == len(PATCH._BACKEND_EDITS)
    # helper defined exactly once, before first use
    assert new.count("def _use_fa2_for_nvfp4_kv_on_sm120") == 1
    assert new.index("def _use_fa2_for_nvfp4_kv_on_sm120") < new.index(
        ") or _use_fa2_for_nvfp4_kv_on_sm120()")


def test_patch_is_idempotently_replayable(patched):
    # Re-running the raw replace-set on already-patched source must NOT
    # silently double-apply: the anchors are gone => PatchDriftError.
    src, new, _ = patched
    with pytest.raises(PATCH.PatchDriftError):
        PATCH.patch_backend_source(new)


def test_supports_kv_cache_dtype_widened(patched):
    _src, new, _ = patched
    assert (
        """            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            ) or _use_fa2_for_nvfp4_kv_on_sm120()""" in new)


def test_backend_forced_to_fa2_on_sm120_route(patched):
    _src, new, _ = patched
    # Both wrapper creation sites (prefill + decode) now pick "fa2" for the
    # SM120 route and keep trtllm-gen for SM100.
    assert new.count('else "trtllm-gen"') == 2
    assert new.count('if getattr(self, "use_fa2_nvfp4_kv", False)') >= 2
    # The stock "trtllm-gen only" comment is gone from both sites.
    assert "fa2/fa3 do not support nvfp4" not in new


def test_wrapper_backends_compile(patched):
    # The transform always leaves a compilable module (compile gate inside
    # patch_backend_source), but pin the invariant for the whole file:
    src, new, _ = patched
    compile(new, "patched_backend.py", "exec")


def test_plan_dtypes_on_fa2_route(patched):
    _src, new, _ = patched
    # o dtype falls back to model dtype on the fa2 route (2 plan sites)
    assert new.count("if self.is_kvcache_nvfp4 and not getattr(") >= 2
    # kv dtype goes torch.uint8 (FI dtype_map_kv -> __nv_fp4x2_e2m1) on the
    # fa2 route at both plan call sites.
    assert new.count("""kv_data_type=(
                            torch.uint8
                            if getattr(self, "use_fa2_nvfp4_kv", False)""") == 1
    assert new.count("""kv_data_type=(
                        torch.uint8
                        if getattr(self, "use_fa2_nvfp4_kv", False)""") == 1


def test_q_dtype_follows_backend(patched):
    _src, new, _ = patched
    assert ("""        if cache_dtype.startswith("nvfp4"):
            if getattr(self, "use_fa2_nvfp4_kv", False):
                return self.model_config.dtype
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")"""
            in new)


def test_fp8_output_detour_disabled_on_fa2_route(patched):
    _src, new, _ = patched
    # 4 forward() needs_fp8_out sites gain the fa2 exclusion; the Impl-side
    # scratch buffer allocation is skipped too.
    assert new.count("and not getattr(self, 'use_fa2_nvfp4_kv', False)") >= 4
    assert ("""        if (
            self.is_kvcache_nvfp4
            and not self.use_fa2_nvfp4_kv
            and vllm_config is not None
        ):""" in new)


def test_xqa_decode_route_stays_strict(patched):
    _src, new, _ = patched
    # decode_with_xqa keeps its `assert not self.is_kvcache_nvfp4` (vLLM
    # 0.30.0 never plumbs SF into the XQA call site) and the route flag
    # suppresses XQA/trtllm decode selection for the fa2 nvfp4 route.
    assert "assert not self.is_kvcache_nvfp4" in new
    assert (") and not self.use_fa2_nvfp4_kv" in new)
    assert ("""        if getattr(self, "use_fa2_nvfp4_kv", False):
            can_use_xqa_or_trtllm_gen_decode = False""" in new)


def test_scope_guards_present(patched):
    _src, new, _ = patched
    assert "head_dim<=256 only" in new          # scope: head_dim<=256
    assert "does not support " in new           # DCP + sinks refusals
    assert "attention sinks; use fp8 KV cache" in new
    assert new.count(
        'UNIFORM_SINGLE_TOKEN_DECODE') >= 2      # cudagraph downgrade added


def test_drift_fails_closed():
    src = BACKEND_FIXTURE.read_text()
    drifted = src.replace(
        "FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype",
        "FP8_DTYPE if self.is_kvcache_nvfp4 else some_other_dtype", 1)
    with pytest.raises(PATCH.PatchDriftError) as exc:
        PATCH.patch_backend_source(drifted)
    assert "plan_o_dtype_model_on_fa2" in str(exc.value)


def test_helper_env_gate_is_self_contained():
    # The injected helper reads the env var + current_platform, NOT the
    # patch package: the patched backend stays importable in workers that
    # never put /plugins back on sys.path.
    _src_new, applied = PATCH.patch_backend_source(BACKEND_FIXTURE.read_text())
    helper = _src_new[_src_new.index("def _use_fa2_for_nvfp4_kv_on_sm120"):
                      _src_new.index("trtllm_workspace_buffer = None")]
    assert 'os.environ.get("SUFFIX_SM120_NVP4KV"' in helper
    assert "is_device_capability_family(120)" in helper
    assert "nvfp4_kv_patch" not in helper


# ---------------------------------------------------------------------------
# FlashInfer header probe (0.6.11 overlay patches 01-03 => no-ops on 0.6.18)
# ---------------------------------------------------------------------------

def _fi_fixture_files():
    return {
        "page.cuh": (FI_FIXTURE / "page.cuh").read_text(),
        "prefill.cuh": (FI_FIXTURE / "prefill.cuh").read_text(),
        "modules.py": (FI_FIXTURE / "jit_attention" / "modules.py").read_text(),
        "tvm_ffi_utils.h": (FI_FIXTURE / "tvm_ffi_utils.h").read_text(),
    }


def test_header_probe_all_features_upstream_on_0_6_18():
    results = PATCH.probe_flashinfer_headers(_fi_fixture_files())
    assert all(results.values()), results


def test_header_probe_fails_on_missing_feature():
    files = _fi_fixture_files()
    # Simulate a pre-#3684 prefill.cuh (no in-kernel de-swizzle branch).
    files["prefill.cuh"] = files["prefill.cuh"].replace(
        "FLASHINFER_PAGED_V_SF_DESWIZZLE", "SOMETHING_ELSE")
    results = PATCH.probe_flashinfer_headers(files)
    assert not results["prefill_cuh_sf_strides"]
    # and an old page.cuh without independent V strides
    files["page.cuh"] = "// legacy"
    results = PATCH.probe_flashinfer_headers(files)
    assert not results["page_cuh_v_strides"]
    assert not results["prefill_cuh_sf_strides"]


def test_deswizzle_flag_management():
    assert PATCH.ensure_deswizzle_flag() is True
    assert os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] == (
        PATCH.DESWIZZLE_FLAG)
    assert PATCH.ensure_deswizzle_flag() is False  # idempotent
    os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = "-DFOO=1"
    assert PATCH.ensure_deswizzle_flag() is True
    assert os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] == (
        f"-DFOO=1 {PATCH.DESWIZZLE_FLAG}")
    assert PATCH.ensure_deswizzle_flag() is False


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def test_gate_off_means_noop(monkeypatch):
    monkeypatch.delenv("SUFFIX_SM120_NVP4KV", raising=False)
    assert PATCH.apply() is False
    assert PATCH.install_post_import_hook() is False
    assert "nvfp4_kv_patch" not in os.environ.get(
        "FLASHINFER_EXTRA_CUDAFLAGS", "")


def test_gate_on_non_sm120_inert(monkeypatch):
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV", "1")
    monkeypatch.setattr(PATCH, "is_sm120", lambda capability=None: False)
    assert PATCH.apply() is False          # inert, not an error
    assert "FLASHINFER_EXTRA_CUDAFLAGS" not in os.environ


def test_capability_helper_never_raises(monkeypatch):
    # No torch installed / no CUDA: _capability() -> None, is_sm120 -> False.
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)  # force import failure
    assert PATCH._capability() is None
    assert PATCH.is_sm120() is False


# ---------------------------------------------------------------------------
# apply() end-to-end against a synthetic fake vllm/flashinfer stack
# ---------------------------------------------------------------------------

FAKE_STACK = textwrap.dedent('''
    import os, shutil, sys, types, importlib.util
    from pathlib import Path

    repo = Path(r"{repo}")
    fixtures = repo / "sm120" / "tests" / "fixtures"

    # --- fake vllm package with the real v0.30.0 backend source on disk and
    # fake heavy deps injected BEFORE the backend module executes.
    vllm = types.ModuleType("vllm"); vllm.__version__ = "0.30.0"
    vllm.__path__ = []
    def _fake(name, **attrs):
        m = types.ModuleType(name); m.__path__ = []
        for k, v in attrs.items(): setattr(m, k, v)
        sys.modules[name] = m; return m
    class _Any:
        def __init__(self, *a, **k): self.__dict__.update(k)
        def __getattr__(self, k): return _Any()
        def __call__(self, *a, **k): return _Any()
        def __or__(self, other): return object
        def __ror__(self, other): return object
    _fake("vllm", **{{}}) if False else sys.modules.setdefault("vllm", vllm)
    _fake("vllm._custom_ops")
    _fake("vllm.envs")
    _fake("vllm.config", CUDAGraphMode=_Any(), VllmConfig=_Any(),
          get_current_vllm_config_or_none=lambda: None)
    _fake("vllm.config.cache", CacheDType=str)
    _fake("vllm.distributed", __path__=[]); _fake("vllm.distributed.parallel_state",
          get_dcp_group=_Any())
    import logging
    _fake("vllm.logger", init_logger=logging.getLogger)
    _fake("vllm.model_executor", __path__=[]); _fake("vllm.model_executor.layers", __path__=[])
    _fake("vllm.model_executor.layers.quantization", __path__=[])
    _fake("vllm.model_executor.layers.quantization.utils", __path__=[])
    _fake("vllm.model_executor.layers.quantization.utils.quant_utils",
          QuantKey=object, kFp8StaticTensorSym=object(), kNvfp4Dynamic=object())
    _fake("vllm.platforms", current_platform=_Any())
    _fake("vllm.platforms.interface", DeviceCapability=tuple)
    _fake("vllm.triton_utils", tl=_Any(), triton=_Any())
    _fake("vllm.utils", __path__=[])
    _fake("vllm.utils.flashinfer",
          can_use_trtllm_attention=lambda *a, **k: False,
          flashinfer_xqa_batch_decode_with_kv_cache=_Any(),
          force_use_trtllm_attention=lambda: None,
          supports_trtllm_attention=lambda **k: False,
          use_trtllm_attention=lambda *a, **k: False,
          pin_host_range_buf=lambda *a, **k: None)
    _fake("vllm.utils.gpu_sync_debug", gpu_sync_allowed=lambda: None)
    _fake("vllm.utils.math_utils", cdiv=lambda a, b: (a + b - 1) // b)
    import torch
    _fake("vllm.utils.torch_utils", PIN_MEMORY=False,
          canonicalize_singleton_dim_strides=lambda t: t,
          get_dtype_size=torch.finfo(torch.float32).bits // 8,
          is_quantized_kv_cache=lambda d: str(d) != "auto",
          is_strictly_contiguous=lambda t: True,
          nvfp4_kv_cache_full_dim=lambda h: h // 2 + h // 16,
          nvfp4_split_data_scale=_Any())
    _fake("vllm.v1", __path__=[]); _fake("vllm.v1.attention", __path__=[])
    _fake("vllm.v1.attention.backend", AttentionBackend=type("AttentionBackend", (), {{
            "supports_kv_cache_dtype": classmethod(lambda cls, d: d is None or True),
            "__class_getitem__": classmethod(lambda cls, item: cls)}}),
          AttentionCGSupport=_Any(), AttentionImpl=object,
          AttentionMetadataBuilder=type("AttentionMetadataBuilder", (), {{
            "__init__": lambda self, *a, **k: None,
            "__class_getitem__": classmethod(lambda cls, item: cls)}}),
          AttentionType=_Any(), CommonAttentionMetadata=_Any(), MultipleOf=_Any())
    _fake("vllm.v1.attention.backends", __path__=[])
    _fake("vllm.v1.attention.backends.utils",
          get_dcp_local_seq_lens=_Any(), get_flashinfer_layout_string=lambda x: "HND",
          get_num_attention_heads_from_layers=lambda *a: 0,
          get_per_layer_parameters=lambda *a: [],
          infer_global_hyperparameters=lambda p: _Any(),
          log2_lse_to_ln=_Any(), split_decodes_and_prefills=_Any())
    _fake("vllm.v1.attention.ops", __path__=[])
    _fake("vllm.v1.attention.ops.dcp", cp_lse_ag_out_rs=_Any(), dcp_a2a_lse_reduce=_Any())
    _fake("vllm.v1.attention.ops.merge_attn_states", merge_attn_states=_Any())
    _fake("vllm.v1.kv_cache_interface", AttentionSpec=_Any(), KVCacheLayout=_Any(),
          KVCacheSpec=_Any(), KVQuantMode=_Any(NONE=0), iter_layer_specs=lambda s: [])
    _fake("vllm.v1.utils", CpuGpuBuffer=_Any())

    # fake flashinfer with the pinned version + real 0.6.18 header fixtures,
    # staged as a package tree so read_installed_flashinfer() finds them
    fi = types.ModuleType("flashinfer"); fi.__version__ = "0.6.18.post1"
    fi_stage = Path({str_backend_pkg!r}).parents[2] / "flashinfer_pkg"
    (fi_stage / "data" / "include" / "flashinfer" / "attention").mkdir(parents=True, exist_ok=True)
    (fi_stage / "data" / "csrc").mkdir(parents=True, exist_ok=True)
    (fi_stage / "jit" / "attention").mkdir(parents=True, exist_ok=True)
    fx = fixtures / "flashinfer_0.6.18.post1"
    shutil.copy(fx / "page.cuh", fi_stage / "data" / "include" / "flashinfer" / "page.cuh")
    shutil.copy(fx / "prefill.cuh", fi_stage / "data" / "include" / "flashinfer" / "attention" / "prefill.cuh")
    shutil.copy(fx / "tvm_ffi_utils.h", fi_stage / "data" / "csrc" / "tvm_ffi_utils.h")
    shutil.copy(fx / "jit_attention" / "modules.py", fi_stage / "jit" / "attention" / "modules.py")
    fi.__path__ = [str(fi_stage)]
    _fake("flashinfer", **{{k: _Any() for k in (
        "BatchAttentionWithAttentionSinkWrapper",
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
        "MultiLevelCascadeAttentionWrapper", "get_seq_lens")}})
    sys.modules["flashinfer"].__version__ = fi.__version__
    sys.modules["flashinfer"].__path__ = fi.__path__
    _fake("flashinfer.decode", fast_decode_plan=_Any(),
          trtllm_batch_decode_with_kv_cache=_Any())
    _fake("flashinfer.prefill", trtllm_batch_context_with_kv_cache=_Any())
    _fake("flashinfer.utils", FP4Tensor=_Any())

    # stage the on-disk backend module exactly where apply() reads it
    pkg = Path({str_backend_pkg!r})
    (pkg / "backends").mkdir(parents=True, exist_ok=True)
    (pkg / "backends" / "flashinfer.py").write_text(
        (repo / "sm120" / "tests" / "fixtures" / "vllm_0.30.0" /
         "flashinfer_backend.py").read_text())

    # fake torch.cuda capability: real torch is present (CPU) — monkey the
    # patch module instead (below).
    spec = importlib.util.spec_from_file_location(
        "nvfp4_kv_patch", repo / "sm120" / "nvfp4_kv_patch" / "__init__.py")
    patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
    patch.is_sm120 = lambda capability=None: True   # pretend cc 12.0
    os.environ["SUFFIX_SM120_NVP4KV"] = "1"

    mod_name = "vllm.v1.attention.backends.flashinfer"
    bspec = importlib.util.spec_from_file_location(
        mod_name, pkg / "backends" / "flashinfer.py")
    backend = importlib.util.module_from_spec(bspec)
    sys.modules[mod_name] = backend
    bspec.loader.exec_module(backend)

    # non-SM120 first: inert (with is_sm120 faked False)
    patch.is_sm120 = lambda capability=None: False
    assert patch.apply(backend) is False
    assert not hasattr(backend, patch.MARKER_ATTR)
    patch.is_sm120 = lambda capability=None: True

    assert patch.apply(backend) is True
    assert getattr(backend, patch.MARKER_ATTR) == patch.PATCH_REVISION
    assert callable(backend._use_fa2_for_nvfp4_kv_on_sm120)
    # idempotent
    assert patch.apply(backend) is True
    # flashinfer JIT env flag got the de-swizzle define
    assert "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1" in os.environ[
        "FLASHINFER_EXTRA_CUDAFLAGS"]
    print("E2E-OK")
''')


def test_apply_end_to_end_with_fakes(tmp_path):
    backend_pkg = tmp_path / "vllm" / "v1" / "attention"
    script = FAKE_STACK.format(repo=str(REPO),
                               str_backend_pkg=str(backend_pkg))
    env = dict(os.environ, SUFFIX_SM120_NVP4KV="1")
    env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, timeout=300)
    assert "E2E-OK" in proc.stdout, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def test_apply_fails_closed_on_version_drift(tmp_path, monkeypatch):
    # vllm reports a different version -> RuntimeError (fail closed), unless
    # ALLOW_DRIFT is set (then anchors still decide).
    import sys, types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "9.9.9.dev0"
    fake_fi = types.ModuleType("flashinfer")
    fake_fi.__version__ = "0.1.0"
    fake_fi.__path__ = [str(tmp_path)]
    fake_fi.__spec__ = importlib.machinery.ModuleSpec(
        "flashinfer", loader=None, is_package=True)
    fake_fi.__spec__.submodule_search_locations = fake_fi.__path__
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV", "1")
    monkeypatch.setattr(PATCH, "is_sm120", lambda capability=None: True)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(PATCH.importlib.util, "find_spec",
                        lambda name: fake_fi.__spec__ if name == "flashinfer"
                        else importlib.util.find_spec(name))
    monkeypatch.setattr(
        PATCH.importlib, "import_module",
        lambda name: fake_fi if name == "flashinfer" else
        importlib.import_module(name))
    with pytest.raises(RuntimeError, match="version drift"):
        PATCH.apply(module=types.ModuleType("m"))
    monkeypatch.setenv("SUFFIX_SM120_NVP4KV_ALLOW_DRIFT", "1")
    # With drift allowed, apply proceeds to the header probe, which fails
    # closed here (empty package dir => missing features).
    with pytest.raises(RuntimeError, match="lacks the upstream NVFP4-FA2"):
        PATCH.apply(module=types.ModuleType("m"))


# ---------------------------------------------------------------------------
# sitecustomize integration (subprocess, PYTHONPATH=/tmp fake bundle)
# ---------------------------------------------------------------------------

def test_sitecustomize_arms_hook_only_when_gated(tmp_path):
    bundle = tmp_path / "plugins"
    (bundle / "nvfp4_kv_patch").mkdir(parents=True)
    (bundle / "nvfp4_kv_patch" / "__init__.py").write_bytes(
        (REPO / "sm120" / "nvfp4_kv_patch" / "__init__.py").read_bytes())
    (bundle / "sitecustomize.py").write_bytes(
        (REPO / "sitecustomize.py").read_bytes())
    probe = textwrap.dedent('''
        import sys
        armed = [f for f in sys.meta_path
                 if type(f).__name__ == "_PostImportFinder"]
        print("ARMED", len(armed))
    ''')
    env = dict(os.environ, PYTHONPATH=str(bundle))
    env.pop("SUFFIX_SM120_NVP4KV", None)
    off = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, env=env, timeout=120)
    assert off.returncode == 0, off.stderr
    assert "ARMED 0" in off.stdout          # gate off: fully inert
    env["SUFFIX_SM120_NVP4KV"] = "1"
    on = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                        text=True, env=env, timeout=120)
    assert on.returncode == 0, on.stderr
    assert "ARMED 1" in on.stdout           # gate on: hook installed
    assert "armed" in on.stderr.lower()     # and announced


# ---------------------------------------------------------------------------
# Bundle builder integration lives in scripts/test_runtime_bundle.py; pin the
# import-name contract here too (sitecustomize imports it top-level).
# ---------------------------------------------------------------------------

def test_patch_module_is_stdlib_only_at_import():
    # Importing the patch module must not pull torch/vllm (it runs from
    # sitecustomize in every process including API servers).
    script = (
        "import importlib.util, sys;"
        "spec = importlib.util.spec_from_file_location("
        "'nvfp4_kv_patch', r'" + str(REPO / 'sm120' / 'nvfp4_kv_patch' /
                                     '__init__.py') + "');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "print('TORCH' if 'torch' in sys.modules else 'NOTORCH',"
        "      'VLLM' if 'vllm' in sys.modules else 'NOVLLM')"
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    assert "NOTORCH NOVLLM" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Layout contract tests (pure CPU; mirror of the pinned vLLM/CUDA constants —
# vllm/utils/torch_utils.py:547 nvfp4_kv_cache_full_dim and the swizzle in
# csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:34-48/285-300)
# ---------------------------------------------------------------------------

def test_nvfp4_full_dim_layout():
    # data (fp4x2-packed: hs/2 bytes) + scales (e4m3, one per 16-element
    # block within a head: hs/16 bytes). Block size is 16, NOT 32.
    for hs in (64, 128, 256):
        assert PATCH.nvfp4_full_dim(hs) == hs // 2 + hs // 16


def test_nvfp4_scale_block_layout():
    # Per head: one e4m3 scale byte per 16 fp4 elements. For a 256 head the
    # packed page dim is 128 data bytes + 16 SF bytes.
    assert PATCH.SF_VEC_SIZE == 16
    assert PATCH.nvfp4_full_dim(256) == 144


def test_v_scale_swizzle_roundtrip():
    # vLLM's store kernel writes V scales in the 4-token swizzled layout
    # (nvfp4_kv_cache_kernels.cu swizzle_scale_offset): (t,s) ->
    # (t',s') = ((t//4)*4 + s//G, (s%G)*4 + t%4) with G = scale_dim//4.
    # FlashInfer's FA2 prefill kernel de-swizzles by READING, for logical
    # (entry, dcol), the byte at (entry&~3 + dcol//G, (dcol%G)*4 + entry&3)
    # — i.e. it applies the same permutation to the read index (verified
    # against prefill.cuh's FLASHINFER_PAGED_V_SF_DESWIZZLE branch). Check
    # the store is a bijection AND the read position equals the store
    # position for every logical (t, s).
    for scale_dim in (4, 8, 16):          # head_size 64/128/256
        for block_size in (4, 16, 64):    # pool uses 64
            g = scale_dim // 4
            written = {}
            for t in range(block_size):
                for s in range(scale_dim):
                    off = PATCH.swizzle_scale_offset(t, s, scale_dim)
                    assert off not in written, "not a bijection"
                    written[off] = (t, s)
                    # prefill.cuh read position for logical (entry=t, dcol=s):
                    a4, e = t & ~3, t & 3
                    read_t = a4 + s // g
                    read_s = (s % g) * 4 + e
                    assert (read_t, read_s) == (
                        off // scale_dim, off % scale_dim
                    ), (scale_dim, t, s)


def test_v_swizzle_inverse_matches_flashinfer_prefill():
    # prefill.cuh de-swizzle: entry-group a4=entry&~3, e=entry&3,
    # swz_entry=a4+dcol/SF_GROUPS, swz_sd=(dcol%SF_GROUPS)*4+e — a pure
    # permutation of the linear (entry, dcol) space.
    for scale_dim in (4, 8, 16):
        s_group = scale_dim // 4
        seen = set()
        for entry in range(64):           # enough page rows
            for dcol in range(scale_dim):
                swz = ((entry & ~3) + dcol // s_group, (dcol % s_group) * 4 + (entry & 3))
                assert swz not in seen
                seen.add(swz)
        assert len(seen) == 64 * scale_dim


def test_capacity_ratio_fp8_to_nvfp4():
    # Pool gemma-spec-dev: 8 KV heads x head_dim 256. fp8 KV = 2*H*hs*1
    # bytes/token; nvfp4 = 2*H*(hs/2+hs/16). => 56.25% of fp8 (1.78x).
    hs, heads = 256, 8
    fp8 = 2 * heads * hs
    nv = 2 * heads * PATCH.nvfp4_full_dim(hs)
    assert fp8 / nv == 16 / 9  # 1.777...
