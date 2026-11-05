# SPDX-License-Identifier: Apache-2.0
"""Every GLM-5.3 source patch replays on the vLLM 0.31.1 nightly the glm pool
moves to (vllm/vllm-openai:nightly-2eb491e18b1ce5b8a25e4c88f32dbb5d028d525c,
0.31.1rc1.dev273+g2eb491e18): pinned upstream copies in
fixtures/vllm_0.31.1rc1.dev273/, the same pure transforms apply() runs."""
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "sm120")]
import fp8_dense_patch as FD  # noqa: E402
import hisparse_mtp_patch as HS  # noqa: E402
import nvfp4_ds_mla_patch as DS  # noqa: E402
import nvfp4_kv_patch as KV  # noqa: E402

from suffix_hybrid import nccl_split as NS  # noqa: E402

NIGHTLY = Path(__file__).resolve().parent / "fixtures/vllm_0.31.1rc1.dev273"
VERSION = "0.31.1rc1.dev273+g2eb491e18"
SHA256 = {
    "cuda_communicator.py": "0a9ad3c5de17595fe5e2490fae7973d2078bc7f481ab40b0f9a48611ceac0588",
    "flashinfer_mla_sparse_sm120.py": "c0f5d6991a0226cb2b793e3aa957ff96f6a94a7055f39c7023f3f0b09842274a",
    "flashinfer_backend.py": "6cddac75c203ae2c8a257b275d42777d83466eb76c36f64d8e73630e79ace760",
    "flashinfer_mla_sparse.py": "18fc3700b535888fc221bad1ca8b3340508d144a7bacd318008720caf04337d1",
    "hisparse_layout.py": "a4fc44db41edd0eb8834c948f014b2e7971220de59c0e4e21917174802605f90",
    "hisparse_runtime.py": "e63c4d78f930106f4bc6b2454633548c258699220aba563fd20dd0ba9091497f",
    "index_group.py": "c84a0b9592c52ef999efeb92713cd6d12f4821c1bcc7203d78540afc688f0407",
    "model_loader_utils.py": "3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6",
    "sparse_mla_attention.py": "59b5cc01f45c9a843db53413fcf970acb90d14cef0282115276b055b98d282d2",
}


def src(name):
    return (NIGHTLY / name).read_text()


def test_fixtures_are_pinned_upstream_copies():
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in NIGHTLY.glob("*.py")} == SHA256


def test_versions_gate_nightly_in_and_unvalidated_out():
    for mod in (HS, DS, FD):
        assert VERSION.startswith(mod.SUPPORTED_VLLM) and "0.30.0".startswith(mod.SUPPORTED_VLLM)
        assert not "0.31.0".startswith(mod.SUPPORTED_VLLM)


def test_hisparse_spec_as_decode_is_upstream_and_marker_still_set():
    new, applied = HS.patch_source(src("sparse_mla_attention.py"))
    assert "hisparse_spec_as_decode(upstream)" in applied
    assert "self._suffix_hisparse_mtp = _suffix_sm120_hisparse_mtp(self)" in new
    assert "hisparse_supports_multi_token_decode" not in new  # upstream dropped the flag
    assert {"hisparse_prefill_without_backend", "prepare_metadata_guard"} <= set(applied)


def test_hisparse_runtime_keeps_pool_pinning_and_leaves_swap_rows_stock():
    new = HS.patch_runtime_source(src("hisparse_runtime.py"), src("index_group.py"))
    assert "_suffix_pin_range" in new and "def _get_max_swap_rows" not in new
    with pytest.raises(HS.PatchDriftError, match="max_swap_rows dispatch"):
        HS.patch_runtime_source(src("hisparse_runtime.py"), "")


def test_hisparse_mirror_staging_is_gone_upstream():
    assert HS.patch_nostage_source(src("hisparse_runtime.py")) is None
    HS.check_layout_source(src("hisparse_layout.py"))


@pytest.mark.parametrize("module,fixture", [
    (DS.BACKEND_MODULE, "flashinfer_mla_sparse.py"),
    (DS.IMPL_MODULE, "flashinfer_mla_sparse_sm120.py"),
    (DS.INDEX_GROUP_MODULE, "index_group.py"),
])
def test_nvfp4_ds_mla_anchors_replay(module, fixture):
    new, applied = DS.patch_source(module, src(fixture))
    assert applied and DS.HELPER_TAG in new


def test_fp8_dense_and_nccl_split_anchors_replay():
    assert FD.patch_source(src("model_loader_utils.py"))[1] == ["convert_after_post_load"]
    assert "_suffix_pick" in NS.patch_all_reduce(src("cuda_communicator.py"))


# ---- SM120 NVFP4 KV (gemma-4 / qwen-fable pools) on the nightly ----
FI070 = Path(__file__).resolve().parent / "fixtures/flashinfer_0.7.0.post1"


def test_nvfp4_kv_full_030_edit_set_drifts_on_nightly():
    # upstream rewrote the route: the 0.30.0 set must refuse, never half-apply
    with pytest.raises(KV.PatchDriftError, match="builder_route_and_gate"):
        KV.patch_backend_source(src("flashinfer_backend.py"))


def test_nvfp4_kv_nightly_keeps_only_mm_prefix_and_hostfree():
    new, applied = KV.patch_backend_source(src("flashinfer_backend.py"), nightly=True)
    assert set(applied) == set(KV._NIGHTLY_KEEP) | {
        "nightly_alias_builder", "nightly_alias_impl", "nightly_alias_default"}
    # our flag rides upstream's (builder + impl), never re-routes
    assert new.count("self.nvfp4_fa2 and _use_fa2_for_nvfp4_kv_on_sm120()") == 2
    assert "self.use_fa2_nvfp4_kv = False" in new
    # mm-prefix + hostfree wiring present; the 0.30.0 route / K2 edits are not
    for needle in ("return _nvfp4_kv_mm_enabled()", "_nvfp4_kv_mm_resplit(",
                   "_nvfp4_kv_mm_prefill_mask(", "self._fi_hostfree = _fi_plan_hostfree(self)",
                   "build = _fi_hostfree_build(build)"):
        assert needle in new, needle
    for gone in ("use_own_nvfp4_attn = _nvfp4_own_attn_gate", "_nvfp4_exact_seq_lens_cpu(self,",
                 "suffix sm120 nvfp4-kv (fa2 route) requires a head-major"):
        assert gone not in new, gone
    # alias lands before the code that reads it (mm mode, hostfree buffers)
    assert new.index("self.use_fa2_nvfp4_kv = (") < new.index("self._fi_hostfree =") \
        < new.index("def build(")
    assert new.index("self.use_fa2_nvfp4_kv = (") < new.index("self.nvfp4_mm_prefix =")
    compile(new, "<nightly flashinfer>", "exec")


def test_nvfp4_kv_header_probe_passes_on_fi_070():
    files = {n: (FI070 / n).read_text() for n in
             ("page.cuh", "prefill.cuh", "tvm_ffi_utils.h", "variants.cuh", "prefill.py")}
    files["modules.py"] = (FI070 / "jit_attention" / "modules.py").read_text()
    assert all(KV.probe_flashinfer_headers(files).values())


def test_nvfp4_kv_nightly_writer_is_linear():
    # why the nightly skips the de-swizzle define and refuses K2: FlashInfer's
    # slot-mapping writer stores block j of token t at v_sf + t*stride_n + j
    page = (FI070 / "page.cuh").read_text()
    kernel = page[page.index("NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel("):]
    assert "v_sf_out + block_idx" in kernel[:6000]
    assert "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping" in src("flashinfer_backend.py")
