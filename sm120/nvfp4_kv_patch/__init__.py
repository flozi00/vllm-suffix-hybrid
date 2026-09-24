# SPDX-License-Identifier: Apache-2.0
"""SM120 NVFP4-KV cache enablement for vLLM 0.30.0 + FlashInfer 0.6.18.post1.

Purpose
-------
Make ``--kv-cache-dtype nvfp4`` work on SM120 (RTX PRO 6000 Blackwell,
cc 12.0/12.1) inside the stock ``vllm/vllm-openai:v0.30.0`` image, as a pure
plugin-bundle member (no image rebuild). SM120 has no trtllm-gen NVFP4 *prefill*
cubins, and vLLM 0.30.0 hard-gates NVFP4 KV on ``trtllm-gen`` for both phases,
so the stock stack rejects the flag outright. FlashInfer 0.6.18.post1 *does*
have the arch-generic FA2 NVFP4 kernels for cc 12 (explicit SF strides +
in-kernel paged V-SF de-swizzle behind ``FLASHINFER_PAGED_V_SF_DESWIZZLE``), so
this patch re-routes NVFP4 KV through the wrapper ``fa2`` backend on SM120 —
the same route the standalone ``vllm-nvfp4-kv-sm120`` overlay proved
end-to-end (1.78x fp8 KV pool, decode parity, MTP + CUDA-graph safe).

Activation
----------
Gated on ``SUFFIX_SM120_NVP4KV=1`` AND compute capability family 12. Set but
not SM120: inert with one log line (safe to leave set fleet-wide, mirrors the
deep_gemm shim's posture). Enabled on SM120: **fail closed** — every failure
raises SystemExit/RuntimeError so a pool that asked for NVFP4 KV never
silently serves with a different config.

Mechanics (why not file-shadowing or plain method rebinding)
------------------------------------------------------------
* vLLM hook point: the bundle is not ``pip install``-ed, so there is no
  ``vllm.general_plugins`` entry point; ``sitecustomize.py`` at the bundle
  root is the hook. It runs before ``vllm`` is imported, so we install a
  tiny ``sys.meta_path`` post-import hook (stdlib only, no torch/vllm import
  at interpreter start). The first import of
  ``vllm.v1.attention.backends.flashinfer`` triggers :func:`apply` on the
  freshly-exec'd module — before any engine/backend selection happens.
* Patch style: exact-anchor *source text* rewrite (see
  ``_BACKEND_EDITS``), ``compile`` + ``exec`` into the live module's own
  ``__dict__``. Anchors are unique-text-verified; any drift from the pinned
  vLLM/FlashInfer versions raises with the failing anchor names instead of
  half-patching. The rewrite is a pure function (``patch_backend_source``),
  replayed byte-exactly in the CPU test suite against the pinned v0.30.0
  fixture, so the whole mechanism is verifiable without a GPU.
* FlashInfer side: the 0.6.11-overlay header patches (01-03) are **no-ops**
  on 0.6.18.post1 — the features they added (independent K/V strides in
  ``page.cuh``; explicit ``sf_stride_*`` + in-kernel paged V-SF de-swizzle +
  ``DTypeK``/``DTypeV`` split in ``prefill.cuh``; the SF-stride codegen in
  ``jit/attention/modules.py`` reading ``GetFP4ScaleStrides``) all landed
  upstream. We *probe* for them (:func:`probe_flashinfer_headers`) and fail
  closed if any is missing — we never blindly overwrite site-packages.
  One knob upstream shipped off-by-default: the V-SF de-swizzle branch is
  compiled only with ``-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1``. vLLM's own
  NVFP4 store kernel writes V scales 4-token swizzled (K linear), so the
  flag is mandatory for the vLLM FA2 route; we append it to
  ``FLASHINFER_EXTRA_CUDAFLAGS`` (honoured by flashinfer.jit.cpp_ext for
  every JIT op) at apply() time — before any attention kernel is planned.

Scope: GQA/MHA only, head_dim <= 256, or 512 with SUFFIX_SM120_NVP4KV_HD512=1
(enforced; set only after ``python -m nvfp4_kv_patch.oracle`` passes). MLA never selects this
backend; DCP, attention sinks, cascade and trtllm-gen/XQA paths stay strict.
XQA NVFP4 decode exists in FI 0.6.18 but vLLM 0.30.0 does not plumb scale
factors into the XQA call site, so the ``decode_with_xqa`` assert is kept.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

PATCH_NAME = "sm120-nvfp4-kv"
PATCH_REVISION = "2026-09-24.2"

TARGET_MODULE = "vllm.v1.attention.backends.flashinfer"

# Pinned targets for this port. The anchor edits + header probe are only
# validated against these; drift fails closed (override: SUFFIX_SM120_NVP4KV_
# ALLOW_DRIFT=1 downgrades the version gate to a warning — anchors still
# fail-close on real text drift).
PINNED_VLLM = "0.30.0"
PINNED_FLASHINFER = "0.6.18.post1"

GATE_ENV = "SUFFIX_SM120_NVP4KV"
DESWIZZLE_FLAG = "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"

MARKER_ATTR = "__suffix_nvfp4_kv_revision__"


class PatchDriftError(RuntimeError):
    """Installed sources do not match the pinned anchor text: refuse to patch."""


# ---------------------------------------------------------------------------
# Gate helpers (stdlib-only at import; torch touched lazily, CUDA never at
# plain-import time so API-server / non-GPU processes stay cheap).
# ---------------------------------------------------------------------------

def gate_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def _capability() -> tuple | None:
    """Device capability without ever exploding in a GPU-less process."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability())
    except Exception:
        return None


def is_sm120(capability: tuple | None = None) -> bool:
    cap = capability if capability is not None else _capability()
    return cap is not None and cap[0] == 12


def allow_drift() -> bool:
    return os.environ.get(GATE_ENV + "_ALLOW_DRIFT", "").strip() == "1"


# ---------------------------------------------------------------------------
# NVFP4 KV layout contract (reference implementations matching the pinned
# vLLM 0.30.0 store kernel, csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu,
# and vllm/utils/torch_utils.py:547 nvfp4_kv_cache_full_dim). Kept here so
# the CPU test suite can byte-verify the layout FI's FA2 kernels assume:
#   * elements are e2m1 packed fp4x2 (hs/2 bytes per head per token),
#   * scales are e4m3 with ONE byte per 16-element block within a head
#     (hs/16 bytes) — block size 16, NOT 32,
#   * page = [K_data | K_scale | V_data | V_scale] (per KV side, contiguous),
#   * K scales linear (t,s); V scales 4-token swizzled (the trtllm
#     swizzle in swizzle_scale_offset) — FI prefill.cuh de-swizzles with the
#     exact inverse behind FLASHINFER_PAGED_V_SF_DESWIZZLE.
# ---------------------------------------------------------------------------

SF_VEC_SIZE = 16  # fp4 elements per e4m3 scale block (kernel CVT_FP4_SF_VEC_SIZE)


def nvfp4_full_dim(head_size: int) -> int:
    """packed last dim = fp4 data (hs/2B) + e4m3 scales (hs/16B)."""
    if head_size % 16:
        raise ValueError("head_size must be divisible by 16 for NVFP4 KV")
    return head_size // 2 + head_size // 16


def swizzle_scale_offset(t: int, s: int, scale_dim: int) -> int:
    """V-SF store swizzle (nvfp4_kv_cache_kernels.cu:42-48)."""
    s_group = scale_dim // 4
    swizzled_t = (t // 4) * 4 + (s // s_group)
    swizzled_s = (s % s_group) * 4 + (t % 4)
    return swizzled_t * scale_dim + swizzled_s


# ---------------------------------------------------------------------------
# FlashInfer header/codegen probe. All overlay header patches (01-03) are
# subsumed by 0.6.18.post1; this proves it on the *installed* package instead
# of overwriting it. Takes the four feature files so it runs on CPU against
# vendored fixtures (sm120/tests/fixtures/flashinfer_0.6.18.post1/).
# ---------------------------------------------------------------------------

HEADER_CHECKS = {
    # page.cuh: independent V strides (overlay patch 01, upstream equivalent).
    "page_cuh_v_strides": ("page.cuh", ("v_stride_page", "v_strides")),
    # prefill.cuh: explicit SF strides + in-kernel paged V-SF de-swizzle
    # (overlay patch 02). The macro ships default-0: apply() turns it on via
    # FLASHINFER_EXTRA_CUDAFLAGS, it must merely EXIST here.
    "prefill_cuh_sf_strides": (
        "prefill.cuh",
        ("page_produce_kv_sf", "sf_stride_page", "FLASHINFER_PAGED_V_SF_DESWIZZLE"),
    ),
    # jit/attention/modules.py: SF-stride setter codegen for fa2 modules
    # (overlay patch 03, upstream equivalent).
    "jit_codegen_sf_strides": (
        "modules.py",
        ("_append_nvfp4_sf_stride_setter", "GetFP4ScaleStrides"),
    ),
    # csrc/tvm_ffi_utils.h: GetFP4ScaleStrides derives page/n/h strides from
    # the SF tensor views vLLM already passes (interleaved-cache aware).
    "ffi_sf_stride_derivation": ("tvm_ffi_utils.h", ("GetFP4ScaleStrides",)),
}


def probe_flashinfer_headers(files: dict) -> dict:
    """files: {basename: text}. Returns {check_name: bool}."""
    results = {}
    for name, (basename, needles) in HEADER_CHECKS.items():
        text = files.get(basename, "")
        results[name] = bool(text) and all(n in text for n in needles)
    return results


def read_installed_flashinfer(fi_root: Path) -> dict:
    """Read the four probe files out of an installed flashinfer package dir."""
    paths = {
        "page.cuh": fi_root / "data" / "include" / "flashinfer" / "page.cuh",
        "prefill.cuh": (
            fi_root / "data" / "include" / "flashinfer" / "attention" / "prefill.cuh"
        ),
        "modules.py": fi_root / "jit" / "attention" / "modules.py",
        "tvm_ffi_utils.h": fi_root / "data" / "csrc" / "tvm_ffi_utils.h",
    }
    out = {}
    for name, p in paths.items():
        try:
            out[name] = p.read_text(errors="replace")
        except OSError:
            out[name] = ""
    return out


def ensure_deswizzle_flag() -> bool:
    """Append the V-SF de-swizzle nvcc flag to FI's JIT env hook.

    Returns True if the env var was modified. flashinfer.jit.cpp_ext reads
    FLASHINFER_EXTRA_CUDAFLAGS at ninja-generation time for every op; the
    build.ninja content changes with the flag, so ninja rebuilds affected
    ops automatically even against a pre-seeded JIT cache. Must run before
    the first attention plan() (the general hook point does)."""
    current = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if DESWIZZLE_FLAG in current:
        return False
    os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (
        (current + " " + DESWIZZLE_FLAG).strip()
    )
    return True


# ---------------------------------------------------------------------------
# vLLM backend source rewrite. Each edit is (name, old, new, expected_count)
# against the pinned v0.30.0 file; counts are verified BEFORE any replacement
# so a drifted file raises PatchDriftError listing every bad anchor at once.
# Mirrors the semantics of the proven overlay patch 04, ported to v0.30.0's
# structure (per-phase q dtypes, FlashInferDecodeKernel routing,
# customize_spec cache layout, fp8-out buffers).
# ---------------------------------------------------------------------------

_HELPER_SRC = '''
# --- suffix sm120 nvfp4-kv patch (%s): SM120 lacks trtllm-gen NVFP4 prefill
# cubins; route NVFP4 KV through the wrapper fa2 backend (FI 0.6.18 FA2 nvfp4
# kernels, cc 12). Gated + fail-closed from the bundle plugin; inert unless
# SUFFIX_SM120_NVP4KV=1 on capability family 12.
def _use_fa2_for_nvfp4_kv_on_sm120() -> bool:
    import os

    if os.environ.get("SUFFIX_SM120_NVP4KV", "").strip() != "1":
        return False
    try:
        return current_platform.is_device_capability_family(120)
    except Exception:
        return False


# head_dim 512 (gemma-4 full-attn) is a separate opt-in: set it only after
# the bundle's on-silicon oracle passed on this image (gemma-hd512 dossier c).
def _nvfp4_kv_head_dim_ok(head_dim: int) -> bool:
    import os

    if head_dim <= 256:
        return True
    return head_dim == 512 and (
        os.environ.get("SUFFIX_SM120_NVP4KV_HD512", "").strip() == "1"
    )


# mm-prefix LMs (gemma-4: bidirectional attention over image-token ranges)
# are not implemented on the fa2 route. Plain causal attention is exact only
# when no multimodal input can ever arrive, i.e. --language-model-only.
def _nvfp4_kv_text_only_mm() -> bool:
    if not _use_fa2_for_nvfp4_kv_on_sm120():
        return False
    cfg = get_current_vllm_config_or_none()
    mm = getattr(getattr(cfg, "model_config", None), "multimodal_config", None)
    return bool(mm is not None and mm.language_model_only)

''' % PATCH_REVISION

_BACKEND_EDITS = [
    # ---- H0: mm-prefix (gemma-4 vision-bidirectional) accepted on the fa2
    # route only under --language-model-only, where no mm range can exist.
    (
        "supports_mm_prefix_text_only",
        """    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:""",
        """    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return _nvfp4_kv_text_only_mm()

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:""",
        1,
    ),
    # ---- H1: module-level helper, right after logger/workspace globals.
    (
        "helper_after_logger",
        "logger = init_logger(__name__)\n\ntrtllm_workspace_buffer = None",
        "logger = init_logger(__name__)\n" + _HELPER_SRC + "\n"
        "trtllm_workspace_buffer = None",
        1,
    ),
    # ---- H2: config gate — accept nvfp4 on SM120 (fa2 route) in addition to
    # the stock SM100 trtllm-gen requirement.
    (
        "supports_kv_cache_dtype",
        """        if kv_cache_dtype is not None and kv_cache_dtype.startswith("nvfp4"):
            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            )""",
        """        if kv_cache_dtype is not None and kv_cache_dtype.startswith("nvfp4"):
            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            ) or _use_fa2_for_nvfp4_kv_on_sm120()""",
        1,
    ),
    # ---- H3a: builder — record the fa2 route, bypass the trtllm-only raise,
    # enforce the verified scope (head_dim<=256, 512 behind its oracle-gated
    # opt-in; no DCP).
    (
        "builder_route_and_gate",
        '''            self.is_kvcache_nvfp4 = self.cache_dtype.startswith("nvfp4")
            if self.is_kvcache_nvfp4:
                if (
                    force_use_trtllm_attention() is False''',
        '''            self.is_kvcache_nvfp4 = self.cache_dtype.startswith("nvfp4")
            self.use_fa2_nvfp4_kv = (
                self.is_kvcache_nvfp4 and _use_fa2_for_nvfp4_kv_on_sm120()
            )
            if self.use_fa2_nvfp4_kv:
                if self.use_dcp:
                    raise ValueError(
                        "suffix sm120 nvfp4-kv (fa2 route) does not support "
                        "decode context parallelism."
                    )
                if not _nvfp4_kv_head_dim_ok(self.head_dim):
                    raise ValueError(
                        "suffix sm120 nvfp4-kv (fa2 route) supports "
                        f"head_dim<=256 only, got {self.head_dim} (512 needs "
                        "SUFFIX_SM120_NVP4KV_HD512=1 after the on-silicon "
                        "oracle `python -m nvfp4_kv_patch.oracle` passed)."
                    )
                logger.info_once(
                    "suffix sm120 nvfp4-kv ACTIVE: NVFP4 KV on SM120 routed "
                    "through the FlashInfer fa2 backend (head_dim=%d).",
                    self.head_dim,
                )
            if self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv:
                if (
                    force_use_trtllm_attention() is False''',
        1,
    ),
    # ---- H3b: builder non-quantized branch keeps the attribute defined.
    (
        "builder_route_default",
        '''            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False''',
        '''            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            self.use_fa2_nvfp4_kv = False''',
        1,
    ),
    # ---- H3c: sinks + fa2 nvfp4 is unsupported (the wrapper sink branch
    # keeps its assert); fail with a precise message at builder init instead.
    (
        "builder_sinks_guard",
        """        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not FlashInferBackend.supports_sink():""",
        """        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and getattr(self, "use_fa2_nvfp4_kv", False):
            raise ValueError(
                "suffix sm120 nvfp4-kv (fa2 route) does not support "
                "attention sinks; use fp8 KV cache for this model."
            )
        if self.has_sinks and not FlashInferBackend.supports_sink():""",
        1,
    ),
    # ---- H4: keep the FA2 wrapper decode route: suppress the XQA/trtllm-gen
    # decode selection for nvfp4 on SM120 (vLLM 0.30.0 does not plumb scale
    # factors into the XQA call site, so the XQA path stays strict).
    (
        "builder_decode_route",
        """        can_use_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        )""",
        """        can_use_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        )
        if getattr(self, "use_fa2_nvfp4_kv", False):
            can_use_xqa_or_trtllm_gen_decode = False""",
        1,
    ),
    # ---- H5: fa2 prefill/decode consume model-dtype queries (FP8-Q belongs
    # to the trtllm-gen route only; q-quant decision follows the backend).
    (
        "q_data_type_nvfp4_fa2",
        '''        if cache_dtype.startswith("nvfp4"):
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")''',
        '''        if cache_dtype.startswith("nvfp4"):
            if getattr(self, "use_fa2_nvfp4_kv", False):
                return self.model_config.dtype
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")''',
        1,
    ),
    # ---- H6: prefill wrapper backend: fa2 instead of trtllm-gen.
    (
        "prefill_wrapper_backend",
        '''                    # NVFP4 KV cache requires the trtllm-gen backend inside
                    # the wrapper; fa2/fa3 do not support nvfp4.
                    backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"''',
        '''                    # suffix nvfp4-kv: SM120 uses the wrapper fa2 backend
                    # (no trtllm-gen nvfp4 cubins on cc 12); SM100 keeps
                    # trtllm-gen.
                    backend = (
                        "fa2"
                        if getattr(self, "use_fa2_nvfp4_kv", False)
                        else "trtllm-gen"
                        if self.is_kvcache_nvfp4
                        else "auto"
                    )''',
        1,
    ),
    # ---- H7: decode wrapper backend: fa2 instead of trtllm-gen.
    (
        "decode_wrapper_backend",
        '''            # NVFP4 KV cache requires the trtllm-gen backend inside
            # the wrapper; fa2/fa3 do not support nvfp4.
            backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"''',
        '''            # suffix nvfp4-kv: SM120 uses the wrapper fa2 backend
            # (no trtllm-gen nvfp4 cubins on cc 12); SM100 keeps
            # trtllm-gen.
            backend = (
                "fa2"
                if getattr(self, "use_fa2_nvfp4_kv", False)
                else "trtllm-gen"
                if self.is_kvcache_nvfp4
                else "auto"
            )''',
        1,
    ),
    # ---- H8: plan() output dtype: fa2 writes model dtype (both the prefill
    # and the decode plan sites). Single-line expression swap =>
    # indentation-agnostic across the two nesting depths.
    (
        "plan_o_dtype_model_on_fa2",
        "FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype",
        "FP8_DTYPE\n                        if self.is_kvcache_nvfp4 and not getattr(\n                            self, 'use_fa2_nvfp4_kv', False)\n                        else self.model_config.dtype",
        2,
    ),
    # ---- H9a: prefill plan kv dtype: torch.uint8 (FI maps uint8 ->
    # __nv_fp4x2_e2m1 in dtype_map_kv) on the fa2 route; the "nvfp4" string
    # is only understood by the trtllm-gen wrapper route.
    (
        "prefill_plan_kv_dtype",
        """                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,""",
        """                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=(
                            torch.uint8
                            if getattr(self, "use_fa2_nvfp4_kv", False)
                            else self.kv_cache_dtype
                        ),
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,""",
        1,
    ),
    # ---- H9b: decode plan (fast_plan_decode) kv dtype, same reasoning.
    (
        "decode_plan_kv_dtype",
        """                    q_data_type=self.q_data_type_decode,
                    kv_data_type=self.kv_cache_dtype,""",
        """                    q_data_type=self.q_data_type_decode,
                    kv_data_type=(
                        torch.uint8
                        if getattr(self, "use_fa2_nvfp4_kv", False)
                        else self.kv_cache_dtype
                    ),""",
        1,
    ),
    # ---- H10: the FP8-out detour exists only for trtllm-gen nvfp4 kernels;
    # fa2 emits model dtype directly (4 forward() sites). Single-line with
    # explicit parens (three sites are bare assignments, one is parenthesized).
    (
        "no_fp8_out_on_fa2",
        "self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE",
        "(self.is_kvcache_nvfp4 and not getattr(self, 'use_fa2_nvfp4_kv', False)\n                        and output.dtype != FP8_DTYPE)",
        4,
    ),
    # ---- H11: Impl-side route flag; keep the XQA/trtllm decode capability
    # strictly false for the fa2 nvfp4 route (XQA nvfp4 decode exists in FI
    # 0.6.18 but vLLM 0.30.0 never passes scale factors into it).
    (
        "impl_route_flag",
        """        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        )""",
        """        self.use_fa2_nvfp4_kv = (
            self.is_kvcache_nvfp4 and _use_fa2_for_nvfp4_kv_on_sm120()
        )
        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        ) and not self.use_fa2_nvfp4_kv""",
        1,
    ),
    # ---- H12: skip the max_num_tokens x heads x hd FP8 scratch on the fa2
    # route (it would just waste a slab; nothing writes it there).
    (
        "skip_fp8_out_buffer",
        """        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        if self.is_kvcache_nvfp4 and vllm_config is not None:""",
        """        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        if (
            self.is_kvcache_nvfp4
            and not self.use_fa2_nvfp4_kv
            and vllm_config is not None
        ):""",
        1,
    ),
    # ---- H13: conservative CUDA-graph support on the fa2 route: the
    # UNIFORM_BATCH tier is derived from trtllm/XQA availability which we
    # just suppressed; single-token decode graphs only (matches the overlay
    # behavior: FULL_AND_PIECEWISE with spec queries on the prefill path).
    (
        "cudagraph_support_fa2",
        '''        """Get the cudagraph support level for FlashInfer attention."""''',
        '''        """Get the cudagraph support level for FlashInfer attention."""
        if (vllm_config.cache_config.cache_dtype or "").startswith(
            "nvfp4"
        ) and _use_fa2_for_nvfp4_kv_on_sm120():
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE''',
        1,
    ),
]


def patch_backend_source(src: str) -> tuple[str, list[str]]:
    """Pure text transform of vllm/v1/attention/backends/flashinfer.py.

    Raises PatchDriftError listing every anchor whose occurrence count
    mismatched (file untouched — nothing is replaced until all counts pass).
    Returns (new_source, [applied anchor names])."""
    bad = [
        f"{name}: expected {count}, found {src.count(old)}"
        for name, old, _new, count in _BACKEND_EDITS
        if src.count(old) != count
    ]
    if bad:
        raise PatchDriftError(
            "vLLM flashinfer backend does not match the pinned anchor text "
            f"(expected vLLM {PINNED_VLLM}): " + "; ".join(bad)
        )
    applied = []
    out = src
    for name, old, new, count in _BACKEND_EDITS:
        out = out.replace(old, new)
        applied.append(name)
    if "suffix sm120 nvfp4-kv patch" not in out:
        raise PatchDriftError("internal error: helper block not injected")
    compile(out, "<nvfp4_kv_patch>", "exec")  # syntax gate
    return out, applied


# ---------------------------------------------------------------------------
# apply(): the runtime entry. Ordering: gate -> versions -> probe -> JIT flag
# -> source rewrite exec into the live module dict. Every step logs loudly.
# ---------------------------------------------------------------------------

def exec_patched_source(module, new_src: str, src_path: Path) -> None:
    """exec the rewritten source into the live module, source-introspectable.

    Triton's @jit reads kernel source via inspect.getsource. Compiled under
    the ORIGINAL filename, that resolves against the on-disk file whose line
    numbers no longer match the (longer) rewrite: late kernels raise "@jit
    functions should be defined in a Python file" (gemma-spec-dev nvfp4 boot
    2026-09-24), earlier ones would silently get the wrong source. Register
    the rewrite in linecache under its own name (mtime None = never
    invalidated by checkcache) and compile against that name."""
    import linecache

    fname = f"{src_path}.{PATCH_NAME}-{PATCH_REVISION}.py"
    linecache.cache[fname] = (
        len(new_src), None, new_src.splitlines(keepends=True), fname)
    exec(compile(new_src, fname, "exec"), module.__dict__)


def _fail(msg: str) -> None:
    raise RuntimeError(f"[suffix {PATCH_NAME}] {msg}")


def apply(module=None, *, force: bool = False) -> bool:
    """Patch the vLLM FlashInfer backend for SM120 NVFP4-KV.

    Returns True if the patch is active in the module afterwards, False if
    cleanly inert (gate off / not SM120). Raises on any inconsistency while
    the gate is enabled on SM120 (fail closed)."""
    if not gate_enabled():
        return False
    if not is_sm120():
        print(
            f"[suffix {PATCH_NAME}] present but inert: {GATE_ENV}=1 is set "
            "but this is not an SM120 (cc 12.x) GPU.",
            file=sys.stderr, flush=True,
        )
        return False

    if module is None:
        module = importlib.import_module(TARGET_MODULE)
    if getattr(module, MARKER_ATTR, None) == PATCH_REVISION and not force:
        return True  # idempotent

    # 1. Version pin. The anchor set + header probe are validated for these
    # exact releases only.
    import vllm

    vllm_ver = getattr(vllm, "__version__", "")
    fi_mod = importlib.import_module("flashinfer")
    try:
        spec = importlib.util.find_spec("flashinfer")
    except ValueError:
        # find_spec raises (not returns None) for modules injected into
        # sys.modules without a __spec__ — fall back to the module's __path__.
        spec = None
    if spec is not None and spec.submodule_search_locations:
        fi_root = Path(list(spec.submodule_search_locations)[0])
    else:
        search = list(getattr(fi_mod, "__path__", []) or [])
        if not search:
            _fail("flashinfer is not installed; cannot enable SM120 NVFP4 KV.")
        fi_root = Path(search[0])
    fi_ver = getattr(fi_mod, "__version__", "")

    drift = []
    if not vllm_ver.startswith(PINNED_VLLM):
        drift.append(f"vllm {vllm_ver!r} != {PINNED_VLLM}")
    if not fi_ver.startswith(PINNED_FLASHINFER):
        drift.append(f"flashinfer {fi_ver!r} != {PINNED_FLASHINFER}")
    if drift:
        message = (
            "version drift: " + ", ".join(drift)
            + f". This patch is validated for vllm {PINNED_VLLM} + "
              f"flashinfer {PINNED_FLASHINFER} only."
        )
        if not allow_drift():
            _fail(message + " Override with " + GATE_ENV + "_ALLOW_DRIFT=1 "
                  "(anchor checks still fail-close on text drift).")
        print(f"[suffix {PATCH_NAME}] WARNING: {message} — proceeding under "
              "ALLOW_DRIFT.", file=sys.stderr, flush=True)

    # 2. Header/codegen probe: overlay patches 01-03 must be no-ops on the
    # installed FlashInfer. Never write into the flashinfer package.
    files = read_installed_flashinfer(fi_root)
    probe = probe_flashinfer_headers(files)
    missing = sorted(k for k, ok in probe.items() if not ok)
    if missing:
        _fail(
            "the installed FlashInfer lacks the upstream NVFP4-FA2 features "
            f"this patch assumes ({', '.join(missing)} in {fi_root}). "
            "Header-level overlay is NOT attempted at runtime (by design); "
            f"the image needs flashinfer >= {PINNED_FLASHINFER}."
        )

    # 3. JIT flags (must land before the first attention plan()).
    if ensure_deswizzle_flag():
        print(
            f"[suffix {PATCH_NAME}] appended {DESWIZZLE_FLAG} to "
            "FLASHINFER_EXTRA_CUDAFLAGS (vLLM's V block-scales are 4-token "
            "swizzled; the FA2 kernel must de-swizzle them in-kernel). "
            "FA2 modules JIT-recompile once on this pod.",
            file=sys.stderr, flush=True,
        )

    # 4. Source rewrite + exec into the live module namespace. Registry
    # resolves backends by qualname, so rebinded classes in the module dict
    # take effect for everything selected after this hook (general_plugins /
    # early post-import hook both precede backend selection).
    src_path = Path(module.__file__)
    src = src_path.read_text(errors="replace")
    if "suffix sm120 nvfp4-kv patch" in src:
        # The on-disk file was already patched (unexpected: we never write to
        # disk). Executing it as-is is correct and idempotent; do not re-run
        # the anchors (they would match the *unpatched* text zero times).
        setattr(module, MARKER_ATTR, PATCH_REVISION)
        return True
    new_src, applied = patch_backend_source(src)
    exec_patched_source(module, new_src, src_path)
    setattr(module, MARKER_ATTR, PATCH_REVISION)

    print(
        f"[suffix {PATCH_NAME}] ACTIVE on SM120: NVFP4 KV -> FlashInfer fa2 "
        f"route (rev {PATCH_REVISION}; {len(applied)} source anchors applied; "
        "vllm " + vllm_ver + ", flashinfer " + fi_ver + ").",
        file=sys.stderr, flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# Deferred installation from sitecustomize (runs before vllm is importable).
# ---------------------------------------------------------------------------

class _PostImportFinder:
    """sys.meta_path hook: run apply() on the freshly-loaded backend module.

    MUST live at the FRONT of sys.meta_path. The import machinery stops at
    the FIRST finder that returns a spec; the stock finders
    (BuiltinImporter / FrozenImporter / PathFinder) answer any regular
    site-packages import before a finder APPENDED at the end is ever
    consulted. Appending was the qwen-nvfp4kv-dev crash-1 root cause
    (2026-09-24): the hook armed loudly, PathFinder answered first, the
    wrapping loader was never used, apply() never ran, and stock
    validate_configuration rejected nvfp4. Front-insertion + spec-return
    keeps the machinery executing the real loader exactly ONCE through our
    wrapper (the sched_sync v3 loader-wrap lesson: never import the target
    yourself and never return a spec for a different execution).
    """

    def __init__(self, target: str, callback):
        self.target = target
        self.callback = callback
        self.armed = True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != self.target or fullname in sys.modules:
            return None
        # Step aside FIRST: importlib.util.find_spec itself walks
        # sys.meta_path and would hit this finder again (armed flag guards
        # the same, but stepping aside is belt and braces).
        self.armed = False
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            self.armed = True
            return None
        if spec is None or spec.loader is None:
            self.armed = True
            return None
        real_loader = spec.loader
        finder = self

        class _WrappingLoader:
            def create_module(inner, spec_):
                return real_loader.create_module(spec_)

            def exec_module(inner, module):
                try:
                    real_loader.exec_module(module)
                    self.callback(module)
                finally:
                    # One-shot: the target loads at most once per process.
                    # (Do NOT touch sys.modules here: the machinery already
                    # registered the module before exec_module and removes
                    # it itself if exec_module raises — re-inserting in a
                    # finally would resurrect a half-loaded module.)
                    sys.meta_path[:] = [
                        f for f in sys.meta_path if f is not finder
                    ]

            def __getattr__(inner, name):
                return getattr(real_loader, name)

        spec.loader = _WrappingLoader()
        return spec


def install_post_import_hook() -> bool:
    """sitecustomize entry: arm the deferred patch. Stdlib-only, cheap.

    Returns True if the hook is now installed (gate on). With the gate ON,
    this must never arm-and-silently-idle: if the backend module was already
    imported we patch it immediately; if the callback ever declines (gate
    off / not SM120) the finder removes itself after firing so a later
    re-activation cannot be missed silently."""
    if not gate_enabled():
        return False
    if any(isinstance(finder, _PostImportFinder)
           and getattr(finder, "target", None) == TARGET_MODULE
           and finder.armed
           for finder in sys.meta_path):
        return True
    mod = sys.modules.get(TARGET_MODULE)
    if mod is not None:
        # Late arm (module imported before sitecustomize ran, e.g. a
        # worker that inherited an already-warm import state): patch NOW —
        # loudly, fail-closed via _hook_callback.
        print(
            f"[suffix {PATCH_NAME}] target already imported at arm time; "
            "applying patch immediately.",
            file=sys.stderr, flush=True,
        )
        _hook_callback(mod)
        return True
    # FRONT insertion is the fix for crash-1: an appended finder is never
    # consulted because PathFinder answers the import first.
    sys.meta_path.insert(0, _PostImportFinder(TARGET_MODULE, _hook_callback))
    print(
        f"[suffix {PATCH_NAME}] armed at sys.meta_path[0]: will patch "
        f"{TARGET_MODULE} on first import (SM120 check + fail-closed "
        "happen at that point).",
        file=sys.stderr, flush=True,
    )
    return True


def _hook_callback(module) -> None:
    try:
        applied = apply(module)
    except Exception as exc:
        # Fail closed: the operator explicitly enabled NVFP4 KV on (presumed)
        # SM120. SystemExit survives the import machinery and site.py's
        # swallowing, so an enabled-but-broken pool cannot serve silently.
        raise SystemExit(
            f"[suffix {PATCH_NAME}] enabled but installation FAILED: {exc}"
        ) from exc
    if not applied and is_sm120():
        # apply() cleanly declined (gate off / not SM120) — but we only armed
        # because the gate was on, and the platform probe says SM120. That
        # combination means the gate flipped off between arm and import, or
        # the capability probe is lying: both are armed-but-inert states and
        # must be loud (fail-closed doctrine), never silent.
        raise SystemExit(
            f"[suffix {PATCH_NAME}] hook fired but apply() declined while "
            f"the gate is on and the platform reports SM120 — refusing to "
            "serve unpatched (armed-but-inert guard)."
        )
    if not applied:
        print(
            f"[suffix {PATCH_NAME}] hook fired; apply() inert (gate off at "
            "import time or non-SM120 host) — NOT patched, stock FlashInfer "
            "backend in use.",
            file=sys.stderr, flush=True,
        )
