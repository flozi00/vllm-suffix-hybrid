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

mm-prefix LMs (gemma-4 vision: bidirectional attention inside each image
range, sliding layers only, clamped to the window) are served WITH images
behind SUFFIX_SM120_NVP4KV_MM=1 (set only after the oracle's mm_prefix cases
pass). FI's fa2 NVFP4 paged prefill instantiates MaskMode::kCustom in the same
JIT module (modules.py: mask_mode 0..3, DefaultAttention<use_custom_mask,...>);
the builder hands it a packed bit mask only for prefill chunks holding >= 2
scheduled tokens of one image range — the only case where the TRITON_ATTN
semantics differ from causal. Text-only / decode steps keep the causal plan.
Mask cost: sum(qo_len * kv_len) bits over the step's prefill rows (worst case
max_num_batched_tokens x max_model_len / 8 = 128 MiB at 8192 x 128k), built
in <= 2 MiB-bool row blocks (no O(q*kv) bool intermediate).
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

PATCH_NAME = "sm120-nvfp4-kv"
PATCH_REVISION = "2026-09-25.11"

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
    # ---- mm-prefix (enforced only with SUFFIX_SM120_NVP4KV_MM=1) ----
    # fa2 codegen instantiates the custom-mask kernel next to causal.
    "mm_fa2_custom_mask_codegen": (
        "modules.py",
        ("DefaultAttention<use_custom_mask,", '"maybe_mask_indptr"',
         "for mask_mode in [0, 1, 2, 3]:"),
    ),
    # Mask layout the builder packs: per-request byte offset, row-major
    # qo x kv little-endian bits, ANDed with the kernel's sliding window.
    "mm_variant_mask_layout": (
        "variants.cuh",
        ("params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx]",
         "static_cast<uint64_t>(qo_idx) * kv_len + kv_idx",
         "(custom_mask_ptr[offset / 8] >> (offset % 8)) & 1",
         "mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx)"),
    ),
    # Wrapper attributes the builder sets after plan() (run() picks
    # MaskMode.CUSTOM off _custom_mask_buf and passes _mask_indptr_buf).
    "mm_wrapper_mask_buffers": (
        "prefill.py",
        ("if self._custom_mask_buf is not None:\n"
         "            mask_mode = MaskMode.CUSTOM.value",
         "self._mask_indptr_buf = mask_indptr.to(",
         "self._mask_indptr_buf,"),
    ),
}

MM_GATE_ENV = "SUFFIX_SM120_NVP4KV_MM"


def mm_gate_enabled() -> bool:
    return os.environ.get(MM_GATE_ENV, "").strip() == "1"


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
        "variants.cuh": (
            fi_root / "data" / "include" / "flashinfer" / "attention"
            / "variants.cuh"
        ),
        "prefill.py": fi_root / "prefill.py",
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


def _nvfp4_kv_cache_selected() -> bool:
    if not _use_fa2_for_nvfp4_kv_on_sm120():
        return False
    cfg = get_current_vllm_config_or_none()
    dtype = getattr(getattr(cfg, "cache_config", None), "cache_dtype", None)
    return str(dtype or "").startswith("nvfp4")


# head_dim 512 (gemma-4 full-attn) is a separate opt-in: set it only after
# the bundle's on-silicon oracle passed on this image (gemma-hd512 dossier c).
def _nvfp4_kv_head_dim_ok(head_dim: int) -> bool:
    import os

    if head_dim <= 256:
        return True
    return head_dim == 512 and (
        os.environ.get("SUFFIX_SM120_NVP4KV_HD512", "").strip() == "1"
    )

''' % PATCH_REVISION

# mm-prefix support, kept as its own block so the CPU tests and the oracle
# exec the SAME text the backend gets (mm_helper_namespace()). Self-contained
# on purpose: the patched backend must not import the bundle package.
_MM_HELPER_SRC = '''
# mm-prefix LMs (gemma-4 vision) on the fa2 route. TRITON_ATTN semantics:
#   allowed(q, k) = k < kv_len and ((k <= q and SW(q, k))
#                   or (q, k in one range [s, e] and SW(q, k) if clamped))
# SW(q, k) = q - k < window (kernel window_left = window - 1). With clamping
# (gemma-4 sliding layers) or no window (full attention) every row is ONE
# interval [.., hi(q)]: hi = q, or min(e, kv_len - 1) inside a range. The
# kernel ANDs its own window, so the packed bits only encode k <= hi(q).
def _nvfp4_kv_mm_enabled() -> bool:
    import os

    return _nvfp4_kv_cache_selected() and (
        os.environ.get("SUFFIX_SM120_NVP4KV_MM", "").strip() == "1"
    )


def _nvfp4_kv_mm_mode(vllm_config, layer_names, window_left, use_fa2) -> bool:
    """Builder-init decision: does this KV group need the mm mask? Raises
    for semantics one fa2 plan cannot reproduce (fail closed)."""
    mc = vllm_config.model_config
    if not (use_fa2 and getattr(mc, "is_mm_prefix_lm", False)):
        return False
    if not _nvfp4_kv_mm_enabled():
        raise ValueError(
            "suffix sm120 nvfp4-kv (fa2 route): mm-prefix model needs "
            "SUFFIX_SM120_NVP4KV_MM=1 (after the oracle mm_prefix cases pass)."
        )
    text = mc.hf_text_config
    # gemma-4 (use_bidirectional_attention == "vision"): the model clears
    # mm ranges on full-attention layers by name (gemma4_mm
    # _clear_mm_prefix_for_full_attn_layers); mirror that exactly.
    gemma4 = getattr(text, "use_bidirectional_attention", None) == "vision"
    cleared = set()
    if gemma4:
        cleared = {
            i for i, t in enumerate(getattr(text, "layer_types", None) or [])
            if t != "sliding_attention"
        }
    layers = vllm_config.compilation_config.static_forward_context
    modes = set()
    drafters = []
    for name in layer_names:
        layer = layers[name]
        # Spec-decode drafter layers (gemma-4 MTP: draft_model.layers.N) are
        # causal by their own config but share the KV group (one fa2 plan)
        # with target layers: they inherit the target layers' mode instead of
        # voting (drafts only; the exact target verifies every token).
        if "draft_model." in name:
            drafters.append(name)
            continue
        if not getattr(layer, "use_mm_prefix", False):
            modes.add(False)
            continue
        idx = None
        if ".layers." in name:
            try:
                idx = int(name.split(".layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                idx = None
        if idx in cleared:
            modes.add(False)
        elif window_left < 0 or getattr(
            layer, "mm_prefix_clamp_sliding_window", False
        ):
            modes.add(True)
        elif gemma4:
            # Only gemma-4's MTP drafter has unclamped sliding mm layers; it
            # gets the clamped mask (differs only for images longer than the
            # window, and drafts are verified by the exact target).
            logger.warning_once(
                "suffix sm120 nvfp4-kv: unclamped mm-prefix sliding layer %s "
                "(gemma-4 drafter) served with the clamped mask.", name
            )
            modes.add(True)
        else:
            raise ValueError(
                "suffix sm120 nvfp4-kv (fa2 route): mm-prefix bidirectional "
                f"ranges that override the sliding window ({name}) are not "
                "expressible on one fa2 plan; use TRITON_ATTN."
            )
    if len(modes) > 1:
        raise ValueError(
            "suffix sm120 nvfp4-kv (fa2 route): mm-prefix and causal layers "
            f"share one KV group ({layer_names[:4]}...); refusing."
        )
    if drafters and True in modes:
        logger.warning_once(
            "suffix sm120 nvfp4-kv: %d drafter layer(s) share an mm-prefix KV "
            "group and use its image mask (e.g. %s).", len(drafters),
            drafters[0]
        )
    return True in modes


def _nvfp4_kv_mm_resplit(cm, num_decodes):
    """Real decodes / spec-verify rows sit after the prompt, so no image
    range (prompt positions) can hold their queries: causal is exact there.
    Decode-classified rows that are still prefilling with >= 2 tokens (short
    extends) may cover 2+ tokens of one range: move the split so they take
    the masked prefill path (batch order is decode -> extend -> prefill).
    Returns the new split tuple or None."""
    if num_decodes == 0 or not any((cm.mm_req_doc_ranges or {}).values()):
        return None  # text-only batch: zero work
    if cm.is_prefilling is None:
        raise RuntimeError(
            "suffix sm120 nvfp4-kv mm-prefix: CommonAttentionMetadata has "
            "mm ranges but no is_prefilling; cannot route short extends."
        )
    qsl = cm.query_start_loc_cpu[: num_decodes + 1]
    hit = ((qsl[1:] - qsl[:-1]) >= 2) & cm.is_prefilling[:num_decodes].cpu()
    if not bool(hit.any()):
        return None
    first = int(hit.int().argmax())
    tok = int(qsl[first])
    return first, cm.num_reqs - first, tok, cm.num_actual_tokens - tok


_NVFP4_MM_BLOCK_BITS = 1 << 24  # per-block bool transient (16 MiB)


def _nvfp4_kv_mm_prefill_mask(mm_ranges, req_offset, qo_indptr, kv_lens,
                              device):
    """FlashInfer packed custom mask for one prefill plan, or None when
    causal is exact (no row holds >= 2 scheduled tokens of one range).
    Returns (packed uint8 [sum ceil(qo*kv/8)], byte indptr int32 [B+1]) —
    segment_packbits' layout; FI's plan() would store a BIT indptr for a
    pre-packed mask, so the caller installs both buffers after plan()."""
    if not mm_ranges:
        return None
    qo = [int(x) for x in qo_indptr.tolist()]
    kv = [int(x) for x in kv_lens.tolist()]
    spans = []
    for j, kvl in enumerate(kv):
        ctx = kvl - (qo[j + 1] - qo[j])
        ext = []
        for s, e in mm_ranges.get(req_offset + j) or ():
            a, b = max(s, ctx), min(e, kvl - 1)
            if s < e and b > a:
                ext.append((a - ctx, b - ctx + 1, b))
        spans.append(ext)
    if not any(spans):
        return None
    import torch

    indptr = [0]
    for j, kvl in enumerate(kv):
        indptr.append(indptr[-1] + ((qo[j + 1] - qo[j]) * kvl + 7) // 8)
    if indptr[-1] >= 1 << 31:
        raise RuntimeError(
            f"suffix sm120 nvfp4-kv mm-prefix mask of {indptr[-1]} bytes "
            "overflows FlashInfer's int32 mask_indptr."
        )
    out = torch.empty(indptr[-1], dtype=torch.uint8, device=device)
    shifts = torch.arange(8, dtype=torch.uint8, device=device)
    for j, kvl in enumerate(kv):
        ql = qo[j + 1] - qo[j]
        if ql == 0 or kvl == 0:
            continue
        hi = torch.arange(kvl - ql, kvl, device=device)
        for a, b, h in spans[j]:
            hi[a:b] = h
        cols = torch.arange(kvl, device=device)
        # Blocks of a multiple of 8 rows start byte-aligned.
        rows = max(8, _NVFP4_MM_BLOCK_BITS // kvl // 8 * 8)
        for r0 in range(0, ql, rows):
            bits = (cols[None, :] <= hi[r0:r0 + rows, None]).flatten()
            pad = -bits.numel() % 8
            if pad:
                bits = torch.cat((bits, bits.new_zeros(pad)))
            packed = (bits.view(-1, 8).to(torch.uint8) << shifts).sum(
                -1, dtype=torch.uint8)
            o = indptr[j] + r0 * kvl // 8
            out[o:o + packed.numel()] = packed
    return out, torch.tensor(indptr, dtype=torch.int32, device=device)

'''


# K2-NVFP4 (our SM120 decode/spec-verify kernel, own_attn.py) hooks. The
# adapter object is injected as the module global ``_nvfp4_own_attn`` by
# apply(); the patched backend never imports the bundle package itself.
# Env unset => both helpers are inert (stock fa2 decode, byte-identical flow).
_OWN_ATTN_HELPER_SRC = '''
def _nvfp4_own_attn_graphs_ok() -> bool:
    """UNIFORM_BATCH (FULL graphs for uniform 1+k verify) on the nvfp4 route,
    only with K2-NVFP4 armed AND the V2 runner's dispatch invariant intact:
    a batch holding ANY still-prefilling request is never uniform-decode
    (vllm/v1/worker/utils.py get_uniform_decode_token_count: `not
    has_prefill`), so an image-bearing prompt chunk can never replay a causal
    decode graph — it always takes the piecewise path, where the builder's
    H17 resplit sends it to the FA2 masked prefill. Invariant missing or
    V1 runner -> single-token graphs (fail closed, loud)."""
    import os

    if os.environ.get("SUFFIX_SM120_NVP4KV_OWN_ATTN", "").strip() != "1":
        return False
    impl = globals().get("_nvfp4_own_attn")
    if impl is None or not impl.uniform_decode_excludes_prefill():
        logger.warning_once(
            "suffix sm120 nvfp4-kv: UNIFORM_BATCH cudagraphs NOT enabled for "
            "K2-NVFP4 (V2 runner has_prefill invariant not verified); spec "
            "verify stays piecewise.")
        return False
    # FULL graphs for every uniform width 1..1+k (q_len 1 -> stock fa2
    # cudagraph decode wrappers), not only 1+k. Same invariant as above.
    impl.install_uniform_graph_lens()
    return True


def _nvfp4_own_attn_gate(builder) -> bool:
    import os

    if os.environ.get("SUFFIX_SM120_NVP4KV_OWN_ATTN", "").strip() != "1":
        return False
    impl = globals().get("_nvfp4_own_attn")
    if impl is None:
        raise RuntimeError(
            "SUFFIX_SM120_NVP4KV_OWN_ATTN=1 but the K2-NVFP4 adapter was not "
            "injected (sm120 nvfp4-kv patch apply() did not run).")
    # window/soft-cap are only set on the builder after the reorder
    # threshold; derive them the way the builder does a few lines later.
    hp = infer_global_hyperparameters(get_per_layer_parameters(
        builder.vllm_config, builder.layer_names, FlashInferImpl))
    return impl.builder_gate(builder, hp.window_left, hp.logits_soft_cap)


def _nvfp4_own_attn_q_len(qo_indptr_cpu, num_decodes) -> int:
    if num_decodes <= 0:
        return 0
    q_lens = qo_indptr_cpu[1:num_decodes + 1] - qo_indptr_cpu[:num_decodes]
    return int(q_lens.max().item())


def _nvfp4_own_attn_decode(builder, block_table, seq_lens, qo_indptr_cpu,
                           num_decodes):
    """K2 metadata for uniform spec-verify rows (q_len > 1); None for
    single-token decode, which stays on FlashInfer's fa2 decode wrapper
    (faster there, and graph-captured by the stock single-token path)."""
    q_lens = qo_indptr_cpu[1:num_decodes + 1] - qo_indptr_cpu[:num_decodes]
    q_len = int(q_lens.max().item())
    if q_len <= 1:
        return None
    real = int((q_lens > 0).sum().item())
    # Uniform rows first, CUDA-graph padding (q_len 0) only at the tail.
    if q_len < 1 or not bool((q_lens[:real] == q_len).all().item()):
        raise RuntimeError(
            "K2-NVFP4 decode needs uniform decode rows, got q_lens="
            f"{q_lens.tolist()}")
    return FIDecode(wrapper=_nvfp4_own_attn.DecodeWrapper(
        block_table[:num_decodes], seq_lens[:num_decodes], q_len,
        builder.num_qo_heads, builder.num_kv_heads, builder.head_dim,
        builder.page_size, builder.window_left, builder.sm_scale,
        builder.logits_soft_cap))

'''


def own_attn_module():
    """own_attn.py by path (the bundle and the CPU tests load this package
    from a file location, not always as an importable package)."""
    name = __name__ + ".own_attn"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("own_attn.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def mm_helper_namespace(**extra) -> dict:
    """Exec _MM_HELPER_SRC standalone (tests/oracle). ``extra`` supplies the
    backend-module globals it references (logger, _nvfp4_kv_cache_selected)."""
    import logging

    ns = {"logger": logging.getLogger(PATCH_NAME),
          "_nvfp4_kv_cache_selected": lambda: True}
    ns["logger"].warning_once = ns["logger"].warning
    ns.update(extra)
    exec(compile(_MM_HELPER_SRC, f"<{PATCH_NAME}-mm>", "exec"), ns)
    return ns


_BACKEND_EDITS = [
    # ---- H15: mm-prefix (gemma-4 vision) accepted on the fa2 nvfp4 route
    # behind its own oracle-gated opt-in; the builder applies the mask.
    (
        "supports_mm_prefix_nvfp4",
        """    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:""",
        """    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return _nvfp4_kv_mm_enabled()

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:""",
        1,
    ),
    # ---- H14: head-major KV layout on the fa2 nvfp4 route, as stock does
    # for SM100 trtllm-gen: the nvfp4 store kernel writes [K|K_sf|V|V_sf]
    # pages head-major, so NHD-family layouts (default LBNHC) corrupt reads.
    (
        "nvfp4_head_major_layouts",
        """        if capability is not None and capability.major == 10:""",
        # Keyed on the route gate alone: workers query layouts OUTSIDE the
        # vllm-config context, so a cache-dtype check reads None there (gemma
        # nvfp4 boot 20:07Z still resolved LBNHC). Head-major is valid for
        # every FlashInfer dtype (stock forces it on SM100 for all dtypes).
        """        if capability is not None and (
            capability.major == 10 or _use_fa2_for_nvfp4_kv_on_sm120()
        ):""",
        1,
    ),
    # ---- H1: module-level helper, right after logger/workspace globals.
    (
        "helper_after_logger",
        "logger = init_logger(__name__)\n\ntrtllm_workspace_buffer = None",
        "logger = init_logger(__name__)\n" + _HELPER_SRC + _MM_HELPER_SRC
        + _OWN_ATTN_HELPER_SRC
        + "\n"
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
                # vLLM's nvfp4 store kernel packs each page head-major; an
                # NHD-family layout makes the read views disagree with it
                # (garbage output, qwen-nvfp4kv-dev 2026-09-24, LBNHC).
                if get_flashinfer_layout_string(self.kv_cache_layout) != "HND":
                    raise ValueError(
                        "suffix sm120 nvfp4-kv (fa2 route) requires a "
                        "head-major KV cache layout (LBHNC/BLHNC), got "
                        f"{self.kv_cache_layout.name}."
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
            # K2-NVFP4 decodes uniform (1+k) verify rows itself with a
            # seq_lens-independent grid -> FULL graphs for spec verify.
            # Without it (FA2 decode wrapper) stay single-token.
            if _nvfp4_own_attn_graphs_ok():
                return AttentionCGSupport.UNIFORM_BATCH
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE''',
        1,
    ),
    # ---- H21: the K2-NVFP4 decode path reads block_table directly; skip the
    # per-step FlashInfer paged_kv_indices build (host work + a Triton copy
    # kernel outside the graph) when only our decode rows would need it.
    (
        "own_attn_no_paged_indices",
        """        needs_native_paged_decode = (
            num_decodes > 0 and not decode_with_flashinfer_trtllm_api
        )""",
        """        needs_native_paged_decode = (
            num_decodes > 0 and not decode_with_flashinfer_trtllm_api
            and not (getattr(self, "use_own_nvfp4_attn", False)
                     and _nvfp4_own_attn_q_len(qo_indptr_cpu, num_decodes) > 1)
        )""",
        1,
    ),
    # ---- H22: a K2-only batch (uniform verify, no prefill rows) never
    # reads seq_lens on the host (K2 takes seq_lens + block_table on the
    # GPU; H21 already skipped the paged indices), so skip the blocking
    # seq_lens.cpu() D2H — one GPU sync per KV group per verify step.
    (
        "own_attn_no_seq_lens_sync",
        """        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
""",
        """        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
        if (needs_seq_lens_cpu and num_prefills == 0 and not self.use_dcp
                and not use_cascade
                and getattr(self, "use_own_nvfp4_attn", False)
                and _nvfp4_own_attn_q_len(qo_indptr_cpu, num_decodes) > 1):
            needs_seq_lens_cpu = False
""",
        1,
    ),
    # ---- H16: per-KV-group mm mask decision at builder init (window_left
    # known here; fail closed on inexpressible semantics).
    (
        "builder_mm_prefix_mode",
        "        self.window_left = self.global_hyperparameters.window_left\n",
        """        self.window_left = self.global_hyperparameters.window_left
        self.nvfp4_mm_prefix = _nvfp4_kv_mm_mode(
            vllm_config, layer_names, self.window_left,
            getattr(self, "use_fa2_nvfp4_kv", False),
        )
""",
        1,
    ),
    # ---- H17: short extends that may hold 2+ image tokens leave the
    # (causal) decode path for the masked prefill path.
    (
        "build_mm_prefix_resplit",
        """            num_prefill_tokens = num_actual_tokens

        page_size = self.page_size""",
        """            num_prefill_tokens = num_actual_tokens
        if getattr(self, "nvfp4_mm_prefix", False):
            _mm_split = _nvfp4_kv_mm_resplit(common_attn_metadata, num_decodes)
            if _mm_split is not None:
                (num_decodes, num_prefills, num_decode_tokens,
                 num_prefill_tokens) = _mm_split

        page_size = self.page_size""",
        1,
    ),
    # ---- H18: install the packed mask on the planned fa2 prefill wrapper
    # (only when a row needs it; run() then dispatches MaskMode.CUSTOM).
    (
        "prefill_mm_prefix_mask",
        """                        disable_split_kv=self.disable_split_kv,
                    )
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)""",
        """                        disable_split_kv=self.disable_split_kv,
                    )
                    if getattr(self, "nvfp4_mm_prefix", False):
                        _mm_mask = _nvfp4_kv_mm_prefill_mask(
                            common_attn_metadata.mm_req_doc_ranges,
                            prefill_start,
                            qo_indptr_prefill_cpu,
                            kv_lens_prefill_cpu,
                            self.device,
                        )
                        if _mm_mask is not None:
                            (prefill_wrapper._custom_mask_buf,
                             prefill_wrapper._mask_indptr_buf) = _mm_mask
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)""",
        1,
    ),
    # ---- H19: K2-NVFP4 armed => uniform spec-verify rows (q_len = 1+k) are
    # decode rows (our kernel) instead of fa2 paged-prefill rows. Gate
    # evaluated once per builder; env unset => False (stock threshold).
    (
        "own_attn_spec_as_decode",
        """        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
            ),""",
        """        self.use_own_nvfp4_attn = _nvfp4_own_attn_gate(self)
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
                or self.use_own_nvfp4_attn
            ),""",
        1,
    ),
    # ---- H20: decode rows -> our op. forward() keeps calling
    # decode_wrapper.run(...); the wrapper is ours (FIDecode-compatible).
    (
        "own_attn_decode_wrapper",
        """            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0""",
        """            elif getattr(self, "use_own_nvfp4_attn", False) and (
                _own_decode := _nvfp4_own_attn_decode(
                    self, block_table_tensor, seq_lens, qo_indptr_cpu,
                    num_decodes)
            ) is not None:
                # uniform spec-verify rows -> K2; q_len 1 -> stock fa2 below
                attn_metadata.decode = _own_decode
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0""",
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
        # An earlier edit must never have consumed a later anchor.
        if out.count(old) != count:
            raise PatchDriftError(f"internal error: anchor {name} overlaps")
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
    # mm_* checks gate only the mm-prefix capability (fail closed there).
    missing = sorted(k for k, ok in probe.items()
                     if not ok and (mm_gate_enabled() or not k.startswith("mm_")))
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
    # K2-NVFP4 adapter (inert unless SUFFIX_SM120_NVP4KV_OWN_ATTN=1).
    module.__dict__["_nvfp4_own_attn"] = own_attn_module()
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
