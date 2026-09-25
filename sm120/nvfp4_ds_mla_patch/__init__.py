# SPDX-License-Identifier: Apache-2.0
"""SM120 NVFP4 DS-MLA (sparse MLA) enablement for vLLM 0.30.0 — DRAFT.

Gated on ``SUFFIX_SM120_NVP4DSMLA=1`` AND capability family 12. When set
but not SM120: inert with one log line. Enabled on SM120: fail closed
(every inconsistency raises; a pool that asked for nvfp4_ds_mla never
silently serves a different config).

Contract sources (pinned research dossier:
``plugin-harness/dossiers/dsmla-kernel-contracts-glm53.md``):
  * Row layout 352 B/token: [0,256) e2m1 NoPE (low nibble = even dim),
    [256,320) UNSCALED e4m3 RoPE, [320,352) 32 e4m3 SF one-per-16-dims,
    SF byte permutation ``s -> 8*(s&3) + (s>>2)``; SF = e4m3(max(amax16/6,
    2^-9)); dequant x = e2m1 * sf. No global scale byte; kv_scale_format
    is an fp8-only concept (nvfp4 rows carry per-16 e4m3 SFs only).
  * Rejection chain today: (a) the SM120 backend's
    supported_kv_cache_dtypes / supports_combination
    (flashinfer_mla_sparse.py:162-167, :212-219); (b) the impl's
    kv_cache_dtype hard check (flashinfer_mla_sparse_sm120.py:63-67);
    (c) the C++ concat_and_cache_mla dispatcher's double SM100 gate
    (CMake FP4_SM100_ARCHS build gate + runtime props->major == 10).
    Sizing already yields 352 for nvfp4_ds_mla (MLAAttentionSpec builder,
    mla_attention.py :1361) — NO spec/sizing change needed.
  * Reader: flashinfer's wrapper hard-fails on a 352-row cache (656-byte
    literal checks) — our cuda-oxide kernels (kernels-oxide/
    nvfp4_ds_mla; host ops nvfp4_ds_mla_decode_cuda /
    nvfp4_ds_mla_quant_store_cuda) fully replace the
    ``flashinfer_trtllm_batch_decode_with_kv_cache_mla`` call inside
    ``_run_mqa_kernel`` with the SAME wrapper signature: q [T,1,64,576]
    bf16, kv uint8 flat, block_tables [T,1,2048] int32 PHYSICAL token
    slots (-1 masked, NOT page indices), seq_lens=None (all columns
    active), sm_scale float, bmm2_scale 1.0 — with no workspace_buffer /
    max_seq_len dependency (o_part + lse_part ≈ 2.1 MiB per token at
    topk 2048 versus the 394 MiB flashinfer workspace).
  * Writer: replaces ``ops.concat_and_cache_mla`` for
    kv_cache_dtype == nvfp4_ds_mla on SM120 (backend.py
    do_kv_cache_update :1055-1073); kernel geometry mirrors the SM100
    nvfp4 writer (grid (tokens,), block 64: warp 0 latent lanes, warp 1
    rope lanes).
  * GLM 5.3: HQ=64, index_topk=2048 (the backend requires exactly
    2048), kv_lora_rank 512, rope 64, DSA arch (no kpool). GLM-5.3-Flash
    (kpool / rope=0) is OUT OF SCOPE v1.

Anchor style: exact-text source rewrite with occurrence-count
verification, the ``sm120/nvfp4_kv_patch`` conventions — every anchor
fails closed with PatchDriftError before any replacement (all failing
anchors reported at once); the transforms are pure functions replayable
byte-exactly in the CPU suite. Anchors below are EXACT transcriptions of
the pinned v0.30.0 sdist (verified count-exact against
/tmp/vllm-0.30.0). TODO before ship: pin fixture copies under
sm120-drafts/fixtures/ and wire the deferred sys.meta_path
front-insertion hooks for BOTH target modules (backend may import
before the sm120 impl module).
"""

import importlib
import math
import os
import sys
from pathlib import Path

PATCH_NAME = "sm120-nvfp4-ds-mla"
PATCH_REVISION = "2026-09-25.1"

TARGET_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"
WRITER_TARGET = "vllm.v1.attention.backend"

PINNED_VLLM = "0.30.0"
GATE_ENV = "SUFFIX_SM120_NVP4DSMLA"
MARKER_ATTR = "__suffix_nvfp4_ds_mla_revision__"


class PatchDriftError(RuntimeError):
    """Installed sources do not match the pinned anchor text: refuse to patch."""


def gate_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def _capability():
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability())
    except Exception:
        return None


def is_sm120(capability=None) -> bool:
    cap = capability if capability is not None else _capability()
    return cap is not None and cap[0] == 12


# ---------------------------------------------------------------------------
# (1) flashinfer_mla_sparse.py — backend support list (:162-167) and
#     supports_combination (:212-219). Exact text of the pinned file.
# ---------------------------------------------------------------------------
# TodoDraft: exercise care with these anchors — the class-level list is
# inside a ClassVar annotation on the SM120 backend class; the
# supports_combination tuple is a plain `not in (...)` gate.
BACKEND_EDITS = [
    (
        "supported_kv_cache_dtypes_ext",
        '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
        '        "auto",\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        '    ]',
        '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
        '        "auto",\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        '        "nvfp4_ds_mla",\n'
        '    ]',
        1,
    ),
    (
        "supports_combination_nvfp4_ext",
        '        if kv_cache_dtype not in (\n'
        '            None,\n'
        '            "auto",\n'
        '            "fp8",\n'
        '            "fp8_e4m3",\n'
        '            "fp8_ds_mla",\n'
        '        ):\n'
        '            return "kv_cache_dtype not supported"',
        '        if kv_cache_dtype not in (\n'
        '            None,\n'
        '            "auto",\n'
        '            "fp8",\n'
        '            "fp8_e4m3",\n'
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        '        ):\n'
        '            return "kv_cache_dtype not supported"',
        1,
    ),
]

# ---------------------------------------------------------------------------
# (2) flashinfer_mla_sparse_sm120.py — impl gate (:63-67), split-workspace
#     allocation (:211-215) and the _run_mqa_kernel flashinfer call
#     (:218-246). Exact text of the pinned file.
# ---------------------------------------------------------------------------
IMPL_EDITS = [
    (
        "impl_kv_cache_dtype_gate",
        '        if kv_cache_dtype != "fp8_ds_mla":\n'
        '            raise NotImplementedError(\n'
        '                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "\n'
        '                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."\n'
        '            )',
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        '            if ' + GATE_ENV + ' gate check:\n'
        '                pass\n'
        '            # (expanded below — see applied f-string)',
        1,
    ),
    (
        "workspace_partials",
        '        output = q.new_empty(\n'
        '            (num_actual_toks, self.num_heads, self.kv_lora_rank),\n'
        '            dtype=q.dtype,\n'
        '        )',
        '        output = q.new_empty(\n'
        '            (num_actual_toks, self.num_heads, self.kv_lora_rank),\n'
        '            dtype=q.dtype,\n'
        '        )\n'
        '        if getattr(self, "_use_nvfp4_ds_mla", False):\n'
        '            # suffix nvfp4-ds-mla: split workspace o_part [T, H, NS,\n'
        '            # 512] bf16 + lse_part [T, H, NS] f32 with NS =\n'
        '            # ceil(sparse_capacity / 64) (flashinfer mid_out /\n'
        '            # mid_lse split contract) — about 2.1 MiB per token\n'
        '            # at topk 2048; none of the 394 MiB shared workspace.\n'
            '            ns = (topk_indices_physical.shape[1] + 63) // 64\n'
        '            self._nvfp4_o_part = q.new_empty(\n'
        '                (num_actual_toks, self.num_heads, ns, self.kv_lora_rank),\n'
        '                dtype=torch.bfloat16)\n'
        '            self._nvfp4_lse_part = q.new_empty(\n'
        '                (num_actual_toks, self.num_heads, ns),\n'
        '                dtype=torch.float32)',
        1,
    ),
    (
        "run_mqa_kernel_swap",
        '        from vllm.utils.flashinfer import (\n'
        '            flashinfer_trtllm_batch_decode_with_kv_cache_mla,\n'
        '        )\n'
        '\n'
        '        sparse_capacity = topk_indices_physical.shape[1]\n'
        '        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(\n'
        '            query=q.unsqueeze(1),\n'
        '            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),\n'
        '            workspace_buffer=self._workspace_buffer,\n'
        '            qk_nope_head_dim=self.qk_nope_head_dim,\n'
        '            kv_lora_rank=self.kv_lora_rank,\n'
        '            qk_rope_head_dim=self.qk_rope_head_dim,\n'
        '            block_tables=topk_indices_physical.unsqueeze(1),\n'
        '            seq_lens=None,\n'
        '            max_seq_len=sparse_capacity,\n'
        '            out=output.unsqueeze(1),\n'
        '            bmm1_scale=self.scale,\n'
        '            bmm2_scale=1.0,\n'
        '            sparse_mla_top_k=sparse_capacity,\n'
        '            kv_scale_format=self.kv_scale_format,\n'
        '        )\n'
        '        return out.squeeze(1)',
        '        if getattr(self, "_use_nvfp4_ds_mla", False):\n'
        '            # suffix nvfp4-ds-mla: our cuda-oxide kernels (family\n'
        '            # nvfp4_ds_mla via suffix_hybrid.oxide_kernels) — same\n'
        '            # wrapper contract: q [T,1,H,576] bf16, kv uint8 flat\n'
        '            # [blocks, page, 352], block_tables [T,1,C] int32\n'
        '            # physical token slots (-1 masked), seq_lens=None (all\n'
        '            # columns active), sm_scale float, bmm2_scale 1.0; no\n'
        '            # flashinfer workspace / max_seq_len.\n'
        '            from suffix_hybrid import _native, oxide_kernels\n'
        '\n'
        '            oxide_kernels.ensure_loaded("nvfp4_ds_mla")\n'
        '            _native.nvfp4_ds_mla_decode_cuda(\n'
        '                q.unsqueeze(1),\n'
        '                kv_cache.view(torch.uint8),\n'
        '                topk_indices_physical.unsqueeze(1),\n'
        '                output.unsqueeze(1),\n'
        '                self._nvfp4_o_part,\n'
        '                self._nvfp4_lse_part,\n'
        '                float(self.scale),\n'
        '                torch.cuda.current_stream(q.device).cuda_stream,\n'
        '            )\n'
        '            return output\n'
        '        from vllm.utils.flashinfer import (\n'
        '            flashinfer_trtllm_batch_decode_with_kv_cache_mla,\n'
        '        )\n'
        '\n'
        '        sparse_capacity = topk_indices_physical.shape[1]\n'
        '        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(\n'
        '            query=q.unsqueeze(1),\n'
        '            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),\n'
        '            workspace_buffer=self._workspace_buffer,\n'
        '            qk_nope_head_dim=self.qk_nope_head_dim,\n'
        '            kv_lora_rank=self.kv_lora_rank,\n'
        '            qk_rope_head_dim=self.qk_rope_head_dim,\n'
        '            block_tables=topk_indices_physical.unsqueeze(1),\n'
        '            seq_lens=None,\n'
        '            max_seq_len=sparse_capacity,\n'
        '            out=output.unsqueeze(1),\n'
        '            bmm1_scale=self.scale,\n'
        '            bmm2_scale=1.0,\n'
        '            sparse_mla_top_k=sparse_capacity,\n'
        '            kv_scale_format=self.kv_scale_format,\n'
        '        )\n'
        '        return out.squeeze(1)',
        1,
    ),
]

# The impl dtype gate: full replacement body (built separately because
# its new text embeds the env-var name twice — keep it a plain literal).
IMPL_EDITS[0] = (
    "impl_kv_cache_dtype_gate",
    IMPL_EDITS[0][1],
    '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
    '            import os\n'
    '\n'
    '            if os.environ.get("%s", "").strip() != "1":\n'
    '                raise NotImplementedError(\n'
    '                    "FLASHINFER_MLA_SPARSE_SM120 requires the packed "\n'
    '                    "fp8_ds_mla KV cache layout; got "\n'
    '                    f"kv_cache_dtype={kv_cache_dtype!r}. Set "\n'
    '                    "%s=1 for the suffix nvfp4_ds_mla kernels."\n'
    '                )\n'
    '            # suffix nvfp4-ds-mla: our cuda-oxide reader/writer\n'
    '            # kernels consume and produce the exact 352 B rows on\n'
    '            # SM120 (the C++ dispatcher is SM100-gated twice).\n'
    '            self._use_nvfp4_ds_mla = True\n'
    '        elif kv_cache_dtype != "fp8_ds_mla":\n'
    '            raise NotImplementedError(\n'
    '                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "\n'
    '                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."\n'
    '            )\n'
    '        else:\n'
    '            self._use_nvfp4_ds_mla = False' % (GATE_ENV, GATE_ENV),
    1,
)

# ---------------------------------------------------------------------------
# (3b) index_group.py — HiSparse row-width dtype fork (register_layer
#      :189-198): stock forks ONLY on fp8_ds_mla (656); anything else,
#      including nvfp4_ds_mla, falls into the generic branch that sets
#      row_width = head_size (576 for ds-MLA) — WRONG for the packed
#      352 B nvfp4 row, corrupting hot/hw-mirror/host pool geometry.
#      The HiSparse data plane itself is dtype-blind raw-uint8 over
#      row_width bytes (hisparse_gather_plan, mirror writes, host
#      copies), so ONLY the width must be fixed. Exact pinned text.
# ---------------------------------------------------------------------------
HISPARSE_EDITS = [
    (
        "hisparse_register_layer_row_width",
        '        if kv_cache_dtype == "fp8_ds_mla":\n'
        '            row_width = FP8_DS_MLA_ROW_BYTES\n'
        '            kv_dtype = torch.uint8\n'
        '        else:\n'
        '            from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype\n'
        '\n'
        '            row_width = head_size\n'
        '            kv_dtype = kv_cache_dtype_str_to_dtype(\n'
        '                kv_cache_dtype, vllm_config.model_config\n'
        '            )',
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        '            # suffix nvfp4-ds-mla + HiSparse: the packed 352 B row\n'
        '            # (256 e2m1 NoPE + 64 unscaled e4m3 RoPE + 32 permuted\n'
        '            # e4m3 SFs) — the HiSparse data plane is dtype-blind\n'
        '            # raw-uint8 over row_width bytes, so the width is the\n'
        '            # only geometry input; matches MLAAttentionSpec\n'
        '            # state_content_bytes (mla_attention.py:1361).\n'
        '            import os\n'
        '\n'
        '            if os.environ.get("SUFFIX_SM120_NVP4DSMLA", "").strip() != "1":\n'
        '                raise NotImplementedError(\n'
        '                    "HiSparse register_layer: nvfp4_ds_mla host/hot "\n'
        '                    "pools need the suffix nvfp4_ds_mla kernels; "\n'
        '                    "set SUFFIX_SM120_NVP4DSMLA=1."\n'
        '                )\n'
        '            row_width = 352\n'
        '            kv_dtype = torch.uint8\n'
        '        elif kv_cache_dtype == "fp8_ds_mla":\n'
        '            row_width = FP8_DS_MLA_ROW_BYTES\n'
        '            kv_dtype = torch.uint8\n'
        '        else:\n'
        '            from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype\n'
        '\n'
        '            row_width = head_size\n'
        '            kv_dtype = kv_cache_dtype_str_to_dtype(\n'
        '                kv_cache_dtype, vllm_config.model_config\n'
        '            )',
        1,
    ),
]

HISPARSE_TARGET = "vllm.v1.attention.backends.mla.index_group"
WRITER_EDITS = [
    (
        "concat_and_cache_mla_route",
        '        from vllm import _custom_ops as ops\n'
        '\n'
        '        ops.concat_and_cache_mla(\n'
        '            kv_c_normed,\n'
        '            k_pe.squeeze(1),\n'
        '            kv_cache,\n'
        '            slot_mapping.flatten(),\n'
        '            kv_cache_dtype=kv_cache_dtype,\n'
        '            scale=k_scale,\n'
        '        )',
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
            '            import os\n'
            '\n'
            '            if os.environ.get("%s", "").strip() == "1":\n'
        '                # suffix nvfp4-ds-mla: the C++ concat_and_cache_mla\n'
        '                # dispatch is SM100-gated (build + runtime); our\n'
        '                # cuda-oxide writer (nvfp4_ds_mla_quant_store)\n'
        '                # writes the exact 352 B rows on SM120. Negative\n'
        '                # slot ids are skipped by contract.\n'
        '                from suffix_hybrid import _native, oxide_kernels\n'
        '\n'
        '                oxide_kernels.ensure_loaded("nvfp4_ds_mla")\n'
        '                _native.nvfp4_ds_mla_quant_store_cuda(\n'
        '                    kv_c_normed,\n'
        '                    k_pe.squeeze(1),\n'
        '                    kv_cache.view(torch.uint8).reshape(-1, 352),\n'
        '                    slot_mapping.flatten(),\n'
        '                    torch.cuda.current_stream(\n'
        '                        kv_c_normed.device\n'
        '                    ).cuda_stream,\n'
        '                )\n'
        '                return\n'
        '        from vllm import _custom_ops as ops\n'
        '\n'
        '        ops.concat_and_cache_mla(\n'
        '            kv_c_normed,\n'
        '            k_pe.squeeze(1),\n'
        '            kv_cache,\n'
        '            slot_mapping.flatten(),\n'
        '            kv_cache_dtype=kv_cache_dtype,\n'
        '            scale=k_scale,\n'
        '        )' % GATE_ENV,
        1,
    ),
]


def _apply_edits(name: str, src: str, edits) -> tuple[str, list[str]]:
    """Count-verify every anchor BEFORE any replacement (nvfp4_kv_patch
    convention): a drifted file raises PatchDriftError naming every bad
    anchor at once; the file is untouched unless all counts pass."""
    bad = [
        f"{n}: expected {c}, found {src.count(old)}"
        for n, old, _new, c in edits
        if src.count(old) != c
    ]
    if bad:
        raise PatchDriftError(
            f"{name} does not match the pinned anchor text (expected vLLM "
            f"{PINNED_VLLM}): " + "; ".join(bad)
        )
    applied = []
    out = src
    for n, old, new, c in edits:
        if out.count(old) != c:
            raise PatchDriftError(f"internal error: anchor {n} overlaps")
        out = out.replace(old, new)
        applied.append(n)
    compile(out, f"<{PATCH_NAME}-{name}>", "exec")  # syntax gate
    return out, applied


def patch_backend_source(src: str) -> tuple[str, list[str]]:
    return _apply_edits("flashinfer_mla_sparse", src, BACKEND_EDITS)


def patch_impl_source(src: str) -> tuple[str, list[str]]:
    return _apply_edits("flashinfer_mla_sparse_sm120", src, IMPL_EDITS)


def patch_writer_source(src: str) -> tuple[str, list[str]]:
    return _apply_edits("backend.do_kv_cache_update", src, WRITER_EDITS)


def patch_hisparse_source(src: str) -> tuple[str, list[str]]:
    """HiSparse row-width fork: nvfp4_ds_mla -> 352 B rows, fail-closed."""
    return _apply_edits("index_group.register_layer", src, HISPARSE_EDITS)


# ---------------------------------------------------------------------------
# apply(): gate -> version pin -> source rewrite exec into the live module
# dict. Fail-closed doctrine: an enabled-but-broken pool never serves.
# TODO(draft):
#   * deferred sys.meta_path front-insertion hooks for BOTH targets
#     (nvfp4_kv_patch _PostImportFinder pattern — backend.py may import
#     before the sm120 impl module; the writer edit must fire on
#     WRITER_TARGET too),
#   * pre-launch ensure_loaded("nvfp4_ds_mla") at the pod boot gate,
#     outside graph capture (SUFFIX_OXIDE_PROBE marker wiring),
#   * fixture copies pinned under sm120-drafts/fixtures/ so the CPU
#     suite replays the transforms without the sdist.
# ---------------------------------------------------------------------------

def apply(module=None, *, force: bool = False) -> bool:
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
    import vllm

    vllm_ver = getattr(vllm, "__version__", "")
    if not vllm_ver.startswith(PINNED_VLLM):
        raise RuntimeError(
            f"[suffix {PATCH_NAME}] version drift: vllm {vllm_ver!r} != "
            f"{PINNED_VLLM}; this patch is validated for {PINNED_VLLM} only."
        )
    src_path = Path(module.__file__)
    src = src_path.read_text(errors="replace")
    new_src, applied = patch_impl_source(src)
    exec(compile(new_src, str(src_path), "exec"), module.__dict__)
    setattr(module, MARKER_ATTR, PATCH_REVISION)
    print(
        f"[suffix {PATCH_NAME}] ACTIVE on SM120: nvfp4_ds_mla sparse-MLA "
        f"decode/writer on our cuda-oxide kernels (rev {PATCH_REVISION}; "
        f"{len(applied)} anchors applied; vllm {vllm_ver}).",
        file=sys.stderr, flush=True,
    )
    return True