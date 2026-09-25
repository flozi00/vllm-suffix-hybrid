# SPDX-License-Identifier: Apache-2.0
"""SM120 HiSparse + MTP: multi-token (spec-verify) decode on the hot buffer.

Gate: ``SUFFIX_SM120_HISPARSE_MTP=1`` AND compute capability family 12.
Set but not SM120: inert with one log line. Enabled on SM120: fail closed.

Why (vLLM 0.30.0)
-----------------
With HiSparse on, ``SparseMLACommonMetadataBuilder._init_reorder_batch_
threshold`` (sparse_mla_attention.py:258-263) forces reorder threshold 1 and
turns spec-as-decode OFF unless the builder sets
``hisparse_supports_multi_token_decode`` — only the SM100 TRTLLM builder
(flashinfer_mla_sparse.py:276) and FlashMLA (flashmla_sparse.py:288) do. SM120
uses the plain ``FlashInferMLASparseMetadataBuilder``, so every MTP verify
step (k+1 tokens) is classified prefill and host-backed requests re-stage the
whole context per layer per step (flashinfer_mla_sparse_sm120.py:142-175).

Nothing below the flag is SM100-specific: the SM120 impl already routes
decode tokens through ``HiSparseMLAIndexGroup.convert_decode_logical_to_
physical_topk`` (index_group.py:263-360, which resolves q_len>1 step by
step into (k+2)*top-k hot rows) and feeds the FlashInfer SM120 kernel ONE
ROW PER TOKEN (q.unsqueeze(1), per-token top-k row; exactly what the SM100
TRTLLM impl does too, flashinfer_mla_sparse.py:686-690). So the flag is the
whole fix; no kernel change.

What the rewrite does (target: vllm.model_executor.layers.attention.
sparse_mla_attention — deliberately NOT flashinfer_mla_sparse*.py /
backend.py / index_group.py, which the nvfp4_ds_mla draft rewrites; the
two patches share no file and compose in any order):
  1. helper block after ``logger = init_logger(__name__)``;
  2. HiSparse branch of ``_init_reorder_batch_threshold``: keep spec-as-decode
     when the builder is exactly ``flashinfer_mla_sparse.
     FlashInferMLASparseMetadataBuilder`` (on SM120 only the SM120 backend
     uses it; SM90's builder is a subclass and is excluded);
  3. ``build()``: count + log (once) real multi-token decode batches —
     marker ``[suffix sm120-hisparse-mtp] HISPARSE-MTP-DECODE``.
"""

import importlib
import os
import sys
from pathlib import Path

PATCH_NAME = "sm120-hisparse-mtp"
PATCH_REVISION = "2026-09-25.1"
TARGET_MODULE = "vllm.model_executor.layers.attention.sparse_mla_attention"
# Modules that subclass the target's builder: once imported they hold the
# pre-rewrite base class, so a late apply() could not reach them.
DEPENDENT_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse"
PINNED_VLLM = "0.30.0"
GATE_ENV = "SUFFIX_SM120_HISPARSE_MTP"
MARKER_ATTR = "__suffix_hisparse_mtp_revision__"
HELPER_TAG = "suffix sm120 hisparse-mtp patch"


class PatchDriftError(RuntimeError):
    """Installed sources do not match the pinned anchor text: refuse to patch."""


def gate_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def is_sm120(capability=None) -> bool:
    if capability is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            capability = tuple(torch.cuda.get_device_capability())
        except Exception:
            return False
    return capability[0] == 12


_HELPER = f'''

# --- {HELPER_TAG} (rev {PATCH_REVISION}) ---
_SUFFIX_HISPARSE_MTP_STATS = {{"multi_token_decode_builds": 0,
                               "max_decode_query_len": 0}}


def _suffix_sm120_hisparse_mtp(builder) -> bool:
    cls = type(builder)
    return (cls.__name__ == "FlashInferMLASparseMetadataBuilder"
            and cls.__module__ == "{DEPENDENT_MODULE}")


def _suffix_hisparse_mtp_mark(num_decodes, num_decode_tokens, decode_max_query_len):
    stats = _SUFFIX_HISPARSE_MTP_STATS
    stats["multi_token_decode_builds"] += 1
    stats["max_decode_query_len"] = max(
        stats["max_decode_query_len"], int(decode_max_query_len))
    if stats["multi_token_decode_builds"] == 1:
        logger.info(
            "[suffix {PATCH_NAME}] HISPARSE-MTP-DECODE first multi-token "
            "decode batch: num_decodes=%d num_decode_tokens=%d "
            "decode_max_query_len=%d", num_decodes, num_decode_tokens,
            decode_max_query_len)
# --- end {HELPER_TAG} ---
'''

# (name, exact old text, new text, expected occurrence count in the pin)
EDITS = [
    (
        "helper_block",
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + _HELPER,
        1,
    ),
    (
        "hisparse_spec_as_decode",
        "        if self.vllm_config.attention_config.hisparse_config is not None:\n"
        "            reorder_batch_threshold = 1\n"
        "            if not self.hisparse_supports_multi_token_decode:\n"
        "                supports_spec_as_decode = False\n",
        "        if self.vllm_config.attention_config.hisparse_config is not None:\n"
        "            reorder_batch_threshold = 1\n"
        "            self._suffix_hisparse_mtp = _suffix_sm120_hisparse_mtp(self)\n"
        "            if self._suffix_hisparse_mtp:\n"
        "                logger.info_once(\n"
        f'                    "[suffix {PATCH_NAME}] HiSparse spec-as-decode "\n'
        '                    "enabled for %s (multi-token hot-buffer decode)",\n'
        "                    type(self).__name__,\n"
        "                )\n"
        "            if not (\n"
        "                self.hisparse_supports_multi_token_decode\n"
        "                or self._suffix_hisparse_mtp\n"
        "            ):\n"
        "                supports_spec_as_decode = False\n",
        1,
    ),
    (
        "build_marker",
        "        prefill_max_seq_len = 0\n"
        "        prefill: SparseMLAPrefillMetadata | None = None\n",
        "        if (\n"
        "            num_decodes\n"
        "            and decode_max_query_len > 1\n"
        '            and getattr(self, "_suffix_hisparse_mtp", False)\n'
        "        ):\n"
        "            _suffix_hisparse_mtp_mark(\n"
        "                num_decodes, num_decode_tokens, decode_max_query_len\n"
        "            )\n"
        "        prefill_max_seq_len = 0\n"
        "        prefill: SparseMLAPrefillMetadata | None = None\n",
        1,
    ),
]


def patch_source(src: str) -> tuple[str, list[str]]:
    """Pure transform of sparse_mla_attention.py. Every anchor is
    count-verified BEFORE any replacement (all failures reported at once)."""
    bad = [f"{n}: expected {c}, found {src.count(old)}"
           for n, old, _new, c in EDITS if src.count(old) != c]
    if bad:
        raise PatchDriftError(
            "sparse_mla_attention.py does not match the pinned anchor text "
            f"(expected vLLM {PINNED_VLLM}): " + "; ".join(bad))
    out = src
    for n, old, new, c in EDITS:
        if out.count(old) != c:
            raise PatchDriftError(f"internal error: anchor {n} overlaps")
        out = out.replace(old, new)
    compile(out, f"<{PATCH_NAME}>", "exec")  # syntax gate
    return out, [n for n, *_ in EDITS]


def exec_patched_source(module, new_src: str, src_path: Path) -> None:
    """exec into the live module under a linecache-registered name (the
    target defines @triton.jit kernels, which read source via inspect)."""
    import linecache

    fname = f"{src_path}.{PATCH_NAME}-{PATCH_REVISION}.py"
    linecache.cache[fname] = (
        len(new_src), None, new_src.splitlines(keepends=True), fname)
    exec(compile(new_src, fname, "exec"), module.__dict__)


def apply(module=None) -> bool:
    """True = patch active; False = cleanly inert (gate off / not SM120).
    Raises on any inconsistency while enabled on SM120."""
    if not gate_enabled():
        return False
    if not is_sm120():
        print(f"[suffix {PATCH_NAME}] present but inert: {GATE_ENV}=1 but "
              "this is not an SM120 (cc 12.x) GPU.", file=sys.stderr, flush=True)
        return False
    if module is None:
        module = importlib.import_module(TARGET_MODULE)
    if getattr(module, MARKER_ATTR, None) == PATCH_REVISION:
        return True
    import vllm

    ver = getattr(vllm, "__version__", "")
    if not ver.startswith(PINNED_VLLM):
        raise PatchDriftError(f"vllm {ver!r} != pinned {PINNED_VLLM}")
    src_path = Path(module.__file__)
    src = src_path.read_text()
    if HELPER_TAG in src:
        raise PatchDriftError(f"{src_path} already carries the rewrite on disk")
    new_src, applied = patch_source(src)
    exec_patched_source(module, new_src, src_path)
    setattr(module, MARKER_ATTR, PATCH_REVISION)
    print(f"[suffix {PATCH_NAME}] ACTIVE on SM120: HiSparse spec-verify tokens "
          f"-> multi-token hot-buffer decode (rev {PATCH_REVISION}; "
          f"{len(applied)} anchors; vllm {ver}).", file=sys.stderr, flush=True)
    return True


def _hook_callback(module) -> None:
    try:
        applied = apply(module)
    except Exception as exc:
        raise SystemExit(
            f"[suffix {PATCH_NAME}] enabled but installation FAILED: {exc}"
        ) from exc
    if not applied and is_sm120():
        raise SystemExit(f"[suffix {PATCH_NAME}] hook fired but apply() declined "
                         "on SM120 with the gate on (armed-but-inert guard).")


def install_post_import_hook() -> bool:
    """sitecustomize entry (stdlib only). Reuses the nvfp4_kv_patch
    front-inserted sys.meta_path finder (its crash-1 lesson applies)."""
    if not gate_enabled():
        return False
    from nvfp4_kv_patch import _PostImportFinder

    if any(isinstance(f, _PostImportFinder) and f.target == TARGET_MODULE
           and f.armed for f in sys.meta_path):
        return True
    mod = sys.modules.get(TARGET_MODULE)
    if mod is not None:
        if DEPENDENT_MODULE in sys.modules:
            raise SystemExit(
                f"[suffix {PATCH_NAME}] {DEPENDENT_MODULE} was imported before "
                "the hook armed; its builder holds the unpatched base class.")
        _hook_callback(mod)
        return True
    sys.meta_path.insert(0, _PostImportFinder(TARGET_MODULE, _hook_callback))
    print(f"[suffix {PATCH_NAME}] armed at sys.meta_path[0]: will patch "
          f"{TARGET_MODULE} on first import.", file=sys.stderr, flush=True)
    return True
