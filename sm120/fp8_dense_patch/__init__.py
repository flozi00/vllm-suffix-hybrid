# SPDX-License-Identifier: Apache-2.0
"""FP8 dense linears (W8A8, per-output-channel weight scales + dynamic
per-token activation scales) for the BF16 projections of GLM 5.3 on vLLM
0.30.0 / SM120, through vLLM's own cutlass_scaled_mm (scaled_mm_c3x_sm120,
M16/M32/M64 tiles for per-token/per-channel; NOT the block-scaled path).

Gate: ``SUFFIX_FP8_DENSE=1`` (default off). ``SUFFIX_FP8_DENSE_LAYERS``: comma
list of fnmatch globs over the module path (``named_modules`` names, e.g.
``model.layers.7.self_attn.o_proj``, MTP ``model.layers.78.eh_proj``); unset =
DEFAULT_LAYERS. DENY_LAYERS always stay BF16 (router gate, indexer
weights_proj, kv_b_proj whose weight the MLA post-load absorbs, lm_head).

What we rewrite (in memory, count-verified anchor, fixture
sm120/tests/fixtures/vllm_0.30.0/model_loader_utils.py):
  model_loader/utils.py process_weights_after_loading: after the per-layer
  quant finalize, the MLA W_UK/W_UV absorb and the model-level hook, call
  ``_suffix_fp8_dense_convert(model)``. It converts every allowlisted module
  that is a LinearBase with ``type(quant_method) is UnquantizedLinearMethod``
  (quant_method swapped) or a plain ``nn.Linear`` (MTP eh_proj; class
  swapped), BF16 weight, N % 16 == K % 16 == 0. BF16 weights are freed.
  The drafter (MTP) model is loaded through the same function.

Runtime per call: dynamic_per_token_scaled_fp8_quant + cutlass_scaled_mm
(static shapes, no host sync: CUDA-graph and torch.compile safe). A boot
self-test per (N, K) shape runs each converted shape once at M=6 against the
BF16 weight before it is freed (fails closed: the SM120 block-scaled
"cutlass_gemm_caller Invalid status" class of failure dies at load, not in
serving).

Not supported: weight reload / RL refit / sleep-mode level 2 on converted
layers (the loaders' BF16 parameters are gone).

NVFP4 mode (nvfp4.py): ``SUFFIX_NVFP4_DENSE=1`` (default off), allowlist
``SUFFIX_NVFP4_DENSE_LAYERS`` (unset = NVFP4_DEFAULT_LAYERS, Gemma 4 language
model), same anchor and DENY_LAYERS. Allowlisted LinearBase layers become
vLLM ModelOpt NVFP4 W4A4 layers (vLLM's own method + kernel selection, so
SUFFIX_NVFP4_GEMM serves them too). It runs BEFORE the FP8 mode, which then
only sees what is still UnquantizedLinearMethod; both gates may be on.
"""

import fnmatch
import os
import sys
from pathlib import Path

PATCH_NAME = "fp8-dense"
PATCH_REVISION = "2026-09-28.1"
PINNED_VLLM = "0.30.0"
GATE_ENV = "SUFFIX_FP8_DENSE"
LAYERS_ENV = "SUFFIX_FP8_DENSE_LAYERS"
NVFP4_GATE_ENV = "SUFFIX_NVFP4_DENSE"
NVFP4_LAYERS_ENV = "SUFFIX_NVFP4_DENSE_LAYERS"
MARKER_ATTR = "__suffix_fp8_dense_revision__"
HELPER_TAG = "suffix fp8-dense patch"
TARGET_MODULE = "vllm.model_executor.model_loader.utils"
FIXTURE = "model_loader_utils.py"

# Decode-path BF16 projections worth the bytes (inventory in the commit /
# dossier). Shared experts and dense MLP are TP-sharded; qkv_a / wq_b /
# eh_proj are replicated per rank.
DEFAULT_LAYERS = (
    "*.self_attn.fused_qkv_a_proj",
    "*.self_attn.q_b_proj",
    "*.self_attn.o_proj",
    "*.indexer.wq_b",
    "*.mlp.gate_up_proj",
    "*.mlp.down_proj",
    "*.shared_experts.gate_up_proj",
    "*.shared_experts.down_proj",
    "*.eh_proj",
)
# Never converted, whatever the allowlist says.
DENY_LAYERS = (
    "*.gate",             # MoE router (fp32 logits, noaux_tc top-k)
    "*weights_proj*",     # indexer wk_weights_proj (top-k head weights)
    "*kv_b_proj*",        # MLA absorb reads the BF16 weight post-load
    "*lm_head*",
    "*shared_head*",      # MTP's lm_head
    "*router*",           # Gemma 4 MoE router (router.proj)
    "*vision_tower*",     # multimodal towers / embedders stay BF16
    "*embed_vision*",
    "*audio_tower*",
    "*embed_audio*",
)
# Gemma 4 (gemma4.py / gemma4_mm.py: language_model.model.layers.N...):
# attention (q_proj = KV-shared layers) and the dense MLP; the routed experts
# are NVFP4 in the checkpoint already.
NVFP4_DEFAULT_LAYERS = (
    "*language_model*self_attn.qkv_proj",
    "*language_model*self_attn.q_proj",
    "*language_model*self_attn.o_proj",
    "*language_model*.mlp.gate_up_proj",
    "*language_model*.mlp.down_proj",
)


class PatchDriftError(RuntimeError):
    """Installed sources do not match the pinned anchor text: refuse to patch."""


def fp8_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def nvfp4_enabled() -> bool:
    return os.environ.get(NVFP4_GATE_ENV, "").strip() == "1"


def gate_enabled() -> bool:
    """Either mode arms the (shared) anchor."""
    return fp8_enabled() or nvfp4_enabled()


def _patterns(env, default) -> tuple:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return default
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def layer_patterns() -> tuple:
    return _patterns(LAYERS_ENV, DEFAULT_LAYERS)


def nvfp4_layer_patterns() -> tuple:
    return _patterns(NVFP4_LAYERS_ENV, NVFP4_DEFAULT_LAYERS)


def selected(name: str, patterns=None) -> bool:
    """Allowlisted and not denied (module path from named_modules)."""
    if any(fnmatch.fnmatchcase(name, d) for d in DENY_LAYERS):
        return False
    return any(fnmatch.fnmatchcase(name, p)
               for p in (layer_patterns() if patterns is None else patterns))


# ---------------------------------------------------------------------------
# Anchor: (name, exact old text, new text, expected count in the pin).
# ---------------------------------------------------------------------------
EDITS = [
    (
        "convert_after_post_load",
        '    # Model-level post-load hook, after the per-layer quant finalize.\n'
        '    if hasattr(model, "process_weights_after_loading"):\n'
        '        model.process_weights_after_loading()\n',
        '    # Model-level post-load hook, after the per-layer quant finalize.\n'
        '    if hasattr(model, "process_weights_after_loading"):\n'
        '        model.process_weights_after_loading()\n'
        f'    # {HELPER_TAG}: BF16 linears -> FP8 W8A8 (after the MLA absorb)\n'
        '    _suffix_fp8_dense_convert(model)\n',
        1,
    ),
]


def patch_source(src: str) -> tuple[str, list[str]]:
    """Pure transform of utils.py; anchors count-verified before replacing."""
    if HELPER_TAG in src:
        raise PatchDriftError(f"{FIXTURE} already carries the {PATCH_NAME} rewrite")
    bad = [f"{n}: expected {c}, found {src.count(old)}"
           for n, old, _new, c in EDITS if src.count(old) != c]
    if bad:
        raise PatchDriftError(
            f"{FIXTURE} does not match the pinned anchor text (expected vLLM "
            f"{PINNED_VLLM}): " + "; ".join(bad))
    out = src
    for _n, old, new, _c in EDITS:
        out = out.replace(old, new)
    compile(out, f"<{PATCH_NAME}:{FIXTURE}>", "exec")  # syntax gate
    return out, [n for n, *_ in EDITS]


def _suffix_fp8_dense_convert(model) -> None:
    if nvfp4_enabled():  # first: FP8 then skips what became NVFP4
        from . import nvfp4  # torch/vllm: worker side only

        nvfp4.convert_model(model)
    if fp8_enabled():
        from . import runtime

        runtime.convert_model(model)


def apply(module) -> bool:
    """Rewrite utils.py in place. True = active; False = gate off. Raises on
    drift. No CUDA probe here (API server imports this too): the SM120 check
    and the kernel self-test run at conversion, on the worker."""
    if not gate_enabled():
        return False
    if module.__name__ != TARGET_MODULE:
        raise PatchDriftError(f"{module.__name__} is not the {PATCH_NAME} target")
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
    module.__dict__["_suffix_fp8_dense_convert"] = _suffix_fp8_dense_convert
    import linecache

    fname = f"{src_path}.{PATCH_NAME}-{PATCH_REVISION}.py"  # introspectable
    linecache.cache[fname] = (len(new_src), None,
                              new_src.splitlines(keepends=True), fname)
    exec(compile(new_src, fname, "exec"), module.__dict__)
    setattr(module, MARKER_ATTR, PATCH_REVISION)
    print(f"[suffix {PATCH_NAME}] ACTIVE: {TARGET_MODULE} rewritten "
          f"({', '.join(applied)}; rev {PATCH_REVISION}; vllm {ver}; fp8 layers "
          f"{','.join(layer_patterns()) if fp8_enabled() else 'off'}; nvfp4 layers "
          f"{','.join(nvfp4_layer_patterns()) if nvfp4_enabled() else 'off'}).",
          file=sys.stderr, flush=True)
    return True


def _hook_callback(module) -> None:
    try:
        applied = apply(module)
    except Exception as exc:
        raise SystemExit(
            f"[suffix {PATCH_NAME}] enabled but installation FAILED on "
            f"{module.__name__}: {exc}") from exc
    if not applied:
        raise SystemExit(f"[suffix {PATCH_NAME}] hook fired but apply() "
                         "declined with the gate on (armed-but-inert guard).")


def install_post_import_hook() -> bool:
    """sitecustomize entry (stdlib only): one front-inserted one-shot finder.
    utils imported before arming = its importers (base_loader) already bound
    the stock function: refuse."""
    if not gate_enabled():
        return False
    from nvfp4_kv_patch import _PostImportFinder

    if TARGET_MODULE in sys.modules:
        raise SystemExit(f"[suffix {PATCH_NAME}] {TARGET_MODULE} imported "
                         "before the hook armed; refusing to patch a live module.")
    if not any(isinstance(f, _PostImportFinder) and f.target == TARGET_MODULE
               and f.armed for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder(TARGET_MODULE, _hook_callback))
    print(f"[suffix {PATCH_NAME}] armed at sys.meta_path[0]: will patch "
          f"{TARGET_MODULE} on first import.", file=sys.stderr, flush=True)
    return True
