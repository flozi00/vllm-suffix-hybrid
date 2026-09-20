# SPDX-License-Identifier: Apache-2.0
"""Opt-in wrap-mode: mix suffix arbitration into eagle-family drafters.

Activated only when env ``SUFFIX_HYBRID_WRAP=1`` (default OFF) and installed
from the top-level ``sitecustomize.py`` (auto-imported by CPython when the
repo root is on PYTHONPATH). A ``sys.meta_path`` watchdog is consulted by
the import machinery on every import; whenever one of the target modules
(``vllm.v1.spec_decode.eagle`` / ``...dflash`` / ``...draft_model``) has
finished executing (present in ``sys.modules``), its Proposer class is
replaced with a subclass that mixes in suffix arbitration: call
``super().propose(...)`` to get the native draft (list[list[int]]), then per
row compare with the suffix-cache draft and return the better one (suffix
wins when its estimated score is high enough to beat the native draft).
isinstance checks in vLLM's gpu_model_runner keep passing because we
subclass.

Correctness is preserved either way because drafts are only proposals
verified by the rejection sampler. But note: for chain-based drafters
(eagle family), replacing draft tokens can degrade the drafter's internal
state for the *next* step (the chain was computed assuming its own tokens)
— output stays correct, it may just be slower. That is why wrap-mode is
opt-in and off by default.

Everything here is wrapped defensively: a failure inside wrap-mode must
never break the pod.
"""

import logging
import os
import sys

logger = logging.getLogger("suffix_hybrid")

_TARGETS = (
    ("vllm.v1.spec_decode.eagle", "EagleProposer"),
    ("vllm.v1.spec_decode.dflash", "DFlashProposer"),
    ("vllm.v1.spec_decode.draft_model", "DraftModelProposer"),
)

# Minimum suffix-cache score required to override the native draft.
_SUFFIX_WIN_THRESHOLD = 0.75


def install() -> bool:
    """Install the meta-path watchdog. No-op unless SUFFIX_HYBRID_WRAP=1.

    Never raises: a wrap-mode failure must never break the pod.
    """
    if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() != "1":
        return False
    try:
        for f in sys.meta_path:
            if getattr(f, "_suffix_hybrid_wrap", False):
                return True  # already installed
        sys.meta_path.insert(0, _WrapWatchdog())
        patch_all()
        logger.info("suffix_hybrid wrap-mode installed")
        return True
    except Exception as exc:
        logger.warning("suffix_hybrid wrap-mode install failed: %s", exc)
        return False


def _patch_module(module_name, class_name):
    """Replace module.class_name with a suffix-arbitrating subclass."""
    try:
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, class_name):
            return False
        base = getattr(module, class_name)
        if getattr(base, "_suffix_hybrid_arbitrated", False):
            return True  # already patched

        from suffix_hybrid.hybrid_proposer import HybridProposer

        hp_holder = []

        def _hp():
            if not hp_holder:
                hp_holder.append(HybridProposer())
            return hp_holder[0]

        def propose(self, *args, **kwargs):
            native = base.propose(self, *args, **kwargs)
            if not isinstance(native, list):
                return native
            patched = []
            for row in native:
                if not isinstance(row, list) or not row:
                    patched.append(row)
                    continue
                try:
                    # Best-effort: propose a suffix draft of the same length
                    # (the suffix cache has no access to the drafter's token
                    # context here; we use the draft itself as the match
                    # seed). On any failure keep the native draft.
                    draft, score, _ml = _hp().suffix_cache.speculate(row, len(row))
                    if draft and score >= _SUFFIX_WIN_THRESHOLD:
                        patched.append(draft)
                    else:
                        patched.append(row)
                except Exception:
                    patched.append(row)
            return patched

        subclass = type(
            "SuffixHybrid" + class_name,
            (base,),
            {"propose": propose, "_suffix_hybrid_arbitrated": True},
        )
        setattr(module, class_name, subclass)
        # Other vllm modules may have already bound the old class; rebind
        # them too (best effort — module objects allow attribute writes).
        for mod_name in list(sys.modules):
            if not mod_name.startswith("vllm."):
                continue
            mod = sys.modules.get(mod_name)
            if mod is not None and getattr(mod, class_name, None) is base:
                try:
                    setattr(mod, class_name, subclass)
                except Exception:
                    pass
        return True
    except Exception as exc:
        logger.warning("suffix_hybrid wrap-mode patch of %s.%s failed: %s",
                       module_name, class_name, exc)
        return False


def patch_all():
    """Patch every already-loaded target module. Returns count patched."""
    n = 0
    for module_name, class_name in _TARGETS:
        if module_name in sys.modules and _patch_module(module_name, class_name):
            n += 1
    return n


class _WrapWatchdog:
    """Meta-path finder consulted on every import.

    We never intercept an import (find_spec always returns None); we only
    opportunistically patch any target module that has finished executing
    (present in sys.modules). Cheap: three dict lookups per import.
    """

    _suffix_hybrid_wrap = True

    def find_spec(self, fullname, path=None, target=None):
        try:
            if fullname.startswith("vllm"):
                patch_all()
        except Exception:
            pass
        return None

    def find_module(self, fullname, path=None):
        self.find_spec(fullname)
        return None

    def invalidate_caches(self):
        pass