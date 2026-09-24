# SPDX-License-Identifier: Apache-2.0
"""SM120-safe ``deep_gemm`` shim (directory-shadowing package).

This package is NOT the real DeepGEMM. ``scripts/runtime_bundle.py`` copies
``sm120/deep_gemm_shim/`` to ``<bundle>/deep_gemm/`` so that, with
``PYTHONPATH=/plugins``, ``import deep_gemm`` lands HERE before any
site-packages copy — which matters because vLLM's
``vllm.utils.deep_gemm._import_deep_gemm`` prefers an external ``deep_gemm``
over its own vendored ``vllm.third_party.deep_gemm``, and
``has_deep_gemm()`` accepts either.

Behaviour:

* Vendored DeepGEMM imports cleanly and this is NOT an SM120 GPU (or the
  ``SUFFIX_SM120=1`` gate is unset): every attribute delegates straight to
  the vendor — zero behaviour change.
* SM120 (``torch.cuda.get_device_capability() == (12, 0)``) with the gate set,
  OR the vendor unusable: the MQA-logits trio
  (``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits`` /
  ``get_paged_mqa_logits_metadata``) is served by a Triton fallback
  (``sm120_fallback``) implementing the FP8 contract of
  ``vllm/utils/deep_gemm.py`` exactly; MXFP4 inputs raise
  ``NotImplementedError`` with guidance.
* No vendor AND not SM120: importing works (cheap, side-effect free — this
  module loads in every process including non-GPU API servers) but any kernel
  call raises a precise RuntimeError.

All other DeepGEMM symbols (fp8_gemm_nt, grouped variants, einsum, layout
helpers, ...) re-export from the vendor when present; when absent they are a
loud proxy, never a silent stub.

Why SM120 needs this: vLLM 0.30's ``support_deep_gemm()`` admits SM120, but
the vendored JIT compiles SM90/SM100 tcgen05/TMA kernels that do not build for
cc 12.0, so the DeepSeek-style sparse attention indexer dies on first use.
"""

import importlib
import importlib.util
import os
import sys
import warnings

__version__ = "2.1.1+suffix.sm120.shim"
__suffix_shim__ = True

# Capabilities whose Triton (not tcgen05/TMA) fallback path is engaged.
# SM120 = Blackwell consumer/workstation (RTX PRO 6000 / RTX 50 / B40).
_SM120 = (12, 0)


def _gate_enabled() -> bool:
    return os.environ.get("SUFFIX_SM120", "").strip() == "1"


def _is_sm120() -> bool:
    """Device check without ever touching CUDA in a non-GPU process."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() == _SM120
    except Exception:
        return False


def _load_vendor():
    """Return the vendored ``vllm.third_party.deep_gemm`` module or None.

    Imports ``vllm`` lazily (it is already in sys.modules whenever this runs
    inside a vLLM process; a bare ``import vllm.third_party.deep_gemm`` from a
    foreign process would drag the whole engine in, so we do the safe
    find_spec dance instead).
    """
    try:
        if importlib.util.find_spec("vllm.third_party.deep_gemm") is None:
            return None
        return importlib.import_module("vllm.third_party.deep_gemm")
    except Exception:
        # In a foreign process ``find_spec`` triggers the parent `vllm`
        # package import; any failure here just means "no vendor for us".
        return None


_vendor = _load_vendor()
_fallback = None

# Engage the Triton MQA fallback when the gate is set on SM120 (the operator
# opted in), or whenever there is no vendor to delegate to (the alternative
# is a hard crash in the indexer anyway — the shim is strictly an upgrade).
_use_fallback = _gate_enabled() and _is_sm120()
if _vendor is None:
    _use_fallback = True

if _use_fallback:
    try:
        # Triton/torch are only touched on GPUs that actually engage the
        # fallback path; never at plain-import time on API-server processes.
        from . import sm120_fallback as _fallback  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised on CUDA hosts only
        if _gate_enabled() and _is_sm120():
            # Fail closed on an enabled SM120 pool: silently delegating to a
            # vendor that cannot JIT for cc 12.0 would surface as a confusing
            # crash deep in the indexer.
            raise RuntimeError(
                "SUFFIX_SM120=1 on an SM120 GPU but the Triton MQA fallback "
                f"failed to import: {exc}"
            ) from exc
        _use_fallback = False


def _fallback_symbol(name):
    """Fallback-backed symbol, or a loud proxy to the vendor/missing path."""
    if _use_fallback and hasattr(_fallback, name):
        return getattr(_fallback, name)
    if _vendor is not None:
        return getattr(_vendor, name)

    def _missing(*args, **kwargs):
        raise RuntimeError(
            f"deep_gemm.{name} is unavailable: the vendored "
            "vllm.third_party.deep_gemm did not import and this environment "
            "has no SM120 Triton fallback for it (gate SUFFIX_SM120=1 "
            "engages the fallback on SM120 for the MQA-logits kernels only)."
        )

    _missing.__name__ = name
    return _missing


# The MQA-logits trio: SM120-gated overrides per the shim contract.
fp8_fp4_mqa_logits = _fallback_symbol("fp8_fp4_mqa_logits")
fp8_fp4_paged_mqa_logits = _fallback_symbol("fp8_fp4_paged_mqa_logits")
get_paged_mqa_logits_metadata = _fallback_symbol(
    "get_paged_mqa_logits_metadata")


class _MissingVendor:
    """Proxy for a DeepGEMM symbol with no vendor: resolves nothing, raises
    a precise RuntimeError on CALL (attribute-preserving, import-cheap)."""

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"deep_gemm.{self._name} is unavailable: the vendored "
            "vllm.third_party.deep_gemm did not import in this process."
        )

    def __getattr__(self, item):
        raise RuntimeError(
            f"deep_gemm.{self._name}.{item} is unavailable: the vendored "
            "vllm.third_party.deep_gemm did not import in this process."
        )

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<missing deep_gemm.{self._name}>"


if _vendor is not None:
    # Re-export EVERYTHING the vendor exposes (vLLM's _lazy_init probes a
    # long getattr() list: fp8_gemm_nt, m_grouped_*, fp8_einsum,
    # tf32_hc_prenorm_gemm, mega_mhc, get_mn_major_tma_aligned_tensor,
    # get_mk_alignment_for_contiguous_layout,
    # get_theoretical_mk_alignment_for_contiguous_layout,
    # transform_sf_into_required_layout, pack_ue8m0_to_int, cublaslt_gemm_nt,
    # set_pdl, get/set_num_sms, ...). Anything not shadowed above passes
    # through verbatim.
    for _name in dir(_vendor):
        if _name.startswith("__"):
            continue
        if _name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits",
                     "get_paged_mqa_logits_metadata"):
            continue
        globals().setdefault(_name, getattr(_vendor, _name))
    del _name
else:
    # No vendor: pre-create loud proxies for every symbol vLLM's
    # _lazy_init()/call sites probe, so `getattr(dg, name, None)` in
    # third-party code still sees a (callable, but raising) object rather
    # than silently taking a None branch that crashes later.
    _PROBED = (
        "cublaslt_gemm_nt", "fp8_gemm_nt", "fp8_einsum",
        "m_grouped_fp8_gemm_nt_contiguous", "m_grouped_fp8_gemm_nt_masked",
        "fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_fp4_gemm_nt_contiguous",
        "tf32_hc_prenorm_gemm", "mega_mhc", "get_mn_major_tma_aligned_tensor",
        "get_mk_alignment_for_contiguous_layout",
        "get_theoretical_mk_alignment_for_contiguous_layout",
        "transform_sf_into_required_layout", "pack_ue8m0_to_int",
        "get_mn_major_tma_aligned_packed_ue8m0_tensor",
        "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
        "set_pdl", "get_num_sms", "set_num_sms",
        "set_mk_alignment_for_contiguous_layout",
    )
    for _name in _PROBED:
        globals().setdefault(_name, _MissingVendor(_name))
    del _name


def _announce():
    """One loud line, only inside a vLLM process (never for unrelated
    importers, never in workers that merely have vllm importable)."""
    if "vllm" not in sys.modules:
        return
    if _use_fallback and _vendor is not None:
        msg = ("suffix deep_gemm shim ACTIVE: Triton SM120 MQA-logits "
               "fallback overrides the vendored DeepGEMM kernels")
    elif _use_fallback:
        msg = ("suffix deep_gemm shim ACTIVE: Triton SM120 MQA-logits "
               "fallback (no vendored DeepGEMM importable)")
    elif _vendor is not None:
        msg = ("suffix deep_gemm shim loaded; delegating verbatim to the "
               "vendored DeepGEMM (not SM120 or SUFFIX_SM120 unset)")
    else:
        msg = ("suffix deep_gemm shim loaded but INERT: no vendored "
               "DeepGEMM importable and this is not an SM120 GPU with "
               "SUFFIX_SM120=1 — kernel calls will raise")
    warnings.warn(msg, stacklevel=2)


_announce()
del _announce
