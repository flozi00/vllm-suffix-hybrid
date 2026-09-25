# SPDX-License-Identifier: Apache-2.0
"""K2-NVFP4: our SM120 NVFP4-KV decode / spec-verify attention (cuda-oxide
kernels kernels-oxide/k2_nvfp4_attn, host op src/nvfp4_attn_oxide.rs)
inside the patched FlashInfer backend.

Gate: ``SUFFIX_SM120_NVP4KV_OWN_ATTN=1`` on top of the fa2 NVFP4 route
(``SUFFIX_SM120_NVP4KV=1``). Default OFF. When set it is FAIL-CLOSED: a
missing feature build, a missing/mismatched prebuilt-cubin manifest, an
out-of-contract shape (head_dim, GQA group, spec width > 16, page size) or
logits soft-capping raises at backend init — the pool never silently keeps
the FA2 decode kernel while the operator believes ours is serving.

What changes (patch anchors H19/H20):
  * H19 builder: ``supports_spec_as_decode`` also when our kernel is armed, so
    uniform MTP/suffix verify rows (q_len = 1+k) become DECODE rows instead of
    riding the FA2 paged prefill kernel. Non-uniform rows, real prefills and
    image-bearing short extends (H17 resplit) stay on FA2 (incl. the
    mm-prefix custom mask) — multimodality untouched.
  * H20 builder decode branch: ``FIDecode(wrapper=DecodeWrapper(...))`` in
    place of FlashInfer's planned BatchDecodeWithPagedKVCacheWrapper, so
    forward()'s existing ``decode_wrapper.run(...)`` call lands on our op with
    zero forward-path edits.
CUDA graphs: unchanged tier (H13 UNIFORM_SINGLE_TOKEN_DECODE). q_len=1 decode
batches replay FULL graphs through our op (grid is seq_lens-independent);
spec-verify batches run piecewise as today — UNIFORM_BATCH needs the image
ranges in-kernel first (dossier sm120-nvfp4-attn-kernel.md §5).

Cubins: pods never JIT. CI (scripts/oxide_build.py) compiles the kernel
crate with cargo-oxide to PTX (ISA <= 9.0) and ptxas 13.0 to sm_120 SASS;
``prepare`` driver-loads the sha256-verified cubin and runs the toolchain
probe (suffix_hybrid.oxide_kernels) before serving.
"""

import json
import os
import sys

ENV = "SUFFIX_SM120_NVP4KV_OWN_ATTN"
LOG2E = 1.4426950408889634
MAX_Q_LEN = 16

_PLANS: dict = {}
_SMS: dict = {}
_META: dict = {}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == "1"


def native():
    """The feature-on native module, or a loud error (never a fallback)."""
    from suffix_hybrid import _native

    if not getattr(_native, "HAS_NVFP4_ATTN_CUDA", False):
        raise RuntimeError(
            f"{ENV}=1 but suffix_hybrid._native was built without the "
            "`nvfp4-attn-kernels` cargo feature (K2-NVFP4 CUDA op missing). "
            "Ship the feature bundle or unset the gate.")
    return _native


def plan(batch, q_len, hq, hkv, d, page_size, num_sms):
    key = (batch, q_len, hq, hkv, d, page_size, num_sms)
    p = _PLANS.get(key)
    if p is None:
        from suffix_hybrid import _native

        p = _PLANS[key] = _native.nvfp4_attn_plan(*key)
    return p


def num_sms(device) -> int:
    n = _SMS.get(device)
    if n is None:
        import torch

        n = _SMS[device] = torch.cuda.get_device_properties(
            device).multi_processor_count
    return n


def max_decode_q_len(vllm_config) -> int:
    """Widest uniform decode row the builder will route here (mirrors
    AttentionMetadataBuilder._init_reorder_batch_threshold)."""
    spec = getattr(vllm_config, "speculative_config", None)
    k = getattr(spec, "num_speculative_tokens", None) if spec else None
    if not k:
        return 1
    return 1 + (2 if getattr(spec, "parallel_drafting", False) else 1) * k


FAMILY = "k2_nvfp4_attn"


def prepare(nat, served, q_lens) -> str:
    """Make the K2 kernels launchable on this device: load the CI-built
    cuda-oxide cubin (sm_120 SASS from ptxas 13.0; sha256-verified, driver
    load = the self-check) and run the toolchain probe once. There is no JIT
    path at all on this track."""
    import torch

    from suffix_hybrid import oxide_kernels

    dev = torch.cuda.current_device()
    oxide_kernels.ensure_loaded(FAMILY, dev)
    if dev not in _PROBED:
        print(oxide_kernels.probe(dev), file=sys.stderr, flush=True)
        _PROBED.add(dev)
    return "cuda-oxide sm_120 cubin driver-loaded + probe PASS, no JIT"


_PROBED: set = set()


def uniform_decode_excludes_prefill(func=None) -> bool:
    """The V2 runner property UNIFORM_BATCH relies on for multimodality: a
    batch containing any still-prefilling request (e.g. a (1+k)-token prompt
    chunk inside an image range) is never classified uniform-decode, hence
    never replays a FULL (causal) decode graph. Checked on the installed
    source (vllm/v1/worker/utils.py:740-746, fed by
    gpu/model_runner.py:1249-1266 has_prefill = is_prefilling.any())."""
    import inspect

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").strip() != "1":
        return False
    try:
        if func is None:
            from vllm.v1.worker.utils import get_uniform_decode_token_count
            func = get_uniform_decode_token_count
        src = inspect.getsource(func)
    except Exception:
        return False
    return "if not has_prefill and" in src


def builder_gate(builder, window_left, logits_soft_cap) -> bool:
    """Patch H19: evaluated once per FlashInferMetadataBuilder (per KV group).
    False = stock fa2 decode; True = our kernel; raises when armed but not
    honourable."""
    if not enabled() or not getattr(builder, "use_fa2_nvfp4_kv", False):
        return False
    nat = native()
    if logits_soft_cap:
        raise ValueError(f"{ENV}=1: logits soft-capping ({logits_soft_cap}) "
                         "is not in the K2-NVFP4 kernel contract.")
    q_max = max_decode_q_len(builder.vllm_config)
    if q_max > MAX_Q_LEN:
        raise ValueError(
            f"{ENV}=1: spec width 1+k={q_max} exceeds the kernel's "
            f"q_len<={MAX_Q_LEN} contract; unset the gate or lower k.")
    d, hq, hkv, page = (builder.head_dim, builder.num_qo_heads,
                        builder.num_kv_heads, builder.page_size)
    # Contract check at the widest shape (ValueError names the violation).
    nat.nvfp4_attn_plan(1, q_max, hq, hkv, d, page, 1)
    wl = -1 if window_left is None else int(window_left)
    how = prepare(nat, (d, hq, hkv, page, wl), tuple(range(1, q_max + 1)))
    print(
        f"[suffix sm120-nvfp4-kv] OWN-ATTN ACTIVE: decode + uniform "
        f"spec-verify (q_len<={q_max}) -> K2-NVFP4 cutile kernel "
        f"(head_dim={d}, heads={hq}/{hkv}, page={page}, window_left={wl}; "
        f"{how}); prefill + mm-prefix stay on FlashInfer fa2.",
        file=sys.stderr, flush=True)
    return True


class DecodeWrapper:
    """Stand-in for BatchDecodeWithPagedKVCacheWrapper inside FIDecode.

    Carries the attributes forward() asserts on and implements the one
    ``run`` signature forward() uses on the fa2 nvfp4 route."""

    def __init__(self, block_table, seq_lens, q_len, num_qo_heads,
                 num_kv_heads, head_dim, page_size, window_left, sm_scale,
                 logits_soft_cap):
        if logits_soft_cap:
            raise ValueError(
                f"{ENV}=1: logits soft-capping ({logits_soft_cap}) is not in "
                "the K2-NVFP4 kernel contract.")
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap or 0.0
        self._sm_scale = sm_scale
        self.block_table = block_table
        self.seq_lens = seq_lens
        self.q_len = q_len
        self.shape = (num_qo_heads, num_kv_heads, head_dim, page_size)

    def run(self, q, kv_cache, *, q_scale=None, k_scale=None, v_scale=None,
            out=None, kv_cache_sf=None, sinks=None, lse=None,
            return_lse=False):
        if sinks is not None or return_lse or lse is not None:
            raise ValueError("K2-NVFP4: sinks / lse outputs not supported")
        if kv_cache_sf is None or out is None:
            raise ValueError("K2-NVFP4 needs kv_cache_sf and a preallocated out")
        k_data, v_data = kv_cache
        k_sf, v_sf = kv_cache_sf
        hq, hkv, d, page = self.shape
        scale = self._sm_scale * (1.0 if q_scale is None else q_scale) * (
            1.0 if k_scale is None else k_scale)
        return run(q, k_data, k_sf, v_data, v_sf, self.block_table,
                   self.seq_lens, out.view(q.shape[0], hq, d), self.q_len,
                   self._window_left, scale,
                   1.0 if v_scale is None else v_scale)


def _meta(device, bt_row_stride):
    """Persistent device word carrying the block-table row stride (keeps it
    out of the kernel key; stable address for CUDA-graph capture)."""
    key = (str(device), int(bt_row_stride))
    t = _META.get(key)
    if t is None:
        import torch

        t = torch.zeros(16, dtype=torch.int32, device=device)
        t[0] = int(bt_row_stride)
        _META[key] = t
    return t


def run(q, k_data, k_sf, v_data, v_sf, block_table, seq_lens, out, q_len,
        window_left, sm_scale, v_scale):
    """out[b*q_len + i] = attention of q row (b, i) over request b's paged
    NVFP4 KV (causal, q at the tail, optional window). Uniform q_len."""
    import torch

    tokens, hq, d = q.shape
    _, hkv, page, _ = k_data.shape
    if tokens == 0:
        return out
    sms = num_sms(q.device)
    p = plan(tokens // q_len, q_len, hq, hkv, d, page, sms)
    rows16 = -(-p["rows"] // 16) * 16
    o_part = torch.empty((rows16, p["ns"], p["m"], d), dtype=torch.bfloat16,
                         device=q.device)
    lse_part = torch.empty((rows16, p["ns"], p["m"]), dtype=torch.float32,
                           device=q.device)
    native().nvfp4_paged_attn_cuda(
        q, k_data, k_sf, v_data, v_sf, block_table,
        _meta(q.device, block_table.stride(0)), seq_lens, out, o_part,
        lse_part, q_len, window_left, sm_scale * LOG2E, v_scale, sms,
        torch.cuda.current_stream(q.device).cuda_stream)
    return out
