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
CUDA graphs (V2 runner): H13 UNIFORM_BATCH. Uniform verify batches (q_len
2..1+k) replay FULL graphs through K2 (grid is seq_lens-independent, H22
drops the host seq_lens sync); uniform q_len-1 batches replay FULL graphs
through FlashInfer's stock fa2 cudagraph decode wrappers
(install_uniform_graph_lens adds the widths vLLM does not capture itself).
Batches holding a prefilling request are never uniform -> piecewise, FA2.

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
    # every register variant (k2_nvfp4_attn_w1/_w2/_w3); the host op picks
    # the one the launch plan names.
    names = [k["name"] for k in oxide_kernels.manifest()["kernels"]
             if k["name"].startswith(FAMILY + "_w")]
    if len(names) != 3:
        raise RuntimeError(f"oxide manifest lacks the K2 variants (found {names})")
    for name in names:
        oxide_kernels.ensure_loaded(name, dev)
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


def install_uniform_graph_lens() -> bool:
    """Target FULL graphs for EVERY uniform decode width 1..1+k, not just
    1+k. vLLM's V2 runner captures uniform-decode graphs only at
    decode_query_len = 1+k (cudagraph_utils._init_candidates), so a uniform
    batch of plain decodes (q_len 1: suffix misses, W0 fast path) or of
    shorter verifies runs PIECEWISE with eager attention in every layer.
    UNIFORM_BATCH (our H13 tier) already promises any uniform width is
    capturable: q_len 1 lands on FlashInfer's stock fa2 cudagraph decode
    wrappers, 2..1+k on K2 with q_len baked per graph. Same multimodal
    guard as 1+k: a batch with any prefilling request is never uniform.
    Only called when K2 graphs are enabled; V1 runner -> no-op.
    SUFFIX_SM120_NVP4KV_GRAPH_ALL_WIDTHS=0 keeps vLLM's 1+k-only set (A/B,
    or to hand the extra graph memory back to KV)."""
    if os.environ.get("SUFFIX_SM120_NVP4KV_GRAPH_ALL_WIDTHS", "1").strip() == "0":
        return False
    try:
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
    except ImportError:
        return False
    return wrap_init_candidates(cgu.ModelCudaGraphManager, cgu.CUDAGraphMode)


def wrap_init_candidates(cls, modes) -> bool:
    cur = cls.__dict__.get("_init_candidates")
    if getattr(cur, "_suffix_k2_lens", False):
        return True
    base = cls._init_candidates

    def _init_candidates(self):
        base(self)
        added = add_uniform_lens(self, base, modes)
        if added:
            print(f"[suffix sm120-nvfp4-kv] K2 graphs: FULL uniform-decode "
                  f"graphs also for q_len {added} (vLLM default: "
                  f"{self.decode_query_len} only)", file=sys.stderr,
                  flush=True)

    _init_candidates._suffix_k2_lens = True
    cls._init_candidates = _init_candidates
    return True


def add_uniform_lens(mgr, base, modes) -> list:
    """Run the stock candidate builder once per extra width and merge its
    FULL uniform descriptors (captures + dispatch candidates, FULL entries
    ahead of PIECEWISE ones as stock orders them)."""
    full = modes.FULL
    mode = mgr.cudagraph_mode
    spec = getattr(mgr.vllm_config, "speculative_config", None)
    dq = mgr.decode_query_len
    dynamic = spec is not None and getattr(
        spec, "uses_dynamic_speculative_decoding", lambda: False)()
    if (dq <= 1 or not mode or not mode.separate_routine()
            or mode.decode_mode() != full or mgr.varlen_decode or dynamic):
        return []
    descs, cands = mgr._capture_descs, mgr._candidates
    added = []
    try:
        for u in range(1, dq):
            mgr._capture_descs, mgr._candidates = {}, {}
            mgr.decode_query_len = u
            base(mgr)
            new = [d for d in mgr._capture_descs.get(full, [])
                   if d.uniform_token_count == u]
            dst = descs.setdefault(full, [])
            dst.extend(d for d in new if d not in dst)
            for key, lst in mgr._candidates.items():
                fu = [d for d in lst
                      if d.cg_mode == full and d.uniform_token_count == u]
                tgt = cands.setdefault(key, [])
                at = next((i for i, d in enumerate(tgt)
                           if d.cg_mode != full), len(tgt))
                tgt[at:at] = [d for d in fu if d not in tgt]
            if new:
                added.append(u)
    finally:
        mgr.decode_query_len = dq
        mgr._capture_descs, mgr._candidates = descs, cands
    descs.get(full, []).sort(key=lambda d: d.num_tokens, reverse=True)
    return added


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
