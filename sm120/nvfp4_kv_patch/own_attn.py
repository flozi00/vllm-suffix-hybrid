# SPDX-License-Identifier: Apache-2.0
"""K2-NVFP4: our SM120 NVFP4-KV decode / spec-verify attention (cutile-rs
kernel, src/nvfp4_attn_gpu.rs) inside the patched FlashInfer backend.

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

Cubins: pods never JIT. CI (scripts/nvfp4_attn_prebuild.py) compiles every
variant with the offline tileiras into suffix_hybrid/nvfp4_attn_cubins/;
``builder_gate`` verifies + installs them. ``..._ALLOW_JIT=1`` (dev boxes
with tileiras only) skips that requirement.
"""

import json
import os
import sys

ENV = "SUFFIX_SM120_NVP4KV_OWN_ATTN"
ALLOW_JIT_ENV = "SUFFIX_SM120_NVP4KV_OWN_ATTN_ALLOW_JIT"
CUBIN_DIR_ENV = "SUFFIX_SM120_NVP4KV_OWN_ATTN_CUBINS"
KERNELS = ("nvfp4_attn_partial", "nvfp4_attn_merge")
LOG2E = 1.4426950408889634
MAX_Q_LEN = 16

_PLANS: dict = {}
_SMS: dict = {}
_META: dict = {}
_INSTALLED: set = set()


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


def cubin_dir() -> str:
    d = os.environ.get(CUBIN_DIR_ENV, "").strip()
    if d:
        return d
    import suffix_hybrid

    return os.path.join(os.path.dirname(suffix_hybrid.__file__),
                        "nvfp4_attn_cubins")


def install_cubins(nat, served, q_lens, gpu, directory=None) -> int:
    """Verify + install every manifest cubin for ``served`` = (d, hq, hkv,
    page, window_left) and q_len in ``q_lens``; each q_len must be covered.
    Returns the number of cubins installed."""
    import hashlib

    directory = directory or cubin_dir()
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"{ENV}=1 but no K2-NVFP4 cubin manifest at {path} "
                           f"(CI prebuild missing; dev boxes: {ALLOW_JIT_ENV}=1)")
    with open(path) as f:
        man = json.load(f)
    want = str(man["bytecode_version"])
    have = os.environ.get("CUTILE_BYTECODE_VERSION")
    if have and have != want:
        raise RuntimeError(f"CUTILE_BYTECODE_VERSION={have} != manifest {want}")
    os.environ["CUTILE_BYTECODE_VERSION"] = want
    d, hq, hkv, page, wl = served
    n, covered = 0, set()
    for e in man["entries"]:
        if (e["d"], e["hq"], e["hkv"], e["page"]) != (d, hq, hkv, page) or (
                e["q_len"] not in q_lens) or not _same_div(e["window_left"], wl):
            continue
        with open(os.path.join(directory, e["file"]), "rb") as f:
            cubin = f.read()
        if hashlib.sha256(cubin).hexdigest() != e["sha256"]:
            raise RuntimeError(f"K2-NVFP4 cubin sha256 mismatch: {e['file']}")
        key = (d, hq, hkv, page, e["q_len"], wl, e["ns"], e["kernel"])
        if key not in _INSTALLED:
            nat.nvfp4_attn_install_cubin(d, hq, hkv, page, e["q_len"], wl,
                                         e["ns"], e["kernel"], gpu,
                                         e["bc_sha256"], cubin)
            _INSTALLED.add(key)
        covered.add(e["q_len"])
        n += 1
    missing = sorted(set(q_lens) - covered)
    if missing:
        raise RuntimeError(
            f"K2-NVFP4 manifest has no cubins for d={d} hq={hq} hkv={hkv} "
            f"page={page} window_left={wl} q_len={missing}; add this served "
            "shape to scripts/nvfp4_attn_prebuild.py --served and rebuild")
    return n


def prepare(nat, served, q_lens) -> str:
    """Make every variant of ``served`` x ``q_lens`` launchable: install the
    CI cubins (serving), or allow JIT under ALLOW_JIT (dev boxes only)."""
    if os.environ.get(ALLOW_JIT_ENV, "").strip() == "1":
        nat.nvfp4_attn_allow_jit(True)
        return "JIT allowed (dev only)"
    import torch

    gpu = nat.nvfp4_attn_gpu_name(torch.cuda.current_device())
    n = install_cubins(nat, served, q_lens, gpu)
    return f"{n} prebuilt {gpu} cubins installed, JIT off"


def _same_div(a: int, b: int) -> bool:
    """window_left only reaches the kernel key via its pow2 divisibility."""
    def div(x):
        x = int(x)
        return 16 if x == 0 else min(x & -x, 16)
    return div(a) == div(b)


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
