# SPDX-License-Identifier: Apache-2.0
"""SM120 NVFP4 DS-MLA KV cache (``--kv-cache-dtype nvfp4_ds_mla``) for vLLM
0.30.0 sparse MLA (GLM 5.3 / DeepSeek-V3.2 geometry) on our cuda-oxide
kernels (kernels-oxide/nvfp4_ds_mla, host ops src/nvfp4_ds_mla_oxide.rs).

Gate: ``SUFFIX_SM120_NVP4DSMLA=1`` AND compute capability family 12. Set but
not SM120: inert with one log line. Enabled on SM120: fail closed (every
inconsistency raises; a pool that asked for nvfp4_ds_mla never silently
serves another config).

Row contract (352 B/token, dossier dsmla-kernel-contracts-glm53.md §2):
[0,256) e2m1 NoPE (low nibble = even dim), [256,320) unscaled e4m3 RoPE,
[320,352) 32 e4m3 SFs at byte 8*(s&3)+(s>>2); sf = e4m3(max(amax16/6,
2^-9)). MLAAttentionSpec already sizes 352 B rows; nothing to patch there.

What stock 0.30.0 does and what we rewrite (in memory, exact-text anchors,
count-verified before any replacement, fixtures sm120/tests/fixtures/
vllm_0.30.0/ pin the sources):
  flashinfer_mla_sparse.py      SM120 backend rejects nvfp4_ds_mla (support
                                list + supports_combination) -> accept it.
  flashinfer_mla_sparse_sm120.py impl __init__ rejects it -> accept, and
                                driver-load our cubin at init (never inside
                                graph capture); _run_mqa_kernel -> our decode
                                op (flashinfer rejects 352 B rows);
                                do_kv_cache_update -> our writer (the C++
                                concat_and_cache_mla nvfp4 path is SM100-only,
                                build- and runtime-gated).
  index_group.py                HiSparse register_layer sizes every non-fp8
                                row as head_size (576) -> 352 for
                                nvfp4_ds_mla (the HiSparse data plane is
                                row-bytes generic; bind_source_cache
                                re-checks the width, fail closed).
The deepseek_v32 fused Triton writer (fused_norm_rope, non-HiSparse path)
already emits the 352 B rows on any arch; the oracle checks our reader
against it too.

Composition with hisparse_mtp_patch: it rewrites sparse_mla_attention.py
only; the files are disjoint and both hooks are one-shot front-inserted
finders, so they compose in either arming order.
"""

import os
import sys
from pathlib import Path

PATCH_NAME = "sm120-nvfp4-ds-mla"
PATCH_REVISION = "2026-09-26.3"
PINNED_VLLM = "0.30.0"
GATE_ENV = "SUFFIX_SM120_NVP4DSMLA"
MARKER_ATTR = "__suffix_nvfp4_ds_mla_revision__"
HELPER_TAG = "suffix sm120 nvfp4-ds-mla patch"
FAMILY = "nvfp4_ds_mla"  # oxide cubin / kernel family
ROW_BYTES = 352

BACKEND_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse"
IMPL_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"
INDEX_GROUP_MODULE = "vllm.v1.attention.backends.mla.index_group"


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


# ---------------------------------------------------------------------------
# Anchors: (name, exact old text, new text, expected count in the pin).
# ---------------------------------------------------------------------------
_TAG = f"        # {HELPER_TAG}\n"

BACKEND_EDITS = [
    (
        "supported_kv_cache_dtypes",
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        '    ]\n'
        '\n'
        '    @staticmethod\n'
        '    def get_name() -> str:\n'
        '        return "FLASHINFER_MLA_SPARSE_SM120"\n',
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        f'        "nvfp4_ds_mla",  # {HELPER_TAG}\n'
        '    ]\n'
        '\n'
        '    @staticmethod\n'
        '    def get_name() -> str:\n'
        '        return "FLASHINFER_MLA_SPARSE_SM120"\n',
        1,
    ),
    (
        "supports_combination",
        '            "fp8_e4m3",\n'
        '            "fp8_ds_mla",\n'
        '        ):\n'
        '            return "kv_cache_dtype not supported"\n',
        '            "fp8_e4m3",\n'
        '            "fp8_ds_mla",\n'
        f'            "nvfp4_ds_mla",  # {HELPER_TAG}\n'
        '        ):\n'
        '            return "kv_cache_dtype not supported"\n',
        1,
    ),
]

IMPL_EDITS = [
    (
        "impl_kv_cache_dtype_gate",
        '        if kv_cache_dtype != "fp8_ds_mla":\n',
        _TAG
        + '        self._use_nvfp4_ds_mla = kv_cache_dtype == "nvfp4_ds_mla"\n'
        '        if kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla"):\n',
        1,
    ),
    (
        "impl_init_load_kernels",
        '        self._workspace_buffer: torch.Tensor | None = None\n',
        '        self._workspace_buffer: torch.Tensor | None = None\n'
        + _TAG
        + '        if self._use_nvfp4_ds_mla:\n'
        '            _suffix_nvfp4_ds_mla_init(self)\n',
        1,
    ),
    (
        "impl_writer_override",
        '    def forward_mqa(\n',
        '    def do_kv_cache_update(\n'
        '        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype,\n'
        '        k_scale,\n'
        '    ) -> None:\n'
        + _TAG
        + '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        '            _suffix_nvfp4_ds_mla_write(kv_c_normed, k_pe, kv_cache, slot_mapping)\n'
        '            return\n'
        '        super().do_kv_cache_update(\n'
        '            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n'
        '        )\n'
        '\n'
        '    def forward_mqa(\n',
        1,
    ),
    (
        "run_mqa_kernel_decode",
        '            dtype=q.dtype,\n'
        '        )\n'
        '\n'
        '        if self._workspace_buffer is None:\n',
        '            dtype=q.dtype,\n'
        '        )\n'
        + _TAG
        + '        if self._use_nvfp4_ds_mla:\n'
        '            return _suffix_nvfp4_ds_mla_decode(\n'
        '                self, q, kv_cache, topk_indices_physical, output\n'
        '            )\n'
        '\n'
        '        if self._workspace_buffer is None:\n',
        1,
    ),
]

INDEX_GROUP_EDITS = [
    (
        "hisparse_row_width",
        '        if kv_cache_dtype == "fp8_ds_mla":\n'
        '            row_width = FP8_DS_MLA_ROW_BYTES\n',
        _TAG
        + '        # packed 352 B nvfp4_ds_mla rows; the HiSparse data plane is\n'
        '        # row-bytes generic and bind_source_cache re-checks the width.\n'
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        f'            row_width = {ROW_BYTES}\n'
        '            kv_dtype = torch.uint8\n'
        '        elif kv_cache_dtype == "fp8_ds_mla":\n'
        '            row_width = FP8_DS_MLA_ROW_BYTES\n',
        1,
    ),
]

# module -> (fixture file name, edits)
TARGETS = {
    BACKEND_MODULE: ("flashinfer_mla_sparse.py", BACKEND_EDITS),
    IMPL_MODULE: ("flashinfer_mla_sparse_sm120.py", IMPL_EDITS),
    INDEX_GROUP_MODULE: ("index_group.py", INDEX_GROUP_EDITS),
}


def patch_source(module_name: str, src: str) -> tuple[str, list[str]]:
    """Pure transform of one target file. Every anchor is count-verified
    BEFORE any replacement (all failures reported at once)."""
    fname, edits = TARGETS[module_name]
    bad = [f"{n}: expected {c}, found {src.count(old)}"
           for n, old, _new, c in edits if src.count(old) != c]
    if bad:
        raise PatchDriftError(
            f"{fname} does not match the pinned anchor text (expected vLLM "
            f"{PINNED_VLLM}): " + "; ".join(bad))
    out = src
    for n, old, new, _c in edits:
        if out.count(old) != 1:
            raise PatchDriftError(f"internal error: anchor {n} overlaps")
        out = out.replace(old, new)
    compile(out, f"<{PATCH_NAME}:{fname}>", "exec")  # syntax gate
    return out, [n for n, *_ in edits]


# ---------------------------------------------------------------------------
# Runtime helpers injected into the impl module (torch/_native imported
# lazily: this package is imported by sitecustomize before torch exists).
# ---------------------------------------------------------------------------
def _native():
    from suffix_hybrid import _native as n

    if not getattr(n, "HAS_NVFP4_DSMLA_CUDA", False):
        raise RuntimeError(
            f"{GATE_ENV}=1 but suffix_hybrid._native lacks the nvfp4_ds_mla "
            "CUDA ops (bundle built without the oxide-kernels feature)")
    return n


def _suffix_nvfp4_ds_mla_init(impl) -> None:
    """Impl __init__: geometry check + driver-load the cubin now, on this
    worker's device, outside any graph capture (fails closed at startup)."""
    import torch
    from suffix_hybrid import oxide_kernels

    geo = (impl.kv_lora_rank, impl.qk_rope_head_dim)
    if geo != (512, 64):
        raise NotImplementedError(
            f"[suffix {PATCH_NAME}] kernels need kv_lora_rank 512 / rope 64, "
            f"got {geo}")
    _native()
    dev = torch.cuda.current_device()
    oxide_kernels.ensure_loaded(FAMILY, dev)
    impl._nvfp4_num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    if dev not in _SELFTESTED:
        _suffix_nvfp4_ds_mla_selftest(impl, torch.device("cuda", dev))
        _SELFTESTED.add(dev)
    print(f"[suffix {PATCH_NAME}] NVFP4-DSMLA impl on cuda:{dev} "
          f"(heads/rank {impl.num_heads}, {impl._nvfp4_num_sms} SMs, rev "
          f"{PATCH_REVISION})", file=sys.stderr, flush=True)


_SELFTESTED = set()


def _suffix_nvfp4_ds_mla_selftest(impl, dev) -> None:
    """Boot guard, once per device at impl init (eager, never in a graph
    capture): writer + decode on 128 rows with max-SF rows and |q| ~ 1e5 —
    the cases a stale (pre-prescale) cubin turns into NaN — against an f64
    reference. Fails closed; costs a few ms."""
    import torch

    from .oracle import attention_ref, dequant_rows_ref, row_rel_l2

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # vLLM inits models under bf16
    try:
        g = torch.Generator().manual_seed(0)
        kv_c = torch.randn(128, 512, generator=g)
        kv_c[:32] = kv_c[:32].sign() * 5000.0  # SF 448, every nibble +-6
        k_pe = torch.randn(128, 64, generator=g)
        q = torch.randn(2, 8, 576, generator=g)
        q[0] *= 2.0 ** 15  # |q| > f16 max on every dim (exact rescale:
        # as well-conditioned as q[1]; single-dim outliers live in the oracle)
        topk = torch.stack([torch.randperm(128, generator=g) for _ in range(2)]).int()
        topk[1, 100:] = -1
        kv_c, k_pe, q, topk = (kv_c.bfloat16().to(dev), k_pe.bfloat16().to(dev),
                               q.bfloat16().to(dev), topk.to(dev))
        cache = torch.zeros(2, 64, ROW_BYTES, dtype=torch.uint8, device=dev)
        _suffix_nvfp4_ds_mla_write(kv_c, k_pe, cache, torch.arange(128, device=dev))
        out = torch.empty(2, 8, 512, dtype=torch.bfloat16, device=dev)
        _suffix_nvfp4_ds_mla_decode(impl, q, cache, topk, out)
        lat, rope = (x.double() for x in dequant_rows_ref(cache.view(-1, ROW_BYTES)))
        ref = attention_ref(q.double(), lat, rope, topk, float(impl.scale))
        err = row_rel_l2(out, ref, torch.ones(2, 8, dtype=torch.bool, device=dev))
    finally:
        torch.set_default_dtype(prev)
    if not (bool(torch.isfinite(out).all()) and err <= 2e-2):
        raise RuntimeError(
            f"[suffix {PATCH_NAME}] boot self-test FAILED on {dev}: max row "
            f"rel-L2 {err:.3e} vs f64 reference, finite "
            f"{bool(torch.isfinite(out).all())} (stale / pre-prescale cubin?)")


def _suffix_nvfp4_ds_mla_decode(impl, q, kv_cache, topk, output):
    """_run_mqa_kernel body: q [T, H, 576] bf16, topk [T, C] int32 physical
    slots (-1 masked), output [T, H, 512] bf16 (filled and returned)."""
    import torch

    n = _native()
    t, h = q.shape[0], q.shape[1]
    if t == 0:
        return output
    ns = n.nvfp4_ds_mla_plan(t, h, topk.shape[-1], impl._nvfp4_num_sms)["ns"]
    o_part = q.new_empty((t, h, ns, 512), dtype=torch.bfloat16)
    lse_part = q.new_empty((t, h, ns), dtype=torch.float32)
    n.nvfp4_ds_mla_decode_cuda(
        q, kv_cache.view(torch.uint8), topk.contiguous(), output, o_part,
        lse_part, float(impl.scale),
        torch.cuda.current_stream(q.device).cuda_stream)
    return output


def _suffix_nvfp4_ds_mla_write(kv_c, k_pe, kv_cache, slot_mapping) -> None:
    """do_kv_cache_update for nvfp4_ds_mla (HiSparse write/mirror targets
    included): kv_c [T, 512], k_pe [T, 64] or [T, 1, 64] bf16."""
    if kv_cache.numel() == 0:
        return
    import torch

    if k_pe.dim() == 3:
        k_pe = k_pe.squeeze(1)
    _native().nvfp4_ds_mla_quant_store_cuda(
        kv_c, k_pe, kv_cache.view(torch.uint8), slot_mapping.flatten(),
        torch.cuda.current_stream(kv_c.device).cuda_stream)


_HELPERS = {
    "_suffix_nvfp4_ds_mla_init": _suffix_nvfp4_ds_mla_init,
    "_suffix_nvfp4_ds_mla_decode": _suffix_nvfp4_ds_mla_decode,
    "_suffix_nvfp4_ds_mla_write": _suffix_nvfp4_ds_mla_write,
}


# ---------------------------------------------------------------------------
# apply(): gate -> SM120 -> version pin -> rewrite -> exec into the live
# module dict (fail closed: an enabled-but-broken pool never serves).
# ---------------------------------------------------------------------------
def exec_patched_source(module, new_src: str, src_path: Path) -> None:
    """exec under a linecache-registered name (source-introspectable)."""
    import linecache

    fname = f"{src_path}.{PATCH_NAME}-{PATCH_REVISION}.py"
    linecache.cache[fname] = (
        len(new_src), None, new_src.splitlines(keepends=True), fname)
    exec(compile(new_src, fname, "exec"), module.__dict__)


def apply(module) -> bool:
    """Patch one target module. True = active; False = cleanly inert (gate
    off / not SM120). Raises on any inconsistency while enabled on SM120."""
    if not gate_enabled():
        return False
    if not is_sm120():
        print(f"[suffix {PATCH_NAME}] present but inert: {GATE_ENV}=1 but "
              "this is not an SM120 (cc 12.x) GPU.", file=sys.stderr, flush=True)
        return False
    name = module.__name__
    if name not in TARGETS:
        raise PatchDriftError(f"{name} is not a {PATCH_NAME} target")
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
    new_src, applied = patch_source(name, src)
    if name == IMPL_MODULE:
        module.__dict__.update(_HELPERS)
    exec_patched_source(module, new_src, src_path)
    setattr(module, MARKER_ATTR, PATCH_REVISION)
    print(f"[suffix {PATCH_NAME}] ACTIVE on SM120: {name} rewritten "
          f"({', '.join(applied)}; rev {PATCH_REVISION}; vllm {ver}).",
          file=sys.stderr, flush=True)
    return True


def _hook_callback(module) -> None:
    try:
        applied = apply(module)
    except Exception as exc:
        raise SystemExit(
            f"[suffix {PATCH_NAME}] enabled but installation FAILED on "
            f"{module.__name__}: {exc}") from exc
    if not applied and is_sm120():
        raise SystemExit(f"[suffix {PATCH_NAME}] hook fired but apply() declined "
                         "on SM120 with the gate on (armed-but-inert guard).")


def install_post_import_hook() -> bool:
    """sitecustomize entry (stdlib only): arm one front-inserted one-shot
    finder per target (nvfp4_kv_patch._PostImportFinder, crash-1 lesson).
    A target imported before arming may already be bound by its importers
    (classes, register_layer) — refuse instead of patching half the world."""
    if not gate_enabled():
        return False
    from nvfp4_kv_patch import _PostImportFinder

    early = sorted(t for t in TARGETS if t in sys.modules)
    if early:
        raise SystemExit(f"[suffix {PATCH_NAME}] {early} imported before the "
                         "hook armed; refusing to patch a live module graph.")
    for target in TARGETS:
        if not any(isinstance(f, _PostImportFinder) and f.target == target
                   and f.armed for f in sys.meta_path):
            sys.meta_path.insert(0, _PostImportFinder(target, _hook_callback))
    print(f"[suffix {PATCH_NAME}] armed at sys.meta_path[0]: will patch "
          f"{len(TARGETS)} sparse-MLA modules on first import.",
          file=sys.stderr, flush=True)
    return True
