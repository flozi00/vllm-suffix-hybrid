# SPDX-License-Identifier: Apache-2.0
"""Serving hook for the DeepSeek-V4.1 Helion sparse-MLA attention (K5).

Gate: ``SUFFIX_HELION_DSV41_ATTN`` = oxide (default: nothing installed,
FlashInfer serves) | helion (every call on our kernels, a missing config or an
unexpected call form is a startup/serving error) | auto (our kernel where a
lab-won config is shipped, FlashInfer elsewhere, counted and logged).
``SUFFIX_DSV41_ATTN_ROUTE=row`` forces the no-atomics body for decode too.

Seam: vLLM's DeepseekV4FlashInferSM120Attention (nightly 39983a5e7,
vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py) derives all metadata
(SWA windows, global compressed slots, lens) itself and then calls the module
global ``flashinfer_trtllm_batch_decode_sparse_mla_dsv4(...)`` from
``_forward_decode`` and ``_forward_prefill``. We replace exactly that module
global with a shim of the same keyword signature, so none of vLLM's metadata
code is copied (smallest drift surface) and no ``vllm._custom_ops`` is
touched. Fail closed: exact-text anchors in both methods are count-verified
before anything is replaced; the class must be the SM120 one; SM120 only.

K5 (startup assert / per-layer path log):
  * ``__init__`` is wrapped: every attention layer loads (compiles every Triton
    specialization + oracle-checks, once per device and geometry) its route BEFORE
    CUDA-graph capture, then runs the boot gate ``dsv41_attn_capture_replay``
    (suffix_hybrid/kernels/dsv41_attn_capture.py: capture on a side stream, replay with
    changing inputs / ctx lengths, bit-exact vs eager; ``SUFFIX_DSV41_ATTN_GATE=0``
    skips it); helion mode with a missing config or a failing gate fails the pod start.
  * ``forward_mqa`` is wrapped: the first call of every layer logs which path
    it took ("[suffix dsv41-attn] model.layers.5.attn: helion dsv41_attn_partial T=6 ...");
    a call form the shim does not implement (other kwargs, other layout,
    foreign scale) raises -- never a silent fallback. In auto mode every
    FlashInfer fallback is counted per (layer, T bucket) and logged once.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading

TAG = "[suffix dsv41-attn]"
KNOB = "SUFFIX_HELION_DSV41_ATTN"
TARGET = "vllm.models.deepseek_v41.nvidia.flashinfer_sparse"
CLASS = "DeepseekV4FlashInferSM120Attention"
SHIM_GLOBAL = "flashinfer_trtllm_batch_decode_sparse_mla_dsv4"

# exact-text anchors in the pinned source; every one must occur exactly once
ANCHORS = {
    "_forward_decode": (
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n",
        "sparse_indices=swa_indices,\n", "compressed_kv_cache=extra_cache,\n",
        "bmm1_scale=self.scale,\n", "sinks=self.attn_sink,\n", 'kv_layout="NHD",\n',
        "swa_topk_lens=swa_lens,\n", "extra_sparse_indices=extra_sparse_indices,\n",
        "extra_sparse_topk_lens=extra_sparse_lengths,\n",
    ),
    "_forward_prefill": (
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n",
        "sparse_indices=swa_indices_chunk,\n", "compressed_kv_cache=extra_kv_paged,\n",
        "bmm1_scale=self.scale,\n", "sinks=self.attn_sink,\n", 'kv_layout="NHD",\n',
        "swa_topk_lens=swa_lens_chunk,\n", "extra_sparse_indices=extra_sparse_indices_chunk,\n",
        "extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n",
    ),
    "forward_mqa": ("self._forward_sparse_impl(\n",),
}
# the keyword set the shim implements (anything else is an unexpected call form)
GATE_ENV = "SUFFIX_DSV41_ATTN_GATE"
KNOWN_KW = {"query", "swa_kv_cache", "workspace_buffer", "sparse_indices", "compressed_kv_cache",
            "out", "bmm1_scale", "sinks", "kv_layout", "swa_topk_lens", "extra_sparse_indices",
            "extra_sparse_topk_lens"}


class AnchorDrift(RuntimeError):
    """The installed vLLM source does not match the pinned anchor text."""


def check_source(name: str, text: str) -> None:
    for anchor in ANCHORS[name]:
        n = text.count(anchor)
        if n != 1:
            raise AnchorDrift(f"{TAG} {CLASS}.{name}: anchor {anchor.strip()!r} occurs {n}x, "
                              "expected 1 (vLLM drifted; port the anchors, do not guess)")


class _Stats:
    """Per-process counters (K5 evidence): helion calls and FlashInfer fallbacks."""

    def __init__(self):
        self.helion = {}
        self.fallback = {}
        self.seen = set()
        self.lock = threading.Lock()


STATS = _Stats()
_CUR = threading.local()
_ROUTES: dict = {}


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def route_for(heads: int, swa_w: int, topk: int, srec: int, crec: int, device, cmp_page: int = 32):
    """The loaded route for a layer geometry (cached across the 43 layers): compile +
    oracle (D.load), then the capture/replay boot gate."""
    from suffix_hybrid.helion_kernels import dsv41_attn as D

    key = (str(device), heads, swa_w, topk, srec, crec, cmp_page)
    if key not in _ROUTES:
        if D.capturing():
            raise RuntimeError(f"{TAG} geometry {key} reached a CUDA graph capture without a loaded route "
                               "(compiling inside a capture is illegal): the layer __init__ loads every "
                               "geometry a layer can see; this one was not predicted")
        route = D.load(heads, swa_w, topk, srec, crec, device, cmp_page)
        if route is not None and os.environ.get(GATE_ENV, "1").strip() != "0":
            from suffix_hybrid.kernels.dsv41_attn_capture import dsv41_attn_capture_replay
            dsv41_attn_capture_replay(route, device)
        _ROUTES[key] = route
    return _ROUTES[key]


def _max_seq_len(layer):
    """Host-known max sequence length of the step from the layer's compressed-cache
    metadata (a Python int: no device sync); None when unavailable. Only a PERFORMANCE
    hint (context class of the tuned config); correctness never depends on it."""
    try:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        src = md.get(layer.compressed_cache_prefix) if (isinstance(md, dict) and layer.compressed_cache_prefix) else None
        return int(src.max_seq_len) if src is not None else None
    except Exception:  # noqa: BLE001 -- a hint must never break serving
        return None


def make_shim(orig):
    """Replacement for the module global ``flashinfer_trtllm_batch_decode_sparse_mla_dsv4``."""
    from suffix_hybrid.helion_kernels import dsv41_attn as D
    from suffix_hybrid.helion_kernels import dsv41_attn_ref as R

    def shim(**kw):
        layer = getattr(_CUR, "layer", None)
        extra = set(kw) - KNOWN_KW
        if (extra or kw.get("kv_layout") != "NHD" or kw.get("swa_topk_lens") is None
                or abs(float(kw.get("bmm1_scale", 0.0)) - R.SCALE) > 1e-12 or layer is None):
            raise RuntimeError(f"{TAG} unexpected FlashInfer call form (extra kwargs {sorted(extra)}, "
                               f"kv_layout {kw.get('kv_layout')!r}, swa_topk_lens "
                               f"{'set' if kw.get('swa_topk_lens') is not None else 'None'}, "
                               f"bmm1_scale {kw.get('bmm1_scale')!r}, layer ctx "
                               f"{'ok' if layer is not None else 'MISSING'}): the SM120 DSV4.1 form "
                               "this hook implements changed; refusing instead of falling back")
        q, out = kw["query"], kw["out"]
        cmp_cache = kw.get("compressed_kv_cache")
        swa = kw["swa_kv_cache"]
        srec = D.pool_view(swa, "swa")[1]
        crec = D.pool_view(cmp_cache, "cmp")[1] if cmp_cache is not None else srec
        sidx = kw["sparse_indices"]
        eidx = kw.get("extra_sparse_indices")
        topk = eidx.shape[-1] if (cmp_cache is not None and eidx is not None) else 0
        cpage = D.pool_view(cmp_cache, "cmp")[2] if topk else 32
        route = route_for(q.shape[1], sidx.shape[-1], topk, srec, crec, q.device, cpage)
        done = None
        if route is not None:
            done = D.attend(route, q, swa, sidx, kw["swa_topk_lens"], cmp_cache, eidx,
                            kw.get("extra_sparse_topk_lens"), kw.get("sinks"), out,
                            D.ctx_class(getattr(_CUR, "max_seq_len", None)))
        t = q.shape[0]
        name = layer.prefix
        with STATS.lock:
            book = STATS.helion if done is not None else STATS.fallback
            b = D.hk.bucket(t, D.ROW_BUCKETS)
            book[(name, b)] = book.get((name, b), 0) + 1
            first = (name, b, done is not None) not in STATS.seen
            STATS.seen.add((name, b, done is not None))
        if first:
            ctx = D.ctx_class(getattr(_CUR, "max_seq_len", None))
            kind = D.pick(route, t, ctx)[0] if done is not None else "FlashInfer"
            _log(f"{name}: {'helion ' + kind if done is not None else 'FALLBACK to FlashInfer'} "
                 f"T={t} (bucket {b}, ctx {ctx}) swa {sidx.shape[-1]} topk {topk} records "
                 f"{R.REC_NAME[srec]}/{R.REC_NAME[crec] if topk else '-'}")
        if done is None:
            if srec != R.REC_V4 or (topk and crec != R.REC_V4):
                # FlashInfer 0.7.2 reads V4.1 records in its own page-footer layout, vLLM writes them
                # interleaved: a fallback would read garbage. Never fall back for these pools.
                raise RuntimeError(f"{TAG} {name}: no tuned Helion kernel for the "
                                   f"{R.REC_NAME[srec]}/{R.REC_NAME[crec] if topk else '-'} pools and FlashInfer "
                                   "cannot read vLLM's V4.1 interleaved records (use SUFFIX_HELION_DSV41_ATTN=helion)")
            return orig(**kw)
        return out

    return shim


def apply(module) -> None:
    """Post-import: verify anchors, wrap __init__/forward_mqa, swap the global."""
    import torch
    from vllm.platforms import current_platform

    from suffix_hybrid import helion_kernels as hk

    mode = hk.backend(KNOB)
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError(f"{TAG} {KNOB}={mode} needs an SM120 device (family 120, not 100)")
    cls = getattr(module, CLASS, None)
    orig_global = getattr(module, SHIM_GLOBAL, None)
    if cls is None or not callable(orig_global):
        raise AnchorDrift(f"{TAG} {TARGET} has no {CLASS} / {SHIM_GLOBAL}")
    for name in ANCHORS:
        check_source(name, inspect.getsource(getattr(cls, name)))
    if getattr(cls, "_suffix_dsv41_attn", False):
        return
    orig_init, orig_mqa = cls.__init__, cls.forward_mqa
    from suffix_hybrid.helion_kernels import dsv41_attn_ref as R

    def __init__(self, vllm_config, *args, **kwargs):
        orig_init(self, vllm_config, *args, **kwargs)
        # K5: load + oracle-check this layer's route before graph capture.
        width = module._required_sm120_sparse_topk(vllm_config, self.window_size)
        topk = 0 if self.compress_ratio == 0 else R.TOPK
        rec = R.REC_BY_BYTES[self.swa_bytes_per_token]
        crec = R.REC_BY_BYTES[self.compressed_bytes_per_token] if topk else rec
        hf = vllm_config.model_config.hf_config
        if (self.padded_heads != self.n_local_heads or self.head_dim != R.HD
                or (topk and getattr(hf, "index_topk", None) != R.TOPK)
                or hf.sliding_window != R.WINDOW):
            raise RuntimeError(f"{TAG} {self.prefix}: geometry unsupported (padded heads "
                               f"{self.padded_heads} vs local {self.n_local_heads}, head_dim "
                               f"{self.head_dim}, index_topk {getattr(hf, 'index_topk', None)}, "
                               f"window {hf.sliding_window})")
        dev = torch.device("cuda", torch.cuda.current_device())
        cpage = vllm_config.cache_config.block_size // self.compress_ratio if topk else 32
        for w in sorted({width, R.WINDOW}):   # decode (maybe non-causal) and prefill (always 128) lists
            route_for(self.padded_heads, w, topk, rec, crec, dev, cpage)

    def forward_mqa(self, q, kv, positions, output):
        _CUR.layer = self
        _CUR.max_seq_len = _max_seq_len(self)
        try:
            return orig_mqa(self, q, kv, positions, output)
        finally:
            _CUR.layer = None
            _CUR.max_seq_len = None

    cls.__init__, cls.forward_mqa = __init__, forward_mqa
    setattr(module, SHIM_GLOBAL, make_shim(orig_global))
    cls._suffix_dsv41_attn = True
    _log(f"ACTIVE ({KNOB}={mode}): {CLASS} decode/prefill attention -> Helion "
         f"{'(FlashInfer where no config is shipped)' if mode == 'auto' else '(no fallback)'}")


def _callback(module) -> None:
    try:
        apply(module)
    except Exception as exc:
        raise SystemExit(f"{TAG} enabled but FAILED on {module.__name__}: {exc!r}") from exc


def install_post_import_hook() -> bool:
    from suffix_hybrid import helion_kernels as hk

    if hk.backend(KNOB) == "oxide":
        return False
    if TARGET in sys.modules:
        raise SystemExit(f"{TAG} {TARGET} imported before the hook armed")
    from nvfp4_kv_patch import _PostImportFinder
    sys.meta_path.insert(0, _PostImportFinder(TARGET, _callback))
    return True
