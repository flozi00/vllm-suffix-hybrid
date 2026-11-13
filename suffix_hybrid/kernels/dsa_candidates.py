# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 two-level candidate selection, fused into the indexer logits
(gate ``SUFFIX_SM120_DSA_INDEXER_CAND=1``, on top of SUFFIX_SM120_DSA_INDEXER).

vLLM nightly's ``sparse_attn_indexer`` decode branch on a consumer layer
(candidate_blocks given, candidate_write False) runs: dense paged logits
[rows, max_model_len] over every row's whole context -> ``_apply_candidate_mask``
(a flags pass + a FULL-WIDTH -inf pass) -> top-k over the whole context. Fused:
the deep_gemm shim's paged-logits call (kernels/dsa_indexer.wrap_paged) finds
its caller's frame, scores only the candidate positions into compressed
[rows, C * B] logits (helion_kernels/dsa_indexer.dsa_cand_logits) and returns
them; the patched ``_apply_candidate_mask`` skips exactly that tensor and the
patched ``get_indexer_topk`` runs vLLM's own top-k on it with per-row lens =
candidate slots, then maps slots back to positions. Same selected set (vLLM
leaves every non-candidate -inf), but past 2048 x 8 = 16k tokens the bytes
read, the mask pass and the top-k all stop growing with the context.

Candidate SOURCE layer (``SUFFIX_SM120_DSA_INDEXER_CANDSEL=1``, independent of
the gate above): vLLM's ``select_candidate_blocks`` scores EVERY block of the
max_model_len-wide logits (FULL_WIDTH on SM120: a full-width read, whatever the
context) and ranks them with ``torch.topk(k=2048)`` over nblocks = 65536 per
row. The patched twin scores the valid blocks only (vLLM's own
``_block_scores_kernel`` with FULL_WIDTH off) and ranks them with vLLM's
persistent top-k (k = 2048 is one of its sizes), valid blocks first and -1
filled like the original. Same set up to ties at the k-th score.

Fail closed: arming refuses to start unless the decode branch matches DECODE_TAIL
verbatim (comment/whitespace-blind) — nothing but the mask and the top-k may
touch those logits; per call it fuses only under vLLM's own masking conditions
(consumer layer, flattened rows, fp32 logits, no DCP, torch.topk-ordered
candidates) and only where a Helion config is shipped (dense otherwise); fused
logits reaching any other top-k raise.
"""
from __future__ import annotations

import inspect
import os
import re
import sys

GATE = "SUFFIX_SM120_DSA_INDEXER_CAND"
SEL_GATE = "SUFFIX_SM120_DSA_INDEXER_CANDSEL"
PERSISTENT_K = (512, 1024, 2048)   # the k persistent_topk serves (indexer_topk.py)
# vLLM 39983a5e7 candidate_blocks._block_scores_kernel parameters (anchor)
SCORES_PARAMS = ["logits", "starts", "ends", "scores", "valid_blocks", "stride_row", "stride_col",
                 "stride_start", "stride_end", "stride_scores", "width", "nblocks", "BLOCK_SIZE",
                 "HAS_STARTS", "ROW_REPEAT", "TILE", "FULL_WIDTH"]
TARGET = "vllm.model_executor.layers.sparse_attn_indexer"
# vLLM 39983a5e7 (sparse_attn_indexer, decode branch), comments + whitespace stripped
DECODE_TAIL = (
    "logits=fp8_fp4_paged_mqa_logits((padded_q_quant_cast,padded_q_scale),kv_cache,decode_weights,"
    "seq_lens,decode_metadata.block_table,decode_metadata.schedule_metadata,max_model_len=max_model_len,"
    "indices=decode_metadata.indices,)iftopk_backend!=\"deep_select\"ordcp_world_size!=1:logits="
    "as_fp32_logits(logits,seq_lens.view(-1))num_rows=logits.shape[0]ifcandidate_blocksisnotNone:vis="
    "seq_lens.reshape(-1)row_repeat=next_nifvis.numel()!=num_rowselse1vis=vis[:num_rows]"
    "decode_candidates=candidate_blocks[:num_rows]ifcandidate_write:_select_candidate_blocks(logits,"
    "None,vis,decode_candidates.shape[1],candidate_block_size,decode_candidates,row_repeat,)else:"
    "_apply_candidate_mask(logits,None,vis,decode_candidates,candidate_block_size,row_repeat,)"
    "topk_indices=topk_indices_buffer[:num_padded_tokens,:topk_tokens]get_indexer_topk(topk_backend)("
    "logits,seq_lens,next_n,topk_indices,topk_tokens,attn_metadata_narrowed.max_seq_len,)if"
    "decode_metadata.global_seq_lensisnotNone:_merge_dcp_topk_global(logits,topk_indices,topk_tokens,"
    "dcp_rank,dcp_world_size,cp_kv_cache_interleave_size,)"
)
_TAIL_END = "_merge_dcp_topk_global("
_state = {"code": None, "pending": None, "sorted": None}


def gate_on() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def sel_on() -> bool:
    return os.environ.get(SEL_GATE, "").strip() == "1"


_REMAP: dict = {}


def _triton_remap(bsz):
    """cand_remap as ONE Triton launch (hdi.cand_remap is ~8 torch ops). One
    stable callable per block size (a constexpr: its own binary), launched
    through dsa_capture.launch so an unwarmed class cannot JIT inside a graph
    capture."""
    if bsz in _REMAP:
        return _REMAP[bsz]
    import triton
    import triton.language as tl

    @triton.jit
    def _remap_kernel(idx, cand, cl, stride_idx, stride_cand, k, BSZ: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < k
        slot = tl.load(idx + row * stride_idx + cols, mask=ok, other=-1)
        sc = tl.maximum(slot, 0)
        blk = tl.load(cand + row * stride_cand + sc // BSZ, mask=ok & (slot >= 0), other=-1).to(tl.int64)
        pos = blk * BSZ + sc % BSZ
        keep = (slot >= 0) & (blk >= 0) & (pos < tl.load(cl + row))
        tl.store(idx + row * stride_idx + cols, tl.where(keep, pos, -1).to(tl.int32), mask=ok)

    class Remap:
        __name__ = f"cand_remap[B={bsz}]"

        def __call__(self, idx, cand, cl, stride_idx, stride_cand, k):
            _remap_kernel[(idx.shape[0], triton.cdiv(k, 512))](
                idx, cand, cl, stride_idx, stride_cand, k, BSZ=bsz, BLOCK=512)
    _REMAP[bsz] = Remap()
    return _REMAP[bsz]


def cand_remap(topk_indices, cand, cl, bsz):
    """Top-k slot ids -> positions, in place: one Triton launch on CUDA
    (int32 topk / cand / cl), the torch twin (helion_kernels) elsewhere."""
    if topk_indices.is_cuda:
        from suffix_hybrid.kernels import dsa_capture as cap
        rows, k = topk_indices.shape
        if rows:
            cap.launch(_triton_remap(bsz), topk_indices, cand, cl, topk_indices.stride(0), cand.stride(0), k)
        return topk_indices
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    return hdi.cand_remap(topk_indices, cand, cl, bsz)


def warm_remap(dev, bsz, k=512, ncand=2048, width=None):
    """Compile the remap for the row-count classes (1 | other | % 16) before any
    graph capture. `k` / `ncand` / `width` mirror the serving tensors' classes."""
    import torch
    width = width or ncand
    for rows in (1, 2, 16):
        idx = torch.full((rows, k), -1, dtype=torch.int32, device=dev)
        cand = torch.zeros((rows, ncand), dtype=torch.int32, device=dev)
        cl = torch.ones(rows, dtype=torch.int32, device=dev)
        cand_remap(idx, cand, cl, bsz)


def _norm(src: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"#[^\n]*", "", src))


def tail_matches(fn_source: str) -> bool:
    """The decode branch of sparse_attn_indexer is exactly DECODE_TAIL."""
    n = _norm(fn_source)
    a = n.find(DECODE_TAIL[:40])
    return a >= 0 and n[a:a + len(DECODE_TAIL)] == DECODE_TAIL


def cands_sorted() -> bool:
    """select_candidate_blocks leaves the valid blocks first (torch.topk,
    sorted) — the fused body's cand_lens prefix relies on it. DeepSelect
    returns them unordered: fusion stays off there."""
    try:
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
        return not cb.has_deep_select()
    except Exception:  # noqa: BLE001 - no such module: nothing to fuse with
        return False


def select_blocks_fn(cb, get_topk, stock):
    """The patched ``select_candidate_blocks`` (module doc). `cb` = vLLM's
    candidate_blocks module, `get_topk` = its indexer_topk dispatcher, `stock` =
    the original function (anything outside the fast path runs it)."""
    import torch

    def select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat=1):
        rows, width = logits.shape
        nblocks = -(-width // block_size)
        if not rows or topk_blocks not in PERSISTENT_K or nblocks < topk_blocks \
                or logits.dtype != torch.float32 or out.dtype != torch.int32 or out.stride(1) != 1 \
                or tuple(out.shape) != (rows, topk_blocks):
            return stock(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat)
        scores = logits.new_empty((rows, nblocks))
        valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
        cb._block_scores_kernel[(rows, 32)](
            logits, row_ks, row_ke, scores, valid_blocks, *logits.stride(),
            row_ks.stride(0) if row_ks is not None else 0, row_ke.stride(0), scores.stride(0),
            width, nblocks, block_size, row_ks is not None, row_repeat, TILE=128, FULL_WIDTH=False)
        # the tail of `scores` past each row's valid blocks was never written: the
        # persistent top-k reads only [0, valid_blocks)
        get_topk("persistent")(scores, valid_blocks, 1, out, topk_blocks, nblocks)
    return select_candidate_blocks


def install(mod=None):
    """Patch vLLM's sparse_attn_indexer module once (fail closed on drift)."""
    if _state["code"] is not None:
        return
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_indexer as di
    mod = mod or sys.modules.get(TARGET)
    fn = getattr(mod, "sparse_attn_indexer", None)
    gate = GATE if gate_on() else SEL_GATE
    if fn is None or not tail_matches(inspect.getsource(inspect.unwrap(fn))):
        raise RuntimeError(f"{di.MARKER} {gate}=1 but {TARGET}.sparse_attn_indexer's decode branch "
                           "drifted from the anchor (kernels/dsa_candidates.DECODE_TAIL): refusing "
                           "to patch candidate selection; unset the gate or re-anchor")
    if sel_on():
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
        fn_ = getattr(cb._block_scores_kernel, "fn", None)
        got = list(inspect.signature(fn_).parameters) if fn_ is not None else None
        if got != SCORES_PARAMS:
            raise RuntimeError(f"{di.MARKER} {SEL_GATE}=1 but candidate_blocks._block_scores_kernel's "
                               f"parameters {got} drifted from the anchor {SCORES_PARAMS}: refusing")
        mod._select_candidate_blocks = select_blocks_fn(cb, mod.get_indexer_topk, mod._select_candidate_blocks)
        di._log("candidate-block selection ARMED (valid blocks only + persistent top-k)")
    if gate_on():
        _arm_fusion(mod, torch, hdi, di)
    _state["code"] = inspect.unwrap(fn).__code__


def _arm_fusion(mod, torch, hdi, di):
    _state["sorted"] = cands_sorted()
    mask, get_topk = mod._apply_candidate_mask, mod.get_indexer_topk

    def _apply_candidate_mask(logits, *args, **kwargs):
        p = _state["pending"]
        if p is not None and logits is p[0]:
            return None  # fused logits hold candidates only
        return mask(logits, *args, **kwargs)

    def _get_indexer_topk(backend):
        topk = get_topk(backend)

        def run(logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len):
            p = _state["pending"]
            if p is None or logits is not p[0]:
                if p is not None:
                    _state["pending"] = None
                    raise RuntimeError(f"{di.MARKER} fused candidate logits were never ranked")
                return topk(logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len)
            _state["pending"] = None
            out, cand, cl, bsz = p
            lens = hdi.cand_lens(cl, cand.shape[1], bsz).to(torch.int32).reshape(-1, 1)
            bound = min(out.shape[1], -(-int(max_seq_len) // bsz) * bsz)
            topk(out, lens, 1, topk_indices, topk_tokens, bound)
            cand_remap(topk_indices, cand, cl, bsz)
        return run

    mod._apply_candidate_mask = _apply_candidate_mask
    mod.get_indexer_topk = _get_indexer_topk
    di._log(f"candidate fusion ARMED ({TARGET} patched; "
            f"{'torch.topk' if _state['sorted'] else 'DeepSelect/none: fusion stays off'} order)")


def _caller_locals(code, depth=6):
    f = sys._getframe(2)
    for _ in range(depth):
        if f is None:
            return None
        if f.f_code is code:
            return f.f_locals
        f = f.f_back
    return None


def maybe_fuse(qv, kv_cache, weights, context_lens, block_tables):
    """Compressed candidate logits for a consumer layer's decode call, or None
    (dense path). Called from the shim's paged-logits entry. Decided from
    static facts only (gate, layer type, shapes, a shipped config): the call
    sits inside a decode graph captured once, so nothing here may depend on the
    step's context. Fused is never slower in a graph: the dense path rewrites
    the whole max_model_len-wide row (a rows x 2 MB mask pass) at every context."""
    if not (gate_on() or sel_on()):
        return None
    if _state["code"] is None:
        from suffix_hybrid.kernels import dsa_capture as cap
        if cap.capturing():
            raise RuntimeError("[suffix dsa-indexer] candidate fusion / block selection is not armed and a CUDA-graph "
                               "capture is open: arming patches vLLM's module and binds state at first use; "
                               "run dsa_candidates.install() (or one eager decode step) before the capture")
        install()
    if not gate_on():
        return None
    if _state["pending"] is not None:
        _state["pending"] = None
        raise RuntimeError("[suffix dsa-indexer] fused candidate logits were never ranked")
    loc = _caller_locals(_state["code"])
    if loc is None or not _state["sorted"]:
        return None
    cand = loc.get("candidate_blocks")
    if cand is None or loc.get("candidate_write") or loc.get("topk_backend") == "deep_select" \
            or loc.get("dcp_world_size", 1) != 1:
        return None
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_indexer as di
    if qv.dim() == 4 and qv.shape[1] != 1:
        return None  # native next_n rows: vLLM shares lens per request (row_repeat)
    rows, heads = qv.shape[0], qv.shape[-2]
    cl = context_lens.reshape(-1)
    bsz, ncand = int(loc.get("candidate_block_size") or 0), cand.shape[1]
    if cl.numel() != rows or block_tables.shape[0] != rows or bsz <= 0 or cand.shape[0] < rows \
            or not hdi.has_config(hdi.CANDK, heads, rows, ncand=ncand, bsz=bsz):
        return None
    di._once("cand", f"candidate logits ACTIVE (helion fused, H={heads}, {ncand} blocks x {bsz})")
    cl = cl.to(torch.int32).contiguous()
    cand = cand[:rows].to(torch.int32).contiguous()
    out = torch.empty((rows, ncand * bsz), dtype=torch.float32, device=qv.device)
    hdi.cand_logits(qv.reshape(rows, heads, 128).contiguous(), weights, kv_cache,
                    block_tables.contiguous(), cl, cand, out)
    _state["pending"] = (out, cand, cl, bsz)
    return out
