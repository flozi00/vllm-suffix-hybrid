# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer lane's contract with the vLLM nightly DeepSeek-V4.1 runs on
(39983a5e7, DeepGEMM bump #61149): pinned upstream copies in
fixtures/vllm_39983a5e7/, checked for exactly what the shim, the lane and the
candidate fusion assume. A failing test here = upstream drift: re-anchor the
lane before serving that nightly."""
import ast
import hashlib
import inspect
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_candidates as dc  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

NIGHTLY = Path(__file__).resolve().parent / "fixtures/vllm_39983a5e7"
SHA256 = {
    "candidate_blocks.py.txt": "604f588fcf96d546fdf230a7b225384a27952acedbcefbd7ea5c181f18e5c621",
    "deep_gemm.py.txt": "0c84d8960edc4e613b79678d3394a2ad6fd3c53b1530a83c2bdaf407dfe41e00",
    "indexer_k_store.py.txt": "298c3c5aa61902ab439a51dd074234064e4f67ff9e1e735374db351ca56c08f9",
    "indexer_backend.py.txt": "b2baac312b6bd213a79e9a555863f186022cd978338343b4be762f72b30058fc",
    "indexer_topk.py.txt": "4dc91f83043c659304e0f0808bd9082574f919361b0b12f7c87fef5e261469bb",
    "sparse_attn_indexer.py.txt": "ed9cfdd83544d15c088b052e287971610d0de96062493b6c757d16ea2248c6aa",
}


def src(name):
    return (NIGHTLY / f"{name}.py.txt").read_text()


def norm(text):
    return re.sub(r"\s+", "", re.sub(r"#[^\n]*", "", text))


def params(source, fn, cls=None):
    tree = ast.parse(source)
    scope = tree.body
    if cls:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    f = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == fn)
    return [a.arg for a in f.args.args + f.args.kwonlyargs]


def test_fixtures_are_pinned_upstream_copies():
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in NIGHTLY.iterdir()} == SHA256


def test_prefill_contract_is_compressed_and_the_shim_reads_it():
    dg = src("deep_gemm")
    assert params(dg, "fp8_fp4_mqa_logits") == ["q", "kv", "weights", "cu_seqlen_ks", "cu_seqlen_ke",
                                                 "max_seqlen_k"]
    assert "Compressedlogitsofshape[M,max_seqlen_k]" in norm(dg)
    # the impl gets max_seqlen_k as its 6th POSITIONAL argument: the shim's
    # wrap_mqa / Triton fallback read an int there as the compressed width
    assert "_fp8_fp4_mqa_logits_impl(q,kv,weights,cu_seqlen_ks,cu_seqlen_ke,max_seqlen_k,)" in norm(dg)
    sai = norm(src("sparse_attn_indexer"))
    assert ("fp8_fp4_mqa_logits((q_slice_cast,q_scale_slice),(k_quant_cast,k_scale_cast),chunk_weights,"
            "cu_seqlen_ks,cu_seqlen_ke,chunk.max_seqlen_k,)") in sai
    assert "row_ends=cu_seqlen_ke-cu_seqlen_ks" in sai   # the consumer reads columns [0, ke - ks)
    sig = list(inspect.signature(di.wrap_mqa(None)).parameters)
    assert sig[:6] == ["q", "kv", "weights", "cu_seqlen_ks", "cu_seqlen_ke", "max_seqlen_k"]


def test_paged_contract_matches_the_shim_signature():
    dg = src("deep_gemm")
    assert params(dg, "fp8_fp4_paged_mqa_logits") == [
        "q", "kv_cache", "weights", "context_lens", "block_tables", "schedule_metadata",
        "max_model_len", "indices"]
    assert ("_fp8_fp4_paged_mqa_logits_impl(q,kv_cache,weights,context_lens,block_tables,"
            "schedule_metadata,max_model_len,**kwargs,)") in norm(dg)
    ours = inspect.signature(di.wrap_paged(None))
    ours.bind(*range(7))                 # vLLM's positional form ...
    ours.bind(*range(7), indices=None)   # ... plus indices as a keyword
    # SM120: fp32 scales + fp32 weights (MX inputs are SM100 only), 64-entry pages
    assert "is_device_capability_family(100" in norm(dg).split("defmqa_logits_use_mx_scales")[1][:600]
    assert "return(64,)ifcurrent_platform.is_device_capability_family(120)else(32,64)" in norm(dg)


def test_indexer_k_cache_is_the_split_page_cache_views_reads():
    ks = norm(src("indexer_k_store"))
    assert "val_ptr=cache_block_ptr+kv_pos_in_block*TOKEN_STRIDE" in ks
    assert "scale_ptr=(cache_block_ptr+kv_cache_block_size*TOKEN_STRIDE+kv_pos_in_block*SCALE_DIM" in ks
    assert "cache_block_ptr=k_cache_ptr+kv_block_idx.to(tl.int64)*KV_BLOCK_STRIDE" in ks  # free stride
    assert "MX_SCALE=notuse_fp4_cacheandmqa_logits_use_mx_scales()" in ks                  # fp32 on SM120
    assert "scale_dim=4" in ks and "token_stride=head_dim" in ks


def test_candidate_selection_order_and_mask_contract():
    cb = src("candidate_blocks")
    n = norm(cb)
    # the non-DeepSelect path ranks with torch.topk (sorted) and writes -1 for
    # -inf scores: valid blocks first, which cand_lens relies on
    assert "top=scores.topk(min(topk_blocks,nblocks),dim=-1)" in n
    assert "tl.where(value!=-float(\"inf\"),index,-1)" in n
    assert "use_deep_select=has_deep_select()andnblocks>=topk_blocks" in n
    assert params(cb, "apply_candidate_mask") == ["logits", "row_ks", "row_ke", "candidate_blocks",
                                                  "block_size", "row_repeat"]
    assert "(cols>=start)&(cols<end)" in n   # decode: start 0, a block covers [B b, B b + B)


def test_topk_call_form_and_the_fused_tail():
    assert params(src("indexer_topk"), "forward", "SparseIndexerTopk") == [
        "self", "logits", "seq_lens", "next_n", "topk_indices", "topk_tokens", "max_seq_len"]
    sai = src("sparse_attn_indexer")
    assert dc.tail_matches(sai)
    # every name the fused path patches or reads off the caller's frame exists
    tree = ast.parse(sai)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer")
    args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert {"candidate_blocks", "candidate_block_size", "candidate_write", "topk_backend",
            "dcp_world_size"} <= args
    assert "from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (\n" \
           "    apply_candidate_mask as _apply_candidate_mask,\n)" in sai
    assert "get_indexer_topk," in sai
    assert hdi.CAND["DeepseekV41ForCausalLM"] == (2048, 8)


def test_sm120_decode_is_flattened_with_per_row_tables_the_group_body_relies_on():
    """vLLM v1/attention/backends/mla/indexer.py: only next_n in (1, 2) decodes
    natively off SM90 / SM100, so DSpark's 1 + 5 rows are flattened to one row
    per token (re-reading each request's K once per row); every flattened row
    carries its request's block-table row and its own causal length, rows of a
    request are contiguous but groups of G need not align with requests."""
    n = norm(src("indexer_backend"))
    assert ("def_supports_native_decode(next_n:int)->bool:" in n)
    tail = n.split("def_supports_native_decode")[1][:700]
    assert "is_device_capability_family(100):returnTrue" in tail
    assert "is_device_capability_family(90):returnnative_next_n_supported(next_n)returnnext_nin(1,2)" in tail
    assert "next_n=1+vllm_config.num_speculative_tokens" in n
    assert "returnnot_supports_native_decode(next_n)or(" in n
    # uniform lengths: the block table row is copied per token; variable lengths: repeat_interleave
    assert "self.expanded_block_table_buffer[:num_decode_tokens]" in n
    assert "torch.repeat_interleave(block_table,decode_lens,dim=0,output_size=actual_expanded)" in n
    assert "self.decode_seq_lens_buffer[actual_expanded:]=0" in n          # padded rows: length 0
    assert "expanded_block_table_buffer[actual_expanded:num_decode_tokens,0]=0" in n
    # the V4.1 indexer backend's kernel block size off SM90 (the plugin hook declares 64 on SM12x)
    assert "return[64ifcurrent_platform.is_device_capability_family(90)else128]" in n


def test_candidate_source_scoring_and_persistent_topk_contract():
    cb, tp = src("candidate_blocks"), src("indexer_topk")
    assert params(cb, "_block_scores_kernel") == dc.SCORES_PARAMS
    n = norm(cb)
    # the call the CANDSEL twin reproduces (FULL_WIDTH is the only difference)
    assert ("_block_scores_kernel[(rows,32)](logits,row_ks,row_ke,scores,valid_blocks,*logits.stride(),"
            "row_ks.stride(0)ifrow_ksisnotNoneelse0,row_ke.stride(0),scores.stride(0),width,nblocks,"
            "block_size,row_ksisnotNone,row_repeat,TILE=128,FULL_WIDTH=notuse_deep_select,)") in n
    assert "num_blocks=nblocksifFULL_WIDTHelsenum_valid" in n          # False: valid blocks only
    assert "iftl.program_id(1)==0:tl.store(valid_blocks+row,num_valid)" in n
    t = norm(tp)
    assert "iftopk_tokensnotin(512,1024,2048)" in t and dc.PERSISTENT_K == (512, 1024, 2048)
    # persistent_topk is the SM12x auto choice (cooperative excludes the 12x family)
    assert "andnotcurrent_platform.is_device_capability_family(120)" in t
    assert "ifself._is_cudaandtopk_tokensin(512,1024,2048):return\"persistent\"" in t
