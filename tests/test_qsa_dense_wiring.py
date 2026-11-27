# SPDX-License-Identifier: Apache-2.0
"""CPU test for the SUFFIX_ROCM_QSA_DENSE wiring (no vLLM, no GPU): the rewritten
forward_qsa (verbatim tail of vLLM 81198e97 vllm/models/qwen4_exp/amd/qsa.py) hands the
main metadata's seq_lens / query_start_loc and the indexer's top-k to the plugin, and
keeps MTP steps that reuse the step-0 selection on the stock path. Silicon check: boot
gate qsa_dense_bench."""
from types import SimpleNamespace

from suffix_hybrid import rocm_patches as rp

PATCH = rp.PATCHES["SUFFIX_ROCM_QSA_DENSE"]
# The gate is a tuple: the dataclass fields, the builder population and the
# dense call travel together (0.34 dropped seq_lens/query_start_loc).
DENSE = PATCH[-1]
FAKE = (
    "def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output,\n"
    "                token_to_req):\n"
    "        num_tokens = attn_metadata.num_actual_tokens\n"
    "        logical_indices = layer.topk_indices_buffer[:num_tokens]\n"
    "        key_cache, value_cache = kv_cache\n"
    + DENSE.old +
    "        return output\n")


def test_forward_qsa_passes_metadata_and_skip_topk():
    src = rp.patch_source(DENSE, FAKE)
    assert "qsa_sparse_paged_attention" not in src
    calls = []
    mod = {"_suffix_qsa_attention": lambda *a, **k: calls.append((a, k))}
    exec(compile(src, "fake_qsa", "exec"), mod)
    md = SimpleNamespace(num_actual_tokens=3, block_table="bt",
                         _suffix_seq_lens="sl", _suffix_query_start_loc="qsl")
    for skip in (False, True):
        layer = SimpleNamespace(topk_indices_buffer=[7, 8, 9, 10],
                                indexer=SimpleNamespace(token_topk=2048, skip_topk=skip))
        out = [0, 1, 2, 3]
        assert mod["forward_qsa"](None, layer, "qqqq", None, None, ("k", "v"), md, out,
                                  "t2r") is out
        args, kwargs = calls.pop()
        assert args == ("qqq", "k", "v", [7, 8, 9], "bt", "t2r", [0, 1, 2], "sl", "qsl", 2048)
        assert kwargs == {"dense": not skip}
    assert DENSE.after == "suffix_hybrid.kernels.qsa_dense_rocm:install"


def test_dense_gate_carries_the_metadata_stash():
    """The dense call reads the stashed fields; the gate must ship the patches
    that declare and populate them, or the boot dies on AttributeError."""
    names = [p.target for p in PATCH]
    assert names.count("vllm.models.qwen4_exp.common.qsa") == 2
    fields = next(p for p in PATCH if "_suffix_seq_lens: torch.Tensor" in p.new)
    build = next(p for p in PATCH if "_suffix_seq_lens=common_attn_metadata.seq_lens" in p.new)
    assert "_suffix_query_start_loc" in fields.new and "_suffix_query_start_loc" in build.new
