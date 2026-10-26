# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 vLLM anchors, replayed against the real vLLM 0.30.0 and 0.32.0
sources (tests/fixtures/qwen_gdn_vllm: excerpts parsed with ast, never
imported): the overridden / called method signatures, the GDNAttentionMetadata
fields the lane reads, and the forward_cuda routing flag 0.32.0 renamed
(enable_fused_gdn_decode -> enable_fused_gdn_spec_decode)."""
import ast
import pathlib

import pytest

from suffix_hybrid.kernels import qwen_gdn

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "qwen_gdn_vllm"


def _parse(version):
    text = (FIX / f"vllm-{version}.py.txt").read_text()
    tree = ast.parse(text)
    cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    methods = {m.name: m for m in cls["QwenGatedDeltaNetAttention"].body
               if isinstance(m, ast.FunctionDef)}
    sigs = {name: [a.arg for a in m.args.args] for name, m in methods.items()}
    fields = [s.target.id for s in cls["GDNAttentionMetadata"].body
              if isinstance(s, ast.AnnAssign)]
    return ast.get_source_segment(text, methods["forward_cuda"]), sigs, fields


@pytest.mark.parametrize("version,flag", [("0.30.0", "enable_fused_gdn_decode"),
                                          ("0.32.0", "enable_fused_gdn_spec_decode")])
def test_anchor_replay(version, flag):
    src, sigs, fields = _parse(version)
    assert qwen_gdn.anchor_flag(src, sigs, fields) == flag
    assert qwen_gdn.ROUTE_FLAGS[flag] == version


def test_anchor_drift_refuses():
    src, sigs, fields = _parse("0.32.0")
    with pytest.raises(RuntimeError, match="no known flag"):
        qwen_gdn.anchor_flag(src.replace("enable_fused_gdn_spec_decode", "fused_v3"), sigs,
                             fields)
    with pytest.raises(RuntimeError, match="forward_cuda reaches"):
        qwen_gdn.anchor_flag(src.replace(qwen_gdn.FUSED_PACKED_OP, "other_op"), sigs, fields)
    bad = dict(sigs, _forward_core_fused_norm=["self", "mixed_qkv", "b", "a", "core_attn_out"])
    with pytest.raises(RuntimeError, match="_forward_core_fused_norm"):
        qwen_gdn.anchor_flag(src, bad, fields)
    with pytest.raises(RuntimeError, match="num_accepted_tokens"):
        qwen_gdn.anchor_flag(src, sigs, [f for f in fields if f != "num_accepted_tokens"])
