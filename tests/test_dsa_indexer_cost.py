# SPDX-License-Identifier: Apache-2.0
"""The byte / launch model of the V4.1 indexer step (kernels/dsa_indexer_cost.py):
the layer map is the checkpoint's, and the model orders the arms the way the
design says (it is an estimate: only these invariants are pinned)."""
from suffix_hybrid.helion_kernels import dsa_indexer as hdi
from suffix_hybrid.kernels import dsa_indexer_cost as c


def test_layer_map_is_deepseek_v41_flash():
    assert sorted(c.INDEX_LAYERS) == [2, 8, 14, 20, 24, 28, 32, 36]
    assert [c.kind(l) for l in sorted(c.INDEX_LAYERS)] == ["dense"] * 3 + ["source"] + ["consumer"] * 4
    # compress ratio 2 on layers 2-19, 1 on 20-39
    assert all((r == 2) == (l < 20) for l, r in c.INDEX_LAYERS.items())
    assert (c.CAND_BLOCKS, c.CAND_BS) == hdi.CAND["DeepseekV41ForCausalLM"] and c.TOPK == 512


def test_ours_moves_several_times_fewer_bytes_at_every_context():
    for reqs in (1, 8, 32):
        for ctx in (8192, 32768, 131072, 524288):
            s, o = c.step_cost(ctx, reqs, 6, False), c.step_cost(ctx, reqs, 6, True)
            assert o["bytes"] * 5 < s["bytes"] and o["launches"] <= s["launches"]


def test_the_mask_pass_costs_max_model_len_not_context():
    """The stock consumer rewrites the whole logits row: a short context at
    c=32 still pays rows x 2 MB per consumer layer."""
    short = c.layer_cost("consumer", 4096, 192, 32, False)
    assert short["mask"] >= 192 * 4 * c.MAX_MODEL_LEN
    fused = c.layer_cost("consumer", 4096, 192, 32, True)
    assert fused["bytes"] < short["bytes"] / 10


def test_consumer_fusion_stops_growing_with_context():
    a, b = (c.layer_cost("consumer", n, 6, 1, True)["bytes"] for n in (262144, 524288))
    assert a == b                                   # 2048 x 8 positions, whatever the context
    s1, s2 = (c.layer_cost("consumer", n, 6, 1, False)["bytes"] for n in (262144, 524288))
    assert s2 > 1.5 * s1


def test_row_sharing_divides_the_k_bytes_by_g():
    stock = c.layer_cost("dense", 131072, 6, 1, False)["k"]
    ours = c.layer_cost("dense", 131072, 6, 1, True)["k"]
    assert stock == 6 * ours


def test_table_renders():
    lines = c.table()
    assert len(lines) == 1 + 3 * 3 and "524288" in lines[-1]
