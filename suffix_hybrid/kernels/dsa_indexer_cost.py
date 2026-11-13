# SPDX-License-Identifier: Apache-2.0
"""HBM-byte / launch model of one DeepSeek-V4.1 decode step's sparse-attention
indexers on one SM120 rank (pure Python; the lab prints it next to its
measurements and docs/dsv41-indexer.md quotes it).

An ESTIMATE, not a measurement: bytes moved over the HBM roofline
(RTX PRO 6000: 1.79 TB/s), every byte counted once as it crosses HBM (L2 reuse
between rows is exactly the unknown the lab measures, so the stock arm is given
no credit and ours the full MTP-row sharing). Nothing here shapes a decision by
itself; it says where the bytes are and which lever removes them.

Layers (config.json of DeepSeek-V4.1-Flash): index_source_layer_ids
[2, 8, 14, 20, 24, 28, 32, 36]; compress_ratios 2 on layers 2-19 and 1 on 20-39,
so the indexers on layers 2/8/14 scan ctx/2 index positions and 20..36 scan ctx;
candidate_source_layer_id 20 writes the top 2048 blocks of 8 positions, the four
later indexers (24, 28, 32, 36) only rank inside them. The indexer is replicated,
not TP-split: every rank runs all of it.
"""
from __future__ import annotations

HBM_BPS = 1.79e12
K_BYTES = 132              # one index position: 128 e4m3 + one f32 scale
TOPK = 512
TOPK_PASSES = 3            # radix select re-reads its row (ESTIMATE; persistent_topk)
TORCH_TOPK_PASSES = 4      # torch.topk(k=2048) over 65536 block scores (ESTIMATE)
HEADS = 32
INDEX_LAYERS = {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}   # layer -> compress ratio
CAND_SOURCE = 20
CAND_BLOCKS, CAND_BS = 2048, 8
MAX_MODEL_LEN = 524288     # the target context (d6-spec.json)


def kind(layer: int) -> str:
    """dense (before the candidate source) | source | consumer."""
    return "source" if layer == CAND_SOURCE else "consumer" if layer > CAND_SOURCE else "dense"


def layer_cost(kind_: str, ctx: int, rows: int, reqs: int, ours: bool, ratio: int = 1,
               max_model_len: int = MAX_MODEL_LEN) -> dict:
    """Bytes and launches of one indexer layer's decode (logits -> selection).
    ctx: the batch's context in tokens; rows: flattened decode rows (reqs x G);
    ours: row-group K sharing + candidate fusion / valid-block selection."""
    n = -(-ctx // ratio)                      # index positions per request
    width = max_model_len // ratio            # logits row width
    nb = -(-width // CAND_BS)                 # candidate blocks per full row
    q = rows * HEADS * 128
    if ours and kind_ == "consumer":
        m = min(n, CAND_BLOCKS * CAND_BS)     # candidate positions only
        b = {"k": reqs * m * K_BYTES, "q": q, "logits": rows * m * 4,
             "topk": rows * (m * 4 * TOPK_PASSES + TOPK * 4)}
        return _sum(b, 3)                     # cand logits, top-k, remap
    k = (reqs if ours else rows) * n * K_BYTES          # stock re-reads K once per flattened row
    b = {"k": k, "q": q, "logits": rows * n * 4}
    launches = 1
    if kind_ == "source":
        if ours:    # valid blocks scored once, persistent top-k over them
            nvalid = -(-n // CAND_BS)
            b["select"] = rows * (n * 4 + nvalid * 4 + nvalid * 4 * TOPK_PASSES + CAND_BLOCKS * 4)
            launches += 2
        else:       # FULL_WIDTH scores of every block + torch.topk(2048) + store
            b["select"] = rows * (width * 4 + nb * 4 + nb * 4 * TORCH_TOPK_PASSES + CAND_BLOCKS * 4 * 2)
            launches += 5
    elif kind_ == "consumer":   # stock: flags + a full-width -inf rewrite of the row
        b["mask"] = rows * (width * 4 + (nb + 1) * 2)
        launches += 2
    b["topk"] = rows * (n * 4 * TOPK_PASSES + TOPK * 4)
    return _sum(b, launches + 1)


def _sum(b: dict, launches: int) -> dict:
    total = sum(b.values())
    return {**b, "bytes": total, "us": total / HBM_BPS * 1e6, "launches": launches}


def step_cost(ctx: int, reqs: int, g: int = 6, ours: bool = False,
              max_model_len: int = MAX_MODEL_LEN) -> dict:
    """Sum over the eight indexer layers of one decode step. {layer: layer_cost,
    'bytes', 'us' (roofline lower bound), 'launches'}."""
    rows = reqs * g
    out, total_b, total_l = {}, 0, 0
    for layer, ratio in INDEX_LAYERS.items():
        c = layer_cost(kind(layer), ctx, rows, reqs, ours, ratio, max_model_len)
        out[layer] = c
        total_b += c["bytes"]
        total_l += c["launches"]
    return {**out, "bytes": total_b, "us": total_b / HBM_BPS * 1e6, "launches": total_l}


def table(ctxs=(32768, 131072, 524288), reqs=(1, 8, 32), g: int = 6) -> list[str]:
    """Rows: ctx x requests -> stock vs ours bytes (MB), roofline us, launches."""
    lines = ["ctx      reqs rows  stock MB   ours MB   stock us  ours us  launches stock/ours"]
    for c in ctxs:
        for r in reqs:
            s, o = step_cost(c, r, g, False), step_cost(c, r, g, True)
            lines.append(f"{c:<8d} {r:<4d} {r * g:<4d} {s['bytes'] / 1e6:9.1f} {o['bytes'] / 1e6:9.1f} "
                         f"{s['us']:9.1f} {o['us']:8.1f}  {s['launches']:>3d}/{o['launches']:<3d}")
    return lines


if __name__ == "__main__":
    print("\n".join(table()))
