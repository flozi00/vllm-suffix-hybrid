# SPDX-License-Identifier: Apache-2.0
"""scripts/dsv41_dense_inventory.py: the per-rank shapes agree with the kernel's lab cases, the byte
math, and the markdown renders (the --verify HTTP check is manual)."""
import importlib.util
import os

os.environ["HELION_INTERPRET"] = "1"

from suffix_hybrid.helion_kernels import mxfp8_gemm as hx  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dsv41_dense_inventory",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "dsv41_dense_inventory.py"))
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)


def test_fp8_rows_match_kernel_shapes():
    ks = hx.dsv41_shapes(8)
    got = {r["name"]: (r["N"], r["K"]) for r in inv.rows(8)}
    for name, key in (("wo_a (grouped o_proj)", "wo_a"), ("fused_wqa_wkv", "fused_wqa_wkv"),
                      ("wq_b", "wq_b"), ("wo_b", "wo_b"), ("indexer.wq_b", "indexer.wq_b"),
                      ("shared.gate_up (w1+w3)", "shared.gate_up"), ("shared.down (w2)", "shared.down"),
                      ("dspark.main_proj", "dspark.main_proj")):
        assert got[name] == ks[key], name


def test_bytes_and_table():
    t = {e["name"]: e for e in inv.table(8)}
    wo_a = t["wo_a (grouped o_proj)"]
    assert abs(wo_a["stock_MB"] - 2 * 1024 * 4096 / 1e6) < 1e-6          # BF16 emulation weight
    assert abs(wo_a["ours_MB"] - 1024 * 4096 * (1 + 1 / 32) / 1e6) < 1e-6  # e4m3 + per-32 scales
    assert wo_a["ours_MB"] < wo_a["stock_MB"] / 1.9
    assert t["lm_head"]["us"][192]["ours_step"] == t["lm_head"]["us"][192]["stock_step"]  # lane M<=64
    assert t["lm_head"]["us"][6]["ours_step"] < t["lm_head"]["us"][6]["stock_step"]
    assert t["engram.wkv"]["ours"] is None
    # a tuned split costs two more launches: never better than the single-launch lane
    for e in t.values():
        if e["ours"]:
            assert e["us"][6]["ours_split_step"] >= e["us"][6]["ours_step"] - 1e-9
    md = inv.markdown(8)
    assert "**total**" in md and "MXFP8 lane subtotal" in md
