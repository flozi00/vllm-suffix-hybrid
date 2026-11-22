"""Tuned launch configs for Inkling 2's SM120 expert GEMMs (moe_sm120.bmm), behind
SUFFIX_INKLING2_MOE_TUNE=1 (off = the upstream configs).

Swept on real captured inputs (RTX 5090, 2026-11-22): every entry is BITWISE equal to the
shipped config on the rows the kernel writes. Only the prefill tile (128) is kept: 1.34-1.51x
per call, rag-8k TTFT -5..-9% and c=32 +5.7% in speed.py. The decode-tile winners (8/16,
1.03-1.32x per call) cost ~1.5% ITL p99 at c=1 end to end and were dropped.
"""
import os

MOE_TUNE = os.getenv("SUFFIX_INKLING2_MOE_TUNE") == "1"

# (dtype, tile, N, K) -> nvfp4: (BM, BN, BK, warps, stages); mxfp8: (BN, BT, BK, warps, stages)
MOE_CONFIGS = {
    ("nvfp4", 128, 2304, 2304): (64, 128, 256, 4, 2),   # 277.0 -> 207.3
    ("nvfp4", 128, 2304, 1152): (64, 128, 256, 4, 2),   # 172.7 -> 114.3
    ("mxfp8", 128, 2304, 2304): (64, 64, 128, 4, 2),    # 1207 -> 864.6
    ("mxfp8", 128, 2304, 1152): (128, 32, 128, 4, 2),   # 620.4 -> 440.8
}


def moe_config(kind, tile, n, k):
    return MOE_CONFIGS.get((kind, tile, n, k)) if MOE_TUNE else None
