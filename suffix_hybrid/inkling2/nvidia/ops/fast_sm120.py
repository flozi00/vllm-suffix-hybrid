"""Tuned launch configs for Inkling 2's SM120 expert GEMMs (moe_sm120.bmm), behind
SUFFIX_INKLING2_MOE_TUNE=1 (off = the upstream configs).

Swept on real captured inputs (RTX 5090, 2026-11-22): every entry is BITWISE equal to the
shipped config on the rows the kernel writes, entries with <3% gain were dropped. Prefill
(tile 128) 1.34-1.51x, decode (tile 8/16) 1.03-1.32x per call; end to end the decode part is
noise at c<=16, the prefill part is what pays.
"""
import os

MOE_TUNE = os.getenv("SUFFIX_INKLING2_MOE_TUNE") == "1"

# (dtype, tile, N, K) -> nvfp4: (BM, BN, BK, warps, stages); mxfp8: (BN, BT, BK, warps, stages)
MOE_CONFIGS = {
    ("nvfp4", 8, 2304, 2304): (32, 32, 512, 2, 3),      # 13.29 -> 12.82 us
    ("nvfp4", 8, 2304, 1152): (32, 128, 512, 2, 2),     # 13.65 -> 12.92
    ("nvfp4", 16, 2304, 2304): (16, 64, 256, 2, 2),     # 25.32 -> 22.80
    ("nvfp4", 128, 2304, 2304): (64, 128, 256, 4, 2),   # 277.0 -> 207.3
    ("nvfp4", 128, 2304, 1152): (64, 128, 256, 4, 2),   # 172.7 -> 114.3
    ("mxfp8", 8, 2304, 2304): (64, 16, 128, 4, 3),      # 18.93 -> 15.25
    ("mxfp8", 8, 2304, 1152): (128, 16, 128, 4, 3),     # 12.94 -> 12.53
    ("mxfp8", 16, 2304, 2304): (64, 16, 128, 4, 2),     # 75.11 -> 63.63
    ("mxfp8", 16, 2304, 1152): (64, 16, 128, 4, 2),     # 20.63 -> 15.61
    ("mxfp8", 128, 2304, 2304): (64, 64, 128, 4, 2),    # 1207 -> 864.6
    ("mxfp8", 128, 2304, 1152): (128, 32, 128, 4, 2),   # 620.4 -> 440.8
}


def moe_config(kind, tile, n, k):
    return MOE_CONFIGS.get((kind, tile, n, k)) if MOE_TUNE else None
