"""Tuned launch configs for Inkling 2's SM120 expert GEMMs (moe_sm120.bmm), behind
SUFFIX_INKLING2_MOE_TUNE=1 (off = the upstream configs).

Swept on real captured inputs (RTX 5090, 2026-11-22): every entry is BITWISE equal to the
shipped config on the rows the kernel writes (20/20 captures, 2026-11-22). Kept: the prefill
tile (128) and the decode tiles 32/64 that c=4..32 hit, >=10% per call. Dropped: tiles 8/16
(c=1..2), whose per-call wins cost ~1.5% ITL p99 at c=1 end to end.
"""
import os

MOE_TUNE = os.getenv("SUFFIX_INKLING2_MOE_TUNE") == "1"

# (dtype, tile, N, K) -> nvfp4: (BM, BN, BK, warps, stages); mxfp8: (BN, BT, BK, warps, stages)
MOE_CONFIGS = {
    # prefill (tile 128), re-swept with BM/BN up to 128/256 (RTX 5090, 2026-11-22)
    ("nvfp4", 128, 2304, 2304): (128, 128, 256, 8, 3),  # 277.9 -> 158.2 us (1.76x)
    ("nvfp4", 128, 2304, 1152): (128, 64, 256, 4, 2),   # 173.7 -> 105.8 (1.64x)
    ("mxfp8", 128, 2304, 2304): (64, 64, 128, 4, 2),    # 1209.8 -> 864.3 (1.40x)
    ("mxfp8", 128, 2304, 1152): (128, 32, 128, 4, 2),   # 623.3 -> 440.6 (1.41x)
    # decode tiles seen at c=4..32 with DSpark k=7 (>=10% only; tiles 8/16 cost c=1 ITL)
    ("nvfp4", 32, 2304, 2304): (32, 256, 256, 4, 2),    # 58.5 -> 52.1 (1.12x)
    ("nvfp4", 32, 2304, 1152): (32, 256, 256, 4, 2),    # 32.1 -> 26.5 (1.21x)
    ("mxfp8", 32, 2304, 2304): (128, 32, 128, 4, 2),    # 325.3 -> 140.0 (2.32x)
    ("mxfp8", 32, 2304, 1152): (128, 32, 128, 4, 2),    # 159.1 -> 71.3 (2.23x)
    ("mxfp8", 64, 2304, 2304): (64, 64, 128, 4, 2),     # 540.2 -> 347.5 (1.55x)
    ("mxfp8", 64, 2304, 1152): (64, 64, 128, 4, 2),     # 285.3 -> 177.8 (1.60x)
}


def moe_config(kind, tile, n, k):
    return MOE_CONFIGS.get((kind, tile, n, k)) if MOE_TUNE else None
