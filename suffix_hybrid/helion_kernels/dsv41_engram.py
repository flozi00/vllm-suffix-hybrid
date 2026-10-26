# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Engram (TP8, DP1) - packed-row transport: gather the MXFP4 rows
as raw bytes, all-gather THOSE, dequantise after the exchange.

Stock (nightly 39983a5e7 + vllm#61530, suffix_hybrid/dsv41_engram_pr61530.txt),
per Engram layer (layers 1 and 14) per step, TP8 / DP1 / no sequence parallel:

  lookup   _engram_lookup_kernel: this rank's 3 of 24 hash heads -> UVA reads of
           the pinned host table (128 B packed e2m1 row + 8 B ue8m0 scales) ->
           dequantised BF16 rows [T, 3, 256]                    (side stream, 1 launch)
  gather   tensor_model_parallel_all_gather(rows, dim=1): [T, 3, 256] bf16 ->
           [T, 24, 256] bf16 (512 B per row on the wire)        (NCCL + copy)
  wkv      ColumnParallelLinear(24*256 -> 5*5120, gather_output=True): activation
           quant + GEMM + a SECOND all-gather of [T, 25600] bf16  (see docs, P2)
  gate     _fused_engram_post_wkv_kernel                         (1 launch)

This module is P1: move the dequantisation after the exchange.
  gather_packed   same ownership / ordering as the stock kernel, but copies the
                  raw record (128 B + 8 B) into a packed staging row [T, LH, 136] u8
  (all-gather the uint8 rows: 136 B per row instead of 512 B, -73 % wire bytes)
  dequant_packed  [T, 24, 136] u8 -> [T, 24, 256] bf16, bit-identical to the stock
                  PACKED_FP4 decode: nibble (even column = low nibble) -> e2m1 value
                  x 2^(scale - 127) ( = bitcast(scale << 23); byte 0 -> +0.0, like stock),
                  rounded once to bf16.
Launches: stock 1 lookup + 1 gather; here gather_packed + 1 gather + dequant = +1
launch per Engram layer (2 per step) for 3.76x fewer bytes through NCCL. Whether
that pays is a measurement (K6): the all-gather of T x 24 x 512 B is ~12 KB/token
(2.4 MB at c32 = 192 tokens) and the lab's host-AR numbers say PCIe-staged
collectives run at single-digit GB/s; the bigger lever is P2 (docs/dsv41-fuse.md).

K1 contract
  ids      i32 [T, NC=24]  bucket id per hash column (stock layout, global row index)
  w_data   u8  [rows, 128] packed e2m1 (2 values per byte, even column low nibble)
  w_scale  u8  [rows, 8]   ue8m0 per 32 columns
  rows owned by this rank: vocab_start <= id < vocab_end (local = id - vocab_start);
  hash heads head_start .. head_start + LH, heads >= NC (padding of an uneven shard)
  and ids outside the rank's range produce zero rows, as in the stock kernel.

Knob: SUFFIX_HELION_DSV41_ENGRAM = oxide (default = stock) | helion | auto.
"""
import sys

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_common as cm

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU host: the module stays importable (lab ABI, tests)
    helion = hl = None
CE = hl.constexpr if hl is not None else int

ENV = "SUFFIX_HELION_DSV41_ENGRAM"
KERNEL_GATHER = "dsv41_engram_gather_packed"
KERNEL_DEQUANT = "dsv41_engram_dequant_packed"
MARKER = "[suffix dsv41-engram-fuse]"
DIM = 256            # engram_head_dim
QB = 32              # MXFP4 block
DATA = DIM // 2      # 128 B
SCALES = DIM // QB   # 8 B
REC = DATA + SCALES  # 136 B packed record
HEADS = 24           # (max_ngram 4 - 1) * 8 heads
BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)


def dsv41_engram_gather_packed(ids: torch.Tensor, w_data: torch.Tensor, w_scale: torch.Tensor,
                               vocab_start: int, vocab_end: int, head_start: int, local_heads: CE,
                               total_heads: CE):
    """ids i32 [T, total_heads]; returns packed u8 [T, local_heads, REC] (zeros for
    rows this rank does not own / padded heads)."""
    T = ids.size(0)
    dd = hl.specialize(w_data.size(1))
    ss = hl.specialize(w_scale.size(1))
    packed = torch.empty([T, local_heads, dd + ss], dtype=torch.uint8, device=ids.device)
    pk = packed.view(T * local_heads, dd + ss)
    pk_d = pk[:, 0:dd]
    pk_s = pk[:, dd:dd + ss]
    flat = ids.reshape(T * total_heads)
    br = hl.register_block_size(16, 128)
    for tile_r in hl.tile(T * local_heads, block_size=br):
        r = tile_r.index
        token = r // local_heads
        head = head_start + r % local_heads
        live = head < total_heads
        idx = flat[token * total_heads + torch.where(live, head, 0)].to(torch.int64)
        owned = live & (idx >= vocab_start) & (idx < vocab_end)
        local = torch.where(owned, idx - vocab_start, 0)
        pk_d[tile_r, :] = torch.where(owned[:, None], w_data[local, :], 0).to(torch.uint8)
        pk_s[tile_r, :] = torch.where(owned[:, None], w_scale[local, :], 0).to(torch.uint8)
    return packed


def _e2m1_f16(code):
    """e2m1 code (int32 0..15) -> exact float16 (bit construction; k2_attn's)."""
    m = code & 7
    bits = torch.where(m >= 2, (m << 9) + 0x3800, m * 0x3800) | ((code & 8) << 12)
    return bits.to(torch.int16).view(torch.float16)


def dsv41_engram_dequant_packed(packed: torch.Tensor):
    """packed u8 [R, REC] (R = T * heads) -> bf16 [R, 2 * DATA]."""
    R = packed.size(0)
    dd = hl.specialize(packed.size(1)) - SCALES
    out = torch.empty([R, dd * 2], dtype=torch.bfloat16, device=packed.device)
    out4 = out.view(R, SCALES, dd // SCALES, 2)
    raw_v = packed[:, 0:dd].view(R, SCALES, dd // SCALES)
    sc_v = packed[:, dd:dd + SCALES]
    br = hl.register_block_size(1, 64)
    for tile_r in hl.tile(R, block_size=br):
        raw = raw_v[tile_r, :, :].to(torch.int32)
        scale = (sc_v[tile_r, :].to(torch.int32) << 23).view(torch.float32)[:, :, None]
        lo = (_e2m1_f16(raw & 15).to(torch.float32) * scale).to(torch.bfloat16)
        hi = (_e2m1_f16(raw >> 4).to(torch.float32) * scale).to(torch.bfloat16)
        out4[tile_r, :, :, :] = hl.join(lo, hi)
    return out


# ---------------------------------------------------------------------------
# exact references (the stock PACKED_FP4 decode, transcribed)
# ---------------------------------------------------------------------------
def gather_reference(ids, w_data, w_scale, vocab_start, vocab_end, head_start, local_heads,
                     total_heads):
    t = ids.shape[0]
    out = torch.zeros(t, local_heads, REC, dtype=torch.uint8, device=ids.device)
    for tok in range(t):
        for h in range(local_heads):
            head = head_start + h
            if head >= total_heads:
                continue
            i = int(ids[tok, head])
            if vocab_start <= i < vocab_end:
                out[tok, h, :DATA] = w_data[i - vocab_start]
                out[tok, h, DATA:] = w_scale[i - vocab_start]
    return out


def dequant_reference(packed):
    """Literal torch transcription of _engram_lookup_kernel's PACKED_FP4 + bf16 store."""
    byte = packed[:, :DATA].to(torch.int32)
    cols = torch.arange(DIM, device=packed.device)
    nib = torch.where(cols[None, :] % 2 == 0, byte[:, cols // 2] & 0xF, (byte[:, cols // 2] >> 4) & 0xF)
    e = (nib & 0x7) >> 1
    m = (nib & 0x1).to(torch.float32)
    pow2 = ((e - 2 + 127) << 23).view(torch.float32)
    mag = torch.where(e == 0, m * 0.5, (2.0 + m) * pow2)
    values = torch.where((nib & 0x8) != 0, -mag, mag)
    scale = (packed[:, DATA:].to(torch.int32) << 23).view(torch.float32)[:, cols // QB]
    return (values * scale).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# plumbing / lab ABI
# ---------------------------------------------------------------------------
def case(tokens: int, local_heads: int = 3) -> dict:
    return {"numtokens": tokens, "heads": HEADS, "local_heads": local_heads, "dim": DIM}


def _is_v41(model: dict) -> bool:
    return str(model.get("architecture", "")).startswith(("DeepseekV41", "DeepseekV4"))


def cases_gather(model: dict) -> list:
    if not _is_v41(model):
        return []
    tp = int(model.get("tp", 8) or 8)
    return [case(b, -(-HEADS // tp)) for b in BUCKETS]


def cases_dequant(model: dict) -> list:
    """Dequant sees the WHOLE gathered row set: T x 24 heads."""
    return [dict(c, local_heads=HEADS) for c in cases_gather(model)]


def make_table(rows: int, seed: int, device="cpu"):
    g = torch.Generator().manual_seed(31 + seed)
    d = torch.randint(0, 256, (rows, DATA), generator=g, dtype=torch.uint8).to(device)
    s = torch.randint(110, 135, (rows, SCALES), generator=g, dtype=torch.uint8).to(device)
    return d, s


def make_gather(c: dict, seed: int, device=None):
    device = device or cm.DEVICE
    t, lh = c["numtokens"], c["local_heads"]
    rows = 4096
    d, s = make_table(rows, seed, device)
    g = torch.Generator().manual_seed(5 + seed)
    ids = torch.randint(0, 3 * rows, (t, HEADS), generator=g, dtype=torch.int32).to(device)
    head_start = min(3, HEADS - lh)        # rank 1 of 8: heads 3..5; owns bucket ids [rows, 2 rows)
    ids[:, head_start:head_start + lh] = rows + (ids[:, head_start:head_start + lh] % rows)   # all local heads owned
    return ids, d, s, rows, 2 * rows, head_start, lh, HEADS


def make_dequant(c: dict, seed: int, device=None):
    device = device or cm.DEVICE
    d, s = make_table(c["numtokens"] * c["local_heads"], seed, device)
    return (torch.cat([d, s], 1).contiguous(),)


def baseline_dequant(packed):
    """Stock: the dequant half of _engram_lookup_kernel is fused into its gather, so the
    only comparable stock launch is the whole lookup; the lab image must carry the
    vllm#61530 overlay (SUFFIX_DSV41_ENGRAM_MXFP4=1)."""
    raise NotImplementedError("K4 for this pair is the in-situ engram layer A/B (docs/dsv41-fuse.md): "
                              "the stock gather+dequant is one kernel")


KERNELS = [
    {"name": KERNEL_GATHER, "body": dsv41_engram_gather_packed, "cases": cases_gather,
     "make": make_gather, "reference": gather_reference,
     "baseline": gather_reference,
     "baseline_name": "eager index_select (stock fuses gather+dequant; see docs: in-situ A/B)",
     "check": lambda out, ref: (bool(torch.equal(out.cpu(), ref.cpu())), "bytes identical"),
     "replaces": [r"_engram_lookup_kernel"]},
    {"name": KERNEL_DEQUANT, "body": dsv41_engram_dequant_packed, "cases": cases_dequant,
     "make": make_dequant, "reference": dequant_reference,
     "baseline": dequant_reference,
     "baseline_name": "eager transcription of the stock PACKED_FP4 decode",
     "check": lambda out, ref: (bool(torch.equal(out.cpu().view(torch.int16),
                                                 ref.cpu().view(torch.int16))), "bits identical"),
     "replaces": []},
]


def wire_bytes(tokens: int, heads: int = HEADS, packed: bool = True) -> int:
    """Bytes the TP all-gather moves per Engram layer (sum over ranks of what each sends)."""
    return tokens * heads * (REC if packed else 2 * DIM)


def default_config(case: dict) -> dict:
    return {"block_sizes": [16], "num_warps": 4}


# ---------------------------------------------------------------------------
# boot gates <kernel>_capture_replay
# ---------------------------------------------------------------------------
def _built(body, case_, kernel, cfg=None):
    import helion as _h
    cfg = cfg or hk.config(kernel, case_) or default_config(case_)
    return _h.kernel(body, config=_h.Config(**cfg), static_shapes=False)


def dsv41_engram_gather_packed_capture_replay(tokens: int = 6, cfg=None, runner_factory=None) -> bool:
    c = case(hk.bucket(tokens, BUCKETS))
    kern = _built(dsv41_engram_gather_packed, c, KERNEL_GATHER, cfg)
    sample = make_gather(case(tokens), 0)

    def make(i):
        ids, d, s, *_ = make_gather(case(tokens), i)
        return [ids, d, s]
    return cm.capture_replay_gate(KERNEL_GATHER, lambda ids, d, s: kern(ids, d, s, *sample[3:]), make,
                                  runner_factory=runner_factory)


def dsv41_engram_dequant_packed_capture_replay(tokens: int = 6, cfg=None, runner_factory=None) -> bool:
    c = case(hk.bucket(tokens, BUCKETS), HEADS)
    kern = _built(dsv41_engram_dequant_packed, c, KERNEL_DEQUANT, cfg)
    return cm.capture_replay_gate(KERNEL_DEQUANT, lambda p: kern(p),
                                  lambda i: [make_dequant(case(tokens, HEADS), i)[0]],
                                  runner_factory=runner_factory)


BOOT_GATES = {f"{KERNEL_GATHER}_capture_replay": dsv41_engram_gather_packed_capture_replay,
              f"{KERNEL_DEQUANT}_capture_replay": dsv41_engram_dequant_packed_capture_replay}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog=__name__)
    ap.add_argument("cmd", choices=["oracle", "bench", "capture_replay"])
    ap.add_argument("--tokens", type=int, default=6)
    ns, _ = ap.parse_known_args(argv)
    if ns.cmd == "capture_replay":
        return 0 if all(g(tokens=ns.tokens) for g in BOOT_GATES.values()) else 1
    from suffix_hybrid.kernels import dsv41_bench
    return dsv41_bench.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
